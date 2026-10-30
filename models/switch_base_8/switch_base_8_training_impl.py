"""Numerical primitives for the generated full Switch Base-8 SGD step.

Reference: transformers 4.57.6. SETEP owns the forward graph, layer/expert
traversals, gradient storage, and weight assignments. PyTorch implements the
opaque kernels and differentiates that graph; no Hugging Face forward is called.
Inputs are one tokenized source/target sequence, with explicit attention masks.
Dropout/jitter are active in train mode. Evaluation disables randomness but the
training entry point still updates weights; use FORWARD for loss-only evaluation.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

import setep
import switch_base_8_training as generated


class Model(generated.Model):
    def __init__(self, config, weights, learning_rate=0.001, max_grad_norm=None):
        super().__init__()
        self.config = dict(config)
        if not math.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError('learning_rate must be finite and nonnegative')
        self.learning_rate = learning_rate
        if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
            raise ValueError('max_grad_norm must be finite and positive, or None')
        self.max_grad_norm = max_grad_norm
        self.last_grad_norm = None
        self._gradient_scale = 1.0
        self.last_loss_components = {}
        expected = dict(d_model=generated.model_width, d_ff=generated.ff_width,
                        d_kv=generated.head_width, num_heads=generated.heads,
                        num_experts=generated.experts, num_layers=generated.encoder_layers,
                        num_decoder_layers=generated.decoder_layers, vocab_size=generated.vocab_size,
                        relative_attention_num_buckets=generated.relative_buckets)
        for key, value in expected.items():
            if self.config[key] != value:
                raise ValueError(f'{key}: checkpoint {self.config[key]} != SETEP {value}')
        if generated.model_width != generated.heads * generated.head_width:
            raise ValueError('This model requires model_width = heads * head_width')
        if self.config.get('router_bias') or self.config.get('dense_act_fn', 'relu') != 'relu':
            raise ValueError('This model requires bias-free routing and ReLU')
        if not self.config.get('tie_word_embeddings', True):
            raise ValueError('This model requires tied embeddings')
        self.mapping = list(self._weight_mapping())
        consumed = set()
        for field, path, name, transpose in self.mapping:
            bank = getattr(self, 'weight_' + field)
            value = weights[name].detach().float()
            if transpose:
                value = value.T
            elif value.ndim == 1:
                value = value.reshape(1, -1)
            if tuple(value.shape) != tuple(bank.declared_shape):
                raise ValueError(f'{name}: incompatible shape {tuple(value.shape)}')
            bank.set(path, value.clone().contiguous())
            consumed.add(name)
        for name in ('encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight'):
            if name in weights:
                if not torch.equal(weights[name], weights['shared.weight']):
                    raise ValueError(f'{name} must share the token embedding')
                consumed.add(name)
        if set(weights) - consumed:
            raise ValueError(f'Unmapped checkpoint tensors: {sorted(set(weights) - consumed)}')
        self._gradients = None
        self.last_gradients = {}

    @classmethod
    def from_checkpoint(cls, directory, **kwargs):
        directory = Path(directory)
        config = json.loads((directory / 'config.json').read_text())
        weights = torch.load(directory / 'pytorch_model.bin', map_location='cpu',
                             weights_only=True, mmap=True)
        return cls(config, weights, **kwargs)

    def sparse(self, side, layer):
        step = self.config[side + '_sparse_step']
        return step > 0 and (step == 1 or layer % step == 1)

    def _weight_mapping(self):
        yield 'wte', (), 'shared.weight', False
        for side, short, count in [('encoder', 'enc', generated.encoder_layers),
                                   ('decoder', 'dec', generated.decoder_layers)]:
            yield short + '_relative_bias', (), side + '.block.0.layer.0.SelfAttention.relative_attention_bias.weight', False
            yield short + '_final_scale', (), side + '.final_layer_norm.weight', False
            for layer in range(count):
                prefix = f'{side}.block.{layer}.layer.'
                for projection in ('q', 'k', 'v', 'o'):
                    yield short + '_' + projection, (layer,), prefix + '0.SelfAttention.' + projection + '.weight', True
                    if side == 'decoder':
                        yield 'cross_' + projection, (layer,), prefix + '1.EncDecAttention.' + projection + '.weight', True
                ff = 1 if side == 'encoder' else 2
                for norm in range(ff + 1):
                    yield short + '_ln' + str(norm + 1), (layer,), prefix + f'{norm}.layer_norm.weight', False
                mlp = prefix + f'{ff}.mlp.'
                if self.sparse(side, layer):
                    yield short + '_router', (layer,), mlp + 'router.classifier.weight', True
                    for expert in range(generated.experts):
                        for projection in ('wi', 'wo'):
                            yield short + '_expert_' + projection, (layer, expert), mlp + f'experts.expert_{expert}.{projection}.weight', True
                else:
                    for projection in ('wi', 'wo'):
                        yield short + '_' + projection, (layer,), mlp + projection + '.weight', True

    def reference_weights(self):
        """Return the current weights in Hugging Face's checkpoint layout."""
        result = {}
        for field, path, name, transpose in self.mapping:
            value = getattr(self, 'weight_' + field).at(path).detach()
            if transpose:
                value = value.T
            elif name.endswith('layer_norm.weight'):
                value = value.flatten()
            result[name] = value.cpu().clone().contiguous()
        for name in ('encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight'):
            result[name] = result['shared.weight']
        return result

    def LOOKUP(self, ids, table, **kwargs):
        return F.embedding(ids.long(), table)

    def SHIFT_RIGHT(self, targets, **kwargs):
        result = torch.empty_like(targets)
        result[0] = self.config['decoder_start_token_id']
        result[1:] = targets[:-1]
        return result.masked_fill(result == -100, self.config['pad_token_id'])

    def RMSNORM(self, value, scale, **kwargs):
        return value * torch.rsqrt(value.float().square().mean(-1, keepdim=True)
                                  + self.config['layer_norm_epsilon']) * scale

    def DROPOUT(self, value, **kwargs):
        return F.dropout(value, self.config['dropout_rate'], self.training)

    def ROUTER_JITTER(self, value, **kwargs):
        noise = self.config.get('router_jitter_noise', 0.01)
        if self.training and noise:
            return value * torch.empty_like(value).uniform_(1 - noise, 1 + noise)
        return value

    def RELU(self, value, **kwargs):
        return F.relu(value)

    def _bias(self, scores, table, head, bidirectional):
        relative = torch.arange(scores.shape[1], device=scores.device)[None, :] - torch.arange(scores.shape[0], device=scores.device)[:, None]
        buckets = generated.relative_buckets
        offset = torch.zeros_like(relative)
        if bidirectional:
            buckets //= 2
            offset = (relative > 0).long() * buckets
            distance = relative.abs()
        else:
            distance = (-relative).clamp_min(0)
        exact = buckets // 2
        large = exact + (torch.log(distance.float().clamp_min(1) / exact)
                         / math.log(self.config['relative_attention_max_distance'] / exact)
                         * (buckets - exact)).long()
        index = offset + torch.where(distance < exact, distance, large.clamp_max(buckets - 1))
        return table[index, head]

    def ENCODER_BIAS(self, scores, table, instance_path, **kwargs):
        return self._bias(scores, table, instance_path[-1], True)

    def DECODER_BIAS(self, scores, table, instance_path, **kwargs):
        return self._bias(scores, table, instance_path[-1], False)

    def SOFTMAX(self, scores, mask, **kwargs):
        return F.softmax(scores.float().masked_fill(~mask.bool(), torch.finfo(scores.dtype).min), -1).to(scores.dtype)

    def CAUSAL_MASK(self, mask, **kwargs):
        return torch.ones((mask.numel(), mask.numel()), dtype=torch.bool, device=mask.device).tril() & mask.bool()

    def HEAD_SCALE(self, value, **kwargs):
        return value * generated.model_width ** -0.5

    def ROUTE(self, logits, **kwargs):
        weights, ids = logits.float().softmax(-1).max(-1)
        assignment = F.one_hot(ids, generated.experts)
        accepted = ((assignment.cumsum(0) <= self.config['expert_capacity']) * assignment).any(-1)
        return generated.ROUTEResult(ids.masked_fill(~accepted, -1), weights)

    def DISPATCH(self, value, ids, weights, instance_path, **kwargs):
        positions = (ids == instance_path[-1]).nonzero().flatten()
        return generated.DISPATCHResult(value[positions], weights[positions], positions)

    def SCALE(self, value, weights, **kwargs):
        return value * weights[:, None]

    def SCATTER(self, value, positions, template, **kwargs):
        return torch.zeros_like(template).index_copy(0, positions, value)

    def ZERO_ROUTER_LOGITS(self, value, **kwargs):
        return value.new_zeros((value.shape[0], generated.experts))

    def UNPACK_FF(self, packed, **kwargs):
        return generated.UNPACK_FFResult(packed[:, :generated.model_width],
                                         packed[:, generated.model_width:])

    def SELECT_ENCODER_FF(self, choice_index, value, repetition=None):
        return torch.tensor([choice_index == int(self.sparse('encoder', repetition.index))], device=value.device)

    def SELECT_DECODER_FF(self, choice_index, value, repetition=None):
        return torch.tensor([choice_index == int(self.sparse('decoder', repetition.index))], device=value.device)

    def _router_loss(self, packed, side):
        logits, indices = [], []
        for layer, values in enumerate(packed.split(generated.experts, dim=-1)):
            if self.sparse(side, layer):
                logits.append(values)
                # HF 4.57.6 argmax of the capacity-masked one-hot mask returns
                # expert zero for overflow tokens. Match that auxiliary loss.
                indices.append(self.ROUTE(values).EXPERT_ID.clamp_min(0))
        if not logits:
            return packed.new_zeros(1)
        scores = torch.cat(logits, 0)
        ids = torch.cat(indices, 0)
        z_loss = scores.logsumexp(-1).square().mean()
        frequencies = F.one_hot(ids, generated.experts).float().mean(0)
        probabilities = scores.float().softmax(-1).mean(0)
        aux_loss = (frequencies * probabilities).mean() * generated.experts ** 2
        return (z_loss * self.config.get('router_z_loss_coef', 0.001)
                + aux_loss * self.config.get('router_aux_loss_coef', 0.001)).reshape(1)

    def ENCODER_ROUTER_LOSS(self, packed, **kwargs):
        return self._router_loss(packed, 'encoder')

    def DECODER_ROUTER_LOSS(self, packed, **kwargs):
        return self._router_loss(packed, 'decoder')

    def CROSS_ENTROPY(self, logits, targets, **kwargs):
        return setep.cross_entropy(logits, targets)

    def GRADIENT(self, loss, parameter, **kwargs):
        # One reverse traversal computes all VJPs of the shared scalar loss.
        # Each SETEP GRADIENT call still names its parameter dependency and
        # materializes the corresponding result before the UPDATE phase.
        if self._gradients is None:
            if not torch.isfinite(loss).all():
                raise ValueError('Nonfinite training loss; no weights were updated')
            parameters = tuple(self.parameters())
            gradients = torch.autograd.grad(loss.sum(), parameters, allow_unused=True)
            self._gradients = {id(p): torch.zeros_like(p) if g is None else g.detach()
                               for p, g in zip(parameters, gradients)}
            norm = torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(g) for g in self._gradients.values()]))
            self.last_grad_norm = norm.item()
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradient norm; no weights were updated')
            if self.max_grad_norm is not None:
                self._gradient_scale = (self.max_grad_norm / (norm + 1e-6)).clamp(max=1.0)
        return self._gradients[id(parameter)]

    def SCALE_GRADIENT(self, gradient, **kwargs):
        return gradient * self._gradient_scale * self.learning_rate

    def REPORT_LOSS(self, loss, **kwargs):
        return loss.detach().clone()

    def FORWARD(self, *args, **kwargs):
        result = super().FORWARD(*args, **kwargs)
        self.last_loss_components = dict(total=result.LOSS.item(), token=result.TOKEN_LOSS.item(),
                                        encoder_router=result.ENC_ROUTER_LOSS.item(),
                                        decoder_router=result.DEC_ROUTER_LOSS.item())
        return result

    def _validate_inputs(self, input_ids, attention_mask, targets, decoder_mask):
        if decoder_mask is None:
            decoder_mask = torch.ones_like(targets, dtype=torch.bool)
        for name, ids, limit in [('input_ids', input_ids, generated.tokens),
                                 ('targets', targets, generated.target_tokens)]:
            if ids.ndim != 1 or not 0 < ids.numel() <= limit:
                raise ValueError(f'{name} must contain 1..{limit} token IDs')
        if attention_mask.shape != input_ids.shape or decoder_mask.shape != targets.shape:
            raise ValueError('Attention mask shapes must match their token sequences')
        if not attention_mask.bool().any() or not decoder_mask.bool().any():
            raise ValueError('Each attention mask must contain at least one valid key')
        return decoder_mask

    def _clear_state(self):
        for module in self.modules():
            if isinstance(module, setep.StateBank):
                module.clear()

    def evaluate_loss(self, input_ids, attention_mask, targets, decoder_mask=None):
        """Deterministic loss only: no dropout, router jitter, backward, or updates."""
        decoder_mask = self._validate_inputs(input_ids, attention_mask, targets, decoder_mask)
        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                self.FORWARD(input_ids, attention_mask, targets, decoder_mask)
            return dict(self.last_loss_components)
        finally:
            self.train(was_training)
            self._clear_state()

    def forward(self, input_ids, attention_mask, targets, decoder_mask=None):
        decoder_mask = self._validate_inputs(input_ids, attention_mask, targets, decoder_mask)
        self._gradients = None
        self.last_gradients = {}
        self.last_grad_norm = None
        self._gradient_scale = 1.0
        try:
            result = super().forward(input_ids, attention_mask, targets, decoder_mask)
            self.last_gradients = {name: self._gradients[id(parameter)]
                                   for name, parameter in self.named_parameters()}
            return result
        finally:
            self._gradients = None
            self._clear_state()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=Path(__file__).with_name('switch-base-8'))
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--save-weights', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Clip the global gradient norm before SGD; 0 disables clipping')
    parser.add_argument('--dropout', type=float, help='Override checkpoint dropout (0 for a single-example fit check)')
    parser.add_argument('--router-jitter', type=float, help='Override checkpoint router noise')
    parser.add_argument('--eval-every', type=int, default=1,
                        help='Report deterministic post-update loss every N steps; 0 disables evaluation')
    parser.add_argument('source')
    parser.add_argument('target')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm < 0:
        parser.error('--max-grad-norm must be finite and nonnegative')
    if args.eval_every < 0:
        parser.error('--eval-every must be nonnegative')
    for name in ('dropout', 'router_jitter'):
        value = getattr(args, name)
        if value is not None and not 0 <= value < 1:
            parser.error('--' + name.replace('_', '-') + ' must be in [0, 1)')
    from transformers import AutoTokenizer
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.weights, local_files_only=True)
    model = Model.from_checkpoint(args.weights, learning_rate=args.learning_rate,
                                  max_grad_norm=args.max_grad_norm or None).to(args.device).train()
    if args.dropout is not None:
        model.config['dropout_rate'] = args.dropout
    if args.router_jitter is not None:
        model.config['router_jitter_noise'] = args.router_jitter
    source = tokenizer(args.source, return_tensors='pt')
    ids, mask = source.input_ids[0].to(args.device), source.attention_mask[0].to(args.device)
    target = tokenizer(args.target, return_tensors='pt').input_ids[0].to(args.device)
    print(f'SGD lr={model.learning_rate:g}, max_grad_norm={model.max_grad_norm}, '
          f'dropout={model.config["dropout_rate"]}, router_jitter={model.config.get("router_jitter_noise", 0.01)}', flush=True)
    if args.eval_every:
        initial = model.evaluate_loss(ids, mask, target)
        print(f'Initial eval_loss={initial["total"]:.6f}, eval_token={initial["token"]:.6f}', flush=True)
    for step in range(args.steps):
        loss = model(ids, mask, target)
        parts = model.last_loss_components
        message = (f'Step {step + 1}: loss={loss.item():.6f}, token={parts["token"]:.6f}, '
                   f'router={parts["encoder_router"] + parts["decoder_router"]:.6f}, '
                   f'grad_norm={model.last_grad_norm:.4f}')
        if args.eval_every and (step + 1) % args.eval_every == 0:
            evaluation = model.evaluate_loss(ids, mask, target)
            message += f', eval_loss={evaluation["total"]:.6f}, eval_token={evaluation["token"]:.6f}'
        print(message, flush=True)
    if args.save_weights:
        torch.save(model.reference_weights(), args.save_weights)


if __name__ == '__main__':
    main()
