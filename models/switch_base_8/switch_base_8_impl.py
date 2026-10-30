"""Handwritten inference operations for SETEP Switch Base-8.

Uses local Hugging Face files; model arithmetic and expert loops are generated.
Reference behavior: transformers 4.57.6, float32 evaluation, greedy decoding.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

import switch_base_8 as generated
import setep


class Model(generated.Model):
    def __init__(self, checkpoint=None):
        super().__init__()
        directory = Path(checkpoint) if checkpoint else Path(__file__).with_name('switch-base-8')
        self.config = json.loads((directory / 'config.json').read_text())
        expected = dict(d_model=generated.model_width, d_ff=generated.ff_width,
                        d_kv=generated.head_width, num_heads=generated.heads,
                        num_experts=generated.experts, num_layers=generated.encoder_layers,
                        num_decoder_layers=generated.decoder_layers, vocab_size=generated.vocab_size)
        for key, value in expected.items():
            if self.config[key] != value:
                raise ValueError(f'{key}: checkpoint {self.config[key]} != SETEP {value}')
        if self.config.get('router_bias') or self.config.get('dense_act_fn') != 'relu':
            raise ValueError('This SETEP model requires bias-free routing and ReLU')
        if not self.config.get('tie_word_embeddings', True):
            raise ValueError('This SETEP model requires tied embeddings')
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError('Install data/switch_base_8/requirements-switch-base-8.txt') from error
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.norms = nn.ParameterDict()
        self.biases = nn.ParameterDict()
        self._load_checkpoint(directory / 'pytorch_model.bin')
        self.last_logits = []
        self.token_ids = []
        self.eval()

    def _load_checkpoint(self, filename):
        tensors = torch.load(filename, map_location='cpu', weights_only=True, mmap=True)
        used = set()
        def take(name, shape):
            value = tensors[name]
            if tuple(value.shape) != tuple(shape):
                raise ValueError(f'{name}: expected {shape}, got {tuple(value.shape)}')
            used.add(name)
            return value.float()
        def bank(field, path, name, transpose=True):
            target = getattr(self, field)
            shape = target.declared_shape
            value = take(name, shape[::-1] if transpose else shape)
            target.set(path, value.T.contiguous() if transpose else value)
        bank('weight_wte', (), 'shared.weight', False)
        for alias in ('encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight'):
            if alias in tensors:
                if not torch.equal(tensors[alias], tensors['shared.weight']):
                    raise ValueError(f'{alias} is not tied to shared.weight')
                used.add(alias)
        for side, count in [('encoder', generated.encoder_layers), ('decoder', generated.decoder_layers)]:
            upper = side.upper()
            sparse_step = self.config[side + '_sparse_step']
            for layer in range(count):
                prefix = f'{side}.block.{layer}.layer.'
                for projection in ('q', 'k', 'v', 'o'):
                    field = 'weight_ENCODER_ATTENTION_' if side == 'encoder' else 'weight_self_'
                    bank(field + projection, (layer,), prefix + '0.SelfAttention.' + projection + '.weight')
                    if side == 'decoder':
                        bank('weight_CROSS_ATTENTION_' + projection, (layer,),
                             prefix + '1.EncDecAttention.' + projection + '.weight')
                ff = 1 if side == 'encoder' else 2
                for norm in range(ff + 1):
                    self.norms[f'{side}_{layer}_{norm}'] = nn.Parameter(take(
                        prefix + f'{norm}.layer_norm.weight', (generated.model_width,)))
                mlp = prefix + f'{ff}.mlp.'
                sparse = sparse_step > 0 and (sparse_step == 1 or layer % sparse_step == 1)
                if sparse:
                    bank('weight_' + upper + '_SWITCH_FF_router', (layer,), mlp + 'router.classifier.weight')
                    for expert in range(generated.experts):
                        for projection in ('wi', 'wo'):
                            bank('weight_' + upper + '_EXPERT_' + projection, (layer, expert),
                                 mlp + f'experts.expert_{expert}.{projection}.weight')
                else:
                    for projection in ('wi', 'wo'):
                        bank('weight_' + upper + '_DENSE_FF_' + projection, (layer,), mlp + projection + '.weight')
            self.norms[side + '_final'] = nn.Parameter(take(side + '.final_layer_norm.weight', (generated.model_width,)))
            self.biases[side] = nn.Parameter(take(side + '.block.0.layer.0.SelfAttention.relative_attention_bias.weight',
                                                  (self.config['relative_attention_num_buckets'], generated.heads)))
        extra = set(tensors) - used
        if extra:
            raise ValueError(f'Unmapped checkpoint parameters: {sorted(extra)}')
        setep._log(f'Loaded {len(used)} checkpoint tensors')

    def TOKENIZE(self, value, **kwargs):
        if value.ndim != 1 or value.dtype != torch.uint8:
            raise TypeError('Expected a vector of UTF-8 bytes')
        text = bytes(value.cpu().tolist()).decode('utf-8')
        encoded = self.tokenizer(text, return_tensors='pt', truncation=False)
        ids = encoded.input_ids[0].to(value.device)
        if ids.numel() > generated.tokens:
            raise ValueError(f'Input exceeds {generated.tokens} tokens including EOS; increase SETEP tokens')
        return generated.TOKENIZEResult(ids, encoded.attention_mask[0].to(value.device).bool(),
                                       torch.arange(ids.numel(), device=value.device))

    def DETOKENIZE(self, value, **kwargs):
        ids = value.tolist()
        eos = self.config['eos_token_id']
        if eos in ids:
            ids = ids[:ids.index(eos)]
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return torch.tensor(list(text.encode('utf-8')), dtype=torch.uint8, device=value.device)

    def LOOKUP(self, ids, table, **kwargs):
        return F.embedding(ids, table)

    def RMSNORM(self, value, instance_path, occurrence):
        side = 'encoder' if occurrence.startswith('ENCODER_') else 'decoder'
        key = f'{side}_{instance_path[-1]}_{occurrence.rsplit("_", 1)[1]}' if '_BLOCK_' in occurrence else side + '_final'
        return value * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + self.config['layer_norm_epsilon']) * self.norms[key]

    def RELU(self, value, **kwargs):
        return F.relu(value)

    def RELATIVE_POSITION_BIAS(self, scores, instance_path, occurrence):
        if len(instance_path) != 2:
            raise ValueError('Regenerate SETEP with RELATIVE_POSITION_BIAS#() to pass the head index')
        side = 'encoder' if occurrence.startswith('ENCODER_') else 'decoder'
        q, k = scores.shape
        query = torch.arange(k - q if side == 'decoder' else 0, k if side == 'decoder' else q, device=scores.device)
        relative = torch.arange(k, device=scores.device)[None, :] - query[:, None]
        buckets = self.config['relative_attention_num_buckets']
        offset = torch.zeros_like(relative)
        if side == 'encoder':
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
        return self.biases[side][index, instance_path[-1]]

    def CAUSAL_MASK(self, scores, **kwargs):
        return torch.ones_like(scores, dtype=torch.bool).tril(diagonal=scores.shape[-1] - scores.shape[-2])

    def SOFTMAX(self, scores, mask, **kwargs):
        # T5 attention does not divide scores by sqrt(head_width).
        return F.softmax(scores.float().masked_fill(~mask.bool(), torch.finfo(scores.dtype).min), dim=-1).to(scores.dtype)

    def ROUTE(self, logits, **kwargs):
        probabilities = logits.float().softmax(-1)
        weights, ids = probabilities.max(-1)
        assignment = F.one_hot(ids, generated.experts)
        priority = assignment.cumsum(0)
        accepted = ((priority <= self.config['expert_capacity']) * assignment).any(-1)
        return generated.ROUTEResult(ids.masked_fill(~accepted, -1), weights)

    def DISPATCH(self, value, ids, weights, instance_path, occurrence):
        positions = (ids == instance_path[-1]).nonzero().flatten()
        return generated.DISPATCHResult(value[positions], weights[positions], positions)

    def SCALE(self, value, weights, **kwargs):
        return value * weights[:, None]

    def SCATTER(self, value, positions, template, **kwargs):
        result = torch.zeros_like(template)
        result[positions] = value
        return result

    def COMBINE_ROUTED(self, value, routed, ids, weights, **kwargs):
        # transformers 4.57.6 leaves capacity-overflow rows zero in the FF branch.
        return routed

    def _select(self, index, value, repetition, side):
        step = self.config[side + '_sparse_step']
        sparse = step > 0 and (step == 1 or repetition.index % step == 1)
        return torch.tensor([index == int(sparse)], device=value.device)

    def SELECT_ENCODER_FF(self, choice_index, value, repetition=None):
        return self._select(choice_index, value, repetition, 'encoder')

    def SELECT_DECODER_FF(self, choice_index, value, repetition=None):
        return self._select(choice_index, value, repetition, 'decoder')

    def DECODER_START(self, value, **kwargs):
        return torch.tensor([self.config['decoder_start_token_id']], device=value.device)

    def DECODE_PREP(self, value, **kwargs):
        return value

    def LMHEAD(self, value, instance_path=(), state_path=()):
        return super().LMHEAD(value * generated.model_width**-.5, instance_path, state_path)

    def SAMPLE(self, value, **kwargs):
        self.last_logits.append(value[-1].detach().clone())
        token = value[-1].argmax().reshape(1)
        self.token_ids.append(token.item())
        return token

    def forward(self, value):
        self.reset_state()
        self.last_logits = []
        self.token_ids = []
        return super().forward(value)


def main():
    parser = argparse.ArgumentParser(description='SETEP Switch Base-8 checkpoint inference')
    parser.add_argument('prompt', nargs='?', default='The capital of France is <extra_id_0>.')
    parser.add_argument('--weights', type=Path, default=Path(__file__).with_name('switch-base-8'), help='Directory containing checkpoint, config and tokenizer files')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    setep.set_debug(args.debug)
    torch.set_num_threads(1)
    model = Model(args.weights).eval()
    value = torch.tensor(list(args.prompt.encode('utf-8')), dtype=torch.uint8)
    with torch.no_grad():
        result = model(value)
    text = bytes(result.tolist()).decode('utf-8')
    setep._log(f'TEXT_OUT={text!r}', always=True)


if __name__ == '__main__':
    main()
