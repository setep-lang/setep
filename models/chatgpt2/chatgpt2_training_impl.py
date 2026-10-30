"""Handwritten implementations for the generated GPT-2 training step.

Text enters as UTF-8 uint8 tensors. Generated SETEP executes forward, reverse
BLOCK traversal, explicit gradients, and in-place SGD weight assignments.
"""
import argparse
import math
import torch
from torch.nn import functional as F
import chatgpt2_training as generated
import setep


class TrainingWeightBank(setep.WeightBank):
    def __init__(self, shape, data_size, learning_rate):
        super().__init__(shape, data_size)
        self.learning_rate = learning_rate

    def update(self, instance_path, value):
        # SETEP applies eta to the gradient; CLI may override that step size.
        parameter = self.at(instance_path)
        value = parameter + (value - parameter) * (self.learning_rate / generated.eta)
        return super().update(instance_path, value)


class Model(generated.Model):
    def __init__(self, seed=0, checkpoint=None, learning_rate=None):
        super().__init__()
        import tiktoken
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.learning_rate = generated.eta if learning_rate is None else learning_rate
        if not math.isfinite(self.learning_rate) or self.learning_rate < 0 or generated.eta == 0:
            raise ValueError('learning rate must be finite and nonnegative; SETEP eta must be nonzero')
        self._gradients = {}
        self._gradient_key = {}
        self.backward_layers = []
        block_names = ('q', 'k', 'v', 'o', 'qb', 'kb', 'vb', 'ob', 'up', 'up_bias',
                       'down', 'down_bias', 'ln1_scale', 'ln1_bias', 'ln2_scale', 'ln2_bias')
        self.banks = torch.nn.ModuleDict()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for name in (*block_names, 'wte', 'wpe', 'lnf_scale', 'lnf_bias'):
                bank = TrainingWeightBank(generated.WEIGHT_SHAPES[name],
                                          generated.WEIGHT_DATA_SIZES[name], self.learning_rate)
                self.banks[name] = bank
                for path in ([(layer,) for layer in range(generated.layers)] if name in block_names else [()]):
                    shape = bank.declared_shape
                    value = torch.ones(shape) if name.endswith('_scale') else torch.zeros(shape) if shape[0] == 1 else torch.randn(shape) * .02
                    bank.set(path, value)
            # Match every generated read and write to the same named weight bank.
            for field, bank in list(self._modules.items()):
                if isinstance(bank, setep.WeightBank):
                    name = next(name for name in sorted(self.banks, key=len, reverse=True)
                                if field.endswith('_' + name))
                    setattr(self, field, self.banks[name])
        if checkpoint:
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, filename):
        from safetensors.torch import load_file
        values = load_file(str(filename))
        copies = []
        def copy(name, path, source):
            target = self.banks[name].at(path)
            if source.numel() != target.numel() or (source.ndim == 2 and source.shape != target.shape):
                raise ValueError(f'Checkpoint shape mismatch for {name} at {path}')
            copies.append((target, source.reshape(target.shape)))
        for name in ('wte', 'wpe'):
            copy(name, (), values[name + '.weight'])
        for layer in range(generated.layers):
            prefix = f'h.{layer}.'
            for name, weight, bias in zip(('q', 'k', 'v'), values[prefix+'attn.c_attn.weight'].chunk(3, -1),
                                        values[prefix+'attn.c_attn.bias'].chunk(3)):
                copy(name, (layer,), weight)
                copy(name+'b', (layer,), bias)
            for name, key in (('o','attn.c_proj.weight'), ('ob','attn.c_proj.bias'),
                              ('up','mlp.c_fc.weight'), ('up_bias','mlp.c_fc.bias'),
                              ('down','mlp.c_proj.weight'), ('down_bias','mlp.c_proj.bias'),
                              ('ln1_scale','ln_1.weight'), ('ln1_bias','ln_1.bias'),
                              ('ln2_scale','ln_2.weight'), ('ln2_bias','ln_2.bias')):
                copy(name, (layer,), values[prefix+key])
        copy('lnf_scale', (), values['ln_f.weight'])
        copy('lnf_bias', (), values['ln_f.bias'])
        with torch.no_grad():
            for target, value in copies:
                target.copy_(value)

    def save_checkpoint(self, filename):
        """Export current weights in the GPT-2 layout accepted by inference."""
        from safetensors.torch import save_file
        values = {}
        def weight(name, path=()):
            return self.banks[name].at(path).detach().cpu().contiguous()
        for name in ('wte', 'wpe'):
            values[name + '.weight'] = weight(name)
        for layer in range(generated.layers):
            path = (layer,)
            prefix = f'h.{layer}.'
            values[prefix + 'attn.c_attn.weight'] = torch.cat(
                [weight(name, path) for name in ('q', 'k', 'v')], dim=1)
            values[prefix + 'attn.c_attn.bias'] = torch.cat(
                [weight(name, path).flatten() for name in ('qb', 'kb', 'vb')])
            for name, key in (('o','attn.c_proj.weight'), ('ob','attn.c_proj.bias'),
                              ('up','mlp.c_fc.weight'), ('up_bias','mlp.c_fc.bias'),
                              ('down','mlp.c_proj.weight'), ('down_bias','mlp.c_proj.bias'),
                              ('ln1_scale','ln_1.weight'), ('ln1_bias','ln_1.bias'),
                              ('ln2_scale','ln_2.weight'), ('ln2_bias','ln_2.bias')):
                value = weight(name, path)
                values[prefix + key] = value.flatten() if name.endswith(('_bias', '_scale')) or name == 'ob' else value
        values['ln_f.weight'] = weight('lnf_scale').flatten()
        values['ln_f.bias'] = weight('lnf_bias').flatten()
        # The token table is shared with the LM head; serialize it only once.
        save_file(values, str(filename), metadata={'format': 'pt'})

    def TRAIN_TOKENIZE(self, value, instance_path=(), occurrence='TRAIN_TOKENIZE'):
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.dtype != torch.uint8:
            raise TypeError('TRAIN_TOKENIZE expects a one-dimensional UTF-8 uint8 tensor')
        text = bytes(value.detach().cpu().tolist()).decode('utf-8')
        ids = self.tokenizer.encode_ordinary(text) or [self.tokenizer.eot_token]
        if len(ids) > generated.tokens:
            raise ValueError(f'Training text exceeds {generated.tokens} tokens')
        device = self.banks['wte'].at(()).device
        padding = generated.tokens - len(ids)
        inputs = torch.tensor(ids + [self.tokenizer.eot_token] * padding, device=device)
        targets = torch.tensor(ids[1:] + [self.tokenizer.eot_token] + [-100] * padding, device=device)
        mask = torch.arange(generated.tokens, device=device) < len(ids)
        positions = torch.arange(generated.tokens, device=device)
        return generated.TRAIN_TOKENIZEResult(inputs, mask, positions, targets)

    def NORMALIZE(self, value, instance_path, occurrence):
        return setep.normalize(value)

    def SCALE(self, value, weight, instance_path, occurrence):
        return value * weight

    def SOFTMAX(self, value, mask, instance_path, occurrence):
        return setep.causal_softmax(value / math.sqrt(generated.model_width // generated.heads), mask)

    def GELU(self, value, instance_path, occurrence):
        return F.gelu(value, approximate='tanh')

    def LOOKUP(self, ids, table, instance_path, occurrence):
        return F.embedding(ids.long(), table)

    def CROSS_ENTROPY(self, logits, targets, instance_path, occurrence):
        return setep.cross_entropy(logits, targets)

    def CROSS_ENTROPY_BACK(self, logits, targets, instance_path, occurrence):
        return setep.local_gradients(lambda x: setep.cross_entropy(x, targets), [logits], torch.ones(1, device=logits.device))[0]

    def LMHEAD_INPUT_GRAD(self, value, gradient, instance_path, occurrence):
        return gradient @ self.banks['wte'].at(())

    def LMHEAD_WEIGHT_GRAD(self, value, gradient, instance_path, occurrence):
        return gradient.T @ value

    def LOOKUP_GRAD(self, ids, gradient, instance_path, occurrence):
        name = 'wte' if occurrence.endswith('_0') else 'wpe'
        result = torch.zeros_like(self.banks[name].at(()))
        return result.index_add_(0, ids.long(), gradient)

    def _group_gradients(self, group, value, upstream, instance_path, occurrence, mask=None):
        path = tuple(instance_path)
        if group == 'LN':
            norm = 'lnf' if not path else 'ln2' if occurrence.endswith('_0') else 'ln1'
            names = (norm+'_scale', norm+'_bias')
            function = lambda x, scale, bias: setep.normalize(x) * scale + bias
            key = (path, norm, value.data_ptr(), upstream.data_ptr())
        elif group == 'MLP':
            names = ('up', 'up_bias', 'down', 'down_bias')
            function = setep.mlp
            key = (path, value.data_ptr(), upstream.data_ptr())
        else:
            names = ('q','k','v','o','qb','kb','vb','ob')
            function = lambda x, *weights: setep.attention(x, mask, *weights, generated.heads)
            key = (path, value.data_ptr(), upstream.data_ptr())
        if self._gradient_key.get(group) != key:
            weights = [self.banks[name].at(path) for name in names]
            self._gradients[group] = dict(zip(('INPUT', *names), setep.local_gradients(function, [value, *weights], upstream)))
            self._gradient_key[group] = key
            if group == 'ATTN':
                self.backward_layers.append(path[-1])
        return self._gradients[group], names

    def forward(self, TEXT_IN):
        self._gradients.clear()
        self._gradient_key.clear()
        self.backward_layers.clear()
        try:
            # Each explicit backward operation recomputes its own local graph.
            with torch.no_grad():
                return super().forward(TEXT_IN).detach().clone()
        finally:
            self._gradients.clear()
            self._gradient_key.clear()
            self.reset_state()


def _backward_operation(group, output):
    def operation(self, value, *args, instance_path, occurrence):
        mask, upstream = args if group == 'ATTN' else (None, args[0])
        gradients, names = self._group_gradients(group, value, upstream, instance_path, occurrence, mask)
        key = 'INPUT' if output == 'INPUT' else names[0 if output == 'SCALE' else 1] if group == 'LN' else output.lower()
        return gradients[key]
    return operation


for _group, _outputs in (('LN', ('INPUT','SCALE','BIAS')),
                         ('MLP', ('INPUT','UP','UP_BIAS','DOWN','DOWN_BIAS')),
                         ('ATTN', ('INPUT','Q','K','V','O','Q_BIAS','K_BIAS','V_BIAS','O_BIAS'))):
    for _output in _outputs:
        _operation = _backward_operation(_group, _output.replace('_BIAS', 'B') if _group == 'ATTN' else _output)
        _operation.__name__ = f'{_group}_{_output}_GRAD'
        setattr(Model, _operation.__name__, _operation)


def main():
    parser = argparse.ArgumentParser(description='One or more real GPT-2 SGD training steps')
    parser.add_argument('prompt', nargs='?', default='Paris is the capital of France.')
    parser.add_argument('--weights', help='Initial GPT-2 safetensors checkpoint')
    parser.add_argument('--save-weights', help='Save final weights as a GPT-2 safetensors checkpoint')
    parser.add_argument('--learning-rate', type=float, default=generated.eta)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    setep.set_debug(args.debug)
    torch.set_num_threads(1)
    model = Model(checkpoint=args.weights, learning_rate=args.learning_rate)
    text = torch.tensor(list(args.prompt.encode('utf-8')), dtype=torch.uint8)
    for step in range(args.steps):
        loss = model(text)
        setep._log(f'Training step {step+1}: LOSS={loss.item():.6f}', always=True)
    if args.save_weights:
        model.save_checkpoint(args.save_weights)
        setep._log(f'Saved weights: {args.save_weights}', always=True)


if __name__ == '__main__':
    main()
