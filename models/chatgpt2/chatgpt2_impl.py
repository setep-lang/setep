"""Handwritten PyTorch implementations for chatgpt2.py; never generated.

Random parameters demonstrate execution; --weights loads a GPT-2 checkpoint.
"""
import argparse
import math

import torch
from torch import nn
import torch.nn.functional as F

import chatgpt2


class Model(chatgpt2.Model):
    def __init__(self, seed=0, checkpoint=None):
        super().__init__()
        self._kv_mode = hasattr(self, 'PREFILL')
        self.register_buffer('_attention_mask', None, persistent=False)
        self._next_position = 0
        try:
            import tiktoken
        except ImportError as error:
            raise ImportError('Install the tokenizer with: python -m pip install tiktoken') from error
        self.tokenizer = tiktoken.get_encoding('gpt2')
        # Construct all parameters before forward so optimizers see them all.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self._initialize_bank(self.weight_wte, ())
            self._initialize_bank(self.weight_wpe, ())
            self.layer_norms = nn.ModuleDict()
            for block in range(chatgpt2.layers):
                for occurrence in ('LN_0', 'LN_1'):
                    key = self._norm_key(
                        (block,), 'CHATGPT2_INFERENCE_BLOCK_' + occurrence)
                    self.layer_norms[key] = nn.LayerNorm(chatgpt2.model_width)
                for name in ('q', 'k', 'v', 'o', 'qb', 'kb', 'vb', 'ob'):
                    self._initialize_bank(
                        getattr(self, 'weight_' + name),
                        (block,))
                self._initialize_bank(self.weight_up, (block,))
                self._initialize_bank(self.weight_down, (block,))
                self._initialize_bank(self.weight_up_bias, (block,))
                self._initialize_bank(self.weight_down_bias, (block,))
            self.layer_norms[self._norm_key(
                (), 'CHATGPT2_INFERENCE_LN_0')] = nn.LayerNorm(
                    chatgpt2.model_width)
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

    @staticmethod
    def _initialize_bank(bank, instance_path):
        rows, cols = bank.declared_shape
        bank.set(instance_path, torch.zeros(rows, cols) if rows == 1 else
                 torch.randn(rows, cols) / math.sqrt(rows))

    def load_checkpoint(self, filename):
        """Load GPT-2 projections and norms; the head reuses the token table."""
        from safetensors.torch import load_file
        tensors = load_file(str(filename))
        assignments = []

        def copy(destination, name, value=None):
            source = tensors[name] if value is None else value
            if tuple(source.shape) != tuple(destination.shape):
                raise ValueError(f'{name}: expected {tuple(destination.shape)}, '
                                 f'got {tuple(source.shape)}')
            assignments.append((destination, source))

        copy(self.weight_wte.at(()), 'wte.weight')
        copy(self.weight_wpe.at(()), 'wpe.weight')
        for block in range(chatgpt2.layers):
            prefix = f'h.{block}.'
            qkv = tensors[prefix + 'attn.c_attn.weight']
            qkv_bias = tensors[prefix + 'attn.c_attn.bias']
            if qkv.shape != (chatgpt2.model_width, 3 * chatgpt2.model_width) or qkv_bias.shape != (3 * chatgpt2.model_width,):
                raise ValueError(f'{prefix}invalid packed Q/K/V shapes')
            for name, weight, bias in zip(('q', 'k', 'v'),
                    qkv.chunk(3, dim=1), qkv_bias.chunk(3)):
                copy(getattr(self, 'weight_' + name).at((block,)),
                     prefix + 'attn.c_attn.weight', weight)
                copy(getattr(self, 'weight_' + name + 'b').at((block,)),
                     prefix + 'attn.c_attn.bias', bias.unsqueeze(0))
            for bank, name in ((self.weight_o, 'attn.c_proj.weight'),
                               (self.weight_ob, 'attn.c_proj.bias'),
                               (self.weight_up, 'mlp.c_fc.weight'),
                               (self.weight_up_bias, 'mlp.c_fc.bias'),
                               (self.weight_down, 'mlp.c_proj.weight'),
                               (self.weight_down_bias, 'mlp.c_proj.bias')):
                value = tensors[prefix + name]
                copy(bank.at((block,)), prefix + name,
                     value.unsqueeze(0) if value.ndim == 1 else value)
            for occurrence, name in (('LN_0', 'ln_1'), ('LN_1', 'ln_2')):
                norm = self.layer_norms[self._norm_key(
                    (block,), 'CHATGPT2_INFERENCE_BLOCK_' + occurrence)]
                copy(norm.weight, prefix + name + '.weight')
                copy(norm.bias, prefix + name + '.bias')
        norm = self.layer_norms[self._norm_key((), 'CHATGPT2_INFERENCE_LN_0')]
        copy(norm.weight, 'ln_f.weight')
        copy(norm.bias, 'ln_f.bias')
        # Validate every shape before changing any parameter. Copying in place
        # preserves optimizer references and the embedding/head weight identity.
        with torch.no_grad():
            for destination, source in assignments:
                destination.copy_(source)
        chatgpt2._log(f'Loaded {len(assignments)} checkpoint parameter tensors')

    @staticmethod
    def _norm_key(instance_path, occurrence):
        return occurrence + '_' + chatgpt2.WeightBank.key(instance_path)

    def TOKEN_WEIGHTS(self, value, instance_path=(), state_path=()):
        return super().TOKEN_WEIGHTS(value, (), state_path)

    def POSITION_WEIGHTS(self, value, instance_path=(), state_path=()):
        return super().POSITION_WEIGHTS(value, (), state_path)

    def MLP(self, value, instance_path=(), state_path=()):
        return super().MLP(value, instance_path[-1:] if self._kv_mode else instance_path, state_path)

    def TOKENIZE(self, value: torch.Tensor, instance_path=(), occurrence='TOKENIZE'):
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.dtype != torch.uint8:
            raise TypeError('TOKENIZE expects a one-dimensional uint8 UTF-8 byte tensor')
        text = bytes(value.detach().cpu().tolist()).decode('utf-8')
        # Treat special-token spellings in user text as ordinary text.
        ids = self.tokenizer.encode_ordinary(text)
        if len(ids) > chatgpt2.tokens:
            raise ValueError(f'Prompt exceeds the {chatgpt2.tokens}-token context')
        chatgpt2._log(f'TOKENIZE {value.numel()} bytes -> {len(ids)} tokens')
        # An empty prompt starts with one real end-of-text token.
        ids = ids or [self.tokenizer.eot_token]
        padding = chatgpt2.tokens - len(ids)
        device = self.weight_wte.at(()).device
        token_ids = torch.tensor([self.tokenizer.eot_token] * padding + ids,
                                 dtype=torch.long, device=device)
        mask = torch.arange(chatgpt2.tokens, device=device) >= padding
        positions = (mask.long().cumsum(0) - 1).clamp_min(0)
        self._attention_mask = mask
        self._next_position = len(ids)
        return chatgpt2.TOKENIZEResult(token_ids, mask, positions)

    def DECODE_PREP(self, value, instance_path, occurrence):
        if self._attention_mask is None:
            raise RuntimeError('Run TOKENIZE and PREFILL before DECODE')
        if value.numel() != 1 or value.dtype not in (torch.int32, torch.int64):
            raise ValueError('DECODE_PREP expects one integer token ID')
        if self._attention_mask.numel() >= chatgpt2.positions:
            raise ValueError('Decode would exceed the model context size')
        self._attention_mask = torch.cat((self._attention_mask,
            torch.ones(1, dtype=torch.bool, device=value.device)))
        position = torch.tensor([self._next_position], device=value.device)
        self._next_position += 1
        chatgpt2._log(f'DECODE_PREP position={position.item()} keys={self._attention_mask.numel()}')
        return chatgpt2.DECODE_PREPResult(value.reshape(1), self._attention_mask, position)

    def DETOKENIZE(self, value, instance_path=(), occurrence='DETOKENIZE'):
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise TypeError('DETOKENIZE expects a one-dimensional token tensor')
        if value.dtype not in (torch.int32, torch.int64):
            raise TypeError('DETOKENIZE expects integer token IDs')
        ids = value.detach().cpu().tolist()
        if any(token < 0 or token >= self.tokenizer.n_vocab for token in ids):
            raise ValueError('DETOKENIZE token ID is outside the GPT-2 vocabulary')
        ids = [token for token in ids if token != self.tokenizer.eot_token]
        # Preserve raw token bytes, including incomplete UTF-8 at generation boundaries.
        data = self.tokenizer.decode_bytes(ids)
        chatgpt2._log(f'DETOKENIZE {value.numel()} tokens -> {len(data)} bytes')
        return torch.tensor(list(data), dtype=torch.uint8, device=value.device)

    def forward(self, TEXT_IN: torch.Tensor) -> torch.Tensor:
        return super().forward(TEXT_IN)

    def LOOKUP(self, ids, table, instance_path, occurrence):
        chatgpt2._log(f'LOOKUP {tuple(ids.shape)} {tuple(table.shape)}')
        if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise TypeError('LOOKUP expects a vector of integer indices')
        return F.embedding(ids, table)

    def LN(self, value, instance_path, occurrence):
        chatgpt2._log(f'LN {instance_path} {occurrence}')
        if self._kv_mode:
            instance_path = instance_path[-1:] if '_BLOCK_' in occurrence else ()
            occurrence = occurrence.replace('PREFILL_', 'CHATGPT2_INFERENCE_').replace('DECODE_', 'CHATGPT2_INFERENCE_')
        return self.layer_norms[self._norm_key(instance_path, occurrence)](value)

    def SOFTMAX(self, value, padding_mask, instance_path, occurrence):
        chatgpt2._log(f'SOFTMAX {tuple(value.shape)}')
        # Each head receives Q @ K.T; apply scaling and causal masking here.
        scores = value / math.sqrt(chatgpt2.model_width // chatgpt2.heads)
        if padding_mask.ndim != 1 or padding_mask.numel() != scores.shape[-1]:
            raise ValueError('SOFTMAX padding mask must match the key sequence')
        valid = torch.ones_like(scores, dtype=torch.bool).tril(
            diagonal=scores.shape[-1] - scores.shape[-2])
        valid = valid & padding_mask.bool().unsqueeze(0)
        # Padded query rows can have no visible keys. Return zero probabilities
        # for those rows instead of allowing softmax(-inf, ...) to produce NaNs.
        nonempty = valid.any(dim=-1, keepdim=True)
        masked = scores.masked_fill(~valid, float('-inf'))
        masked = torch.where(nonempty, masked, torch.zeros_like(masked))
        return F.softmax(masked, dim=-1).masked_fill(~valid, 0)

    def GELU(self, value, instance_path, occurrence):
        chatgpt2._log(f'GELU {tuple(value.shape)}')
        return F.gelu(value, approximate='tanh')

    def SAMPLE(self, value, instance_path, occurrence):
        chatgpt2._log(f'SAMPLE {tuple(value.shape)}')
        # Greedy next-token selection from the final position's logits.
        return value[-1].argmax(dim=-1, keepdim=True)


def main():
    parser = argparse.ArgumentParser(description='GPT-2 demo with random weights')
    parser.add_argument('prompt', nargs='?', default='Hello, world!')
    parser.add_argument('--weights', help='Path to a GPT-2 safetensors checkpoint')
    parser.add_argument('--debug', action='store_true', help='Print model progress and operation sizes')
    args = parser.parse_args()
    chatgpt2.set_debug(args.debug)
    torch.set_num_threads(1)
    chatgpt2._log('GPT-2: ' + ('checkpoint weights' if args.weights else 'random weights'))
    model = Model(checkpoint=args.weights).eval()
    chatgpt2._log(f'TEXT_IN={args.prompt!r}')
    with torch.no_grad():
        device = model.weight_wte.at(()).device
        text_in = torch.tensor(list(args.prompt.encode('utf-8')), dtype=torch.uint8, device=device)
        result = model(text_in)
    text_out = bytes(result.detach().cpu().tolist()).decode('utf-8', errors='replace')
    chatgpt2._log(f'TEXT_OUT={text_out!r}', always=True)


if __name__ == '__main__':
    main()
