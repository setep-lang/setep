"""Probe which loss convention reproduces the SETEP training loss 3.074863."""
import sys
import torch, tiktoken
from transformers import GPT2LMHeadModel
tok = tiktoken.get_encoding('gpt2'); ids = tok.encode_ordinary('Paris is the capital of France.')
m = GPT2LMHeadModel.from_pretrained('gpt2', torch_dtype=torch.float32, attn_implementation='eager').eval()
F = torch.nn.functional.cross_entropy
with torch.no_grad():
    lg = m(torch.tensor([ids])).logits[0]
    pad = ids + [tok.eot_token] * (10 - len(ids)); lg10 = m(torch.tensor([pad])).logits[0]
print('ids', ids)
print('7 targets incl eot', F(lg, torch.tensor(ids[1:] + [tok.eot_token])).item())
print('6 targets', F(lg[:-1], torch.tensor(ids[1:])).item())
t10 = torch.tensor(ids[1:] + [tok.eot_token] + [-100] * (10 - len(ids)))
print('padded10 ignore', F(lg10, t10).item())
print('padded10 all eot targets', F(lg10, torch.tensor(ids[1:] + [tok.eot_token] * (11 - len(ids)))).item())
print('sum/10', F(lg, torch.tensor(ids[1:] + [tok.eot_token]), reduction='sum').item() / 10)
