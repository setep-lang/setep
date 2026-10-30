"""Write the HF `gpt2` transformer weights as a GPT-2-format safetensors file.

Usage: python make_checkpoint.py OUT.safetensors
Keys follow the original GPT-2 naming (wte.weight, h.N.attn.c_attn.weight, ...),
which is what models/chatgpt2/chatgpt2_impl.py expects.
"""
import sys
import torch
from safetensors.torch import save_file
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2', torch_dtype=torch.float32)
state = {k: v.detach().contiguous().clone()
         for k, v in model.transformer.state_dict().items()}
save_file(state, sys.argv[1], metadata={'source': 'huggingface gpt2',
                                        'revision': str(getattr(model.config, '_commit_hash', None))})
print(len(state), 'tensors;', 'commit', getattr(model.config, '_commit_hash', None))
print(sorted(state)[:6])
