"""GPT-2 training check: 2 SGD steps (lr 1e-4) on one prompt, SETEP-generated
explicit backward/update vs HF GPT2LMHeadModel + torch.optim.SGD (fp32).

Usage: python train_vs_hf.py CHECKPOINT TRAINED_BY_SETEP.safetensors
The SETEP losses are taken from train.log (printed by chatgpt2_training_impl.py).
Inputs = prompt ids, targets = ids[1:] + [eot] (as in TRAIN_TOKENIZE); padded
positions are ignored by both.
"""
import sys, torch, tiktoken
from safetensors.torch import load_file
from transformers import GPT2LMHeadModel

ckpt, trained = sys.argv[1:3]
torch.set_num_threads(4)
tok = tiktoken.get_encoding('gpt2')
ids = tok.encode_ordinary('Paris is the capital of France.')
targets = ids[1:] + [tok.eot_token]
model = GPT2LMHeadModel.from_pretrained('gpt2', torch_dtype=torch.float32, attn_implementation='eager').eval()  # eval: no dropout; grads still flow
opt = torch.optim.SGD(model.parameters(), lr=1e-4)
x = torch.tensor([ids]); y = torch.tensor(targets)
for step in range(2):
    opt.zero_grad()
    logits = model(x).logits[0]
    loss = torch.nn.functional.cross_entropy(logits, y)
    print(f'HF step {step+1} loss {loss.item():.6f}')
    loss.backward()
    opt.step()
hf = {k: v.detach() for k, v in model.transformer.state_dict().items()}
setep = load_file(trained)
orig = load_file(ckpt)
worst = (0, None); worst_rel = (0, None); moved = []
for k, v in setep.items():
    ref = hf[k].reshape(v.shape)
    d = (v.double() - ref.double()).abs().max().item()
    upd_ref = (ref.double() - orig[k].double()).abs().max().item()
    upd_err = ((v.double() - orig[k].double()) - (ref.double() - orig[k].double())).abs().max().item()
    moved.append((k, upd_ref, upd_err))
    if d > worst[0]:
        worst = (d, k)
    if upd_ref > 0 and upd_err / upd_ref > worst_rel[0]:
        worst_rel = (upd_err / upd_ref, k)
print('tensors saved by SETEP', len(setep), 'HF transformer tensors', len(hf),
      'missing in SETEP file', sorted(set(hf) - set(setep))[:5])
print('max |W_setep - W_hf| after 2 steps', worst)
print('max (update error / max update) per tensor', worst_rel)
print('max update magnitude', max(m[1] for m in moved))
