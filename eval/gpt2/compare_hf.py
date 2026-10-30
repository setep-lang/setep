"""Result set 2: generated SETEP PyTorch GPT-2 vs Hugging Face GPT2LMHeadModel (fp32).

Usage: python compare_hf.py GEN_DIR CHECKPOINT OUT_JSON
GEN_DIR holds chatgpt2.py + setep.py (from setep --python) and a copy of
models/chatgpt2/chatgpt2_impl.py.  The SETEP model is run through its own
forward() (TOKENIZE -> PREFILL -> 10x DECODE -> DETOKENIZE); SAMPLE and LN
are wrapped only to record their inputs.  HF runs on the unpadded token IDs,
eager attention, with past_key_values for decode, fed the same tokens.
"""
import json, platform, sys
import torch

gen_dir, ckpt, out_json = sys.argv[1:4]
sys.path.insert(0, gen_dir)
import chatgpt2  # noqa: E402
import chatgpt2_impl  # noqa: E402
import transformers  # noqa: E402
from transformers import GPT2LMHeadModel  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)


class Recording(chatgpt2_impl.Model):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.logits, self.ln_inputs = [], []

    def SAMPLE(self, value, instance_path, occurrence):
        self.logits.append(value[-1].detach().clone())
        return super().SAMPLE(value, instance_path, occurrence)

    def LN(self, value, instance_path, occurrence):
        self.ln_inputs.append((occurrence, tuple(instance_path), value.detach().clone()))
        return super().LN(value, instance_path, occurrence)


def err(a, b):
    a, b = a.double(), b.double()
    d = (a - b).abs()
    return {'max_abs': d.max().item(),
            'max_rel': (d / b.abs().clamp_min(1e-30)).max().item(),
            'max_rel_floor1e-3': (d / b.abs().clamp_min(1e-3)).max().item(),
            'max_abs_over_max_ref': (d.max() / b.abs().max()).item(),
            'ref_max_abs': b.abs().max().item()}


def hf_ln_inputs(hf):
    store = []
    hooks = []
    for i, block in enumerate(hf.transformer.h):
        hooks.append(block.ln_1.register_forward_pre_hook(
            lambda m, args, i=i: store.append(('ln_1', i, args[0][0].detach().clone()))))
        hooks.append(block.ln_2.register_forward_pre_hook(
            lambda m, args, i=i: store.append(('ln_2', i, args[0][0].detach().clone()))))
    hooks.append(hf.transformer.ln_f.register_forward_pre_hook(
        lambda m, args: store.append(('ln_f', -1, args[0][0].detach().clone()))))
    return store, hooks


def run(prompt, hf, setep_model=None):
    model = setep_model or Recording(checkpoint=ckpt).eval()
    model.logits.clear(); model.ln_inputs.clear()
    with torch.no_grad():
        text_in = torch.tensor(list(prompt.encode('utf-8')), dtype=torch.uint8)
        text_out = model(text_in)
    out_text = bytes(text_out.tolist()).decode('utf-8', errors='replace')
    ids = model.tokenizer.encode_ordinary(prompt) or [model.tokenizer.eot_token]
    pad = chatgpt2.tokens - len(ids)
    setep_tokens = [int(l.argmax()) for l in model.logits]

    # HF reference: prefill on unpadded ids, then cached decode fed SETEP's tokens.
    store, hooks = hf_ln_inputs(hf)
    hf_logits, hf_greedy = [], []
    with torch.no_grad():
        o = hf(torch.tensor([ids]), use_cache=True)
        past = o.past_key_values
        hf_logits.append(o.logits[0, -1]); hf_greedy.append(int(o.logits[0, -1].argmax()))
        hf_prefill_store = list(store); store.clear()
        for t in setep_tokens[:-1]:
            o = hf(torch.tensor([[t]]), past_key_values=past, use_cache=True)
            past = o.past_key_values
            hf_logits.append(o.logits[0, -1]); hf_greedy.append(int(o.logits[0, -1].argmax()))
    for h in hooks:
        h.remove()

    steps = []
    for s, (a, b) in enumerate(zip(model.logits, hf_logits)):
        e = err(a, b)
        e.update(step='prefill' if s == 0 else f'decode{s}',
                 setep_token=setep_tokens[s], hf_token=hf_greedy[s],
                 greedy_match=setep_tokens[s] == hf_greedy[s],
                 top5_match=torch.topk(a, 5).indices.tolist() == torch.topk(b, 5).indices.tolist())
        steps.append(e)

    # Intermediate activations of the prefill: LN inputs = residual stream.
    prefill_ln = [x for x in model.ln_inputs if x[0].startswith('PREFILL_')]
    acts = []
    hf_map = {(n, i): v for n, i, v in hf_prefill_store}
    for occ, path, value in prefill_ln:
        if occ == 'PREFILL_BLOCK_LN_0':
            key = ('ln_1', path[-1])
        elif occ == 'PREFILL_BLOCK_LN_1':
            key = ('ln_2', path[-1])
        else:
            key = ('ln_f', -1)
        e = err(value[pad:], hf_map[key])
        e['activation'] = f'{key[0]} input' + (f' block {key[1]}' if key[1] >= 0 else '')
        acts.append(e)
    return model, dict(prompt=prompt, n_prompt_tokens=len(ids), left_padding=pad,
                       prompt_ids=ids, setep_text_out=out_text,
                       setep_generated_ids=setep_tokens, hf_greedy_ids=hf_greedy,
                       steps=steps, prefill_activations=acts)


hf = GPT2LMHeadModel.from_pretrained('gpt2', torch_dtype=torch.float32,
                                     attn_implementation='eager').eval()
prompts = ['Paris is the capital of France.',
           'The quick brown fox jumps over the lazy dog because']
results = []
shared = None
for p in prompts:
    shared, r = run(p, hf, shared)  # same SETEP object reused: second run tests cache reset
    results.append(r)
# Independence: second prompt on a fresh model must equal the reused-model run.
_, fresh = run(prompts[1], hf)
same = all(a['setep_token'] == b['setep_token'] and abs(a['max_abs'] - b['max_abs']) == 0
           for a, b in zip(results[1]['steps'], fresh['steps']))

meta = dict(python=platform.python_version(), torch=torch.__version__,
            transformers=transformers.__version__, threads=torch.get_num_threads(),
            seed=0, dtype='float32', hf_attn='eager', checkpoint=ckpt,
            reused_model_matches_fresh_model_bitwise_err=same)
json.dump(dict(meta=meta, results=results), open(out_json, 'w'), indent=1)

print(json.dumps(meta))
for r in results:
    print(f"\nPROMPT {r['prompt']!r}: {r['n_prompt_tokens']} tokens, left pad {r['left_padding']}")
    print('  SETEP text:', repr(r['setep_text_out']))
    print(f"  {'step':8} {'max_abs':>10} {'max_rel':>10} {'rel(fl1e-3)':>11} {'abs/max|ref|':>12} {'|ref|max':>9} tok_setep tok_hf match top5")
    for s in r['steps']:
        print(f"  {s['step']:8} {s['max_abs']:10.3e} {s['max_rel']:10.3e} {s['max_rel_floor1e-3']:11.3e} {s['max_abs_over_max_ref']:12.3e} {s['ref_max_abs']:9.2f} {s['setep_token']:9d} {s['hf_token']:6d} {s['greedy_match']} {s['top5_match']}")
    for a in r['prefill_activations']:
        if a['activation'].startswith(('ln_1 input block 0', 'ln_1 input block 6', 'ln_2 input block 11', 'ln_f')):
            print(f"  act {a['activation']:22} max_abs {a['max_abs']:.3e} abs/max|ref| {a['max_abs_over_max_ref']:.3e} |ref|max {a['ref_max_abs']:.1f}")
    print('  max over all 25 prefill LN inputs: abs', max(a['max_abs'] for a in r['prefill_activations']))
