"""Full-size GPT-2 ONNX export from models/chatgpt2/chatgpt2.setep with HF weights,
run in onnxruntime and compared with the generated PyTorch model and HF.

Usage: python onnx_gpt2.py SETEP_ROOT WORK_DIR GEN_DIR CHECKPOINT OUT_JSON
  GEN_DIR    holds chatgpt2.py + setep.py (setep --python) and chatgpt2_impl.py
  CHECKPOINT GPT-2 safetensors from make_checkpoint.py
First exports with --zero-weights to learn the weight names, writes
WORK_DIR/chatgpt2.weights.npz from the checkpoint, then exports with it.
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import onnx, onnxruntime as ort

root, work, gen_dir, ckpt, out_json = map(Path, sys.argv[1:6])
work.mkdir(parents=True, exist_ok=True)
setep = work / 'chatgpt2.setep'
shutil.copy(root / 'models/chatgpt2/chatgpt2.setep', setep)
tool = root / 'build' / ('setep.exe' if os.name == 'nt' else 'setep')


def run(*options):
    command = [str(tool), str(setep), '--onnx', '--onnx-profile', 'gpt2-token', '--overwrite',
               '--interpreter', sys.executable, *options]
    t0 = time.perf_counter()
    call = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    dt = time.perf_counter() - t0
    assert call.returncode == 0, call.stdout + call.stderr
    return dt


t_placeholder = run('--zero-weights')
placeholder = onnx.load(work / 'chatgpt2.onnx', load_external_data=False)
props = {p.key: p.value for p in placeholder.metadata_props}
keys = json.loads(props['setep.placeholder_weights'])
dims = {t.name: tuple(t.dims) for t in placeholder.graph.initializer}
del placeholder

from safetensors.torch import load_file
hf = {k: v.numpy() for k, v in load_file(str(ckpt)).items()}
weights = {}
for key in keys:
    if key in ('weight_wte', 'weight_wte_shared'):
        value = hf['wte.weight']
    elif key in ('weight_wpe', 'weight_wpe_shared'):
        value = hf['wpe.weight']
    elif key.startswith('weight_'):
        name, layer = key[len('weight_'):].split('/')
        p = f'h.{layer}.'
        qkv_w = np.split(hf[p + 'attn.c_attn.weight'], 3, axis=1)
        qkv_b = np.split(hf[p + 'attn.c_attn.bias'], 3)
        value = {'q': qkv_w[0], 'k': qkv_w[1], 'v': qkv_w[2],
                 'qb': qkv_b[0], 'kb': qkv_b[1], 'vb': qkv_b[2],
                 'o': hf[p + 'attn.c_proj.weight'], 'ob': hf[p + 'attn.c_proj.bias'],
                 'up': hf[p + 'mlp.c_fc.weight'], 'up_bias': hf[p + 'mlp.c_fc.bias'],
                 'down': hf[p + 'mlp.c_proj.weight'], 'down_bias': hf[p + 'mlp.c_proj.bias']}[name]
    else:
        value = hf[key]  # h.N.ln_1.weight / ln_f.bias etc.
    value = np.ascontiguousarray(value, dtype=np.float32).reshape(dims[key])
    weights[key] = value
np.savez(work / 'chatgpt2.weights.npz', **weights)
t_export = run('--weights', str(work / 'chatgpt2.weights.npz'))
model_bytes = (work / 'chatgpt2.onnx').stat().st_size
m = onnx.load(work / 'chatgpt2.onnx', load_external_data=False)
weight_status = {p.key: p.value for p in m.metadata_props}.get('setep.weight_status')
n_nodes = len(m.graph.node)
inputs = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
del m

options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.log_severity_level = 3
sess = ort.InferenceSession(str(work / 'chatgpt2.onnx'), options, providers=['CPUExecutionProvider'])
out_names = [o.name for o in sess.get_outputs()]

# Generated PyTorch model (same as compare_hf.py) for the same prompts.
sys.path.insert(0, str(gen_dir))
import chatgpt2, chatgpt2_impl  # noqa: E402
from transformers import GPT2LMHeadModel  # noqa: E402
torch.set_num_threads(4)


class Recording(chatgpt2_impl.Model):
    def SAMPLE(self, value, instance_path, occurrence):
        self.logits.append(value[-1].detach().clone())
        return super().SAMPLE(value, instance_path, occurrence)


pt = Recording(checkpoint=str(ckpt)).eval()
ref = GPT2LMHeadModel.from_pretrained('gpt2', torch_dtype=torch.float32, attn_implementation='eager').eval()


def err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = np.abs(a - b)
    return float(d.max()), float((d / np.maximum(np.abs(b), 1e-30)).max())


results = []
for prompt in ['Paris is the capital of France.', 'The quick brown fox jumps over the lazy dog because']:
    pt.logits = []
    with torch.no_grad():
        pt(torch.tensor(list(prompt.encode()), dtype=torch.uint8))
    ids = pt.tokenizer.encode_ordinary(prompt)
    pad = chatgpt2.tokens - len(ids)
    onnx_ids = np.array([pt.tokenizer.eot_token] * pad + ids, np.int32)
    mask = np.array([0] * pad + [1] * len(ids), np.int32)
    t0 = time.perf_counter()
    values = dict(zip(out_names, sess.run(None, {'input_ids': onnx_ids, 'attention_mask': mask})))
    t_ort = time.perf_counter() - t0
    tokens = next(v for n, v in values.items() if n.startswith('output_tokens_')).tolist()
    # HF reference on unpadded ids, cached decode fed the ONNX tokens.
    hf_logits = []
    with torch.no_grad():
        o = ref(torch.tensor([ids]), use_cache=True); past = o.past_key_values
        hf_logits.append(o.logits[0, -1].numpy())
        for t in tokens[:-1]:
            o = ref(torch.tensor([[t]]), past_key_values=past, use_cache=True); past = o.past_key_values
            hf_logits.append(o.logits[0, -1].numpy())
    steps = []
    for s in range(len(tokens)):
        lg = next(v for n, v in values.items() if n.startswith(f'output_logits_{s}_'))[-1]
        a_hf, r_hf = err(lg, hf_logits[s])
        a_pt, r_pt = err(lg, pt.logits[s].numpy())
        steps.append(dict(step=s, onnx_token=tokens[s], pt_token=int(pt.logits[s].argmax()),
                          hf_token=int(hf_logits[s].argmax()),
                          onnx_vs_hf_max_abs=a_hf, onnx_vs_hf_max_rel=r_hf,
                          onnx_vs_pt_max_abs=a_pt, onnx_vs_pt_max_rel=r_pt))
    results.append(dict(prompt=prompt, pad=pad, ort_seconds=t_ort, steps=steps))

meta = dict(onnx=onnx.__version__, onnxruntime=ort.__version__, torch=torch.__version__,
            export_seconds_placeholder=t_placeholder, export_seconds_weights=t_export,
            onnx_bytes=model_bytes, onnx_nodes=n_nodes, inputs=inputs,
            n_weight_tensors=len(weights), weight_status=weight_status,
            n_outputs=len(out_names))
json.dump(dict(meta=meta, results=results), open(out_json, 'w'), indent=1)
print(json.dumps(meta))
for r in results:
    print(f"\n{r['prompt']!r} pad={r['pad']} ort={r['ort_seconds']:.2f}s")
    for s in r['steps']:
        print(f"  step {s['step']:2d} tok onnx/pt/hf {s['onnx_token']}/{s['pt_token']}/{s['hf_token']}  "
              f"vsHF abs {s['onnx_vs_hf_max_abs']:.3e} rel {s['onnx_vs_hf_max_rel']:.3e}  "
              f"vsPT abs {s['onnx_vs_pt_max_abs']:.3e} rel {s['onnx_vs_pt_max_rel']:.3e}")
