"""ONNX export: numerical checks with onnxruntime against independent NumPy
references, weight handling, overwrite protection, and a reduced GPT-2 through
the gpt2-token profile (prefill, two cached decode steps, shared weights)."""
import json
import os
import re

from common import ROOT, arguments, require_modules, setep, source

require_modules('numpy', 'onnx', 'onnxruntime')
import numpy as np
import onnx
import onnxruntime as ort

args = arguments()
interpreter = os.environ.get('SETEP_TEST_INTERPRETER') or __import__('sys').executable


def run(path, success=True, *options):
    code, output = setep(args, path, '--onnx', '--interpreter', interpreter, *options, check=False)
    assert (code == 0) == success, output
    return output


def fixture(name, text, weights=None):
    path = source(args, name, text)
    if weights is not None:
        np.savez(path.with_suffix('.weights.npz'), **weights)
    return path


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def props(path):
    return {p.key: p.value for p in onnx.load(path).metadata_props}


x = np.arange(8, dtype=np.float32).reshape(2, 4) / 4 - 1
path = fixture('arithmetic', 'A=2%4@fp32 TWICE(A){A+A >> B} RUN(A){A >> TWICE() >> C; REPEAT/2{C+C >> C}; C-A >> B}')
run(path)
np.testing.assert_allclose(session(path.with_suffix('.onnx')).run(None, {'A': x})[0], 7 * x)
original = path.with_suffix('.onnx').read_bytes()
assert 'collision' in run(path, False).lower()
assert original == path.with_suffix('.onnx').read_bytes()
run(path, True, '--overwrite')
path.with_suffix('.onnx').write_bytes(original + b'edited')
assert 'modified' in run(path, False, '--overwrite').lower()
path.with_suffix('.onnx').write_bytes(original)
# -o writes elsewhere, with its own receipt.
run(path, True, '-o', args.work / 'elsewhere')
assert (args.work / 'elsewhere' / 'arithmetic.onnx').exists()
assert json.loads((args.work / 'elsewhere' / 'arithmetic.onnx-export.json').read_text())['format'] == 'setep.onnx-export'

# Weight values, matrix orientation, and activation mathematics.
w = np.arange(12, dtype=np.float32).reshape(4, 3) / 9 - .5
path = fixture('mlp', 'A=2%4@fp32 w=4%3@fp32 GELU(A){} RUN(A){A*w >> GELU() >> B}', {'w': w})
run(path)
z = x @ w
expected = .5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + .044715 * z ** 3)))
np.testing.assert_allclose(session(path.with_suffix('.onnx')).run(None, {'A': x})[0], expected, rtol=2e-5, atol=1e-6)
restored = onnx.load(path.with_suffix('.onnx'))
assert np.array_equal(onnx.numpy_helper.to_array(next(t for t in restored.graph.initializer if t.name == 'weight_w')), w)
assert props(path.with_suffix('.onnx'))['setep.weight_status'] == 'supplied'
# Explicit --weights file.
path = fixture('mlp_explicit', 'A=2%4@fp32 w=4%3@fp32 GELU(A){} RUN(A){A*w >> GELU() >> B}')
np.savez(args.work / 'explicit.npz', w=w)
run(path, True, '--weights', args.work / 'explicit.npz')
np.testing.assert_allclose(session(path.with_suffix('.onnx')).run(None, {'A': x})[0], expected, rtol=2e-5, atol=1e-6)
assert 'not found' in run(fixture('mlp_absent', 'A=2%4@fp32 w=4%3@fp32 RUN(A){A*w >> B}'), False, '--weights', args.work / 'absent.npz')

# Last-axis normalization and softmax.
path = fixture('normalization', 'A=2%4@fp32 LN(A){} RUN(A){A >> LN() >> SOFTMAX_LAST() >> B}')
run(path)
centered = x - x.mean(-1, keepdims=True)
normalized = centered / np.sqrt((centered ** 2).mean(-1, keepdims=True) + 1e-5)
expected = np.exp(normalized - normalized.max(-1, keepdims=True))
expected /= expected.sum(-1, keepdims=True)
np.testing.assert_allclose(session(path.with_suffix('.onnx')).run(None, {'A': x})[0], expected, rtol=1e-5, atol=1e-6)
path = fixture('arrays', 'A=2%4@fp32 RUN(A){A >[2]> P; REPEAT/2[]{P[]+P[]} >[]> B}')
run(path)
np.testing.assert_allclose(session(path.with_suffix('.onnx')).run(None, {'A': x})[0], 2 * x)
# Attention scores from two projections and a transpose.
q = np.eye(4, dtype=np.float32); k = q * .5; v = q * 2
path = fixture('qkv', 'A=2%4@fp32 q=4%4@fp32 k=4%4@fp32 v=4%4@fp32 RUN(A){A*q >> Q; A*k >> K; A*v >> V; Q*K^ >> SOFTMAX_LAST() >> P; P*V >> B}',
               {'q': q, 'k': k, 'v': v})
run(path)
scores = x @ q @ (x @ k).T
p = np.exp(scores - scores.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
np.testing.assert_allclose(session(path.with_suffix('.onnx')).run(None, {'A': x})[0], p @ (x @ v), rtol=1e-5, atol=1e-6)

# Missing weights need --zero-weights; wrong dtypes are refused.
path = fixture('missing', 'A=2%4@fp32 w=4%3@fp32 RUN(A){A*w >> B}')
assert 'Missing weight' in run(path, False)
assert not path.with_suffix('.onnx').exists()
run(path, True, '--zero-weights')
assert props(path.with_suffix('.onnx'))['setep.weight_status'] == 'placeholder'
assert np.count_nonzero(session(path.with_suffix('.onnx')).run(None, {'A': x})[0]) == 0
path = fixture('bad_weights', 'A=2%4@fp32 w=4%3@fp32 RUN(A){A*w >> B}', {'w': w.astype(np.float64)})
assert 'dtype mismatch' in run(path, False)
# Array weights get one bank per element.
path = fixture('indexed', 'A=2%4@fp32 w=4%4@fp32 RUN(A){REPEAT/2{A*w[] >> A}; A >> B}')
run(path, True, '--zero-weights')
banks = [t.name for t in onnx.load(path.with_suffix('.onnx')).graph.initializer if t.name.startswith('weight_')]
assert len(banks) == 2 and len(set(banks)) == 2, banks
np.savez(path.with_suffix('.weights.npz'), **{name: np.eye(4, dtype=np.float32) * (i + 1) for i, name in enumerate(banks)})
run(path, True, '--overwrite')
np.testing.assert_allclose(session(path.with_suffix('.onnx')).run(None, {'A': x})[0], 2 * x)
path = fixture('unsupported', 'A=2%4@fp32 TOKENIZE(A){} RUN(A){A >> TOKENIZE() >> B}')
assert 'mathematical' in run(path, False)
assert not path.with_suffix('.onnx').exists()

# Persistent state, and reading state before it is written.
path = fixture('state', 'A=1%2@fp32 RUN(A){A+A >> X; X >> K!; K!^:X^ >:> NEXT; NEXT^ >> K!; K! >> B}')
run(path)
for value in session(path.with_suffix('.onnx')).run(None, {'A': np.array([[1, 2]], np.float32)}):
    np.testing.assert_array_equal(value, [[2, 4], [2, 4]])
path = fixture('uninitialized', 'A=1%2@fp32 RUN(A){K! >> B}')
assert 'Uninitialized persistent state' in run(path, False)
assert not path.with_suffix('.onnx').exists()

# Reduced GPT-2 through the gpt2-token profile against a NumPy reference.
text = (ROOT / 'models/chatgpt2/chatgpt2.setep').read_text()
for key, value in dict(tokens=3, decode_tokens=2, layers=2, heads=2, model_width=8, mlp_width=16, vocab_size=17, positions=16).items():
    text = re.sub(r'(?m)^' + key + r' = \d+', f'{key} = {value}', text)
path = fixture('native_gpt2', text)
assert 'operation' in run(path, False).lower()  # No implicit model-specific semantics.
run(path, True, '--onnx-profile', 'gpt2-token', '--zero-weights')
keys = json.loads(props(path.with_suffix('.onnx'))['setep.placeholder_weights'])
placeholder = onnx.load(path.with_suffix('.onnx'))
rng = np.random.default_rng(42)
weights = {t.name: rng.normal(0, .15, t.dims).astype(np.float32) for t in placeholder.graph.initializer if t.name in keys}
for key in weights:
    if ('.ln_' in key or key.startswith('ln_f.')) and key.endswith('.weight'):
        weights[key] += 1
np.savez(path.with_suffix('.weights.npz'), **weights)
run(path, True, '--onnx-profile', 'gpt2-token', '--overwrite')
model = onnx.load(path.with_suffix('.onnx'))
assert props(path.with_suffix('.onnx'))['setep.weight_status'] == 'supplied'
# Parameter ownership is shared between prefill/decode and embedding/head.
assert len([t for t in model.graph.initializer if t.name == 'weight_wte']) == 1
assert set(keys) == set(weights)
runtime = session(path.with_suffix('.onnx'))


def reference(ids, mask):
    cache = {}; logits = []; generated = []
    positions = np.maximum(np.cumsum(mask) - 1, 0); next_position = int(mask.sum())

    def norm(x, key):
        centered = x - x.mean(-1, keepdims=True)
        return centered / np.sqrt((centered ** 2).mean(-1, keepdims=True) + 1e-5) * weights[key + '.weight'] + weights[key + '.bias']

    def bank(name, layer):
        return weights[f'weight_{name}/{layer}']
    for step in range(3):
        x = weights['weight_wte'][ids] + weights['weight_wpe'][positions]
        for layer in range(2):
            a = norm(x, f'h.{layer}.ln_1')
            q, k, v = [a @ bank(n, layer) + bank(n + 'b', layer) for n in ('q', 'k', 'v')]
            if step:
                k = np.concatenate([cache[layer][0], k]); v = np.concatenate([cache[layer][1], v])
            cache[layer] = (k, v)
            qh = q.reshape(len(q), 2, 4).transpose(1, 0, 2)
            kh = k.reshape(len(k), 2, 4).transpose(1, 0, 2)
            vh = v.reshape(len(v), 2, 4).transpose(1, 0, 2)
            scores = qh @ kh.transpose(0, 2, 1) / 2
            valid = np.tri(len(q), len(k), len(k) - len(q), dtype=bool) & mask.astype(bool)
            scores = np.where(valid, scores, -np.inf)
            scores = np.where(valid.any(-1, keepdims=True), scores, 0)
            p = np.exp(scores - scores.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True); p = np.where(valid, p, 0)
            attention = (p @ vh).transpose(1, 0, 2).reshape(len(q), 8)
            x = x + attention @ bank('o', layer) + bank('ob', layer)
            z = norm(x, f'h.{layer}.ln_2') @ bank('up', layer) + bank('up_bias', layer)
            z = .5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + .044715 * z ** 3)))
            x = x + z @ bank('down', layer) + bank('down_bias', layer)
        out = norm(x, 'ln_f') @ weights['weight_wte'].T
        logits.append(out); token = int(out[-1].argmax()); generated.append(token)
        ids = np.array([token]); positions = np.array([next_position]); next_position += 1
        mask = np.append(mask, 1)
    return generated, logits, cache


for mask in (np.array([1, 1, 1], np.int32), np.array([0, 1, 1], np.int32), np.array([0, 0, 1], np.int32)):
    ids = np.array([1, 2, 3], np.int32)
    expected, logits, cache = reference(ids, mask)
    values = dict(zip([o.name for o in runtime.get_outputs()], runtime.run(None, {'input_ids': ids, 'attention_mask': mask})))

    def result(prefix):
        return next(v for n, v in values.items() if n.startswith(prefix))
    np.testing.assert_array_equal(result('output_tokens_'), expected)
    for step in range(3):
        np.testing.assert_allclose(result(f'output_logits_{step}_'), logits[step], rtol=3e-5, atol=2e-6)
    for layer in range(2):
        for i, kind in enumerate(('K', 'V')):
            np.testing.assert_allclose(result(f'state_state_BLOCK_{kind}_{layer}_'), cache[layer][i], rtol=3e-5, atol=2e-6)
print('ONNX numerical export checks passed (including the reduced GPT-2 profile)')
