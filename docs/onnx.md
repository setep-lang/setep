# Exporting SETEP to ONNX

```text
setep model.setep --onnx [--weights model.weights.npz | --zero-weights] [--onnx-profile gpt2-token] [--interpreter python]
```

This writes `<stem>.onnx` and `<stem>.onnx-export.json` into the output
directory (`-o`, default: beside the `.setep`). The receipt records ownership,
model/source hashes and placeholder status. The executable graph comes from
SETEP's semantic IR; the tool serializes it and runs `python/setep_onnx.py`,
which builds the graph with the `onnx` package, checks it with the ONNX checker
and strict shape inference, and writes the file. Install
`python/requirements-onnx.txt` into the interpreter given by `--interpreter`.
ONNX Runtime is only needed for the numerical tests, not for export.

An existing export is replaced only with `--overwrite`, and only when its
receipt shows the file is an unmodified export of the same source.

## Supported mathematics

Native support includes concrete typed input declarations, matrix multiplication,
transpose, addition/subtraction, elementwise multiplication/division, static
composition/repetition, split/concat and supported array flows. Last-axis reduction
and normalization builtins have numerical implementations. Recognized abstract
operations include `LN`, `LAYERNORM`, `RMSNORM`, `SOFTMAX`, `RELU`, `SIGMOID`,
`TANH`, `SILU` and `GELU`, with strict input cardinality checks. LN/RMSNorm are
**non-affine**, last-axis, with epsilon `1e-5`; GELU uses the **tanh approximation**.
ONNX metadata records these conventions. Express affine scale/bias separately.

For example, this MLP exports directly:

```text
A = 2%4@fp32
w = 4%3@fp32
GELU(A) {}
RUN(A) {A*w >> GELU() >> B}
```

## Weights

Supply values in `model.weights.npz` beside the model (or name the file with
`--weights`):

```python
import numpy as np
np.savez('model.weights.npz', w=np.asarray(your_weights, dtype=np.float32))
```

NPZ keys use semantic weight-bank identities such as `weight_w`, with static
instance suffixes (`weight_w/0`, etc.) for array weights. Root weights also accept
the declaration name (`w`). Local declarations use qualified names; missing-weight
errors report the required identity. Shapes/dtypes must match exactly. Repeated
use of one bank/path shares one initializer. NPZ loading disables pickle.

Missing values fail unless `--zero-weights` explicitly permits zero placeholders.
Model metadata (`setep.weight_status`, `setep.placeholder_weights`) and the
receipt list every placeholder; a placeholder model does not reproduce a trained
model. A simple way to obtain the full list of weight identities is a first
export with `--zero-weights`.

## State, limits and errors

Persistent tensors (`K!`) initialized within the entry computation export as static
SSA dataflow: writes replace the current value, reads reuse it, and append grows
axis zero. Bank and instance paths keep separate caches isolated. Final persistent
values are additional graph outputs. Reading an uninitialized cache fails; exporting
a standalone decoder with externally supplied initial state is unsupported.

Unknown abstract functions (including tokenizer/sampling hooks), dynamic choices,
weight mutation and unsupported structured entry inputs fail explicitly. Cost
annotations do not supply mathematics. Native exports use concrete declared shapes
and expand static repetition; they are not dynamic-batch or training graphs. The
writer embeds tensor data and rejects models at or above 2 GiB.

## GPT-2 token-level profile

`--onnx-profile gpt2-token` exports `models/chatgpt2/chatgpt2.setep` with the
semantics of `chatgpt2_impl.py`: embedding lookup, affine layer normalization
(epsilon `1e-5`), head-width scaling, causal and padding masks (all-masked rows
produce zero), tanh GELU, and greedy sampling from the final position. It uses
the source's `PREFILL` and `DECODE` computations and their cache updates, rather
than a separate hardcoded transformer graph. The text entry wrapper is replaced
by a token API.

Inputs are rank-one `int32` `input_ids` and binary `attention_mask`, of length
the SETEP scalar `tokens`. Use left padding and at least one real token. Position
IDs are derived from the mask. Batch size is one. Tokenization and detokenization
remain external. The graph expands prefill and `decode_tokens` decode calls.
Outputs include `output_tokens_*` (the `decode_tokens + 1` new token IDs),
`output_logits_N_*` for each iteration, and final `state_*` K/V tensors. Re-export
to change these fixed dimensions/counts.

Supply projections as `weight_q/N`, `weight_k/N`, `weight_v/N`, `weight_o/N`,
`weight_up/N`, `weight_down/N` and the corresponding `weight_qb/N`, `weight_kb/N`,
`weight_vb/N`, `weight_ob/N`, `weight_up_bias/N`, `weight_down_bias/N` NPZ entries.
Shared embeddings use `weight_wte` and `weight_wpe`. Affine norm parameters use
`h.N.ln_1.weight`, `h.N.ln_1.bias`, `h.N.ln_2.weight`, `h.N.ln_2.bias`,
`ln_f.weight`, and `ln_f.bias`. Projection biases have shape `[1,width]`; norm
parameters have shape `[width]`. All are `float32`. Prefill/decode and the
embedding/head reuse the same weight identities. `eval/gpt2/onnx_gpt2.py` builds
this NPZ from the Hugging Face `gpt2` checkpoint and compares the exported model
with the generated PyTorch model and Hugging Face.

Metadata `setep.schedule` records the prefill/decode phase of every node, the
prompt length and the sampled tokens.
