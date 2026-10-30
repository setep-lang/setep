# SETEP

SETEP is a compact language for describing machine-learning models as
**s**copes, **e**xpressions, **t**ransfers, **e**xternal computations and
**p**ersistent state: tensors with typed, optionally partitioned shapes; matrix
arithmetic; repeated and alternative scopes; weights and caches; and opaque
computations whose implementation is supplied elsewhere. One description is
compiled to an executable PyTorch model, to distributed Chakra execution traces
for [ASTRA-sim](https://astra-sim.github.io/), and to ONNX.

```text
A = 2%4@fp32
w = 4%3@fp32
GELU(A) {}
RUN(A) {A*w >> GELU() >> B}
```

The language is specified in [docs/SETEP.txt](docs/SETEP.txt); the backends are
described in [docs/codegen.md](docs/codegen.md) and [docs/onnx.md](docs/onnx.md).

## Build

Requires CMake 3.16+ and a C++17 compiler; there are no other C++ dependencies.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

On Windows with MSYS2 UCRT64 (the toolchain this was developed with):

```sh
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
```

The result is `build/setep` (`build/setep.exe`).

## Usage

```text
setep MODEL.setep [-o DIR] [--python [--partitioned]]
                  [--chakra [--nodes N] [--processors-per-node N] [--hosts-per-node N]
                            [--memory-mb MB] [--host-memory-mb MB] [--replicas N] [--sequences N]
                            [--gflop-per-cycle X] [--mb-per-cycle X] [--clock-hz X]]
                  [--onnx [--weights FILE.npz | --zero-weights] [--onnx-profile gpt2-token]
                          [--overwrite] [--interpreter PYTHON]]
```

With no backend option the model is parsed and checked only. Outputs go to `DIR`
(default: beside the model). `setep --help` lists every option with its default.
Errors are reported as `MODEL: line N: message` and exit with status 1; usage
errors exit with 2.

### PyTorch

```sh
build/setep models/chatgpt2/chatgpt2_simple.setep --python -o out
python out/chatgpt2_simple.py                         # placeholder demo, random weights
build/setep models/chatgpt2/chatgpt2.setep --python -o out
cp models/chatgpt2/chatgpt2_impl.py out/
python out/chatgpt2_impl.py "Hello, world!"           # handwritten hooks (needs tiktoken)
```

This writes `out/chatgpt2.py` (a `torch.nn.Module` with one method per
computation and an overridable method per external computation) and
`out/setep.py`, the shared runtime. The `*_impl.py` files next to the models are
handwritten subclasses that implement the external computations.

### Chakra traces for ASTRA-sim

```sh
build/setep models/chatgpt2/chatgpt2.setep --chakra -o out/gpt2 --memory-mb 500
python tools/chakra_trace.py check out/gpt2/chatgpt2
```

The model is expanded into its operation graph and placed with memory-aware
next-fit packing on an inventory of `--nodes` x `--processors-per-node`
accelerators with `--memory-mb` each (default 64 x 64, 80000 MB). This writes one
`chatgpt2.<rank>.et` per occupied accelerator plus `chatgpt2.chakra.json`;
cross-processor dependencies become matching send/receive pairs. Give ASTRA-sim
`--workload-configuration=out/gpt2/chatgpt2` and a network with `num_ranks` NPUs
(see `eval/astra/one.sh`). At 500 MB per accelerator GPT-2 occupies 12 ranks
with 7468 compute nodes and 285 transfers.

### ONNX

```sh
python -m pip install -r python/requirements-onnx.txt
build/setep models/chatgpt2/chatgpt2.setep --onnx --onnx-profile gpt2-token --zero-weights -o out/onnx
```

The tool serializes the semantic program and runs `python/setep_onnx.py` with
`--interpreter` (default `python`), which builds, checks and writes
`out/onnx/chatgpt2.onnx` and an ownership receipt. Weights come from
`MODEL.weights.npz` beside the model or `--weights FILE.npz`; `--zero-weights`
fills missing ones with zeros and records them in the model metadata.
`eval/gpt2/onnx_gpt2.py` exports GPT-2 with the Hugging Face weights. The Python
files are found next to the source tree; set `SETEP_PYTHON_DIR` if the
executable is moved.

## Models

`models/` holds the SETEP sources and their supporting files:

| model | files |
|---|---|
| `chatgpt2` | GPT-2 inference (`chatgpt2.setep`, with prefill/decode cache), a simplified variant, and training (explicit backward and SGD); handwritten `*_impl.py` hooks; ASTRA-sim system/network configs |
| `switch_base_8` | Switch Transformer (8 experts) inference and training with `*_impl.py`; the hooks load the Hugging Face `google/switch-base-8` checkpoint and tokenizer from `models/switch_base_8/switch-base-8/` (download separately) |
| `deepseek_r1`, `qwen3_5`, `llama`, `mistral`, `kimi2`, `gpt_oss` | published architectures as text-inference workload models |

Each `README.md` / `model-info.json` records the sources and sizing choices.

## Tests

```sh
ctest --test-dir build --output-on-failure
```

| test | needs | checks |
|---|---|---|
| `graph` | - | operation costs, shapes, weight instances, storage and placement (C++) |
| `errors` | Python | parse and semantic errors, including `!` on a weight, and CLI errors |
| `python_models` | Python (PyTorch optional) | every model generates and compiles; with PyTorch each imports |
| `python_numerical` | PyTorch | generated Python of small programs against expected values, plus a Chakra export of each |
| `chakra` | Python | decoded traces, placement, replicas/sequences, and the paper's GPT-2/Switch counts |
| `onnx` | numpy, onnx, onnxruntime | ONNX numerics against NumPy, weights, overwrite protection, reduced GPT-2 |

Tests needing missing packages are reported as skipped. Choose the interpreter
with `-DSETEP_TEST_PYTHON=/path/to/python` (default: the Python 3 found by
CMake); `python/requirements-test.txt` installs everything. Python 3.11 or newer
is required.

## Paper evaluation

`eval/` contains the scripts behind the evaluation in
the SETEP paper: description size and coverage, the
GPT-2 comparison against Hugging Face (inference, ONNX, training), the Chakra
placement sweep, and ASTRA-sim runs. `eval/NOTES.txt` gives the exact commands
and settings; `eval/requirements-eval.txt` the Python packages.

## Repository layout

| path | contents |
|---|---|
| `src/` | the `setep` tool: generated parser (`SetepParser.*`, `JimJamLexer.*`), syntax tree, semantic IR, and the Python, operation-graph/Chakra and ONNX backends |
| `grammar/` | `Setep.jim` / `Setep.jam`, the JimJam grammar the parser is generated from |
| `python/` | the ONNX graph builder used by `--onnx` |
| `tools/chakra_trace.py` | standard-library Chakra decoder: summary, structural check, comparison |
| `models/` | SETEP models and handwritten hooks |
| `tests/` | ctest suite |
| `eval/` | paper evaluation scripts |
| `docs/` | language specification and backend notes |

The parser sources in `src/` are generated by the JimJam parser generator from
`grammar/`; JimJam itself is not part of this repository.
