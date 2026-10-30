# Llama 4 Maverick 17B-128E (text decoder)

Entry: `LLAMA` in `llama.setep`. Source checked 2026-09-29.

48 layers: 24 dense and 24 MoE. MoE uses 128 routed experts plus one shared expert; real routing selects one expert/token. Three chunked RoPE layers then one full NoPE layer repeat 12 times. QK normalization is disabled for Maverick. Vision and long-context temperature tuning are omitted.

Source: [published reference](https://github.com/meta-pytorch/torchtune/blob/main/torchtune/models/llama4/_model_builders.py). The numerical sizing choices are
also recorded in `model-info.json`.

This is a text-inference workload model for SETEP operation-graph and Chakra analysis,
not a pretrained numerical implementation. `tokens=1` means stateless one-token
prefill, not cached decoding. Change the scalar in the file to model larger
prefills. No training, tokenizer, image/audio path, persistent KV-cache memory,
or cache-read bandwidth is included. BF16 (2 bytes/element) is the sizing policy,
even if the published checkpoint uses quantized weights. Generated Python custom
operations are placeholders and cannot reproduce the released model's logits.

Projection and FFN matrix products and per-head attention matrix products are
explicit. RMS normalization, RoPE, embedding lookup, grouped-KV expansion, masks,
and routing are external shape contracts. Each layer instance has separate
weights; expert instances have separate expert weights. The LM head is untied.

**MoE:** the operation graph and Chakra traces execute every expert on all
tokens; `active_experts` records the published top-k selection but does not make
the generated Python or Chakra traces sparse.

Export Chakra traces (one replica, 80 GB per accelerator) from the repository root:

```text
build/setep models/llama/llama.setep --chakra -o out --memory-mb 80000
```

Whole operations must fit the per-accelerator memory
capacity. The planner uses a resident-memory proxy, not tensor-lifetime peak
memory. Kernel timings are not calibrated to measured hardware.
