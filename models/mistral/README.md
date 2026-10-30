# Mistral Medium 3.5 128B (text decoder)

Entry: `MISTRAL` in `mistral.setep`. Source checked 2026-09-29.

88 dense decoder layers, GQA with 96 query / 8 KV heads, 128-dimensional heads. The vision encoder/projector and FP8 checkpoint quantization are omitted.

Source: [published reference](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B/blob/f38972c0ef3023b423b0c9157edb11d02c865b28/config.json). The numerical sizing choices are
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

Export Chakra traces (one replica, 80 GB per accelerator) from the repository root:

```text
build/setep models/mistral/mistral.setep --chakra -o out --memory-mb 80000
```

Whole operations must fit the per-accelerator memory
capacity. The planner uses a resident-memory proxy, not tensor-lifetime peak
memory. Kernel timings are not calibrated to measured hardware.
