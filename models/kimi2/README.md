# Kimi K2 Instruct

Entry: `KIMI2` in `kimi2.setep`. Source checked 2026-09-29.

61 MLA layers: one dense FFN layer followed by 60 MoE layers. 384 routed experts plus one shared expert; real routing is top-8. Query rank 1536, KV rank 512, 64 attention heads, 128 non-RoPE/value dimensions and 64 RoPE dimensions. Sigmoid router scores, correction bias, top-k normalization and routed scale 2.827 are contracted inside ROUTER/SCALE_ROUTED. The correction-bias storage and FP8 block-scale storage are omitted.

Source: [published reference](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json). The numerical sizing choices are
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
build/setep models/kimi2/kimi2.setep --chakra -o out --memory-mb 80000
```

Whole operations must fit the per-accelerator memory
capacity. The planner uses a resident-memory proxy, not tensor-lifetime peak
memory. Kernel timings are not calibrated to measured hardware.
