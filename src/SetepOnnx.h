#pragma once

#include <string>

namespace setep {
struct Module;

struct OnnxOptions {
  std::string python;      // interpreter with python/requirements-onnx.txt installed
  std::string bridgeDir;   // directory holding setep_onnx.py
  std::string weights;     // NPZ file; empty uses SOURCE_STEM.weights.npz when present
  bool zeroWeights = false; // allow deterministic zero placeholders for missing weights
  std::string profile;     // "" for generic math, "gpt2-token" for the GPT-2 adapter
  bool overwrite = false;  // replace an earlier export carrying a matching receipt
};

// Serialize the partition-lowered module and run the Python ONNX lowering,
// which writes OUTPUT and OUTPUT_STEM.onnx-export.json. Throws on failure.
void exportOnnx(const Module& module, const std::string& source, const std::string& output,
                const OnnxOptions& options);
}
