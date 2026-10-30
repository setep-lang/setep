// Operation-graph and placement checks: standard operation costs, shapes,
// weight instances, storage identities and memory-aware placement.
#include "SetepDistribution.h"
#include "SetepFrontend.h"
#include "SetepGraph.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {
void check(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
void near(double actual, double expected, const std::string& message) {
  check(std::abs(actual - expected) <= 1e-5 * std::max(1e-9, std::abs(expected)),
        message + ": got " + std::to_string(actual) + ", expected " + std::to_string(expected));
}
setep::Graph load(const std::string& text) {
  return setep::buildGraph(setep::lowerPartitions(setep::parseSource(text)));
}
std::string rejection(const std::string& text) {
  try {
    load(text);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}
setep::Processor processor(double capacityMb) {
  setep::Processor p;
  p.capacityMb = capacityMb;
  return p;
}

void costs() {
  for (const auto& entry : std::vector<std::pair<std::string, double>>{
       {"LN", 44}, {"LAYERNORM", 44}, {"RMSNORM", 28}, {"SOFTMAX", 40},
       {"GELU", 72}, {"SILU", 16}, {"RELU", 8}, {"CUSTOM", 8}, {"SAMPLE", 3}}) {
    const auto g = load("A=2%4@fp32 " + entry.first + "(A){} RUN(A){A >> " + entry.first + "() >> B}");
    near(g.operations.back().ops * 1e9, entry.second, "Standard operation cost: " + entry.first);
  }
  near(load("A=2%4@fp32 LN(A){ops_per_element=0} RUN(A){A >> LN() >> B}").operations.back().ops, 0,
       "Explicit zero must override standard operation cost");
  near(load("A=2%4@fp32 GELU(A){ops_per_element=3} RUN(A){A >> GELU() >> B}").operations.back().ops * 1e9, 24,
       "Explicit operation coefficient must take precedence");
  const auto reads = load("A=2%4@fp32 LN(A){input_reads=2} RUN(A){A >> LN() >> B}");
  near(reads.operations.back().ops * 1e9, 44, "Memory-only override must retain standard arithmetic");
  near(reads.operations.back().mem * 1e6, 64, "input_reads=2 doubles the input bytes read");
  const auto overhead = load("A=2%4@fp32 F(A){overhead_ns=2500; output_writes=3} RUN(A){A >> F() >> B}");
  near(overhead.operations.back().overheadNs, 2500, "Declared overhead");
  near(overhead.operations.back().outputWrites, 3, "Declared output writes");
}

void shapes() {
  auto g = load("A=8%4@fp32 w=4%6@fp32 b=1%6@fp32 CUSTOM(A){} RUN(A){A >> CUSTOM() >> *w +b >> B}");
  check(g.operations.back().outputs.at(0).dims == std::vector<int64_t>({8, 6}), "Bias collapsed flexible projection rows");
  g = load("A=8%6@fp32 b=1%6@fp32 RUN(A){b+A >> B}");
  check(g.operations.back().outputs.at(0).dims == std::vector<int64_t>({8, 6}), "Reversed bias broadcast changed shape");
  g = load("IDS=3@int32 w=7%4@fp16 LOOKUP(IDS:B){} LN(A){} RUN(IDS){IDS:w >> LOOKUP() >> LN() >> B}");
  for (const auto& op : g.operations) {
    check(op.outputs.at(0).dims == std::vector<int64_t>({3, 4}), "Embedding/normalization shape");
    check(op.outputs.at(0).dataSize == "fp16", "Embedding inherited integer index dtype");
  }
  near(g.operations[0].mem * 1e6, 36, "Embedding reads indices plus selected rows");
  near(g.operations[0].ops, 0, "Embedding gather must not charge arithmetic");
  double table = 0;
  for (const auto& entry : g.operations[0].storageMb) if (entry.first.rfind("weight/", 0) == 0) table += entry.second;
  near(table * 1e6, 56, "Embedding table must remain fully resident");
  setep::DistributionOptions tiny; tiny.replicas = 1;
  setep::DistributionPlan plan; std::string error;
  check(!setep::buildDistributionPlan(g.operations, {processor(80e-6)}, tiny, plan, error),
        "Selected-row reads must not make an oversized embedding table fit");
  g = load("A=2%4@fp32 B=2%3@fp32 RUN(A:B){A:B >:> C}");
  check(g.operations.size() == 1 && g.operations.front().name == "CONCAT", "Concatenation must create a copy operation");
  near(g.operations.front().mem * 1e6, 56, "Concatenation source reads");
  near(g.operations.front().ops, 0, "Concatenation must not add arithmetic");
}

void weights() {
  // Only array weights differ between instances; a plain weight has one value
  // even inside a repeated scope, and ! is reserved for persistent tensors.
  const auto g = load("A=2%4@fp32 w=4%4@fp32 u[3]=4%4@fp32 RUN(A){A >> REPEAT/3{*w >> *u[]} >> B}");
  const auto instances = [&](const std::string& field) {
    return std::count_if(g.weightShapes.begin(), g.weightShapes.end(),
      [&](const auto& item) { return item.first.rfind(field + "/", 0) == 0 || item.first == field; });
  };
  check(instances("weight_w") == 1, "Plain weight must have one value across repetitions");
  check(instances("weight_u") == 3, "Array weight must have one value per repetition");
  check(rejection("A=2%4@fp32 w=4%4@fp32 RUN(A){A >> *w! >> B}").find("remove ! from weight") != std::string::npos,
        "! on a weight must be rejected");
}

void placement() {
  setep::DistributionOptions options; options.replicas = 1; options.sequences = 2;
  setep::DistributionPlan plan; std::string error;
  auto g = load("A=2%4@fp32 w=4%4@fp32 RUN(A){A*w >> B; B*w^ >> C}");
  check(setep::buildDistributionPlan(g.operations, {processor(1)}, options, plan, error), error);
  near(plan.residentMb.at(0) * 1e6, 64 + 2 * (32 + 32 + 32), "Shared/transposed weight residency and separate sequence buffers");
  g = load("A=2%4@fp32 w=4%4@fp32 v=4%4@fp32 RUN(A){A*w >> B; B*v >> C}");
  check(setep::buildDistributionPlan(g.operations, {processor(1)}, options, plan, error), error);
  near(plan.residentMb.at(0) * 1e6, 128 + 2 * (32 + 32 + 32), "Equal-shaped distinct weight banks must not share storage");
  check(plan.instances.size() == 4 && plan.transfers.empty(), "Two sequences of two operations on one processor");
  // Next-fit: a small capacity forces each matrix product onto its own processor.
  setep::DistributionOptions one; one.replicas = 1;
  check(setep::buildDistributionPlan(g.operations, {processor(130e-6), processor(130e-6)}, one, plan, error), error);
  check(plan.processors.size() == 2 && plan.transfers.size() == 1, "Split placement transfers one activation");
  check(plan.transfers[0].bytes == 32, "Transfer carries the 2x4 fp32 activation");
  // Persistent state: a cache written once and read later shares one allocation.
  g = load("A=2%4@fp32 v=4%3@fp32 RUN(A){A >> K!; K!*v >> B}");
  bool cache = false;
  for (const auto& entry : g.operations.back().storageMb) cache = cache || entry.first.rfind("cache/", 0) == 0;
  check(cache, "Persistent input must be stored as cache");
}

void gpt2(const std::string& root) {
  std::ifstream input(std::filesystem::path(root) / "models/chatgpt2/chatgpt2.setep");
  std::string text((std::istreambuf_iterator<char>(input)), {});
  for (const auto& replacement : std::vector<std::pair<std::string, std::string>>{
       {"tokens = 10", "tokens = 8"}, {"decode_tokens = 10", "decode_tokens = 2"}}) {
    const auto at = text.find(replacement.first);
    check(at != std::string::npos, "GPT-2 fixture scalar missing");
    text.replace(at, replacement.first.size(), replacement.second);
  }
  const auto g = load(text);
  std::map<int, int> scores; int masks = 0, samples = 0; double matmulOps = 0;
  for (const auto& op : g.operations) {
    const auto& shape = op.outputs.at(0);
    if (op.name == "MATMUL") {
      check(shape.dataSize == "fp32", "GPT-2 matrix output must be floating point");
      matmulOps += op.ops * 1e9;
      if (shape.dims == std::vector<int64_t>({8, 8})) ++scores[8];
      if (shape.dims == std::vector<int64_t>({1, 9})) ++scores[9];
      if (shape.dims == std::vector<int64_t>({1, 10})) ++scores[10];
    }
    if (op.name == "DECODE_PREP") {
      check(shape.dims == std::vector<int64_t>({9 + masks}), "Decode padding mask must grow with cache");
      ++masks;
    }
    if (op.name == "SAMPLE") {
      check(shape.dims == std::vector<int64_t>({1}) && shape.dataSize == "int32", "Sample shape/type");
      ++samples;
    }
  }
  for (int length : {8, 9, 10}) check(scores[length] == 144, "Attention cache length/head count mismatch");
  check(masks == 2 && samples == 3, "Prefill/decode iteration count mismatch");
  double expected = 0;
  for (int length : {8, 9, 10}) {
    const double m = length == 8 ? 8 : 1, d = 768, f = 3072, v = 50257;
    expected += 12 * (8 * m * d * d + 4 * m * d * f + 4 * m * length * d) + 2 * m * d * v;
  }
  near(matmulOps, expected, "GPT-2 matrix arithmetic differs from the independent QKV/attention/MLP/head formula");
}
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: setep_graph_test SOURCE_DIR\n";
    return 2;
  }
  try {
    costs();
    shapes();
    weights();
    placement();
    gpt2(argv[1]);
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << '\n';
    return 1;
  }
  std::cout << "Operation graph and placement checks passed\n";
  return 0;
}
