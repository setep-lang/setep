#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace setep {
struct Module;

struct TensorShape {
  std::vector<int64_t> dims;
  int bits = 0;
  std::string dataSize;
};

// One executed operation of the elaborated SETEP program: every repetition,
// array element and alternative is expanded. Costs are per execution.
struct Operation {
  int id = -1;
  std::string name;                // SETEP operation or computation name
  std::vector<int> froms, tos;     // dependencies, as operation IDs
  float ops = 0;                   // arithmetic, GFLOP
  float mem = 0;                   // bytes read while executing, MB
  float out = 0;                   // output bytes sent to each consumer, MB
  float residentMem = 0;           // sum of storage, MB
  double outputWrites = 1;         // memory writes per output byte
  double overheadNs = 0;           // declared fixed overhead
  // Allocation identities (weights, activations, inputs, caches) and their
  // full backing sizes in MB. Equal identities share storage on one processor.
  std::map<std::string, double> storageMb;
  std::vector<TensorShape> outputs; // flattened output tensors
};

struct Graph {
  std::vector<Operation> operations; // topological: dependencies precede users
  // Weight instances (bank field plus instance path) and their shapes.
  std::map<std::string, TensorShape> weightShapes;
};

// Elaborate the partition-lowered module into its operation graph. Throws
// std::runtime_error with a "SETEP graph lowering" message on invalid input.
Graph buildGraph(const Module& module);
}
