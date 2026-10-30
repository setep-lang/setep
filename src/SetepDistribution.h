#pragma once

#include "SetepGraph.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace setep {

// Processor inventory, in stable rank order. Host (CPU) processors are
// separate endpoints; accelerators are never silently folded into hosts.
struct Processor {
  int group = 0, node = 0, slot = 0; // reported address
  bool host = false;
  double capacityMb = 0;
  double gFlopsPerCycle = 1;
  double memoryMbPerCycle = 1;
};

// Accelerator nodes with an optional number of host processors per node.
// Each node holds processorsPerNode accelerators followed by its hosts.
struct Inventory {
  int nodes = 64;
  int processorsPerNode = 64;
  int hostsPerNode = 0;
  double memoryMb = 80000, hostMemoryMb = 128000;
  double gFlopsPerCycle = 1, memoryMbPerCycle = 10;
  double hostGFlopsPerCycle = 1, hostMemoryMbPerCycle = 10;
};

struct DistributionOptions {
  int replicas = 0; // zero: as many complete replicas as fit
  int sequences = 1;
  double cyclesPerSecond = 1e9;
};

struct DistributedInstance {
  size_t operationIndex = 0;
  int runtimeId = -1;
  int replica = 0;
  int sequence = 0;
  size_t rank = 0;
  int effortCycles = 0;
  // Indices into DistributionPlan::instances, not operation IDs.
  std::vector<size_t> dependencies;
  std::vector<size_t> outgoingTransfers;
};

struct DistributedTransfer {
  size_t source = 0;
  size_t destinationRank = 0;
  // One producer output per destination processor, even with local fan-out.
  std::vector<size_t> consumers;
  uint64_t bytes = 0;
};

struct DistributionPlan {
  std::vector<Processor> processors; // occupied processors only
  DistributionOptions options;
  std::vector<std::vector<size_t>> replicaRanks;
  std::vector<DistributedInstance> instances; // topological, oldest sequence first
  std::vector<DistributedTransfer> transfers;
  std::vector<double> residentMb;
  size_t operationsPerReplica = 0;
};

bool buildInventory(const Inventory& inventory, std::vector<Processor>& processors, std::string& error);

// Memory-aware next-fit placement of whole replicas, then per-sequence instances
// and the coalesced transfers between processors. On failure, plan is empty and
// error explains why placement or dependency planning failed.
bool buildDistributionPlan(const std::vector<Operation>& operations,
    const std::vector<Processor>& processors,
    const DistributionOptions& options, DistributionPlan& plan,
    std::string& error);
}
