#include "SetepDistribution.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <set>

namespace setep {

bool buildInventory(const Inventory& inventory, std::vector<Processor>& processors, std::string& error) {
  processors.clear();
  error.clear();
  if (inventory.nodes <= 0 || inventory.processorsPerNode <= 0 || inventory.hostsPerNode < 0 ||
      int64_t(inventory.nodes) * (inventory.processorsPerNode + inventory.hostsPerNode) > INT_MAX) {
    error = "Invalid processor inventory: need at least one node and one processor per node";
    return false;
  }
  for (int node = 0; node < inventory.nodes; ++node)
    for (int slot = 0; slot < inventory.processorsPerNode + inventory.hostsPerNode; ++slot) {
      const bool host = slot >= inventory.processorsPerNode;
      processors.push_back({0, node, slot, host,
        host ? inventory.hostMemoryMb : inventory.memoryMb,
        host ? inventory.hostGFlopsPerCycle : inventory.gFlopsPerCycle,
        host ? inventory.hostMemoryMbPerCycle : inventory.memoryMbPerCycle});
    }
  return true;
}

bool buildDistributionPlan(const std::vector<Operation>& operations,
    const std::vector<Processor>& processors,
    const DistributionOptions& options, DistributionPlan& plan,
    std::string& error) {
  plan = {};
  error.clear();
  auto fail = [&](const std::string& message) { error = message; return false; };
  if (operations.empty() || processors.empty())
    return fail("Cannot distribute a workload without operations and processors");
  if (options.sequences <= 0 || options.replicas < 0 ||
      !std::isfinite(options.cyclesPerSecond) || options.cyclesPerSecond <= 0)
    return fail("Invalid sequence count, replica count, or clock rate");

  std::vector<size_t> processorRanks[2];
  double capacity[2] = {0, 0};
  for (size_t rank = 0; rank < processors.size(); ++rank) {
    const auto& processor = processors[rank];
    if (!std::isfinite(processor.capacityMb) || processor.capacityMb <= 0 ||
        !std::isfinite(processor.gFlopsPerCycle) || processor.gFlopsPerCycle <= 0 ||
        !std::isfinite(processor.memoryMbPerCycle) || processor.memoryMbPerCycle <= 0)
      return fail("Invalid processor capacity or throughput");
    // One capacity for each processor type.
    if (!processorRanks[processor.host].empty() &&
        capacity[processor.host] != processor.capacityMb)
      return fail("Mixed capacities within one processor type are unsupported");
    capacity[processor.host] = processor.capacityMb;
    processorRanks[processor.host].push_back(rank);
  }
  if (processorRanks[0].empty()) return fail("Cannot place a workload without accelerators");

  std::map<int, size_t> operationById;
  int64_t stride = 1;
  for (size_t i = 0; i < operations.size(); ++i) {
    if (operations[i].id < 0 || !operationById.emplace(operations[i].id, i).second)
      return fail("Operation IDs must be unique and nonnegative");
    stride = std::max(stride, int64_t(operations[i].id) + 1);
  }
  std::vector<int> indegree(operations.size(), 0);
  std::vector<std::vector<size_t>> successors(operations.size());
  for (size_t i = 0; i < operations.size(); ++i) {
    std::set<int> unique;
    for (int id : operations[i].froms) {
      auto source = operationById.find(id);
      if (source == operationById.end() || !unique.insert(id).second)
        return fail("Missing or duplicate operation dependency");
      successors[source->second].push_back(i);
      ++indegree[i];
    }
  }
  using Ready = std::pair<int, size_t>;
  std::priority_queue<Ready, std::vector<Ready>, std::greater<Ready>> ready;
  for (size_t i = 0; i < operations.size(); ++i)
    if (!indegree[i]) ready.push({operations[i].id, i});
  std::vector<size_t> order;
  while (!ready.empty()) {
    const size_t i = ready.top().second;
    ready.pop();
    order.push_back(i);
    for (size_t next : successors[i])
      if (!--indegree[next]) ready.push({operations[next].id, next});
  }
  if (order.size() != operations.size()) return fail("Cannot schedule a workload with a dependency cycle");

  // Every SETEP operation runs on an accelerator; hosts stay available but empty.
  const bool host = false;
  struct Slot { double memory; };
  std::vector<Slot> slots;
  std::map<int, std::map<std::string, double>> slotStorage;
  // Weights are shared by sequences on one processor; other storage is per sequence.
  auto storageSize = [&](const std::pair<const std::string, double>& entry) {
    return entry.second * (entry.first.rfind("weight/", 0) != 0 ? options.sequences : 1);
  };
  auto additionalStorage = [&](int slot, const Operation& operation, double legacy) {
    if (operation.storageMb.empty()) return legacy;
    double extra = 0;
    for (const auto& entry : operation.storageMb) {
      if (!std::isfinite(entry.second) || entry.second < 0) return std::numeric_limits<double>::infinity();
      extra += std::max(0.0, storageSize(entry) - slotStorage[slot][entry.first]);
    }
    return extra;
  };
  auto addStorage = [&](int slot, const Operation& operation) {
    for (const auto& entry : operation.storageMb)
      slotStorage[slot][entry.first] = std::max(slotStorage[slot][entry.first], storageSize(entry));
  };
  std::vector<int> operationSlots(operations.size(), -1);
  std::vector<double> resident(operations.size());
  for (size_t i = 0; i < operations.size(); ++i) {
    const auto& operation = operations[i];
    resident[i] = operation.residentMem >= 0 ? operation.residentMem : operation.mem;
    if (!operation.storageMb.empty()) {
      resident[i] = 0;
      for (const auto& entry : operation.storageMb) resident[i] += storageSize(entry);
    }
    if (!std::isfinite(resident[i]) || resident[i] < 0 || resident[i] > capacity[host])
      return fail("Operation " + std::to_string(operation.id) + " (" + operation.name +
                  ") exceeds the per-processor memory capacity");
  }
  // Next-fit packing in operation order: only the current processor can accept
  // another operation. Never backfill closed processors.
  int current = -1;
  for (size_t i = 0; i < operations.size(); ++i) {
    const auto& operation = operations[i];
    if (current < 0 || slots[current].memory + additionalStorage(current, operation, resident[i]) > capacity[host]) {
      current = static_cast<int>(slots.size());
      slots.push_back({0});
    }
    operationSlots[i] = current;
    slots[current].memory += additionalStorage(current, operation, resident[i]);
    addStorage(current, operation);
  }
  const int replicasThatFit = static_cast<int>(processorRanks[host].size() / slots.size());
  const int replicaCount = options.replicas > 0 ? options.replicas : replicasThatFit;
  if (replicaCount <= 0 || replicaCount > replicasThatFit)
    return fail("Requested workload replicas do not fit on the available processors (one replica needs " +
                std::to_string(slots.size()) + ")");
  if (stride > INT_MAX / int64_t(options.sequences) / replicaCount)
    return fail("Distributed runtime IDs exceed the supported range");

  DistributionPlan result;
  result.processors = processors;
  result.options = options;
  result.operationsPerReplica = operations.size();
  result.residentMb.resize(processors.size(), 0);
  std::set<size_t> claimed;
  size_t nextRank = 0;
  for (int replica = 0; replica < replicaCount; ++replica) {
    std::vector<size_t> ranks;
    for (const auto& slot : slots) {
      auto& candidates = processorRanks[host];
      if (nextRank == candidates.size()) return fail("Unable to place a complete workload replica");
      const size_t rank = candidates[nextRank++];
      claimed.insert(rank);
      ranks.push_back(rank);
      result.residentMb[rank] = slot.memory;
    }
    result.replicaRanks.push_back(ranks);
    for (int sequence = 0; sequence < options.sequences; ++sequence) {
      std::map<size_t, size_t> instanceByOperation;
      for (size_t operationIndex : order) {
        const auto& operation = operations[operationIndex];
        DistributedInstance instance;
        instance.operationIndex = operationIndex;
        instance.runtimeId = static_cast<int>((replica * int64_t(options.sequences) + sequence) * stride + operation.id);
        instance.replica = replica;
        instance.sequence = sequence;
        instance.rank = ranks[operationSlots[operationIndex]];
        const auto& processor = processors[instance.rank];
        // Cycles: arithmetic, then reads plus output writes, then declared overhead.
        const double effort = operation.ops / processor.gFlopsPerCycle
          + (operation.mem + 0.0 + std::max(0.0, operation.outputWrites) * operation.out) / processor.memoryMbPerCycle
          + operation.overheadNs * options.cyclesPerSecond / 1e9 + 1;
        if (!std::isfinite(effort) || effort < 0 || effort > INT_MAX ||
            operation.ops < 0 || operation.mem < 0 ||
            !std::isfinite(operation.out) || operation.out < 0 ||
            double(operation.out) * 1e6 >= double(std::numeric_limits<int64_t>::max()))
          return fail("Operation effort or output size is outside the supported range");
        instance.effortCycles = static_cast<int>(effort);
        for (int dependency : operation.froms)
          instance.dependencies.push_back(instanceByOperation.at(operationById.at(dependency)));
        instanceByOperation[operationIndex] = result.instances.size();
        result.instances.push_back(std::move(instance));
      }
    }
  }
  std::map<std::pair<size_t, size_t>, size_t> transferByDestination;
  for (size_t consumer = 0; consumer < result.instances.size(); ++consumer) {
    const auto& instance = result.instances[consumer];
    for (size_t source : instance.dependencies) {
      if (result.instances[source].rank == instance.rank) continue;
      const auto key = std::make_pair(source, instance.rank);
      auto inserted = transferByDestination.emplace(key, result.transfers.size());
      if (inserted.second) {
        DistributedTransfer transfer;
        transfer.source = source;
        transfer.destinationRank = instance.rank;
        transfer.bytes = static_cast<uint64_t>(std::llround(double(operations[result.instances[source].operationIndex].out) * 1e6));
        result.instances[source].outgoingTransfers.push_back(result.transfers.size());
        result.transfers.push_back(std::move(transfer));
      }
      result.transfers[inserted.first->second].consumers.push_back(consumer);
    }
  }
  // Export only occupied processors. Compact in inventory order so rank IDs
  // are stable and the manifest gives the required ASTRA-sim endpoint count.
  std::vector<size_t> compactRank(processors.size());
  result.processors.clear();
  std::vector<double> compactMemory;
  for (size_t rank = 0; rank < processors.size(); ++rank) {
    if (!claimed.count(rank)) continue;
    compactRank[rank] = result.processors.size();
    result.processors.push_back(processors[rank]);
    compactMemory.push_back(result.residentMb[rank]);
  }
  result.residentMb = std::move(compactMemory);
  for (auto& ranks : result.replicaRanks)
    for (auto& rank : ranks) rank = compactRank[rank];
  for (auto& instance : result.instances) instance.rank = compactRank[instance.rank];
  for (auto& transfer : result.transfers) transfer.destinationRank = compactRank[transfer.destinationRank];
  plan = std::move(result);
  return true;
}
}
