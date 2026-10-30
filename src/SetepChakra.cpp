// Chakra execution trace (ET) writer for ASTRA-sim. The protobuf wire format is
// encoded directly, so neither Chakra nor protobuf is a build dependency.
#include "SetepChakra.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace setep {
namespace {

using Bytes = std::vector<uint8_t>;

void varint(Bytes& out, uint64_t value) {
  while (value >= 0x80) {
    out.push_back(static_cast<uint8_t>(value) | 0x80);
    value >>= 7;
  }
  out.push_back(static_cast<uint8_t>(value));
}

void key(Bytes& out, unsigned field, unsigned wire) {
  varint(out, (static_cast<uint64_t>(field) << 3) | wire);
}

void integer(Bytes& out, unsigned field, uint64_t value) {
  key(out, field, 0);
  varint(out, value);
}

void stringField(Bytes& out, unsigned field, const std::string& value) {
  key(out, field, 2);
  varint(out, value.size());
  out.insert(out.end(), value.begin(), value.end());
}

void message(Bytes& out, unsigned field, const Bytes& value) {
  key(out, field, 2);
  varint(out, value.size());
  out.insert(out.end(), value.begin(), value.end());
}

Bytes booleanAttribute(const std::string& name, bool value) {
  Bytes attr;
  stringField(attr, 1, name);
  integer(attr, 27, value);
  return attr;
}

Bytes integerAttribute(const std::string& name, uint64_t value, unsigned valueField) {
  Bytes attr;
  stringField(attr, 1, name);
  integer(attr, valueField, value);
  return attr;
}

std::string shapeText(const std::vector<TensorShape>& shapes) {
  std::string text = "[";
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (i) text += ',';
    text += '[';
    for (size_t j = 0; j < shapes[i].dims.size(); ++j) {
      if (j) text += ',';
      text += std::to_string(shapes[i].dims[j]);
    }
    text += ']';
  }
  return text + ']';
}

Bytes ioInfo(const std::vector<TensorShape>& shapes) {
  Bytes io;
  stringField(io, 2, shapeText(shapes));
  std::string types = "[";
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (i) types += ',';
    types += shapes[i].dataSize.empty()
      ? "fp" + std::to_string(shapes[i].bits ? shapes[i].bits : 32)
      : shapes[i].dataSize;
  }
  stringField(io, 3, types + ']');
  return io;
}

void writeRecord(std::ofstream& out, const Bytes& record) {
  Bytes length;
  varint(length, record.size());
  out.write(reinterpret_cast<const char*>(length.data()), length.size());
  out.write(reinterpret_cast<const char*>(record.data()), record.size());
}

uint64_t scaled(double value, double factor) {
  const double result = std::max(0.0, value) * factor;
  return result >= static_cast<double>(std::numeric_limits<uint64_t>::max())
    ? std::numeric_limits<uint64_t>::max()
    : static_cast<uint64_t>(std::round(result));
}

std::string jsonString(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char character : value) {
    if (character == '"' || character == '\\') out << '\\' << character;
    else if (character < 0x20)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(character);
    else out << character;
  }
  out << '"';
  return out.str();
}

} // namespace

void writeChakra(const std::string& prefixText, const Graph& graph, const DistributionPlan& plan) {
  const auto& operations = graph.operations;
  if (plan.instances.empty() || plan.processors.empty())
    throw std::runtime_error("Cannot export Chakra without a distributed workload");
  if (plan.transfers.size() >= static_cast<size_t>(std::numeric_limits<int32_t>::max()))
    throw std::runtime_error("Too many Chakra communication tags");
  const std::filesystem::path prefix(prefixText);
  // Stage every file before publishing, so generation errors leave existing
  // traces intact. A manifest is published last and defines the rank count.
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  std::filesystem::path staging(prefix.string() + ".chakra-tmp-" + std::to_string(nonce));
  std::vector<std::pair<std::filesystem::path, std::filesystem::path>> files;
  auto cleanup = [&]() {
    std::error_code ignored;
    for (const auto& file : files) std::filesystem::remove(file.first, ignored);
    std::filesystem::remove(staging, ignored);
  };
  try {
    if (!std::filesystem::create_directory(staging))
      throw std::runtime_error("Cannot create Chakra staging directory");
    uint64_t nextId = 1;
    for (const auto& instance : plan.instances)
      nextId = std::max(nextId, static_cast<uint64_t>(instance.runtimeId) + 2);
    const uint64_t communicationBase = nextId;
    std::map<std::pair<size_t, size_t>, size_t> transferByDestination;
    std::vector<std::vector<size_t>> instancesByRank(plan.processors.size());
    for (size_t i = 0; i < plan.instances.size(); ++i)
      instancesByRank.at(plan.instances[i].rank).push_back(i);
    for (size_t i = 0; i < plan.transfers.size(); ++i) {
      const auto& transfer = plan.transfers[i];
      transferByDestination[{transfer.source, transfer.destinationRank}] = i;
    }
    auto computeId = [&](size_t instance) {
      return static_cast<uint64_t>(plan.instances.at(instance).runtimeId) + 1;
    };
    auto sendId = [&](size_t transfer) { return communicationBase + transfer * 2; };
    auto receiveId = [&](size_t transfer) { return sendId(transfer) + 1; };
    auto communication = [&](size_t index, bool send) {
      const auto& transfer = plan.transfers[index];
      Bytes node;
      integer(node, 1, send ? sendId(index) : receiveId(index));
      stringField(node, 2, std::string(send ? "send_" : "recv_") + std::to_string(index + 1));
      integer(node, 3, send ? 5 : 6); // COMM_SEND_NODE / COMM_RECV_NODE
      if (send) integer(node, 5, computeId(transfer.source));
      // Communication is issued by the endpoint's NIC, including host ranks.
      message(node, 10, booleanAttribute("is_cpu_op", false));
      message(node, 10, integerAttribute("comm_src", plan.instances[transfer.source].rank, 11));
      message(node, 10, integerAttribute("comm_dst", transfer.destinationRank, 11));
      message(node, 10, integerAttribute("comm_tag", index + 1, 11));
      message(node, 10, integerAttribute("comm_size", transfer.bytes, 13));
      return node;
    };
    for (size_t rank = 0; rank < plan.processors.size(); ++rank) {
      const std::filesystem::path target(prefix.string() + "." + std::to_string(rank) + ".et");
      const auto temporary = staging / target.filename();
      files.push_back({temporary, target});
      std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
      out.exceptions(std::ios::failbit | std::ios::badbit);
      Bytes metadata;
      stringField(metadata, 1, "0.0.4");
      writeRecord(out, metadata);
      std::set<size_t> emittedReceives;
      for (size_t index : instancesByRank[rank]) {
        const auto& instance = plan.instances[index];
        const Operation& operation = operations.at(instance.operationIndex);
        Bytes node;
        integer(node, 1, computeId(index));
        stringField(node, 2, operation.name.empty()
          ? "operation_" + std::to_string(operation.id) : operation.name);
        integer(node, 3, 4); // COMP_NODE
        for (size_t dependency : instance.dependencies) {
          uint64_t id = computeId(dependency);
          if (plan.instances[dependency].rank != rank) {
            const size_t transfer = transferByDestination.at({dependency, rank});
            // Keep receives near their first consumer for streaming feeders.
            // They have no foreign-file dependency IDs: readiness comes from
            // the matching send, and local compute waits for this receive.
            if (emittedReceives.insert(transfer).second)
              writeRecord(out, communication(transfer, false));
            id = receiveId(transfer);
          }
          integer(node, 5, id);
        }
        // Chakra stores integral microseconds. Preserve the modeled compute
        // effort for replay, rounding up to its minimum nonzero unit.
        integer(node, 7, std::max<uint64_t>(1, scaled(
          std::ceil(instance.effortCycles * 1e6 / plan.options.cyclesPerSecond), 1)));
        std::vector<TensorShape> inputs;
        for (size_t dependency : instance.dependencies) {
          const auto& shapes = operations.at(plan.instances[dependency].operationIndex).outputs;
          inputs.insert(inputs.end(), shapes.begin(), shapes.end());
        }
        message(node, 8, ioInfo(inputs));
        message(node, 9, ioInfo(operation.outputs));
        message(node, 10, booleanAttribute("is_cpu_op", plan.processors[rank].host));
        message(node, 10, integerAttribute("num_ops", scaled(operation.ops, 1e9), 13));
        // ASTRA-sim's roofline tensor_size measures memory traffic, not output
        // bytes. Output payload sizing is carried by the communication nodes.
        message(node, 10, integerAttribute("tensor_size", scaled(operation.mem, 1e6), 13));
        message(node, 10, integerAttribute("setep_operation_id", operation.id, 9));
        message(node, 10, integerAttribute("setep_replica", instance.replica, 9));
        message(node, 10, integerAttribute("setep_sequence", instance.sequence, 9));
        writeRecord(out, node);
        for (size_t transfer : instance.outgoingTransfers)
          writeRecord(out, communication(transfer, true));
      }
      out.close();
    }
    const std::filesystem::path manifestPath(prefix.string() + ".chakra.json");
    const auto temporaryManifest = staging / manifestPath.filename();
    files.push_back({temporaryManifest, manifestPath});
    std::ofstream manifest(temporaryManifest);
    manifest.exceptions(std::ios::failbit | std::ios::badbit);
    manifest << "{\n  \"schema_version\": 1,\n  \"workload_prefix\": "
      << jsonString(prefix.generic_string())
      << ",\n  \"num_ranks\": " << plan.processors.size()
      << ",\n  \"replicas\": " << plan.replicaRanks.size()
      << ",\n  \"sequences_per_replica\": " << plan.options.sequences
      << ",\n  \"cycles_per_second\": " << plan.options.cyclesPerSecond
      << ",\n  \"compute_instances\": " << plan.instances.size()
      << ",\n  \"transfers\": " << plan.transfers.size()
      << ",\n  \"ranks\": [\n";
    for (size_t rank = 0; rank < plan.processors.size(); ++rank) {
      const auto& processor = plan.processors[rank];
      manifest << (rank ? ",\n" : "") << "    {\"rank\": " << rank
        << ", \"processor\": \"" << (processor.host ? "cpu" : "gpu")
        << "\", \"address\": [" << processor.group << ", "
        << processor.node << ", " << processor.slot
        << "], \"capacity_mb\": " << processor.capacityMb
        << ", \"resident_mb\": " << plan.residentMb.at(rank) << "}";
    }
    manifest << "\n  ]\n}\n";
    manifest.close();
    // All content is now complete. Replace only this workload's generated
    // files; publish the manifest last. Do not traverse/remove directories.
    for (const auto& file : files)
      std::filesystem::copy_file(file.first, file.second,
        std::filesystem::copy_options::overwrite_existing);
    // Remove obsolete rank files after a successful smaller export.
    const auto parent = prefix.has_parent_path() ? prefix.parent_path() : std::filesystem::path(".");
    const std::string basename = prefix.filename().string() + ".";
    for (const auto& entry : std::filesystem::directory_iterator(parent)) {
      const std::string name = entry.path().filename().string();
      if (!entry.is_regular_file() || name.compare(0, basename.size(), basename) != 0 ||
          entry.path().extension() != ".et") continue;
      const std::string number = name.substr(basename.size(), name.size() - basename.size() - 3);
      if (number.empty() || number.find_first_not_of("0123456789") != std::string::npos) continue;
      // Avoid parsing unbounded filenames and remove only canonical rank names.
      if (number.size() > 9 || (number.size() > 1 && number.front() == '0')) continue;
      if (std::stoul(number) >= plan.processors.size()) std::filesystem::remove(entry.path());
    }
    cleanup();
  } catch (const std::exception& exception) {
    cleanup();
    throw std::runtime_error("Unable to export Chakra traces '" + prefix.string() + "': " + exception.what());
  }
}
}
