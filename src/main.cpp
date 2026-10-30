// setep: SETEP model compiler. Parses a .setep file and generates PyTorch
// source, Chakra execution traces for ASTRA-sim, and ONNX models.
#include "SetepChakra.h"
#include "SetepDistribution.h"
#include "SetepFrontend.h"
#include "SetepGraph.h"
#include "SetepOnnx.h"
#include "SetepSemantic.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

#ifndef SETEP_PYTHON_DIR
#define SETEP_PYTHON_DIR ""
#endif

namespace {
namespace fs = std::filesystem;

const char* usage =
R"(usage: setep MODEL.setep [options]

Parse and check a SETEP model, then generate the requested outputs. With no
backend option the model is only checked.

General:
  -o, --output-dir DIR       write outputs to DIR (default: the model's directory)
  -h, --help                 show this help

PyTorch (--python):
  --python                   write MODEL.py and the shared setep.py runtime
  --partitioned              generate Python from the partition-lowered program

Chakra execution traces for ASTRA-sim (--chakra):
  --chakra                   write MODEL.<rank>.et per occupied processor and
                             MODEL.chakra.json (the manifest)
  --nodes N                  accelerator nodes in the inventory (default 64)
  --processors-per-node N    accelerators per node (default 64)
  --hosts-per-node N         host (CPU) processors per node (default 0)
  --memory-mb MB             memory capacity per accelerator (default 80000)
  --host-memory-mb MB        memory capacity per host processor (default 128000)
  --replicas N               model replicas; 0 places as many as fit (default 1)
  --sequences N              sequences per replica (default 1)
  --gflop-per-cycle X        accelerator arithmetic per cycle (default 1)
  --mb-per-cycle X           accelerator memory bandwidth per cycle (default 10)
  --clock-hz X               cycles per second (default 1e9)

ONNX (--onnx; runs python/setep_onnx.py):
  --onnx                     write MODEL.onnx and MODEL.onnx-export.json
  --weights FILE.npz         weight values (default: MODEL.weights.npz if present)
  --zero-weights             use zero placeholders for weights that are missing
  --onnx-profile NAME        opaque-operation semantics: gpt2-token (GPT-2 adapter)
  --overwrite                replace an earlier export with a matching receipt
  --interpreter EXE          Python interpreter with onnx installed (default python)

Environment: SETEP_PYTHON_DIR overrides the directory holding setep_onnx.py.
)";

struct Options {
  std::string model, outputDir;
  bool python = false, partitioned = false, chakra = false, onnx = false;
  setep::Inventory inventory;
  setep::DistributionOptions distribution;
  setep::OnnxOptions onnxOptions;
};

[[noreturn]] void usageError(const std::string& message) {
  throw std::invalid_argument(message);
}

double number(const std::string& option, const std::string& text) {
  try {
    size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size()) throw std::invalid_argument(text);
    return value;
  } catch (const std::exception&) {
    usageError("option " + option + " needs a number, got '" + text + "'");
  }
}

int whole(const std::string& option, const std::string& text) {
  const double value = number(option, text);
  if (value != static_cast<int>(value)) usageError("option " + option + " needs an integer, got '" + text + "'");
  return static_cast<int>(value);
}

Options parse(int argc, char** argv) {
  Options options;
#ifdef _WIN32
  options.onnxOptions.python = "python";
#else
  options.onnxOptions.python = "python3";
#endif
  options.distribution.replicas = 1;
  const char* bridge = std::getenv("SETEP_PYTHON_DIR");
  options.onnxOptions.bridgeDir = bridge && *bridge ? bridge : SETEP_PYTHON_DIR;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) usageError("option " + arg + " needs a value");
      return argv[++i];
    };
    if (arg == "-h" || arg == "--help") { std::cout << usage; std::exit(0); }
    else if (arg == "-o" || arg == "--output-dir") options.outputDir = value();
    else if (arg == "--python") options.python = true;
    else if (arg == "--partitioned") options.partitioned = true;
    else if (arg == "--chakra") options.chakra = true;
    else if (arg == "--nodes") options.inventory.nodes = whole(arg, value());
    else if (arg == "--processors-per-node") options.inventory.processorsPerNode = whole(arg, value());
    else if (arg == "--hosts-per-node") options.inventory.hostsPerNode = whole(arg, value());
    else if (arg == "--memory-mb") options.inventory.memoryMb = number(arg, value());
    else if (arg == "--host-memory-mb") options.inventory.hostMemoryMb = number(arg, value());
    else if (arg == "--replicas") options.distribution.replicas = whole(arg, value());
    else if (arg == "--sequences") options.distribution.sequences = whole(arg, value());
    else if (arg == "--gflop-per-cycle") options.inventory.gFlopsPerCycle = number(arg, value());
    else if (arg == "--mb-per-cycle") options.inventory.memoryMbPerCycle = number(arg, value());
    else if (arg == "--clock-hz") options.distribution.cyclesPerSecond = number(arg, value());
    else if (arg == "--onnx") options.onnx = true;
    else if (arg == "--weights") options.onnxOptions.weights = value();
    else if (arg == "--zero-weights") options.onnxOptions.zeroWeights = true;
    else if (arg == "--onnx-profile") options.onnxOptions.profile = value();
    else if (arg == "--overwrite") options.onnxOptions.overwrite = true;
    else if (arg == "--interpreter") options.onnxOptions.python = value();
    else if (!arg.empty() && arg[0] == '-') usageError("unknown option " + arg);
    else if (options.model.empty()) options.model = arg;
    else usageError("more than one model given: " + arg);
  }
  if (options.model.empty()) usageError("no model given");
  if (!options.onnxOptions.weights.empty() && options.onnxOptions.zeroWeights)
    usageError("--weights and --zero-weights are exclusive");
  if (options.onnxOptions.bridgeDir.empty() || !fs::exists(fs::path(options.onnxOptions.bridgeDir) / "setep_onnx.py")) {
    const fs::path executable = fs::absolute(argv[0]).parent_path();
    for (const fs::path& candidate : {executable / "python", executable.parent_path() / "python"})
      if (fs::exists(candidate / "setep_onnx.py")) { options.onnxOptions.bridgeDir = candidate.u8string(); break; }
  }
  return options;
}

void write(const fs::path& path, const std::string& text) {
  std::ofstream out(path);
  out << text;
  if (!out) throw std::runtime_error("cannot write " + path.u8string());
}

int run(const Options& options) {
  std::ifstream input(options.model, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open " + options.model);
  const std::string source{std::istreambuf_iterator<char>(input), {}};
  const setep::Module module = setep::parseSource(source);

  const fs::path model(options.model);
  const fs::path directory = options.outputDir.empty()
    ? (model.has_parent_path() ? model.parent_path() : fs::path("."))
    : fs::path(options.outputDir);
  if (!options.outputDir.empty()) fs::create_directories(directory);
  const fs::path stem = directory / model.stem();

  if (options.python) {
    const fs::path path = fs::path(stem).replace_extension(".py");
    write(path, setep::emitPython(options.partitioned ? setep::lowerPartitions(module) : module));
    write(directory / "setep.py", setep::pythonSupport());
    std::cout << "wrote " << path.u8string() << " and " << (directory / "setep.py").u8string() << '\n';
  }
  if (options.chakra) {
    const setep::Graph graph = setep::buildGraph(setep::lowerPartitions(module));
    std::vector<setep::Processor> processors;
    setep::DistributionPlan plan;
    std::string error;
    if (!setep::buildInventory(options.inventory, processors, error) ||
        !setep::buildDistributionPlan(graph.operations, processors, options.distribution, plan, error))
      throw std::runtime_error("Chakra placement: " + error);
    setep::writeChakra(stem.u8string(), graph, plan);
    std::cout << "wrote " << plan.processors.size() << " Chakra rank file(s) and "
              << stem.u8string() << ".chakra.json (" << plan.instances.size()
              << " compute nodes, " << plan.transfers.size() << " transfers)\n";
  }
  if (options.onnx) {
    const fs::path path = fs::path(stem).replace_extension(".onnx");
    setep::exportOnnx(module, options.model, path.u8string(), options.onnxOptions);
  }
  if (!options.python && !options.chakra && !options.onnx)
    std::cout << options.model << ": ok\n";
  return 0;
}
}

int main(int argc, char** argv) {
  Options options;
  try {
    options = parse(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "setep: " << e.what() << "\n(run setep --help for usage)\n";
    return 2;
  }
  try {
    return run(options);
  } catch (const std::exception& e) {
    // Parser messages may span lines; prefix each with the model path.
    std::string message = e.what();
    for (size_t at = 0; (at = message.find('\n', at)) != std::string::npos; at += options.model.size() + 3)
      message.replace(at, 1, "\n" + options.model + ": ");
    std::cerr << options.model << ": " << message << '\n';
    return 1;
  }
}
