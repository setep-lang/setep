// ONNX backend: the semantic module is handed to python/setep_onnx.py, which
// builds and checks the ONNX graph with the onnx package.
#include "SetepOnnx.h"
#include "SetepSemantic.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace setep {
namespace {
namespace fs = std::filesystem;

// Minimal JSON writer for the request; values are strings, numbers, booleans,
// arrays and objects with string keys.
std::string quote(const std::string& value) {
  std::string out = "\"";
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') { out += '\\'; out += static_cast<char>(c); }
    else if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else if (c < 0x20) { char buffer[8]; std::snprintf(buffer, sizeof buffer, "\\u%04x", c); out += buffer; }
    else out += static_cast<char>(c);
  }
  return out + "\"";
}
std::string json(const std::string& value) { return quote(value); }
std::string json(bool value) { return value ? "true" : "false"; }
std::string json(int value) { return std::to_string(value); }
template <class T> std::string json(const std::vector<T>& values) {
  std::string out = "[";
  for (size_t i = 0; i < values.size(); ++i) out += (i ? "," : "") + json(values[i]);
  return out + "]";
}
template <class T> std::string json(const std::map<std::string, T>& values) {
  std::string out = "{";
  bool first = true;
  for (const auto& item : values) { out += (first ? "" : ",") + quote(item.first) + ":" + json(item.second); first = false; }
  return out + "}";
}
class Object {
  std::string text_ = "{";
public:
  template <class T> Object& add(const std::string& key, const T& value) { return raw(key, json(value)); }
  Object& raw(const std::string& key, const std::string& value) {
    text_ += (text_.size() > 1 ? "," : "") + quote(key) + ":" + value;
    return *this;
  }
  std::string str() const { return text_ + "}"; }
};

std::string expression(const Expression& e) {
  static const char* kinds[] = {"Local","Weight","Call","Opaque","TensorOp","Add","Subtract","Matmul","Transpose",
    "Merge","Split","Member","IndexedPath","StringLiteral","Predicate","Group","GroupMember","Array","ArrayAt",
    "ArraySize","ValidateShape","ReduceScatter","Record","ValidateArray","StateRead","StateWrite","StateConcat","WeightWrite","Scale"};
  std::string args = "[";
  for (size_t i = 0; i < e.args.size(); ++i) args += (i ? "," : "") + expression(e.args[i]);
  return Object().add("kind", std::string(kinds[e.kind])).add("name", e.name).raw("args", args + "]").str();
}
std::string statements(const std::vector<Statement>& input) {
  static const char* kinds[] = {"Assign","Loop","Append","Branch","Extend","Check"};
  std::string out = "[";
  for (size_t i = 0; i < input.size(); ++i) {
    const auto& s = input[i];
    if (s.kind == Statement::Branch) throw std::runtime_error("ONNX export does not support dynamic SETEP choices");
    out += (i ? "," : "") + Object().add("kind", std::string(kinds[s.kind])).add("name", s.name).add("count", s.count)
      .add("selector", s.selector).add("group", s.group).add("reverse", s.reverse).add("declaration", s.tensorDeclaration)
      .raw("value", expression(s.value)).raw("body", statements(s.body)).str();
  }
  return out + "]";
}
std::string function(const Function& f) {
  std::string results = "[";
  for (size_t i = 0; i < f.results.size(); ++i) results += (i ? "," : "") + expression(f.results[i]);
  return Object().add("name", f.name).add("inputs", f.inputs).add("outputs", f.outputs)
    .add("needs_instance", f.needsInstance).add("consumes_input", f.consumesInput)
    .add("returns_group", f.returnsGroup).add("array_depth", f.arrayDepth)
    .add("input_types", f.inputTypes).add("input_declarations", f.inputDeclarations)
    .add("input_array_sizes", f.inputArraySizes).raw("body", statements(f.body))
    .raw("results", results + "]").str();
}

int launch(const std::vector<std::string>& args) {
#ifdef _WIN32
  // Windows process creation has a command line but no shell. Quote using the
  // documented CRT argv rules, including trailing backslashes and quotes.
  std::wstring command;
  for (const auto& arg : args) {
    if (!command.empty()) command += L' ';
    const auto wide = fs::u8path(arg).wstring();
    command += L'"'; size_t slashes = 0;
    for (wchar_t c : wide) {
      if (c == L'\\') { ++slashes; continue; }
      command.append(c == L'"' ? slashes * 2 + 1 : slashes, L'\\'); slashes = 0; command += c;
    }
    command.append(slashes * 2, L'\\'); command += L'"';
  }
  STARTUPINFOW start{}; start.cb = sizeof(start);
  start.dwFlags = STARTF_USESTDHANDLES;
  start.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  start.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
  start.hStdError = GetStdHandle(STD_ERROR_HANDLE);
  PROCESS_INFORMATION process{};
  if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &start, &process))
    throw std::runtime_error("cannot launch the Python interpreter '" + args.front() + "'; set it with --python");
  WaitForSingleObject(process.hProcess, INFINITE);
  DWORD code = 1; GetExitCodeProcess(process.hProcess, &code);
  CloseHandle(process.hThread); CloseHandle(process.hProcess);
  return static_cast<int>(code);
#else
  pid_t pid = fork();
  if (pid < 0) throw std::runtime_error("cannot fork the ONNX exporter");
  if (pid == 0) {
    std::vector<char*> argv;
    for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);
    execvp(argv[0], argv.data());
    std::perror(argv[0]);
    _exit(127);
  }
  int status = 0;
  while (waitpid(pid, &status, 0) < 0) if (errno != EINTR) throw std::runtime_error("cannot wait for the ONNX exporter");
  return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
#endif
}

struct TemporaryFile {
  fs::path path;
  TemporaryFile() {
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    path = fs::temp_directory_path() / ("setep-onnx-" + std::to_string(stamp) + ".json");
  }
  ~TemporaryFile() { std::error_code ignored; fs::remove(path, ignored); }
};
}

void exportOnnx(const Module& input, const std::string& source, const std::string& output,
                const OnnxOptions& options) {
  const Module m = lowerPartitions(input);
  std::string functions = "[";
  for (size_t i = 0; i < m.functions.size(); ++i) functions += (i ? "," : "") + function(m.functions[i]);
  const std::string module = Object().add("scalars", m.scalars).add("tensor_shapes", m.tensorShapes)
    .add("tensor_dtypes", m.tensorDataSizes).add("weight_shapes", m.weightShapes)
    .add("weight_dtypes", m.weightDataSizes).add("weight_declarations", m.weightDeclarations)
    .raw("functions", functions + "]").add("states", m.states).add("input_records", m.inputRecords).str();
  const std::string request = Object().add("operation", std::string("setep_to_onnx"))
    .add("model", fs::absolute(source).u8string()).add("output", fs::absolute(output).u8string())
    .raw("module", module).add("placeholder_weights", options.zeroWeights)
    .add("weights", options.weights.empty() ? std::string() : fs::absolute(options.weights).u8string())
    .add("profile", options.profile).add("overwrite_generated", options.overwrite).str();
  const fs::path script = fs::path(options.bridgeDir) / "setep_onnx.py";
  if (!fs::exists(script))
    throw std::runtime_error("ONNX exporter not found: " + script.u8string() + " (set SETEP_PYTHON_DIR)");
  TemporaryFile file;
  {
    std::ofstream out(file.path, std::ios::binary);
    out << request;
    if (!out) throw std::runtime_error("cannot write ONNX request " + file.path.u8string());
  }
  if (launch({options.python, script.u8string(), "--request", file.path.u8string()}) != 0)
    throw std::runtime_error("ONNX export failed (see the message above; the Python environment needs python/requirements-onnx.txt)");
}
}
