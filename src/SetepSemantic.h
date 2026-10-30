#pragma once

#include <map>
#include <string>
#include <vector>

namespace setepSyntax { class SetepFile; }

// Backend-independent, structured tensor program shared by the Python, Chakra
// and ONNX backends. No placement or scheduling information is stored here.
namespace setep {
// Physical partition intent, separate from explicit array structure.
struct TensorLayout {
  std::vector<std::string> logicalShape;
  std::vector<int> splitCounts;
};
struct Expression {
  enum Kind { Local, Weight, Call, Opaque, TensorOp, Add, Subtract, Matmul, Transpose,
              Merge, Split, Member, IndexedPath, StringLiteral, Predicate, Group, GroupMember,
              Array, ArrayAt, ArraySize, ValidateShape, ReduceScatter, Record, ValidateArray, StateRead, StateWrite, StateConcat, WeightWrite, Scale } kind = Local;
  std::string name;
  std::vector<Expression> args;
  std::vector<int> splitAxes;
  TensorLayout layout;
  bool logicalAssembly = false; // Partition reconstruction for validation, not a physical copy.
};
struct Statement {
  enum Kind { Assign, Loop, Append, Branch, Extend, Check } kind = Assign;
  std::string name, count, selector;
  std::string tensorDeclaration;
  std::string sourceName; // Original tensor name before fresh-local renaming.
  Expression value;
  std::vector<Statement> body;
  std::vector<std::vector<Statement>> alternatives;
  std::vector<Expression> predicates;
  int arrayDepth = 1;
  bool exclusive = false;
  bool group = false, reverse = false;
  bool named = false; // Explicit SETEP tensor destination, not an anonymous result.
};
struct Function {
  std::string name;
  bool needsInstance = false;
  bool consumesInput = true;
  bool returnsGroup = false;
  int arrayDepth = 0;
  std::vector<std::string> inputs, outputs;
  std::vector<int> outputArrayDepths;
  std::vector<std::vector<int>> outputSplitAxes;
  std::map<std::string, std::string> inputTypes, inputDeclarations, inputArraySizes;
  std::vector<Statement> body;
  std::vector<Expression> results;
};
struct AbstractCost {
  double opsPerElement=1, inputReads=1, outputWrites=1, weightPercent=100, overheadNs=0;
  bool explicitOperations=false;
};
struct Module {
  std::string entryName;
  std::map<std::string, std::string> scalars;
  std::map<std::string, std::vector<std::string>> tensorShapes;
  std::map<std::string, std::string> tensorDataSizes;
  std::map<std::string, std::vector<int>> shapeSplits;
  std::map<std::string, std::vector<std::string>> weightShapes;
  std::map<std::string, std::string> weightDataSizes, weightDeclarations;
  std::map<std::string, std::vector<std::string>> inputRecords;
  std::vector<Function> functions;
  std::vector<std::string> weights, operations;
  std::vector<std::string> states;
  std::map<std::string, Function> opaqueSignatures;
  std::map<std::string, AbstractCost> abstractCosts;
  std::vector<std::string> selectors;
};
Module lower(setepSyntax::SetepFile& file);
// Expand supported physical partitions; behavioral lowering remains unchanged.
Module lowerPartitions(const Module& module);
// PyTorch source for the model, and the shared setep.py support module.
std::string emitPython(const Module& module);
std::string pythonSupport();
}
