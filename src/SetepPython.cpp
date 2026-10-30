// PyTorch source backend: one generated model class per SETEP file, plus the
// shared setep.py runtime. Opaque computations become overridable methods.
#include "SetepSemantic.h"
#include "SetepPythonRuntime.h"
#include <set>
#include <sstream>
#include <stdexcept>

namespace setep {
namespace {
class PythonEmitter {
  const Module& m;
  std::ostringstream out;
  std::string pad(int i) { return std::string(i*4,' '); }
  std::string join(const std::vector<std::string>& a,const std::string& sep=", ") {
    std::string r; for(const std::string& s:a) { if(!r.empty()) r+=sep; r+=s; } return r;
  }
  std::string expression(const Expression& e) {
    using E=Expression;
    std::vector<std::string> args; for(const Expression& a:e.args) args.push_back(expression(a));
    const std::string receiver="self.";
    switch(e.kind) {
    case E::TensorOp: return "tensor_op(\""+e.name+"\", ["+join(args)+"])";
    case E::ValidateShape: { std::vector<std::string> dims(args.begin()+1,args.end()); return "require_shape("+args[0]+", ["+join(dims)+"])"; }
    case E::ValidateArray: return "require_array_size("+join(args)+")";
    case E::Record: return e.name+"("+join(args)+")";
    case E::Array:
      if(!e.name.empty()) return args[0];
      return "checked_array(["+join(args)+"])";
    case E::ArrayAt: return args[0]+"["+args[1]+"]";
    case E::ArraySize: return "len("+args[0]+")";
    case E::ReduceScatter: return "reduce_scatter("+args[0]+")";
    case E::Local: return e.name;
    case E::StringLiteral: return "\""+e.name+"\"";
    case E::WeightWrite: return receiver+e.name+".update("+args[0]+", "+args[1]+")";
    case E::Scale: return "("+args[0]+" * "+args[1]+")";
    case E::Weight: return receiver+e.name+".at("+join(args)+")";
    case E::StateRead: return receiver+e.name+".at("+join(args)+")";
    case E::StateWrite: return receiver+e.name+".set("+join(args)+")";
    case E::StateConcat: return "concat_state(["+join(args)+"])";
    case E::Call: return receiver+e.name+"("+join(args)+")";
    case E::Opaque: {
      const std::vector<std::string> inputs(args.begin(),args.end()-2);
      return receiver+e.name+"("+(inputs.empty()?"":join(inputs)+", ")+"instance_path="+args[args.size()-2]+", occurrence="+args.back()+")";
    }
    case E::Predicate: {
      std::string call=receiver+e.name+"("+args[0]+", "+args[1];
      if(args.size()==4) call+=", repetition=RepetitionInfo("+args[2]+", "+args[3]+")";
      return "predicate_value("+call+"))";
    }
    case E::Add: return "_add("+join(args)+")";
    case E::Subtract: return "("+args[0]+" - "+args[1]+")";
    case E::Matmul: return "_matmul("+join(args)+")";
    case E::Transpose: return args[0]+".transpose(-2, -1)";
    case E::Split: return e.name.empty()?"split("+join(args)+")":"split_axis("+join(args)+", "+e.name+")";
    case E::Member: return args[0]+"."+e.name;
    case E::Group: return e.name=="literal" ? "["+join(args)+"]" : args[0];
    case E::GroupMember: return "group_member("+join(args)+")";
    case E::IndexedPath: return args.empty() ? "()" : "child_path("+join(args)+")";
    case E::Merge:
      if(e.name.rfind("axis:",0)==0) return "merge_axis("+args[0]+", "+e.name.substr(5)+")";
      if(e.name=="group") return "merge_group(["+join(args)+"])";
      if(e.name=="sum") return "reduce_sum("+args[0]+")";
      if(e.name=="family") return "merge("+args[0]+")";
      return "merge(["+join(args)+"])";
    }
    throw std::logic_error("invalid semantic expression");
  }
  static std::string pythonType(const std::string& type) { return type=="std::vector<Tensor>"?"list":type; }
  void statements(const std::vector<Statement>& ss,int depth,const std::string& function,std::string instancePath="instance_path") {
    for(const Statement& s:ss) {
      if(s.kind==Statement::Check) {
        out<<pad(depth)<<"if not "<<expression(s.value)<<": raise ValueError(\""<<s.name<<"\")\n";
      } else if(s.kind==Statement::Branch) {
        std::vector<std::string> choices;
        for(size_t i=0;i<s.alternatives.size();++i) {
          const std::string choice=s.name+"_"+std::to_string(i);
          choices.push_back(choice);
          out<<pad(depth)<<choice<<" = "<<expression(s.predicates[i])<<"\n";
        }
        if(s.exclusive)
          out<<pad(depth)<<"if ("<<join(choices," + ")<<") != 1: raise ValueError(\"enumerated choice requires exactly one selection\")\n";
        for(size_t i=0;i<s.alternatives.size();++i) {
          out<<pad(depth)<<"if "<<choices[i]<<":\n";
          statements(s.alternatives[i],depth+1,function,instancePath);
        }
      } else if(s.kind==Statement::Loop) {
        if(!s.selector.empty()) {
          out<<pad(depth)<<"for "<<s.name<<" in selected("<<s.selector<<", "<<s.count<<", "<<(s.reverse?"True":"False")<<"):\n";
        } else {
          out<<pad(depth)<<"for "<<s.name<<" in range(";
          if(s.reverse) out<<s.count<<" - 1, -1, -1"; else out<<s.count;
          out<<"):\n";
        }
        out<<pad(depth+1)<<"_log(f\""<<function<<" loop "<<s.name<<" {"<<s.name<<"}/{"<<s.count<<"} at {"<<instancePath<<"}\")\n";
        statements(s.body,depth+1,function,instancePath);
      } else if(s.kind==Statement::Extend) {
        out<<pad(depth)<<s.name<<".extend("<<expression(s.value)<<")\n";
      } else if(s.kind==Statement::Append) {
        if(s.reverse) out<<pad(depth)<<s.name<<".insert(0, "<<expression(s.value)<<")\n";
        else out<<pad(depth)<<"array_push("<<s.name<<", "<<expression(s.value)<<")\n";
      } else {
        out<<pad(depth)<<s.name<<" = "<<(s.group?"[]":expression(s.value))<<"\n";
        if(!s.tensorDeclaration.empty()) out<<pad(depth)<<s.name<<" = _declared(self, "<<s.name<<", \""<<s.tensorDeclaration<<"\")\n";
        if(s.value.kind==Expression::IndexedPath && s.name.find("scope_instance_path_")==0) instancePath=s.name;
      }
    }
  }
  std::string resultType(const Function& f) {
    return f.arrayDepth?"list":f.returnsGroup?"list[Tensor]":f.outputs.size()==1?"Tensor":f.name+"Result";
  }
public:
  explicit PythonEmitter(const Module& module):m(module) {}
  std::string run() {
    out<<"\"\"\"Generated SETEP model; shared runtime is in setep.py.\"\"\"\n"
      "from dataclasses import dataclass\nimport torch\nfrom torch import Tensor, nn\n"
      "from setep import (WeightBank, StateBank, concat_state, RepetitionInfo, child_path, split, merge,\n"
      "    tensor_op, require_shape, split_axis, merge_axis, reduce_sum, reduce_scatter, require_array_size, checked_array, array_push, merge_group, group_member, selected, predicate_value, run_demo,\n"
      "    set_debug, _log, _matmul, _add, _declared, _placeholder, _Shape, _ShapePlan)\n\n";
    for(const auto& s:m.scalars) out<<s.first<<" = "<<s.second<<"\n";
    out<<"TENSOR_SHAPES = {\n";
    for(const auto& shape:m.tensorShapes) out<<"    \""<<shape.first<<"\": ("<<join(shape.second)<<",),\n";
    out<<"}\n";
    out<<"TENSOR_DATA_SIZES = {\n";
    for(const auto& size:m.tensorDataSizes) out<<"    \""<<size.first<<"\": \""<<size.second<<"\",\n";
    out<<"}\n";
    out<<"WEIGHT_SHAPES = {\n";
    for(const auto& shape:m.weightShapes) out<<"    \""<<shape.first<<"\": ("<<join(shape.second)<<",),\n";
    out<<"}\nWEIGHT_DATA_SIZES = {\n";
    for(const auto& size:m.weightDataSizes) out<<"    \""<<size.first<<"\": \""<<size.second<<"\",\n";
    out<<"}\n";
    for(size_t i=0;i<m.functions.size();++i) out<<m.functions[i].name<<"_family = "<<i<<"\n";
    for(const auto& record:m.inputRecords) {
      out<<"\n@dataclass\nclass "<<record.first<<":\n";
      for(const std::string& field:record.second) out<<pad(1)<<field<<": Tensor\n";
    }
    std::vector<Function> resultFunctions=m.functions;
    for(const auto& signature:m.opaqueSignatures) resultFunctions.push_back(signature.second);
    for(const Function& f:resultFunctions) if(f.outputs.size()>1) {
      out<<"\n@dataclass\nclass "<<f.name<<"Result:\n";
      for(size_t i=0;i<f.outputs.size();++i) {
        const int depth=i<f.outputArrayDepths.size()?f.outputArrayDepths[i]:0;
        out<<pad(1)<<f.outputs[i]<<(depth?": list":": Tensor")<<"\n";
      }
    }
    out<<"\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n";
    out<<"        self._input_names = [\""<<join(m.functions.back().inputs,"\", \"")<<"\"]\n";
    out<<"        self._input_types = {";
    for(const auto& item:m.functions.back().inputTypes) out<<"\""<<item.first<<"\": "<<pythonType(item.second)<<", ";
    out<<"}\n        self._input_array_sizes = {";
    for(const auto& item:m.functions.back().inputArraySizes) out<<"\""<<item.first<<"\": "<<item.second<<", ";
    out<<"}\n        self._input_declarations = {";
    for(const auto& item:m.functions.back().inputDeclarations) out<<"\""<<item.first<<"\": \""<<item.second<<"\", ";
    out<<"}\n        self._tensor_shapes = TENSOR_SHAPES\n";
    for(const std::string& w:m.weights) {
      out<<"        self."<<w;
      if(m.weightDeclarations.count(w)) {
        const std::string& declaration=m.weightDeclarations.at(w);
        out<<" = WeightBank(WEIGHT_SHAPES[\""<<declaration<<"\"], WEIGHT_DATA_SIZES[\""<<declaration<<"\"])\n";
      } else out<<" = WeightBank()\n";
    }
    for(const std::string& state:m.states) out<<"        self."<<state<<" = StateBank()\n";
    if(!m.states.empty()) {
      out<<"\n    def reset_state(self):\n";
      for(const std::string& state:m.states) out<<pad(2)<<"self."<<state<<".clear()\n";
    }
    for(const std::string& op:m.operations) {
      const Function signature=m.opaqueSignatures.count(op)?m.opaqueSignatures.at(op):Function{};
      const std::vector<std::string>& inputs=signature.inputs;
      const std::string result=signature.outputs.size()>1?op+"Result":"Tensor";
      // With no inputs the demo placeholder takes its shape from the declared destination.
      const std::string shapeSource=inputs.empty()?"None":inputs.front();
      std::vector<std::string> parameters;
      for(const std::string& input:inputs) parameters.push_back(input+": Tensor");
      const std::string leading=parameters.empty()?"":join(parameters)+", ";
      out<<"\n    def "<<op<<"(self, "<<leading<<"instance_path, occurrence: str) -> "<<result<<":\n"
        "        # Not implemented: return shape-compatible demo placeholders.\n"
        "        _log(f\""<<op<<" not implemented at {instance_path}: {occurrence}\")\n        return ";
      if(signature.outputs.size()>1) {
        out<<result<<"(\n";
        for(const std::string& field:signature.outputs) out<<"            _placeholder(self, "<<shapeSource<<", instance_path, occurrence + \"_"<<field<<"\"),\n";
        out<<"        )\n";
      } else out<<"_placeholder(self, "<<shapeSource<<", instance_path, occurrence)\n";
    }
    for(const std::string& selector:m.selectors)
      out<<"\n    def "<<selector<<"(self, choice_index: int, value: Tensor, repetition=None) -> Tensor:\n        # Not implemented: the demo selects the first choice.\n        _log(f\""<<selector<<" not implemented: choice={choice_index}, repetition={repetition}\")\n        return torch.tensor([int(choice_index == 0)])\n";
    for(const Function& f:m.functions) {
      out<<"\n    def "<<f.name<<"(self, ";
      for(const std::string& arg:f.inputs)
        out<<arg<<": "<<pythonType(f.inputTypes.count(arg)?f.inputTypes.at(arg):"Tensor")<<", ";
      out<<"instance_path=()";
      if(f.needsInstance) out<<", instance=0";
      out<<", state_path=()) -> "<<resultType(f)<<":\n";
      out<<pad(2)<<"_log(f\""<<f.name<<" start at {instance_path}\")\n";
      statements(f.body,2,f.name);
      out<<pad(2)<<"_log(f\""<<f.name<<" done at {instance_path}\")\n";
      out<<pad(2)<<"return ";
      if(f.outputs.size()>1) out<<f.name<<"Result(";
      for(size_t i=0;i<f.results.size();++i) { if(i) out<<", "; out<<expression(f.results[i]); }
      if(f.outputs.size()>1) out<<")";
      out<<"\n";
    }
    const Function& entry=m.functions.back();
    out<<"\n    def forward(self, ";
    std::vector<std::string> args;
    for(const std::string& arg:entry.inputs)
      args.push_back(arg+": "+pythonType(entry.inputTypes.count(arg)?entry.inputTypes.at(arg):"Tensor"));
    out<<join(args)<<"):\n        return self."<<entry.name<<"("<<join(entry.inputs)<<")\n";
    out<<"\nif __name__ == \"__main__\":\n    import argparse\n    parser = argparse.ArgumentParser(description=\"SETEP PyTorch demo\")\n    parser.add_argument(\"--debug\", action=\"store_true\", help=\"Print model progress and operation sizes\")\n    args = parser.parse_args()\n    set_debug(args.debug)\n    torch.set_num_threads(1)\n    run_demo(Model(), "<<entry.inputs.size()<<")\n";
    return out.str();
  }
};
}
std::string emitPython(const Module& module) { return PythonEmitter(module).run(); }
std::string pythonSupport() { return pythonRuntime; }
}
