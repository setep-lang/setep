#include "SetepSemantic.h"
#include "SetepSyntax.h"
#include "SetepDefinitions.h"
#include <algorithm>
#include <cctype>
#include <set>
#include <stdexcept>

namespace setep {
namespace {
using namespace setepSyntax;
using E = Expression;
using Values = std::vector<E>;
std::string base(std::string s) {
  while (!s.empty() && (s.back()==']' || s.back()=='[' || s.back()=='!' || s.back()=='^')) s.pop_back();
  return s;
}
std::string tensorKey(std::string name) {
  size_t element=name.find("[]"); if(element!=std::string::npos) name.erase(element,2);
  if(!name.empty() && name.back()=='^') name.pop_back();
  return name;
}
std::string id(const std::string& s) {
  std::string r;
  for (unsigned char c : base(s)) {
    if (c=='\'') r += "_prime";
    else r += std::isalnum(c) || c=='_' ? char(c) : '_';
  }
  static const std::set<std::string> reserved={"class","def","return","for","while","if","else","try","except","auto","int","float","double","char","bool","void","const","public","private","protected","virtual","static","new","delete","template","typename","namespace","struct","union","using","import","from","with","as","pass","raise","lambda","self","instance_path","state_path","instance","input","True","False","None","Model","Tensor","WeightBank","StateBank","forward"};
  return reserved.count(r) ? "setep_"+r : r;
}
bool weightName(const std::string& name) {
  return !name.empty() && std::islower(static_cast<unsigned char>(name.front()));
}
E local(std::string s) { return {E::Local, std::move(s), {}, {}, {}}; }
E expr(E::Kind k, Values a, std::string n={}) { return {k,std::move(n),std::move(a),{}, {}}; }
bool hasPartitions(const TensorLayout& layout) {
  return std::any_of(layout.splitCounts.begin(),layout.splitCounts.end(),[](int n){return n>1;});
}
int layoutDimension(const Module& module,const std::string& value) {
  return std::stoi(module.scalars.count(value)?module.scalars.at(value):value);
}
TensorLayout elementwiseLayout(const Module& module,const TensorLayout& a,const TensorLayout& b) {
  if(a.logicalShape.empty() || b.logicalShape.empty()) return {};
  const size_t rank=std::max(a.logicalShape.size(),b.logicalShape.size());
  TensorLayout result; result.logicalShape.resize(rank); result.splitCounts.resize(rank);
  for(size_t i=0;i<rank;++i) {
    const int ai=static_cast<int>(i)-static_cast<int>(rank-a.logicalShape.size());
    const int bi=static_cast<int>(i)-static_cast<int>(rank-b.logicalShape.size());
    const std::string ad=ai<0?"1":a.logicalShape[ai], bd=bi<0?"1":b.logicalShape[bi];
    const int av=layoutDimension(module,ad), bv=layoutDimension(module,bd);
    const int ac=ai<0?1:a.splitCounts[ai], bc=bi<0?1:b.splitCounts[bi];
    if(av!=bv && av!=1 && bv!=1) throw std::runtime_error("elementwise broadcast dimension mismatch");
    if(ac>1 && bc>1 && ac!=bc) throw std::runtime_error("conflicting elementwise split counts");
    result.logicalShape[i]=av==1?bd:ad; result.splitCounts[i]=std::max(ac,bc);
  }
  return result;
}
TensorLayout matmulLayout(const Module& module,const TensorLayout& left,const TensorLayout& right) {
  if(left.logicalShape.empty() || right.logicalShape.empty()) return {};
  if((left.logicalShape.size()!=1 && left.logicalShape.size()!=2) || right.logicalShape.size()!=2) {
    if(hasPartitions(left) || hasPartitions(right))
      throw std::runtime_error("partitioned multiplication supports vector/matrix times matrix");
    return {};
  }
  if(layoutDimension(module,left.logicalShape.back())!=layoutDimension(module,right.logicalShape[0]))
    throw std::runtime_error("multiplication contraction dimension mismatch");
  if(left.splitCounts.back()>1 && right.splitCounts[0]>1 && left.splitCounts.back()!=right.splitCounts[0])
    throw std::runtime_error("conflicting multiplication contraction split counts");
  TensorLayout result=left;
  result.logicalShape.back()=right.logicalShape.back();
  result.splitCounts.back()=right.splitCounts.back();
  return result;
}
int arrayDepth(const E& value) {
  if(value.kind!=E::Array) return 0;
  return value.name.empty() ? 1+arrayDepth(value.args.front()) : std::stoi(value.name);
}
E merged(const Values& v) {
  if (v.empty()) throw std::runtime_error("empty tensor group");
  Values tensors;
  for(const E& value:v) tensors.push_back(value.kind==E::Array || value.kind==E::Group?expr(E::Merge,{value},"family"):value);
  return tensors.size()==1 ? tensors.front() : expr(E::Merge,tensors,"group");
}
std::vector<std::string> names(Tensor* t) {
  std::vector<std::string> r;
  for (TensorGroup* g:t->tensorGroup_) {
    if (!g->tensorEnum_) throw std::runtime_error("explicit tensor families require layout metadata");
    r.insert(r.end(),g->tensorEnum_->upperName_.begin(),g->tensorEnum_->upperName_.end());
  }
  if (t->tensorGroup_.size()!=1) throw std::runtime_error("nested tensor groups require layout metadata");
  return r;
}

void countUses(const E& value,std::map<std::string,size_t>& uses) {
  if(value.kind==E::Local) ++uses[value.name];
  for(const E& arg:value.args) countUses(arg,uses);
}
void countLocals(const std::vector<Statement>& body,std::map<std::string,size_t>& uses,
                 std::map<std::string,size_t>& definitions) {
  for(const Statement& statement:body) {
    if(statement.kind==Statement::Assign) ++definitions[statement.name];
    if(statement.kind==Statement::Append) ++uses[statement.name];
    if(!statement.selector.empty()) ++uses[statement.selector];
    countUses(statement.value,uses);
    for(const E& predicate:statement.predicates) countUses(predicate,uses);
    countLocals(statement.body,uses,definitions);
    for(const std::vector<Statement>& branch:statement.alternatives) countLocals(branch,uses,definitions);
  }
}
void nameResults(std::vector<Statement>& body,const std::map<std::string,size_t>& uses,
                 const std::map<std::string,size_t>& definitions) {
  for(Statement& statement:body) {
    nameResults(statement.body,uses,definitions);
    for(std::vector<Statement>& branch:statement.alternatives) nameResults(branch,uses,definitions);
  }
  // Rename at the producer's original position: do not move computations past
  // calls, weight reads, or loops. Only a single-use/single-definition local
  // can lose its copy; loop-carried values and explicitly named tensors remain.
  for(size_t i=0;i<body.size();++i) {
    Statement& producer=body[i];
    if(producer.kind!=Statement::Assign || producer.group || producer.named ||
       producer.value.kind==E::Local || producer.value.kind==E::IndexedPath ||
       definitions.at(producer.name)!=1 || !uses.count(producer.name) ||
       uses.at(producer.name)!=1) continue;
    for(size_t j=i+1;j<body.size();++j) {
      const Statement& destination=body[j];
      if(destination.kind==Statement::Assign && destination.named && !destination.group &&
         destination.value.kind==E::Local && destination.value.name==producer.name &&
         definitions.at(destination.name)==1) {
        producer.name=destination.name;
        producer.named=true;
        producer.tensorDeclaration=destination.tensorDeclaration;
        producer.sourceName=destination.sourceName;
        body.erase(body.begin()+j);
        break;
      }
    }
  }
}
void nameResults(Function& function) {
  std::map<std::string,size_t> uses, definitions;
  countLocals(function.body,uses,definitions);
  for(const E& result:function.results) countUses(result,uses);
  nameResults(function.body,uses,definitions);
}

class Lowering {
  Module module;
  std::map<std::string,E> symbols;
  std::map<std::string,Function> definitions;
  std::map<std::string,size_t> opaqueOccurrences;
  std::vector<Statement>* body=nullptr;
  std::string owner, index="instance", repetitionCount;
  E instancePath=local("instance_path");
  E sharedPath=local("instance_path");
  E weightPath=local("instance_path");
  std::map<std::string,std::string> weightArrays, declarationArrays;
  E statePath=local("state_path");
  std::string stateOwner;
  std::vector<std::string> stateRepetitions;
  std::map<std::string,std::vector<std::string>> stateLayouts;
  int serial=0;
  Function* currentFunction=nullptr;
  std::map<std::string,std::string> scalarBindings, tensorBindings, weightBindings;
  std::string card(Card* c) {
    if(!c) throw std::runtime_error("cardinality is required");
    if(!c->number_.empty()) {
      if(std::stoll(c->number_)<=0) throw std::runtime_error("cardinality must be positive");
      return c->number_;
    }
    const std::string plain=id(c->lowerName_);
    const std::string name=scalarBindings.count(plain)?scalarBindings.at(plain):plain;
    std::map<std::string,std::string>::const_iterator found=module.scalars.find(name);
    if(found==module.scalars.end()) throw std::runtime_error("undefined scalar "+c->lowerName_);
    if(std::stoll(found->second)<=0) throw std::runtime_error("cardinality must be positive");
    return name;
  }
  // NAME[n] = shape declares an array of n values of that shape.
  void registerArray(Card* count,const std::string& name) {
    if(count) declarationArrays[name]=card(count);
  }
  void registerSplits(ShapeDef* shape,const std::string& name) {
    std::vector<int> counts;
    for(size_t axis=0;axis<shape->card_.size();++axis) {
      const std::string dim=card(shape->card_[axis]);
      const std::string split=axis<shape->splits_.size() && shape->splits_[axis]?card(shape->splits_[axis]):"1";
      const int size=std::stoi(module.scalars.count(dim)?module.scalars.at(dim):dim);
      const int count=std::stoi(module.scalars.count(split)?module.scalars.at(split):split);
      if(size%count) throw std::runtime_error("shape dimension is not divisible by split count: "+name);
      counts.push_back(count);
    }
    module.shapeSplits[name]=counts;
  }
  E declaredTensor(E value,const std::string& declaration,bool weight=false) {
    const auto& shapes=weight?module.weightShapes:module.tensorShapes;
    if(!shapes.count(declaration)) return value;
    TensorLayout layout{shapes.at(declaration),module.shapeSplits.at(declaration)};
    if(!value.layout.logicalShape.empty()) {
      auto resolved=[&](const std::vector<std::string>& dims) {
        std::vector<std::string> result;
        for(const auto& dim:dims) result.push_back(module.scalars.count(dim)?module.scalars.at(dim):dim);
        return result;
      };
      if(resolved(value.layout.logicalShape)!=resolved(layout.logicalShape) || value.layout.splitCounts!=layout.splitCounts)
        throw std::runtime_error("conflicting declared tensor dimensions or splits: "+declaration);
    }
    value.layout=std::move(layout);
    return value;
  }
  E declaredArray(E value,const std::string& declaration) {
    if(value.kind!=E::Array || arrayDepth(value)!=1) throw std::runtime_error("array declaration requires a single layer array: "+declaration);
    const auto axes=value.splitAxes;
    value=expr(E::Array,{expr(E::ValidateArray,{value,local(declarationArrays.at(declaration))})},"1");
    value.splitAxes=axes;
    return mapArray(value,[&](const E& element,const E&) {
      E typed=declaredTensor(element,declaration);
      Values arguments{element};
      for(const auto& dim:module.tensorShapes.at(declaration)) arguments.push_back(local(dim));
      E checked=expr(E::ValidateShape,arguments); checked.layout=typed.layout;
      return checked;
    });
  }
  E transposed(const E& value) {
    if(value.kind!=E::Array) {
      E result=expr(E::Transpose,{value}); result.layout=value.layout;
      if(result.layout.logicalShape.size()>=2) {
        const size_t last=result.layout.logicalShape.size()-1;
        std::swap(result.layout.logicalShape[last],result.layout.logicalShape[last-1]);
        std::swap(result.layout.splitCounts[last],result.layout.splitCounts[last-1]);
      }
      return result;
    }
    E result=mapArray(value,[&](const E& child,const E&){return transposed(child);});
    result.splitAxes=value.splitAxes;
    for(int& axis:result.splitAxes) { if(axis==0) axis=1; else if(axis==1) axis=0; }
    return result;
  }
  E assemble(const E& value) {
    if(value.kind!=E::Array) throw std::runtime_error("array concatenation requires an array");
    E flat=value;
    if(arrayDepth(value)>1) flat=mapArray(value,[&](const E& child,const E&){return assemble(child);});
    return assign(expr(E::Merge,{flat},value.splitAxes.empty()?"family":"axis:"+std::to_string(value.splitAxes.front())),"concatenated");
  }
  // Zero-input abstract calls (call statement -> computation) whose result has not yet
  // reached a declared tensor, and the "call.field" members of such calls that have.
  std::map<std::string,std::string> undeclaredSources;
  std::set<std::string> declaredMembers;
  void declaredSource(const E& value) {
    if(value.kind==E::Local) undeclaredSources.erase(value.name);
    else if(value.kind==E::Member && value.args.size()==1 && value.args.front().kind==E::Local)
      declaredMembers.insert(value.args.front().name+"."+value.name);
  }
  void requireDeclaredSources() {
    for(const auto& pending:undeclaredSources) {
      const Function& signature=module.opaqueSignatures.at(id(pending.second));
      bool declared=signature.outputs.size()>1;
      // Each output member needs a declared destination or a global declaration of its name.
      for(const std::string& field:signature.outputs)
        declared=declared && (declaredMembers.count(pending.first+"."+field) || module.tensorShapes.count(field));
      if(!declared) throw std::runtime_error("abstract computation "+pending.second+" has no inputs, so its result needs a declared destination: "+
        pending.second+"() >> R with R = shape");
    }
    undeclaredSources.clear(); declaredMembers.clear();
  }
  std::string fresh(std::string hint="tensor") { return id(hint)+"_"+std::to_string(serial++); }
  E assign(E value, const std::string& hint="tensor",bool named=false) {
    Statement statement;
    statement.name=fresh(hint);
    statement.value=std::move(value);
    statement.named=named;
    if(named) statement.sourceName=hint;
    if(named && statement.value.kind!=E::Array && tensorBindings.count(base(hint))) statement.tensorDeclaration=tensorBindings.at(base(hint));
    if(!statement.tensorDeclaration.empty()) declaredSource(statement.value);
    E result=local(statement.name);
    result.layout=statement.value.layout;
    body->push_back(std::move(statement));
    return result;
  }
  void unique(std::vector<std::string>& list,const std::string& name) {
    if (std::find(list.begin(),list.end(),name)==list.end()) list.push_back(name);
  }
  std::string persistentField(const std::string& name) {
    const std::string field="state_"+(stateOwner.empty()?"root":stateOwner)+"_"+id(name);
    if(stateLayouts.count(field) && stateLayouts.at(field)!=stateRepetitions)
      throw std::runtime_error("persistent scope repetition mismatch: "+field);
    stateLayouts[field]=stateRepetitions;
    unique(module.states,field);
    return field;
  }
  E reference(const std::string& name) {
    if(weightName(name)) {
      if(scalarBindings.count(base(name)) || module.scalars.count(base(name)))
        throw std::runtime_error("scalar cannot be a tensor/weight group member: "+name);
      return weightValue(name);
    }
    const std::string key=tensorKey(name);
    const size_t dot=key.find('.');
    if(dot!=std::string::npos && !symbols.count(key)) {
      const std::string root=id(key.substr(0,dot)), field=id(key.substr(dot+1));
      if(currentFunction && std::find(currentFunction->inputs.begin(),currentFunction->inputs.end(),root)!=currentFunction->inputs.end()) {
        const std::string record=currentFunction->name+"_"+root+"Input";
        currentFunction->inputTypes[root]=record;
        symbols[root]=local(root);
        unique(module.inputRecords[record],field);
        if(tensorBindings.count(key)) currentFunction->inputDeclarations[root+"."+field]=tensorBindings.at(key);
        symbols[key]=expr(E::Member,{local(root)},field);
      }
    }
    E v;
    {
      std::map<std::string,E>::const_iterator it=symbols.find(key);
      if(name.find('!')!=std::string::npos) {
        const std::string field=persistentField(name);
        v=expr(E::StateRead,{statePath},field);
        symbols[key]=v;
      } else {
        if (it==symbols.end()) throw std::runtime_error("undefined tensor "+name);
        v=it->second;
      }
    }
    if(name.find("[]")!=std::string::npos && v.kind!=E::Array) throw std::runtime_error("indexed tensor is not an array: "+name);
    if(name.find("[]")!=std::string::npos && v.kind==E::Array) {
      const int depth=arrayDepth(v); const auto axes=v.splitAxes;
      if(!repetitionCount.empty()) v=expr(E::Array,{expr(E::ValidateArray,{v,local(repetitionCount)})},std::to_string(depth));
      v=expr(E::ArrayAt,{v,local(index)});
      if(depth>1) { v=expr(E::Array,{v},std::to_string(depth-1)); if(!axes.empty()) v.splitAxes.assign(axes.begin()+1,axes.end()); }
    }
    return name.back()=='^' ? transposed(v) : v;
  }
  Values read(Tensor* t) {
    Values r;
    for(const std::string& n:names(t)) {
      E value=reference(n);
      if(t->arrayCounts_.count(n)) {
        const std::string count=card(t->arrayCounts_.at(n));
        const int size=std::stoi(module.scalars.count(count)?module.scalars.at(count):count);
        if(weightName(n)) value=weightArray(n,count);
        else if(value.kind==E::Array) {
          const auto axes=value.splitAxes;
          value=expr(E::Array,{expr(E::ValidateArray,{value,local(count)})},std::to_string(arrayDepth(value)));
          value.splitAxes=axes;
        } else { value=expr(E::Array,Values(size,value)); symbols[tensorKey(n)]=value; }
      }
      if(value.kind==E::Group && value.name=="literal") r.insert(r.end(),value.args.begin(),value.args.end());
      else r.push_back(value);
    }
    return r;
  }
  Values transfer(Values input, TransferOp* op) {
    if (!op) return input;
    if (op->transferQualifier_ && op->transferQualifier_->card_) {
      const std::string count=card(op->transferQualifier_->card_);
      const int n=std::stoi(module.scalars.count(count)?module.scalars.at(count):count);
      Values arrays;
      for(const E& value:input) {
        if(value.kind==E::Array) throw std::runtime_error("split transfer requires tensors, not arrays");
        Values elements;
        for(int i=0;i<n;++i) elements.push_back(expr(E::Split,{value,local(count),local(std::to_string(i))}));
        arrays.push_back(expr(E::Array,elements));
      }
      return arrays;
    }
    if(op->strings_.count("+")) {
      Values result;
      for(const E& value:input) {
        if(value.kind!=E::Array || arrayDepth(value)!=1) throw std::runtime_error("array reduction requires a single layer array");
        result.push_back(assign(expr(E::Merge,{value},"sum"),"reduced"));
      }
      return result;
    }
    if(op->strings_.count("[")) {
      Values result;
      for(const E& value:input) {
        if(value.kind!=E::Array) throw std::runtime_error("array concatenation requires an array");
        result.push_back(assemble(value));
      }
      return result;
    }
    if(op->strings_.count(":")) return {concatenateGroup(input)};
    return input;
  }
  E concatenateGroup(const Values& input) {
    for(const E& value:input) if(value.kind==E::Array) {
      return mapArray(value,[&](const E&,const E& indexValue) {
        Values members;
        for(const E& other:input) {
          if(other.kind!=E::Array || arrayDepth(other)!=arrayDepth(value) || other.splitAxes!=value.splitAxes) throw std::runtime_error("group concatenation requires matching array layouts");
          E member=expr(E::GroupMember,{other,expr(E::ArraySize,{value}),indexValue});
          if(arrayDepth(other)>1) { member=expr(E::Array,{member},std::to_string(arrayDepth(other)-1)); if(!other.splitAxes.empty()) member.splitAxes.assign(other.splitAxes.begin()+1,other.splitAxes.end()); }
          members.push_back(member);
        }
        return concatenateGroup(members);
      });
    }
    return assign(expr(E::Merge,input,"group"),"concatenated_group");
  }

  Values receive(Tensor* t,Values input, bool corresponding) {
    const std::vector<std::string> ns=names(t);
    if(t->arrayCount_ && input.size()==1 && input.front().kind==E::Array) {
      input={expr(E::Array,{expr(E::ValidateArray,{input.front(),local(card(t->arrayCount_))})},std::to_string(arrayDepth(input.front())))};
    }
    if(corresponding && ns.size()>1 && input.size()==1 && input.front().kind==E::Local) {
      for(Statement& producer:*body) if(producer.name==input.front().name && producer.value.kind==E::Opaque && inferredOpaque.count(producer.value.name)) {
        Function& signature=module.opaqueSignatures.at(producer.value.name);
        if(signature.outputs.size()!=1 && signature.outputs.size()!=ns.size()) throw std::runtime_error("abstract output cardinality mismatch");
        signature.outputs.clear();
        const E call=input.front(); input.clear();
        for(const std::string& name:ns) { signature.outputs.push_back(id(name)); input.push_back(expr(E::Member,{call},id(name))); }
        break;
      }
    }
    if(corresponding && ns.size()>1 && input.size()==1 && input.front().kind==E::Array)
      throw std::runtime_error("cannot transfer an array to a group");
    if(corresponding && input.size()==1 && input.front().kind==E::Group) {
      const E group=input.front().args.front();
      input.clear();
      for(size_t i=0;i<ns.size();++i)
        input.push_back(expr(E::GroupMember,{group,local(std::to_string(ns.size())),local(std::to_string(i))}));
    }
    if(corresponding && ns.size()==1 && input.size()>1) {
      const std::string root=tensorKey(ns.front());
      const std::map<std::string,E> prior=symbols;
      for(const E& member:input) for(const std::pair<const std::string,E>& item:prior)
        if(member.kind==item.second.kind && member.name==item.second.name && !member.name.empty()) symbols[root+"."+item.first]=member;
      symbols[root]=expr(E::Group,input,"literal"); return input;
    }
    if (corresponding && input.size()!=ns.size())
      throw std::runtime_error("corresponding transfer cardinality mismatch at "+ns.front());
    Values r;
    for(size_t i=0;i<ns.size();++i) {
      E v=corresponding && input.size()==ns.size() ? input[i] : merged(input);
      if(corresponding && input.size()==1 && ns.size()>1)
        v=expr(E::Split,{v,local(std::to_string(ns.size())),local(std::to_string(i))});
      if(weightName(ns[i])) {
        if(t->arrayCounts_.count(ns[i])) throw std::runtime_error("weight array destinations require an explicit repeated scope");
        E weight=weightValue(ns[i]);
        if(weight.kind!=E::Weight) throw std::runtime_error("weight destination must be a declared weight");
        if(!weight.layout.logicalShape.empty()) v=declaredTensor(v,module.weightDeclarations.at(weight.name),true);
        E update=expr(E::WeightWrite,{weight.args.front(),v},weight.name); update.layout=weight.layout;
        r.push_back(assign(std::move(update),"updated_weight"));
        continue;
      }
      if(t->arrayCounts_.count(ns[i])) {
        if(v.kind!=E::Array) throw std::runtime_error("array destination requires an array");
        E checked=expr(E::Array,{expr(E::ValidateArray,{v,local(card(t->arrayCounts_.at(ns[i])))})},std::to_string(arrayDepth(v))); checked.splitAxes=v.splitAxes; v=checked;
      }
      if(tensorBindings.count(base(ns[i]))) {
        const std::string declaration=tensorBindings.at(base(ns[i]));
        if(declarationArrays.count(declaration)) v=declaredArray(v,declaration);
        else if(v.kind!=E::Array) v=declaredTensor(v,declaration);
      }
      if(ns[i].find('!')!=std::string::npos) {
        const std::string field=persistentField(ns[i]);
        v=assign(expr(E::StateWrite,{statePath,v},field),base(ns[i])+"_saved",true);
      } else if(v.kind==E::Array) {
        const int depth=arrayDepth(v); const auto axes=v.splitAxes;
        v=expr(E::Array,{assign(v,base(ns[i]),true)},std::to_string(depth)); v.splitAxes=axes;
      } else v=assign(v,base(ns[i]),true);
      symbols[tensorKey(ns[i])]=v; r.push_back(v);
    }
    return r;
  }
  std::set<std::string> inferredOpaque;
  Values invoke(const std::string& name,Values input) {
    if(!name.empty() && name.back()=='^') {
      Values result=invoke(name.substr(0,name.size()-1),input);
      for(E& value:result) value=transposed(value);
      return result;
    }
    const std::string key=base(name);
    std::map<std::string,Function>::const_iterator d=definitions.find(key);
    auto array=std::find_if(input.begin(),input.end(),[](const E& value){return value.kind==E::Array;});
    if(array!=input.end() && (d==definitions.end() || d->second.inputTypes.empty())) {
      return {mapArray(*array,[&](const E& element,const E& at) {
        Values arguments;
        for(const E& value:input) {
          if(value.kind!=E::Array) { arguments.push_back(value); continue; }
          if(value.splitAxes!=array->splitAxes || arrayDepth(value)!=arrayDepth(*array)) throw std::runtime_error("computation input array layouts must match");
          E member=expr(E::GroupMember,{value,expr(E::ArraySize,{*array}),at});
          if(arrayDepth(value)>1) { member=expr(E::Array,{member},std::to_string(arrayDepth(value)-1)); member.splitAxes=element.splitAxes; }
          arguments.push_back(member);
        }
        const std::string savedIndex=index; const E savedPath=instancePath;
        if(name.find("[]")!=std::string::npos) { index=at.name; instancePath=expr(E::IndexedPath,{savedPath,at}); }
        Values output=invoke(name,arguments);
        index=savedIndex; instancePath=savedPath;
        if(output.size()!=1) throw std::runtime_error("array element computation must return one value");
        return output.front();
      })};
    }
    static const std::map<std::string,std::string> builtins={
      {"SUM_LAST","sum_last"},{"MEAN_LAST","mean_last"},{"MAX_LAST","max_last"},
      {"SOFTMAX_LAST","softmax_last"},{"LAYERNORM_LAST","layernorm_last"},{"RMSNORM_LAST","rmsnorm_last"},
      {"MUL_ELEMENTS","mul"},{"DIV_ELEMENTS","div"}};
    if(d==definitions.end() && !module.opaqueSignatures.count(id(key)) && builtins.count(key)) {
      const std::string op=builtins.at(key);
      const bool binary=op=="mul" || op=="div";
      if(input.size()!=(binary?2u:1u)) throw std::runtime_error("builtin input cardinality mismatch: "+key);
      E value=expr(E::TensorOp,input,op);
      value.layout=binary?elementwiseLayout(module,input[0].layout,input[1].layout):input[0].layout;
      if(!value.layout.logicalShape.empty() && (op=="sum_last" || op=="mean_last" || op=="max_last")) {
        value.layout.logicalShape.back()="1"; value.layout.splitCounts.back()=1;
      }
      return {assign(value,key)};
    }
    if (d==definitions.end()) {
      if(!module.opaqueSignatures.count(id(key))) throw std::runtime_error("undefined computation "+key);
      const std::string op=id(key); unique(module.operations,op);
      const std::string site=owner+"_"+op;
      const std::string occurrence=site+"_"+std::to_string(opaqueOccurrences[site]++);
      const E operationPath=name.find("[]")!=std::string::npos ? instancePath : weightPath;
      // Declared parameters fix the inputs. With none, the incoming flow value is not
      // passed and the result takes its shape from a declared destination.
      const size_t parameters=module.opaqueSignatures.at(op).inputs.size();
      Values args;
      if(parameters) {
        if(input.size()!=parameters)
          throw std::runtime_error("abstract computation "+key+" takes "+std::to_string(parameters)+
            " input"+(parameters==1?"":"s")+" but receives "+std::to_string(input.size()));
        args=parameters==1 ? Values{merged(input)} : input;
      }
      args.push_back(operationPath); args.push_back(expr(E::StringLiteral,{},occurrence));
      E call=assign(expr(E::Opaque,args,op),key);
      if(!parameters) undeclaredSources[call.name]=key;
      if(!module.opaqueSignatures.count(op) || module.opaqueSignatures.at(op).outputs.size()==1) return {call};
      Values result;
      for(const std::string& field:module.opaqueSignatures.at(op).outputs) result.push_back(expr(E::Member,{call},field));
      return result;
    }
    if(d->second.inputs.size()==1 && d->second.inputTypes.count(d->second.inputs.front())) {
      const std::string type=d->second.inputTypes.at(d->second.inputs.front());
      if(module.inputRecords.count(type) && input.size()>1) {
        if(module.inputRecords.at(type).size()!=input.size()) throw std::runtime_error("group computation input cardinality mismatch: "+key);
        input={expr(E::Record,input,type)};
      }
    }
    if(d->second.consumesInput && d->second.inputs.size()==1 && input.size()>1) {
      Values result;
      for(const E& member:input) { Values one=invoke(name,{member}); result.insert(result.end(),one.begin(),one.end()); }
      return result;
    }
    Values args=input;
    if(args.size()==1 && args.front().kind==E::Group) {
      const E group=args.front().args.front(); args.clear();
      for(size_t i=0;i<d->second.inputs.size();++i)
        args.push_back(expr(E::GroupMember,{group,local(std::to_string(d->second.inputs.size())),local(std::to_string(i))}));
    }
    for(size_t i=0;i<args.size() && i<d->second.inputs.size();++i) {
      const std::string& formal=d->second.inputs[i];
      if(!d->second.inputTypes.count(formal)) continue;
      for(const std::string& actual:currentFunction->inputs) {
        const bool same=args[i].kind==E::Local && (args[i].name==actual ||
          (symbols.count(actual) && symbols.at(actual).kind==E::Local && symbols.at(actual).name==args[i].name));
        if(!same) continue;
        currentFunction->inputTypes[actual]=d->second.inputTypes.at(formal);
        for(const std::pair<const std::string,std::string>& declaration:d->second.inputDeclarations)
          if(declaration.first.find(formal+".")==0) currentFunction->inputDeclarations[actual+declaration.first.substr(formal.size())]=declaration.second;
      }
    }
    if(args.size()!=d->second.inputs.size()) {
      if(d->second.inputs.size()!=1) throw std::runtime_error("call input cardinality mismatch: "+key);
      args={merged(input)};
    }
    args.push_back(name.find("[]")!=std::string::npos ? instancePath : weightPath);
    if(d->second.needsInstance) args.push_back(local(index));
    args.push_back(statePath);
    E call=assign(expr(E::Call,args,id(key)),key);
    if(d->second.arrayDepth) { E result=expr(E::Array,{call},std::to_string(d->second.arrayDepth)); if(!d->second.outputSplitAxes.empty()) result.splitAxes=d->second.outputSplitAxes.front(); return {result}; }
    if(d->second.returnsGroup) return {expr(E::Group,{call})};
    if(d->second.outputs.size()==1) return {call};
    Values result;
    for(size_t i=0;i<d->second.outputs.size();++i) {
      E value=expr(E::Member,{call},d->second.outputs[i]);
      if(i<d->second.outputArrayDepths.size() && d->second.outputArrayDepths[i]) value=expr(E::Array,{value},std::to_string(d->second.outputArrayDepths[i]));
      result.push_back(value);
    }
    return result;
  }
  E weightArray(const std::string& w,const std::string& count) {
    const std::string declaration=weightBindings.count(base(w))?weightBindings.at(base(w)):"";
    auto resolved=[&](const std::string& size){return module.scalars.count(size)?module.scalars.at(size):size;};
    if(declarationArrays.count(declaration) && resolved(declarationArrays.at(declaration))!=resolved(count))
      throw std::runtime_error("weight array count conflicts with declaration: "+w);
    weightArrays[base(w)]=count;
    Values elements; const E saved=weightPath;
    for(int j=0;j<std::stoi(resolved(count));++j) { weightPath=expr(E::IndexedPath,{saved,local(std::to_string(j))}); elements.push_back(weightValue(w)); }
    weightPath=saved; return expr(E::Array,elements);
  }
  E weightValue(const std::string& w) {
    if(scalarBindings.count(base(w))) return local(scalarBindings.at(base(w)));
    if(module.scalars.count(id(w))) return local(id(w));
    // Weights are always persistent; ! marks persistent tensors only.
    if(w.find('!')!=std::string::npos) throw std::runtime_error("weights are always persistent; remove ! from weight "+w);
    const std::string field=weightBindings.count(base(w))?"weight_"+weightBindings.at(base(w)):owner+"_"+id(w); unique(module.weights,field);
    if(weightBindings.count(base(w))) module.weightDeclarations[field]=weightBindings.at(base(w));
    const bool indexed=w.find("[]")!=std::string::npos;
    const std::string declaration=weightBindings.count(base(w))?weightBindings.at(base(w)):"";
    const std::string arrayCount=declarationArrays.count(declaration)?declarationArrays.at(declaration):weightArrays.count(base(w))?weightArrays.at(base(w)):"";
    if(indexed && !arrayCount.empty() && !repetitionCount.empty()) {
      const std::string count=arrayCount;
      if((module.scalars.count(count)?module.scalars.at(count):count)!=(module.scalars.count(repetitionCount)?module.scalars.at(repetitionCount):repetitionCount)) throw std::runtime_error("weight array count does not match repetition: "+w);
    }
    // Only array weights differ between instances: w[] selects the current element and
    // an unindexed array weight takes the instance of the enclosing indexed call. Any
    // other weight has one value wherever it is used.
    const E path=indexed ? instancePath : !arrayCount.empty() ? weightPath : expr(E::IndexedPath,{},"root");
    E weight=expr(E::Weight,{path},field);
    if(weightBindings.count(base(w))) weight=declaredTensor(weight,weightBindings.at(base(w)),true);
    if(w.back()=='^') weight=transposed(weight);
    return weight;
  }
  template<class Apply> E mapArray(const E& array,Apply apply) {
    const int depth=arrayDepth(array);
    const E size=assign(expr(E::ArraySize,{array}),"array_size");
    Statement loop; loop.kind=Statement::Loop; loop.name=fresh("element"); loop.count=size.name;
    Statement init; init.name=fresh("array_result"); init.group=true;
    std::vector<Statement>* outer=body; body=&loop.body;
    E element=expr(E::ArrayAt,{array,local(loop.name)});
    if(depth>1) { element=expr(E::Array,{element},std::to_string(depth-1)); if(!array.splitAxes.empty()) element.splitAxes.assign(array.splitAxes.begin()+1,array.splitAxes.end()); }
    const E value=apply(element,local(loop.name));
    init.arrayDepth=1+arrayDepth(value);
    Statement append; append.kind=Statement::Append; append.name=init.name; append.value=value; body->push_back(append);
    body=outer; body->push_back(init); body->push_back(loop);
    E result=expr(E::Array,{local(init.name)},std::to_string(init.arrayDepth));
    if(!array.splitAxes.empty()) result.splitAxes.push_back(array.splitAxes.front());
    result.splitAxes.insert(result.splitAxes.end(),value.splitAxes.begin(),value.splitAxes.end()); return result;
  }
  E binary(const std::string& op,const E& left,const E& right) {
    if(left.kind==E::Array || right.kind==E::Array) {
      const E& array=left.kind==E::Array?left:right;
      return mapArray(array,[&](const E& member,const E& indexValue) {
        E lhs=left, rhs=right;
        if(left.kind==E::Array) lhs=member;
        if(right.kind==E::Array) {
          rhs=left.kind==E::Array?expr(E::GroupMember,{right,expr(E::ArraySize,{left}),indexValue}):member;
          if(left.kind==E::Array && arrayDepth(right)>1) rhs=expr(E::Array,{rhs},std::to_string(arrayDepth(right)-1));
        }
        return binary(op,lhs,rhs);
      });
    }
    const E::Kind kind=op=="*"?(right.kind==E::Local && module.scalars.count(right.name)?E::Scale:E::Matmul):op=="+"?E::Add:E::Subtract;
    E value=expr(kind,{left,right});
    if(kind==E::Matmul) value.layout=matmulLayout(module,left.layout,right.layout);
    else if(kind==E::Scale) value.layout=left.layout;
    else value.layout=elementwiseLayout(module,left.layout,right.layout);
    return assign(value,op=="*"?"product":"sum");
  }
  Values computationExpression(CompExpression* e,Values input) {
    std::string op=e->first_ ? *e->first_->strings_.begin() : "";
    Values current=input;
    for(size_t i=0;i<e->arg_.size();++i) {
      Arg* a=e->arg_[i]; Values rhs;
      if(a->scope_) rhs=scope(a->scope_,current);
      else if(a->tensor_) {
        const std::vector<std::string> ns=names(a->tensor_);
        for(size_t n=0;n<ns.size();++n) {
          if(n<a->calls_.size() && a->calls_[n]) {
            Values outputs=invoke(ns[n],input);
            rhs.insert(rhs.end(),outputs.begin(),outputs.end());
          } else rhs.push_back(reference(ns[n]));
        }
      } else {
        for(const std::string& w:a->weight_->weightEnum_->lowerName_) {
          if(a->weight_->arrayCounts_.count(w)) {
            rhs.push_back(weightArray(w,card(a->weight_->arrayCounts_.at(w))));
          } else rhs.push_back(weightValue(w));
        }
      }
      if(op.empty()) current=rhs;
      else {
        if(op!="*" && op!="+" && op!="-") throw std::runtime_error("unsupported computation operator "+op);
        Values result;
        if(current.size()!=1 && rhs.size()!=1 && current.size()!=rhs.size())
          throw std::runtime_error("arithmetic group cardinality mismatch");
        for(size_t j=0;j<std::max(current.size(),rhs.size());++j)
          result.push_back(binary(op,current[current.size()==1?0:j],rhs[rhs.size()==1?0:j]));
        current=result;
      }
      if(i<e->compOp_.size()) op=*e->compOp_[i]->strings_.begin();
    }
    return current;
  }
  template<class Body> Values repeat(Repetition* rep,Values input,Body generate,bool scopeInstances=false,bool namedInstances=false,bool namedFlow=false) {
    if(!rep) return generate(input);
    Statement loop; loop.kind=Statement::Loop; loop.name=fresh("i"); if(rep->card_) loop.count=card(rep->card_);
    else {
      auto array=std::find_if(input.begin(),input.end(),[](const E& value){return value.kind==E::Array;});
      if(array==input.end()) throw std::runtime_error("inferred repetition requires an input array");
      loop.count=assign(expr(E::ArraySize,{*array}),"iteration_count").name;
    }
    loop.reverse=rep->repOp_->strings_.count("\\")!=0;
    bool peer=rep->repType_!=nullptr;
    if(peer) {
      const E* layout=nullptr;
      for(const E& value:input) if(value.kind==E::Array) {
        if(layout && (arrayDepth(value)!=arrayDepth(*layout) || value.splitAxes!=layout->splitAxes)) throw std::runtime_error("scope input array layouts must match");
        layout=&value;
      }
    }
    E carry; std::vector<int> peerAxes;
    const std::map<std::string,E> enclosingSymbols=symbols;
    std::map<std::string,E> namedCarries;
    if(namedFlow) for(const std::pair<const std::string,E>& item:enclosingSymbols) {
      if(item.second.kind==E::Local) namedCarries[item.first]=assign(item.second,"named_state");
    }
    for(const std::pair<const std::string,E>& item:namedCarries) symbols[item.first]=item.second;
    Values carries;
    if(peer) {
      carry=local(fresh("outputs")); Statement init; init.name=carry.name; init.group=true; body->push_back(init);
    } else for(const E& value:input) carries.push_back(assign(value,"state"));
    std::vector<Statement>* outer=body; std::map<std::string,E> savedSymbols=symbols;
    const std::string savedIndex=index, savedRepetition=repetitionCount;
    repetitionCount=loop.count;
    const E savedInstancePath=instancePath, savedStatePath=statePath;
    index=loop.name; body=&loop.body;
    if(scopeInstances)
      instancePath=assign(expr(E::IndexedPath,{savedInstancePath,local(index)}),"scope_instance_path");
    if(namedInstances) {
      statePath=assign(expr(E::IndexedPath,{savedStatePath,local(index)}),"scope_state_path");
      stateRepetitions.push_back(module.scalars.count(loop.count)?module.scalars.at(loop.count):loop.count);
    }
    Values iteration=peer ? input : carries;
    if(peer) for(E& value:iteration) if(value.kind==E::Array) {
      const int depth=arrayDepth(value); const auto axes=value.splitAxes;
      value=expr(E::ArrayAt,{expr(E::ValidateArray,{value,local(loop.count)}),local(index)});
      if(depth>1) { value=expr(E::Array,{value},std::to_string(depth-1)); if(!axes.empty()) value.splitAxes.assign(axes.begin()+1,axes.end()); }
    }
    Values result=generate(iteration);
    if(peer) {
      if(result.size()!=1) throw std::runtime_error("peer scopes must produce a tensor or array, not a group");
      loop.arrayDepth=1+arrayDepth(result.front());
      for(const E& source:input) if(!source.splitAxes.empty()) { peerAxes.push_back(source.splitAxes.front()); break; }
      peerAxes.insert(peerAxes.end(),result.front().splitAxes.begin(),result.front().splitAxes.end());
      Statement end; end.kind=Statement::Append; end.name=carry.name; end.value=result.front(); end.reverse=loop.reverse; body->push_back(end);
    } else if(namedFlow) {
      Values outputs;
      for(const E& value:result) {
        body=outer;
        E output=assign(input.front(),"scope_result");
        body=&loop.body;
        Statement save; save.name=output.name; save.value=value; body->push_back(save);
        outputs.push_back(output);
      }
      carries=outputs;
      Values next;
      for(const std::pair<const std::string,E>& item:namedCarries) next.push_back(assign(symbols.at(item.first),"next_named_state"));
      size_t n=0;
      for(const std::pair<const std::string,E>& item:namedCarries) {
        Statement save; save.name=item.second.name; save.value=next[n++]; body->push_back(save);
      }
    } else {
      if(carries.size()==1) result={merged(result)};
      if(result.size()!=carries.size()) throw std::runtime_error("sequential scope changes tensor-group cardinality");
      // Capture all outputs before updating any loop-carried input.
      Values next;
      for(const E& value:result) next.push_back(assign(value,"next_state"));
      for(size_t i=0;i<carries.size();++i) {
        Statement end; end.name=carries[i].name; end.value=next[i]; body->push_back(end);
      }
    }
    body=outer; symbols=savedSymbols; index=savedIndex; repetitionCount=savedRepetition;
    if(namedFlow) {
      symbols=enclosingSymbols;
      for(const std::pair<const std::string,E>& item:namedCarries) symbols[item.first]=item.second;
    }
    instancePath=savedInstancePath; statePath=savedStatePath;
    if(namedInstances) stateRepetitions.pop_back();
    if(peer) for(Statement& init:*body) if(init.name==carry.name) init.arrayDepth=loop.arrayDepth;
    const int outputDepth=loop.arrayDepth;
    body->push_back(std::move(loop));
    if(peer) {
      E result=expr(E::Array,{carry},std::to_string(outputDepth)); result.splitAxes=peerAxes; return {result};
    }
    return carries;
  }
  Values computation(Computation* c,Values input) {
    const std::string savedOwner=owner, savedStateOwner=stateOwner;
    const E savedSharedPath=sharedPath;
    if(c->upperName_.find('!')!=std::string::npos) sharedPath=instancePath;
    if(!c->upperName_.empty()) {
      owner+="_"+id(c->upperName_);
      stateOwner+=(stateOwner.empty()?"":"_")+id(c->upperName_);
    }
    Values result=repeat(c->repetition_,input,[&](Values v){
      for(CompStmt* s:c->compBody_->compStmt_)
        for(size_t i=0;i<s->compExpression_.size();++i) {
          if(i) v=transfer(v,s->transferOp_[i-1]);
          v=computationExpression(s->compExpression_[i],v);
        }
      return v;
    },false,!c->upperName_.empty());
    owner=savedOwner; stateOwner=savedStateOwner;
    sharedPath=savedSharedPath;
    return result;
  }
  Values scope(Scope* s,Values input,bool standalone=false) {
    const std::string choiceSelector=s->choiceSel_;
    if(s->scopeBody_->scopeStmt_.size()==1) {
      ScopeStmt* st=s->scopeBody_->scopeStmt_.front();
      const bool choice=std::any_of(st->scopeSep_.begin(),st->scopeSep_.end(),[](ScopeSep* sep){return !sep->transferOp_;});
      if(choice) {
        if(choiceSelector.empty()) throw std::runtime_error("scope alternatives require a selector");
        const std::string selector=id(choiceSelector);
        unique(module.selectors,selector);
        if(module.opaqueSignatures.count(selector) && module.opaqueSignatures.at(selector).inputs.size()!=1)
          throw std::runtime_error("selector "+choiceSelector+" receives the scope's input value and needs one parameter: "+selector+"(A) {}");
        E result=local(fresh("choice_outputs"));
        Statement init; init.name=result.name; init.group=true; body->push_back(init);
        Statement branch; branch.exclusive=true; branch.kind=Statement::Branch; branch.name=fresh("choice");
        std::vector<Statement>* parent=body;
        const std::map<std::string,E> savedSymbols=symbols;
        std::vector<std::vector<ScopeExpression*>> branches(1);
        std::vector<std::vector<TransferOp*>> transfers(1);
        for(size_t i=0;i<st->scopeExpression_.size();++i) {
          if(i) {
            if(st->scopeSep_[i-1]->transferOp_) transfers.back().push_back(st->scopeSep_[i-1]->transferOp_);
            else { branches.emplace_back(); transfers.emplace_back(); }
          }
          branches.back().push_back(st->scopeExpression_[i]);
        }
        size_t outputCount=0;
        for(size_t alternative=0;alternative<branches.size();++alternative) {
          Values arguments={local(std::to_string(branch.predicates.size())),merged(input)};
          if(choiceSelector.find("[]")!=std::string::npos) {
            arguments.push_back(instancePath); arguments.push_back(local(index));
          }
          branch.predicates.push_back(expr(E::Predicate,arguments,selector));
          branch.alternatives.emplace_back(); body=&branch.alternatives.back(); symbols=savedSymbols;
          Values output=statement(branches[alternative],transfers[alternative],input,false);
          if(alternative && output.size()!=outputCount) throw std::runtime_error("enumerated choices must return matching group cardinalities");
          outputCount=output.size();
          for(const E& value:output) {
            Statement save; save.kind=value.kind==E::Group?Statement::Extend:Statement::Append;
            save.name=result.name; save.value=value; body->push_back(save);
          }
        }
        body=parent; symbols=savedSymbols; body->push_back(std::move(branch));
        Values selected;
        for(size_t i=0;i<outputCount;++i) selected.push_back(assign(expr(E::GroupMember,{result,local(std::to_string(outputCount)),local(std::to_string(i))}),"selected"));
        return selected;
      }
    }
    const std::map<std::string,std::string> savedScalars=scalarBindings, savedTensors=tensorBindings, savedWeights=weightBindings;
    std::map<std::string,E> savedSymbols=symbols; const std::string savedOwner=owner, savedStateOwner=stateOwner;
    const E savedSharedPath=sharedPath;
    if(s->scopeName_.find('!')!=std::string::npos) sharedPath=instancePath;
    if(!s->scopeName_.empty()) {
      owner+="_"+id(s->scopeName_);
      stateOwner+=(stateOwner.empty()?"":"_")+id(s->scopeName_);
    }
    bool namedFlow=standalone && (!s->repetition_ || !s->repetition_->repType_);
    if(namedFlow) {
      ScopeStmt* firstStatement=s->scopeBody_->scopeStmt_.front();
      Tensor* firstTensor=nullptr;
      if(!firstStatement->scopeExpression_.empty()) {
        ScopeExpression* firstExpression=firstStatement->scopeExpression_.front();
        firstTensor=firstExpression->tensor_;
        if(!firstTensor && firstExpression->computation_) {
          CompExpression* expression=firstExpression->computation_->compBody_->compStmt_.front()->compExpression_.front();
          Arg* argument=expression->arg_.front();
          if(!expression->first_ && std::none_of(argument->calls_.begin(),argument->calls_.end(),[](bool call){return call;}))
            firstTensor=argument->tensor_;
        }
      }
      if(!firstTensor) namedFlow=false;
      else for(const std::string& name:names(firstTensor))
        if(!weightName(name) && !symbols.count(tensorKey(name))) namedFlow=false;
    }
    Values result=repeat(s->repetition_,input,[&](Values v){
      bool first=!namedFlow;
      for(ScopeStmt* st:s->scopeBody_->scopeStmt_) {
        if(st->lowerDef_) {
          if(isCostField(st->lowerDef_->lowerName_)) throw std::runtime_error("cost parameters cannot appear in executable scopes");
          const std::string name=base(st->lowerDef_->lowerName_), qualified=fresh(owner+"_"+name);
          if(isScalar(*st->lowerDef_)) { scalarBindings[name]=qualified; module.scalars[qualified]=st->lowerDef_->shapeDef_->card_.front()->number_; }
          else { requireTypedShape(*st->lowerDef_->shapeDef_,name); weightBindings[name]=qualified; for(Card* c:st->lowerDef_->shapeDef_->card_) module.weightShapes[qualified].push_back(card(c)); registerArray(st->lowerDef_->arrayCount_,qualified); registerSplits(st->lowerDef_->shapeDef_,qualified); module.weightDataSizes[qualified]=st->lowerDef_->shapeDef_->lowerName_; }
          continue;
        }
        if(st->upperDef_) {
          const std::string name=base(st->upperDef_->upperName_), qualified=fresh(owner+"_"+name);
          requireTypedShape(*st->upperDef_->shapeDef_,name); tensorBindings[name]=qualified;
          for(Card* c:st->upperDef_->shapeDef_->card_) module.tensorShapes[qualified].push_back(card(c));
          registerArray(st->upperDef_->arrayCount_,qualified); registerSplits(st->upperDef_->shapeDef_,qualified); module.tensorDataSizes[qualified]=st->upperDef_->shapeDef_->lowerName_; continue;
        }
        std::vector<TransferOp*> ops;
        for(ScopeSep* sep:st->scopeSep_) {
          if(!sep->transferOp_) throw std::runtime_error("explicit scope alternatives require a selector layout");
          ops.push_back(sep->transferOp_);
        }
        v=statement(st->scopeExpression_,ops,v,first); first=false;
      }
      return v;
    },true,!s->scopeName_.empty(),namedFlow);
    if(namedFlow) for(std::pair<const std::string,E>& item:savedSymbols) if(symbols.count(item.first)) item.second=symbols.at(item.first);
    symbols=savedSymbols; owner=savedOwner; stateOwner=savedStateOwner; sharedPath=savedSharedPath;
    scalarBindings=savedScalars; tensorBindings=savedTensors; weightBindings=savedWeights; return result;
  }
  Values statement(const std::vector<ScopeExpression*>& expressions,const std::vector<TransferOp*>& ops,Values v,bool entry) {
    for(size_t i=0;i<expressions.size();++i) {
      ScopeExpression* e=expressions[i];
      if(i) v=transfer(v,ops[i-1]);
      if(e->tensor_) {
        const std::vector<std::string> tensorNames=names(e->tensor_);
        const bool persistentInput=std::any_of(tensorNames.begin(),tensorNames.end(),[](const std::string& n){return n.find('!')!=std::string::npos;});
        const bool indexedInput=std::all_of(tensorNames.begin(),tensorNames.end(),[&](const std::string& n){return n.find("[]")!=std::string::npos;});
        const bool weightInput=std::any_of(tensorNames.begin(),tensorNames.end(),weightName);
        if(i==0 && entry && (persistentInput || weightInput)) {
          size_t inputIndex=0;
          for(const std::string& name:tensorNames) if(!weightName(name) && name.find('!')==std::string::npos) {
            if(inputIndex>=v.size()) throw std::runtime_error("persistent input cardinality mismatch");
            symbols[tensorKey(name)]=v[inputIndex++];
          }
          v=read(e->tensor_);
        } else if(i==0 && (!entry || indexedInput)) v=read(e->tensor_);
        else v=receive(e->tensor_,v,true);
      } else if(e->computation_) v=computation(e->computation_,v);
      else if(!e->lowerName_.empty()) {
        E weight=weightValue(e->lowerName_);
        if(weight.kind!=E::Weight) throw std::runtime_error("weight destination must be a declared weight");
        if(i) {
          E value=merged(v);
          if(!weight.layout.logicalShape.empty())
            value=declaredTensor(std::move(value),module.weightDeclarations.at(weight.name),true);
          E update=expr(E::WeightWrite,{weight.args.front(),value},weight.name);
          update.layout=weight.layout;
          v={assign(std::move(update),"updated_weight")};
        }
        else v={weight};
      } else v=scope(e->scope_,v,i==0);
    }
    return v;
  }
public:
  Module run(SetepFile& file) {
    for(LowerDef* d:file.lowerDef_) if(isScalar(*d)) module.scalars[id(d->lowerName_)]=d->shapeDef_->card_.front()->number_;
    for(LowerDef* d:file.lowerDef_) if(!isScalar(*d)) {
      requireTypedShape(*d->shapeDef_,d->lowerName_);
      const std::string name=id(d->lowerName_);
      for(Card* dimension:d->shapeDef_->card_) module.weightShapes[name].push_back(card(dimension));
      registerArray(d->arrayCount_,name);
      registerSplits(d->shapeDef_,name); module.weightDataSizes[name]=d->shapeDef_->lowerName_; weightBindings[base(d->lowerName_)]=name;
    }
    for(Definition* d:file.definition_) if(d->defBody_->shapeDef_) {
      requireTypedShape(*d->defBody_->shapeDef_,d->upperName_);
      const std::vector<Card*>& cards=d->defBody_->shapeDef_->card_;
      if(cards.empty()) throw std::runtime_error("tensor declaration requires dimensions");
      std::vector<std::string>& shape=module.tensorShapes[id(d->upperName_)];
      for(Card* dimension:cards) shape.push_back(card(dimension));
      registerArray(d->defBody_->arrayCount_,id(d->upperName_));
      registerSplits(d->defBody_->shapeDef_,id(d->upperName_)); module.tensorDataSizes[id(d->upperName_)]=d->defBody_->shapeDef_->lowerName_;
    }
    for(const std::pair<const std::string,std::vector<std::string>>& shape:module.tensorShapes) tensorBindings[shape.first]=shape.first;
    for(Definition* d:file.definition_) if(d->defBody_->shapeDef_) tensorBindings[base(d->upperName_)]=id(d->upperName_);
    const std::map<std::string,std::string> globalTensors=tensorBindings, globalWeights=weightBindings;
    for(Definition* d:file.definition_) if(d->defBody_->compDef_) {
      if(isOpaqueSignature(*d->defBody_->compDef_)) {
        const std::string name=id(d->upperName_); unique(module.operations,name);
        AbstractCost cost;
        std::set<std::string> assigned;
        for(const auto* statement:d->defBody_->compDef_->compDefStmt_) {
          const auto* field=statement->lowerDef_;
          if(!field || !isCostField(field->lowerName_))
            throw std::runtime_error("abstract cost declarations may contain only cost parameters: "+name);
          const std::string key=field->lowerName_;
          if(!assigned.insert(key).second) throw std::runtime_error("duplicate cost parameter: "+key);
          const auto& shape=*field->shapeDef_;
          if(!shape.lowerName_.empty() || shape.card_.size()!=1 ||
              std::any_of(shape.splits_.begin(),shape.splits_.end(),[](const Card* c){return c!=nullptr;}))
            throw std::runtime_error("cost parameter requires an integer or defined scalar: "+key);
          const auto* value=shape.card_.front();
          std::string text=value->number_;
          if(text.empty()) {
            const auto scalar=module.scalars.find(id(value->lowerName_));
            if(scalar==module.scalars.end()) throw std::runtime_error("undefined cost scalar: "+value->lowerName_);
            text=scalar->second;
          }
          if(text.empty() || !std::all_of(text.begin(),text.end(),[](unsigned char c){return std::isdigit(c);}))
            throw std::runtime_error("cost parameter must be a nonnegative integer: "+key);
          const double number=std::stod(text);
          if(number>1e12 || (key=="weight_percent" && number>100))
            throw std::runtime_error("cost parameter out of range: "+key);
          if(key=="ops_per_element") { cost.opsPerElement=number; cost.explicitOperations=true; }
          else if(key=="input_reads") cost.inputReads=number;
          else if(key=="output_writes") cost.outputWrites=number;
          else if(key=="weight_percent") cost.weightPercent=number;
          else cost.overheadNs=number;
        }
        module.abstractCosts[name]=cost;
        Function signature; signature.name=name; signature.inputs={"value"}; signature.outputs={"output"};
        signature.inputs.clear();
        // Declared parameters fix the inputs; () means none. Outputs are inferred from the
        // destination, as in TOKENIZE(TEXT) {} >> A:B:C.
        for(const std::string& parameter:d->defBody_->compDef_->parameters_) signature.inputs.push_back(id(parameter));
        inferredOpaque.insert(name);
        module.opaqueSignatures[name]=signature;
        continue;
      }
      Function f; f.name=id(d->upperName_); owner=f.name; stateOwner.clear(); symbols.clear(); serial=0;
      currentFunction=&f; scalarBindings.clear(); tensorBindings=globalTensors; weightBindings=globalWeights;
      body=&f.body;
      const std::vector<CompDefStmt*>& stmts=d->defBody_->compDef_->compDefStmt_;
      std::vector<CompDefExpression*> expressions;
      for(CompDefStmt* st:stmts) {
        if(st->lowerDef_ && isScalar(*st->lowerDef_)) {
          const std::string name=id(st->lowerDef_->lowerName_), qualified=f.name+"_"+name;
          if(scalarBindings.count(name)) throw std::runtime_error("duplicate local scalar "+name);
          scalarBindings[name]=qualified; module.scalars[qualified]=st->lowerDef_->shapeDef_->card_.front()->number_;
        }
        if(st->compDefExpression_) expressions.push_back(st->compDefExpression_);
      }
      for(CompDefStmt* st:stmts) if(st->lowerDef_ && !isScalar(*st->lowerDef_)) {
        const std::string name=base(st->lowerDef_->lowerName_), qualified=f.name+"_"+id(name);
        requireTypedShape(*st->lowerDef_->shapeDef_,name);
        weightBindings[name]=qualified;
        for(Card* dimension:st->lowerDef_->shapeDef_->card_) module.weightShapes[qualified].push_back(card(dimension));
        registerArray(st->lowerDef_->arrayCount_,qualified); registerSplits(st->lowerDef_->shapeDef_,qualified); module.weightDataSizes[qualified]=st->lowerDef_->shapeDef_->lowerName_;
      }
      for(CompDefStmt* st:stmts) if(st->upperDef_) {
        requireTypedShape(*st->upperDef_->shapeDef_,st->upperDef_->upperName_);
        const std::string name=base(st->upperDef_->upperName_), qualified=f.name+"_"+id(name);
        tensorBindings[name]=qualified;
        for(Card* dimension:st->upperDef_->shapeDef_->card_) module.tensorShapes[qualified].push_back(card(dimension));
        registerArray(st->upperDef_->arrayCount_,qualified); registerSplits(st->upperDef_->shapeDef_,qualified); module.tensorDataSizes[qualified]=st->upperDef_->shapeDef_->lowerName_;
      }
      if(expressions.empty()) throw std::runtime_error("computation has no expressions: "+f.name);
      ScopeExpression* first=expressions.front()->scopeExpression_.front();
      f.consumesInput=first->lowerName_.empty();
      if(first->tensor_) for(const auto& item:first->tensor_->arrayCounts_) { f.inputTypes[id(item.first)]="std::vector<Tensor>"; f.inputArraySizes[id(item.first)]=card(item.second); }
      const std::vector<std::string> ins=first->tensor_ ? names(first->tensor_) : std::vector<std::string>{"input"};
      for(const std::string& n:ins) if((!first->tensor_ || !weightName(n)) && n.find('!')==std::string::npos) unique(f.inputs,id(base(n).substr(0,base(n).find('.'))));
      // Later leading members of the same aggregate are independent inputs.
      for(CompDefExpression* st:expressions) if(st->scopeExpression_.front()->tensor_)
        for(const std::string& n:names(st->scopeExpression_.front()->tensor_)) if(n.find('.')!=std::string::npos)
          unique(f.inputs,id(base(n).substr(0,n.find('.'))));
      Values values;
      for(const std::string& n:ins) {
        if((first->tensor_ && weightName(n)) || n.find('!')!=std::string::npos) continue;
        if(n.find('.')!=std::string::npos) values.push_back(reference(n));
        else {
          E value=first->tensor_ && first->tensor_->arrayCounts_.count(n) ? expr(E::Array,{local(id(n))},"1") : local(id(n));
          const std::string declaration=tensorBindings.count(base(n))?tensorBindings.at(base(n)):"";
          if(declarationArrays.count(declaration)) {
            // An input declared as A[n] = shape is a whole array of n checked tensors.
            f.inputTypes[id(n)]="std::vector<Tensor>"; f.inputArraySizes[id(n)]=declarationArrays.at(declaration);
            value=declaredArray(value.kind==E::Array?value:expr(E::Array,{value},"1"),declaration);
          } else if(value.kind!=E::Array && !declaration.empty()) value=declaredTensor(value,declaration);
          values.push_back(value);
          if(tensorBindings.count(base(n))) f.inputDeclarations[id(n)]=tensorBindings.at(base(n));
        }
      }
      for(size_t i=0;i<expressions.size();++i) values=statement(expressions[i]->scopeExpression_,expressions[i]->transferOp_,values,i==0);
      requireDeclaredSources();
      f.results=values;
      if(values.size()==1) f.arrayDepth=arrayDepth(values.front());
      if(values.size()==1 && values.front().kind==E::Group) {
        f.returnsGroup=true;
        f.results=values.front().args;
      }
      for(size_t i=0;i<values.size();++i) {
        std::string field="output_"+std::to_string(i);
        const E& resultValue=values[i].kind==E::Array && !values[i].name.empty()?values[i].args.front():values[i];
        for(const std::pair<const std::string,E>& symbol:symbols) {
          const E& candidate=symbol.second.kind==E::Array && !symbol.second.name.empty()?symbol.second.args.front():symbol.second;
          if(resultValue.kind==E::Local && candidate.kind==E::Local && candidate.name==resultValue.name) field=id(symbol.first);
        }
        if(std::find(f.outputs.begin(),f.outputs.end(),field)!=f.outputs.end()) field+="_"+std::to_string(i);
        f.outputs.push_back(field);
        f.outputArrayDepths.push_back(arrayDepth(values[i])); f.outputSplitAxes.push_back(values[i].splitAxes);
      }
      nameResults(f);
      std::map<std::string,size_t> uses, localDefinitions;
      countLocals(f.body,uses,localDefinitions);
      for(const E& result:f.results) countUses(result,uses);
      f.needsInstance=uses.count("instance")!=0;
      definitions[base(d->upperName_)]=f; module.functions.push_back(std::move(f));
    }
    Definition* entry=nullptr;
    for(Definition* definition:file.definition_) if(definition->defBody_->compDef_) entry=definition;
    if(entry && isOpaqueSignature(*entry->defBody_->compDef_)) {
      Function function; function.name=id(entry->upperName_)+"_entry"; function.inputs={"A"}; function.outputs={"B"};
      owner=function.name; body=&function.body; currentFunction=&function;
      function.results=invoke(entry->upperName_,{local("A")});
      module.functions.push_back(function);
    }
    if(module.functions.empty()) throw std::runtime_error("missing computation definition");
    module.entryName=id(entry->upperName_);
    return module;
  }
};
}
namespace {
// This pass is shared by the operation graph (Chakra) and the ONNX backend.
// It emits ordinary Split/Matmul/Merge IR, never backend-specific operations.
class PartitionLowering {
  Module module;
  // Tiles are row-major over the physical output grid. 'whole' is only a
  // logical view for source validation; consumers use tiles directly.
  struct Value { E whole; std::vector<E> tiles; };
  using Environment=std::map<std::string,Value>;
  size_t serial=0;
  std::set<std::string> usedNames;
  [[noreturn]] static void unsupported(const std::string& detail) {
    throw std::runtime_error("shape partition lowering is not implemented: "+detail);
  }
  E save(E expression,std::vector<Statement>& output) {
    Statement statement;
    do { statement.name="partition_tile_"+std::to_string(serial++); } while(usedNames.count(statement.name));
    usedNames.insert(statement.name);
    E result=local(statement.name); result.layout=expression.layout;
    statement.value=std::move(expression); output.push_back(std::move(statement));
    return result;
  }
  E slice(E value,int count,int index,int axis) {
    return count==1?value:expr(E::Split,{value,local(std::to_string(count)),local(std::to_string(index))},std::to_string(axis));
  }
  E tile(const Value& value,int row,int column) {
    const auto& layout=value.whole.layout;
    const int columns=layout.splitCounts.back();
    if(!value.tiles.empty()) return value.tiles.at(row*columns+column);
    E result=value.whole;
    if(layout.logicalShape.size()==2) result=slice(result,layout.splitCounts.front(),row,0);
    return slice(result,columns,column,-1);
  }
  E concatenate(const std::vector<E>& values,int axis) {
    if(values.size()==1) return values.front();
    E result=expr(E::Merge,{expr(E::Array,values)},"axis:"+std::to_string(axis));
    result.logicalAssembly=true;
    return result;
  }
  Value assemble(TensorLayout layout,std::vector<E> tiles) {
    const int rows=layout.logicalShape.size()==2?layout.splitCounts[0]:1;
    const int columns=layout.splitCounts.back();
    std::vector<E> assembledRows;
    for(int row=0;row<rows;++row) {
      std::vector<E> values(tiles.begin()+row*columns,tiles.begin()+(row+1)*columns);
      assembledRows.push_back(concatenate(values,-1));
    }
    E whole=concatenate(assembledRows,0); whole.layout=std::move(layout);
    return {whole,std::move(tiles)};
  }
  E broadcastTile(const Value& value,const TensorLayout& output,int row,int column) {
    const auto& layout=value.whole.layout;
    if(layout.logicalShape.empty() || layout.logicalShape.size()>2) unsupported("elementwise operand shape");
    const bool matrix=layout.logicalShape.size()==2;
    const bool rowBroadcast=!matrix || layoutDimension(module,layout.logicalShape[0])==1;
    const bool colBroadcast=layoutDimension(module,layout.logicalShape.back())==1;
    const int sourceRows=matrix?layout.splitCounts[0]:1, sourceCols=layout.splitCounts.back();
    E result=tile(value,sourceRows==1?0:row,sourceCols==1?0:column);
    if(!rowBroadcast && sourceRows==1 && output.logicalShape.size()==2) result=slice(result,output.splitCounts[0],row,0);
    if(!colBroadcast && sourceCols==1) result=slice(result,output.splitCounts.back(),column,-1);
    return result;
  }
  Value elementwise(E::Kind kind,const std::vector<Value>& args,std::vector<Statement>& output,const std::string& name={}) {
    TensorLayout layout=kind==E::Scale?args[0].whole.layout:elementwiseLayout(module,args[0].whole.layout,args[1].whole.layout);
    if(layout.logicalShape.empty() || layout.logicalShape.size()>2) unsupported("elementwise operand shapes");
    const int rows=layout.logicalShape.size()==2?layout.splitCounts[0]:1, cols=layout.splitCounts.back();
    std::vector<E> tiles;
    for(int i=0;i<rows;++i) for(int j=0;j<cols;++j) {
      E a=broadcastTile(args[0],layout,i,j);
      E b=kind==E::Scale?args[1].whole:broadcastTile(args[1],layout,i,j);
      tiles.push_back(save(expr(kind,{a,b},name),output));
    }
    return assemble(layout,std::move(tiles));
  }
  Value transpose(const Value& input) {
    TensorLayout layout=input.whole.layout;
    if(layout.logicalShape.size()!=2) unsupported("transpose requires a matrix");
    std::swap(layout.logicalShape[0],layout.logicalShape[1]); std::swap(layout.splitCounts[0],layout.splitCounts[1]);
    std::vector<E> tiles;
    for(int i=0;i<layout.splitCounts[0];++i) for(int j=0;j<layout.splitCounts[1];++j)
      tiles.push_back(expr(E::Transpose,{tile(input,j,i)}));
    return assemble(layout,std::move(tiles));
  }
  E operation(const std::string& name,Values args,std::vector<Statement>& output) {
    return save(expr(E::TensorOp,std::move(args),name),output);
  }
  E reduceLast(const std::vector<E>& tiles,const std::string& op,int width,std::vector<Statement>& output) {
    std::vector<E> partials;
    for(const E& tile:tiles) partials.push_back(operation(op=="max_last"?"max_last":"sum_last",{tile},output));
    E result=partials.front();
    if(op=="max_last") {
      for(size_t i=1;i<partials.size();++i) result=operation("maximum",{result,partials[i]},output);
    } else if(partials.size()>1) result=save(expr(E::Merge,{expr(E::Array,partials)},"sum"),output);
    if(op=="mean_last") result=operation("divide_by:"+std::to_string(width),{result},output);
    return result;
  }
  Value normalized(const Value& input,const std::string& op,std::vector<Statement>& output) {
    TensorLayout layout=input.whole.layout;
    if(layout.logicalShape.empty() || layout.logicalShape.size()>2) unsupported("last-axis operation requires vector/matrix shape");
    const int rows=layout.logicalShape.size()==2?layout.splitCounts[0]:1, columns=layout.splitCounts.back();
    const int width=layoutDimension(module,layout.logicalShape.back());
    const bool reduction=op=="sum_last" || op=="mean_last" || op=="max_last";
    std::vector<E> results;
    for(int row=0;row<rows;++row) {
      std::vector<E> values;
      for(int col=0;col<columns;++col) values.push_back(tile(input,row,col));
      if(reduction) { results.push_back(reduceLast(values,op,width,output)); continue; }
      if(op=="softmax_last") {
        E maximum=reduceLast(values,"max_last",width,output);
        std::vector<E> exponentials;
        for(const E& value:values) {
          E centered=save(expr(E::Subtract,{value,maximum}),output);
          exponentials.push_back(operation("exp",{centered},output));
        }
        E denominator=reduceLast(exponentials,"sum_last",width,output);
        for(const E& value:exponentials) results.push_back(operation("div",{value,denominator},output));
      } else {
        if(op!="layernorm_last" && op!="rmsnorm_last") unsupported("unknown normalization contract");
        if(op=="layernorm_last") {
          E mean=reduceLast(values,"mean_last",width,output);
          for(E& value:values) value=save(expr(E::Subtract,{value,mean}),output);
        }
        std::vector<E> squares;
        for(const E& value:values) squares.push_back(operation("mul",{value,value},output));
        E variance=reduceLast(squares,"mean_last",width,output);
        E scale=operation("rsqrt_eps",{variance},output);
        for(const E& value:values) results.push_back(operation("mul",{value,scale},output));
      }
    }
    if(reduction) { layout.logicalShape.back()="1"; layout.splitCounts.back()=1; }
    return assemble(layout,std::move(results));
  }
  Value multiply(const Value& left,const Value& right,std::vector<Statement>& output) {
    const auto& lhs=left.whole.layout; const auto& rhs=right.whole.layout;
    TensorLayout layout=matmulLayout(module,lhs,rhs);
    if(layout.logicalShape.empty()) unsupported("multiplication requires known operand shapes");
    const int rows=lhs.logicalShape.size()==2?lhs.splitCounts[0]:1;
    const int columns=rhs.splitCounts[1];
    const int contraction=std::max(lhs.splitCounts.back(),rhs.splitCounts[0]);
    if(static_cast<double>(rows)*columns*(static_cast<double>(contraction)+1)>1000000)
      throw std::runtime_error("partitioned multiplication expansion limit exceeded");
    Value result; std::vector<E> assembledRows;
    for(int row=0;row<rows;++row) {
      std::vector<E> assembledColumns;
      for(int column=0;column<columns;++column) {
        std::vector<E> products;
        for(int k=0;k<contraction;++k) {
          E a=tile(left,row,lhs.splitCounts.back()==1?0:k);
          E b=tile(right,rhs.splitCounts[0]==1?0:k,column);
          if(lhs.splitCounts.back()==1) a=slice(a,contraction,k,-1);
          if(rhs.splitCounts[0]==1) b=slice(b,contraction,k,0);
          products.push_back(save(expr(E::Matmul,{a,b}),output));
        }
        E value=products.size()==1?products.front():save(expr(E::Merge,{expr(E::Array,products)},"sum"),output);
        result.tiles.push_back(value); assembledColumns.push_back(value);
      }
      assembledRows.push_back(concatenate(assembledColumns,-1));
    }
    result.whole=concatenate(assembledRows,0); result.whole.layout=std::move(layout);
    return result;
  }
  Value visit(E expression,const Environment& env,std::vector<Statement>& output) {
    if(expression.kind==E::Local && env.count(expression.name)) return env.at(expression.name);
    std::vector<Value> args;
    for(E& arg:expression.args) { args.push_back(visit(std::move(arg),env,output)); arg=args.back().whole; }
    const bool partitioned=std::any_of(args.begin(),args.end(),[](const Value& arg){return hasPartitions(arg.whole.layout);});
    if(expression.kind==E::WeightWrite) {
      validate(args.back().whole.layout,expression.layout,expression.name);
      // All tile values are evaluated before the atomic behavioral update.
      // Graph lowering emits one physical write for each declared partition.
      expression.args.insert(expression.args.end(),args.back().tiles.begin(),args.back().tiles.end());
      return {std::move(expression),args.back().tiles};
    }
    if(partitioned && (expression.kind==E::Add || expression.kind==E::Subtract || expression.kind==E::Scale))
      return elementwise(expression.kind,args,output);
    if(partitioned && expression.kind==E::Transpose) return transpose(args[0]);
    if(expression.kind==E::TensorOp) {
      if(expression.name=="mul" || expression.name=="div") {
        if(partitioned) return elementwise(E::TensorOp,args,output,expression.name);
        expression.layout=elementwiseLayout(module,args[0].whole.layout,args[1].whole.layout);
      } else if(partitioned) return normalized(args[0],expression.name,output);
    }
    if(expression.kind==E::Matmul) {
      const bool partitioned=hasPartitions(args[0].whole.layout) || hasPartitions(args[1].whole.layout);
      if(partitioned) return multiply(args[0],args[1],output);
      expression.layout=matmulLayout(module,args[0].whole.layout,args[1].whole.layout);
    }
    for(const auto& arg:args) if(hasPartitions(arg.whole.layout)) unsupported("operation on a partitioned tensor");
    if(expression.kind==E::Call) {
      for(const auto& function:module.functions) if(function.name==expression.name)
        for(const auto& result:function.results) if(hasPartitions(result.layout)) unsupported("partitioned function result");
    }
    return {std::move(expression),{}};
  }
  void validate(const TensorLayout& inferred,const TensorLayout& declared,const std::string& name) {
    if(inferred.logicalShape.empty()) return;
    bool same=inferred.logicalShape.size()==declared.logicalShape.size() && inferred.splitCounts==declared.splitCounts;
    for(size_t i=0;same && i<inferred.logicalShape.size();++i)
      same=layoutDimension(module,inferred.logicalShape[i])==layoutDimension(module,declared.logicalShape[i]);
    if(!same) throw std::runtime_error("conflicting inferred tensor dimensions or splits: "+name);
  }
  void statements(std::vector<Statement>& body,Environment& env) {
    std::vector<Statement> output;
    for(Statement statement:body) {
      Value value=visit(std::move(statement.value),env,output);
      if(!statement.tensorDeclaration.empty()) {
        const auto& name=statement.tensorDeclaration;
        TensorLayout declared{module.tensorShapes.at(name),module.shapeSplits.at(name)};
        validate(value.whole.layout,declared,name); value.whole.layout=std::move(declared);
      }
      statement.value=value.whole;
      for(E& predicate:statement.predicates) predicate=visit(std::move(predicate),env,output).whole;
      if(statement.kind==Statement::Assign) {
        value.whole=local(statement.name); value.whole.layout=statement.value.layout;
        env[statement.name]=std::move(value);
      }
      if(!statement.body.empty()) {
        Environment nested=env; statements(statement.body,nested);
        for(const auto& item:nested) if(hasPartitions(item.second.whole.layout)) unsupported("partitioned loop state");
      }
      for(auto& alternative:statement.alternatives) {
        Environment nested=env; statements(alternative,nested);
        for(const auto& item:nested) if(hasPartitions(item.second.whole.layout)) unsupported("partitioned branch state");
      }
      output.push_back(std::move(statement));
    }
    body=std::move(output);
  }
  void collectNames(const std::vector<Statement>& body) {
    for(const auto& statement:body) {
      usedNames.insert(statement.name); collectNames(statement.body);
      for(const auto& alternative:statement.alternatives) collectNames(alternative);
    }
  }
public:
  explicit PartitionLowering(const Module& input):module(input) {}
  Module run() {
    bool needed=false;
    for(const auto& item:module.shapeSplits) for(int count:item.second) needed=needed || count>1;
    if(!needed) return module;
    for(const auto& scalar:module.scalars) usedNames.insert(scalar.first);
    for(const auto& function:module.functions) {
      usedNames.insert(function.inputs.begin(),function.inputs.end()); collectNames(function.body);
    }
    for(Function& function:module.functions) {
      Environment env;
      for(const auto& item:function.inputDeclarations) {
        E value=local(item.first);
        value.layout={module.tensorShapes.at(item.second),module.shapeSplits.at(item.second)};
        if(hasPartitions(value.layout) && function.inputTypes.count(item.first)) unsupported("partitioned array/record input");
        env[item.first]={value,{}};
      }
      statements(function.body,env);
      for(E& result:function.results) {
        Value value=visit(std::move(result),env,function.body);
        // A partitioned input cannot be exported as an unsplit operation graph.
        if(hasPartitions(value.whole.layout) && value.tiles.empty()) unsupported("unconsumed partitioned output");
        result=std::move(value.whole);
      }
    }
    return module;
  }
};
}
Module lowerPartitions(const Module& module) { return PartitionLowering(module).run(); }

Module lower(setepSyntax::SetepFile& file) { return Lowering().run(file); }
}
