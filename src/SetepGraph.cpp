// Elaborates the semantic IR into a static operation graph: shapes, arithmetic,
// memory reads, outputs, storage identities and dependencies of every executed
// operation. The Chakra backend places and exports this graph.
#include "SetepGraph.h"
#include "SetepSemantic.h"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <map>
#include <set>
#include <stdexcept>

namespace setep {
namespace {
using E=Expression;
struct Value {
  enum Kind { Tensor, Array, Record, Integer, Path } kind=Tensor;
  std::vector<int64_t> dims;
  std::string dtype="fp32";
  std::set<int> producers;
  std::vector<Value> elements;
  std::map<std::string,Value> fields;
  int64_t integer=0;
  bool flexible=false;
  bool weight=false;
  std::map<std::string,double> storage; // Allocation identities and full backing sizes, MB.
};
using Environment=std::map<std::string,Value>;
int64_t elements(const Value& value) {
  int64_t result=1;
  for(int64_t dim:value.dims) result*=dim;
  return result;
}
int bits(const Value& value) {
  const auto digit=value.dtype.find_first_of("0123456789");
  return digit==std::string::npos?32:std::stoi(value.dtype.substr(digit));
}
double bytes(const Value& value) {
  double result=0;
  if(value.kind==Value::Array) { for(const Value& child:value.elements) result+=bytes(child); return result; }
  if(value.kind==Value::Record) { for(const std::pair<const std::string,Value>& child:value.fields) result+=bytes(child.second); return result; }
  return value.kind==Value::Tensor?std::ceil(elements(value)*bits(value)/8.0):0;
}
void require(bool condition,const std::string& message) { if(!condition) throw std::runtime_error("SETEP graph lowering: "+message); }
class Elaborator {
  const Module& module;
  std::vector<Operation>& operations;
  std::map<std::string,TensorShape>& weightShapes;
  std::map<std::string,Value> states;
  std::map<std::string,Value> updatedWeights;
  std::set<std::string> cacheAllocations;
  std::map<std::string,const Function*> functions;
  int steps=0;
  int64_t decodeMaskLength=0;
  std::set<int> controlDependencies;
  double activationElements(const Value& value) const {
    if(value.kind==Value::Tensor) return value.weight?0:elements(value);
    double total=0;
    for(const auto& member:value.elements) total+=activationElements(member);
    for(const auto& field:value.fields) total+=activationElements(field.second);
    return total;
  }
  double costReadBytes(const Value& value,const AbstractCost& cost) const {
    if(value.kind==Value::Tensor)
      return bytes(value)*(value.weight?cost.weightPercent/100:cost.inputReads);
    double total=0;
    for(const auto& member:value.elements) total+=costReadBytes(member,cost);
    for(const auto& field:value.fields) total+=costReadBytes(field.second,cost);
    return total;
  }
  Value integer(int64_t number) { Value result; result.kind=Value::Integer; result.integer=number; return result; }
  int64_t number(const std::string& name) const { return std::stoll(module.scalars.count(name)?module.scalars.at(name):name); }
  Value declared(const std::vector<std::string>& shape,const std::string& dtype) {
    Value value; value.dtype=dtype;
    for(const std::string& dimension:shape) value.dims.push_back(number(dimension));
    return value;
  }
  // Scalar-operation convention shared with the ONNX lowering. Transcendental
  // functions count as one operation; this is arithmetic work, not kernel time.
  double standardOperations(const std::string& name,const Value& input,double fallback) const {
    if(input.kind!=Value::Tensor || input.dims.empty()) return fallback;
    const double n=elements(input), rows=n/input.dims.back();
    if(name=="LN" || name=="LAYERNORM" || name=="layernorm_last") return 5*n+2*rows;
    if(name=="RMSNORM" || name=="rmsnorm_last") return 3*n+2*rows;
    if(name=="SOFTMAX" || name=="softmax_last") return 5*n;
    if(name=="GELU") return 9*n; // Tanh approximation, including x*x*x.
    if(name=="SILU") return 2*n; // Sigmoid followed by multiplication.
    if(name=="RELU" || name=="SIGMOID" || name=="TANH" || name=="exp") return n;
    if(name=="rsqrt_eps") return 3*n;
    if(name=="LOOKUP") return 0; // Gather moves data, without arithmetic.
    if(name=="SAMPLE") { // Greedy last-position argmax for each batch item.
      const double batches=input.dims.size()>2?rows/input.dims[input.dims.size()-2]:1;
      return batches*(input.dims.back()-1);
    }
    return fallback;
  }
  // Shape contracts for the standard opaque hooks used by the supplied models.
  // Unknown hooks retain a flexible result until a declaration resolves it.
  void opaqueShape(const std::string& name,const std::vector<Value>& args,Value& result) {
    if(name=="LOOKUP" && args.size()==4) { // Two tensors, then path and occurrence.
      const auto& ids=args[0]; const auto& table=args[1];
      require(ids.kind==Value::Tensor && table.kind==Value::Tensor && table.dims.size()==2,
              "LOOKUP requires indices and a rank-two embedding table");
      require(ids.dtype.find("int")==0 || ids.dtype.find("uint")==0,"LOOKUP indices must be integers");
      result=ids; result.dims.push_back(table.dims.back()); result.dtype=table.dtype;
      result.flexible=ids.flexible || table.flexible;
    } else if(name=="LN" || name=="LAYERNORM" || name=="RMSNORM" || name=="SOFTMAX" ||
              name=="GELU" || name=="RELU" || name=="SILU" || name=="SIGMOID" || name=="TANH") {
      result=args.front(); // Normalization/activation preserves resolved shape and dtype.
    } else if(name=="SAMPLE") {
      result=args.front();
      if(result.dims.size()>2) { result.dims.pop_back(); result.dims.back()=1; }
      else result.dims={1};
      // Greedy last-position sampling is the default; an explicit output
      // declaration can still specify a different sampling contract.
      result.dtype="int32"; result.flexible=true;
    } else if(name=="TOKENIZE" && result.kind==Value::Record) {
      const auto mask=result.fields.find("ATTENTION_MASK");
      if(mask!=result.fields.end() && mask->second.dims.size()==1) decodeMaskLength=mask->second.dims[0];
    } else if(name=="DECODE_PREP") {
      if(result.kind==Value::Record && result.fields.count("MASK")) {
        require(decodeMaskLength>0,"DECODE_PREP requires a preceding TOKENIZE mask");
        ++decodeMaskLength;
        for(auto& field:result.fields) {
          field.second=args.front(); field.second.dtype="int32"; field.second.flexible=false;
          field.second.dims={field.first=="MASK"?decodeMaskLength:1};
        }
      } else { result=args.front(); result.flexible=args.front().flexible; }
    }
  }
  void dependencies(const Value& value,std::set<int>& result) {
    result.insert(value.producers.begin(),value.producers.end());
    for(const Value& member:value.elements) dependencies(member,result);
    for(const std::pair<const std::string,Value>& field:value.fields) dependencies(field.second,result);
  }
  void flatten(const Value& value,std::vector<TensorShape>& shapes) {
    if(value.kind==Value::Tensor) shapes.push_back({value.dims,bits(value),value.dtype});
    for(const Value& member:value.elements) flatten(member,shapes);
    for(const std::pair<const std::string,Value>& field:value.fields) flatten(field.second,shapes);
  }
  void producer(Value& value,int id) {
    value.weight=false;
    value.storage.clear();
    if(value.kind==Value::Tensor) value.producers={id};
    for(Value& member:value.elements) producer(member,id);
    for(std::pair<const std::string,Value>& field:value.fields) producer(field.second,id);
  }
  void allocations(Value& value,int id,const std::string& suffix="") {
    if(value.kind==Value::Tensor) value.storage["activation/"+std::to_string(id)+suffix]=bytes(value)/1e6;
    for(size_t i=0;i<value.elements.size();++i) allocations(value.elements[i],id,suffix+"/"+std::to_string(i));
    for(auto& field:value.fields) allocations(field.second,id,suffix+"/"+field.first);
  }
  void resident(const Value& value,Operation& operation) {
    for(const auto& entry:value.storage) operation.storageMb[entry.first]=std::max(operation.storageMb[entry.first],entry.second);
    for(const auto& member:value.elements) resident(member,operation);
    for(const auto& field:value.fields) resident(field.second,operation);
  }
  Value operation(const std::string& name,const std::vector<Value>& args,Value result,double arithmetic) {
    result.weight=false; // Computed values are activations, even when a weight is the first operand.
    Operation item; item.id=static_cast<int>(operations.size()); item.name=name; item.mem=0;
    item.outputWrites=1;
    std::set<int> from=controlDependencies;
    for(const Value& arg:args) { dependencies(arg,from); item.mem+=static_cast<float>(bytes(arg)/1e6); resident(arg,item); }
    item.froms.assign(from.begin(),from.end()); item.ops=static_cast<float>(arithmetic/1e9); item.out=static_cast<float>(bytes(result)/1e6);
    for(int id:from) { require(id>=0 && id<item.id,"invalid dependency"); operations[id].tos.push_back(item.id); }
    producer(result,item.id); allocations(result,item.id); resident(result,item);
    item.residentMem=0; for(const auto& entry:item.storageMb) item.residentMem+=entry.second;
    flatten(result,item.outputs);
    operations.push_back(item);
    return result;
  }
  std::string path(const Value& value) {
    std::string text;
    for(int64_t index:value.dims) text+="/"+std::to_string(index);
    return text;
  }
  Value merge(const std::vector<Value>& values,bool sum=false,bool state=false,int mergeAxis=-1,bool materialize=true) {
    require(!values.empty(),"empty merge/reduction"); Value result=values.front();
    require(result.kind==Value::Tensor,"merge requires tensors");
    for(size_t i=1;i<values.size();++i) {
      const Value& rhs=values[i]; require(rhs.kind==Value::Tensor && rhs.dims.size()==result.dims.size(),"merge rank mismatch");
      result.weight=result.weight && rhs.weight;
      for(const auto& entry:rhs.storage) result.storage[entry.first]=std::max(result.storage[entry.first],entry.second);
      const size_t axis=mergeAxis>=0?static_cast<size_t>(mergeAxis):state?0:result.dims.size()-1;
      for(size_t d=0;d<result.dims.size();++d) if(sum || d!=axis) require(rhs.dims[d]==result.dims[d] || rhs.flexible || result.flexible,"merge shape mismatch");
      if(!sum) result.dims[axis]+=rhs.dims[axis];
      result.producers.insert(rhs.producers.begin(),rhs.producers.end());
    }
    if(sum) return operation("REDUCE",values,result,elements(result)*(values.size()-1));
    // Distinct tensor buffers require an actual concatenation, including cache
    // append. Single nonempty inputs remain views; physical partitions are not
    // assumed to share a contiguous allocation after placement.
    if(std::count_if(values.begin(),values.end(),[&](const Value& v){return bytes(v)>0;})<=1) return result;
    return materialize?operation("CONCAT",values,result,0):result;
  }
  Value evaluate(const E& expression,Environment& env) {
    if(expression.kind==E::Local) {
      if(env.count(expression.name)) return env.at(expression.name);
      return integer(number(expression.name));
    }
    // Opaque-call occurrence labels are metadata, not four-byte tensors.
    if(expression.kind==E::StringLiteral) { Value value; value.kind=Value::Path; return value; }
    std::vector<Value> args;
    for(const E& arg:expression.args) args.push_back(evaluate(arg,env));
    switch(expression.kind) {
    case E::ValidateArray: require(args[0].kind==Value::Array && args[0].elements.size()==static_cast<size_t>(args[1].integer),"array count mismatch"); return args[0];
    case E::Record: { Value result; result.kind=Value::Record; const std::vector<std::string>& fields=module.inputRecords.at(expression.name); for(size_t i=0;i<fields.size();++i) result.fields[fields[i]]=args[i]; return result; }
    case E::Array: { if(!expression.name.empty()) return args.at(0); Value result; result.kind=Value::Array; result.elements=args; return result; }
    case E::Group: { if(expression.name!="literal") return args.at(0); Value result; result.kind=Value::Array; result.elements=args; return result; }
    case E::IndexedPath: { Value result; result.kind=Value::Path; if(!args.empty()) result=args.front(); for(size_t i=1;i<args.size();++i) result.dims.push_back(args[i].integer); return result; }
    case E::ValidateShape: {
      Value result=args.front(); std::vector<int64_t> dims;
      for(size_t i=1;i<args.size();++i) dims.push_back(args[i].integer);
      require(result.flexible || result.dims==dims,"declared tensor shape mismatch"); result.dims=dims; result.flexible=false; return result;
    }
    case E::ArraySize: return integer(args.at(0).elements.size());
    case E::ArrayAt: return args.at(0).elements.at(static_cast<size_t>(args.at(1).integer));
    case E::GroupMember: require(args[0].elements.size()==static_cast<size_t>(args[1].integer),"corresponding array size mismatch"); return args[0].elements.at(args[2].integer);
    case E::Member: return args.at(0).fields.at(expression.name);
    case E::Weight: {
      const std::string address=expression.name+path(args.front());
      if(updatedWeights.count(address)) return updatedWeights.at(address);
      Value value;
      if(module.weightDeclarations.count(expression.name)) { const std::string& key=module.weightDeclarations.at(expression.name); value=declared(module.weightShapes.at(key),module.weightDataSizes.at(key)); }
      else { value.dims={1,1}; value.flexible=true; }
      value.weight=true;
      value.storage["weight/"+address]=bytes(value)/1e6;
      weightShapes[address]={value.dims,bits(value),value.dtype}; return value;
    }
    case E::WeightWrite: {
      Value result=args[1], tile=result;
      size_t count=1;
      for(size_t axis=0;axis<expression.layout.splitCounts.size();++axis) {
        const int splits=expression.layout.splitCounts[axis];
        require(splits>0 && axis<tile.dims.size() && tile.dims[axis]%splits==0,"invalid weight update partition");
        tile.dims[axis]/=splits; count*=splits;
      }
      result.producers.clear();
      std::set<int> priorComputations;
      for(const Operation& previous:operations)
        if(previous.tos.empty()) priorComputations.insert(previous.id);
      for(size_t i=0;i<count;++i) {
        Value input=args.size()>2?args.at(2+i):tile;
        // Wait for the complete update expression before mutating any tile.
        // Keep input byte accounting local to this partition.
        input.producers.insert(args[1].producers.begin(),args[1].producers.end());
        input.producers.insert(priorComputations.begin(),priorComputations.end());
        Value written=operation("WEIGHT_UPDATE",{input},tile,elements(tile));
        result.producers.insert(written.producers.begin(),written.producers.end());
      }
      result.weight=true;
      result.storage={{"weight/"+expression.name+path(args.front()),bytes(result)/1e6}};
      updatedWeights[expression.name+path(args.front())]=result;
      // Mutations form a conservative barrier: later computations cannot read
      // a mixture of old and new partitions, including tied weight uses.
      controlDependencies.insert(result.producers.begin(),result.producers.end());
      return result;
    }
    case E::StateRead: require(states.count(expression.name+path(args.front())),"uninitialized persistent tensor "+expression.name); return states.at(expression.name+path(args.front()));
    case E::StateWrite: {
      Value saved=args.back(); saved.weight=false;
      for(const auto& entry:saved.storage) cacheAllocations.insert(entry.first);
      states[expression.name+path(args.front())]=saved; return saved;
    }
    case E::StateConcat: return merge(args,false,true);
    case E::Call: return call(*functions.at(expression.name),args);
    case E::Opaque: {
      const Function& signature=module.opaqueSignatures.at(expression.name);
      // Arguments end with the instance path and occurrence. A zero-input operation's
      // result is shaped by its declared destination (the semantic layer requires one).
      Value result;
      if(signature.inputs.empty()) result.kind=Value::Tensor;
      else result=args.front();
      result.flexible=true;
      if(signature.outputs.size()>1) {
        Value record; record.kind=Value::Record;
        for(const std::string& field:signature.outputs) {
          // Named external-operation outputs can have different declared
          // widths and data types (for example MLA latent and RoPE outputs).
          record.fields[field]=module.tensorShapes.count(field)
            ? declared(module.tensorShapes.at(field),module.tensorDataSizes.at(field)) : result;
        }
        result=record;
      }
      if(!signature.inputs.empty()) opaqueShape(expression.name,args,result);
      const auto& cost=module.abstractCosts.at(expression.name);
      double arithmetic=0, readBytes=0;
      for(const auto& input:args) {
        arithmetic+=activationElements(input)*cost.opsPerElement;
        readBytes+=costReadBytes(input,cost);
      }
      if(!cost.explicitOperations && !signature.inputs.empty())
        arithmetic=standardOperations(expression.name,args.front(),arithmetic);
      // Embedding table residency and table read traffic are different quantities.
      // Charge one selected row per index (no data-dependent duplicate-index reuse).
      if(expression.name=="LOOKUP" && args.size()==4 && args[1].weight) {
        readBytes-=costReadBytes(args[1],cost);
        readBytes+=bytes(result)*cost.weightPercent/100;
      }
      Value output=operation(expression.name,args,result,arithmetic);
      auto& item=operations.back();
      item.mem=static_cast<float>(readBytes/1e6);
      item.outputWrites=cost.outputWrites; item.overheadNs=cost.overheadNs;
      return output;
    }
    case E::TensorOp: {
      const std::string& op=expression.name;
      Value result=args.front(); double arithmetic=elements(result);
      if(op=="sum_last" || op=="mean_last" || op=="max_last") {
        require(!result.dims.empty(),"last-axis reduction requires a tensor");
        result.dims.back()=1;
        arithmetic=elements(args.front())-(op=="mean_last"?0:elements(result));
      } else if(op=="mul" || op=="div" || op=="maximum") {
        const auto& rhs=args.at(1); const size_t rank=std::max(result.dims.size(),rhs.dims.size());
        std::vector<int64_t> dims(rank,1);
        for(size_t i=0;i<rank;++i) {
          const int64_t a=i<result.dims.size()?result.dims[result.dims.size()-1-i]:1;
          const int64_t b=i<rhs.dims.size()?rhs.dims[rhs.dims.size()-1-i]:1;
          require(a==b || a==1 || b==1,"elementwise size mismatch"); dims[rank-1-i]=std::max(a,b);
        }
        result.dims=dims; arithmetic=elements(result);
      } else arithmetic=standardOperations(op,args.front(),arithmetic);
      return operation(op,args,result,arithmetic);
    }
    case E::Predicate: { Value result; result.dims={1}; result.dtype="int8"; return operation(expression.name,args,result,1); }
    case E::Transpose: { Value result=args.front(); require(result.dims.size()>=2,"transpose requires rank two or higher"); std::swap(result.dims[result.dims.size()-1],result.dims[result.dims.size()-2]); return result; }
    case E::Split: {
      Value result=args.front(); const int64_t count=args[1].integer;
      const size_t axis=expression.name.empty() || expression.name=="-1"?result.dims.size()-1:static_cast<size_t>(std::stoi(expression.name));
      require(count>0 && axis<result.dims.size() && (result.flexible || result.dims[axis]%count==0),"invalid split count");
      result.dims[axis]=std::max<int64_t>(1,result.dims[axis]/count);
      // A physical partition owns only its tile, including partitioned inputs
      // and gradients. Charging the whole backing tensor prevents valid tiles
      // from fitting on the device that executes them.
      if(result.storage.size()==1) {
        const auto key=result.storage.begin()->first+"/slice/"+std::to_string(axis)+"/"+std::to_string(count)+"/"+std::to_string(args[2].integer);
        result.storage={{key,bytes(result)/1e6}};
      }
      return result;
    }
    case E::Merge:
      if(expression.name.rfind("axis:",0)==0) {
        std::vector<Value> values=args.front().elements;
        require(!values.empty(),"empty concatenation");
        const size_t axis=expression.name.substr(5)=="-1"?values.front().dims.size()-1:static_cast<size_t>(std::stoi(expression.name.substr(5)));
        for(const Value& value:values) require(axis<value.dims.size(),"invalid concatenation axis");
        return merge(values,false,false,static_cast<int>(axis),!expression.logicalAssembly);
      }
      return merge(expression.name.empty() || expression.name=="group"?args:args.front().elements,expression.name=="sum");
    case E::ReduceScatter: {
      require(!args.front().elements.empty(),"empty reduce scatter"); Value result=args.front().elements.front();
      for(size_t i=0;i<result.elements.size();++i) { std::vector<Value> column; for(const Value& row:args.front().elements) { require(row.elements.size()==result.elements.size(),"reduce scatter size mismatch"); column.push_back(row.elements[i]); } result.elements[i]=merge(column,true); }
      return result;
    }
    case E::Matmul: {
      Value left=args[0], right=args[1];
      if(left.flexible && right.dims.size()>=2) { if(left.dims.size()<2) left.dims={left.dims.empty()?1:left.dims.front(),right.dims[right.dims.size()-2]}; else left.dims.back()=right.dims[right.dims.size()-2]; }
      if(right.flexible && left.dims.size()>=2) { if(right.dims.size()<2) right.dims={left.dims.back(),left.dims.back()}; else right.dims[right.dims.size()-2]=left.dims.back(); }
      require(left.dims.size()>=1 && right.dims.size()>=2 && left.dims.back()==right.dims[right.dims.size()-2],"matrix multiplication size mismatch");
      Value result=left; result.dims.back()=right.dims.back();
      // A resolved product must not be reinterpreted as the shape of a later bias.
      result.flexible=false;
      if(left.flexible) result.dtype=right.dtype;
      else if(right.flexible) result.dtype=left.dtype;
      else require(left.dtype==right.dtype,"matrix multiplication dtype mismatch");
      return operation("MATMUL",args,result,2.0*elements(result)*left.dims.back());
    }
    case E::Scale: return operation("SCALE",args,args.front(),elements(args.front()));
    case E::Add: case E::Subtract: {
      Value result=args[0]; const Value& rhs=args[1];
      const size_t rank=std::max(result.dims.size(),rhs.dims.size()); std::vector<int64_t> shape(rank,1);
      for(size_t i=0;i<rank;++i) { const int64_t a=i<result.dims.size()?result.dims[result.dims.size()-1-i]:1, b=i<rhs.dims.size()?rhs.dims[rhs.dims.size()-1-i]:1; require(a==b || a==1 || b==1 || result.flexible || rhs.flexible,"addition size mismatch"); shape[rank-1-i]=std::max(a,b); }
      result.dims=shape; return operation(expression.kind==E::Add?"ADD":"SUBTRACT",args,result,elements(result));
    }
    default: throw std::runtime_error("unsupported SETEP semantic expression in the operation graph");
    }
  }
  void combineProducers(Value& target,const Value& source) {
    target.producers.insert(source.producers.begin(),source.producers.end());
    for(size_t i=0;i<std::min(target.elements.size(),source.elements.size());++i) combineProducers(target.elements[i],source.elements[i]);
    for(std::pair<const std::string,Value>& field:target.fields) if(source.fields.count(field.first)) combineProducers(field.second,source.fields.at(field.first));
  }
  void statements(const std::vector<Statement>& body,Environment& env) {
    for(const Statement& statement:body) {
      require(++steps<1000000,"operation expansion limit exceeded");
      if(statement.kind==Statement::Check) { require(evaluate(statement.value,env).integer!=0,statement.name); }
      else if(statement.kind==Statement::Loop) {
        const int64_t count=env.count(statement.count)?env.at(statement.count).integer:number(statement.count);
        require(count>0 && count<=100000,"invalid repetition count");
        for(int64_t i=0;i<count;++i) {
          const int index=static_cast<int>(statement.reverse?count-1-i:i);
          env[statement.name]=integer(index); statements(statement.body,env);
        }
      } else if(statement.kind==Statement::Branch) {
        // Static traces contain all candidate work. Exactly-one choices retain
        // one representative result; run-time source backends execute predicates.
        const Environment before=env;
        const std::set<int> savedControl=controlDependencies;
        std::vector<Value> predicates;
        for(const E& predicate:statement.predicates) predicates.push_back(evaluate(predicate,env));
        for(size_t i=0;i<statement.alternatives.size();++i) {
          controlDependencies=savedControl;
          dependencies(predicates.at(i),controlDependencies);
          Environment branch=statement.exclusive?before:env;
          statements(statement.alternatives[i],branch);
          if(!statement.exclusive || i==0) env=branch;
          else for(std::pair<const std::string,Value>& item:env) if(branch.count(item.first)) combineProducers(item.second,branch.at(item.first));
        }
        controlDependencies=savedControl;
      } else if(statement.kind==Statement::Append) {
        Value value=evaluate(statement.value,env);
        if(statement.reverse) env[statement.name].elements.insert(env[statement.name].elements.begin(),value);
        else env[statement.name].elements.push_back(value);
      }
      else if(statement.kind==Statement::Extend) { Value value=evaluate(statement.value,env); env[statement.name].elements.insert(env[statement.name].elements.end(),value.elements.begin(),value.elements.end()); }
      else {
        Value value;
        if(statement.group) value.kind=Value::Array;
        else value=evaluate(statement.value,env);
        if(!statement.tensorDeclaration.empty() && value.kind==Value::Tensor) {
          const Value shape=declared(module.tensorShapes.at(statement.tensorDeclaration),module.tensorDataSizes.at(statement.tensorDeclaration));
          require(value.flexible || value.dims==shape.dims,"declared tensor shape mismatch: "+statement.tensorDeclaration);
          const bool inferred=value.flexible;
          value.dims=shape.dims; value.dtype=shape.dtype; value.flexible=false;
          for(int id:value.producers) {
            Operation& op=operations[id];
            if(op.outputs.size()==1 && (inferred || op.outputs[0].dims==value.dims)) {
              op.outputs[0]={value.dims,bits(value),value.dtype}; op.out=static_cast<float>(bytes(value)/1e6);
              const std::string key="activation/"+std::to_string(id);
              if(value.storage.count(key)) { value.storage[key]=bytes(value)/1e6; op.storageMb[key]=bytes(value)/1e6; }
              op.residentMem=0; for(const auto& entry:op.storageMb) op.residentMem+=entry.second;
            }
          }
        }
        env[statement.name]=value;
      }
    }
  }
  Value call(const Function& function,const std::vector<Value>& args) {
    Environment env; for(size_t i=0;i<function.inputs.size();++i) env[function.inputs[i]]=args.at(i);
    size_t offset=function.inputs.size(); env["instance_path"]=args.at(offset++);
    if(function.needsInstance) env["instance"]=args.at(offset++); else env["instance"]=integer(0);
    env["state_path"]=args.at(offset);
    statements(function.body,env);
    if(function.results.size()==1) return evaluate(function.results.front(),env);
    Value result; result.kind=Value::Record;
    for(size_t i=0;i<function.results.size();++i) result.fields[function.outputs[i]]=evaluate(function.results[i],env);
    return result;
  }
public:
  Elaborator(const Module& input,Graph& graph)
    :module(input),operations(graph.operations),weightShapes(graph.weightShapes) {
    for(const Function& function:module.functions) functions[function.name]=&function;
  }
  void run() {
    const Function& entry=module.functions.back(); std::vector<Value> inputs;
    for(const std::string& name:entry.inputs) {
      Value value;
      if(entry.inputDeclarations.count(name)) { const std::string key=entry.inputDeclarations.at(name); value=declared(module.tensorShapes.at(key),module.tensorDataSizes.at(key)); }
      else if(module.tensorShapes.count(name)) value=declared(module.tensorShapes.at(name),module.tensorDataSizes.at(name));
      else { value.dims={1,1}; value.flexible=true; }
      if(entry.inputArraySizes.count(name)) { Value array; array.kind=Value::Array; array.elements=std::vector<Value>(number(entry.inputArraySizes.at(name)),value); value=array; }
      if(value.kind==Value::Tensor) value.storage["input/"+name]=bytes(value)/1e6;
      inputs.push_back(value);
    }
    Value pathValue; pathValue.kind=Value::Path; inputs.push_back(pathValue);
    if(entry.needsInstance) inputs.push_back(integer(0));
    inputs.push_back(pathValue);
    call(entry,inputs);
    // State assignment aliases existing buffers. Canonicalize all references,
    // including the producing operation, rather than allocating a second cache.
    for(auto& op:operations) {
      std::map<std::string,double> storage;
      for(const auto& allocation:op.storageMb) {
        const std::string key=cacheAllocations.count(allocation.first)?"cache/"+allocation.first:allocation.first;
        storage[key]=std::max(storage[key],allocation.second);
      }
      op.storageMb=std::move(storage);
    }
  }
};
}
Graph buildGraph(const Module& module) {
  Graph graph;
  Elaborator(module,graph).run();
  return graph;
}
}
