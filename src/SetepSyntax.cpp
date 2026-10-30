#include "SetepSyntax.h"
#include <algorithm>
#include <stdexcept>

namespace setepSyntax {
namespace {
// REPEAT and CHOOSE prefix a repetition or selector without naming the scope.
bool anonymousScope(const std::string& name) { return name=="REPEAT" || name=="CHOOSE"; }

class Normalize {
  SetepFile& file;
  template<class T, class S> T* make(const S* source) {
    std::shared_ptr<T> node=std::make_shared<T>();
    node->sourceStart_=source->sourceStart_; node->sourceEnd_=source->sourceEnd_;
    file.storage_.push_back(node); return node.get();
  }
  Card* card(const SetepParser::Card* source) {
    if(!source) return nullptr;
    Card* node=make<Card>(source); node->number_=source->number_; node->lowerName_=source->lowerName_; return node;
  }
  ShapeDef* shape(const SetepParser::Shape* source) {
    ShapeDef* node=make<ShapeDef>(source); node->lowerName_=source->lowerName_; node->strings_=source->strings_;
    for(const SetepParser::ShapeDim* dimension:source->shapeDim_) { node->card_.push_back(card(dimension->dim_)); node->splits_.push_back(card(dimension->split_)); }
    return node;
  }
  TransferOp* transfer(const SetepParser::TransferOp* source) {
    if(source->strings_.count("+") && (!source->strings_.count("[") || source->card_))
      throw std::runtime_error("array reduction requires >+[]> without a count");
    TransferOp* node=make<TransferOp>(source); node->strings_=source->strings_;
    if(source->card_) {
      node->transferQualifier_=make<TransferQualifier>(source);
      node->transferQualifier_->card_=card(source->card_);
    }
    return node;
  }
  static bool isWeight(const SetepParser::Value* value) { return !value->valueName_->name_->lowerName_.empty(); }
  static const SetepParser::Card* count(const SetepParser::Value* value) { return value->valueName_->card_; }
  std::string suffix(const SetepParser::Value* value) {
    const SetepParser::ValueName* valueName=value->valueName_;
    std::string name=isWeight(value) ? valueName->name_->lowerName_ : valueName->name_->upperName_;
    if(valueName->strings_.count("[") && !valueName->card_) name+="[]";
    if(value->strings_.count("^")) name+="^";
    return name;
  }
  Tensor* tensor(const SetepParser::Group* source) {
    Tensor* node=make<Tensor>(source); TensorGroup* group=make<TensorGroup>(source);
    group->tensorEnum_=make<TensorEnum>(source);
    for(const SetepParser::Value* value:source->value_) {
      if(isWeight(value) && value->strings_.count("(")) throw std::runtime_error("weights cannot be called as computations");
      const std::string name=suffix(value);
      group->tensorEnum_->upperName_.push_back(name);
      if(count(value)) node->arrayCounts_[name]=card(count(value));
    }
    if(source->value_.size()==1 && !node->arrayCounts_.empty()) node->arrayCount_=node->arrayCounts_.begin()->second;
    node->tensorGroup_.push_back(group); return node;
  }
  bool weightsOnly(const SetepParser::Group* source) {
    if(!source) return false;
    for(const SetepParser::Value* value:source->value_) {
      if(isWeight(value) && value->strings_.count("("))
        throw std::runtime_error("weights cannot be called as computations");
      if(!isWeight(value)) return false;
    }
    return true;
  }
  Weight* weight(const SetepParser::Group* source) {
    Weight* node=make<Weight>(source); node->weightEnum_=make<WeightEnum>(source);
    for(const SetepParser::Value* value:source->value_) {
      const std::string name=suffix(value);
      node->weightEnum_->lowerName_.push_back(name);
      if(count(value)) node->arrayCounts_[name]=card(count(value));
    }
    return node;
  }
  CompOp* operation(const SetepParser::ScopeOp* source) {
    if(!source) return nullptr;
    CompOp* node=make<CompOp>(source); node->strings_=source->strings_; return node;
  }
  ScopeExpression* expression(const SetepParser::ScopeExpr* source) {
    ScopeExpression* node=make<ScopeExpression>(source);
    const SetepParser::ScopeArg* first=source->scopeArg_.front();
    bool simple=!source->first_ && source->scopeOp_.empty();
    if(simple && first->scope_) { node->scope_=scope(first->scope_); return node; }
    if(simple && weightsOnly(first->group_) && first->group_->value_.size()==1
        && !count(first->group_->value_.front())) {
      node->lowerName_=suffix(first->group_->value_.front()); return node;
    }
    if(simple && first->group_ && !weightsOnly(first->group_)) {
      bool calls=false;
      for(const SetepParser::Value* value:first->group_->value_) calls=calls || value->strings_.count("(");
      if(!calls) { node->tensor_=tensor(first->group_); return node; }
    }
    node->computation_=make<Computation>(source);
    node->computation_->compBody_=make<CompBody>(source);
    CompStmt* stmt=make<CompStmt>(source); CompExpression* expr=make<CompExpression>(source);
    expr->first_=operation(source->first_);
    for(const SetepParser::ScopeOp* op:source->scopeOp_) expr->compOp_.push_back(operation(op));
    for(const SetepParser::ScopeArg* arg:source->scopeArg_) {
      Arg* value=make<Arg>(arg);
      if(arg->scope_) value->scope_=scope(arg->scope_);
      if(weightsOnly(arg->group_)) value->weight_=weight(arg->group_);
      else if(arg->group_) {
        value->tensor_=tensor(arg->group_);
        for(const SetepParser::Value* item:arg->group_->value_) value->calls_.push_back(item->strings_.count("(")!=0);
        if(value->tensor_->arrayCount_) for(bool call:value->calls_) if(call) throw std::runtime_error("repeat computation calls using a scope: REPEAT/n{F()}");
      }
      expr->arg_.push_back(value);
    }
    stmt->compExpression_.push_back(expr); node->computation_->compBody_->compStmt_.push_back(stmt); return node;
  }
  // NAME = shape or NAME[n] = shape: one plain value, with nothing after the shape.
  static const SetepParser::ValueName* defined(const SetepParser::ScopeStmt* source) {
    const SetepParser::ScopeExpr* expr=source->scopeExpr_.front();
    const SetepParser::ScopeArg* arg=expr->scopeArg_.front();
    if(!source->scopeSep_.empty() || expr->first_ || !expr->scopeOp_.empty() || !arg->group_
        || arg->group_->value_.size()!=1 || !arg->group_->value_.front()->strings_.empty())
      throw std::runtime_error("a definition names one value: NAME = shape or NAME[n] = shape");
    const SetepParser::ValueName* name=arg->group_->value_.front()->valueName_;
    if(name->strings_.count("[") && !name->card_) throw std::runtime_error("array declaration requires an explicit count: NAME[n] = shape");
    return name;
  }
  ScopeStmt* statement(const SetepParser::ScopeStmt* source) {
    ScopeStmt* node=make<ScopeStmt>(source);
    if(source->shape_) {
      const SetepParser::ValueName* name=defined(source);
      if(name->card_ && source->shape_->lowerName_.empty()) throw std::runtime_error("array declaration requires a typed tensor or weight: NAME[n] = shape@dtype");
      if(!name->name_->lowerName_.empty()) {
        node->lowerDef_=make<LowerDef>(source); node->lowerDef_->lowerName_=name->name_->lowerName_;
        node->lowerDef_->shapeDef_=shape(source->shape_); node->lowerDef_->arrayCount_=card(name->card_);
      } else {
        node->upperDef_=make<UpperDef>(source); node->upperDef_->upperName_=name->name_->upperName_;
        node->upperDef_->shapeDef_=shape(source->shape_); node->upperDef_->arrayCount_=card(name->card_);
      }
      return node;
    }
    for(const SetepParser::ScopeExpr* expr:source->scopeExpr_) node->scopeExpression_.push_back(expression(expr));
    for(const SetepParser::ScopeSep* sep:source->scopeSep_) {
      ScopeSep* out=make<ScopeSep>(sep); out->strings_=sep->strings_;
      if(sep->transferOp_) out->transferOp_=transfer(sep->transferOp_);
      node->scopeSep_.push_back(out);
    }
    return node;
  }
  // NAME{...}, NAME/n{...}, NAME/[]{...}, REPEAT/n{...} or CHOOSE?SELECT(){a | b}.
  Scope* scope(const SetepParser::Scope* source) {
    Scope* node=make<Scope>(source);
    node->scopeBody_=make<ScopeBody>(source->scopeBody_);
    for(const SetepParser::ScopeStmt* stmt:source->scopeBody_->scopeStmt_) node->scopeBody_->scopeStmt_.push_back(statement(stmt));
    const SetepParser::ScopeName* prefix=source->scopeName_;
    if(!prefix) return node;
    if(prefix->scopeName_=="REPEAT" && !prefix->repetition_) throw std::runtime_error("REPEAT requires a repetition: REPEAT/n{...}");
    if(prefix->scopeName_=="CHOOSE" && !prefix->choice_) throw std::runtime_error("CHOOSE requires a selector: CHOOSE?SELECT(){a | b}");
    if(!anonymousScope(prefix->scopeName_)) node->scopeName_=prefix->scopeName_;
    if(prefix->choice_) node->choiceSel_=prefix->choice_->choiceComp_+(prefix->choice_->strings_.count("[")?"[]":"");
    const SetepParser::Repetition* rep=prefix->repetition_;
    if(!rep) return node;
    if(!rep->repType_->card_ && rep->repType_->strings_.empty()) throw std::runtime_error("scope repetition requires a count or []");
    if(prefix->choice_) throw std::runtime_error("selectors require explicit alternatives without repetition");
    node->repetition_=make<Repetition>(rep); node->repetition_->card_=card(rep->repType_->card_);
    node->repetition_->repOp_=make<RepOp>(rep); node->repetition_->repOp_->strings_=rep->repDirection_->strings_;
    if(!rep->repType_->strings_.empty()) { node->repetition_->repType_=make<RepType>(rep); node->repetition_->repType_->strings_=rep->repType_->strings_; }
    return node;
  }
public:
  explicit Normalize(SetepFile& out):file(out) {}
  // Value definitions (NAME = shape, NAME[n] = shape) precede computations (NAME{...});
  // each keeps source order. Top-level statements must be definitions.
  void run(SetepParser::SetepModel& source) {
    for(const SetepParser::ScopeStmt* stmt:source.scopeStmt_) {
      if(!stmt->shape_) throw std::runtime_error("top-level statements must be definitions: NAME = shape");
      const ScopeStmt* def=statement(stmt);
      if(def->lowerDef_) { file.lowerDef_.push_back(def->lowerDef_); continue; }
      Definition* node=make<Definition>(stmt); node->upperName_=def->upperDef_->upperName_; node->defBody_=make<DefBody>(stmt);
      node->defBody_->shapeDef_=def->upperDef_->shapeDef_; node->defBody_->arrayCount_=def->upperDef_->arrayCount_;
      file.definition_.push_back(node);
    }
    for(const SetepParser::CompDef* def:source.compDef_) {
      // upperName_ holds the computation name followed by its parameters. Parentheses are
      // required, empty for no inputs; without them NAME{...} only names a scope.
      if(!def->strings_.count("("))
        throw std::runtime_error("computation definitions require parentheses: "+def->upperName_.front()+"() {...}");
      Definition* node=make<Definition>(def); node->upperName_=def->upperName_.front(); node->defBody_=make<DefBody>(def);
      CompDef* comp=node->defBody_->compDef_=make<CompDef>(def);
      comp->parameters_.assign(def->upperName_.begin()+1,def->upperName_.end());
      const auto* body=def->scopeBody_;
      const std::vector<SetepParser::ScopeStmt*> empty;
      for(const SetepParser::ScopeStmt* stmt:body ? body->scopeStmt_ : empty) {
        ScopeStmt* normalized=statement(stmt); CompDefStmt* out=make<CompDefStmt>(stmt);
        out->lowerDef_=normalized->lowerDef_; out->upperDef_=normalized->upperDef_;
        if(!stmt->shape_) {
          out->compDefExpression_=make<CompDefExpression>(stmt); out->compDefExpression_->scopeExpression_=normalized->scopeExpression_;
          for(ScopeSep* sep:normalized->scopeSep_) {
            if(!sep->transferOp_) throw std::runtime_error("choice alternatives must be inside a scope with CHOOSE?SELECT()");
            out->compDefExpression_->transferOp_.push_back(sep->transferOp_);
          }
        }
        comp->compDefStmt_.push_back(out);
      }
      bindParameters(def,*comp);
      file.definition_.push_back(node);
    }
  }
  // NAME(A:B){...} binds its inputs as a leading A:B statement would, placed before the
  // first expression. A body already starting with exactly A:B needs nothing added; an
  // abstract body (no expressions) keeps the parameters as its declared signature.
  void bindParameters(const SetepParser::CompDef* def, CompDef& comp) {
    if(comp.parameters_.empty()) return;
    auto first=std::find_if(comp.compDefStmt_.begin(),comp.compDefStmt_.end(),[](const CompDefStmt* s){return s->compDefExpression_!=nullptr;});
    if(first==comp.compDefStmt_.end()) return;
    const ScopeExpression* lead=(*first)->compDefExpression_->scopeExpression_.front();
    if(lead->tensor_ && lead->tensor_->tensorGroup_.size()==1 && lead->tensor_->tensorGroup_.front()->tensorEnum_
        && lead->tensor_->tensorGroup_.front()->tensorEnum_->upperName_==comp.parameters_) return;
    CompDefStmt* out=make<CompDefStmt>(def); out->compDefExpression_=make<CompDefExpression>(def);
    ScopeExpression* expr=make<ScopeExpression>(def); expr->tensor_=make<Tensor>(def);
    TensorGroup* group=make<TensorGroup>(def); group->tensorEnum_=make<TensorEnum>(def);
    group->tensorEnum_->upperName_=comp.parameters_;
    expr->tensor_->tensorGroup_.push_back(group);
    out->compDefExpression_->scopeExpression_.push_back(expr);
    comp.compDefStmt_.insert(first,out);
  }
};
}
SetepFile::SetepFile(SetepParser::SetepModel& model) { Normalize(*this).run(model); }
}
