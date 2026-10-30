#pragma once
#include "SetepSyntax.h"
#include <stdexcept>

namespace setep {
inline bool isScalar(const setepSyntax::LowerDef& definition) {
  const setepSyntax::ShapeDef& shape=*definition.shapeDef_;
  bool split=false; for(const auto* count:shape.splits_) split=split || count!=nullptr;
  return !split && shape.lowerName_.empty() && shape.card_.size()==1 && !shape.card_.front()->number_.empty();
}
inline void requireTypedShape(const setepSyntax::ShapeDef& shape,const std::string& name) {
  if(shape.card_.empty() || shape.lowerName_.empty())
    throw std::runtime_error("shape declaration '"+name+"' requires dimensions and @dtype");
}
inline bool isCostField(const std::string& name) {
  return name=="ops_per_element" || name=="input_reads" || name=="output_writes"
    || name=="weight_percent" || name=="overhead_ns";
}
inline bool isOpaqueSignature(const setepSyntax::CompDef& definition) {
  if(definition.compDefStmt_.empty()) return true;
  for(const auto* statement:definition.compDefStmt_)
    if(statement->lowerDef_ && isCostField(statement->lowerDef_->lowerName_)) return true;
  return false;
}
}
