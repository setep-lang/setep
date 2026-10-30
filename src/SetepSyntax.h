#pragma once
#include "SetepParser.h"
#include <memory>
#include <map>

// Owned normalization of the generated grammar for the source and graph backends.
// Nodes retain source locations; this layer never reparses or rewrites input text.
namespace setepSyntax {
class ScopeExpression;
class Scope;
class LowerDef;
class UpperDef;

class RepOp
{
public:
   std::set<std::string> strings_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class RepType
{
public:
   std::set<std::string> strings_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class Card
{
public:
   std::string lowerName_;
   std::string number_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class Repetition
{
public:
   Card* card_ = nullptr;
   RepOp* repOp_ = nullptr;
   RepType* repType_ = nullptr;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class TensorEnum
{
public:
   std::vector<std::string> upperName_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class TensorArray
{
public:
   Card* card_ = nullptr;
   std::string upperName_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class TensorGroup
{
public:
   TensorArray* tensorArray_ = nullptr;
   TensorEnum* tensorEnum_ = nullptr;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class Tensor
{
public:
   Card* arrayCount_ = nullptr;
   std::map<std::string, Card*> arrayCounts_;
   std::vector<TensorGroup*> tensorGroup_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class WeightEnum
{
public:
   std::vector<std::string> lowerName_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class WeightArray
{
public:
   Card* card_ = nullptr;
   std::string lowerName_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class Weight
{
public:
   Card* arrayCount_ = nullptr;
   std::map<std::string, Card*> arrayCounts_;
   WeightArray* weightArray_ = nullptr;
   WeightEnum* weightEnum_ = nullptr;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class Arg
{
public:
   Scope* scope_ = nullptr;
   std::vector<bool> calls_;
   Tensor* tensor_ = nullptr;
   Weight* weight_ = nullptr;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class TransferQualifier
{
public:
   Card* card_ = nullptr;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class TransferOp
{
public:
   TransferQualifier* transferQualifier_ = nullptr;
   std::set<std::string> strings_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class CompOp
{
public:
   std::set<std::string> strings_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class CompExpression
{
public:
   std::vector<Arg*> arg_;
   std::vector<CompOp*> compOp_;
   CompOp* first_ = nullptr;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class CompStmt
{
public:
   std::vector<CompExpression*> compExpression_;
   std::vector<TransferOp*> transferOp_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class CompBody
{
public:
   std::vector<CompStmt*> compStmt_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class Computation
{
public:
   CompBody* compBody_ = nullptr;
   Repetition* repetition_ = nullptr;
   std::string upperName_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class ScopeSep
{
public:
   TransferOp* transferOp_ = nullptr;
   std::set<std::string> strings_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class ScopeStmt
{
public:
   LowerDef* lowerDef_ = nullptr;
   UpperDef* upperDef_ = nullptr;
   std::vector<ScopeExpression*> scopeExpression_;
   std::vector<ScopeSep*> scopeSep_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class ScopeBody
{
public:
   std::vector<ScopeStmt*> scopeStmt_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class Scope
{
public:
   Repetition* repetition_ = nullptr;
   ScopeBody* scopeBody_ = nullptr;
   std::string choiceSel_;
   std::string scopeName_;
   std::set<std::string> strings_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class ScopeExpression
{
public:
   Computation* computation_ = nullptr;
   Scope* scope_ = nullptr;
   Tensor* tensor_ = nullptr;
   std::string lowerName_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class CompDefExpression
{
public:
   std::vector<ScopeExpression*> scopeExpression_;
   std::vector<TransferOp*> transferOp_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class ShapeDef
{
public:
   std::vector<Card*> card_;
   std::vector<Card*> splits_;
   std::string lowerName_;
   std::set<std::string> strings_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class LowerDef
{
public:
   Card* arrayCount_ = nullptr; // NAME[n] = shape
   ShapeDef* shapeDef_ = nullptr;
   std::string lowerName_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class UpperDef
{
public:
   Card* arrayCount_ = nullptr; // NAME[n] = shape
   ShapeDef* shapeDef_ = nullptr;
   std::string upperName_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class CompDefStmt
{
public:
   CompDefExpression* compDefExpression_ = nullptr;
   LowerDef* lowerDef_ = nullptr;
   UpperDef* upperDef_ = nullptr;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class CompDef
{
public:
   std::vector<CompDefStmt*> compDefStmt_;
   std::vector<std::string> parameters_; // NAME(A:B){...} inputs, in order
   std::set<std::string> strings_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class DefBody
{
public:
   Card* arrayCount_ = nullptr; // NAME[n] = shape
   CompDef* compDef_ = nullptr;
   ShapeDef* shapeDef_ = nullptr;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class Definition
{
public:
   DefBody* defBody_ = nullptr;
   std::string upperName_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

class SetepFile
{
public:
   explicit SetepFile(SetepParser::SetepModel& model);
   std::vector<std::shared_ptr<void>> storage_;
   std::vector<Definition*> definition_;
   std::vector<LowerDef*> lowerDef_;
   size_t sourceStart_ = 0, sourceEnd_ = 0;
};

}
