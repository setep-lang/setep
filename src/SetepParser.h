#pragma once

#include "JimJamLexer.h"
#include <set>
#include <string>
#include <vector>

namespace SetepParser {

class SetepLexer : public JimJamLexer
{
public:
   void initKeywords();
   void getNext();
   std::string typeToStr(int type);
   enum { 
      endTok,
      keywordTok,
      whitespaceTok,
      lineCommentTok,
      upperNameTok,
      lowerNameTok,
      numberTok,
      puncTok,
      operatorTok
   };
};

extern SetepLexer setepLexer;

class Scope;

// card : number | lowerName ;
class Card
{
public:
   Card();
   ~Card();
   std::string lowerName_;
   std::string number_;
   size_t sourceStart_, sourceEnd_;
};

// name : upperName | lowerName;
class Name
{
public:
   Name();
   ~Name();
   std::string lowerName_;
   std::string upperName_;
   size_t sourceStart_, sourceEnd_;
};

// valueName : name [ "[" [card] "]" ] ;
class ValueName
{
public:
   ValueName();
   ~ValueName();
   Card* card_;
   Name* name_;
   std::set<std::string> strings_;
   size_t sourceStart_, sourceEnd_;
};

// value : valueName [ "(" ")" ] ["^"] ;
class Value
{
public:
   Value();
   ~Value();
   ValueName* valueName_;
   std::set<std::string> strings_;
   size_t sourceStart_, sourceEnd_;
};

// shapeDim : card>dim [ "[" card>split "]" ] ;
class ShapeDim
{
public:
   ShapeDim();
   ~ShapeDim();
   Card* dim_;
   Card* split_;
   std::set<std::string> strings_;
   size_t sourceStart_, sourceEnd_;
};

// shape : shapeDim { "%" shapeDim } ["@" lowerName] ;
class Shape
{
public:
   Shape();
   ~Shape();
   std::vector<ShapeDim*> shapeDim_;
   std::string lowerName_;
   std::set<std::string> strings_;
   size_t sourceStart_, sourceEnd_;
};

// group : value  { ":" value }  ;
class Group
{
public:
   Group();
   ~Group();
   std::vector<Value*> value_;
   size_t sourceStart_, sourceEnd_;
};

// transferOp : ">" ["+" ] [ ":" | "[" [card] "]" ] ">" ;
class TransferOp
{
public:
   TransferOp();
   ~TransferOp();
   Card* card_;
   std::set<std::string> strings_;
   size_t sourceStart_, sourceEnd_;
};

// scopeOp : "+" | "-" | "*" ;
class ScopeOp
{
public:
   ScopeOp();
   ~ScopeOp();
   std::set<std::string> strings_;
   size_t sourceStart_, sourceEnd_;
};

// scopeArg : scope | group ;
class ScopeArg
{
public:
   ScopeArg();
   ~ScopeArg();
   Group* group_;
   Scope* scope_;
   size_t sourceStart_, sourceEnd_;
};

// scopeExpr : [scopeOp>first] scopeArg {scopeOp  scopeArg} ;
class ScopeExpr
{
public:
   ScopeExpr();
   ~ScopeExpr();
   ScopeOp* first_;
   std::vector<ScopeArg*> scopeArg_;
   std::vector<ScopeOp*> scopeOp_;
   size_t sourceStart_, sourceEnd_;
};

// scopeSep : "|" | transferOp;
class ScopeSep
{
public:
   ScopeSep();
   ~ScopeSep();
   TransferOp* transferOp_;
   std::set<std::string> strings_;
   size_t sourceStart_, sourceEnd_;
};

// scopeStmt : scopeExpr ["=" shape] { scopeSep scopeExpr } ;
class ScopeStmt
{
public:
   ScopeStmt();
   ~ScopeStmt();
   std::vector<ScopeExpr*> scopeExpr_;
   std::vector<ScopeSep*> scopeSep_;
   Shape* shape_;
   std::set<std::string> strings_;
   size_t sourceStart_, sourceEnd_;
};

// scopeBody : scopeStmt {";" scopeStmt} ;
class ScopeBody
{
public:
   ScopeBody();
   ~ScopeBody();
   std::vector<ScopeStmt*> scopeStmt_;
   size_t sourceStart_, sourceEnd_;
};

// repDirection : "/" | "\\" ;
class RepDirection
{
public:
   RepDirection();
   ~RepDirection();
   std::set<std::string> strings_;
   size_t sourceStart_, sourceEnd_;
};

// repType : [card] ["[" "]"] ;
class RepType
{
public:
   RepType();
   ~RepType();
   Card* card_;
   std::set<std::string> strings_;
   size_t sourceStart_, sourceEnd_;
};

// repetition : repDirection repType ;
class Repetition
{
public:
   Repetition();
   ~Repetition();
   RepDirection* repDirection_;
   RepType* repType_;
   size_t sourceStart_, sourceEnd_;
};

// choice : "?" upperName>choiceComp ["[" "]"] "(" ")" ;
class Choice
{
public:
   Choice();
   ~Choice();
   std::string choiceComp_;
   std::set<std::string> strings_;
   size_t sourceStart_, sourceEnd_;
};

// scopeName : upperName>scopeName [ repetition ] [choice] ;
class ScopeName
{
public:
   ScopeName();
   ~ScopeName();
   Choice* choice_;
   Repetition* repetition_;
   std::string scopeName_;
   size_t sourceStart_, sourceEnd_;
};

// scope : [scopeName] "{" scopeBody "}" ;
class Scope
{
public:
   Scope();
   ~Scope();
   ScopeBody* scopeBody_;
   ScopeName* scopeName_;
   size_t sourceStart_, sourceEnd_;
};

// compDef : upperName ["(" [upperName {":" upperName}] ")"] "{" [scopeBody] "}" ;
class CompDef
{
public:
   CompDef();
   ~CompDef();
   ScopeBody* scopeBody_;
   std::vector<std::string> upperName_;
   std::set<std::string> strings_;
   size_t sourceStart_, sourceEnd_;
};

// setepModel : { compDef | scopeStmt } ;
class SetepModel
{
public:
   SetepModel();
   ~SetepModel();
   std::vector<CompDef*> compDef_;
   std::vector<ScopeStmt*> scopeStmt_;
   size_t sourceStart_, sourceEnd_;
};

}
