#include "SetepParser.h"
using namespace std;

namespace SetepParser {

SetepLexer setepLexer;

string SetepLexer::typeToStr(int type)
{
   switch (type)
   {
   case whitespaceTok:
      return ("`whitespaceTok");
   case lineCommentTok:
      return ("`lineCommentTok");
   case upperNameTok:
      return ("`upperNameTok");
   case lowerNameTok:
      return ("`lowerNameTok");
   case numberTok:
      return ("`numberTok");
   case puncTok:
      return ("`puncTok");
   case operatorTok:
      return ("`operatorTok");
   }
   return "`unknownTok";
}

void SetepLexer::getNext()
{
   // Copy next to cur
   curTok_ = nextTok_;
   curTokPos_ = nextTokPos_;
   curType_ = nextType_;
   // Skip whitespace and comments
   whitespace_ = "";
   comment_ = "";
   while (true)
   {
      initCheckPattern();
      // whitespace : [ \t\r\n]+ ;
      if (
         checkPattern(whitespace_, " \\t\\r\\n", patClass, req1N, true)
      )
         nextType_ = whitespaceTok;
      // lineComment : (//)[^\n]* ;
      else if (
         checkPattern(comment_, "//", patGrp, req1, true) &&
         checkPattern(comment_, "\\n", patNegClass, req0N)
      )
         nextType_ = lineCommentTok;
      else
         break;
   }
   nextTok_ = "";
   nextTokPos_ = textPos();
   initCheckPattern();
   // upperName : [A-Z][A-Z0-9_']*[!]? ;
   if (
      checkPattern(nextTok_, "A-Z", patClass, req1, true) &&
      checkPattern(nextTok_, "A-Z0-9_'", patClass, req0N) &&
      checkPattern(nextTok_, "!", patClass, req01)
   )
      nextType_ = upperNameTok;
   // lowerName : [a-z][a-z0-9_']*[!]? ;
   else if (
      checkPattern(nextTok_, "a-z", patClass, req1, true) &&
      checkPattern(nextTok_, "a-z0-9_'", patClass, req0N) &&
      checkPattern(nextTok_, "!", patClass, req01)
   )
      nextType_ = lowerNameTok;
   // number : [0-9]+ ;
   else if (
      checkPattern(nextTok_, "0-9", patClass, req1N, true)
   )
      nextType_ = numberTok;
   // punc : [\]\[\(\){}] ;
   else if (
      checkPattern(nextTok_, "][(){}", patClass, req1, true)
   )
      nextType_ = puncTok;
   // operator : [=:;/\\%@?|*+->\^] ;
   else if (
      checkPattern(nextTok_, "=:;/\\%@?|*+->^", patClass, req1, true)
   )
      nextType_ = operatorTok;
   else if (endOfText())
   {
      nextTok_ = "";
      nextType_ = endTok;
      nextTokPos_ = textPos();
      return;
   }
   else
   {
      addError(string("unhandled character : ") + getChar());
   }
   // If the token is a keyword set the type to keywordTok
   if (keywords_.find(nextTok_) != keywords_.end())
   {
         nextType_ = keywordTok;
   }
}

void SetepLexer::initKeywords()
{
   keywords_.add("%");
   keywords_.add("(");
   keywords_.add(")");
   keywords_.add("*");
   keywords_.add("+");
   keywords_.add("-");
   keywords_.add("/");
   keywords_.add(":");
   keywords_.add(";");
   keywords_.add("=");
   keywords_.add(">");
   keywords_.add("?");
   keywords_.add("@");
   keywords_.add("[");
   keywords_.add("\\");
   keywords_.add("]");
   keywords_.add("^");
   keywords_.add("{");
   keywords_.add("|");
   keywords_.add("}");
}

//************************ Card ***********************
// card : number | lowerName ;
Card::Card()
{
   sourceStart_ = setepLexer.curTokPos_;
   if (setepLexer.checkType(SetepLexer::numberTok))
   {
      setepLexer.getToken(SetepLexer::numberTok, number_);
      goto choiceEnd0;
   }
   if (setepLexer.checkType(SetepLexer::lowerNameTok))
   {
      setepLexer.getToken(SetepLexer::lowerNameTok, lowerName_);
      goto choiceEnd0;
   }
   // If no choice above is found it is an error
   setepLexer.addError("no valid choice found");
   // If choice above is found goto here
choiceEnd0: ;
   sourceEnd_ = setepLexer.curTokPos_;
}

Card::~Card()
{
}

//************************ Name ***********************
// name : upperName | lowerName;
Name::Name()
{
   sourceStart_ = setepLexer.curTokPos_;
   if (setepLexer.checkType(SetepLexer::upperNameTok))
   {
      setepLexer.getToken(SetepLexer::upperNameTok, upperName_);
      goto choiceEnd1;
   }
   if (setepLexer.checkType(SetepLexer::lowerNameTok))
   {
      setepLexer.getToken(SetepLexer::lowerNameTok, lowerName_);
      goto choiceEnd1;
   }
   // If no choice above is found it is an error
   setepLexer.addError("no valid choice found");
   // If choice above is found goto here
choiceEnd1: ;
   sourceEnd_ = setepLexer.curTokPos_;
}

Name::~Name()
{
}

//************************ ValueName ***********************
// valueName : name [ "[" [card] "]" ] ;
ValueName::ValueName()
{
   sourceStart_ = setepLexer.curTokPos_;
   card_ = nullptr;
   name_ = nullptr;
   static TokenSet tokens2("`lowerNameTok `numberTok");
   name_ = new Name();
   if (setepLexer.checkToken("["))
   {
      setepLexer.require("[");
      strings_.insert("[");
      if (setepLexer.checkType(SetepLexer::lowerNameTok) || setepLexer.checkType(SetepLexer::numberTok))
      {
         card_ = new Card();
      }
      setepLexer.require("]");
      strings_.insert("]");
   }
   sourceEnd_ = setepLexer.curTokPos_;
}

ValueName::~ValueName()
{
   if (card_ != nullptr)
      delete card_; 
   if (name_ != nullptr)
      delete name_; 
}

//************************ Value ***********************
// value : valueName [ "(" ")" ] ["^"] ;
Value::Value()
{
   sourceStart_ = setepLexer.curTokPos_;
   valueName_ = nullptr;
   valueName_ = new ValueName();
   if (setepLexer.checkToken("("))
   {
      setepLexer.require("(");
      strings_.insert("(");
      setepLexer.require(")");
      strings_.insert(")");
   }
   if (setepLexer.checkToken("^"))
   {
      setepLexer.require("^");
      strings_.insert("^");
   }
   sourceEnd_ = setepLexer.curTokPos_;
}

Value::~Value()
{
   if (valueName_ != nullptr)
      delete valueName_; 
}

//************************ ShapeDim ***********************
// shapeDim : card>dim [ "[" card>split "]" ] ;
ShapeDim::ShapeDim()
{
   sourceStart_ = setepLexer.curTokPos_;
   dim_ = nullptr;
   split_ = nullptr;
   dim_ = new Card();
   if (setepLexer.checkToken("["))
   {
      setepLexer.require("[");
      strings_.insert("[");
      split_ = new Card();
      setepLexer.require("]");
      strings_.insert("]");
   }
   sourceEnd_ = setepLexer.curTokPos_;
}

ShapeDim::~ShapeDim()
{
   if (dim_ != nullptr)
      delete dim_; 
   if (split_ != nullptr)
      delete split_; 
}

//************************ Shape ***********************
// shape : shapeDim { "%" shapeDim } ["@" lowerName] ;
Shape::Shape()
{
   sourceStart_ = setepLexer.curTokPos_;
   shapeDim_.push_back(new ShapeDim());
   while (setepLexer.checkToken("%"))
   {
      setepLexer.require("%");
      shapeDim_.push_back(new ShapeDim());
   }
   if (setepLexer.checkToken("@"))
   {
      setepLexer.require("@");
      strings_.insert("@");
      setepLexer.getToken(SetepLexer::lowerNameTok, lowerName_);
   }
   sourceEnd_ = setepLexer.curTokPos_;
}

Shape::~Shape()
{
   for (ShapeDim* shapeDim : shapeDim_)
   {
      delete shapeDim;
   }
}

//************************ Group ***********************
// group : value  { ":" value }  ;
Group::Group()
{
   sourceStart_ = setepLexer.curTokPos_;
   value_.push_back(new Value());
   while (setepLexer.checkToken(":"))
   {
      setepLexer.require(":");
      value_.push_back(new Value());
   }
   sourceEnd_ = setepLexer.curTokPos_;
}

Group::~Group()
{
   for (Value* value : value_)
   {
      delete value;
   }
}

//************************ TransferOp ***********************
// transferOp : ">" ["+" ] [ ":" | "[" [card] "]" ] ">" ;
TransferOp::TransferOp()
{
   sourceStart_ = setepLexer.curTokPos_;
   card_ = nullptr;
   static TokenSet tokens2(": [");
   static TokenSet tokens5("`lowerNameTok `numberTok");
   try
   {
      setepLexer.require(">");
      if (setepLexer.checkToken("+"))
      {
         setepLexer.require("+");
         strings_.insert("+");
      }
      if (setepLexer.checkToken(":") || setepLexer.checkToken("["))
      {
         if (setepLexer.checkToken(":"))
         {
            setepLexer.require(":");
            strings_.insert(":");
            goto choiceEnd2;
         }
         if (setepLexer.checkToken("["))
         {
            setepLexer.require("[");
            strings_.insert("[");
            if (setepLexer.checkType(SetepLexer::lowerNameTok) || setepLexer.checkType(SetepLexer::numberTok))
            {
               card_ = new Card();
            }
            setepLexer.require("]");
            strings_.insert("]");
            goto choiceEnd2;
         }
         // If no choice above is found it is an error
         setepLexer.addError("no valid choice found");
         // If choice above is found goto here
choiceEnd2: ;
      }
      setepLexer.require(">");
   }
   catch (const JimJamException&)
   {
      setepLexer.skipToTok(">"); 
   }
   sourceEnd_ = setepLexer.curTokPos_;
}

TransferOp::~TransferOp()
{
   if (card_ != nullptr)
      delete card_; 
}

//************************ ScopeOp ***********************
// scopeOp : "+" | "-" | "*" ;
ScopeOp::ScopeOp()
{
   sourceStart_ = setepLexer.curTokPos_;
   if (setepLexer.checkToken("+"))
   {
      setepLexer.require("+");
      strings_.insert("+");
      goto choiceEnd3;
   }
   if (setepLexer.checkToken("-"))
   {
      setepLexer.require("-");
      strings_.insert("-");
      goto choiceEnd3;
   }
   if (setepLexer.checkToken("*"))
   {
      setepLexer.require("*");
      strings_.insert("*");
      goto choiceEnd3;
   }
   // If no choice above is found it is an error
   setepLexer.addError("no valid choice found");
   // If choice above is found goto here
choiceEnd3: ;
   sourceEnd_ = setepLexer.curTokPos_;
}

ScopeOp::~ScopeOp()
{
}

//************************ ScopeArg ***********************
// scopeArg : scope | group ;
ScopeArg::ScopeArg()
{
   sourceStart_ = setepLexer.curTokPos_;
   group_ = nullptr;
   scope_ = nullptr;
   static TokenSet tokens1("`upperNameTok {");
   static TokenSet tokens2("`lowerNameTok `upperNameTok");
   if (setepLexer.checkTypeAndToken(SetepLexer::upperNameTok, "/") || setepLexer.checkTypeAndToken(SetepLexer::upperNameTok, "?") || setepLexer.checkTypeAndToken(SetepLexer::upperNameTok, "\\") || setepLexer.checkTypeAndToken(SetepLexer::upperNameTok, "{") || setepLexer.checkTokens("{", "*") || setepLexer.checkTokens("{", "+") || setepLexer.checkTokens("{", "-") || setepLexer.checkTokenAndType("{", SetepLexer::lowerNameTok) || setepLexer.checkTokenAndType("{", SetepLexer::upperNameTok) || setepLexer.checkTokens("{", "{"))
   {
      scope_ = new Scope();
      goto choiceEnd4;
   }
   if (setepLexer.checkType(SetepLexer::lowerNameTok) || setepLexer.checkType(SetepLexer::upperNameTok))
   {
      group_ = new Group();
      goto choiceEnd4;
   }
   // If no choice above is found it is an error
   setepLexer.addError("no valid choice found");
   // If choice above is found goto here
choiceEnd4: ;
   sourceEnd_ = setepLexer.curTokPos_;
}

ScopeArg::~ScopeArg()
{
   if (group_ != nullptr)
      delete group_; 
   if (scope_ != nullptr)
      delete scope_; 
}

//************************ ScopeExpr ***********************
// scopeExpr : [scopeOp>first] scopeArg {scopeOp  scopeArg} ;
ScopeExpr::ScopeExpr()
{
   sourceStart_ = setepLexer.curTokPos_;
   first_ = nullptr;
   static TokenSet tokens1("* + -");
   static TokenSet tokens2("* + -");
   if (setepLexer.checkToken("*") || setepLexer.checkToken("+") || setepLexer.checkToken("-"))
   {
      first_ = new ScopeOp();
   }
   scopeArg_.push_back(new ScopeArg());
   while (setepLexer.checkToken("*") || setepLexer.checkToken("+") || setepLexer.checkToken("-"))
   {
      scopeOp_.push_back(new ScopeOp());
      scopeArg_.push_back(new ScopeArg());
   }
   sourceEnd_ = setepLexer.curTokPos_;
}

ScopeExpr::~ScopeExpr()
{
   if (first_ != nullptr)
      delete first_; 
   for (ScopeArg* scopeArg : scopeArg_)
   {
      delete scopeArg;
   }
   for (ScopeOp* scopeOp : scopeOp_)
   {
      delete scopeOp;
   }
}

//************************ ScopeSep ***********************
// scopeSep : "|" | transferOp;
ScopeSep::ScopeSep()
{
   sourceStart_ = setepLexer.curTokPos_;
   transferOp_ = nullptr;
   if (setepLexer.checkTokens(">", "+") || setepLexer.checkTokens(">", ":") || setepLexer.checkTokens(">", ">") || setepLexer.checkTokens(">", "["))
   {
      transferOp_ = new TransferOp();
      goto choiceEnd5;
   }
   if (setepLexer.checkToken("|"))
   {
      setepLexer.require("|");
      strings_.insert("|");
      goto choiceEnd5;
   }
   // If no choice above is found it is an error
   setepLexer.addError("no valid choice found");
   // If choice above is found goto here
choiceEnd5: ;
   sourceEnd_ = setepLexer.curTokPos_;
}

ScopeSep::~ScopeSep()
{
   if (transferOp_ != nullptr)
      delete transferOp_; 
}

//************************ ScopeStmt ***********************
// scopeStmt : scopeExpr ["=" shape] { scopeSep scopeExpr } ;
ScopeStmt::ScopeStmt()
{
   sourceStart_ = setepLexer.curTokPos_;
   shape_ = nullptr;
   static TokenSet tokens2("> |");
   scopeExpr_.push_back(new ScopeExpr());
   if (setepLexer.checkToken("="))
   {
      setepLexer.require("=");
      strings_.insert("=");
      shape_ = new Shape();
   }
   while (setepLexer.checkTokens(">", "+") || setepLexer.checkTokens(">", ":") || setepLexer.checkTokens(">", ">") || setepLexer.checkTokens(">", "[") || setepLexer.checkToken("|"))
   {
      scopeSep_.push_back(new ScopeSep());
      scopeExpr_.push_back(new ScopeExpr());
   }
   sourceEnd_ = setepLexer.curTokPos_;
}

ScopeStmt::~ScopeStmt()
{
   for (ScopeExpr* scopeExpr : scopeExpr_)
   {
      delete scopeExpr;
   }
   for (ScopeSep* scopeSep : scopeSep_)
   {
      delete scopeSep;
   }
   if (shape_ != nullptr)
      delete shape_; 
}

//************************ ScopeBody ***********************
// scopeBody : scopeStmt {";" scopeStmt} ;
ScopeBody::ScopeBody()
{
   sourceStart_ = setepLexer.curTokPos_;
   scopeStmt_.push_back(new ScopeStmt());
   while (setepLexer.checkToken(";"))
   {
      setepLexer.require(";");
      scopeStmt_.push_back(new ScopeStmt());
   }
   sourceEnd_ = setepLexer.curTokPos_;
}

ScopeBody::~ScopeBody()
{
   for (ScopeStmt* scopeStmt : scopeStmt_)
   {
      delete scopeStmt;
   }
}

//************************ RepDirection ***********************
// repDirection : "/" | "\\" ;
RepDirection::RepDirection()
{
   sourceStart_ = setepLexer.curTokPos_;
   if (setepLexer.checkToken("/"))
   {
      setepLexer.require("/");
      strings_.insert("/");
      goto choiceEnd6;
   }
   if (setepLexer.checkToken("\\"))
   {
      setepLexer.require("\\");
      strings_.insert("\\");
      goto choiceEnd6;
   }
   // If no choice above is found it is an error
   setepLexer.addError("no valid choice found");
   // If choice above is found goto here
choiceEnd6: ;
   sourceEnd_ = setepLexer.curTokPos_;
}

RepDirection::~RepDirection()
{
}

//************************ RepType ***********************
// repType : [card] ["[" "]"] ;
RepType::RepType()
{
   sourceStart_ = setepLexer.curTokPos_;
   card_ = nullptr;
   static TokenSet tokens1("`lowerNameTok `numberTok");
   if (setepLexer.checkType(SetepLexer::lowerNameTok) || setepLexer.checkType(SetepLexer::numberTok))
   {
      card_ = new Card();
   }
   if (setepLexer.checkToken("["))
   {
      setepLexer.require("[");
      strings_.insert("[");
      setepLexer.require("]");
      strings_.insert("]");
   }
   sourceEnd_ = setepLexer.curTokPos_;
}

RepType::~RepType()
{
   if (card_ != nullptr)
      delete card_; 
}

//************************ Repetition ***********************
// repetition : repDirection repType ;
Repetition::Repetition()
{
   sourceStart_ = setepLexer.curTokPos_;
   repDirection_ = nullptr;
   repType_ = nullptr;
   repDirection_ = new RepDirection();
   repType_ = new RepType();
   sourceEnd_ = setepLexer.curTokPos_;
}

Repetition::~Repetition()
{
   if (repDirection_ != nullptr)
      delete repDirection_; 
   if (repType_ != nullptr)
      delete repType_; 
}

//************************ Choice ***********************
// choice : "?" upperName>choiceComp ["[" "]"] "(" ")" ;
Choice::Choice()
{
   sourceStart_ = setepLexer.curTokPos_;
   try
   {
      setepLexer.require("?");
      setepLexer.getToken(SetepLexer::upperNameTok, choiceComp_);
      if (setepLexer.checkToken("["))
      {
         setepLexer.require("[");
         strings_.insert("[");
         setepLexer.require("]");
         strings_.insert("]");
      }
      setepLexer.require("(");
      setepLexer.require(")");
   }
   catch (const JimJamException&)
   {
      setepLexer.skipToTok(")"); 
   }
   sourceEnd_ = setepLexer.curTokPos_;
}

Choice::~Choice()
{
}

//************************ ScopeName ***********************
// scopeName : upperName>scopeName [ repetition ] [choice] ;
ScopeName::ScopeName()
{
   sourceStart_ = setepLexer.curTokPos_;
   choice_ = nullptr;
   repetition_ = nullptr;
   static TokenSet tokens1("/ \\");
   setepLexer.getToken(SetepLexer::upperNameTok, scopeName_);
   if (setepLexer.checkToken("/") || setepLexer.checkToken("\\"))
   {
      repetition_ = new Repetition();
   }
   if (setepLexer.checkTokenAndType("?", SetepLexer::upperNameTok))
   {
      choice_ = new Choice();
   }
   sourceEnd_ = setepLexer.curTokPos_;
}

ScopeName::~ScopeName()
{
   if (choice_ != nullptr)
      delete choice_; 
   if (repetition_ != nullptr)
      delete repetition_; 
}

//************************ Scope ***********************
// scope : [scopeName] "{" scopeBody "}" ;
Scope::Scope()
{
   sourceStart_ = setepLexer.curTokPos_;
   scopeBody_ = nullptr;
   scopeName_ = nullptr;
   try
   {
      if (setepLexer.checkType(SetepLexer::upperNameTok))
      {
         scopeName_ = new ScopeName();
      }
      setepLexer.require("{");
      scopeBody_ = new ScopeBody();
      setepLexer.require("}");
   }
   catch (const JimJamException&)
   {
      setepLexer.skipToTok("}"); 
   }
   sourceEnd_ = setepLexer.curTokPos_;
}

Scope::~Scope()
{
   if (scopeBody_ != nullptr)
      delete scopeBody_; 
   if (scopeName_ != nullptr)
      delete scopeName_; 
}

//************************ CompDef ***********************
// compDef : upperName ["(" [upperName {":" upperName}] ")"] "{" [scopeBody] "}" ;
CompDef::CompDef()
{
   sourceStart_ = setepLexer.curTokPos_;
   scopeBody_ = nullptr;
   static TokenSet tokens4("* + - `lowerNameTok `upperNameTok {");
   try
   {
      setepLexer.getToken(SetepLexer::upperNameTok, upperName_.emplace_back());
      if (setepLexer.checkToken("("))
      {
         setepLexer.require("(");
         strings_.insert("(");
         if (setepLexer.checkType(SetepLexer::upperNameTok))
         {
            setepLexer.getToken(SetepLexer::upperNameTok, upperName_.emplace_back());
            while (setepLexer.checkToken(":"))
            {
               setepLexer.require(":");
               strings_.insert(":");
               setepLexer.getToken(SetepLexer::upperNameTok, upperName_.emplace_back());
            }
         }
         setepLexer.require(")");
         strings_.insert(")");
      }
      setepLexer.require("{");
      if (setepLexer.checkTokenAndType("*", SetepLexer::lowerNameTok) || setepLexer.checkTokenAndType("*", SetepLexer::upperNameTok) || setepLexer.checkTokens("*", "{") || setepLexer.checkTokenAndType("+", SetepLexer::lowerNameTok) || setepLexer.checkTokenAndType("+", SetepLexer::upperNameTok) || setepLexer.checkTokens("+", "{") || setepLexer.checkTokenAndType("-", SetepLexer::lowerNameTok) || setepLexer.checkTokenAndType("-", SetepLexer::upperNameTok) || setepLexer.checkTokens("-", "{") || setepLexer.checkType(SetepLexer::lowerNameTok) || setepLexer.checkType(SetepLexer::upperNameTok) || setepLexer.checkTokens("{", "*") || setepLexer.checkTokens("{", "+") || setepLexer.checkTokens("{", "-") || setepLexer.checkTokenAndType("{", SetepLexer::lowerNameTok) || setepLexer.checkTokenAndType("{", SetepLexer::upperNameTok) || setepLexer.checkTokens("{", "{"))
      {
         scopeBody_ = new ScopeBody();
      }
      setepLexer.require("}");
   }
   catch (const JimJamException&)
   {
      setepLexer.skipToTok("}"); 
   }
   sourceEnd_ = setepLexer.curTokPos_;
}

CompDef::~CompDef()
{
   if (scopeBody_ != nullptr)
      delete scopeBody_; 
}

//************************ SetepModel ***********************
// setepModel : { compDef | scopeStmt } ;
SetepModel::SetepModel()
{
   sourceStart_ = setepLexer.curTokPos_;
   static TokenSet tokens1("* + - `lowerNameTok `upperNameTok {");
   static TokenSet tokens3("* + - `lowerNameTok `upperNameTok {");
   while (setepLexer.checkTokenAndType("*", SetepLexer::lowerNameTok) || setepLexer.checkTokenAndType("*", SetepLexer::upperNameTok) || setepLexer.checkTokens("*", "{") || setepLexer.checkTokenAndType("+", SetepLexer::lowerNameTok) || setepLexer.checkTokenAndType("+", SetepLexer::upperNameTok) || setepLexer.checkTokens("+", "{") || setepLexer.checkTokenAndType("-", SetepLexer::lowerNameTok) || setepLexer.checkTokenAndType("-", SetepLexer::upperNameTok) || setepLexer.checkTokens("-", "{") || setepLexer.checkType(SetepLexer::lowerNameTok) || setepLexer.checkType(SetepLexer::upperNameTok) || setepLexer.checkTokens("{", "*") || setepLexer.checkTokens("{", "+") || setepLexer.checkTokens("{", "-") || setepLexer.checkTokenAndType("{", SetepLexer::lowerNameTok) || setepLexer.checkTokenAndType("{", SetepLexer::upperNameTok) || setepLexer.checkTokens("{", "{"))
   {
      if (setepLexer.checkTypeAndToken(SetepLexer::upperNameTok, "(") || setepLexer.checkTypeAndToken(SetepLexer::upperNameTok, "{"))
      {
         compDef_.push_back(new CompDef());
         goto choiceEnd7;
      }
      if (setepLexer.checkTokenAndType("*", SetepLexer::lowerNameTok) || setepLexer.checkTokenAndType("*", SetepLexer::upperNameTok) || setepLexer.checkTokens("*", "{") || setepLexer.checkTokenAndType("+", SetepLexer::lowerNameTok) || setepLexer.checkTokenAndType("+", SetepLexer::upperNameTok) || setepLexer.checkTokens("+", "{") || setepLexer.checkTokenAndType("-", SetepLexer::lowerNameTok) || setepLexer.checkTokenAndType("-", SetepLexer::upperNameTok) || setepLexer.checkTokens("-", "{") || setepLexer.checkType(SetepLexer::lowerNameTok) || setepLexer.checkType(SetepLexer::upperNameTok) || setepLexer.checkTokens("{", "*") || setepLexer.checkTokens("{", "+") || setepLexer.checkTokens("{", "-") || setepLexer.checkTokenAndType("{", SetepLexer::lowerNameTok) || setepLexer.checkTokenAndType("{", SetepLexer::upperNameTok) || setepLexer.checkTokens("{", "{"))
      {
         scopeStmt_.push_back(new ScopeStmt());
         goto choiceEnd7;
      }
      // If no choice above is found it is an error
      setepLexer.addError("no valid choice found");
      // If choice above is found goto here
choiceEnd7: ;
   }
   sourceEnd_ = setepLexer.curTokPos_;
}

SetepModel::~SetepModel()
{
   for (CompDef* compDef : compDef_)
   {
      delete compDef;
   }
   for (ScopeStmt* scopeStmt : scopeStmt_)
   {
      delete scopeStmt;
   }
}

}
