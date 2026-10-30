#pragma once

#ifdef WIN32
#include <string>
#else
#include <string.h>
#endif
#include <set>
#include <map>
#include <vector>
#include <sstream>

#ifdef WIN32
#define for_tok(str, seps, tok) char* dup##tok = _strdup(str); char* context##tok = nullptr; for (const char* tok = strtok_s(dup##tok, seps, &context##tok); tok ? true : (free(dup##tok), false); tok = strtok_s(NULL, seps, &context##tok))
#else
#define for_tok(str, seps, tok) char* dup##tok = strdup(str); char* context##tok = nullptr; for (const char* tok = strtok_r(dup##tok, seps, &context##tok); tok ? true : (free(dup##tok),false) ; tok = strtok_r(NULL, seps, &context##tok))
#endif
// Class for set of tokens
class TokenSet : public std::set<std::string>
{
public:
   TokenSet() {}
   TokenSet(const char* toks)
   {
      for_tok(toks, " ", tok)
         insert(tok);
   }
   void add(const TokenSet& tset)
   {
      insert(tset.begin(), tset.end());
   }
   std::string out()
   {
      std::stringstream ss;
      for (std::string tok : *this)
      {
         if (!ss.str().empty())
            ss << " ";
         ss << tok;
      }
      return ss.str();
   }
};

// Error class for lexer/parser errors
struct JimJamError
{
   // Constructor
   JimJamError(const std::string& errMsg, size_t lineNum, const std::string line)
   {
      errMsg_ = errMsg;
      lineNum_ = lineNum;
      line_ = line;
   }
   // Member variables
   std::string errMsg_;
   int lineNum_;
   std::string line_;
};

// Typedef for vector of errors
typedef std::vector<JimJamError> JimJamErrorVect;

// Exception for errors in parsing
class JimJamException : public std::exception
{
};

// Exception for errors in parsing
class JimJamOptionException : public std::exception
{
};

// Base class for Lexer
class JimJamLexer
{
public:
   JimJamLexer();
   ~JimJamLexer();
   void init(const char* textStart, const char* textEnd = NULL);
   void advance();
   void retreat();
   enum { req1, req01, req0N, req1N };
   enum { patClass, patNegClass, patGrp, patNegGrp };
   void initCheckPattern();
   bool checkPattern(std::string& tok, const std::string& pat, int patType, int count, bool clear = false);
   bool endOfText() { return textPtr_ >= textEnd_; }
   size_t textPos() { return textPtr_ - textStart_; }
   char getChar(int offset = 0) {
      if (textPtr_ + offset >= textEnd_)
         return 0;
      return *(textPtr_ + offset); 
   }
   virtual void getNext() = 0;
   virtual void initKeywords() = 0;
   bool checkToken(const std::string& expected);
   bool checkTokens(const std::string& expected1, const std::string& expected2);
   bool checkTokens(const std::string& expected);
   bool checkType(int type);
   bool checkTypes(int type1, int type2);
   bool checkTypeAndToken(int type, const std::string& expected);
   bool checkTokenAndType(const std::string& expected, int type);
   bool checkSet(const TokenSet& allowed);
   void requireSet(const TokenSet& expected); // , std::string& tok);
   void require(const std::string& expected);
   void requireType(int type); // , std::string& tok);
   virtual std::string typeToStr(int type) = 0;
   void getToken(int type, std::string& tok);
   //void getName(std::string& name, bool required = true);
   //void getValue(double& value, bool required = true);
   //void getLiteral(std::string& text);
   bool skipToChar(char ch);
   void skipToTok(const std::string& tok);
   std::string source(size_t start, size_t end);
   std::string curLine();
   void addError(const std::string& errMsg);
   void saveState();
   void restoreState();
   void discardState();
   // Member variables
   TokenSet keywords_;
   const char* textStart_;
   const char* textEnd_;
   const char* textPtr_;
   const char* lexPtr_;
   //char nextCh_;
   //char nextNextCh_;
   std::string curTok_;
   std::string nextTok_;
   std::string whitespace_;
   std::string comment_;
   int curType_;
   int nextType_;
   size_t curTokPos_;
   size_t nextTokPos_;
   size_t lineNum_;
   const char* lineStart_;
   JimJamErrorVect errors_;
   // State variables
   bool stateSaved_;
   const char* savedTextPtr_;
   char savedNextCh_;
   char savedNextNextCh_;
   std::string savedCurTok_;
   std::string savedWhitespace_;
   std::string savedComment_;
   int savedCurType_;
   size_t savedCurTokPos_;
   size_t savedLineNum_;
   const char* savedLineStart_;
   JimJamErrorVect savedErrors_;
};
