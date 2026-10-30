#include "JimJamLexer.h"
#include <sstream>
#include <map>
using namespace std;


// Class Implementation : JimJam Lexer
// 

// Constructor
JimJamLexer::JimJamLexer() 
{ 
   lineNum_ = 0; 
   textStart_ = textEnd_ = textPtr_ = lineStart_ = lexPtr_ = NULL; 
   stateSaved_ = false;
}

// Destructor
JimJamLexer::~JimJamLexer() 
{
}

// Init function - sets text, finds first token
void JimJamLexer::init(const char* textStart, const char* textEnd)
{
   // Initialize keywords - Virtual from derived class
   initKeywords();
   // Set the text for the lexer
   textPtr_ = textStart_ = textStart;
   textEnd_ = textEnd;
   lineStart_ = textStart_;
   curTokPos_ = nextTokPos_ = 0;
   curType_ = nextType_ = 0;
   // Get the cur and next token
   getNext();
   getNext();
}

// Advance gets the next character
void JimJamLexer::advance()
{
    // If at end of text return 0
  if (textPtr_ >= textEnd_)
     return;
   // advance
   textPtr_++;
   // If line end increment line number and set line start
   if (*textPtr_ == '\n')
   {
      lineNum_++;
      lineStart_ = textPtr_+1 ;
   }
 }

void JimJamLexer::retreat()
{
   if (textPtr_ > textStart_)
      textPtr_--;
}

void JimJamLexer::initCheckPattern()
{
   lexPtr_ = textPtr_;
}

bool JimJamLexer::checkPattern(std::string& tok, const string& pat, int patType, int req, bool clear)
{
   bool first = true;
   if (endOfText())
      return false;
   // For groups, compare to text
   if (patType == patGrp)
   {
      while (true)
      {
         // If not found
         bool match = true;
         string matchStr;
         const char* startPtr = textPtr_;
         for (size_t i = 0; i < pat.size(); i++)
         {
            char curCh = getChar();
            // If escape in jim being parsed
            if (curCh == '\\')
            {
               char nextCh = getChar(1);
               // If not whitespace get next char
               if (nextCh != 'n' && nextCh != 'r' && nextCh != 't' && nextCh != 's')
               {
                  advance();
                  nextCh = getChar();
              }
            }
            char ch = pat[i];
            // If escacped in pattern convert to actual
            if (ch == '\\' && i < pat.size() - 1)
            {
               ch = pat[i++];
               if (ch == 'n')
                  ch = '\n';
               else if (ch == 'r')
                  ch = '\r';
               else if (ch == 't')
                  ch = '\t';
               else if (ch == 's')
                  ch = ' ';
               else
                  bool shouldNotHappen = true; // Because slashers replaced in lexer code
            }
            // If a match
            if (ch == curCh)
            {
               matchStr += ch;
               advance();
            }
            else
            {
               match = false;
               break;
            }
         }
         if (!match)
         {
            // If it didn't match restore ptr
            textPtr_ = startPtr;
            // If this is the first time through
            if (first)
            {
               // If one or more expected it failed
               if (req == req1 || req == req1N)
               {
                  textPtr_ = lexPtr_;
                  return false;
               }
               // If zero or more expected it passes
               if (req == req0N)
                  return true;
            }
            // not the first so one found so everything else passes
            return true;
         }
         // Add to the current token
         tok += matchStr;
         // If only looking for zero or one or one it passes
         if (req == req1 || req == req01)
            return true;
         // Otherwise see if there is another
         first = false;
      }
   }
   else if (patType == patNegGrp)
   {
      while (true)
      {
         // If not found
         const char* matchPtr = textPtr_;
         for (size_t i = 0; i < pat.size(); i++)
         {
            // If escape in jim being parsed
            if (*matchPtr == '\\')
            {
               matchPtr++;
               // If whitespace leave as is
               if (*matchPtr == 'n' || *matchPtr == 'r' || *matchPtr == 't' || *matchPtr == 's')
                  matchPtr--;
               // Otherwise just use character after slash without slash
            }
            char ch = pat[i];
            // If escacped in pattern convert to actual
            if (ch == '\\' && i < pat.size() - 1)
            {
               ch = pat[i++];
               if (ch == 'n')
                  ch = '\n';
               else if (ch == 'r')
                  ch = '\r';
               else if (ch == 't')
                  ch = '\t';
               else if (ch == 's')
                  ch = ' ';
               else
                  bool shouldNotHappen = true; // Because slashers replaced in lexer code
            }
            // If a match
            if (ch == *matchPtr)
            {
               matchPtr++;
            }
            else
            {
               matchPtr = nullptr;
               break;
            }
         }
         // If it didn't match add the last character
         if (matchPtr == nullptr)
         {
            tok += getChar();
            advance();
         }
         // If it did match we are done 
         else
            return true;
         // Add to the current character
      }
   }
   // Char class
   else if (patType == patClass || patType == patNegClass)
   {
      while (true)
      {
         // If escape in jim being parsed
         bool escaped = false;
         char curCh = getChar();
         // SETEP source characters are literal; escapes belong to the pattern.
         // Look at all chars in class
         bool neg = patType == patNegClass;
         bool matched = false;
         if (neg && escaped)
         {
            if (curCh == '-')
               tok += "\\\\"; 
            tok += curCh;
            advance();
            if (req == req1 || req == req01)
               return true;
            matched = true;
         }
         else
         {
            size_t size = pat.size();
            // Go through chars in class
            for (size_t i = 0; i < size; i++)
            {
               char ch = pat[i];
               // Handle range like a-z
               if (ch != '\\' && i < size - 2 && pat[i + 1] == '-')
               {
                  i += 2;
                  char ch1 = pat[i];
                  // Match found
                  if (ch <= curCh && curCh <= ch1)
                  {
                     if (neg)
                     {
                        if (req == req1 || (first && req == req1N))
                        {
                           textPtr_ = lexPtr_;
                           return false;
                        }
                        return true;
                     }
                     // Match so add to token
                     if (escaped && curCh == '-')
                        tok += "\\\\";
                     tok += curCh;
                     advance();
                     if (req == req1 || req == req01)
                        return true;
                     matched = true;
                     break;
                  }
               }
               // Zero never matches
               else if (endOfText())
                  break;
               // Handle single char match
               else
               {
                  // Handle escape char like \]
                  if (ch == '\\' && i < size - 1 &&
                      (pat[i+1]=='n' || pat[i+1]=='r' || pat[i+1]=='t' ||
                       pat[i+1]=='s' || pat[i+1]=='\\' || pat[i+1]==']' || pat[i+1]=='-'))
                  {
                     i++;
                     ch = pat[i];
                     if (ch == 'n')
                        ch = '\n';
                     else if (ch == 'r')
                        ch = '\r';
                     else if (ch == 't')
                        ch = '\t';
                     else if (ch == 's')
                        ch = ' ';
                     else
                        bool shouldNotHappen = true;
                  }
                  if (curCh == ch)
                  {
                     if (neg)
                     {
                        if (req == req1 || (first && req == req1N))
                        {
                           textPtr_ = lexPtr_;
                           return false;
                        }
                        return true;
                     }
                     // Match so add to token
                     tok += curCh;
                     advance();
                     if (req == req1 || req == req01)
                        return true;
                     matched = true;
                     break;
                  }
               }
            }
         }
         if (matched)
            ;
         // Last did not match
         else if (neg)
         {
            // Didn't match so add to tok
            tok += curCh;
            advance();
            // If zero or one or one expected pass
            if (req == req1 || req == req01)
               return true;
            // if (req == req0N || req == req1N)
            //    continue;
         }
         else
         {
            // If no match must exit
            if (first && (req == req1 || req == req1N))
            {
               textPtr_ = lexPtr_;
               return false;
            }
            // if (req == req01 || req == req1N)
            // If no match but last escaped, add back
            if (escaped)
               retreat();
            return true;
         }
         first = false;
      }
   }
   // Should never get here
   textPtr_ = lexPtr_;
   return false;
}

// Get the current line with a pointer to the current position marked by ^^^
string JimJamLexer::curLine()
{
   const char* linePtr = lineStart_;
   string line;
   const char* tokStart = textStart_ + curTokPos_;
   while (*linePtr != '\n' && *linePtr != 0)
   {
      if (linePtr == tokStart)
         line += "^^^";
      line += *linePtr;
      linePtr++;
   }
   return line;
}

// Add an error to the error list and throw an exception
// Parser will put catch when there is a final token to a rule
void JimJamLexer::addError(const std::string& errMsg)
{
   string line = curLine();
   errors_.push_back(JimJamError(errMsg, lineNum_, line));
   if (stateSaved_)
      throw JimJamOptionException();
   else
      throw JimJamException();
}

// Skip tokens until a certain token is found
// Used in error recovery
void JimJamLexer::skipToTok(const string& tok)
{
   while (curTok_ != tok && *textPtr_ != 0)
   {
      getNext();
   }
   if (*textPtr_ == 0)
   {
      addError("cannot recover");
      throw JimJamException();
   }
   getNext();
}

// Get the source starting at start and ending at end
string JimJamLexer::source(size_t start, size_t end)
{
   string src = string(textStart_ + start, end - start);
   // Remove leading white space
   size_t wsPos = src.find_first_not_of(" \r\n\t");
   if (wsPos != string::npos)
      src = src.substr(wsPos);
   // Remove trailing white space
   wsPos = src.find_last_not_of(" \r\n\t");
   if (wsPos != string::npos)
      src.resize(wsPos+1);
   return src;
}

// Compare token against an expected token
bool JimJamLexer::checkToken(const string& expected)
{
  return expected == curTok_;
}

// Compare tokens against an expected tokens
bool JimJamLexer::checkTokens(const string& expected1, const string& expected2)
{
  return expected1 == curTok_ && expected2 == nextTok_;
}

// See if a token is in a string of space separated tokens
bool JimJamLexer::checkTokens(const string& expected)
{
   string curType = typeToStr(curType_);
   // Single token, just check it
   if (expected.find(' ') == string::npos)
   {
      if (expected == curTok_ || expected == curType)
         return true;
      return false;
   }
   // More than one, look at each
   for_tok(expected.c_str(), " ", tok)
   {
      if (tok == curTok_ || tok == curType)
      {
         return true;
      }
   }
   return false;
}

// Check currrent type against a type
bool JimJamLexer::checkType(int type)
{
   return curType_ == type;
}

bool JimJamLexer::checkTypeAndToken(int type, const std::string& expected)
{
  return curType_ == type && expected == nextTok_;

}

bool JimJamLexer::checkTypes(int type1, int type2)
{
  return curType_ == type1 && nextType_ == type2;

}

bool JimJamLexer::checkTokenAndType(const std::string& expected, int type)
{
  return  expected == curTok_ && nextType_ == type;
}


// Check current token against a set of tokens
bool JimJamLexer::checkSet(const TokenSet& allowed)
{
   return (allowed.find(curTok_) != allowed.end()) ||
      (allowed.find(typeToStr(curType_)) != allowed.end());
}

// Require current token to be in a set of tokens
void JimJamLexer::requireSet(const TokenSet& expected) //, string& tok)
{
   if (expected.find(curTok_) == expected.end())
   {
      string exp;
      for (string s : expected)
      {
         exp += s + "/";
      }
      addError("expected \"" + exp + "\" found \"" + curTok_ + "\"");
   }
   // tok = curTok_;
   getNext();
}

// Require current token be an expected token
void JimJamLexer::require(const string& expected)
{
   if (expected != curTok_)
   {
      addError("expected \"" + expected + "\" found \"" + curTok_ + "\"");
      throw JimJamException();

   }
   getNext();
}

// Require current token to be of a certain type
void JimJamLexer::requireType(int type) //, string& tok)
{
   if (type != curType_)
   {
      stringstream ss;
      ss << "expected type " << type << " found type " << curType_;
      addError(ss.str());
   }
   // tok = curTok_;
   getNext();
}

// Get a token of a require type
void JimJamLexer::getToken(int type, string& tok)
{
   if (curType_ != type)
   {
      //if (required)
         addError("expected \"" + typeToStr(type) +  "\" found \"" + typeToStr(curType_) + "\"");
      //else
      //{
      //   tok = "";
      //   return;
      //}
   }
   tok = curTok_;
   getNext();

}

// Save the current state
void JimJamLexer::saveState()
{
   stateSaved_ = true;
   savedTextPtr_ = textPtr_;
   //savedNextCh_ = nextCh_;
   //savedNextNextCh_ = nextNextCh_;
   savedCurTok_ = curTok_;
   savedWhitespace_ = whitespace_;
   savedComment_ = comment_;
   savedCurType_ = curType_;
   savedCurTokPos_ = curTokPos_;
   savedLineNum_ = lineNum_;
   savedLineStart_ = lineStart_;
   savedErrors_  = errors_;
}

void JimJamLexer::discardState()
{
   stateSaved_ = false;
}

// Restore the current state
void JimJamLexer::restoreState()
{
   stateSaved_ = false;
   textPtr_ = savedTextPtr_;
   //nextCh_ = savedNextCh_;
   //nextNextCh_ = savedNextNextCh_;
   curTok_ = savedCurTok_;
   whitespace_ = savedWhitespace_;
   comment_ = savedComment_;
   curType_ = savedCurType_;
   curTokPos_ = savedCurTokPos_;
   lineNum_ = savedLineNum_;
   lineStart_ = savedLineStart_;
   errors_ = savedErrors_;
}
