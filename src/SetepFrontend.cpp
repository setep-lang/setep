#include "SetepFrontend.h"
#include "SetepSyntax.h"

namespace setep {
namespace {
std::string parserErrors() {
  std::string text;
  for (const JimJamError& error : SetepParser::setepLexer.errors_) {
    if (!text.empty()) text += "\n";
    text += "line " + std::to_string(error.lineNum_ + 1) + ": " + error.errMsg_;
  }
  return text;
}
}

Module parseSource(const std::string& source) {
  SetepParser::setepLexer.errors_.clear();
  try {
    SetepParser::setepLexer.init(source.data(), source.data() + source.size());
    SetepParser::SetepModel parsed;
    if (!SetepParser::setepLexer.errors_.empty()) throw SourceError(parserErrors());
    setepSyntax::SetepFile file(parsed);
    if (!SetepParser::setepLexer.checkType(SetepParser::SetepLexer::endTok))
      throw SourceError("unexpected trailing SETEP input");
    return lower(file);
  } catch (const SourceError&) {
    throw;
  } catch (const std::exception& e) {
    if (!SetepParser::setepLexer.errors_.empty()) throw SourceError(parserErrors());
    throw SourceError(e.what());
  } catch (...) {
    if (!SetepParser::setepLexer.errors_.empty()) throw SourceError(parserErrors());
    throw SourceError("SETEP parsing failed");
  }
}
}
