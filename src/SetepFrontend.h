#pragma once

#include "SetepSemantic.h"
#include <stdexcept>
#include <string>

namespace setep {
// A parse or semantic error. what() starts with "line N: " when the parser
// located it; lowering errors carry the semantic message only.
struct SourceError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Parse SETEP source text and lower it to the semantic IR. The generated
// parser keeps one global lexer, so calls must not run concurrently.
Module parseSource(const std::string& source);
}
