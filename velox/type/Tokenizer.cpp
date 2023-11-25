/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "velox/type/Tokenizer.h"

namespace facebook::velox::common {

DefaultTokenizer::DefaultTokenizer(const std::string& path)
    : path_(path), separators_(Separators::get()) {
  state = State::kNotReady;
  index_ = 0;
}

bool DefaultTokenizer::hasNext() {
  switch (state) {
    case State::kDone:
      return false;
    case State::kReady:
      return true;
    case State::kNotReady:
      break;
    case State::kFailed:
      VELOX_FAIL("Illegal state");
  }
  return tryToComputeNext();
}

std::unique_ptr<Subfield::PathElement> DefaultTokenizer::next() {
  if (!hasNext()) {
    VELOX_FAIL("No more tokens");
  }
  state = State::kNotReady;
  return std::move(next_);
}

bool DefaultTokenizer::hasNextCharacter() {
  return index_ < path_.length();
}

std::unique_ptr<Subfield::PathElement> DefaultTokenizer::computeNext() {
  if (!hasNextCharacter()) {
    state = State::kDone;
    return nullptr;
  }

  if (tryMatchSeparator(separators_->dot)) {
    std::unique_ptr<Subfield::PathElement> token = matchPathSegment();
    firstSegment = false;
    return token;
  }

  if (tryMatchSeparator(separators_->openBracket)) {
    std::unique_ptr<Subfield::PathElement> token =
        tryMatchSeparator(separators_->quote)      ? matchQuotedSubscript()
        : tryMatchSeparator(separators_->wildCard) ? matchWildcardSubscript()
                                                   : matchUnquotedSubscript();

    match(separators_->closeBracket);
    firstSegment = false;
    return token;
  }

  if (firstSegment) {
    std::unique_ptr<Subfield::PathElement> token = matchPathSegment();
    firstSegment = false;
    return token;
  }

  VELOX_UNREACHABLE();
}

bool DefaultTokenizer::tryMatchSeparator(char expected) {
  return separators_->isSeparator(expected) && tryMatch(expected);
}

void DefaultTokenizer::match(char expected) {
  if (!tryMatch(expected)) {
    invalidSubfieldPath();
  }
}

bool DefaultTokenizer::tryMatch(char expected) {
  if (!hasNextCharacter() || peekCharacter() != expected) {
    return false;
  }
  index_++;
  return true;
}

void DefaultTokenizer::nextCharacter() {
  index_++;
}

char DefaultTokenizer::peekCharacter() {
  return path_[index_];
}

std::unique_ptr<Subfield::PathElement> DefaultTokenizer::matchPathSegment() {
  // seek until we see a special character or whitespace
  int start = index_;
  while (hasNextCharacter() && !separators_->isSeparator(peekCharacter()) &&
         isUnquotedPathCharacter(peekCharacter())) {
    nextCharacter();
  }
  int end = index_;

  std::string token = path_.substr(start, end - start);

  // an empty unquoted token is not allowed
  if (token.empty()) {
    invalidSubfieldPath();
  }

  return std::make_unique<Subfield::NestedField>(token);
}

std::unique_ptr<Subfield::PathElement>
DefaultTokenizer::matchUnquotedSubscript() {
  // seek until we see a special character or whitespace
  int start = index_;
  while (hasNextCharacter() && isUnquotedSubscriptCharacter(peekCharacter())) {
    nextCharacter();
  }
  int end = index_;

  std::string token = path_.substr(start, end);

  // an empty unquoted token is not allowed
  if (token.empty()) {
    invalidSubfieldPath();
  }
  long index = 0;
  try {
    index = std::stol(token);
  } catch (...) {
    VELOX_FAIL("Invalid index {}", token);
  }
  return std::make_unique<Subfield::LongSubscript>(index);
}

bool DefaultTokenizer::isUnquotedPathCharacter(char c) {
  return c == ':' || c == '$' || c == '-' || c == '/' || c == '@' || c == '|' ||
      c == '#' || c == '.' || isUnquotedSubscriptCharacter(c);
}

bool DefaultTokenizer::isUnquotedSubscriptCharacter(char c) {
  return c == '-' || c == '_' || isalnum(c);
}

std::unique_ptr<Subfield::PathElement>
DefaultTokenizer::matchQuotedSubscript() {
  // quote has already been matched

  // seek until we see the close quote
  std::string token;
  bool escaped = false;

  while (hasNextCharacter() &&
         (escaped || peekCharacter() != separators_->quote)) {
    if (escaped) {
      switch (peekCharacter()) {
        case '\"':
        case '\\':
          token += peekCharacter();
          break;
        default:
          invalidSubfieldPath();
      }
      escaped = false;
    } else {
      if (peekCharacter() == separators_->backSlash) {
        escaped = true;
      } else {
        token += peekCharacter();
      }
    }
    nextCharacter();
  }
  if (escaped) {
    invalidSubfieldPath();
  }

  match(separators_->quote);

  if (token == "*") {
    return std::make_unique<Subfield::AllSubscripts>();
  }
  return std::make_unique<Subfield::StringSubscript>(token);
}

std::unique_ptr<Subfield::PathElement>
DefaultTokenizer::matchWildcardSubscript() {
  return std::make_unique<Subfield::AllSubscripts>();
}

void DefaultTokenizer::invalidSubfieldPath() {
  VELOX_FAIL("Invalid subfield path: {}", this->toString());
}

std::string DefaultTokenizer::toString() {
  return path_.substr(0, index_) + separators_->unicodeCaret +
      path_.substr(index_);
}

bool DefaultTokenizer::tryToComputeNext() {
  state = State::kFailed; // temporary pessimism
  next_ = computeNext();
  if (state != State::kDone) {
    state = State::kReady;
    return true;
  }
  return false;
}

std::function<std::unique_ptr<Tokenizer>(const std::string&)>
    Tokenizer::tokenizerFactory_ = nullptr;

// static
std::unique_ptr<Tokenizer> Tokenizer::getInstance(const std::string& path) {
  if (!tokenizerFactory_) {
    tokenizerFactory_ = [](const std::string& p) {
      return std::make_unique<DefaultTokenizer>(p);
    };
  }
  return tokenizerFactory_(path);
}

// static
void Tokenizer::registerInstanceFactory(
    std::function<std::unique_ptr<Tokenizer>(const std::string&)>
        tokenizerFactory) {
  tokenizerFactory_ = tokenizerFactory;
}
} // namespace facebook::velox::common
