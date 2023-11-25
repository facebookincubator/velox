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
#pragma once

#include "velox/type/Subfield.h"

namespace facebook::velox::common {

class Tokenizer {
 public:
  enum class State {
    // We have computed the next element and haven't returned it yet.
    kReady,

    // We haven't yet computed or have already returned the element.
    kNotReady,

    // We have reached the end of the data and are finished.
    kDone,

    // We've suffered an exception and are kaput.
    kFailed,
  };

  virtual ~Tokenizer() = default;

  virtual bool hasNext() = 0;

  virtual std::unique_ptr<Subfield::PathElement> next() = 0;

  static std::unique_ptr<Tokenizer> getInstance(const std::string& path);

  static void registerInstanceFactory(
      std::function<std::unique_ptr<Tokenizer>(const std::string&)>
          tokenizerFactory);

 private:
  static std::function<std::unique_ptr<Tokenizer>(const std::string&)>
      tokenizerFactory_;
};

class DefaultTokenizer : public Tokenizer {
 public:
  explicit DefaultTokenizer(const std::string& path);

  bool hasNext() override;

  std::unique_ptr<Subfield::PathElement> next() override;

 private:
  const std::string path_;
  // Customized separators to tokenize field name.
  std::shared_ptr<Separators> separators_;

  int index_;
  State state;
  bool firstSegment = true;
  std::unique_ptr<Subfield::PathElement> next_;

  bool hasNextCharacter();

  std::unique_ptr<Subfield::PathElement> computeNext();

  // Returns whether the expected char is a separator and
  // can be found.
  bool tryMatchSeparator(char expected);

  void match(char expected);

  bool tryMatch(char expected);

  std::unique_ptr<Subfield::PathElement> matchPathSegment();

  std::unique_ptr<Subfield::PathElement> matchUnquotedSubscript();

  std::unique_ptr<Subfield::PathElement> matchQuotedSubscript();

  std::string toString();

  bool tryToComputeNext();

  void invalidSubfieldPath();

  bool isUnquotedPathCharacter(char c);

  bool isUnquotedSubscriptCharacter(char c);

  void nextCharacter();

  char peekCharacter();

  std::unique_ptr<Subfield::PathElement> matchWildcardSubscript();
};
} // namespace facebook::velox::common
