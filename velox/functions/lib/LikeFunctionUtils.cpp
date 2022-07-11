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
#include "velox/functions/lib/LikeFunctionUtils.h"

#include <optional>

#include "velox/expression/EvalCtx.h"
#include "velox/expression/Expr.h"
#include "velox/functions/lib/ArrayBuilder.h"
#include "velox/type/StringView.h"

namespace facebook::velox::functions {
namespace likeutils {

using ::facebook::velox::exec::EvalCtx;
using ::facebook::velox::exec::Expr;
using ::facebook::velox::exec::VectorFunction;
using ::facebook::velox::exec::VectorFunctionArg;

// This function checks if the string 'pattern' has wildcard characters; a
// wildcard character could either be '_' or '%'. If wildcard characters are
// detected in the string 'pattern', the function returns true; else it returns
// false.
bool checkWildcardInPattern(const std::string& pattern) {
  size_t patternLength = pattern.size();
  for (size_t patternIndex = 0; patternIndex < patternLength; patternIndex++) {
    if ((pattern[patternIndex] == '%') || (pattern[patternIndex] == '_')) {
      return true;
    }
  }

  return false;
}

// This function is called if the string 'pattern' has no wildcard characters; a
// wildcard character could either be '_' or '%'. When there are no wildcard
// characters in the pattern, the 'inputString' can be matched with the string
// 'pattern' by comparing the characters at the same indices in 'inputString'
// and 'pattern'. This character by character comparison is performed in this
// function and if there is no mismatch till the end of both 'inputString' and
// 'pattern' are reached, we return true.
bool matchExactPattern(
    const std::string& inputString,
    const std::string& pattern) {
  size_t inputStringIndex = 0;
  size_t patternIndex = 0;
  size_t inputStringLength = inputString.size();
  size_t patternLength = pattern.size();

  if (inputStringLength != patternLength) {
    return false;
  }

  for (;
       (inputStringIndex < inputStringLength) && (patternIndex < patternLength);
       inputStringIndex++, patternIndex++) {
    if (inputString[inputStringIndex] != pattern[patternIndex]) {
      return false;
    }
  }
  return (
      (inputStringIndex == inputStringLength) &&
      (patternIndex == patternLength));
}

// We use the Knuth-Morris-Pratt (KMP) string search algorithm to search for the
// string 'pattern' in a given 'inputString'. More details about the KMP
// algorithm can be found here:
// https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/
// https://cmps-people.ok.ubc.ca/ylucet/DS/KnuthMorrisPratt.html
// This function does the preprocessing of the string 'pattern' to return the
// array lps, which is used by the KMP algorithm. lps[i] stores the length of
// the longest proper prefix that is also a suffix of the pat[0-i].
void kmpSearchPreprocessPattern(
    const std::string& pattern,
    std::vector<int>& lps) {
  size_t patternIndex = 1;
  size_t lpsLength = 0;
  size_t patternLength = pattern.size();

  while (patternIndex < patternLength) {
    if (pattern[patternIndex] == pattern[lpsLength]) {
      lpsLength++;
      lps[patternIndex] = lpsLength;
      patternIndex++;
    } else {
      if (lpsLength != 0) {
        lpsLength = lps[lpsLength - 1];
      } else {
        lps[patternIndex] = 0;
        patternIndex++;
      }
    }
  }
}

// This function checks if the string 'pattern' is an 'exact' pattern followed
// by one or more occurences of the wildcard character, '%'; let us refer to
// such patterns as prefix patterns. We define an 'exact' pattern as a string
// that doesn't contain any wildcard characters; a wildcard character could
// either be '_' or '%'. If 'pattern' is a prefix pattern, we return true.
bool checkPrefixPattern(const std::string& pattern) {
  size_t patternLength = pattern.size();
  size_t patternIndex = patternLength - 1;

  for (; patternIndex >= 0; patternIndex--) {
    if (pattern[patternIndex] != '%') {
      break;
    }
  }
  if (patternIndex == patternLength - 1) {
    return false;
  }

  for (; patternIndex >= 0; patternIndex--) {
    if ((pattern[patternIndex] == '%') || (pattern[patternIndex] == '_')) {
      return false;
    }
  }

  return true;
}

// This function is called when the string 'pattern' is a prefix pattern, as
// defined in and as determined by the return value of the function bool
// checkPrefixPattern(const std::string& pattern); When 'pattern' is a prefix
// pattern, we match the exact pattern in the prefix pattern with the first 'm'
// characters of the 'inputString', where m = length(pattern). If the first m
// characters match in the two strings, we return true.
bool matchPrefixPattern(
    const std::string& inputString,
    const std::string& pattern) {
  size_t inputStringIndex = 0;
  size_t patternIndex = 0;
  size_t inputStringLength = inputString.size();
  size_t patternLength = pattern.size();

  for (; (patternIndex < patternLength) &&
       (inputStringIndex < inputStringLength) && (pattern[patternIndex] != '%');
       inputStringIndex++, patternIndex++) {
    if (pattern[patternIndex] != inputString[inputStringIndex]) {
      return false;
    }
  }

  return (pattern[patternIndex] == '%');
}

// This function checks if the string 'pattern' is an 'exact' pattern preceded
// by one or more occurences of the wildcard character, '%'; let us refer to
// such patterns as suffix patterns. We define an 'exact' pattern as a string
// that doesn't contain any wildcard characters; a wildcard character could
// either be '_' or '%'. If 'pattern' is a suffix pattern, we return true.
bool checkSuffixPattern(const std::string& pattern) {
  size_t patternIndex = 0;
  size_t patternLength = pattern.size();

  for (; patternIndex < patternLength; patternIndex++) {
    if (pattern[patternIndex] != '%') {
      break;
    }
  }
  if (patternIndex == 0) {
    return false;
  }

  for (; patternIndex < patternLength; patternIndex++) {
    if ((pattern[patternIndex] == '%') || (pattern[patternIndex] == '_')) {
      return false;
    }
  }

  return true;
}

// This function is called when the string 'pattern' is a suffix pattern, as
// defined in and as determined by the return value of the function bool
// checkSuffixPattern(const std::string& pattern); When 'pattern' is a suffix
// pattern, we match the exact pattern in the suffix pattern with the last 'm'
// characters of the 'inputString', where m = length(pattern). If the last m
// characters match in the two strings, we return true.
bool matchSuffixPattern(
    const std::string& inputString,
    const std::string& pattern) {
  int inputStringLength = inputString.size();
  int patternLength = pattern.size();
  int inputStringIndex = inputStringLength - 1;
  int patternIndex = patternLength - 1;

  for (; (patternIndex >= 0) && (inputStringIndex >= 0) &&
       (pattern[patternIndex] != '%');
       patternIndex--, inputStringIndex--) {
    if (pattern[patternIndex] != inputString[inputStringIndex]) {
      return false;
    }
  }

  return (pattern[patternIndex] == '%');
}

} // namespace likeutils
} // namespace facebook::velox::functions