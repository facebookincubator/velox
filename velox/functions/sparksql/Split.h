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

#include <cctype>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

#include "velox/functions/Udf.h"
#include "velox/functions/lib/Re2Functions.h"
#include "velox/functions/lib/Utf8Utils.h"

namespace facebook::velox::functions::sparksql {
namespace detail {

/// Splits input strings around matches of a regex delimiter, with fast paths
/// for empty, single-character, escaped-character, octal, and literal-string
/// delimiters. Shared by Spark's split and str_to_map functions.
class StringSplitter {
 public:
  StringSplitter() : cache_(0) {}

  void initialize(
      uint64_t maxCompiledRegexes,
      const StringView* constantDelimiter) {
    cache_.setMaxCompiledRegexes(maxCompiledRegexes);
    isConstantDelimiter_ = constantDelimiter != nullptr;
    if (isConstantDelimiter_) {
      initializeFastPath(constantDelimiter->data(), constantDelimiter->size());
    } else {
      fastPathKind_ = FastPathKind::kNone;
    }
  }

  void initializeConstant(
      uint64_t maxCompiledRegexes,
      std::string_view constantDelimiter) {
    cache_.setMaxCompiledRegexes(maxCompiledRegexes);
    isConstantDelimiter_ = true;
    initializeFastPath(constantDelimiter.data(), constantDelimiter.size());
  }

  template <typename OnPiece>
  void split(
      std::string_view input,
      std::string_view delimiter,
      int32_t limit,
      OnPiece&& onPiece) {
    if (delimiter.empty()) {
      splitEmptyDelimiter(input, limit, onPiece);
      return;
    }

    // Trivial case of converting string to 1 piece.
    if (limit == 1) {
      onPiece(input);
      return;
    }

    if (!isConstantDelimiter_) {
      initializeFastPath(delimiter.data(), delimiter.size());
    }

    int32_t addedElements{0};
    const size_t end = input.size();
    const char* start = input.data();
    size_t pos = 0;

    if (fastPathKind_ == FastPathKind::kSingleChar) {
      for (size_t i = 0; i < end; ++i) {
        if (start[i] == singleCharDelimiter_) {
          onPiece(std::string_view(start + pos, i - pos));
          pos = i + 1;
          ++addedElements;
          if (addedElements + 1 == limit) {
            break;
          }
        }
      }
    } else if (fastPathKind_ == FastPathKind::kLiteralString) {
      while (true) {
        const size_t found = input.find(literalDelimiter_, pos);
        if (found == std::string_view::npos) {
          break;
        }

        onPiece(std::string_view(start + pos, found - pos));
        pos = found + literalDelimiter_.size();

        ++addedElements;
        // If the next element should be the last, leave the loop.
        if (addedElements + 1 == limit) {
          break;
        }
      }
    } else {
      auto* re = cache_.findOrCompile(StringView(delimiter));
      const auto re2String = re2::StringPiece(start, end);
      re2::StringPiece subMatches[1];
      // Matches a regular expression against a portion of the input string,
      // starting from 'pos' to the end of the input string. The match is not
      // anchored, which means it can start at any position in the string. If a
      // match is found, the matched portion of the string is stored in
      // 'subMatches'. The '1' indicates that we are only interested in the
      // first match found from the current position 'pos' in each iteration of
      // the loop.
      while (re->Match(
          re2String, pos, end, RE2::Anchor::UNANCHORED, subMatches, 1)) {
        const auto fullMatch = subMatches[0];
        auto offset = static_cast<size_t>(fullMatch.data() - start);
        const auto size = fullMatch.size();
        if (offset >= end) {
          break;
        }

        // When hitting an empty match, split the character at the current 'pos'
        // of the input string and put it into the result, followed by an
        // empty tail string at last, e.g., the result for
        // split('abc','d|') is ["a","b","c",""].
        if (size == 0) {
          int32_t codePoint;
          auto charLength =
              tryGetUtf8CharLength(start + pos, end - pos, codePoint);
          if (charLength <= 0) {
            // Invalid UTF-8 character, the length of the invalid
            // character is the absolute value of result of
            // `tryGetUtf8CharLength`.
            charLength = -charLength;
          }
          offset += charLength;
        }
        onPiece(std::string_view(start + pos, offset - pos));
        pos = offset + size;

        ++addedElements;
        // If the next element should be the last, leave the loop.
        if (addedElements + 1 == limit) {
          break;
        }
      }
    }

    // Add the rest of the string and we are done.
    // Note that the rest of the string can be empty - we still add it.
    onPiece(std::string_view(start + pos, end - pos));
  }

 private:
  enum class FastPathKind {
    kNone,
    kSingleChar,
    kLiteralString,
  };

  // Checks if a string represents an octal-escaped ascii character (e.g.,
  // \123). If valid, converts it to its decimal value.
  static bool isOctalString(const char* str, char& decimal, size_t size) {
    if (size < 2 || size > 4 || str[0] != '\\') {
      return false;
    }
    int decimalValue = 0;
    int octalValue = 0;
    for (size_t i = 1; i < size; ++i) {
      const char ch = str[i];
      if (ch < '0' || ch > '7') {
        return false;
      }
      decimalValue = decimalValue * 10 + (ch - '0');
      octalValue = octalValue * 8 + (ch - '0');
    }
    // The ascii octal number should not exceed 177.
    if (decimalValue > 177) {
      return false;
    }
    decimal = static_cast<char>(octalValue);
    return true;
  }

  // When pattern is empty, split each character out. Since Spark 3.4, when
  // delimiter is empty, the result does not include an empty tail string, e.g.
  // split('abc', '') outputs ["a", "b", "c"] instead of ["a", "b", "c", ""].
  // The result does not include remaining string when limit is smaller than the
  // string size, e.g. split('abc', '', 2) outputs ["a", "b"] instead of ["a",
  // "bc"].
  template <typename OnPiece>
  static void splitEmptyDelimiter(
      std::string_view input,
      int32_t limit,
      OnPiece&& onPiece) {
    if (input.empty()) {
      onPiece(std::string_view{});
      return;
    }

    const size_t end = input.size();
    const char* start = input.data();
    size_t pos = 0;
    int32_t count = 0;
    while (pos < end && count < limit) {
      int32_t codePoint;
      auto charLength = tryGetUtf8CharLength(start + pos, end - pos, codePoint);
      if (charLength <= 0) {
        // Invalid UTF-8 character, the length of the invalid
        // character is the absolute value of result of `tryGetUtf8CharLength`.
        charLength = -charLength;
      }
      onPiece(std::string_view(start + pos, charLength));
      pos += charLength;
      count += 1;
    }
  }

  // Fast path if the delimiter is a
  // (1)one-char string and this character is not one of the RegEx's meta
  // characters ".$|()[{^?*+\\", or
  // (2)two-char string and the first char is the backslash and the second is
  // not the ascii digit or ascii letter, or
  // (3)octal string that falls within the range of ASCII characters, or
  // (4)non-empty plain string without any RegEx meta character.
  void initializeFastPath(const char* delimiterStr, size_t delimiterSize) {
    // Referencing from
    // https://github.com/openjdk/jdk8/blob/master/jdk/src/share/classes/java/lang/String.java#L2325
    // to align with Spark.
    static constexpr std::string_view kReservedChars = ".$|()[{^?*+\\";

    if (delimiterSize == 1) {
      singleCharDelimiter_ = delimiterStr[0];
      if (kReservedChars.find(singleCharDelimiter_) == std::string_view::npos) {
        fastPathKind_ = FastPathKind::kSingleChar;
        return;
      }
    } else if (delimiterSize == 2 && delimiterStr[0] == '\\') {
      singleCharDelimiter_ = delimiterStr[1];
      const auto delimiterChar =
          static_cast<unsigned char>(singleCharDelimiter_);
      if (!std::isdigit(delimiterChar) && !std::isalpha(delimiterChar)) {
        fastPathKind_ = FastPathKind::kSingleChar;
        return;
      }
    } else if (isOctalString(
                   delimiterStr, singleCharDelimiter_, delimiterSize)) {
      fastPathKind_ = FastPathKind::kSingleChar;
      return;
    }

    if (delimiterSize > 1 &&
        std::string_view{delimiterStr, delimiterSize}.find_first_of(
            kReservedChars) == std::string_view::npos) {
      fastPathKind_ = FastPathKind::kLiteralString;
      literalDelimiter_.assign(delimiterStr, delimiterSize);
      return;
    }

    fastPathKind_ = FastPathKind::kNone;
  }

  mutable facebook::velox::functions::detail::ReCache cache_;
  FastPathKind fastPathKind_{FastPathKind::kNone};
  // Single character delimiter for fast path.
  char singleCharDelimiter_{};
  // Non-empty delimiter for the literal-string fast path.
  std::string literalDelimiter_;
  bool isConstantDelimiter_{false};
};

} // namespace detail

/// split(string, delimiter[, limit]) -> array(varchar)
///
/// Splits string on delimiter and returns an array of size at most limit.
/// delimiter is a string representing regular expression.
/// limit is an integer which controls the number of times the regex is applied.
/// By default, limit is -1, which means 'no limit', the delimiter will be
/// applied as many times as possible.
template <typename T>
struct Split {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Results refer to strings in the first argument.
  static constexpr int32_t reuse_strings_from_arg = 0;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& config,
      const arg_type<Varchar>* input,
      const arg_type<Varchar>* delimiter) {
    initialize(inputTypes, config, input, delimiter, nullptr);
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* /*input*/,
      const arg_type<Varchar>* delimiter,
      const arg_type<int32_t>* /*limit*/) {
    splitter_.initialize(config.exprMaxCompiledRegexes(), delimiter);
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Array<Varchar>>& result,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& delimiter) {
    doCall(result, input, delimiter, std::numeric_limits<int32_t>::max());
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Array<Varchar>>& result,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& delimiter,
      const arg_type<int32_t>& limit) {
    doCall(
        result,
        input,
        delimiter,
        limit > 0 ? limit : std::numeric_limits<int32_t>::max());
  }

 private:
  void doCall(
      out_type<Array<Varchar>>& result,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& delimiter,
      int32_t limit) {
    splitter_.split(
        std::string_view(input.data(), input.size()),
        std::string_view(delimiter.data(), delimiter.size()),
        limit,
        [&](std::string_view piece) {
          result.add_item().setNoCopy(StringView(piece));
        });
  }

  detail::StringSplitter splitter_;
};
} // namespace facebook::velox::functions::sparksql
