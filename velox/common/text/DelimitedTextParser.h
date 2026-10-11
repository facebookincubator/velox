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

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace facebook::velox::text {

/// Configures the delimiter, escape, and quote bytes used to scan text fields.
struct DelimitedTextOptions {
  /// Ends an unquoted field. Delimiter classification takes precedence when
  /// the byte is also the configured escape.
  char delimiter;

  /// Escapes the following byte. std::nullopt disables escaping.
  std::optional<char> escape;

  /// Opens a quoted field when it is the field's first byte. A null byte
  /// disables quoting.
  char quote{'\0'};
};

/// Owns reusable output storage populated by DelimitedTextParser::splitLine.
struct DelimitedTextBuffers {
  /// Holds field views into either the input record or decodedFields.
  std::vector<std::string_view> fields;

  /// Owns fields that require quote or escape decoding. Modifying this string
  /// invalidates any field views that point into it.
  std::string decodedFields;
};

namespace detail {

/// Describes how a scanned byte affects the current field.
enum class DelimitedTextScanAction {
  kAppend,
  kSkip,
  kDelimiter,
  kMalformedQuote,
};

/// Contains the action and decoded value produced for a scanned byte.
struct DelimitedTextScanResult {
  /// Controls whether the caller appends, skips, or ends the field.
  DelimitedTextScanAction action;

  /// Holds the decoded byte when `action` is `kAppend`.
  char value;

  /// Number of literal quote bytes to append before applying `action`.
  size_t numQuotes{0};

  /// Indicates that the caller must switch from a zero-copy view to decoded
  /// storage before applying this result.
  bool requiresDecodedOutput{false};

  /// Indicates that the configured escape byte must be appended before value.
  bool appendEscape{false};
};

/// Applies delimiter, escape, and quote transitions while scanning one field.
class DelimitedTextScanner {
 public:
  /// Creates a scanner using `options`. When an unquoted byte matches both the
  /// delimiter and escape, the delimiter takes precedence for TextReader
  /// compatibility.
  DelimitedTextScanner(
      DelimitedTextOptions options,
      bool decodeEscapedNewlines);

  /// Resets the scanner before reading another field.
  void reset();

  /// Returns true when the next byte must be consumed as escaped field data
  /// without delimiter classification.
  bool isEscaping() const {
    return escaping_;
  }

  /// Consumes raw unquoted input without decoding escaped bytes. This is the
  /// streaming fast path for callers that preserve the raw field and decode it
  /// after delimiter detection. Quoting must be disabled.
  DelimitedTextScanAction consumeUnquotedRaw(char value, bool isDelimiter);

  /// Returns the number of ordinary quoted bytes at the start of `input`.
  /// Advances scanner state only by recording that quoted content was seen.
  size_t quotedRawRunLength(std::string_view input);

  /// Consumes one byte using the configured delimiter.
  DelimitedTextScanResult consume(char value);

  /// Consumes one byte and reports whether to append, skip, or end the field.
  /// `isDelimiter` allows streaming callers to apply nested delimiter rules.
  DelimitedTextScanResult consume(char value, bool isDelimiter);

  /// Finishes the current field and returns any bytes delayed by quote or
  /// escape lookahead.
  DelimitedTextScanResult finish();

  /// Returns whitespace tentatively skipped after a possible closing quote.
  std::string_view pendingWhitespace() const;

 private:
  enum class State {
    kStart,
    kUnquoted,
    kQuoted,
    kQuoteRun,
    kAfterQuote,
    kMalformedQuote,
  };

  DelimitedTextScanResult consumeUnquoted(char value, bool isDelimiter);

  DelimitedTextScanResult
  consumeAfterQuote(char value, bool isDelimiter, size_t numQuotes);

  // Configures delimiter, escape, and quote bytes.
  const DelimitedTextOptions options_;

  // Controls conversion of textual `\r` and `\n` escape sequences.
  const bool decodeEscapedNewlines_;

  // Tracks the current tokenizer state.
  State state_{State::kStart};

  // Tracks whether the preceding byte was the escape character.
  bool escaping_{false};

  // Tracks whether an unclosed quoted field contains bytes after its opener.
  bool hasQuotedContent_{false};

  // Counts a run of quotes until the following byte identifies the last quote
  // as a closing quote.
  size_t quoteRunLength_{0};

  // Holds whitespace after a possible closing quote until the field is known
  // to be valid or malformed.
  std::string pendingWhitespace_;
};

inline DelimitedTextScanAction DelimitedTextScanner::consumeUnquotedRaw(
    char value,
    bool isDelimiter) {
  if (escaping_) {
    escaping_ = false;
    return DelimitedTextScanAction::kAppend;
  }

  state_ = State::kUnquoted;
  if (isDelimiter) {
    return DelimitedTextScanAction::kDelimiter;
  }
  if (options_.escape.has_value() && value == *options_.escape) {
    escaping_ = true;
    return DelimitedTextScanAction::kSkip;
  }
  return DelimitedTextScanAction::kAppend;
}

inline size_t DelimitedTextScanner::quotedRawRunLength(std::string_view input) {
  if (state_ != State::kQuoted || escaping_) {
    return 0;
  }

  size_t runLength{0};
  while (
      runLength < input.size() && input[runLength] != options_.quote &&
      (!options_.escape.has_value() || input[runLength] != *options_.escape)) {
    ++runLength;
  }
  hasQuotedContent_ = hasQuotedContent_ || runLength > 0;
  return runLength;
}

} // namespace detail

/// Splits delimited records and provides shared escape and null handling.
class DelimitedTextParser {
 public:
  /// Splits one record into fields. Quoted fields are recognized only when the
  /// first byte is the configured quote character. The returned views remain
  /// valid until `line` or `buffers` is modified.
  static void splitLine(
      std::string_view line,
      const DelimitedTextOptions& options,
      size_t maxFields,
      DelimitedTextBuffers& buffers);

  /// Decodes escape sequences in place. Escaped 'r' and 'n' can optionally be
  /// converted to carriage return and newline for Hive text compatibility.
  static void unescapeField(
      std::string& field,
      std::optional<char> escape,
      bool decodeEscapedNewlines);

  /// Returns true when `field` matches the configured null marker.
  static bool isNullField(std::string_view field, std::string_view nullMarker);
};

} // namespace facebook::velox::text
