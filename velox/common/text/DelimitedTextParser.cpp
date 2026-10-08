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

#include "velox/common/text/DelimitedTextParser.h"

#include <algorithm>

namespace facebook::velox::text {
namespace detail {

DelimitedTextScanner::DelimitedTextScanner(
    DelimitedTextOptions options,
    bool decodeEscapedNewlines)
    : options_{options}, decodeEscapedNewlines_{decodeEscapedNewlines} {}

void DelimitedTextScanner::reset() {
  state_ = State::kStart;
  escaping_ = false;
  hasQuotedContent_ = false;
  quoteRunLength_ = 0;
  pendingWhitespace_.clear();
}

DelimitedTextScanResult DelimitedTextScanner::consume(char value) {
  return consume(value, value == options_.delimiter);
}

DelimitedTextScanResult DelimitedTextScanner::consumeUnquoted(
    char value,
    bool isDelimiter) {
  if (escaping_) {
    escaping_ = false;
    if (decodeEscapedNewlines_ && value == 'r') {
      return {DelimitedTextScanAction::kAppend, '\r'};
    }
    if (decodeEscapedNewlines_ && value == 'n') {
      return {DelimitedTextScanAction::kAppend, '\n'};
    }
    return {DelimitedTextScanAction::kAppend, value};
  }

  if (isDelimiter) {
    return {DelimitedTextScanAction::kDelimiter, value};
  }

  if (options_.escape.has_value() && value == *options_.escape) {
    escaping_ = true;
    return {
        DelimitedTextScanAction::kSkip,
        value,
        0,
        true,
    };
  }

  return {DelimitedTextScanAction::kAppend, value};
}

DelimitedTextScanResult DelimitedTextScanner::consumeAfterQuote(
    char value,
    bool isDelimiter,
    size_t numQuotes) {
  if (isDelimiter) {
    pendingWhitespace_.clear();
    return {DelimitedTextScanAction::kDelimiter, value, numQuotes};
  }
  if (static_cast<unsigned char>(value) <= ' ') {
    pendingWhitespace_.push_back(value);
    return {DelimitedTextScanAction::kSkip, value, numQuotes};
  }

  state_ = State::kMalformedQuote;
  return {
      DelimitedTextScanAction::kMalformedQuote,
      value,
      numQuotes,
      true,
  };
}

DelimitedTextScanResult DelimitedTextScanner::consume(
    char value,
    bool isDelimiter) {
  switch (state_) {
    case State::kStart:
      if (options_.quote != '\0' && value == options_.quote) {
        state_ = State::kQuoted;
        return {
            DelimitedTextScanAction::kSkip,
            value,
            0,
            true,
        };
      }
      state_ = State::kUnquoted;
      return consumeUnquoted(value, isDelimiter);

    case State::kUnquoted:
      return consumeUnquoted(value, isDelimiter);

    case State::kQuoted:
      if (escaping_) {
        escaping_ = false;
        hasQuotedContent_ = true;
        if (value == options_.quote ||
            (options_.escape.has_value() && value == *options_.escape)) {
          return {DelimitedTextScanAction::kAppend, value};
        }
        return {
            DelimitedTextScanAction::kAppend,
            value,
            0,
            true,
            true,
        };
      }
      if (options_.escape.has_value() && value == *options_.escape) {
        escaping_ = true;
        hasQuotedContent_ = true;
        return {DelimitedTextScanAction::kSkip, value};
      }
      if (value == options_.quote) {
        state_ = State::kQuoteRun;
        quoteRunLength_ = 1;
        return {DelimitedTextScanAction::kSkip, value};
      }
      hasQuotedContent_ = true;
      return {DelimitedTextScanAction::kAppend, value};

    case State::kQuoteRun:
      if (value == options_.quote) {
        ++quoteRunLength_;
        return {DelimitedTextScanAction::kSkip, value};
      } else {
        const auto numQuotes = quoteRunLength_ - 1;
        quoteRunLength_ = 0;
        state_ = State::kAfterQuote;
        return consumeAfterQuote(value, isDelimiter, numQuotes);
      }

    case State::kAfterQuote:
      return consumeAfterQuote(value, isDelimiter, 0);

    case State::kMalformedQuote:
      if (isDelimiter) {
        return {DelimitedTextScanAction::kDelimiter, value};
      }
      return {DelimitedTextScanAction::kAppend, value};
  }

  return {DelimitedTextScanAction::kSkip, value};
}

DelimitedTextScanResult DelimitedTextScanner::finish() {
  switch (state_) {
    case State::kStart:
    case State::kAfterQuote:
    case State::kMalformedQuote:
      return {DelimitedTextScanAction::kSkip, '\0'};
    case State::kUnquoted:
      if (escaping_) {
        escaping_ = false;
        return {DelimitedTextScanAction::kAppend, *options_.escape};
      }
      return {DelimitedTextScanAction::kSkip, '\0'};
    case State::kQuoted:
      if (escaping_) {
        escaping_ = false;
        return {DelimitedTextScanAction::kAppend, *options_.escape};
      }
      if (!hasQuotedContent_) {
        return {DelimitedTextScanAction::kAppend, options_.quote};
      }
      return {DelimitedTextScanAction::kSkip, '\0'};
    case State::kQuoteRun: {
      const auto numQuotes = quoteRunLength_ - 1;
      quoteRunLength_ = 0;
      state_ = State::kAfterQuote;
      return {DelimitedTextScanAction::kSkip, '\0', numQuotes};
    }
  }

  return {DelimitedTextScanAction::kSkip, '\0'};
}

std::string_view DelimitedTextScanner::pendingWhitespace() const {
  return pendingWhitespace_;
}

} // namespace detail

void DelimitedTextParser::splitLine(
    std::string_view line,
    const DelimitedTextOptions& options,
    size_t maxFields,
    DelimitedTextBuffers& buffers) {
  auto& fields = buffers.fields;
  auto& decodedFields = buffers.decodedFields;
  fields.clear();
  decodedFields.clear();
  if (maxFields > 0) {
    fields.reserve(maxFields);
  }

  size_t position{0};
  bool trailingDelimiter{false};
  while (position < line.size()) {
    if (maxFields > 0 && fields.size() >= maxFields) {
      break;
    }

    trailingDelimiter = false;
    const size_t start = position;
    const size_t decodedStart = decodedFields.size();
    size_t fieldEnd{line.size()};
    bool decoded{false};

    if (options.quote == '\0' || line[start] != options.quote) {
      size_t end = start;
      while (end < line.size() && line[end] != options.delimiter &&
             (!options.escape.has_value() || line[end] != *options.escape)) {
        ++end;
      }
      if (end == line.size() || line[end] == options.delimiter) {
        fields.push_back(line.substr(start, end - start));
        position = end;
        if (position < line.size()) {
          ++position;
          trailingDelimiter = true;
        }
        continue;
      }
    }

    detail::DelimitedTextScanner scanner(
        options, /*decodeEscapedNewlines=*/false);

    const auto ensureDecoded = [&](size_t copyEnd) {
      if (!decoded) {
        if (decodedFields.capacity() < line.size()) {
          decodedFields.reserve(line.size());
        }
        decodedFields.append(line.data() + start, copyEnd - start);
        decoded = true;
      }
    };

    const auto appendResult =
        [&](const detail::DelimitedTextScanResult& result) {
          if (result.requiresDecodedOutput) {
            ensureDecoded(position);
          }
          if (result.numQuotes > 0) {
            ensureDecoded(position);
            decodedFields.append(result.numQuotes, options.quote);
          }
          if (result.appendEscape) {
            ensureDecoded(position);
            decodedFields.push_back(*options.escape);
          }
          if (result.action == detail::DelimitedTextScanAction::kAppend &&
              decoded) {
            decodedFields.push_back(result.value);
          } else if (
              result.action ==
              detail::DelimitedTextScanAction::kMalformedQuote) {
            ensureDecoded(position);
            decodedFields.push_back(options.quote);
            decodedFields.append(scanner.pendingWhitespace());
            decodedFields.push_back(result.value);
            decodedFields.push_back(options.quote);
            std::rotate(
                decodedFields.begin() + decodedStart,
                decodedFields.end() - 1,
                decodedFields.end());
          }
        };

    while (position < line.size()) {
      if (decoded) {
        const size_t runLength =
            scanner.quotedRawRunLength(line.substr(position));
        if (runLength > 0) {
          decodedFields.append(line.data() + position, runLength);
          position += runLength;
          continue;
        }
      }

      const auto result = scanner.consume(line[position]);
      if (result.action == detail::DelimitedTextScanAction::kDelimiter) {
        appendResult(result);
        fieldEnd = position;
        ++position;
        trailingDelimiter = true;
        break;
      }
      appendResult(result);
      ++position;
    }

    if (position == line.size() && !trailingDelimiter) {
      fieldEnd = position;
      appendResult(scanner.finish());
    }

    fields.push_back(
        decoded ? std::string_view(decodedFields).substr(decodedStart)
                : line.substr(start, fieldEnd - start));
  }

  if ((trailingDelimiter || line.empty()) &&
      (maxFields == 0 || fields.size() < maxFields)) {
    fields.push_back(std::string_view{});
  }
}

void DelimitedTextParser::unescapeField(
    std::string& field,
    std::optional<char> escape,
    bool decodeEscapedNewlines) {
  if (!escape.has_value()) {
    return;
  }

  detail::DelimitedTextScanner scanner(
      DelimitedTextOptions{.delimiter = '\0', .escape = escape, .quote = '\0'},
      decodeEscapedNewlines);
  size_t outputPosition{0};
  for (const auto value : field) {
    const auto result = scanner.consume(value, /*isDelimiter=*/false);
    if (result.action == detail::DelimitedTextScanAction::kAppend) {
      field[outputPosition++] = result.value;
    }
  }
  const auto finalResult = scanner.finish();
  if (finalResult.action == detail::DelimitedTextScanAction::kAppend) {
    field[outputPosition++] = finalResult.value;
  }
  field.resize(outputPosition);
}

bool DelimitedTextParser::isNullField(
    std::string_view field,
    std::string_view nullMarker) {
  return field == nullMarker;
}

} // namespace facebook::velox::text
