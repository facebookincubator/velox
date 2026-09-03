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

#include "velox/exec/rpc/BackendErrorSummary.h"

#include <algorithm>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec::rpc {

namespace {

// Appended whenever any of the backend text was dropped, so a reader knows the
// row is showing a summary and the full text is in the logs.
constexpr std::string_view kElisionMarker = "... (truncated)";

constexpr std::string_view kWhitespace = " \t\n\r\f\v";

// Drops the whitespace around a backend message. Leading whitespace is what
// makes the first line empty, and a lone trailing newline is not text worth
// telling the reader was dropped.
std::string_view trimWhitespace(std::string_view text) {
  const auto begin = text.find_first_not_of(kWhitespace);
  if (begin == std::string_view::npos) {
    return {};
  }
  return text.substr(begin, text.find_last_not_of(kWhitespace) + 1 - begin);
}

// Backs 'length' off to a UTF-8 sequence boundary. Presto reads this text as a
// UTF-8 VARCHAR, and a backend message echoing prompt text is not necessarily
// ASCII, so cutting on a raw byte index can leave half a character in the
// column.
size_t utf8SafeLength(std::string_view text, size_t length) {
  if (length >= text.size()) {
    return text.size();
  }
  while (length > 0 &&
         (static_cast<unsigned char>(text[length]) & 0xC0) == 0x80) {
    --length;
  }
  return length;
}

} // namespace

std::string summarizeBackendError(std::string_view rawError, size_t maxLength) {
  VELOX_CHECK_GT(
      maxLength,
      kElisionMarker.size(),
      "Backend error cap must leave room for the elision marker");

  // Trim first: a message that leads with a newline has an empty first line,
  // which would leave the row carrying the marker and nothing else, and a
  // message that ends with one has nothing dropped worth marking.
  const std::string_view trimmed = trimWhitespace(rawError);

  // Servers put their stack frames on their own lines, so the first newline is
  // the boundary between the sentence worth keeping and the trace. A backend
  // that appends frames to the message instead leaves no boundary; the cap
  // below is what bounds the row in that case.
  const std::string_view firstLine = trimmed.substr(0, trimmed.find('\n'));

  if (firstLine.size() == trimmed.size() && firstLine.size() <= maxLength) {
    return std::string(firstLine);
  }
  // The marker is appended below, so the kept text has to leave room for it or
  // the result overruns 'maxLength'. The VELOX_CHECK_GT above makes the
  // subtraction safe.
  const auto kept = utf8SafeLength(
      firstLine, std::min(firstLine.size(), maxLength - kElisionMarker.size()));
  return std::string(firstLine.substr(0, kept)) + std::string(kElisionMarker);
}

std::vector<facebook::velox::rpc::RPCResponse> makeBatchErrorResponses(
    size_t numRows,
    std::string_view errorPrefix,
    std::string_view rawError) {
  std::vector<facebook::velox::rpc::RPCResponse> responses(numRows);
  const std::string rowError = std::string(errorPrefix) +
      summarizeBackendError(rawError, kMaxBackendErrorBytes);
  for (auto& response : responses) {
    response.error = rowError;
    response.errorKind = facebook::velox::rpc::RPCErrorKind::kBackendError;
  }
  return responses;
}

} // namespace facebook::velox::exec::rpc
