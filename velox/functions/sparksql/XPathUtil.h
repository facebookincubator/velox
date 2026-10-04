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

#include <folly/Expected.h>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

#include "velox/common/base/Status.h"

namespace facebook::velox::functions::sparksql {
namespace xpath {

/// Owns the string representation of an XPath result.
class XPathStringValue {
 public:
  /// Owns an already-formatted XPath string.
  explicit XPathStringValue(std::string value);

  /// Releases the owned string.
  ~XPathStringValue();

  /// Moves an owned XPath string.
  XPathStringValue(XPathStringValue&& other) noexcept;

  /// Replaces this value with an owned XPath string.
  XPathStringValue& operator=(XPathStringValue&& other) noexcept;

  XPathStringValue(const XPathStringValue&) = delete;
  XPathStringValue& operator=(const XPathStringValue&) = delete;

  /// Returns the owned string without copying it.
  std::string_view view() const;

 private:
  friend folly::Expected<std::optional<XPathStringValue>, Status> evalString(
      std::string_view xml,
      std::string_view path);

  // Takes ownership of a libxml2-allocated string.
  XPathStringValue(void* value, size_t size);

  // Stores a string formatted by Velox.
  std::string value_;

  // Stores a string allocated by libxml2.
  void* xmlValue_{nullptr};

  // Number of bytes in 'xmlValue_'.
  size_t xmlSize_{0};
};

/// Evaluates a boolean XPath expression on a non-empty UTF-8 XML string.
/// Returns a user error on malformed XML or XPath syntax/evaluation errors.
/// Error text does not include the document or entity system identifiers, and
/// an invalid path is reported as a short UTF-8 prefix with controls and
/// invalid bytes replaced, not cut inside a code point.
/// Returns nullopt when no value is available, including retained entity
/// references and unregistered namespace prefixes.
folly::Expected<std::optional<bool>, Status> evalBoolean(
    std::string_view xml,
    std::string_view path);

/// Evaluates a string XPath expression on a non-empty UTF-8 XML string.
/// Returns a user error on malformed XML or XPath syntax/evaluation errors.
/// Error text does not include the document or entity system identifiers, and
/// an invalid path is reported as a short UTF-8 prefix with controls and
/// invalid bytes replaced, not cut inside a code point.
/// Returns nullopt when no value is available, including retained entity
/// references and unregistered namespace prefixes. A no-match node-set returns
/// an empty string.
folly::Expected<std::optional<XPathStringValue>, Status> evalString(
    std::string_view xml,
    std::string_view path);

} // namespace xpath
} // namespace facebook::velox::functions::sparksql
