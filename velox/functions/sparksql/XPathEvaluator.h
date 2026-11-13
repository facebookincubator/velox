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

#include <optional>
#include <string>
#include <string_view>

#include "velox/common/base/Status.h"

namespace facebook::velox::functions::sparksql {
namespace xpath {

/// Evaluates a boolean XPath expression on a non-empty UTF-8 XML string.
/// Returns a user error on malformed XML or XPath syntax/evaluation errors.
/// Error text does not include the document or entity system identifiers, and
/// an invalid path is reported as a short UTF-8 prefix with controls and
/// invalid bytes replaced, not cut inside a code point.
/// Returns nullopt for empty inputs or retained entity references.
/// Prefixed element and attribute name tests do not match. Evaluated
/// unregistered functions or variables return a user error.
folly::Expected<std::optional<bool>, Status> evalBoolean(
    std::string_view xml,
    std::string_view path);

/// Evaluates a string XPath expression on a non-empty UTF-8 XML string.
/// Returns a user error on malformed XML or XPath syntax/evaluation errors.
/// Error text does not include the document or entity system identifiers, and
/// an invalid path is reported as a short UTF-8 prefix with controls and
/// invalid bytes replaced, not cut inside a code point.
/// Returns nullopt for empty inputs or retained entity references.
/// Prefixed element and attribute name tests do not match. Evaluated
/// unregistered functions or variables return a user error. A no-match
/// node-set returns an empty string.
folly::Expected<std::optional<std::string>, Status> evalString(
    std::string_view xml,
    std::string_view path);

} // namespace xpath
} // namespace facebook::velox::functions::sparksql
