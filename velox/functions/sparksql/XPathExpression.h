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

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace facebook::velox::functions::sparksql::xpath {
// Share the JDK operation codes between iterator analysis and numeric coercion.
enum Operation {
  kEnd = -1,
  kEmpty = -2,
  kWildcard = -3,
  kXPath = 1,
  kOr = 2,
  kAnd = 3,
  kNotEqual = 4,
  kEqual = 5,
  kLessEqual = 6,
  kLess = 7,
  kGreaterEqual = 8,
  kGreater = 9,
  kAdd = 10,
  kSubtract = 11,
  kMultiply = 12,
  kDivide = 13,
  kModulo = 14,
  kNegate = 16,
  kUnion = 20,
  kLiteral = 21,
  kVariable = 22,
  kGroup = 23,
  kExtensionFunction = 24,
  kFunction = 25,
  kArgument = 26,
  kNumber = 27,
  kLocation = 28,
  kPredicate = 29,
  kNameTest = 34,
  kRootTest = 35,
  kAncestor = 37,
  kCommentTest = 1030,
  kTextTest = 1031,
  kProcessingInstructionTest = 1032,
  kNodeTest = 1033,
};

/// Retains the rewritten expression and the names needed by its contexts.
struct XPathExpression {
  /// Stores the namespace-unaware expression with strict numeric conversions.
  std::string text;
  /// Stores per-node selectors used by namespace-axis continuations.
  std::vector<std::string> namespaceSelectors;
  /// Stores prefixes from QName tokens, excluding string literals.
  std::vector<std::string> prefixes;
};

/// Normalizes a syntactically valid XPath, rejecting unsupported plans and
/// attempts to call the evaluator's internal functions.
std::optional<XPathExpression> normalizeXPath(std::string_view path);

} // namespace facebook::velox::functions::sparksql::xpath
