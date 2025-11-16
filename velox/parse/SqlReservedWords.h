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

#include <algorithm>
#include <string>
#include <unordered_set>

namespace facebook::velox::core {

/// Checks if a function name is a SQL reserved word that needs escaping.
inline bool isSqlReservedWord(const std::string& name) {
  // Convert to lowercase for case-insensitive comparison
  std::string lowerName = name;
  std::transform(
      lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);

  // Common SQL reserved words that might appear as function names
  static const std::unordered_set<std::string> reservedWords = {
      // Logical operators
      "and",
      "or",
      "not",
      // Comparison and set operators
      "in",
      "is",
      "like",
      "between",
      "exists",
      "all",
      "any",
      "some",
      // Control flow
      "case",
      "when",
      "then",
      "else",
      "end",
      // DML keywords
      "select",
      "from",
      "where",
      "having",
      "group",
      "order",
      "by",
      "as",
      "on",
      // Join types
      "join",
      "inner",
      "outer",
      "left",
      "right",
      "full",
      "cross",
      // Set operations
      "union",
      "intersect",
      "except",
      // Data modification
      "insert",
      "update",
      "delete",
      // DDL
      "create",
      "drop",
      "alter",
      "table",
      // Literals and constants
      "null",
      "true",
      "false",
      // Other common keywords
      "distinct",
      "limit",
      "offset",
      "with",
      "recursive",
  };

  return reservedWords.find(lowerName) != reservedWords.end();
}

} // namespace facebook::velox::core
