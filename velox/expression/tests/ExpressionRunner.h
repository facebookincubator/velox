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

#include <string>
#include "velox/parse/Expressions.h"
#include "velox/vector/TypeAliases.h"

namespace facebook::velox::test {

/// Utility class that helps to run any expressions standalone. It takes input
/// data, SQL, and dirty result vector if any from disk and run the expression
/// described by SQL. It supports 3 modes:
///    - "verify": run expression and compare results between common and
///                simplified path
///    - "common": run expression only using common paths. (to be supported)
///    - "simplified": run expression only using simplified path. (to be
///                supported)
class ExpressionRunner {
 public:
  /// @param inputPath The path to the on-disk vector that will be used as input
  ///        to feed to the expression.
  /// @param sql Comma-separated SQL expressions.
  /// @param complexConstantsPath The path to on-disk vector that stores complex
  ///        subexpressions that aren't expressable in SQL (if any), used with
  ///        sql to construct the complete plan
  /// @param resultPath The path to the on-disk vector
  ///        that will be used as the result buffer to which the expression
  ///        evaluation results will be written.
  /// @param mode The expression evaluation mode, one of ["verify", "common",
  ///        "simplified"]
  /// @param numRows Maximum number of rows to process. 0 means 'all' rows.
  ///         Applies to "common" and "simplified" modes only.
  /// @param storeResultPath The path to a directory on disk where the results
  /// of expression or query evaluation will be stored. If empty, the results
  /// will not be stored.
  /// @param lazyColumnListPath The path to on-disk vector of column indices
  /// that specify which columns of the input row vector should be wrapped in
  /// lazy.
  ///
  /// User can refer to 'VectorSaver' class to see how to serialize/preserve
  /// vectors to disk.
  static void run(
      const std::string& inputPath,
      const std::string& sql,
      const std::string& complexConstantsPath,
      const std::string& resultPath,
      const std::string& mode,
      vector_size_t numRows,
      const std::string& storeResultPath,
      const std::string& lazyColumnListPath);

  /// Parse comma-separated SQL expressions. This should be treated as private
  /// except for tests.
  static std::vector<core::TypedExprPtr> parseSql(
      const std::string& sql,
      const TypePtr& inputType,
      memory::MemoryPool* pool,
      const VectorPtr& complexConstants);
};

} // namespace facebook::velox::test
