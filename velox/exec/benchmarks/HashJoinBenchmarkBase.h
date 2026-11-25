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

#include <functional>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "velox/core/PlanNode.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::exec::test {

/// Base class for hash join benchmarks. A benchmark makes probe and build
/// batches and registers a benchmark for each join to run on them.
///
/// The probe key is 't0' and the build key is 'u0'. Each side may have a
/// second column, 't1' or 'u1', that holds the row number. The join outputs
/// the non-key columns its type allows, plus 'match' for a semi project join.
/// A count(1) on top counts the output rows. The task runs serially.
class HashJoinBenchmarkBase : public velox::test::VectorTestBase {
 public:
  /// A join to run.
  struct Join {
    core::JoinType joinType;

    /// Filter on top of the key equality. Empty means no filter.
    std::string filter{};

    /// Query configs.
    std::unordered_map<std::string, std::string> configs{};

    /// Checks the join output row count against this value when set.
    std::optional<int64_t> expectedNumOutputRows{};
  };

  /// Registers a benchmark that runs 'join' on 'probe' and 'build' and reports
  /// the number of output rows as 'outputRows'. The benchmark is named
  /// '<join type> <name>', followed by ', <filter>' if there is a filter.
  /// 'location' gives the file to list the benchmark under.
  void addBenchmark(
      std::string_view name,
      const Join& join,
      const std::vector<RowVectorPtr>& probe,
      const std::vector<RowVectorPtr>& build,
      std::source_location location = std::source_location::current());

 protected:
  /// Makes batches of 'batchRows' rows.
  explicit HashJoinBenchmarkBase(vector_size_t batchRows);

  /// Returns 'numRows' rows in batches. 'names' has one or two columns. The
  /// first holds 'key(row)' and the second holds the row number.
  std::vector<RowVectorPtr> makeBatches(
      const std::vector<std::string>& names,
      int64_t numRows,
      const std::function<int64_t(int64_t row)>& key);

 private:
  const vector_size_t batchRows_;

  // Plans of the registered benchmarks. They are destroyed before the pool in
  // VectorTestBase that holds their vectors.
  std::vector<core::PlanNodePtr> plans_;
};

} // namespace facebook::velox::exec::test
