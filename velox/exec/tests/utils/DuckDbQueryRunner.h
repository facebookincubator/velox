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

#include "velox/vector/ComplexVector.h"

#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif
#include <duckdb.hpp> // @manual

namespace facebook::velox::exec::test {

using MaterializedRow = std::vector<velox::Variant>;

/// Multiset that compares floating-point values directly.
using MaterializedRowMultiset = std::multiset<MaterializedRow>;

/// Runs SQL queries against an in-memory DuckDB to compute expected results in
/// tests. Not thread-safe: a single instance must be accessed from one thread
/// at a time.
class DuckDbQueryRunner {
 public:
  DuckDbQueryRunner() = default;

  void createTable(
      const std::string& name,
      const std::vector<RowVectorPtr>& data);

  using QueryResult = std::unique_ptr<::duckdb::MaterializedQueryResult>;

  QueryResult execute(const std::string& sql);

  MaterializedRowMultiset execute(
      const std::string& sql,
      const RowTypePtr& resultRowType) {
    MaterializedRowMultiset allRows;
    execute(
        sql,
        resultRowType,
        [&allRows](std::vector<MaterializedRow>& rows) mutable {
          std::copy(
              rows.begin(), rows.end(), std::inserter(allRows, allRows.end()));
        });
    return allRows;
  }

  std::vector<MaterializedRow> executeOrdered(
      const std::string& sql,
      const RowTypePtr& resultRowType) {
    std::vector<MaterializedRow> allRows;
    execute(
        sql,
        resultRowType,
        [&allRows](std::vector<MaterializedRow>& rows) mutable {
          std::copy(rows.begin(), rows.end(), std::back_inserter(allRows));
        });
    return allRows;
  }

 private:
  // Returns the in-memory DuckDB, building it on the first call. Construction
  // is expensive, so it is deferred until the first query.
  ::duckdb::DuckDB& database();

  // shared_ptr (not unique_ptr) keeps DuckDbQueryRunner copyable; some callers
  // hold it by value and rely on copy semantics.
  std::shared_ptr<::duckdb::DuckDB> db_;

  void execute(
      const std::string& sql,
      const RowTypePtr& resultRowType,
      std::function<void(std::vector<MaterializedRow>&)> resultCallback);
};

} // namespace facebook::velox::exec::test
