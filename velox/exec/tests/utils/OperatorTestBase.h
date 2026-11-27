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

#include <folly/executors/IOThreadPoolExecutor.h>
#include <gtest/gtest.h>

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/core/Expressions.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::exec::test {
class OperatorTestBase : public virtual testing::Test,
                         public velox::test::VectorTestBase {
 public:
  /// Sets up the memory system with leak checks and the shared arbitrator,
  /// registers Presto scalar and aggregate functions, and enables TestValue.
  /// Called once per test suite. Public because benchmarks such as ReduceAgg
  /// call it to set up the same environment.
  static void SetUpTestCase();

  /// Shuts down the async data cache, waits for all tasks to be deleted, and
  /// unregisters the shared arbitrator. Called once per test suite.
  static void TearDownTestCase();

  /// Sets up the velox memory system.
  ///
  /// NOTE: a second call to this will clear the previous memory system
  /// instances and create a new set.
  static void setupMemory(
      int64_t allocatorCapacity,
      int64_t arbitratorCapacity,
      int64_t arbitratorReservedCapacity,
      int64_t memoryPoolInitCapacity,
      int64_t memoryPoolReservedCapacity,
      int64_t memoryPoolMinReclaimBytes,
      int64_t memoryPoolAbortCapacityLimit);

  static void resetMemory();

 protected:
  OperatorTestBase();
  ~OperatorTestBase() override;

  /// Registers the default vector serde via registerVectorSerde() and the
  /// named Presto, CompactRow and UnsafeRow serdes, creates driver and IO
  /// executors, and starts the periodic stats reporter and the local exchange
  /// source.
  void SetUp() override;

  /// Waits for all tasks to be deleted, stops the stats reporter and the local
  /// exchange source, releases memory pools, and resets the memory system.
  void TearDown() override;

  /// Allow base classes to register custom vector serde.
  /// By default, registers Presto-compatible serde.
  virtual void registerVectorSerde();

  void createDuckDbTable(const std::vector<RowVectorPtr>& data) {
    duckDbQueryRunner_.createTable("tmp", data);
  }

  void createDuckDbTable(
      const std::string& tableName,
      const std::vector<RowVectorPtr>& data) {
    duckDbQueryRunner_.createTable(tableName, data);
  }

  TaskPtr assertQuery(
      const core::PlanNodePtr& plan,
      const std::string& duckDbSql);

  TaskPtr assertQuery(
      const core::PlanNodePtr& plan,
      const RowVectorPtr& expectedResults);

  /// Assumes plan has a single leaf node. All splits are added to that node.
  TaskPtr assertQuery(
      const core::PlanNodePtr& plan,
      const std::vector<std::shared_ptr<connector::ConnectorSplit>>&
          connectorSplits,
      const std::string& duckDbSql);

  /// Assumes plan has a single leaf node. All splits are added to that node.
  TaskPtr assertQuery(
      const core::PlanNodePtr& plan,
      std::vector<exec::Split>&& splits,
      const std::string& duckDbSql);

  /// Parses a DuckDB SQL expression and resolves its types against 'rowType'.
  core::TypedExprPtr parseExpr(
      const std::string& sql,
      const RowTypePtr& rowType,
      const parse::ParseOptions& options = {});

 public:
  static void deleteTaskAndCheckSpillDirectory(TaskPtr& task);

 protected:
  // Used as default AsyncDataCache.
  static inline std::shared_ptr<cache::AsyncDataCache> asyncDataCache_;

  DuckDbQueryRunner duckDbQueryRunner_;

  // Used for driver thread execution.
  std::unique_ptr<folly::CPUThreadPoolExecutor> driverExecutor_;

  // Used for IO prefetch and spilling.
  std::unique_ptr<folly::IOThreadPoolExecutor> ioExecutor_;
};
} // namespace facebook::velox::exec::test
