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

#ifdef VELOX_ENABLE_BACKWARD_COMPATIBILITY
// TODO: Remove once Prestissimo migrates.
#include "velox/exec/HashProbe.h"
#endif

namespace facebook::velox::exec::test {
class OperatorTestBase : public virtual testing::Test,
                         public velox::test::VectorTestBase {
 public:
  /// The following methods are used by google unit test framework to do
  /// one-time setup/teardown for all the unit tests from OperatorTestBase. We
  /// make them public as some benchmark like ReduceAgg also call these methods
  /// to setup/teardown benchmark test environment.
  static void SetUpTestCase();
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

  void SetUp() override;

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

  static std::shared_ptr<core::FieldAccessTypedExpr> toFieldExpr(
      const std::string& name,
      const RowTypePtr& rowType);

  core::TypedExprPtr parseExpr(
      const std::string& text,
      RowTypePtr rowType,
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
