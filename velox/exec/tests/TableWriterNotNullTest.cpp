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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/exec/TableWriter.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::exec {
using namespace facebook::velox::common::testutil;
namespace {

// Writes data through a TableWriter whose target columns carry a NOT NULL
// constraint, and checks which nulls the operator rejects and which it lets
// through.
class TableWriterNotNullTest : public HiveConnectorTestBase {
 protected:
  // Plans a write of 'input' into a fresh directory.
  core::PlanNodePtr writePlan(
      const std::vector<RowVectorPtr>& input,
      const folly::F14FastSet<std::string>& notNullColumns) {
    directories_.push_back(TempDirectoryPath::create());
    return PlanBuilder()
        .values(input)
        .startTableWriter()
        .outputDirectoryPath(directories_.back()->getPath())
        .notNullColumns(notNullColumns)
        .endTableWriter()
        .planNode();
  }

  int64_t writtenRows(const core::PlanNodePtr& plan) {
    const auto results = AssertQueryBuilder(plan).copyResults(pool());
    const auto* rowCount = results->childAt(TableWriteTraits::kRowCountChannel)
                               ->as<FlatVector<int64_t>>();
    VELOX_CHECK(!rowCount->isNullAt(0));
    return rowCount->valueAt(0);
  }

  // Keeps the plans' output directories alive.
  std::vector<std::shared_ptr<TempDirectoryPath>> directories_;
};

TEST_F(TableWriterNotNullTest, enforcement) {
  auto data = makeRowVector(
      {"c0", "c1"},
      {
          makeFlatVector<int32_t>({0, 1, 2}),
          makeNullableFlatVector<int32_t>({0, std::nullopt, 2}),
      });

  VELOX_ASSERT_USER_THROW(
      AssertQueryBuilder(writePlan({data}, {"c0", "c1"})).countResults(),
      "NULL value not allowed for NOT NULL column: c1");

  ASSERT_EQ(writtenRows(writePlan({data}, {"c0"})), data->size());
}

// A dictionary wrap decides the column's nulls: it may add nulls the base does
// not have, and hide nulls the base does have.
TEST_F(TableWriterNotNullTest, dictionaryEncoding) {
  auto nullFree = makeFlatVector<int32_t>({0, 1, 2});
  auto withNulls = makeNullableFlatVector<int32_t>({0, std::nullopt, 2});

  // The wrap adds a null over a null-free base.
  auto addsNulls = makeRowVector(
      {"c0"},
      {BaseVector::wrapInDictionary(
          makeNulls({false, true, false}),
          makeIndices({0, 1, 2}),
          3,
          nullFree)});
  VELOX_ASSERT_USER_THROW(
      AssertQueryBuilder(writePlan({addsNulls}, {"c0"})).countResults(),
      "NULL value not allowed for NOT NULL column: c0");

  // The wrap selects only the base's non-null rows.
  auto hidesNulls =
      makeRowVector({"c0"}, {wrapInDictionary(makeIndices({2, 0}), withNulls)});
  ASSERT_EQ(writtenRows(writePlan({hidesNulls}, {"c0"})), 2);

  // The wrap selects some of the base's rows, one of them null.
  auto keepsNull =
      makeRowVector({"c0"}, {wrapInDictionary(makeIndices({2, 1}), withNulls)});
  VELOX_ASSERT_USER_THROW(
      AssertQueryBuilder(writePlan({keepsNull}, {"c0"})).countResults(),
      "NULL value not allowed for NOT NULL column: c0");
}

TEST_F(TableWriterNotNullTest, constantNull) {
  auto data = makeRowVector({"c0"}, {makeNullConstant(TypeKind::INTEGER, 3)});

  VELOX_ASSERT_USER_THROW(
      AssertQueryBuilder(writePlan({data}, {"c0"})).countResults(),
      "NULL value not allowed for NOT NULL column: c0");
}

// A RowVector may be shorter than its children, as in Limit's output. Nulls
// past its size are not part of the batch.
TEST_F(TableWriterNotNullTest, ignoresNullsBeyondBatchSize) {
  const vector_size_t batchSize = 2;
  auto data = std::make_shared<RowVector>(
      pool(),
      ROW("c0", INTEGER()),
      nullptr,
      batchSize,
      std::vector<VectorPtr>{
          makeNullableFlatVector<int32_t>({0, 1, std::nullopt})});

  ASSERT_EQ(writtenRows(writePlan({data}, {"c0"})), batchSize);
}

// The constraint names the table's columns, which may be renamed and reordered
// relative to the input's. PlanBuilder writes under the input names, so build
// the node directly.
TEST_F(TableWriterNotNullTest, renamedColumns) {
  auto data = makeRowVector(
      {"c0", "c1"},
      {
          makeFlatVector<int32_t>({0, 1, 2}),
          makeNullableFlatVector<int32_t>({0, std::nullopt, 2}),
      });
  // 'value' is input column c1, 'key' is c0.
  const auto targetColumns = ROW({"c1", "c0"}, INTEGER());
  const std::vector<std::string> tableColumnNames = {"value", "key"};

  auto directory = TempDirectoryPath::create();
  auto writePlan = [&](const folly::F14FastSet<std::string>& notNullColumns) {
    auto insertHandle = std::make_shared<core::InsertTableHandle>(
        kHiveConnectorId,
        makeHiveInsertTableHandle(
            tableColumnNames,
            targetColumns->children(),
            /*partitionedBy=*/{},
            makeLocationHandle(directory->getPath())),
        notNullColumns);
    return PlanBuilder()
        .values({data})
        .addNode([&](const core::PlanNodeId& nodeId, core::PlanNodePtr source) {
          return std::make_shared<core::TableWriteNode>(
              nodeId,
              targetColumns,
              tableColumnNames,
              /*columnStatsSpec=*/std::nullopt,
              insertHandle,
              /*hasPartitioningScheme=*/false,
              TableWriteTraits::outputType(std::nullopt),
              connector::CommitStrategy::kNoCommit,
              std::move(source));
        })
        .planNode();
  };

  ASSERT_EQ(writtenRows(writePlan({"key"})), data->size());

  VELOX_ASSERT_USER_THROW(
      AssertQueryBuilder(writePlan({"value"})).countResults(),
      "NULL value not allowed for NOT NULL column: value");
}

// The reused DecodedVector must not break on a change of batch size or
// encoding, nor mask a null in a later batch.
TEST_F(TableWriterNotNullTest, multipleBatches) {
  auto flat = makeRowVector({"c0"}, {makeFlatVector<int32_t>({0, 1, 2})});
  auto dictionary = makeRowVector(
      {"c0"},
      {wrapInDictionary(makeIndices({1, 0}), makeFlatVector<int32_t>({0, 1}))});
  auto withNull =
      makeRowVector({"c0"}, {makeNullableFlatVector<int32_t>({std::nullopt})});

  ASSERT_EQ(writtenRows(writePlan({flat, dictionary}, {"c0"})), 5);

  VELOX_ASSERT_USER_THROW(
      AssertQueryBuilder(writePlan({flat, dictionary, withNull}, {"c0"}))
          .countResults(),
      "NULL value not allowed for NOT NULL column: c0");
}

} // namespace
} // namespace facebook::velox::exec
