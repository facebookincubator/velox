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

namespace facebook::velox::exec::test {

using namespace facebook::velox::common::testutil;

// Writes data through a TableWriter whose target columns carry a NOT NULL
// constraint, and checks which nulls the operator rejects and which it lets
// through.
class TableWriterNotNullTest : public HiveConnectorTestBase {
 protected:
  // Plans a write of 'input' into a fresh directory. 'targetColumns', when set,
  // selects and reorders the written columns.
  core::PlanNodePtr writePlan(
      const std::vector<RowVectorPtr>& input,
      const folly::F14FastSet<std::string>& notNullColumns,
      const RowTypePtr& targetColumns = nullptr) {
    directories_.push_back(TempDirectoryPath::create());
    PlanBuilder planBuilder;
    auto& writerBuilder =
        planBuilder.values(input)
            .startTableWriter()
            .outputDirectoryPath(directories_.back()->getPath())
            .notNullColumns(notNullColumns);
    if (targetColumns != nullptr) {
      writerBuilder.targetColumns(targetColumns);
    }
    return writerBuilder.endTableWriter().planNode();
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
  const vector_size_t size = 100;
  auto nonNulls = makeFlatVector<int32_t>(size, [](auto row) { return row; });
  auto values =
      makeFlatVector<int32_t>(size, [](auto row) { return row; }, nullEvery(3));
  auto indices = makeIndices(size, [](auto row) { return row; });
  // The same nulls presented flat, dictionary- and constant-wrapped, and as a
  // struct that is itself null.
  const std::vector<VectorPtr> nullColumns = {
      values,
      wrapInDictionary(indices, size, values),
      makeNullConstant(TypeKind::INTEGER, size),
      makeRowVector({nonNulls}, nullEvery(3)),
  };

  for (const auto& nullColumn : nullColumns) {
    SCOPED_TRACE(nullColumn->toString());
    auto data = makeRowVector({"c0", "c1"}, {nonNulls, nullColumn});

    // Constraining c0 leaves the nulls in c1 alone.
    ASSERT_EQ(writtenRows(writePlan({data}, {"c0"})), size);

    VELOX_ASSERT_USER_THROW(
        AssertQueryBuilder(writePlan({data}, {"c0", "c1"})).countResults(),
        "NULL value not allowed for NOT NULL column: c1");
  }
}

// Nulls inside a struct's fields are the fields' nulls, not the column's.
TEST_F(TableWriterNotNullTest, nullsInsideStruct) {
  const vector_size_t size = 10;
  auto data = makeRowVector(
      {"c0"},
      {makeRowVector({makeFlatVector<int32_t>(
          size, [](auto row) { return row; }, nullEvery(3))})});

  ASSERT_EQ(writtenRows(writePlan({data}, {"c0"})), size);
}

// A RowVector may be shorter than its children, as in Limit's output. Nulls
// past its size are not part of the batch.
TEST_F(TableWriterNotNullTest, ignoresNullsBeyondBatchSize) {
  const vector_size_t childSize = 10;
  const vector_size_t batchSize = 4;
  auto indices = makeIndices(childSize, [](auto row) { return row; });
  auto base = makeFlatVector<int32_t>(
      childSize,
      [](auto row) { return row; },
      [batchSize](auto row) { return row >= batchSize; });
  auto data = std::make_shared<RowVector>(
      pool(),
      ROW({"c0"}, {INTEGER()}),
      nullptr,
      batchSize,
      std::vector<VectorPtr>{wrapInDictionary(indices, childSize, base)});

  ASSERT_EQ(writtenRows(writePlan({data}, {"c0"})), batchSize);
}

// The constraint names target table columns, which may be a reordered subset
// of the input columns.
TEST_F(TableWriterNotNullTest, reorderedColumns) {
  const vector_size_t size = 10;
  auto data = makeRowVector(
      {"c0", "c1", "c2"},
      {
          makeFlatVector<int32_t>(size, [](auto row) { return row; }),
          makeFlatVector<int32_t>(size, [](auto row) { return row * 2; }),
          makeFlatVector<int32_t>(
              size, [](auto row) { return row * 3; }, nullEvery(3)),
      });
  auto targetColumns = ROW({"c2", "c0"}, {INTEGER(), INTEGER()});

  ASSERT_EQ(writtenRows(writePlan({data}, {"c0"}, targetColumns)), size);

  VELOX_ASSERT_USER_THROW(
      AssertQueryBuilder(writePlan({data}, {"c2"}, targetColumns))
          .countResults(),
      "NULL value not allowed for NOT NULL column: c2");
}

// The constraint names the table's columns, which may differ from the input's.
// PlanBuilder always writes under the input names, so build the node directly.
TEST_F(TableWriterNotNullTest, renamedColumns) {
  const vector_size_t size = 10;
  auto data = makeRowVector(
      {"c0", "c1"},
      {
          makeFlatVector<int32_t>(size, [](auto row) { return row; }),
          makeFlatVector<int32_t>(
              size, [](auto row) { return row; }, nullEvery(3)),
      });
  const auto inputColumns = asRowType(data->type());
  const std::vector<std::string> tableColumnNames = {"key", "value"};

  auto directory = TempDirectoryPath::create();
  auto writePlan = [&](const folly::F14FastSet<std::string>& notNullColumns) {
    auto insertHandle = std::make_shared<core::InsertTableHandle>(
        kHiveConnectorId,
        makeHiveInsertTableHandle(
            tableColumnNames,
            inputColumns->children(),
            /*partitionedBy=*/{},
            makeLocationHandle(directory->getPath())),
        notNullColumns);
    return PlanBuilder()
        .values({data})
        .addNode([&](const core::PlanNodeId& nodeId, core::PlanNodePtr source) {
          return std::make_shared<core::TableWriteNode>(
              nodeId,
              inputColumns,
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

  // 'key' maps to input column c0, which has no nulls.
  ASSERT_EQ(writtenRows(writePlan({"key"})), size);

  VELOX_ASSERT_USER_THROW(
      AssertQueryBuilder(writePlan({"value"})).countResults(),
      "NULL value not allowed for NOT NULL column: value");
}

// The reused DecodedVector must not break on a change of size or encoding, nor
// mask a null in a later batch.
TEST_F(TableWriterNotNullTest, multipleBatches) {
  auto makeBatch = [&](vector_size_t size, bool dictionary, bool hasNulls) {
    std::function<bool(vector_size_t)> isNullAt;
    if (hasNulls) {
      isNullAt = nullEvery(3);
    }
    auto values =
        makeFlatVector<int32_t>(size, [](auto row) { return row; }, isNullAt);
    if (!dictionary) {
      return makeRowVector({"c0"}, {values});
    }
    auto indices = makeIndices(size, [](auto row) { return row; });
    return makeRowVector({"c0"}, {wrapInDictionary(indices, size, values)});
  };

  ASSERT_EQ(
      writtenRows(writePlan(
          {makeBatch(10, false, false),
           makeBatch(50, true, false),
           makeBatch(1, true, false)},
          {"c0"})),
      61);

  VELOX_ASSERT_USER_THROW(
      AssertQueryBuilder(
          writePlan(
              {makeBatch(10, false, false), makeBatch(5, true, true)}, {"c0"}))
          .countResults(),
      "NULL value not allowed for NOT NULL column: c0");
}

} // namespace facebook::velox::exec::test
