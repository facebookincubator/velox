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

#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

#include <folly/json.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

#ifdef VELOX_ENABLE_PARQUET

class IcebergEqualityDeleteWriterTest : public test::IcebergTestBase {
 protected:
  std::vector<std::string> write(
      const std::vector<RowVectorPtr>& vectors,
      const std::string& outputDirectory,
      std::vector<int32_t> equalityFieldIds,
      const std::vector<test::PartitionField>& partitionFields = {}) {
    auto sink = createDataSinkAndAppendData(
        vectors,
        outputDirectory,
        partitionFields,
        IcebergInsertTableHandle::WriteKind::kEqualityDelete,
        std::move(equalityFieldIds));
    return sink->close();
  }

  void assertCommitMetadata(
      const std::vector<std::string>& commitMessages,
      const std::vector<int32_t>& equalityFieldIds,
      bool expectPartitionData = false) {
    ASSERT_FALSE(commitMessages.empty());
    for (const auto& message : commitMessages) {
      const auto commit = folly::parseJson(message);
      ASSERT_EQ(commit["content"].asString(), "EQUALITY_DELETES");
      ASSERT_TRUE(commit["equalityFieldIds"].isArray());
      ASSERT_EQ(commit["equalityFieldIds"].size(), equalityFieldIds.size());
      for (size_t i = 0; i < equalityFieldIds.size(); ++i) {
        EXPECT_EQ(commit["equalityFieldIds"][i].asInt(), equalityFieldIds[i]);
      }
      EXPECT_EQ(commit.count("partitionDataJson"), expectPartitionData ? 1 : 0);
    }
  }

  void assertWrittenRows(
      const std::string& outputDirectory,
      const RowTypePtr& rowType,
      const std::vector<RowVectorPtr>& expected) {
    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(rowType)
                    .endTableScan()
                    .planNode();
    exec::test::AssertQueryBuilder(plan)
        .splits(createSplitsForDirectory(outputDirectory))
        .assertResults(expected);
  }
};

TEST_F(IcebergEqualityDeleteWriterTest, singleEqualityField) {
  const auto outputDirectory = test::TempDirectoryPath::create();
  const auto rowType = ROW({"id"}, {BIGINT()});
  const std::vector<RowVectorPtr> vectors = {
      makeRowVector(rowType->names(), {makeFlatVector<int64_t>({11, 22, 33})})};

  const auto commitMessages = write(vectors, outputDirectory->getPath(), {1});

  ASSERT_EQ(commitMessages.size(), 1);
  assertCommitMetadata(commitMessages, {1});
  assertWrittenRows(outputDirectory->getPath(), rowType, vectors);
}

TEST_F(IcebergEqualityDeleteWriterTest, compositeEqualityFields) {
  const auto outputDirectory = test::TempDirectoryPath::create();
  const auto rowType = ROW({"id", "category"}, {BIGINT(), VARCHAR()});
  const std::vector<RowVectorPtr> vectors = {
      makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({1, 2}),
           makeFlatVector<std::string>({"a", "b"})}),
      makeRowVector(
          rowType->names(),
          {makeFlatVector<int64_t>({3, 4}),
           makeFlatVector<std::string>({"c", "d"})})};

  const auto commitMessages =
      write(vectors, outputDirectory->getPath(), {1, 2});

  ASSERT_EQ(commitMessages.size(), 1);
  assertCommitMetadata(commitMessages, {1, 2});
  assertWrittenRows(outputDirectory->getPath(), rowType, vectors);
}

TEST_F(IcebergEqualityDeleteWriterTest, partitioned) {
  const auto outputDirectory = test::TempDirectoryPath::create();
  const auto rowType = ROW({"category", "id"}, {VARCHAR(), BIGINT()});
  const std::vector<RowVectorPtr> vectors = {makeRowVector(
      rowType->names(),
      {makeFlatVector<std::string>({"a", "a", "b", "b"}),
       makeFlatVector<int64_t>({1, 2, 3, 4})})};
  const std::vector<test::PartitionField> partitionFields = {
      {0, TransformType::kIdentity, std::nullopt}};

  const auto commitMessages =
      write(vectors, outputDirectory->getPath(), {2}, partitionFields);

  ASSERT_EQ(commitMessages.size(), 2);
  assertCommitMetadata(commitMessages, {2}, true);
  assertWrittenRows(outputDirectory->getPath(), rowType, vectors);
}

#endif

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
