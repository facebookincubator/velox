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

#include "velox/common/encode/Base64.h"
#include "velox/connectors/hive/iceberg/IcebergConfig.h"
#include "velox/connectors/hive/iceberg/PartitionSpec.h"
#include "velox/connectors/hive/iceberg/TransformEvaluator.h"
#include "velox/connectors/hive/iceberg/TransformExprBuilder.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

class TransformTest : public test::IcebergTestBase {
 protected:
  void testTransform(
      const IcebergPartitionSpecPtr& spec,
      const RowVectorPtr& input,
      const RowVectorPtr& expected) const {
    std::vector<column_index_t> partitionChannels;
    for (const auto& field : spec->fields) {
      partitionChannels.push_back(input->rowType()->getChildIdx(field.name));
    }
    // Build and evaluate transform expressions.
    auto transformExprs = TransformExprBuilder::toExpressions(
        spec,
        partitionChannels,
        input->rowType(),
        std::string(IcebergConfig::kDefaultFunctionPrefix));
    auto transformEvaluator = std::make_unique<TransformEvaluator>(
        transformExprs, connectorQueryCtx_.get());
    auto result = transformEvaluator->evaluate(input);

    ASSERT_EQ(result.size(), expected->childrenSize());
    for (auto i = 0; i < result.size(); ++i) {
      velox::test::assertEqualVectors(expected->childAt(i), result[i]);
    }
  }
};

TEST_F(TransformTest, identity) {
  const auto& rowType =
      ROW({"c0", "c1", "c2", "c3", "c4"},
          {INTEGER(), BIGINT(), VARCHAR(), VARBINARY(), TIMESTAMP()});
  const auto& partitionSpec = createPartitionSpec(
      rowType,
      {
          {0, TransformType::kIdentity, std::nullopt},
          {1, TransformType::kIdentity, std::nullopt},
          {2, TransformType::kIdentity, std::nullopt},
          {3, TransformType::kIdentity, std::nullopt},
          {4, TransformType::kIdentity, std::nullopt},
      });

  const std::vector<VectorPtr> input = {
      makeFlatVector<int32_t>({1, -1}),
      makeFlatVector<int64_t>({1L, -1L}),
      makeFlatVector<std::string>({("test data"), ("")}),
      makeFlatVector<std::string>({("\x01\x02\x03"), ("")}, VARBINARY()),
      makeFlatVector<Timestamp>({Timestamp(0, 0), Timestamp(1609459200, 0)}),
  };

  testTransform(partitionSpec, makeRowVector(input), makeRowVector(input));
}

TEST_F(TransformTest, nulls) {
  const auto& rowType =
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
          {INTEGER(),
           VARCHAR(),
           VARBINARY(),
           DATE(),
           TIMESTAMP(),
           TIMESTAMP(),
           TIMESTAMP()});
  const auto& partitionSpec = createPartitionSpec(
      rowType,
      {
          {0, TransformType::kIdentity, std::nullopt},
          {1, TransformType::kBucket, 8},
          {2, TransformType::kTruncate, 16},
          {3, TransformType::kYear, std::nullopt},
          {4, TransformType::kMonth, std::nullopt},
          {5, TransformType::kDay, std::nullopt},
          {6, TransformType::kHour, std::nullopt},
      });
  testTransform(
      partitionSpec,
      makeRowVector({
          makeNullableFlatVector<int32_t>({std::nullopt}),
          makeNullableFlatVector<std::string>({std::nullopt}),
          makeNullableFlatVector<std::string>({std::nullopt}, VARBINARY()),
          makeNullableFlatVector<int32_t>({std::nullopt}, DATE()),
          makeNullableFlatVector<Timestamp>({std::nullopt}),
          makeNullableFlatVector<Timestamp>({std::nullopt}),
          makeNullableFlatVector<Timestamp>({std::nullopt}),
      }),
      makeRowVector({
          makeNullableFlatVector<int32_t>({std::nullopt}),
          makeNullableFlatVector<int32_t>({std::nullopt}),
          makeNullableFlatVector<std::string>({std::nullopt}, VARBINARY()),
          makeNullableFlatVector<int32_t>({std::nullopt}),
          makeNullableFlatVector<int32_t>({std::nullopt}),
          makeNullableFlatVector<int32_t>({std::nullopt}, DATE()),
          makeNullableFlatVector<int32_t>({std::nullopt}),
      }));
}

TEST_F(TransformTest, bucket) {
  const auto& rowType =
      ROW({"c0", "c1", "c2", "c3", "c4", "c5"},
          {INTEGER(), BIGINT(), VARCHAR(), VARBINARY(), DATE(), TIMESTAMP()});
  const auto& partitionSpec = createPartitionSpec(
      rowType,
      {
          {0, TransformType::kBucket, 4},
          {1, TransformType::kBucket, 8},
          {2, TransformType::kBucket, 16},
          {3, TransformType::kBucket, 32},
          {4, TransformType::kBucket, 10},
          {5, TransformType::kBucket, 8},
      });

  testTransform(
      partitionSpec,
      makeRowVector({
          makeFlatVector<int32_t>({8, 34, 0}),
          makeFlatVector<int64_t>({34L, 0L, -34L}),
          makeFlatVector<std::string>({"abcdefg", "测试", ""}),
          makeFlatVector<std::string>(
              {"\x61\x62\x64\x00\x00", "\x01\x02\x03\x04", "\x00"},
              VARBINARY()),
          makeFlatVector<int32_t>({0, 365, 18'262}),
          makeFlatVector<Timestamp>(
              {Timestamp(0, 0),
               Timestamp(-31536000, 0),
               Timestamp(1612224000, 0)}),
      }),
      makeRowVector({
          makeFlatVector<int32_t>({3, 3, 0}),
          makeFlatVector<int32_t>({3, 4, 5}),
          makeFlatVector<int32_t>({6, 8, 0}),
          makeFlatVector<int32_t>({26, 5, 0}),
          makeFlatVector<int32_t>({6, 1, 3}),
          makeFlatVector<int32_t>({4, 3, 5}),
      }));
}

TEST_F(TransformTest, year) {
  const auto& rowType = ROW({"c0", "c1"}, {DATE(), TIMESTAMP()});
  const auto& partitionSpec = createPartitionSpec(
      rowType,
      {
          {0, TransformType::kYear, std::nullopt},
          {1, TransformType::kYear, std::nullopt},
      });

  testTransform(
      partitionSpec,
      makeRowVector({
          makeFlatVector<int32_t>({0, 18'262, -365}),
          makeFlatVector<Timestamp>(
              {Timestamp(0, 0),
               Timestamp(31536000, 0),
               Timestamp(-31536000, 0)}),
      }),
      makeRowVector({
          makeFlatVector<int32_t>({0, 50, -1}),
          makeFlatVector<int32_t>({0, 1, -1}),
      }));
}

TEST_F(TransformTest, month) {
  const auto& rowType = ROW({"c0", "c1"}, {DATE(), TIMESTAMP()});
  const auto& partitionSpec = createPartitionSpec(
      rowType,
      {
          {0, TransformType::kMonth, std::nullopt},
          {1, TransformType::kMonth, std::nullopt},
      });

  testTransform(
      partitionSpec,
      makeRowVector({
          makeFlatVector<int32_t>({0, 18'262, -365}),
          makeFlatVector<Timestamp>(
              {Timestamp(0, 0),
               Timestamp(31536000, 0),
               Timestamp(-2678400, 0)}),
      }),
      makeRowVector({
          makeFlatVector<int32_t>({0, 600, -12}),
          makeFlatVector<int32_t>({0, 12, -1}),
      }));
}

TEST_F(TransformTest, day) {
  const auto& rowType = ROW({"c0", "c1"}, {DATE(), TIMESTAMP()});
  const auto& partitionSpec = createPartitionSpec(
      rowType,
      {
          {0, TransformType::kDay, std::nullopt},
          {1, TransformType::kDay, std::nullopt},
      });

  testTransform(
      partitionSpec,
      makeRowVector({
          makeFlatVector<int32_t>({0, 17532, -1}, DATE()),
          makeFlatVector<Timestamp>(
              {Timestamp(0, 0),
               Timestamp(1514764800, 0),
               Timestamp(-86400, 0)}),
      }),
      makeRowVector({
          makeFlatVector<int32_t>({0, 17532, -1}, DATE()),
          makeFlatVector<int32_t>({0, 17532, -1}, DATE()),
      }));
}

TEST_F(TransformTest, hour) {
  const auto& partitionSpec = createPartitionSpec(
      ROW({"c0"}, {TIMESTAMP()}), {{0, TransformType::kHour, std::nullopt}});

  testTransform(
      partitionSpec,
      makeRowVector({makeFlatVector<Timestamp>({
          Timestamp(0, 0),
          Timestamp(3600, 0),
          Timestamp(-3600, 0),
      })}),
      makeRowVector({makeFlatVector<int32_t>({0, 1, -1})}));
}

TEST_F(TransformTest, truncate) {
  const auto& rowType = ROW(
      {"c0", "c1", "c2", "c3"}, {INTEGER(), BIGINT(), VARCHAR(), VARBINARY()});
  const auto& partitionSpec = createPartitionSpec(
      rowType,
      {
          {0, TransformType::kTruncate, 10},
          {1, TransformType::kTruncate, 100},
          {2, TransformType::kTruncate, 5},
          {3, TransformType::kTruncate, 3},
      });

  testTransform(
      partitionSpec,
      makeRowVector({
          makeFlatVector<int32_t>({11, -11, 5}),
          makeFlatVector<int64_t>({123L, -123L, 50L}),
          makeFlatVector<std::string>({"abcdefg", "测试data", "x"}),
          makeFlatVector<std::string>(
              {"abcdefg", "\x01\x02\x03\x04", "\x05"}, VARBINARY()),
      }),
      makeRowVector({
          makeFlatVector<int32_t>({10, -20, 0}),
          makeFlatVector<int64_t>({100L, -200L, 0L}),
          makeFlatVector<std::string>({"abcde", "测试dat", "x"}),
          makeFlatVector<std::string>(
              {"abc", "\x01\x02\x03", "\x05"}, VARBINARY()),
      }));
}

TEST_F(TransformTest, multipleTransforms) {
  const auto& rowType = ROW({"c0", "c1", "c2"}, {INTEGER(), DATE(), VARCHAR()});
  const auto& partitionSpec = createPartitionSpec(
      rowType,
      {
          {0, TransformType::kBucket, 4},
          {1, TransformType::kYear, std::nullopt},
          {2, TransformType::kTruncate, 3},
      });

  testTransform(
      partitionSpec,
      makeRowVector({
          makeFlatVector<int32_t>({8, 34}),
          makeFlatVector<int32_t>({0, 17532}),
          makeFlatVector<std::string>({"abcdefg", "ab c"}),
      }),
      makeRowVector({
          makeFlatVector<int32_t>({3, 3}),
          makeFlatVector<int32_t>({0, 48}),
          makeFlatVector<std::string>({"abc", "ab "}),
      }));
}

TEST_F(TransformTest, multipleTransformsOnSameColumn) {
  const auto& rowType = ROW({"c0", "c1"}, {DATE(), VARCHAR()});
  const auto& partitionSpec = createPartitionSpec(
      rowType,
      {
          {0, TransformType::kYear, std::nullopt},
          {0, TransformType::kBucket, 10},
          {1, TransformType::kTruncate, 5},
          {1, TransformType::kBucket, 8},
      });

  testTransform(
      partitionSpec,
      makeRowVector(
          rowType->names(),
          {
              makeFlatVector<int32_t>({0, 17532}),
              makeFlatVector<std::string>({"abcdefg", "test"}),
          }),
      makeRowVector({
          makeFlatVector<int32_t>({0, 48}),
          makeFlatVector<int32_t>({6, 7}),
          makeFlatVector<std::string>({"abcde", "test"}),
          makeFlatVector<int32_t>({6, 3}),
      }));
}

} // namespace

} // namespace facebook::velox::connector::hive::iceberg
