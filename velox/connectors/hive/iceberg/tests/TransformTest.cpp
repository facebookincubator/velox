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
#include "velox/connectors/hive/iceberg/PartitionSpec.h"
#include "velox/connectors/hive/iceberg/TransformEvaluator.h"
#include "velox/connectors/hive/iceberg/TransformExprBuilder.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

class TransformTest : public test::IcebergTestBase {
 protected:
  template <typename T>
  void testIdentityTransform(
      const IcebergPartitionSpec::Field& field,
      const std::vector<T>& inputValues,
      const TypePtr& type = nullptr) {
    // Convert input values to optional for testTransform
    std::vector<std::optional<T>> optionalValues;
    optionalValues.reserve(inputValues.size());
    for (const auto& value : inputValues) {
      optionalValues.push_back(value);
    }
    testTransform<T, T>(field, optionalValues, optionalValues, type);
  }

  template <typename IN, typename OUT>
  void testTransform(
      const IcebergPartitionSpec::Field& field,
      const std::vector<std::optional<IN>>& inputValues,
      const std::vector<std::optional<OUT>>& expectedValues,
      const TypePtr& type = nullptr) {
    VectorPtr inputVector;
    std::vector<IcebergPartitionSpec::Field> fields = {field};
    const auto& spec = std::make_shared<const IcebergPartitionSpec>(0, fields);
    std::vector<column_index_t> channels{0};

    if constexpr (std::is_same_v<IN, StringView>) {
      auto size = inputValues.size();
      auto vectorType = type ? type : VARCHAR();
      inputVector = BaseVector::create<FlatVector<StringView>>(
          vectorType, size, opPool_.get());
      const auto& flatVector = inputVector->asFlatVector<StringView>();
      for (vector_size_t i = 0; i < size; i++) {
        if (inputValues[i].has_value()) {
          flatVector->set(i, inputValues[i].value());
        } else {
          flatVector->setNull(i, true);
        }
      }
    } else {
      auto size = inputValues.size();
      inputVector = BaseVector::create<FlatVector<IN>>(
          type ? type : CppToType<IN>::create(), size, opPool_.get());
      const auto& flatVector = inputVector->asFlatVector<IN>();
      for (auto i = 0; i < size; i++) {
        if (inputValues[i].has_value()) {
          flatVector->set(i, inputValues[i].value());
        } else {
          flatVector->setNull(i, true);
        }
      }
    }

    const auto& rowVector = makeRowVector({field.name}, {inputVector});

    // Build transform expressions from partition spec.
    auto transformExprs = TransformExprBuilder::toExpressions(
        spec, channels, asRowType(rowVector->type()));

    // Create evaluator and evaluate expressions.
    auto transformEvaluator = std::make_unique<TransformEvaluator>(
        transformExprs, connectorQueryCtx_.get());
    const auto& resultVector = transformEvaluator->evaluate(rowVector);

    VectorPtr expectedVector;
    if constexpr (std::is_same_v<OUT, StringView>) {
      auto size = expectedValues.size();
      auto vectorType = type ? type : VARCHAR();
      expectedVector = BaseVector::create<FlatVector<StringView>>(
          vectorType, size, opPool_.get());
      const auto& flatVector = expectedVector->asFlatVector<StringView>();
      for (auto i = 0; i < size; i++) {
        if (expectedValues[i].has_value()) {
          flatVector->set(i, expectedValues[i].value());
        } else {
          flatVector->setNull(i, true);
        }
      }
    } else {
      auto size = expectedValues.size();
      expectedVector = BaseVector::create<FlatVector<OUT>>(
          CppToType<OUT>::create(), size, opPool_.get());
      const auto& flatVector = expectedVector->asFlatVector<OUT>();
      for (auto i = 0; i < size; i++) {
        if (expectedValues[i].has_value()) {
          flatVector->set(i, expectedValues[i].value());
        } else {
          flatVector->setNull(i, true);
        }
      }
    }

    ASSERT_EQ(resultVector[0]->size(), expectedValues.size());
    for (auto i = 0; i < resultVector[0]->size(); i++) {
      if (expectedValues[i].has_value()) {
        EXPECT_FALSE(resultVector[0]->isNullAt(i));
        EXPECT_EQ(
            resultVector[0]->as<SimpleVector<OUT>>()->valueAt(i),
            expectedVector->as<SimpleVector<OUT>>()->valueAt(i));
      } else {
        EXPECT_TRUE(resultVector[0]->isNullAt(i));
      }
    }
  }
};

TEST_F(TransformTest, identity) {
  auto partitionSpec = createPartitionSpec(
      {
          {0, TransformType::kIdentity, std::nullopt}, // c_int.
          {1, TransformType::kIdentity, std::nullopt}, // c_bigint.
          {2, TransformType::kIdentity, std::nullopt}, // c_varchar.
          {4, TransformType::kIdentity, std::nullopt}, // c_varbinary.
          {5, TransformType::kIdentity, std::nullopt}, // c_decimal.
          {6, TransformType::kIdentity, std::nullopt}, // c_timestamp.
      },
      ROW(
          {
              "c_int",
              "c_bigint",
              "c_varchar",
              "c_date",
              "c_varbinary",
              "c_decimal",
              "c_timestamp",
          },
          {
              INTEGER(),
              BIGINT(),
              VARCHAR(),
              DATE(),
              VARBINARY(),
              DECIMAL(18, 3),
              TIMESTAMP(),
          }));

  const auto& intTransform = partitionSpec->fields[0];
  EXPECT_EQ(intTransform.transformType, TransformType::kIdentity);
  testIdentityTransform<int32_t>(
      intTransform,
      {
          1,
          0,
          -1,
          std::numeric_limits<int32_t>::min(),
          std::numeric_limits<int32_t>::max(),
      });

  const auto& bigintTransform = partitionSpec->fields[1];
  EXPECT_EQ(bigintTransform.transformType, TransformType::kIdentity);
  EXPECT_EQ(bigintTransform.type->kind(), TypeKind::BIGINT);
  testIdentityTransform<int64_t>(
      bigintTransform,
      {
          1L,
          0L,
          -1L,
          std::numeric_limits<int64_t>::min(),
          std::numeric_limits<int64_t>::max(),
      });

  const auto& varcharTransform = partitionSpec->fields[2];
  EXPECT_EQ(varcharTransform.transformType, TransformType::kIdentity);
  EXPECT_EQ(varcharTransform.type->kind(), TypeKind::VARCHAR);
  testIdentityTransform<StringView>(
      varcharTransform,
      {
          StringView("a"),
          StringView(""),
          StringView("velox"),
          StringView(
              "Velox is a composable execution engine distributed as an open source C++ library. It provides reusable, extensible, and high-performance data processing components that can be (re-)used to build data management systems focused on different analytical workloads, including batch, interactive, stream processing, and AI/ML. Velox was created by Meta and it is currently developed in partnership with IBM/Ahana, Intel, Voltron Data, Microsoft, ByteDance and many other companies."),
      });

  const auto& varbinaryTransform = partitionSpec->fields[3];
  EXPECT_EQ(varbinaryTransform.transformType, TransformType::kIdentity);
  EXPECT_EQ(varbinaryTransform.type->kind(), TypeKind::VARBINARY);
  testIdentityTransform<StringView>(
      varbinaryTransform,
      {
          StringView("\x01\x02\x03", 3),
          StringView("\x04\x05\x06\x07", 4),
          StringView("\x08\x09", 2),
          StringView("", 0),
          StringView("\xFF\xFE\xFD\xFC", 4),
      },
      VARBINARY());

  const auto& timestampTransform = partitionSpec->fields[5];
  EXPECT_EQ(timestampTransform.transformType, TransformType::kIdentity);
  EXPECT_EQ(timestampTransform.type->kind(), TypeKind::TIMESTAMP);
  testIdentityTransform<Timestamp>(
      timestampTransform,
      {
          Timestamp(0, 0),
          Timestamp(1609459200, 0),
          Timestamp(1640995200, 0),
          Timestamp(1672531200, 0),
          Timestamp(9223372036854775, 999999999),
      });
}

TEST_F(TransformTest, nulls) {
  auto spec = createPartitionSpec(
      {{0, TransformType::kIdentity, std::nullopt}},
      ROW({"c_int"}, {INTEGER()}));

  std::vector<std::optional<int32_t>> intValues = {
      5, std::nullopt, 15, std::nullopt, 25};
  testTransform<int32_t, int32_t>(spec->fields[0], intValues, intValues);

  std::vector<std::optional<StringView>> varcharInput = {
      StringView("abc"),
      std::nullopt,
      StringView("def"),
      std::nullopt,
      StringView("ghi"),
  };

  spec = createPartitionSpec(
      {{0, TransformType::kIdentity, std::nullopt}},
      ROW({"c_varchar"}, {VARCHAR()}));
  testTransform<StringView, StringView>(
      spec->fields[0], varcharInput, varcharInput);

  std::vector<std::optional<StringView>> varbinaryInput = {
      StringView("\x01\x02\x03", 3),
      std::nullopt,
      StringView("\x04\x05\x06", 3),
      std::nullopt,
      StringView("\x07\x08\x09", 3),
  };

  spec = createPartitionSpec(
      {{0, TransformType::kIdentity, std::nullopt}},
      ROW({"c_varbinary"}, {VARBINARY()}));
  testTransform<StringView>(spec->fields[0], varbinaryInput, varbinaryInput);
}

TEST_F(TransformTest, bucketTransform) {
  auto rowType =
      ROW({"c_int", "c_bigint", "c_varchar", "c_varbinary", "c_date"},
          {INTEGER(), BIGINT(), VARCHAR(), VARBINARY(), DATE()});

  const auto partitionSpec = createPartitionSpec(
      {{0, TransformType::kBucket, 4},
       {1, TransformType::kBucket, 8},
       {2, TransformType::kBucket, 16},
       {3, TransformType::kBucket, 32},
       {4, TransformType::kBucket, 10}},
      rowType);

  auto& intBucketTransform = partitionSpec->fields[0];
  EXPECT_EQ(intBucketTransform.transformType, TransformType::kBucket);

  testTransform<int32_t, int32_t>(
      intBucketTransform,
      {8,
       34,
       0,
       1,
       -1,
       42,
       100,
       1000,
       std::numeric_limits<int32_t>::min(),
       std::numeric_limits<int32_t>::max()},
      {3, 3, 0, 0, 0, 2, 0, 0, 0, 2});

  auto& bigintBucketTransform = partitionSpec->fields[1];
  EXPECT_EQ(bigintBucketTransform.transformType, TransformType::kBucket);

  testTransform<int64_t, int32_t>(
      bigintBucketTransform,
      {34L,
       0L,
       -34L,
       -1L,
       1L,
       42L,
       123'456'789L,
       -123'456'789L,
       std::numeric_limits<int64_t>::min(),
       std::numeric_limits<int64_t>::max()},
      {3, 4, 5, 0, 4, 6, 1, 4, 5, 7});

  auto& varcharBucketTransform = partitionSpec->fields[2];
  EXPECT_EQ(varcharBucketTransform.transformType, TransformType::kBucket);

  testTransform<StringView, int32_t>(
      varcharBucketTransform,
      {StringView("abcdefg"),
       StringView("测试"),
       StringView("测试ping试测"),
       StringView(""),
       StringView("🚀🔥"),
       StringView("a\u0300\u0301"), // Combining characters.
       StringView("To be or not to be, that is the question.")},
      {6, 8, 11, 0, 14, 11, 9});

  auto& varbinaryBucketTransform = partitionSpec->fields[3];
  EXPECT_EQ(varbinaryBucketTransform.transformType, TransformType::kBucket);

  testTransform<StringView, int32_t>(
      varbinaryBucketTransform,
      {StringView("abc\0\0", 5),
       StringView("\x01\x02\x03\x04", 4),
       StringView("\xFF\xFE\xFD\xFC", 4),
       StringView("\x00\x00\x00\x00", 4),
       StringView("\xDE\xAD\xBE\xEF", 4),
       StringView(std::string(100, 'x').c_str(), 100)},
      {11, 5, 15, 30, 10, 18},
      VARBINARY());

  auto& dateBucketTransform = partitionSpec->fields[4];
  EXPECT_EQ(dateBucketTransform.transformType, TransformType::kBucket);

  testTransform<int32_t, int32_t>(
      dateBucketTransform,
      {
          0, // 1970-01-01.
          365, // 1971-01-01.
          18'262, // 2020-01-01.
          -365, // 1969-01-01.
          -1, // 1969-12-31.
          20'181, // 2025-04-03.
          -36889, // 1869-01-01.
          18'628 // 2021-01-01.
      },
      {6, 1, 3, 6, 2, 5, 9, 0});
}

} // namespace

} // namespace facebook::velox::connector::hive::iceberg
