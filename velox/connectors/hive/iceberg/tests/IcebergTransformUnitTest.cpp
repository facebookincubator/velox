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
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

namespace facebook::velox::connector::hive::iceberg::test {

class IcebergTransformUnitTest : public IcebergTestBase {
 protected:
  template <typename TIN, typename TOUT>
  void testTransform(
      const IcebergPartitionSpec::Field& field,
      const std::vector<TIN>& inputValues,
      const std::vector<std::optional<TOUT>>& expectedValues,
      const TypePtr& type = nullptr) {
    VectorPtr inputVector;
    std::vector<std::shared_ptr<Transform>> transforms =
        parsePartitionTransformSpecs({field}, opPool_.get());
    auto transform = transforms[0];
    if constexpr (std::is_same_v<TIN, StringView>) {
      auto size = inputValues.size();
      auto vectorType = type ? type : VARCHAR();
      inputVector = BaseVector::create<FlatVector<StringView>>(
          vectorType, size, opPool_.get());
      const auto flatVector = inputVector->asFlatVector<StringView>();
      for (vector_size_t i = 0; i < size; i++) {
        if (i < inputValues.size()) {
          flatVector->set(i, inputValues[i]);
        } else {
          flatVector->setNull(i, true);
        }
      }
    } else {
      auto size = inputValues.size();
      inputVector = BaseVector::create<FlatVector<TIN>>(
          type ? type : CppToType<TIN>::create(), size, opPool_.get());
      const auto flatVector = inputVector->asFlatVector<TIN>();
      for (vector_size_t i = 0; i < size; i++) {
        if (i < inputValues.size()) {
          flatVector->set(i, inputValues[i]);
        } else {
          flatVector->setNull(i, true);
        }
      }
    }

    std::vector<VectorPtr> children = {inputVector};
    std::vector<std::string> names = {field.name};
    auto rowVector = makeRowVector(names, children);
    const auto resultVector = transform->transform(rowVector, 0);

    ASSERT_EQ(resultVector->size(), expectedValues.size());
    for (vector_size_t i = 0; i < resultVector->size(); i++) {
      if (expectedValues[i].has_value()) {
        if constexpr (
            std::is_same_v<TIN, StringView> &&
            std::is_same_v<TOUT, StringView>) {
          if (type && type->isVarbinary()) {
            EXPECT_EQ(
                resultVector->as<SimpleVector<TIN>>()->valueAt(i).str(),
                encoding::Base64::encode(expectedValues[i].value().str()));
          } else {
            EXPECT_EQ(
                resultVector->as<SimpleVector<TIN>>()->valueAt(i).str(),
                expectedValues[i].value().str());
          }
        } else {
          EXPECT_EQ(
              resultVector->as<SimpleVector<TOUT>>()->valueAt(i),
              expectedValues[i].value());
        }
      } else {
        EXPECT_TRUE(resultVector->isNullAt(i));
      }
    }
  }
};

TEST_F(IcebergTransformUnitTest, testIdentityTransform) {
  rowType_ =
      ROW({"c_int",
           "c_bigint",
           "c_varchar",
           "c_date",
           "c_varbinary",
           "c_decimal",
           "c_timestamp"},
          {INTEGER(),
           BIGINT(),
           VARCHAR(),
           DATE(),
           VARBINARY(),
           DECIMAL(18, 3),
           TIMESTAMP()});

  // Create partition spec with identity transforms.
  const auto partitionSpec = createPartitionSpec(
      {{0, TransformType::kIdentity, std::nullopt}, // c_int.
       {1, TransformType::kIdentity, std::nullopt}, // c_bigint.
       {2, TransformType::kIdentity, std::nullopt}, // c_varchar.
       {4, TransformType::kIdentity, std::nullopt}, // c_varbinary.
       {5, TransformType::kIdentity, std::nullopt}, // c_decimal.
       {6, TransformType::kIdentity, std::nullopt}}, // c_timestamp.
      rowType_);

  auto& intTransform = partitionSpec->fields[0];
  EXPECT_EQ(intTransform.transformType, TransformType::kIdentity);
  testTransform<int32_t, int32_t>(
      intTransform,
      {1,
       0,
       -1,
       std::numeric_limits<int32_t>::min(),
       std::numeric_limits<int32_t>::max()},
      {1,
       0,
       -1,
       std::numeric_limits<int32_t>::min(),
       std::numeric_limits<int32_t>::max()});

  auto& bigintTransform = partitionSpec->fields[1];
  EXPECT_EQ(bigintTransform.transformType, TransformType::kIdentity);
  EXPECT_EQ(bigintTransform.type->kind(), TypeKind::BIGINT);
  testTransform<int64_t, int64_t>(
      bigintTransform,
      {1L,
       0L,
       -1L,
       std::numeric_limits<int64_t>::min(),
       std::numeric_limits<int64_t>::max()},
      {1,
       0,
       -1,
       std::numeric_limits<int64_t>::min(),
       std::numeric_limits<int64_t>::max()});

  auto& varcharTransform = partitionSpec->fields[2];
  EXPECT_EQ(varcharTransform.transformType, TransformType::kIdentity);
  EXPECT_EQ(varcharTransform.type->kind(), TypeKind::VARCHAR);
  testTransform<StringView, StringView>(
      varcharTransform,
      {StringView("a"),
       StringView(""),
       StringView("velox"),
       StringView(
           "Velox is a composable execution engine distributed as an open source C++ library. It provides reusable, extensible, and high-performance data processing components that can be (re-)used to build data management systems focused on different analytical workloads, including batch, interactive, stream processing, and AI/ML. Velox was created by Meta and it is currently developed in partnership with IBM/Ahana, Intel, Voltron Data, Microsoft, ByteDance and many other companies.")},
      {StringView("a"),
       StringView(""),
       StringView("velox"),
       StringView(
           "Velox is a composable execution engine distributed as an open source C++ library. It provides reusable, extensible, and high-performance data processing components that can be (re-)used to build data management systems focused on different analytical workloads, including batch, interactive, stream processing, and AI/ML. Velox was created by Meta and it is currently developed in partnership with IBM/Ahana, Intel, Voltron Data, Microsoft, ByteDance and many other companies.")});

  auto& varbinaryTransform = partitionSpec->fields[3];
  EXPECT_EQ(varbinaryTransform.transformType, TransformType::kIdentity);
  EXPECT_EQ(varbinaryTransform.type->kind(), TypeKind::VARBINARY);
  testTransform<StringView, StringView>(
      varbinaryTransform,
      {
          StringView("\x01\x02\x03", 3),
          StringView("\x04\x05\x06\x07", 4),
          StringView("\x08\x09", 2),
          StringView("", 0),
          StringView("\xFF\xFE\xFD\xFC", 4),
      },
      {
          StringView("\x01\x02\x03", 3),
          StringView("\x04\x05\x06\x07", 4),
          StringView("\x08\x09", 2),
          StringView("", 0),
          StringView("\xFF\xFE\xFD\xFC", 4),
      },
      VARBINARY());

  auto& timestampTransform = partitionSpec->fields[5];
  EXPECT_EQ(timestampTransform.transformType, TransformType::kIdentity);
  EXPECT_EQ(timestampTransform.type->kind(), TypeKind::TIMESTAMP);
  testTransform<Timestamp, Timestamp>(
      timestampTransform,
      {
          Timestamp(0, 0),
          Timestamp(1609459200, 0),
          Timestamp(1640995200, 0),
          Timestamp(1672531200, 0),
          Timestamp(9223372036854775, 999999999),
      },
      {
          Timestamp(0, 0),
          Timestamp(1609459200, 0),
          Timestamp(1640995200, 0),
          Timestamp(1672531200, 0),
          Timestamp(9223372036854775, 999999999),
      });
}

TEST_F(IcebergTransformUnitTest, testTruncateTransform) {
  rowType_ =
      ROW({"c_int", "c_decimal", "c_varchar", "c_varbinary"},
          {INTEGER(), DECIMAL(18, 3), VARCHAR(), VARBINARY()});

  const auto partitionSpec = createPartitionSpec(
      {{0, TransformType::kTruncate, 10},
       {1, TransformType::kTruncate, 10},
       {2, TransformType::kTruncate, 2},
       {3, TransformType::kTruncate, 3}},
      rowType_);

  auto& intTruncateTransform = partitionSpec->fields[0];
  testTransform<int32_t, int32_t>(
      intTruncateTransform,
      {
          std::numeric_limits<int32_t>::min(),
          std::numeric_limits<int32_t>::min() + 1,
          std::numeric_limits<int32_t>::min() + 9,
          std::numeric_limits<int32_t>::min() + 10,
          -1,
          0,
          1,
          9,
          std::numeric_limits<int32_t>::max() - 10,
          std::numeric_limits<int32_t>::max() - 9,
          std::numeric_limits<int32_t>::max() - 1,
          std::numeric_limits<int32_t>::max(),
      },
      {
          2'147'483'646,
          2'147'483'646,
          -2'147'483'640,
          -2'147'483'640,
          -10,
          0,
          0,
          0,
          2'147'483'630,
          2'147'483'630,
          2'147'483'640,
          2'147'483'640,
      });

  auto& decimalTruncateTransform = partitionSpec->fields[1];
  testTransform<int64_t, int64_t>(
      decimalTruncateTransform,
      {
          5000,
          5010,
          5011,
          5019,
          5020,
          5021,
          -5000,
          -5010,
          -5011,
          -5019,
          -5020,
          -5021,
          1234,
          1230,
          1229,
          5,
          -5,
          -10,
          -9,
          -1,
          0,
          1,
          9,
          10,
          995,
          1000,
          1005,
          1010,

          // Large values.
          999'999'999'999'999'990L,
          999'999'999'999'999'995L,
          999'999'999'999'999'999L,
          // Small values.
          -999'999'999'999'999'990L,
          -999'999'999'999'999'995L,
          -999'999'999'999'999'999L,
      },
      {
          5000,
          5010,
          5010,
          5010,
          5020,
          5020,
          -5000,
          -5010,
          -5020,
          -5020,
          -5020,
          -5030,
          1230,
          1230,
          1220,
          0,
          -10,
          -10,
          -10,
          -10,
          0,
          0,
          0,
          10,
          990,
          1000,
          1000,
          1010,
          // Expected results for large values.
          999'999'999'999'999'990L,
          999'999'999'999'999'990L,
          999'999'999'999'999'990L,
          // Expected results for small values.
          -999'999'999'999'999'990L,
          -1'000'000'000'000'000'000L,
          -1'000'000'000'000'000'000L,
      });

  auto& varcharTruncateTransform = partitionSpec->fields[2];
  testTransform<StringView, StringView>(
      varcharTruncateTransform,
      {
          StringView(""),
          StringView("a"),
          StringView("ab"),
          StringView("abc"),
          StringView("abcd"),
          StringView("测"), // 1 code point, 3 bytes.
          StringView("测试"), // 2 code points, 6 bytes.
          StringView("测试abc"), // 5 code points.
          StringView("a测b试c"), // 5 code points.
          StringView("🚀"), // 1 code point, 4 bytes.
          StringView("🚀🔥"), // 2 code points, 8 bytes.
          StringView("abc🚀🔥"), // 5 code points.
          StringView("é"), // 1 code point (e + combining acute accent).
          StringView("éfac"), // 4 code points.
          StringView("a\u0300"), // 'a' + combining grave accent = 1 code point.
      },
      {
          StringView(""),
          StringView("a"),
          StringView("ab"),
          StringView("ab"),
          StringView("ab"),
          StringView("测"),
          StringView("测试"),
          StringView("测试"),
          StringView("a测"),
          StringView("🚀"),
          StringView("🚀🔥"),
          StringView("ab"),
          StringView("é"),
          StringView("éf"),
          StringView("a\u0300"),
      });

  auto& varbinaryTransform = partitionSpec->fields[3];
  testTransform<StringView, StringView>(
      varbinaryTransform,
      {
          StringView("\x01\x02\x03", 3),
          StringView("\x04\x05\x06\x07", 4),
          StringView("\x08\x09", 2),
          StringView("", 0),
          StringView(
              "\xFF\xFE\xFD\xFC\xFA\xFB\xFC\xF1\xF2\xF3\xF4\xF5\xF6\xF7", 14),
      },
      {
          StringView("\x01\x02\x03", 3),
          StringView("\x04\x05\x06", 3),
          StringView("\x08\x09", 2),
          StringView("", 0),
          StringView("\xFF\xFE\xFD", 3),
      },
      VARBINARY());
}

TEST_F(IcebergTransformUnitTest, testBucketTransform) {
  rowType_ =
      ROW({"c_int", "c_bigint", "c_varchar", "c_varbinary", "c_date"},
          {INTEGER(), BIGINT(), VARCHAR(), VARBINARY(), DATE()});

  const auto partitionSpec = createPartitionSpec(
      {{0, TransformType::kBucket, 4},
       {1, TransformType::kBucket, 8},
       {2, TransformType::kBucket, 16},
       {3, TransformType::kBucket, 32},
       {4, TransformType::kBucket, 10}},
      rowType_);

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

TEST_F(IcebergTransformUnitTest, testTemporalTransforms) {
  rowType_ = ROW({"c_date"}, {DATE()});

  const auto partitionSpec = createPartitionSpec(
      {{0, TransformType::kYear, std::nullopt},
       {0, TransformType::kMonth, std::nullopt},
       {0, TransformType::kDay, std::nullopt},
       {0, TransformType::kHour, std::nullopt},
       {0, TransformType::kBucket, 8},
       {0, TransformType::kIdentity, std::nullopt}},
      rowType_);

  auto& yearTransform = partitionSpec->fields[0];
  EXPECT_EQ(yearTransform.transformType, TransformType::kYear);
  // Create test dates (days since epoch).
  testTransform<int32_t, int32_t>(
      yearTransform,
      {
          -36889, // 1869-01-01.
          -18628, // 1919-01-01.
          -365, // 1969-01-01.
          -1, // 1969-12-31.
          0, // 1970-01-01 (epoch).
          31, // 1970-02-01.
          365, // 1971-01-01.
          18'262, // 2020-01-01.
          20'181 // 2025-04-03.
      },
      {
          -101, // 1869 - 1970 = -101.
          -51, // 1919 - 1970 = -51.
          -1, // 1969 - 1970 = -1.
          -1, // 1969 - 1970 = -1.
          0, // 1970 - 1970 = 0.
          0, // 1970 - 1970 = 0.
          1, // 1971 - 1970 = 1.
          50, // 2020 - 1970 = 50.
          55 // 2025 - 1970 = 55.
      });
  // Test month transform.
  auto& monthTransform = partitionSpec->fields[1];
  EXPECT_EQ(monthTransform.transformType, TransformType::kMonth);

  testTransform<int32_t, int32_t>(
      monthTransform,
      {-36525, -18263, -365, -1, 0, 31, 365, 18'262, 20'181},
      {-1201, -600, -12, -1, 0, 1, 12, 600, 663});
  // Test day transform.
  auto& dayTransform = partitionSpec->fields[2];
  EXPECT_EQ(dayTransform.transformType, TransformType::kDay);
  testTransform<int32_t, int32_t>(
      dayTransform,
      {-36525, -18263, -365, -1, 0, 31, 365, 18'262, 20'181},
      {-36525, -18263, -365, -1, 0, 31, 365, 18'262, 20'181});
}

TEST_F(IcebergTransformUnitTest, testTransformOnTimestamp) {
  rowType_ = ROW({"c_timestamp"}, {TIMESTAMP()});

  const auto partitionSpec = createPartitionSpec(
      {{0, TransformType::kYear, std::nullopt},
       {0, TransformType::kMonth, std::nullopt},
       {0, TransformType::kDay, std::nullopt},
       {0, TransformType::kHour, std::nullopt},
       {0, TransformType::kBucket, 8},
       {0, TransformType::kIdentity, std::nullopt}},
      rowType_);

  auto& yearTransform = partitionSpec->fields[0];
  EXPECT_EQ(yearTransform.transformType, TransformType::kYear);
  testTransform<Timestamp, int32_t>(
      yearTransform,
      {
          Timestamp(0, 0),
          Timestamp(31536000, 0), // 1971-01-01 00:00:00.
          Timestamp(1609459200, 0), // 2021-01-01 00:00:00.
          Timestamp(1612224000, 0), // 2021-02-01 00:00:00.
      },
      {
          0, // 1970 - 1970 = 0.
          1, // 1971 - 1970 = 1.
          51, // 2021 - 1970 = 51.
          51 // 2021 - 1970 = 51.
      });

  auto& monthTransform = partitionSpec->fields[1];
  EXPECT_EQ(monthTransform.transformType, TransformType::kMonth);

  testTransform<Timestamp, int32_t>(
      monthTransform,
      {Timestamp(0, 0),
       Timestamp(31536000, 0),
       Timestamp(1609459200, 0),
       Timestamp(1612224000, 0)},
      {0, 12, 612, 613});

  auto& dayTransform = partitionSpec->fields[2];
  EXPECT_EQ(dayTransform.transformType, TransformType::kDay);
  testTransform<Timestamp, int32_t>(
      dayTransform,
      {Timestamp(0, 0),
       Timestamp(31536000, 0),
       Timestamp(1609459200, 0),
       Timestamp(1612224000, 0)},
      {0, 365, 18628, 18660});

  auto& hourTransform = partitionSpec->fields[3];
  EXPECT_EQ(hourTransform.transformType, TransformType::kHour);
  testTransform<Timestamp, int32_t>(
      hourTransform,
      {Timestamp(0, 0),
       Timestamp(31536000, 0),
       Timestamp(1609459200, 0),
       Timestamp(1612224000, 0)},
      {0, 8760, 447072, 447840});

  auto& bucketTransform = partitionSpec->fields[4];
  EXPECT_EQ(bucketTransform.transformType, TransformType::kBucket);
  testTransform<Timestamp, int32_t>(
      bucketTransform,
      {
          Timestamp(0, 0),
          Timestamp(31536000, 0),
          Timestamp(1609459200, 0),
          Timestamp(1612224000, 0),
          Timestamp(-31536000, 0),
      },
      {4, 4, 6, 5, 3});

  auto& identityTransform = partitionSpec->fields[5];
  EXPECT_EQ(identityTransform.transformType, TransformType::kIdentity);
  testTransform<Timestamp, Timestamp>(
      identityTransform,
      {Timestamp(0, 0),
       Timestamp(31536000, 0),
       Timestamp(1609459200, 0),
       Timestamp(1612224000, 0)},
      {Timestamp(0, 0),
       Timestamp(31536000, 0),
       Timestamp(1609459200, 0),
       Timestamp(1612224000, 0)});
}

TEST_F(IcebergTransformUnitTest, testTransformsWithNulls) {
  rowType_ = ROW(
      {"c_int", "c_bigint", "c_decimal", "c_varchar", "c_varbinary", "c_date"},
      {INTEGER(), BIGINT(), DECIMAL(18, 3), VARCHAR(), VARBINARY(), DATE()});

  const auto partitionSpec = createPartitionSpec(
      {{0, TransformType::kIdentity, std::nullopt},
       {2, TransformType::kTruncate, 100},
       {1, TransformType::kBucket, 16},
       {5, TransformType::kYear, std::nullopt},
       {5, TransformType::kMonth, std::nullopt},
       {5, TransformType::kDay, std::nullopt}},
      rowType_);

  auto& identityTransform = partitionSpec->fields[0];
  EXPECT_EQ(identityTransform.transformType, TransformType::kIdentity);

  auto intInput =
      makeNullableFlatVector<int32_t>({5, std::nullopt, 15, std::nullopt, 25});
  std::vector<VectorPtr> children = {intInput};
  std::vector<std::string> names = {identityTransform.name};
  auto rowVector = makeRowVector(names, children);

  std::vector<std::shared_ptr<Transform>> transforms =
      parsePartitionTransformSpecs({identityTransform}, opPool_.get());
  auto transform = transforms[0];
  auto identityResult = transform->transform(rowVector, 0);
  ASSERT_EQ(identityResult->size(), 5);
  EXPECT_EQ(identityResult->as<SimpleVector<int32_t>>()->valueAt(0), 5);
  EXPECT_TRUE(identityResult->isNullAt(1));
  EXPECT_EQ(identityResult->as<SimpleVector<int32_t>>()->valueAt(2), 15);
  EXPECT_TRUE(identityResult->isNullAt(3));
  EXPECT_EQ(identityResult->as<SimpleVector<int32_t>>()->valueAt(4), 25);

  auto& truncateTransform = partitionSpec->fields[1];
  EXPECT_EQ(truncateTransform.transformType, TransformType::kTruncate);

  auto decimalInput = makeNullableFlatVector<int64_t>(
      {5'000, std::nullopt, 15'000, std::nullopt, 25'000});
  children = {decimalInput};
  names = {truncateTransform.name};
  rowVector = makeRowVector(names, children);
  transforms = parsePartitionTransformSpecs({truncateTransform}, opPool_.get());
  transform = transforms[0];
  auto truncateResult = transform->transform(rowVector, 0);
  ASSERT_EQ(truncateResult->size(), 5);
  EXPECT_EQ(truncateResult->as<SimpleVector<int64_t>>()->valueAt(0), 5000);
  EXPECT_TRUE(truncateResult->isNullAt(1));
  EXPECT_EQ(truncateResult->as<SimpleVector<int64_t>>()->valueAt(2), 15'000);
  EXPECT_TRUE(truncateResult->isNullAt(3));
  EXPECT_EQ(truncateResult->as<SimpleVector<int64_t>>()->valueAt(4), 25'000);

  auto& bucketTransform = partitionSpec->fields[2];
  EXPECT_EQ(bucketTransform.transformType, TransformType::kBucket);

  auto bigintInput = makeNullableFlatVector<int64_t>(
      {50L, std::nullopt, 150L, std::nullopt, 250L});
  children = {bigintInput};
  names = {bucketTransform.name};
  rowVector = makeRowVector(names, children);
  transforms = parsePartitionTransformSpecs({bucketTransform}, opPool_.get());
  transform = transforms[0];
  auto bucketResult = transform->transform(rowVector, 0);
  ASSERT_EQ(bucketResult->size(), 5);
  EXPECT_TRUE(bucketResult->isNullAt(1));
  EXPECT_TRUE(bucketResult->isNullAt(3));

  auto& yearTransform = partitionSpec->fields[3];
  EXPECT_EQ(yearTransform.transformType, TransformType::kYear);

  auto dateInput = makeNullableFlatVector<int32_t>(
      {0, std::nullopt, 365, std::nullopt, 20'175});
  children = {dateInput};
  names = {yearTransform.name};
  rowVector = makeRowVector(names, children);
  transforms = parsePartitionTransformSpecs({yearTransform}, opPool_.get());
  transform = transforms[0];
  auto yearResult = transform->transform(rowVector, 0);
  ASSERT_EQ(yearResult->size(), 5);
  EXPECT_EQ(yearResult->as<SimpleVector<int32_t>>()->valueAt(0), 0);
  EXPECT_TRUE(yearResult->isNullAt(1));
  EXPECT_EQ(yearResult->as<SimpleVector<int32_t>>()->valueAt(2), 1);
  EXPECT_TRUE(yearResult->isNullAt(3));
  EXPECT_EQ(yearResult->as<SimpleVector<int32_t>>()->valueAt(4), 55);

  auto& monthTransform = partitionSpec->fields[4];
  EXPECT_EQ(monthTransform.transformType, TransformType::kMonth);
  children = {dateInput};
  names = {monthTransform.name};
  rowVector = makeRowVector(names, children);
  transforms = parsePartitionTransformSpecs({monthTransform}, opPool_.get());
  transform = transforms[0];
  auto monthResult = transform->transform(rowVector, 0);
  ASSERT_EQ(monthResult->size(), 5);
  EXPECT_EQ(monthResult->as<SimpleVector<int32_t>>()->valueAt(0), 0);
  EXPECT_TRUE(monthResult->isNullAt(1));
  EXPECT_EQ(monthResult->as<SimpleVector<int32_t>>()->valueAt(2), 12);
  EXPECT_TRUE(monthResult->isNullAt(3));
  EXPECT_EQ(monthResult->as<SimpleVector<int32_t>>()->valueAt(4), 662);

  auto& dayTransform = partitionSpec->fields[5];
  EXPECT_EQ(dayTransform.transformType, TransformType::kDay);
  names = {dayTransform.name};
  rowVector = makeRowVector(names, children);
  transforms = parsePartitionTransformSpecs({dayTransform}, opPool_.get());
  transform = transforms[0];
  auto dayResult = transform->transform(rowVector, 0);
  ASSERT_EQ(dayResult->size(), 5);
  EXPECT_EQ(dayResult->as<SimpleVector<int32_t>>()->valueAt(0), 0);
  EXPECT_TRUE(dayResult->isNullAt(1));
  EXPECT_EQ(dayResult->as<SimpleVector<int32_t>>()->valueAt(2), 365);
  EXPECT_TRUE(dayResult->isNullAt(3));
  EXPECT_EQ(dayResult->as<SimpleVector<int32_t>>()->valueAt(4), 20'175);

  auto varcharInput = makeNullableFlatVector<StringView>(
      {StringView("abc"),
       std::nullopt,
       StringView("def"),
       std::nullopt,
       StringView("ghi")});

  rowType_ = ROW({"c_varchar"}, {VARCHAR()});
  auto varcharIdentityTransform =
      createPartitionSpec(
          {{0, TransformType::kIdentity, std::nullopt}}, rowType_)
          ->fields[0];
  children = {varcharInput};
  names = {varcharIdentityTransform.name};
  rowVector = makeRowVector(names, children);

  transforms =
      parsePartitionTransformSpecs({varcharIdentityTransform}, opPool_.get());
  transform = transforms[0];
  auto varcharIdentityResult = transform->transform(rowVector, 0);
  ASSERT_EQ(varcharIdentityResult->size(), 5);
  EXPECT_EQ(
      varcharIdentityResult->as<SimpleVector<StringView>>()->valueAt(0).str(),
      "abc");
  EXPECT_TRUE(varcharIdentityResult->isNullAt(1));
  EXPECT_EQ(
      varcharIdentityResult->as<SimpleVector<StringView>>()->valueAt(2).str(),
      "def");
  EXPECT_TRUE(varcharIdentityResult->isNullAt(3));
  EXPECT_EQ(
      varcharIdentityResult->as<SimpleVector<StringView>>()->valueAt(4).str(),
      "ghi");

  auto varbinaryInput = makeNullableFlatVector<StringView>(
      {StringView("\x01\x02\x03", 3),
       std::nullopt,
       StringView("\x04\x05\x06", 3),
       std::nullopt,
       StringView("\x07\x08\x09", 3)},
      VARBINARY());

  rowType_ = ROW({"c_varbinary"}, {VARBINARY()});
  auto varbinaryIdentityTransform =
      createPartitionSpec(
          {{0, TransformType::kIdentity, std::nullopt}}, rowType_)
          ->fields[0];
  children = {varbinaryInput};
  names = {varbinaryIdentityTransform.name};
  rowVector = makeRowVector(names, children);
  transforms =
      parsePartitionTransformSpecs({varbinaryIdentityTransform}, opPool_.get());
  transform = transforms[0];
  auto varbinaryIdentityResult = transform->transform(rowVector, 0);
  ASSERT_EQ(varbinaryIdentityResult->size(), 5);
  EXPECT_TRUE(varbinaryIdentityResult->isNullAt(1));
  EXPECT_TRUE(varbinaryIdentityResult->isNullAt(3));
}

} // namespace facebook::velox::connector::hive::iceberg::test
