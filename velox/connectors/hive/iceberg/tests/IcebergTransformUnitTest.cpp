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
  struct SchemaField {
    int32_t id;
    std::string name;
    TypePtr type;
  };

  static std::shared_ptr<IcebergPartitionSpec::Schema> createTestSchema(
      const std::vector<SchemaField>& fields) {
    std::unordered_map<std::string, std::int32_t> columnNameToIdMapping;
    std::unordered_map<std::string, TypePtr> columnNameToTypeMapping;

    for (const auto& [id, name, type] : fields) {
      columnNameToIdMapping[name] = id;
      columnNameToTypeMapping[name] = type;
    }

    return std::make_shared<IcebergPartitionSpec::Schema>(
        columnNameToIdMapping, columnNameToTypeMapping);
  }

  template <typename TIN, typename TOUT>
  void testTransform(
      const ColumnTransform& transform,
      const std::vector<TIN>& inputValues,
      const std::vector<std::optional<TOUT>>& expectedValues,
      const TypePtr& type = nullptr) {
    VectorPtr inputVector;
    if constexpr (std::is_same_v<TIN, StringView>) {
      auto size = inputValues.size();
      auto vectorType = type ? type : VARCHAR();
      inputVector = BaseVector::create<FlatVector<StringView>>(
          vectorType, size, opPool_.get());
      const auto flatVector = inputVector->asFlatVector<StringView>();

      for (auto i = 0; i < size; i++) {
        flatVector->set(i, inputValues[i]);
      }
    } else {
      inputVector = makeFlatVector<TIN>(inputValues);
    }

    // Apply transform.
    const auto resultVector = transform.transformVector(inputVector);

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
  const auto schema = createTestSchema(
      {{1, "c_int", INTEGER()},
       {2, "c_bigint", BIGINT()},
       {3, "c_varchar", VARCHAR()},
       {4, "c_varbinary", VARBINARY()},
       {5, "c_decimal", DECIMAL(15, 3)}});

  // Create partition spec with identity transforms.
  const auto partitionSpec = createPartitionSpec(
      schema,
      {"c_int", "c_bigint", "c_varchar", "c_varbinary", "c_decimal"},
      opPool_.get());

  auto& intTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[0];
  EXPECT_EQ(intTransform.transformName(), "identity");
  EXPECT_EQ(intTransform.resultType()->toString(), "INTEGER");
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

  auto& bigintTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[1];
  EXPECT_EQ(bigintTransform.transformName(), "identity");
  EXPECT_EQ(bigintTransform.resultType()->toString(), "BIGINT");
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

  auto& varcharTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[2];
  EXPECT_EQ(varcharTransform.transformName(), "identity");
  EXPECT_EQ(varcharTransform.resultType()->toString(), "VARCHAR");
  testTransform<StringView, StringView>(
      varcharTransform,
      {StringView("a"), StringView(""), StringView("velox")},
      {StringView("a"), StringView(""), StringView("velox")});

  auto& varbinaryTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[3];
  EXPECT_EQ(varbinaryTransform.transformName(), "identity");
  EXPECT_EQ(varbinaryTransform.resultType()->toString(), "VARBINARY");
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
}

TEST_F(IcebergTransformUnitTest, testTruncateTransform) {
  const auto schema = createTestSchema(
      {{1, "c_int", INTEGER()},
       {2, "c_decimal", DECIMAL(18, 3)},
       {3, "c_varchar", VARCHAR()},
       {4, "c_varbinary", VARBINARY()}});

  const auto partitionSpec = createPartitionSpec(
      schema,
      {"truncate(c_int, 10)",
       "truncate(c_decimal, 10)",
       "truncate(c_varchar, 2)",
       "truncate(c_varbinary, 3)"},
      opPool_.get());

  auto& intTruncateTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[0];
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

  auto& decimalTruncateTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[1];
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

  auto& varcharTruncateTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[2];
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
          // 'a' + multiple combining accents = 1 code point.
          StringView("a\u0300\u0301"),
          // Family emoji (multiple code points but visually one character).
          StringView("👨‍👩‍👧‍👦"),
      },
      {
          // Expected results (truncated to 2 code points).
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
          StringView("a\u0300"),
          StringView("👨‍"), // First two code points of the family emoji.
      });
  auto& varbinaryTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[3];
  EXPECT_EQ(varbinaryTransform.transformName(), "trunc");
  EXPECT_EQ(varbinaryTransform.resultType()->toString(), "VARBINARY");
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
  const auto schema = createTestSchema({
      {1, "c_int", INTEGER()},
      {2, "c_bigint", BIGINT()},
      {3, "c_varchar", VARCHAR()},
      {4, "c_varbinary", VARBINARY()},
  });
  const auto partitionSpec = createPartitionSpec(
      schema,
      {"bucket(c_int, 4)",
       "bucket(c_bigint, 8)",
       "bucket(c_varchar, 16)",
       "bucket(c_varbinary, 32)"},
      opPool_.get());

  auto& intBucketTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[0];
  EXPECT_EQ(intBucketTransform.transformName(), "bucket");
  EXPECT_EQ(intBucketTransform.resultType()->toString(), "INTEGER");

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

  auto& bigintBucketTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[1];
  EXPECT_EQ(bigintBucketTransform.transformName(), "bucket");
  EXPECT_EQ(bigintBucketTransform.resultType()->toString(), "INTEGER");

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

  auto& varcharBucketTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[2];
  EXPECT_EQ(varcharBucketTransform.transformName(), "bucket");
  EXPECT_EQ(varcharBucketTransform.resultType()->toString(), "INTEGER");

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

  auto& varbinaryBucketTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[3];
  EXPECT_EQ(varbinaryBucketTransform.transformName(), "bucket");
  EXPECT_EQ(varbinaryBucketTransform.resultType()->toString(), "INTEGER");

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
}

TEST_F(IcebergTransformUnitTest, testTemporalTransforms) {
  const auto schema = createTestSchema({
      {1, "c_date", DATE()},
  });

  const auto partitionSpec = createPartitionSpec(
      schema, {"year(c_date)", "month(c_date)", "day(c_date)"}, opPool_.get());

  auto& yearTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[0];
  EXPECT_EQ(yearTransform.transformName(), "year");
  EXPECT_EQ(yearTransform.resultType()->toString(), "INTEGER");
  // Create test dates (days since epoch).
  testTransform<int32_t, int32_t>(
      yearTransform,
      {
          0, // 1970-01-01 (epoch).
          31, // 1970-02-01.
          365, // 1971-01-01.
          18'262, // 2020-01-01.
          20'181 // 2025-04-03.
      },
      {
          0, // 1970 - 1970 = 0.
          0, // 1970 - 1970 = 0.
          1, // 1971 - 1970 = 1.
          50, // 2020 - 1970 = 50.
          55 // 2025 - 1970 = 55.
      });
  // Test month transform.
  auto& monthTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[1];
  EXPECT_EQ(monthTransform.transformName(), "month");
  EXPECT_EQ(monthTransform.resultType()->toString(), "INTEGER");

  testTransform<int32_t, int32_t>(
      monthTransform, {0, 31, 365, 18'262, 20'181}, {0, 1, 12, 600, 663});
  // Test day transform.
  auto& dayTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[2];
  EXPECT_EQ(dayTransform.transformName(), "day");
  EXPECT_EQ(dayTransform.resultType()->toString(), "INTEGER");
  testTransform<int32_t, int32_t>(
      dayTransform, {0, 31, 365, 18'262, 20'181}, {0, 31, 365, 18'262, 20'181});
}

TEST_F(IcebergTransformUnitTest, testTransformsWithNulls) {
  const auto schema = createTestSchema({
      {1, "c_int", INTEGER()},
      {2, "c_bigint", BIGINT()},
      {3, "c_decimal", DECIMAL(18, 3)},
      {4, "c_varchar", VARCHAR()},
      {5, "c_varbinary", VARBINARY()},
      {6, "c_date", DATE()},
  });

  const auto partitionSpec = createPartitionSpec(
      schema,
      {"c_int",
       "truncate(c_decimal, 100)",
       "bucket(c_bigint, 16)",
       "year(c_date)",
       "month(c_date)",
       "day(c_date)"},
      opPool_.get());

  auto& identityTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[0];
  EXPECT_EQ(identityTransform.transformName(), "identity");

  auto intInput =
      makeNullableFlatVector<int32_t>({5, std::nullopt, 15, std::nullopt, 25});

  auto identityResult = identityTransform.transformVector(intInput);
  ASSERT_EQ(identityResult->size(), 5);
  EXPECT_EQ(identityResult->as<SimpleVector<int32_t>>()->valueAt(0), 5);
  EXPECT_TRUE(identityResult->isNullAt(1));
  EXPECT_EQ(identityResult->as<SimpleVector<int32_t>>()->valueAt(2), 15);
  EXPECT_TRUE(identityResult->isNullAt(3));
  EXPECT_EQ(identityResult->as<SimpleVector<int32_t>>()->valueAt(4), 25);

  auto& truncateTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[1];
  EXPECT_EQ(truncateTransform.transformName(), "trunc");

  auto decimalInput = makeNullableFlatVector<int64_t>(
      {5'000, std::nullopt, 15'000, std::nullopt, 25'000});

  auto truncateResult = truncateTransform.transformVector(decimalInput);
  ASSERT_EQ(truncateResult->size(), 5);
  EXPECT_EQ(truncateResult->as<SimpleVector<int64_t>>()->valueAt(0), 5000);
  EXPECT_TRUE(truncateResult->isNullAt(1));
  EXPECT_EQ(truncateResult->as<SimpleVector<int64_t>>()->valueAt(2), 15'000);
  EXPECT_TRUE(truncateResult->isNullAt(3));
  EXPECT_EQ(truncateResult->as<SimpleVector<int64_t>>()->valueAt(4), 25'000);

  auto& bucketTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[2];
  EXPECT_EQ(bucketTransform.transformName(), "bucket");

  auto bigintInput = makeNullableFlatVector<int64_t>(
      {50L, std::nullopt, 150L, std::nullopt, 250L});

  auto bucketResult = bucketTransform.transformVector(bigintInput);
  ASSERT_EQ(bucketResult->size(), 5);
  EXPECT_TRUE(bucketResult->isNullAt(1));
  EXPECT_TRUE(bucketResult->isNullAt(3));

  auto& yearTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[3];
  EXPECT_EQ(yearTransform.transformName(), "year");

  auto dateInput = makeNullableFlatVector<int32_t>(
      {0, std::nullopt, 365, std::nullopt, 20'175});

  auto yearResult = yearTransform.transformVector(dateInput);
  ASSERT_EQ(yearResult->size(), 5);
  EXPECT_EQ(yearResult->as<SimpleVector<int32_t>>()->valueAt(0), 0);
  EXPECT_TRUE(yearResult->isNullAt(1));
  EXPECT_EQ(yearResult->as<SimpleVector<int32_t>>()->valueAt(2), 1);
  EXPECT_TRUE(yearResult->isNullAt(3));
  EXPECT_EQ(yearResult->as<SimpleVector<int32_t>>()->valueAt(4), 55);

  auto& monthTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[4];
  EXPECT_EQ(monthTransform.transformName(), "month");

  auto monthResult = monthTransform.transformVector(dateInput);
  ASSERT_EQ(monthResult->size(), 5);
  EXPECT_EQ(monthResult->as<SimpleVector<int32_t>>()->valueAt(0), 0);
  EXPECT_TRUE(monthResult->isNullAt(1));
  EXPECT_EQ(monthResult->as<SimpleVector<int32_t>>()->valueAt(2), 12);
  EXPECT_TRUE(monthResult->isNullAt(3));
  EXPECT_EQ(monthResult->as<SimpleVector<int32_t>>()->valueAt(4), 662);

  auto& dayTransform =
      partitionSpec->columnTransforms->getColumnTransforms()[5];
  EXPECT_EQ(dayTransform.transformName(), "day");
  auto dayResult = dayTransform.transformVector(dateInput);
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

  auto varcharIdentityTransform =
      createPartitionSpec(
          createTestSchema({{1, "c_varchar", VARCHAR()}}),
          {"c_varchar"},
          opPool_.get())
          ->columnTransforms->getColumnTransforms()[0];
  auto varcharIdentityResult =
      varcharIdentityTransform.transformVector(varcharInput);
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

  auto varbinaryIdentityTransform =
      createPartitionSpec(
          createTestSchema({{1, "c_varbinary", VARBINARY()}}),
          {"c_varbinary"},
          opPool_.get())
          ->columnTransforms->getColumnTransforms()[0];
  auto varbinaryIdentityResult =
      varbinaryIdentityTransform.transformVector(varbinaryInput);
  ASSERT_EQ(varbinaryIdentityResult->size(), 5);
  EXPECT_TRUE(varbinaryIdentityResult->isNullAt(1));
  EXPECT_TRUE(varbinaryIdentityResult->isNullAt(3));
}

} // namespace facebook::velox::connector::hive::iceberg::test
