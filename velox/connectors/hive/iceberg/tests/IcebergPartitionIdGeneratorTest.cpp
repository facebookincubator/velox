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

#include "velox/connectors/hive/iceberg/IcebergPartitionIdGenerator.h"
#include "velox/common/encode/Base64.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

namespace facebook::velox::connector::hive::iceberg {

using namespace facebook::velox;

namespace {

class IcebergPartitionIdGeneratorTest : public test::IcebergTestBase {
 protected:
  // Creates a generator with partition channels [0, 1, 2, ...] corresponding
  // to the input columns in order. All input vectors must have the same size.
  //
  // @param columnNames Names of the partition columns, each name should be
  // unique.
  // @param types Data types of the partition columns.
  // @param transformTypes Transform types to apply to each column (e.g.,
  // kIdentity, kBucket, kYear).
  // @param parameters Optional transform parameters (e.g., bucket count for
  // kBucket, width for kTruncate). If empty or shorter than columnNames,
  // missing parameters default to std::nullopt.
  // @return A configured IcebergPartitionIdGenerator with maxPartitions=128.
  std::unique_ptr<IcebergPartitionIdGenerator> createGenerator(
      const std::vector<std::string>& columnNames,
      const std::vector<TypePtr>& types,
      const std::vector<TransformType>& transformTypes,
      const std::vector<std::optional<int32_t>>& parameters = {}) {
    VELOX_CHECK_EQ(columnNames.size(), types.size());
    VELOX_CHECK_EQ(types.size(), transformTypes.size());
    VELOX_CHECK(parameters.empty() || parameters.size() == columnNames.size());
    std::vector<IcebergPartitionSpec::Field> fields;
    fields.reserve(columnNames.size());
    std::vector<column_index_t> partitionChannels(columnNames.size());
    std::iota(partitionChannels.begin(), partitionChannels.end(), 0);
    for (auto i = 0; i < columnNames.size(); ++i) {
      auto parameter = parameters.size() > i ? parameters.at(i) : std::nullopt;
      fields.emplace_back(
          IcebergPartitionSpec::Field{
              columnNames[i], types[i], transformTypes[i], parameter});
    }

    return std::make_unique<IcebergPartitionIdGenerator>(
        ROW(columnNames, types),
        partitionChannels,
        std::make_shared<const IcebergPartitionSpec>(0, fields),
        /*maxPartitions=*/128,
        connectorQueryCtx_.get());
  }

  // Runs the generator on the input rowVector and verifies that the generated
  // partition paths match the expected paths. Supports two verification modes:
  // 1. If expectedPaths.size() == rowVector.size(): Verifies each row's
  //    complete partition path matches the corresponding expected path.
  // 2. If rowVector.size() == 1: Splits the generated partition path by '/'
  //    and verifies each component matches the corresponding expected path.
  //
  // @param generator The IcebergPartitionIdGenerator to test.
  // @param rowVector Input data to generate partition IDs from.
  // @param expectedPaths Expected partition path strings or path components.
  void verifyPartitionPaths(
      const std::unique_ptr<IcebergPartitionIdGenerator>& generator,
      const RowVectorPtr& rowVector,
      const std::vector<std::string>& expectedPaths) {
    raw_vector<uint64_t> partitionIds(rowVector->size());
    generator->run(rowVector, partitionIds);

    if (expectedPaths.size() == rowVector->size()) {
      for (auto i = 0; i < rowVector->size(); ++i) {
        std::string partitionName = generator->partitionName(partitionIds[i]);
        ASSERT_EQ(partitionName, expectedPaths[i]);
      }
    } else {
      ASSERT_EQ(rowVector->size(), 1);
      std::string partitionName = generator->partitionName(partitionIds[0]);
      std::vector<std::string> actualPaths;
      folly::split('/', partitionName, actualPaths);
      ASSERT_EQ(actualPaths.size(), expectedPaths.size());
      for (auto i = 0; i < expectedPaths.size(); ++i) {
        ASSERT_EQ(actualPaths[i], expectedPaths[i]);
      }
    }
  }
};

TEST_F(IcebergPartitionIdGeneratorTest, identityTransforms) {
  std::vector<std::string> columnNames = {
      "c_int",
      "c_bigint",
      "c_varchar",
      "c_varbinary",
      "c_decimal",
      "c_bool",
      "c_date"};

  std::vector<TypePtr> types = {
      INTEGER(),
      BIGINT(),
      VARCHAR(),
      VARBINARY(),
      DECIMAL(18, 4),
      BOOLEAN(),
      DATE(),
  };
  std::vector<TransformType> transformTypes(
      columnNames.size(), TransformType::kIdentity);
  const auto& generator = createGenerator(columnNames, types, transformTypes);
  verifyPartitionPaths(
      generator,
      makeRowVector(
          columnNames,
          {
              makeConstant<int32_t>(42, 1),
              makeConstant<int64_t>(9'876'543'210, 1),
              makeConstant<StringView>("test string partition column name", 1),
              makeConstant<StringView>("\x48\x65\x6c\x6c\x6f", 1, VARBINARY()),
              makeConstant<int64_t>(12'345'678'901'234, 1, DECIMAL(18, 4)),
              makeConstant<bool>(true, 1),
              makeConstant<int32_t>(18'262, 1, DATE()),
          }),
      {
          "c_int=42",
          "c_bigint=9876543210",
          "c_varchar=test+string+partition+column+name",
          "c_varbinary=SGVsbG8%3D",
          "c_decimal=1234567890.1234",
          "c_bool=true",
          "c_date=2020-01-01",
      });
}

TEST_F(IcebergPartitionIdGeneratorTest, timestampIdentitySpecialValues) {
  std::vector<Timestamp> timestamps = {
      Timestamp(253402300800, 100000000), // +10000-01-01T00:00:00.1.
      Timestamp(-62170000000, 0), // -0001-11-29T19:33:20.
      Timestamp(-62135577748, 999000000), // 0001-01-01T05:17:32.999.
      Timestamp(0, 0), // 1970-01-01T00:00.
      Timestamp(1609459200, 999000000), // 2021-01-01T00:00.
      Timestamp(1640995200, 500000000), // 2022-01-01T00:00:00.5.
      Timestamp(1672531200, 123000000), // 2023-01-01T00:00:00.123.
      Timestamp(-1, 999000000), // 1969-12-31T23:59:59.999.
      Timestamp(1, 1000000), // 1970-01-01T00:00:01.001.
      Timestamp(-62167219199, 0), // 0000-01-01T00:00:01.
      Timestamp(-377716279140, 321000000), // -10000-01-01T01:01:00.321.
      Timestamp(253402304660, 321000000), // +10000-01-01T01:01:00.321.
      Timestamp(951782400, 0), // 2000-02-29T00:00:00 (leap year).
      Timestamp(4107456000, 0), // 2100-02-28T00:00:00.
      Timestamp(-86400, 0), // 1969-12-31T00:00:00.
  };

  std::vector<std::string> expectedPartitionNames = {
      "c_timestamp=%2B10000-01-01T00%3A00%3A00.1",
      "c_timestamp=-0001-11-29T19%3A33%3A20",
      "c_timestamp=0001-01-01T05%3A17%3A32.999",
      "c_timestamp=1970-01-01T00%3A00%3A00",
      "c_timestamp=2021-01-01T00%3A00%3A00.999",
      "c_timestamp=2022-01-01T00%3A00%3A00.5",
      "c_timestamp=2023-01-01T00%3A00%3A00.123",
      "c_timestamp=1969-12-31T23%3A59%3A59.999",
      "c_timestamp=1970-01-01T00%3A00%3A01.001",
      "c_timestamp=0000-01-01T00%3A00%3A01",
      "c_timestamp=-10000-08-24T19%3A21%3A00.321",
      "c_timestamp=%2B10000-01-01T01%3A04%3A20.321",
      "c_timestamp=2000-02-29T00%3A00%3A00",
      "c_timestamp=2100-02-28T00%3A00%3A00",
      "c_timestamp=1969-12-31T00%3A00%3A00",
  };

  std::vector<std::string> columnNames = {"c_timestamp"};
  const auto& generator =
      createGenerator(columnNames, {TIMESTAMP()}, {TransformType::kIdentity});
  verifyPartitionPaths(
      generator,
      makeRowVector(columnNames, {makeFlatVector<Timestamp>(timestamps)}),
      expectedPartitionNames);
}

TEST_F(IcebergPartitionIdGeneratorTest, nullValues) {
  std::vector<std::string> columnNames = {"c_int", "c_varchar", "c_decimal"};

  const auto& generator = createGenerator(
      columnNames,
      {INTEGER(), VARCHAR(), DECIMAL(18, 3)},
      {
          TransformType::kBucket,
          TransformType::kTruncate,
          TransformType::kIdentity,
      },
      {4, 100, std::nullopt});

  verifyPartitionPaths(
      generator,
      makeRowVector(
          columnNames,
          {
              makeConstant<int32_t>(std::nullopt, 1),
              makeConstant<StringView>(std::nullopt, 1),
              makeConstant<int64_t>(std::nullopt, 1, DECIMAL(18, 3)),
          }),
      {"c_int_bucket=null/c_varchar_trunc=null/c_decimal=null"});
}

TEST_F(IcebergPartitionIdGeneratorTest, specialChars) {
  std::vector<std::pair<std::string, std::string>> testCases = {
      {"space test", "space+test"},
      {"slash/test", "slash%2Ftest"},
      {"question?test", "question%3Ftest"},
      {"percent%test", "percent%25test"},
      {"hash#test", "hash%23test"},
      {"ampersand&test", "ampersand%26test"},
      {"equals=test", "equals%3Dtest"},
      {"plus+test", "plus%2Btest"},
      {"comma,test", "comma%2Ctest"},
      {"semicolon;test", "semicolon%3Btest"},
      {"at@test", "at%40test"},
      {"dollar$test", "dollar%24test"},
      {"backslash\\test", "backslash%5Ctest"},
      {"quote\"test", "quote%22test"},
      {"apostrophe'test", "apostrophe%27test"},
      {"less<than", "less%3Cthan"},
      {"greater>than", "greater%3Ethan"},
      {"colon:test", "colon%3Atest"},
      {"pipe|test", "pipe%7Ctest"},
      {"bracket[test", "bracket%5Btest"},
      {"bracket]test", "bracket%5Dtest"},
      {"brace{test", "brace%7Btest"},
      {"brace}test", "brace%7Dtest"},
      {"caret^test", "caret%5Etest"},
      {"tilde~test", "tilde%7Etest"},
      {"backtick`test", "backtick%60test"},
      {"unicode\u00A9test", "unicode%C2%A9test"},
      {"email@example.com", "email%40example.com"},
      {"user:password@host:port/path", "user%3Apassword%40host%3Aport%2Fpath"},
      {"https://github.ibm.com/IBM/velox",
       "https%3A%2F%2Fgithub.ibm.com%2FIBM%2Fvelox"},
      {"a+b=c&d=e+f", "a%2Bb%3Dc%26d%3De%2Bf"},
      {"special!@#$%^&*()_+", "special%21%40%23%24%25%5E%26*%28%29_%2B"},
  };

  std::vector<std::string> columnNames = {"ColumnWithSpecialChars"};
  const auto& generator =
      createGenerator(columnNames, {VARCHAR()}, {TransformType::kIdentity});

  for (const auto& [input, expectedEncoded] : testCases) {
    std::vector<std::string> expectedPartitionNames = {
        fmt::format("{}={}", columnNames[0], expectedEncoded)};
    verifyPartitionPaths(
        generator,
        makeRowVector(
            columnNames, {makeConstant<StringView>(StringView(input), 1)}),
        expectedPartitionNames);
  }
}

TEST_F(IcebergPartitionIdGeneratorTest, multipleRows) {
  std::vector<std::string> columnNames = {"c_int", "c_varchar"};
  std::vector<TransformType> transformTypes(
      columnNames.size(), TransformType::kIdentity);
  const auto& generator =
      createGenerator(columnNames, {INTEGER(), VARCHAR()}, transformTypes);

  verifyPartitionPaths(
      generator,
      makeRowVector(
          columnNames,
          {
              makeFlatVector<int32_t>({10, 20, 30}),
              makeFlatVector<StringView>({"value1", "value2", "value3"}),
          }),
      {
          "c_int=10/c_varchar=value1",
          "c_int=20/c_varchar=value2",
          "c_int=30/c_varchar=value3",
      });
}

} // namespace

} // namespace facebook::velox::connector::hive::iceberg
