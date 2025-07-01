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
#include "velox/connectors/hive/iceberg/Transforms.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

using namespace facebook::velox;

namespace facebook::velox::connector::hive::iceberg::test {

class IcebergPartitionIdGeneratorTest : public IcebergTestBase {
 protected:
  std::vector<std::shared_ptr<Transform>> createColumnTransforms(
      const std::vector<std::string>& columnNames,
      const std::vector<TypePtr>& types,
      const std::vector<TransformType>& transformTypes,
      const std::vector<std::optional<int32_t>>& parameters = {}) {
    std::vector<IcebergPartitionSpec::Field> fields;
    fields.reserve(columnNames.size());

    for (size_t i = 0; i < columnNames.size(); ++i) {
      std::optional<int32_t> parameter =
          parameters.size() > i ? parameters[i] : std::nullopt;

      fields.emplace_back(
          columnNames[i], types[i], transformTypes[i], parameter);
    }

    return parsePartitionTransformSpecs(fields, pool_.get());
  }

  std::unique_ptr<IcebergPartitionIdGenerator> createGenerator(
      const std::vector<std::shared_ptr<Transform>>& transforms,
      bool partitionPathAsLowerCase = false) {
    std::vector<column_index_t> partitionChannels;
    for (size_t i = 0; i < transforms.size(); ++i) {
      partitionChannels.push_back(i);
    }

    return std::make_unique<IcebergPartitionIdGenerator>(
        partitionChannels,
        128,
        pool_.get(),
        transforms,
        partitionPathAsLowerCase);
  }

  void verifyPartitionComponents(
      const std::string& partitionName,
      const std::vector<std::string>& expectedComponents) {
    std::vector<std::string> actualComponents;
    folly::split('/', partitionName, actualComponents);
    ASSERT_EQ(actualComponents.size(), expectedComponents.size());
    for (size_t i = 0; i < expectedComponents.size(); ++i) {
      ASSERT_EQ(actualComponents[i], expectedComponents[i]);
    }
  }
};

TEST_F(IcebergPartitionIdGeneratorTest, partitionNameWithIdentityTransforms) {
  std::vector<std::string> columnNames = {
      "c_int", "c_bigint", "c_varchar", "c_decimal", "c_bool", "c_date"};

  std::vector<VectorPtr> columns = {
      makeConstant<int32_t>(42, 1),
      makeConstant<int64_t>(9'876'543'210, 1),
      makeConstant<StringView>("test string", 1),
      makeConstant<int64_t>(12'345'678'901'234, 1, DECIMAL(18, 4)),
      makeConstant<bool>(true, 1),
      makeConstant<int32_t>(18'262, 1, DATE())};

  std::vector<TypePtr> types = {
      INTEGER(), BIGINT(), VARCHAR(), DECIMAL(18, 4), BOOLEAN(), DATE()};
  auto rowVector = makeRowVector(columnNames, columns);
  std::vector<TransformType> transformTypes(
      columnNames.size(), TransformType::kIdentity);
  auto transforms = createColumnTransforms(columnNames, types, transformTypes);
  auto generator = createGenerator(transforms);
  raw_vector<uint64_t> partitionIds(1);
  generator->run(rowVector, partitionIds);

  std::string partitionName = generator->partitionName(partitionIds[0]);
  std::vector<std::string> expectedComponents = {
      "c_int=42",
      "c_bigint=9876543210",
      "c_varchar=test+string",
      "c_decimal=1234567890.1234",
      "c_bool=true",
      "c_date=2020-01-01"};
  verifyPartitionComponents(partitionName, expectedComponents);
}

TEST_F(
    IcebergPartitionIdGeneratorTest,
    partitionNameWithTimestampIdentitySpecialValues) {
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
      Timestamp(4107456000, 0), // 2100-02-28T00:00:00 (not leap year).
      Timestamp(86400, 0), // 1970-01-02T00:00:00.
      Timestamp(-86400, 0), // 1969-12-31T00:00:00.
      Timestamp(1672531200, 456000000), // 2023-01-01T00:00:00.456.
      Timestamp(1672531200, 789000000), // 2023-01-01T00:00:00.789.
  };

  std::vector<std::string> expectedPartitionNames = {
      "c_timestamp=%2B10000-01-01T00%3A00%3A00.1",
      "c_timestamp=-0001-11-29T19%3A33%3A20",
      "c_timestamp=0001-01-01T05%3A17%3A32.999",
      "c_timestamp=1970-01-01T00%3A00",
      "c_timestamp=2021-01-01T00%3A00%3A00.999",
      "c_timestamp=2022-01-01T00%3A00%3A00.5",
      "c_timestamp=2023-01-01T00%3A00%3A00.123",
      "c_timestamp=1969-12-31T23%3A59%3A59.999",
      "c_timestamp=1970-01-01T00%3A00%3A01.001",
      "c_timestamp=0000-01-01T00%3A00%3A01",
      "c_timestamp=-10000-08-24T19%3A21%3A00.321",
      "c_timestamp=%2B10000-01-01T01%3A04%3A20.321",
      "c_timestamp=2000-02-29T00%3A00",
      "c_timestamp=2100-02-28T00%3A00",
      "c_timestamp=1970-01-02T00%3A00",
      "c_timestamp=1969-12-31T00%3A00",
      "c_timestamp=2023-01-01T00%3A00%3A00.456",
      "c_timestamp=2023-01-01T00%3A00%3A00.789",
  };

  auto timestampVector = makeFlatVector<Timestamp>(timestamps);
  std::vector<std::string> columnNames = {"c_timestamp"};
  std::vector<VectorPtr> columns = {timestampVector};
  std::vector<TypePtr> types = {TIMESTAMP()};
  auto rowVector = makeRowVector(columnNames, columns);

  std::vector<TransformType> transformTypes = {TransformType::kIdentity};
  auto transforms = createColumnTransforms(columnNames, types, transformTypes);
  auto generator = createGenerator(transforms);
  raw_vector<uint64_t> partitionIds(timestamps.size());
  generator->run(rowVector, partitionIds);

  for (size_t i = 0; i < timestamps.size(); ++i) {
    std::string partitionName = generator->partitionName(partitionIds[i]);
    ASSERT_EQ(partitionName, expectedPartitionNames[i]);
  }
}

TEST_F(IcebergPartitionIdGeneratorTest, partitionNameWithMixedTransforms) {
  std::vector<std::string> columnNames = {
      "c_int",
      "c_bigint",
      "c_varchar",
      "c_year",
      "c_month",
      "c_day",
      "c_hour",
      "c_bool"};

  std::vector<VectorPtr> columns = {
      makeConstant<int32_t>(42, 1),
      makeConstant<int64_t>(9'876'543'210, 1),
      makeConstant<StringView>("test string", 1),
      makeConstant<Timestamp>(Timestamp(1'577'836'800, 0), 1),
      makeConstant<Timestamp>(Timestamp(1'578'836'800, 0), 1),
      makeConstant<Timestamp>(Timestamp(1'579'836'800, 0), 1),
      makeConstant<Timestamp>(Timestamp(1'57'936'800, 0), 1),
      makeConstant<bool>(true, 1)};

  std::vector<TypePtr> types = {
      INTEGER(),
      BIGINT(),
      VARCHAR(),
      TIMESTAMP(),
      TIMESTAMP(),
      TIMESTAMP(),
      TIMESTAMP(),
      BOOLEAN()};

  auto rowVector = makeRowVector(columnNames, columns);

  std::vector<TransformType> transformTypes = {
      TransformType::kBucket,
      TransformType::kTruncate,
      TransformType::kTruncate,
      TransformType::kYear,
      TransformType::kMonth,
      TransformType::kDay,
      TransformType::kHour,
      TransformType::kIdentity};

  std::vector<std::optional<int32_t>> parameters = {4, 1'000, 5, std::nullopt};
  auto transforms =
      createColumnTransforms(columnNames, types, transformTypes, parameters);

  auto generator = createGenerator(transforms);
  raw_vector<uint64_t> partitionIds(1);
  generator->run(rowVector, partitionIds);

  std::string partitionName = generator->partitionName(partitionIds[0]);
  std::vector<std::string> expectedComponents = {
      "c_int_bucket=2",
      "c_bigint_trunc=9876543000",
      "c_varchar_trunc=test+",
      "c_year_year=2020",
      "c_month_month=2020-01",
      "c_day_day=2020-01-24",
      "c_hour_hour=1975-01-02-23",
      "c_bool=true"};
  verifyPartitionComponents(partitionName, expectedComponents);
}

TEST_F(IcebergPartitionIdGeneratorTest, partitionNameWithNullValues) {
  std::vector<std::string> columnNames = {"c_int", "c_varchar", "c_decimal"};
  std::vector<VectorPtr> columns = {
      makeConstant<int32_t>(std::nullopt, 1),
      makeConstant<StringView>(std::nullopt, 1),
      makeConstant<int64_t>(std::nullopt, 1, DECIMAL(18, 4))};
  std::vector<TypePtr> types = {INTEGER(), VARCHAR(), DECIMAL(18, 3)};
  auto rowVector = makeRowVector(columnNames, columns);

  std::vector<TransformType> transformTypes = {
      TransformType::kBucket,
      TransformType::kTruncate,
      TransformType::kIdentity};
  std::vector<std::optional<int32_t>> parameters = {4, 1'000, std::nullopt};
  auto transforms =
      createColumnTransforms(columnNames, types, transformTypes, parameters);
  auto generator = createGenerator(transforms);
  raw_vector<uint64_t> partitionIds(1);
  generator->run(rowVector, partitionIds);

  std::string partitionName = generator->partitionName(partitionIds[0]);
  std::vector<std::string> expectedComponents = {
      "c_int_bucket=null", "c_varchar_trunc=null", "c_decimal=null"};
  verifyPartitionComponents(partitionName, expectedComponents);
}

TEST_F(IcebergPartitionIdGeneratorTest, partitionNameWithLowerCase) {
  auto varcharVector = makeConstant<StringView>("MiXeD_CaSe", 1);
  std::vector<std::string> columnNames = {"MiXeD_CoLuMn"};
  std::vector<VectorPtr> columns = {varcharVector};
  std::vector<TypePtr> types = {VARCHAR()};
  auto rowVector = makeRowVector(columnNames, columns);
  std::vector<TransformType> transformTypes = {TransformType::kIdentity};
  auto transforms = createColumnTransforms(columnNames, types, transformTypes);
  auto generator = createGenerator(transforms, true);
  raw_vector<uint64_t> partitionIds(1);
  generator->run(rowVector, partitionIds);
  std::string partitionName = generator->partitionName(partitionIds[0]);
  std::vector<std::string> expectedPartitionName = {"mixed_column=MiXeD_CaSe"};
  verifyPartitionComponents(partitionName, expectedPartitionName);

  generator = createGenerator(transforms);
  generator->run(rowVector, partitionIds);
  partitionName = generator->partitionName(partitionIds[0]);
  expectedPartitionName = {"MiXeD_CoLuMn=MiXeD_CaSe"};
  verifyPartitionComponents(partitionName, expectedPartitionName);
}

TEST_F(IcebergPartitionIdGeneratorTest, urlEncodingForSpecialChars) {
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

  std::vector<TransformType> transformTypes = {TransformType::kIdentity};
  std::vector<TypePtr> types = {VARCHAR()};
  std::vector<std::string> columnNames = {"ColumnWithSpecialChars"};
  auto transforms = createColumnTransforms(columnNames, types, transformTypes);
  raw_vector<uint64_t> partitionIds(1);
  auto generator = createGenerator(transforms);

  for (const auto& [input, expectedEncoded] : testCases) {
    auto varcharVector = makeConstant<StringView>(StringView(input), 1);
    auto rowVector = makeRowVector(columnNames, {varcharVector});
    generator->run(rowVector, partitionIds);
    std::string partitionName = generator->partitionName(partitionIds[0]);
    std::string expectedPartitionName =
        fmt::format("{}={}", columnNames[0], expectedEncoded);
    ASSERT_EQ(partitionName, expectedPartitionName);
  }
}

TEST_F(IcebergPartitionIdGeneratorTest, multipleRows) {
  std::vector<std::string> columnNames = {"c_int", "c_varchar"};
  auto rowVector = makeRowVector(
      columnNames,
      {makeFlatVector<int32_t>({10, 20, 30}),
       makeFlatVector<StringView>({"value1", "value2", "value3"})});

  std::vector<TypePtr> types = {INTEGER(), VARCHAR()};
  std::vector<TransformType> transformTypes(
      columnNames.size(), TransformType::kIdentity);
  auto transforms = createColumnTransforms(columnNames, types, transformTypes);
  auto generator = createGenerator(transforms);
  raw_vector<uint64_t> partitionIds(3);
  generator->run(rowVector, partitionIds);

  std::vector<std::string> expectedNames = {
      "c_int=10/c_varchar=value1",
      "c_int=20/c_varchar=value2",
      "c_int=30/c_varchar=value3"};

  for (size_t i = 0; i < 3; ++i) {
    std::string partitionName = generator->partitionName(partitionIds[i]);
    ASSERT_EQ(partitionName, expectedNames[i]);
  }
}

} // namespace facebook::velox::connector::hive::iceberg::test
