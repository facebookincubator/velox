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
#include "velox/connectors/hive/iceberg/ColumnTransform.h"
#include "velox/connectors/hive/iceberg/Transforms.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

using namespace facebook::velox;

namespace facebook::velox::connector::hive::iceberg::test {

class IcebergPartitionIdGeneratorTest : public IcebergTestBase {
 protected:
  std::vector<ColumnTransform> createColumnTransforms(
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
      const std::vector<ColumnTransform>& transforms,
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

  RowVectorPtr createRowVector(
      const std::vector<std::string>& names,
      const std::vector<VectorPtr>& children) {
    std::vector<TypePtr> types;
    for (const auto& child : children) {
      types.push_back(child->type());
    }

    return makeRowVector(names, children);
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
  auto intVector = makeFlatVector<int32_t>(1, [](auto) { return 42; });
  auto bigintVector =
      makeFlatVector<int64_t>(1, [](auto) { return 9'876'543'210; });
  auto varcharVector =
      BaseVector::create<FlatVector<StringView>>(VARCHAR(), 1, opPool_.get());
  varcharVector->set(0, StringView("test string"));
  auto decimalVector =
      BaseVector::create<FlatVector<int64_t>>(DECIMAL(18, 4), 1, opPool_.get());
  decimalVector->set(0, 12'345'678'901'234);
  auto boolVector = makeFlatVector<bool>(1, [](auto) { return true; });
  auto dateVector =
      BaseVector::create<FlatVector<int32_t>>(DATE(), 1, opPool_.get());
  dateVector->set(0, 18'262);

  std::vector<std::string> columnNames = {
      "c_int", "c_bigint", "c_varchar", "c_decimal", "c_bool", "c_date"};

  std::vector<VectorPtr> columns = {
      intVector,
      bigintVector,
      varcharVector,
      decimalVector,
      boolVector,
      dateVector};

  std::vector<TypePtr> types = {
      INTEGER(), BIGINT(), VARCHAR(), DECIMAL(18, 4), BOOLEAN(), DATE()};
  auto rowVector = createRowVector(columnNames, columns);
  std::vector<TransformType> transformTypes(
      columnNames.size(), TransformType::kIdentity);
  auto transforms = createColumnTransforms(columnNames, types, transformTypes);
  auto generator = createGenerator(transforms);
  raw_vector<uint64_t> partitionIds(1);
  generator->run(rowVector, partitionIds);

  std::string partitionName = generator->partitionName(partitionIds[0], "null");
  std::vector<std::string> expectedComponents = {
      "c_int=42",
      "c_bigint=9876543210",
      "c_varchar=test+string",
      "c_decimal=1234567890.1234",
      "c_bool=true",
      "c_date=2020-01-01"};
  verifyPartitionComponents(partitionName, expectedComponents);
}

TEST_F(IcebergPartitionIdGeneratorTest, partitionNameWithMixedTransforms) {
  auto intVector = makeFlatVector<int32_t>(1, [](auto) { return 42; });
  auto bigintVector =
      makeFlatVector<int64_t>(1, [](auto) { return 9'876'543'210; });
  auto varcharVector = makeFlatVector<StringView>({"test string"});
  auto yearVector = makeFlatVector<Timestamp>(
      1, [](auto) { return Timestamp(1'577'836'800, 0); });
  auto monthVector = makeFlatVector<Timestamp>(
      1, [](auto) { return Timestamp(1'578'836'800, 0); });
  auto dayVector = makeFlatVector<Timestamp>(
      1, [](auto) { return Timestamp(1'579'836'800, 0); });
  auto hourVector = makeFlatVector<Timestamp>(
      1, [](auto) { return Timestamp(1'57'936'800, 0); });
  auto boolVector = makeFlatVector<bool>(1, [](auto) { return true; });

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
      intVector,
      bigintVector,
      varcharVector,
      yearVector,
      monthVector,
      dayVector,
      hourVector,
      boolVector};

  std::vector<TypePtr> types = {
      INTEGER(),
      BIGINT(),
      VARCHAR(),
      TIMESTAMP(),
      TIMESTAMP(),
      TIMESTAMP(),
      TIMESTAMP(),
      BOOLEAN()};

  auto rowVector = createRowVector(columnNames, columns);

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

  std::string partitionName = generator->partitionName(partitionIds[0], "null");
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
  auto intVector = makeNullableFlatVector<int32_t>({std::nullopt});
  auto varcharVector = makeNullableFlatVector<StringView>({std::nullopt});
  auto decimalVector =
      BaseVector::create<FlatVector<int64_t>>(DECIMAL(18, 4), 1, opPool_.get());
  decimalVector->setNull(0, true);

  std::vector<std::string> columnNames = {"c_int", "c_varchar", "c_decimal"};
  std::vector<VectorPtr> columns = {intVector, varcharVector, decimalVector};
  std::vector<TypePtr> types = {INTEGER(), VARCHAR(), DECIMAL(18, 3)};
  auto rowVector = createRowVector(columnNames, columns);

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

  std::string partitionName = generator->partitionName(partitionIds[0], "null");
  std::vector<std::string> expectedComponents = {
      "c_int_bucket=null", "c_varchar_trunc=null", "c_decimal=null"};
  verifyPartitionComponents(partitionName, expectedComponents);
}

TEST_F(IcebergPartitionIdGeneratorTest, partitionNameWithLowerCase) {
  auto varcharVector = makeFlatVector<StringView>({"MiXeD_CaSe"});
  std::vector<std::string> columnNames = {"MiXeD_CoLuMn"};
  std::vector<VectorPtr> columns = {varcharVector};
  std::vector<TypePtr> types = {VARCHAR()};
  auto rowVector = createRowVector(columnNames, columns);
  std::vector<TransformType> transformTypes = {TransformType::kIdentity};
  auto transforms = createColumnTransforms(columnNames, types, transformTypes);
  auto generator = createGenerator(transforms, true);
  raw_vector<uint64_t> partitionIds(1);
  generator->run(rowVector, partitionIds);
  std::string partitionName = generator->partitionName(partitionIds[0], "null");
  std::vector<std::string> expectedComponents = {"mixed_column=MiXeD_CaSe"};
  verifyPartitionComponents(partitionName, expectedComponents);

  generator = createGenerator(transforms);
  generator->run(rowVector, partitionIds);
  partitionName = generator->partitionName(partitionIds[0], "null");
  expectedComponents = {"MiXeD_CoLuMn=MiXeD_CaSe"};
  verifyPartitionComponents(partitionName, expectedComponents);
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

  for (const auto& [input, expectedEncoded] : testCases) {
    auto varcharVector = makeFlatVector<StringView>({StringView(input)});
    std::vector<std::string> columnNames = {"ColumnWithSpecialChars"};
    auto rowVector = createRowVector(columnNames, {varcharVector});

    std::vector<TransformType> transformTypes = {TransformType::kIdentity};
    std::vector<TypePtr> types = {VARCHAR()};
    auto transforms =
        createColumnTransforms(columnNames, types, transformTypes);

    auto generator = createGenerator(transforms);
    raw_vector<uint64_t> partitionIds(1);
    generator->run(rowVector, partitionIds);

    std::string partitionName =
        generator->partitionName(partitionIds[0], "null");
    std::string expectedComponent =
        fmt::format("{}={}", columnNames[0], expectedEncoded);
    ASSERT_EQ(partitionName, expectedComponent);
  }
}

TEST_F(IcebergPartitionIdGeneratorTest, multipleRows) {
  auto intVector = makeFlatVector<int32_t>({10, 20, 30});
  auto varcharVector =
      makeFlatVector<StringView>({"value1", "value2", "value3"});
  std::vector<std::string> columnNames = {"c_int", "c_varchar"};
  auto rowVector = createRowVector(columnNames, {intVector, varcharVector});

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
    std::string partitionName =
        generator->partitionName(partitionIds[i], "null");
    ASSERT_EQ(partitionName, expectedNames[i]);
  }
}

} // namespace facebook::velox::connector::hive::iceberg::test
