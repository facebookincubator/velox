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

#include "velox/connectors/hive/iceberg/IcebergConnector.h"

#include <numeric>

#include "velox/common/encode/Base64.h"
#include "velox/connectors/hive/iceberg/IcebergPartitionName.h"
#include "velox/connectors/hive/iceberg/TransformEvaluator.h"
#include "velox/connectors/hive/iceberg/TransformExprBuilder.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

namespace facebook::velox::connector::hive::iceberg {

using namespace facebook::velox;

namespace {

class PartitionNameTest : public test::IcebergTestBase {
 protected:
  // Generates partition IDs for the input rows and verifies that the resulting
  // partition paths match the expected paths. Each row is processed
  // independently, and its generated partition path is compared against the
  // corresponding entry in expectedPaths.
  //
  // @param input Input data to generate partition IDs from. Must have the
  // same size as expectedPaths.
  // @param partitionSpec The IcebergPartitionSpec defining the partition
  // transforms. The partition channels are determined by matching field names
  // from the spec to column names in the input's type.
  // @param expectedPaths Expected partition path strings, one per row. Each
  // path should be the complete partition directory name (e.g., "col1=val1").
  void verifyPartitionPaths(
      const RowVectorPtr& input,
      const std::shared_ptr<IcebergPartitionSpec>& partitionSpec,
      const std::vector<std::string>& expectedPaths) const {
    ASSERT_EQ(expectedPaths.size(), input->size());
    std::vector<column_index_t> partitionChannels(partitionSpec->fields.size());
    auto rowType = input->rowType();
    for (auto i = 0; i < partitionSpec->fields.size(); ++i) {
      partitionChannels[i] =
          rowType->getChildIdx(partitionSpec->fields[i].name);
    }

    // Step 1: Build transform expressions and create evaluator.
    auto transformExpressions = TransformExprBuilder::toExpressions(
        partitionSpec,
        partitionChannels,
        rowType,
        std::string(kDefaultIcebergFunctionPrefix));
    auto transformEvaluator = std::make_unique<TransformEvaluator>(
        transformExpressions, connectorQueryCtx_.get());

    // Step 2: Apply transforms to input partition columns.
    auto transformedColumns = transformEvaluator->evaluate(input);

    std::vector<TypePtr> partitionKeyTypes;
    std::vector<std::string> partitionKeyNames;
    for (const auto& field : partitionSpec->fields) {
      partitionKeyTypes.emplace_back(field.resultType());
      std::string key = field.transformType == TransformType::kIdentity
          ? field.name
          : fmt::format(
                "{}_{}",
                field.name,
                TransformTypeName::toName(field.transformType));
      partitionKeyNames.emplace_back(std::move(key));
    }

    auto partitionRowType =
        ROW(std::move(partitionKeyNames), std::move(partitionKeyTypes));
    // Step 3: Create RowVector based on transformed columns.
    auto transformedRowVector = std::make_shared<RowVector>(
        connectorQueryCtx_->memoryPool(),
        partitionRowType,
        nullptr,
        input->size(),
        std::move(transformedColumns));

    // Step 4: Generate partition IDs from transformed data.
    // The transformed row vector has columns in the same order as partition
    // spec fields, so channels are sequential: 0, 1, 2, ...
    std::vector<column_index_t> transformedChannels(
        partitionSpec->fields.size());
    std::iota(transformedChannels.begin(), transformedChannels.end(), 0);

    auto idGenerator = std::make_unique<PartitionIdGenerator>(
        partitionRowType,
        transformedChannels,
        /*maxPartitions=*/128,
        connectorQueryCtx_->memoryPool());

    auto nameGenerator = std::make_unique<IcebergPartitionName>(partitionSpec);

    raw_vector<uint64_t> partitionIds(input->size());
    idGenerator->run(transformedRowVector, partitionIds);

    for (auto i = 0; i < input->size(); ++i) {
      std::string partitionName = nameGenerator->partitionName(
          partitionIds[i], idGenerator->partitionValues(), false);
      ASSERT_EQ(partitionName, expectedPaths[i]);
    }
  }
};

TEST_F(PartitionNameTest, identity) {
  std::vector<std::tuple<TypePtr, VectorPtr, std::string>> input = {
      {INTEGER(), makeConstant<int32_t>(42, 1), "42"},
      {BIGINT(), makeConstant<int64_t>(9'876'543'210, 1), "9876543210"},
      {VARCHAR(),
       makeConstant<std::string>("test string partition column name", 1),
       "test+string+partition+column+name"},
      {VARBINARY(),
       makeConstant<std::string>("\x48\x65\x6c\x6c\x6f", 1, VARBINARY()),
       "SGVsbG8%3D"},
      {DECIMAL(18, 4),
       makeConstant<int64_t>(12'345'678'901'234, 1, DECIMAL(18, 4)),
       "1234567890.1234"},
      {BOOLEAN(), makeConstant<bool>(true, 1), "true"},
      {DATE(), makeConstant<int32_t>(18'262, 1, DATE()), "2020-01-01"},
  };

  for (const auto& [type, value, expectedValue] : input) {
    const auto& partitionSpec = createPartitionSpec(
        ROW({"c0"}, {type}), {{0, TransformType::kIdentity, std::nullopt}});

    verifyPartitionPaths(
        makeRowVector({value}),
        partitionSpec,
        {fmt::format("c0={}", expectedValue)});
  }
}

TEST_F(PartitionNameTest, timestamp) {
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
      "c0=%2B10000-01-01T00%3A00%3A00.1",
      "c0=-0001-11-29T19%3A33%3A20",
      "c0=0001-01-01T05%3A17%3A32.999",
      "c0=1970-01-01T00%3A00%3A00",
      "c0=2021-01-01T00%3A00%3A00.999",
      "c0=2022-01-01T00%3A00%3A00.5",
      "c0=2023-01-01T00%3A00%3A00.123",
      "c0=1969-12-31T23%3A59%3A59.999",
      "c0=1970-01-01T00%3A00%3A01.001",
      "c0=0000-01-01T00%3A00%3A01",
      "c0=-10000-08-24T19%3A21%3A00.321",
      "c0=%2B10000-01-01T01%3A04%3A20.321",
      "c0=2000-02-29T00%3A00%3A00",
      "c0=2100-02-28T00%3A00%3A00",
      "c0=1969-12-31T00%3A00%3A00",
  };

  const auto& partitionSpec = createPartitionSpec(
      ROW({"c0"}, {TIMESTAMP()}),
      {{0, TransformType::kIdentity, std::nullopt}});

  verifyPartitionPaths(
      makeRowVector({makeFlatVector<Timestamp>(timestamps)}),
      partitionSpec,
      expectedPartitionNames);
}

TEST_F(PartitionNameTest, null) {
  std::vector<
      std::tuple<TypePtr, TransformType, std::optional<int32_t>, VectorPtr>>
      input = {
          {INTEGER(),
           TransformType::kBucket,
           32,
           makeConstant<int32_t>(std::nullopt, 1)},
          {VARCHAR(),
           TransformType::kTruncate,
           100,
           makeConstant<std::string>(std::nullopt, 1)},
          {DECIMAL(18, 3),
           TransformType::kIdentity,
           std::nullopt,
           makeConstant<int64_t>(std::nullopt, 1, DECIMAL(18, 3))},
          {TIMESTAMP(),
           TransformType::kYear,
           std::nullopt,
           makeConstant<Timestamp>(std::nullopt, 1)},
          {TIMESTAMP(),
           TransformType::kMonth,
           std::nullopt,
           makeConstant<Timestamp>(std::nullopt, 1)},
          {DATE(),
           TransformType::kDay,
           std::nullopt,
           makeConstant<int32_t>(std::nullopt, 1, DATE())},
          {TIMESTAMP(),
           TransformType::kHour,
           std::nullopt,
           makeConstant<Timestamp>(std::nullopt, 1)},
      };

  for (const auto& [type, transformType, parameter, value] : input) {
    auto rowType = ROW({"c0"}, {type});
    const auto& partitionSpec =
        createPartitionSpec(rowType, {{0, transformType, parameter}});
    if (transformType == TransformType::kIdentity) {
      verifyPartitionPaths(makeRowVector({value}), partitionSpec, {"c0=null"});
    } else {
      verifyPartitionPaths(
          makeRowVector({value}),
          partitionSpec,
          {fmt::format(
              "c0_{}=null", TransformTypeName::toName(transformType))});
    }
  }
}

// test both partition column name and partition key encoding.
TEST_F(PartitionNameTest, specialChars) {
  std::vector<std::pair<std::string, std::string>> inputs = {
      {"abc123", "abc123"},
      {"ABC123", "ABC123"},
      {"a.b-c_d*e", "a.b-c_d*e"},
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
      {"exclamation!test", "exclamation%21test"},
      {"dollar$test", "dollar%24test"},
      {"backslash\\test", "backslash%5Ctest"},
      {"quote\"test", "quote%22test"},
      {"apostrophe'test", "apostrophe%27test"},
      {"paren(test", "paren%28test"},
      {"paren)test", "paren%29test"},
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
      {"newline\ntest", "newline%0Atest"},
      {"carriage\rreturn", "carriage%0Dreturn"},
      {"tab\ttest", "tab%09test"},
      {"unicode\u00A9test", "unicode%C2%A9test"},
      {"email@example.com", "email%40example.com"},
      {"user:password@host:port/path", "user%3Apassword%40host%3Aport%2Fpath"},
      {"https://github.com/facebookincubator/velox",
       "https%3A%2F%2Fgithub.com%2Ffacebookincubator%2Fvelox"},
      {"a+b=c&d=e+f", "a%2Bb%3Dc%26d%3De%2Bf"},
      {"a#b=c/d e", "a%23b%3Dc%2Fd+e"},
      {"special!@#$%^&*()_+", "special%21%40%23%24%25%5E%26*%28%29_%2B"},
  };

  for (const auto& [input, encodedValue] : inputs) {
    const auto& partitionSpec = createPartitionSpec(
        ROW({input}, {VARCHAR()}),
        {{0, TransformType::kIdentity, std::nullopt}});

    verifyPartitionPaths(
        makeRowVector(
            {input}, {makeConstant<StringView>(StringView(input), 1)}),
        partitionSpec,
        {fmt::format("{}={}", encodedValue, encodedValue)});
  }
}

TEST_F(PartitionNameTest, multipleRows) {
  const auto& partitionSpec = createPartitionSpec(
      ROW({"c0", "c1"}, {INTEGER(), VARCHAR()}),
      {
          {0, TransformType::kBucket, 8},
          {1, TransformType::kIdentity, std::nullopt},
      });

  verifyPartitionPaths(
      makeRowVector({
          makeFlatVector<int32_t>({10, 20, 30, -100}),
          makeFlatVector<std::string>({"value1", "VALue2", "VALUE3", ""}),
      }),
      partitionSpec,
      {
          "c0_bucket=4/c1=value1",
          "c0_bucket=3/c1=VALue2",
          "c0_bucket=3/c1=VALUE3",
          "c0_bucket=6/c1=",
      });
}

TEST_F(PartitionNameTest, year) {
  const auto& partitionSpec = createPartitionSpec(
      ROW({"c0"}, {TIMESTAMP()}), {{0, TransformType::kYear, std::nullopt}});

  std::vector<Timestamp> timestamps = {
      Timestamp(0, 0),
      Timestamp(1609459200, 0),
      Timestamp(1640995200, 0),
      Timestamp(-31536000, 0),
      Timestamp(253402300800, 0),
  };

  verifyPartitionPaths(
      makeRowVector({makeFlatVector<Timestamp>(timestamps)}),
      partitionSpec,
      {
          "c0_year=1970",
          "c0_year=2021",
          "c0_year=2022",
          "c0_year=1969",
          "c0_year=10000",
      });
}

TEST_F(PartitionNameTest, yearWithDate) {
  const auto& partitionSpec = createPartitionSpec(
      ROW({"c0"}, {DATE()}), {{0, TransformType::kYear, std::nullopt}});

  verifyPartitionPaths(
      makeRowVector({makeFlatVector<int32_t>({0, 365, 18262, -365}, DATE())}),
      partitionSpec,
      {
          "c0_year=1970",
          "c0_year=1971",
          "c0_year=2020",
          "c0_year=1969",
      });
}

TEST_F(PartitionNameTest, month) {
  const auto& partitionSpec = createPartitionSpec(
      ROW({"c0"}, {TIMESTAMP()}), {{0, TransformType::kMonth, std::nullopt}});

  verifyPartitionPaths(
      makeRowVector({makeFlatVector<Timestamp>({
          Timestamp(0, 0),
          Timestamp(2678400, 0),
          Timestamp(1609459200, 0),
          Timestamp(1640995200, 0),
          Timestamp(-2678400, 0),
      })}),
      partitionSpec,
      {
          "c0_month=1970-01",
          "c0_month=1970-02",
          "c0_month=2021-01",
          "c0_month=2022-01",
          "c0_month=1969-12",
      });
}

TEST_F(PartitionNameTest, monthWithDate) {
  const auto& partitionSpec = createPartitionSpec(
      ROW({"c0"}, {DATE()}), {{0, TransformType::kMonth, std::nullopt}});

  verifyPartitionPaths(
      makeRowVector({makeFlatVector<int32_t>({0, 31, 365, -31}, DATE())}),
      partitionSpec,
      {
          "c0_month=1970-01",
          "c0_month=1970-02",
          "c0_month=1971-01",
          "c0_month=1969-12",
      });
}

TEST_F(PartitionNameTest, day) {
  const auto& partitionSpec = createPartitionSpec(
      ROW({"c0"}, {TIMESTAMP()}), {{0, TransformType::kDay, std::nullopt}});

  std::vector<Timestamp> timestamps = {
      Timestamp(0, 0),
      Timestamp(86400, 0),
      Timestamp(1577836800, 0),
      Timestamp(-86400, 0),
  };

  verifyPartitionPaths(
      makeRowVector({makeFlatVector<Timestamp>(timestamps)}),
      partitionSpec,
      {
          "c0_day=1970-01-01",
          "c0_day=1970-01-02",
          "c0_day=2020-01-01",
          "c0_day=1969-12-31",
      });
}

TEST_F(PartitionNameTest, dayWithDate) {
  const auto& partitionSpec = createPartitionSpec(
      ROW({"c0"}, {DATE()}), {{0, TransformType::kDay, std::nullopt}});

  verifyPartitionPaths(
      makeRowVector({makeFlatVector<int32_t>({0, 1, 18262, -1}, DATE())}),
      partitionSpec,
      {
          "c0_day=1970-01-01",
          "c0_day=1970-01-02",
          "c0_day=2020-01-01",
          "c0_day=1969-12-31",
      });
}

TEST_F(PartitionNameTest, hour) {
  const auto& partitionSpec = createPartitionSpec(
      ROW({"c0"}, {TIMESTAMP()}), {{0, TransformType::kHour, std::nullopt}});

  std::vector<Timestamp> timestamps = {
      Timestamp(0, 0),
      Timestamp(3600, 0),
      Timestamp(86400, 0),
      Timestamp(1577836800, 0),
      Timestamp(-3600, 0),
  };

  verifyPartitionPaths(
      makeRowVector({makeFlatVector<Timestamp>(timestamps)}),
      partitionSpec,
      {
          "c0_hour=1970-01-01-00",
          "c0_hour=1970-01-01-01",
          "c0_hour=1970-01-02-00",
          "c0_hour=2020-01-01-00",
          "c0_hour=1969-12-31-23",
      });
}

TEST_F(PartitionNameTest, multipleTransformsSameColumn) {
  const auto& partitionSpec = createPartitionSpec(
      ROW({"c0"}, {TIMESTAMP()}),
      {
          {0, TransformType::kIdentity, std::nullopt},
          {0, TransformType::kYear, std::nullopt},
          {0, TransformType::kBucket, 10},
      });

  verifyPartitionPaths(
      makeRowVector({makeFlatVector<Timestamp>({Timestamp(1609459200, 0)})}),
      partitionSpec,
      {"c0=2021-01-01T00%3A00%3A00/c0_year=2021/c0_bucket=0"});
}

} // namespace

} // namespace facebook::velox::connector::hive::iceberg
