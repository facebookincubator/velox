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
#include "velox/connectors/hive/paimon/PaimonTableHandle.h"

#include <gtest/gtest.h>
#include <limits>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/Expressions.h"

namespace facebook::velox::connector::hive::paimon {
namespace {

class PaimonTableHandleTest : public testing::Test {
 protected:
  static constexpr auto kReadProfile =
      PaimonTableHandle::ReadProfile::kSameSchemaAppendBatch;

  void SetUp() override {
    Type::registerSerDe();
    common::Filter::registerSerDe();
    core::ITypedExpr::registerSerDe();
    HiveColumnHandle::registerSerDe();
    PaimonTableHandle::registerSerDe();
  }

  static PaimonSchemaPtr schema(std::vector<int32_t> primaryKeys = {8, 7}) {
    return std::make_shared<PaimonSchema>(
        int64_t{1} << 40,
        2,
        ROW({{"id", BIGINT()}, {"p", VARCHAR()}, {"value", BIGINT()}}),
        PaimonTypeMetadata{
            "ROW",
            false,
            std::nullopt,
            {},
            {{"BIGINT", false, 7},
             {"VARCHAR", true, 8, {100}},
             {"BIGINT", true, 9}}},
        std::vector<int32_t>{8},
        std::move(primaryKeys),
        std::unordered_map<std::string, std::string>{
            {"type", "table"},
            {"batch-scan-mode", "none"},
            {"deletion-vectors.enabled", "false"},
            {"data-evolution.enabled", "false"},
            {"data-file.thin-mode", "false"},
            {"row-tracking.enabled", "false"},
            {"file.format", "nimble"},
            {"bucket", "-1"}});
  }

  static std::shared_ptr<PaimonTableHandle> table(
      PaimonSchemaPtr tableSchema = schema()) {
    return std::make_shared<PaimonTableHandle>(
        "test-paimon", "orders", std::move(tableSchema), kReadProfile);
  }
};

TEST_F(PaimonTableHandleTest, tableDefinition) {
  const FileTableHandlePtr handle = table();
  EXPECT_EQ(handle->connectorId(), "test-paimon");
  EXPECT_EQ(handle->name(), "orders");
  EXPECT_EQ(handle->dbName(), "");
  EXPECT_EQ(
      *handle->dataColumns(), *ROW({{"id", BIGINT()}, {"value", BIGINT()}}));
  EXPECT_TRUE(handle->tableParameters().empty());
  EXPECT_TRUE(handle->filterColumnHandles().empty());
  EXPECT_TRUE(handle->subfieldFilters().empty());
  EXPECT_EQ(handle->remainingFilter(), nullptr);
  EXPECT_EQ(handle->sampleRate(), 1);
  EXPECT_FALSE(handle->supportsIndexLookup());
  const auto* paimonTable = handle->as<PaimonTableHandle>();
  ASSERT_NE(paimonTable, nullptr);
  EXPECT_EQ(paimonTable->tableType(), PaimonTableType::kPrimaryKey);
  EXPECT_EQ(paimonTable->schema()->primaryKeys(), (std::vector<int32_t>{8, 7}));
  EXPECT_EQ(table(schema({}))->tableType(), PaimonTableType::kAppendOnly);
  EXPECT_EQ(
      handle->toString(),
      "PaimonTableHandle[table orders, schemaId 1099511627776, tableType "
      "PRIMARY_KEY, readProfile SAME_SCHEMA_APPEND_BATCH, dataColumns "
      "ROW<id:BIGINT,value:BIGINT>, sampleRate 1]");

  const auto onlyPartition = std::make_shared<PaimonSchema>(
      0,
      3,
      ROW({{"p", BIGINT()}}),
      PaimonTypeMetadata{"ROW", false, std::nullopt, {}, {{"BIGINT", true, 0}}},
      std::vector<int32_t>{0});
  EXPECT_EQ(table(onlyPartition)->dataColumns()->size(), 0);
  const auto empty = std::make_shared<PaimonSchema>(
      0, 3, ROW({}), PaimonTypeMetadata{"ROW", false});
  EXPECT_EQ(table(empty)->dataColumns(), empty->rowType());
}

TEST_F(PaimonTableHandleTest, roundTrip) {
  common::SubfieldFilters filters;
  filters.emplace(
      common::Subfield("id"),
      std::make_unique<common::BigintRange>(1, 9, false));
  const auto remainingFilter =
      std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), Variant(true));
  const auto filterColumn = std::make_shared<HiveColumnHandle>(
      "value", FileColumnHandle::ColumnType::kRegular, BIGINT(), BIGINT());
  const std::unordered_map<std::string, std::string> queryOptions{
      {"query-custom", ""}, {"file.format", "orc"}};
  const auto original = std::make_shared<PaimonTableHandle>(
      "test-paimon",
      "orders",
      schema(),
      kReadProfile,
      std::move(filters),
      remainingFilter,
      std::vector<FileColumnHandlePtr>{filterColumn},
      queryOptions,
      0.5,
      "analytics");
  const auto obj = original->serialize();
  const auto copy = ISerializable::deserialize<PaimonTableHandle>(obj);
  ASSERT_NE(copy, nullptr);
  EXPECT_EQ(copy->serialize(), obj);
  EXPECT_EQ(*copy->schema(), *original->schema());
  EXPECT_EQ(copy->tableParameters(), queryOptions);
  EXPECT_EQ(copy->schema()->options().at("file.format"), "nimble");
  EXPECT_EQ(copy->remainingFilter()->toString(), remainingFilter->toString());
  ASSERT_EQ(copy->filterColumnHandles().size(), 1);
  EXPECT_EQ(
      copy->filterColumnHandles()[0]->toString(), filterColumn->toString());
  EXPECT_EQ(copy->dbName(), "analytics");
  EXPECT_EQ(copy->sampleRate(), 0.5);
  EXPECT_FALSE(obj.count("dataColumns"));
  EXPECT_FALSE(obj.count("tableType"));
}

TEST_F(PaimonTableHandleTest, structuredFilterPaths) {
  common::SubfieldFilters filters;
  filters.emplace(
      std::move(*common::Subfield::create("a.b")),
      std::make_unique<common::IsNotNull>());
  filters.emplace(common::Subfield("a.b"), std::make_unique<common::IsNull>());
  filters.emplace(
      common::Subfield("m[\"key\"].v"), std::make_unique<common::IsNotNull>());
  filters.emplace(
      common::Subfield("items[3]"), std::make_unique<common::IsNotNull>());
  filters.emplace(
      common::Subfield("items[*]"), std::make_unique<common::IsNotNull>());
  const auto rowType = ROW(
      {{"a.b", BIGINT()},
       {"a", ROW({{"b", BIGINT()}})},
       {"m", MAP(VARCHAR(), ROW({{"v", BIGINT()}}))},
       {"items", ARRAY(BIGINT())}});
  const auto tableSchema = std::make_shared<PaimonSchema>(
      0,
      3,
      rowType,
      PaimonTypeMetadata{
          "ROW",
          false,
          std::nullopt,
          {},
          {{"BIGINT", true, 0},
           {"ROW", true, 1, {}, {{"BIGINT", true, 2}}},
           {"MAP",
            true,
            3,
            {},
            {{"VARCHAR", false, std::nullopt, {100}},
             {"ROW", true, std::nullopt, {}, {{"BIGINT", true, 4}}}}},
           {"ARRAY", true, 5, {}, {{"BIGINT", true}}}}});
  filters.emplace(
      common::Subfield("m[*].v"), std::make_unique<common::IsNotNull>());
  const auto original = std::make_shared<PaimonTableHandle>(
      "test-paimon", "orders", tableSchema, kReadProfile, std::move(filters));
  const auto copy =
      ISerializable::deserialize<PaimonTableHandle>(original->serialize());
  ASSERT_EQ(copy->subfieldFilters().size(), original->subfieldFilters().size());
  for (const auto& [path, filter] : original->subfieldFilters()) {
    auto it = copy->subfieldFilters().find(path);
    ASSERT_NE(it, copy->subfieldFilters().end());
    EXPECT_EQ(it->second->serialize(), filter->serialize());
  }
}

TEST_F(PaimonTableHandleTest, splitValidation) {
  const auto handle = table(schema({}));
  handle->validateReadProfile();
  const auto makeSplit = [](PaimonTableType type,
                            const auto& partitions,
                            std::string connectorId = "test-paimon") {
    PaimonDataFile file;
    file.path = "/not-opened";
    return PaimonConnectorSplit(
        connectorId,
        1,
        type,
        dwio::common::FileFormat::DWRF,
        std::vector<PaimonDataFile>{file},
        partitions,
        std::nullopt);
  };
  const std::unordered_map<std::string, std::optional<std::string>> good{
      {"p", std::nullopt}};
  handle->validateSplit(makeSplit(PaimonTableType::kAppendOnly, good));
  handle->validateSplit(makeSplit(
      PaimonTableType::kAppendOnly,
      std::unordered_map<std::string, std::optional<std::string>>{{"p", ""}}));
  VELOX_ASSERT_THROW(
      handle->validateSplit(makeSplit(PaimonTableType::kPrimaryKey, good)),
      "Paimon split table type does not match table schema");
  VELOX_ASSERT_THROW(
      handle->validateSplit(
          makeSplit(PaimonTableType::kAppendOnly, good, "other")),
      "Paimon split connector does not match table handle");
  VELOX_ASSERT_THROW(
      handle->validateSplit(makeSplit(
          PaimonTableType::kAppendOnly,
          std::unordered_map<std::string, std::optional<std::string>>{})),
      "Paimon split partition names do not match table schema");
  VELOX_ASSERT_THROW(
      handle->validateSplit(makeSplit(
          PaimonTableType::kAppendOnly,
          std::unordered_map<std::string, std::optional<std::string>>{
              {"q", "1"}})),
      "Missing Paimon split partition 'p' declared by table schema");
  auto notNullable = handle->schema()->serialize();
  notNullable["metadata"]["children"][1]["nullable"] = false;
  VELOX_ASSERT_USER_THROW(
      table(PaimonSchema::create(notNullable))
          ->validateSplit(makeSplit(PaimonTableType::kAppendOnly, good)),
      "NULL Paimon partition 'p' is not nullable in target schema");
}

TEST_F(PaimonTableHandleTest, executableProfile) {
  const auto good = schema({})->serialize();
  table(PaimonSchema::create(good))->validateReadProfile();
  VELOX_ASSERT_USER_THROW(
      table()->validateReadProfile(),
      "Paimon same-schema append profile does not support primary-key tables; "
      "primary-key reading is not implemented yet");
  for (const auto& key :
       {"type",
        "batch-scan-mode",
        "deletion-vectors.enabled",
        "data-evolution.enabled",
        "data-file.thin-mode"}) {
    SCOPED_TRACE(key);
    auto wire = good;
    wire["options"].erase(key);
    VELOX_ASSERT_USER_THROW(
        table(PaimonSchema::create(wire))->validateReadProfile(),
        fmt::format(
            "Missing explicit Paimon profile option '{}'; provide normalized "
            "options because defaults are not resolved by the reader",
            key));
    wire["options"][key] = "unsupported";
    VELOX_ASSERT_USER_THROW(
        table(PaimonSchema::create(wire))->validateReadProfile(),
        fmt::format(
            "Paimon same-schema append profile requires '{}'='{}'; the current "
            "reader only implements ordinary same-schema append batch scans",
            key,
            good["options"][key].asString()));
  }
  for (const auto& key :
       {"scan.mode",
        "log.scan",
        "merge-engine",
        "sequence.field",
        "file.format.per.level",
        "scan.ignore-lost-file",
        "custom"}) {
    auto wire = good;
    wire["options"][key] = "unverified";
    VELOX_ASSERT_USER_THROW(
        table(PaimonSchema::create(wire))->validateReadProfile(),
        fmt::format(
            "Paimon same-schema append profile does not support option '{}'; "
            "its applicability to the current reader has not been classified",
            key));
  }
  auto writerOption = good;
  writerOption["options"]["write-buffer-size"] = "256 mb";
  VELOX_ASSERT_USER_THROW(
      table(PaimonSchema::create(writerOption))->validateReadProfile(),
      "Paimon same-schema append profile does not support option "
      "'write-buffer-size'; its applicability to the current reader has not "
      "been classified");
  for (const auto& value : {"-2", "0", "2147483648", "unknown"}) {
    auto wire = good;
    wire["options"]["bucket"] = value;
    VELOX_ASSERT_USER_THROW(
        table(PaimonSchema::create(wire))->validateReadProfile(),
        "Unsupported Paimon append bucket option");
  }
  auto wire = good;
  wire["options"]["file.format"] = "avro";
  VELOX_ASSERT_USER_THROW(
      table(PaimonSchema::create(wire))->validateReadProfile(),
      "Unsupported Paimon schema file.format 'avro'; the current reader "
      "supports nimble, dwrf, orc and parquet");
  wire = good;
  wire["options"]["row-tracking.enabled"] = "true";
  table(PaimonSchema::create(wire))->validateReadProfile();
  wire["options"]["row-tracking.enabled"] = "unknown";
  VELOX_ASSERT_USER_THROW(
      table(PaimonSchema::create(wire))->validateReadProfile(),
      "Paimon profile requires explicit boolean 'row-tracking.enabled'; "
      "provide normalized options because defaults are not resolved by the reader");
  wire["options"].erase("row-tracking.enabled");
  VELOX_ASSERT_USER_THROW(
      table(PaimonSchema::create(wire))->validateReadProfile(),
      "Paimon profile requires explicit boolean 'row-tracking.enabled'; "
      "provide normalized options because defaults are not resolved by the reader");
  const auto queryOptions = std::make_shared<PaimonTableHandle>(
      "test-paimon",
      "orders",
      schema({}),
      kReadProfile,
      common::SubfieldFilters{},
      nullptr,
      std::vector<FileColumnHandlePtr>{},
      std::unordered_map<std::string, std::string>{
          {"skip.header.line.count", "1"}});
  VELOX_ASSERT_USER_THROW(
      queryOptions->validateReadProfile(),
      "Paimon same-schema append profile does not support query options; "
      "query-option interpretation is not implemented yet");
  const auto dateSchema = std::make_shared<PaimonSchema>(
      0,
      3,
      ROW({{"date", DATE()}}),
      PaimonTypeMetadata{"ROW", false, std::nullopt, {}, {{"DATE", true, 0}}},
      std::vector<int32_t>{},
      std::vector<int32_t>{},
      schema({})->options());
  VELOX_ASSERT_USER_THROW(
      table(dateSchema)->validateReadProfile(),
      "Unsupported Paimon same-schema append logical type 'DATE'; the current "
      "reader is limited to BOOLEAN, INTEGER, BIGINT, DOUBLE and VARCHAR");
}

TEST_F(PaimonTableHandleTest, conditionValidation) {
  const auto makeHandle = [](common::SubfieldFilters filters,
                             core::TypedExprPtr expression) {
    return std::make_shared<PaimonTableHandle>(
        "test-paimon",
        "orders",
        schema({}),
        kReadProfile,
        std::move(filters),
        std::move(expression));
  };
  common::SubfieldFilters badName;
  badName.emplace(
      common::Subfield("missing"), std::make_unique<common::IsNull>());
  VELOX_ASSERT_USER_THROW(
      makeHandle(std::move(badName), nullptr), "Unknown Paimon field: missing");
  common::SubfieldFilters badType;
  badType.emplace(
      common::Subfield("p"),
      std::make_unique<common::BigintRange>(1, 9, false));
  VELOX_ASSERT_USER_THROW(
      makeHandle(std::move(badType), nullptr),
      "is not supported for target type VARCHAR");
  const auto field =
      std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "id");
  const auto expression = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(), std::vector<core::TypedExprPtr>{field}, "test_predicate");
  VELOX_ASSERT_USER_THROW(
      makeHandle({}, expression),
      "Paimon expression field 'id' type does not match target schema");
  const auto missing =
      std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "missing");
  VELOX_ASSERT_USER_THROW(
      makeHandle(
          {},
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(),
              std::vector<core::TypedExprPtr>{missing},
              "test_predicate")),
      "Unknown Paimon column 'missing' in target schema");
  const auto lambda = std::make_shared<core::LambdaTypedExpr>(
      ROW({{"id", BIGINT()}}),
      std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "id"));
  VELOX_ASSERT_USER_THROW(
      makeHandle(
          {},
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(),
              std::vector<core::TypedExprPtr>{lambda},
              "test_predicate")),
      "Paimon same-schema append profile does not support Lambda filters; "
      "scope-aware column validation is not implemented yet");
  for (const auto& inputType :
       {schema({})->rowType(), ROW({{"id", INTEGER()}})}) {
    const auto explicitInput = std::make_shared<core::CallTypedExpr>(
        BOOLEAN(),
        std::vector<core::TypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(
                BIGINT(),
                std::make_shared<core::InputTypedExpr>(inputType),
                "id")},
        "test_predicate");
    if (*inputType == *schema({})->rowType()) {
      EXPECT_NE(makeHandle({}, explicitInput), nullptr);
    } else {
      VELOX_ASSERT_USER_THROW(
          makeHandle({}, explicitInput),
          "Paimon expression input type does not match target schema");
    }
  }
  VELOX_ASSERT_USER_THROW(
      makeHandle(
          {},
          std::make_shared<core::CallTypedExpr>(
              BOOLEAN(),
              std::vector<core::TypedExprPtr>{
                  std::make_shared<core::InputTypedExpr>(
                      schema({})->rowType())},
              "test_predicate")),
      "Paimon remaining filter must access input columns by name");
}

TEST_F(PaimonTableHandleTest, invalidDefinition) {
  VELOX_ASSERT_THROW(table(nullptr), "Missing Paimon table schema");
  VELOX_ASSERT_THROW(
      PaimonTableHandle("", "orders", schema(), kReadProfile),
      "Empty Paimon connector ID");
  VELOX_ASSERT_THROW(
      PaimonTableHandle("test-paimon", "", schema(), kReadProfile),
      "Empty Paimon table name");
  for (double rate :
       {0.0,
        -1.0,
        1.1,
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    VELOX_ASSERT_THROW(
        PaimonTableHandle(
            "test-paimon",
            "orders",
            schema(),
            kReadProfile,
            {},
            nullptr,
            {},
            {},
            rate),
        "Paimon sample rate must be finite and in (0, 1]");
  }
  VELOX_ASSERT_THROW(
      PaimonTableHandle(
          "test-paimon",
          "orders",
          schema(),
          kReadProfile,
          {},
          std::make_shared<core::ConstantTypedExpr>(
              BIGINT(), Variant(int64_t{1}))),
      "Paimon remaining filter must have BOOLEAN type");
  VELOX_ASSERT_THROW(
      PaimonTableHandle(
          "test-paimon",
          "orders",
          schema(),
          kReadProfile,
          {},
          nullptr,
          {nullptr}),
      "Missing Paimon filter column handle");
  common::SubfieldFilters missingFilter;
  missingFilter.emplace(common::Subfield("id"), nullptr);
  VELOX_ASSERT_THROW(
      PaimonTableHandle(
          "test-paimon",
          "orders",
          schema(),
          kReadProfile,
          std::move(missingFilter)),
      "Missing Paimon subfield filter");
}

TEST_F(PaimonTableHandleTest, malformedWire) {
  const auto good = table()->serialize();
  for (const auto& key : good.keys()) {
    auto obj = good;
    obj.erase(key);
    EXPECT_THROW(PaimonTableHandle::create(obj, nullptr), std::exception)
        << key;
  }
  for (const folly::dynamic& version :
       {folly::dynamic(0), folly::dynamic(2), folly::dynamic("1")}) {
    auto obj = good;
    obj["version"] = version;
    VELOX_ASSERT_THROW(
        PaimonTableHandle::create(obj, nullptr),
        "Unsupported Paimon table handle wire version");
  }
  auto obj = good;
  obj["queryOptions"]["bucket"] = 1;
  VELOX_ASSERT_THROW(
      PaimonTableHandle::create(obj, nullptr),
      "Paimon query options must be strings");
  obj = good;
  obj["schema"]["metadata"]["children"][0]["fieldId"] = 8;
  VELOX_ASSERT_THROW(
      PaimonTableHandle::create(obj, nullptr), "Duplicate Paimon field ID");
  obj = good;
  obj["filterColumnHandles"] = folly::dynamic::object;
  VELOX_ASSERT_THROW(
      PaimonTableHandle::create(obj, nullptr),
      "Expected Paimon filter column handle array");
  obj = good;
  obj["sampleRate"] = "1";
  VELOX_ASSERT_THROW(
      PaimonTableHandle::create(obj, nullptr),
      "Expected Paimon numeric sample rate");
  obj = good;
  const folly::dynamic entry = folly::dynamic::object(
      "subfield",
      folly::dynamic::array(
          folly::dynamic::object("kind", "field")("value", "id")))(
      "filter", common::IsNull().serialize());
  obj["subfieldFilters"] = folly::dynamic::array(entry, entry);
  VELOX_ASSERT_THROW(
      PaimonTableHandle::create(obj, nullptr),
      "Duplicate Paimon subfield filter");
  obj["subfieldFilters"] = folly::dynamic::array(entry);
  obj["subfieldFilters"][0]["subfield"][0]["kind"] = "invalid";
  VELOX_ASSERT_THROW(
      PaimonTableHandle::create(obj, nullptr), "Unknown Paimon path kind");
}

} // namespace
} // namespace facebook::velox::connector::hive::paimon
