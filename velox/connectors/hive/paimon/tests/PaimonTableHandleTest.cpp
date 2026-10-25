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
#include "velox/connectors/hive/paimon/PaimonColumnHandle.h"

#include <gtest/gtest.h>
#include <limits>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/Expressions.h"

namespace facebook::velox::connector::hive::paimon {
namespace {

class PaimonTableHandleTest : public testing::Test {
 protected:
  void SetUp() override {
    Type::registerSerDe();
    common::Filter::registerSerDe();
    core::ITypedExpr::registerSerDe();
    PaimonColumnHandle::registerSerDe();
    PaimonTableHandle::registerSerDe();
  }

  static RowTypePtr rowType() {
    return ROW({{"id", BIGINT()}, {"p", VARCHAR()}, {"value", BIGINT()}});
  }

  static std::shared_ptr<PaimonTableHandle> table(
      PaimonTableType type = PaimonTableType::kAppendOnly) {
    return std::make_shared<PaimonTableHandle>(
        "test-paimon",
        "orders",
        int64_t{1} << 40,
        rowType(),
        std::vector<int32_t>{7, 8, 20},
        type,
        std::vector<std::string>{"p"});
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
  const auto* paimon = handle->as<PaimonTableHandle>();
  ASSERT_NE(paimon, nullptr);
  EXPECT_EQ(paimon->schemaId(), int64_t{1} << 40);
  EXPECT_EQ(*paimon->rowType(), *rowType());
  EXPECT_EQ(paimon->fieldIds(), (std::vector<int32_t>{7, 8, 20}));
  EXPECT_EQ(paimon->partitionKeys(), (std::vector<std::string>{"p"}));
  EXPECT_EQ(paimon->tableType(), PaimonTableType::kAppendOnly);
  // The handle can describe primary-key tables even though reading them
  // is not yet supported.
  EXPECT_EQ(
      table(PaimonTableType::kPrimaryKey)->tableType(),
      PaimonTableType::kPrimaryKey);
  EXPECT_EQ(
      handle->toString(),
      "PaimonTableHandle[table orders, schemaId 1099511627776, tableType "
      "APPEND_ONLY, dataColumns ROW<id:BIGINT,value:BIGINT>, sampleRate 1]");

  const PaimonTableHandle onlyPartition(
      "test-paimon",
      "t",
      0,
      ROW({{"p", BIGINT()}}),
      {0},
      PaimonTableType::kAppendOnly,
      {"p"});
  EXPECT_EQ(onlyPartition.dataColumns()->size(), 0);
  const auto emptyType = ROW({});
  const PaimonTableHandle empty(
      "test-paimon", "t", 0, emptyType, {}, PaimonTableType::kAppendOnly);
  EXPECT_EQ(empty.dataColumns(), emptyType);
}

TEST_F(PaimonTableHandleTest, roundTrip) {
  common::SubfieldFilters filters;
  filters.emplace(
      common::Subfield("id"),
      std::make_unique<common::BigintRange>(1, 9, false));
  const auto remaining =
      std::make_shared<core::ConstantTypedExpr>(BOOLEAN(), Variant(true));
  const auto filterColumn = std::make_shared<PaimonColumnHandle>(
      "value", 20, FileColumnHandle::ColumnType::kRegular, BIGINT());
  const std::unordered_map<std::string, std::string> options{
      {"deletion-vectors.enabled", "false"},
      {"data-evolution.enabled", "false"},
      {"data-file.thin-mode", "false"},
      {"row-tracking.enabled", "false"},
      {"bucket", "8"},
      {"skip.header.line.count", "1"}};
  const PaimonTableHandle original(
      "test-paimon",
      "orders",
      int64_t{1} << 40,
      rowType(),
      {7, 8, 20},
      PaimonTableType::kAppendOnly,
      {"p"},
      options,
      std::move(filters),
      remaining,
      {filterColumn},
      0.5,
      "analytics");
  const auto obj = original.serialize();
  const auto copy = ISerializable::deserialize<PaimonTableHandle>(obj);
  ASSERT_NE(copy, nullptr);
  EXPECT_EQ(copy->serialize(), obj);
  EXPECT_EQ(copy->schemaId(), original.schemaId());
  EXPECT_EQ(*copy->rowType(), *original.rowType());
  EXPECT_EQ(copy->fieldIds(), original.fieldIds());
  EXPECT_EQ(copy->partitionKeys(), original.partitionKeys());
  EXPECT_EQ(copy->tableType(), original.tableType());
  EXPECT_EQ(copy->options(), options);
  EXPECT_TRUE(copy->tableParameters().empty());
  EXPECT_EQ(copy->remainingFilter()->toString(), remaining->toString());
  ASSERT_EQ(copy->filterColumnHandles().size(), 1);
  EXPECT_EQ(
      copy->filterColumnHandles()[0]->toString(), filterColumn->toString());
  EXPECT_EQ(copy->dbName(), "analytics");
  EXPECT_EQ(copy->sampleRate(), 0.5);
  EXPECT_FALSE(obj.count("dataColumns"));
  EXPECT_FALSE(obj.count("schema"));
  EXPECT_FALSE(obj.count("readProfile"));
  EXPECT_FALSE(obj.count("queryOptions"));
}

TEST_F(PaimonTableHandleTest, defaultValues) {
  const folly::dynamic requiredFields = folly::dynamic::object(
      "name", "PaimonTableHandle")("version", 1)("connectorId", "test-paimon")(
      "tableName", "orders")("schemaId", int64_t{1} << 40)(
      "rowType", rowType()->serialize())(
      "fieldIds", folly::dynamic::array(7, 8, 20))("tableType", "APPEND_ONLY")(
      "partitionKeys", folly::dynamic::array("p"))(
      "options", folly::dynamic::object)(
      "subfieldFilters", folly::dynamic::array);
  for (bool explicitDefaults : {false, true}) {
    auto obj = requiredFields;
    if (explicitDefaults) {
      obj["remainingFilter"] = nullptr;
      obj["sampleRate"] = 1.0;
      obj["dbName"] = "";
      obj["filterColumnHandles"] = folly::dynamic::array;
    }
    const auto copy = ISerializable::deserialize<PaimonTableHandle>(obj);
    EXPECT_EQ(copy->remainingFilter(), nullptr);
    EXPECT_EQ(copy->sampleRate(), 1.0);
    EXPECT_TRUE(copy->dbName().empty());
    EXPECT_TRUE(copy->filterColumnHandles().empty());
    EXPECT_EQ(copy->serialize(), requiredFields);
  }
}

TEST_F(PaimonTableHandleTest, subfieldStringSerde) {
  const auto nestedType = ROW({{"a", BIGINT()}});
  common::SubfieldFilters filters;
  filters.emplace(
      common::Subfield("r.a"), std::make_unique<common::IsNotNull>());
  std::vector<common::Subfield> requiredSubfields;
  requiredSubfields.emplace_back("r.a");
  const auto filterColumn = std::make_shared<PaimonColumnHandle>(
      "r",
      7,
      FileColumnHandle::ColumnType::kRegular,
      nestedType,
      std::move(requiredSubfields));
  const PaimonTableHandle original(
      "test-paimon",
      "orders",
      0,
      ROW({{"r", nestedType}}),
      {7},
      PaimonTableType::kAppendOnly,
      {},
      {},
      std::move(filters),
      nullptr,
      {filterColumn});
  const auto obj = original.serialize();
  EXPECT_EQ(obj["subfieldFilters"][0]["subfield"].asString(), "r.a");
  EXPECT_EQ(
      obj["filterColumnHandles"][0]["requiredSubfields"],
      folly::dynamic::array("r.a"));
  const auto copy = ISerializable::deserialize<PaimonTableHandle>(obj);
  ASSERT_EQ(copy->subfieldFilters().size(), 1);
  const auto& path = copy->subfieldFilters().begin()->first;
  EXPECT_EQ(path, common::Subfield("r.a"));
  EXPECT_EQ(
      copy->subfieldFilters().begin()->second->serialize(),
      original.subfieldFilters().begin()->second->serialize());
  ASSERT_EQ(copy->filterColumnHandles().size(), 1);
  EXPECT_EQ(
      copy->filterColumnHandles()[0]->requiredSubfields(),
      filterColumn->requiredSubfields());
}

TEST_F(PaimonTableHandleTest, invalidDefinition) {
  const auto make = [](RowTypePtr type,
                       std::vector<int32_t> ids,
                       std::vector<std::string> partitions = {},
                       int64_t schemaId = 0,
                       double sampleRate = 1.0) {
    return PaimonTableHandle(
        "test-paimon",
        "orders",
        schemaId,
        std::move(type),
        std::move(ids),
        PaimonTableType::kAppendOnly,
        std::move(partitions),
        {},
        {},
        nullptr,
        {},
        sampleRate);
  };
  VELOX_ASSERT_USER_THROW(make(nullptr, {}), "Missing Paimon table row type");
  VELOX_ASSERT_USER_THROW(
      make(rowType(), {7, 8}),
      "Paimon field IDs must be aligned to table columns");
  VELOX_ASSERT_USER_THROW(
      make(rowType(), {7, -1, 20}), "Paimon field ID must be nonnegative");
  VELOX_ASSERT_USER_THROW(
      make(rowType(), {7, 7, 20}), "Duplicate Paimon field ID");
  VELOX_ASSERT_USER_THROW(
      make(rowType(), {7, 8, 20}, {"missing"}),
      "Unknown Paimon partition column 'missing'");
  VELOX_ASSERT_USER_THROW(
      make(rowType(), {7, 8, 20}, {"p", "p"}),
      "Duplicate Paimon partition column 'p'");
  VELOX_ASSERT_USER_THROW(
      make(rowType(), {7, 8, 20}, {}, -1),
      "Paimon schema ID must be nonnegative");
  for (double rate :
       {0.0,
        -1.0,
        1.1,
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    VELOX_ASSERT_USER_THROW(
        make(rowType(), {7, 8, 20}, {}, 0, rate),
        "Paimon sample rate must be finite and in (0, 1]");
  }
}

TEST_F(PaimonTableHandleTest, malformedWire) {
  const auto good = table()->serialize();
  auto obj = good;
  obj["version"] = 2;
  VELOX_ASSERT_USER_THROW(
      PaimonTableHandle::create(obj, nullptr),
      "Unsupported Paimon table handle wire version");
  obj = good;
  obj["fieldIds"][0] = int64_t{1} << 40;
  EXPECT_THROW(PaimonTableHandle::create(obj, nullptr), VeloxUserError);
  obj = good;
  obj["fieldIds"][1] = 7;
  VELOX_ASSERT_USER_THROW(
      PaimonTableHandle::create(obj, nullptr), "Duplicate Paimon field ID");
  obj = good;
  obj["rowType"] = nullptr;
  EXPECT_THROW(PaimonTableHandle::create(obj, nullptr), VeloxUserError);
}

TEST_F(PaimonTableHandleTest, localCallbackIsNotWireState) {
  const auto column = std::make_shared<PaimonColumnHandle>(
      "id",
      7,
      FileColumnHandle::ColumnType::kRegular,
      BIGINT(),
      std::vector<common::Subfield>{},
      std::vector<NamedExtraction>{},
      /*partitionDateValueDaysSinceEpoch=*/false,
      [](VectorPtr&) {});
  const PaimonTableHandle handle(
      "test-paimon",
      "orders",
      0,
      rowType(),
      {7, 8, 20},
      PaimonTableType::kAppendOnly,
      {"p"},
      {},
      {},
      nullptr,
      {column});
  EXPECT_TRUE(handle.filterColumnHandles()[0]->postProcessor());
  VELOX_ASSERT_USER_THROW(
      handle.serialize(), "Paimon column callbacks cannot be serialized");
}

} // namespace
} // namespace facebook::velox::connector::hive::paimon
