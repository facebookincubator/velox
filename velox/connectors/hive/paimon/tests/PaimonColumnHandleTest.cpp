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
#include "velox/connectors/hive/paimon/PaimonColumnHandle.h"

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox::connector::hive::paimon {
namespace {

using ColumnType = FileColumnHandle::ColumnType;

class PaimonColumnHandleTest : public testing::Test {
 protected:
  void SetUp() override {
    Type::registerSerDe();
    PaimonColumnHandle::registerSerDe();
  }
};

TEST_F(PaimonColumnHandleTest, columnMetadata) {
  const PaimonColumnHandle column("id", 73, ColumnType::kRegular, BIGINT());
  EXPECT_EQ(column.name(), "id");
  EXPECT_EQ(column.fieldId(), 73);
  EXPECT_EQ(column.columnType(), ColumnType::kRegular);
  EXPECT_EQ(column.schemaType(), BIGINT());
  EXPECT_EQ(column.dataType(), column.schemaType());
  EXPECT_TRUE(column.requiredSubfields().empty());
  EXPECT_TRUE(column.extractions().empty());
  EXPECT_FALSE(column.postProcessor());
  EXPECT_FALSE(column.isPartitionKey());
  EXPECT_FALSE(column.isPartitionDateValueDaysSinceEpoch());
  EXPECT_EQ(
      column.toString(),
      "PaimonColumnHandle[name id, fieldId 73, columnType Regular, "
      "schemaType BIGINT, dataType BIGINT]");
  const folly::dynamic expected =
      folly::dynamic::object("name", "PaimonColumnHandle")("version", 1)(
          "columnName", "id")("fieldId", 73)("columnType", "Regular")(
          "schemaType", BIGINT()->serialize())(
          "requiredSubfields", folly::dynamic::array)(
          "extractions", folly::dynamic::array)(
          "partitionDateValueDaysSinceEpoch", false);
  EXPECT_EQ(column.serialize(), expected);
}

TEST_F(PaimonColumnHandleTest, roundTrip) {
  std::vector<common::Subfield> paths;
  paths.emplace_back("r.items[*].value");
  const PaimonColumnHandle column(
      "r",
      29,
      ColumnType::kRegular,
      ROW({{"items", ARRAY(ROW({{"value", BIGINT()}, {"other", VARCHAR()}}))}}),
      std::move(paths));
  const auto obj = column.serialize();
  const auto copy = ISerializable::deserialize<PaimonColumnHandle>(obj);
  ASSERT_NE(copy, nullptr);
  EXPECT_EQ(copy->serialize(), obj);
  EXPECT_EQ(copy->fieldId(), 29);
  ASSERT_EQ(copy->requiredSubfields().size(), 1);
  EXPECT_EQ(copy->requiredSubfields()[0], column.requiredSubfields()[0]);

  const PaimonColumnHandle partition(
      "day", 5, ColumnType::kPartitionKey, DATE(), {}, {}, true);
  const auto partitionCopy =
      ISerializable::deserialize<PaimonColumnHandle>(partition.serialize());
  EXPECT_TRUE(partitionCopy->isPartitionKey());
  EXPECT_TRUE(partitionCopy->isPartitionDateValueDaysSinceEpoch());
  EXPECT_EQ(partitionCopy->schemaType(), DATE());
}

TEST_F(PaimonColumnHandleTest, literalNamesRoundTrip) {
  std::vector<common::Subfield> paths;
  paths.push_back(std::move(*common::Subfield::create("a.b[0]")));
  const PaimonColumnHandle column(
      "a.b[0]", 91, ColumnType::kRegular, BIGINT(), std::move(paths));
  const auto copy =
      ISerializable::deserialize<PaimonColumnHandle>(column.serialize());
  ASSERT_EQ(copy->requiredSubfields().size(), 1);
  EXPECT_EQ(copy->requiredSubfields()[0], column.requiredSubfields()[0]);
  EXPECT_EQ(copy->requiredSubfields()[0].path().size(), 1);
}

TEST_F(PaimonColumnHandleTest, extractionRoundTrip) {
  const auto schemaType = MAP(VARCHAR(), BIGINT());
  const NamedExtraction keys{
      "keys",
      {ExtractionPathElement::simple(ExtractionStep::kMapKeys)},
      ARRAY(VARCHAR())};
  const NamedExtraction size{
      "size", {ExtractionPathElement::simple(ExtractionStep::kSize)}, BIGINT()};
  for (const auto& extractions :
       {std::vector<NamedExtraction>{keys},
        std::vector<NamedExtraction>{keys, size}}) {
    const PaimonColumnHandle column(
        "m", 30, ColumnType::kRegular, schemaType, {}, extractions);
    const FileColumnHandle& fileColumn = column;
    EXPECT_EQ(fileColumn.schemaType(), schemaType);
    EXPECT_EQ(fileColumn.extractions(), extractions);
    const auto copy =
        ISerializable::deserialize<PaimonColumnHandle>(column.serialize());
    EXPECT_EQ(copy->serialize(), column.serialize());
    EXPECT_EQ(copy->extractions(), extractions);
    TypePtr expectedType = ARRAY(VARCHAR());
    if (extractions.size() > 1) {
      expectedType = ROW({{"keys", ARRAY(VARCHAR())}, {"size", BIGINT()}});
    }
    EXPECT_EQ(*copy->dataType(), *expectedType);
    EXPECT_EQ(*copy->schemaType(), *schemaType);
    EXPECT_EQ(*fileColumn.dataType(), *expectedType);
  }
}

TEST_F(PaimonColumnHandleTest, extractionOutputNames) {
  const auto output = ROW({{"alias", BIGINT()}});
  const PaimonColumnHandle column(
      "r",
      8,
      ColumnType::kRegular,
      ROW({{"value", ROW({{"x", BIGINT()}})}}),
      {},
      {{"value", {ExtractionPathElement::structField("value")}, output}});
  const auto copy =
      ISerializable::deserialize<PaimonColumnHandle>(column.serialize());
  EXPECT_EQ(*column.dataType(), *output);
  EXPECT_EQ(*copy->dataType(), *output);
}

TEST_F(PaimonColumnHandleTest, invalidMetadata) {
  VELOX_ASSERT_USER_THROW(
      PaimonColumnHandle("id", -1, ColumnType::kRegular, BIGINT()),
      "Paimon field ID must be nonnegative");
  VELOX_ASSERT_USER_THROW(
      PaimonColumnHandle("", 1, ColumnType::kRegular, BIGINT()),
      "Paimon column name must not be empty");
  VELOX_ASSERT_USER_THROW(
      PaimonColumnHandle("id", 1, ColumnType::kRegular, nullptr),
      "Missing Paimon column schema type");

  VELOX_ASSERT_USER_THROW(
      PaimonColumnHandle(
          "m",
          1,
          ColumnType::kRegular,
          MAP(VARCHAR(), BIGINT()),
          {},
          {{"size",
            {ExtractionPathElement::simple(ExtractionStep::kSize)},
            VARCHAR()}}),
      "Paimon extraction 'size' output type must be BIGINT");
}

TEST_F(PaimonColumnHandleTest, malformedWire) {
  const PaimonColumnHandle column("id", 7, ColumnType::kRegular, BIGINT());
  auto obj = column.serialize();
  obj["version"] = 2;
  VELOX_ASSERT_USER_THROW(
      PaimonColumnHandle::create(obj),
      "Unsupported Paimon column handle wire version");
  obj = column.serialize();
  obj["fieldId"] = int64_t{1} << 40;
  EXPECT_THROW(PaimonColumnHandle::create(obj), VeloxUserError);
  obj = column.serialize();
  obj["fieldId"] = -1;
  VELOX_ASSERT_USER_THROW(
      PaimonColumnHandle::create(obj), "Paimon field ID must be nonnegative");
  obj = column.serialize();
  obj["requiredSubfields"].push_back("id");
  VELOX_ASSERT_USER_THROW(
      PaimonColumnHandle::create(obj), "Expected Paimon subfield path");
}

TEST_F(PaimonColumnHandleTest, localCallbackIsNotWireState) {
  bool called = false;
  const PaimonColumnHandle column(
      "id",
      7,
      ColumnType::kRegular,
      BIGINT(),
      {},
      {},
      false,
      [&called](VectorPtr&) { called = true; });
  const FileColumnHandle& fileColumn = column;
  ASSERT_TRUE(fileColumn.postProcessor());
  VectorPtr unused;
  fileColumn.postProcessor()(unused);
  EXPECT_TRUE(called);
  VELOX_ASSERT_USER_THROW(
      column.serialize(), "Paimon column callbacks cannot be serialized");
}

} // namespace
} // namespace facebook::velox::connector::hive::paimon
