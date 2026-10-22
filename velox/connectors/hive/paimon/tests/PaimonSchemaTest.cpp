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
#include "velox/connectors/hive/paimon/PaimonSchema.h"

#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>
#include <limits>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/type/TypeSerde.h"

namespace facebook::velox::connector::hive::paimon {
namespace {

using common::Subfield;

class PaimonSchemaTest : public testing::Test {
 protected:
  void SetUp() override {
    Type::registerSerDe();
  }

  static PaimonSchemaPtr nestedSchema(
      int64_t id = 10,
      int32_t version = 3,
      std::unordered_map<std::string, std::string> options = {}) {
    return std::make_shared<PaimonSchema>(
        id,
        version,
        ROW(
            {{"id", BIGINT()},
             {"region", VARCHAR()},
             {"r", ROW({{"x", INTEGER()}})},
             {"a", ARRAY(ROW({{"value", VARCHAR()}}))},
             {"m", MAP(VARCHAR(), ROW({{"n", DECIMAL(20, 3)}}))}}),
        PaimonTypeMetadata{
            "ROW",
            false,
            std::nullopt,
            {},
            {{"BIGINT", false, 0},
             {"VARCHAR", true, 1, {20}},
             {"ROW", true, 20, {}, {{"INTEGER", false, 21}}},
             {"ARRAY",
              true,
              30,
              {},
              {{"ROW",
                true,
                std::nullopt,
                {},
                {{"VARCHAR", true, 31, {100}}}}}},
             {"MAP",
              true,
              40,
              {},
              {{"VARCHAR", true, std::nullopt, {40}},
               {"ROW",
                false,
                std::nullopt,
                {},
                {{"DECIMAL", true, 41, {20, 3}}}}}}}},
        std::vector<int32_t>{1},
        std::vector<int32_t>{1, 0},
        std::move(options));
  }
};

TEST_F(PaimonSchemaTest, nativeTypesAndLogicalAttributes) {
  const std::vector<std::pair<TypePtr, PaimonTypeMetadata>> cases{
      {BOOLEAN(), {"BOOLEAN", false}},
      {TINYINT(), {"TINYINT", true}},
      {SMALLINT(), {"SMALLINT", false}},
      {INTEGER(), {"INTEGER", true}},
      {BIGINT(), {"BIGINT", false}},
      {REAL(), {"FLOAT", true}},
      {DOUBLE(), {"DOUBLE", false}},
      {DATE(), {"DATE", true}},
      {VARCHAR(), {"CHAR", false, std::nullopt, {8}}},
      {VARCHAR(), {"VARCHAR", true, std::nullopt, {INT32_MAX}}},
      {VARBINARY(), {"BINARY", true, std::nullopt, {16}}},
      {VARBINARY(), {"VARBINARY", false, std::nullopt, {INT32_MAX}}},
      {DECIMAL(10, 2), {"DECIMAL", true, std::nullopt, {10, 2}}},
      {DECIMAL(38, 0), {"DECIMAL", false, std::nullopt, {38, 0}}},
      {TIMESTAMP(), {"TIMESTAMP_WITHOUT_TIME_ZONE", true, std::nullopt, {9}}},
      {TIMESTAMP(),
       {"TIMESTAMP_WITH_LOCAL_TIME_ZONE", false, std::nullopt, {0}}}};
  for (const auto& [type, metadata] : cases) {
    SCOPED_TRACE(metadata.logicalType);
    metadata.validate(type, false);
    const auto copy = PaimonTypeMetadata::create(metadata.serialize());
    EXPECT_EQ(copy, metadata);
    copy.validate(type, false);
    auto field = metadata;
    field.fieldId = 7;
    const PaimonSchema schema(
        0,
        3,
        ROW({{"field", type}}),
        PaimonTypeMetadata{"ROW", false, std::nullopt, {}, {field}});
    const auto schemaCopy = PaimonSchema::create(schema.serialize());
    EXPECT_EQ(*schemaCopy->fieldType(7), *type);
    EXPECT_EQ(schemaCopy->fieldMetadata(7), field);
  }
}

TEST_F(PaimonSchemaTest, roundTrip) {
  const auto schema = nestedSchema(
      int64_t{1} << 40,
      3,
      {{"file.format", "nimble"}, {"bucket", "-1"}, {"nested-field", "true"}});
  const auto serialized = schema->serialize();
  const auto copy = PaimonSchema::create(serialized);
  EXPECT_EQ(copy->serialize(), serialized);
  EXPECT_EQ(copy->schemaId(), int64_t{1} << 40);
  EXPECT_EQ(copy->schemaVersion(), 3);
  EXPECT_EQ(*copy->rowType(), *schema->rowType());
  EXPECT_EQ(copy->metadata(), schema->metadata());
  EXPECT_EQ(copy->partitionKeys(), (std::vector<int32_t>{1}));
  EXPECT_EQ(copy->primaryKeys(), (std::vector<int32_t>{1, 0}));
  EXPECT_TRUE(copy->fieldMetadata(40).children[0].nullable);
  EXPECT_FALSE(copy->fieldMetadata(40).children[1].nullable);
  EXPECT_FALSE(copy->fieldMetadata(30).children[0].fieldId.has_value());

  const PaimonSchema empty(
      INT64_MAX, 1, ROW({}), PaimonTypeMetadata{"ROW", false});
  EXPECT_EQ(
      PaimonSchema::create(empty.serialize())->serialize(), empty.serialize());
}

TEST_F(PaimonSchemaTest, selfContainedTypeWire) {
  auto& cache = serializedTypeCache();
  ASSERT_FALSE(cache.isEnabled());
  cache.enable({.minRowTypeSize = 1});
  SCOPE_EXIT {
    cache.disable();
    cache.clear();
  };
  const auto schema = nestedSchema();
  EXPECT_TRUE(schema->rowType()->serialize().count("ref"));
  const auto cacheSize = cache.size();
  const auto serialized = schema->serialize();
  const auto copy = PaimonSchema::create(serialized);
  EXPECT_EQ(copy->serialize(), serialized);
  EXPECT_EQ(cache.size(), cacheSize);
  EXPECT_EQ(deserializedTypeCache().size(), 0);
}

TEST_F(PaimonSchemaTest, fieldIdentityPaths) {
  const auto schema = nestedSchema();
  EXPECT_EQ(schema->fieldIndexPath(0), (std::vector<column_index_t>{0}));
  EXPECT_EQ(schema->fieldIndexPath(21), (std::vector<column_index_t>{2, 0}));
  EXPECT_EQ(schema->fieldIndexPath(31), (std::vector<column_index_t>{3, 0, 0}));
  EXPECT_EQ(schema->fieldIndexPath(41), (std::vector<column_index_t>{4, 1, 0}));
  EXPECT_EQ(*schema->fieldType(21), *INTEGER());
  EXPECT_EQ(*schema->fieldType(41), *DECIMAL(20, 3));
  EXPECT_EQ(
      schema->fieldIdPath(Subfield("r.x")), (std::vector<int32_t>{20, 21}));
  EXPECT_EQ(
      schema->fieldIdPath(Subfield("a[3].value")),
      (std::vector<int32_t>{30, 31}));
  EXPECT_EQ(
      schema->fieldIdPath(Subfield("a[*].value")),
      (std::vector<int32_t>{30, 31}));
  EXPECT_EQ(
      schema->fieldIdPath(Subfield("m[\"key\"].n")),
      (std::vector<int32_t>{40, 41}));
  EXPECT_EQ(
      schema->fieldIdPath(Subfield("m[*].n")), (std::vector<int32_t>{40, 41}));
  EXPECT_EQ(schema->fieldIdPath(Subfield("id")), (std::vector<int32_t>{0}));

  VELOX_ASSERT_THROW(schema->fieldIndexPath(999), "Unknown Paimon field ID");
  VELOX_ASSERT_THROW(
      schema->fieldIdPath(Subfield("missing")), "Unknown Paimon field");
  VELOX_ASSERT_THROW(
      schema->fieldIdPath(Subfield("r.y")), "Unknown Paimon subfield");
  VELOX_ASSERT_THROW(schema->fieldIdPath(Subfield("id.x")), "ROW required");
  VELOX_ASSERT_THROW(
      schema->fieldIdPath(Subfield("a[0].value")), "Invalid ARRAY subscript");
  VELOX_ASSERT_THROW(
      schema->fieldIdPath(Subfield("a[-1].value")), "Invalid ARRAY subscript");
  VELOX_ASSERT_THROW(
      schema->fieldIdPath(Subfield("a[\"key\"].value")),
      "Invalid ARRAY subscript");
  VELOX_ASSERT_THROW(
      schema->fieldIdPath(Subfield("m[1].n")), "Invalid MAP subscript");

  const auto integerMap = std::make_shared<PaimonSchema>(
      1,
      3,
      ROW({{"m", MAP(BIGINT(), BIGINT())}}),
      PaimonTypeMetadata{
          "ROW",
          false,
          std::nullopt,
          {},
          {{"MAP", true, 5, {}, {{"BIGINT", false}, {"BIGINT", true}}}}});
  EXPECT_EQ(
      integerMap->fieldIdPath(Subfield("m[-1]")), (std::vector<int32_t>{5}));
  VELOX_ASSERT_THROW(
      integerMap->fieldIdPath(Subfield("m[\"key\"]")), "Invalid MAP subscript");
  VELOX_ASSERT_THROW(schema->fieldType(-1), "Unknown Paimon field ID");
  VELOX_ASSERT_THROW(schema->fieldMetadata(999), "Unknown Paimon field ID");
  VELOX_ASSERT_THROW(
      schema->fieldIdPath(Subfield("r[1]")), "Invalid Paimon subfield path");
}

TEST_F(PaimonSchemaTest, fieldNamesAndIndicesAreNotIdentities) {
  const PaimonSchema original(
      1,
      3,
      ROW({{"r.x", BIGINT()}, {"r", ROW({{"x", BIGINT()}})}}),
      PaimonTypeMetadata{
          "ROW",
          false,
          std::nullopt,
          {},
          {{"BIGINT", true, 8}, {"ROW", true, 9, {}, {{"BIGINT", true, 10}}}}});
  std::vector<std::unique_ptr<Subfield::PathElement>> literalPath;
  literalPath.push_back(std::make_unique<Subfield::NestedField>("r.x"));
  EXPECT_EQ(
      original.fieldIdPath(Subfield(std::move(literalPath))),
      (std::vector<int32_t>{8}));
  EXPECT_EQ(
      original.fieldIdPath(Subfield("r.x")), (std::vector<int32_t>{9, 10}));

  const PaimonSchema reordered(
      2,
      3,
      ROW({{"renamed", ROW({{"x", BIGINT()}})}, {"r.x", BIGINT()}}),
      PaimonTypeMetadata{
          "ROW",
          false,
          std::nullopt,
          {},
          {{"ROW", true, 9, {}, {{"BIGINT", true, 10}}},
           {"BIGINT", true, 11}}});
  EXPECT_EQ(original.fieldIndexPath(10), (std::vector<column_index_t>{1, 0}));
  EXPECT_EQ(reordered.fieldIndexPath(10), (std::vector<column_index_t>{0, 0}));
  EXPECT_EQ(
      reordered.fieldIdPath(Subfield("renamed.x")),
      (std::vector<int32_t>{9, 10}));
  VELOX_ASSERT_THROW(reordered.fieldType(8), "Unknown Paimon field ID");
}

TEST_F(PaimonSchemaTest, optionsArePreserved) {
  for (int version : {1, 2, 3}) {
    SCOPED_TRACE(version);
    EXPECT_EQ(
        nestedSchema(1, version)->serialize()["version"],
        PaimonSchema::kSerdeVersion);
    EXPECT_EQ(nestedSchema(1, version)->serialize()["schemaVersion"], version);
    EXPECT_TRUE(nestedSchema(1, version)->options().empty());
    EXPECT_TRUE(
        PaimonSchema::create(nestedSchema(1, version)->serialize())
            ->options()
            .empty());
    const std::unordered_map<std::string, std::string> options{
        {"bucket", "-1"},
        {"file.format", "nimble"},
        {"custom", ""},
        {"", "opaque"}};
    const auto schema = nestedSchema(1, version, options);
    EXPECT_EQ(schema->options(), options);
    EXPECT_EQ(PaimonSchema::create(schema->serialize())->options(), options);
  }
}

TEST_F(PaimonSchemaTest, malformedMetadataWire) {
  const auto good = nestedSchema()->serialize();
  for (const auto& key : good.keys()) {
    auto obj = good;
    obj.erase(key);
    EXPECT_THROW(PaimonSchema::create(obj), std::exception) << key;
  }
  for (const auto& key : good["metadata"].keys()) {
    auto obj = good;
    obj["metadata"].erase(key);
    EXPECT_THROW(PaimonSchema::create(obj), std::exception) << key;
  }
  auto obj = good;
  obj["metadata"]["children"][0]["nullable"] = 1;
  VELOX_ASSERT_THROW(
      PaimonSchema::create(obj), "Expected Paimon nullability boolean");
  obj = good;
  obj["metadata"]["children"][1]["parameters"] =
      folly::dynamic::array(int64_t{1} << 32);
  VELOX_ASSERT_THROW(PaimonSchema::create(obj), "Paimon integer out of range");
  obj = good;
  obj["options"]["bucket"] = 1;
  VELOX_ASSERT_THROW(
      PaimonSchema::create(obj), "schema options must be strings");
  obj = good;
  obj["rowType"] = BIGINT()->serialize();
  VELOX_ASSERT_THROW(PaimonSchema::create(obj), "schema must have a ROW type");
}

TEST_F(PaimonSchemaTest, malformedTypeWire) {
  const auto good = nestedSchema()->serialize();
  auto obj = good;
  obj["rowType"] = folly::dynamic::object("name", "Type")("ref", 0);
  VELOX_ASSERT_THROW(PaimonSchema::create(obj), "type must be self-contained");
  obj = good;
  obj["rowType"]["names"] = folly::dynamic::array("id");
  VELOX_ASSERT_THROW(PaimonSchema::create(obj), "Paimon ROW shape mismatch");
  obj = good;
  obj["rowType"]["names"][0] = 0;
  VELOX_ASSERT_THROW(PaimonSchema::create(obj), "Expected Paimon field name");
  obj = good;
  obj["rowType"]["cTypes"][0]["cTypes"] =
      folly::dynamic::array(BIGINT()->serialize());
  VELOX_ASSERT_THROW(
      PaimonSchema::create(obj), "scalar type must not have children");
  obj = good;
  obj["rowType"]["name"] = "DateType";
  VELOX_ASSERT_THROW(
      PaimonSchema::create(obj), "Invalid Paimon type serde name");
  obj = good;
  obj["rowType"]["cTypes"][3]["cTypes"] = folly::dynamic::array;
  VELOX_ASSERT_THROW(PaimonSchema::create(obj), "Paimon type shape mismatch");
  obj = good;
  obj["rowType"]["cTypes"][0]["type"] = "HUGEINT";
  VELOX_ASSERT_THROW(
      PaimonSchema::create(obj), "Unsupported Paimon Velox type");

  for (int64_t precision :
       {int64_t{-1}, int64_t{0}, int64_t{39}, int64_t{276}, int64_t{1} << 32}) {
    SCOPED_TRACE(precision);
    obj = good;
    obj["rowType"]["cTypes"][4]["cTypes"][1]["cTypes"][0]["precision"] =
        precision;
    EXPECT_THROW(PaimonSchema::create(obj), VeloxException);
  }
  for (int scale : {-1, 21, 259}) {
    SCOPED_TRACE(scale);
    obj = good;
    obj["rowType"]["cTypes"][4]["cTypes"][1]["cTypes"][0]["scale"] = scale;
    VELOX_ASSERT_THROW(
        PaimonSchema::create(obj), "Invalid Paimon decimal precision/scale");
  }
}

TEST_F(PaimonSchemaTest, mapKeyAndValueRowIdentity) {
  const auto key = ROW({{"key_id", BIGINT()}});
  const auto value = ROW({{"value_id", BIGINT()}});
  const auto schema = std::make_shared<PaimonSchema>(
      1,
      3,
      ROW({{"m", MAP(key, value)}}),
      PaimonTypeMetadata{
          "ROW",
          false,
          std::nullopt,
          {},
          {{"MAP",
            true,
            5,
            {},
            {{"ROW", true, std::nullopt, {}, {{"BIGINT", false, 6}}},
             {"ROW", false, std::nullopt, {}, {{"BIGINT", true, 7}}}}}}});
  const auto copy = PaimonSchema::create(schema->serialize());
  EXPECT_EQ(copy->fieldIndexPath(6), (std::vector<column_index_t>{0, 0, 0}));
  EXPECT_EQ(copy->fieldIndexPath(7), (std::vector<column_index_t>{0, 1, 0}));
  EXPECT_TRUE(copy->fieldMetadata(5).children[0].nullable);
  EXPECT_FALSE(copy->fieldMetadata(5).children[1].nullable);
  EXPECT_EQ(
      copy->fieldIdPath(Subfield("m[*].value_id")),
      (std::vector<int32_t>{5, 7}));
  auto invalid = schema->serialize();
  invalid["metadata"]["children"][0]["children"][1]["children"][0]["fieldId"] =
      6;
  VELOX_ASSERT_THROW(
      PaimonSchema::create(invalid), "Duplicate Paimon field ID");
}

TEST_F(PaimonSchemaTest, invalidIdentityAndVersions) {
  VELOX_ASSERT_THROW(nestedSchema(-1), "schema ID must be non-negative");
  for (int version : {0, 4, INT32_MAX}) {
    VELOX_ASSERT_THROW(
        nestedSchema(1, version), "Unsupported Paimon schema version");
  }
  for (const folly::dynamic& version :
       {folly::dynamic(0),
        folly::dynamic(2),
        folly::dynamic(int64_t{1} << 32),
        folly::dynamic("1"),
        folly::dynamic(nullptr)}) {
    auto wire = nestedSchema()->serialize();
    wire["version"] = version;
    VELOX_ASSERT_THROW(
        PaimonSchema::create(wire), "Unsupported Paimon schema wire version");
  }
  auto obj = nestedSchema()->serialize();
  obj.erase("schemaVersion");
  EXPECT_THROW(PaimonSchema::create(obj), std::exception);
  obj = nestedSchema()->serialize();
  obj["schemaVersion"] = int64_t{1} << 32;
  VELOX_ASSERT_THROW(PaimonSchema::create(obj), "Paimon integer out of range");
  obj["schemaVersion"] = "3";
  VELOX_ASSERT_THROW(PaimonSchema::create(obj), "Expected a Paimon integer");
  obj = nestedSchema()->serialize();
  obj["schemaId"] = "10";
  VELOX_ASSERT_THROW(PaimonSchema::create(obj), "Expected Paimon schema ID");
  obj = nestedSchema()->serialize();
  obj["metadata"]["children"][0]["fieldId"] = int64_t{1} << 32;
  VELOX_ASSERT_THROW(PaimonSchema::create(obj), "Paimon integer out of range");
  obj["metadata"]["children"][0]["fieldId"] = -1;
  VELOX_ASSERT_THROW(PaimonSchema::create(obj), "Invalid Paimon DataField ID");
  obj["metadata"]["children"][0]["fieldId"] = INT32_MAX / 2;
  VELOX_ASSERT_THROW(PaimonSchema::create(obj), "Invalid Paimon DataField ID");
  obj["metadata"]["children"][0]["fieldId"] = nullptr;
  VELOX_ASSERT_THROW(PaimonSchema::create(obj), "DataField IDs are required");
  obj = nestedSchema()->serialize();
  obj["metadata"]["children"][0]["fieldId"] = INT32_MAX / 2 - 1;
  obj["primaryKeys"] = folly::dynamic::array(INT32_MAX / 2 - 1);
  EXPECT_EQ(
      PaimonSchema::create(obj)->fieldIndexPath(INT32_MAX / 2 - 1),
      (std::vector<column_index_t>{0}));
}

TEST_F(PaimonSchemaTest, invalidFieldAndContainerMetadata) {
  auto obj = nestedSchema()->serialize();
  obj["metadata"]["children"][2]["children"][0]["fieldId"] = 0;
  VELOX_ASSERT_THROW(PaimonSchema::create(obj), "Duplicate Paimon field ID");
  obj = nestedSchema()->serialize();
  obj["metadata"]["children"][3]["children"][0]["fieldId"] = 99;
  VELOX_ASSERT_THROW(
      PaimonSchema::create(obj),
      "DataField IDs are required only for ROW children");
  obj = nestedSchema()->serialize();
  obj["metadata"]["fieldId"] = 100;
  VELOX_ASSERT_THROW(
      PaimonSchema::create(obj),
      "DataField IDs are required only for ROW children");
  obj = nestedSchema()->serialize();
  obj["metadata"]["children"][4]["children"] = folly::dynamic::array;
  VELOX_ASSERT_THROW(PaimonSchema::create(obj), "metadata shape mismatch");

  const PaimonTypeMetadata duplicateNames{
      "ROW",
      false,
      std::nullopt,
      {},
      {{"BIGINT", true, 1}, {"BIGINT", true, 2}}};
  VELOX_ASSERT_THROW(
      duplicateNames.validate(ROW({{"a", BIGINT()}, {"a", BIGINT()}}), false),
      "Duplicate Paimon field name");
  VELOX_ASSERT_THROW(
      duplicateNames.validate(ROW({{"", BIGINT()}, {"b", BIGINT()}}), false),
      "field name must not be empty");
  VELOX_ASSERT_THROW(
      duplicateNames.validate(
          ROW({{" \t\n", BIGINT()}, {"b", BIGINT()}}), false),
      "field name must not be empty or blank");
}

TEST_F(PaimonSchemaTest, invalidTypesAndParameters) {
  const auto check = [](const TypePtr& type,
                        const PaimonTypeMetadata& metadata,
                        const std::string& error) {
    VELOX_ASSERT_THROW(metadata.validate(type, false), error);
  };
  check(nullptr, {"BIGINT", true}, "Missing Paimon field type");
  check(INTEGER(), {"BIGINT", true}, "does not match Velox type");
  check(DATE(), {"INTEGER", true}, "does not match Velox type");
  check(BIGINT(), {"MULTISET", true}, "Unsupported Paimon logical type");
  check(VARCHAR(), {"VARCHAR", true}, "Invalid parameters");
  check(
      VARCHAR(), {"CHAR", true, std::nullopt, {0}}, "length must be positive");
  check(
      VARBINARY(),
      {"BINARY", true, std::nullopt, {-1}},
      "length must be positive");
  check(
      TIMESTAMP(),
      {"TIMESTAMP_WITHOUT_TIME_ZONE", true, std::nullopt, {10}},
      "Invalid Paimon timestamp precision");
  check(
      TIMESTAMP(),
      {"TIMESTAMP_WITH_LOCAL_TIME_ZONE", true, std::nullopt, {-1}},
      "Invalid Paimon timestamp precision");
  check(
      DECIMAL(10, 2),
      {"DECIMAL", true, std::nullopt, {10, 3}},
      "precision/scale do not match");
  check(
      BIGINT(),
      {"DECIMAL", true, std::nullopt, {10, 2}},
      "requires a decimal type");
  check(BIGINT(), {"BIGINT", true, std::nullopt, {10}}, "Invalid parameters");
  check(BIGINT(), {"ROW", true}, "requires a ROW type");
  check(BIGINT(), {"ARRAY", true}, "requires an ARRAY type");
  check(BIGINT(), {"MAP", true}, "requires a MAP type");
}

TEST_F(PaimonSchemaTest, invalidKeyDeclarations) {
  auto obj = nestedSchema()->serialize();
  obj["partitionKeys"] = folly::dynamic::array(1, 1);
  VELOX_ASSERT_THROW(
      PaimonSchema::create(obj), "Duplicate Paimon partition key");
  obj["partitionKeys"] = folly::dynamic::array(21);
  VELOX_ASSERT_THROW(
      PaimonSchema::create(obj), "must reference top-level fields");
  obj["partitionKeys"] = folly::dynamic::array(999);
  VELOX_ASSERT_THROW(PaimonSchema::create(obj), "Unknown Paimon field ID");
  obj = nestedSchema()->serialize();
  obj["primaryKeys"] = folly::dynamic::array(0, 0);
  VELOX_ASSERT_THROW(PaimonSchema::create(obj), "Duplicate Paimon primary key");
  obj["primaryKeys"] = folly::dynamic::array(int64_t{1} << 32);
  VELOX_ASSERT_THROW(PaimonSchema::create(obj), "Paimon integer out of range");
  obj["primaryKeys"] = folly::dynamic::array(1);
  VELOX_ASSERT_THROW(
      PaimonSchema::create(obj),
      "primary keys must include a non-partition field");
  for (const auto* key : {"partitionKeys", "primaryKeys"}) {
    for (int fieldId : {20, 30, 40}) {
      SCOPED_TRACE(key);
      SCOPED_TRACE(fieldId);
      obj = nestedSchema()->serialize();
      obj[key] = folly::dynamic::array(fieldId);
      VELOX_ASSERT_THROW(
          PaimonSchema::create(obj), "must reference scalar fields");
    }
  }
  obj = nestedSchema()->serialize();
  obj["primaryKeys"] = folly::dynamic::array(0);
  const auto copy = PaimonSchema::create(obj);
  EXPECT_EQ(copy->primaryKeys(), (std::vector<int32_t>{0}));
  EXPECT_EQ(copy->partitionKeys(), (std::vector<int32_t>{1}));
  obj["primaryKeys"] = folly::dynamic::array;
  obj["partitionKeys"] = folly::dynamic::array(1, 0);
  EXPECT_EQ(
      PaimonSchema::create(obj)->partitionKeys(), (std::vector<int32_t>{1, 0}));
}

TEST_F(PaimonSchemaTest, nestingLimit) {
  TypePtr type = BIGINT();
  PaimonTypeMetadata metadata{"BIGINT", true};
  for (int i = 0; i < 130; ++i) {
    type = ARRAY(type);
    metadata = PaimonTypeMetadata{
        "ARRAY", true, std::nullopt, {}, {std::move(metadata)}};
  }
  VELOX_ASSERT_THROW(
      metadata.validate(type, false), "Paimon type nesting is too deep");
  VELOX_ASSERT_THROW(
      PaimonTypeMetadata::create(metadata.serialize()),
      "Paimon type nesting is too deep");
  auto obj = nestedSchema()->serialize();
  obj["rowType"] = ROW({{"deep", type}})->serialize();
  VELOX_ASSERT_THROW(
      PaimonSchema::create(obj), "Paimon type nesting is too deep");

  type = BIGINT();
  metadata = PaimonTypeMetadata{"BIGINT", true};
  for (int i = 0; i < 127; ++i) {
    type = ARRAY(type);
    metadata = PaimonTypeMetadata{
        "ARRAY", true, std::nullopt, {}, {std::move(metadata)}};
  }
  metadata.fieldId = 1;
  const PaimonSchema schema(
      1,
      3,
      ROW({{"deep", type}}),
      PaimonTypeMetadata{
          "ROW", false, std::nullopt, {}, {std::move(metadata)}});
  EXPECT_EQ(*PaimonSchema::create(schema.serialize())->fieldType(1), *type);
}

} // namespace
} // namespace facebook::velox::connector::hive::paimon
