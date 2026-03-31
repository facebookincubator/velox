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

#include "velox/connectors/hive/TableHandle.h"

#include <gmock/gmock.h>
#include "gtest/gtest.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::connector::hive;

TEST(FileHandleTest, hiveColumnHandle) {
  Type::registerSerDe();
  connector::hive::HiveColumnHandle::registerSerDe();
  auto columnType = ROW(
      {{"c0c0", BIGINT()},
       {"c0c1",
        ARRAY(MAP(
            VARCHAR(), ROW({{"c0c1c0", BIGINT()}, {"c0c1c1", BIGINT()}})))}});
  auto columnHandle = exec::test::HiveConnectorTestBase::makeColumnHandle(
      "columnHandle", columnType, columnType, {"c0.c0c1[3][\"foo\"].c0c1c0"});
  ASSERT_EQ(columnHandle->name(), "columnHandle");
  ASSERT_EQ(
      columnHandle->columnType(),
      connector::hive::HiveColumnHandle::ColumnType::kRegular);
  ASSERT_EQ(columnHandle->dataType(), columnType);
  ASSERT_EQ(columnHandle->hiveType(), columnType);
  ASSERT_FALSE(columnHandle->isPartitionKey());

  auto str = columnHandle->toString();
  auto obj = columnHandle->serialize();
  auto clone =
      ISerializable::deserialize<connector::hive::HiveColumnHandle>(obj);
  ASSERT_EQ(clone->toString(), str);

  auto incompatibleHiveType = ROW({{"c0c0", BIGINT()}, {"c0c1", BIGINT()}});
  VELOX_ASSERT_THROW(
      exec::test::HiveConnectorTestBase::makeColumnHandle(
          "columnHandle",
          columnType,
          incompatibleHiveType,
          {"c0.c0c1[3][\"foo\"].c0c1c0"}),
      "data type ROW<c0c0:BIGINT,c0c1:ARRAY<MAP<VARCHAR,ROW<c0c1c0:BIGINT,c0c1c1:BIGINT>>>> and hive type ROW<c0c0:BIGINT,c0c1:BIGINT> do not match");
}

TEST(TableHandleTest, hiveTableHandleDbName) {
  connector::hive::HiveTableHandle::registerSerDe();

  // Default dbName is empty.
  auto handleNoDb = std::make_shared<connector::hive::HiveTableHandle>(
      "test-connector",
      "test_table",
      common::SubfieldFilters{},
      /*remainingFilter=*/nullptr);
  ASSERT_TRUE(handleNoDb->dbName().empty());
  ASSERT_EQ(handleNoDb->tableName(), "test_table");

  // Explicit dbName is preserved.
  auto handleWithDb = std::make_shared<connector::hive::HiveTableHandle>(
      "test-connector",
      "test_table",
      common::SubfieldFilters{},
      /*remainingFilter=*/nullptr,
      /*dataColumns=*/nullptr,
      /*indexColumns=*/std::vector<std::string>{},
      /*tableParameters=*/std::unordered_map<std::string, std::string>{},
      /*filterColumnHandles=*/
      std::vector<connector::hive::HiveColumnHandlePtr>{},
      /*sampleRate=*/1.0,
      /*dbName=*/"test_db");
  ASSERT_EQ(handleWithDb->dbName(), "test_db");
  ASSERT_EQ(handleWithDb->tableName(), "test_table");

  // Serialization round-trip preserves dbName.
  auto obj = handleWithDb->serialize();
  auto clone = ISerializable::deserialize<connector::hive::HiveTableHandle>(
      obj, /*context=*/nullptr);
  ASSERT_EQ(clone->dbName(), "test_db");
  ASSERT_EQ(clone->tableName(), "test_table");

  // Round-trip with empty dbName omits the field.
  auto objNoDb = handleNoDb->serialize();
  auto cloneNoDb = ISerializable::deserialize<connector::hive::HiveTableHandle>(
      objNoDb, /*context=*/nullptr);
  ASSERT_TRUE(cloneNoDb->dbName().empty());
}

TEST(TableHandleTest, hiveTableHandleIndexSupport) {
  // Test HiveTableHandle without index columns.
  auto tableHandleWithoutIndex =
      std::make_shared<connector::hive::HiveTableHandle>(
          "test-connector",
          "test_table",
          common::SubfieldFilters{},
          /*remainingFilter=*/nullptr,
          /*dataColumns=*/nullptr,
          /*indexColumns=*/std::vector<std::string>{});

  ASSERT_FALSE(tableHandleWithoutIndex->supportsIndexLookup());
  ASSERT_TRUE(tableHandleWithoutIndex->needsIndexSplit());
  ASSERT_TRUE(tableHandleWithoutIndex->indexColumns().empty());

  // Test HiveTableHandle with index columns.
  auto tableHandleWithIndex =
      std::make_shared<connector::hive::HiveTableHandle>(
          "test-connector",
          "test_table",
          common::SubfieldFilters{},
          /*remainingFilter=*/nullptr,
          /*dataColumns=*/nullptr,
          /*indexColumns=*/std::vector<std::string>{"col1", "col2"});

  ASSERT_TRUE(tableHandleWithIndex->supportsIndexLookup());
  ASSERT_TRUE(tableHandleWithIndex->needsIndexSplit());
  ASSERT_EQ(tableHandleWithIndex->indexColumns().size(), 2);
  ASSERT_EQ(tableHandleWithIndex->indexColumns()[0], "col1");
  ASSERT_EQ(tableHandleWithIndex->indexColumns()[1], "col2");
}

// --- Column extraction pushdown tests ---

using EPE = ExtractionPathElementPtr;

TEST(ExtractionStepTest, nameRoundTrip) {
  // Verify all extraction steps can be converted to names and back.
  std::vector<ExtractionStep> steps = {
      ExtractionStep::kStructField,
      ExtractionStep::kMapKeys,
      ExtractionStep::kMapValues,
      ExtractionStep::kMapKeyFilter,
      ExtractionStep::kArrayElements,
      ExtractionStep::kSize,
  };
  for (auto step : steps) {
    auto name = extractionStepName(step);
    ASSERT_EQ(extractionStepFromName(name), step);
  }
}

TEST(ExtractionStepTest, nameValues) {
  ASSERT_EQ(extractionStepName(ExtractionStep::kStructField), "STRUCT_FIELD");
  ASSERT_EQ(extractionStepName(ExtractionStep::kMapKeys), "MAP_KEYS");
  ASSERT_EQ(extractionStepName(ExtractionStep::kMapValues), "MAP_VALUES");
  ASSERT_EQ(
      extractionStepName(ExtractionStep::kMapKeyFilter), "MAP_KEY_FILTER");
  ASSERT_EQ(
      extractionStepName(ExtractionStep::kArrayElements), "ARRAY_ELEMENTS");
  ASSERT_EQ(extractionStepName(ExtractionStep::kSize), "SIZE");
}

TEST(DeriveExtractionOutputTypeTest, mapKeys) {
  // map_keys(col) where col: MAP(VARCHAR, BIGINT) -> ARRAY(VARCHAR).
  auto hiveType = MAP(VARCHAR(), BIGINT());
  std::vector<EPE> chain = {
      ExtractionPathElement::simple(ExtractionStep::kMapKeys)};
  auto outputType = deriveExtractionOutputType(hiveType, chain);
  ASSERT_TRUE(outputType->equivalent(*ARRAY(VARCHAR())));
}

TEST(DeriveExtractionOutputTypeTest, mapValues) {
  // map_values(col) where col: MAP(VARCHAR, BIGINT) -> ARRAY(BIGINT).
  auto hiveType = MAP(VARCHAR(), BIGINT());
  std::vector<EPE> chain = {
      ExtractionPathElement::simple(ExtractionStep::kMapValues)};
  auto outputType = deriveExtractionOutputType(hiveType, chain);
  ASSERT_TRUE(outputType->equivalent(*ARRAY(BIGINT())));
}

TEST(DeriveExtractionOutputTypeTest, size) {
  // cardinality(col) where col: MAP(VARCHAR, BIGINT) -> BIGINT.
  auto mapType = MAP(VARCHAR(), BIGINT());
  std::vector<EPE> chain = {
      ExtractionPathElement::simple(ExtractionStep::kSize)};
  auto outputType = deriveExtractionOutputType(mapType, chain);
  ASSERT_TRUE(outputType->equivalent(*BIGINT()));

  // cardinality(col) where col: ARRAY(BIGINT) -> BIGINT.
  auto arrayType = ARRAY(BIGINT());
  outputType = deriveExtractionOutputType(arrayType, chain);
  ASSERT_TRUE(outputType->equivalent(*BIGINT()));
}

TEST(DeriveExtractionOutputTypeTest, structFieldMapKeys) {
  // map_keys(col.a.b) where col: ROW(a: ROW(b: MAP(K, V))) -> ARRAY(K).
  auto hiveType = ROW({{"a", ROW({{"b", MAP(VARCHAR(), BIGINT())}})}});
  std::vector<EPE> chain = {
      ExtractionPathElement::structField("a"),
      ExtractionPathElement::structField("b"),
      ExtractionPathElement::simple(ExtractionStep::kMapKeys)};
  auto outputType = deriveExtractionOutputType(hiveType, chain);
  ASSERT_TRUE(outputType->equivalent(*ARRAY(VARCHAR())));
}

TEST(DeriveExtractionOutputTypeTest, structFieldSize) {
  // cardinality(col.features) where col: ROW(features: ARRAY(FLOAT),
  // label: INT) -> BIGINT.
  auto hiveType = ROW({{"features", ARRAY(REAL())}, {"label", INTEGER()}});
  std::vector<EPE> chain = {
      ExtractionPathElement::structField("features"),
      ExtractionPathElement::simple(ExtractionStep::kSize)};
  auto outputType = deriveExtractionOutputType(hiveType, chain);
  ASSERT_TRUE(outputType->equivalent(*BIGINT()));
}

TEST(DeriveExtractionOutputTypeTest, mapValuesArrayElementsMapKeys) {
  // map_keys(map_values(col)) where col: MAP(K1, MAP(K2, V))
  // -> ARRAY(ARRAY(K2)).
  auto hiveType = MAP(VARCHAR(), MAP(INTEGER(), DOUBLE()));
  std::vector<EPE> chain = {
      ExtractionPathElement::simple(ExtractionStep::kMapValues),
      ExtractionPathElement::simple(ExtractionStep::kArrayElements),
      ExtractionPathElement::simple(ExtractionStep::kMapKeys)};
  auto outputType = deriveExtractionOutputType(hiveType, chain);
  ASSERT_TRUE(outputType->equivalent(*ARRAY(ARRAY(INTEGER()))));
}

TEST(DeriveExtractionOutputTypeTest, nestedArrayElements) {
  // map_keys(array_elements(map_values(col)))
  // where col: MAP(K1, ARRAY(MAP(K2, V))) -> ARRAY(ARRAY(ARRAY(K2))).
  auto hiveType = MAP(VARCHAR(), ARRAY(MAP(INTEGER(), DOUBLE())));
  std::vector<EPE> chain = {
      ExtractionPathElement::simple(ExtractionStep::kMapValues),
      ExtractionPathElement::simple(ExtractionStep::kArrayElements),
      ExtractionPathElement::simple(ExtractionStep::kArrayElements),
      ExtractionPathElement::simple(ExtractionStep::kMapKeys)};
  auto outputType = deriveExtractionOutputType(hiveType, chain);
  ASSERT_TRUE(outputType->equivalent(*ARRAY(ARRAY(ARRAY(INTEGER())))));
}

TEST(DeriveExtractionOutputTypeTest, mapValuesStructField) {
  // map_values(col).x where col: MAP(K, ROW(x: INT, y: INT))
  // -> ARRAY(INT).
  auto hiveType = MAP(VARCHAR(), ROW({{"x", INTEGER()}, {"y", INTEGER()}}));
  std::vector<EPE> chain = {
      ExtractionPathElement::simple(ExtractionStep::kMapValues),
      ExtractionPathElement::simple(ExtractionStep::kArrayElements),
      ExtractionPathElement::structField("x")};
  auto outputType = deriveExtractionOutputType(hiveType, chain);
  ASSERT_TRUE(outputType->equivalent(*ARRAY(INTEGER())));
}

TEST(DeriveExtractionOutputTypeTest, mapKeyFilter) {
  // map_subset(col, ARRAY['a', 'b']) where col: MAP(VARCHAR, BIGINT)
  // -> MAP(VARCHAR, BIGINT).
  auto hiveType = MAP(VARCHAR(), BIGINT());
  std::vector<EPE> chain = {
      ExtractionPathElement::mapKeyFilter(std::vector<std::string>{"a", "b"})};
  auto outputType = deriveExtractionOutputType(hiveType, chain);
  ASSERT_TRUE(outputType->equivalent(*MAP(VARCHAR(), BIGINT())));
}

TEST(DeriveExtractionOutputTypeTest, mapKeyFilterMapValuesStructField) {
  // element_at(col, 'foo').x via extraction chain:
  // [MapKeyFilter(["foo"]), MapValues, ArrayElements, StructField("x")]
  // where col: MAP(VARCHAR, ROW(x: INT, y: INT)) -> ARRAY(INT).
  auto hiveType = MAP(VARCHAR(), ROW({{"x", INTEGER()}, {"y", INTEGER()}}));
  std::vector<EPE> chain = {
      ExtractionPathElement::mapKeyFilter(std::vector<std::string>{"foo"}),
      ExtractionPathElement::simple(ExtractionStep::kMapValues),
      ExtractionPathElement::simple(ExtractionStep::kArrayElements),
      ExtractionPathElement::structField("x")};
  auto outputType = deriveExtractionOutputType(hiveType, chain);
  ASSERT_TRUE(outputType->equivalent(*ARRAY(INTEGER())));
}

TEST(DeriveExtractionOutputTypeTest, nestedKeyFilter) {
  // Nested key filter: MAP(K1, MAP(VARCHAR, ROW(x: INT, y: INT)))
  // Chain: [MapValues, AE, MapKeyFilter(["foo"]), MapValues, AE,
  // StructField("x")]
  // -> ARRAY(ARRAY(INT)).
  auto hiveType =
      MAP(VARCHAR(), MAP(VARCHAR(), ROW({{"x", INTEGER()}, {"y", DOUBLE()}})));
  std::vector<EPE> chain = {
      ExtractionPathElement::simple(ExtractionStep::kMapValues),
      ExtractionPathElement::simple(ExtractionStep::kArrayElements),
      ExtractionPathElement::mapKeyFilter(std::vector<std::string>{"foo"}),
      ExtractionPathElement::simple(ExtractionStep::kMapValues),
      ExtractionPathElement::simple(ExtractionStep::kArrayElements),
      ExtractionPathElement::structField("x")};
  auto outputType = deriveExtractionOutputType(hiveType, chain);
  ASSERT_TRUE(outputType->equivalent(*ARRAY(ARRAY(INTEGER()))));
}

TEST(DeriveExtractionOutputTypeTest, emptyChain) {
  // Empty chain means pass-through.
  auto hiveType = MAP(VARCHAR(), BIGINT());
  std::vector<EPE> chain = {};
  auto outputType = deriveExtractionOutputType(hiveType, chain);
  ASSERT_TRUE(outputType->equivalent(*hiveType));
}

TEST(DeriveExtractionOutputTypeTest, errorMissingArrayElementsAfterMapValues) {
  // [MapValues, MapKeys] on MAP(K1, ARRAY(MAP(K2, V))) — missing
  // ArrayElements.
  auto hiveType = MAP(VARCHAR(), ARRAY(MAP(INTEGER(), DOUBLE())));
  std::vector<EPE> chain = {
      ExtractionPathElement::simple(ExtractionStep::kMapValues),
      ExtractionPathElement::simple(ExtractionStep::kMapKeys)};
  VELOX_ASSERT_THROW(
      deriveExtractionOutputType(hiveType, chain),
      "MapKeys requires MAP input, got: ARRAY");
}

TEST(DeriveExtractionOutputTypeTest, errorMissingArrayElementsAfterMapKeys) {
  // [MapKeys, StructField("x")] on MAP(ROW(x: INT, y: INT), V) — missing
  // ArrayElements.
  auto hiveType = MAP(ROW({{"x", INTEGER()}, {"y", INTEGER()}}), BIGINT());
  std::vector<EPE> chain = {
      ExtractionPathElement::simple(ExtractionStep::kMapKeys),
      ExtractionPathElement::structField("x")};
  VELOX_ASSERT_THROW(
      deriveExtractionOutputType(hiveType, chain),
      "StructField requires ROW input, got: ARRAY");
}

TEST(DeriveExtractionOutputTypeTest, errorSizeNotTerminal) {
  // Size must be the last step.
  auto hiveType = MAP(VARCHAR(), BIGINT());
  std::vector<EPE> chain = {
      ExtractionPathElement::simple(ExtractionStep::kSize),
      ExtractionPathElement::simple(ExtractionStep::kMapKeys)};
  VELOX_ASSERT_THROW(
      deriveExtractionOutputType(hiveType, chain),
      "Size must be the last step");
}

TEST(DeriveExtractionOutputTypeTest, errorStructFieldOnMap) {
  auto hiveType = MAP(VARCHAR(), BIGINT());
  std::vector<EPE> chain = {ExtractionPathElement::structField("x")};
  VELOX_ASSERT_THROW(
      deriveExtractionOutputType(hiveType, chain),
      "StructField requires ROW input");
}

TEST(DeriveExtractionOutputTypeTest, errorMapKeysOnArray) {
  auto hiveType = ARRAY(BIGINT());
  std::vector<EPE> chain = {
      ExtractionPathElement::simple(ExtractionStep::kMapKeys)};
  VELOX_ASSERT_THROW(
      deriveExtractionOutputType(hiveType, chain),
      "MapKeys requires MAP input");
}

TEST(DeriveExtractionOutputTypeTest, errorSizeOnScalar) {
  auto hiveType = BIGINT();
  std::vector<EPE> chain = {
      ExtractionPathElement::simple(ExtractionStep::kSize)};
  VELOX_ASSERT_THROW(
      deriveExtractionOutputType(hiveType, chain),
      "Size requires MAP or ARRAY input");
}

TEST(DeriveExtractionOutputTypeTest, errorNonExistentField) {
  auto hiveType = ROW({{"a", INTEGER()}});
  std::vector<EPE> chain = {ExtractionPathElement::structField("nonexistent")};
  VELOX_ASSERT_THROW(
      deriveExtractionOutputType(hiveType, chain),
      "non-existent field: nonexistent");
}

TEST(ColumnExtractionTest, singleExtraction) {
  // Single extraction: map_keys(col) where col: MAP(VARCHAR, BIGINT).
  Type::registerSerDe();
  HiveColumnHandle::registerSerDe();

  auto hiveType = MAP(VARCHAR(), BIGINT());
  auto outputType = ARRAY(VARCHAR());
  std::vector<NamedExtraction> extractions = {
      {"keys",
       {ExtractionPathElement::simple(ExtractionStep::kMapKeys)},
       outputType}};

  auto handle = std::make_shared<HiveColumnHandle>(
      "col",
      HiveColumnHandle::ColumnType::kRegular,
      outputType,
      hiveType,
      std::vector<common::Subfield>{},
      std::move(extractions));

  ASSERT_EQ(handle->name(), "col");
  ASSERT_TRUE(handle->dataType()->equivalent(*ARRAY(VARCHAR())));
  ASSERT_TRUE(handle->hiveType()->equivalent(*MAP(VARCHAR(), BIGINT())));
  ASSERT_TRUE(handle->requiredSubfields().empty());
  ASSERT_EQ(handle->extractions().size(), 1);
  ASSERT_EQ(handle->extractions()[0].outputName, "keys");
  ASSERT_EQ(handle->extractions()[0].chain.size(), 1);
  ASSERT_EQ(
      handle->extractions()[0].chain[0]->step(), ExtractionStep::kMapKeys);
  ASSERT_TRUE(handle->extractions()[0].dataType->equivalent(*ARRAY(VARCHAR())));

  // Serialization round-trip.
  auto obj = handle->serialize();
  auto clone = ISerializable::deserialize<HiveColumnHandle>(obj);
  ASSERT_EQ(clone->toString(), handle->toString());
  ASSERT_EQ(clone->extractions().size(), 1);
  ASSERT_EQ(clone->extractions()[0].outputName, "keys");
  ASSERT_TRUE(clone->extractions()[0].dataType->equivalent(*ARRAY(VARCHAR())));
}

TEST(ColumnExtractionTest, multipleExtractions) {
  // Multiple extractions from the same column:
  // map_keys(col), cardinality(col) where col: MAP(VARCHAR, BIGINT).
  Type::registerSerDe();
  HiveColumnHandle::registerSerDe();

  auto hiveType = MAP(VARCHAR(), BIGINT());
  auto keysType = ARRAY(VARCHAR());
  auto sizeType = BIGINT();
  auto rowOutputType = ROW({{"keys", keysType}, {"size", sizeType}});
  std::vector<NamedExtraction> extractions = {
      {"keys",
       {ExtractionPathElement::simple(ExtractionStep::kMapKeys)},
       keysType},
      {"size",
       {ExtractionPathElement::simple(ExtractionStep::kSize)},
       sizeType}};

  auto handle = std::make_shared<HiveColumnHandle>(
      "col",
      HiveColumnHandle::ColumnType::kRegular,
      rowOutputType,
      hiveType,
      std::vector<common::Subfield>{},
      std::move(extractions));

  ASSERT_EQ(handle->extractions().size(), 2);
  ASSERT_EQ(handle->extractions()[0].outputName, "keys");
  ASSERT_EQ(handle->extractions()[1].outputName, "size");

  // Serialization round-trip.
  auto obj = handle->serialize();
  auto clone = ISerializable::deserialize<HiveColumnHandle>(obj);
  ASSERT_EQ(clone->extractions().size(), 2);
  ASSERT_EQ(clone->extractions()[0].outputName, "keys");
  ASSERT_EQ(clone->extractions()[1].outputName, "size");
  ASSERT_TRUE(clone->extractions()[0].dataType->equivalent(*ARRAY(VARCHAR())));
  ASSERT_TRUE(clone->extractions()[1].dataType->equivalent(*BIGINT()));
}

TEST(ColumnExtractionTest, complexChainWithKeyFilter) {
  // Chain: [MapKeyFilter(["foo", "bar"]), MapValues, AE, StructField("x")]
  // on MAP(VARCHAR, ROW(x: INT, y: INT)) -> ARRAY(INT).
  Type::registerSerDe();
  HiveColumnHandle::registerSerDe();

  auto hiveType = MAP(VARCHAR(), ROW({{"x", INTEGER()}, {"y", INTEGER()}}));
  auto outputType = ARRAY(INTEGER());
  std::vector<NamedExtraction> extractions = {
      {"col_x",
       {ExtractionPathElement::mapKeyFilter(
            std::vector<std::string>{"foo", "bar"}),
        ExtractionPathElement::simple(ExtractionStep::kMapValues),
        ExtractionPathElement::simple(ExtractionStep::kArrayElements),
        ExtractionPathElement::structField("x")},
       outputType}};

  auto handle = std::make_shared<HiveColumnHandle>(
      "col",
      HiveColumnHandle::ColumnType::kRegular,
      outputType,
      hiveType,
      std::vector<common::Subfield>{},
      std::move(extractions));

  ASSERT_EQ(handle->extractions().size(), 1);
  ASSERT_EQ(handle->extractions()[0].chain.size(), 4);

  // Verify key filter keys are preserved.
  ASSERT_THAT(
      static_cast<const StringMapKeyFilterExtractionPathElement&>(
          *handle->extractions()[0].chain[0])
          .filterKeys(),
      testing::ElementsAre("foo", "bar"));

  // Serialization round-trip.
  auto obj = handle->serialize();
  auto clone = ISerializable::deserialize<HiveColumnHandle>(obj);
  ASSERT_EQ(clone->extractions().size(), 1);
  ASSERT_EQ(clone->extractions()[0].chain.size(), 4);
  ASSERT_THAT(
      static_cast<const StringMapKeyFilterExtractionPathElement&>(
          *clone->extractions()[0].chain[0])
          .filterKeys(),
      testing::ElementsAre("foo", "bar"));
}

TEST(ColumnExtractionTest, intFilterKeys) {
  // MapKeyFilter with integer keys.
  Type::registerSerDe();
  HiveColumnHandle::registerSerDe();

  auto hiveType = MAP(BIGINT(), VARCHAR());
  auto outputType = MAP(BIGINT(), VARCHAR());
  std::vector<NamedExtraction> extractions = {
      {"filtered",
       {ExtractionPathElement::mapKeyFilter(std::vector<int64_t>{1, 2, 3})},
       outputType}};

  auto handle = std::make_shared<HiveColumnHandle>(
      "col",
      HiveColumnHandle::ColumnType::kRegular,
      outputType,
      hiveType,
      std::vector<common::Subfield>{},
      std::move(extractions));

  ASSERT_THAT(
      static_cast<const IntMapKeyFilterExtractionPathElement&>(
          *handle->extractions()[0].chain[0])
          .filterKeys(),
      testing::ElementsAre(1, 2, 3));

  // Serialization round-trip.
  auto obj = handle->serialize();
  auto clone = ISerializable::deserialize<HiveColumnHandle>(obj);
  ASSERT_THAT(
      static_cast<const IntMapKeyFilterExtractionPathElement&>(
          *clone->extractions()[0].chain[0])
          .filterKeys(),
      testing::ElementsAre(1, 2, 3));
}

TEST(ColumnExtractionTest, mutualExclusivity) {
  // Cannot set both extractions and requiredSubfields.
  auto hiveType = MAP(VARCHAR(), BIGINT());
  std::vector<common::Subfield> requiredSubfields;
  requiredSubfields.emplace_back("col[\"foo\"]");
  std::vector<NamedExtraction> extractions = {
      {"keys",
       {ExtractionPathElement::simple(ExtractionStep::kMapKeys)},
       ARRAY(VARCHAR())}};

  VELOX_ASSERT_THROW(
      std::make_shared<HiveColumnHandle>(
          "col",
          HiveColumnHandle::ColumnType::kRegular,
          ARRAY(VARCHAR()),
          hiveType,
          std::move(requiredSubfields),
          std::move(extractions)),
      "mutually exclusive");
}

TEST(ColumnExtractionTest, extractionOutputTypeMismatch) {
  // Declared output type doesn't match derived type.
  auto hiveType = MAP(VARCHAR(), BIGINT());
  std::vector<NamedExtraction> extractions = {
      {"keys",
       {ExtractionPathElement::simple(ExtractionStep::kMapKeys)},
       // Wrong: should be ARRAY(VARCHAR), not ARRAY(BIGINT).
       ARRAY(BIGINT())}};

  VELOX_ASSERT_THROW(
      std::make_shared<HiveColumnHandle>(
          "col",
          HiveColumnHandle::ColumnType::kRegular,
          ARRAY(BIGINT()),
          hiveType,
          std::vector<common::Subfield>{},
          std::move(extractions)),
      "does not match derived type");
}

TEST(ColumnExtractionTest, noExtractionsBackwardCompatible) {
  // Empty extractions: existing behavior, dataType must match hiveType.
  auto type = MAP(VARCHAR(), BIGINT());
  auto handle = std::make_shared<HiveColumnHandle>(
      "col", HiveColumnHandle::ColumnType::kRegular, type, type);

  ASSERT_TRUE(handle->extractions().empty());
  ASSERT_TRUE(handle->requiredSubfields().empty());
  ASSERT_TRUE(handle->dataType()->equivalent(*type));
}

TEST(ColumnExtractionTest, threeExtractionsFromSameColumn) {
  // Three extractions: size, keys, and value subfield.
  // col: MAP(BIGINT, ROW(a: VARCHAR, b: DOUBLE, c: INT))
  Type::registerSerDe();
  HiveColumnHandle::registerSerDe();

  auto hiveType =
      MAP(BIGINT(), ROW({{"a", VARCHAR()}, {"b", DOUBLE()}, {"c", INTEGER()}}));
  auto szType = BIGINT();
  auto keysType = ARRAY(BIGINT());
  auto valsAType = ARRAY(VARCHAR());
  auto rowOutputType =
      ROW({{"sz", szType}, {"keys", keysType}, {"vals_a", valsAType}});

  std::vector<NamedExtraction> extractions = {
      {"sz", {ExtractionPathElement::simple(ExtractionStep::kSize)}, szType},
      {"keys",
       {ExtractionPathElement::simple(ExtractionStep::kMapKeys)},
       keysType},
      {"vals_a",
       {ExtractionPathElement::simple(ExtractionStep::kMapValues),
        ExtractionPathElement::simple(ExtractionStep::kArrayElements),
        ExtractionPathElement::structField("a")},
       valsAType}};

  auto handle = std::make_shared<HiveColumnHandle>(
      "col",
      HiveColumnHandle::ColumnType::kRegular,
      rowOutputType,
      hiveType,
      std::vector<common::Subfield>{},
      std::move(extractions));

  ASSERT_EQ(handle->extractions().size(), 3);

  // Serialization round-trip.
  auto obj = handle->serialize();
  auto clone = ISerializable::deserialize<HiveColumnHandle>(obj);
  ASSERT_EQ(clone->extractions().size(), 3);
  ASSERT_EQ(clone->extractions()[0].outputName, "sz");
  ASSERT_EQ(clone->extractions()[1].outputName, "keys");
  ASSERT_EQ(clone->extractions()[2].outputName, "vals_a");
  ASSERT_TRUE(clone->extractions()[0].dataType->equivalent(*BIGINT()));
  ASSERT_TRUE(clone->extractions()[1].dataType->equivalent(*ARRAY(BIGINT())));
  ASSERT_TRUE(clone->extractions()[2].dataType->equivalent(*ARRAY(VARCHAR())));
}
