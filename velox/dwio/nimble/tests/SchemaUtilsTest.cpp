/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/dwio/nimble/common/SchemaUtils.h"

#include <gtest/gtest.h>

#include <folly/Random.h>

#include <memory>
#include "velox/dwio/nimble/common/SchemaBuilder.h"
#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/tests/SchemaUtils.h"
#include "velox/type/Subfield.h"
#include "velox/type/Type.h"

using namespace facebook;
using namespace facebook::nimble;
using Subfield = velox::common::Subfield;

namespace {

// Recursively compares two nimble Type trees for structural equivalence.
// Checks kind, names, children count, scalar kinds, and FlatMap key scalar
// kinds. Ignores stream offsets (they differ between the two overloads).
void expectSameType(const Type& a, const Type& b, const std::string& path) {
  SCOPED_TRACE(path);
  ASSERT_EQ(a.kind(), b.kind());

  switch (a.kind()) {
    case Kind::Scalar:
      EXPECT_EQ(
          a.asScalar().scalarDescriptor().scalarKind(),
          b.asScalar().scalarDescriptor().scalarKind());
      break;
    case Kind::TimestampMicroNano:
      break;
    case Kind::Row: {
      const auto& ra = a.asRow();
      const auto& rb = b.asRow();
      ASSERT_EQ(ra.childrenCount(), rb.childrenCount());
      for (size_t i = 0; i < ra.childrenCount(); ++i) {
        EXPECT_EQ(ra.nameAt(i), rb.nameAt(i));
        expectSameType(
            *ra.childAt(i), *rb.childAt(i), path + "." + ra.nameAt(i));
      }
      break;
    }
    case Kind::Array:
      expectSameType(
          *a.asArray().elements(), *b.asArray().elements(), path + "[]");
      break;
    case Kind::ArrayWithOffsets:
      expectSameType(
          *a.asArrayWithOffsets().elements(),
          *b.asArrayWithOffsets().elements(),
          path + "[]");
      break;
    case Kind::Map:
      expectSameType(*a.asMap().keys(), *b.asMap().keys(), path + ".keys");
      expectSameType(
          *a.asMap().values(), *b.asMap().values(), path + ".values");
      break;
    case Kind::SlidingWindowMap:
      expectSameType(
          *a.asSlidingWindowMap().keys(),
          *b.asSlidingWindowMap().keys(),
          path + ".keys");
      expectSameType(
          *a.asSlidingWindowMap().values(),
          *b.asSlidingWindowMap().values(),
          path + ".values");
      break;
    case Kind::FlatMap: {
      const auto& fa = a.asFlatMap();
      const auto& fb = b.asFlatMap();
      EXPECT_EQ(fa.keyScalarKind(), fb.keyScalarKind());
      ASSERT_EQ(fa.childrenCount(), fb.childrenCount());
      for (size_t i = 0; i < fa.childrenCount(); ++i) {
        EXPECT_EQ(fa.nameAt(i), fb.nameAt(i));
        expectSameType(
            *fa.childAt(i), *fb.childAt(i), path + "[" + fa.nameAt(i) + "]");
      }
      break;
    }
    default:
      FAIL() << "Unexpected kind at " << path;
  }
}

} // namespace

// --- convertToVeloxType tests ---

TEST(SchemaUtilsTest, convertScalarToVelox) {
  struct TestCase {
    ScalarKind scalarKind;
    velox::TypeKind expectedVeloxKind;
  };

  std::vector<TestCase> cases = {
      {ScalarKind::Int8, velox::TypeKind::TINYINT},
      {ScalarKind::Int16, velox::TypeKind::SMALLINT},
      {ScalarKind::Int32, velox::TypeKind::INTEGER},
      {ScalarKind::Int64, velox::TypeKind::BIGINT},
      {ScalarKind::Float, velox::TypeKind::REAL},
      {ScalarKind::Double, velox::TypeKind::DOUBLE},
      {ScalarKind::Bool, velox::TypeKind::BOOLEAN},
      {ScalarKind::String, velox::TypeKind::VARCHAR},
      {ScalarKind::Binary, velox::TypeKind::VARBINARY},
  };

  for (const auto& tc : cases) {
    SchemaBuilder schemaBuilder;
    NIMBLE_SCHEMA(
        schemaBuilder,
        NIMBLE_ROW({{"f", builder.createScalarTypeBuilder(tc.scalarKind)}}));
    auto nimbleType = SchemaReader::getSchema(schemaBuilder.schemaNodes());
    auto& row = nimbleType->asRow();
    auto veloxType = convertToVeloxType(*row.childAt(0));
    EXPECT_EQ(tc.expectedVeloxKind, veloxType->kind())
        << toString(tc.scalarKind);
  }
}

TEST(SchemaUtilsTest, convertUnsupportedScalarToVeloxThrows) {
  std::vector<ScalarKind> unsupported = {
      ScalarKind::UInt8,
      ScalarKind::UInt16,
      ScalarKind::UInt32,
      ScalarKind::UInt64,
      ScalarKind::Undefined,
  };

  for (auto kind : unsupported) {
    SchemaBuilder schemaBuilder;
    NIMBLE_SCHEMA(
        schemaBuilder,
        NIMBLE_ROW({{"f", builder.createScalarTypeBuilder(kind)}}));
    auto nimbleType = SchemaReader::getSchema(schemaBuilder.schemaNodes());
    auto& row = nimbleType->asRow();
    EXPECT_THROW(convertToVeloxType(*row.childAt(0)), NimbleUserError)
        << toString(kind);
  }
}

TEST(SchemaUtilsTest, convertTimestampMicroNanoToVelox) {
  SchemaBuilder schemaBuilder;
  NIMBLE_SCHEMA(
      schemaBuilder, NIMBLE_ROW({{"ts", NIMBLE_TIMESTAMPMICRONANO()}}));
  auto nimbleType = SchemaReader::getSchema(schemaBuilder.schemaNodes());
  auto& row = nimbleType->asRow();
  auto veloxType = convertToVeloxType(*row.childAt(0));
  EXPECT_EQ(velox::TypeKind::TIMESTAMP, veloxType->kind());
}

TEST(SchemaUtilsTest, convertRowToVelox) {
  SchemaBuilder schemaBuilder;
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW({{"a", NIMBLE_TINYINT()}, {"b", NIMBLE_BIGINT()}}));
  auto nimbleType = SchemaReader::getSchema(schemaBuilder.schemaNodes());
  auto veloxType = convertToVeloxType(*nimbleType);
  EXPECT_EQ(velox::TypeKind::ROW, veloxType->kind());
  auto& veloxRow = veloxType->asRow();
  ASSERT_EQ(2, veloxRow.size());
  EXPECT_EQ("a", veloxRow.nameOf(0));
  EXPECT_EQ("b", veloxRow.nameOf(1));
  EXPECT_EQ(velox::TypeKind::TINYINT, veloxRow.childAt(0)->kind());
  EXPECT_EQ(velox::TypeKind::BIGINT, veloxRow.childAt(1)->kind());
}

TEST(SchemaUtilsTest, convertArrayToVelox) {
  SchemaBuilder schemaBuilder;
  NIMBLE_SCHEMA(
      schemaBuilder, NIMBLE_ROW({{"arr", NIMBLE_ARRAY(NIMBLE_BIGINT())}}));
  auto nimbleType = SchemaReader::getSchema(schemaBuilder.schemaNodes());
  auto& row = nimbleType->asRow();
  auto veloxType = convertToVeloxType(*row.childAt(0));
  EXPECT_EQ(velox::TypeKind::ARRAY, veloxType->kind());
  EXPECT_EQ(
      velox::TypeKind::BIGINT, veloxType->asArray().elementType()->kind());
}

TEST(SchemaUtilsTest, convertArrayWithOffsetsToVelox) {
  SchemaBuilder schemaBuilder;
  NIMBLE_SCHEMA(
      schemaBuilder, NIMBLE_ROW({{"oa", NIMBLE_OFFSETARRAY(NIMBLE_BIGINT())}}));
  auto nimbleType = SchemaReader::getSchema(schemaBuilder.schemaNodes());
  auto& row = nimbleType->asRow();
  auto veloxType = convertToVeloxType(*row.childAt(0));
  EXPECT_EQ(velox::TypeKind::ARRAY, veloxType->kind());
  EXPECT_EQ(
      velox::TypeKind::BIGINT, veloxType->asArray().elementType()->kind());
}

TEST(SchemaUtilsTest, convertMapToVelox) {
  SchemaBuilder schemaBuilder;
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW({{"m", NIMBLE_MAP(NIMBLE_STRING(), NIMBLE_INTEGER())}}));
  auto nimbleType = SchemaReader::getSchema(schemaBuilder.schemaNodes());
  auto& row = nimbleType->asRow();
  auto veloxType = convertToVeloxType(*row.childAt(0));
  EXPECT_EQ(velox::TypeKind::MAP, veloxType->kind());
  EXPECT_EQ(velox::TypeKind::VARCHAR, veloxType->asMap().keyType()->kind());
  EXPECT_EQ(velox::TypeKind::INTEGER, veloxType->asMap().valueType()->kind());
}

TEST(SchemaUtilsTest, convertSlidingWindowMapToVelox) {
  SchemaBuilder schemaBuilder;
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW(
          {{"swm",
            NIMBLE_SLIDINGWINDOWMAP(NIMBLE_STRING(), NIMBLE_INTEGER())}}));
  auto nimbleType = SchemaReader::getSchema(schemaBuilder.schemaNodes());
  auto& row = nimbleType->asRow();
  auto veloxType = convertToVeloxType(*row.childAt(0));
  EXPECT_EQ(velox::TypeKind::MAP, veloxType->kind());
  EXPECT_EQ(velox::TypeKind::VARCHAR, veloxType->asMap().keyType()->kind());
  EXPECT_EQ(velox::TypeKind::INTEGER, veloxType->asMap().valueType()->kind());
}

TEST(SchemaUtilsTest, convertFlatMapToVelox) {
  SchemaBuilder schemaBuilder;
  test::FlatMapChildAdder adder;
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW({{"fm", NIMBLE_FLATMAP(String, NIMBLE_INTEGER(), adder)}}));
  adder.addChild("key1");
  auto nimbleType = SchemaReader::getSchema(schemaBuilder.schemaNodes());
  auto& row = nimbleType->asRow();
  auto veloxType = convertToVeloxType(*row.childAt(0));
  EXPECT_EQ(velox::TypeKind::MAP, veloxType->kind());
  EXPECT_EQ(velox::TypeKind::VARCHAR, veloxType->asMap().keyType()->kind());
  EXPECT_EQ(velox::TypeKind::INTEGER, veloxType->asMap().valueType()->kind());
}

// --- convertToNimbleType tests ---

TEST(SchemaUtilsTest, convertVeloxScalarToNimble) {
  struct TestCase {
    velox::TypePtr veloxType;
    ScalarKind expectedScalarKind;
  };

  std::vector<TestCase> cases = {
      {velox::BOOLEAN(), ScalarKind::Bool},
      {velox::TINYINT(), ScalarKind::Int8},
      {velox::SMALLINT(), ScalarKind::Int16},
      {velox::INTEGER(), ScalarKind::Int32},
      {velox::BIGINT(), ScalarKind::Int64},
      {velox::REAL(), ScalarKind::Float},
      {velox::DOUBLE(), ScalarKind::Double},
      {velox::VARCHAR(), ScalarKind::String},
      {velox::VARBINARY(), ScalarKind::Binary},
  };

  for (const auto& tc : cases) {
    auto nimbleType = convertToNimbleType(*tc.veloxType);
    ASSERT_EQ(Kind::Scalar, nimbleType->kind()) << tc.veloxType->toString();
    EXPECT_EQ(
        tc.expectedScalarKind,
        nimbleType->asScalar().scalarDescriptor().scalarKind())
        << tc.veloxType->toString();
  }
}

TEST(SchemaUtilsTest, convertVeloxTimestampToNimble) {
  auto nimbleType = convertToNimbleType(*velox::TIMESTAMP());
  ASSERT_EQ(Kind::TimestampMicroNano, nimbleType->kind());
}

TEST(SchemaUtilsTest, convertVeloxArrayToNimble) {
  auto vType = velox::ARRAY(velox::BIGINT());
  auto nimbleType = convertToNimbleType(*vType);
  ASSERT_EQ(Kind::Array, nimbleType->kind());
  auto& arr = nimbleType->asArray();
  EXPECT_EQ(Kind::Scalar, arr.elements()->kind());
  EXPECT_EQ(
      ScalarKind::Int64,
      arr.elements()->asScalar().scalarDescriptor().scalarKind());
}

TEST(SchemaUtilsTest, convertVeloxMapToNimble) {
  auto vType = velox::MAP(velox::VARCHAR(), velox::INTEGER());
  auto nimbleType = convertToNimbleType(*vType);
  ASSERT_EQ(Kind::Map, nimbleType->kind());
  auto& map = nimbleType->asMap();
  EXPECT_EQ(
      ScalarKind::String,
      map.keys()->asScalar().scalarDescriptor().scalarKind());
  EXPECT_EQ(
      ScalarKind::Int32,
      map.values()->asScalar().scalarDescriptor().scalarKind());
}

TEST(SchemaUtilsTest, convertVeloxRowToNimble) {
  auto vType = velox::ROW({"x", "y"}, {velox::TINYINT(), velox::DOUBLE()});
  auto nimbleType = convertToNimbleType(*vType);
  ASSERT_EQ(Kind::Row, nimbleType->kind());
  auto& row = nimbleType->asRow();
  ASSERT_EQ(2, row.childrenCount());
  EXPECT_EQ("x", row.nameAt(0));
  EXPECT_EQ("y", row.nameAt(1));
  EXPECT_EQ(
      ScalarKind::Int8,
      row.childAt(0)->asScalar().scalarDescriptor().scalarKind());
  EXPECT_EQ(
      ScalarKind::Double,
      row.childAt(1)->asScalar().scalarDescriptor().scalarKind());
}

// --- Round-trip tests ---

TEST(SchemaUtilsTest, roundTripScalarTypes) {
  std::vector<velox::TypePtr> scalarTypes = {
      velox::BOOLEAN(),
      velox::TINYINT(),
      velox::SMALLINT(),
      velox::INTEGER(),
      velox::BIGINT(),
      velox::REAL(),
      velox::DOUBLE(),
      velox::VARCHAR(),
      velox::VARBINARY(),
  };

  for (const auto& vType : scalarTypes) {
    auto nimbleType = convertToNimbleType(*vType);
    auto roundTripped = convertToVeloxType(*nimbleType);
    EXPECT_TRUE(vType->equivalent(*roundTripped))
        << "Failed round-trip for " << vType->toString() << " got "
        << roundTripped->toString();
  }
}

TEST(SchemaUtilsTest, roundTripTimestamp) {
  auto vType = velox::TIMESTAMP();
  auto nimbleType = convertToNimbleType(*vType);
  auto roundTripped = convertToVeloxType(*nimbleType);
  EXPECT_TRUE(vType->equivalent(*roundTripped));
}

TEST(SchemaUtilsTest, roundTripArray) {
  auto vType = velox::ARRAY(velox::INTEGER());
  auto nimbleType = convertToNimbleType(*vType);
  auto roundTripped = convertToVeloxType(*nimbleType);
  EXPECT_TRUE(vType->equivalent(*roundTripped)) << roundTripped->toString();
}

TEST(SchemaUtilsTest, roundTripMap) {
  auto vType = velox::MAP(velox::VARCHAR(), velox::BIGINT());
  auto nimbleType = convertToNimbleType(*vType);
  auto roundTripped = convertToVeloxType(*nimbleType);
  EXPECT_TRUE(vType->equivalent(*roundTripped)) << roundTripped->toString();
}

TEST(SchemaUtilsTest, roundTripRow) {
  auto vType = velox::ROW({"a", "b"}, {velox::TINYINT(), velox::DOUBLE()});
  auto nimbleType = convertToNimbleType(*vType);
  auto roundTripped = convertToVeloxType(*nimbleType);
  EXPECT_TRUE(vType->equivalent(*roundTripped)) << roundTripped->toString();
}

// --- convertToNimbleType with projected subfields tests ---

TEST(SchemaUtilsTest, projectionScalarOnly) {
  auto veloxType = velox::ROW(
      {"col_a", "col_b", "col_c"},
      {velox::INTEGER(), velox::BIGINT(), velox::VARCHAR()});

  std::vector<Subfield> subfields;
  subfields.emplace_back("col_a");
  subfields.emplace_back("col_b");
  subfields.emplace_back("col_c");

  auto projectedNimbleTypeFromVelox =
      buildProjectedNimbleType(*veloxType, subfields);
  ASSERT_EQ(Kind::Row, projectedNimbleTypeFromVelox->kind());
  const auto& row = projectedNimbleTypeFromVelox->asRow();
  ASSERT_EQ(3, row.childrenCount());
  EXPECT_EQ("col_a", row.nameAt(0));
  EXPECT_EQ("col_b", row.nameAt(1));
  EXPECT_EQ("col_c", row.nameAt(2));

  EXPECT_EQ(Kind::Scalar, row.childAt(0)->kind());
  EXPECT_EQ(
      ScalarKind::Int32,
      row.childAt(0)->asScalar().scalarDescriptor().scalarKind());
  EXPECT_EQ(Kind::Scalar, row.childAt(1)->kind());
  EXPECT_EQ(
      ScalarKind::Int64,
      row.childAt(1)->asScalar().scalarDescriptor().scalarKind());
  EXPECT_EQ(Kind::Scalar, row.childAt(2)->kind());
  EXPECT_EQ(
      ScalarKind::String,
      row.childAt(2)->asScalar().scalarDescriptor().scalarKind());
}

TEST(SchemaUtilsTest, projectionFlatMap) {
  // Velox type has MAP columns (FlatMap in file).
  auto veloxType = velox::ROW(
      {"user_id", "int_traits_map", "long_traits_map"},
      {velox::BIGINT(),
       velox::MAP(velox::INTEGER(), velox::ARRAY(velox::INTEGER())),
       velox::MAP(velox::INTEGER(), velox::ARRAY(velox::BIGINT()))});

  std::vector<Subfield> subfields;
  subfields.emplace_back("user_id");
  subfields.emplace_back("int_traits_map[2]");
  subfields.emplace_back("int_traits_map[387]");
  subfields.emplace_back("long_traits_map[10]");

  ColumnEncodings encodings;
  encodings.flatMapColumns.insert("int_traits_map");
  encodings.flatMapColumns.insert("long_traits_map");

  auto projectedNimbleTypeFromVelox =
      buildProjectedNimbleType(*veloxType, subfields, encodings);
  ASSERT_EQ(Kind::Row, projectedNimbleTypeFromVelox->kind());
  const auto& row = projectedNimbleTypeFromVelox->asRow();
  ASSERT_EQ(3, row.childrenCount());

  // user_id: scalar.
  EXPECT_EQ("user_id", row.nameAt(0));
  EXPECT_EQ(Kind::Scalar, row.childAt(0)->kind());
  EXPECT_EQ(
      ScalarKind::Int64,
      row.childAt(0)->asScalar().scalarDescriptor().scalarKind());

  // int_traits_map: FlatMap with 2 keys.
  EXPECT_EQ("int_traits_map", row.nameAt(1));
  ASSERT_EQ(Kind::FlatMap, row.childAt(1)->kind());
  const auto& intFlatMap = row.childAt(1)->asFlatMap();
  EXPECT_EQ(ScalarKind::Int32, intFlatMap.keyScalarKind());
  ASSERT_EQ(2, intFlatMap.childrenCount());
  EXPECT_EQ("2", intFlatMap.nameAt(0));
  EXPECT_EQ("387", intFlatMap.nameAt(1));
  // Value type: Array<Int32>.
  ASSERT_EQ(Kind::Array, intFlatMap.childAt(0)->kind());
  EXPECT_EQ(
      ScalarKind::Int32,
      intFlatMap.childAt(0)
          ->asArray()
          .elements()
          ->asScalar()
          .scalarDescriptor()
          .scalarKind());

  // long_traits_map: FlatMap with 1 key.
  EXPECT_EQ("long_traits_map", row.nameAt(2));
  ASSERT_EQ(Kind::FlatMap, row.childAt(2)->kind());
  const auto& longFlatMap = row.childAt(2)->asFlatMap();
  EXPECT_EQ(ScalarKind::Int32, longFlatMap.keyScalarKind());
  ASSERT_EQ(1, longFlatMap.childrenCount());
  EXPECT_EQ("10", longFlatMap.nameAt(0));
  // Value type: Array<Int64>.
  ASSERT_EQ(Kind::Array, longFlatMap.childAt(0)->kind());
  EXPECT_EQ(
      ScalarKind::Int64,
      longFlatMap.childAt(0)
          ->asArray()
          .elements()
          ->asScalar()
          .scalarDescriptor()
          .scalarKind());
}

TEST(SchemaUtilsTest, projectionDictionaryArray) {
  // dictionaryArrayColumns: ARRAY columns become ArrayWithOffsets.
  auto veloxType = velox::ROW(
      {"id", "tags", "scores"},
      {velox::BIGINT(),
       velox::ARRAY(velox::VARCHAR()),
       velox::ARRAY(velox::INTEGER())});

  std::vector<Subfield> subfields;
  subfields.emplace_back("id");
  subfields.emplace_back("tags");
  subfields.emplace_back("scores");

  ColumnEncodings encodings;
  encodings.dictionaryArrayColumns.insert("tags");

  auto projectedNimbleTypeFromVelox =
      buildProjectedNimbleType(*veloxType, subfields, encodings);
  ASSERT_EQ(Kind::Row, projectedNimbleTypeFromVelox->kind());
  const auto& row = projectedNimbleTypeFromVelox->asRow();
  ASSERT_EQ(3, row.childrenCount());

  // id: scalar.
  EXPECT_EQ("id", row.nameAt(0));
  EXPECT_EQ(Kind::Scalar, row.childAt(0)->kind());

  // tags: ArrayWithOffsets (dictionaryArray encoding).
  EXPECT_EQ("tags", row.nameAt(1));
  ASSERT_EQ(Kind::ArrayWithOffsets, row.childAt(1)->kind());
  const auto& tagsArray = row.childAt(1)->asArrayWithOffsets();
  EXPECT_EQ(Kind::Scalar, tagsArray.elements()->kind());
  EXPECT_EQ(
      ScalarKind::String,
      tagsArray.elements()->asScalar().scalarDescriptor().scalarKind());

  // scores: plain Array (not in dictionaryArrayColumns).
  EXPECT_EQ("scores", row.nameAt(2));
  ASSERT_EQ(Kind::Array, row.childAt(2)->kind());
  const auto& scoresArray = row.childAt(2)->asArray();
  EXPECT_EQ(Kind::Scalar, scoresArray.elements()->kind());
  EXPECT_EQ(
      ScalarKind::Int32,
      scoresArray.elements()->asScalar().scalarDescriptor().scalarKind());
}

TEST(SchemaUtilsTest, projectionDeduplicatedMap) {
  // deduplicatedMapColumns: MAP columns become SlidingWindowMap.
  auto veloxType = velox::ROW(
      {"id", "dedup_map", "regular_map"},
      {velox::BIGINT(),
       velox::MAP(velox::VARCHAR(), velox::INTEGER()),
       velox::MAP(velox::VARCHAR(), velox::BIGINT())});

  std::vector<Subfield> subfields;
  subfields.emplace_back("id");
  subfields.emplace_back("dedup_map");
  subfields.emplace_back("regular_map");

  ColumnEncodings encodings;
  encodings.deduplicatedMapColumns.insert("dedup_map");

  auto projectedNimbleTypeFromVelox =
      buildProjectedNimbleType(*veloxType, subfields, encodings);
  ASSERT_EQ(Kind::Row, projectedNimbleTypeFromVelox->kind());
  const auto& row = projectedNimbleTypeFromVelox->asRow();
  ASSERT_EQ(3, row.childrenCount());

  // id: scalar.
  EXPECT_EQ("id", row.nameAt(0));
  EXPECT_EQ(Kind::Scalar, row.childAt(0)->kind());

  // dedup_map: SlidingWindowMap (deduplicatedMap encoding).
  EXPECT_EQ("dedup_map", row.nameAt(1));
  ASSERT_EQ(Kind::SlidingWindowMap, row.childAt(1)->kind());
  const auto& dedupMap = row.childAt(1)->asSlidingWindowMap();
  EXPECT_EQ(Kind::Scalar, dedupMap.keys()->kind());
  EXPECT_EQ(
      ScalarKind::String,
      dedupMap.keys()->asScalar().scalarDescriptor().scalarKind());
  EXPECT_EQ(Kind::Scalar, dedupMap.values()->kind());
  EXPECT_EQ(
      ScalarKind::Int32,
      dedupMap.values()->asScalar().scalarDescriptor().scalarKind());

  // regular_map: plain Map (not in deduplicatedMapColumns).
  EXPECT_EQ("regular_map", row.nameAt(2));
  ASSERT_EQ(Kind::Map, row.childAt(2)->kind());
  const auto& regularMap = row.childAt(2)->asMap();
  EXPECT_EQ(
      ScalarKind::String,
      regularMap.keys()->asScalar().scalarDescriptor().scalarKind());
  EXPECT_EQ(
      ScalarKind::Int64,
      regularMap.values()->asScalar().scalarDescriptor().scalarKind());
}

TEST(SchemaUtilsTest, projectionMixedEncodings) {
  // All three encoding types together.
  auto veloxType = velox::ROW(
      {"id", "dict_arr", "dedup_map", "flat_map"},
      {velox::BIGINT(),
       velox::ARRAY(velox::VARCHAR()),
       velox::MAP(velox::VARCHAR(), velox::INTEGER()),
       velox::MAP(velox::INTEGER(), velox::DOUBLE())});

  std::vector<Subfield> subfields;
  subfields.emplace_back("id");
  subfields.emplace_back("dict_arr");
  subfields.emplace_back("dedup_map");
  subfields.emplace_back("flat_map[1]");
  subfields.emplace_back("flat_map[5]");

  ColumnEncodings encodings;
  encodings.dictionaryArrayColumns.insert("dict_arr");
  encodings.deduplicatedMapColumns.insert("dedup_map");
  encodings.flatMapColumns.insert("flat_map");

  auto projectedNimbleTypeFromVelox =
      buildProjectedNimbleType(*veloxType, subfields, encodings);
  ASSERT_EQ(Kind::Row, projectedNimbleTypeFromVelox->kind());
  const auto& row = projectedNimbleTypeFromVelox->asRow();
  ASSERT_EQ(4, row.childrenCount());

  // id: scalar.
  EXPECT_EQ("id", row.nameAt(0));
  EXPECT_EQ(Kind::Scalar, row.childAt(0)->kind());

  // dict_arr: ArrayWithOffsets.
  EXPECT_EQ("dict_arr", row.nameAt(1));
  ASSERT_EQ(Kind::ArrayWithOffsets, row.childAt(1)->kind());

  // dedup_map: SlidingWindowMap.
  EXPECT_EQ("dedup_map", row.nameAt(2));
  ASSERT_EQ(Kind::SlidingWindowMap, row.childAt(2)->kind());

  // flat_map: FlatMap with 2 keys.
  EXPECT_EQ("flat_map", row.nameAt(3));
  ASSERT_EQ(Kind::FlatMap, row.childAt(3)->kind());
  const auto& flatMap = row.childAt(3)->asFlatMap();
  EXPECT_EQ(ScalarKind::Int32, flatMap.keyScalarKind());
  ASSERT_EQ(2, flatMap.childrenCount());
  EXPECT_EQ("1", flatMap.nameAt(0));
  EXPECT_EQ("5", flatMap.nameAt(1));
}

TEST(SchemaUtilsTest, fullFlatMapProjectionFails) {
  auto veloxType = velox::ROW(
      {"user_id", "traits_map"},
      {velox::BIGINT(),
       velox::MAP(velox::INTEGER(), velox::ARRAY(velox::INTEGER()))});

  // Projecting entire FlatMap column without key subscripts should fail.
  std::vector<Subfield> subfields;
  subfields.emplace_back("user_id");
  subfields.emplace_back("traits_map");

  ColumnEncodings encodings;
  encodings.flatMapColumns.insert("traits_map");

  NIMBLE_ASSERT_THROW(
      buildProjectedNimbleType(*veloxType, subfields, encodings),
      "Cannot project entire FlatMap column without key subscripts");
}

TEST(SchemaUtilsTest, projectionOnlyProjectedColumns) {
  // Verify that unprojected columns are excluded from the output.
  auto veloxType = velox::ROW(
      {"col_a", "col_b", "col_c"},
      {velox::INTEGER(), velox::BIGINT(), velox::VARCHAR()});

  std::vector<Subfield> subfields;
  subfields.emplace_back("col_a");
  subfields.emplace_back("col_c");

  auto projectedNimbleTypeFromVelox =
      buildProjectedNimbleType(*veloxType, subfields);
  ASSERT_EQ(Kind::Row, projectedNimbleTypeFromVelox->kind());
  const auto& row = projectedNimbleTypeFromVelox->asRow();
  ASSERT_EQ(2, row.childrenCount());
  EXPECT_EQ("col_a", row.nameAt(0));
  EXPECT_EQ("col_c", row.nameAt(1));
}

TEST(SchemaUtilsTest, projectionDeep) {
  auto veloxType = velox::ROW(
      {"map_col"},
      {velox::MAP(
          velox::VARCHAR(), velox::ROW({"inner"}, {velox::INTEGER()}))});

  // "map_col["key"].inner" — depth 3, projects FlatMap key "key" with only
  // the "inner" field of the value struct.
  std::vector<std::unique_ptr<velox::common::Subfield::PathElement>> path;
  path.push_back(std::make_unique<Subfield::NestedField>("map_col"));
  path.push_back(std::make_unique<Subfield::StringSubscript>("key"));
  path.push_back(std::make_unique<Subfield::NestedField>("inner"));

  std::vector<Subfield> subfields;
  subfields.emplace_back(std::move(path));

  ColumnEncodings encodings;
  encodings.flatMapColumns.insert("map_col");

  auto projectedNimbleTypeFromVelox =
      buildProjectedNimbleType(*veloxType, subfields, encodings);
  ASSERT_EQ(projectedNimbleTypeFromVelox->kind(), Kind::Row);
  const auto& root = projectedNimbleTypeFromVelox->asRow();
  ASSERT_EQ(root.childrenCount(), 1);
  EXPECT_EQ(root.nameAt(0), "map_col");

  // map_col should be a FlatMap with one key "key".
  ASSERT_EQ(root.childAt(0)->kind(), Kind::FlatMap);
  const auto& flatMap = root.childAt(0)->asFlatMap();
  ASSERT_EQ(flatMap.childrenCount(), 1);
  EXPECT_EQ(flatMap.nameAt(0), "key");

  // The value type should be a Row with only the "inner" field.
  ASSERT_EQ(flatMap.childAt(0)->kind(), Kind::Row);
  const auto& valueRow = flatMap.childAt(0)->asRow();
  ASSERT_EQ(valueRow.childrenCount(), 1);
  EXPECT_EQ(valueRow.nameAt(0), "inner");
  EXPECT_EQ(valueRow.childAt(0)->kind(), Kind::Scalar);
}

TEST(SchemaUtilsTest, projectionDeepWithStreamOffsets) {
  // Build nimble schema: root Row with a FlatMap column whose values are
  // Row(inner_a: INT, inner_b: STRING).
  //
  // Schema tree (stream offsets assigned by SchemaBuilder in DFS order):
  //   Row nulls [0]
  //     FlatMap nulls [1]
  //       key1: Row nulls [2]
  //         inner_a: INT [3]
  //         inner_b: STRING [4]
  //       key1 inMap: BOOL [5]
  //       key2: Row nulls [6]
  //         inner_a: INT [7]
  //         inner_b: STRING [8]
  //       key2 inMap: BOOL [9]
  SchemaBuilder schemaBuilder;
  test::FlatMapChildAdder childAdder;
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW({
          {"map_col",
           NIMBLE_FLATMAP(
               String,
               NIMBLE_ROW({
                   {"inner_a", NIMBLE_INTEGER()},
                   {"inner_b", NIMBLE_STRING()},
               }),
               childAdder)},
      }));
  childAdder.addChild("key1");
  childAdder.addChild("key2");
  auto convertedNimbleType =
      SchemaReader::getSchema(schemaBuilder.schemaNodes());

  // Project "map_col["key1"].inner_b" — depth 3: FlatMap key + struct field.
  std::vector<std::unique_ptr<velox::common::Subfield::PathElement>> path;
  path.push_back(std::make_unique<Subfield::NestedField>("map_col"));
  path.push_back(std::make_unique<Subfield::StringSubscript>("key1"));
  path.push_back(std::make_unique<Subfield::NestedField>("inner_b"));

  std::vector<Subfield> subfields;
  subfields.emplace_back(std::move(path));

  // Drive the projection through the same nimble-source API the projectors
  // use: derive projected schema metadata in one pass.
  auto projection =
      buildProjectedNimbleType(convertedNimbleType.get(), subfields);

  // Verify projected schema structure.
  ASSERT_EQ(projection.nimbleType->kind(), Kind::Row);
  const auto& root = projection.nimbleType->asRow();
  ASSERT_EQ(root.childrenCount(), 1);
  EXPECT_EQ(root.nameAt(0), "map_col");

  ASSERT_EQ(root.childAt(0)->kind(), Kind::FlatMap);
  const auto& flatMap = root.childAt(0)->asFlatMap();
  ASSERT_EQ(flatMap.childrenCount(), 1);
  EXPECT_EQ(flatMap.nameAt(0), "key1");

  // Value should be a Row with only "inner_b" (not "inner_a").
  ASSERT_EQ(flatMap.childAt(0)->kind(), Kind::Row);
  const auto& valueRow = flatMap.childAt(0)->asRow();
  ASSERT_EQ(valueRow.childrenCount(), 1);
  EXPECT_EQ(valueRow.nameAt(0), "inner_b");
  EXPECT_EQ(valueRow.childAt(0)->kind(), Kind::Scalar);

  // Verify projected stream offsets enumerate source positions in DFS +
  // FlatMap-alphabetical order. Source offsets here reflect the order the
  // SchemaBuilder allocates them (FlatMap nulls first because the
  // FlatMapTypeBuilder is constructed before the outer Row, value subtrees
  // for each key are built lazily inside addChild):
  //   FlatMap.nulls = 0, outer Row.nulls = 1,
  //   key1.inner_a = 2, key1.inner_b = 3, key1.Row.nulls = 4, key1.inMap = 5,
  //   key2.inner_a = 6, key2.inner_b = 7, key2.Row.nulls = 8, key2.inMap = 9.
  // The walker visits: outer-Row.nulls (1), map_col-FlatMap.nulls (0),
  // key1.Row.nulls (4), key1.inner_b (3), key1.inMap (5).
  EXPECT_EQ(projection.streamOffsets, std::vector<uint32_t>({1, 0, 4, 3, 5}));
  EXPECT_EQ(
      projection.rowOrFlatMapNullStreams,
      std::vector<bool>({true, true, true, false, false}));
}

TEST(SchemaUtilsTest, projectionMarksRowOrFlatMapNullStreams) {
  SchemaBuilder schemaBuilder;
  test::FlatMapChildAdder featuresAdder;
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW({
          {"id", NIMBLE_BIGINT()},
          {"profile",
           NIMBLE_ROW({
               {"age", NIMBLE_INTEGER()},
               {"name", NIMBLE_STRING()},
           })},
          {"features",
           NIMBLE_FLATMAP(
               String,
               NIMBLE_ROW({
                   {"score", NIMBLE_INTEGER()},
                   {"label", NIMBLE_STRING()},
               }),
               featuresAdder)},
      }));
  featuresAdder.addChild("a");
  featuresAdder.addChild("b");
  auto sourceNimbleType = SchemaReader::getSchema(schemaBuilder.schemaNodes());

  const auto rowChild = [](const RowType& row,
                           const char* name) -> const Type& {
    return *row.childAt(row.findChild(name).value());
  };
  const auto scalarOffset = [](const Type& type) {
    return type.asScalar().scalarDescriptor().offset();
  };
  struct SourceStreamOffsets {
    uint32_t rootNull;
    uint32_t id;
    uint32_t profileNull;
    uint32_t profileAge;
    uint32_t featuresNull;
    uint32_t featureARowNull;
    uint32_t featureAScore;
    uint32_t featureAInMap;
  };
  const auto sourceOffsets = [&]() {
    const auto& root = sourceNimbleType->asRow();
    const auto& profile = rowChild(root, "profile").asRow();
    const auto& features = rowChild(root, "features").asFlatMap();
    const auto featureAIdx = features.findChild("a").value();
    const auto& featureARow = features.childAt(featureAIdx)->asRow();
    return SourceStreamOffsets{
        root.nullsDescriptor().offset(),
        scalarOffset(rowChild(root, "id")),
        profile.nullsDescriptor().offset(),
        scalarOffset(rowChild(profile, "age")),
        features.nullsDescriptor().offset(),
        featureARow.nullsDescriptor().offset(),
        scalarOffset(rowChild(featureARow, "score")),
        features.inMapDescriptorAt(featureAIdx).offset()};
  }();

  auto project = [&](const std::vector<Subfield>& subfields) {
    return buildProjectedNimbleType(sourceNimbleType.get(), subfields);
  };

  enum class ProjectedShape {
    Scalar,
    RowWithAge,
    FlatMapWithScore,
    FlatMapWithFullRow,
  };
  struct TestCase {
    std::string name;
    std::string subfield;
    ProjectedShape shape;
    std::vector<uint32_t> expectedStreamOffsets;
    std::vector<bool> expectedRowOrFlatMapNullStreams;
  };
  const std::vector<TestCase> testCases{
      {"scalar",
       "id",
       ProjectedShape::Scalar,
       {sourceOffsets.rootNull, sourceOffsets.id},
       {true, false}},
      {"nested-row",
       "profile.age",
       ProjectedShape::RowWithAge,
       {sourceOffsets.rootNull,
        sourceOffsets.profileNull,
        sourceOffsets.profileAge},
       {true, true, false}},
      {"flatmap-row-value",
       "features[\"a\"].score",
       ProjectedShape::FlatMapWithScore,
       {sourceOffsets.rootNull,
        sourceOffsets.featuresNull,
        sourceOffsets.featureARowNull,
        sourceOffsets.featureAScore,
        sourceOffsets.featureAInMap},
       {true, true, true, false, false}},
      {"missing-flatmap-row-value",
       "features[\"missing\"]",
       ProjectedShape::FlatMapWithFullRow,
       {sourceOffsets.rootNull,
        sourceOffsets.featuresNull,
        UINT32_MAX,
        UINT32_MAX,
        UINT32_MAX,
        UINT32_MAX},
       {true, true, true, false, false, false}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    std::vector<Subfield> subfields;
    subfields.emplace_back(testCase.subfield);

    auto metadata = project(subfields);

    ASSERT_EQ(metadata.nimbleType->kind(), Kind::Row);
    const auto& projectedRoot = metadata.nimbleType->asRow();
    ASSERT_EQ(projectedRoot.childrenCount(), 1);
    switch (testCase.shape) {
      case ProjectedShape::Scalar:
        EXPECT_EQ(projectedRoot.nameAt(0), "id");
        EXPECT_EQ(projectedRoot.childAt(0)->kind(), Kind::Scalar);
        break;
      case ProjectedShape::RowWithAge: {
        EXPECT_EQ(projectedRoot.nameAt(0), "profile");
        ASSERT_EQ(projectedRoot.childAt(0)->kind(), Kind::Row);
        const auto& projectedProfile = projectedRoot.childAt(0)->asRow();
        ASSERT_EQ(projectedProfile.childrenCount(), 1);
        EXPECT_EQ(projectedProfile.nameAt(0), "age");
        break;
      }
      case ProjectedShape::FlatMapWithScore: {
        EXPECT_EQ(projectedRoot.nameAt(0), "features");
        ASSERT_EQ(projectedRoot.childAt(0)->kind(), Kind::FlatMap);
        const auto& projectedFeatures = projectedRoot.childAt(0)->asFlatMap();
        ASSERT_EQ(projectedFeatures.childrenCount(), 1);
        EXPECT_EQ(projectedFeatures.nameAt(0), "a");
        ASSERT_EQ(projectedFeatures.childAt(0)->kind(), Kind::Row);
        const auto& projectedFeatureA = projectedFeatures.childAt(0)->asRow();
        ASSERT_EQ(projectedFeatureA.childrenCount(), 1);
        EXPECT_EQ(projectedFeatureA.nameAt(0), "score");
        break;
      }
      case ProjectedShape::FlatMapWithFullRow: {
        EXPECT_EQ(projectedRoot.nameAt(0), "features");
        ASSERT_EQ(projectedRoot.childAt(0)->kind(), Kind::FlatMap);
        const auto& projectedFeatures = projectedRoot.childAt(0)->asFlatMap();
        ASSERT_EQ(projectedFeatures.childrenCount(), 1);
        EXPECT_EQ(projectedFeatures.nameAt(0), "missing");
        ASSERT_EQ(projectedFeatures.childAt(0)->kind(), Kind::Row);
        const auto& projectedFeature = projectedFeatures.childAt(0)->asRow();
        ASSERT_EQ(projectedFeature.childrenCount(), 2);
        EXPECT_EQ(projectedFeature.nameAt(0), "score");
        EXPECT_EQ(projectedFeature.nameAt(1), "label");
        break;
      }
    }
    EXPECT_EQ(metadata.streamOffsets, testCase.expectedStreamOffsets);
    EXPECT_EQ(
        metadata.rowOrFlatMapNullStreams,
        testCase.expectedRowOrFlatMapNullStreams);
  }
}

TEST(SchemaUtilsTest, nestedFlatMapProjectionFails) {
  SchemaBuilder schemaBuilder;
  test::FlatMapChildAdder featuresAdder;
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW({
          {"nested",
           NIMBLE_ROW({
               {"features",
                NIMBLE_FLATMAP(String, NIMBLE_INTEGER(), featuresAdder)},
           })},
      }));
  featuresAdder.addChild("a");
  auto sourceNimbleType = SchemaReader::getSchema(schemaBuilder.schemaNodes());

  std::vector<Subfield> subfields;
  subfields.emplace_back("nested");

  NIMBLE_ASSERT_THROW(
      buildProjectedNimbleType(sourceNimbleType.get(), subfields),
      "FlatMap projection is supported only for top-level columns");
}

TEST(SchemaUtilsTest, projectionEncodingHintsSlidingWindowMap) {
  // Exercise `deriveColumnEncodings` + `emitOffsets` SlidingWindowMap branch
  // at the buildProjectedNimbleType(const Type*) level. No Projector test
  // serializes a
  // SlidingWindowMap top-level column, so this is the only test that locks
  // the encoding-hint-driven schema shape + per-descriptor offset order.
  //
  // Source schema (offsets assigned by SchemaBuilder in construction order):
  //   id BIGINT = 0
  //   dedup_map.key STRING = 1
  //   dedup_map.value INTEGER = 2
  //   dedup_map.offsetsDesc = 3, dedup_map.lengthsDesc = 4
  //   outer Row.nulls = 5
  SchemaBuilder schemaBuilder;
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW({
          {"id", NIMBLE_BIGINT()},
          {"dedup_map",
           NIMBLE_SLIDINGWINDOWMAP(NIMBLE_STRING(), NIMBLE_INTEGER())},
      }));
  auto sourceNimbleType = SchemaReader::getSchema(schemaBuilder.schemaNodes());

  std::vector<Subfield> subfields;
  subfields.emplace_back("id");
  subfields.emplace_back("dedup_map");

  auto projection = buildProjectedNimbleType(sourceNimbleType.get(), subfields);

  // Verify the SlidingWindowMap encoding is preserved (this is the path
  // through getColumnEncodings → deduplicatedMapColumns → velox-source
  // builder's SlidingWindowMap branch).
  ASSERT_EQ(projection.nimbleType->kind(), Kind::Row);
  const auto& root = projection.nimbleType->asRow();
  ASSERT_EQ(root.childrenCount(), 2);
  EXPECT_EQ(root.nameAt(0), "id");
  EXPECT_EQ(root.childAt(0)->kind(), Kind::Scalar);
  EXPECT_EQ(root.nameAt(1), "dedup_map");
  EXPECT_EQ(root.childAt(1)->kind(), Kind::SlidingWindowMap);

  // Walker visits: outer-Row.nulls (5), id Scalar (0), dedup_map offsets (3),
  // lengths (4), keys (1), values (2).
  EXPECT_EQ(
      projection.streamOffsets, std::vector<uint32_t>({5, 0, 3, 4, 1, 2}));
  EXPECT_EQ(
      projection.rowOrFlatMapNullStreams,
      std::vector<bool>({true, false, false, false, false, false}));
}

TEST(SchemaUtilsTest, projectionEncodingHintsMixed) {
  // Combine all three encoding-bearing top-level Kinds (ArrayWithOffsets,
  // SlidingWindowMap, FlatMap) plus a plain Scalar, with one missing FlatMap
  // key to exercise the UINT32_MAX placeholder path of `emitOffsets`
  // (via `emitPlaceholderOffsets` on a Scalar value template).
  //
  // Source schema offsets (construction order; FlatMap value subtrees are
  // built lazily inside addChild after the outer Row's ctor runs):
  //   id BIGINT = 0
  //   tags.element STRING = 1
  //   tags.offsetsDesc = 2, tags.lengthsDesc = 3
  //   dedup_map.key STRING = 4, dedup_map.value INTEGER = 5
  //   dedup_map.offsetsDesc = 6, dedup_map.lengthsDesc = 7
  //   features.nulls = 8 (FlatMap ctor)
  //   outer Row.nulls = 9 (outer Row ctor)
  //   features["a"].value INTEGER = 10, features["a"].inMap = 11
  //   features["c"].value INTEGER = 12, features["c"].inMap = 13
  SchemaBuilder schemaBuilder;
  test::FlatMapChildAdder featuresAdder;
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW({
          {"id", NIMBLE_BIGINT()},
          {"tags", NIMBLE_OFFSETARRAY(NIMBLE_STRING())},
          {"dedup_map",
           NIMBLE_SLIDINGWINDOWMAP(NIMBLE_STRING(), NIMBLE_INTEGER())},
          {"features", NIMBLE_FLATMAP(String, NIMBLE_INTEGER(), featuresAdder)},
      }));
  featuresAdder.addChild("a");
  featuresAdder.addChild("c");
  auto sourceNimbleType = SchemaReader::getSchema(schemaBuilder.schemaNodes());

  std::vector<Subfield> subfields;
  subfields.emplace_back("id");
  subfields.emplace_back("tags");
  subfields.emplace_back("dedup_map");
  subfields.emplace_back("features[\"a\"]");
  subfields.emplace_back("features[\"missing\"]");

  auto projection = buildProjectedNimbleType(sourceNimbleType.get(), subfields);

  // All three encoding-specific Kinds survive the projection.
  ASSERT_EQ(projection.nimbleType->kind(), Kind::Row);
  const auto& root = projection.nimbleType->asRow();
  ASSERT_EQ(root.childrenCount(), 4);
  EXPECT_EQ(root.childAt(0)->kind(), Kind::Scalar);
  EXPECT_EQ(root.childAt(1)->kind(), Kind::ArrayWithOffsets);
  EXPECT_EQ(root.childAt(2)->kind(), Kind::SlidingWindowMap);
  ASSERT_EQ(root.childAt(3)->kind(), Kind::FlatMap);

  // FlatMap children are alphabetical: real "a" then synthetic "missing".
  const auto& flatMap = root.childAt(3)->asFlatMap();
  ASSERT_EQ(flatMap.childrenCount(), 2);
  EXPECT_EQ(flatMap.nameAt(0), "a");
  EXPECT_EQ(flatMap.nameAt(1), "missing");

  // Walker visits, in order:
  //   outer Row.nulls (9), id Scalar (0),
  //   tags offsets (2), lengths (3), element (1),
  //   dedup_map offsets (6), lengths (7), key (4), value (5),
  //   features.nulls (8),
  //     "a" value (10), "a" inMap (11),
  //     "missing" value (UINT32_MAX), "missing" inMap (UINT32_MAX).
  EXPECT_EQ(
      projection.streamOffsets,
      std::vector<uint32_t>(
          {9, 0, 2, 3, 1, 6, 7, 4, 5, 8, 10, 11, UINT32_MAX, UINT32_MAX}));
  EXPECT_EQ(
      projection.rowOrFlatMapNullStreams,
      std::vector<bool>(
          {true,
           false,
           false,
           false,
           false,
           false,
           false,
           false,
           false,
           true,
           false,
           false,
           false,
           false}));
}

TEST(SchemaUtilsTest, projectionEmptySubfieldsFails) {
  auto type = velox::ROW({{"a", velox::BIGINT()}, {"b", velox::VARCHAR()}});
  std::vector<Subfield> subfields;
  NIMBLE_ASSERT_THROW(
      buildProjectedNimbleType(type->asRow(), subfields, {}),
      "projectedSubfields must not be empty");
}
