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
#include "velox/dwio/nimble/common/SchemaSerialization.h"
#include <gtest/gtest.h>
#include "velox/dwio/nimble/common/SchemaBuilder.h"
#include "velox/dwio/nimble/common/SchemaGenerated.h"
#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/dwio/nimble/common/SchemaTypes.h"
#include "velox/dwio/nimble/tests/SchemaUtils.h"

using namespace facebook;
using namespace facebook::nimble;

namespace {

std::shared_ptr<const Type> roundTrip(SchemaBuilder& schemaBuilder) {
  SchemaSerializer serializer;
  auto serialized = serializer.serialize(schemaBuilder);
  return SchemaDeserializer::deserialize(serialized);
}

std::shared_ptr<const Type> roundTrip(const Type& type) {
  SchemaSerializer serializer;
  auto serialized = serializer.serialize(type);
  return SchemaDeserializer::deserialize(serialized);
}

void expectSchemaNodesEqual(
    const std::vector<SchemaNode>& expected,
    const std::vector<SchemaNode>& actual) {
  ASSERT_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(expected[i].kind(), actual[i].kind());
    EXPECT_EQ(expected[i].offset(), actual[i].offset());
    EXPECT_EQ(expected[i].name(), actual[i].name());
    EXPECT_EQ(expected[i].scalarKind(), actual[i].scalarKind());
    EXPECT_EQ(expected[i].childrenCount(), actual[i].childrenCount());
    EXPECT_EQ(expected[i].attributes(), actual[i].attributes());
  }
}

} // namespace

TEST(SchemaTypesTest, isIntegerScalarKind) {
  struct TestCase {
    ScalarKind scalarKind;
    bool expected;
  };
  const std::vector<TestCase> testCases{
      {ScalarKind::Int8, true},
      {ScalarKind::UInt8, true},
      {ScalarKind::Int16, true},
      {ScalarKind::UInt16, true},
      {ScalarKind::Int32, true},
      {ScalarKind::UInt32, true},
      {ScalarKind::Int64, true},
      {ScalarKind::UInt64, true},
      {ScalarKind::Float, false},
      {ScalarKind::Double, false},
      {ScalarKind::Bool, false},
      {ScalarKind::String, false},
      {ScalarKind::Binary, false},
      {ScalarKind::Undefined, false},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(toString(testCase.scalarKind));
    EXPECT_EQ(isIntegerScalarKind(testCase.scalarKind), testCase.expected);
  }
}

TEST(SchemaSerializationTest, scalarInt8) {
  SchemaBuilder schemaBuilder;
  NIMBLE_SCHEMA(schemaBuilder, NIMBLE_ROW({{"field", NIMBLE_TINYINT()}}));
  auto root = roundTrip(schemaBuilder);
  ASSERT_EQ(Kind::Row, root->kind());
  auto& row = root->asRow();
  ASSERT_EQ(1, row.childrenCount());
  EXPECT_EQ("field", row.nameAt(0));
  EXPECT_EQ(Kind::Scalar, row.childAt(0)->kind());
  EXPECT_EQ(
      ScalarKind::Int8,
      row.childAt(0)->asScalar().scalarDescriptor().scalarKind());
}

TEST(SchemaSerializationTest, scalarTypes) {
  struct TestCase {
    std::string name;
    ScalarKind expectedKind;
  };

  std::vector<TestCase> cases = {
      {"int8", ScalarKind::Int8},
      {"uint8", ScalarKind::UInt8},
      {"int16", ScalarKind::Int16},
      {"uint16", ScalarKind::UInt16},
      {"int32", ScalarKind::Int32},
      {"uint32", ScalarKind::UInt32},
      {"int64", ScalarKind::Int64},
      {"uint64", ScalarKind::UInt64},
      {"float", ScalarKind::Float},
      {"double", ScalarKind::Double},
      {"bool", ScalarKind::Bool},
      {"string", ScalarKind::String},
      {"binary", ScalarKind::Binary},
  };

  for (const auto& tc : cases) {
    SchemaBuilder schemaBuilder;
    NIMBLE_SCHEMA(
        schemaBuilder,
        NIMBLE_ROW({{"f", builder.createScalarTypeBuilder(tc.expectedKind)}}));
    auto root = roundTrip(schemaBuilder);
    auto& row = root->asRow();
    ASSERT_EQ(1, row.childrenCount()) << tc.name;
    EXPECT_EQ(Kind::Scalar, row.childAt(0)->kind()) << tc.name;
    EXPECT_EQ(
        tc.expectedKind,
        row.childAt(0)->asScalar().scalarDescriptor().scalarKind())
        << tc.name;
  }
}

TEST(SchemaSerializationTest, rowType) {
  SchemaBuilder schemaBuilder;
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW(
          {{"a", NIMBLE_TINYINT()},
           {"b", NIMBLE_INTEGER()},
           {"c", NIMBLE_STRING()}}));
  auto root = roundTrip(schemaBuilder);
  ASSERT_EQ(Kind::Row, root->kind());
  auto& row = root->asRow();
  ASSERT_EQ(3, row.childrenCount());
  EXPECT_EQ("a", row.nameAt(0));
  EXPECT_EQ("b", row.nameAt(1));
  EXPECT_EQ("c", row.nameAt(2));
  EXPECT_EQ(
      ScalarKind::Int8,
      row.childAt(0)->asScalar().scalarDescriptor().scalarKind());
  EXPECT_EQ(
      ScalarKind::Int32,
      row.childAt(1)->asScalar().scalarDescriptor().scalarKind());
  EXPECT_EQ(
      ScalarKind::String,
      row.childAt(2)->asScalar().scalarDescriptor().scalarKind());
}

TEST(SchemaSerializationTest, arrayType) {
  SchemaBuilder schemaBuilder;
  NIMBLE_SCHEMA(
      schemaBuilder, NIMBLE_ROW({{"arr", NIMBLE_ARRAY(NIMBLE_BIGINT())}}));
  auto root = roundTrip(schemaBuilder);
  auto& row = root->asRow();
  ASSERT_EQ(1, row.childrenCount());
  auto& arr = row.childAt(0)->asArray();
  EXPECT_EQ(Kind::Array, row.childAt(0)->kind());
  EXPECT_EQ(Kind::Scalar, arr.elements()->kind());
  EXPECT_EQ(
      ScalarKind::Int64,
      arr.elements()->asScalar().scalarDescriptor().scalarKind());
}

TEST(SchemaSerializationTest, mapType) {
  SchemaBuilder schemaBuilder;
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW({{"m", NIMBLE_MAP(NIMBLE_STRING(), NIMBLE_INTEGER())}}));
  auto root = roundTrip(schemaBuilder);
  auto& row = root->asRow();
  ASSERT_EQ(1, row.childrenCount());
  auto& map = row.childAt(0)->asMap();
  EXPECT_EQ(Kind::Map, row.childAt(0)->kind());
  EXPECT_EQ(
      ScalarKind::String,
      map.keys()->asScalar().scalarDescriptor().scalarKind());
  EXPECT_EQ(
      ScalarKind::Int32,
      map.values()->asScalar().scalarDescriptor().scalarKind());
}

TEST(SchemaSerializationTest, arrayWithOffsetsType) {
  SchemaBuilder schemaBuilder;
  NIMBLE_SCHEMA(
      schemaBuilder, NIMBLE_ROW({{"oa", NIMBLE_OFFSETARRAY(NIMBLE_BIGINT())}}));
  auto root = roundTrip(schemaBuilder);
  auto& row = root->asRow();
  ASSERT_EQ(1, row.childrenCount());
  EXPECT_EQ(Kind::ArrayWithOffsets, row.childAt(0)->kind());
  auto& arr = row.childAt(0)->asArrayWithOffsets();
  EXPECT_EQ(Kind::Scalar, arr.elements()->kind());
  EXPECT_EQ(
      ScalarKind::Int64,
      arr.elements()->asScalar().scalarDescriptor().scalarKind());
}

TEST(SchemaSerializationTest, slidingWindowMapType) {
  SchemaBuilder schemaBuilder;
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW(
          {{"swm",
            NIMBLE_SLIDINGWINDOWMAP(NIMBLE_STRING(), NIMBLE_INTEGER())}}));
  auto root = roundTrip(schemaBuilder);
  auto& row = root->asRow();
  ASSERT_EQ(1, row.childrenCount());
  EXPECT_EQ(Kind::SlidingWindowMap, row.childAt(0)->kind());
  auto& swm = row.childAt(0)->asSlidingWindowMap();
  EXPECT_EQ(
      ScalarKind::String,
      swm.keys()->asScalar().scalarDescriptor().scalarKind());
  EXPECT_EQ(
      ScalarKind::Int32,
      swm.values()->asScalar().scalarDescriptor().scalarKind());
}

TEST(SchemaSerializationTest, flatMapType) {
  SchemaBuilder schemaBuilder;
  test::FlatMapChildAdder adder;
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW({{"fm", NIMBLE_FLATMAP(String, NIMBLE_INTEGER(), adder)}}));
  adder.addChild("key1");
  auto root = roundTrip(schemaBuilder);
  auto& row = root->asRow();
  ASSERT_EQ(1, row.childrenCount());
  EXPECT_EQ(Kind::FlatMap, row.childAt(0)->kind());
  auto& fm = row.childAt(0)->asFlatMap();
  EXPECT_EQ(ScalarKind::String, fm.keyScalarKind());
  ASSERT_EQ(1, fm.childrenCount());
  EXPECT_EQ(
      ScalarKind::Int32,
      fm.childAt(0)->asScalar().scalarDescriptor().scalarKind());
}

TEST(SchemaSerializationTest, timestampMicroNanoType) {
  SchemaBuilder schemaBuilder;
  NIMBLE_SCHEMA(
      schemaBuilder, NIMBLE_ROW({{"ts", NIMBLE_TIMESTAMPMICRONANO()}}));
  auto root = roundTrip(schemaBuilder);
  auto& row = root->asRow();
  ASSERT_EQ(1, row.childrenCount());
  EXPECT_EQ(Kind::TimestampMicroNano, row.childAt(0)->kind());
}

TEST(SchemaSerializationTest, nestedComplex) {
  // ROW containing ARRAY of MAP
  SchemaBuilder schemaBuilder;
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW(
          {{"nested",
            NIMBLE_ARRAY(NIMBLE_MAP(NIMBLE_STRING(), NIMBLE_BIGINT()))}}));
  auto root = roundTrip(schemaBuilder);
  auto& row = root->asRow();
  ASSERT_EQ(1, row.childrenCount());
  EXPECT_EQ("nested", row.nameAt(0));

  auto& arr = row.childAt(0)->asArray();
  EXPECT_EQ(Kind::Array, row.childAt(0)->kind());

  auto& map = arr.elements()->asMap();
  EXPECT_EQ(Kind::Map, arr.elements()->kind());
  EXPECT_EQ(
      ScalarKind::String,
      map.keys()->asScalar().scalarDescriptor().scalarKind());
  EXPECT_EQ(
      ScalarKind::Int64,
      map.values()->asScalar().scalarDescriptor().scalarKind());
}

// ---------------------------------------------------------------------------
// Per-SchemaNode attributes round-trip.
//
// The flatbuffer wire format for `SchemaNode.attributes` is already validated
// in NimbleSchemaAttributesTest. These tests cover the C++ plumbing:
// TypeBuilder::setAttributes(...) -> SchemaBuilder::addNode ->
// SchemaSerializer -> flatbuffer -> SchemaDeserializer ->
// SchemaReader::getSchema -> Type::attributes().
// ---------------------------------------------------------------------------

TEST(SchemaSerializationTest, attributesRoundTripOnLeaf) {
  // Set string-keyed attributes on the leaf TypeBuilder before it is
  // attached to the row. The keys must survive serialize -> deserialize and
  // surface on the deserialized Type with insertion order preserved.
  const std::vector<std::pair<std::string, std::string>> kAttrs = {
      {"key.a", "12"},
      {"key.b", "true"},
      {"key.c", "LONG"},
  };

  SchemaBuilder schemaBuilder;
  auto leaf = schemaBuilder.createScalarTypeBuilder(ScalarKind::Int64);
  leaf->setAttributes(kAttrs);
  auto row = schemaBuilder.createRowTypeBuilder(1);
  row->addChild("field", leaf);

  auto root = roundTrip(schemaBuilder);
  ASSERT_EQ(Kind::Row, root->kind());
  // Row itself was not annotated.
  EXPECT_TRUE(root->attributes().empty());

  const auto& child = root->asRow().childAt(0);
  EXPECT_EQ(Kind::Scalar, child->kind());
  EXPECT_EQ(kAttrs, child->attributes());
}

TEST(SchemaSerializationTest, attributesRoundTripOnParentAndChild) {
  // Both a nested struct (RowType) and one of its leaves carry distinct
  // attribute bags. Verify each set lands on the corresponding deserialized
  // Type independently and does not bleed between nodes.
  const std::vector<std::pair<std::string, std::string>> kParentAttrs = {
      {"key.a", "1"},
      {"key.b", "Variant"},
  };
  const std::vector<std::pair<std::string, std::string>> kChildAttrs = {
      {"key.a", "2"},
      {"key.b", "UUID"},
      {"key.c", "16"},
  };

  SchemaBuilder schemaBuilder;
  auto leaf = schemaBuilder.createScalarTypeBuilder(ScalarKind::Binary);
  leaf->setAttributes(kChildAttrs);
  auto innerRow = schemaBuilder.createRowTypeBuilder(1);
  innerRow->addChild("uuid", leaf);
  innerRow->setAttributes(kParentAttrs);
  auto outerRow = schemaBuilder.createRowTypeBuilder(1);
  outerRow->addChild("variant", innerRow);

  auto root = roundTrip(schemaBuilder);
  ASSERT_EQ(Kind::Row, root->kind());
  EXPECT_TRUE(root->attributes().empty());

  const auto& inner = root->asRow().childAt(0);
  ASSERT_EQ(Kind::Row, inner->kind());
  EXPECT_EQ(kParentAttrs, inner->attributes());

  const auto& innerLeaf = inner->asRow().childAt(0);
  EXPECT_EQ(Kind::Scalar, innerLeaf->kind());
  EXPECT_EQ(kChildAttrs, innerLeaf->attributes());
}

TEST(SchemaSerializationTest, attributesEmptyByDefault) {
  // A builder without setAttributes(...) must produce a deserialized Type
  // tree where every node exposes an empty (not throwing, not null) vector.
  // This is the no-op upgrade path for every NIMBLE writer that does not
  // know about attributes.
  SchemaBuilder schemaBuilder;
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW({{"a", NIMBLE_INTEGER()}, {"b", NIMBLE_STRING()}}));

  auto root = roundTrip(schemaBuilder);
  ASSERT_EQ(Kind::Row, root->kind());
  EXPECT_TRUE(root->attributes().empty());
  const auto& row = root->asRow();
  ASSERT_EQ(2, row.childrenCount());
  EXPECT_TRUE(row.childAt(0)->attributes().empty());
  EXPECT_TRUE(row.childAt(1)->attributes().empty());
}

// ---------------------------------------------------------------------------
// Backward-compatibility regressions.
//
// These tests pin invariants that protect every NIMBLE file produced before
// the `attributes` plumbing landed. They are written to FAIL if a future
// refactor either:
//   (a) stops accepting legacy flatbuffer buffers that omit the
//       `attributes` vtable slot, or
//   (b) starts emitting an `attributes` vtable slot for nodes whose
//       TypeBuilder never called setAttributes, which would silently change
//       on-disk bytes for callers that haven't opted in to attributes.
// ---------------------------------------------------------------------------

TEST(SchemaSerializationTest, legacyWireBufferDeserializesWithEmptyAttributes) {
  // Construct a flatbuffer Schema with NO `attributes` vtable slot on any
  // node -- byte-for-byte the wire format used by every NIMBLE writer
  // before this stack landed. SchemaDeserializer must accept it, produce
  // a correct Type tree, and surface empty attributes on every node.
  //
  // Schema shape: ROW{"a": Int32, "b": String}
  // DFS node layout (matches SchemaBuilder::addNode emit order):
  //   nodes[0] = ROW(name="root_field", offset=0, children=2)
  //   nodes[1] = Int32(name="a",        offset=1)
  //   nodes[2] = String(name="b",       offset=2)
  namespace fbs = facebook::nimble::serialization;
  flatbuffers::FlatBufferBuilder builder;

  auto makeNode = [&](fbs::Kind kind,
                      uint32_t offset,
                      uint32_t children,
                      const std::string& name) {
    auto nameOffset = builder.CreateString(name);
    fbs::SchemaNodeBuilder nodeBuilder(builder);
    nodeBuilder.add_kind(kind);
    nodeBuilder.add_children(children);
    nodeBuilder.add_name(nameOffset);
    nodeBuilder.add_offset(offset);
    // Deliberately NOT calling add_attributes(...) so the legacy vtable
    // shape is preserved.
    return nodeBuilder.Finish();
  };

  std::vector<flatbuffers::Offset<fbs::SchemaNode>> nodeOffsets;
  // The Row is always wrapped at the top by the writer; the root passed to
  // the deserializer is the row itself, which has no enclosing name from
  // SchemaBuilder's perspective. Mirror that here with an empty name on the
  // root.
  {
    fbs::SchemaNodeBuilder nodeBuilder(builder);
    nodeBuilder.add_kind(fbs::Kind_Row);
    nodeBuilder.add_children(2);
    nodeBuilder.add_offset(0);
    nodeOffsets.push_back(nodeBuilder.Finish());
  }
  nodeOffsets.push_back(makeNode(fbs::Kind_Int32, 1, 0, "a"));
  nodeOffsets.push_back(makeNode(fbs::Kind_String, 2, 0, "b"));

  auto nodesVector = builder.CreateVector(nodeOffsets);
  builder.Finish(fbs::CreateSchema(builder, nodesVector));

  // Sanity-check that the legacy buffer truly omits the attributes slot
  // on every node -- if a future refactor changes that, the rest of this
  // test becomes meaningless and we want to know.
  const auto* schema = fbs::GetSchema(builder.GetBufferPointer());
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->nodes(), nullptr);
  ASSERT_EQ(schema->nodes()->size(), 3u);
  for (const auto* node : *schema->nodes()) {
    EXPECT_EQ(node->attributes(), nullptr)
        << "Legacy fixture must not encode attributes vtable slot.";
  }

  auto serialized = std::string_view(
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize());
  auto root = SchemaDeserializer::deserialize(serialized);

  ASSERT_EQ(Kind::Row, root->kind());
  // The empty attribute bag must surface on every Type, not crash, and not
  // signal "I have attributes" to callers that branch on size.
  EXPECT_TRUE(root->attributes().empty());

  const auto& row = root->asRow();
  ASSERT_EQ(2, row.childrenCount());
  EXPECT_EQ("a", row.nameAt(0));
  EXPECT_EQ(Kind::Scalar, row.childAt(0)->kind());
  EXPECT_EQ(
      ScalarKind::Int32,
      row.childAt(0)->asScalar().scalarDescriptor().scalarKind());
  EXPECT_TRUE(row.childAt(0)->attributes().empty());

  EXPECT_EQ("b", row.nameAt(1));
  EXPECT_EQ(Kind::Scalar, row.childAt(1)->kind());
  EXPECT_EQ(
      ScalarKind::String,
      row.childAt(1)->asScalar().scalarDescriptor().scalarKind());
  EXPECT_TRUE(row.childAt(1)->attributes().empty());
}

TEST(SchemaSerializationTest, noAttributesWireShapeMatchesLegacy) {
  // When no TypeBuilder calls setAttributes(...), the new serializer must
  // NOT emit the `attributes` vtable slot on any node. This is the
  // byte-shape invariant for every NIMBLE writer that has not opted in to
  // attributes: their on-disk files stay structurally identical to
  // pre-attributes output, so any reader (Velox, external ORC-spec readers
  // that treat NIMBLE files as ORC) sees the same vtable layout it always
  // did.
  namespace fbs = facebook::nimble::serialization;
  SchemaBuilder schemaBuilder;
  // Cover one of each non-trivial container kind to guard against a future
  // change that accidentally stamps an empty attributes vector through one
  // codepath but not another.
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW(
          {{"i", NIMBLE_INTEGER()},
           {"arr", NIMBLE_ARRAY(NIMBLE_BIGINT())},
           {"m", NIMBLE_MAP(NIMBLE_STRING(), NIMBLE_INTEGER())},
           {"ts", NIMBLE_TIMESTAMPMICRONANO()}}));

  SchemaSerializer serializer;
  auto serialized = serializer.serialize(schemaBuilder);

  const auto* schema = fbs::GetSchema(serialized.data());
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->nodes(), nullptr);
  ASSERT_GT(schema->nodes()->size(), 0u);
  for (const auto* node : *schema->nodes()) {
    EXPECT_EQ(node->attributes(), nullptr)
        << "Serializer must not emit attributes vtable slot when no "
           "TypeBuilder set attributes -- this would silently change "
           "on-disk bytes for legacy callers.";
  }

  // Sanity: deserialize still works and produces empty attributes on every
  // Type. Belt-and-suspenders next to LegacyWireBufferDeserializes... .
  auto root = SchemaDeserializer::deserialize(serialized);
  ASSERT_EQ(Kind::Row, root->kind());
  EXPECT_TRUE(root->attributes().empty());
  const auto& row = root->asRow();
  for (size_t i = 0; i < row.childrenCount(); ++i) {
    EXPECT_TRUE(row.childAt(i)->attributes().empty()) << "child " << i;
  }
}

TEST(SchemaSerializationTest, schemaNodesFromTypeMatchesBuilderNodes) {
  const std::vector<std::pair<std::string, std::string>> kRootAttrs = {
      {"root", "attrs"},
  };
  const std::vector<std::pair<std::string, std::string>> kArrayAttrs = {
      {"array", "attrs"},
  };
  const std::vector<std::pair<std::string, std::string>> kLeafAttrs = {
      {"leaf", "attrs"},
  };
  const std::vector<std::pair<std::string, std::string>> kFlatMapAttrs = {
      {"flatmap", "attrs"},
  };

  SchemaBuilder schemaBuilder;
  auto offsetArrayElement =
      schemaBuilder.createScalarTypeBuilder(ScalarKind::Int64);
  offsetArrayElement->setAttributes(kLeafAttrs);
  auto offsetArray = schemaBuilder.createArrayWithOffsetsTypeBuilder();
  offsetArray->setChildren(offsetArrayElement);
  offsetArray->setAttributes(kArrayAttrs);

  auto slidingMap = schemaBuilder.createSlidingWindowMapTypeBuilder();
  slidingMap->setChildren(
      schemaBuilder.createScalarTypeBuilder(ScalarKind::String),
      schemaBuilder.createScalarTypeBuilder(ScalarKind::Int32));

  auto flatMap = schemaBuilder.createFlatMapTypeBuilder(ScalarKind::String);
  flatMap->setAttributes(kFlatMapAttrs);
  flatMap->addChild(
      "key1", schemaBuilder.createScalarTypeBuilder(ScalarKind::Double));
  auto flatMapArrayElement =
      schemaBuilder.createScalarTypeBuilder(ScalarKind::UInt32);
  auto flatMapArray = schemaBuilder.createArrayTypeBuilder();
  flatMapArray->setChildren(flatMapArrayElement);
  flatMap->addChild("key2", flatMapArray);

  auto timestamp = schemaBuilder.createTimestampMicroNanoTypeBuilder();

  auto root = schemaBuilder.createRowTypeBuilder(4);
  root->setAttributes(kRootAttrs);
  root->addChild("offsetArray", offsetArray);
  root->addChild("slidingMap", slidingMap);
  root->addChild("flatMap", flatMap);
  root->addChild("timestamp", timestamp);

  const auto deserializedType = roundTrip(schemaBuilder);

  expectSchemaNodesEqual(
      schemaBuilder.schemaNodes(), schemaNodes(*deserializedType));
}

TEST(SchemaSerializationTest, serializeTypePreservesSchemaShape) {
  SchemaBuilder schemaBuilder;
  test::FlatMapChildAdder adder;
  NIMBLE_SCHEMA(
      schemaBuilder,
      NIMBLE_ROW(
          {{"offsetArray", NIMBLE_OFFSETARRAY(NIMBLE_BIGINT())},
           {"slidingMap",
            NIMBLE_SLIDINGWINDOWMAP(NIMBLE_STRING(), NIMBLE_INTEGER())},
           {"flatMap", NIMBLE_FLATMAP(String, NIMBLE_DOUBLE(), adder)},
           {"timestamp", NIMBLE_TIMESTAMPMICRONANO()}}));
  adder.addChild("key1");
  adder.addChild("key2");

  const auto deserializedType = roundTrip(schemaBuilder);
  const auto reserializedType = roundTrip(*deserializedType);

  test::compareSchema(schemaBuilder.schemaNodes(), reserializedType);
}

TEST(SchemaSerializationTest, serializeTypePreservesAttributes) {
  const std::vector<std::pair<std::string, std::string>> kRowAttrs = {
      {"row", "root"},
  };
  const std::vector<std::pair<std::string, std::string>> kArrayAttrs = {
      {"array", "items"},
  };
  const std::vector<std::pair<std::string, std::string>> kLeafAttrs = {
      {"leaf", "value"},
  };

  SchemaBuilder schemaBuilder;
  auto leaf = schemaBuilder.createScalarTypeBuilder(ScalarKind::Binary);
  leaf->setAttributes(kLeafAttrs);
  auto array = schemaBuilder.createArrayTypeBuilder();
  array->setChildren(leaf);
  array->setAttributes(kArrayAttrs);
  auto row = schemaBuilder.createRowTypeBuilder(1);
  row->addChild("field", array);
  row->setAttributes(kRowAttrs);

  const auto deserializedType = roundTrip(schemaBuilder);
  const auto reserializedType = roundTrip(*deserializedType);

  ASSERT_EQ(Kind::Row, reserializedType->kind());
  EXPECT_EQ(kRowAttrs, reserializedType->attributes());

  const auto& arrayType = reserializedType->asRow().childAt(0);
  ASSERT_EQ(Kind::Array, arrayType->kind());
  EXPECT_EQ(kArrayAttrs, arrayType->attributes());

  const auto& leafType = arrayType->asArray().elements();
  ASSERT_EQ(Kind::Scalar, leafType->kind());
  EXPECT_EQ(kLeafAttrs, leafType->attributes());
}
