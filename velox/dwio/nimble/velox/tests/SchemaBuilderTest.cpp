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

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <sstream>
#include <type_traits>

#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/velox/HybridFlatMap.h"
#include "velox/dwio/nimble/velox/SchemaBuilder.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"

namespace facebook::nimble {
namespace {

static_assert(!std::is_base_of_v<FlatMapTypeBuilder, HybridFlatMapTypeBuilder>);
static_assert(!std::is_base_of_v<FlatMapType, HybridFlatMapType>);

std::vector<std::pair<std::string, std::string>> hybridFlatMapAttributes(
    const HybridFlatMap& metadata) {
  return {{
      std::string{HybridFlatMap::kAttributeName},
      metadata.serialize(),
  }};
}

std::shared_ptr<TypeBuilder> makeComplexValue(SchemaBuilder& builder) {
  auto array = builder.createArrayTypeBuilder();
  array->setChildren(builder.createScalarTypeBuilder(ScalarKind::Int32));

  auto offsetArray = builder.createArrayWithOffsetsTypeBuilder();
  offsetArray->setChildren(builder.createScalarTypeBuilder(ScalarKind::UInt32));

  auto map = builder.createMapTypeBuilder();
  map->setChildren(
      builder.createScalarTypeBuilder(ScalarKind::String),
      builder.createScalarTypeBuilder(ScalarKind::Double));

  auto slidingMap = builder.createSlidingWindowMapTypeBuilder();
  slidingMap->setChildren(
      builder.createScalarTypeBuilder(ScalarKind::Int16),
      builder.createScalarTypeBuilder(ScalarKind::Bool));

  auto flatMap = builder.createFlatMapTypeBuilder(ScalarKind::String);
  flatMap->addChild(
      "flat-key", builder.createScalarTypeBuilder(ScalarKind::Int64));

  auto row = builder.createRowTypeBuilder(6);
  row->addChild("timestamp", builder.createTimestampMicroNanoTypeBuilder());
  row->addChild("array", array);
  row->addChild("offsetArray", offsetArray);
  row->addChild("map", map);
  row->addChild("slidingMap", slidingMap);
  row->addChild("flatMap", flatMap);
  return row;
}

TEST(SchemaBuilderTest, hybridFlatMapUsesGroupChildrenAndKeyStreams) {
  SchemaBuilder builder;
  auto root = builder.createRowTypeBuilder(1);
  auto hybridMap = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);
  EXPECT_EQ(&hybridMap->asHybridFlatMap(), hybridMap.get());
  EXPECT_EQ(hybridMap->kind(), Kind::HybridFlatMap);

  auto groupValue = builder.createScalarTypeBuilder(ScalarKind::Int64);
  const auto nodeCountBeforeGroup = builder.nodeCount();
  const auto streams = hybridMap->addGroup(7, {"a", "b"}, groupValue);
  auto defaultValue = builder.createScalarTypeBuilder(ScalarKind::Int64);
  const auto defaultStreams =
      hybridMap->addGroup(HybridFlatMap::kDefaultGroupId, {}, defaultValue);
  root->addChild("features", hybridMap);

  EXPECT_EQ(builder.nodeCount(), nodeCountBeforeGroup + 5);
  EXPECT_EQ(streams.keyDescriptor.scalarKind(), ScalarKind::String);
  EXPECT_EQ(streams.inMapDescriptor.scalarKind(), ScalarKind::Bool);
  EXPECT_EQ(defaultStreams.keyDescriptor.scalarKind(), ScalarKind::String);
  EXPECT_EQ(defaultStreams.inMapDescriptor.scalarKind(), ScalarKind::Bool);
  EXPECT_NE(streams.keyDescriptor.offset(), streams.inMapDescriptor.offset());
  EXPECT_NE(
      streams.keyDescriptor.offset(), groupValue->scalarDescriptor().offset());
  EXPECT_NE(
      streams.inMapDescriptor.offset(),
      groupValue->scalarDescriptor().offset());
  ASSERT_EQ(hybridMap->groupCount(), 2);
  const auto storedGroup = hybridMap->groupAt(0);
  EXPECT_EQ(storedGroup.groupId, 7);
  EXPECT_EQ(storedGroup.groupKeys, (std::vector<std::string>{"a", "b"}));
  const auto storedDefaultGroup = hybridMap->groupAt(1);
  EXPECT_EQ(storedDefaultGroup.groupId, HybridFlatMap::kDefaultGroupId);
  EXPECT_TRUE(storedDefaultGroup.groupKeys.empty());
  EXPECT_EQ(storedGroup.keyDescriptor.offset(), streams.keyDescriptor.offset());
  EXPECT_EQ(
      storedGroup.inMapDescriptor.offset(), streams.inMapDescriptor.offset());
  EXPECT_EQ(&storedGroup.valueType, groupValue.get());

  const auto nodes = builder.schemaNodes();
  ASSERT_EQ(nodes.size(), 8);
  EXPECT_EQ(nodes[1].kind(), Kind::HybridFlatMap);
  EXPECT_EQ(nodes[1].childrenCount(), 2);
  EXPECT_EQ(nodes[1].scalarKind(), ScalarKind::String);
  ASSERT_EQ(nodes[1].attributes().size(), 1);
  EXPECT_EQ(nodes[1].attributes()[0].first, HybridFlatMap::kAttributeName);
  auto attributes = nodes[1].attributes();
  const auto metadata = HybridFlatMap::extractAttribute(attributes);
  EXPECT_TRUE(attributes.empty());
  ASSERT_EQ(metadata.groups.size(), 2);
  EXPECT_EQ(metadata.groups[0].groupId, 7);
  EXPECT_EQ(metadata.groups[0].groupKeys, (std::vector<std::string>{"a", "b"}));
  EXPECT_TRUE(metadata.groups[1].groupKeys.empty());
  EXPECT_EQ(nodes[2].scalarKind(), ScalarKind::String);
  EXPECT_EQ(nodes[3].scalarKind(), ScalarKind::Bool);

  const auto schema = SchemaReader::getSchema(nodes);
  NIMBLE_ASSERT_THROW(
      schema->asHybridFlatMap(),
      "Cannot cast to HybridFlatMap. Current type is Row");
  const auto& child = schema->asRow().childAt(0);
  EXPECT_FALSE(child->isFlatMap());
  EXPECT_TRUE(child->isHybridFlatMap());
  const auto& hybridMapType = child->asHybridFlatMap();
  EXPECT_EQ(hybridMapType.kind(), Kind::HybridFlatMap);
  EXPECT_EQ(hybridMapType.keyScalarKind(), ScalarKind::String);
  ASSERT_EQ(hybridMapType.groupCount(), 2);
  EXPECT_EQ(hybridMapType.groupAt(0).groupId, 7);
  EXPECT_EQ(
      hybridMapType.groupAt(0).groupKeys, (std::vector<std::string>{"a", "b"}));
  EXPECT_EQ(
      hybridMapType.groupAt(0).keyDescriptor.scalarKind(), ScalarKind::String);
  EXPECT_EQ(
      hybridMapType.groupAt(0).inMapDescriptor.scalarKind(), ScalarKind::Bool);
  EXPECT_EQ(
      hybridMapType.groupAt(0)
          .valueType->asScalar()
          .scalarDescriptor()
          .offset(),
      groupValue->scalarDescriptor().offset());
  EXPECT_EQ(
      hybridMapType.defaultGroup().groupId, HybridFlatMap::kDefaultGroupId);
  EXPECT_TRUE(hybridMapType.defaultGroup().groupKeys.empty());
  EXPECT_EQ(hybridMapType.findGroup("a"), 0);
  EXPECT_EQ(hybridMapType.findGroup("b"), 0);
  EXPECT_FALSE(hybridMapType.findGroup("unknown").has_value());
  EXPECT_EQ(
      hybridMapType.valueType().asScalar().scalarDescriptor().scalarKind(),
      ScalarKind::Int64);
  EXPECT_EQ(toString(Kind::HybridFlatMap), "HybridFlatMap");

  NIMBLE_ASSERT_THROW(hybridMap->groupAt(2), "Index out of range");
  NIMBLE_ASSERT_THROW(hybridMapType.groupAt(2), "Index out of range");

  std::vector<offset_size> valueOffsets;
  EXPECT_FALSE(visitValueStreamLeaves(*child, [&](offset_size offset) {
    valueOffsets.push_back(offset);
    return false;
  }));
  EXPECT_EQ(
      valueOffsets,
      (std::vector<offset_size>{
          groupValue->scalarDescriptor().offset(),
          defaultValue->scalarDescriptor().offset()}));

  std::vector<offset_size> presenceOffsets;
  EXPECT_FALSE(visitPresenceStreamOffsets(*child, [&](offset_size offset) {
    presenceOffsets.push_back(offset);
    return false;
  }));
  EXPECT_EQ(presenceOffsets.front(), hybridMapType.nullsDescriptor().offset());
  EXPECT_NE(
      std::find(
          presenceOffsets.begin(),
          presenceOffsets.end(),
          streams.keyDescriptor.offset()),
      presenceOffsets.end());
  EXPECT_NE(
      std::find(
          presenceOffsets.begin(),
          presenceOffsets.end(),
          streams.inMapDescriptor.offset()),
      presenceOffsets.end());

  std::vector<size_t> traversedGroupOrdinals;
  SchemaReader::traverseSchema(
      schema,
      [&](uint32_t, const Type&, const SchemaReader::NodeInfo& nodeInfo) {
        if (nodeInfo.parentType == child.get()) {
          EXPECT_EQ(nodeInfo.name, "valueType");
          traversedGroupOrdinals.push_back(nodeInfo.placeInSibling);
        }
      });
  EXPECT_EQ(traversedGroupOrdinals, (std::vector<size_t>{0, 1}));

  std::ostringstream builderDescription;
  builderDescription << builder;
  EXPECT_NE(builderDescription.str().find("HYBRIDFLATMAP"), std::string::npos);
  std::ostringstream readerDescription;
  readerDescription << schema;
  EXPECT_NE(readerDescription.str().find("HYBRIDFLATMAP"), std::string::npos);
}

TEST(SchemaBuilderTest, hybridFlatMapSupportsComplexGroupValueTypes) {
  SchemaBuilder builder;
  auto hybridMap = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);
  hybridMap->addGroup(0, {"a"}, makeComplexValue(builder));
  hybridMap->addGroup(1, {"b"}, makeComplexValue(builder));
  hybridMap->addGroup(
      HybridFlatMap::kDefaultGroupId, {}, makeComplexValue(builder));

  const auto schema = SchemaReader::getSchema(builder.schemaNodes());
  const auto& value = schema->asHybridFlatMap().valueType().asRow();
  ASSERT_EQ(value.childrenCount(), 6);
  EXPECT_TRUE(value.childAt(0)->isTimestampMicroNano());
  EXPECT_TRUE(value.childAt(1)->isArray());
  EXPECT_TRUE(value.childAt(2)->isArrayWithOffsets());
  EXPECT_TRUE(value.childAt(3)->isMap());
  EXPECT_TRUE(value.childAt(4)->isSlidingWindowMap());
  EXPECT_TRUE(value.childAt(5)->isFlatMap());
}

TEST(SchemaBuilderTest, hybridFlatMapStreamVisitorsShortCircuit) {
  SchemaBuilder builder;
  auto hybridMap = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);
  hybridMap->addGroup(
      0, {"a"}, builder.createScalarTypeBuilder(ScalarKind::Int64));
  hybridMap->addGroup(
      HybridFlatMap::kDefaultGroupId,
      {},
      builder.createScalarTypeBuilder(ScalarKind::Int64));
  const auto schema = SchemaReader::getSchema(builder.schemaNodes());
  const auto& map = schema->asHybridFlatMap();

  std::vector<offset_size> valueOffsets;
  EXPECT_TRUE(visitValueStreamLeaves(map, [&](offset_size offset) {
    valueOffsets.push_back(offset);
    return true;
  }));
  EXPECT_EQ(
      valueOffsets,
      (std::vector<offset_size>{
          map.groupAt(0).valueType->asScalar().scalarDescriptor().offset()}));

  std::vector<offset_size> presenceOffsets;
  const auto firstKeysOffset = map.groupAt(0).keyDescriptor.offset();
  EXPECT_TRUE(visitPresenceStreamOffsets(map, [&](offset_size offset) {
    presenceOffsets.push_back(offset);
    return offset == firstKeysOffset;
  }));
  EXPECT_EQ(
      presenceOffsets,
      (std::vector<offset_size>{
          map.nullsDescriptor().offset(), firstKeysOffset}));

  presenceOffsets.clear();
  const auto firstValueOffset =
      map.groupAt(0).valueType->asScalar().scalarDescriptor().offset();
  EXPECT_TRUE(visitPresenceStreamOffsets(map, [&](offset_size offset) {
    presenceOffsets.push_back(offset);
    return offset == firstValueOffset;
  }));
  EXPECT_EQ(
      presenceOffsets,
      (std::vector<offset_size>{
          map.nullsDescriptor().offset(),
          map.groupAt(0).keyDescriptor.offset(),
          map.groupAt(0).inMapDescriptor.offset(),
          firstValueOffset}));
}

TEST(SchemaBuilderTest, hybridFlatMapRejectsNestedHybridFlatMap) {
  {
    SchemaBuilder builder;
    auto outer = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);
    auto nested = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);
    nested->addGroup(
        1, {"nested"}, builder.createScalarTypeBuilder(ScalarKind::Int64));
    nested->addGroup(
        HybridFlatMap::kDefaultGroupId,
        {},
        builder.createScalarTypeBuilder(ScalarKind::Int64));
    outer->addGroup(0, {"outer"}, nested);
    outer->addGroup(
        HybridFlatMap::kDefaultGroupId,
        {},
        builder.createScalarTypeBuilder(ScalarKind::Int64));

    NIMBLE_ASSERT_THROW(
        builder.schemaNodes(),
        "Hybrid FlatMap groups must have the same logical value type");
  }

  {
    SchemaBuilder builder;
    auto outer = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);
    auto nested = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);
    nested->addGroup(
        1, {"nested"}, builder.createScalarTypeBuilder(ScalarKind::Int64));
    nested->addGroup(
        HybridFlatMap::kDefaultGroupId,
        {},
        builder.createScalarTypeBuilder(ScalarKind::Int64));
    auto row = builder.createRowTypeBuilder(1);
    row->addChild("nested", nested);
    outer->addGroup(0, {"outer"}, row);
    outer->addGroup(
        HybridFlatMap::kDefaultGroupId,
        {},
        builder.createScalarTypeBuilder(ScalarKind::Int64));

    NIMBLE_ASSERT_THROW(
        builder.schemaNodes(),
        "Hybrid FlatMap groups must have the same logical value type");
  }

  {
    SchemaBuilder builder;
    auto outer = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);
    auto nested = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);
    nested->addGroup(
        1, {"nested"}, builder.createScalarTypeBuilder(ScalarKind::Int64));
    nested->addGroup(
        HybridFlatMap::kDefaultGroupId,
        {},
        builder.createScalarTypeBuilder(ScalarKind::Int64));
    auto flatMap = builder.createFlatMapTypeBuilder(ScalarKind::String);
    flatMap->addChild(
        "value", builder.createScalarTypeBuilder(ScalarKind::Int64));
    flatMap->addChild("nested", nested);
    outer->addGroup(0, {"outer"}, flatMap);
    outer->addGroup(
        HybridFlatMap::kDefaultGroupId,
        {},
        builder.createScalarTypeBuilder(ScalarKind::Int64));

    NIMBLE_ASSERT_THROW(
        builder.schemaNodes(),
        "Hybrid FlatMap groups must have the same logical value type");
  }

  {
    const auto nestedGroupScalar =
        std::make_shared<ScalarType>(StreamDescriptor{3, ScalarKind::Int64});
    const auto nestedDefaultScalar =
        std::make_shared<ScalarType>(StreamDescriptor{6, ScalarKind::Int64});
    const auto nested = std::make_shared<HybridFlatMapType>(
        StreamDescriptor{0, ScalarKind::Bool},
        ScalarKind::String,
        std::vector<HybridFlatMapType::Group>{
            {.groupId = 1,
             .groupKeys = {"nested"},
             .keyDescriptor = StreamDescriptor{1, ScalarKind::String},
             .inMapDescriptor = StreamDescriptor{2, ScalarKind::Bool},
             .valueType = nestedGroupScalar},
            {.groupId = HybridFlatMap::kDefaultGroupId,
             .groupKeys = {},
             .keyDescriptor = StreamDescriptor{4, ScalarKind::String},
             .inMapDescriptor = StreamDescriptor{5, ScalarKind::Bool},
             .valueType = nestedDefaultScalar},
        });
    const auto outerDefaultScalar =
        std::make_shared<ScalarType>(StreamDescriptor{10, ScalarKind::Int64});
    NIMBLE_ASSERT_THROW(
        std::make_shared<HybridFlatMapType>(
            StreamDescriptor{7, ScalarKind::Bool},
            ScalarKind::String,
            std::vector<HybridFlatMapType::Group>{
                {.groupId = 0,
                 .groupKeys = {"outer"},
                 .keyDescriptor = StreamDescriptor{8, ScalarKind::String},
                 .inMapDescriptor = StreamDescriptor{9, ScalarKind::Bool},
                 .valueType = nested},
                {.groupId = HybridFlatMap::kDefaultGroupId,
                 .groupKeys = {},
                 .keyDescriptor = StreamDescriptor{11, ScalarKind::String},
                 .inMapDescriptor = StreamDescriptor{12, ScalarKind::Bool},
                 .valueType = outerDefaultScalar},
            }),
        "Hybrid FlatMap groups must have the same logical value type");

    const auto nestedRow = std::make_shared<RowType>(
        StreamDescriptor{13, ScalarKind::Bool},
        std::vector<std::string>{"nested"},
        std::vector<std::shared_ptr<const Type>>{nested});
    NIMBLE_ASSERT_THROW(
        std::make_shared<HybridFlatMapType>(
            StreamDescriptor{14, ScalarKind::Bool},
            ScalarKind::String,
            std::vector<HybridFlatMapType::Group>{
                {.groupId = 0,
                 .groupKeys = {"outer"},
                 .keyDescriptor = StreamDescriptor{15, ScalarKind::String},
                 .inMapDescriptor = StreamDescriptor{16, ScalarKind::Bool},
                 .valueType = nestedRow},
                {.groupId = HybridFlatMap::kDefaultGroupId,
                 .groupKeys = {},
                 .keyDescriptor = StreamDescriptor{17, ScalarKind::String},
                 .inMapDescriptor = StreamDescriptor{18, ScalarKind::Bool},
                 .valueType = outerDefaultScalar},
            }),
        "Hybrid FlatMap groups must have the same logical value type");

    const auto validFlatMapChild =
        std::make_shared<ScalarType>(StreamDescriptor{19, ScalarKind::Int64});
    std::vector<std::unique_ptr<StreamDescriptor>> inMapDescriptors;
    inMapDescriptors.push_back(
        std::make_unique<StreamDescriptor>(20, ScalarKind::Bool));
    inMapDescriptors.push_back(
        std::make_unique<StreamDescriptor>(21, ScalarKind::Bool));
    const auto nestedFlatMap = std::make_shared<FlatMapType>(
        StreamDescriptor{22, ScalarKind::Bool},
        ScalarKind::String,
        std::vector<std::string>{"value", "nested"},
        std::move(inMapDescriptors),
        std::vector<std::shared_ptr<const Type>>{validFlatMapChild, nested});
    NIMBLE_ASSERT_THROW(
        std::make_shared<HybridFlatMapType>(
            StreamDescriptor{23, ScalarKind::Bool},
            ScalarKind::String,
            std::vector<HybridFlatMapType::Group>{
                {.groupId = 0,
                 .groupKeys = {"outer"},
                 .keyDescriptor = StreamDescriptor{24, ScalarKind::String},
                 .inMapDescriptor = StreamDescriptor{25, ScalarKind::Bool},
                 .valueType = nestedFlatMap},
                {.groupId = HybridFlatMap::kDefaultGroupId,
                 .groupKeys = {},
                 .keyDescriptor = StreamDescriptor{26, ScalarKind::String},
                 .inMapDescriptor = StreamDescriptor{27, ScalarKind::Bool},
                 .valueType = outerDefaultScalar},
            }),
        "Hybrid FlatMap groups must have the same logical value type");
  }
}

TEST(SchemaBuilderTest, hybridFlatMapRejectsNestedRightFlatMapChild) {
  SchemaBuilder builder;
  auto hybridMap = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);

  auto homogeneousFlatMap =
      builder.createFlatMapTypeBuilder(ScalarKind::String);
  homogeneousFlatMap->addChild(
      "first", builder.createScalarTypeBuilder(ScalarKind::Int64));
  homogeneousFlatMap->addChild(
      "second", builder.createScalarTypeBuilder(ScalarKind::Int64));
  hybridMap->addGroup(0, {"configured"}, homogeneousFlatMap);

  auto nested = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);
  nested->addGroup(
      1, {"nested"}, builder.createScalarTypeBuilder(ScalarKind::Int64));
  nested->addGroup(
      HybridFlatMap::kDefaultGroupId,
      {},
      builder.createScalarTypeBuilder(ScalarKind::Int64));
  auto heterogeneousFlatMap =
      builder.createFlatMapTypeBuilder(ScalarKind::String);
  heterogeneousFlatMap->addChild(
      "first", builder.createScalarTypeBuilder(ScalarKind::Int64));
  heterogeneousFlatMap->addChild("second", nested);
  hybridMap->addGroup(HybridFlatMap::kDefaultGroupId, {}, heterogeneousFlatMap);

  NIMBLE_ASSERT_THROW(
      builder.schemaNodes(),
      "Hybrid FlatMap groups must have the same logical value type");
}

TEST(SchemaBuilderTest, hybridFlatMapRejectsUnsupportedKeyKinds) {
  for (const auto keyKind : {
           ScalarKind::UInt8,
           ScalarKind::UInt16,
           ScalarKind::UInt32,
           ScalarKind::UInt64,
           ScalarKind::Float,
           ScalarKind::Double,
           ScalarKind::Bool,
           ScalarKind::Binary,
           ScalarKind::Undefined,
       }) {
    SCOPED_TRACE(keyKind);
    const auto error =
        "Hybrid FlatMap key kind is unsupported: " + toString(keyKind);
    SchemaBuilder builder;
    NIMBLE_ASSERT_THROW(builder.createHybridFlatMapTypeBuilder(keyKind), error);
    NIMBLE_ASSERT_THROW(
        std::make_shared<HybridFlatMapType>(
            StreamDescriptor{0, ScalarKind::Bool},
            keyKind,
            std::vector<HybridFlatMapType::Group>{}),
        error);
  }
}

TEST(SchemaBuilderTest, hybridFlatMapRejectsTruncatedGroup) {
  // The Default group's value node is missing, so its stream nodes run past
  // the end of the schema.
  std::vector<SchemaNode> nodes{
      SchemaNode{
          Kind::HybridFlatMap,
          0,
          ScalarKind::String,
          std::nullopt,
          2,
          hybridFlatMapAttributes(
              HybridFlatMap{
                  .groups =
                      {
                          {.groupId = 0, .groupKeys = {"a"}},
                          {.groupId = HybridFlatMap::kDefaultGroupId,
                           .groupKeys = {}},
                      },
              })},
      SchemaNode{Kind::Scalar, 1, ScalarKind::String},
      SchemaNode{Kind::Scalar, 2, ScalarKind::Bool},
      SchemaNode{Kind::Scalar, 3, ScalarKind::Int64},
      SchemaNode{Kind::Scalar, 4, ScalarKind::String},
      SchemaNode{Kind::Scalar, 5, ScalarKind::Bool},
  };

  NIMBLE_ASSERT_THROW(
      SchemaReader::getSchema(nodes), "Incomplete Hybrid FlatMap group");
}

TEST(SchemaBuilderTest, hybridFlatMapValidatesGroupContract) {
  {
    SchemaBuilder builder;
    builder.createHybridFlatMapTypeBuilder(ScalarKind::String);

    NIMBLE_ASSERT_THROW(
        builder.schemaNodes(), "Hybrid FlatMap requires at least two groups");
  }

  {
    SchemaBuilder builder;
    auto hybridMap = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);
    hybridMap->addGroup(
        HybridFlatMap::kDefaultGroupId,
        {},
        builder.createScalarTypeBuilder(ScalarKind::Int64));

    NIMBLE_ASSERT_THROW(
        builder.schemaNodes(), "Hybrid FlatMap requires at least two groups");
  }

  {
    SchemaBuilder builder;
    auto hybridMap = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);
    hybridMap->addGroup(
        0, {"a"}, builder.createScalarTypeBuilder(ScalarKind::Int64));
    hybridMap->addGroup(
        1, {"b"}, builder.createScalarTypeBuilder(ScalarKind::Int64));

    NIMBLE_ASSERT_THROW(
        builder.schemaNodes(),
        "Hybrid FlatMap must have exactly one Default group");
  }

  {
    SchemaBuilder builder;
    auto hybridMap = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);
    hybridMap->addGroup(
        HybridFlatMap::kDefaultGroupId,
        {},
        builder.createScalarTypeBuilder(ScalarKind::Int64));
    hybridMap->addGroup(
        HybridFlatMap::kDefaultGroupId,
        {},
        builder.createScalarTypeBuilder(ScalarKind::Int64));

    NIMBLE_ASSERT_THROW(
        builder.schemaNodes(), "Duplicate Hybrid FlatMap group ID: 4294967295");
  }

  {
    SchemaBuilder builder;
    auto hybridMap = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);
    hybridMap->addGroup(
        0, {""}, builder.createScalarTypeBuilder(ScalarKind::Int64));
    hybridMap->addGroup(
        HybridFlatMap::kDefaultGroupId,
        {},
        builder.createScalarTypeBuilder(ScalarKind::Int64));

    NIMBLE_ASSERT_THROW(
        builder.schemaNodes(), "Hybrid FlatMap key cannot be empty");
  }

  {
    SchemaBuilder builder;
    auto hybridMap = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);
    hybridMap->addGroup(
        0, {"a"}, builder.createScalarTypeBuilder(ScalarKind::Int64));
    hybridMap->addGroup(
        HybridFlatMap::kDefaultGroupId,
        {"default-key"},
        builder.createScalarTypeBuilder(ScalarKind::Int64));

    NIMBLE_ASSERT_THROW(
        builder.schemaNodes(),
        "Hybrid FlatMap Default group cannot contain group keys");
  }

  {
    SchemaBuilder builder;
    auto hybridMap = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);
    hybridMap->addGroup(
        0, {"a"}, builder.createScalarTypeBuilder(ScalarKind::Int64));
    hybridMap->addGroup(
        1, {"a"}, builder.createScalarTypeBuilder(ScalarKind::Int64));
    hybridMap->addGroup(
        HybridFlatMap::kDefaultGroupId,
        {},
        builder.createScalarTypeBuilder(ScalarKind::Int64));

    NIMBLE_ASSERT_THROW(
        builder.schemaNodes(), "Duplicate Hybrid FlatMap key: 'a'");
  }

  {
    SchemaBuilder builder;
    auto hybridMap = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);
    hybridMap->addGroup(
        0, {"a"}, builder.createScalarTypeBuilder(ScalarKind::Int64));
    hybridMap->addGroup(
        0, {"b"}, builder.createScalarTypeBuilder(ScalarKind::Int64));
    hybridMap->addGroup(
        HybridFlatMap::kDefaultGroupId,
        {},
        builder.createScalarTypeBuilder(ScalarKind::Int64));

    NIMBLE_ASSERT_THROW(
        builder.schemaNodes(), "Duplicate Hybrid FlatMap group ID: 0");
  }

  {
    SchemaBuilder builder;
    auto hybridMap = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);
    hybridMap->addGroup(
        0, {}, builder.createScalarTypeBuilder(ScalarKind::Int64));
    hybridMap->addGroup(
        HybridFlatMap::kDefaultGroupId,
        {},
        builder.createScalarTypeBuilder(ScalarKind::Int64));

    NIMBLE_ASSERT_THROW(
        builder.schemaNodes(),
        "Hybrid FlatMap group must contain at least one key: 0");
  }

  using ValueFactory =
      std::function<std::shared_ptr<TypeBuilder>(SchemaBuilder&)>;
  const auto scalar = [](ScalarKind kind) -> ValueFactory {
    return [kind](SchemaBuilder& builder) {
      return builder.createScalarTypeBuilder(kind);
    };
  };
  const auto array = [](ScalarKind kind) -> ValueFactory {
    return [kind](SchemaBuilder& builder) {
      auto value = builder.createArrayTypeBuilder();
      value->setChildren(builder.createScalarTypeBuilder(kind));
      return value;
    };
  };
  const auto row = [](std::vector<std::pair<std::string, ScalarKind>> fields)
      -> ValueFactory {
    return [fields = std::move(fields)](SchemaBuilder& builder) {
      auto value = builder.createRowTypeBuilder(fields.size());
      for (const auto& [name, kind] : fields) {
        value->addChild(name, builder.createScalarTypeBuilder(kind));
      }
      return value;
    };
  };
  const auto flatMap = [](ScalarKind keyKind) -> ValueFactory {
    return [keyKind](SchemaBuilder& builder) {
      auto value = builder.createFlatMapTypeBuilder(keyKind);
      value->addChild(
          "key", builder.createScalarTypeBuilder(ScalarKind::Int64));
      return value;
    };
  };
  struct MismatchCase {
    std::string_view name;
    ValueFactory configured;
    ValueFactory defaultValue;
  };
  const std::vector<MismatchCase> mismatches{
      {"kind", scalar(ScalarKind::Int64), array(ScalarKind::Int64)},
      {"row arity",
       row({{"x", ScalarKind::Int64}}),
       row({{"x", ScalarKind::Int64}, {"y", ScalarKind::Int64}})},
      {"row name",
       row({{"x", ScalarKind::Int64}}),
       row({{"y", ScalarKind::Int64}})},
      {"row child type",
       row({{"x", ScalarKind::Int64}}),
       row({{"x", ScalarKind::String}})},
      {"FlatMap key kind",
       flatMap(ScalarKind::String),
       flatMap(ScalarKind::Int32)},
  };
  for (const auto& mismatch : mismatches) {
    SCOPED_TRACE(mismatch.name);
    SchemaBuilder builder;
    auto hybridMap = builder.createHybridFlatMapTypeBuilder(ScalarKind::String);
    hybridMap->addGroup(0, {"a"}, mismatch.configured(builder));
    hybridMap->addGroup(
        HybridFlatMap::kDefaultGroupId, {}, mismatch.defaultValue(builder));

    NIMBLE_ASSERT_THROW(
        builder.schemaNodes(),
        "Hybrid FlatMap groups must have the same logical value type");
  }
}

TEST(SchemaBuilderTest, hybridFlatMapReaderValidatesGroupMetadata) {
  const auto makeNodes = [](const HybridFlatMap& metadata) {
    std::vector<SchemaNode> nodes;
    nodes.emplace_back(
        Kind::HybridFlatMap,
        0,
        ScalarKind::String,
        std::nullopt,
        metadata.groups.size(),
        hybridFlatMapAttributes(metadata));
    for (size_t i = 0; i < nodes.front().childrenCount(); ++i) {
      nodes.emplace_back(Kind::Scalar, 1 + 3 * i, ScalarKind::String);
      nodes.emplace_back(Kind::Scalar, 2 + 3 * i, ScalarKind::Bool);
      nodes.emplace_back(Kind::Scalar, 3 + 3 * i, ScalarKind::Int64);
    }
    return nodes;
  };

  NIMBLE_ASSERT_THROW(
      SchemaReader::getSchema(makeNodes(HybridFlatMap{})),
      "Hybrid FlatMap requires at least two groups");

  NIMBLE_ASSERT_THROW(
      SchemaReader::getSchema(makeNodes(
          HybridFlatMap{
              .groups =
                  {
                      {.groupId = HybridFlatMap::kDefaultGroupId,
                       .groupKeys = {}},
                  },
          })),
      "Hybrid FlatMap requires at least two groups");

  NIMBLE_ASSERT_THROW(
      SchemaReader::getSchema(makeNodes(
          HybridFlatMap{
              .groups =
                  {
                      {.groupId = HybridFlatMap::kDefaultGroupId,
                       .groupKeys = {}},
                      {.groupId = HybridFlatMap::kDefaultGroupId,
                       .groupKeys = {}},
                  },
          })),
      "Duplicate Hybrid FlatMap group ID: 4294967295");

  NIMBLE_ASSERT_THROW(
      SchemaReader::getSchema(makeNodes(
          HybridFlatMap{
              .groups =
                  {
                      {.groupId = 0, .groupKeys = {"a"}},
                      {.groupId = 1, .groupKeys = {"a"}},
                      {.groupId = HybridFlatMap::kDefaultGroupId,
                       .groupKeys = {}},
                  },
          })),
      "Duplicate Hybrid FlatMap key: 'a'");

  NIMBLE_ASSERT_THROW(
      SchemaReader::getSchema(makeNodes(
          HybridFlatMap{
              .groups =
                  {
                      {.groupId = 0, .groupKeys = {"a"}},
                      {.groupId = 0, .groupKeys = {"b"}},
                      {.groupId = HybridFlatMap::kDefaultGroupId,
                       .groupKeys = {}},
                  },
          })),
      "Duplicate Hybrid FlatMap group ID: 0");

  NIMBLE_ASSERT_THROW(
      SchemaReader::getSchema(makeNodes(
          HybridFlatMap{
              .groups =
                  {
                      {.groupId = 0, .groupKeys = {"a"}},
                      {.groupId = HybridFlatMap::kDefaultGroupId,
                       .groupKeys = {"default"}},
                  },
          })),
      "Hybrid FlatMap Default group cannot contain group keys");

  NIMBLE_ASSERT_THROW(
      SchemaReader::getSchema(makeNodes(
          HybridFlatMap{
              .groups =
                  {
                      {.groupId = 0, .groupKeys = {}},
                      {.groupId = HybridFlatMap::kDefaultGroupId,
                       .groupKeys = {}},
                  },
          })),
      "Hybrid FlatMap group must contain at least one key: 0");

  NIMBLE_ASSERT_THROW(
      SchemaReader::getSchema(makeNodes(
          HybridFlatMap{
              .groups =
                  {
                      {.groupId = 0, .groupKeys = {""}},
                      {.groupId = HybridFlatMap::kDefaultGroupId,
                       .groupKeys = {}},
                  },
          })),
      "Hybrid FlatMap key cannot be empty");
}

TEST(SchemaBuilderTest, hybridFlatMapReaderRejectsMismatchedValueShapes) {
  using ValueFactory = std::function<std::shared_ptr<const Type>()>;
  const auto scalar = [](ScalarKind kind) -> ValueFactory {
    return [kind]() {
      return std::make_shared<ScalarType>(StreamDescriptor{10, kind});
    };
  };
  const auto array = [](ScalarKind kind) -> ValueFactory {
    return [kind]() {
      return std::make_shared<ArrayType>(
          StreamDescriptor{11, ScalarKind::UInt32},
          std::make_shared<ScalarType>(StreamDescriptor{12, kind}));
    };
  };
  const auto row = [](std::vector<std::pair<std::string, ScalarKind>> fields)
      -> ValueFactory {
    return [fields = std::move(fields)]() {
      std::vector<std::string> names;
      std::vector<std::shared_ptr<const Type>> children;
      names.reserve(fields.size());
      children.reserve(fields.size());
      for (const auto& [name, kind] : fields) {
        names.push_back(name);
        children.push_back(
            std::make_shared<ScalarType>(StreamDescriptor{13, kind}));
      }
      return std::make_shared<RowType>(
          StreamDescriptor{14, ScalarKind::Bool},
          std::move(names),
          std::move(children));
    };
  };
  const auto flatMap = [](ScalarKind keyKind) -> ValueFactory {
    return [keyKind]() {
      std::vector<std::unique_ptr<StreamDescriptor>> inMapDescriptors;
      inMapDescriptors.push_back(
          std::make_unique<StreamDescriptor>(15, ScalarKind::Bool));
      std::vector<std::shared_ptr<const Type>> children;
      children.push_back(
          std::make_shared<ScalarType>(
              StreamDescriptor{16, ScalarKind::Int64}));
      return std::make_shared<FlatMapType>(
          StreamDescriptor{17, ScalarKind::Bool},
          keyKind,
          std::vector<std::string>{"key"},
          std::move(inMapDescriptors),
          std::move(children));
    };
  };
  struct MismatchCase {
    std::string_view name;
    ValueFactory configured;
    ValueFactory defaultValue;
  };
  const std::vector<MismatchCase> mismatches{
      {"kind", scalar(ScalarKind::Int64), array(ScalarKind::Int64)},
      {"row arity",
       row({{"x", ScalarKind::Int64}}),
       row({{"x", ScalarKind::Int64}, {"y", ScalarKind::Int64}})},
      {"row name",
       row({{"x", ScalarKind::Int64}}),
       row({{"y", ScalarKind::Int64}})},
      {"row child type",
       row({{"x", ScalarKind::Int64}}),
       row({{"x", ScalarKind::String}})},
      {"FlatMap key kind",
       flatMap(ScalarKind::String),
       flatMap(ScalarKind::Int32)},
  };
  for (const auto& mismatch : mismatches) {
    SCOPED_TRACE(mismatch.name);
    NIMBLE_ASSERT_THROW(
        std::make_shared<HybridFlatMapType>(
            StreamDescriptor{0, ScalarKind::Bool},
            ScalarKind::String,
            std::vector<HybridFlatMapType::Group>{
                {.groupId = 0,
                 .groupKeys = {"a"},
                 .keyDescriptor = StreamDescriptor{1, ScalarKind::String},
                 .inMapDescriptor = StreamDescriptor{2, ScalarKind::Bool},
                 .valueType = mismatch.configured()},
                {.groupId = HybridFlatMap::kDefaultGroupId,
                 .groupKeys = {},
                 .keyDescriptor = StreamDescriptor{3, ScalarKind::String},
                 .inMapDescriptor = StreamDescriptor{4, ScalarKind::Bool},
                 .valueType = mismatch.defaultValue()},
            }),
        "Hybrid FlatMap groups must have the same logical value type");
  }
}

TEST(SchemaBuilderTest, hybridFlatMapReaderRejectsMalformedPhysicalGroups) {
  const HybridFlatMap metadata{
      .groups =
          {
              {.groupId = 0, .groupKeys = {"a"}},
              {.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {}},
          },
  };
  const auto makeNodes = [&](ScalarKind parentKeyKind,
                             ScalarKind keysKind,
                             ScalarKind inMapKind,
                             std::optional<std::string> valueName,
                             size_t childCount = 2) {
    return std::vector<SchemaNode>{
        SchemaNode{
            Kind::HybridFlatMap,
            0,
            parentKeyKind,
            std::nullopt,
            childCount,
            hybridFlatMapAttributes(metadata)},
        SchemaNode{Kind::Scalar, 1, keysKind},
        SchemaNode{Kind::Scalar, 2, inMapKind},
        SchemaNode{Kind::Scalar, 3, ScalarKind::Int64, std::move(valueName)},
        SchemaNode{Kind::Scalar, 4, parentKeyKind},
        SchemaNode{Kind::Scalar, 5, ScalarKind::Bool},
        SchemaNode{Kind::Scalar, 6, ScalarKind::Int64},
    };
  };

  NIMBLE_ASSERT_THROW(
      SchemaReader::getSchema(
          {SchemaNode{Kind::HybridFlatMap, 0, ScalarKind::String}}),
      "Hybrid FlatMap metadata attribute is missing");
  NIMBLE_ASSERT_THROW(
      SchemaReader::getSchema(makeNodes(
          ScalarKind::String,
          ScalarKind::String,
          ScalarKind::Bool,
          std::nullopt,
          1)),
      "Hybrid FlatMap group metadata must match its child count");
  NIMBLE_ASSERT_THROW(
      SchemaReader::getSchema(makeNodes(
          ScalarKind::String,
          ScalarKind::Int64,
          ScalarKind::Bool,
          std::nullopt)),
      "Hybrid FlatMap keys stream must match its key type");
  NIMBLE_ASSERT_THROW(
      SchemaReader::getSchema(makeNodes(
          ScalarKind::String,
          ScalarKind::String,
          ScalarKind::Int64,
          std::nullopt)),
      "Hybrid FlatMap in-map stream must be boolean");
  NIMBLE_ASSERT_THROW(
      SchemaReader::getSchema(makeNodes(
          ScalarKind::String,
          ScalarKind::String,
          ScalarKind::Bool,
          "named-value")),
      "Hybrid FlatMap group value child must be unnamed");
  NIMBLE_ASSERT_THROW(
      SchemaReader::getSchema(makeNodes(
          ScalarKind::Undefined,
          ScalarKind::Undefined,
          ScalarKind::Bool,
          std::nullopt)),
      "Hybrid FlatMap key kind is unsupported: Undefined");

  auto mismatchedValues = makeNodes(
      ScalarKind::String, ScalarKind::String, ScalarKind::Bool, std::nullopt);
  mismatchedValues.back() = SchemaNode{Kind::Scalar, 6, ScalarKind::String};
  NIMBLE_ASSERT_THROW(
      SchemaReader::getSchema(mismatchedValues),
      "Hybrid FlatMap groups must have the same logical value type");
}

} // namespace
} // namespace facebook::nimble
