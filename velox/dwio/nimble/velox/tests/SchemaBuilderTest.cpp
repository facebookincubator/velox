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

#include <type_traits>

#include "velox/dwio/nimble/velox/SchemaBuilder.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"

namespace facebook::nimble {
namespace {

static_assert(!std::is_base_of_v<FlatMapTypeBuilder, HybridFlatMapTypeBuilder>);
static_assert(!std::is_base_of_v<FlatMapType, HybridFlatMapType>);

TEST(SchemaBuilderTest, setAttributeReplacesInPlaceAndPreservesOthers) {
  SchemaBuilder builder;
  auto type = builder.createScalarTypeBuilder(ScalarKind::Int64);
  type->setAttributes({{"first", "1"}, {"replace", "old"}});

  type->setAttribute("replace", "new");
  type->setAttribute("last", "3");

  EXPECT_EQ(
      type->attributes(),
      (std::vector<std::pair<std::string, std::string>>{
          {"first", "1"}, {"replace", "new"}, {"last", "3"}}));
}

TEST(SchemaBuilderTest, hybridFlatMapStreamsRemainOutsideLogicalChildren) {
  SchemaBuilder builder;
  auto root = builder.createRowTypeBuilder(1);
  auto flatMap = builder.createHybridFlatMapTypeBuilder(ScalarKind::Int32);
  EXPECT_EQ(&flatMap->asHybridFlatMap(), flatMap.get());
  EXPECT_EQ(flatMap->kind(), Kind::HybridFlatMap);
  auto templateValue = builder.createScalarTypeBuilder(ScalarKind::Int64);
  flatMap->setValueTemplate(templateValue);

  auto groupValue = builder.createScalarTypeBuilder(ScalarKind::Int64);
  const auto nodeCountBeforeGroup = builder.nodeCount();
  const auto streams = flatMap->addGroup(groupValue);
  root->addChild("features", flatMap);

  EXPECT_EQ(builder.nodeCount(), nodeCountBeforeGroup + 2);
  EXPECT_EQ(streams.keyHasEntries.scalarKind(), ScalarKind::Bool);
  EXPECT_EQ(streams.inMap.scalarKind(), ScalarKind::Bool);
  EXPECT_NE(streams.keyHasEntries.offset(), streams.inMap.offset());
  EXPECT_NE(
      streams.keyHasEntries.offset(), groupValue->scalarDescriptor().offset());
  EXPECT_NE(streams.inMap.offset(), groupValue->scalarDescriptor().offset());
  ASSERT_EQ(flatMap->groupCount(), 1);
  const auto storedStreams = flatMap->groupStreamDescriptorsAt(0);
  EXPECT_EQ(
      storedStreams.keyHasEntries.offset(), streams.keyHasEntries.offset());
  EXPECT_EQ(storedStreams.inMap.offset(), streams.inMap.offset());
  EXPECT_EQ(&flatMap->groupValuesAt(0), groupValue.get());

  const auto nodes = builder.schemaNodes();
  ASSERT_EQ(nodes.size(), 3);
  EXPECT_EQ(nodes[1].kind(), Kind::HybridFlatMap);
  EXPECT_EQ(nodes[1].childrenCount(), 1);
  EXPECT_EQ(nodes[1].scalarKind(), ScalarKind::Int32);
  EXPECT_FALSE(nodes[2].name().has_value());

  const auto schema = SchemaReader::getSchema(nodes);
  const auto& child = schema->asRow().childAt(0);
  EXPECT_FALSE(child->isFlatMap());
  EXPECT_TRUE(child->isHybridFlatMap());
  const auto& hybridMapType = child->asHybridFlatMap();
  EXPECT_EQ(hybridMapType.kind(), Kind::HybridFlatMap);
  EXPECT_EQ(hybridMapType.keyScalarKind(), ScalarKind::Int32);
  EXPECT_EQ(
      hybridMapType.valueTemplate()->asScalar().scalarDescriptor().offset(),
      templateValue->scalarDescriptor().offset());

  std::vector<offset_size> valueOffsets;
  EXPECT_FALSE(visitValueStreamLeaves(*child, [&](offset_size offset) {
    valueOffsets.push_back(offset);
    return false;
  }));
  EXPECT_TRUE(valueOffsets.empty());

  std::vector<offset_size> presenceOffsets;
  EXPECT_FALSE(visitPresenceStreamOffsets(*child, [&](offset_size offset) {
    presenceOffsets.push_back(offset);
    return false;
  }));
  EXPECT_EQ(
      presenceOffsets,
      std::vector<offset_size>{hybridMapType.nullsDescriptor().offset()});
}

TEST(SchemaBuilderTest, hybridFlatMapRequiresOneUnnamedValueTemplate) {
  std::vector<SchemaNode> missingTemplate{
      SchemaNode{Kind::HybridFlatMap, 0, ScalarKind::Int32, std::nullopt, 0},
  };
  EXPECT_THROW(SchemaReader::getSchema(missingTemplate), NimbleInternalError);

  std::vector<SchemaNode> namedTemplate{
      SchemaNode{Kind::HybridFlatMap, 0, ScalarKind::Int32, std::nullopt, 1},
      SchemaNode{
          Kind::Scalar,
          1,
          ScalarKind::Int64,
          std::optional<std::string>{"unexpected"}},
  };
  EXPECT_THROW(SchemaReader::getSchema(namedTemplate), NimbleInternalError);
}

} // namespace
} // namespace facebook::nimble
