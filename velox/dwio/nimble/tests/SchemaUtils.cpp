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
#include <optional>

#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/dwio/nimble/tests/SchemaUtils.h"

namespace facebook::nimble::test {
void verifySchemaNodes(
    const std::vector<nimble::SchemaNode>& nodes,
    std::vector<nimble::SchemaNode> expected) {
  ASSERT_EQ(expected.size(), nodes.size());
  for (auto i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(expected[i].kind(), nodes[i].kind()) << "i = " << i;
    EXPECT_EQ(expected[i].offset(), nodes[i].offset()) << "i = " << i;
    EXPECT_EQ(expected[i].name(), nodes[i].name()) << "i = " << i;
    EXPECT_EQ(expected[i].childrenCount(), nodes[i].childrenCount())
        << "i = " << i;
    EXPECT_EQ(expected[i].scalarKind(), nodes[i].scalarKind()) << "i = " << i;
  }
}

void compareSchema(
    uint32_t& index,
    const std::vector<nimble::SchemaNode>& nodes,
    const std::shared_ptr<const nimble::Type>& type,
    std::optional<std::string> name = std::nullopt) {
  const auto& node = nodes[index++];
  EXPECT_EQ(node.name().has_value(), name.has_value());
  if (name.has_value()) {
    EXPECT_EQ(name.value(), node.name().value());
  }

  EXPECT_EQ(type->kind(), node.kind());

  switch (type->kind()) {
    case nimble::Kind::Scalar: {
      auto& scalar = type->asScalar();
      EXPECT_EQ(scalar.scalarDescriptor().offset(), node.offset());
      EXPECT_EQ(scalar.scalarDescriptor().scalarKind(), node.scalarKind());
      EXPECT_EQ(0, node.childrenCount());
      break;
    }
    case nimble::Kind::TimestampMicroNano: {
      auto& timestamp = type->asTimestampMicroNano();
      EXPECT_EQ(timestamp.microsDescriptor().offset(), node.offset());
      EXPECT_EQ(nimble::ScalarKind::Int64, node.scalarKind());
      EXPECT_EQ(0, node.childrenCount());

      const auto& nanosNode = nodes[index++];
      EXPECT_FALSE(nanosNode.name().has_value());
      EXPECT_EQ(Kind::Scalar, nanosNode.kind());
      EXPECT_EQ(ScalarKind::UInt16, nanosNode.scalarKind());
      EXPECT_EQ(timestamp.nanosDescriptor().offset(), nanosNode.offset());
      break;
    }
    case nimble::Kind::Row: {
      auto& row = type->asRow();
      EXPECT_EQ(row.nullsDescriptor().offset(), node.offset());
      EXPECT_EQ(nimble::ScalarKind::Bool, node.scalarKind());
      EXPECT_EQ(row.childrenCount(), node.childrenCount());

      for (auto i = 0; i < row.childrenCount(); ++i) {
        compareSchema(index, nodes, row.childAt(i), row.nameAt(i));
      }

      break;
    }
    case nimble::Kind::Array: {
      auto& array = type->asArray();
      EXPECT_EQ(array.lengthsDescriptor().offset(), node.offset());
      EXPECT_EQ(nimble::ScalarKind::UInt32, node.scalarKind());
      EXPECT_EQ(0, node.childrenCount());

      compareSchema(index, nodes, array.elements());

      break;
    }
    case nimble::Kind::ArrayWithOffsets: {
      auto& arrayWithOffsets = type->asArrayWithOffsets();
      EXPECT_EQ(arrayWithOffsets.lengthsDescriptor().offset(), node.offset());
      EXPECT_EQ(nimble::ScalarKind::UInt32, node.scalarKind());
      EXPECT_EQ(0, node.childrenCount());

      const auto& offsetNode = nodes[index++];
      EXPECT_FALSE(offsetNode.name().has_value());
      EXPECT_EQ(Kind::Scalar, offsetNode.kind());
      EXPECT_EQ(ScalarKind::UInt32, offsetNode.scalarKind());
      EXPECT_EQ(
          arrayWithOffsets.offsetsDescriptor().offset(), offsetNode.offset());

      compareSchema(index, nodes, arrayWithOffsets.elements());

      break;
    }
    case nimble::Kind::Map: {
      auto& map = type->asMap();
      EXPECT_EQ(map.lengthsDescriptor().offset(), node.offset());
      EXPECT_EQ(nimble::ScalarKind::UInt32, node.scalarKind());
      EXPECT_EQ(0, node.childrenCount());

      compareSchema(index, nodes, map.keys());
      compareSchema(index, nodes, map.values());

      break;
    }
    case nimble::Kind::SlidingWindowMap: {
      auto& map = type->asSlidingWindowMap();
      EXPECT_EQ(map.offsetsDescriptor().offset(), node.offset());
      EXPECT_EQ(nimble::ScalarKind::UInt32, node.scalarKind());
      EXPECT_EQ(0, node.childrenCount());

      const auto& lengthNode = nodes[index++];
      EXPECT_FALSE(lengthNode.name().has_value());
      EXPECT_EQ(Kind::Scalar, lengthNode.kind());
      EXPECT_EQ(ScalarKind::UInt32, lengthNode.scalarKind());
      EXPECT_EQ(map.lengthsDescriptor().offset(), lengthNode.offset());

      compareSchema(index, nodes, map.keys());
      compareSchema(index, nodes, map.values());

      break;
    }
    case nimble::Kind::FlatMap: {
      auto& map = type->asFlatMap();
      EXPECT_EQ(map.nullsDescriptor().offset(), node.offset());
      EXPECT_EQ(map.keyScalarKind(), node.scalarKind());
      EXPECT_EQ(map.childrenCount(), node.childrenCount());

      for (auto i = 0; i < map.childrenCount(); ++i) {
        const auto& inMapNode = nodes[index++];
        ASSERT_TRUE(inMapNode.name().has_value());
        EXPECT_EQ(map.nameAt(i), inMapNode.name().value());
        EXPECT_EQ(Kind::Scalar, inMapNode.kind());
        EXPECT_EQ(ScalarKind::Bool, inMapNode.scalarKind());
        EXPECT_EQ(map.inMapDescriptorAt(i).offset(), inMapNode.offset());
        compareSchema(index, nodes, map.childAt(i));
      }

      break;
    }
    default:
      FAIL() << "Unknown type kind: " << (int)type->kind();
  }
}

void compareSchema(
    const std::vector<nimble::SchemaNode>& nodes,
    const std::shared_ptr<const nimble::Type>& root) {
  uint32_t index = 0;
  compareSchema(index, nodes, root);
  EXPECT_EQ(nodes.size(), index);
}

std::shared_ptr<nimble::RowTypeBuilder> row(
    nimble::SchemaBuilder& builder,
    std::vector<std::pair<std::string, std::shared_ptr<nimble::TypeBuilder>>>
        children) {
  auto row = builder.createRowTypeBuilder(children.size());
  for (const auto& pair : children) {
    row->addChild(pair.first, pair.second);
  }
  return row;
}

std::shared_ptr<nimble::ArrayTypeBuilder> array(
    nimble::SchemaBuilder& builder,
    std::shared_ptr<nimble::TypeBuilder> elements) {
  auto array = builder.createArrayTypeBuilder();
  array->setChildren(std::move(elements));

  return array;
}

std::shared_ptr<nimble::ArrayWithOffsetsTypeBuilder> arrayWithOffsets(
    nimble::SchemaBuilder& builder,
    std::shared_ptr<nimble::TypeBuilder> elements) {
  auto arrayWithOffsets = builder.createArrayWithOffsetsTypeBuilder();
  arrayWithOffsets->setChildren(std::move(elements));

  return arrayWithOffsets;
}

std::shared_ptr<nimble::MapTypeBuilder> map(
    nimble::SchemaBuilder& builder,
    std::shared_ptr<nimble::TypeBuilder> keys,
    std::shared_ptr<nimble::TypeBuilder> values) {
  auto map = builder.createMapTypeBuilder();
  map->setChildren(std::move(keys), std::move(values));

  return map;
}

std::shared_ptr<nimble::SlidingWindowMapTypeBuilder> slidingWindowMap(
    nimble::SchemaBuilder& builder,
    std::shared_ptr<nimble::TypeBuilder> keys,
    std::shared_ptr<nimble::TypeBuilder> values) {
  auto map = builder.createSlidingWindowMapTypeBuilder();
  map->setChildren(std::move(keys), std::move(values));
  return map;
}

std::shared_ptr<nimble::FlatMapTypeBuilder> flatMap(
    nimble::SchemaBuilder& builder,
    nimble::ScalarKind keyScalarKind,
    std::function<std::shared_ptr<nimble::TypeBuilder>(nimble::SchemaBuilder&)>
        valueFactory,
    FlatMapChildAdder& childAdder) {
  auto map = builder.createFlatMapTypeBuilder(keyScalarKind);
  childAdder.initialize(builder, *map, valueFactory);
  return map;
}

void schema(
    nimble::SchemaBuilder& builder,
    std::function<void(nimble::SchemaBuilder&)> factory) {
  factory(builder);
}

} // namespace facebook::nimble::test
