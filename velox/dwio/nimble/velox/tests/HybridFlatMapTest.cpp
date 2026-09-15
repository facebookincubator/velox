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
#include "velox/dwio/nimble/velox/HybridFlatMap.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/velox/SchemaBuilder.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"

namespace facebook::nimble {
namespace {

using testing::ElementsAre;
using testing::FieldsAre;
using testing::Optional;

TEST(HybridFlatMapTest, metadataIsAbsentByDefault) {
  SchemaBuilder schemaBuilder;
  auto row = schemaBuilder.createRowTypeBuilder(1);
  auto features =
      schemaBuilder.createHybridFlatMapTypeBuilder(ScalarKind::Int64);
  features->setValueTemplate(
      schemaBuilder.createScalarTypeBuilder(ScalarKind::Double));
  row->addChild("features", features);
  const auto schema = SchemaReader::getSchema(schemaBuilder.schemaNodes());

  const auto& hybridMap = schema->asRow().childAt(0)->asHybridFlatMap();
  EXPECT_FALSE(hybridFlatMapLayout(hybridMap).has_value());
  EXPECT_FALSE(hybridFlatMapPhysicalLayout(hybridMap).has_value());
}

TEST(HybridFlatMapTest, metadataRejectsEmptyAttributes) {
  SchemaBuilder schemaBuilder;
  auto row = schemaBuilder.createRowTypeBuilder(1);
  auto features =
      schemaBuilder.createHybridFlatMapTypeBuilder(ScalarKind::Int64);
  features->setValueTemplate(
      schemaBuilder.createScalarTypeBuilder(ScalarKind::Double));
  features->setAttributes({
      {std::string(kHybridFlatMapLogicalGroupingAttribute), ""},
      {std::string(kHybridFlatMapPhysicalStreamsAttribute), ""},
  });
  row->addChild("features", features);
  const auto schema = SchemaReader::getSchema(schemaBuilder.schemaNodes());

  const auto& hybridMap = schema->asRow().childAt(0)->asHybridFlatMap();
  NIMBLE_ASSERT_THROW(
      hybridFlatMapLayout(hybridMap),
      "Invalid hybrid FlatMap layout: empty payload");
  NIMBLE_ASSERT_THROW(
      hybridFlatMapPhysicalLayout(hybridMap),
      "Invalid hybrid FlatMap physical layout: empty payload");
}

TEST(HybridFlatMapTest, emptyLayoutsRoundTrip) {
  const HybridFlatMapLayout layout;
  const auto serializedLayout = serializeHybridFlatMapLayout(layout);
  EXPECT_EQ(serializedLayout, "0;");
  const auto roundTrippedLayout =
      deserializeHybridFlatMapLayout(serializedLayout);
  ASSERT_TRUE(roundTrippedLayout.has_value());
  EXPECT_TRUE(roundTrippedLayout->groups.empty());

  const HybridFlatMapPhysicalLayout physicalLayout;
  const auto serializedPhysicalLayout =
      serializeHybridFlatMapPhysicalLayout(physicalLayout);
  EXPECT_EQ(serializedPhysicalLayout, "all_bitmap;0;");
  const auto roundTrippedPhysicalLayout =
      deserializeHybridFlatMapPhysicalLayout(serializedPhysicalLayout);
  ASSERT_TRUE(roundTrippedPhysicalLayout.has_value());
  EXPECT_TRUE(roundTrippedPhysicalLayout->groups.empty());
}

TEST(HybridFlatMapTest, layoutRejectsMalformedPayload) {
  const std::vector<std::string> malformedPayloads{
      "",
      "0;trailing",
      "groups;",
      "groups;1;",
      "1;0;1;4:key",
      "1;0;1;3:key;trailing",
  };

  for (const auto& payload : malformedPayloads) {
    SCOPED_TRACE(payload);
    NIMBLE_ASSERT_THROW(
        deserializeHybridFlatMapLayout(payload), "Invalid hybrid FlatMap");
  }
}

TEST(HybridFlatMapTest, layoutPreservesDelimiterCharacters) {
  const HybridFlatMapLayout layout{
      .groups = {HybridFlatMapLogicalGroup{
          .groupId = 7,
          .keys =
              {"feature;one-with-a-long-suffix",
               "feature:two-with-a-long-suffix",
               "feature-three-with-a-long-suffix"}}},
  };

  const auto roundTrippedLayout =
      deserializeHybridFlatMapLayout(serializeHybridFlatMapLayout(layout));
  ASSERT_TRUE(roundTrippedLayout.has_value());
  EXPECT_THAT(
      roundTrippedLayout->groups,
      ElementsAre(FieldsAre(
          7,
          ElementsAre(
              "feature;one-with-a-long-suffix",
              "feature:two-with-a-long-suffix",
              "feature-three-with-a-long-suffix"))));
}

TEST(HybridFlatMapTest, layoutRejectsMultipleDefaultGroups) {
  const HybridFlatMapLayout layout{
      .groups =
          {
              HybridFlatMapLogicalGroup{
                  .groupId = kHybridFlatMapDefaultGroupId,
                  .keys = {"1"},
              },
              HybridFlatMapLogicalGroup{
                  .groupId = kHybridFlatMapDefaultGroupId,
                  .keys = {"2"},
              },
          },
  };

  NIMBLE_ASSERT_THROW(
      serializeHybridFlatMapLayout(layout),
      "Found groups with reserved default group ID 4294967295");
}

TEST(HybridFlatMapTest, layoutRoundTripsExplicitGroupIds) {
  const std::string serialized{"2;0;2;1:1;1:2;4294967295;2;3:100;2:99;"};
  const auto layout = deserializeHybridFlatMapLayout(serialized);
  ASSERT_TRUE(layout.has_value());
  EXPECT_THAT(
      layout->groups,
      ElementsAre(
          FieldsAre(0, ElementsAre("1", "2")),
          FieldsAre(kHybridFlatMapDefaultGroupId, ElementsAre("100", "99"))));
  EXPECT_EQ(serializeHybridFlatMapLayout(*layout), serialized);
}

TEST(HybridFlatMapTest, physicalLayoutRoundTrips) {
  const HybridFlatMapPhysicalLayout layout{
      .groups =
          {
              HybridFlatMapGroup{
                  .groupId = 0,
                  .keyHasEntriesStreamOffset = 11,
                  .inMapStreamOffset = 12,
                  .valueStreamOffsets = {13, 14},
              },
              HybridFlatMapGroup{
                  .groupId = kHybridFlatMapDefaultGroupId,
                  .keyHasEntriesStreamOffset = 21,
                  .inMapStreamOffset = 22,
                  .valueStreamOffsets = {23},
              },
          },
  };

  const auto serializedLayout = serializeHybridFlatMapPhysicalLayout(layout);
  EXPECT_EQ(
      serializedLayout, "all_bitmap;2;0;11;12;2;13;14;4294967295;21;22;1;23;");
  const auto roundTrippedLayout =
      deserializeHybridFlatMapPhysicalLayout(serializedLayout);

  ASSERT_TRUE(roundTrippedLayout.has_value());
  EXPECT_THAT(
      roundTrippedLayout->groups,
      ElementsAre(
          FieldsAre(0, 11, 12, ElementsAre(13, 14)),
          FieldsAre(kHybridFlatMapDefaultGroupId, 21, 22, ElementsAre(23))));
}

TEST(HybridFlatMapTest, physicalLayoutRejectsMalformedPayload) {
  const std::vector<std::string> malformedPayloads{
      "",
      "1;",
      "1;0;11;",
      "1;0;11;12;",
      "1;0;11;12;0;",
      "1;0;11;12;2;13;",
      "0;trailing",
      "bitmap;",
      "bitmap;0;trailing",
      "bitmap;0;",
      "all_bitmap;",
      "all_bitmap;0;trailing",
  };

  for (const auto& payload : malformedPayloads) {
    SCOPED_TRACE(payload);
    NIMBLE_ASSERT_THROW(
        deserializeHybridFlatMapPhysicalLayout(payload),
        "Invalid hybrid FlatMap physical layout");
  }
}

TEST(HybridFlatMapTest, selectedKeysRoundTrip) {
  const std::vector<std::string> selectedKeys{"1", "feature:2", "feature;3"};
  const auto serialized = serializeHybridFlatMapSelectedKeys(selectedKeys);
  EXPECT_EQ(serialized, "3;1:1;9:feature:2;9:feature;3;");
  EXPECT_EQ(deserializeHybridFlatMapSelectedKeys(serialized), selectedKeys);

  EXPECT_EQ(serializeHybridFlatMapSelectedKeys({}), "0;");
  EXPECT_TRUE(deserializeHybridFlatMapSelectedKeys("0;").empty());
}

TEST(HybridFlatMapTest, selectedKeysRejectInvalidPayloads) {
  NIMBLE_ASSERT_THROW(
      serializeHybridFlatMapSelectedKeys({"2", "1"}),
      "selected keys must be sorted");
  NIMBLE_ASSERT_THROW(
      serializeHybridFlatMapSelectedKeys({"1", "1"}), "Duplicate key");
  for (const auto& payload : {"", "keys;", "2;1:1;", "1;2:1;"}) {
    SCOPED_TRACE(payload);
    NIMBLE_ASSERT_THROW(
        deserializeHybridFlatMapSelectedKeys(payload),
        "Invalid hybrid FlatMap selected keys");
  }
}

TEST(HybridFlatMapTest, readsOptionalSelectedKeysAttribute) {
  SchemaBuilder schemaBuilder;
  auto row = schemaBuilder.createRowTypeBuilder(1);
  auto hybridMap =
      schemaBuilder.createHybridFlatMapTypeBuilder(ScalarKind::Int64);
  hybridMap->setValueTemplate(
      schemaBuilder.createScalarTypeBuilder(ScalarKind::Double));
  row->addChild("features", hybridMap);

  auto schema = SchemaReader::getSchema(schemaBuilder.schemaNodes());
  EXPECT_FALSE(
      hybridFlatMapSelectedKeys(schema->asRow().childAt(0)->asHybridFlatMap())
          .has_value());

  hybridMap->setAttribute(
      std::string(kHybridFlatMapSelectedKeysAttribute),
      serializeHybridFlatMapSelectedKeys({"1", "2"}));
  schema = SchemaReader::getSchema(schemaBuilder.schemaNodes());
  EXPECT_THAT(
      hybridFlatMapSelectedKeys(schema->asRow().childAt(0)->asHybridFlatMap()),
      Optional(ElementsAre("1", "2")));
}

} // namespace
} // namespace facebook::nimble
