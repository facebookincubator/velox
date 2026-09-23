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
#include "velox/dwio/nimble/velox/SchemaGenerated.h"

namespace facebook::nimble {
namespace {

using testing::ElementsAre;

HybridFlatMap metadata() {
  return HybridFlatMap{
      .groups =
          {
              {.groupId = 7,
               .groupKeys =
                   {"feature,with:delimiters|and;slashes/",
                    std::string{"prefix\0suffix", 13}}},
              {.groupId = 11, .groupKeys = {"C"}},
              {.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {}},
          },
  };
}

std::string encodeRawMetadata(
    const std::vector<uint32_t>& groupIds,
    const std::vector<uint32_t>& groupKeyCounts,
    const std::vector<std::string>& groupKeys) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<flatbuffers::String>> keyOffsets;
  keyOffsets.reserve(groupKeys.size());
  for (const auto& key : groupKeys) {
    keyOffsets.push_back(builder.CreateString(key));
  }
  const auto serialized = serialization::CreateHybridFlatMap(
      builder,
      builder.CreateVector(groupIds),
      builder.CreateVector(groupKeyCounts),
      builder.CreateVector(keyOffsets));
  builder.Finish(serialized);
  return {
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
}

TEST(HybridFlatMapTest, supportedKeyKinds) {
  for (const auto kind : {
           ScalarKind::Int8,
           ScalarKind::Int16,
           ScalarKind::Int32,
           ScalarKind::Int64,
           ScalarKind::String,
       }) {
    SCOPED_TRACE(kind);
    EXPECT_TRUE(HybridFlatMap::supportedKeyKind(kind));
  }
  for (const auto kind : {
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
    SCOPED_TRACE(kind);
    EXPECT_FALSE(HybridFlatMap::supportedKeyKind(kind));
  }
}

TEST(HybridFlatMapTest, metadataRoundTripPreservesFlattenedOrderAndBinaryKeys) {
  const auto expected = metadata();
  const auto serialized = expected.serialize();
  flatbuffers::Verifier verifier{
      reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size()};
  ASSERT_TRUE(verifier.VerifyBuffer<serialization::HybridFlatMap>());

  const auto* flat =
      flatbuffers::GetRoot<serialization::HybridFlatMap>(serialized.data());
  ASSERT_NE(flat->group_ids(), nullptr);
  EXPECT_THAT(
      *flat->group_ids(), ElementsAre(7, 11, HybridFlatMap::kDefaultGroupId));
  ASSERT_NE(flat->group_key_counts(), nullptr);
  EXPECT_THAT(*flat->group_key_counts(), ElementsAre(2, 1, 0));
  ASSERT_NE(flat->group_keys(), nullptr);
  ASSERT_EQ(flat->group_keys()->size(), 3);
  EXPECT_EQ(flat->group_keys()->Get(0)->str(), expected.groups[0].groupKeys[0]);
  EXPECT_EQ(flat->group_keys()->Get(1)->str(), expected.groups[0].groupKeys[1]);
  EXPECT_EQ(flat->group_keys()->Get(1)->size(), 13);
  EXPECT_EQ(flat->group_keys()->Get(2)->str(), "C");

  EXPECT_EQ(HybridFlatMap::deserialize(serialized), expected);
}

TEST(HybridFlatMapTest, rejectsOmittedVectors) {
  flatbuffers::FlatBufferBuilder builder;
  serialization::HybridFlatMapBuilder metadataBuilder{builder};
  const auto serialized = metadataBuilder.Finish();
  builder.Finish(serialized);
  const std::string_view bytes{
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};

  NIMBLE_ASSERT_THROW(
      HybridFlatMap::deserialize(bytes),
      "Hybrid FlatMap group IDs are missing");
}

TEST(HybridFlatMapTest, rejectsMalformedMetadata) {
  NIMBLE_ASSERT_THROW(
      HybridFlatMap::deserialize("malformed"),
      "Hybrid FlatMap metadata attribute is malformed");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap::deserialize(encodeRawMetadata({0, 1}, {1}, {})),
      "Hybrid FlatMap group IDs and key counts must have the same size");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap::deserialize(encodeRawMetadata({0, 1}, {1, 1}, {"a"})),
      "Hybrid FlatMap group key counts must match group keys size");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap::deserialize(
          encodeRawMetadata({0, 1}, {0, 0}, {"unexpected"})),
      "Hybrid FlatMap group key counts must match group keys size");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap::deserialize(encodeRawMetadata(
          {0, 1, HybridFlatMap::kDefaultGroupId}, {0, 0, 0}, {})),
      "Hybrid FlatMap group count must not exceed group key count plus one");
}

TEST(HybridFlatMapTest, deserializeEnforcesGroupContract) {
  const auto defaultGroupId = HybridFlatMap::kDefaultGroupId;

  NIMBLE_ASSERT_THROW(
      HybridFlatMap::deserialize(encodeRawMetadata({}, {}, {})),
      "Hybrid FlatMap requires at least two groups");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap::deserialize(encodeRawMetadata({defaultGroupId}, {0}, {})),
      "Hybrid FlatMap requires at least two groups");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap::deserialize(encodeRawMetadata({0, 0}, {1, 1}, {"a", "b"})),
      "Duplicate Hybrid FlatMap group ID: 0");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap::deserialize(
          encodeRawMetadata({defaultGroupId, defaultGroupId}, {0, 1}, {"a"})),
      "Duplicate Hybrid FlatMap group ID: 4294967295");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap::deserialize(encodeRawMetadata({0, 1}, {1, 1}, {"a", "b"})),
      "Hybrid FlatMap must have exactly one Default group");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap::deserialize(
          encodeRawMetadata({0, defaultGroupId}, {1, 1}, {"a", "b"})),
      "Hybrid FlatMap Default group cannot contain group keys");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap::deserialize(
          encodeRawMetadata({0, defaultGroupId}, {0, 1}, {"a"})),
      "Hybrid FlatMap group must contain at least one key: 0");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap::deserialize(
          encodeRawMetadata({0, defaultGroupId}, {1, 0}, {""})),
      "Hybrid FlatMap key cannot be empty");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap::deserialize(
          encodeRawMetadata({0, 1, defaultGroupId}, {1, 1, 0}, {"a", "a"})),
      "Duplicate Hybrid FlatMap key: 'a'");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap::deserialize(
          encodeRawMetadata({0, defaultGroupId}, {2, 0}, {"b", "a"})),
      "Hybrid FlatMap group keys must be sorted: 0");
}

TEST(HybridFlatMapTest, setMetadataAttributeAppendsToUserAttributes) {
  std::vector<std::pair<std::string, std::string>> attributes{
      {"user.a", "1"},
      {"user.b", "2"},
  };
  const auto expected = metadata();

  expected.setAttribute(attributes);

  ASSERT_EQ(attributes.size(), 3);
  EXPECT_EQ(
      attributes[0], (std::pair<std::string, std::string>{"user.a", "1"}));
  EXPECT_EQ(
      attributes[1], (std::pair<std::string, std::string>{"user.b", "2"}));
  EXPECT_EQ(attributes[2].first, HybridFlatMap::kAttributeName);
  EXPECT_EQ(HybridFlatMap::deserialize(attributes[2].second), expected);
}

TEST(HybridFlatMapTest, setMetadataAttributeRejectsExisting) {
  std::vector<std::pair<std::string, std::string>> attributes{
      {"user.a", "1"},
      {std::string{HybridFlatMap::kAttributeName}, "stale"},
      {"user.b", "2"},
  };
  const auto before = attributes;

  NIMBLE_ASSERT_THROW(
      metadata().setAttribute(attributes),
      "Hybrid FlatMap metadata attribute already exists");
  EXPECT_EQ(attributes, before);
}

TEST(HybridFlatMapTest, getAttributeLeavesAttributesUnchanged) {
  const auto expected = metadata();
  const auto serialized = expected.serialize();
  std::vector<std::pair<std::string, std::string>> attributes{
      {"user.a", "1"},
      {std::string{HybridFlatMap::kAttributeName}, serialized},
      {"user.b", "2"},
  };

  EXPECT_EQ(HybridFlatMap::getAttribute(attributes), expected);
  EXPECT_EQ(
      attributes,
      (std::vector<std::pair<std::string, std::string>>{
          {"user.a", "1"},
          {std::string{HybridFlatMap::kAttributeName}, serialized},
          {"user.b", "2"},
      }));
}

TEST(HybridFlatMapTest, extractMetadataAttributePreservesUserAttributes) {
  const auto expected = metadata();
  std::vector<std::pair<std::string, std::string>> attributes{
      {"user.a", "1"},
      {std::string{HybridFlatMap::kAttributeName}, expected.serialize()},
      {"user.b", "2"},
  };

  EXPECT_EQ(HybridFlatMap::extractAttribute(attributes), expected);
  EXPECT_EQ(
      attributes,
      (std::vector<std::pair<std::string, std::string>>{
          {"user.a", "1"}, {"user.b", "2"}}));
}

TEST(HybridFlatMapTest, extractFailureLeavesAttributesUnchanged) {
  {
    std::vector<std::pair<std::string, std::string>> attributes{
        {"user.a", "1"}, {"user.b", "2"}};
    const auto before = attributes;
    NIMBLE_ASSERT_THROW(
        HybridFlatMap::extractAttribute(attributes),
        "Hybrid FlatMap metadata attribute is missing");
    EXPECT_EQ(attributes, before);
  }
  {
    const auto serialized = metadata().serialize();
    std::vector<std::pair<std::string, std::string>> attributes{
        {"user.a", "1"},
        {std::string{HybridFlatMap::kAttributeName}, serialized},
        {"user.b", "2"},
        {std::string{HybridFlatMap::kAttributeName}, serialized},
    };
    const auto before = attributes;
    NIMBLE_ASSERT_THROW(
        HybridFlatMap::extractAttribute(attributes),
        "Hybrid FlatMap metadata attribute must be unique");
    EXPECT_EQ(attributes, before);
  }
  {
    std::vector<std::pair<std::string, std::string>> attributes{
        {"user.a", "1"},
        {std::string{HybridFlatMap::kAttributeName}, "malformed"},
        {"user.b", "2"},
    };
    const auto before = attributes;
    NIMBLE_ASSERT_THROW(
        HybridFlatMap::extractAttribute(attributes),
        "Hybrid FlatMap metadata attribute is malformed");
    EXPECT_EQ(attributes, before);
  }
}

} // namespace
} // namespace facebook::nimble
