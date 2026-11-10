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

#include "velox/dwio/nimble/common/HybridFlatMap.h"

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <optional>

#include "folly/container/F14Map.h"
#include "velox/dwio/nimble/common/SchemaGenerated.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"

namespace facebook::nimble {
namespace {

using testing::ElementsAre;

HybridFlatMap metadata() {
  return HybridFlatMap::fromSchema(
      {
          {.groupId = 7,
           .groupKeys =
               {"feature,with:delimiters|and;slashes/",
                std::string{"prefix\0suffix", 13}}},
          {.groupId = 11, .groupKeys = {"C"}},
          {.groupId = HybridFlatMap::kDefaultGroupId,
           .groupKeys = {"seen-second", "seen-first"}},
      },
      HybridFlatMap::KeyPresenceMode::kBitmap);
}

std::string encodeRawMetadata(
    const std::vector<uint32_t>& groupIds,
    const std::vector<uint32_t>& groupKeyCounts,
    const std::vector<std::string>& groupKeys,
    std::optional<uint8_t> defaultGroupKeyPresenceMode =
        serialization::HybridFlatMapKeyPresenceMode_Bitmap) {
  flatbuffers::FlatBufferBuilder builder;
  builder.ForceDefaults(true);
  std::vector<flatbuffers::Offset<flatbuffers::String>> keyOffsets;
  keyOffsets.reserve(groupKeys.size());
  for (const auto& key : groupKeys) {
    keyOffsets.push_back(builder.CreateString(key));
  }
  const auto groupIdsOffset = builder.CreateVector(groupIds);
  const auto groupKeyCountsOffset = builder.CreateVector(groupKeyCounts);
  const auto groupKeysOffset = builder.CreateVector(keyOffsets);
  serialization::HybridFlatMapBuilder metadataBuilder{builder};
  metadataBuilder.add_group_ids(groupIdsOffset);
  metadataBuilder.add_group_key_counts(groupKeyCountsOffset);
  metadataBuilder.add_group_keys(groupKeysOffset);
  // Writes the raw byte, since the generated enum cannot hold unknown values.
  if (defaultGroupKeyPresenceMode.has_value()) {
    builder.AddElement<uint8_t>(
        serialization::HybridFlatMap::VT_DEFAULT_GROUP_KEY_PRESENCE_MODE,
        *defaultGroupKeyPresenceMode,
        serialization::HybridFlatMapKeyPresenceMode_Bitmap);
  }
  builder.Finish(metadataBuilder.Finish());
  return {
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
}

std::string encodeMetadataWithoutGroupKeys(
    const std::vector<uint32_t>& groupIds,
    const std::vector<uint32_t>& groupKeyCounts) {
  flatbuffers::FlatBufferBuilder builder;
  builder.ForceDefaults(true);
  builder.Finish(
      serialization::CreateHybridFlatMap(
          builder,
          builder.CreateVector(groupIds),
          builder.CreateVector(groupKeyCounts)));
  return {
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
}

HybridFlatMap defaultOnlyMetadata() {
  return HybridFlatMap::fromSchema(
      {{.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {}}},
      HybridFlatMap::KeyPresenceMode::kBitmap);
}

void validateGroups(const HybridFlatMap& hybridMap, bool hasDefault) {
  folly::F14FastMap<std::string_view, size_t> groupIndices;
  detail::validateHybridFlatMapGroups(
      hybridMap.groups().size(),
      hasDefault,
      [&hybridMap](size_t index) { return hybridMap.groups()[index].groupId; },
      [&hybridMap](size_t index) -> const auto& {
        return hybridMap.groups()[index].groupKeys;
      },
      groupIndices);
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

// Integer keys of `kind`, whose values are the range of T, [min, max]:
// canonical forms pass through; a non-canonical form, a non-integer and a value
// out of range are rejected with the same user error by canonicalKey() and
// parseCanonicalIntegerKey().
template <typename T>
void expectCanonicalIntegerKeys(
    ScalarKind kind,
    const std::string& min,
    const std::string& max,
    const std::string& pastMax) {
  for (const auto& key : {min, std::string{"-1"}, std::string{"0"}, max}) {
    EXPECT_EQ(HybridFlatMap::canonicalKey(kind, key), key);
    EXPECT_EQ(
        HybridFlatMap::parseCanonicalIntegerKey<T>(key),
        static_cast<T>(std::stoll(key)));
  }
  for (const auto& key :
       {std::string{"07"}, std::string{"+7"}, std::string{"-0"}}) {
    const auto message =
        fmt::format("Hybrid FlatMap key '{}' is not canonical.", key);
    NIMBLE_ASSERT_USER_THROW(HybridFlatMap::canonicalKey(kind, key), message);
    NIMBLE_ASSERT_USER_THROW(
        HybridFlatMap::parseCanonicalIntegerKey<T>(key), message);
  }
  for (const auto& key : {std::string{"ten"}, std::string{"1.5"}, pastMax}) {
    const auto message = fmt::format(
        "Hybrid FlatMap key '{}' cannot be parsed as {}.", key, kind);
    NIMBLE_ASSERT_USER_THROW(HybridFlatMap::canonicalKey(kind, key), message);
    NIMBLE_ASSERT_USER_THROW(
        HybridFlatMap::parseCanonicalIntegerKey<T>(key), message);
  }
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap::canonicalKey(kind, ""),
      "Hybrid FlatMap key cannot be empty.");
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap::parseCanonicalIntegerKey<T>(""),
      fmt::format("Hybrid FlatMap key '' cannot be parsed as {}.", kind));
}

TEST(HybridFlatMapTest, canonicalInt8Keys) {
  expectCanonicalIntegerKeys<int8_t>(ScalarKind::Int8, "-128", "127", "128");
}

TEST(HybridFlatMapTest, canonicalInt16Keys) {
  expectCanonicalIntegerKeys<int16_t>(
      ScalarKind::Int16, "-32768", "32767", "32768");
}

TEST(HybridFlatMapTest, canonicalInt32Keys) {
  expectCanonicalIntegerKeys<int32_t>(
      ScalarKind::Int32, "-2147483648", "2147483647", "2147483648");
}

TEST(HybridFlatMapTest, canonicalInt64Keys) {
  expectCanonicalIntegerKeys<int64_t>(
      ScalarKind::Int64,
      "-9223372036854775808",
      "9223372036854775807",
      "9223372036854775808");
}

// A String key is kept as written; only an empty one is rejected.
TEST(HybridFlatMapTest, canonicalStringKeys) {
  EXPECT_EQ(HybridFlatMap::canonicalKey(ScalarKind::String, "07"), "07");
  EXPECT_EQ(HybridFlatMap::canonicalKey(ScalarKind::String, " a,b "), " a,b ");
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap::canonicalKey(ScalarKind::String, ""),
      "Hybrid FlatMap key cannot be empty.");
}

TEST(HybridFlatMapTest, canonicalKeyRejectsUnsupportedKinds) {
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap::canonicalKey(ScalarKind::Double, "1"),
      "Unsupported Hybrid FlatMap key kind");
}

std::vector<HybridFlatMap::Group> config(
    std::vector<std::pair<uint32_t, std::vector<std::string>>> groups) {
  std::vector<HybridFlatMap::Group> configured;
  configured.reserve(groups.size());
  for (auto& [groupId, keys] : groups) {
    configured.push_back({.groupId = groupId, .groupKeys = std::move(keys)});
  }
  return configured;
}

constexpr uint32_t kDefault = HybridFlatMap::kDefaultGroupId;
constexpr auto kBitmap = HybridFlatMap::KeyPresenceMode::kBitmap;
constexpr auto kKeyList = HybridFlatMap::KeyPresenceMode::kKeyList;

// Keys come back canonical and sorted, as strings: 10 before 9.
TEST(HybridFlatMapTest, constructorSortsKeys) {
  EXPECT_EQ(
      HybridFlatMap(
          ScalarKind::Int64,
          config({{0, {"9", "10", "-3"}}, {4, {"2"}}, {kDefault, {}}}),
          kBitmap)
          .groups(),
      config({{0, {"-3", "10", "9"}}, {4, {"2"}}, {kDefault, {}}}));
  EXPECT_EQ(
      HybridFlatMap(
          ScalarKind::String,
          config({{0, {"b", "a"}}, {kDefault, {}}}),
          kBitmap)
          .groups(),
      config({{0, {"a", "b"}}, {kDefault, {}}}));
  // Only the last key is out of order, so the whole group must be checked.
  EXPECT_EQ(
      HybridFlatMap(
          ScalarKind::String,
          config({{0, {"b", "c", "d", "a"}}, {kDefault, {}}}),
          kBitmap)
          .groups(),
      config({{0, {"a", "b", "c", "d"}}, {kDefault, {}}}));
  // A Default-only configuration is valid: Default holds every key.
  EXPECT_EQ(
      HybridFlatMap(ScalarKind::Int32, config({{kDefault, {}}}), kBitmap)
          .groups(),
      config({{kDefault, {}}}));
}

TEST(HybridFlatMapTest, constructorRejectsNonCanonicalKeys) {
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap(
          ScalarKind::Int64,
          config({{0, {"1", "07"}}, {kDefault, {}}}),
          kBitmap),
      "Hybrid FlatMap key '07' is not canonical.");
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap(
          ScalarKind::Int8, config({{0, {"128"}}, {kDefault, {}}}), kBitmap),
      "Hybrid FlatMap key '128' cannot be parsed as");
}

TEST(HybridFlatMapTest, constructorRejectsRepeatedKeys) {
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap(
          ScalarKind::Int64,
          config({{0, {"1", "1"}}, {kDefault, {}}}),
          kBitmap),
      "Duplicate Hybrid FlatMap key: '1'.");
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap(
          ScalarKind::Int64,
          config({{0, {"1"}}, {1, {"1"}}, {kDefault, {}}}),
          kBitmap),
      "Duplicate Hybrid FlatMap key: '1'.");
}

// The first repeated key, in input order, fails before the keys and groups
// after it are checked.
TEST(HybridFlatMapTest, constructorRejectsRepeatedKeyBeforeLaterErrors) {
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap(
          ScalarKind::Int8,
          config({{0, {"1"}}, {1, {"1", "300"}}, {kDefault, {}}}),
          kBitmap),
      "Duplicate Hybrid FlatMap key: '1'.");
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap(ScalarKind::Int64, config({{0, {"1", "1"}}}), kBitmap),
      "Duplicate Hybrid FlatMap key: '1'.");
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap(
          ScalarKind::String,
          config({{0, {"b", "a"}}, {1, {"b", "a"}}, {kDefault, {}}}),
          kBitmap),
      "Duplicate Hybrid FlatMap key: 'b'.");
}

TEST(HybridFlatMapTest, constructorRejectsUnorderedGroupIds) {
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap(
          ScalarKind::Int64,
          config({{1, {"1"}}, {0, {"2"}}, {kDefault, {}}}),
          kBitmap),
      "Hybrid FlatMap group IDs must be in ascending order.");
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap(
          ScalarKind::Int64,
          config({{0, {"1"}}, {0, {"2"}}, {kDefault, {}}}),
          kBitmap),
      "Hybrid FlatMap group IDs must be in ascending order.");
}

TEST(HybridFlatMapTest, constructorRejectsEmptyConfiguredGroups) {
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap(
          ScalarKind::Int64, config({{0, {}}, {kDefault, {}}}), kBitmap),
      "Hybrid FlatMap group must contain at least one key: 0.");
}

TEST(HybridFlatMapTest, constructorRequiresOneDefaultGroupWithoutKeys) {
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap(ScalarKind::Int64, config({{0, {"1"}}}), kBitmap),
      "Hybrid FlatMap must have exactly one Default group.");
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap(
          ScalarKind::Int64,
          config({{0, {"1"}}, {kDefault, {}}, {kDefault, {}}}),
          kBitmap),
      "Hybrid FlatMap has multiple Default groups.");
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap(
          ScalarKind::Int64, config({{0, {"1"}}, {kDefault, {"2"}}}), kBitmap),
      "Hybrid FlatMap Default group keys are appended in first-seen order and "
      "cannot be configured.");
}

TEST(HybridFlatMapTest, constructorRejectsUnsupportedKeyKinds) {
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap(ScalarKind::Double, config({{kDefault, {}}}), kBitmap),
      "Unsupported Hybrid FlatMap key kind");
}

// Schema metadata has no grouped-key index for findGroup().
TEST(HybridFlatMapTest, findGroupNeedsLookup) {
  NIMBLE_ASSERT_THROW(
      metadata().findGroup("a"), "Hybrid FlatMap has no grouped-key index");
  NIMBLE_ASSERT_THROW(
      metadata().numGroupedKeys(), "Hybrid FlatMap has no grouped-key index");
}

// Copies of a HybridFlatMap share its groups and grouped-key index, as copies
// into SerializerOptions::hybridFlatMapColumns do.
TEST(HybridFlatMapTest, copiesShareGroupsBuiltForKeyKind) {
  const HybridFlatMap hybridMap{
      ScalarKind::Int8, config({{0, {"9", "10"}}, {kDefault, {}}}), kKeyList};
  const folly::F14FastMap<std::string, HybridFlatMap> columns{
      {"column", hybridMap}};
  const auto& copy = columns.at("column");
  EXPECT_EQ(&copy.groups(), &hybridMap.groups());
  EXPECT_EQ(copy, hybridMap);
  EXPECT_EQ(copy.keyKind(), ScalarKind::Int8);
  EXPECT_EQ(
      copy.defaultGroupKeyPresenceMode(),
      HybridFlatMap::KeyPresenceMode::kKeyList);
  EXPECT_EQ(copy.findGroup("10"), size_t{0});
  EXPECT_EQ(copy.findGroup("1"), std::nullopt);
  EXPECT_EQ(copy.numGroupedKeys(), size_t{2});
}

TEST(HybridFlatMapTest, identifiesDefaultGroup) {
  EXPECT_TRUE(HybridFlatMap::isDefaultGroup(HybridFlatMap::kDefaultGroupId));
  EXPECT_FALSE(HybridFlatMap::isDefaultGroup(0));
  EXPECT_FALSE(
      HybridFlatMap::isDefaultGroup(HybridFlatMap::kDefaultGroupId - 1));
}

TEST(HybridFlatMapTest, usesKeyPresentListOnlyForKeyListDefault) {
  {
    const HybridFlatMap hybridMap{
        ScalarKind::Int64, config({{kDefault, {}}}), kBitmap};
    EXPECT_FALSE(hybridMap.useKeyPresentList(HybridFlatMap::kDefaultGroupId));
  }
  {
    const HybridFlatMap hybridMap{
        ScalarKind::Int64, config({{kDefault, {}}}), kKeyList};
    EXPECT_TRUE(hybridMap.useKeyPresentList(HybridFlatMap::kDefaultGroupId));
    EXPECT_FALSE(hybridMap.useKeyPresentList(0));
  }
}

TEST(HybridFlatMapTest, formatsKeyPresenceMode) {
  EXPECT_EQ(
      fmt::format("{}", HybridFlatMap::KeyPresenceMode::kBitmap), "Bitmap");
  EXPECT_EQ(
      fmt::format("{}", HybridFlatMap::KeyPresenceMode::kKeyList), "KeyList");
}

TEST(HybridFlatMapTest, getsGroupById) {
  const auto hybridMap = metadata();

  EXPECT_EQ(hybridMap.groupById(7), hybridMap.groups()[0]);
  EXPECT_EQ(hybridMap.groupById(11), hybridMap.groups()[1]);
  EXPECT_EQ(
      hybridMap.groupById(HybridFlatMap::kDefaultGroupId),
      hybridMap.groups()[2]);
  NIMBLE_ASSERT_THROW(
      hybridMap.groupById(8), "Hybrid FlatMap group ID is missing: 8");
}

TEST(HybridFlatMapTest, releaseGroupsMovesGroupsOut) {
  auto hybridMap = metadata();
  const auto expected = hybridMap.groups();

  EXPECT_EQ(std::move(hybridMap).releaseGroups(), expected);
}

// A HybridFlatMap shares its groups with its copies, so it releases a copy of
// them and the copies keep theirs.
TEST(HybridFlatMapTest, releaseGroupsCopiesSharedGroups) {
  const std::vector<HybridFlatMap::Group> groups{
      {.groupId = 0, .groupKeys = {"1", "2"}},
      {.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {}},
  };
  HybridFlatMap hybridMap{ScalarKind::Int64, groups, kBitmap};
  const auto copy = hybridMap;

  EXPECT_EQ(std::move(hybridMap).releaseGroups(), groups);
  EXPECT_EQ(copy.groups(), groups);
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
  EXPECT_THAT(*flat->group_key_counts(), ElementsAre(2, 1, 2));
  ASSERT_NE(flat->group_keys(), nullptr);
  ASSERT_EQ(flat->group_keys()->size(), 5);
  EXPECT_EQ(
      flat->group_keys()->Get(0)->str(), expected.groups()[0].groupKeys[0]);
  EXPECT_EQ(
      flat->group_keys()->Get(1)->str(), expected.groups()[0].groupKeys[1]);
  EXPECT_EQ(flat->group_keys()->Get(1)->size(), 13);
  EXPECT_EQ(flat->group_keys()->Get(2)->str(), "C");
  EXPECT_EQ(flat->group_keys()->Get(3)->str(), "seen-second");
  EXPECT_EQ(flat->group_keys()->Get(4)->str(), "seen-first");

  EXPECT_EQ(HybridFlatMap::deserialize(serialized), expected);
}

TEST(HybridFlatMapTest, metadataRoundTripsDefaultOnly) {
  const auto expected = defaultOnlyMetadata();
  const auto serialized = expected.serialize();
  flatbuffers::Verifier verifier{
      reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size()};
  ASSERT_TRUE(verifier.VerifyBuffer<serialization::HybridFlatMap>());

  const auto* flat =
      flatbuffers::GetRoot<serialization::HybridFlatMap>(serialized.data());
  ASSERT_NE(flat->group_ids(), nullptr);
  EXPECT_THAT(*flat->group_ids(), ElementsAre(HybridFlatMap::kDefaultGroupId));
  ASSERT_NE(flat->group_key_counts(), nullptr);
  EXPECT_THAT(*flat->group_key_counts(), ElementsAre(0));
  ASSERT_NE(flat->group_keys(), nullptr);
  EXPECT_EQ(flat->group_keys()->size(), 0);

  EXPECT_EQ(HybridFlatMap::deserialize(serialized), expected);
}

TEST(HybridFlatMapTest, metadataRoundTripsKeyList) {
  // The key presence mode is written even when it is Bitmap.
  const auto hasKeyPresenceField = [](std::string_view serialized) {
    return flatbuffers::GetRoot<flatbuffers::Table>(serialized.data())
        ->CheckField(
            serialization::HybridFlatMap::VT_DEFAULT_GROUP_KEY_PRESENCE_MODE);
  };
  EXPECT_TRUE(hasKeyPresenceField(metadata().serialize()));

  const auto expected = HybridFlatMap::fromSchema(
      defaultOnlyMetadata().groups(), HybridFlatMap::KeyPresenceMode::kKeyList);
  const auto serialized = expected.serialize();
  flatbuffers::Verifier verifier{
      reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size()};
  ASSERT_TRUE(verifier.VerifyBuffer<serialization::HybridFlatMap>());

  EXPECT_TRUE(hasKeyPresenceField(serialized));
  EXPECT_EQ(
      flatbuffers::GetRoot<serialization::HybridFlatMap>(serialized.data())
          ->default_group_key_presence_mode(),
      serialization::HybridFlatMapKeyPresenceMode_KeyList);
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

TEST(HybridFlatMapTest, deserializeAcceptsOmittedGroupKeysWithoutKeys) {
  EXPECT_EQ(
      HybridFlatMap::deserialize(encodeMetadataWithoutGroupKeys(
          {HybridFlatMap::kDefaultGroupId}, {0})),
      defaultOnlyMetadata());

  NIMBLE_ASSERT_THROW(
      HybridFlatMap::deserialize(encodeMetadataWithoutGroupKeys(
          {7, HybridFlatMap::kDefaultGroupId}, {1, 0})),
      "Hybrid FlatMap group key counts must match group keys size");
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
          {HybridFlatMap::kDefaultGroupId},
          {1},
          {"listed"},
          serialization::HybridFlatMapKeyPresenceMode_KeyList)),
      "Hybrid FlatMap Default group cannot list keys when batches list their "
      "Default keys");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap::deserialize(encodeRawMetadata(
          {HybridFlatMap::kDefaultGroupId},
          {0},
          {},
          /*defaultGroupKeyPresenceMode=*/2)),
      "Hybrid FlatMap Default key presence is unknown");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap::deserialize(encodeRawMetadata(
          {HybridFlatMap::kDefaultGroupId},
          {0},
          {},
          /*defaultGroupKeyPresenceMode=*/std::nullopt)),
      "Hybrid FlatMap Default key presence is missing");
}

TEST(HybridFlatMapTest, projectedConfiguredGroupRequiresKeys) {
  const auto projected = HybridFlatMap::fromSchema(
      {{.groupId = 7, .groupKeys = {}}},
      HybridFlatMap::KeyPresenceMode::kBitmap);
  folly::F14FastMap<std::string_view, size_t> groupIndices;

  NIMBLE_ASSERT_THROW(
      detail::validateHybridFlatMapGroups(
          projected.groups().size(),
          /*hasDefault=*/false,
          [&projected](size_t index) {
            return projected.groups()[index].groupId;
          },
          [&projected](size_t index) -> const auto& {
            return projected.groups()[index].groupKeys;
          },
          groupIndices),
      "Hybrid FlatMap group must contain at least one key: 7");
}

TEST(HybridFlatMapTest, validatesMinimumGroupCount) {
  for (const bool hasDefault : {false, true}) {
    SCOPED_TRACE(hasDefault);
    EXPECT_NO_THROW(validateGroups(defaultOnlyMetadata(), hasDefault));
    NIMBLE_ASSERT_THROW(
        validateGroups(
            HybridFlatMap::fromSchema(
                {}, HybridFlatMap::KeyPresenceMode::kBitmap),
            hasDefault),
        "Hybrid FlatMap requires at least 1 group(s)");
  }

  // A projection may keep only a configured group, but a complete schema's
  // single group must be Default.
  const auto explicitOnly = HybridFlatMap::fromSchema(
      {{.groupId = 7, .groupKeys = {"a"}}},
      HybridFlatMap::KeyPresenceMode::kBitmap);
  EXPECT_NO_THROW(validateGroups(explicitOnly, /*hasDefault=*/false));
  NIMBLE_ASSERT_THROW(
      validateGroups(explicitOnly, /*hasDefault=*/true),
      "Hybrid FlatMap single group must be Default: 7");
}

TEST(HybridFlatMapTest, deserializeAcceptsPhysicalProjectionMetadata) {
  const auto metadata = HybridFlatMap::deserialize(
      encodeRawMetadata({7, HybridFlatMap::kDefaultGroupId}, {0, 0}, {}));
  ASSERT_EQ(metadata.groups().size(), 2);
  EXPECT_EQ(metadata.groups()[0].groupId, 7);
  EXPECT_TRUE(metadata.groups()[0].groupKeys.empty());
  EXPECT_EQ(metadata.groups()[1].groupId, HybridFlatMap::kDefaultGroupId);
  EXPECT_TRUE(metadata.groups()[1].groupKeys.empty());
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

TEST(HybridFlatMapTest, setMetadataAttributeOverridesExisting) {
  std::vector<std::pair<std::string, std::string>> attributes{
      {"user.a", "1"},
      {std::string{HybridFlatMap::kAttributeName}, "stale"},
      {"user.b", "2"},
  };

  const auto expected = metadata();
  expected.setAttribute(attributes);

  ASSERT_EQ(attributes.size(), 3);
  EXPECT_EQ(
      attributes[0], (std::pair<std::string, std::string>{"user.a", "1"}));
  EXPECT_EQ(attributes[1].first, HybridFlatMap::kAttributeName);
  EXPECT_EQ(HybridFlatMap::deserialize(attributes[1].second), expected);
  EXPECT_EQ(
      attributes[2], (std::pair<std::string, std::string>{"user.b", "2"}));
}

TEST(HybridFlatMapTest, setMetadataAttributeRejectsDuplicates) {
  std::vector<std::pair<std::string, std::string>> attributes{
      {std::string{HybridFlatMap::kAttributeName}, "stale"},
      {"user.a", "1"},
      {std::string{HybridFlatMap::kAttributeName}, "also stale"},
  };
  const auto expected = metadata();

  NIMBLE_ASSERT_THROW(
      expected.setAttribute(attributes),
      "Hybrid FlatMap metadata attribute must be unique");
  EXPECT_EQ(HybridFlatMap::deserialize(attributes[0].second), expected);
  EXPECT_EQ(
      attributes[1], (std::pair<std::string, std::string>{"user.a", "1"}));
  EXPECT_EQ(
      attributes[2],
      (std::pair<std::string, std::string>{
          HybridFlatMap::kAttributeName, "also stale"}));
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
