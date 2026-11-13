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
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/SchemaGenerated.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"

namespace facebook::nimble {
namespace {

using testing::ElementsAre;

// Returns the pool that Hybrid FlatMap schemas are read and written with.
velox::memory::MemoryPool& schemaPool() {
  // Owns its manager, which tests that reset the default manager cannot free.
  static velox::memory::MemoryManager manager;
  static const auto pool = manager.addLeafPool();
  return *pool;
}

// Returns the groups a schema records for keys of `keyKind`, as a schema
// reader parses them.
HybridFlatMap schemaGroups(
    ScalarKind keyKind,
    const std::vector<HybridFlatMap::Group>& groups,
    HybridFlatMap::KeyPresenceMode mode) {
  return HybridFlatMap{
      keyKind,
      HybridFlatMap::serialize(
          keyKind, groups, mode, /*groupKeysLayout=*/nullptr, &schemaPool()),
      &schemaPool()};
}

HybridFlatMap metadata() {
  return schemaGroups(
      ScalarKind::String,
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

// Groups with Int64 keys: an explicit group sorted by value, then a bitmap
// Default group in first-seen order.
std::vector<HybridFlatMap::Group> integerGroups() {
  return {
      {.groupId = 7, .groupKeys = {"-5", "9", "10"}},
      {.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {"42", "-1"}},
  };
}

// Returns an uncompressed FixedBitWidth layout, which has no nested encodings.
EncodingLayout fixedBitWidthLayout() {
  return EncodingLayout{
      EncodingType::FixedBitWidth, {}, CompressionType::Uncompressed};
}

// Returns the encoded_group_keys stream of `serialized` metadata.
std::string_view encodedGroupKeys(std::string_view serialized) {
  const auto* encoded =
      flatbuffers::GetRoot<serialization::HybridFlatMap>(serialized.data())
          ->encoded_group_keys();
  NIMBLE_CHECK_NOT_NULL(encoded, "Expected encoded group keys.");
  return {reinterpret_cast<const char*>(encoded->data()), encoded->size()};
}

// Returns the size of each group's stream in the encoded_group_keys of
// `serialized` metadata.
std::vector<uint32_t> encodedGroupKeySizes(std::string_view serialized) {
  const auto* sizes =
      flatbuffers::GetRoot<serialization::HybridFlatMap>(serialized.data())
          ->encoded_group_key_sizes();
  NIMBLE_CHECK_NOT_NULL(sizes, "Expected encoded group key sizes.");
  return {sizes->begin(), sizes->end()};
}

// Returns the values of `encoded`, one group's stream.
std::vector<int64_t> decodeGroupStream(std::string_view encoded) {
  const auto pool = velox::memory::memoryManager()->addLeafPool();
  Encoding::Options options;
  options.useVarintRowCount = true;
  const auto encoding = EncodingFactory{options}.create(
      *pool, encoded, /*stringBufferFactory=*/nullptr);
  std::vector<int64_t> values(encoding->rowCount());
  encoding->materialize(encoding->rowCount(), values.data());
  return values;
}

// Writes Bitmap metadata whose keys are `encodedKeys`, with `sizes` as their
// group sizes when set, plus `groupKeys` strings when not empty.
std::string encodeMetadataWithEncodedKeys(
    const std::vector<uint32_t>& groupIds,
    const std::vector<uint32_t>& groupKeyCounts,
    std::string_view encodedKeys,
    const std::optional<std::vector<uint32_t>>& sizes,
    const std::vector<std::string>& groupKeys = {}) {
  flatbuffers::FlatBufferBuilder builder;
  builder.ForceDefaults(true);
  std::vector<flatbuffers::Offset<flatbuffers::String>> keyOffsets;
  keyOffsets.reserve(groupKeys.size());
  for (const auto& key : groupKeys) {
    keyOffsets.push_back(builder.CreateString(key));
  }
  const auto groupIdsOffset = builder.CreateVector(groupIds);
  const auto groupKeyCountsOffset = builder.CreateVector(groupKeyCounts);
  const auto groupKeysOffset = groupKeys.empty()
      ? flatbuffers::Offset<
            flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>>{}
      : builder.CreateVector(keyOffsets);
  const auto encodedKeysOffset = builder.CreateVector(
      reinterpret_cast<const uint8_t*>(encodedKeys.data()), encodedKeys.size());
  const auto sizesOffset = sizes.has_value()
      ? builder.CreateVector(*sizes)
      : flatbuffers::Offset<flatbuffers::Vector<uint32_t>>{};
  builder.Finish(
      serialization::CreateHybridFlatMap(
          builder,
          groupIdsOffset,
          groupKeyCountsOffset,
          groupKeysOffset,
          serialization::HybridFlatMapKeyPresenceMode_Bitmap,
          encodedKeysOffset,
          sizesOffset));
  return {
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
}

HybridFlatMap defaultOnlyMetadata() {
  return schemaGroups(
      ScalarKind::String,
      {{.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {}}},
      HybridFlatMap::KeyPresenceMode::kBitmap);
}

void validateGroups(
    const std::vector<HybridFlatMap::Group>& groups,
    bool hasDefault) {
  folly::F14FastMap<std::string_view, size_t> groupIndices;
  detail::validateHybridFlatMapGroups(
      groups.size(),
      hasDefault,
      [&groups](size_t index) { return groups[index].groupId; },
      [&groups](size_t index) -> const auto& {
        return groups[index].groupKeys;
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

// Keys come back canonical and sorted, by value for integer keys: 9 before 10.
TEST(HybridFlatMapTest, constructorSortsKeys) {
  EXPECT_EQ(
      HybridFlatMap(
          ScalarKind::Int64,
          config({{0, {"9", "10", "-3"}}, {4, {"2"}}, {kDefault, {}}}),
          kBitmap)
          .groups(),
      config({{0, {"-3", "9", "10"}}, {4, {"2"}}, {kDefault, {}}}));
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

  // The schema constructor checks the key kind before it parses the bytes.
  NIMBLE_ASSERT_THROW(
      HybridFlatMap(ScalarKind::Double, "malformed", /*pool=*/nullptr),
      "Hybrid FlatMap key kind is unsupported");
}

// A schema's bitmap Default lists the keys it recorded, so findGroup() finds
// them as it finds the keys of explicit groups.
TEST(HybridFlatMapTest, findGroupIncludesSchemaDefaultKeys) {
  const auto hybridMap = metadata();
  EXPECT_EQ(hybridMap.findGroup("C"), size_t{1});
  EXPECT_EQ(hybridMap.findGroup("seen-first"), size_t{2});
  EXPECT_EQ(hybridMap.findGroup("a"), std::nullopt);
  EXPECT_EQ(hybridMap.numGroupedKeys(), size_t{5});
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

TEST(HybridFlatMapTest, metadataRoundTripPreservesFlattenedOrderAndBinaryKeys) {
  const auto expected = metadata();
  const auto serialized = HybridFlatMap::serialize(
      ScalarKind::String,
      expected.groups(),
      expected.defaultGroupKeyPresenceMode(),
      /*groupKeysLayout=*/nullptr,
      &schemaPool());
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

  EXPECT_EQ(
      HybridFlatMap(ScalarKind::String, serialized, &schemaPool()), expected);
}

TEST(HybridFlatMapTest, metadataRoundTripsDefaultOnly) {
  const auto expected = defaultOnlyMetadata();
  const auto serialized = HybridFlatMap::serialize(
      ScalarKind::String,
      expected.groups(),
      expected.defaultGroupKeyPresenceMode(),
      /*groupKeysLayout=*/nullptr,
      &schemaPool());
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

  EXPECT_EQ(
      HybridFlatMap(ScalarKind::String, serialized, &schemaPool()), expected);
}

TEST(HybridFlatMapTest, metadataRoundTripsKeyList) {
  // The key presence mode is written even when it is Bitmap.
  const auto hasKeyPresenceField = [](std::string_view serialized) {
    return flatbuffers::GetRoot<flatbuffers::Table>(serialized.data())
        ->CheckField(
            serialization::HybridFlatMap::VT_DEFAULT_GROUP_KEY_PRESENCE_MODE);
  };
  const auto bitmapMetadata = metadata();
  EXPECT_TRUE(hasKeyPresenceField(
      HybridFlatMap::serialize(
          ScalarKind::String,
          bitmapMetadata.groups(),
          bitmapMetadata.defaultGroupKeyPresenceMode(),
          /*groupKeysLayout=*/nullptr,
          &schemaPool())));

  const auto expected = schemaGroups(
      ScalarKind::String,
      defaultOnlyMetadata().groups(),
      HybridFlatMap::KeyPresenceMode::kKeyList);
  const auto serialized = HybridFlatMap::serialize(
      ScalarKind::String,
      expected.groups(),
      expected.defaultGroupKeyPresenceMode(),
      /*groupKeysLayout=*/nullptr,
      &schemaPool());
  flatbuffers::Verifier verifier{
      reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size()};
  ASSERT_TRUE(verifier.VerifyBuffer<serialization::HybridFlatMap>());

  EXPECT_TRUE(hasKeyPresenceField(serialized));
  EXPECT_EQ(
      flatbuffers::GetRoot<serialization::HybridFlatMap>(serialized.data())
          ->default_group_key_presence_mode(),
      serialization::HybridFlatMapKeyPresenceMode_KeyList);
  EXPECT_EQ(
      HybridFlatMap(ScalarKind::String, serialized, &schemaPool()), expected);
}

TEST(HybridFlatMapTest, metadataEncodesIntegerKeys) {
  velox::memory::MemoryManager::testingSetInstance({});
  const auto pool = velox::memory::memoryManager()->addLeafPool();
  const auto groups = integerGroups();
  const auto layout = fixedBitWidthLayout();
  const auto serialized = HybridFlatMap::serialize(
      ScalarKind::Int64,
      groups,
      HybridFlatMap::KeyPresenceMode::kBitmap,
      &layout,
      pool.get());
  flatbuffers::Verifier verifier{
      reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size()};
  ASSERT_TRUE(verifier.VerifyBuffer<serialization::HybridFlatMap>());

  const auto* flat =
      flatbuffers::GetRoot<serialization::HybridFlatMap>(serialized.data());
  EXPECT_THAT(*flat->group_key_counts(), ElementsAre(3, 2));
  // The explicit group is a stream, and the bitmap Default's keys stay
  // strings.
  EXPECT_THAT(encodedGroupKeySizes(serialized), ElementsAre(testing::Gt(0), 0));
  EXPECT_THAT(
      decodeGroupStream(encodedGroupKeys(serialized)), ElementsAre(-5, 9, 10));
  ASSERT_NE(flat->group_keys(), nullptr);
  ASSERT_EQ(flat->group_keys()->size(), 2);
  EXPECT_EQ(flat->group_keys()->Get(0)->str(), "42");
  EXPECT_EQ(flat->group_keys()->Get(1)->str(), "-1");
  EXPECT_EQ(
      EncodingPrefix::dataType(encodedGroupKeys(serialized)), DataType::Int64);
  const HybridFlatMap recorded{ScalarKind::Int64, serialized, pool.get()};
  EXPECT_EQ(recorded.groups(), groups);
  EXPECT_FALSE(recorded.groups()[0].encodedKeys.empty());
  EXPECT_TRUE(recorded.groups()[1].encodedKeys.empty());

  // A projection copies the explicit group's stream and the Default strings.
  EXPECT_EQ(
      HybridFlatMap::serialize(
          ScalarKind::Int64,
          recorded.groups(),
          HybridFlatMap::KeyPresenceMode::kBitmap,
          /*groupKeysLayout=*/nullptr,
          pool.get()),
      serialized);
}

// Each group with keys is its own stream, ascending by value, so a projection
// can copy it and EliasFano stores it.
TEST(HybridFlatMapTest, metadataEncodesEachGroup) {
  velox::memory::MemoryManager::testingSetInstance({});
  const auto pool = velox::memory::memoryManager()->addLeafPool();
  const std::vector<HybridFlatMap::Group> groups{
      {.groupId = 7, .groupKeys = {"-5", "9", "10"}},
      {.groupId = 8, .groupKeys = {"3"}},
      {.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {}},
  };
  const EncodingLayout eliasFano{
      EncodingType::EliasFano, {}, CompressionType::Uncompressed};
  const auto serialized = HybridFlatMap::serialize(
      ScalarKind::Int64,
      groups,
      HybridFlatMap::KeyPresenceMode::kKeyList,
      &eliasFano,
      pool.get());
  const auto encoded = encodedGroupKeys(serialized);
  const auto sizes = encodedGroupKeySizes(serialized);
  ASSERT_THAT(sizes, ElementsAre(sizes[0], sizes[1], 0));
  EXPECT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::EliasFano);
  EXPECT_THAT(
      decodeGroupStream(encoded.substr(0, sizes[0])), ElementsAre(-5, 9, 10));
  EXPECT_THAT(
      decodeGroupStream(encoded.substr(sizes[0], sizes[1])), ElementsAre(3));

  // The schema keeps each group's stream, which serialize() copies.
  const HybridFlatMap recorded{ScalarKind::Int64, serialized, pool.get()};
  EXPECT_EQ(recorded.groups(), groups);
  EXPECT_EQ(recorded.groups()[0].encodedKeys, encoded.substr(0, sizes[0]));
  EXPECT_TRUE(recorded.groups()[2].encodedKeys.empty());
  EXPECT_EQ(
      HybridFlatMap::serialize(
          ScalarKind::Int64,
          recorded.groups(),
          HybridFlatMap::KeyPresenceMode::kKeyList,
          /*groupKeysLayout=*/nullptr,
          pool.get()),
      serialized);

  // Negative keys and keys of different lengths, in value order.
  const std::vector<HybridFlatMap::Group> mixedGroups{
      {.groupId = 0, .groupKeys = {"-10", "-9", "-1", "0", "9", "10", "105"}},
      {.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {}},
  };
  EXPECT_EQ(
      HybridFlatMap(
          ScalarKind::Int64,
          HybridFlatMap::serialize(
              ScalarKind::Int64,
              mixedGroups,
              HybridFlatMap::KeyPresenceMode::kKeyList,
              &eliasFano,
              pool.get()),
          pool.get())
          .groups(),
      mixedGroups);
}

TEST(HybridFlatMapTest, metadataWritesKeysAsStringsWithoutEncoding) {
  velox::memory::MemoryManager::testingSetInstance({});
  const auto pool = velox::memory::memoryManager()->addLeafPool();
  // Integer keys without a layout.
  const auto groups = integerGroups();
  const auto serialized = HybridFlatMap::serialize(
      ScalarKind::Int64,
      groups,
      HybridFlatMap::KeyPresenceMode::kBitmap,
      /*groupKeysLayout=*/nullptr,
      pool.get());
  const auto* flat =
      flatbuffers::GetRoot<serialization::HybridFlatMap>(serialized.data());
  EXPECT_NE(flat->group_keys(), nullptr);
  EXPECT_EQ(flat->encoded_group_keys(), nullptr);
  EXPECT_EQ(
      HybridFlatMap(ScalarKind::Int64, serialized, pool.get()).groups(),
      groups);

  // String keys ignore the layout, which covers every node of a schema.
  const auto stringMetadata = metadata();
  const auto layout = fixedBitWidthLayout();
  const auto stringSerialized = HybridFlatMap::serialize(
      ScalarKind::String,
      stringMetadata.groups(),
      stringMetadata.defaultGroupKeyPresenceMode(),
      &layout,
      pool.get());
  const auto* stringFlat = flatbuffers::GetRoot<serialization::HybridFlatMap>(
      stringSerialized.data());
  EXPECT_NE(stringFlat->group_keys(), nullptr);
  EXPECT_EQ(stringFlat->encoded_group_keys(), nullptr);
  EXPECT_EQ(
      HybridFlatMap(ScalarKind::String, stringSerialized, pool.get()),
      stringMetadata);
}

// Writers that predate sorting integer keys by value sorted them as strings,
// and the key-presence bitmaps of their batches follow that order.
TEST(HybridFlatMapTest, schemaKeepsIntegerKeysInStringOrder) {
  const std::vector<HybridFlatMap::Group> groups{
      {.groupId = 0, .groupKeys = {"-1", "-20", "10", "100", "9"}},
      {.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {"50", "5"}},
  };
  const auto serialized = HybridFlatMap::serialize(
      ScalarKind::Int64,
      groups,
      HybridFlatMap::KeyPresenceMode::kBitmap,
      /*groupKeysLayout=*/nullptr,
      &schemaPool());
  const HybridFlatMap schema{ScalarKind::Int64, serialized, &schemaPool()};
  EXPECT_EQ(schema.groups(), groups);
  EXPECT_EQ(schema.findGroup("9"), size_t{0});
  EXPECT_EQ(schema.findGroup("5"), size_t{1});

  // A projection writes the keys back in the order the schema recorded.
  EXPECT_EQ(
      HybridFlatMap::serialize(
          ScalarKind::Int64,
          schema.groups(),
          schema.defaultGroupKeyPresenceMode(),
          /*groupKeysLayout=*/nullptr,
          &schemaPool()),
      serialized);
}

TEST(HybridFlatMapTest, metadataReplaysGroupKeysLayout) {
  velox::memory::MemoryManager::testingSetInstance({});
  const auto pool = velox::memory::memoryManager()->addLeafPool();
  const auto groups = integerGroups();
  const auto serializeWith = [&](const EncodingLayout& layout) {
    return HybridFlatMap::serialize(
        ScalarKind::Int64,
        groups,
        HybridFlatMap::KeyPresenceMode::kBitmap,
        &layout,
        pool.get());
  };
  for (const auto encodingType :
       {EncodingType::FixedBitWidth, EncodingType::Varint}) {
    SCOPED_TRACE(toString(encodingType));
    const auto serialized = serializeWith(
        EncodingLayout{encodingType, {}, CompressionType::Uncompressed});
    EXPECT_EQ(
        EncodingPrefix::encodingType(encodedGroupKeys(serialized)),
        encodingType);
    EXPECT_EQ(
        HybridFlatMap(ScalarKind::Int64, serialized, pool.get()).groups(),
        groups);
  }

  // A nested encoding replays each child the layout names.
  {
    const EncodingLayout trivial{
        EncodingType::Trivial, {}, CompressionType::Uncompressed};
    const auto serialized = serializeWith(
        EncodingLayout{
            EncodingType::Delta,
            {},
            CompressionType::Uncompressed,
            {trivial, trivial, trivial}});
    EXPECT_EQ(
        EncodingPrefix::encodingType(encodedGroupKeys(serialized)),
        EncodingType::Delta);
    EXPECT_EQ(
        HybridFlatMap(ScalarKind::Int64, serialized, pool.get()).groups(),
        groups);
  }

  // A FixedBitWidth stream holds its compression type right after the
  // encoding prefix. The keys stay uncompressed even when the layout names a
  // compression.
  const auto serialized = serializeWith(
      EncodingLayout{EncodingType::FixedBitWidth, {}, CompressionType::Zstd});
  EXPECT_EQ(
      static_cast<CompressionType>(
          encodedGroupKeys(serialized)[EncodingPrefix::kFixedPrefixSize]),
      CompressionType::Uncompressed);
  EXPECT_EQ(
      HybridFlatMap(ScalarKind::Int64, serialized, pool.get()).groups(),
      groups);
}

// A layout that cannot encode a group's keys fails the write rather than
// writing them another way.
TEST(HybridFlatMapTest, serializeRejectsKeysTheLayoutCannotEncode) {
  velox::memory::MemoryManager::testingSetInstance({});
  const auto pool = velox::memory::memoryManager()->addLeafPool();
  const auto serialize = [&](ScalarKind keyKind,
                             const std::vector<HybridFlatMap::Group>& groups,
                             const EncodingLayout& layout) {
    return HybridFlatMap::serialize(
        keyKind,
        groups,
        HybridFlatMap::KeyPresenceMode::kBitmap,
        &layout,
        pool.get());
  };
  const auto groups = integerGroups();
  {
    SCOPED_TRACE("Prefix encodes only strings");
    EXPECT_THROW(
        serialize(
            ScalarKind::Int32,
            groups,
            EncodingLayout{
                EncodingType::Prefix, {}, CompressionType::Uncompressed}),
        NimbleException);
  }
  {
    SCOPED_TRACE("missing nested encoding");
    const EncodingLayout trivial{
        EncodingType::Trivial, {}, CompressionType::Uncompressed};
    NIMBLE_ASSERT_USER_THROW(
        serialize(
            ScalarKind::Int64,
            groups,
            EncodingLayout{
                EncodingType::Delta,
                {},
                CompressionType::Uncompressed,
                {trivial, std::nullopt, trivial}}),
        "Hybrid FlatMap group keys layout is missing a nested encoding");
  }
  {
    SCOPED_TRACE("under one bit per key");
    // Keys 1000 through 1999 are 999 deltas of 1, which Constant holds in a
    // few bytes.
    std::vector<std::string> keys;
    for (int key = 1000; key < 2000; ++key) {
      keys.push_back(std::to_string(key));
    }
    const EncodingLayout constant{
        EncodingType::Constant, {}, CompressionType::Uncompressed};
    NIMBLE_ASSERT_USER_THROW(
        serialize(
            ScalarKind::Int64,
            {{.groupId = 0, .groupKeys = keys},
             {.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {}}},
            EncodingLayout{
                EncodingType::Delta,
                {},
                CompressionType::Uncompressed,
                {constant,
                 constant,
                 EncodingLayout{
                     EncodingType::SparseBool,
                     {},
                     CompressionType::Uncompressed,
                     {EncodingLayout{
                         EncodingType::Trivial,
                         {},
                         CompressionType::Uncompressed}}}}}),
        "Hybrid FlatMap group keys layout encodes the keys of group 0 in under "
        "one bit per key");
  }
}

// Every explicit group lists keys, and keys that are neither recorded nor
// encodable with a layout cannot be written as a stream.
TEST(HybridFlatMapTest, serializeRejectsGroupsItCannotWriteAsStreams) {
  velox::memory::MemoryManager::testingSetInstance({});
  const auto pool = velox::memory::memoryManager()->addLeafPool();
  const auto layout = fixedBitWidthLayout();
  NIMBLE_ASSERT_THROW(
      HybridFlatMap::serialize(
          ScalarKind::Int64,
          {{.groupId = 7, .groupKeys = {}},
           {.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {}}},
          HybridFlatMap::KeyPresenceMode::kBitmap,
          &layout,
          pool.get()),
      "Hybrid FlatMap explicit group 7 has no keys");

  // Group 7 carries the stream a schema recorded, and group 8 has keys with
  // no stream to copy and no layout to encode them.
  const HybridFlatMap recorded{
      ScalarKind::Int64,
      HybridFlatMap::serialize(
          ScalarKind::Int64,
          integerGroups(),
          HybridFlatMap::KeyPresenceMode::kBitmap,
          &layout,
          pool.get()),
      pool.get()};
  auto groups = recorded.groups();
  groups.insert(
      groups.begin() + 1,
      HybridFlatMap::Group{.groupId = 8, .groupKeys = {"11"}});
  NIMBLE_ASSERT_THROW(
      HybridFlatMap::serialize(
          ScalarKind::Int64,
          groups,
          HybridFlatMap::KeyPresenceMode::kBitmap,
          /*groupKeysLayout=*/nullptr,
          pool.get()),
      "Hybrid FlatMap group 8 has neither a recorded stream nor a layout to "
      "encode its keys");
}

// Encoding and decoding integer keys allocate from the caller's pool, which
// schemas require even with string keys.
TEST(HybridFlatMapTest, encodedKeysRequireMemoryPool) {
  velox::memory::MemoryManager::testingSetInstance({});
  const auto pool = velox::memory::memoryManager()->addLeafPool();
  const auto layout = fixedBitWidthLayout();
  const auto groups = integerGroups();
  const auto serialized = HybridFlatMap::serialize(
      ScalarKind::Int64,
      groups,
      HybridFlatMap::KeyPresenceMode::kBitmap,
      &layout,
      pool.get());
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap(ScalarKind::Int64, serialized, /*pool=*/nullptr),
      "Hybrid FlatMap schema requires a memory pool");
  EXPECT_EQ(
      HybridFlatMap(ScalarKind::Int64, serialized, pool.get()).groups(),
      groups);
}

TEST(HybridFlatMapTest, metadataEncodesNarrowIntegerKeys) {
  velox::memory::MemoryManager::testingSetInstance({});
  const auto pool = velox::memory::memoryManager()->addLeafPool();
  const auto layout = fixedBitWidthLayout();
  // Each key kind's keys are a stream of its own integer type.
  for (const auto& [keyKind, dataType, keys] :
       std::vector<std::tuple<ScalarKind, DataType, std::vector<std::string>>>{
           {ScalarKind::Int8, DataType::Int8, {"-128", "127"}},
           {ScalarKind::Int16, DataType::Int16, {"-32768", "32767"}},
           {ScalarKind::Int32, DataType::Int32, {"-2147483648", "2147483647"}},
           {ScalarKind::Int64,
            DataType::Int64,
            {"-9223372036854775808", "9223372036854775807"}},
       }) {
    SCOPED_TRACE(toString(keyKind));
    const std::vector<HybridFlatMap::Group> groups{
        {.groupId = 0, .groupKeys = keys},
        {.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {}},
    };
    const auto serialized = HybridFlatMap::serialize(
        keyKind,
        groups,
        HybridFlatMap::KeyPresenceMode::kKeyList,
        &layout,
        pool.get());
    EXPECT_EQ(EncodingPrefix::dataType(encodedGroupKeys(serialized)), dataType);
    EXPECT_EQ(HybridFlatMap(keyKind, serialized, pool.get()).groups(), groups);
  }
}

TEST(HybridFlatMapTest, metadataOmitsEncodedKeysWithoutKeys) {
  velox::memory::MemoryManager::testingSetInstance({});
  const auto pool = velox::memory::memoryManager()->addLeafPool();
  const auto layout = fixedBitWidthLayout();
  const auto serialized = HybridFlatMap::serialize(
      ScalarKind::Int64,
      defaultOnlyMetadata().groups(),
      HybridFlatMap::KeyPresenceMode::kBitmap,
      &layout,
      pool.get());
  const auto* flat =
      flatbuffers::GetRoot<serialization::HybridFlatMap>(serialized.data());
  ASSERT_NE(flat->group_keys(), nullptr);
  EXPECT_EQ(flat->group_keys()->size(), 0);
  EXPECT_EQ(flat->encoded_group_keys(), nullptr);
  EXPECT_EQ(
      HybridFlatMap(ScalarKind::Int64, serialized, pool.get()).groups(),
      defaultOnlyMetadata().groups());

  // A Default-only node's keys stay strings, even when it lists keys.
  const std::vector<HybridFlatMap::Group> defaultKeys{
      {.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {"42", "-1"}}};
  const auto withKeys = HybridFlatMap::serialize(
      ScalarKind::Int64,
      defaultKeys,
      HybridFlatMap::KeyPresenceMode::kBitmap,
      &layout,
      pool.get());
  const auto* withKeysFlat =
      flatbuffers::GetRoot<serialization::HybridFlatMap>(withKeys.data());
  ASSERT_NE(withKeysFlat->group_keys(), nullptr);
  EXPECT_EQ(withKeysFlat->group_keys()->size(), 2);
  EXPECT_EQ(withKeysFlat->encoded_group_keys(), nullptr);
  EXPECT_EQ(
      HybridFlatMap(ScalarKind::Int64, withKeys, pool.get()).groups(),
      defaultKeys);
}

TEST(HybridFlatMapTest, serializeRejectsNonCanonicalIntegerKeys) {
  velox::memory::MemoryManager::testingSetInstance({});
  const auto pool = velox::memory::memoryManager()->addLeafPool();
  const auto layout = fixedBitWidthLayout();
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap::serialize(
          ScalarKind::Int64,
          {{.groupId = 0, .groupKeys = {"07"}},
           {.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {}}},
          HybridFlatMap::KeyPresenceMode::kBitmap,
          &layout,
          pool.get()),
      "Hybrid FlatMap key '07' is not canonical");
  NIMBLE_ASSERT_USER_THROW(
      HybridFlatMap::serialize(
          ScalarKind::Int32,
          {{.groupId = 0, .groupKeys = {"2147483648"}},
           {.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {}}},
          HybridFlatMap::KeyPresenceMode::kBitmap,
          &layout,
          pool.get()),
      "Hybrid FlatMap key '2147483648' cannot be parsed as Int32");
}

TEST(HybridFlatMapTest, rejectsMalformedEncodedKeys) {
  velox::memory::MemoryManager::testingSetInstance({});
  const auto pool = velox::memory::memoryManager()->addLeafPool();
  const auto layout = fixedBitWidthLayout();
  // Group 7 has a stream, and the bitmap Default's keys are strings.
  const auto serialized = HybridFlatMap::serialize(
      ScalarKind::Int64,
      integerGroups(),
      HybridFlatMap::KeyPresenceMode::kBitmap,
      &layout,
      pool.get());
  const std::string encoded{encodedGroupKeys(serialized)};
  const auto sizes = encodedGroupKeySizes(serialized);
  ASSERT_THAT(sizes, ElementsAre(encoded.size(), 0));
  const std::vector<uint32_t> groupIds{7, HybridFlatMap::kDefaultGroupId};
  const std::vector<std::string> defaultKeys{"42", "-1"};
  const auto metadata =
      [&](const std::vector<uint32_t>& groupKeyCounts,
          std::string_view encodedKeys,
          const std::optional<std::vector<uint32_t>>& groupSizes) {
        return encodeMetadataWithEncodedKeys(
            groupIds, groupKeyCounts, encodedKeys, groupSizes, defaultKeys);
      };

  NIMBLE_ASSERT_FILE_THROW(
      HybridFlatMap(ScalarKind::String, serialized, pool.get()),
      "Hybrid FlatMap encoded group keys require an integer key kind");
  NIMBLE_ASSERT_FILE_THROW(
      HybridFlatMap(
          ScalarKind::Int64,
          metadata({3, 2}, encoded.substr(0, 2), std::vector<uint32_t>{2, 0}),
          pool.get()),
      "Hybrid FlatMap encoded group keys are truncated");
  {
    auto int32Stream = encoded;
    int32Stream[EncodingPrefix::kDataTypeOffset] =
        static_cast<char>(DataType::Int32);
    NIMBLE_ASSERT_FILE_THROW(
        HybridFlatMap(
            ScalarKind::Int64,
            metadata({3, 2}, int32Stream, sizes),
            pool.get()),
        "Hybrid FlatMap encoded group keys must be of the key type");
  }
  NIMBLE_ASSERT_FILE_THROW(
      HybridFlatMap(
          ScalarKind::Int64, metadata({2, 2}, encoded, sizes), pool.get()),
      "Hybrid FlatMap encoded group keys must match the group key counts");
  NIMBLE_ASSERT_FILE_THROW(
      HybridFlatMap(
          ScalarKind::Int64,
          metadata(
              {static_cast<uint32_t>(8 * encoded.size() + 1), 2},
              encoded,
              sizes),
          pool.get()),
      "Hybrid FlatMap encoded group keys are too short for the group key "
      "counts");
  NIMBLE_ASSERT_FILE_THROW(
      HybridFlatMap(
          ScalarKind::Int64,
          metadata({3, 2}, encoded, std::nullopt),
          pool.get()),
      "Hybrid FlatMap encoded group keys must list the size of each group's "
      "keys");
  NIMBLE_ASSERT_FILE_THROW(
      HybridFlatMap(
          ScalarKind::Int64,
          metadata({3, 2}, encoded, std::vector<uint32_t>{sizes[0]}),
          pool.get()),
      "Hybrid FlatMap encoded group key sizes must match the group count");
  NIMBLE_ASSERT_FILE_THROW(
      HybridFlatMap(
          ScalarKind::Int64,
          metadata({3, 2}, encoded, std::vector<uint32_t>{sizes[0], 1}),
          pool.get()),
      "Hybrid FlatMap encoded group keys are truncated");
  NIMBLE_ASSERT_FILE_THROW(
      HybridFlatMap(
          ScalarKind::Int64,
          metadata({3, 2}, encoded + "x", sizes),
          pool.get()),
      "Hybrid FlatMap encoded group key sizes must add up to the encoded group "
      "keys");
  NIMBLE_ASSERT_FILE_THROW(
      HybridFlatMap(
          ScalarKind::Int64, metadata({0, 2}, encoded, sizes), pool.get()),
      "Hybrid FlatMap group without keys must have no encoded keys");

  // The Default group's keys are strings, never a stream.
  NIMBLE_ASSERT_FILE_THROW(
      HybridFlatMap(
          ScalarKind::Int64,
          metadata(
              {3, 2},
              encoded + encoded,
              std::vector<uint32_t>{sizes[0], sizes[0]}),
          pool.get()),
      "Hybrid FlatMap Default group must have no encoded keys");

  // group_keys holds only the Default group's keys next to encoded keys.
  NIMBLE_ASSERT_THROW(
      HybridFlatMap(
          ScalarKind::Int64,
          encodeMetadataWithEncodedKeys(
              groupIds, {3, 2}, encoded, sizes, {"-5", "9", "10", "42", "-1"}),
          pool.get()),
      "Hybrid FlatMap group key counts must match group keys size");

  // An Int8 map cannot read the Int64 stream of Int64 keys.
  NIMBLE_ASSERT_FILE_THROW(
      HybridFlatMap(
          ScalarKind::Int8,
          HybridFlatMap::serialize(
              ScalarKind::Int64,
              {{.groupId = 0, .groupKeys = {"300"}},
               {.groupId = HybridFlatMap::kDefaultGroupId, .groupKeys = {}}},
              HybridFlatMap::KeyPresenceMode::kBitmap,
              &layout,
              pool.get()),
          pool.get()),
      "Hybrid FlatMap encoded group keys must be of the key type");
}

TEST(HybridFlatMapTest, rejectsOmittedVectors) {
  flatbuffers::FlatBufferBuilder builder;
  // Writes the Default key presence mode, so only the vectors are missing.
  builder.ForceDefaults(true);
  serialization::HybridFlatMapBuilder metadataBuilder{builder};
  metadataBuilder.add_default_group_key_presence_mode(
      serialization::HybridFlatMapKeyPresenceMode_Bitmap);
  const auto serialized = metadataBuilder.Finish();
  builder.Finish(serialized);
  const std::string_view bytes{
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};

  NIMBLE_ASSERT_THROW(
      HybridFlatMap(ScalarKind::String, bytes, &schemaPool()),
      "Hybrid FlatMap group IDs are missing");
}

TEST(HybridFlatMapTest, schemaAcceptsOmittedGroupKeysWithoutKeys) {
  EXPECT_EQ(
      HybridFlatMap(
          ScalarKind::String,
          encodeMetadataWithoutGroupKeys({HybridFlatMap::kDefaultGroupId}, {0}),
          &schemaPool()),
      defaultOnlyMetadata());

  NIMBLE_ASSERT_THROW(
      HybridFlatMap(
          ScalarKind::String,
          encodeMetadataWithoutGroupKeys(
              {7, HybridFlatMap::kDefaultGroupId}, {1, 0}),
          &schemaPool()),
      "Hybrid FlatMap group key counts must match group keys size");
}

TEST(HybridFlatMapTest, rejectsMalformedMetadata) {
  NIMBLE_ASSERT_THROW(
      HybridFlatMap(ScalarKind::String, "malformed", &schemaPool()),
      "Hybrid FlatMap metadata attribute is malformed");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap(
          ScalarKind::String,
          encodeRawMetadata({0, 1}, {1}, {}),
          &schemaPool()),
      "Hybrid FlatMap group IDs and key counts must have the same size");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap(
          ScalarKind::String,
          encodeRawMetadata({0, 1}, {1, 1}, {"a"}),
          &schemaPool()),
      "Hybrid FlatMap group key counts must match group keys size");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap(
          ScalarKind::String,
          encodeRawMetadata({0, 1}, {0, 0}, {"unexpected"}),
          &schemaPool()),
      "Hybrid FlatMap group key counts must match group keys size");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap(
          ScalarKind::String,
          encodeRawMetadata(
              {HybridFlatMap::kDefaultGroupId},
              {1},
              {"listed"},
              serialization::HybridFlatMapKeyPresenceMode_KeyList),
          &schemaPool()),
      "Hybrid FlatMap Default group cannot list keys when batches list their "
      "Default keys");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap(
          ScalarKind::String,
          encodeRawMetadata(
              {HybridFlatMap::kDefaultGroupId},
              {0},
              {},
              /*defaultGroupKeyPresenceMode=*/2),
          &schemaPool()),
      "Hybrid FlatMap Default key presence is unknown");
  NIMBLE_ASSERT_THROW(
      HybridFlatMap(
          ScalarKind::String,
          encodeRawMetadata(
              {HybridFlatMap::kDefaultGroupId},
              {0},
              {},
              /*defaultGroupKeyPresenceMode=*/std::nullopt),
          &schemaPool()),
      "Hybrid FlatMap Default key presence is missing");
}

TEST(HybridFlatMapTest, projectedConfiguredGroupRequiresKeys) {
  const std::vector<HybridFlatMap::Group> projected{
      {.groupId = 7, .groupKeys = {}},
  };
  folly::F14FastMap<std::string_view, size_t> groupIndices;

  NIMBLE_ASSERT_THROW(
      detail::validateHybridFlatMapGroups(
          projected.size(),
          /*hasDefault=*/false,
          [&projected](size_t index) { return projected[index].groupId; },
          [&projected](size_t index) -> const auto& {
            return projected[index].groupKeys;
          },
          groupIndices),
      "Hybrid FlatMap group must contain at least one key: 7");
}

TEST(HybridFlatMapTest, validatesMinimumGroupCount) {
  for (const bool hasDefault : {false, true}) {
    SCOPED_TRACE(hasDefault);
    EXPECT_NO_THROW(validateGroups(defaultOnlyMetadata().groups(), hasDefault));
    NIMBLE_ASSERT_THROW(
        validateGroups({}, hasDefault),
        "Hybrid FlatMap requires at least 1 group(s)");
  }

  // A projection may keep only a configured group, but a complete schema's
  // single group must be Default.
  const std::vector<HybridFlatMap::Group> explicitOnly{
      {.groupId = 7, .groupKeys = {"a"}},
  };
  EXPECT_NO_THROW(validateGroups(explicitOnly, /*hasDefault=*/false));
  NIMBLE_ASSERT_THROW(
      validateGroups(explicitOnly, /*hasDefault=*/true),
      "Hybrid FlatMap single group must be Default: 7");
}

// A written schema, projected or not, lists at least one key in each explicit
// group.
TEST(HybridFlatMapTest, schemaRejectsKeylessExplicitGroup) {
  NIMBLE_ASSERT_THROW(
      HybridFlatMap(
          ScalarKind::String,
          encodeRawMetadata({7, HybridFlatMap::kDefaultGroupId}, {0, 0}, {}),
          &schemaPool()),
      "Hybrid FlatMap group must contain at least one key: 7");
}

} // namespace
} // namespace facebook::nimble
