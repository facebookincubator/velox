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

#include <algorithm>
#include <iterator>

#include <folly/Conv.h>
#include <folly/container/F14Set.h>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/SchemaGenerated.h"

namespace facebook::nimble {

bool HybridFlatMap::supportedKeyKind(ScalarKind kind) {
  switch (kind) {
    case ScalarKind::Int8:
    case ScalarKind::Int16:
    case ScalarKind::Int32:
    case ScalarKind::Int64:
    case ScalarKind::String:
      return true;
    case ScalarKind::UInt8:
    case ScalarKind::UInt16:
    case ScalarKind::UInt32:
    case ScalarKind::UInt64:
    case ScalarKind::Float:
    case ScalarKind::Double:
    case ScalarKind::Bool:
    case ScalarKind::Binary:
    case ScalarKind::Undefined:
      return false;
  }
  NIMBLE_UNREACHABLE("Unknown scalar kind: {}.", kind);
}

namespace {

// The decimal form of `key`'s value as an integer of type T, which must be
// `key` itself.
template <typename T>
std::string canonicalIntegerKey(ScalarKind keyKind, std::string_view key) {
  const auto parsed = folly::tryTo<T>(key);
  NIMBLE_USER_CHECK(
      parsed.hasValue(),
      "Hybrid FlatMap key '{}' cannot be parsed as {}.",
      key,
      keyKind);
  auto canonical = std::to_string(parsed.value());
  NIMBLE_USER_CHECK_EQ(
      canonical, key, "Hybrid FlatMap key '{}' is not canonical.", key);
  return canonical;
}

} // namespace

std::string HybridFlatMap::canonicalKey(
    ScalarKind keyKind,
    std::string_view key) {
  NIMBLE_USER_CHECK(!key.empty(), "Hybrid FlatMap key cannot be empty.");
  switch (keyKind) {
    case ScalarKind::Int8:
      return canonicalIntegerKey<int8_t>(keyKind, key);
    case ScalarKind::Int16:
      return canonicalIntegerKey<int16_t>(keyKind, key);
    case ScalarKind::Int32:
      return canonicalIntegerKey<int32_t>(keyKind, key);
    case ScalarKind::Int64:
      return canonicalIntegerKey<int64_t>(keyKind, key);
    case ScalarKind::String:
      return std::string(key);
    case ScalarKind::UInt8:
    case ScalarKind::UInt16:
    case ScalarKind::UInt32:
    case ScalarKind::UInt64:
    case ScalarKind::Float:
    case ScalarKind::Double:
    case ScalarKind::Bool:
    case ScalarKind::Binary:
    case ScalarKind::Undefined:
      break;
  }
  NIMBLE_USER_FAIL("Unsupported Hybrid FlatMap key kind: {}.", keyKind);
}

HybridFlatMap::HybridFlatMap(ScalarKind keyKind, std::vector<Group> groups)
    : keyKind_{keyKind}, groups_{std::move(groups)} {
  NIMBLE_USER_CHECK(
      supportedKeyKind(keyKind),
      "Unsupported Hybrid FlatMap key kind: {}.",
      keyKind);
  folly::F14FastSet<std::string> keys;
  bool foundDefault{false};
  for (size_t i = 0; i < groups_.size(); ++i) {
    auto& group = groups_[i];
    if (isDefaultGroup(group.groupId)) {
      NIMBLE_USER_CHECK(
          !foundDefault, "Hybrid FlatMap has multiple Default groups.");
    }
    // Ascending IDs are also distinct, and put the Default group, whose ID is
    // the largest, last.
    if (i > 0) {
      NIMBLE_USER_CHECK_LT(
          groups_[i - 1].groupId,
          group.groupId,
          "Hybrid FlatMap group IDs must be in ascending order.");
    }
    if (isDefaultGroup(group.groupId)) {
      NIMBLE_USER_CHECK(
          group.groupKeys.empty(),
          "Hybrid FlatMap Default group keys are appended in first-seen order "
          "and cannot be configured.");
      foundDefault = true;
      continue;
    }
    NIMBLE_USER_CHECK(
        !group.groupKeys.empty(),
        "Hybrid FlatMap group must contain at least one key: {}.",
        group.groupId);
    for (const auto& key : group.groupKeys) {
      // Throws unless `key` is canonical; a canonical key is returned as is.
      canonicalKey(keyKind, key);
      NIMBLE_USER_CHECK(
          keys.insert(key).second, "Duplicate Hybrid FlatMap key: '{}'.", key);
    }
    std::sort(group.groupKeys.begin(), group.groupKeys.end());
  }
  NIMBLE_USER_CHECK(
      foundDefault, "Hybrid FlatMap must have exactly one Default group.");
}

HybridFlatMap::HybridFlatMap()
    : groups_{{Group{.groupId = kDefaultGroupId, .groupKeys = {}}}} {}

HybridFlatMap::HybridFlatMap(ScalarKind keyKind)
    : keyKind_{keyKind},
      groups_{{Group{.groupId = kDefaultGroupId, .groupKeys = {}}}} {
  NIMBLE_USER_CHECK(
      supportedKeyKind(keyKind),
      "Unsupported Hybrid FlatMap key kind: {}.",
      keyKind);
}

HybridFlatMap::HybridFlatMap(
    ScalarKind keyKind,
    const HybridFlatMap& configuration)
    : HybridFlatMap{keyKind, groupsForKeyKind(keyKind, configuration)} {}

std::vector<HybridFlatMap::Group> HybridFlatMap::groupsForKeyKind(
    ScalarKind keyKind,
    const HybridFlatMap& configuration) {
  if (configuration.keyKind_.has_value()) {
    NIMBLE_USER_CHECK_EQ(
        *configuration.keyKind_,
        keyKind,
        "Hybrid FlatMap configuration is for keys of another type.");
  }
  return configuration.groups_;
}

void HybridFlatMap::addGroup(uint32_t groupId, std::vector<std::string> keys) {
  // A writer configuration ends with its Default group; schema metadata
  // (fromSchema()) need not.
  NIMBLE_USER_CHECK(
      !groups_.empty(),
      "Hybrid FlatMap groups can be added only to a writer configuration.");
  NIMBLE_USER_CHECK(
      isDefaultGroup(groups_.back().groupId),
      "Hybrid FlatMap groups can be added only to a writer configuration.");
  NIMBLE_USER_CHECK(
      !isDefaultGroup(groupId),
      "Hybrid FlatMap Default group keys are appended in first-seen order "
      "and cannot be configured.");
  // groups_ ends with the Default group, so the last explicit group, if any,
  // is the one before it.
  if (groups_.size() > 1) {
    NIMBLE_USER_CHECK_LT(
        groups_[groups_.size() - 2].groupId,
        groupId,
        "Hybrid FlatMap group IDs must be in ascending order.");
  }
  NIMBLE_USER_CHECK(
      !keys.empty(),
      "Hybrid FlatMap group must contain at least one key: {}.",
      groupId);
  folly::F14FastSet<std::string_view> existingKeys;
  for (const auto& group : groups_) {
    existingKeys.insert(group.groupKeys.begin(), group.groupKeys.end());
  }
  for (const auto& key : keys) {
    if (keyKind_.has_value()) {
      // Throws unless `key` is canonical; a canonical key is returned as is.
      canonicalKey(*keyKind_, key);
    } else {
      NIMBLE_USER_CHECK(!key.empty(), "Hybrid FlatMap key cannot be empty.");
    }
    const bool inserted = existingKeys.insert(key).second;
    NIMBLE_USER_CHECK(inserted, "Duplicate Hybrid FlatMap key: '{}'.", key);
  }
  std::sort(keys.begin(), keys.end());
  groups_.insert(
      groups_.end() - 1,
      Group{.groupId = groupId, .groupKeys = std::move(keys)});
}

HybridFlatMap HybridFlatMap::fromSchema(std::vector<Group> groups) {
  return HybridFlatMap{AsRecorded{}, std::move(groups)};
}

const HybridFlatMap::Group& HybridFlatMap::groupById(uint32_t groupId) const {
  const auto iterator = std::find_if(
      groups_.begin(), groups_.end(), [groupId](const auto& group) {
        return group.groupId == groupId;
      });
  NIMBLE_CHECK(
      iterator != groups_.end(),
      "Hybrid FlatMap group ID is missing: {}.",
      groupId);
  return *iterator;
}

std::string HybridFlatMap::serialize() const {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<uint32_t> groupIds;
  std::vector<uint32_t> groupKeyCounts;
  std::vector<flatbuffers::Offset<flatbuffers::String>> groupKeys;
  groupIds.reserve(groups_.size());
  groupKeyCounts.reserve(groups_.size());
  for (const auto& group : groups_) {
    NIMBLE_CHECK_LE(
        group.groupKeys.size(),
        std::numeric_limits<uint32_t>::max(),
        "Hybrid FlatMap group has too many keys: {}.",
        group.groupKeys.size());
    groupIds.push_back(group.groupId);
    groupKeyCounts.push_back(static_cast<uint32_t>(group.groupKeys.size()));
    for (const auto& key : group.groupKeys) {
      groupKeys.push_back(builder.CreateString(key));
    }
  }
  builder.Finish(
      serialization::CreateHybridFlatMap(
          builder,
          builder.CreateVector(groupIds),
          builder.CreateVector(groupKeyCounts),
          builder.CreateVector(groupKeys)));
  return {
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
}

HybridFlatMap HybridFlatMap::deserialize(std::string_view serialized) {
  flatbuffers::Verifier verifier{
      reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size()};
  NIMBLE_CHECK(
      verifier.VerifyBuffer<serialization::HybridFlatMap>(),
      "Hybrid FlatMap metadata attribute is malformed.");
  const auto* serializedMetadata =
      flatbuffers::GetRoot<serialization::HybridFlatMap>(serialized.data());
  const auto* groupIds = serializedMetadata->group_ids();
  const auto* groupKeyCounts = serializedMetadata->group_key_counts();
  const auto* groupKeys = serializedMetadata->group_keys();
  NIMBLE_CHECK_NOT_NULL(groupIds, "Hybrid FlatMap group IDs are missing.");
  NIMBLE_CHECK_NOT_NULL(
      groupKeyCounts, "Hybrid FlatMap group key counts are missing.");
  const auto numGroupIds = groupIds->size();
  // FlatBuffers writers may omit an empty vector. An omitted group_keys holds
  // zero keys, so the key count checks below accept it only when no group has
  // keys, as in Default-only metadata.
  const auto numKeys = groupKeys == nullptr ? 0 : groupKeys->size();
  NIMBLE_CHECK_EQ(
      numGroupIds,
      groupKeyCounts->size(),
      "Hybrid FlatMap group IDs and key counts must have the same size");
  std::vector<Group> groups;
  groups.reserve(numGroupIds);
  flatbuffers::uoffset_t groupKeyIndex{0};
  for (flatbuffers::uoffset_t i = 0; i < numGroupIds; ++i) {
    const auto numGroupKeys = groupKeyCounts->Get(i);
    NIMBLE_CHECK_LE(
        groupKeyIndex,
        numKeys,
        "Hybrid FlatMap group key counts must match group keys size");
    NIMBLE_CHECK_LE(
        numGroupKeys,
        numKeys - groupKeyIndex,
        "Hybrid FlatMap group key counts must match group keys size");
    auto& group = groups.emplace_back(
        Group{
            .groupId = groupIds->Get(i),
            .groupKeys = {},
        });
    group.groupKeys.reserve(numGroupKeys);
    for (uint32_t j = 0; j < numGroupKeys; ++j) {
      const auto* key = groupKeys->Get(groupKeyIndex++);
      NIMBLE_CHECK_NOT_NULL(key, "Hybrid FlatMap group key cannot be null.");
      group.groupKeys.push_back(key->str());
    }
  }
  NIMBLE_CHECK_EQ(
      groupKeyIndex,
      numKeys,
      "Hybrid FlatMap group key counts must match group keys size");
  return fromSchema(std::move(groups));
}

void HybridFlatMap::setAttribute(
    std::vector<std::pair<std::string, std::string>>& attributes) const {
  const auto isHybridFlatMapAttribute = [](const auto& attribute) {
    return attribute.first == kAttributeName;
  };
  const auto existing = std::find_if(
      attributes.begin(), attributes.end(), isHybridFlatMapAttribute);
  auto serialized = serialize();
  if (existing == attributes.end()) {
    attributes.emplace_back(kAttributeName, std::move(serialized));
  } else {
    existing->second = std::move(serialized);
  }

  NIMBLE_CHECK_EQ(
      std::count_if(
          attributes.begin(), attributes.end(), isHybridFlatMapAttribute),
      1,
      "Hybrid FlatMap metadata attribute must be unique.");
}

HybridFlatMap HybridFlatMap::getAttribute(
    const std::vector<std::pair<std::string, std::string>>& attributes) {
  const std::string* serialized{nullptr};
  for (const auto& attribute : attributes) {
    if (attribute.first == kAttributeName) {
      NIMBLE_CHECK_NULL(
          serialized, "Hybrid FlatMap metadata attribute must be unique.");
      serialized = &attribute.second;
    }
  }
  NIMBLE_CHECK_NOT_NULL(
      serialized, "Hybrid FlatMap metadata attribute is missing.");
  return deserialize(*serialized);
}

HybridFlatMap HybridFlatMap::extractAttribute(
    std::vector<std::pair<std::string, std::string>>& attributes) {
  auto hybridMap = getAttribute(attributes);
  std::erase_if(attributes, [](const auto& attribute) {
    return attribute.first == kAttributeName;
  });
  return hybridMap;
}

namespace detail {

void checkHybridFlatMapGroupKeys(
    uint32_t groupId,
    const std::vector<std::string>& groupKeys) {
  const bool isDefaultGroup = HybridFlatMap::isDefaultGroup(groupId);
  NIMBLE_CHECK(
      isDefaultGroup || !groupKeys.empty(),
      "Hybrid FlatMap group must contain at least one key: {}.",
      groupId);
  for (size_t i = 0; i < groupKeys.size(); ++i) {
    NIMBLE_CHECK(!groupKeys[i].empty(), "Hybrid FlatMap key cannot be empty.");
    if (isDefaultGroup || i == 0) {
      continue;
    }
    NIMBLE_CHECK_NE(
        groupKeys[i - 1],
        groupKeys[i],
        "Duplicate Hybrid FlatMap key: '{}'.",
        groupKeys[i]);
    NIMBLE_CHECK_LT(
        groupKeys[i - 1],
        groupKeys[i],
        "Hybrid FlatMap group keys must be sorted: {}.",
        groupId);
  }
}

} // namespace detail

} // namespace facebook::nimble
