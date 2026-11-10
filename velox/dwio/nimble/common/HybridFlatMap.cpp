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
#include <numeric>

#include <folly/container/F14Map.h>
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
std::string canonicalIntegerKey(std::string_view key) {
  HybridFlatMap::parseCanonicalIntegerKey<T>(key);
  return std::string(key);
}

// Sorts `keys` as strings. Grouped keys usually arrive sorted, so a sorted
// group costs one comparison per key.
void sortGroupKeys(std::vector<std::string>& keys) {
  if (!std::is_sorted(keys.begin(), keys.end())) {
    std::sort(keys.begin(), keys.end());
  }
}

} // namespace

std::string HybridFlatMap::canonicalKey(
    ScalarKind keyKind,
    std::string_view key) {
  NIMBLE_USER_CHECK(!key.empty(), "Hybrid FlatMap key cannot be empty.");
  switch (keyKind) {
    case ScalarKind::Int8:
      return canonicalIntegerKey<int8_t>(key);
    case ScalarKind::Int16:
      return canonicalIntegerKey<int16_t>(key);
    case ScalarKind::Int32:
      return canonicalIntegerKey<int32_t>(key);
    case ScalarKind::Int64:
      return canonicalIntegerKey<int64_t>(key);
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

namespace {

// Returns the number of keys in `groups`.
size_t totalNumKeys(const std::vector<HybridFlatMap::Group>& groups) {
  return std::accumulate(
      groups.begin(),
      groups.end(),
      size_t{0},
      [](size_t numKeys, const HybridFlatMap::Group& group) {
        return numKeys + group.groupKeys.size();
      });
}

// Returns the index in `groups` of the group of each key, viewing the keys of
// `groups`. The writer constructor has rejected a key repeated within or
// across groups, and a key in the Default group.
folly::F14FastMap<std::string_view, size_t> buildIndexGroupsByKey(
    const std::vector<HybridFlatMap::Group>& groups) {
  // Reserves room for every grouped key, which can number in the thousands,
  // so the index never rehashes as it grows.
  folly::F14FastMap<std::string_view, size_t> groupIndexByKey;
  groupIndexByKey.reserve(totalNumKeys(groups));
  for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
    for (const auto& key : groups[groupIndex].groupKeys) {
      const bool inserted = groupIndexByKey.emplace(key, groupIndex).second;
      NIMBLE_CHECK(inserted, "Duplicate Hybrid FlatMap key: '{}'.", key);
    }
  }
  return groupIndexByKey;
}

} // namespace

HybridFlatMap::HybridFlatMap(
    ScalarKind keyKind,
    std::vector<Group> groups,
    KeyPresenceMode defaultGroupKeyPresenceMode)
    : keyKind_{keyKind},
      defaultGroupKeyPresenceMode_{defaultGroupKeyPresenceMode} {
  NIMBLE_USER_CHECK(
      supportedKeyKind(keyKind),
      "Unsupported Hybrid FlatMap key kind: {}.",
      keyKind);

  // Reserves room for every grouped key, which can number in the thousands,
  // so the duplicate check never rehashes the set as it grows. The set views
  // the keys of `groups`, so they are sorted only after every check.
  folly::F14FastSet<std::string_view> keys;
  keys.reserve(totalNumKeys(groups));
  bool foundDefault{false};
  for (size_t i = 0; i < groups.size(); ++i) {
    const auto& group = groups[i];
    if (isDefaultGroup(group.groupId)) {
      NIMBLE_USER_CHECK(
          !foundDefault, "Hybrid FlatMap has multiple Default groups.");
    }
    // Ascending IDs are also distinct, and put the Default group, whose ID is
    // the largest, last.
    if (i > 0) {
      NIMBLE_USER_CHECK_LT(
          groups[i - 1].groupId,
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
      const bool inserted = keys.insert(key).second;
      NIMBLE_USER_CHECK(inserted, "Duplicate Hybrid FlatMap key: '{}'.", key);
    }
  }
  NIMBLE_USER_CHECK(
      foundDefault, "Hybrid FlatMap must have exactly one Default group.");

  for (auto& group : groups) {
    sortGroupKeys(group.groupKeys);
  }
  groups_ = std::make_shared<std::vector<Group>>(std::move(groups));
  groupedKeyIndex_ =
      std::make_shared<const folly::F14FastMap<std::string_view, size_t>>(
          buildIndexGroupsByKey(*groups_));
}

HybridFlatMap::HybridFlatMap(
    std::vector<Group> groups,
    KeyPresenceMode defaultGroupKeyPresenceMode)
    : groups_{std::make_shared<std::vector<Group>>(std::move(groups))},
      defaultGroupKeyPresenceMode_{defaultGroupKeyPresenceMode} {}

HybridFlatMap HybridFlatMap::fromSchema(
    std::vector<Group> groups,
    KeyPresenceMode defaultGroupKeyPresenceMode) {
  return HybridFlatMap{std::move(groups), defaultGroupKeyPresenceMode};
}

const std::vector<HybridFlatMap::Group>& HybridFlatMap::groups() const {
  return *groups_;
}

std::vector<HybridFlatMap::Group> HybridFlatMap::releaseGroups() && {
  // The grouped-key index views the keys of groups_ and is shared exactly as
  // groups_ is, so a sole owner may move them out of this expiring
  // HybridFlatMap.
  if (groups_.use_count() == 1) {
    return std::move(*groups_);
  }
  return *groups_;
}

std::optional<size_t> HybridFlatMap::findGroup(std::string_view key) const {
  NIMBLE_CHECK_NOT_NULL(
      groupedKeyIndex_,
      "Hybrid FlatMap has no grouped-key index: schema metadata holds none.");
  const auto it = groupedKeyIndex_->find(key);
  if (it == groupedKeyIndex_->end()) {
    return std::nullopt;
  }
  return it->second;
}

size_t HybridFlatMap::numGroupedKeys() const {
  NIMBLE_CHECK_NOT_NULL(
      groupedKeyIndex_,
      "Hybrid FlatMap has no grouped-key index: schema metadata holds none.");
  return groupedKeyIndex_->size();
}

bool HybridFlatMap::operator==(const HybridFlatMap& other) const {
  return keyKind_ == other.keyKind_ && groups() == other.groups() &&
      defaultGroupKeyPresenceMode_ == other.defaultGroupKeyPresenceMode_;
}

const HybridFlatMap::Group& HybridFlatMap::groupById(uint32_t groupId) const {
  const auto& allGroups = groups();
  const auto iterator = std::find_if(
      allGroups.begin(), allGroups.end(), [groupId](const auto& group) {
        return group.groupId == groupId;
      });
  NIMBLE_CHECK(
      iterator != allGroups.end(),
      "Hybrid FlatMap group ID is missing: {}.",
      groupId);
  return *iterator;
}

std::string HybridFlatMap::serialize() const {
  flatbuffers::FlatBufferBuilder builder;
  // Writes the Default key presence mode even when it is Bitmap, since
  // deserialize() requires it.
  builder.ForceDefaults(true);
  std::vector<uint32_t> groupIds;
  std::vector<uint32_t> groupKeyCounts;
  std::vector<flatbuffers::Offset<flatbuffers::String>> groupKeys;
  const auto& allGroups = groups();
  groupIds.reserve(allGroups.size());
  groupKeyCounts.reserve(allGroups.size());
  for (const auto& group : allGroups) {
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
          builder.CreateVector(groupKeys),
          static_cast<serialization::HybridFlatMapKeyPresenceMode>(
              defaultGroupKeyPresenceMode_)));
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
  const auto* table =
      flatbuffers::GetRoot<flatbuffers::Table>(serialized.data());
  NIMBLE_CHECK(
      table->CheckField(
          serialization::HybridFlatMap::VT_DEFAULT_GROUP_KEY_PRESENCE_MODE),
      "Hybrid FlatMap Default key presence is missing.");
  // The generated accessor converts an unknown value to the enum, which is
  // undefined behavior, so check the raw byte first.
  const auto defaultGroupKeyPresenceMode = table->GetField<uint8_t>(
      serialization::HybridFlatMap::VT_DEFAULT_GROUP_KEY_PRESENCE_MODE, 0);
  NIMBLE_CHECK_LE(
      defaultGroupKeyPresenceMode,
      static_cast<uint8_t>(serialization::HybridFlatMapKeyPresenceMode_MAX),
      "Hybrid FlatMap Default key presence is unknown.");
  const auto keyPresenceMode =
      static_cast<KeyPresenceMode>(defaultGroupKeyPresenceMode);
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
    detail::checkHybridFlatMapDefaultKeys(
        keyPresenceMode, group.groupId, group.groupKeys);
  }
  NIMBLE_CHECK_EQ(
      groupKeyIndex,
      numKeys,
      "Hybrid FlatMap group key counts must match group keys size");
  return fromSchema(std::move(groups), keyPresenceMode);
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

std::string toString(HybridFlatMap::KeyPresenceMode mode) {
  switch (mode) {
    case HybridFlatMap::KeyPresenceMode::kBitmap:
      return "Bitmap";
    case HybridFlatMap::KeyPresenceMode::kKeyList:
      return "KeyList";
  }
  NIMBLE_UNREACHABLE(
      "Unknown Hybrid FlatMap key presence mode: {}.",
      static_cast<int32_t>(mode));
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

    // Readers check every schema for every batch. One comparison accepts
    // ascending keys; the checks below only name the failure.
    if (groupKeys[i - 1] < groupKeys[i]) {
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

void checkHybridFlatMapDefaultKeys(
    HybridFlatMap::KeyPresenceMode defaultGroupKeyPresenceMode,
    uint32_t groupId,
    const std::vector<std::string>& groupKeys) {
  if (HybridFlatMap::hasKeyPresenceList(defaultGroupKeyPresenceMode, groupId)) {
    NIMBLE_CHECK(
        groupKeys.empty(),
        "Hybrid FlatMap Default group cannot list keys when batches list "
        "their Default keys.");
  }
}

} // namespace detail

} // namespace facebook::nimble
