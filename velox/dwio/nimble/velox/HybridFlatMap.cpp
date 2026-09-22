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

#include <algorithm>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/velox/SchemaGenerated.h"

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

std::string HybridFlatMap::serialize() const {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<uint32_t> groupIds;
  std::vector<uint32_t> groupKeyCounts;
  std::vector<flatbuffers::Offset<flatbuffers::String>> groupKeys;
  groupIds.reserve(groups.size());
  groupKeyCounts.reserve(groups.size());
  for (const auto& group : groups) {
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
  NIMBLE_CHECK_NOT_NULL(groupKeys, "Hybrid FlatMap group keys are missing.");
  const auto numGroupIds = groupIds->size();
  const auto numKeys = groupKeys->size();
  NIMBLE_CHECK_EQ(
      numGroupIds,
      groupKeyCounts->size(),
      "Hybrid FlatMap group IDs and key counts must have the same size");

  HybridFlatMap hybridMap;
  hybridMap.groups.reserve(numGroupIds);
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
    auto& group =
        hybridMap.groups.emplace_back(Group{.groupId = groupIds->Get(i)});
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
  validate(
      hybridMap.groups.size(),
      [&hybridMap](size_t index) { return hybridMap.groups[index].groupId; },
      [&hybridMap](size_t index) -> const auto& {
        return hybridMap.groups[index].groupKeys;
      });
  return hybridMap;
}

void HybridFlatMap::setAttribute(
    std::vector<std::pair<std::string, std::string>>& attributes) const {
  auto serialized = serialize();
  const bool alreadySet = std::any_of(
      attributes.begin(), attributes.end(), [](const auto& attribute) {
        return attribute.first == kAttributeName;
      });
  NIMBLE_CHECK(
      !alreadySet, "Hybrid FlatMap metadata attribute already exists.");
  attributes.emplace_back(kAttributeName, std::move(serialized));
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

} // namespace facebook::nimble
