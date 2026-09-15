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
#include <charconv>
#include <system_error>
#include <utility>

#include "folly/container/F14Set.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/velox/SchemaReader.h"

namespace facebook::nimble {
namespace {

constexpr std::string_view kLayoutName{"hybrid FlatMap layout"};
constexpr std::string_view kPhysicalLayoutName{
    "hybrid FlatMap physical layout"};
constexpr std::string_view kSelectedKeysName{"hybrid FlatMap selected keys"};
constexpr std::string_view kAllBitmapPrefix{"all_bitmap;"};

template <typename T>
T parseUnsignedToken(
    std::string_view token,
    std::string_view layoutName,
    std::string_view fieldName) {
  T result{0};
  const auto [end, error] =
      std::from_chars(token.data(), token.data() + token.size(), result);
  NIMBLE_CHECK(
      error == std::errc{} && end == token.data() + token.size(),
      "Invalid {}: malformed {}.",
      layoutName,
      fieldName);
  return result;
}

std::string_view readDelimitedToken(
    std::string_view value,
    size_t& offset,
    char delimiter,
    std::string_view layoutName,
    std::string_view fieldName) {
  const auto delimiterOffset = value.find(delimiter, offset);
  NIMBLE_CHECK(
      delimiterOffset != std::string_view::npos,
      "Invalid {}: missing {} delimiter.",
      layoutName,
      fieldName);
  const auto token = value.substr(offset, delimiterOffset - offset);
  offset = delimiterOffset + 1;
  return token;
}

std::string readLengthPrefixedString(
    std::string_view value,
    size_t& offset,
    std::string_view layoutName) {
  const auto length = parseUnsignedToken<size_t>(
      readDelimitedToken(value, offset, ':', layoutName, "key length"),
      layoutName,
      "key length");
  NIMBLE_CHECK_LE(
      length, value.size() - offset, "Invalid {}: truncated key.", layoutName);
  auto result = std::string(value.substr(offset, length));
  offset += length;
  return result;
}

void appendLengthPrefixedString(std::string& output, std::string_view value) {
  output.append(std::to_string(value.size()));
  output.push_back(':');
  output.append(value);
}

std::optional<std::string_view> schemaAttribute(
    const Type& type,
    std::string_view attributeName) {
  for (const auto& [key, value] : type.attributes()) {
    if (key == attributeName) {
      return value;
    }
  }
  return std::nullopt;
}

} // namespace

std::string serializeHybridFlatMapLayout(const HybridFlatMapLayout& layout) {
  folly::F14FastSet<uint32_t> groupIds;
  folly::F14FastSet<std::string_view> keys;
  std::string output;
  output.append(std::to_string(layout.groups.size()));
  output.push_back(';');
  for (const auto& group : layout.groups) {
    const bool uniqueGroupId = groupIds.insert(group.groupId).second;
    if (group.groupId == kHybridFlatMapDefaultGroupId) {
      NIMBLE_CHECK(
          uniqueGroupId,
          "Found groups with reserved default group ID {}",
          kHybridFlatMapDefaultGroupId);
    } else {
      NIMBLE_CHECK(
          uniqueGroupId,
          "Duplicate group ID {} in hybrid FlatMap logical layout.",
          group.groupId);
    }
    output.append(std::to_string(group.groupId));
    output.push_back(';');
    output.append(std::to_string(group.keys.size()));
    output.push_back(';');
    for (const auto& key : group.keys) {
      NIMBLE_CHECK(!key.empty(), "Hybrid FlatMap key cannot be empty.");
      NIMBLE_CHECK(
          keys.insert(key).second,
          "Duplicate key '{}' in hybrid FlatMap logical groups.",
          key);
      appendLengthPrefixedString(output, key);
      output.push_back(';');
    }
  }
  return output;
}

std::optional<HybridFlatMapLayout> deserializeHybridFlatMapLayout(
    std::string_view value) {
  NIMBLE_CHECK(!value.empty(), "Invalid {}: empty payload.", kLayoutName);

  size_t offset{0};
  const auto numGroups = parseUnsignedToken<size_t>(
      readDelimitedToken(value, offset, ';', kLayoutName, "group count"),
      kLayoutName,
      "group count");
  NIMBLE_CHECK_LE(
      numGroups,
      value.size() - offset,
      "Invalid {}: group count is too large.",
      kLayoutName);

  HybridFlatMapLayout layout;
  layout.groups.reserve(numGroups);
  for (size_t groupIndex = 0; groupIndex < numGroups; ++groupIndex) {
    const auto groupId = parseUnsignedToken<uint32_t>(
        readDelimitedToken(value, offset, ';', kLayoutName, "group ID"),
        kLayoutName,
        "group ID");
    const auto numKeys = parseUnsignedToken<size_t>(
        readDelimitedToken(value, offset, ';', kLayoutName, "key count"),
        kLayoutName,
        "key count");
    NIMBLE_CHECK_LE(
        numKeys,
        value.size() - offset,
        "Invalid {}: key count is too large.",
        kLayoutName);

    auto& group = layout.groups.emplace_back(
        HybridFlatMapLogicalGroup{.groupId = groupId});
    group.keys.reserve(numKeys);
    for (size_t keyIndex = 0; keyIndex < numKeys; ++keyIndex) {
      group.keys.push_back(
          readLengthPrefixedString(value, offset, kLayoutName));
      NIMBLE_CHECK_LT(
          offset,
          value.size(),
          "Invalid {}: missing key delimiter.",
          kLayoutName);
      NIMBLE_CHECK_EQ(
          value[offset++],
          ';',
          "Invalid {}: malformed key delimiter.",
          kLayoutName);
    }
  }
  NIMBLE_CHECK_EQ(
      offset, value.size(), "Invalid {}: trailing bytes.", kLayoutName);
  return layout;
}

std::optional<HybridFlatMapLayout> hybridFlatMapLayout(
    const HybridFlatMapType& hybridMap) {
  const auto value =
      schemaAttribute(hybridMap, kHybridFlatMapLogicalGroupingAttribute);
  return value.has_value() ? deserializeHybridFlatMapLayout(*value)
                           : std::nullopt;
}

std::string serializeHybridFlatMapPhysicalLayout(
    const HybridFlatMapPhysicalLayout& layout) {
  std::string output{kAllBitmapPrefix};
  output.append(std::to_string(layout.groups.size()));
  output.push_back(';');
  for (const auto& group : layout.groups) {
    output.append(std::to_string(group.groupId));
    output.push_back(';');
    output.append(std::to_string(group.keyHasEntriesStreamOffset));
    output.push_back(';');
    output.append(std::to_string(group.inMapStreamOffset));
    output.push_back(';');
    output.append(std::to_string(group.valueStreamOffsets.size()));
    output.push_back(';');
    for (const auto streamOffset : group.valueStreamOffsets) {
      output.append(std::to_string(streamOffset));
      output.push_back(';');
    }
  }
  return output;
}

std::optional<HybridFlatMapPhysicalLayout>
deserializeHybridFlatMapPhysicalLayout(std::string_view value) {
  NIMBLE_CHECK(
      !value.empty(), "Invalid {}: empty payload.", kPhysicalLayoutName);
  NIMBLE_CHECK(
      value.starts_with(kAllBitmapPrefix),
      "Invalid {}: unsupported key-has-entries encoding.",
      kPhysicalLayoutName);

  size_t offset = kAllBitmapPrefix.size();
  auto readPhysicalField = [&](std::string_view fieldName) {
    return parseUnsignedToken<uint32_t>(
        readDelimitedToken(value, offset, ';', kPhysicalLayoutName, fieldName),
        kPhysicalLayoutName,
        fieldName);
  };

  HybridFlatMapPhysicalLayout layout;
  const auto numGroups = parseUnsignedToken<size_t>(
      readDelimitedToken(
          value, offset, ';', kPhysicalLayoutName, "group count"),
      kPhysicalLayoutName,
      "group count");
  NIMBLE_CHECK_LE(
      numGroups,
      value.size() - offset,
      "Invalid {}: group count is too large.",
      kPhysicalLayoutName);
  layout.groups.reserve(numGroups);
  for (size_t groupIndex = 0; groupIndex < numGroups; ++groupIndex) {
    HybridFlatMapGroup group{
        .groupId = readPhysicalField("group id"),
        .keyHasEntriesStreamOffset =
            readPhysicalField("key-has-entries stream offset"),
        .inMapStreamOffset = readPhysicalField("in-map stream offset"),
    };
    const auto valueStreamCount = readPhysicalField("value stream count");
    NIMBLE_CHECK_GT(
        valueStreamCount,
        0,
        "Invalid {}: value stream count must be positive.",
        kPhysicalLayoutName);
    NIMBLE_CHECK_LE(
        valueStreamCount,
        value.size() - offset,
        "Invalid {}: value stream count is too large.",
        kPhysicalLayoutName);
    group.valueStreamOffsets.reserve(valueStreamCount);
    for (uint32_t i = 0; i < valueStreamCount; ++i) {
      group.valueStreamOffsets.push_back(
          readPhysicalField("value stream offset"));
    }
    layout.groups.push_back(std::move(group));
  }
  NIMBLE_CHECK_EQ(
      offset, value.size(), "Invalid {}: trailing bytes.", kPhysicalLayoutName);
  return layout;
}

std::optional<HybridFlatMapPhysicalLayout> hybridFlatMapPhysicalLayout(
    const HybridFlatMapType& hybridMap) {
  const auto value =
      schemaAttribute(hybridMap, kHybridFlatMapPhysicalStreamsAttribute);
  return value.has_value() ? deserializeHybridFlatMapPhysicalLayout(*value)
                           : std::nullopt;
}

std::string serializeHybridFlatMapSelectedKeys(
    const std::vector<std::string>& selectedKeys) {
  NIMBLE_CHECK(
      std::is_sorted(selectedKeys.begin(), selectedKeys.end()),
      "Hybrid FlatMap selected keys must be sorted.");
  folly::F14FastSet<std::string_view> uniqueKeys;
  std::string output = std::to_string(selectedKeys.size());
  output.push_back(';');
  for (const auto& key : selectedKeys) {
    NIMBLE_CHECK(
        uniqueKeys.insert(key).second,
        "Duplicate key '{}' in hybrid FlatMap selection.",
        key);
    appendLengthPrefixedString(output, key);
    output.push_back(';');
  }
  return output;
}

std::vector<std::string> deserializeHybridFlatMapSelectedKeys(
    std::string_view value) {
  NIMBLE_CHECK(!value.empty(), "Invalid {}: empty payload.", kSelectedKeysName);
  size_t offset{0};
  const auto keyCount = parseUnsignedToken<size_t>(
      readDelimitedToken(value, offset, ';', kSelectedKeysName, "key count"),
      kSelectedKeysName,
      "key count");
  NIMBLE_CHECK_LE(
      keyCount,
      value.size() - offset,
      "Invalid {}: key count is too large.",
      kSelectedKeysName);
  std::vector<std::string> selectedKeys;
  selectedKeys.reserve(keyCount);
  folly::F14FastSet<std::string_view> uniqueKeys;
  for (size_t i = 0; i < keyCount; ++i) {
    auto key = readLengthPrefixedString(value, offset, kSelectedKeysName);
    NIMBLE_CHECK_LT(
        offset,
        value.size(),
        "Invalid {}: missing key delimiter.",
        kSelectedKeysName);
    NIMBLE_CHECK_EQ(
        value[offset++],
        ';',
        "Invalid {}: malformed key delimiter.",
        kSelectedKeysName);
    selectedKeys.push_back(std::move(key));
    NIMBLE_CHECK(
        uniqueKeys.insert(selectedKeys.back()).second,
        "Duplicate key '{}' in hybrid FlatMap selection.",
        selectedKeys.back());
  }
  NIMBLE_CHECK_EQ(
      offset, value.size(), "Invalid {}: trailing bytes.", kSelectedKeysName);
  return selectedKeys;
}

std::optional<std::vector<std::string>> hybridFlatMapSelectedKeys(
    const HybridFlatMapType& hybridMap) {
  const auto value =
      schemaAttribute(hybridMap, kHybridFlatMapSelectedKeysAttribute);
  return value.has_value()
      ? std::make_optional(deserializeHybridFlatMapSelectedKeys(*value))
      : std::nullopt;
}

} // namespace facebook::nimble
