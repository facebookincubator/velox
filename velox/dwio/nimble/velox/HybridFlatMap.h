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
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace facebook::nimble {

class HybridFlatMapType;

/// Reserved group ID for features absent from every configured group.
/// Reserved identifier for the Default group, which may appear at most once.
constexpr uint32_t kHybridFlatMapDefaultGroupId = UINT32_MAX;

/// Defines the feature catalog for one logical group.
struct HybridFlatMapLogicalGroup {
  /// Stable group ID, including `kHybridFlatMapDefaultGroupId`.
  uint32_t groupId{0};
  /// Canonical FlatMap key names in key-has-entries bitmap order.
  std::vector<std::string> keys;
};

/// Defines the complete logical feature catalog for a hybrid FlatMap.
/// Every feature is listed in exactly one explicitly identified group.
struct HybridFlatMapLayout {
  std::vector<HybridFlatMapLogicalGroup> groups;
};

/// Identifies the physical streams for one hybrid FlatMap group.
struct HybridFlatMapGroup {
  /// Explicit logical group ID or `kHybridFlatMapDefaultGroupId`.
  uint32_t groupId{0};
  /// For each encoded batch containing this group, one bit per catalog key.
  /// A true bit means the key has at least one map entry and therefore owns
  /// the next key-major segment in the in-map stream. Dense and sparse keys
  /// both set this bit.
  uint32_t keyHasEntriesStreamOffset{0};
  /// Concatenated key-major row-presence bits.
  uint32_t inMapStreamOffset{0};
  /// Value-subtree streams in descriptor traversal order.
  std::vector<uint32_t> valueStreamOffsets;
};

/// Maps hybrid FlatMap groups to their physical stream offsets.
struct HybridFlatMapPhysicalLayout {
  /// Physical streams keyed by logical group ID.
  std::vector<HybridFlatMapGroup> groups;
};

/// Type attribute containing logical group IDs and their ordered key catalogs.
/// The key order defines bit positions in each group's key-has-entries stream.
constexpr std::string_view kHybridFlatMapLogicalGroupingAttribute{
    "nimble.hybrid_flatmap.logical_grouping.v1"};
/// Type attribute mapping each logical group ID to its key-has-entries, in-map,
/// and value-subtree stream offsets. Projection rewrites these to
/// response-local offsets. A cacheable schema retains slots for groups absent
/// from a payload.
constexpr std::string_view kHybridFlatMapPhysicalStreamsAttribute{
    "nimble.hybrid_flatmap.physical_streams.v1"};
/// Optional request-specific attribute containing the exact projected keys.
/// It is omitted from cacheable all-group schemas, where the caller performs
/// exact key filtering after decoding. Absence makes all catalog keys eligible.
constexpr std::string_view kHybridFlatMapSelectedKeysAttribute{
    "nimble.hybrid_flatmap.selected_keys.v1"};

std::string serializeHybridFlatMapLayout(const HybridFlatMapLayout& layout);
std::optional<HybridFlatMapLayout> deserializeHybridFlatMapLayout(
    std::string_view value);
std::optional<HybridFlatMapLayout> hybridFlatMapLayout(
    const HybridFlatMapType& hybridMap);

std::string serializeHybridFlatMapPhysicalLayout(
    const HybridFlatMapPhysicalLayout& layout);
std::optional<HybridFlatMapPhysicalLayout>
deserializeHybridFlatMapPhysicalLayout(std::string_view value);
std::optional<HybridFlatMapPhysicalLayout> hybridFlatMapPhysicalLayout(
    const HybridFlatMapType& hybridMap);

std::string serializeHybridFlatMapSelectedKeys(
    const std::vector<std::string>& selectedKeys);
std::vector<std::string> deserializeHybridFlatMapSelectedKeys(
    std::string_view value);
std::optional<std::vector<std::string>> hybridFlatMapSelectedKeys(
    const HybridFlatMapType& hybridMap);

} // namespace facebook::nimble
