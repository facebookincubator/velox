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

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>
#include <folly/Conv.h>
#include "folly/container/F14Map.h"
#include "folly/container/F14Set.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/SchemaTypes.h"
#include "velox/dwio/nimble/common/Types.h"

namespace facebook::velox::memory {
class MemoryPool;
} // namespace facebook::velox::memory

namespace facebook::nimble {

class EncodingLayout;

/// Groups the keys of a MAP column: a writer configuration, or the groups a
/// written schema records. Read-only once built. Its copies share its groups
/// and the key index, which maps each key a group lists to that group, so a
/// copy costs reference counts and threads may read copies concurrently.
class HybridFlatMap {
 public:
  /// How a group's key-presence stream records which keys a batch contains.
  /// Configured groups always use kBitmap; defaultGroupKeyPresenceMode()
  /// selects the mode of the Default group.
  enum class KeyPresenceMode : uint8_t {
    /// One bit per schema key of the group. Default appends newly seen keys to
    /// the schema in first-seen order, so the schema can change with the data,
    /// and a key the schema has not seen projects as no group.
    kBitmap = 0,
    /// Default lists no schema keys; each batch lists the typed Default keys
    /// it contains. The schema depends only on the configured groups, and a
    /// key in no configured group projects as the whole Default group.
    kKeyList = 1,
  };

  /// Describes one group in logical schema order.
  struct Group {
    uint32_t groupId{0};
    /// Keys in key-presence bitmap order. A writer configuration sorts
    /// explicit groups, by value for integer keys and as strings for String
    /// keys. A schema keeps the order it records: one written before integer
    /// keys sorted by value lists them as strings, e.g. "10" before "9".
    /// Default preserves first-seen order, and lists no keys with
    /// KeyPresenceMode::kKeyList.
    std::vector<std::string> groupKeys;
    /// The group's `encoded_group_keys` stream, when a schema recorded one.
    /// serialize() copies it instead of encoding the keys again, e.g. for a
    /// projection that keeps the group.
    std::string encodedKeys{};

    /// Compares the ID and the keys, however they were stored.
    bool operator==(const Group& other) const {
      return groupId == other.groupId && groupKeys == other.groupKeys;
    }
  };

  /// Reserved identifier for the Default group.
  static constexpr uint32_t kDefaultGroupId =
      std::numeric_limits<uint32_t>::max();

  /// Returns whether `groupId` identifies the reserved Default group.
  static constexpr bool isDefaultGroup(uint32_t groupId) {
    return groupId == kDefaultGroupId;
  }

  /// A writer configuration for keys of `keyKind`, whose batches record their
  /// Default keys with `defaultGroupKeyPresenceMode`. Checked with user errors
  /// and normalized: every key canonical (canonicalKey()), group IDs
  /// ascending, no key repeated within or across groups, every explicit
  /// group non-empty, and exactly one Default group, without keys. groups()
  /// then holds each group's keys sorted, by value for integer keys and as
  /// strings for String keys, the schema and key-presence bitmap order. Builds
  /// the key index of findGroup(), which copies share.
  ///
  /// Example: for Int64 keys, groups {0: ["10", "9"], Default: []} hold
  /// {0: ["9", "10"], Default: []}; {0: ["07"], Default: []} and
  /// {1: ["1"], 0: ["2"], Default: []} are rejected.
  HybridFlatMap(
      ScalarKind keyKind,
      std::vector<Group> groups,
      KeyPresenceMode defaultGroupKeyPresenceMode);

  /// The groups a written schema records for keys of `keyKind`, parsed from
  /// `serialized`, as serialize() writes them. Checked as a written schema:
  /// group IDs ascending, explicit groups non-empty, no key listed twice, and
  /// no Default keys with KeyPresenceMode::kKeyList. Keys keep the order the
  /// schema records, their key-presence bitmap order, so schemas written
  /// before integer keys sorted by value read with their keys sorted as
  /// strings. A bitmap Default lists the keys found while writing, in
  /// first-seen order, and a projection may leave groups out, Default
  /// included. Decodes `encoded_group_keys` back to the keys' canonical
  /// strings, allocating from `pool`, which must not be null.
  /// Also accepts omitted key vectors when no group has keys. Builds the key
  /// index of findGroup(), which copies share.
  HybridFlatMap(
      ScalarKind keyKind,
      std::string_view serialized,
      velox::memory::MemoryPool* pool);

  /// The key kind the groups were built or recorded for.
  ScalarKind keyKind() const {
    return keyKind_;
  }

  /// Logical groups in schema order. Copies share them, so they keep their
  /// address while any copy lives.
  const std::vector<Group>& groups() const;

  /// Returns the index in groups() of the group that lists `key`, or
  /// std::nullopt when no group lists it. A writer configuration's Default
  /// lists no keys, so a key it doesn't find belongs to Default. A schema's
  /// bitmap Default lists the keys it recorded.
  std::optional<size_t> findGroup(std::string_view key) const;

  /// Returns the number of keys the groups list. A writer configuration's
  /// Default lists none, so these are its grouped keys.
  size_t numGroupedKeys() const;

  /// How batches record their Default keys: the mode a writer configuration
  /// was built with, or the mode a schema records.
  KeyPresenceMode defaultGroupKeyPresenceMode() const {
    return defaultGroupKeyPresenceMode_;
  }

  /// Returns whether group `groupId` writes its key-presence stream as a key
  /// list: the batch's present keys, typed by the map key kind, instead of a
  /// bitmap over schema keys. True only for Default with
  /// KeyPresenceMode::kKeyList. Depends only on the schema, never on
  /// batch data.
  static constexpr bool hasKeyPresenceList(
      KeyPresenceMode defaultGroupKeyPresenceMode,
      uint32_t groupId) {
    return isDefaultGroup(groupId) &&
        defaultGroupKeyPresenceMode == KeyPresenceMode::kKeyList;
  }

  /// Returns whether group `groupId` lists its keys under
  /// defaultGroupKeyPresenceMode(). See hasKeyPresenceList().
  bool useKeyPresentList(uint32_t groupId) const {
    return hasKeyPresenceList(defaultGroupKeyPresenceMode_, groupId);
  }

  /// Returns the logical group with `groupId`. Fails when it is absent.
  const Group& groupById(uint32_t groupId) const;

  /// Compares keyKind(), groups() and defaultGroupKeyPresenceMode().
  bool operator==(const HybridFlatMap& other) const;

  /// Returns whether `kind` is supported for Hybrid FlatMap keys.
  static bool supportedKeyKind(ScalarKind kind);

  /// Returns `key` as a Hybrid FlatMap schema writes keys of `keyKind`: a
  /// String key unchanged, an integer key as the decimal form of its value.
  /// Throws a user error when `key` is empty, does not parse as `keyKind`, or
  /// is not written that way, e.g. "07" or "+7" for an integer kind.
  static std::string canonicalKey(ScalarKind keyKind, std::string_view key);

  /// Returns the value of `key` as a key of the integer type T. Throws the user
  /// error of canonicalKey() when it would reject `key`. Builds no string for a
  /// canonical key, so a reader can parse every schema key of every batch.
  template <typename T>
  static T parseCanonicalIntegerKey(std::string_view key);

  /// Serializes `groups` and `defaultGroupKeyPresenceMode` of a schema node
  /// whose keys are of `keyKind` into the flattened HybridFlatMap FlatBuffer
  /// table. `group_ids` and `group_key_counts` are parallel per-group vectors.
  /// The keys of every group follow in group order, so group `i` owns the
  /// `group_key_counts[i]` keys that follow the keys of the preceding groups.
  /// Keys are written as `group_keys` strings, except when some group has keys
  /// and either `groupKeysLayout` is not null or a group carries its recorded
  /// stream (Group::encodedKeys). Those are written as `encoded_group_keys`,
  /// one Nimble-encoded stream per group of the keys' type, e.g. Int32 for
  /// Int32 keys and String for String keys, empty for a group without keys:
  /// the recorded stream as is, e.g. for a projection, or the keys encoded in
  /// exactly `groupKeysLayout`, which must name every nested encoding. A bitmap
  /// Default's keys are encoded in their first-seen order. When it writes
  /// `encoded_group_keys`, serialize() throws for keys the layout cannot
  /// encode, e.g. Prefix for integer keys, EliasFano for String keys or for a
  /// bitmap Default's keys out of order, a missing nested encoding, or a stream
  /// under one bit per key, which readers reject. It also throws then for an
  /// explicit group without keys, and for a group with keys but neither a
  /// recorded stream nor a layout. Compression is always off, including any
  /// that the layout names, so readers need no codec. Encoding allocates from
  /// `pool`, which must not be null.
  /// The Default key presence mode is always written. Takes groups rather
  /// than a HybridFlatMap, since a type builder's bitmap Default lists the keys
  /// found while writing, which no writer configuration holds.
  static std::string serialize(
      ScalarKind keyKind,
      const std::vector<Group>& groups,
      KeyPresenceMode defaultGroupKeyPresenceMode,
      const EncodingLayout* groupKeysLayout,
      velox::memory::MemoryPool* pool);

 private:
  // See keyKind().
  const ScalarKind keyKind_;
  // See groups(). Const and shared by every copy, on any thread.
  std::shared_ptr<const std::vector<Group>> groups_;
  // The index in groups_ of the group of each key a group lists, viewing the
  // keys of groups_. Built once groups_ holds the keys, since sorting or moving
  // them moves them. Const and shared by every copy, on any thread. Declared
  // after groups_, so it is destroyed first.
  std::shared_ptr<const folly::F14FastMap<std::string_view, size_t>>
      groupedKeyIndex_;
  // See defaultGroupKeyPresenceMode().
  const KeyPresenceMode defaultGroupKeyPresenceMode_;
};

// Defined in the header so that readers inline it into their per-key loops.
template <typename T>
T HybridFlatMap::parseCanonicalIntegerKey(std::string_view key) {
  T value{};
  const auto* end = key.data() + key.size();
  const auto [parsedEnd, error] = std::from_chars(key.data(), end, value);
  if (error == std::errc{} && parsedEnd == end) {
    // std::from_chars accepts leading zeros, which only the key "0" may have.
    const size_t firstDigit{key.front() == '-' ? 1U : 0U};
    if (key[firstDigit] != '0' || (firstDigit == 0 && key.size() == 1)) {
      return value;
    }
  }
  NIMBLE_USER_CHECK(
      folly::tryTo<T>(key).hasValue(),
      "Hybrid FlatMap key '{}' cannot be parsed as {}.",
      key,
      TypeTraits<T>::dataType);
  NIMBLE_USER_FAIL("Hybrid FlatMap key '{}' is not canonical.", key);
}

/// Returns the name of `mode` as spelled in the serialized schema.
std::string toString(HybridFlatMap::KeyPresenceMode mode);

/// Locates one Hybrid FlatMap group's entries in one decode run, which is a
/// file stripe or a Serializer batch.
///
/// Example: keys {"7", "9", "11"}, key presence [1, 0, 1] and in-map
/// [1, 0, 1, 0, 1, 1] describe three non-null maps. "7" is in maps 0 and 2,
/// and "11" is in maps 1 and 2. The layout is numNonNullRows 3, presentKeys
/// {0, 2} and valueStarts {0, 2, 4}.
struct HybridFlatMapGroupLayout {
  /// Non-null maps covered by each present key's in-map bits. 0 when empty().
  uint32_t numNonNullRows{0};
  /// Ordinals of the present keys among the group's keys, ascending. Present
  /// key p owns in-map bits [p * numNonNullRows, (p + 1) * numNonNullRows).
  std::vector<uint32_t> presentKeys;
  /// Present key p owns values [valueStarts[p], valueStarts[p + 1]) of the
  /// group's key-major value stream. Holds presentKeys.size() + 1 entries, or
  /// none when empty().
  std::vector<uint32_t> valueStarts;

  bool empty() const {
    return presentKeys.empty();
  }

  uint32_t numValues() const {
    return valueStarts.empty() ? 0 : valueStarts.back();
  }

  /// Empties the layout and keeps the capacity of its vectors.
  void clear();
};

/// Sizes of one Hybrid FlatMap group's metadata in one decode run.
struct HybridFlatMapGroupMetadata {
  /// Keys that key presence refers to: the group's schema keys, or the keys
  /// the group lists in this decode run with KeyPresenceMode::kKeyList.
  uint32_t numKeys{0};
  /// All bits in the key-presence bitmap, at most numKeys. 0 when the bitmap
  /// is omitted or the group lists its keys, in which case every key is
  /// present.
  uint32_t numAllKeyPresenceBits{0};
  /// Set bits in the key-presence bitmap: the present keys. Readers count them
  /// over their own bit layout, e.g. a word popcount over packed bits, so the
  /// layout is found in one pass over the keys. Ignored when
  /// numAllKeyPresenceBits is 0.
  uint32_t numSetKeyPresenceBits{0};
  /// Bits in the group's key-major in-map stream.
  uint32_t numInMap{0};
  /// Whether the group lists its keys in each decode run.
  bool listsKeys{false};
  /// Whether any stream of the group's value subtree has data in this decode
  /// run.
  bool hasValueStreams{false};
};

/// Builds `layout` from one group's metadata by splitting its in-map bits
/// evenly among the present keys and starting each present key's values after
/// those of the present keys before it.
template <typename IsKeyPresent, typename CountKeyValues, typename KeyName>
void decodeHybridFlatMapGroupLayout(
    const HybridFlatMapGroupMetadata& metadata,
    const IsKeyPresent& isKeyPresent,
    const CountKeyValues& countKeyValues,
    const KeyName& keyName,
    HybridFlatMapGroupLayout& layout) {
  layout.clear();

  // The writer omits all-false in-map metadata. With no value streams, the
  // group therefore has no entries in this decode run, even if key presence was
  // retained to preserve the schema prefix.
  if (metadata.numInMap == 0 && !metadata.hasValueStreams) {
    if (metadata.listsKeys) {
      NIMBLE_CHECK_FILE_EQ(
          metadata.numKeys,
          0,
          "Hybrid FlatMap group lists keys without in-map rows.");
    }
    return;
  }

  // A key list carries no presence bits; every listed key is present. An
  // omitted key-presence bitmap with in-map rows applies to all schema keys.
  // All-false metadata is omitted only when the group is empty.
  const bool allKeysPresent = metadata.numAllKeyPresenceBits == 0;
  NIMBLE_CHECK_FILE_LE(
      metadata.numAllKeyPresenceBits,
      metadata.numKeys,
      "Hybrid FlatMap key-presence count exceeds schema key count.");
  const uint32_t numKeys =
      allKeysPresent ? metadata.numKeys : metadata.numAllKeyPresenceBits;

  // numNonNullRows locates each key's in-map bits, and it needs the number of
  // present keys before the pass over the keys.
  const uint32_t numPresentKeys =
      allKeysPresent ? numKeys : metadata.numSetKeyPresenceBits;
  if (numPresentKeys == 0) {
    NIMBLE_CHECK_FILE_EQ(
        metadata.numInMap,
        0,
        "Hybrid FlatMap group has in-map rows without present keys.");
    NIMBLE_CHECK_FILE(
        !metadata.hasValueStreams,
        "Hybrid FlatMap group has value streams without present keys.");
    return;
  }
  NIMBLE_CHECK_FILE_NE(
      metadata.numInMap,
      0,
      "Hybrid FlatMap group with present keys is missing in-map rows.");
  NIMBLE_CHECK_FILE_EQ(
      metadata.numInMap % numPresentKeys,
      0,
      "Hybrid FlatMap in-map count is not divisible by present key "
      "count.");

  // Each present key has one in-map bit per non-null map.
  layout.numNonNullRows = metadata.numInMap / numPresentKeys;
  layout.presentKeys.reserve(numPresentKeys);
  layout.valueStarts.reserve(numPresentKeys + 1);

  // Records each present key, in key order, with the start of its values.
  // `isKeyPresent(key)` reads the key's presence bit. `countKeyValues(key,
  // presentKey, firstValue)` returns the number of set in-map bits of present
  // key `presentKey`, which are its values, starting at value `firstValue`.
  // It can locate those bits because numNonNullRows is already set. The
  // function is defined in the header so that readers inline these callbacks
  // into the loop.
  uint32_t numValues{0};
  for (uint32_t key = 0; key < numKeys; ++key) {
    if (!allKeysPresent && !isKeyPresent(key)) {
      continue;
    }
    const auto presentKey = static_cast<uint32_t>(layout.presentKeys.size());

    // Keeps the in-map bits of `presentKey` within numInMap.
    NIMBLE_CHECK_LT(
        presentKey,
        numPresentKeys,
        "Hybrid FlatMap present key count does not match the key-presence "
        "bitmap.");
    const auto numKeyValues = countKeyValues(key, presentKey, numValues);
    NIMBLE_CHECK_FILE_GT(
        numKeyValues,
        0,
        "Hybrid FlatMap key '{}' has no present rows.",
        keyName(key));
    layout.presentKeys.push_back(key);
    layout.valueStarts.push_back(numValues);
    numValues += numKeyValues;
  }
  NIMBLE_CHECK_EQ(
      layout.presentKeys.size(),
      numPresentKeys,
      "Hybrid FlatMap present key count does not match the key-presence "
      "bitmap.");

  // Ends the values of the last present key.
  layout.valueStarts.push_back(numValues);
}

namespace detail {

// Checks the rules for one group's own keys, used only by whole-schema
// validation (validateHybridFlatMapGroups()): an explicit group must list a
// key, and no key may be empty. Keys may come in any order, which is their
// key-presence bitmap order. A writer configuration sorts explicit groups, but
// schemas written before integer keys sorted by value list them as strings,
// e.g. "10" before "9", and Default keeps its first-seen order. Callers check
// that every key is unique across all groups.
void checkHybridFlatMapGroupKeys(
    uint32_t groupId,
    const std::vector<std::string>& groupKeys);

// Rejects Default schema keys when batches list their Default keys, since the
// schema must then depend only on the configured groups.
void checkHybridFlatMapDefaultKeys(
    HybridFlatMap::KeyPresenceMode defaultGroupKeyPresenceMode,
    uint32_t groupId,
    const std::vector<std::string>& groupKeys);

// Validates the group IDs and keys of a whole schema, and maps each key to its
// group's index in `groupIndices`, which must start empty. The index also
// detects repeated keys, so a caller that needs it hashes each key once.
// Callers keep every group's keys alive and unchanged while they use
// `groupIndices`, so views avoid copying each key.
template <typename GroupIdAt, typename GroupKeysAt>
void validateHybridFlatMapGroups(
    size_t groupCount,
    bool hasDefault,
    GroupIdAt groupIdAt,
    GroupKeysAt groupKeysAt,
    folly::F14FastMap<std::string_view, size_t>& groupIndices) {
  NIMBLE_CHECK_GE(
      groupCount, size_t{1}, "Hybrid FlatMap requires at least 1 group(s).");
  // Default alone is valid because it holds every key. A projection may keep a
  // single configured group, so only a complete schema requires its single
  // group to be Default.
  if (hasDefault && groupCount == 1) {
    NIMBLE_CHECK(
        HybridFlatMap::isDefaultGroup(groupIdAt(0)),
        "Hybrid FlatMap single group must be Default: {}.",
        groupIdAt(0));
  }
  size_t numKeys{0};
  for (size_t i = 0; i < groupCount; ++i) {
    numKeys += groupKeysAt(i).size();
  }
  groupIndices.reserve(numKeys);
  folly::F14FastSet<uint32_t> groupIds;
  bool foundDefault{false};
  for (size_t i = 0; i < groupCount; ++i) {
    const auto groupId = groupIdAt(i);
    const auto& groupKeys = groupKeysAt(i);
    const bool isNewGroupId = groupIds.insert(groupId).second;
    NIMBLE_CHECK(
        isNewGroupId, "Duplicate Hybrid FlatMap group ID: {}.", groupId);
    if (i > 0) {
      NIMBLE_CHECK_LT(
          groupIdAt(i - 1),
          groupId,
          "Hybrid FlatMap group IDs must be in ascending order.");
    }
    foundDefault |= HybridFlatMap::isDefaultGroup(groupId);
    checkHybridFlatMapGroupKeys(groupId, groupKeys);
    for (const auto& key : groupKeys) {
      const bool isNewKey = groupIndices.emplace(key, i).second;
      NIMBLE_CHECK(isNewKey, "Duplicate Hybrid FlatMap key: '{}'.", key);
    }
  }
  if (hasDefault) {
    NIMBLE_CHECK(
        foundDefault, "Hybrid FlatMap must have exactly one Default group.");
  }
}

// Normalizes child accessors across the two schema-tree representations:
// TypeBuilder exposes children by reference (`const TypeBuilder&`) while Type
// exposes them through shared ownership (`const std::shared_ptr<const Type>&`).
template <typename T>
const T& dereferenceType(const T& type) {
  return type;
}

template <typename T>
const T& dereferenceType(const std::shared_ptr<T>& type) {
  return *type;
}

// Compares logical type shapes while ignoring stream descriptors and FlatMap
// key catalogs. Each FlatMap must use one consistent value type internally.
// Hybrid FlatMap returns false so it cannot appear inside a value subtree.
//
// `TypeLike` is duck-typed over Type and TypeBuilder, which share no base
// class, so no non-template signature accepts both. Both parameters bind the
// same `TypeLike`: a reader type cannot be compared against a writer type.
// The writer and reader paths instantiate it from separate translation units,
// so the definition has to stay in a header visible to both.
template <typename TypeLike>
bool sameLogicalType(const TypeLike& lhs, const TypeLike& rhs) {
  if (lhs.kind() != rhs.kind()) {
    return false;
  }
  switch (lhs.kind()) {
    case Kind::Scalar:
      return lhs.asScalar().scalarDescriptor().scalarKind() ==
          rhs.asScalar().scalarDescriptor().scalarKind();
    case Kind::TimestampMicroNano:
      return true;
    case Kind::Array:
      return sameLogicalType(
          dereferenceType(lhs.asArray().elements()),
          dereferenceType(rhs.asArray().elements()));
    case Kind::ArrayWithOffsets:
      return sameLogicalType(
          dereferenceType(lhs.asArrayWithOffsets().elements()),
          dereferenceType(rhs.asArrayWithOffsets().elements()));
    case Kind::SlidingWindowArray:
      return sameLogicalType(
          dereferenceType(lhs.asSlidingWindowArray().elements()),
          dereferenceType(rhs.asSlidingWindowArray().elements()));
    case Kind::Map:
      return sameLogicalType(
                 dereferenceType(lhs.asMap().keys()),
                 dereferenceType(rhs.asMap().keys())) &&
          sameLogicalType(
                 dereferenceType(lhs.asMap().values()),
                 dereferenceType(rhs.asMap().values()));
    case Kind::SlidingWindowMap:
      return sameLogicalType(
                 dereferenceType(lhs.asSlidingWindowMap().keys()),
                 dereferenceType(rhs.asSlidingWindowMap().keys())) &&
          sameLogicalType(
                 dereferenceType(lhs.asSlidingWindowMap().values()),
                 dereferenceType(rhs.asSlidingWindowMap().values()));
    case Kind::Row: {
      const auto& lhsRow = lhs.asRow();
      const auto& rhsRow = rhs.asRow();
      if (lhsRow.childrenCount() != rhsRow.childrenCount()) {
        return false;
      }
      for (size_t i = 0; i < lhsRow.childrenCount(); ++i) {
        if (lhsRow.nameAt(i) != rhsRow.nameAt(i) ||
            !sameLogicalType(
                dereferenceType(lhsRow.childAt(i)),
                dereferenceType(rhsRow.childAt(i)))) {
          return false;
        }
      }
      return true;
    }
    case Kind::FlatMap: {
      const auto& lhsMap = lhs.asFlatMap();
      const auto& rhsMap = rhs.asFlatMap();
      const auto lhsChildrenCount = lhsMap.childrenCount();
      const auto rhsChildrenCount = rhsMap.childrenCount();
      if (lhsChildrenCount == 0 || rhsChildrenCount == 0) {
        return lhsChildrenCount == rhsChildrenCount &&
            lhsMap.keyScalarKind() == rhsMap.keyScalarKind();
      }
      const auto& lhsValueType = dereferenceType(lhsMap.childAt(0));
      for (size_t i = 1; i < lhsChildrenCount; ++i) {
        if (!sameLogicalType(
                lhsValueType, dereferenceType(lhsMap.childAt(i)))) {
          return false;
        }
      }
      const auto& rhsValueType = dereferenceType(rhsMap.childAt(0));
      for (size_t i = 1; i < rhsChildrenCount; ++i) {
        if (!sameLogicalType(
                rhsValueType, dereferenceType(rhsMap.childAt(i)))) {
          return false;
        }
      }
      return lhsMap.keyScalarKind() == rhsMap.keyScalarKind() &&
          sameLogicalType(lhsValueType, rhsValueType);
    }
    case Kind::HybridFlatMap:
      return false;
  }
  NIMBLE_UNREACHABLE("Unknown type kind: {}.", lhs.kind());
}

} // namespace detail

} // namespace facebook::nimble

template <>
struct fmt::formatter<facebook::nimble::HybridFlatMap::KeyPresenceMode>
    : fmt::formatter<std::string> {
  auto format(
      facebook::nimble::HybridFlatMap::KeyPresenceMode mode,
      format_context& ctx) const {
    return fmt::formatter<std::string>::format(
        facebook::nimble::toString(mode), ctx);
  }
};
