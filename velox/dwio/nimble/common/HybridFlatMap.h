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
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <folly/Conv.h>
#include "folly/container/F14Map.h"
#include "folly/container/F14Set.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/SchemaTypes.h"
#include "velox/dwio/nimble/common/Types.h"

namespace facebook::nimble {

namespace detail {

struct GroupLookupIndex;

} // namespace detail

/// Groups the keys of a MAP column for the Hybrid FlatMap writer, or holds the
/// groups a written schema records (fromSchema()). A HybridFlatMap built with
/// every group at once, or bound to a column, holds its groups and a lookup of
/// each grouped key's group in immutable state that its copies share, so a
/// copy costs one reference count and threads may read copies concurrently.
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
    /// Keys in key-presence bitmap order. Configured groups are sorted;
    /// Default preserves first-seen order, and lists no keys with
    /// KeyPresenceMode::kKeyList.
    std::vector<std::string> groupKeys;

    bool operator==(const Group&) const = default;
  };

  /// Reserved identifier for the Default group.
  static constexpr uint32_t kDefaultGroupId =
      std::numeric_limits<uint32_t>::max();

  /// Returns whether `groupId` identifies the reserved Default group.
  static constexpr bool isDefaultGroup(uint32_t groupId) {
    return groupId == kDefaultGroupId;
  }

  /// An empty writer configuration: only the Default group, without keys.
  /// addGroup() adds the explicit groups. Without a key kind, as when the
  /// caller does not know the map's key type yet, the keys are checked for one
  /// when a writer binds the configuration to its column (see the binding
  /// constructor below).
  HybridFlatMap();

  /// As above, for keys of `keyKind`, so addGroup() checks each key for it
  /// on the spot.
  explicit HybridFlatMap(ScalarKind keyKind);

  /// A writer configuration for keys of `keyKind`, checked with user errors
  /// and normalized: every key canonical (canonicalKey()), group IDs
  /// ascending, no key repeated within or across groups, every explicit
  /// group non-empty, and exactly one Default group, without keys. groups()
  /// then holds each group's keys sorted, the schema and key-presence bitmap
  /// order. Builds the lookup of findGroup(), which copies share.
  ///
  /// Example: for Int64 keys, groups {0: ["9", "10"], Default: []} hold
  /// {0: ["10", "9"], Default: []}; {0: ["07"], Default: []} and
  /// {1: ["1"], 0: ["2"], Default: []} are rejected.
  HybridFlatMap(ScalarKind keyKind, std::vector<Group> groups);

  /// `configuration` bound to a column whose keys are of `keyKind`, as a
  /// writer does when it meets the column. A configuration built without a
  /// key kind is checked for `keyKind` as the constructor above checks one; a
  /// configuration built for another key kind is rejected. Keeps the
  /// configuration's defaultGroupKeyPresenceMode().
  ///
  /// Example: a configuration with group {0: ["7", "300"]} and no key kind
  /// binds to an Int16 column, and is rejected for an Int8 column, where
  /// "300" does not parse.
  HybridFlatMap(ScalarKind keyKind, const HybridFlatMap& configuration);

  /// The groups of a written schema, as recorded: the Default group holds
  /// the keys found while writing, in first-seen order, and a projection may
  /// leave groups out. Not checked as a writer configuration; schema readers
  /// and builders apply their own checks. Its key kind is the schema type's,
  /// and `defaultGroupKeyPresenceMode` is the mode the schema records.
  static HybridFlatMap fromSchema(
      std::vector<Group> groups,
      KeyPresenceMode defaultGroupKeyPresenceMode);

  /// The key kind a writer configuration was built for; empty for a
  /// configuration built without one, until a writer binds it, and for schema
  /// metadata (fromSchema()), whose key kind is the schema type's.
  std::optional<ScalarKind> keyKind() const {
    return keyKind_;
  }

  /// Logical groups in schema order.
  const std::vector<Group>& groups() const;

  /// Returns the index in groups() of the explicit group that holds `key`, or
  /// std::nullopt when no explicit group holds it. Needs the lookup
  /// (hasGroupLookupIndex()); fails without it.
  std::optional<size_t> findGroup(std::string_view key) const;

  /// Returns the number of keys in explicit groups. Needs the lookup, as
  /// findGroup() does.
  size_t numGroupedKeys() const;

  /// Returns whether this HybridFlatMap holds the lookup of findGroup(), which
  /// the constructor taking every group builds and copies share. A writer for
  /// a column of keyKind() uses such a HybridFlatMap as is instead of binding
  /// it again.
  bool hasGroupLookupIndex() const {
    return groupLookupIndex_ != nullptr;
  }

  /// Moves the groups out, for a reader that builds its own type from them.
  /// Returns a copy of groups that copies of this HybridFlatMap share.
  std::vector<Group> releaseGroups() &&;

  /// Adds explicit group `groupId` holding `keys`, checked with user errors
  /// on the spot: `groupId` above every explicit group added before and below
  /// the Default group's, `keys` non-empty, no key empty or already in a
  /// group, and, with a key kind, every key canonical for it. The group goes
  /// before the Default group with its keys sorted, the key-presence bitmap
  /// order. Rejects a HybridFlatMap with the lookup of findGroup(), whose
  /// copies share its groups. A call that fails a check changes nothing.
  ///
  /// Example: on HybridFlatMap{ScalarKind::Int64}, addGroup(0, {"9", "10"})
  /// adds {0: ["10", "9"]}; a later addGroup(0, {"1"}) or addGroup(1, {"9"})
  /// is rejected.
  void addGroup(uint32_t groupId, std::vector<std::string> keys);

  /// How batches record their Default keys. A writer configuration uses
  /// KeyPresenceMode::kBitmap unless setDefaultGroupKeyPresenceMode() changes
  /// it; schema metadata holds the mode its schema records.
  KeyPresenceMode defaultGroupKeyPresenceMode() const {
    return defaultGroupKeyPresenceMode_;
  }

  /// Sets defaultGroupKeyPresenceMode() of a writer configuration.
  void setDefaultGroupKeyPresenceMode(KeyPresenceMode mode) {
    defaultGroupKeyPresenceMode_ = mode;
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

  /// Reserved schema attribute containing serialized group metadata.
  static constexpr std::string_view kAttributeName{"hybridFlatMap"};

  /// Compares keyKind(), groups() and defaultGroupKeyPresenceMode(), whether
  /// or not the lookup of findGroup() is built.
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

  /// Serializes the groups into the flattened HybridFlatMap FlatBuffer table.
  /// `group_ids` and `group_key_counts` are parallel per-group vectors, and
  /// `group_keys` concatenates the keys of every group in group order, so
  /// group `i` owns the `group_key_counts[i]` keys that follow the keys of the
  /// preceding groups. The Default key presence mode is always written.
  std::string serialize() const;

  /// Deserializes and validates groups produced by `serialize()`, as schema
  /// metadata (fromSchema()). Also accepts an omitted `group_keys` vector when
  /// no group has keys. Fails when the Default key presence mode is missing.
  static HybridFlatMap deserialize(std::string_view serialized);

  /// Deserializes the reserved metadata attribute, leaving `attributes`
  /// unchanged.
  static HybridFlatMap getAttribute(
      const std::vector<std::pair<std::string, std::string>>& attributes);

  /// Deserializes the reserved metadata attribute and removes it from
  /// `attributes`. Leaves `attributes` unchanged when deserialization fails.
  static HybridFlatMap extractAttribute(
      std::vector<std::pair<std::string, std::string>>& attributes);

  /// Appends the reserved metadata attribute when absent, or replaces its
  /// stale value in place.
  void setAttribute(
      std::vector<std::pair<std::string, std::string>>& attributes) const;

 private:
  // fromSchema(): the groups as given, without a key kind.
  struct AsRecorded {};
  HybridFlatMap(
      AsRecorded,
      std::vector<Group> groups,
      KeyPresenceMode defaultGroupKeyPresenceMode)
      : groups_{std::move(groups)},
        defaultGroupKeyPresenceMode_{defaultGroupKeyPresenceMode} {}

  // Returns a copy of the groups of `configuration`, which the binding
  // constructor checks for `keyKind`. Throws when `configuration` was built
  // for a different key kind, e.g. Int64 for an Int32 column.
  static std::vector<Group> groupsForKeyKind(
      ScalarKind keyKind,
      const HybridFlatMap& configuration);

  // See keyKind(). Fixed at construction: binding to a column builds a new
  // HybridFlatMap.
  const std::optional<ScalarKind> keyKind_;
  // The groups of a HybridFlatMap built by the other constructors and
  // addGroup(), and the groups of schema metadata (fromSchema()). Empty while
  // groupLookupIndex_ holds the groups.
  std::vector<Group> groups_;
  // The groups and the lookup of findGroup(), built by the constructor taking
  // every group and shared by every copy. Null otherwise.
  std::shared_ptr<const detail::GroupLookupIndex> groupLookupIndex_;
  // See defaultGroupKeyPresenceMode().
  KeyPresenceMode defaultGroupKeyPresenceMode_{KeyPresenceMode::kBitmap};
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

namespace detail {

// Checks the rules for one group's own keys, shared by whole-schema validation
// and HybridFlatMapTypeBuilder::addGroup(). Configured keys must be non-empty
// and strictly ascending because lookups binary search them. Default keys keep
// their first-seen order, which is their key-presence bitmap order, so only
// empty keys are rejected here. Callers check that every key is unique across
// all groups.
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
