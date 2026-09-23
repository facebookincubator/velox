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
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "folly/container/F14Set.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/velox/SchemaTypes.h"

namespace facebook::nimble {

class HybridFlatMap {
 public:
  /// Describes one group in logical schema order.
  struct Group {
    uint32_t groupId{0};
    std::vector<std::string> groupKeys;

    bool operator==(const Group&) const = default;
  };

  /// Reserved identifier for the Default group.
  static constexpr uint32_t kDefaultGroupId =
      std::numeric_limits<uint32_t>::max();

  /// Reserved schema attribute containing serialized group metadata.
  static constexpr std::string_view kAttributeName{"hybridFlatMap"};

  /// Logical groups in schema order.
  std::vector<Group> groups;

  bool operator==(const HybridFlatMap&) const = default;

  /// Returns whether `kind` is supported for Hybrid FlatMap keys.
  static bool supportedKeyKind(ScalarKind kind);

  /// Serializes the groups into the flattened HybridFlatMap FlatBuffer table.
  /// `group_ids` and `group_key_counts` are parallel per-group vectors, and
  /// `group_keys` concatenates the keys of every group in group order, so
  /// group `i` owns the `group_key_counts[i]` keys that follow the keys of the
  /// preceding groups.
  std::string serialize() const;

  /// Deserializes and validates groups produced by `serialize()`.
  static HybridFlatMap deserialize(std::string_view serialized);

  /// Deserializes the reserved metadata attribute, leaving `attributes`
  /// unchanged.
  static HybridFlatMap getAttribute(
      const std::vector<std::pair<std::string, std::string>>& attributes);

  /// Deserializes the reserved metadata attribute and removes it from
  /// `attributes`. Leaves `attributes` unchanged when deserialization fails.
  static HybridFlatMap extractAttribute(
      std::vector<std::pair<std::string, std::string>>& attributes);

  /// Appends the reserved metadata attribute to `attributes`, which must not
  /// already contain it. The attribute is owned by the schema, so a caller
  /// supplying its own copy is setting reserved state it does not control.
  void setAttribute(
      std::vector<std::pair<std::string, std::string>>& attributes) const;

  /// Validates full-plan group IDs and keys. Accessors decouple validation
  /// from writer- and reader-side group representations.
  template <typename GroupIdAt, typename GroupKeysAt>
  static void
  validate(size_t groupCount, GroupIdAt groupIdAt, GroupKeysAt groupKeysAt) {
    NIMBLE_CHECK_GT(
        groupCount, 1, "Hybrid FlatMap requires at least two groups.");
    folly::F14FastSet<uint32_t> groupIds;
    folly::F14FastSet<std::string> keys;
    bool hasDefaultGroup{false};
    for (size_t i = 0; i < groupCount; ++i) {
      const auto groupId = groupIdAt(i);
      const auto& groupKeys = groupKeysAt(i);
      const bool uniqueGroupId = groupIds.insert(groupId).second;
      NIMBLE_CHECK(
          uniqueGroupId, "Duplicate Hybrid FlatMap group ID: {}.", groupId);
      if (groupId == kDefaultGroupId) {
        hasDefaultGroup = true;
        NIMBLE_CHECK(
            groupKeys.empty(),
            "Hybrid FlatMap Default group cannot contain group keys.");
      } else {
        NIMBLE_CHECK(
            !groupKeys.empty(),
            "Hybrid FlatMap group must contain at least one key: {}.",
            groupId);
      }
      NIMBLE_CHECK(
          std::is_sorted(groupKeys.begin(), groupKeys.end()),
          "Hybrid FlatMap group keys must be sorted: {}.",
          groupId);
      for (const auto& key : groupKeys) {
        NIMBLE_CHECK(!key.empty(), "Hybrid FlatMap key cannot be empty.");
        const bool uniqueKey = keys.insert(key).second;
        NIMBLE_CHECK(uniqueKey, "Duplicate Hybrid FlatMap key: '{}'.", key);
      }
    }
    NIMBLE_CHECK(
        hasDefaultGroup, "Hybrid FlatMap must have exactly one Default group.");
  }
};

namespace detail {

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
