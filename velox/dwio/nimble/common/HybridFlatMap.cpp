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
#include <array>
#include <iterator>
#include <numeric>
#include <span>
#include <type_traits>

#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>

#include "velox/buffer/Buffer.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/SchemaGenerated.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

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

// Returns whether key `lhs` sorts before key `rhs` in a group of keys of
// `keyKind`: by value for canonical integer keys, so "9" before "10", and as
// strings otherwise.
bool groupKeyPrecedes(
    ScalarKind keyKind,
    std::string_view lhs,
    std::string_view rhs) {
  if (keyKind == ScalarKind::String || keyKind == ScalarKind::Binary ||
      lhs.empty() || rhs.empty()) {
    return lhs < rhs;
  }
  // Canonical integers have no leading zeros, so the longer of two
  // non-negative keys is larger, and the longer of two negative keys smaller.
  const bool lhsNegative = lhs.front() == '-';
  const bool rhsNegative = rhs.front() == '-';
  if (lhsNegative != rhsNegative) {
    return lhsNegative;
  }
  if (lhs.size() != rhs.size()) {
    return lhsNegative ? lhs.size() > rhs.size() : lhs.size() < rhs.size();
  }
  return lhsNegative ? rhs < lhs : lhs < rhs;
}

// Sorts `keys`, the keys of one group of `keyKind`, in key-presence bitmap
// order (groupKeyPrecedes()). Grouped keys usually arrive sorted, so a sorted
// group costs one comparison per key.
void sortGroupKeys(ScalarKind keyKind, std::vector<std::string>& keys) {
  const auto less = [keyKind](const std::string& lhs, const std::string& rhs) {
    return groupKeyPrecedes(keyKind, lhs, rhs);
  };
  if (!std::is_sorted(keys.begin(), keys.end(), less)) {
    std::sort(keys.begin(), keys.end(), less);
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
    sortGroupKeys(keyKind, group.groupKeys);
  }
  groups_ = std::make_shared<const std::vector<Group>>(std::move(groups));
  groupedKeyIndex_ =
      std::make_shared<const folly::F14FastMap<std::string_view, size_t>>(
          buildIndexGroupsByKey(*groups_));
}

const std::vector<HybridFlatMap::Group>& HybridFlatMap::groups() const {
  return *groups_;
}

std::optional<size_t> HybridFlatMap::findGroup(std::string_view key) const {
  const auto it = groupedKeyIndex_->find(key);
  if (it == groupedKeyIndex_->end()) {
    return std::nullopt;
  }
  return it->second;
}

size_t HybridFlatMap::numGroupedKeys() const {
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

namespace {

// Calls `fn` with a value of the type that keys of `keyKind`, a supported key
// kind (HybridFlatMap::supportedKeyKind()), are encoded as, and returns its
// result: the integer type of integer keys, and std::string_view for String
// keys.
template <typename Fn>
auto withKeyType(ScalarKind keyKind, Fn&& fn) {
  switch (keyKind) {
    case ScalarKind::Int8:
      return fn(int8_t{});
    case ScalarKind::Int16:
      return fn(int16_t{});
    case ScalarKind::Int32:
      return fn(int32_t{});
    case ScalarKind::Int64:
      return fn(int64_t{});
    case ScalarKind::String:
      return fn(std::string_view{});
    default:
      NIMBLE_UNREACHABLE(
          "Hybrid FlatMap key kind is unsupported: {}.", keyKind);
  }
}

// The most bytes the String keys of one encoded_group_keys stream may hold per
// byte of the stream. A String stream can hold keys far longer than itself,
// e.g. a Constant key repeated, or Prefix keys that each extend the one before
// with no restart in between, so readers stop decoding a stream once its keys
// pass the cap, as they bound its key count. Writers check the same cap. It
// leaves room for Prefix on keys with long shared prefixes.
constexpr uint64_t kMaxStringKeyBytesPerStreamByte{64};

// Returns the options of every encoded_group_keys stream, whose prefix records
// its row count as a varint. Readers decode the stream with the same options,
// since the stream does not record which row count format it uses.
Encoding::Options encodingOptions() {
  Encoding::Options options;
  options.useVarintRowCount = true;
  return options;
}

// Encodes `keys`, keys of type T (withKeyType()), as one Nimble stream of T in
// `buffer` with `layout`, which must name every nested encoding: integer keys
// parsed from their canonical strings, and String keys as they are.
// Compression stays off, so that readers without Meta-internal codecs decode
// the schema too. Throws when `layout` cannot encode keys of T, e.g. Prefix for
// integer keys, EliasFano for String keys, a missing nested encoding, or
// Constant for varying keys, and the user error of
// HybridFlatMap::canonicalKey() for an integer key it would reject.
template <typename T>
std::string_view encodeKeys(
    const std::vector<std::string>& keys,
    const EncodingLayout& layout,
    Buffer& buffer) {
  std::vector<T> values;
  values.reserve(keys.size());
  for (const auto& key : keys) {
    if constexpr (std::is_same_v<T, std::string_view>) {
      values.emplace_back(key);
    } else {
      values.push_back(HybridFlatMap::parseCanonicalIntegerKey<T>(key));
    }
  }
  auto policy = std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
      layout,
      /*compressionOptions=*/std::nullopt,
      [](DataType dataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
        NIMBLE_USER_FAIL(
            "Hybrid FlatMap group keys layout is missing a nested encoding: "
            "{}.",
            dataType);
      });
  return EncodingFactory::encode<T>(
      std::move(policy), values, buffer, encodingOptions());
}

// Returns the encoded_group_keys stream of each of `groups`, whose keys are of
// type T (withKeyType()). A group without keys, a KeyList Default or a bitmap
// Default before its first key, has an empty stream. A group with keys,
// including a bitmap Default that lists the keys found while writing, has the
// stream a schema recorded for it, e.g. for a projection, or else its keys
// encoded with `groupKeysLayout` in `buffer`, which holds the streams while the
// caller uses them. `buffer` is set exactly when `groupKeysLayout` is. Throws
// for an explicit group without keys, for a group with neither a recorded
// stream nor a layout, and for keys the layout cannot encode, e.g. a bitmap
// Default's first-seen keys out of order for EliasFano, or encodes in under one
// bit per key, or String keys in under 1/kMaxStringKeyBytesPerStreamByte of
// their bytes, which readers reject.
template <typename T>
std::vector<std::string_view> encodeGroupKeys(
    const std::vector<HybridFlatMap::Group>& groups,
    const EncodingLayout* groupKeysLayout,
    Buffer* buffer) {
  std::vector<std::string_view> streams;
  streams.reserve(groups.size());
  for (const auto& group : groups) {
    // Only a Default group has no keys: a KeyList Default, or a bitmap Default
    // before its first key.
    if (group.groupKeys.empty()) {
      NIMBLE_CHECK_EQ(
          group.groupId,
          HybridFlatMap::kDefaultGroupId,
          "Hybrid FlatMap explicit group {} has no keys.",
          group.groupId);
      NIMBLE_CHECK_EQ(
          group.encodedKeys.size(),
          0,
          "Hybrid FlatMap group without keys has encoded keys.");
      streams.emplace_back();
      continue;
    }
    // The group comes from a schema that recorded its stream, e.g. a group a
    // projection keeps, so the stream is copied instead of encoded again.
    if (!group.encodedKeys.empty()) {
      streams.emplace_back(group.encodedKeys);
      continue;
    }
    NIMBLE_CHECK_NOT_NULL(
        groupKeysLayout,
        "Hybrid FlatMap group {} has neither a recorded stream nor a layout to "
        "encode its keys.",
        group.groupId);
    const auto stream =
        encodeKeys<T>(group.groupKeys, *groupKeysLayout, *buffer);

    // Readers bound the keys a stream holds by its size before decoding it.
    NIMBLE_USER_CHECK_LE(
        group.groupKeys.size(),
        uint64_t{8} * stream.size(),
        "Hybrid FlatMap group keys layout encodes the keys of group {} in "
        "under one bit per key, which readers reject.",
        group.groupId);
    if constexpr (std::is_same_v<T, std::string_view>) {
      const auto keyBytes = std::accumulate(
          group.groupKeys.begin(),
          group.groupKeys.end(),
          uint64_t{0},
          [](uint64_t bytes, const std::string& key) {
            return bytes + key.size();
          });
      NIMBLE_USER_CHECK_LE(
          keyBytes,
          kMaxStringKeyBytesPerStreamByte * stream.size(),
          "Hybrid FlatMap group keys layout encodes the String keys of group {} "
          "in under 1/{} of their bytes, which readers reject.",
          group.groupId,
          kMaxStringKeyBytesPerStreamByte);
    }
    NIMBLE_CHECK_LE(
        stream.size(),
        std::numeric_limits<uint32_t>::max(),
        "Hybrid FlatMap encoded group keys are too large.");
    streams.push_back(stream);
  }
  return streams;
}

// Writes the HybridFlatMap table of `groups` and
// `defaultGroupKeyPresenceMode`: the keys as `encodedGroupKeys`, one stream
// per group, when not null, and as group_keys strings otherwise.
std::string writeMetadata(
    const std::vector<HybridFlatMap::Group>& groups,
    HybridFlatMap::KeyPresenceMode defaultGroupKeyPresenceMode,
    const std::vector<std::string_view>* encodedGroupKeys) {
  flatbuffers::FlatBufferBuilder builder;
  // Writes the Default key presence mode even when it is Bitmap, since the
  // schema constructor requires it.
  builder.ForceDefaults(true);
  std::vector<uint32_t> groupIds;
  std::vector<uint32_t> groupKeyCounts;
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
  }
  flatbuffers::Offset<
      flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>>
      groupKeysOffset;
  flatbuffers::Offset<flatbuffers::Vector<uint8_t>> encodedGroupKeysOffset;
  flatbuffers::Offset<flatbuffers::Vector<uint32_t>> encodedGroupKeySizesOffset;
  if (encodedGroupKeys == nullptr) {
    std::vector<flatbuffers::Offset<flatbuffers::String>> groupKeys;
    groupKeys.reserve(totalNumKeys(groups));
    for (const auto& group : groups) {
      for (const auto& key : group.groupKeys) {
        groupKeys.push_back(builder.CreateString(key));
      }
    }
    groupKeysOffset = builder.CreateVector(groupKeys);
  } else {
    std::vector<uint32_t> sizes;
    sizes.reserve(encodedGroupKeys->size());
    for (const auto stream : *encodedGroupKeys) {
      sizes.push_back(static_cast<uint32_t>(stream.size()));
    }
    std::string bytes;
    bytes.reserve(std::accumulate(sizes.begin(), sizes.end(), size_t{0}));
    for (const auto stream : *encodedGroupKeys) {
      bytes.append(stream);
    }
    encodedGroupKeysOffset = builder.CreateVector(
        reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
    encodedGroupKeySizesOffset = builder.CreateVector(sizes);
  }
  const auto groupIdsOffset = builder.CreateVector(groupIds);
  const auto groupKeyCountsOffset = builder.CreateVector(groupKeyCounts);
  builder.Finish(
      serialization::CreateHybridFlatMap(
          builder,
          groupIdsOffset,
          groupKeyCountsOffset,
          groupKeysOffset,
          static_cast<serialization::HybridFlatMapKeyPresenceMode>(
              defaultGroupKeyPresenceMode),
          encodedGroupKeysOffset,
          encodedGroupKeySizesOffset));
  return {
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      builder.GetSize()};
}

} // namespace

std::string HybridFlatMap::serialize(
    ScalarKind keyKind,
    const std::vector<Group>& groups,
    KeyPresenceMode defaultGroupKeyPresenceMode,
    const EncodingLayout* groupKeysLayout,
    velox::memory::MemoryPool* pool) {
  NIMBLE_USER_CHECK_NOT_NULL(
      pool, "Hybrid FlatMap serialization requires a memory pool.");
  // A node whose groups have no keys has nothing to encode, e.g. a projection
  // of a key the node lacks, which keeps only its empty Default group. Without
  // a layout, the keys stay strings unless a schema recorded their streams,
  // which a projection copies.
  const auto hasKeys = [](const Group& group) {
    return !group.groupKeys.empty();
  };
  const auto isEncoded = [](const Group& group) {
    return !group.encodedKeys.empty();
  };
  if (std::none_of(groups.begin(), groups.end(), hasKeys) ||
      (groupKeysLayout == nullptr &&
       std::none_of(groups.begin(), groups.end(), isEncoded))) {
    return writeMetadata(
        groups, defaultGroupKeyPresenceMode, /*encodedGroupKeys=*/nullptr);
  }
  // Holds the streams that encodeGroupKeys() encodes until writeMetadata()
  // copies them. Created only with a layout, since a Buffer allocates its first
  // 1 MiB chunk up front.
  std::optional<Buffer> buffer;
  if (groupKeysLayout != nullptr) {
    buffer.emplace(*pool);
  }
  const auto streams = withKeyType(keyKind, [&](auto type) {
    return encodeGroupKeys<decltype(type)>(
        groups, groupKeysLayout, buffer.has_value() ? &*buffer : nullptr);
  });
  return writeMetadata(groups, defaultGroupKeyPresenceMode, &streams);
}

namespace {

// Returns `keyKind`, the key kind of a written schema, once it is checked.
// Initializes the const HybridFlatMap::keyKind_ in the schema constructor, so
// the key kind is checked before defaultGroupKeyPresenceMode_'s initializer
// reads any byte.
ScalarKind checkRecordedKeyKind(ScalarKind keyKind) {
  NIMBLE_CHECK(
      HybridFlatMap::supportedKeyKind(keyKind),
      "Hybrid FlatMap key kind is unsupported: {}.",
      keyKind);
  return keyKind;
}

// Verifies `serialized` as a HybridFlatMap FlatBuffer table and returns the
// Default key presence mode it records, which every written schema has.
// Initializes the const HybridFlatMap::defaultGroupKeyPresenceMode_ in the
// schema constructor, whose body reads the rest of the table only once this
// has verified it.
HybridFlatMap::KeyPresenceMode readDefaultGroupKeyPresenceMode(
    std::string_view serialized) {
  flatbuffers::Verifier verifier{
      reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size()};
  NIMBLE_CHECK(
      verifier.VerifyBuffer<serialization::HybridFlatMap>(),
      "Hybrid FlatMap metadata attribute is malformed.");
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
  return static_cast<HybridFlatMap::KeyPresenceMode>(
      defaultGroupKeyPresenceMode);
}

// Decodes `encoded`, the encoded_group_keys stream of `numKeys` keys of type T
// (withKeyType()), into the keys' canonical strings, as the writer held them,
// allocating from `pool`.
template <typename T>
std::vector<std::string> decodeKeys(
    std::string_view encoded,
    uint64_t numKeys,
    velox::memory::MemoryPool& pool) {
  // The Encoding constructor reads the varint row count unchecked, so the
  // prefix is checked first, and encodings trust their payload as value streams
  // do. Writers keep at least one bit per key, which bounds the keys a stream
  // can hold before any allocation.
  NIMBLE_CHECK_FILE_GT(
      encoded.size(),
      EncodingPrefix::kRowCountOffset,
      "Hybrid FlatMap encoded group keys are truncated.");
  auto payload = encoded;
  EncodingPrefix::consume(payload, encodingOptions().useVarintRowCount);
  NIMBLE_CHECK_FILE_LE(
      numKeys,
      uint64_t{8} * encoded.size(),
      "Hybrid FlatMap encoded group keys are too short for the group key "
      "counts.");
  NIMBLE_CHECK_FILE_EQ(
      EncodingPrefix::dataType(encoded),
      TypeTraits<T>::dataType,
      "Hybrid FlatMap encoded group keys must be of the key type.");
  // Holds the strings that a String stream decodes into until the keys copy
  // them.
  std::vector<velox::BufferPtr> stringBuffers;
  const auto encoding = EncodingFactory{encodingOptions()}.create(
      pool, encoded, [&](uint32_t totalLength) -> void* {
        return stringBuffers
            .emplace_back(
                velox::AlignedBuffer::allocate<char>(totalLength, &pool))
            ->asMutable<void>();
      });
  NIMBLE_CHECK_FILE_EQ(
      encoding->rowCount(),
      numKeys,
      "Hybrid FlatMap encoded group keys must match the group key counts.");
  std::vector<std::string> keys;
  keys.reserve(numKeys);
  if constexpr (std::is_same_v<T, std::string_view>) {
    // A String stream can decode to far more bytes than it holds, e.g. a
    // Constant key repeated, or Prefix keys that each extend the one before.
    // Decodes the keys in batches and caps their bytes as they grow, so a
    // stream allocates at most one batch past the cap.
    constexpr uint64_t kBatchSize{64};
    const uint64_t maxKeyBytes =
        kMaxStringKeyBytesPerStreamByte * encoded.size();
    uint64_t keyBytes{0};
    std::vector<std::string_view> batch;
    for (uint64_t begin = 0; begin < numKeys; begin += kBatchSize) {
      batch.resize(std::min(kBatchSize, numKeys - begin));
      encoding->materialize(static_cast<uint32_t>(batch.size()), batch.data());
      for (const auto key : batch) {
        keyBytes += key.size();
        NIMBLE_CHECK_FILE_LE(
            keyBytes,
            maxKeyBytes,
            "Hybrid FlatMap encoded group keys decode to too many bytes.");
        keys.emplace_back(key);
      }
    }
  } else {
    std::vector<T> values(numKeys);
    encoding->materialize(encoding->rowCount(), values.data());
    // The buffer std::to_chars writes each integer key into. It fits the
    // longest key of T: a minus sign and digits10 + 1 decimal digits, e.g. the
    // 20 characters of int64_t's "-9223372036854775808".
    std::array<char, std::numeric_limits<T>::digits10 + 2> digits{};
    for (const auto value : values) {
      const auto result =
          std::to_chars(digits.data(), digits.data() + digits.size(), value);
      keys.emplace_back(digits.data(), result.ptr);
    }
  }
  return keys;
}

// Validates encoded_group_keys and its sizes, and returns the stream of each of
// the `numGroupIds` groups, or std::nullopt when `metadata` holds group_keys
// strings instead.
std::optional<std::vector<std::string_view>> getEncodedGroupKeys(
    const serialization::HybridFlatMap& metadata,
    flatbuffers::uoffset_t numGroupIds) {
  const auto* encodedGroupKeys = metadata.encoded_group_keys();
  const auto* encodedGroupKeySizes = metadata.encoded_group_key_sizes();
  if (encodedGroupKeys == nullptr) {
    NIMBLE_CHECK_FILE_NULL(
        encodedGroupKeySizes,
        "Hybrid FlatMap encoded group key sizes require encoded group keys.");
    return std::nullopt;
  }
  NIMBLE_CHECK_FILE_NULL(
      metadata.group_keys(),
      "Hybrid FlatMap metadata must not hold both group keys and encoded "
      "group keys.");
  NIMBLE_CHECK_FILE_NOT_NULL(
      encodedGroupKeySizes,
      "Hybrid FlatMap encoded group keys must list the size of each group's "
      "keys.");
  NIMBLE_CHECK_FILE_EQ(
      encodedGroupKeySizes->size(),
      numGroupIds,
      "Hybrid FlatMap encoded group key sizes must match the group count.");
  std::vector<std::string_view> encodedGroups;
  encodedGroups.reserve(numGroupIds);
  flatbuffers::uoffset_t encodedGroupBegin{0};
  for (const auto encodedGroupSize : *encodedGroupKeySizes) {
    NIMBLE_CHECK_FILE_LE(
        encodedGroupSize,
        encodedGroupKeys->size() - encodedGroupBegin,
        "Hybrid FlatMap encoded group keys are truncated.");
    encodedGroups.emplace_back(
        reinterpret_cast<const char*>(encodedGroupKeys->data()) +
            encodedGroupBegin,
        encodedGroupSize);
    encodedGroupBegin += encodedGroupSize;
  }
  NIMBLE_CHECK_FILE_EQ(
      encodedGroupBegin,
      encodedGroupKeys->size(),
      "Hybrid FlatMap encoded group key sizes must add up to the encoded "
      "group keys.");
  return encodedGroups;
}

} // namespace

HybridFlatMap::HybridFlatMap(
    ScalarKind keyKind,
    std::string_view serialized,
    velox::memory::MemoryPool* pool)
    : keyKind_{checkRecordedKeyKind(keyKind)},
      defaultGroupKeyPresenceMode_{
          readDefaultGroupKeyPresenceMode(serialized)} {
  NIMBLE_USER_CHECK_NOT_NULL(
      pool, "Hybrid FlatMap schema requires a memory pool.");
  // readDefaultGroupKeyPresenceMode() has verified the buffer.
  const auto* serializedMetadata =
      flatbuffers::GetRoot<serialization::HybridFlatMap>(serialized.data());
  const auto* groupIds = serializedMetadata->group_ids();
  const auto* groupKeyCounts = serializedMetadata->group_key_counts();
  const auto* groupKeys = serializedMetadata->group_keys();
  NIMBLE_CHECK_NOT_NULL(groupIds, "Hybrid FlatMap group IDs are missing.");
  NIMBLE_CHECK_NOT_NULL(
      groupKeyCounts, "Hybrid FlatMap group key counts are missing.");
  const auto numGroupIds = groupIds->size();
  NIMBLE_CHECK_EQ(
      numGroupIds,
      groupKeyCounts->size(),
      "Hybrid FlatMap group IDs and key counts must have the same size");
  const auto encodedGroups =
      getEncodedGroupKeys(*serializedMetadata, numGroupIds);
  // FlatBuffers writers may omit an empty vector. Omitted key vectors hold
  // zero keys, so the key count checks below accept them only when no group
  // has keys, as in Default-only metadata.
  size_t numKeys{groupKeys == nullptr ? 0 : groupKeys->size()};
  // Decodes the keys of each group, picked once for the key kind, which
  // checkRecordedKeyKind() has checked.
  std::vector<std::string> (*decodeGroupKeys)(
      std::string_view, uint64_t, velox::memory::MemoryPool&){nullptr};
  if (encodedGroups.has_value()) {
    numKeys = std::accumulate(
        groupKeyCounts->begin(), groupKeyCounts->end(), size_t{0});
    decodeGroupKeys = withKeyType(
        keyKind_, [](auto type) { return &decodeKeys<decltype(type)>; });
  }
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
    if (encodedGroups.has_value()) {
      const auto encodedGroup = (*encodedGroups)[i];
      groupKeyIndex += numGroupKeys;
      if (numGroupKeys == 0) {
        NIMBLE_CHECK_FILE(
            encodedGroup.empty(),
            "Hybrid FlatMap group without keys must have no encoded keys.");
      } else {
        group.groupKeys = decodeGroupKeys(encodedGroup, numGroupKeys, *pool);
        group.encodedKeys = std::string(encodedGroup);
      }
    } else {
      for (uint32_t j = 0; j < numGroupKeys; ++j) {
        const auto* key = groupKeys->Get(groupKeyIndex++);
        NIMBLE_CHECK_NOT_NULL(key, "Hybrid FlatMap group key cannot be null.");
        group.groupKeys.push_back(key->str());
      }
    }
    detail::checkHybridFlatMapDefaultKeys(
        defaultGroupKeyPresenceMode_, group.groupId, group.groupKeys);
  }
  NIMBLE_CHECK_EQ(
      groupKeyIndex,
      numKeys,
      "Hybrid FlatMap group key counts must match group keys size");

  // The index views the keys, so it is built once groups_ holds them.
  groups_ = std::make_shared<const std::vector<Group>>(std::move(groups));
  folly::F14FastMap<std::string_view, size_t> groupIndices;
  detail::validateHybridFlatMapGroups(
      groups_->size(),
      /*hasDefault=*/false,
      [this](size_t index) { return (*groups_)[index].groupId; },
      [this](size_t index) -> const auto& {
        return (*groups_)[index].groupKeys;
      },
      groupIndices);
  groupedKeyIndex_ =
      std::make_shared<const folly::F14FastMap<std::string_view, size_t>>(
          std::move(groupIndices));
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
  if (!HybridFlatMap::isDefaultGroup(groupId)) {
    NIMBLE_CHECK_GT(
        groupKeys.size(),
        0,
        "Hybrid FlatMap group must contain at least one key: {}.",
        groupId);
  }
  for (const auto& key : groupKeys) {
    NIMBLE_CHECK_GT(key.size(), 0, "Hybrid FlatMap key cannot be empty.");
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
