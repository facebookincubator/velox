/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include <string>
#include <vector>

#include <folly/container/F14Map.h>

#include "velox/common/base/Exceptions.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::functions::sparksql {

/// Equality and hashing of the keys of one map the way Spark's map builder
/// compares them, read from a decoded vector starting at an offset. A top-level
/// floating-point key compares the way Java compares boxed floats and doubles,
/// so -0.0 and 0.0 are two keys and every NaN is one key. Every other key,
/// including a floating-point value nested in an array, map or struct, compares
/// with Velox equality, where nested -0.0 equals 0.0.
class SparkMapKeyEquality {
 public:
  SparkMapKeyEquality() = default;

  /// Compares the keys of a map that starts at the given offset in the decoded
  /// keys, which must outlive this object.
  SparkMapKeyEquality(const DecodedVector& keys, vector_size_t offset)
      : keys_{&keys}, offset_{offset} {}

  /// Returns a hash of a non-null key that agrees with the equality below.
  uint64_t hash(vector_size_t position) const;

  /// Returns whether two non-null keys are the same key.
  bool equals(vector_size_t lhs, vector_size_t rhs) const;

  /// Returns whether a key is null.
  bool isNull(vector_size_t position) const {
    return keys_->isNullAt(offset_ + position);
  }

  /// Returns a key as text for error messages.
  std::string describe(vector_size_t position) const;

 private:
  // Decoded keys the map's keys are read from.
  const DecodedVector* keys_{nullptr};
  // Where this map's keys start in the decoded keys.
  vector_size_t offset_{0};
};

/// The positions of the key and the value a map keeps for one distinct key,
/// counted from the start of the map's keys.
struct KeptPositions {
  /// Position of the occurrence whose key is kept.
  vector_size_t key;
  /// Position of the occurrence whose value is kept.
  vector_size_t value;
};

/// Resolves repeated keys among the keys of one map, comparing them with the
/// given rule. Reusing one object across maps means a call allocates only when
/// a map has more keys than any seen before.
template <typename TRule>
class LastWin {
 public:
  LastWin() : entryByKey_{0, Hash{}, EqualTo{&rule_}} {}

  LastWin(const LastWin&) = delete;
  LastWin& operator=(const LastWin&) = delete;

  /// Resolves the keys of one map, read from a decoded vector starting at the
  /// given offset. Returns the positions to keep for each distinct key, in
  /// order of first occurrence. Checks each key in order and throws a user
  /// error on the first null key or, if repeated keys should throw, the first
  /// key that repeats an earlier one. Otherwise a repeated key keeps the key
  /// and position of its first occurrence and the value of its last occurrence.
  /// The result stays valid until the next call.
  const std::vector<KeptPositions>& resolve(
      const DecodedVector& keys,
      vector_size_t offset,
      vector_size_t numKeys,
      bool throwOnDuplicateKey) {
    rule_ = TRule{keys, offset};
    entries_.clear();
    entries_.reserve(numKeys);
    entryByKey_.clear();
    entryByKey_.reserve(numKeys);

    for (vector_size_t index = 0; index < numKeys; ++index) {
      VELOX_USER_CHECK(!rule_.isNull(index), "Cannot use null as map key!");
      const auto [it, inserted] = entryByKey_.try_emplace(
          Key{index, rule_.hash(index)}, entries_.size());
      if (!inserted) {
        VELOX_USER_CHECK(
            !throwOnDuplicateKey,
            "Duplicate map key was found: {}",
            rule_.describe(index));
        entries_[it->second].value = index;
      } else {
        entries_.push_back({index, index});
      }
    }
    return entries_;
  }

 private:
  // A key's position and its hash, computed once.
  struct Key {
    vector_size_t position;
    uint64_t hash;
  };

  struct Hash {
    size_t operator()(const Key& key) const {
      return key.hash;
    }
  };

  struct EqualTo {
    const TRule* rule;

    bool operator()(const Key& lhs, const Key& rhs) const {
      return rule->equals(lhs.position, rhs.position);
    }
  };

  // Compares the keys of the map being resolved.
  TRule rule_;
  // One entry per distinct key resolved so far.
  std::vector<KeptPositions> entries_;
  // The entry of each distinct key resolved so far.
  folly::F14FastMap<Key, vector_size_t, Hash, EqualTo> entryByKey_;
};

} // namespace facebook::velox::functions::sparksql
