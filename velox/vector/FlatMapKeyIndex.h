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

#include <optional>
#include <unordered_map>

#include "velox/vector/BaseVector.h"

namespace facebook::velox::detail {

/// Maps the hash of each distinct key of a FlatMapVector to its channel, the
/// position of that key's values. Keying on the hash supports any key type
/// without templating this class, so lookups must compare the actual keys to
/// rule out hash collisions.
class FlatMapKeyIndex {
 public:
  /// Indexes every key in `distinctKeys`, in order.
  explicit FlatMapKeyIndex(const BaseVector& distinctKeys);

  /// Indexes the next channel, whose key hashes to `hash`.
  void add(uint64_t hash);

  /// Returns the first channel under `hash` for which `isMatch(channel)` holds.
  template <typename TIsMatch>
  std::optional<column_index_t> find(uint64_t hash, TIsMatch&& isMatch) const {
    auto range = keyToChannel_.equal_range(hash);
    for (auto it = range.first; it != range.second; ++it) {
      if (isMatch(it->second)) {
        return it->second;
      }
    }
    return std::nullopt;
  }

 private:
  std::unordered_multimap<uint64_t, column_index_t> keyToChannel_;
};

} // namespace facebook::velox::detail
