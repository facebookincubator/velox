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

#include <cstdint>
#include <cstring>
#include <optional>
#include <string_view>

#include <folly/container/F14Set.h>
#include <folly/container/detail/F14Defaults.h>

namespace facebook::velox::detail {

/// A lookup structure that maps string names to uint32_t indices.
/// This is written to decrease memory footprint.
/// In general it can be replaced with Map<string_view, size_t>.
// TODO: Consider using absl::flat_hash_set instead.
class NameToIndex {
 public:
  NameToIndex() = default;

  /// Reserves space for the specified number of elements.
  void reserve(size_t size) {
    set_.reserve(size);
  }

  /// Inserts a name with its corresponding index.
  /// @param name The string to insert.
  /// @param index The index to associate with the name.
  void insert(std::string_view name, uint32_t index) {
    set_.emplace(
        NameIndex{name.data(), static_cast<uint32_t>(name.size()), index});
  }

  /// Checks if a name exists in the lookup.
  /// @param name The name to check.
  /// @return true if the name exists, false otherwise.
  bool contains(std::string_view name) const {
    return set_.contains(
        NameIndex{name.data(), static_cast<uint32_t>(name.size()), 0});
  }

  /// Finds the index associated with a name.
  /// @param name The name to find.
  /// @return The index if found, std::nullopt otherwise.
  std::optional<uint32_t> find(std::string_view name) const {
    auto it = set_.find(
        NameIndex{name.data(), static_cast<uint32_t>(name.size()), 0});
    if (it != set_.end()) {
      return it->index;
    }
    return std::nullopt;
  }

  /// Returns the number of elements in the lookup.
  /// @return The number of name-index pairs stored.
  size_t size() const {
    return set_.size();
  }

 private:
  struct NameIndex {
    const char* data = nullptr;
    uint32_t size = 0;
    uint32_t index = 0;

    bool operator==(const NameIndex& other) const {
      return size == other.size && std::memcmp(data, other.data, size) == 0;
    }
  };

  struct NameIndexHasher {
    size_t operator()(const NameIndex& nameIndex) const {
      return folly::f14::DefaultHasher<std::string_view>{}(
          std::string_view{nameIndex.data, nameIndex.size});
    }
  };

  folly::F14ValueSet<NameIndex, NameIndexHasher> set_;
};

} // namespace facebook::velox::detail
