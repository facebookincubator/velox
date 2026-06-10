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

#include <folly/container/F14Map.h>
#include <folly/dynamic.h>

#include "velox/type/Type.h"

namespace facebook::velox {

/// Cache of serialized RowType instances. Useful to reduce the size of
/// serialized expressions and plans. Disabled by default. Not thread safe.
///
/// To enable, call 'serializedTypeCache().enable()'. This enables the cache for
/// the current thread. To disable, call 'serializedTypeCache().disable()'.
/// While enables, type serialization will use the cache and serialize the types
/// using IDs stored in the cache. The caller is responsible for saving
/// serialized types from the cache and using these to hidrate
/// 'deserializedTypeCache()' before deserializing the types.
class SerializedTypeCache {
 public:
  struct Options {
    // Caching applies to RowType's with at least this many fields.
    size_t minRowTypeSize = 10;
  };

  bool isEnabled() const {
    return enabled_;
  }

  const Options& options() const {
    return options_;
  }

  void enable(const Options& options = {.minRowTypeSize = 10}) {
    enabled_ = true;
    options_ = options;
  }

  void disable() {
    enabled_ = false;
  }

  size_t size() const {
    return cache_.size();
  }

  void clear() {
    cache_.clear();
  }

  /// Returns the ID of the type if it is in the cache. Returns std::nullopt if
  /// type is not found in the cache. Cache key is type instance pointer. Hence,
  /// equal but different instances are stored separately.
  std::optional<int32_t> get(const Type& type) const;

  /// Stores the type in the cache. Returns the ID of the type. Reports an error
  /// if type is already present in the cache. IDs are monotonically increasing.
  /// Serialized type may refer to types stored previously in the cache. When
  /// deserializing type cache, make sure to deserialize types in the order of
  /// cache IDs.
  int32_t put(const Type& type, folly::dynamic serialized);

  /// Serialized the types stored in the cache. Use
  /// DeserializedTypeCache::deserialize to deserialize.
  folly::dynamic serialize();

 private:
  bool enabled_{false};
  Options options_;
  folly::F14FastMap<const Type*, std::pair<int32_t, folly::dynamic>> cache_;
};

/// Thread local cache of serialized RowType instances. Used by
/// RowType::serialize.
SerializedTypeCache& serializedTypeCache();

/// Thread local cache of deserialized RowType instances. Used when
/// deserializing Type objects.
class DeserializedTypeCache {
 public:
  void deserialize(const folly::dynamic& obj);

  size_t size() const {
    return cache_.size();
  }

  const TypePtr& get(int32_t id) const;

  void clear() {
    cache_.clear();
  }

 private:
  folly::F14FastMap<int32_t, TypePtr> cache_;
};

DeserializedTypeCache& deserializedTypeCache();

} // namespace facebook::velox
