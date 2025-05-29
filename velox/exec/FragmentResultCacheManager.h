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

// #include "dwio/common/Options.h"
// #include "serializers/PrestoSerializer.h"
#include "velox/connectors/Connector.h"

namespace facebook::velox::exec {
struct CacheKey {
  std::string planIdentifier;
  std::string splitIdentifier;
  CacheKey() = default;
  CacheKey(const std::string& string, const std::string& basic_string) {
    planIdentifier = string;
    splitIdentifier = basic_string;
  }
};

struct CacheKeyHasher {
  size_t operator()(const CacheKey& key) const {
    return std::hash<std::string>()(key.planIdentifier + key.splitIdentifier);
  }
};

struct CacheKeyEqual {
  bool operator()(const CacheKey& lhs, const CacheKey& rhs) const {
    return lhs.planIdentifier == rhs.planIdentifier && lhs.splitIdentifier == rhs.splitIdentifier;

  }
};

class FragmentResultCacheManager {
 public:
  void put(
      std::string planIdentifier,
      std::string splitIdentifier,
      std::vector<RowVectorPtr> result);
  bool get(
      std::string planIdentifier,
      std::string splitIdentifier,
      std::vector<RowVectorPtr>& result);

 private:
  std::unordered_map<
      CacheKey,
      std::vector<RowVectorPtr>,
      CacheKeyHasher,
      CacheKeyEqual>
      cache_;

  size_t capacity_ = 1000;
  std::list<CacheKey> cache_list_;
  // std::unique_ptr<serializer::presto::PrestoVectorSerde> serde_ =
  //     std::make_unique<serializer::presto::PrestoVectorSerde>();
  // VectorSerde::Options serdeOptions_{};
  // Memory pool placeholder
  // std::shared_ptr<velox::memory::MemoryPool> rootPool_{
  //   velox::memory::memoryManager()->addRootPool()};
  // std::shared_ptr<velox::memory::MemoryPool> pool_{
  //   rootPool_->addLeafChild("leaf")};
};

} // namespace facebook::velox::exec
