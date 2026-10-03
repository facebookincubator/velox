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

#include <cstring>
#include <optional>
#include <vector>

#include "velox/common/base/SplitBlockBloomFilter.h"
#include "velox/core/QueryConfig.h"
#include "velox/functions/Macros.h"

namespace facebook::velox::functions::sparksql {

template <typename T>
struct BloomFilterMightContainFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig&,
      const arg_type<Varbinary>* serialized,
      const arg_type<int64_t>*) {
    if (serialized == nullptr) {
      return;
    }

    VELOX_USER_CHECK_GT(
        serialized->size(), 0, "Serialized split-block Bloom filter is empty");
    VELOX_USER_CHECK_EQ(
        serialized->size() % sizeof(SplitBlockBloomFilter::Block),
        0,
        "Invalid serialized split-block Bloom filter size: {}",
        serialized->size());
    using Block = SplitBlockBloomFilter::Block;
    const auto numBlocks =
        static_cast<int32_t>(serialized->size() / sizeof(Block));
    const auto* serializedBlocks = serialized->data();
    Block* blocks;
    if (reinterpret_cast<uintptr_t>(serializedBlocks) % alignof(Block) == 0) {
      // Reuse aligned serialized blocks without copying.
      blocks = reinterpret_cast<Block*>(const_cast<char*>(serializedBlocks));
    } else {
      // Copy unaligned serialized blocks into aligned storage.
      ownedBlocks_.resize(numBlocks);
      std::memcpy(
          ownedBlocks_.data(), serializedBlocks, numBlocks * sizeof(Block));
      blocks = ownedBlocks_.data();
    }
    bloomFilter_.emplace(std::span<Block>(blocks, numBlocks));
  }

  FOLLY_ALWAYS_INLINE void
  call(bool& result, const arg_type<Varbinary>&, const int64_t& input) {
    result = bloomFilter_.has_value() &&
        bloomFilter_->mayContain(folly::hasher<int64_t>()(input));
  }

 private:
  std::vector<SplitBlockBloomFilter::Block> ownedBlocks_;
  std::optional<SplitBlockBloomFilter> bloomFilter_;
};

} // namespace facebook::velox::functions::sparksql
