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
#include <string>
#include <vector>

#include "velox/common/base/SplitBlockBloomFilter.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/exec/AggregateUtil.h"

namespace facebook::velox::functions::aggregate::sparksql {

class BloomFilterAccumulator {
 public:
  explicit BloomFilterAccumulator(HashStringAllocator* allocator);

  int32_t serializedSize() const;

  void serialize(char* output) const;

  void merge(const StringView& serialized);

  bool initialized() const;

  void init(int32_t numBlocks);

  void insert(int64_t value);

  SplitBlockBloomFilter* bloomFilter();

  const SplitBlockBloomFilter* bloomFilter() const;

 private:
  using Block = SplitBlockBloomFilter::Block;
  using BlockAllocator = AlignedStlAllocator<Block, alignof(Block)>;

  std::vector<Block, BlockAllocator> blocks_;
  std::optional<SplitBlockBloomFilter> bloomFilter_;
};

exec::AggregateRegistrationResult registerBloomFilterAggAggregate(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite);

} // namespace facebook::velox::functions::aggregate::sparksql
