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

#include "velox/functions/lib/HllAccumulator.h"
#include "velox/type/StringView.h"

namespace facebook::velox::functions::aggregate::sparksql {

class HyperLogLogPlusPlusHelper final {
 public:
  explicit HyperLogLogPlusPlusHelper(HashStringAllocator* allocator)
      : hll_(allocator) {}

  HyperLogLogPlusPlusHelper(
      int8_t indexBitLength,
      HashStringAllocator* allocator)
      : hll_(indexBitLength, allocator) {}

  void insertHash(uint64_t hash) {
    hll_.insertHash(hash);
  }

  int64_t cardinality() const {
    return hll_.cardinality();
  }

  void mergeWith(const HyperLogLogPlusPlusHelper& other) {
    hll_.mergeWith(other.hll_);
  }

  void mergeWith(StringView serialized, HashStringAllocator* allocator) {
    hll_.mergeWith(serialized, allocator);
  }

  int32_t serializedSize() {
    return hll_.serializedSize();
  }

  void serialize(char* outputBuffer) {
    hll_.serialize(outputBuffer);
  }

 private:
  common::hll::HllAccumulator<int64_t, false, HashStringAllocator> hll_;
};

} // namespace facebook::velox::functions::aggregate::sparksql
