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

#include "velox/common/base/BloomFilter.h"

namespace facebook::velox {

namespace {
// We use 4 independent hash functions by taking 24 bits of
// the hash code and breaking these up into 4 groups of 6 bits. Each group
// represents a number between 0 and 63 (2^6-1) and maps to one bit in a
// 64-bit number. We combine these to get a 64-bit number with up to 4 bits
// set.
uint64_t bloomMask(uint64_t hashCode) {
  return (1L << (hashCode & 63)) | (1L << ((hashCode >> 6) & 63)) |
      (1L << ((hashCode >> 12) & 63)) | (1L << ((hashCode >> 18) & 63));
}

// Skip 24 bits used for bloomMask and use the next N bits of the hash code
// as index. N = log2(bloomSize). bloomSize must be a power of 2.
uint32_t bloomIndex(uint32_t bloomSize, uint64_t hashCode) {
  return ((hashCode >> 24) & (bloomSize - 1));
}
} // namespace

bool BloomFilterBase::test(
    const uint64_t* bloom,
    int32_t bloomSize,
    uint64_t hashCode) {
  auto mask = bloomMask(hashCode);
  auto index = bloomIndex(bloomSize, hashCode);
  return mask == (bloom[index] & mask);
}

void BloomFilterBase::set(
    uint64_t* bloom,
    int32_t bloomSize,
    uint64_t hashCode) {
  auto mask = bloomMask(hashCode);
  auto index = bloomIndex(bloomSize, hashCode);
  bloom[index] |= mask;
}

BloomFilterView::BloomFilterView(const char* serializedBloom) {
  common::InputByteStream stream(serializedBloom);
  version_ = stream.read<int8_t>();
  VELOX_USER_CHECK_EQ(kBloomFilterV1, version_);
  size_ = stream.read<int32_t>();
  VELOX_USER_CHECK_GT(size_, 0);
  bitsData_ =
      reinterpret_cast<const uint64_t*>(serializedBloom + stream.offset());
}
} // namespace facebook::velox
