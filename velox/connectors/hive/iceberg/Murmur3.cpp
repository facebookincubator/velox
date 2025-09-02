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

#include "velox/connectors/hive/iceberg/Murmur3.h"
#include "velox/type/DecimalUtil.h"
#include "velox/type/HugeInt.h"

namespace facebook::velox::connector::hive::iceberg {

int32_t Murmur3Hash32::hash(const char* const input, size_t len) {
  uint32_t h1{kDefaultSeed};
  uint32_t k1{0};
  const uint8_t* data = reinterpret_cast<const uint8_t*>(input);
  const size_t nblocks = len / 4;

  // Body.
  for (size_t i = 0; i < nblocks; i++) {
    uint32_t k1 = *reinterpret_cast<const uint32_t*>(data + i * 4);
    k1 = mixK1(k1);
    h1 = mixH1(h1, k1);
  }

  k1 = 0;
  data = data + nblocks * 4;

  // Tail.
  switch (len & 3) {
    case 3:
      k1 ^= (static_cast<uint32_t>(data[2])) << 16;
      [[fallthrough]];
    case 2:
      k1 ^= (static_cast<uint32_t>(data[1])) << 8;
      [[fallthrough]];
    case 1:
      k1 ^= data[0];
      k1 = mixK1(k1);
      h1 ^= k1;
  };

  // Finalization.
  return fmix32(h1, len);
}

int32_t Murmur3Hash32::hash(const StringView& value) {
  return hash(value.data(), value.size());
}

int32_t Murmur3Hash32::hash(uint64_t value) {
  auto h1 = kDefaultSeed;
  const auto low = static_cast<uint32_t>(value & 0xFFFFFFFF);
  const auto high = static_cast<uint32_t>((value >> 32) & 0xFFFFFFFF);

  auto k1 = mixK1(low);
  h1 = mixH1(h1, k1);

  k1 = mixK1(high);
  h1 = mixH1(h1, k1);

  return fmix32(h1, sizeof(uint64_t));
}

int32_t Murmur3Hash32::hashDecimal(int128_t value) {
  char bytes[16];
  const auto length = DecimalUtil::getByteArrayLength(value);
  DecimalUtil::toByteArray(value, bytes);
  return hash(bytes, length);
}

} // namespace facebook::velox::connector::hive::iceberg
