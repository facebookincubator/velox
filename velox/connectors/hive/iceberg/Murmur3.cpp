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

#include "Murmur3.h"

#include <cstddef>
#include <cstring>

#include "velox/type/HugeInt.h"

namespace facebook::velox::connectors::hive::iceberg {

std::shared_ptr<uint8_t> Murmur3_32::intToMinimalBytes(
    int64_t value,
    int* length) {
  auto bytes = std::shared_ptr<uint8_t>(
      new uint8_t[9], std::default_delete<uint8_t[]>());
  const bool isNegative = (value < 0);
  int32_t idx = 0;
  bool started = false;

  for (int32_t i = 7; i >= 0; --i) {
    auto byte = static_cast<uint8_t>((value >> (i * 8)) & 0xFF);
    if (!started) {
      if (i > 0) {
        auto nextByte = static_cast<uint8_t>((value >> ((i - 1) * 8)) & 0xFF);
        if (isNegative && byte == 0xFF && (nextByte & 0x80)) {
          continue;
        }
        if (!isNegative && byte == 0x00 && !(nextByte & 0x80)) {
          continue;
        }
      }
      started = true;
    }

    bytes.get()[idx++] = byte;
  }

  if (idx == 0) {
    // Value=0.
    bytes.get()[idx++] = 0x00;
  } else {
    bool needSignByte = (isNegative && (bytes.get()[0] & 0x80) == 0) ||
        (!isNegative && (bytes.get()[0] & 0x80) != 0);
    if (needSignByte) {
      memmove(bytes.get() + 1, bytes.get(), idx);
      bytes.get()[0] = isNegative ? 0xFF : 0x00;
      idx++;
    }
  }

  *length = idx;
  return bytes;
}

uint32_t Murmur3_32::hash(const uint8_t* data, size_t len, uint32_t seed) {
  uint32_t h1{seed};
  uint32_t k1{0};
  constexpr size_t bsize = sizeof(k1);
  const size_t nblocks = len / bsize;

  // Body.
  for (size_t i = 0; i < nblocks; i++, data += bsize) {
    memcpy(&k1, data, bsize);

    k1 = mixK1(k1);
    h1 = mixH1(h1, k1);
  }

  k1 = 0;

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
  h1 ^= static_cast<uint32_t>(len);
  FMIX32(h1);
  return h1;
}

uint32_t Murmur3_32::hash(
    const facebook::velox::StringView& value,
    uint32_t seed) {
  return hash(
      reinterpret_cast<const uint8_t*>(value.data()), value.size(), seed);
}

uint32_t Murmur3_32::hash(int32_t value, uint32_t seed) {
  return hash(static_cast<uint64_t>(value), seed);
}

uint32_t Murmur3_32::hash(uint32_t value, uint32_t seed) {
  return hash(static_cast<uint64_t>(value), seed);
}

uint32_t Murmur3_32::hash(int64_t value, uint32_t seed) {
  return hash(static_cast<uint64_t>(value), seed);
}

uint32_t Murmur3_32::hash(uint64_t value, uint32_t seed) {
  auto h1 = seed;
  const auto low = static_cast<uint32_t>(value & 0xFFFFFFFF);
  const auto high = static_cast<uint32_t>((value >> 32) & 0xFFFFFFFF);

  auto k1 = mixK1(low);
  h1 = mixH1(h1, k1);

  k1 = mixK1(high);
  h1 = mixH1(h1, k1);

  h1 ^= sizeof(uint64_t);
  FMIX32(h1);
  return h1;
}

uint32_t Murmur3_32::hash(folly::int128_t value, uint32_t seed) {
  char buf[sizeof(folly::int128_t)] = {};
  HugeInt::serialize(value, buf);
  return hash(reinterpret_cast<const uint8_t*>(buf), sizeof(buf), seed);
}

uint32_t Murmur3_32::hashDecimal(int64_t value, uint32_t seed) {
  int32_t length{0};
  const auto bytes = intToMinimalBytes(value, &length);
  return hash(bytes.get(), length, seed);
}
} // namespace facebook::velox::connectors::hive::iceberg
