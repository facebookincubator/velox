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

#include "velox/dwio/parquet/common/BitStreamUtilsInternal.h"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

namespace facebook::velox::parquet {
namespace {

// More than 3 blocks of 32 values, so that a batch also has a tail that
// fastpforlib doesn't unpack.
constexpr int kNumValues{100};

uint64_t maxValue(int numBits) {
  return numBits == 64 ? ~0ULL : (1ULL << numBits) - 1;
}

// Returns 'kNumValues' random values that fit in 'numBits' bits.
std::vector<uint64_t> makeValues(int numBits) {
  std::mt19937_64 generator{static_cast<uint64_t>(numBits)};
  std::vector<uint64_t> values(kNumValues);
  for (auto& value : values) {
    value = generator() & maxValue(numBits);
  }
  // Cover the values with all bits set and with no bits set.
  values[1] = maxValue(numBits);
  values[2] = 0;
  return values;
}

int numPackedBytes(int numBits) {
  return (kNumValues * numBits + 7) / 8;
}

// Packs 'values' with 'numBits' bits each, least significant bit first, after
// 'offset' bytes of padding. The padding makes the packed data start at
// 'offset' bytes from an aligned address. One more byte keeps the buffer
// non-empty when 'numBits' is 0, because BitReader rejects a null buffer.
std::vector<uint8_t>
pack(const std::vector<uint64_t>& values, int numBits, int offset) {
  std::vector<uint8_t> buffer(offset + numPackedBytes(numBits) + 1);
  uint64_t bitPosition = offset * 8;
  for (const auto value : values) {
    for (int bit = 0; bit < numBits; ++bit) {
      if ((value >> bit) & 1) {
        buffer[bitPosition / 8] |= 1 << (bitPosition % 8);
      }
      ++bitPosition;
    }
  }
  return buffer;
}

template <typename T>
void testGetBatch(int numBits, int numLeadingValues) {
  const auto values = makeValues(numBits);
  const std::vector<T> expected(values.begin(), values.end());
  for (int offset = 0; offset < 4; ++offset) {
    SCOPED_TRACE(
        fmt::format(
            "numBits: {}, offset: {}, numLeadingValues: {}",
            numBits,
            offset,
            numLeadingValues));
    const auto buffer = pack(values, numBits, offset);
    BitReader reader(buffer.data() + offset, numPackedBytes(numBits));
    std::vector<T> result(kNumValues);
    for (int i = 0; i < numLeadingValues; ++i) {
      ASSERT_TRUE(reader.GetValue(numBits, &result[i]));
    }
    ASSERT_EQ(
        reader.GetBatch(
            numBits,
            result.data() + numLeadingValues,
            kNumValues - numLeadingValues),
        kNumValues - numLeadingValues);
    EXPECT_EQ(result, expected);
  }
}

template <typename T>
void testGetBatch() {
  for (int numBits = 0; numBits <= static_cast<int>(sizeof(T) * 8); ++numBits) {
    testGetBatch<T>(numBits, 0);
  }
}

TEST(BitReaderTest, getBatchUint8) {
  testGetBatch<uint8_t>();
}

TEST(BitReaderTest, getBatchUint16) {
  testGetBatch<uint16_t>();
}

TEST(BitReaderTest, getBatchUint32) {
  testGetBatch<uint32_t>();
}

TEST(BitReaderTest, getBatchUint64) {
  testGetBatch<uint64_t>();
}

// Reading single values first leaves the batch at a bit offset inside a byte
// and at any byte offset.
TEST(BitReaderTest, getBatchAfterGetValue) {
  for (int numLeadingValues = 1; numLeadingValues <= 8; ++numLeadingValues) {
    for (int numBits = 1; numBits <= 32; ++numBits) {
      testGetBatch<uint32_t>(numBits, numLeadingValues);
    }
    for (int numBits = 33; numBits <= 64; ++numBits) {
      testGetBatch<uint64_t>(numBits, numLeadingValues);
    }
  }
}

} // namespace
} // namespace facebook::velox::parquet
