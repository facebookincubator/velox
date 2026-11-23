/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <numeric>

#include "folly/Random.h"
#include "folly/Range.h"
#include "folly/Varint.h"
#include "velox/dwio/nimble/common/Varint.h"

using namespace ::facebook;

namespace {
const int kNumElements = 10000;

// Encode a vector of values into a varint buffer, returning the buffer and its
// size.
template <typename T>
std::pair<std::unique_ptr<char[]>, size_t> encodeValues(
    const std::vector<T>& values) {
  auto buf =
      std::make_unique<char[]>(values.size() * folly::kMaxVarintLength64);
  char* pos = buf.get();
  for (auto val : values) {
    nimble::varint::writeVarint(val, &pos);
  }
  return {std::move(buf), static_cast<size_t>(pos - buf.get())};
}

// Bulk-decode and verify the result matches the expected values.
template <typename T>
void verifyBulkDecode(const std::vector<T>& expected, const char* encoded) {
  std::vector<T> decoded(expected.size());
  if constexpr (sizeof(T) == 4) {
    nimble::varint::bulkVarintDecode32(
        expected.size(), encoded, decoded.data());
  } else {
    nimble::varint::bulkVarintDecode64(
        expected.size(), encoded, decoded.data());
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(expected[i], decoded[i])
        << "mismatch at index " << i << " of " << expected.size();
  }
}

} // namespace

TEST(VarintTests, varintSize32) {
  // Boundary values for varint encoding.
  EXPECT_EQ(nimble::varint::varintSize(uint32_t{0}), 1);
  EXPECT_EQ(nimble::varint::varintSize(uint32_t{1}), 1);
  EXPECT_EQ(nimble::varint::varintSize(uint32_t{127}), 1);
  EXPECT_EQ(nimble::varint::varintSize(uint32_t{128}), 2);
  EXPECT_EQ(nimble::varint::varintSize(uint32_t{16383}), 2);
  EXPECT_EQ(nimble::varint::varintSize(uint32_t{16384}), 3);
  EXPECT_EQ(nimble::varint::varintSize(uint32_t{2097151}), 3);
  EXPECT_EQ(nimble::varint::varintSize(uint32_t{2097152}), 4);
  EXPECT_EQ(nimble::varint::varintSize(uint32_t{268435455}), 4);
  EXPECT_EQ(nimble::varint::varintSize(uint32_t{268435456}), 5);
  EXPECT_EQ(
      nimble::varint::varintSize(std::numeric_limits<uint32_t>::max()), 5);

  // Verify consistency with writeVarint for random values.
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);
  char buf[folly::kMaxVarintLength32];
  for (int i = 0; i < kNumElements; ++i) {
    const int bitShift = folly::Random::rand32(rng) % 32;
    uint32_t val = folly::Random::rand32(rng) >> bitShift;
    char* pos = buf;
    nimble::varint::writeVarint(val, &pos);
    ASSERT_EQ(nimble::varint::varintSize(val), static_cast<uint32_t>(pos - buf))
        << "mismatch for val=" << val;
  }
}

TEST(VarintTests, varintSize64) {
  // Boundary values for varint encoding.
  EXPECT_EQ(nimble::varint::varintSize(uint64_t{0}), 1);
  EXPECT_EQ(nimble::varint::varintSize(uint64_t{127}), 1);
  EXPECT_EQ(nimble::varint::varintSize(uint64_t{128}), 2);
  EXPECT_EQ(nimble::varint::varintSize(uint64_t{16383}), 2);
  EXPECT_EQ(nimble::varint::varintSize(uint64_t{16384}), 3);
  EXPECT_EQ(
      nimble::varint::varintSize(std::numeric_limits<uint64_t>::max()), 10);

  // Verify consistency with writeVarint for random values.
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);
  char buf[folly::kMaxVarintLength64];
  for (int i = 0; i < kNumElements; ++i) {
    const int bitShift = folly::Random::rand64(rng) % 64;
    uint64_t val = folly::Random::rand64(rng) >> bitShift;
    char* pos = buf;
    nimble::varint::writeVarint(val, &pos);
    ASSERT_EQ(nimble::varint::varintSize(val), static_cast<uint32_t>(pos - buf))
        << "mismatch for val=" << val;
  }
}

TEST(VarintTests, maxVarintSizeForBitWidth) {
  EXPECT_EQ(nimble::varint::maxVarintSizeForBitWidth(0), 0);
  EXPECT_EQ(nimble::varint::maxVarintSizeForBitWidth(1), 1);
  EXPECT_EQ(nimble::varint::maxVarintSizeForBitWidth(7), 1);
  EXPECT_EQ(nimble::varint::maxVarintSizeForBitWidth(8), 2);
  EXPECT_EQ(nimble::varint::maxVarintSizeForBitWidth(63), 9);
  EXPECT_EQ(nimble::varint::maxVarintSizeForBitWidth(64), 10);
  EXPECT_EQ(nimble::varint::maxVarintSizeForBitWidth(65), 10);
}

TEST(VarintTests, writeRead32) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  std::vector<uint32_t> data;
  // Generate data with uniform bit length.
  for (int i = 0; i < kNumElements; ++i) {
    const int bitShift = folly::Random::rand32(rng) % 32;
    data.push_back(folly::Random::rand32(rng) >> bitShift);
  }

  auto buffer =
      std::make_unique<char[]>(kNumElements * folly::kMaxVarintLength32);
  char* pos = buffer.get();
  for (int i = 0; i < kNumElements; ++i) {
    nimble::varint::writeVarint(data[i], &pos);
  }

  auto follyBuffer =
      std::make_unique<uint8_t[]>(kNumElements * folly::kMaxVarintLength32);
  uint8_t* fpos = follyBuffer.get();
  for (int i = 0; i < kNumElements; ++i) {
    fpos += folly::encodeVarint(data[i], fpos);
  }

  ASSERT_EQ(pos - buffer.get(), fpos - follyBuffer.get());

  ASSERT_EQ(nimble::varint::bulkVarintSize32(data), pos - buffer.get());

  const char* cpos = buffer.get();
  for (int i = 0; i < kNumElements; ++i) {
    ASSERT_EQ(data[i], nimble::varint::readVarint32(&cpos));
  }

  const uint8_t* fstart = follyBuffer.get();
  const uint8_t* fend = follyBuffer.get() + (fpos - follyBuffer.get());
  folly::Range<const uint8_t*> frange(fstart, fend);
  for (int i = 0; i < kNumElements; ++i) {
    ASSERT_EQ(data[i], folly::decodeVarint(frange));
  }

  std::vector<uint32_t> bulk(kNumElements);
  cpos = buffer.get();
  nimble::varint::bulkVarintDecode32(kNumElements, cpos, bulk.data());
  for (int i = 0; i < kNumElements; ++i) {
    ASSERT_EQ(data[i], bulk[i]);
  }
}

TEST(VarintTests, writeRead64) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  std::vector<uint64_t> data;
  // Generate data with uniform bit length.
  for (int i = 0; i < kNumElements; ++i) {
    const int bitShift = folly::Random::rand64(rng) % 64;
    data.push_back(folly::Random::rand64(rng) >> bitShift);
  }

  auto buffer =
      std::make_unique<char[]>(kNumElements * folly::kMaxVarintLength64);
  char* pos = buffer.get();
  for (int i = 0; i < kNumElements; ++i) {
    nimble::varint::writeVarint(data[i], &pos);
  }

  auto follyBuffer =
      std::make_unique<uint8_t[]>(kNumElements * folly::kMaxVarintLength64);
  uint8_t* fpos = follyBuffer.get();
  for (int i = 0; i < kNumElements; ++i) {
    fpos += folly::encodeVarint(data[i], fpos);
  }

  ASSERT_EQ(pos - buffer.get(), fpos - follyBuffer.get());

  ASSERT_EQ(nimble::varint::bulkVarintSize64(data), pos - buffer.get());

  const char* cpos = buffer.get();
  for (int i = 0; i < kNumElements; ++i) {
    ASSERT_EQ(data[i], nimble::varint::readVarint64(&cpos));
  }

  const uint8_t* fstart = follyBuffer.get();
  const uint8_t* fend = follyBuffer.get() + (fpos - follyBuffer.get());
  folly::Range<const uint8_t*> frange(fstart, fend);
  for (int i = 0; i < kNumElements; ++i) {
    ASSERT_EQ(data[i], folly::decodeVarint(frange));
  }

  std::vector<uint64_t> bulk(kNumElements);
  cpos = buffer.get();
  nimble::varint::bulkVarintDecode64(kNumElements, cpos, bulk.data());
  for (int i = 0; i < kNumElements; ++i) {
    ASSERT_EQ(data[i], bulk[i]);
  }
}

// ============================================================================
// Single-byte varint tests: exercise the SIMD decodeSingleByteRun path.
// The function has three loops:
//   1. Wide loop: processes kU8BatchSize bytes (32 on AVX2, 16 on SSE/NEON)
//   2. 8-byte loop: processes 8 bytes at a time
//   3. Tail loop: processes 1 byte at a time
// These tests cover boundary conditions for all three loops.
// ============================================================================

// All 128 single-byte values (0-127) decode correctly for uint32_t.
TEST(VarintTests, singleByte32AllValues) {
  std::vector<uint32_t> data(128);
  std::iota(data.begin(), data.end(), 0);
  auto [buf, size] = encodeValues(data);
  ASSERT_EQ(size, 128u);
  verifyBulkDecode(data, buf.get());
}

// All 128 single-byte values (0-127) decode correctly for uint64_t.
TEST(VarintTests, singleByte64AllValues) {
  std::vector<uint64_t> data(128);
  std::iota(data.begin(), data.end(), 0);
  auto [buf, size] = encodeValues(data);
  ASSERT_EQ(size, 128u);
  verifyBulkDecode(data, buf.get());
}

// Test every count from 0 to 100 with all-zero values.
// Exercises exact boundary transitions between wide/8-byte/tail loops.
TEST(VarintTests, singleByte32AllCountsZero) {
  for (int count = 0; count <= 100; ++count) {
    std::vector<uint32_t> data(count, 0);
    auto [buf, size] = encodeValues(data);
    verifyBulkDecode(data, buf.get());
  }
}

// Test every count from 0 to 100 with value 127 (max single-byte varint).
TEST(VarintTests, singleByte32AllCountsMax) {
  for (int count = 0; count <= 100; ++count) {
    std::vector<uint32_t> data(count, 127);
    auto [buf, size] = encodeValues(data);
    verifyBulkDecode(data, buf.get());
  }
}

// Test counts at specific SIMD boundaries with uint64_t.
TEST(VarintTests, singleByte64SimdBoundaries) {
  for (int count : {0,  1,  2,  7,  8,  9,  15,  16,  17,  31,
                    32, 33, 63, 64, 65, 96, 127, 128, 256, 1000}) {
    std::vector<uint64_t> data(count);
    for (int i = 0; i < count; ++i) {
      data[i] = i % 128;
    }
    auto [buf, size] = encodeValues(data);
    ASSERT_EQ(size, static_cast<size_t>(count));
    verifyBulkDecode(data, buf.get());
  }
}

// A multi-byte varint (>=128) interrupts the single-byte run at each position
// within a 64-element window. Verifies the SIMD path correctly bails out and
// the remaining elements are decoded by the fallback path.
TEST(VarintTests, singleByte32MultiByteInterrupt) {
  for (int interruptPos = 0; interruptPos < 64; ++interruptPos) {
    const int count = 64;
    std::vector<uint32_t> data(count);
    for (int i = 0; i < count; ++i) {
      data[i] = (i == interruptPos) ? 200 : (i % 128);
    }
    auto [buf, size] = encodeValues(data);
    verifyBulkDecode(data, buf.get());
  }
}

TEST(VarintTests, singleByte64MultiByteInterrupt) {
  for (int interruptPos = 0; interruptPos < 64; ++interruptPos) {
    const int count = 64;
    std::vector<uint64_t> data(count);
    for (int i = 0; i < count; ++i) {
      data[i] = (i == interruptPos) ? 200 : (i % 128);
    }
    auto [buf, size] = encodeValues(data);
    verifyBulkDecode(data, buf.get());
  }
}

// Single-byte values followed by progressively longer multi-byte varints.
// Tests the transition from decodeSingleByteRun into the BMI2/scalar path.
TEST(VarintTests, singleByte32TransitionToMultiByte) {
  for (int singleCount : {0, 1, 7, 8, 15, 16, 31, 32, 33, 64}) {
    for (int multiCount : {0, 1, 5, 10}) {
      std::vector<uint32_t> data;
      data.reserve(singleCount + multiCount);
      for (int i = 0; i < singleCount; ++i) {
        data.push_back(i % 128);
      }
      for (int i = 0; i < multiCount; ++i) {
        data.push_back(128 + i * 1000);
      }
      auto [buf, size] = encodeValues(data);
      verifyBulkDecode(data, buf.get());
    }
  }
}

TEST(VarintTests, singleByte64TransitionToMultiByte) {
  for (int singleCount : {0, 1, 7, 8, 15, 16, 31, 32, 33, 64}) {
    for (int multiCount : {0, 1, 5, 10}) {
      std::vector<uint64_t> data;
      data.reserve(singleCount + multiCount);
      for (int i = 0; i < singleCount; ++i) {
        data.push_back(i % 128);
      }
      for (int i = 0; i < multiCount; ++i) {
        data.push_back(128 + static_cast<uint64_t>(i) * 1000);
      }
      auto [buf, size] = encodeValues(data);
      verifyBulkDecode(data, buf.get());
    }
  }
}

// Alternating single-byte and multi-byte varints. The SIMD path must
// correctly handle frequent bail-outs and re-entries.
TEST(VarintTests, singleByte32AlternatingSingleMulti) {
  std::vector<uint32_t> data;
  data.reserve(200);
  for (int i = 0; i < 200; ++i) {
    data.push_back(i % 2 == 0 ? (i % 128) : (128 + i));
  }
  auto [buf, size] = encodeValues(data);
  verifyBulkDecode(data, buf.get());
}

// Large run of single-byte varints to stress the wide SIMD loop.
TEST(VarintTests, singleByte32LargeRun) {
  const int count = 100000;
  std::vector<uint32_t> data(count);
  for (int i = 0; i < count; ++i) {
    data[i] = i % 128;
  }
  auto [buf, size] = encodeValues(data);
  ASSERT_EQ(size, static_cast<size_t>(count));
  verifyBulkDecode(data, buf.get());
}

TEST(VarintTests, singleByte64LargeRun) {
  const int count = 100000;
  std::vector<uint64_t> data(count);
  for (int i = 0; i < count; ++i) {
    data[i] = i % 128;
  }
  auto [buf, size] = encodeValues(data);
  ASSERT_EQ(size, static_cast<size_t>(count));
  verifyBulkDecode(data, buf.get());
}

// Random mix: ~80% single-byte, ~20% multi-byte, with a random seed.
TEST(VarintTests, singleByte32RandomMix) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  std::vector<uint32_t> data(kNumElements);
  for (int i = 0; i < kNumElements; ++i) {
    if (folly::Random::rand32(rng) % 5 != 0) {
      data[i] = folly::Random::rand32(rng) % 128;
    } else {
      data[i] = 128 + folly::Random::rand32(rng) % 10000;
    }
  }
  auto [buf, size] = encodeValues(data);
  verifyBulkDecode(data, buf.get());
}

TEST(VarintTests, singleByte64RandomMix) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  std::vector<uint64_t> data(kNumElements);
  for (int i = 0; i < kNumElements; ++i) {
    if (folly::Random::rand32(rng) % 5 != 0) {
      data[i] = folly::Random::rand32(rng) % 128;
    } else {
      data[i] = 128 + folly::Random::rand64(rng) % 1000000;
    }
  }
  auto [buf, size] = encodeValues(data);
  verifyBulkDecode(data, buf.get());
}

// Constant value runs for each single-byte value.
TEST(VarintTests, singleByte32ConstantRuns) {
  for (uint32_t val = 0; val < 128; ++val) {
    std::vector<uint32_t> data(37, val);
    auto [buf, size] = encodeValues(data);
    verifyBulkDecode(data, buf.get());
  }
}

// Verify that a single multi-byte varint at the very start works.
TEST(VarintTests, singleByte32MultiByteFirst) {
  std::vector<uint32_t> data = {300};
  for (int i = 0; i < 50; ++i) {
    data.push_back(i % 128);
  }
  auto [buf, size] = encodeValues(data);
  verifyBulkDecode(data, buf.get());
}

// Verify that a single multi-byte varint at the very end works.
TEST(VarintTests, singleByte32MultiByteLast) {
  std::vector<uint32_t> data;
  data.reserve(51);
  for (int i = 0; i < 50; ++i) {
    data.push_back(i % 128);
  }
  data.push_back(300);
  auto [buf, size] = encodeValues(data);
  verifyBulkDecode(data, buf.get());
}

// ============================================================================
// Two-byte varint tests: exercise the bulkDecodeTwoByteRun fast path.
// Two-byte varints have values in [128, 16383]. The fast path processes
// 4 varints (8 bytes) at a time, then handles trailing varints one at a time.
// ============================================================================

// All two-byte boundary values decode correctly for uint32_t.
TEST(VarintTests, twoByte32BoundaryValues) {
  std::vector<uint32_t> data = {128, 255, 256, 1000, 8191, 16383};
  auto [buf, size] = encodeValues(data);
  ASSERT_EQ(size, 12u);
  verifyBulkDecode(data, buf.get());
}

// All two-byte boundary values decode correctly for uint64_t.
TEST(VarintTests, twoByte64BoundaryValues) {
  std::vector<uint64_t> data = {128, 255, 256, 1000, 8191, 16383};
  auto [buf, size] = encodeValues(data);
  ASSERT_EQ(size, 12u);
  verifyBulkDecode(data, buf.get());
}

// Test every count from 0 to 100 with uniform two-byte varints.
// Exercises the 4-at-a-time wide loop and trailing scalar loop boundaries.
TEST(VarintTests, twoByte32AllCounts) {
  for (int count = 0; count <= 100; ++count) {
    std::vector<uint32_t> data(count, 200);
    auto [buf, size] = encodeValues(data);
    ASSERT_EQ(size, static_cast<size_t>(count * 2));
    verifyBulkDecode(data, buf.get());
  }
}

// Test counts at multiples of 4 (wide loop boundary) with uint64_t.
TEST(VarintTests, twoByte64WideBoundaries) {
  for (int count : {0, 1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 32, 64, 100, 1000}) {
    std::vector<uint64_t> data(count);
    for (int i = 0; i < count; ++i) {
      data[i] = 128 + (i % 16256);
    }
    auto [buf, size] = encodeValues(data);
    ASSERT_EQ(size, static_cast<size_t>(count * 2));
    verifyBulkDecode(data, buf.get());
  }
}

// Constant value runs for representative two-byte values.
TEST(VarintTests, twoByte32ConstantRuns) {
  for (uint32_t val : {128u, 200u, 1000u, 8192u, 16383u}) {
    std::vector<uint32_t> data(37, val);
    auto [buf, size] = encodeValues(data);
    verifyBulkDecode(data, buf.get());
  }
}

// Large run of two-byte varints to stress the wide loop.
TEST(VarintTests, twoByte32LargeRun) {
  const int count = 100000;
  std::vector<uint32_t> data(count);
  for (int i = 0; i < count; ++i) {
    data[i] = 128 + (i % 16256);
  }
  auto [buf, size] = encodeValues(data);
  ASSERT_EQ(size, static_cast<size_t>(count * 2));
  verifyBulkDecode(data, buf.get());
}

TEST(VarintTests, twoByte64LargeRun) {
  const int count = 100000;
  std::vector<uint64_t> data(count);
  for (int i = 0; i < count; ++i) {
    data[i] = 128 + (i % 16256);
  }
  auto [buf, size] = encodeValues(data);
  ASSERT_EQ(size, static_cast<size_t>(count * 2));
  verifyBulkDecode(data, buf.get());
}

// A 3+ byte varint interrupts the two-byte run at each position within a
// 32-element window. Verifies the fast path correctly bails out.
TEST(VarintTests, twoByte32MultiByteInterrupt) {
  for (int interruptPos = 0; interruptPos < 32; ++interruptPos) {
    const int count = 32;
    std::vector<uint32_t> data(count);
    for (int i = 0; i < count; ++i) {
      data[i] = (i == interruptPos) ? 20000 : 200;
    }
    auto [buf, size] = encodeValues(data);
    verifyBulkDecode(data, buf.get());
  }
}

// Two-byte values followed by multi-byte values.
TEST(VarintTests, twoByte32TransitionToMultiByte) {
  for (int twoByteCount : {0, 1, 3, 4, 5, 8, 16, 32}) {
    for (int multiCount : {0, 1, 5, 10}) {
      std::vector<uint32_t> data;
      data.reserve(twoByteCount + multiCount);
      for (int i = 0; i < twoByteCount; ++i) {
        data.push_back(128 + (i % 16256));
      }
      for (int i = 0; i < multiCount; ++i) {
        data.push_back(20000 + i * 1000);
      }
      auto [buf, size] = encodeValues(data);
      verifyBulkDecode(data, buf.get());
    }
  }
}

// ============================================================================
// Dispatch loop tests: exercise the bulkVarintDecodeDispatch re-entry logic.
// The dispatch loop cycles: single-byte fast path -> two-byte fast path ->
// BMI2 general decoder, and the BMI2 decoder yields back when it detects
// a single-byte boundary (no carryover). These tests verify correct decoding
// across multiple dispatch loop iterations.
// ============================================================================

// Single-byte run, then two-byte run, then single-byte run again.
// The BMI2 decoder should not be entered at all.
TEST(VarintTests, dispatch32SingleTwoSingle) {
  std::vector<uint32_t> data;
  data.reserve(150);
  for (int i = 0; i < 50; ++i) {
    data.push_back(i % 128);
  }
  for (int i = 0; i < 50; ++i) {
    data.push_back(128 + (i % 16256));
  }
  for (int i = 0; i < 50; ++i) {
    data.push_back(i % 128);
  }
  auto [buf, size] = encodeValues(data);
  verifyBulkDecode(data, buf.get());
}

// A large 5-byte varint followed by single-byte varints.
// Tests the dispatch loop re-entering single-byte fast path after BMI2.
TEST(VarintTests, dispatch32LargeHeadThenSingleByte) {
  std::vector<uint32_t> data;
  data.reserve(201);
  data.push_back(UINT32_MAX);
  for (int i = 0; i < 200; ++i) {
    data.push_back(i % 128);
  }
  auto [buf, size] = encodeValues(data);
  verifyBulkDecode(data, buf.get());
}

// A large 5-byte varint followed by two-byte varints.
// Tests the dispatch loop re-entering two-byte fast path after BMI2.
TEST(VarintTests, dispatch32LargeHeadThenTwoByte) {
  std::vector<uint32_t> data;
  data.reserve(201);
  data.push_back(UINT32_MAX);
  for (int i = 0; i < 200; ++i) {
    data.push_back(128 + (i % 16256));
  }
  auto [buf, size] = encodeValues(data);
  verifyBulkDecode(data, buf.get());
}

TEST(VarintTests, dispatch64LargeHeadThenSingleByte) {
  std::vector<uint64_t> data;
  data.reserve(201);
  data.push_back(UINT64_MAX);
  for (int i = 0; i < 200; ++i) {
    data.push_back(i % 128);
  }
  auto [buf, size] = encodeValues(data);
  verifyBulkDecode(data, buf.get());
}

TEST(VarintTests, dispatch64LargeHeadThenTwoByte) {
  std::vector<uint64_t> data;
  data.reserve(201);
  data.push_back(UINT64_MAX);
  for (int i = 0; i < 200; ++i) {
    data.push_back(128 + (i % 16256));
  }
  auto [buf, size] = encodeValues(data);
  verifyBulkDecode(data, buf.get());
}

// Sporadic large values every N elements among single-byte values.
// Each large value forces the BMI2 decoder, then the dispatch loop must
// re-enter the single-byte fast path.
TEST(VarintTests, dispatch32SporadicLargeAmongSingleByte) {
  for (int interval : {4, 8, 16, 64, 256}) {
    const int count = 1024;
    std::vector<uint32_t> data(count);
    for (int i = 0; i < count; ++i) {
      data[i] = (i % interval == 0) ? UINT32_MAX : (i % 128);
    }
    auto [buf, size] = encodeValues(data);
    verifyBulkDecode(data, buf.get());
  }
}

// Sporadic large values every N elements among two-byte values.
TEST(VarintTests, dispatch32SporadicLargeAmongTwoByte) {
  for (int interval : {4, 8, 16, 64, 256}) {
    const int count = 1024;
    std::vector<uint32_t> data(count);
    for (int i = 0; i < count; ++i) {
      data[i] = (i % interval == 0) ? UINT32_MAX : (128 + i % 16256);
    }
    auto [buf, size] = encodeValues(data);
    verifyBulkDecode(data, buf.get());
  }
}

TEST(VarintTests, dispatch64SporadicLargeAmongSingleByte) {
  for (int interval : {4, 8, 16, 64, 256}) {
    const int count = 1024;
    std::vector<uint64_t> data(count);
    for (int i = 0; i < count; ++i) {
      data[i] = (i % interval == 0) ? UINT64_MAX : uint64_t(i % 128);
    }
    auto [buf, size] = encodeValues(data);
    verifyBulkDecode(data, buf.get());
  }
}

// Repeating pattern: single-byte block, two-byte block, large value.
// Exercises all three dispatch loop phases in sequence, multiple times.
TEST(VarintTests, dispatch32RepeatingThreePhase) {
  std::vector<uint32_t> data;
  data.reserve(20 * 21);
  for (int round = 0; round < 20; ++round) {
    for (int i = 0; i < 10; ++i) {
      data.push_back(i % 128);
    }
    for (int i = 0; i < 10; ++i) {
      data.push_back(128 + (i % 16256));
    }
    data.push_back(UINT32_MAX);
  }
  auto [buf, size] = encodeValues(data);
  verifyBulkDecode(data, buf.get());
}

TEST(VarintTests, dispatch64RepeatingThreePhase) {
  std::vector<uint64_t> data;
  data.reserve(20 * 21);
  for (int round = 0; round < 20; ++round) {
    for (int i = 0; i < 10; ++i) {
      data.push_back(i % 128);
    }
    for (int i = 0; i < 10; ++i) {
      data.push_back(128 + (i % 16256));
    }
    data.push_back(UINT64_MAX);
  }
  auto [buf, size] = encodeValues(data);
  verifyBulkDecode(data, buf.get());
}

// All 3-byte, 4-byte, and 5-byte varints (no fast path applies).
// Everything goes through BMI2 general decoder.
TEST(VarintTests, dispatch32AllMultiByte) {
  for (auto [lo, hi] : std::vector<std::pair<uint32_t, uint32_t>>{
           {16384, 2097151}, {2097152, 268435455}, {268435456, UINT32_MAX}}) {
    auto seed = folly::Random::rand32();
    LOG(INFO) << "seed: " << seed;
    std::mt19937 rng(seed);
    const int count = 500;
    std::vector<uint32_t> data(count);
    for (int i = 0; i < count; ++i) {
      data[i] = lo + folly::Random::rand32(rng) % (hi - lo + 1);
    }
    auto [buf, size] = encodeValues(data);
    verifyBulkDecode(data, buf.get());
  }
}

// Random mix of all varint widths with dispatch loop stress.
TEST(VarintTests, dispatch32RandomAllWidths) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  std::vector<uint32_t> data(kNumElements);
  for (int i = 0; i < kNumElements; ++i) {
    int width = folly::Random::rand32(rng) % 5;
    switch (width) {
      case 0:
        data[i] = folly::Random::rand32(rng) % 128;
        break;
      case 1:
        data[i] = 128 + folly::Random::rand32(rng) % 16256;
        break;
      case 2:
        data[i] = 16384 + folly::Random::rand32(rng) % 2080768;
        break;
      case 3:
        data[i] = 2097152 + folly::Random::rand32(rng) % 266338304;
        break;
      case 4:
        data[i] =
            268435456 + folly::Random::rand32(rng) % (UINT32_MAX - 268435456);
        break;
    }
  }
  auto [buf, size] = encodeValues(data);
  verifyBulkDecode(data, buf.get());
}

TEST(VarintTests, dispatch64RandomAllWidths) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  std::vector<uint64_t> data(kNumElements);
  for (int i = 0; i < kNumElements; ++i) {
    int width = folly::Random::rand32(rng) % 5;
    switch (width) {
      case 0:
        data[i] = folly::Random::rand32(rng) % 128;
        break;
      case 1:
        data[i] = 128 + folly::Random::rand32(rng) % 16256;
        break;
      case 2:
        data[i] = 16384 + folly::Random::rand32(rng) % 2080768;
        break;
      case 3:
        data[i] = uint64_t(2097152) + folly::Random::rand64(rng) % 266338304;
        break;
      case 4:
        data[i] = folly::Random::rand64(rng);
        break;
    }
  }
  auto [buf, size] = encodeValues(data);
  verifyBulkDecode(data, buf.get());
}

// Edge case: exactly n=1 for each varint width.
TEST(VarintTests, dispatch32SingleElement) {
  for (uint32_t val : {0u, 127u, 128u, 16383u, 16384u, 2097151u, UINT32_MAX}) {
    std::vector<uint32_t> data = {val};
    auto [buf, size] = encodeValues(data);
    verifyBulkDecode(data, buf.get());
  }
}
