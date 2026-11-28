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

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <folly/String.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingV2Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/deprecated/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

namespace facebook::nimble {
namespace {

// Pins the bytes that BlockBitPacking (V1) and BlockBitPackingV2 write and
// slice, which share one implementation.
class BlockBitPackingWireFormatTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  // Blocks of 16 rows: constant, narrow, one wide value forcing a raw block,
  // and a short last block.
  static std::vector<uint32_t> values() {
    std::vector<uint32_t> values;
    values.reserve(16 * 3 + 5);
    for (uint32_t i = 0; i < 16; ++i) {
      values.push_back(1'000);
    }
    for (uint32_t i = 0; i < 16; ++i) {
      values.push_back(500 + (i * 7) % 13);
    }
    for (uint32_t i = 0; i < 16; ++i) {
      values.push_back(i == 3 ? std::numeric_limits<uint32_t>::max() : i);
    }
    for (uint32_t i = 0; i < 5; ++i) {
      values.push_back(42 + i);
    }
    return values;
  }

  // Hex of the encoding of values(), then of its slices over kRanges.
  template <typename EncodingT>
  void expectBytes(const std::vector<std::string>& expected) {
    Vector<uint32_t> input{pool_.get()};
    for (const auto value : values()) {
      input.push_back(value);
    }
    const Encoding::Options options{.blockBitPackingBlockSize = 16};
    const auto encoded = test::Encoder<EncodingT>::encode(
        *buffer_, input, CompressionType::Uncompressed, options);
    ASSERT_EQ(expected.size(), kRanges.size() + 1);
    EXPECT_EQ(folly::hexlify(encoded), expected[0]);
    Buffer sliceBuffer{*pool_};
    EXPECT_EQ(
        EncodingFactory::slice(encoded, 0, input.size(), sliceBuffer, options),
        encoded);
    for (size_t i = 0; i < kRanges.size(); ++i) {
      const auto [offset, length] = kRanges[i];
      SCOPED_TRACE(fmt::format("offset={} length={}", offset, length));
      EXPECT_EQ(
          folly::hexlify(
              EncodingFactory::slice(
                  encoded, offset, length, sliceBuffer, options)),
          expected[i + 1]);
    }
  }

  // Mid-block ranges come back wrapped in a Slice encoding; block-aligned ones
  // stay a BlockBitPacking.
  static constexpr std::array<std::pair<uint32_t, uint32_t>, 5> kRanges{{
      {5, 20},
      {16, 16},
      {20, 30},
      {48, 5},
      {47, 2},
  }};

  std::shared_ptr<velox::memory::MemoryPool> pool_{
      velox::memory::memoryManager()->addLeafPool()};
  std::unique_ptr<Buffer> buffer_{std::make_unique<Buffer>(*pool_)};
};

TEST_F(BlockBitPackingWireFormatTest, pinsV1Bytes) {
  expectBytes<deprecated::BlockBitPackingEncoding<uint32_t>>({
      "0f06350000000010041700060400000000e8030000f4010000000000002a0000"
      "000b000204000000000004ff0317000604000000000000000007000000160000"
      "00560000001000000000000000708192a3b4c506170000000000000000000000"
      "0100000002000000ffffffff0400000005000000060000000700000008000000"
      "090000000a0000000b0000000c0000000d0000000e0000000f00000088460000"
      "0000000000",
      "17061400000005000000000f06200000000010020f00060200000000e8030000"
      "f4010000090002020000000000041a1706020000000000000000000602000000"
      "0000000000070000001000000000000000708192a3b4c5061700000000000000",
      "0f06100000000010010b00060100000000f40100000800020100000000041617"
      "06010000000000000000000601000000000000000010708192a3b4c506170000"
      "0000000000",
      "17061e00000004000000000f06250000000010031300060300000000f4010000"
      "000000002a0000000a0002030000000004ff031e170603000000000000000000"
      "060300000000000000000f0000004f00000010708192a3b4c506170000000000"
      "0000000000000100000002000000ffffffff0400000005000000060000000700"
      "000008000000090000000a0000000b0000000c0000000d0000000e0000000f00"
      "0000884600000000000000",
      "0f06050000000010010b000601000000002a0000000800020100000000031617"
      "06010000000000000000000601000000000000000005884600000000000000",
      "1706020000000f000000000f06150000000010020f0006020000000000000000"
      "2a0000000900020200000000ff031a1706020000000000000000000602000000"
      "00000000004000000010000000000100000002000000ffffffff040000000500"
      "0000060000000700000008000000090000000a0000000b0000000c0000000d00"
      "00000e0000000f000000884600000000000000",
  });
}

TEST_F(BlockBitPackingWireFormatTest, pinsV2Bytes) {
  expectBytes<BlockBitPackingV2Encoding<uint32_t>>({
      "1c06350000000010041700060400000000e8030000f4010000000000002a0000"
      "000b000204000000000004ff0317000604000000000000000007000000160000"
      "005600000000000000000000708192a3b4c50617000000000000000000000001"
      "00000002000000ffffffff040000000500000006000000070000000800000009"
      "0000000a0000000b0000000c0000000d0000000e0000000f0000008846000000"
      "00000000",
      "17061400000005000000001c06200000000010020f00060200000000e8030000"
      "f4010000090002020000000000041a1706020000000000000000000602000000"
      "00000000000700000000000000000000708192a3b4c5061700000000000000",
      "1c06100000000010010b00060100000000f40100000800020100000000041617"
      "060100000000000000000006010000000000000000708192a3b4c50617000000"
      "00000000",
      "17061e00000004000000001c06250000000010031300060300000000f4010000"
      "000000002a0000000a0002030000000004ff031e170603000000000000000000"
      "060300000000000000000f0000004f000000708192a3b4c50617000000000000"
      "00000000000100000002000000ffffffff040000000500000006000000070000"
      "0008000000090000000a0000000b0000000c0000000d0000000e0000000f0000"
      "00884600000000000000",
      "1c06050000000010010b000601000000002a0000000800020100000000031617"
      "060100000000000000000006010000000000000000884600000000000000",
      "1706020000000f000000001c06150000000010020f0006020000000000000000"
      "2a0000000900020200000000ff031a1706020000000000000000000602000000"
      "000000000040000000000000000100000002000000ffffffff04000000050000"
      "00060000000700000008000000090000000a0000000b0000000c0000000d0000"
      "000e0000000f000000884600000000000000",
  });
}

} // namespace
} // namespace facebook::nimble
