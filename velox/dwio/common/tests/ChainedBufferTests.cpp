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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/dwio/common/ChainedBuffer.h"

using namespace ::testing;

namespace facebook::velox::dwio::common {

class ChainedBufferTests : public Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
};

TEST_F(ChainedBufferTests, testCreate) {
  ChainedBuffer<int32_t> buf{*pool_, 128, 1024};
  ASSERT_EQ(buf.capacity(), 128);
  ASSERT_EQ(buf.size(), 0);
  ASSERT_EQ(buf.pageCount(), 1);
  ChainedBuffer<int32_t> buf2{*pool_, 256, 1024};
  ASSERT_EQ(buf2.capacity(), 256);
  ASSERT_EQ(buf2.pageCount(), 1);
  ASSERT_EQ(buf2.size(), 0);
  ChainedBuffer<int32_t> buf3{*pool_, 257, 1024};
  ASSERT_EQ(buf3.capacity(), 512);
  ASSERT_EQ(buf3.pageCount(), 2);
  ASSERT_EQ(buf3.size(), 0);

  VELOX_ASSERT_THROW(
      (ChainedBuffer<int32_t>{*pool_, 256, 257}),
      "(2 vs. 1) must be power of 2: 257");

  ChainedBuffer<int32_t> buf0{*pool_, 0, 1024};
  ASSERT_EQ(buf0.capacity(), 0);
  ASSERT_EQ(buf0.pageCount(), 0);
  ASSERT_EQ(buf0.size(), 0);
}

TEST_F(ChainedBufferTests, testReserve) {
  for (const uint32_t initialCapacityBytes : {0, 16}) {
    SCOPED_TRACE(
        fmt::format(
            "initialCapacityBytes ", succinctBytes(initialCapacityBytes)));
    ChainedBuffer<int32_t> buf{*pool_, initialCapacityBytes, 1024};
    ASSERT_EQ(buf.capacity(), initialCapacityBytes);
    ASSERT_EQ(buf.size(), 0);
    buf.reserve(16);
    buf.reserve(17);
    ASSERT_EQ(buf.capacity(), 32);
    ASSERT_EQ(buf.pageCount(), 1);
    buf.reserve(112);
    ASSERT_EQ(buf.capacity(), 128);
    ASSERT_EQ(buf.pageCount(), 1);
    buf.reserve(257);
    ASSERT_EQ(buf.capacity(), 512);
    ASSERT_EQ(buf.pageCount(), 2);
    buf.reserve(1025);
    ASSERT_EQ(buf.capacity(), 1024 + 256);
    ASSERT_EQ(buf.pageCount(), 5);
  }
}

TEST_F(ChainedBufferTests, testAppend) {
  ChainedBuffer<int32_t> buf{*pool_, 16, 64};
  for (size_t i = 0; i < 16; ++i) {
    buf.unsafeAppend(i);
    ASSERT_EQ(buf.capacity(), 16);
    ASSERT_EQ(buf.size(), i + 1);
    ASSERT_EQ(buf.pageCount(), 1);
  }
  buf.reserve(32);
  for (size_t i = 0; i < 16; ++i) {
    buf.unsafeAppend(i + 16);
    ASSERT_EQ(buf.capacity(), 32);
    ASSERT_EQ(buf.size(), i + 17);
    ASSERT_EQ(buf.pageCount(), 2);
  }
  for (size_t i = 0; i < 32; ++i) {
    ASSERT_EQ(buf[i], i);
  }
  buf.append(100);
  ASSERT_EQ(buf.capacity(), 48);
  ASSERT_EQ(buf.pageCount(), 3);
  ASSERT_EQ(buf[buf.size() - 1], 100);
}

TEST_F(ChainedBufferTests, testClear) {
  ChainedBuffer<int32_t> buf{*pool_, 128, 1024};
  buf.clear();
  ASSERT_EQ(buf.capacity(), 128);
  ASSERT_EQ(buf.size(), 0);
  ASSERT_EQ(buf.pageCount(), 1);

  ChainedBuffer<int32_t> buf2{*pool_, 1024, 1024};
  buf2.clear(false);
  ASSERT_EQ(buf2.capacity(), 256);
  ASSERT_EQ(buf2.size(), 0);
  ASSERT_EQ(buf2.pageCount(), 1);
}

TEST_F(ChainedBufferTests, testApplyRange) {
  std::vector<std::tuple<uint64_t, uint64_t, int32_t>> result;
  auto fn = [&](auto ptr, auto begin, auto end) {
    result.push_back({begin, end, *ptr});
  };

  ChainedBuffer<int32_t> buf{*pool_, 64, 64};
  for (size_t i = 0; i < 64 / 16; ++i) {
    for (size_t j = 0; j < 16; ++j) {
      buf.unsafeAppend(i);
    }
  }
  VELOX_ASSERT_THROW(buf.applyRange(2, 1, fn), "(2 vs. 1)");
  VELOX_ASSERT_THROW(buf.applyRange(1, 65, fn), "(65 vs. 64)");

  result.clear();
  buf.applyRange(1, 5, fn);
  ASSERT_THAT(
      result, ElementsAre(std::tuple<uint64_t, uint64_t, int32_t>{1, 5, 0}));

  result.clear();
  buf.applyRange(3, 16, fn);
  ASSERT_THAT(
      result, ElementsAre(std::tuple<uint64_t, uint64_t, int32_t>{3, 16, 0}));

  result.clear();
  buf.applyRange(1, 17, fn);
  ASSERT_THAT(
      result,
      ElementsAre(
          std::tuple<uint64_t, uint64_t, int32_t>{1, 16, 0},
          std::tuple<uint64_t, uint64_t, int32_t>{0, 1, 1}));

  result.clear();
  buf.applyRange(1, 37, fn);
  ASSERT_THAT(
      result,
      ElementsAre(
          std::tuple<uint64_t, uint64_t, int32_t>{1, 16, 0},
          std::tuple<uint64_t, uint64_t, int32_t>{0, 16, 1},
          std::tuple<uint64_t, uint64_t, int32_t>{0, 5, 2}));

  result.clear();
  buf.applyRange(1, 64, fn);
  ASSERT_THAT(
      result,
      ElementsAre(
          std::tuple<uint64_t, uint64_t, int32_t>{1, 16, 0},
          std::tuple<uint64_t, uint64_t, int32_t>{0, 16, 1},
          std::tuple<uint64_t, uint64_t, int32_t>{0, 16, 2},
          std::tuple<uint64_t, uint64_t, int32_t>{0, 16, 3}));
}

TEST_F(ChainedBufferTests, testPageAccess) {
  ChainedBuffer<int32_t> buf{*pool_, 1024, 1024};
  for (int32_t i = 0; i < 1024; ++i) {
    buf.append(i);
  }
  ASSERT_EQ(buf.pageCount(), 4);
  EXPECT_EQ(buf[0], 0);
  EXPECT_EQ(buf[255], 255);
  EXPECT_EQ(buf[256], 256);
  EXPECT_EQ(buf[1023], 1023);

  ChainedBuffer<int64_t> buf2{*pool_, 1024, 1024};
  for (int64_t i = 0; i < 1024; ++i) {
    buf2.append(i);
  }
  ASSERT_EQ(buf2.pageCount(), 8);
  EXPECT_EQ(buf2[0], 0);
  EXPECT_EQ(buf2[127], 127);
  EXPECT_EQ(buf2[128], 128);
  EXPECT_EQ(buf2[1023], 1023);

  ChainedBuffer<int8_t> buf3{*pool_, 1024, 1024};
  for (int32_t i = 0; i <= 1024; ++i) {
    buf3.append(static_cast<int8_t>(i));
  }
  ASSERT_EQ(buf3.pageCount(), 2);
  buf3[0] = 11;
  buf3[1023] = 22;
  buf3[1024] = 33;
  EXPECT_EQ(buf3[0], 11);
  EXPECT_EQ(buf3[1023], 22);
  EXPECT_EQ(buf3[1024], 33);
}

TEST_F(ChainedBufferTests, testBitCount) {
  ASSERT_EQ(detail::bitCount(0), 0);
  ASSERT_EQ(detail::bitCount(1), 1);
  ASSERT_EQ(detail::bitCount(4), 1);
  ASSERT_EQ(detail::bitCount(15), 4);
}

TEST_F(ChainedBufferTests, testTrailingZeros) {
  ASSERT_EQ(detail::trailingZeros(1), 0);
  ASSERT_EQ(detail::trailingZeros(12), 2);
  ASSERT_EQ(detail::trailingZeros(1u << 31), 31);
  VELOX_ASSERT_THROW(detail::trailingZeros(0), "(0 vs. 0)");
}

TEST_F(ChainedBufferTests, testClearAll) {
  for (const uint32_t initialCapacityBytes : {0, 128}) {
    SCOPED_TRACE(
        fmt::format(
            "initialCapacityBytes ", succinctBytes(initialCapacityBytes)));
    ChainedBuffer<int32_t> buf{*pool_, initialCapacityBytes, 1024};
    ASSERT_EQ(buf.capacity(), initialCapacityBytes);
    ASSERT_EQ(buf.size(), 0);

    buf.clear(false);
    ASSERT_EQ(buf.capacity(), initialCapacityBytes);
    ASSERT_EQ(buf.size(), 0);
    ASSERT_EQ(buf.pageCount(), initialCapacityBytes == 0 ? 0 : 1);
    buf.clear(true);
    ASSERT_EQ(buf.capacity(), 0);
    ASSERT_EQ(buf.size(), 0);
    ASSERT_EQ(buf.pageCount(), 0);

    buf.reserve(256);
    ASSERT_EQ(buf.capacity(), 256);
    ASSERT_EQ(buf.size(), 0);

    buf.unsafeAppend(32);
    ASSERT_EQ(buf.size(), 1);
    for (int i = 1; i < 256; ++i) {
      buf.unsafeAppend(32);
    }
    ASSERT_EQ(buf.capacity(), 256);
    ASSERT_EQ(buf.size(), 256);
    ASSERT_EQ(buf.pageCount(), 1);
    buf.append(32);
    ASSERT_EQ(buf.capacity(), 512);
    ASSERT_EQ(buf.size(), 257);
    ASSERT_EQ(buf.pageCount(), 2);

    buf.clear(true);
    ASSERT_EQ(buf.capacity(), 0);
    ASSERT_EQ(buf.size(), 0);
    ASSERT_EQ(buf.pageCount(), 0);

    for (int i = 0; i <= 256; ++i) {
      buf.append(32);
    }
    ASSERT_EQ(buf.capacity(), 512);
    ASSERT_EQ(buf.size(), 257);
    ASSERT_EQ(buf.pageCount(), 2);
    buf.clear(true);

    ASSERT_EQ(buf.capacity(), 0);
    ASSERT_EQ(buf.size(), 0);
    ASSERT_EQ(buf.pageCount(), 0);

    for (int i = 0; i <= 2048; ++i) {
      buf.append(32);
    }
    ASSERT_EQ(buf.capacity(), 2304);
    ASSERT_EQ(buf.size(), 2049);
    ASSERT_EQ(buf.pageCount(), 9);

    buf.clear(true);
    ASSERT_EQ(buf.capacity(), 0);
    ASSERT_EQ(buf.size(), 0);
    ASSERT_EQ(buf.pageCount(), 0);

    for (int i = 0; i <= 2048; ++i) {
      buf.append(32);
    }
    ASSERT_EQ(buf.capacity(), 2304);
    ASSERT_EQ(buf.size(), 2049);
    ASSERT_EQ(buf.pageCount(), 9);
  }
}

} // namespace facebook::velox::dwio::common
