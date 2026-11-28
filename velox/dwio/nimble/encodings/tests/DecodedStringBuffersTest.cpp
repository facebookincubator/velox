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

#include "velox/dwio/nimble/encodings/views/DecodedStringBuffers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"

namespace facebook::nimble::test {
namespace {

// The room of a buffer that takes 'bytes' from the pool with its header.
constexpr size_t roomOf(size_t bytes) {
  return bytes - velox::AlignedBuffer::kPaddedSize;
}

class DecodedStringBuffersTest : public ::testing::Test {
 protected:
  velox::memory::MemoryManager manager_;
  const std::shared_ptr<velox::memory::MemoryPool> rootPool_{
      manager_.addRootPool()};
  const std::shared_ptr<velox::memory::MemoryPool> pool_{
      rootPool_->addLeafChild("decoded-string-buffers")};
};

TEST_F(DecodedStringBuffersTest, packsRooms) {
  DecodedStringBuffers buffers{pool_.get()};

  // The first buffer takes 32 KB from the pool, its header included.
  const auto first = buffers.reserve(/*numReservedBytes=*/100);
  EXPECT_EQ(first.size(), roomOf(32 << 10));
  EXPECT_EQ(pool_->usedBytes(), 32 << 10);
  std::fill_n(first.data(), 60, 'a');
  const auto firstBuffer = buffers.finish(/*numUsedBytes=*/60);
  ASSERT_NE(firstBuffer, nullptr);
  EXPECT_EQ(first.data(), firstBuffer->as<char>());
  EXPECT_EQ(firstBuffer->size(), roomOf(32 << 10));

  // The next room starts right after the strings of the first, in the rest of
  // the buffer.
  const auto second = buffers.reserve(/*numReservedBytes=*/200);
  EXPECT_EQ(second.data(), firstBuffer->as<char>() + 60);
  EXPECT_EQ(second.size(), roomOf(32 << 10) - 60);
  std::fill_n(second.data(), 10, 'b');
  EXPECT_EQ(buffers.finish(/*numUsedBytes=*/10), firstBuffer);

  // A reservation larger than the rest of the buffer takes a new buffer, twice
  // as large, and the strings before it stay where they are.
  const auto third = buffers.reserve(/*numReservedBytes=*/roomOf(32 << 10));
  EXPECT_EQ(third.size(), roomOf(64 << 10));
  const auto secondBuffer = buffers.finish(/*numUsedBytes=*/30);
  EXPECT_NE(secondBuffer, firstBuffer);
  EXPECT_EQ(third.data(), secondBuffer->as<char>());
  EXPECT_EQ(
      std::string_view(firstBuffer->as<char>(), 70),
      std::string(60, 'a') + std::string(10, 'b'));

  // The rooms after it come from the new buffer.
  EXPECT_EQ(
      buffers.reserve(/*numReservedBytes=*/10).data(),
      secondBuffer->as<char>() + 30);
}

TEST_F(DecodedStringBuffersTest, rejectsReserveBeforeTheLastRoomFinishes) {
  DecodedStringBuffers buffers{pool_.get()};
  const auto room = buffers.reserve(/*numReservedBytes=*/100);

  // Each reserve() must be followed by a finish() before the next.
  NIMBLE_ASSERT_THROW(
      buffers.reserve(/*numReservedBytes=*/100),
      "A decode must not start before the decode in progress finishes");
  buffers.finish(/*numUsedBytes=*/10);
  NIMBLE_ASSERT_THROW(
      buffers.finish(/*numUsedBytes=*/0), "No decode is in progress to finish");

  // A decode that fails finishes with no strings, so that the next room
  // starts where its room did.
  EXPECT_EQ(buffers.reserve(/*numReservedBytes=*/100).data(), room.data() + 10);
  buffers.finish(/*numUsedBytes=*/0);
  EXPECT_EQ(buffers.reserve(/*numReservedBytes=*/100).data(), room.data() + 10);
}

TEST_F(DecodedStringBuffersTest, doublesBuffersUpTo1MB) {
  // Each room fills its buffer, so that the next one takes a new buffer, which
  // takes twice the bytes of the one before from the pool, up to 1 MB.
  DecodedStringBuffers buffers{pool_.get()};
  std::vector<size_t> sizes;
  std::vector<int64_t> usedBytes;
  for (int i = 0; i < 7; ++i) {
    const auto room = buffers.reserve(/*numReservedBytes=*/1);
    sizes.push_back(room.size());
    usedBytes.push_back(pool_->usedBytes());
    buffers.finish(/*numUsedBytes=*/room.size());
  }
  EXPECT_THAT(
      sizes,
      ::testing::ElementsAre(
          roomOf(32 << 10),
          roomOf(64 << 10),
          roomOf(128 << 10),
          roomOf(256 << 10),
          roomOf(512 << 10),
          roomOf(1 << 20),
          roomOf(1 << 20)));
  EXPECT_THAT(
      usedBytes,
      ::testing::ElementsAre(
          32 << 10,
          64 << 10,
          128 << 10,
          256 << 10,
          512 << 10,
          1 << 20,
          1 << 20));
}

TEST_F(DecodedStringBuffersTest, reservesMoreThanTheNextBuffer) {
  // A reservation larger than the next buffer takes a buffer of its size.
  DecodedStringBuffers buffers{pool_.get()};
  constexpr size_t kSize{100 << 10};
  EXPECT_EQ(buffers.reserve(/*numReservedBytes=*/kSize).size(), kSize);
}

TEST_F(DecodedStringBuffersTest, givesOversizeRoomABufferOfItsOwn) {
  DecodedStringBuffers buffers{pool_.get()};
  const auto first = buffers.reserve(/*numReservedBytes=*/100);
  std::fill_n(first.data(), 60, 'a');
  const auto firstBuffer = buffers.finish(/*numUsedBytes=*/60);

  // A room larger than a buffer of the largest size holds takes a buffer of
  // exactly its size, which no later room shares.
  constexpr size_t kSize{DecodedStringBuffers::kMaxBufferBytes + 1};
  const auto oversize = buffers.reserve(/*numReservedBytes=*/kSize);
  EXPECT_EQ(oversize.size(), kSize);
  std::fill_n(oversize.data(), 10, 'b');
  const auto oversizeBuffer = buffers.finish(/*numUsedBytes=*/10);
  EXPECT_NE(oversizeBuffer, firstBuffer);
  EXPECT_EQ(oversize.data(), oversizeBuffer->as<char>());

  // The next room goes on in the first buffer, right after its strings.
  const auto next = buffers.reserve(/*numReservedBytes=*/100);
  EXPECT_EQ(next.data(), firstBuffer->as<char>() + 60);
  EXPECT_EQ(buffers.finish(/*numUsedBytes=*/0), firstBuffer);
  EXPECT_EQ(
      std::string_view(firstBuffer->as<char>(), 60), std::string(60, 'a'));
  EXPECT_EQ(
      std::string_view(oversizeBuffer->as<char>(), 10), std::string(10, 'b'));
}

TEST_F(DecodedStringBuffersTest, isNeitherCopyableNorMovable) {
  // Views keep the address of an instance, and a copy would give out the same
  // bytes as the original.
  static_assert(!std::is_copy_constructible_v<DecodedStringBuffers>);
  static_assert(!std::is_copy_assignable_v<DecodedStringBuffers>);
  static_assert(!std::is_move_constructible_v<DecodedStringBuffers>);
  static_assert(!std::is_move_assignable_v<DecodedStringBuffers>);
}

TEST_F(DecodedStringBuffersTest, clearStartsOverWith32KB) {
  DecodedStringBuffers buffers{pool_.get()};
  const auto room = buffers.reserve(/*numReservedBytes=*/100);
  std::fill_n(room.data(), 80, 'a');
  const auto firstBuffer = buffers.finish(/*numUsedBytes=*/80);
  buffers.reserve(/*numReservedBytes=*/roomOf(32 << 10));
  buffers.finish(/*numUsedBytes=*/0);

  // clear() drops the last buffer, so that the next room takes a new buffer of
  // 32 KB. The strings of the rooms before stay valid.
  buffers.clear();
  EXPECT_EQ(
      std::string_view(firstBuffer->as<char>(), 80), std::string(80, 'a'));
  const auto next = buffers.reserve(/*numReservedBytes=*/100);
  EXPECT_NE(next.data(), firstBuffer->as<char>() + 80);
  EXPECT_EQ(next.size(), roomOf(32 << 10));
}

} // namespace
} // namespace facebook::nimble::test
