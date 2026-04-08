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

#include "velox/buffer/BufferPool.h"

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"

namespace facebook::velox::test {

class BufferPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = memoryManager_.addLeafPool("BufferPoolTest");
  }

  memory::MemoryManager memoryManager_;
  std::shared_ptr<memory::MemoryPool> pool_;
};

TEST_F(BufferPoolTest, emptyPool) {
  BufferPool bufferPool;
  EXPECT_EQ(bufferPool.size(), 0);
  EXPECT_EQ(bufferPool.get(), nullptr);
  EXPECT_EQ(bufferPool.get(100), nullptr);
}

TEST_F(BufferPoolTest, recycleAndGet) {
  BufferPool bufferPool;
  auto buffer = AlignedBuffer::allocate<char>(1'024, pool_.get());
  auto* rawPtr = buffer.get();
  const auto capacity = buffer->capacity();

  bufferPool.release(std::move(buffer));
  EXPECT_EQ(bufferPool.size(), 1);

  auto retrieved = bufferPool.get();
  EXPECT_NE(retrieved, nullptr);
  EXPECT_EQ(retrieved.get(), rawPtr);
  EXPECT_EQ(retrieved->capacity(), capacity);
  EXPECT_EQ(bufferPool.size(), 0);
}

TEST_F(BufferPoolTest, getWithMinBytes) {
  BufferPool bufferPool;

  auto small = AlignedBuffer::allocate<char>(128, pool_.get());
  auto large = AlignedBuffer::allocate<char>(4'096, pool_.get());
  auto* largePtr = large.get();

  bufferPool.release(std::move(small));
  bufferPool.release(std::move(large));
  EXPECT_EQ(bufferPool.size(), 2);

  // Request minBytes that only the large buffer satisfies.
  auto retrieved = bufferPool.get(1'024);
  EXPECT_NE(retrieved, nullptr);
  EXPECT_EQ(retrieved.get(), largePtr);
  EXPECT_EQ(bufferPool.size(), 1);

  // The small buffer is still there.
  auto remaining = bufferPool.get();
  EXPECT_NE(remaining, nullptr);
  EXPECT_EQ(bufferPool.size(), 0);
}

TEST_F(BufferPoolTest, getWithMinBytesNoMatch) {
  BufferPool bufferPool;

  auto small = AlignedBuffer::allocate<char>(128, pool_.get());
  bufferPool.release(std::move(small));

  auto retrieved = bufferPool.get(1'000'000);
  EXPECT_EQ(retrieved, nullptr);
  EXPECT_EQ(bufferPool.size(), 1);
}

TEST_F(BufferPoolTest, recycleNullptr) {
  BufferPool bufferPool;
  bufferPool.release(nullptr);
  EXPECT_EQ(bufferPool.size(), 0);
}

TEST_F(BufferPoolTest, maxCachedLimit) {
  BufferPool bufferPool;

  // Fill beyond the max cached limit.
  for (size_t i = 0; i < 40; ++i) {
    auto buffer = AlignedBuffer::allocate<char>(64, pool_.get());
    bufferPool.release(std::move(buffer));
  }

  // Should be capped at 32.
  EXPECT_EQ(bufferPool.size(), 32);
}

TEST_F(BufferPoolTest, multipleRecycleAndGet) {
  BufferPool bufferPool;

  for (int i = 0; i < 5; ++i) {
    auto buffer = AlignedBuffer::allocate<char>((i + 1) * 256, pool_.get());
    bufferPool.release(std::move(buffer));
  }
  EXPECT_EQ(bufferPool.size(), 5);

  // Get all back out.
  for (int i = 0; i < 5; ++i) {
    auto retrieved = bufferPool.get();
    EXPECT_NE(retrieved, nullptr);
  }
  EXPECT_EQ(bufferPool.size(), 0);
  EXPECT_EQ(bufferPool.get(), nullptr);
}

} // namespace facebook::velox::test
