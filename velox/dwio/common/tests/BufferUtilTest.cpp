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

#include "velox/dwio/common/BufferUtil.h"

#include <gtest/gtest.h>
#include "velox/common/memory/Memory.h"

namespace facebook::velox::dwio::common {
namespace {

class BufferUtilTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  const std::shared_ptr<memory::MemoryPool> pool_ =
      memory::memoryManager()->addLeafPool();
};

TEST_F(BufferUtilTest, ensureCapacityNullBuffer) {
  BufferPtr buffer;
  ensureCapacity<int8_t>(buffer, 64, pool_.get(), false, true);
  ASSERT_NE(buffer, nullptr);
  ASSERT_GE(buffer->capacity(), 64);
  auto* raw = buffer->as<int8_t>();
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(raw[i], 0);
  }
}

TEST_F(BufferUtilTest, ensureCapacityGrowWithoutPreserveOldData) {
  BufferPtr buffer;
  // Allocate initial small buffer and fill with non-zero bytes.
  ensureCapacity<int8_t>(buffer, 32, pool_.get(), false, true);
  auto* raw = buffer->asMutable<int8_t>();
  std::memset(raw, 0x7F, 32);

  // Grow buffer beyond capacity so a new buffer is allocated.
  // With preserveOldData = false and clearBits = true, all bytes in [0, newCapacity)
  // must be cleared to 0 (including the range [0, oldSize)).
  const size_t newElements = buffer->capacity() * 2;
  ensureCapacity<int8_t>(buffer, newElements, pool_.get(), false, true);
  ASSERT_GE(buffer->capacity(), newElements);

  auto* newRaw = buffer->as<int8_t>();
  for (size_t i = 0; i < newElements; ++i) {
    EXPECT_EQ(newRaw[i], 0) << "Byte at index " << i << " was not cleared";
  }
}

TEST_F(BufferUtilTest, ensureCapacityNoReallocWithoutPreserveOldData) {
  BufferPtr buffer;
  // Allocate a buffer with enough capacity for 128 elements.
  ensureCapacity<int8_t>(buffer, 128, pool_.get(), false, true);
  auto* raw = buffer->asMutable<int8_t>();
  std::memset(raw, 0x7F, 128);

  // Call ensureCapacity with smaller or equal size (no reallocation).
  // With preserveOldData = false and clearBits = true, the entire range
  // [0, newCapacity) must be cleared even when no reallocation occurs.
  ensureCapacity<int8_t>(buffer, 64, pool_.get(), false, true);
  auto* clearedRaw = buffer->as<int8_t>();
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(clearedRaw[i], 0) << "Byte at index " << i << " was not cleared";
  }
}

TEST_F(BufferUtilTest, ensureCapacityGrowWithPreserveOldData) {
  BufferPtr buffer;
  ensureCapacity<int8_t>(buffer, 32, pool_.get(), false, true);
  auto* raw = buffer->asMutable<int8_t>();
  for (size_t i = 0; i < 32; ++i) {
    raw[i] = static_cast<int8_t>(i + 1);
  }

  const size_t newElements = buffer->capacity() * 2;
  ensureCapacity<int8_t>(buffer, newElements, pool_.get(), true, true);
  ASSERT_GE(buffer->capacity(), newElements);

  auto* newRaw = buffer->as<int8_t>();
  for (size_t i = 0; i < 32; ++i) {
    EXPECT_EQ(newRaw[i], static_cast<int8_t>(i + 1));
  }
  for (size_t i = 32; i < newElements; ++i) {
    EXPECT_EQ(newRaw[i], 0);
  }
}

TEST_F(BufferUtilTest, ensureCapacityNoReallocWithPreserveOldData) {
  BufferPtr buffer;
  // Allocate a buffer with capacity for 128 elements, but only initialize 32.
  ensureCapacity<int8_t>(buffer, 128, pool_.get(), false, true);
  buffer->setSize(32);
  auto* raw = buffer->asMutable<int8_t>();
  for (size_t i = 0; i < 32; ++i) {
    raw[i] = static_cast<int8_t>(i + 1);
  }

  // Request 64 elements with preserveOldData = true, clearBits = true.
  // Capacity (>= 128) is sufficient, so no reallocation occurs.
  // [0, 32) should be preserved, and [32, 64) should be cleared.
  ensureCapacity<int8_t>(buffer, 64, pool_.get(), true, true);
  auto* result = buffer->as<int8_t>();
  for (size_t i = 0; i < 32; ++i) {
    EXPECT_EQ(result[i], static_cast<int8_t>(i + 1));
  }
  for (size_t i = 32; i < 64; ++i) {
    EXPECT_EQ(result[i], 0) << "Byte at index " << i << " was not cleared";
  }
}

TEST_F(BufferUtilTest, ensureCapacityDownsizeSharedBufferWithPreserveOldData) {
  BufferPtr buffer;
  ensureCapacity<int8_t>(buffer, 64, pool_.get(), false, true);
  auto* raw = buffer->asMutable<int8_t>();
  for (size_t i = 0; i < 64; ++i) {
    raw[i] = static_cast<int8_t>(i + 1);
  }

  // Make the buffer shared so !buffer->isMutable() triggers reallocation.
  BufferPtr sharedBuffer = buffer;
  ASSERT_FALSE(buffer->unique());

  // Downsize to 16 elements with preserveOldData = true.
  // Reallocation occurs; std::min(oldSize, newCapacity) must prevent copying past
  // the allocated size of the new buffer.
  ensureCapacity<int8_t>(buffer, 16, pool_.get(), true, false);
  ASSERT_TRUE(buffer->unique());
  ASSERT_GE(buffer->capacity(), 16);

  auto* downsizedRaw = buffer->as<int8_t>();
  for (size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(downsizedRaw[i], static_cast<int8_t>(i + 1));
  }
}

TEST_F(BufferUtilTest, ensureCapacityZeroElements) {
  BufferPtr buffer;
  ensureCapacity<int8_t>(buffer, 0, pool_.get(), false, true);
  ASSERT_NE(buffer, nullptr);

  // Calling again with zero elements should be a clean no-op.
  ensureCapacity<int8_t>(buffer, 0, pool_.get(), true, true);
  ensureCapacity<int8_t>(buffer, 0, pool_.get(), false, false);
}

} // namespace
} // namespace facebook::velox::dwio::common
