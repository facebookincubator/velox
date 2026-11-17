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

} // namespace
} // namespace facebook::velox::dwio::common
