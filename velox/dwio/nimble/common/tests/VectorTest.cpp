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
#include "velox/dwio/nimble/common/Vector.h"
#include <gtest/gtest.h>
#include <numeric>
#include "velox/buffer/BufferPool.h"
#include "velox/common/memory/Memory.h"

DECLARE_bool(velox_enable_memory_usage_track_in_default_memory_pool);

using namespace ::facebook;

namespace {

// Verifies that a Vector is in a clean empty state with no buffer.
template <typename T>
void checkVectorEmpty(const nimble::Vector<T>& v) {
  EXPECT_EQ(v.size(), 0);
  EXPECT_EQ(v.capacity(), 0);
  EXPECT_EQ(v.testingBuffer(), nullptr);
}

// Verifies that a Vector holds the expected buffer with the given capacity
// and zero logical size.
template <typename T>
void checkVectorBuffer(
    const nimble::Vector<T>& v,
    const velox::Buffer* expectedBuffer,
    uint64_t expectedCapacity) {
  EXPECT_EQ(v.size(), 0);
  EXPECT_EQ(v.capacity(), expectedCapacity);
  EXPECT_EQ(v.testingBuffer().get(), expectedBuffer);
}

} // namespace

class VectorTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    FLAGS_velox_enable_memory_usage_track_in_default_memory_pool = true;
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    rootPool_ = velox::memory::memoryManager()->addRootPool("VectorTest");
    pool_ = rootPool_->addLeafChild("leaf");
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

TEST_F(VectorTest, fromRange) {
  std::vector<int32_t> source{4, 5, 6};
  nimble::Vector<int32_t> v1(pool_.get(), source.begin(), source.end());
  EXPECT_EQ(3, v1.size());
  EXPECT_EQ(4, v1[0]);
  EXPECT_EQ(5, v1[1]);
  EXPECT_EQ(6, v1[2]);
}

TEST_F(VectorTest, equalOp1) {
  nimble::Vector<int32_t> v1(pool_.get());
  v1.push_back(1);
  v1.emplace_back(2);
  v1.push_back(3);

  nimble::Vector<int32_t> v2(pool_.get());
  v2.push_back(4);
  v2.emplace_back(5);

  EXPECT_EQ(3, v1.size());
  EXPECT_EQ(1, v1[0]);
  EXPECT_EQ(2, v1[1]);
  EXPECT_EQ(3, v1[2]);
  EXPECT_EQ(2, v2.size());
  EXPECT_EQ(4, v2[0]);
  EXPECT_EQ(5, v2[1]);

  v1 = v2;
  EXPECT_EQ(2, v1.size());
  EXPECT_EQ(4, v1[0]);
  EXPECT_EQ(5, v1[1]);
  EXPECT_EQ(2, v2.size());
  EXPECT_EQ(4, v2[0]);
  EXPECT_EQ(5, v2[1]);
}

TEST_F(VectorTest, explicitMoveEqualOp) {
  nimble::Vector<int32_t> v1(pool_.get());
  v1.push_back(1);
  v1.emplace_back(2);
  v1.push_back(3);

  nimble::Vector<int32_t> v2(pool_.get());
  v2.push_back(4);
  v2.emplace_back(5);

  EXPECT_EQ(3, v1.size());
  ASSERT_FALSE(v1.empty());
  EXPECT_EQ(1, v1[0]);
  EXPECT_EQ(2, v1[1]);
  EXPECT_EQ(3, v1[2]);
  EXPECT_EQ(2, v2.size());
  ASSERT_FALSE(v2.empty());
  EXPECT_EQ(4, v2[0]);
  EXPECT_EQ(5, v2[1]);

  v1 = std::move(v2);
  EXPECT_EQ(2, v1.size());
  EXPECT_EQ(4, v1[0]);
  EXPECT_EQ(5, v1[1]);
  // @lint-ignore CLANGTIDY bugprone-use-after-move
  EXPECT_EQ(0, v2.size());
  ASSERT_TRUE(v2.empty());
}

TEST_F(VectorTest, moveEqualOp1) {
  nimble::Vector<int32_t> v1(pool_.get());
  v1.push_back(1);
  v1.emplace_back(2);
  v1.push_back(3);
  EXPECT_EQ(3, v1.size());
  EXPECT_EQ(1, v1[0]);
  EXPECT_EQ(2, v1[1]);
  EXPECT_EQ(3, v1[2]);
  v1 = nimble::Vector(pool_.get(), {4, 5});
  EXPECT_EQ(2, v1.size());
  EXPECT_EQ(4, v1[0]);
  EXPECT_EQ(5, v1[1]);
}

TEST_F(VectorTest, copyCtr) {
  nimble::Vector<int32_t> v2(pool_.get());
  v2.push_back(3);
  v2.emplace_back(4);
  EXPECT_EQ(2, v2.size());
  EXPECT_EQ(3, v2[0]);
  EXPECT_EQ(4, v2[1]);
  nimble::Vector<int32_t> v1(v2);
  EXPECT_EQ(2, v1.size());
  EXPECT_EQ(3, v1[0]);
  EXPECT_EQ(4, v1[1]);

  // make sure they do not share buffer
  v1[0] = 1;
  v1[1] = 2;
  EXPECT_EQ(2, v1.size());
  EXPECT_EQ(1, v1[0]);
  EXPECT_EQ(2, v1[1]);
  EXPECT_EQ(2, v2.size());
  EXPECT_EQ(3, v2[0]);
  EXPECT_EQ(4, v2[1]);
}

TEST_F(VectorTest, boolInitializerList) {
  nimble::Vector<bool> v1(pool_.get(), {true, false, true});
  EXPECT_EQ(3, v1.size());
  EXPECT_EQ(true, v1[0]);
  EXPECT_EQ(false, v1[1]);
  EXPECT_EQ(true, v1[2]);
}

TEST_F(VectorTest, boolEqualOp1) {
  nimble::Vector<bool> v1(pool_.get());
  v1.push_back(false);
  v1.emplace_back(true);
  v1.push_back(true);

  EXPECT_EQ(3, v1.size());
  EXPECT_EQ(false, v1[0]);
  EXPECT_EQ(true, v1[1]);
  EXPECT_EQ(true, v1[2]);

  nimble::Vector<bool> v2(pool_.get());
  v2.push_back(true);
  v2.emplace_back(false);

  EXPECT_EQ(2, v2.size());
  EXPECT_EQ(true, v2[0]);
  EXPECT_EQ(false, v2[1]);

  v1 = v2;
  EXPECT_EQ(2, v1.size());
  EXPECT_EQ(true, v1[0]);
  EXPECT_EQ(false, v1[1]);
  EXPECT_EQ(2, v2.size());
  EXPECT_EQ(true, v2[0]);
  EXPECT_EQ(false, v2[1]);
}

TEST_F(VectorTest, boolMoveEqualOp1) {
  nimble::Vector<bool> v1(pool_.get());
  v1.push_back(true);
  v1.emplace_back(false);
  v1.push_back(false);

  EXPECT_EQ(3, v1.size());
  EXPECT_EQ(true, v1[0]);
  EXPECT_EQ(false, v1[1]);
  EXPECT_EQ(false, v1[2]);

  v1 = nimble::Vector(pool_.get(), {false, true});
  EXPECT_EQ(2, v1.size());
  EXPECT_EQ(false, v1[0]);
  EXPECT_EQ(true, v1[1]);
}

TEST_F(VectorTest, boolCopyCtr) {
  nimble::Vector<bool> v2(pool_.get());
  v2.push_back(true);
  v2.emplace_back(false);
  EXPECT_EQ(2, v2.size());
  EXPECT_EQ(true, v2[0]);
  EXPECT_EQ(false, v2[1]);
  nimble::Vector<bool> v1(v2);
  EXPECT_EQ(2, v1.size());
  EXPECT_EQ(true, v1[0]);
  EXPECT_EQ(false, v1[1]);
  EXPECT_EQ(2, v2.size());
  EXPECT_EQ(true, v2[0]);
  EXPECT_EQ(false, v2[1]);
}

TEST_F(VectorTest, scopedVectorReleasesBufferToPool) {
  velox::BufferPool bufferPool{velox::BufferPool::kDefaultCapacity};

  {
    nimble::ScopedVector<uint32_t> scratch{
        /*size=*/4, pool_.get(), &bufferPool};
    ASSERT_EQ(scratch.size(), 4);
    scratch[0] = 123;
    EXPECT_EQ(scratch[0], 123);
    EXPECT_EQ(bufferPool.size(), 0);
  }
  EXPECT_EQ(bufferPool.size(), 1);

  {
    nimble::ScopedVector<uint32_t> scratch{
        /*size=*/2, pool_.get(), &bufferPool};
    EXPECT_EQ(scratch.size(), 2);
    EXPECT_EQ(bufferPool.size(), 0);
  }
  EXPECT_EQ(bufferPool.size(), 1);
}

TEST_F(VectorTest, scopedVectorWorksWithoutBufferPool) {
  nimble::ScopedVector<bool> scratch{
      /*size=*/3, pool_.get(), /*bufferPool=*/nullptr};
  ASSERT_EQ(scratch.size(), 3);
  scratch[0] = true;
  scratch[1] = false;
  EXPECT_TRUE(scratch[0]);
  EXPECT_FALSE(scratch[1]);
}

TEST_F(VectorTest, scopedVectorOperations) {
  nimble::ScopedVector<uint32_t> scratch{
      /*size=*/0, pool_.get(), /*bufferPool=*/nullptr};
  EXPECT_TRUE(scratch.empty());

  scratch.reserve(3);
  EXPECT_GE(scratch.capacity(), 3);
  scratch.push_back(1);
  scratch.emplace_back(2);
  scratch.resize(3);
  scratch[2] = 3;

  EXPECT_EQ(scratch.size(), 3);

  nimble::Vector<uint32_t>& values = scratch;
  EXPECT_EQ(values.size(), scratch.size());
  EXPECT_EQ(values.data(), scratch.data());
  EXPECT_EQ(values[0], 1);

  EXPECT_EQ(scratch->capacity(), scratch.capacity());
  EXPECT_EQ((*scratch)[1], 2);
  EXPECT_EQ(scratch[2], 3);
  EXPECT_EQ(std::accumulate(scratch.begin(), scratch.end(), uint32_t{0}), 6);

  const auto& constScratch = scratch;
  const nimble::Vector<uint32_t>& constValues = constScratch;
  EXPECT_EQ(constValues.data(), constScratch.data());
  EXPECT_EQ(constValues[0], 1);
  EXPECT_EQ(constScratch->size(), 3);
  EXPECT_EQ((*constScratch)[1], 2);
  EXPECT_EQ(constScratch[2], 3);
  EXPECT_EQ(
      std::accumulate(constScratch.begin(), constScratch.end(), uint32_t{0}),
      6);
}

TEST_F(VectorTest, memoryCleanup) {
  EXPECT_EQ(0, pool_->usedBytes());
  {
    nimble::Vector<int32_t> v(pool_.get());
    EXPECT_EQ(0, pool_->usedBytes());
    v.resize(1000, 10);
    EXPECT_NE(0, pool_->usedBytes());
  }
  EXPECT_EQ(0, pool_->usedBytes());
  {
    nimble::Vector<int32_t> v(pool_.get());
    EXPECT_EQ(0, pool_->usedBytes());
    v.resize(1000, 10);
    EXPECT_NE(0, pool_->usedBytes());

    auto vCopy(v);
  }
  EXPECT_EQ(0, pool_->usedBytes());
  {
    nimble::Vector<int32_t> v(pool_.get());
    EXPECT_EQ(0, pool_->usedBytes());
    v.resize(1000, 10);
    EXPECT_NE(0, pool_->usedBytes());

    auto vCopy(std::move(v));
  }
  EXPECT_EQ(0, pool_->usedBytes());
}

TEST_F(VectorTest, reserveActualSize) {
  EXPECT_EQ(0, pool_->usedBytes());

  // There is no good way to assert the exact expected size because of padding
  // logic in the AlignedBuffer::allocate.
  // We know that without the exactSize flag being passed into the
  // AlignedBuffer, it would allocate 12,582,912 bytes for any requested size in
  // the range [8,388,609 - 12,582,912]. What we can do is to pick some value
  // in the middle of this range and assert that it's roughly what we expect it
  // to be, e.g allocatedSize is in [X, X+1MB].
  {
    // 1 byte type
    nimble::Vector<int8_t> v(pool_.get());
    EXPECT_EQ(0, pool_->usedBytes());

    const size_t lowerBound = 9 * 1024 * 1024;
    const size_t upperBound = lowerBound + 1024 * 1024;
    v.reserve(lowerBound);
    EXPECT_GE(pool_->usedBytes(), lowerBound);
    EXPECT_LE(pool_->usedBytes(), upperBound);
  }

  {
    // 4 byte type
    nimble::Vector<int32_t> v(pool_.get());
    EXPECT_EQ(0, pool_->usedBytes());
    const size_t lowerBound = 9 * 1024 * 1024;
    const size_t upperBound = lowerBound + 1024 * 1024;
    const uint64_t valueCount = lowerBound / sizeof(int32_t);
    v.reserve(valueCount);
    EXPECT_GE(pool_->usedBytes(), lowerBound);
    EXPECT_LE(pool_->usedBytes(), upperBound);
  }
}

TEST_F(VectorTest, releaseBuffer) {
  nimble::Vector<int32_t> v(pool_.get());
  v.push_back(10);
  v.push_back(20);
  v.push_back(30);
  EXPECT_EQ(v.size(), 3);

  auto buf = v.releaseBuffer();
  EXPECT_NE(buf, nullptr);
  EXPECT_GE(buf->capacity(), 3 * sizeof(int32_t));

  // Vector is fully reset after release.
  checkVectorEmpty(v);

  // Releasing again returns nullptr.
  auto buf2 = v.releaseBuffer();
  EXPECT_EQ(buf2, nullptr);
}

TEST_F(VectorTest, constructFromBuffer) {
  auto buf = velox::AlignedBuffer::allocate<int32_t>(100, pool_.get());
  const auto expectedCapacity = buf->capacity() / sizeof(int32_t);
  auto* rawBuf = buf.get();

  nimble::Vector<int32_t> v(std::move(buf));

  checkVectorBuffer(v, rawBuf, expectedCapacity);
  EXPECT_EQ(v.pool(), pool_.get());

  // Can use the vector normally.
  v.push_back(42);
  EXPECT_EQ(v.size(), 1);
  EXPECT_EQ(v[0], 42);
}

TEST_F(VectorTest, releaseAndConstructRoundTrip) {
  nimble::Vector<int32_t> v1(pool_.get());
  for (int i = 0; i < 100; ++i) {
    v1.push_back(i);
  }
  const auto originalCapacity = v1.capacity();

  // Release from v1, construct v2 from the buffer.
  auto buf = v1.releaseBuffer();
  checkVectorEmpty(v1);

  auto* rawBuf = buf.get();
  nimble::Vector<int32_t> v2(std::move(buf));

  checkVectorBuffer(v2, rawBuf, originalCapacity);
  EXPECT_EQ(v2.pool(), pool_.get());

  // v2 can be used normally.
  v2.resize(50, 7);
  EXPECT_EQ(v2.size(), 50);
  EXPECT_EQ(v2[0], 7);
}
