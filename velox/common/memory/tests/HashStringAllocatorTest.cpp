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
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/common/base/tests/GTestUtils.h"

#include <folly/Random.h>

#include <folly/container/F14Map.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

namespace facebook::velox {
namespace {

using HSA = HashStringAllocator;

struct Multipart {
  HSA::Position start;
  HSA::Position current;
  uint64_t size = 0;
  std::string reference;
};

class HashStringAllocatorTest : public testing::Test {
 protected:
  void SetUp() override {
    pool_ = memory::addDefaultLeafMemoryPool();
    allocator_ = std::make_unique<HashStringAllocator>(pool_.get());
    rng_.seed(1);
  }

  HSA::Header* allocate(int32_t numBytes) {
    auto result = allocator_->allocate(numBytes);
    EXPECT_GE(result->size(), numBytes);
    initializeContents(result);
    return result;
  }

  void initializeContents(HSA::Header* header) {
    auto sequence = ++sequence_;
    int32_t numWords = header->size() / sizeof(void*);
    void** ptr = reinterpret_cast<void**>(header->begin());
    ptr[0] = reinterpret_cast<void*>(sequence);
    for (int32_t offset = 1; offset < numWords; offset++) {
      ptr[offset] = ptr + offset + sequence;
    }
  }

  static void checkMultipart(const Multipart& data) {
    std::string storage;
    auto contiguous = HSA::contiguousString(
        StringView(data.start.position, data.reference.size()), storage);
    EXPECT_EQ(StringView(data.reference), contiguous);
  }

  void checkAndFree(Multipart& data) {
    checkMultipart(data);
    data.reference.clear();
    allocator_->free(data.start.header);
    data.start = HSA::Position::null();
  }

  uint32_t rand32() {
    return folly::Random::rand32(rng_);
  }

  std::string randomString() {
    std::string result;
    result.resize(
        20 + (rand32() % 10 > 8 ? rand32() % 200 : 1000 + rand32() % 1000));
    for (auto i = 0; i < result.size(); ++i) {
      result[i] = 32 + (rand32() % 96);
    }
    return result;
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::unique_ptr<HashStringAllocator> allocator_;
  int32_t sequence_ = 0;
  folly::Random::DefaultGenerator rng_;
};

TEST_F(HashStringAllocatorTest, allocate) {
  for (auto count = 0; count < 3; ++count) {
    std::vector<HSA::Header*> headers;
    for (auto i = 0; i < 10'000; ++i) {
      headers.push_back(allocate((i % 10) * 10));
    }
    allocator_->checkConsistency();
    for (int32_t step = 7; step >= 1; --step) {
      for (auto i = 0; i < headers.size(); i += step) {
        if (headers[i]) {
          allocator_->free(headers[i]);
          headers[i] = nullptr;
        }
      }
      allocator_->checkConsistency();
    }
  }
  // We allow for some free overhead for free lists after all is freed.
  EXPECT_LE(allocator_->retainedSize() - allocator_->freeSpace(), 250);
}

TEST_F(HashStringAllocatorTest, allocateLarge) {
  // Verify that allocate() can handle sizes larger than the largest class size
  // supported by memory allocators, that is, 256 pages.
  auto size =
      memory::AllocationTraits::pageBytes(pool_->largestSizeClass() + 1);
  auto header = allocate(size);
  allocator_->free(header);
  EXPECT_EQ(0, allocator_->retainedSize());
}

TEST_F(HashStringAllocatorTest, finishWrite) {
  ByteStream stream(allocator_.get());
  auto start = allocator_->newWrite(stream);

  // Write a short string.
  stream.appendStringPiece(folly::StringPiece("abc"));
  auto [firstStart, firstFinish] = allocator_->finishWrite(stream, 0);

  ASSERT_EQ(start.header, firstStart.header);
  ASSERT_EQ(HSA::offset(firstStart.header, firstFinish), 3);

  // Replace short string with a long string that uses two bytes short of
  // available space.
  allocator_->extendWrite(start, stream);
  auto longString = std::string(start.header->size() - 2, 'x');
  stream.appendStringPiece(folly::StringPiece(longString));
  auto [longStart, longFinish] = allocator_->finishWrite(stream, 0);

  ASSERT_EQ(start.header, longStart.header);
  ASSERT_EQ(HSA::offset(longStart.header, longFinish), longString.size());

  // Append another string after the long string.
  allocator_->extendWrite(longFinish, stream);
  stream.appendStringPiece(folly::StringPiece("abc"));
  auto [appendStart, appendFinish] = allocator_->finishWrite(stream, 0);

  ASSERT_NE(appendStart.header, longFinish.header);
  ASSERT_EQ(HSA::offset(longStart.header, appendStart), longString.size());
  ASSERT_EQ(
      HSA::offset(appendStart.header, appendFinish), appendStart.offset() + 3);

  // Replace last string.
  allocator_->extendWrite(appendStart, stream);
  stream.appendStringPiece(folly::StringPiece("abcd"));
  auto [replaceStart, replaceFinish] = allocator_->finishWrite(stream, 0);

  ASSERT_EQ(appendStart.header, replaceStart.header);
  ASSERT_EQ(
      HSA::offset(replaceStart.header, replaceFinish),
      replaceStart.offset() + 4);

  // Read back long and short strings.
  HSA::prepareRead(longStart.header, stream);

  std::string copy;
  copy.resize(longString.size());
  stream.readBytes(copy.data(), copy.size());
  ASSERT_EQ(copy, longString);

  copy.resize(4);
  stream.readBytes(copy.data(), 4);
  ASSERT_EQ(copy, "abcd");
}

TEST_F(HashStringAllocatorTest, multipart) {
  constexpr int32_t kNumSamples = 10'000;
  std::vector<Multipart> data(kNumSamples);
  for (auto count = 0; count < 3; ++count) {
    for (auto i = 0; i < kNumSamples; ++i) {
      if (data[i].start.header && rand32() % 10 > 7) {
        checkAndFree(data[i]);
        continue;
      }
      auto chars = randomString();
      ByteStream stream(allocator_.get());
      if (data[i].start.header) {
        if (rand32() % 5) {
          // 4/5 of cases append to the end.
          allocator_->extendWrite(data[i].current, stream);
        } else {
          // 1/5 of cases rewrite from the start.
          allocator_->extendWrite(data[i].start, stream);
          data[i].current = data[i].start;
          data[i].reference.clear();
        }
      } else {
        data[i].start = allocator_->newWrite(stream, chars.size());
        data[i].current = data[i].start;
        EXPECT_EQ(
            data[i].start.header, HSA::headerOf(stream.ranges()[0].buffer));
      }
      stream.appendStringPiece(folly::StringPiece(chars.data(), chars.size()));
      auto reserve = rand32() % 100;
      data[i].current = allocator_->finishWrite(stream, reserve).second;
      data[i].reference.insert(
          data[i].reference.end(), chars.begin(), chars.end());
    }
    allocator_->checkConsistency();
  }
  for (const auto& d : data) {
    if (d.start.isSet()) {
      checkMultipart(d);
    }
  }
  for (auto& d : data) {
    if (d.start.isSet()) {
      checkAndFree(d);
      allocator_->checkConsistency();
    }
  }
  allocator_->checkConsistency();
}

TEST_F(HashStringAllocatorTest, rewrite) {
  ByteStream stream(allocator_.get());
  auto header = allocator_->allocate(5);
  EXPECT_EQ(16, header->size()); // Rounds up to kMinAlloc.
  HSA::Position current = HSA::Position::atOffset(header, 0);
  for (auto i = 0; i < 10; ++i) {
    allocator_->extendWrite(current, stream);
    stream.appendOne(123456789012345LL);
    current = allocator_->finishWrite(stream, 0).second;
    auto offset = HSA::offset(header, current);
    EXPECT_EQ((i + 1) * sizeof(int64_t), offset);
    // The allocated writable space from 'header' is at least the amount
    // written.
    auto available = HSA::available(HSA::Position::atOffset(header, 0));
    EXPECT_LE((i + 1) * sizeof(int64_t), available);
  }
  EXPECT_EQ(-1, HSA::offset(header, HSA::Position::null()));
  for (auto repeat = 0; repeat < 2; ++repeat) {
    auto position = HSA::seek(header, sizeof(int64_t));
    // We write the words at index 1 and 2.
    allocator_->extendWrite(position, stream);
    stream.appendOne(12345LL);
    stream.appendOne(67890LL);
    position = allocator_->finishWrite(stream, 0).second;
    EXPECT_EQ(3 * sizeof(int64_t), HSA::offset(header, position));
    HSA::prepareRead(header, stream);
    EXPECT_EQ(123456789012345LL, stream.read<int64_t>());
    EXPECT_EQ(12345LL, stream.read<int64_t>());
    EXPECT_EQ(67890LL, stream.read<int64_t>());
  }
  // The stream contains 3 int64_t's.
  auto end = HSA::seek(header, 3 * sizeof(int64_t));
  EXPECT_EQ(0, HSA::available(end));
  allocator_->ensureAvailable(32, end);
  EXPECT_EQ(32, HSA::available(end));
}

TEST_F(HashStringAllocatorTest, stlAllocator) {
  {
    std::vector<double, StlAllocator<double>> data(
        StlAllocator<double>(allocator_.get()));
    uint32_t counter{0};
    {
      RowSizeTracker trackSize(counter, *allocator_);

      // The contiguous size goes to 80K, rounded to 128K by
      // std::vector. This covers making an extra-large slab in the
      // allocator.
      for (auto i = 0; i < 10'000; i++) {
        data.push_back(i);
      }
    }
    EXPECT_LE(128 * 1024, counter);
    for (auto i = 0; i < 10'000; i++) {
      ASSERT_EQ(i, data[i]);
    }

    data.clear();
    for (auto i = 0; i < 10'000; i++) {
      data.push_back(i);
    }

    for (auto i = 0; i < 10'000; i++) {
      ASSERT_EQ(i, data[i]);
    }

    data.clear();

    // Repeat allocations, now peaking at a largest contiguous block of 256K
    for (auto i = 0; i < 20'000; i++) {
      data.push_back(i);
    }

    for (auto i = 0; i < 20'000; i++) {
      ASSERT_EQ(i, data[i]);
    }
  }

  allocator_->checkConsistency();

  // We allow for some overhead for free lists after all is freed.
  EXPECT_LE(allocator_->retainedSize() - allocator_->freeSpace(), 100);
}

TEST_F(HashStringAllocatorTest, stlAllocatorWithSet) {
  {
    std::unordered_set<
        double,
        std::hash<double>,
        std::equal_to<double>,
        StlAllocator<double>>
        set(StlAllocator<double>(allocator_.get()));

    for (auto i = 0; i < 10'000; i++) {
      set.insert(i);
    }
    for (auto i = 0; i < 10'000; i++) {
      ASSERT_EQ(1, set.count(i));
    }

    set.clear();
    for (auto i = 0; i < 10'000; i++) {
      ASSERT_EQ(0, set.count(i));
    }

    for (auto i = 10'000; i < 20'000; i++) {
      set.insert(i);
    }
    for (auto i = 10'000; i < 20'000; i++) {
      ASSERT_EQ(1, set.count(i));
    }
  }

  allocator_->checkConsistency();

  // We allow for some overhead for free lists after all is freed.
  EXPECT_LE(allocator_->retainedSize() - allocator_->freeSpace(), 180);
}

TEST_F(HashStringAllocatorTest, alignedStlAllocatorWithF14Map) {
  {
    folly::F14FastMap<
        int32_t,
        double,
        std::hash<int32_t>,
        std::equal_to<int32_t>,
        AlignedStlAllocator<std::pair<const int32_t, double>, 16>>
        map(AlignedStlAllocator<std::pair<const int32_t, double>, 16>(
            allocator_.get()));

    for (auto i = 0; i < 10'000; i++) {
      map.try_emplace(i, i + 0.05);
    }
    for (auto i = 0; i < 10'000; i++) {
      ASSERT_EQ(1, map.count(i));
    }

    map.clear();
    for (auto i = 0; i < 10'000; i++) {
      ASSERT_EQ(0, map.count(i));
    }

    for (auto i = 10'000; i < 20'000; i++) {
      map.try_emplace(i, i + 0.15);
    }
    for (auto i = 10'000; i < 20'000; i++) {
      ASSERT_EQ(1, map.count(i));
    }
  }

  allocator_->checkConsistency();

  // We allow for some overhead for free lists after all is freed. Map tends to
  // generate more free blocks at the end, so we loosen the upper bound a bit.
  EXPECT_LE(allocator_->retainedSize() - allocator_->freeSpace(), 130);
}

TEST_F(HashStringAllocatorTest, stlAllocatorOverflow) {
  StlAllocator<int64_t> alloc(allocator_.get());
  VELOX_ASSERT_THROW(alloc.allocate(1ULL << 62), "integer overflow");
  AlignedStlAllocator<int64_t, 16> alignedAlloc(allocator_.get());
  VELOX_ASSERT_THROW(alignedAlloc.allocate(1ULL << 62), "integer overflow");
}

} // namespace
} // namespace facebook::velox
