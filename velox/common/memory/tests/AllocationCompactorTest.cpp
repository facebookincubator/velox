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
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/vector/SelectivityVector.h"

#include <folly/Random.h>

#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

namespace facebook::velox {
namespace {

using HSA = HashStringAllocator;
using Header = HSA::Header;
using AllocationCompactor = HSA::AllocationCompactor;

class AllocationCompactorTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::initialize({});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool();
    rng_.seed(1);
  }

  // TODO: support multi-arena
  // Adds an allocation with size of 'size', and creates blocks with size
  // specified by 'blockSizes'. Sum of 'blockSizes' should equal to 'size' -
  // sizeof(Header) * block number. 'isFree' indicates if block is a free block.
  // There should be no adjacent free blocks. 'multipartPairs' are the pairs
  // whose first has continued pointer pointing to second. Returns the range of
  // allocation.
  folly::Range<char*> addAllocation(
      int64_t size,
      const std::vector<int64_t>& blockSizes,
      const SelectivityVector& isFree,
      const std::vector<std::pair<int32_t, int32_t>>& multipartPairs = {}) {
    VELOX_CHECK(
        size % memory::AllocationTraits::kPageSize == 0 ||
        size % memory::AllocationTraits::kHugePageSize == 0);
    auto blockSizeSum = std::accumulate(
        blockSizes.begin(),
        blockSizes.end(),
        blockSizes.size() * sizeof(Header));
    VELOX_CHECK_EQ(blockSizeSum, size - simd::kPadding);
    VELOX_CHECK_EQ(blockSizes.size(), isFree.size());

    auto data = pool_->allocate(size);
    ranges_.emplace_back(reinterpret_cast<char*>(data), size);

    auto header = reinterpret_cast<Header*>(data);
    bool previousIsFree{false};
    for (auto i = 0; i < blockSizes.size(); ++i) {
      *header = Header(blockSizes[i]);
      if (previousIsFree) {
        header->setPreviousFree();
      }
      if (isFree.isValid(i)) {
        VELOX_CHECK(!previousIsFree);
        header->setFree();
        previousIsFree = true;
      } else {
        previousIsFree = false;
      }
      header = reinterpret_cast<Header*>(header->end());
    }

    VELOX_CHECK_EQ(
        reinterpret_cast<char*>(header) - ranges_.back().data(),
        size - simd::kPadding);
    *reinterpret_cast<uint32_t*>(header) = Header::kArenaEnd;

    std::vector<Header*> blocks;
    blocks.reserve(blockSizes.size());
    header = reinterpret_cast<Header*>(data);
    while (header != nullptr) {
      blocks.push_back(header);
      header = header->next();
    }
    for (const auto& [prev, next] : multipartPairs) {
      VELOX_CHECK_LT(prev, blockSizes.size());
      VELOX_CHECK_LT(next, blockSizes.size());

      auto prevHeader = blocks[prev];
      auto nextHeader = blocks[next];
      VELOX_CHECK(!prevHeader->isFree());
      VELOX_CHECK(!nextHeader->isFree());
      VELOX_CHECK(!prevHeader->isContinued());
      prevHeader->setContinued();
      prevHeader->setNextContinued(nextHeader);
    }

    return ranges_.back();
  }

  void clear() {
    for (auto& range : ranges_) {
      pool_->free(range.data(), range.size());
    }
    ranges_.clear();
  }

  // Fill random strings into non-free blocks of 'ranges_'. Returns a map from
  // non-free blocks' header to the filled string. Multipart's string are
  // concatenated.
  folly::F14FastMap<Header*, std::string> fillContent(
      int32_t allocationIndex = -1) {
    folly::F14FastSet<Header*> continuedHeaders;
    for (auto i = 0; i < ranges_.size(); ++i) {
      if (allocationIndex >= 0 && i != allocationIndex) {
        continue;
      }
      const auto& range = ranges_[i];
      for (int64_t offset = 0; offset < range.size();
           offset += memory::AllocationTraits::kHugePageSize) {
        auto header = reinterpret_cast<Header*>(range.data() + offset);
        while (header != nullptr) {
          if (header->isContinued()) {
            continuedHeaders.insert(header->nextContinued());
          }
          header = header->next();
        }
      }
    }

    folly::F14FastMap<Header*, std::string> result;
    for (auto i = 0; i < ranges_.size(); ++i) {
      if (allocationIndex >= 0 && i != allocationIndex) {
        continue;
      }
      const auto& range = ranges_[i];
      for (int64_t offset = 0; offset < range.size();
           offset += memory::AllocationTraits::kHugePageSize) {
        auto header = reinterpret_cast<Header*>(range.data() + offset);
        while (header != nullptr) {
          if (!header->isFree() && !continuedHeaders.contains(header)) {
            std::string totalString{};
            auto currentHeader = header;
            while (true) {
              auto currentString = randomString(currentHeader->usableSize());
              memcpy(
                  currentHeader->begin(),
                  currentString.data(),
                  currentHeader->usableSize());
              totalString += currentString;
              if (!currentHeader->isContinued()) {
                break;
              }
              currentHeader = currentHeader->nextContinued();
            }
            result[header] = std::move(totalString);
          }
          header = header->next();
        }
      }
    }
    return result;
  }

  // TODO: Consider reuse of Header::toString().
  std::string allocationToString(int32_t index) const {
    VELOX_USER_CHECK_LT(index, ranges_.size());

    std::stringstream out;

    out << '[';
    const auto& range = ranges_.at(index);
    auto header = reinterpret_cast<Header*>(range.data());
    const auto end =
        reinterpret_cast<Header*>(range.data() + range.size() - simd::kPadding);
    while (header != end) {
      if (header->isPreviousFree()) {
        out << 'P';
      }
      if (header->isFree()) {
        out << 'F';
      }
      if (header->isContinued()) {
        out << 'C';
      }
      out << header->size();
      if (header->next()) {
        out << "|";
      }
      header = reinterpret_cast<Header*>(header->end());
    }
    out << ']';
    return out.str();
  }

  uint32_t rand32() {
    return folly::Random::rand32(rng_);
  }

  std::string randomString(int32_t size = 0) {
    std::string result;
    result.resize(
        size != 0 ? size
                  : 20 +
                (rand32() % 10 > 8 ? rand32() % 200 : 1000 + rand32() % 1000));
    for (auto i = 0; i < result.size(); ++i) {
      result[i] = 32 + (rand32() % 96);
    }
    return result;
  }

  void verifyInitialization(
      int32_t allocationIndex,
      const AllocationCompactor& ac) {
    VELOX_CHECK_LT(allocationIndex, ranges_.size());
    const auto& range{ranges_[allocationIndex]};

    EXPECT_EQ(ac.size(), range.size());
    EXPECT_EQ(
        ac.usableSize(),
        range.size() - simd::kPadding -
            AllocationCompactor::kReservationPerArena);

    auto expectedNonFreeBlockSize{0};
    AllocationCompactor::foreachBlock(
        range, [&expectedNonFreeBlockSize](Header* header) {
          if (!header->isFree()) {
            expectedNonFreeBlockSize += header->size() + sizeof(Header);
          }
        });
    EXPECT_EQ(ac.nonFreeBlockSize(), expectedNonFreeBlockSize);
  }

  void verifyAccumulateMultipartMap(
      int32_t allocationIndex,
      const AllocationCompactor& ac,
      std::vector<std::pair<int32_t, int32_t>>& multipartPairs) {
    folly::F14FastMap<Header*, Header*> multipartMap;
    ac.accumulateMultipartMap(multipartMap);
    EXPECT_EQ(multipartMap.size(), multipartPairs.size());

    auto& allocation = ranges_[allocationIndex];
    // TODO: support multi-arena
    auto blockAt = [&allocation](int32_t index) {
      auto header = reinterpret_cast<Header*>(allocation.data());
      int32_t cnt{0};
      while (header != nullptr) {
        if (cnt == index) {
          return header;
        }
        header = header->next();
        cnt++;
      }
      VELOX_UNREACHABLE();
    };

    for (const auto& [prev, next] : multipartPairs) {
      const auto nextBlock{blockAt(next)};
      const auto prevBlock{blockAt(prev)};
      EXPECT_TRUE(nextBlock != nullptr);
      EXPECT_TRUE(prevBlock != nullptr);
      EXPECT_TRUE(prevBlock->nextContinued() == nextBlock);
      EXPECT_TRUE(multipartMap[nextBlock] == prevBlock);
    }
  }

  void verifySqueeze(int32_t allocationIndex, AllocationCompactor& ac) {
    auto& range = ranges_[allocationIndex];

    folly::F14FastMap<Header*, std::string> headerToContent =
        fillContent(allocationIndex);

    std::vector<int32_t> nonFreeBlockSizes;
    ac.foreachBlock([&nonFreeBlockSizes](Header* header) {
      if (!header->isFree()) {
        nonFreeBlockSizes.push_back(header->size());
      }
    });

    folly::F14FastMap<Header*, Header*> multipartMap;
    folly::F14FastMap<Header*, Header*> updatedBlocks;

    ac.accumulateMultipartMap(multipartMap);

    auto resultFreeBlocks = ac.squeeze(multipartMap, updatedBlocks);
    std::cout << allocationToString(0) << '\n';

    // Verify the last free block.
    const auto remainingSize = range.size() - simd::kPadding -
        std::accumulate(nonFreeBlockSizes.begin(),
                        nonFreeBlockSizes.end(),
                        sizeof(Header) * (nonFreeBlockSizes.size()));
    EXPECT_TRUE(
        remainingSize == 0 ||
        remainingSize >= AllocationCompactor::kMinBlockSize);
    if (remainingSize == 0) {
      EXPECT_TRUE(resultFreeBlocks.empty());
    } else {
      const auto expectedLastFreeSize = remainingSize - sizeof(Header);
      EXPECT_EQ(resultFreeBlocks.size(), 1);
      EXPECT_TRUE(resultFreeBlocks[0]->isFree());
      EXPECT_FALSE(resultFreeBlocks[0]->isPreviousFree());
      EXPECT_FALSE(resultFreeBlocks[0]->isContinued());
      EXPECT_EQ(resultFreeBlocks[0]->size(), expectedLastFreeSize);
    }

    // Verify blocks before last free are non-free.
    Header* searchEnd = (remainingSize > 0)
        ? resultFreeBlocks[0]
        : reinterpret_cast<Header*>(
              range.data() + range.size() - simd::kPadding);
    auto header = reinterpret_cast<Header*>(range.data());
    while (header != searchEnd) {
      EXPECT_FALSE(header->isFree());
      EXPECT_FALSE(header->isPreviousFree());
      header = reinterpret_cast<Header*>(header->end());
    }

    // Verify content did not change.
    for (const auto& [header, content] : headerToContent) {
      auto offset{0};
      Header* headerNow = header;
      if (updatedBlocks.contains(headerNow)) {
        headerNow = updatedBlocks[headerNow];
        continue;
      }
      EXPECT_TRUE(!headerNow->isFree());

      while (content.size() - offset > 0) {
        const auto compareSize =
            std::min<int32_t>(headerNow->usableSize(), content.size() - offset);
        EXPECT_GT(compareSize, 0);
        EXPECT_EQ(
            0,
            memcmp(headerNow->begin(), content.data() + offset, compareSize));
        if (headerNow->isContinued()) {
          EXPECT_NE(headerNow->nextContinued(), nullptr);
          headerNow = headerNow->nextContinued();
        }
        offset += compareSize;
      }
    }
  }

 private:
  std::shared_ptr<memory::MemoryPool> pool_;
  folly::Random::DefaultGenerator rng_;
  std::vector<folly::Range<char*>> ranges_;
};

TEST_F(AllocationCompactorTest, basic) {
  static const auto kHugePageSize = memory::AllocationTraits::kHugePageSize;
  static const auto kPageSize = memory::AllocationTraits::kPageSize;
  static const auto smallAllocationSize =
      memory::AllocationPool::kMinPages * kPageSize;
  static const auto kMinBlockSize = AllocationCompactor::kMinBlockSize;

  // TODO: generalize to multi-arena
  auto complementLastBlockSize = [](int64_t size,
                                    std::vector<int64_t>& blockSizes) {
    const auto remainingSize = size - simd::kPadding -
        std::accumulate(blockSizes.begin(),
                        blockSizes.end(),
                        sizeof(Header) * (blockSizes.size()));
    VELOX_USER_CHECK_GE(remainingSize, 0);
    if (remainingSize == 0) {
      return;
    }
    VELOX_USER_CHECK_GE(remainingSize, kMinBlockSize);
    blockSizes.push_back(remainingSize - sizeof(Header));
  };

  // [42 | F65 | P240 | 1024 | F40 | P32 | 50 | F63981]
  {
    // Build data for initializing allocation
    // Size of blocks.
    const auto allocationSize{smallAllocationSize};
    std::vector<int64_t> blockSizes{40, 65, 240, 1024, 40, 32, 50};
    complementLastBlockSize(allocationSize, blockSizes);
    // isFree vector.
    SelectivityVector isFree(blockSizes.size(), false);
    std::vector<int32_t> freeBlocks{
        1, 4, static_cast<int32_t>(blockSizes.size() - 1)};
    for (auto i = 0; i < freeBlocks.size(); ++i) {
      if (i > 0) {
        VELOX_CHECK_GT(freeBlocks[i] - freeBlocks[i - 1], 1);
      }
      isFree.setValid(freeBlocks[i], true);
    }
    isFree.updateBounds();
    // multipart pairs.
    std::vector<std::pair<int32_t, int32_t>> multipartPairs{{2, 6}, {5, 3}};

    auto allocation =
        addAllocation(allocationSize, blockSizes, isFree, multipartPairs);

    std::cout << allocationToString(0) << '\n';

    AllocationCompactor ac(allocation);
    verifyInitialization(0, ac);
    verifyAccumulateMultipartMap(0, ac, multipartPairs);
    verifySqueeze(0, ac);
    clear();
  }

  // Empty
  {
    const auto allocationSize{smallAllocationSize};
    std::vector<int64_t> blockSizes{};
    complementLastBlockSize(allocationSize, blockSizes);
    // isFree vector.
    SelectivityVector isFree(blockSizes.size(), true);
    isFree.updateBounds();
    // multipart pairs.
    std::vector<std::pair<int32_t, int32_t>> multipartPairs{};

    auto allocation =
        addAllocation(allocationSize, blockSizes, isFree, multipartPairs);

    std::cout << allocationToString(0) << '\n';

    AllocationCompactor ac(allocation);
    verifyInitialization(0, ac);
    verifyAccumulateMultipartMap(0, ac, multipartPairs);
    verifySqueeze(0, ac);
    clear();
  }

  // Full, single block
  {
    const auto allocationSize{smallAllocationSize};
    std::vector<int64_t> blockSizes{};
    complementLastBlockSize(allocationSize, blockSizes);
    // isFree vector.
    SelectivityVector isFree(blockSizes.size(), false);
    isFree.updateBounds();
    // multipart pairs.
    std::vector<std::pair<int32_t, int32_t>> multipartPairs{};

    auto allocation =
        addAllocation(allocationSize, blockSizes, isFree, multipartPairs);

    std::cout << allocationToString(0) << '\n';

    AllocationCompactor ac(allocation);
    verifyInitialization(0, ac);
    verifyAccumulateMultipartMap(0, ac, multipartPairs);
    verifySqueeze(0, ac);
    clear();
  }

  // Full, multi-block
  {
    const auto allocationSize{smallAllocationSize};
    std::vector<int64_t> blockSizes{64, 128, 256, 512, 1024, 2048, 4096, 8192,
                                    64, 128, 256, 512, 1024, 2048, 4096, 8192,
                                    64, 128, 256, 512, 1024, 2048, 4096, 8192,
                                    64, 128, 256, 512, 1024, 2048, 4096, 8192};
    complementLastBlockSize(allocationSize, blockSizes);
    // isFree vector.
    SelectivityVector isFree(blockSizes.size(), false);
    isFree.updateBounds();
    // multipart pairs.
    std::vector<std::pair<int32_t, int32_t>> multipartPairs{};

    auto allocation =
        addAllocation(allocationSize, blockSizes, isFree, multipartPairs);

    std::cout << allocationToString(0) << '\n';

    AllocationCompactor ac(allocation);
    verifyInitialization(0, ac);
    verifyAccumulateMultipartMap(0, ac, multipartPairs);
    verifySqueeze(0, ac);
    clear();
  }

  // Mix of free / non-free.
  {
    const auto allocationSize{smallAllocationSize};
    std::vector<int64_t> blockSizes{64, 128, 256, 512, 1024, 2048, 4096, 8192,
                                    64, 128, 256, 512, 1024, 2048, 4096, 8192,
                                    64, 128, 256, 512, 1024, 2048, 4096, 8192,
                                    64, 128, 256, 512, 1024, 2048, 4096, 8192};
    complementLastBlockSize(allocationSize, blockSizes);
    // isFree vector.
    SelectivityVector isFree(blockSizes.size(), false);
    for (auto i = 1; i < blockSizes.size(); i += 2) {
      isFree.setValid(i, true);
    }
    isFree.updateBounds();
    // multipart pairs.
    std::vector<std::pair<int32_t, int32_t>> multipartPairs{};

    auto allocation =
        addAllocation(allocationSize, blockSizes, isFree, multipartPairs);

    std::cout << allocationToString(0) << '\n';

    AllocationCompactor ac(allocation);
    verifyInitialization(0, ac);
    verifyAccumulateMultipartMap(0, ac, multipartPairs);
    verifySqueeze(0, ac);
    clear();
  }
}

TEST_F(AllocationCompactorTest, moveBlock) {
  static const auto kMinBlockSize = AllocationCompactor::kMinBlockSize;

  struct DummyBlock {
    explicit DummyBlock(int32_t size)
        : data{new char[size + sizeof(Header)]},
          header{new (data.get()) Header(size)} {}

    std::string_view string_view(int32_t upto = 0) const {
      if (upto == 0) {
        return std::string_view(header->begin(), header->usableSize());
      }
      VELOX_CHECK_LE(upto, header->size());
      return std::string_view(header->begin(), upto);
    }

    std::unique_ptr<char[]> data;
    Header* header;
  };

  constexpr int32_t srcSize{128};
  DummyBlock srcBlocks[2] = {DummyBlock(srcSize), DummyBlock(srcSize)};

  auto srcContent = randomString(srcSize);
  // srcBlocks[0] is a standalone block.
  memcpy(srcBlocks[0].header->begin(), srcContent.data(), srcSize);

  // srcBlocks[1] is a continued block that continues to contBlock.
  DummyBlock contBlock(HSA::kMinAlloc);
  srcBlocks[1].header->setContinued();
  srcBlocks[1].header->setNextContinued(contBlock.header);
  memcpy(
      srcBlocks[1].header->begin(),
      srcContent.data(),
      srcBlocks[1].header->usableSize());

  // srcSize == destSize
  for (auto i = 0; i < 2; ++i) {
    auto& src = srcBlocks[i];

    folly::F14FastMap<Header*, Header*> multipartMap;
    if (src.header->isContinued()) {
      multipartMap[src.header->nextContinued()] = src.header;
    }
    folly::F14FastMap<Header*, Header*> updatedBlocks;

    DummyBlock dest(srcSize);
    dest.header->setFree();

    auto result = AllocationCompactor::moveBlock(
        src.header, 0, nullptr, dest.header, multipartMap, updatedBlocks);

    EXPECT_EQ(result.srcMovedSize, src.header->size());
    EXPECT_EQ(result.prevContPtr, nullptr);
    EXPECT_EQ(result.remainingDestBlock, nullptr);
    EXPECT_EQ(result.destDiscardedSize, 0);

    EXPECT_EQ(src.header->size(), dest.header->size());
    EXPECT_EQ(src.string_view(), dest.string_view());
    EXPECT_FALSE(dest.header->isFree());
    EXPECT_FALSE(dest.header->isPreviousFree());
    EXPECT_EQ(dest.header->isContinued(), src.header->isContinued());
    if (src.header->isContinued()) {
      EXPECT_EQ(dest.header->nextContinued(), src.header->nextContinued());
    }

    EXPECT_TRUE(updatedBlocks.contains(src.header));
    EXPECT_EQ(updatedBlocks[src.header], dest.header);
    if (src.header->isContinued()) {
      EXPECT_EQ(multipartMap[src.header->nextContinued()], dest.header);
    } else {
      EXPECT_TRUE(multipartMap.empty());
    }
  }

  // srcSize + kMinBlockSize <= destSize, remaining of dest block forms a new
  // free block.
  for (auto i = 0; i < 2; ++i) {
    auto& src = srcBlocks[i];

    folly::F14FastMap<Header*, Header*> multipartMap;
    if (src.header->isContinued()) {
      multipartMap[src.header->nextContinued()] = src.header;
    }
    folly::F14FastMap<Header*, Header*> updatedBlocks;

    DummyBlock dest(srcSize + kMinBlockSize);
    dest.header->setFree();

    auto result = AllocationCompactor::moveBlock(
        src.header, 0, nullptr, dest.header, multipartMap, updatedBlocks);

    EXPECT_EQ(result.srcMovedSize, src.header->size());
    EXPECT_EQ(result.prevContPtr, nullptr);
    EXPECT_NE(result.remainingDestBlock, nullptr);
    EXPECT_EQ(result.destDiscardedSize, 0);

    EXPECT_EQ(src.header->size(), dest.header->size());
    EXPECT_EQ(src.string_view(), dest.string_view());
    EXPECT_FALSE(dest.header->isFree());
    EXPECT_FALSE(dest.header->isPreviousFree());
    EXPECT_EQ(dest.header->isContinued(), src.header->isContinued());
    if (src.header->isContinued()) {
      EXPECT_EQ(dest.header->nextContinued(), src.header->nextContinued());
    }

    auto destRemaining = result.remainingDestBlock;
    EXPECT_EQ(destRemaining, dest.header->next());
    EXPECT_EQ(destRemaining->size(), kMinBlockSize - sizeof(Header));
    EXPECT_TRUE(destRemaining->isFree());
    EXPECT_FALSE(destRemaining->isPreviousFree());

    EXPECT_TRUE(updatedBlocks.contains(src.header));
    EXPECT_EQ(updatedBlocks[src.header], dest.header);
    if (src.header->isContinued()) {
      EXPECT_EQ(multipartMap[src.header->nextContinued()], dest.header);
    } else {
      EXPECT_TRUE(multipartMap.empty());
    }
  }

  // srcSize >= destSize + kMinAlloc - sizeof(ptr), split src block into two
  // valid blocks.
  for (auto i = 0; i < 2; ++i) {
    auto& src = srcBlocks[i];

    folly::F14FastMap<Header*, Header*> multipartMap;
    if (src.header->isContinued()) {
      multipartMap[src.header->nextContinued()] = src.header;
    }
    folly::F14FastMap<Header*, Header*> updatedBlocks;

    constexpr auto destSize1{
        srcSize - HSA::kMinAlloc + Header::kContinuedPtrSize};
    DummyBlock dest1(destSize1);
    dest1.header->setFree();

    auto result1 = AllocationCompactor::moveBlock(
        src.header, 0, nullptr, dest1.header, multipartMap, updatedBlocks);

    constexpr auto expectedMovedSize{destSize1 - Header::kContinuedPtrSize};
    EXPECT_EQ(result1.srcMovedSize, expectedMovedSize);
    EXPECT_EQ(
        result1.prevContPtr,
        reinterpret_cast<Header**>(
            dest1.header->end() - Header::kContinuedPtrSize));
    EXPECT_EQ(result1.remainingDestBlock, nullptr);
    EXPECT_EQ(result1.destDiscardedSize, 0);

    EXPECT_EQ(destSize1, dest1.header->size());
    EXPECT_EQ(
        src.string_view(expectedMovedSize),
        dest1.string_view(expectedMovedSize));
    EXPECT_FALSE(dest1.header->isFree());
    EXPECT_FALSE(dest1.header->isPreviousFree());
    EXPECT_TRUE(dest1.header->isContinued());

    EXPECT_TRUE(updatedBlocks.contains(src.header));
    EXPECT_EQ(updatedBlocks[src.header], dest1.header);
    if (src.header->isContinued()) {
      EXPECT_EQ(multipartMap[src.header->nextContinued()], src.header);
    } else {
      EXPECT_TRUE(multipartMap.empty());
    }

    // Move the rest of src block.
    const auto offset{result1.srcMovedSize};
    constexpr auto destSize2{srcSize};
    DummyBlock dest2(destSize2);
    dest2.header->setFree();

    auto result2 = AllocationCompactor::moveBlock(
        src.header,
        offset,
        result1.prevContPtr,
        dest2.header,
        multipartMap,
        updatedBlocks);

    EXPECT_EQ(result2.srcMovedSize, srcSize - expectedMovedSize);
    EXPECT_EQ(result2.prevContPtr, nullptr);
    EXPECT_NE(result2.remainingDestBlock, nullptr);
    EXPECT_EQ(result2.destDiscardedSize, 0);

    EXPECT_EQ(dest2.header->size(), result2.srcMovedSize);
    EXPECT_EQ(
        dest2.string_view(dest2.header->usableSize()),
        std::string_view(
            src.header->begin() + expectedMovedSize,
            dest2.header->usableSize()));
    EXPECT_FALSE(dest2.header->isFree());
    EXPECT_FALSE(dest2.header->isPreviousFree());
    EXPECT_EQ(dest2.header->isContinued(), src.header->isContinued());
    EXPECT_EQ(dest1.header->nextContinued(), dest2.header);

    EXPECT_EQ(updatedBlocks.size(), 1);
    EXPECT_EQ(updatedBlocks[src.header], dest1.header);
    if (src.header->isContinued()) {
      EXPECT_EQ(multipartMap[src.header->nextContinued()], dest2.header);
    } else {
      EXPECT_TRUE(multipartMap.empty());
    }
  }

  // destSize - kMinBlockSize < srcSize < destSize and src block is not
  // continued, the remaining of dest block is not enough for a valid block,
  // use it up.
  {
    auto& src = srcBlocks[0];

    folly::F14FastMap<Header*, Header*> multipartMap;
    folly::F14FastMap<Header*, Header*> updatedBlocks;

    constexpr auto destSize{srcSize + kMinBlockSize - 1};
    DummyBlock dest(destSize);
    dest.header->setFree();

    auto result = AllocationCompactor::moveBlock(
        src.header, 0, nullptr, dest.header, multipartMap, updatedBlocks);

    EXPECT_EQ(result.srcMovedSize, src.header->size());
    EXPECT_EQ(result.prevContPtr, nullptr);
    EXPECT_EQ(result.remainingDestBlock, nullptr);
    EXPECT_EQ(result.destDiscardedSize, 0);

    EXPECT_EQ(destSize, dest.header->size());
    EXPECT_EQ(src.string_view(), dest.string_view(src.header->size()));
    EXPECT_FALSE(dest.header->isFree());
    EXPECT_FALSE(dest.header->isPreviousFree());
    EXPECT_FALSE(dest.header->isContinued());

    EXPECT_TRUE(updatedBlocks.contains(src.header));
    EXPECT_EQ(updatedBlocks[src.header], dest.header);
    EXPECT_TRUE(multipartMap.empty());
  }

  // destSize-kMinBlockSize < srcSize < destSize and destSize >=
  // kMinBlockSize+kMinAlloc and srcSize >= 2*kMinAlloc-sizeof(ptr), and src
  // block is continued. Split src block into A and B so that:
  // 1. sizeof(A)+sizeof(ptr) >= kMinAlloc.
  // 2. sizeof(B) >= kMinAlloc.
  // 3. destSize-(sizeof(A)+sizeof(ptr)) >= kMinBlockSize.
  {
    auto& src = srcBlocks[1];

    folly::F14FastMap<Header*, Header*> multipartMap;
    multipartMap[src.header->nextContinued()] = src.header;
    folly::F14FastMap<Header*, Header*> updatedBlocks;

    constexpr auto destSize{srcSize + kMinBlockSize - 1};
    DummyBlock dest(destSize);
    dest.header->setFree();

    auto result = AllocationCompactor::moveBlock(
        src.header, 0, nullptr, dest.header, multipartMap, updatedBlocks);

    const auto expectedMovedSize{std::min<int32_t>(
        destSize - kMinBlockSize - Header::kContinuedPtrSize,
        srcSize - HSA::kMinAlloc)};
    EXPECT_EQ(result.srcMovedSize, expectedMovedSize);
    EXPECT_EQ(
        result.prevContPtr,
        reinterpret_cast<Header**>(
            dest.header->end() - Header::kContinuedPtrSize));
    EXPECT_EQ(result.remainingDestBlock, nullptr);
    EXPECT_EQ(
        result.destDiscardedSize,
        destSize - expectedMovedSize - Header::kContinuedPtrSize);

    EXPECT_EQ(
        dest.header->size(), expectedMovedSize + Header::kContinuedPtrSize);
    EXPECT_EQ(src.string_view(expectedMovedSize), dest.string_view());
    EXPECT_FALSE(dest.header->isFree());
    EXPECT_FALSE(dest.header->isPreviousFree());
    EXPECT_TRUE(dest.header->isContinued());

    EXPECT_TRUE(updatedBlocks.contains(src.header));
    EXPECT_EQ(updatedBlocks[src.header], dest.header);
    EXPECT_EQ(multipartMap[src.header->nextContinued()], src.header);
  }

  // destSize-kMinBlockSize < srcSize < destSize and destSize >=
  // kMinBlockSize+kMinAlloc but srcSize < 2*kMinAlloc-sizeof(ptr), and src
  // block is continued. Unable to split src block, skip.
  {
    const auto srcSize1{2 * HSA::kMinAlloc - Header::kContinuedPtrSize - 1};
    auto src = DummyBlock(srcSize1);
    auto srcCont = DummyBlock(srcSize);
    src.header->setContinued();
    src.header->setNextContinued(srcCont.header);

    folly::F14FastMap<Header*, Header*> multipartMap;
    multipartMap[src.header->nextContinued()] = src.header;
    folly::F14FastMap<Header*, Header*> updatedBlocks;

    constexpr auto destSize{srcSize1 + kMinBlockSize - 1};
    DummyBlock dest(destSize);
    dest.header->setFree();

    auto result = AllocationCompactor::moveBlock(
        src.header, 0, nullptr, dest.header, multipartMap, updatedBlocks);

    EXPECT_EQ(result.srcMovedSize, 0);
    EXPECT_EQ(result.prevContPtr, nullptr);
    EXPECT_EQ(result.remainingDestBlock, nullptr);
    EXPECT_EQ(result.destDiscardedSize, destSize);

    EXPECT_TRUE(updatedBlocks.empty());
    EXPECT_EQ(multipartMap[src.header->nextContinued()], src.header);
  }

  // destSize1 < srcSize < destSize1 + kMinAlloc - sizeof(ptr) and src block
  // is not continued, split the src block and the second part is smaller than
  // kMinAlloc, needs padding.
  {
    auto& src = srcBlocks[0];

    folly::F14FastMap<Header*, Header*> multipartMap;
    folly::F14FastMap<Header*, Header*> updatedBlocks;

    constexpr auto destSize1{
        srcSize - HSA::kMinAlloc + Header::kContinuedPtrSize + 1};
    DummyBlock dest1(destSize1);
    dest1.header->setFree();

    auto result1 = AllocationCompactor::moveBlock(
        src.header, 0, nullptr, dest1.header, multipartMap, updatedBlocks);

    constexpr auto expectedMovedSize{destSize1 - Header::kContinuedPtrSize};
    EXPECT_EQ(result1.srcMovedSize, expectedMovedSize);
    EXPECT_EQ(
        result1.prevContPtr,
        reinterpret_cast<Header**>(
            dest1.header->end() - Header::kContinuedPtrSize));
    EXPECT_EQ(result1.remainingDestBlock, nullptr);
    EXPECT_EQ(result1.destDiscardedSize, 0);

    EXPECT_EQ(destSize1, dest1.header->size());
    EXPECT_EQ(
        src.string_view(expectedMovedSize),
        dest1.string_view(expectedMovedSize));
    EXPECT_FALSE(dest1.header->isFree());
    EXPECT_FALSE(dest1.header->isPreviousFree());
    EXPECT_TRUE(dest1.header->isContinued());

    EXPECT_TRUE(updatedBlocks.contains(src.header));
    EXPECT_EQ(updatedBlocks[src.header], dest1.header);
    EXPECT_TRUE(multipartMap.empty());

    const auto offset{result1.srcMovedSize};
    constexpr auto destSize2{srcSize};
    DummyBlock dest2(destSize2);
    dest2.header->setFree();

    auto result2 = AllocationCompactor::moveBlock(
        src.header,
        offset,
        result1.prevContPtr,
        dest2.header,
        multipartMap,
        updatedBlocks);

    EXPECT_EQ(result2.srcMovedSize, srcSize - expectedMovedSize);
    EXPECT_LT(result2.srcMovedSize, HSA::kMinAlloc);
    EXPECT_EQ(result2.prevContPtr, nullptr);
    EXPECT_NE(result2.remainingDestBlock, nullptr);
    EXPECT_EQ(result2.destDiscardedSize, 0);

    EXPECT_EQ(dest2.header->size(), HSA::kMinAlloc);
    EXPECT_EQ(
        dest2.string_view(result2.srcMovedSize),
        std::string_view(
            src.header->begin() + expectedMovedSize, result2.srcMovedSize));
    EXPECT_FALSE(dest2.header->isFree());
    EXPECT_FALSE(dest2.header->isPreviousFree());
    EXPECT_FALSE(dest2.header->isContinued());
    EXPECT_EQ(dest1.header->nextContinued(), dest2.header);

    EXPECT_TRUE(updatedBlocks.contains(src.header));
    EXPECT_EQ(updatedBlocks[src.header], dest1.header);
    EXPECT_TRUE(multipartMap.empty());
  };

  // destSize < srcSize < destSize+kMinAlloc-sizeof(ptr), destSize
  // >=kMinBlockSize+kMinAlloc and src block is continued, split the src block
  // into A and B so that: block is continued. Split src block into A and B so
  // that:
  // 1. sizeof(A)+sizeof(ptr) >= kMinAlloc.
  // 2. sizeof(B) >= kMinAlloc.
  // 3. destSize-(sizeof(A)+sizeof(ptr)) >= kMinBlockSize.
  {
    auto& src = srcBlocks[1];

    folly::F14FastMap<Header*, Header*> multipartMap;
    multipartMap[src.header->nextContinued()] = src.header;
    folly::F14FastMap<Header*, Header*> updatedBlocks;

    constexpr auto destSize{
        srcSize - HSA::kMinAlloc + Header::kContinuedPtrSize + 1};
    DummyBlock dest(destSize);
    dest.header->setFree();

    auto result = AllocationCompactor::moveBlock(
        src.header, 0, nullptr, dest.header, multipartMap, updatedBlocks);
    const auto expectedMovedSize{
        destSize - kMinBlockSize - Header::kContinuedPtrSize};
    EXPECT_EQ(result.srcMovedSize, expectedMovedSize);
    EXPECT_EQ(
        result.prevContPtr,
        reinterpret_cast<Header**>(
            dest.header->end() - Header::kContinuedPtrSize));
    EXPECT_EQ(result.remainingDestBlock, nullptr);
    EXPECT_EQ(
        result.destDiscardedSize,
        destSize - expectedMovedSize - Header::kContinuedPtrSize);

    EXPECT_EQ(
        dest.header->size(), expectedMovedSize + Header::kContinuedPtrSize);
    EXPECT_EQ(src.string_view(expectedMovedSize), dest.string_view());
    EXPECT_FALSE(dest.header->isFree());
    EXPECT_FALSE(dest.header->isPreviousFree());
    EXPECT_TRUE(dest.header->isContinued());

    EXPECT_TRUE(updatedBlocks.contains(src.header));
    EXPECT_EQ(updatedBlocks[src.header], dest.header);
    EXPECT_EQ(multipartMap[src.header->nextContinued()], src.header);
  };

  // destSize < srcSize < destSize+kMinAlloc-sizeof(ptr), destSize <
  // kMinBlockSize+kMinAlloc and src block is continued, split the src block
  // into A and B so that: block is continued. Unable to split src block, skip.
  {
    constexpr auto destSize{kMinBlockSize + HSA::kMinAlloc - 1};
    DummyBlock dest(destSize);
    dest.header->setFree();

    const auto srcSize1{
        destSize + HSA::kMinAlloc - Header::kContinuedPtrSize - 1};
    auto src = DummyBlock(srcSize1);
    auto srcCont = DummyBlock(srcSize);
    src.header->setContinued();
    src.header->setNextContinued(srcCont.header);

    folly::F14FastMap<Header*, Header*> multipartMap;
    multipartMap[src.header->nextContinued()] = src.header;
    folly::F14FastMap<Header*, Header*> updatedBlocks;

    auto result = AllocationCompactor::moveBlock(
        src.header, 0, nullptr, dest.header, multipartMap, updatedBlocks);

    EXPECT_EQ(result.srcMovedSize, 0);
    EXPECT_EQ(result.prevContPtr, nullptr);
    EXPECT_EQ(result.remainingDestBlock, nullptr);
    EXPECT_EQ(result.destDiscardedSize, destSize);

    EXPECT_TRUE(updatedBlocks.empty());
    EXPECT_EQ(multipartMap[src.header->nextContinued()], src.header);
  }
}

} // namespace
} // namespace facebook::velox
