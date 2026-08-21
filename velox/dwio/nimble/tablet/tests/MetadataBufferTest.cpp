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

#include "velox/dwio/nimble/tablet/MetadataBuffer.h"

#include <gtest/gtest.h>

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MmapAllocator.h"
#include "velox/common/memory/tests/MmapAllocatorCompatibility.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/tablet/Compression.h"

using namespace ::facebook;

namespace facebook::nimble::test {

class MetadataBufferTestHelper {
 public:
  static uint64_t offset(const MetadataBuffer& buffer) {
    return buffer.offset_;
  }

  static uint64_t size(const MetadataBuffer& buffer) {
    return buffer.size_;
  }

  static void verifySlice(
      const MetadataBuffer& buffer,
      uint64_t expectedOffset,
      uint64_t expectedSize,
      std::string_view expectedContent) {
    EXPECT_EQ(offset(buffer), expectedOffset);
    EXPECT_EQ(size(buffer), expectedSize);
    EXPECT_EQ(buffer.content(), expectedContent);
  }
};

} // namespace facebook::nimble::test

namespace {

using TestHelper = nimble::test::MetadataBufferTestHelper;

velox::BufferPtr toBufferPtr(
    std::string_view data,
    velox::memory::MemoryPool* pool) {
  auto buffer = velox::AlignedBuffer::allocate<char>(data.size(), pool);
  std::memcpy(buffer->asMutable<char>(), data.data(), data.size());
  return buffer;
}

class MetadataBufferTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {}

  std::shared_ptr<velox::memory::MemoryPool> rootPool_{
      velox::memory::memoryManager()->addRootPool("MetadataBufferTest")};
  std::shared_ptr<velox::memory::MemoryPool> pool_{
      rootPool_->addLeafChild("MetadataBufferTest")};
};

const std::string kTestData = "Hello, MetadataBuffer!";
const std::string kLargeTestData = std::string(1'000, 'A');

TEST_F(MetadataBufferTest, decompressBufferPtr) {
  struct TestParam {
    std::string inputData;
    nimble::CompressionType compressionType;
    std::string debugString() const {
      return fmt::format(
          "size {}, compressionType {}",
          inputData.size(),
          static_cast<int>(compressionType));
    }
  };
  for (const auto& testData : std::vector<TestParam>{
           {"", nimble::CompressionType::Uncompressed},
           {"", nimble::CompressionType::Zstd},
           {kTestData, nimble::CompressionType::Uncompressed},
           {kLargeTestData, nimble::CompressionType::Uncompressed},
           {kLargeTestData, nimble::CompressionType::Zstd}}) {
    SCOPED_TRACE(testData.debugString());

    velox::BufferPtr inputBuffer;
    if (testData.compressionType == nimble::CompressionType::Uncompressed) {
      inputBuffer = toBufferPtr(testData.inputData, pool_.get());
    } else {
      auto compressed =
          nimble::ZstdCompression::compress(testData.inputData, pool_.get());
      ASSERT_TRUE(compressed.has_value());
      inputBuffer = std::move(compressed.value());
    }

    nimble::MetadataBuffer buffer(
        nimble::MetadataBuffer::decompress(
            std::move(inputBuffer), testData.compressionType, pool_.get()));

    auto content = buffer.content();
    EXPECT_EQ(content, testData.inputData);
    EXPECT_EQ(content.size(), testData.inputData.size());
  }
}

TEST_F(MetadataBufferTest, decompressStringView) {
  struct TestParam {
    std::string inputData;
    nimble::CompressionType compressionType;
    bool compressible;
    std::string debugString() const {
      return fmt::format(
          "size {}, compressionType {}, compressible {}",
          inputData.size(),
          static_cast<int>(compressionType),
          compressible);
    }
  };
  std::vector<TestParam> testSettings;
  for (const auto& inputData : {std::string(""), kTestData, kLargeTestData}) {
    for (const auto compressionType :
         {nimble::CompressionType::Uncompressed,
          nimble::CompressionType::Zstd}) {
      const bool compressible =
          compressionType == nimble::CompressionType::Uncompressed ||
          nimble::ZstdCompression::compress(inputData, pool_.get()).has_value();
      testSettings.push_back({inputData, compressionType, compressible});
    }
  }
  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());

    if (!testData.compressible) {
      auto compressed =
          nimble::ZstdCompression::compress(testData.inputData, pool_.get());
      EXPECT_FALSE(compressed.has_value());
      continue;
    }

    std::string rawData;
    if (testData.compressionType == nimble::CompressionType::Zstd) {
      auto compressed =
          nimble::ZstdCompression::compress(testData.inputData, pool_.get());
      ASSERT_TRUE(compressed.has_value());
      rawData = std::string(
          compressed.value()->as<char>(), compressed.value()->size());
    } else {
      rawData = testData.inputData;
    }

    nimble::MetadataBuffer buffer(
        nimble::MetadataBuffer::decompress(
            rawData, testData.compressionType, pool_.get()));

    auto content = buffer.content();
    EXPECT_EQ(content, testData.inputData);
  }
}

TEST_F(MetadataBufferTest, sectionConstruction) {
  nimble::MetadataBuffer buffer(
      nimble::MetadataBuffer::decompress(
          toBufferPtr(kTestData, pool_.get()),
          nimble::CompressionType::Uncompressed,
          pool_.get()));

  nimble::Section section{std::move(buffer)};

  const auto content = section.content();
  EXPECT_EQ(content, kTestData);
}

TEST_F(MetadataBufferTest, sectionStringViewConversion) {
  nimble::MetadataBuffer buffer(
      nimble::MetadataBuffer::decompress(
          toBufferPtr(kTestData, pool_.get()),
          nimble::CompressionType::Uncompressed,
          pool_.get()));

  nimble::Section section{std::move(buffer)};

  std::string_view content = static_cast<std::string_view>(section);
  EXPECT_EQ(content, kTestData);
}

TEST_F(MetadataBufferTest, sectionBuffer) {
  nimble::MetadataBuffer buffer(
      nimble::MetadataBuffer::decompress(
          toBufferPtr(kTestData, pool_.get()),
          nimble::CompressionType::Uncompressed,
          pool_.get()));

  nimble::Section section{std::move(buffer)};

  const nimble::MetadataBuffer& bufferRef = section.buffer();
  EXPECT_EQ(bufferRef.content(), kTestData);
  EXPECT_EQ(bufferRef.content(), section.content());
}

TEST_F(MetadataBufferTest, metadataSection) {
  {
    nimble::MetadataSection section{100, 200, nimble::CompressionType::Zstd};
    EXPECT_EQ(section.offset(), 100);
    EXPECT_EQ(section.size(), 200);
    EXPECT_EQ(section.compressionType(), nimble::CompressionType::Zstd);
    EXPECT_FALSE(section.uncompressedSize().has_value());
  }

  {
    nimble::MetadataSection section;
    EXPECT_EQ(section.offset(), 0);
    EXPECT_EQ(section.size(), 0);
    EXPECT_EQ(section.compressionType(), nimble::CompressionType::Uncompressed);
    EXPECT_FALSE(section.uncompressedSize().has_value());
  }
}

TEST_F(MetadataBufferTest, metadataSectionUncompressedSize) {
  struct TestParam {
    uint32_t size;
    nimble::CompressionType compressionType;
    std::optional<uint32_t> uncompressedSize;
    std::string debugString() const {
      return fmt::format(
          "size {}, compressionType {}, uncompressedSize {}",
          size,
          static_cast<int>(compressionType),
          uncompressedSize.has_value()
              ? std::to_string(uncompressedSize.value())
              : std::string("nullopt"));
    }
  };
  for (const auto& testData : std::vector<TestParam>{
           {50, nimble::CompressionType::Zstd, 200},
           {50, nimble::CompressionType::Zstd, std::nullopt},
           {100, nimble::CompressionType::Uncompressed, 100},
           {100, nimble::CompressionType::Uncompressed, std::nullopt}}) {
    SCOPED_TRACE(testData.debugString());
    nimble::MetadataSection section{
        0, testData.size, testData.compressionType, testData.uncompressedSize};

    EXPECT_EQ(section.size(), testData.size);
    EXPECT_EQ(section.compressionType(), testData.compressionType);
    EXPECT_EQ(section.uncompressedSize(), testData.uncompressedSize);
  }
}

TEST_F(MetadataBufferTest, metadataBufferMove) {
  // Move construction.
  {
    nimble::MetadataBuffer buffer(
        nimble::MetadataBuffer::decompress(
            toBufferPtr(kTestData, pool_.get()),
            nimble::CompressionType::Uncompressed,
            pool_.get()));

    nimble::MetadataBuffer moved{std::move(buffer)};
    EXPECT_EQ(moved.content(), kTestData);
  }

  // Move assignment.
  {
    nimble::MetadataBuffer buffer(
        nimble::MetadataBuffer::decompress(
            toBufferPtr(kTestData, pool_.get()),
            nimble::CompressionType::Uncompressed,
            pool_.get()));

    nimble::MetadataBuffer other(
        nimble::MetadataBuffer::decompress(
            toBufferPtr("other", pool_.get()),
            nimble::CompressionType::Uncompressed,
            pool_.get()));

    other = std::move(buffer);
    EXPECT_EQ(other.content(), kTestData);
  }
}

TEST_F(MetadataBufferTest, metadataSectionErrors) {
  NIMBLE_ASSERT_THROW(
      nimble::MetadataSection(
          0, 100, nimble::CompressionType::Uncompressed, 200),
      "Uncompressed size must equal size for uncompressed data");
  NIMBLE_ASSERT_THROW(
      nimble::MetadataSection(0, 100, nimble::CompressionType::Zstd, 50),
      "Uncompressed size must be >= compressed size");
}

class MetadataBufferCacheTest : public ::testing::Test {
 protected:
  static constexpr uint64_t kCacheSize = 64 << 20;

  void SetUp() override {
    // This suite unconditionally constructs an MmapAllocator.
    SKIP_IF_MMAP_ALLOCATOR_UNSUPPORTED();
    velox::memory::MemoryManager::Options options;
    options.useMmapAllocator = true;
    options.allocatorCapacity = kCacheSize;
    options.arbitratorCapacity = kCacheSize;
    options.trackDefaultUsage = true;
    manager_ = std::make_unique<velox::memory::MemoryManager>(options);
    allocator_ =
        static_cast<velox::memory::MmapAllocator*>(manager_->allocator());
    cache_ = velox::cache::AsyncDataCache::create(allocator_);
    fileId_ = std::make_unique<velox::StringIdLease>(
        velox::fileIds(), "MetadataBufferCacheTest");
  }

  void TearDown() override {
    fileId_.reset();
    if (cache_ != nullptr) {
      cache_->shutdown();
    }
    cache_.reset();
    manager_.reset();
  }

  velox::cache::CachePin makePin(uint64_t cacheOffset, std::string_view data) {
    velox::cache::RawFileCacheKey cacheKey{fileId_->id(), cacheOffset};
    auto pin = cache_->findOrCreate(cacheKey, data.size());
    EXPECT_FALSE(pin.empty());
    EXPECT_TRUE(pin.entry()->isExclusive());
    auto ranges = pin.entry()->dataRanges(data.size());
    size_t offset = 0;
    for (auto& range : ranges) {
      const auto copySize = std::min(range.size(), data.size() - offset);
      std::memcpy(range.data(), data.data() + offset, copySize);
      offset += copySize;
    }
    pin.entry()->setExclusiveToShared();
    return pin;
  }

  std::unique_ptr<velox::memory::MemoryManager> manager_;
  velox::memory::MmapAllocator* allocator_{nullptr};
  std::shared_ptr<velox::cache::AsyncDataCache> cache_;
  std::unique_ptr<velox::StringIdLease> fileId_;
};

TEST_F(MetadataBufferCacheTest, cachePinData) {
  for (const auto& data : {std::string("tiny"), std::string(1'000, 'X')}) {
    SCOPED_TRACE(fmt::format("size {}", data.size()));

    {
      auto pin = makePin(0, data);
      EXPECT_FALSE(pin.empty());
      nimble::MetadataBuffer pinBuffer{std::move(pin)};
      EXPECT_EQ(pinBuffer.content(), data);
    }

    {
      auto pool = manager_->addLeafPool("bufferTest");
      auto buffer = toBufferPtr(data, pool.get());
      nimble::MetadataBuffer bufferObj{std::move(buffer)};
      EXPECT_EQ(bufferObj.content(), data);
    }
  }
}

TEST_F(MetadataBufferCacheTest, metadataBufferMoveWithPin) {
  {
    const std::string data = "move-test";
    auto pin = makePin(100, data);
    nimble::MetadataBuffer buffer{std::move(pin)};
    nimble::MetadataBuffer moved{std::move(buffer)};
    EXPECT_EQ(moved.content(), data);
  }

  {
    const std::string data1 = "first";
    const std::string data2 = "second";
    auto pin1 = makePin(200, data1);
    auto pin2 = makePin(300, data2);
    nimble::MetadataBuffer buffer1{std::move(pin1)};
    nimble::MetadataBuffer buffer2{std::move(pin2)};
    buffer2 = std::move(buffer1);
    EXPECT_EQ(buffer2.content(), data1);
  }
}

TEST_F(MetadataBufferCacheTest, cachePinSection) {
  const std::string data = "section-cache-test";
  auto pin = makePin(400, data);

  nimble::Section section{nimble::MetadataBuffer{std::move(pin)}};
  EXPECT_EQ(section.content(), data);
}

TEST_F(MetadataBufferTest, cloneBufferPtr) {
  auto buffer = toBufferPtr(kTestData, pool_.get());
  nimble::MetadataBuffer original{std::move(buffer)};
  EXPECT_EQ(original.content(), kTestData);

  auto cloned = original.clone();
  EXPECT_EQ(cloned.content(), kTestData);
  EXPECT_EQ(original.content(), cloned.content());
  EXPECT_EQ(original.content().data(), cloned.content().data());
}

TEST_F(MetadataBufferCacheTest, cloneCachePin) {
  const std::string data = "clone-pin-test";
  auto pin = makePin(500, data);
  nimble::MetadataBuffer original{std::move(pin)};
  EXPECT_EQ(original.content(), data);

  auto cloned = original.clone();
  EXPECT_EQ(cloned.content(), data);
  EXPECT_EQ(original.content(), cloned.content());
}

TEST_F(MetadataBufferTest, sliceBufferPtr) {
  struct SliceOp {
    uint64_t offset;
    uint64_t size;
    uint64_t expectedAbsoluteOffset;
    std::string expectedContent;

    std::string debugString() const {
      return fmt::format(
          "offset {}, size {}, expectedAbsoluteOffset {}, expectedContent '{}'",
          offset,
          size,
          expectedAbsoluteOffset,
          expectedContent);
    }
  };

  struct TestParam {
    std::string data;
    std::vector<std::vector<SliceOp>> sliceChains;

    std::string debugString() const {
      return fmt::format("data '{}'", data);
    }
  };

  // clang-format off
  std::vector<TestParam> testSettings = {
      {kTestData,
       {
           {{0, kTestData.size(), 0, kTestData}},
           {{0, 0, 0, ""}},
           {{7, 14, 7, "MetadataBuffer"}},
           {{7, 14, 7, "MetadataBuffer"}, {8, 6, 15, "Buffer"}},
           {{7, 14, 7, "MetadataBuffer"}, {0, 8, 7, "Metadata"}, {4, 4, 11, "data"}},
           {{0, 5, 0, "Hello"}},
           {{kTestData.size() - 1, 1, kTestData.size() - 1, "!"}},
       }},
      {kLargeTestData,
       {
           {{0, 1'000, 0, kLargeTestData}},
           {{500, 500, 500, std::string(500, 'A')}},
           {{0, 100, 0, std::string(100, 'A')}, {50, 50, 50, std::string(50, 'A')}},
       }},
  };
  // clang-format on

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto buffer = toBufferPtr(testData.data, pool_.get());
    nimble::MetadataBuffer original{std::move(buffer)};
    TestHelper::verifySlice(original, 0, testData.data.size(), testData.data);

    for (const auto& chain : testData.sliceChains) {
      ASSERT_FALSE(chain.empty());
      SCOPED_TRACE(chain[0].debugString());

      auto current = original.slice(chain[0].offset, chain[0].size);
      TestHelper::verifySlice(
          *current,
          chain[0].expectedAbsoluteOffset,
          chain[0].size,
          chain[0].expectedContent);
      TestHelper::verifySlice(original, 0, testData.data.size(), testData.data);

      for (size_t i = 1; i < chain.size(); ++i) {
        SCOPED_TRACE(chain[i].debugString());
        current = current->slice(chain[i].offset, chain[i].size);
        TestHelper::verifySlice(
            *current,
            chain[i].expectedAbsoluteOffset,
            chain[i].size,
            chain[i].expectedContent);
      }
    }
  }
}

TEST_F(MetadataBufferTest, sliceBufferPtrOutOfRange) {
  auto buffer = toBufferPtr(kTestData, pool_.get());
  nimble::MetadataBuffer original{std::move(buffer)};

  NIMBLE_ASSERT_THROW(
      original.slice(0, kTestData.size() + 1),
      "Slice range exceeds buffer content");
  NIMBLE_ASSERT_THROW(
      original.slice(kTestData.size(), 1),
      "Slice range exceeds buffer content");

  auto sliced = original.slice(7, 14);
  NIMBLE_ASSERT_THROW(
      sliced->slice(0, 15), "Slice range exceeds buffer content");
  NIMBLE_ASSERT_THROW(
      sliced->slice(14, 1), "Slice range exceeds buffer content");
}

TEST_F(MetadataBufferCacheTest, sliceCachePin) {
  struct SliceOp {
    uint64_t offset;
    uint64_t size;
    uint64_t expectedAbsoluteOffset;
    std::string expectedContent;

    std::string debugString() const {
      return fmt::format(
          "offset {}, size {}, expectedAbsoluteOffset {}, expectedContent '{}'",
          offset,
          size,
          expectedAbsoluteOffset,
          expectedContent);
    }
  };

  const std::string data = "Hello, CacheSlice!";
  auto pin = makePin(600, data);
  nimble::MetadataBuffer original{std::move(pin)};
  TestHelper::verifySlice(original, 0, data.size(), data);

  // clang-format off
  std::vector<std::vector<SliceOp>> sliceChains = {
      {{0, data.size(), 0, data}},
      {{0, 0, 0, ""}},
      {{7, 10, 7, "CacheSlice"}},
      {{7, 10, 7, "CacheSlice"}, {5, 5, 12, "Slice"}},
      {{7, 10, 7, "CacheSlice"}, {0, 5, 7, "Cache"}, {0, 3, 7, "Cac"}},
      {{data.size() - 1, 1, data.size() - 1, "!"}},
  };
  // clang-format on

  for (const auto& chain : sliceChains) {
    ASSERT_FALSE(chain.empty());
    SCOPED_TRACE(chain[0].debugString());

    auto current = original.slice(chain[0].offset, chain[0].size);
    TestHelper::verifySlice(
        *current,
        chain[0].expectedAbsoluteOffset,
        chain[0].size,
        chain[0].expectedContent);
    TestHelper::verifySlice(original, 0, data.size(), data);

    for (size_t i = 1; i < chain.size(); ++i) {
      SCOPED_TRACE(chain[i].debugString());
      current = current->slice(chain[i].offset, chain[i].size);
      TestHelper::verifySlice(
          *current,
          chain[i].expectedAbsoluteOffset,
          chain[i].size,
          chain[i].expectedContent);
    }
  }
}

TEST_F(MetadataBufferCacheTest, sliceCachePinOutOfRange) {
  const std::string data = "short";
  auto pin = makePin(700, data);
  nimble::MetadataBuffer original{std::move(pin)};

  NIMBLE_ASSERT_THROW(
      original.slice(0, data.size() + 1), "Slice range exceeds buffer content");

  auto sliced = original.slice(1, 3);
  NIMBLE_ASSERT_THROW(
      sliced->slice(0, 4), "Slice range exceeds buffer content");
}

} // namespace
