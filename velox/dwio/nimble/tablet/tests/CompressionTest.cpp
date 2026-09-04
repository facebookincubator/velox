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

#include "velox/dwio/nimble/tablet/Compression.h"

#include <gtest/gtest.h>
#include <zstd.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"

using namespace ::facebook;

namespace {

class CompressionTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {}

  std::shared_ptr<velox::memory::MemoryPool> rootPool_{
      velox::memory::memoryManager()->addRootPool("CompressionTest")};
  std::shared_ptr<velox::memory::MemoryPool> pool_{
      rootPool_->addLeafChild("CompressionTest")};
};

// Generate a large string of random data
const std::string kLargeDataForTest = [] {
  std::mt19937 rng{std::random_device{}()};
  std::uniform_int_distribution<int> dist('A', 'Z'); // uppercase letters

  size_t kLargeTestDataSize = 2001;
  std::string data;
  data.reserve(kLargeTestDataSize);

  for (auto i = 0; i < kLargeTestDataSize; ++i) {
    auto randomChar = dist(rng);
    data.push_back(randomChar);
  }
  return data;
}();

TEST_F(CompressionTest, compressRoundTrip) {
  struct TestParam {
    size_t size;
    char fillChar;
    std::string debugString() const {
      return fmt::format("size {}, fillChar {}", size, fillChar);
    }
  };
  for (const auto& testData : std::vector<TestParam>{
           {0, '\0'}, {100, 'A'}, {2'001, 'B'}, {100'000, 'X'}}) {
    SCOPED_TRACE(testData.debugString());
    const std::string data(testData.size, testData.fillChar);
    auto compressed = nimble::ZstdCompression::compress(data, pool_.get());
    ASSERT_TRUE(compressed.has_value());

    std::string_view compressedView{
        compressed.value()->as<char>(), compressed.value()->size()};
    auto decompressed =
        nimble::ZstdCompression::uncompress(compressedView, pool_.get());

    std::string_view result{decompressed->as<char>(), decompressed->size()};
    EXPECT_EQ(result, data);
  }
}

TEST_F(CompressionTest, compressIncompressible) {
  std::string data;
  data.reserve(256);
  for (int i = 0; i < 256; ++i) {
    data.push_back(static_cast<char>(i));
  }

  auto compressed = nimble::ZstdCompression::compress(data, pool_.get());
  EXPECT_FALSE(compressed.has_value());
}

TEST_F(CompressionTest, invalidCompressionLevel) {
  const std::string data(1'000, 'A');
  NIMBLE_ASSERT_THROW(
      nimble::ZstdCompression::compress(
          data, pool_.get(), ZSTD_maxCLevel() + 1),
      "Compression level above maximum");
  NIMBLE_ASSERT_THROW(
      nimble::ZstdCompression::compress(
          data, pool_.get(), ZSTD_minCLevel() - 1),
      "Compression level below minimum");
}

} // namespace
