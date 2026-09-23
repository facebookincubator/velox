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

#include <gtest/gtest.h>

#include "velox/buffer/BufferPool.h"
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/tests/EncodingViewTestUtils.h"

using namespace facebook;

using HuffmanEncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(HuffmanEncodingViewTest, readsAcrossCheckpoints) {
  nimble::Vector<int32_t> values{pool_.get()};
  values.reserve(1025);
  for (uint32_t row = 0; row < 1025; ++row) {
    values.push_back(row % 13 == 0 ? -7 : static_cast<int32_t>(row % 5));
  }
  expectReads<nimble::HuffmanEncoding<int32_t>>(
      values, {1024, 0, 255, 256, 257, 511, 512, 768});
}

TEST_F(HuffmanEncodingViewTest, readsAllIntegralTypes) {
  expectReads<nimble::HuffmanEncoding<int8_t>>(
      makeVector<int8_t>({-2, -1, 0, 1, 2}), {4, 0, 2});
  expectReads<nimble::HuffmanEncoding<uint8_t>>(
      makeVector<uint8_t>({0, 1, 2, 3, 4}), {3, 1, 4});
  expectReads<nimble::HuffmanEncoding<int16_t>>(
      makeVector<int16_t>({-100, 0, 100, -100}), {3, 2, 0});
  expectReads<nimble::HuffmanEncoding<uint16_t>>(
      makeVector<uint16_t>({0, 100, 1000, 100}), {2, 0, 3});
  expectReads<nimble::HuffmanEncoding<int32_t>>(
      makeVector<int32_t>({-1000, 0, 1000, -1000}), {1, 3, 2});
  expectReads<nimble::HuffmanEncoding<uint32_t>>(
      makeVector<uint32_t>({0, 1000, 100000, 1000}), {2, 1, 0});
  expectReads<nimble::HuffmanEncoding<int64_t>>(
      makeVector<int64_t>({-100000, 0, 100000, -100000}), {3, 0, 2});
  expectReads<nimble::HuffmanEncoding<uint64_t>>(
      makeVector<uint64_t>({0, 100000, 1000000, 100000}), {2, 0, 3});
}

TEST_F(HuffmanEncodingViewTest, concurrent) {
  expectConcurrentReads<nimble::HuffmanEncoding<uint32_t>>(
      randomNarrowUnsigned<uint32_t>(/*seed=*/41),
      randomizedPositions(/*seed=*/42));
}

TEST_F(HuffmanEncodingViewTest, reusesMetadataBuffers) {
  auto values = randomNarrowUnsigned<uint32_t>(/*seed=*/43);
  const auto encoded =
      nimble::test::Encoder<nimble::HuffmanEncoding<uint32_t>>::encode(
          *buffer_, values);
  const std::vector<char> encodedStorage{encoded.begin(), encoded.end()};

  auto decodePool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  velox::BufferPool bufferPool{velox::BufferPool::kDefaultCapacity};
  const nimble::Encoding::Options options{.bufferPool = &bufferPool};

  auto readAndRelease = [&]() {
    auto view = nimble::createEncodingView(
        {encodedStorage.data(), encodedStorage.size()},
        decodePool.get(),
        options);
    uint32_t value;
    view->readAt(768, &value);
    EXPECT_EQ(value, values[768]);
  };

  readAndRelease();
  const auto numAllocsAfterFirstView = decodePool->stats().numAllocs;
  EXPECT_GT(bufferPool.size(), 0);

  readAndRelease();
  EXPECT_EQ(decodePool->stats().numAllocs, numAllocsAfterFirstView);
}
