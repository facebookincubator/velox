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

#include "velox/dwio/nimble/serializer/ChunkedStreamPayload.h"

#include <zstd.h>

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"

namespace facebook::nimble::serde {
namespace {

std::string makeUncompressedChunk(std::string_view payload) {
  std::string chunk(kChunkHeaderSize, '\0');
  auto* pos = chunk.data();
  writeChunkHeader(
      static_cast<uint32_t>(payload.size()),
      CompressionType::Uncompressed,
      pos);
  chunk.append(payload);
  return chunk;
}

std::string makeZstdChunk(std::string_view payload) {
  const auto maxCompressedSize = ZSTD_compressBound(payload.size());
  std::string compressed(sizeof(uint32_t) + maxCompressedSize, '\0');
  auto* compressedData = compressed.data();
  encoding::writeUint32(payload.size(), compressedData);
  const auto compressedSize = ZSTD_compress(
      compressedData,
      maxCompressedSize,
      payload.data(),
      payload.size(),
      /*compressionLevel=*/1);
  NIMBLE_CHECK(!ZSTD_isError(compressedSize));
  compressed.resize(sizeof(uint32_t) + compressedSize);

  std::string chunk(kChunkHeaderSize, '\0');
  auto* pos = chunk.data();
  writeChunkHeader(
      static_cast<uint32_t>(compressed.size()), CompressionType::Zstd, pos);
  chunk.append(compressed);
  return chunk;
}

class ChunkedStreamPayloadTest : public ::testing::TestWithParam<bool> {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool(
        "chunked_stream_decoder_test");
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

TEST_P(ChunkedStreamPayloadTest, chunkOwnership) {
  struct TestCase {
    std::string_view name;
    std::vector<std::string> chunks;
    std::string expectedPayload;
    bool expectedUsesInputStorage;
  };
  const std::vector<TestCase> testCases = {
      {"single uncompressed",
       {makeUncompressedChunk("payload")},
       "payload",
       true},
      {"multiple uncompressed",
       {makeUncompressedChunk("first"), makeUncompressedChunk("second")},
       "firstsecond",
       false},
      {"single compressed",
       {makeZstdChunk("compressed payload")},
       "compressed payload",
       false},
  };
  EncodingBufferPool bufferPool{pool_.get(), /*maxCachedBuffers=*/1};
  auto* const optionalBufferPool = GetParam() ? &bufferPool : nullptr;

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    std::string chunks;
    for (const auto& chunk : testCase.chunks) {
      chunks.append(chunk);
    }

    ScopedEncodingBuffer strippedStreamBuffer{pool_.get(), optionalBufferPool};
    const auto stripped = stripChunkHeaders(chunks, strippedStreamBuffer.get());

    EXPECT_EQ(stripped, testCase.expectedPayload);
    if (testCase.expectedUsesInputStorage) {
      EXPECT_EQ(stripped.data(), chunks.data() + kChunkHeaderSize);
    } else {
      EXPECT_NE(stripped.data(), chunks.data() + kChunkHeaderSize);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    BufferPool,
    ChunkedStreamPayloadTest,
    ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "withBufferPool" : "withoutBufferPool";
    });

} // namespace
} // namespace facebook::nimble::serde
