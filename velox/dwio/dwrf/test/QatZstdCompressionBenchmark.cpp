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

#include "velox/common/base/VeloxException.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/common/encryption/TestProvider.h"
#include "velox/dwio/dwrf/common/Compression.h"
#include "velox/dwio/dwrf/common/wrap/dwrf-proto-wrapper.h"
#include "velox/dwio/dwrf/test/OrcTest.h"

#include <folly/Benchmark.h>
#include <folly/Random.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include <algorithm>

using namespace ::testing;
using namespace facebook::velox::common;
using namespace facebook::velox::dwio;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::dwio::common::encryption;
using namespace facebook::velox::dwio::common::encryption::test;
using namespace facebook::velox::dwrf;
using namespace facebook::velox::memory;
using namespace folly;
using facebook::velox::VeloxException;

const int32_t DEFAULT_MEM_STREAM_SIZE = 1024 * 1024 * 2; // 2M

class TestBufferPool : public CompressionBufferPool {
 public:
  TestBufferPool(MemoryPool& pool, uint64_t blockSize)
      : buffer_{std::make_unique<DataBuffer<char>>(
            pool,
            blockSize + PAGE_HEADER_SIZE)} {}

  std::unique_ptr<DataBuffer<char>> getBuffer(uint64_t /* unused */) override {
    VELOX_CHECK_NOT_NULL(buffer_);
    return std::move(buffer_);
  }

  void returnBuffer(std::unique_ptr<DataBuffer<char>> buffer) override {
    VELOX_CHECK_NULL(buffer_);
    VELOX_CHECK_NOT_NULL(buffer);
    buffer_ = std::move(buffer);
  }

 private:
  std::unique_ptr<DataBuffer<char>> buffer_;
};

void generateRandomData(char* data, size_t size, bool letter) {
  for (size_t i = 0; i < size; ++i) {
    if (letter) {
      bool capitalized = folly::Random::rand32() % 2 == 0;
      data[i] = capitalized
          ? static_cast<char>('A' + folly::Random::rand32() % 26)
          : static_cast<char>('a' + folly::Random::rand32() % 26);
    } else {
      data[i] = static_cast<char>(folly::Random::rand32() % 256);
    }
  }
}

void compress(
    CompressionKind kind,
    FileSink& sink,
    uint64_t block,
    MemoryPool& pool,
    const char* data,
    size_t dataSize,
    const Encrypter* encrypter) {
  TestBufferPool bufferPool(pool, block);
  DataBufferHolder holder{
      pool, block, 0, DEFAULT_PAGE_GROW_RATIO, std::addressof(sink)};
  Config config;
  config.set<uint32_t>(Config::COMPRESSION_THRESHOLD, 128);
  std::unique_ptr<BufferedOutputStream> compressStream =
      createCompressor(kind, bufferPool, holder, config, encrypter);

  size_t pos = 0;
  char* compressBuffer;
  int32_t compressBufferSize = 0;
  while (dataSize > 0 &&
         compressStream->Next(
             reinterpret_cast<void**>(&compressBuffer), &compressBufferSize)) {
    size_t copy_size =
        std::min(static_cast<size_t>(compressBufferSize), dataSize);
    memcpy(compressBuffer, data + pos, copy_size);

    if (copy_size == dataSize) {
      compressStream->BackUp(
          compressBufferSize - static_cast<int32_t>(dataSize));
    }

    pos += copy_size;
    dataSize -= copy_size;
  }

  compressStream->flush();
}

BENCHMARK(compressZstd) {
  folly::BenchmarkSuspender suspender;

  auto pool = addDefaultLeafMemoryPool();
  MemorySink memSink(DEFAULT_MEM_STREAM_SIZE, {.pool = pool.get()});

  uint64_t block = 1024 * 4;
  auto dataSize = 1 * 1024 * 1024;
  char testData[dataSize];
  generateRandomData(testData, dataSize, true);

  suspender.dismiss();

  compress(
      CompressionKind_ZSTD,
      memSink,
      block,
      *pool,
      testData,
      sizeof(testData),
      NULL);
}

int main(int argc, char** argv) {
  folly::init(&argc, &argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  folly::runBenchmarks();
  return 0;
}