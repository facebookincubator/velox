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

#include "velox/dwio/nimble/encodings/legacy/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"
#include "velox/dwio/nimble/reader/ChunkedStreamDecoder.h"
#include "velox/dwio/nimble/writer/ChunkedStreamWriter.h"

using namespace facebook;

namespace {

DEFINE_uint32(seed, 0, "Override test seed.");
DEFINE_uint32(
    min_chunk_count,
    2,
    "Minumum chunks to use in multi-chunk tests.");
DEFINE_uint32(
    max_chunk_count,
    10,
    "Maximum chunks to use in multi-chunk tests.");
DEFINE_uint32(max_chunk_size, 10, "Maximum items to store in each chunk.");

DEFINE_uint32(
    reset_iterations,
    2,
    "How many iterations to run, resetting the decoder in between.");

size_t paddedBitmapByteCount(uint32_t count) {
  return velox::bits::nbytes(count) + velox::simd::kPadding;
}

template <typename T, typename RNG>
nimble::Vector<T>
generateData(RNG& rng, velox::memory::MemoryPool& memoryPool, size_t size) {
  nimble::Vector<T> data{&memoryPool, size};
  for (auto i = 0; i < size; ++i) {
    uint64_t value =
        folly::Random::rand64(std::numeric_limits<uint64_t>::max(), rng);
    data[i] = *reinterpret_cast<T*>(&value);
    LOG(INFO) << "Data[" << i << "]: " << data[i];
  }
  return data;
}

template <typename RNG>
nimble::Vector<std::string_view> generateStringData(
    RNG& rng,
    velox::memory::MemoryPool& memoryPool,
    size_t size,
    nimble::Buffer& buffer) {
  nimble::Vector<std::string_view> data{&memoryPool, size};
  for (auto i = 0; i < size; ++i) {
    uint64_t length = folly::Random::rand32(100, rng);
    auto content = buffer.reserve(length);
    for (auto j = 0; j < length; ++j) {
      content[j] = folly::Random::rand32(std::numeric_limits<char>::max(), rng);
    }
    data[i] = std::string_view(content, length);
    LOG(INFO) << "Data[" << i << "]: " << data[i].length();
  }
  return data;
}

template <typename RNG>
nimble::Vector<bool>
generateNullData(RNG& rng, velox::memory::MemoryPool& memoryPool, size_t size) {
  nimble::Vector<bool> data{&memoryPool, size};
  for (auto i = 0; i < size; ++i) {
    data[i] = static_cast<bool>(folly::Random::oneIn(2, rng));
    LOG(INFO) << "Null[" << i << "]: " << data[i];
  }
  return data;
}

class TestStreamLoader : public nimble::StreamLoader {
 public:
  explicit TestStreamLoader(std::string stream) : stream_{std::move(stream)} {}

  const std::string_view getStream() const override {
    return stream_;
  }

 private:
  const std::string stream_;
};

template <typename E>
std::unique_ptr<nimble::StreamLoader> createStream(
    velox::memory::MemoryPool& memoryPool,
    const std::vector<nimble::Vector<typename E::cppDataType>>& values,
    const std::vector<std::optional<nimble::Vector<bool>>>& nulls,
    nimble::CompressionParams compressionParams = {
        .type = nimble::CompressionType::Uncompressed}) {
  NIMBLE_CHECK_EQ(values.size(), nulls.size(), "Data and nulls size mismatch.");

  std::string stream;

  for (auto i = 0; i < values.size(); ++i) {
    nimble::Buffer buffer{memoryPool};
    nimble::ChunkedStreamWriter writer{buffer, compressionParams};
    std::vector<std::string_view> segments;

    if (nulls[i].has_value()) {
      // Remove null entries from data
      nimble::Vector<typename E::cppDataType> data(
          &memoryPool,
          std::accumulate(
              nulls[i]->data(), nulls[i]->data() + nulls[i]->size(), 0UL));
      uint32_t offset = 0;
      for (auto j = 0; j < nulls[i]->size(); ++j) {
        if (nulls[i].value()[j]) {
          data[offset++] = values[i][j];
        }
      }
      segments = writer.encode(
          nimble::test::Encoder<E>::encodeNullable(
              buffer, data, nulls[i].value()));
    } else {
      segments =
          writer.encode(nimble::test::Encoder<E>::encode(buffer, values[i]));
    }

    for (const auto& segment : segments) {
      stream += segment;
    }
  }

  return std::make_unique<TestStreamLoader>(std::move(stream));
}

template <typename T>
T getValue(const std::vector<nimble::Vector<T>>& data, size_t offset) {
  size_t i = 0;
  while (data[i].size() <= offset) {
    offset -= data[i].size();
    ++i;
  }

  return data[i][offset];
}

template <typename T>
bool getNullValue(
    const std::vector<nimble::Vector<T>>& data,
    const std::vector<std::optional<nimble::Vector<bool>>>& nulls,
    size_t offset) {
  size_t i = 0;
  while (data[i].size() <= offset) {
    offset -= data[i].size();
    ++i;
  }

  if (!nulls[i].has_value()) {
    return true;
  }

  return nulls[i].value()[offset];
}

template <typename T>
void test(
    bool multipleChunks,
    bool hasNulls,
    bool skips,
    bool scatter,
    bool compress,
    bool optimizeStringBufferHandling) {
  uint32_t seed = FLAGS_seed == 0 ? folly::Random::rand32() : FLAGS_seed;
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng{seed};

  auto memoryPool = velox::memory::deprecatedAddDefaultLeafMemoryPool();

  auto chunkCount = multipleChunks
      ? folly::Random::rand32(FLAGS_min_chunk_count, FLAGS_max_chunk_count, rng)
      : 1;
  std::vector<nimble::Vector<T>> data;
  std::vector<std::optional<nimble::Vector<bool>>> nulls(chunkCount);

  nimble::Buffer buffer{*memoryPool};
  size_t totalSize = 0;
  for (auto i = 0; i < chunkCount; ++i) {
    auto size = folly::Random::rand32(1, FLAGS_max_chunk_size, rng);
    LOG(INFO) << "Chunk: " << i << ", Size: " << size;
    if constexpr (std::is_same_v<T, std::string_view>) {
      data.push_back(generateStringData(rng, *memoryPool, size, buffer));
    } else {
      data.push_back(generateData<T>(rng, *memoryPool, size));
    }
    if (hasNulls && folly::Random::oneIn(2, rng)) {
      nulls[i] = generateNullData(rng, *memoryPool, size);
    }
    totalSize += size;
  }

  auto streamLoader = createStream<nimble::TrivialEncoding<T>>(
      *memoryPool,
      data,
      nulls,
      compress
          ? nimble::CompressionParams{.type = nimble::CompressionType::Zstd}
          : nimble::CompressionParams{
                .type = nimble::CompressionType::Uncompressed});

  nimble::ChunkedStreamDecoder decoder{
      *memoryPool,
      std::make_unique<nimble::InMemoryChunkedStream>(
          *memoryPool, std::move(streamLoader)),
      optimizeStringBufferHandling ? [](velox::memory::MemoryPool& pool,
         std::string_view data, std::function<void*(uint32_t)> stringBufferFactory) -> std::unique_ptr<nimble::Encoding> {
        return nimble::EncodingFactory().create(
            pool, data, std::move(stringBufferFactory), nimble::Encoding::Options{});
      } : [](velox::memory::MemoryPool& pool,
         std::string_view data, std::function<void*(uint32_t)> stringBufferFactory) -> std::unique_ptr<nimble::Encoding> {
        return nimble::legacy::EncodingFactory().create(
            pool, data, std::move(stringBufferFactory), nimble::Encoding::Options{});
      },
      optimizeStringBufferHandling,
      /* metricLogger */ {}};

  for (auto batchSize = 1; batchSize <= totalSize; ++batchSize) {
    for (auto iteration = 0; iteration < FLAGS_reset_iterations; ++iteration) {
      LOG(INFO) << "batchSize: " << batchSize;
      auto offset = 0;
      while (offset < totalSize) {
        const auto outputSize = folly::Random::oneIn(5, rng)
            ? 0
            : std::min<size_t>(batchSize, totalSize - offset);
        if (skips && folly::Random::oneIn(2, rng)) {
          LOG(INFO) << "Skipping " << outputSize;
          decoder.skip(outputSize);
        } else {
          std::vector<uint8_t> scatterMap;
          std::optional<velox::bits::Bitmap> scatterBitmap;
          auto scatterSize = outputSize;
          if (scatter) {
            scatterSize =
                folly::Random::rand32(outputSize, outputSize * 2, rng);
            scatterMap.resize(paddedBitmapByteCount(scatterSize));
            std::vector<size_t> indices(scatterSize);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), rng);
            for (auto i = 0; i < outputSize; ++i) {
              velox::bits::setBit(
                  reinterpret_cast<uint8_t*>(scatterMap.data()), indices[i]);
            }
            for (auto i = 0; i < scatterSize; ++i) {
              LOG(INFO) << "Scatter[" << i << "]: "
                        << velox::bits::isBitSet(
                               reinterpret_cast<const uint8_t*>(
                                   scatterMap.data()),
                               i);
            }
            scatterBitmap.emplace(scatterMap.data(), scatterSize);
          }

          std::vector<velox::BufferPtr> stringBuffers;

          if (hasNulls || scatter) {
            std::vector<T> output(scatterSize);
            std::vector<uint8_t> outputNulls(
                paddedBitmapByteCount(scatterSize));
            uint32_t count = 0;
            const auto nonNullCount = decoder.next(
                outputSize,
                output.data(),
                stringBuffers,
                [&]() {
                  ++count;
                  return outputNulls.data();
                },
                scatterBitmap.has_value() ? &scatterBitmap.value() : nullptr);

            LOG(INFO) << "offset: " << offset << ", batchSize: " << batchSize
                      << ", outputSize: " << outputSize
                      << ", scatterSize: " << scatterSize
                      << ", nonNullCount: " << nonNullCount;
            if (nonNullCount != scatterSize || (outputSize == 0 && scatter)) {
              EXPECT_EQ(1, count);
              EXPECT_EQ(
                  velox::bits::countBits(
                      reinterpret_cast<const uint64_t*>(outputNulls.data()),
                      0,
                      scatterSize),
                  nonNullCount);
            } else {
              EXPECT_EQ(0, count);
            }

            if (nonNullCount == scatterSize) {
              for (auto i = 0; i < scatterSize; ++i) {
                EXPECT_EQ(getValue(data, offset + i), output[i])
                    << "Index: " << i << ", Offset: " << offset;
              }
            } else {
              if (scatter) {
                size_t scatterOffset = 0;
                for (auto i = 0; i < scatterSize; ++i) {
                  if (!velox::bits::isBitSet(
                          reinterpret_cast<const uint8_t*>(scatterMap.data()),
                          i)) {
                    EXPECT_FALSE(
                        velox::bits::isBitSet(
                            reinterpret_cast<const uint8_t*>(
                                outputNulls.data()),
                            i));
                  } else {
                    const auto isNotNull = velox::bits::isBitSet(
                        reinterpret_cast<const uint8_t*>(outputNulls.data()),
                        i);
                    EXPECT_EQ(
                        getNullValue(data, nulls, offset + scatterOffset),
                        isNotNull)
                        << "Index: " << i << ", Offset: " << offset
                        << ", scatterOffset: " << scatterOffset;
                    if (isNotNull) {
                      EXPECT_EQ(
                          getValue(data, offset + scatterOffset), output[i])
                          << "Index: " << i << ", Offset: " << offset
                          << ", scatterOffset: " << scatterOffset;
                    }
                    ++scatterOffset;
                  }
                }
              } else {
                for (auto i = 0; i < scatterSize; ++i) {
                  const auto isNotNull = velox::bits::isBitSet(
                      reinterpret_cast<const uint8_t*>(outputNulls.data()), i);
                  EXPECT_EQ(getNullValue(data, nulls, offset + i), isNotNull)
                      << "Index: " << i << ", Offset: " << offset;
                  if (isNotNull) {
                    EXPECT_EQ(getValue(data, offset + i), output[i])
                        << "Index: " << i << ", Offset: " << offset;
                  }
                }
              }
            }
          } else {
            std::vector<T> output(outputSize);
            const auto actualOutputSize =
                decoder.next(outputSize, output.data(), stringBuffers);
            EXPECT_EQ(outputSize, actualOutputSize);

            for (auto i = 0; i < outputSize; ++i) {
              EXPECT_EQ(getValue(data, offset + i), output[i])
                  << "Index: " << i << ", Offset: " << offset;
            }
          }
        }

        offset += outputSize;
      }

      decoder.reset();
    }
  }
}
} // namespace

class ChunkedStreamDecoderTests
    : public ::testing::Test,
      public ::testing::WithParamInterface<
          std::tuple<bool, bool, bool, bool, bool, bool>> {};

TEST_P(ChunkedStreamDecoderTests, decode) {
  const auto
      [multipleChunks,
       hasNulls,
       skip,
       scatter,
       compress,
       optimizeStringBufferHandling] = GetParam();
  for (int i = 0; i < 3; ++i) {
    LOG(INFO) << "Interation: " << i << ", Multiple Chunks: " << multipleChunks
              << ", Has Nulls: " << hasNulls << ", Skips: " << skip
              << ", Scatter: " << scatter
              << ", Optimize String Buffer Handling: "
              << optimizeStringBufferHandling;
    test<int32_t>(
        multipleChunks,
        hasNulls,
        skip,
        scatter,
        compress,
        optimizeStringBufferHandling);
  }
}

TEST_P(ChunkedStreamDecoderTests, decodeStrings) {
  const auto
      [multipleChunks,
       hasNulls,
       skip,
       scatter,
       compress,
       optimizeStringBufferHandling] = GetParam();
  for (int i = 0; i < 3; ++i) {
    LOG(INFO) << "Interation: " << i << ", Multiple Chunks: " << multipleChunks
              << ", Has Nulls: " << hasNulls << ", Skips: " << skip
              << ", Scatter: " << scatter << ", Compress: " << compress
              << ", Optimize String Buffer Handling: "
              << optimizeStringBufferHandling;
    test<std::string_view>(
        multipleChunks,
        hasNulls,
        skip,
        scatter,
        compress,
        optimizeStringBufferHandling);
  }
}

INSTANTIATE_TEST_CASE_P(
    ChunkedStreamDecoderTestSuite,
    ChunkedStreamDecoderTests,
    testing::Combine(
        testing::Values(false, true),
        testing::Values(false, true),
        testing::Values(false, true),
        testing::Values(false, true),
        testing::Values(false, true),
        testing::Values(false, true)));
