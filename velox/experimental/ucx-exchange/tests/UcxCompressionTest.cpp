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
#include "velox/experimental/ucx-exchange/UcxCompression.h"
#include "velox/experimental/ucx-exchange/UcxColumnCodec.h"

#include <chrono>
#include <limits>
#include <random>

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <rmm/cuda_stream.hpp>

namespace facebook::velox::ucx_exchange {
namespace {

// Uploads host bytes, round-trips through compress/decompress, checks
// byte-exactness, and reports {ratio, encode GB/s, decode GB/s}.
struct RoundTripResult {
  bool compressed;
  bool attempted;
  std::size_t inputBytes;
  std::size_t candidateBytes;
  double ratio;
  double encodeGbps;
  double decodeGbps;
};

RoundTripResult roundTrip(const std::vector<uint8_t>& host) {
  rmm::cuda_stream stream;
  rmm::device_buffer input(host.data(), host.size(), stream.view());
  cudaStreamSynchronize(stream.value());

  auto timedCompress = [&] {
    auto start = std::chrono::steady_clock::now();
    auto result = compressBlob(input.data(), input.size(), stream.view());
    cudaStreamSynchronize(stream.value());
    auto seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    return std::make_pair(std::move(result), seconds);
  };
  // Warm-up then timed run (kernel JIT + pool warm-up dominate run one).
  timedCompress();
  auto [result, encodeSeconds] = timedCompress();

  RoundTripResult out{};
  out.compressed = result.used;
  out.attempted = result.stats.attempted;
  out.inputBytes = result.stats.inputBytes;
  out.candidateBytes = result.stats.candidateBytes;
  if (!result.used) {
    return out;
  }
  std::size_t compressedBytes = 0;
  for (auto size : result.segSizes) {
    compressedBytes += size;
  }
  out.ratio = static_cast<double>(host.size()) / compressedBytes;
  out.encodeGbps = host.size() / encodeSeconds / 1e9;

  auto start = std::chrono::steady_clock::now();
  auto decoded = decompressBlob(
      result.data.data(), result.segSizes, host.size(), stream.view());
  cudaStreamSynchronize(stream.value());
  out.decodeGbps = host.size() /
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count() /
      1e9;

  std::vector<uint8_t> back(host.size());
  cudaMemcpy(back.data(), decoded.data(), back.size(), cudaMemcpyDeviceToHost);
  EXPECT_EQ(back, host) << "round-trip not byte-exact";
  return out;
}

std::vector<uint8_t> skewedBytes(std::size_t size, uint32_t seed) {
  // Geometric-ish byte distribution: what packed decimal mantissa planes and
  // dictionary codes look like after transforms.
  std::mt19937 gen(seed);
  std::geometric_distribution<int> dist(0.25);
  std::vector<uint8_t> data(size);
  for (auto& byte : data) {
    byte = static_cast<uint8_t>(std::min(dist(gen), 255));
  }
  return data;
}

TEST(UcxCompressionTest, tinyInputSkipped) {
  std::vector<uint8_t> host(1024, 7);
  auto result = roundTrip(host);
  EXPECT_FALSE(result.compressed);
  EXPECT_FALSE(result.attempted);
  EXPECT_EQ(result.inputBytes, host.size());
  EXPECT_EQ(result.candidateBytes, 0);
}

TEST(UcxCompressionTest, incompressibleSkipped) {
  std::mt19937 gen(42);
  std::vector<uint8_t> host(8u << 20);
  for (auto& byte : host) {
    byte = static_cast<uint8_t>(gen());
  }
  auto result = roundTrip(host);
  EXPECT_FALSE(result.compressed) << "random bytes must not be compressed";
  EXPECT_TRUE(result.attempted);
  EXPECT_EQ(result.inputBytes, host.size());
  EXPECT_GT(result.candidateBytes, 0);
}

TEST(UcxCompressionTest, skewedSingleSegment) {
  auto result = roundTrip(skewedBytes(8u << 20, 1));
  EXPECT_TRUE(result.compressed);
  EXPECT_TRUE(result.attempted);
  EXPECT_GT(result.candidateBytes, 0);
  EXPECT_LT(result.candidateBytes, result.inputBytes);
  EXPECT_GT(result.ratio, 1.5);
}

TEST(UcxCompressionTest, skewedMultiSegmentUnaligned) {
  // Crosses several 32 MiB segments with a ragged tail.
  auto result = roundTrip(skewedBytes((96u << 20) + 12345, 2));
  EXPECT_TRUE(result.compressed);
  EXPECT_GT(result.ratio, 1.5);
  // Report throughput as useful context without making it a hardware-specific
  // test requirement.
  LOG(INFO) << "whole-blob 96MiB: ratio=" << result.ratio
            << " enc=" << result.encodeGbps << " GB/s dec=" << result.decodeGbps
            << " GB/s";
}

TEST(UcxCompressionTest, constantBytes) {
  std::vector<uint8_t> host(40u << 20, 0);
  auto result = roundTrip(host);
  EXPECT_TRUE(result.compressed);
  EXPECT_GT(result.ratio, 25.0); // byte-rANS per-block framing floor
}

TEST(UcxCompressionTest, dictionaryPforDescriptorRoundTrip) {
  PackedCompressResult packed;
  EncodedRegion region;
  region.blobOffset = 128;
  region.rawBytes = 8u << 20;
  region.codec = RegionCodec::kDictPfor;
  region.elemWidth = 8;
  region.base = -9'000;
  region.dictionary = {0, 3, 255, 256, 32'768, 65'535, 42};
  packed.regions.push_back(region);

  std::vector<int64_t> descriptor;
  serializeRegions(packed, 9u << 20, descriptor);
  ASSERT_EQ(descriptor.front(), kPerColumnMagic);

  std::vector<EncodedRegion> decoded;
  std::size_t uncompressedBytes = 0;
  ASSERT_TRUE(deserializeRegions(descriptor, decoded, uncompressedBytes));
  ASSERT_EQ(uncompressedBytes, 9u << 20);
  ASSERT_EQ(decoded.size(), 1);
  EXPECT_EQ(decoded.front().codec, RegionCodec::kDictPfor);
  EXPECT_EQ(decoded.front().base, region.base);
  EXPECT_TRUE(decoded.front().segSizes.empty());
  EXPECT_EQ(decoded.front().dictionary, region.dictionary);
}

TEST(UcxCompressionTest, frequencyPforDescriptorRoundTrip) {
  PackedCompressResult packed;
  EncodedRegion region;
  region.blobOffset = 512;
  region.rawBytes = 16u << 20;
  region.codec = RegionCodec::kFreqPfor;
  region.elemWidth = 8;
  region.base = -32'000;
  region.segSizes = {12'345, 6'789};
  region.dictionarySize = 12'345;
  region.exceptionCount = 17;
  packed.regions.push_back(region);

  std::vector<int64_t> descriptor;
  serializeRegions(packed, 20u << 20, descriptor);
  ASSERT_EQ(descriptor.front(), kPerColumnMagic);

  std::vector<EncodedRegion> decoded;
  std::size_t uncompressedBytes = 0;
  ASSERT_TRUE(deserializeRegions(descriptor, decoded, uncompressedBytes));
  ASSERT_EQ(uncompressedBytes, 20u << 20);
  ASSERT_EQ(decoded.size(), 1);
  EXPECT_EQ(decoded.front().codec, RegionCodec::kFreqPfor);
  EXPECT_EQ(decoded.front().base, region.base);
  EXPECT_EQ(decoded.front().segSizes, region.segSizes);
  EXPECT_EQ(decoded.front().dictionarySize, region.dictionarySize);
  EXPECT_EQ(decoded.front().exceptionCount, region.exceptionCount);
}
TEST(UcxCompressionTest, deltaFrequencyPforDescriptorRoundTrip) {
  PackedCompressResult packed;
  EncodedRegion region;
  region.blobOffset = 1'024;
  region.rawBytes = 32u << 20;
  region.codec = RegionCodec::kDeltaFreqPfor;
  region.elemWidth = 4;
  region.base = -31'991;
  region.first = 9'876'543;
  region.segSizes = {54'321, 9'876};
  region.dictionarySize = 8'765;
  region.exceptionCount = 29;
  packed.regions.push_back(region);

  std::vector<int64_t> descriptor;
  serializeRegions(packed, 40u << 20, descriptor);
  ASSERT_EQ(descriptor.front(), kPerColumnMagic);

  std::vector<EncodedRegion> decoded;
  std::size_t uncompressedBytes = 0;
  ASSERT_TRUE(deserializeRegions(descriptor, decoded, uncompressedBytes));
  ASSERT_EQ(uncompressedBytes, 40u << 20);
  ASSERT_EQ(decoded.size(), 1);
  EXPECT_EQ(decoded.front().codec, RegionCodec::kDeltaFreqPfor);
  EXPECT_EQ(decoded.front().base, region.base);
  EXPECT_EQ(decoded.front().first, region.first);
  EXPECT_EQ(decoded.front().segSizes, region.segSizes);
  EXPECT_EQ(decoded.front().dictionarySize, region.dictionarySize);
  EXPECT_EQ(decoded.front().exceptionCount, region.exceptionCount);
}

TEST(UcxCompressionTest, dictionaryPforGpuDecodeIsByteExact) {
  constexpr uint32_t kValues = 1u << 20;
  constexpr uint32_t kDistinct = 50;
  constexpr int64_t kBase = 100;
  std::vector<uint8_t> ranks(kValues);
  std::vector<int64_t> expected(kValues);
  for (uint32_t i = 0; i < kValues; ++i) {
    ranks[i] = static_cast<uint8_t>(i % kDistinct);
    expected[i] = kBase + static_cast<int64_t>(ranks[i]) * 100;
  }

  EncodedRegion region;
  region.rawBytes = expected.size() * sizeof(int64_t);
  region.codec = RegionCodec::kDictPfor;
  region.elemWidth = sizeof(int64_t);
  region.base = kBase;
  for (uint32_t rank = 0; rank < kDistinct; ++rank) {
    region.dictionary.push_back(static_cast<uint16_t>(rank * 100));
  }

  rmm::cuda_stream stream;
  rmm::device_buffer encoded(ranks.data(), ranks.size(), stream.view());
  auto decoded = decompressPacked(
      encoded.data(), {region}, region.rawBytes, stream.view());
  std::vector<int64_t> actual(kValues);
  ASSERT_EQ(
      cudaMemcpyAsync(
          actual.data(),
          decoded.data(),
          region.rawBytes,
          cudaMemcpyDeviceToHost,
          stream.value()),
      cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream.value()), cudaSuccess);
  EXPECT_EQ(actual, expected);
}

TEST(UcxCompressionTest, frequencyPforGpuDecodeAndPatchIsByteExact) {
  constexpr uint32_t kValues = 1u << 20;
  constexpr uint32_t kDistinct = 200;
  constexpr int64_t kBase = -10'000;
  const auto aligned = [](std::size_t bytes) {
    return (bytes + 15) & ~static_cast<std::size_t>(15);
  };

  std::vector<uint16_t> dictionary(kDistinct);
  for (uint32_t rank = 0; rank < kDistinct; ++rank) {
    dictionary[rank] = static_cast<uint16_t>(rank * 251);
  }
  std::vector<uint8_t> ranks(kValues);
  std::vector<int64_t> expected(kValues);
  for (uint32_t i = 0; i < kValues; ++i) {
    ranks[i] = static_cast<uint8_t>((i * 17) % kDistinct);
    expected[i] = kBase + dictionary[ranks[i]];
  }
  std::vector<uint32_t> exceptionPositions{7, 99'999, kValues - 1};
  std::vector<int64_t> exceptionValues{
      std::numeric_limits<int64_t>::min() + 19,
      9'000'000'000LL,
      std::numeric_limits<int64_t>::max() - 23};
  for (std::size_t i = 0; i < exceptionPositions.size(); ++i) {
    ranks[exceptionPositions[i]] = 0;
    expected[exceptionPositions[i]] = exceptionValues[i];
  }

  rmm::cuda_stream stream;
  rmm::device_buffer rankDevice(ranks.data(), ranks.size(), stream.view());
  auto compressed =
      compressBlob(rankDevice.data(), ranks.size(), stream.view(), 0.0, 1);
  ASSERT_TRUE(compressed.used);
  ASSERT_EQ(compressed.segSizes.size(), 1);

  const std::size_t rankBytes = aligned(compressed.segSizes.front());
  const std::size_t dictionaryBytes = dictionary.size() * sizeof(uint16_t);
  const std::size_t positionBytes =
      exceptionPositions.size() * sizeof(uint32_t);
  const std::size_t valueBytes = exceptionValues.size() * sizeof(int64_t);
  const std::size_t totalBytes = rankBytes + aligned(dictionaryBytes) +
      aligned(positionBytes) + aligned(valueBytes);
  rmm::device_buffer wire(totalBytes, stream.view());
  std::size_t offset = 0;
  ASSERT_EQ(
      cudaMemcpyAsync(
          wire.data(),
          compressed.data.data(),
          rankBytes,
          cudaMemcpyDeviceToDevice,
          stream.value()),
      cudaSuccess);
  offset += rankBytes;
  ASSERT_EQ(
      cudaMemcpyAsync(
          static_cast<uint8_t*>(wire.data()) + offset,
          dictionary.data(),
          dictionaryBytes,
          cudaMemcpyHostToDevice,
          stream.value()),
      cudaSuccess);
  offset += aligned(dictionaryBytes);
  ASSERT_EQ(
      cudaMemcpyAsync(
          static_cast<uint8_t*>(wire.data()) + offset,
          exceptionPositions.data(),
          positionBytes,
          cudaMemcpyHostToDevice,
          stream.value()),
      cudaSuccess);
  offset += aligned(positionBytes);
  ASSERT_EQ(
      cudaMemcpyAsync(
          static_cast<uint8_t*>(wire.data()) + offset,
          exceptionValues.data(),
          valueBytes,
          cudaMemcpyHostToDevice,
          stream.value()),
      cudaSuccess);

  EncodedRegion region;
  region.rawBytes = expected.size() * sizeof(int64_t);
  region.codec = RegionCodec::kFreqPfor;
  region.elemWidth = sizeof(int64_t);
  region.base = kBase;
  region.segSizes.assign(
      compressed.segSizes.begin(), compressed.segSizes.end());
  region.dictionarySize = dictionary.size();
  region.exceptionCount = exceptionPositions.size();

  auto decoded =
      decompressPacked(wire.data(), {region}, region.rawBytes, stream.view());
  std::vector<int64_t> actual(kValues);
  ASSERT_EQ(
      cudaMemcpyAsync(
          actual.data(),
          decoded.data(),
          region.rawBytes,
          cudaMemcpyDeviceToHost,
          stream.value()),
      cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream.value()), cudaSuccess);
  EXPECT_EQ(actual, expected);
}

TEST(UcxCompressionTest, deltaFrequencyPforGpuDecodeAndPatchIsByteExact) {
  constexpr uint32_t kValues = 1u << 20;
  constexpr int64_t kFirst = 1'000'000;
  const auto aligned = [](std::size_t bytes) {
    return (bytes + 15) & ~static_cast<std::size_t>(15);
  };

  // Zigzag codes for deltas {0, +1, -1, +2, -2}.
  const std::vector<uint16_t> dictionary{0, 2, 1, 4, 3};
  std::vector<uint8_t> ranks(kValues);
  std::vector<int64_t> zigzags(kValues);
  for (uint32_t i = 0; i < kValues; ++i) {
    ranks[i] = i == 0 ? 0 : static_cast<uint8_t>((i * 17) % dictionary.size());
    zigzags[i] = dictionary[ranks[i]];
  }
  std::vector<uint32_t> exceptionPositions{100'003, 200'007, 700'011, 800'009};
  // Zigzag(+100000), zigzag(-100000), repeated to keep int32 output bounded.
  std::vector<int64_t> exceptionValues{200'000, 199'999, 200'000, 199'999};
  for (std::size_t i = 0; i < exceptionPositions.size(); ++i) {
    ranks[exceptionPositions[i]] = 0;
    zigzags[exceptionPositions[i]] = exceptionValues[i];
  }

  std::vector<int32_t> expected(kValues);
  int64_t current = kFirst;
  for (uint32_t i = 0; i < kValues; ++i) {
    const uint64_t zigzag = static_cast<uint64_t>(zigzags[i]);
    const int64_t delta =
        static_cast<int64_t>((zigzag >> 1) ^ (~(zigzag & 1) + 1));
    current += delta;
    expected[i] = static_cast<int32_t>(current);
  }

  rmm::cuda_stream stream;
  rmm::device_buffer rankDevice(ranks.data(), ranks.size(), stream.view());
  auto compressed =
      compressBlob(rankDevice.data(), ranks.size(), stream.view(), 0.0, 1);
  ASSERT_TRUE(compressed.used);
  ASSERT_EQ(compressed.segSizes.size(), 1);

  const std::size_t rankBytes = aligned(compressed.segSizes.front());
  const std::size_t dictionaryBytes = dictionary.size() * sizeof(uint16_t);
  const std::size_t positionBytes =
      exceptionPositions.size() * sizeof(uint32_t);
  const std::size_t valueBytes = exceptionValues.size() * sizeof(int64_t);
  const std::size_t totalBytes = rankBytes + aligned(dictionaryBytes) +
      aligned(positionBytes) + aligned(valueBytes);
  rmm::device_buffer wire(totalBytes, stream.view());
  std::size_t offset = 0;
  ASSERT_EQ(
      cudaMemcpyAsync(
          wire.data(),
          compressed.data.data(),
          rankBytes,
          cudaMemcpyDeviceToDevice,
          stream.value()),
      cudaSuccess);
  offset += rankBytes;
  ASSERT_EQ(
      cudaMemcpyAsync(
          static_cast<uint8_t*>(wire.data()) + offset,
          dictionary.data(),
          dictionaryBytes,
          cudaMemcpyHostToDevice,
          stream.value()),
      cudaSuccess);
  offset += aligned(dictionaryBytes);
  ASSERT_EQ(
      cudaMemcpyAsync(
          static_cast<uint8_t*>(wire.data()) + offset,
          exceptionPositions.data(),
          positionBytes,
          cudaMemcpyHostToDevice,
          stream.value()),
      cudaSuccess);
  offset += aligned(positionBytes);
  ASSERT_EQ(
      cudaMemcpyAsync(
          static_cast<uint8_t*>(wire.data()) + offset,
          exceptionValues.data(),
          valueBytes,
          cudaMemcpyHostToDevice,
          stream.value()),
      cudaSuccess);

  EncodedRegion region;
  region.rawBytes = expected.size() * sizeof(int32_t);
  region.codec = RegionCodec::kDeltaFreqPfor;
  region.elemWidth = sizeof(int32_t);
  region.base = 0;
  region.first = kFirst;
  region.segSizes.assign(
      compressed.segSizes.begin(), compressed.segSizes.end());
  region.dictionarySize = dictionary.size();
  region.exceptionCount = exceptionPositions.size();

  auto decoded =
      decompressPacked(wire.data(), {region}, region.rawBytes, stream.view());
  std::vector<int32_t> actual(kValues);
  ASSERT_EQ(
      cudaMemcpyAsync(
          actual.data(),
          decoded.data(),
          region.rawBytes,
          cudaMemcpyDeviceToHost,
          stream.value()),
      cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream.value()), cudaSuccess);
  EXPECT_EQ(actual, expected);
}

} // namespace
} // namespace facebook::velox::ucx_exchange
