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

#include "velox/dwio/nimble/encodings/benchmarks/NimbleEncodingRunner.h"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <folly/json/json.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/encodings/benchmarks/BlockBitPackingBenchmarkData.h"
#include "velox/dwio/nimble/encodings/benchmarks/FsstBenchmarkData.h"
#include "velox/dwio/nimble/encodings/benchmarks/PFOREncodingBenchmarkData.h"
#include "velox/dwio/nimble/encodings/benchmarks/PrefixBenchmarkData.h"
#include "velox/dwio/nimble/encodings/benchmarks/SimdForBitpackBenchmarkData.h"
#include "velox/dwio/nimble/encodings/benchmarks/VarintBenchmarkData.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "xplat/secure_lib/secure_string.h"

namespace facebook::nimble::benchmarks {
namespace {

struct PforArtifactOffsets {
  size_t baseBitWidth;
  size_t exceptionCount;
  uint32_t numExceptions;
  size_t positionsSize;
  size_t positions;
  uint32_t positionsBytes;
  size_t valuesSize;
  size_t values;
  uint32_t valuesBytes;
  size_t packed;
};

PforArtifactOffsets readPforArtifactOffsets(std::string_view artifact) {
  const char* cursor =
      artifact.data() + EncodingPrefix::kFixedPrefixSize + sizeof(uint32_t);
  const size_t baseBitWidth = cursor - artifact.data();
  ++cursor;
  const size_t exceptionCount = cursor - artifact.data();
  const uint32_t numExceptions = varint::readVarint32(&cursor);
  const size_t positionsSize = cursor - artifact.data();
  const uint32_t positionsBytes = varint::readVarint32(&cursor);
  const size_t positions = cursor - artifact.data();
  cursor += positionsBytes;
  const size_t valuesSize = cursor - artifact.data();
  const uint32_t valuesBytes = varint::readVarint32(&cursor);
  const size_t values = cursor - artifact.data();
  cursor += valuesBytes;
  return {
      .baseBitWidth = baseBitWidth,
      .exceptionCount = exceptionCount,
      .numExceptions = numExceptions,
      .positionsSize = positionsSize,
      .positions = positions,
      .positionsBytes = positionsBytes,
      .valuesSize = valuesSize,
      .values = values,
      .valuesBytes = valuesBytes,
      .packed = static_cast<size_t>(cursor - artifact.data()),
  };
}

struct BlockBitPackingArtifactOffsets {
  size_t compression;
  size_t blockSize;
  size_t numBlocks;
  uint32_t blockCount;
  size_t baselinesSize;
  size_t baselines;
  uint32_t baselinesBytes;
  size_t bitWidthsSize;
  size_t bitWidths;
  uint32_t bitWidthsBytes;
  size_t blockOffsetsSize;
  size_t blockOffsets;
  uint32_t blockOffsetsBytes;
  size_t firstBlockRows;
  size_t packed;
};

BlockBitPackingArtifactOffsets readBlockBitPackingArtifactOffsets(
    std::string_view artifact) {
  const char* cursor = artifact.data() + EncodingPrefix::kFixedPrefixSize;
  const size_t compression = cursor - artifact.data();
  ++cursor;
  const size_t blockSize = cursor - artifact.data();
  varint::readVarint32(&cursor);
  const size_t numBlocks = cursor - artifact.data();
  const uint32_t blockCount = varint::readVarint32(&cursor);
  const size_t baselinesSize = cursor - artifact.data();
  const uint32_t baselinesBytes = varint::readVarint32(&cursor);
  const size_t baselines = cursor - artifact.data();
  cursor += baselinesBytes;
  const size_t bitWidthsSize = cursor - artifact.data();
  const uint32_t bitWidthsBytes = varint::readVarint32(&cursor);
  const size_t bitWidths = cursor - artifact.data();
  cursor += bitWidthsBytes;
  const size_t blockOffsetsSize = cursor - artifact.data();
  const uint32_t blockOffsetsBytes = varint::readVarint32(&cursor);
  const size_t blockOffsets = cursor - artifact.data();
  cursor += blockOffsetsBytes;
  const size_t firstBlockRows = cursor - artifact.data();
  varint::readVarint32(&cursor);
  return {
      .compression = compression,
      .blockSize = blockSize,
      .numBlocks = numBlocks,
      .blockCount = blockCount,
      .baselinesSize = baselinesSize,
      .baselines = baselines,
      .baselinesBytes = baselinesBytes,
      .bitWidthsSize = bitWidthsSize,
      .bitWidths = bitWidths,
      .bitWidthsBytes = bitWidthsBytes,
      .blockOffsetsSize = blockOffsetsSize,
      .blockOffsets = blockOffsets,
      .blockOffsetsBytes = blockOffsetsBytes,
      .firstBlockRows = firstBlockRows,
      .packed = static_cast<size_t>(cursor - artifact.data()),
  };
}

uint32_t readUint32(std::string_view data, size_t offset) {
  uint32_t value;
  checked_memcpy_robust(
      &value,
      sizeof(value),
      data.data() + offset,
      data.size() - offset,
      sizeof(value));
  return value;
}

void writeUint32(std::string& data, size_t offset, uint32_t value) {
  checked_memcpy_offset(
      data.data(), data.size(), offset, &value, sizeof(value));
}

void writeVarint32WithWidth(
    std::string& data,
    size_t offset,
    size_t width,
    uint32_t value) {
  for (size_t byte = 0; byte < width; ++byte) {
    const bool hasNext = byte + 1 < width;
    data[offset + byte] = static_cast<char>(
        (value & 0x7F) | (hasNext ? uint32_t{0x80} : uint32_t{0}));
    value >>= 7;
  }
  EXPECT_EQ(0, value);
}

template <typename Operation>
void expectRuntimeErrorContaining(
    Operation&& operation,
    std::string_view expectedMessage) {
  try {
    operation();
    FAIL() << "expected std::runtime_error";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(
        std::string_view{error.what()}.find(expectedMessage),
        std::string_view::npos)
        << error.what();
  }
}

struct PrefixArtifactOffsets {
  uint32_t restartInterval;
  size_t restartOffsets;
  size_t entries;
  std::vector<size_t> entryOffsets;
};

PrefixArtifactOffsets readPrefixArtifactOffsets(
    std::string_view artifact,
    uint32_t rowCount) {
  const size_t intervalOffset = EncodingPrefix::kFixedPrefixSize;
  const uint32_t restartInterval = readUint32(artifact, intervalOffset);
  const uint32_t numRestarts = 1 + (rowCount - 1) / restartInterval;
  const size_t restartOffsets = intervalOffset + sizeof(uint32_t);
  const size_t entries = restartOffsets + numRestarts * sizeof(uint32_t);
  std::vector<size_t> entryOffsets;
  entryOffsets.reserve(rowCount);
  size_t offset = entries;
  for (uint32_t row = 0; row < rowCount; ++row) {
    entryOffsets.push_back(offset);
    offset += 2 * sizeof(uint32_t);
    offset += readUint32(artifact, offset - sizeof(uint32_t));
  }
  EXPECT_EQ(artifact.size(), offset);
  return {
      .restartInterval = restartInterval,
      .restartOffsets = restartOffsets,
      .entries = entries,
      .entryOffsets = std::move(entryOffsets),
  };
}

struct FsstArtifactOffsets {
  size_t symbolTableSize;
  size_t symbolTable;
  uint32_t symbolTableBytes;
  size_t lengthsSize;
  size_t lengths;
  uint32_t lengthsBytes;
  size_t blob;
};

FsstArtifactOffsets readFsstArtifactOffsets(std::string_view artifact) {
  const char* cursor = artifact.data() + EncodingPrefix::kFixedPrefixSize;
  const size_t symbolTableSize = cursor - artifact.data();
  const uint32_t symbolTableBytes = varint::readVarint32(&cursor);
  const size_t symbolTable = cursor - artifact.data();
  cursor += symbolTableBytes;
  const size_t lengthsSize = cursor - artifact.data();
  const uint32_t lengthsBytes = varint::readVarint32(&cursor);
  const size_t lengths = cursor - artifact.data();
  cursor += lengthsBytes;
  return {
      .symbolTableSize = symbolTableSize,
      .symbolTable = symbolTable,
      .symbolTableBytes = symbolTableBytes,
      .lengthsSize = lengthsSize,
      .lengths = lengths,
      .lengthsBytes = lengthsBytes,
      .blob = static_cast<size_t>(cursor - artifact.data()),
  };
}

uint32_t readFsstCompressedLength(
    std::string_view artifact,
    const FsstArtifactOffsets& offsets,
    uint32_t row) {
  constexpr size_t kBaselineOffset =
      EncodingPrefix::kFixedPrefixSize + sizeof(uint8_t);
  constexpr size_t kBitWidthOffset = kBaselineOffset + sizeof(uint32_t);
  constexpr size_t kPayloadOffset = kBitWidthOffset + sizeof(uint8_t);
  const auto child = artifact.substr(offsets.lengths, offsets.lengthsBytes);
  const uint32_t baseline = readUint32(child, kBaselineOffset);
  const auto bitWidth = static_cast<uint8_t>(child[kBitWidthOffset]);
  return baseline +
      FixedBitArray{child.data() + kPayloadOffset, bitWidth}.get(row);
}

class NimbleEncodingRunnerTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    rootPool_ =
        velox::memory::memoryManager()->addRootPool("NimbleEncodingRunnerTest");
    pool_ = rootPool_->addLeafChild("runner");
  }

  EncodingRunnerConfig config(std::string taskId) const {
    return EncodingRunnerConfig{
        .taskId = std::move(taskId),
        .rowCount = 257,
        .seed = 0xC0FFEE,
        .warmups = 0,
        .samples = 5,
        .minSampleTimeMicros = 0,
        .innerIterations = 1,
    };
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

TEST_F(NimbleEncodingRunnerTest, sameSeedProducesIdenticalArtifacts) {
  const auto runnerConfig = config("nimble.dictionary.encode.v1");

  const auto first = runEncodingBenchmark(runnerConfig, *pool_);
  const auto second = runEncodingBenchmark(runnerConfig, *pool_);

  EXPECT_EQ(first.encodedArtifact, second.encodedArtifact);
  EXPECT_EQ(first.inputDigest, second.inputDigest);
  EXPECT_EQ(first.outputDigest, second.outputDigest);
  EXPECT_EQ(first.artifactDigest, second.artifactDigest);
  EXPECT_EQ(first.inputDigest, first.outputDigest);
  EXPECT_EQ(64, first.inputDigest.size());
  EXPECT_EQ(64, first.artifactDigest.size());
  ASSERT_EQ(first.samplesSeconds.size(), runnerConfig.samples);
  for (const auto sample : first.samplesSeconds) {
    EXPECT_GT(sample, 0.0);
  }
}

TEST_F(
    NimbleEncodingRunnerTest,
    executableArtifactsRoundTripAcrossRunnerBoundary) {
  const std::array<std::string, 31> taskIds{
      "nimble.rle.encode.v1",
      "nimble.rle.selection_e2e.v1",
      "nimble.dictionary.decode_dense.v1",
      "nimble.fixed_bit_width.encode.v1",
      "nimble.fixed_bit_width.selection_e2e.v1",
      "nimble.delta.encode.v1",
      "nimble.delta.decode_dense.v1",
      "nimble.sparse_bool.encode.v1",
      "nimble.sparse_bool.decode_dense.v1",
      "nimble.sparse_bool.skip_seek.v1",
      "nimble.pfor.encode.v1",
      "nimble.pfor.decode_dense.v1",
      "nimble.pfor.skip_seek.v1",
      "nimble.simd_for_bitpack.encode.v1",
      "nimble.simd_for_bitpack.decode_dense.v1",
      "nimble.simd_for_bitpack.skip_seek.v1",
      "nimble.block_bit_packing.encode.v1",
      "nimble.block_bit_packing.decode_dense.v1",
      "nimble.block_bit_packing.skip_seek.v1",
      "nimble.varint.encode.v1",
      "nimble.varint.decode_dense.v1",
      "nimble.varint.skip_seek.v1",
      "nimble.prefix.encode.v1",
      "nimble.prefix.decode_dense.v1",
      "nimble.prefix.skip_seek.v1",
      "nimble.fsst.encode.v1",
      "nimble.fsst.decode_dense.v1",
      "nimble.fsst.skip_seek.v1",
      "nimble.nullable.decode_dense.v1",
      "nimble.alp.decode_dense.v1",
      "nimble.delta_block.decode_dense.v1",
  };

  for (const auto& taskId : taskIds) {
    SCOPED_TRACE(taskId);
    const auto runnerConfig = config(taskId);
    const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
    const auto verification = verifyEncodingArtifact(
        runnerConfig, measurement.encodedArtifact, *pool_);

    EXPECT_EQ(measurement.taskId, verification.taskId);
    EXPECT_EQ(measurement.encoding, verification.encoding);
    EXPECT_EQ(measurement.dataType, verification.dataType);
    EXPECT_EQ(measurement.inputDigest, verification.inputDigest);
    EXPECT_EQ(measurement.outputDigest, verification.outputDigest);
    EXPECT_EQ(measurement.artifactDigest, verification.artifactDigest);
    EXPECT_EQ(measurement.encodedBytes, verification.encodedBytes);
  }
}

TEST_F(NimbleEncodingRunnerTest, taskMatrixProducesRawTimingSamples) {
  const std::array<std::string, 10> lanes{
      "encode",
      "decode_construct",
      "decode_dense",
      "decode_range50",
      "decode_scatter10",
      "decode_scatter1",
      "skip_seek",
      "view_random",
      "slice",
      "selection_e2e",
  };
  struct EncodingCase {
    std::string_view slug;
    std::string_view dataType;
    bool supportsView;
  };
  const std::array<EncodingCase, 14> encodings{{
      {"rle", "int64", true},
      {"dictionary", "int64", true},
      {"fixed_bit_width", "uint64", true},
      {"delta", "uint32", false},
      {"sparse_bool", "bool", false},
      {"pfor", "uint32", false},
      {"simd_for_bitpack", "uint32", false},
      {"block_bit_packing", "uint32", false},
      {"varint", "uint32", false},
      {"prefix", "string", false},
      {"fsst", "string", false},
      {"nullable", "int64", false},
      {"alp", "double", true},
      {"delta_block", "int64", true},
  }};

  for (const auto& encoding : encodings) {
    for (const auto& lane : lanes) {
      if (lane == "view_random" && !encoding.supportsView) {
        continue;
      }
      if (encoding.slug == "fixed_bit_width" && lane != "encode" &&
          lane != "decode_dense" && lane != "decode_scatter10" &&
          lane != "decode_scatter1" && lane != "skip_seek" &&
          lane != "selection_e2e") {
        continue;
      }
      if (encoding.slug == "delta" && lane != "encode" &&
          lane != "decode_dense" && lane != "skip_seek") {
        continue;
      }
      if (encoding.slug == "sparse_bool" && lane != "encode" &&
          lane != "decode_dense" && lane != "skip_seek") {
        continue;
      }
      if (encoding.slug == "pfor" && lane != "encode" &&
          lane != "decode_dense" && lane != "skip_seek") {
        continue;
      }
      if (encoding.slug == "simd_for_bitpack" && lane != "encode" &&
          lane != "decode_dense" && lane != "skip_seek") {
        continue;
      }
      if (encoding.slug == "block_bit_packing" && lane != "encode" &&
          lane != "decode_dense" && lane != "skip_seek") {
        continue;
      }
      if (encoding.slug == "varint" && lane != "encode" &&
          lane != "decode_dense" && lane != "skip_seek") {
        continue;
      }
      if (encoding.slug == "prefix" && lane != "encode" &&
          lane != "decode_dense" && lane != "skip_seek") {
        continue;
      }
      if (encoding.slug == "fsst" && lane != "encode" &&
          lane != "decode_dense" && lane != "skip_seek") {
        continue;
      }
      SCOPED_TRACE(std::string{encoding.slug} + "." + lane);
      auto runnerConfig =
          config("nimble." + std::string{encoding.slug} + "." + lane + ".v1");
      runnerConfig.rowCount = 100;
      const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);

      EXPECT_EQ(lane, measurement.lane);
      EXPECT_EQ(encoding.slug, measurement.encoding);
      EXPECT_EQ(encoding.dataType, measurement.dataType);
      EXPECT_EQ(measurement.inputDigest, measurement.outputDigest);
      EXPECT_EQ(runnerConfig.samples, measurement.samplesSeconds.size());
    }
  }
}

TEST_F(NimbleEncodingRunnerTest, fixedBitWidthUsesNativeBenchmarkProfile) {
  auto runnerConfig = config("nimble.fixed_bit_width.encode.v1");
  runnerConfig.rowCount = 4096;

  const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
  const auto verification =
      verifyEncodingArtifact(runnerConfig, measurement.encodedArtifact, *pool_);

  EXPECT_EQ("fixed_bit_width", measurement.encoding);
  EXPECT_EQ("uint64", measurement.dataType);
  EXPECT_EQ(
      "ccd36b95b0e2673446ac128d52ba998a5c686df3d4a9dee2c7cffabf4e48d5bf",
      measurement.inputDigest);
  EXPECT_EQ(measurement.inputDigest, measurement.outputDigest);
  EXPECT_EQ(measurement.artifactDigest, verification.artifactDigest);
  EXPECT_LT(measurement.encodedBytes, runnerConfig.rowCount * sizeof(uint64_t));
}

TEST_F(
    NimbleEncodingRunnerTest,
    fixedBitWidthProfileCrossVerifiesBoundarySeeds) {
  for (const uint64_t seed :
       {uint64_t{0},
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())}) {
    SCOPED_TRACE(seed);
    auto runnerConfig = config("nimble.fixed_bit_width.encode.v1");
    runnerConfig.seed = seed;
    runnerConfig.rowCount = 100;

    const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
    const auto verification = verifyEncodingArtifact(
        runnerConfig, measurement.encodedArtifact, *pool_);

    EXPECT_EQ(measurement.inputDigest, verification.outputDigest);
    EXPECT_EQ(measurement.artifactDigest, verification.artifactDigest);
  }
}

TEST_F(NimbleEncodingRunnerTest, deltaUsesNativeIncreasingProfile) {
  auto runnerConfig = config("nimble.delta.encode.v1");
  runnerConfig.rowCount = 4096;

  const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
  const auto verification =
      verifyEncodingArtifact(runnerConfig, measurement.encodedArtifact, *pool_);

  EXPECT_EQ("delta", measurement.encoding);
  EXPECT_EQ("uint32", measurement.dataType);
  EXPECT_EQ(
      "a199f8087c72ce90daf78cf570a3de46f7e4bbf3afce25713aba583bed6d80f6",
      measurement.inputDigest);
  EXPECT_EQ(measurement.inputDigest, measurement.outputDigest);
  EXPECT_EQ(measurement.artifactDigest, verification.artifactDigest);
  EXPECT_LT(measurement.encodedBytes, runnerConfig.rowCount * sizeof(uint32_t));
}

TEST_F(
    NimbleEncodingRunnerTest,
    deltaProfileCrossVerifiesChunkRemaindersAndBoundarySeeds) {
  for (const uint64_t seed :
       {uint64_t{0},
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())}) {
    for (const uint32_t rowCount : {112, 113, 127, 128, 129}) {
      SCOPED_TRACE(
          "seed=" + std::to_string(seed) +
          ", rows=" + std::to_string(rowCount));
      auto runnerConfig = config("nimble.delta.decode_dense.v1");
      runnerConfig.seed = seed;
      runnerConfig.rowCount = rowCount;

      const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
      const auto verification = verifyEncodingArtifact(
          runnerConfig, measurement.encodedArtifact, *pool_);

      EXPECT_EQ(measurement.inputDigest, verification.outputDigest);
      EXPECT_EQ(measurement.artifactDigest, verification.artifactDigest);
    }
  }
}

TEST_F(NimbleEncodingRunnerTest, sparseBoolUsesNativeSparseProfile) {
  auto runnerConfig = config("nimble.sparse_bool.encode.v1");
  runnerConfig.rowCount = 4096;

  const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
  const auto verification =
      verifyEncodingArtifact(runnerConfig, measurement.encodedArtifact, *pool_);

  EXPECT_EQ("sparse_bool", measurement.encoding);
  EXPECT_EQ("bool", measurement.dataType);
  EXPECT_EQ(
      "e70d715fd4ee19fe52e5806ea68f3c656c1ee227ceadd64731537d3feaae0dce",
      measurement.inputDigest);
  EXPECT_EQ(measurement.inputDigest, measurement.outputDigest);
  EXPECT_EQ(measurement.artifactDigest, verification.artifactDigest);
  EXPECT_LT(measurement.encodedBytes, runnerConfig.rowCount * sizeof(bool));
}

TEST_F(
    NimbleEncodingRunnerTest,
    sparseBoolProfileCrossVerifiesBoundarySeedsAndRows) {
  for (const uint64_t seed :
       {uint64_t{0},
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())}) {
    for (const uint32_t rowCount : {100, 127, 128, 129}) {
      SCOPED_TRACE(
          "seed=" + std::to_string(seed) +
          ", rows=" + std::to_string(rowCount));
      auto runnerConfig = config("nimble.sparse_bool.decode_dense.v1");
      runnerConfig.seed = seed;
      runnerConfig.rowCount = rowCount;

      const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
      const auto verification = verifyEncodingArtifact(
          runnerConfig, measurement.encodedArtifact, *pool_);

      EXPECT_EQ(measurement.inputDigest, verification.outputDigest);
      EXPECT_EQ(measurement.artifactDigest, verification.artifactDigest);
    }
  }
}

TEST_F(NimbleEncodingRunnerTest, pforUsesNativeOutlierProfile) {
  auto runnerConfig = config("nimble.pfor.encode.v1");
  runnerConfig.rowCount = 4096;

  const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
  const auto verification =
      verifyEncodingArtifact(runnerConfig, measurement.encodedArtifact, *pool_);

  EXPECT_EQ("pfor", measurement.encoding);
  EXPECT_EQ("uint32", measurement.dataType);
  EXPECT_EQ(
      "bedb99cdfe074a4761ede05d305fea3d6e2156a90a34723976f7052c61ef588b",
      measurement.inputDigest);
  EXPECT_EQ(measurement.inputDigest, measurement.outputDigest);
  EXPECT_EQ(measurement.artifactDigest, verification.artifactDigest);
  EXPECT_LT(measurement.encodedBytes, runnerConfig.rowCount * sizeof(uint32_t));

  ++runnerConfig.seed;
  const auto differentSeed = runEncodingBenchmark(runnerConfig, *pool_);
  EXPECT_NE(measurement.inputDigest, differentSeed.inputDigest);
}

TEST_F(NimbleEncodingRunnerTest, pforProfileCrossVerifiesBoundarySeedsAndRows) {
  for (const uint64_t seed :
       {uint64_t{0},
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())}) {
    for (const uint32_t rowCount : {100, 127, 128, 129, 255, 256, 257}) {
      SCOPED_TRACE(
          "seed=" + std::to_string(seed) +
          ", rows=" + std::to_string(rowCount));
      auto runnerConfig = config("nimble.pfor.decode_dense.v1");
      runnerConfig.seed = seed;
      runnerConfig.rowCount = rowCount;

      const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
      const auto verification = verifyEncodingArtifact(
          runnerConfig, measurement.encodedArtifact, *pool_);

      EXPECT_EQ(measurement.inputDigest, verification.outputDigest);
      EXPECT_EQ(measurement.artifactDigest, verification.artifactDigest);
    }
  }
}

TEST_F(NimbleEncodingRunnerTest, simdForBitpackUsesNative16BitProfile) {
  auto runnerConfig = config("nimble.simd_for_bitpack.encode.v1");
  runnerConfig.rowCount = 4096;

  const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
  const auto verification =
      verifyEncodingArtifact(runnerConfig, measurement.encodedArtifact, *pool_);

  EXPECT_EQ("simd_for_bitpack", measurement.encoding);
  EXPECT_EQ("uint32", measurement.dataType);
  EXPECT_EQ(
      "8a5c6bee57b4678306c18c74a076c9a7ac2d98f8c1db75b07b25652cb275d463",
      measurement.inputDigest);
  EXPECT_EQ(measurement.inputDigest, measurement.outputDigest);
  EXPECT_EQ(measurement.artifactDigest, verification.artifactDigest);
  EXPECT_LT(measurement.encodedBytes, runnerConfig.rowCount * sizeof(uint32_t));

  ++runnerConfig.seed;
  const auto differentSeed = runEncodingBenchmark(runnerConfig, *pool_);
  EXPECT_NE(measurement.inputDigest, differentSeed.inputDigest);
}

TEST_F(
    NimbleEncodingRunnerTest,
    simdForBitpackProfileCrossVerifiesBoundarySeedsAndRows) {
  for (const uint64_t seed :
       {uint64_t{0},
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())}) {
    for (const uint32_t rowCount : {100, 127, 128, 129, 255, 256, 257}) {
      SCOPED_TRACE(
          "seed=" + std::to_string(seed) +
          ", rows=" + std::to_string(rowCount));
      auto runnerConfig = config("nimble.simd_for_bitpack.decode_dense.v1");
      runnerConfig.seed = seed;
      runnerConfig.rowCount = rowCount;

      const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
      const auto verification = verifyEncodingArtifact(
          runnerConfig, measurement.encodedArtifact, *pool_);

      EXPECT_EQ(measurement.inputDigest, verification.outputDigest);
      EXPECT_EQ(measurement.artifactDigest, verification.artifactDigest);
    }
  }
}

TEST_F(NimbleEncodingRunnerTest, blockBitPackingUsesNativeBlockLocalProfile) {
  auto runnerConfig = config("nimble.block_bit_packing.encode.v1");
  runnerConfig.rowCount = 4096;

  const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
  const auto verification =
      verifyEncodingArtifact(runnerConfig, measurement.encodedArtifact, *pool_);

  EXPECT_EQ("block_bit_packing", measurement.encoding);
  EXPECT_EQ("uint32", measurement.dataType);
  EXPECT_EQ(
      "29b034edff59da27be03cb8bd347e2528eb71bb49ca9817ed20bd7a2e01cab2d",
      measurement.inputDigest);
  EXPECT_EQ(measurement.inputDigest, measurement.outputDigest);
  EXPECT_EQ(measurement.artifactDigest, verification.artifactDigest);
  EXPECT_LT(measurement.encodedBytes, runnerConfig.rowCount * sizeof(uint32_t));

  ++runnerConfig.seed;
  const auto differentSeed = runEncodingBenchmark(runnerConfig, *pool_);
  EXPECT_NE(measurement.inputDigest, differentSeed.inputDigest);
}

TEST_F(
    NimbleEncodingRunnerTest,
    blockBitPackingProfileCrossVerifiesBlockBoundaries) {
  for (const uint64_t seed :
       {uint64_t{0},
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())}) {
    for (const uint32_t rowCount : {100, 1023, 1024, 1025, 2047, 2048, 2049}) {
      SCOPED_TRACE(
          "seed=" + std::to_string(seed) +
          ", rows=" + std::to_string(rowCount));
      auto runnerConfig = config("nimble.block_bit_packing.decode_dense.v1");
      runnerConfig.seed = seed;
      runnerConfig.rowCount = rowCount;

      const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
      const auto verification = verifyEncodingArtifact(
          runnerConfig, measurement.encodedArtifact, *pool_);

      EXPECT_EQ(measurement.inputDigest, verification.outputDigest);
      EXPECT_EQ(measurement.artifactDigest, verification.artifactDigest);
    }
  }
}

TEST_F(NimbleEncodingRunnerTest, blockBitPackingConsumerUsesPristineArtifact) {
  const auto runnerConfig = config("nimble.block_bit_packing.skip_seek.v1");
  const auto reference = runEncodingBenchmark(runnerConfig, *pool_);

  const auto candidate = runEncodingBenchmark(
      runnerConfig,
      *pool_,
      std::optional<std::string_view>{reference.encodedArtifact});

  EXPECT_EQ(reference.encodedArtifact, candidate.encodedArtifact);
  EXPECT_EQ(reference.artifactDigest, candidate.artifactDigest);
  EXPECT_EQ(reference.encodedBytes, candidate.encodedBytes);
}

TEST_F(NimbleEncodingRunnerTest, varintUsesNativeMixedWidthProfile) {
  auto runnerConfig = config("nimble.varint.encode.v1");
  runnerConfig.rowCount = 4096;

  const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
  const auto verification =
      verifyEncodingArtifact(runnerConfig, measurement.encodedArtifact, *pool_);

  EXPECT_EQ("varint", measurement.encoding);
  EXPECT_EQ("uint32", measurement.dataType);
  EXPECT_EQ(
      "c5177276571a8ae00666b9760d3ca4cc6fc31096753a4ac738954c47bac8bc5d",
      measurement.inputDigest);
  EXPECT_EQ(
      varintBenchmarkBaseline(runnerConfig.seed),
      readUint32(
          measurement.encodedArtifact, EncodingPrefix::kFixedPrefixSize));
  EXPECT_EQ(measurement.inputDigest, measurement.outputDigest);
  EXPECT_EQ(measurement.artifactDigest, verification.artifactDigest);
  EXPECT_LT(measurement.encodedBytes, runnerConfig.rowCount * sizeof(uint32_t));

  const char* cursor = measurement.encodedArtifact.data() +
      EncodingPrefix::kFixedPrefixSize + sizeof(uint32_t);
  size_t expectedEncodedBytes =
      EncodingPrefix::kFixedPrefixSize + sizeof(uint32_t);
  constexpr std::array<uint32_t, 10> kBoundaryResiduals{
      0,
      0x7F,
      0x80,
      0x3FFF,
      0x4000,
      0x1FFFFF,
      0x200000,
      0x0FFFFFFF,
      0x10000000,
      std::numeric_limits<uint32_t>::max() -
          varintBenchmarkBaseline(kVarintBenchmarkDefaultSeed),
  };
  constexpr std::array<size_t, 10> kBoundaryWidths{
      1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  for (uint32_t row = 0; row < runnerConfig.rowCount; ++row) {
    const char* before = cursor;
    const uint32_t residual = varint::readVarint32(&cursor);
    EXPECT_EQ(varintBenchmarkResidual(row, runnerConfig.seed), residual);
    expectedEncodedBytes += cursor - before;
    if (row < kBoundaryResiduals.size()) {
      EXPECT_EQ(kBoundaryResiduals[row], residual);
      EXPECT_EQ(kBoundaryWidths[row], cursor - before);
    }
  }
  EXPECT_EQ(
      measurement.encodedArtifact.data() + measurement.encodedArtifact.size(),
      cursor);
  EXPECT_EQ(expectedEncodedBytes, measurement.encodedBytes);

  ++runnerConfig.seed;
  const auto differentSeed = runEncodingBenchmark(runnerConfig, *pool_);
  EXPECT_NE(measurement.inputDigest, differentSeed.inputDigest);
}

TEST_F(
    NimbleEncodingRunnerTest,
    varintProfileCrossVerifiesBoundarySeedsAndRows) {
  for (const uint64_t seed :
       {uint64_t{0},
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())}) {
    for (const uint32_t rowCount : {100, 127, 128, 129, 255, 256, 257}) {
      SCOPED_TRACE(
          "seed=" + std::to_string(seed) +
          ", rows=" + std::to_string(rowCount));
      auto runnerConfig = config("nimble.varint.decode_dense.v1");
      runnerConfig.seed = seed;
      runnerConfig.rowCount = rowCount;

      const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
      const auto verification = verifyEncodingArtifact(
          runnerConfig, measurement.encodedArtifact, *pool_);

      EXPECT_EQ(measurement.inputDigest, verification.outputDigest);
      EXPECT_EQ(measurement.artifactDigest, verification.artifactDigest);
    }
  }
}

TEST_F(NimbleEncodingRunnerTest, varintConsumerUsesPristineArtifact) {
  const auto runnerConfig = config("nimble.varint.skip_seek.v1");
  const auto reference = runEncodingBenchmark(runnerConfig, *pool_);

  const auto candidate = runEncodingBenchmark(
      runnerConfig,
      *pool_,
      std::optional<std::string_view>{reference.encodedArtifact});

  EXPECT_EQ(reference.encodedArtifact, candidate.encodedArtifact);
  EXPECT_EQ(reference.artifactDigest, candidate.artifactDigest);
  EXPECT_EQ(reference.encodedBytes, candidate.encodedBytes);
}

TEST_F(NimbleEncodingRunnerTest, rleCorpusExercisesRunCompression) {
  auto runnerConfig = config("nimble.rle.encode.v1");
  runnerConfig.rowCount = 4096;

  const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);

  EXPECT_EQ("rle", measurement.encoding);
  EXPECT_EQ("int64", measurement.dataType);
  EXPECT_EQ(measurement.inputDigest, measurement.outputDigest);
  EXPECT_LT(measurement.encodedBytes, runnerConfig.rowCount * sizeof(int64_t));
}

TEST_F(NimbleEncodingRunnerTest, rleEncodeProfileCrossVerifiesBoundarySeeds) {
  for (const uint64_t seed :
       {uint64_t{0},
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())}) {
    SCOPED_TRACE(seed);
    auto runnerConfig = config("nimble.rle.encode.v1");
    runnerConfig.seed = seed;
    runnerConfig.rowCount = 100;

    const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
    const auto verification = verifyEncodingArtifact(
        runnerConfig, measurement.encodedArtifact, *pool_);

    EXPECT_EQ(measurement.inputDigest, verification.outputDigest);
    EXPECT_EQ(measurement.artifactDigest, verification.artifactDigest);
  }
}

TEST_F(NimbleEncodingRunnerTest, invalidContractRejectsConfiguration) {
  auto zeroSamples = config("nimble.dictionary.decode_dense.v1");
  zeroSamples.samples = 0;
  EXPECT_THROW(
      runEncodingBenchmark(zeroSamples, *pool_), std::invalid_argument);

  auto tooFewRows = config("nimble.dictionary.decode_dense.v1");
  tooFewRows.rowCount = 99;
  EXPECT_THROW(runEncodingBenchmark(tooFewRows, *pool_), std::invalid_argument);

  auto tooManyRows = config("nimble.dictionary.decode_dense.v1");
  tooManyRows.rowCount = 1'048'577;
  EXPECT_THROW(
      runEncodingBenchmark(tooManyRows, *pool_), std::invalid_argument);

  auto tooManySamples = config("nimble.dictionary.decode_dense.v1");
  tooManySamples.samples = 101;
  EXPECT_THROW(
      runEncodingBenchmark(tooManySamples, *pool_), std::invalid_argument);

  EXPECT_THROW(
      runEncodingBenchmark(config("nimble.unknown.decode_dense.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(config("nimble.nullable.view_random.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(
          config("nimble.fixed_bit_width.decode_construct.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(config("nimble.delta.decode_construct.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(
          config("nimble.sparse_bool.decode_construct.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(config("nimble.pfor.decode_construct.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(config("nimble.pfor.view_random.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(config("nimble.pfor.slice.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(config("nimble.pfor.selection_e2e.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(
          config("nimble.simd_for_bitpack.decode_construct.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(
          config("nimble.simd_for_bitpack.view_random.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(config("nimble.simd_for_bitpack.slice.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(
          config("nimble.simd_for_bitpack.selection_e2e.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(
          config("nimble.block_bit_packing.decode_construct.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(
          config("nimble.block_bit_packing.view_random.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(config("nimble.block_bit_packing.slice.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(
          config("nimble.block_bit_packing.selection_e2e.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(config("nimble.varint.decode_construct.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(config("nimble.varint.view_random.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(config("nimble.varint.slice.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(config("nimble.varint.selection_e2e.v1"), *pool_),
      std::invalid_argument);
}

TEST_F(NimbleEncodingRunnerTest, singleSampleSupportsHostSidePairScheduling) {
  auto runnerConfig = config("nimble.dictionary.decode_dense.v1");
  runnerConfig.samples = 1;

  const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);

  ASSERT_EQ(1, measurement.samplesSeconds.size());
  EXPECT_GT(measurement.samplesSeconds.front(), 0.0);
}

TEST_F(NimbleEncodingRunnerTest, mismatchedTaskRejectsArtifact) {
  const auto dictionary =
      runEncodingBenchmark(config("nimble.dictionary.decode_dense.v1"), *pool_);

  EXPECT_THROW(
      verifyEncodingArtifact(
          config("nimble.delta_block.decode_dense.v1"),
          dictionary.encodedArtifact,
          *pool_),
      std::runtime_error);
}

TEST_F(
    NimbleEncodingRunnerTest,
    consumerLaneAcceptsCanonicalArtifactAndProducerLaneRejectsIt) {
  const auto consumerConfig = config("nimble.dictionary.decode_dense.v1");
  const auto reference = runEncodingBenchmark(consumerConfig, *pool_);

  const auto candidate = runEncodingBenchmark(
      consumerConfig,
      *pool_,
      std::optional<std::string_view>{reference.encodedArtifact});

  EXPECT_EQ(reference.encodedArtifact, candidate.encodedArtifact);
  EXPECT_EQ(reference.artifactDigest, candidate.artifactDigest);
  EXPECT_EQ(reference.encodedBytes, candidate.encodedBytes);
  EXPECT_THROW(
      runEncodingBenchmark(
          config("nimble.dictionary.encode.v1"),
          *pool_,
          std::optional<std::string_view>{reference.encodedArtifact}),
      std::invalid_argument);
}

TEST_F(NimbleEncodingRunnerTest, emptyArtifactIsRejected) {
  EXPECT_THROW(
      verifyEncodingArtifact(
          config("nimble.dictionary.decode_dense.v1"), "", *pool_),
      std::runtime_error);
}

TEST_F(NimbleEncodingRunnerTest, truncatedRleArtifactsAreRejected) {
  const auto runnerConfig = config("nimble.rle.decode_dense.v1");
  const auto reference = runEncodingBenchmark(runnerConfig, *pool_);

  for (size_t size = 0; size < reference.encodedArtifact.size(); ++size) {
    const std::string truncated = reference.encodedArtifact.substr(0, size);
    EXPECT_THROW(
        verifyEncodingArtifact(runnerConfig, truncated, *pool_),
        std::runtime_error)
        << "accepted RLE artifact truncated to " << size << " bytes";
  }
}

TEST_F(NimbleEncodingRunnerTest, malformedRleMetadataIsRejected) {
  const auto runnerConfig = config("nimble.rle.decode_dense.v1");
  const auto reference = runEncodingBenchmark(runnerConfig, *pool_);
  constexpr size_t kRunLengthsSizeOffset = EncodingPrefix::kFixedPrefixSize;
  constexpr size_t kRunLengthsOffset = kRunLengthsSizeOffset + sizeof(uint32_t);
  constexpr size_t kNestedRowCountOffset =
      kRunLengthsOffset + EncodingPrefix::kRowCountOffset;
  constexpr size_t kRunLengthsCompressionOffset =
      kRunLengthsOffset + EncodingPrefix::kFixedPrefixSize;
  constexpr size_t kFirstRunLengthOffset =
      kRunLengthsCompressionOffset + sizeof(uint8_t);
  uint32_t runLengthsSize;
  checked_memcpy_robust(
      &runLengthsSize,
      sizeof(runLengthsSize),
      reference.encodedArtifact.data() + kRunLengthsSizeOffset,
      reference.encodedArtifact.size() - kRunLengthsSizeOffset,
      sizeof(runLengthsSize));
  uint32_t runCount;
  checked_memcpy_robust(
      &runCount,
      sizeof(runCount),
      reference.encodedArtifact.data() + kNestedRowCountOffset,
      reference.encodedArtifact.size() - kNestedRowCountOffset,
      sizeof(runCount));
  const size_t runValuesOffset = kRunLengthsOffset + runLengthsSize;
  const size_t runValuesRowCountOffset =
      runValuesOffset + EncodingPrefix::kRowCountOffset;
  const size_t runValuesBitWidthOffset = runValuesOffset +
      EncodingPrefix::kFixedPrefixSize + sizeof(uint8_t) + sizeof(uint64_t);

  auto oversizedChild = reference.encodedArtifact;
  const uint32_t oversizedChildSize = std::numeric_limits<uint32_t>::max();
  checked_memcpy_offset(
      oversizedChild.data(),
      oversizedChild.size(),
      kRunLengthsSizeOffset,
      &oversizedChildSize,
      sizeof(oversizedChildSize));

  auto oversizedRunCount = reference.encodedArtifact;
  const uint32_t invalidRunCount = runnerConfig.rowCount + 1;
  checked_memcpy_offset(
      oversizedRunCount.data(),
      oversizedRunCount.size(),
      kNestedRowCountOffset,
      &invalidRunCount,
      sizeof(invalidRunCount));

  auto zeroRunLength = reference.encodedArtifact;
  const uint32_t zero = 0;
  checked_memcpy_offset(
      zeroRunLength.data(),
      zeroRunLength.size(),
      kFirstRunLengthOffset,
      &zero,
      sizeof(zero));

  auto overflowingRunLength = reference.encodedArtifact;
  const uint32_t overflowing = std::numeric_limits<uint32_t>::max();
  checked_memcpy_offset(
      overflowingRunLength.data(),
      overflowingRunLength.size(),
      kFirstRunLengthOffset,
      &overflowing,
      sizeof(overflowing));

  auto compressedRunLengths = reference.encodedArtifact;
  compressedRunLengths[kRunLengthsCompressionOffset] =
      static_cast<char>(CompressionType::Zstd);

  auto mismatchedValueCount = reference.encodedArtifact;
  checked_memcpy_offset(
      mismatchedValueCount.data(),
      mismatchedValueCount.size(),
      runValuesRowCountOffset,
      &invalidRunCount,
      sizeof(invalidRunCount));

  auto invalidBitWidth = reference.encodedArtifact;
  invalidBitWidth[runValuesBitWidthOffset] = 65;

  auto zeroBitWidth = reference.encodedArtifact;
  zeroBitWidth.resize(
      runValuesBitWidthOffset + sizeof(uint8_t) +
      FixedBitArray::bufferSize(runCount, 0));
  zeroBitWidth[runValuesBitWidthOffset] = 0;

  auto trailingPayload = reference.encodedArtifact;
  trailingPayload.push_back('\0');

  for (const auto& malformed :
       {oversizedChild,
        oversizedRunCount,
        zeroRunLength,
        overflowingRunLength,
        compressedRunLengths,
        mismatchedValueCount,
        invalidBitWidth,
        zeroBitWidth,
        trailingPayload}) {
    EXPECT_THROW(
        verifyEncodingArtifact(runnerConfig, malformed, *pool_),
        std::runtime_error);
  }
}

TEST_F(NimbleEncodingRunnerTest, truncatedDeltaArtifactsAreRejected) {
  const auto runnerConfig = config("nimble.delta.decode_dense.v1");
  const auto reference = runEncodingBenchmark(runnerConfig, *pool_);

  for (size_t size = 0; size < reference.encodedArtifact.size(); ++size) {
    const std::string truncated = reference.encodedArtifact.substr(0, size);
    EXPECT_THROW(
        verifyEncodingArtifact(runnerConfig, truncated, *pool_),
        std::runtime_error)
        << "accepted Delta artifact truncated to " << size << " bytes";
  }
}

TEST_F(NimbleEncodingRunnerTest, malformedDeltaMetadataIsRejected) {
  auto runnerConfig = config("nimble.delta.decode_dense.v1");
  runnerConfig.rowCount = 4095;
  const auto reference = runEncodingBenchmark(runnerConfig, *pool_);
  constexpr size_t kDeltasSizeOffset = EncodingPrefix::kFixedPrefixSize;
  constexpr size_t kRestatementsSizeOffset =
      kDeltasSizeOffset + sizeof(uint32_t);
  constexpr size_t kDeltasOffset = kRestatementsSizeOffset + sizeof(uint32_t);
  uint32_t deltasSize;
  checked_memcpy_robust(
      &deltasSize,
      sizeof(deltasSize),
      reference.encodedArtifact.data() + kDeltasSizeOffset,
      reference.encodedArtifact.size() - kDeltasSizeOffset,
      sizeof(deltasSize));
  uint32_t restatementsSize;
  checked_memcpy_robust(
      &restatementsSize,
      sizeof(restatementsSize),
      reference.encodedArtifact.data() + kRestatementsSizeOffset,
      reference.encodedArtifact.size() - kRestatementsSizeOffset,
      sizeof(restatementsSize));
  const size_t restatementsOffset = kDeltasOffset + deltasSize;
  const size_t flagsOffset = restatementsOffset + restatementsSize;
  const size_t deltasRowCountOffset =
      kDeltasOffset + EncodingPrefix::kRowCountOffset;
  const size_t deltasCompressionOffset =
      kDeltasOffset + EncodingPrefix::kFixedPrefixSize;
  const size_t deltasBitWidthOffset =
      deltasCompressionOffset + sizeof(uint8_t) + sizeof(uint32_t);
  uint32_t deltasRowCount;
  checked_memcpy_robust(
      &deltasRowCount,
      sizeof(deltasRowCount),
      reference.encodedArtifact.data() + deltasRowCountOffset,
      reference.encodedArtifact.size() - deltasRowCountOffset,
      sizeof(deltasRowCount));
  const size_t restatementsCompressionOffset =
      restatementsOffset + EncodingPrefix::kFixedPrefixSize;
  const size_t restatementsRowCountOffset =
      restatementsOffset + EncodingPrefix::kRowCountOffset;
  const size_t flagsRowCountOffset =
      flagsOffset + EncodingPrefix::kRowCountOffset;
  const size_t flagsCompressionOffset =
      flagsOffset + EncodingPrefix::kFixedPrefixSize;
  const size_t flagsPayloadOffset = flagsCompressionOffset + sizeof(uint8_t);

  auto oversizedDeltas = reference.encodedArtifact;
  const uint32_t oversizedChildSize = std::numeric_limits<uint32_t>::max();
  checked_memcpy_offset(
      oversizedDeltas.data(),
      oversizedDeltas.size(),
      kDeltasSizeOffset,
      &oversizedChildSize,
      sizeof(oversizedChildSize));

  auto oversizedRestatements = reference.encodedArtifact;
  checked_memcpy_offset(
      oversizedRestatements.data(),
      oversizedRestatements.size(),
      kRestatementsSizeOffset,
      &oversizedChildSize,
      sizeof(oversizedChildSize));

  auto wrongDeltaEncoding = reference.encodedArtifact;
  wrongDeltaEncoding[kDeltasOffset] = static_cast<char>(EncodingType::Trivial);

  auto wrongDeltaType = reference.encodedArtifact;
  wrongDeltaType[kDeltasOffset + 1] = static_cast<char>(DataType::Uint64);

  auto wrongDeltaRowCount = reference.encodedArtifact;
  const uint32_t invalidRowCount = runnerConfig.rowCount + 1;
  checked_memcpy_offset(
      wrongDeltaRowCount.data(),
      wrongDeltaRowCount.size(),
      deltasRowCountOffset,
      &invalidRowCount,
      sizeof(invalidRowCount));

  auto compressedDeltas = reference.encodedArtifact;
  compressedDeltas[deltasCompressionOffset] =
      static_cast<char>(CompressionType::Zstd);

  auto zeroDeltaBitWidth = reference.encodedArtifact;
  zeroDeltaBitWidth[deltasBitWidthOffset] = 0;

  auto exactSizeZeroDeltaBitWidth = reference.encodedArtifact;
  const uint32_t zeroWidthDeltasSize = static_cast<uint32_t>(
      EncodingPrefix::kFixedPrefixSize + sizeof(uint8_t) + sizeof(uint32_t) +
      sizeof(uint8_t) + FixedBitArray::bufferSize(deltasRowCount, 0));
  ASSERT_LT(zeroWidthDeltasSize, deltasSize);
  exactSizeZeroDeltaBitWidth.erase(
      kDeltasOffset + zeroWidthDeltasSize, deltasSize - zeroWidthDeltasSize);
  checked_memcpy_offset(
      exactSizeZeroDeltaBitWidth.data(),
      exactSizeZeroDeltaBitWidth.size(),
      kDeltasSizeOffset,
      &zeroWidthDeltasSize,
      sizeof(zeroWidthDeltasSize));
  exactSizeZeroDeltaBitWidth[deltasBitWidthOffset] = 0;

  auto oversizedDeltaBitWidth = reference.encodedArtifact;
  oversizedDeltaBitWidth[deltasBitWidthOffset] = 33;

  auto wrongRestatementEncoding = reference.encodedArtifact;
  wrongRestatementEncoding[restatementsOffset] =
      static_cast<char>(EncodingType::FixedBitWidth);

  auto wrongRestatementType = reference.encodedArtifact;
  wrongRestatementType[restatementsOffset + 1] =
      static_cast<char>(DataType::Uint64);

  auto compressedRestatements = reference.encodedArtifact;
  compressedRestatements[restatementsCompressionOffset] =
      static_cast<char>(CompressionType::Zstd);

  auto wrongRestatementRowCount = reference.encodedArtifact;
  checked_memcpy_offset(
      wrongRestatementRowCount.data(),
      wrongRestatementRowCount.size(),
      restatementsRowCountOffset,
      &invalidRowCount,
      sizeof(invalidRowCount));

  auto wrongFlagRowCount = reference.encodedArtifact;
  checked_memcpy_offset(
      wrongFlagRowCount.data(),
      wrongFlagRowCount.size(),
      flagsRowCountOffset,
      &invalidRowCount,
      sizeof(invalidRowCount));

  auto wrongFlagEncoding = reference.encodedArtifact;
  wrongFlagEncoding[flagsOffset] =
      static_cast<char>(EncodingType::FixedBitWidth);

  auto wrongFlagType = reference.encodedArtifact;
  wrongFlagType[flagsOffset + 1] = static_cast<char>(DataType::Uint32);

  auto compressedFlags = reference.encodedArtifact;
  compressedFlags[flagsCompressionOffset] =
      static_cast<char>(CompressionType::Zstd);

  auto firstRowIsDelta = reference.encodedArtifact;
  firstRowIsDelta[flagsPayloadOffset] &= static_cast<char>(~uint8_t{1});

  auto mismatchedRestatementCount = reference.encodedArtifact;
  mismatchedRestatementCount[flagsPayloadOffset] |= static_cast<char>(2);

  auto nonZeroFlagPadding = reference.encodedArtifact;
  const size_t lastFlagByte =
      flagsPayloadOffset + (runnerConfig.rowCount + 7) / 8 - 1;
  nonZeroFlagPadding[lastFlagByte] |= static_cast<char>(0x80);

  auto nonZeroFlagSlop = reference.encodedArtifact;
  nonZeroFlagSlop.back() |= static_cast<char>(1);

  auto trailingPayload = reference.encodedArtifact;
  trailingPayload.push_back('\0');

  for (const auto& malformed :
       {oversizedDeltas,
        oversizedRestatements,
        wrongDeltaEncoding,
        wrongDeltaType,
        wrongDeltaRowCount,
        compressedDeltas,
        zeroDeltaBitWidth,
        exactSizeZeroDeltaBitWidth,
        oversizedDeltaBitWidth,
        wrongRestatementEncoding,
        wrongRestatementType,
        compressedRestatements,
        wrongRestatementRowCount,
        wrongFlagRowCount,
        wrongFlagEncoding,
        wrongFlagType,
        compressedFlags,
        firstRowIsDelta,
        mismatchedRestatementCount,
        nonZeroFlagPadding,
        nonZeroFlagSlop,
        trailingPayload}) {
    EXPECT_THROW(
        verifyEncodingArtifact(runnerConfig, malformed, *pool_),
        std::runtime_error);
  }
}

TEST_F(NimbleEncodingRunnerTest, truncatedSparseBoolArtifactsAreRejected) {
  const auto runnerConfig = config("nimble.sparse_bool.decode_dense.v1");
  const auto reference = runEncodingBenchmark(runnerConfig, *pool_);

  for (size_t size = 0; size < reference.encodedArtifact.size(); ++size) {
    const std::string truncated = reference.encodedArtifact.substr(0, size);
    EXPECT_THROW(
        verifyEncodingArtifact(runnerConfig, truncated, *pool_),
        std::runtime_error)
        << "accepted SparseBool artifact truncated to " << size << " bytes";
  }
}

TEST_F(NimbleEncodingRunnerTest, malformedSparseBoolMetadataIsRejected) {
  const auto runnerConfig = config("nimble.sparse_bool.decode_dense.v1");
  const auto reference = runEncodingBenchmark(runnerConfig, *pool_);
  constexpr size_t kSparseValueOffset = EncodingPrefix::kFixedPrefixSize;
  constexpr size_t kIndicesOffset = kSparseValueOffset + sizeof(uint8_t);
  constexpr size_t kIndicesRowCountOffset =
      kIndicesOffset + EncodingPrefix::kRowCountOffset;
  constexpr size_t kIndicesCompressionOffset =
      kIndicesOffset + EncodingPrefix::kFixedPrefixSize;
  constexpr size_t kIndicesBitWidthOffset =
      kIndicesCompressionOffset + sizeof(uint8_t) + sizeof(uint32_t);
  constexpr size_t kIndicesPayloadOffset =
      kIndicesBitWidthOffset + sizeof(uint8_t);

  uint32_t indexCount;
  checked_memcpy_robust(
      &indexCount,
      sizeof(indexCount),
      reference.encodedArtifact.data() + kIndicesRowCountOffset,
      reference.encodedArtifact.size() - kIndicesRowCountOffset,
      sizeof(indexCount));

  auto invalidSparseValue = reference.encodedArtifact;
  invalidSparseValue[kSparseValueOffset] = 2;

  auto wrongIndexEncoding = reference.encodedArtifact;
  wrongIndexEncoding[kIndicesOffset] = static_cast<char>(EncodingType::Trivial);

  auto wrongIndexType = reference.encodedArtifact;
  wrongIndexType[kIndicesOffset + 1] = static_cast<char>(DataType::Uint64);

  auto zeroIndexCount = reference.encodedArtifact;
  const uint32_t zero = 0;
  checked_memcpy_offset(
      zeroIndexCount.data(),
      zeroIndexCount.size(),
      kIndicesRowCountOffset,
      &zero,
      sizeof(zero));

  auto oversizedIndexCount = reference.encodedArtifact;
  const uint32_t oversizedCount = runnerConfig.rowCount + 2;
  checked_memcpy_offset(
      oversizedIndexCount.data(),
      oversizedIndexCount.size(),
      kIndicesRowCountOffset,
      &oversizedCount,
      sizeof(oversizedCount));

  auto compressedIndices = reference.encodedArtifact;
  compressedIndices[kIndicesCompressionOffset] =
      static_cast<char>(CompressionType::Zstd);

  auto invalidBitWidth = reference.encodedArtifact;
  invalidBitWidth[kIndicesBitWidthOffset] = 33;

  auto invalidSentinel = reference.encodedArtifact;
  const auto bitWidth =
      static_cast<uint8_t>(invalidSentinel[kIndicesBitWidthOffset]);
  FixedBitArray indices{
      invalidSentinel.data() + kIndicesPayloadOffset, bitWidth};
  indices.zeroAndSet(indexCount - 1, 0);

  auto duplicateIndex = reference.encodedArtifact;
  FixedBitArray duplicateIndices{
      duplicateIndex.data() + kIndicesPayloadOffset, bitWidth};
  duplicateIndices.zeroAndSet(1, 0);

  auto outOfRangeIndex = reference.encodedArtifact;
  FixedBitArray outOfRangeIndices{
      outOfRangeIndex.data() + kIndicesPayloadOffset, bitWidth};
  outOfRangeIndices.zeroAndSet(indexCount - 2, runnerConfig.rowCount);

  auto trailingPayload = reference.encodedArtifact;
  trailingPayload.push_back('\0');

  for (const auto& malformed :
       {invalidSparseValue,
        wrongIndexEncoding,
        wrongIndexType,
        zeroIndexCount,
        oversizedIndexCount,
        compressedIndices,
        invalidBitWidth,
        invalidSentinel,
        duplicateIndex,
        outOfRangeIndex,
        trailingPayload}) {
    EXPECT_THROW(
        verifyEncodingArtifact(runnerConfig, malformed, *pool_),
        std::runtime_error);
  }
}

TEST_F(NimbleEncodingRunnerTest, truncatedPforArtifactsAreRejected) {
  const auto runnerConfig = config("nimble.pfor.decode_dense.v1");
  const auto reference = runEncodingBenchmark(runnerConfig, *pool_);

  for (size_t size = 0; size < reference.encodedArtifact.size(); ++size) {
    const std::string truncated = reference.encodedArtifact.substr(0, size);
    EXPECT_THROW(
        verifyEncodingArtifact(runnerConfig, truncated, *pool_),
        std::runtime_error)
        << "accepted PFOR artifact truncated to " << size << " bytes";
  }
}

TEST_F(NimbleEncodingRunnerTest, pforPackedResidualOverflowIsRejected) {
  auto runnerConfig = config("nimble.pfor.decode_dense.v1");
  runnerConfig.rowCount = 100;
  const auto reference = runEncodingBenchmark(runnerConfig, *pool_);

  std::string wrappingArtifact =
      reference.encodedArtifact.substr(0, EncodingPrefix::kFixedPrefixSize);
  const size_t baselineOffset = wrappingArtifact.size();
  wrappingArtifact.resize(baselineOffset + sizeof(uint32_t));
  writeUint32(
      wrappingArtifact, baselineOffset, std::numeric_limits<uint32_t>::max());
  constexpr uint8_t kFullBitWidth = 32;
  wrappingArtifact.push_back(static_cast<char>(kFullBitWidth));
  wrappingArtifact.append(3, '\0');

  const size_t packedOffset = wrappingArtifact.size();
  const auto packedSize = static_cast<size_t>(
      FixedBitArray::bufferSize(runnerConfig.rowCount, kFullBitWidth));
  wrappingArtifact.resize(packedOffset + packedSize, '\0');
  FixedBitArray packedResiduals{
      wrappingArtifact.data() + packedOffset, kFullBitWidth};
  for (uint32_t row = 0; row < runnerConfig.rowCount; ++row) {
    packedResiduals.zeroAndSet(
        row, pforBenchmarkValue(row, runnerConfig.seed) + uint32_t{1});
  }

  EXPECT_THROW(
      verifyEncodingArtifact(runnerConfig, wrappingArtifact, *pool_),
      std::runtime_error);
}

TEST_F(NimbleEncodingRunnerTest, malformedPforMetadataIsRejected) {
  auto runnerConfig = config("nimble.pfor.decode_dense.v1");
  runnerConfig.rowCount = 100;
  const auto reference = runEncodingBenchmark(runnerConfig, *pool_);
  const auto offsets = readPforArtifactOffsets(reference.encodedArtifact);
  ASSERT_GE(offsets.numExceptions, 2);
  ASSERT_LT(offsets.positions, reference.encodedArtifact.size());
  ASSERT_LT(offsets.values, reference.encodedArtifact.size());
  ASSERT_LT(offsets.packed, reference.encodedArtifact.size());

  auto invalidBaseBitWidth = reference.encodedArtifact;
  invalidBaseBitWidth[offsets.baseBitWidth] = 33;

  auto oversizedExceptionCount = reference.encodedArtifact;
  oversizedExceptionCount[offsets.exceptionCount] =
      static_cast<char>(runnerConfig.rowCount + 1);

  auto zeroCountWithChildren = reference.encodedArtifact;
  zeroCountWithChildren[offsets.exceptionCount] = 0;

  auto truncatedExceptionCount = reference.encodedArtifact.substr(
      0, offsets.exceptionCount + sizeof(uint8_t));
  truncatedExceptionCount[offsets.exceptionCount] = static_cast<char>(0x80);

  auto overlongExceptionCount = reference.encodedArtifact;
  for (size_t byte = 0; byte < 5; ++byte) {
    overlongExceptionCount[offsets.exceptionCount + byte] =
        static_cast<char>(0x80);
  }

  auto oversizedPositions = reference.encodedArtifact;
  oversizedPositions[offsets.positionsSize] = 0x7f;

  auto oversizedValues = reference.encodedArtifact;
  oversizedValues[offsets.valuesSize] = 0x7f;

  auto wrongPositionEncoding = reference.encodedArtifact;
  wrongPositionEncoding[offsets.positions] =
      static_cast<char>(EncodingType::Trivial);

  auto wrongPositionType = reference.encodedArtifact;
  wrongPositionType[offsets.positions + 1] =
      static_cast<char>(DataType::Uint64);

  auto wrongPositionRowCount = reference.encodedArtifact;
  writeUint32(
      wrongPositionRowCount,
      offsets.positions + EncodingPrefix::kRowCountOffset,
      offsets.numExceptions + 1);

  auto compressedPositions = reference.encodedArtifact;
  compressedPositions[offsets.positions + EncodingPrefix::kFixedPrefixSize] =
      static_cast<char>(CompressionType::Zstd);

  auto wrongValueEncoding = reference.encodedArtifact;
  wrongValueEncoding[offsets.values] = static_cast<char>(EncodingType::Trivial);

  auto wrongValueType = reference.encodedArtifact;
  wrongValueType[offsets.values + 1] = static_cast<char>(DataType::Uint64);

  auto wrongValueRowCount = reference.encodedArtifact;
  writeUint32(
      wrongValueRowCount,
      offsets.values + EncodingPrefix::kRowCountOffset,
      offsets.numExceptions + 1);

  auto compressedValues = reference.encodedArtifact;
  compressedValues[offsets.values + EncodingPrefix::kFixedPrefixSize] =
      static_cast<char>(CompressionType::Zstd);

  constexpr size_t kFixedBitWidthHeaderSize = EncodingPrefix::kFixedPrefixSize +
      sizeof(uint8_t) + sizeof(uint32_t) + sizeof(uint8_t);
  const size_t positionBaselineOffset =
      offsets.positions + EncodingPrefix::kFixedPrefixSize + sizeof(uint8_t);
  const size_t positionBitWidthOffset =
      positionBaselineOffset + sizeof(uint32_t);
  const size_t positionPayloadOffset = positionBitWidthOffset + sizeof(uint8_t);
  const uint32_t positionBaseline =
      readUint32(reference.encodedArtifact, positionBaselineOffset);
  const uint8_t positionBitWidth =
      static_cast<uint8_t>(reference.encodedArtifact[positionBitWidthOffset]);

  auto duplicatePosition = reference.encodedArtifact;
  FixedBitArray duplicatePositions{
      duplicatePosition.data() + positionPayloadOffset, positionBitWidth};
  duplicatePositions.zeroAndSet(1, duplicatePositions.get(0));

  auto outOfRangePosition = reference.encodedArtifact;
  FixedBitArray outOfRangePositions{
      outOfRangePosition.data() + positionPayloadOffset, positionBitWidth};
  outOfRangePositions.zeroAndSet(
      offsets.numExceptions - 1, runnerConfig.rowCount - positionBaseline);

  const size_t valueBaselineOffset =
      offsets.values + EncodingPrefix::kFixedPrefixSize + sizeof(uint8_t);
  const size_t valueBitWidthOffset = valueBaselineOffset + sizeof(uint32_t);
  auto residualWithinBaseMask = reference.encodedArtifact;
  writeUint32(residualWithinBaseMask, valueBaselineOffset, 0);

  auto overflowingResidual = reference.encodedArtifact;
  writeUint32(
      overflowingResidual,
      valueBaselineOffset,
      std::numeric_limits<uint32_t>::max());

  const uint8_t baseBitWidth =
      static_cast<uint8_t>(reference.encodedArtifact[offsets.baseBitWidth]);
  auto nonZeroExceptionBaseSlot = reference.encodedArtifact;
  const uint64_t firstExceptionPosition = positionBaseline +
      FixedBitArray{
          reference.encodedArtifact.data() + positionPayloadOffset,
          positionBitWidth}
          .get(0);
  FixedBitArray baseResiduals{
      nonZeroExceptionBaseSlot.data() + offsets.packed, baseBitWidth};
  baseResiduals.zeroAndSet(firstExceptionPosition, 1);

  auto invalidPositionBitWidth = reference.encodedArtifact;
  invalidPositionBitWidth[positionBitWidthOffset] = 33;

  auto invalidValueBitWidth = reference.encodedArtifact;
  invalidValueBitWidth[valueBitWidthOffset] = 33;

  auto nonZeroPackedSlop = reference.encodedArtifact;
  nonZeroPackedSlop.back() |= 1;

  auto trailingPayload = reference.encodedArtifact;
  trailingPayload.push_back('\0');

  EXPECT_EQ(offsets.positions + offsets.positionsBytes, offsets.valuesSize);
  EXPECT_EQ(offsets.values + offsets.valuesBytes, offsets.packed);
  EXPECT_GE(offsets.positionsBytes, kFixedBitWidthHeaderSize);
  EXPECT_GE(offsets.valuesBytes, kFixedBitWidthHeaderSize);

  for (const auto& malformed :
       {invalidBaseBitWidth,     oversizedExceptionCount,
        zeroCountWithChildren,   truncatedExceptionCount,
        overlongExceptionCount,  oversizedPositions,
        oversizedValues,         wrongPositionEncoding,
        wrongPositionType,       wrongPositionRowCount,
        compressedPositions,     wrongValueEncoding,
        wrongValueType,          wrongValueRowCount,
        compressedValues,        duplicatePosition,
        outOfRangePosition,      residualWithinBaseMask,
        overflowingResidual,     nonZeroExceptionBaseSlot,
        invalidPositionBitWidth, invalidValueBitWidth,
        nonZeroPackedSlop,       trailingPayload}) {
    EXPECT_THROW(
        verifyEncodingArtifact(runnerConfig, malformed, *pool_),
        std::runtime_error);
  }
}

TEST_F(NimbleEncodingRunnerTest, truncatedSimdForBitpackArtifactsAreRejected) {
  const auto runnerConfig = config("nimble.simd_for_bitpack.decode_dense.v1");
  const auto reference = runEncodingBenchmark(runnerConfig, *pool_);

  for (size_t size = 0; size < reference.encodedArtifact.size(); ++size) {
    const std::string truncated = reference.encodedArtifact.substr(0, size);
    EXPECT_THROW(
        verifyEncodingArtifact(runnerConfig, truncated, *pool_),
        std::runtime_error)
        << "accepted SIMD_FOR artifact truncated to " << size << " bytes";
  }
}

TEST_F(NimbleEncodingRunnerTest, malformedSimdForBitpackMetadataIsRejected) {
  const auto runnerConfig = config("nimble.simd_for_bitpack.decode_dense.v1");
  const auto reference = runEncodingBenchmark(runnerConfig, *pool_);
  constexpr size_t kBaselineOffset = EncodingPrefix::kFixedPrefixSize;
  constexpr size_t kBitWidthOffset = kBaselineOffset + sizeof(uint32_t);
  constexpr size_t kFirstGroupRowsOffset = kBitWidthOffset + sizeof(uint8_t);
  const char* cursor = reference.encodedArtifact.data() + kFirstGroupRowsOffset;
  EXPECT_EQ(32, varint::readVarint32(&cursor));
  const size_t payloadOffset = cursor - reference.encodedArtifact.data();

  auto wrongRootEncoding = reference.encodedArtifact;
  wrongRootEncoding[0] = static_cast<char>(EncodingType::PFOR);

  auto wrongDataType = reference.encodedArtifact;
  wrongDataType[1] = static_cast<char>(DataType::Uint64);

  auto wrongRowCount = reference.encodedArtifact;
  writeUint32(
      wrongRowCount,
      EncodingPrefix::kRowCountOffset,
      runnerConfig.rowCount + 1);

  auto invalidBitWidth = reference.encodedArtifact;
  invalidBitWidth[kBitWidthOffset] = 33;

  auto zeroFirstGroup = reference.encodedArtifact;
  zeroFirstGroup[kFirstGroupRowsOffset] = 0;

  auto oversizedFirstGroup = reference.encodedArtifact;
  oversizedFirstGroup[kFirstGroupRowsOffset] = 33;

  auto truncatedFirstGroup =
      reference.encodedArtifact.substr(0, kFirstGroupRowsOffset + 1);
  truncatedFirstGroup[kFirstGroupRowsOffset] = static_cast<char>(0x80);

  std::string overlongFirstGroup =
      reference.encodedArtifact.substr(0, kFirstGroupRowsOffset);
  overlongFirstGroup.append(5, static_cast<char>(0x80));
  overlongFirstGroup.append(reference.encodedArtifact.substr(payloadOffset));

  std::string nonMinimalFirstGroup =
      reference.encodedArtifact.substr(0, kFirstGroupRowsOffset);
  nonMinimalFirstGroup.push_back(static_cast<char>(0xA0));
  nonMinimalFirstGroup.push_back('\0');
  nonMinimalFirstGroup.append(reference.encodedArtifact.substr(payloadOffset));

  auto mismatchedBaseline = reference.encodedArtifact;
  writeUint32(
      mismatchedBaseline,
      kBaselineOffset,
      kSimdForBitpackBenchmarkBaseline + 1);

  auto trailingPayload = reference.encodedArtifact;
  trailingPayload.push_back('\0');

  for (const auto& malformed :
       {wrongRootEncoding,
        wrongDataType,
        wrongRowCount,
        invalidBitWidth,
        zeroFirstGroup,
        oversizedFirstGroup,
        truncatedFirstGroup,
        overlongFirstGroup,
        nonMinimalFirstGroup,
        mismatchedBaseline,
        trailingPayload}) {
    EXPECT_THROW(
        verifyEncodingArtifact(runnerConfig, malformed, *pool_),
        std::runtime_error);
  }
}

TEST_F(NimbleEncodingRunnerTest, truncatedBlockBitPackingArtifactsAreRejected) {
  const auto runnerConfig = config("nimble.block_bit_packing.decode_dense.v1");
  const auto reference = runEncodingBenchmark(runnerConfig, *pool_);

  for (size_t size = 0; size < reference.encodedArtifact.size(); ++size) {
    const std::string truncated = reference.encodedArtifact.substr(0, size);
    EXPECT_THROW(
        verifyEncodingArtifact(runnerConfig, truncated, *pool_),
        std::runtime_error)
        << "accepted BlockBitPacking artifact truncated to " << size
        << " bytes";
  }
}

TEST_F(NimbleEncodingRunnerTest, malformedBlockBitPackingMetadataIsRejected) {
  const auto runnerConfig = config("nimble.block_bit_packing.decode_dense.v1");
  const auto reference = runEncodingBenchmark(runnerConfig, *pool_);
  const auto offsets =
      readBlockBitPackingArtifactOffsets(reference.encodedArtifact);
  constexpr size_t kTrivialPayloadOffset =
      EncodingPrefix::kFixedPrefixSize + sizeof(uint8_t);
  ASSERT_EQ(1, offsets.blockCount);
  ASSERT_LT(offsets.packed, reference.encodedArtifact.size());

  auto compressedPayload = reference.encodedArtifact;
  compressedPayload[offsets.compression] =
      static_cast<char>(CompressionType::Zstd);

  auto zeroBlockSize = reference.encodedArtifact;
  zeroBlockSize[offsets.blockSize] = 0;

  auto oversizedBlockSize = reference.encodedArtifact;
  oversizedBlockSize[offsets.blockSize] = static_cast<char>(0xFF);
  oversizedBlockSize[offsets.blockSize + 1] = 0x7F;

  auto zeroBlockCount = reference.encodedArtifact;
  zeroBlockCount[offsets.numBlocks] = 0;

  auto wrongBaselineEncoding = reference.encodedArtifact;
  wrongBaselineEncoding[offsets.baselines] =
      static_cast<char>(EncodingType::FixedBitWidth);

  auto wrongBaselineType = reference.encodedArtifact;
  wrongBaselineType[offsets.baselines + 1] =
      static_cast<char>(DataType::Uint64);

  auto wrongBaselineRows = reference.encodedArtifact;
  writeUint32(
      wrongBaselineRows,
      offsets.baselines + EncodingPrefix::kRowCountOffset,
      offsets.blockCount + 1);

  auto wrongBitWidthType = reference.encodedArtifact;
  wrongBitWidthType[offsets.bitWidths + 1] =
      static_cast<char>(DataType::Uint32);

  auto wrongBlockOffsetEncoding = reference.encodedArtifact;
  wrongBlockOffsetEncoding[offsets.blockOffsets] =
      static_cast<char>(EncodingType::FixedBitWidth);

  auto wrongBlockOffsetType = reference.encodedArtifact;
  wrongBlockOffsetType[offsets.blockOffsets + 1] =
      static_cast<char>(DataType::Uint64);

  auto wrongBlockOffsetRows = reference.encodedArtifact;
  writeUint32(
      wrongBlockOffsetRows,
      offsets.blockOffsets + EncodingPrefix::kRowCountOffset,
      offsets.blockCount + 1);

  auto invalidBitWidth = reference.encodedArtifact;
  invalidBitWidth[offsets.bitWidths + kTrivialPayloadOffset] = 33;

  auto rawBitWidthWithPackedPayload = reference.encodedArtifact;
  rawBitWidthWithPackedPayload[offsets.bitWidths + kTrivialPayloadOffset] =
      static_cast<char>(0xFF);

  auto nonZeroFirstOffset = reference.encodedArtifact;
  writeUint32(
      nonZeroFirstOffset, offsets.blockOffsets + kTrivialPayloadOffset, 1);

  auto overflowingBaseline = reference.encodedArtifact;
  writeUint32(
      overflowingBaseline,
      offsets.baselines + kTrivialPayloadOffset,
      std::numeric_limits<uint32_t>::max());

  std::string nonMinimalFirstBlockRows =
      reference.encodedArtifact.substr(0, offsets.firstBlockRows);
  nonMinimalFirstBlockRows.push_back(static_cast<char>(0x81));
  nonMinimalFirstBlockRows.push_back(static_cast<char>(0x82));
  nonMinimalFirstBlockRows.push_back('\0');
  nonMinimalFirstBlockRows.append(
      reference.encodedArtifact.substr(offsets.packed));

  auto zeroFirstBlockRows = reference.encodedArtifact;
  zeroFirstBlockRows[offsets.firstBlockRows] = 0;

  auto oversizedFirstBlockRows = reference.encodedArtifact;
  oversizedFirstBlockRows[offsets.firstBlockRows] = static_cast<char>(0x82);

  auto nonZeroPackedPadding = reference.encodedArtifact;
  nonZeroPackedPadding.back() |= static_cast<char>(0x80);

  auto trailingPayload = reference.encodedArtifact;
  trailingPayload.push_back('\0');

  for (const auto& malformed :
       {compressedPayload,
        zeroBlockSize,
        oversizedBlockSize,
        zeroBlockCount,
        wrongBaselineEncoding,
        wrongBaselineType,
        wrongBaselineRows,
        wrongBitWidthType,
        wrongBlockOffsetEncoding,
        wrongBlockOffsetType,
        wrongBlockOffsetRows,
        invalidBitWidth,
        rawBitWidthWithPackedPayload,
        nonZeroFirstOffset,
        overflowingBaseline,
        nonMinimalFirstBlockRows,
        zeroFirstBlockRows,
        oversizedFirstBlockRows,
        nonZeroPackedPadding,
        trailingPayload}) {
    EXPECT_THROW(
        verifyEncodingArtifact(runnerConfig, malformed, *pool_),
        std::runtime_error);
  }
}

TEST_F(NimbleEncodingRunnerTest, truncatedVarintArtifactsAreRejected) {
  const auto runnerConfig = config("nimble.varint.decode_dense.v1");
  const auto reference = runEncodingBenchmark(runnerConfig, *pool_);

  for (size_t size = 0; size < reference.encodedArtifact.size(); ++size) {
    const std::string truncated = reference.encodedArtifact.substr(0, size);
    EXPECT_THROW(
        verifyEncodingArtifact(runnerConfig, truncated, *pool_),
        std::runtime_error)
        << "accepted Varint artifact truncated to " << size << " bytes";
  }
}

TEST_F(NimbleEncodingRunnerTest, malformedVarintPayloadIsRejected) {
  const auto runnerConfig = config("nimble.varint.decode_dense.v1");
  const auto reference = runEncodingBenchmark(runnerConfig, *pool_);
  constexpr size_t kBaselineOffset = EncodingPrefix::kFixedPrefixSize;
  constexpr size_t kPayloadOffset = kBaselineOffset + sizeof(uint32_t);

  auto wrongRootEncoding = reference.encodedArtifact;
  wrongRootEncoding[0] = static_cast<char>(EncodingType::Trivial);

  auto wrongDataType = reference.encodedArtifact;
  wrongDataType[1] = static_cast<char>(DataType::Uint64);

  auto wrongRowCount = reference.encodedArtifact;
  writeUint32(
      wrongRowCount,
      EncodingPrefix::kRowCountOffset,
      runnerConfig.rowCount + 1);

  std::string continuedAtEnd =
      reference.encodedArtifact.substr(0, kPayloadOffset);
  continuedAtEnd.push_back(static_cast<char>(0x80));

  std::string nonMinimalResidual =
      reference.encodedArtifact.substr(0, kPayloadOffset);
  nonMinimalResidual.push_back(static_cast<char>(0x80));
  nonMinimalResidual.push_back('\0');
  nonMinimalResidual.append(
      reference.encodedArtifact.substr(kPayloadOffset + 1));

  std::string nonMinimalFiveByteResidual =
      reference.encodedArtifact.substr(0, kPayloadOffset);
  nonMinimalFiveByteResidual.append(4, static_cast<char>(0x80));
  nonMinimalFiveByteResidual.push_back('\0');
  nonMinimalFiveByteResidual.append(
      reference.encodedArtifact.substr(kPayloadOffset + 1));

  std::string overflowingResidual =
      reference.encodedArtifact.substr(0, kPayloadOffset);
  overflowingResidual.append(4, static_cast<char>(0x80));
  overflowingResidual.push_back(static_cast<char>(0x10));
  overflowingResidual.append(
      reference.encodedArtifact.substr(kPayloadOffset + 1));

  auto overflowingBaseline = reference.encodedArtifact;
  writeUint32(
      overflowingBaseline,
      kBaselineOffset,
      varintBenchmarkBaseline(runnerConfig.seed) + 1);

  auto missingMinimum = reference.encodedArtifact;
  const char* cursor = reference.encodedArtifact.data() + kPayloadOffset;
  for (uint32_t row = 0; row < runnerConfig.rowCount; ++row) {
    const size_t residualOffset = cursor - reference.encodedArtifact.data();
    if (varint::readVarint32(&cursor) == 0) {
      missingMinimum[residualOffset] = 1;
    }
  }

  auto trailingPayload = reference.encodedArtifact;
  trailingPayload.push_back('\0');

  for (const auto& malformed :
       {wrongRootEncoding,
        wrongDataType,
        wrongRowCount,
        continuedAtEnd,
        nonMinimalResidual,
        nonMinimalFiveByteResidual,
        overflowingResidual,
        overflowingBaseline,
        missingMinimum,
        trailingPayload}) {
    EXPECT_THROW(
        verifyEncodingArtifact(runnerConfig, malformed, *pool_),
        std::runtime_error);
  }
}

TEST_F(NimbleEncodingRunnerTest, prefixSupportsOnlyItsThreeStringLanes) {
  for (const std::string_view lane : {"encode", "decode_dense", "skip_seek"}) {
    SCOPED_TRACE(lane);
    const auto runnerConfig =
        config("nimble.prefix." + std::string{lane} + ".v1");
    const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
    EXPECT_EQ("prefix", measurement.encoding);
    EXPECT_EQ("string", measurement.dataType);
    EXPECT_EQ(lane, measurement.lane);
    EXPECT_EQ(measurement.inputDigest, measurement.outputDigest);
  }

  for (const std::string_view lane :
       {"decode_construct",
        "decode_range50",
        "decode_scatter10",
        "decode_scatter1",
        "view_random",
        "slice",
        "selection_e2e"}) {
    EXPECT_THROW(
        runEncodingBenchmark(
            config("nimble.prefix." + std::string{lane} + ".v1"), *pool_),
        std::invalid_argument)
        << lane;
  }
}

TEST_F(NimbleEncodingRunnerTest, prefixCrossVerifiesBoundaryRowsAndSeeds) {
  struct Boundary {
    uint32_t rows;
    uint64_t seed;
  };
  constexpr std::array<Boundary, 3> kBoundaries{{
      {100, 0},
      {256, 0xC0FFEE},
      {257, static_cast<uint64_t>(std::numeric_limits<int64_t>::max())},
  }};
  for (const auto& boundary : kBoundaries) {
    for (const std::string_view lane :
         {"encode", "decode_dense", "skip_seek"}) {
      SCOPED_TRACE(
          std::to_string(boundary.rows) + "/" + std::to_string(boundary.seed) +
          "/" + std::string{lane});
      auto runnerConfig = config("nimble.prefix." + std::string{lane} + ".v1");
      runnerConfig.rowCount = boundary.rows;
      runnerConfig.seed = boundary.seed;
      const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
      const auto verification = verifyEncodingArtifact(
          runnerConfig, measurement.encodedArtifact, *pool_);
      EXPECT_EQ(measurement.inputDigest, verification.outputDigest);
      EXPECT_EQ(measurement.artifactDigest, verification.artifactDigest);
    }
  }
}

TEST_F(NimbleEncodingRunnerTest, prefixConsumersKeepCanonicalArtifactPristine) {
  auto encodeConfig = config("nimble.prefix.encode.v1");
  const auto reference = runEncodingBenchmark(encodeConfig, *pool_);
  const std::string pristine = reference.encodedArtifact;

  for (const std::string_view lane : {"decode_dense", "skip_seek"}) {
    auto consumerConfig = config("nimble.prefix." + std::string{lane} + ".v1");
    const auto measurement = runEncodingBenchmark(
        consumerConfig, *pool_, std::string_view{reference.encodedArtifact});
    EXPECT_EQ(pristine, reference.encodedArtifact);
    EXPECT_EQ(pristine, measurement.encodedArtifact);
    EXPECT_EQ(reference.artifactDigest, measurement.artifactDigest);
    EXPECT_EQ(reference.inputDigest, measurement.outputDigest);
  }
}

TEST_F(NimbleEncodingRunnerTest, stringDigestUsesContentNotViewBacking) {
  const std::vector<std::string> firstStorage{
      "", std::string{"a\0b", 3}, "same", "tail"};
  const std::vector<std::string> secondStorage = firstStorage;
  const std::array<std::string_view, 4> firstViews{
      firstStorage[0], firstStorage[1], firstStorage[2], firstStorage[3]};
  const std::array<std::string_view, 4> secondViews{
      secondStorage[0], secondStorage[1], secondStorage[2], secondStorage[3]};
  const std::array<bool, 4> nonNulls{true, true, false, true};

  const auto first =
      detail::semanticStringDigestForTesting(firstViews, nonNulls);
  const auto second =
      detail::semanticStringDigestForTesting(secondViews, nonNulls);
  EXPECT_EQ(first, second);

  auto changedStorage = secondStorage;
  changedStorage.back().back() = 'X';
  const std::array<std::string_view, 4> changedViews{
      changedStorage[0],
      changedStorage[1],
      changedStorage[2],
      changedStorage[3]};
  EXPECT_NE(
      first, detail::semanticStringDigestForTesting(changedViews, nonNulls));

  changedStorage = secondStorage;
  changedStorage[2] = "ignored-null-content";
  const std::array<std::string_view, 4> changedNullViews{
      changedStorage[0],
      changedStorage[1],
      changedStorage[2],
      changedStorage[3]};
  EXPECT_EQ(
      first,
      detail::semanticStringDigestForTesting(changedNullViews, nonNulls));
  auto changedNulls = nonNulls;
  changedNulls[2] = true;
  EXPECT_NE(
      first, detail::semanticStringDigestForTesting(firstViews, changedNulls));
}

TEST_F(NimbleEncodingRunnerTest, prefixDefaultCorpusDigestIsStable) {
  auto runnerConfig = config("nimble.prefix.encode.v1");
  runnerConfig.rowCount = kPrefixBenchmarkDefaultRowCount;
  runnerConfig.seed = kPrefixBenchmarkDefaultSeed;
  const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
  EXPECT_EQ(
      "94190f95466dd1ef33d8255fe8d6dfe0b20155b76b5331711dd1c0185e8a1b98",
      measurement.inputDigest);
}

TEST_F(NimbleEncodingRunnerTest, malformedPrefixArtifactsAreRejected) {
  const auto runnerConfig = config("nimble.prefix.decode_dense.v1");
  const auto reference = runEncodingBenchmark(runnerConfig, *pool_);
  const auto offsets = readPrefixArtifactOffsets(
      reference.encodedArtifact, runnerConfig.rowCount);

  for (const size_t size :
       {size_t{0},
        size_t{EncodingPrefix::kFixedPrefixSize - 1},
        offsets.restartOffsets - 1,
        offsets.entries - 1,
        reference.encodedArtifact.size() - 1}) {
    EXPECT_THROW(
        verifyEncodingArtifact(
            runnerConfig, reference.encodedArtifact.substr(0, size), *pool_),
        std::runtime_error)
        << "accepted Prefix artifact truncated to " << size << " bytes";
  }

  auto wrongRootEncoding = reference.encodedArtifact;
  wrongRootEncoding[0] = static_cast<char>(EncodingType::Trivial);

  auto wrongDataType = reference.encodedArtifact;
  wrongDataType[1] = static_cast<char>(DataType::Uint64);

  auto wrongRowCount = reference.encodedArtifact;
  writeUint32(
      wrongRowCount,
      EncodingPrefix::kRowCountOffset,
      runnerConfig.rowCount + 1);

  auto zeroInterval = reference.encodedArtifact;
  writeUint32(zeroInterval, EncodingPrefix::kFixedPrefixSize, 0);

  auto wrongFirstOffset = reference.encodedArtifact;
  writeUint32(wrongFirstOffset, offsets.restartOffsets, 1);

  auto wrongLaterOffset = reference.encodedArtifact;
  writeUint32(
      wrongLaterOffset,
      offsets.restartOffsets + sizeof(uint32_t),
      readUint32(wrongLaterOffset, offsets.restartOffsets + sizeof(uint32_t)) +
          1);

  auto restartShared = reference.encodedArtifact;
  writeUint32(restartShared, offsets.entryOffsets[offsets.restartInterval], 1);

  auto excessiveShared = reference.encodedArtifact;
  writeUint32(excessiveShared, offsets.entryOffsets[1], UINT32_MAX);

  auto truncatedSuffix = reference.encodedArtifact;
  writeUint32(
      truncatedSuffix,
      offsets.entryOffsets[1] + sizeof(uint32_t),
      static_cast<uint32_t>(truncatedSuffix.size()));

  auto excessiveDecodedLength = reference.encodedArtifact;
  writeUint32(
      excessiveDecodedLength,
      offsets.entryOffsets[1] + sizeof(uint32_t),
      UINT32_MAX);

  auto trailingPayload = reference.encodedArtifact;
  trailingPayload.push_back('\0');

  for (const auto& malformed :
       {wrongRootEncoding,
        wrongDataType,
        wrongRowCount,
        zeroInterval,
        wrongFirstOffset,
        wrongLaterOffset,
        restartShared,
        excessiveShared,
        truncatedSuffix,
        excessiveDecodedLength,
        trailingPayload}) {
    EXPECT_THROW(
        verifyEncodingArtifact(runnerConfig, malformed, *pool_),
        std::runtime_error);
  }
}

TEST_F(NimbleEncodingRunnerTest, fsstSupportsOnlyItsThreeStringLanes) {
  for (const std::string_view lane : {"encode", "decode_dense", "skip_seek"}) {
    SCOPED_TRACE(lane);
    auto runnerConfig = config("nimble.fsst." + std::string{lane} + ".v1");
    runnerConfig.rowCount = kFsstBenchmarkDefaultRowCount;
    const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
    EXPECT_EQ("fsst", measurement.encoding);
    EXPECT_EQ("string", measurement.dataType);
    EXPECT_EQ(lane, measurement.lane);
    EXPECT_EQ(measurement.inputDigest, measurement.outputDigest);
  }

  for (const std::string_view lane :
       {"decode_construct",
        "decode_range50",
        "decode_scatter10",
        "decode_scatter1",
        "view_random",
        "slice",
        "selection_e2e"}) {
    EXPECT_THROW(
        runEncodingBenchmark(
            config("nimble.fsst." + std::string{lane} + ".v1"), *pool_),
        std::invalid_argument)
        << lane;
  }
}

TEST_F(NimbleEncodingRunnerTest, fsstConsumersKeepCanonicalArtifactPristine) {
  auto encodeConfig = config("nimble.fsst.encode.v1");
  encodeConfig.rowCount = kFsstBenchmarkDefaultRowCount;
  const auto reference = runEncodingBenchmark(encodeConfig, *pool_);
  const std::string pristine = reference.encodedArtifact;
  const auto corpus =
      makeFsstBenchmarkCorpus(encodeConfig.rowCount, encodeConfig.seed);

  EXPECT_LT(reference.encodedBytes, corpus.rawBytes);
  for (const std::string_view lane : {"decode_dense", "skip_seek"}) {
    auto consumerConfig = config("nimble.fsst." + std::string{lane} + ".v1");
    consumerConfig.rowCount = encodeConfig.rowCount;
    const auto measurement = runEncodingBenchmark(
        consumerConfig, *pool_, std::string_view{reference.encodedArtifact});
    EXPECT_EQ(pristine, reference.encodedArtifact);
    EXPECT_EQ(pristine, measurement.encodedArtifact);
    EXPECT_EQ(reference.artifactDigest, measurement.artifactDigest);
    EXPECT_EQ(reference.inputDigest, measurement.outputDigest);
  }
}

TEST_F(NimbleEncodingRunnerTest, malformedFsstArtifactsAreRejected) {
  auto runnerConfig = config("nimble.fsst.decode_dense.v1");
  runnerConfig.rowCount = 100;
  const auto reference = runEncodingBenchmark(runnerConfig, *pool_);
  const auto offsets = readFsstArtifactOffsets(reference.encodedArtifact);

  for (const size_t size :
       {size_t{0},
        size_t{EncodingPrefix::kFixedPrefixSize - 1},
        offsets.symbolTable - 1,
        offsets.symbolTable + offsets.symbolTableBytes - 1,
        offsets.lengths - 1,
        offsets.lengths + offsets.lengthsBytes - 1,
        reference.encodedArtifact.size() - 1}) {
    EXPECT_THROW(
        verifyEncodingArtifact(
            runnerConfig, reference.encodedArtifact.substr(0, size), *pool_),
        std::runtime_error)
        << "accepted FSST artifact truncated to " << size << " bytes";
  }

  auto wrongRootEncoding = reference.encodedArtifact;
  wrongRootEncoding[0] = static_cast<char>(EncodingType::Trivial);

  auto wrongDataType = reference.encodedArtifact;
  wrongDataType[1] = static_cast<char>(DataType::Uint64);

  auto wrongRowCount = reference.encodedArtifact;
  writeUint32(
      wrongRowCount,
      EncodingPrefix::kRowCountOffset,
      runnerConfig.rowCount + 1);

  auto zeroSymbolTable = reference.encodedArtifact;
  zeroSymbolTable[offsets.symbolTableSize] = 0;

  auto nonCanonicalSymbolTableSize = reference.encodedArtifact;
  nonCanonicalSymbolTableSize.replace(
      offsets.symbolTableSize,
      offsets.symbolTable - offsets.symbolTableSize,
      std::string{"\x80\x00", 2});

  auto overflowingSymbolTableSize = reference.encodedArtifact;
  overflowingSymbolTableSize.replace(
      offsets.symbolTableSize,
      offsets.symbolTable - offsets.symbolTableSize,
      std::string{"\xff\xff\xff\xff\x10", 5});

  auto continuedSymbolTableSize = reference.encodedArtifact;
  continuedSymbolTableSize.replace(
      offsets.symbolTableSize,
      offsets.symbolTable - offsets.symbolTableSize,
      std::string{"\x80\x80\x80\x80\x80", 5});

  auto oversizedSymbolTable = reference.encodedArtifact;
  oversizedSymbolTable.replace(
      offsets.symbolTableSize,
      offsets.symbolTable - offsets.symbolTableSize,
      std::string{"\xff\xff\xff\xff\x0f", 5});

  auto zeroLengths = reference.encodedArtifact;
  zeroLengths[offsets.lengthsSize] = 0;

  auto oversizedLengths = reference.encodedArtifact;
  oversizedLengths.replace(
      offsets.lengthsSize,
      offsets.lengths - offsets.lengthsSize,
      std::string{"\xff\xff\xff\xff\x0f", 5});

  auto lengthsPayloadIncludesBlob = reference.encodedArtifact;
  writeVarint32WithWidth(
      lengthsPayloadIncludesBlob,
      offsets.lengthsSize,
      offsets.lengths - offsets.lengthsSize,
      offsets.lengthsBytes + 1);

  auto wrongLengthsEncoding = reference.encodedArtifact;
  wrongLengthsEncoding[offsets.lengths] =
      static_cast<char>(EncodingType::Trivial);

  auto wrongLengthsType = reference.encodedArtifact;
  wrongLengthsType[offsets.lengths + 1] = static_cast<char>(DataType::Uint64);

  auto wrongLengthsRows = reference.encodedArtifact;
  writeUint32(
      wrongLengthsRows,
      offsets.lengths + EncodingPrefix::kRowCountOffset,
      runnerConfig.rowCount + 1);

  auto compressedLengths = reference.encodedArtifact;
  compressedLengths[offsets.lengths + EncodingPrefix::kFixedPrefixSize] =
      static_cast<char>(CompressionType::Zstd);

  constexpr size_t kLengthsBitWidthOffset =
      EncodingPrefix::kFixedPrefixSize + sizeof(uint8_t) + sizeof(uint32_t);
  auto zeroLengthsBitWidth = reference.encodedArtifact;
  zeroLengthsBitWidth[offsets.lengths + kLengthsBitWidthOffset] = 0;

  constexpr size_t kLengthsBaselineOffset =
      EncodingPrefix::kFixedPrefixSize + sizeof(uint8_t);
  auto oversizedRowExpansion = reference.encodedArtifact;
  writeUint32(
      oversizedRowExpansion,
      offsets.lengths + kLengthsBaselineOffset,
      16 * 1024 * 1024 / 8 + 1);
  expectRuntimeErrorContaining(
      [&] {
        verifyEncodingArtifact(runnerConfig, oversizedRowExpansion, *pool_);
      },
      "per-row expansion");

  auto oversizedTotalExpansion = reference.encodedArtifact;
  writeUint32(
      oversizedTotalExpansion,
      offsets.lengths + kLengthsBaselineOffset,
      static_cast<uint32_t>(
          (256ULL * 1024 * 1024 / 8) / runnerConfig.rowCount + 1));
  expectRuntimeErrorContaining(
      [&] {
        verifyEncodingArtifact(runnerConfig, oversizedTotalExpansion, *pool_);
      },
      "total expansion");

  auto mismatchedLengthSum = reference.encodedArtifact;
  const auto lengthBitWidth = static_cast<uint8_t>(
      mismatchedLengthSum[offsets.lengths + kLengthsBitWidthOffset]);
  constexpr size_t kLengthsPayloadOffset =
      kLengthsBitWidthOffset + sizeof(uint8_t);
  FixedBitArray{
      mismatchedLengthSum.data() + offsets.lengths + kLengthsPayloadOffset,
      lengthBitWidth}
      .zeroAndSet(0, 1);

  auto escapeAtRowEnd = reference.encodedArtifact;
  size_t compressedOffset{0};
  bool mutatedEscape{false};
  for (uint32_t row = 0; row < runnerConfig.rowCount; ++row) {
    const uint32_t length =
        readFsstCompressedLength(reference.encodedArtifact, offsets, row);
    if (!mutatedEscape && length > 0) {
      escapeAtRowEnd[offsets.blob + compressedOffset + length - 1] =
          static_cast<char>(255);
      mutatedEscape = true;
    }
    compressedOffset += length;
  }
  ASSERT_TRUE(mutatedEscape);

  auto trailingPayload = reference.encodedArtifact;
  trailingPayload.push_back('\0');

  for (const auto& malformed :
       {wrongRootEncoding,
        wrongDataType,
        wrongRowCount,
        zeroSymbolTable,
        nonCanonicalSymbolTableSize,
        overflowingSymbolTableSize,
        continuedSymbolTableSize,
        oversizedSymbolTable,
        zeroLengths,
        oversizedLengths,
        lengthsPayloadIncludesBlob,
        wrongLengthsEncoding,
        wrongLengthsType,
        wrongLengthsRows,
        compressedLengths,
        zeroLengthsBitWidth,
        mismatchedLengthSum,
        escapeAtRowEnd,
        trailingPayload}) {
    EXPECT_THROW(
        verifyEncodingArtifact(runnerConfig, malformed, *pool_),
        std::runtime_error);
  }
}

TEST_F(NimbleEncodingRunnerTest, rawMeasurementDoesNotClaimFinalGraderScore) {
  const auto measurement =
      runEncodingBenchmark(config("nimble.alp.decode_range50.v1"), *pool_);

  const auto json = folly::parseJson(measurementToJson(measurement));

  EXPECT_EQ(1, json["schema_version"].asInt());
  EXPECT_EQ("nimble_encoding_measurement", json["kind"].asString());
  EXPECT_EQ(measurement.taskId, json["task_id"].asString());
  EXPECT_EQ(measurement.lane, json["lane"].asString());
  EXPECT_TRUE(json["correctness"].asBool());
  EXPECT_EQ(measurement.samplesSeconds.size(), json["samples_seconds"].size());
  EXPECT_EQ(0, json.count("reward"));
  EXPECT_EQ(0, json.count("baseline_seconds"));
  EXPECT_EQ(0, json.count("encoded_artifact"));
}

TEST_F(NimbleEncodingRunnerTest, invalidRawMeasurementIsNotSerialized) {
  auto measurement =
      runEncodingBenchmark(config("nimble.alp.decode_dense.v1"), *pool_);
  measurement.samplesSeconds.front() = 0.0;

  EXPECT_THROW(measurementToJson(measurement), std::runtime_error);
}

} // namespace
} // namespace facebook::nimble::benchmarks
