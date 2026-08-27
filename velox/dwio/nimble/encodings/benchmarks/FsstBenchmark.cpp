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

// Micro-benchmarks comparing string compression: FSST vs Zstd vs Zstrong.
// Measures encode throughput, decode throughput, and compression ratio.
//
// Based on the FSST paper (VLDB 2020) benchmark methodology:
// - Varied string workloads: URLs, log messages, UUIDs (high entropy)
// - Throughput measured in MB/s of raw (uncompressed) data
// - Compression ratio = raw_size / compressed_size

#include <numeric>

#include "folly/Benchmark.h"
#include "folly/init/Init.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/compression/Compression.h"
#include "velox/dwio/nimble/encodings/FsstEncoding.h"

using namespace facebook::nimble;

namespace {

// ---------------------------------------------------------------------------
// Data generators — produce realistic string workloads.
// ---------------------------------------------------------------------------

struct StringDataset {
  std::vector<std::string> storage;
  std::vector<std::string_view> views;
  size_t totalRawBytes{0};

  // Concatenated blob for block compressors (Zstd/Zstrong).
  std::string blob;

  void finalize() {
    views.reserve(storage.size());
    totalRawBytes = 0;
    for (const auto& s : storage) {
      views.emplace_back(s);
      totalRawBytes += s.size();
    }
    blob.clear();
    blob.reserve(totalRawBytes);
    for (const auto& s : storage) {
      blob.append(s);
    }
  }
};

StringDataset makeUrls(int n) {
  StringDataset ds;
  ds.storage.reserve(n);
  for (int i = 0; i < n; ++i) {
    ds.storage.emplace_back(
        fmt::format(
            "https://www.example.com/api/v2/users/{}/profile?session=abc{}def&lang=en",
            i,
            i * 7));
  }
  ds.finalize();
  return ds;
}

StringDataset makeLogMessages(int n) {
  StringDataset ds;
  ds.storage.reserve(n);
  const std::array<std::string_view, 5> levels = {
      "INFO", "WARN", "ERROR", "DEBUG", "TRACE"};
  const std::array<std::string_view, 4> components = {
      "StorageEngine", "QueryPlanner", "NetworkHandler", "CacheManager"};
  for (int i = 0; i < n; ++i) {
    ds.storage.emplace_back(
        fmt::format(
            "[{}] {} - Processing request #{} from client 10.0.{}.{}: operation completed in {}ms",
            levels[i % levels.size()],
            components[i % components.size()],
            i,
            (i / 256) % 256,
            i % 256,
            (i * 13) % 5'000));
  }
  ds.finalize();
  return ds;
}

StringDataset makeUuids(int n) {
  StringDataset ds;
  ds.storage.reserve(n);
  for (int i = 0; i < n; ++i) {
    ds.storage.emplace_back(
        fmt::format(
            "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
            static_cast<uint32_t>(i * 2654435761u),
            (i * 31) & 0xFFFF,
            i & 0xFFF,
            0x8000 | (i & 0x3FFF),
            static_cast<uint64_t>(i) * 6364136223846793005ULL));
  }
  ds.finalize();
  return ds;
}

constexpr int kNumStrings = 100'000;

// ---------------------------------------------------------------------------
// FSST helpers — raw C API wrappers.
// ---------------------------------------------------------------------------

struct FsstInput {
  std::vector<size_t> lengths;
  std::vector<const unsigned char*> ptrs;
  size_t outputBufferSize{0};
  size_t maxStringLength{0};
};

FsstInput makeFsstInput(const StringDataset& ds) {
  NIMBLE_CHECK(!ds.views.empty(), "Benchmark dataset must not be empty.");
  FsstInput input;
  input.lengths.reserve(ds.views.size());
  input.ptrs.reserve(ds.views.size());
  for (const auto value : ds.views) {
    input.lengths.push_back(value.size());
    input.ptrs.push_back(reinterpret_cast<const unsigned char*>(value.data()));
    input.maxStringLength = std::max(input.maxStringLength, value.size());
  }
  input.outputBufferSize = 7 + 2 * ds.totalRawBytes;
  return input;
}

const unsigned char** fsstInputPtrs(const FsstInput& input) {
  return const_cast<const unsigned char**>(input.ptrs.data());
}

struct FsstTrainedState {
  std::vector<size_t> compressedLengths;
  // Offsets into compressedBuf for each compressed string.
  std::vector<size_t> compressedOffsets;
  std::vector<unsigned char> compressedBuf;
  size_t totalCompressedBytes{0};

  unsigned char symbolTableBuf[FSST_MAXHEADER]{};
  unsigned int symbolTableSize{0};
  size_t maxStringLength{0};
};

FsstTrainedState fsstTrain(const FsstInput& input) {
  FsstTrainedState state;
  const auto n = input.lengths.size();
  state.maxStringLength = input.maxStringLength;

  auto* encoder =
      nimble_fsst_create(n, input.lengths.data(), fsstInputPtrs(input), 0);
  NIMBLE_CHECK_NOT_NULL(encoder, "FSST encoder creation failed.");
  state.symbolTableSize = nimble_fsst_export(encoder, state.symbolTableBuf);

  state.compressedBuf.resize(input.outputBufferSize);
  state.compressedLengths.resize(n);
  std::vector<unsigned char*> rawPtrs(n);

  const auto compressedCount = nimble_fsst_compress(
      encoder,
      n,
      input.lengths.data(),
      fsstInputPtrs(input),
      input.outputBufferSize,
      state.compressedBuf.data(),
      state.compressedLengths.data(),
      rawPtrs.data());
  NIMBLE_CHECK_EQ(
      compressedCount, n, "FSST compression did not compress all strings.");

  nimble_fsst_destroy(encoder);

  // Convert raw pointers to stable offsets.
  state.compressedOffsets.resize(n);
  state.totalCompressedBytes = 0;
  for (size_t i = 0; i < n; ++i) {
    state.compressedOffsets[i] = rawPtrs[i] - state.compressedBuf.data();
    state.totalCompressedBytes += state.compressedLengths[i];
  }

  return state;
}

void fsstEncodeBenchmark(unsigned int iters, const FsstInput& input) {
  std::vector<unsigned char> outBuf;
  std::vector<size_t> outLens;
  std::vector<unsigned char*> outPtrs;
  BENCHMARK_SUSPEND {
    outBuf.resize(input.outputBufferSize);
    outLens.resize(input.lengths.size());
    outPtrs.resize(input.lengths.size());
  }

  while (iters--) {
    auto* enc = nimble_fsst_create(
        input.lengths.size(), input.lengths.data(), fsstInputPtrs(input), 0);
    NIMBLE_CHECK_NOT_NULL(enc, "FSST encoder creation failed.");
    const auto compressedCount = nimble_fsst_compress(
        enc,
        input.lengths.size(),
        input.lengths.data(),
        fsstInputPtrs(input),
        input.outputBufferSize,
        outBuf.data(),
        outLens.data(),
        outPtrs.data());
    NIMBLE_CHECK_EQ(
        compressedCount,
        input.lengths.size(),
        "FSST compression did not compress all strings.");
    nimble_fsst_destroy(enc);
  }
}

void fsstDecodeBenchmark(unsigned int iters, const FsstTrainedState& state) {
  fsst_decoder_t decoder{};
  nimble_fsst_import(
      &decoder, const_cast<unsigned char*>(state.symbolTableBuf));
  std::vector<unsigned char> outBuf(state.maxStringLength);
  NIMBLE_CHECK_EQ(
      state.compressedLengths.size(),
      state.compressedOffsets.size(),
      "FSST benchmark compressed length/offset count mismatch.");
  const auto n = state.compressedLengths.size();
  const auto* lengths = state.compressedLengths.data();
  const auto* offsets = state.compressedOffsets.data();
  const auto* compressed = state.compressedBuf.data();

  while (iters--) {
    size_t totalOut = 0;
    for (size_t i = 0; i < n; ++i) {
      totalOut += nimble_fsst_decompress(
          &decoder,
          lengths[i],
          compressed + offsets[i],
          outBuf.size(),
          outBuf.data());
    }
    folly::doNotOptimizeAway(totalOut);
  }
}

// ---------------------------------------------------------------------------
// Zstd/Zstrong helpers — use Nimble's Compression API on the concatenated blob.
// ---------------------------------------------------------------------------

class BenchmarkCompressionPolicy : public CompressionPolicy {
 public:
  explicit BenchmarkCompressionPolicy(CompressionType type, int level = 3)
      : type_{type}, level_{level} {}

  CompressionConfig config() const override {
    CompressionConfig info{.compressionType = type_};
    if (type_ == CompressionType::Zstd) {
      info.parameters.zstd.compressionLevel = level_;
    } else if (type_ == CompressionType::MetaInternal) {
      info.parameters.metaInternal.compressionLevel = level_;
      info.parameters.metaInternal.decompressionLevel = 2;
    }
    info.minCompressionSize = 0;
    return info;
  }

  bool shouldAccept(CompressionType, uint64_t, uint64_t) const override {
    return true;
  }

 private:
  CompressionType type_;
  int level_;
};

// Pre-compressed state for decode benchmarks.
struct BlockCompressedState {
  CompressionType type{CompressionType::Uncompressed};
  std::string compressedData;
  size_t rawSize{0};
};

BlockCompressedState blockCompress(
    facebook::velox::memory::MemoryPool& pool,
    const StringDataset& ds,
    CompressionType type,
    int level = 3) {
  BenchmarkCompressionPolicy policy{type, level};
  auto result = Compression::compress(
      pool, ds.blob, DataType::String, /*bitWidth=*/0, policy);
  NIMBLE_CHECK(
      result.buffer.has_value(),
      "Benchmark block compression unexpectedly returned raw data.");
  NIMBLE_CHECK_EQ(result.compressionType, type, "Unexpected compression type.");
  return {
      .type = result.compressionType,
      .compressedData =
          std::string{result.buffer->data(), result.buffer->size()},
      .rawSize = ds.blob.size(),
  };
}

struct BenchmarkState {
  std::shared_ptr<facebook::velox::memory::MemoryPool> pool{
      facebook::velox::memory::memoryManager()->addLeafPool("benchmark")};

  StringDataset urlData{makeUrls(kNumStrings)};
  StringDataset logData{makeLogMessages(kNumStrings)};
  StringDataset uuidData{makeUuids(kNumStrings)};

  FsstInput urlFsstInput{makeFsstInput(urlData)};
  FsstInput logFsstInput{makeFsstInput(logData)};
  FsstInput uuidFsstInput{makeFsstInput(uuidData)};

  FsstTrainedState urlFsst{fsstTrain(urlFsstInput)};
  FsstTrainedState logFsst{fsstTrain(logFsstInput)};
  FsstTrainedState uuidFsst{fsstTrain(uuidFsstInput)};

  BlockCompressedState urlZstd{
      blockCompress(*pool, urlData, CompressionType::Zstd, 3)};
  BlockCompressedState logZstd{
      blockCompress(*pool, logData, CompressionType::Zstd, 3)};
  BlockCompressedState uuidZstd{
      blockCompress(*pool, uuidData, CompressionType::Zstd, 3)};
  BlockCompressedState urlZstrong{
      blockCompress(*pool, urlData, CompressionType::MetaInternal, 4)};
  BlockCompressedState logZstrong{
      blockCompress(*pool, logData, CompressionType::MetaInternal, 4)};
  BlockCompressedState uuidZstrong{
      blockCompress(*pool, uuidData, CompressionType::MetaInternal, 4)};
};

const BenchmarkState& benchmarkState() {
  static const auto* state = new BenchmarkState();
  return *state;
}

void blockEncodeBenchmark(
    unsigned int iters,
    const BenchmarkState& state,
    const StringDataset& ds,
    CompressionType type,
    int level) {
  BenchmarkCompressionPolicy policy{type, level};
  while (iters--) {
    auto result = Compression::compress(
        *state.pool, ds.blob, DataType::String, 0, policy);
    folly::doNotOptimizeAway(result);
  }
}

void blockDecodeBenchmark(
    unsigned int iters,
    const BenchmarkState& state,
    const BlockCompressedState& compressed) {
  while (iters--) {
    auto result = Compression::uncompress(
        *state.pool,
        compressed.type,
        DataType::String,
        compressed.compressedData,
        /*decompressCounter=*/nullptr);
    folly::doNotOptimizeAway(result);
  }
}

// ---------------------------------------------------------------------------
// Print compression ratio summary (runs once, not timed).
// ---------------------------------------------------------------------------

void printCompressionRatios() {
  const auto& state = benchmarkState();
  auto ratio = [](size_t raw, size_t compressed) {
    return static_cast<double>(raw) / static_cast<double>(compressed);
  };

  fmt::print("\n=== Compression Ratios (higher is better) ===\n");
  fmt::print(
      "{:<16} {:>10} {:>10} {:>10} {:>10}\n",
      "Dataset",
      "Raw (KB)",
      "FSST",
      "Zstd",
      "Zstrong");
  fmt::print("{:-<60}\n", "");

  auto printRow = [&](const char* name,
                      const StringDataset& ds,
                      const FsstTrainedState& fsst,
                      const BlockCompressedState& zstd,
                      const BlockCompressedState& zstrong) {
    const auto fsstTotal = fsst.totalCompressedBytes + fsst.symbolTableSize;
    fmt::print(
        "{:<16} {:>10.1f} {:>10.2f} {:>10.2f} {:>10.2f}\n",
        name,
        ds.totalRawBytes / 1024.0,
        ratio(ds.totalRawBytes, fsstTotal),
        ratio(ds.totalRawBytes, zstd.compressedData.size()),
        ratio(ds.totalRawBytes, zstrong.compressedData.size()));
  };

  printRow(
      "URLs", state.urlData, state.urlFsst, state.urlZstd, state.urlZstrong);
  printRow(
      "Log Messages",
      state.logData,
      state.logFsst,
      state.logZstd,
      state.logZstrong);
  printRow(
      "UUIDs",
      state.uuidData,
      state.uuidFsst,
      state.uuidZstd,
      state.uuidZstrong);

  fmt::print("\n=== Compressed Sizes (bytes) ===\n");
  fmt::print(
      "{:<16} {:>12} {:>12} {:>12} {:>12}\n",
      "Dataset",
      "Raw",
      "FSST",
      "Zstd",
      "Zstrong");
  fmt::print("{:-<64}\n", "");

  auto printSizeRow = [&](const char* name,
                          const StringDataset& ds,
                          const FsstTrainedState& fsst,
                          const BlockCompressedState& zstd,
                          const BlockCompressedState& zstrong) {
    const auto fsstTotal = fsst.totalCompressedBytes + fsst.symbolTableSize;
    fmt::print(
        "{:<16} {:>12} {:>12} {:>12} {:>12}\n",
        name,
        ds.totalRawBytes,
        fsstTotal,
        zstd.compressedData.size(),
        zstrong.compressedData.size());
  };

  printSizeRow(
      "URLs", state.urlData, state.urlFsst, state.urlZstd, state.urlZstrong);
  printSizeRow(
      "Log Messages",
      state.logData,
      state.logFsst,
      state.logZstd,
      state.logZstrong);
  printSizeRow(
      "UUIDs",
      state.uuidData,
      state.uuidFsst,
      state.uuidZstd,
      state.uuidZstrong);
  fmt::print("\n");
}

} // namespace

// ============================================================================
// FSST Encode — train + compress (full pipeline)
// ============================================================================

BENCHMARK(FSST_Encode_URLs, iters) {
  fsstEncodeBenchmark(iters, benchmarkState().urlFsstInput);
}

BENCHMARK(FSST_Encode_LogMessages, iters) {
  fsstEncodeBenchmark(iters, benchmarkState().logFsstInput);
}

BENCHMARK(FSST_Encode_UUIDs, iters) {
  fsstEncodeBenchmark(iters, benchmarkState().uuidFsstInput);
}

BENCHMARK_DRAW_LINE();

// ============================================================================
// Zstd Encode (level 3)
// ============================================================================

BENCHMARK(Zstd_Encode_URLs, iters) {
  const auto& state = benchmarkState();
  blockEncodeBenchmark(iters, state, state.urlData, CompressionType::Zstd, 3);
}

BENCHMARK(Zstd_Encode_LogMessages, iters) {
  const auto& state = benchmarkState();
  blockEncodeBenchmark(iters, state, state.logData, CompressionType::Zstd, 3);
}

BENCHMARK(Zstd_Encode_UUIDs, iters) {
  const auto& state = benchmarkState();
  blockEncodeBenchmark(iters, state, state.uuidData, CompressionType::Zstd, 3);
}

BENCHMARK_DRAW_LINE();

// ============================================================================
// Zstrong Encode
// ============================================================================

BENCHMARK(Zstrong_Encode_URLs, iters) {
  const auto& state = benchmarkState();
  blockEncodeBenchmark(
      iters, state, state.urlData, CompressionType::MetaInternal, 4);
}

BENCHMARK(Zstrong_Encode_LogMessages, iters) {
  const auto& state = benchmarkState();
  blockEncodeBenchmark(
      iters, state, state.logData, CompressionType::MetaInternal, 4);
}

BENCHMARK(Zstrong_Encode_UUIDs, iters) {
  const auto& state = benchmarkState();
  blockEncodeBenchmark(
      iters, state, state.uuidData, CompressionType::MetaInternal, 4);
}

BENCHMARK_DRAW_LINE();

// ============================================================================
// FSST Decode — decompress all strings using pre-trained symbol table
// ============================================================================

BENCHMARK(FSST_Decode_URLs, iters) {
  fsstDecodeBenchmark(iters, benchmarkState().urlFsst);
}

BENCHMARK(FSST_Decode_LogMessages, iters) {
  fsstDecodeBenchmark(iters, benchmarkState().logFsst);
}

BENCHMARK(FSST_Decode_UUIDs, iters) {
  fsstDecodeBenchmark(iters, benchmarkState().uuidFsst);
}

BENCHMARK_DRAW_LINE();

// ============================================================================
// Zstd Decode
// ============================================================================

BENCHMARK(Zstd_Decode_URLs, iters) {
  const auto& state = benchmarkState();
  blockDecodeBenchmark(iters, state, state.urlZstd);
}

BENCHMARK(Zstd_Decode_LogMessages, iters) {
  const auto& state = benchmarkState();
  blockDecodeBenchmark(iters, state, state.logZstd);
}

BENCHMARK(Zstd_Decode_UUIDs, iters) {
  const auto& state = benchmarkState();
  blockDecodeBenchmark(iters, state, state.uuidZstd);
}

BENCHMARK_DRAW_LINE();

// ============================================================================
// Zstrong Decode
// ============================================================================

BENCHMARK(Zstrong_Decode_URLs, iters) {
  const auto& state = benchmarkState();
  blockDecodeBenchmark(iters, state, state.urlZstrong);
}

BENCHMARK(Zstrong_Decode_LogMessages, iters) {
  const auto& state = benchmarkState();
  blockDecodeBenchmark(iters, state, state.logZstrong);
}

BENCHMARK(Zstrong_Decode_UUIDs, iters) {
  const auto& state = benchmarkState();
  blockDecodeBenchmark(iters, state, state.uuidZstrong);
}

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize({});

  printCompressionRatios();

  folly::runBenchmarks();
  return 0;
}
