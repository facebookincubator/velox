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

// Benchmarks for Delta encoding materialize (decode) performance.
//
// Tests materialize throughput with different data distributions:
// - Monotonic: strictly increasing by 1 (best case: 0% restatements)
// - Sawtooth: repeating ramp pattern (periodic restatements)
// - Timestamps: monotonic with small random jitter
// - Random: uniformly random values (worst case: ~50% restatements)
// - Constant: all same value (0% restatements, 0 deltas)
//
// Usage:
//   buck2 run @mode/opt //velox/dwio/nimble/encodings/tests:delta_benchmark

#include "common/init/light.h"
#include "folly/Benchmark.h"
#include "folly/Random.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/DeltaEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

using namespace facebook::nimble;

namespace {

constexpr uint32_t kRowCount = 1'000'000;
constexpr uint32_t kRowCountSmall = 100'000;
constexpr uint32_t kRowCountLarge = 10'000'000;

std::shared_ptr<facebook::velox::memory::MemoryPool> pool;

// --- Data generators ---

// Strictly monotonically increasing: value[i] = i.
// Best case for delta: 0 restatements, all deltas = 1.
template <typename T>
std::vector<T> generateMonotonic(uint32_t rowCount) {
  std::vector<T> data(rowCount);
  for (uint32_t i = 0; i < rowCount; ++i) {
    data[i] = static_cast<T>(i);
  }
  return data;
}

// Sawtooth: repeating ramp [0..period), then reset.
// Produces restatements at every period boundary.
template <typename T>
std::vector<T> generateSawtooth(uint32_t rowCount, uint32_t period) {
  std::vector<T> data(rowCount);
  for (uint32_t i = 0; i < rowCount; ++i) {
    data[i] = static_cast<T>(i % period);
  }
  return data;
}

// Timestamps: monotonically increasing base + small random jitter.
// Realistic workload: mostly deltas, occasional restatement when jitter
// causes a decrease.
template <typename T>
std::vector<T> generateTimestamps(uint32_t rowCount) {
  std::vector<T> data(rowCount);
  T base = 1'000'000;
  for (uint32_t i = 0; i < rowCount; ++i) {
    // Each step increases by ~100, with jitter in [-5, +5].
    base += static_cast<T>(95 + folly::Random::rand32() % 11);
    data[i] = base;
  }
  return data;
}

// Random: uniformly random values. Worst case for delta encoding:
// ~50% of transitions are decreasing, causing restatements.
template <typename T>
std::vector<T> generateRandom(uint32_t rowCount) {
  std::vector<T> data(rowCount);
  for (uint32_t i = 0; i < rowCount; ++i) {
    data[i] = static_cast<T>(folly::Random::rand64() % 1'000'000);
  }
  return data;
}

// Constant: all same value.
// Edge case: 0 deltas, all restatements except first.
// Actually for delta, first is restatement, rest are deltas with delta=0.
template <typename T>
std::vector<T> generateConstant(uint32_t rowCount) {
  return std::vector<T>(rowCount, static_cast<T>(42));
}

// --- Encode helper ---

struct EncodedData {
  std::string_view encoded;
  std::unique_ptr<Buffer> buffer;
};

template <typename T>
EncodedData encodeDelta(const std::vector<T>& data) {
  auto buffer = std::make_unique<Buffer>(*pool);
  ManualEncodingSelectionPolicyFactory factory{
      {{EncodingType::Delta, 1.0}},
      /*compressionOptions=*/std::nullopt};
  auto policy = std::unique_ptr<EncodingSelectionPolicy<T>>(
      static_cast<EncodingSelectionPolicy<T>*>(
          factory.createPolicy(TypeTraits<T>::dataType).release()));
  auto encoded = EncodingFactory::encode<T>(
      std::move(policy), std::span<const T>(data), *buffer);
  return {encoded, std::move(buffer)};
}

// --- Benchmark functions ---

// Full materialize: decode all rows in one call.
template <typename T>
void benchMaterialize(unsigned iters, const EncodedData& data, uint32_t rows) {
  std::vector<T> output(rows);
  for (unsigned i = 0; i < iters; ++i) {
    auto encoding = EncodingFactory().create(*pool, data.encoded, {});
    encoding->materialize(rows, output.data());
    folly::doNotOptimizeAway(output[0]);
  }
}

// Batched materialize: decode in fixed-size batches.
template <typename T>
void benchMaterializeBatched(
    unsigned iters,
    const EncodedData& data,
    uint32_t rows,
    uint32_t batchSize) {
  std::vector<T> output(batchSize);
  for (unsigned i = 0; i < iters; ++i) {
    auto encoding = EncodingFactory().create(*pool, data.encoded, {});
    uint32_t remaining = rows;
    while (remaining > 0) {
      auto count = std::min(remaining, batchSize);
      encoding->materialize(count, output.data());
      remaining -= count;
    }
    folly::doNotOptimizeAway(output[0]);
  }
}

// Skip + materialize: alternating skip/read pattern.
template <typename T>
void benchSkipAndMaterialize(
    unsigned iters,
    const EncodedData& data,
    uint32_t rows,
    uint32_t skipCount,
    uint32_t readCount) {
  std::vector<T> output(readCount);
  for (unsigned i = 0; i < iters; ++i) {
    auto encoding = EncodingFactory().create(*pool, data.encoded, {});
    uint32_t pos = 0;
    while (pos < rows) {
      auto toSkip = std::min(skipCount, rows - pos);
      encoding->skip(toSkip);
      pos += toSkip;
      if (pos >= rows) {
        break;
      }
      auto toRead = std::min(readCount, rows - pos);
      encoding->materialize(toRead, output.data());
      pos += toRead;
    }
    folly::doNotOptimizeAway(output[0]);
  }
}

// --- Pre-encoded benchmark data ---

template <typename T>
struct BenchmarkData {
  EncodedData monotonic;
  EncodedData sawtooth100; // period=100
  EncodedData sawtooth1000; // period=1000
  EncodedData timestamps;
  EncodedData random;
  EncodedData constant;

  void init(uint32_t rows) {
    monotonic = encodeDelta(generateMonotonic<T>(rows));
    sawtooth100 = encodeDelta(generateSawtooth<T>(rows, 100));
    sawtooth1000 = encodeDelta(generateSawtooth<T>(rows, 1000));
    timestamps = encodeDelta(generateTimestamps<T>(rows));
    random = encodeDelta(generateRandom<T>(rows));
    constant = encodeDelta(generateConstant<T>(rows));
  }
};

// 1M rows — primary benchmark size.
std::unique_ptr<BenchmarkData<int32_t>> int32Data;
std::unique_ptr<BenchmarkData<int64_t>> int64Data;

// 100K rows — for batched/skip benchmarks.
std::unique_ptr<BenchmarkData<int64_t>> int64DataSmall;

// 10M rows — for large-scale throughput.
std::unique_ptr<BenchmarkData<int64_t>> int64DataLarge;

} // namespace

// ============================================================================
// int32_t full materialize (1M rows)
// ============================================================================

BENCHMARK(Int32_Materialize_Monotonic, n) {
  benchMaterialize<int32_t>(n, int32Data->monotonic, kRowCount);
}
BENCHMARK_RELATIVE(Int32_Materialize_Sawtooth100, n) {
  benchMaterialize<int32_t>(n, int32Data->sawtooth100, kRowCount);
}
BENCHMARK_RELATIVE(Int32_Materialize_Sawtooth1000, n) {
  benchMaterialize<int32_t>(n, int32Data->sawtooth1000, kRowCount);
}
BENCHMARK_RELATIVE(Int32_Materialize_Timestamps, n) {
  benchMaterialize<int32_t>(n, int32Data->timestamps, kRowCount);
}
BENCHMARK_RELATIVE(Int32_Materialize_Random, n) {
  benchMaterialize<int32_t>(n, int32Data->random, kRowCount);
}
BENCHMARK_RELATIVE(Int32_Materialize_Constant, n) {
  benchMaterialize<int32_t>(n, int32Data->constant, kRowCount);
}

BENCHMARK_DRAW_LINE();

// ============================================================================
// int64_t full materialize (1M rows)
// ============================================================================

BENCHMARK(Int64_Materialize_Monotonic, n) {
  benchMaterialize<int64_t>(n, int64Data->monotonic, kRowCount);
}
BENCHMARK_RELATIVE(Int64_Materialize_Sawtooth100, n) {
  benchMaterialize<int64_t>(n, int64Data->sawtooth100, kRowCount);
}
BENCHMARK_RELATIVE(Int64_Materialize_Sawtooth1000, n) {
  benchMaterialize<int64_t>(n, int64Data->sawtooth1000, kRowCount);
}
BENCHMARK_RELATIVE(Int64_Materialize_Timestamps, n) {
  benchMaterialize<int64_t>(n, int64Data->timestamps, kRowCount);
}
BENCHMARK_RELATIVE(Int64_Materialize_Random, n) {
  benchMaterialize<int64_t>(n, int64Data->random, kRowCount);
}
BENCHMARK_RELATIVE(Int64_Materialize_Constant, n) {
  benchMaterialize<int64_t>(n, int64Data->constant, kRowCount);
}

BENCHMARK_DRAW_LINE();

// ============================================================================
// int64_t batched materialize (100K rows, batch=1024)
// ============================================================================

BENCHMARK(Int64_Batched1024_Monotonic, n) {
  benchMaterializeBatched<int64_t>(
      n, int64DataSmall->monotonic, kRowCountSmall, 1024);
}
BENCHMARK_RELATIVE(Int64_Batched1024_Sawtooth100, n) {
  benchMaterializeBatched<int64_t>(
      n, int64DataSmall->sawtooth100, kRowCountSmall, 1024);
}
BENCHMARK_RELATIVE(Int64_Batched1024_Timestamps, n) {
  benchMaterializeBatched<int64_t>(
      n, int64DataSmall->timestamps, kRowCountSmall, 1024);
}
BENCHMARK_RELATIVE(Int64_Batched1024_Random, n) {
  benchMaterializeBatched<int64_t>(
      n, int64DataSmall->random, kRowCountSmall, 1024);
}
BENCHMARK_RELATIVE(Int64_Batched1024_Constant, n) {
  benchMaterializeBatched<int64_t>(
      n, int64DataSmall->constant, kRowCountSmall, 1024);
}

BENCHMARK_DRAW_LINE();

// ============================================================================
// int64_t skip + materialize (100K rows, skip 512, read 512)
// ============================================================================

BENCHMARK(Int64_SkipRead512_Monotonic, n) {
  benchSkipAndMaterialize<int64_t>(
      n, int64DataSmall->monotonic, kRowCountSmall, 512, 512);
}
BENCHMARK_RELATIVE(Int64_SkipRead512_Sawtooth100, n) {
  benchSkipAndMaterialize<int64_t>(
      n, int64DataSmall->sawtooth100, kRowCountSmall, 512, 512);
}
BENCHMARK_RELATIVE(Int64_SkipRead512_Timestamps, n) {
  benchSkipAndMaterialize<int64_t>(
      n, int64DataSmall->timestamps, kRowCountSmall, 512, 512);
}
BENCHMARK_RELATIVE(Int64_SkipRead512_Random, n) {
  benchSkipAndMaterialize<int64_t>(
      n, int64DataSmall->random, kRowCountSmall, 512, 512);
}
BENCHMARK_RELATIVE(Int64_SkipRead512_Constant, n) {
  benchSkipAndMaterialize<int64_t>(
      n, int64DataSmall->constant, kRowCountSmall, 512, 512);
}

BENCHMARK_DRAW_LINE();

// ============================================================================
// int64_t full materialize (10M rows) — large-scale throughput
// ============================================================================

BENCHMARK(Int64_10M_Materialize_Monotonic, n) {
  benchMaterialize<int64_t>(n, int64DataLarge->monotonic, kRowCountLarge);
}
BENCHMARK_RELATIVE(Int64_10M_Materialize_Sawtooth100, n) {
  benchMaterialize<int64_t>(n, int64DataLarge->sawtooth100, kRowCountLarge);
}
BENCHMARK_RELATIVE(Int64_10M_Materialize_Timestamps, n) {
  benchMaterialize<int64_t>(n, int64DataLarge->timestamps, kRowCountLarge);
}
BENCHMARK_RELATIVE(Int64_10M_Materialize_Random, n) {
  benchMaterialize<int64_t>(n, int64DataLarge->random, kRowCountLarge);
}
BENCHMARK_RELATIVE(Int64_10M_Materialize_Constant, n) {
  benchMaterialize<int64_t>(n, int64DataLarge->constant, kRowCountLarge);
}

// ============================================================================
// Micro-benchmarks: isolate the reconstruction loop from child materialization.
// These benchmark just the merge + prefix-sum step using pre-built arrays,
// to identify whether the bottleneck is child materialization or the loop.
// ============================================================================

namespace {

struct ReconstructionData {
  // Use uint8_t instead of bool to match nimble::Vector<bool> (byte-per-bool).
  std::vector<uint8_t> isRestatements;
  std::vector<int64_t> deltas;
  std::vector<int64_t> restatements;
  uint32_t rowCount{};
};

// Build raw arrays from source data as DeltaEncoding::computeDeltas would.
ReconstructionData buildReconstructionData(const std::vector<int64_t>& values) {
  ReconstructionData rd;
  rd.rowCount = static_cast<uint32_t>(values.size());
  rd.isRestatements.resize(rd.rowCount, 0);
  rd.isRestatements[0] = 1;
  rd.restatements.push_back(values[0]);
  for (uint32_t i = 1; i < rd.rowCount; ++i) {
    if (values[i] >= values[i - 1]) {
      rd.deltas.push_back(values[i] - values[i - 1]);
    } else {
      rd.isRestatements[i] = 1;
      rd.restatements.push_back(values[i]);
    }
  }
  return rd;
}

// Original reconstruction: per-element branch.
void reconstructBranchy(const ReconstructionData& rd, int64_t* output) {
  const uint8_t* isRestData = rd.isRestatements.data();
  const int64_t* restPtr = rd.restatements.data();
  const int64_t* deltaPtr = rd.deltas.data();
  int64_t current = 0;
  for (uint32_t i = 0; i < rd.rowCount; ++i) {
    if (isRestData[i]) {
      current = *restPtr++;
    } else {
      current += *deltaPtr++;
    }
    output[i] = current;
  }
}

// Segment-based reconstruction: process runs between restatements.
void reconstructSegmented(const ReconstructionData& rd, int64_t* output) {
  const uint8_t* isRestData = rd.isRestatements.data();
  const int64_t* restPtr = rd.restatements.data();
  const int64_t* deltaPtr = rd.deltas.data();
  int64_t current = 0;
  uint32_t i = 0;
  while (i < rd.rowCount) {
    if (isRestData[i]) {
      current = *restPtr++;
      output[i] = current;
      ++i;
      uint32_t segEnd = i;
      while (segEnd < rd.rowCount && !isRestData[segEnd]) {
        ++segEnd;
      }
      const uint32_t segLen = segEnd - i;
      for (uint32_t j = 0; j < segLen; ++j) {
        current += deltaPtr[j];
        output[i + j] = current;
      }
      deltaPtr += segLen;
      i = segEnd;
    } else {
      current += *deltaPtr++;
      output[i] = current;
      ++i;
    }
  }
}

// Branchless reconstruction: merge deltas and restatements into a single
// array, then do a conditional prefix-sum with no branches.
void reconstructBranchless(const ReconstructionData& rd, int64_t* output) {
  const uint8_t* isRestData = rd.isRestatements.data();
  const int64_t* restPtr = rd.restatements.data();
  const int64_t* deltaPtr = rd.deltas.data();

  // First, merge deltas and restatements into the output array.
  // For restatement positions: store the restatement value.
  // For delta positions: store the delta.
  for (uint32_t i = 0; i < rd.rowCount; ++i) {
    // Branchless select: isRest ? restatement : delta.
    const bool isRest = isRestData[i];
    output[i] = isRest ? *restPtr : *deltaPtr;
    restPtr += isRest;
    deltaPtr += !isRest;
  }

  // Second pass: convert to absolute values.
  // At restatement positions, reset the running sum.
  // At delta positions, add to the running sum.
  int64_t current = 0;
  for (uint32_t i = 0; i < rd.rowCount; ++i) {
    // Branchless: isRest ? output[i] : current + output[i]
    const bool isRest = isRestData[i];
    // If restatement, current = output[i] (the restatement value).
    // If delta, current = current + output[i] (accumulate delta).
    current = isRest ? output[i] : (current + output[i]);
    output[i] = current;
  }
}

// Single-pass branchless: combine merge and prefix-sum.
void reconstructBranchlessSinglePass(
    const ReconstructionData& rd,
    int64_t* output) {
  const uint8_t* isRestData = rd.isRestatements.data();
  const int64_t* restPtr = rd.restatements.data();
  const int64_t* deltaPtr = rd.deltas.data();
  int64_t current = 0;

  for (uint32_t i = 0; i < rd.rowCount; ++i) {
    const bool isRest = isRestData[i];
    // Branchless: pick restatement or (current + delta).
    const int64_t restVal = *restPtr;
    const int64_t deltaVal = current + *deltaPtr;
    current = isRest ? restVal : deltaVal;
    restPtr += isRest;
    deltaPtr += !isRest;
    output[i] = current;
  }
}

std::unique_ptr<ReconstructionData> reconMonotonic;
std::unique_ptr<ReconstructionData> reconSawtooth100;
std::unique_ptr<ReconstructionData> reconTimestamps;
std::unique_ptr<ReconstructionData> reconRandom;

} // namespace

BENCHMARK_DRAW_LINE();

// --- Branchy reconstruction (original approach) ---
BENCHMARK(ReconBranchy_Monotonic, n) {
  std::vector<int64_t> output(reconMonotonic->rowCount);
  for (unsigned i = 0; i < n; ++i) {
    reconstructBranchy(*reconMonotonic, output.data());
    folly::doNotOptimizeAway(output[0]);
  }
}
BENCHMARK_RELATIVE(ReconBranchy_Sawtooth100, n) {
  std::vector<int64_t> output(reconSawtooth100->rowCount);
  for (unsigned i = 0; i < n; ++i) {
    reconstructBranchy(*reconSawtooth100, output.data());
    folly::doNotOptimizeAway(output[0]);
  }
}
BENCHMARK_RELATIVE(ReconBranchy_Timestamps, n) {
  std::vector<int64_t> output(reconTimestamps->rowCount);
  for (unsigned i = 0; i < n; ++i) {
    reconstructBranchy(*reconTimestamps, output.data());
    folly::doNotOptimizeAway(output[0]);
  }
}
BENCHMARK_RELATIVE(ReconBranchy_Random, n) {
  std::vector<int64_t> output(reconRandom->rowCount);
  for (unsigned i = 0; i < n; ++i) {
    reconstructBranchy(*reconRandom, output.data());
    folly::doNotOptimizeAway(output[0]);
  }
}

BENCHMARK_DRAW_LINE();

// --- Segment-based reconstruction (optimized approach) ---
BENCHMARK(ReconSegmented_Monotonic, n) {
  std::vector<int64_t> output(reconMonotonic->rowCount);
  for (unsigned i = 0; i < n; ++i) {
    reconstructSegmented(*reconMonotonic, output.data());
    folly::doNotOptimizeAway(output[0]);
  }
}
BENCHMARK_RELATIVE(ReconSegmented_Sawtooth100, n) {
  std::vector<int64_t> output(reconSawtooth100->rowCount);
  for (unsigned i = 0; i < n; ++i) {
    reconstructSegmented(*reconSawtooth100, output.data());
    folly::doNotOptimizeAway(output[0]);
  }
}
BENCHMARK_RELATIVE(ReconSegmented_Timestamps, n) {
  std::vector<int64_t> output(reconTimestamps->rowCount);
  for (unsigned i = 0; i < n; ++i) {
    reconstructSegmented(*reconTimestamps, output.data());
    folly::doNotOptimizeAway(output[0]);
  }
}
BENCHMARK_RELATIVE(ReconSegmented_Random, n) {
  std::vector<int64_t> output(reconRandom->rowCount);
  for (unsigned i = 0; i < n; ++i) {
    reconstructSegmented(*reconRandom, output.data());
    folly::doNotOptimizeAway(output[0]);
  }
}

BENCHMARK_DRAW_LINE();

// --- Branchless two-pass reconstruction ---
BENCHMARK(ReconBranchless_Monotonic, n) {
  std::vector<int64_t> output(reconMonotonic->rowCount);
  for (unsigned i = 0; i < n; ++i) {
    reconstructBranchless(*reconMonotonic, output.data());
    folly::doNotOptimizeAway(output[0]);
  }
}
BENCHMARK_RELATIVE(ReconBranchless_Sawtooth100, n) {
  std::vector<int64_t> output(reconSawtooth100->rowCount);
  for (unsigned i = 0; i < n; ++i) {
    reconstructBranchless(*reconSawtooth100, output.data());
    folly::doNotOptimizeAway(output[0]);
  }
}
BENCHMARK_RELATIVE(ReconBranchless_Timestamps, n) {
  std::vector<int64_t> output(reconTimestamps->rowCount);
  for (unsigned i = 0; i < n; ++i) {
    reconstructBranchless(*reconTimestamps, output.data());
    folly::doNotOptimizeAway(output[0]);
  }
}
BENCHMARK_RELATIVE(ReconBranchless_Random, n) {
  std::vector<int64_t> output(reconRandom->rowCount);
  for (unsigned i = 0; i < n; ++i) {
    reconstructBranchless(*reconRandom, output.data());
    folly::doNotOptimizeAway(output[0]);
  }
}

BENCHMARK_DRAW_LINE();

// --- Branchless single-pass reconstruction ---
BENCHMARK(ReconBranchless1P_Monotonic, n) {
  std::vector<int64_t> output(reconMonotonic->rowCount);
  for (unsigned i = 0; i < n; ++i) {
    reconstructBranchlessSinglePass(*reconMonotonic, output.data());
    folly::doNotOptimizeAway(output[0]);
  }
}
BENCHMARK_RELATIVE(ReconBranchless1P_Sawtooth100, n) {
  std::vector<int64_t> output(reconSawtooth100->rowCount);
  for (unsigned i = 0; i < n; ++i) {
    reconstructBranchlessSinglePass(*reconSawtooth100, output.data());
    folly::doNotOptimizeAway(output[0]);
  }
}
BENCHMARK_RELATIVE(ReconBranchless1P_Timestamps, n) {
  std::vector<int64_t> output(reconTimestamps->rowCount);
  for (unsigned i = 0; i < n; ++i) {
    reconstructBranchlessSinglePass(*reconTimestamps, output.data());
    folly::doNotOptimizeAway(output[0]);
  }
}
BENCHMARK_RELATIVE(ReconBranchless1P_Random, n) {
  std::vector<int64_t> output(reconRandom->rowCount);
  for (unsigned i = 0; i < n; ++i) {
    reconstructBranchlessSinglePass(*reconRandom, output.data());
    folly::doNotOptimizeAway(output[0]);
  }
}

int main(int argc, char** argv) {
  facebook::init::InitFacebookLight init(&argc, &argv);
  facebook::velox::memory::MemoryManager::initialize({});
  pool =
      facebook::velox::memory::memoryManager()->addLeafPool("delta_benchmark");

  int32Data = std::make_unique<BenchmarkData<int32_t>>();
  int32Data->init(kRowCount);

  int64Data = std::make_unique<BenchmarkData<int64_t>>();
  int64Data->init(kRowCount);

  int64DataSmall = std::make_unique<BenchmarkData<int64_t>>();
  int64DataSmall->init(kRowCountSmall);

  int64DataLarge = std::make_unique<BenchmarkData<int64_t>>();
  int64DataLarge->init(kRowCountLarge);

  // Build reconstruction micro-benchmark data (1M rows).
  reconMonotonic = std::make_unique<ReconstructionData>(
      buildReconstructionData(generateMonotonic<int64_t>(kRowCount)));
  reconSawtooth100 = std::make_unique<ReconstructionData>(
      buildReconstructionData(generateSawtooth<int64_t>(kRowCount, 100)));
  reconTimestamps = std::make_unique<ReconstructionData>(
      buildReconstructionData(generateTimestamps<int64_t>(kRowCount)));
  reconRandom = std::make_unique<ReconstructionData>(
      buildReconstructionData(generateRandom<int64_t>(kRowCount)));

  folly::runBenchmarks();

  reconRandom.reset();
  reconTimestamps.reset();
  reconSawtooth100.reset();
  reconMonotonic.reset();
  int64DataLarge.reset();
  int64DataSmall.reset();
  int64Data.reset();
  int32Data.reset();
  pool.reset();
  return 0;
}
