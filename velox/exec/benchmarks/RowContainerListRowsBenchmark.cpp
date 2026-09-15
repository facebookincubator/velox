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

// Benchmarks the software prefetch that RowContainer::listRows() issues on x86.
//
// Drives the real listRows() (not a copy of its loop) and probes an 8 MiB table
// once per row -- a pointer-only consumer would leave a pure sequential walk the
// hardware prefetcher already covers, hiding the DRAM latency the prefetch
// targets. Containers are built before timing.
//
// The distance is a template parameter of listRows() defaulting to the shipped
// kListRowsPrefetchDistanceBytes, so production and the benchmark instantiate the
// same scan. Each binary measures one distance (0 disables the prefetch for the
// baseline); compare medians across per-distance processes:
//
//   c++ ... -DLIST_ROWS_BENCHMARK_DISTANCE_BYTES=0    ...  # baseline
//   c++ ...                                           ...  # shipped default, 2048
//   c++ ... -DLIST_ROWS_BENCHMARK_DISTANCE_BYTES=4096 ...

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/exec/RowContainer.h"

namespace facebook::velox::exec {
namespace {

// Representative batch size for the target workload.
constexpr int32_t kBatchSize = 1'024;

// Defaults to the shipped constant; override at compile time to sweep.
#ifndef LIST_ROWS_BENCHMARK_DISTANCE_BYTES
#define LIST_ROWS_BENCHMARK_DISTANCE_BYTES \
  RowContainer::kListRowsPrefetchDistanceBytes
#endif
constexpr int32_t kBenchmarkDistance = LIST_ROWS_BENCHMARK_DISTANCE_BYTES;

// Pseudo-random probe table, sized above private-cache capacity to model the
// cache pressure of a downstream hash or aggregation lookup. Power of two for
// key & mask.
constexpr int64_t kProbeSlots = (8LL << 20) / sizeof(uint64_t);
constexpr uint64_t kProbeMask = static_cast<uint64_t>(kProbeSlots) - 1;

// Key layouts that land fixedRowSize() exactly on the target width (packed keys
// plus one flag byte holding the adjacent free and probed bits: 28+1, 60+1,
// 127+1); VELOX_CHECK_EQ below guards it.
std::vector<TypePtr> keyTypesForWidth(int32_t width) {
  switch (width) {
    case 29:
      return {BIGINT(), BIGINT(), BIGINT(), INTEGER()};
    case 61:
      return {
          BIGINT(),
          BIGINT(),
          BIGINT(),
          BIGINT(),
          BIGINT(),
          BIGINT(),
          BIGINT(),
          INTEGER()};
    case 128: {
      std::vector<TypePtr> types(15, BIGINT());
      for (int32_t i = 0; i < 7; ++i) {
        types.push_back(TINYINT());
      }
      return types;
    }
    default:
      VELOX_FAIL("Unsupported benchmark row width: {}", width);
  }
}

std::unique_ptr<RowContainer> makeContainer(
    int32_t rowWidth,
    int64_t targetBytes,
    memory::MemoryPool* pool) {
  auto container = std::make_unique<RowContainer>(
      keyTypesForWidth(rowWidth),
      /*nullableKeys=*/false,
      std::vector<Accumulator>{},
      std::vector<TypePtr>{},
      /*hasNext=*/false,
      /*isJoinBuild=*/false,
      /*hasProbedFlag=*/true,
      /*hasCountFlag=*/false,
      /*hasNormalizedKey=*/false,
      /*useListRowIndex=*/false,
      pool);
  VELOX_CHECK_EQ(container->fixedRowSize(), rowWidth);

  const int64_t numRows = std::max<int64_t>(kBatchSize, targetBytes / rowWidth);
  std::mt19937_64 rng(42);
  for (int64_t i = 0; i < numRows; ++i) {
    char* row = container->newRow();
    // Random BIGINT in the first key indexes the probe table. The flag byte sits
    // past the keys and stays zero, so the probed bit is clear and kNotProbed
    // returns every row.
    const uint64_t key = rng();
    std::memcpy(row, &key, sizeof(key));
  }
  return container;
}

// Scans the container with the real listRows(), probing once per row. Returns a
// checksum so nothing is optimized away.
template <int32_t prefetchDistanceBytes>
uint64_t scanAndProbeOnce(
    const RowContainer& container,
    const std::vector<uint64_t>& probeTable) {
  RowContainerIterator iter;
  std::array<char*, kBatchSize> rows;
  uint64_t checksum = 0;
  while (true) {
    const auto count = container.listRows<
        RowContainer::ProbeType::kNotProbed,
        prefetchDistanceBytes>(
        &iter, kBatchSize, RowContainer::kUnlimited, rows.data());
    if (count == 0) {
      break;
    }
    for (int32_t i = 0; i < count; ++i) {
      uint64_t key;
      std::memcpy(&key, rows[i], sizeof(key));
      checksum += probeTable[key & kProbeMask];
    }
  }
  return checksum;
}

// Runs 'times' scans and returns 'times' for folly to divide by. The
// makeUnpredictable() reload stops the compiler hoisting the scan out of loop.
template <int32_t prefetchDistanceBytes>
unsigned runScanAndProbe(
    const RowContainer& container,
    const std::vector<uint64_t>& probeTable,
    unsigned times) {
  uint64_t checksum = 0;
  for (unsigned i = 0; i < times; ++i) {
    const RowContainer* containerPtr = &container;
    folly::makeUnpredictable(containerPtr);
    checksum += scanAndProbeOnce<prefetchDistanceBytes>(*containerPtr, probeTable);
  }
  folly::doNotOptimizeAway(checksum);
  return times;
}

} // namespace
} // namespace facebook::velox::exec

int main(int argc, char** argv) {
  using namespace facebook::velox;
  using facebook::velox::exec::kBenchmarkDistance;
  using facebook::velox::exec::kProbeSlots;
  using facebook::velox::exec::makeContainer;
  using facebook::velox::exec::runScanAndProbe;

  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  auto pool = memory::memoryManager()->addLeafPool();

  std::vector<uint64_t> probeTable(static_cast<size_t>(kProbeSlots));
  std::mt19937_64 rng(1337);
  for (auto& slot : probeTable) {
    slot = rng();
  }

  // 256 MiB exercises the DRAM-sensitive regime; 256 KiB provides a
  // small-row-working-set case to bound the prefetch cost when row data is
  // cache-friendly. Local to main() so they release before the MemoryManager
  // singleton is torn down.
  constexpr int64_t kLargeBytes = 256LL << 20;
  constexpr int64_t kResidentBytes = 256LL << 10;
  auto c29 = makeContainer(29, kLargeBytes, pool.get());
  auto c61 = makeContainer(61, kLargeBytes, pool.get());
  auto c128 = makeContainer(128, kLargeBytes, pool.get());
  auto c29Resident = makeContainer(29, kResidentBytes, pool.get());

  fprintf(
      stderr,
      "listRows prefetch distance under test: %d bytes\n",
      kBenchmarkDistance);

  folly::addBenchmark(__FILE__, "listRows_29B_dram", [&](unsigned times) {
    return runScanAndProbe<kBenchmarkDistance>(*c29, probeTable, times);
  });
  folly::addBenchmark(__FILE__, "listRows_61B_dram", [&](unsigned times) {
    return runScanAndProbe<kBenchmarkDistance>(*c61, probeTable, times);
  });
  folly::addBenchmark(__FILE__, "listRows_128B_dram", [&](unsigned times) {
    return runScanAndProbe<kBenchmarkDistance>(*c128, probeTable, times);
  });
  folly::addBenchmark(__FILE__, "listRows_29B_resident", [&](unsigned times) {
    return runScanAndProbe<kBenchmarkDistance>(*c29Resident, probeTable, times);
  });

  folly::runBenchmarks();
  return 0;
}
