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
// It walks fixed-size rows in an allocation run, collecting up to kBatchSize
// row pointers before handing the batch to a consumer -- the batched
// producer-consumer pattern of the real caller. The prefetch fetches rows a
// fixed distance ahead so they are in cache by the time the flag check and the
// consumer reach them.
//
// The rows are narrow (dominant TPC-DS build/probe layouts are tens of bytes)
// and the run is far larger than the LLC, so the row stream misses to DRAM;
// hardware prefetch does not fully hide that latency in this access pattern,
// and the software prefetch restores the run-ahead.
//
// The consumer is load-bearing: it does one pseudo-random probe per collected
// row into an 8 MiB table, modeling the hash-probe or aggregation lookup the
// caller drives. With a trivial (key-only) consumer the row stream is a
// sequential walk the hardware prefetcher already covers, and the measured win
// then grows monotonically with distance -- an artifact that over-credits far
// prefetch.
//
// The distance is the compile-time constant LIST_ROWS_PREFETCH_DISTANCE
// (default 2048); each binary measures the baseline against that one distance.
// Sweep by rebuilding one binary per distance and comparing medians --
// per-distance process isolation is what keeps the comparison trustworthy:
//   for d in 1024 2048 4096 8192; do
//     c++ ... -DLIST_ROWS_PREFETCH_DISTANCE=$d ...; done

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace {

// Rows are collected in batches of this many, matching listRows()' caller.
constexpr int32_t kBatchSize = 1'024;

// Offset of the 8-byte key the consumer reads, relative to the logical row
// start; every modeled row is wide enough that it never overlaps the free flag.
constexpr int32_t kKeyOffset = 8;

// Normalized-key prefix size while normalized keys are live, matching
// RowContainer's originalNormalizedKeySize_ for the common case.
constexpr int32_t kNormalizedKeySize = 8;

// Prefetch distance under test, in bytes. Compile-time so this binary holds one
// distance only; rebuild per distance to sweep (see the header). Default is the
// production value.
#ifndef LIST_ROWS_PREFETCH_DISTANCE
#define LIST_ROWS_PREFETCH_DISTANCE 2048
#endif
constexpr int32_t kPrefetchDistance = LIST_ROWS_PREFETCH_DISTANCE;

// The consumer probes pseudo-random 8-byte slots in an 8 MiB table -- above
// private cache capacity, below the row scan's DRAM footprint. Power of two so
// key & mask selects a slot.
constexpr int64_t kProbeSlots = (8LL << 20) / sizeof(uint64_t);
constexpr uint64_t kProbeMask = static_cast<uint64_t>(kProbeSlots) - 1;
std::vector<uint64_t> gProbeTable;

// Normalized-key state of a scan, mirroring how listRows() offsets each row
// pointer and steps the row size down when the normalized keys run out.
enum class NormalizedKeyMode {
  kOff, // No normalized keys: rows are dataRowSize throughout.
  kAll, // Every row carries a normalized key.
  kTransition, // Normalized keys run out mid-scan; the row size steps down.
};

// One contiguous run of fixed-size rows, modeling an allocation range that
// listRows() scans. Each row has a clear free flag at its logical start and an
// 8-byte key at kKeyOffset. While normalized keys are live the physical stride
// is dataRowSize + kNormalizedKeySize and the logical row starts that far in.
struct FakeRowContainer {
  FakeRowContainer(
      int32_t dataRowSize,
      int64_t targetBytes,
      NormalizedKeyMode mode = NormalizedKeyMode::kOff)
      : dataRowSize_(dataRowSize),
        normalizedKeySize_(
            mode == NormalizedKeyMode::kOff ? 0 : kNormalizedKeySize) {
    const int32_t wideSize = dataRowSize_ + normalizedKeySize_;
    // The allocation is sized for the wide stride. In kTransition the narrower
    // post-transition stride therefore yields more logical rows than numRows,
    // so absolute times are not comparable across NK modes.
    const int64_t numRows =
        std::max<int64_t>(kBatchSize, targetBytes / wideSize);
    usedBytes_ = numRows * wideSize;
    data_.resize(static_cast<size_t>(usedBytes_));

    if (mode == NormalizedKeyMode::kTransition) {
      normalizedKeysLeft_ = numRows / 2;
    } else if (mode == NormalizedKeyMode::kAll) {
      normalizedKeysLeft_ = numRows;
    } else {
      normalizedKeysLeft_ = 0;
    }

    // Fill every 8 bytes with non-zero data so each page is really backed (not
    // a shared zero page) and the scan truly hits DRAM.
    std::mt19937_64 rng(42);
    for (size_t i = 0; i + sizeof(uint64_t) <= data_.size();
         i += sizeof(uint64_t)) {
      const uint64_t value = rng();
      std::memcpy(data_.data() + i, &value, sizeof(value));
    }
    // Replay the exact stride walk to clear the free flag on every visited row,
    // so all rows are live and the boundaries match the scan.
    int64_t row = 0;
    int32_t stride = wideSize;
    int64_t left = normalizedKeysLeft_;
    while (row + stride <= usedBytes_) {
      const int64_t logical = row + (left > 0 ? normalizedKeySize_ : 0);
      data_[logical] &= ~static_cast<char>(1); // Free flag clear: row is live.
      row += stride;
      if (left > 0 && --left == 0) {
        stride = dataRowSize_;
      }
    }
  }

  const char* data() const {
    return data_.data();
  }
  int64_t usedBytes() const {
    return usedBytes_;
  }
  // Initial (widest) stride the scan starts with.
  int32_t rowSize() const {
    return dataRowSize_ + normalizedKeySize_;
  }
  int32_t normalizedKeySize() const {
    return normalizedKeySize_;
  }
  int64_t normalizedKeysLeft() const {
    return normalizedKeysLeft_;
  }

 private:
  int32_t dataRowSize_;
  int32_t normalizedKeySize_;
  int64_t usedBytes_;
  int64_t normalizedKeysLeft_;
  std::vector<char> data_;
};

// The RowContainer::listRows() producer loop, then one probe per collected row.
// kPrefetchBytes == 0 is the no-prefetch baseline; non-zero reproduces the x86
// hint at that distance. As in production the hint is unconditional and its
// address is formed with integer arithmetic, avoiding out-of-bounds pointer
// arithmetic even when the target lies past the range end.
template <int32_t kPrefetchBytes>
uint64_t scanAndConsumeOnce(
    const char* data,
    int64_t usedBytes,
    int32_t rowSize,
    int32_t normalizedKeySize,
    int64_t normalizedKeysLeft,
    const uint64_t* probeTable) {
  const char* rows[kBatchSize];
  uint64_t checksum = 0;
  int64_t row = 0;
  int32_t curRowSize = rowSize;
  while (row + curRowSize <= usedBytes) {
    int32_t count = 0;
    while (row + curRowSize <= usedBytes) {
      if constexpr (kPrefetchBytes != 0) {
        __builtin_prefetch(
            reinterpret_cast<const char*>(
                reinterpret_cast<uintptr_t>(data) + row + kPrefetchBytes));
      }
      const char* r =
          data + row + (normalizedKeysLeft > 0 ? normalizedKeySize : 0);
      rows[count++] = r;
      row += curRowSize;
      if (--normalizedKeysLeft == 0) {
        curRowSize -= normalizedKeySize;
      }
      if (r[0] & 1) { // Free flag: skip freed rows, as listRows() does.
        --count;
        continue;
      }
      if (count == kBatchSize) {
        break;
      }
    }
    // Each collected row's key selects one pseudo-random probe-table entry,
    // modeling a downstream hash probe or aggregation lookup.
    for (int32_t i = 0; i < count; ++i) {
      uint64_t key;
      std::memcpy(&key, rows[i] + kKeyOffset, sizeof(key));
      checksum += probeTable[key & kProbeMask];
    }
    if (count == 0) {
      break;
    }
  }
  return checksum;
}

template <int32_t kPrefetchBytes>
void runScanAndConsume(const FakeRowContainer& container, int32_t iters) {
  uint64_t checksum = 0;
  for (int32_t i = 0; i < iters; ++i) {
    // The scan reads only const globals; without this the compiler hoists the
    // whole call out of the loop as loop-invariant and the baseline reports an
    // impossible bandwidth. makeUnpredictable applies equally to every arm.
    const char* data = container.data();
    int64_t usedBytes = container.usedBytes();
    int32_t rowSize = container.rowSize();
    int32_t normalizedKeySize = container.normalizedKeySize();
    int64_t normalizedKeysLeft = container.normalizedKeysLeft();
    const uint64_t* probeTable = gProbeTable.data();
    folly::makeUnpredictable(data);
    folly::makeUnpredictable(usedBytes);
    folly::makeUnpredictable(rowSize);
    folly::makeUnpredictable(normalizedKeySize);
    folly::makeUnpredictable(normalizedKeysLeft);
    folly::makeUnpredictable(probeTable);
    checksum += scanAndConsumeOnce<kPrefetchBytes>(
        data,
        usedBytes,
        rowSize,
        normalizedKeySize,
        normalizedKeysLeft,
        probeTable);
  }
  folly::doNotOptimizeAway(checksum);
}

// Larger-than-LLC buffers exercise the DRAM regime the prefetch targets; a
// small buffer exercises the cache-resident regime the prefetch must not hurt.
std::unique_ptr<FakeRowContainer> g29Dram;
std::unique_ptr<FakeRowContainer> g61Dram;
std::unique_ptr<FakeRowContainer> g128Dram;
std::unique_ptr<FakeRowContainer> g29Resident;
std::unique_ptr<FakeRowContainer> g29DramNkAll;
std::unique_ptr<FakeRowContainer> g29DramNkTransition;

// Passes per timed iteration. DRAM buffers are scanned once; the tiny resident
// buffer is scanned many times so the timed work dwarfs per-call overhead.
constexpr int32_t kDramPasses = 1;
constexpr int32_t kResidentPasses = 512;

// Part A: baseline vs the compile-time distance across row widths, DRAM regime.
BENCHMARK(listRowsConsume_29B_dram_off) {
  runScanAndConsume<0>(*g29Dram, kDramPasses);
}
BENCHMARK_RELATIVE(listRowsConsume_29B_dram_pf) {
  runScanAndConsume<kPrefetchDistance>(*g29Dram, kDramPasses);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(listRowsConsume_61B_dram_off) {
  runScanAndConsume<0>(*g61Dram, kDramPasses);
}
BENCHMARK_RELATIVE(listRowsConsume_61B_dram_pf) {
  runScanAndConsume<kPrefetchDistance>(*g61Dram, kDramPasses);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(listRowsConsume_128B_dram_off) {
  runScanAndConsume<0>(*g128Dram, kDramPasses);
}
BENCHMARK_RELATIVE(listRowsConsume_128B_dram_pf) {
  runScanAndConsume<kPrefetchDistance>(*g128Dram, kDramPasses);
}

BENCHMARK_DRAW_LINE();

// Part B: small, cache-resident row working set (consumer unchanged). Bounds
// the prefetch's cost when the rows are already cached, versus the DRAM win.
BENCHMARK(listRowsConsume_29B_resident_off) {
  runScanAndConsume<0>(*g29Resident, kResidentPasses);
}
BENCHMARK_RELATIVE(listRowsConsume_29B_resident_pf) {
  runScanAndConsume<kPrefetchDistance>(*g29Resident, kResidentPasses);
}

BENCHMARK_DRAW_LINE();

// Part C: normalized-key states. Compare off vs pf within a state only; times
// are not comparable across states.
BENCHMARK(listRowsConsume_29B_nkoff_off) {
  runScanAndConsume<0>(*g29Dram, kDramPasses);
}
BENCHMARK_RELATIVE(listRowsConsume_29B_nkoff_pf) {
  runScanAndConsume<kPrefetchDistance>(*g29Dram, kDramPasses);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(listRowsConsume_29B_nkall_off) {
  runScanAndConsume<0>(*g29DramNkAll, kDramPasses);
}
BENCHMARK_RELATIVE(listRowsConsume_29B_nkall_pf) {
  runScanAndConsume<kPrefetchDistance>(*g29DramNkAll, kDramPasses);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(listRowsConsume_29B_nktransition_off) {
  runScanAndConsume<0>(*g29DramNkTransition, kDramPasses);
}
BENCHMARK_RELATIVE(listRowsConsume_29B_nktransition_pf) {
  runScanAndConsume<kPrefetchDistance>(*g29DramNkTransition, kDramPasses);
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);

  // 256 MiB keeps the row working set well beyond private-cache capacity and
  // exercises the DRAM-sensitive regime on the target platform.
  constexpr int64_t kDramBytes = 256LL << 20;
  // 256 KiB models a cache-resident row working set.
  constexpr int64_t kResidentBytes = 256LL << 10;

  // Fill the probe table with non-zero data so a probed slot is a real load.
  gProbeTable.resize(static_cast<size_t>(kProbeSlots));
  std::mt19937_64 rng(1337);
  for (auto& slot : gProbeTable) {
    slot = rng();
  }

  g29Dram = std::make_unique<FakeRowContainer>(29, kDramBytes);
  g61Dram = std::make_unique<FakeRowContainer>(61, kDramBytes);
  g128Dram = std::make_unique<FakeRowContainer>(128, kDramBytes);
  g29Resident = std::make_unique<FakeRowContainer>(29, kResidentBytes);
  g29DramNkAll = std::make_unique<FakeRowContainer>(
      29, kDramBytes, NormalizedKeyMode::kAll);
  g29DramNkTransition = std::make_unique<FakeRowContainer>(
      29, kDramBytes, NormalizedKeyMode::kTransition);

  fprintf(
      stderr, "prefetch distance under test: %d bytes\n", kPrefetchDistance);

  folly::runBenchmarks();
  return 0;
}
