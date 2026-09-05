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

// Public-API-only ALP encode/decode profile over a dataset catalog. Reports
// compressed size, CPU time, throughput, dispersion, and peak memory per
// dataset x dtype so a change to ALP can be judged on every axis at once.
//
// ALP competes as the outer encoding via ManualEncodingSelectionPolicy
// (candidates = [ALP, Trivial, FixedBitWidth], nested policy sees the default
// cost-based set). It is not forced: a dataset where ALP's estimate loses to
// Trivial reports Trivial's size, which is itself a useful signal.
//
// Columns:
//
//   * bytes       encoded size
//   * ratio       encoded / (rows * sizeof(T)), lower is better
//   * bits/val    encoded bits per value, the unit the ALP literature uses
//   * w-cpu       encode CPU time us / megavalue, the fastest measurement
//   * w-GB/s      encode throughput, (rows * sizeof(T)) / w-cpu
//   * w-iter      encode spread across --iters within one process, percent
//   * w-trial     encode spread across the --trials processes, percent
//   * w-mem       encode peak bytes, from a leaf pool scoped to one encode
//   * r-*         the same five columns for decode
//   * ok          NimbleCompare::equals element-wise round-trip check
//
// Throughput is on the raw input size, matching the compression literature's
// Thrpt = Orig. Size / [De]Comp. Time.
//
// Reported time is a minimum, because the fastest measurement is the one
// interference perturbed least. A minimum on its own cannot say whether it was
// measured cleanly or got lucky, so two dispersion columns qualify it.
//
// w-iter is the relative standard deviation of the --iters samples taken inside
// a single process. It catches interference that comes and goes during a run:
// another process taking the core, a frequency change, a cold cache on the
// first datasets of the catalog.
//
// w-trial is the relative standard deviation of the per-process minima across
// --trials independent child processes. It catches what w-iter is blind to:
// variance that is fixed for the life of a process and therefore identical in
// every one of its iterations, such as code and heap layout under ASLR, or a
// larger environment block shifting the stack. A row can show w-iter under 1%
// and still move several percent between processes; only w-trial reveals that,
// and it is the column to read before quoting a speedup as fact.
//
// --trials defaults to 1, which measures in this process and leaves w-trial at
// zero. A larger value re-executes this binary that many times with --child,
// each child emitting one tab-separated record per row on stdout.
//
// Wall-clock is measured but not tabulated -- this driver is single-threaded
// and in-memory, so wall tracks CPU. A row where it does not gets a warning
// line, which means the machine was busy and that row should be rerun. Both
// dispersion columns are nonetheless computed from the wall samples, because
// getrusage reports whole microseconds and a measurement wraps a single encode
// or decode. Float decode of a well compressed dataset runs under ten
// microseconds, so its samples round onto two or three distinct integers and
// the spread reads 0.0% however much the runs really differ. Wall time has
// nanosecond resolution, and wherever the CPU clock is not rounding the two
// agree to within hundredths of a percent.
//
// Memory is the pool's own peak, not process RSS, so it is comparable across
// datasets and free of allocator noise. Encode peak covers the Buffer plus all
// nested-encoding scratch, so it includes the encoded output. Decode peak
// covers only the decoder's scratch: the materialize destination is allocated
// from a separate pool so a constant rows * sizeof(T) does not mask it. A
// decode peak of zero is a real result, not a broken counter -- the encodings
// read straight out of the encoded buffer, and whatever scratch they do use
// comes from the C++ allocator, which no pool sees.

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sys/resource.h>
#include <sys/time.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/common/tests/NimbleCompare.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

DEFINE_uint32(warmup, 3, "Untimed encode/decode iterations before measuring.");
DEFINE_uint32(iters, 10, "Timed encode/decode iterations per dataset.");
DEFINE_uint32(rows, 65'536, "Values per dataset.");
DEFINE_uint32(
    trials,
    1,
    "Independent child processes to measure in. 1 measures in this process "
    "and leaves the cross-process spread at zero.");
DEFINE_bool(
    child,
    false,
    "Internal. Emit one tab-separated record per row on stdout instead of a "
    "table. Set by the parent when --trials > 1.");
DEFINE_string(
    only,
    "",
    "Substring filter on the dataset name; empty runs every dataset.");
DEFINE_string(
    dtype,
    "",
    "Restrict to one of 'double' or 'float'; empty runs both.");

namespace {

using facebook::nimble::Buffer;
using facebook::nimble::EncodingFactory;
using facebook::nimble::EncodingType;
using facebook::nimble::ManualEncodingSelectionPolicy;
using facebook::nimble::NimbleCompare;
using facebook::nimble::Vector;
using facebook::velox::memory::MemoryPool;

// Parent of every pool below. Created through addRootPool() rather than
// addLeafPool() because only that path turns on usage tracking: leaves under
// the manager's default root inherit
// FLAGS_velox_enable_memory_usage_track_in_default_memory_pool, which is off,
// and an untracked pool reports peakBytes() == 0.
std::shared_ptr<MemoryPool>& rootPool() {
  static auto pool =
      facebook::velox::memory::memoryManager()->addRootPool("alp_profile");
  return pool;
}

// Holds the input datasets and the materialize destinations. Kept out of the
// measured pools so their allocations never enter a reported peak.
std::shared_ptr<MemoryPool>& benchPool() {
  static auto pool = rootPool()->addLeafChild("data");
  return pool;
}

// Creates a leaf pool scoped to a single encode or decode. Peak bytes are read
// off it once the operation finishes and the pool is then discarded, so each
// measurement starts from zero.
std::shared_ptr<MemoryPool> makeRunPool(std::string_view purpose) {
  static uint64_t sequence{0};
  return rootPool()->addLeafChild(fmt::format("{}_{}", purpose, sequence++));
}

uint64_t cpuMicros() {
  struct rusage ru{};
  ::getrusage(RUSAGE_SELF, &ru);
  auto toMicros = [](const struct timeval& tv) -> uint64_t {
    return static_cast<uint64_t>(tv.tv_sec) * 1'000'000ULL +
        static_cast<uint64_t>(tv.tv_usec);
  };
  return toMicros(ru.ru_utime) + toMicros(ru.ru_stime);
}

struct Timer {
  std::chrono::steady_clock::time_point wallStart;
  uint64_t cpuStart{0};

  void start() {
    wallStart = std::chrono::steady_clock::now();
    cpuStart = cpuMicros();
  }

  std::pair<double, double> stop() const {
    const auto wallEnd = std::chrono::steady_clock::now();
    const uint64_t cpuEnd = cpuMicros();
    return {
        std::chrono::duration<double, std::micro>(wallEnd - wallStart).count(),
        static_cast<double>(cpuEnd - cpuStart)};
  }
};

// -----------------------------------------------------------------------------
// Datasets. Same catalog the selection and E2E benchmarks use.
// -----------------------------------------------------------------------------

constexpr uint64_t kSeed = 0x51ED0FF1CEULL;

template <typename T>
Vector<T> makeEmpty(uint32_t n) {
  Vector<T> v{benchPool().get()};
  v.reserve(n);
  return v;
}

template <typename T>
Vector<T> makeIntegers(uint32_t n) {
  auto v = makeEmpty<T>(n);
  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<int32_t> dist(-1'000'000, 1'000'000);
  for (uint32_t i = 0; i < n; ++i) {
    v.push_back(static_cast<T>(dist(rng)));
  }
  return v;
}

template <typename T>
Vector<T> makeTwoDecimalUniform(uint32_t n) {
  auto v = makeEmpty<T>(n);
  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<int32_t> dist(0, 999'999);
  for (uint32_t i = 0; i < n; ++i) {
    v.push_back(static_cast<T>(dist(rng)) / static_cast<T>(100));
  }
  return v;
}

template <typename T>
Vector<T> makeSensorLike(uint32_t n, double outlierRate = 0.02) {
  auto v = makeEmpty<T>(n);
  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<int32_t> clean(0, 99'999);
  std::uniform_real_distribution<double> pOut(0.0, 1.0);
  std::uniform_int_distribution<int64_t> outlier(
      5'000'000'000LL, 9'999'999'999LL);
  for (uint32_t i = 0; i < n; ++i) {
    if (pOut(rng) < outlierRate) {
      v.push_back(static_cast<T>(outlier(rng)) / static_cast<T>(1'000'000));
    } else {
      v.push_back(static_cast<T>(clean(rng)) / static_cast<T>(1'000));
    }
  }
  return v;
}

template <typename T>
Vector<T> makePrices(uint32_t n) {
  auto v = makeEmpty<T>(n);
  std::mt19937_64 rng(kSeed);
  std::normal_distribution<double> dist(500.0, 120.0);
  for (uint32_t i = 0; i < n; ++i) {
    double d = dist(rng);
    if (d < 0) {
      d = -d;
    }
    int64_t cents = static_cast<int64_t>(std::llround(d * 100.0));
    v.push_back(static_cast<T>(cents) / static_cast<T>(100));
  }
  return v;
}

template <typename T>
Vector<T> makeTimestampsMs(uint32_t n) {
  auto v = makeEmpty<T>(n);
  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<int64_t> secs(1'700'000'000LL, 1'800'000'000LL);
  std::uniform_int_distribution<int32_t> millis(0, 999);
  for (uint32_t i = 0; i < n; ++i) {
    int64_t ms = secs(rng) * 1'000LL + millis(rng);
    v.push_back(static_cast<T>(ms) / static_cast<T>(1'000));
  }
  return v;
}

template <typename T>
Vector<T> makeLowCardinality(uint32_t n) {
  auto v = makeEmpty<T>(n);
  const std::vector<double> palette = {
      0.1234, 1.5000, 2.7182, 3.1415, 42.0000, 100.9999, 999.0001, 12.3456};
  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<uint32_t> pick(0, palette.size() - 1);
  for (uint32_t i = 0; i < n; ++i) {
    v.push_back(static_cast<T>(palette[pick(rng)]));
  }
  return v;
}

template <typename T>
Vector<T> makeMixedPrecisionHeavy(uint32_t n) {
  return makeSensorLike<T>(n, /*outlierRate=*/0.30);
}

template <typename T>
Vector<T> makeChaotic(uint32_t n) {
  auto v = makeEmpty<T>(n);
  std::mt19937_64 rng(kSeed);
  std::uniform_real_distribution<double> dist(-1e6, 1e6);
  for (uint32_t i = 0; i < n; ++i) {
    v.push_back(static_cast<T>(dist(rng)));
  }
  return v;
}

// -----------------------------------------------------------------------------
// Encoding selection
// -----------------------------------------------------------------------------

// Puts ALP on the candidate list at cost parity with Trivial and
// FixedBitWidth. ALP still has to win on its own estimate.
template <typename T>
std::unique_ptr<ManualEncodingSelectionPolicy<T>> makeAlpCandidatePolicy() {
  std::vector<std::pair<EncodingType, float>> factors{
      {EncodingType::ALP, 1.0},
      {EncodingType::Trivial, 1.0},
      {EncodingType::FixedBitWidth, 1.0},
  };
  return std::make_unique<ManualEncodingSelectionPolicy<T>>(
      std::move(factors),
      /*compressionOptions=*/std::nullopt,
      /*identifier=*/std::nullopt);
}

// One numeric series. Timing series hold microseconds, one sample per measured
// iteration; cross-process series hold one sample per trial.
struct Series {
  std::vector<double> samples;

  void add(double sample) {
    samples.push_back(sample);
  }

  double min() const {
    if (samples.empty()) {
      return 0.0;
    }
    return *std::min_element(samples.begin(), samples.end());
  }

  double mean() const {
    if (samples.empty()) {
      return 0.0;
    }
    return std::accumulate(samples.begin(), samples.end(), 0.0) /
        static_cast<double>(samples.size());
  }

  // Relative standard deviation as a percentage of the mean. Uses the sample
  // standard deviation (N-1), so a single sample reports zero spread.
  double relativeStdDevPercent() const {
    if (samples.size() < 2) {
      return 0.0;
    }
    const double avg = mean();
    if (avg == 0.0) {
      return 0.0;
    }
    double sumSquares{0.0};
    for (const double sample : samples) {
      const double delta = sample - avg;
      sumSquares += delta * delta;
    }
    const double variance =
        sumSquares / static_cast<double>(samples.size() - 1);
    return std::sqrt(variance) / avg * 100.0;
  }
};

// What one process measured for one dataset x dtype.
struct RunStats {
  uint64_t encodedBytes{0};
  Series writeWall;
  Series writeCpu;
  Series readWall;
  Series readCpu;
  int64_t writePeakBytes{0};
  int64_t readPeakBytes{0};
  bool correct{true};
};

// One process's contribution to one table row, flattened to the few numbers a
// child sends to the parent. The timing fields are that process's own minimum
// and within-process spread.
struct TrialRecord {
  std::string dataset;
  std::string dtype;
  uint64_t encodedBytes{0};
  double writeCpuMin{0.0};
  double writeIterRsd{0.0};
  double writeWallMin{0.0};
  double readCpuMin{0.0};
  double readIterRsd{0.0};
  double readWallMin{0.0};
  int64_t writePeakBytes{0};
  int64_t readPeakBytes{0};
  bool correct{true};
};

// Flattens one process's stats for one row into the record it reports.
//
// Both spreads come from the wall series rather than the CPU series, for the
// reason given on RowSummary: the CPU clock has whole-microsecond resolution
// and rounds the spread of a short operation to zero.
TrialRecord toRecord(
    const std::string& dataset,
    const std::string& dtype,
    const RunStats& s) {
  TrialRecord r;
  r.dataset = dataset;
  r.dtype = dtype;
  r.encodedBytes = s.encodedBytes;
  r.writeCpuMin = s.writeCpu.min();
  r.writeIterRsd = s.writeWall.relativeStdDevPercent();
  r.writeWallMin = s.writeWall.min();
  r.readCpuMin = s.readCpu.min();
  r.readIterRsd = s.readWall.relativeStdDevPercent();
  r.readWallMin = s.readWall.min();
  r.writePeakBytes = s.writePeakBytes;
  r.readPeakBytes = s.readPeakBytes;
  r.correct = s.correct;
  return r;
}

// Everything the parent knows about one table row once every trial has
// reported. The series carry one entry per trial, so their relative standard
// deviation is the cross-process spread.
//
// The reported time comes from the CPU clock, but the cross-process spread is
// computed from the wall series. getrusage reports whole microseconds and a
// measurement wraps a single encode or decode, so a row whose operation takes
// only a few microseconds -- float decode of a well compressed dataset is under
// ten -- has its samples rounded onto two or three distinct integers. The
// resulting spread reads 0.0% no matter how much the processes actually differ.
// Wall time has nanosecond resolution and, wherever the CPU clock is not
// rounding, the two agree to within hundredths of a percent.
struct RowSummary {
  std::string dataset;
  std::string dtype;
  uint64_t encodedBytes{0};
  Series writeCpuAcrossTrials;
  Series readCpuAcrossTrials;
  Series writeWallAcrossTrials;
  Series readWallAcrossTrials;
  // Widest within-process spread any trial saw, so a single noisy trial is not
  // averaged away.
  double writeIterRsd{0.0};
  double readIterRsd{0.0};
  double writeWallMin{0.0};
  double readWallMin{0.0};
  int64_t writePeakBytes{0};
  int64_t readPeakBytes{0};
  bool correct{true};

  void absorb(const TrialRecord& r) {
    dataset = r.dataset;
    dtype = r.dtype;
    encodedBytes = r.encodedBytes;
    writeCpuAcrossTrials.add(r.writeCpuMin);
    readCpuAcrossTrials.add(r.readCpuMin);
    writeWallAcrossTrials.add(r.writeWallMin);
    readWallAcrossTrials.add(r.readWallMin);
    writeIterRsd = std::max(writeIterRsd, r.writeIterRsd);
    readIterRsd = std::max(readIterRsd, r.readIterRsd);
    writeWallMin = writeWallMin == 0.0 ? r.writeWallMin
                                       : std::min(writeWallMin, r.writeWallMin);
    readWallMin = readWallMin == 0.0 ? r.readWallMin
                                     : std::min(readWallMin, r.readWallMin);
    writePeakBytes = std::max(writePeakBytes, r.writePeakBytes);
    readPeakBytes = std::max(readPeakBytes, r.readPeakBytes);
    correct = correct && r.correct;
  }
};

template <typename T>
RunStats runOne(const Vector<T>& values) {
  const facebook::nimble::Encoding::Options options{};
  const std::span<const T> logical{values.data(), values.size()};

  for (uint32_t i = 0; i < FLAGS_warmup; ++i) {
    auto pool = makeRunPool("warmup");
    Buffer buf{*pool};
    const auto encoded = EncodingFactory::encode<T>(
        makeAlpCandidatePolicy<T>(), logical, buf, options);
    auto encoding = EncodingFactory(options).create(
        *pool, encoded, [](uint32_t) { return nullptr; });
    Vector<T> out{pool.get(), values.size()};
    encoding->materialize(values.size(), out.data());
  }

  RunStats agg;

  // Destination for materialize, reused across iterations and allocated
  // outside the measured pools.
  Vector<T> out{benchPool().get(), values.size()};

  for (uint32_t iter = 0; iter < FLAGS_iters; ++iter) {
    auto writePool = makeRunPool("encode");
    Buffer buf{*writePool};

    Timer wt;
    wt.start();
    const auto encoded = EncodingFactory::encode<T>(
        makeAlpCandidatePolicy<T>(), logical, buf, options);
    const auto [wallW, cpuW] = wt.stop();
    agg.writeWall.add(wallW);
    agg.writeCpu.add(cpuW);
    agg.writePeakBytes = std::max(agg.writePeakBytes, writePool->peakBytes());
    if (iter == 0) {
      agg.encodedBytes = encoded.size();
    }

    auto readPool = makeRunPool("decode");
    Timer rt;
    rt.start();
    auto encoding = EncodingFactory(options).create(
        *readPool, encoded, [](uint32_t) { return nullptr; });
    encoding->materialize(values.size(), out.data());
    const auto [wallR, cpuR] = rt.stop();
    agg.readWall.add(wallR);
    agg.readCpu.add(cpuR);
    agg.readPeakBytes = std::max(agg.readPeakBytes, readPool->peakBytes());

    if (iter == 0) {
      for (uint32_t j = 0; j < values.size(); ++j) {
        if (!NimbleCompare<T>::equals(out[j], values[j])) {
          agg.correct = false;
          break;
        }
      }
    }
  }

  return agg;
}

// -----------------------------------------------------------------------------
// Output
// -----------------------------------------------------------------------------

std::string formatKiB(int64_t bytes) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1)
      << (static_cast<double>(bytes) / 1024.0) << "K";
  return oss.str();
}

void printHeader() {
  {
    std::ostringstream oss;
    oss << "| " << std::left << std::setw(28) << "dataset" << " | "
        << std::setw(6) << "dtype" << " | " << std::right << std::setw(10)
        << "bytes" << " | " << std::setw(6) << "ratio" << " | " << std::setw(8)
        << "bits/val" << " | " << std::setw(9) << "w-cpu" << " | "
        << std::setw(7) << "w-GB/s" << " | " << std::setw(6) << "w-iter"
        << " | " << std::setw(7) << "w-trial" << " | " << std::setw(9)
        << "w-mem" << " | " << std::setw(9) << "r-cpu" << " | " << std::setw(7)
        << "r-GB/s" << " | " << std::setw(6) << "r-iter" << " | "
        << std::setw(7) << "r-trial" << " | " << std::setw(9) << "r-mem"
        << " | " << std::setw(3) << "ok" << " |";
    LOG(INFO) << oss.str();
  }
  {
    std::ostringstream oss;
    oss << "|" << std::string(30, '-') << "|" << std::string(8, '-') << "|"
        << std::string(12, '-') << "|" << std::string(8, '-') << "|"
        << std::string(10, '-') << "|" << std::string(11, '-') << "|"
        << std::string(9, '-') << "|" << std::string(8, '-') << "|"
        << std::string(9, '-') << "|" << std::string(11, '-') << "|"
        << std::string(11, '-') << "|" << std::string(9, '-') << "|"
        << std::string(8, '-') << "|" << std::string(9, '-') << "|"
        << std::string(11, '-') << "|" << std::string(5, '-') << "|";
    LOG(INFO) << oss.str();
  }
}

// Converts a per-operation time into throughput over the raw input, matching
// the compression literature's Thrpt = Orig. Size / [De]Comp. Time.
double gigabytesPerSecond(double rawBytes, double micros) {
  if (micros <= 0.0) {
    return 0.0;
  }
  return rawBytes / micros / 1'000.0;
}

void printRow(const RowSummary& row, uint32_t rows) {
  const double megaVals = static_cast<double>(rows) / 1'000'000.0;
  const double bytesPerValue = row.dtype == "float" ? 4.0 : 8.0;
  const double rawBytes = static_cast<double>(rows) * bytesPerValue;
  const double ratio = rawBytes == 0
      ? 0.0
      : static_cast<double>(row.encodedBytes) / rawBytes * 100.0;
  const double bitsPerValue = rows == 0
      ? 0.0
      : static_cast<double>(row.encodedBytes) * 8.0 / static_cast<double>(rows);
  // Across trials the representative time is again the minimum: the fastest
  // process is the one whose layout and environment cost it least.
  const double writeCpuMin = row.writeCpuAcrossTrials.min();
  const double readCpuMin = row.readCpuAcrossTrials.min();
  std::ostringstream oss;
  oss << "| " << std::left << std::setw(28) << row.dataset << " | "
      << std::setw(6) << row.dtype << " | " << std::right << std::setw(10)
      << row.encodedBytes << " | " << std::fixed << std::setprecision(2)
      << std::setw(5) << ratio << "% | " << std::setprecision(2) << std::setw(8)
      << bitsPerValue << " | " << std::setprecision(1) << std::setw(7)
      << (writeCpuMin / megaVals) << "us | " << std::setprecision(2)
      << std::setw(7) << gigabytesPerSecond(rawBytes, writeCpuMin) << " | "
      << std::setprecision(1) << std::setw(5) << row.writeIterRsd << "% | "
      << std::setw(6) << row.writeWallAcrossTrials.relativeStdDevPercent()
      << "% | " << std::setw(9) << formatKiB(row.writePeakBytes) << " | "
      << std::setw(7) << (readCpuMin / megaVals) << "us | "
      << std::setprecision(2) << std::setw(7)
      << gigabytesPerSecond(rawBytes, readCpuMin) << " | "
      << std::setprecision(1) << std::setw(5) << row.readIterRsd << "% | "
      << std::setw(6) << row.readWallAcrossTrials.relativeStdDevPercent()
      << "% | " << std::setw(9) << formatKiB(row.readPeakBytes) << " | "
      << std::setw(3) << (row.correct ? "OK" : "BAD") << " |";
  LOG(INFO) << oss.str();

  // Wall-clock is not tabulated because this driver is single-threaded and in
  // memory. If it diverges from CPU time the machine was contended and the row
  // is not trustworthy, so say so rather than silently reporting it.
  auto flagDivergence = [&row](
                            const char* phase, double cpuMin, double wallMin) {
    if (cpuMin > 0.0 && wallMin > cpuMin * 1.25) {
      LOG(WARNING) << "  " << row.dataset << " (" << row.dtype << "): " << phase
                   << " wall exceeds cpu by "
                   << static_cast<int>((wallMin / cpuMin - 1.0) * 100.0)
                   << "%, machine likely contended; rerun this row";
    }
  };
  flagDivergence("encode", writeCpuMin, row.writeWallMin);
  flagDivergence("decode", readCpuMin, row.readWallMin);
}

template <typename T>
struct Dataset {
  std::string name;
  std::function<Vector<T>(uint32_t)> maker;
};

template <typename T>
void collectAll(std::vector<TrialRecord>& out) {
  const uint32_t n = FLAGS_rows;
  const std::string dtype = std::is_same_v<T, float> ? "float" : "double";
  std::vector<Dataset<T>> datasets{
      {"integers", &makeIntegers<T>},
      {"two-decimal uniform", &makeTwoDecimalUniform<T>},
      {"prices (2dp gaussian)", &makePrices<T>},
      {"timestamps (ms)", &makeTimestampsMs<T>},
      {"low cardinality (8)", &makeLowCardinality<T>},
      {"sensor + 2% outliers", [](uint32_t k) { return makeSensorLike<T>(k); }},
      {"mixed precision 30%", &makeMixedPrecisionHeavy<T>},
      {"chaotic (all exceptions)", &makeChaotic<T>},
  };
  for (const auto& d : datasets) {
    if (!FLAGS_only.empty() && d.name.find(FLAGS_only) == std::string::npos) {
      continue;
    }
    auto data = d.maker(n);
    out.push_back(toRecord(d.name, dtype, runOne<T>(data)));
  }
}

// Measures every selected dataset once, in this process.
std::vector<TrialRecord> collectTrial() {
  std::vector<TrialRecord> records;
  if (FLAGS_dtype != "float") {
    collectAll<double>(records);
  }
  if (FLAGS_dtype != "double") {
    collectAll<float>(records);
  }
  return records;
}

// -----------------------------------------------------------------------------
// Parent/child protocol
//
// A child writes one tab-separated record per row on stdout. The dataset name
// comes last because it is the only field that can contain spaces; every other
// field is a number, so the parent can split on tabs without quoting rules.
// -----------------------------------------------------------------------------

constexpr char kRecordPrefix[] = "TRIAL\t";

void writeRecord(const TrialRecord& r) {
  std::printf(
      "%s%s\t%llu\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%lld\t%lld\t%d\t%s\n",
      kRecordPrefix,
      r.dtype.c_str(),
      static_cast<unsigned long long>(r.encodedBytes),
      r.writeCpuMin,
      r.writeIterRsd,
      r.writeWallMin,
      r.readCpuMin,
      r.readIterRsd,
      r.readWallMin,
      static_cast<long long>(r.writePeakBytes),
      static_cast<long long>(r.readPeakBytes),
      r.correct ? 1 : 0,
      r.dataset.c_str());
}

// Parses one line a child produced. Returns false for any line that is not a
// record, which is how glog output on the shared stderr/stdout is skipped.
bool parseRecord(const std::string& line, TrialRecord& out) {
  const std::string prefix{kRecordPrefix};
  if (line.compare(0, prefix.size(), prefix) != 0) {
    return false;
  }
  std::vector<std::string> fields;
  size_t start = prefix.size();
  // The last field is the dataset name and may itself be anything, so stop
  // splitting once the fixed-width fields have been taken.
  constexpr int kFixedFields = 11;
  for (int i = 0; i < kFixedFields; ++i) {
    const size_t tab = line.find('\t', start);
    if (tab == std::string::npos) {
      return false;
    }
    fields.push_back(line.substr(start, tab - start));
    start = tab + 1;
  }
  out.dtype = fields[0];
  out.encodedBytes = std::stoull(fields[1]);
  out.writeCpuMin = std::stod(fields[2]);
  out.writeIterRsd = std::stod(fields[3]);
  out.writeWallMin = std::stod(fields[4]);
  out.readCpuMin = std::stod(fields[5]);
  out.readIterRsd = std::stod(fields[6]);
  out.readWallMin = std::stod(fields[7]);
  out.writePeakBytes = std::stoll(fields[8]);
  out.readPeakBytes = std::stoll(fields[9]);
  out.correct = fields[10] == "1";
  out.dataset = line.substr(start);
  return true;
}

// Re-executes this binary once with --child and returns the records it emitted.
// Child stderr is left attached so its glog warnings reach the terminal.
std::vector<TrialRecord> runChild(const std::string& self) {
  std::ostringstream cmd;
  cmd << "'" << self << "'"
      << " --child"
      << " --rows=" << FLAGS_rows << " --warmup=" << FLAGS_warmup
      << " --iters=" << FLAGS_iters;
  if (!FLAGS_only.empty()) {
    cmd << " --only='" << FLAGS_only << "'";
  }
  if (!FLAGS_dtype.empty()) {
    cmd << " --dtype='" << FLAGS_dtype << "'";
  }

  std::vector<TrialRecord> records;
  FILE* pipe = ::popen(cmd.str().c_str(), "r");
  if (pipe == nullptr) {
    LOG(ERROR) << "Failed to launch trial child: " << cmd.str();
    return records;
  }
  std::array<char, 4096> buffer{};
  while (std::fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    std::string line{buffer.data()};
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
      line.pop_back();
    }
    TrialRecord record;
    if (parseRecord(line, record)) {
      records.push_back(record);
    }
  }
  const int status = ::pclose(pipe);
  if (status != 0) {
    LOG(ERROR) << "Trial child exited with status " << status;
  }
  return records;
}

} // namespace

int main(int argc, char** argv) {
  const std::string self = argv[0];
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize({});

  if (FLAGS_child) {
    for (const auto& record : collectTrial()) {
      writeRecord(record);
    }
    std::fflush(stdout);
    return 0;
  }

  const uint32_t trials = std::max(FLAGS_trials, 1u);

  LOG(INFO) << "=== ALP encode/decode profile ===";
  LOG(INFO) << "rows / dataset: " << FLAGS_rows
            << ", warmup iters: " << FLAGS_warmup
            << ", measure iters: " << FLAGS_iters << ", trials: " << trials;
  LOG(INFO) << "outer policy: ManualEncodingSelectionPolicy "
               "[ALP, Trivial, FixedBitWidth]";
  LOG(INFO) << "times: cpu microseconds per megavalue, the fastest measurement";
  LOG(INFO) << "GB/s: (rows * sizeof(T)) / fastest cpu time";
  LOG(INFO) << "iter: spread across the measure iters of one process; a large "
               "value means that process hit interference";
  LOG(INFO) << "trial: spread across the trial processes; a large value with a "
               "small iter means per-process layout, not noise, and repeated "
               "iterations cannot average it away";
  LOG(INFO) << "mem: peak bytes of a leaf pool scoped to one encode or decode "
               "(max over all measurements)";
  LOG(INFO) << "ratio: encoded / (rows * sizeof(T)), lower is better";
  LOG(INFO) << "bits/val: encoded bits per value, lower is better";
  if (trials < 2) {
    LOG(INFO) << "note: --trials=1 measures in this process, so the trial "
                 "columns read 0.0%; pass --trials=5 or more to populate them";
  }

  // Insertion-ordered so the table keeps the catalog's order regardless of the
  // order trials report in.
  std::vector<std::string> rowOrder;
  std::map<std::string, RowSummary> rows;
  auto absorb = [&](const TrialRecord& record) {
    const std::string key = record.dtype + "\t" + record.dataset;
    if (rows.find(key) == rows.end()) {
      rowOrder.push_back(key);
    }
    rows[key].absorb(record);
  };

  if (trials == 1) {
    for (const auto& record : collectTrial()) {
      absorb(record);
    }
  } else {
    for (uint32_t trial = 0; trial < trials; ++trial) {
      const auto records = runChild(self);
      if (records.empty()) {
        LOG(ERROR) << "Trial " << trial << " produced no records";
        continue;
      }
      for (const auto& record : records) {
        absorb(record);
      }
    }
  }

  printHeader();
  for (const auto& key : rowOrder) {
    printRow(rows[key], FLAGS_rows);
  }
  return 0;
}
