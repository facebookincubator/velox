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

// Asks whether ALP can replace a general-purpose compressor on floating-point
// columns. Two arms encode the same dataset through the public
// EncodingFactory API and are compared on size first, speed second:
//
//   * alp    ManualEncodingSelectionPolicy [ALP, Trivial, FixedBitWidth] with
//            no compression options at all, so nothing in the tree -- outer
//            encoding or nested stream -- is handed to a compressor.
//   * rival  ManualEncodingSelectionPolicy [Trivial] with compression options
//            naming OpenZL, so the raw values go straight to the compressor.
//
// Size is the admission gate, not a tiebreaker. A speedup only counts if the
// alp arm writes no more bytes than the rival arm, which is what the size
// column reports. Speed is the payoff: ALP is a per-value transform with no
// entropy stage, so it should encode and decode faster.
//
// Columns:
//
//   * a-type    encoding the alp arm's policy actually selected. ALP has to
//               win on its own size estimate; a row reading Trivial or
//               FixedBitWidth is a row where it lost, and its timings then
//               describe that encoding, not ALP.
//   * a-bytes   encoded size of the alp arm
//   * z-comp    compression the rival arm's stream ended up carrying. Reads
//               Uncompressed when the compressor's output was not smaller
//               than its input and the policy rejected it.
//   * z-bytes   encoded size of the rival arm
//   * size      a-bytes / z-bytes. Under 100% means ALP compressed better and
//               the gate is passed.
//   * a-enc     alp arm encode throughput, (rows * sizeof(T)) / cpu time
//   * z-enc     rival arm encode throughput
//   * enc x     a-enc / z-enc, above 1.00x means ALP encodes faster
//   * a-dec     alp arm decode throughput
//   * z-dec     rival arm decode throughput
//   * dec x     a-dec / z-dec
//   * iter      widest spread across the --iters of one process, over all four
//               measured series, percent
//   * trial     widest spread across the --trials processes, same four series
//   * ok        both arms passed an element-wise NimbleCompare round trip
//
// Throughput is on the raw input size, matching the compression literature's
// Thrpt = Orig. Size / [De]Comp. Time. Reported time is a minimum, because the
// fastest measurement is the one interference perturbed least.
//
// The two arms alternate inside one iteration of one process: alp encode, alp
// decode, rival encode, rival decode. That matters for the ratio columns. A
// speedup assembled from two separately captured tables carries the drift
// between the captures, which on this workload has been seen to reach low
// double-digit percent even on byte-identical code, and neither dispersion
// column can see it. Interleaving puts both arms under the same drift, so it
// divides out of the ratio.
//
// One asymmetry to keep in mind when reading the double rows. OpenZL picks its
// compression graph from the stream's data type, and only Float reaches the
// numeric pipeline; Double falls through to plain zstd. So the double half of
// the table is really ALP against Trivial + zstd, and only the float half puts
// ALP against OpenZL's floating-point graph. Pass --rival=zstd to make that
// explicit on both halves.
//
// The rival arm's compression options override two defaults that would
// otherwise let small or marginally compressible inputs slip through
// uncompressed and quietly turn the comparison into ALP against raw Trivial:
// the accept ratio is raised to 1.0 so any size reduction is kept, and the
// per-compressor minimum input size is dropped to zero.
//
// Peak memory is not tabulated. The velox pool sees the Buffer and the nested
// encoding scratch but not what a compressor allocates through the C++
// allocator, so the two arms would not be measured on the same footing.

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
DEFINE_string(
    rival,
    "openzl",
    "Compressor the Trivial arm runs against ALP: 'openzl', 'zstd' or 'lz4'.");

namespace {

using facebook::nimble::Buffer;
using facebook::nimble::CompressionOptions;
using facebook::nimble::CompressionType;
using facebook::nimble::EncodingFactory;
using facebook::nimble::EncodingType;
using facebook::nimble::ManualEncodingSelectionPolicy;
using facebook::nimble::NimbleCompare;
using facebook::nimble::Vector;
using facebook::velox::memory::MemoryPool;

// Parent of every pool below. Created through addRootPool() rather than
// addLeafPool() because only that path turns on usage tracking.
std::shared_ptr<MemoryPool>& rootPool() {
  static auto pool =
      facebook::velox::memory::memoryManager()->addRootPool("alp_vs_openzl");
  return pool;
}

// Holds the input datasets and the materialize destinations, so their
// allocations never land in a measured pool.
std::shared_ptr<MemoryPool>& benchPool() {
  static auto pool = rootPool()->addLeafChild("data");
  return pool;
}

// Creates a leaf pool scoped to a single encode or decode, so each operation
// allocates from a pool that starts empty and is discarded right after.
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
// Datasets. Same catalog ALPProfileBenchmark uses, so the two tables line up
// row for row.
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

CompressionType rivalCompressionType() {
  if (FLAGS_rival == "openzl") {
    return CompressionType::OpenZL;
  }
  if (FLAGS_rival == "zstd") {
    return CompressionType::Zstd;
  }
  if (FLAGS_rival == "lz4") {
    return CompressionType::Lz4;
  }
  LOG(FATAL) << "Unknown --rival, expected openzl, zstd or lz4: "
             << FLAGS_rival;
}

// Puts ALP on the candidate list at cost parity with Trivial and
// FixedBitWidth, and passes no compression options, which leaves every stream
// in the tree uncompressed.
template <typename T>
std::unique_ptr<ManualEncodingSelectionPolicy<T>> makeAlpPolicy() {
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

// Leaves Trivial as the only candidate so the raw values reach the compressor
// untransformed, which is what a deployment without ALP does today.
template <typename T>
std::unique_ptr<ManualEncodingSelectionPolicy<T>>
makeCompressedTrivialPolicy() {
  CompressionOptions compressionOptions;
  compressionOptions.compressionType = rivalCompressionType();
  // Keep any size reduction at all, and never skip a stream for being short,
  // so a rejected compression means the compressor genuinely could not shrink
  // the data rather than that a threshold got in the way.
  compressionOptions.compressionAcceptRatio = 1.0f;
  compressionOptions.compressionAcceptRatioOverrides.clear();
  compressionOptions.openzlMinCompressionSize = 0;
  compressionOptions.zstdMinCompressionSize = 0;
  compressionOptions.lz4MinCompressionSize = 0;

  std::vector<std::pair<EncodingType, float>> factors{
      {EncodingType::Trivial, 1.0},
  };
  return std::make_unique<ManualEncodingSelectionPolicy<T>>(
      std::move(factors),
      std::move(compressionOptions),
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

// What one process measured for one arm of one dataset x dtype.
struct ArmStats {
  uint64_t encodedBytes{0};
  // Encoding the policy selected, and the compression the stream ended up
  // carrying. Both are read back off the encoded bytes rather than assumed.
  std::string encodingName{"?"};
  std::string compressionName{"?"};
  Series encodeWall;
  Series encodeCpu;
  Series decodeWall;
  Series decodeCpu;
  bool correct{true};
};

// -----------------------------------------------------------------------------
// Measurement
// -----------------------------------------------------------------------------

// Runs one encode and one decode of the whole dataset and appends the two
// timings to the arm. On the capture iteration it also records the encoded
// size, what the policy selected, and the round-trip result.
template <typename T, typename PolicyFactory>
void measureOnce(
    const Vector<T>& values,
    Vector<T>& out,
    const facebook::nimble::Encoding::Options& options,
    PolicyFactory makePolicy,
    bool capture,
    ArmStats& arm) {
  const std::span<const T> logical{values.data(), values.size()};

  auto writePool = makeRunPool("encode");
  Buffer buffer{*writePool};
  Timer encodeTimer;
  encodeTimer.start();
  const auto encoded =
      EncodingFactory::encode<T>(makePolicy(), logical, buffer, options);
  const auto [encodeWall, encodeCpu] = encodeTimer.stop();
  arm.encodeWall.add(encodeWall);
  arm.encodeCpu.add(encodeCpu);

  auto readPool = makeRunPool("decode");
  Timer decodeTimer;
  decodeTimer.start();
  auto encoding = EncodingFactory(options).create(
      *readPool, encoded, [](uint32_t) { return nullptr; });
  encoding->materialize(values.size(), out.data());
  const auto [decodeWall, decodeCpu] = decodeTimer.stop();
  arm.decodeWall.add(decodeWall);
  arm.decodeCpu.add(decodeCpu);

  if (!capture) {
    return;
  }
  arm.encodedBytes = encoded.size();
  arm.encodingName = toString(encoding->encodingType());
  // A Trivial stream stores its compression type in the first byte after the
  // common prefix. No other encoding here has that layout, so nothing is
  // claimed about them.
  if (encoding->encodingType() == EncodingType::Trivial) {
    arm.compressionName =
        toString(static_cast<CompressionType>(encoded[encoding->dataOffset()]));
  } else {
    arm.compressionName = "-";
  }
  for (uint32_t i = 0; i < values.size(); ++i) {
    if (!NimbleCompare<T>::equals(out[i], values[i])) {
      arm.correct = false;
      break;
    }
  }
}

// Both arms of one dataset x dtype, measured by one process.
struct PairStats {
  ArmStats alp;
  ArmStats rival;
};

template <typename T>
PairStats runOne(const Vector<T>& values) {
  const facebook::nimble::Encoding::Options options{};
  Vector<T> out{benchPool().get(), values.size()};

  PairStats warm;
  for (uint32_t i = 0; i < FLAGS_warmup; ++i) {
    measureOnce<T>(
        values, out, options, &makeAlpPolicy<T>, /*capture=*/false, warm.alp);
    measureOnce<T>(
        values,
        out,
        options,
        &makeCompressedTrivialPolicy<T>,
        /*capture=*/false,
        warm.rival);
  }

  PairStats stats;
  // The arms alternate inside the iteration so that whatever the machine is
  // doing at this moment lands on both of them, and cancels in the ratio.
  for (uint32_t iter = 0; iter < FLAGS_iters; ++iter) {
    const bool capture = iter == 0;
    measureOnce<T>(values, out, options, &makeAlpPolicy<T>, capture, stats.alp);
    measureOnce<T>(
        values,
        out,
        options,
        &makeCompressedTrivialPolicy<T>,
        capture,
        stats.rival);
  }
  return stats;
}

// -----------------------------------------------------------------------------
// Aggregation
// -----------------------------------------------------------------------------

// One process's contribution to one table row. The timing fields are that
// process's own minimum and within-process spread.
//
// Both spreads come from the wall series rather than the CPU series. getrusage
// reports whole microseconds and a measurement wraps a single encode or
// decode, so a short operation has its samples rounded onto two or three
// distinct integers and its spread reads 0.0% however much the runs differ.
// Wall time has nanosecond resolution, and wherever the CPU clock is not
// rounding the two agree to within hundredths of a percent.
struct TrialRecord {
  std::string dataset;
  std::string dtype;
  std::string alpEncoding;
  std::string rivalCompression;
  uint64_t alpBytes{0};
  uint64_t rivalBytes{0};
  double alpEncodeCpuMin{0.0};
  double alpEncodeIterRsd{0.0};
  double alpDecodeCpuMin{0.0};
  double alpDecodeIterRsd{0.0};
  double rivalEncodeCpuMin{0.0};
  double rivalEncodeIterRsd{0.0};
  double rivalDecodeCpuMin{0.0};
  double rivalDecodeIterRsd{0.0};
  double alpEncodeWallMin{0.0};
  double alpDecodeWallMin{0.0};
  double rivalEncodeWallMin{0.0};
  double rivalDecodeWallMin{0.0};
  bool correct{true};
};

TrialRecord toRecord(
    const std::string& dataset,
    const std::string& dtype,
    const PairStats& stats) {
  TrialRecord r;
  r.dataset = dataset;
  r.dtype = dtype;
  r.alpEncoding = stats.alp.encodingName;
  r.rivalCompression = stats.rival.compressionName;
  r.alpBytes = stats.alp.encodedBytes;
  r.rivalBytes = stats.rival.encodedBytes;
  r.alpEncodeCpuMin = stats.alp.encodeCpu.min();
  r.alpEncodeIterRsd = stats.alp.encodeWall.relativeStdDevPercent();
  r.alpDecodeCpuMin = stats.alp.decodeCpu.min();
  r.alpDecodeIterRsd = stats.alp.decodeWall.relativeStdDevPercent();
  r.rivalEncodeCpuMin = stats.rival.encodeCpu.min();
  r.rivalEncodeIterRsd = stats.rival.encodeWall.relativeStdDevPercent();
  r.rivalDecodeCpuMin = stats.rival.decodeCpu.min();
  r.rivalDecodeIterRsd = stats.rival.decodeWall.relativeStdDevPercent();
  r.alpEncodeWallMin = stats.alp.encodeWall.min();
  r.alpDecodeWallMin = stats.alp.decodeWall.min();
  r.rivalEncodeWallMin = stats.rival.encodeWall.min();
  r.rivalDecodeWallMin = stats.rival.decodeWall.min();
  r.correct = stats.alp.correct && stats.rival.correct;
  return r;
}

// Everything the parent knows about one table row once every trial has
// reported. The wall series carry one entry per trial, so their relative
// standard deviation is the cross-process spread.
struct RowSummary {
  std::string dataset;
  std::string dtype;
  std::string alpEncoding;
  std::string rivalCompression;
  uint64_t alpBytes{0};
  uint64_t rivalBytes{0};
  Series alpEncodeCpu;
  Series alpDecodeCpu;
  Series rivalEncodeCpu;
  Series rivalDecodeCpu;
  Series alpEncodeWall;
  Series alpDecodeWall;
  Series rivalEncodeWall;
  Series rivalDecodeWall;
  // Widest within-process spread any trial saw on any of the four series, so a
  // single noisy trial is not averaged away.
  double iterRsd{0.0};
  bool correct{true};

  void absorb(const TrialRecord& r) {
    dataset = r.dataset;
    dtype = r.dtype;
    alpEncoding = r.alpEncoding;
    rivalCompression = r.rivalCompression;
    alpBytes = r.alpBytes;
    rivalBytes = r.rivalBytes;
    alpEncodeCpu.add(r.alpEncodeCpuMin);
    alpDecodeCpu.add(r.alpDecodeCpuMin);
    rivalEncodeCpu.add(r.rivalEncodeCpuMin);
    rivalDecodeCpu.add(r.rivalDecodeCpuMin);
    alpEncodeWall.add(r.alpEncodeWallMin);
    alpDecodeWall.add(r.alpDecodeWallMin);
    rivalEncodeWall.add(r.rivalEncodeWallMin);
    rivalDecodeWall.add(r.rivalDecodeWallMin);
    iterRsd = std::max(
        {iterRsd,
         r.alpEncodeIterRsd,
         r.alpDecodeIterRsd,
         r.rivalEncodeIterRsd,
         r.rivalDecodeIterRsd});
    correct = correct && r.correct;
  }

  double trialRsd() const {
    return std::max(
        {alpEncodeWall.relativeStdDevPercent(),
         alpDecodeWall.relativeStdDevPercent(),
         rivalEncodeWall.relativeStdDevPercent(),
         rivalDecodeWall.relativeStdDevPercent()});
  }
};

// -----------------------------------------------------------------------------
// Output
// -----------------------------------------------------------------------------

struct Column {
  const char* label;
  // Printed width, not counting the space on either side of the separator.
  int width;
  bool leftAligned;
};

constexpr std::array<Column, 16> kColumns{{
    {"dataset", 28, true},
    {"dtype", 6, true},
    {"a-type", 9, true},
    {"a-bytes", 10, false},
    {"z-comp", 12, true},
    {"z-bytes", 10, false},
    {"size", 7, false},
    {"a-enc", 6, false},
    {"z-enc", 6, false},
    {"enc x", 6, false},
    {"a-dec", 6, false},
    {"z-dec", 6, false},
    {"dec x", 6, false},
    {"iter", 6, false},
    {"trial", 6, false},
    {"ok", 3, false},
}};

void printHeader() {
  {
    std::ostringstream oss;
    oss << "|";
    for (const auto& column : kColumns) {
      oss << " " << (column.leftAligned ? std::left : std::right)
          << std::setw(column.width) << column.label << " |";
    }
    LOG(INFO) << oss.str();
  }
  {
    std::ostringstream oss;
    oss << "|";
    for (const auto& column : kColumns) {
      oss << std::string(column.width + 2, '-') << "|";
    }
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

// Speedup of the alp arm over the rival arm. Both are times, so the faster arm
// is the smaller one and the ratio inverts.
double speedup(double alpMicros, double rivalMicros) {
  if (alpMicros <= 0.0) {
    return 0.0;
  }
  return rivalMicros / alpMicros;
}

void printRow(const RowSummary& row, uint32_t rows) {
  const double bytesPerValue = row.dtype == "float" ? 4.0 : 8.0;
  const double rawBytes = static_cast<double>(rows) * bytesPerValue;
  // Across trials the representative time is again the minimum: the fastest
  // process is the one whose layout and environment cost it least.
  const double alpEncode = row.alpEncodeCpu.min();
  const double alpDecode = row.alpDecodeCpu.min();
  const double rivalEncode = row.rivalEncodeCpu.min();
  const double rivalDecode = row.rivalDecodeCpu.min();
  const double sizeRatio = row.rivalBytes == 0
      ? 0.0
      : static_cast<double>(row.alpBytes) /
          static_cast<double>(row.rivalBytes) * 100.0;

  std::ostringstream oss;
  oss << "| " << std::left << std::setw(kColumns[0].width) << row.dataset
      << " | " << std::setw(kColumns[1].width) << row.dtype << " | "
      << std::setw(kColumns[2].width) << row.alpEncoding << " | " << std::right
      << std::setw(kColumns[3].width) << row.alpBytes << " | " << std::left
      << std::setw(kColumns[4].width) << row.rivalCompression << " | "
      << std::right << std::setw(kColumns[5].width) << row.rivalBytes << " | "
      << std::fixed << std::setprecision(2) << std::setw(6) << sizeRatio
      << "% | " << std::setw(kColumns[7].width)
      << gigabytesPerSecond(rawBytes, alpEncode) << " | "
      << std::setw(kColumns[8].width)
      << gigabytesPerSecond(rawBytes, rivalEncode) << " | " << std::setw(5)
      << speedup(alpEncode, rivalEncode) << "x | "
      << std::setw(kColumns[10].width)
      << gigabytesPerSecond(rawBytes, alpDecode) << " | "
      << std::setw(kColumns[11].width)
      << gigabytesPerSecond(rawBytes, rivalDecode) << " | " << std::setw(5)
      << speedup(alpDecode, rivalDecode) << "x | " << std::setprecision(1)
      << std::setw(5) << row.iterRsd << "% | " << std::setw(5) << row.trialRsd()
      << "% | " << std::setw(kColumns[15].width) << (row.correct ? "OK" : "BAD")
      << " |";
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
  flagDivergence("alp encode", alpEncode, row.alpEncodeWall.min());
  flagDivergence("alp decode", alpDecode, row.alpDecodeWall.min());
  flagDivergence("rival encode", rivalEncode, row.rivalEncodeWall.min());
  flagDivergence("rival decode", rivalDecode, row.rivalDecodeWall.min());
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
// field is a number or an enum name, so the parent can split on tabs without
// quoting rules.
// -----------------------------------------------------------------------------

constexpr char kRecordPrefix[] = "TRIAL\t";

void writeRecord(const TrialRecord& r) {
  std::printf(
      "%s%s\t%s\t%s\t%llu\t%llu"
      "\t%.6f\t%.6f\t%.6f\t%.6f"
      "\t%.6f\t%.6f\t%.6f\t%.6f"
      "\t%.6f\t%.6f\t%.6f\t%.6f"
      "\t%d\t%s\n",
      kRecordPrefix,
      r.dtype.c_str(),
      r.alpEncoding.c_str(),
      r.rivalCompression.c_str(),
      static_cast<unsigned long long>(r.alpBytes),
      static_cast<unsigned long long>(r.rivalBytes),
      r.alpEncodeCpuMin,
      r.alpEncodeIterRsd,
      r.alpDecodeCpuMin,
      r.alpDecodeIterRsd,
      r.rivalEncodeCpuMin,
      r.rivalEncodeIterRsd,
      r.rivalDecodeCpuMin,
      r.rivalDecodeIterRsd,
      r.alpEncodeWallMin,
      r.alpDecodeWallMin,
      r.rivalEncodeWallMin,
      r.rivalDecodeWallMin,
      r.correct ? 1 : 0,
      r.dataset.c_str());
}

// Parses one line a child produced. Returns false for any line that is not a
// record, which is how glog output on the shared stdout is skipped.
bool parseRecord(const std::string& line, TrialRecord& out) {
  const std::string prefix{kRecordPrefix};
  if (line.compare(0, prefix.size(), prefix) != 0) {
    return false;
  }
  std::vector<std::string> fields;
  size_t start = prefix.size();
  // The last field is the dataset name and may itself be anything, so stop
  // splitting once the fixed-width fields have been taken.
  constexpr int kFixedFields = 18;
  for (int i = 0; i < kFixedFields; ++i) {
    const size_t tab = line.find('\t', start);
    if (tab == std::string::npos) {
      return false;
    }
    fields.push_back(line.substr(start, tab - start));
    start = tab + 1;
  }
  out.dtype = fields[0];
  out.alpEncoding = fields[1];
  out.rivalCompression = fields[2];
  out.alpBytes = std::stoull(fields[3]);
  out.rivalBytes = std::stoull(fields[4]);
  out.alpEncodeCpuMin = std::stod(fields[5]);
  out.alpEncodeIterRsd = std::stod(fields[6]);
  out.alpDecodeCpuMin = std::stod(fields[7]);
  out.alpDecodeIterRsd = std::stod(fields[8]);
  out.rivalEncodeCpuMin = std::stod(fields[9]);
  out.rivalEncodeIterRsd = std::stod(fields[10]);
  out.rivalDecodeCpuMin = std::stod(fields[11]);
  out.rivalDecodeIterRsd = std::stod(fields[12]);
  out.alpEncodeWallMin = std::stod(fields[13]);
  out.alpDecodeWallMin = std::stod(fields[14]);
  out.rivalEncodeWallMin = std::stod(fields[15]);
  out.rivalDecodeWallMin = std::stod(fields[16]);
  out.correct = fields[17] == "1";
  out.dataset = line.substr(start);
  return true;
}

// Re-executes this binary once with --child and returns the records it
// emitted. Child stderr is left attached so its glog warnings reach the
// terminal.
std::vector<TrialRecord> runChild(const std::string& self) {
  std::ostringstream cmd;
  cmd << "'" << self << "'"
      << " --child"
      << " --rows=" << FLAGS_rows << " --warmup=" << FLAGS_warmup
      << " --iters=" << FLAGS_iters << " --rival='" << FLAGS_rival << "'";
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
  // Rejects a bad --rival before any measurement runs.
  const auto rival = rivalCompressionType();

  if (FLAGS_child) {
    for (const auto& record : collectTrial()) {
      writeRecord(record);
    }
    std::fflush(stdout);
    return 0;
  }

  const uint32_t trials = std::max(FLAGS_trials, 1u);

  LOG(INFO) << "=== ALP without compression vs Trivial + " << toString(rival)
            << " ===";
  LOG(INFO) << "rows / dataset: " << FLAGS_rows
            << ", warmup iters: " << FLAGS_warmup
            << ", measure iters: " << FLAGS_iters << ", trials: " << trials;
  LOG(INFO) << "alp arm: ManualEncodingSelectionPolicy "
               "[ALP, Trivial, FixedBitWidth], no compression options";
  LOG(INFO) << "rival arm: ManualEncodingSelectionPolicy [Trivial], "
            << toString(rival)
            << " with accept ratio 1.0 and no minimum input size";
  LOG(INFO) << "the two arms alternate within each iteration, so drift lands "
               "on both and divides out of the x columns";
  LOG(INFO) << "size: alp bytes / rival bytes; under 100% means ALP "
               "compressed better and the speedups below are earned";
  LOG(INFO) << "enc x, dec x: rival time / alp time; above 1.00x means ALP is "
               "faster";
  LOG(INFO) << "GB/s: (rows * sizeof(T)) / fastest cpu time";
  LOG(INFO) << "iter: widest spread across the measure iters of one process";
  LOG(INFO) << "trial: widest spread across the trial processes; a large value "
               "with a small iter means per-process layout, not noise";
  if (rival == CompressionType::OpenZL) {
    LOG(INFO) << "note: OpenZL routes Double to plain zstd and only Float to "
                 "its numeric graph, so the double rows below are really ALP "
                 "against Trivial + zstd";
  }
  if (trials < 2) {
    LOG(INFO) << "note: --trials=1 measures in this process, so the trial "
                 "column reads 0.0%; pass --trials=5 or more to populate it";
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
