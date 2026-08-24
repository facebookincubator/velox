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

#pragma once

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <array>
#include <cstdint>
#include <fstream>
#include <functional>
#include <map>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include <folly/FileUtil.h>
#include <folly/Random.h>
#include <folly/dynamic.h>
#include <folly/json/json.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/CachePolicy.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

// ---------------------------------------------------------------------------
// CLI flags shared across benchmark binaries
// ---------------------------------------------------------------------------

// Defined in MlIdBenchmarkFlags.cpp; link nimble_ml_id_benchmark_common to get them.
DECLARE_string(mlidc_output_csv);
DECLARE_string(mlidc_output_manifest);
DECLARE_int32(mlidc_rows);
DECLARE_int32(mlidc_iters);
DECLARE_int64(mlidc_seed);

namespace facebook::nimble::mlidc {

// ---------------------------------------------------------------------------
// NimbleBenchTarget<EncodingT>
// ---------------------------------------------------------------------------
// Wraps a single encode/decode cycle.  After encode() the object holds the
// serialised bytes in an internal Buffer together with a live Encoding object
// ready for decode operations.

template <typename EncodingT>
class NimbleBenchTarget {
 public:
  using T = typename EncodingT::cppDataType;

  NimbleBenchTarget()
      : pool_(benchmarks::benchmarkPool()) {}

  // Encode data.  Destroys any previously encoded state.
  void encode(
      const Vector<T>& data,
      const Encoding::Options& options = {},
      bool realNestedSelection = false) {
    Buffer buf{*pool_};
    encoded_ = std::string(
        test::Encoder<EncodingT>::encode(
            buf, data, CompressionType::Uncompressed, options,
            realNestedSelection));
    // Construct the Encoding directly from the encoded bytes rather than
    // re-encoding via createEncoding(), which would silently drop
    // realNestedSelection and produce different encoded data.
    encoding_ = std::make_unique<EncodingT>(
        *pool_,
        std::string_view(encoded_),
        benchmarks::nullFactory(),
        options);
  }

  // reset + materialize all n rows into dst.
  void materializeAll(T* dst, uint32_t n) {
    encoding_->reset();
    encoding_->materialize(n, dst);
  }

  // reset + skip begin rows + materialize count rows into dst.
  void materializeRange(uint32_t begin, uint32_t count, T* dst) {
    encoding_->reset();
    if (begin > 0) {
      encoding_->skip(begin);
    }
    encoding_->materialize(count, dst);
  }

  // Gather pattern: for each [begin, count) range in sorted order, skip then
  // materialize.  dst must have space for the total number of rows across all
  // ranges.
  void skipThenMaterialize(
      const std::vector<std::pair<uint32_t, uint32_t>>& ranges,
      T* dst) {
    encoding_->reset();
    uint32_t cursor = 0;
    for (auto& [begin, count] : ranges) {
      if (begin > cursor) {
        encoding_->skip(begin - cursor);
        cursor = begin;
      }
      encoding_->materialize(count, dst);
      dst += count;
      cursor += count;
    }
  }

  std::span<const std::byte> payloadBytes() const {
    return {
        reinterpret_cast<const std::byte*>(encoded_.data()), encoded_.size()};
  }

  size_t payloadSize() const {
    return encoded_.size();
  }

  // Spans covering all internal buffer regions — useful for cache eviction.
  std::vector<std::span<const std::byte>> internalBuffers() const {
    return {payloadBytes()};
  }

  Encoding* encoding() {
    return encoding_.get();
  }

 private:
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::string encoded_;
  std::unique_ptr<Encoding> encoding_;
};

// ---------------------------------------------------------------------------
// CsvResultWriter
// ---------------------------------------------------------------------------

class CsvResultWriter {
 public:
  CsvResultWriter(
      const std::string& path,
      std::vector<std::string> columns)
      : path_(path), columns_(std::move(columns)) {
    file_.open(path_, std::ios::out | std::ios::trunc);
    NIMBLE_CHECK(file_.is_open(), "Cannot open CSV output: " + path_);
    // Write header.
    for (size_t i = 0; i < columns_.size(); ++i) {
      if (i) {
        file_ << ',';
      }
      file_ << columns_[i];
    }
    file_ << '\n';
  }

  void beginRow() {
    row_.clear();
  }

  void set(const std::string& col, const std::string& value) {
    // Minimal CSV quoting: wrap in quotes if value contains comma, quote, or
    // newline.
    if (value.find_first_of(",\"\n") != std::string::npos) {
      std::string quoted = "\"";
      for (char c : value) {
        if (c == '"') {
          quoted += "\"\"";
        } else {
          quoted += c;
        }
      }
      quoted += '"';
      row_[col] = std::move(quoted);
    } else {
      row_[col] = value;
    }
  }

  void set(const std::string& col, int64_t value) {
    row_[col] = std::to_string(value);
  }

  void set(const std::string& col, double value) {
    std::ostringstream oss;
    oss << value;
    row_[col] = oss.str();
  }

  void endRow() {
    for (size_t i = 0; i < columns_.size(); ++i) {
      if (i) {
        file_ << ',';
      }
      auto it = row_.find(columns_[i]);
      if (it != row_.end()) {
        file_ << it->second;
      }
      // Missing column → empty (null in CSV).
    }
    file_ << '\n';
  }

  void flush() {
    file_.flush();
  }

 private:
  std::string path_;
  std::vector<std::string> columns_;
  std::ofstream file_;
  std::map<std::string, std::string> row_;
};

// ---------------------------------------------------------------------------
// RunManifest
// ---------------------------------------------------------------------------

namespace detail {

inline std::string readFile(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    return "";
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  std::string s = ss.str();
  // Strip trailing newline.
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
    s.pop_back();
  }
  return s;
}

inline std::string cpuModel() {
  std::ifstream f("/proc/cpuinfo");
  if (!f.is_open()) {
    return "unknown";
  }
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("model name", 0) == 0) {
      auto colon = line.find(':');
      if (colon != std::string::npos) {
        auto s = line.substr(colon + 1);
        // ltrim
        s.erase(0, s.find_first_not_of(" \t"));
        return s;
      }
    }
  }
  return "unknown";
}

inline folly::dynamic cacheTopology() {
  auto topo = CacheTopology::detect();
  return folly::dynamic::object(
      "l1d_bytes", static_cast<int64_t>(topo.l1dBytes))(
      "l2_bytes", static_cast<int64_t>(topo.l2Bytes))(
      "llc_bytes", static_cast<int64_t>(topo.llcBytes))(
      "line_bytes", static_cast<int64_t>(topo.lineBytes));
}

inline std::string scalingGovernor() {
  return readFile(
      "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
}

inline std::string hostname() {
  char buf[256] = {};
  if (::gethostname(buf, sizeof(buf)) == 0) {
    return buf;
  }
  return "unknown";
}

} // namespace detail

// Write a JSON manifest sidecar describing the current run environment.
inline void writeRunManifest(const std::string& path) {
  folly::dynamic manifest = folly::dynamic::object;

  manifest["hostname"] = detail::hostname();
  manifest["cpu_model"] = detail::cpuModel();
  manifest["cache_topology"] = detail::cacheTopology();
  manifest["scaling_governor"] = detail::scalingGovernor();
  manifest["compiler_version"] = __VERSION__;

#ifdef MLIDC_GIT_SHA
  manifest["build_sha"] = MLIDC_GIT_SHA;
#else
  manifest["build_sha"] = "unknown";
#endif

  // Reproduce key flags.
  manifest["flags"] = folly::dynamic::object(
      "mlidc_rows", FLAGS_mlidc_rows)(
      "mlidc_iters", FLAGS_mlidc_iters)(
      "mlidc_seed", FLAGS_mlidc_seed);

  auto json = folly::toPrettyJson(manifest);
  if (folly::writeFile(json, path.c_str())) {
    LOG(INFO) << "Manifest written to " << path;
  } else {
    LOG(WARNING) << "Failed to write manifest to " << path;
  }
}

// ---------------------------------------------------------------------------
// EncoderEntry
// ---------------------------------------------------------------------------

template <typename T>
struct NimbleBenchTargetBase {
  virtual ~NimbleBenchTargetBase() = default;
  virtual void encode(const Vector<T>& data, const Encoding::Options& opts) = 0;
  virtual void materializeAll(T* dst, uint32_t n) = 0;
  virtual void materializeRange(uint32_t begin, uint32_t count, T* dst) = 0;
  virtual void skipThenMaterialize(
      const std::vector<std::pair<uint32_t, uint32_t>>& ranges,
      T* dst) = 0;
  virtual size_t payloadSize() const = 0;
  virtual std::vector<std::span<const std::byte>> internalBuffers() const = 0;
};

template <typename EncodingT>
struct NimbleBenchTargetImpl : NimbleBenchTargetBase<typename EncodingT::cppDataType> {
  using T = typename EncodingT::cppDataType;

  NimbleBenchTarget<EncodingT> target;

  void encode(const Vector<T>& data, const Encoding::Options& opts) override {
    target.encode(data, opts);
  }
  void materializeAll(T* dst, uint32_t n) override {
    target.materializeAll(dst, n);
  }
  void materializeRange(uint32_t begin, uint32_t count, T* dst) override {
    target.materializeRange(begin, count, dst);
  }
  void skipThenMaterialize(
      const std::vector<std::pair<uint32_t, uint32_t>>& ranges,
      T* dst) override {
    target.skipThenMaterialize(ranges, dst);
  }
  size_t payloadSize() const override {
    return target.payloadSize();
  }
  std::vector<std::span<const std::byte>> internalBuffers() const override {
    return target.internalBuffers();
  }
};

template <typename T>
struct EncoderEntry {
  std::string name;
  std::string family;   // "baseline", "sis-manual", "sis-auto", "fpe-index"
  std::string variant;
  bool isSequential{true};
  bool fastSkip{false};
  bool randomAccess{false};

  // Factory: construct a fresh target and encode the given data.
  std::function<std::unique_ptr<NimbleBenchTargetBase<T>>(
      const Vector<T>&,
      const Encoding::Options&)>
      factory;
};

// Convenience builder for a concrete EncodingT.
template <typename EncodingT>
EncoderEntry<typename EncodingT::cppDataType> makeEncoderEntry(
    std::string name,
    std::string family,
    std::string variant,
    bool isSequential = true,
    bool fastSkip = false,
    bool randomAccess = false) {
  using T = typename EncodingT::cppDataType;
  EncoderEntry<T> entry;
  entry.name = std::move(name);
  entry.family = std::move(family);
  entry.variant = std::move(variant);
  entry.isSequential = isSequential;
  entry.fastSkip = fastSkip;
  entry.randomAccess = randomAccess;
  entry.factory = [](const Vector<T>& data, const Encoding::Options& opts) {
    auto impl = std::make_unique<NimbleBenchTargetImpl<EncodingT>>();
    impl->encode(data, opts);
    return impl;
  };
  return entry;
}

// ---------------------------------------------------------------------------
// DatasetEntry and default int64 datasets
// ---------------------------------------------------------------------------

template <typename T>
struct DatasetEntry {
  std::string name;
  std::function<Vector<T>(uint32_t n, uint64_t seed)> generate;
};

namespace detail {

// Seed-controlled wrappers around BenchmarkUtils generators.  The seed is
// applied by seeding folly::Random before calling the generator — the
// generators use folly::Random::secureRand* which is thread-local state
// (unseedable), so we provide a best-effort deterministic path via a simple
// linear-congruential RNG to fill the buffer directly.

template <typename T>
Vector<T> makeRandomSeeded(uint32_t n, uint64_t seed) {
  auto& pool = benchmarks::benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);
  // LCG for reproducibility (Knuth parameters).
  uint64_t state = seed ^ 0x9e3779b97f4a7c15ULL;
  for (uint32_t i = 0; i < n; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    if constexpr (sizeof(T) <= 4) {
      data[i] = static_cast<T>(static_cast<uint32_t>(state >> 33));
    } else {
      uint64_t hi = state;
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      data[i] = static_cast<T>((hi & 0xFFFFFFFF00000000ULL) | (state >> 33));
    }
  }
  return data;
}

template <typename T>
Vector<T> makeNarrowSeeded(int bitWidth, uint32_t n, uint64_t seed) {
  auto& pool = benchmarks::benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);
  using U = std::make_unsigned_t<T>;
  U mask = (bitWidth >= static_cast<int>(sizeof(T) * 8))
      ? static_cast<U>(~U{0})
      : (static_cast<U>(1) << bitWidth) - 1;
  uint64_t state = seed ^ 0x9e3779b97f4a7c15ULL;
  for (uint32_t i = 0; i < n; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    data[i] = static_cast<T>(static_cast<U>(state >> (64 - sizeof(T) * 8)) & mask);
  }
  return data;
}

template <typename T>
Vector<T> makeIncreasingSeeded(uint32_t n, uint64_t seed) {
  auto& pool = benchmarks::benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);
  uint64_t state = seed ^ 0x9e3779b97f4a7c15ULL;
  T val = 0;
  for (uint32_t i = 0; i < n; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    val += static_cast<T>((state >> 61) + 1); // delta in [1,8]
    data[i] = val;
  }
  return data;
}

template <typename T>
Vector<T> makeLowCardinalitySeeded(
    uint32_t cardinality,
    uint32_t n,
    uint64_t seed) {
  auto& pool = benchmarks::benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);
  uint64_t state = seed ^ 0x9e3779b97f4a7c15ULL;
  for (uint32_t i = 0; i < n; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    data[i] = static_cast<T>((state >> 33) % cardinality);
  }
  return data;
}

template <typename T>
Vector<T> makeRunLengthSeeded(uint32_t n, uint64_t seed) {
  auto& pool = benchmarks::benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);
  uint64_t state = seed ^ 0x9e3779b97f4a7c15ULL;
  auto next = [&]() -> uint64_t {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state >> 33;
  };
  uint32_t i = 0;
  while (i < n) {
    T val = static_cast<T>(next() % 1000);
    uint32_t runLen = static_cast<uint32_t>(10 + next() % 50);
    runLen = std::min(runLen, n - i);
    for (uint32_t j = 0; j < runLen; ++j) {
      data[i + j] = val;
    }
    i += runLen;
  }
  return data;
}

} // namespace detail

// Default dataset suite for int64 — covers the main distribution shapes
// expected for ML ID workloads.
template <typename T>
std::vector<DatasetEntry<T>> defaultInt64Datasets() {
  std::vector<DatasetEntry<T>> out;

  out.push_back({"uniform-64bit", [](uint32_t n, uint64_t seed) {
                   return detail::makeRandomSeeded<T>(n, seed);
                 }});

  out.push_back({"narrow-20bit", [](uint32_t n, uint64_t seed) {
                   return detail::makeNarrowSeeded<T>(20, n, seed);
                 }});

  out.push_back({"narrow-40bit", [](uint32_t n, uint64_t seed) {
                   return detail::makeNarrowSeeded<T>(40, n, seed);
                 }});

  out.push_back({"increasing-small-delta", [](uint32_t n, uint64_t seed) {
                   return detail::makeIncreasingSeeded<T>(n, seed);
                 }});

  out.push_back({"low-cardinality-256", [](uint32_t n, uint64_t seed) {
                   return detail::makeLowCardinalitySeeded<T>(256, n, seed);
                 }});

  out.push_back({"run-length", [](uint32_t n, uint64_t seed) {
                   return detail::makeRunLengthSeeded<T>(n, seed);
                 }});

  return out;
}

// ---------------------------------------------------------------------------
// Default 9-encoder suite shared across decode/encode/smoke drivers.
// ---------------------------------------------------------------------------

template <typename T>
std::vector<EncoderEntry<T>> buildDefaultEncoders() {
  std::vector<EncoderEntry<T>> encoders;

  encoders.push_back(makeEncoderEntry<TrivialEncoding<T>>(
      "Trivial", "Baseline", "trivial", true, true, false));
  encoders.push_back(makeEncoderEntry<FixedBitWidthEncoding<T>>(
      "FixedBitWidth", "Baseline", "fbw", true, true, false));
  encoders.push_back(makeEncoderEntry<DictionaryEncoding<T>>(
      "Dictionary", "Baseline", "dict", true, false, false));
  encoders.push_back(makeEncoderEntry<RLEEncoding<T>>(
      "RLE", "Baseline", "rle", true, true, false));

  const std::array<std::string, 4> fpeNames = {
      "fpe_noindex", "fpe_pertier", "fpe_tagtag", "fpe_elias"};
  const std::array<bool, 4> fpeRA = {false, true, true, true};
  const std::array<bool, 4> fpeSkip = {false, true, true, true};

  for (int idx = 0; idx < 4; ++idx) {
    EncoderEntry<T> entry;
    entry.name = "FPE/" + fpeNames[idx];
    entry.family = "FrequencyPartition";
    entry.variant = fpeNames[idx];
    entry.isSequential = true;
    entry.fastSkip = fpeSkip[idx];
    entry.randomAccess = fpeRA[idx];
    entry.factory = [idx](const Vector<T>& data,
                          const Encoding::Options& opts) {
      auto impl = std::make_unique<
          NimbleBenchTargetImpl<FrequencyPartitionEncoding<T>>>();
      Encoding::Options o = opts;
      o.frequencyPartitionIndex = static_cast<uint8_t>(idx);
      impl->target.encode(data, o);
      return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
    };
    encoders.push_back(std::move(entry));
  }

  {
    EncoderEntry<T> entry;
    entry.name = "SIS/realNested";
    entry.family = "SubIntSplit";
    entry.variant = "real_nested";
    entry.isSequential = false;
    entry.fastSkip = false;
    entry.randomAccess = false;
    entry.factory = [](const Vector<T>& data,
                       const Encoding::Options& opts) {
      auto impl =
          std::make_unique<NimbleBenchTargetImpl<SubIntSplitEncoding<T>>>();
      impl->target.encode(data, opts, /*realNestedSelection=*/true);
      return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
    };
    encoders.push_back(std::move(entry));
  }

  return encoders;
}

} // namespace facebook::nimble::mlidc

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
