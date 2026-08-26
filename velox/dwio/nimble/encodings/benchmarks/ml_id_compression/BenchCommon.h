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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <span>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <folly/FileUtil.h>
#include <folly/Random.h>
#include <folly/dynamic.h>
#include <folly/json/json.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/compression/Compression.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/CachePolicy.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ResultWriter.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/SubstreamCompression.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

// ---------------------------------------------------------------------------
// CLI flags shared across benchmark binaries
// ---------------------------------------------------------------------------

// Defined in MlIdBenchmarkFlags.cpp; link nimble_ml_id_benchmark_common to get
// them.
DECLARE_string(mlidc_output_csv);
DECLARE_string(mlidc_output_manifest);
DECLARE_int32(mlidc_rows);
DECLARE_int32(mlidc_iters);
DECLARE_int64(mlidc_seed);
DECLARE_string(mlidc_file);
DECLARE_string(mlidc_dataset_name);
DECLARE_string(mlidc_substream_compression);
DECLARE_string(mlidc_outer_compression);
DECLARE_int32(mlidc_block_codec_iters);
DECLARE_string(mlidc_datasets);
DECLARE_bool(mlidc_dump_encoding);
DECLARE_int32(mlidc_block_codec_probes);
DECLARE_string(mlidc_dtype);

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

  NimbleBenchTarget() : pool_(benchmarks::benchmarkPool()) {}

  // Encode data.  Destroys any previously encoded state.
  void encode(
      const Vector<T>& data,
      const Encoding::Options& options = {},
      bool realNestedSelection = false) {
    Buffer buf{*pool_};
    // Not test::Encoder::encode: its policy silently redirects any compressor
    // other than Zstd, and leaves nested sub-streams on the default one. See
    // SubstreamCompression.h.
    encoded_ = std::string(
        encodeWithCompression<EncodingT, T>(
            buf,
            data,
            parseCompressionType(FLAGS_mlidc_substream_compression),
            options,
            realNestedSelection));
    // Construct the Encoding directly from the encoded bytes rather than
    // re-encoding via createEncoding(), which would silently drop
    // realNestedSelection and produce different encoded data.
    encoding_ = std::make_unique<EncodingT>(
        *pool_, std::string_view(encoded_), benchmarks::nullFactory(), options);
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
  /// Returns the encoding tree, for reporting which nested encodings a
  /// selection policy actually chose. Empty for targets that are not Nimble
  /// encodings and so have no tree to show.
  virtual std::string describe() {
    return {};
  }
};

template <typename EncodingT>
struct NimbleBenchTargetImpl
    : NimbleBenchTargetBase<typename EncodingT::cppDataType> {
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
  std::string describe() override {
    auto* encoding = target.encoding();
    return encoding != nullptr ? encoding->debugString(0) : std::string{};
  }
};

// ---------------------------------------------------------------------------
// NimbleViewBenchTargetImpl<EncodingT>
// ---------------------------------------------------------------------------
// Encodes exactly as NimbleBenchTargetImpl does, then reads through an
// EncodingView instead of an Encoding.
//
// The two differ only in how a read is addressed. An Encoding carries a
// sequential cursor, so reaching row i means traversing from wherever the
// cursor is; a view is addressed by index. Pairing each view entry with its
// sequential twin in the encoder list makes that the only variable between
// them, since the encoded bytes are identical.
template <typename EncodingT>
class NimbleViewBenchTargetImpl
    : public NimbleBenchTargetBase<typename EncodingT::cppDataType> {
 public:
  using T = typename EncodingT::cppDataType;

  void encode(const Vector<T>& data, const Encoding::Options& opts) override {
    encodeWith(data, opts, /*realNestedSelection=*/false);
  }

  void encodeWith(
      const Vector<T>& data,
      const Encoding::Options& opts,
      bool realNestedSelection) {
    Buffer buf{*pool_};
    encoded_ = std::string(encodeWithCompression<EncodingT, T>(
        buf,
        data,
        parseCompressionType(FLAGS_mlidc_substream_compression),
        opts,
        realNestedSelection));
    options_ = opts;
    view_ = createEncodingView(std::string_view(encoded_), pool_.get(), opts);
    NIMBLE_CHECK_NOT_NULL(view_);
  }

  void materializeAll(T* dst, uint32_t n) override {
    view_->read(0, n, dst);
  }

  // A single-row read goes through readAt rather than a length-1 range: that is
  // the API a point lookup would actually use, and the one the point driver is
  // meant to be measuring.
  void materializeRange(uint32_t begin, uint32_t count, T* dst) override {
    if (count == 1) {
      view_->readAt(begin, dst);
    } else {
      view_->read(begin, count, dst);
    }
  }

  // No cursor, so no skip: each range is resolved from its own index. That is
  // the whole point of a view on a gather workload.
  void skipThenMaterialize(
      const std::vector<std::pair<uint32_t, uint32_t>>& ranges,
      T* dst) override {
    for (const auto& [begin, count] : ranges) {
      if (count == 1) {
        view_->readAt(begin, dst);
      } else {
        view_->read(begin, count, dst);
      }
      dst += count;
    }
  }

  size_t payloadSize() const override {
    return encoded_.size();
  }

  std::vector<std::span<const std::byte>> internalBuffers() const override {
    // The view's own index structures (a run-end array, say) are private to it,
    // so eviction reaches the payload only. Cache-state rows for view encoders
    // are therefore a lower bound on a genuinely cold read.
    return {
        {reinterpret_cast<const std::byte*>(encoded_.data()), encoded_.size()}};
  }

  // A view has no debugString, so build the encoding just to report its tree.
  // Only --mlidc_dump_encoding calls this, always outside a timed region.
  std::string describe() override {
    if (encoded_.empty()) {
      return {};
    }
    EncodingT encoding{
        *pool_,
        std::string_view(encoded_),
        benchmarks::nullFactory(),
        options_};
    return encoding.debugString(0);
  }

 private:
  std::shared_ptr<velox::memory::MemoryPool> pool_{benchmarks::benchmarkPool()};
  std::string encoded_;
  Encoding::Options options_;
  std::unique_ptr<EncodingView> view_;
};

template <typename T>
struct EncoderEntry {
  std::string name;
  std::string family; // "baseline", "sis-manual", "sis-auto", "fpe-index"
  std::string variant;
  bool isSequential{true};
  bool fastSkip{false};
  bool randomAccess{false};
  // True when every read, however small, must first decompress the entire
  // payload. Drivers use this to cap iterations so a block codec does not
  // dominate wall-clock time on the fine-grained access sweeps.
  bool wholePayloadCodec{false};

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
// Outer (whole-payload) compression
// ---------------------------------------------------------------------------

// Wraps an encoded column in a single block compressor, modelling shipping the
// whole encoded payload through a codec such as OpenZL.
//
// The point of measuring this separately is the read cost. A block codec has
// no addressable interior, so every access, including a one-element point
// lookup, must first decompress the entire payload. Any skip-based advantage
// the inner encoding has is therefore erased while the payload stays
// compressed, which is what the decode drivers are meant to expose.
//
// Decorates NimbleBenchTargetBase so it composes with any inner encoding
// without those encodings knowing about it.
template <typename T>
class OuterCompressedTarget : public NimbleBenchTargetBase<T> {
 public:
  // Takes an inner target that the encoder entry's factory has already
  // encoded, and compresses its payload.
  OuterCompressedTarget(
      std::unique_ptr<NimbleBenchTargetBase<T>> inner,
      CompressionType compressionType)
      : inner_{std::move(inner)}, compressionType_{compressionType} {
    compressInner();
  }

  void encode(const Vector<T>& data, const Encoding::Options& opts) override {
    inner_->encode(data, opts);
    compressInner();
  }
  void materializeAll(T* dst, uint32_t n) override {
    decompressAll();
    inner_->materializeAll(dst, n);
  }

  void materializeRange(uint32_t begin, uint32_t count, T* dst) override {
    decompressAll();
    inner_->materializeRange(begin, count, dst);
  }

  void skipThenMaterialize(
      const std::vector<std::pair<uint32_t, uint32_t>>& ranges,
      T* dst) override {
    decompressAll();
    inner_->skipThenMaterialize(ranges, dst);
  }

  // The stored size, which is what an outer codec is chosen for.
  size_t payloadSize() const override {
    return compressed_.size();
  }

  std::vector<std::span<const std::byte>> internalBuffers() const override {
    return {
        {reinterpret_cast<const std::byte*>(compressed_.data()),
         compressed_.size()}};
  }

  std::string describe() override {
    return inner_->describe();
  }

 private:
  void compressInner() {
    // payloadBytes() is not on the base interface; internalBuffers() exposes
    // the same bytes and is.
    auto buffers = inner_->internalBuffers();
    NIMBLE_CHECK(!buffers.empty(), "Inner target exposed no payload buffer");
    std::string_view view{
        reinterpret_cast<const char*>(buffers.front().data()),
        buffers.front().size()};

    BenchCompressPolicy policy{compressionType_};
    auto result = Compression::compress(
        *pool_, view, DataType::Int8, /*bitWidth=*/8, policy);

    // A compressor may decline, in which case the payload is stored as is and
    // reads skip the decompress step.
    if (result.buffer.has_value()) {
      compressed_.assign(result.buffer->data(), result.buffer->size());
      storedType_ = result.compressionType;
    } else {
      compressed_.assign(view.data(), view.size());
      storedType_ = CompressionType::Uncompressed;
    }
  }

  // Charged to every access, as it would be in a reader holding only the
  // compressed block.
  //
  // This times the decompression but discards the output: the inner target
  // still holds the payload it encoded, so decoding stays correct without
  // re-parsing. A real reader would also rebuild the Encoding from the
  // decompressed bytes, so the penalty measured here is a lower bound.
  void decompressAll() {
    if (storedType_ == CompressionType::Uncompressed) {
      return;
    }
    auto buffer = Compression::uncompress(
        *pool_,
        storedType_,
        DataType::Int8,
        std::string_view{compressed_.data(), compressed_.size()},
        /*decompressCounter=*/nullptr);
    // Kept so the compiler cannot elide the decompression.
    lastDecompressed_ = std::move(buffer);
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_{benchmarks::benchmarkPool()};
  std::unique_ptr<NimbleBenchTargetBase<T>> inner_;
  CompressionType compressionType_;
  CompressionType storedType_{CompressionType::Uncompressed};
  std::string compressed_;
  velox::BufferPtr lastDecompressed_;
};

// Wraps entry's factory so every target it builds carries the outer codec.
// Returns the entry unchanged when no outer compression is configured.
template <typename T>
EncoderEntry<T> withOuterCompression(
    EncoderEntry<T> entry,
    CompressionType compressionType) {
  if (compressionType == CompressionType::Uncompressed) {
    return entry;
  }
  entry.name += "+outer:" + nimble::toString(compressionType);
  // An outer block codec removes any interior addressability the inner
  // encoding had.
  entry.fastSkip = false;
  entry.randomAccess = false;
  entry.wholePayloadCodec = true;
  auto inner = std::move(entry.factory);
  entry.factory = [inner = std::move(inner), compressionType](
                      const Vector<T>& data, const Encoding::Options& opts) {
    auto target = std::make_unique<OuterCompressedTarget<T>>(
        inner(data, opts), compressionType);
    // The constructor compresses the payload the inner factory just encoded;
    // calling encode() here would encode a second time.
    return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(target));
  };
  return entry;
}

// ---------------------------------------------------------------------------
// DatasetEntry and the default dataset suites
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
    data[i] =
        static_cast<T>(static_cast<U>(state >> (64 - sizeof(T) * 8)) & mask);
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
Vector<T>
makeLowCardinalitySeeded(uint32_t cardinality, uint32_t n, uint64_t seed) {
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

// ---------------------------------------------------------------------------
// Float generators
// ---------------------------------------------------------------------------
// The integer generators above build values in the bit domain: they mask and
// shift a raw LCG word. That is meaningless for float and double, and
// makeNarrowSeeded does not even compile for them, because std::make_unsigned_t
// is ill-formed on a floating point type.
//
// These are value-domain analogues carrying the same six names, so the same
// --mlidc_datasets selection works whatever --mlidc_dtype is set to, and a
// float row lines up with the int64 row of the same dataset. They deliberately
// produce ordinary finite values with fractional parts rather than reusing the
// integer bit patterns: bit-casting random words into floats yields mostly NaNs
// and denormals, which compress unlike any real float column.

// Advances the shared LCG and returns a value in [0, 1).
inline double nextUnitDouble(uint64_t& state) {
  state = state * 6364136223846793005ULL + 1442695040888963407ULL;
  // Top 53 bits: one full double mantissa, so the result is uniform.
  return static_cast<double>(state >> 11) / 9007199254740992.0;
}

template <typename T>
Vector<T> makeFloatUniformSeeded(uint32_t n, uint64_t seed) {
  auto& pool = benchmarks::benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);
  uint64_t state = seed ^ 0x9e3779b97f4a7c15ULL;
  for (uint32_t i = 0; i < n; ++i) {
    // Spread over a wide signed range, the float counterpart of a full-width
    // integer draw.
    data[i] = static_cast<T>((nextUnitDouble(state) - 0.5) * 2.0e9);
  }
  return data;
}

// Values whose integral part spans `bitWidth` bits, plus a fractional part.
// The bit width is what the integer suite varies, so keeping it as the knob
// makes the narrow-20bit and narrow-40bit rows comparable across dtypes.
template <typename T>
Vector<T> makeFloatNarrowSeeded(int bitWidth, uint32_t n, uint64_t seed) {
  auto& pool = benchmarks::benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);
  const double range = std::exp2(static_cast<double>(bitWidth));
  uint64_t state = seed ^ 0x9e3779b97f4a7c15ULL;
  for (uint32_t i = 0; i < n; ++i) {
    data[i] = static_cast<T>(nextUnitDouble(state) * range);
  }
  return data;
}

template <typename T>
Vector<T> makeFloatIncreasingSeeded(uint32_t n, uint64_t seed) {
  auto& pool = benchmarks::benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);
  uint64_t state = seed ^ 0x9e3779b97f4a7c15ULL;
  double val = 0.0;
  for (uint32_t i = 0; i < n; ++i) {
    val += nextUnitDouble(state) * 8.0; // delta in [0,8), mirrors [1,8]
    data[i] = static_cast<T>(val);
  }
  return data;
}

template <typename T>
Vector<T>
makeFloatLowCardinalitySeeded(uint32_t cardinality, uint32_t n, uint64_t seed) {
  auto& pool = benchmarks::benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);
  // Draw from a fixed palette so the value count is exactly `cardinality`;
  // scaling a random draw would leave it approximate.
  std::vector<T> palette(cardinality);
  uint64_t paletteState = seed ^ 0x9e3779b97f4a7c15ULL;
  for (uint32_t c = 0; c < cardinality; ++c) {
    palette[c] = static_cast<T>(nextUnitDouble(paletteState) * 1000.0);
  }
  uint64_t state = seed ^ 0xa5a5a5a5a5a5a5a5ULL;
  for (uint32_t i = 0; i < n; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    data[i] = palette[(state >> 33) % cardinality];
  }
  return data;
}

template <typename T>
Vector<T> makeFloatRunLengthSeeded(uint32_t n, uint64_t seed) {
  auto& pool = benchmarks::benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);
  uint64_t state = seed ^ 0x9e3779b97f4a7c15ULL;
  uint32_t i = 0;
  while (i < n) {
    const T val = static_cast<T>(nextUnitDouble(state) * 1000.0);
    uint32_t runLen = static_cast<uint32_t>(10 + (state >> 33) % 50);
    runLen = std::min(runLen, n - i);
    for (uint32_t j = 0; j < runLen; ++j) {
      data[i + j] = val;
    }
    i += runLen;
  }
  return data;
}

// Parses one value of T from a line of the real-data file.
//
// std::stoll stops at the '.', so parsing a float column with it would load
// 1.5 as 1: a wrong value that looks exactly like a legitimate benchmark
// result. Each type gets the parser that matches it, and the narrower integer
// types are range-checked rather than silently truncated by a static_cast.
template <typename T>
T parseColumnValue(const std::string& line) {
  if constexpr (std::is_same_v<T, float>) {
    return std::stof(line);
  } else if constexpr (std::is_same_v<T, double>) {
    return std::stod(line);
  } else if constexpr (std::is_unsigned_v<T>) {
    const unsigned long long raw = std::stoull(line);
    if (raw > static_cast<unsigned long long>(std::numeric_limits<T>::max())) {
      throw std::runtime_error(
          "Value out of range for the selected --mlidc_dtype: " + line);
    }
    return static_cast<T>(raw);
  } else {
    const long long raw = std::stoll(line);
    if (raw < static_cast<long long>(std::numeric_limits<T>::min()) ||
        raw > static_cast<long long>(std::numeric_limits<T>::max())) {
      throw std::runtime_error(
          "Value out of range for the selected --mlidc_dtype: " + line);
    }
    return static_cast<T>(raw);
  }
}

// Loads a real-data column from a text file holding one value per line, parsed
// as the type selected by --mlidc_dtype. Reads exactly the first n values and
// throws if the file is missing or shorter than requested: a silently short
// read would be indistinguishable from a legitimate benchmark result.
//
// The int64 path is the format read by the --file flag of
// velox/dwio/nimble/tools/encoding_bench, so the same column dump feeds both
// tools and the results cross-check. That tool parses int64 only
// (tools/encoding_bench/EncodingBench.cpp:121), so the other types are an
// extension this suite makes alone.
template <typename T>
Vector<T> loadColumnLines(const std::string& path, uint32_t n) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Cannot open data file: " + path);
  }

  auto& pool = benchmarks::benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);

  std::string line;
  uint32_t count = 0;
  while (count < n && std::getline(file, line)) {
    if (!line.empty()) {
      data[count++] = parseColumnValue<T>(line);
    }
  }

  if (count < n) {
    throw std::runtime_error(
        "Data file has fewer values than requested. Path: " + path +
        ", available: " + std::to_string(count) +
        ", requested: " + std::to_string(n));
  }
  return data;
}

} // namespace detail

// Default dataset suite: the main distribution shapes expected for ML ID
// workloads, plus any real-data column supplied at run time.
//
// Both the integer and float suites carry the same six names, so a
// --mlidc_datasets selection means the same thing at every --mlidc_dtype and
// rows for different types line up dataset by dataset.
template <typename T>
std::vector<DatasetEntry<T>> defaultDatasets() {
  std::vector<DatasetEntry<T>> out;

  if constexpr (std::is_floating_point_v<T>) {
    out.push_back({"uniform-full", [](uint32_t n, uint64_t seed) {
                     return detail::makeFloatUniformSeeded<T>(n, seed);
                   }});

    out.push_back({"narrow-20bit", [](uint32_t n, uint64_t seed) {
                     return detail::makeFloatNarrowSeeded<T>(20, n, seed);
                   }});

    // Only for 8-byte types: see the integer branch below.
    if constexpr (sizeof(T) == 8) {
      out.push_back({"narrow-40bit", [](uint32_t n, uint64_t seed) {
                       return detail::makeFloatNarrowSeeded<T>(40, n, seed);
                     }});
    }

    out.push_back({"increasing-small-delta", [](uint32_t n, uint64_t seed) {
                     return detail::makeFloatIncreasingSeeded<T>(n, seed);
                   }});

    out.push_back({"low-cardinality-256", [](uint32_t n, uint64_t seed) {
                     return detail::makeFloatLowCardinalitySeeded<T>(
                         256, n, seed);
                   }});

    out.push_back({"run-length", [](uint32_t n, uint64_t seed) {
                     return detail::makeFloatRunLengthSeeded<T>(n, seed);
                   }});
  } else {
    out.push_back({"uniform-full", [](uint32_t n, uint64_t seed) {
                     return detail::makeRandomSeeded<T>(n, seed);
                   }});

    out.push_back({"narrow-20bit", [](uint32_t n, uint64_t seed) {
                     return detail::makeNarrowSeeded<T>(20, n, seed);
                   }});

    // A 40-bit draw has no meaning in a 32-bit type, and makeNarrowSeeded
    // would shift by more than the width of T, which is undefined behaviour.
    if constexpr (sizeof(T) == 8) {
      out.push_back({"narrow-40bit", [](uint32_t n, uint64_t seed) {
                       return detail::makeNarrowSeeded<T>(40, n, seed);
                     }});
    }

    out.push_back({"increasing-small-delta", [](uint32_t n, uint64_t seed) {
                     return detail::makeIncreasingSeeded<T>(n, seed);
                   }});

    out.push_back({"low-cardinality-256", [](uint32_t n, uint64_t seed) {
                     return detail::makeLowCardinalitySeeded<T>(256, n, seed);
                   }});

    out.push_back({"run-length", [](uint32_t n, uint64_t seed) {
                     return detail::makeRunLengthSeeded<T>(n, seed);
                   }});
  }

  // Real-data column supplied at run time. Unlike the synthetic generators
  // above this is not regenerated per seed, so the seed is ignored.
  if (!FLAGS_mlidc_file.empty()) {
    out.push_back({FLAGS_mlidc_dataset_name, [](uint32_t n, uint64_t /*seed*/) {
                     return detail::loadColumnLines<T>(FLAGS_mlidc_file, n);
                   }});
  }

  // Applied last so a real-data dataset can be selected by name too.
  if (!FLAGS_mlidc_datasets.empty()) {
    std::vector<DatasetEntry<T>> filtered;
    std::stringstream names(FLAGS_mlidc_datasets);
    std::string want;
    while (std::getline(names, want, ',')) {
      if (want.empty()) {
        continue;
      }
      auto it = std::find_if(out.begin(), out.end(), [&](const auto& entry) {
        return entry.name == want;
      });
      // A typo would otherwise run nothing and look like a clean empty result.
      if (it == out.end()) {
        throw std::runtime_error("Unknown dataset name: " + want);
      }
      filtered.push_back(*it);
    }
    out = std::move(filtered);
  }

  return out;
}

// ---------------------------------------------------------------------------
// Default 9-encoder suite shared across decode/encode/smoke drivers.
// ---------------------------------------------------------------------------

template <typename T>
std::vector<EncoderEntry<T>> buildDefaultEncoders() {
  std::vector<EncoderEntry<T>> encoders;

  encoders.push_back(
      makeEncoderEntry<TrivialEncoding<T>>(
          "Trivial", "Baseline", "trivial", true, true, false));
  encoders.push_back(
      makeEncoderEntry<FixedBitWidthEncoding<T>>(
          "FixedBitWidth", "Baseline", "fbw", true, true, false));
  encoders.push_back(
      makeEncoderEntry<DictionaryEncoding<T>>(
          "Dictionary", "Baseline", "dict", true, false, false));
  encoders.push_back(
      makeEncoderEntry<RLEEncoding<T>>(
          "RLE", "Baseline", "rle", true, true, false));

  // Read-path variants. Each encodes byte-for-byte identically to the entry it
  // shadows and differs only in reading by index rather than by cursor, so the
  // pair isolates what indexed access is worth.
  {
    EncoderEntry<T> entry;
    entry.name = "RLE/view";
    entry.family = "Baseline";
    entry.variant = "rle_view";
    entry.isSequential = false;
    entry.fastSkip = true;
    entry.randomAccess = true;
    entry.factory = [](const Vector<T>& data, const Encoding::Options& opts) {
      auto impl = std::make_unique<NimbleViewBenchTargetImpl<RLEEncoding<T>>>();
      impl->encode(data, opts);
      return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
    };
    encoders.push_back(std::move(entry));
  }

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
    entry.factory = [idx](
                        const Vector<T>& data, const Encoding::Options& opts) {
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
    entry.factory = [](const Vector<T>& data, const Encoding::Options& opts) {
      auto impl =
          std::make_unique<NimbleBenchTargetImpl<SubIntSplitEncoding<T>>>();
      impl->target.encode(data, opts, /*realNestedSelection=*/true);
      return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
    };
    encoders.push_back(std::move(entry));
  }

  {
    EncoderEntry<T> entry;
    entry.name = "SIS/realNested+view";
    entry.family = "SubIntSplit";
    entry.variant = "real_nested_view";
    entry.isSequential = false;
    entry.fastSkip = true;
    entry.randomAccess = true;
    entry.factory = [](const Vector<T>& data, const Encoding::Options& opts) {
      auto impl = std::make_unique<
          NimbleViewBenchTargetImpl<SubIntSplitEncoding<T>>>();
      impl->encodeWith(data, opts, /*realNestedSelection=*/true);
      return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
    };
    encoders.push_back(std::move(entry));
  }

  // Applied last so it wraps whatever the entries above produced.
  const auto outerType = parseCompressionType(FLAGS_mlidc_outer_compression);
  for (auto& entry : encoders) {
    entry = withOuterCompression<T>(std::move(entry), outerType);
  }

  return encoders;
}

} // namespace facebook::nimble::mlidc

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
