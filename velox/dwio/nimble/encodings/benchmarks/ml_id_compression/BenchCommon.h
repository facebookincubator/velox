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
#include <unordered_set>
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
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/AccessStructure.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/CachePolicy.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/EncodeCache.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/EncodingNodeEstimates.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/InputOrder.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ResultWriter.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/SubstreamCompression.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/tools/EncodingUtilities.h"

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
DECLARE_string(mlidc_input_order);
DECLARE_string(mlidc_substream_compression);
DECLARE_bool(mlidc_sis_row_frame);
DECLARE_string(mlidc_sis_upstream_features);
DECLARE_bool(mlidc_sis_estimate_compression_guard);
DECLARE_bool(mlidc_sis_estimate_bitflip_screen);
DECLARE_string(mlidc_outer_compression);
DECLARE_int32(mlidc_block_codec_iters);
DECLARE_string(mlidc_datasets);
DECLARE_string(mlidc_encoders);
DECLARE_double(mlidc_sis_decode_weight);
DECLARE_int32(mlidc_sis_decode_access_pattern);
DECLARE_int32(mlidc_sis_decode_read_path);
DECLARE_int32(mlidc_sis_admission);
DECLARE_bool(mlidc_sis_admission_forces);
DECLARE_double(mlidc_sis_max_size_regression);
DECLARE_bool(mlidc_dump_encoding);
DECLARE_string(mlidc_encode_cache_dir);
DECLARE_bool(mlidc_allow_delta_block);
DECLARE_bool(mlidc_sis_withdraw_frequency_partition);
DECLARE_int32(mlidc_block_codec_probes);
DECLARE_string(mlidc_dtype);

namespace facebook::nimble::mlidc {

// ---------------------------------------------------------------------------
// Per-target memory pools
// ---------------------------------------------------------------------------

// Returns a leaf pool of this target's own, so allocations are attributable
// to it rather than pooled with every other arm in the sweep. A view's index
// structures are allocated here, so they are measured rather than estimated.
inline std::shared_ptr<velox::memory::MemoryPool> makeTargetPool() {
  static std::atomic<size_t> nextId{0};
  return velox::memory::memoryManager()->addLeafPool(
      "mlidc_target_" + std::to_string(nextId++));
}

// ---------------------------------------------------------------------------
// NimbleBenchTarget<EncodingT>
// ---------------------------------------------------------------------------
// Wraps a single encode/decode cycle.  After encode() the object holds the
// serialised bytes in an internal Buffer together with a live Encoding object
// ready for decode operations.

/// Whether NimbleBenchTarget::encode() leaves the decoder to be built on
/// first use. The encode driver sets it, so decoder construction is not
/// charged to encode(); every other driver keeps the eager build.
inline bool& deferDecoderConstruction() {
  static bool defer{false};
  return defer;
}

template <typename EncodingT>
class NimbleBenchTarget {
 public:
  using T = typename EncodingT::cppDataType;

  NimbleBenchTarget() : pool_(makeTargetPool()) {}

  // Encode data.  Destroys any previously encoded state.
  void encode(
      const Vector<T>& data,
      const Encoding::Options& options = {},
      bool realNestedSelection = false) {
    Buffer buf{*pool_};
    constexpr auto kType = test::EncodingTypeTraits<EncodingT>::encodingType;
    // The key hashes every input value and two fingerprints, so it is built
    // only where a cache will read it.
    const bool caching = !cacheDir().empty();
    const auto armId = caching ? cacheArmIdentity(options, realNestedSelection)
                               : std::string{};
    const auto key = caching
        ? encodeCacheKey<T>(data.data(), data.size(), armId, kType)
        : std::string{};
    if (!caching || !loadCached(key, armId, kType, encoded_)) {
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
      if (caching) {
        storeCached(key, armId, kType, encoded_);
      }
    }
    options_ = options;
    encoding_.reset();
    if (!deferDecoderConstruction()) {
      decoder();
    }
  }

  // reset + materialize all n rows into dst.
  void materializeAll(T* dst, uint32_t n) {
    decoder().reset();
    decoder().materialize(n, dst);
  }

  // reset + skip begin rows + materialize count rows into dst.
  void materializeRange(uint32_t begin, uint32_t count, T* dst) {
    auto& encoding = decoder();
    encoding.reset();
    if (begin > 0) {
      encoding.skip(begin);
    }
    encoding.materialize(count, dst);
  }

  // Gather pattern: for each [begin, count) range in sorted order, skip then
  // materialize.  dst must have space for the total number of rows across all
  // ranges.
  void skipThenMaterialize(std::span<const nimble::RowRange> ranges, T* dst) {
    auto& encoding = decoder();
    encoding.reset();
    uint32_t cursor = 0;
    for (const auto& range : ranges) {
      const uint32_t begin = range.startRow;
      const uint32_t count = range.numRows();
      if (begin > cursor) {
        encoding.skip(begin - cursor);
        cursor = begin;
      }
      encoding.materialize(count, dst);
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
    return &decoder();
  }

  // The encoded bytes and whatever the Encoding allocated from this target's
  // own pool.
  size_t residentBytes() const {
    return encoded_.size() + static_cast<size_t>(pool_->usedBytes());
  }

  bool retainsDecodeCache() const {
    return encoding_ != nullptr && encoding_->retainsDecodeCache();
  }

  void dropDecodeCache() {
    if (encoding_ != nullptr) {
      encoding_->dropDecodeCache();
    }
  }

 private:
  Encoding& decoder() {
    if (encoding_ == nullptr) {
      // Constructed directly from the encoded bytes rather than by
      // re-encoding via createEncoding(), which would silently drop
      // realNestedSelection and produce different encoded data.
      encoding_ = std::make_unique<EncodingT>(
          *pool_,
          std::string_view(encoded_),
          benchmarks::nullFactory(),
          options_);
    }
    return *encoding_;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::string encoded_;
  Encoding::Options options_;
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
      std::span<const nimble::RowRange> ranges,
      T* dst) = 0;
  virtual size_t payloadSize() const = 0;

  /// Bytes this target holds in memory to serve reads, including the encoded
  /// payload, anything decoded and kept, and any index structure built over
  /// it. Pure, so a target that quietly held a decoded copy cannot be
  /// compared on time alone against one holding only compressed bytes.
  /// Sampled after a workload, since lazily materialising targets have no
  /// final footprint until something has read from them.
  virtual size_t residentBytes() const = 0;

  virtual std::vector<std::span<const std::byte>> internalBuffers() const = 0;

  /// How a partial read reaches its rows.
  ///
  /// Pure, so every target answers rather than a driver inferring it from
  /// the arm's name.
  virtual ReadPath readPath() const = 0;

  /// Whether reads are served from a structure this target builds once and
  /// then reuses: a view's indexed accessors, or a decoded buffer. False
  /// means every read pays the same cost, which an amortisation curve needs
  /// in order to show the flat line a cursor arm draws.
  virtual bool buildsAccessStructure() const {
    return false;
  }

  /// Builds that structure now, so a read that follows does not pay for it.
  /// Idempotent, and a no-op where there is nothing to build.
  virtual void buildAccessStructure() {}

  /// Drops it, so the next read builds it again. Paired with
  /// buildAccessStructure() this lets one run report construction cost and
  /// per-read cost separately.
  virtual void discardAccessStructure() {}

  /// Returns the encoding tree, for reporting which nested encodings a
  /// selection policy actually chose. Empty for targets that are not Nimble
  /// encodings and so have no tree to show.
  virtual std::string describe() {
    return {};
  }

  /// The same tree as describe(), one node per line and keyed by path.
  /// describe() nests children inside their parent's line, which reads well
  /// but parses badly since a node's position depends on its siblings. See
  /// tools::getEncodingTreeLabel.
  virtual std::string describeTree() {
    return {};
  }

  /// Per node, what it cost against what its selection was quoted. Dumped
  /// under default options, which is what the drivers encode with.
  virtual std::string describeNodeEstimates() {
    return {};
  }

  /// For a SubIntSplit stream, what each section and the whole value were
  /// quoted by section selection and by the split planner. Empty otherwise.
  virtual std::string describeSectionChoices() {
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
  void skipThenMaterialize(std::span<const nimble::RowRange> ranges, T* dst)
      override {
    target.skipThenMaterialize(ranges, dst);
  }
  size_t payloadSize() const override {
    return target.payloadSize();
  }
  std::vector<std::span<const std::byte>> internalBuffers() const override {
    return target.internalBuffers();
  }
  // An Encoding carries a cursor, so materializeRange reaches row i by
  // resetting and skipping i rows.
  ReadPath readPath() const override {
    return ReadPath::kCursor;
  }

  size_t residentBytes() const override {
    return target.residentBytes();
  }

  // Usually nothing is built once and reused, and this is false. The
  // exception is a transformed SubIntSplit plan, whose first probe decodes
  // the whole column into a cache that later probes copy out of, giving the
  // arm a one-time build despite its sequential interface.
  bool buildsAccessStructure() const override {
    return target.retainsDecodeCache();
  }

  // Forces that first decode now. A one-row read is what populates the cache,
  // and for an unblocked transformed plan the span it decodes is the whole
  // column, so this is the cost a first probe pays.
  void buildAccessStructure() override {
    if (!target.retainsDecodeCache()) {
      return;
    }
    T value{};
    target.materializeRange(0, 1, &value);
  }

  void discardAccessStructure() override {
    target.dropDecodeCache();
  }
  std::string describe() override {
    auto* encoding = target.encoding();
    return encoding != nullptr ? encoding->debugString(0) : std::string{};
  }

  std::string describeTree() override {
    const auto payload = target.payloadBytes();
    if (payload.empty()) {
      return {};
    }
    return nimble::tools::getEncodingTreeLabel(
        std::string_view(
            reinterpret_cast<const char*>(payload.data()), payload.size()));
  }

  std::string describeNodeEstimates() override {
    const auto payload = target.payloadBytes();
    if (payload.empty()) {
      return {};
    }
    return describeEncodingNodeEstimates(
        std::string_view(
            reinterpret_cast<const char*>(payload.data()), payload.size()),
        *benchmarks::benchmarkPool(),
        Encoding::Options{});
  }

  std::string describeSectionChoices() override {
    const auto payload = target.payloadBytes();
    if (payload.empty()) {
      return {};
    }
    return describeSubIntSplitSectionChoices(
        std::string_view(
            reinterpret_cast<const char*>(payload.data()), payload.size()),
        *benchmarks::benchmarkPool());
  }
};

// ---------------------------------------------------------------------------
// NimbleViewBenchTargetImpl<EncodingT>
// ---------------------------------------------------------------------------
// Encodes exactly as NimbleBenchTargetImpl does, then reads through an
// EncodingView instead of an Encoding. The two differ only in how a read is
// addressed: sequential cursor vs. index. Pairing each view entry with its
// sequential twin makes that the only variable between them.
template <typename EncodingT>
class NimbleViewBenchTargetImpl
    : public NimbleBenchTargetBase<typename EncodingT::cppDataType> {
 public:
  using T = typename EncodingT::cppDataType;

  // True for the same reason makeEncoderEntry defaults to it: a composite
  // encoding is nothing but its sub-streams, and writing them Trivial reports
  // the encoding at its worst.
  void encode(const Vector<T>& data, const Encoding::Options& opts) override {
    encodeWith(data, opts, /*realNestedSelection=*/true);
  }

  void encodeWith(
      const Vector<T>& data,
      const Encoding::Options& opts,
      bool realNestedSelection) {
    Buffer buf{*pool_};
    constexpr auto kType = test::EncodingTypeTraits<EncodingT>::encodingType;
    // Keyed only where a cache will read the key; see NimbleBenchTarget.
    const bool caching = !cacheDir().empty();
    const auto armId =
        caching ? cacheArmIdentity(opts, realNestedSelection) : std::string{};
    const auto key = caching
        ? encodeCacheKey<T>(data.data(), data.size(), armId, kType)
        : std::string{};
    if (!caching || !loadCached(key, armId, kType, encoded_)) {
      encoded_ = std::string(
          encodeWithCompression<EncodingT, T>(
              buf,
              data,
              parseCompressionType(FLAGS_mlidc_substream_compression),
              opts,
              realNestedSelection));
      if (caching) {
        storeCached(key, armId, kType, encoded_);
      }
    }
    options_ = opts;
    view_ = createEncodingView(std::string_view(encoded_), pool_.get(), opts);
    NIMBLE_CHECK_NOT_NULL(view_);
  }

  ReadPath readPath() const override {
    return ReadPath::kIndexed;
  }

  // Reading through a view is two costs, not one. Where a section's encoding
  // has no real view -- FrequencyPartition and Delta are the two that matter
  // here -- the fallback MaterializedEncodingView decodes that whole section in
  // its constructor, so a read that finds the view already built is reporting a
  // partial decode. Separating the two is what lets one measurement report both
  // numbers, rather than a second arm reporting the other one.
  bool buildsAccessStructure() const override {
    return true;
  }

  void buildAccessStructure() override {
    if (view_ != nullptr) {
      return;
    }
    view_ =
        createEncodingView(std::string_view(encoded_), pool_.get(), options_);
    NIMBLE_CHECK_NOT_NULL(view_);
  }

  void discardAccessStructure() override {
    view_.reset();
  }

  // Rebuilds the view inside every materializeAll, so a +view+ctor arm reports
  // the construction and the decode as one figure, like the cursor path.
  void setTimeViewConstruction(bool value) {
    timeViewConstruction_ = value;
  }

  void materializeAll(T* dst, uint32_t n) override {
    if (timeViewConstruction_) {
      discardAccessStructure();
    }
    buildAccessStructure();
    view_->read(0, n, dst);
  }

  // A single-row read goes through readAt rather than a length-1 range: that
  // is the API a point lookup would actually use.
  void materializeRange(uint32_t begin, uint32_t count, T* dst) override {
    buildAccessStructure();
    if (count == 1) {
      view_->readAt(begin, dst);
    } else {
      view_->read(begin, count, dst);
    }
  }

  // No cursor, so no skip: each range is resolved from its own index. The
  // whole list goes to the view in one call, so it can plan across ranges
  // rather than answer them one by one.
  void skipThenMaterialize(std::span<const nimble::RowRange> ranges, T* dst)
      override {
    buildAccessStructure();
    view_->readRanges(ranges, dst);
  }

  size_t payloadSize() const override {
    return encoded_.size();
  }

  // The encoded bytes plus whatever the view allocated from this target's own
  // pool: a view's index structures, and any viewless section's decode
  // buffer, are pool-backed and so measured here rather than estimated.
  size_t residentBytes() const override {
    return encoded_.size() + static_cast<size_t>(pool_->usedBytes());
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

  // Unlike describe(), this needs no Encoding at all: the tree is read
  // straight off the encoded bytes this target already holds.
  std::string describeTree() override {
    return encoded_.empty()
        ? std::string{}
        : nimble::tools::getEncodingTreeLabel(std::string_view(encoded_));
  }

  std::string describeNodeEstimates() override {
    return encoded_.empty() ? std::string{}
                            : describeEncodingNodeEstimates(
                                  std::string_view(encoded_), *pool_, options_);
  }

 private:
  std::shared_ptr<velox::memory::MemoryPool> pool_{makeTargetPool()};
  std::string encoded_;
  Encoding::Options options_;
  std::unique_ptr<EncodingView> view_;
  bool timeViewConstruction_{false};
};

template <typename T>
struct EncoderEntry {
  std::string name;
  std::string family; // "baseline", "sis-manual", "sis-auto", "fpe-index"
  std::string variant;
  // Which cost-model inventory SubIntSplit was allowed to choose from. Only
  // meaningful for the SubIntSplit family; empty elsewhere.
  std::string inventory;
  // Section transform applied, empty when none. The ablation arm.
  std::string transform;
  bool isSequential{true};
  bool fastSkip{false};
  bool randomAccess{false};
  // Whether a read must first decompress the entire payload is deliberately
  // not a field here: drivers ask the target through
  // NimbleBenchTargetBase::readPath() instead.

  // Factory: construct a fresh target and encode the given data.
  std::function<std::unique_ptr<NimbleBenchTargetBase<T>>(
      const Vector<T>&,
      const Encoding::Options&)>
      factory;
};

// Convenience builder for a concrete EncodingT.
//
// realNestedSelection defaults to true because that is what a writer does.
// An encoding built from sub-streams -- Dictionary's alphabet and indices,
// RLE's values and lengths -- is nothing but those sub-streams, and with
// selection off they are written Trivial, reporting the encoding at its
// worst rather than as anyone would ship it. Encodings with no sub-streams
// encode byte-identically either way, so this is a no-op for the flat ones.
template <typename EncodingT>
EncoderEntry<typename EncodingT::cppDataType> makeEncoderEntry(
    std::string name,
    std::string family,
    std::string variant,
    bool isSequential = true,
    bool fastSkip = false,
    bool randomAccess = false,
    bool realNestedSelection = true) {
  using T = typename EncodingT::cppDataType;
  EncoderEntry<T> entry;
  entry.name = std::move(name);
  entry.family = std::move(family);
  entry.variant = std::move(variant);
  entry.isSequential = isSequential;
  entry.fastSkip = fastSkip;
  entry.randomAccess = randomAccess;
  entry.factory = [realNestedSelection](
                      const Vector<T>& data, const Encoding::Options& opts) {
    auto impl = std::make_unique<NimbleBenchTargetImpl<EncodingT>>();
    impl->target.encode(data, opts, realNestedSelection);
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

  void skipThenMaterialize(std::span<const nimble::RowRange> ranges, T* dst)
      override {
    decompressAll();
    inner_->skipThenMaterialize(ranges, dst);
  }

  // Whatever the inner encoding could do, a read here decompresses the whole
  // payload first, which is the erasure this arm exists to measure.
  ReadPath readPath() const override {
    return ReadPath::kWholePayload;
  }

  // The stored size, which is what an outer codec is chosen for.
  size_t payloadSize() const override {
    return compressed_.size();
  }

  // The compressed payload, whatever the inner target still holds, and the
  // buffer the last read decompressed into: all three are resident at once
  // while a read is being served.
  size_t residentBytes() const override {
    return compressed_.size() + inner_->residentBytes() +
        (lastDecompressed_ != nullptr
             ? static_cast<size_t>(lastDecompressed_->capacity())
             : 0);
  }

  std::vector<std::span<const std::byte>> internalBuffers() const override {
    return {
        {reinterpret_cast<const std::byte*>(compressed_.data()),
         compressed_.size()}};
  }

  std::string describeTree() override {
    return inner_->describeTree();
  }

  std::string describeNodeEstimates() override {
    return inner_->describeNodeEstimates();
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

  std::shared_ptr<velox::memory::MemoryPool> pool_{makeTargetPool()};
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
// Materialise on first access
// ---------------------------------------------------------------------------

// Decodes the column once on the first access and serves everything after it
// from the decoded buffer.
//
// A blackbox codec has no addressable interior, so OpenZLBenchTarget
// decompresses the whole column on every probe. A reader that expects to
// come back can instead decompress once and serve the rest from memory; this
// decorator is that reader, putting a blackbox codec on the same
// amortisation curve as a view. The structure is MaterializedEncodingView's,
// which does exactly this for a section whose own encoding has no view.
template <typename T>
class MaterializingTarget : public NimbleBenchTargetBase<T> {
 public:
  // Takes an inner target the encoder entry's factory has already encoded, and
  // the row count to decode, which the factory reads off the input column.
  MaterializingTarget(
      std::unique_ptr<NimbleBenchTargetBase<T>> inner,
      uint32_t rowCount)
      : inner_{std::move(inner)}, rowCount_{rowCount} {}

  void encode(const Vector<T>& data, const Encoding::Options& opts) override {
    inner_->encode(data, opts);
    rowCount_ = data.size();
    discardAccessStructure();
  }

  void materializeAll(T* dst, uint32_t n) override {
    buildAccessStructure();
    std::copy_n(values_.data(), n, dst);
  }

  void materializeRange(uint32_t begin, uint32_t count, T* dst) override {
    buildAccessStructure();
    std::copy_n(values_.data() + begin, count, dst);
  }

  void skipThenMaterialize(std::span<const nimble::RowRange> ranges, T* dst)
      override {
    buildAccessStructure();
    for (const auto& range : ranges) {
      std::copy_n(values_.data() + range.startRow, range.numRows(), dst);
      dst += range.numRows();
    }
  }

  // Indexed once the buffer exists. What it costs to get there is the build
  // cost, reported separately rather than folded in.
  ReadPath readPath() const override {
    return ReadPath::kIndexed;
  }

  bool buildsAccessStructure() const override {
    return true;
  }

  void buildAccessStructure() override {
    if (rowCount_ == 0 || !values_.empty()) {
      return;
    }
    values_.resize(rowCount_);
    inner_->materializeAll(values_.data(), rowCount_);
    ++numBuilds_;
  }

  // Frees the buffer rather than marking it stale, because a genuine first
  // access allocates it. Holding the allocation across a discard would make
  // every rebuild cheaper than the one a reader actually pays for.
  void discardAccessStructure() override {
    values_ = std::vector<T>{};
  }

  /// Times the inner target has been decoded since construction.
  /// Instrumentation for tests pinning "decoded once, not once per read".
  size_t numBuilds() const {
    return numBuilds_;
  }

  // The inner codec's stored bytes. Materialising changes what a read costs,
  // not what the column occupies.
  size_t payloadSize() const override {
    return inner_->payloadSize();
  }

  // What it occupies on disk is payloadSize(); what it occupies in memory is
  // this, and for a materialised arm the two differ by a whole decoded
  // column.
  size_t residentBytes() const override {
    return inner_->residentBytes() + values_.capacity() * sizeof(T);
  }

  std::vector<std::span<const std::byte>> internalBuffers() const override {
    return inner_->internalBuffers();
  }

  std::string describe() override {
    return inner_->describe();
  }

  std::string describeTree() override {
    return inner_->describeTree();
  }

  std::string describeNodeEstimates() override {
    return inner_->describeNodeEstimates();
  }

 private:
  std::unique_ptr<NimbleBenchTargetBase<T>> inner_;
  uint32_t rowCount_{0};
  std::vector<T> values_;
  size_t numBuilds_{0};
};

// Wraps entry's factory so its target decodes once and serves reads from the
// decoded buffer. Named "+materialize" for the reason "+view" is: the pair
// encodes to the same bytes and differs only in how a read is addressed.
template <typename T>
EncoderEntry<T> withMaterializedAccess(EncoderEntry<T> entry) {
  entry.name += "+materialize";
  entry.variant += "_materialize";
  entry.isSequential = false;
  entry.fastSkip = true;
  entry.randomAccess = true;
  auto inner = std::move(entry.factory);
  entry.factory = [inner = std::move(inner)](
                      const Vector<T>& data, const Encoding::Options& opts) {
    auto target = std::make_unique<MaterializingTarget<T>>(
        inner(data, opts), static_cast<uint32_t>(data.size()));
    // The inner factory has already encoded; calling encode() here would
    // encode a second time.
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
// Presents the column in the arrival order the run asked for. The order is
// applied once, to the input, before anything is encoded, so every encoder
// sees the same rows in the same order.
template <typename T>
Vector<T> applyInputOrder(Vector<T> data) {
  const auto order = mlidc::parseInputOrder(FLAGS_mlidc_input_order);
  if (order.kind.empty() || order.kind == "shipped") {
    return data;
  }

  std::vector<uint64_t> values(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    uint64_t bits = 0;
    __builtin_memcpy(&bits, &data[i], sizeof(T));
    values[i] = bits;
  }

  // mergekey partitions by a field the value already carries, so the key is
  // a real section of the real split rather than an arbitrary bit range.
  std::function<uint64_t(uint32_t)> keyOf = [](uint32_t) {
    return uint64_t{0};
  };
  if (order.kind == "mergekey") {
    const auto plan = ::facebook::nimble::subintsplit::selectSplits(
        values, static_cast<int>(sizeof(T) * 8), values.size());
    NIMBLE_CHECK(!plan.sections.empty(), "Split selection found no sections.");
    const size_t which = std::min<size_t>(
        static_cast<size_t>(std::max(0, order.param)),
        plan.sections.size() - 1);
    const int bitStart = plan.sections[which].bitStart;
    const int width =
        plan.sections[which].bitEnd - plan.sections[which].bitStart + 1;
    const uint64_t mask =
        (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
    const std::vector<uint64_t>* source = &values;
    keyOf = [source, bitStart, mask](uint32_t row) -> uint64_t {
      return ((*source)[row] >> bitStart) & mask;
    };
  }

  const auto rows = mlidc::buildInputOrder(
      order, values, keyOf, static_cast<uint64_t>(FLAGS_mlidc_seed));

  auto& pool = benchmarks::benchmarkPool();
  Vector<T> reordered{pool.get()};
  reordered.resize(data.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    reordered[i] = data[rows[i]];
  }
  return reordered;
}

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
                     auto data =
                         detail::loadColumnLines<T>(FLAGS_mlidc_file, n);
                     return detail::applyInputOrder<T>(std::move(data));
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
  // Top value plus exceptions: a dominant value, a boolean vector marking the
  // rows that hold it, and a child encoding for the rest. Corresponds to
  // BtrBlocks' FREQUENCY64 and FastLanes' frequency. FPE is not this
  // encoding -- it partitions into frequency tiers.
  encoders.push_back(
      makeEncoderEntry<MainlyConstantEncoding<T>>(
          "MainlyConstant", "Baseline", "mainly_constant", true, false, false));
  // Huffman has no top-level arm of its own, only the SIS/huffOn and huffOff
  // planner flag.
  encoders.push_back(
      makeEncoderEntry<HuffmanEncoding<T>>(
          "Huffman", "Baseline", "huffman", true, false, false));

  // Read-path variants. Each encodes byte-for-byte identically to the entry
  // it shadows and differs only in reading by index rather than by cursor,
  // isolating what indexed access is worth.
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

  {
    EncoderEntry<T> entry;
    entry.name = "FixedBitWidth/view";
    entry.family = "Baseline";
    entry.variant = "fbw_view";
    entry.isSequential = false;
    entry.fastSkip = true;
    entry.randomAccess = true;
    entry.factory = [](const Vector<T>& data, const Encoding::Options& opts) {
      auto impl = std::make_unique<
          NimbleViewBenchTargetImpl<FixedBitWidthEncoding<T>>>();
      impl->encode(data, opts);
      return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
    };
    encoders.push_back(std::move(entry));
  }

  {
    EncoderEntry<T> entry;
    entry.name = "MainlyConstant/view";
    entry.family = "Baseline";
    entry.variant = "mainly_constant_view";
    entry.isSequential = false;
    entry.fastSkip = true;
    entry.randomAccess = true;
    entry.factory = [](const Vector<T>& data, const Encoding::Options& opts) {
      auto impl = std::make_unique<
          NimbleViewBenchTargetImpl<MainlyConstantEncoding<T>>>();
      impl->encode(data, opts);
      return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
    };
    encoders.push_back(std::move(entry));
  }

  {
    EncoderEntry<T> entry;
    entry.name = "Dictionary/view";
    entry.family = "Baseline";
    entry.variant = "dict_view";
    entry.isSequential = false;
    entry.fastSkip = true;
    entry.randomAccess = true;
    entry.factory = [](const Vector<T>& data, const Encoding::Options& opts) {
      auto impl =
          std::make_unique<NimbleViewBenchTargetImpl<DictionaryEncoding<T>>>();
      impl->encode(data, opts);
      return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
    };
    encoders.push_back(std::move(entry));
  }

  // PFOR and SimdForBitpack are integral-only encodings.
  if constexpr (std::is_integral_v<T>) {
    {
      EncoderEntry<T> entry;
      entry.name = "PFOR/view";
      entry.family = "Baseline";
      entry.variant = "pfor_view";
      entry.isSequential = false;
      entry.fastSkip = true;
      entry.randomAccess = true;
      entry.factory = [](const Vector<T>& data, const Encoding::Options& opts) {
        auto impl =
            std::make_unique<NimbleViewBenchTargetImpl<PFOREncoding<T>>>();
        impl->encode(data, opts);
        return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
      };
      encoders.push_back(std::move(entry));
    }

    {
      EncoderEntry<T> entry;
      entry.name = "SimdForBitpack/view";
      entry.family = "Baseline";
      entry.variant = "simdfor_view";
      entry.isSequential = false;
      entry.fastSkip = true;
      entry.randomAccess = true;
      entry.factory = [](const Vector<T>& data, const Encoding::Options& opts) {
        auto impl = std::make_unique<
            NimbleViewBenchTargetImpl<SimdForBitpackEncoding<T>>>();
        impl->encode(data, opts);
        return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
      };
      encoders.push_back(std::move(entry));
    }
  }

  // All four index types are carried. TierTagArray is the cheapest correct
  // index where frequency partitioning does real work rather than
  // degenerating to a dictionary, though it is still not usable as a default
  // on speed: materializeImpl is a per-row random-access loop for every
  // indexed mode, and tagtag rescans up to kRankSampleStride tags per row
  // where a bitmap reads one superblock and a few popcounts.
  //
  // fpe_noindex is not a candidate at all. It materializes in tier-reordered
  // space -- it encodes the multiset, not the sequence -- which is why every
  // driver skips its validation. It is carried only as the floor the correct
  // modes are paying above, and it must never enter a comparison as an
  // option.
  //
  // The index type is carried explicitly rather than derived from the loop
  // position, since FreqPartIndexType's own ordering does not match the
  // surviving name list.
  const std::array<std::string, 4> fpeNames = {
      "fpe_noindex", "fpe_pertier", "fpe_tagtag", "fpe_elias"};
  const std::array<uint8_t, 4> fpeIndexType = {0, 1, 2, 3};
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
    const uint8_t indexType = fpeIndexType[idx];
    entry.factory = [indexType](
                        const Vector<T>& data, const Encoding::Options& opts) {
      auto impl = std::make_unique<
          NimbleBenchTargetImpl<FrequencyPartitionEncoding<T>>>();
      Encoding::Options o = opts;
      o.frequencyPartitionIndex = indexType;
      // Real nested selection, for the same reason the baseline arms need
      // it: FrequencyPartition is its per-tier key arrays and tier
      // dictionaries, and with selection off those are written Trivial, so
      // tiering cannot pay by construction.
      impl->target.encode(data, o, /*realNestedSelection=*/true);
      return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
    };
    encoders.push_back(std::move(entry));
  }

  {
    // Same as FPE/fpe_tagtag, with
    // Options::frequencyPartitionResolveTierValues set: TierTagArray
    // resolves indices[rank] -> dictionary index into a per-tier
    // rank -> value table at decode construction, so decode reads it
    // directly instead of chasing dictionary[indices[rank]].
    EncoderEntry<T> entry;
    entry.name = "FPE/fpe_tagtag_resolved";
    entry.family = "FrequencyPartition";
    entry.variant = "fpe_tagtag_resolved";
    entry.isSequential = true;
    entry.fastSkip = true;
    entry.randomAccess = true;
    entry.factory = [](const Vector<T>& data, const Encoding::Options& opts) {
      auto impl = std::make_unique<
          NimbleBenchTargetImpl<FrequencyPartitionEncoding<T>>>();
      Encoding::Options o = opts;
      o.frequencyPartitionIndex =
          static_cast<uint8_t>(FreqPartIndexType::TierTagArray);
      o.frequencyPartitionResolveTierValues = true;
      impl->target.encode(data, o, /*realNestedSelection=*/true);
      return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
    };
    encoders.push_back(std::move(entry));
  }

  {
    EncoderEntry<T> entry;
    entry.name = "SIS/realNested";
    entry.family = "SubIntSplit";
    entry.variant = "real_nested";
    entry.inventory = "full";
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
    // The +view arms above build the view in encodeWith, outside the timed
    // region. Where a section's encoding has no real view -- FrequencyPartition
    // and Delta are the two that matter here -- the fallback
    // MaterializedEncodingView decodes that whole section in its constructor,
    // so those arms report a partial decode. This pair repeats them with
    // construction inside the measurement, the like-for-like number against
    // the cursor arms.
    EncoderEntry<T> entry;
    entry.name = "SIS/realNested+view+ctor";
    entry.family = "SubIntSplit";
    entry.variant = "real_nested_view_ctor";
    entry.inventory = "full";
    entry.isSequential = false;
    entry.fastSkip = true;
    entry.randomAccess = true;
    entry.factory = [](const Vector<T>& data, const Encoding::Options& opts) {
      auto impl =
          std::make_unique<NimbleViewBenchTargetImpl<SubIntSplitEncoding<T>>>();
      impl->encodeWith(data, opts, /*realNestedSelection=*/true);
      impl->setTimeViewConstruction(true);
      return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
    };
    encoders.push_back(std::move(entry));
  }

  {
    // SIS/hybrid+view with construction timed, the like-for-like arm for plans
    // priced on a read path that pays for opening the stream.
    EncoderEntry<T> entry;
    entry.name = "SIS/hybrid+view+ctor";
    entry.family = "SubIntSplit";
    entry.variant = "hybrid_view_ctor";
    entry.inventory = "full";
    entry.isSequential = false;
    entry.fastSkip = true;
    entry.randomAccess = true;
    entry.factory = [](const Vector<T>& data, const Encoding::Options& opts) {
      Encoding::Options hybridOptions = opts;
      hybridOptions.subIntSplitHybridPlanner = true;
      auto impl =
          std::make_unique<NimbleViewBenchTargetImpl<SubIntSplitEncoding<T>>>();
      impl->encodeWith(data, hybridOptions, /*realNestedSelection=*/true);
      impl->setTimeViewConstruction(true);
      return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
    };
    encoders.push_back(std::move(entry));
  }

  {
    EncoderEntry<T> entry;
    entry.name = "SIS/key_derived+view+ctor";
    entry.family = "SubIntSplit";
    entry.variant = "key_derived_view_ctor";
    entry.inventory = "full";
    entry.isSequential = false;
    entry.fastSkip = true;
    entry.randomAccess = true;
    entry.factory = [](const Vector<T>& data, const Encoding::Options& opts) {
      auto impl =
          std::make_unique<NimbleViewBenchTargetImpl<SubIntSplitEncoding<T>>>();
      Encoding::Options o = opts;
      o.subIntSplitTransform =
          static_cast<uint8_t>(subintsplit::TransformId::KeyDerived);
      o.subIntSplitKeySection = 0xFF;
      impl->encodeWith(data, o, /*realNestedSelection=*/true);
      impl->setTimeViewConstruction(true);
      return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
    };
    encoders.push_back(std::move(entry));
  }

  {
    // Same plan as SIS/realNested, with
    // Options::frequencyPartitionResolveTierValues set. SubIntSplit's
    // sectionEncodingOptions copies the caller's options, so the flag reaches
    // every section: a FrequencyPartition section then resolves
    // indices[rank] -> dictionary index into a per-tier rank -> value table
    // at decode construction, and the read does one load where it did two.
    // Payload is identical to the arm above, since the table is never
    // serialised; the cost is memory.
    EncoderEntry<T> entry;
    entry.name = "SIS/realNested+resolved";
    entry.family = "SubIntSplit";
    entry.variant = "real_nested_resolved";
    entry.inventory = "full";
    entry.isSequential = false;
    entry.fastSkip = false;
    entry.randomAccess = false;
    entry.factory = [](const Vector<T>& data, const Encoding::Options& opts) {
      auto impl =
          std::make_unique<NimbleBenchTargetImpl<SubIntSplitEncoding<T>>>();
      Encoding::Options o = opts;
      o.frequencyPartitionResolveTierValues = true;
      impl->target.encode(data, o, /*realNestedSelection=*/true);
      return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
    };
    encoders.push_back(std::move(entry));
  }

  {
    // The transformed counterpart of the arm above, asking the same question
    // of the key-derived path.
    EncoderEntry<T> entry;
    entry.name = "SIS/key_derived+resolved";
    entry.family = "SubIntSplit";
    entry.variant = "key_derived_resolved";
    entry.inventory = "full";
    entry.isSequential = false;
    entry.fastSkip = false;
    entry.randomAccess = false;
    entry.factory = [](const Vector<T>& data, const Encoding::Options& opts) {
      auto impl =
          std::make_unique<NimbleBenchTargetImpl<SubIntSplitEncoding<T>>>();
      Encoding::Options o = opts;
      o.frequencyPartitionResolveTierValues = true;
      o.subIntSplitTransform =
          static_cast<uint8_t>(subintsplit::TransformId::KeyDerived);
      // 0xFF: let the encoder find the section worth keying on rather than
      // assert one, matching the generated key_derived arms.
      o.subIntSplitKeySection = 0xFF;
      impl->target.encode(data, o, /*realNestedSelection=*/true);
      return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
    };
    encoders.push_back(std::move(entry));
  }

  {
    EncoderEntry<T> entry;
    entry.name = "SIS/realNested+view";
    entry.family = "SubIntSplit";
    entry.variant = "real_nested_view";
    entry.inventory = "full";
    entry.isSequential = false;
    entry.fastSkip = true;
    entry.randomAccess = true;
    entry.factory = [](const Vector<T>& data, const Encoding::Options& opts) {
      auto impl =
          std::make_unique<NimbleViewBenchTargetImpl<SubIntSplitEncoding<T>>>();
      impl->encodeWith(data, opts, /*realNestedSelection=*/true);
      return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
    };
    encoders.push_back(std::move(entry));
  }

  // SIS/realNested and its view, with split boundaries from the hybrid planner
  // (Encoding::Options::subIntSplitHybridPlanner) instead of the DP's argmin.
  // Everything else is the realNested arm, so the pair differs only by plan.
  for (const bool view : {false, true}) {
    EncoderEntry<T> entry;
    entry.name = view ? "SIS/hybrid+view" : "SIS/hybrid";
    entry.family = "SubIntSplit";
    entry.variant = view ? "hybrid_view" : "hybrid";
    entry.inventory = "full";
    entry.isSequential = false;
    entry.fastSkip = view;
    entry.randomAccess = view;
    entry.factory = [view](
                        const Vector<T>& data, const Encoding::Options& opts) {
      Encoding::Options hybridOptions = opts;
      hybridOptions.subIntSplitHybridPlanner = true;
      if (view) {
        auto impl = std::make_unique<
            NimbleViewBenchTargetImpl<SubIntSplitEncoding<T>>>();
        impl->encodeWith(data, hybridOptions, /*realNestedSelection=*/true);
        return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
      }
      auto impl =
          std::make_unique<NimbleBenchTargetImpl<SubIntSplitEncoding<T>>>();
      impl->target.encode(data, hybridOptions, /*realNestedSelection=*/true);
      return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
    };
    encoders.push_back(std::move(entry));
  }

  // One entry per transform, so the ablation is an encoder row rather than a
  // new loop in every driver: bulk, gather and point pick these up unchanged.
  {
    const std::vector<std::pair<subintsplit::TransformId, const char*>> arms{
        {subintsplit::TransformId::KeyDerived, "key_derived"},
    };
    for (const auto& [transformId, name] : arms) {
      const auto rawId = static_cast<uint8_t>(transformId);

      {
        EncoderEntry<T> entry;
        entry.name = std::string("SIS/") + name;
        entry.family = "SubIntSplit";
        entry.variant = "real_nested";
        entry.inventory = "full";
        entry.transform = name;
        entry.isSequential = false;
        entry.fastSkip = false;
        entry.randomAccess = false;
        entry.factory = [rawId](
                            const Vector<T>& data,
                            const Encoding::Options& opts) {
          Encoding::Options o = opts;
          o.subIntSplitTransform = rawId;
          // 0xFF: let the encoder find the section worth keying on rather
          // than assert one, since that is a property of the column.
          o.subIntSplitKeySection = 0xFF;
          auto impl =
              std::make_unique<NimbleBenchTargetImpl<SubIntSplitEncoding<T>>>();
          impl->target.encode(data, o, /*realNestedSelection=*/true);
          return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
        };
        encoders.push_back(std::move(entry));
      }

      {
        EncoderEntry<T> entry;
        entry.name = std::string("SIS/") + name + "+view";
        entry.family = "SubIntSplit";
        entry.variant = "real_nested_view";
        entry.inventory = "full";
        entry.transform = name;
        entry.isSequential = false;
        entry.fastSkip = true;
        entry.randomAccess = true;
        entry.factory =
            [rawId](const Vector<T>& data, const Encoding::Options& opts) {
              Encoding::Options o = opts;
              o.subIntSplitTransform = rawId;
              // 0xFF: let the encoder find the section worth keying on rather
              // than assert one, since that is a property of the column.
              o.subIntSplitKeySection = 0xFF;
              auto impl = std::make_unique<
                  NimbleViewBenchTargetImpl<SubIntSplitEncoding<T>>>();
              impl->encodeWith(data, o, /*realNestedSelection=*/true);
              return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
            };
        encoders.push_back(std::move(entry));
      }
    }
  }

  // The transform arms above each pin one transform for the whole column.
  // This pair instead lets the encoder pick per section, which is what
  // production would run. Priced against SIS/realNested, the same encoder
  // with the search switched off.
  {
    // SIS/hybrid_auto and its view take their split boundaries from the
    // hybrid planner and then run the same per-section transform search.
    struct AutoArm {
      const char* name;
      bool withView;
      bool hybrid;
    };
    const std::vector<AutoArm> arms{
        {"SIS/auto", false, false},
        {"SIS/auto+view", true, false},
        {"SIS/hybrid_auto", false, true},
        {"SIS/hybrid_auto+view", true, true},
    };
    for (const auto& [name, withView, hybrid] : arms) {
      EncoderEntry<T> entry;
      entry.name = name;
      entry.family = "SubIntSplit";
      entry.variant = hybrid ? (withView ? "hybrid_view" : "hybrid")
                             : (withView ? "real_nested_view" : "real_nested");
      entry.inventory = "full";
      entry.transform = "auto";
      entry.isSequential = false;
      entry.fastSkip = withView;
      entry.randomAccess = withView;
      entry.factory = [withView, hybrid](
                          const Vector<T>& data,
                          const Encoding::Options& opts) {
        Encoding::Options o = opts;
        o.subIntSplitAutoTransform = true;
        o.subIntSplitKeySection = 0xFF;
        o.subIntSplitHybridPlanner = hybrid;
        if (withView) {
          auto impl = std::make_unique<
              NimbleViewBenchTargetImpl<SubIntSplitEncoding<T>>>();
          impl->encodeWith(data, o, /*realNestedSelection=*/true);
          return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
        }
        auto impl =
            std::make_unique<NimbleBenchTargetImpl<SubIntSplitEncoding<T>>>();
        impl->target.encode(data, o, /*realNestedSelection=*/true);
        return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
      };
      encoders.push_back(std::move(entry));
    }
  }

  // Whether SubIntSplit's planner may cost a bit range as Huffman. Off is
  // now the default, so huffOn is the arm that reproduces the older
  // behaviour. Huffman itself is never chosen for a section here; what
  // moves is where the split boundaries fall, since the planner costs
  // Huffman when scoring a bit range and the DP minimises over those costs.
  //
  // Names avoid commas: the --mlidc_encoders filter splits its list on them.
  {
    const std::vector<std::pair<uint8_t, const char*>> transformArms{
        {0, ""},
        {static_cast<uint8_t>(subintsplit::TransformId::KeyDerived),
         "key_derived"},
    };
    // Only huffOn. The huffOff arms encoded byte-identically to the
    // corresponding non-huffman arms already in the table, so only huffOn
    // is a real ablation of what costing Huffman does to split boundaries.
    for (const bool allowHuffman : {true}) {
      for (const auto& transformArm : transformArms) {
        const uint8_t rawId = transformArm.first;
        const std::string transformName = transformArm.second;
        for (const bool view : {false, true}) {
          EncoderEntry<T> entry;
          std::string name =
              std::string("SIS/") + (allowHuffman ? "huffOn" : "huffOff");
          if (!transformName.empty()) {
            name += "/" + transformName;
          }
          if (view) {
            name += "+view";
          }
          entry.name = std::move(name);
          entry.family = "SubIntSplit";
          entry.variant = view ? "real_nested_view" : "real_nested";
          // DeltaBlock is withdrawn by default too, so "full" would name an
          // inventory no arm here actually runs unless the flag is set.
          entry.inventory = FLAGS_mlidc_allow_delta_block
              ? (allowHuffman ? "full" : "no_huffman")
              : (allowHuffman ? "no_delta_block" : "no_huffman_no_delta_block");
          entry.transform = transformName;
          entry.isSequential = false;
          entry.fastSkip = view;
          entry.randomAccess = view;
          entry.factory = [allowHuffman, rawId, view](
                              const Vector<T>& data,
                              const Encoding::Options& opts) {
            Encoding::Options o = opts;
            o.subIntSplitAllowHuffman = allowHuffman;
            o.subIntSplitAllowDeltaBlock = FLAGS_mlidc_allow_delta_block;
            if (rawId != 0) {
              o.subIntSplitTransform = rawId;
              // 0xFF: let the encoder find the section worth keying on
              // rather than assert one, as the transform arms above do.
              o.subIntSplitKeySection = 0xFF;
            }
            if (view) {
              auto impl = std::make_unique<
                  NimbleViewBenchTargetImpl<SubIntSplitEncoding<T>>>();
              impl->encodeWith(data, o, /*realNestedSelection=*/true);
              return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
            }
            auto impl = std::make_unique<
                NimbleBenchTargetImpl<SubIntSplitEncoding<T>>>();
            impl->target.encode(data, o, /*realNestedSelection=*/true);
            return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(impl));
          };
          encoders.push_back(std::move(entry));
        }
      }
    }
  }

  // The whitebox baselines are pure: none of their nested streams may be
  // SubIntSplit, so a baseline's bytes are that encoding's and any gain from
  // nesting SubIntSplit is charged to SubIntSplit. The writer's own
  // selection does nest it, though, so the "+nestedSIS" arms keep the
  // writer's candidates for encodings where nesting has been seen to matter.
  {
    const std::vector<std::string> pureArms{
        "Trivial",
        "FixedBitWidth",
        "Dictionary",
        "RLE",
        "MainlyConstant",
        "Huffman",
        "RLE/view",
        "FixedBitWidth/view",
        "MainlyConstant/view",
        "Dictionary/view",
        "PFOR/view",
        "SimdForBitpack/view",
        "FPE/fpe_noindex",
        "FPE/fpe_pertier",
        "FPE/fpe_tagtag",
        "FPE/fpe_elias",
        "FPE/fpe_tagtag_resolved",
    };
    const std::vector<std::string> nestedArms{
        "RLE", "Dictionary", "MainlyConstant", "FPE/fpe_pertier"};
    const auto withNestedStreams = [](EncoderEntry<T> entry, bool allowed) {
      entry.factory = [factory = std::move(entry.factory), allowed](
                          const Vector<T>& data,
                          const Encoding::Options& opts) {
        Encoding::Options o = opts;
        o.subIntSplitInNestedStreams = allowed;
        return factory(data, o);
      };
      return entry;
    };
    std::vector<EncoderEntry<T>> nested;
    for (auto& entry : encoders) {
      if (std::find(nestedArms.begin(), nestedArms.end(), entry.name) !=
          nestedArms.end()) {
        auto copy = withNestedStreams(entry, true);
        copy.name += "+nestedSIS";
        copy.variant += "_nested_sis";
        nested.push_back(std::move(copy));
      }
      if (std::find(pureArms.begin(), pureArms.end(), entry.name) !=
          pureArms.end()) {
        entry = withNestedStreams(std::move(entry), false);
      }
    }
    for (auto& entry : nested) {
      encoders.push_back(std::move(entry));
    }
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
