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

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <latch>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "folly/Executor.h"
#include "folly/container/F14Set.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/DecoderUtil.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#include "velox/dwio/nimble/encodings/subintsplit/DecodeCost.h"
#include "velox/dwio/nimble/encodings/subintsplit/DeltaTransform.h"
#include "velox/dwio/nimble/encodings/subintsplit/Format.h"
#include "velox/dwio/nimble/encodings/subintsplit/PlanRefiner.h"
#include "velox/dwio/nimble/encodings/subintsplit/RowFrame.h"
#include "velox/dwio/nimble/encodings/subintsplit/Sampler.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionAccumulator.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionTransform.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitBoundaries.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitSelector.h"
#include "velox/dwio/nimble/encodings/subintsplit/TopLevelPolicy.h"
#include "velox/dwio/nimble/encodings/subintsplit/TuningConfig.h"
#ifdef __AVX2__
#include <immintrin.h>
#endif

// SubIntSplitEncoding: decomposes each value in a 32- or 64-bit integer stream
// into bit-range sub-streams, selects an optimal encoding for each sub-stream
// via a sample-driven DP algorithm, and stitches the encoded sub-streams back
// together for efficient decoding.
//
// Only supported for 32- and 64-bit types (int32_t, uint32_t, int64_t,
// uint64_t, float, double). The physical type for float is uint32_t and for
// double is uint64_t; bit patterns are preserved across encode/decode.
//
// Each section is encoded as the narrowest unsigned integer type that fits
// its bit width, to avoid paying an 8-byte-per-value penalty for narrow
// sections under e.g. Dictionary or Trivial encoding.
//
// The pieces live in encodings/subintsplit/: SplitSelector plans the bit
// ranges over a cost grid priced by CostModel, PlanRefiner re-prices a
// shortlist, RowFrame subtracts a predictor from the values, SectionTransform
// reorders sections, and Format.h documents the binary layout.

namespace facebook::nimble::subintsplit {

/// Options one SubIntSplit section is encoded under, derived from the
/// enclosing column's options.
///
/// A section overrides several of the column's options, changing encoded size,
/// so anything pricing a section's cost must derive options from here
/// rather than restate them.
inline Encoding::Options sectionEncodingOptions(
    const Encoding::Options& options,
    const TuningConfig& tuning = kDefaultTuningConfig) {
  Encoding::Options sectionOptions = options;
  // Pack each section at its exact bit width instead of rounding up to a
  // byte; sections dominate encoded size for multi-field values, so byte
  // rounding there is costly. FixedBitWidth records its own bit width, so
  // decode is unaffected.
  sectionOptions.fixedBitWidthUseExactBits = true;
  // Sections are priced against each other, so their estimates must track
  // the bytes an encoding really writes rather than only rank candidates.
  sectionOptions.subIntSplit.sectionEstimatorRefinements = true;
  // NoIndex would output values in tier-reordered order, desyncing this
  // section from siblings at decode time, so a section always carries an
  // index.
  //
  // TierTagArray costs ceilLog2(tiers + 1) bits per row versus one bit per
  // row per tier for PerTierBitmaps, cheaper from three tiers up (where
  // sections land) and faster on a contiguous read. Its cost is random
  // access: a point read scans a sampled stride for rank, where bitmaps
  // answer from a Rank9 superblock. Sections are read contiguously far more
  // often than probed, so this favors the contiguous case; a
  // point-read-heavy workload would want the other choice.
  sectionOptions.frequencyPartitionIndex =
      2u; // FreqPartIndexType::TierTagArray
  // Marks these as a section's options so subIntSplit.decodeWeight applies to
  // the section's own encoding candidates, not just the planner.
  sectionOptions.subIntSplit.sectionSelection = true;
  sectionOptions.subIntSplit.decodeWeight =
      tuning.selector.decodeWeighting.weight;
  sectionOptions.subIntSplit.decodeAccessPattern =
      tuning.selector.decodeWeighting.accessPattern;
  sectionOptions.subIntSplit.decodeReadPath =
      tuning.selector.decodeWeighting.readPath;
  sectionOptions.subIntSplit.maxSizeRegression = tuning.maxSizeRegression;
  return sectionOptions;
}

/// How far above a plan's bytes selection's estimate of a whole-value
/// encoding may be and still be worth encoding to check, since that
/// estimate can overshoot what the encoding actually writes.
///
/// RLE and Dictionary/FrequencyPartition estimates overshoot their actual
/// encoded size by different margins, and bit-packing estimates are exact,
/// so each gets its own slack rather than one shared value.
inline double wholeValueEstimateSlack(EncodingType encodingType) {
  switch (encodingType) {
    case EncodingType::RLE:
      return 2.5;
    case EncodingType::Dictionary:
    case EncodingType::FrequencyPartition:
    case EncodingType::MainlyConstant:
      return 1.25;
    default:
      return 1.0;
  }
}

} // namespace facebook::nimble::subintsplit

namespace facebook::nimble {

namespace detail {
// Defined in selection/EncodingSizeEstimation.h, which includes this header
// on some branches; only named here, from template code, so its definition is
// needed where SubIntSplitEncoding is instantiated rather than here.
template <typename T>
struct EncodingSizeEstimation;
} // namespace detail

template <typename T>
class SubIntSplitEncoding
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  static_assert(
      sizeof(physicalType) == 4 || sizeof(physicalType) == 8,
      "SubIntSplitEncoding only supports 32- and 64-bit types");
  static_assert(
      isNumericType<physicalType>(),
      "SubIntSplitEncoding only supports numeric types");

  SubIntSplitEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory,
      const Encoding::Options& options = {},
      const subintsplit::TuningConfig& tuning =
          subintsplit::kDefaultTuningConfig);

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  // Bulk-decodes the contiguous span covering the selected rows once, then
  // gathers/scatters into the visitor. Invoked by detail::readWithVisitorFast.
  template <bool kScatter, typename Visitor>
  void bulkScan(
      Visitor& visitor,
      vector_size_t currentRow,
      const vector_size_t* selectedRows,
      vector_size_t numSelected,
      const vector_size_t* scatterRows);

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {},
      const subintsplit::TuningConfig& tuning =
          subintsplit::kDefaultTuningConfig);

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
  /// Estimates what a split would store `values` in by running the same
  /// split DP the encoder uses, over a smaller sample and priced on size
  /// alone. The answer is floored at FixedBitWidth's exact estimate, which
  /// the encoder guarantees a split never exceeds. nullopt when the streams
  /// will be handed to a substream compressor and
  /// subintsplit::Options::estimateCompressionGuard is set.
  static std::optional<uint64_t> estimateSize(
      uint64_t rowCount,
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options);

  /// Bounds estimateSize() from below without planning a split, for a caller
  /// that only needs to know whether one can win. Returns FixedBitWidth's
  /// exact estimate when the bit-flip gradient gate finds no heterogeneity
  /// in `values` to split on; nullopt otherwise, and always nullopt unless
  /// subintsplit::Options::estimateBitFlipScreen is set.
  ///
  /// The bound is the gate's prediction, not a proof: it holds only where
  /// the gate is right that a rejected stream has nothing to exploit, so it
  /// can cost a false-reject column its SubIntSplit in exchange for
  /// skipping the split DP elsewhere.
  static std::optional<uint64_t> estimateSizeLowerBound(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options);
#endif

  std::string debugString(int offset) const final;

 private:
  struct SectionInfo {
    int bitStart{0};
    int bitEnd{0};
    uint64_t mask{0}; // (1 << width) - 1, or ~0 for full 64-bit section
    uint8_t storageBytes{8}; // 1, 2, 4, or 8 — matches the section's DataType
    std::unique_ptr<Encoding> encoding;
  };

  // Read context:
  //
  //   section headers + payloads -> [child decoders] -> decodeChunk -> values
  //
  // All dynamic children represent the same logical row. Reset, skip, and
  // decode must advance them together; folded Constant sections contribute
  // bits without a decoder.
  std::vector<SectionInfo> sections_;

  // The configuration the split planner runs under for `tuning`.
  static subintsplit::SelectorConfig plannerSelectorConfig(
      const subintsplit::TuningConfig& tuning,
      size_t rowCount);

  // Whether nested selection can estimate `type`, and so could ever choose
  // it for a section.
  static bool sectionSelectionEstimates(EncodingType type);

  // The sample estimateSize plans over. Smaller than the encoder's, because
  // the estimate only has to rank a split against the other candidates, not
  // choose the boundaries the encoder will use, and the DP is linear in the
  // sample: see estimateSize for what the difference costs and buys.
  static subintsplit::SamplerConfig estimatorSamplerConfig();

  // Bytes a split's header costs beyond its sections: the outer prefix and
  // compression type, plus a prefix and a relative offset per section.
  static constexpr uint64_t kOuterOverheadBytes{6u + 2u};
  static constexpr uint64_t kPerSectionOverheadBytes{6u + 8u};

  // A column's planner sample and the split grid costed over it. Costing the
  // grid is most of what planning costs, so the row frame decision, which has
  // to cost a grid for the values and for the residuals, hands the winner's to
  // the encode instead of the encode costing it a third time.
  struct PlanningSample {
    std::vector<uint64_t> sample;
    std::vector<subintsplit::SectionCost> grid;
  };

  // Samples `values` and costs the split grid over the sample.
  static PlanningSample costPlanningSample(
      std::span<const physicalType> values,
      const Encoding::Options& options,
      const subintsplit::TuningConfig& tuning);

  // Encodes `values` as they are, fitting a row frame first when the options
  // ask for one. encode() calls this once, or twice when it also tries the
  // delta pre-transform.
  static std::string_view encodeValues(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options,
      const subintsplit::TuningConfig& tuning);

  // Plans `values` into sections and encodes each. `values` have `rowFrame`
  // already subtracted from them, and the header records the frame so reads
  // add it back. `planning`, when not null, is the sample and grid already
  // costed for exactly these values. `extraFlags` goes into the header's flag
  // byte, for instance to record that `values` are zigzag deltas.
  // `stepFrame`, when not null, is a step frame fitted to the same column, and
  // `stepResiduals` the column with it subtracted: the whole-value floor may
  // store those residuals as its one section instead.
  static std::string_view encodeResiduals(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options,
      const subintsplit::TuningConfig& tuning,
      const subintsplit::RowFrame& rowFrame = {},
      PlanningSample* planning = nullptr,
      uint8_t extraFlags = 0,
      const subintsplit::RowFrame* stepFrame = nullptr,
      std::span<const physicalType> stepResiduals = {});

  // What section selection's pick is quoted for storing `values` as one
  // whole-value section, priced on contiguous blocks of them
  // and scaled to all of them: the encoding, its quote, and the quote divided
  // by how far that encoding's estimate has been measured above what it
  // writes. Nothing when no such encoding can be priced.
  struct SampledWholeValue {
    EncodingType encoding;
    double estimatedBytes;
    double lowerBoundBytes;
  };
  static std::optional<SampledWholeValue> sampleWholeValue(
      EncodingSelectionPolicy<physicalType>& sectionPolicy,
      std::span<const physicalType> values,
      const Encoding::Options& sectionOptions);

  // The whole-value sections a plan is held against: FixedBitWidth's exact
  // estimate, and, unless the plan is already one section, section
  // selection's sampled pick for the whole value.
  //
  // A class rather than one call so the fallback can be priced before the
  // plan is encoded, letting a losing plan's encode be abandoned partway
  // instead of finished and thrown away; each candidate is still encoded at
  // most once however many times it is asked for.
  class WholeValueFloor {
   public:
    WholeValueFloor(
        EncodingSelection<physicalType>& selection,
        std::span<const physicalType> values,
        Buffer& sectionBuffer,
        const Encoding::Options& sectionOptions,
        bool planIsWholeValue,
        bool valuesAreColumn);

    // Whether a policy narrowed to a single encoding could be had at all,
    // without which there is no fallback to offer.
    bool usable() const {
      return sectionPolicy_ != nullptr;
    }

    // The sample's quote, or nothing when no candidate could be priced.
    const std::optional<SampledWholeValue>& quote();

    // The smallest whole-value section that stores `values` in fewer than
    // `bytesToBeat` bytes, or nothing. Encodes a candidate the first time the
    // bound admits it and reuses it afterwards. `replayingEncoding` is the
    // Delta or Varint a plan's only varying section is stored in, which the
    // whole value may then be stored in too.
    std::optional<std::string_view> under(
        uint64_t bytesToBeat,
        std::optional<EncodingType> replayingEncoding);

    // Bytes a plan has to come in under for `under` to be answering the same
    // question at every bound above it: past this, every candidate is admitted
    // and the answer is the smallest of them. Only meaningful once `under` has
    // been called with no bound.
    uint64_t decidedAbove() const {
      return decidedAbove_;
    }

   private:
    std::string_view encodeAs(EncodingType encodingType);

    EncodingSelection<physicalType>& selection_;
    std::span<const physicalType> values_;
    Buffer& sectionBuffer_;
    const Encoding::Options& sectionOptions_;
    const bool planIsWholeValue_;
    const bool valuesAreColumn_;
    std::unique_ptr<EncodingSelectionPolicy<physicalType>> sectionPolicy_;

    bool quoted_{false};
    std::optional<SampledWholeValue> quote_;
    std::optional<std::string_view> sampledEncoded_;
    std::optional<uint64_t> fixedBitWidthEstimate_;
    std::optional<std::string_view> fixedBitWidthEncoded_;
    std::optional<std::string_view> replayingEncoded_;
    uint64_t decidedAbove_{0};
  };
  // Predictor the encoder subtracted before planning, inactive when it did
  // not. Every value leaving this class has it added back.
  subintsplit::RowFrame rowFrame_;

  // Per-section transform metadata from the header. Empty ids mean the stream
  // predates transforms, or chose none.
  subintsplit::TransformInfo transformInfo_;
  // Widened section values for the block being decoded, reused across
  // blocks. Populated only for a section that needs a uint64 span: one that
  // is itself transformed (invert() requires it) or that serves as another
  // section's key (keySpan is handed to invert() the same way). Every other
  // section decodes straight into sectionNative_ below and assembly reads it
  // at its own width, so most streams never widen at all.
  std::vector<std::vector<uint64_t>> sectionScratch_;
  // Byte-packed decode buffer for a section sectionNeedsWide_ marks false, at
  // the section's own storage width, one vector per section.
  std::vector<std::vector<uint8_t>> sectionNative_;
  // Whether section s must be widened to uint64: it carries a transform, or
  // it is the key section another section's transform reads. Computed once
  // at construction, since transformIds and keySection never change after.
  std::vector<bool> sectionNeedsWide_;
  // The row the sections stand at. Only meaningful for a transformed stream.
  uint32_t sectionsAt_{0};
  std::vector<physicalType> blockCache_;

  // The key's run bookkeeping for the block currently being decoded, built
  // once and shared by every section keyed on it instead of each one's
  // invert() rebuilding it. Reused across blocks purely for capacity; every
  // field is fully overwritten before being read.
  subintsplit::KeyRunState keyRunState_;
  // Working run-start cursor for the fused key-derived accumulate path
  // (decodeTransformedColumn's assembly loop, not invert()), reused across
  // sections and blocks.
  std::vector<uint32_t> fusedCursor_;

  // Decodes a stream whose sections carry a transform. A transform is undone
  // over the whole column, so a partial read is served from blockCache_.
  void materializeTransformed(uint32_t rowCount, physicalType* output);

  // Decodes and inverts the whole column into blockCache_, or, when
  // `directOutput` is not null, straight into it instead. Only a read of
  // every row may pass a non-null directOutput: doing so leaves blockCache_
  // untouched, so a later partial read or point probe decodes normally rather
  // than reading stale cache contents.
  void decodeTransformedColumn(physicalType* directOutput);

  // Persistent scratch buffer reused across materialize() calls. Grown to one
  // chunk of the current read, at most decodeChunkSize_ values of
  // sizeof(physicalType) bytes.
  Vector<uint8_t> scratchBuf_;

  // Whether the stored sections hold zigzag deltas rather than values. Such a
  // stream carries no frame and no transform, and is read strictly in order.
  bool deltaEncoded_{false};
  // Running prefix sum for a delta stream, carried across materialize() calls.
  physicalType deltaAccumulator_{0};

  // Sections the untransformed path decodes per chunk. Every section, unless
  // subintsplit::TuningConfig::foldConstantSections folded the Constant ones
  // into constantOr_ at construction.
  std::vector<uint32_t> dynamicSections_;
  // Combined, pre-shifted bits of every folded Constant section, OR-ed into
  // each value by the first dynamic section.
  physicalType constantOr_{0};
  // One dynamic section reproduces each value verbatim, so it decodes straight
  // into the caller's buffer. Only with subintsplit::TuningConfig::passThrough.
  bool passThrough_{false};
  // Output elements combined per chunk on the untransformed path.
  uint32_t decodeChunkSize_{subintsplit::kDecodeChunkSize};

  // Values the readWithVisitor slow path decoded ahead of the read cursor, when
  // subintsplit::TuningConfig::visitorBlockBuffer is set. The sections then
  // stand at row_ + pendingAvailable(), and skip() and materialize() consume
  // this buffer before touching them.
  static constexpr uint32_t kSlowPathBlock = 256;
  bool visitorBlockBuffer_{false};
  Vector<physicalType> pendingBuf_;
  uint32_t pendingOffset_{0};
  uint32_t pendingCount_{0};

  uint32_t pendingAvailable() const noexcept {
    return pendingCount_ - pendingOffset_;
  }

  // Hands back values the slow path decoded ahead of the cursor. Returns how
  // many were written to `output`.
  uint32_t takeFromPending(uint32_t rowCount, physicalType* output);

  // Decodes the next block of values into pendingBuf_, clamped to what the
  // stream has left.
  void refillPending();

  // Decodes the untransformed sections for `rowCount` rows into `output`.
  void decodeUntransformed(uint32_t rowCount, physicalType* output);

  // Logical read cursor (rows consumed so far). Maintained across skip(),
  // materialize(), and the readWithVisitor slow path so the fast path can map
  // external row numbers onto the section cursors.
  uint32_t row_{0};

  // Scratch buffer for the readWithVisitor fast path. Holds the decoded span of
  // physical values before they are gathered/widened into the reader output.
  Vector<physicalType> decodeBuf_;

  // Return the storage byte width for a section of the given bit width.
  static constexpr uint8_t sectionStorageBytes(int bitWidth) noexcept {
    return subintsplit::sectionStorageBytes(bitWidth);
  }

  // Forwards to the kernel shared with SubIntSplitEncodingView.
  template <typename SectionT, bool IsFirst>
  static void accumulateSection(
      const SectionT* __restrict__ src,
      physicalType* __restrict__ dst,
      uint32_t count,
      uint64_t mask,
      int shift,
      physicalType orConstant = 0) noexcept {
    subintsplit::accumulateSection<IsFirst>(
        src, dst, count, mask, shift, orConstant);
  }
};

//
// End of public API. Implementation follows.
//

template <typename T>
SubIntSplitEncoding<T>::SubIntSplitEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory,
    const Encoding::Options& options,
    const subintsplit::TuningConfig& tuning)
    : TypedEncoding<T, physicalType>{pool, data, options},
      sections_{},
      scratchBuf_{&pool},
      pendingBuf_{&pool},
      decodeBuf_{&pool} {
  uint8_t flags{0};
  const auto parsed = subintsplit::parseSections(
      data, this->dataOffset(), &flags, &rowFrame_, &transformInfo_);
  deltaEncoded_ = (flags & subintsplit::kFlagDelta) != 0;
  if (tuning.decodeChunkSize > 0) {
    decodeChunkSize_ = tuning.decodeChunkSize;
  }
  visitorBlockBuffer_ = tuning.visitorBlockBuffer;
  NIMBLE_CHECK(!parsed.empty(), "SubIntSplit stream has no sections.");
  // Validate every id before decoding anything: a transform this reader does
  // not know would otherwise be skipped, returning transformed values as
  // though they were the originals.
  for (uint8_t id : transformInfo_.transformIds) {
    subintsplit::transformForRaw(id);
  }
  // Only the reordered type carries transforms, and a reordered stream must
  // carry one, so the type and the header cannot disagree about whether the
  // sections need undoing.
  NIMBLE_CHECK_FILE(
      transformInfo_.anyTransform() ==
          (this->encodingType() == EncodingType::SubIntSplitReordered),
      "SubIntSplit encoding type does not match its transform block.");

  sections_.resize(parsed.size());
  for (size_t s = 0; s < parsed.size(); ++s) {
    auto& sec = sections_[s];
    sec.bitStart = parsed[s].bitStart;
    sec.bitEnd = parsed[s].bitEnd;
    sec.mask = parsed[s].mask;
    sec.storageBytes = parsed[s].storageBytes;
    NIMBLE_CHECK_FILE(
        parsed[s].stream.size() >= EncodingPrefix::kRowCountOffset,
        "SubIntSplit section encoding prefix is truncated.");
    const auto expectedDataType = subintsplit::dispatchStorageType(
        sec.storageBytes, []<typename StorageType>() {
          return TypeTraits<StorageType>::dataType;
        });
    NIMBLE_CHECK_FILE(
        EncodingPrefix::dataType(parsed[s].stream) == expectedDataType,
        "SubIntSplit section data type does not match its bit width.");
    sec.encoding = EncodingFactory().create(
        *this->pool_, parsed[s].stream, stringBufferFactory, options);
  }

  // A section needs a uint64 span only if invert() is called on it directly,
  // or if it is handed to another section's invert() as the key. Every other
  // section is assembled straight from its own storage width.
  sectionNeedsWide_.assign(sections_.size(), false);
  for (size_t s = 0; s < transformInfo_.transformIds.size(); ++s) {
    if (transformInfo_.transformIds[s] != 0) {
      sectionNeedsWide_[s] = true;
    }
  }
  if (transformInfo_.keySection != subintsplit::TransformInfo::kNoKeySection &&
      transformInfo_.keySection < sectionNeedsWide_.size()) {
    sectionNeedsWide_[transformInfo_.keySection] = true;
  }

  // A Constant section contributes the same bits to every row, so decoding it
  // materialises copies of one number to OR them in a row at a time. With the
  // switch on it is read once here and left out of the chunk loop. Only the
  // untransformed path folds: a transformed stream assembles every section
  // through the block cache, and a key section must stay decodable.
  dynamicSections_.reserve(sections_.size());
  for (uint32_t s = 0; s < sections_.size(); ++s) {
    const auto& sec = sections_[s];
    if (tuning.foldConstantSections && !transformInfo_.anyTransform() &&
        sec.encoding->encodingType() == EncodingType::Constant) {
      // ConstantEncoding ignores the read position, so reading it here leaves
      // the cursor the decode path relies on where it was.
      const uint64_t value = subintsplit::dispatchStorageType(
          sec.storageBytes, [&]<typename StorageType>() -> uint64_t {
            StorageType stored{0};
            sec.encoding->materialize(1, &stored);
            return static_cast<uint64_t>(stored);
          });
      constantOr_ |= static_cast<physicalType>(value & sec.mask)
          << sec.bitStart;
      continue;
    }
    dynamicSections_.push_back(s);
  }
  if (tuning.passThrough && dynamicSections_.size() == 1 && constantOr_ == 0 &&
      !transformInfo_.anyTransform()) {
    const auto& only = sections_[dynamicSections_.front()];
    constexpr uint64_t kFullMask =
        ~uint64_t{0} >> (64 - sizeof(physicalType) * 8);
    passThrough_ = only.bitStart == 0 &&
        only.storageBytes == sizeof(physicalType) && only.mask == kFullMask;
  }
}

template <typename T>
void SubIntSplitEncoding<T>::reset() {
  for (auto& sec : sections_) {
    sec.encoding->reset();
  }
  row_ = 0;
  sectionsAt_ = 0;
  deltaAccumulator_ = 0;
  pendingOffset_ = 0;
  pendingCount_ = 0;
  // blockCache_ is deliberately kept. It holds decoded rows addressed by their
  // absolute position, which rewinding the cursor does not invalidate, and a
  // point read reaches this class by resetting before every probe: dropping it
  // would make each probe decode the column again.
}

template <typename T>
void SubIntSplitEncoding<T>::skip(uint32_t rowCount) {
  // A delta stream rebuilds each value from every step before it, so skipping
  // rows means decoding them. Reads stay correct, and become sequential.
  if (deltaEncoded_) {
    decodeBuf_.resize(std::min(rowCount, decodeChunkSize_));
    while (rowCount > 0) {
      const uint32_t count = std::min(rowCount, decodeChunkSize_);
      materialize(count, decodeBuf_.data());
      rowCount -= count;
    }
    return;
  }
  // The sections already sit past anything the visitor slow path buffered, so
  // those rows are skipped by dropping them rather than by moving the cursors.
  const uint32_t fromPending = std::min(rowCount, pendingAvailable());
  pendingOffset_ += fromPending;
  row_ += fromPending;
  rowCount -= fromPending;
  // A transformed stream is decoded as a whole column, so a skip only moves
  // the logical position.
  if (transformInfo_.anyTransform()) {
    row_ += rowCount;
    return;
  }
  for (auto& sec : sections_) {
    sec.encoding->skip(rowCount);
  }
  row_ += rowCount;
}

template <typename T>
void SubIntSplitEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  physicalType* output = static_cast<physicalType*>(buffer);
  const uint32_t fromPending = takeFromPending(rowCount, output);
  rowCount -= fromPending;
  if (rowCount == 0) {
    return;
  }
  output += fromPending;
  const uint32_t firstRow = row_;
  // A transformed stream takes a separate path, because its sections have to
  // be inverted before they are assembled.
  if (transformInfo_.anyTransform()) {
    materializeTransformed(rowCount, output);
  } else {
    decodeUntransformed(rowCount, output);
    row_ += rowCount;
  }
  if (rowFrame_.active()) {
    subintsplit::addRowFrame(rowFrame_, firstRow, output, rowCount);
  }
  if (deltaEncoded_) {
    subintsplit::decodeDeltas<physicalType>(
        {output, rowCount}, deltaAccumulator_, firstRow == 0);
  }
}

template <typename T>
uint32_t SubIntSplitEncoding<T>::takeFromPending(
    uint32_t rowCount,
    physicalType* output) {
  if (pendingAvailable() == 0) [[likely]] {
    return 0;
  }
  const uint32_t taken = std::min(rowCount, pendingAvailable());
  std::copy_n(pendingBuf_.data() + pendingOffset_, taken, output);
  pendingOffset_ += taken;
  row_ += taken;
  return taken;
}

template <typename T>
void SubIntSplitEncoding<T>::refillPending() {
  if (pendingBuf_.size() < kSlowPathBlock) [[unlikely]] {
    pendingBuf_.resize(kSlowPathBlock);
  }
  const uint32_t total = this->rowCount();
  NIMBLE_CHECK(
      row_ < total, "SubIntSplitEncoding: read past the end of the stream.");
  const uint32_t block = std::min(kSlowPathBlock, total - row_);
  // Decoded without moving row_, which stays the logical cursor; the sections
  // run ahead of it by what is buffered.
  decodeUntransformed(block, pendingBuf_.data());
  pendingOffset_ = 0;
  pendingCount_ = block;
}

template <typename T>
void SubIntSplitEncoding<T>::decodeUntransformed(
    uint32_t rowCount,
    physicalType* output) {
  // A single section holding each value verbatim needs no masking, shifting
  // or OR-ing, so it writes the caller's buffer directly.
  if (passThrough_) {
    const uint32_t s = dynamicSections_.front();
    sections_[s].encoding->materialize(rowCount, output);
    return;
  }
  // Every section was constant, so each value is the folded constant.
  if (dynamicSections_.empty()) [[unlikely]] {
    std::fill_n(output, rowCount, constantOr_);
    return;
  }

  // Lazily size the scratch buffer. It holds one chunk's worth of section
  // values at the widest storage type, and a chunk is never longer than this
  // read, so a large configured chunk size costs nothing here. Computed in 64
  // bits: the chunk size is a caller option, and multiplying it by the value
  // width in 32 bits can wrap to a buffer smaller than a chunk.
  const uint64_t scratchBytes =
      static_cast<uint64_t>(std::min(rowCount, decodeChunkSize_)) *
      sizeof(physicalType);
  if (scratchBuf_.size() < scratchBytes) [[unlikely]] {
    scratchBuf_.resize(scratchBytes);
  }

  // Chunks decodeChunkSize_ elements at a time, accumulating all sections
  // for a chunk before moving on, so the output slice and scratch buffer
  // stay cache-resident across the section loop.
  uint32_t chunkCount{0};
  for (uint32_t chunkStart = 0; chunkStart < rowCount;
       chunkStart += chunkCount) {
    chunkCount = std::min(decodeChunkSize_, rowCount - chunkStart);
    physicalType* chunkOutput = output + chunkStart;

    // With nothing folded every section is dynamic and in order, so the
    // indirection through dynamicSections_ is skipped: on a one-row read it is
    // a dependent load per section, and those reads are what a gather issues.
    const bool allDynamic = dynamicSections_.size() == sections_.size();
    const uint32_t* dynamic = dynamicSections_.data();
    const size_t numDynamic = dynamicSections_.size();
    for (size_t d = 0; d < numDynamic; ++d) {
      const size_t s = allDynamic ? d : dynamic[d];
      const auto& sec = sections_[s];
      const int shift = sec.bitStart;
      const uint64_t mask = sec.mask;
      // The first section initialises each output element (pure write) and
      // ORs in the folded constants; subsequent sections OR their bits in.
      // This avoids a separate std::fill pass.
      const bool isFirst = (d == 0);

      switch (sec.storageBytes) {
        case 1: {
          auto* scratch = reinterpret_cast<uint8_t*>(scratchBuf_.data());
          sec.encoding->materialize(chunkCount, scratch);
          if (isFirst)
            accumulateSection<uint8_t, true>(
                scratch, chunkOutput, chunkCount, mask, shift, constantOr_);
          else
            accumulateSection<uint8_t, false>(
                scratch, chunkOutput, chunkCount, mask, shift);
          break;
        }
        case 2: {
          auto* scratch = reinterpret_cast<uint16_t*>(scratchBuf_.data());
          sec.encoding->materialize(chunkCount, scratch);
          if (isFirst)
            accumulateSection<uint16_t, true>(
                scratch, chunkOutput, chunkCount, mask, shift, constantOr_);
          else
            accumulateSection<uint16_t, false>(
                scratch, chunkOutput, chunkCount, mask, shift);
          break;
        }
        case 4: {
          auto* scratch = reinterpret_cast<uint32_t*>(scratchBuf_.data());
          sec.encoding->materialize(chunkCount, scratch);
          if (isFirst)
            accumulateSection<uint32_t, true>(
                scratch, chunkOutput, chunkCount, mask, shift, constantOr_);
          else
            accumulateSection<uint32_t, false>(
                scratch, chunkOutput, chunkCount, mask, shift);
          break;
        }
        case 8: {
          auto* scratch = reinterpret_cast<uint64_t*>(scratchBuf_.data());
          sec.encoding->materialize(chunkCount, scratch);
          if (isFirst)
            accumulateSection<uint64_t, true>(
                scratch, chunkOutput, chunkCount, mask, shift, constantOr_);
          else
            accumulateSection<uint64_t, false>(
                scratch, chunkOutput, chunkCount, mask, shift);
          break;
        }
        default:
          NIMBLE_UNREACHABLE("Invalid SubIntSplit section storage width.");
      }
    }
  }
}

template <typename T>
template <typename V>
void SubIntSplitEncoding<T>::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  using OutputType = detail::ValueType<typename V::DataType>;
  constexpr bool kIsSuitableWidth =
      (isFourByteIntegralType<physicalType>() ||
       isEightByteIntegralType<physicalType>());
  constexpr bool kIsFluidCast = sizeof(OutputType) >= sizeof(physicalType) &&
      std::is_integral_v<OutputType> && std::is_integral_v<physicalType>;

  // Fast path: bulk-decode for integral 4/8-byte physical types into a
  // compatible output type. Float/double fall through to the slow path,
  // which applies castFromPhysicalType; useFastPath also requires a
  // deterministic filter, AVX2, and null/filter/hook compatibility.
  if constexpr (
      kIsSuitableWidth &&
      std::is_same_v<
          typename V::Extract,
          velox::dwio::common::ExtractToReader> &&
      kIsFluidCast) {
    auto* nulls = visitor.reader().rawNullsInReadRange();
    if (velox::dwio::common::useFastPath(visitor, nulls)) {
      detail::readWithVisitorFast(*this, visitor, params, nulls);
      return;
    }
  }

  // Slow path: reconstruct one value at a time from the section encodings.
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&] {
        // A delta stream rebuilds each value from every earlier row, a row
        // frame is added back by the row a value sits at, and reassembling a
        // reordered stream from its sections returns the permuted values; only
        // materialize() handles each, so this path defers to it.
        if (deltaEncoded_ || rowFrame_.active() ||
            transformInfo_.anyTransform()) {
          physicalType value = 0;
          materialize(1, &value);
          return value;
        }
        // Decoding a block at a time turns a virtual call per section per value
        // into one per section per block.
        if (visitorBlockBuffer_) {
          if (pendingAvailable() == 0) {
            refillPending();
          }
          ++row_;
          return pendingBuf_[pendingOffset_++];
        }
        physicalType value = 0;
        for (const auto& sec : sections_) {
          switch (sec.storageBytes) {
            case 1: {
              uint8_t sectionValue = 0;
              sec.encoding->materialize(1, &sectionValue);
              value |= static_cast<physicalType>(sectionValue & sec.mask)
                  << sec.bitStart;
              break;
            }
            case 2: {
              uint16_t sectionValue = 0;
              sec.encoding->materialize(1, &sectionValue);
              value |= static_cast<physicalType>(sectionValue & sec.mask)
                  << sec.bitStart;
              break;
            }
            case 4: {
              uint32_t sectionValue = 0;
              sec.encoding->materialize(1, &sectionValue);
              value |= static_cast<physicalType>(sectionValue & sec.mask)
                  << sec.bitStart;
              break;
            }
            case 8: {
              uint64_t sectionValue = 0;
              sec.encoding->materialize(1, &sectionValue);
              value |= static_cast<physicalType>(sectionValue & sec.mask)
                  << sec.bitStart;
              break;
            }
            default: {
              NIMBLE_UNREACHABLE("Invalid SubIntSplit section storage width.");
            }
          }
        }
        // Keep row_ in sync so a subsequent fast-path chunk maps rows
        // correctly.
        ++row_;
        return value;
      });
}

template <typename T>
template <bool kScatter, typename V>
void SubIntSplitEncoding<T>::bulkScan(
    V& visitor,
    vector_size_t currentRow,
    const vector_size_t* selectedRows,
    vector_size_t numSelected,
    const vector_size_t* scatterRows) {
  using OutputType = detail::ValueType<typename V::DataType>;
  static_assert(
      isFourByteIntegralType<physicalType>() ||
          isEightByteIntegralType<physicalType>(),
      "bulkScan only supports 4-byte or 8-byte integral types");

  if (numSelected == 0) {
    return;
  }

  const auto numRows = visitor.numRows() - visitor.rowIndex();

  // Map external row numbers onto the section cursors. Nulls can make the
  // encoding (non-null) position lag the logical row number.
  const auto offset =
      static_cast<int32_t>(row_) - static_cast<int32_t>(currentRow);

  // The selected rows all lie within one contiguous span of stored (non-null)
  // values. Decode that whole span once, then gather the selected positions.
  const vector_size_t spanStart = selectedRows[0] + offset;
  const vector_size_t spanEnd = selectedRows[numSelected - 1] + offset;
  const uint32_t spanLength = static_cast<uint32_t>(spanEnd - spanStart + 1);

  // Advance the section cursors to the start of the span.
  if (spanStart > static_cast<vector_size_t>(row_)) {
    skip(static_cast<uint32_t>(spanStart - static_cast<vector_size_t>(row_)));
  }

  auto* values = detail::mutableValues<OutputType>(visitor, numRows);

  // Same-size integral output shares the physical bit pattern, so we can decode
  // straight into the reader buffer; otherwise stage in decodeBuf_ and widen.
  constexpr bool kSameSize = sizeof(physicalType) == sizeof(OutputType);

  if constexpr (V::dense) {
    // Dense: the span is exactly the selected rows (spanLength == numSelected).
    if constexpr (kSameSize) {
      materialize(spanLength, values);
    } else {
      decodeBuf_.resize(spanLength);
      materialize(spanLength, decodeBuf_.data());
      for (vector_size_t i = 0; i < numSelected; ++i) {
        values[i] = static_cast<OutputType>(decodeBuf_[i]);
      }
    }
  } else {
    // Sparse: decode the span, then gather the selected positions.
    decodeBuf_.resize(spanLength);
    materialize(spanLength, decodeBuf_.data());
    for (vector_size_t i = 0; i < numSelected; ++i) {
      values[i] = static_cast<OutputType>(
          decodeBuf_[selectedRows[i] - selectedRows[0]]);
    }
  }

  // No scatter, filter, or hook: values are already in the output buffer.
  if constexpr (!kScatter && !V::kHasFilter && !V::kHasHook) {
    visitor.addNumValues(numRows);
    visitor.setRowIndex(visitor.numRows());
    return;
  }

  detail::applyFixedWidthRun<kScatter>(
      visitor, selectedRows, numSelected, scatterRows, values, numRows);
  visitor.setRowIndex(visitor.numRows());
}

template <typename T>
void SubIntSplitEncoding<T>::materializeTransformed(
    uint32_t rowCount,
    physicalType* output) {
  if (rowCount == 0) {
    return;
  }
  // The cache exists so that a probe does not rebuild the column it has just
  // rebuilt. It must not answer a read that asks for every row: that read is
  // a decode, and serving it from a cache would report the cost of a copy in
  // place of the cost of decoding.
  if (rowCount >= this->rowCount()) {
    decodeTransformedColumn(output);
  } else {
    if (blockCache_.empty()) {
      decodeTransformedColumn(nullptr);
    }
    std::copy_n(blockCache_.data() + row_, rowCount, output);
  }
  row_ += rowCount;
}

template <typename T>
void SubIntSplitEncoding<T>::decodeTransformedColumn(
    physicalType* directOutput) {
  // The sections decode forwards only, so decoding the column again means
  // starting them over.
  if (sectionsAt_ > 0) {
    for (auto& sec : sections_) {
      sec.encoding->reset();
    }
    sectionsAt_ = 0;
  }

  const uint32_t numRows = this->rowCount();

  const uint32_t neededBytes =
      numRows * static_cast<uint32_t>(sizeof(physicalType));
  if (scratchBuf_.size() < neededBytes) [[unlikely]] {
    scratchBuf_.resize(neededBytes);
  }

  // A section that carries a transform, or serves as another section's key,
  // decodes into a widened uint64 scratch entry since invert() needs that
  // span; other sections decode straight into their native-width buffer.
  // Buffers are held across blocks rather than reallocated per block, since
  // a bulk decode walks thousands of them.
  sectionScratch_.resize(sections_.size());
  sectionNative_.resize(sections_.size());
  for (size_t s = 0; s < sections_.size(); ++s) {
    auto& sec = sections_[s];
    if (!sectionNeedsWide_[s] && sec.storageBytes != 8) {
      auto& native = sectionNative_[s];
      native.resize(numRows * sec.storageBytes);
      switch (sec.storageBytes) {
        case 1:
          sec.encoding->materialize(numRows, native.data());
          break;
        case 2:
          sec.encoding->materialize(
              numRows, reinterpret_cast<uint16_t*>(native.data()));
          break;
        default:
          sec.encoding->materialize(
              numRows, reinterpret_cast<uint32_t*>(native.data()));
          break;
      }
      continue;
    }
    auto& values = sectionScratch_[s];
    values.resize(numRows);
    if (sec.storageBytes == 8) {
      // Already the target width: decode directly, skipping the
      // narrow-to-wide copy entirely.
      sec.encoding->materialize(numRows, values.data());
      continue;
    }
    const uint32_t neededBytes =
        numRows * static_cast<uint32_t>(sizeof(physicalType));
    if (scratchBuf_.size() < neededBytes) [[unlikely]] {
      scratchBuf_.resize(neededBytes);
    }
    switch (sec.storageBytes) {
      case 1: {
        auto* scratch = reinterpret_cast<uint8_t*>(scratchBuf_.data());
        sec.encoding->materialize(numRows, scratch);
        for (uint32_t i = 0; i < numRows; ++i) {
          values[i] = scratch[i];
        }
        break;
      }
      case 2: {
        auto* scratch = reinterpret_cast<uint16_t*>(scratchBuf_.data());
        sec.encoding->materialize(numRows, scratch);
        for (uint32_t i = 0; i < numRows; ++i) {
          values[i] = scratch[i];
        }
        break;
      }
      default: {
        auto* scratch = reinterpret_cast<uint32_t*>(scratchBuf_.data());
        sec.encoding->materialize(numRows, scratch);
        for (uint32_t i = 0; i < numRows; ++i) {
          values[i] = scratch[i];
        }
        break;
      }
    }
  }
  sectionsAt_ = numRows;

  // The key section is stored in original order precisely so it can order the
  // sections that were permuted by it, so it is never itself transformed.
  std::span<const uint64_t> keySpan;
  if (transformInfo_.keySection != subintsplit::TransformInfo::kNoKeySection) {
    keySpan =
        std::span<const uint64_t>(sectionScratch_[transformInfo_.keySection]);
  }

  // Builds the key's run bookkeeping once, here, when at least one section
  // in this block is keyed on it, and shares it with every such section
  // instead of letting each one rebuild it.
  bool haveSharedKeyRunState = false;
  if (!keySpan.empty()) {
    for (size_t s = 0; s < sections_.size(); ++s) {
      const uint8_t id = transformInfo_.transformIds[s];
      if (id != 0 &&
          subintsplit::transformForRaw(id)->id() ==
              subintsplit::TransformId::KeyDerived) {
        haveSharedKeyRunState = true;
        break;
      }
    }
    if (haveSharedKeyRunState) {
      subintsplit::buildKeyRunState(keySpan, keyRunState_);
    }
  }

  // Accumulates one section at a time across the whole block, reusing the
  // same SIMD kernel the untransformed path calls from materialize(), so
  // width/widening dispatch happens once per section rather than once per
  // (row, section). A section that was never widened accumulates straight
  // from its native-width buffer; only a widened section reads through
  // sectionScratch_.
  //
  // A whole-block bulk read hands its own output buffer in as directOutput,
  // so assembly writes straight into it and blockCache_ is left untouched.
  const bool direct = directOutput != nullptr;
  if (!direct) {
    blockCache_.resize(numRows);
  }
  physicalType* dst = direct ? directOutput : blockCache_.data();
  for (size_t s = 0; s < sections_.size(); ++s) {
    const auto& sec = sections_[s];
    const int shift = sec.bitStart;
    const uint64_t mask = sec.mask;
    const bool isFirst = (s == 0);
    const uint8_t id = transformInfo_.transformIds[s];
    // Undoes the key-derived permutation and ORs its bits into dst in one
    // pass, rather than invert() merging into a temporary buffer that
    // accumulateSection would read back out of sectionScratch_.
    if (id != 0 &&
        subintsplit::transformForRaw(id)->id() ==
            subintsplit::TransformId::KeyDerived) {
      // A key-derived section is only ever written beside a key section.
      NIMBLE_CHECK(
          haveSharedKeyRunState,
          "SubIntSplit key-derived section has no key section.");
      fusedCursor_.assign(
          keyRunState_.runStart.begin(), keyRunState_.runStart.end());
      const uint64_t* src = sectionScratch_[s].data();
      const uint32_t* runOfRow = keyRunState_.runOfRow.data();
      const uint32_t* sortedRank = keyRunState_.sortedRank.data();
      uint32_t* cursor = fusedCursor_.data();
      if (isFirst) {
        for (uint32_t i = 0; i < numRows; ++i) {
          const uint64_t v = src[cursor[sortedRank[runOfRow[i]]]++];
          dst[i] = static_cast<physicalType>((v & mask) << shift);
        }
      } else {
        for (uint32_t i = 0; i < numRows; ++i) {
          const uint64_t v = src[cursor[sortedRank[runOfRow[i]]]++];
          dst[i] |= static_cast<physicalType>((v & mask) << shift);
        }
      }
      continue;
    }
    if (sec.storageBytes == 8 || sectionNeedsWide_[s]) {
      const auto* src = sectionScratch_[s].data();
      if (isFirst) {
        accumulateSection<uint64_t, true>(src, dst, numRows, mask, shift);
      } else {
        accumulateSection<uint64_t, false>(src, dst, numRows, mask, shift);
      }
      continue;
    }
    const uint8_t* native = sectionNative_[s].data();
    switch (sec.storageBytes) {
      case 1: {
        if (isFirst) {
          accumulateSection<uint8_t, true>(native, dst, numRows, mask, shift);
        } else {
          accumulateSection<uint8_t, false>(native, dst, numRows, mask, shift);
        }
        break;
      }
      case 2: {
        const auto* src = reinterpret_cast<const uint16_t*>(native);
        if (isFirst) {
          accumulateSection<uint16_t, true>(src, dst, numRows, mask, shift);
        } else {
          accumulateSection<uint16_t, false>(src, dst, numRows, mask, shift);
        }
        break;
      }
      default: {
        const auto* src = reinterpret_cast<const uint32_t*>(native);
        if (isFirst) {
          accumulateSection<uint32_t, true>(src, dst, numRows, mask, shift);
        } else {
          accumulateSection<uint32_t, false>(src, dst, numRows, mask, shift);
        }
        break;
      }
    }
  }
}

/// Whether a plan of `candidateBytes` displaces the smallest found so far.
///
/// Strictly smaller, so the first candidate to reach the minimum keeps it,
/// independent of try order.
///
/// The search also abandons a candidate the moment its running total stops
/// satisfying this test, which is sound because a plan only grows as
/// sections are added: an abandoned candidate could not have displaced the
/// incumbent had it been finished. Loosening this to `<=` without loosening
/// the abandon test the same way would silently make the search stop
/// pricing plans it had just decided it wanted.
inline bool improvesOnBest(size_t candidateBytes, size_t bestBytes) noexcept {
  return candidateBytes < bestBytes;
}

// Whether sorting by these values would group anything.
//
// A key-derived permutation earns its keep by bringing like rows together;
// a key with nearly as many values as there are rows has nothing to bring
// together, and the decoder still carries the full apparatus for it: sorting
// as many runs as there are rows and probing a table that large once per row.
inline bool groupsEnoughToKey(
    const std::vector<uint64_t>& key,
    int boundBits,
    size_t* distinctOut = nullptr) {
  // Below this, a run averages fewer than four rows and there is little to
  // gather.
  constexpr size_t kMinRowsPerRun = 4;
  if (key.empty()) {
    return false;
  }
  const size_t rowCount = key.size();

  // Counted exactly rather than estimated from a sample: cardinality cannot
  // be estimated from a small sample within a constant factor, so a sampled
  // count would misjudge which columns should be refused.
  //
  // Counted only up to the point where the answer stops being in doubt: the
  // test is distinct * kMinRowsPerRun <= rowCount, so a key is refused once
  // its distinct count passes a quarter of the rows, and nothing after that
  // changes the outcome. This matters because the keys that would take
  // longest to count fully are exactly the ones this early exit refuses.
  const size_t distinctLimit = rowCount / kMinRowsPerRun;

  // One bit per value the section can hold: no hashing, and stays cache
  // resident for a key narrow enough to be worth keying on. Only used while
  // the bitmap costs no more bytes than the key has rows.
  const auto bitmapAffordable = [rowCount](int bits) {
    return bits < 64 && (size_t{1} << bits) <= rowCount * 8;
  };

  // The caller's bound comes free from the section's bit range. ORing the
  // values bounds them more tightly, but that is a pass over every row, so
  // it is only worth taking where it might rescue a key the caller's bound
  // would otherwise send to the hash.
  int significantBits = std::min(boundBits, 64);
  if (!bitmapAffordable(significantBits)) {
    uint64_t orOfKeys = 0;
    for (const uint64_t value : key) {
      orOfKeys |= value;
    }
    significantBits = std::bit_width(orOfKeys);
  }

  size_t distinct = 0;
  if (bitmapAffordable(significantBits)) {
    std::vector<uint64_t> seen(
        ((size_t{1} << significantBits) + 63) / 64, uint64_t{0});
    for (const uint64_t value : key) {
      uint64_t& word = seen[value >> 6];
      const uint64_t bit = uint64_t{1} << (value & 63);
      if ((word & bit) == 0) {
        word |= bit;
        if (++distinct > distinctLimit) {
          return false;
        }
      }
    }
    if (distinctOut != nullptr) {
      *distinctOut = distinct;
    }
    return true;
  }

  // Wide keys still hash, but the set is reserved for what the early exit
  // allows rather than for the whole column, and it stops at the same point.
  folly::F14FastSet<uint64_t> seen;
  seen.reserve(std::min(rowCount, distinctLimit + 1));
  for (const uint64_t value : key) {
    if (seen.insert(value).second && ++distinct > distinctLimit) {
      return false;
    }
  }
  // Exact on every path that reaches here: the early exits above return
  // false the moment the count passes the bound, so an admitted key was
  // counted to completion.
  if (distinctOut != nullptr) {
    *distinctOut = distinct;
  }
  return true;
}

template <typename T>
std::string_view SubIntSplitEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options,
    const subintsplit::TuningConfig& tuning) {
  if (!options.subIntSplitDeltaPreTransform || values.size() < 2) {
    return encodeValues(selection, values, buffer, options, tuning);
  }
  // Encode both forms and keep the smaller, so the pre-transform can never
  // regress a stream it does not suit -- interleaved counters, for instance,
  // are worse under delta because interleaved shards break monotonicity.
  const std::string_view plain =
      encodeValues(selection, values, buffer, options, tuning);
  Vector<physicalType> residuals{&buffer.getMemoryPool(), values.size()};
  subintsplit::encodeDeltas<physicalType>(
      values, {residuals.data(), residuals.size()});
  // Deltas are undone by a running sum over every earlier row, which neither a
  // frame nor a section transform can sit beneath, so the delta form plans its
  // sections over the residuals alone.
  auto deltaTuning = tuning;
  deltaTuning.autoTransform = false;
  deltaTuning.forceApply = false;
  deltaTuning.transform = static_cast<uint8_t>(subintsplit::TransformId::None);
  const std::string_view delta = encodeResiduals(
      selection,
      std::span<const physicalType>(residuals.data(), residuals.size()),
      buffer,
      options,
      deltaTuning,
      subintsplit::RowFrame{},
      nullptr,
      subintsplit::kFlagDelta);
  return delta.size() < plain.size() ? delta : plain;
}

template <typename T>
std::string_view SubIntSplitEncoding<T>::encodeValues(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options,
    const subintsplit::TuningConfig& tuning) {
  if (!tuning.rowFrame) {
    return encodeResiduals(selection, values, buffer, options, tuning);
  }
  constexpr int kBits = static_cast<int>(sizeof(physicalType) * 8);
  // Fitted in place first, so a column that follows no line, which is most of
  // them, is not copied just to learn that.
  const auto lineFrame = subintsplit::fitRowFrame<physicalType>(values);
  // Whether the frame came from adjacent steps rather than from a line
  // through the stream, which decides how it is priced below.
  const bool stepFrame = !lineFrame.active();
  const auto rowFrame =
      stepFrame ? subintsplit::fitStepFrame<physicalType>(values) : lineFrame;
  if (!rowFrame.active()) {
    return encodeResiduals(selection, values, buffer, options, tuning);
  }
  std::vector<physicalType> residuals;
  subintsplit::subtractRowFrame<physicalType>(rowFrame, values, residuals);

  // A preserve-mode encode replays boundaries without running the planner,
  // so it has nothing to price a frame with. It takes one exactly when the
  // captured layout carried one, since those boundaries were planned on that
  // stream's residuals or values specifically.
  const auto modeConfig =
      selection.getConfig(std::string(subintsplit::kSplitModeConfigKey));
  if (modeConfig.has_value() &&
      *modeConfig == subintsplit::kSplitModePreserve) {
    const auto frameConfig =
        selection.getConfig(std::string(subintsplit::kRowFrameConfigKey));
    const bool captured = frameConfig.has_value() &&
        *frameConfig == subintsplit::kRowFramePresent;
    return captured
        ? encodeResiduals(
              selection, residuals, buffer, options, tuning, rowFrame)
        : encodeResiduals(selection, values, buffer, options, tuning);
  }

  if (tuning.rowFrameForceApply) {
    return encodeResiduals(
        selection, residuals, buffer, options, tuning, rowFrame);
  }

  // A step frame produces runs of whole values, which the planner's run
  // models can misprice, so the two streams are first compared on the
  // whole-value quotes the floor uses, from a sample. Where one is decisively
  // cheaper only it is planned and encoded, and the residuals are still
  // offered to the floor as one whole-value section when the values are
  // planned. Encoding both in full on every such column is several times the
  // base encode time.
  if (stepFrame) {
    auto sectionPolicy = std::unique_ptr<EncodingSelectionPolicy<physicalType>>(
        static_cast<EncodingSelectionPolicy<physicalType>*>(
            selection
                .template createNestedPolicy<physicalType>(
                    selection.encodingType(), NestedEncodingIdentifier{0})
                .release()));
    const auto sectionOptions =
        subintsplit::sectionEncodingOptions(options, tuning);
    const auto framedQuote =
        sampleWholeValue(*sectionPolicy, residuals, sectionOptions);
    const auto valuesQuote =
        sampleWholeValue(*sectionPolicy, values, sectionOptions);
    // Where the quotes are within a factor of two of each other the sample
    // cannot tell the streams apart, and both are encoded, into a scratch
    // buffer so the loser takes no space in the stream. Deciding within that
    // factor risks a measurable loss on run-heavy values.
    constexpr double kDecisiveQuoteRatio{2.0};
    const bool decisive = framedQuote.has_value() && valuesQuote.has_value() &&
        std::max(framedQuote->lowerBoundBytes, valuesQuote->lowerBoundBytes) >=
            kDecisiveQuoteRatio *
                std::min(
                    framedQuote->lowerBoundBytes, valuesQuote->lowerBoundBytes);
    if (!decisive) {
      Buffer scratch{buffer.getMemoryPool()};
      const auto framed = encodeResiduals(
          selection, residuals, scratch, options, tuning, rowFrame);
      const auto unframed =
          encodeResiduals(selection, values, scratch, options, tuning);
      const std::string_view smaller =
          framed.size() < unframed.size() ? framed : unframed;
      char* reserved = buffer.reserve(smaller.size());
      std::memcpy(reserved, smaller.data(), smaller.size());
      return {reserved, smaller.size()};
    }
    if (framedQuote->lowerBoundBytes < valuesQuote->lowerBoundBytes) {
      return encodeResiduals(
          selection, residuals, buffer, options, tuning, rowFrame);
    }
    return encodeResiduals(
        selection,
        values,
        buffer,
        options,
        tuning,
        subintsplit::RowFrame{},
        /*planning=*/nullptr,
        /*extraFlags=*/0,
        &rowFrame,
        residuals);
  }

  // Fitting only says the column follows a line, not that sections encode
  // the distance from it more cheaply than the values themselves. So both
  // are priced with the planner's own DP over the same inventory and
  // configuration, and the frame is kept only when its estimate, header
  // included, is smaller; the winner's grid is the one the encode plans on.
  const auto selectorConfig = plannerSelectorConfig(tuning, values.size());
  auto residualPlanning = costPlanningSample(residuals, options, tuning);
  auto valuePlanning = costPlanningSample(values, options, tuning);
  const double frameBits = subintsplit::selectSplitsOverGrid(
                               residualPlanning.grid, kBits, selectorConfig)
                               .totalSizeBits +
      8.0 * subintsplit::kRowFrameHeaderSize;
  const double valueBits = subintsplit::selectSplitsOverGrid(
                               valuePlanning.grid, kBits, selectorConfig)
                               .totalSizeBits;
  if (frameBits >= valueBits) {
    return encodeResiduals(
        selection,
        values,
        buffer,
        options,
        tuning,
        subintsplit::RowFrame{},
        &valuePlanning);
  }
  return encodeResiduals(
      selection,
      residuals,
      buffer,
      options,
      tuning,
      rowFrame,
      &residualPlanning);
}

template <typename T>
typename SubIntSplitEncoding<T>::PlanningSample
SubIntSplitEncoding<T>::costPlanningSample(
    std::span<const physicalType> values,
    const Encoding::Options& options,
    const subintsplit::TuningConfig& tuning) {
  constexpr int kBits = static_cast<int>(sizeof(physicalType) * 8);
  PlanningSample planning;
  subintsplit::sampleIntoU64<physicalType>(
      values, planning.sample, tuning.sampler);
  NIMBLE_CHECK(
      !planning.sample.empty(), "SubIntSplit planner sample is empty.");
  planning.grid = subintsplit::buildRestrictedCostGrid(
      planning.sample,
      kBits,
      values.size(),
      tuning.allowedEncodings,
      plannerSelectorConfig(tuning, values.size()));
  return planning;
}

template <typename T>
subintsplit::SelectorConfig SubIntSplitEncoding<T>::plannerSelectorConfig(
    const subintsplit::TuningConfig& tuning,
    size_t rowCount) {
  auto selectorConfig = tuning.selector;
  selectorConfig.streamRowCount = rowCount;
  // The cost models cover encodings that nestedEncodingReadFactors offers a
  // section but that EncodingSizeEstimation may not price. select() skips an
  // encoding it cannot price, so a section is never given one, and a plan
  // shaped around one is paid for with whatever selection picks instead. A
  // FrequencyPartition price, for instance, splits a correlated low bit off a
  // delta'd Snowflake residual, and the section it was priced for is then
  // stored as MainlyConstant anyway. Withheld until selection can price it,
  // which the probe below detects, so it returns once its estimator lands.
  //
  // Delta and FOR are in the same position but stay priced: withholding them
  // moves plans that currently land well, such as XMark's, where a Delta
  // price keeps a rarely changing high field out of a noisy low section.
  for (const auto type : {EncodingType::FrequencyPartition}) {
    if (!sectionSelectionEstimates(type)) {
      selectorConfig.excludedEncodings.insert(type);
    }
  }
  return selectorConfig;
}

template <typename T>
bool SubIntSplitEncoding<T>::sectionSelectionEstimates(EncodingType type) {
  // Asked of a two-row stream, as any estimator the type has answers for it.
  // Sections are estimated at their own storage width, but whether a type has
  // an estimator at all does not depend on the width.
  const std::array<physicalType, 2> probe{physicalType{0}, physicalType{1}};
  const std::span<const physicalType> values{probe.data(), probe.size()};
  return detail::EncodingSizeEstimation<physicalType>::estimateSize(
             type,
             values,
             Statistics<physicalType>::create(values),
             Encoding::Options{})
      .has_value();
}

template <typename T>
subintsplit::SamplerConfig SubIntSplitEncoding<T>::estimatorSamplerConfig() {
  // A quarter of the encoder's sample, in blocks half as long. The DP is
  // O(kBits^2 * sampleSize), so this is the term that decides what the
  // estimate costs; halving the block keeps the same number of distinct
  // stretches of the stream in a smaller sample, which is what the run-length
  // and frame-residual models in the cost grid read.
  //
  // Halving it again does not pay: fitting the row frame and drawing the
  // sample are passes over the whole column, so the DP is not all of the
  // cost, while the estimate/actual ratio worsens enough to lose selections.
  return subintsplit::SamplerConfig{.maxSamples = 512, .blockSize = 64};
}

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
template <typename T>
std::optional<uint64_t> SubIntSplitEncoding<T>::estimateSize(
    uint64_t rowCount,
    std::span<const physicalType> values,
    const Statistics<physicalType>& statistics,
    const Encoding::Options& options) {
  // Priced under the production tuning, which is what the encoder will run
  // with when selection's call reaches it.
  const auto& tuning = subintsplit::kDefaultTuningConfig;
  constexpr int kBits = static_cast<int>(sizeof(physicalType) * 8);
  // Exact, and an upper bound on what a split writes: the encoder's
  // WholeValueFloor stores the values as one FixedBitWidth section rather than
  // let a plan come in above it. Every path below therefore falls back to it
  // rather than to nullopt, so that a stream a plan cannot be costed for is
  // still costed as the split that would be written for it.
  const uint64_t fixedBitWidthEstimate =
      FixedBitWidthEncoding<physicalType>::estimateSize(
          rowCount, statistics, options);
  if (values.empty()) {
    return fixedBitWidthEstimate;
  }

  // A split is ranked here on uncompressed bytes, and where a substream
  // compressor follows the encode those are not the bytes on disk. See
  // subintsplit::Options::estimateCompressionGuard. Declined rather than
  // priced at the floor: SubIntSplit's read factor can be below
  // FixedBitWidth's, so a split priced at the floor would still win.
  if (options.subIntSplit.substreamCompression &&
      options.subIntSplit.estimateCompressionGuard) {
    return std::nullopt;
  }

  std::vector<uint64_t> samples;
  std::vector<size_t> sampleRows;
  subintsplit::sampleIntoU64WithRows<physicalType>(
      values, samples, &sampleRows, estimatorSamplerConfig());
  if (samples.empty()) {
    return fixedBitWidthEstimate;
  }

  // Priced on size alone. The planner's objective carries a per-boundary
  // penalty and, when a caller asks for it, a decode term; neither is bytes on
  // disk, and selection is comparing bytes here.
  auto selectorConfig = plannerSelectorConfig(tuning, rowCount);
  selectorConfig.decodeWeighting = subintsplit::DecodeCostWeighting{};
  const auto planBytes =
      [&](const std::vector<uint64_t>& planSamples) -> std::optional<uint64_t> {
    const auto plan = subintsplit::selectSplitsRestricted(
        planSamples, kBits, rowCount, tuning.allowedEncodings, selectorConfig);
    if (!std::isfinite(plan.totalSizeBits) || plan.sections.empty()) {
      return std::nullopt;
    }
    return static_cast<uint64_t>(std::ceil(plan.totalSizeBits / 8.0)) +
        kOuterOverheadBytes + plan.sections.size() * kPerSectionOverheadBytes;
  };

  uint64_t estimated = fixedBitWidthEstimate;
  if (const auto valueBytes = planBytes(samples)) {
    estimated = std::min(estimated, *valueBytes);
  }

  // The encoder fits a row frame and plans over the residuals as well as
  // the values, keeping whichever costs less. Estimating only the values
  // can put a monotone column far above what the encoder actually writes
  // for it, because block-stratified sampling breaks monotonicity and the
  // models see the jumps between blocks rather than the line through them.
  // The frame is fitted over the whole column, as the encoder fits it, and
  // then subtracted from the sample alone, which is all the DP reads.
  if (tuning.rowFrame) {
    auto frame = subintsplit::fitRowFrame<physicalType>(values);
    if (!frame.active()) {
      frame = subintsplit::fitStepFrame<physicalType>(values);
    }
    if (frame.active()) {
      constexpr uint64_t kWidthMask =
          kBits >= 64 ? ~uint64_t{0} : (uint64_t{2} << (kBits - 1)) - 1;
      std::vector<uint64_t> residualSamples(samples.size());
      for (size_t i = 0; i < samples.size(); ++i) {
        residualSamples[i] =
            (samples[i] - (frame.slope * sampleRows[i] + frame.base)) &
            kWidthMask;
      }
      if (const auto residualBytes = planBytes(residualSamples)) {
        estimated = std::min(
            estimated, *residualBytes + subintsplit::kRowFrameHeaderSize);
      }
    }
  }
  return estimated;
}

template <typename T>
std::optional<uint64_t> SubIntSplitEncoding<T>::estimateSizeLowerBound(
    std::span<const physicalType> values,
    const Statistics<physicalType>& statistics,
    const Encoding::Options& options) {
  if (!options.subIntSplit.estimateBitFlipScreen || values.size() < 2) {
    return std::nullopt;
  }
  // The gradient gate only, whatever admission mode the caller runs: the
  // entropy guard's whole-stream varying-bit pass costs more than the bound
  // saves on the columns it would change.
  const auto profile = subintsplit::bitFlipAdmissionProfile(
      values,
      subintsplit::SubIntSplitAdmission::kBitFlip,
      options.subIntSplit.admissionProfilePairs);
  if (subintsplit::bitFlipGradientGate(
          profile, subintsplit::TopLevelPolicyConfig{})) {
    return std::nullopt;
  }
  return FixedBitWidthEncoding<physicalType>::estimateSize(
      values.size(), statistics, options);
}
#endif

template <typename T>
std::string_view SubIntSplitEncoding<T>::encodeResiduals(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options,
    const subintsplit::TuningConfig& tuning,
    const subintsplit::RowFrame& rowFrame,
    PlanningSample* planning,
    uint8_t extraFlags,
    const subintsplit::RowFrame* stepFrame,
    std::span<const physicalType> stepResiduals) {
  // The frame the stream records: `rowFrame`, unless the floor stores the step
  // frame's residuals instead.
  subintsplit::RowFrame writtenFrame = rowFrame;
  const bool useVarint = options.useVarintRowCount;
  const uint32_t valueCount = static_cast<uint32_t>(values.size());

  if (values.empty()) {
    NIMBLE_INCOMPATIBLE_ENCODING("SubIntSplitEncoding cannot be empty.");
  }

  constexpr int kBits = static_cast<int>(sizeof(physicalType) * 8);

  std::vector<subintsplit::SectionPlan> segments;
  // What the planner thinks its own plan stores the column in. Infinite for a
  // replayed layout, which was not planned here and so was never priced. Read
  // only to decide whether the whole-value fallback is worth pricing before
  // the plan is encoded; nothing is chosen on it.
  double planEstimatedBits{std::numeric_limits<double>::infinity()};
  const auto modeConfig =
      selection.getConfig(std::string(subintsplit::kSplitModeConfigKey));
  if (modeConfig.has_value() &&
      *modeConfig == subintsplit::kSplitModePreserve) {
    const auto boundaryConfig = selection.getConfig(
        std::string(subintsplit::kSplitBoundariesConfigKey));
    NIMBLE_CHECK(
        boundaryConfig.has_value(),
        "SubIntSplit preserve mode requires boundaries config.");
    auto parsed = subintsplit::parseSplitBoundaries(*boundaryConfig, kBits);
    NIMBLE_CHECK(parsed.has_value(), "Invalid SubIntSplit boundaries config.");
    segments = std::move(parsed.value());
  } else {
    // Default behavior: recompute the split boundaries from the sampled data.
    std::vector<uint64_t> sampleBuf;
    if (planning != nullptr) {
      sampleBuf = std::move(planning->sample);
    } else {
      subintsplit::sampleIntoU64<physicalType>(
          values, sampleBuf, tuning.sampler);
    }

    // An empty allowed set costs every encoding, so this is the production
    // path unless a caller has deliberately narrowed the inventory.
    const auto selectorConfig = plannerSelectorConfig(tuning, valueCount);
    // The hybrid planner replaces the DP's argmin with a shortlist re-priced
    // by section selection's own estimators and refined, bounded on size by
    // the same cap. See subintsplit::TuningConfig::hybridPlanner. It costs
    // the grid once and takes the DP's own plan as the first of its shortlist,
    // so it replaces the call below rather than adding to it.
    bool planned = false;
    if constexpr (
        std::is_same_v<physicalType, uint32_t> ||
        std::is_same_v<physicalType, uint64_t>) {
      if (tuning.hybridPlanner) {
        // Bit-flip gradient boundaries nominate plans and bound where
        // refinement splits a segment. They do not constrain the DP itself,
        // since doing so measurably worsened plan quality.
        const auto profile = subintsplit::computeBitFlipProfile<uint64_t>(
            std::span<const uint64_t>(sampleBuf.data(), sampleBuf.size()));
        std::vector<bool> cuts(kBits + 1, false);
        cuts[0] = true;
        cuts[kBits] = true;
        for (const int boundary : subintsplit::bitFlipGradientBoundaries(
                 profile, subintsplit::TopLevelPolicyConfig{})) {
          if (boundary > 0 && boundary < kBits) {
            cuts[boundary] = true;
          }
        }
        const auto shortlist = planning != nullptr
            ? subintsplit::shortlistSplitsOverGrid(
                  planning->grid,
                  kBits,
                  selectorConfig,
                  subintsplit::kHybridShortlist,
                  cuts)
            : subintsplit::shortlistSplitsRestricted(
                  sampleBuf,
                  kBits,
                  valueCount,
                  tuning.allowedEncodings,
                  selectorConfig,
                  subintsplit::kHybridShortlist,
                  cuts);
        auto refined =
            subintsplit::SubIntSplitPlanRefiner::refine<physicalType>(
                values,
                kBits,
                shortlist,
                cuts,
                selectorConfig,
                subintsplit::sectionEncodingOptions(options, tuning));
        if (!refined.sections.empty()) {
          segments = std::move(refined.sections);
          planEstimatedBits = refined.sizeBits;
          planned = true;
        }
      }
    }

    if (!planned) {
      auto selectorResult = planning != nullptr
          ? subintsplit::selectSplitsOverGrid(
                planning->grid, kBits, selectorConfig)
          : subintsplit::selectSplitsRestricted(
                sampleBuf,
                kBits,
                valueCount,
                tuning.allowedEncodings,
                selectorConfig);

      // What the weighted plan gave up in bytes, bounded against what size
      // alone would have stored the column in. The DP minimises size plus a
      // decode term in the same units, so on a column with structure it will
      // keep buying decode with bytes for as long as the weight makes that
      // arithmetic work, and there is no point at which it stops on its own.
      //
      // Costed rather than estimated: the size-only plan is a second run of the
      // same DP over the same sample, which is the only way to know what was
      // given up, since the weighted plan's own totalSizeBits says what it
      // stores and not what it could have stored. Paid only when the weight is
      // on, and the sample is the same one already extracted.
      if (selectorConfig.decodeWeighting.weight != 0.0) {
        auto sizeOnlyConfig = selectorConfig;
        sizeOnlyConfig.decodeWeighting = subintsplit::DecodeCostWeighting{};
        auto sizeOnly = subintsplit::selectSplitsRestricted(
            sampleBuf,
            kBits,
            valueCount,
            tuning.allowedEncodings,
            sizeOnlyConfig);
        // Compared on estimated size alone, not on totalCost: bytes are what is
        // being bounded, and totalCost is the objective that has just been
        // shown not to bound them.
        const double allowedSizeBits =
            sizeOnly.totalSizeBits * (1.0 + tuning.maxSizeRegression);
        if (selectorResult.totalSizeBits > allowedSizeBits) {
          selectorResult = std::move(sizeOnly);
        }
      }

      planEstimatedBits = selectorResult.totalSizeBits;
      segments = std::move(selectorResult.sections);
    }
  }

  NIMBLE_CHECK(
      !segments.empty(), "SubIntSplitEncoding: selector returned no segments");
  uint8_t splitCount = static_cast<uint8_t>(segments.size());

  // Encode each section into a temporary buffer.
  // Each section is encoded as the narrowest unsigned integer type that fits
  // its bit width, so narrow sections don't pay an 8-byte-per-value penalty.
  auto* pool = &buffer.getMemoryPool();
  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  Buffer& sectionBuffer = scopedBuffer.get();
  auto* sectionPool = &sectionBuffer.getMemoryPool();
  std::vector<std::string_view> sectionData;
  sectionData.reserve(splitCount);

  // What these options override, and why, is documented on
  // sectionEncodingOptions. Derived there rather than here so that a driver
  // measuring section costs can ask for the same options instead of restating
  // them.
  const Encoding::Options sectionOptions =
      subintsplit::sectionEncodingOptions(options, tuning);

  // A replayed layout is an instruction, not a plan, so it is left alone.
  const bool replayed =
      modeConfig.has_value() && *modeConfig == subintsplit::kSplitModePreserve;
  const uint64_t singleSectionHeader = subintsplit::specificHeaderSize(1);
  // With the decode weight on, the plan may exceed the floor by as much as it
  // may exceed the size-only plan, and no more.
  const double allowedRegression = tuning.selector.decodeWeighting.weight != 0.0
      ? 1.0 + tuning.maxSizeRegression
      : 1.0;

  // Write context:
  //
  //   full values + SectionPlan -> [extract narrow values] -> nested bytes
  //
  // Each child receives one right-aligned value per input row. Its storage
  // type is the narrowest unsigned type that fits the section, while exact
  // bit width and planner exclusions flow into nested selection.
  //
  // Encodes one section at its storage width, reading row i's section value
  // from sectionValueAt(i).
  const auto encodeSectionInto = [&](uint8_t s,
                                     uint8_t storageBytes,
                                     const auto& sectionValueAt,
                                     Buffer& targetBuffer,
                                     const Encoding::Options& targetOptions) {
    std::string_view encoded;
    switch (storageBytes) {
      case 1: {
        Vector<uint8_t> sectionValues{sectionPool, valueCount};
        for (uint32_t i = 0; i < valueCount; ++i) {
          sectionValues[i] = static_cast<uint8_t>(sectionValueAt(i));
        }
        encoded = selection.template encodeNested<uint8_t>(
            static_cast<NestedEncodingIdentifier>(s),
            std::span<const uint8_t>(
                sectionValues.data(), sectionValues.size()),
            targetBuffer,
            targetOptions);
        break;
      }
      case 2: {
        Vector<uint16_t> sectionValues{sectionPool, valueCount};
        for (uint32_t i = 0; i < valueCount; ++i) {
          sectionValues[i] = static_cast<uint16_t>(sectionValueAt(i));
        }
        encoded = selection.template encodeNested<uint16_t>(
            static_cast<NestedEncodingIdentifier>(s),
            std::span<const uint16_t>(
                sectionValues.data(), sectionValues.size()),
            targetBuffer,
            targetOptions);
        break;
      }
      case 4: {
        Vector<uint32_t> sectionValues{sectionPool, valueCount};
        for (uint32_t i = 0; i < valueCount; ++i) {
          sectionValues[i] = static_cast<uint32_t>(sectionValueAt(i));
        }
        encoded = selection.template encodeNested<uint32_t>(
            static_cast<NestedEncodingIdentifier>(s),
            std::span<const uint32_t>(
                sectionValues.data(), sectionValues.size()),
            targetBuffer,
            targetOptions);
        break;
      }
      case 8: {
        Vector<uint64_t> sectionValues{sectionPool, valueCount};
        for (uint32_t i = 0; i < valueCount; ++i) {
          sectionValues[i] = sectionValueAt(i);
        }
        encoded = selection.template encodeNested<uint64_t>(
            static_cast<NestedEncodingIdentifier>(s),
            std::span<const uint64_t>(
                sectionValues.data(), sectionValues.size()),
            targetBuffer,
            targetOptions);
        break;
      }
      default: {
        NIMBLE_UNREACHABLE("Invalid SubIntSplit section storage width.");
      }
    }
    return encoded;
  };
  const auto encodeSectionFrom =
      [&](uint8_t s, uint8_t storageBytes, const auto& sectionValueAt) {
        return encodeSectionInto(
            s, storageBytes, sectionValueAt, sectionBuffer, sectionOptions);
      };
  // A section is extracted once into 64-bit form, transformed there, and only
  // then narrowed to its storage width, so a transform never has to know which
  // width it is working in.
  const auto extractSection = [&](const auto& seg) {
    const int width = seg.bitEnd - seg.bitStart + 1;
    const uint64_t mask = subintsplit::widthMask(width);
    // Appended rather than sized and then overwritten. Sizing it first would
    // clear a buffer as long as the column, once per section, that the loop
    // below writes over completely.
    std::vector<uint64_t> out;
    out.reserve(valueCount);
    for (uint32_t i = 0; i < valueCount; ++i) {
      uint64_t v = 0;
      __builtin_memcpy(&v, &values[i], sizeof(physicalType));
      out.push_back((v >> seg.bitStart) & mask);
    }
    return out;
  };
  const auto encodeSection = [&](uint8_t s,
                                 uint8_t storageBytes,
                                 const std::vector<uint64_t>& sectionU64) {
    return encodeSectionFrom(
        s, storageBytes, [&sectionU64](uint32_t i) { return sectionU64[i]; });
  };

  const auto requestedTransform =
      static_cast<subintsplit::TransformId>(tuning.transform);
  const auto* transform = subintsplit::transformFor(requestedTransform);
  const uint8_t keySection = tuning.keySection;

  // Transforms the per-section search may choose between when the caller asks
  // for selection rather than naming one.
  //
  // There is one row order per stream -- every section is a bit-slice of the
  // same rows -- so the key-derived permutation is built once per candidate
  // key and sections opt into it individually.
  std::vector<const subintsplit::SectionTransform*> candidates;
  if (tuning.autoTransform) {
    NIMBLE_CHECK(
        !tuning.forceApply,
        "subIntSplit.autoTransform and subIntSplit.forceApply are exclusive: "
        "one asks the encoder to choose, the other to obey.");
    candidates.push_back(
        subintsplit::transformFor(subintsplit::TransformId::KeyDerived));
  } else if (transform != nullptr) {
    candidates.push_back(transform);
  }
  if (tuning.forceApply) {
    NIMBLE_CHECK_NOT_NULL(
        transform, "subIntSplit.forceApply requires a real transform.");
  }
  // Whether any candidate needs a key decides if the key search runs at all.
  const bool anyCandidateNeedsKey = std::any_of(
      candidates.begin(),
      candidates.end(),
      [](const subintsplit::SectionTransform* candidate) {
        return candidate->needsKeySection();
      });

  // Rewrites one section with the transform. A transformed section costs one
  // id byte on the wire; the margin it has to clear is larger, so that a
  // transform has to save more than noise to be kept.
  constexpr size_t kTransformMarginBytes{8};
  const auto applyTransform =
      [&](const subintsplit::SectionTransform* candidate,
          int width,
          const std::vector<uint64_t>& keyValues,
          std::span<const uint32_t> keyOrder,
          std::vector<uint64_t>& sectionU64) {
        subintsplit::TransformState state;
        subintsplit::TransformContext context{
            .keySection = keyValues, .width = width, .keyOrder = keyOrder};
        candidate->apply(sectionU64, context, state);
        NIMBLE_CHECK(
            state.codebook.empty(),
            "A section transform has no wire field for its state.");
      };

  // Neither a section's extracted values nor its untransformed encoding
  // depends on which section is being tried as the key, so both are done
  // once per section here rather than once per candidate key, which would
  // make encode quadratic in the split count. With no transform to price,
  // nothing reads a section's 64-bit form after its plain encode, so it is
  // sliced straight into its storage width instead.

  // A plan the whole value beats is encoded and then thrown away, which for
  // some columns is the dominant cost of running SubIntSplit at all. So
  // where the planner's own estimate is already above what a sample quotes
  // one whole-value section at, the fallback is priced first and the plan's
  // sections are then encoded against what the fallback really costs: the
  // section that takes the plan past it ends the plan.
  //
  // This reorders the work and nothing else: the fallback still has to beat
  // the plan on encoded bytes, WholeValueFloor::under applies the same tests
  // whichever order candidates were priced in, and a plan is only abandoned
  // once the bytes already written put the comparison beyond doubt. Only a
  // multi-section plan with no transform to search and no step frame to
  // offer qualifies, since the bytes a section has already cost bound the
  // plan from below only while no transform can replace that section with a
  // smaller one.
  std::optional<WholeValueFloor> valuesFloor;
  std::optional<std::string_view> earlyFloor;
  if (!replayed && stepFrame == nullptr && splitCount > 1 &&
      candidates.empty()) {
    valuesFloor.emplace(
        selection,
        values,
        sectionBuffer,
        sectionOptions,
        /*planIsWholeValue=*/false,
        /*valuesAreColumn=*/!rowFrame.active());
    // Reordering the plan's encode around the fallback is not free: sections
    // stop being encoded concurrently, since a plan cannot be abandoned
    // partway while every part is already in flight. So the quote must be
    // far enough under the plan's estimate that the plan is expected to be
    // abandoned after its first section or two; a quote merely below the
    // estimate risks paying for the plan serially and still finishing it.
    constexpr double kDecisiveQuoteRatio{2.0};
    const double planEstimatedBytes = planEstimatedBits / 8.0;
    if (valuesFloor->usable() && valuesFloor->quote().has_value() &&
        kDecisiveQuoteRatio * valuesFloor->quote()->lowerBoundBytes <
            planEstimatedBytes &&
        planEstimatedBytes > static_cast<double>(singleSectionHeader)) {
      earlyFloor = valuesFloor->under(
          static_cast<uint64_t>(planEstimatedBytes / allowedRegression) -
              singleSectionHeader,
          /*replayingEncoding=*/std::nullopt);
    }
  }
  // Plan bytes past which the fallback is certain to win, so that the plan can
  // be abandoned at the first section that reaches them. Never reached when
  // there is no fallback to abandon it for.
  const uint64_t planAbandonBytes = earlyFloor.has_value()
      ? static_cast<uint64_t>(std::ceil(
            static_cast<double>(
                valuesFloor->decidedAbove() + singleSectionHeader) *
            allowedRegression))
      : std::numeric_limits<uint64_t>::max();
  std::vector<std::vector<uint64_t>> sectionValues64(splitCount);
  std::vector<uint8_t> sectionStorage(splitCount);
  std::vector<std::string_view> plainEncoded(splitCount);
  // Each concurrent section writes into a buffer of its own, which outlives
  // the loop because plainEncoded views into it, and draws no scratch from the
  // encoding buffer pool, which is not thread-safe.
  std::vector<std::unique_ptr<Buffer>> concurrentBuffers;
  // Set once the sections encoded so far already cost more than the fallback,
  // which is as much of the plan as there is any reason to encode.
  bool planAbandoned = false;
  // Whatever the sections encoded so far cost, plus the header every plan
  // carries. A lower bound on the plan, since no section still to come can
  // take bytes away from it, which is what makes abandoning on it sound.
  uint64_t planBytesSoFar = subintsplit::specificHeaderSize(splitCount);
  // The one section a plan with a priced fallback encodes before the rest.
  // Abandoning needs a section's real bytes, and a plan whose every section is
  // already in flight cannot be abandoned at all, so the widest one -- the
  // likeliest to take the plan past the fallback on its own -- is encoded
  // first and the rest still go to the executor. A plan that survives the
  // probe pays for one section serially instead of for all of them.
  constexpr uint8_t kNoProbeSection = 0xFF;
  uint8_t probeSection = kNoProbeSection;
  if (earlyFloor.has_value() && splitCount > 1) {
    probeSection = 0;
    for (uint8_t s = 1; s < splitCount; ++s) {
      if (segments[s].bitEnd - segments[s].bitStart >
          segments[probeSection].bitEnd - segments[probeSection].bitStart) {
        probeSection = s;
      }
    }
    const auto& seg = segments[probeSection];
    const int width = seg.bitEnd - seg.bitStart + 1;
    sectionStorage[probeSection] = sectionStorageBytes(width);
    const uint64_t mask = subintsplit::widthMask(width);
    plainEncoded[probeSection] = encodeSectionFrom(
        probeSection,
        sectionStorage[probeSection],
        [&values, &seg, mask](uint32_t i) {
          uint64_t value = 0;
          __builtin_memcpy(&value, &values[i], sizeof(physicalType));
          return (value >> seg.bitStart) & mask;
        });
    planBytesSoFar += plainEncoded[probeSection].size();
    planAbandoned = planBytesSoFar >= planAbandonBytes;
  }
  // Transformed attempts read each section's 64-bit form, which only the
  // serial loop below extracts.
  if (!planAbandoned && candidates.empty() &&
      tuning.sectionExecutor != nullptr && splitCount > 1) {
    Encoding::Options concurrentOptions = sectionOptions;
    concurrentOptions.encodingBufferPool = nullptr;
    concurrentBuffers.resize(splitCount);
    std::vector<std::exception_ptr> failures(splitCount);
    std::latch remaining(
        splitCount - (probeSection == kNoProbeSection ? 0 : 1));
    for (uint8_t s = 0; s < splitCount; ++s) {
      if (s == probeSection) {
        continue;
      }
      const int width = segments[s].bitEnd - segments[s].bitStart + 1;
      sectionStorage[s] = sectionStorageBytes(width);
      concurrentBuffers[s] = std::make_unique<Buffer>(*sectionPool);
      tuning.sectionExecutor->add([&, s, width]() {
        try {
          const auto& segment = segments[s];
          const uint64_t mask = subintsplit::widthMask(width);
          plainEncoded[s] = encodeSectionInto(
              s,
              sectionStorage[s],
              [&values, &segment, mask](uint32_t i) {
                uint64_t value = 0;
                __builtin_memcpy(&value, &values[i], sizeof(physicalType));
                return (value >> segment.bitStart) & mask;
              },
              *concurrentBuffers[s],
              concurrentOptions);
        } catch (...) {
          failures[s] = std::current_exception();
        }
        remaining.count_down();
      });
    }
    remaining.wait();
    for (const auto& failure : failures) {
      if (failure) {
        std::rethrow_exception(failure);
      }
    }
  }
  for (uint8_t s = 0; s < splitCount && concurrentBuffers.empty(); ++s) {
    if (s == probeSection) {
      continue;
    }
    if (planAbandoned || planBytesSoFar >= planAbandonBytes) {
      planAbandoned = true;
      break;
    }
    const auto& seg = segments[s];
    const int width = seg.bitEnd - seg.bitStart + 1;
    sectionStorage[s] = sectionStorageBytes(width);
    if (candidates.empty()) {
      const uint64_t mask = subintsplit::widthMask(width);
      plainEncoded[s] = encodeSectionFrom(
          s, sectionStorage[s], [&values, &seg, mask](uint32_t i) {
            uint64_t value = 0;
            __builtin_memcpy(&value, &values[i], sizeof(physicalType));
            return (value >> seg.bitStart) & mask;
          });
      planBytesSoFar += plainEncoded[s].size();
      continue;
    }
    sectionValues64[s] = extractSection(seg);
    plainEncoded[s] = encodeSection(s, sectionStorage[s], sectionValues64[s]);
    planBytesSoFar += plainEncoded[s].size();
  }
  if (planBytesSoFar >= planAbandonBytes) {
    planAbandoned = true;
  }

  // One choice of key section, priced. kNoKeySection means the transform does
  // not use a key, in which case every section is a candidate to transform.
  struct Attempt {
    std::vector<std::string_view> sections;
    subintsplit::TransformInfo info;
    size_t totalBytes{0};
    // What the search minimises: the attempt's bytes plus the size-equivalent
    // of the decode a row-permuting transform adds. Equal to totalBytes at the
    // default decode weight of zero, so the plan chosen is the one size alone
    // would choose unless a caller asks for decode to count.
    size_t costBytes{0};
  };
  // Returns nothing when the plan being built has already grown past
  // `bound`: a plan only grows as sections are added, so it cannot come back
  // under the bound, and improvesOnBest is the same test the finished plan
  // would face, so abandoning here is not a heuristic.
  const auto attemptWithKey = [&](uint8_t candidateKey,
                                  size_t bound) -> std::optional<Attempt> {
    Attempt attempt;
    attempt.sections.assign(splitCount, std::string_view{});
    attempt.info.transformIds.assign(splitCount, 0);
    attempt.info.keySection = subintsplit::TransformInfo::kNoKeySection;

    const std::vector<uint64_t> noKey;
    const bool hasKey =
        candidateKey != subintsplit::TransformInfo::kNoKeySection;
    const std::vector<uint64_t>& keyValues =
        hasKey ? sectionValues64[candidateKey] : noKey;
    bool keyGroups = true;
    // Distinct values in the candidate key, which is what the reader's merge
    // rotates through and so what the transform costs per row to undo. Zero
    // where there is no key, where the penalty is not charged at all.
    size_t keyDistinct = 0;
    if (hasKey) {
      attempt.info.keySection = candidateKey;
      const auto& keySegment = segments[candidateKey];
      keyGroups = groupsEnoughToKey(
          keyValues, keySegment.bitEnd - keySegment.bitStart + 1, &keyDistinct);
    }

    // Priced per candidate key rather than once for the column, since two
    // keys over the same sections can cost the reader very differently.
    // Zero at the default weight, leaving the transform chosen on size.
    const size_t keyDerivedDecodePenaltyBytes = static_cast<size_t>(
        subintsplit::decodeCostBits(
            subintsplit::keyDerivedTransformNanosPerRow(keyDistinct),
            valueCount,
            tuning.selector.decodeWeighting.weight) /
        8.0);

    // The permutation a key-derived transform gathers by depends only on the
    // candidate key, so it is built once here instead of once per section.
    // Local to the attempt on purpose: a permutation left over from a
    // previous candidate key would reorder rows by a key the stream does not
    // name.
    std::vector<uint32_t> keyPermutation;
    if (hasKey && anyCandidateNeedsKey && (keyGroups || tuning.forceApply)) {
      keyPermutation = subintsplit::buildKeyOrder(keyValues);
    }

    // A section that cannot be transformed contributes its plain size
    // regardless, so those are settled first and already in the running
    // total before any transform is priced against the bound. The key
    // section is always one of them, since it rebuilds the order of the
    // others and so is never itself transformed.
    std::vector<uint8_t> transformable;
    transformable.reserve(splitCount);
    for (uint8_t s = 0; s < splitCount; ++s) {
      const bool mayTransform = !candidates.empty() && s != candidateKey &&
          (keyGroups || tuning.forceApply);
      if (mayTransform) {
        transformable.push_back(s);
        continue;
      }
      attempt.sections[s] = plainEncoded[s];
      attempt.totalBytes += plainEncoded[s].size();
      attempt.costBytes += plainEncoded[s].size();
    }
    if (!improvesOnBest(attempt.costBytes, bound)) {
      return std::nullopt;
    }

    // Biggest plain section first, so the running total climbs toward the
    // bound fastest and a losing candidate is abandoned after fewer encodes.
    // Results are written at their section's index, so this order only
    // decides how soon the search gives up, never what a kept candidate is
    // made of.
    std::sort(
        transformable.begin(),
        transformable.end(),
        [&plainEncoded](uint8_t a, uint8_t b) {
          const size_t sizeA = plainEncoded[a].size();
          const size_t sizeB = plainEncoded[b].size();
          return sizeA != sizeB ? sizeA > sizeB : a < b;
        });

    for (const uint8_t s : transformable) {
      const auto& seg = segments[s];
      const int width = seg.bitEnd - seg.bitStart + 1;
      const std::string_view plain = plainEncoded[s];

      // Plain is the incumbent: a candidate must be strictly cheaper, margin
      // included, to displace it, so the untransformed result is the default
      // whenever a transform does not pay.
      size_t bestBytes = plain.size();
      size_t bestCost = plain.size();
      std::string_view bestEncoded = plain;
      const subintsplit::SectionTransform* bestTransform = nullptr;
      for (const auto* candidate : candidates) {
        // A key-derived candidate has nothing to gather by when this attempt
        // found no usable key.
        if (candidate->needsKeySection() && keyPermutation.empty()) {
          continue;
        }
        auto transformed = sectionValues64[s];
        applyTransform(
            candidate,
            width,
            keyValues,
            std::span<const uint32_t>(keyPermutation),
            transformed);
        const std::string_view alternative =
            encodeSection(s, sectionStorage[s], transformed);
        const size_t total = alternative.size() + kTransformMarginBytes;
        // Only a transform that keys on another section moves rows, and only
        // moving rows costs the reader the scatter back.
        const size_t cost = total +
            (candidate->needsKeySection() ? keyDerivedDecodePenaltyBytes : 0);
        if (cost < bestCost || tuning.forceApply) {
          bestBytes = total;
          bestCost = cost;
          bestEncoded = alternative;
          bestTransform = candidate;
        }
      }

      attempt.sections[s] = bestEncoded;
      attempt.totalBytes += bestBytes;
      attempt.costBytes += bestCost;
      if (bestTransform != nullptr) {
        attempt.info.transformIds[s] =
            static_cast<uint8_t>(bestTransform->id());
      }
      if (!improvesOnBest(attempt.costBytes, bound)) {
        return std::nullopt;
      }
    }
    return attempt;
  };

  // Which section to key on is a property of the data, not a constant, so
  // every section is tried and the smallest encode wins; guessing wrong would
  // blame the transform for what was really a bad guess.
  constexpr size_t kNoBound = std::numeric_limits<size_t>::max();
  std::optional<Attempt> best;
  // The smallest attempt on bytes alone, which is what the cost-minimal
  // attempt is held against. Tracked only while the decode weight is on,
  // since without it the two are the same attempt and the copy would be
  // waste.
  const bool boundTransformSize = tuning.selector.decodeWeighting.weight != 0.0;
  std::optional<Attempt> smallestAttempt;
  const auto offerToSmallest = [&](const std::optional<Attempt>& attempt) {
    if (!boundTransformSize || !attempt.has_value()) {
      return;
    }
    if (!smallestAttempt.has_value() ||
        attempt->totalBytes < smallestAttempt->totalBytes) {
      smallestAttempt = attempt;
    }
  };
  if (planAbandoned) {
    // Nothing to search: the plan these attempts would choose between has
    // already lost to the fallback on bytes already written.
  } else if (anyCandidateNeedsKey) {
    // The split is planned from the data, so a pinned key past its last
    // section names nothing and is inert: the key is searched as if unpinned.
    if (keySection < splitCount) {
      best = attemptWithKey(keySection, kNoBound);
    } else {
      // Keying on nothing is a real candidate, not the absence of one: it is
      // the untransformed plan, priced first so it becomes the bound the keyed
      // attempts have to beat. A forced search skips it, since every attempt
      // it keeps must carry the transform.
      if (!tuning.forceApply) {
        best =
            attemptWithKey(subintsplit::TransformInfo::kNoKeySection, kNoBound);
        offerToSmallest(best);
      }
      for (uint8_t candidate = 0; candidate < splitCount; ++candidate) {
        // Bounded by the incumbent, so an attempt that comes back has
        // already beaten it: anything that would not have displaced the
        // incumbent was abandoned rather than finished. Bounding on cost,
        // though, can prune an attempt that stores the column better but
        // reads it worse, which is exactly the attempt the size cap may need
        // as a fallback. So while the cap is live every candidate is priced
        // to the end.
        auto attempt = attemptWithKey(
            candidate,
            boundTransformSize
                ? kNoBound
                : (best.has_value() ? best->costBytes : kNoBound));
        if (attempt.has_value()) {
          offerToSmallest(attempt);
          if (!best.has_value() || attempt->costBytes < best->costBytes) {
            best = std::move(attempt);
          }
        }
      }
    }
  } else {
    best = attemptWithKey(subintsplit::TransformInfo::kNoKeySection, kNoBound);
  }

  // The transform is bounded on size for the same reason the split is: a
  // decode penalty charged in bytes can always be made to outweigh bytes.
  // Where the weighted search's pick stores the column worse than the
  // smallest attempt by more than the caller allowed, the smallest attempt
  // is what gets written.
  if (smallestAttempt.has_value() && best.has_value()) {
    const double allowedBytes =
        static_cast<double>(smallestAttempt->totalBytes) *
        (1.0 + tuning.maxSizeRegression);
    if (static_cast<double>(best->totalBytes) > allowedBytes) {
      best = std::move(smallestAttempt);
    }
  }

  subintsplit::TransformInfo transformInfo;
  if (!planAbandoned) {
    NIMBLE_CHECK(
        best.has_value(), "SubIntSplit transform search found no plan.");
    sectionData = std::move(best->sections);
    transformInfo = std::move(best->info);
    // A key section is only worth holding back if some other section was
    // actually keyed on it.
    if (!transformInfo.anyTransform()) {
      transformInfo.keySection = subintsplit::TransformInfo::kNoKeySection;
    }
  }

  // The plan is chosen on estimates, and a whole-value encoding it priced
  // wrongly can beat it outright, including storing the column worse than
  // raw. Holding the plan to what one whole-value section actually encodes
  // to makes that impossible. With the decode weight on, the plan may
  // exceed the floor by as much as it may exceed the size-only plan, and no
  // more.
  {
    std::optional<std::string_view> floor;
    uint64_t bytesToBeat = 0;
    if (planAbandoned) {
      // The plan crossed what the fallback costs while being encoded, and
      // decidedAbove is the point past which every candidate is admitted,
      // so the answer is just the smallest candidate: `earlyFloor`.
      floor = earlyFloor;
    } else {
      uint64_t planBytes = subintsplit::specificHeaderSize(splitCount) +
          subintsplit::transformHeaderSize(transformInfo);
      for (const auto& section : sectionData) {
        planBytes += section.size();
      }
      bytesToBeat = static_cast<uint64_t>(
          static_cast<double>(planBytes) / allowedRegression);
      if (!replayed && bytesToBeat > singleSectionHeader) {
        if (!valuesFloor.has_value()) {
          valuesFloor.emplace(
              selection,
              values,
              sectionBuffer,
              sectionOptions,
              /*planIsWholeValue=*/splitCount == 1 &&
                  !transformInfo.anyTransform(),
              /*valuesAreColumn=*/!rowFrame.active());
        }
        // A plan whose sections are constant but for one stored in Delta or
        // Varint already replays every earlier row to reach one, so the whole
        // value gives nothing up by being stored the same way, and saves the
        // constant sections and their headers.
        std::optional<EncodingType> replayingEncoding;
        if (splitCount > 1 && !transformInfo.anyTransform()) {
          size_t numVarying{0};
          EncodingType varyingEncoding{EncodingType::Constant};
          for (const auto& section : sectionData) {
            const auto encodingType = EncodingPrefix::encodingType(section);
            if (encodingType != EncodingType::Constant) {
              ++numVarying;
              varyingEncoding = encodingType;
            }
          }
          if (numVarying == 1 &&
              (varyingEncoding == EncodingType::Delta ||
               varyingEncoding == EncodingType::Varint)) {
            replayingEncoding = varyingEncoding;
          }
        }
        floor = valuesFloor->under(
            bytesToBeat - singleSectionHeader, replayingEncoding);
      }
    }
    // The step frame's residuals pay for the frame block on top of the
    // section, and have to beat whatever the values' floor already reached.
    const uint64_t framedHeader =
        singleSectionHeader + subintsplit::kRowFrameHeaderSize;
    const uint64_t framedBytesToBeat =
        floor.has_value() ? floor->size() + singleSectionHeader : bytesToBeat;
    if (!replayed && stepFrame != nullptr && framedBytesToBeat > framedHeader) {
      WholeValueFloor framedValuesFloor{
          selection,
          stepResiduals,
          sectionBuffer,
          sectionOptions,
          /*planIsWholeValue=*/false,
          /*valuesAreColumn=*/false};
      const auto framedFloor = framedValuesFloor.under(
          framedBytesToBeat - framedHeader,
          /*replayingEncoding=*/std::nullopt);
      if (framedFloor.has_value()) {
        floor = framedFloor;
        writtenFrame = *stepFrame;
      }
    }
    if (floor.has_value()) {
      segments.assign(1, {.bitStart = 0, .bitEnd = kBits - 1});
      splitCount = 1;
      sectionData.assign(1, *floor);
      transformInfo = subintsplit::TransformInfo{};
    }
    NIMBLE_CHECK(
        !sectionData.empty(),
        "SubIntSplitEncoding: a plan was abandoned with no fallback to write.");
  }

  // Write final encoding to main buffer.
  const uint32_t prefixSize =
      Encoding::serializePrefixSize(valueCount, useVarint);
  const uint32_t specificHeader = subintsplit::specificHeaderSize(splitCount) +
      subintsplit::rowFrameHeaderSize(writtenFrame) +
      subintsplit::transformHeaderSize(transformInfo);
  uint32_t sectionsSize = 0;
  for (const auto& sv : sectionData) {
    sectionsSize += static_cast<uint32_t>(sv.size());
  }
  const uint32_t encodingSize = prefixSize + specificHeader + sectionsSize;

  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;

  // A stream with no transform keeps the original encoding type and header,
  // so it stays readable by an older SubIntSplit reader. A transformed
  // stream announces a type an older reader does not know, so it fails in
  // the factory rather than decoding sections and skipping the inverse.
  const bool transformed = transformInfo.anyTransform();
  Encoding::serializePrefix(
      transformed ? EncodingType::SubIntSplitReordered
                  : EncodingType::SubIntSplit,
      TypeTraits<T>::dataType,
      valueCount,
      useVarint,
      pos);

  encoding::write<uint8_t>(splitCount, pos);
  const uint8_t flags = extraFlags |
      (writtenFrame.active() ? subintsplit::kFlagRowFrame : uint8_t{0}) |
      (transformed ? subintsplit::kFlagTransforms : uint8_t{0});
  encoding::write<uint8_t>(flags, pos);

  if (writtenFrame.active()) {
    encoding::write<uint8_t>(subintsplit::kRowFrameGuard, pos);
    encoding::write<uint64_t>(writtenFrame.slope, pos);
    encoding::write<uint64_t>(writtenFrame.base, pos);
  }

  if (transformed) {
    encoding::write<uint8_t>(transformInfo.keySection, pos);
    for (uint8_t s = 0; s < splitCount; ++s) {
      encoding::write<uint8_t>(transformInfo.transformIds[s], pos);
    }
  }

  for (uint8_t s = 0; s < splitCount; ++s) {
    const auto& seg = segments[s];
    encoding::write<uint8_t>(static_cast<uint8_t>(seg.bitStart), pos);
    encoding::write<uint8_t>(static_cast<uint8_t>(seg.bitEnd), pos);
    encoding::writeUint32(static_cast<uint32_t>(sectionData[s].size()), pos);
  }
  for (const auto& sv : sectionData) {
    encoding::writeBytes(sv, pos);
  }

  NIMBLE_DCHECK_EQ(
      static_cast<uint32_t>(pos - reserved),
      encodingSize,
      "SubIntSplitEncoding: encoding size mismatch");

  return {reserved, encodingSize};
}

template <typename T>
std::optional<typename SubIntSplitEncoding<T>::SampledWholeValue>
SubIntSplitEncoding<T>::sampleWholeValue(
    EncodingSelectionPolicy<physicalType>& sectionPolicy,
    std::span<const physicalType> values,
    const Encoding::Options& sectionOptions) {
  // Priced on contiguous blocks spread over the column rather than on the
  // whole column, since pricing distinct values and runs over every row
  // costs as much as the rest of the encode. The planner's own sample is
  // too small for these estimates, whose alphabet overhead does not scale
  // with rows.
  constexpr size_t kSampleBlocks{8};
  constexpr size_t kSampleBlockRows{8'192};
  std::vector<physicalType> sample;
  std::span<const physicalType> priced = values;
  if (values.size() > 2 * kSampleBlocks * kSampleBlockRows) {
    sample = sampleSpreadBlocks(values, kSampleBlocks, kSampleBlockRows);
    priced = sample;
  }
  const auto sampleStatistics = Statistics<physicalType>::create(priced);
  // Every encoding a section may take is priced, as section selection would
  // price the whole value, except RLE where the sample averages fewer than
  // two rows a run: without runs an RLE stream is its values stream plus
  // lengths, and its quoted slack admits trials that always lose. Narrowing
  // the candidate set further was tried and rejected: it can miss a
  // whole-value encoding that beats the plan outright.
  const bool hasRuns =
      2 * sampleStatistics.consecutiveRepeatCount() <= priced.size();
  const auto trialPolicy =
      sectionPolicy.narrowed([hasRuns](EncodingType encodingType) {
        return encodingType != EncodingType::RLE || hasRuns;
      });
  if (trialPolicy == nullptr) {
    return std::nullopt;
  }
  const auto picked =
      trialPolicy->select(priced, sampleStatistics, sectionOptions);
  if (!picked.estimatedSize.has_value()) {
    return std::nullopt;
  }
  const double estimatedBytes = static_cast<double>(*picked.estimatedSize) *
      static_cast<double>(values.size()) / static_cast<double>(priced.size());
  return SampledWholeValue{
      .encoding = picked.encodingType,
      .estimatedBytes = estimatedBytes,
      .lowerBoundBytes = estimatedBytes /
          subintsplit::wholeValueEstimateSlack(picked.encodingType)};
}

template <typename T>
SubIntSplitEncoding<T>::WholeValueFloor::WholeValueFloor(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& sectionBuffer,
    const Encoding::Options& sectionOptions,
    bool planIsWholeValue,
    bool valuesAreColumn)
    : selection_{selection},
      values_{values},
      sectionBuffer_{sectionBuffer},
      sectionOptions_{sectionOptions},
      planIsWholeValue_{planIsWholeValue},
      valuesAreColumn_{valuesAreColumn} {
  // A whole-value section is stored at the physical type's own width, so the
  // section's values are the column's values and need no slicing. Its
  // candidates, their read factors, their compression and their children's
  // candidates are the ones a section would be offered, taken from the policy
  // a section is selected by. A policy that cannot be narrowed to one encoding
  // gets no floor, which `usable` reports.
  sectionPolicy_ = std::unique_ptr<EncodingSelectionPolicy<physicalType>>(
      static_cast<EncodingSelectionPolicy<physicalType>*>(
          selection
              .template createNestedPolicy<physicalType>(
                  selection.encodingType(), NestedEncodingIdentifier{0})
              .release()));
  if (sectionPolicy_ != nullptr &&
      sectionPolicy_->narrowed([](EncodingType candidate) {
        return candidate == EncodingType::FixedBitWidth;
      }) == nullptr) {
    sectionPolicy_.reset();
  }
}

template <typename T>
std::string_view SubIntSplitEncoding<T>::WholeValueFloor::encodeAs(
    EncodingType encodingType) {
  return EncodingFactory::encode<physicalType>(
      sectionPolicy_->narrowed([encodingType](EncodingType candidate) {
        return candidate == encodingType;
      }),
      values_,
      sectionBuffer_,
      sectionOptions_);
}

template <typename T>
const std::optional<typename SubIntSplitEncoding<T>::SampledWholeValue>&
SubIntSplitEncoding<T>::WholeValueFloor::quote() {
  if (!quoted_) {
    quoted_ = true;
    // The candidates sampleWholeValue prices, and why only those, are
    // documented there.
    if (!planIsWholeValue_) {
      quote_ = sampleWholeValue(*sectionPolicy_, values_, sectionOptions_);
      // Delta and Varint are never quoted: a whole column in either replays
      // every earlier row to reach one. `under` offers them only to a plan
      // that already does.
      if (quote_.has_value() &&
          (quote_->encoding == EncodingType::Delta ||
           quote_->encoding == EncodingType::Varint)) {
        quote_.reset();
      }
    }
  }
  return quote_;
}

template <typename T>
std::optional<std::string_view> SubIntSplitEncoding<T>::WholeValueFloor::under(
    uint64_t bytesToBeat,
    std::optional<EncodingType> replayingEncoding) {
  if (!usable()) {
    return std::nullopt;
  }
  std::optional<std::string_view> floor;
  uint64_t budget = bytesToBeat;
  // Encoded only where the quote, scaled down by wholeValueEstimateSlack's
  // margin, still undercuts the plan.
  if (quote().has_value() &&
      quote_->lowerBoundBytes < static_cast<double>(budget)) {
    decidedAbove_ = std::max<uint64_t>(
        decidedAbove_, static_cast<uint64_t>(quote_->lowerBoundBytes) + 1);
    if (!sampledEncoded_.has_value()) {
      sampledEncoded_ = encodeAs(quote_->encoding);
    }
    decidedAbove_ =
        std::max<uint64_t>(decidedAbove_, sampledEncoded_->size() + 1);
    if (sampledEncoded_->size() < budget) {
      floor = sampledEncoded_;
      budget = sampledEncoded_->size();
    }
  }

  if (replayingEncoding.has_value() && !replayingEncoded_.has_value()) {
    auto replayingPolicy = sectionPolicy_->narrowed(
        [encodingType = *replayingEncoding](EncodingType candidate) {
          return candidate == encodingType;
        });
    if (replayingPolicy != nullptr) {
      replayingEncoded_ = EncodingFactory::encode<physicalType>(
          std::move(replayingPolicy), values_, sectionBuffer_, sectionOptions_);
    }
  }
  if (replayingEncoding.has_value() && replayingEncoded_.has_value() &&
      replayingEncoded_->size() < budget) {
    floor = replayingEncoded_;
    budget = replayingEncoded_->size();
  }

  // FixedBitWidth's size is exact and needs only the range, which the column's
  // statistics already hold when these are its values.
  if (!fixedBitWidthEstimate_.has_value()) {
    std::optional<Statistics<physicalType>> residualStatistics;
    if (!valuesAreColumn_) {
      residualStatistics.emplace(Statistics<physicalType>::create(values_));
    }
    const auto& statistics =
        valuesAreColumn_ ? selection_.statistics() : *residualStatistics;
    fixedBitWidthEstimate_ = FixedBitWidthEncoding<physicalType>::estimateSize(
        values_.size(), statistics, sectionOptions_);
  }
  if (*fixedBitWidthEstimate_ < budget) {
    decidedAbove_ =
        std::max<uint64_t>(decidedAbove_, *fixedBitWidthEstimate_ + 1);
    if (!fixedBitWidthEncoded_.has_value()) {
      fixedBitWidthEncoded_ = encodeAs(EncodingType::FixedBitWidth);
    }
    decidedAbove_ =
        std::max<uint64_t>(decidedAbove_, fixedBitWidthEncoded_->size() + 1);
    if (fixedBitWidthEncoded_->size() < budget) {
      floor = fixedBitWidthEncoded_;
    }
  }
  return floor;
}

template <typename T>
std::string SubIntSplitEncoding<T>::debugString(int offset) const {
  std::string indent(offset, ' ');
  std::string result = indent +
      "SubIntSplitEncoding sections=" + std::to_string(sections_.size());
  // Which section the permutation sorts by, and which sections took a
  // transform, since both decide what the plan costs to read.
  if (transformInfo_.keySection != subintsplit::TransformInfo::kNoKeySection) {
    result += " keySection=" + std::to_string(transformInfo_.keySection);
  }
  if (rowFrame_.active()) {
    result += fmt::format(
        " rowFrame=(slope={:#x} base={:#x})", rowFrame_.slope, rowFrame_.base);
  }
  if (deltaEncoded_) {
    result += " delta=yes";
  }
  result += "\n";
  for (size_t s = 0; s < sections_.size(); ++s) {
    const auto& sec = sections_[s];
    result += indent + "  [" + std::to_string(sec.bitStart) + ".." +
        std::to_string(sec.bitEnd) +
        "] storageBytes=" + std::to_string(sec.storageBytes);
    if (s < transformInfo_.transformIds.size() &&
        transformInfo_.transformIds[s] != 0) {
      result += " transform=" +
          subintsplit::toString(
                    static_cast<subintsplit::TransformId>(
                        transformInfo_.transformIds[s]));
    }
    result += "\n";
    result += sec.encoding->debugString(offset + 4);
    result += "\n";
  }
  return result;
}

} // namespace facebook::nimble
