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
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/DecoderUtil.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#include "velox/dwio/nimble/encodings/subintsplit/DeltaTransform.h"
#include "velox/dwio/nimble/encodings/subintsplit/Format.h"
#include "velox/dwio/nimble/encodings/subintsplit/RowFrame.h"
#include "velox/dwio/nimble/encodings/subintsplit/Sampler.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionAccumulator.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitBoundaries.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitSelector.h"
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
// ranges over a cost grid priced by CostModel, RowFrame subtracts a
// predictor from the values, and Format.h documents the binary layout.

namespace facebook::nimble::subintsplit {

/// Options one SubIntSplit section is encoded under, derived from the
/// enclosing column's options.
///
/// A section overrides two of the column's options, changing encoded size,
/// so anything pricing a section's cost must derive options from here
/// rather than restate them.
inline Encoding::Options sectionEncodingOptions(
    const Encoding::Options& options) {
  Encoding::Options sectionOptions = options;
  // Pack each section at its exact bit width instead of rounding up to a
  // byte; sections dominate encoded size for multi-field values, so byte
  // rounding there is costly. FixedBitWidth records its own bit width, so
  // decode is unaffected.
  sectionOptions.fixedBitWidthUseExactBits = true;
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
  return sectionOptions;
}

} // namespace facebook::nimble::subintsplit

namespace facebook::nimble {

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
      const Encoding::Options& options = {});

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
      const Encoding::Options& options = {});

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

  // The configuration the split planner runs under for `options`.
  static subintsplit::SelectorConfig plannerSelectorConfig(
      const Encoding::Options& options,
      size_t rowCount);

  // The sample the planner draws, honouring
  // Options::subIntSplitPlannerMaxSamples.
  static subintsplit::SamplerConfig plannerSamplerConfig(
      const Encoding::Options& options);

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
      const Encoding::Options& options);

  // Encodes `values` as they are, fitting a row frame first when the options
  // ask for one. encode() calls this once, or twice when it also tries the
  // delta pre-transform.
  static std::string_view encodeValues(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options);

  // Plans `values` into sections and encodes each. `values` have `rowFrame`
  // already subtracted from them, and the header records the frame so reads
  // add it back. `planning`, when not null, is the sample and grid already
  // costed for exactly these values. `extraFlags` goes into the header's flag
  // byte, for instance to record that `values` are zigzag deltas.
  static std::string_view encodeResiduals(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options,
      const subintsplit::RowFrame& rowFrame = {},
      PlanningSample* planning = nullptr,
      uint8_t extraFlags = 0);

  // Predictor the encoder subtracted before planning, inactive when it did
  // not. Every value leaving this class has it added back.
  subintsplit::RowFrame rowFrame_;

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
  // subintsplit::Options::foldConstantSections folded the Constant ones into
  // constantOr_ at construction.
  std::vector<uint32_t> dynamicSections_;
  // Combined, pre-shifted bits of every folded Constant section, OR-ed into
  // each value by the first dynamic section.
  physicalType constantOr_{0};
  // One dynamic section reproduces each value verbatim, so it decodes straight
  // into the caller's buffer. Only with subintsplit::Options::passThrough.
  bool passThrough_{false};
  // Output elements combined per chunk on the untransformed path.
  uint32_t decodeChunkSize_{subintsplit::kDecodeChunkSize};

  // Values the readWithVisitor slow path decoded ahead of the read cursor, when
  // subintsplit::Options::visitorBlockBuffer is set. The sections then stand at
  // row_ + pendingAvailable(), and skip() and materialize() consume this buffer
  // before touching them.
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
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>{pool, data, options},
      sections_{},
      scratchBuf_{&pool},
      pendingBuf_{&pool},
      decodeBuf_{&pool} {
  uint8_t flags{0};
  const auto parsed = subintsplit::parseSections(
      data, this->dataOffset(), &flags, &rowFrame_);
  deltaEncoded_ = (flags & subintsplit::kFlagDelta) != 0;
  if (options.subIntSplitDecodeChunkSize > 0) {
    decodeChunkSize_ = options.subIntSplitDecodeChunkSize;
  }
  visitorBlockBuffer_ = options.subIntSplit.visitorBlockBuffer;
  NIMBLE_CHECK(!parsed.empty(), "SubIntSplit stream has no sections.");

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
        sec.storageBytes,
        []<typename StorageType>() { return TypeTraits<StorageType>::dataType; });
    NIMBLE_CHECK_FILE(
        EncodingPrefix::dataType(parsed[s].stream) == expectedDataType,
        "SubIntSplit section data type does not match its bit width.");
    sec.encoding = EncodingFactory().create(
        *this->pool_, parsed[s].stream, stringBufferFactory, options);
  }

  // A Constant section contributes the same bits to every row, so decoding it
  // materialises copies of one number to OR them in a row at a time. With the
  // switch on it is read once here and left out of the chunk loop.
  dynamicSections_.reserve(sections_.size());
  for (uint32_t s = 0; s < sections_.size(); ++s) {
    const auto& sec = sections_[s];
    if (options.subIntSplit.foldConstantSections &&
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
  if (options.subIntSplit.passThrough && dynamicSections_.size() == 1 &&
      constantOr_ == 0) {
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
  deltaAccumulator_ = 0;
  pendingOffset_ = 0;
  pendingCount_ = 0;
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
  decodeUntransformed(rowCount, output);
  row_ += rowCount;
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
        // A delta stream rebuilds each value from every earlier row, and a row
        // frame is added back by the row a value sits at, both of which only
        // materialize() tracks, so this path defers to it.
        if (deltaEncoded_ || rowFrame_.active()) {
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

  // processFixedWidthRun handles scatter (null gaps), filter evaluation, and
  // hook forwarding. For non-hook paths it operates in place on the reader's
  // rawValues; for hooks, values stays as the staged buffer.
  if constexpr (!V::kHasHook) {
    values = reinterpret_cast<OutputType*>(visitor.reader().rawValues());
  }

  auto numValues = visitor.reader().numValues();
  int32_t* filterHits = nullptr;
  if constexpr (V::kHasFilter) {
    filterHits = visitor.outputRows(numSelected) - numValues;
  }

  velox::dwio::common::
      processFixedWidthRun<OutputType, V::kFilterOnly, kScatter, V::dense>(
          velox::RowSet(selectedRows, numSelected),
          0,
          numSelected,
          scatterRows,
          values,
          filterHits,
          numValues,
          visitor.filter(),
          visitor.hook());

  if constexpr (!V::kHasHook) {
    // Filter: count passing rows; no filter: all rows produce values.
    visitor.addNumValues(
        V::kHasFilter ? numValues - visitor.reader().numValues() : numRows);
  }
  visitor.setRowIndex(visitor.numRows());
}

template <typename T>
std::string_view SubIntSplitEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  if (!options.subIntSplitDeltaPreTransform || values.size() < 2) {
    return encodeValues(selection, values, buffer, options);
  }
  // Encode both forms and keep the smaller, so the pre-transform can never
  // regress a stream it does not suit -- interleaved counters, for instance,
  // are worse under delta because interleaved shards break monotonicity.
  const std::string_view plain =
      encodeValues(selection, values, buffer, options);
  Vector<physicalType> residuals{&buffer.getMemoryPool(), values.size()};
  subintsplit::encodeDeltas<physicalType>(
      values, {residuals.data(), residuals.size()});
  // Deltas are undone by a running sum over every earlier row, which a frame
  // cannot sit beneath, so the delta form plans its sections over the
  // residuals alone.
  const std::string_view delta = encodeResiduals(
      selection,
      std::span<const physicalType>(residuals.data(), residuals.size()),
      buffer,
      options,
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
    const Encoding::Options& options) {
  if (!options.subIntSplit.rowFrame) {
    return encodeResiduals(selection, values, buffer, options);
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
    return encodeResiduals(selection, values, buffer, options);
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
        ? encodeResiduals(selection, residuals, buffer, options, rowFrame)
        : encodeResiduals(selection, values, buffer, options);
  }

  if (options.subIntSplit.rowFrameForceApply) {
    return encodeResiduals(selection, residuals, buffer, options, rowFrame);
  }

  // A step frame produces runs of whole values, which the planner's run
  // models can misprice, so both streams are encoded, into a scratch buffer
  // so the loser takes no space in the stream, and the smaller is kept.
  if (stepFrame) {
    Buffer scratch{buffer.getMemoryPool()};
    const auto framed =
        encodeResiduals(selection, residuals, scratch, options, rowFrame);
    const auto unframed = encodeResiduals(selection, values, scratch, options);
    const std::string_view smaller =
        framed.size() < unframed.size() ? framed : unframed;
    char* reserved = buffer.reserve(smaller.size());
    std::memcpy(reserved, smaller.data(), smaller.size());
    return {reserved, smaller.size()};
  }

  // Fitting only says the column follows a line, not that sections encode
  // the distance from it more cheaply than the values themselves. So both
  // are priced with the planner's own DP over the same inventory and
  // configuration, and the frame is kept only when its estimate, header
  // included, is smaller; the winner's grid is the one the encode plans on.
  const auto selectorConfig = plannerSelectorConfig(options, values.size());
  auto residualPlanning = costPlanningSample(residuals, options);
  auto valuePlanning = costPlanningSample(values, options);
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
        subintsplit::RowFrame{},
        &valuePlanning);
  }
  return encodeResiduals(
      selection, residuals, buffer, options, rowFrame, &residualPlanning);
}

template <typename T>
typename SubIntSplitEncoding<T>::PlanningSample
SubIntSplitEncoding<T>::costPlanningSample(
    std::span<const physicalType> values,
    const Encoding::Options& options) {
  constexpr int kBits = static_cast<int>(sizeof(physicalType) * 8);
  PlanningSample planning;
  subintsplit::sampleIntoU64<physicalType>(
      values, planning.sample, plannerSamplerConfig(options));
  NIMBLE_CHECK(
      !planning.sample.empty(), "SubIntSplit planner sample is empty.");
  planning.grid = subintsplit::buildRestrictedCostGrid(
      planning.sample,
      kBits,
      values.size(),
      options.subIntSplit.allowedEncodings,
      plannerSelectorConfig(options, values.size()));
  return planning;
}

template <typename T>
subintsplit::SelectorConfig SubIntSplitEncoding<T>::plannerSelectorConfig(
    const Encoding::Options& options,
    size_t rowCount) {
  // Huffman and DeltaBlock are withdrawn by default: each was priced into
  // where boundaries fall while being unselectable for the sections they
  // produce, steering the planner toward splits nothing would read well.
  // Both also cost a pass over the sample per grid cell, so withdrawing
  // either buys encode time as well as better plans. See
  // subintsplit::Options::allowHuffman and allowDeltaBlock.
  //
  // Each of these must stay in step with nestedEncodingReadFactors, which
  // decides what a section may actually be encoded as: a mismatch either
  // way gives the planner an encoding selection will not use, or leaves it
  // carving boundaries around one that is unavailable.
  auto selectorConfig = subintsplit::defaultSelectorConfig();
  selectorConfig.allowHuffman = options.subIntSplit.allowHuffman;
  selectorConfig.allowDeltaBlock = options.subIntSplit.allowDeltaBlock;
  // The planner restrictions, every one off by default: each narrows the
  // grid the DP searches, and so can move the plan.
  selectorConfig.trimConstantPlanes = options.subIntSplit.trimConstantPlanes;
  if (options.subIntSplitBoundaryPruneThreshold >= 0.0) {
    selectorConfig.boundaryPruneThreshold =
        options.subIntSplitBoundaryPruneThreshold;
  }
  selectorConfig.maxCandidateBoundaries =
      options.subIntSplitMaxCandidateBoundaries;
  selectorConfig.maxSectionWidth =
      static_cast<int>(options.subIntSplitMaxSectionWidth);
  selectorConfig.frequencyMetricsMaxWidth =
      static_cast<int>(options.subIntSplitFrequencyMetricsMaxWidth);
  selectorConfig.decodeCostBitsPerValue =
      options.subIntSplitDecodeCostBitsPerValue;
  selectorConfig.streamRowCount = rowCount;
  return selectorConfig;
}

template <typename T>
subintsplit::SamplerConfig SubIntSplitEncoding<T>::plannerSamplerConfig(
    const Encoding::Options& options) {
  auto samplerConfig = subintsplit::defaultSamplerConfig();
  if (options.subIntSplitPlannerMaxSamples > 0) {
    samplerConfig.maxSamples = options.subIntSplitPlannerMaxSamples;
  }
  return samplerConfig;
}

template <typename T>
std::string_view SubIntSplitEncoding<T>::encodeResiduals(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options,
    const subintsplit::RowFrame& rowFrame,
    PlanningSample* planning,
    uint8_t extraFlags) {
  const bool useVarint = options.useVarintRowCount;
  const uint32_t valueCount = static_cast<uint32_t>(values.size());

  if (values.empty()) {
    NIMBLE_INCOMPATIBLE_ENCODING("SubIntSplitEncoding cannot be empty.");
  }

  constexpr int kBits = static_cast<int>(sizeof(physicalType) * 8);

  std::vector<subintsplit::SectionPlan> segments;
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
    const auto selectorConfig = plannerSelectorConfig(options, valueCount);
    subintsplit::SelectorResult selectorResult;
    if (planning != nullptr) {
      selectorResult = subintsplit::selectSplitsOverGrid(
          planning->grid, kBits, selectorConfig);
    } else {
      std::vector<uint64_t> sampleBuf;
      subintsplit::sampleIntoU64<physicalType>(
          values, sampleBuf, plannerSamplerConfig(options));

      // An empty allowed set costs every encoding, so this is the production
      // path unless a caller has deliberately narrowed the inventory.
      selectorResult = subintsplit::selectSplitsRestricted(
          sampleBuf,
          kBits,
          valueCount,
          options.subIntSplit.allowedEncodings,
          selectorConfig);
    }
    segments = std::move(selectorResult.sections);
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
      subintsplit::sectionEncodingOptions(options);

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
  std::vector<uint8_t> sectionStorage(splitCount);
  std::vector<std::string_view> plainEncoded(splitCount);
  for (uint8_t s = 0; s < splitCount; ++s) {
    const auto& seg = segments[s];
    const int width = seg.bitEnd - seg.bitStart + 1;
    sectionStorage[s] = sectionStorageBytes(width);
    const uint64_t mask = subintsplit::widthMask(width);
    plainEncoded[s] = encodeSectionFrom(
        s, sectionStorage[s], [&values, &seg, mask](uint32_t i) {
          uint64_t value = 0;
          __builtin_memcpy(&value, &values[i], sizeof(physicalType));
          return (value >> seg.bitStart) & mask;
        });
  }

  sectionData = std::move(plainEncoded);

  // Write final encoding to main buffer.
  const uint32_t prefixSize =
      Encoding::serializePrefixSize(valueCount, useVarint);
  const uint32_t specificHeader = subintsplit::specificHeaderSize(splitCount) +
      subintsplit::rowFrameHeaderSize(rowFrame);
  uint32_t sectionsSize = 0;
  for (const auto& sv : sectionData) {
    sectionsSize += static_cast<uint32_t>(sv.size());
  }
  const uint32_t encodingSize = prefixSize + specificHeader + sectionsSize;

  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;

  Encoding::serializePrefix(
      EncodingType::SubIntSplit,
      TypeTraits<T>::dataType,
      valueCount,
      useVarint,
      pos);

  encoding::write<uint8_t>(splitCount, pos);
  const uint8_t flags = extraFlags |
      (rowFrame.active() ? subintsplit::kFlagRowFrame : uint8_t{0});
  encoding::write<uint8_t>(flags, pos);

  if (rowFrame.active()) {
    encoding::write<uint8_t>(subintsplit::kRowFrameGuard, pos);
    encoding::write<uint64_t>(rowFrame.slope, pos);
    encoding::write<uint64_t>(rowFrame.base, pos);
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
std::string SubIntSplitEncoding<T>::debugString(int offset) const {
  std::string indent(offset, ' ');
  std::string result = indent +
      "SubIntSplitEncoding sections=" + std::to_string(sections_.size());
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
    result += "\n";
    result += sec.encoding->debugString(offset + 4);
    result += "\n";
  }
  return result;
}

} // namespace facebook::nimble
