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
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/DecoderUtil.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/subintsplit/DeltaTransform.h"
#include "velox/dwio/nimble/encodings/subintsplit/Format.h"
#include "velox/dwio/nimble/encodings/subintsplit/Sampler.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionEncoder.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionTable.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitBoundaries.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitSelector.h"
#include "velox/dwio/nimble/encodings/subintsplit/TuningConfig.h"

// SubIntSplitEncoding: decomposes each value in a 32- or 64-bit integer stream
// into bit-range sub-streams, selects an optimal encoding for each sub-stream
// via a sample-driven DP algorithm, and stitches the encoded sub-streams back
// together for efficient decoding.
//
// Only supported for 32- and 64-bit types (int32_t, uint32_t, int64_t,
// uint64_t, float, double). The physical type for float is uint32_t and for
// double is uint64_t; bit patterns are preserved across encode/decode.
//
// The pieces live in encodings/subintsplit/: SplitSelector plans the bit
// ranges, SectionEncoder writes one, SectionTable reads them all back, and
// Format.h documents the binary layout.

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
      const Encoding::Options& options = {},
      const subintsplit::TuningConfig& tuning =
          subintsplit::kDefaultTuningConfig);

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  /// Bulk scan for the readWithVisitor fast path. Decodes the contiguous span
  /// covering the selected rows once, then gathers/scatters the requested
  /// positions through the visitor. Invoked by detail::readWithVisitorFast.
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

  /// Encodes one candidate form. `flags` goes into the header byte and records
  /// whether `values` are raw or zigzag deltas.
  static std::string_view encodeImpl(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options,
      uint8_t flags,
      const subintsplit::TuningConfig& tuning =
          subintsplit::kDefaultTuningConfig);

  std::string debugString(int offset) const final;

 private:
  // Values decoded per refill by the readWithVisitor slow path. Small enough
  // that a visitor skipping most rows wastes little, large enough to amortise
  // the per-section virtual dispatch.
  static constexpr uint32_t kSlowPathBlock = 256;

  uint32_t pendingAvailable() const noexcept {
    return pendingCount_ - pendingOffset_;
  }

  // Hands back values the slow path decoded ahead of the cursor. Returns how
  // many were written to `output`.
  uint32_t takeFromPending(uint32_t rowCount, physicalType* output);

  // Decodes the next block of values into pendingBuf_, clamped to what the
  // stream has left.
  void refillPending();

  // Walks the output in chunk-sized steps, combining every section for a chunk
  // before moving on, so the output slice and the section scratch both stay in
  // cache across the whole section loop.
  void decodeChunked(uint32_t rowCount, physicalType* output);

  // Chooses the bit ranges to split into, either from the preserve-mode config
  // or by running the DP planner over a sample.
  static std::vector<subintsplit::SectionPlan> planSections(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      const subintsplit::TuningConfig& tuning);

  // Assembles the prefix, the section headers and the section payloads into
  // `buffer`.
  static std::string_view writeEncoding(
      std::span<const subintsplit::SectionPlan> sections,
      std::span<const std::string_view> payloads,
      uint32_t valueCount,
      uint8_t flags,
      Buffer& buffer,
      bool useVarintRowCount);

  subintsplit::SectionTable<physicalType> sections_;

  // Whether the stored sections hold zigzag deltas rather than raw values.
  bool deltaEncoded_{false};

  // Running prefix-sum accumulator for delta-encoded streams, carried across
  // the chunk loop and across successive materialize() calls.
  physicalType deltaAccumulator_{0};

  // Logical read cursor (rows consumed so far). Maintained across skip(),
  // materialize(), and the readWithVisitor slow path so the fast path can map
  // external row numbers onto the section cursors.
  uint32_t row_{0};

  // Holds the decoded span of physical values in the readWithVisitor fast path
  // before they are gathered/widened into the reader output.
  Vector<physicalType> decodeBuf_;

  // Values the readWithVisitor slow path decoded ahead of the read cursor.
  //
  // That path asks for one value at a time, which otherwise costs a virtual
  // materialize(1, ...) per section per value. Decoding a block at a time
  // amortises the dispatch and lets the section kernels vectorise, at the cost
  // of running the section cursors ahead of row_. The section cursors therefore
  // sit at row_ + pendingAvailable(), and skip() and materialize() both consume
  // this buffer before touching the sections.
  Vector<physicalType> pendingBuf_;
  uint32_t pendingOffset_{0};
  uint32_t pendingCount_{0};
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
      sections_{pool, tuning.decodeChunkSize},
      decodeBuf_{&pool},
      pendingBuf_{&pool} {
  NIMBLE_CHECK_FILE(
      data.size() >= this->dataOffset() + subintsplit::kStreamHeaderSize,
      "SubIntSplit stream header is truncated.");
  const char* pos = data.data() + this->dataOffset();
  const auto header = subintsplit::readStreamHeader(pos);
  NIMBLE_CHECK_FILE(
      header.numSections > 0,
      "SubIntSplit stream must contain at least one section.");
  NIMBLE_CHECK_FILE(
      header.numSections <= sizeof(physicalType) * 8,
      "SubIntSplit stream has too many sections.");
  NIMBLE_CHECK_FILE(
      (header.flags & ~subintsplit::kKnownFlags) == 0,
      "SubIntSplit stream has unsupported flags.");
  deltaEncoded_ = (header.flags & subintsplit::kFlagDelta) != 0;
  sections_.load(
      {pos, data.size() - this->dataOffset() - subintsplit::kStreamHeaderSize},
      header.numSections,
      stringBufferFactory,
      options);
}

template <typename T>
void SubIntSplitEncoding<T>::reset() {
  sections_.reset();
  row_ = 0;
  deltaAccumulator_ = 0;
  pendingOffset_ = 0;
  pendingCount_ = 0;
}

template <typename T>
void SubIntSplitEncoding<T>::skip(uint32_t rowCount) {
  // The sections already sit past anything still buffered, so those rows are
  // skipped by dropping them rather than by moving the section cursors.
  const uint32_t fromPending = std::min(rowCount, pendingAvailable());
  pendingOffset_ += fromPending;
  row_ += fromPending;

  uint32_t remaining = rowCount - fromPending;
  if (remaining == 0) {
    return;
  }

  if (!deltaEncoded_) {
    sections_.skip(remaining);
    row_ += remaining;
    return;
  }

  const uint32_t chunkSize = sections_.decodeChunkSize();
  decodeBuf_.resize(std::min(remaining, chunkSize));
  while (remaining > 0) {
    const uint32_t count = std::min(remaining, chunkSize);
    decodeChunked(count, decodeBuf_.data());
    row_ += count;
    remaining -= count;
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
  const uint32_t block =
      std::min({kSlowPathBlock, sections_.decodeChunkSize(), total - row_});
  decodeChunked(block, pendingBuf_.data());
  pendingOffset_ = 0;
  pendingCount_ = block;
}

template <typename T>
void SubIntSplitEncoding<T>::decodeChunked(
    uint32_t rowCount,
    physicalType* output) {
  const uint32_t chunkSize = sections_.decodeChunkSize();

  // Delta streams store zigzag steps, so the accumulator carries across chunks
  // and across successive materialize() calls -- which is why they can only be
  // read sequentially.
  bool atStreamStart = (row_ == 0);

  for (uint32_t start = 0; start < rowCount; start += chunkSize) {
    const uint32_t count = std::min(chunkSize, rowCount - start);
    sections_.decodeChunk(count, output + start);

    if (deltaEncoded_) {
      subintsplit::decodeDeltas<physicalType>(
          {output + start, count}, deltaAccumulator_, atStreamStart);
      atStreamStart = false;
    }
  }
}

template <typename T>
void SubIntSplitEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  auto* output = static_cast<physicalType*>(buffer);

  const uint32_t fromPending = takeFromPending(rowCount, output);
  rowCount -= fromPending;
  if (rowCount == 0) {
    return;
  }
  output += fromPending;

  // A pass-through stream needs no masking, shifting or OR-ing, so the sole
  // section can write the caller's buffer directly. Delta streams still need
  // the prefix-sum pass, so they never take this route.
  if (sections_.isPassThrough() && !deltaEncoded_) {
    sections_.decodePassThrough(rowCount, output);
  } else {
    decodeChunked(rowCount, output);
  }
  row_ += rowCount;
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

  // Fast path: bulk-decode for integral 4/8-byte physical types extracted into
  // the reader with a compatible (at-least-as-wide integral) output type.
  // Float/double fall through here (kIsFluidCast is false for them) and use the
  // slow path, which applies castFromPhysicalType. The runtime useFastPath
  // check additionally requires a deterministic filter, AVX2, and the bulk path
  // being enabled with null+filter/hook compatibility.
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
        // Decoding a block at a time turns a virtual call per section per value
        // into one per section per block.
        if (pendingAvailable() == 0) {
          refillPending();
        }
        // Keep row_ in sync so a subsequent fast-path chunk maps rows
        // correctly.
        ++row_;
        return pendingBuf_[pendingOffset_++];
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
std::vector<subintsplit::SectionPlan> SubIntSplitEncoding<T>::planSections(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    const subintsplit::TuningConfig& tuning) {
  constexpr int kBits = static_cast<int>(sizeof(physicalType) * 8);

  const auto mode =
      selection.getConfig(std::string(subintsplit::kSplitModeConfigKey));
  if (mode.has_value() && *mode == subintsplit::kSplitModePreserve) {
    const auto boundaries = selection.getConfig(
        std::string(subintsplit::kSplitBoundariesConfigKey));
    NIMBLE_CHECK(
        boundaries.has_value(),
        "SubIntSplit preserve mode requires boundaries config.");
    auto parsed = subintsplit::parseSplitBoundaries(*boundaries, kBits);
    NIMBLE_CHECK(parsed.has_value(), "Invalid SubIntSplit boundaries config.");
    return std::move(parsed.value());
  }

  std::vector<uint64_t> samples;
  subintsplit::sampleIntoU64<physicalType>(values, samples, tuning.sampler);

  return subintsplit::selectSplits(
             samples, kBits, values.size(), tuning.selector)
      .sections;
}

template <typename T>
std::string_view SubIntSplitEncoding<T>::writeEncoding(
    std::span<const subintsplit::SectionPlan> sections,
    std::span<const std::string_view> payloads,
    uint32_t valueCount,
    uint8_t flags,
    Buffer& buffer,
    bool useVarintRowCount) {
  const auto numSections = static_cast<uint8_t>(sections.size());

  uint32_t payloadSize = 0;
  for (const auto& payload : payloads) {
    payloadSize += static_cast<uint32_t>(payload.size());
  }
  const uint32_t encodingSize =
      Encoding::serializePrefixSize(valueCount, useVarintRowCount) +
      subintsplit::specificHeaderSize(numSections) + payloadSize;

  char* const reserved = buffer.reserve(encodingSize);
  char* pos = reserved;

  Encoding::serializePrefix(
      EncodingType::SubIntSplit,
      TypeTraits<T>::dataType,
      valueCount,
      useVarintRowCount,
      pos);

  encoding::write<uint8_t>(numSections, pos);
  encoding::write<uint8_t>(flags, pos);
  for (uint8_t i = 0; i < numSections; ++i) {
    subintsplit::writeSectionHeader(
        sections[i].range(), static_cast<uint32_t>(payloads[i].size()), pos);
  }
  for (const auto& payload : payloads) {
    encoding::writeBytes(payload, pos);
  }

  NIMBLE_DCHECK_EQ(
      static_cast<uint32_t>(pos - reserved),
      encodingSize,
      "SubIntSplitEncoding: encoding size mismatch");

  return {reserved, encodingSize};
}

template <typename T>
std::string_view SubIntSplitEncoding<T>::encodeImpl(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options,
    uint8_t flags,
    const subintsplit::TuningConfig& tuning) {
  if (values.empty()) {
    NIMBLE_INCOMPATIBLE_ENCODING("SubIntSplitEncoding cannot be empty.");
  }

  const auto sections = planSections(selection, values, tuning);
  NIMBLE_CHECK(
      !sections.empty(), "SubIntSplitEncoding: selector returned no sections");

  ScopedEncodingBuffer scopedBuffer{
      &buffer.getMemoryPool(), options.encodingBufferPool};
  Buffer& sectionBuffer = scopedBuffer.get();

  // Pack each section at its exact bit width instead of rounding up to a byte,
  // so e.g. a 12-bit section costs 12 bits/value rather than 16. Sections
  // dominate the encoded size for multi-field values, where byte rounding
  // wasted up to 7 bits/value per section. FixedBitWidth records its own bit
  // width, so the decode path is unaffected.
  Encoding::Options sectionOptions = options;
  sectionOptions.fixedBitWidthUseExactBits = true;

  std::vector<std::string_view> payloads;
  payloads.reserve(sections.size());
  for (size_t i = 0; i < sections.size(); ++i) {
    payloads.push_back(
        subintsplit::encodeSection<physicalType>(
            selection,
            values,
            sections[i].range(),
            static_cast<NestedEncodingIdentifier>(i),
            sectionBuffer,
            sectionOptions));
  }

  return writeEncoding(
      sections,
      payloads,
      static_cast<uint32_t>(values.size()),
      flags,
      buffer,
      options.useVarintRowCount);
}

template <typename T>
std::string_view SubIntSplitEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options,
    const subintsplit::TuningConfig& tuning) {
  if (!options.subIntSplitDeltaPreTransform || values.size() < 2) {
    return encodeImpl(
        selection,
        values,
        buffer,
        options,
        /*flags=*/0,
        tuning);
  }

  // Encode both forms and keep the smaller, so the pre-transform can never
  // regress a stream it does not suit -- InterleavedCounters, for instance,
  // is worse under delta because interleaved shards break monotonicity.
  const std::string_view plain = encodeImpl(
      selection,
      values,
      buffer,
      options,
      /*flags=*/0,
      tuning);

  Vector<physicalType> residuals{&buffer.getMemoryPool(), values.size()};
  subintsplit::encodeDeltas<physicalType>(
      values, {residuals.data(), residuals.size()});

  const std::string_view delta = encodeImpl(
      selection,
      std::span<const physicalType>(residuals.data(), residuals.size()),
      buffer,
      options,
      subintsplit::kFlagDelta,
      tuning);

  return delta.size() < plain.size() ? delta : plain;
}

template <typename T>
std::string SubIntSplitEncoding<T>::debugString(int offset) const {
  const std::string indent(offset, ' ');
  return indent + "SubIntSplitEncoding sections=" +
      std::to_string(sections_.numSections()) +
      (deltaEncoded_ ? " delta=yes" : " delta=no") + "\n" +
      sections_.debugString(offset);
}

} // namespace facebook::nimble
