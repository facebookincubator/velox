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
#include <atomic>
#include <memory>
#include <utility>
#include <vector>

#include <folly/CPortability.h>

#include "velox/common/memory/RawVector.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/subintsplit/Format.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionAccumulator.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionTransform.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble {

namespace detail {

/// Serves indexed reads over a stream that has no EncodingView of its own by
/// decoding it once into an owned array and serving each indexed read from
/// that. Construction cost and memory are O(rowCount), so this is a fallback
/// path, not the common one.
///
/// Not SubIntSplit-specific; SharedDictionaryAlphabet hand-rolls the same
/// fallback and could reuse this if moved to views/.
template <typename T>
class MaterializedEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  MaterializedEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        values_{this->template getVectorBuffer<physicalType>()} {
    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    auto encoding = EncodingFactory{options}.create(
        *this->pool_, data, noStringBufferFactory);
    NIMBLE_CHECK_NOT_NULL(encoding);
    NIMBLE_CHECK_EQ(encoding->rowCount(), this->rowCount_);
    values_.resize(this->rowCount_);
    if (this->rowCount_ > 0) {
      encoding->materialize(this->rowCount_, values_.data());
    }
  }

  ~MaterializedEncodingView() override {
    this->releaseVectorBuffer(values_);
  }

 private:
  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    return detail::castFromPhysicalType<T>(values_[index]);
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    std::copy_n(values_.data() + offset, length, output);
  }

  Vector<physicalType> values_;
};

// Prefers a view over a stream, decoding once when it cannot have one.
// Attempting construction is the only available test: compression nests, so
// a stream can report viewable while a nested stream is not, and views
// signal both that and an incompatible type by throwing.
template <typename SectionT>
std::unique_ptr<EncodingView> makeSectionView(
    std::string_view stream,
    velox::memory::MemoryPool* pool,
    const Encoding::Options& options) {
  if (supportsEncodingView(EncodingPrefix::encodingType(stream))) {
    try {
      return createTypedEncodingView<SectionT>(stream, pool, options);
    } catch (const NimbleException&) {
      // Fall through to the materialized fallback below.
    }
  }
  return std::make_unique<MaterializedEncodingView<SectionT>>(
      stream, pool, options);
}

} // namespace detail

/// Random-access view over a SubIntSplit stream.
///
/// Holds one indexed accessor per bit-range section and reassembles the word
/// from them, where SubIntSplitEncoding holds an Encoding per section and can
/// only reach row i by traversing from row zero.
///
/// Sections that cannot be viewed fall back to MaterializedEncodingView, so
/// indexed access survives whatever the selection picked. Of the eight
/// encodings in the default nested inventory only Varint has no view.
template <typename T>
class SubIntSplitEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  static_assert(
      sizeof(physicalType) == 4 || sizeof(physicalType) == 8,
      "SubIntSplitEncodingView only supports 32- and 64-bit types");
  static_assert(
      isNumericType<physicalType>(),
      "SubIntSplitEncodingView only supports numeric types");

  SubIntSplitEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        // Identifies this instance for PositionCache below, for the lifetime
        // of the process: an address can be reused the moment a view is
        // destroyed, which is not true of a counter that only ever
        // increases.
        viewId_{nextViewId_.fetch_add(1, std::memory_order_relaxed)} {
    NIMBLE_CHECK(
        this->encodingType_ == EncodingType::SubIntSplit ||
            this->encodingType_ == EncodingType::SubIntSplitReordered,
        "SubIntSplitEncodingView built over a stream that is not SubIntSplit.");

    uint8_t flags{0};
    const auto parsed = subintsplit::parseSections(
        data, this->dataOffset_, &transformInfo_, &rowFrame_, &flags);
    NIMBLE_CHECK(!parsed.empty(), "SubIntSplit stream has no sections.");
    // A delta stream's sections hold steps rather than values, so no section
    // can be read at an index; the factory routes such streams to a full
    // decode instead.
    NIMBLE_CHECK(
        (flags & subintsplit::kFlagDelta) == 0,
        "SubIntSplitEncodingView cannot index a delta stream.");
    // Validated before any section is built: a transform this reader does not
    // know would otherwise be skipped, and the values it returned would look
    // like ordinary ones.
    for (uint8_t id : transformInfo_.transformIds) {
      subintsplit::transformForRaw(id);
    }

    for (size_t wireIndex = 0; wireIndex < parsed.size(); ++wireIndex) {
      const auto& meta = parsed[wireIndex];
      Section section;
      switch (meta.storageBytes) {
        case 1:
          section = makeSection<uint8_t>(meta, pool, options);
          break;
        case 2:
          section = makeSection<uint16_t>(meta, pool, options);
          break;
        case 4:
          section = makeSection<uint32_t>(meta, pool, options);
          break;
        case 8:
          section = makeSection<uint64_t>(meta, pool, options);
          break;
        default:
          NIMBLE_UNREACHABLE("Invalid SubIntSplit section storage width.");
      }
      NIMBLE_CHECK_EQ(section.view->rowCount(), this->rowCount_);

      // Recorded on every section, since it is what addresses per-section
      // transform state, independent of position in sections_.
      section.wireIndex = wireIndex;
      if (!transformInfo_.transformIds.empty() &&
          transformInfo_.transformIds[wireIndex] != 0) {
        section.transform = subintsplit::transformForRaw(
            transformInfo_.transformIds[wireIndex]);
        const auto mapping = section.transform->positionMapping();
        permutedSection_ = permutedSection_ ||
            mapping == subintsplit::PositionMapping::Permuted;
      }

      // A transformed section is not constant in the values it yields, so
      // folding it away would drop the inverse along with it. Nor may the key
      // section be folded, even though it is untransformed and may well be
      // constant: it is what orders the sections that were permuted by it, and
      // they need it row by row.
      const bool isKeySection = wireIndex == transformInfo_.keySection;
      if (this->rowCount_ > 0 && section.transform == nullptr &&
          !isKeySection &&
          section.view->encodingType() == EncodingType::Constant) {
        constantBits_ |= static_cast<physicalType>(
                             section.valueAt(*section.view, 0) & section.mask)
            << section.bitStart;
        continue;
      }
      sections_.push_back(std::move(section));
    }
  }

 private:
  struct Section {
    int bitStart{0};
    uint64_t mask{0};
    uint8_t storageBytes{8};
    // Bit width of the section, which a transform needs to know how wide a
    // value it is working with.
    int width{0};
    std::unique_ptr<EncodingView> view;
    // Position on the wire, which is how the header's per-section transform
    // state is addressed. Not the position in sections_, since folded
    // constants are dropped from that.
    size_t wireIndex{0};
    // Null where the section carries no transform.
    const subintsplit::SectionTransform* transform{nullptr};
    // What the transform's inverse needs back. Section-wide; anything that
    // varies by block lives in transformInfo_.
    subintsplit::TransformState transformState;
    // Resolved from the storage width at construction. The chunked path
    // switches instead, so that the accumulate kernel stays inlinable.
    uint64_t (*valueAt)(const EncodingView&, uint32_t){nullptr};
  };

  template <typename SectionT>
  static Section makeSection(
      const subintsplit::StoredSection& meta,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options) {
    return Section{
        .bitStart = meta.bitStart,
        .mask = meta.mask,
        .storageBytes = meta.storageBytes,
        .width = meta.bitEnd - meta.bitStart + 1,
        .view = detail::makeSectionView<SectionT>(meta.stream, pool, options),
        .valueAt = &readValueAt<SectionT>,
    };
  }

  // readAt() writes exactly the section's storage width, so the value is read
  // into that width rather than through a wider one, which would depend on byte
  // order.
  template <typename SectionT>
  static uint64_t readValueAt(const EncodingView& view, uint32_t index) {
    SectionT value;
    view.readAt(index, &value);
    return static_cast<uint64_t>(value);
  }

  template <typename SectionT>
  static void readSectionChunk(
      const Section& section,
      uint32_t offset,
      uint32_t count,
      physicalType* output,
      bool isFirst,
      uint8_t* scratch) {
    auto* values = reinterpret_cast<SectionT*>(scratch);
    section.view->read(offset, count, values);
    if (isFirst) {
      subintsplit::accumulateSection<true>(
          values,
          output,
          count,
          section.mask,
          section.bitStart,
          physicalType{0});
    } else {
      subintsplit::accumulateSection<false>(
          values,
          output,
          count,
          section.mask,
          section.bitStart,
          physicalType{0});
    }
  }

  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    physicalType value = readResidualAt(index);
    if (rowFrame_.active()) {
      value = static_cast<physicalType>(
          value +
          subintsplit::rowFramePrediction<physicalType>(rowFrame_, index));
    }
    return detail::castFromPhysicalType<T>(value);
  }

  // Everything below the three read overrides works on residuals, the values
  // as the sections store them, and the overrides add the row frame back once
  // on the way out. Internal reads therefore call the residual forms rather
  // than the overrides, which would add the frame a second time.
  physicalType readResidualAt(uint32_t index) const {
    return readOneRow(index);
  }

  // Reads each range on its own, the way TypedEncodingView's default range
  // list read does, but in residuals.
  void readResidualRangesSeparately(
      std::span<const RowRange> ranges,
      physicalType* output) const {
    for (const auto& range : ranges) {
      const uint32_t offset = range.startRow;
      const uint32_t length = range.numRows();
      if (length == 1) {
        *output = readResidualAt(offset);
      } else {
        readResidualRange(offset, length, output);
      }
      output += length;
    }
  }

  // Reads one row. A key-derived section is followed through the position
  // map, which is O(1) once the map exists.
  physicalType readOneRow(uint32_t index) const {
    const velox::raw_vector<uint32_t>* positions =
        permutedSection_ ? &positionMap() : nullptr;
    physicalType value = constantBits_;
    for (const auto& section : sections_) {
      const uint64_t sectionValue = section.transform == nullptr
          ? section.valueAt(*section.view, index)
          : section.valueAt(*section.view, (*positions)[index]);
      value |= static_cast<physicalType>(sectionValue & section.mask)
          << section.bitStart;
    }
    return value;
  }

  // Maps each original row to the source position it was permuted from, built
  // once per view and reused. Held per thread, since a view is read
  // concurrently and keeps no mutable state of its own; the cache is keyed on
  // viewId_ rather than `this`, since a destroyed view's address can be
  // reused by an unrelated one and a stale hit would index this array with a
  // mismatched row count.
  const velox::raw_vector<uint32_t>& positionMap() const {
    thread_local PositionCache cache;
    if (cache.owner == viewId_) {
      // A hit relies entirely on viewId_ being unique; this checks that it
      // actually was, before the caller indexes this array by row up to
      // rowCount_.
      NIMBLE_DCHECK_EQ(
          cache.positions.size(),
          this->rowCount_,
          "Stale position cache does not match this view's row count.");
      return cache.positions;
    }

    NIMBLE_CHECK(
        transformInfo_.keySection != subintsplit::TransformInfo::kNoKeySection,
        "A computable transform needs the key section it was ordered by.");
    // Prefers the key section's own dense run ids when it has them, avoiding
    // a value-at-a-time read of the whole key section.
    std::vector<uint32_t> runIds;
    std::vector<uint64_t> runValues;
    std::vector<uint64_t> keyValues;
    for (const auto& section : sections_) {
      if (section.wireIndex != transformInfo_.keySection) {
        continue;
      }
      if (section.view->denseRunIds(0, this->rowCount_, runIds, runValues)) {
        break;
      }
      keyValues.resize(this->rowCount_);
      for (uint32_t row = 0; row < this->rowCount_; ++row) {
        keyValues[row] = section.valueAt(*section.view, row);
      }
      break;
    }

    cache.positions.resize(this->rowCount_);
    const subintsplit::TransformContext context{
        .keySection = keyValues,
        .width = 0,
        .keyRunIds = runIds,
        .keyRunValues = runValues};
    for (const auto& section : sections_) {
      if (section.transform == nullptr ||
          section.transform->positionMapping() !=
              subintsplit::PositionMapping::Permuted) {
        continue;
      }
      section.transform->positionMap(
          context,
          section.transformState,
          std::span<uint32_t>(cache.positions.data(), cache.positions.size()));
      break;
    }
    cache.owner = viewId_;
    return cache.positions;
  }

  // Chunked the same way as SubIntSplitEncoding::materialize: with the chunk on
  // the outside and the sections on the inside, the output slice and the
  // scratch both stay resident across the section loop.
  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    readResidualRange(offset, length, output);
    if (rowFrame_.active()) {
      subintsplit::addRowFrame(rowFrame_, offset, output, length);
    }
  }

  void readResidualRange(uint32_t offset, uint32_t length, physicalType* output)
      const {
    if (length == 0) {
      return;
    }

    // A permuted section is read one of three ways depending on range length.
    // A key-derived permutation is a stable sort by key, so a run's scattered
    // occurrences map to a contiguous block of source indices; readPermutedSpan
    // sorts the requested rows' source indices to bring each run's occurrences
    // together, then reads each block in one bulk call. That sort has a fixed
    // cost that only pays off once the range is long enough (kMinSpanLength);
    // shorter ranges probe row by row instead. Above kSpanAdvantageNumerator/
    // kSpanAdvantageDenominator of the column, decoding it whole and keeping
    // the slice wins outright.
    if (transformInfo_.anyTransform()) {
      if (length < kMinSpanLength) {
        readPermutedProbes(offset, length, output);
        return;
      }
      if (length * kSpanAdvantageNumerator <
          this->rowCount_ * kSpanAdvantageDenominator) {
        readPermutedSpan(offset, length, output);
        return;
      }
      readWholeSpan(offset, length, output);
      return;
    }

    // On the stack because a view is read concurrently and so cannot hold
    // scratch of its own. At 1024 rows this and the output slice together are
    // 16 KB and stay in L1 across the section loop.
    alignas(64) uint8_t scratch[kViewChunkSize * sizeof(physicalType)];

    // Section 0 initialises each output element and the rest OR into it, which
    // avoids a separate fill pass. That only works when there is nothing to
    // seed with, so a non-zero constant contribution is filled first instead.
    const bool seedWithConstant = constantBits_ != 0 || sections_.empty();

    for (uint32_t chunkStart = 0; chunkStart < length;
         chunkStart += kViewChunkSize) {
      const uint32_t chunkCount = std::min(kViewChunkSize, length - chunkStart);
      physicalType* chunkOutput = output + chunkStart;
      const uint32_t sourceOffset = offset + chunkStart;

      if (seedWithConstant) {
        std::fill(chunkOutput, chunkOutput + chunkCount, constantBits_);
      }
      for (size_t s = 0; s < sections_.size(); ++s) {
        const auto& section = sections_[s];
        const bool isFirst = !seedWithConstant && s == 0;
        // Switched rather than called through a function pointer, so the
        // compiler can inline the accumulate kernel into the loop.
        switch (section.storageBytes) {
          case 1:
            readSectionChunk<uint8_t>(
                section,
                sourceOffset,
                chunkCount,
                chunkOutput,
                isFirst,
                scratch);
            break;
          case 2:
            readSectionChunk<uint16_t>(
                section,
                sourceOffset,
                chunkCount,
                chunkOutput,
                isFirst,
                scratch);
            break;
          case 4:
            readSectionChunk<uint32_t>(
                section,
                sourceOffset,
                chunkCount,
                chunkOutput,
                isFirst,
                scratch);
            break;
          case 8:
            readSectionChunk<uint64_t>(
                section,
                sourceOffset,
                chunkCount,
                chunkOutput,
                isFirst,
                scratch);
            break;
          default:
            NIMBLE_UNREACHABLE("Invalid SubIntSplit section storage width.");
        }
      }
    }
  }

  // Plans a scattered read across the whole range list rather than probing
  // each range independently, since reading a dense stretch and discarding
  // unwanted rows is far cheaper than a per-row virtual probe per section.
  void readPhysicalRanges(
      std::span<const RowRange> ranges,
      physicalType* output) const final {
    for (const auto& range : ranges) {
      this->checkReadRange(range.startRow, range.numRows());
    }
    readResidualRanges(ranges, output);
    if (rowFrame_.active()) {
      for (const auto& range : ranges) {
        const uint32_t offset = range.startRow;
        const uint32_t length = range.numRows();
        subintsplit::addRowFrame(rowFrame_, offset, output, length);
        output += length;
      }
    }
  }

  void readResidualRanges(
      std::span<const RowRange> ranges,
      physicalType* output) const {
    if (transformInfo_.anyTransform()) {
      readTransformedRanges(ranges, output);
      return;
    }
    readUntransformedRanges(ranges, output);
  }

  // Groups nearby ranges and decodes each group's covering span once. A gap
  // between ranges is worth bridging exactly when it costs at most
  // kProbeCostInDecodedRows, since that is the cost of a separate read call
  // instead. Groups are capped at kViewChunkSize rows so the staging buffer
  // stays in L1.
  void readUntransformedRanges(
      std::span<const RowRange> ranges,
      physicalType* output) const {
    alignas(64) physicalType staged[kViewChunkSize];
    size_t rangeIndex = 0;
    while (rangeIndex < ranges.size()) {
      const uint32_t groupOffset = ranges[rangeIndex].startRow;
      const uint32_t groupFirstLength = ranges[rangeIndex].numRows();
      if (groupFirstLength == 0) {
        ++rangeIndex;
        continue;
      }
      // Anything this long gains nothing from staging: readPhysical() already
      // decodes it chunk by chunk straight into the output.
      if (groupFirstLength >= kViewChunkSize) {
        readResidualRange(groupOffset, groupFirstLength, output);
        output += groupFirstLength;
        ++rangeIndex;
        continue;
      }
      uint32_t groupEnd = groupOffset + groupFirstLength;
      size_t groupLast = rangeIndex;
      for (size_t next = rangeIndex + 1; next < ranges.size(); ++next) {
        const uint32_t nextOffset = ranges[next].startRow;
        const uint32_t nextLength = ranges[next].numRows();
        if (nextLength == 0) {
          // Taken into the group so the copy loop below skips it in order.
          groupLast = next;
          continue;
        }
        // A range that starts before the group's end is out of order or
        // overlapping, and starts a group of its own rather than being
        // assumed to lie inside this one's span.
        if (nextOffset < groupEnd ||
            nextOffset - groupEnd > kProbeCostInDecodedRows ||
            static_cast<uint64_t>(nextOffset) + nextLength - groupOffset >
                kViewChunkSize) {
          break;
        }
        groupEnd = nextOffset + nextLength;
        groupLast = next;
      }

      if (groupLast == rangeIndex) {
        if (groupFirstLength == 1) {
          *output = readOneRow(groupOffset);
        } else {
          readResidualRange(groupOffset, groupFirstLength, output);
        }
        output += groupFirstLength;
        ++rangeIndex;
        continue;
      }

      readResidualRange(groupOffset, groupEnd - groupOffset, staged);
      for (; rangeIndex <= groupLast; ++rangeIndex) {
        const uint32_t offset = ranges[rangeIndex].startRow;
        const uint32_t length = ranges[rangeIndex].numRows();
        std::copy_n(staged + (offset - groupOffset), length, output);
        output += length;
      }
    }
  }

  // Chooses between one whole-column decode and reading each range the way
  // readPhysical() would read it alone. A transformed stream's bulk path is
  // always a whole-column decode, since a permuted section's rows are
  // scattered across it, so the choice is made once for the list by pricing
  // each range at what it would cost read alone, in units of one row of
  // whole-column decode, and decoding the column once that total reaches its
  // row count.
  void readTransformedRanges(
      std::span<const RowRange> ranges,
      physicalType* output) const {
    // One range is exactly what readPhysical() plans for, and it can decode
    // a whole-column request straight into the output.
    if (ranges.size() == 1) {
      readResidualRange(ranges[0].startRow, ranges[0].numRows(), output);
      return;
    }
    const uint64_t rowCount = this->rowCount_;
    // Prices range by range rather than pricing the whole list at a uniform
    // per-row rate: a row from a long range shares its decoded block with its
    // neighbours while a row from a short, scattered range does not, so the
    // same total row count can cost very differently depending on how it is
    // split into ranges.
    uint64_t totalRows = 0;
    for (const auto& range : ranges) {
      const uint32_t rangeLength = range.numRows();
      totalRows += rangeLength;
    }
    uint64_t piecewiseCost = 0;
    for (const auto& range : ranges) {
      const uint32_t rangeLength = range.numRows();
      const uint64_t length = rangeLength;
      if (length < kMinSpanLength) {
        piecewiseCost += length * kProbeCostInDecodedRows;
      } else if (
          length * kSpanAdvantageNumerator <
          rowCount * kSpanAdvantageDenominator) {
        // readPermutedSpan() levels with a whole-column decode at half the
        // column, which makes one of its rows cost about two decoded ones.
        piecewiseCost +=
            length * kSpanAdvantageNumerator / kSpanAdvantageDenominator;
      } else {
        piecewiseCost += rowCount;
      }
      if (piecewiseCost >= rowCount) {
        break;
      }
    }
    if (piecewiseCost < rowCount) {
      // Worth reading piecewise; the whole list is read as one span rather
      // than range by range, since a run's occurrences may be split across
      // ranges. Below kMinSpanLength rows total, the sort has nothing to
      // amortise, so the list falls back to per-range probes.
      if (totalRows >= kMinSpanLength) {
        readPermutedSpanRanges(
            ranges, static_cast<uint32_t>(totalRows), output);
        return;
      }
      readResidualRangesSeparately(ranges, output);
      return;
    }
    readWholeColumnRanges(ranges, output);
  }

  // Decodes the column once and keeps the rows the list asked for.
  void readWholeColumnRanges(
      std::span<const RowRange> ranges,
      physicalType* output) const {
    // Fully overwritten by readPhysicalBlock() below before being read.
    thread_local velox::raw_vector<physicalType> whole;
    whole.resize(this->rowCount_);
    readPhysicalBlock(0, this->rowCount_, whole.data());
    for (const auto& range : ranges) {
      const uint32_t offset = range.startRow;
      const uint32_t length = range.numRows();
      std::copy_n(whole.data() + offset, length, output);
      output += length;
    }
  }

  // What one row read on its own costs, in rows of bulk decode. Set
  // conservatively below the measured probe-to-decode ratio, since a value
  // set too high decodes gaps that a column with cheap probes should skip.
  static constexpr uint64_t kProbeCostInDecodedRows = 64;

  // The position map for one view, held per thread. Keyed on the view, since
  // one thread may read several.
  struct PositionCache {
    // 0 never matches a real viewId_, which starts at 1.
    uint64_t owner{0};
    // Always fully overwritten by positionMap() before being read, so an
    // uninitialised resize costs nothing here.
    velox::raw_vector<uint32_t> positions;
  };

  // The ratio of wanted rows to total rows below which spans beat decoding
  // the column. Must stay at or above 1: below it, length * numerator <
  // rowCount_ * denominator becomes true even at length == rowCount_,
  // wrongly sending a whole-column request down the span path, which then
  // pays an O(n) sort to rediscover that sequential reading was right all
  // along.
  static constexpr uint32_t kSpanAdvantageNumerator = 2;
  static constexpr uint32_t kSpanAdvantageDenominator = 1;

  // Below this many rows, the radix sort's own fixed cost has too little to
  // amortise and readPermutedProbes() wins instead.
  static constexpr uint32_t kMinSpanLength = 24;

  // Decodes every section in order across the whole column, undoes the
  // transform over that span, and keeps the requested rows: sections come
  // off their encodings sequentially and the inverse is applied once, rather
  // than a random access per row.
  void readWholeSpan(uint32_t offset, uint32_t length, physicalType* output)
      const {
    // Reading the entire column needs no staging buffer since the span and
    // the output are the same rows.
    if (offset == 0 && length == this->rowCount_) {
      readPhysicalBlock(0, this->rowCount_, output);
      return;
    }
    // Fully overwritten by readPhysicalBlock() below before being read.
    thread_local velox::raw_vector<physicalType> whole;
    whole.resize(this->rowCount_);
    readPhysicalBlock(0, this->rowCount_, whole.data());
    std::copy_n(whole.data() + offset, length, output);
  }

  // Reads a partial range of a Permuted-mapped stream in O(length + spans
  // touched) rather than O(length) point probes. The first ranged read on a
  // view pays the position map's O(n) build, cached thereafter per view; a
  // workload reading one small range per view will not see the amortised
  // benefit.

  // One requested row's source index, paired with where in the request it
  // belongs. A plain struct rather than std::pair, since some compilers this
  // project builds against do not treat std::pair as trivially copyable even
  // when both members are, which fails raw_vector's static_assert.
  struct SourceRow {
    uint32_t source;
    uint32_t row;
  };

  // Sorts `order`'s first `length` entries by .source in O(length) using an
  // 8-bit-digit radix sort, rather than the O(length log length) a
  // comparison sort pays or the O(rowCount_) an array sized to the source
  // range would cost. Only the digits maxSource can carry are swept, so a
  // narrower source range skips passes it does not need.
  static void radixSortBySource(
      velox::raw_vector<SourceRow>& order,
      uint32_t length,
      uint32_t maxSource) {
    thread_local velox::raw_vector<SourceRow> radixScratch;
    radixScratch.resize(length);
    SourceRow* src = order.data();
    SourceRow* dst = radixScratch.data();
    for (int shift = 0; shift < 32 && (maxSource >> shift) != 0; shift += 8) {
      uint32_t count[257] = {};
      for (uint32_t i = 0; i < length; ++i) {
        ++count[((src[i].source >> shift) & 0xFF) + 1];
      }
      for (uint32_t digit = 0; digit < 256; ++digit) {
        count[digit + 1] += count[digit];
      }
      for (uint32_t i = 0; i < length; ++i) {
        const uint32_t digit = (src[i].source >> shift) & 0xFF;
        dst[count[digit]++] = src[i];
      }
      std::swap(src, dst);
    }
    if (src != order.data()) {
      std::copy_n(src, length, order.data());
    }
  }

  void readPermutedSpan(uint32_t offset, uint32_t length, physicalType* output)
      const {
    const RowRange one{offset, offset + length};
    readPermutedSpanRanges({&one, 1}, length, output);
  }

  // The span read, over a whole range list rather than one range. Sorting the
  // whole request at once amortises the fixed per-call cost and finds the
  // longer runs, since a run's occurrences may be split across ranges;
  // reading each range on its own would pay that cost per range. `totalRows`
  // is the sum of the lengths, already computed by the caller.
  void readPermutedSpanRanges(
      std::span<const RowRange> ranges,
      uint32_t totalRows,
      physicalType* output) const {
    const auto& positions = positionMap();
    const bool seedWithConstant = constantBits_ != 0 || sections_.empty();
    if (seedWithConstant) {
      std::fill(output, output + totalRows, constantBits_);
    }

    // A run's occurrences are a contiguous block of source indices, but not
    // necessarily adjacent in the request, since another run's rows can fall
    // between them. Sorting (source index, request-relative row) pairs by
    // source index brings each run's occurrences together regardless of
    // arrival order; this grouping is shared across every section below,
    // since it depends only on the position map. Every element is
    // overwritten by the loop directly below, so an uninitialised resize
    // costs nothing here.
    thread_local velox::raw_vector<SourceRow> order;
    order.resize(totalRows);
    uint32_t at = 0;
    for (const auto& range : ranges) {
      const uint32_t rangeOffset = range.startRow;
      const uint32_t rangeLength = range.numRows();
      for (uint32_t i = 0; i < rangeLength; ++i, ++at) {
        order[at] = {positions[rangeOffset + i], at};
      }
    }
    radixSortBySource(order, totalRows, this->rowCount_ - 1);

    // Sized to the whole range rather than chunked: this path already pays
    // for a heap scratch buffer for its permuted sections, so a plain
    // section gains nothing here from the stack-sized chunking the bulk path
    // uses for cache residency.
    thread_local velox::raw_vector<uint8_t> scratch;
    scratch.resize(static_cast<size_t>(totalRows) * sizeof(physicalType));

    for (size_t s = 0; s < sections_.size(); ++s) {
      const auto& section = sections_[s];
      const bool isFirst = !seedWithConstant && s == 0;
      const bool permuted = section.transform != nullptr &&
          section.transform->positionMapping() ==
              subintsplit::PositionMapping::Permuted;
      if (!permuted) {
        // Untransformed or declined: values sit in original row order
        // already, so this is exactly readSectionChunk's job, one range at a
        // time.
        physicalType* rangeOutput = output;
        for (const auto& range : ranges) {
          const uint32_t rangeOffset = range.startRow;
          const uint32_t rangeLength = range.numRows();
          switch (section.storageBytes) {
            case 1:
              readSectionChunk<uint8_t>(
                  section,
                  rangeOffset,
                  rangeLength,
                  rangeOutput,
                  isFirst,
                  scratch.data());
              break;
            case 2:
              readSectionChunk<uint16_t>(
                  section,
                  rangeOffset,
                  rangeLength,
                  rangeOutput,
                  isFirst,
                  scratch.data());
              break;
            case 4:
              readSectionChunk<uint32_t>(
                  section,
                  rangeOffset,
                  rangeLength,
                  rangeOutput,
                  isFirst,
                  scratch.data());
              break;
            default:
              readSectionChunk<uint64_t>(
                  section,
                  rangeOffset,
                  rangeLength,
                  rangeOutput,
                  isFirst,
                  scratch.data());
              break;
          }
          rangeOutput += rangeLength;
        }
        continue;
      }
      switch (section.storageBytes) {
        case 1:
          readPermutedSpanSection<uint8_t>(
              section, totalRows, order, output, isFirst, scratch);
          break;
        case 2:
          readPermutedSpanSection<uint16_t>(
              section, totalRows, order, output, isFirst, scratch);
          break;
        case 4:
          readPermutedSpanSection<uint32_t>(
              section, totalRows, order, output, isFirst, scratch);
          break;
        default:
          readPermutedSpanSection<uint64_t>(
              section, totalRows, order, output, isFirst, scratch);
          break;
      }
    }
  }

  // Reads a range too short for readPermutedSpan()'s sort to pay. Sections
  // left in place are read as one run each, and only permuted sections pay a
  // probe per row, through a position map fetched once for the range, rather
  // than readOneRow()'s per-row map access and per-section dispatch. Kept out
  // of line so the untransformed read path it branches from stays small.
  FOLLY_NOINLINE void readPermutedProbes(
      uint32_t offset,
      uint32_t length,
      physicalType* output) const {
    const auto& positions = positionMap();
    const bool seedWithConstant = constantBits_ != 0 || sections_.empty();
    if (seedWithConstant) {
      std::fill(output, output + length, constantBits_);
    }
    thread_local velox::raw_vector<uint8_t> scratch;
    scratch.resize(static_cast<size_t>(length) * sizeof(physicalType));
    thread_local velox::raw_vector<uint64_t> probed;
    probed.resize(length);
    for (size_t s = 0; s < sections_.size(); ++s) {
      const auto& section = sections_[s];
      const bool isFirst = !seedWithConstant && s == 0;
      const bool permuted = section.transform != nullptr &&
          section.transform->positionMapping() ==
              subintsplit::PositionMapping::Permuted;
      if (!permuted) {
        switch (section.storageBytes) {
          case 1:
            readSectionChunk<uint8_t>(
                section, offset, length, output, isFirst, scratch.data());
            break;
          case 2:
            readSectionChunk<uint16_t>(
                section, offset, length, output, isFirst, scratch.data());
            break;
          case 4:
            readSectionChunk<uint32_t>(
                section, offset, length, output, isFirst, scratch.data());
            break;
          default:
            readSectionChunk<uint64_t>(
                section, offset, length, output, isFirst, scratch.data());
            break;
        }
        continue;
      }
      for (uint32_t i = 0; i < length; ++i) {
        probed[i] = section.valueAt(*section.view, positions[offset + i]);
      }
      if (isFirst) {
        subintsplit::accumulateSection<true>(
            probed.data(),
            output,
            length,
            section.mask,
            section.bitStart,
            physicalType{0});
      } else {
        subintsplit::accumulateSection<false>(
            probed.data(),
            output,
            length,
            section.mask,
            section.bitStart,
            physicalType{0});
      }
    }
  }

  // Reads one section's values for a range already grouped into (source
  // index, request-relative row) pairs sorted by source index, one bulk call
  // per consecutive-value block. Each block's values come back in source
  // order, not request order, so they are scattered into `scratch` at their
  // recorded row rather than appended.
  template <typename SectionT>
  static void readPermutedSpanSection(
      const Section& section,
      uint32_t length,
      const velox::raw_vector<SourceRow>& order,
      physicalType* output,
      bool isFirst,
      velox::raw_vector<uint8_t>& scratch) {
    auto* gathered = reinterpret_cast<SectionT*>(scratch.data());
    // The blocks are handed to the section as one ascending range list rather
    // than read one at a time, since an encoding whose random access is a
    // search (e.g. RLE) can then walk its runs once for all of them instead
    // of a binary search per block.
    thread_local std::vector<RowRange> blocks;
    blocks.clear();
    uint32_t j = 0;
    while (j < length) {
      uint32_t blockEnd = j + 1;
      while (blockEnd < length &&
             order[blockEnd].source == order[blockEnd - 1].source + 1) {
        ++blockEnd;
      }
      blocks.emplace_back(order[j].source, order[j].source + (blockEnd - j));
      j = blockEnd;
    }
    // Same zero-fill hazard as `order` above: resized once per call now, to
    // the whole request, and every element is written by the read below.
    thread_local velox::raw_vector<SectionT> block;
    block.resize(length);
    section.view->readRanges(blocks, block.data());
    for (uint32_t k = 0; k < length; ++k) {
      gathered[order[k].row] = block[k];
    }
    if (isFirst) {
      subintsplit::accumulateSection<true>(
          gathered,
          output,
          length,
          section.mask,
          section.bitStart,
          physicalType{0});
    } else {
      subintsplit::accumulateSection<false>(
          gathered,
          output,
          length,
          section.mask,
          section.bitStart,
          physicalType{0});
    }
  }

  // Reads a permuted section at its own width, puts its rows back where they
  // belong, and accumulates them through the same kernel an untransformed
  // section uses. The whole section is read because the permutation scatters
  // across it, so a row's value can sit anywhere. Kept out of line so its
  // code does not grow readPhysicalBlock's other, more common paths.
  template <typename SectionT>
  FOLLY_NOINLINE void permuteSection(
      const Section& section,
      uint32_t blockStart,
      uint32_t blockCount,
      physicalType* output,
      bool isFirst) const {
    const auto& positions = positionMap();

    // Fully overwritten below before being read, by the sequential section
    // read.
    thread_local velox::raw_vector<uint8_t> whole;
    whole.resize(static_cast<size_t>(this->rowCount_) * sizeof(SectionT));

    auto* source = reinterpret_cast<SectionT*>(whole.data());
    section.view->read(0, this->rowCount_, source);

    // Gathered straight into the output word rather than through a staging
    // buffer, since the gather is already one load per output element and
    // doing the mask and shift here adds nothing to that.
    const uint64_t mask = section.mask;
    const int shift = section.bitStart;
    const uint32_t* __restrict__ rows = positions.data() + blockStart;
    physicalType* __restrict__ out = output;
    // Seeding writes the whole word, so the caller does not have to clear the
    // output first. Two loops rather than a branch per row, which is what the
    // accumulate kernel does for the same reason.
    if (isFirst) {
      for (uint32_t row = 0; row < blockCount; ++row) {
        out[row] = static_cast<physicalType>(source[rows[row]] & mask) << shift;
      }
      return;
    }
    for (uint32_t row = 0; row < blockCount; ++row) {
      out[row] |= static_cast<physicalType>(source[rows[row]] & mask) << shift;
    }
  }

  // Reads `blockCount` rows of a transformed stream from `blockStart`, putting
  // every permuted section back in original row order. The key section is
  // stored in original order, so it is read like an untransformed section.
  void readPhysicalBlock(
      uint32_t blockStart,
      uint32_t blockCount,
      physicalType* output) const {
    // One chunk's worth, held per thread because a view is read concurrently
    // and holds no mutable state of its own. Kept to a chunk so a section's
    // values stay in L1 on the way from its view to the accumulate kernel.
    thread_local velox::raw_vector<uint8_t> scratch;
    scratch.resize(static_cast<size_t>(kViewChunkSize) * sizeof(physicalType));
    const auto isPermuted = [](const Section& section) {
      return section.transform != nullptr &&
          section.transform->positionMapping() ==
          subintsplit::PositionMapping::Permuted;
    };

    // A section that seeds writes the whole word, so the clear below is only
    // needed where nothing will.
    const bool seedWithConstant = constantBits_ != 0 || sections_.empty();
    if (seedWithConstant) {
      std::fill_n(output, blockCount, constantBits_);
    }
    bool isFirst = !seedWithConstant;
    for (size_t i = 0; i < sections_.size(); ++i) {
      const auto& section = sections_[i];
      const bool sectionSeeds = isFirst;
      isFirst = false;
      if (isPermuted(section)) {
        switch (section.storageBytes) {
          case 1:
            permuteSection<uint8_t>(
                section, blockStart, blockCount, output, sectionSeeds);
            break;
          case 2:
            permuteSection<uint16_t>(
                section, blockStart, blockCount, output, sectionSeeds);
            break;
          case 4:
            permuteSection<uint32_t>(
                section, blockStart, blockCount, output, sectionSeeds);
            break;
          default:
            permuteSection<uint64_t>(
                section, blockStart, blockCount, output, sectionSeeds);
            break;
        }
        continue;
      }
      // Straight through the accumulate kernel, from the section's own
      // storage width, a chunk at a time so the scratch stays resident.
      for (uint32_t chunk = 0; chunk < blockCount; chunk += kViewChunkSize) {
        const uint32_t chunkCount =
            std::min(kViewChunkSize, blockCount - chunk);
        const uint32_t chunkStart = blockStart + chunk;
        physicalType* chunkOutput = output + chunk;
        switch (section.storageBytes) {
          case 1:
            readSectionChunk<uint8_t>(
                section,
                chunkStart,
                chunkCount,
                chunkOutput,
                sectionSeeds,
                scratch.data());
            break;
          case 2:
            readSectionChunk<uint16_t>(
                section,
                chunkStart,
                chunkCount,
                chunkOutput,
                sectionSeeds,
                scratch.data());
            break;
          case 4:
            readSectionChunk<uint32_t>(
                section,
                chunkStart,
                chunkCount,
                chunkOutput,
                sectionSeeds,
                scratch.data());
            break;
          default:
            readSectionChunk<uint64_t>(
                section,
                chunkStart,
                chunkCount,
                chunkOutput,
                sectionSeeds,
                scratch.data());
            break;
        }
      }
    }
  }

  // Rows per chunk in readPhysical. Smaller than the encoding's chunk because
  // the scratch is on the stack, which is what keeps the view const and safe to
  // read concurrently.
  static constexpr uint32_t kViewChunkSize = 1024;

  // Source of viewId_ below, shared across every view of this T so the id
  // space has no gaps for a reused address to fall into.
  inline static std::atomic<uint64_t> nextViewId_{1};
  // Identifies this instance for PositionCache, assigned once at
  // construction and never reused, unlike `this`.
  const uint64_t viewId_;

  // Per-section transform metadata from the header, indexed by wire position.
  subintsplit::TransformInfo transformInfo_;
  // Predictor the encoder subtracted before planning, inactive when it did
  // not. Added back by the three read overrides and nowhere else.
  subintsplit::RowFrame rowFrame_;
  // True where some section carries a Permuted transform, so reads go through
  // the position map.
  bool permutedSection_{false};

  // Sections that vary per row. Constant sections are folded into constantBits_
  // at construction and do not appear here.
  std::vector<Section> sections_;
  physicalType constantBits_{0};
};

} // namespace facebook::nimble
