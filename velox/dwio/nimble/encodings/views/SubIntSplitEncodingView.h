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

#include <memory>
#include <vector>

#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/SubIntSplitAccumulate.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble {

namespace detail {

/// Serves indexed reads over a stream that has no EncodingView of its own —
/// e.g. a SubIntSplit section whose sub-stream is Zstd-compressed. Nimble
/// decompresses eagerly inside the Encoding constructor and has no
/// self-describing compressed-stream format for a view to attach to, so
/// there is nothing to wrap. This class instead decodes the stream once,
/// into an owned physicalType[rowCount] array, and serves each indexed read
/// from that array directly. Construction is not cheap and the array costs
/// rowCount * sizeof(physicalType), so it is the fallback path, not the
/// common one: of the eight encodings in the default nested inventory only
/// Varint has no view, so on an uncompressed column this class is rarely
/// built.
///
/// Nothing here is SubIntSplit-specific. SharedDictionaryAlphabet hand-rolls
/// the same fallback and could be simplified by this class; move it to views/
/// if that is done.
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
//
// Attempting construction is the only available test. A predicate cannot
// replace it: compression nests, so an RLE stream reports viewable while its
// run values are compressed a level down, and views signal both that and an
// incompatible type by throwing. See the
// compressionNestsBelowTheOuterEncoding test.
//
// Not specific to SubIntSplit sections: any caller assembling indexed
// accessors over sub-streams of unknown viewability can reuse this.
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
      : TypedEncodingView<T>{data, pool, options} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::SubIntSplit);

    const auto parsed =
        detail::parseSubIntSplitSections(data, this->dataOffset_);
    NIMBLE_CHECK(!parsed.empty(), "SubIntSplit stream has no sections.");

    for (const auto& meta : parsed) {
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

      // A Constant section contributes the same bits to every row, so resolve
      // it now and keep it out of the per-row work entirely.
      if (this->rowCount_ > 0 &&
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
    std::unique_ptr<EncodingView> view;
    // Resolved from the storage width at construction. The chunked path
    // switches instead, so that the accumulate kernel stays inlinable.
    uint64_t (*valueAt)(const EncodingView&, uint32_t){nullptr};
  };

  template <typename SectionT>
  static Section makeSection(
      const detail::SubIntSplitSection& meta,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options) {
    return Section{
        .bitStart = meta.bitStart,
        .mask = meta.mask,
        .storageBytes = meta.storageBytes,
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
      detail::accumulateSubIntSplitSection<physicalType, SectionT, true>(
          values, output, count, section.mask, section.bitStart);
    } else {
      detail::accumulateSubIntSplitSection<physicalType, SectionT, false>(
          values, output, count, section.mask, section.bitStart);
    }
  }

  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    physicalType value = constantBits_;
    for (const auto& section : sections_) {
      value |= static_cast<physicalType>(
                   section.valueAt(*section.view, index) & section.mask)
          << section.bitStart;
    }
    return detail::castFromPhysicalType<T>(value);
  }

  // Chunked the same way as SubIntSplitEncoding::materialize: with the chunk on
  // the outside and the sections on the inside, the output slice and the
  // scratch both stay resident across the section loop.
  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    if (length == 0) {
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
        // Switched rather than called through section.readChunk, which costs
        // 4% of bulk throughput: an indirect call stops the compiler inlining
        // the AVX2 accumulate kernel into the loop. The switch itself is noise
        // at one per chunk per section.
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

  // Rows per chunk in readPhysical. Smaller than the encoding's chunk because
  // the scratch is on the stack, which is what keeps the view const and safe to
  // read concurrently.
  static constexpr uint32_t kViewChunkSize = 1024;

  // Sections that vary per row. Constant sections are folded into constantBits_
  // at construction and do not appear here.
  std::vector<Section> sections_;
  physicalType constantBits_{0};
};

} // namespace facebook::nimble
