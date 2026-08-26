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

/// Serves an encoding by index after decoding it once into an owned array.
///
/// This is the fallback for a sub-stream that createTypedEncodingView() cannot
/// wrap, which in practice means a compressed stream or a Varint one. It is
/// worth having rather than falling back to a sequential cursor because
/// decompression in Nimble is eager and private: an Encoding decompresses into
/// its own buffer in its constructor (TrivialEncoding.h), and
/// Compression::uncompress yields the values body rather than a self-describing
/// stream, so there is no decompressed form for a real view to attach to.
/// Decoding once into an array recovers indexed access at the cost of holding
/// the section decoded, which is the same trade SharedDictionaryAlphabet makes
/// for a non-viewable alphabet (SharedDictionaryEncoding.cpp).
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

} // namespace detail

/// Random-access view over a SubIntSplit stream.
///
/// SubIntSplitEncoding splits each value into contiguous bit ranges and encodes
/// each range as its own sub-stream. Reading row i therefore only needs row i of
/// every section, but the encoding holds its sections as Encoding objects, whose
/// only read API is a sequential cursor -- so a point lookup costs a traversal
/// from row zero. This view holds one indexed accessor per section instead, and
/// reassembles the word from the pieces.
///
/// The sections a real selection produces are overwhelmingly viewable: of the
/// eight encodings in the default nested inventory, Varint alone has no view.
/// Anything that is not viewable, and any section whose sub-stream is
/// compressed, falls back to MaterializedEncodingView, so indexed access
/// survives either way.
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

    for (const auto& parsed :
         detail::parseSubIntSplitSections(data, this->dataOffset_)) {
      Section section;
      section.bitStart = parsed.bitStart;
      section.mask = parsed.mask;
      section.storageBytes = parsed.storageBytes;
      switch (section.storageBytes) {
        case 1:
          section.view = makeSectionView<uint8_t>(parsed.stream, pool, options);
          break;
        case 2:
          section.view =
              makeSectionView<uint16_t>(parsed.stream, pool, options);
          break;
        case 4:
          section.view =
              makeSectionView<uint32_t>(parsed.stream, pool, options);
          break;
        case 8:
          section.view =
              makeSectionView<uint64_t>(parsed.stream, pool, options);
          break;
        default:
          NIMBLE_UNREACHABLE("Invalid SubIntSplit section storage width.");
      }
      NIMBLE_CHECK_EQ(section.view->rowCount(), this->rowCount_);
      sections_.push_back(std::move(section));
    }
    NIMBLE_CHECK(!sections_.empty(), "SubIntSplit stream has no sections.");
  }

 private:
  struct Section {
    int bitStart{0};
    uint64_t mask{0};
    uint8_t storageBytes{8};
    std::unique_ptr<EncodingView> view;
  };

  // Prefers a real view over the section's sub-stream, and decodes the section
  // once when it cannot have one.
  //
  // Whether a stream is compressed is not visible from the outside -- each
  // encoding stores its compression type at its own offset, and
  // supportsEncodingView() sees only the encoding type -- so the attempt is the
  // test. createTypedEncodingView() throws on a compressed stream and on a type
  // it has no view for, and both are exactly the cases the fallback exists to
  // serve. A stream that is genuinely malformed still fails, because the
  // fallback decodes it through EncodingFactory and lets that error escape.
  template <typename SectionT>
  static std::unique_ptr<EncodingView> makeSectionView(
      std::string_view stream,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options) {
    if (supportsEncodingView(EncodingPrefix::encodingType(stream))) {
      try {
        return detail::createTypedEncodingView<SectionT>(stream, pool, options);
      } catch (const NimbleException&) {
        // Fall through to the materialized fallback below.
      }
    }
    return std::make_unique<detail::MaterializedEncodingView<SectionT>>(
        stream, pool, options);
  }

  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    physicalType value{0};
    for (const auto& section : sections_) {
      value |= static_cast<physicalType>(
                   sectionValueAt(section, index) & section.mask)
          << section.bitStart;
    }
    return detail::castFromPhysicalType<T>(value);
  }

  // Reads one section's raw value. The width switch is explicit rather than
  // writing through a uint64_t, because readAt() writes exactly the section's
  // storage width and reading the rest back would depend on byte order.
  static uint64_t sectionValueAt(const Section& section, uint32_t index) {
    switch (section.storageBytes) {
      case 1: {
        uint8_t v;
        section.view->readAt(index, &v);
        return v;
      }
      case 2: {
        uint16_t v;
        section.view->readAt(index, &v);
        return v;
      }
      case 4: {
        uint32_t v;
        section.view->readAt(index, &v);
        return v;
      }
      default: {
        uint64_t v;
        section.view->readAt(index, &v);
        return v;
      }
    }
  }

  // Chunked the same way as SubIntSplitEncoding::materialize, and for the same
  // reason: with the chunk on the outside and the sections on the inside, the
  // output slice and the scratch buffer both stay resident across the whole
  // section loop. Only the source of each section's values differs -- an
  // indexed read here against a sequential one there.
  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    if (length == 0) {
      return;
    }

    // One chunk of one section's values, on the stack because a view is read
    // concurrently and so cannot hold scratch of its own. kViewChunkSize is
    // smaller than the encoding's chunk for the same reason it exists at all:
    // at 1024 the scratch and the output slice together are 16 KB and stay in
    // L1 across the whole section loop.
    alignas(64) uint8_t scratch[kViewChunkSize * sizeof(physicalType)];

    for (uint32_t chunkStart = 0; chunkStart < length;
         chunkStart += kViewChunkSize) {
      const uint32_t chunkCount = std::min(kViewChunkSize, length - chunkStart);
      physicalType* chunkOutput = output + chunkStart;
      const uint32_t sourceOffset = offset + chunkStart;

      for (size_t s = 0; s < sections_.size(); ++s) {
        const auto& section = sections_[s];
        // Section 0 initialises each output element; the rest OR into it.
        const bool isFirst = (s == 0);
        switch (section.storageBytes) {
          case 1:
            readSectionChunk<uint8_t>(
                section, sourceOffset, chunkCount, chunkOutput, isFirst,
                scratch);
            break;
          case 2:
            readSectionChunk<uint16_t>(
                section, sourceOffset, chunkCount, chunkOutput, isFirst,
                scratch);
            break;
          case 4:
            readSectionChunk<uint32_t>(
                section, sourceOffset, chunkCount, chunkOutput, isFirst,
                scratch);
            break;
          case 8:
            readSectionChunk<uint64_t>(
                section, sourceOffset, chunkCount, chunkOutput, isFirst,
                scratch);
            break;
          default:
            NIMBLE_UNREACHABLE("Invalid SubIntSplit section storage width.");
        }
      }
    }
  }

  template <typename SectionT>
  static void readSectionChunk(
      const Section& section,
      uint32_t sourceOffset,
      uint32_t count,
      physicalType* chunkOutput,
      bool isFirst,
      uint8_t* scratch) {
    auto* values = reinterpret_cast<SectionT*>(scratch);
    section.view->read(sourceOffset, count, values);
    if (isFirst) {
      detail::accumulateSubIntSplitSection<physicalType, SectionT, true>(
          values, chunkOutput, count, section.mask, section.bitStart);
    } else {
      detail::accumulateSubIntSplitSection<physicalType, SectionT, false>(
          values, chunkOutput, count, section.mask, section.bitStart);
    }
  }

  // Rows per chunk in readPhysical. See the scratch declaration there.
  static constexpr uint32_t kViewChunkSize = 1024;

  std::vector<Section> sections_;
};

} // namespace facebook::nimble
