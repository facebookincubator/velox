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
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/subintsplit/BitSection.h"
#include "velox/dwio/nimble/encodings/subintsplit/Format.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionAccumulator.h"
#include "velox/dwio/nimble/encodings/subintsplit/TuningConfig.h"

/// Read context:
///
///   section headers + payloads -> [child decoders and cursors]
///                                      |
///                                      v
///                         decodeChunk -> accumulated values
///
/// All dynamic children represent the same logical row. Reset, skip, and decode
/// must advance them together; constant children contribute bits without a
/// cursor.

namespace facebook::nimble::subintsplit {

/// The nested encodings a SubIntSplit stream is split into, and the machinery
/// to stitch them back into whole values.
///
/// Owns the read cursor of every section: skip() and decodeChunk() advance all
/// of them together, so the table always presents one consistent stream
/// position.
template <typename PhysicalType>
class SectionTable {
 public:
  SectionTable(velox::memory::MemoryPool& pool, uint32_t decodeChunkSize)
      : decodeChunkSize_{decodeChunkSize}, pool_{pool}, scratch_{&pool} {}

  uint32_t decodeChunkSize() const noexcept {
    return decodeChunkSize_;
  }

  /// Parses `numSections` section headers at `pos` and builds a nested encoding
  /// for each.
  ///
  /// Separate from the constructor because the owning encoding only knows where
  /// the section headers start once its base class has parsed the common
  /// Encoding prefix, by which point the table already needs its memory pool.
  void load(
      std::string_view data,
      uint8_t numSections,
      const std::function<void*(uint32_t)>& stringBufferFactory,
      const Encoding::Options& options);

  void reset();

  void skip(uint32_t numRows);

  /// Combines every section for `numValues` values, starting at the current
  /// cursors, into `output`. `numValues` must not exceed decodeChunkSize().
  void decodeChunk(uint32_t numValues, PhysicalType* output);

  /// True when a single section reproduces each value verbatim -- it spans the
  /// full width, starts at bit 0, and no constant section contributes. The
  /// mask, the shift and the OR are then all identities, so the caller can hand
  /// its own buffer to decodePassThrough() and skip the scratch round-trip.
  bool isPassThrough() const noexcept {
    return passThrough_;
  }

  /// Decodes straight into `output` with no combining. Only valid when
  /// isPassThrough().
  void decodePassThrough(uint32_t numValues, PhysicalType* output) {
    sections_[dynamicSections_.front()].encoding->materialize(
        numValues, output);
  }

  size_t numSections() const noexcept {
    return sections_.size();
  }

  std::string debugString(int offset) const;

 private:
  struct Section {
    BitSection range;
    uint64_t mask{0};

    // 1, 2, 4, or 8 -- matches the section's nested DataType.
    uint8_t storageBytes{8};

    std::unique_ptr<Encoding> encoding;
  };

  // Reads a Constant section's single value and folds it into constantOr_.
  void foldConstantSection(const Section& section);

  // Decides whether one section can be handed the caller's buffer directly.
  void computePassThrough();

  std::vector<Section> sections_;

  // Indices into sections_ that have to be decoded per chunk.
  //
  // A Constant section contributes the same bits to every value, so decoding it
  // would mean materializing thousands of copies of one number and OR-ing them
  // in a value at a time. Those sections are resolved once at construction into
  // constantOr_ and dropped from the per-chunk loop. This is the common case
  // rather than a corner one: the selector emits a Constant section for every
  // constant high prefix or low suffix it trims.
  std::vector<uint32_t> dynamicSections_;

  // Combined contribution of every Constant section, pre-masked and
  // pre-shifted, OR-ed into each output value by the first dynamic section.
  PhysicalType constantOr_{0};

  bool passThrough_{false};

  const uint32_t decodeChunkSize_;

  velox::memory::MemoryPool& pool_;

  // Holds one chunk of one section's values between materialize() and the
  // accumulate pass. Sized for the widest storage type on first use.
  Vector<uint8_t> scratch_;
};

template <typename PhysicalType>
void SectionTable<PhysicalType>::load(
    std::string_view data,
    uint8_t numSections,
    const std::function<void*(uint32_t)>& stringBufferFactory,
    const Encoding::Options& options) {
  const uint32_t sectionHeaderBytes =
      static_cast<uint32_t>(numSections) * kSectionHeaderSize;
  NIMBLE_CHECK_FILE(
      data.size() >= sectionHeaderBytes,
      "SubIntSplit section headers are truncated.");

  const char* pos = data.data();
  std::vector<SectionHeader> headers;
  headers.reserve(numSections);
  uint32_t expectedBitStart{0};
  uint64_t totalEncodedSize{0};
  for (uint8_t i = 0; i < numSections; ++i) {
    headers.push_back(readSectionHeader(pos));
    const auto& header = headers.back();
    NIMBLE_CHECK_FILE(
        header.range.bitStart == expectedBitStart &&
            header.range.bitEnd >= header.range.bitStart &&
            header.range.bitEnd < sizeof(PhysicalType) * 8,
        "SubIntSplit sections must cover the value bits once in order.");
    expectedBitStart = header.range.bitEnd + 1;
    totalEncodedSize += header.encodedSize;
  }
  NIMBLE_CHECK_FILE(
      expectedBitStart == sizeof(PhysicalType) * 8,
      "SubIntSplit sections must cover the value bits once in order.");
  NIMBLE_CHECK_FILE(
      totalEncodedSize == data.size() - sectionHeaderBytes,
      "SubIntSplit section payload sizes do not match the stream.");

  sections_.resize(numSections);
  for (uint8_t i = 0; i < numSections; ++i) {
    auto& section = sections_[i];
    section.range = headers[i].range;
    section.mask = section.range.mask();
    section.storageBytes = sectionStorageBytes(section.range.width());

    const std::string_view sectionData{pos, headers[i].encodedSize};
    NIMBLE_CHECK_FILE(
        sectionData.size() >= EncodingPrefix::kRowCountOffset,
        "SubIntSplit section encoding prefix is truncated.");
    const auto expectedDataType =
        dispatchStorageType(section.storageBytes, []<typename StorageType>() {
          return TypeTraits<StorageType>::dataType;
        });
    NIMBLE_CHECK_FILE(
        EncodingPrefix::dataType(sectionData) == expectedDataType,
        "SubIntSplit section data type does not match its bit width.");
    section.encoding = EncodingFactory().create(
        pool_, sectionData, stringBufferFactory, options);
    pos += headers[i].encodedSize;

    if (section.encoding->encodingType() == EncodingType::Constant) {
      foldConstantSection(section);
    } else {
      dynamicSections_.push_back(i);
    }
  }

  computePassThrough();
}

template <typename PhysicalType>
void SectionTable<PhysicalType>::foldConstantSection(const Section& section) {
  // ConstantEncoding is stateless -- its reset() and skip() are no-ops and
  // materialize() ignores the read position -- so reading its value here does
  // not disturb the cursor the decode path relies on, and the section can be
  // left out of the chunk loop entirely.
  const uint64_t value = dispatchStorageType(
      section.storageBytes, [&]<typename StorageType>() -> uint64_t {
        StorageType stored{0};
        section.encoding->materialize(1, &stored);
        return static_cast<uint64_t>(stored);
      });

  constantOr_ |= static_cast<PhysicalType>(value & section.mask)
      << section.range.bitStart;
}

template <typename PhysicalType>
void SectionTable<PhysicalType>::computePassThrough() {
  if (dynamicSections_.size() != 1 || constantOr_ != 0) {
    return;
  }
  const auto& only = sections_[dynamicSections_.front()];
  constexpr uint64_t kFullMask =
      ~uint64_t{0} >> (64 - sizeof(PhysicalType) * 8);
  passThrough_ = only.range.bitStart == 0 &&
      only.storageBytes == sizeof(PhysicalType) && only.mask == kFullMask;
}

template <typename PhysicalType>
void SectionTable<PhysicalType>::reset() {
  for (auto& section : sections_) {
    section.encoding->reset();
  }
}

template <typename PhysicalType>
void SectionTable<PhysicalType>::skip(uint32_t numRows) {
  for (auto& section : sections_) {
    section.encoding->skip(numRows);
  }
}

template <typename PhysicalType>
void SectionTable<PhysicalType>::decodeChunk(
    uint32_t numValues,
    PhysicalType* output) {
  // Every section was constant, so there is nothing to decode and each value is
  // just the folded constant.
  if (dynamicSections_.empty()) [[unlikely]] {
    std::fill_n(output, numValues, constantOr_);
    return;
  }

  const uint64_t scratchBytes =
      static_cast<uint64_t>(numValues) * sizeof(PhysicalType);
  if (scratch_.size() < scratchBytes) [[unlikely]] {
    scratch_.resize(scratchBytes);
  }

  for (size_t i = 0; i < dynamicSections_.size(); ++i) {
    const auto& section = sections_[dynamicSections_[i]];

    // The first dynamic section initialises each output element with a pure
    // write, OR-ing in the constant sections' bits; the rest OR theirs in. That
    // saves a separate fill pass over the chunk.
    const bool isFirst = (i == 0);

    dispatchStorageType(section.storageBytes, [&]<typename StorageType>() {
      auto* values = reinterpret_cast<StorageType*>(scratch_.data());
      section.encoding->materialize(numValues, values);
      if (isFirst) {
        accumulateSection<true>(
            values,
            output,
            numValues,
            section.mask,
            section.range.bitStart,
            constantOr_);
      } else {
        accumulateSection<false>(
            values,
            output,
            numValues,
            section.mask,
            section.range.bitStart,
            PhysicalType{0});
      }
    });
  }
}

template <typename PhysicalType>
std::string SectionTable<PhysicalType>::debugString(int offset) const {
  const std::string indent(offset, ' ');
  std::string result;
  for (const auto& section : sections_) {
    result += indent + "  [" + std::to_string(section.range.bitStart) + ".." +
        std::to_string(section.range.bitEnd) +
        "] storageBytes=" + std::to_string(section.storageBytes) + "\n";
    result += section.encoding->debugString(offset + 4);
    result += "\n";
  }
  return result;
}

} // namespace facebook::nimble::subintsplit
