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
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <folly/ScopeGuard.h>
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/encodings/views/TrivialEncodingView.h"

/// Implements the read view for SubIntSplit, whose serialized payload contains
/// metadata for contiguous, non-overlapping bit sections followed by one
/// independently encoded child stream per section. Construction validates that
/// the sections cover the physical integer exactly once, creates each child
/// view, folds constant children into a cached value, and identifies a child
/// that can decode directly into the caller's physical output buffer.
///
/// Serialized payload
/// +--------+-------------+-------------------------------+
/// | header | descriptors | encoded section child streams |
/// +--------+-------------+-------------------------------+
///              |                         |
///              v                         v
///       validate full bit         create child views
///            coverage                    |
///                                        v
///                          +-------------+--------------+
///                          | Constant: cache bits once  |
///                          | Direct: decode into output |
///                          | Other: decode then merge   |
///                          +-------------+--------------+
///                                        |
///                    +-------------------+-------------------+
///                    v                   v                   v
///              scalar read       contiguous batch     projected batch
///              merge each        direct + generic     ordered fusion or
///              child             section merges       generic merges
///                    +-------------------+-------------------+
///                                        |
///                                        v
///                         complete physical value -> T
///
/// Every read reconstructs values by masking, shifting, and OR-ing section
/// values into their physical positions. Contiguous reads reuse typed scratch
/// buffers for non-direct children. Projected reads additionally access
/// Trivial children in place and fuse an ordered direct-plus-Trivial layout
/// into one output traversal. Layouts outside these cases use the same generic
/// child-decoding and merge path.

namespace facebook::nimble {

/// Provides random access to integers encoded as independent bit-range
/// streams.
template <typename T>
class SubIntSplitEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  SubIntSplitEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options} {
    static_assert(
        sizeof(physicalType) == 4 || sizeof(physicalType) == 8,
        "SubIntSplit only supports 32- and 64-bit physical types");
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::SubIntSplit);

    const char* position = data.data() + this->dataOffset_;
    const auto numSections = encoding::read<uint8_t>(position);
    // Skip the reserved section-order byte.
    encoding::read<uint8_t>(position);
    NIMBLE_CHECK_GT(numSections, 0);

    struct SerializedSection {
      uint8_t bitStart;
      uint8_t bitEnd;
      uint32_t encodedSize;
    };
    std::vector<SerializedSection> serializedSections;
    serializedSections.reserve(numSections);
    uint8_t expectedBitStart{0};
    for (uint8_t section{0}; section < numSections; ++section) {
      const auto bitStart = encoding::read<uint8_t>(position);
      const auto bitEnd = encoding::read<uint8_t>(position);
      const auto encodedSize = encoding::readUint32(position);
      NIMBLE_CHECK_EQ(bitStart, expectedBitStart);
      NIMBLE_CHECK_GE(bitEnd, bitStart);
      NIMBLE_CHECK_LT(bitEnd, sizeof(physicalType) * 8);
      serializedSections.push_back({bitStart, bitEnd, encodedSize});
      expectedBitStart = bitEnd + 1;
    }
    NIMBLE_CHECK_EQ(expectedBitStart, sizeof(physicalType) * 8);

    sections_.reserve(numSections);
    for (const auto& serialized : serializedSections) {
      NIMBLE_CHECK_LE(
          serialized.encodedSize,
          static_cast<size_t>(data.data() + data.size() - position));
      auto view = createEncodingView(
          {position, serialized.encodedSize}, this->pool_, options);
      NIMBLE_CHECK_EQ(view->rowCount(), this->rowCount_);
      const auto width = serialized.bitEnd - serialized.bitStart + 1;
      const auto storageBytes = sectionStorageBytes(width);
      NIMBLE_CHECK_EQ(view->dataType(), sectionDataType(storageBytes));
      Section section{
          .bitStart = serialized.bitStart,
          .mask = width == 64 ? ~uint64_t{0} : (uint64_t{1} << width) - 1,
          .storageBytes = storageBytes,
          .view = std::move(view),
      };
      // Constant children produce the same positioned bits for every row.
      // Fold them now and omit them from all scalar and batch read loops.
      if (section.view->encodingType() == EncodingType::Constant &&
          this->rowCount_ > 0) {
        constantValue_ = mergeConstantSection(section, constantValue_);
      } else {
        sections_.push_back(std::move(section));
        // A physical-width child can write directly into the caller's output;
        // narrower children still require typed scratch storage.
        if (storageBytes == sizeof(physicalType)) {
          directSectionIndex_ = sections_.size() - 1;
        }
      }
      position += serialized.encodedSize;
    }
    NIMBLE_CHECK_EQ(position, data.data() + data.size());
  }

 private:
  struct Section {
    uint8_t bitStart{0};
    uint64_t mask{0};
    uint8_t storageBytes{0};
    std::unique_ptr<EncodingView> view;
  };

  static constexpr uint8_t sectionStorageBytes(uint8_t bitWidth) {
    if (bitWidth <= 8) {
      return 1;
    }
    if (bitWidth <= 16) {
      return 2;
    }
    if (bitWidth <= 32) {
      return 4;
    }
    return 8;
  }

  static DataType sectionDataType(uint8_t storageBytes) {
    switch (storageBytes) {
      case 1:
        return DataType::Uint8;
      case 2:
        return DataType::Uint16;
      case 4:
        return DataType::Uint32;
      case 8:
        return DataType::Uint64;
      default:
        NIMBLE_UNREACHABLE("Invalid SubIntSplit storage width.");
    }
  }

  /// Reads a constant child once and folds its positioned bits into value.
  static physicalType mergeConstantSection(
      const Section& section,
      physicalType value) {
    switch (section.storageBytes) {
      case 1: {
        uint8_t sectionValue{0};
        section.view->readAt(0, &sectionValue);
        return mergeValue(value, sectionValue, section);
      }
      case 2: {
        uint16_t sectionValue{0};
        section.view->readAt(0, &sectionValue);
        return mergeValue(value, sectionValue, section);
      }
      case 4: {
        uint32_t sectionValue{0};
        section.view->readAt(0, &sectionValue);
        return mergeValue(value, sectionValue, section);
      }
      case 8: {
        uint64_t sectionValue{0};
        section.view->readAt(0, &sectionValue);
        return mergeValue(value, sectionValue, section);
      }
      default:
        NIMBLE_UNREACHABLE(
            "Invalid SubIntSplit storage width: {}", section.storageBytes);
    }
  }

  template <typename SectionType>
  static physicalType mergeValue(
      physicalType value,
      SectionType sectionValue,
      const Section& section) {
    return value |
        static_cast<physicalType>(
               (static_cast<uint64_t>(sectionValue) & section.mask)
               << section.bitStart);
  }

  template <typename SectionType>
  void mergeSection(
      const Section& section,
      uint32_t offset,
      uint32_t length,
      physicalType* output) const {
    auto values = this->template getVectorBuffer<SectionType>();
    SCOPE_EXIT {
      this->releaseVectorBuffer(values);
    };
    values.resize(length);
    section.view->read(offset, length, values.data());
    for (uint32_t i{0}; i < length; ++i) {
      output[i] = mergeValue(output[i], values[i], section);
    }
  }

  template <typename SectionType>
  /// Merges sparse values directly for trivial children, avoiding a temporary
  /// vector and a second traversal.
  void mergeSectionAt(
      const Section& section,
      std::span<const uint32_t> indices,
      physicalType* output) const {
    if (section.view->encodingType() == EncodingType::Trivial) {
      const auto* view = static_cast<const TrivialEncodingView<SectionType>*>(
          section.view.get());
      for (size_t i{0}; i < indices.size(); ++i) {
        output[i] = mergeValue(output[i], view->readAt(indices[i]), section);
      }
      return;
    }
    auto values = this->template getVectorBuffer<SectionType>();
    SCOPE_EXIT {
      this->releaseVectorBuffer(values);
    };
    values.resize(indices.size());
    section.view->readAt(indices, values.data());
    for (size_t i{0}; i < indices.size(); ++i) {
      output[i] = mergeValue(output[i], values[i], section);
    }
  }

  template <typename SectionType>
  /// Fuses the direct and trivial child merges into one output traversal.
  static void mergeDirectAndTrivialSectionAt(
      const Section& directSection,
      const Section& trivialSection,
      std::span<const uint32_t> indices,
      physicalType constantValue,
      physicalType* output) {
    const auto* view = static_cast<const TrivialEncodingView<SectionType>*>(
        trivialSection.view.get());
    for (size_t i{0}; i < indices.size(); ++i) {
      output[i] = mergeValue(
          mergeValue(constantValue, output[i], directSection),
          view->readAt(indices[i]),
          trivialSection);
    }
  }

  /// Uses the fused ordered-read path when the layout and batch amortize its
  /// setup; returns false when the generic path should handle the read.
  bool tryMergeDirectAndTrivialSectionAt(
      size_t directSectionIndex,
      std::span<const uint32_t> indices,
      physicalType* output) const {
    if (sections_.size() != 2 || indices.size() < 16 ||
        !std::is_sorted(indices.begin(), indices.end())) {
      return false;
    }
    const auto trivialSectionIndex = 1 - directSectionIndex;
    const auto& trivialSection = sections_[trivialSectionIndex];
    if (trivialSection.view->encodingType() != EncodingType::Trivial) {
      return false;
    }

    const auto& directSection = sections_[directSectionIndex];
    switch (trivialSection.storageBytes) {
      case 1:
        mergeDirectAndTrivialSectionAt<uint8_t>(
            directSection, trivialSection, indices, constantValue_, output);
        return true;
      case 2:
        mergeDirectAndTrivialSectionAt<uint16_t>(
            directSection, trivialSection, indices, constantValue_, output);
        return true;
      case 4:
        mergeDirectAndTrivialSectionAt<uint32_t>(
            directSection, trivialSection, indices, constantValue_, output);
        return true;
      case 8:
        mergeDirectAndTrivialSectionAt<uint64_t>(
            directSection, trivialSection, indices, constantValue_, output);
        return true;
      default:
        NIMBLE_UNREACHABLE(
            "Invalid SubIntSplit storage width: {}",
            trivialSection.storageBytes);
    }
  }

  physicalType readPhysicalAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    physicalType value{constantValue_};
    for (const auto& section : sections_) {
      switch (section.storageBytes) {
        case 1: {
          uint8_t sectionValue{0};
          section.view->readAt(index, &sectionValue);
          value = mergeValue(value, sectionValue, section);
          break;
        }
        case 2: {
          uint16_t sectionValue{0};
          section.view->readAt(index, &sectionValue);
          value = mergeValue(value, sectionValue, section);
          break;
        }
        case 4: {
          uint32_t sectionValue{0};
          section.view->readAt(index, &sectionValue);
          value = mergeValue(value, sectionValue, section);
          break;
        }
        case 8: {
          uint64_t sectionValue{0};
          section.view->readAt(index, &sectionValue);
          value = mergeValue(value, sectionValue, section);
          break;
        }
        default:
          NIMBLE_UNREACHABLE(
              "Invalid SubIntSplit storage width: {}", section.storageBytes);
      }
    }
    return value;
  }

  T readTypedAt(uint32_t index) const final {
    return detail::castFromPhysicalType<T>(readPhysicalAt(index));
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    // Seed output through the widest child when available. This avoids a
    // zero-fill, scratch allocation, and copy for that section.
    if (directSectionIndex_.has_value()) {
      const auto& section = sections_[*directSectionIndex_];
      section.view->read(offset, length, output);
      if (directSectionNeedsMerge(section)) {
        for (uint32_t i{0}; i < length; ++i) {
          output[i] = mergeValue(constantValue_, output[i], section);
        }
      }
    } else {
      // With no direct child, constants are the only bits known before the
      // remaining sections are decoded and merged.
      std::fill(output, output + length, constantValue_);
    }
    // Merge every non-direct child exactly once; the direct child already
    // occupies output and folded constants no longer appear in sections_.
    for (size_t sectionIndex{0}; sectionIndex < sections_.size();
         ++sectionIndex) {
      if (directSectionIndex_ == sectionIndex) {
        continue;
      }
      const auto& section = sections_[sectionIndex];
      switch (section.storageBytes) {
        case 1:
          mergeSection<uint8_t>(section, offset, length, output);
          break;
        case 2:
          mergeSection<uint16_t>(section, offset, length, output);
          break;
        case 4:
          mergeSection<uint32_t>(section, offset, length, output);
          break;
        case 8:
          mergeSection<uint64_t>(section, offset, length, output);
          break;
        default:
          NIMBLE_UNREACHABLE(
              "Invalid SubIntSplit storage width: {}", section.storageBytes);
      }
    }
  }

  void readPhysicalAt(std::span<const uint32_t> indices, physicalType* output)
      const final {
    // Projected reads use the same direct-output seed as contiguous reads, but
    // may additionally fuse an ordered Trivial child into that traversal.
    if (directSectionIndex_.has_value()) {
      const auto directSectionIndex = *directSectionIndex_;
      const auto& section = sections_[directSectionIndex];
      section.view->readAt(indices, output);
      if (directSectionNeedsMerge(section)) {
        if (tryMergeDirectAndTrivialSectionAt(
                directSectionIndex, indices, output)) {
          return;
        }
        for (size_t i{0}; i < indices.size(); ++i) {
          output[i] = mergeValue(constantValue_, output[i], section);
        }
      }
    } else {
      std::fill(output, output + indices.size(), constantValue_);
    }
    // The fused helper returns early after consuming both live sections;
    // otherwise this loop handles every non-direct section generically.
    for (size_t sectionIndex{0}; sectionIndex < sections_.size();
         ++sectionIndex) {
      if (directSectionIndex_ == sectionIndex) {
        continue;
      }
      const auto& section = sections_[sectionIndex];
      switch (section.storageBytes) {
        case 1:
          mergeSectionAt<uint8_t>(section, indices, output);
          break;
        case 2:
          mergeSectionAt<uint16_t>(section, indices, output);
          break;
        case 4:
          mergeSectionAt<uint32_t>(section, indices, output);
          break;
        case 8:
          mergeSectionAt<uint64_t>(section, indices, output);
          break;
        default:
          NIMBLE_UNREACHABLE(
              "Invalid SubIntSplit storage width: {}", section.storageBytes);
      }
    }
  }

  /// Returns whether direct decoding still needs bit placement or constants.
  bool directSectionNeedsMerge(const Section& section) const {
    return section.bitStart != 0 ||
        section.mask != std::numeric_limits<physicalType>::max() ||
        constantValue_ != 0;
  }

  // Sections collectively cover every physical bit exactly once.
  std::vector<Section> sections_;
  // Positioned bits shared by every row after Constant children are folded.
  physicalType constantValue_{0};
  // Live child whose storage width matches physicalType and can decode into
  // the caller-provided output buffer without a typed scratch vector.
  std::optional<size_t> directSectionIndex_;
};

} // namespace facebook::nimble
