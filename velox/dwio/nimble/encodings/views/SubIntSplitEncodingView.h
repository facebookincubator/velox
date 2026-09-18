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
#include <span>
#include <string_view>
#include <vector>

#include <folly/ScopeGuard.h>
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

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
          serialized.encodedSize, static_cast<size_t>(data.end() - position));
      auto view = createEncodingView(
          {position, serialized.encodedSize}, this->pool_, options);
      NIMBLE_CHECK_EQ(view->rowCount(), this->rowCount_);
      const auto width = serialized.bitEnd - serialized.bitStart + 1;
      const auto storageBytes = sectionStorageBytes(width);
      NIMBLE_CHECK_EQ(view->dataType(), sectionDataType(storageBytes));
      sections_.push_back({
          .bitStart = serialized.bitStart,
          .mask = width == 64 ? ~uint64_t{0} : (uint64_t{1} << width) - 1,
          .storageBytes = storageBytes,
          .view = std::move(view),
      });
      position += serialized.encodedSize;
    }
    NIMBLE_CHECK_EQ(position, data.end());
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
  void mergeSectionAt(
      const Section& section,
      std::span<const uint32_t> indices,
      physicalType* output) const {
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

  physicalType readPhysicalAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    physicalType value{0};
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
    std::fill(output, output + length, physicalType{0});
    for (const auto& section : sections_) {
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
    std::fill(output, output + indices.size(), physicalType{0});
    for (const auto& section : sections_) {
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

  // Sections collectively cover every physical bit exactly once.
  std::vector<Section> sections_;
};

} // namespace facebook::nimble
