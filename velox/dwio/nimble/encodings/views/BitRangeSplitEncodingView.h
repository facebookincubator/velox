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

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "folly/ScopeGuard.h"
#include "velox/dwio/nimble/encodings/BitRangeSplitEncoding.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble {

/// Provides direct random access to a self-describing bit-range split stream.
template <typename T>
class BitRangeSplitEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  /// Constructs a random-access view over a self-describing encoded chunk.
  BitRangeSplitEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        header_{Base::parseHeader(data, options)} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::BitRangeSplit);
    NIMBLE_CHECK_EQ(header_.rowCount, this->rowCount_);
    nestedViews_.reserve(header_.sections.size());
    for (const auto& section : header_.sections) {
      auto nested = createEncodingView(
          {section.data, section.dataBytes}, this->pool_, options);
      NIMBLE_CHECK_EQ(nested->rowCount(), header_.rowCount);
      NIMBLE_CHECK_EQ(
          nested->dataType(), Base::sectionDataType(section.storageBytes));
      nestedViews_.push_back(std::move(nested));
    }
  }

 private:
  using Base = detail::BitRangeSplitEncodingBase;
  using Section = Base::Section;

  // Places one decoded child value into its range in the output value.
  template <typename SectionType>
  static physicalType mergeValue(
      physicalType value,
      SectionType sectionValue,
      const Section& section) {
    const auto shifted =
        (static_cast<uint64_t>(sectionValue) & Base::sectionMask(section))
        << section.bitStart;
    return static_cast<physicalType>(static_cast<uint64_t>(value) | shifted);
  }

  // Reads and merges one child for a contiguous row range.
  template <typename SectionType>
  void mergeSection(
      size_t sectionIndex,
      uint32_t offset,
      uint32_t length,
      physicalType* output) const {
    auto values = this->template getVectorBuffer<SectionType>();
    SCOPE_EXIT {
      this->releaseVectorBuffer(values);
    };
    values.resize(length);
    nestedViews_[sectionIndex]->read(offset, length, values.data());
    const auto& section = header_.sections[sectionIndex];
    for (uint32_t row{0}; row < length; ++row) {
      output[row] = mergeValue(output[row], values[row], section);
    }
  }

  // Reads and merges one child for arbitrary row indices.
  template <typename SectionType>
  void mergeSectionAt(
      size_t sectionIndex,
      std::span<const uint32_t> indices,
      physicalType* output) const {
    auto values = this->template getVectorBuffer<SectionType>();
    SCOPE_EXIT {
      this->releaseVectorBuffer(values);
    };
    values.resize(indices.size());
    nestedViews_[sectionIndex]->readAt(indices, values.data());
    const auto& section = header_.sections[sectionIndex];
    for (size_t indexOffset{0}; indexOffset < indices.size(); ++indexOffset) {
      output[indexOffset] =
          mergeValue(output[indexOffset], values[indexOffset], section);
    }
  }

  // Dispatches each child using the integer type that contains its range.
  template <typename Callback>
  void forEachNestedSection(Callback&& callback) const {
    for (size_t sectionIndex{0}; sectionIndex < header_.sections.size();
         ++sectionIndex) {
      [&] {
        NIMBLE_RETURN_BY_UNSIGNED_INTEGER_DATA_TYPE(
            Base::sectionDataType(header_.sections[sectionIndex].storageBytes),
            SectionType,
            callback.template operator()<SectionType>(sectionIndex));
      }();
    }
  }

  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    physicalType value{0};
    forEachNestedSection([&]<typename SectionType>(size_t sectionIndex) {
      SectionType sectionValue{0};
      nestedViews_[sectionIndex]->readAt(index, &sectionValue);
      value = mergeValue(value, sectionValue, header_.sections[sectionIndex]);
    });
    return detail::castFromPhysicalType<T>(value);
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    std::fill(output, output + length, physicalType{0});
    forEachNestedSection([&]<typename SectionType>(size_t sectionIndex) {
      mergeSection<SectionType>(sectionIndex, offset, length, output);
    });
  }

  void readPhysicalAt(std::span<const uint32_t> indices, physicalType* output)
      const final {
    std::fill(output, output + indices.size(), physicalType{0});
    forEachNestedSection([&]<typename SectionType>(size_t sectionIndex) {
      mergeSectionAt<SectionType>(sectionIndex, indices, output);
    });
  }

  // Validated metadata and child-stream boundaries for this chunk.
  Base::Header header_{};
  // Random-access views ordered by the corresponding range in header_.sections.
  std::vector<std::unique_ptr<EncodingView>> nestedViews_{};
};

} // namespace facebook::nimble
