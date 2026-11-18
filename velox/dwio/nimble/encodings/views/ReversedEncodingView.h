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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/ReversedEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble {

/// Reads a Reversed stream by mapping back the rows its nested view returns.
/// The view keeps no state of its own, so Encoding::Options::cacheEncodingView
/// applies only to the nested view, which it is passed to.
template <typename T>
class ReversedEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  /// Opens a Reversed stream and the view of its nested stream.
  ReversedEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        values_{createValues(data, options)} {
    NIMBLE_CHECK_FILE_EQ(
        values_->rowCount(),
        this->rowCount_,
        "Reversed nested stream has the wrong row count.");
  }

 private:
  // Returns the nested view of the stored values in Reversed stream 'data'.
  std::unique_ptr<TypedEncodingView<physicalType>> createValues(
      std::string_view data,
      const Encoding::Options& options) const {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::Reversed);
    const auto nested = data.substr(this->dataOffset_);

    // Checks the data type before creating the nested view, which would read
    // the values as that type.
    NIMBLE_CHECK_FILE_GE(
        nested.size(),
        EncodingPrefix::kRowCountOffset,
        "Truncated encoding prefix.");
    NIMBLE_CHECK_FILE_EQ(
        EncodingPrefix::readDataType(nested),
        TypeTraits<physicalType>::dataType,
        "Reversed nested stream has the wrong data type.");
    auto values = detail::createTypedEncodingView<physicalType>(
        nested, this->pool_, options);
    NIMBLE_CHECK_NOT_NULL(values);
    return values;
  }

  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    physicalType value;
    values_->readAt(index, &value);
    ReversedEncoding<T>::flip(&value, 1);
    return static_cast<T>(value);
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    values_->read(offset, length, output);
    ReversedEncoding<T>::flip(output, length);
  }

  void readPhysicalAt(std::span<const uint32_t> indices, physicalType* output)
      const final {
    values_->readAt(indices, output);
    ReversedEncoding<T>::flip(output, indices.size());
  }

  // Nested view of the stored values.
  const std::unique_ptr<TypedEncodingView<physicalType>> values_;
};

} // namespace facebook::nimble
