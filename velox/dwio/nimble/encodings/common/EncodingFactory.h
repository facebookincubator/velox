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
#include <span>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"

namespace facebook::nimble {

template <typename T>
class EncodingSelection;

template <typename T>
class EncodingSelectionPolicy;

class EncodingFactory {
 public:
  explicit EncodingFactory(Encoding::Options options = {})
      : options_{options} {}

  virtual ~EncodingFactory() = default;

  /// Creates an Encoding from serialized data using this factory's options.
  virtual std::unique_ptr<Encoding> create(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory) const;

  /// Creates an Encoding from serialized data using the provided options while
  /// preserving this factory's concrete implementation.
  virtual std::unique_ptr<Encoding> create(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory,
      const Encoding::Options& options) const;

  const Encoding::Options& options() const {
    return options_;
  }

  template <typename T>
  static std::string_view encode(
      std::unique_ptr<EncodingSelectionPolicy<T>>&& selectorPolicy,
      std::span<const T> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

  template <typename T>
  static std::string_view encodeWithCapturedLayout(
      std::string_view encodedLayoutSource,
      std::span<const T> values,
      Buffer& buffer,
      const Encoding::Options& options = {},
      std::string_view missingChildContext = "Captured encoding layout");

  template <typename T>
  static std::string_view encodeNullable(
      std::unique_ptr<EncodingSelectionPolicy<T>>&& selectorPolicy,
      std::span<const T> values,
      std::span<const bool> nulls,
      Buffer& buffer,
      const Encoding::Options& options = {});

  /// Creates a serialized encoding over the row range [offset, offset + length)
  /// of an existing encoded buffer. Native slices are used when possible;
  /// otherwise the range is materialized and re-encoded.
  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {});

 private:
  Encoding::Options options_{};

  template <typename T>
  static std::string_view encode(
      EncodingSelection<typename TypeTraits<T>::physicalType>&& selection,
      std::span<const typename TypeTraits<T>::physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

  template <typename T>
  static std::string_view encodeNullable(
      EncodingSelection<typename TypeTraits<T>::physicalType>&& selection,
      std::span<const typename TypeTraits<T>::physicalType> values,
      std::span<const bool> nulls,
      Buffer& buffer,
      const Encoding::Options& options = {});

  friend class EncodingSelection<int8_t>;
  friend class EncodingSelection<uint8_t>;
  friend class EncodingSelection<int16_t>;
  friend class EncodingSelection<uint16_t>;
  friend class EncodingSelection<int32_t>;
  friend class EncodingSelection<uint32_t>;
  friend class EncodingSelection<int64_t>;
  friend class EncodingSelection<uint64_t>;
  friend class EncodingSelection<float>;
  friend class EncodingSelection<double>;
  friend class EncodingSelection<bool>;
  friend class EncodingSelection<std::string_view>;
};

} // namespace facebook::nimble
