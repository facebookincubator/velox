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
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

namespace facebook::nimble::legacy {

class EncodingFactory : public ::facebook::nimble::EncodingFactory {
 public:
  using ::facebook::nimble::EncodingFactory::EncodingFactory;

  std::unique_ptr<Encoding> create(
      velox::memory::MemoryPool& memoryPool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory) const override;

  std::unique_ptr<Encoding> create(
      velox::memory::MemoryPool& memoryPool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory,
      const Encoding::Options& options) const override;

  template <typename T>
  static std::string_view encode(
      std::unique_ptr<EncodingSelectionPolicy<T>>&& selectorPolicy,
      std::span<const T> values,
      Buffer& buffer);

  template <typename T>
  static std::string_view encodeNullable(
      std::unique_ptr<EncodingSelectionPolicy<T>>&& selectorPolicy,
      std::span<const T> values,
      std::span<const bool> nulls,
      Buffer& buffer);

 private:
  template <typename T>
  static std::string_view encode(
      EncodingSelection<
          typename ::facebook::nimble::TypeTraits<T>::physicalType>&& selection,
      std::span<const typename ::facebook::nimble::TypeTraits<T>::physicalType>
          values,
      Buffer& buffer);

  template <typename T>
  static std::string_view encodeNullable(
      EncodingSelection<
          typename ::facebook::nimble::TypeTraits<T>::physicalType>&& selection,
      std::span<const typename ::facebook::nimble::TypeTraits<T>::physicalType>
          values,
      std::span<const bool> nulls,
      Buffer& buffer);

  friend class ::facebook::nimble::EncodingSelection<int8_t>;
  friend class ::facebook::nimble::EncodingSelection<uint8_t>;
  friend class ::facebook::nimble::EncodingSelection<int16_t>;
  friend class ::facebook::nimble::EncodingSelection<uint16_t>;
  friend class ::facebook::nimble::EncodingSelection<int32_t>;
  friend class ::facebook::nimble::EncodingSelection<uint32_t>;
  friend class ::facebook::nimble::EncodingSelection<int64_t>;
  friend class ::facebook::nimble::EncodingSelection<uint64_t>;
  friend class ::facebook::nimble::EncodingSelection<float>;
  friend class ::facebook::nimble::EncodingSelection<double>;
  friend class ::facebook::nimble::EncodingSelection<bool>;
  friend class ::facebook::nimble::EncodingSelection<std::string_view>;
};

} // namespace facebook::nimble::legacy
