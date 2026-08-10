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
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

#include <algorithm>
#include <array>

#include "velox/dwio/nimble/encodings/views/ALPEncodingView.h"
#include "velox/dwio/nimble/encodings/views/BlockBitPackingEncodingView.h"
#include "velox/dwio/nimble/encodings/views/ConstantEncodingView.h"
#include "velox/dwio/nimble/encodings/views/DeltaBlockEncodingView.h"
#include "velox/dwio/nimble/encodings/views/DictionaryEncodingView.h"
#include "velox/dwio/nimble/encodings/views/FOREncodingView.h"
#include "velox/dwio/nimble/encodings/views/FixedBitWidthEncodingView.h"
#include "velox/dwio/nimble/encodings/views/HuffmanEncodingView.h"
#include "velox/dwio/nimble/encodings/views/MainlyConstantEncodingView.h"
#include "velox/dwio/nimble/encodings/views/PFOREncodingView.h"
#include "velox/dwio/nimble/encodings/views/RLEEncodingView.h"
#include "velox/dwio/nimble/encodings/views/SimdForBitpackEncodingView.h"
#include "velox/dwio/nimble/encodings/views/SparseBoolEncodingView.h"
#include "velox/dwio/nimble/encodings/views/TrivialEncodingView.h"

namespace facebook::nimble {
namespace detail {

template <typename T>
std::unique_ptr<TypedEncodingView<T>> createTypedEncodingView(
    std::string_view data,
    velox::memory::MemoryPool* pool,
    const Encoding::Options& options) {
  using physicalType = typename TypeTraits<T>::physicalType;
  const auto encodingType =
      static_cast<EncodingType>(data[EncodingPrefix::kEncodingTypeOffset]);
  switch (encodingType) {
    case EncodingType::Constant:
      return std::make_unique<ConstantEncodingView<T>>(data, pool, options);
    case EncodingType::Trivial:
      return std::make_unique<TrivialEncodingView<T>>(data, pool, options);
    case EncodingType::MainlyConstant:
      // MainlyConstant supports all leaf types except bool.
      if constexpr (!std::is_same_v<T, bool>) {
        return std::make_unique<MainlyConstantEncodingView<T>>(
            data, pool, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "MainlyConstant encoding cannot be used for boolean data.");
    case EncodingType::ALP:
      if constexpr (isFloatingPointType<T>()) {
        return std::make_unique<ALPEncodingView<T>>(data, pool, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "ALP encoding only supports float and double data types, got {}.",
          TypeTraits<T>::dataType);
    case EncodingType::FixedBitWidth:
      if constexpr (isNumericType<physicalType>()) {
        return std::make_unique<FixedBitWidthEncodingView<T>>(
            data, pool, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "FixedBitWidth encoding should not be selected for non-numeric data types.");
    case EncodingType::Dictionary:
      if constexpr (!std::is_same_v<T, bool>) {
        return std::make_unique<DictionaryEncodingView<T>>(data, pool, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "Dictionary encoding cannot be used for boolean data.");
    case EncodingType::SparseBool:
      if constexpr (std::is_same_v<T, bool>) {
        return std::make_unique<SparseBoolEncodingView>(data, pool, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "SparseBool encoding only supports boolean data.");
    case EncodingType::RLE:
      return std::make_unique<RLEEncodingView<T>>(data, pool, options);
    case EncodingType::FOR:
      if constexpr (isIntegralType<physicalType>()) {
        return std::make_unique<FOREncodingView<T>>(data, pool, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "FOR encoding should not be selected for non-integral data type: {}.",
          TypeTraits<T>::dataType);
    case EncodingType::DeltaBlock:
      if constexpr (isIntegralType<T>()) {
        return std::make_unique<DeltaBlockEncodingView<T>>(data, pool, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "DeltaBlock encoding only supports integral data types, got {}.",
          TypeTraits<T>::dataType);
    case EncodingType::Huffman:
      if constexpr (
          isIntegralType<physicalType>() &&
          !std::is_same_v<physicalType, bool>) {
        return std::make_unique<HuffmanEncodingView<T>>(data, pool, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "Huffman encoding should not be selected for non-integral data type: {}.",
          TypeTraits<T>::dataType);
    case EncodingType::PFOR:
      if constexpr (isIntegralType<physicalType>()) {
        return std::make_unique<PFOREncodingView<T>>(data, pool, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "PFOR encoding should not be selected for non-integral data type: {}.",
          TypeTraits<T>::dataType);
    case EncodingType::SimdForBitpack:
      if constexpr (isIntegralType<physicalType>()) {
        return std::make_unique<SimdForBitpackEncodingView<T>>(
            data, pool, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "SimdForBitpack encoding only supports integral data types, got {}.",
          TypeTraits<T>::dataType);
    case EncodingType::BlockBitPacking:
      if constexpr (isNumericType<physicalType>()) {
        return std::make_unique<BlockBitPackingEncodingView<T>>(
            data, pool, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "BlockBitPacking encoding should not be selected for non-numeric data types.");
    default:
      NIMBLE_UNSUPPORTED("{} does not support EncodingView.", encodingType);
  }
}

#define INSTANTIATE_CREATE_TYPED_ENCODING_VIEW(T)                         \
  template std::unique_ptr<TypedEncodingView<T>> createTypedEncodingView( \
      std::string_view data,                                              \
      velox::memory::MemoryPool* pool,                                    \
      const Encoding::Options& options)

INSTANTIATE_CREATE_TYPED_ENCODING_VIEW(int8_t);
INSTANTIATE_CREATE_TYPED_ENCODING_VIEW(uint8_t);
INSTANTIATE_CREATE_TYPED_ENCODING_VIEW(int16_t);
INSTANTIATE_CREATE_TYPED_ENCODING_VIEW(uint16_t);
INSTANTIATE_CREATE_TYPED_ENCODING_VIEW(int32_t);
INSTANTIATE_CREATE_TYPED_ENCODING_VIEW(uint32_t);
INSTANTIATE_CREATE_TYPED_ENCODING_VIEW(int64_t);
INSTANTIATE_CREATE_TYPED_ENCODING_VIEW(uint64_t);
INSTANTIATE_CREATE_TYPED_ENCODING_VIEW(float);
INSTANTIATE_CREATE_TYPED_ENCODING_VIEW(double);
INSTANTIATE_CREATE_TYPED_ENCODING_VIEW(bool);
INSTANTIATE_CREATE_TYPED_ENCODING_VIEW(std::string_view);

#undef INSTANTIATE_CREATE_TYPED_ENCODING_VIEW

} // namespace detail

bool supportsEncodingView(EncodingType encodingType) {
  // Keep in sync with the encoding dispatch in createTypedEncodingView().
  static constexpr std::array kViewableEncodings{
      EncodingType::Constant,
      EncodingType::Trivial,
      EncodingType::MainlyConstant,
      EncodingType::ALP,
      EncodingType::FixedBitWidth,
      EncodingType::Dictionary,
      EncodingType::SparseBool,
      EncodingType::RLE,
      EncodingType::FOR,
      EncodingType::DeltaBlock,
      EncodingType::Huffman,
      EncodingType::PFOR,
      EncodingType::SimdForBitpack,
      EncodingType::BlockBitPacking};
  return std::find(
             kViewableEncodings.begin(),
             kViewableEncodings.end(),
             encodingType) != kViewableEncodings.end();
}

std::unique_ptr<EncodingView> createEncodingView(
    std::string_view data,
    velox::memory::MemoryPool* pool,
    const Encoding::Options& options) {
  const auto dataType = EncodingPrefix::dataType(data);
  switch (dataType) {
    case DataType::Int8:
      return detail::createTypedEncodingView<int8_t>(data, pool, options);
    case DataType::Uint8:
      return detail::createTypedEncodingView<uint8_t>(data, pool, options);
    case DataType::Int16:
      return detail::createTypedEncodingView<int16_t>(data, pool, options);
    case DataType::Uint16:
      return detail::createTypedEncodingView<uint16_t>(data, pool, options);
    case DataType::Int32:
      return detail::createTypedEncodingView<int32_t>(data, pool, options);
    case DataType::Uint32:
      return detail::createTypedEncodingView<uint32_t>(data, pool, options);
    case DataType::Int64:
      return detail::createTypedEncodingView<int64_t>(data, pool, options);
    case DataType::Uint64:
      return detail::createTypedEncodingView<uint64_t>(data, pool, options);
    case DataType::Float:
      return detail::createTypedEncodingView<float>(data, pool, options);
    case DataType::Double:
      return detail::createTypedEncodingView<double>(data, pool, options);
    case DataType::Bool:
      return detail::createTypedEncodingView<bool>(data, pool, options);
    case DataType::String:
      return detail::createTypedEncodingView<std::string_view>(
          data, pool, options);
    default:
      NIMBLE_UNSUPPORTED("{} does not support EncodingView.", dataType);
  }
}

} // namespace facebook::nimble
