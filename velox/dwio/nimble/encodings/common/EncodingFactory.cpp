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
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"

#include <utility>

#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaBlockEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/EncodingSliceFactory.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/ForEncoding.h"
#include "velox/dwio/nimble/encodings/FsstEncoding.h"
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/PFOREncoding.h"
#include "velox/dwio/nimble/encodings/PrefixEncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"
#include "velox/dwio/nimble/encodings/SparseBoolEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/VarintEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingTypeDispatch.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

namespace facebook::nimble {

namespace {

template <typename T>
std::span<const typename TypeTraits<T>::physicalType> toPhysicalSpan(
    std::span<const T> values) {
  return std::span<const typename TypeTraits<T>::physicalType>(
      reinterpret_cast<const typename TypeTraits<T>::physicalType*>(
          values.data()),
      values.size());
}

} // namespace

std::unique_ptr<Encoding> EncodingFactory::create(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory) const {
  // Maybe we should have a magic number of encodings too? Hrm.
  const EncodingType encodingType = EncodingPrefix::encodingType(data);
  const DataType dataType = EncodingPrefix::dataType(data);
  const auto& options = options_;

  switch (encodingType) {
    case EncodingType::Trivial: {
      RETURN_ENCODING_BY_DATA_TYPE(TrivialEncoding, dataType);
    }
    case EncodingType::RLE: {
      RETURN_ENCODING_BY_DATA_TYPE(RLEEncoding, dataType);
    }
    case EncodingType::Dictionary: {
      RETURN_ENCODING_BY_NON_BOOL_TYPE(DictionaryEncoding, dataType);
    }
    case EncodingType::SharedDictionary: {
      RETURN_ENCODING_BY_INTEGER_TYPE(SharedDictionaryEncoding, dataType);
    }
    case EncodingType::FixedBitWidth: {
      RETURN_ENCODING_BY_NUMERIC_TYPE(FixedBitWidthEncoding, dataType);
    }
    case EncodingType::Nullable: {
      RETURN_ENCODING_BY_DATA_TYPE(NullableEncoding, dataType);
    }
    case EncodingType::SparseBool: {
      NIMBLE_CHECK_EQ(
          dataType,
          DataType::Bool,
          "Trying to deserialize a SparseBoolEncoding with a non-bool data type.");
      return std::make_unique<SparseBoolEncoding>(
          pool, data, stringBufferFactory, options);
    }
    case EncodingType::Varint: {
      RETURN_ENCODING_BY_VARINT_TYPE(VarintEncoding, dataType);
    }
    case EncodingType::Constant: {
      RETURN_ENCODING_BY_DATA_TYPE(ConstantEncoding, dataType);
    }
    case EncodingType::MainlyConstant: {
      RETURN_ENCODING_BY_NON_BOOL_TYPE(MainlyConstantEncoding, dataType);
    }
    case EncodingType::Prefix: {
      NIMBLE_CHECK_EQ(
          dataType,
          DataType::String,
          "Trying to deserialize a PrefixEncoding with a non-string data type.");
      return std::make_unique<PrefixEncoding>(
          pool, data, stringBufferFactory, options);
    }
    case EncodingType::Fsst: {
      NIMBLE_CHECK_EQ(
          dataType,
          DataType::String,
          "Trying to deserialize a FsstEncoding with a non-string data type.");
      return std::make_unique<FsstEncoding>(
          pool, data, stringBufferFactory, options);
    }
    case EncodingType::Delta: {
      RETURN_ENCODING_BY_NUMERIC_TYPE(DeltaEncoding, dataType);
    }
    case EncodingType::DeltaBlock: {
      RETURN_ENCODING_BY_INTEGER_TYPE(DeltaBlockEncoding, dataType);
    }
    case EncodingType::ALP: {
      switch (dataType) {
        case DataType::Float:
          return std::make_unique<ALPEncoding<float>>(
              pool, data, stringBufferFactory, options);
        case DataType::Double:
          return std::make_unique<ALPEncoding<double>>(
              pool, data, stringBufferFactory, options);
        default:
          NIMBLE_INCOMPATIBLE_ENCODING(
              "ALP encoding only supports float and double data types, got {}.",
              dataType);
      }
    }
    case EncodingType::BlockBitPacking: {
      RETURN_ENCODING_BY_NUMERIC_TYPE(BlockBitPackingEncoding, dataType);
    }
    case EncodingType::PFOR: {
      RETURN_ENCODING_BY_NUMERIC_TYPE(PFOREncoding, dataType);
    }
    case EncodingType::SimdForBitpack: {
      RETURN_ENCODING_BY_NUMERIC_TYPE(SimdForBitpackEncoding, dataType);
    }
    case EncodingType::Huffman: {
      RETURN_ENCODING_BY_INTEGER_TYPE(HuffmanEncoding, dataType);
    }
    case EncodingType::FOR: {
      RETURN_ENCODING_BY_INTEGER_TYPE(ForEncoding, dataType);
    }
    default: {
      NIMBLE_UNREACHABLE(
          "Trying to deserialize invalid EncodingType:{} -- garbage input?",
          static_cast<int>(encodingType));
    }
  }
}

std::unique_ptr<Encoding> EncodingFactory::create(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory,
    const Encoding::Options& options) const {
  return EncodingFactory{options}.create(
      pool, data, std::move(stringBufferFactory));
}

std::string_view EncodingFactory::slice(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  return EncodingSliceFactory::slice(encoded, offset, length, buffer, options);
}

template <typename T>
std::string_view EncodingFactory::encode(
    std::unique_ptr<EncodingSelectionPolicy<T>>&& selectorPolicy,
    std::span<const T> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  using physicalType = typename TypeTraits<T>::physicalType;
  auto physicalValues = toPhysicalSpan(values);
  auto statistics = Statistics<physicalType>::create(physicalValues);
  auto selectionResult =
      selectorPolicy->select(physicalValues, statistics, options);
  EncodingSelection<physicalType> selection{
      std::move(selectionResult),
      std::move(statistics),
      std::move(selectorPolicy)};
  return EncodingFactory::encode<T>(
      std::move(selection), physicalValues, buffer, options);
}

// The encoding layout is honored, except for nullable encodings, which are
// replaced with the appropriate nullable encoding.
template <typename T>
std::string_view EncodingFactory::encodeWithCapturedLayout(
    std::string_view encodedLayoutSource,
    std::span<const T> values,
    Buffer& buffer,
    const Encoding::Options& options,
    std::string_view missingChildContext) {
  auto policy = std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
      EncodingLayoutCapture::capture(encodedLayoutSource, options),
      CompressionOptions{},
      [missingChildContext](
          DataType dataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
        NIMBLE_FAIL(
            "{} is missing nested child for {}.",
            missingChildContext,
            dataType);
      });
  return EncodingFactory::encode<T>(std::move(policy), values, buffer, options);
}

template <typename T>
std::string_view EncodingFactory::encodeNullable(
    std::unique_ptr<EncodingSelectionPolicy<T>>&& selectorPolicy,
    std::span<const T> values,
    std::span<const bool> nulls,
    Buffer& buffer,
    const Encoding::Options& options) {
  using physicalType = typename TypeTraits<T>::physicalType;
  auto physicalValues = toPhysicalSpan(values);
  auto statistics = Statistics<physicalType>::create(physicalValues);
  auto selectionResult = selectorPolicy->selectNullable(
      physicalValues, nulls, statistics, options);
  EncodingSelection<physicalType> selection{
      std::move(selectionResult),
      std::move(statistics),
      std::move(selectorPolicy)};
  return EncodingFactory::encodeNullable<T>(
      std::move(selection), physicalValues, nulls, buffer, options);
}

template <typename T>
std::string_view EncodingFactory::encode(
    EncodingSelection<typename TypeTraits<T>::physicalType>&& selection,
    std::span<const typename TypeTraits<T>::physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  using physicalType = typename TypeTraits<T>::physicalType;
  auto castedValues = toPhysicalSpan(values);
  switch (selection.encodingType()) {
    case EncodingType::Constant: {
      return ConstantEncoding<T>::encode(
          selection, castedValues, buffer, options);
    }
    case EncodingType::Trivial: {
      return TrivialEncoding<T>::encode(
          selection, castedValues, buffer, options);
    }
    case EncodingType::RLE: {
      return RLEEncoding<T>::encode(selection, castedValues, buffer, options);
    }
    case EncodingType::Dictionary: {
      if constexpr (std::is_same<T, bool>::value) {
        NIMBLE_INCOMPATIBLE_ENCODING(
            "Dictionary encoding cannot be used for boolean data.");
      }
      if (castedValues.empty()) {
        // A replayed Dictionary layout can land on an empty stream (e.g. a
        // MainlyConstant whose OtherValues are all the common value). Reject it
        // as an incompatible encoding -- like the other data-requiring
        // encodings here -- so the writer retries without the captured layout
        // instead of aborting on an empty dictionary.
        NIMBLE_INCOMPATIBLE_ENCODING(
            "Dictionary encoding cannot be used with 0 rows.");
      }
      return DictionaryEncoding<T>::encode(
          selection, castedValues, buffer, options);
    }
    case EncodingType::SharedDictionary: {
      if constexpr (isIntegralType<T>() && !std::is_same_v<T, bool>) {
        return SharedDictionaryEncoding<T>::encode(
            selection, castedValues, buffer, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "SharedDictionary encoding only supports non-bool integer data types, got {}.",
          TypeTraits<T>::dataType);
    }
    case EncodingType::FixedBitWidth: {
      if constexpr (isNumericType<physicalType>()) {
        return FixedBitWidthEncoding<T>::encode(
            selection, castedValues, buffer, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "FixedBitWidth encoding should not be selected for non-numeric data types.");
    }
    case EncodingType::Varint: {
      // TODO: we can support floating point types, but currently Statistics
      // doesn't calculate buckets for floating point types. We should convert
      // floating point types to their physical type, and then Statistics and
      // Varint encoding will just work.
      if constexpr (
          isNumericType<physicalType>() &&
          (sizeof(physicalType) == 4 || sizeof(T) == 8)) {
        return VarintEncoding<T>::encode(
            selection, castedValues, buffer, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "Varint encoding can only be selected for large numeric data types.");
    }
    case EncodingType::MainlyConstant: {
      if constexpr (!std::is_same<T, bool>::value) {
        return MainlyConstantEncoding<T>::encode(
            selection, castedValues, buffer, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "MainlyConstant encoding should not be selected for bool data types.");
    }
    case EncodingType::SparseBool: {
      if constexpr (std::is_same<T, bool>::value) {
        return SparseBoolEncoding::encode(
            selection, castedValues, buffer, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "SparseBool encoding should not be selected for non-bool data types.");
    }
    case EncodingType::Prefix: {
      if constexpr (std::is_same<T, std::string_view>::value) {
        return PrefixEncoding::encode(selection, castedValues, buffer, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "Prefix encoding should only be selected for string_view data types.");
    }
    case EncodingType::Fsst: {
      if constexpr (std::is_same<T, std::string_view>::value) {
        return FsstEncoding::encode(selection, castedValues, buffer, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "Fsst encoding should only be selected for string_view data types.");
    }
    case EncodingType::Delta: {
      if constexpr (isNumericType<physicalType>()) {
        return DeltaEncoding<T>::encode(
            selection, castedValues, buffer, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "Delta encoding should not be selected for non-numeric data type: {}.",
          TypeTraits<T>::dataType);
    }
    case EncodingType::DeltaBlock: {
      if constexpr (isIntegralType<T>()) {
        return DeltaBlockEncoding<T>::encode(
            selection, castedValues, buffer, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "DeltaBlock encoding only supports integral data types, got {}.",
          TypeTraits<T>::dataType);
    }
    case EncodingType::ALP: {
      if constexpr (isFloatingPointType<T>()) {
        return ALPEncoding<T>::encode(selection, castedValues, buffer, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "ALP encoding should only be selected for float or double data types, got {}.",
          TypeTraits<T>::dataType);
    }
    case EncodingType::BlockBitPacking: {
      if constexpr (isNumericType<physicalType>()) {
        return BlockBitPackingEncoding<T>::encode(
            selection, castedValues, buffer, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "BlockBitPacking encoding should not be selected for non-numeric data types.");
    }
    case EncodingType::PFOR: {
      if constexpr (isIntegralType<physicalType>()) {
        return PFOREncoding<T>::encode(
            selection, castedValues, buffer, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "PFOR encoding should not be selected for non-integral data type: {}.",
          TypeTraits<T>::dataType);
    }
    case EncodingType::SimdForBitpack: {
      if constexpr (isIntegralType<physicalType>()) {
        return SimdForBitpackEncoding<T>::encode(
            selection, castedValues, buffer, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "SimdForBitpack encoding only supports integral data types, got {}.",
          TypeTraits<T>::dataType);
    }
    case EncodingType::Huffman: {
      if constexpr (isIntegralType<physicalType>()) {
        return HuffmanEncoding<T>::encode(
            selection, castedValues, buffer, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "Huffman encoding only supports integral data types, got {}.",
          TypeTraits<T>::dataType);
    }
    case EncodingType::FOR: {
      if constexpr (isIntegralType<physicalType>()) {
        return ForEncoding<T>::encode(selection, castedValues, buffer, options);
      }
      NIMBLE_INCOMPATIBLE_ENCODING(
          "FOR encoding only supports integral data types, got {}.",
          TypeTraits<T>::dataType);
    }
    default: {
      NIMBLE_UNSUPPORTED(
          "Encoding {} is not supported.", selection.encodingType());
    }
  }
}

template <typename T>
std::string_view EncodingFactory::encodeNullable(
    EncodingSelection<typename TypeTraits<T>::physicalType>&& selection,
    std::span<const typename TypeTraits<T>::physicalType> values,
    std::span<const bool> nulls,
    Buffer& buffer,
    const Encoding::Options& options) {
  auto physicalValues = toPhysicalSpan(values);
  switch (selection.encodingType()) {
    case EncodingType::Nullable: {
      return NullableEncoding<T>::encodeNullable(
          selection, physicalValues, nulls, buffer, options);
    }
    default: {
      NIMBLE_UNSUPPORTED(
          "Encoding {} is not supported for nullable data.",
          selection.encodingType());
    }
  }
}

#define DEFINE_TEMPLATES(type)                                                 \
  template std::string_view EncodingFactory::encode<type>(                     \
      std::unique_ptr<EncodingSelectionPolicy<type>> && selectorPolicy,        \
      std::span<const type> values,                                            \
      Buffer & buffer,                                                         \
      const Encoding::Options& options);                                       \
  template std::string_view EncodingFactory::encodeWithCapturedLayout<type>(   \
      std::string_view encodedLayoutSource,                                    \
      std::span<const type> values,                                            \
      Buffer & buffer,                                                         \
      const Encoding::Options& options,                                        \
      std::string_view missingChildContext);                                   \
  template std::string_view EncodingFactory::encodeNullable<type>(             \
      std::unique_ptr<EncodingSelectionPolicy<type>> && selectorPolicy,        \
      std::span<const type> values,                                            \
      std::span<const bool> nulls,                                             \
      Buffer & buffer,                                                         \
      const Encoding::Options& options);                                       \
  template std::string_view EncodingFactory::encode<type>(                     \
      EncodingSelection<typename TypeTraits<type>::physicalType> && selection, \
      std::span<const typename TypeTraits<type>::physicalType> values,         \
      Buffer & buffer,                                                         \
      const Encoding::Options& options);                                       \
  template std::string_view EncodingFactory::encodeNullable<type>(             \
      EncodingSelection<typename TypeTraits<type>::physicalType> && selection, \
      std::span<const typename TypeTraits<type>::physicalType> values,         \
      std::span<const bool> nulls,                                             \
      Buffer & buffer,                                                         \
      const Encoding::Options& options);

DEFINE_TEMPLATES(int8_t);
DEFINE_TEMPLATES(uint8_t);
DEFINE_TEMPLATES(int16_t);
DEFINE_TEMPLATES(uint16_t);
DEFINE_TEMPLATES(int32_t);
DEFINE_TEMPLATES(uint32_t);
DEFINE_TEMPLATES(int64_t);
DEFINE_TEMPLATES(uint64_t);
DEFINE_TEMPLATES(float);
DEFINE_TEMPLATES(double);
DEFINE_TEMPLATES(bool);
DEFINE_TEMPLATES(std::string_view);

} // namespace facebook::nimble
