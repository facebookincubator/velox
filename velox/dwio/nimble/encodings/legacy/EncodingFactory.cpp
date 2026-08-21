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
#include "velox/dwio/nimble/encodings/legacy/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaBlockEncoding.h"
#include "velox/dwio/nimble/encodings/FsstEncoding.h"
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/PFOREncoding.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"
// SubIntSplit integration commented out (disabled):
/*
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#endif
*/
#include "velox/dwio/nimble/encodings/legacy/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/DeltaEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/PrefixEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/SparseBoolEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/VarintEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

// Import necessary types from parent namespace
using ::facebook::nimble::isNumericType;
using ::facebook::nimble::Statistics;
using ::facebook::nimble::toString;
using ::facebook::nimble::TypeTraits;

namespace facebook::nimble::legacy {

namespace {

template <typename T>
static std::span<const typename TypeTraits<T>::physicalType> toPhysicalSpan(
    std::span<const T> values) {
  return std::span<const typename TypeTraits<T>::physicalType>(
      reinterpret_cast<const typename TypeTraits<T>::physicalType*>(
          values.data()),
      values.size());
}
} // namespace

std::unique_ptr<Encoding> EncodingFactory::create(
    velox::memory::MemoryPool& memoryPool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory) const {
  // Maybe we should have a magic number of encodings too? Hrm.
  const EncodingType encodingType = static_cast<EncodingType>(data[0]);
  const DataType dataType = static_cast<DataType>(data[1]);
#define RETURN_ENCODING_BY_LEAF_TYPE(Encoding, dataType)                  \
  switch (dataType) {                                                     \
    case DataType::Int8:                                                  \
      return std::make_unique<Encoding<int8_t>>(                          \
          memoryPool, data, stringBufferFactory);                         \
    case DataType::Uint8:                                                 \
      return std::make_unique<Encoding<uint8_t>>(                         \
          memoryPool, data, stringBufferFactory);                         \
    case DataType::Int16:                                                 \
      return std::make_unique<Encoding<int16_t>>(                         \
          memoryPool, data, stringBufferFactory);                         \
    case DataType::Uint16:                                                \
      return std::make_unique<Encoding<uint16_t>>(                        \
          memoryPool, data, stringBufferFactory);                         \
    case DataType::Int32:                                                 \
      return std::make_unique<Encoding<int32_t>>(                         \
          memoryPool, data, stringBufferFactory);                         \
    case DataType::Uint32:                                                \
      return std::make_unique<Encoding<uint32_t>>(                        \
          memoryPool, data, stringBufferFactory);                         \
    case DataType::Int64:                                                 \
      return std::make_unique<Encoding<int64_t>>(                         \
          memoryPool, data, stringBufferFactory);                         \
    case DataType::Uint64:                                                \
      return std::make_unique<Encoding<uint64_t>>(                        \
          memoryPool, data, stringBufferFactory);                         \
    case DataType::Float:                                                 \
      return std::make_unique<Encoding<float>>(                           \
          memoryPool, data, stringBufferFactory);                         \
    case DataType::Double:                                                \
      return std::make_unique<Encoding<double>>(                          \
          memoryPool, data, stringBufferFactory);                         \
    case DataType::Bool:                                                  \
      return std::make_unique<Encoding<bool>>(                            \
          memoryPool, data, stringBufferFactory);                         \
    case DataType::String:                                                \
      return std::make_unique<Encoding<std::string_view>>(                \
          memoryPool, data, stringBufferFactory);                         \
    default:                                                              \
      NIMBLE_UNREACHABLE("Unknown encoding type {}.", toString(dataType)) \
  }

#define RETURN_ENCODING_BY_VARINT_TYPE(Encoding, dataType) \
  switch (dataType) {                                      \
    case DataType::Int32:                                  \
      return std::make_unique<Encoding<int32_t>>(          \
          memoryPool, data, stringBufferFactory);          \
    case DataType::Uint32:                                 \
      return std::make_unique<Encoding<uint32_t>>(         \
          memoryPool, data, stringBufferFactory);          \
    case DataType::Int64:                                  \
      return std::make_unique<Encoding<int64_t>>(          \
          memoryPool, data, stringBufferFactory);          \
    case DataType::Uint64:                                 \
      return std::make_unique<Encoding<uint64_t>>(         \
          memoryPool, data, stringBufferFactory);          \
    case DataType::Float:                                  \
      return std::make_unique<Encoding<float>>(            \
          memoryPool, data, stringBufferFactory);          \
    case DataType::Double:                                 \
      return std::make_unique<Encoding<double>>(           \
          memoryPool, data, stringBufferFactory);          \
    default:                                               \
      NIMBLE_UNREACHABLE(                                  \
          "Trying to deserialize a varint stream for "     \
          "an incompatible data type {}.",                 \
          toString(dataType));                             \
  }

#define RETURN_ENCODING_BY_NON_BOOL_TYPE(Encoding, dataType) \
  switch (dataType) {                                        \
    case DataType::Int8:                                     \
      return std::make_unique<Encoding<int8_t>>(             \
          memoryPool, data, stringBufferFactory);            \
    case DataType::Uint8:                                    \
      return std::make_unique<Encoding<uint8_t>>(            \
          memoryPool, data, stringBufferFactory);            \
    case DataType::Int16:                                    \
      return std::make_unique<Encoding<int16_t>>(            \
          memoryPool, data, stringBufferFactory);            \
    case DataType::Uint16:                                   \
      return std::make_unique<Encoding<uint16_t>>(           \
          memoryPool, data, stringBufferFactory);            \
    case DataType::Int32:                                    \
      return std::make_unique<Encoding<int32_t>>(            \
          memoryPool, data, stringBufferFactory);            \
    case DataType::Uint32:                                   \
      return std::make_unique<Encoding<uint32_t>>(           \
          memoryPool, data, stringBufferFactory);            \
    case DataType::Int64:                                    \
      return std::make_unique<Encoding<int64_t>>(            \
          memoryPool, data, stringBufferFactory);            \
    case DataType::Uint64:                                   \
      return std::make_unique<Encoding<uint64_t>>(           \
          memoryPool, data, stringBufferFactory);            \
    case DataType::Float:                                    \
      return std::make_unique<Encoding<float>>(              \
          memoryPool, data, stringBufferFactory);            \
    case DataType::Double:                                   \
      return std::make_unique<Encoding<double>>(             \
          memoryPool, data, stringBufferFactory);            \
    case DataType::String:                                   \
      return std::make_unique<Encoding<std::string_view>>(   \
          memoryPool, data, stringBufferFactory);            \
    default:                                                 \
      NIMBLE_UNREACHABLE(                                    \
          "Trying to deserialize a non-bool stream for "     \
          "the bool data type {}.",                          \
          toString(dataType));                               \
  }

#define RETURN_ENCODING_BY_NUMERIC_TYPE(Encoding, dataType) \
  switch (dataType) {                                       \
    case DataType::Int8:                                    \
      return std::make_unique<Encoding<int8_t>>(            \
          memoryPool, data, stringBufferFactory);           \
    case DataType::Uint8:                                   \
      return std::make_unique<Encoding<uint8_t>>(           \
          memoryPool, data, stringBufferFactory);           \
    case DataType::Int16:                                   \
      return std::make_unique<Encoding<int16_t>>(           \
          memoryPool, data, stringBufferFactory);           \
    case DataType::Uint16:                                  \
      return std::make_unique<Encoding<uint16_t>>(          \
          memoryPool, data, stringBufferFactory);           \
    case DataType::Int32:                                   \
      return std::make_unique<Encoding<int32_t>>(           \
          memoryPool, data, stringBufferFactory);           \
    case DataType::Uint32:                                  \
      return std::make_unique<Encoding<uint32_t>>(          \
          memoryPool, data, stringBufferFactory);           \
    case DataType::Int64:                                   \
      return std::make_unique<Encoding<int64_t>>(           \
          memoryPool, data, stringBufferFactory);           \
    case DataType::Uint64:                                  \
      return std::make_unique<Encoding<uint64_t>>(          \
          memoryPool, data, stringBufferFactory);           \
    case DataType::Float:                                   \
      return std::make_unique<Encoding<float>>(             \
          memoryPool, data, stringBufferFactory);           \
    case DataType::Double:                                  \
      return std::make_unique<Encoding<double>>(            \
          memoryPool, data, stringBufferFactory);           \
    default:                                                \
      NIMBLE_UNREACHABLE(                                   \
          "Trying to deserialize a non-numeric stream for " \
          "a numeric data type {}.",                        \
          toString(dataType));                              \
  }

#define RETURN_ENCODING_BY_INTEGRAL_TYPE(Encoding, dataType)               \
  switch (dataType) {                                                      \
    case DataType::Int8:                                                   \
      return std::make_unique<Encoding<int8_t>>(                           \
          memoryPool, data, stringBufferFactory);                          \
    case DataType::Uint8:                                                  \
      return std::make_unique<Encoding<uint8_t>>(                          \
          memoryPool, data, stringBufferFactory);                          \
    case DataType::Int16:                                                  \
      return std::make_unique<Encoding<int16_t>>(                          \
          memoryPool, data, stringBufferFactory);                          \
    case DataType::Uint16:                                                 \
      return std::make_unique<Encoding<uint16_t>>(                         \
          memoryPool, data, stringBufferFactory);                          \
    case DataType::Int32:                                                  \
      return std::make_unique<Encoding<int32_t>>(                          \
          memoryPool, data, stringBufferFactory);                          \
    case DataType::Uint32:                                                 \
      return std::make_unique<Encoding<uint32_t>>(                         \
          memoryPool, data, stringBufferFactory);                          \
    case DataType::Int64:                                                  \
      return std::make_unique<Encoding<int64_t>>(                          \
          memoryPool, data, stringBufferFactory);                          \
    case DataType::Uint64:                                                 \
      return std::make_unique<Encoding<uint64_t>>(                         \
          memoryPool, data, stringBufferFactory);                          \
    case DataType::Undefined:                                              \
    case DataType::Float:                                                  \
    case DataType::Double:                                                 \
    case DataType::Bool:                                                   \
    case DataType::String:                                                 \
    default:                                                               \
      NIMBLE_INCOMPATIBLE_ENCODING(                                        \
          "{} only supports integer types, got {}.", #Encoding, dataType); \
  }

  switch (encodingType) {
    case EncodingType::Trivial: {
      RETURN_ENCODING_BY_LEAF_TYPE(TrivialEncoding, dataType);
    }
    case EncodingType::RLE: {
      RETURN_ENCODING_BY_LEAF_TYPE(RLEEncoding, dataType);
    }
    case EncodingType::Dictionary: {
      RETURN_ENCODING_BY_NON_BOOL_TYPE(DictionaryEncoding, dataType);
    }
    case EncodingType::FixedBitWidth: {
      RETURN_ENCODING_BY_NUMERIC_TYPE(FixedBitWidthEncoding, dataType);
    }
    case EncodingType::Nullable: {
      RETURN_ENCODING_BY_LEAF_TYPE(NullableEncoding, dataType);
    }
    case EncodingType::SparseBool: {
      NIMBLE_CHECK_EQ(
          dataType,
          DataType::Bool,
          "Trying to deserialize a SparseBoolEncoding with a non-bool data type.");
      return std::make_unique<SparseBoolEncoding>(
          memoryPool, data, stringBufferFactory);
    }
    case EncodingType::Varint: {
      RETURN_ENCODING_BY_VARINT_TYPE(VarintEncoding, dataType);
    }
    case EncodingType::Constant: {
      RETURN_ENCODING_BY_LEAF_TYPE(ConstantEncoding, dataType);
    }
    case EncodingType::MainlyConstant: {
      RETURN_ENCODING_BY_NON_BOOL_TYPE(MainlyConstantEncoding, dataType);
    }
    case EncodingType::Delta: {
      RETURN_ENCODING_BY_NUMERIC_TYPE(DeltaEncoding, dataType);
    }
    case EncodingType::DeltaBlock: {
      RETURN_ENCODING_BY_INTEGRAL_TYPE(
          ::facebook::nimble::DeltaBlockEncoding, dataType);
    }
    // SubIntSplit integration commented out (disabled):
    /*
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
    case EncodingType::SubIntSplit: {
      RETURN_ENCODING_BY_VARINT_TYPE(SubIntSplitEncoding, dataType);
    }
#endif
    */
    case EncodingType::BlockBitPacking: {
      RETURN_ENCODING_BY_NUMERIC_TYPE(
          facebook::nimble::BlockBitPackingEncoding, dataType);
    }
    case EncodingType::Prefix: {
      NIMBLE_CHECK_EQ(
          dataType,
          DataType::String,
          "Trying to deserialize a PrefixEncoding with a non-string data type.");
      return std::make_unique<PrefixEncoding>(
          memoryPool, data, stringBufferFactory);
    }
    case EncodingType::PFOR: {
      RETURN_ENCODING_BY_NUMERIC_TYPE(PFOREncoding, dataType);
    }
    case EncodingType::SimdForBitpack: {
      RETURN_ENCODING_BY_NUMERIC_TYPE(
          ::facebook::nimble::SimdForBitpackEncoding, dataType);
    }
    case EncodingType::Huffman: {
      RETURN_ENCODING_BY_INTEGRAL_TYPE(
          ::facebook::nimble::HuffmanEncoding, dataType);
    }
    case EncodingType::ALP: {
      switch (dataType) {
        case DataType::Float:
          return std::make_unique<::facebook::nimble::ALPEncoding<float>>(
              memoryPool, data, stringBufferFactory);
        case DataType::Double:
          return std::make_unique<::facebook::nimble::ALPEncoding<double>>(
              memoryPool, data, stringBufferFactory);
        default:
          NIMBLE_INCOMPATIBLE_ENCODING(
              "ALP encoding only supports float and double data types, got {}.",
              dataType);
      }
    }
    case EncodingType::Fsst: {
      NIMBLE_CHECK_EQ(
          dataType,
          DataType::String,
          "Trying to deserialize a FsstEncoding with a non-string data type.");
      return std::make_unique<::facebook::nimble::FsstEncoding>(
          memoryPool, data, stringBufferFactory);
    }
    default: {
      NIMBLE_UNREACHABLE(
          "Trying to deserialize invalid EncodingType -- garbage input?");
    }
  }
}

std::unique_ptr<Encoding> EncodingFactory::create(
    velox::memory::MemoryPool& memoryPool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory,
    const Encoding::Options& options) const {
  return EncodingFactory{options}.create(memoryPool, data, stringBufferFactory);
}

template <typename T>
std::string_view EncodingFactory::encode(
    std::unique_ptr<EncodingSelectionPolicy<T>>&& selectorPolicy,
    std::span<const T> values,
    Buffer& buffer) {
  using physicalType = typename TypeTraits<T>::physicalType;
  auto physicalValues = toPhysicalSpan(values);
  auto statistics = Statistics<physicalType>::create(physicalValues);
  auto selectionResult =
      selectorPolicy->select(physicalValues, statistics, Encoding::Options{});
  EncodingSelection<physicalType> selection{
      std::move(selectionResult),
      std::move(statistics),
      std::move(selectorPolicy)};
  return EncodingFactory::encode<T>(
      std::move(selection), physicalValues, buffer);
}

template <typename T>
std::string_view EncodingFactory::encodeNullable(
    std::unique_ptr<EncodingSelectionPolicy<T>>&& selectorPolicy,
    std::span<const T> values,
    std::span<const bool> nulls,
    Buffer& buffer) {
  using physicalType = typename TypeTraits<T>::physicalType;
  auto physicalValues = toPhysicalSpan(values);
  auto statistics = Statistics<physicalType>::create(physicalValues);
  auto selectionResult = selectorPolicy->selectNullable(
      physicalValues, nulls, statistics, Encoding::Options{});
  EncodingSelection<physicalType> selection{
      std::move(selectionResult),
      std::move(statistics),
      std::move(selectorPolicy)};
  return EncodingFactory::encodeNullable<T>(
      std::move(selection), physicalValues, nulls, buffer);
}

template <typename T>
std::string_view EncodingFactory::encode(
    EncodingSelection<typename TypeTraits<T>::physicalType>&& selection,
    std::span<const typename TypeTraits<T>::physicalType> values,
    Buffer& buffer) {
  using physicalType = typename TypeTraits<T>::physicalType;
  auto castedValues = toPhysicalSpan(values);
  switch (selection.encodingType()) {
    case EncodingType::Constant: {
      return ConstantEncoding<T>::encode(selection, castedValues, buffer);
    }
    case EncodingType::Trivial: {
      return TrivialEncoding<T>::encode(selection, castedValues, buffer);
    }
    case EncodingType::RLE: {
      return RLEEncoding<T>::encode(selection, castedValues, buffer);
    }
    case EncodingType::Dictionary: {
      NIMBLE_DCHECK(
          (!std::is_same<T, bool>::value && !castedValues.empty()),
          "Invalid DictionaryEncoding selection.");
      return DictionaryEncoding<T>::encode(selection, castedValues, buffer);
    }
    case EncodingType::FixedBitWidth: {
      if constexpr (isNumericType<physicalType>()) {
        return FixedBitWidthEncoding<T>::encode(
            selection, castedValues, buffer);
      } else {
        NIMBLE_INCOMPATIBLE_ENCODING(
            "FixedBitWidth encoding should not be selected for non-numeric data types.");
      }
    }
    case EncodingType::Varint: {
      // TODO: we can support floating point types, but currently Statistics
      // doesn't calculate buckets for floating point types. We should convert
      // floating point types to their physical type, and then Statistics and
      // Varint encoding will just work.
      if constexpr (
          isNumericType<physicalType>() &&
          (sizeof(physicalType) == 4 || sizeof(T) == 8)) {
        return VarintEncoding<T>::encode(selection, castedValues, buffer);
      } else {
        NIMBLE_INCOMPATIBLE_ENCODING(
            "Varint encoding can only be selected for large numeric data types.");
      }
    }
    case EncodingType::MainlyConstant: {
      if constexpr (std::is_same<T, bool>::value) {
        NIMBLE_INCOMPATIBLE_ENCODING(
            "MainlyConstant encoding should not be selected for bool data types.");
      } else {
        return MainlyConstantEncoding<T>::encode(
            selection, castedValues, buffer);
      }
    }
    case EncodingType::SparseBool: {
      if constexpr (!std::is_same<T, bool>::value) {
        NIMBLE_INCOMPATIBLE_ENCODING(
            "SparseBool encoding should not be selected for non-bool data types.");
      } else {
        return SparseBoolEncoding::encode(selection, castedValues, buffer);
      }
    }
    default: {
      NIMBLE_UNSUPPORTED(
          "Encoding {} is not supported.", toString(selection.encodingType()));
    }
  }
}

template <typename T>
std::string_view EncodingFactory::encodeNullable(
    EncodingSelection<typename TypeTraits<T>::physicalType>&& selection,
    std::span<const typename TypeTraits<T>::physicalType> values,
    std::span<const bool> nulls,
    Buffer& buffer) {
  auto physicalValues = toPhysicalSpan(values);
  switch (selection.encodingType()) {
    case EncodingType::Nullable: {
      return NullableEncoding<T>::encodeNullable(
          selection, physicalValues, nulls, buffer);
    }
    default: {
      NIMBLE_UNSUPPORTED(
          "Encoding {} is not supported for nullable data.",
          toString(selection.encodingType()));
    }
  }
}

#define DEFINE_TEMPLATES(type)                                                 \
  template std::string_view EncodingFactory::encode<type>(                     \
      std::unique_ptr<EncodingSelectionPolicy<type>> && selectorPolicy,        \
      std::span<const type> values,                                            \
      Buffer & buffer);                                                        \
  template std::string_view EncodingFactory::encodeNullable<type>(             \
      std::unique_ptr<EncodingSelectionPolicy<type>> && selectorPolicy,        \
      std::span<const type> values,                                            \
      std::span<const bool> nulls,                                             \
      Buffer & buffer);                                                        \
  template std::string_view EncodingFactory::encode<type>(                     \
      EncodingSelection<typename TypeTraits<type>::physicalType> && selection, \
      std::span<const typename TypeTraits<type>::physicalType> values,         \
      Buffer & buffer);                                                        \
  template std::string_view EncodingFactory::encodeNullable<type>(             \
      EncodingSelection<typename TypeTraits<type>::physicalType> && selection, \
      std::span<const typename TypeTraits<type>::physicalType> values,         \
      std::span<const bool> nulls,                                             \
      Buffer & buffer);

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

} // namespace facebook::nimble::legacy
