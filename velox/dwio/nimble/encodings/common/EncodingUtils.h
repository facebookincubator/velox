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

#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaBlockEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
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

namespace facebook::nimble {

template <typename DecoderVisitor>
void callReadWithVisitor(
    Encoding& encoding,
    DecoderVisitor& visitor,
    ReadWithVisitorParams& params);

namespace detail {

inline int dataTypeSize(DataType type) {
  switch (type) {
    case DataType::Int8:
    case DataType::Uint8:
    case DataType::Bool:
      return 1;
    case DataType::Int16:
    case DataType::Uint16:
      return 2;
    case DataType::Int32:
    case DataType::Uint32:
    case DataType::Float:
      return 4;
    case DataType::Int64:
    case DataType::Uint64:
    case DataType::Double:
      return 8;
    default:
      NIMBLE_UNSUPPORTED("{}", type);
  }
}

template <typename F>
auto encodingTypeDispatchString(Encoding& encoding, F f) {
  NIMBLE_CHECK_EQ(
      encoding.dataType(), DataType::String, "{}", encoding.dataType());
  switch (encoding.encodingType()) {
    case EncodingType::Trivial:
      return f(static_cast<TrivialEncoding<std::string_view>&>(encoding));
    case EncodingType::RLE:
      return f(static_cast<RLEEncoding<std::string_view>&>(encoding));
    case EncodingType::Dictionary:
      return f(static_cast<DictionaryEncoding<std::string_view>&>(encoding));
    case EncodingType::Nullable:
      return f(static_cast<NullableEncoding<std::string_view>&>(encoding));
    case EncodingType::Constant:
      return f(static_cast<ConstantEncoding<std::string_view>&>(encoding));
    case EncodingType::MainlyConstant:
      return f(
          static_cast<MainlyConstantEncoding<std::string_view>&>(encoding));
    case EncodingType::Prefix:
      return f(static_cast<PrefixEncoding&>(encoding));
    case EncodingType::Fsst:
      return f(static_cast<FsstEncoding&>(encoding));
    default:
      NIMBLE_UNSUPPORTED("{}", encoding.encodingType());
  }
}

template <typename T, typename F>
auto encodingTypeDispatchNonString(Encoding& encoding, F&& f) {
  NIMBLE_CHECK_EQ(
      dataTypeSize(encoding.dataType()), sizeof(T), "{}", encoding.dataType());
  switch (encoding.encodingType()) {
    case EncodingType::Trivial:
      return f(static_cast<TrivialEncoding<T>&>(encoding));
    case EncodingType::RLE:
      return f(static_cast<RLEEncoding<T>&>(encoding));
    case EncodingType::Dictionary:
      return f(static_cast<DictionaryEncoding<T>&>(encoding));
    case EncodingType::SharedDictionary:
      if constexpr (isIntegralType<T>() && !std::is_same_v<T, bool>) {
        return f(static_cast<SharedDictionaryEncoding<T>&>(encoding));
      }
      NIMBLE_UNREACHABLE(
          "Shared dictionary only supports integer types, got {}.",
          encoding.dataType());
    case EncodingType::FixedBitWidth:
      return f(static_cast<FixedBitWidthEncoding<T>&>(encoding));
    case EncodingType::Nullable:
      return f(static_cast<NullableEncoding<T>&>(encoding));
    case EncodingType::SparseBool:
      if constexpr (std::is_same_v<T, bool>) {
        return f(static_cast<SparseBoolEncoding&>(encoding));
      }
      NIMBLE_UNREACHABLE(
          "SparseBool encoding only supports bool data types, got {}.",
          encoding.dataType());
    case EncodingType::Varint:
      if constexpr (folly::IsOneOf<
                        T,
                        int32_t,
                        uint32_t,
                        int64_t,
                        uint64_t,
                        float,
                        double>::value) {
        return f(static_cast<VarintEncoding<T>&>(encoding));
      }
      NIMBLE_UNREACHABLE(
          "Varint encoding only supports 32- and 64-bit numeric data types, got {}.",
          encoding.dataType());
    case EncodingType::Constant:
      return f(static_cast<ConstantEncoding<T>&>(encoding));
    case EncodingType::MainlyConstant:
      return f(static_cast<MainlyConstantEncoding<T>&>(encoding));
    case EncodingType::Delta:
      return f(static_cast<DeltaEncoding<T>&>(encoding));
    case EncodingType::DeltaBlock:
      if constexpr (isIntegralType<T>()) {
        return f(static_cast<DeltaBlockEncoding<T>&>(encoding));
      }
      NIMBLE_UNREACHABLE(
          "DeltaBlock encoding only supports integral data types, got {}.",
          encoding.dataType());
    case EncodingType::ALP:
      if constexpr (isFloatingPointType<T>()) {
        return f(static_cast<ALPEncoding<T>&>(encoding));
      }
      NIMBLE_UNSUPPORTED(
          "ALP encoding only supports float and double data types, got {}.",
          encoding.dataType());
    case EncodingType::BlockBitPacking:
      return f(static_cast<BlockBitPackingEncoding<T>&>(encoding));
    case EncodingType::PFOR:
      if constexpr (isIntegralType<T>()) {
        return f(static_cast<PFOREncoding<T>&>(encoding));
      }
      NIMBLE_UNREACHABLE(
          "PFOR encoding only supports integral data types, got {}.",
          encoding.dataType());
    case EncodingType::SimdForBitpack:
      if constexpr (isIntegralType<T>()) {
        return f(static_cast<SimdForBitpackEncoding<T>&>(encoding));
      }
      NIMBLE_UNREACHABLE(
          "SimdForBitpack encoding only supports integral data types, got {}.",
          encoding.dataType());
    case EncodingType::Huffman:
      if constexpr (isIntegralType<T>()) {
        return f(static_cast<HuffmanEncoding<T>&>(encoding));
      }
      NIMBLE_UNREACHABLE(
          "Huffman encoding only supports integral data types, got {}.",
          encoding.dataType());
    default:
      NIMBLE_UNSUPPORTED("{}", encoding.encodingType());
  }
}

} // namespace detail

template <typename V>
void callReadWithVisitor(
    Encoding& encoding,
    V& visitor,
    ReadWithVisitorParams& params) {
  using T = typename V::DataType;
  if constexpr (std::is_same_v<T, std::string_view>) {
    detail::encodingTypeDispatchString(encoding, [&](auto& typedEncoding) {
      typedEncoding.readWithVisitor(visitor, params);
    });
  } else if constexpr (std::is_same_v<T, velox::int128_t>) {
    NIMBLE_UNSUPPORTED("Int128 is not supported in Nimble");
  } else if constexpr (std::is_same_v<T, int8_t>) {
    if (encoding.dataType() == DataType::Bool) {
      detail::encodingTypeDispatchNonString<bool>(
          encoding, [&](auto& typedEncoding) {
            typedEncoding.readWithVisitor(visitor, params);
          });
    } else {
      detail::encodingTypeDispatchNonString<int8_t>(
          encoding, [&](auto& typedEncoding) {
            typedEncoding.readWithVisitor(visitor, params);
          });
    }
  } else {
    detail::encodingTypeDispatchNonString<T>(
        encoding, [&](auto& typedEncoding) {
          typedEncoding.readWithVisitor(visitor, params);
        });
  }
}

/// Encoding trait for non-legacy encodings. Dispatches to the standard
/// callReadWithVisitor which casts to non-legacy concrete encoding types.
struct DefaultEncodingTrait {
  template <typename V>
  static void callReadWithVisitor(
      Encoding& encoding,
      V& visitor,
      ReadWithVisitorParams& params) {
    nimble::callReadWithVisitor(encoding, visitor, params);
  }
};

/// Dispatches readIndicesWithVisitor to the correct encoding type.
/// Supports DictionaryEncoding and dictionary-enabled NullableEncoding,
/// MainlyConstantEncoding, RLEEncoding, and ConstantEncoding wrappers.
/// Non-legacy encodings only.
///
/// T is the encoding's logical value type, which cannot be deduced from the
/// visitor: the visitor reads dictionary indices, so its DataType is the index
/// type rather than the alphabet's value type.
template <typename T, typename V>
void callReadIndicesWithVisitor(
    Encoding& encoding,
    V& visitor,
    ReadWithVisitorParams& params) {
  static_assert(
      !std::is_same_v<T, std::string>,
      "String encodings use std::string_view as their logical value type");
  NIMBLE_CHECK_EQ(
      encoding.dataType(),
      TypeTraits<T>::dataType,
      "Unexpected encoding data type: {}",
      encoding.dataType());
  switch (encoding.encodingType()) {
    case EncodingType::Dictionary: {
      static_cast<DictionaryEncoding<T>&>(encoding).readIndicesWithVisitor(
          visitor, params);
      return;
    }
    case EncodingType::Nullable: {
      static_cast<NullableEncoding<T>&>(encoding).readIndicesWithVisitor(
          visitor, params);
      return;
    }
    case EncodingType::MainlyConstant: {
      static_cast<MainlyConstantEncoding<T>&>(encoding).readIndicesWithVisitor(
          visitor, params);
      return;
    }
    case EncodingType::RLE: {
      static_cast<RLEEncoding<T>&>(encoding).readIndicesWithVisitor(
          visitor, params);
      return;
    }
    case EncodingType::Constant: {
      static_cast<ConstantEncoding<T>&>(encoding).readIndicesWithVisitor(
          visitor, params);
      return;
    }
    default:
      NIMBLE_UNREACHABLE(
          "Dictionary indices dispatch on unsupported encoding: {}",
          encoding.encodingType());
  }
}

/// Builds the dictionary alphabet from any dictionary-enabled encoding.
/// Uses dictionaryEntries() which returns a contiguous array for all
/// encoding types (Dict, Nullable→Dict, MC→Dict).
template <typename T>
inline std::vector<T> buildEncodingDictionaryAlphabet(
    const Encoding* encoding) {
  NIMBLE_CHECK(
      encoding->dictionaryEnabled(),
      "buildEncodingDictionaryAlphabet requires a dictionary-enabled encoding");
  if (encoding->encodingType() == EncodingType::Nullable) {
    return buildEncodingDictionaryAlphabet<T>(
        static_cast<const NullableEncoding<T>*>(encoding)->nonNulls());
  }
  const auto size = encoding->dictionarySize();
  const auto* entries = static_cast<const T*>(encoding->dictionaryEntries());
  return {entries, entries + size};
}

} // namespace facebook::nimble
