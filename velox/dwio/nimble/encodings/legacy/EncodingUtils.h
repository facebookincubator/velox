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
#include "velox/dwio/nimble/encodings/DeltaBlockEncoding.h"
#include "velox/dwio/nimble/encodings/FsstEncoding.h"
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/PFOREncoding.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"
// SubIntSplit integration (re-enabled for NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS;
// was commented out by #636):
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#endif
// FOR and FrequencyPartition integration (re-enabled for
// NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS; was commented out by #636):
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
#include "velox/dwio/nimble/encodings/ForEncoding.h"
#include "velox/dwio/nimble/encodings/FrequencyPartitionEncoding.h"
#endif
#include "velox/dwio/nimble/encodings/common/EncodingUtils.h"
#include "velox/dwio/nimble/encodings/legacy/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/DeltaEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/SparseBoolEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/legacy/VarintEncoding.h"

namespace facebook::nimble::legacy {

template <typename DecoderVisitor>
void callReadWithVisitor(
    Encoding& encoding,
    DecoderVisitor& visitor,
    ReadWithVisitorParams& params);

namespace detail {

template <typename F>
auto encodingTypeDispatchString(Encoding& encoding, F f) {
  NIMBLE_CHECK_EQ(
      encoding.dataType(),
      DataType::String,
      "{}",
      toString(encoding.dataType()));
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
    case EncodingType::Fsst:
      return f(static_cast<::facebook::nimble::FsstEncoding&>(encoding));
    default:
      NIMBLE_UNSUPPORTED(toString(encoding.encodingType()));
  }
}

template <typename T, typename F>
auto encodingTypeDispatchNonString(Encoding& encoding, F&& f) {
  NIMBLE_CHECK_EQ(
      ::facebook::nimble::detail::dataTypeSize(encoding.dataType()),
      sizeof(T),
      "{}",
      toString(encoding.dataType()));
  switch (encoding.encodingType()) {
    case EncodingType::Trivial:
      return f(static_cast<TrivialEncoding<T>&>(encoding));
    case EncodingType::RLE:
      return f(static_cast<RLEEncoding<T>&>(encoding));
    case EncodingType::Dictionary:
      return f(static_cast<DictionaryEncoding<T>&>(encoding));
    case EncodingType::FixedBitWidth:
      return f(static_cast<FixedBitWidthEncoding<T>&>(encoding));
    case EncodingType::Nullable:
      return f(static_cast<NullableEncoding<T>&>(encoding));
    case EncodingType::SparseBool:
      if constexpr (std::is_same_v<T, bool>) {
        return f(static_cast<SparseBoolEncoding&>(encoding));
      } else {
        NIMBLE_UNREACHABLE(toString(encoding.dataType()));
      }
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
      } else {
        NIMBLE_UNREACHABLE(toString(encoding.dataType()));
      }
    case EncodingType::Constant:
      return f(static_cast<ConstantEncoding<T>&>(encoding));
    case EncodingType::MainlyConstant:
      return f(static_cast<MainlyConstantEncoding<T>&>(encoding));
    case EncodingType::Delta:
      return f(static_cast<DeltaEncoding<T>&>(encoding));
    case EncodingType::BlockBitPacking:
      return f(
          static_cast<facebook::nimble::BlockBitPackingEncoding<T>&>(encoding));
    // New Encoding has no legacy-specialised class; legacy reads dispatch into
    // the modern Encoding classes
    case EncodingType::DeltaBlock:
      if constexpr (isIntegralType<T>()) {
        return f(
            static_cast<::facebook::nimble::DeltaBlockEncoding<T>&>(encoding));
      } else {
        NIMBLE_UNREACHABLE(
            "DeltaBlock encoding only supports integral data types, got {}.",
            encoding.dataType());
      }
    case EncodingType::PFOR:
      if constexpr (isIntegralType<T>()) {
        return f(static_cast<::facebook::nimble::PFOREncoding<T>&>(encoding));
      } else {
        NIMBLE_UNREACHABLE("{}", encoding.dataType());
      }
    case EncodingType::SimdForBitpack:
      if constexpr (isIntegralType<T>()) {
        return f(
            static_cast<::facebook::nimble::SimdForBitpackEncoding<T>&>(
                encoding));
      } else {
        NIMBLE_UNREACHABLE("{}", encoding.dataType());
      }
    case EncodingType::Huffman:
      if constexpr (isIntegralType<T>()) {
        return f(
            static_cast<::facebook::nimble::HuffmanEncoding<T>&>(encoding));
      } else {
        NIMBLE_UNREACHABLE("{}", encoding.dataType());
      }
    case EncodingType::ALP:
      if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        return f(static_cast<::facebook::nimble::ALPEncoding<T>&>(encoding));
      } else {
        NIMBLE_UNREACHABLE(
            "ALP encoding only supports float and double data types, got {}.",
            encoding.dataType());
      }
    // SubIntSplit integration (re-enabled for
    // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS; was commented out by #636):
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
    case EncodingType::SubIntSplit:
      if constexpr (isNumericType<T>() && (sizeof(T) == 4 || sizeof(T) == 8)) {
        return f(
            static_cast<::facebook::nimble::SubIntSplitEncoding<T>&>(encoding));
      } else {
        NIMBLE_UNREACHABLE("{}", encoding.dataType());
      }
#endif
    // FOR and FrequencyPartition integration (re-enabled for
    // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS; was commented out by #636):
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
    case EncodingType::FOR:
      if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
        return f(static_cast<::facebook::nimble::ForEncoding<T>&>(encoding));
      } else {
        NIMBLE_UNREACHABLE(toString(encoding.dataType()));
      }
    case EncodingType::FrequencyPartition:
      if constexpr (!std::is_same_v<T, bool>) {
        return f(static_cast<
                 ::facebook::nimble::FrequencyPartitionEncoding<T>&>(encoding));
      } else {
        NIMBLE_UNREACHABLE(toString(encoding.dataType()));
      }
#endif
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

} // namespace facebook::nimble::legacy
