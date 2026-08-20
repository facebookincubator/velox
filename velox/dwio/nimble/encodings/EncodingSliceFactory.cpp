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
#include "velox/dwio/nimble/encodings/EncodingSliceFactory.h"

#include <algorithm>
#include <span>
#include <vector>

#include "velox/dwio/nimble/common/DataTypeDispatch.h"
#include "velox/dwio/nimble/common/NimbleException.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/ForEncoding.h"
#include "velox/dwio/nimble/encodings/FsstEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/PFOREncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"
#include "velox/dwio/nimble/encodings/SparseBoolEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingTypeDispatch.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

namespace facebook::nimble {
namespace {

template <typename T>
std::string_view encodeValuesWithLayout(
    EncodingLayout encodingLayout,
    std::span<const T> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  auto policy = std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
      std::move(encodingLayout),
      CompressionOptions{},
      [](DataType dataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
        NIMBLE_FAIL(
            "Captured encoding layout is missing nested child for {}.",
            dataType);
      });
  return EncodingFactory::encode<T>(std::move(policy), values, buffer, options);
}

template <typename T>
std::string_view sliceByMaterializing(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  using physicalType = typename TypeTraits<T>::physicalType;
  auto* pool = &buffer.getMemoryPool();
  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  Vector<physicalType> physicalValues{pool, length};

  auto encoding = EncodingFactory{options}.create(
      *pool, encoded, [&scopedBuffer](uint32_t size) -> void* {
        return scopedBuffer.get().reserve(size);
      });
  encoding->skip(offset);
  encoding->materialize(length, physicalValues.data());

  return encodeValuesWithLayout<T>(
      EncodingLayoutCapture::capture(encoded, options),
      {reinterpret_cast<const T*>(physicalValues.data()),
       physicalValues.size()},
      buffer,
      options);
}

std::string_view sliceByMaterializing(
    std::string_view encoded,
    EncodingType encodingType,
    DataType dataType,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_CHECK_NE(
      encodingType,
      EncodingType::Nullable,
      "Slicing nullable {} encoding by materializing is not supported.",
      encodingType);
  NIMBLE_RETURN_BY_DATA_TYPE(
      dataType,
      T,
      sliceByMaterializing<T>(encoded, offset, length, buffer, options));
}

std::string_view sliceNullable(
    std::string_view encoded,
    DataType dataType,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_RETURN_BY_DATA_TYPE(
      dataType,
      T,
      NullableEncoding<T>::slice(encoded, offset, length, buffer, options));
}

std::string_view sliceMainlyConstant(
    std::string_view encoded,
    DataType dataType,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_RETURN_BY_NON_BOOL_DATA_TYPE(
      dataType,
      T,
      MainlyConstantEncoding<T>::slice(
          encoded, offset, length, buffer, options));
}

std::string_view sliceTrivial(
    std::string_view encoded,
    DataType dataType,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_RETURN_BY_DATA_TYPE(
      dataType,
      T,
      TrivialEncoding<T>::slice(encoded, offset, length, buffer, options));
}

std::string_view sliceRLE(
    std::string_view encoded,
    DataType dataType,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_RETURN_BY_DATA_TYPE(
      dataType,
      T,
      RLEEncoding<T>::slice(encoded, offset, length, buffer, options));
}

std::string_view sliceConstant(
    std::string_view encoded,
    DataType dataType,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_RETURN_BY_DATA_TYPE(
      dataType,
      T,
      ConstantEncoding<T>::slice(encoded, offset, length, buffer, options));
}

std::string_view sliceDictionary(
    std::string_view encoded,
    DataType dataType,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_RETURN_BY_NON_BOOL_DATA_TYPE(
      dataType,
      T,
      DictionaryEncoding<T>::slice(encoded, offset, length, buffer, options));
}

std::string_view sliceSharedDictionary(
    std::string_view encoded,
    DataType dataType,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_RETURN_BY_INTEGER_DATA_TYPE(
      dataType,
      T,
      SharedDictionaryEncoding<T>::slice(
          encoded, offset, length, buffer, options));
}

std::string_view sliceFixedBitWidth(
    std::string_view encoded,
    DataType dataType,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_RETURN_BY_NUMERIC_DATA_TYPE(
      dataType,
      T,
      FixedBitWidthEncoding<T>::slice(
          encoded, offset, length, buffer, options));
}

std::string_view sliceBlockBitPacking(
    std::string_view encoded,
    DataType dataType,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_RETURN_BY_NUMERIC_DATA_TYPE(
      dataType,
      T,
      BlockBitPackingEncoding<T>::slice(
          encoded, offset, length, buffer, options));
}

std::string_view slicePFOR(
    std::string_view encoded,
    DataType dataType,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_RETURN_BY_NUMERIC_DATA_TYPE(
      dataType,
      T,
      PFOREncoding<T>::slice(encoded, offset, length, buffer, options));
}

std::string_view sliceSimdForBitpack(
    std::string_view encoded,
    DataType dataType,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_RETURN_BY_NUMERIC_DATA_TYPE(
      dataType,
      T,
      SimdForBitpackEncoding<T>::slice(
          encoded, offset, length, buffer, options));
}

std::string_view sliceFOR(
    std::string_view encoded,
    DataType dataType,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_RETURN_BY_INTEGER_DATA_TYPE(
      dataType,
      T,
      ForEncoding<T>::slice(encoded, offset, length, buffer, options));
}

std::string_view sliceFsst(
    std::string_view encoded,
    DataType dataType,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_CHECK_EQ(
      dataType,
      DataType::String,
      "Trying to slice FsstEncoding with a non-string data type.");
  return FsstEncoding::slice(encoded, offset, length, buffer, options);
}

std::string_view sliceALP(
    std::string_view encoded,
    DataType dataType,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_RETURN_BY_FLOATING_POINT_DATA_TYPE_OR(
      dataType,
      T,
      ALPEncoding<T>::slice(encoded, offset, length, buffer, options),
      NIMBLE_INCOMPATIBLE_ENCODING(
          "ALP encoding only supports float and double data types, got {}.",
          dataType));
}

} // namespace

std::string_view EncodingSliceFactory::slice(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  const auto encodingType = EncodingPrefix::encodingType(encoded);
  const auto rowCount =
      EncodingPrefix::readRowCount(encoded, options.useVarintRowCount);
  NIMBLE_CHECK_LE(offset, rowCount);
  NIMBLE_CHECK_LE(length, rowCount - offset);
  NIMBLE_CHECK_GT(length, 0, "Cannot slice zero rows.");
  if (offset == 0 && length == rowCount &&
      encodingType != EncodingType::SharedDictionary) {
    return buffer.writeString(encoded);
  }

  const auto dataType = EncodingPrefix::dataType(encoded);
  switch (encodingType) {
    case EncodingType::Constant:
      return sliceConstant(encoded, dataType, offset, length, buffer, options);
    case EncodingType::Trivial:
      return sliceTrivial(encoded, dataType, offset, length, buffer, options);
    case EncodingType::RLE:
      return sliceRLE(encoded, dataType, offset, length, buffer, options);
    case EncodingType::Dictionary:
      return sliceDictionary(
          encoded, dataType, offset, length, buffer, options);
    case EncodingType::SharedDictionary:
      return sliceSharedDictionary(
          encoded, dataType, offset, length, buffer, options);
    case EncodingType::FixedBitWidth:
      return sliceFixedBitWidth(
          encoded, dataType, offset, length, buffer, options);
    case EncodingType::BlockBitPacking:
      return sliceBlockBitPacking(
          encoded, dataType, offset, length, buffer, options);
    case EncodingType::PFOR:
      return slicePFOR(encoded, dataType, offset, length, buffer, options);
    case EncodingType::SimdForBitpack:
      return sliceSimdForBitpack(
          encoded, dataType, offset, length, buffer, options);
    case EncodingType::FOR:
      return sliceFOR(encoded, dataType, offset, length, buffer, options);
    case EncodingType::Fsst:
      return sliceFsst(encoded, dataType, offset, length, buffer, options);
    case EncodingType::ALP:
      return sliceALP(encoded, dataType, offset, length, buffer, options);
    case EncodingType::Nullable:
      return sliceNullable(encoded, dataType, offset, length, buffer, options);
    case EncodingType::SparseBool:
      return SparseBoolEncoding::slice(
          encoded, offset, length, buffer, options);
    case EncodingType::MainlyConstant:
      return sliceMainlyConstant(
          encoded, dataType, offset, length, buffer, options);
    default:
      return sliceByMaterializing(
          encoded, encodingType, dataType, offset, length, buffer, options);
  }
}

} // namespace facebook::nimble
