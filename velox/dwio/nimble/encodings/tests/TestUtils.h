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

#include <algorithm>
#include <optional>

#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaBlockEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/ForEncoding.h"
#include "velox/dwio/nimble/encodings/FrequencyPartitionEncoding.h"
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/PFOREncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"
#include "velox/dwio/nimble/encodings/SparseBoolEncoding.h"
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#endif
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/VarintEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

namespace facebook::nimble::test {

template <typename Encoding>
struct EncodingTypeTraits;

template <typename T>
struct EncodingTypeTraits<nimble::ALPEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::ALP;
};

template <typename T>
struct EncodingTypeTraits<nimble::BlockBitPackingEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::BlockBitPacking;
};

template <typename T>
struct EncodingTypeTraits<nimble::ConstantEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::Constant;
};

template <typename T>
struct EncodingTypeTraits<nimble::DeltaEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::Delta;
};

template <typename T>
struct EncodingTypeTraits<nimble::DeltaBlockEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::DeltaBlock;
};

template <typename T>
struct EncodingTypeTraits<nimble::DictionaryEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::Dictionary;
};

template <typename T>
struct EncodingTypeTraits<nimble::FixedBitWidthEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::FixedBitWidth;
};

template <typename T>
struct EncodingTypeTraits<nimble::FrequencyPartitionEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::FrequencyPartition;
};

template <typename T>
struct EncodingTypeTraits<nimble::ForEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::FOR;
};

template <typename T>
struct EncodingTypeTraits<nimble::HuffmanEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::Huffman;
};

template <typename T>
struct EncodingTypeTraits<nimble::MainlyConstantEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::MainlyConstant;
};

template <typename T>
struct EncodingTypeTraits<nimble::RLEEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::RLE;
};

template <>
struct EncodingTypeTraits<nimble::SparseBoolEncoding> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::SparseBool;
};

template <typename T>
struct EncodingTypeTraits<nimble::TrivialEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::Trivial;
};

template <typename T>
struct EncodingTypeTraits<nimble::VarintEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::Varint;
};

template <typename T>
struct EncodingTypeTraits<nimble::NullableEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::Nullable;
};

template <typename T>
struct EncodingTypeTraits<nimble::PFOREncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::PFOR;
};

template <typename T>
struct EncodingTypeTraits<nimble::SimdForBitpackEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::SimdForBitpack;
};

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
template <typename T>
struct EncodingTypeTraits<nimble::SubIntSplitEncoding<T>> {
  static constexpr inline nimble::EncodingType encodingType =
      nimble::EncodingType::SubIntSplit;
};
#endif

template <typename E>
class Encoder {
  using T = typename E::cppDataType;

 private:
  class TestCompressPolicy : public nimble::CompressionPolicy {
   public:
    explicit TestCompressPolicy(CompressionType compressionType)
        : compressionType_{compressionType} {}

    nimble::CompressionConfig config() const override {
      if (compressionType_ == CompressionType::Uncompressed) {
        return {.compressionType = CompressionType::Uncompressed};
      }

      if (compressionType_ == CompressionType::Zstd) {
        nimble::CompressionConfig information{
            .compressionType = nimble::CompressionType::Zstd};
        information.parameters.zstd.compressionLevel = 3;
        return information;
      }

#ifdef DISABLE_META_INTERNAL_COMPRESSOR
      // The Meta internal compressor has no OSS implementation and is not put
      // in the registry by Compression.cpp, so returning it here would throw
      // from getCompressor(). Redirect to Zstd; tests that assert on the
      // resulting compression type already account for this mapping.
      nimble::CompressionConfig information{
          .compressionType = nimble::CompressionType::Zstd};
      information.parameters.zstd.compressionLevel = 3;
      return information;
#else
      nimble::CompressionConfig information{
          .compressionType = nimble::CompressionType::MetaInternal};
      information.parameters.metaInternal.compressionLevel = 9;
      information.parameters.metaInternal.decompressionLevel = 2;
      return information;
#endif
    }

    virtual bool shouldAccept(
        nimble::CompressionType /* compressionType */,
        uint64_t /* uncompressedSize */,
        uint64_t /* compressedSize */) const override {
      return true;
    }

   private:
    CompressionType compressionType_;
  };

  template <typename TInner>
  class TestTrivialEncodingSelectionPolicy
      : public nimble::EncodingSelectionPolicy<TInner> {
    using physicalType = typename nimble::TypeTraits<TInner>::physicalType;

   public:
    explicit TestTrivialEncodingSelectionPolicy(
        CompressionType compressionType,
        bool realNestedSelection = false)
        : compressionType_{compressionType},
          realNestedSelection_{realNestedSelection} {}

    nimble::EncodingSelectionResult select(
        std::span<const physicalType> /* values */,
        const nimble::Statistics<physicalType>& /* statistics */,
        const nimble::Encoding::Options& /* options */) override {
      return {
          .encodingType = nimble::EncodingType::Trivial,
          .compressionPolicyFactory = [this]() {
            return std::make_unique<TestCompressPolicy>(compressionType_);
          }};
    }

    EncodingSelectionResult selectNullable(
        std::span<const physicalType> /* values */,
        std::span<const bool> /* nulls */,
        const Statistics<physicalType>& /* statistics */,
        const Encoding::Options& /* options */) override {
      return {
          .encodingType = EncodingType::Nullable,
      };
    }

    std::unique_ptr<nimble::EncodingSelectionPolicyBase> createImpl(
        nimble::EncodingType /* encodingType */,
        nimble::NestedEncodingIdentifier /* identifier */,
        nimble::DataType type) override {
      // When realNestedSelection_ is set, nested (sub-stream) encodings are
      // chosen by the normal cost-based factory instead of being forced to
      // Trivial -- e.g. so SubIntSplit exercises its diverse per-section
      // encoders. SubIntSplit is removed from the candidates to avoid infinite
      // recursion (it is also not registered in EncodingFactory dispatch).
      if (realNestedSelection_) {
        auto readFactors = nimble::ManualEncodingSelectionPolicyFactory::
            defaultEncodingReadFactors();
        readFactors.erase(
            std::remove_if(
                readFactors.begin(),
                readFactors.end(),
                [](const auto& factor) {
                  return factor.first == nimble::EncodingType::SubIntSplit;
                }),
            readFactors.end());
        return nimble::ManualEncodingSelectionPolicyFactory{
            std::move(readFactors), std::nullopt}
            .createPolicy(type);
      }
      UNIQUE_PTR_FACTORY(
          type,
          TestTrivialEncodingSelectionPolicy,
          compressionType_,
          realNestedSelection_);
    }

   private:
    CompressionType compressionType_;
    bool realNestedSelection_;
  };

 public:
  static constexpr EncodingType encodingType() {
    return EncodingTypeTraits<E>::encodingType;
  }

  static std::string_view encode(
      nimble::Buffer& buffer,
      const nimble::Vector<T>& values,
      CompressionType compressionType = CompressionType::Uncompressed,
      const nimble::Encoding::Options& options = {},
      bool realNestedSelection = false) {
    using physicalType = typename nimble::TypeTraits<T>::physicalType;

    auto physicalValues = std::span<const physicalType>(
        reinterpret_cast<const physicalType*>(values.data()), values.size());
    nimble::EncodingSelection<physicalType> selection{
        {.encodingType = EncodingTypeTraits<E>::encodingType,
         .compressionPolicyFactory =
             [compressionType]() {
               return std::make_unique<TestCompressPolicy>(compressionType);
             }},
        nimble::Statistics<physicalType>::create(physicalValues),
        std::make_unique<TestTrivialEncodingSelectionPolicy<T>>(
            compressionType, realNestedSelection)};

    return E::encode(selection, physicalValues, buffer, options);
  }

  static std::string_view encodeNullable(
      nimble::Buffer& buffer,
      const nimble::Vector<T>& values,
      const nimble::Vector<bool>& nulls,
      nimble::CompressionType compressionType =
          nimble::CompressionType::Uncompressed,
      const nimble::Encoding::Options& options = {}) {
    using physicalType = typename nimble::TypeTraits<T>::physicalType;

    auto physicalValues = std::span<const physicalType>(
        reinterpret_cast<const physicalType*>(values.data()), values.size());
    nimble::EncodingSelection<physicalType> selection{
        {.encodingType = EncodingTypeTraits<E>::encodingType,
         .compressionPolicyFactory =
             [compressionType]() {
               return std::make_unique<TestCompressPolicy>(compressionType);
             }},
        nimble::Statistics<physicalType>::create(physicalValues),
        std::make_unique<TestTrivialEncodingSelectionPolicy<T>>(
            compressionType)};

    return NullableEncoding<typename E::cppDataType>::encodeNullable(
        selection, physicalValues, nulls, buffer, options);
  }

  static std::unique_ptr<nimble::Encoding> createEncoding(
      nimble::Buffer& buffer,
      const nimble::Vector<T>& values,
      const std::function<void*(uint32_t)>& stringBufferFactory,
      CompressionType compressionType = CompressionType::Uncompressed,
      const nimble::Encoding::Options& options = {}) {
    return std::make_unique<E>(
        buffer.getMemoryPool(),
        encode(buffer, values, compressionType, options),
        stringBufferFactory,
        options);
  }

  static std::unique_ptr<nimble::Encoding> createNullableEncoding(
      nimble::Buffer& buffer,
      const nimble::Vector<T>& values,
      const nimble::Vector<bool>& nulls,
      const std::function<void*(uint32_t)>& stringBufferFactory,
      CompressionType compressionType = CompressionType::Uncompressed,
      const nimble::Encoding::Options& options = {}) {
    return std::make_unique<E>(
        buffer.getMemoryPool(),
        encodeNullable(buffer, values, nulls, compressionType, options),
        stringBufferFactory,
        options);
  }
};

class TestUtils {
 public:
  static uint64_t getRawDataSize(
      velox::memory::MemoryPool& memoryPool,
      std::string_view encodingStr);
};
} // namespace facebook::nimble::test
