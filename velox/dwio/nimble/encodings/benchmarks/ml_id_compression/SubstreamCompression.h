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

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <optional>
#include <span>
#include <string>

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/compression/CompressionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

// Encodes benchmark data with a caller-chosen compressor applied to the
// encoding's streams, including the sub-streams of a nested encoding such as
// SubIntSplit.
//
// This exists because test::Encoder cannot express the choice. Its
// TestCompressPolicy handles only Uncompressed and Zstd, and silently
// redirects everything else to Zstd level 3 under
// DISABLE_META_INTERNAL_COMPRESSOR, which would report OpenZL numbers that are
// really Zstd. Its policy classes are private, so they cannot be reused. Its
// nested path also hard-codes ManualEncodingSelectionPolicyFactory{...,
// std::nullopt}, which leaves sub-streams on the default compressor whatever
// the caller asked for.
//
// Compressor names are parsed by nimble::toCompressionType, so a codec added
// to nimble becomes available here with no change to this file.

namespace facebook::nimble::mlidc {

// Rejects compressors that have no OSS implementation, rather than quietly
// substituting a different one.
inline CompressionType parseCompressionType(const std::string& name) {
  auto type = nimble::toCompressionType(name);
#ifdef DISABLE_META_INTERNAL_COMPRESSOR
  if (type == CompressionType::MetaInternal) {
    throw std::runtime_error(
        "MetaInternal has no OSS implementation and is not in the compressor "
        "registry. Use Uncompressed, Zstd, Lz4 or OpenZL.");
  }
#endif
  return type;
}

// Builds the per-stream compression config for one chosen compressor. Unlike
// test::Encoder's policy this never substitutes a different compressor.
class BenchCompressPolicy : public nimble::CompressionPolicy {
 public:
  explicit BenchCompressPolicy(CompressionType compressionType)
      : compressionType_{compressionType} {}

  nimble::CompressionConfig config() const override {
    nimble::CompressionConfig config{.compressionType = compressionType_};
    // Defaults mirror CompressionOptions so a stream compressed here matches
    // one compressed through the production selection path.
    config.parameters.zstd.compressionLevel = 3;
    return config;
  }

  bool shouldAccept(
      nimble::CompressionType /* compressionType */,
      uint64_t /* uncompressedSize */,
      uint64_t /* compressedSize */) const override {
    return true;
  }

 private:
  CompressionType compressionType_;
};

// Returns the CompressionOptions handed to nested selection, so sub-streams
// use the same compressor as the top-level stream.
inline CompressionOptions compressionOptionsFor(CompressionType type) {
  CompressionOptions options;
  options.compressionType = type;
  // The default 0.98 accept ratio would silently leave a stream uncompressed
  // when the codec barely helps, which makes a compressor comparison read as a
  // codec difference. Always keep the requested compressor's output.
  options.compressionAcceptRatio = 1.0f;
  // Nimble skips compression for streams below these sizes. Benchmarks want
  // the requested compressor applied uniformly.
  options.zstdMinCompressionSize = 0;
  options.lz4MinCompressionSize = 0;
  options.openzlMinCompressionSize = 0;
  return options;
}

// Chooses encodings the same way test::Encoder does, so encoded layouts stay
// comparable, while routing the chosen compressor into nested selection.
template <typename TInner>
class BenchEncodingSelectionPolicy
    : public nimble::EncodingSelectionPolicy<TInner> {
  using physicalType = typename nimble::TypeTraits<TInner>::physicalType;

 public:
  BenchEncodingSelectionPolicy(
      CompressionType compressionType,
      bool realNestedSelection)
      : compressionType_{compressionType},
        realNestedSelection_{realNestedSelection} {}

  nimble::EncodingSelectionResult select(
      std::span<const physicalType> /* values */,
      const nimble::Statistics<physicalType>& /* statistics */,
      const nimble::Encoding::Options& /* options */) override {
    return {
        .encodingType = nimble::EncodingType::Trivial,
        .compressionPolicyFactory = [this]() {
          return std::make_unique<BenchCompressPolicy>(compressionType_);
        }};
  }

  nimble::EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const nimble::Statistics<physicalType>& /* statistics */,
      const nimble::Encoding::Options& /* options */) override {
    return {.encodingType = nimble::EncodingType::Nullable};
  }

  std::unique_ptr<nimble::EncodingSelectionPolicyBase> createImpl(
      nimble::EncodingType /* encodingType */,
      nimble::NestedEncodingIdentifier /* identifier */,
      nimble::DataType type) override {
    // Mirrors test::Encoder's nested path: when realNestedSelection is set the
    // sub-stream encodings are chosen by the normal cost-based factory rather
    // than forced to Trivial, so SubIntSplit exercises its per-section
    // encoders. SubIntSplit is removed from the candidates to avoid infinite
    // recursion. The one difference is that the chosen compressor is passed
    // down instead of std::nullopt.
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
          std::move(readFactors), compressionOptionsFor(compressionType_)}
          .createPolicy(type);
    }
    UNIQUE_PTR_FACTORY(
        type,
        BenchEncodingSelectionPolicy,
        compressionType_,
        realNestedSelection_);
  }

 private:
  CompressionType compressionType_;
  bool realNestedSelection_;
};

// Encodes values with the given encoding, applying compressionType to the
// encoding's own stream and to any sub-streams it creates.
template <typename E, typename T>
std::string_view encodeWithCompression(
    nimble::Buffer& buffer,
    const nimble::Vector<T>& values,
    CompressionType compressionType,
    const nimble::Encoding::Options& options,
    bool realNestedSelection) {
  using physicalType = typename nimble::TypeTraits<T>::physicalType;

  auto physicalValues = std::span<const physicalType>(
      reinterpret_cast<const physicalType*>(values.data()), values.size());

  nimble::EncodingSelection<physicalType> selection{
      {.encodingType = test::EncodingTypeTraits<E>::encodingType,
       .compressionPolicyFactory =
           [compressionType]() {
             return std::make_unique<BenchCompressPolicy>(compressionType);
           }},
      nimble::Statistics<physicalType>::create(physicalValues),
      std::make_unique<BenchEncodingSelectionPolicy<T>>(
          compressionType, realNestedSelection)};

  return E::encode(selection, physicalValues, buffer, options);
}

} // namespace facebook::nimble::mlidc

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
