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
#include <utility>
#include "velox/dwio/nimble/common/Constants.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingSizeEstimation.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

namespace facebook::nimble::legacy {

using EncodingSelectionPolicyCreator =
    std::function<std::unique_ptr<EncodingSelectionPolicyBase>(DataType)>;

#define COMMA ,
#define UNIQUE_PTR_FACTORY_EXTRA(data_type, class, extra_types, ...)        \
  switch (data_type) {                                                      \
    case facebook::nimble::DataType::Uint8: {                               \
      return std::make_unique<class<uint8_t extra_types>>(__VA_ARGS__);     \
    }                                                                       \
    case facebook::nimble::DataType::Int8: {                                \
      return std::make_unique<class<int8_t extra_types>>(__VA_ARGS__);      \
    }                                                                       \
    case facebook::nimble::DataType::Uint16: {                              \
      return std::make_unique<class<uint16_t extra_types>>(__VA_ARGS__);    \
    }                                                                       \
    case facebook::nimble::DataType::Int16: {                               \
      return std::make_unique<class<int16_t extra_types>>(__VA_ARGS__);     \
    }                                                                       \
    case facebook::nimble::DataType::Uint32: {                              \
      return std::make_unique<class<uint32_t extra_types>>(__VA_ARGS__);    \
    }                                                                       \
    case facebook::nimble::DataType::Int32: {                               \
      return std::make_unique<class<int32_t extra_types>>(__VA_ARGS__);     \
    }                                                                       \
    case facebook::nimble::DataType::Uint64: {                              \
      return std::make_unique<class<uint64_t extra_types>>(__VA_ARGS__);    \
    }                                                                       \
    case facebook::nimble::DataType::Int64: {                               \
      return std::make_unique<class<int64_t extra_types>>(__VA_ARGS__);     \
    }                                                                       \
    case facebook::nimble::DataType::Float: {                               \
      return std::make_unique<class<float extra_types>>(__VA_ARGS__);       \
    }                                                                       \
    case facebook::nimble::DataType::Double: {                              \
      return std::make_unique<class<double extra_types>>(__VA_ARGS__);      \
    }                                                                       \
    case facebook::nimble::DataType::Bool: {                                \
      return std::make_unique<class<bool extra_types>>(__VA_ARGS__);        \
    }                                                                       \
    case facebook::nimble::DataType::String: {                              \
      return std::make_unique<class<std::string_view extra_types>>(         \
          __VA_ARGS__);                                                     \
    }                                                                       \
    default: {                                                              \
      NIMBLE_UNREACHABLE("Unsupported data type {}.", toString(data_type)); \
    }                                                                       \
  }

#define UNIQUE_PTR_FACTORY(data_type, class, ...) \
  UNIQUE_PTR_FACTORY_EXTRA(data_type, class, , __VA_ARGS__)

struct CompressionOptions {
  /// Rejects compression when compressedSize exceeds uncompressedSize
  /// multiplied by this ratio. Currently only Zstrong/MetaInternal enforces
  /// this through CompressionPolicy::shouldAccept().
  /// TODO: Apply this consistently for Zstd and other compression types.
  float compressionAcceptRatio = 0.98f;
  uint64_t zstdMinCompressionSize = kZstdMinCompressionSize;
  uint32_t zstdCompressionLevel = 3;
  uint64_t internalMinCompressionSize = kMetaInternalMinCompressionSize;
  uint32_t internalCompressionLevel = 4;
  uint32_t internalDecompressionLevel = 2;
  bool useVariableBitWidthCompressor = false;
  MetaInternalCompressionKey metaInternalCompressionKey;
  std::vector<std::pair<EncodingType, float>> compressionAcceptRatioOverrides;
};

// This is the manual encoding selection implementation.
// The manual selection is using a manually crafted model to choose the most
// appropriate encoding based on the provided statistics.
template <typename T, bool FixedByteWidth = true>
class ManualEncodingSelectionPolicy : public EncodingSelectionPolicy<T> {
 public:
  using physicalType = typename TypeTraits<T>::physicalType;

  ManualEncodingSelectionPolicy(
      std::vector<std::pair<EncodingType, float>> encodingReadFactors,
      std::optional<CompressionOptions> compressionOptions,
      std::optional<NestedEncodingIdentifier> identifier)
      : candidateEncodingReadFactors_{std::move(encodingReadFactors)},
        compressionOptions_{std::move(compressionOptions)},
        identifier_{identifier} {}

  EncodingSelectionResult select(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& /* options */ = {}) override {
    if (values.empty()) {
      return {
          .encodingType = EncodingType::Trivial,
      };
    }

    float minCost = std::numeric_limits<float>::max();
    EncodingType selectedEncoding = EncodingType::Trivial;
    std::optional<uint64_t> selectedEstimatedSize;
    // Iterate on all candidate encodings, and pick the encoding with the
    // minimal cost.
    for (const auto& entry : candidateEncodingReadFactors_) {
      auto encodingType = entry.first;
      auto size =
          nimble::legacy::detail::EncodingSizeEstimation<T, FixedByteWidth>::
              estimateSize(encodingType, values.size(), statistics);
      if (!size.has_value()) {
        continue;
      }

      // We use read factor weights to raise/lower the favorability of each
      // encoding.
      const auto readFactor = entry.second;
      const auto cost = size.value() * readFactor;
      if (cost < minCost) {
        minCost = cost;
        selectedEncoding = encodingType;
        selectedEstimatedSize = size;
      }
    }

    // Currently, we always attempt to compress leaf data streams. The logic
    // behind this is that encoding selection optimizes the in-memory layout of
    // data, while compression provides extra saving for persistent storage (and
    // bandwidth).
    class AlwaysCompressPolicy : public CompressionPolicy {
     public:
      AlwaysCompressPolicy(
          CompressionOptions compressionOptions,
          EncodingType encodingType)
          : compressionOptions_{std::move(compressionOptions)},
            effectiveAcceptRatio_{getAcceptRatio(encodingType)} {}

      CompressionConfig config() const override {
#ifndef DISABLE_META_INTERNAL_COMPRESSOR
        CompressionConfig information{
            .compressionType = CompressionType::MetaInternal,
            .minCompressionSize =
                compressionOptions_.internalMinCompressionSize};
        information.parameters.metaInternal.compressionLevel =
            compressionOptions_.internalCompressionLevel;
        information.parameters.metaInternal.decompressionLevel =
            compressionOptions_.internalDecompressionLevel;
        information.parameters.metaInternal.useVariableBitWidthCompressor =
            compressionOptions_.useVariableBitWidthCompressor;
        information.parameters.metaInternal.compressionKey =
            compressionOptions_.metaInternalCompressionKey;
        return information;
#else
        CompressionConfig information{
            .compressionType = CompressionType::Zstd,
            .minCompressionSize = compressionOptions_.zstdMinCompressionSize};
        information.parameters.zstd.compressionLevel =
            compressionOptions_.zstdCompressionLevel;
        return information;
#endif
      }

      virtual bool shouldAccept(
          CompressionType /* compressionType */,
          uint64_t uncompressedSize,
          uint64_t compressedSize) const override {
        if (uncompressedSize * effectiveAcceptRatio_ < compressedSize) {
          return false;
        }

        return true;
      }

     private:
      float getAcceptRatio(EncodingType encodingType) const {
        for (const auto& [enc, ratio] :
             compressionOptions_.compressionAcceptRatioOverrides) {
          if (enc == encodingType) {
            return ratio;
          }
        }
        return compressionOptions_.compressionAcceptRatio;
      }

      const CompressionOptions compressionOptions_;
      const float effectiveAcceptRatio_;
    };

    if (!compressionOptions_.has_value()) {
      return {
          .encodingType = selectedEncoding,
          .estimatedSize = selectedEstimatedSize};
    }
    return {
        .encodingType = selectedEncoding,
        .estimatedSize = selectedEstimatedSize,
        .compressionPolicyFactory = [compressionOptions =
                                         compressionOptions_.value(),
                                     selectedEncoding]() {
          return std::make_unique<AlwaysCompressPolicy>(
              compressionOptions, selectedEncoding);
        }};
  }

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */ = {}) override {
    return {
        .encodingType = EncodingType::Nullable,
    };
  }

  const std::vector<std::pair<EncodingType, float>>&
  candidateEncodingReadFactors() const {
    return candidateEncodingReadFactors_;
  }

 protected:
  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType parentEncodingType,
      NestedEncodingIdentifier nestedEncodingIdentifier,
      DataType nestedDataType) override {
    // In each sub-level of the encoding selection, we exclude the encodings
    // selected in parent levels. Although this is not required (as hopefully,
    // the model will not pick a nested encoding of the same type as the
    // parent), it provides an additional safty net, making sure the encoding
    // selection will eventually converge, and also slightly speeds up nested
    // encoding selection.
    // TODO: validate the assumptions here compared to brute forcing, to see if
    // the same encoding is selected multiple times in the tree (for example,
    // should we allow trivial string lengths to be encoded using trivial
    // encoding?)
    std::vector<std::pair<EncodingType, float>> nestedEncodingReadFactors;
    nestedEncodingReadFactors.reserve(candidateEncodingReadFactors_.size() - 1);
    for (const auto& entry : candidateEncodingReadFactors_) {
      if (entry.first != parentEncodingType) {
        nestedEncodingReadFactors.emplace_back(entry);
      }
    }
    UNIQUE_PTR_FACTORY_EXTRA(
        nestedDataType,
        ManualEncodingSelectionPolicy,
        COMMA FixedByteWidth,
        std::move(nestedEncodingReadFactors),
        compressionOptions_,
        nestedEncodingIdentifier);
  }

 private:
  // Candidate encodings and their read-cost factors. Encoding selection uses
  // estimatedSize * readFactor as the cost, so a lower factor makes an
  // encoding more likely to be picked. Right now, these represent mostly the
  // CPU cost to decode values. Trivial is boosted with a lower factor because
  // it also benefits from applying compression.
  // See ManualEncodingSelectionPolicyFactory::defaultReadFactors for the
  // default.
  const std::vector<std::pair<EncodingType, float>>
      candidateEncodingReadFactors_;
  // When nullopt, compression is disabled for this policy and all nested
  // policies created from it.
  const std::optional<CompressionOptions> compressionOptions_;
  const std::optional<NestedEncodingIdentifier> identifier_;
};

class ManualEncodingSelectionPolicyFactory {
 public:
  ManualEncodingSelectionPolicyFactory(
      std::vector<std::pair<EncodingType, float>> encodingReadFactors =
          defaultReadFactors(),
      std::optional<CompressionOptions> compressionOptions =
          CompressionOptions{})
      : encodingReadFactors_{std::move(encodingReadFactors)},
        compressionOptions_{std::move(compressionOptions)} {}

  std::unique_ptr<EncodingSelectionPolicyBase> createPolicy(
      DataType dataType) const {
    UNIQUE_PTR_FACTORY(
        dataType,
        ManualEncodingSelectionPolicy,
        encodingReadFactors_,
        compressionOptions_,
        std::nullopt);
  }

  static std::vector<EncodingType> possibleEncodings() {
    return {
        EncodingType::Constant,
        EncodingType::Trivial,
        EncodingType::FixedBitWidth,
        EncodingType::MainlyConstant,
        EncodingType::SparseBool,
        EncodingType::Dictionary,
        EncodingType::RLE,
        EncodingType::Varint,
    };
  }

  static std::vector<std::pair<EncodingType, float>> defaultReadFactors() {
    return {
        {EncodingType::Constant, 1.0},
        {EncodingType::Trivial, 0.7},
        {EncodingType::FixedBitWidth, 0.9},
        {EncodingType::MainlyConstant, 1.0},
        {EncodingType::SparseBool, 1.0},
        {EncodingType::Dictionary, 1.0},
        {EncodingType::RLE, 1.0},
        {EncodingType::Varint, 1.0},
    };
  }

  /// Parses semicolon-delimited runtime read-factor config.
  static std::vector<std::pair<nimble::EncodingType, float>>
  parseEncodingReadFactors(const std::string& readFactorsConfig) {
    std::vector<std::pair<nimble::EncodingType, float>> encodingReadFactors;
    std::vector<std::string> parts;
    folly::split(';', folly::trimWhitespace(readFactorsConfig), parts);
    encodingReadFactors.reserve(parts.size());

    std::vector<nimble::EncodingType> possibleEncodings =
        ManualEncodingSelectionPolicyFactory::possibleEncodings();
    std::vector<std::string> possibleEncodingStrs;
    possibleEncodingStrs.reserve(possibleEncodings.size());
    std::transform(
        possibleEncodings.cbegin(),
        possibleEncodings.cend(),
        std::back_inserter(possibleEncodingStrs),
        [](const auto& encoding) { return toString(encoding); });
    for (const auto& part : parts) {
      std::vector<std::string> kv;
      folly::split('=', part, kv);
      NIMBLE_CHECK_EQ(
          kv.size(),
          2,
          "Invalid read factor format. "
          "Expected format is <EncodingType>=<factor>;<EncodingType>=<factor>. "
          "Unable to parse '{}'.",
          part);
      const auto value = folly::tryTo<float>(kv[1]);
      NIMBLE_CHECK(
          value.hasValue(),
          "Unable to parse read factor value '{}' in '{}'. Expected valid float value.",
          kv[1],
          part);
      bool found = false;
      const auto key = folly::trimWhitespace(kv[0]);
      for (auto i = 0; i < possibleEncodings.size(); ++i) {
        auto encoding = possibleEncodings[i];
        // @lint-ignore CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
        if (key == possibleEncodingStrs[i]) {
          found = true;
          encodingReadFactors.emplace_back(encoding, value.value());
          break;
        }
      }
      NIMBLE_CHECK(
          found,
          "Unknown or unexpected read factor encoding '{}'. Allowed values: {}",
          key,
          folly::join(",", possibleEncodingStrs));
    }
    return encodingReadFactors;
  }

 private:
  const std::vector<std::pair<EncodingType, float>> encodingReadFactors_;
  const std::optional<CompressionOptions> compressionOptions_;
};

// This is a learned encoding selection implementation.
template <typename T>
class LearnedEncodingSelectionPolicy : public EncodingSelectionPolicy<T> {
  using physicalType = typename TypeTraits<T>::physicalType;

 public:
  // This model is trained offline and parameters are updated here. The
  // parameters are relatively robust and do not need to be updated unless we
  // add or remove an encoding type.
  struct EncodingPredictionModel {
    explicit EncodingPredictionModel()
        : maxRepeatParam(1.52), minRepeatParam(1.13), uniqueParam(2.589) {}

    float maxRepeatParam;
    float minRepeatParam;
    float uniqueParam;

    float predict(const Statistics<physicalType>& statistics) {
      // TODO: Utilize more features within statistics for prediction.
      const auto maxRepeat = statistics.maxRepeat();
      const auto minRepeat = statistics.minRepeat();
      const auto unique = statistics.uniqueCounts().value().size();
      return maxRepeatParam * maxRepeat + minRepeatParam * minRepeat +
          uniqueParam * unique;
    }
  };

  LearnedEncodingSelectionPolicy(
      std::vector<EncodingType> encodingChoices,
      std::optional<NestedEncodingIdentifier> identifier,
      EncodingPredictionModel encodingModel = EncodingPredictionModel())
      : encodingChoices_{std::move(encodingChoices)},
        identifier_{identifier},
        mlModel_{encodingModel} {}

  LearnedEncodingSelectionPolicy()
      : LearnedEncodingSelectionPolicy{
            possibleEncodingChoices(),
            std::nullopt,
            EncodingPredictionModel()} {}

  EncodingSelectionResult select(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options = {}) override;

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */ = {}) override {
    return {
        .encodingType = EncodingType::Nullable,
    };
  }

  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType parentEncodingType,
      NestedEncodingIdentifier nestedEncodingIdentifier,
      DataType nestedDataType) override {
    // In each sub-level of the encoding selection, we exclude the encodings
    // selected in parent levels. Although this is not required (as hopefully,
    // the model will not pick a nested encoding of the same type as the
    // parent), it provides an additional safty net, making sure the encoding
    // selection will eventually converge, and also slightly speeds up nested
    // encoding selection.
    // TODO: validate the assumptions here compared to brute forcing, to see if
    // the same encoding is selected multiple times in the tree (for example,
    // should we allow trivial string lengths to be encoded using trivial
    // encoding?)
    std::vector<EncodingType> nestedEncodingChoices;
    nestedEncodingChoices.reserve(encodingChoices_.size() - 1);
    for (const auto& encodingType_ : encodingChoices_) {
      if (encodingType_ != parentEncodingType) {
        nestedEncodingChoices.push_back(encodingType_);
      }
    }
    UNIQUE_PTR_FACTORY(
        nestedDataType,
        LearnedEncodingSelectionPolicy,
        std::move(nestedEncodingChoices),
        nestedEncodingIdentifier);
  }

 private:
  static std::vector<EncodingType> possibleEncodingChoices() {
    return {
        EncodingType::Constant,
        EncodingType::Trivial,
        EncodingType::FixedBitWidth,
        EncodingType::MainlyConstant,
        EncodingType::SparseBool,
        EncodingType::Dictionary,
        EncodingType::RLE,
        EncodingType::Varint,
    };
  }

  std::vector<EncodingType> encodingChoices_;
  std::optional<NestedEncodingIdentifier> identifier_;
  EncodingPredictionModel mlModel_;
};

template <typename T>
EncodingSelectionResult LearnedEncodingSelectionPolicy<T>::select(
    std::span<const typename LearnedEncodingSelectionPolicy<T>::physicalType>
        values,
    const Statistics<typename LearnedEncodingSelectionPolicy<T>::physicalType>&
        statistics,
    const Encoding::Options& /* options */) {
  if (values.empty()) {
    return {
        .encodingType = EncodingType::Trivial,
    };
  }

  const auto prediction = mlModel_.predict(statistics);
  if (prediction > 0.1) {
    return {
        .encodingType = EncodingType::Trivial,
    };
  }
  // TODO: Implement a multi-class Encoding model so that we can predict not
  // only a trivial encoding but also other encodings.

  return {
      .encodingType = EncodingType::Trivial,
  };
}

template <typename T>
class ReplayedEncodingSelectionPolicy : public EncodingSelectionPolicy<T> {
 public:
  using physicalType = typename nimble::TypeTraits<T>::physicalType;

  ReplayedEncodingSelectionPolicy(
      EncodingLayout encodingLayout,
      std::optional<CompressionOptions> compressionOptions,
      const EncodingSelectionPolicyCreator& encodingSelectionPolicyCreator)
      : compressionOptions_{std::move(compressionOptions)},
        encodingSelectionPolicyCreator_{encodingSelectionPolicyCreator},
        encodingLayout_{std::move(encodingLayout)} {}

  EncodingSelectionResult select(
      std::span<const physicalType> /* values */,
      const nimble::Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */ = {}) override {
    if (!compressionOptions_.has_value()) {
      return {
          .encodingType = encodingLayout_.encodingType(),
      };
    }
    return {
        .encodingType = encodingLayout_.encodingType(),
        .compressionPolicyFactory = [this]() {
          return std::make_unique<ReplayedCompressionPolicy>(
              encodingLayout_.compressionType(), compressionOptions_.value());
        }};
  }

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */ = {}) override {
    // NullableEncoding asks createImpl() for nullable data and nulls children.
    // The replay policy is initialized with the data layout, so synthesize the
    // nullable parent shape here.
    encodingLayout_ = EncodingLayout{
        EncodingType::Nullable,
        {},
        CompressionType::Uncompressed,
        {
            /* Data */ std::move(encodingLayout_),
            /* Nulls */ std::nullopt,
        }};
    return {
        .encodingType = EncodingType::Nullable,
    };
  }

 protected:
  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      nimble::EncodingType /* parentEncodingType */,
      nimble::NestedEncodingIdentifier nestedEncodingIdentifier,
      nimble::DataType nestedDataType) override {
    NIMBLE_CHECK_LT(
        nestedEncodingIdentifier,
        encodingLayout_.childrenCount(),
        "Sub-encoding identifier out of range.");
    auto child = encodingLayout_.child(nestedEncodingIdentifier);

    if (child.has_value()) {
      UNIQUE_PTR_FACTORY(
          nestedDataType,
          ReplayedEncodingSelectionPolicy,
          child.value(),
          compressionOptions_,
          encodingSelectionPolicyCreator_);
    } else {
      return encodingSelectionPolicyCreator_(nestedDataType);
    }
  }

 private:
  const std::optional<CompressionOptions> compressionOptions_;
  const EncodingSelectionPolicyCreator& encodingSelectionPolicyCreator_;
  EncodingLayout encodingLayout_;
};

} // namespace facebook::nimble::legacy
