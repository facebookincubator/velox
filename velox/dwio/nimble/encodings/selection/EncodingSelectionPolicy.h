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

#include <glog/logging.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "velox/common/base/SuccinctPrinter.h"
#include "velox/dwio/nimble/common/Constants.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSizeEstimation.h"

namespace facebook::nimble {

using EncodingSelectionPolicyCreator =
    std::function<std::unique_ptr<EncodingSelectionPolicyBase>(DataType)>;

// The following enables encoding selection debug messages. By default, these
// logs are turned off (with zero overhead). In tests (or in debug sessions), we
// enable these logs.
#ifndef NIMBLE_ENCODING_SELECTION_DEBUG_MAX_ITEMS
#define NIMBLE_ENCODING_SELECTION_DEBUG_MAX_ITEMS 50
#endif

#ifdef NIMBLE_ENCODING_SELECTION_DEBUG
#define NIMBLE_SELECTION_LOG(stream) LOG(INFO) << stream
#else
#define NIMBLE_SELECTION_LOG(stream)
#endif

#define COMMA ,
#define UNIQUE_PTR_FACTORY_EXTRA(data_type, class, extra_types, ...)     \
  switch (data_type) {                                                   \
    case facebook::nimble::DataType::Uint8: {                            \
      return std::make_unique<class<uint8_t extra_types>>(__VA_ARGS__);  \
    }                                                                    \
    case facebook::nimble::DataType::Int8: {                             \
      return std::make_unique<class<int8_t extra_types>>(__VA_ARGS__);   \
    }                                                                    \
    case facebook::nimble::DataType::Uint16: {                           \
      return std::make_unique<class<uint16_t extra_types>>(__VA_ARGS__); \
    }                                                                    \
    case facebook::nimble::DataType::Int16: {                            \
      return std::make_unique<class<int16_t extra_types>>(__VA_ARGS__);  \
    }                                                                    \
    case facebook::nimble::DataType::Uint32: {                           \
      return std::make_unique<class<uint32_t extra_types>>(__VA_ARGS__); \
    }                                                                    \
    case facebook::nimble::DataType::Int32: {                            \
      return std::make_unique<class<int32_t extra_types>>(__VA_ARGS__);  \
    }                                                                    \
    case facebook::nimble::DataType::Uint64: {                           \
      return std::make_unique<class<uint64_t extra_types>>(__VA_ARGS__); \
    }                                                                    \
    case facebook::nimble::DataType::Int64: {                            \
      return std::make_unique<class<int64_t extra_types>>(__VA_ARGS__);  \
    }                                                                    \
    case facebook::nimble::DataType::Float: {                            \
      return std::make_unique<class<float extra_types>>(__VA_ARGS__);    \
    }                                                                    \
    case facebook::nimble::DataType::Double: {                           \
      return std::make_unique<class<double extra_types>>(__VA_ARGS__);   \
    }                                                                    \
    case facebook::nimble::DataType::Bool: {                             \
      return std::make_unique<class<bool extra_types>>(__VA_ARGS__);     \
    }                                                                    \
    case facebook::nimble::DataType::String: {                           \
      return std::make_unique<class<std::string_view extra_types>>(      \
          __VA_ARGS__);                                                  \
    }                                                                    \
    case facebook::nimble::DataType::Undefined:                          \
      break;                                                             \
  }                                                                      \
  NIMBLE_UNREACHABLE("Unsupported data type {}.", toString(data_type))

#define UNIQUE_PTR_FACTORY(data_type, class, ...) \
  UNIQUE_PTR_FACTORY_EXTRA(data_type, class, , __VA_ARGS__)

/// Manual encoding selection implementation.
/// Uses a manually crafted model to choose the most appropriate encoding based
/// on the provided statistics.
template <typename T>
class ManualEncodingSelectionPolicy : public EncodingSelectionPolicy<T> {
 public:
  using physicalType = typename TypeTraits<T>::physicalType;

  ManualEncodingSelectionPolicy(
      std::vector<std::pair<EncodingType, float>> encodingReadFactors,
      std::optional<CompressionOptions> compressionOptions,
      std::optional<NestedEncodingIdentifier> identifier,
      std::optional<std::vector<std::pair<EncodingType, float>>>
          nestedEncodingReadFactors = std::nullopt)
      : candidateEncodingReadFactors_{std::move(encodingReadFactors)},
        compressionOptions_{std::move(compressionOptions)},
        identifier_{identifier},
        nestedEncodingReadFactorsOverride_{
            std::move(nestedEncodingReadFactors)} {}

  EncodingSelectionResult select(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) override {
    if (values.empty()) {
      return {
          .encodingType = EncodingType::Trivial,
          .encodingConfig = {},
          .estimatedSize = std::nullopt,
      };
    }

    auto candidateEncodingReadFactors = candidateEncodingReadFactors_;
    // TODO: Remove this opt-in once ALP is production-ready for default
    // selection.
    if constexpr (isFloatingPointType<T>()) {
      if (options.allowNestedAlpSelection &&
          (identifier_ == EncodingIdentifiers::Dictionary::Alphabet ||
           identifier_ == EncodingIdentifiers::MainlyConstant::OtherValues ||
           identifier_ == EncodingIdentifiers::RunLength::RunValues) &&
          std::none_of(
              candidateEncodingReadFactors.begin(),
              candidateEncodingReadFactors.end(),
              [](const auto& entry) {
                return entry.first == EncodingType::ALP;
              })) {
        candidateEncodingReadFactors.emplace_back(EncodingType::ALP, 1.0);
      }
    }

    // Fast path: when there are no candidate encodings, fall back to Trivial.
    if (candidateEncodingReadFactors.empty()) {
      return {
          .encodingType = EncodingType::Trivial,
          .encodingConfig = {},
          .estimatedSize = std::nullopt,
      };
    }

    float minCost = std::numeric_limits<float>::max();
    EncodingType selectedEncoding = EncodingType::Trivial;
    std::optional<uint64_t> selectedEstimatedSize;
    // Iterate on all candidate encodings, and pick the encoding with the
    // minimal cost.
    for (const auto& entry : candidateEncodingReadFactors) {
      const auto encodingType = entry.first;
      const auto estimatedSize =
          detail::EncodingSizeEstimation<T>::estimateSize(
              encodingType, values, statistics, options);
      if (!estimatedSize.has_value()) {
        NIMBLE_SELECTION_LOG(encodingType << " encoding is incompatible.");
        continue;
      }

      // We use read factor weights to raise/lower the favorability of each
      // encoding.
      const auto readFactor = entry.second;
      const auto cost = estimatedSize.value() * readFactor;
      NIMBLE_SELECTION_LOG(
          "Encoding: " << encodingType << ", Size: "
                       << velox::succinctBytes(estimatedSize.value())
                       << ", Factor: " << readFactor << ", Cost: " << cost);
      if (cost < minCost) {
        minCost = cost;
        selectedEncoding = encodingType;
        selectedEstimatedSize = estimatedSize;
      }
    }

    NIMBLE_SELECTION_LOG(
        "Selected Encoding"
        << (identifier_.has_value()
                ? folly::to<std::string>(
                      " [NestedEncodingIdentifier: ", identifier_.value(), "]")
                : "")
        << ": " << selectedEncoding << ", Sampled Data: "
        << folly::join(
               ",",
               std::span<const physicalType>{
                   values.data(),
                   std::min(
                       size_t(NIMBLE_ENCODING_SELECTION_DEBUG_MAX_ITEMS),
                       values.size())})
        << (values.size() > size_t(NIMBLE_ENCODING_SELECTION_DEBUG_MAX_ITEMS)
                ? "..."
                : ""));
    if (!compressionOptions_.has_value()) {
      return {
          .encodingType = selectedEncoding,
          .encodingConfig = {},
          .estimatedSize = selectedEstimatedSize};
    }
    // Encoding selection optimizes the in-memory layout. Compression is still
    // attempted for leaf data streams to reduce persistent storage size.
    return {
        .encodingType = selectedEncoding,
        .encodingConfig = {},
        .estimatedSize = selectedEstimatedSize,
        .compressionPolicyFactory = [compressionOptions =
                                         compressionOptions_.value(),
                                     selectedEncoding]() {
          return std::make_unique<ConfiguredCompressionPolicy>(
              compressionOptions, selectedEncoding);
        }};
  }

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) override {
    return {
        .encodingType = EncodingType::Nullable,
        .encodingConfig = {},
        .estimatedSize = std::nullopt,
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
    // parent), it provides an additional safety net, making sure the encoding
    // selection will eventually converge, and also slightly speeds up nested
    // encoding selection.
    // TODO: validate the assumptions here compared to brute forcing, to see if
    // the same encoding is selected multiple times in the tree (for example,
    // should we allow trivial string lengths to be encoded using trivial
    // encoding?)
    std::vector<std::pair<EncodingType, float>> nestedEncodingReadFactors;
    const auto& sourceEncodingReadFactors =
        nestedEncodingReadFactorsOverride_.has_value()
        ? nestedEncodingReadFactorsOverride_.value()
        : candidateEncodingReadFactors_;
    nestedEncodingReadFactors.reserve(sourceEncodingReadFactors.size());
    for (const auto& entry : sourceEncodingReadFactors) {
      if (entry.first != parentEncodingType) {
        nestedEncodingReadFactors.emplace_back(entry);
      }
    }
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
    // SubIntSplit decomposes its input into bit-range segments, each
    // independently re-encoded via encodeNested(). Segments often look very
    // different from the original column (narrow, possibly skewed
    // residuals), so offer additional integer-compression candidates here
    // that aren't part of the global default read factors. This only affects
    // direct children of a SubIntSplit node: recursion is bounded because a
    // child's own encodingType (e.g. PFOR) -- not SubIntSplit -- is what gets
    // passed to *its* children's createImpl.
    if (parentEncodingType == EncodingType::SubIntSplit) {
      for (const auto& pair :
           {std::pair{EncodingType::PFOR, 0.9f},
            std::pair{EncodingType::SimdForBitpack, 0.9f},
            std::pair{EncodingType::BlockBitPacking, 0.9f}}) {
        nestedEncodingReadFactors.push_back(pair);
      }
    }
#endif
    UNIQUE_PTR_FACTORY(
        nestedDataType,
        ManualEncodingSelectionPolicy,
        std::move(nestedEncodingReadFactors),
        compressionOptions_,
        nestedEncodingIdentifier,
        std::nullopt);
  }

 private:
  // Candidate encodings and their read-cost factors. Encoding selection uses
  // estimatedSize * readFactor as the cost, so a lower factor makes an
  // encoding more likely to be picked. Right now, these represent mostly the
  // CPU cost to decode values. Trivial is boosted with a lower factor because
  // it also benefits from applying compression.
  // See ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors for
  // the default.
  const std::vector<std::pair<EncodingType, float>>
      candidateEncodingReadFactors_;
  // When nullopt, compression is disabled for this policy and all nested
  // policies created from it.
  const std::optional<CompressionOptions> compressionOptions_;
  const std::optional<NestedEncodingIdentifier> identifier_;
  // An override for the next nested policy. Deeper policies inherit their
  // parent's already-filtered candidates.
  const std::optional<std::vector<std::pair<EncodingType, float>>>
      nestedEncodingReadFactorsOverride_;
};

class ManualEncodingSelectionPolicyFactory {
 public:
  /// TODO: Add ALP once it is production-ready for default manual selection.
  static std::vector<std::pair<EncodingType, float>>
  defaultEncodingReadFactors();

  /// Parses semicolon-delimited runtime read-factor config.
  /// Allows explicit opt-in to parseable encodings that are not part of
  /// defaultEncodingReadFactors().
  static std::vector<std::pair<nimble::EncodingType, float>>
  parseEncodingReadFactors(const std::string& readFactorsConfig);

  /// Builds a factory from a nimble.encoding_selection_config string of the
  /// form "type:default[,read_factors:<E1>=<f1>;<E2>=<f2>;...]". Ignores the
  /// 'type' key (already used by createEncodingSelectionPolicyFactory). Returns
  /// nullopt when no 'read_factors' are given, so the caller keeps the
  /// nimble.manual_encoding_selection_read_factors default; otherwise a factory
  /// whose read factors override that config. NimbleUserError on a malformed
  /// entry or unknown key.
  static std::optional<ManualEncodingSelectionPolicyFactory> create(
      std::string_view configStr,
      std::optional<CompressionOptions> compressionOptions =
          CompressionOptions{});

  ManualEncodingSelectionPolicyFactory(
      std::vector<std::pair<EncodingType, float>> encodingReadFactors =
          defaultEncodingReadFactors(),
      std::optional<CompressionOptions> compressionOptions =
          CompressionOptions{},
      std::optional<std::vector<std::pair<EncodingType, float>>>
          nestedEncodingReadFactors = std::nullopt);

  std::unique_ptr<EncodingSelectionPolicyBase> createPolicy(
      DataType dataType) const;

  /// All encodings the string-config parsers accept (production +
  /// experimental). Also used to resolve encoding names to EncodingType.
  static std::vector<EncodingType> possibleEncodings();

 private:
  const std::vector<std::pair<EncodingType, float>> encodingReadFactors_;
  const std::optional<CompressionOptions> compressionOptions_;
  const std::optional<std::vector<std::pair<EncodingType, float>>>
      nestedEncodingReadFactors_;
};

/// Learned encoding selection implementation.
template <typename T>
class LearnedEncodingSelectionPolicy : public EncodingSelectionPolicy<T> {
  using physicalType = typename TypeTraits<T>::physicalType;

 public:
  /// Model trained offline for learned encoding selection.
  /// Parameters are relatively robust and do not need updates unless encodings
  /// are added or removed.
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
      const Encoding::Options& options) override;

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) override {
    return {
        .encodingType = EncodingType::Nullable,
    };
  }

 protected:
  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType parentEncodingType,
      NestedEncodingIdentifier nestedEncodingIdentifier,
      DataType nestedDataType) override {
    // In each sub-level of the encoding selection, we exclude the encodings
    // selected in parent levels. Although this is not required (as hopefully,
    // the model will not pick a nested encoding of the same type as the
    // parent), it provides an additional safety net, making sure the encoding
    // selection will eventually converge, and also slightly speeds up nested
    // encoding selection.
    // TODO: validate the assumptions here compared to brute forcing, to see if
    // the same encoding is selected multiple times in the tree (for example,
    // should we allow trivial string lengths to be encoded using trivial
    // encoding?)
    std::vector<std::pair<EncodingType, float>> nestedEncodingReadFactors;
    const auto& sourceEncodingReadFactors =
        nestedEncodingReadFactorsOverride_.has_value()
        ? nestedEncodingReadFactorsOverride_.value()
        : candidateEncodingReadFactors_;
    nestedEncodingReadFactors.reserve(sourceEncodingReadFactors.size());
    for (const auto& entry : sourceEncodingReadFactors) {
      if (entry.first != parentEncodingType) {
        nestedEncodingReadFactors.emplace_back(entry);
      }
    }
    UNIQUE_PTR_FACTORY(
        nestedDataType,
        ManualEncodingSelectionPolicy,
        std::move(nestedEncodingReadFactors),
        compressionOptions_,
        nestedEncodingIdentifier,
        std::nullopt);
  }

 private:
  // Candidate encodings and their read-cost factors. Encoding selection uses
  // estimatedSize * readFactor as the cost, so a lower factor makes an
  // encoding more likely to be picked. Right now, these represent mostly the
  // CPU cost to decode values. Trivial is boosted with a lower factor because
  // it also benefits from applying compression.
  // See ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors for
  // the default.
  const std::vector<std::pair<EncodingType, float>>
      candidateEncodingReadFactors_;
  // When nullopt, compression is disabled for this policy and all nested
  // policies created from it.
  const std::optional<CompressionOptions> compressionOptions_;
  const std::optional<NestedEncodingIdentifier> identifier_;
  // An override for the next nested policy. Deeper policies inherit their
  // parent's already-filtered candidates.
  const std::optional<std::vector<std::pair<EncodingType, float>>>
      nestedEncodingReadFactorsOverride_;
};

class ManualEncodingSelectionPolicyFactory {
 public:
  /// TODO: Add ALP once it is production-ready for default manual selection.
  static std::vector<std::pair<EncodingType, float>>
  defaultEncodingReadFactors();

  /// Parses semicolon-delimited runtime read-factor config.
  /// Allows explicit opt-in to parseable encodings that are not part of
  /// defaultEncodingReadFactors().
  static std::vector<std::pair<nimble::EncodingType, float>>
  parseEncodingReadFactors(const std::string& readFactorsConfig);

  /// Builds a factory from a nimble.encoding_selection_config string of the
  /// form "type:default[,read_factors:<E1>=<f1>;<E2>=<f2>;...]". Ignores the
  /// 'type' key (already used by createEncodingSelectionPolicyFactory). Returns
  /// nullopt when no 'read_factors' are given, so the caller keeps the
  /// nimble.manual_encoding_selection_read_factors default; otherwise a factory
  /// whose read factors override that config. NimbleUserError on a malformed
  /// entry or unknown key.
  static std::optional<ManualEncodingSelectionPolicyFactory> create(
      std::string_view configStr,
      std::optional<CompressionOptions> compressionOptions =
          CompressionOptions{});

  ManualEncodingSelectionPolicyFactory(
      std::vector<std::pair<EncodingType, float>> encodingReadFactors =
          defaultEncodingReadFactors(),
      std::optional<CompressionOptions> compressionOptions =
          CompressionOptions{},
      std::optional<std::vector<std::pair<EncodingType, float>>>
          nestedEncodingReadFactors = std::nullopt);

  std::unique_ptr<EncodingSelectionPolicyBase> createPolicy(
      DataType dataType) const;

  /// All encodings the string-config parsers accept (production +
  /// experimental). Also used to resolve encoding names to EncodingType.
  static std::vector<EncodingType> possibleEncodings();

 private:
  const std::vector<std::pair<EncodingType, float>> encodingReadFactors_;
  const std::optional<CompressionOptions> compressionOptions_;
  const std::optional<std::vector<std::pair<EncodingType, float>>>
      nestedEncodingReadFactors_;
};

/// Learned encoding selection implementation.
template <typename T>
class LearnedEncodingSelectionPolicy : public EncodingSelectionPolicy<T> {
  using physicalType = typename TypeTraits<T>::physicalType;

 public:
  /// Model trained offline for learned encoding selection.
  /// Parameters are relatively robust and do not need updates unless encodings
  /// are added or removed.
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
      const Encoding::Options& options) override;

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) override {
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
    // parent), it provides an additional safety net, making sure the encoding
    // selection will eventually converge, and also slightly speeds up nested
    // encoding selection.
    //
    // TODO: validate the assumptions here compared to brute forcing, to see if
    // the same encoding is selected multiple times in the tree (for example,
    // should we allow trivial string lengths to be encoded using trivial
    // encoding?)
    std::vector<EncodingType> nestedEncodingChoices;
    nestedEncodingChoices.reserve(encodingChoices_.size());
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
  // TODO: Add ALP once it is production-ready for learned selection.
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
        .encodingConfig = {},
        .estimatedSize = std::nullopt,
    };
  }

  const auto prediction = mlModel_.predict(statistics);
  if (prediction > 0.1) {
    return {
        .encodingType = EncodingType::Trivial,
        .encodingConfig = {},
        .estimatedSize = std::nullopt,
    };
  }
  // TODO: Implement a multi-class Encoding model so that we can predict not
  // only a trivial encoding but also other encodings.

  return {
      .encodingType = EncodingType::Trivial,
      .encodingConfig = {},
      .estimatedSize = std::nullopt,
  };
}

template <typename T>
class ReplayedEncodingSelectionPolicy
    : public nimble::EncodingSelectionPolicy<T> {
 public:
  using physicalType = typename nimble::TypeTraits<T>::physicalType;

  ReplayedEncodingSelectionPolicy(
      EncodingLayout encodingLayout,
      std::optional<CompressionOptions> compressionOptions,
      const EncodingSelectionPolicyCreator& encodingSelectionPolicyCreator)
      : compressionOptions_{std::move(compressionOptions)},
        encodingSelectionPolicyCreator_{encodingSelectionPolicyCreator},
        encodingLayout_{std::move(encodingLayout)} {}

  nimble::EncodingSelectionResult select(
      std::span<const physicalType> /* values */,
      const nimble::Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) override {
    if (!compressionOptions_.has_value()) {
      return {
          .encodingType = encodingLayout_.encodingType(),
          .encodingConfig = encodingLayout_.config(),
          .estimatedSize = std::nullopt,
      };
    }
    return {
        .encodingType = encodingLayout_.encodingType(),
        .encodingConfig = encodingLayout_.config(),
        .estimatedSize = std::nullopt,
        .compressionPolicyFactory = [this]() {
          return std::make_unique<ReplayedCompressionPolicy>(
              encodingLayout_.compressionType(), compressionOptions_.value());
        }};
  }

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) override {
    // NullableEncoding asks createImpl() for nullable data and nulls children.
    // The replay policy is initialized with the data layout, so synthesize the
    // nullable parent shape here.
    encodingLayout_ = EncodingLayout{
        EncodingType::Nullable,
        {},
        CompressionType::Uncompressed,
        {
            /*Data=*/std::move(encodingLayout_),
            /*Nulls=*/std::nullopt,
        }};
    return {
        .encodingType = EncodingType::Nullable,
        .encodingConfig = {},
        .estimatedSize = std::nullopt,
    };
  }

 protected:
  std::unique_ptr<nimble::EncodingSelectionPolicyBase> createImpl(
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
  const EncodingSelectionPolicyCreator encodingSelectionPolicyCreator_;
  EncodingLayout encodingLayout_;
};

#undef COMMA

} // namespace facebook::nimble
