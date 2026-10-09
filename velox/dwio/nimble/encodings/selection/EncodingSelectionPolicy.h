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
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "velox/common/base/SuccinctPrinter.h"
#include "velox/dwio/nimble/common/Constants.h"
#include "velox/dwio/nimble/encodings/BitRangeSplitEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSizeEstimation.h"

namespace facebook::nimble {

using EncodingSelectionPolicyCreator =
    std::function<std::unique_ptr<EncodingSelectionPolicyBase>(DataType)>;

using NestedEncodingCompressionRatiosProvider =
    std::function<std::optional<std::vector<std::pair<EncodingType, float>>>(
        EncodingType,
        NestedEncodingIdentifier,
        DataType)>;

namespace detail {

/// Checks whether the candidates contain an AlpLike encoding (ALP or ALP_RD).
bool hasAlpLikeCandidate(
    const std::vector<std::pair<EncodingType, float>>& candidates);

/// Checks whether a layout tree contains an AlpLike encoding (ALP or ALP_RD).
bool hasAlpLikeEncoding(const EncodingLayout& layout);

/// Checks whether a value child that can select ALP or ALPRD lacks a layout.
/// Auxiliary streams such as null flags and dictionary indices are excluded.
bool hasUnspecifiedValueEncoding(const EncodingLayout& layout);

} // namespace detail

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
          nestedEncodingReadFactors = std::nullopt,
      std::optional<std::vector<std::pair<EncodingType, float>>>
          estimatedCompressionRatios = std::nullopt,
      NestedEncodingCompressionRatiosProvider
          nestedEncodingCompressionRatiosProvider = {})
      : candidateEncodingReadFactors_{std::move(encodingReadFactors)},
        compressionOptions_{std::move(compressionOptions)},
        identifier_{identifier},
        nestedEncodingReadFactorsOverride_{
            std::move(nestedEncodingReadFactors)},
        estimatedCompressionRatios_{std::move(estimatedCompressionRatios)},
        nestedEncodingCompressionRatiosProvider_{
            std::move(nestedEncodingCompressionRatiosProvider)} {
    if (estimatedCompressionRatios_.has_value()) {
      for (size_t i = 0; i < estimatedCompressionRatios_->size(); ++i) {
        const auto& [encodingType, compressionRatio] =
            estimatedCompressionRatios_->at(i);
        NIMBLE_USER_CHECK(
            std::isfinite(compressionRatio) && compressionRatio > 0 &&
                compressionRatio <= 1,
            "Estimated compression ratio for {} must be finite and in (0, 1], got {}.",
            toString(encodingType),
            compressionRatio);
        for (size_t j = 0; j < i; ++j) {
          NIMBLE_USER_CHECK(
              estimatedCompressionRatios_->at(j).first != encodingType,
              "Duplicate estimated compression ratio for encoding {}.",
              toString(encodingType));
        }
      }
    }
  }

  EncodingSelectionResult select(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) override {
    return selectScored(values, statistics, options).result;
  }

  /// Selects an encoding and retains its weighted cost for a compound parent.
  /// Public select() intentionally exposes only the established result type.
  ScoredEncodingSelection selectScored(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) override {
    if (values.empty()) {
      return {
          .result =
              {
                  .encodingType = EncodingType::Trivial,
                  .encodingConfig = {},
                  .estimatedSize = std::nullopt,
              },
          .estimatedSize = 0,
          .cost = 0,
      };
    }

    const auto& candidateEncodingReadFactors =
        this->candidateEncodingReadFactors();
    const auto estimateTrivial = [&]() {
      auto estimatedSize = detail::EncodingSizeEstimation<T>::estimateSize(
          EncodingType::Trivial, values, statistics, options);
      if (!estimatedSize.has_value()) {
        if constexpr (std::is_same_v<physicalType, std::string_view>) {
          estimatedSize = TrivialEncoding<physicalType>::estimateSize(
              values.size(), statistics, options);
        } else {
          estimatedSize =
              TrivialEncoding<physicalType>::estimateSize(values.size());
        }
      }
      return estimatedSize.value();
    };

    // Fast path: when there are no candidate encodings, fall back to Trivial.
    if (candidateEncodingReadFactors.empty()) {
      const auto estimatedSize = estimateTrivial();
      return {
          .result =
              {
                  .encodingType = EncodingType::Trivial,
                  .encodingConfig = {},
                  .estimatedSize = std::nullopt,
              },
          .estimatedSize = estimatedSize,
          .cost = static_cast<double>(estimatedSize) *
              estimatedCompressionRatio(EncodingType::Trivial),
      };
    }

    double minCost = std::numeric_limits<double>::max();
    EncodingType selectedEncoding = EncodingType::Trivial;
    std::optional<uint64_t> selectedEstimatedSize;
    bool selectedFallback{false};
    // Iterate on all candidate encodings, and pick the encoding with the
    // minimal cost.
    for (const auto& entry : candidateEncodingReadFactors) {
      const auto encodingType = entry.first;
      const auto score = detail::EncodingSizeEstimation<T>::estimateScore(
          encodingType,
          values,
          statistics,
          options,
          entry.second * estimatedCompressionRatio(encodingType),
          [this]<typename NestedT>(
              EncodingType parentEncodingType,
              NestedEncodingIdentifier identifier,
              std::span<const NestedT> nestedValues,
              uint32_t targetRowCount,
              const Encoding::Options& nestedOptions) {
            return this->template scoreNestedChild<NestedT>(
                parentEncodingType,
                identifier,
                nestedValues,
                targetRowCount,
                nestedOptions);
          });
      if (!score.has_value()) {
        NIMBLE_SELECTION_LOG(encodingType << " encoding is incompatible.");
        continue;
      }

      // We use read factor weights to raise/lower the favorability of each
      // encoding.
      NIMBLE_SELECTION_LOG(
          "Encoding: " << encodingType << ", Size: "
                       << velox::succinctBytes(score->estimatedSize)
                       << ", Read Factor: " << entry.second
                       << ", Estimated Compression Ratio: "
                       << estimatedCompressionRatio(encodingType)
                       << ", Cost: " << score->cost);
      if (score->cost < minCost) {
        minCost = score->cost;
        selectedEncoding = encodingType;
        selectedEstimatedSize = score->estimatedSize;
      }
    }

    // A configured candidate list can still contain no encoding compatible
    // with this physical child type. Preserve the established Trivial
    // fallback, and give recursive selection an actual size to score.
    if (!selectedEstimatedSize.has_value()) {
      selectedEstimatedSize = estimateTrivial();
      minCost = static_cast<double>(selectedEstimatedSize.value()) *
          estimatedCompressionRatio(EncodingType::Trivial);
      selectedFallback = true;
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
          .result =
              {.encodingType = selectedEncoding,
               .encodingConfig = {},
               .estimatedSize =
                   selectedFallback ? std::nullopt : selectedEstimatedSize},
          .estimatedSize = selectedEstimatedSize,
          .cost = minCost};
    }
    // Encoding selection optimizes the in-memory layout. Compression is still
    // attempted for leaf data streams to reduce persistent storage size.
    return {
        .result =
            {.encodingType = selectedEncoding,
             .encodingConfig = {},
             .estimatedSize =
                 selectedFallback ? std::nullopt : selectedEstimatedSize,
             .compressionPolicyFactory =
                 [compressionOptions = compressionOptions_.value(),
                  selectedEncoding]() {
                   return std::make_unique<ConfiguredCompressionPolicy>(
                       compressionOptions, selectedEncoding);
                 }},
        .estimatedSize = selectedEstimatedSize,
        .cost = minCost};
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

  bool hasAlpLikeCandidates() const override {
    return detail::hasAlpLikeCandidate(candidateEncodingReadFactors_) ||
        (nestedEncodingReadFactorsOverride_ &&
         detail::hasAlpLikeCandidate(*nestedEncodingReadFactorsOverride_));
  }

  /// Returns the configured candidates for this selection node.
  const std::vector<std::pair<EncodingType, float>>&
  candidateEncodingReadFactors() const {
    return candidateEncodingReadFactors_;
  }

  const std::optional<std::vector<std::pair<EncodingType, float>>>&
  estimatedCompressionRatios() const {
    return estimatedCompressionRatios_;
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
      const bool isCandidate = parentEncodingType == EncodingType::BitRangeSplit
          ? detail::BitRangeSplitEncodingBase::isValidSectionEncodingCandidate(
                entry.first)
          : entry.first != parentEncodingType;
      if (isCandidate) {
        nestedEncodingReadFactors.emplace_back(entry);
      }
    }
    UNIQUE_PTR_FACTORY(
        nestedDataType,
        ManualEncodingSelectionPolicy,
        std::move(nestedEncodingReadFactors),
        compressionOptions_,
        nestedEncodingIdentifier,
        std::nullopt,
        nestedEncodingCompressionRatiosProvider_
            ? nestedEncodingCompressionRatiosProvider_(
                  parentEncodingType, nestedEncodingIdentifier, nestedDataType)
            : std::nullopt,
        nestedEncodingCompressionRatiosProvider_);
  }

 private:
  // Returns this node's configured compressed-size ratio, or no adjustment.
  float estimatedCompressionRatio(EncodingType encodingType) const {
    if (compressionOptions_.has_value() &&
        estimatedCompressionRatios_.has_value()) {
      for (const auto& [compressionEncoding, compressionRatio] :
           estimatedCompressionRatios_.value()) {
        if (compressionEncoding == encodingType) {
          return compressionRatio;
        }
      }
    }
    return 1.0;
  }

  // Selects an immediate child on sampled values and projects its size and
  // weighted cost to the target child row count.
  template <typename NestedT>
  std::optional<EncodingCandidateScore> scoreNestedChild(
      EncodingType parentEncodingType,
      NestedEncodingIdentifier identifier,
      std::span<const NestedT> values,
      uint32_t targetRowCount,
      const Encoding::Options& options) {
    if (values.empty()) {
      NIMBLE_CHECK_EQ(targetRowCount, 0);
      return EncodingCandidateScore{0, 0};
    }
    NIMBLE_CHECK_LE(values.size(), targetRowCount);
    auto child = this->template create<NestedT>(parentEncodingType, identifier);
    auto* typed = static_cast<EncodingSelectionPolicy<NestedT>*>(child.get());
    const auto statistics = Statistics<NestedT>::create(values);
    auto scored = typed->selectScored(values, statistics, options);
    if (!scored.estimatedSize.has_value()) {
      scored.estimatedSize =
          detail::EncodingSizeEstimation<NestedT>::estimateSize(
              scored.result.encodingType, values, statistics, options);
    }
    if (!scored.estimatedSize.has_value()) {
      return std::nullopt;
    }
    const auto sampledSize = scored.estimatedSize.value();
    uint64_t estimatedSize{sampledSize};
    if (targetRowCount != values.size() &&
        scored.result.encodingType == EncodingType::Constant) {
      // Constant payload size is independent of row count. Replace only the
      // prefix instead of linearly scaling its fixed header and value.
      const auto sampledPrefix = EncodingPrefix::serializedSize(
          static_cast<uint32_t>(values.size()), options.useVarintRowCount);
      NIMBLE_CHECK_GE(sampledSize, sampledPrefix);
      estimatedSize = sampledSize - sampledPrefix +
          EncodingPrefix::serializedSize(
                          targetRowCount, options.useVarintRowCount);
    } else if (targetRowCount != values.size()) {
      // Encoded size need not scale linearly with row count.
      const auto projectedSize =
          detail::EncodingSizeEstimation<NestedT>::estimateProjectedSize(
              scored.result.encodingType, targetRowCount, statistics, options);
      if (!projectedSize.has_value()) {
        return std::nullopt;
      }
      estimatedSize = projectedSize.value();
    }
    const auto sampledCost = scored.cost.value_or(sampledSize);
    return EncodingCandidateScore{
        estimatedSize,
        sampledSize == 0
            ? 0
            : static_cast<double>(estimatedSize) * sampledCost / sampledSize};
  }

  // Candidate encodings and their read-cost factors. Encoding selection uses
  // estimatedSize * readFactor * estimatedCompressionRatio as the cost, so a
  // lower factor or ratio makes an encoding more likely to be picked. Right
  // now, read factors represent mostly the CPU cost to decode values. Trivial
  // is boosted with a lower read factor because it also benefits from applying
  // compression.
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
  // Optional estimates of compressed size / encoded size for candidates at
  // this node. These adjust selection cost without changing eligibility.
  const std::optional<std::vector<std::pair<EncodingType, float>>>
      estimatedCompressionRatios_;
  // Supplies compression estimates for a specific child of a composite
  // encoding. Returning nullopt preserves unadjusted selection.
  const NestedEncodingCompressionRatiosProvider
      nestedEncodingCompressionRatiosProvider_;
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
          nestedEncodingReadFactors = std::nullopt,
      NestedEncodingCompressionRatiosProvider
          nestedEncodingCompressionRatiosProvider = {});

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
  const NestedEncodingCompressionRatiosProvider
      nestedEncodingCompressionRatiosProvider_;
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
    // NullableEncoding asks createImpl() for non-null value and null-flag
    // children.
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

  bool hasAlpLikeCandidates() const override {
    // Only an unspecified value encoding needs the fallback's capabilities.
    return detail::hasAlpLikeEncoding(encodingLayout_) ||
        (detail::hasUnspecifiedValueEncoding(encodingLayout_) &&
         encodingSelectionPolicyCreator_(TypeTraits<T>::dataType)
             ->hasAlpLikeCandidates());
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
