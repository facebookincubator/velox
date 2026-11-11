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

#include <folly/hash/Hash.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

// Random encoding selection lives in the nimble::testing namespace and its own
// target to keep it out of the core encoding-selection library: it is meant for
// fuzz/stress testing and validation tooling, never for production table
// ingestion.
namespace facebook::nimble::testing {

/// Randomized encoding selection for fuzz/stress testing. For each encoded
/// chunk it picks uniformly at random among the encodings that are compatible
/// with the data. Compatibility is defined exactly as encoding-size
/// estimability:
/// EncodingSizeEstimation returns nullopt for any encoding that cannot encode
/// the given physical type or data statistics (e.g. Dictionary/FixedBitWidth/
/// Varint on bool, Varint on non-integers, Constant on non-constant data), so
/// reusing that signal keeps the random pick compatible without a
/// hand-maintained rule table, and the writer's one-shot IncompatibleEncoding
/// fallback is never relied upon.
template <typename T>
class RandomEncodingSelectionPolicy : public EncodingSelectionPolicy<T> {
  using physicalType = typename TypeTraits<T>::physicalType;

 public:
  RandomEncodingSelectionPolicy(
      uint64_t seed,
      std::vector<EncodingType> candidateEncodingTypes,
      std::optional<CompressionOptions> compressionOptions =
          CompressionOptions{})
      : seed_{seed},
        candidateEncodingTypes_{std::move(candidateEncodingTypes)},
        compressionOptions_{std::move(compressionOptions)} {}

  EncodingSelectionResult select(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) override {
    return select(values, statistics, values.size(), options).value();
  }

  /// Selects among candidates that can estimate the target row count.
  std::optional<EncodingSelectionResult> select(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      size_t numValues,
      const Encoding::Options& options) override {
    NIMBLE_CHECK_LE(values.size(), numValues);
    if (values.empty()) {
      NIMBLE_CHECK_EQ(numValues, 0);
      return EncodingSelectionResult{
          .encodingType = EncodingType::Trivial,
          .estimatedSize = 0,
      };
    }

    // Keep the selected size available to parent estimators.
    std::vector<EncodingSelectionResult> compatibleEncodings;
    compatibleEncodings.reserve(candidateEncodingTypes_.size());
    for (const auto& encodingType : candidateEncodingTypes_) {
      const auto size =
          estimateSize(encodingType, values, statistics, numValues, options);
      if (size.has_value()) {
        compatibleEncodings.push_back(
            EncodingSelectionResult{
                .encodingType = encodingType,
                .estimatedSize = size,
            });
      }
    }

    if (compatibleEncodings.empty()) {
      // A fallback size lets parent estimators account for streams whose
      // configured candidates cannot estimate the target row count.
      const auto size = estimateSize(
          EncodingType::Trivial, values, statistics, numValues, options);
      if (values.size() != numValues && !size.has_value()) {
        return std::nullopt;
      }
      return EncodingSelectionResult{
          .encodingType = EncodingType::Trivial,
          .estimatedSize = size,
      };
    }

    const auto selectionSeed = folly::hash::hash_combine(
        seed_, folly::hash::hash_range(values.begin(), values.end()));
    std::mt19937_64 generator{selectionSeed};
    std::uniform_int_distribution<size_t> distribution(
        0, compatibleEncodings.size() - 1);
    auto selected = compatibleEncodings[distribution(generator)];

    if (compressionOptions_.has_value()) {
      selected.compressionPolicyFactory =
          [compressionOptions = compressionOptions_.value(),
           selectedEncoding = selected.encodingType]() {
            return std::make_unique<ConfiguredCompressionPolicy>(
                compressionOptions, selectedEncoding);
          };
    }
    return selected;
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
    return std::any_of(
        candidateEncodingTypes_.begin(),
        candidateEncodingTypes_.end(),
        [](EncodingType type) {
          return type == EncodingType::ALP || type == EncodingType::ALPRD;
        });
  }

 protected:
  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType parentEncodingType,
      NestedEncodingIdentifier nestedEncodingIdentifier,
      DataType nestedDataType) override {
    // Exclude the parent encoding from nested choices so the recursive
    // selection always converges (mirrors Manual/Learned).
    std::vector<EncodingType> nestedCandidateEncodingTypes;
    nestedCandidateEncodingTypes.reserve(candidateEncodingTypes_.size());
    for (const auto& encodingType : candidateEncodingTypes_) {
      if (encodingType != parentEncodingType) {
        nestedCandidateEncodingTypes.push_back(encodingType);
      }
    }
    // Fold this policy's seed with the parent encoding and the child slot so
    // every node in the tree gets a distinct seed that depends only on its
    // structural path, never on encode timing. This keeps the whole random
    // layout reproducible from the single base seed even when streams are
    // encoded concurrently.
    const uint64_t nestedSeed = folly::hash::hash_combine(
        seed_,
        static_cast<std::underlying_type_t<EncodingType>>(parentEncodingType),
        nestedEncodingIdentifier);
    UNIQUE_PTR_FACTORY(
        nestedDataType,
        RandomEncodingSelectionPolicy,
        nestedSeed,
        std::move(nestedCandidateEncodingTypes),
        compressionOptions_);
  }

 private:
  // Only encodings with statistics-based estimates support projection.
  std::optional<uint64_t> estimateSize(
      EncodingType encodingType,
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      size_t numValues,
      const Encoding::Options& options) {
    using Estimator = detail::EncodingSizeEstimation<T>;
    if (values.size() == numValues) {
      return Estimator::estimateSize(
          *this, encodingType, values, statistics, options);
    }
    if (!detail::canProjectSizeEstimation<physicalType>(encodingType)) {
      return std::nullopt;
    }
    return Estimator::estimateSize(
        encodingType, numValues, statistics, options);
  }

  const uint64_t seed_;
  const std::vector<EncodingType> candidateEncodingTypes_;
  const std::optional<CompressionOptions> compressionOptions_;
};

/// Produces RandomEncodingSelectionPolicy instances seeded deterministically
/// from a single base seed. Each selection additionally derives its seed from
/// the input values, keeping the result reproducible independently of policy
/// creation and encode thread order. Intended for fuzz/stress testing only.
class RandomEncodingSelectionPolicyFactory {
 public:
  /// The candidate encodings the random policy draws from (the production
  /// Learned/Manual default set). EncodingSizeEstimation filters this per
  /// selection down to the encodings compatible with the actual data.
  static std::vector<EncodingType> defaultEncodingChoices();

  /// Builds a factory from a nimble.encoding_selection_config string of the
  /// form "type:random,seed:<n>[,encodings:<E1>;<E2>;...]". 'seed' is required;
  /// 'encodings' (optional) restricts the candidate set. Ignores the 'type' key
  /// (already used by createEncodingSelectionPolicyFactory). Reports bad input
  /// as a NimbleUserError.
  static RandomEncodingSelectionPolicyFactory create(
      std::string_view configStr,
      std::optional<CompressionOptions> compressionOptions =
          CompressionOptions{});

  explicit RandomEncodingSelectionPolicyFactory(
      uint64_t seed,
      std::vector<EncodingType> candidateEncodingTypes =
          defaultEncodingChoices(),
      std::optional<CompressionOptions> compressionOptions =
          CompressionOptions{});

  std::unique_ptr<EncodingSelectionPolicyBase> createPolicy(
      DataType dataType) const;

 private:
  const uint64_t seed_;
  const std::vector<EncodingType> candidateEncodingTypes_;
  const std::optional<CompressionOptions> compressionOptions_;
};

} // namespace facebook::nimble::testing
