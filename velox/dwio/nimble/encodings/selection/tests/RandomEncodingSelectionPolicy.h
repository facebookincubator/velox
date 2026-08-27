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

/// Randomized encoding selection for fuzz/stress testing. For each stream it
/// picks uniformly at random among the encodings that are compatible with the
/// data. Compatibility is defined exactly as encoding-size estimability:
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
    // Empty streams (e.g. the values of an all-null column, or an empty nested
    // stream) can only be encoded trivially; several encodings hard-fail on
    // zero rows. Matches Manual/Learned selection.
    if (values.empty()) {
      return {
          .encodingType = EncodingType::Trivial,
          .encodingConfig = {},
          .estimatedSize = std::nullopt,
      };
    }

    // Keep only encodings that can estimate a size for this data. An absent
    // estimate means the encoding is incompatible with the physical type or the
    // data's statistics, so excluding it here is what keeps the random pick
    // compatible.
    std::vector<EncodingType> compatibleEncodings;
    compatibleEncodings.reserve(candidateEncodingTypes_.size());
    for (const auto& encodingType : candidateEncodingTypes_) {
      if (detail::EncodingSizeEstimation<T>::estimateSize(
              encodingType, values, statistics, options)
              .has_value()) {
        compatibleEncodings.push_back(encodingType);
      }
    }

    if (compatibleEncodings.empty()) {
      return {
          .encodingType = EncodingType::Trivial,
          .encodingConfig = {},
          .estimatedSize = std::nullopt,
      };
    }

    // Seed a fresh generator from this policy's derived seed so the single pick
    // is deterministic and independent of encode thread order.
    std::mt19937_64 generator{seed_};
    std::uniform_int_distribution<size_t> distribution(
        0, compatibleEncodings.size() - 1);
    const auto selectedEncoding = compatibleEncodings[distribution(generator)];

    if (!compressionOptions_.has_value()) {
      return {
          .encodingType = selectedEncoding,
          .encodingConfig = {},
          .estimatedSize = std::nullopt,
      };
    }
    // Encoding selection optimizes the in-memory layout. Compression is still
    // attempted for leaf data streams to reduce persistent storage size.
    return {
        .encodingType = selectedEncoding,
        .encodingConfig = {},
        .estimatedSize = std::nullopt,
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
  const uint64_t seed_;
  const std::vector<EncodingType> candidateEncodingTypes_;
  const std::optional<CompressionOptions> compressionOptions_;
};

/// Produces RandomEncodingSelectionPolicy instances seeded deterministically
/// from a single base seed, so an entire file's random encoding tree is
/// reproducible from that seed. Intended for fuzz/stress testing only.
class RandomEncodingSelectionPolicyFactory {
 public:
  /// The candidate encodings the random policy draws from (the production
  /// Learned/Manual default set). EncodingSizeEstimation filters this per
  /// stream down to the encodings compatible with the actual data.
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
