/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include <cstdint>
#include "velox/functions/lib/aggregates/noisy_aggregation/sketch/BitMap.h"
#include "velox/functions/lib/aggregates/noisy_aggregation/sketch/SecureRandomizationStrategy.h"

namespace facebook::velox::functions::aggregate {

/// For technical details of SfmSketch, please refer to the paper
/// <a href="https://arxiv.org/pdf/2302.02056.pdf">Sketch-Flip-Merge: Mergeable
/// Sketches for Private Distinct Counting</a>.
class SfmSketch {
 public:
  static constexpr double NON_PRIVATE_EPSILON =
      std::numeric_limits<double>::infinity();
  static constexpr int32_t MAX_ESTIMATION_ITERATIONS = 1000;
  // Create a non-private sketch.
  static SfmSketch create(uint32_t numberOfBuckets, uint32_t precision);
  static uint32_t numberOfTrailingZeros(uint64_t hash, uint32_t indexBitLength);

  void add(int64_t value);
  void addHash(uint64_t hash);
  void addIndexAndZeros(uint32_t bucketIndex, uint32_t zeros);
  void mergeWith(SfmSketch& other);
  uint64_t cardinality() const;
  size_t serializedSize() const;
  void serialize(char* out) const;
  static SfmSketch deserialize(const char* in);

  template <typename RandomizationStrategy>
  void mergeWith(
      SfmSketch& other,
      RandomizationStrategy randomizationStrategy) {
    VELOX_CHECK_EQ(
        precision_,
        other.precision_,
        "Cannot merge two SFM sketches with different precision: {} vs. {}",
        precision_,
        other.precision_);
    VELOX_CHECK_EQ(
        indexBitLength_,
        other.indexBitLength_,
        "Cannot merge two SFM sketches with different indexBitLength: {} vs. {}",
        indexBitLength_,
        other.indexBitLength_);

    if (!privacyEnabled() && !other.privacyEnabled()) {
      // If neither sketch is private, we just take the OR of the sketches
      bitMap_.bitwiseOr(other.getBitMap());
    } else {
      // If either sketch is private, we combine using a randomized merge
      double p1 = randomizedResponseProbability_;
      double p2 = other.randomizedResponseProbability_;
      double p = mergeRandomizedResponseProbabilities(p1, p2);
      double normalizer = (1 - 2 * p) / ((1 - 2 * p1) * (1 - 2 * p2));

      for (uint32_t i = 0; i < bitMap_.length(); i++) {
        double bit1 = bitMap_.getBit(i) ? 1.0 : 0.0;
        double bit2 = other.bitMap_.getBit(i) ? 1.0 : 0.0;
        double x = 1 - 2 * p - normalizer * (1 - p1 - bit1) * (1 - p2 - bit2);
        double probability = p + normalizer * x;
        probability = std::min(1.0, std::max(0.0, probability));
        bitMap_.setBit(i, randomizationStrategy.nextBoolean(probability));
      }
    }

    randomizedResponseProbability_ = mergeRandomizedResponseProbabilities(
        randomizedResponseProbability_, other.randomizedResponseProbability_);
  }

  void enablePrivacy(double epsilon) {
    enablePrivacy(epsilon, getDefaultRandomizationStrategy());
  }

  static SecureRandomizationStrategy getDefaultRandomizationStrategy() {
    return SecureRandomizationStrategy();
  }

  template <typename RandomizationStrategy>
  void enablePrivacy(
      double epsilon,
      RandomizationStrategy randomizationStrategy) {
    VELOX_CHECK_NOT_NULL(
        &randomizationStrategy, "randomizationStrategy can't be null.");
    VELOX_CHECK(!privacyEnabled(), "privacy is already enabled.");
    validateEpsilon(epsilon);
    randomizedResponseProbability_ =
        calculateRandomizedResponseProbability(epsilon);
    bitMap_.flipAll(randomizedResponseProbability_, randomizationStrategy);
  }

  // For math details, see Theorem 4.8, <a
  // href="https://arxiv.org/pdf/2302.02056.pdf">arXiv:2302.02056</a>.
  static double mergeRandomizedResponseProbabilities(double p1, double p2) {
    return (p1 + p2 - 3 * p1 * p2) / (1 - 2 * p1 * p2);
  }

  static uint32_t computeIndex(uint64_t hash, uint32_t indexBitLength) {
    VELOX_CHECK(indexBitLength < 32, "indexBitLength must be less than 32.");
    constexpr uint32_t kBitWidth = sizeof(uint64_t) * 8;
    return static_cast<uint32_t>(hash >> (kBitWidth - indexBitLength));
  }

  static uint32_t numberOfBuckets(uint32_t indexBitLength) {
    VELOX_CHECK(indexBitLength < 32, "indexBitLength must be less than 32.");
    return 1 << indexBitLength;
  }

  static uint32_t indexBitLength(uint32_t numberOfBuckets) {
    VELOX_CHECK(
        numberOfBuckets > 0 && (numberOfBuckets & (numberOfBuckets - 1)) == 0,
        "numberOfBuckets must be a power of 2.");
    return std::log2<uint32_t>(numberOfBuckets);
  }

  double getOnProbability() const {
    // Probability of a 1-bit remaining a 1-bit under randomized response
    return 1 - randomizedResponseProbability_;
  }

  double getRandomizedResponseProbability() const {
    return randomizedResponseProbability_;
  }

  static double calculateRandomizedResponseProbability(double epsilon) {
    if (epsilon == NON_PRIVATE_EPSILON) {
      return 0.0;
    }
    return 1.0 / (1.0 + exp(epsilon));
  }

  uint32_t getIndexBitLength() const {
    return indexBitLength_;
  }

  uint32_t getPrecision() const {
    return precision_;
  }

  const BitMap& getBitMap() const {
    return bitMap_;
  }

  BitMap& getMutableBitMap() {
    return bitMap_;
  }

  bool privacyEnabled() const {
    return randomizedResponseProbability_ > 0;
  }

 private:
  SfmSketch(
      BitMap& bitMap,
      uint32_t indexBitLength,
      uint32_t precision,
      double randomizedResponseProbability);

  static void validateEpsilon(double epsilon);
  static void validatePrecision(uint32_t precision, uint32_t indexBitLength);
  static void validatePrefixLength(uint32_t indexBitLength);
  static void validateRandomizedResponseProbability(double p);
  double observationProbability(uint32_t level) const;
  void flipBitOn(uint32_t bucketIndex, uint32_t zeros);
  double logLikelihoodFirstDerivative(double n) const;
  double logLikelihoodTermFirstDerivative(uint32_t level, bool on, double n)
      const;
  double logLikelihoodSecondDerivative(double n) const;
  double logLikelihoodTermSecondDerivative(uint32_t level, bool on, double n)
      const;

  const uint32_t indexBitLength_;
  const uint32_t precision_;
  double randomizedResponseProbability_;
  BitMap bitMap_;
};
} // namespace facebook::velox::functions::aggregate
