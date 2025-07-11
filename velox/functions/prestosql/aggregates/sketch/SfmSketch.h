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

#include <folly/Range.h>
#include <xxhash.h>
#include <cstdint>
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/functions/prestosql/aggregates/sketch/RandomizationStrategy.h"

namespace facebook::velox::functions::aggregate {

/// For technical details of SfmSketch, please refer to the paper
/// <a href="https://arxiv.org/pdf/2302.02056.pdf">Sketch-Flip-Merge: Mergeable
/// Sketches for Private Distinct Counting</a>.
/// A typical work flow is: initialize -> add -> merge -> enable-privacy ->
/// cardinality.
class SfmSketch {
 public:
  /// Creates an uninitialized instance. The caller must call initialize()
  /// before using this instance.
  explicit SfmSketch(HashStringAllocator* allocator);

  /// Set the sketch size. The size is determined by the number of buckets and
  /// the precision. This function should be called before any other functions.
  /// @param buckets The number of buckets, should be a power of 2.
  /// @param precision The number of bits to count trailing zeros in a hash,
  /// precision should be less than 64 - indexBitLength, indexBitLength can
  /// be obtained by calling indexBitLength().
  void initialize(int32_t buckets, int32_t precision);

  /// @tparam T StringView, bigint, or double.
  template <typename T>
  void add(T value) {
    addHash(XXH64(
        reinterpret_cast<const void*>(&value),
        sizeof(value),
        /*seed*/ 0));
  }

  /// Another way to add an element to the sketch. Explicitly toggle the bit at
  /// 'bucketIndex' and 'zeros' position.
  void addIndexAndZeros(int32_t bucketIndex, int32_t zeros);

  /// Compute the bit index with the given hash and indexBitLength.
  static int32_t computeIndex(uint64_t hash, int32_t indexBitLength);

  /// Compute the number of buckets given the indexBitLength.
  static int32_t numBuckets(int32_t indexBitLength);

  /// Compute the number of bits needed to represent the index of a bucket.
  /// range of indexBitLength is [1, 16].
  static int32_t indexBitLength(int32_t buckets);

  /// Merge another sketch into the current sketch. This function is called
  /// after adding all the elements to the sketch to add noise to the sketch.
  void mergeWith(const SfmSketch& other);

  /// Make a non-private sketch private.
  /// @param epsilon The privacy parameter, the larger the epsilon, the less
  /// noise will be added.
  void enablePrivacy(double epsilon);

  /// Merge another sketch into the current sketch with a randomization
  /// strategy.
  void mergeWith(
      const SfmSketch& other,
      RandomizationStrategy& randomizationStrategy);

  /// Make a non-private sketch private with a randomization strategy.
  void enablePrivacy(
      double epsilon,
      RandomizationStrategy& randomizationStrategy);

  /// Estimate the number of distinct elements added to the sketch via maximum
  /// pseudolikelihood (Newton's method).
  int64_t cardinality() const;

  /// Return the size of the sketch in bytes.
  int32_t serializedSize() const;

  /// Serialize the sketch into a char array.
  /// The Java serialization format is used in
  /// production, so we need to ensure that the C++ implementation can read and
  /// write the same format.
  /// The Java serialization format is as follows:
  /// 1. 1 bytes for the FORMAT_TAG = 7, constant.
  /// 2. 4 bytes for the indexBitLength
  /// 3. 4 bytes for the precision
  /// 4. 8 bytes for the randomized response probability
  /// 5. 4 bytes for the actual number of bytes in the serialized bitmap
  /// 6. The bitmap data
  void serialize(char* out) const;

  /// Deserialize the sketch from a pre-allocated buffer of at least
  /// 'serializedSize' bytes.
  static SfmSketch deserialize(const char* in, HashStringAllocator* allocator);

  int32_t numberOfBits() const {
    return static_cast<int32_t>(bits_.size() * 8);
  }

  double randomizedResponseProbability() const {
    return randomizedResponseProbability_;
  }

  int32_t indexBitLength() const {
    return indexBitLength_;
  }

  int32_t precision() const {
    return precision_;
  }

  folly::Range<const int8_t*> bits() const {
    return folly::Range<const int8_t*>(bits_.data(), bits_.size());
  }

  bool privacyEnabled() const {
    return randomizedResponseProbability_ > 0;
  }

 private:
  // Epsilon for non-private sketch.
  static constexpr double kNonPrivateEpsilon =
      std::numeric_limits<double>::infinity();

  // Maximum number of iterations for Newton's method.
  static constexpr int32_t kMaxIteration = 1000;

  // Java implementation use a format_tag to identify the sketch.
  // We need this tag to be able to deserialize the sketch from Java.
  static constexpr int8_t kFormatTag = 7;

  // Calculate the RandomizedResponseProbabilities for merged sketches.
  // For math details, see Theorem 4.8,
  // <a href="https://arxiv.org/pdf/2302.02056.pdf">arXiv:2302.02056</a>.
  static double mergeRandomizedResponseProbabilities(double p1, double p2) {
    return (p1 + p2 - 3 * p1 * p2) / (1 - 2 * p1 * p2);
  }

  // Calculate the randomized response probability given epsilon.
  static double calculateRandomizedResponseProbability(double epsilon);

  // Add a hash to the sketch.
  void addHash(uint64_t hash);

  // Probability of a 1-bit remaining a 1-bit under randomized response.
  double onProbability() const {
    return 1 - randomizedResponseProbability_;
  }

  // Get the number of bits that are set to 1.
  int32_t countBits() const;

  // Return the size of the bitset in bytes after dropping the trailing zeros.
  int32_t compactBitSize() const;

  // Probability of observing a run of zeros of length level in any single
  // bucket
  double observationProbability(int32_t level) const;

  // Set the bit at given bucketIndex and zeros position.
  void setBitTrue(int32_t bucketIndex, int32_t zeros);

  // Recast the bits to a uint64_t pointer. This function should be called
  // in initialize() and deserialize().
  void recomputeRawBits();

  // Helper functions for cardinality estimation using Newton's method.
  double logLikelihoodFirstDerivative(double n) const;
  double logLikelihoodTermFirstDerivative(int32_t level, bool on, double n)
      const;
  double logLikelihoodSecondDerivative(double n) const;
  double logLikelihoodTermSecondDerivative(int32_t level, bool on, double n)
      const;

  // Number of buckets in the sketch.
  int32_t numBuckets_{0};

  // Number of bits to represent the index of a bucket.
  int32_t indexBitLength_{0};

  // Number of bits to count trailing zeros in a hash value.
  int32_t precision_{0};

  // Probability of a bit being flipped.
  double randomizedResponseProbability_{0.0};

  // The underline bit representation of the sketch.
  std::vector<int8_t, facebook::velox::StlAllocator<int8_t>> bits_;

  // Pointer to the raw bits.
  const uint64_t* rawBits_{nullptr};
};
} // namespace facebook::velox::functions::aggregate
