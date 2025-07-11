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

#include "velox/functions/prestosql/aggregates/sketch/SfmSketch.h"
#include <cmath>
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/IOUtils.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/functions/prestosql/aggregates/sketch/MersenneTwisterRandomizationStrategy.h"
#include "velox/functions/prestosql/aggregates/sketch/SecureRandomizationStrategy.h"

namespace facebook::velox::functions::aggregate {

const int8_t SfmSketch::kFormatTag;

namespace {
void validateIndexBitLength(int32_t indexBitLength) {
  VELOX_CHECK_LE(indexBitLength, 16, "indexBitLength must be <= 16.");
  VELOX_CHECK_GE(indexBitLength, 1, "indexBitLength must be >= 1.");
}

int32_t numberOfTrailingZeros(uint64_t hash, int32_t indexBitLength) {
  // Set the lowest bit in the prefix to ensure value is non-zero
  constexpr int32_t kBitWidth = sizeof(uint64_t) * 8;
  uint64_t value = hash | (1ULL << (kBitWidth - indexBitLength));

  // Safe because value is guaranteed non-zero
  return __builtin_ctzll(value);
}
} // namespace

using Allocator = StlAllocator<int8_t>;

SfmSketch::SfmSketch(HashStringAllocator* allocator)
    : bits_(Allocator(allocator)) {}

void SfmSketch::initialize(int32_t buckets, int32_t precision) {
  auto indexBitLen = indexBitLength(buckets);
  validateIndexBitLength(indexBitLen);

  VELOX_CHECK_LE(
      precision + indexBitLen,
      64,
      "Precision + indexBitLength cannot exceed 64");
  numBuckets_ = buckets;
  indexBitLength_ = indexBitLen;
  precision_ = precision;
  bits_.resize(bits::nbytes(buckets * precision));
  recomputeRawBits();
}

void SfmSketch::addHash(uint64_t hash) {
  auto bucketIndex = computeIndex(hash, indexBitLength_);
  // Cap the number of trailing zeros to precision - 1, to avoid out of
  // bounds.
  auto zeros =
      std::min(precision_ - 1, numberOfTrailingZeros(hash, indexBitLength_));
  setBitTrue(bucketIndex, zeros);
}

void SfmSketch::addIndexAndZeros(int32_t bucketIndex, int32_t zeros) {
  VELOX_CHECK_GE(bucketIndex, 0, "Bucket index out of range.");
  VELOX_CHECK_LT(bucketIndex, numBuckets_, "Bucket index out of range.");
  VELOX_CHECK_GE(zeros, 0, "Zeros must be greater than or equal to 0.");
  VELOX_CHECK_LE(zeros, 64, "Zeros must be less than or equal to 64.");

  // count of zeros in range [0, precision - 1]
  zeros = std::min(precision_ - 1, zeros);
  setBitTrue(bucketIndex, zeros);
}

// static
int32_t SfmSketch::computeIndex(uint64_t hash, int32_t indexBitLength) {
  validateIndexBitLength(indexBitLength);
  constexpr int32_t kBitWidth = sizeof(uint64_t) * 8;
  return static_cast<int32_t>(hash >> (kBitWidth - indexBitLength));
}

// static
int32_t SfmSketch::numBuckets(int32_t indexBitLength) {
  validateIndexBitLength(indexBitLength);
  return 1 << indexBitLength;
}

// static
int32_t SfmSketch::indexBitLength(int32_t buckets) {
  VELOX_CHECK_GT(buckets, 0, "Number of buckets must be greater than 0.");
  VELOX_CHECK_EQ(
      buckets & (buckets - 1), 0, "Number of buckets must be power of 2.");
  return static_cast<int32_t>(std::log2(buckets));
}

// static
double SfmSketch::calculateRandomizedResponseProbability(double epsilon) {
  if (epsilon == kNonPrivateEpsilon) {
    return 0.0;
  }
  return 1.0 / (1.0 + exp(epsilon));
}

void SfmSketch::mergeWith(const SfmSketch& other) {
  MersenneTwisterRandomizationStrategy randomizationStrategy;
  mergeWith(other, randomizationStrategy);
}

void SfmSketch::mergeWith(
    const SfmSketch& other,
    RandomizationStrategy& randomizationStrategy) {
  VELOX_CHECK_EQ(
      precision_,
      other.precision_,
      "Cannot merge two SFM sketches with different precision");
  VELOX_CHECK_EQ(
      indexBitLength_,
      other.indexBitLength_,
      "Cannot merge two SFM sketches with different indexBitLength");

  auto numBits = numberOfBits();
  // If neither sketch is private, we just take the OR of the sketches.
  if (!privacyEnabled() && !other.privacyEnabled()) {
    bits::orBits(const_cast<uint64_t*>(rawBits_), other.rawBits_, 0, numBits);
  } else {
    // If either sketch is private, we combine using a randomized merge
    const double p1 = randomizedResponseProbability_;
    const double p2 = other.randomizedResponseProbability_;
    double p = mergeRandomizedResponseProbabilities(p1, p2);
    double normalizer = (1 - 2 * p) / ((1 - 2 * p1) * (1 - 2 * p2));

    for (int32_t i = 0; i < numBits; i++) {
      double bit1 = bits::isBitSet(rawBits_, i) ? 1.0 : 0.0;
      double bit2 = bits::isBitSet(other.rawBits_, i) ? 1.0 : 0.0;
      double x = 1 - 2 * p - normalizer * (1 - p1 - bit1) * (1 - p2 - bit2);
      double probability = p + normalizer * x;
      probability = std::min(1.0, std::max(0.0, probability));
      bits::setBit(
          const_cast<uint64_t*>(rawBits_),
          i,
          randomizationStrategy.nextBoolean(probability));
    }
  }

  randomizedResponseProbability_ = mergeRandomizedResponseProbabilities(
      randomizedResponseProbability_, other.randomizedResponseProbability_);
}

void SfmSketch::enablePrivacy(double epsilon) {
  SecureRandomizationStrategy randomizationStrategy;
  enablePrivacy(epsilon, randomizationStrategy);
}

void SfmSketch::enablePrivacy(
    double epsilon,
    RandomizationStrategy& randomizationStrategy) {
  VELOX_CHECK_NOT_NULL(
      &randomizationStrategy, "randomizationStrategy can't be null.");
  VELOX_CHECK(!privacyEnabled(), "privacy is already enabled.");
  VELOX_CHECK_GT(
      epsilon, 0, "Epsilon must be greater than zero or equal to infinity");
  randomizedResponseProbability_ =
      calculateRandomizedResponseProbability(epsilon);

  // Toggle each bit with a securely generated probability.
  for (int32_t i = 0; i < numberOfBits(); ++i) {
    if (randomizationStrategy.nextBoolean(randomizedResponseProbability_)) {
      bits::negateBit(bits_.data(), i);
    }
  }
}

int32_t SfmSketch::countBits() const {
  return bits::countBits(rawBits_, 0, numberOfBits());
}

int64_t SfmSketch::cardinality() const {
  // Handle empty sketch case.
  if (countBits() == 0) {
    return 0;
  }

  double guess = 1.0;
  double changeInGuess = std::numeric_limits<double>::infinity();
  int32_t iterations = 0;

  while (std::abs(changeInGuess) > 0.1 && iterations < kMaxIteration) {
    changeInGuess = -logLikelihoodFirstDerivative(guess) /
        logLikelihoodSecondDerivative(guess);
    guess += changeInGuess;
    iterations++;
  }

  // Handle NaN values that can occur in Newton's method iterations,
  // especially with privacy-enabled sketches where randomized response
  // can create unstable mathematical conditions
  if (std::isnan(guess)) {
    return 0;
  }

  VELOX_CHECK_LE(guess, std::numeric_limits<uint64_t>::max());
  // Clamp negative values to 0 before rounding to avoid undefined behavior
  // when casting negative values to unsigned types
  double clampedGuess = std::max(0.0, guess);
  return static_cast<int64_t>(std::round(clampedGuess));
}

// Java-compatible format: FORMAT_TAG + indexBitLength + precision +
// randomizedResponseProbability + bitsetByteLength + bitset_data
int32_t SfmSketch::serializedSize() const {
  return static_cast<int32_t>(
      sizeof(int8_t) + sizeof(int32_t) + sizeof(int32_t) + sizeof(double) +
      sizeof(int32_t) + compactBitSize());
}

void SfmSketch::serialize(char* out) const {
  common::OutputByteStream stream(out);
  stream.appendOne(kFormatTag);
  stream.appendOne(indexBitLength_);
  stream.appendOne(precision_);
  stream.appendOne(randomizedResponseProbability_);
  auto compactBitSize = this->compactBitSize();
  stream.appendOne(compactBitSize);

  // Only append data if the vector is not empty to avoid null pointer issues
  if (compactBitSize > 0) {
    stream.append(reinterpret_cast<const char*>(bits_.data()), compactBitSize);
  }
}

SfmSketch SfmSketch::deserialize(
    const char* in,
    HashStringAllocator* allocator) {
  common::InputByteStream stream(in);
  const auto formatTag = stream.read<int8_t>(); // FORMAT_TAG
  auto indexBitLength = stream.read<int32_t>();
  auto precision = stream.read<int32_t>();
  auto randomizedResponseProbability = stream.read<double>();
  auto bitSetBytesLength = stream.read<int32_t>();

  // Validate format tag
  VELOX_CHECK_EQ(formatTag, kFormatTag, "Invalid format tag");

  // Validate indexBitLength before calling numBuckets
  validateIndexBitLength(indexBitLength);

  // Create the sketch.
  SfmSketch sketch(allocator);
  auto buckets = numBuckets(indexBitLength);
  sketch.initialize(buckets, precision);

  // Set the randomized response probability
  sketch.randomizedResponseProbability_ = randomizedResponseProbability;

  // Read the serialized bitmap data directly into the sketch's bitset
  // The sketch's bitset is already the correct size and zero-initialized
  // We just need to copy the serialized data (which may be truncated)
  stream.copyTo(sketch.bits_.data(), bitSetBytesLength);

  sketch.recomputeRawBits();
  return sketch;
}

double SfmSketch::observationProbability(int32_t level) const {
  return std::pow(2.0, -(level + 1.0)) / numBuckets_;
}

int32_t SfmSketch::compactBitSize() const {
  // Find the last non-zero byte.
  int32_t lastSetBit = bits::findLastBit(rawBits_, 0, numberOfBits());

  if (lastSetBit < 0) {
    // No bits are set
    return 0;
  }
  return bits::nbytes(lastSetBit + 1);
}

void SfmSketch::recomputeRawBits() {
  rawBits_ = reinterpret_cast<const uint64_t*>(bits_.data());
}

double SfmSketch::logLikelihoodFirstDerivative(double n) const {
  double result = 0.0;
  for (int32_t level = 0; level < precision_; level++) {
    double termOn = logLikelihoodTermFirstDerivative(level, true, n);
    double termOff = logLikelihoodTermFirstDerivative(level, false, n);
    for (int32_t bucket = 0; bucket < numBuckets_; bucket++) {
      int32_t bitPosition = level * numBuckets_ + bucket;
      result += bits::isBitSet(bits_.data(), bitPosition) ? termOn : termOff;
    }
  }
  return result;
}

double SfmSketch::logLikelihoodTermFirstDerivative(
    int32_t level,
    bool on,
    double n) const {
  double p = observationProbability(level);
  int32_t sign = on ? -1 : 1;
  double c1 = on ? onProbability() : 1 - onProbability();
  double c2 = onProbability() - randomizedResponseProbability();
  return std::log1p(-p) * (1 - c1 / (c1 + sign * c2 * std::pow(1 - p, n)));
}

double SfmSketch::logLikelihoodSecondDerivative(double n) const {
  double result = 0.0;
  for (int32_t level = 0; level < precision_; level++) {
    double termOn = logLikelihoodTermSecondDerivative(level, true, n);
    double termOff = logLikelihoodTermSecondDerivative(level, false, n);
    for (int32_t bucket = 0; bucket < numBuckets_; bucket++) {
      int32_t bitPosition = level * numBuckets_ + bucket;
      result += bits::isBitSet(bits_.data(), bitPosition) ? termOn : termOff;
    }
  }
  return result;
}

double SfmSketch::logLikelihoodTermSecondDerivative(
    int32_t level,
    bool on,
    double n) const {
  double p = observationProbability(level);
  int32_t sign = on ? -1 : 1;
  double c1 = on ? onProbability() : 1 - onProbability();
  double c2 = onProbability() - randomizedResponseProbability();
  return sign * c1 * c2 * std::pow(std::log1p(-p), 2) * std::pow(1 - p, n) *
      std::pow(c1 + sign * c2 * std::pow(1 - p, n), -2);
}

void SfmSketch::setBitTrue(int32_t bucketIndex, int32_t zeros) {
  VELOX_CHECK(!privacyEnabled(), "private sketch is immutable.");
  // It's more likely to have a hash with less zeros than more zeros.
  // we keep the bitmap in the form of a bit matrix, where each row is a zero
  // level instead of a bucket.
  // In this way, we can save space by dropping the trailing zeros in
  // serialization.
  auto bitPosition = zeros * numBuckets_ + bucketIndex;
  bits::setBit(bits_.data(), bitPosition, true);
}

} // namespace facebook::velox::functions::aggregate
