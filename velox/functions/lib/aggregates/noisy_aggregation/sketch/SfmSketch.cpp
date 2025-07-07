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

#include "velox/functions/lib/aggregates/noisy_aggregation/sketch/SfmSketch.h"
#include <cmath>
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/IOUtils.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/functions/lib/aggregates/noisy_aggregation/sketch/MersenneTwisterRandomizationStrategy.h"

namespace facebook::velox::functions::aggregate {
const int8_t kFormatTag = 7;

using Allocator = facebook::velox::StlAllocator<int8_t>;

SfmSketch::SfmSketch(HashStringAllocator* allocator)
    : bits_(Allocator(allocator)) {}

void SfmSketch::setSketchSize(uint32_t numberOfBuckets, uint32_t precision) {
  validatePrefixLength(indexBitLength(numberOfBuckets));
  indexBitLength_ = indexBitLength(numberOfBuckets);
  validatePrecision(precision, indexBitLength_);
  precision_ = precision;
  bits_.resize(velox::bits::nbytes(numberOfBuckets * precision_));
}

uint32_t SfmSketch::numberOfTrailingZeros(
    uint64_t hash,
    uint32_t indexBitLength) {
  // Set the lowest bit in the prefix to ensure value is non-zero
  constexpr uint32_t kBitWidth = sizeof(uint64_t) * 8;
  uint64_t value = hash | (1ULL << (kBitWidth - indexBitLength));

  // Safe because value is guaranteed non-zero
  return static_cast<uint32_t>(__builtin_ctzll(value));
}

void SfmSketch::addHash(uint64_t hash) {
  auto bucketIndex = computeIndex(hash, indexBitLength_);
  // Cap the number of trailing zeros to precision - 1, to avoid out of
  // bound.
  auto zeros =
      std::min(precision_ - 1, numberOfTrailingZeros(hash, indexBitLength_));
  setBitTrue(bucketIndex, zeros);
}

void SfmSketch::addIndexAndZeros(uint32_t bucketIndex, uint32_t zeros) {
  auto numOfbuckets = numberOfBuckets(indexBitLength_);
  VELOX_CHECK(
      bucketIndex >= 0 && bucketIndex < numOfbuckets,
      "Bucket index {} must be between zero (inclusive) and the number of buckets-1 {}",
      bucketIndex,
      numOfbuckets - 1);
  VELOX_CHECK(
      zeros >= 0 && zeros <= 64, "Zeros {} must be between 0 and 64", zeros);

  // count of zeros in range [0, precision - 1]
  zeros = std::min(precision_ - 1, zeros);
  setBitTrue(bucketIndex, zeros);
}

uint32_t SfmSketch::computeIndex(uint64_t hash, uint32_t indexBitLength) {
  VELOX_CHECK(
      indexBitLength <= kMaxIndexBitLength, "indexBitLength must be <= 16.");
  constexpr uint32_t kBitWidth = sizeof(uint64_t) * 8;
  return static_cast<uint32_t>(hash >> (kBitWidth - indexBitLength));
}

uint32_t SfmSketch::numberOfBuckets(uint32_t indexBitLength) {
  VELOX_CHECK(
      indexBitLength <= kMaxIndexBitLength, "indexBitLength must be <= 16.");
  return 1 << indexBitLength;
}

uint32_t SfmSketch::indexBitLength(uint32_t numberOfBuckets) {
  VELOX_CHECK(
      numberOfBuckets > 0 && (numberOfBuckets & (numberOfBuckets - 1)) == 0,
      "numberOfBuckets must be a power of 2.");
  return static_cast<uint32_t>(std::log2(numberOfBuckets));
}

double SfmSketch::calculateRandomizedResponseProbability(double epsilon) {
  if (epsilon == kNonPrivateEpsilon) {
    return 0.0;
  }
  return 1.0 / (1.0 + exp(epsilon));
}

void SfmSketch::mergeWith(const SfmSketch& other) {
  mergeWith(other, MersenneTwisterRandomizationStrategy());
}

void SfmSketch::enablePrivacy(double epsilon) {
  enablePrivacy(epsilon, getDefaultRandomizationStrategy());
}

int32_t SfmSketch::countBits() const {
  const uint64_t* wordBits = reinterpret_cast<const uint64_t*>(bits_.data());
  return velox::bits::countBits(
      wordBits, 0, static_cast<int32_t>(getNumberOfBits()));
}

uint64_t SfmSketch::cardinality() const {
  // Handle empty sketch case to avoid NaN in updating guess, where denominator
  // will be 0.
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
  return static_cast<uint64_t>(std::round(clampedGuess));
}

size_t SfmSketch::serializedSize() const {
  // Java-compatible format: FORMAT_TAG + indexBitLength + precision +
  // randomizedResponseProbability + bitsetByteLength + bitset_data
  return sizeof(int8_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(double) +
      sizeof(uint32_t) + compactBitSize();
}

void SfmSketch::serialize(char* out) const {
  common::OutputByteStream stream(out);
  stream.appendOne(kFormatTag);
  stream.appendOne(indexBitLength_);
  stream.appendOne(precision_);
  stream.appendOne(randomizedResponseProbability_);

  // Get bitset bytes in Java-compatible format (truncated trailing zeros)
  auto bitSetBytes = toCompactIntVector();
  stream.appendOne(static_cast<uint32_t>(bitSetBytes.size()));

  // Only append data if the vector is not empty to avoid null pointer issues
  if (!bitSetBytes.empty()) {
    stream.append(
        reinterpret_cast<const char*>(bitSetBytes.data()),
        static_cast<int32_t>(bitSetBytes.size()));
  }
}

SfmSketch SfmSketch::deserialize(
    const char* in,
    HashStringAllocator* allocator) {
  common::InputByteStream stream(in);
  auto formatTag = stream.read<int8_t>(); // FORMAT_TAG
  auto indexBitLength = stream.read<uint32_t>();
  auto precision = stream.read<uint32_t>();
  auto randomizedResponseProbability = stream.read<double>();
  auto bitSetBytesLength =
      stream.read<uint32_t>(); // Java stores bitset byte length

  // Validate format tag
  VELOX_CHECK_EQ(
      formatTag,
      kFormatTag,
      "Invalid format tag: expected {}, got {}",
      kFormatTag,
      formatTag);

  // Validate indexBitLength before calling numberOfBuckets
  VELOX_CHECK(
      indexBitLength >= 1 && indexBitLength <= 16,
      "indexBitLength {} is out of range [1, 16]",
      indexBitLength);

  // Create the sketch - constructor already sizes bits_ to the expected length
  SfmSketch sketch(allocator);
  sketch.setSketchSize(numberOfBuckets(indexBitLength), precision);

  // Set the randomized response probability
  sketch.randomizedResponseProbability_ = randomizedResponseProbability;

  // Read the serialized bitmap data directly into the sketch's bitset
  // The sketch's bitset is already the correct size and zero-initialized
  // We just need to copy the serialized data (which may be truncated)
  for (uint32_t i = 0; i < bitSetBytesLength && i < sketch.bits_.size(); ++i) {
    sketch.bits_[i] = stream.read<int8_t>();
  }

  return sketch;
}

void SfmSketch::validateEpsilon(double epsilon) {
  VELOX_CHECK(
      epsilon > 0,
      "Epsilon must be greater than zero or equal to kNonPrivateEpsilon");
}

void SfmSketch::validatePrecision(uint32_t precision, uint32_t indexBitLength) {
  VELOX_CHECK(
      precision + indexBitLength <= 64,
      "Precision + indexBitLength cannot exceed 64");
}

void SfmSketch::validatePrefixLength(uint32_t indexBitLength) {
  VELOX_CHECK(
      indexBitLength >= 1 && indexBitLength <= kMaxIndexBitLength,
      "IndexBitLength is out of range, should be in the interval [1, 16]");
}

void SfmSketch::validateRandomizedResponseProbability(double p) {
  VELOX_CHECK(
      p >= 0 && p <= 0.5,
      "RandomizedResponseProbability should be in the interval [0, 0.5]");
}

double SfmSketch::observationProbability(uint32_t level) const {
  return std::pow(2.0, -(level + 1.0)) / numberOfBuckets(indexBitLength_);
}

size_t SfmSketch::compactBitSize() const {
  // Find the last non-zero byte.
  const uint64_t* wordBits = reinterpret_cast<const uint64_t*>(bits_.data());
  int32_t lastSetBit =
      bits::findLastBit(wordBits, 0, static_cast<int32_t>(getNumberOfBits()));

  if (lastSetBit < 0) {
    // No bits are set
    return 0;
  }
  return static_cast<size_t>(lastSetBit / 8) + 1;
}

const std::vector<int8_t, facebook::velox::StlAllocator<int8_t>>
SfmSketch::toCompactIntVector() const {
  if (bits_.empty()) {
    return std::vector<int8_t, Allocator>(Allocator(bits_.get_allocator()));
  }

  auto compactBitSize = this->compactBitSize();

  return std::vector<int8_t, Allocator>(
      bits_.begin(),
      bits_.begin() + static_cast<int64_t>(compactBitSize),
      Allocator(bits_.get_allocator()));
}

double SfmSketch::logLikelihoodFirstDerivative(double n) const {
  double result = 0.0;
  for (uint32_t level = 0; level < precision_; level++) {
    double termOn = logLikelihoodTermFirstDerivative(level, true, n);
    double termOff = logLikelihoodTermFirstDerivative(level, false, n);
    for (uint32_t bucket = 0; bucket < numberOfBuckets(indexBitLength_);
         bucket++) {
      uint32_t bitPosition = level * numberOfBuckets(indexBitLength_) + bucket;
      result +=
          velox::bits::isBitSet(bits_.data(), bitPosition) ? termOn : termOff;
    }
  }
  return result;
}

double SfmSketch::logLikelihoodTermFirstDerivative(
    uint32_t level,
    bool on,
    double n) const {
  double p = observationProbability(level);
  int32_t sign = on ? -1 : 1;
  double c1 = on ? getOnProbability() : 1 - getOnProbability();
  double c2 = getOnProbability() - getRandomizedResponseProbability();
  return std::log1p(-p) * (1 - c1 / (c1 + sign * c2 * std::pow(1 - p, n)));
}

double SfmSketch::logLikelihoodSecondDerivative(double n) const {
  double result = 0.0;
  for (uint32_t level = 0; level < precision_; level++) {
    double termOn = logLikelihoodTermSecondDerivative(level, true, n);
    double termOff = logLikelihoodTermSecondDerivative(level, false, n);
    for (uint32_t bucket = 0; bucket < numberOfBuckets(indexBitLength_);
         bucket++) {
      uint32_t bitPosition = level * numberOfBuckets(indexBitLength_) + bucket;
      result +=
          velox::bits::isBitSet(bits_.data(), bitPosition) ? termOn : termOff;
    }
  }
  return result;
}

double SfmSketch::logLikelihoodTermSecondDerivative(
    uint32_t level,
    bool on,
    double n) const {
  double p = observationProbability(level);
  int32_t sign = on ? -1 : 1;
  double c1 = on ? getOnProbability() : 1 - getOnProbability();
  double c2 = getOnProbability() - getRandomizedResponseProbability();
  return sign * c1 * c2 * std::pow(std::log1p(-p), 2) * std::pow(1 - p, n) *
      std::pow(c1 + sign * c2 * std::pow(1 - p, n), -2);
}

void SfmSketch::setBitTrue(uint32_t bucketIndex, uint32_t zeros) {
  VELOX_CHECK(!privacyEnabled(), "private sketch is immutable.");
  // It's more likely to have a hash with less zeros than more zeros.
  // we keep the bitmap in the form of a bit matrix, where each row is a zero
  // level instead of a bucket.
  // In this way, we can save space by dropping the trailing zeros in
  // serialization.
  auto bitPosition = zeros * numberOfBuckets(indexBitLength_) + bucketIndex;
  velox::bits::setBit(bits_.data(), static_cast<uint64_t>(bitPosition), true);
}

} // namespace facebook::velox::functions::aggregate
