// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/IOUtils.h"

namespace facebook::velox::functions::aggregate {

class NoisySumAccumulator {
 public:
  NoisySumAccumulator(
      double sum,
      double noiseScale,
      std::optional<int32_t> randomSeed)
      : sum_{sum}, noiseScale_{noiseScale}, randomSeed_{randomSeed} {}

  NoisySumAccumulator() = default;

  void checkAndSetNoiseScale(double noiseScale) {
    VELOX_USER_CHECK_GE(
        noiseScale, 0.0, "Noise scale must be non-negative value.");
    this->noiseScale_ = noiseScale;
  }

  void setRandomSeed(int32_t randomSeed) {
    this->randomSeed_ = randomSeed;
  }

  // This function is used to update the sum
  void update(double value) {
    this->sum_ += value;
  }

  double getSum() const {
    return this->sum_;
  }

  double getNoiseScale() const {
    return this->noiseScale_;
  }

  std::optional<int32_t> getRandomSeed() const {
    return this->randomSeed_;
  }

  static size_t serializedSize() {
    // The serialized size should be the sum of:
    // - sizeof(double) for sum_
    // - sizeof(double) for noiseScale_
    // - sizeof(bool) for randomSeed_ has_value flag
    // - sizeof(int32_t) for randomSeed_ value
    return sizeof(double) + sizeof(double) + sizeof(bool) + sizeof(int32_t);
  }

  void serialize(char* buffer) {
    common::OutputByteStream stream(buffer);
    stream.appendOne(sum_);
    stream.appendOne(noiseScale_);

    // Serialize randomSeed_.(append 0 if has_value is false)
    stream.appendOne(randomSeed_.has_value());
    stream.appendOne(randomSeed_.has_value() ? randomSeed_.value() : 0);
  }

  static NoisySumAccumulator deserialize(const char* intermediate) {
    common::InputByteStream stream(intermediate);
    auto sum = stream.read<double>();
    auto noiseScale = stream.read<double>();

    // Deserialize randomSeed_
    bool hasRandomSeed = stream.read<bool>();
    int32_t randomSeed = stream.read<int32_t>();

    return hasRandomSeed ? NoisySumAccumulator{sum, noiseScale, randomSeed}
                         : NoisySumAccumulator{sum, noiseScale, std::nullopt};
  }

 private:
  double sum_{0.0};
  // Initial noise scale is an invalid noise scale,
  // indicating that we have not updated it yet
  double noiseScale_{-1.0};
  std::optional<int32_t> randomSeed_{std::nullopt};
};

} // namespace facebook::velox::functions::aggregate
