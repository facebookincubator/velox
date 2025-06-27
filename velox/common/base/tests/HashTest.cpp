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

#include "velox/common/base/Hash.h"

#include <folly/Random.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <functional>
#include <iomanip>

namespace facebook::velox {

namespace {

template <typename T>
T getBitMask(int postition) {
  return T{1} << postition;
}

template <typename T>
T flipBit(T value, int postition) {
  return value ^ getBitMask<T>(postition);
}

template <typename T>
bool isBitFlipped(T original, T modified, int position) {
  T mask = getBitMask<T>(position);
  T originalBit = original & mask;
  T modifiedBit = modified & mask;
  return originalBit != modifiedBit;
}

template <typename T>
std::array<std::array<double, 64>, sizeof(T) * 8> calculateFlipRates(
    std::function<uint64_t(T)> hash,
    int iterations = 1000000) {
  constexpr int kBits = sizeof(T) * 8;
  std::array<std::array<double, 64>, kBits> flips;

  folly::Random::DefaultGenerator rng;
  rng.seed(0xDEADBEAF);

  for (int valueBit = 0; valueBit < kBits; ++valueBit) {
    flips[valueBit].fill(0);
    for (int i = 0; i < iterations; ++i) {
      constexpr int words = std::max(sizeof(T) / sizeof(uint64_t), size_t{1});
      uint64_t value[words];
      for (int j = 0; j < words; ++j) {
        value[j] = folly::Random::rand64(rng);
      }
      T originalValue = *reinterpret_cast<T*>(value);
      uint64_t originalHash = hash(originalValue);
      T modifiedValue = flipBit(originalValue, valueBit);
      uint64_t modifiedHash = hash(modifiedValue);
      for (int hashBit = 0; hashBit < 64; ++hashBit) {
        if (isBitFlipped(originalHash, modifiedHash, hashBit)) {
          ++flips[valueBit][hashBit];
        }
      }
    }
  }

  std::array<std::array<double, 64>, kBits> flipRates;
  for (int valueBit = 0; valueBit < kBits; ++valueBit) {
    for (int hashBit = 0; hashBit < 64; ++hashBit) {
      flipRates[valueBit][hashBit] =
          static_cast<double>(flips[valueBit][hashBit]) / iterations;
    }
  }

  return flipRates;
}

template <typename T>
std::array<std::array<double, 64>, sizeof(T) * 8>
calculateVeloxHasherFlipRates() {
  return calculateFlipRates<T>(velox::hasher<T>{});
}

template <size_t N, size_t M>
void assertFlipRates(const std::array<std::array<double, M>, N>& flipRates) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      ASSERT_NEAR(flipRates[i][j], 0.5, 0.01);
    }
  }
}

template <size_t N, size_t M>
std::ostream& operator<<(
    std::ostream& os,
    const std::array<std::array<double, M>, N>& arr) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < M; ++j) {
      os << std::fixed << std::setprecision(2) << arr[i][j];
      if (j != M - 1) {
        os << "\t";
      }
    }
    if (i != N - 1) {
      os << "\n";
    }
  }
  return os;
}

template <typename T>
void runAvalancheExperiment(
    std::string_view name,
    std::function<uint64_t(T)> hash) {
  LOG(INFO) << name;
  std::cerr << calculateFlipRates(hash) << std::endl;
}

TEST(HashTest, avalanche) {
  assertFlipRates(calculateVeloxHasherFlipRates<uint16_t>());
  assertFlipRates(calculateVeloxHasherFlipRates<uint32_t>());
  assertFlipRates(calculateVeloxHasherFlipRates<uint64_t>());
  assertFlipRates(calculateVeloxHasherFlipRates<__int128>());
}
} // namespace
} // namespace facebook::velox
