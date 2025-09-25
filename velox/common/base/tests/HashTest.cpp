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
T generateValue(folly::Random::DefaultGenerator& rng) {
  constexpr int words = std::max(sizeof(T) / sizeof(uint64_t), size_t{1});
  uint64_t value[words];
  for (int j = 0; j < words; ++j) {
    value[j] = folly::Random::rand64(rng);
  }
  return *reinterpret_cast<T*>(value);
}

template <typename T>
std::array<std::array<double, 64>, sizeof(T) * 8> calculateFlipRates(
    const std::function<uint64_t(T)>& hash,
    int iterations = 1000000) {
  constexpr int kBits = sizeof(T) * 8;
  std::array<std::array<double, 64>, kBits> flips{};

  folly::Random::DefaultGenerator rng;
  rng.seed(0xDEADBEAF);

  if (sizeof(T) <= 2) {
    iterations =
        static_cast<int>(std::numeric_limits<std::make_unsigned_t<T>>::max());
  }

  for (int valueBit = 0; valueBit < kBits; ++valueBit) {
    flips[valueBit].fill(0);
    for (int i = 0; i <= iterations; ++i) {
      T originalValue;
      if (sizeof(T) <= 2) {
        originalValue = static_cast<T>(i);
      } else {
        originalValue = generateValue<T>(rng);
      }
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

  std::array<std::array<double, 64>, kBits> flipRates{};
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
void assertFlipRates(
    const std::array<std::array<double, M>, N>& flipRates,
    float precision) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      ASSERT_NEAR(flipRates[i][j], 0.5, precision);
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
  assertFlipRates(calculateVeloxHasherFlipRates<uint8_t>(), 0.15);
  assertFlipRates(calculateVeloxHasherFlipRates<int8_t>(), 0.15);
  assertFlipRates(calculateVeloxHasherFlipRates<uint16_t>(), 0.012);
  assertFlipRates(calculateVeloxHasherFlipRates<int16_t>(), 0.012);
  assertFlipRates(calculateVeloxHasherFlipRates<uint32_t>(), 0.002);
  assertFlipRates(calculateVeloxHasherFlipRates<int32_t>(), 0.002);
  assertFlipRates(calculateVeloxHasherFlipRates<uint64_t>(), 0.002);
  assertFlipRates(calculateVeloxHasherFlipRates<int64_t>(), 0.002);
  assertFlipRates(calculateVeloxHasherFlipRates<unsigned __int128>(), 0.002);
  assertFlipRates(calculateVeloxHasherFlipRates<__int128>(), 0.002);
}

TEST(HashTest, partition) {
  EXPECT_EQ(velox::hasher<uint64_t>{}(0) % 2, 0);
  EXPECT_EQ(velox::hasher<uint64_t>{}(1) % 2, 1);
  EXPECT_EQ(velox::hasher<uint64_t>{}(2) % 2, 0);
  EXPECT_EQ(velox::hasher<uint64_t>{}(3) % 2, 1);
  EXPECT_EQ(velox::hasher<uint64_t>{}(4) % 2, 1);
  EXPECT_EQ(velox::hasher<uint64_t>{}(5) % 2, 0);
  EXPECT_EQ(velox::hasher<uint64_t>{}(6) % 2, 1);
  EXPECT_EQ(velox::hasher<uint64_t>{}(7) % 2, 0);
  EXPECT_EQ(velox::hasher<uint64_t>{}(8) % 2, 0);
  EXPECT_EQ(velox::hasher<uint64_t>{}(9) % 2, 1);
  EXPECT_EQ(velox::hasher<uint64_t>{}(10) % 2, 1);
}
} // namespace
} // namespace facebook::velox
