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
#include <gtest/gtest.h>
#include <cstring>
#include <memory>

#include "fmt/format.h"
#include "folly/Benchmark.h"
#include "folly/Random.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"

using namespace ::facebook;

TEST(FixedBitArrayTests, setThenGet) {
  constexpr int bitWidth = 10;
  constexpr int elementCount = 10000;
  auto buffer = std::make_unique<char[]>(
      nimble::FixedBitArray::bufferSize(elementCount, bitWidth));
  nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);
  for (int i = 0; i < elementCount; ++i) {
    fixedBitArray.set(i, (i + i * i) % (1 << bitWidth));
  }
  for (int i = 0; i < elementCount; ++i) {
    ASSERT_EQ(fixedBitArray.get(i), (i + i * i) % (1 << bitWidth));
  }
}

TEST(FixedBitArrayTests, zeroAndSet) {
  constexpr int bitWidth = 4;
  constexpr int elementCount = 1234;
  auto buffer = std::make_unique<char[]>(
      nimble::FixedBitArray::bufferSize(elementCount, bitWidth));
  nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);
  for (int i = 0; i < elementCount; ++i) {
    fixedBitArray.set(i, (i + i * i) % (1 << bitWidth));
  }
  for (int i = 0; i < elementCount; ++i) {
    ASSERT_EQ(fixedBitArray.get(i), (i + i * i) % (1 << bitWidth));
  }
  for (int i = 0; i < elementCount; ++i) {
    if (i % 3 == 0) {
      fixedBitArray.zeroAndSet(i, i % (1 << bitWidth));
    }
  }
  for (int i = 0; i < elementCount; ++i) {
    if (i % 3 == 0) {
      ASSERT_EQ(fixedBitArray.get(i), i % (1 << bitWidth));
    } else {
      ASSERT_EQ(fixedBitArray.get(i), (i + i * i) % (1 << bitWidth));
    }
  }
}

constexpr int kNumTestsPerBitWidth = 10;
constexpr int kMaxElements = 1000;

TEST(FixedBitArrayTests, setThenGetRandom) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  for (int bitWidth = 1; bitWidth <= 64; ++bitWidth) {
    const uint64_t elementMask =
        bitWidth == 64 ? (~0ULL) : ((1ULL << bitWidth) - 1);
    for (int test = 0; test < kNumTestsPerBitWidth; ++test) {
      const int elementCount = folly::Random::rand32(rng) % kMaxElements;
      auto buffer = std::make_unique<char[]>(
          nimble::FixedBitArray::bufferSize(elementCount, bitWidth));
      nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);
      std::vector<uint64_t> randomValues(elementCount);
      for (int i = 0; i < elementCount; ++i) {
        randomValues[i] = folly::Random::rand64(rng) & elementMask;
      }
      for (int i = 0; i < elementCount; ++i) {
        fixedBitArray.set(i, randomValues[i]);
      }
      for (int i = 0; i < elementCount; ++i) {
        ASSERT_EQ(fixedBitArray.get(i), randomValues[i]);
      }
    }
  }
}

TEST(FixedBitArrayTests, zeroAndSetThenGetRandom) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  for (int bitWidth = 1; bitWidth <= 64; ++bitWidth) {
    const uint64_t elementMask =
        bitWidth == 64 ? (~0ULL) : ((1ULL << bitWidth) - 1);
    for (int test = 0; test < kNumTestsPerBitWidth; ++test) {
      const int elementCount = folly::Random::rand32(rng) % kMaxElements;
      std::vector<char> buffer(
          nimble::FixedBitArray::bufferSize(elementCount, bitWidth), 0xFF);
      nimble::FixedBitArray fixedBitArray(buffer.data(), bitWidth);
      std::vector<uint64_t> randomValues(elementCount);
      for (int i = 0; i < elementCount; ++i) {
        randomValues[i] = folly::Random::rand64(rng) & elementMask;
      }
      for (int i = 0; i < elementCount; ++i) {
        fixedBitArray.zeroAndSet(i, randomValues[i]);
      }
      for (int i = 0; i < elementCount; ++i) {
        ASSERT_EQ(fixedBitArray.get(i), randomValues[i]) << bitWidth;
      }
    }
  }
}

TEST(FixedBitArrayTests, set32ThenGet32Random) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  for (int bitWidth = 1; bitWidth <= 32; ++bitWidth) {
    const uint64_t elementMask = (1ULL << bitWidth) - 1;
    for (int test = 0; test < kNumTestsPerBitWidth; ++test) {
      const int elementCount = folly::Random::rand32(rng) % kMaxElements;
      auto buffer = std::make_unique<char[]>(
          nimble::FixedBitArray::bufferSize(elementCount, bitWidth));
      nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);
      std::vector<uint32_t> randomValues(elementCount);
      for (int i = 0; i < elementCount; ++i) {
        randomValues[i] = folly::Random::rand32(rng) & elementMask;
      }
      for (int i = 0; i < elementCount; ++i) {
        fixedBitArray.set32(i, randomValues[i]);
      }
      for (int i = 0; i < elementCount; ++i) {
        ASSERT_EQ(fixedBitArray.get32(i), randomValues[i])
            << i << " " << bitWidth;
      }
    }
  }
}

TEST(FixedBitArrayTests, bulkGet32) {
  constexpr int kBitWidth = 7;
  constexpr int kNumElements = 27;
  auto buffer = std::make_unique<char[]>(
      nimble::FixedBitArray::bufferSize(kNumElements, kBitWidth));
  nimble::FixedBitArray fixedBitArray(buffer.get(), kBitWidth);
  for (int i = 0; i < kNumElements; ++i) {
    fixedBitArray.set32(i, 100 - i);
  }
  std::vector<uint32_t> values(kNumElements);
  fixedBitArray.bulkGet32(0, kNumElements, values.data());
  for (int i = 0; i < kNumElements; ++i) {
    ASSERT_EQ(values[i], 100 - i);
  }
  std::vector<uint64_t> values64(kNumElements);
  fixedBitArray.bulkGet32Into64(0, kNumElements, values64.data());
  for (int i = 0; i < kNumElements; ++i) {
    ASSERT_EQ(values64[i], 100 - i);
  }
}

TEST(FixedBitArrayTests, bulkGet32Random) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  for (int bitWidth = 1; bitWidth <= 32; ++bitWidth) {
    const uint64_t elementMask = (1ULL << bitWidth) - 1;
    for (int test = 0; test < kNumTestsPerBitWidth; ++test) {
      const int elementCount = folly::Random::rand32(rng) % kMaxElements;
      auto buffer = std::make_unique<char[]>(
          nimble::FixedBitArray::bufferSize(elementCount, bitWidth));
      nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);
      std::vector<uint32_t> randomValues(elementCount);
      for (int i = 0; i < elementCount; ++i) {
        randomValues[i] = folly::Random::rand32(rng) & elementMask;
      }
      for (int i = 0; i < elementCount; ++i) {
        fixedBitArray.set32(i, randomValues[i]);
      }
      std::vector<uint32_t> values(elementCount);
      fixedBitArray.bulkGet32(0, elementCount, values.data());
      for (int i = 0; i < elementCount; ++i) {
        ASSERT_EQ(values[i], randomValues[i]);
      }
      for (int i = 0; i < elementCount; ++i) {
        uint32_t element;
        fixedBitArray.bulkGet32(i, 1, &element);
        ASSERT_EQ(element, values[i]);
      }
      std::vector<uint64_t> values64(elementCount);
      fixedBitArray.bulkGet32Into64(0, elementCount, values64.data());
      for (int i = 0; i < elementCount; ++i) {
        ASSERT_EQ(values64[i], randomValues[i]);
      }
      for (int i = 0; i < elementCount; ++i) {
        uint64_t element;
        fixedBitArray.bulkGet32Into64(i, 1, &element);
        ASSERT_EQ(element, values64[i]);
      }
    }
  }
}

TEST(FixedBitArrayTests, bulkGetWithBaseline32Random) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  for (int bitWidth = 4; bitWidth <= 32; ++bitWidth) {
    const uint64_t maxElement = (1ULL << bitWidth);
    const uint64_t baseline = folly::Random::rand32(rng) % maxElement;
    for (int test = 0; test < kNumTestsPerBitWidth; ++test) {
      const int elementCount = folly::Random::rand32(rng) % kMaxElements;
      auto buffer = std::make_unique<char[]>(
          nimble::FixedBitArray::bufferSize(elementCount, bitWidth));
      nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);
      std::vector<uint32_t> randomValues(elementCount);
      for (int i = 0; i < elementCount; ++i) {
        randomValues[i] = folly::Random::rand32(rng) % (maxElement - baseline);
      }
      for (int i = 0; i < elementCount; ++i) {
        fixedBitArray.set32(i, randomValues[i]);
      }
      std::vector<uint32_t> values(elementCount);
      fixedBitArray.bulkGetWithBaseline(
          0, elementCount, values.data(), baseline);
      for (int i = 0; i < elementCount; ++i) {
        ASSERT_EQ(values[i], randomValues[i] + baseline)
            << "Bit Width: " << bitWidth << ", i: " << i;
      }
      for (int i = 0; i < elementCount; ++i) {
        uint32_t element;
        fixedBitArray.bulkGetWithBaseline(i, 1, &element, baseline);
        ASSERT_EQ(element, randomValues[i] + baseline);
      }
      std::vector<uint64_t> values64(elementCount);
      fixedBitArray.bulkGetWithBaseline32Into64(
          0, elementCount, values64.data(), baseline);
      for (int i = 0; i < elementCount; ++i) {
        ASSERT_EQ(values64[i], randomValues[i] + baseline);
      }
      for (int i = 0; i < elementCount; ++i) {
        uint64_t element;
        fixedBitArray.bulkGetWithBaseline32Into64(i, 1, &element, baseline);
        ASSERT_EQ(element, randomValues[i] + baseline);
      }
    }
  }
}

TEST(FixedBitArrayTests, bulkGet64WithBaseline) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  // Test all three code paths:
  // - bitWidth <= 32: delegates to bulkGetWithBaseline32Into64
  // - bitWidth 33-58: branchless byte-aligned loads
  // - bitWidth > 58: inline cross-word boundary handling
  for (int bitWidth = 1; bitWidth <= 64; ++bitWidth) {
    SCOPED_TRACE(fmt::format("bitWidth={}", bitWidth));
    const uint64_t maxElement = bitWidth == 64
        ? std::numeric_limits<uint64_t>::max()
        : (1ULL << bitWidth) - 1;
    const uint64_t baseline = folly::Random::rand64(rng) % (maxElement / 2 + 1);
    const uint64_t valueRange = maxElement - baseline;

    for (int test = 0; test < kNumTestsPerBitWidth; ++test) {
      const int elementCount = folly::Random::rand32(rng) % kMaxElements;
      const auto bufferBytes =
          nimble::FixedBitArray::bufferSize(elementCount, bitWidth);
      auto buffer = std::make_unique<char[]>(bufferBytes);
      std::memset(buffer.get(), 0, bufferBytes);
      nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);

      std::vector<uint64_t> randomValues(elementCount);
      for (int i = 0; i < elementCount; ++i) {
        randomValues[i] = valueRange == std::numeric_limits<uint64_t>::max()
            ? folly::Random::rand64(rng)
            : folly::Random::rand64(rng) % (valueRange + 1);
      }
      for (int i = 0; i < elementCount; ++i) {
        fixedBitArray.set(i, randomValues[i]);
      }

      // Bulk read all elements.
      std::vector<uint64_t> values(elementCount);
      fixedBitArray.bulkGetWithBaseline(
          0, elementCount, values.data(), baseline);
      for (int i = 0; i < elementCount; ++i) {
        ASSERT_EQ(values[i], randomValues[i] + baseline)
            << "bitWidth: " << bitWidth << ", i: " << i;
      }

      // Single-element reads at each position.
      for (int i = 0; i < elementCount; ++i) {
        uint64_t element;
        fixedBitArray.bulkGetWithBaseline(i, 1, &element, baseline);
        ASSERT_EQ(element, randomValues[i] + baseline);
      }

      // Read from a random offset.
      if (elementCount > 1) {
        const int offset = folly::Random::rand32(rng) % (elementCount - 1);
        const int count = elementCount - offset;
        std::vector<uint64_t> partial(count);
        fixedBitArray.bulkGetWithBaseline(
            offset, count, partial.data(), baseline);
        for (int i = 0; i < count; ++i) {
          ASSERT_EQ(partial[i], randomValues[offset + i] + baseline);
        }
      }
    }
  }
}

TEST(FixedBitArrayTests, bulkGet64WithBaselineZeroBaseline) {
  // Verify bulkGet64WithBaseline with baseline=0 matches per-element get().
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  for (int bitWidth : {1, 8, 16, 32, 33, 40, 48, 56, 57, 58, 60, 64}) {
    SCOPED_TRACE(fmt::format("bitWidth={}", bitWidth));
    const int elementCount = 100 + folly::Random::rand32(rng) % 200;
    const uint64_t maxElement = bitWidth == 64
        ? std::numeric_limits<uint64_t>::max()
        : (1ULL << bitWidth) - 1;
    const auto bufferBytes =
        nimble::FixedBitArray::bufferSize(elementCount, bitWidth);
    auto buffer = std::make_unique<char[]>(bufferBytes);
    std::memset(buffer.get(), 0, bufferBytes);
    nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);

    for (int i = 0; i < elementCount; ++i) {
      const uint64_t value = bitWidth == 64
          ? folly::Random::rand64(rng)
          : folly::Random::rand64(rng) % (maxElement + 1);
      fixedBitArray.set(i, value);
    }

    std::vector<uint64_t> bulkValues(elementCount);
    fixedBitArray.bulkGetWithBaseline(0, elementCount, bulkValues.data(), 0);

    for (int i = 0; i < elementCount; ++i) {
      ASSERT_EQ(bulkValues[i], fixedBitArray.get(i))
          << "bitWidth: " << bitWidth << ", i: " << i;
    }
  }
}

TEST(FixedBitArrayTests, bulkSet32Random) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);
  for (int bitWidth = 1; bitWidth <= 32; ++bitWidth) {
    const uint64_t maxElement = (1ULL << bitWidth);
    for (int test = 0; test < kNumTestsPerBitWidth; ++test) {
      const int elementCount = 1 + folly::Random::rand32(rng) % kMaxElements;
      auto buffer = std::make_unique<char[]>(
          nimble::FixedBitArray::bufferSize(elementCount, bitWidth));
      nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);
      std::vector<uint32_t> randomValues(elementCount);
      for (int i = 0; i < elementCount; ++i) {
        randomValues[i] = folly::Random::rand32(rng) % maxElement;
      }
      const int offset = folly::Random::rand32(rng) % elementCount;
      const int size = elementCount - offset;
      fixedBitArray.bulkSet32(offset, size, randomValues.data() + offset);
      for (int i = 0; i < size; ++i) {
        ASSERT_EQ(fixedBitArray.get32(offset + i), randomValues[offset + i]);
      }
    }
  }
}

TEST(FixedBitArrayTests, bulkSet32WithBaselineRandom) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);
  for (int bitWidth = 4; bitWidth <= 32; ++bitWidth) {
    const uint64_t maxElement = (1ULL << bitWidth);
    const uint64_t baseline = folly::Random::rand32(rng) % maxElement;
    for (int test = 0; test < kNumTestsPerBitWidth; ++test) {
      const int elementCount = 1 + folly::Random::rand32(rng) % kMaxElements;
      auto buffer = std::make_unique<char[]>(
          nimble::FixedBitArray::bufferSize(elementCount, bitWidth));
      nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);
      std::vector<uint32_t> randomValues(elementCount);
      std::vector<uint32_t> randomValuesWithBaseline(elementCount);
      for (int i = 0; i < elementCount; ++i) {
        randomValues[i] = folly::Random::rand32(rng) % (maxElement - baseline);
        randomValuesWithBaseline[i] = randomValues[i] + baseline;
      }
      const int offset = folly::Random::rand32(rng) % elementCount;
      const int size = elementCount - offset;
      fixedBitArray.bulkSetWithBaseline(
          offset, size, randomValuesWithBaseline.data() + offset, baseline);
      for (int i = 0; i < size; ++i) {
        ASSERT_EQ(fixedBitArray.get32(offset + i), randomValues[offset + i])
            << "Bit Width: " << bitWidth << ", i: " << i;
      }
    }
  }
}

TEST(FixedBitArrayTests, equals32Random) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  for (int bitWidth = 1; bitWidth <= 32; ++bitWidth) {
    const uint64_t elementMask = (1ULL << bitWidth) - 1;
    for (int test = 0; test < kNumTestsPerBitWidth; ++test) {
      const int elementCount = 1 + folly::Random::rand32(rng) % kMaxElements;
      auto buffer = std::make_unique<char[]>(
          nimble::FixedBitArray::bufferSize(elementCount, bitWidth));
      nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);
      std::vector<uint32_t> randomValues(elementCount);
      for (int i = 0; i < elementCount; ++i) {
        // Test both the full range and also a case where equals are actually
        // likely.
        randomValues[i] = folly::Random::rand32(rng) & elementMask;
        if (test % 2 == 1) {
          randomValues[i] = randomValues[i] % 10;
        }
        fixedBitArray.set32(i, randomValues[i]);
      }
      // Remember start needs to be a multiple of 64.
      const int start =
          (folly::Random::rand32(rng) % elementCount) & (0xFFFFFF00);
      const int length = elementCount - start;
      auto equalsBuffer = std::make_unique<char[]>(
          nimble::FixedBitArray::bufferSize(length, 1));
      const uint32_t equalsValue =
          randomValues[folly::Random::rand32(rng) % elementCount];
      fixedBitArray.equals32(start, length, equalsValue, equalsBuffer.get());
      for (int i = 0; i < length; ++i) {
        ASSERT_EQ(
            velox::bits::isBitSet(
                reinterpret_cast<const uint8_t*>(equalsBuffer.get()), i),
            randomValues[start + i] == equalsValue);
      }
    }
  }
}

TEST(FixedBitArrayTests, bulkGet64WithBaselineBitWidth58Boundary) {
  constexpr int bitWidth = 58;
  constexpr int elementCount = 5;
  constexpr int start = 3;
  constexpr uint64_t baseline = 17;
  constexpr uint64_t maxElement = (1ULL << bitWidth) - 1;

  const auto bufferBytes =
      nimble::FixedBitArray::bufferSize(elementCount, bitWidth);
  auto buffer = std::make_unique<char[]>(bufferBytes);
  std::memset(buffer.get(), 0, bufferBytes);
  nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);

  // start * bitWidth leaves a bit remainder of 6. For bitWidth 58 this is the
  // highest possible remainder, and 58 + 6 exactly fills one 64-bit load.
  fixedBitArray.set(start, maxElement);

  uint64_t value = 0;
  fixedBitArray.bulkGetWithBaseline(start, 1, &value, baseline);
  ASSERT_EQ(value, maxElement + baseline);
}

TEST(FixedBitArrayTests, bulkSet64WithBaseline) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  // Test all three code paths:
  // - bitWidth <= 32: delegates to template-unrolled bulkSetInternal32
  // - bitWidth 33-58: branchless byte-aligned stores
  // - bitWidth > 58: inline cross-word boundary handling
  for (int bitWidth = 1; bitWidth <= 64; ++bitWidth) {
    SCOPED_TRACE(fmt::format("bitWidth={}", bitWidth));
    const uint64_t maxElement = bitWidth == 64
        ? std::numeric_limits<uint64_t>::max()
        : (1ULL << bitWidth) - 1;
    const uint64_t baseline = folly::Random::rand64(rng) % (maxElement / 2 + 1);
    const uint64_t valueRange = maxElement - baseline;

    for (int test = 0; test < kNumTestsPerBitWidth; ++test) {
      const int elementCount = 1 + folly::Random::rand32(rng) % kMaxElements;
      const auto bufferBytes =
          nimble::FixedBitArray::bufferSize(elementCount, bitWidth);
      auto buffer = std::make_unique<char[]>(bufferBytes);
      std::memset(buffer.get(), 0, bufferBytes);
      nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);

      std::vector<uint64_t> inputValues(elementCount);
      std::vector<uint64_t> expectedResiduals(elementCount);
      for (int i = 0; i < elementCount; ++i) {
        expectedResiduals[i] =
            valueRange == std::numeric_limits<uint64_t>::max()
            ? folly::Random::rand64(rng)
            : folly::Random::rand64(rng) % (valueRange + 1);
        inputValues[i] = expectedResiduals[i] + baseline;
      }

      fixedBitArray.bulkSetWithBaseline(
          0, elementCount, inputValues.data(), baseline);

      for (int i = 0; i < elementCount; ++i) {
        ASSERT_EQ(fixedBitArray.get(i), expectedResiduals[i]) << "i=" << i;
      }

      std::vector<uint64_t> recovered(elementCount);
      fixedBitArray.bulkGetWithBaseline(
          0, elementCount, recovered.data(), baseline);
      ASSERT_EQ(recovered, inputValues);
    }
  }
}

TEST(FixedBitArrayTests, bulkSet64WithBaselinePartialRange) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  for (int bitWidth : {4, 16, 32, 33, 40, 48, 56, 57, 58, 64}) {
    SCOPED_TRACE(fmt::format("bitWidth={}", bitWidth));
    const uint64_t maxElement = bitWidth == 64
        ? std::numeric_limits<uint64_t>::max()
        : (1ULL << bitWidth) - 1;
    const uint64_t baseline = folly::Random::rand64(rng) % (maxElement / 2 + 1);
    const uint64_t valueRange = maxElement - baseline;

    const int elementCount = 200;
    const auto bufferBytes =
        nimble::FixedBitArray::bufferSize(elementCount, bitWidth);
    auto buffer = std::make_unique<char[]>(bufferBytes);
    std::memset(buffer.get(), 0, bufferBytes);
    nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);

    const int offset = 50;
    const int count = 100;
    std::vector<uint64_t> inputValues(count);
    for (int i = 0; i < count; ++i) {
      const uint64_t residual =
          valueRange == std::numeric_limits<uint64_t>::max()
          ? folly::Random::rand64(rng)
          : folly::Random::rand64(rng) % (valueRange + 1);
      inputValues[i] = residual + baseline;
    }

    fixedBitArray.bulkSetWithBaseline(
        offset, count, inputValues.data(), baseline);

    for (int i = 0; i < count; ++i) {
      ASSERT_EQ(fixedBitArray.get(offset + i), inputValues[i] - baseline)
          << "i=" << i;
    }

    // Round-trip the partial range through the bulk read path. A non-zero start
    // offset exercises the byte-aligned masked load at a non-zero base pointer
    // (its last element also reads into bufferSize's trailing slop).
    std::vector<uint64_t> recovered(count);
    fixedBitArray.bulkGetWithBaseline(
        offset, count, recovered.data(), baseline);
    ASSERT_EQ(recovered, inputValues);

    for (int i = 0; i < offset; ++i) {
      ASSERT_EQ(fixedBitArray.get(i), 0)
          << "pre-range slot " << i << " should be zero";
    }
    for (int i = offset + count; i < elementCount; ++i) {
      ASSERT_EQ(fixedBitArray.get(i), 0)
          << "post-range slot " << i << " should be zero";
    }
  }
}

// Packs random values of element type T through the unified
// bulkSetWithBaseline and confirms the stored residuals round-trip. bitWidth is
// chosen so values fit in T.
template <typename T>
void verifyBulkSetWithBaseline(
    int bitWidth,
    int elementCount,
    std::mt19937& rng) {
  SCOPED_TRACE(
      fmt::format(
          "typeBytes={} bitWidth={} elementCount={}",
          sizeof(T),
          bitWidth,
          elementCount));
  const auto bufferBytes =
      nimble::FixedBitArray::bufferSize(elementCount, bitWidth);
  auto buffer = std::make_unique<char[]>(bufferBytes);
  std::memset(buffer.get(), 0, bufferBytes);
  nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);

  const T baseline = static_cast<T>(folly::Random::rand32(rng) % 7);
  const uint64_t residualMask =
      bitWidth == 64 ? ~0ULL : ((1ULL << bitWidth) - 1);
  std::vector<T> inputValues(elementCount);
  std::vector<uint64_t> expectedResiduals(elementCount);
  for (int i = 0; i < elementCount; ++i) {
    expectedResiduals[i] = folly::Random::rand64(rng) & residualMask;
    inputValues[i] = static_cast<T>(expectedResiduals[i] + baseline);
  }

  fixedBitArray.bulkSetWithBaseline(
      0, elementCount, inputValues.data(), baseline);

  std::vector<uint64_t> actualResiduals(elementCount);
  for (int i = 0; i < elementCount; ++i) {
    actualResiduals[i] = fixedBitArray.get(i);
  }
  EXPECT_EQ(actualResiduals, expectedResiduals);
}

TEST(FixedBitArrayTests, bulkSetWithBaselineDispatchesByElementWidth) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  const int smallCount = 1 + folly::Random::rand32(rng) % kMaxElements;
  // One element type per compile-time branch: 1- and 2-byte exercise the widen
  // path, 4-byte delegates to bulkSet32, 8-byte to bulkSet64.
  verifyBulkSetWithBaseline<uint8_t>(/*bitWidth=*/5, smallCount, rng);
  verifyBulkSetWithBaseline<uint16_t>(/*bitWidth=*/12, smallCount, rng);
  verifyBulkSetWithBaseline<uint32_t>(/*bitWidth=*/20, smallCount, rng);
  verifyBulkSetWithBaseline<uint64_t>(/*bitWidth=*/40, smallCount, rng);

  // Exercise the narrow widen path across multiple stack chunks
  // (length > kWidenChunk = 1024), including a partial final chunk.
  verifyBulkSetWithBaseline<uint8_t>(
      /*bitWidth=*/5, /*elementCount=*/2500, rng);
  verifyBulkSetWithBaseline<uint16_t>(
      /*bitWidth=*/12, /*elementCount=*/2500, rng);
}

// Round-trips values of element type T through bulkSetWithBaseline +
// bulkGetWithBaseline and confirms each compile-time branch recovers them.
template <typename T>
void verifyBulkGetWithBaseline(
    int bitWidth,
    int elementCount,
    std::mt19937& rng) {
  SCOPED_TRACE(
      fmt::format(
          "typeBytes={} bitWidth={} elementCount={}",
          sizeof(T),
          bitWidth,
          elementCount));
  const auto bufferBytes =
      nimble::FixedBitArray::bufferSize(elementCount, bitWidth);
  auto buffer = std::make_unique<char[]>(bufferBytes);
  std::memset(buffer.get(), 0, bufferBytes);
  nimble::FixedBitArray fixedBitArray(buffer.get(), bitWidth);

  const T baseline = static_cast<T>(folly::Random::rand32(rng) % 7);
  const uint64_t residualMask =
      bitWidth == 64 ? ~0ULL : ((1ULL << bitWidth) - 1);
  std::vector<T> inputValues(elementCount);
  for (int i = 0; i < elementCount; ++i) {
    inputValues[i] =
        static_cast<T>((folly::Random::rand64(rng) & residualMask) + baseline);
  }

  fixedBitArray.bulkSetWithBaseline(
      0, elementCount, inputValues.data(), baseline);

  std::vector<T> actual(elementCount);
  fixedBitArray.bulkGetWithBaseline(0, elementCount, actual.data(), baseline);
  EXPECT_EQ(actual, inputValues);
}

TEST(FixedBitArrayTests, bulkGetWithBaselineDispatchesByElementWidth) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  const int smallCount = 1 + folly::Random::rand32(rng) % kMaxElements;
  verifyBulkGetWithBaseline<uint8_t>(/*bitWidth=*/5, smallCount, rng);
  verifyBulkGetWithBaseline<uint16_t>(/*bitWidth=*/12, smallCount, rng);
  verifyBulkGetWithBaseline<uint32_t>(/*bitWidth=*/20, smallCount, rng);
  verifyBulkGetWithBaseline<uint64_t>(/*bitWidth=*/40, smallCount, rng);

  // Narrow read path across multiple stack chunks (length > kWidenChunk).
  verifyBulkGetWithBaseline<uint8_t>(
      /*bitWidth=*/5, /*elementCount=*/2500, rng);
  verifyBulkGetWithBaseline<uint16_t>(
      /*bitWidth=*/12, /*elementCount=*/2500, rng);
}
