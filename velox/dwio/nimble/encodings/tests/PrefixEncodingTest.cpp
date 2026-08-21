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
#include "velox/dwio/nimble/encodings/PrefixEncoding.h"
#include <fmt/core.h>
#include <gtest/gtest.h>
#include "folly/Conv.h"
#include "velox/buffer/Buffer.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

using namespace ::facebook;

namespace facebook::nimble::test {

class PrefixEncodingTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    if (!velox::memory::MemoryManager::testInstance()) {
      velox::memory::MemoryManager::initialize({});
    }
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool();
  }

  std::unique_ptr<EncodingSelectionPolicy<std::string_view>>
  createSelectionPolicy() {
    // Create an EncodingLayout that specifies PrefixEncoding for the top level.
    // Nested encodings (lengths, data) will be handled by the factory.
    EncodingLayout layout{
        EncodingType::Prefix, {}, CompressionType::Uncompressed};
    return std::make_unique<ReplayedEncodingSelectionPolicy<std::string_view>>(
        std::move(layout),
        CompressionOptions{},
        encodingSelectionPolicyCreator_);
  }

  // Helper function to create a string buffer factory for decoding
  std::function<void*(uint32_t)> createStringBufferFactory() {
    return [this](uint32_t totalLength) {
      auto& buffer = stringBuffers_.emplace_back(
          velox::AlignedBuffer::allocate<char>(totalLength, pool_.get()));
      return buffer->asMutable<void>();
    };
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::vector<velox::BufferPtr> stringBuffers_;
  ManualEncodingSelectionPolicyFactory manualPolicyFactory_;
  EncodingSelectionPolicyCreator encodingSelectionPolicyCreator_ =
      [this](DataType dataType) {
        return manualPolicyFactory_.createPolicy(dataType);
      };
};

TEST_F(PrefixEncodingTest, materialize) {
  struct TestCase {
    std::string_view name;
    std::vector<std::string_view> values;
  };

  const std::vector<TestCase> testCases = {
      {"basic sorted strings",
       {"apple", "application", "apply", "banana", "bandana", "band"}},
      {"empty strings", {"", "", "a", "ab", "abc"}},
      {"all empty strings", {"", "", "", ""}},
      {"long common prefix",
       {"aaaaaaaaaaaaaaaaaaaa1",
        "aaaaaaaaaaaaaaaaaaaa2",
        "aaaaaaaaaaaaaaaaaaaa3",
        "aaaaaaaaaaaaaaaaaaaa4"}},
      {"single value", {"single"}},
      {"all same values", {"same", "same", "same"}},
      {"no common prefix", {"abc", "def", "ghi", "jkl"}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);

    Buffer buffer{*pool_};

    auto encoded = EncodingFactory::encode<std::string_view>(
        createSelectionPolicy(), testCase.values, buffer);
    stringBuffers_.clear();
    auto encoding =
        EncodingFactory().create(*pool_, encoded, createStringBufferFactory());

    EXPECT_EQ(encoding->rowCount(), testCase.values.size());
    EXPECT_EQ(encoding->encodingType(), EncodingType::Prefix);

    std::vector<std::string_view> decoded(testCase.values.size());
    encoding->materialize(testCase.values.size(), decoded.data());

    for (size_t i = 0; i < testCase.values.size(); ++i) {
      EXPECT_EQ(decoded[i], testCase.values[i]) << "Mismatch at index " << i;
    }
  }
}

TEST_F(PrefixEncodingTest, resetReusesMultipleStringBufferPages) {
  constexpr uint32_t kValueCount = 48;
  constexpr size_t kValueSize = 20 * 1024;

  std::vector<std::string> storage;
  storage.reserve(kValueCount);
  const std::string commonPrefix(kValueSize, 'p');
  for (uint32_t i = 0; i < kValueCount; ++i) {
    storage.push_back(commonPrefix + fmt::format("/{:04}", i));
  }

  std::vector<std::string_view> values;
  values.reserve(storage.size());
  for (const auto& value : storage) {
    values.push_back(value);
  }

  Buffer buffer{*pool_};
  const auto encoded = EncodingFactory::encode<std::string_view>(
      createSelectionPolicy(), values, buffer);
  auto encoding =
      EncodingFactory().create(*pool_, encoded, createStringBufferFactory());

  std::vector<std::string_view> decoded(values.size());
  encoding->materialize(values.size(), decoded.data());
  ASSERT_EQ(values, decoded);
  const auto pageCount = stringBuffers_.size();
  ASSERT_GT(pageCount, 2);

  for (uint32_t round = 0; round < 100; ++round) {
    encoding->reset();
    if (round % 2 == 0) {
      encoding->materialize(values.size(), decoded.data());
    } else {
      uint32_t offset = 0;
      while (offset < values.size()) {
        const uint32_t count =
            std::min<uint32_t>(1 + offset % 7, values.size() - offset);
        encoding->materialize(count, decoded.data() + offset);
        offset += count;
      }
    }
    ASSERT_EQ(values, decoded) << "round=" << round;
    ASSERT_EQ(pageCount, stringBuffers_.size()) << "round=" << round;
  }
}

// Test with varying number of restarts (restart interval = 16)
// numRestarts = ceil(numValues / restartInterval)
TEST_F(PrefixEncodingTest, varyingNumRestarts) {
  struct TestCase {
    std::string_view name;
    uint32_t numValues;
    uint32_t expectedRestarts;
  };

  const std::vector<TestCase> testCases = {
      {"1 value, 1 restart", 1, 1},
      {"16 values, 1 restart", 16, 1},
      {"17 values, 2 restarts", 17, 2},
      {"32 values, 2 restarts", 32, 2},
      {"33 values, 3 restarts", 33, 3},
      {"48 values, 3 restarts", 48, 3},
      {"50 values, 4 restarts", 50, 4},
      {"100 values, 7 restarts", 100, 7},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);

    // Generate sorted string values with common prefix
    std::vector<std::string> stringStorage;
    std::vector<std::string_view> values;
    stringStorage.reserve(testCase.numValues);
    values.reserve(testCase.numValues);

    for (uint32_t i = 0; i < testCase.numValues; ++i) {
      stringStorage.push_back("prefix_" + folly::to<std::string>(i));
    }
    std::sort(stringStorage.begin(), stringStorage.end());
    for (const auto& s : stringStorage) {
      values.push_back(s);
    }

    Buffer buffer{*pool_};

    auto encoded = EncodingFactory::encode<std::string_view>(
        createSelectionPolicy(), values, buffer);
    stringBuffers_.clear();
    auto encoding =
        EncodingFactory().create(*pool_, encoded, createStringBufferFactory());

    EXPECT_EQ(encoding->rowCount(), values.size());
    EXPECT_EQ(encoding->encodingType(), EncodingType::Prefix);

    // Verify debug string contains expected restart count
    const std::string debug = encoding->debugString();
    EXPECT_TRUE(
        debug.find(
            "num_restarts=" +
            folly::to<std::string>(testCase.expectedRestarts)) !=
        std::string::npos)
        << "Debug: " << debug;

    // Verify all values decode correctly
    std::vector<std::string_view> decoded(values.size());
    encoding->materialize(values.size(), decoded.data());

    for (size_t i = 0; i < values.size(); ++i) {
      EXPECT_EQ(decoded[i], values[i]) << "Mismatch at index " << i;
    }

    // Verify values again after reset
    encoding->reset();
    decoded.clear();
    decoded.resize(values.size());
    encoding->materialize(values.size(), decoded.data());
    for (size_t i = 0; i < values.size(); ++i) {
      EXPECT_EQ(decoded[i], values[i]) << "Mismatch after reset at index " << i;
    }
  }
}

// Test with a sequence of skip/materialize steps.
// Each step specifies skipCount and materializeCount.
// Zero skipCount means no skip, zero materializeCount means no materialize.
TEST_F(PrefixEncodingTest, skipAndMaterializeSteps) {
  struct Step {
    uint32_t skipCount;
    uint32_t materializeCount;
  };

  struct TestCase {
    std::string_view name;
    uint32_t numValues;
    std::vector<Step> steps;
  };

  const std::vector<TestCase> testCases = {
      // Single restart (<=16 values)
      {"single restart, exhaust all",
       10,
       {{0, 3},
        {2, 2},
        {0, 3}}}, // materialize 3, skip 2, materialize 2, materialize 3 = 10
      {"single restart, stop in middle",
       10,
       {{0, 3},
        {2, 2}}}, // materialize 3, skip 2, materialize 2 = 7, stop early
      {"single restart, skip only",
       10,
       {{5, 0}, {0, 5}}}, // skip 5, materialize 5 = 10
      {"single restart, alternating",
       16,
       {{1, 2}, {1, 2}, {1, 2}, {1, 2}, {0, 4}}}, // (skip 1, mat 2) x 4 + mat 4
                                                  // = 16

      // Multiple restarts (>16 values, restart interval = 16)
      {"two restarts, exhaust all",
       32,
       {{0, 10}, {5, 10}, {0, 7}}}, // mat 10, skip 5, mat 10, mat 7 = 32
      {"two restarts, cross restart boundary",
       32,
       {{0, 14}, {0, 4}, {0, 14}}}, // mat 14, mat 4 (crosses 16), mat 14 = 32
      {"three restarts, skip across restarts",
       50,
       {{20, 5},
        {10, 10},
        {0,
         5}}}, // skip 20 (crosses restart), mat 5, skip 10, mat 10, mat 5 = 50
      {"three restarts, stop in middle",
       50,
       {{0, 10},
        {25,
         5}}}, // mat 10, skip 25 (crosses two restarts), mat 5 = 40, stop early
      {"four restarts, many small ops",
       64,
       {{2, 4}, {3, 5}, {4, 6}, {5, 7}, {6, 8}, {7, 7}}}, // various small ops
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);

    // Generate sorted string values
    std::vector<std::string> stringStorage;
    std::vector<std::string_view> values;
    stringStorage.reserve(testCase.numValues);
    values.reserve(testCase.numValues);

    for (uint32_t i = 0; i < testCase.numValues; ++i) {
      stringStorage.push_back("value_" + folly::to<std::string>(i));
    }
    std::sort(stringStorage.begin(), stringStorage.end());
    for (const auto& s : stringStorage) {
      values.push_back(s);
    }

    Buffer buffer{*pool_};

    auto encoded = EncodingFactory::encode<std::string_view>(
        createSelectionPolicy(), values, buffer);
    stringBuffers_.clear();
    auto encoding =
        EncodingFactory().create(*pool_, encoded, createStringBufferFactory());

    // Run through steps twice

    // Run through steps twice (second time after reset) to verify repeatability
    for (int round = 0; round < 2; ++round) {
      SCOPED_TRACE(fmt::format("round {}", round));

      if (round > 0) {
        encoding->reset();
      }

      uint32_t currentRow = 0;
      for (size_t stepIdx = 0; stepIdx < testCase.steps.size(); ++stepIdx) {
        const auto& step = testCase.steps[stepIdx];
        SCOPED_TRACE(fmt::format("step {}", stepIdx));

        // Skip if skipCount > 0
        if (step.skipCount > 0) {
          encoding->skip(step.skipCount);
          currentRow += step.skipCount;
        }

        // Materialize if materializeCount > 0
        if (step.materializeCount > 0) {
          std::vector<std::string_view> decoded(step.materializeCount);
          encoding->materialize(step.materializeCount, decoded.data());

          for (uint32_t i = 0; i < step.materializeCount; ++i) {
            EXPECT_EQ(decoded[i], values[currentRow + i])
                << "Mismatch at row " << (currentRow + i);
          }
          currentRow += step.materializeCount;
        }
      }
    }
  }
}

// Fuzzer-style test with random inputs and random skip/materialize steps.
// Each iteration generates random values and random steps, then runs two passes
// with reset to verify repeatability.
TEST_F(PrefixEncodingTest, fuzzerMaterialize) {
  constexpr uint32_t kNumIterations = 100;
  constexpr uint32_t kMaxNumValues = 200;
  constexpr uint32_t kMinNumValues = 1;
  constexpr uint32_t kMaxNumSteps = 20;

  std::mt19937 rng(42); // Fixed seed for reproducibility

  for (uint32_t iter = 0; iter < kNumIterations; ++iter) {
    SCOPED_TRACE(fmt::format("iteration {}", iter));

    // Random number of values (covers single and multiple restarts)
    std::uniform_int_distribution<uint32_t> numValuesDist(
        kMinNumValues, kMaxNumValues);
    const uint32_t numValues = numValuesDist(rng);

    // Generate sorted string values
    std::vector<std::string> stringStorage;
    std::vector<std::string_view> values;
    stringStorage.reserve(numValues);
    values.reserve(numValues);

    for (uint32_t i = 0; i < numValues; ++i) {
      stringStorage.push_back("key_" + folly::to<std::string>(i));
    }
    std::sort(stringStorage.begin(), stringStorage.end());
    for (const auto& s : stringStorage) {
      values.push_back(s);
    }

    Buffer buffer{*pool_};
    auto encoded = EncodingFactory::encode<std::string_view>(
        createSelectionPolicy(), values, buffer);
    stringBuffers_.clear();
    auto encoding =
        EncodingFactory().create(*pool_, encoded, createStringBufferFactory());

    // Generate random steps
    std::uniform_int_distribution<uint32_t> numStepsDist(1, kMaxNumSteps);
    const uint32_t numSteps = numStepsDist(rng);

    struct Step {
      uint32_t skipCount;
      uint32_t materializeCount;
    };
    std::vector<Step> steps;
    steps.reserve(numSteps);

    uint32_t remainingRows = numValues;
    for (uint32_t s = 0; s < numSteps && remainingRows > 0; ++s) {
      Step step{};

      // Random skip (0 to remaining rows)
      std::uniform_int_distribution<uint32_t> skipDist(0, remainingRows);
      step.skipCount = skipDist(rng);
      remainingRows -= step.skipCount;

      if (remainingRows > 0) {
        // Random materialize (0 to remaining rows)
        std::uniform_int_distribution<uint32_t> matDist(0, remainingRows);
        step.materializeCount = matDist(rng);
        remainingRows -= step.materializeCount;
      }

      steps.push_back(step);
    }

    // Run through steps twice (second time after reset) to verify repeatability
    for (int round = 0; round < 2; ++round) {
      SCOPED_TRACE(fmt::format("round {}", round));

      if (round > 0) {
        encoding->reset();
      }

      uint32_t currentRow = 0;
      for (size_t stepIdx = 0; stepIdx < steps.size(); ++stepIdx) {
        const auto& step = steps[stepIdx];

        // Skip if skipCount > 0
        if (step.skipCount > 0) {
          encoding->skip(step.skipCount);
          currentRow += step.skipCount;
        }

        // Materialize if materializeCount > 0
        if (step.materializeCount > 0) {
          std::vector<std::string_view> decoded(step.materializeCount);
          encoding->materialize(step.materializeCount, decoded.data());

          for (uint32_t i = 0; i < step.materializeCount; ++i) {
            EXPECT_EQ(decoded[i], values[currentRow + i])
                << "Mismatch at row " << (currentRow + i);
          }
          currentRow += step.materializeCount;
        }
      }
    }
  }
}

TEST_F(PrefixEncodingTest, debugString) {
  std::vector<std::string_view> values = {"apple", "banana", "cherry"};

  Buffer buffer{*pool_};

  auto encoded = EncodingFactory::encode<std::string_view>(
      createSelectionPolicy(), values, buffer);
  stringBuffers_.clear();
  auto encoding =
      EncodingFactory().create(*pool_, encoded, createStringBufferFactory());

  std::string debug = encoding->debugString();
  EXPECT_TRUE(debug.find("Prefix") != std::string::npos);
  EXPECT_TRUE(debug.find("restart_interval=") != std::string::npos);
  EXPECT_TRUE(debug.find("num_restarts=") != std::string::npos);
}

// Test that custom restart interval from EncodingLayout config is respected.
TEST_F(PrefixEncodingTest, customRestartInterval) {
  std::vector<std::string_view> values = {
      "apple", "application", "apply", "banana", "bandana", "band"};

  // nullopt tests that the default interval is used when no config is provided
  const std::vector<std::optional<uint32_t>> customRestartIntervals = {
      std::nullopt, 1, 2, 3, 4, 5, 6, 10, 16, 32};

  for (const auto& customRestartInterval : customRestartIntervals) {
    SCOPED_TRACE(
        fmt::format(
            "restart_interval={}",
            customRestartInterval.has_value()
                ? std::to_string(customRestartInterval.value())
                : "default"));

    Buffer buffer{*pool_};

    // Create EncodingLayout with optional custom restart interval in config
    std::unordered_map<std::string, std::string> configMap;
    if (customRestartInterval.has_value()) {
      configMap[std::string(PrefixEncoding::kRestartIntervalConfigKey)] =
          std::to_string(customRestartInterval.value());
    }

    EncodingLayout layout{
        EncodingType::Prefix,
        EncodingLayout::Config{std::move(configMap)},
        CompressionType::Uncompressed};

    auto policy =
        std::make_unique<ReplayedEncodingSelectionPolicy<std::string_view>>(
            std::move(layout),
            CompressionOptions{},
            encodingSelectionPolicyCreator_);

    auto encoded = EncodingFactory::encode<std::string_view>(
        std::move(policy), values, buffer);
    stringBuffers_.clear();
    auto encoding =
        EncodingFactory().create(*pool_, encoded, createStringBufferFactory());

    // Verify restart interval is correctly set from config or default
    const std::string debug = encoding->debugString();
    const uint32_t expectedInterval =
        customRestartInterval.value_or(PrefixEncoding::kDefaultRestartInterval);
    EXPECT_TRUE(
        debug.find("restart_interval=" + std::to_string(expectedInterval)) !=
        std::string::npos)
        << "Debug: " << debug;
  }
}

// Test that invalid restart intervals (0 or negative) throw an exception.
TEST_F(PrefixEncodingTest, invalidRestartIntervalThrows) {
  const std::vector<std::string_view> values = {"apple", "banana", "cherry"};

  for (const auto invalidInterval : {0, -1, -100}) {
    SCOPED_TRACE(fmt::format("restart_interval={}", invalidInterval));

    Buffer buffer{*pool_};

    std::unordered_map<std::string, std::string> configMap;
    configMap[std::string(PrefixEncoding::kRestartIntervalConfigKey)] =
        std::to_string(invalidInterval);

    EncodingLayout layout{
        EncodingType::Prefix,
        EncodingLayout::Config{std::move(configMap)},
        CompressionType::Uncompressed};

    auto policy =
        std::make_unique<ReplayedEncodingSelectionPolicy<std::string_view>>(
            std::move(layout),
            CompressionOptions{},
            encodingSelectionPolicyCreator_);

    NIMBLE_ASSERT_THROW(
        EncodingFactory::encode<std::string_view>(
            std::move(policy), values, buffer),
        "Restart interval must be greater than 0");
  }
}

// Test encode with various input patterns and verify with materialize.
// This test generates different input patterns and verifies that encode/decode
// round-trip produces correct results. Also tests restart interval boundary
// conditions.
TEST_F(PrefixEncodingTest, encode) {
  struct TestCase {
    std::string_view name;
    std::function<std::vector<std::string>(uint32_t)> generator;
    uint32_t numValues;
  };

  // Generator: sequential numbers with common prefix
  auto sequentialWithPrefix = [](uint32_t n) {
    std::vector<std::string> result;
    result.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
      result.push_back(fmt::format("prefix_{:06d}", i));
    }
    std::sort(result.begin(), result.end());
    return result;
  };

  // Generator: varying prefix lengths
  auto varyingPrefixLengths = [](uint32_t n) {
    std::vector<std::string> result;
    result.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
      // Create strings with different prefix patterns
      std::string prefix(i % 20 + 1, 'a'); // Prefix length 1-20
      result.push_back(fmt::format("{}_{:04d}", prefix, i));
    }
    std::sort(result.begin(), result.end());
    return result;
  };

  // Generator: strings with long common prefix
  auto longCommonPrefix = [](uint32_t n) {
    std::vector<std::string> result;
    result.reserve(n);
    std::string commonPrefix(100, 'x'); // 100-char common prefix
    for (uint32_t i = 0; i < n; ++i) {
      result.push_back(fmt::format("{}_{:06d}", commonPrefix, i));
    }
    std::sort(result.begin(), result.end());
    return result;
  };

  // Generator: strings with no common prefix
  auto noCommonPrefix = [](uint32_t n) {
    std::vector<std::string> result;
    result.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
      // Use different starting characters
      char startChar = 'a' + (i % 26);
      result.push_back(fmt::format("{}_{:06d}", startChar, i));
    }
    std::sort(result.begin(), result.end());
    return result;
  };

  // Generator: mixed empty and non-empty strings
  auto mixedEmpty = [](uint32_t n) {
    std::vector<std::string> result;
    result.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
      if (i % 5 == 0) {
        result.emplace_back("");
      } else {
        result.push_back(fmt::format("str_{:04d}", i));
      }
    }
    std::sort(result.begin(), result.end());
    return result;
  };

  // Generator: varying prefix patterns
  auto varyingPrefixPattern = [](uint32_t n) {
    std::vector<std::string> result;
    result.reserve(n);
    const int patternLength = velox::bits::divRoundUp(n, 26);
    for (uint32_t i = 0; i < n; ++i) {
      // Create strings with different prefix patterns.
      const std::string prefix(4, 'a' + (i / patternLength) % 26);
      result.push_back(fmt::format("{}_{:04d}", prefix, i));
    }
    std::sort(result.begin(), result.end());
    return result;
  };

  // Test cases with different sizes to cover restart interval boundaries
  // Restart interval is 16, so we test: <16, =16, >16, multiples, etc.
  const std::vector<TestCase> testCases = {
      // Single restart (<=16 values)
      {"sequential, 1 value", sequentialWithPrefix, 1},
      {"sequential, 8 values", sequentialWithPrefix, 8},
      {"sequential, 15 values (boundary-1)", sequentialWithPrefix, 15},
      {"sequential, 16 values (exact boundary)", sequentialWithPrefix, 16},

      // Two restarts (17-32 values)
      {"sequential, 17 values (boundary+1)", sequentialWithPrefix, 17},
      {"sequential, 24 values", sequentialWithPrefix, 24},
      {"sequential, 31 values (boundary-1)", sequentialWithPrefix, 31},
      {"sequential, 32 values (exact boundary)", sequentialWithPrefix, 32},

      // Three restarts (33-48 values)
      {"sequential, 33 values (boundary+1)", sequentialWithPrefix, 33},
      {"sequential, 47 values (boundary-1)", sequentialWithPrefix, 47},
      {"sequential, 48 values (exact boundary)", sequentialWithPrefix, 48},

      // Four restarts (49-64 values)
      {"sequential, 49 values (boundary+1)", sequentialWithPrefix, 49},
      {"sequential, 63 values (boundary-1)", sequentialWithPrefix, 63},
      {"sequential, 64 values (exact boundary)", sequentialWithPrefix, 64},
      {"sequential, 65 values (boundary+1)", sequentialWithPrefix, 65},

      // Multiple restarts - larger sizes
      {"sequential, 100 values (7 restarts)", sequentialWithPrefix, 100},

      // Different patterns with varying sizes
      {"varying prefix, 50 values", varyingPrefixLengths, 50},
      {"varying prefix pattern, 50 values", varyingPrefixPattern, 50},
      {"long common prefix, 50 values", longCommonPrefix, 50},
      {"no common prefix, 50 values", noCommonPrefix, 50},
      {"mixed empty, 50 values", mixedEmpty, 50},

      // Large datasets
      {"sequential, 200 values (13 restarts)", sequentialWithPrefix, 200},
      {"varying prefix, 200 values", varyingPrefixLengths, 200},
      {"varying prefix pattern, 200 values", varyingPrefixPattern, 200},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);

    // Generate input data
    std::vector<std::string> stringStorage =
        testCase.generator(testCase.numValues);
    std::vector<std::string_view> values;
    values.reserve(stringStorage.size());
    for (const auto& s : stringStorage) {
      values.push_back(s);
    }

    Buffer buffer{*pool_};

    // Encode
    auto encoded = EncodingFactory::encode<std::string_view>(
        createSelectionPolicy(), values, buffer);
    stringBuffers_.clear();
    auto encoding =
        EncodingFactory().create(*pool_, encoded, createStringBufferFactory());

    // Verify basic properties
    EXPECT_EQ(encoding->rowCount(), values.size());
    EXPECT_EQ(encoding->encodingType(), EncodingType::Prefix);

    // Verify restart count
    const uint32_t expectedRestarts = (testCase.numValues + 15) / 16;
    const std::string debug = encoding->debugString();
    EXPECT_TRUE(
        debug.find(
            "num_restarts=" + folly::to<std::string>(expectedRestarts)) !=
        std::string::npos)
        << "Debug: " << debug << ", expected restarts: " << expectedRestarts;

    // Materialize all values and verify
    std::vector<std::string_view> decoded(values.size());
    encoding->materialize(values.size(), decoded.data());

    for (size_t i = 0; i < values.size(); ++i) {
      EXPECT_EQ(decoded[i], values[i]) << "Mismatch at index " << i;
    }

    // Reset and verify again
    encoding->reset();
    decoded.clear();
    decoded.resize(values.size());
    encoding->materialize(values.size(), decoded.data());

    for (size_t i = 0; i < values.size(); ++i) {
      EXPECT_EQ(decoded[i], values[i]) << "Mismatch after reset at index " << i;
    }

    // Test accessing values at restart boundaries
    // Restart points are at indices: 0, 16, 32, 48, ...
    for (uint32_t restartIdx = 0; restartIdx < expectedRestarts; ++restartIdx) {
      const uint32_t restartPos = restartIdx * 16;
      if (restartPos >= testCase.numValues) {
        break;
      }

      // Reset and skip to restart position, then materialize
      encoding->reset();
      if (restartPos > 0) {
        encoding->skip(restartPos);
      }

      // Materialize values starting from restart position
      const uint32_t toMaterialize =
          std::min(16u, testCase.numValues - restartPos);
      std::vector<std::string_view> restartDecoded(toMaterialize);
      encoding->materialize(toMaterialize, restartDecoded.data());

      for (uint32_t i = 0; i < toMaterialize; ++i) {
        EXPECT_EQ(restartDecoded[i], values[restartPos + i])
            << "Mismatch at restart " << restartIdx << " offset " << i;
      }
    }

    // Test materialize across restart boundaries with varying chunk sizes
    encoding->reset();
    const std::vector<uint32_t> chunkSizes = {10, 15, 20, 25};
    uint32_t currentRow = 0;
    for (uint32_t chunkSize : chunkSizes) {
      if (currentRow >= testCase.numValues) {
        break;
      }
      const uint32_t toRead =
          std::min(chunkSize, testCase.numValues - currentRow);
      std::vector<std::string_view> chunkDecoded(toRead);
      encoding->materialize(toRead, chunkDecoded.data());

      for (uint32_t i = 0; i < toRead; ++i) {
        EXPECT_EQ(chunkDecoded[i], values[currentRow + i])
            << "Cross-boundary mismatch at row " << (currentRow + i);
      }
      currentRow += toRead;
    }
  }
}

// Fuzzer-style encode test with random data generation.
// Tests encode/decode round-trip with randomly generated string data.
TEST_F(PrefixEncodingTest, fuzzerEncode) {
  constexpr uint32_t kNumIterations = 50;
  constexpr uint32_t kMaxNumValues = 200;
  constexpr uint32_t kMinNumValues = 1;
  constexpr uint32_t kMaxStringLength = 100;

  std::mt19937 rng(12345); // Fixed seed for reproducibility

  for (uint32_t iter = 0; iter < kNumIterations; ++iter) {
    SCOPED_TRACE(fmt::format("iteration {}", iter));

    // Random number of values
    std::uniform_int_distribution<uint32_t> numValuesDist(
        kMinNumValues, kMaxNumValues);
    const uint32_t numValues = numValuesDist(rng);

    // Generate random strings
    std::vector<std::string> stringStorage;
    stringStorage.reserve(numValues);

    std::uniform_int_distribution<uint32_t> lengthDist(0, kMaxStringLength);
    std::uniform_int_distribution<char> charDist('a', 'z');

    for (uint32_t i = 0; i < numValues; ++i) {
      const uint32_t len = lengthDist(rng);
      std::string s;
      s.reserve(len);
      for (uint32_t j = 0; j < len; ++j) {
        s.push_back(charDist(rng));
      }
      stringStorage.push_back(std::move(s));
    }

    // Sort to ensure valid input for prefix encoding
    std::sort(stringStorage.begin(), stringStorage.end());

    std::vector<std::string_view> values;
    values.reserve(stringStorage.size());
    for (const auto& s : stringStorage) {
      values.push_back(s);
    }

    Buffer buffer{*pool_};

    // Encode
    auto encoded = EncodingFactory::encode<std::string_view>(
        createSelectionPolicy(), values, buffer);
    stringBuffers_.clear();
    auto encoding =
        EncodingFactory().create(*pool_, encoded, createStringBufferFactory());

    // Verify properties
    EXPECT_EQ(encoding->rowCount(), values.size());
    EXPECT_EQ(encoding->encodingType(), EncodingType::Prefix);

    // Verify restart count
    const uint32_t expectedRestarts = (numValues + 15) / 16;
    const std::string debug = encoding->debugString();
    EXPECT_TRUE(
        debug.find(
            "num_restarts=" + folly::to<std::string>(expectedRestarts)) !=
        std::string::npos)
        << "Debug: " << debug;

    // Materialize and verify all values
    std::vector<std::string_view> decoded(values.size());
    encoding->materialize(values.size(), decoded.data());

    for (size_t i = 0; i < values.size(); ++i) {
      EXPECT_EQ(decoded[i], values[i]) << "Mismatch at index " << i;
    }

    // Test partial materialize across restart boundaries
    encoding->reset();
    uint32_t currentRow = 0;
    while (currentRow < values.size()) {
      // Random batch size (1 to remaining rows, max 30)
      const uint32_t remaining = values.size() - currentRow;
      std::uniform_int_distribution<uint32_t> batchDist(
          1, std::min(remaining, 30u));
      const uint32_t batchSize = batchDist(rng);

      std::vector<std::string_view> batch(batchSize);
      encoding->materialize(batchSize, batch.data());

      for (uint32_t i = 0; i < batchSize; ++i) {
        EXPECT_EQ(batch[i], values[currentRow + i])
            << "Batch mismatch at row " << (currentRow + i);
      }
      currentRow += batchSize;
    }
  }
}

TEST_F(PrefixEncodingTest, encodeLargeInputRoundTrip) {
  constexpr uint32_t kNumValues = 500'000;

  std::vector<std::string> keyStrings;
  keyStrings.reserve(kNumValues);
  for (uint32_t i = 0; i < kNumValues; ++i) {
    keyStrings.push_back(fmt::format("key_{:010d}_suffix_{:05d}", i / 100, i));
  }
  std::vector<std::string_view> values;
  values.reserve(kNumValues);
  for (const auto& s : keyStrings) {
    values.push_back(s);
  }

  Buffer buffer{*pool_};

  const auto startTime = std::chrono::steady_clock::now();
  auto encoded = EncodingFactory::encode<std::string_view>(
      createSelectionPolicy(), values, buffer);
  const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::steady_clock::now() - startTime)
                           .count();
  EXPECT_FALSE(encoded.empty());
  // Without the reserve() fix, encoding 500K values takes 600+ seconds due to
  // O(n^2) reallocations. With the fix it completes in under a second. Use a
  // generous 120s limit to avoid flakes under ASAN/TSAN/loaded machines.
  EXPECT_LT(elapsed, 120) << "Encoding " << kNumValues << " values took "
                          << elapsed << "s";

  stringBuffers_.clear();
  auto encoding =
      EncodingFactory().create(*pool_, encoded, createStringBufferFactory());
  EXPECT_EQ(encoding->rowCount(), kNumValues);

  std::vector<std::string_view> decoded(kNumValues);
  encoding->materialize(kNumValues, decoded.data());
  for (uint32_t i = 0; i < kNumValues; ++i) {
    EXPECT_EQ(decoded[i], values[i]) << "Mismatch at row " << i;
  }
}

} // namespace facebook::nimble::test
