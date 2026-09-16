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

#include <random>
#include <thread>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/NimbleException.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/index/KeyEncoding.h"

namespace facebook::nimble::index {
namespace {

class KeyEncodingTest : public ::testing::TestWithParam<EncodingType> {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addRootPool("KeyEncodingTest");
    leafPool_ = pool_->addLeafChild("leaf");
  }

  // Pins the key-stream encoding the way ClusterIndexWriter does. A
  // selection policy would not: EncodingSizeEstimation has no Prefix case
  // for std::string_view, so it reports Prefix as incompatible and the
  // policy silently falls back to Trivial.
  static EncodingLayout keyEncodingLayout(EncodingType encodingType) {
    if (encodingType == EncodingType::Prefix) {
      return EncodingLayout{
          EncodingType::Prefix, {}, CompressionType::Uncompressed};
    }
    return EncodingLayout{
        EncodingType::Trivial,
        {},
        CompressionType::Uncompressed,
        {EncodingLayout{
            EncodingType::Trivial, {}, CompressionType::Uncompressed}}};
  }

  std::unique_ptr<KeyEncoding> createKeyEncoding(
      const std::vector<std::string_view>& keys) {
    Buffer encodingBuffer{*leafPool_};
    auto policy =
        std::make_unique<ReplayedEncodingSelectionPolicy<std::string_view>>(
            keyEncodingLayout(GetParam()),
            CompressionOptions{},
            [](DataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
              return nullptr;
            });
    auto encoded = EncodingFactory::encode<std::string_view>(
        std::move(policy), keys, encodingBuffer);
    encodedData_ = std::string(encoded.data(), encoded.size());
    stringBuffers_.clear();
    auto keyEncoding = KeyEncoding::create(
        *leafPool_, encodedData_, [this](uint32_t totalLength) -> void* {
          auto& buffer = stringBuffers_.emplace_back(
              velox::AlignedBuffer::allocate<char>(
                  totalLength, leafPool_.get()));
          return buffer->asMutable<void>();
        });
    // Guards against the encoding silently degrading to Trivial, which would
    // make the /Prefix instantiation test the same path twice.
    EXPECT_EQ(keyEncoding->encodingType(), GetParam());
    return keyEncoding;
  }

  // Generates sorted string values.
  static std::pair<std::vector<std::string>, std::vector<std::string_view>>
  generateSortedKeys(uint32_t count, uint32_t step = 1) {
    std::vector<std::string> storage;
    storage.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      storage.push_back(fmt::format("key_{:03d}", i * step));
    }
    std::sort(storage.begin(), storage.end());
    std::vector<std::string_view> views;
    views.reserve(count);
    for (const auto& s : storage) {
      views.emplace_back(s);
    }
    return {std::move(storage), std::move(views)};
  }

  // Linear search reference for fuzzer verification.
  static std::optional<uint32_t> linearSearch(
      const std::vector<std::string_view>& values,
      std::string_view target,
      bool inclusive) {
    for (size_t i = 0; i < values.size(); ++i) {
      const bool found = inclusive ? values[i] >= target : values[i] > target;
      if (found) {
        return static_cast<uint32_t>(i);
      }
    }
    return std::nullopt;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
  std::string encodedData_;
  std::vector<velox::BufferPtr> stringBuffers_;
};

// ---------------------------------------------------------------------------
// seek() tests
// ---------------------------------------------------------------------------

TEST_P(KeyEncodingTest, seekExactMatch) {
  struct TestParam {
    std::string_view name;
    uint32_t numValues;
    std::vector<uint32_t> seekPositions;
    std::string debugString() const {
      return std::string(name);
    }
  };

  const std::vector<TestParam> testParams = {
      {"small (10 values)", 10, {0, 3, 5, 7, 9}},
      {"boundary (16 values)", 16, {0, 4, 8, 12, 15}},
      {"two restarts (17 values)", 17, {0, 8, 15, 16}},
      {"two restarts (32 values)", 32, {0, 5, 10, 15, 16, 20, 25, 31}},
      {"three restarts (48 values)", 48, {0, 7, 15, 16, 24, 31, 32, 40, 47}},
      {"many restarts (100 values)",
       100,
       {0, 7, 15, 16, 24, 31, 32, 40, 47, 48, 55, 63, 64, 72, 80, 90, 99}},
  };

  for (const auto& param : testParams) {
    SCOPED_TRACE(param.debugString());

    auto [storage, values] = generateSortedKeys(param.numValues);
    auto keyEncoding = createKeyEncoding(values);

    for (bool inclusive : {true, false}) {
      SCOPED_TRACE(fmt::format("inclusive={}", inclusive));

      for (uint32_t pos : param.seekPositions) {
        SCOPED_TRACE(fmt::format("seek position {}", pos));
        auto result = keyEncoding->seek(values[pos], inclusive);
        if (inclusive) {
          ASSERT_TRUE(result.has_value());
          EXPECT_EQ(result.value(), pos);
        } else {
          if (pos + 1 < param.numValues) {
            ASSERT_TRUE(result.has_value());
            EXPECT_EQ(result.value(), pos + 1);
          } else {
            EXPECT_FALSE(result.has_value());
          }
        }
      }

      {
        SCOPED_TRACE("value larger than max");
        auto result = keyEncoding->seek("zzz_larger_than_all", inclusive);
        EXPECT_FALSE(result.has_value());
      }
    }
  }
}

TEST_P(KeyEncodingTest, seekNonExactMatch) {
  struct TestParam {
    std::string_view name;
    uint32_t numValues;
    std::vector<uint32_t> seekBeforePositions;
    std::string debugString() const {
      return std::string(name);
    }
  };

  const std::vector<TestParam> testParams = {
      {"small (10 values)", 10, {1, 3, 5, 7, 9}},
      {"boundary (16 values)", 16, {1, 4, 8, 12, 15}},
      {"two restarts (17 values)", 17, {1, 8, 15, 16}},
      {"two restarts (32 values)", 32, {1, 5, 10, 15, 16, 20, 25, 31}},
      {"many restarts (100 values)",
       100,
       {1, 7, 15, 16, 24, 31, 32, 40, 47, 48, 55, 63, 64, 72, 80, 90, 99}},
  };

  for (const auto& param : testParams) {
    SCOPED_TRACE(param.debugString());

    // Generate with gaps (step=2) so we can seek for non-existing keys.
    auto [storage, values] = generateSortedKeys(param.numValues, /*step=*/2);
    auto keyEncoding = createKeyEncoding(values);

    for (bool inclusive : {true, false}) {
      SCOPED_TRACE(fmt::format("inclusive={}", inclusive));

      for (uint32_t pos : param.seekBeforePositions) {
        SCOPED_TRACE(fmt::format("seek before position {}", pos));
        auto targetKey = fmt::format("key_{:03d}", pos * 2 - 1);
        auto result = keyEncoding->seek(targetKey, inclusive);
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result.value(), pos);
      }

      {
        SCOPED_TRACE("value before all");
        auto result = keyEncoding->seek("aaa_before_all", inclusive);
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result.value(), 0);
      }

      {
        SCOPED_TRACE("value larger than max");
        auto result = keyEncoding->seek("zzz_larger_than_all", inclusive);
        EXPECT_FALSE(result.has_value());
      }
    }
  }
}

TEST_P(KeyEncodingTest, seekBeforeAll) {
  std::vector<std::string_view> values = {"banana", "cherry", "date"};
  auto keyEncoding = createKeyEncoding(values);

  for (bool inclusive : {true, false}) {
    SCOPED_TRACE(fmt::format("inclusive={}", inclusive));
    auto result = keyEncoding->seek("apple", inclusive);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 0);
  }
}

TEST_P(KeyEncodingTest, seekAfterAll) {
  std::vector<std::string_view> values = {"apple", "banana", "cherry"};
  auto keyEncoding = createKeyEncoding(values);

  for (bool inclusive : {true, false}) {
    SCOPED_TRACE(fmt::format("inclusive={}", inclusive));
    auto result = keyEncoding->seek("zucchini", inclusive);
    EXPECT_FALSE(result.has_value());
  }
}

TEST_P(KeyEncodingTest, seekWithDuplicateValues) {
  std::vector<std::string_view> values = {
      "aaa", "aaa", "bbb", "bbb", "bbb", "ccc", "ddd", "ddd"};
  auto keyEncoding = createKeyEncoding(values);

  auto result = keyEncoding->seek("aaa", /*inclusive=*/true);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 0);

  result = keyEncoding->seek("aaa", /*inclusive=*/false);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 2);

  result = keyEncoding->seek("bbb", /*inclusive=*/true);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 2);

  result = keyEncoding->seek("bbb", /*inclusive=*/false);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 5);

  result = keyEncoding->seek("ddd", /*inclusive=*/false);
  EXPECT_FALSE(result.has_value());
}

TEST_P(KeyEncodingTest, fuzzerSeek) {
  constexpr uint32_t kNumIterations = 100;
  constexpr uint32_t kSeeksPerIteration = 20;
  const std::vector<uint32_t> numValuesCases = {5, 16, 17, 32, 50, 100, 200};

  std::mt19937 rng(42);

  for (uint32_t iter = 0; iter < kNumIterations; ++iter) {
    SCOPED_TRACE(fmt::format("iteration {}", iter));

    std::uniform_int_distribution<size_t> sizeDist(
        0, numValuesCases.size() - 1);
    const uint32_t numValues = numValuesCases[sizeDist(rng)];

    auto [storage, values] = generateSortedKeys(numValues, /*step=*/2);
    auto keyEncoding = createKeyEncoding(values);

    for (bool inclusive : {true, false}) {
      SCOPED_TRACE(fmt::format("inclusive={}", inclusive));
      for (uint32_t s = 0; s < kSeeksPerIteration; ++s) {
        // Generate random seek target that may or may not exist in values
        std::uniform_int_distribution<uint32_t> keyDist(0, numValues * 2 + 10);
        auto targetKey = fmt::format("key_{:03d}", keyDist(rng));
        auto result = keyEncoding->seek(targetKey, inclusive);
        auto expected = linearSearch(values, targetKey, inclusive);
        EXPECT_EQ(result, expected)
            << "target=" << targetKey << " inclusive=" << inclusive;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// get() tests
// ---------------------------------------------------------------------------

TEST_P(KeyEncodingTest, getByRow) {
  auto [storage, values] = generateSortedKeys(50);
  auto keyEncoding = createKeyEncoding(values);

  for (uint32_t i = 0; i < values.size(); ++i) {
    SCOPED_TRACE(fmt::format("row={}", i));
    EXPECT_EQ(keyEncoding->get(i), values[i]);
  }
}

TEST_P(KeyEncodingTest, getByRowRepeatedCalls) {
  std::vector<std::string_view> values = {"alpha", "beta", "gamma"};
  auto keyEncoding = createKeyEncoding(values);

  for (int repeat = 0; repeat < 3; ++repeat) {
    EXPECT_EQ(keyEncoding->get(1), "beta");
  }
  EXPECT_EQ(keyEncoding->get(0), "alpha");
  EXPECT_EQ(keyEncoding->get(2), "gamma");
}

TEST_P(KeyEncodingTest, getSingleRow) {
  std::vector<std::string_view> values = {"only"};
  auto keyEncoding = createKeyEncoding(values);
  EXPECT_EQ(keyEncoding->get(0), "only");
}

// ---------------------------------------------------------------------------
// materialize() tests
// ---------------------------------------------------------------------------

TEST_P(KeyEncodingTest, materializeFullRange) {
  std::vector<std::string_view> values = {"aaa", "bbb", "ccc", "ddd", "eee"};
  auto keyEncoding = createKeyEncoding(values);

  auto result = keyEncoding->materialize(0, 5);
  ASSERT_EQ(result.size(), 5);
  for (uint32_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(result[i], values[i]);
  }
}

TEST_P(KeyEncodingTest, materializeSubRange) {
  auto [storage, values] = generateSortedKeys(50);
  auto keyEncoding = createKeyEncoding(values);

  auto result = keyEncoding->materialize(10, 5);
  ASSERT_EQ(result.size(), 5);
  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(result[i], values[10 + i]);
  }
}

TEST_P(KeyEncodingTest, materializeEmptyRangeAtEnd) {
  // startRow == rowCount with count == 0 is the natural way to ask for
  // "nothing left", and both encodings must answer it the same way.
  auto [storage, values] = generateSortedKeys(50);
  auto keyEncoding = createKeyEncoding(values);
  ASSERT_EQ(keyEncoding->rowCount(), 50);

  EXPECT_TRUE(keyEncoding->materialize(50, 0).empty());
  EXPECT_TRUE(keyEncoding->materialize(17, 0).empty());
}

TEST_P(KeyEncodingTest, materializeSingleRow) {
  std::vector<std::string_view> values = {"aaa", "bbb", "ccc"};
  auto keyEncoding = createKeyEncoding(values);

  auto result = keyEncoding->materialize(1, 1);
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "bbb");
}

// ---------------------------------------------------------------------------
// cursor() tests
// ---------------------------------------------------------------------------

TEST_P(KeyEncodingTest, cursorFullScan) {
  auto [storage, values] = generateSortedKeys(50);
  auto keyEncoding = createKeyEncoding(values);

  std::vector<std::string> scanned;
  scanned.reserve(values.size());
  auto cursor = keyEncoding->cursor(0);
  for (uint32_t row = 0; row < values.size(); ++row) {
    scanned.emplace_back(cursor->next());
  }
  EXPECT_EQ(scanned, storage);
}

TEST_P(KeyEncodingTest, cursorFromMiddleRow) {
  // 50 keys span several prefix restart blocks, so row 19 starts inside a
  // block rather than on a restart boundary.
  auto [storage, values] = generateSortedKeys(50);
  auto keyEncoding = createKeyEncoding(values);

  constexpr uint32_t kStartRow = 19;
  auto cursor = keyEncoding->cursor(kStartRow);
  for (uint32_t row = kStartRow; row < values.size(); ++row) {
    SCOPED_TRACE(fmt::format("row={}", row));
    EXPECT_EQ(cursor->next(), values[row]);
  }
}

TEST_P(KeyEncodingTest, cursorAcrossRestartBlocks) {
  // Prefix encoding emits a full key every kDefaultRestartInterval rows, so
  // 50 keys span four restart blocks. Start on a block boundary, just after
  // one, and just before the next, and in each case run to the end so the
  // cursor crosses every later boundary.
  constexpr uint32_t kNumKeys = 50;
  auto [storage, values] = generateSortedKeys(kNumKeys);
  auto keyEncoding = createKeyEncoding(values);

  for (const uint32_t startRow : {0u, 1u, 15u, 16u, 17u, 31u, 32u, 49u}) {
    SCOPED_TRACE(fmt::format("startRow={}", startRow));
    auto cursor = keyEncoding->cursor(startRow);
    for (uint32_t row = startRow; row < kNumKeys; ++row) {
      ASSERT_TRUE(cursor->hasNext());
      EXPECT_EQ(cursor->next(), values[row]);
    }
    EXPECT_FALSE(cursor->hasNext());
  }
}

TEST_P(KeyEncodingTest, cursorRefusesToAdvancePastTheEnd) {
  auto [storage, values] = generateSortedKeys(20);
  auto keyEncoding = createKeyEncoding(values);

  auto cursor = keyEncoding->cursor(18);
  EXPECT_TRUE(cursor->hasNext());
  EXPECT_EQ(cursor->next(), values[18]);
  EXPECT_EQ(cursor->next(), values[19]);
  EXPECT_FALSE(cursor->hasNext());
  // Enforced in optimized builds too: over-advancing would read out of
  // bounds rather than fail.
  EXPECT_THROW(cursor->next(), NimbleInternalError);
}

TEST_P(KeyEncodingTest, cursorSingleRow) {
  std::vector<std::string_view> values = {"only"};
  auto keyEncoding = createKeyEncoding(values);

  auto cursor = keyEncoding->cursor(0);
  EXPECT_EQ(cursor->next(), "only");
}

TEST_P(KeyEncodingTest, cursorsAreIndependent) {
  auto [storage, values] = generateSortedKeys(50);
  auto keyEncoding = createKeyEncoding(values);

  auto first = keyEncoding->cursor(0);
  auto second = keyEncoding->cursor(25);
  for (uint32_t row = 0; row < 25; ++row) {
    SCOPED_TRACE(fmt::format("row={}", row));
    EXPECT_EQ(first->next(), values[row]);
    EXPECT_EQ(second->next(), values[25 + row]);
  }
}

TEST_P(KeyEncodingTest, cursorStartRowOutOfRange) {
  std::vector<std::string_view> values = {"aaa", "bbb"};
  auto keyEncoding = createKeyEncoding(values);

  EXPECT_THROW(keyEncoding->cursor(2), NimbleInternalError);
}

// ---------------------------------------------------------------------------
// Concurrent tests
// ---------------------------------------------------------------------------

TEST_P(KeyEncodingTest, concurrentSeek) {
  auto [storage, values] = generateSortedKeys(100);
  auto keyEncoding = createKeyEncoding(values);

  constexpr int kNumThreads = 16;
  constexpr auto kDuration = std::chrono::seconds(3);
  std::atomic_bool stop{false};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&, t] {
      while (!stop.load(std::memory_order_relaxed)) {
        const uint32_t idx = t % values.size();
        auto pos = keyEncoding->seek(values[idx], /*inclusive=*/true);
        ASSERT_TRUE(pos.has_value());
        EXPECT_EQ(pos.value(), idx);
      }
    });
  }

  std::this_thread::sleep_for(kDuration);
  stop.store(true);
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_P(KeyEncodingTest, concurrentGet) {
  auto [storage, values] = generateSortedKeys(100);
  auto keyEncoding = createKeyEncoding(values);

  constexpr int kNumThreads = 16;
  constexpr auto kDuration = std::chrono::seconds(3);
  std::atomic_bool stop{false};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&, t] {
      while (!stop.load(std::memory_order_relaxed)) {
        const uint32_t idx = t % values.size();
        EXPECT_EQ(keyEncoding->get(idx), values[idx]);
      }
    });
  }

  std::this_thread::sleep_for(kDuration);
  stop.store(true);
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_P(KeyEncodingTest, concurrentMaterialize) {
  auto [storage, values] = generateSortedKeys(100);
  auto keyEncoding = createKeyEncoding(values);

  constexpr int kNumThreads = 16;
  constexpr auto kDuration = std::chrono::seconds(3);
  std::atomic_bool stop{false};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&, t] {
      while (!stop.load(std::memory_order_relaxed)) {
        const uint32_t start = (t * 7) % 80;
        const uint32_t count = 20;
        auto entries = keyEncoding->materialize(start, count);
        ASSERT_EQ(entries.size(), count);
        for (uint32_t i = 0; i < count; ++i) {
          EXPECT_EQ(entries[i], values[start + i]);
        }
      }
    });
  }

  std::this_thread::sleep_for(kDuration);
  stop.store(true);
  for (auto& thread : threads) {
    thread.join();
  }
}

INSTANTIATE_TEST_SUITE_P(
    EncodingTypes,
    KeyEncodingTest,
    ::testing::Values(EncodingType::Trivial, EncodingType::Prefix),
    [](const ::testing::TestParamInfo<EncodingType>& info) {
      return info.param == EncodingType::Trivial ? "Trivial" : "Prefix";
    });

} // namespace
} // namespace facebook::nimble::index
