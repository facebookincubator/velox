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
#include "velox/dwio/nimble/index/BloomFilter.h"

#include <array>
#include <limits>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/index/BlockedBloomFilter.h"

namespace facebook::nimble::index {
namespace {

// The serialized format, spelled out independently of the implementation so
// that these tests fail if the bytes on disk ever change shape.
constexpr size_t kTrailerSize{4};
constexpr size_t kBlockSizeBytes{32};
constexpr char kBlockedTrailer[kTrailerSize] = {0x01, 0x00, 0x00, 0x00};

class BloomFilterTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addRootPool("BloomFilterTest");
    leafPool_ = pool_->addLeafChild("leaf");
  }

  // Returns a blocked config at 'bitsPerKey' that tells the builder to expect
  // 'numKeys' keys, or that leaves it to count them when 'numKeys' is unset.
  static BlockedBloomFilterConfig blockedConfig(
      float bitsPerKey,
      std::optional<uint64_t> numKeys) {
    BlockedBloomFilterConfig config{bitsPerKey};
    config.numKeys = numKeys;
    return config;
  }

  // The two ways a builder can learn the size of its filter: from 'numKeys'
  // given up front, or by counting the keys it receives before finish().
  static std::vector<std::optional<uint64_t>> numKeysModes(uint64_t numKeys) {
    return {numKeys, std::nullopt};
  }

  // Builds a filter over 'keys' with 'config' and returns its serialized
  // bytes.
  std::string build(
      const std::vector<std::string>& keys,
      const BlockedBloomFilterConfig& config) {
    auto builder = createBloomFilterBuilder(config, pool());
    for (const auto& key : keys) {
      builder->insert(key);
    }
    return std::string{builder->finish()};
  }

  // Builds a filter over 'keys', telling the builder up front how many there
  // are, and returns its serialized bytes.
  std::string build(
      const std::vector<std::string>& keys,
      float bitsPerKey = 10.0f) {
    return build(keys, blockedConfig(bitsPerKey, keys.size()));
  }

  std::unique_ptr<BloomFilterReader> open(std::string_view serialized) {
    return createBloomFilterReader(serialized, pool());
  }

  // Returns 'count' distinct keys sharing 'prefix'.
  static std::vector<std::string> makeKeys(
      std::string_view prefix,
      size_t count) {
    std::vector<std::string> keys;
    keys.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      keys.emplace_back(std::string{prefix} + std::to_string(i));
    }
    return keys;
  }

  velox::memory::MemoryPool* pool() {
    return leafPool_.get();
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
};

TEST_F(BloomFilterTest, insertedKeysAreFound) {
  const auto keys = makeKeys("key_", 1'000);
  const auto reader = open(build(keys));
  for (const auto& key : keys) {
    ASSERT_TRUE(reader->maybeContains(key)) << key;
  }
}

TEST_F(BloomFilterTest, emptyStringKey) {
  const auto reader = open(build({""}));
  EXPECT_TRUE(reader->maybeContains(""));
  EXPECT_FALSE(reader->maybeContains("notempty"));
}

TEST_F(BloomFilterTest, duplicateInsertsAreIdempotent) {
  // Both filters must be sized identically, or they would differ for a reason
  // that has nothing to do with the repeated inserts.
  auto once = createBloomFilterBuilder(blockedConfig(10.0f, 1'000), pool());
  auto repeated = createBloomFilterBuilder(blockedConfig(10.0f, 1'000), pool());
  once->insert("dup");
  for (int i = 0; i < 3; ++i) {
    repeated->insert("dup");
  }
  EXPECT_EQ(once->finish(), repeated->finish());
}

TEST_F(BloomFilterTest, emptyFilterMatchesNothing) {
  const auto reader = open(build({}));
  EXPECT_FALSE(reader->maybeContains(""));
  EXPECT_FALSE(reader->maybeContains("anything"));
}

TEST_F(BloomFilterTest, sizingAtFinishWithNoKeys) {
  const auto serialized = build({}, blockedConfig(10.0f, std::nullopt));
  EXPECT_EQ(serialized.size(), kBlockSizeBytes + kTrailerSize);

  const auto reader = open(serialized);
  EXPECT_FALSE(reader->maybeContains(""));
  EXPECT_FALSE(reader->maybeContains("anything"));
}

TEST_F(BloomFilterTest, falsePositiveRateNearTarget) {
  constexpr size_t kNumKeys{10'000};
  const auto reader = open(build(makeKeys("key_", kNumKeys)));

  const auto misses = makeKeys("miss_", kNumKeys);
  size_t numFalsePositives{0};
  for (const auto& key : misses) {
    if (reader->maybeContains(key)) {
      ++numFalsePositives;
    }
  }
  // At 10 bits per key the split-block layout gives a little under 1%. The
  // bound is loose enough to stay stable, tight enough to catch a filter that
  // is sized or probed wrongly.
  EXPECT_LT(
      static_cast<double>(numFalsePositives) / static_cast<double>(kNumKeys),
      0.05);
}

TEST_F(BloomFilterTest, moreKeysProduceLargerFilter) {
  EXPECT_LT(
      build(makeKeys("k", 10)).size(), build(makeKeys("k", 10'000)).size());
}

TEST_F(BloomFilterTest, filterSizeMatchesTheSizingFormula) {
  // Expected block counts worked out by hand from "round numKeys * bitsPerKey
  // up to whole 256-bit blocks, and never fewer than one block", not read back
  // from the implementation. Pins the size a given config produces, which is
  // what a reader derives the block count from.
  struct {
    uint64_t numKeys;
    float bitsPerKey;
    size_t expectedNumBlocks;
  } static constexpr kCases[]{
      {0, 10.0f, 1}, // max(0, 256) bits -> 1 block.
      {1'000, 10.0f, 40}, // 10'000 bits / 256 -> 39.06 -> 40.
      {100, 20.0f, 8}, // 2'000 bits / 256 -> 7.81 -> 8.
      {513, 0.5f, 1}, // 256.5 bits truncates to 256 -> exactly 1 block.
  };
  for (const auto& testCase : kCases) {
    SCOPED_TRACE(
        fmt::format(
            "numKeys: {}, bitsPerKey: {}",
            testCase.numKeys,
            testCase.bitsPerKey));
    auto builder = createBloomFilterBuilder(
        blockedConfig(testCase.bitsPerKey, testCase.numKeys), pool());
    EXPECT_EQ(
        builder->finish().size(),
        testCase.expectedNumBlocks * kBlockSizeBytes + kTrailerSize);
  }
}

TEST_F(BloomFilterTest, sizingAtFinishMatchesUpFrontSizing) {
  // A multiple of 128: at 10 bits per key that many keys fill whole 256-bit
  // blocks exactly, so counting even one key too many adds a block.
  const auto keys = makeKeys("key_", 9'984);
  EXPECT_EQ(
      build(keys, blockedConfig(10.0f, std::nullopt)),
      build(keys, blockedConfig(10.0f, keys.size())));
}

TEST_F(BloomFilterTest, sizingAtFinishCollapsesAdjacentRepeats) {
  // Sorted input repeats a key on consecutive rows. Sized at finish(), the
  // filter must count each such key once. The key count is a multiple of 128
  // for the same reason as above.
  const auto keys = makeKeys("key_", 1'024);
  std::vector<std::string> repeated;
  repeated.reserve(2 * keys.size());
  for (const auto& key : keys) {
    repeated.push_back(key);
    repeated.push_back(key);
  }
  EXPECT_EQ(
      build(repeated, blockedConfig(10.0f, std::nullopt)),
      build(keys, blockedConfig(10.0f, keys.size())));
}

TEST_F(BloomFilterTest, moreBitsPerKeyProduceLargerFilter) {
  const auto keys = makeKeys("k", 1'000);
  EXPECT_LT(build(keys, 5.0f).size(), build(keys, 20.0f).size());
}

TEST_F(BloomFilterTest, invalidBitsPerKey) {
  // Rejected when the builder is created, whether or not the count is known,
  // so a bad config fails before any keys are written.
  for (const auto& numKeys : numKeysModes(100)) {
    SCOPED_TRACE(numKeys.has_value() ? "count known" : "count unknown");
    for (const auto bitsPerKey :
         {0.0f,
          -1.0f,
          std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::quiet_NaN()}) {
      SCOPED_TRACE(bitsPerKey);
      NIMBLE_ASSERT_THROW(
          createBloomFilterBuilder(blockedConfig(bitsPerKey, numKeys), pool()),
          "Bloom filter bits per key must be finite and positive");
    }
  }
}

TEST_F(BloomFilterTest, bitsPerKeyTooLargeToSize) {
  // Finite and positive, but it asks for a filter that cannot be addressed.
  // Sizing must reject it rather than overflow into a small one.
  const auto bitsPerKey = std::numeric_limits<float>::max();
  NIMBLE_ASSERT_THROW(
      createBloomFilterBuilder(blockedConfig(bitsPerKey, 1'000'000), pool()),
      "Bloom filter is too large");

  // Without a count, sizing waits for finish(), so that is where it fails.
  auto builder =
      createBloomFilterBuilder(blockedConfig(bitsPerKey, std::nullopt), pool());
  builder->insert("key");
  NIMBLE_ASSERT_THROW(builder->finish(), "Bloom filter is too large");
}

TEST_F(BloomFilterTest, batchAgreesWithSingleKey) {
  const auto keys = makeKeys("key_", 20);
  const auto reader = open(build(keys));

  // Spans more than one prefetch run, and ends on a partial one.
  std::vector<std::string> probes = keys;
  const auto misses = makeKeys("miss_", 20);
  probes.insert(probes.end(), misses.begin(), misses.end());

  std::vector<std::string_view> probeViews{probes.begin(), probes.end()};
  std::vector<bool> expected;
  expected.reserve(probes.size());
  for (const auto& probe : probes) {
    expected.push_back(reader->maybeContains(probe));
  }

  std::array<bool, 40> actual{};
  reader->maybeContains(probeViews, actual);
  EXPECT_THAT(
      std::vector<bool>(actual.begin(), actual.end()),
      ::testing::ElementsAreArray(expected));
}

TEST_F(BloomFilterTest, batchHandlesEmptyAndSingleKey) {
  const auto reader = open(build({"present"}));

  std::array<bool, 1> single{};
  const std::array<std::string_view, 1> singleKey{"present"};
  reader->maybeContains(singleKey, single);
  EXPECT_TRUE(single[0]);

  // An empty batch must leave the output alone rather than fail.
  reader->maybeContains({}, single);
  EXPECT_TRUE(single[0]);
}

TEST_F(BloomFilterTest, serializedFilterRecordsItsType) {
  const auto serialized = build({"a", "b"});
  ASSERT_GT(serialized.size(), kTrailerSize);

  // Compare against the bytes spelled out above rather than against the
  // encoder, so that a change to what gets written has to be made here too.
  EXPECT_EQ(
      serialized.substr(serialized.size() - kTrailerSize),
      std::string_view(kBlockedTrailer, kTrailerSize));
  EXPECT_EQ((serialized.size() - kTrailerSize) % kBlockSizeBytes, 0);
}

TEST_F(BloomFilterTest, readerDoesNotAliasItsInput) {
  const auto keys = makeKeys("key_", 500);
  std::unique_ptr<BloomFilterReader> reader;
  {
    const auto serialized = build(keys);
    reader = open(serialized);
  }
  // The bytes the reader was opened over are gone. It must answer from its own
  // copy, which is what lets a caller drop the enclosing metadata buffer.
  for (const auto& key : keys) {
    ASSERT_TRUE(reader->maybeContains(key)) << key;
  }
}

TEST_F(BloomFilterTest, unknownTypeFailsToOpen) {
  auto serialized = build(makeKeys("key_", 100));
  // Stand in for a filter written by a future version.
  serialized[serialized.size() - kTrailerSize] = 0x7f;

  NIMBLE_ASSERT_THROW(
      open(serialized), "No bloom filter factory is registered for type: 127");
}

TEST_F(BloomFilterTest, malformedTrailerFailsToOpen) {
  // Too short to hold a trailer at all.
  NIMBLE_ASSERT_THROW(open("ab"), "Bloom filter trailer is unreadable");

  // A later revision put something in the reserved bytes.
  auto serialized = build(makeKeys("key_", 100));
  serialized[serialized.size() - 1] = 0x01;
  NIMBLE_ASSERT_THROW(open(serialized), "Bloom filter trailer is unreadable");
}

TEST_F(BloomFilterTest, knownTypeWithMalformedPayloadThrows) {
  // A payload whose layout this build does implement still has to be well
  // formed. Unlike the cases above, these are rejected by the blocked reader
  // rather than by the trailer.
  const std::string trailer(kBlockedTrailer, kTrailerSize);
  NIMBLE_ASSERT_THROW(
      open(std::string(10, '\0') + trailer),
      "Blocked bloom filter payload is not a whole number of blocks");
  NIMBLE_ASSERT_THROW(open(trailer), "Blocked bloom filter payload is empty");
}

// Stands in for an implementation registered from outside this library, such
// as one that is not part of the open source build.
constexpr auto kTestType = static_cast<BloomFilterType>(200);

class CountingBloomFilterReader final : public BloomFilterReader {
 public:
  bool maybeContains(std::string_view key) const override {
    return key == "present";
  }
};

class CountingBloomFilterFactory final : public BloomFilterFactory {
 public:
  BloomFilterType type() const override {
    return kTestType;
  }

  std::unique_ptr<BloomFilterBuilder> createBuilder(
      const BloomFilterConfig&,
      velox::memory::MemoryPool*) const override {
    NIMBLE_FAIL("Not needed by these tests");
  }

  std::unique_ptr<BloomFilterReader> createReader(
      std::string_view,
      velox::memory::MemoryPool*) const override {
    return std::make_unique<CountingBloomFilterReader>();
  }
};

// Names a layout nothing ever registers, unlike kTestType below.
class UnregisteredBloomFilterConfig final : public BloomFilterConfig {
 public:
  UnregisteredBloomFilterConfig()
      : BloomFilterConfig{static_cast<BloomFilterType>(201), 10.0f} {}

  std::unique_ptr<BloomFilterConfig> clone() const override {
    return std::make_unique<UnregisteredBloomFilterConfig>(*this);
  }
};

// Claims the blocked layout while not being the config that layout expects.
class MislabeledBloomFilterConfig final : public BloomFilterConfig {
 public:
  MislabeledBloomFilterConfig()
      : BloomFilterConfig{BloomFilterType::kBlocked, 10.0f} {}

  std::unique_ptr<BloomFilterConfig> clone() const override {
    return std::make_unique<MislabeledBloomFilterConfig>(*this);
  }
};

TEST_F(BloomFilterTest, builderRejectsUnregisteredType) {
  // Unlike the read side, a writer that cannot honor its configured layout
  // must fail rather than quietly produce a filter of some other shape.
  NIMBLE_ASSERT_THROW(
      createBloomFilterBuilder(UnregisteredBloomFilterConfig{}, pool()),
      "No bloom filter factory is registered for type: 201");
}

TEST_F(BloomFilterTest, builderRejectsConfigOfTheWrongType) {
  // The downcast a factory does on its config is checked, so a config whose
  // type byte disagrees with its class is caught rather than reinterpreted.
  VELOX_ASSERT_THROW(
      createBloomFilterBuilder(MislabeledBloomFilterConfig{}, pool()),
      "Failed to cast");
}

TEST_F(BloomFilterTest, cloneKeepsConcreteType) {
  // A caller holding only the base, like the hash index writer setting its
  // entry count, must get back a copy the factory can still downcast.
  const auto original = blockedConfig(7.0f, 123);
  const BloomFilterConfig& base = original;
  const auto copy = base.clone();

  const auto& copied =
      checkedBloomFilterConfig<BlockedBloomFilterConfig>(*copy);
  EXPECT_EQ(copied.type, BloomFilterType::kBlocked);
  EXPECT_FLOAT_EQ(copied.bitsPerKey, 7.0f);
  EXPECT_EQ(copied.numKeys, std::optional<uint64_t>{123});
}

TEST_F(BloomFilterTest, unregisteredTypeReadableOnceRegistered) {
  // A filter carrying a type this build has no factory for cannot be read at
  // all, so an implementation registered from outside this library is what
  // makes those bytes interpretable.
  auto serialized = build(makeKeys("key_", 100));
  serialized[serialized.size() - kTrailerSize] =
      static_cast<char>(static_cast<uint8_t>(kTestType));
  ASSERT_EQ(bloomFilterFactory(kTestType), nullptr);
  NIMBLE_ASSERT_THROW(
      open(serialized), "No bloom filter factory is registered for type: 200");

  // Once the implementation is registered, the same bytes are interpreted.
  registerBloomFilterFactory(
      std::make_shared<const CountingBloomFilterFactory>());
  ASSERT_NE(bloomFilterFactory(kTestType), nullptr);
  const auto reader = open(serialized);
  EXPECT_TRUE(reader->maybeContains("present"));
  EXPECT_FALSE(reader->maybeContains("definitely absent"));

  // This reader overrides only the single key form, so the batch call lands on
  // the base class implementation that every other reader inherits.
  const std::array<std::string_view, 2> keys{"present", "absent"};
  std::array<bool, 2> results{};
  reader->maybeContains(keys, results);
  EXPECT_THAT(results, ::testing::ElementsAre(true, false));

  NIMBLE_ASSERT_THROW(
      registerBloomFilterFactory(
          std::make_shared<const CountingBloomFilterFactory>()),
      "Bloom filter factory is already registered for type");
}

TEST_F(BloomFilterTest, builderRejectsUseAfterFinish) {
  for (const auto& numKeys : numKeysModes(10)) {
    SCOPED_TRACE(numKeys.has_value() ? "count known" : "count unknown");
    auto builder =
        createBloomFilterBuilder(blockedConfig(10.0f, numKeys), pool());
    builder->insert("key");
    builder->finish();

    NIMBLE_ASSERT_THROW(
        builder->insert("late"), "Cannot insert into a finished bloom filter");
    NIMBLE_ASSERT_THROW(builder->finish(), "Bloom filter is already finished");
  }
}

TEST_F(BloomFilterTest, memoryIsPoolTrackedAndReleased) {
  const auto memoryBefore = pool()->usedBytes();
  {
    auto builder =
        createBloomFilterBuilder(blockedConfig(10.0f, 10'000), pool());
    EXPECT_GT(pool()->usedBytes(), memoryBefore);
    const auto reader = open(builder->finish());
    EXPECT_GT(pool()->usedBytes(), memoryBefore);
  }
  EXPECT_EQ(pool()->usedBytes(), memoryBefore);
}

TEST_F(BloomFilterTest, sizingAtFinishReleasesHashes) {
  const auto memoryBefore = pool()->usedBytes();
  {
    auto builder =
        createBloomFilterBuilder(blockedConfig(10.0f, std::nullopt), pool());
    for (const auto& key : makeKeys("key_", 10'000)) {
      builder->insert(key);
    }
    const auto memoryWithHashes = pool()->usedBytes();
    EXPECT_GT(memoryWithHashes, memoryBefore);

    // A hash per key costs 8 bytes; the filter they size costs 1.25 bytes per
    // key at 10 bits per key. Once finish() releases the hashes, the builder
    // holds much less than it did.
    builder->finish();
    EXPECT_LT(pool()->usedBytes(), memoryWithHashes);
  }
  EXPECT_EQ(pool()->usedBytes(), memoryBefore);
}

} // namespace
} // namespace facebook::nimble::index
