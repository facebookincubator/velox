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

#include "velox/functions/sparksql/aggregates/SparkNumericHistogram.h"

#include <bit>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <folly/String.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/Memory.h"

namespace facebook::velox::functions::aggregate::sparksql::test {
namespace {

using Bin = SparkNumericHistogram::Bin;
using testing::IsEmpty;

// Hand-derived fixtures, not output from the implementation under test or an
// executed JVM oracle. The public baseline is Apache Spark revision
// b84dc909a8856388faddc154c6a1d3aba271474e:
// https://github.com/apache/spark/blob/b84dc909a8856388faddc154c6a1d3aba271474e/sql/catalyst/src/main/java/org/apache/spark/sql/util/NumericHistogram.java
// Its add/merge/trim operations and NumericHistogramSerializer in
// sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/aggregate/HistogramNumeric.scala
// define every operation sequence below. Finite arithmetic fixtures assume
// binary64 round-to-nearest, ties-to-even, with no contraction. Arithmetic NaNs
// are checked by classification only: their payload is not portable across
// JDK/CPU combinations. Untouched transport payloads are checked bit-for-bit.
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr int32_t kMaxNumBins = std::numeric_limits<int32_t>::max();

// Constructs an opaque Spark payload independently of the production codec.
std::string payload(int32_t numBins, const std::vector<Bin>& bins) {
  std::string bytes;
  auto append = [&](uint64_t value, int32_t width) {
    for (auto i = width - 1; i >= 0; --i) {
      bytes.push_back(static_cast<char>((value >> (8 * i)) & 0xff));
    }
  };
  append(static_cast<uint32_t>(numBins), 4);
  append(bins.size(), 4);
  for (const auto& bin : bins) {
    append(std::bit_cast<uint64_t>(bin.x), 8);
    append(std::bit_cast<uint64_t>(bin.y), 8);
  }
  return bytes;
}

std::string unhex(std::string_view hex) {
  std::string bytes;
  VELOX_CHECK(
      folly::unhexlify(folly::StringPiece(hex.data(), hex.size()), bytes));
  return bytes;
}

std::string serialize(const SparkNumericHistogram& histogram) {
  std::string bytes(histogram.serializedSize(), '\0');
  histogram.serialize(bytes.data());
  return bytes;
}

void assertBins(
    const SparkNumericHistogram& histogram,
    const std::vector<Bin>& expected) {
  ASSERT_EQ(histogram.bins().size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(
        std::bit_cast<uint64_t>(histogram.bins()[i].x),
        std::bit_cast<uint64_t>(expected[i].x));
    EXPECT_EQ(
        std::bit_cast<uint64_t>(histogram.bins()[i].y),
        std::bit_cast<uint64_t>(expected[i].y));
  }
}

class SparkNumericHistogramTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void TearDown() override {
    EXPECT_EQ(allocator_.currentBytes(), 0);
    EXPECT_EQ(allocator_.checkConsistency(), 0);
  }

  std::shared_ptr<memory::MemoryPool> pool_ =
      memory::memoryManager()->addLeafPool();
  HashStringAllocator allocator_{pool_.get()};
};

TEST_F(SparkNumericHistogramTest, unboundAndKnownEmpty) {
  SparkNumericHistogram histogram(&allocator_);
  EXPECT_EQ(histogram.numBins(), 0);
  EXPECT_THAT(histogram.bins(), IsEmpty());
  VELOX_ASSERT_THROW(histogram.serializedSize(), "not initialized");
  VELOX_ASSERT_THROW(histogram.add(1), "not initialized");

  histogram.initialize(3);
  EXPECT_EQ(histogram.numBins(), 3);
  EXPECT_THAT(histogram.bins(), IsEmpty());
  EXPECT_EQ(serialize(histogram), unhex("0000000300000000"));
  EXPECT_EQ(allocator_.currentBytes(), 0);
  SparkNumericHistogram fromWire(&allocator_);
  fromWire.mergeSerialized(unhex("0000000300000000"));
  EXPECT_EQ(fromWire.numBins(), 3);
  EXPECT_THAT(fromWire.bins(), IsEmpty());
  EXPECT_EQ(serialize(fromWire), serialize(histogram));
  EXPECT_EQ(allocator_.currentBytes(), 0);
  // The aggregate adapter uses empty bins for a null final, and skips null
  // intermediate transport without calling mergeSerialized. A zero-length
  // non-null payload, in contrast, is structurally invalid.
  VELOX_ASSERT_USER_THROW(histogram.mergeSerialized({}), "header");
}

TEST_F(SparkNumericHistogramTest, initializeRejectsInvalidAndChangedCapacity) {
  SparkNumericHistogram histogram(&allocator_);
  for (auto numBins : {std::numeric_limits<int32_t>::min(), -1, 0, 1}) {
    VELOX_ASSERT_USER_THROW(histogram.initialize(numBins), "at least 2");
    EXPECT_EQ(histogram.numBins(), 0);
  }
  histogram.initialize(2);
  histogram.add(1);
  histogram.initialize(2);
  const auto before = serialize(histogram);
  VELOX_ASSERT_USER_THROW(histogram.initialize(3), "numBins mismatch");
  VELOX_ASSERT_USER_THROW(histogram.initialize(1), "at least 2");
  EXPECT_EQ(serialize(histogram), before);
}

TEST_F(SparkNumericHistogramTest, primitiveSearchAndDuplicates) {
  SparkNumericHistogram histogram(&allocator_);
  histogram.initialize(8);
  for (auto value : {4, 1, 3, 2, 0, 5, 3}) {
    histogram.add(value);
  }
  assertBins(histogram, {{0, 1}, {1, 1}, {2, 1}, {3, 2}, {4, 1}, {5, 1}});
}

TEST_F(SparkNumericHistogramTest, primitiveSignedZeros) {
  for (double first : {-0.0, 0.0}) {
    SparkNumericHistogram histogram(&allocator_);
    histogram.initialize(2);
    histogram.add(first);
    histogram.add(-first);
    assertBins(histogram, {{first, 2}});
  }
}

TEST_F(SparkNumericHistogramTest, primitiveNaNPositions) {
  const auto nan = std::bit_cast<double>(0x7ff8000000000042ULL);
  // An unordered comparison stops at the midpoint, but primitive equality
  // remains false, so the new value is inserted before that midpoint.
  struct Case {
    std::vector<double> input;
    std::vector<Bin> expected;
  };
  const std::vector<Case> cases{
      {{1, 2, nan}, {{1, 1}, {nan, 1}, {2, 1}}},
      {{nan, 1, 2}, {{1, 1}, {2, 1}, {nan, 1}}},
      {{1, nan, 2}, {{nan, 1}, {1, 1}, {2, 1}}},
      {{nan, nan, nan}, {{nan, 1}, {nan, 1}, {nan, 1}}},
  };
  for (const auto& test : cases) {
    SparkNumericHistogram histogram(&allocator_);
    histogram.initialize(4);
    for (auto value : test.input) {
      histogram.add(value);
    }
    assertBins(histogram, test.expected);
  }
}

TEST_F(SparkNumericHistogramTest, primitiveInfinities) {
  SparkNumericHistogram histogram(&allocator_);
  histogram.initialize(4);
  for (auto value : {kInfinity, -kInfinity, 0.0, kInfinity, -kInfinity}) {
    histogram.add(value);
  }
  assertBins(histogram, {{-kInfinity, 2}, {0, 1}, {kInfinity, 2}});
}

TEST_F(SparkNumericHistogramTest, duplicateMidpointAfterMerge) {
  SparkNumericHistogram histogram(&allocator_);
  histogram.mergeSerialized(payload(8, {{2, 1}, {2, 2}}));
  histogram.mergeSerialized(payload(8, {{2, 4}, {2, 8}}));
  histogram.add(2);
  // Four comparator-equal bins survive merge. Raw search lands at index 2,
  // not the leftmost equal center as a lower_bound-based update would.
  assertBins(histogram, {{2, 1}, {2, 2}, {2, 5}, {2, 8}});
}

TEST_F(SparkNumericHistogramTest, stableJavaOrderMerge) {
  const auto nan1 = std::bit_cast<double>(0xfff8000000000001ULL);
  const auto nan2 = std::bit_cast<double>(0x7ff8000000000002ULL);
  SparkNumericHistogram histogram(&allocator_);
  histogram.mergeSerialized(
      payload(16, {{nan1, 1}, {0.0, 2}, {2, 3}, {-0.0, 4}, {2, 5}}));
  histogram.mergeSerialized(
      payload(16, {{nan2, 6}, {-0.0, 7}, {2, 8}, {0.0, 9}, {-kInfinity, 10}}));
  assertBins(
      histogram,
      {{-kInfinity, 10},
       {-0.0, 4},
       {-0.0, 7},
       {0.0, 2},
       {0.0, 9},
       {2, 3},
       {2, 5},
       {2, 8},
       {nan1, 1},
       {nan2, 6}});
}

TEST_F(SparkNumericHistogramTest, emptyDestinationRetainsSourceOrder) {
  const auto bytes = payload(3, {{3, 1}, {1, 2}, {2, 3}});
  for (bool bindFirst : {false, true}) {
    SparkNumericHistogram histogram(&allocator_);
    if (bindFirst) {
      histogram.initialize(3);
    }
    auto source = bytes;
    histogram.mergeSerialized(source);
    source.assign(source.size(), '!');
    EXPECT_EQ(serialize(histogram), bytes);
  }
}

TEST_F(SparkNumericHistogramTest, nonNullEmptySourceSorts) {
  const auto nan = std::bit_cast<double>(0x7ff800000000002aULL);
  SparkNumericHistogram histogram(&allocator_);
  histogram.initialize(3);
  histogram.add(1);
  histogram.add(2);
  histogram.add(nan);
  assertBins(histogram, {{1, 1}, {nan, 1}, {2, 1}});
  histogram.mergeSerialized(unhex("0000000300000000"));
  assertBins(histogram, {{1, 1}, {2, 1}, {nan, 1}});
}

// Independently applying the Java Random specification to seed 31183 gives
// initial state 0x0005deec9fa2. The first six successive 48-bit LCG states are
// 1b8427838405, 80980637b42c, e4c4f2273ec7, d7ebce9084c6, 2338e55a6c59,
// 1456f4e417f0.
// Each nextDouble takes 26 high bits from one state and 27 from the next:
// draw 1 is in (0.10, 0.11), draw 2 in (0.89, 0.90), draw 3 in (0.13, 0.14).
// These bounds alone establish the pair choices below without a JVM run.
// https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/Random.html
TEST_F(SparkNumericHistogramTest, repeatedTieDraws) {
  SparkNumericHistogram histogram(&allocator_);
  histogram.initialize(2);
  for (auto value : {0, 1, 2}) {
    histogram.add(value);
  }
  assertBins(histogram, {{0, 1}, {1.5, 2}});
  histogram.add(3);
  assertBins(histogram, {{1, 3}, {3, 1}});
  histogram.add(5);
  assertBins(histogram, {{1, 3}, {4, 2}});
}

TEST_F(SparkNumericHistogramTest, multiwayTies) {
  SparkNumericHistogram histogram(&allocator_);
  histogram.initialize(3);
  for (auto value : {0, 1, 2, 3}) {
    histogram.add(value);
  }
  // Draw 1 selects pair 1 at threshold 1/2. Draw 2 does not replace it at
  // threshold 1/3. Each equality consumes one draw, not one per final choice.
  assertBins(histogram, {{0, 1}, {1.5, 2}, {3, 1}});
}

TEST_F(SparkNumericHistogramTest, serializedResumeResetsRandom) {
  SparkNumericHistogram original(&allocator_);
  original.initialize(2);
  for (auto value : {0, 1, 2}) {
    original.add(value);
  }
  SparkNumericHistogram resumed(&allocator_);
  resumed.mergeSerialized(serialize(original));
  original.add(3);
  resumed.add(3);
  // The bytes carry bins and capacity, not the consumed first RNG draw.
  assertBins(original, {{1, 3}, {3, 1}});
  assertBins(resumed, {{0, 1}, {2, 3}});
}

TEST_F(SparkNumericHistogramTest, mergeKeepsDestinationRandom) {
  SparkNumericHistogram histogram(&allocator_);
  histogram.initialize(2);
  for (auto value : {0, 1, 2}) {
    histogram.add(value);
  }
  histogram.initialize(2);
  histogram.mergeSerialized(payload(3, {}));
  EXPECT_EQ(histogram.numBins(), 2);
  VELOX_ASSERT_USER_THROW(histogram.mergeSerialized("short"), "header");
  histogram.mergeSerialized(payload(2, {{3, 1}}));
  assertBins(histogram, {{1, 3}, {3, 1}});
}

TEST_F(SparkNumericHistogramTest, independentRandomStreams) {
  SparkNumericHistogram first(&allocator_);
  SparkNumericHistogram second(&allocator_);
  first.initialize(2);
  second.initialize(2);
  for (auto value : {0, 1, 2}) {
    first.add(value);
    second.add(value);
  }
  assertBins(first, {{0, 1}, {1.5, 2}});
  assertBins(second, {{0, 1}, {1.5, 2}});
  first.add(3);
  first.add(5);
  second.add(3);
  assertBins(first, {{1, 3}, {4, 2}});
  assertBins(second, {{1, 3}, {3, 1}});
}

TEST_F(SparkNumericHistogramTest, nonMutatingSerialization) {
  SparkNumericHistogram histogram(&allocator_);
  histogram.initialize(2);
  histogram.add(0);
  histogram.add(1);
  const auto before = serialize(histogram);
  EXPECT_EQ(serialize(histogram), before);
  histogram.add(2);
  const auto after = serialize(histogram);
  EXPECT_EQ(serialize(histogram), after);
  histogram.add(3);
  assertBins(histogram, {{1, 3}, {3, 1}});
}

TEST_F(SparkNumericHistogramTest, firstGapNaN) {
  SparkNumericHistogram histogram(&allocator_);
  histogram.mergeSerialized(
      payload(3, {{-kInfinity, 1}, {-kInfinity, 2}, {0, 1}}));
  histogram.add(1);
  // -Inf - -Inf is NaN: even the finite later gap of 1 cannot replace it.
  assertBins(histogram, {{-kInfinity, 3}, {0, 1}, {1, 1}});

  SparkNumericHistogram raw(&allocator_);
  raw.initialize(2);
  raw.add(1);
  raw.add(2);
  raw.add(std::numeric_limits<double>::quiet_NaN());
  ASSERT_EQ(raw.bins().size(), 2);
  EXPECT_TRUE(std::isnan(raw.bins()[0].x));
  EXPECT_EQ(raw.bins()[0].y, 2);
  EXPECT_EQ(raw.bins()[1].x, 2);
  EXPECT_EQ(raw.bins()[1].y, 1);
}

TEST_F(SparkNumericHistogramTest, infiniteAndMaximalGaps) {
  for (auto extreme : {kInfinity, std::numeric_limits<double>::max()}) {
    SparkNumericHistogram histogram(&allocator_);
    histogram.initialize(2);
    histogram.add(-extreme);
    histogram.add(0);
    histogram.add(extreme);
    // The first actual gap is the baseline. The one tie draws 1 and picks
    // pair 1, even if both gaps are infinite or equal to DBL_MAX.
    assertBins(histogram, {{-extreme, 1}, {extreme / 2, 2}});
  }
}

TEST_F(SparkNumericHistogramTest, separateRoundingSteps) {
  SparkNumericHistogram tiny(&allocator_);
  tiny.initialize(2);
  tiny.add(0x0.0000000000001p-1022);
  tiny.add(0x0.0000000000002p-1022);
  tiny.add(1);
  // First term underflows to even zero; second term is one subnormal unit.
  // A single rounding of the mathematical mean would give TWO units.
  assertBins(tiny, {{0x0.0000000000001p-1022, 2}, {1, 1}});

  SparkNumericHistogram large(&allocator_);
  large.initialize(2);
  large.add(0x1.ffffffffffffep1023);
  large.add(0x1.fffffffffffffp1023);
  large.add(kInfinity);
  // Half + half is a midpoint rounded to the even lower significand;
  // summing before dividing would overflow.
  assertBins(large, {{0x1.ffffffffffffep1023, 2}, {kInfinity, 1}});
}

TEST_F(SparkNumericHistogramTest, noContractedMultiplyAdd) {
  SparkNumericHistogram histogram(&allocator_);
  histogram.mergeSerialized(payload(
      2, {{-0x1.ffffffcp0, 0x1.0000002p0}, {0x1.0000002p1, 0x1.ffffffcp-1}}));
  histogram.add(16);
  // Let e=2^-27. d=2, first term rounds -(1-e^2) to -1, second
  // rounds (1-e^2) to +1, so the result is +0. A fused final multiply-add
  // instead produces -2^-54. This also detects extra-precision accumulation.
  assertBins(histogram, {{0.0, 2}, {16, 1}});
}

TEST_F(SparkNumericHistogramTest, serializedSizeRejectsOversizedOutput) {
  // Preflight the actual output count without constructing multi-gigabyte bin
  // storage. Instance serializedSize() and serialize() use this same check.
  EXPECT_EQ(SparkNumericHistogram::serializedSize(0), 8);
  EXPECT_EQ(SparkNumericHistogram::serializedSize(1), 24);
  // floor((INT32_MAX - 8) / 16) = 134217727. One more bin needs
  // 2147483656 bytes, beyond StringWriter's and Java's int-sized payloads.
  EXPECT_EQ(SparkNumericHistogram::serializedSize(134'217'727), 2'147'483'640);
  for (uint64_t numUsedBins : {
           uint64_t{134'217'728},
           uint64_t{kMaxNumBins},
           uint64_t{1} << 60, // Multiplying by 16 would wrap to zero.
           std::numeric_limits<uint64_t>::max(),
       }) {
    SCOPED_TRACE(numUsedBins);
    VELOX_ASSERT_USER_THROW(
        SparkNumericHistogram::serializedSize(numUsedBins),
        "histogram_numeric serialized state is too large");
    EXPECT_EQ(allocator_.currentBytes(), 0);
  }
}

TEST_F(SparkNumericHistogramTest, exactWireBytesEmptyAndSingle) {
  SparkNumericHistogram histogram(&allocator_);
  histogram.initialize(3);
  EXPECT_EQ(histogram.serializedSize(), 8);
  EXPECT_EQ(serialize(histogram), unhex("0000000300000000"));
  histogram.add(1);
  EXPECT_EQ(histogram.serializedSize(), 24);
  EXPECT_EQ(
      serialize(histogram),
      unhex("00000003000000013ff00000000000003ff0000000000000"));
}

TEST_F(SparkNumericHistogramTest, payloadBitsAndUnalignedIo) {
  // Header and all x/y bits are literal. Include signaling and quiet NaNs,
  // both signs of zero, and distinct signs/payloads without any arithmetic.
  const auto bytes = unhex(
      "0000000400000004"
      "7ff00000000000427ff8000000000043"
      "fff80000000012348000000000000000"
      "80000000000000000000000000000000"
      "0000000000000000fff0000000000044");
  const auto padded = "!" + bytes;
  SparkNumericHistogram histogram(&allocator_);
  histogram.mergeSerialized(std::string_view(padded).substr(1));
  std::string output(bytes.size() + 2, '!');
  histogram.serialize(output.data() + 1);
  EXPECT_EQ(output.front(), '!');
  EXPECT_EQ(output.back(), '!');
  EXPECT_EQ(output.substr(1, bytes.size()), bytes);
  histogram.mergeSerialized(payload(4, {}));
  EXPECT_EQ(
      serialize(histogram),
      unhex(
          "0000000400000004"
          "80000000000000000000000000000000"
          "0000000000000000fff0000000000044"
          "7ff00000000000427ff8000000000043"
          "fff80000000012348000000000000000"));
}

TEST_F(SparkNumericHistogramTest, rejectsShortHeaders) {
  SparkNumericHistogram histogram(&allocator_);
  const auto header = unhex("0000000300000000");
  for (size_t size = 0; size < 8; ++size) {
    VELOX_ASSERT_USER_THROW(
        histogram.mergeSerialized(std::string_view(header).substr(0, size)),
        "header");
    EXPECT_EQ(histogram.numBins(), 0);
    EXPECT_THAT(histogram.bins(), IsEmpty());
    EXPECT_EQ(allocator_.currentBytes(), 0);
  }
}

TEST_F(SparkNumericHistogramTest, rejectsInvalidHeaderCounts) {
  SparkNumericHistogram histogram(&allocator_);
  histogram.initialize(3);
  histogram.add(1);
  const auto before = serialize(histogram);
  for (const auto* hex : {
           "ffffffff00000000", // Negative capacity.
           "0000000000000000",
           "0000000100000000",
           "00000003ffffffff", // Negative used count.
           "0000000380000000",
           "0000000300000004", // Used count exceeds capacity.
       }) {
    const auto padded = "!" + unhex(hex);
    VELOX_ASSERT_USER_THROW(
        histogram.mergeSerialized(std::string_view(padded).substr(1)), "");
    EXPECT_EQ(serialize(histogram), before);
  }
}

TEST_F(SparkNumericHistogramTest, rejectsTruncatedOrTrailing) {
  SparkNumericHistogram histogram(&allocator_);
  histogram.initialize(3);
  histogram.add(2);
  const auto before = serialize(histogram);
  const auto source = payload(3, {{1, 1}, {3, 1}});
  const auto memoryBefore = allocator_.currentBytes();
  for (size_t size = 8; size < source.size(); ++size) {
    VELOX_ASSERT_USER_THROW(
        histogram.mergeSerialized(std::string_view(source).substr(0, size)),
        "length");
    EXPECT_EQ(serialize(histogram), before);
    EXPECT_EQ(allocator_.currentBytes(), memoryBefore);
  }
  for (const auto& bytes : {source + "x", payload(3, {}) + "x"}) {
    VELOX_ASSERT_USER_THROW(histogram.mergeSerialized(bytes), "length");
    EXPECT_EQ(serialize(histogram), before);
  }
}

TEST_F(SparkNumericHistogramTest, rejectsOversizedStateBeforeAllocation) {
  SparkNumericHistogram histogram(&allocator_);
  for (const auto* hex : {
           "7fffffff7fffffff",
           "7fffffff08000000", // 8+16*2^27 exceeds INT32_MAX.
       }) {
    VELOX_ASSERT_USER_THROW(histogram.mergeSerialized(unhex(hex)), "too large");
    EXPECT_EQ(histogram.numBins(), 0);
    EXPECT_EQ(allocator_.currentBytes(), 0);
  }
  // Just below the size limit, the header still cannot claim absent pairs.
  VELOX_ASSERT_USER_THROW(
      histogram.mergeSerialized(unhex("7fffffff07ffffff")), "length");
  histogram.initialize(kMaxNumBins);
  histogram.add(1);
  EXPECT_EQ(serialize(histogram), payload(kMaxNumBins, {{1, 1}}));
}

TEST_F(SparkNumericHistogramTest, emptyDestinationAdoptsSourceCapacity) {
  SparkNumericHistogram histogram(&allocator_);
  histogram.initialize(2);
  histogram.mergeSerialized(payload(3, {{3, 1}, {1, 2}, {2, 3}}));

  EXPECT_EQ(histogram.numBins(), 3);
  assertBins(histogram, {{3, 1}, {1, 2}, {2, 3}});
}

TEST_F(SparkNumericHistogramTest, nonemptyDestinationRetainsCapacity) {
  SparkNumericHistogram histogram(&allocator_);
  histogram.initialize(2);
  histogram.add(0);
  histogram.mergeSerialized(payload(3, {{2, 1}, {4, 1}}));

  EXPECT_EQ(histogram.numBins(), 2);
  assertBins(histogram, {{0, 1}, {3, 2}});
}

TEST_F(SparkNumericHistogramTest, validOpaquePayloads) {
  SparkNumericHistogram histogram(&allocator_);
  const auto bytes = payload(
      4, {{kInfinity, -1}, {-1, -0.0}, {-kInfinity, kInfinity}, {1, 0.0}});
  histogram.mergeSerialized(bytes);
  EXPECT_EQ(serialize(histogram), bytes);
  // A changed but structurally valid double is not detectable corruption.
  auto changed = bytes;
  changed.back() = 1;
  SparkNumericHistogram other(&allocator_);
  other.mergeSerialized(changed);
  EXPECT_EQ(serialize(other), changed);
}

TEST_F(SparkNumericHistogramTest, mergeHistoriesRetainDuplicates) {
  SparkNumericHistogram raw(&allocator_);
  raw.initialize(4);
  raw.add(1);
  raw.add(1);
  assertBins(raw, {{1, 2}});

  SparkNumericHistogram merged(&allocator_);
  merged.mergeSerialized(payload(4, {{1, 1}}));
  merged.mergeSerialized(payload(4, {{1, 1}}));
  assertBins(merged, {{1, 1}, {1, 1}});
  EXPECT_NE(serialize(raw), serialize(merged));

  // Same multiset, different explicit merge trees. With capacity 2, merging
  // {0},{1},{2} consumes draw 1 and merges (1,2); starting with {1},{2},{3}
  // merges (2,3). The remaining gaps and the next tie draw determine the
  // final results. No topology-invariance assertion is appropriate here.
  SparkNumericHistogram left(&allocator_);
  SparkNumericHistogram right(&allocator_);
  for (auto value : {0, 1, 2, 3}) {
    left.mergeSerialized(payload(2, {{static_cast<double>(value), 1}}));
  }
  for (auto value : {1, 2, 3, 0}) {
    right.mergeSerialized(payload(2, {{static_cast<double>(value), 1}}));
  }
  assertBins(left, {{1, 3}, {3, 1}});
  assertBins(right, {{0.5, 2}, {2.5, 2}});
}

TEST_F(SparkNumericHistogramTest, lazyAlignedTrackedStorage) {
  const auto bytes = payload(kMaxNumBins, {{2, 1}, {1, 2}});
  {
    SparkNumericHistogram histogram(&allocator_);
    histogram.initialize(kMaxNumBins);
    EXPECT_EQ(allocator_.currentBytes(), 0);
    histogram.mergeSerialized(bytes);
    histogram.mergeSerialized(payload(kMaxNumBins, {{3, 1}}));
    EXPECT_EQ(
        reinterpret_cast<uintptr_t>(histogram.bins().data()) % alignof(Bin), 0);
    EXPECT_GT(allocator_.currentBytes(), 0);
    EXPECT_LT(allocator_.currentBytes(), 1'024);
    EXPECT_EQ(allocator_.checkConsistency(), allocator_.currentBytes());
    assertBins(histogram, {{1, 2}, {2, 1}, {3, 1}});
  }
  EXPECT_EQ(allocator_.currentBytes(), 0);
}

TEST_F(SparkNumericHistogramTest, mergeAllocationFailureDoesNotMutate) {
  auto root = memory::memoryManager()->addRootPool("histogramMerge", 1 << 20);
  auto pool = root->addLeafChild("histogramMergeLeaf");
  HashStringAllocator allocator(pool.get());
  {
    SparkNumericHistogram histogram(&allocator);
    histogram.initialize(kMaxNumBins);
    for (auto i = 0; i < 8'192; ++i) {
      histogram.add(i);
    }
    const auto before = serialize(histogram);
    const auto memoryBefore = allocator.currentBytes();
    // Exercise both staging failure with a live destination and failure on
    // initial deserialization. Staging must be pool-accounted, too.
    for (auto numSourceBins : {24'576, 100'000}) {
      const auto source =
          payload(kMaxNumBins, std::vector<Bin>(numSourceBins, {0, 1}));
      VELOX_ASSERT_THROW(histogram.mergeSerialized(source), "");
      EXPECT_EQ(serialize(histogram), before);
      EXPECT_EQ(allocator.currentBytes(), memoryBefore);
    }
    SparkNumericHistogram empty(&allocator);
    const auto source = payload(kMaxNumBins, std::vector<Bin>(100'000, {0, 1}));
    VELOX_ASSERT_THROW(empty.mergeSerialized(source), "");
    EXPECT_EQ(empty.numBins(), 0);
    EXPECT_THAT(empty.bins(), IsEmpty());
    EXPECT_EQ(allocator.currentBytes(), memoryBefore);
    histogram.add(8'192);
    EXPECT_EQ(histogram.bins().size(), 8'193);
  }
  EXPECT_EQ(allocator.currentBytes(), 0);
  EXPECT_EQ(allocator.checkConsistency(), 0);
}

TEST_F(SparkNumericHistogramTest, addAllocationFailureDoesNotMutate) {
  auto root = memory::memoryManager()->addRootPool("histogramAdd", 1 << 20);
  auto pool = root->addLeafChild("histogramAddLeaf");
  HashStringAllocator allocator(pool.get());
  {
    SparkNumericHistogram histogram(&allocator);
    histogram.initialize(kMaxNumBins);
    for (auto i = 0; i < 32'768; ++i) {
      histogram.add(i);
    }
    const auto before = serialize(histogram);
    const auto memoryBefore = allocator.currentBytes();
    VELOX_ASSERT_THROW(histogram.add(32'768), "");
    EXPECT_EQ(serialize(histogram), before);
    EXPECT_EQ(allocator.currentBytes(), memoryBefore);
    // An existing bin can still be updated without allocating.
    histogram.add(0);
    EXPECT_EQ(histogram.bins()[0].y, 2);
  }
  EXPECT_EQ(allocator.currentBytes(), 0);
  EXPECT_EQ(allocator.checkConsistency(), 0);
}

} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
