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

#include "velox/dwio/common/DecoderUtil.h"
#include <folly/Random.h>
#include "velox/common/base/Nulls.h"
#include "velox/dwio/common/SelectiveColumnReader.h"
#include "velox/type/Filter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <functional>

using namespace facebook::velox;
using namespace facebook::velox::dwio::common;

class DecoderUtilTest : public testing::Test {
 protected:
  void SetUp() override {
    rng_.seed(1);
  }

  void randomBits(std::vector<uint64_t>& bits, int32_t onesPer1000) {
    for (auto i = 0; i < bits.size() * 64; ++i) {
      if (folly::Random::rand32(rng_) % 1000 < onesPer1000) {
        bits::setBit(bits.data(), i);
      }
    }
  }

  void randomRows(
      int32_t numRows,
      int32_t rowsPer1000,
      raw_vector<int32_t>& result) {
    for (auto i = 0; i < numRows; ++i) {
      if (folly::Random::rand32(rng_) % 1000 < rowsPer1000) {
        result.push_back(i);
      }
    }
  }

  template <bool isFilter, bool outputNulls>
  bool nonNullRowsFromSparseReference(
      const uint64_t* nulls,
      RowSet rows,
      raw_vector<int32_t>& innerRows,
      raw_vector<int32_t>& outerRows,
      uint64_t* resultNulls,
      int32_t& tailSkip) {
    bool anyNull = false;
    auto numIn = rows.size();
    innerRows.resize(numIn);
    outerRows.resize(numIn);
    int32_t lastRow = -1;
    int32_t numNulls = 0;
    int32_t numInner = 0;
    int32_t lastNonNull = -1;
    for (auto i = 0; i < numIn; ++i) {
      auto row = rows[i];
      if (row > lastRow + 1) {
        numNulls += bits::countNulls(nulls, lastRow + 1, row);
      }
      if (bits::isBitNull(nulls, row)) {
        ++numNulls;
        lastRow = row;
        if (!isFilter && outputNulls) {
          bits::setNull(resultNulls, i);
          anyNull = true;
        }
      } else {
        innerRows[numInner] = row - numNulls;
        outerRows[numInner++] = isFilter ? row : i;
        lastNonNull = row;
        lastRow = row;
      }
    }
    innerRows.resize(numInner);
    outerRows.resize(numInner);
    tailSkip = bits::countBits(nulls, lastNonNull + 1, lastRow);
    return anyNull;
  }

  // Maps 'rows' where the row falls on a non-null in 'nulls' to an
  // index in non-null rows. This uses both a reference implementation
  // and the SIMDized fast path and checks consistent results.
  template <bool isFilter, bool outputNulls>
  void testNonNullFromSparse(uint64_t* nulls, RowSet rows) {
    raw_vector<int32_t> referenceInner;
    raw_vector<int32_t> referenceOuter;
    std::vector<uint64_t> referenceNulls(bits::nwords(rows.size()), ~0ULL);
    int32_t referenceSkip;
    auto referenceAnyNull =
        nonNullRowsFromSparseReference<isFilter, outputNulls>(
            nulls,
            rows,
            referenceInner,
            referenceOuter,
            referenceNulls.data(),
            referenceSkip);
    raw_vector<int32_t> testInner;
    raw_vector<int32_t> testOuter;
    std::vector<uint64_t> testNulls(bits::nwords(rows.size()), ~0ULL);
    int32_t testSkip;
    auto testAnyNull = nonNullRowsFromSparse<isFilter, outputNulls>(
        nulls, rows, testInner, testOuter, testNulls.data(), testSkip);

    EXPECT_EQ(testAnyNull, referenceAnyNull);
    EXPECT_EQ(testSkip, referenceSkip);
    for (auto i = 0; i < testInner.size() && i < testOuter.size(); ++i) {
      EXPECT_EQ(testInner[i], referenceInner[i]);
      EXPECT_EQ(testOuter[i], referenceOuter[i]);
    }
    EXPECT_EQ(testInner.size(), referenceInner.size());
    EXPECT_EQ(testOuter.size(), referenceOuter.size());

    if (outputNulls) {
      for (auto i = 0; i < rows.size(); ++i) {
        EXPECT_EQ(
            bits::isBitSet(testNulls.data(), i),
            bits::isBitSet(referenceNulls.data(), i));
      }
    }
  }

  void testNonNullFromSparseCases(uint64_t* nulls, RowSet rows) {
    testNonNullFromSparse<false, true>(nulls, rows);
    testNonNullFromSparse<true, false>(nulls, rows);
  }

  folly::Random::DefaultGenerator rng_;
};

// Running for about 13 seconds.
TEST_F(DecoderUtilTest, nonNullsFromSparse) {
  // We cover cases with different null frequencies and different density of
  // access.
  constexpr int32_t kSize = 2000;
  for (auto nullsIn1000 = 1; nullsIn1000 < 1011; nullsIn1000 += 10) {
    for (auto rowsIn1000 = 1; rowsIn1000 < 1011; rowsIn1000 += 10) {
      raw_vector<int32_t> rows;
      // Have an extra word at the end to allow 64 bit access.
      std::vector<uint64_t> nulls(bits::nwords(kSize) + 1);
      randomBits(nulls, 1000 - nullsIn1000);
      randomRows(kSize, rowsIn1000, rows);
      if (rows.empty()) {
        // The operation is not defined for 0 rows.
        rows.push_back(1234);
      }
      testNonNullFromSparseCases(nulls.data(), rows);
    }
  }
}

TEST_F(DecoderUtilTest, processFixedWithRun) {
  // Tests processing consecutive batches of integers with processFixedWidthRun.
  constexpr int kSize = 100;
  constexpr int32_t kStep = 17;
  raw_vector<int32_t> data;
  raw_vector<int32_t> scatter;
  data.reserve(kSize);
  scatter.reserve(kSize);
  // Data is 0, 100,  2, 98 ... 98, 2.
  // scatter is 0, 2, 4,6 ... 196, 198.
  for (auto i = 0; i < kSize; i += 2) {
    data.push_back(i / 2);
    data.push_back(kSize - i);
    scatter.push_back(i * 2);
    scatter.push_back((i + 1) * 2);
  }

  // the row numbers that pass the filter come here, translated via scatter.
  raw_vector<int32_t> hits(kSize);
  // Each valid index in 'data'
  raw_vector<int32_t> rows(kSize);
  auto filter = std::make_unique<common::BigintRange>(40, 1000, false);
  std::iota(rows.begin(), rows.end(), 0);
  // The passing values are gathered here. Before each call to
  // processFixedWidthRun, the candidate values are appended here and
  // processFixedWidthRun overwrites them with the passing values and sets
  // numValues to be the first unused index after the passing values.
  raw_vector<int32_t> results;
  int32_t numValues = 0;
  for (auto rowIndex = 0; rowIndex < kSize; rowIndex += kStep) {
    int32_t numInput = std::min<int32_t>(kStep, kSize - rowIndex);
    results.resize(numValues + numInput);
    std::memcpy(
        results.data() + numValues,
        data.data() + rowIndex,
        numInput * sizeof(results[0]));

    NoHook noHook;
    processFixedWidthRun<int32_t, false, true, false>(
        rows,
        rowIndex,
        numInput,
        scatter.data(),
        results.data(),
        hits.data(),
        numValues,
        *filter,
        noHook);
  }
  // Check that each value that passes the filter is in 'results' and that   its
  // index times 2 is in 'data' is in 'hits'. The 2x is because the scatter maps
  // each row to 2x the row number.
  int32_t passedCount = 0;
  for (auto i = 0; i < kSize; ++i) {
    if (data[i] >= 40) {
      EXPECT_EQ(data[i], results[passedCount]);
      EXPECT_EQ(i * 2, hits[passedCount]);
      ++passedCount;
    }
  }
}

TEST_F(DecoderUtilTest, fixedWidthScanMemcpyFastPath) {
  constexpr int kSize = 10;
  int32_t rows[kSize];
  std::iota(std::begin(rows), std::end(rows), 0);
  float expectedValues[kSize], actualValues[kSize];
  for (int i = 0; i < kSize; ++i) {
    expectedValues[i] = std::sin(i);
    actualValues[i] = NAN;
  }
  int32_t numValues = 0;
  SeekableArrayInputStream input(
      reinterpret_cast<const uint8_t*>(expectedValues), sizeof(expectedValues));
  const char* bufferStart = nullptr;
  const char* bufferEnd = nullptr;
  NoHook noHook;
  fixedWidthScan<float, false, false>(
      {rows, kSize},
      nullptr,
      actualValues,
      nullptr,
      numValues,
      input,
      bufferStart,
      bufferEnd,
      common::AlwaysTrue(),
      noHook);
  for (int i = 0; i < kSize; ++i) {
    ASSERT_EQ(actualValues[i], expectedValues[i]);
  }
  ASSERT_EQ(numValues, kSize);
}

namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

// Number of int32 lanes per SIMD batch on this build's architecture. The kernel
// processes the input one batch at a time, so the tests parameterize sizes off
// this to exercise both full and masked-tail batches regardless of arch.
constexpr int32_t kSimdWidth = xsimd::batch<int32_t>::size;

// Distinctive value pre-filled into the output buffers so a test can tell which
// slots filterDictionaryRunSimd actually wrote.
constexpr int32_t kSentinel = -7;

// filterDictionaryRunSimd gathers each cache byte via a misaligned load at
// 'filterCache + index - 3', so for index 0 it reads up to 3 bytes before the
// pointer. This holds the cache in a std::vector with that much leading slack
// (the real callers satisfy the same contract with raw_vector, whose data() is
// offset by simd::kPadding). The pad value is irrelevant: only bits 6/7 of the
// byte at 'index' drive the kernel's unknown/success masks.
constexpr int32_t kLeadingPad = 16;

class FilterCache {
 public:
  FilterCache(int32_t numIndices, uint8_t fill)
      : storage_(kLeadingPad + numIndices + kLeadingPad, fill) {}

  // Pointer to dictionary index 0, with kLeadingPad valid bytes before it.
  uint8_t* data() {
    return storage_.data() + kLeadingPad;
  }

  uint8_t& operator[](int32_t index) {
    return data()[index];
  }

 private:
  std::vector<uint8_t> storage_;
};

FilterCache makeFilterCache(
    int32_t numIndices,
    uint8_t fill = FilterResult::kUnknown) {
  return FilterCache(numIndices, fill);
}

// Outputs of one filterDictionaryRunSimd call, sliced to just the entries this
// call produced (i.e. [startNumValues, returnedNumValues)) so tests assert
// against independently constructed expectations.
struct FilterRunResult {
  std::vector<int32_t> hits; // passing rows, in input order
  std::vector<int32_t>
      values; // passing dict indices (sentinel-filled if kFilterOnly)
  int32_t returnedNumValues{0};
  std::vector<int32_t>
      testedIndices; // indices handed to testIndex, in call order
};

// Drives filterDictionaryRunSimd against constructed input. 'input' and 'rows'
// are passed by value so this can grow them by one SIMD width -- the kernel
// loads a full batch at the tail, reading past the logical end, so the read
// (input/rows) and write (hits/values) buffers all need that much slack.
template <bool kFilterOnly>
FilterRunResult runFilterRun(
    std::vector<int32_t> input,
    std::vector<int32_t> rows,
    FilterCache& filterCache,
    const std::function<bool(int32_t)>& predicate,
    int32_t startNumValues = 0) {
  const int32_t numInput = static_cast<int32_t>(input.size());
  input.resize(numInput + kSimdWidth, 0);
  rows.resize(numInput + kSimdWidth, 0);

  std::vector<int32_t> hits(startNumValues + numInput + kSimdWidth, kSentinel);
  std::vector<int32_t> values(
      startNumValues + numInput + kSimdWidth, kSentinel);

  std::vector<int32_t> testedIndices;
  auto testIndex = [&](int32_t dictionaryIndex) {
    testedIndices.push_back(dictionaryIndex);
    return predicate(dictionaryIndex);
  };

  const int32_t returnedNumValues = filterDictionaryRunSimd<kFilterOnly>(
      input.data(),
      numInput,
      rows.data(),
      filterCache.data(),
      hits.data(),
      values.data(),
      startNumValues,
      testIndex);

  FilterRunResult result;
  result.returnedNumValues = returnedNumValues;
  result.testedIndices = std::move(testedIndices);
  result.hits.assign(
      hits.begin() + startNumValues, hits.begin() + returnedNumValues);
  result.values.assign(
      values.begin() + startNumValues, values.begin() + returnedNumValues);
  return result;
}

// Every entry passes on a cold cache: all rows/indices flow through and each
// distinct index is recorded as kSuccess.
TEST(FilterDictionaryRunSimdTest, allPassColdCache) {
  auto cache = makeFilterCache(/*numIndices=*/10);
  auto result = runFilterRun</*kFilterOnly=*/false>(
      /*input=*/{3, 1, 4, 1, 5},
      /*rows=*/{100, 101, 102, 103, 104},
      cache,
      /*predicate=*/[](int32_t) { return true; });

  EXPECT_EQ(result.returnedNumValues, 5);
  EXPECT_THAT(result.hits, ElementsAre(100, 101, 102, 103, 104));
  EXPECT_THAT(result.values, ElementsAre(3, 1, 4, 1, 5));
  EXPECT_EQ(cache[3], FilterResult::kSuccess);
  EXPECT_EQ(cache[1], FilterResult::kSuccess);
  EXPECT_EQ(cache[4], FilterResult::kSuccess);
  EXPECT_EQ(cache[5], FilterResult::kSuccess);
}

// Every entry fails on a cold cache: no output, each distinct index recorded as
// kFailure.
TEST(FilterDictionaryRunSimdTest, allFailColdCache) {
  auto cache = makeFilterCache(10);
  auto result = runFilterRun<false>(
      {3, 1, 4, 1, 5}, {100, 101, 102, 103, 104}, cache, [](int32_t) {
        return false;
      });

  EXPECT_EQ(result.returnedNumValues, 0);
  EXPECT_THAT(result.hits, IsEmpty());
  EXPECT_THAT(result.values, IsEmpty());
  EXPECT_EQ(cache[3], FilterResult::kFailure);
  EXPECT_EQ(cache[1], FilterResult::kFailure);
  EXPECT_EQ(cache[4], FilterResult::kFailure);
  EXPECT_EQ(cache[5], FilterResult::kFailure);
}

// A mix of passing/failing indices compacts to just the survivors (preserving
// order) and records the right verdict per index.
TEST(FilterDictionaryRunSimdTest, mixedColdCacheCompactsAndRecords) {
  auto cache = makeFilterCache(10);
  auto result = runFilterRun<false>(
      /*input=*/{2, 4, 5, 6, 7},
      /*rows=*/{10, 11, 12, 13, 14},
      cache,
      /*predicate=*/[](int32_t index) { return index % 2 == 0; });

  EXPECT_EQ(result.returnedNumValues, 3);
  EXPECT_THAT(result.hits, ElementsAre(10, 11, 13)); // rows of 2, 4, 6
  EXPECT_THAT(result.values, ElementsAre(2, 4, 6));
  EXPECT_EQ(cache[2], FilterResult::kSuccess);
  EXPECT_EQ(cache[4], FilterResult::kSuccess);
  EXPECT_EQ(cache[6], FilterResult::kSuccess);
  EXPECT_EQ(cache[5], FilterResult::kFailure);
  EXPECT_EQ(cache[7], FilterResult::kFailure);
}

// A warm cache short-circuits: testIndex is only invoked for cache-unknown
// indices; pre-recorded kSuccess/kFailure verdicts are honored without calling
// it.
TEST(FilterDictionaryRunSimdTest, warmCacheShortCircuitsTestIndex) {
  auto cache = makeFilterCache(10);
  cache[2] = FilterResult::kSuccess;
  cache[5] = FilterResult::kFailure;
  // Index 7 stays kUnknown, so only it should reach testIndex.

  auto result = runFilterRun<false>(
      {2, 5, 7}, {10, 11, 12}, cache, [](int32_t) { return true; });

  EXPECT_EQ(result.returnedNumValues, 2); // 2 (warm pass) + 7 (cold pass)
  EXPECT_THAT(result.hits, ElementsAre(10, 12));
  EXPECT_THAT(result.values, ElementsAre(2, 7));
  EXPECT_THAT(result.testedIndices, ElementsAre(7));
  EXPECT_EQ(cache[7], FilterResult::kSuccess);
}

// With kFilterOnly the values (dict index) buffer is left untouched; only the
// row hits are emitted.
TEST(FilterDictionaryRunSimdTest, filterOnlyLeavesValuesBufferUntouched) {
  auto cache = makeFilterCache(10);
  auto result = runFilterRun</*kFilterOnly=*/true>(
      {2, 4, 5, 6, 7}, {10, 11, 12, 13, 14}, cache, [](int32_t index) {
        return index % 2 == 0;
      });

  EXPECT_EQ(result.returnedNumValues, 3);
  EXPECT_THAT(result.hits, ElementsAre(10, 11, 13));
  // The values buffer slice is still the pre-filled sentinel -> not written.
  EXPECT_THAT(result.values, ElementsAre(kSentinel, kSentinel, kSentinel));
}

// Output is appended at the supplied starting numValues offset, not overwritten
// from zero, and the returned count reflects the offset.
TEST(FilterDictionaryRunSimdTest, appendsAtStartingNumValues) {
  auto cache = makeFilterCache(10);
  auto result = runFilterRun<false>(
      {2, 5, 4},
      {10, 11, 12},
      cache,
      [](int32_t index) { return index % 2 == 0; },
      /*startNumValues=*/4);

  // 2 and 4 pass, 5 fails -> 2 appended after offset 4 -> returns 6.
  EXPECT_EQ(result.returnedNumValues, 6);
  EXPECT_THAT(result.hits, ElementsAre(10, 12));
  EXPECT_THAT(result.values, ElementsAre(2, 4));
}

// The same index appearing in a later batch hits the verdict cached by the
// earlier batch, so testIndex runs exactly once for it.
TEST(FilterDictionaryRunSimdTest, repeatedIndexAcrossBatchesHitsCache) {
  constexpr int32_t kRepeatedIndex = 2;
  const int32_t numInput = kSimdWidth + 1;
  std::vector<int32_t> input(numInput);
  std::vector<int32_t> rows(numInput);
  for (int32_t i = 0; i < numInput; ++i) {
    input[i] = 100 + i; // distinct, all != kRepeatedIndex
    rows[i] = i;
  }
  // Same index in batch 0 (pos 0) and batch 1 (pos kSimdWidth).
  input[0] = kRepeatedIndex;
  input[kSimdWidth] = kRepeatedIndex;

  auto cache = makeFilterCache(/*numIndices=*/200);
  auto result =
      runFilterRun<false>(input, rows, cache, [](int32_t) { return true; });

  const int32_t timesTested = static_cast<int32_t>(std::count(
      result.testedIndices.begin(),
      result.testedIndices.end(),
      kRepeatedIndex));
  EXPECT_EQ(timesTested, 1);
  EXPECT_EQ(cache[kRepeatedIndex], FilterResult::kSuccess);
  EXPECT_EQ(result.returnedNumValues, numInput);
}

// Exercises full and masked-tail batches across a range of input sizes; the
// expected survivors are computed independently from the same predicate.
class FilterDictionaryRunSimdSizeTest : public testing::TestWithParam<int32_t> {
};

TEST_P(FilterDictionaryRunSimdSizeTest, fullAndPartialBatches) {
  const int32_t numInput = GetParam();
  std::vector<int32_t> input(numInput);
  std::vector<int32_t> rows(numInput);
  for (int32_t i = 0; i < numInput; ++i) {
    input[i] = i;
    rows[i] = 1000 + i;
  }
  const auto predicate = [](int32_t index) { return index % 3 != 0; };

  std::vector<int32_t> expectedHits;
  std::vector<int32_t> expectedValues;
  for (int32_t i = 0; i < numInput; ++i) {
    if (predicate(i)) {
      expectedHits.push_back(1000 + i);
      expectedValues.push_back(i);
    }
  }

  auto cache = makeFilterCache(numInput + 1);
  auto result = runFilterRun<false>(input, rows, cache, predicate);

  EXPECT_EQ(
      result.returnedNumValues, static_cast<int32_t>(expectedValues.size()));
  EXPECT_THAT(result.hits, ElementsAreArray(expectedHits));
  EXPECT_THAT(result.values, ElementsAreArray(expectedValues));
}

INSTANTIATE_TEST_SUITE_P(
    Sizes,
    FilterDictionaryRunSimdSizeTest,
    testing::Values(
        1,
        kSimdWidth - 1,
        kSimdWidth,
        kSimdWidth + 1,
        2 * kSimdWidth,
        3 * kSimdWidth + 3));

} // namespace
