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

#include "velox/experimental/cudf/functions/GpuExec.h"
#include "velox/experimental/cudf/types/GpuTimestamp.cuh"

#include "velox/type/SimpleFunctionTags.h"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

// Included by path: this target does not put gpu_shadows/ on its include
// path, and nothing else here includes the real BitUtil.h.
#include "velox/experimental/cudf/functions/gpu_shadows/velox/common/base/BitUtil.h"

namespace facebook::velox::gpu {
namespace {

// True when GpuExec maps T to Expected in every position a call() sees it.
template <typename T, typename Expected>
constexpr bool resolvesTo =
    std::is_same_v<typename GpuExec::resolver<T>::in_type, Expected> &&
    std::is_same_v<typename GpuExec::resolver<T>::out_type, Expected> &&
    std::is_same_v<typename GpuExec::resolver<T>::null_free_in_type, Expected>;

// Passes primitives through and maps each Velox type tag to the physical type
// a kernel reads.
TEST(GpuTypesTest, resolver) {
  static_assert(resolvesTo<bool, bool>);
  static_assert(resolvesTo<int32_t, int32_t>);
  static_assert(resolvesTo<int64_t, int64_t>);
  static_assert(resolvesTo<float, float>);
  static_assert(resolvesTo<double, double>);
  static_assert(resolvesTo<Date, int32_t>);
  static_assert(resolvesTo<IntervalYearMonth, int32_t>);
  static_assert(resolvesTo<IntervalDayTime, int64_t>);
  static_assert(resolvesTo<Time, int64_t>);
  static_assert(resolvesTo<ShortDecimal<P1, S1>, int64_t>);
  static_assert(resolvesTo<LongDecimal<P1, S1>, __int128>);
  static_assert(resolvesTo<Timestamp, GpuTimestamp>);
}

// Orders by seconds, then by nanos, and defaults to the epoch.
TEST(GpuTypesTest, gpuTimestampOrdering) {
  EXPECT_EQ(GpuTimestamp{}.seconds, 0);
  EXPECT_EQ(GpuTimestamp{}.nanos, 0u);

  // Strictly ascending, so the result of each operator on a pair follows from
  // the positions alone.
  const std::vector<GpuTimestamp> ascending = {
      {-1, 999'999'999},
      {0, 0},
      {0, 1},
      {100, 500},
      {100, 600},
      {101, 0},
  };
  for (size_t i = 0; i < ascending.size(); ++i) {
    for (size_t j = 0; j < ascending.size(); ++j) {
      SCOPED_TRACE(testing::Message() << "i=" << i << " j=" << j);
      const auto& left = ascending[i];
      const auto& right = ascending[j];
      EXPECT_EQ(left == right, i == j);
      EXPECT_EQ(left != right, i != j);
      EXPECT_EQ(left < right, i < j);
      EXPECT_EQ(left <= right, i <= j);
      EXPECT_EQ(left > right, i > j);
      EXPECT_EQ(left >= right, i >= j);
    }
  }
}

// Counts the set bits of [begin, end) across word boundaries, and returns 0
// for an empty or negative range instead of shifting by a negative amount.
TEST(GpuTypesTest, countBits) {
  const uint64_t words[3] = {
      0xB5,
      0xF0F0F0F0F0F0F0F0ULL,
      ~uint64_t{0},
  };
  constexpr int32_t kNumBits = 3 * 64;
  const auto reference = [&](int32_t begin, int32_t end) {
    int32_t count{0};
    for (int32_t bit = begin; bit < end; ++bit) {
      count += static_cast<int32_t>((words[bit / 64] >> (bit % 64)) & 1);
    }
    return count;
  };
  for (int32_t begin = 0; begin <= kNumBits; ++begin) {
    for (int32_t end = begin; end <= kNumBits; ++end) {
      ASSERT_EQ(bits::countBits(words, begin, end), reference(begin, end))
          << "begin=" << begin << " end=" << end;
    }
  }

  const std::vector<std::pair<int32_t, int32_t>> invalidRanges = {
      {-1, 64},
      {-10, -5},
      {10, 5},
  };
  for (const auto& [begin, end] : invalidRanges) {
    SCOPED_TRACE(testing::Message() << "begin=" << begin << " end=" << end);
    EXPECT_EQ(bits::countBits(words, begin, end), 0);
  }
}

} // namespace
} // namespace facebook::velox::gpu
