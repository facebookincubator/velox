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

// Included by path: this target does not put gpu_shadows/ on its include
// path, and nothing else in this translation unit includes the real BitUtil.h,
// whose countBits the shadow redefines for the device.
#include "velox/experimental/cudf/functions/gpu_shadows/velox/common/base/BitUtil.h"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

namespace facebook::velox::bits {
namespace {

// Counts the set bits of [begin, end) across word boundaries, and returns 0
// for an empty or negative range instead of shifting by a negative amount.
TEST(GpuShadowBitUtilTest, countBits) {
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
      ASSERT_EQ(countBits(words, begin, end), reference(begin, end))
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
    EXPECT_EQ(countBits(words, begin, end), 0);
  }
}

} // namespace
} // namespace facebook::velox::bits
