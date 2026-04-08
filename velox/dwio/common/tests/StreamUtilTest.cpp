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

#include "velox/dwio/common/StreamUtil.h"

#include <gtest/gtest.h>

namespace facebook::velox::dwio::common {
namespace {

template <int kStep>
void testRowLoop(
    const int32_t* rows,
    int32_t begin,
    int32_t end,
    const std::vector<int32_t>& expectedDense,
    const std::vector<int32_t>& expectedSparse,
    const std::vector<std::pair<int32_t, int32_t>>& expectedSparseN) {
  std::vector<int32_t> actualDense, actualSparse;
  std::vector<std::pair<int32_t, int32_t>> actualSparseN;
  rowLoop<kStep>(
      rows,
      begin,
      end,
      [&](auto i) { actualDense.push_back(i); },
      [&](auto i) { actualSparse.push_back(i); },
      [&](auto i, auto size) { actualSparseN.emplace_back(i, size); });
  ASSERT_EQ(actualDense, expectedDense);
  ASSERT_EQ(actualSparse, expectedSparse);
  ASSERT_EQ(actualSparseN, expectedSparseN);
}

TEST(StreamUtilTest, rowLoop) {
  const int32_t rows[] = {
      0,  1,  2,  3,  4, // Dense
      5,  7,  9,  11, 13, // Sparse
      14, 15, 16, 17, 18, // Dense
      19, 21, 23, 25, 27, // Sparse
  };
  testRowLoop<2>(rows, 0, 6, {0, 2, 4}, {}, {});
  testRowLoop<4>(rows, 0, 6, {0}, {}, {{4, 2}});
  testRowLoop<4>(rows, 0, 10, {0}, {4}, {{8, 2}});
  testRowLoop<4>(rows, 10, 20, {10}, {14}, {{18, 2}});
}

} // namespace
} // namespace facebook::velox::dwio::common
