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

#include "velox/dwio/common/SelectiveColumnReader.h"

#include <gtest/gtest.h>

namespace facebook::velox::dwio::common {
namespace {

TEST(IsDenseTest, empty) {
  const RowSet rows(nullptr, nullptr);
  EXPECT_TRUE(isDense(rows));
}

TEST(IsDenseTest, singleElement) {
  const std::vector<int32_t> data{0};
  const RowSet rows(data.data(), data.size());
  EXPECT_TRUE(isDense(rows));
}

TEST(IsDenseTest, contiguousFromZero) {
  const std::vector<int32_t> data{0, 1, 2, 3, 4};
  const RowSet rows(data.data(), data.size());
  EXPECT_TRUE(isDense(rows));
}

TEST(IsDenseTest, sparseRows) {
  const std::vector<int32_t> data{0, 2, 4};
  const RowSet rows(data.data(), data.size());
  EXPECT_FALSE(isDense(rows));
}

TEST(IsDenseTest, startingFromNonZero) {
  const std::vector<int32_t> data{1, 2, 3};
  const RowSet rows(data.data(), data.size());
  EXPECT_FALSE(isDense(rows));
}

TEST(IsDenseTest, singleNonZeroElement) {
  const std::vector<int32_t> data{5};
  const RowSet rows(data.data(), data.size());
  EXPECT_FALSE(isDense(rows));
}

} // namespace
} // namespace facebook::velox::dwio::common
