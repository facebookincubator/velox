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

#include "velox/common/testutil/PageBeforeGuard.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox::common::testutil {
namespace {

TEST(PageBeforeGuardTest, readPastEndDies) {
  PageBeforeGuard page;
  // Volatile, so that the compiler keeps the reads.
  volatile uint8_t* end = page.end();
  end[-1] = 7;
  EXPECT_EQ(end[-1], 7);
  EXPECT_DEATH({ [[maybe_unused]] uint8_t value = end[0]; }, "");
}

// The page holds no constructed objects, so it takes only trivial types.
template <typename T>
concept Placeable = requires(const PageBeforeGuard& page) { page.atEnd<T>(1); };
static_assert(Placeable<int64_t>);
static_assert(!Placeable<std::string>);

TEST(PageBeforeGuardTest, atEnd) {
  PageBeforeGuard page;
  auto values = page.atEnd<int64_t>(3);
  EXPECT_EQ(values.size(), 3);
  EXPECT_EQ(reinterpret_cast<uint8_t*>(values.data() + 3), page.end());

  VELOX_ASSERT_THROW(
      page.atEnd<int64_t>(size_t{1} << 30), "Values must fit in one page.");
}

TEST(PageBeforeGuardTest, copyToEnd) {
  PageBeforeGuard page;
  const std::vector<int32_t> source = {1, 2, 3};
  auto copy = page.copyToEnd(source);
  // Older gmock needs a const_iterator, which std::span lacks before C++23.
  EXPECT_THAT(
      std::vector<int32_t>(copy.begin(), copy.end()),
      testing::ElementsAre(1, 2, 3));
  EXPECT_EQ(reinterpret_cast<uint8_t*>(copy.data() + 3), page.end());

  auto list = page.copyToEnd({int16_t{4}, int16_t{5}});
  EXPECT_THAT(
      std::vector<int16_t>(list.begin(), list.end()),
      testing::ElementsAre(4, 5));
  EXPECT_EQ(reinterpret_cast<uint8_t*>(list.data() + 2), page.end());
}

} // namespace
} // namespace facebook::velox::common::testutil
