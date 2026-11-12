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

#include <gtest/gtest.h>

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

} // namespace
} // namespace facebook::velox::common::testutil
