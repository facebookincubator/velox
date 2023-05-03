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

#include "velox/common/base/Cleanup.h"

#include <gtest/gtest.h>

namespace facebook {
namespace velox {

TEST(CleanupTest, basic) {
  int count = 0;
  {
    MakeCleanup cleanup([&]{ ++count; });
    ASSERT_EQ(count, 0);
  }
  ASSERT_EQ(count, 1);
  {
    MakeCleanup cleanup([&]{ ++count; });
    ASSERT_EQ(count, 1);
  }
  ASSERT_EQ(count, 2);
  {
    MakeCleanup cleanup([&]{ ++count; });
    ASSERT_EQ(count, 2);
    cleanup.cancel();
    ASSERT_EQ(count, 2);
  }
  ASSERT_EQ(count, 2);
  {
    MakeCleanup cleanup([&]{ ++count; });
    ASSERT_EQ(count, 2);
  }
  ASSERT_EQ(count, 3);
}

} // namespace velox
} // namespace facebook
