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

#include <gtest/gtest.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/ScopedTestTime.h"
#include "velox/common/time/Timer.h"

namespace {

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;

// NOTE: we can only construct ScopedTestTime in debug builds.
DEBUG_ONLY_TEST(TestScopedTestTime, testSetCurrentTimeMs) {
  {
    ScopedTestTime scopedTestTime;
    scopedTestTime.setCurrentTestTimeMs(1);
    ASSERT_EQ(EpochTime::now<std::chrono::milliseconds>(), 1);
    ASSERT_EQ(EpochTime::now<std::chrono::microseconds>(), 1000);
    scopedTestTime.setCurrentTestTimeMs(2);
    ASSERT_EQ(EpochTime::now<std::chrono::milliseconds>(), 2);
    ASSERT_EQ(EpochTime::now<std::chrono::microseconds>(), 2000);
  }

  // This should be the actual time, so we don't know what it is, but it
  // shouldn't be equal to the overridden value.
  ASSERT_NE(EpochTime::now<std::chrono::milliseconds>(), 2);
  ASSERT_NE(EpochTime::now<std::chrono::microseconds>(), 2000);
}

DEBUG_ONLY_TEST(TestScopedTestTime, testSetCurrentTimeMicro) {
  {
    ScopedTestTime scopedTestTime;
    scopedTestTime.setCurrentTestTimeMicro(1000);
    ASSERT_EQ(EpochTime::now<std::chrono::milliseconds>(), 1);
    ASSERT_EQ(EpochTime::now<std::chrono::microseconds>(), 1000);
    scopedTestTime.setCurrentTestTimeMicro(2000);
    ASSERT_EQ(EpochTime::now<std::chrono::milliseconds>(), 2);
    ASSERT_EQ(EpochTime::now<std::chrono::microseconds>(), 2000);
  }

  // This should be the actual time, so we don't know what it is, but it
  // shouldn't be equal to the overridden value.
  ASSERT_NE(EpochTime::now<std::chrono::milliseconds>(), 2);
  ASSERT_NE(EpochTime::now<std::chrono::microseconds>(), 2000);
}

DEBUG_ONLY_TEST(TestScopedTestTime, testSetCurrentTimeNano) {
  {
    ScopedTestTime scopedTestTime;
    scopedTestTime.setCurrentTestTimeNano(1000);
    ASSERT_EQ(EpochTime::now<std::chrono::microseconds>(), 1);
    ASSERT_EQ(EpochTime::now<std::chrono::nanoseconds>(), 1000);
    scopedTestTime.setCurrentTestTimeNano(2000);
    ASSERT_EQ(EpochTime::now<std::chrono::microseconds>(), 2);
    ASSERT_EQ(EpochTime::now<std::chrono::nanoseconds>(), 2000);
  }

  // This should be the actual time, so we don't know what it is, but it
  // shouldn't be equal to the overridden value.
  ASSERT_NE(EpochTime::now<std::chrono::microseconds>(), 2);
  ASSERT_NE(EpochTime::now<std::chrono::nanoseconds>(), 2000);
}

DEBUG_ONLY_TEST(TestScopedTestTime, multipleScopedTestTimes) {
  {
    ScopedTestTime scopedTestTime;
    scopedTestTime.setCurrentTestTimeMs(1);
    ASSERT_EQ(EpochTime::now<std::chrono::milliseconds>(), 1);
    ASSERT_EQ(EpochTime::now<std::chrono::microseconds>(), 1000);
  }

  {
    ScopedTestTime scopedTestTime;
    // The previous scoped test time should have been cleared.
    ASSERT_NE(EpochTime::now<std::chrono::milliseconds>(), 1);
    ASSERT_NE(EpochTime::now<std::chrono::microseconds>(), 1000);

    scopedTestTime.setCurrentTestTimeMs(1);
    ASSERT_EQ(EpochTime::now<std::chrono::milliseconds>(), 1);
    ASSERT_EQ(EpochTime::now<std::chrono::microseconds>(), 1000);

    // Trying to create another ScopedTestTime with one already in scope should
    // fail.
    auto createScopedTestTime = []() { ScopedTestTime scopedTestTime2; };
    VELOX_ASSERT_THROW(
        createScopedTestTime(),
        "Only one ScopedTestTime can be active at a time");
  }
}
} // namespace
