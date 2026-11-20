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

#include "velox/connectors/hive/storage_adapters/test_common/ServerStartup.h"

#include <gtest/gtest.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox::test {
namespace {

TEST(ServerStartupTest, readyAfterSeveralChecks) {
  int numChecks = 0;
  ServerStartup::wait(
      "testServer", [&] { return ++numChecks == 3; }, [] { return true; });
  EXPECT_EQ(numChecks, 3);
}

TEST(ServerStartupTest, exited) {
  VELOX_ASSERT_THROW(
      ServerStartup::wait(
          "testServer", [] { return false; }, [] { return false; }),
      "testServer exited before it was ready");
}

TEST(ServerStartupTest, timeout) {
  VELOX_ASSERT_THROW(
      ServerStartup::wait(
          "testServer",
          [] { return false; },
          [] { return true; },
          std::chrono::seconds(1)),
      "Timed out after 1 seconds waiting for testServer to start");
}

} // namespace
} // namespace facebook::velox::test
