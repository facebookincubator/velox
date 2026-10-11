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

#include "velox/common/process/ProcessBase.h"

#include <gtest/gtest.h>
#include <filesystem>
#include <future>
#include <thread>

namespace facebook::velox::process {

TEST(ProcessBaseTest, identity) {
  EXPECT_GT(getProcessId(), 0);
  EXPECT_FALSE(getAppName().empty());
  EXPECT_FALSE(getHostName().empty());
  EXPECT_TRUE(
      std::filesystem::equivalent(
          getCurrentDirectory(), std::filesystem::current_path()));
}

TEST(ProcessBaseTest, threadIdentityAndCpuClock) {
  const auto parentId = getThreadId();
  EXPECT_TRUE(pthread_equal(parentId, getThreadId()));
  std::promise<pthread_t> promise;
  auto future = promise.get_future();
  std::thread worker([&] {
    const auto start = threadCpuNanos();
    promise.set_value(getThreadId());
    EXPECT_GE(threadCpuNanos(), start);
  });
  const auto workerId = future.get();
  worker.join();
  EXPECT_FALSE(pthread_equal(parentId, workerId));
}

} // namespace facebook::velox::process
