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
#include "velox/common/memory/Memory.h"
#include "velox/common/process/ThreadDebugInfo.h"

#include <folly/Unit.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

DEFINE_bool(
    enable_parallel_barrier_tests,
    false,
    "Run parallel cases of LimitTest.barrier and "
    "IndexLookupJoinTest.{barrier,equalJoin}; disabled by default pending "
    "https://github.com/facebookincubator/velox/issues/19832");

// This main is needed for some tests on linux.
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  // Signal handler required for ThreadDebugInfoTest
  facebook::velox::process::addDefaultFatalSignalHandler();
  folly::Init init(&argc, &argv, false);
  facebook::velox::memory::MemoryManager::initialize({});
  return RUN_ALL_TESTS();
}
