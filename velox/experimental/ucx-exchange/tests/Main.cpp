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
#include "velox/common/process/ThreadDebugInfo.h"
#include "velox/experimental/cudf/CudfConfig.h"

#include <folly/Unit.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

DEFINE_int32(
    exchange_log_level,
    0,
    "VLOG level for ucx-exchange modules (0=silent, 1-3=increasing verbosity)");

// This main is needed for some tests on linux.
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  // Signal handler required for ThreadDebugInfoTest
  facebook::velox::process::addDefaultFatalSignalHandler();
  folly::Init init(&argc, &argv, false);
  facebook::velox::cudf_velox::CudfConfig::getInstance().exchangeLogLevel =
      FLAGS_exchange_log_level;
  return RUN_ALL_TESTS();
}
