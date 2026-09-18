/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/dwio/nimble/fuzzer/stripe_stats_pruning/StripeStatsPruningFuzzer.h"

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"

namespace facebook::nimble::fuzzer {
namespace {

class StripeStatsPruningFuzzerTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    rootPool_ = velox::memory::memoryManager()->addRootPool(
        "stripe_stats_pruning_fuzzer_test");
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
};

// Fixed seed so a failure here is reproducible; the standalone binary sweeps
// other seeds.
TEST_F(StripeStatsPruningFuzzerTest, prunedReadMatchesPredicate) {
  StripeStatsPruningFuzzerOptions options;
  options.numIterations = 48;
  options.seed = 0xB16017;

  StripeStatsPruningFuzzer fuzzer{options, *rootPool_};
  const auto stats = fuzzer.run();

  EXPECT_EQ(stats.numIterations, options.numIterations);
  // A run that pruned nothing would pass the correctness check vacuously.
  EXPECT_GT(stats.numStripesSkipped, 0);
  // Nulls forbid skipping a stripe under a null-admitting filter, so the run
  // must reach that path rather than only testing non-null keys.
  EXPECT_GT(stats.numNullableKeyIterations, 0);
}

} // namespace
} // namespace facebook::nimble::fuzzer
