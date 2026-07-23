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

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/exec/GroupingSet.h"
#include "velox/exec/tests/AggregateSpillBenchmarkBase.h"
#include "velox/exec/tests/JoinSpillInputBenchmarkBase.h"

// The spiller_benchmark_* flags set below are declared in
// SpillerBenchmarkBase.h (included transitively) and defined in
// SpillerBenchmarkBase.cpp.

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace {

// Smoke test that exercises the spill benchmarks' code path end to end with a
// tiny workload so they are built and run in CI. The benchmarks themselves are
// cpp_binary targets CI never builds, so a regression in the spill path -- most
// notably the named-serde registration (getNamedVectorSerde("Presto") in
// SpillWriter/SpillReadFile) -- would otherwise only surface on a manual run.
//
// Crucially this fixture does NOT derive from OperatorTestBase: that base
// registers the named Presto serde itself in SetUp(), which would mask a
// regression in SpillerBenchmarkBase::registerSerde() and make this test pass
// even if the fix were reverted. Instead we register serdes through the exact
// same shared entry point the benchmark mains use, so dropping the named-serde
// registration fails this test.
class SpillBenchmarkSmokeTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!memory::MemoryManager::testInstance()) {
      memory::MemoryManager::initialize(memory::MemoryManager::Options{});
    }
    filesystems::registerLocalFileSystem();
    SpillerBenchmarkBase::registerSerde();
  }

  void SetUp() override {
    FLAGS_spiller_benchmark_num_spill_vectors = 4;
    FLAGS_spiller_benchmark_spill_vector_size = 16;
    FLAGS_spiller_benchmark_spill_executor_size = 1;
  }
};

TEST_F(SpillBenchmarkSmokeTest, aggregateInput) {
  const std::string spillerType{exec::AggregationInputSpiller::kType};
  AggregateSpillBenchmarkBase benchmark(spillerType);
  benchmark.setUp();
  benchmark.run();
  benchmark.cleanup();
}

TEST_F(SpillBenchmarkSmokeTest, joinInput) {
  JoinSpillInputBenchmarkBase benchmark;
  benchmark.setUp();
  benchmark.run();
  benchmark.cleanup();
}

} // namespace
