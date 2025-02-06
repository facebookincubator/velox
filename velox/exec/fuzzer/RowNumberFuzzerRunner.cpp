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

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <unordered_set>
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/exec/MemoryReclaimer.h"
#include "velox/exec/fuzzer/DuckQueryRunner.h"
#include "velox/exec/fuzzer/FuzzerUtil.h"
#include "velox/exec/fuzzer/PrestoQueryRunner.h"
#include "velox/exec/fuzzer/ReferenceQueryRunner.h"
#include "velox/exec/fuzzer/RowNumberFuzzer.h"
#include "velox/serializers/PrestoSerializer.h"

/// RowNumberFuzzerRunner leverages RowNumberFuzzer and VectorFuzzer to
/// automatically generate and execute the fuzzer. It works as follows:
///
///  1. Plan Generation: Generate two equivalent query plans, one is row-number
///     over ValuesNode and the other is over TableScanNode.
///  2. Executes a variety of logically equivalent query plans and checks the
///     results are the same.
///  3. Rinse and repeat.
///
/// It is used as follows:
///
///  $ ./velox_row_number_fuzzer --duration_sec 600
///
/// The flags that configure RowNumberFuzzer's behavior are:
///
///  --steps: how many iterations to run.
///  --duration_sec: alternatively, for how many seconds it should run (takes
///          precedence over --steps).
///  --seed: pass a deterministic seed to reproduce the behavior (each iteration
///          will print a seed as part of the logs).
///  --v=1: verbose logging; print a lot more details about the execution.
///  --batch_size: size of input vector batches generated.
///  --num_batches: number if input vector batches to generate.
///  --enable_spill: test plans with spilling enabled.
///  --enable_oom_injection: randomly trigger OOM while executing query plans.
/// e.g:
///
///  $ ./velox_row_number_fuzzer \
///         --seed 123 \
///         --duration_sec 600 \
///         --v=1

DEFINE_int64(
    seed,
    0,
    "Initial seed for random number generator used to reproduce previous "
    "results (0 means start with random seed).");

DEFINE_string(
    presto_url,
    "",
    "Presto coordinator URI along with port. If set, we use Presto "
    "source of truth. Otherwise, use DuckDB. Example: "
    "--presto_url=http://127.0.0.1:8080");

DEFINE_uint32(
    req_timeout_ms,
    1000,
    "Timeout in milliseconds for HTTP requests made to reference DB, "
    "such as Presto. Example: --req_timeout_ms=2000");

DEFINE_int64(allocator_capacity, 8L << 30, "Allocator capacity in bytes.");

DEFINE_int64(arbitrator_capacity, 6L << 30, "Arbitrator capacity in bytes.");

using namespace facebook::velox::exec;

namespace {
std::unique_ptr<test::ReferenceQueryRunner> setupReferenceQueryRunner(
    facebook::velox::memory::MemoryPool* aggregatePool,
    const std::string& prestoUrl,
    const std::string& runnerName,
    const uint32_t& reqTimeoutMs) {
  if (prestoUrl.empty()) {
    auto duckQueryRunner =
        std::make_unique<test::DuckQueryRunner>(aggregatePool);
    LOG(INFO) << "Using DuckDB as the reference DB.";
    return duckQueryRunner;
  }

  LOG(INFO) << "Using Presto as the reference DB.";
  return std::make_unique<test::PrestoQueryRunner>(
      aggregatePool,
      prestoUrl,
      runnerName,
      static_cast<std::chrono::milliseconds>(reqTimeoutMs));
}
} // namespace

int main(int argc, char** argv) {
  // Calls common init functions in the necessary order, initializing
  // singletons, installing proper signal handlers for better debugging
  // experience, and initialize glog and gflags.
  folly::Init init(&argc, &argv);
  test::setupMemory(FLAGS_allocator_capacity, FLAGS_arbitrator_capacity);
  std::shared_ptr<facebook::velox::memory::MemoryPool> rootPool{
      facebook::velox::memory::memoryManager()->addRootPool()};
  auto referenceQueryRunner = setupReferenceQueryRunner(
      rootPool.get(),
      FLAGS_presto_url,
      "row_number_fuzzer",
      FLAGS_req_timeout_ms);
  const size_t initialSeed = FLAGS_seed == 0 ? std::time(nullptr) : FLAGS_seed;
  facebook::velox::serializer::presto::PrestoVectorSerde::registerVectorSerde();
  facebook::velox::filesystems::registerLocalFileSystem();
  rowNumberFuzzer(initialSeed, std::move(referenceQueryRunner));
}
