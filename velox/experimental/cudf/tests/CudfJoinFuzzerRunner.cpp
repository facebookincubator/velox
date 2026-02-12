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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/CudfJoinFuzzer.h"

#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/exec/fuzzer/FuzzerUtil.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/serializers/CompactRowSerializer.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/serializers/UnsafeRowSerializer.h"

#include <folly/init/Init.h>
#include <gflags/gflags.h>

/// cuDF Join FuzzerRunner leverages CudfJoinFuzzer and VectorFuzzer to
/// automatically generate and execute GPU hash join tests. It works by:
///
///  1. Picking a random join type supported by cuDF.
///  2. Generating a random set of input data (vectors), with a variety of
///     encodings and data layouts.
///  3. Executing the join using cuDF GPU operators.
///  4. Comparing results against DuckDB reference.
///  5. Rinse and repeat.
///
/// The common usage pattern is as following:
///
///  $ ./velox_cudf_join_fuzzer --steps 10000
///
/// The important flags that control the fuzzer's behavior are:
///
///  --steps: how many iterations to run.
///  --duration_sec: alternatively, for how many seconds it should run (takes
///          precedence over --steps).
///  --seed: pass a deterministic seed to reproduce the behavior (each iteration
///          will print a seed as part of the logs).
///  --v=1: verbose logging; print a lot more details about the execution.
///  --batch_size: size of input vector batches generated.
///  --num_batches: number of input vector batches to generate.
///
/// e.g:
///
///  $ ./velox_cudf_join_fuzzer \
///         --steps 10000 \
///         --seed 123 \
///         --v=1

DEFINE_int64(
    seed,
    0,
    "Initial seed for random number generator used to reproduce previous "
    "results (0 means start with random seed).");

DEFINE_int64(allocator_capacity, 8L << 30, "Allocator capacity in bytes.");

DEFINE_int64(arbitrator_capacity, 6L << 30, "Arbitrator capacity in bytes.");

using namespace facebook::velox;

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);

  exec::test::setupMemory(FLAGS_allocator_capacity, FLAGS_arbitrator_capacity);

  // Register cuDF operators - this causes hash joins to use GPU.
  cudf_velox::registerCudf();

  // Disable CPU fallback to ensure we're testing GPU code paths.
  cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;

  std::shared_ptr<memory::MemoryPool> rootPool{
      memory::memoryManager()->addRootPool()};
  auto referenceQueryRunner = exec::test::setupReferenceQueryRunner(
      rootPool.get(), "", "cudf_join_fuzzer", 1000);
  const size_t initialSeed = FLAGS_seed == 0 ? std::time(nullptr) : FLAGS_seed;

  serializer::presto::PrestoVectorSerde::registerVectorSerde();
  filesystems::registerLocalFileSystem();
  functions::prestosql::registerAllScalarFunctions();
  parse::registerTypeResolver();

  if (!isRegisteredNamedVectorSerde(VectorSerde::Kind::kPresto)) {
    serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
  }
  if (!isRegisteredNamedVectorSerde(VectorSerde::Kind::kCompactRow)) {
    serializer::CompactRowVectorSerde::registerNamedVectorSerde();
  }
  if (!isRegisteredNamedVectorSerde(VectorSerde::Kind::kUnsafeRow)) {
    serializer::spark::UnsafeRowVectorSerde::registerNamedVectorSerde();
  }

  cudf_velox::test::cudfJoinFuzzer(
      initialSeed, std::move(referenceQueryRunner));

  cudf_velox::unregisterCudf();
  return 0;
}
