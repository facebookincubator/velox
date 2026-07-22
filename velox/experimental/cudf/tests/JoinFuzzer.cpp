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

/// cuDF Join Fuzzer Runner (CPU fallback enabled)
///
/// Runs the standard JoinFuzzer with cuDF registered and CPU fallback
/// enabled.  Operators that can run on GPU are routed there by the
/// DriverAdapter; unsupported operators fall back to CPU transparently.
/// Results are verified against DuckDB.
///
/// Usage:
///   ./velox_cudf_join_fuzzer --steps 100 --v=1

#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/exec/fuzzer/FuzzerUtil.h"
#include "velox/exec/fuzzer/JoinFuzzer.h"
#include "velox/exec/fuzzer/ReferenceQueryRunner.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/serializers/CompactRowSerializer.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/serializers/UnsafeRowSerializer.h"

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

using namespace facebook::velox;

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  exec::test::setupMemory(FLAGS_allocator_capacity, FLAGS_arbitrator_capacity);

  // Register cuDF with CPU fallback enabled.  Unsupported operators
  // fall back to CPU transparently.
  cudf_velox::CudfConfig::getInstance().allowCpuFallback = true;
  // Log which operators are replaced with GPU equivalents per pipeline.
  cudf_velox::CudfConfig::getInstance().debugEnabled = true;
  cudf_velox::registerCudf();

  std::shared_ptr<memory::MemoryPool> rootPool{
      memory::memoryManager()->addRootPool()};
  auto referenceQueryRunner = exec::test::setupReferenceQueryRunner(
      rootPool.get(), FLAGS_presto_url, "join_fuzzer", FLAGS_req_timeout_ms);
  const size_t initialSeed = FLAGS_seed == 0 ? std::time(nullptr) : FLAGS_seed;

  serializer::presto::PrestoVectorSerde::registerVectorSerde();
  filesystems::registerLocalFileSystem();
  functions::prestosql::registerAllScalarFunctions();
  parse::registerTypeResolver();
  if (!isRegisteredNamedVectorSerde("Presto")) {
    serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
  }
  if (!isRegisteredNamedVectorSerde("CompactRow")) {
    serializer::CompactRowVectorSerde::registerNamedVectorSerde();
  }
  if (!isRegisteredNamedVectorSerde("UnsafeRow")) {
    serializer::spark::UnsafeRowVectorSerde::registerNamedVectorSerde();
  }

  exec::joinFuzzer(initialSeed, std::move(referenceQueryRunner));

  cudf_velox::unregisterCudf();
}
