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
#include <unordered_set>

#include "velox/core/QueryConfig.h"
#include "velox/exec/fuzzer/PrestoQueryRunner.h"
#ifdef VELOX_ENABLE_LOCAL_RUNNER_SERVICE
#include "velox/exec/fuzzer/VeloxQueryRunner.h"
#endif
#include "velox/expression/fuzzer/ArgTypesGenerator.h"
#include "velox/expression/fuzzer/ArgValuesGenerators.h"
#include "velox/expression/fuzzer/ExpressionFuzzer.h"
#include "velox/expression/fuzzer/FuzzerRunner.h"
#include "velox/expression/fuzzer/PrestoSkippedFunctions.h"
#include "velox/expression/fuzzer/SpecialFormSignatureGenerator.h"
#include "velox/functions/prestosql/fuzzer/DivideArgTypesGenerator.h"
#include "velox/functions/prestosql/fuzzer/FloorCeilRoundArgTypesGenerator.h"
#include "velox/functions/prestosql/fuzzer/ModulusArgTypesGenerator.h"
#include "velox/functions/prestosql/fuzzer/MultiplyArgTypesGenerator.h"
#include "velox/functions/prestosql/fuzzer/PlusMinusArgTypesGenerator.h"
#include "velox/functions/prestosql/fuzzer/SkipIPAddressArgTypesGenerator.h"

#include "velox/functions/prestosql/fuzzer/SortArrayTransformer.h"
#include "velox/functions/prestosql/fuzzer/TruncateArgTypesGenerator.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

DEFINE_int64(
    seed,
    0,
    "Initial seed for random number generator used to reproduce previous "
    "results (0 means start with random seed).");

DEFINE_string(
    presto_url,
    "",
    "Presto coordinator URI along with port. If set, we use Presto as the "
    "source of truth. Otherwise, use the Velox simplified expression evaluation. Example: "
    "--presto_url=http://127.0.0.1:8080");

#ifdef VELOX_ENABLE_LOCAL_RUNNER_SERVICE
DEFINE_string(
    velox_runner_url,
    "",
    "URI for thrift requests to LocalRunnerService. If set, we use a "
    "LocalRunnerService instance as the source of truth. Otherwise, use the "
    "Velox simplified expression evaluation. Example: "
    "--velox_runner_url=http://127.0.0.1:9091");
#endif

DEFINE_uint32(
    req_timeout_ms,
    10000,
    "Timeout in milliseconds for HTTP requests made to reference DB, "
    "such as Presto. Example: --req_timeout_ms=2000");

using namespace facebook::velox::exec::test;
using facebook::velox::exec::test::PrestoQueryRunner;
using facebook::velox::fuzzer::ArgTypesGenerator;
using facebook::velox::fuzzer::ArgValuesGenerator;
using facebook::velox::fuzzer::AtTimezoneArgValuesGenerator;
using facebook::velox::fuzzer::CastVarcharAndJsonArgValuesGenerator;
using facebook::velox::fuzzer::ExpressionFuzzer;
using facebook::velox::fuzzer::FuzzerRunner;
using facebook::velox::fuzzer::JsonExtractArgValuesGenerator;
using facebook::velox::fuzzer::JsonParseArgValuesGenerator;
using facebook::velox::fuzzer::prestoSkippedFunctions;
using facebook::velox::fuzzer::prestoSkippedFunctionsSOT;
using facebook::velox::fuzzer::S2CellIdArgValuesGenerator;
using facebook::velox::fuzzer::S2CellTokenArgValuesGenerator;
using facebook::velox::fuzzer::URLArgValuesGenerator;
using facebook::velox::test::ReferenceQueryRunner;

std::unordered_map<std::string, std::shared_ptr<ArgTypesGenerator>>
    argTypesGenerators = {
        {"plus", std::make_shared<PlusMinusArgTypesGenerator>()},
        {"minus", std::make_shared<PlusMinusArgTypesGenerator>()},
        {"multiply", std::make_shared<MultiplyArgTypesGenerator>()},
        {"divide", std::make_shared<DivideArgTypesGenerator>()},
        {"floor", std::make_shared<FloorCeilRoundArgTypesGenerator>()},
        {"ceil", std::make_shared<FloorCeilRoundArgTypesGenerator>()},
        {"round", std::make_shared<FloorCeilRoundArgTypesGenerator>()},
        {"mod", std::make_shared<ModulusArgTypesGenerator>()},
        {"truncate", std::make_shared<TruncateArgTypesGenerator>()},
        // Block IPADDRESS in containers for functions whose hash-based
        // deduplication calls compareTo() on Int128ArrayBlock, which is not
        // implemented. See: https://github.com/prestodb/presto/issues/26836
        {"distinct_from", std::make_shared<SkipIPAddressArgTypesGenerator>()},
        {"array_union", std::make_shared<SkipIPAddressArgTypesGenerator>()}};

std::unordered_map<std::string, std::shared_ptr<ExprTransformer>>
    exprTransformers = {
        {"array_intersect", std::make_shared<SortArrayTransformer>()},
        {"array_except", std::make_shared<SortArrayTransformer>()},
        {"array_duplicates", std::make_shared<SortArrayTransformer>()},
        {"map_entries", std::make_shared<SortArrayTransformer>()},
        {"map_keys", std::make_shared<SortArrayTransformer>()},
        {"map_values", std::make_shared<SortArrayTransformer>()}};

std::unordered_map<std::string, std::shared_ptr<ArgValuesGenerator>>
    argValuesGenerators = {
        {"at_timezone", std::make_shared<AtTimezoneArgValuesGenerator>()},
        {"cast", std::make_shared<CastVarcharAndJsonArgValuesGenerator>()},
        {"json_parse", std::make_shared<JsonParseArgValuesGenerator>()},
        {"json_extract", std::make_shared<JsonExtractArgValuesGenerator>()},
        {"url_extract_fragment", std::make_shared<URLArgValuesGenerator>()},
        {"url_extract_host", std::make_shared<URLArgValuesGenerator>()},
        {"url_extract_parameter", std::make_shared<URLArgValuesGenerator>()},
        {"url_extract_path", std::make_shared<URLArgValuesGenerator>()},
        {"url_extract_port", std::make_shared<URLArgValuesGenerator>()},
        {"url_extract_protocol", std::make_shared<URLArgValuesGenerator>()},
        {"url_extract_query", std::make_shared<URLArgValuesGenerator>()},
        {"s2_cell_area_sq_km", std::make_shared<S2CellIdArgValuesGenerator>()},
        {"s2_cell_contains", std::make_shared<S2CellIdArgValuesGenerator>()},
        {"s2_cell_level", std::make_shared<S2CellIdArgValuesGenerator>()},
        {"s2_cell_parent", std::make_shared<S2CellIdArgValuesGenerator>()},
        {"s2_cell_to_token", std::make_shared<S2CellIdArgValuesGenerator>()},
        {"s2_cell_from_token",
         std::make_shared<S2CellTokenArgValuesGenerator>()}};

const std::unordered_set<std::string> skipFunctionsLocalRunner{};

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  facebook::velox::functions::prestosql::registerAllScalarFunctions();
  facebook::velox::functions::prestosql::registerInternalFunctions();

  // Calls common init functions in the necessary order, initializing
  // singletons, installing proper signal handlers for better debugging
  // experience, and initialize glog and gflags.
  folly::Init init(&argc, &argv);

  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});

  std::unordered_set<std::string> skipFunctions = prestoSkippedFunctions();

  size_t initialSeed = FLAGS_seed == 0 ? std::time(nullptr) : FLAGS_seed;

  std::shared_ptr<facebook::velox::memory::MemoryPool> rootPool{
      facebook::velox::memory::memoryManager()->addRootPool()};
  std::shared_ptr<ReferenceQueryRunner> referenceQueryRunner{nullptr};
  const char* shouldAdjustTimestampToSessionTimezone{"true"};

  if (!FLAGS_presto_url.empty()) {
    // Add additional functions to skip since we are now querying Presto
    // directly and are aware of certain failures.
    const auto& skipFunctionsSOT = prestoSkippedFunctionsSOT();
    skipFunctions.insert(skipFunctionsSOT.begin(), skipFunctionsSOT.end());

    referenceQueryRunner = std::make_shared<PrestoQueryRunner>(
        rootPool.get(),
        FLAGS_presto_url,
        "expression_fuzzer",
        static_cast<std::chrono::milliseconds>(FLAGS_req_timeout_ms));
    LOG(INFO) << "Using Presto as the reference DB.";
#ifdef VELOX_ENABLE_LOCAL_RUNNER_SERVICE
  } else if (!FLAGS_velox_runner_url.empty()) {
    // LocalRunnerService sets only session_timezone on the reference side, so
    // adjust_timestamp_to_session_timezone keeps its default of false there.
    // The contender has to match it or timezone-less timestamp conversions
    // differ by the session's UTC offset on every comparison.
    shouldAdjustTimestampToSessionTimezone = "false";
    skipFunctions.insert(
        skipFunctionsLocalRunner.begin(), skipFunctionsLocalRunner.end());
    referenceQueryRunner = std::make_shared<VeloxQueryRunner>(
        rootPool.get(),
        FLAGS_velox_runner_url,
        std::chrono::milliseconds(FLAGS_req_timeout_ms));
    LOG(INFO) << "Using LocalQueryRunner as the reference engine.";
#endif
  }
  FuzzerRunner::runFromGtest(
      initialSeed,
      skipFunctions,
      exprTransformers,
      {{facebook::velox::core::QueryConfig::kSessionTimezone,
        "America/Los_Angeles"},
       {facebook::velox::core::QueryConfig::kAdjustTimestampToTimezone,
        shouldAdjustTimestampToSessionTimezone},
       {facebook::velox::core::QueryConfig::kMinRowsForPeeling, "50"}},
      argTypesGenerators,
      argValuesGenerators,
      referenceQueryRunner,
      std::make_shared<
          facebook::velox::fuzzer::SpecialFormSignatureGenerator>());
}
