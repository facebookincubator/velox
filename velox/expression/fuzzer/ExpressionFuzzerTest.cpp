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
#include <gtest/gtest.h>
#include <unordered_set>

#include "velox/expression/fuzzer/ArgGenerator.h"
#include "velox/expression/fuzzer/FuzzerRunner.h"
#include "velox/functions/prestosql/fuzzer/DivideArgGenerator.h"
#include "velox/functions/prestosql/fuzzer/FloorAndRoundArgGenerator.h"
#include "velox/functions/prestosql/fuzzer/ModulusArgGenerator.h"
#include "velox/functions/prestosql/fuzzer/MultiplyArgGenerator.h"
#include "velox/functions/prestosql/fuzzer/PlusMinusArgGenerator.h"
#include "velox/functions/prestosql/fuzzer/TruncateArgGenerator.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/exec/fuzzer/PrestoQueryRunner.h"

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
    10000,
    "Timeout in milliseconds for HTTP requests made to reference DB, "
    "such as Presto. Example: --req_timeout_ms=2000");

using namespace facebook::velox::exec::test;
using facebook::velox::fuzzer::ArgGenerator;
using facebook::velox::fuzzer::FuzzerRunner;
using facebook::velox::exec::test::PrestoQueryRunner;
using facebook::velox::test::ReferenceQueryRunner;

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Calls common init functions in the necessary order, initializing
  // singletons, installing proper signal handlers for better debugging
  // experience, and initialize glog and gflags.
  folly::Init init(&argc, &argv);

  facebook::velox::functions::prestosql::registerAllScalarFunctions();
  facebook::velox::memory::MemoryManager::initialize({});

  // TODO: List of the functions that at some point crash or fail and need to
  // be fixed before we can enable.
  // This list can include a mix of function names and function signatures.
  // Use function name to exclude all signatures of a given function from
  // testing. Use function signature to exclude only a specific signature.
  std::unordered_set<std::string> skipFunctions = {
      // Fuzzer and the underlying engine are confused about cardinality(HLL)
      // (since HLL is a user defined type), and end up trying to use
      // cardinality passing a VARBINARY (since HLL's implementation uses an
      // alias to VARBINARY).
      "cardinality",
      "element_at",
      "width_bucket",
      // Fuzzer cannot generate valid 'comparator' lambda.
      "array_sort(array(T),constant function(T,T,bigint)) -> array(T)",
      "split_to_map(varchar,varchar,varchar,function(varchar,varchar,varchar,varchar)) -> map(varchar,varchar)",
      // https://github.com/facebookincubator/velox/issues/8919
      "plus(date,interval year to month) -> date",
      "minus(date,interval year to month) -> date",
      "plus(timestamp,interval year to month) -> timestamp",
      "plus(interval year to month,timestamp) -> timestamp",
      "minus(timestamp,interval year to month) -> timestamp",
      // https://github.com/facebookincubator/velox/issues/8438#issuecomment-1907234044
      "regexp_extract",
      "regexp_extract_all",
      "regexp_like",
      "regexp_replace",
      "regexp_split",
      // date_format and format_datetime throw VeloxRuntimeError when input
      // timestamp is out of the supported range.
      "date_format",
      "format_datetime",
      // from_unixtime can generate timestamps out of the supported range that
      // make other functions throw VeloxRuntimeErrors.
      "from_unixtime",
      // Presto not support
      "plus",
      "minus",
      "divide",
      "multiply",
      "subscript",
      "array_sort", //for array_sort with lambda, --seed=1144235377
      "array_sort_desc", //same as above
      "array_remove", //different try behavior
      "is_null", //not registered in Presto
      "codepoint", // expect varchar(1) as parameter type
      "json_array_contains", //Velox throws, Presto returns NULL, SELECT json_array_contains('{asce', '{asce')
      "like", // Presto not supporting this as function name
      "lt",
      "switch",
      "eq",
      "neq",
      "le",
      "ge",
      "gt",
      "negate",
      "clamp",
      "between",
      // --enable_variadic_signatures --velox_fuzzer_enable_complex_types --lazy_vector_generation_ratio 0.2 --velox_fuzzer_enable_column_reuse --velox_fuzzer_enable_expression_reuse --max_expression_trees_per_step 2 --duration_sec 60 --logtostderr=1 --minloglevel=0 --presto_url=http://127.0.0.1:8080 --batch_size=10 --seed=374405688
      
  };
  size_t initialSeed = FLAGS_seed == 0 ? std::time(nullptr) : FLAGS_seed;

  std::unordered_map<std::string, std::shared_ptr<ArgGenerator>> argGenerators =
      {{"plus", std::make_shared<PlusMinusArgGenerator>()},
       {"minus", std::make_shared<PlusMinusArgGenerator>()},
       {"multiply", std::make_shared<MultiplyArgGenerator>()},
       {"divide", std::make_shared<DivideArgGenerator>()},
       {"floor", std::make_shared<FloorAndRoundArgGenerator>()},
       {"round", std::make_shared<FloorAndRoundArgGenerator>()},
       {"mod", std::make_shared<ModulusArgGenerator>()},
       {"truncate", std::make_shared<TruncateArgGenerator>()}};

  std::shared_ptr<facebook::velox::memory::MemoryPool> rootPool{
      facebook::velox::memory::memoryManager()->addRootPool()};
  std::shared_ptr<ReferenceQueryRunner> referenceQueryRunner{nullptr};
  if (!FLAGS_presto_url.empty()) {
    referenceQueryRunner = std::make_shared<PrestoQueryRunner>(
        rootPool.get(),
        FLAGS_presto_url,
        "expression_fuzzer",
        static_cast<std::chrono::milliseconds>(FLAGS_req_timeout_ms));
    LOG(INFO) << "Using Presto as the reference DB.";
  }
  return FuzzerRunner::run(
      initialSeed, skipFunctions, {{"session_timezone", "America/Los_Angeles"}}, argGenerators, referenceQueryRunner);
}
