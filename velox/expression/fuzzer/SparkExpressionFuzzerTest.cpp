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

#include <folly/String.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/exec/fuzzer/ReferenceQueryRunner.h"
#include "velox/expression/fuzzer/FuzzerRunner.h"
#include "velox/expression/fuzzer/SparkSpecialFormSignatureGenerator.h"
#include "velox/functions/prestosql/fuzzer/FloorAndRoundArgTypesGenerator.h"
#include "velox/functions/sparksql/fuzzer/AddSubtractArgTypesGenerator.h"
#include "velox/functions/sparksql/fuzzer/DivideArgTypesGenerator.h"
#include "velox/functions/sparksql/fuzzer/MakeTimestampArgTypesGenerator.h"
#include "velox/functions/sparksql/fuzzer/MultiplyArgTypesGenerator.h"
#include "velox/functions/sparksql/fuzzer/SparkQueryRunner.h"
#include "velox/functions/sparksql/fuzzer/UnscaledValueArgTypesGenerator.h"
#include "velox/functions/sparksql/registration/Register.h"

using namespace facebook::velox::functions::sparksql::fuzzer;
using facebook::velox::fuzzer::ArgTypesGenerator;
using facebook::velox::test::ReferenceQueryRunner;

DEFINE_int64(
    seed,
    123456,
    "Initial seed for random number generator "
    "(use it to reproduce previous results).");

DEFINE_string(
    spark_url,
    "",
    "Spark connect server URI along with port. If set, we use Spark as the "
    "source of truth. Otherwise, use the Velox simplified expression evaluation. Example: "
    "--spark_url=localhost:15002");

using facebook::velox::fuzzer::FuzzerRunner;

int main(int argc, char** argv) {
  facebook::velox::functions::sparksql::registerFunctions("");

  ::testing::InitGoogleTest(&argc, argv);

  // Calls common init functions in the necessary order, initializing
  // singletons, installing proper signal handlers for better debugging
  // experience, and initialize glog and gflags.
  folly::Init init(&argc, &argv);
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});

  facebook::velox::parquet::registerParquetReaderFactory();
  facebook::velox::parquet::registerParquetWriterFactory();

  // The following list are the Spark UDFs that hit issues
  // For rlike you need the following combo in the only list:
  // rlike, md5 and upper
  std::unordered_set<std::string> skipFunctions = {
      "regexp_extract",
      // https://github.com/facebookincubator/velox/issues/8438
      "regexp_replace",
      "rlike",
      "chr",
      "replace",
      "might_contain",
      // unix_timestamp with empty parameter returns current unix timestamp so
      // the results are different for each evaluation.
      "unix_timestamp",
      // from_unixtime throws VeloxRuntimeError when the timestamp is out of the
      // supported range.
      "from_unixtime",
      // timestamp_millis(bigint) can generate timestamps out of the supported
      // range that make other functions throw VeloxRuntimeErrors.
      "timestamp_millis(bigint) -> timestamp",
  };

  // Functions need to be skipped when using Spark as the source of truth.
  std::unordered_set<std::string> skipFunctionsSpark = {
      // Spark internal functions.
      "get_timestamp",
      "might_contain",
      "unscaled_value",
      "empty2null",
      "varchar_type_write_side_check",
      "char_type_write_side_check",
      // Used when Spark configuration
      // "spark.sql.decimalOperations.allowPrecisionLoss" is disabled.
      "add_deny_precision_loss",
      "subtract_deny_precision_loss",
      "multiply_deny_precision_loss",
      "divide_deny_precision_loss",
      // Allows different precisions and scales for inputs, which is different
      // with Spark comparison functions but can be used when the cast before
      // comparison is dropped.
      "decimal_greaterthan",
      "decimal_greaterthanorequal",
      "decimal_lessthan",
      "decimal_lessthanorequal",
      "decimal_equalto",
      "decimal_notequalto",
      // Depends on Spark partition id.
      "spark_partition_id",
      // unix_timestamp with empty parameter returns current unix timestamp so
      // the results are different for each evaluation.
      "unix_timestamp() -> bigint",
      // Result mismatch.
      "unix_timestamp",
      "to_unix_timestamp",
      "from_unixtime",
      "date_sub",
      "bit_get",
      "round",
      "multiply",
      "divide",
      "checked_divide",
      "equalnullsafe",
      "between",
      // String functions.
      "regexp_extract",
      // https: // github.com/facebookincubator/velox/issues/8438
      "regexp_replace",
      "rlike",
      "like",
      "reverse",
      "startswith",
      "contains",
      "locate",
      "lpad",
      "rpad",
      "overlay",
      "length",
      "repeat",
      "mask",
      "base64",
      "unbase64",
      // Spark throw but Velox return NULL.
      // https://github.com/facebookincubator/velox/issues/14059
      "add_months",
      "to_utc_timestamp",
      "from_utc_timestamp",
      "timestamp_millis",
      "date_format",
      // Spark throws when unit is invalid.
      "timestampdiff",
      // Spark throws when input type is not supported.
      "to_json",
      "make_ym_interval",
      "timestampadd",
  };

  // Required by spark_partition_id function.
  std::unordered_map<std::string, std::string> queryConfigs = {
      {facebook::velox::core::QueryConfig::kSparkPartitionId, "123"},
      {facebook::velox::core::QueryConfig::kSessionTimezone,
       "America/Los_Angeles"}};

  std::unordered_map<std::string, std::shared_ptr<ArgTypesGenerator>>
      argTypesGenerators = {
          {"add", std::make_shared<AddSubtractArgTypesGenerator>(true)},
          {"add_deny_precision_loss",
           std::make_shared<AddSubtractArgTypesGenerator>(false)},
          {"subtract", std::make_shared<AddSubtractArgTypesGenerator>(true)},
          {"subtract_deny_precision_loss",
           std::make_shared<AddSubtractArgTypesGenerator>(false)},
          {"multiply", std::make_shared<MultiplyArgTypesGenerator>(true)},
          {"multiply_deny_precision_loss",
           std::make_shared<MultiplyArgTypesGenerator>(false)},
          {"divide", std::make_shared<DivideArgTypesGenerator>(true)},
          {"divide_deny_precision_loss",
           std::make_shared<DivideArgTypesGenerator>(false)},
          {"ceil",
           std::make_shared<
               facebook::velox::exec::test::FloorAndRoundArgTypesGenerator>()},
          {"floor",
           std::make_shared<
               facebook::velox::exec::test::FloorAndRoundArgTypesGenerator>()},
          {"unscaled_value",
           std::make_shared<UnscaledValueArgTypesGenerator>()},
          {"make_timestamp",
           std::make_shared<MakeTimestampArgTypesGenerator>()}};

  std::shared_ptr<facebook::velox::memory::MemoryPool> rootPool{
      facebook::velox::memory::memoryManager()->addRootPool()};
  std::shared_ptr<ReferenceQueryRunner> referenceQueryRunner{nullptr};
  const bool useSpark = !FLAGS_spark_url.empty();
  if (useSpark) {
    referenceQueryRunner = std::make_shared<SparkQueryRunner>(
        rootPool.get(), FLAGS_spark_url, "fuzzer", "expression");
    LOG(INFO) << "Using Spark as the reference DB.";
  }

  return FuzzerRunner::run(
      FLAGS_seed,
      useSpark ? skipFunctionsSpark : skipFunctions,
      {{}},
      queryConfigs,
      argTypesGenerators,
      {{}},
      referenceQueryRunner,
      std::make_shared<
          facebook::velox::fuzzer::SparkSpecialFormSignatureGenerator>());
}
