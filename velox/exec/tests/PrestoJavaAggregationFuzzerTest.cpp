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

#include "velox/exec/fuzzer/PrestoQueryRunner.h"
#include "velox/exec/tests/AggregationFuzzerCommon.h"

DEFINE_string(
    presto_url,
    "",
    "Presto coordinator URI along with port. This is required to use Presto "
    "as source of truth. Example: "
    "--presto_url=http://127.0.0.1:8080");

int main(int argc, char** argv) {
  using namespace facebook::velox::exec::test;

  VELOX_CHECK(
      !FLAGS_presto_url.empty(),
      "Presto Aggregation fuzzer requires Presto"
      " coordinator URL!");

  // List of presto specific functions that have known bugs that cause crashes
  // or failures.
  static const std::unordered_set<std::string> skipFunctions = {
      // https://github.com/facebookincubator/velox/issues/3493
      "stddev_pop",
      // Lambda functions are not supported yet.
      "reduce_agg",
  };

  auto prestoQueryRunner = std::make_unique<PrestoQueryRunner>(
      FLAGS_presto_url, "aggregation_fuzzer");

  return runAggregationFuzzer(
      argc, argv, skipFunctions, std::move(prestoQueryRunner));
}
