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
/**
 * This file tests the performance of the content in each JsonXXXFunction.call
 * () without any use of expression framework and Velox vectors. It is the
 * pure function benchmarks.
 */
#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include "velox/functions/prestosql/SIMDJsonFunctions.h"
#include "velox/functions/prestosql/benchmarks/JsonBenchmarkUtil.h"
#include "velox/functions/prestosql/benchmarks/JsonFileReader.h"
#include "velox/functions/prestosql/json/JsonExtractor.h"
#include "velox/functions/prestosql/json/SimdJsonExtractor.h"

namespace facebook::velox::functions::prestosql {
namespace {

std::string prepareData(const std::string& fileSize) {
  JsonFileReader reader;
  return reader.readJsonStringFromFile(fileSize);
}

void VeloxJsonExtract(
    int,
    int iter,
    const std::string& fileSize,
    const std::string& jsonPath) {
  folly::BenchmarkSuspender suspender;
  auto json = prepareData(fileSize);
  auto result = velox::functions::jsonExtract(json, jsonPath);
  suspender.dismiss();
  for (auto i = 0; i < iter; i++) {
    result = velox::functions::jsonExtract(json, jsonPath);
  }
  folly::doNotOptimizeAway(result);
}

void SimdJsonExtract(
    int,
    int iter,
    const std::string& fileSize,
    const std::string& jsonPath) {
  folly::BenchmarkSuspender suspender;
  std::vector<std::string> token;
  auto json = prepareData(fileSize);
  if (!tokenize(jsonPath, token)) {
    VELOX_USER_FAIL("Invalid JSON path: {}", jsonPath);
  }
  auto result = velox::functions::simdJsonExtractString(json, token);
  suspender.dismiss();
  for (auto i = 0; i < iter; i++) {
    result = velox::functions::simdJsonExtractString(json, token);
  }
  folly::doNotOptimizeAway(result);
}

BENCHMARK_DRAW_LINE();

JSONEXTRACT_BENCHMARK_NAMED_PARAM_TWO_FUNCS(
    func,
    VeloxJsonExtract,
    SimdJsonExtract,
    100,
    1K,
    "$.statuses[0].friends_count",
    "$.statuses[0].user.entities.description.urls",
    "$.statuses[0].metadata.result_type")

JSONEXTRACT_BENCHMARK_NAMED_PARAM_TWO_FUNCS(
    func,
    VeloxJsonExtract,
    SimdJsonExtract,
    100,
    10K,
    "$.statuses[0].metadata.result_type",
    "$.statuses[5].metadata.result_type",
    "$.statuses[9].metadata.result_type")

JSONEXTRACT_BENCHMARK_NAMED_PARAM_TWO_FUNCS(
    func,
    VeloxJsonExtract,
    SimdJsonExtract,
    100,
    100K,
    "$.statuses[0].metadata.result_type",
    "$.statuses[8].metadata.result_type",
    "$.statuses[15].metadata.result_type")

JSONEXTRACT_BENCHMARK_NAMED_PARAM_TWO_FUNCS(
    func,
    VeloxJsonExtract,
    SimdJsonExtract,
    100,
    1000K,
    "$.statuses[0].metadata.result_type",
    "$.statuses[500].metadata.result_type",
    "$.statuses[999].metadata.result_type")

JSONEXTRACT_BENCHMARK_NAMED_PARAM_TWO_FUNCS(
    func,
    VeloxJsonExtract,
    SimdJsonExtract,
    100,
    10000K,
    "$.statuses[0].metadata.result_type",
    "$.statuses[5000].metadata.result_type",
    "$.statuses[9999].metadata.result_type")
BENCHMARK_DRAW_LINE();

} // namespace
} // namespace facebook::velox::functions::prestosql

int main(int argc, char** argv) {
  folly::init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
