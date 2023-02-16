/**
 * This file tests the performance of the content in each JsonXXXFunction.call
 * () without any use of expression framework and Velox vectors. It is the
 * pure function benchmarks.
 */
#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include "velox/functions/prestosql/benchmarks/JsonBenchmarkUtil.h"
#include "velox/functions/prestosql/benchmarks/JsonFileReader.h"
#include "velox/functions/prestosql/json/SimdJsonExtractor.h"
#include "velox/functions/prestosql/json/JsonExtractor.h"

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
  auto json = prepareData(fileSize);
  auto result = velox::functions::SimdJsonExtractString(json, jsonPath);
  suspender.dismiss();
  for (auto i = 0; i < iter; i++) {
    result = velox::functions::SimdJsonExtractString(json, jsonPath);
  }
  folly::doNotOptimizeAway(result);
}

BENCHMARK_DRAW_LINE();
JSON_BENCHMARK_NAMED_PARAM_TWO_FUNCS(
    func,
    VeloxJsonExtract,
    SimdJsonExtract,
    100,
    100B,
    "$.statuses[0].metadata.result_type",
    "$.statuses[0].metadata.result_type",
    "$.statuses[0].metadata.result_type")

JSON_BENCHMARK_NAMED_PARAM_TWO_FUNCS(
    func,
    VeloxJsonExtract,
    SimdJsonExtract,
    100,
    1K,
    "$.statuses[0].metadata.result_type",
    "$.statuses[0].truncated",
    "$.statuses[0].in_reply_to_screen_name")

JSON_BENCHMARK_NAMED_PARAM_TWO_FUNCS(
    func,
    VeloxJsonExtract,
    SimdJsonExtract,
    100,
    10K,
    "$.statuses[0].metadata.result_type",
    "$.statuses[1].metadata.result_type",
    "$.statuses[1].retweeted_status.user.profile_sidebar_border_color")

JSON_BENCHMARK_NAMED_PARAM_TWO_FUNCS(
    func,
    VeloxJsonExtract,
    SimdJsonExtract,
    100,
    100K,
    "$.statuses[0].metadata.result_type",
    "$.statuses[8].metadata.result_type",
    "$.statuses[15].metadata.result_type")

JSON_BENCHMARK_NAMED_PARAM_TWO_FUNCS(
    func,
    VeloxJsonExtract,
    SimdJsonExtract,
    100,
    1M,
    "$.statuses[0].metadata.result_type",
    "$.statuses[141].metadata.result_type",
    "$.search_metadata.since_id_str")

JSON_BENCHMARK_NAMED_PARAM_TWO_FUNCS(
    func,
    VeloxJsonExtract,
    SimdJsonExtract,
    100,
    10M,
    "$.statuses[0].metadata.result_type",
    "$.statuses[1442].metadata.result_type",
    "$.search_metadata.since_id_str")
BENCHMARK_DRAW_LINE();

} // namespace
} // namespace facebook::velox::functions::prestosql

int main(int argc, char** argv) {
  folly::init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
