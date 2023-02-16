/**
 * This file tests the performance of each JsonXXXFunction.call() with
 * expression framework and Velox vectors.
 */
#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/JsonFunctions.h"
#include "velox/functions/prestosql/SIMDJsonFunctions.h"
#include "velox/functions/prestosql/benchmarks/JsonBenchmarkUtil.h"
#include "velox/functions/prestosql/benchmarks/JsonFileReader.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

namespace facebook::velox::functions::prestosql {
namespace {

/// This function is only for test.
template <typename T>
struct JsonExtractFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Json>& json,
      const arg_type<Varchar>& jsonPath) {
    const folly::StringPiece& jsonStringPiece = json;
    const folly::StringPiece& jsonPathStringPiece = jsonPath;

    auto extractResult = jsonExtract(jsonStringPiece, jsonPathStringPiece);

    if (extractResult.hasValue()) {
      UDFOutputString::assign(result, folly::toJson(extractResult.value()));
      return true;
    } else {
      return false;
    }
  }
};

class JsonBenchmark : public velox::functions::test::FunctionBenchmarkBase {
 public:
  JsonBenchmark() : FunctionBenchmarkBase() {
    velox::functions::prestosql::registerJsonFunctions();
    velox::functions::prestosql::registerSIMDJsonFunctions();
    registerFunction<JsonExtractFunction, Varchar, Json, Varchar>(
        {"json_extract"});
  }

  std::string prepareData(const std::string& fileSize) {
    JsonFileReader reader;
    return reader.readJsonStringFromFile(fileSize);
  }

  velox::VectorPtr makeJsonData(const std::string& json, int vectorSize) {
    auto jsonVector = vectorMaker_.flatVector<velox::StringView>(vectorSize);
    for (auto i = 0; i < vectorSize; i++) {
      jsonVector->set(i, velox::StringView(json));
    }
    return jsonVector;
  }

  void runWithJsonPath(
      int iter,
      int vectorSize,
      const std::string& fnName,
      const std::string& json,
      const std::string& jsonPath) {
    folly::BenchmarkSuspender suspender;

    auto jsonVector = makeJsonData(json, vectorSize);
    auto jsonPathVector = velox::BaseVector::createConstant(
        jsonPath.data(), vectorSize, execCtx_.pool());

    auto rowVector = vectorMaker_.rowVector({jsonVector, jsonPathVector});
    auto exprSet =
        compileExpression(fmt::format("{}(c0, c1)", fnName), rowVector->type());
    suspender.dismiss();
    doRun(iter, exprSet, rowVector);
  }

  void doRun(
      const int iter,
      velox::exec::ExprSet& exprSet,
      const velox::RowVectorPtr& rowVector) {
    uint32_t cnt = 0;
    for (auto i = 0; i < iter; i++) {
      cnt += evaluate(exprSet, rowVector)->size();
    }
    folly::doNotOptimizeAway(cnt);
  }
};

void VeloxJsonExtract(
    int iter,
    int vectorSize,
    const std::string& fileSize,
    const std::string& jsonPath) {
  folly::BenchmarkSuspender suspender;
  JsonBenchmark benchmark;
  auto json = benchmark.prepareData(fileSize);
  suspender.dismiss();
  benchmark.runWithJsonPath(iter, vectorSize, "json_extract", json, jsonPath);
}

void SIMDJsonExtract(
    int iter,
    int vectorSize,
    const std::string& fileSize,
    const std::string& jsonPath) {
  folly::BenchmarkSuspender suspender;
  JsonBenchmark benchmark;
  auto json = benchmark.prepareData(fileSize);
  suspender.dismiss();
  benchmark.runWithJsonPath(
      iter, vectorSize, "simd_json_extract_scalar", json, jsonPath);
}

BENCHMARK_DRAW_LINE();
JSON_BENCHMARK_NAMED_PARAM_TWO_FUNCS(
    expr,
    VeloxJsonExtract,
    SIMDJsonExtract,
    100,
    100B,
    "$.statuses[0].metadata.result_type",
    "$.statuses[0].metadata.result_type",
    "$.statuses[0].metadata.result_type")

JSON_BENCHMARK_NAMED_PARAM_TWO_FUNCS(
    func,
    VeloxJsonExtract,
    SIMDJsonExtract,
    100,
    1K,
    "$.statuses[0].metadata.result_type",
    "$.statuses[0].truncated",
    "$.statuses[0].in_reply_to_screen_name")

JSON_BENCHMARK_NAMED_PARAM_TWO_FUNCS(
    func,
    VeloxJsonExtract,
    SIMDJsonExtract,
    100,
    10K,
    "$.statuses[0].metadata.result_type",
    "$.statuses[1].metadata.result_type",
    "$.statuses[1].retweeted_status.user.profile_sidebar_border_color")

JSON_BENCHMARK_NAMED_PARAM_TWO_FUNCS(
    func,
    VeloxJsonExtract,
    SIMDJsonExtract,
    100,
    100K,
    "$.statuses[0].metadata.result_type",
    "$.statuses[8].metadata.result_type",
    "$.statuses[15].metadata.result_type")

JSON_BENCHMARK_NAMED_PARAM_TWO_FUNCS(
    func,
    VeloxJsonExtract,
    SIMDJsonExtract,
    100,
    1M,
    "$.statuses[0].metadata.result_type",
    "$.statuses[141].metadata.result_type",
    "$.search_metadata.since_id_str")

JSON_BENCHMARK_NAMED_PARAM_TWO_FUNCS(
    func,
    VeloxJsonExtract,
    SIMDJsonExtract,
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
