#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include "velox/connectors/hive/HivePartitionFunction.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

namespace facebook::velox::test {
namespace {

void BM_HivePartFunc(
    size_t,
    size_t vectorSize,
    size_t bucketCount,
    const TypePtr& type) {
  functions::test::FunctionBenchmarkBase benchmarkBase;
  folly::BenchmarkSuspender suspender;

  VectorFuzzer::Options opts;
  opts.vectorSize = vectorSize;
  VectorFuzzer fuzzer(opts, benchmarkBase.pool());
  auto vector = fuzzer.fuzzRow(ROW({type}));

  std::vector<int> bucketToPartition(bucketCount);
  std::vector<ChannelIndex> keyChannels;
  std::vector<uint32_t> partitions(vectorSize);

  std::iota(bucketToPartition.begin(), bucketToPartition.end(), 0);
  keyChannels.emplace_back(0);
  connector::hive::HivePartitionFunction partitionFunction(
      bucketCount, bucketToPartition, keyChannels);

  suspender.dismiss();

  partitionFunction.partition(
      *reinterpret_cast<RowVector*>(vector.get()), partitions);
}

} // namespace

BENCHMARK_NAMED_PARAM(
    BM_HivePartFunc,
    BIGINT_10M_ROWS_1K_BUCKETS,
    10'000'000,
    1'000,
    BIGINT());

BENCHMARK_NAMED_PARAM(
    BM_HivePartFunc,
    BOOLEAN_10M_ROWS_1K_BUCKETS,
    10'000'000,
    1'000,
    BOOLEAN());

BENCHMARK_NAMED_PARAM(
    BM_HivePartFunc,
    TINYINT_10M_ROWS_1K_BUCKETS,
    10'000'000,
    1'000,
    TINYINT());

BENCHMARK_NAMED_PARAM(
    BM_HivePartFunc,
    SMALLINT_10M_ROWS_1K_BUCKETS,
    10'000'000,
    1'000,
    SMALLINT());

BENCHMARK_NAMED_PARAM(
    BM_HivePartFunc,
    INTEGER_10M_ROWS_1K_BUCKETS,
    10'000'000,
    1'000,
    INTEGER());

BENCHMARK_NAMED_PARAM(
    BM_HivePartFunc,
    REAL_10M_ROWS_1K_BUCKETS,
    10'000'000,
    1'000,
    REAL());

BENCHMARK_NAMED_PARAM(
    BM_HivePartFunc,
    DOUBLE_10M_ROWS_1K_BUCKETS,
    10'000'000,
    1'000,
    DOUBLE());

BENCHMARK_NAMED_PARAM(
    BM_HivePartFunc,
    TIMESTAMP_10M_ROWS_1K_BUCKETS,
    10'000'000,
    1'000,
    TIMESTAMP());

BENCHMARK_NAMED_PARAM(
    BM_HivePartFunc,
    DATE_10M_ROWS_1K_BUCKETS,
    10'000'000,
    1'000,
    DATE());

} // namespace facebook::velox::test

int main(int argc, char** argv) {
  folly::init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
