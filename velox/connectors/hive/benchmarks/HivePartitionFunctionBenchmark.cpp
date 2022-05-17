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
#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include "velox/connectors/hive/HivePartitionFunction.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/VectorMaker.h"

DEFINE_int64(fuzzer_seed, 99887766, "Seed for random input dataset generator");

using namespace facebook::velox;
using namespace facebook::velox::test;
using connector::hive::HivePartitionFunction;

namespace {

constexpr std::array<TypeKind, 10> supportedTypes{
    TypeKind::BOOLEAN,
    TypeKind::TINYINT,
    TypeKind::SMALLINT,
    TypeKind::INTEGER,
    TypeKind::BIGINT,
    TypeKind::REAL,
    TypeKind::DOUBLE,
    TypeKind::VARCHAR,
    TypeKind::TIMESTAMP,
    TypeKind::DATE};

class HivePartitionFunctionBenchmark
    : public functions::test::FunctionBenchmarkBase {
 public:
  explicit HivePartitionFunctionBenchmark(
      size_t vectorSize,
      size_t smallBucketCount,
      size_t largeBucketCount,
      size_t stringLength = 20)
      : FunctionBenchmarkBase() {
    // Prepare input data
    VectorFuzzer::Options opts;
    opts.vectorSize = vectorSize;
    opts.stringLength = stringLength;
    VectorFuzzer fuzzer(opts, pool(), FLAGS_fuzzer_seed);
    VectorMaker vm{pool_.get()};
    for (auto typeKind : supportedTypes) {
      auto flatVector = fuzzer.fuzzFlat(createScalarType(typeKind));
      auto rowVecotr = vm.rowVector({flatVector});
      rowVectors_[typeKind] = std::move(rowVecotr);
    }

    // Prepare HivePartitionFunction
    smallBucketFunction_ = createHivePartitionFunction(smallBucketCount);
    largeBucketFunction_ = createHivePartitionFunction(largeBucketCount);

    partitions_.resize(vectorSize);
  }

  template <TypeKind KIND>
  void runSmall() {
    run<KIND>(smallBucketFunction_.get());
  }

  template <TypeKind KIND>
  void runLarge() {
    run<KIND>(largeBucketFunction_.get());
  }

 private:
  std::unique_ptr<HivePartitionFunction> createHivePartitionFunction(
      size_t bucketCount) {
    std::vector<int> bucketToPartition(bucketCount);
    std::iota(bucketToPartition.begin(), bucketToPartition.end(), 0);
    std::vector<ChannelIndex> keyChannels;
    keyChannels.emplace_back(0);
    return std::make_unique<HivePartitionFunction>(
        bucketCount, bucketToPartition, keyChannels);
  }

  template <TypeKind KIND>
  void run(HivePartitionFunction* function) {
    function->partition(*rowVectors_[KIND], partitions_);
  }

  std::unordered_map<TypeKind, RowVectorPtr> rowVectors_;
  std::unique_ptr<HivePartitionFunction> smallBucketFunction_;
  std::unique_ptr<HivePartitionFunction> largeBucketFunction_;
  std::vector<uint32_t> partitions_;
};

constexpr size_t smallVectorSize = 1'000;
constexpr size_t largeVectorSize = 10'000;
constexpr size_t smallBucketCount = 20;
constexpr size_t largeBucketCount = 100;

std::unique_ptr<HivePartitionFunctionBenchmark> benchmarkSmall;
std::unique_ptr<HivePartitionFunctionBenchmark> benchmarkLarge;

BENCHMARK(booleanSmallRowsSmallBuckets) {
  benchmarkSmall->runSmall<TypeKind::BOOLEAN>();
}

BENCHMARK_RELATIVE(booleanSmallRowsLargeBuckets) {
  benchmarkSmall->runLarge<TypeKind::BOOLEAN>();
}

BENCHMARK(booleanLargeRowsSmallBuckets) {
  benchmarkLarge->runSmall<TypeKind::BOOLEAN>();
}

BENCHMARK_RELATIVE(booleanLargeRowsLargeBuckets) {
  benchmarkLarge->runLarge<TypeKind::BOOLEAN>();
}

BENCHMARK_DRAW_LINE();

BENCHMARK(tinyintSmallRowsSmallBuckets) {
  benchmarkSmall->runSmall<TypeKind::TINYINT>();
}

BENCHMARK_RELATIVE(tinyintSmallRowsLargeBuckets) {
  benchmarkSmall->runLarge<TypeKind::TINYINT>();
}

BENCHMARK(tinyintLargeRowsSmallBuckets) {
  benchmarkLarge->runSmall<TypeKind::TINYINT>();
}

BENCHMARK_RELATIVE(tinyintLargeRowsLargeBuckets) {
  benchmarkLarge->runLarge<TypeKind::TINYINT>();
}

BENCHMARK_DRAW_LINE();

BENCHMARK(smallintSmallRowsSmallBuckets) {
  benchmarkSmall->runSmall<TypeKind::SMALLINT>();
}

BENCHMARK_RELATIVE(smallintSmallRowsLargeBuckets) {
  benchmarkSmall->runLarge<TypeKind::SMALLINT>();
}

BENCHMARK(smallintLargeRowsSmallBuckets) {
  benchmarkLarge->runSmall<TypeKind::SMALLINT>();
}

BENCHMARK_RELATIVE(smallintLargeRowsLargeBuckets) {
  benchmarkLarge->runLarge<TypeKind::SMALLINT>();
}

BENCHMARK_DRAW_LINE();

BENCHMARK(integerSmallRowsSmallBuckets) {
  benchmarkSmall->runSmall<TypeKind::INTEGER>();
}

BENCHMARK_RELATIVE(integerSmallRowsLargeBuckets) {
  benchmarkSmall->runLarge<TypeKind::INTEGER>();
}

BENCHMARK(integerLargeRowsSmallBuckets) {
  benchmarkLarge->runSmall<TypeKind::INTEGER>();
}

BENCHMARK_RELATIVE(integerLargeRowsLargeBuckets) {
  benchmarkLarge->runLarge<TypeKind::INTEGER>();
}

BENCHMARK_DRAW_LINE();

BENCHMARK(bigintSmallRowsSmallBuckets) {
  benchmarkSmall->runSmall<TypeKind::BIGINT>();
}

BENCHMARK_RELATIVE(bigintSmallRowsLargeBuckets) {
  benchmarkSmall->runLarge<TypeKind::BIGINT>();
}

BENCHMARK(bigintLargeRowsSmallBuckets) {
  benchmarkLarge->runSmall<TypeKind::BIGINT>();
}

BENCHMARK_RELATIVE(bigintLargeRowsLargeBuckets) {
  benchmarkLarge->runLarge<TypeKind::BIGINT>();
}

BENCHMARK_DRAW_LINE();

BENCHMARK(realSmallRowsSmallBuckets) {
  benchmarkSmall->runSmall<TypeKind::REAL>();
}

BENCHMARK_RELATIVE(realSmallRowsLargeBuckets) {
  benchmarkSmall->runLarge<TypeKind::REAL>();
}

BENCHMARK(realLargeRowsSmallBuckets) {
  benchmarkLarge->runSmall<TypeKind::REAL>();
}

BENCHMARK_RELATIVE(realLargeRowsLargeBuckets) {
  benchmarkLarge->runLarge<TypeKind::REAL>();
}

BENCHMARK_DRAW_LINE();

BENCHMARK(doubleSmallRowsSmallBuckets) {
  benchmarkSmall->runSmall<TypeKind::DOUBLE>();
}

BENCHMARK_RELATIVE(doubleSmallRowsLargeBuckets) {
  benchmarkSmall->runLarge<TypeKind::DOUBLE>();
}

BENCHMARK(doubleLargeRowsSmallBuckets) {
  benchmarkLarge->runSmall<TypeKind::DOUBLE>();
}

BENCHMARK_RELATIVE(doubleLargeRowsLargeBuckets) {
  benchmarkLarge->runLarge<TypeKind::DOUBLE>();
}

BENCHMARK_DRAW_LINE();

BENCHMARK(varcharSmallRowsSmallBuckets) {
  benchmarkSmall->runSmall<TypeKind::VARCHAR>();
}

BENCHMARK_RELATIVE(varcharSmallRowsLargeBuckets) {
  benchmarkSmall->runLarge<TypeKind::VARCHAR>();
}

BENCHMARK(varcharLargeRowsSmallBuckets) {
  benchmarkLarge->runSmall<TypeKind::VARCHAR>();
}

BENCHMARK_RELATIVE(varcharLargeRowsLargeBuckets) {
  benchmarkLarge->runLarge<TypeKind::VARCHAR>();
}

BENCHMARK_DRAW_LINE();

BENCHMARK(timestampSmallRowsSmallBuckets) {
  benchmarkSmall->runSmall<TypeKind::TIMESTAMP>();
}

BENCHMARK_RELATIVE(timestampSmallRowsLargeBuckets) {
  benchmarkSmall->runLarge<TypeKind::TIMESTAMP>();
}

BENCHMARK(timestampLargeRowsSmallBuckets) {
  benchmarkLarge->runSmall<TypeKind::TIMESTAMP>();
}

BENCHMARK_RELATIVE(timestampLargeRowsLargeBuckets) {
  benchmarkLarge->runLarge<TypeKind::TIMESTAMP>();
}

BENCHMARK_DRAW_LINE();

BENCHMARK(dateSmallRowsSmallBuckets) {
  benchmarkSmall->runSmall<TypeKind::DATE>();
}

BENCHMARK_RELATIVE(dateSmallRowsLargeBuckets) {
  benchmarkSmall->runLarge<TypeKind::DATE>();
}

BENCHMARK(dateLargeRowsSmallBuckets) {
  benchmarkLarge->runSmall<TypeKind::DATE>();
}

BENCHMARK_RELATIVE(dateLargeRowsLargeBuckets) {
  benchmarkLarge->runLarge<TypeKind::DATE>();
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  benchmarkSmall = std::make_unique<HivePartitionFunctionBenchmark>(
      smallVectorSize, smallBucketCount, largeBucketCount);
  benchmarkLarge = std::make_unique<HivePartitionFunctionBenchmark>(
      largeVectorSize, smallBucketCount, largeBucketCount);

  folly::runBenchmarks();

  benchmarkSmall.reset();
  benchmarkLarge.reset();

  return 0;
}
