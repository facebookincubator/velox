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
#include "velox/dwio/dwrf/test/utils/DataFiles.h"
#include "velox/dwio/parquet/reader/NativeParquetReader.h"
#include "velox/type/Type.h"
#include "velox/type/tests/FilterBuilder.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/tests/VectorMaker.h"

// using namespace ::testing;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox;
using namespace facebook::velox::parquet;

namespace {

class ParquetReaderBenchmark {
 public:
  ParquetReaderBenchmark()
      : min_(1),
        max_(2000),
        scanSpec_(""),
        rowReaderOpts_(),
        readerOptions_(ReaderOptions()),
        reader_(
            std::make_unique<FileInputStream>(getExampleFilePath(
                "10000_random_integers_plain_nocompression.parquet")),
            readerOptions_) {
    auto rowType = ROW({"a"}, {INTEGER()});
    auto cs = std::make_shared<ColumnSelector>(
        rowType, std::vector<std::string>{"a"});
    rowReaderOpts_.select(cs);
    rowReaderOpts_.setScanSpec(&scanSpec_);
  }

  void readParquet(const double& selectionRate) {
    folly::BenchmarkSuspender suspender;

    VELOX_CHECK_LT(
        selectionRate, 1.0, "Unexpected filterRate {}", selectionRate);

    scanSpec_.getOrCreateChild(common::Subfield("a"))
        ->setFilter(common::test::between(
            min_, min_ + (max_ - min_ + 1) * selectionRate));

    //    rowReaderOpts_.setUseAhanaParquetReader(useAhanaReader);
    auto rowReader = reader_.createRowReader(rowReaderOpts_);

    uint64_t total = 0;
    uint64_t count = 0;
    VectorPtr result = BaseVector::create(
        reader_.rowType(), 0, &readerOptions_.getMemoryPool());

    suspender.dismiss();
    for (int i = 0; i < 1; i++) {
      do {
        count = rowReader->next(10000, result);
        total += count;
      } while (count != 0);

      folly::doNotOptimizeAway(result);
      //      printf("total %lld\n", total);
    }

    folly::doNotOptimizeAway(total);

    suspender.rehire();

    //    printf("total = %llu", total);
  }

 private:
  std::string getExampleFilePath(const std::string& fileName) {
    return test::getDataFilePath(
        "velox/dwio/parquet/tests", "examples/" + fileName);
  }

  const uint32_t min_;
  const uint32_t max_;

  common::ScanSpec scanSpec_;
  ReaderOptions readerOptions_;
  RowReaderOptions rowReaderOpts_;
  NativeParquetReader reader_;
  VectorPtr result; // = BaseVector::create(outputType, 0, &memoryPool);
};

// BENCHMARK(readIntegerInPlainNoCompression_005) {
//   ParquetReaderBenchmark benchmark;
//   benchmark.readParquet(0.05, true);
// }
// BENCHMARK(readIntegerInPlainNoCompression_01) {
//   ParquetReaderBenchmark benchmark;
//   benchmark.readParquet(0.1, true);
// }
// BENCHMARK(readIntegerInPlainNoCompression_02) {
//   ParquetReaderBenchmark benchmark;
//   benchmark.readParquet(0.2, true);
// }
// BENCHMARK(readIntegerInPlainNoCompression_03) {
//   ParquetReaderBenchmark benchmark;
//   benchmark.readParquet(0.3, true);
// }
// BENCHMARK(readIntegerInPlainNoCompression_04) {
//   ParquetReaderBenchmark benchmark;
//   benchmark.readParquet(0.4, true);
// }
BENCHMARK(readIntegerInPlainNoCompression_05) {
  ParquetReaderBenchmark benchmark;
  benchmark.readParquet(0.5);
  //}
  // BENCHMARK(readIntegerInPlainNoCompression_06) {
  //  ParquetReaderBenchmark benchmark;
  //  benchmark.readParquet(0.6, true);
  //}
  // BENCHMARK(readIntegerInPlainNoCompression_07) {
  //  ParquetReaderBenchmark benchmark;
  //  benchmark.readParquet(0.7, true);
  //}
  // BENCHMARK(readIntegerInPlainNoCompression_08) {
  //  ParquetReaderBenchmark benchmark;
  //  benchmark.readParquet(0.8, true);
  //}
  // BENCHMARK(readIntegerInPlainNoCompression_09) {
  //  ParquetReaderBenchmark benchmark;
  //  benchmark.readParquet(0.9, true);
  //}
  // BENCHMARK(readIntegerInPlainNoCompression_095) {
  //  ParquetReaderBenchmark benchmark;
  //  benchmark.readParquet(0.95, true);
}

// BENCHMARK(readIntegerInPlainNoCompressionAhana) {
//   ParquetReaderBenchmark benchmark;
//   benchmark.readParquet(0.5, true);
// }

BENCHMARK_DRAW_LINE();

} // namespace

//} // namespace facebook::velox::test

int main(int argc, char** argv) {
  folly::init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
