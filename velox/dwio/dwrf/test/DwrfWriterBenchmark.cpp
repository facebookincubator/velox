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

#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/dwio/common/tests/utils/MapBuilder.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/dwrf/test/OrcTest.h"
#include "velox/dwio/dwrf/test/utils/E2EWriterTestUtil.h"
#include "velox/dwio/dwrf/writer/Writer.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include "velox/tpch/gen/TpchGen.h"

DEFINE_string(table_name, "lineitem", "table name");
DEFINE_bool(
    VELOX_ENABLE_QAT_ZSTD_OT,
    true,
    "if to use qat for zstd compression");
const double scale_factor = 10;
using std::chrono::system_clock;

using namespace ::testing;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::test;
using namespace facebook::velox::dwrf;
using namespace facebook::velox;
using namespace facebook::velox::tpch;
using facebook::velox::memory::MemoryPool;

class DwrfWriterBenchmark {
 public:
  DwrfWriterBenchmark() {
    rootPool_ = memory::defaultMemoryManager().addRootPool("DwrfWriterTest");
    leafPool_ = rootPool_->addLeafChild("leaf");

    // write file to memory
    auto config = std::make_shared<Config>();
    // seems like default is ZSTD compression
    config->set(Config::COMPRESSION, common::CompressionKind_ZSTD);

    auto sink = std::make_unique<MemorySink>(*leafPool_, 800 * 1024 * 1024);
    auto sinkPtr = sink.get();

    dwrf::WriterOptions options;
    options.config = config;
    options.schema = getTableSchema(Table::TBL_LINEITEM);
    options.memoryPool = rootPool_.get();
    writer_ = std::make_unique<dwrf::Writer>(std::move(sink), options);
  }

  ~DwrfWriterBenchmark() {
    writer_->close();
  }

  void writeToFile() {
    RowVectorPtr rowVector1;
    LOG(INFO) << "table name: " << FLAGS_table_name;

    folly::BenchmarkSuspender suspender;

    if (FLAGS_table_name.compare("part") == 0) {
      rowVector1 = facebook::velox::tpch::genTpchPart(
          leafPool_.get(), 200000, 0, scale_factor);
    } else if (FLAGS_table_name.compare("partsupp") == 0) {
      rowVector1 = facebook::velox::tpch::genTpchPartSupp(
          leafPool_.get(), 800000, 0, 10);
    } else if (FLAGS_table_name.compare("orders") == 0) {
      rowVector1 = facebook::velox::tpch::genTpchOrders(
          leafPool_.get(), 150000, 0, scale_factor);
    } else if (FLAGS_table_name.compare("customer") == 0) {
      rowVector1 = facebook::velox::tpch::genTpchCustomer(
          leafPool_.get(), 150000, 0, scale_factor);
    } else if (FLAGS_table_name.compare("lineitem") == 0) {
      rowVector1 = facebook::velox::tpch::genTpchLineItem(
          leafPool_.get(), 600000, 0, scale_factor);
    } else if (FLAGS_table_name.compare("region") == 0) {
      rowVector1 = facebook::velox::tpch::genTpchRegion(
          leafPool_.get(), 5, 0, scale_factor);
    } else if (FLAGS_table_name.compare("supplier") == 0) {
      rowVector1 = facebook::velox::tpch::genTpchSupplier(
          leafPool_.get(), 100000, 0, 10);
    } else if (FLAGS_table_name.compare("nation") == 0) {
      rowVector1 = facebook::velox::tpch::genTpchNation(
          leafPool_.get(), 25, 0, scale_factor);
    }

    suspender.dismiss();

    for (int i = 0; i < 10; i++) {
      LOG(INFO) << "i: " << i << ", num row: " << rowVector1->size()
                << std::endl;
      writer_->write(rowVector1);
    }

    suspender.rehire();

    LOG(INFO) << "success write " << FLAGS_table_name << std::endl;
    writer_->flush();
  }

 private:
  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> leafPool_;
  dwio::common::DataSink* sinkPtr_;
  std::unique_ptr<dwrf::Writer> writer_;
  RuntimeStatistics runtimeStats_;
};

void run() {
  DwrfWriterBenchmark benchmark;
  benchmark.writeToFile();
}

BENCHMARK(dwrfZSTDWrite) {
  run();
}

int main(int argc, char** argv) {
  folly::init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
