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

#include <folly/logging/xlog.h>
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
#include <sys/syscall.h>
#include <unistd.h> // for syscall()
#include "velox/tpch/gen/TpchGen.h"

DEFINE_int32(SCALE_FACTOR, 10, "scale factor");
DEFINE_int32(MAX_ORDER_ROWS, 6000000, "max order rows");
DEFINE_int32(BASE_SCALE_FACTOR, 1, "scale factor used in tpchgen");

using std::chrono::system_clock;

using namespace ::testing;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::test;
using namespace facebook::velox::dwrf;
using namespace facebook::velox;
using namespace facebook::velox::tpch;
using facebook::velox::common::CompressionKind;
using facebook::velox::memory::MemoryPool;

class DwrfWriterBenchmark {
 public:
  DwrfWriterBenchmark() {
    rootPool_ = memory::defaultMemoryManager().addRootPool("DwrfWriterTest");
    leafPool_ = rootPool_->addLeafChild("leaf");

    auto config = std::make_shared<dwrf::Config>();
    config->set(
        dwrf::Config::COMPRESSION, CompressionKind::CompressionKind_ZSTD);

    int id = syscall(SYS_gettid);
    auto path = "/tmp/zstd_compressed_" + std::to_string(id) + ".orc";
    auto localWriteFile = std::make_unique<LocalWriteFile>(path, true, false);
    auto sink =
        std::make_unique<WriteFileSink>(std::move(localWriteFile), path);

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
    int total_writes = FLAGS_SCALE_FACTOR / FLAGS_BASE_SCALE_FACTOR;
    XLOG_FIRST_N(INFO, 1) << fmt::format("scale factor: ")
                          << FLAGS_SCALE_FACTOR;
    XLOG_FIRST_N(INFO, 1) << fmt::format("base scale factor: ")
                          << FLAGS_BASE_SCALE_FACTOR;
    XLOG_FIRST_N(INFO, 1) << fmt::format("FLAGS_MAX_ORDER_ROWS: ") << FLAGS_MAX_ORDER_ROWS;
    XLOG_FIRST_N(INFO, 1) << fmt::format("total writes: ") << total_writes;

    folly::BenchmarkSuspender suspender;
    rowVector1 = facebook::velox::tpch::genTpchLineItem(
        leafPool_.get(), FLAGS_MAX_ORDER_ROWS, 0, FLAGS_BASE_SCALE_FACTOR);

    suspender.dismiss();

    for (int i = 0; i < total_writes; i++) {
      XLOG(INFO) << "i: " << i << ", num row: " << rowVector1->size()
                 << std::endl;
      writer_->write(rowVector1);
    }

    suspender.rehire();

    XLOG(INFO) << "success write." << std::endl;
    writer_->flush();
  }

 private:
  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> leafPool_;
  std::unique_ptr<dwrf::Writer> writer_;
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
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  folly::runBenchmarks();
  return 0;
}