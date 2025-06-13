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

#include "velox/benchmarks/tpch/TpchBenchmark.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::dwio::common;

DEFINE_string(
    data_path,
    "",
    "Root path of TPC-H data. Data layout must follow Hive-style partitioning. "
    "Example layout for '-data_path=/data/tpch10'\n"
    "       /data/tpch10/customer\n"
    "       /data/tpch10/lineitem\n"
    "       /data/tpch10/nation\n"
    "       /data/tpch10/orders\n"
    "       /data/tpch10/part\n"
    "       /data/tpch10/partsupp\n"
    "       /data/tpch10/region\n"
    "       /data/tpch10/supplier\n"
    "If the above are directories, they contain the data files for "
    "each table. If they are files, they contain a file system path for each "
    "data file, one per line. This allows running against cloud storage or "
    "HDFS");
namespace {
static bool notEmpty(const char* /*flagName*/, const std::string& value) {
  return !value.empty();
}
} // namespace

DEFINE_validator(data_path, &notEmpty);

DEFINE_int32(
    run_query_verbose,
    -1,
    "Run a given query and print execution statistics");
DEFINE_int32(
    io_meter_column_pct,
    0,
    "Percentage of lineitem columns to "
    "include in IO meter query. The columns are sorted by name and the n% first "
    "are scanned");

void TpchBenchmark::initQueryBuilder() {
  queryBuilder =
      std::make_shared<TpchQueryBuilder>(toFileFormat(FLAGS_data_format));
  queryBuilder->initialize(FLAGS_data_path);
}

std::unique_ptr<TpchBenchmark> benchmark;

BENCHMARK(q1) {
  benchmark->runQuery(1);
}

BENCHMARK(q2) {
  benchmark->runQuery(2);
}

BENCHMARK(q3) {
  benchmark->runQuery(3);
}

BENCHMARK(q4) {
  benchmark->runQuery(4);
}

BENCHMARK(q5) {
  benchmark->runQuery(5);
}

BENCHMARK(q6) {
  benchmark->runQuery(6);
}

BENCHMARK(q7) {
  benchmark->runQuery(7);
}

BENCHMARK(q8) {
  benchmark->runQuery(8);
}

BENCHMARK(q9) {
  benchmark->runQuery(9);
}

BENCHMARK(q10) {
  benchmark->runQuery(10);
}

BENCHMARK(q11) {
  benchmark->runQuery(11);
}

BENCHMARK(q12) {
  benchmark->runQuery(12);
}

BENCHMARK(q13) {
  benchmark->runQuery(13);
}

BENCHMARK(q14) {
  benchmark->runQuery(14);
}

BENCHMARK(q15) {
  benchmark->runQuery(15);
}

BENCHMARK(q16) {
  benchmark->runQuery(16);
}

BENCHMARK(q17) {
  benchmark->runQuery(17);
}

BENCHMARK(q18) {
  benchmark->runQuery(18);
}

BENCHMARK(q19) {
  benchmark->runQuery(19);
}

BENCHMARK(q20) {
  benchmark->runQuery(20);
}

BENCHMARK(q21) {
  benchmark->runQuery(21);
}

BENCHMARK(q22) {
  benchmark->runQuery(22);
}

void tpchBenchmarkMain() {
  benchmark->initialize();
  if (FLAGS_test_flags_file.empty()) {
    RunStats ignore;
    benchmark->runMain(std::cout, ignore);
  } else {
    benchmark->runAllCombinations();
  }
  benchmark->shutdown();
}
