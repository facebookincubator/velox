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
#include "SortSpillInputBenchmarkBase.h"

#include <gflags/gflags.h>

using namespace facebook::velox;
using namespace facebook::velox::exec;

void runTest(RowTypePtr rowType, bool serializeRowContainer, int strLength) {
  folly::BenchmarkSuspender suspender;
  auto test = std::make_unique<test::SortSpillInputBenchmarkBase>();
  test->setUp(rowType, serializeRowContainer, strLength);
  suspender.dismiss();
  test->run();
  suspender.rehire();
  test->printStats();
  test->cleanup();
}

BENCHMARK(integer_type) {
  runTest(ROW({"c0", "c1"}, {INTEGER(), INTEGER()}), false, 0);
}

BENCHMARK_RELATIVE(integer_type_serialize_rows) {
  runTest(ROW({"c0", "c1"}, {INTEGER(), INTEGER()}), true, 0);
}

BENCHMARK(bigint_type) {
  runTest(ROW({"c0", "c1"}, {INTEGER(), BIGINT()}), false, 0);
}

BENCHMARK_RELATIVE(bigint_type_serialize_rows) {
  runTest(ROW({"c0", "c1"}, {INTEGER(), BIGINT()}), true, 0);
}

BENCHMARK(string_type_10bytes) {
  runTest(ROW({"c0", "c1"}, {INTEGER(), VARBINARY()}), false, 10);
}

BENCHMARK_RELATIVE(string_type_10bytes_serialize_rows) {
  runTest(ROW({"c0", "c1"}, {INTEGER(), VARBINARY()}), true, 10);
}

BENCHMARK(string_type_50bytes) {
  runTest(ROW({"c0", "c1"}, {INTEGER(), VARBINARY()}), false, 50);
}

BENCHMARK_RELATIVE(string_type_50bytes_long_serialize_rows) {
  runTest(ROW({"c0", "c1"}, {INTEGER(), VARBINARY()}), true, 50);
}

BENCHMARK(array_of_integer_type) {
  runTest(ROW({"c0", "c1"}, {INTEGER(), ARRAY(INTEGER())}), false, 10);
}

BENCHMARK_RELATIVE(array_of_integer_type_serialize_rows) {
  runTest(ROW({"c0", "c1"}, {INTEGER(), ARRAY(INTEGER())}), true, 10);
}

BENCHMARK(array_of_10bytes_varbinary_type) {
  runTest(ROW({"c0", "c1"}, {INTEGER(), ARRAY(VARBINARY())}), false, 10);
}

BENCHMARK_RELATIVE(array_of_10bytes_varbinary_type_serialize_rows) {
  runTest(ROW({"c0", "c1"}, {INTEGER(), ARRAY(VARBINARY())}), true, 10);
}

BENCHMARK(array_of_50bytes_varbinary_type) {
  runTest(ROW({"c0", "c1"}, {INTEGER(), ARRAY(VARBINARY())}), false, 50);
}

BENCHMARK_RELATIVE(array_of_50bytes_varbinary_type_serialize_rows) {
  runTest(ROW({"c0", "c1"}, {INTEGER(), ARRAY(VARBINARY())}), true, 50);
}

BENCHMARK(array_of_100bytes_varbinary_type) {
  runTest(ROW({"c0", "c1"}, {INTEGER(), ARRAY(VARBINARY())}), false, 100);
}

BENCHMARK_RELATIVE(array_of_100bytes_varbinary_type_serialize_rows) {
  runTest(ROW({"c0", "c1"}, {INTEGER(), ARRAY(VARBINARY())}), true, 100);
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  serializer::presto::PrestoVectorSerde::registerVectorSerde();
  serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
  filesystems::registerLocalFileSystem();

  folly::runBenchmarks();
  return 0;
}
