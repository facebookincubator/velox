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

#include "velox/connectors/hive/iceberg/tests/IcebergSplitReaderBenchmark.h"

using namespace facebook::velox;
using namespace facebook::velox::dwio;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::dwrf;
using namespace facebook::velox::iceberg::reader::test;
using namespace facebook::velox::test;

#define PARQUET_BENCHMARKS_FILTER_DELETES(_type_, _name_, _filter_, _deletes_) \
  BENCHMARK_NAMED_PARAM(                                                       \
      run,                                                                     \
      _name_##_Filter_##_filter_##_Delete_##_deletes_##_next_5k,               \
      #_name_,                                                                 \
      _type_,                                                                  \
      _filter_,                                                                \
      _deletes_,                                                               \
      5000);                                                                   \
  BENCHMARK_NAMED_PARAM(                                                       \
      run,                                                                     \
      _name_##_Filter_##_filter_##_Delete_##_deletes_##_next_10k,              \
      #_name_,                                                                 \
      _type_,                                                                  \
      _filter_,                                                                \
      _deletes_,                                                               \
      10000);                                                                  \
  BENCHMARK_DRAW_LINE();

#define PARQUET_BENCHMARKS_FILTERS(_type_, _name_, _filter_)      \
  PARQUET_BENCHMARKS_FILTER_DELETES(_type_, _name_, _filter_, 0)  \
  PARQUET_BENCHMARKS_FILTER_DELETES(_type_, _name_, _filter_, 20) \
  PARQUET_BENCHMARKS_FILTER_DELETES(_type_, _name_, _filter_, 50) \
  PARQUET_BENCHMARKS_FILTER_DELETES(_type_, _name_, _filter_, 70) \
  PARQUET_BENCHMARKS_FILTER_DELETES(_type_, _name_, _filter_, 100)

#define PARQUET_BENCHMARKS(_type_, _name_)        \
  PARQUET_BENCHMARKS_FILTERS(_type_, _name_, 0)   \
  PARQUET_BENCHMARKS_FILTERS(_type_, _name_, 20)  \
  PARQUET_BENCHMARKS_FILTERS(_type_, _name_, 50)  \
  PARQUET_BENCHMARKS_FILTERS(_type_, _name_, 70)  \
  PARQUET_BENCHMARKS_FILTERS(_type_, _name_, 100) \
  BENCHMARK_DRAW_LINE();

PARQUET_BENCHMARKS(BIGINT(), BigInt);

// Same deleted rows, stored three ways: one V2 positional delete file, ten V2
// positional delete files (deletes accumulated over several commits), and one
// V3 deletion vector. Every row passes the filter, so the read cost is the
// data scan plus applying the deletes.
#define DELETE_FILE_KIND_BENCHMARKS(_deletes_)  \
  BENCHMARK_NAMED_PARAM(                        \
      runDeletes,                               \
      Delete_##_deletes_##_Positional_1_file,   \
      "BigInt",                                 \
      BIGINT(),                                 \
      _deletes_,                                \
      DeleteFileKind::kPositional,              \
      1,                                        \
      10000);                                   \
  BENCHMARK_RELATIVE_NAMED_PARAM(               \
      runDeletes,                               \
      Delete_##_deletes_##_Positional_10_files, \
      "BigInt",                                 \
      BIGINT(),                                 \
      _deletes_,                                \
      DeleteFileKind::kPositional,              \
      10,                                       \
      10000);                                   \
  BENCHMARK_RELATIVE_NAMED_PARAM(               \
      runDeletes,                               \
      Delete_##_deletes_##_DeletionVector,      \
      "BigInt",                                 \
      BIGINT(),                                 \
      _deletes_,                                \
      DeleteFileKind::kDeletionVector,          \
      1,                                        \
      10000);                                   \
  BENCHMARK_DRAW_LINE();

DELETE_FILE_KIND_BENCHMARKS(1)
DELETE_FILE_KIND_BENCHMARKS(5)
DELETE_FILE_KIND_BENCHMARKS(20)
DELETE_FILE_KIND_BENCHMARKS(50)

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  folly::runBenchmarks();
  return 0;
}
