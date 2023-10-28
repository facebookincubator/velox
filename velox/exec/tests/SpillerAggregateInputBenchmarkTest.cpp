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

#include "velox/exec/tests/SpillerAggregateInputBenchmarkTest.h"
#include "velox/serializers/PrestoSerializer.h"

#include <gflags/gflags.h>
#include <deque>

DEFINE_string(
    spiller_benchmark_name,
    "SpillerAggregateInputBenchmarkTest",
    "The name of this benchmark");
DEFINE_uint64(
    spiller_benchmark_max_spill_file_size,
    2 << 30,
    "The max spill file size");
DEFINE_uint64(
    spiller_benchmark_write_buffer_size,
    0,
    "The spill write buffer size");
DEFINE_uint64(
    spiller_benchmark_min_spill_run_size,
    1 << 30,
    "The file directory path for spiller benchmark");
DEFINE_uint32(
    spiller_benchmark_num_spill_vectors,
    1'000,
    "The number of vectors for spilling");
DEFINE_uint32(
    spiller_benchmark_spill_vector_size,
    500,
    "The number of rows per each spill vector");
DEFINE_string(
    spiller_benchmark_compression_kind,
    "none",
    "The compression kind to compress spill rows before write to disk");
DEFINE_uint32(
    spiller_benchmark_spill_dependent_key_num,
    2,
    "The number of aggregation dependent key");

using namespace facebook::velox;
using namespace facebook::velox::common;
using namespace facebook::velox::memory;
using namespace facebook::velox::exec;

namespace facebook::velox::exec::test {

void SpillerAggregateInputBenchmarkTest::setUp() {
  SpillerBenchmarkBase::setUp();

  rowContainer_ = setupSpillContainer(
      rowType_, FLAGS_spiller_benchmark_spill_dependent_key_num);
  spiller_ = std::make_unique<Spiller>(
      exec::Spiller::Type::kAggregateInput,
      rowContainer_.get(),
      [&](folly::Range<char**> rows) { rowContainer_->eraseRows(rows); },
      rowType_,
      HashBitRange{29, 29},
      rowContainer_->keyTypes().size(),
      std::vector<CompareFlags>{},
      fmt::format("{}/{}", spillDir_, FLAGS_spiller_benchmark_name),
      FLAGS_spiller_benchmark_max_spill_file_size,
      FLAGS_spiller_benchmark_write_buffer_size,
      FLAGS_spiller_benchmark_min_spill_run_size,
      stringToCompressionKind(FLAGS_spiller_benchmark_compression_kind),
      memory::spillMemoryPool(),
      executor_.get());
  spiller_->setPartitionsSpilled({0});
  writeSpillData();
}

void SpillerAggregateInputBenchmarkTest::run() {
  MicrosecondTimer timer(&executionTimeUs_);
  spiller_->spill(0, 0);
}

void SpillerAggregateInputBenchmarkTest::writeSpillData() {
  vector_size_t numRows = 0;
  for (const auto& rowVector : rowVectors_) {
    numRows += rowVector->size();
  }

  std::vector<char*> rows;
  rows.reserve(numRows);
  for (int i = 0; i < numRows; ++i) {
    rows[i] = rowContainer_->newRow();
  }

  vector_size_t nextRow = 0;
  for (const auto& rowVector : rowVectors_) {
    const SelectivityVector allRows(rowVector->size());
    for (int index = 0; index < rowVector->size(); ++index, ++nextRow) {
      for (int i = 0; i < rowType_->size(); ++i) {
        DecodedVector decodedVector(*rowVector->childAt(i), allRows);
        rowContainer_->store(decodedVector, index, rows[nextRow], i);
      }
    }
  }
}

std::unique_ptr<RowContainer>
SpillerAggregateInputBenchmarkTest::makeRowContainer(
    const std::vector<TypePtr>& keyTypes,
    const std::vector<TypePtr>& dependentTypes) const {
  auto container = std::make_unique<RowContainer>(
      keyTypes,
      true, // nullableKeys
      std::vector<Accumulator>{},
      dependentTypes,
      false, // hasNext
      false, // isJoinBuild
      false, // hasProbedFlag
      false, // hasNormalizedKey
      pool_.get());
  return container;
}

std::unique_ptr<RowContainer>
SpillerAggregateInputBenchmarkTest::setupSpillContainer(
    const RowTypePtr& rowType,
    uint32_t numKeys) const {
  const auto& childTypes = rowType->children();
  std::vector<TypePtr> keys(childTypes.begin(), childTypes.begin() + numKeys);
  std::vector<TypePtr> dependents;
  if (numKeys < childTypes.size()) {
    dependents.insert(
        dependents.end(), childTypes.begin() + numKeys, childTypes.end());
  }
  return makeRowContainer(keys, dependents);
}
} // namespace facebook::velox::exec::test

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  serializer::presto::PrestoVectorSerde::registerVectorSerde();
  filesystems::registerLocalFileSystem();
  auto test = std::make_unique<
      facebook::velox::exec::test::SpillerAggregateInputBenchmarkTest>();
  test->setUp();
  test->run();
  test->printStats();
  test->cleanup();
  return 0;
}
