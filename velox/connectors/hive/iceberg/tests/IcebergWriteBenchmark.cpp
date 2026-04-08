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
#include <gflags/gflags.h>

#include <algorithm>
#include <atomic>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "velox/common/base/Exceptions.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/IcebergConfig.h"
#include "velox/connectors/hive/iceberg/TransformEvaluator.h"
#include "velox/connectors/hive/iceberg/TransformExprBuilder.h"
#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/functions/iceberg/Register.h"
#include "velox/type/Type.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;
using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;
using namespace facebook::velox::connector::hive::iceberg;

DEFINE_int32(num_batches, 2, "Number of input batches per benchmark run.");
DEFINE_int32(rows_per_batch, 5'000, "Rows per input batch.");
DEFINE_double(null_ratio, 0.1, "Null ratio for generated data.");
DEFINE_int32(string_length, 32, "String length for generated data.");
DEFINE_int32(bucket_count, 32, "Bucket count for bucket transform.");
DEFINE_int32(truncate_width, 16, "Truncate width for truncate transform.");
DEFINE_int32(
    max_partitions_per_writers,
    10'000,
    "Max partitions per writer (HiveConfig session override).");
DEFINE_int32(
    root_pool_capacity_gb,
    8,
    "Root memory pool capacity in GB for the benchmark.");

namespace facebook::velox::connector::hive::iceberg {

using test::PartitionField;

class BenchmarkContext : public test::IcebergTestBase {
 public:
  BenchmarkContext() {
    SetUp();
    setConnectorSessionProperty(
        HiveConfig::kMaxPartitionsPerWritersSession,
        std::to_string(FLAGS_max_partitions_per_writers));
  }

  ~BenchmarkContext() override {
    TearDown();
  }

  void TestBody() override {}

  RowVectorPtr makeBatch(const RowTypePtr& rowType) const {
    VectorFuzzer::Options options;
    options.vectorSize = FLAGS_rows_per_batch;
    options.stringLength = FLAGS_string_length;
    options.nullRatio = FLAGS_null_ratio;
    options.allowDictionaryVector = false;
    options.timestampPrecision =
        fuzzer::FuzzerTimestampPrecision::kMilliSeconds;
    VectorFuzzer fuzzer(options, opPool_.get());
    return fuzzer.fuzzRow(rowType, FLAGS_rows_per_batch, false);
  }

  std::vector<RowVectorPtr> makeBatches(const RowTypePtr& rowType) const {
    std::vector<RowVectorPtr> batches;
    batches.reserve(FLAGS_num_batches);
    for (int32_t i = 0; i < FLAGS_num_batches; ++i) {
      batches.push_back(makeBatch(rowType));
    }
    return batches;
  }

  std::shared_ptr<IcebergDataSink> makeDataSink(
      const RowTypePtr& rowType,
      const std::string& outputDirectoryPath,
      const std::vector<PartitionField>& partitionFields) {
    return createDataSink(rowType, outputDirectoryPath, partitionFields);
  }

  IcebergPartitionSpecPtr makePartitionSpec(
      const RowTypePtr& rowType,
      const std::vector<PartitionField>& partitionFields) {
    return createPartitionSpec(rowType, partitionFields);
  }

  const ConnectorQueryCtx* connectorQueryCtx() const {
    return connectorQueryCtx_.get();
  }

 private:
  int64_t rootPoolCapacityBytes() const override {
    return static_cast<int64_t>(FLAGS_root_pool_capacity_gb) << 30;
  }
};

class TransformBenchmark {
 public:
  TransformBenchmark(
      BenchmarkContext& context,
      RowTypePtr rowType,
      const std::vector<PartitionField>& partitionFields)
      : input_(context.makeBatch(rowType)) {
    auto partitionSpec = context.makePartitionSpec(rowType, partitionFields);
    std::vector<column_index_t> partitionChannels;
    partitionChannels.reserve(partitionFields.size());
    for (const auto& field : partitionFields) {
      partitionChannels.push_back(field.id);
    }
    auto transformExprs = TransformExprBuilder::toExpressions(
        partitionSpec,
        partitionChannels,
        rowType,
        IcebergConfig::kDefaultFunctionPrefix);
    transformEvaluator_ = std::make_unique<TransformEvaluator>(
        transformExprs, context.connectorQueryCtx());
  }

  void run() const {
    folly::doNotOptimizeAway(transformEvaluator_->evaluate(input_));
  }

 private:
  RowVectorPtr input_;
  std::unique_ptr<TransformEvaluator> transformEvaluator_;
};

class WriteBenchmark {
 public:
  WriteBenchmark(
      BenchmarkContext& context,
      RowTypePtr rowType,
      std::vector<PartitionField> partitionFields)
      : context_(context),
        rowType_(std::move(rowType)),
        partitionFields_(std::move(partitionFields)),
        batches_(context_.makeBatches(rowType_)) {}

  void run() const {
    auto dataSink = context_.makeDataSink(
        rowType_,
        exec::test::TempDirectoryPath::create()->getPath(),
        partitionFields_);
    for (const auto& batch : batches_) {
      dataSink->appendData(batch);
    }
    dataSink->finish();
    folly::doNotOptimizeAway(dataSink->close());
  }

 private:
  BenchmarkContext& context_;
  RowTypePtr rowType_;
  std::vector<PartitionField> partitionFields_;
  std::vector<RowVectorPtr> batches_;
};

std::unique_ptr<BenchmarkContext> context;

std::unique_ptr<TransformBenchmark> transformIdentity;
std::unique_ptr<TransformBenchmark> transformBucket;
std::unique_ptr<TransformBenchmark> transformTruncate;
std::unique_ptr<TransformBenchmark> transformYear;
std::unique_ptr<TransformBenchmark> transformMonth;
std::unique_ptr<TransformBenchmark> transformDay;
std::unique_ptr<TransformBenchmark> transformHour;

std::unique_ptr<WriteBenchmark> writeUnpartitioned;
std::unique_ptr<WriteBenchmark> writeIdentity;
std::unique_ptr<WriteBenchmark> writeBucket;
std::unique_ptr<WriteBenchmark> writeTruncate;
std::unique_ptr<WriteBenchmark> writeYear;
std::unique_ptr<WriteBenchmark> writeMonth;
std::unique_ptr<WriteBenchmark> writeDay;
std::unique_ptr<WriteBenchmark> writeHour;

BENCHMARK(icebergTransformIdentity) {
  transformIdentity->run();
}

BENCHMARK_RELATIVE(icebergTransformBucket) {
  transformBucket->run();
}

BENCHMARK_RELATIVE(icebergTransformTruncate) {
  transformTruncate->run();
}

BENCHMARK_RELATIVE(icebergTransformYear) {
  transformYear->run();
}

BENCHMARK_RELATIVE(icebergTransformMonth) {
  transformMonth->run();
}

BENCHMARK_RELATIVE(icebergTransformDay) {
  transformDay->run();
}

BENCHMARK_RELATIVE(icebergTransformHour) {
  transformHour->run();
}

BENCHMARK_DRAW_LINE();

BENCHMARK(icebergWriteUnpartitioned) {
  writeUnpartitioned->run();
}

BENCHMARK_RELATIVE(icebergWriteIdentity) {
  writeIdentity->run();
}

BENCHMARK_RELATIVE(icebergWriteBucket) {
  writeBucket->run();
}

BENCHMARK_RELATIVE(icebergWriteTruncate) {
  writeTruncate->run();
}

BENCHMARK_RELATIVE(icebergWriteYear) {
  writeYear->run();
}

BENCHMARK_RELATIVE(icebergWriteMonth) {
  writeMonth->run();
}

BENCHMARK_RELATIVE(icebergWriteDay) {
  writeDay->run();
}

BENCHMARK_RELATIVE(icebergWriteHour) {
  writeHour->run();
}

} // namespace facebook::velox::connector::hive::iceberg

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  memory::SharedArbitrator::registerFactory();

  VELOX_CHECK_GT(FLAGS_rows_per_batch, 0);
  VELOX_CHECK_GT(FLAGS_num_batches, 0);
  VELOX_CHECK_GT(FLAGS_bucket_count, 0);
  VELOX_CHECK_GT(FLAGS_truncate_width, 0);
  VELOX_CHECK_GT(FLAGS_max_partitions_per_writers, 0);
  VELOX_CHECK_GT(FLAGS_root_pool_capacity_gb, 0);

  filesystems::registerLocalFileSystem();
  dwio::common::LocalFileSink::registerFactory();
  functions::iceberg::registerFunctions();

  context = std::make_unique<BenchmarkContext>();

  transformIdentity = std::make_unique<TransformBenchmark>(
      *context,
      ROW({"c0"}, {BIGINT()}),
      std::vector<PartitionField>{{0, TransformType::kIdentity, std::nullopt}});

  transformBucket = std::make_unique<TransformBenchmark>(
      *context,
      ROW({"c0"}, {BIGINT()}),
      std::vector<PartitionField>{
          {0, TransformType::kBucket, FLAGS_bucket_count}});

  transformTruncate = std::make_unique<TransformBenchmark>(
      *context,
      ROW({"c0"}, {VARCHAR()}),
      std::vector<PartitionField>{
          {0, TransformType::kTruncate, FLAGS_truncate_width}});

  transformYear = std::make_unique<TransformBenchmark>(
      *context,
      ROW({"c0"}, {DATE()}),
      std::vector<PartitionField>{{0, TransformType::kYear, std::nullopt}});

  transformMonth = std::make_unique<TransformBenchmark>(
      *context,
      ROW({"c0"}, {DATE()}),
      std::vector<PartitionField>{{0, TransformType::kMonth, std::nullopt}});

  transformDay = std::make_unique<TransformBenchmark>(
      *context,
      ROW({"c0"}, {DATE()}),
      std::vector<PartitionField>{{0, TransformType::kDay, std::nullopt}});

  transformHour = std::make_unique<TransformBenchmark>(
      *context,
      ROW({"c0"}, {TIMESTAMP()}),
      std::vector<PartitionField>{{0, TransformType::kHour, std::nullopt}});

  writeUnpartitioned = std::make_unique<WriteBenchmark>(
      *context, ROW({"c0"}, {BIGINT()}), std::vector<PartitionField>{});

  writeIdentity = std::make_unique<WriteBenchmark>(
      *context,
      ROW({"c0"}, {BIGINT()}),
      std::vector<PartitionField>{{0, TransformType::kIdentity, std::nullopt}});

  writeBucket = std::make_unique<WriteBenchmark>(
      *context,
      ROW({"c0"}, {BIGINT()}),
      std::vector<PartitionField>{
          {0, TransformType::kBucket, FLAGS_bucket_count}});

  writeTruncate = std::make_unique<WriteBenchmark>(
      *context,
      ROW({"c0"}, {VARCHAR()}),
      std::vector<PartitionField>{
          {0, TransformType::kTruncate, FLAGS_truncate_width}});

  writeYear = std::make_unique<WriteBenchmark>(
      *context,
      ROW({"c0"}, {DATE()}),
      std::vector<PartitionField>{{0, TransformType::kYear, std::nullopt}});

  writeMonth = std::make_unique<WriteBenchmark>(
      *context,
      ROW({"c0"}, {DATE()}),
      std::vector<PartitionField>{{0, TransformType::kMonth, std::nullopt}});

  writeDay = std::make_unique<WriteBenchmark>(
      *context,
      ROW({"c0"}, {DATE()}),
      std::vector<PartitionField>{{0, TransformType::kDay, std::nullopt}});

  writeHour = std::make_unique<WriteBenchmark>(
      *context,
      ROW({"c0"}, {TIMESTAMP()}),
      std::vector<PartitionField>{{0, TransformType::kHour, std::nullopt}});

  folly::runBenchmarks();

  writeHour.reset();
  writeDay.reset();
  writeMonth.reset();
  writeYear.reset();
  writeTruncate.reset();
  writeBucket.reset();
  writeIdentity.reset();
  writeUnpartitioned.reset();

  transformHour.reset();
  transformDay.reset();
  transformMonth.reset();
  transformYear.reset();
  transformTruncate.reset();
  transformBucket.reset();
  transformIdentity.reset();

  context.reset();

  return 0;
}
