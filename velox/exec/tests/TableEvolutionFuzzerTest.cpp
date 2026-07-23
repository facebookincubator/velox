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

#include "velox/exec/tests/TableEvolutionFuzzer.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/dwrf/RegisterDwrfReader.h"
#include "velox/dwio/dwrf/RegisterDwrfWriter.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include "velox/parse/TypeResolver.h"

DEFINE_uint32(seed, 0, "");
DEFINE_int32(table_evolution_fuzzer_duration_sec, 30, "");
DEFINE_int32(column_count, 5, "");
DEFINE_int32(evolution_count, 5, "");

// Defined in TableEvolutionFuzzer.cpp; declared here for the flag-validation
// test below.
DECLARE_int32(batches_per_file);
DECLARE_int64(batch_target_bytes);

namespace facebook::velox::exec::test {

namespace {

void registerFactories(folly::Executor* ioExecutor) {
  filesystems::registerLocalFileSystem();
  connector::hive::HiveConnectorFactory factory;
  auto hiveConnector = factory.newConnector(
      TableEvolutionFuzzer::connectorId(),
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>{
              {connector::hive::HiveConfig::kEnableFileHandleCache, "false"}}),
      ioExecutor);
  connector::ConnectorRegistry::global().insert(
      hiveConnector->connectorId(), hiveConnector);
  dwio::common::registerFileSinks();
  dwrf::registerDwrfReaderFactory();
  dwrf::registerDwrfWriterFactory();
}

TableEvolutionFuzzer::Config makeDwrfConfig(
    memory::MemoryPool* pool,
    int evolutionCount) {
  TableEvolutionFuzzer::Config config;
  config.pool = pool;
  config.columnCount = 5;
  config.evolutionCount = evolutionCount;
  config.formats = TableEvolutionFuzzer::parseFileFormats("dwrf");
  return config;
}

// Constructs the fuzzer with various evolutionCount values to pin the relaxed
// constructor assert (VELOX_CHECK_GT(evolutionCount, 0)). evolutionCount == 1
// is the no-evolution mode (single setup, no schema evolution) and must
// construct cleanly; evolutionCount == 0 must throw.
TEST(TableEvolutionFuzzerTest, constructorEvolutionCountBoundary) {
  auto pool = memory::memoryManager()->addLeafPool("TableEvolutionFuzzer");

  EXPECT_NO_THROW(
      { TableEvolutionFuzzer fuzzer(makeDwrfConfig(pool.get(), 1)); });
  EXPECT_NO_THROW(
      { TableEvolutionFuzzer fuzzer(makeDwrfConfig(pool.get(), 2)); });

  EXPECT_THROW(
      { TableEvolutionFuzzer fuzzer(makeDwrfConfig(pool.get(), 0)); },
      VeloxException);
}

// The constructor validates the batch-shaping gflags: a non-positive
// batches_per_file or batch_target_bytes must throw.
TEST(TableEvolutionFuzzerTest, constructorRejectsNonPositiveBatchFlags) {
  auto pool = memory::memoryManager()->addLeafPool("TableEvolutionFuzzer");

  {
    gflags::FlagSaver flagSaver;
    FLAGS_batches_per_file = 0;
    EXPECT_THROW(
        { TableEvolutionFuzzer fuzzer(makeDwrfConfig(pool.get(), 1)); },
        VeloxException);
  }
  {
    gflags::FlagSaver flagSaver;
    FLAGS_batch_target_bytes = 0;
    EXPECT_THROW(
        { TableEvolutionFuzzer fuzzer(makeDwrfConfig(pool.get(), 1)); },
        VeloxException);
  }
}

// Runs the fuzzer in no-evolution mode (evolutionCount == 1) for a small,
// fixed number of deterministic iterations. This exercises the no-evolution
// path together with filter-no-project (dropped filter-only columns) and the
// shared flatmap-as-struct read schema end to end; the fuzzer's internal
// pushdown-vs-FilterNode oracle is the assertion (run() throws on divergence).
TEST(TableEvolutionFuzzerTest, noEvolutionBoundedRun) {
  auto pool = memory::memoryManager()->addLeafPool("TableEvolutionFuzzer");
  TableEvolutionFuzzer fuzzer(makeDwrfConfig(pool.get(), 1));
  fuzzer.setSeed(20260629);
  constexpr int kIterations = 4;
  for (int i = 0; i < kIterations; ++i) {
    LOG(INFO) << "noEvolutionBoundedRun iteration " << i
              << ", seed=" << fuzzer.seed();
    EXPECT_NO_THROW(fuzzer.run());
    fuzzer.reSeed();
  }
}

// A column is "used by aggregation" if it is a grouping key or appears in an
// aggregate expression.
TEST(TableEvolutionFuzzerTest, isColumnUsedByAggregation) {
  AggregationConfig aggConfig;
  aggConfig.groupingKeys = {"g0", "g1"};
  aggConfig.aggregates = {"sum(a0)", "max(a1)"};

  EXPECT_TRUE(TableEvolutionFuzzer::isColumnUsedByAggregation("g0", aggConfig));
  EXPECT_TRUE(TableEvolutionFuzzer::isColumnUsedByAggregation("g1", aggConfig));
  EXPECT_TRUE(TableEvolutionFuzzer::isColumnUsedByAggregation("a0", aggConfig));
  EXPECT_TRUE(TableEvolutionFuzzer::isColumnUsedByAggregation("a1", aggConfig));
  EXPECT_FALSE(
      TableEvolutionFuzzer::isColumnUsedByAggregation("c0", aggConfig));
}

// projectedColumnNames returns the schema's columns in order, minus the dropped
// set; names not present in the schema are ignored.
TEST(TableEvolutionFuzzerTest, projectedColumnNames) {
  auto schema = ROW({{"c0", INTEGER()}, {"c1", BIGINT()}, {"c2", VARCHAR()}});

  EXPECT_EQ(
      TableEvolutionFuzzer::projectedColumnNames(schema, {"c1"}),
      (std::vector<std::string>{"c0", "c2"}));
  EXPECT_EQ(
      TableEvolutionFuzzer::projectedColumnNames(schema, {}),
      (std::vector<std::string>{"c0", "c1", "c2"}));
  EXPECT_EQ(
      TableEvolutionFuzzer::projectedColumnNames(schema, {"absent"}),
      (std::vector<std::string>{"c0", "c1", "c2"}));
}

// Only a top-level, non-map, non-bucket, type-stable, non-aggregation filtered
// column is eligible to be dropped filter-only; everything else is always
// projected. Across many seeds the eligible column is sometimes dropped and the
// ineligible ones never are.
TEST(TableEvolutionFuzzerTest, selectFilterOnlyColumns) {
  auto schema = ROW({
      {"c_scalar", INTEGER()}, // eligible
      {"c_map", MAP(VARCHAR(), INTEGER())}, // map -> excluded
      {"c_bucket", INTEGER()}, // bucket column -> excluded
      {"c_unstable", DOUBLE()}, // differs from an earlier setup -> excluded
      {"c_agg", INTEGER()}, // used by the aggregation -> excluded
      {"c_unfiltered", INTEGER()}, // not filtered -> excluded
  });
  // An earlier setup where c_unstable (positional index 3) was REAL.
  auto earlierSetup = ROW({
      {"c_scalar", INTEGER()},
      {"c_map", MAP(VARCHAR(), INTEGER())},
      {"c_bucket", INTEGER()},
      {"c_unstable", REAL()},
      {"c_agg", INTEGER()},
      {"c_unfiltered", INTEGER()},
  });
  const std::unordered_set<std::string> filtered = {
      "c_scalar", "c_map", "c_bucket", "c_unstable", "c_agg"};
  const std::vector<column_index_t> bucketColumnIndices = {2}; // c_bucket
  AggregationConfig aggConfig;
  aggConfig.aggregates = {"sum(c_agg)"};

  bool everDroppedScalar = false;
  for (uint32_t seed = 0; seed < 50; ++seed) {
    FuzzerGenerator rng(seed);
    const auto dropped = TableEvolutionFuzzer::selectFilterOnlyColumns(
        schema,
        filtered,
        bucketColumnIndices,
        {schema, earlierSetup},
        aggConfig,
        rng);
    for (const auto& name : dropped) {
      EXPECT_EQ(name, "c_scalar") << "unexpectedly dropped: " << name;
    }
    everDroppedScalar |= dropped.count("c_scalar") > 0;
  }
  EXPECT_TRUE(everDroppedScalar);
}
// Runs the fuzzer in evolution mode (evolutionCount > 1) for a small, fixed
// number of deterministic iterations, exercising the
// multiple-batches-per-file write path together with schema evolution. Each
// file holds batches_per_file independently fuzzed batches; the actual file and
// the lifted expected file are written from the SAME batches, and the caller
// merges the final setup's batches for filter generation, so this covers the
// multi-batch write + liftToType + merge path across evolving schemas, which
// the single-schema noEvolutionBoundedRun does not. The pushdown-vs-FilterNode
// oracle inside run() is the assertion.
TEST(TableEvolutionFuzzerTest, multiBatchEvolutionBoundedRun) {
  auto pool = memory::memoryManager()->addLeafPool("TableEvolutionFuzzer");
  TableEvolutionFuzzer fuzzer(makeDwrfConfig(pool.get(), 3));
  fuzzer.setSeed(20260629);
  constexpr int kIterations = 4;
  for (int i = 0; i < kIterations; ++i) {
    LOG(INFO) << "multiBatchEvolutionBoundedRun iteration " << i
              << ", seed=" << fuzzer.seed();
    EXPECT_NO_THROW(fuzzer.run());
    fuzzer.reSeed();
  }
}

// Exercises the pure adaptive batch-size clamp: narrow rows (few bytes each)
// saturate at the max row count, very wide rows floor at the min, and in the
// unclamped band the row count is kTargetBatchBytes / bytesPerRow and is
// monotonically non-increasing in bytesPerRow.
TEST(TableEvolutionFuzzerTest, adaptiveVectorSizeClampsToByteTarget) {
  using Fuzzer = TableEvolutionFuzzer;

  // Narrow rows -> many rows, capped at the max. A sub-byte estimate is treated
  // as 1 byte/row, so it also saturates at the cap.
  EXPECT_EQ(
      Fuzzer::adaptiveVectorSizeForBytesPerRow(1.0),
      Fuzzer::kMaxAdaptiveVectorSize);
  EXPECT_EQ(
      Fuzzer::adaptiveVectorSizeForBytesPerRow(0.0),
      Fuzzer::kMaxAdaptiveVectorSize);

  // Very wide rows -> few rows, floored at the min.
  EXPECT_EQ(
      Fuzzer::adaptiveVectorSizeForBytesPerRow(Fuzzer::kTargetBatchBytes),
      Fuzzer::kMinAdaptiveVectorSize);
  EXPECT_EQ(
      Fuzzer::adaptiveVectorSizeForBytesPerRow(1e12),
      Fuzzer::kMinAdaptiveVectorSize);

  // In the unclamped band, rows == kTargetBatchBytes / bytesPerRow.
  constexpr double kBytesPerRow = 1024.0;
  const int expected =
      static_cast<int>(Fuzzer::kTargetBatchBytes / kBytesPerRow);
  EXPECT_EQ(Fuzzer::adaptiveVectorSizeForBytesPerRow(kBytesPerRow), expected);
  EXPECT_GT(expected, Fuzzer::kMinAdaptiveVectorSize);
  EXPECT_LT(expected, Fuzzer::kMaxAdaptiveVectorSize);

  // Monotonic non-increasing: wider rows never yield more rows.
  EXPECT_GE(
      Fuzzer::adaptiveVectorSizeForBytesPerRow(512.0),
      Fuzzer::adaptiveVectorSizeForBytesPerRow(4096.0));
}

// The byte target (the batch_target_bytes gflag, passed as the second arg)
// scales the per-batch row count: in the unclamped band the count is
// targetBatchBytes / bytesPerRow, so doubling the target doubles the rows. The
// one-arg form defaults to kTargetBatchBytes.
TEST(TableEvolutionFuzzerTest, adaptiveVectorSizeScalesWithByteTarget) {
  using Fuzzer = TableEvolutionFuzzer;
  constexpr double kBytesPerRow = 512.0;

  EXPECT_EQ(
      Fuzzer::adaptiveVectorSizeForBytesPerRow(kBytesPerRow),
      Fuzzer::adaptiveVectorSizeForBytesPerRow(
          kBytesPerRow, Fuzzer::kTargetBatchBytes));

  const int base =
      Fuzzer::adaptiveVectorSizeForBytesPerRow(kBytesPerRow, 4LL << 20);
  const int doubled =
      Fuzzer::adaptiveVectorSizeForBytesPerRow(kBytesPerRow, 8LL << 20);

  // Both land in the unclamped band, and the larger target yields exactly twice
  // the rows.
  EXPECT_GT(base, Fuzzer::kMinAdaptiveVectorSize);
  EXPECT_LT(doubled, Fuzzer::kMaxAdaptiveVectorSize);
  EXPECT_EQ(doubled, 2 * base);
}

TEST(TableEvolutionFuzzerTest, run) {
  auto pool = memory::memoryManager()->addLeafPool("TableEvolutionFuzzer");
  exec::test::TableEvolutionFuzzer::Config config;
  config.pool = pool.get();
  config.columnCount = FLAGS_column_count;
  config.evolutionCount = FLAGS_evolution_count;
  config.formats = TableEvolutionFuzzer::parseFileFormats("dwrf");
  LOG(INFO) << "Running TableEvolutionFuzzer with seed " << FLAGS_seed;
  exec::test::TableEvolutionFuzzer fuzzer(config);
  fuzzer.setSeed(FLAGS_seed);
  const auto startTime = std::chrono::system_clock::now();
  const auto deadline = startTime +
      std::chrono::seconds(FLAGS_table_evolution_fuzzer_duration_sec);
  for (int i = 0; std::chrono::system_clock::now() < deadline; ++i) {
    LOG(INFO) << "Starting iteration " << i << ", seed=" << fuzzer.seed();
    fuzzer.run();
    fuzzer.reSeed();
  }
}

} // namespace

} // namespace facebook::velox::exec::test

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  if (gflags::GetCommandLineFlagInfoOrDie("seed").is_default) {
    FLAGS_seed = std::random_device{}();
    LOG(INFO) << "Use generated random seed " << FLAGS_seed;
  }
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});
  auto ioExecutor = folly::getGlobalIOExecutor();
  facebook::velox::exec::test::registerFactories(ioExecutor.get());
  facebook::velox::functions::prestosql::registerAllScalarFunctions();
  facebook::velox::aggregate::prestosql::registerAllAggregateFunctions();
  facebook::velox::parse::registerTypeResolver();
  return RUN_ALL_TESTS();
}
