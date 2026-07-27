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

#pragma once

#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include "velox/expression/fuzzer/ExpressionFuzzer.h"

namespace facebook::velox::exec::test {

class TableEvolutionFuzzer {
 public:
  struct Config {
    int columnCount;
    int evolutionCount;
    std::vector<dwio::common::FileFormat> formats;
    memory::MemoryPool* pool;

    /// Returns extra writer serde params to merge for one file, or none when
    /// unset. Called once per written file with the file's format and the
    /// fuzzer rng, so a driver can exercise format-specific write options,
    /// randomized yet reproducible from the seed. The core stays
    /// format-agnostic.
    std::function<std::unordered_map<std::string, std::string>(
        dwio::common::FileFormat,
        FuzzerGenerator&)>
        extraWriteSerdeParams;
  };

  /// Per-batch raw-byte target and clamp bounds for adaptive batch sizing. A
  /// batch is sized to about kTargetBatchBytes raw bytes regardless of schema
  /// width: narrow schemas get many rows, wide/nested schemas get few. Public
  /// so the adaptive sizing can be unit tested.
  static constexpr int64_t kTargetBatchBytes = 768 * 1024L;
  static constexpr int kMinAdaptiveVectorSize = 16;
  static constexpr int kMaxAdaptiveVectorSize = 50'000;

  /// Maps an estimated per-row raw byte cost to a per-batch row count
  /// (~targetBatchBytes per batch), clamped to [FLAGS_min_adaptive_vector_size,
  /// FLAGS_max_adaptive_vector_size] (which default to kMinAdaptiveVectorSize /
  /// kMaxAdaptiveVectorSize). A per-row cost below 1 byte is treated as 1.
  /// Defined in the .cpp so it can read the clamp-bound gflags.
  static int adaptiveVectorSizeForBytesPerRow(
      double bytesPerRow,
      int64_t targetBatchBytes = kTargetBatchBytes);

  explicit TableEvolutionFuzzer(const Config& config);

  static const std::string& connectorId();

  unsigned seed() const;

  void setSeed(unsigned seed);

  void reSeed();

  // Parses command-line inputted, comma separated string of file formats to be
  // used in the TableEvolutionFuzzer test run. The output is a vector of
  // dwio-packaged defined file formats.
  static const std::vector<dwio::common::FileFormat> parseFileFormats(
      std::string input);

  /// Returns true if 'columnName' is referenced by 'aggregationConfig's
  /// grouping keys or aggregate expressions.
  static bool isColumnUsedByAggregation(
      const std::string& columnName,
      const AggregationConfig& aggregationConfig);

  /// Selects a random subset of 'filteredColumns' to read filter-only (filtered
  /// but dropped from the scan output), exercising the selective reader's
  /// filter-only column path. A column is eligible only when it is a top-level,
  /// non-map, non-bucket column whose type is identical across all
  /// 'perEvolutionSchemas', and, when 'aggregationConfig' is set, is not used
  /// by the aggregation. Each eligible column is dropped with probability 1/2.
  static folly::F14FastSet<std::string> selectFilterOnlyColumns(
      const RowTypePtr& schema,
      const std::unordered_set<std::string>& filteredColumns,
      const std::vector<column_index_t>& bucketColumnIndices,
      const std::vector<RowTypePtr>& perEvolutionSchemas,
      const std::optional<AggregationConfig>& aggregationConfig,
      FuzzerGenerator& rng);

  /// Returns the column names in 'schema' order, excluding 'droppedColumns'.
  static std::vector<std::string> projectedColumnNames(
      const RowTypePtr& schema,
      const folly::F14FastSet<std::string>& droppedColumns);

  void run();

  virtual ~TableEvolutionFuzzer() = default;

 private:
  struct Setup {
    // Potentially with different field names, widened types, and additional
    // fields compared to previous setup.
    RowTypePtr schema;

    // New bucket count, must be a multiple of the bucket count in previous
    // setup.
    int log2BucketCount;

    dwio::common::FileFormat fileFormat;

    int bucketCount() const;
  };

  friend std::ostream& operator<<(
      std::ostream& out,
      const TableEvolutionFuzzer::Setup& setup);

  std::string makeNewName();

  TypePtr makeNewType(int maxDepth);

  RowTypePtr makeInitialSchema(
      const std::vector<std::string>& additionalColumnNames = {},
      const std::vector<TypePtr>& additionalColumnTypes = {});

  TypePtr evolveType(const TypePtr& old);

  RowTypePtr evolveRowType(
      const RowType& old,
      const std::vector<column_index_t>& bucketColumnIndices,
      std::unordered_map<std::string, std::string>* columnNameMapping =
          nullptr);

  std::vector<Setup> makeSetups(
      const std::vector<column_index_t>& bucketColumnIndices,
      const std::vector<std::string>& additionalColumnNames = {},
      const std::vector<TypePtr>& additionalColumnTypes = {},
      std::unordered_map<std::string, std::string>* columnNameMapping =
          nullptr);

  static std::unique_ptr<TaskCursor> makeWriteTask(
      const Setup& setup,
      const std::vector<RowVectorPtr>& dataBatches,
      const std::string& outputDir,
      const std::vector<column_index_t>& bucketColumnIndices,
      FuzzerGenerator& rng,
      bool enableFlatMap,
      folly::F14FastMap<int, folly::F14FastSet<std::string>>&
          globalMapColumnKeys,
      std::vector<int>& globallyCompatibleFlatmapColumns,
      const std::unordered_map<std::string, std::string>& extraSerdeParams);

  template <typename To, typename From>
  VectorPtr liftToPrimitiveType(
      const FlatVector<From>& input,
      const TypePtr& type);

  VectorPtr liftToType(const VectorPtr& input, const TypePtr& type);

  /// Builds a TableScan TaskCursor for one setup. When 'useFiltersAsNode' is
  /// true the filters are realized as a separate FilterNode above the scan (the
  /// reference plan that the pushdown plan is validated against); when false
  /// the filters are pushed down into the TableScan (the plan under test).
  std::unique_ptr<TaskCursor> makeScanTask(
      const RowTypePtr& tableSchema,
      std::vector<Split> splits,
      const PushdownConfig& pushdownConfig,
      bool useFiltersAsNode,
      bool insertProjectToBlockPushdown,
      const RowTypePtr& fullOutSchema,
      const std::vector<std::string>& outputColumnNames);

  /// Builds schema for flatmap as struct reading by converting selected map
  /// columns to struct types.
  RowTypePtr buildFlatmapAsStructSchema(
      const RowTypePtr& tableSchema,
      const folly::F14FastMap<int, folly::F14FastSet<std::string>>&
          globalMapColumnKeys,
      const std::vector<int>& globallyCompatibleFlatmapColumns);

  /// Randomly generates bucket column indices for partitioning data.
  /// Returns a vector of column indices that will be used for bucketing,
  /// with each column having a 1/(2*columnCount) probability of being selected.
  std::vector<column_index_t> generateBucketColumnIndices();

  /// Creates write tasks for all evolution steps.
  /// Generates test data and creates TaskCursor objects for writing data
  /// to temporary directories. Populates the writeTasks vector and collects the
  /// last evolution step's batches into finalExpectedBatches.
  void createWriteTasks(
      const std::vector<Setup>& testSetups,
      const std::vector<column_index_t>& bucketColumnIndices,
      const std::string& tableOutputRootDirPath,
      std::vector<std::shared_ptr<TaskCursor>>& writeTasks,
      std::vector<RowVectorPtr>& finalExpectedBatches,
      folly::F14FastMap<int, folly::F14FastSet<std::string>>&
          globalMapColumnKeys,
      std::vector<int>& globallyConsistentColumnIndexVector);

  /// Creates scan splits from write results.
  /// Converts the output of write tasks into scan splits that can be used
  /// for reading the written data back during the scan phase.
  std::pair<std::vector<Split>, std::vector<Split>>
  createScanSplitsFromWriteResults(
      const std::vector<std::vector<RowVectorPtr>>& writeResults,
      const std::vector<Setup>& testSetups,
      const std::vector<column_index_t>& bucketColumnIndices,
      std::optional<int32_t> selectedBucket);

  /// Applies remaining filters with updated column names.
  /// Updates filter expressions to use evolved column names based on the
  /// column name mapping tracked during schema evolution.
  void applyRemainingFilters(
      const fuzzer::ExpressionFuzzer::FuzzedExpressionData&
          generatedRemainingFilters,
      const std::unordered_map<std::string, std::string>& columnNameMapping,
      PushdownConfig& pushdownConfig,
      const std::unordered_set<std::string>& subfieldFilteredFields);

  /// Generates a single fresh query shape over the already-written files and
  /// verifies the pushdown plan against the FilterNode reference plan. Draws
  /// subfield filters, remaining-filter application, dropped filter-only
  /// columns, aggregation config, and the flatmap-as-struct read schema, then
  /// rebuilds the scan splits (no rewrite) and runs both scan tasks. Called
  /// once per query shape so a single write amortizes many shapes.
  void runQueryShape(
      const std::vector<std::vector<RowVectorPtr>>& writeResults,
      const std::vector<Setup>& testSetups,
      const std::vector<column_index_t>& bucketColumnIndices,
      const RowVectorPtr& finalExpectedData,
      const folly::F14FastMap<int, folly::F14FastSet<std::string>>&
          globalMapColumnKeys,
      const std::vector<int>& globallyConsistentColumnIndexVector,
      bool shouldGenerateRemainingFilters,
      const fuzzer::ExpressionFuzzer::FuzzedExpressionData&
          generatedRemainingFilters,
      const std::unordered_map<std::string, std::string>& columnNameMapping,
      folly::Executor& executor);

  const Config config_;
  VectorFuzzer vectorFuzzer_;
  unsigned currentSeed_;
  FuzzerGenerator rng_;
  int64_t sequenceNumber_ = 0;
};

} // namespace facebook::velox::exec::test
