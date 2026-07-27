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
#include "velox/common/config/Config.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/core/QueryCtx.h"
#include "velox/dwio/common/tests/utils/FilterGenerator.h"
#include "velox/dwio/dwrf/common/Config.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/expression/fuzzer/ExpressionFuzzer.h"
#include "velox/expression/fuzzer/FuzzerToolkit.h"
#include "velox/functions/FunctionRegistry.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

#include <algorithm>
#include <filesystem>

#include <re2/re2.h>

DEFINE_bool(
    enable_oom_injection_write_path,
    false,
    "When enabled OOMs will randomly be triggered while executing the write path "
    "The goal of this mode is to ensure unexpected exceptions "
    "aren't thrown and the process isn't killed in the process of cleaning "
    "up after failures. Therefore, results are not compared when this is "
    "enabled. Note that this option only works in debug builds.");

DEFINE_bool(
    enable_oom_injection_read_path,
    false,
    "When enabled OOMs will randomly be triggered while executing scan "
    "plans. The goal of this mode is to ensure unexpected exceptions "
    "aren't thrown and the process isn't killed in the process of cleaning "
    "up after failures. Therefore, results are not compared when this is "
    "enabled. Note that this option only works in debug builds.");

DEFINE_int32(
    aggregation_pushdown_frequency,
    5,
    "Controls the frequency of aggregation pushdown. The aggregation pushdown "
    "is enabled with probability 1/N where N is this value. For example, "
    "N=5 means 20% chance, N=2 means 50% chance.");

DEFINE_int32(
    batches_per_file,
    8,
    "Number of independently-fuzzed batches written per file (one writer "
    "write() call each). A fixed count is sufficient: chunk and stripe flush "
    "randomness already varies the physical layout across files.");

DEFINE_bool(
    adaptive_batch_sizing,
    true,
    "When true, each written batch is sized adaptively to about "
    "--batch_target_bytes raw bytes (the per-batch row count is derived from "
    "the schema's per-row cost, clamped to [--min_adaptive_vector_size, "
    "--max_adaptive_vector_size]). When false, batches use a fixed row count.");

DEFINE_int64(
    batch_target_bytes,
    facebook::velox::exec::test::TableEvolutionFuzzer::kTargetBatchBytes,
    "Target raw size in bytes for each written batch. The per-batch row count "
    "is derived from this target so batch bytes stay schema-independent across "
    "narrow and wide/nested schemas. Only used when --adaptive_batch_sizing.");

DEFINE_int32(
    min_adaptive_vector_size,
    facebook::velox::exec::test::TableEvolutionFuzzer::kMinAdaptiveVectorSize,
    "Lower bound on the adaptive per-batch row count "
    "(used when --adaptive_batch_sizing).");

DEFINE_int32(
    max_adaptive_vector_size,
    facebook::velox::exec::test::TableEvolutionFuzzer::kMaxAdaptiveVectorSize,
    "Upper bound on the adaptive per-batch row count "
    "(used when --adaptive_batch_sizing).");

namespace facebook::velox::exec::test {
using namespace facebook::velox::common::testutil;

std::ostream& operator<<(
    std::ostream& out,
    const TableEvolutionFuzzer::Setup& setup) {
  out << "schema=" << setup.schema->toString()
      << " log2BucketCount=" << setup.log2BucketCount
      << " fileFormat=" << setup.fileFormat;
  return out;
}

namespace {

// Default vector size for the fuzzer; the actual per-batch row count is chosen
// adaptively per setup to hit a byte target (see computeAdaptiveVectorSize).
constexpr int kDefaultVectorSize = 101;

// Number of rows in the probe batch used to estimate per-row raw size. The
// per-batch byte target and clamp bounds live on TableEvolutionFuzzer
// (kTargetBatchBytes / kMinAdaptiveVectorSize / kMaxAdaptiveVectorSize) so the
// adaptive sizing can be unit tested.
constexpr int kProbeRows = 64;

// Number of distinct query shapes run against each written set of files. The
// (now expensive) write happens once per run(); this many shapes amortize it.
constexpr int kQueryShapesPerFile = 20;

VectorFuzzer::Options makeVectorFuzzerOptions() {
  VectorFuzzer::Options options;
  options.vectorSize = kDefaultVectorSize;
  options.allowSlice = false;
  return options;
}

// Estimates the per-row raw byte cost of 'schema' by fuzzing a small probe
// batch, then returns the adaptive per-batch row count for that cost (see
// TableEvolutionFuzzer::adaptiveVectorSizeForBytesPerRow).
int computeAdaptiveVectorSize(
    VectorFuzzer& vectorFuzzer,
    const RowTypePtr& schema) {
  auto probe = vectorFuzzer.fuzzRow(schema, kProbeRows, false);
  const uint64_t probeRawSize = probe->estimateFlatSize();
  return TableEvolutionFuzzer::adaptiveVectorSizeForBytesPerRow(
      static_cast<double>(probeRawSize) / kProbeRows, FLAGS_batch_target_bytes);
}

template <typename T>
void removeFromVector(std::vector<T>& vec, const T& value) {
  auto it = std::find(vec.begin(), vec.end(), value);
  if (it != vec.end()) {
    vec.erase(it);
  }
}

bool hasUnsupportedMapKey(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::MAP: {
      auto mapType = type->asMap();
      // FlatMapColumnWriter only supports TINYINT, SMALLINT, INTEGER, BIGINT,
      // VARCHAR, VARBINARY as KeyType
      auto keyKind = mapType.keyType()->kind();
      if (keyKind != TypeKind::TINYINT && keyKind != TypeKind::SMALLINT &&
          keyKind != TypeKind::INTEGER && keyKind != TypeKind::BIGINT &&
          keyKind != TypeKind::VARCHAR && keyKind != TypeKind::VARBINARY) {
        return true;
      }
      return hasUnsupportedMapKey(mapType.valueType());
    }
    case TypeKind::ARRAY:
      return hasUnsupportedMapKey(type->asArray().elementType());
    case TypeKind::ROW: {
      auto& rowType = type->asRow();
      for (int i = 0; i < rowType.size(); ++i) {
        if (hasUnsupportedMapKey(rowType.childAt(i))) {
          return true;
        }
      }
      return false;
    }
    default:
      return false;
  }
}

bool hasMapColumns(const RowTypePtr& schema) {
  VLOG(1) << "Checking if schema has map columns";
  for (int i = 0; i < schema->size(); ++i) {
    if (schema->childAt(i)->isMap()) {
      return true;
    }
  }
  return false;
}

bool hasEmptyElement(const RowVectorPtr& data, int columnIndex) {
  auto mapVector = data->childAt(columnIndex)->as<MapVector>();
  if (!mapVector) {
    return true;
  }

  // Check if any map entry is empty (null or size = 0)
  auto sizes = mapVector->sizes();
  for (int j = 0; j < mapVector->size(); ++j) {
    if (mapVector->isNullAt(j) || sizes->as<vector_size_t>()[j] == 0) {
      return true; // Found an empty map
    }
  }
  return false; // No empty maps found
}

} // namespace

int TableEvolutionFuzzer::adaptiveVectorSizeForBytesPerRow(
    double bytesPerRow,
    int64_t targetBatchBytes) {
  const double safeBytesPerRow = std::max(1.0, bytesPerRow);
  // Clamp as double before narrowing: a large --batch_target_bytes over a tiny
  // per-row cost can exceed INT_MAX, and casting that to int before clamping
  // would be undefined behavior. The clamp bounds are well within int range.
  const double rows = static_cast<double>(targetBatchBytes) / safeBytesPerRow;
  return static_cast<int>(std::clamp<double>(
      rows,
      static_cast<double>(FLAGS_min_adaptive_vector_size),
      static_cast<double>(FLAGS_max_adaptive_vector_size)));
}

TableEvolutionFuzzer::TableEvolutionFuzzer(const Config& config)
    : config_(config), vectorFuzzer_(makeVectorFuzzerOptions(), config.pool) {
  VELOX_CHECK_GT(config_.columnCount, 0);
  VELOX_CHECK_GT(config_.evolutionCount, 0);
  VELOX_CHECK_GT(FLAGS_batches_per_file, 0);
  VELOX_CHECK_GT(FLAGS_batch_target_bytes, 0);
  VELOX_CHECK_GT(FLAGS_min_adaptive_vector_size, 0);
  VELOX_CHECK_GE(
      FLAGS_max_adaptive_vector_size, FLAGS_min_adaptive_vector_size);
}

const std::string& TableEvolutionFuzzer::connectorId() {
  static const std::string connectorId(PlanBuilder::kHiveDefaultConnectorId);
  return connectorId;
}

unsigned TableEvolutionFuzzer::seed() const {
  return currentSeed_;
}

void TableEvolutionFuzzer::setSeed(unsigned seed) {
  currentSeed_ = seed;
  rng_.seed(seed);
  vectorFuzzer_.reSeed(rng_());
}

void TableEvolutionFuzzer::reSeed() {
  setSeed(rng_());
}

const std::vector<dwio::common::FileFormat>
TableEvolutionFuzzer::parseFileFormats(std::string input) {
  std::vector<std::string> formatsAsStrings;
  folly::split(",", input, formatsAsStrings);
  VELOX_CHECK(!formatsAsStrings.empty(), "No file formats specified");
  std::vector<dwio::common::FileFormat> formats;
  for (const auto& formatAsString : formatsAsStrings) {
    auto format = dwio::common::toFileFormat(formatAsString);
    VELOX_CHECK_NE(
        format,
        dwio::common::FileFormat::UNKNOWN,
        "Config contains UNKNOWN file format");
    formats.push_back(format);
  }
  return formats;
}

namespace {

// Helper function to randomly select aggregates from available columns
// without replacement. Returns a list of aggregate expressions.
void generateAggregatesForColumns(
    const std::vector<int>& availableColumns,
    const std::vector<std::string>& supportedAggFuncs,
    const RowTypePtr& schema,
    FuzzerGenerator& rng,
    std::vector<std::string>& aggregates) {
  if (availableColumns.empty()) {
    return;
  }

  int numAggregates = std::min(
      static_cast<int>(availableColumns.size()),
      std::min(
          static_cast<int>(5),
          static_cast<int>(
              folly::Random::rand32(1, availableColumns.size() + 1, rng))));

  std::unordered_set<int> selectedIndices;
  for (int i = 0; i < numAggregates; ++i) {
    if (folly::Random::oneIn(2, rng)) {
      int randomIdx;
      do {
        randomIdx = folly::Random::rand32(availableColumns.size(), rng);
      } while (selectedIndices.count(randomIdx) > 0);
      selectedIndices.insert(randomIdx);

      int colIdx = availableColumns[randomIdx];
      std::string aggFunc = supportedAggFuncs[folly::Random::rand32(
          supportedAggFuncs.size(), rng)];
      aggregates.push_back(
          fmt::format("{}({})", aggFunc, schema->nameOf(colIdx)));
    }
  }
}

std::vector<std::vector<RowVectorPtr>> runTaskCursors(
    const std::vector<std::shared_ptr<TaskCursor>>& cursors,
    folly::Executor& executor) {
  std::vector<folly::SemiFuture<std::vector<RowVectorPtr>>> futures;
  for (int i = 0; i < cursors.size(); ++i) {
    auto [promise, future] =
        folly::makePromiseContract<std::vector<RowVectorPtr>>();
    futures.push_back(std::move(future));
    auto cursorPtr = cursors[i];
    auto task = cursorPtr->task();
    executor.add([cursorPtr, task, promise = std::move(promise)]() mutable {
      std::vector<RowVectorPtr> results;
      try {
        while (cursorPtr->moveNext()) {
          auto& result = cursorPtr->current();
          result->loadedVector();
          results.push_back(std::move(result));
        }
        promise.setValue(std::move(results));
      } catch (VeloxRuntimeError& e) {
        if (FLAGS_enable_oom_injection_write_path &&
            e.errorCode() == facebook::velox::error_code::kMemCapExceeded &&
            e.message() == ScopedOOMInjector::kErrorMessage) {
          // If we enabled OOM injection we expect the exception thrown by the
          // ScopedOOMInjector.
          LOG(INFO) << "OOM injection triggered in write path: " << e.what();
          promise.setValue(std::move(results));
        } else if (
            FLAGS_enable_oom_injection_read_path &&
            e.errorCode() == facebook::velox::error_code::kMemCapExceeded &&
            e.message() == ScopedOOMInjector::kErrorMessage) {
          // If we enabled OOM injection we expect the exception thrown by the
          // ScopedOOMInjector.
          LOG(INFO) << "OOM injection triggered in read path: " << e.what();
          promise.setValue(std::move(results));
        } else {
          LOG(ERROR) << e.what();
          promise.setException(e);
        }
      } catch (const std::exception& e) {
        LOG(ERROR) << e.what();
        promise.setException(e);
      }
    });
  }
  std::vector<std::vector<RowVectorPtr>> results;
  results.reserve(futures.size());
  for (auto& future : futures) {
    results.push_back(std::move(future).get());
  }
  return results;
}
// `tableBucketCount' is the bucket count of current table setup when reading.
// `partitionBucketCount' is the bucket count when the partition was written.
// `tableBucketCount' must be a multiple of `partitionBucketCount'.
void buildScanSplitFromTableWriteResult(
    const RowTypePtr& tableSchema,
    const std::vector<column_index_t>& bucketColumnIndices,
    std::optional<int32_t> tableBucket,
    int tableBucketCount,
    int partitionBucketCount,
    dwio::common::FileFormat fileFormat,
    const std::vector<RowVectorPtr>& writeResult,
    std::vector<Split>& splits) {
  if (FLAGS_enable_oom_injection_write_path) {
    return;
  }
  VELOX_CHECK_EQ(writeResult.size(), 1);
  auto* fragments =
      writeResult[0]->childAt(1)->asChecked<SimpleVector<StringView>>();
  for (int i = 1; i < writeResult[0]->size(); ++i) {
    auto fragment = folly::parseJson(std::string_view(fragments->valueAt(i)));
    auto fileName = fragment["fileWriteInfos"][0]["writeFileName"].asString();
    auto hiveSplit = std::make_shared<connector::hive::HiveConnectorSplit>(
        TableEvolutionFuzzer::connectorId(),
        fmt::format("{}/{}", fragment["writePath"].asString(), fileName),
        fileFormat);
    if (!tableBucket.has_value()) {
      splits.emplace_back(std::move(hiveSplit));
      continue;
    }

    auto fileBucketEnd = fileName.find('_');
    VELOX_CHECK_NE(fileBucketEnd, fileName.npos);
    auto fileBucket = folly::to<int32_t>(fileName.substr(0, fileBucketEnd));
    if (*tableBucket % partitionBucketCount != fileBucket) {
      continue;
    }
    hiveSplit->tableBucketNumber = tableBucket;
    if (partitionBucketCount != tableBucketCount) {
      auto& bucketConversion = hiveSplit->bucketConversion.emplace();
      bucketConversion.tableBucketCount = tableBucketCount;
      bucketConversion.partitionBucketCount = partitionBucketCount;
      for (auto bucketColumnIndex : bucketColumnIndices) {
        auto handle = std::make_unique<connector::hive::HiveColumnHandle>(
            tableSchema->nameOf(bucketColumnIndex),
            connector::hive::FileColumnHandle::ColumnType::kRegular,
            tableSchema->childAt(bucketColumnIndex),
            tableSchema->childAt(bucketColumnIndex));
        bucketConversion.bucketColumnHandles.push_back(std::move(handle));
      }
    }
    splits.emplace_back(std::move(hiveSplit));
  }
}

void checkResultsEqual(
    const std::vector<RowVectorPtr>& actual,
    const std::vector<RowVectorPtr>& expected) {
  int actualVectorIndex = 0;
  int expectedVectorIndex = 0;
  int actualRowIndex = 0, expectedRowIndex = 0;
  while (actualVectorIndex < actual.size() &&
         expectedVectorIndex < expected.size()) {
    if (actualRowIndex == actual[actualVectorIndex]->size()) {
      ++actualVectorIndex;
      actualRowIndex = 0;
      continue;
    }
    if (expectedRowIndex == expected[expectedVectorIndex]->size()) {
      ++expectedVectorIndex;
      expectedRowIndex = 0;
      continue;
    }
    VELOX_CHECK(
        actual[actualVectorIndex]->equalValueAt(
            expected[expectedVectorIndex].get(),
            actualRowIndex,
            expectedRowIndex),
        "actualVectorIndex={} actualRowIndex={} expectedVectorIndex={} expectedRowIndex={}\nactual={}\nexpected={}",
        actualVectorIndex,
        actualRowIndex,
        expectedVectorIndex,
        expectedRowIndex,
        actual[actualVectorIndex]->toString(actualRowIndex),
        expected[expectedVectorIndex]->toString(expectedRowIndex));
    ++actualRowIndex;
    ++expectedRowIndex;
  }
  if (actualVectorIndex < actual.size() &&
      actualRowIndex == actual[actualVectorIndex]->size()) {
    ++actualVectorIndex;
    actualRowIndex = 0;
  }
  if (expectedVectorIndex < expected.size() &&
      expectedRowIndex == expected[expectedVectorIndex]->size()) {
    ++expectedVectorIndex;
    expectedRowIndex = 0;
  }
  VELOX_CHECK_EQ(actualVectorIndex, actual.size());
  VELOX_CHECK_EQ(expectedVectorIndex, expected.size());
}

common::SubfieldFilters generateSubfieldFilters(
    RowTypePtr& rowType,
    const RowVectorPtr& finalExpectedData) {
  dwio::common::MutationSpec mutations;
  std::vector<uint64_t> hitRows;

  std::unique_ptr<velox::dwio::common::FilterGenerator> filterGenerator =
      std::make_unique<velox::dwio::common::FilterGenerator>(rowType, 0);

  auto subfieldsVector = filterGenerator->makeFilterables(rowType->size(), 100);

  const auto& filterSpecs =
      filterGenerator->makeRandomSpecs(subfieldsVector, 100);

  return filterGenerator->makeSubfieldFilters(
      filterSpecs, {finalExpectedData}, &mutations, hitRows);
}

fuzzer::ExpressionFuzzer::FuzzedExpressionData generateRemainingFilters(
    const TableEvolutionFuzzer::Config& config,
    unsigned currentSeed) {
  // Use ExpressionFuzzer to generate complex expressions, but use the actual
  // data types from finalExpectedData
  // Configure ExpressionFuzzer to generate simpler expressions suitable for
  // filters
  fuzzer::ExpressionFuzzer::Options options;
  options.enableComplexTypes = false; // Disable complex types to avoid issues
  options.enableDecimalType = false; // Disable decimal types
  options.maxLevelOfNesting = 3; // Reduce nesting to avoid complexity
  options.nullRatio = 0.0; // No null values to avoid type resolution issues
  // Only use simple comparison and logical functions suitable for filters
  options.useOnlyFunctions = "eq,neq,lt,lte,gt,gte,and,or,not";
  options.specialForms = "and,or"; // Only simple special forms

  // Skip complex functions that generate unparseable expressions
  options.skipFunctions = {
      "regexp_like",
      "regexp_extract",
      "replace",
      "replace_first",
      "json_format",
      "json_extract",
      "json_parse",
      "from_utf8",
      "to_utf8",
      "reverse",
      "upper",
      "lower",
      "st_coorddim",
      "is_null",
      "is_not_null"};

  auto signatureMap = getVectorFunctionSignatures();

  // Configure VectorFuzzer to avoid null values and use the actual data types
  VectorFuzzer::Options vectorFuzzerOptions;
  vectorFuzzerOptions.nullRatio = 0.0; // No nulls
  vectorFuzzerOptions.vectorSize = 100;
  auto vectorFuzzer =
      std::make_shared<VectorFuzzer>(vectorFuzzerOptions, config.pool);

  fuzzer::ExpressionFuzzer expressionFuzzer(
      signatureMap, currentSeed, vectorFuzzer, options);

  return expressionFuzzer.fuzzExpressions(1);
}

// Generate random aggregation configuration for pushdown testing.
// Only generates aggregations that are eligible for pushdown:
// - Supported aggregate functions: min, max, bool_and, bool_or
// - Each column can only be used by at most one aggregate
// - Grouping keys are optional (can be empty for global aggregation)
// - Columns with filters (subfield or remaining) are excluded to enable
// pushdown
std::optional<AggregationConfig> generateAggregationConfig(
    const RowTypePtr& schema,
    FuzzerGenerator& rng,
    const std::unordered_set<std::string>& filteredColumns) {
  // List of aggregate functions that support pushdown
  // Note: Excluding 'sum' to avoid integer overflow in fuzzer with random data
  static const std::vector<std::string> supportedNumericAggs = {"min", "max"};
  static const std::vector<std::string> supportedBooleanAggs = {
      "bool_and", "bool_or"};
  static const std::vector<std::string> supportedIntegerAggs = {
      "bitwise_and_agg", "bitwise_or_agg", "bitwise_xor_agg"};

  // Randomly decide number of grouping keys (0 to 2)
  int numGroupingKeys = folly::Random::rand32(3, rng);
  std::vector<std::string> groupingKeys;
  std::unordered_set<int> usedColumnIndices;

  // Select random columns for grouping keys
  for (int i = 0; i < numGroupingKeys && i < schema->size(); ++i) {
    int colIdx = folly::Random::rand32(schema->size(), rng);
    if (usedColumnIndices.count(colIdx) == 0) {
      groupingKeys.push_back(schema->nameOf(colIdx));
      usedColumnIndices.insert(colIdx);
    }
  }

  // Generate aggregates on remaining columns
  // For aggregation pushdown to work, each column should only be used once
  // and columns with filters should be excluded
  std::vector<std::string> aggregates;
  std::vector<int> availableNumericColumns;
  std::vector<int> availableIntegerColumns;
  std::vector<int> availableBooleanColumns;
  for (int i = 0; i < schema->size(); ++i) {
    if (usedColumnIndices.count(i) == 0) {
      auto columnName = schema->nameOf(i);
      // Skip columns that have filters (subfield or remaining)
      if (filteredColumns.count(columnName) > 0) {
        continue;
      }

      auto type = schema->childAt(i);
      // Integer types: randomly choose between min/max or bitwise aggregations
      // Note: Exclude DATE/Interval type as it doesn't support bitwise
      // aggregations
      if ((type->isInteger() || type->isBigint() || type->isSmallint() ||
           type->isTinyint()) &&
          !type->isDate() && !type->isIntervalDayTime() &&
          !type->isIntervalYearMonth()) {
        if (folly::Random::oneIn(2, rng)) {
          availableIntegerColumns.push_back(i);
        } else {
          availableNumericColumns.push_back(i);
        }
      }
      // Float types support min/max only
      else if ((type->isReal() || type->isDouble()) && !type->isDecimal()) {
        availableNumericColumns.push_back(i);
      }
      // Boolean types support bool_and/bool_or
      else if (type->isBoolean()) {
        availableBooleanColumns.push_back(i);
      }
    }
  }

  // Need at least one column to aggregate
  if (availableNumericColumns.empty() && availableBooleanColumns.empty() &&
      availableIntegerColumns.empty()) {
    return std::nullopt;
  }

  // Randomly pick columns for aggregates without replacement
  generateAggregatesForColumns(
      availableNumericColumns, supportedNumericAggs, schema, rng, aggregates);
  generateAggregatesForColumns(
      availableBooleanColumns, supportedBooleanAggs, schema, rng, aggregates);
  generateAggregatesForColumns(
      availableIntegerColumns, supportedIntegerAggs, schema, rng, aggregates);

  if (aggregates.empty()) {
    return std::nullopt;
  }

  return AggregationConfig{
      .groupingKeys = std::move(groupingKeys),
      .aggregates = std::move(aggregates)};
}

} // namespace

VectorPtr TableEvolutionFuzzer::liftToType(
    const VectorPtr& input,
    const TypePtr& type) {
  switch (input->typeKind()) {
    case TypeKind::TINYINT: {
      auto* typed = input->asChecked<FlatVector<int8_t>>();
      switch (type->kind()) {
        case TypeKind::TINYINT:
          return input;
        case TypeKind::SMALLINT:
          return liftToPrimitiveType<int16_t>(*typed, type);
        case TypeKind::INTEGER:
          return liftToPrimitiveType<int32_t>(*typed, type);
        case TypeKind::BIGINT:
          return liftToPrimitiveType<int64_t>(*typed, type);
        default:
          VELOX_UNREACHABLE();
      }
    }
    case TypeKind::SMALLINT: {
      auto* typed = input->asChecked<FlatVector<int16_t>>();
      switch (type->kind()) {
        case TypeKind::SMALLINT:
          return input;
        case TypeKind::INTEGER:
          return liftToPrimitiveType<int32_t>(*typed, type);
        case TypeKind::BIGINT:
          return liftToPrimitiveType<int64_t>(*typed, type);
        default:
          VELOX_UNREACHABLE();
      }
    }
    case TypeKind::INTEGER: {
      auto* typed = input->asChecked<FlatVector<int32_t>>();
      switch (type->kind()) {
        case TypeKind::INTEGER:
          return input;
        case TypeKind::BIGINT:
          return liftToPrimitiveType<int64_t>(*typed, type);
        default:
          VELOX_UNREACHABLE();
      }
    }
    case TypeKind::REAL: {
      auto* typed = input->asChecked<FlatVector<float>>();
      switch (type->kind()) {
        case TypeKind::REAL:
          return input;
        case TypeKind::DOUBLE:
          return liftToPrimitiveType<double>(*typed, type);
        default:
          VELOX_UNREACHABLE();
      }
    }
    case TypeKind::ARRAY: {
      VELOX_CHECK_EQ(type->kind(), TypeKind::ARRAY);
      auto* array = input->asChecked<ArrayVector>();
      return std::make_shared<ArrayVector>(
          config_.pool,
          type,
          array->nulls(),
          array->size(),
          array->offsets(),
          array->sizes(),
          liftToType(array->elements(), type->asArray().elementType()));
    }
    case TypeKind::MAP: {
      VELOX_CHECK_EQ(type->kind(), TypeKind::MAP);
      auto& mapType = type->asMap();
      auto* map = input->asChecked<MapVector>();
      return std::make_shared<MapVector>(
          config_.pool,
          type,
          map->nulls(),
          map->size(),
          map->offsets(),
          map->sizes(),
          liftToType(map->mapKeys(), mapType.keyType()),
          liftToType(map->mapValues(), mapType.valueType()));
    }
    case TypeKind::ROW: {
      VELOX_CHECK_EQ(type->kind(), TypeKind::ROW);
      auto& rowType = type->asRow();
      auto* row = input->asChecked<RowVector>();
      auto children = row->children();
      for (int i = 0; i < rowType.size(); ++i) {
        auto& childType = rowType.childAt(i);
        if (i < children.size()) {
          children[i] = liftToType(children[i], childType);
        } else {
          children.push_back(
              BaseVector::createNullConstant(
                  childType, row->size(), config_.pool));
        }
      }
      return std::make_shared<RowVector>(
          config_.pool, type, row->nulls(), row->size(), std::move(children));
    }
    default:
      return input;
  }
}

namespace {
// Returns true when 'name' occurs in 'expr' as a complete identifier token
// rather than as a substring of a longer identifier. Aggregate expressions look
// like "func(name)" and column names are plain identifiers, so this avoids
// false positives such as column "name_4" matching aggregate "min(name_42)".
// Mirrors the word-boundary check in applyRemainingFilters.
bool containsIdentifier(const std::string& expr, const std::string& name) {
  const auto isIdentChar = [](char c) {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
  };
  size_t pos = 0;
  while ((pos = expr.find(name, pos)) != std::string::npos) {
    const bool boundedLeft = pos == 0 || !isIdentChar(expr[pos - 1]);
    const size_t end = pos + name.size();
    const bool boundedRight = end >= expr.size() || !isIdentChar(expr[end]);
    if (boundedLeft && boundedRight) {
      return true;
    }
    pos += name.size();
  }
  return false;
}
} // namespace

bool TableEvolutionFuzzer::isColumnUsedByAggregation(
    const std::string& columnName,
    const AggregationConfig& aggregationConfig) {
  for (const auto& groupingKey : aggregationConfig.groupingKeys) {
    if (groupingKey == columnName) {
      return true;
    }
  }
  for (const auto& aggregate : aggregationConfig.aggregates) {
    if (containsIdentifier(aggregate, columnName)) {
      return true;
    }
  }
  return false;
}

folly::F14FastSet<std::string> TableEvolutionFuzzer::selectFilterOnlyColumns(
    const RowTypePtr& schema,
    const std::unordered_set<std::string>& filteredColumns,
    const std::vector<column_index_t>& bucketColumnIndices,
    const std::vector<RowTypePtr>& perEvolutionSchemas,
    const std::optional<AggregationConfig>& aggregationConfig,
    FuzzerGenerator& rng) {
  folly::F14FastSet<std::string> bucketColumnNames;
  for (const auto bucketColumnIndex : bucketColumnIndices) {
    bucketColumnNames.insert(schema->nameOf(bucketColumnIndex));
  }
  folly::F14FastSet<std::string> droppedColumns;
  for (int i = 0; i < schema->size(); ++i) {
    const auto& name = schema->nameOf(i);
    if (filteredColumns.count(name) == 0) {
      continue;
    }
    // Skip map columns: a filtered map column is read in flatmap-as-struct mode
    // (buildFlatmapAsStructSchema swaps its type to a struct in the output
    // schema) and its filter is a map key/value subfield filter. Dropping it
    // filter-only makes the two oracle plans read it through divergent paths --
    // the pushdown plan pushes the subfield filter onto the struct-read column,
    // while the FilterNode plan realizes the filter as a node and then projects
    // the column away -- so their results can disagree.
    if (schema->childAt(i)->isMap()) {
      continue;
    }
    if (bucketColumnNames.count(name) > 0) {
      continue;
    }
    // Columns evolve positionally; index 'i' in the final schema corresponds to
    // index 'i' in every earlier setup that has that position. Skip columns
    // whose type differs across setups: the selective reader applies coercion
    // for evolved columns differently when a column is filter-only versus also
    // projected, which would make the pushdown-vs-FilterNode oracle diverge.
    bool typeStableAcrossSetups = true;
    for (const auto& evolutionSchema : perEvolutionSchemas) {
      if (i >= evolutionSchema->size() ||
          !evolutionSchema->childAt(i)->equivalent(*schema->childAt(i))) {
        typeStableAcrossSetups = false;
        break;
      }
    }
    if (!typeStableAcrossSetups) {
      continue;
    }
    if (aggregationConfig.has_value() &&
        isColumnUsedByAggregation(name, *aggregationConfig)) {
      continue;
    }
    if (folly::Random::oneIn(2, rng)) {
      droppedColumns.insert(name);
    }
  }
  return droppedColumns;
}

std::vector<std::string> TableEvolutionFuzzer::projectedColumnNames(
    const RowTypePtr& schema,
    const folly::F14FastSet<std::string>& droppedColumns) {
  std::vector<std::string> outputColumnNames;
  for (int i = 0; i < schema->size(); ++i) {
    const auto& name = schema->nameOf(i);
    if (droppedColumns.count(name) == 0) {
      outputColumnNames.push_back(name);
    }
  }
  return outputColumnNames;
}

void TableEvolutionFuzzer::run() {
  ScopedOOMInjector oomInjectorWritePath(
      [this]() -> bool { return folly::Random::oneIn(10, rng_); },
      10); // Check the condition every 10 ms.
  if (FLAGS_enable_oom_injection_write_path) {
    oomInjectorWritePath.enable();
  }

  // Step 1: Randomly decide whether to generate remaining filters (50% chance)
  bool shouldGenerateRemainingFilters = folly::Random::oneIn(2, rng_);

  fuzzer::ExpressionFuzzer::FuzzedExpressionData generatedRemainingFilters;
  std::vector<std::string> additionalColumnNames;
  std::vector<TypePtr> additionalColumnTypes;

  if (shouldGenerateRemainingFilters) {
    // Generate remaining filters and extract new columns
    generatedRemainingFilters = generateRemainingFilters(config_, currentSeed_);

    VLOG(1) << "Generated remaining filters from expression fuzzer: "
            << generatedRemainingFilters.expressions[0]->toString();

    // Extract all columns from generatedRemainingFilters.inputType
    if (generatedRemainingFilters.inputType) {
      for (int i = 0; i < generatedRemainingFilters.inputType->size(); ++i) {
        const auto& columnName = generatedRemainingFilters.inputType->nameOf(i);
        additionalColumnNames.push_back(columnName);
        additionalColumnTypes.push_back(
            generatedRemainingFilters.inputType->childAt(i));
      }
    }

    if (!additionalColumnNames.empty()) {
      VLOG(1)
          << "Found " << additionalColumnNames.size()
          << " columns from generateRemainingFilters, will add to schema evolution";
    }
  } else {
    VLOG(1) << "Skipping remaining filter generation (50% randomization)";
  }

  // Step 2: Test setup and bucketColumnIndices generation with additional
  // columns
  auto bucketColumnIndices = generateBucketColumnIndices();

  // Track column name mappings during evolution
  std::unordered_map<std::string, std::string> columnNameMapping;
  for (const auto& columnName : additionalColumnNames) {
    columnNameMapping[columnName] = columnName; // Initially map to itself
  }

  auto testSetups = makeSetups(
      bucketColumnIndices,
      additionalColumnNames,
      additionalColumnTypes,
      &columnNameMapping);

  // Step 3: Create and execute write tasks
  auto tableOutputRootDir = TempDirectoryPath::create();
  std::vector<std::shared_ptr<TaskCursor>> writeTasks(
      2 * config_.evolutionCount - 1);
  std::vector<RowVectorPtr> finalExpectedBatches;

  folly::F14FastMap<int, folly::F14FastSet<std::string>> globalMapColumnKeys;
  std::vector<int> globallyConsistentColumnIndexVector;

  createWriteTasks(
      testSetups,
      bucketColumnIndices,
      tableOutputRootDir->getPath(),
      writeTasks,
      finalExpectedBatches,
      globalMapColumnKeys,
      globallyConsistentColumnIndexVector);

  auto executor = folly::getGlobalCPUExecutor();
  auto writeResults = runTaskCursors(writeTasks, *executor);

  // Merge the final setup's batches into one vector, once, just before the
  // query-shape loop that uses it to generate subfield filters over every row.
  const RowVectorPtr finalExpectedData =
      fuzzer::mergeRowVectors(finalExpectedBatches, config_.pool);

  // Step 4: Amortize the expensive write by running many query shapes over the
  // same files. Each shape independently draws a fresh query (subfield filters,
  // remaining-filter application, dropped filter-only columns, aggregation
  // config, flatmap-as-struct read schema) and rebuilds the scan splits from
  // the existing write results (no rewrite), then compares the pushdown plan
  // against the FilterNode reference plan over the SAME files.
  for (int shape = 0; shape < kQueryShapesPerFile; ++shape) {
    VLOG(1) << "Running query shape " << shape;
    runQueryShape(
        writeResults,
        testSetups,
        bucketColumnIndices,
        finalExpectedData,
        globalMapColumnKeys,
        globallyConsistentColumnIndexVector,
        shouldGenerateRemainingFilters,
        generatedRemainingFilters,
        columnNameMapping,
        *executor);
  }
}

void TableEvolutionFuzzer::runQueryShape(
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
    folly::Executor& executor) {
  // Rebuild the scan splits per shape: makeScanTask moves them, and a fresh
  // bucket selection lets shapes exercise different buckets. This only
  // reconstructs Split objects from the existing write results; it does not
  // rewrite any files.
  std::optional<int32_t> selectedBucket;
  if (!bucketColumnIndices.empty()) {
    selectedBucket =
        folly::Random::rand32(testSetups.back().bucketCount(), rng_);
    VLOG(1) << "selectedBucket=" << *selectedBucket;
  }

  auto [actualSplits, expectedSplits] = createScanSplitsFromWriteResults(
      writeResults, testSetups, bucketColumnIndices, selectedBucket);

  // Step 5: Setup scan tasks with filters and optional aggregation pushdown
  auto rowType = testSetups.back().schema;
  PushdownConfig pushdownConfig;

  // Generate subfield filters first
  pushdownConfig.subfieldFiltersMap =
      generateSubfieldFilters(rowType, finalExpectedData);

  // Extract field names used by subfield filters to avoid conflicts
  std::unordered_set<std::string> subfieldFilteredFields;
  for (const auto& [subfield, filter] : pushdownConfig.subfieldFiltersMap) {
    auto fieldName = subfield.toString();
    VLOG(1) << "Raw subfield: " << fieldName << ' ' << filter->toString();
    // Extract the root field name (before any nested access)
    const size_t dotPos = fieldName.find('.');
    if (dotPos != std::string::npos) {
      fieldName = fieldName.substr(0, dotPos);
      VLOG(1) << "Subfield filter targets field: " << fieldName;
    }
    subfieldFilteredFields.insert(fieldName);
  }

  // Apply generated remaining filters with updated column names, avoiding
  // conflicts
  if (shouldGenerateRemainingFilters) {
    // Apply generated remaining filters
    applyRemainingFilters(
        generatedRemainingFilters,
        columnNameMapping,
        pushdownConfig,
        subfieldFilteredFields);
  }

  // Collect all filtered columns (both subfield and remaining filters)
  std::unordered_set<std::string> allFilteredColumns = subfieldFilteredFields;

  // Extract columns from remaining filter if present
  if (!pushdownConfig.remainingFilter.empty()) {
    for (const auto& name : rowType->names()) {
      // Check if the column name appears in the remaining filter
      if (pushdownConfig.remainingFilter.find(name) != std::string::npos) {
        allFilteredColumns.insert(name);
      }
    }
  }

  // Enable aggregation testing
  std::optional<AggregationConfig> aggConfig;
  bool shouldTestAggregation =
      folly::Random::oneIn(FLAGS_aggregation_pushdown_frequency, rng_);
  if (shouldTestAggregation) {
    aggConfig = generateAggregationConfig(rowType, rng_, allFilteredColumns);
    if (aggConfig.has_value()) {
      VLOG(1) << "Testing aggregation pushdown with grouping keys: ["
              << folly::join(", ", aggConfig->groupingKeys)
              << "] and aggregates: ["
              << folly::join(", ", aggConfig->aggregates) << "]";
    } else {
      VLOG(1) << "Could not generate valid aggregation configuration";
      aggConfig = std::nullopt;
    }
  }

  // Read a random subset of the filtered columns filter-only (drop them from
  // the scan output while still filtering on them) to exercise the selective
  // reader's filter-only column path.
  std::vector<RowTypePtr> perEvolutionSchemas;
  perEvolutionSchemas.reserve(testSetups.size());
  for (const auto& setup : testSetups) {
    perEvolutionSchemas.emplace_back(setup.schema);
  }
  const folly::F14FastSet<std::string> droppedColumns = selectFilterOnlyColumns(
      rowType,
      allFilteredColumns,
      bucketColumnIndices,
      perEvolutionSchemas,
      aggConfig,
      rng_);
  if (!droppedColumns.empty()) {
    VLOG(1) << "Dropping filter-only columns from scan output: ["
            << folly::join(", ", droppedColumns) << "]";
  }
  const std::vector<std::string> outputColumnNames =
      projectedColumnNames(rowType, droppedColumns);

  // Decide the flatmap-as-struct read schema once and share it across both
  // plans. buildFlatmapAsStructSchema draws an rng coin per compatible map
  // column; computing it separately per plan could make the pushdown and
  // FilterNode plans disagree on a column's read mode (MAP vs struct).
  RowTypePtr fullOutSchema = buildFlatmapAsStructSchema(
      rowType, globalMapColumnKeys, globallyConsistentColumnIndexVector);

  std::vector<std::shared_ptr<TaskCursor>> scanTasks(2);
  // actual: TableScan -> Aggregation (allows pushdown)
  pushdownConfig.aggregationConfig = aggConfig;
  scanTasks[0] = makeScanTask(
      rowType,
      std::move(actualSplits),
      pushdownConfig,
      false,
      false, // insertProjectToBlockPushdown
      fullOutSchema,
      outputColumnNames);

  // expected: TableScan -> Project -> Aggregation (blocks pushdown)
  // Insert a Project node to prevent aggregation pushdown
  pushdownConfig.aggregationConfig = aggConfig;
  scanTasks[1] = makeScanTask(
      rowType,
      std::move(expectedSplits),
      pushdownConfig,
      true,
      true, // insertProjectToBlockPushdown
      fullOutSchema,
      outputColumnNames);

  ScopedOOMInjector oomInjectorReadPath(
      [this]() -> bool { return folly::Random::oneIn(10, rng_); },
      10); // Check the condition every 10 ms.
  if (FLAGS_enable_oom_injection_read_path) {
    oomInjectorReadPath.enable();
  }

  // Step 6: Execute scan tasks and verify results
  auto scanResults = runTaskCursors(scanTasks, executor);

  // Skip result verification when OOM injection is enabled
  if (!FLAGS_enable_oom_injection_write_path &&
      !FLAGS_enable_oom_injection_read_path) {
    checkResultsEqual(scanResults[0], scanResults[1]);
  }
}

int TableEvolutionFuzzer::Setup::bucketCount() const {
  return 1 << log2BucketCount;
}

std::string TableEvolutionFuzzer::makeNewName() {
  return fmt::format("name_{}", ++sequenceNumber_);
}

TypePtr TableEvolutionFuzzer::makeNewType(int maxDepth) {
  // All types that can be written to file directly.
  static const std::vector<TypePtr> scalarTypes = {
      BOOLEAN(),
      TINYINT(),
      SMALLINT(),
      INTEGER(),
      BIGINT(),
      REAL(),
      DOUBLE(),
      VARCHAR(),
      VARBINARY(),
  };
  return vectorFuzzer_.randType(scalarTypes, maxDepth);
}

RowTypePtr TableEvolutionFuzzer::makeInitialSchema(
    const std::vector<std::string>& additionalColumnNames,
    const std::vector<TypePtr>& additionalColumnTypes) {
  std::vector<std::string> names(config_.columnCount);
  std::vector<TypePtr> types(config_.columnCount);
  for (int i = 0; i < config_.columnCount; ++i) {
    names[i] = makeNewName();
    types[i] = makeNewType(3);
  }

  // Add additional columns from generateRemainingFilters
  for (int i = 0; i < additionalColumnNames.size(); ++i) {
    names.push_back(additionalColumnNames[i]);
    types.push_back(additionalColumnTypes[i]);
    VLOG(1) << "Adding additional column to initial schema: "
            << additionalColumnNames[i] << " of type "
            << additionalColumnTypes[i]->toString();
  }

  return ROW(std::move(names), std::move(types));
}

TypePtr TableEvolutionFuzzer::evolveType(const TypePtr& old) {
  switch (old->kind()) {
    case TypeKind::ARRAY:
      return ARRAY(evolveType(old->asArray().elementType()));
    case TypeKind::MAP: {
      auto& mapType = old->asMap();
      return MAP(
          evolveType(mapType.keyType()), evolveType(mapType.valueType()));
    }
    case TypeKind::ROW:
      return evolveRowType(old->asRow(), {});
    default:
      if (!folly::Random::oneIn(4, rng_)) {
        return old;
      }
  }

  switch (old->kind()) {
    case TypeKind::TINYINT:
      return SMALLINT();
    case TypeKind::SMALLINT:
      return INTEGER();
    case TypeKind::INTEGER:
      // Don't evolve DATE type to BIGINT
      if (old->isDate()) {
        return old;
      }
      return BIGINT();
    case TypeKind::REAL:
      return DOUBLE();
    default:
      return old;
  }
}

RowTypePtr TableEvolutionFuzzer::evolveRowType(
    const RowType& old,
    const std::vector<column_index_t>& bucketColumnIndices,
    std::unordered_map<std::string, std::string>* columnNameMapping) {
  auto names = old.names();
  auto types = old.children();
  for (int i = 0, j = 0; i < old.size(); ++i) {
    // Skip evolving bucket column.
    while (j < bucketColumnIndices.size() && bucketColumnIndices[j] < i) {
      ++j;
    }
    if (j < bucketColumnIndices.size() && bucketColumnIndices[j] == i) {
      continue;
    }
    if (folly::Random::oneIn(4, rng_)) {
      auto oldName = names[i];
      auto newName = makeNewName();
      names[i] = newName;

      // Update column name mapping if provided
      if (columnNameMapping) {
        // Find if this column was originally from generateRemainingFilters
        for (auto& [originalName, currentName] : *columnNameMapping) {
          if (currentName == oldName) {
            currentName = newName;
            VLOG(1) << "Updated column name mapping: " << originalName << " -> "
                    << newName;
            break;
          }
        }
      }
    }
    types[i] = evolveType(types[i]);
  }
  if (folly::Random::oneIn(4, rng_)) {
    names.push_back(makeNewName());
    types.push_back(makeNewType(2));
  }
  return ROW(std::move(names), std::move(types));
}

std::vector<TableEvolutionFuzzer::Setup> TableEvolutionFuzzer::makeSetups(
    const std::vector<column_index_t>& bucketColumnIndices,
    const std::vector<std::string>& additionalColumnNames,
    const std::vector<TypePtr>& additionalColumnTypes,
    std::unordered_map<std::string, std::string>* columnNameMapping) {
  std::vector<Setup> setups(config_.evolutionCount);
  for (int i = 0; i < config_.evolutionCount; ++i) {
    if (i == 0) {
      setups[i].schema =
          makeInitialSchema(additionalColumnNames, additionalColumnTypes);
    } else {
      setups[i].schema = evolveRowType(
          *setups[i - 1].schema, bucketColumnIndices, columnNameMapping);
    }
    if (!bucketColumnIndices.empty()) {
      if (i == 0) {
        setups[i].log2BucketCount = folly::Random::rand32(1, 4, rng_);
      } else {
        setups[i].log2BucketCount = std::min<int>(
            8, setups[i - 1].log2BucketCount + folly::Random::rand32(3, rng_));
      }
    } else {
      setups[i].log2BucketCount = 0;
    }
    setups[i].fileFormat =
        config_.formats[folly::Random::rand32(config_.formats.size(), rng_)];
    VLOG(1) << "Setup " << i << ": " << setups[i];
  }
  return setups;
}

std::unique_ptr<TaskCursor> TableEvolutionFuzzer::makeWriteTask(
    const Setup& setup,
    const std::vector<RowVectorPtr>& dataBatches,
    const std::string& outputDir,
    const std::vector<column_index_t>& bucketColumnIndices,
    FuzzerGenerator& rng,
    bool enableFlatMap,
    folly::F14FastMap<int, folly::F14FastSet<std::string>>& globalMapColumnKeys,
    std::vector<int>& globallyCompatibleFlatmapColumns,
    const std::unordered_map<std::string, std::string>& extraSerdeParams) {
  // Emit each batch as its own Values output, so the writer sees multiple
  // write() calls per file. This drives multiple stripes per file and multiple
  // chunks per stripe when chunking is enabled.
  auto builder = PlanBuilder().values(dataBatches);

  // Create serdeParameters using proper dwrf::Config for flatmap configuration
  std::unordered_map<std::string, std::string> serdeParameters;

  if (hasMapColumns(setup.schema)) {
    // Find all top-level map column indices that support flatmap
    std::vector<uint32_t> supportedMapColumnIndices;

    for (int i = 0; i < setup.schema->size(); ++i) {
      if (setup.schema->childAt(i)->isMap()) {
        // Check if this specific map column has any empty elements in any
        // batch. A column is flatmap-compatible only if every batch is free of
        // empty maps.
        // TODO: Coverage gap, not a correctness bug. A present-key/null-value
        // entry (e.g. {k: null}) is NOT excluded here -- it still contributes
        // key k -- yet under flatmap-as-struct read it collapses to the same
        // null child as an absent key. That collapse is symmetric across both
        // compared plans (both read flatmap-as-struct), so it does not diverge
        // the oracle; it only leaves the present-null-vs-absent case
        // unexercised. Empty/null maps are excluded for a different, real
        // reason: an all-empty/all-null column collects zero keys, and
        // buildFlatmapAsStructSchema would then build a zero-field struct that
        // the flatmap-as-struct reader rejects.
        bool anyEmptyElement = false;
        for (const auto& batch : dataBatches) {
          if (hasEmptyElement(batch, i)) {
            anyEmptyElement = true;
            break;
          }
        }
        if (anyEmptyElement) {
          removeFromVector(globallyCompatibleFlatmapColumns, i);
          continue;
        }

        if (!hasUnsupportedMapKey(setup.schema->childAt(i))) {
          // %50 chance to enable flatmap for this map column.
          if (enableFlatMap && folly::Random::oneIn(2, rng)) {
            supportedMapColumnIndices.push_back(static_cast<uint32_t>(i));
            VLOG(1) << "Write column " << setup.schema->nameOf(i)
                    << " as flatmap";

            // Collect keys directly into the global set
            auto& uniqueKeys = globalMapColumnKeys[static_cast<int>(i)];

            // Extract actual keys from every batch's map data and collect
            // directly into the global set.
            for (const auto& batch : dataBatches) {
              SelectivityVector allRows(batch->childAt(i)->size());
              DecodedVector decodedMap(*batch->childAt(i), allRows);
              auto* mapVector = decodedMap.base()->asChecked<MapVector>();
              if (mapVector->size() == 0) {
                continue;
              }
              auto keys = mapVector->mapKeys();
              if (!keys) {
                continue;
              }

              // Iterate through the decoded rows, not the raw mapVector
              // indices
              for (vector_size_t row = 0; row < batch->childAt(i)->size();
                   ++row) {
                auto decodedIndex = decodedMap.index(row);
                if (!decodedMap.isNullAt(row) &&
                    !mapVector->isNullAt(decodedIndex)) {
                  // Get the map entry for this decoded row
                  auto mapOffset = mapVector->offsetAt(decodedIndex);
                  auto mapSize = mapVector->sizeAt(decodedIndex);

                  // Process all keys in this map entry
                  for (vector_size_t keyIdx = 0; keyIdx < mapSize; ++keyIdx) {
                    auto keyPosition = mapOffset + keyIdx;
                    if (!keys->isNullAt(keyPosition)) {
                      std::string keyStr;
                      if (keys->type()->isVarchar() ||
                          keys->type()->isVarbinary()) {
                        auto* keyVector = keys->asFlatVector<StringView>();
                        auto keyView = keyVector->valueAt(keyPosition);
                        keyStr = std::string(keyView);
                      } else if (keys->type()->isInteger()) {
                        auto* keyVector = keys->asFlatVector<int32_t>();
                        auto keyVal = keyVector->valueAt(keyPosition);
                        keyStr = std::to_string(keyVal);
                      } else if (keys->type()->isBigint()) {
                        auto* keyVector = keys->asFlatVector<int64_t>();
                        auto keyVal = keyVector->valueAt(keyPosition);
                        keyStr = std::to_string(keyVal);
                      } else if (keys->type()->isSmallint()) {
                        auto* keyVector = keys->asFlatVector<int16_t>();
                        auto keyVal = keyVector->valueAt(keyPosition);
                        keyStr = std::to_string(keyVal);
                      } else if (keys->type()->isTinyint()) {
                        auto* keyVector = keys->asFlatVector<int8_t>();
                        auto keyVal = keyVector->valueAt(keyPosition);
                        keyStr = std::to_string(keyVal);
                      } else {
                        // This should not be reached since
                        // hasUnsupportedMapKey filters out unsupported types
                        VELOX_UNREACHABLE(
                            "Unsupported map key type: {}",
                            keys->type()->toString());
                      }
                      uniqueKeys.insert(keyStr);
                    }
                  }
                }
              }
            }
          } else {
            // Remove this column from globallyCompatibleFlatmapColumns
            removeFromVector(globallyCompatibleFlatmapColumns, i);
          }
        } else {
          removeFromVector(globallyCompatibleFlatmapColumns, i);
        }
      }
    }

    if (!supportedMapColumnIndices.empty()) {
      auto config = std::make_shared<dwrf::Config>();
      config->set(dwrf::Config::FLATTEN_MAP, true);
      config->set<const std::vector<uint32_t>>(
          dwrf::Config::MAP_FLAT_COLS, supportedMapColumnIndices);

      // Convert to serdeParameters
      auto configParams = config->toSerdeParams();
      serdeParameters.insert(configParams.begin(), configParams.end());
    }
  }

  // Driver-injected, format-specific overrides win over the defaults above.
  for (const auto& [key, value] : extraSerdeParams) {
    serdeParameters.insert_or_assign(key, value);
  }

  if (bucketColumnIndices.empty()) {
    if (!serdeParameters.empty()) {
      builder.tableWrite(
          outputDir,
          /*partitionBy=*/{},
          /*bucketCount=*/0,
          /*bucketedBy=*/{},
          /*sortBy=*/{},
          setup.fileFormat,
          /*aggregates=*/{},
          /*connectorId=*/PlanBuilder::kHiveDefaultConnectorId,
          serdeParameters);
    } else {
      builder.tableWrite(outputDir, setup.fileFormat);
    }
  } else {
    std::vector<std::string> bucketColumnNames;
    bucketColumnNames.reserve(bucketColumnIndices.size());
    for (auto i : bucketColumnIndices) {
      bucketColumnNames.push_back(setup.schema->nameOf(i));
    }
    if (!serdeParameters.empty()) {
      builder.tableWrite(
          outputDir,
          /*partitionBy=*/{},
          setup.bucketCount(),
          bucketColumnNames,
          /*sortBy=*/{},
          setup.fileFormat,
          /*aggregates=*/{},
          /*connectorId=*/PlanBuilder::kHiveDefaultConnectorId,
          serdeParameters);
    } else {
      builder.tableWrite(
          outputDir,
          /*partitionBy=*/{},
          setup.bucketCount(),
          bucketColumnNames,
          /*sortBy=*/{},
          setup.fileFormat);
    }
  }
  CursorParameters params;
  params.serialExecution = true;
  params.planNode = builder.planNode();
  // A bucketed write opens one writer per bucket, and generated setups can
  // reach 2^8 buckets -- past the Hive connector's default 128 open-writer cap.
  // Raise the cap so high-bucket-count setups exercise bucket conversion
  // instead of failing with "Exceeded open writer limit".
  params.queryCtx = core::QueryCtx::create(
      /*executor=*/nullptr,
      core::QueryConfig{{}},
      {{std::string(PlanBuilder::kHiveDefaultConnectorId),
        std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>{
                {"max_partitions_per_writers", "1024"}})}});
  return TaskCursor::create(params);
}

template <typename To, typename From>
VectorPtr TableEvolutionFuzzer::liftToPrimitiveType(
    const FlatVector<From>& input,
    const TypePtr& type) {
  auto targetBuffer = AlignedBuffer::allocate<To>(input.size(), config_.pool);
  auto* rawTargetValues = targetBuffer->template asMutable<To>();
  auto* rawSourceValues = input.rawValues();
  for (vector_size_t i = 0; i < input.size(); ++i) {
    rawTargetValues[i] = rawSourceValues[i];
  }
  return std::make_shared<FlatVector<To>>(
      config_.pool,
      type,
      input.nulls(),
      input.size(),
      std::move(targetBuffer),
      std::vector<BufferPtr>({}));
}

RowTypePtr TableEvolutionFuzzer::buildFlatmapAsStructSchema(
    const RowTypePtr& tableSchema,
    const folly::F14FastMap<int, folly::F14FastSet<std::string>>&
        globalMapColumnKeys,
    const std::vector<int>& globallyCompatibleFlatmapColumns) {
  if (globallyCompatibleFlatmapColumns.empty()) {
    return tableSchema;
  }

  VLOG(1) << "Setting up struct reading for "
          << globallyCompatibleFlatmapColumns.size()
          << " flatmap columns with real keys";

  auto names = tableSchema->names();
  auto types = tableSchema->children();

  // Filter globalMapColumnKeys to only include globally compatible columns
  std::unordered_map<int, folly::F14FastSet<std::string>> filteredMapColumnKeys;
  for (int mapColumnIndex : globallyCompatibleFlatmapColumns) {
    if (globalMapColumnKeys.find(mapColumnIndex) != globalMapColumnKeys.end()) {
      // Add 50% probability to include this column in filteredMapColumnKeys
      if (folly::Random::oneIn(2, rng_)) {
        filteredMapColumnKeys[mapColumnIndex] =
            globalMapColumnKeys.at(mapColumnIndex);
      }
    }
  }

  // Use the filteredMapColumnKeys for struct reading
  for (const auto& [mapColumnIndex, keysSet] : filteredMapColumnKeys) {
    // Convert map type to struct type for struct reading
    auto finalMapType = types[mapColumnIndex]->asMap();
    auto finalValueType = finalMapType.valueType();
    // Convert F14FastSet to vector for ROW constructor
    std::vector<std::string> keys(keysSet.begin(), keysSet.end());
    // Construct struct schema with real keys from write time + final value
    // type
    std::vector<TypePtr> finalStructFieldTypes(keys.size(), finalValueType);
    auto finalStructSchema = ROW(keys, finalStructFieldTypes);

    // Replace the map type with struct type in the schema
    types[mapColumnIndex] = finalStructSchema;
  }

  // Build new schema using struct reading for flatmap columns
  return ROW(names, types);
}

std::unique_ptr<TaskCursor> TableEvolutionFuzzer::makeScanTask(
    const RowTypePtr& tableSchema,
    std::vector<Split> splits,
    const PushdownConfig& pushdownConfig,
    bool useFiltersAsNode,
    bool insertProjectToBlockPushdown,
    const RowTypePtr& fullOutSchema,
    const std::vector<std::string>& outputColumnNames) {
  // 'insertProjectToBlockPushdown' only takes effect on the reference plan,
  // where the Project below is what blocks aggregation pushdown; it is
  // meaningless (and never applied) on the pushdown plan. Enforce the pairing
  // so a caller that sets it without 'useFiltersAsNode' fails loudly instead of
  // silently skipping the intended Project.
  VELOX_CHECK(
      !insertProjectToBlockPushdown || useFiltersAsNode,
      "insertProjectToBlockPushdown requires useFiltersAsNode");
  // Build the projected output schema by keeping only the columns named in
  // 'outputColumnNames', preserving the order of 'fullOutSchema'. An empty
  // 'outputColumnNames' means no pruning, i.e. project the full schema.
  std::unordered_set<std::string> outputColumnNameSet(
      outputColumnNames.begin(), outputColumnNames.end());
  std::vector<std::string> projectedNames;
  std::vector<TypePtr> projectedTypes;
  for (uint32_t i = 0; i < fullOutSchema->size(); ++i) {
    const auto& name = fullOutSchema->nameOf(i);
    if (outputColumnNames.empty() || outputColumnNameSet.count(name) > 0) {
      projectedNames.push_back(name);
      projectedTypes.push_back(fullOutSchema->childAt(i));
    }
  }
  RowTypePtr projectedSchema =
      ROW(std::move(projectedNames), std::move(projectedTypes));
  const bool isPruned = projectedSchema->size() < fullOutSchema->size();

  CursorParameters params;
  params.serialExecution = true;

  PlanBuilder builder;
  builder.filtersAsNode(useFiltersAsNode);
  if (!useFiltersAsNode && isPruned) {
    // Pushdown path with pruning: output only the projected columns while
    // still reading every column so the dropped columns can be used for
    // pushed-down filters. The assignments map covers all columns of the full
    // output schema; the extra columns (not in 'projectedSchema') are read
    // for filters but not output, exercising the filter-only-column path.
    connector::ColumnHandleMap assignments;
    for (uint32_t i = 0; i < fullOutSchema->size(); ++i) {
      const auto& name = fullOutSchema->nameOf(i);
      const auto& type = fullOutSchema->childAt(i);
      assignments.emplace(
          name,
          std::make_shared<connector::hive::HiveColumnHandle>(
              name,
              connector::hive::FileColumnHandle::ColumnType::kRegular,
              type,
              type));
    }
    builder.tableScanWithPushDown(
        projectedSchema,
        /*pushdownConfig=*/pushdownConfig,
        tableSchema, // Original schema as dataColumns
        assignments);
  } else {
    builder.tableScanWithPushDown(
        fullOutSchema, // Use struct schema for flatmap reading
        /*pushdownConfig=*/pushdownConfig,
        tableSchema, // Original schema as dataColumns
        {});
  }

  // Reference path: when filters are realized as a FilterNode, the filtered
  // columns must be output by the scan so the FilterNode can reference them.
  // Add a Project to drop the filter-only columns and emit the same projected
  // schema as the pushdown path. Projecting also blocks aggregation pushdown,
  // so the identity-project case for aggregation is preserved when not pruned.
  if (useFiltersAsNode &&
      (isPruned ||
       (insertProjectToBlockPushdown &&
        pushdownConfig.aggregationConfig.has_value()))) {
    builder.project(
        isPruned ? projectedSchema->names() : fullOutSchema->names());
  }

  // Add aggregation if enabled in pushdown config
  if (pushdownConfig.aggregationConfig.has_value()) {
    builder.singleAggregation(
        pushdownConfig.aggregationConfig->groupingKeys,
        pushdownConfig.aggregationConfig->aggregates);
  }

  params.planNode = builder.planNode();

  auto cursor = TaskCursor::create(params);
  for (auto& split : splits) {
    cursor->task()->addSplit("0", std::move(split));
  }
  cursor->task()->noMoreSplits("0");
  return cursor;
}

std::vector<column_index_t>
TableEvolutionFuzzer::generateBucketColumnIndices() {
  std::vector<column_index_t> bucketColumnIndices;
  for (int i = 0; i < config_.columnCount; ++i) {
    if (folly::Random::oneIn(2 * config_.columnCount, rng_)) {
      bucketColumnIndices.push_back(i);
    }
  }
  VLOG(1) << "bucketColumnIndices: [" << folly::join(", ", bucketColumnIndices)
          << "]";
  return bucketColumnIndices;
}

std::pair<std::vector<Split>, std::vector<Split>>
TableEvolutionFuzzer::createScanSplitsFromWriteResults(
    const std::vector<std::vector<RowVectorPtr>>& writeResults,
    const std::vector<Setup>& testSetups,
    const std::vector<column_index_t>& bucketColumnIndices,
    std::optional<int32_t> selectedBucket) {
  std::vector<Split> actualSplits, expectedSplits;

  for (int i = 0; i < config_.evolutionCount; ++i) {
    auto* result = &writeResults[2 * i];
    buildScanSplitFromTableWriteResult(
        testSetups.back().schema,
        bucketColumnIndices,
        selectedBucket,
        testSetups.back().bucketCount(),
        testSetups[i].bucketCount(),
        testSetups[i].fileFormat,
        *result,
        actualSplits);

    if (i < config_.evolutionCount - 1) {
      result = &writeResults[2 * i + 1];
    }
    buildScanSplitFromTableWriteResult(
        testSetups.back().schema,
        bucketColumnIndices,
        selectedBucket,
        testSetups.back().bucketCount(),
        testSetups.back().bucketCount(),
        testSetups.back().fileFormat,
        *result,
        expectedSplits);
  }

  return {std::move(actualSplits), std::move(expectedSplits)};
}

void TableEvolutionFuzzer::createWriteTasks(
    const std::vector<Setup>& testSetups,
    const std::vector<column_index_t>& bucketColumnIndices,
    const std::string& tableOutputRootDirPath,
    std::vector<std::shared_ptr<TaskCursor>>& writeTasks,
    std::vector<RowVectorPtr>& finalExpectedBatches,
    folly::F14FastMap<int, folly::F14FastSet<std::string>>& globalMapColumnKeys,
    std::vector<int>& globallyConsistentColumnIndexVector) {
  // Initialize globallyConsistentColumnIndexVector with all map column indices
  // from the first schema, then filter out incompatible ones during processing
  if (hasMapColumns(testSetups[0].schema)) {
    for (int j = 0; j < testSetups[0].schema->size(); ++j) {
      if (testSetups[0].schema->childAt(j)->isMap() &&
          !hasUnsupportedMapKey(testSetups[0].schema->childAt(j))) {
        globallyConsistentColumnIndexVector.push_back(j);
      }
    }
  }

  // Generate data and create write tasks in a single loop
  for (int i = 0; i < config_.evolutionCount; ++i) {
    // Write several independently fuzzed batches per file (one writer write()
    // call each) to drive multiple stripes per file and multiple chunks per
    // stripe. A fixed count suffices: chunk and stripe flush randomness already
    // varies the physical layout across files.
    const int numBatches = FLAGS_batches_per_file;
    // Size each batch to a byte target so file/stripe sizes are stable across
    // wildly different schema widths, unless adaptive sizing is disabled via
    // --adaptive_batch_sizing, in which case use a fixed per-batch row count.
    const int vectorSize = FLAGS_adaptive_batch_sizing
        ? computeAdaptiveVectorSize(vectorFuzzer_, testSetups[i].schema)
        : kDefaultVectorSize;
    std::vector<RowVectorPtr> dataBatches;
    dataBatches.reserve(numBatches);
    for (int batch = 0; batch < numBatches; ++batch) {
      auto data =
          vectorFuzzer_.fuzzRow(testSetups[i].schema, vectorSize, false);
      for (auto& child : data->children()) {
        BaseVector::flattenVector(child);
      }
      dataBatches.push_back(std::move(data));
    }

    auto actualDir = fmt::format("{}/actual_{}", tableOutputRootDirPath, i);
    VELOX_CHECK(std::filesystem::create_directory(actualDir));

    // Pass globally consistent columns to restrict flatmap usage
    writeTasks[2 * i] = makeWriteTask(
        testSetups[i],
        dataBatches,
        actualDir,
        bucketColumnIndices,
        rng_,
        true,
        globalMapColumnKeys,
        globallyConsistentColumnIndexVector,
        config_.extraWriteSerdeParams
            ? config_.extraWriteSerdeParams(testSetups[i].fileFormat, rng_)
            : std::unordered_map<std::string, std::string>{});

    if (i == config_.evolutionCount - 1) {
      // The final setup has no separate expected file; its actual file is the
      // oracle. Keep its batches for subfield-filter generation; the caller
      // merges them into one vector just before use.
      finalExpectedBatches = std::move(dataBatches);
      continue;
    }
    auto expectedDir = fmt::format("{}/expected_{}", tableOutputRootDirPath, i);
    VELOX_CHECK(std::filesystem::create_directory(expectedDir));

    // Write the same batches lifted to the final schema, so the expected file
    // holds the identical logical rows as the actual file and the oracle holds.
    std::vector<RowVectorPtr> expectedBatches;
    expectedBatches.reserve(dataBatches.size());
    for (const auto& data : dataBatches) {
      expectedBatches.push_back(
          std::static_pointer_cast<RowVector>(
              liftToType(data, testSetups.back().schema)));
    }

    writeTasks[2 * i + 1] = makeWriteTask(
        testSetups.back(),
        expectedBatches,
        expectedDir,
        bucketColumnIndices,
        rng_,
        true,
        globalMapColumnKeys,
        globallyConsistentColumnIndexVector,
        config_.extraWriteSerdeParams
            ? config_.extraWriteSerdeParams(testSetups.back().fileFormat, rng_)
            : std::unordered_map<std::string, std::string>{});
  }
}

void TableEvolutionFuzzer::applyRemainingFilters(
    const fuzzer::ExpressionFuzzer::FuzzedExpressionData&
        generatedRemainingFilters,
    const std::unordered_map<std::string, std::string>& columnNameMapping,
    PushdownConfig& pushownConfig,
    const std::unordered_set<std::string>& subfieldFilteredFields) {
  if (generatedRemainingFilters.expressions.empty() ||
      columnNameMapping.empty()) {
    return;
  }

  std::vector<std::string> filterStrings;
  for (const auto& expr : generatedRemainingFilters.expressions) {
    auto filterString = expr->toString();
    VLOG(1) << "Processing remaining filter expression: " << filterString;

    // First, update column names in the filter string using columnNameMapping
    for (const auto& [originalName, currentName] : columnNameMapping) {
      // Simple string replacement - this is a basic approach
      // In a more robust implementation, we would parse the expression tree
      size_t pos = 0;
      while ((pos = filterString.find(originalName, pos)) !=
             std::string::npos) {
        // Check if this is a complete word (not part of another identifier)
        bool isCompleteWord = true;
        if (pos > 0 &&
            (std::isalnum(filterString[pos - 1]) ||
             filterString[pos - 1] == '_')) {
          isCompleteWord = false;
        }
        if (pos + originalName.length() < filterString.length() &&
            (std::isalnum(filterString[pos + originalName.length()]) ||
             filterString[pos + originalName.length()] == '_')) {
          isCompleteWord = false;
        }

        if (isCompleteWord) {
          filterString.replace(pos, originalName.length(), currentName);
          pos += currentName.length();
        } else {
          pos += originalName.length();
        }
      }
    }

    VLOG(1) << "After column name mapping: " << filterString;

    // Now check if this filter expression conflicts with subfield filters
    bool hasConflict = false;
    for (const auto& subfieldField : subfieldFilteredFields) {
      // Check if the filter string contains references to fields that are
      // already filtered by subfield filters
      size_t pos = 0;
      while ((pos = filterString.find(subfieldField, pos)) !=
             std::string::npos) {
        // Check if this is a complete word (not part of another identifier)
        bool isCompleteWord = true;
        if (pos > 0 &&
            (std::isalnum(filterString[pos - 1]) ||
             filterString[pos - 1] == '_')) {
          isCompleteWord = false;
        }
        if (pos + subfieldField.length() < filterString.length() &&
            (std::isalnum(filterString[pos + subfieldField.length()]) ||
             filterString[pos + subfieldField.length()] == '_')) {
          isCompleteWord = false;
        }

        if (isCompleteWord) {
          hasConflict = true;
          VLOG(1)
              << "CONFLICT DETECTED! Skipping remaining filter due to conflict with subfield filter on field: "
              << subfieldField << ", filter: " << filterString;
          break;
        }
        pos += subfieldField.length();
      }
      if (hasConflict) {
        break;
      }
    }

    // Skip this filter if it conflicts with subfield filters
    if (hasConflict) {
      VLOG(1) << "Skipping filter due to conflict: " << filterString;
      continue;
    }

    VLOG(1) << "No conflict detected, proceeding with filter: " << filterString;

    // Fix DATE literal format: convert bare date to DATE literal format
    // to prevent DuckDB parser from interpreting it as arithmetic expression
    RE2 datePattern(R"(\b(\d{4}-\d{2}-\d{2})\b)");
    RE2::GlobalReplace(&filterString, datePattern, "DATE '\\1'");

    filterStrings.push_back(filterString);
    VLOG(1) << "Updated filter expression: " << filterString;
  }

  if (filterStrings.size() == 1) {
    pushownConfig.remainingFilter = filterStrings[0];
  } else if (filterStrings.size() > 1) {
    pushownConfig.remainingFilter =
        "(" + folly::join(") AND (", filterStrings) + ")";
  }
}

} // namespace facebook::velox::exec::test
