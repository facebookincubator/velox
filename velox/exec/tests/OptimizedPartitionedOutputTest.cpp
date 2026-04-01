/*
 * Copyright (c) International Business Machines Corporation
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

#include <future>
#include <random>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/memory/ByteStream.h"
#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/OptimizedPartitionedOutput.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/serializers/PrestoSerializer.h"

namespace facebook::velox::exec::test {

/// How null values are distributed in value columns.
enum class NullMode {
  kNoNull, // no null values
  kPartialNull, // row i is null if i % 2 == 0
  kAllNull, // all values are null
};

/// Describes one parameterized test configuration.
struct TestParam {
  /// Short lowercase name used as the gtest parameter suffix.
  std::string name;
  /// Element type for value columns. Ignored when numValueCols == 0.
  TypePtr valueType;
  /// Number of partition-key columns (all INTEGER).
  int numPartitionCols;
  /// Number of value columns of valueType.
  int numValueCols;
  /// Null pattern applied to value columns.
  NullMode nullMode;
};

/// Returns the full set of TestParam combinations:
///   - numValueCols==0: 1 entry per numPartitionCols (type/nullMode irrelevant)
///   - numValueCols∈{1,256}: all 4 types × 2 pk counts × 3 null modes
std::vector<TestParam> testParams() {
  std::vector<TestParam> params;

  const std::vector<std::pair<std::string, TypePtr>> types = {
      {"bool", BOOLEAN()},
      {"tinyint", TINYINT()},
      {"bigint", BIGINT()},
      {"hugeint", HUGEINT()},
  };

  const std::vector<std::pair<std::string, NullMode>> nullModes = {
      {"no_null", NullMode::kNoNull},
      {"partial_null", NullMode::kPartialNull},
      {"all_null", NullMode::kAllNull},
  };

  // Zero value columns: type and null mode do not affect test behavior.
  for (int numPk : {1, 4}) {
    params.push_back({
        .name = "pk" + std::to_string(numPk) + "_val0",
        .valueType = BIGINT(),
        .numPartitionCols = numPk,
        .numValueCols = 0,
        .nullMode = NullMode::kNoNull,
    });
  }

  // One and many value columns: all type × pk-count × null-mode combinations.
  for (int numVal : {1, 256}) {
    for (const auto& [typeName, type] : types) {
      for (int numPk : {1, 4}) {
        for (const auto& [nullName, nullMode] : nullModes) {
          params.push_back({
              .name = "pk" + std::to_string(numPk) + "_val" +
                  std::to_string(numVal) + "_" + typeName + "_" + nullName,
              .valueType = type,
              .numPartitionCols = numPk,
              .numValueCols = numVal,
              .nullMode = nullMode,
          });
        }
      }
    }
  }

  return params;
}

/// Collected output from a single run of runPartitionedOutput().
struct PartitionedOutputResult {
  // Declared first so it is destroyed last: the IOBufs in pages reference the
  // task's memory pool, so the task must outlive all the pages.
  std::shared_ptr<Task> task;
  /// Serialized output pages per partition, indexed by partition ID.
  std::vector<std::vector<std::unique_ptr<folly::IOBuf>>> pages;
  /// Number of pages received by each partition.
  std::vector<size_t> pageCounts;
  /// Total rows deserialized from each partition's pages.
  std::vector<int64_t> rowCounts;
  /// Number of partitions that received at least one page.
  int numNonEmptyPartitions{0};
  /// Sum of operator's numAppends runtime stat.
  int64_t numAppends{0};
  /// Sum of operator's numFlushes runtime stat.
  int64_t numFlushes{0};
  /// Sum of operator's numBlockedTimes runtime stat.
  int64_t numBlockedTimes{0};
};

/// Shared infrastructure for all OptimizedPartitionedOutput tests.
class OptimizedPartitionedOutputTest : public OperatorTestBase {
 protected:
  std::shared_ptr<core::QueryCtx> createQueryContext(
      std::unordered_map<std::string, std::string> config) {
    config[core::QueryConfig::kOptimizedPartitionedOutputEnabled] = "true";
    return core::QueryCtx::create(
        executor_.get(), core::QueryConfig(std::move(config)));
  }

  /// Fetches one batch of serialized pages from the output buffer for the given
  /// destination. Returns the pages via a promise/future callback.
  std::vector<std::unique_ptr<folly::IOBuf>>
  getData(const std::string& taskId, int destination, int64_t sequence) {
    auto [promise, semiFuture] = folly::makePromiseContract<
        std::vector<std::unique_ptr<folly::IOBuf>>>();
    VELOX_CHECK(bufferManager_->getData(
        taskId,
        destination,
        OptimizedPartitionedOutput::kMinDestinationSize,
        sequence,
        [result = std::make_shared<
             folly::Promise<std::vector<std::unique_ptr<folly::IOBuf>>>>(
             std::move(promise))](
            std::vector<std::unique_ptr<folly::IOBuf>> pages,
            int64_t /*sequence*/,
            std::vector<int64_t> /*remainingBytes*/) {
          result->setValue(std::move(pages));
        }));
    auto future = std::move(semiFuture).via(executor_.get());
    future.wait(std::chrono::seconds{10});
    VELOX_CHECK(future.isReady());
    return std::move(future).value();
  }

  /// Drains all pages for a destination until the null sentinel is received.
  std::vector<std::unique_ptr<folly::IOBuf>> getAllData(
      const std::string& taskId,
      int destination) {
    std::vector<std::unique_ptr<folly::IOBuf>> result;
    int attempts = 0;
    bool done = false;
    while (!done) {
      VELOX_CHECK_LT(++attempts, 10'000);
      auto pages = getData(taskId, destination, result.size());
      for (auto& page : pages) {
        if (page) {
          result.push_back(std::move(page));
        } else {
          bufferManager_->deleteResults(taskId, destination);
          done = true;
          break;
        }
      }
    }
    return result;
  }

  /// Deserializes a single Presto-serialized IOBuf page into a RowVector.
  RowVectorPtr deserializePage(
      const folly::IOBuf* iobuf,
      const RowTypePtr& rowType) {
    auto byteRanges = byteRangesFromIOBuf(const_cast<folly::IOBuf*>(iobuf));
    auto byteStream =
        std::make_unique<BufferInputStream>(std::move(byteRanges));
    serializer::presto::PrestoVectorSerde serde;
    RowVectorPtr result;
    serde.deserialize(byteStream.get(), pool(), rowType, &result, 0, nullptr);
    return result;
  }

  /// Deserializes and concatenates all pages for one partition into a single
  /// RowVector. Returns an empty RowVector when pages is empty.
  RowVectorPtr concatPages(
      const std::vector<std::unique_ptr<folly::IOBuf>>& pages,
      const RowTypePtr& rowType) {
    RowVectorPtr result;
    for (const auto& iobuf : pages) {
      auto page = deserializePage(iobuf.get(), rowType);
      if (!result) {
        result = page;
      } else {
        result->append(page.get());
      }
    }
    if (!result) {
      result = std::static_pointer_cast<RowVector>(
          BaseVector::create(rowType, 0, pool()));
    }
    return result;
  }

  int64_t getIntRuntimeStat(Task* task, const std::string& statName) {
    const auto taskStats = task->taskStats();
    const auto& runtimeStats =
        taskStats.pipelineStats[0].operatorStats.back().runtimeStats;
    auto it = runtimeStats.find(statName);
    return it != runtimeStats.end() ? it->second.sum : 0;
  }

  /// Builds a plan from inputBatches, creates and starts a task, drains all
  /// numPartitions destinations concurrently, waits for task completion, and
  /// returns the collected pages, per-partition row counts, and operator
  /// runtime stats. extraConfig is merged into the query config on top of the
  /// OptimizedPartitionedOutput enable flag.
  PartitionedOutputResult runPartitionedOutput(
      const std::string& taskId,
      const std::vector<RowVectorPtr>& inputBatches,
      const std::vector<std::string>& partitionKeys,
      int numPartitions,
      std::unordered_map<std::string, std::string> extraConfig = {},
      std::chrono::seconds timeout = std::chrono::seconds{30}) {
    VELOX_CHECK(!inputBatches.empty());
    const auto rowType =
        std::dynamic_pointer_cast<const RowType>(inputBatches[0]->type());

    auto plan = PlanBuilder()
                    .values(inputBatches)
                    .partitionedOutput(partitionKeys, numPartitions)
                    .planNode();

    auto task = Task::create(
        taskId,
        core::PlanFragment{plan},
        0,
        createQueryContext(std::move(extraConfig)),
        Task::ExecutionMode::kParallel);
    task->start(1);

    // Drain all partitions concurrently to avoid deadlock with the driver.
    std::vector<std::future<std::vector<std::unique_ptr<folly::IOBuf>>>>
        futures;
    futures.reserve(numPartitions);
    for (int p = 0; p < numPartitions; ++p) {
      futures.push_back(std::async(std::launch::async, [&, p] {
        return getAllData(taskId, p);
      }));
    }

    const auto taskWaitUs =
        std::chrono::duration_cast<std::chrono::microseconds>(timeout).count();
    EXPECT_TRUE(waitForTaskCompletion(task.get(), taskWaitUs));

    PartitionedOutputResult result;
    result.pages.resize(numPartitions);
    result.pageCounts.resize(numPartitions, 0);
    result.rowCounts.resize(numPartitions, 0);

    for (int p = 0; p < numPartitions; ++p) {
      result.pages[p] = futures[p].get();
      result.pageCounts[p] = result.pages[p].size();
      if (result.pageCounts[p] > 0) {
        ++result.numNonEmptyPartitions;
      }
      result.rowCounts[p] = concatPages(result.pages[p], rowType)->size();
    }

    result.numAppends = getIntRuntimeStat(task.get(), "numAppends");
    result.numFlushes = getIntRuntimeStat(task.get(), "numFlushes");
    result.numBlockedTimes = getIntRuntimeStat(task.get(), "numBlockedTimes");
    result.task = task;

    return result;
  }

 private:
  const std::shared_ptr<OutputBufferManager> bufferManager_{
      OutputBufferManager::getInstanceRef()};
};

// ─── Parameterized fixture ───────────────────────────────────────────────────

/// Parameterized fixture that exercises every TestParam combination.
class OptimizedPartitionedOutputParamTest
    : public OptimizedPartitionedOutputTest,
      public ::testing::WithParamInterface<TestParam> {
 protected:
  const TestParam& param() const {
    return GetParam();
  }

  /// Names for pk columns: ["p1"] or ["p1","p2","p3","p4"].
  std::vector<std::string> pkColNames() const {
    std::vector<std::string> names;
    for (int i = 0; i < param().numPartitionCols; ++i) {
      names.push_back("p" + std::to_string(i + 1));
    }
    return names;
  }

  /// Names for value columns: ["v0", ..., "v{N-1}"].
  std::vector<std::string> valueColNames() const {
    std::vector<std::string> names;
    for (int i = 0; i < param().numValueCols; ++i) {
      names.push_back("v" + std::to_string(i));
    }
    return names;
  }

  /// Full input ROW type: pk cols (INTEGER) followed by value cols.
  RowTypePtr inputType() const {
    std::vector<std::string> names = pkColNames();
    std::vector<TypePtr> types(param().numPartitionCols, INTEGER());
    for (const auto& name : valueColNames()) {
      names.push_back(name);
      types.push_back(param().valueType);
    }
    return ROW(std::move(names), std::move(types));
  }

  /// Channel indices of the pk columns within the input type.
  std::vector<column_index_t> pkChannels() const {
    std::vector<column_index_t> channels(param().numPartitionCols);
    std::iota(channels.begin(), channels.end(), 0);
    return channels;
  }

  /// Returns true if row i should be null in value columns for the current
  /// null mode.
  bool isNull(int rowIdx) const {
    switch (param().nullMode) {
      case NullMode::kNoNull:
        return false;
      case NullMode::kAllNull:
        return true;
      case NullMode::kPartialNull:
        return rowIdx % 2 == 0;
    }
    VELOX_UNREACHABLE();
  }

  /// Creates a flat vector of the param's value type with random values and
  /// nulls applied according to nullMode.
  VectorPtr makeRandomValueVector(int numRows, std::mt19937_64& rng) {
    auto isNullFn = [this](vector_size_t i) -> bool { return isNull(i); };

    switch (param().valueType->kind()) {
      case TypeKind::BOOLEAN:
        return vectorMaker_.flatVector<bool>(
            numRows,
            [&](auto /*i*/) -> bool { return rng() % 2 == 0; },
            isNullFn);
      case TypeKind::TINYINT:
        return vectorMaker_.flatVector<int8_t>(
            numRows,
            [&](auto /*i*/) -> int8_t { return static_cast<int8_t>(rng()); },
            isNullFn);
      case TypeKind::BIGINT:
        return vectorMaker_.flatVector<int64_t>(
            numRows,
            [&](auto /*i*/) -> int64_t { return static_cast<int64_t>(rng()); },
            isNullFn);
      case TypeKind::HUGEINT:
        return vectorMaker_.flatVector<int128_t>(
            numRows,
            [&](auto /*i*/) -> int128_t {
              int64_t hi = static_cast<int64_t>(rng());
              uint64_t lo = rng();
              return (static_cast<int128_t>(hi) << 64) |
                  static_cast<int128_t>(lo);
            },
            isNullFn);
      default:
        VELOX_UNREACHABLE(
            "Unsupported value type: {}", param().valueType->toString());
    }
  }

  /// Builds one input RowVector. p0Values holds the first pk column; each
  /// subsequent pk column i is p0 + i. Value columns are filled with
  /// independent random data drawn from rng.
  RowVectorPtr makeInputBatch(
      const std::vector<int32_t>& p0Values,
      std::mt19937_64& rng) {
    const int numRows = p0Values.size();
    std::vector<std::string> names;
    std::vector<VectorPtr> vecs;

    // pk columns
    for (int k = 0; k < param().numPartitionCols; ++k) {
      names.push_back("p" + std::to_string(k + 1));
      vecs.push_back(vectorMaker_.flatVector<int32_t>(
          numRows, [&, k](auto i) { return p0Values[i] + k; }));
    }

    // value columns
    for (int v = 0; v < param().numValueCols; ++v) {
      names.push_back("v" + std::to_string(v));
      vecs.push_back(makeRandomValueVector(numRows, rng));
    }

    return makeRowVector(names, vecs);
  }

  /// Sorts a vector by value for order-independent comparison. Returns a
  /// dictionary vector with rows sorted in ascending order.
  VectorPtr canonicalize(const VectorPtr& vector) {
    const auto numRows = vector->size();
    auto indices = makeIndices(numRows, [](auto i) { return i; });
    auto* data = indices->asMutable<vector_size_t>();
    std::stable_sort(data, data + numRows, [&](auto a, auto b) {
      return vector->compare(vector.get(), a, b) < 0;
    });
    return BaseVector::wrapInDictionary(nullptr, indices, numRows, vector);
  }

  /// Builds a RowVector by gathering rows from inputBatches at the given
  /// (batchIdx, rowIdx) positions. Used to construct the per-partition expected
  /// RowVector.
  RowVectorPtr gatherRows(
      const std::vector<RowVectorPtr>& batches,
      const std::vector<std::pair<int, int>>& rowList,
      const RowTypePtr& rowType) {
    const auto numRows = static_cast<vector_size_t>(rowList.size());
    auto result = std::static_pointer_cast<RowVector>(
        BaseVector::create(rowType, numRows, pool()));
    for (vector_size_t r = 0; r < numRows; ++r) {
      result->copy(batches[rowList[r].first].get(), r, rowList[r].second, 1);
    }
    return result;
  }

  /// Verifies that the deserialized pages for each partition exactly match the
  /// rows from inputBatches that were routed to that partition. Both expected
  /// and actual rows are sorted (canonicalized) before comparison to allow
  /// order-independent matching.
  void verifyDataIntegrity(
      const std::vector<RowVectorPtr>& inputBatches,
      const std::vector<std::vector<std::unique_ptr<folly::IOBuf>>>& allPages,
      int numPartitions) {
    // Compute expected per-partition row list using the same hash function as
    // the operator.
    auto partitionFn = std::make_unique<HashPartitionFunction>(
        false, numPartitions, inputType(), pkChannels());

    std::vector<std::vector<std::pair<int, int>>> expectedRows(numPartitions);
    for (int batchIdx = 0; batchIdx < static_cast<int>(inputBatches.size());
         ++batchIdx) {
      std::vector<uint32_t> assignments(inputBatches[batchIdx]->size());
      partitionFn->partition(*inputBatches[batchIdx], assignments);
      for (int rowIdx = 0; rowIdx < static_cast<int>(assignments.size());
           ++rowIdx) {
        expectedRows[assignments[rowIdx]].emplace_back(batchIdx, rowIdx);
      }
    }

    const auto rowType = inputType();
    int64_t totalRows = 0;

    for (int p = 0; p < numPartitions; ++p) {
      auto expected = gatherRows(inputBatches, expectedRows[p], rowType);
      auto actual = concatPages(allPages[p], rowType);

      totalRows += expected->size();
      ASSERT_EQ(expected->size(), actual->size())
          << "partition " << p << " row count mismatch";

      // Sort both vectors before comparing to allow order-independent matching.
      auto expectedSorted = canonicalize(expected);
      auto actualSorted = canonicalize(actual);
      velox::test::assertEqualVectors(expectedSorted, actualSorted);
    }

    int64_t sentRows = 0;
    for (const auto& batch : inputBatches) {
      sentRows += batch->size();
    }
    EXPECT_EQ(totalRows, sentRows);
  }
};

// ─── singleFlush ─────────────────────────────────────────────────────────────

// Sends one batch into a large-buffer operator. All data is buffered without
// triggering an intermediate flush; the final noMoreInput flush serializes
// everything once. Verifies numFlushes==1, numBlockedTimes==0, and that every
// deserialized row matches its source.
TEST_P(OptimizedPartitionedOutputParamTest, singleFlush) {
  constexpr int kNumPartitions = 4;
  // One row per partition key, so every partition gets data.
  std::vector<int32_t> p0Values;
  for (int i = 0; i < kNumPartitions; ++i) {
    p0Values.push_back(i);
  }

  std::mt19937_64 rng(42);
  const std::vector<RowVectorPtr> inputBatches = {
      makeInputBatch(p0Values, rng)};

  auto result = runPartitionedOutput(
      "local://test-single-flush-" + param().name,
      inputBatches,
      pkColNames(),
      kNumPartitions);

  verifyDataIntegrity(inputBatches, result.pages, kNumPartitions);
  EXPECT_EQ(result.numAppends, 1);
  EXPECT_EQ(result.numFlushes, 1);
  EXPECT_EQ(result.numBlockedTimes, 0);
}

// ─── multipleFlushes ─────────────────────────────────────────────────────────

// Sends multiple batches through a 1-byte serializer ceiling so each addInput
// triggers its own flush. A 10-byte OutputBuffer ceiling forces blocking.
// Concurrent consumers drain each partition so the driver can unblock.
// Verifies numFlushes==kBatches, numBlockedTimes>=1, and full data integrity.
TEST_P(OptimizedPartitionedOutputParamTest, multipleFlushes) {
  constexpr int kNumPartitions = 4;
  constexpr int kBatches = 10;

  // For wide schemas, reduce rows per batch so each batch stays small.
  const int kRowsPerBatch = param().numValueCols >= 64 ? 2 : kNumPartitions;

  std::vector<int32_t> p0Values(kRowsPerBatch);
  for (int i = 0; i < kRowsPerBatch; ++i) {
    p0Values[i] = i % kNumPartitions;
  }
  std::mt19937_64 rng(42);
  std::vector<RowVectorPtr> inputBatches;
  inputBatches.reserve(kBatches);
  for (int b = 0; b < kBatches; ++b) {
    inputBatches.push_back(makeInputBatch(p0Values, rng));
  }

  auto result = runPartitionedOutput(
      "local://test-multiple-flushes-" + param().name,
      inputBatches,
      pkColNames(),
      kNumPartitions,
      // 1-byte serializer ceiling flushes before every addInput.
      // 10-byte OutputBuffer ceiling forces blocking on every enqueue.
      {{core::QueryConfig::kMaxPartitionedOutputBufferSize, "1"},
       {core::QueryConfig::kMaxOutputBufferSize, "10"}},
      std::chrono::seconds{30});

  verifyDataIntegrity(inputBatches, result.pages, kNumPartitions);
  EXPECT_EQ(result.numAppends, kBatches);
  EXPECT_EQ(result.numFlushes, kBatches);
  EXPECT_EQ(result.numBlockedTimes, kBatches);
}

// ─── uniformDistribution ─────────────────────────────────────────────────────

// Sends many batches with p1 cycling through all partition keys so every
// partition receives rows. Uses the default buffer size (no intermediate
// flush). Verifies that all partitions are non-empty and that data integrity
// holds across all rows.
TEST_P(OptimizedPartitionedOutputParamTest, uniformDistribution) {
  constexpr int kNumPartitions = 4;
  constexpr int kBatches = 10;

  std::mt19937_64 rng(123);
  // Use enough distinct p1 values across a wide range so all partitions receive
  // rows regardless of how the hash distributes them. With 50 distinct p1
  // values and 4 partitions the probability of any partition being empty is <
  // 1e-6.
  constexpr int kRowsPerBatch = 50;
  std::uniform_int_distribution<int32_t> dist(0, 999);

  std::vector<RowVectorPtr> inputBatches;
  inputBatches.reserve(kBatches);
  for (int b = 0; b < kBatches; ++b) {
    std::vector<int32_t> p0Values(kRowsPerBatch);
    for (auto& v : p0Values) {
      v = dist(rng);
    }
    inputBatches.push_back(makeInputBatch(p0Values, rng));
  }

  auto result = runPartitionedOutput(
      "local://test-uniform-" + param().name,
      inputBatches,
      pkColNames(),
      kNumPartitions);

  verifyDataIntegrity(inputBatches, result.pages, kNumPartitions);

  // With 50 distinct p1 values per batch and 4 partitions, every partition must
  // receive rows (probability of any bucket being empty is < 1e-6).
  EXPECT_EQ(result.numNonEmptyPartitions, kNumPartitions);
}

// ─── skewed distributions
// ──────────────────────────────────────────────────────

// Sends batches with 6 distinct key values whose frequencies decrease by
// roughly 2x per step, so non-empty partitions end up with very different row
// counts. Because 6 < 8 some partitions stay empty; because 6 > 8/2 most
// partitions receive rows. This sits between uniformDistribution (all full)
// and skewedDistribution (at most 2 of 64 filled).
TEST_P(OptimizedPartitionedOutputParamTest, moderateSkew) {
  constexpr int kNumPartitions = 8;
  constexpr int kBatches = 5;

  // Key i appears 2^(5-i) times per batch: key 0 → 32 rows, key 1 → 16,
  // key 2 → 8, key 3 → 4, key 4 → 2, key 5 → 1. Total: 63 rows per batch.
  std::vector<int32_t> keyPattern;
  for (int key = 0; key < 6; ++key) {
    const int count = 1 << (5 - key); // 32, 16, 8, 4, 2, 1
    for (int j = 0; j < count; ++j) {
      keyPattern.push_back(key);
    }
  }

  std::mt19937_64 rng(55);
  std::vector<RowVectorPtr> inputBatches;
  inputBatches.reserve(kBatches);
  for (int b = 0; b < kBatches; ++b) {
    auto p0Values = keyPattern;
    std::shuffle(p0Values.begin(), p0Values.end(), rng);
    inputBatches.push_back(makeInputBatch(p0Values, rng));
  }

  auto result = runPartitionedOutput(
      "local://test-moderate-skew-" + param().name,
      inputBatches,
      pkColNames(),
      kNumPartitions);

  verifyDataIntegrity(inputBatches, result.pages, kNumPartitions);

  // 6 distinct keys → at most 6 non-empty partitions; 6 < 8 → at least one
  // empty partition.
  EXPECT_LE(result.numNonEmptyPartitions, 6);

  // Verify a wide spread in per-partition row counts: the heaviest non-empty
  // partition must have at least 2x the average non-empty partition size.
  // This remains stable even when several low-frequency keys hash to the same
  // bucket, unlike a comparison against the minimum non-empty partition.
  int64_t maxRows = 0;
  int64_t totalNonZeroRows = 0;
  int64_t numNonZeroPartitions = 0;
  for (int p = 0; p < kNumPartitions; ++p) {
    if (result.rowCounts[p] > 0) {
      maxRows = std::max(maxRows, result.rowCounts[p]);
      totalNonZeroRows += result.rowCounts[p];
      ++numNonZeroPartitions;
    }
  }
  ASSERT_GT(numNonZeroPartitions, 0);
  EXPECT_GE(maxRows * numNonZeroPartitions, totalNonZeroRows * 2);
}

// Sends many batches with p1 restricted to {0, 1} into a 64-partition
// operator. At most 2 of the 64 partitions will receive any rows; the rest
// must be empty. Verifies data integrity and the empty-partition invariant.
TEST_P(OptimizedPartitionedOutputParamTest, twoDestinations) {
  constexpr int kNumPartitions = 64;
  constexpr int kBatches = 10;
  constexpr int kRowsPerBatch = 4;

  std::mt19937_64 rng(7);
  std::vector<RowVectorPtr> inputBatches;
  inputBatches.reserve(kBatches);
  for (int b = 0; b < kBatches; ++b) {
    // p1 only takes values 0 and 1; at most 2 of 64 partitions receive rows.
    std::vector<int32_t> p0Values(kRowsPerBatch);
    for (int i = 0; i < kRowsPerBatch; ++i) {
      p0Values[i] = i % 2;
    }
    inputBatches.push_back(makeInputBatch(p0Values, rng));
  }

  auto result = runPartitionedOutput(
      "local://test-skewed-" + param().name,
      inputBatches,
      pkColNames(),
      kNumPartitions);

  verifyDataIntegrity(inputBatches, result.pages, kNumPartitions);

  // p1 ∈ {0, 1}: at most 2 distinct hash buckets receive rows.
  EXPECT_LE(result.numNonEmptyPartitions, 2);
  EXPECT_GE(result.numNonEmptyPartitions, 1);
}

// Sends multiple batches where every row carries the same partition key value
// so all rows hash to a single destination. Verifies that exactly one partition
// receives all rows and the remaining partitions stay empty.
TEST_P(OptimizedPartitionedOutputParamTest, singleDestination) {
  constexpr int kNumPartitions = 8;
  constexpr int kBatches = 5;
  constexpr int kRowsPerBatch = 10;

  // Every row has p1=0 (p2=1, p3=2, p4=3 for multi-pk params), so the hash is
  // identical for every row and all rows land in one partition.
  std::mt19937_64 rng(99);
  std::vector<RowVectorPtr> inputBatches;
  inputBatches.reserve(kBatches);
  for (int b = 0; b < kBatches; ++b) {
    inputBatches.push_back(
        makeInputBatch(std::vector<int32_t>(kRowsPerBatch, 0), rng));
  }

  auto result = runPartitionedOutput(
      "local://test-single-dest-" + param().name,
      inputBatches,
      pkColNames(),
      kNumPartitions);

  verifyDataIntegrity(inputBatches, result.pages, kNumPartitions);

  // All rows must land in exactly one partition.
  EXPECT_EQ(result.numNonEmptyPartitions, 1);

  // That one partition must hold every row from every batch.
  const int64_t totalInputRows = static_cast<int64_t>(kBatches) * kRowsPerBatch;
  for (int p = 0; p < kNumPartitions; ++p) {
    if (result.rowCounts[p] > 0) {
      EXPECT_EQ(result.rowCounts[p], totalInputRows) << "partition " << p;
    }
  }
}

// ─── instantiation ───────────────────────────────────────────────────────────

INSTANTIATE_TEST_SUITE_P(
    Params,
    OptimizedPartitionedOutputParamTest,
    ::testing::ValuesIn(testParams()),
    [](const ::testing::TestParamInfo<TestParam>& info) {
      return info.param.name;
    });

// ─── non-parameterized tests ─────────────────────────────────────────────────

// Verifies that replicateNullsAndAny raises an error since it is not yet
// supported by OptimizedPartitionedOutput.
TEST_F(OptimizedPartitionedOutputTest, replicateNullsAndAnyUnsupported) {
  auto input = makeRowVector(
      {"p1", "v1"},
      {makeNullableFlatVector<int32_t>({0, std::nullopt, 1}),
       makeFlatVector<std::string>({"a", "b", "c"})});

  auto plan =
      PlanBuilder()
          .values({input})
          .partitionedOutput({"p1"}, 2, /*replicateNullsAndAny=*/true, {"v1"})
          .planNode();

  auto taskId = "local://test-replicate-nulls-unsupported-0";
  auto task = Task::create(
      taskId,
      core::PlanFragment{plan},
      0,
      createQueryContext({}),
      Task::ExecutionMode::kParallel);
  task->start(1);

  const auto taskWaitUs = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::seconds{10})
                              .count();
  ASSERT_TRUE(waitForTaskFailure(task.get(), taskWaitUs));
  ASSERT_THAT(
      task->errorMessage(),
      testing::HasSubstr(
          "replicateNullsAndAny is not yet supported by OptimizedPartitionedOutput"));
}

} // namespace facebook::velox::exec::test
