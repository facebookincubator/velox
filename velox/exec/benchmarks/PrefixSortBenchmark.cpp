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

#include "glog/logging.h"
#include "velox/exec/PrefixSort.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {

class TestCase {
 public:
  TestCase(
      memory::MemoryPool* pool,
      const std::string& testName,
      size_t numRows,
      const RowTypePtr& rowType)
      : testName_(testName), numRows_(numRows) {
    data_ = std::make_unique<RowContainer>(rowType->children(), pool);
    RowVectorPtr sortedRows = fuzzRows(pool, numRows, rowType);
    storeRows(numRows, sortedRows, data_.get(), rows_);
  };

  const std::string& testName() const {
    return testName_;
  }

  size_t numRows() const {
    return numRows_;
  }

  const std::vector<char*>& rows() const {
    return rows_;
  }

  RowContainer* rowContainer() const {
    return data_.get();
  }

 private:
  // Store data into a RowContainer to mock the behavior of SortBuffer.
  void storeRows(
      int numRows,
      const RowVectorPtr& data,
      RowContainer* rowContainer,
      std::vector<char*>& rows) {
    rows.resize(numRows);
    for (int row = 0; row < numRows; ++row) {
      rows[row] = rowContainer->newRow();
    }
    for (int column = 0; column < data->childrenSize(); ++column) {
      DecodedVector decoded(*data->childAt(column));
      for (int i = 0; i < numRows; ++i) {
        char* row = rows[i];
        rowContainer->store(decoded, i, row, column);
      }
    }
  }

  // Fuzz test rows : for front columns (column 0 to size -2) use high
  // nullRatio to enforce all columns to be compared.
  RowVectorPtr fuzzRows(
      memory::MemoryPool* pool,
      size_t numRows,
      const RowTypePtr& rowType) {
    VectorFuzzer fuzzer({.vectorSize = numRows}, pool);
    VectorFuzzer fuzzerWithNulls(
        {.vectorSize = numRows, .nullRatio = 0.7}, pool);
    if (rowType->children().size() == 1) {
      return fuzzer.fuzzRow(rowType);
    }
    std::vector<VectorPtr> children;
    for (int column = 0; column < rowType->children().size() - 1; ++column) {
      children.push_back(fuzzerWithNulls.fuzz(rowType->childAt(column)));
    }
    children.push_back(fuzzer.fuzz(rowType->children().back()));
    return std::make_shared<RowVector>(
        pool, rowType, nullptr, numRows, std::move(children));
  }

  const std::string testName_;
  const size_t numRows_;
  // Rows address stored in RowContainer
  std::vector<char*> rows_;
  std::unique_ptr<RowContainer> data_;
};

class PrefixSortBenchmark {
 public:
  PrefixSortBenchmark(memory::MemoryPool* pool) : pool_(pool) {}

  void runPrefixSort(
      const std::vector<char*>& rows,
      RowContainer* rowContainer,
      const std::vector<CompareFlags>& compareFlags) {
    PrefixSortConfig prefixSortConfig(1024);
    PrefixSort prefixSort(pool_, rowContainer, compareFlags, prefixSortConfig);
    // Copy rows to sortedRows to avoid sort rows already sorted.
    std::vector<char*> sortedRows = rows;
    prefixSort.sort(sortedRows);
  }

  void runStdSort(
      const std::vector<char*>& rows,
      RowContainer* rowContainer,
      const std::vector<CompareFlags>& compareFlags) {
    std::vector<char*> sortedRows = rows;
    std::sort(
        sortedRows.begin(),
        sortedRows.end(),
        [&](const char* leftRow, const char* rightRow) {
          for (vector_size_t index = 0; index < compareFlags.size(); ++index) {
            if (auto result = rowContainer->compare(
                    leftRow, rightRow, index, compareFlags[index])) {
              return result < 0;
            }
          }
          return false;
        });
  }

  // Add benchmark manually to avoid writing a lot of BENCHMARK.
  void benchmark(
      const std::string& testName,
      size_t numRows,
      const RowTypePtr& rowType,
      int iterateTimes = 1) {
    // CompareFlags could be same in benchmark.
    std::vector<CompareFlags> compareFlags;
    for (int i = 0; i < rowType->children().size(); ++i) {
      compareFlags.push_back(
          {true, true, false, CompareFlags::NullHandlingMode::kNullAsValue});
    }
    auto testCase =
        std::make_unique<TestCase>(pool_, testName, numRows, rowType);
    // Add benchmarks for std-sort and prefix-sort.
    {
      folly::addBenchmark(
           __FILE__,
           "StdSort_" + testCase->testName(),
           [rows = testCase->rows(),
            container = testCase->rowContainer(),
            sortFlags = compareFlags,
            iterateTimes = iterateTimes,
            this]() {
             for (auto i = 0; i < iterateTimes; ++i) {
               runStdSort(rows, container, sortFlags);
             }
             return rows.size() * iterateTimes;
           });
      folly::addBenchmark(
          __FILE__,
          "%PrefixSort_" + testCase->testName(),
          [rows = testCase->rows(),
           container = testCase->rowContainer(),
           sortFlags = compareFlags,
           iterateTimes = iterateTimes,
           this]() {
            for (auto i = 0; i < iterateTimes; ++i) {
              runPrefixSort(rows, container, sortFlags);
            }
            return rows.size() * iterateTimes;
          });
    }
    testCases_.push_back(std::move(testCase));
  }

 private:
  std::vector<std::unique_ptr<TestCase>> testCases_;
  memory::MemoryPool* pool_;
};
} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);

  memory::MemoryManager::initialize({});
  auto rootPool = memory::memoryManager()->addRootPool();
  auto leafPool = rootPool->addLeafChild("leaf");

  PrefixSortBenchmark bm(leafPool.get());

  const auto iterations = 100;
  const std::vector<vector_size_t> batchSizes = {100, 1'000, 10'000, 100'000};

  std::vector<RowTypePtr> rowTypes = {
      ROW({BIGINT()}),
      ROW({BIGINT(), BIGINT()}),
      ROW({BIGINT(), BIGINT(), BIGINT()}),
      ROW({BIGINT(), BIGINT(), BIGINT(), BIGINT()}),
  };

  for (auto batchSize : batchSizes) {
    for (auto i = 0; i < rowTypes.size(); ++i) {
      auto name = fmt::format("{}_Bigint_{}k", i + 1, batchSize / 1000);
      bm.benchmark(name, batchSize, rowTypes[i], iterations);
    }
  }

  // Varchar keys do not support normalization, hence, the performance of prefix
  // and std sort are the same.
  const RowTypePtr rowVarchar = ROW({VARCHAR()});
  for (auto batchSize : batchSizes) {
    auto name = fmt::format("Varchar_{}k", batchSize / 1000);
    bm.benchmark(name, batchSize, rowVarchar, iterations);
  }

  folly::runBenchmarks();

  return 0;
}