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

#include "velox/common/caching/FileGroupStats.h"
#include "velox/common/caching/FileIds.h"

#include <folly/Random.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::cache;

struct TestFile {
  StringIdLease id;
  int32_t numStripes;
};

struct TestGroup {
  StringIdLease id;
  std::vector<TestFile> files;
  std::vector<int32_t> columnSizes;
};

struct TestTable {
  std::vector<TestGroup> groups;
};

class GroupTrackerTest : public testing::Test {
 protected:
  void SetUp() override {
    rng_.seed(1);
    stats_ = std::make_unique<FileGroupStats>();
  }

  // Makes a test population of tables. A table has many file groups
  // (partitions) with the same number of columns and same column sizes.
  void makeTables(int32_t numTables) {
    for (auto i = 0; i < numTables; ++i) {
      int32_t sizeClass = random(100);
      int32_t numColumns = 0;
      if (sizeClass < 30) {
        numColumns = 15 + random(10);
      } else if (sizeClass < 95) {
        numColumns = 30 + random(20);
      } else {
        numColumns = (1000) + random(1000);
      }

      std::vector<int32_t> sizes;
      for (auto i = 0; i < numColumns; ++i) {
        sizes.push_back(columnSizes_[random(columnSizes_.size())]);
      }
      int32_t numGroups = 10 + random(30);
      std::vector<TestGroup> groups;
      for (auto i = 0; i < numGroups; ++i) {
        TestGroup group;
        group.id = StringIdLease(fileIds(), fmt::format("group{}", i));
        int32_t numFiles = 10 + random(6);
        for (auto fileNum = 0; fileNum < numFiles; ++fileNum) {
          group.files.push_back(TestFile{
              StringIdLease(
                  fileIds(),
                  fmt::format(
                      "{}/file{}", fileIds().string(group.id.id()), fileNum)),
              10});
        }
        group.columnSizes = sizes;
        groups.push_back(std::move(group));
      }
      TestTable table;
      table.groups = std::move(groups);
      tables_.push_back(std::move(table));
    }
  }

  // Reads a random set of groups from a random table, selecting a biased
  // random set of columns with some being sparsely read.
  void query(int32_t tableIndex) {
    std::vector<int32_t> columns;
    auto& table = tables_[tableIndex];
    auto numColumns = table.groups[0].columnSizes.size();
    std::unordered_set<int32_t> readColumns;
    int32_t toRead =
        readAllColumns_ ? numColumns : 5 + random(numColumns > 20 ? 10 : 5);
    for (auto i = 0; i < toRead; ++i) {
      if (readAllColumns_) {
        readColumns.insert(i);
      } else {
        readColumns.insert(random(numColumns, numColumns));
      }
    }
    auto numGroups = table.groups.size();
    auto readGroups = std::min<int32_t>(numGroups, 5 + random(numGroups));
    for (auto groupIndex = numGroups - readGroups; groupIndex < numGroups;
         ++groupIndex) {
      auto& group = table.groups[groupIndex];
      for (auto& file : group.files) {
        stats_->recordFile(file.id.id(), group.id.id(), file.numStripes);
        for (auto column : readColumns) {
          stats_->recordReference(
              file.id.id(),
              group.id.id(),
              TrackingId(column, 0),
              group.columnSizes[column]);
          auto readBytes = shouldRead(column, group.columnSizes[column]);
          if (readBytes) {
            if (stats_->shouldSaveToSsd(group.id.id(), TrackingId(column, 0))) {
              numFromSsd_ += readBytes;
            } else {
              numFromDisk_ += readBytes;
            }
            stats_->recordRead(
                file.id.id(), group.id.id(), TrackingId(column, 0), readBytes);
          }
        }
      }
    }
  }

  // For a stable 1/3, read 1/3 of the data.
  int32_t shouldRead(int32_t column, int32_t size) {
    if (readAll_) {
      return size;
    }
    int32_t rnd = random(100);
    if (column % 21 == 0) {
      // 5% are read for 1/10.
      return rnd > 95 ? size / 10 : 0;
    }
    if (column % 7 == 0) {
      // 1/7 is read for 1/1 1/2 of the time.
      if (rnd > 50) {
        return size / 3;
      }
      return 0;
    }

    return size;
  }

  // A bitwise or of random numbers will make 3/4 of the bits in
  // common true, giving a biased distribution.
  int32_t random(int32_t range, int32_t maskRange) {
    return ((folly::Random::rand32(rng_) % range) |
            (folly::Random::rand32(rng_) % maskRange)) %
        range;
  }

  int32_t random(int32_t pctLow, int32_t lowRange, int32_t highRange) {
    return folly::Random::rand32(rng_) %
        (folly::Random::rand32() % 100 > pctLow ? highRange : lowRange);
  }

  int32_t random(int32_t range) {
    return folly::Random::rand32(rng_) % range;
  }

  std::unique_ptr<FileGroupStats> stats_;
  std::vector<TestTable> tables_;
  std::vector<int32_t>
      columnSizes_{1000, 2000, 3000, 3000, 3000, 4000, 10000, 15000, 100000};
  uint64_t numFromSsd_{0};
  uint64_t numFromDisk_{0};
  folly::Random::DefaultGenerator rng_;
  bool readAll_{false};
  bool readAllColumns_{false};
};

TEST_F(GroupTrackerTest, randomScan) {
  // Runs 1K random queries on 100 files. The file and column read
  // probabilities are biased. Checks that cache covers a small
  // percentage of the data but selects the more frequently hit
  // columns and groups.
  constexpr int32_t kQueries = 1000;
  constexpr uint64_t kSsdSize = 1UL << 30;
  makeTables(101);
  auto numTables = tables_.size();
  for (auto i = 0; i < kQueries; ++i) {
    int32_t tableIndex = random(numTables, numTables);
    query(tableIndex);
    if (i % (kQueries / 10) == 0) {
      stats_->updateSsdFilter(kSsdSize);
    }
  }
  LOG(INFO) << stats_->toString(kSsdSize);

  auto dataPct = stats_->cachableDataPct();
  auto readPct = stats_->cachableReadPct();
  EXPECT_LE(3, dataPct);
  EXPECT_GE(4, dataPct);
  EXPECT_LE(45, readPct);
  EXPECT_GE(55, readPct);
}

TEST_F(GroupTrackerTest, limitAndDecay) {
  constexpr uint64_t kSsdSize = 1UL << 30;
  makeTables(1000);
  auto numTables = tables_.size();
  // We read all the  tables once and every 5th twice.
  readAllColumns_ = true;
  for (auto i = 0; i < numTables; ++i) {
    query(i);
    if (i % 5 == 0) {
      query(i);
    }

    if (i % 20 == 0) {
      stats_->updateSsdFilter(kSsdSize, 2);
    }
  }
  LOG(INFO) << stats_->toString(kSsdSize);
  auto dataPct = stats_->cachableDataPct();
  auto readPct = stats_->cachableReadPct();
  EXPECT_LE(0.4, dataPct);
  EXPECT_GE(1, dataPct);
  EXPECT_LE(10, readPct);
  EXPECT_GE(16, readPct);
}
