/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/dwio/nimble/tablet/DataInput.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <random>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"

#include "velox/common/file/File.h"
#include "velox/common/file/IoUringReader.h"
#include "velox/common/file/LocalFile.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/memory/Allocation.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/common/testutil/TestValue.h"

namespace facebook::nimble {
namespace {

struct OptionParams {
  int32_t maxCoalesceDistance;
  int64_t maxCoalesceBytes;

  std::string toString() const {
    return fmt::format("dist{}_bytes{}", maxCoalesceDistance, maxCoalesceBytes);
  }
};

class DirectIoInMemoryReadFile : public velox::InMemoryReadFile {
 public:
  using velox::InMemoryReadFile::InMemoryReadFile;

  bool directIo(uint64_t& alignment) const override {
    alignment = velox::memory::AllocationTraits::kPageSize;
    return true;
  }
};

class FailingPreadvInMemoryReadFile : public velox::InMemoryReadFile {
 public:
  using velox::InMemoryReadFile::InMemoryReadFile;
  using velox::InMemoryReadFile::preadv;

  uint64_t preadv(
      folly::Range<const velox::common::Region*> /*regions*/,
      folly::Range<const folly::Range<char*>*> /*buffers*/,
      const velox::FileIoContext& /*context*/ = {}) const override {
    NIMBLE_FAIL("Injected preadv failure");
  }
};

class DataInputTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance({});
    velox::common::testutil::TestValue::enable();
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool("DataInputTest");
    ioStats_ = std::make_shared<velox::io::IoStatistics>();
  }

  std::unique_ptr<velox::InMemoryReadFile> createFile(
      const std::string& content) {
    return std::make_unique<velox::InMemoryReadFile>(content);
  }

  DirectDataInput::Options makeOptions(
      int32_t maxCoalesceDistance = 1'000'000,
      int64_t maxCoalesceBytes = 128 << 20) {
    DirectDataInput::Options options;
    options.pool = pool_.get();
    options.ioStats = ioStats_;
    options.maxCoalesceDistance = maxCoalesceDistance;
    options.maxCoalesceBytes = maxCoalesceBytes;
    return options;
  }

  DirectDataInput::Options makeOptions(const OptionParams& params) {
    DirectDataInput::Options options;
    options.pool = pool_.get();
    options.ioStats = ioStats_;
    options.maxCoalesceDistance = params.maxCoalesceDistance;
    options.maxCoalesceBytes = params.maxCoalesceBytes;
    return options;
  }

  static std::string_view refData(const DataInput::BufferRef& ref) {
    return {ref.data, ref.length};
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::shared_ptr<velox::io::IoStatistics> ioStats_;
};

// Parameterized test fixture for running functional tests across
// different option combinations.
class DataInputParamTest : public DataInputTest,
                           public ::testing::WithParamInterface<OptionParams> {
};

TEST_P(DataInputParamTest, singleGroup) {
  // Use alignment-friendly size (multiple of 4096).
  std::string data(102'400, '\0');
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<char>(i % 256);
  }
  auto file = createFile(data);

  // Each scenario is a list of {offset, length} regions in one group.
  using Regions = std::vector<std::pair<uint64_t, uint64_t>>;
  std::vector<Regions> scenarios = {
      {{100, 200}},
      {{5'000, 500}, {1'000, 300}, {3'000, 200}},
      {{300, 50}, {100, 50}, {200, 50}},
      {{0, 100}, {50'000, 100}, {98'000, 100}},
      {{0, 1}},
      {{0, 4'096}},
  };

  for (size_t scenario = 0; scenario < scenarios.size(); ++scenario) {
    const auto& regions = scenarios[scenario];
    std::string desc;
    for (const auto& [offset, length] : regions) {
      desc += fmt::format("[{},{}] ", offset, length);
    }
    SCOPED_TRACE(fmt::format("scenario {}: {}", scenario, desc));
    DirectDataInput input(file.get(), makeOptions(GetParam()));
    input.reserve(regions.size());
    input.startGroup();
    std::vector<uint32_t> indices;
    for (const auto& [offset, length] : regions) {
      indices.push_back(input.enqueue({offset, length}));
    }
    auto handle = input.load();
    for (size_t i = 0; i < regions.size(); ++i) {
      EXPECT_EQ(
          refData(input.bufferRef(indices[i])),
          data.substr(regions[i].first, regions[i].second));
    }
  }
}

TEST_P(DataInputParamTest, multipleGroups) {
  std::string data(102'400, '\0');
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<char>(i % 256);
  }
  auto file = createFile(data);

  // Each scenario is a list of groups, each group is a list of regions.
  using Groups = std::vector<std::vector<std::pair<uint64_t, uint64_t>>>;
  std::vector<Groups> scenarios = {
      {{{100, 200}}, {{500, 100}}},
      {{{500, 100}, {100, 200}, {800, 50}},
       {{50'500, 300}, {50'000, 400}, {60'000, 100}}},
      {{{100, 50}}, {{200, 50}}, {{300, 50}}},
      {{{500, 50}, {100, 50}}, {{550, 50}, {600, 50}}},
      {{{0, 100}}, {{98'000, 100}}},
  };

  for (size_t scenario = 0; scenario < scenarios.size(); ++scenario) {
    const auto& groups = scenarios[scenario];
    std::string desc;
    for (size_t g = 0; g < groups.size(); ++g) {
      desc += fmt::format("G{}: ", g);
      for (const auto& [offset, length] : groups[g]) {
        desc += fmt::format("[{},{}] ", offset, length);
      }
    }
    SCOPED_TRACE(fmt::format("scenario {}: {}", scenario, desc));
    DirectDataInput input(file.get(), makeOptions(GetParam()));
    uint32_t total = 0;
    for (const auto& g : groups) {
      total += g.size();
    }
    input.reserve(total);

    std::vector<std::pair<uint32_t, std::pair<uint64_t, uint64_t>>> indices;
    for (const auto& group : groups) {
      input.startGroup();
      for (const auto& [offset, length] : group) {
        indices.emplace_back(
            input.enqueue({offset, length}), std::make_pair(offset, length));
      }
    }
    auto handle = input.load();
    for (const auto& [idx, region] : indices) {
      EXPECT_EQ(
          refData(input.bufferRef(idx)),
          data.substr(region.first, region.second));
    }
  }
}

TEST_P(DataInputParamTest, clearAndReuse) {
  std::string data(100'000, '\0');
  for (int i = 0; i < 100'000; ++i) {
    data[i] = static_cast<char>(i % 256);
  }
  auto file = createFile(data);

  DirectDataInput input(file.get(), makeOptions(GetParam()));

  for (int cycle = 0; cycle < 3; ++cycle) {
    SCOPED_TRACE(fmt::format("cycle {}", cycle));
    input.reserve(2);
    input.startGroup();
    const auto idx0 =
        input.enqueue({static_cast<uint64_t>(cycle * 1'000), 100});
    input.startGroup();
    const auto idx1 =
        input.enqueue({static_cast<uint64_t>(50'000 + cycle * 1'000), 100});

    auto handle = input.load();
    EXPECT_EQ(refData(input.bufferRef(idx0)), data.substr(cycle * 1'000, 100));
    EXPECT_EQ(
        refData(input.bufferRef(idx1)),
        data.substr(50'000 + cycle * 1'000, 100));
    input.clear();
  }
}

TEST_P(DataInputParamTest, fuzz) {
  const size_t fileSize = 102'400;
  std::string data(fileSize, '\0');
  for (size_t i = 0; i < fileSize; ++i) {
    data[i] = static_cast<char>(i % 256);
  }
  auto file = createFile(data);

  std::mt19937 rng(42);
  const int numIterations = 10;

  for (int iter = 0; iter < numIterations; ++iter) {
    SCOPED_TRACE(fmt::format("iteration {}", iter));
    DirectDataInput input(file.get(), makeOptions(GetParam()));

    const int numGroups = 1 + rng() % 5;
    uint32_t totalRegions = 0;
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> groups(numGroups);
    uint64_t nextOffset = rng() % 100;
    for (int group = 0; group < numGroups; ++group) {
      const int numRegions = 1 + rng() % 10;
      uint64_t offset = nextOffset;
      for (int region = 0; region < numRegions; ++region) {
        const auto maxLen = std::min<uint64_t>(512, fileSize - offset);
        const auto length = 1 + rng() % maxLen;
        groups[group].emplace_back(offset, length);
        offset += length + rng() % 128;
      }
      std::shuffle(groups[group].begin(), groups[group].end(), rng);
      nextOffset = offset + 4'096;
      totalRegions += groups[group].size();
    }

    input.reserve(totalRegions);
    std::vector<std::pair<uint32_t, std::pair<uint64_t, uint64_t>>> indices;
    for (const auto& group : groups) {
      input.startGroup();
      for (const auto& [offset, length] : group) {
        indices.emplace_back(
            input.enqueue({offset, length}), std::make_pair(offset, length));
      }
    }

    auto handle = input.load();

    for (const auto& [idx, region] : indices) {
      EXPECT_EQ(
          refData(input.bufferRef(idx)),
          data.substr(region.first, region.second));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    DataInputOptions,
    DataInputParamTest,
    ::testing::Values(
        OptionParams{100, 128 << 20},
        OptionParams{1'000'000, 128 << 20},
        OptionParams{100, 1'024},
        OptionParams{100, 512}),
    [](const auto& info) { return info.param.toString(); });

// --- Non-parameterized tests ---

TEST_F(DataInputTest, alignment) {
  std::string data(16'384, '\0');
  for (int i = 0; i < 16'384; ++i) {
    data[i] = static_cast<char>(i % 256);
  }

  struct TestParam {
    bool directIo;
    uint64_t offset;
    uint64_t length;
    uint64_t expectedOverread;

    std::string debugString() const {
      return fmt::format(
          "directIo {} offset {} length {} expectedOverread {}",
          directIo,
          offset,
          length,
          expectedOverread);
    }
  };

  auto alignDown = [](uint64_t value, uint64_t align) {
    return value & ~(align - 1);
  };
  auto alignUp = [](uint64_t value, uint64_t align) {
    return (value + align - 1) & ~(align - 1);
  };
  auto expectedOverread = [&](uint64_t align,
                              uint64_t offset,
                              uint64_t length) {
    return alignUp(offset + length, align) - alignDown(offset, align) - length;
  };

  constexpr uint64_t kPageSize = velox::memory::AllocationTraits::kPageSize;
  std::vector<TestParam> testSettings = {
      // O_DIRECT files are page-aligned. offset 4100 -> alignDown=4096,
      // alignUp(4300)=8192 -> overread=8192-4096-200=3896.
      {true, 4'100, 200, expectedOverread(kPageSize, 4'100, 200)},

      // Offset already aligned.
      {true, 4'096, 200, expectedOverread(kPageSize, 4'096, 200)},

      // Offset 0, short length.
      {true, 0, 100, expectedOverread(kPageSize, 0, 100)},

      // Buffered files use byte alignment and do not overread for alignment.
      {false, 4'100, 200, 0},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    std::unique_ptr<velox::InMemoryReadFile> file;
    if (testData.directIo) {
      file = std::make_unique<DirectIoInMemoryReadFile>(data);
    } else {
      file = createFile(data);
    }
    auto stats = std::make_shared<velox::io::IoStatistics>();
    auto options = makeOptions();
    options.ioStats = stats;

    DirectDataInput input(file.get(), options);
    input.reserve(1);
    input.startGroup();
    const auto idx = input.enqueue({testData.offset, testData.length});
    auto handle = input.load();

    EXPECT_EQ(input.bufferRef(idx).length, testData.length);
    EXPECT_EQ(
        refData(input.bufferRef(idx)),
        data.substr(testData.offset, testData.length));
    EXPECT_EQ(stats->rawBytesRead(), testData.length);
    EXPECT_EQ(stats->rawOverreadBytes(), testData.expectedOverread);
    EXPECT_EQ(stats->read().count(), 1);
  }
}

TEST_F(DataInputTest, intraGroupCoalescing) {
  std::string data(10'000, '\0');
  for (int i = 0; i < 10'000; ++i) {
    data[i] = static_cast<char>(i % 256);
  }
  auto file = createFile(data);

  struct TestParam {
    std::vector<std::pair<uint64_t, uint64_t>> regions;
    int32_t maxCoalesceDistance;
    uint64_t expectedReadCount;

    std::string debugString() const {
      return fmt::format(
          "regions={}, dist={}, expectedReads={}",
          regions.size(),
          maxCoalesceDistance,
          expectedReadCount);
    }
  };

  std::vector<TestParam> testSettings = {
      // Two nearby + one far → 2 reads.
      {.regions = {{100, 50}, {200, 50}, {5'000, 50}},
       .maxCoalesceDistance = 100,
       .expectedReadCount = 2},

      // All within coalesce distance → 1 read.
      {.regions = {{100, 50}, {200, 50}, {300, 50}},
       .maxCoalesceDistance = 100,
       .expectedReadCount = 1},

      // All far apart → 3 reads.
      {.regions = {{100, 50}, {5'000, 50}, {9'000, 50}},
       .maxCoalesceDistance = 10,
       .expectedReadCount = 3},

      // Unsorted input, nearby after sort → 1 read.
      {.regions = {{300, 50}, {100, 50}, {200, 50}},
       .maxCoalesceDistance = 100,
       .expectedReadCount = 1},

      // Single region → 1 read.
      {.regions = {{100, 50}},
       .maxCoalesceDistance = 100,
       .expectedReadCount = 1},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto stats = std::make_shared<velox::io::IoStatistics>();
    auto options = makeOptions(testData.maxCoalesceDistance);
    options.ioStats = stats;
    DirectDataInput input(file.get(), options);

    input.reserve(testData.regions.size());
    input.startGroup();
    std::vector<std::pair<uint32_t, std::pair<uint64_t, uint64_t>>> indices;
    for (const auto& [offset, length] : testData.regions) {
      const auto idx = input.enqueue({offset, length});
      indices.emplace_back(idx, std::make_pair(offset, length));
    }

    auto handle = input.load();

    for (const auto& [idx, region] : indices) {
      EXPECT_EQ(
          refData(input.bufferRef(idx)),
          data.substr(region.first, region.second));
    }
    EXPECT_EQ(stats->read().count(), testData.expectedReadCount);
  }
}

DEBUG_ONLY_TEST_F(DataInputTest, coalescesDuplicateRegions) {
  std::string data(1'000, '\0');
  for (int i = 0; i < 1'000; ++i) {
    data[i] = static_cast<char>(i % 256);
  }
  auto file = createFile(data);

  DirectDataInput input(file.get(), makeOptions());
  input.reserve(4);
  input.startGroup();
  const auto duplicate0 = input.enqueue({100, 50});
  const auto duplicate1 = input.enqueue({100, 50});
  const auto adjacent0 = input.enqueue({150, 50});
  const auto adjacent1 = input.enqueue({200, 25});

  uint64_t allocatedBytes = 0;
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::DirectDataInput::allocateBuffer",
      std::function<void(uint64_t*)>(
          [&](uint64_t* bytes) { allocatedBytes = *bytes; }));
  auto handle = input.load();
  EXPECT_EQ(allocatedBytes, 125);

  EXPECT_EQ(refData(input.bufferRef(duplicate0)), data.substr(100, 50));
  EXPECT_EQ(refData(input.bufferRef(duplicate1)), data.substr(100, 50));
  EXPECT_EQ(refData(input.bufferRef(adjacent0)), data.substr(150, 50));
  EXPECT_EQ(refData(input.bufferRef(adjacent1)), data.substr(200, 25));
  EXPECT_EQ(input.bufferRef(duplicate0).data, input.bufferRef(duplicate1).data);
  // BufferRef::canonicalIndex exposes the dedup: both duplicates resolve to
  // the stored copy (the first-enqueued index of the duplicate set), while
  // unique regions resolve to themselves.
  EXPECT_EQ(input.bufferRef(adjacent0).canonicalIndex, adjacent0);
  EXPECT_EQ(input.bufferRef(adjacent1).canonicalIndex, adjacent1);
  EXPECT_EQ(input.bufferRef(duplicate0).canonicalIndex, duplicate0);
  EXPECT_EQ(input.bufferRef(duplicate1).canonicalIndex, duplicate0);
  EXPECT_EQ(ioStats_->read().count(), 1);
  EXPECT_EQ(ioStats_->read().sum(), 125);
  EXPECT_EQ(ioStats_->rawBytesRead(), 125);
  EXPECT_EQ(ioStats_->rawOverreadBytes(), 0);
  EXPECT_EQ(ioStats_->duplicateReadRegions(), 1);
  EXPECT_EQ(ioStats_->duplicateReadBytes(), 50);
}

TEST_F(DataInputTest, crossGroupCoalescing) {
  std::string data(10'000, '\0');
  for (int i = 0; i < 10'000; ++i) {
    data[i] = static_cast<char>(i % 256);
  }
  auto file = createFile(data);

  struct TestParam {
    // Regions per group: {offset, length}.
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> groups;
    int32_t maxCoalesceDistance;
    uint64_t expectedReadCount;

    std::string debugString() const {
      return fmt::format(
          "groups={}, dist={}, expectedReads={}",
          groups.size(),
          maxCoalesceDistance,
          expectedReadCount);
    }
  };

  std::vector<TestParam> testSettings = {
      // Groups in file order, nearby offsets → coalesced into 1 read.
      {.groups = {{{100, 50}}, {{200, 50}}},
       .maxCoalesceDistance = 1'000,
       .expectedReadCount = 1},

      // Groups in file order but far apart → 2 reads.
      {.groups = {{{100, 50}}, {{5'000, 50}}},
       .maxCoalesceDistance = 100,
       .expectedReadCount = 2},

      // Three groups, all nearby → 1 read.
      {.groups = {{{100, 50}}, {{200, 50}}, {{300, 50}}},
       .maxCoalesceDistance = 1'000,
       .expectedReadCount = 1},

      // Three groups, first two nearby, third far → 2 reads.
      {.groups = {{{100, 50}}, {{200, 50}}, {{5'000, 50}}},
       .maxCoalesceDistance = 100,
       .expectedReadCount = 2},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto stats = std::make_shared<velox::io::IoStatistics>();
    auto options = makeOptions(testData.maxCoalesceDistance);
    options.ioStats = stats;
    DirectDataInput input(file.get(), options);

    std::vector<std::pair<uint32_t, std::pair<uint64_t, uint64_t>>> indices;
    uint32_t totalRegions = 0;
    for (const auto& group : testData.groups) {
      totalRegions += group.size();
    }
    input.reserve(totalRegions);

    for (const auto& group : testData.groups) {
      input.startGroup();
      for (const auto& [offset, length] : group) {
        const auto idx = input.enqueue({offset, length});
        indices.emplace_back(idx, std::make_pair(offset, length));
      }
    }

    auto handle = input.load();

    for (const auto& [idx, region] : indices) {
      EXPECT_EQ(
          refData(input.bufferRef(idx)),
          data.substr(region.first, region.second));
    }
    EXPECT_EQ(stats->read().count(), testData.expectedReadCount);
  }
}

TEST_F(DataInputTest, rejectsInvalidGroupBoundaries) {
  std::string data(10'000, '\0');
  auto file = createFile(data);

  using Groups = std::vector<std::vector<std::pair<uint64_t, uint64_t>>>;
  struct TestParam {
    std::string name;
    Groups groups;
  };
  std::vector<TestParam> testSettings = {
      {"interleavedGroups", {{{500, 50}, {100, 50}}, {{150, 50}, {600, 50}}}},
      {"overlappingGroups", {{{100, 100}}, {{150, 50}}}},
      {"backwardGroups", {{{500, 50}}, {{100, 50}}}},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.name);
    DirectDataInput input(
        file.get(), makeOptions(/*maxCoalesceDistance=*/1'000));

    uint32_t totalRegions = 0;
    for (const auto& group : testData.groups) {
      totalRegions += group.size();
    }
    input.reserve(totalRegions);

    for (const auto& group : testData.groups) {
      input.startGroup();
      for (const auto& [offset, length] : group) {
        input.enqueue({offset, length});
      }
    }

    NIMBLE_ASSERT_THROW(
        input.load(),
        "Read groups must be sorted and non-overlapping by file offset");
  }
}

TEST_F(DataInputTest, clearAndReuse) {
  std::string data(1'000, 'A');
  auto file = createFile(data);

  DirectDataInput input(file.get(), makeOptions());

  input.reserve(1);
  input.startGroup();
  const auto idx0 = input.enqueue({0, 100});
  auto loadedData = input.load();
  EXPECT_EQ(refData(input.bufferRef(idx0)), data.substr(0, 100));

  input.clear();
  input.reserve(1);
  input.startGroup();
  const auto idx1 = input.enqueue({500, 100});
  loadedData = input.load();
  EXPECT_EQ(refData(input.bufferRef(idx1)), data.substr(500, 100));
}

TEST_F(DataInputTest, bufferRefLifetimeAfterClear) {
  std::string data(1'000, '\0');
  for (int i = 0; i < 1'000; ++i) {
    data[i] = static_cast<char>(i % 256);
  }
  auto file = createFile(data);

  DirectDataInput input(file.get(), makeOptions());
  input.reserve(1);
  input.startGroup();
  const auto idx = input.enqueue({100, 200});
  auto loadedData = input.load();

  // Copy the BufferRef before clear — loadedData handle keeps memory alive.
  const auto savedRef = input.bufferRef(idx);
  const auto expected = data.substr(100, 200);

  input.clear();

  EXPECT_EQ(refData(savedRef), expected);
}

TEST_F(DataInputTest, emptyLoad) {
  const std::string data(100, 'X');
  auto file = createFile(data);

  DirectDataInput input(file.get(), makeOptions());

  // Load without startGroup() should fail.
  NIMBLE_ASSERT_THROW(input.load(), "");

  // Load with empty group (no enqueue) should fail.
  input.startGroup();
  NIMBLE_ASSERT_THROW(input.load(), "No regions enqueued");
}

TEST_F(DataInputTest, emptyGroup) {
  std::string data(1'000, '\0');
  for (int i = 0; i < 1'000; ++i) {
    data[i] = static_cast<char>(i % 256);
  }
  auto file = createFile(data);

  DirectDataInput input(file.get(), makeOptions());
  input.reserve(1);

  // Enqueue without startGroup() should fail.
  NIMBLE_ASSERT_THROW(input.enqueue({100, 50}), "");

  // Empty group followed by startGroup() should fail.
  input.startGroup();
  NIMBLE_ASSERT_THROW(input.startGroup(), "Previous group is empty");

  // Enqueue into the first group, then start a new group.
  const auto idx = input.enqueue({100, 50});
  input.startGroup();
  const auto idx2 = input.enqueue({500, 50});
  auto loadedData = input.load();

  EXPECT_EQ(refData(input.bufferRef(idx)), data.substr(100, 50));
  EXPECT_EQ(refData(input.bufferRef(idx2)), data.substr(500, 50));

  DirectDataInput emptyFinalGroup(file.get(), makeOptions());
  emptyFinalGroup.reserve(1);
  emptyFinalGroup.startGroup();
  emptyFinalGroup.enqueue({100, 50});
  emptyFinalGroup.startGroup();
  NIMBLE_ASSERT_THROW(
      emptyFinalGroup.load(), "Read group must contain a region");
}

TEST_F(DataInputTest, handleOutlivesDataInput) {
  std::string data(1'000, '\0');
  for (int i = 0; i < 1'000; ++i) {
    data[i] = static_cast<char>(i % 256);
  }
  auto file = createFile(data);

  const auto statsBefore = pool_->stats();

  DataInput::Handle handle;
  DataInput::BufferRef savedRef;
  const auto expected = data.substr(200, 100);
  {
    DirectDataInput input(file.get(), makeOptions());
    input.reserve(1);
    input.startGroup();
    const auto idx = input.enqueue({200, 100});
    handle = input.load();
    savedRef = input.bufferRef(idx);
    EXPECT_EQ(refData(savedRef), expected);

    const auto statsAfterLoad = pool_->stats();
    EXPECT_EQ(statsAfterLoad.numAllocs, statsBefore.numAllocs + 1);
  }

  // DirectDataInput is destroyed, but handle keeps the buffer alive.
  const auto statsAfterDestroy = pool_->stats();
  EXPECT_EQ(statsAfterDestroy.numFrees, statsBefore.numFrees);
  EXPECT_EQ(std::string_view(savedRef.data, savedRef.length), expected);

  handle.reset();

  // After releasing the handle, memory is freed.
  const auto statsAfterRelease = pool_->stats();
  EXPECT_EQ(statsAfterRelease.numFrees, statsBefore.numFrees + 1);
}

TEST_F(DataInputTest, singleRequest) {
  std::string data(4'096, '\0');
  for (int i = 0; i < 4'096; ++i) {
    data[i] = static_cast<char>(i % 256);
  }
  auto file = createFile(data);

  struct TestParam {
    uint64_t offset;
    uint64_t length;
    std::string debugString() const {
      return fmt::format("offset {} length {}", offset, length);
    }
  };

  std::vector<TestParam> testSettings = {
      {3, 1},
      {0, 1},
      {0, 4'096},
      {100, 200},
      {4'000, 96},
  };

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    DirectDataInput input(file.get(), makeOptions());
    input.reserve(1);
    input.startGroup();
    const auto idx = input.enqueue({testData.offset, testData.length});
    auto handle = input.load();
    EXPECT_EQ(
        refData(input.bufferRef(idx)),
        data.substr(testData.offset, testData.length));
  }
}

TEST_F(DataInputTest, noCoalesceDistantRequests) {
  std::string data(100'000, '\0');
  for (int i = 0; i < 100'000; ++i) {
    data[i] = static_cast<char>(i % 256);
  }
  auto file = createFile(data);

  DirectDataInput input(file.get(), makeOptions(/*maxCoalesceDistance=*/10));
  input.reserve(2);
  input.startGroup();
  input.enqueue({100, 50});
  input.enqueue({50'000, 50});

  auto handle = input.load();

  // With 10-byte coalesce distance, two distant regions should not merge.
  EXPECT_EQ(ioStats_->read().count(), 2);
  EXPECT_EQ(ioStats_->rawBytesRead(), 100);
  EXPECT_LT(ioStats_->rawOverreadBytes(), 100);
}

TEST_F(DataInputTest, maxCoalesceDistanceCap) {
  // Verify that maxCoalesceDistance is capped at kMaxCoalesceDistance.
  // Two regions beyond kMaxCoalesceDistance apart should not be coalesced.
  std::string data(1'000'000, '\0');
  for (int i = 0; i < 1'000'000; ++i) {
    data[i] = static_cast<char>(i % 256);
  }
  auto file = createFile(data);

  auto stats = std::make_shared<velox::io::IoStatistics>();
  DirectDataInput::Options options;
  options.pool = pool_.get();
  options.ioStats = stats;
  options.maxCoalesceDistance = 1'000'000;

  DirectDataInput input(file.get(), options);
  input.reserve(2);
  input.startGroup();
  const auto idx0 = input.enqueue({0, 10});
  const auto idx1 = input.enqueue({500'000, 10});
  auto handle = input.load();

  EXPECT_EQ(refData(input.bufferRef(idx0)), data.substr(0, 10));
  EXPECT_EQ(refData(input.bufferRef(idx1)), data.substr(500'000, 10));
  EXPECT_EQ(stats->rawBytesRead(), 20);
  // Gap of 500K exceeds kMaxCoalesceDistance (256K), so two separate IO groups.
  EXPECT_EQ(stats->read().count(), 2);
  EXPECT_LT(stats->rawOverreadBytes(), 1'000);
}

TEST_F(DataInputTest, invalidOptions) {
  std::string data(100, 'X');
  auto file = createFile(data);

  auto validOptions = makeOptions();

  auto nullPool = validOptions;
  nullPool.pool = nullptr;
  NIMBLE_ASSERT_THROW(DirectDataInput(file.get(), nullPool), "");

  auto nullIoStats = validOptions;
  nullIoStats.ioStats = nullptr;
  NIMBLE_ASSERT_THROW(DirectDataInput(file.get(), nullIoStats), "");

  NIMBLE_ASSERT_THROW(DirectDataInput(nullptr, validOptions), "");
}

DEBUG_ONLY_TEST_F(DataInputTest, readError) {
  std::string data(10'000, '\0');
  for (int i = 0; i < 10'000; ++i) {
    data[i] = static_cast<char>(i % 256);
  }
  auto file = std::make_unique<FailingPreadvInMemoryReadFile>(data);

  DirectDataInput input(file.get(), makeOptions(/*maxCoalesceDistance=*/10));
  input.reserve(2);
  input.startGroup();
  input.enqueue({100, 50});
  input.startGroup();
  input.enqueue({5'000, 50});

  NIMBLE_ASSERT_THROW(input.load(), "Injected preadv failure");
}

TEST_F(DataInputTest, loadTracksLocalReadModeStats) {
  constexpr uint64_t kPageSize = velox::memory::AllocationTraits::kPageSize;
  std::string data(8 * kPageSize, '\0');
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<char>((i * 131) % 256);
  }
  auto tempFile = velox::common::testutil::TempFilePath::create();
  {
    velox::LocalWriteFile writeFile(
        tempFile->getPath(),
        /*shouldCreateParentDirectories=*/false,
        /*shouldThrowOnFileAlreadyExists=*/false);
    writeFile.append(data);
    writeFile.close();
  }

  const std::vector<std::vector<std::pair<uint64_t, uint64_t>>> groups = {
      {{100, 200}},
      {{kPageSize + 10, 1'000}},
      {{3 * kPageSize, 4'096}},
      {{7 * kPageSize, 100}},
  };
  constexpr uint64_t kPayloadBytes = 200 + 1'000 + 4'096 + 100;
  constexpr uint64_t kDirectReadBytes = 4 * kPageSize;

  struct TestCase {
    const char* name;
    bool bufferIo;
    bool useIoUring;
    uint64_t expectedReadBytes;
    uint64_t expectedOverreadBytes;
    uint64_t expectedIoUringReadCalls;
    uint64_t expectedIoUringRegions;
  };

  const std::vector<TestCase> testCases = {
      {
          "buffered",
          /*bufferIo=*/true,
          /*useIoUring=*/false,
          /*expectedReadBytes=*/kPayloadBytes,
          /*expectedOverreadBytes=*/0,
          /*expectedIoUringReadCalls=*/0,
          /*expectedIoUringRegions=*/0,
      },
      {
          "direct",
          /*bufferIo=*/false,
          /*useIoUring=*/false,
          /*expectedReadBytes=*/kDirectReadBytes,
          /*expectedOverreadBytes=*/kDirectReadBytes - kPayloadBytes,
          /*expectedIoUringReadCalls=*/0,
          /*expectedIoUringRegions=*/0,
      },
      {
          "directIoUring",
          /*bufferIo=*/false,
          /*useIoUring=*/true,
          /*expectedReadBytes=*/kDirectReadBytes,
          /*expectedOverreadBytes=*/kDirectReadBytes - kPayloadBytes,
          /*expectedIoUringReadCalls=*/1,
          /*expectedIoUringRegions=*/groups.size(),
      },
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);
    velox::ThreadLocalIoUringReader::testingClear();
    uint64_t numIoUringReaders{0};
    ASSERT_EQ(velox::getIoUringReaderStats(numIoUringReaders).readCalls, 0);
    ASSERT_EQ(numIoUringReaders, 0);

    auto file = std::make_unique<velox::LocalReadFile>(
        tempFile->getPath(),
        /*executor=*/nullptr,
        testCase.bufferIo,
        testCase.useIoUring);

    auto stats = std::make_shared<velox::io::IoStatistics>();
    auto options = makeOptions(/*maxCoalesceDistance=*/0);
    options.ioStats = stats;
    DirectDataInput input(file.get(), options);
    uint32_t totalRegions = 0;
    for (const auto& group : groups) {
      totalRegions += group.size();
    }
    input.reserve(totalRegions);
    std::vector<std::pair<uint32_t, std::pair<uint64_t, uint64_t>>> indices;
    for (const auto& group : groups) {
      input.startGroup();
      for (const auto& [offset, length] : group) {
        indices.emplace_back(
            input.enqueue({offset, length}), std::make_pair(offset, length));
      }
    }

    auto handle = input.load();

    for (const auto& [idx, region] : indices) {
      EXPECT_EQ(
          refData(input.bufferRef(idx)),
          data.substr(region.first, region.second));
    }
    EXPECT_EQ(file->bytesRead(), testCase.expectedReadBytes);
    EXPECT_EQ(stats->rawBytesRead(), kPayloadBytes);
    EXPECT_EQ(stats->rawOverreadBytes(), testCase.expectedOverreadBytes);
    EXPECT_EQ(stats->read().count(), groups.size());
    EXPECT_EQ(stats->read().sum(), testCase.expectedReadBytes);

    const auto ioUringStats = velox::getIoUringReaderStats(numIoUringReaders);
    EXPECT_EQ(ioUringStats.readCalls, testCase.expectedIoUringReadCalls);
    EXPECT_EQ(ioUringStats.regions, testCase.expectedIoUringRegions);
    if (testCase.useIoUring) {
      EXPECT_EQ(numIoUringReaders, 1);
      EXPECT_EQ(ioUringStats.batches, 1);
      EXPECT_EQ(ioUringStats.minRegionsPerRead, groups.size());
      EXPECT_EQ(ioUringStats.maxRegionsPerRead, groups.size());
      EXPECT_EQ(ioUringStats.minBatchSize, groups.size());
      EXPECT_EQ(ioUringStats.maxBatchSize, groups.size());
    } else {
      EXPECT_EQ(numIoUringReaders, 0);
      EXPECT_EQ(ioUringStats.batches, 0);
    }
  }
  velox::ThreadLocalIoUringReader::testingClear();
}

} // namespace
} // namespace facebook::nimble
