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

#include <gtest/gtest.h>
#include <algorithm>
#include <sstream>

#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/index/ClusterIndexConfig.h"
#include "velox/dwio/nimble/tools/NimbleDumpLib.h"
#include "velox/vector/FlatVector.h"

using namespace facebook;
using namespace facebook::nimble;

class NimbleDumpLibTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addRootPool("test");
    leafPool_ = pool_->addLeafChild("test_leaf");
  }

  std::shared_ptr<velox::ReadFile> makeReadFile(const std::string& content) {
    return std::make_shared<velox::InMemoryReadFile>(content);
  }

  // Returns 'numBatches' batches of 'rowsPerBatch' rows of a BIGINT "key"
  // column, whose keys run from 0 across the batches.
  std::vector<velox::VectorPtr> makeKeyBatches(
      int numBatches,
      int rowsPerBatch) {
    std::vector<velox::VectorPtr> batches;
    batches.reserve(numBatches);
    for (int batch = 0; batch < numBatches; ++batch) {
      auto keyVector = velox::BaseVector::create(
          velox::BIGINT(), rowsPerBatch, leafPool_.get());
      auto* flatKeys = keyVector->asFlatVector<int64_t>();
      for (int row = 0; row < rowsPerBatch; ++row) {
        flatKeys->set(row, batch * rowsPerBatch + row);
      }
      batches.emplace_back(
          std::make_shared<velox::RowVector>(
              leafPool_.get(),
              velox::ROW({"key"}, {velox::BIGINT()}),
              nullptr,
              rowsPerBatch,
              std::vector<velox::VectorPtr>{keyVector}));
    }
    return batches;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
};

TEST_F(NimbleDumpLibTest, emitStatsVectorizedStats) {
  auto type =
      velox::ROW({"intCol", "bigintCol"}, {velox::INTEGER(), velox::BIGINT()});

  auto intVector =
      velox::BaseVector::create(velox::INTEGER(), 100, leafPool_.get());
  auto* flatInt = intVector->asFlatVector<int32_t>();
  for (int i = 0; i < 100; ++i) {
    flatInt->set(i, i * 10);
  }

  auto bigintVector =
      velox::BaseVector::create(velox::BIGINT(), 100, leafPool_.get());
  auto* flatBigint = bigintVector->asFlatVector<int64_t>();
  for (int i = 0; i < 100; ++i) {
    flatBigint->set(i, i * 1000L);
  }

  auto rowVector = std::make_shared<velox::RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      100,
      std::vector<velox::VectorPtr>{intVector, bigintVector});

  WriterOptions writerOptions;
  writerOptions.enableStatsCollection = true;
  writerOptions.enableVectorizedStats = true;

  auto fileContent =
      test::createNimbleFile(*leafPool_, rowVector, writerOptions);
  auto readFile = makeReadFile(fileContent);

  std::ostringstream output;
  tools::NimbleDumpLib dumpLib{readFile, false, output};
  dumpLib.emitStats(false);

  auto result = output.str();
  EXPECT_NE(result.find("Vectorized Stats:"), std::string::npos);
  EXPECT_NE(result.find("index"), std::string::npos);
  EXPECT_NE(result.find("schema_node"), std::string::npos);
  EXPECT_NE(result.find("stat_type"), std::string::npos);
  EXPECT_NE(result.find("root"), std::string::npos);
  EXPECT_NE(result.find("intCol"), std::string::npos);
  EXPECT_NE(result.find("bigintCol"), std::string::npos);
  EXPECT_NE(result.find("INTEGRAL"), std::string::npos);
}

TEST_F(NimbleDumpLibTest, emitStatsVectorizedStatsNoHeader) {
  auto type = velox::ROW({"col"}, {velox::INTEGER()});

  auto intVector =
      velox::BaseVector::create(velox::INTEGER(), 10, leafPool_.get());
  auto* flatInt = intVector->asFlatVector<int32_t>();
  for (int i = 0; i < 10; ++i) {
    flatInt->set(i, i);
  }

  auto rowVector = std::make_shared<velox::RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      10,
      std::vector<velox::VectorPtr>{intVector});

  auto fileContent = test::createNimbleFile(*leafPool_, rowVector);
  auto readFile = makeReadFile(fileContent);

  std::ostringstream output;
  tools::NimbleDumpLib dumpLib{readFile, false, output};
  dumpLib.emitStats(true);

  auto result = output.str();
  EXPECT_EQ(result.find("index"), std::string::npos);
  EXPECT_EQ(result.find("schema_node"), std::string::npos);
  EXPECT_NE(result.find("root"), std::string::npos);
}

TEST_F(NimbleDumpLibTest, emitStatsLegacyStats) {
  auto type = velox::ROW({"col"}, {velox::INTEGER()});

  auto intVector =
      velox::BaseVector::create(velox::INTEGER(), 10, leafPool_.get());
  auto* flatInt = intVector->asFlatVector<int32_t>();
  for (int i = 0; i < 10; ++i) {
    flatInt->set(i, i);
  }

  auto rowVector = std::make_shared<velox::RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      10,
      std::vector<velox::VectorPtr>{intVector});

  WriterOptions writerOptions;
  writerOptions.enableStatsCollection = true;
  writerOptions.enableVectorizedStats = false;

  auto fileContent =
      test::createNimbleFile(*leafPool_, rowVector, writerOptions);
  auto readFile = makeReadFile(fileContent);

  std::ostringstream output;
  tools::NimbleDumpLib dumpLib{readFile, false, output};
  dumpLib.emitStats(false);

  auto result = output.str();
  EXPECT_NE(result.find("Legacy Stats:"), std::string::npos);
  EXPECT_NE(result.find("raw_size"), std::string::npos);
  EXPECT_EQ(result.find("schema_node"), std::string::npos);
}

TEST_F(NimbleDumpLibTest, emitStatsNoStats) {
  auto type = velox::ROW({"col"}, {velox::INTEGER()});

  auto intVector =
      velox::BaseVector::create(velox::INTEGER(), 10, leafPool_.get());
  auto* flatInt = intVector->asFlatVector<int32_t>();
  for (int i = 0; i < 10; ++i) {
    flatInt->set(i, i);
  }

  auto rowVector = std::make_shared<velox::RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      10,
      std::vector<velox::VectorPtr>{intVector});

  WriterOptions writerOptions;
  writerOptions.enableStatsCollection = false;

  auto fileContent =
      test::createNimbleFile(*leafPool_, rowVector, writerOptions);
  auto readFile = makeReadFile(fileContent);

  std::ostringstream output;
  tools::NimbleDumpLib dumpLib{readFile, false, output};
  dumpLib.emitStats(false);

  auto result = output.str();
  EXPECT_NE(result.find("No stats section found"), std::string::npos);
}

TEST_F(NimbleDumpLibTest, emitSchemaPrintsColumnAttributes) {
  auto type = velox::ROW({"id", "name"}, {velox::BIGINT(), velox::VARCHAR()});

  auto idVector =
      velox::BaseVector::create(velox::BIGINT(), 1, leafPool_.get());
  idVector->asFlatVector<int64_t>()->set(0, 42);
  auto nameVector =
      velox::BaseVector::create(velox::VARCHAR(), 1, leafPool_.get());
  nameVector->asFlatVector<velox::StringView>()->set(0, velox::StringView("a"));

  auto rowVector = std::make_shared<velox::RowVector>(
      leafPool_.get(),
      type,
      nullptr,
      1,
      std::vector<velox::VectorPtr>{idVector, nameVector});

  // Stamp Iceberg field-ids onto the top-level columns (as the write path does)
  // and confirm the dumped schema round-trips and surfaces them. Node ids are
  // pre-order: id=1, name=2 (root row = 0).
  WriterOptions writerOptions;
  writerOptions.schemaAttributes = {
      {1, {{"iceberg.id", "1"}}}, {2, {{"iceberg.id", "2"}}}};

  auto fileContent =
      test::createNimbleFile(*leafPool_, rowVector, writerOptions);
  auto readFile = makeReadFile(fileContent);

  std::ostringstream output;
  tools::NimbleDumpLib dumpLib{readFile, false, output};
  dumpLib.emitSchema(/*collapseFlatMap=*/false);

  auto result = output.str();
  EXPECT_NE(result.find("iceberg.id=1"), std::string::npos);
  EXPECT_NE(result.find("iceberg.id=2"), std::string::npos);
}

TEST_F(NimbleDumpLibTest, emitIndexShowsPartitionKeyStreams) {
  // Two stripes of ten rows with keys 0 to 19, each in a stripe group and so
  // in a partition of its own.
  constexpr int kRowsPerStripe{10};
  const auto batches = makeKeyBatches(/*numBatches=*/2, kRowsPerStripe);
  WriterOptions writerOptions;
  writerOptions.clusterIndexConfig =
      index::ClusterIndexConfigBuilder{}
          .withKeyColumns({"key"})
          .withSortOrders({SortOrder{.ascending = true}})
          .withEnforceKeyOrder(true)
          .withNoDuplicateKey(true)
          .build();
  writerOptions.metadataFlushThreshold = 0;
  const auto fileContent =
      test::createNimbleFile(*leafPool_, batches, writerOptions);

  std::ostringstream output;
  tools::NimbleDumpLib dumpLib{makeReadFile(fileContent), false, output};
  dumpLib.emitIndex();

  // Reads the numbers of the partition rows: partition, chunks, rows, key
  // stream bytes, metadata bytes and key stream offset.
  const auto parseNumber = [](std::string token) {
    token.erase(std::remove(token.begin(), token.end(), ','), token.end());
    return std::stoull(token);
  };
  std::istringstream lines{output.str()};
  std::string line;
  while (std::getline(lines, line) && line != "Partition Details:") {
  }
  ASSERT_TRUE(std::getline(lines, line)) << output.str();
  std::vector<std::vector<uint64_t>> partitions;
  while (std::getline(lines, line) && line.rfind("Total Chunks:", 0) != 0) {
    std::istringstream tokens{line};
    std::vector<uint64_t> numbers;
    std::string token;
    while (tokens >> token) {
      numbers.emplace_back(parseNumber(token));
    }
    ASSERT_EQ(numbers.size(), 6) << line;
    partitions.emplace_back(std::move(numbers));
  }
  ASSERT_EQ(partitions.size(), 2) << output.str();
  for (const auto& partition : partitions) {
    EXPECT_EQ(partition[2], kRowsPerStripe);
    EXPECT_GT(partition[3], 0);
  }
  // Each partition writes its key stream after the stripes of its group.
  EXPECT_GE(partitions[1][5], partitions[0][5] + partitions[0][3]);
  EXPECT_NE(
      line.find(
          fmt::format(
              "Key Stream: {} bytes", partitions[0][3] + partitions[1][3])),
      std::string::npos)
      << line;
}
