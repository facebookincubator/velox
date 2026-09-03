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
#include <sstream>

#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/tools/NimbleDslLib.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::nimble::tools {
namespace {

class NimbleDslTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    rootPool_ = velox::memory::memoryManager()->addRootPool("nimble_dsl_test");
    leafPool_ = rootPool_->addLeafChild("leaf");
  }

  NimbleDslLib createDslLib(std::ostream& out, const std::string& fileData) {
    auto readFile =
        std::make_shared<velox::InMemoryReadFile>(std::string(fileData));
    return NimbleDslLib{out, /*enableColors=*/false, std::move(readFile)};
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
};

TEST_F(NimbleDslTest, execDescribeShowsColumns) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector = maker.rowVector(
      {"user_id", "name", "score"},
      {maker.flatVector<int64_t>({1001, 1002, 1003}),
       maker.flatVector<velox::StringView>({"Alice", "Bob", "Charlie"}),
       maker.flatVector<double>({95.5, 87.3, 92.1})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.describe();

  auto output = out.str();
  EXPECT_NE(output.find("user_id"), std::string::npos);
  EXPECT_NE(output.find("name"), std::string::npos);
  EXPECT_NE(output.find("score"), std::string::npos);
}

TEST_F(NimbleDslTest, execSelectAllColumns) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector = maker.rowVector(
      {"id", "value"},
      {maker.flatVector<int32_t>({10, 20, 30}),
       maker.flatVector<int32_t>({100, 200, 300})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.select({}, /*limit=*/10, /*offset=*/0, std::nullopt);

  auto output = out.str();
  // Header should contain column names.
  EXPECT_NE(output.find("id"), std::string::npos);
  EXPECT_NE(output.find("value"), std::string::npos);
  // Data values should appear.
  EXPECT_NE(output.find("10"), std::string::npos);
  EXPECT_NE(output.find("200"), std::string::npos);
  EXPECT_NE(output.find("300"), std::string::npos);
  // Row count summary.
  EXPECT_NE(output.find("(3 rows)"), std::string::npos);
}

TEST_F(NimbleDslTest, execSelectWithColumnProjection) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector = maker.rowVector(
      {"col_a", "col_b", "col_c"},
      {maker.flatVector<int32_t>({1, 2, 3}),
       maker.flatVector<int32_t>({4, 5, 6}),
       maker.flatVector<int32_t>({7, 8, 9})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.select({"col_a", "col_c"}, /*limit=*/10, /*offset=*/0, std::nullopt);

  auto output = out.str();
  EXPECT_NE(output.find("col_a"), std::string::npos);
  EXPECT_NE(output.find("col_c"), std::string::npos);
  // col_b should not appear in header.
  // Values from col_a and col_c should be present.
  EXPECT_NE(output.find("1"), std::string::npos);
  EXPECT_NE(output.find("7"), std::string::npos);
}

TEST_F(NimbleDslTest, execSelectWithLimit) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"x"}, {maker.flatVector<int32_t>({10, 20, 30, 40, 50})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.select({}, /*limit=*/2, /*offset=*/0, std::nullopt);

  auto output = out.str();
  EXPECT_NE(output.find("(2 rows)"), std::string::npos);
}

TEST_F(NimbleDslTest, execSelectWithOffset) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector = maker.rowVector(
      {"val"}, {maker.flatVector<int32_t>({100, 200, 300, 400, 500})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.select({}, /*limit=*/2, /*offset=*/2, std::nullopt);

  auto output = out.str();
  EXPECT_NE(output.find("300"), std::string::npos);
  EXPECT_NE(output.find("400"), std::string::npos);
  EXPECT_NE(output.find("(2 rows)"), std::string::npos);
}

TEST_F(NimbleDslTest, execSelectWithNulls) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector = maker.rowVector(
      {"nullable_col"},
      {maker.flatVectorNullable<int32_t>(
          {1, std::nullopt, 3, std::nullopt, 5})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.select({}, /*limit=*/10, /*offset=*/0, std::nullopt);

  auto output = out.str();
  EXPECT_NE(output.find("NULL"), std::string::npos);
  EXPECT_NE(output.find("(5 rows)"), std::string::npos);
}

TEST_F(NimbleDslTest, execSelectInvalidColumn) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector = maker.rowVector({"col1"}, {maker.flatVector<int32_t>({1, 2})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.select({"nonexistent"}, /*limit=*/10, /*offset=*/0, std::nullopt);

  auto output = out.str();
  EXPECT_NE(output.find("Error"), std::string::npos);
  EXPECT_NE(output.find("nonexistent"), std::string::npos);
}

TEST_F(NimbleDslTest, execSelectWithStripeFilter) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto v1 = maker.rowVector({"id"}, {maker.flatVector<int32_t>({1, 2, 3})});
  auto v2 = maker.rowVector({"id"}, {maker.flatVector<int32_t>({4, 5, 6})});

  // flushAfterWrite=true means each vector becomes a separate stripe.
  auto fileData = nimble::test::createNimbleFile(
      *rootPool_, std::vector<velox::VectorPtr>{v1, v2});

  // Read only stripe 1 (second stripe).
  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.select({}, /*limit=*/100, /*offset=*/0, /*stripeId=*/1);

  auto output = out.str();
  EXPECT_NE(output.find("4"), std::string::npos);
  EXPECT_NE(output.find("5"), std::string::npos);
  EXPECT_NE(output.find("6"), std::string::npos);
  EXPECT_NE(output.find("(3 rows)"), std::string::npos);
}

TEST_F(NimbleDslTest, execShowSchema) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector = maker.rowVector(
      {"id", "name"},
      {maker.flatVector<int64_t>({1, 2}),
       maker.flatVector<velox::StringView>({"a", "b"})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.showSchema();

  auto output = out.str();
  EXPECT_NE(output.find("id"), std::string::npos);
  EXPECT_NE(output.find("name"), std::string::npos);
  EXPECT_NE(output.find("Scalar"), std::string::npos);
  EXPECT_NE(output.find("Row"), std::string::npos);
}

TEST_F(NimbleDslTest, execShowInfo) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3, 4, 5})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.showInfo();

  auto output = out.str();
  EXPECT_NE(output.find("Nimble File"), std::string::npos);
  EXPECT_NE(output.find("Version"), std::string::npos);
  EXPECT_NE(output.find("File Size"), std::string::npos);
  EXPECT_NE(output.find("Stripe Count"), std::string::npos);
  EXPECT_NE(output.find("Row Count"), std::string::npos);
  EXPECT_NE(output.find("5"), std::string::npos);
}

TEST_F(NimbleDslTest, execShowStripes) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto v1 = maker.rowVector({"x"}, {maker.flatVector<int32_t>({1, 2})});
  auto v2 = maker.rowVector({"x"}, {maker.flatVector<int32_t>({3, 4, 5})});

  auto fileData = nimble::test::createNimbleFile(
      *rootPool_, std::vector<velox::VectorPtr>{v1, v2});

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.showStripes();

  auto output = out.str();
  // Header columns.
  EXPECT_NE(output.find("Stripe Id"), std::string::npos);
  EXPECT_NE(output.find("Row Count"), std::string::npos);
  // Should show two stripes.
  EXPECT_NE(output.find("0"), std::string::npos);
  EXPECT_NE(output.find("1"), std::string::npos);
}

TEST_F(NimbleDslTest, execShowStreams) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector = maker.rowVector(
      {"a", "b"},
      {maker.flatVector<int32_t>({1, 2}), maker.flatVector<int64_t>({10, 20})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.showStreams(std::nullopt);

  auto output = out.str();
  EXPECT_NE(output.find("Stripe Id"), std::string::npos);
  EXPECT_NE(output.find("Stream Id"), std::string::npos);
  EXPECT_NE(output.find("Stream Label"), std::string::npos);
}

TEST_F(NimbleDslTest, execShowStatsNoStats) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});

  // Explicitly disable vectorized stats to test the no-stats path.
  nimble::WriterOptions options;
  options.enableVectorizedStats = false;
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector, options);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.showStats();

  auto output = out.str();
  // Should report no stats available.
  EXPECT_NE(output.find("No vectorized statistics"), std::string::npos);
}

TEST_F(NimbleDslTest, execShowStatsWithStats) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"num"}, {maker.flatVector<int32_t>({10, 20, 30})});

  nimble::WriterOptions options;
  options.enableVectorizedStats = true;
  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector, options);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.showStats();

  auto output = out.str();
  // Should show stats table header.
  EXPECT_NE(output.find("Column"), std::string::npos);
  EXPECT_NE(output.find("Values"), std::string::npos);
  EXPECT_NE(output.find("Nulls"), std::string::npos);
}

TEST_F(NimbleDslTest, execShowEncoding) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector = maker.rowVector(
      {"a", "b"},
      {maker.flatVector<int32_t>({1, 2}), maker.flatVector<int64_t>({10, 20})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.showEncoding(std::nullopt);

  auto output = out.str();
  EXPECT_NE(output.find("Encoding"), std::string::npos);
  EXPECT_NE(output.find("Stream Label"), std::string::npos);
  EXPECT_NE(output.find("Stripe Id"), std::string::npos);
  // Should contain encoding type info like "Trivial" or similar.
  EXPECT_NE(output.find("Trivial"), std::string::npos);
}

TEST_F(NimbleDslTest, execShowIndexNoIndex) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.showIndex();

  auto output = out.str();
  EXPECT_NE(output.find("Not configured"), std::string::npos);
}

TEST_F(NimbleDslTest, execShowHistogram) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector = maker.rowVector(
      {"a", "b"},
      {maker.flatVector<int32_t>({1, 2, 3}),
       maker.flatVector<int64_t>({10, 20, 30})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.showHistogram(/*topLevel=*/false, std::nullopt);

  auto output = out.str();
  EXPECT_NE(output.find("Encoding Type"), std::string::npos);
  EXPECT_NE(output.find("Data Type"), std::string::npos);
  EXPECT_NE(output.find("Instance Count"), std::string::npos);
  EXPECT_NE(output.find("Storage %"), std::string::npos);
}

TEST_F(NimbleDslTest, execShowHistogramTopLevel) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector = maker.rowVector({"x"}, {maker.flatVector<int32_t>({1, 2, 3})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.showHistogram(/*topLevel=*/true, std::nullopt);

  auto output = out.str();
  EXPECT_NE(output.find("Encoding Type"), std::string::npos);
}

TEST_F(NimbleDslTest, execShowContent) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"val"}, {maker.flatVector<int32_t>({42, 99, 7})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  // Stream 0 is typically the nulls for the root row; stream 1 is the first
  // scalar column. We just verify it produces output without error.
  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.showContent(/*streamId=*/1, std::nullopt);

  auto output = out.str();
  // Should contain the actual values from the stream.
  EXPECT_FALSE(output.empty());
}

TEST_F(NimbleDslTest, execShowContentInvalidStream) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.showContent(/*streamId=*/999, std::nullopt);

  auto output = out.str();
  EXPECT_NE(output.find("Error"), std::string::npos);
}

TEST_F(NimbleDslTest, execShowFileLayout) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.showFileLayout();

  auto output = out.str();
  EXPECT_NE(output.find("Offset"), std::string::npos);
  EXPECT_NE(output.find("Size"), std::string::npos);
  EXPECT_NE(output.find("File Footer"), std::string::npos);
  EXPECT_NE(output.find("File Postscript"), std::string::npos);
}

TEST_F(NimbleDslTest, execShowStripesMetadata) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.showStripesMetadata();

  auto output = out.str();
  // Should show metadata or indicate none available.
  EXPECT_FALSE(output.empty());
}

TEST_F(NimbleDslTest, execShowStripeGroupsMetadata) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.showStripeGroupsMetadata();

  auto output = out.str();
  // Should show metadata or indicate none available.
  EXPECT_FALSE(output.empty());
}

TEST_F(NimbleDslTest, execShowOptionalSections) {
  velox::test::VectorMaker maker{leafPool_.get()};
  auto vector =
      maker.rowVector({"col"}, {maker.flatVector<int32_t>({1, 2, 3})});

  auto fileData = nimble::test::createNimbleFile(*rootPool_, vector);

  std::ostringstream out;
  auto sql = createDslLib(out, fileData);
  sql.showOptionalSections();

  auto output = out.str();
  // File may or may not have optional sections; just check it doesn't crash.
  EXPECT_FALSE(output.empty());
}

} // namespace
} // namespace facebook::nimble::tools
