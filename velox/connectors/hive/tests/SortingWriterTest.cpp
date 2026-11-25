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

#include "velox/connectors/hive/SortingWriter.h"
#include <gtest/gtest.h>

#include <tuple>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/SortBuffer.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::memory;

namespace facebook::velox::connector::hive::test {

class MockWriter : public dwio::common::Writer {
 public:
  MockWriter() {
    setState(State::kRunning);
  }

  void write(const VectorPtr& data) override {
    writtenData_.push_back(data);
    totalRowsWritten_ += data->size();
  }

  const std::vector<VectorPtr>& writtenData() const {
    return writtenData_;
  }

  bool finish() override {
    return true;
  }

  void flush() override {}

  std::unique_ptr<dwio::common::FileMetadata> close() override {
    setState(State::kClosed);
    return nullptr;
  }

  void abort() override {
    setState(State::kAborted);
  }

  uint64_t getTotalRowsWritten() const {
    return totalRowsWritten_;
  }

 private:
  std::vector<VectorPtr> writtenData_;
  uint64_t totalRowsWritten_ = 0;
};

namespace {

// A row of the keyed test schema. The first three columns form the unique key.
struct KeyedRow {
  int64_t crc;
  std::string docId;
  std::string columnName;
  int64_t timestamp;
  std::string value;

  bool operator==(const KeyedRow& other) const {
    return std::tie(crc, docId, columnName, timestamp, value) ==
        std::tie(
               other.crc,
               other.docId,
               other.columnName,
               other.timestamp,
               other.value);
  }
};

std::ostream& operator<<(std::ostream& out, const KeyedRow& row) {
  return out << "(" << row.crc << ", " << row.docId << ", " << row.columnName
             << ", " << row.timestamp << ", " << row.value << ")";
}

} // namespace

class SortingWriterTest : public testing::Test,
                          public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  std::unique_ptr<SortBuffer> createSortBuffer() {
    const RowTypePtr inputType =
        ROW({{"c0", BIGINT()}, {"c1", INTEGER()}, {"c2", VARCHAR()}});

    const std::vector<column_index_t> sortColumnIndices{1};
    const std::vector<CompareFlags> sortCompareFlags{
        {true, true, false, CompareFlags::NullHandlingMode::kNullAsValue}};

    const velox::common::PrefixSortConfig prefixSortConfig{
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
        12};

    return std::make_unique<SortBuffer>(
        inputType,
        sortColumnIndices,
        sortCompareFlags,
        pool_.get(),
        &nonReclaimableSection_,
        prefixSortConfig);
  }

  RowVectorPtr createTestData(uint64_t numRows = 1000) {
    return makeRowVector(
        {makeFlatVector<int64_t>(
             numRows, [](vector_size_t row) { return row; }),
         makeFlatVector<int32_t>(
             numRows, [numRows](vector_size_t row) { return numRows - row; }),
         makeFlatVector<std::string>(numRows, [](vector_size_t row) {
           return fmt::format("row_{}", row);
         })});
  }

  static RowTypePtr keyedType() {
    return ROW(
        {{"crc", BIGINT()},
         {"doc_id_str", VARCHAR()},
         {"column_name", VARCHAR()},
         {"timestamp", BIGINT()},
         {"value", VARCHAR()}});
  }

  // Keys ascending, then timestamp descending, then value ascending.
  std::unique_ptr<SortBuffer> createKeyedSortBuffer() {
    const std::vector<column_index_t> sortColumnIndices{0, 1, 2, 3, 4};
    const CompareFlags ascending{
        true, true, false, CompareFlags::NullHandlingMode::kNullAsValue};
    const CompareFlags descending{
        true, false, false, CompareFlags::NullHandlingMode::kNullAsValue};
    const std::vector<CompareFlags> sortCompareFlags{
        ascending, ascending, ascending, descending, ascending};
    const velox::common::PrefixSortConfig prefixSortConfig{
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
        12};
    return std::make_unique<SortBuffer>(
        keyedType(),
        sortColumnIndices,
        sortCompareFlags,
        pool_.get(),
        &nonReclaimableSection_,
        prefixSortConfig);
  }

  RowVectorPtr makeKeyedBatch(const std::vector<KeyedRow>& rows) {
    const auto size = static_cast<vector_size_t>(rows.size());
    return makeRowVector(
        {makeFlatVector<int64_t>(
             size, [&](vector_size_t i) { return rows[i].crc; }),
         makeFlatVector<std::string>(
             size, [&](vector_size_t i) { return rows[i].docId; }),
         makeFlatVector<std::string>(
             size, [&](vector_size_t i) { return rows[i].columnName; }),
         makeFlatVector<int64_t>(
             size, [&](vector_size_t i) { return rows[i].timestamp; }),
         makeFlatVector<std::string>(
             size, [&](vector_size_t i) { return rows[i].value; })});
  }

  static std::vector<KeyedRow> collect(const MockWriter& writer) {
    std::vector<KeyedRow> rows;
    for (const auto& data : writer.writtenData()) {
      auto* batch = data->as<RowVector>();
      DecodedVector crc(*batch->childAt(0));
      DecodedVector docId(*batch->childAt(1));
      DecodedVector columnName(*batch->childAt(2));
      DecodedVector timestamp(*batch->childAt(3));
      DecodedVector value(*batch->childAt(4));
      for (vector_size_t i = 0; i < batch->size(); ++i) {
        rows.push_back(
            {crc.valueAt<int64_t>(i),
             std::string(docId.valueAt<StringView>(i)),
             std::string(columnName.valueAt<StringView>(i)),
             timestamp.valueAt<int64_t>(i),
             std::string(value.valueAt<StringView>(i))});
      }
    }
    return rows;
  }

  // Duplicates within a write, across writes and as exact copies.
  void writeDuplicates(SortingWriter& writer) {
    writer.write(makeKeyedBatch(
        {{2, "b", "c1", 10, "q"},
         {1, "a", "c1", 100, "x"},
         {1, "a", "c1", 200, "y"}}));
    writer.write(makeKeyedBatch(
        {{1, "a", "c1", 200, "b"},
         {1, "a", "c2", 50, "z"},
         {2, "b", "c1", 10, "q"}}));
  }

  tsan_atomic<bool> nonReclaimableSection_{false};
};

TEST_F(SortingWriterTest, largeRowSizeExceedsMaxOutputBytes) {
  auto mockWriter = std::make_unique<MockWriter>();
  auto mockWriterPtr = mockWriter.get();

  auto sortBuffer = createSortBuffer();

  const vector_size_t maxOutputRowsConfig = 1000;
  const uint64_t maxOutputBytesConfig = 1;
  const uint64_t outputTimeSliceLimitMs = 1000;

  SortingWriter sortingWriter(
      std::move(mockWriter),
      std::move(sortBuffer),
      {},
      maxOutputRowsConfig,
      maxOutputBytesConfig,
      outputTimeSliceLimitMs);

  RowVectorPtr testData = createTestData(10);
  sortingWriter.write(testData);
  ASSERT_TRUE(sortingWriter.finish());
  ASSERT_GT(mockWriterPtr->getTotalRowsWritten(), 0);
}

TEST_F(SortingWriterTest, dedupeKeepsWinnerPerKey) {
  auto mockWriter = std::make_unique<MockWriter>();
  auto* written = mockWriter.get();
  SortingWriter writer(
      std::move(mockWriter),
      createKeyedSortBuffer(),
      {0, 1, 2},
      1'024,
      10 << 20,
      1'000);
  writeDuplicates(writer);
  ASSERT_TRUE(writer.finish());

  const std::vector<KeyedRow> expected = {
      {1, "a", "c1", 200, "b"},
      {1, "a", "c2", 50, "z"},
      {2, "b", "c1", 10, "q"}};
  ASSERT_EQ(collect(*written), expected);
  const auto stats = writer.runtimeStats();
  ASSERT_EQ(stats.at(SortingWriter::kSortingWriterDedupedRows).sum, 3);
  ASSERT_EQ(writer.numDuplicateRows(), 3);
}

TEST_F(SortingWriterTest, dedupeAcrossOutputBatches) {
  auto mockWriter = std::make_unique<MockWriter>();
  auto* written = mockWriter.get();
  // One row per output batch puts every duplicate across a boundary.
  SortingWriter writer(
      std::move(mockWriter),
      createKeyedSortBuffer(),
      {0, 1, 2},
      1,
      10 << 20,
      1'000);
  writeDuplicates(writer);
  ASSERT_TRUE(writer.finish());

  const std::vector<KeyedRow> expected = {
      {1, "a", "c1", 200, "b"},
      {1, "a", "c2", 50, "z"},
      {2, "b", "c1", 10, "q"}};
  ASSERT_EQ(collect(*written), expected);
  ASSERT_EQ(
      writer.runtimeStats().at(SortingWriter::kSortingWriterDedupedRows).sum,
      3);
}

TEST_F(SortingWriterTest, dedupeAcrossOutputBatchesWithLongKeys) {
  auto mockWriter = std::make_unique<MockWriter>();
  auto* written = mockWriter.get();
  SortingWriter writer(
      std::move(mockWriter),
      createKeyedSortBuffer(),
      {0, 1, 2},
      1,
      10 << 20,
      1'000);
  const std::string docA = "document-identifier-0000000001";
  const std::string docB = "document-identifier-0000000002";
  const std::string column = "granulated-column-name";
  writer.write(makeKeyedBatch(
      {{1, docA, column, 100, "x"},
       {1, docA, column, 200, "y"},
       {1, docB, column, 10, "q"}}));
  writer.write(makeKeyedBatch(
      {{1, docA, column, 200, "b"}, {1, docB, column, 10, "q"}}));
  ASSERT_TRUE(writer.finish());

  const std::vector<KeyedRow> expected = {
      {1, docA, column, 200, "b"}, {1, docB, column, 10, "q"}};
  ASSERT_EQ(collect(*written), expected);
  ASSERT_EQ(writer.numDuplicateRows(), 3);
}

TEST_F(SortingWriterTest, dedupeKeyRunSpansManyOutputBatches) {
  auto mockWriter = std::make_unique<MockWriter>();
  auto* written = mockWriter.get();
  // Two rows per output batch leave whole batches that repeat the last key.
  SortingWriter writer(
      std::move(mockWriter),
      createKeyedSortBuffer(),
      {0, 1, 2},
      2,
      10 << 20,
      1'000);
  const std::string docA = "document-identifier-0000000001";
  const std::string docB = "document-identifier-0000000002";
  const std::string column = "granulated-column-name";
  writer.write(makeKeyedBatch(
      {{1, docA, column, 100, "a"},
       {1, docA, column, 200, "b"},
       {1, docA, column, 300, "c"},
       {1, docA, column, 400, "d"},
       {1, docA, column, 500, "e"},
       {1, docB, column, 10, "q"}}));
  ASSERT_TRUE(writer.finish());

  const std::vector<KeyedRow> expected = {
      {1, docA, column, 500, "e"}, {1, docB, column, 10, "q"}};
  ASSERT_EQ(collect(*written), expected);
  ASSERT_EQ(writer.numDuplicateRows(), 4);
}

TEST_F(SortingWriterTest, dedupeTreatsNullKeysAsEqual) {
  auto mockWriter = std::make_unique<MockWriter>();
  auto* written = mockWriter.get();
  SortingWriter writer(
      std::move(mockWriter),
      createKeyedSortBuffer(),
      {0, 1, 2},
      1'024,
      10 << 20,
      1'000);
  writer.write(makeRowVector(
      {makeFlatVector<int64_t>({1, 1, 1, 1}),
       makeNullableFlatVector<std::string>(
           {std::nullopt, "a", std::nullopt, std::nullopt}),
       makeNullableFlatVector<std::string>({"c1", "c1", "c1", std::nullopt}),
       makeFlatVector<int64_t>({100, 5, 200, 7}),
       makeFlatVector<std::string>({"x", "z", "y", "w"})}));
  ASSERT_TRUE(writer.finish());

  // The two rows with a null document and column c1 are one key.
  ASSERT_EQ(written->writtenData().size(), 1);
  velox::test::assertEqualVectors(
      makeRowVector(
          {makeFlatVector<int64_t>({1, 1, 1}),
           makeNullableFlatVector<std::string>(
               {std::nullopt, std::nullopt, "a"}),
           makeNullableFlatVector<std::string>({std::nullopt, "c1", "c1"}),
           makeFlatVector<int64_t>({7, 200, 5}),
           makeFlatVector<std::string>({"w", "y", "z"})}),
      written->writtenData()[0]);
  ASSERT_EQ(writer.numDuplicateRows(), 1);
}

TEST_F(SortingWriterTest, dedupeDisabledKeepsAllRows) {
  auto mockWriter = std::make_unique<MockWriter>();
  auto* written = mockWriter.get();
  SortingWriter writer(
      std::move(mockWriter),
      createKeyedSortBuffer(),
      {},
      1'024,
      10 << 20,
      1'000);
  writeDuplicates(writer);
  ASSERT_TRUE(writer.finish());
  ASSERT_EQ(written->getTotalRowsWritten(), 6u);
  ASSERT_EQ(
      writer.runtimeStats().count(SortingWriter::kSortingWriterDedupedRows), 0);
  ASSERT_EQ(writer.numDuplicateRows(), 0);
}

TEST_F(SortingWriterTest, dedupeWithoutDuplicatesKeepsBatchUnchanged) {
  auto mockWriter = std::make_unique<MockWriter>();
  auto* written = mockWriter.get();
  SortingWriter writer(
      std::move(mockWriter),
      createKeyedSortBuffer(),
      {0, 1, 2},
      1'024,
      10 << 20,
      1'000);
  writer.write(makeKeyedBatch(
      {{3, "c", "c1", 1, "v"},
       {1, "a", "c1", 1, "v"},
       {2, "b", "c1", 1, "v"}}));
  ASSERT_TRUE(writer.finish());
  ASSERT_EQ(written->getTotalRowsWritten(), 3u);
  ASSERT_EQ(
      writer.runtimeStats().at(SortingWriter::kSortingWriterDedupedRows).sum,
      0);
}

} // namespace facebook::velox::connector::hive::test
