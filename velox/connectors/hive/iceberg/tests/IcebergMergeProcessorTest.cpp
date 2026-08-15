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

#include "velox/connectors/hive/iceberg/IcebergMergeProcessor.h"

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/type/Type.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

using IMP = IcebergMergeProcessor;

// Channel indices used by every test. The transform only reads
// `targetRowIdChannel` and `mergeRowChannel`; the other channels of the
// input RowVector exist purely to mirror the OSS contract documented by
// `MergeRowChangeProcessor.transformPage` and are populated with valid but
// unused data.
constexpr column_index_t kTargetRowIdChannel = 1;
constexpr column_index_t kMergeRowChannel = 2;

// Shapes used across the test fixture. Two target columns: BIGINT (id) and
// VARCHAR (name). The row id is a small ROW(VARCHAR file_path, BIGINT pos)
// — enough to verify verbatim row-id copy semantics without dragging the
// full Iceberg row id schema (file_path, pos, spec_id, partition_data) into
// the test.
const RowTypePtr kRowIdType = ROW({"file_path", "pos"}, {VARCHAR(), BIGINT()});
const RowTypePtr kMergeRowType =
    ROW({"id", "name", "operation", "case_number"},
        {BIGINT(), VARCHAR(), TINYINT(), INTEGER()});

class IcebergMergeProcessorTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool("IcebergMergeProcessorTest");
    vectorMaker_ = std::make_unique<velox::test::VectorMaker>(pool_.get());
  }

  // Builds the 5-channel input RowVector specified by
  // MergeRowChangeProcessor.transformPage:
  //   0 unique_id BIGINT
  //   1 target_row_id ROW(file_path, pos)
  //   2 merge_row ROW(id, name, operation, case_number)
  //   3 case_number INTEGER
  //   4 is_distinct BOOLEAN
  // Channels 0/3/4 are populated with deterministic-but-unused values so
  // any accidental dereference would be obvious in a debugger.
  RowVectorPtr makeInput(
      const std::vector<std::optional<std::string>>& filePaths,
      const std::vector<std::optional<int64_t>>& positions,
      const std::vector<std::optional<int64_t>>& mergeIds,
      const std::vector<std::optional<std::string>>& mergeNames,
      const std::vector<int8_t>& operations,
      const std::vector<int32_t>& caseNumbers) {
    const auto numRows = static_cast<vector_size_t>(operations.size());
    auto uniqueId = vectorMaker_->flatVector<int64_t>(
        numRows, [](vector_size_t i) { return 1'000 + i; });

    auto filePathField =
        vectorMaker_->flatVectorNullable<StringView>(toStringViews(filePaths));
    auto posField = vectorMaker_->flatVectorNullable<int64_t>(positions);
    auto rowId = std::make_shared<RowVector>(
        pool_.get(),
        kRowIdType,
        /*nulls=*/nullptr,
        numRows,
        std::vector<VectorPtr>{filePathField, posField});

    auto idField = vectorMaker_->flatVectorNullable<int64_t>(mergeIds);
    auto nameField =
        vectorMaker_->flatVectorNullable<StringView>(toStringViews(mergeNames));
    auto operationField = vectorMaker_->flatVector<int8_t>(operations);
    auto caseNumberField = vectorMaker_->flatVector<int32_t>(caseNumbers);
    auto mergeRow = std::make_shared<RowVector>(
        pool_.get(),
        kMergeRowType,
        /*nulls=*/nullptr,
        numRows,
        std::vector<VectorPtr>{
            idField, nameField, operationField, caseNumberField});

    auto caseNumberCh = vectorMaker_->flatVector<int32_t>(caseNumbers);
    auto isDistinct = vectorMaker_->flatVector<bool>(
        numRows, [](vector_size_t /*i*/) { return true; });

    return std::make_shared<RowVector>(
        pool_.get(),
        ROW({"unique_id",
             "target_row_id",
             "merge_row",
             "case_number",
             "is_distinct"},
            {BIGINT(), kRowIdType, kMergeRowType, INTEGER(), BOOLEAN()}),
        /*nulls=*/nullptr,
        numRows,
        std::vector<VectorPtr>{
            uniqueId, rowId, mergeRow, caseNumberCh, isDistinct});
  }

  IMP makeProcessor() const {
    return IMP(
        /*targetColumnTypes=*/{BIGINT(), VARCHAR()},
        /*outputColumnNames=*/
        {"id",
         "name",
         "operation",
         "$target_table_row_id",
         "insert_from_update"},
        /*rowIdType=*/kRowIdType,
        /*targetRowIdChannel=*/kTargetRowIdChannel,
        /*mergeRowChannel=*/kMergeRowChannel);
  }

  static std::vector<std::optional<StringView>> toStringViews(
      const std::vector<std::optional<std::string>>& values) {
    std::vector<std::optional<StringView>> result;
    result.reserve(values.size());
    for (const auto& v : values) {
      if (v.has_value()) {
        result.emplace_back(StringView(*v));
      } else {
        result.emplace_back(std::nullopt);
      }
    }
    return result;
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::unique_ptr<velox::test::VectorMaker> vectorMaker_;
};

TEST_F(IcebergMergeProcessorTest, emptyInputProducesEmptyOutput) {
  auto processor = makeProcessor();
  auto input = makeInput({}, {}, {}, {}, {}, {});
  auto output = processor.transform(input, pool_.get());

  EXPECT_EQ(output->size(), 0);
  EXPECT_EQ(*output->type(), *processor.outputType());
  EXPECT_EQ(output->childrenSize(), 5);
}

TEST_F(IcebergMergeProcessorTest, allInsertOperationsCopyTargetColumns) {
  auto processor = makeProcessor();
  // 3 INSERT rows. row_id is irrelevant for INSERT — set to null to make
  // it obvious if the transform mistakenly copies it onto INSERT output.
  auto input = makeInput(
      /*filePaths=*/{std::nullopt, std::nullopt, std::nullopt},
      /*positions=*/{std::nullopt, std::nullopt, std::nullopt},
      /*mergeIds=*/{10, 20, 30},
      /*mergeNames=*/
      {{std::string("a")}, {std::string("b")}, {std::string("c")}},
      /*operations=*/
      {IMP::kInsertOperationNumber,
       IMP::kInsertOperationNumber,
       IMP::kInsertOperationNumber},
      /*caseNumbers=*/{0, 0, 0});

  auto output = processor.transform(input, pool_.get());
  ASSERT_EQ(output->size(), 3);

  auto* ids = output->childAt(0)->asFlatVector<int64_t>();
  auto* names = output->childAt(1)->asFlatVector<StringView>();
  auto* operations = output->childAt(2)->asFlatVector<int8_t>();
  const auto& rowIds = output->childAt(3);
  auto* insertFromUpdate = output->childAt(4)->asFlatVector<int8_t>();

  for (vector_size_t i = 0; i < 3; ++i) {
    EXPECT_FALSE(ids->isNullAt(i));
    EXPECT_EQ(ids->valueAt(i), 10 * (i + 1));
    EXPECT_FALSE(names->isNullAt(i));
    EXPECT_EQ(operations->valueAt(i), IMP::kInsertOperationNumber);
    EXPECT_TRUE(rowIds->isNullAt(i));
    EXPECT_EQ(insertFromUpdate->valueAt(i), 0);
  }
  EXPECT_EQ(names->valueAt(0).str(), "a");
  EXPECT_EQ(names->valueAt(1).str(), "b");
  EXPECT_EQ(names->valueAt(2).str(), "c");
}

TEST_F(IcebergMergeProcessorTest, allDeleteOperationsNullTargetCopyRowId) {
  auto processor = makeProcessor();
  // 3 DELETE rows. mergeId / mergeName are irrelevant for DELETE — set
  // to non-null so we can verify the transform does NOT copy them through.
  auto input = makeInput(
      /*filePaths=*/
      {{std::string("file-1.parquet")},
       {std::string("file-2.parquet")},
       {std::string("file-3.parquet")}},
      /*positions=*/{100, 200, 300},
      /*mergeIds=*/{999, 999, 999},
      /*mergeNames=*/
      {{std::string("ignore")},
       {std::string("ignore")},
       {std::string("ignore")}},
      /*operations=*/
      {IMP::kDeleteOperationNumber,
       IMP::kDeleteOperationNumber,
       IMP::kDeleteOperationNumber},
      /*caseNumbers=*/{0, 0, 0});

  auto output = processor.transform(input, pool_.get());
  ASSERT_EQ(output->size(), 3);

  auto* ids = output->childAt(0)->asFlatVector<int64_t>();
  auto* names = output->childAt(1)->asFlatVector<StringView>();
  auto* operations = output->childAt(2)->asFlatVector<int8_t>();
  const auto* rowIds = output->childAt(3)->as<RowVector>();
  auto* insertFromUpdate = output->childAt(4)->asFlatVector<int8_t>();
  ASSERT_NE(rowIds, nullptr);

  auto* outFilePath = rowIds->childAt(0)->asFlatVector<StringView>();
  auto* outPos = rowIds->childAt(1)->asFlatVector<int64_t>();

  for (vector_size_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(ids->isNullAt(i)) << "DELETE must null out the id column.";
    EXPECT_TRUE(names->isNullAt(i)) << "DELETE must null out the name column.";
    EXPECT_EQ(operations->valueAt(i), IMP::kDeleteOperationNumber);
    EXPECT_FALSE(rowIds->isNullAt(i));
    EXPECT_EQ(outPos->valueAt(i), 100 * (i + 1));
    EXPECT_EQ(insertFromUpdate->valueAt(i), 0)
        << "Plain DELETE is not derived from UPDATE.";
  }
  EXPECT_EQ(outFilePath->valueAt(0).str(), "file-1.parquet");
  EXPECT_EQ(outFilePath->valueAt(1).str(), "file-2.parquet");
  EXPECT_EQ(outFilePath->valueAt(2).str(), "file-3.parquet");
}

TEST_F(IcebergMergeProcessorTest, updateOperationsFanOutDeleteThenInsert) {
  auto processor = makeProcessor();
  // 2 UPDATE rows → 4 output rows in (DELETE, INSERT) order per input row.
  auto input = makeInput(
      /*filePaths=*/
      {{std::string("u-1.parquet")}, {std::string("u-2.parquet")}},
      /*positions=*/{50, 60},
      /*mergeIds=*/{42, 84},
      /*mergeNames=*/
      {{std::string("new-a")}, {std::string("new-b")}},
      /*operations=*/
      {IMP::kUpdateOperationNumber, IMP::kUpdateOperationNumber},
      /*caseNumbers=*/{0, 0});

  auto output = processor.transform(input, pool_.get());
  ASSERT_EQ(output->size(), 4);

  auto* ids = output->childAt(0)->asFlatVector<int64_t>();
  auto* names = output->childAt(1)->asFlatVector<StringView>();
  auto* operations = output->childAt(2)->asFlatVector<int8_t>();
  const auto* rowIds = output->childAt(3)->as<RowVector>();
  auto* insertFromUpdate = output->childAt(4)->asFlatVector<int8_t>();
  ASSERT_NE(rowIds, nullptr);

  auto* outFilePath = rowIds->childAt(0)->asFlatVector<StringView>();
  auto* outPos = rowIds->childAt(1)->asFlatVector<int64_t>();

  // Output row 0 — DELETE half of UPDATE #0.
  EXPECT_TRUE(ids->isNullAt(0));
  EXPECT_TRUE(names->isNullAt(0));
  EXPECT_EQ(operations->valueAt(0), IMP::kDeleteOperationNumber);
  EXPECT_FALSE(rowIds->isNullAt(0));
  EXPECT_EQ(outFilePath->valueAt(0).str(), "u-1.parquet");
  EXPECT_EQ(outPos->valueAt(0), 50);
  EXPECT_EQ(insertFromUpdate->valueAt(0), 0);

  // Output row 1 — INSERT half of UPDATE #0.
  EXPECT_FALSE(ids->isNullAt(1));
  EXPECT_EQ(ids->valueAt(1), 42);
  EXPECT_FALSE(names->isNullAt(1));
  EXPECT_EQ(names->valueAt(1).str(), "new-a");
  EXPECT_EQ(operations->valueAt(1), IMP::kInsertOperationNumber);
  EXPECT_TRUE(rowIds->isNullAt(1));
  EXPECT_EQ(insertFromUpdate->valueAt(1), 1)
      << "INSERT half of UPDATE must mark insert_from_update=1.";

  // Output row 2 — DELETE half of UPDATE #1.
  EXPECT_TRUE(ids->isNullAt(2));
  EXPECT_EQ(operations->valueAt(2), IMP::kDeleteOperationNumber);
  EXPECT_EQ(outFilePath->valueAt(2).str(), "u-2.parquet");
  EXPECT_EQ(outPos->valueAt(2), 60);
  EXPECT_EQ(insertFromUpdate->valueAt(2), 0);

  // Output row 3 — INSERT half of UPDATE #1.
  EXPECT_EQ(ids->valueAt(3), 84);
  EXPECT_EQ(names->valueAt(3).str(), "new-b");
  EXPECT_EQ(operations->valueAt(3), IMP::kInsertOperationNumber);
  EXPECT_TRUE(rowIds->isNullAt(3));
  EXPECT_EQ(insertFromUpdate->valueAt(3), 1);
}

TEST_F(IcebergMergeProcessorTest, defaultCaseRowsAreDroppedFromOutput) {
  auto processor = makeProcessor();
  // 2 DEFAULT rows surrounding 1 INSERT — output should contain only the
  // INSERT row. This mirrors the OSS Java behavior where MERGE WHEN cases
  // that match no clause emit DEFAULT_CASE rows that the processor must
  // drop.
  auto input = makeInput(
      /*filePaths=*/{std::nullopt, std::nullopt, std::nullopt},
      /*positions=*/{std::nullopt, std::nullopt, std::nullopt},
      /*mergeIds=*/{77, 88, 99},
      /*mergeNames=*/
      {{std::string("x")}, {std::string("y")}, {std::string("z")}},
      /*operations=*/
      {IMP::kDefaultCaseOperationNumber,
       IMP::kInsertOperationNumber,
       IMP::kDefaultCaseOperationNumber},
      /*caseNumbers=*/{0, 0, 0});

  auto output = processor.transform(input, pool_.get());
  ASSERT_EQ(output->size(), 1);

  auto* ids = output->childAt(0)->asFlatVector<int64_t>();
  auto* names = output->childAt(1)->asFlatVector<StringView>();
  auto* operations = output->childAt(2)->asFlatVector<int8_t>();
  EXPECT_EQ(ids->valueAt(0), 88);
  EXPECT_EQ(names->valueAt(0).str(), "y");
  EXPECT_EQ(operations->valueAt(0), IMP::kInsertOperationNumber);
}

TEST_F(IcebergMergeProcessorTest, mixedOperationsProduceExpectedRowCount) {
  auto processor = makeProcessor();
  // 1 INSERT + 2 DELETE + 2 UPDATE + 1 DEFAULT = 1 + 2 + 4 + 0 = 7 output
  // rows. Verifies the precomputed total counts agree with the second-pass
  // emission so the internal VELOX_CHECK_EQ guard holds.
  auto input = makeInput(
      /*filePaths=*/
      {std::nullopt,
       {std::string("d-1.parquet")},
       {std::string("d-2.parquet")},
       {std::string("u-1.parquet")},
       {std::string("u-2.parquet")},
       std::nullopt},
      /*positions=*/
      {std::nullopt, 1, 2, 3, 4, std::nullopt},
      /*mergeIds=*/{1, 999, 999, 30, 40, 555},
      /*mergeNames=*/
      {{std::string("ins")},
       {std::string("ignore")},
       {std::string("ignore")},
       {std::string("upd-a")},
       {std::string("upd-b")},
       {std::string("ignore")}},
      /*operations=*/
      {IMP::kInsertOperationNumber,
       IMP::kDeleteOperationNumber,
       IMP::kDeleteOperationNumber,
       IMP::kUpdateOperationNumber,
       IMP::kUpdateOperationNumber,
       IMP::kDefaultCaseOperationNumber},
      /*caseNumbers=*/{0, 0, 0, 0, 0, 0});

  auto output = processor.transform(input, pool_.get());
  EXPECT_EQ(output->size(), 7);

  auto* operations = output->childAt(2)->asFlatVector<int8_t>();
  auto* insertFromUpdate = output->childAt(4)->asFlatVector<int8_t>();
  // Order should be:
  //   0: INSERT     (from input row 0)
  //   1: DELETE     (from input row 1)
  //   2: DELETE     (from input row 2)
  //   3,4: DELETE,INSERT (from UPDATE input row 3)
  //   5,6: DELETE,INSERT (from UPDATE input row 4)
  EXPECT_EQ(operations->valueAt(0), IMP::kInsertOperationNumber);
  EXPECT_EQ(operations->valueAt(1), IMP::kDeleteOperationNumber);
  EXPECT_EQ(operations->valueAt(2), IMP::kDeleteOperationNumber);
  EXPECT_EQ(operations->valueAt(3), IMP::kDeleteOperationNumber);
  EXPECT_EQ(operations->valueAt(4), IMP::kInsertOperationNumber);
  EXPECT_EQ(operations->valueAt(5), IMP::kDeleteOperationNumber);
  EXPECT_EQ(operations->valueAt(6), IMP::kInsertOperationNumber);

  // INSERT-derived-from-UPDATE markers: only positions 4 and 6 (the INSERT
  // halves of the two UPDATE input rows) should be 1.
  EXPECT_EQ(insertFromUpdate->valueAt(0), 0);
  EXPECT_EQ(insertFromUpdate->valueAt(1), 0);
  EXPECT_EQ(insertFromUpdate->valueAt(2), 0);
  EXPECT_EQ(insertFromUpdate->valueAt(3), 0);
  EXPECT_EQ(insertFromUpdate->valueAt(4), 1);
  EXPECT_EQ(insertFromUpdate->valueAt(5), 0);
  EXPECT_EQ(insertFromUpdate->valueAt(6), 1);
}

TEST_F(IcebergMergeProcessorTest, unknownOperationByteThrowsUserError) {
  auto processor = makeProcessor();
  auto input = makeInput(
      /*filePaths=*/{std::nullopt},
      /*positions=*/{std::nullopt},
      /*mergeIds=*/{1},
      /*mergeNames=*/{{std::string("oops")}},
      /*operations=*/{static_cast<int8_t>(7)},
      /*caseNumbers=*/{0});

  VELOX_ASSERT_USER_THROW(
      processor.transform(input, pool_.get()), "Unknown merge operation byte");
}

TEST_F(IcebergMergeProcessorTest, outputTypeReflectsConstructorArgs) {
  auto processor = makeProcessor();
  const auto& outType = processor.outputType();
  ASSERT_EQ(outType->size(), 5);
  EXPECT_TRUE(outType->childAt(0)->equivalent(*BIGINT()));
  EXPECT_TRUE(outType->childAt(1)->equivalent(*VARCHAR()));
  EXPECT_TRUE(outType->childAt(2)->equivalent(*TINYINT()));
  EXPECT_TRUE(outType->childAt(3)->equivalent(*kRowIdType));
  EXPECT_TRUE(outType->childAt(4)->equivalent(*TINYINT()));
  // All output column names come from the outputColumnNames ctor arg,
  // including the trailing operation / rowId / insert_from_update names.
  EXPECT_EQ(outType->nameOf(0), "id");
  EXPECT_EQ(outType->nameOf(1), "name");
  EXPECT_EQ(outType->nameOf(2), "operation");
  EXPECT_EQ(outType->nameOf(3), "$target_table_row_id");
  EXPECT_EQ(outType->nameOf(4), "insert_from_update");
}

TEST_F(IcebergMergeProcessorTest, outputColumnNamesArityMismatchThrows) {
  VELOX_ASSERT_THROW(
      IMP(
          /*targetColumnTypes=*/{BIGINT(), VARCHAR()},
          /*outputColumnNames=*/{"id", "name"}, // intentionally wrong arity.
          /*rowIdType=*/kRowIdType,
          /*targetRowIdChannel=*/kTargetRowIdChannel,
          /*mergeRowChannel=*/kMergeRowChannel),
      "outputColumnNames size must be targetColumnTypes.size() + 3");
}

// Upstream operators (e.g. SWITCH/ROW_CONSTRUCTOR in a Project) can wrap
// the merge_row column in a dictionary or constant encoding rather than
// materializing it flat. The transform must decode the wrapper before per-
// row indexing rather than failing the strict flat-RowVector check.
TEST_F(IcebergMergeProcessorTest, dictionaryEncodedMergeRowIsFlattened) {
  auto processor = makeProcessor();
  // Build a 2-row input via the regular helper, then dictionary-wrap the
  // merge_row column with an identity index buffer. Identity indices are
  // sufficient to exercise the flatten path — the decoded result must
  // equal the original flat data, and the transform's output must match
  // what an un-wrapped input would have produced.
  auto input = makeInput(
      /*filePaths=*/{std::string("/d.parquet"), std::nullopt},
      /*positions=*/{int64_t(5), std::nullopt},
      /*mergeIds=*/{int64_t(1), int64_t(2)},
      /*mergeNames=*/{std::string("a"), std::string("b")},
      /*operations=*/
      {IMP::kDeleteOperationNumber, IMP::kInsertOperationNumber},
      /*caseNumbers=*/{1, 1});

  const auto numRows = input->size();
  auto indices = AlignedBuffer::allocate<vector_size_t>(numRows, pool_.get());
  auto* rawIndices = indices->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    rawIndices[i] = i;
  }
  auto wrappedMergeRow = BaseVector::wrapInDictionary(
      /*nulls=*/nullptr, indices, numRows, input->childAt(kMergeRowChannel));

  std::vector<VectorPtr> children;
  children.reserve(input->childrenSize());
  for (size_t c = 0; c < input->childrenSize(); ++c) {
    children.push_back(
        c == kMergeRowChannel ? wrappedMergeRow
                              : input->childAt(static_cast<column_index_t>(c)));
  }
  auto wrappedInput = std::make_shared<RowVector>(
      pool_.get(),
      input->type(),
      /*nulls=*/nullptr,
      numRows,
      std::move(children));

  // Should not throw "merge_row column must be a flat RowVector".
  auto output = processor.transform(wrappedInput, pool_.get());
  // 1 DELETE + 1 INSERT = 2 output rows; operation column reflects each.
  ASSERT_EQ(output->size(), 2);
  const auto* operationOut = output->childAt(2)->asFlatVector<int8_t>();
  ASSERT_NE(operationOut, nullptr);
  EXPECT_EQ(operationOut->valueAt(0), IMP::kDeleteOperationNumber);
  EXPECT_EQ(operationOut->valueAt(1), IMP::kInsertOperationNumber);
}

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
