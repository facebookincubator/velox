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

#include "velox/exec/FlatRowSerde.h"

#include <gtest/gtest.h>
#include <utility>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/FlatRowToVector.h"
#include "velox/exec/tests/utils/FlatRowTestBase.h"
#include "velox/type/tests/utils/CustomTypesForTesting.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

namespace facebook::velox::exec::test {

class FlatRowSerdeTest : public FlatRowTestBase {
 public:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
  }

 protected:
  // Lists all rows from the container and serializes them into an aligned
  // buffer. Returns the buffer and a vector of per-row start pointers.
  std::pair<std::vector<char>, std::vector<char*>> serializeContainer(
      RowContainer* container,
      const RowContainerLayout& layout) {
    std::vector<char*> rows = listRows(container);
    return serializeRows(rows, layout);
  }

  // Serializes and deserializes all rows in 'input', then asserts
  // extractColumnFromFlatRow agrees with RowContainer::extractColumn.
  void verifyRoundtrip(const RowVectorPtr& input, int numKeys) {
    const int32_t numCols = static_cast<int32_t>(input->childrenSize());

    std::vector<TypePtr> keyTypes;
    std::vector<TypePtr> depTypes;
    for (int32_t i = 0; i < numCols; ++i) {
      if (i < numKeys) {
        keyTypes.push_back(input->childAt(i)->type());
      } else {
        depTypes.push_back(input->childAt(i)->type());
      }
    }

    auto container =
        makeRowContainer(keyTypes, depTypes, /*nullableKeys=*/true);
    storeRows(container.get(), input, numKeys);
    std::vector<char*> liveRows = listRows(container.get());

    RowContainerLayout layout(container.get());
    auto [buffer, serializedRows] = serializeRows(liveRows, layout);
    for (char* ser : serializedRows) {
      char* mutableRow = ser;
      deserializeRow(mutableRow, layout);
    }

    const int32_t numRows = static_cast<int32_t>(liveRows.size());
    for (int32_t col = 0; col < numCols; ++col) {
      const RowColumn& rowCol = layout.rowColumns[col];
      const TypePtr& colType = container->columnTypes()[col];

      // Oracle: live RowContainer rows, same order as serialization.
      auto liveCol = BaseVector::create(colType, numRows, pool());
      RowContainer::extractColumn(
          liveRows.data(), numRows, rowCol, /*columnHasNulls=*/true, liveCol);

      // Actual: extract from deserialized flat rows.
      auto flatCol = BaseVector::create(colType, numRows, pool());
      extractColumnFromFlatRow(
          reinterpret_cast<const char* const*>(serializedRows.data()),
          numRows,
          rowCol,
          colType,
          flatCol);

      for (int32_t row = 0; row < numRows; ++row) {
        ASSERT_EQ(liveCol->isNullAt(row), flatCol->isNullAt(row))
            << "Col " << col << " row " << row << " null mismatch";
        if (!liveCol->isNullAt(row)) {
          EXPECT_EQ(
              liveCol->compare(flatCol.get(), row, row, CompareFlags{}), 0)
              << "Col " << col << " row " << row << " value mismatch";
        }
      }
    }
  }
};

TEST_F(FlatRowSerdeTest, fuzzedScalarRoundtrip) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 1'024;
  opts.nullRatio = 0.1;
  opts.stringLength = 20;
  opts.stringVariableLength = true;
  VectorFuzzer fuzzer(opts, pool(), /*seed=*/42);
  verifyRoundtrip(
      makeRowVector({
          fuzzer.fuzzFlat(BIGINT()),
          fuzzer.fuzzFlat(INTEGER()),
          fuzzer.fuzzFlat(SMALLINT()),
          fuzzer.fuzzFlat(TINYINT()),
          fuzzer.fuzzFlat(BOOLEAN()),
          fuzzer.fuzzFlat(REAL()),
          fuzzer.fuzzFlat(DOUBLE()),
          fuzzer.fuzzFlat(TIMESTAMP()),
          fuzzer.fuzzFlat(VARCHAR()),
          fuzzer.fuzzFlat(VARBINARY()),
          fuzzer.fuzz(ARRAY(BIGINT())),
          fuzzer.fuzz(MAP(VARCHAR(), DOUBLE())),
          fuzzer.fuzz(ROW({BIGINT(), VARCHAR()})),
      }),
      /*numKeys=*/1);
}

TEST_F(FlatRowSerdeTest, unsupportedTypeThrows) {
  auto container = makeRowContainer(
      {facebook::velox::test::BIGINT_TYPE_WITH_CUSTOM_COMPARISON()},
      {},
      /*nullableKeys=*/true);

  VELOX_ASSERT_THROW(
      RowContainerLayout(container.get()),
      "Row-based spill does not support type with custom comparison");
}

TEST_F(FlatRowSerdeTest, complexTypeSerializedSlotLayout) {
  // ARRAY(BIGINT) as a sort key so it lands in the variable-column list.
  auto container =
      makeRowContainer({ARRAY(BIGINT())}, {}, /*nullableKeys=*/true);

  // Three-element array so the out-of-line byte count is deterministic and
  // verifiable.
  auto arrays = makeArrayVector<int64_t>({{1, 2, 3}});

  SelectivityVector allRows(1);
  DecodedVector decodedArr(*arrays, allRows);
  char* row = container->newRow();
  container->store(decodedArr, 0, row, /*column=*/0);

  RowContainerLayout layout(container.get());
  ASSERT_EQ(layout.variableColumns.size(), 1u);
  EXPECT_EQ(layout.variableColumns[0].kind, TypeKind::ARRAY)
      << "ARRAY column must be a non-string variable column";

  const size_t totalSize = layout.serializedSize(row);
  const size_t alignment = static_cast<size_t>(layout.alignment);
  std::vector<char> buffer(totalSize + alignment * 2, 0);
  void* rawStart = buffer.data();
  size_t space = buffer.size();
  void* aligned = std::align(alignment, totalSize, rawStart, space);
  ASSERT_NE(aligned, nullptr);
  char* current = reinterpret_cast<char*>(aligned);
  const char* bufferEnd = buffer.data() + buffer.size();
  char* serialized = current;
  serializeRow(row, current, bufferEnd, layout);

  // Locate the ARRAY column's slot in the fixed area.
  const RowColumn& arrCol = container->columnAt(0);
  const size_t colOffset = static_cast<size_t>(arrCol.offset());

  // slot[0] must equal the out-of-line byte count for the serialized array
  // data.
  const size_t slot0 = *reinterpret_cast<const size_t*>(serialized + colOffset);
  EXPECT_GT(slot0, 0u)
      << "Out-of-line byte count must be non-zero for a 3-element array";

  // slot[1] must be zero — the protocol zeroes the remainder of the
  // std::string_view field after writing the out-of-line byte count at
  // offset 0.
  const size_t slot1 =
      *reinterpret_cast<const size_t*>(serialized + colOffset + sizeof(size_t));
  EXPECT_EQ(slot1, 0u)
      << "Second size_t word of complex-type slot must be zero";

  verifyRoundtrip(makeRowVector({arrays}), /*numKeys=*/1);
}

TEST_F(FlatRowSerdeTest, complexTypeKeyRoundtrip) {
  // ROW(DOUBLE) sort key exercises the BufferInputStream path for complex
  // keys.
  verifyRoundtrip(
      makeRowVector({makeRowVector({makeFlatVector<double>({1.5, 2.5, 3.5})})}),
      /*numKeys=*/1);
}

TEST_F(FlatRowSerdeTest, deserializeAdvanceMovesPointer) {
  auto container =
      makeRowContainer({BIGINT()}, {VARCHAR()}, /*nullableKeys=*/true);

  auto keyVec = BaseVector::create(BIGINT(), 2, pool());
  auto valVec = BaseVector::create(VARCHAR(), 2, pool());
  auto* flatKey = keyVec->asFlatVector<int64_t>();
  auto* flatVal = valVec->asFlatVector<StringView>();
  flatKey->set(0, 10LL);
  flatKey->set(1, 20LL);
  flatVal->set(0, StringView("hello"));
  flatVal->set(1, StringView("world"));

  SelectivityVector allRows(2);
  DecodedVector decodedKey(*keyVec, allRows);
  DecodedVector decodedVal(*valVec, allRows);
  for (int i = 0; i < 2; ++i) {
    char* row = container->newRow();
    container->store(decodedKey, i, row, /*column=*/0);
    container->store(decodedVal, i, row, /*column=*/1);
  }

  RowContainerLayout layout(container.get());
  std::vector<char*> rows = listRows(container.get());

  size_t totalSize = 0;
  for (char* row : rows) {
    totalSize += layout.serializedSize(row);
  }
  const size_t alignment = static_cast<size_t>(layout.alignment);
  std::vector<char> buffer(totalSize + alignment * 2, 0);
  void* rawStart = buffer.data();
  size_t space = buffer.size();
  void* aligned = std::align(alignment, totalSize, rawStart, space);
  char* current = reinterpret_cast<char*>(aligned);
  const char* bufferEnd = buffer.data() + buffer.size();

  const char* row0Start = current;
  serializeRow(rows[0], current, bufferEnd, layout);
  const char* row1Start = current;
  serializeRow(rows[1], current, bufferEnd, layout);

  // Deserialize row0: current should advance to row1Start.
  char* ptr = const_cast<char*>(row0Start);
  deserializeRow(ptr, layout);

  EXPECT_EQ(ptr, row1Start)
      << "deserializeRow should advance pointer to start of next row";

  const RowColumn& keyCol = container->columnAt(0);
  const RowColumn& valCol = container->columnAt(1);
  const int64_t key0 =
      *reinterpret_cast<const int64_t*>(row0Start + keyCol.offset());
  EXPECT_EQ(key0, 10LL);
  const StringView sv0 =
      *reinterpret_cast<const StringView*>(row0Start + valCol.offset());
  EXPECT_EQ(std::string(sv0.data(), sv0.size()), "hello");

  // Deserialize row1 and verify values from the original pointer.
  char* ptr1 = const_cast<char*>(row1Start);
  deserializeRow(ptr1, layout);
  const int64_t key1 =
      *reinterpret_cast<const int64_t*>(row1Start + keyCol.offset());
  EXPECT_EQ(key1, 20LL);
  const StringView sv1 =
      *reinterpret_cast<const StringView*>(row1Start + valCol.offset());
  EXPECT_EQ(std::string(sv1.data(), sv1.size()), "world");
}

TEST_F(FlatRowSerdeTest, emptyRowContainerSerialize) {
  auto container =
      makeRowContainer({BIGINT()}, {VARCHAR()}, /*nullableKeys=*/true);
  RowContainerLayout layout(container.get());
  auto [buffer, serializedRows] = serializeContainer(container.get(), layout);
  EXPECT_TRUE(serializedRows.empty());
}

TEST_F(FlatRowSerdeTest, pureFixedWidthRoundtrip) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 256;
  opts.nullRatio = 0.15;
  VectorFuzzer fuzzer(opts, pool(), /*seed=*/7);

  auto input = makeRowVector({
      fuzzer.fuzzFlat(BIGINT()), // sort key
      fuzzer.fuzzFlat(INTEGER()),
      fuzzer.fuzzFlat(SMALLINT()),
      fuzzer.fuzzFlat(TINYINT()),
      fuzzer.fuzzFlat(BOOLEAN()),
      fuzzer.fuzzFlat(REAL()),
      fuzzer.fuzzFlat(DOUBLE()),
      fuzzer.fuzzFlat(TIMESTAMP()),
  });

  const int32_t numCols = static_cast<int32_t>(input->childrenSize());
  std::vector<TypePtr> keyTypes;
  std::vector<TypePtr> depTypes;
  for (int32_t i = 0; i < numCols; ++i) {
    if (i < 1) {
      keyTypes.push_back(input->childAt(i)->type());
    } else {
      depTypes.push_back(input->childAt(i)->type());
    }
  }

  auto container = makeRowContainer(keyTypes, depTypes, /*nullableKeys=*/true);
  RowContainerLayout layout(container.get());
  EXPECT_TRUE(layout.variableColumns.empty())
      << "Pure fixed-width schema must have no variable columns";

  verifyRoundtrip(input, /*numKeys=*/1);
}

} // namespace facebook::velox::exec::test
