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

#include <gtest/gtest.h>
#include "velox/exec/SortBuffer.h"

#include "velox/exec/PrefixSort.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox;
using namespace facebook::velox::memory;
using namespace facebook::velox::test;

namespace facebook::velox::exec::prefixsort::test {

class TestSetting {
 public:
  TestSetting(
      const std::vector<CompareFlags>& sortCompareFlags,
      const RowVectorPtr& expectedResult)
      : sortCompareFlags_(sortCompareFlags), expectedResult_(expectedResult){};

  const std::vector<CompareFlags>& sortCompareFlags() const {
    return sortCompareFlags_;
  }

  const RowVectorPtr& expectedResult() const {
    return expectedResult_;
  }

  std::string debugString() const {
    const auto numRows = expectedResult_->size();
    const std::string expectedResultStr =
        expectedResult_->toString(0, numRows > 100 ? 100 : numRows);
    std::stringstream sortCompareFlagsStr;
    for (const auto sortCompareFlag : sortCompareFlags_) {
      sortCompareFlagsStr << sortCompareFlag.toString();
    }
    return fmt::format(
        "sortCompareFlags:{}, \nexpectedResult:\n{}",
        sortCompareFlagsStr.str(),
        expectedResultStr);
  }

 private:
  std::vector<CompareFlags> sortCompareFlags_;
  RowVectorPtr expectedResult_;
};

class PrefixSortTest : public OperatorTestBase {
 protected:
  std::vector<char*>
  storeRows(int numRows, const RowVectorPtr& sortedRows, RowContainer* data);

  void testSort(
      int numRows,
      const std::vector<TestSetting> testSettings,
      const RowVectorPtr& sortedRows);

  // Use std::sort to generate expected result.
  const RowVectorPtr generateExpectedResult(
      const std::vector<CompareFlags> compareFlags,
      int numRows,
      const RowVectorPtr& sortedRows);

  // 20 MB
  const int64_t maxBytes_ = 20LL << 20;
  const std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::MemoryManager::getInstance()->addRootPool(
          "PrefixSortTest",
          maxBytes_)};

  const std::shared_ptr<memory::MemoryPool> pool_{
      rootPool_->addLeafChild("PrefixSortTest", maxBytes_)};

  const std::shared_ptr<memory::MemoryPool> fuzzerPool =
      rootPool_->addLeafChild("VectorFuzzer");
};

std::vector<char*> PrefixSortTest::storeRows(
    int numRows,
    const RowVectorPtr& sortedRows,
    RowContainer* data) {
  std::vector<char*> rows;
  SelectivityVector allRows(numRows);
  rows.resize(numRows);
  for (int row = 0; row < numRows; ++row) {
    rows[row] = data->newRow();
  }
  for (int column = 0; column < sortedRows->childrenSize(); ++column) {
    DecodedVector decoded(*sortedRows->childAt(column), allRows);
    for (int i = 0; i < numRows; ++i) {
      char* row = rows[i];
      data->store(decoded, i, row, column);
    }
  }
  return rows;
}

const RowVectorPtr PrefixSortTest::generateExpectedResult(
    const std::vector<CompareFlags> compareFlags,
    int numRows,
    const RowVectorPtr& sortedRows) {
  const auto sortedRowsType = asRowType(sortedRows->type());
  const int sortKeysNum = sortedRows->childrenSize();
  RowContainer rowContainer(sortedRowsType->children(), pool_.get());

  std::vector<char*> rows = storeRows(numRows, sortedRows, &rowContainer);
  std::sort(
      rows.begin(), rows.end(), [&](const char* leftRow, const char* rightRow) {
        for (vector_size_t index = 0; index < sortKeysNum; ++index) {
          if (auto result = rowContainer.compare(
                  leftRow, rightRow, index, compareFlags[index])) {
            return result < 0;
          }
        }
        return false;
      });
  const RowVectorPtr result =
      BaseVector::create<RowVector>(sortedRowsType, numRows, pool_.get());
  for (int column = 0; column < compareFlags.size(); ++column) {
    rowContainer.extractColumn(
        rows.data(), numRows, column, result->childAt(column));
  }
  return result;
}

void PrefixSortTest::testSort(
    const int numRows,
    const std::vector<TestSetting> testSettings,
    const RowVectorPtr& sortedRows) {
  const auto sortedRowsType = asRowType(sortedRows->type());
  for (const auto& setting : testSettings) {
    SCOPED_TRACE(setting.debugString());
    // 1. Store test rows into rowContainer.
    RowContainer rowContainer(sortedRowsType->children(), pool_.get());
    std::vector<char*> rows = storeRows(numRows, sortedRows, &rowContainer);
    // 2. Use PrefixSort sort rows.
    PrefixSort::sort(
        rows,
        pool_.get(),
        &rowContainer,
        setting.sortCompareFlags(),
        {1024,
         // Set threshold to 0 to enable prefix-sort in small dataset.
         0});

    // 3. Extract columns from row container.
    const RowVectorPtr actual =
        BaseVector::create<RowVector>(sortedRowsType, numRows, pool_.get());
    for (int column = 0; column < setting.sortCompareFlags().size(); ++column) {
      rowContainer.extractColumn(
          rows.data(), numRows, column, actual->childAt(column));
    }
    // 4. Compare acutal & expected
    assertEqualVectors(actual, setting.expectedResult());
  }
}

TEST_F(PrefixSortTest, singleKey) {
  const int numRows = 5;
  const int columnsSize = 7;

  // Vectors without nulls.
  const std::vector<VectorPtr> testData = {
      makeFlatVector<int64_t>({5, 4, 3, 2, 1}),
      makeFlatVector<int32_t>({5, 4, 3, 2, 1}),
      makeFlatVector<int16_t>({5, 4, 3, 2, 1}),
      makeFlatVector<float>({5.5, 4.4, 3.3, 2.2, 1.1}),
      makeFlatVector<double>({5.5, 4.4, 3.3, 2.2, 1.1}),
      makeFlatVector<Timestamp>(
          {Timestamp(5, 5),
           Timestamp(4, 4),
           Timestamp(3, 3),
           Timestamp(2, 2),
           Timestamp(1, 1)}),
      makeFlatVector<std::string_view>({"eee", "ddd", "ccc", "bbb", "aaa"})};
  for (int i = 5; i < columnsSize; ++i) {
    const auto sortedRows = makeRowVector({testData[i]});
    std::vector<CompareFlags> ascFlag = {
        {true, true, false, CompareFlags::NullHandlingMode::kNullAsValue}};
    std::vector<CompareFlags> dscFlag = {
        {true, false, false, CompareFlags::NullHandlingMode::kNullAsValue}};
    const auto ascResult = generateExpectedResult(ascFlag, numRows, sortedRows);
    const auto dscResult = generateExpectedResult(dscFlag, numRows, sortedRows);
    std::vector<TestSetting> testSettings = {
        TestSetting(ascFlag, ascResult), TestSetting(dscFlag, dscResult)};
    testSort(numRows, testSettings, sortedRows);
  }
}

TEST_F(PrefixSortTest, singleKeyWithNulls) {
  const int numRows = 5;
  const int columnsSize = 7;

  Timestamp ts = {5, 5};
  // Vectors with nulls.
  const std::vector<VectorPtr> testData = {
      makeNullableFlatVector<int64_t>({5, 4, std::nullopt, 2, 1}),
      makeNullableFlatVector<int32_t>({5, 4, std::nullopt, 2, 1}),
      makeNullableFlatVector<int16_t>({5, 4, std::nullopt, 2, 1}),
      makeNullableFlatVector<float>({5.5, 4.4, std::nullopt, 2.2, 1.1}),
      makeNullableFlatVector<double>({5.5, 4.4, std::nullopt, 2.2, 1.1}),
      makeNullableFlatVector<Timestamp>(
          {Timestamp(5, 5),
           Timestamp(4, 4),
           std::nullopt,
           Timestamp(2, 2),
           Timestamp(1, 1)}),
      makeNullableFlatVector<std::string_view>(
          {"eee", "ddd", std::nullopt, "bbb", "aaa"})};

  for (int i = 5; i < columnsSize; ++i) {
    const auto sortedRows = makeRowVector({testData[i]});
    std::vector<CompareFlags> ascFlag = {
        {true, true, false, CompareFlags::NullHandlingMode::kNullAsValue}};
    std::vector<CompareFlags> dscFlag = {
        {true, false, false, CompareFlags::NullHandlingMode::kNullAsValue}};

    const auto ascResult = generateExpectedResult(ascFlag, numRows, sortedRows);
    const auto dscResult = generateExpectedResult(dscFlag, numRows, sortedRows);
    std::vector<TestSetting> testSettings = {
        TestSetting(ascFlag, ascResult), TestSetting(dscFlag, dscResult)};
    testSort(numRows, testSettings, sortedRows);
  }
}

TEST_F(PrefixSortTest, multipleKeys) {
  // Test all keys normalized : bigint, integer
  {
    const int numRows = 5;
    const std::vector<VectorPtr> testData = {
        makeNullableFlatVector<int64_t>({5, 2, std::nullopt, 2, 1}),
        makeNullableFlatVector<int32_t>({5, 4, std::nullopt, 2, 1})};
    const auto sortedRows = makeRowVector(testData);
    std::vector<CompareFlags> ascFlag = {
        {true, true, false, CompareFlags::NullHandlingMode::kNullAsValue},
        {true, true, false, CompareFlags::NullHandlingMode::kNullAsValue}};
    std::vector<CompareFlags> dscFlag = {
        {true, false, false, CompareFlags::NullHandlingMode::kNullAsValue},
        {true, false, false, CompareFlags::NullHandlingMode::kNullAsValue}};

    const auto ascResult = generateExpectedResult(ascFlag, numRows, sortedRows);
    const auto dscResult = generateExpectedResult(dscFlag, numRows, sortedRows);
    std::vector<TestSetting> testSettings = {
        TestSetting(ascFlag, ascResult), TestSetting(dscFlag, dscResult)};
    testSort(numRows, testSettings, sortedRows);
  }

  // Test keys with semi-normalized : bigint, varchar
  {
    const int numRows = 5;
    const std::vector<VectorPtr> testData = {
        makeNullableFlatVector<int64_t>({5, 2, std::nullopt, 2, 1}),
        makeNullableFlatVector<std::string_view>(
            {"eee", "ddd", std::nullopt, "bbb", "aaa"})};
    const auto sortedRows = makeRowVector(testData);
    std::vector<CompareFlags> ascFlag = {
        {true, true, false, CompareFlags::NullHandlingMode::kNullAsValue},
        {true, true, false, CompareFlags::NullHandlingMode::kNullAsValue}};
    std::vector<CompareFlags> dscFlag = {
        {true, false, false, CompareFlags::NullHandlingMode::kNullAsValue},
        {true, false, false, CompareFlags::NullHandlingMode::kNullAsValue}};

    const auto ascResult = generateExpectedResult(ascFlag, numRows, sortedRows);
    const auto dscResult = generateExpectedResult(dscFlag, numRows, sortedRows);
    std::vector<TestSetting> testSettings = {
        TestSetting(ascFlag, ascResult), TestSetting(dscFlag, dscResult)};
    testSort(numRows, testSettings, sortedRows);
  }
}

TEST_F(PrefixSortTest, fuzz) {
  std::vector<TypePtr> allTypes = {
      INTEGER(),
      BOOLEAN(),
      TINYINT(),
      SMALLINT(),
      BIGINT(),
      HUGEINT(),
      REAL(),
      DOUBLE(),
      TIMESTAMP(),
      VARCHAR(),
      VARBINARY()};
  const int numRows = 10240;
  for (const auto& type : allTypes) {
    VectorFuzzer fuzzer(
        {.vectorSize = numRows, .nullRatio = 0.1}, fuzzerPool.get());
    RowVectorPtr sortedRows = fuzzer.fuzzRow(ROW({type}));
    std::vector<CompareFlags> ascFlag = {
        {true, true, false, CompareFlags::NullHandlingMode::kNullAsValue}};
    std::vector<CompareFlags> dscFlag = {
        {true, false, false, CompareFlags::NullHandlingMode::kNullAsValue}};
    const auto ascResult = generateExpectedResult(ascFlag, numRows, sortedRows);
    const auto dscResult = generateExpectedResult(dscFlag, numRows, sortedRows);
    std::vector<TestSetting> testSettings = {
        TestSetting(ascFlag, ascResult), TestSetting(dscFlag, dscResult)};
    testSort(numRows, testSettings, sortedRows);
  }
}

TEST_F(PrefixSortTest, fuzzMulti) {
  std::vector<TypePtr> allTypes = {
      INTEGER(),
      BOOLEAN(),
      TINYINT(),
      SMALLINT(),
      BIGINT(),
      HUGEINT(),
      REAL(),
      DOUBLE(),
      TIMESTAMP(),
      VARCHAR(),
      VARBINARY()};
  const int numRows = 10240;
  const TypePtr payload = VARCHAR();
  VectorFuzzer fuzzer(
      {.vectorSize = numRows, .nullRatio = 0.1}, fuzzerPool.get());
  for (const auto& type1 : allTypes) {
    for (const auto& type2 : allTypes) {
      RowVectorPtr sortedRows = fuzzer.fuzzRow(ROW({type1, type2, payload}));
      std::vector<CompareFlags> ascFlag = {
          {true, true, false, CompareFlags::NullHandlingMode::kNullAsValue},
          {true, true, false, CompareFlags::NullHandlingMode::kNullAsValue}};
      std::vector<CompareFlags> dscFlag = {
          {true, false, false, CompareFlags::NullHandlingMode::kNullAsValue},
          {true, false, false, CompareFlags::NullHandlingMode::kNullAsValue}};
      const auto ascResult =
          generateExpectedResult(ascFlag, numRows, sortedRows);
      const auto dscResult =
          generateExpectedResult(dscFlag, numRows, sortedRows);
      std::vector<TestSetting> testSettings = {
          TestSetting(ascFlag, ascResult), TestSetting(dscFlag, dscResult)};
      testSort(numRows, testSettings, sortedRows);
    }
  }
}
} // namespace facebook::velox::exec::prefixsort::test
