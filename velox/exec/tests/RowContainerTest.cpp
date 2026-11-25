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
#include <gmock/gmock.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/Spiller.h"
#include "velox/exec/VectorHasher.h"
#include "velox/exec/tests/utils/RowContainerTestBase.h"
#include "velox/type/tests/utils/CustomTypesForTesting.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox::test;

namespace facebook::velox::exec::test {
namespace {
// Ensures that 'column is non-null by copying a non-null value over nulls.
void makeNonNull(RowVectorPtr batch, int32_t column) {
  auto values = batch->childAt(column);
  std::vector<vector_size_t> nonNulls;
  for (auto i = 0; i < values->size(); ++i) {
    if (!values->isNullAt(i)) {
      nonNulls.push_back(i);
    }
  }
  VELOX_CHECK(!nonNulls.empty(), "Whole column must not be null");
  vector_size_t nonNull = 0;
  for (auto i = 0; i < values->size(); ++i) {
    if (values->isNullAt(i)) {
      values->copy(values.get(), i, nonNulls[nonNull++ % nonNulls.size()], 1);
    }
  }
}

// Makes a string with deterministic distinct content.
void makeLargeString(int32_t approxSize, std::string& string) {
  string.reserve(approxSize);
  while (string.size() < string.capacity() - 20) {
    auto size = string.size();
    auto number =
        fmt::format("{}", (size + 1 + approxSize) * 0xc6a4a7935bd1e995L);
    string.resize(size + number.size());
    memcpy(&string[size], &number[0], number.size());
  }
}

int32_t sign(int32_t n) {
  return n < 0 ? -1 : n == 0 ? 0 : 1;
}

class RowContainerTest : public RowContainerTestBase,
                         public testing::WithParamInterface<bool> {
 protected:
  // Whether the container keeps an array of row pointers.
  bool useListRowIndex() const {
    return GetParam();
  }

  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    velox::common::testutil::TestValue::enable();
  }

  void testExtractColumn(
      RowContainer& container,
      const std::vector<char*>& rows,
      int column,
      const VectorPtr& expected) {
    // This set of tests uses all variations of the extractColumn API to copy
    // column value for all rows in the container. Run these with offset 0
    // and offset to skip the first 1/3 rows.
    testExtractColumnAllRows(container, rows, column, expected, 0);
    testExtractColumnAllRows(
        container, rows, column, expected, rows.size() / 3);
    // This set of tests use extractColumn with an out-of-order pattern
    // and repetitions. Run these with offset 0 and offset to skip the first
    // 1/3 rows.
    testOutOfOrderRowNumbers(container, rows, column, expected, 0);
    testOutOfOrderRowNumbers(
        container, rows, column, expected, rows.size() / 3);

    testRepeatedRowNumbers(container, rows, column, expected);
    testNegativeRowNumbers(container, rows, column, expected);
  }

  void testExtractColumnAllRows(
      RowContainer& container,
      const std::vector<char*>& rows,
      int column,
      const VectorPtr& expected,
      vector_size_t offset) {
    auto size = rows.size();

    auto testEqualVectors = [](const VectorPtr& lhs,
                               const VectorPtr& rhs,
                               vector_size_t lhsIndex,
                               vector_size_t rhsIndex) {
      for (auto i = 0; i < lhs->size() - lhsIndex; i++) {
        EXPECT_TRUE(lhs->equalValueAt(rhs.get(), lhsIndex + i, rhsIndex + i));
      }
    };

    auto testBasic = [&]() {
      // Test the extractColumn API that didn't use the offset parameter and
      // copied to the start of the result vector.
      auto result = BaseVector::create(expected->type(), size, pool_.get());
      container.extractColumn(rows.data(), size, column, result);
      assertEqualVectors(expected, result);
    };

    auto testBasicWithOffset = [&]() {
      // Test extractColumn from offset.
      auto result = BaseVector::create(expected->type(), size, pool_.get());
      container.extractColumn(
          rows.data(), size - offset, column, result, offset);
      testEqualVectors(result, expected, offset, 0);
    };

    auto testRowNumbers = [&]() {
      auto result = BaseVector::create(expected->type(), size, pool_.get());

      std::vector<vector_size_t> rowNumbers;
      rowNumbers.resize(size - offset);
      for (int i = 0; i < size - offset; i++) {
        rowNumbers[i] = i + offset;
      }
      folly::Range<const vector_size_t*> rowNumbersRange =
          folly::Range(rowNumbers.data(), size - offset);
      container.extractColumn(
          rows.data(), rowNumbersRange, column, result, offset);
      testEqualVectors(result, expected, offset, offset);
    };

    // Test using extractColumn API.
    testBasic();
    testBasicWithOffset();
    // Test using extractColumn (with rowNumbers) API.
    testRowNumbers();
  }

  void testOutOfOrderRowNumbers(
      RowContainer& container,
      const std::vector<char*>& rows,
      int column,
      const VectorPtr& expected,
      vector_size_t offset) {
    auto size = rows.size();

    // In this test the extractColumn API is operated on an out-of-order
    // input (either the base input data or the rowNumbers). The new ordering
    // has null in even positions and retains odd positions from the original
    // input.

    // This is a helper method for result verification for the out of order
    // results.
    auto verifyResults = [&](const VectorPtr& result, int offset = 0) {
      EXPECT_EQ(size, result->size());
      for (vector_size_t i = offset; i < size; ++i) {
        if (i % 2 == 0) {
          EXPECT_TRUE(result->isNullAt(i)) << "at " << i;
        } else {
          EXPECT_TRUE(expected->equalValueAt(result.get(), i, i))
              << "at " << i << ": expected " << expected->toString(i)
              << ", got " << result->toString();
        }
      }
    };

    auto testBasic = [&]() {
      // Construct an input source with null in even
      // positions and that retains the original rows in
      // odd positions. Copy its rows
      std::vector<char*> input(size, nullptr);
      for (auto i = 1; i < size; i += 2) {
        input[i] = rows[i];
      }

      // Test the extractColumn API that didn't use the offset parameter
      // and copies to the start of the result vector.
      auto result = BaseVector::create(expected->type(), size, pool_.get());
      container.extractColumn(input.data(), size, column, result);
      verifyResults(result);

      // Test extractColumn from offset.
      container.extractColumn(
          input.data() + offset, size - offset, column, result, offset);
      verifyResults(result, offset);
    };

    auto testRowNumbers = [&]() {
      // Setup input rows vector so that the first row is null.
      std::vector<char*> input(rows);
      input[0] = nullptr;

      // The rowNumbersBuffer has values like 0, 1, 0, 3, 0, 5, 0, 7, etc
      // This tests the case of column extraction where the rowNumber values
      // are repeated and are also out of order.
      std::vector<vector_size_t> rowNumbers;
      auto rowNumbersSize = size - offset;
      rowNumbers.resize(rowNumbersSize);

      for (int i = 0; i < rowNumbersSize; i++) {
        if ((i + offset) % 2 == 0) {
          rowNumbers[i] = 0;
        } else {
          rowNumbers[i] = i + offset;
        }
      }

      auto result = BaseVector::create(expected->type(), size, pool_.get());
      folly::Range<const vector_size_t*> rowNumbersRange =
          folly::Range(rowNumbers.data(), rowNumbersSize);
      container.extractColumn(
          input.data(), rowNumbersRange, column, result, offset);
      verifyResults(result, offset);
    };

    testBasic();
    testRowNumbers();
  }

  void testRepeatedRowNumbers(
      RowContainer& container,
      const std::vector<char*>& rows,
      int column,
      const VectorPtr& expected) {
    auto size = rows.size();
    // Copy the rows so that we rotate over the values at positions 0, 1, 2.
    // This is for testing copying repeated row numbers.
    std::vector<vector_size_t> rowNumbers(size, 0);
    rowNumbers.resize(size);
    for (int i = 0; i < size; i++) {
      rowNumbers[i] = i % 3;
    }
    auto rowNumbersRange = folly::Range(rowNumbers.data(), size);

    auto result = BaseVector::create(expected->type(), size, pool_.get());
    container.extractColumn(rows.data(), rowNumbersRange, column, result);

    EXPECT_EQ(size, result->size());
    for (vector_size_t i = 0; i < size; ++i) {
      EXPECT_TRUE(result->equalValueAt(expected.get(), i, i % 3))
          << "at " << i << ": expected " << expected->toString(i % 3)
          << ", got " << result->toString(i);
    }
  }

  void testNegativeRowNumbers(
      RowContainer& container,
      const std::vector<char*>& rows,
      int column,
      const VectorPtr& expected) {
    auto size = rows.size();
    // Negative row numbers result in nullptr being copied to that position.
    // Alternate between nulls at odd positions and a copy of the values
    // at even positions.
    std::vector<vector_size_t> rowNumbers;
    rowNumbers.resize(size);
    for (int i = 0; i < size; i++) {
      if (i % 2) {
        rowNumbers[i] = -1 * i;
      } else {
        rowNumbers[i] = i;
      }
    }
    auto rowNumbersRange = folly::Range(rowNumbers.data(), size);

    auto result = BaseVector::create(expected->type(), size, pool_.get());
    container.extractColumn(rows.data(), rowNumbersRange, column, result);

    EXPECT_EQ(size, result->size());
    EXPECT_TRUE(result->equalValueAt(expected.get(), 0, 0))
        << "at 0 expected " << expected->toString(0) << ", got "
        << result->toString(0);
    for (vector_size_t i = 1; i < size; ++i) {
      if (i % 2) {
        EXPECT_TRUE(result->isNullAt(i))
            << "at " << i << "expected null, got " << result->toString(i);
      } else {
        EXPECT_TRUE(result->equalValueAt(expected.get(), i, i))
            << "at " << i << ": expected " << expected->toString(i) << ", got "
            << result->toString(i);
      }
    }
  }

  void checkSizes(std::vector<char*>& rows, RowContainer& rowContainer) {
    int64_t sum = 0;
    for (auto row : rows) {
      sum += rowContainer.rowSize(row) - rowContainer.fixedRowSize();
    }
    auto usage = rowContainer.stringAllocator().currentBytes();
    if (rowContainer.testingRowPointers().empty()) {
      EXPECT_EQ(usage, sum);
    }
  }

  std::vector<char*> store(
      RowContainer& rowContainer,
      const RowVectorPtr& data) {
    std::vector<DecodedVector> decodedVectors;
    for (auto& vector : data->children()) {
      decodedVectors.emplace_back(*vector);
    }

    std::vector<char*> rows;
    for (auto i = 0; i < data->size(); ++i) {
      auto* row = rowContainer.newRow();
      rows.push_back(row);

      for (auto j = 0; j < decodedVectors.size(); ++j) {
        rowContainer.store(decodedVectors[j], i, row, j);
      }
    }

    return rows;
  }

  std::vector<char*> store(
      RowContainer& rowContainer,
      const DecodedVector& decodedVector,
      vector_size_t size) {
    std::vector<char*> rows(size);
    for (size_t row = 0; row < size; ++row) {
      rows[row] = rowContainer.newRow();
      rowContainer.store(decodedVector, row, rows[row], 0);
    }
    return rows;
  }

  // Like store(), but makes all the rows first and then stores one column at a
  // time with the batch store().
  std::vector<char*> storeColumns(
      RowContainer& rowContainer,
      const RowVectorPtr& data) {
    std::vector<char*> rows(data->size());
    for (auto& row : rows) {
      row = rowContainer.newRow();
    }
    for (auto i = 0; i < data->childrenSize(); ++i) {
      DecodedVector decoded(*data->childAt(i));
      rowContainer.store(decoded, folly::Range(rows.data(), rows.size()), i);
    }
    return rows;
  }

  // Returns a batch of 'rowType' in which every 11th value of the 'string_val'
  // column is a large string.
  RowVectorPtr makeBatchWithLargeStrings(
      const RowTypePtr& rowType,
      vector_size_t numRows) {
    return makeDataset(rowType, numRows, [](RowVectorPtr rows) {
      auto strings =
          rows->childAt(rows->type()->asRow().getChildIdx("string_val"))
              ->as<FlatVector<StringView>>();
      for (auto i = 0; i < strings->size(); i += 11) {
        std::string chars;
        makeLargeString(i * 10000, chars);
        strings->set(i, StringView(chars));
      }
    });
  }

  // Stores 'batch' in the rows of 'rowContainer', which has as many rows as
  // 'batch' and non-nullable keys. Makes the key columns of 'batch' non-null
  // first. Returns the rows.
  std::vector<char*> storeKeysAndDependents(
      const RowVectorPtr& batch,
      RowContainer& rowContainer) {
    const auto numRows = batch->size();
    std::vector<char*> rows(numRows);
    RowContainerIterator iter;
    EXPECT_EQ(rowContainer.listRows(&iter, std::span(rows)), numRows);
    EXPECT_EQ(rowContainer.listRows(&iter, std::span(rows)), 0);

    checkSizes(rows, rowContainer);
    for (auto column = 0; column < batch->childrenSize(); ++column) {
      if (column < rowContainer.keyTypes().size()) {
        makeNonNull(batch, column);
        EXPECT_EQ(rowContainer.columnAt(column).nullMask(), 0);
      } else {
        EXPECT_NE(rowContainer.columnAt(column).nullMask(), 0);
      }
      DecodedVector decoded(*batch->childAt(column));
      for (auto index = 0; index < numRows; ++index) {
        rowContainer.store(decoded, index, rows[index], column);
      }
    }
    checkSizes(rows, rowContainer);
    rowContainer.checkConsistency();
    return rows;
  }

  // Stores the input vector in Row Container, extracts it and compares. Returns
  // the container.
  std::unique_ptr<RowContainer> roundTrip(const VectorPtr& input) {
    // Create row container.
    std::vector<TypePtr> types{input->type()};

    // Store the vector in the rowContainer.
    auto rowContainer = std::make_unique<RowContainer>(types, pool());
    auto size = input->size();
    DecodedVector decoded(*input);
    auto rows = store(*rowContainer, decoded, size);

    testExtractColumn(*rowContainer, rows, 0, input);
    return rowContainer;
  }

  // Count number of nulls at the start of the list.
  template <typename T>
  static size_t countLeadingNulls(const std::vector<std::optional<T>>& values) {
    for (auto i = 0; i < values.size(); ++i) {
      if (values[i].has_value()) {
        return i;
      }
    }
    return values.size();
  }

  // Given a list of values sorted in ascending order with nulls first re-orders
  // the elements to reverse the order and / or put nulls last.
  template <typename T>
  void changeSortingOrder(
      std::vector<std::optional<T>>& values,
      const CompareFlags& flags) {
    if (flags.ascending) {
      if (flags.nullsFirst) {
        // Nothing to do. 'values' are expected to be already sorted in
        // ascending order with nulls first.
      } else {
        // Move the nulls from the start of the list to the end of the list.
        const auto numNulls = countLeadingNulls(values);
        for (auto i = 0; i < values.size() - numNulls; ++i) {
          values[i] = values[i + numNulls];
        }

        for (auto i = values.size() - numNulls; i < values.size(); ++i) {
          values[i].reset();
        }
      }
    } else {
      if (flags.nullsFirst) {
        // Keeps nulls at the start of the list. Reverse the order of remaining
        // non-null elements.
        const auto numNulls = countLeadingNulls(values);
        std::reverse(values.begin() + numNulls, values.end());
      } else {
        // Reverse the order of all elements.
        std::reverse(values.begin(), values.end());
      }
    }
  }

  using IndexAndRow = std::pair<vector_size_t, char*>;
  // A list of rows with unique zero-based indices.
  using IndexAndRowVector = std::vector<IndexAndRow>;

  static IndexAndRowVector indexRows(const std::vector<char*>& rows) {
    IndexAndRowVector indexedRows;
    indexedRows.reserve(rows.size());
    for (auto i = 0; i < rows.size(); ++i) {
      indexedRows.push_back({i, rows[i]});
    }
    return indexedRows;
  }

  VectorPtr copyValues(
      const IndexAndRowVector& indexedRows,
      const VectorPtr& values) {
    const auto numRows = indexedRows.size();
    auto copy = BaseVector::create(values->type(), numRows, pool_.get());
    for (auto i = 0; i < numRows; ++i) {
      copy->copy(values.get(), i, indexedRows[i].first, 1);
    }
    return copy;
  }

  void testRowContainerCompareAPIs(
      const TypePtr& type,
      const VectorPtr& values,
      const VectorPtr& expected,
      std::optional<CompareFlags> flags) {
    // If no flags provided then it must be the default of {true, true}.
    SCOPED_TRACE(
        fmt::format(
            "{}, ascending = {}, nullsFirst = {}",
            type->toString(),
            flags.has_value() ? flags.value().ascending : true,
            flags.has_value() ? flags.value().nullsFirst : true));

    RowContainer rowContainer({type}, {type}, pool());
    int numRows = values->size();

    DecodedVector decoded(*values);
    auto rows = store(rowContainer, decoded, numRows);
    auto indexedRows = indexRows(rows);

    std::function<int(const char*, const char*)> compareRows =
        [&](const auto& l, const auto& r) {
          return rowContainer.compare(l, r, 0) < 0;
        };

    // If compare flags are specified pass them into the compare function.
    if (flags.has_value()) {
      compareRows = [&](const auto& l, const auto& r) {
        return rowContainer.compare(l, r, 0, flags.value()) < 0;
      };
    }

    // Verify compare method with two rows as input - using compareRows
    {
      std::sort(rows.begin(), rows.end(), compareRows);
      VectorPtr result = BaseVector::create(type, numRows, pool_.get());
      rowContainer.extractColumn(rows.data(), numRows, 0, result);
      assertEqualVectors(expected, result);
    }

    std::function<int(
        const std::pair<int, char*>&, const std::pair<int, char*>&)>
        compareDecodedAndRows = [&](const auto& l, const auto& r) {
          return rowContainer.compare(
                     l.second, rowContainer.columnAt(0), decoded, r.first) < 0;
        };

    if (flags.has_value()) {
      compareDecodedAndRows = [&](const auto& l, const auto& r) {
        return rowContainer.compare(
                   l.second,
                   rowContainer.columnAt(0),
                   decoded,
                   r.first,
                   flags.value()) < 0;
      };
    }

    // Verify compare method with row and decoded vector as input - using
    // compareDecodedAndRows.
    // This test assumes that the rows and decoded vector are in the same order
    // so that indexRows can be sorted. The decoded vector is a different
    // representation of the same row data. This allows for using sort which
    // compares each element in decoded vector and each row value to each other
    // testing the compare API as if the same row is sorted.
    {
      std::sort(indexedRows.begin(), indexedRows.end(), compareDecodedAndRows);
      auto result = copyValues(indexedRows, values);
      assertEqualVectors(expected, result);
    }
  }

  template <bool canHandleNulls>
  void testRowContainerEqualsAPI(
      const TypePtr& type,
      const VectorPtr& lhs,
      const VectorPtr& rhs,
      const std::vector<bool>& expected) {
    auto numRows = lhs->size();
    EXPECT_EQ(numRows, rhs->size());
    EXPECT_EQ(numRows, expected.size());

    RowContainer rowContainer({type}, {type}, pool());
    DecodedVector lhsDecoded(*lhs);
    auto rows = store(rowContainer, lhsDecoded, numRows);

    DecodedVector rhsDecoded(*rhs);

    int32_t index{0};
    for (auto row : rows) {
      ASSERT_EQ(
          expected[index],
          rowContainer.compare<canHandleNulls>(
              row, rowContainer.columnAt(0), rhsDecoded, index) == 0)
          << fmt::format(
                 "Mismatch at index {} with canHandleNulls {}",
                 index,
                 canHandleNulls);
      ++index;
    }
  }

  void testRowContainerEqualityEdgeValues(
      const TypePtr& type,
      const VectorPtr& values) {
    auto numRows = values->size();
    if (values->mayHaveNulls()) {
      auto numNulls = BaseVector::countNulls(values->nulls(), 0, numRows);
      auto vecWithoutNulls = values->slice(numNulls, numRows - numNulls);
      std::vector<bool> expectedResults(vecWithoutNulls->size(), true);
      testRowContainerEqualsAPI<false>(
          type, vecWithoutNulls, vecWithoutNulls, expectedResults);
    } else {
      std::vector<bool> expectedResults(numRows, true);
      testRowContainerEqualsAPI<true>(type, values, values, expectedResults);
    }
  }

  template <typename T>
  void testOrderAndEqualsWithNullsFirstVariations(
      const TypePtr& type,
      const VectorPtr& values,
      const std::vector<std::optional<T>>& ascNullsFirstOrder,
      std::function<VectorPtr(const std::vector<std::optional<T>>&)>
          generateExpectedVector) {
    // The default flags are ascending == true, nullsFirst == true.
    // std::nullopt means use the default for the compare function.
    const std::vector<std::optional<CompareFlags>> compareFlags{
        {{std::nullopt}}, {{true, false}}, {{false, true}}, {{false, false}}};

    for (auto flags : compareFlags) {
      std::vector<std::optional<T>> expectedOrder{ascNullsFirstOrder};
      // The expectedOrder is ascNullsFirstOrder for the case
      // where no compare flags are provided.
      if (flags.has_value()) {
        changeSortingOrder<T>(expectedOrder, flags.value());
      }
      auto expected = generateExpectedVector(expectedOrder);

      testRowContainerCompareAPIs(type, values, expected, flags);
    }

    testRowContainerEqualityEdgeValues(type, values);
  }

#define TEST_FLOATING_TYPE_LIMIT_VARIABLES        \
  auto lowest = std::numeric_limits<T>::lowest(); \
  auto min = std::numeric_limits<T>::min();       \
  auto max = std::numeric_limits<T>::max();       \
  auto nan = std::numeric_limits<T>::quiet_NaN(); \
  auto inf = std::numeric_limits<T>::infinity()

  template <typename T>
  void testComparePrimitiveFloats(const TypePtr& type) {
    TEST_FLOATING_TYPE_LIMIT_VARIABLES;

    auto values = makeNullableFlatVector<T>(
        {std::nullopt, max, inf, 0.0, nan, lowest, min});

    std::vector<std::optional<T>> ascNullsFirstOrder = {
        std::nullopt, lowest, 0.0, min, max, inf, nan};

    testOrderAndEqualsWithNullsFirstVariations<T>(
        type, values, ascNullsFirstOrder, [&](const auto& expectedOrder) {
          return makeNullableFlatVector<T>(expectedOrder);
        });
  }

  /// This function constructions a vector of random floating point
  /// values including special floating point values as well as an
  /// boolean result vector. The result vector contains the expected
  /// boolean result if a value at data vector position i is compared
  /// with a NaN.
  template <typename T>
  void populateFloatingPointTestVectorsForNaNComparison(
      int32_t numRows,
      std::vector<std::optional<T>>& rawData,
      std::vector<bool>& rawExpected) {
    TEST_FLOATING_TYPE_LIMIT_VARIABLES;
    std::mt19937 gen(numRows);

    std::vector<std::optional<T>> rawSpecialValues = {
        std::nullopt, min, 0.0, lowest, max, inf};
    std::vector<bool> rawExpectedSpecialValues(rawSpecialValues.size(), false);
    // Make sure the null value is at the beginning of the dataset.
    // This knowledge is later used to build vectors without null values.
    rawData.insert(
        rawData.end(), rawSpecialValues.begin(), rawSpecialValues.end());
    rawExpected.insert(
        rawExpected.end(),
        rawExpectedSpecialValues.begin(),
        rawExpectedSpecialValues.end());

    while (numRows) {
      auto value = folly::Random::randDouble(min, max, gen);
      // Intersperse nan values.
      if (static_cast<int64_t>(std::fmod(value, 3.0)) == 0) {
        rawData.push_back(nan);
        rawExpected.push_back(true);
      } else {
        rawData.push_back(value);
        rawExpected.push_back(false);
      }
      --numRows;
    }
  }

  template <typename T>
  void runTestNanEquals(
      const TypePtr& type,
      const VectorPtr& lhs,
      const VectorPtr& rhs,
      const std::vector<bool>& rawExpected) {
    auto numRows = lhs->size();
    EXPECT_EQ(numRows, rawExpected.size());
    EXPECT_EQ(numRows, rhs->size());

    testRowContainerEqualsAPI<true>(type, lhs, rhs, rawExpected);

    // Remove nulls and re-run.
    auto numNulls = BaseVector::countNulls(lhs->nulls(), 0, numRows);
    auto lhsNoNulls = lhs->slice(numNulls, numRows - numNulls);
    auto rhsNoNulls = rhs->slice(numNulls, numRows - numNulls);
    std::vector<bool> rawExpectedNoNulls(
        rawExpected.cbegin() + numNulls, rawExpected.cend());
    EXPECT_EQ(lhsNoNulls->size(), rawExpectedNoNulls.size());
    testRowContainerEqualsAPI<false>(
        type, lhsNoNulls, rhsNoNulls, rawExpectedNoNulls);
  }

  template <typename T>
  void testNanEqualsPrimitiveFloats(const TypePtr& type) {
    constexpr auto nan = std::numeric_limits<T>::quiet_NaN();
    constexpr int32_t kNumRows = 100;
    std::vector<std::optional<T>> rawValues;
    std::vector<bool> rawExpected;

    populateFloatingPointTestVectorsForNaNComparison<T>(
        kNumRows, rawValues, rawExpected);
    auto nanVector = makeConstant<T>(nan, rawValues.size());
    auto values = makeNullableFlatVector<T>(rawValues);
    runTestNanEquals<T>(type, values, nanVector, rawExpected);
  }

  template <typename T>
  void testCompareArrayFloats(const TypePtr& type) {
    TEST_FLOATING_TYPE_LIMIT_VARIABLES;

    auto values = makeNullableArrayVector<T>(
        {std::nullopt,
         {{std::nullopt}},
         {{max}},
         {{inf}},
         {{0.0}},
         {{nan}},
         {{lowest}},
         {{min}}});

    std::vector<std::optional<std::vector<std::optional<T>>>>
        ascNullsFirstOrder = {
            std::nullopt,
            {{std::nullopt}},
            {{lowest}},
            {{0.0}},
            {{min}},
            {{max}},
            {{inf}},
            {{nan}}};

    testOrderAndEqualsWithNullsFirstVariations<std::vector<std::optional<T>>>(
        type, values, ascNullsFirstOrder, [&](const auto& expectedOrder) {
          return makeNullableArrayVector<T>(expectedOrder);
        });
  }

  template <typename T>
  void testNanEqualsArrayFloats(const TypePtr& type) {
    constexpr auto nan = std::numeric_limits<T>::quiet_NaN();
    constexpr int32_t kNumRows = 100;
    std::vector<std::optional<T>> rawValues;
    std::vector<bool> rawExpected;
    std::vector<std::optional<std::vector<std::optional<T>>>> rawArrayData;
    std::vector<std::vector<T>> rawNanArrayData;

    rawArrayData.push_back(std::nullopt);
    rawExpected.push_back(false);
    rawNanArrayData.push_back({nan});

    populateFloatingPointTestVectorsForNaNComparison<T>(
        kNumRows, rawValues, rawExpected);

    for (auto value : rawValues) {
      std::vector<std::optional<T>> temp = {value};
      rawArrayData.push_back(temp);
      rawNanArrayData.push_back({nan});
    }

    auto arrayData = makeNullableArrayVector<T>(rawArrayData);
    auto nanArrayData = makeArrayVector<T>(rawNanArrayData);
    runTestNanEquals<T>(type, arrayData, nanArrayData, rawExpected);
  }

  template <typename T>
  void testCompareMapFloats(const TypePtr& type) {
    TEST_FLOATING_TYPE_LIMIT_VARIABLES;

    auto values = makeNullableMapVector<T, int32_t>(
        {{{std::nullopt}},
         {{{max, 1}}},
         {{{inf, 2}}},
         {{{0.0, 3}}},
         {{{nan, 4}}},
         {{{lowest, 5}}},
         {{{min, 6}}}});

    std::vector<
        std::optional<std::vector<std::pair<T, std::optional<int32_t>>>>>
        ascNullsFirstOrder{
            {{std::nullopt}},
            {{{lowest, 5}}},
            {{{0.0, 3}}},
            {{{min, 6}}},
            {{{max, 1}}},
            {{{inf, 2}}},
            {{{nan, 4}}}};

    testOrderAndEqualsWithNullsFirstVariations<
        std::vector<std::pair<T, std::optional<int32_t>>>>(
        type, values, ascNullsFirstOrder, [&](const auto& expectedOrder) {
          return makeNullableMapVector<T, int32_t>(expectedOrder);
        });
  }

  template <typename T>
  void testNanEqualsMapFloats(const TypePtr& type) {
    constexpr auto nan = std::numeric_limits<T>::quiet_NaN();
    constexpr int32_t kNumRows = 100;
    std::vector<std::optional<T>> rawValues;
    std::vector<bool> rawExpected;
    std::vector<
        std::optional<std::vector<std::pair<T, std::optional<int32_t>>>>>
        rawMapData;
    std::vector<std::vector<std::pair<T, std::optional<int32_t>>>>
        rawNanMapData;

    rawMapData.push_back({std::nullopt});
    rawExpected.push_back(false);
    rawNanMapData.push_back({{{nan, std::nullopt}}});

    populateFloatingPointTestVectorsForNaNComparison<T>(
        kNumRows, rawValues, rawExpected);
    // Remove nulls in data because keys cannot be null.
    auto numNulls = countLeadingNulls(rawValues);
    rawValues.erase(rawValues.cbegin(), rawValues.cbegin() + numNulls);
    rawExpected.erase(rawExpected.cbegin(), rawExpected.cbegin() + numNulls);
    std::optional<int32_t> second = 0;
    for (auto value : rawValues) {
      std::vector<std::pair<T, std::optional<int32_t>>> temp{
          {{value.value(), second}}};
      rawMapData.push_back(temp);
      rawNanMapData.push_back({{{nan, second}}});
      ++second.value();
    }

    auto mapData = makeNullableMapVector<T, int32_t>(rawMapData);
    auto nanMapData = makeMapVector<T, int32_t>(rawNanMapData);
    runTestNanEquals<T>(type, mapData, nanMapData, rawExpected);
  }

  template <typename T>
  void testCompareRowFloats(const TypePtr& type) {
    TEST_FLOATING_TYPE_LIMIT_VARIABLES;

    auto values = makeRowVector({makeNullableFlatVector<T>(
        {std::nullopt, max, inf, 0.0, nan, lowest, min})});

    std::vector<std::optional<T>> ascNullsFirstOrder = {
        std::nullopt, lowest, 0.0, min, max, inf, nan};

    testOrderAndEqualsWithNullsFirstVariations<T>(
        type, values, ascNullsFirstOrder, [&](const auto& expectedOrder) {
          return makeRowVector({makeNullableFlatVector<T>(expectedOrder)});
        });
  }

  template <typename T>
  void testNanEqualsRowFloats(const TypePtr& type) {
    constexpr auto nan = std::numeric_limits<T>::quiet_NaN();
    constexpr int32_t kNumRows = 100;
    std::vector<std::optional<T>> rawValues;
    std::vector<bool> rawExpected;

    populateFloatingPointTestVectorsForNaNComparison<T>(
        kNumRows, rawValues, rawExpected);

    auto rowData = makeRowVector({makeNullableFlatVector<T>(rawValues)});
    auto nanRowData = makeRowVector({makeConstant<T>(nan, rawValues.size())});
    runTestNanEquals<T>(type, rowData, nanRowData, rawExpected);
  }
#undef TEST_FLOATING_TYPE_LIMIT_VARIABLES

  template <typename T>
  void testEqualAndCompareRowContainerTypeFloat(const TypePtr& type) {
    if (type->isPrimitiveType()) {
      testComparePrimitiveFloats<T>(type);
      testNanEqualsPrimitiveFloats<T>(type);
    } else if (type->isArray()) {
      testCompareArrayFloats<T>(type);
      testNanEqualsArrayFloats<T>(type);
    } else if (type->isMap()) {
      testCompareMapFloats<T>(type);
      testNanEqualsMapFloats<T>(type);
    } else if (type->isRow()) {
      testCompareRowFloats<T>(type);
      testNanEqualsRowFloats<T>(type);
    }
  }
};

TEST_P(RowContainerTest, extractWithNullsAndTargetOffset) {
  constexpr int32_t kNumRows = 100;
  // The second column must have no nulls in the first batch.
  auto rowVector1 = makeRowVector({
      makeFlatVector<bool>(
          kNumRows, [](auto row) { return row % 11 + 1; }, nullEvery(17)),
      makeFlatVector<StringView>(
          kNumRows, [](auto /* row */) { return StringView("abcd"); }),
      makeFlatVector<int8_t>(
          kNumRows, [](auto /* row */) { return 4; }, nullEvery(1)),
  });
  // The second column must have at least one null in the second batch.
  auto rowVector2 = makeRowVector({
      makeFlatVector<bool>(
          kNumRows, [](auto row) { return row % 11 + 1; }, nullEvery(23)),
      makeFlatVector<StringView>(
          kNumRows,
          [](auto /* row */) { return StringView(""); },
          nullEvery(3)),
      makeFlatVector<int8_t>(
          kNumRows, [](auto /* row */) { return 5; }, nullEvery(5)),
  });

  // Create and fill up two row containers from two row vectors.
  const auto rowType = makeRowType({BOOLEAN(), VARCHAR(), TINYINT()});

  auto makeContainer = [&]() {
    return RowContainer(
        {},
        {},
        rowType->children(),
        pool(),
        {.useListRowIndex = useListRowIndex()});
  };

  RowContainer data1 = makeContainer();
  RowContainer data2 = makeContainer();
  const auto rows1 = storeColumns(data1, rowVector1);
  const auto rows2 = storeColumns(data2, rowVector2);

  // Now create the result row vector and extract two row containers into it.
  auto result =
      BaseVector::create<RowVector>(rowType, kNumRows * 2, pool_.get());

  for (int32_t col = 0; col < 3; col++) {
    data1.extractColumn(rows1.data(), kNumRows, col, result->childAt(col));
  }
  for (int32_t col = 0; col < 3; col++) {
    data2.extractColumn(
        rows2.data(), kNumRows, col, result->childAt(col), kNumRows);
  }

  // Check we have all nulls correct.
  ASSERT_EQ(result->size(), kNumRows * 2);
  for (int32_t col = 0; col < 3; col++) {
    auto resultColVector = result->childAt(col);
    auto batch1ColVector = rowVector1->childAt(col);
    auto batch2ColVector = rowVector2->childAt(col);
    for (int32_t row = 0; row < kNumRows; row++) {
      ASSERT_EQ(resultColVector->isNullAt(row), batch1ColVector->isNullAt(row));
      ASSERT_EQ(
          resultColVector->isNullAt(row + kNumRows),
          batch2ColVector->isNullAt(row));
    }
  }
}

TEST_F(RowContainerTest, storeExtractArrayOfVarchar) {
  // Make an array-of-strings vector with two rows each having 2 elements.
  // Here it is important, that one of the 1st row's elements has more than 12
  // characters (14 and 13 in this case), since 12 is the number of chars being
  // inlined in the String View.
  auto input = makeArrayVector<StringView>(
      {{"aaaaaaaaaaaaaa", "bbbbbbbbbbbbb"}, {"cccccccc", "ddddd"}});
  roundTrip(input);
}

TEST_P(RowContainerTest, types) {
  constexpr int32_t kNumRows = 100;
  auto batch = makeBatchWithLargeStrings(
      ROW(
          {{"bool_val", BOOLEAN()},
           {"tiny_val", TINYINT()},
           {"small_val", SMALLINT()},
           {"int_val", INTEGER()},
           {"long_val", BIGINT()},
           {"float_val", REAL()},
           {"double_val", DOUBLE()},
           {"long_decimal_val", DECIMAL(20, 3)},
           {"short_decimal_val", DECIMAL(10, 2)},
           {"string_val", VARCHAR()},
           {"array_val", ARRAY(VARCHAR())},
           {"struct_val",
            ROW({{"s_int", INTEGER()}, {"s_array", ARRAY(REAL())}})},
           {"map_val",
            MAP(VARCHAR(),
                MAP(BIGINT(),
                    ROW({{"s2_int", INTEGER()}, {"s2_string", VARCHAR()}})))},

           {"bool_val2", BOOLEAN()},
           {"tiny_val2", TINYINT()},
           {"small_val2", SMALLINT()},
           {"int_val2", INTEGER()},
           {"long_val2", BIGINT()},
           {"float_val2", REAL()},
           {"double_val2", DOUBLE()},
           {"long_decimal_val2", DECIMAL(20, 3)},
           {"short_decimal_val2", DECIMAL(10, 2)},
           {"string_val2", VARCHAR()},
           {"array_val2", ARRAY(VARCHAR())},
           {"struct_val2",
            ROW({{"s_int", INTEGER()}, {"s_array", ARRAY(REAL())}})},
           {"map_val2",
            MAP(VARCHAR(),
                MAP(BIGINT(),
                    ROW({{"s2_int", INTEGER()}, {"s2_string", VARCHAR()}})))}}),
      kNumRows);
  // The same types twice: the first half as non-null keys, the second half as
  // nullable dependents.
  const auto& types = batch->rowType()->children();
  const auto numKeys = types.size() / 2;
  const std::vector<TypePtr> keys(types.begin(), types.begin() + numKeys);
  const std::vector<TypePtr> dependents(types.begin() + numKeys, types.end());
  RowContainer rowContainer(
      keys,
      {},
      dependents,
      pool(),
      {.nullableKeys = false,
       .hasNext = true,
       .hasNormalizedKey = true,
       .useListRowIndex = useListRowIndex()});

  EXPECT_GT(rowContainer.nextOffset(), 0);
  for (int i = 0; i < kNumRows; ++i) {
    auto row = rowContainer.newRow();
    if (i < kNumRows / 2) {
      RowContainer::normalizedKey(row) = i;
    } else {
      rowContainer.disableNormalizedKeys();
    }
  }
  EXPECT_EQ(kNumRows, rowContainer.numRows());
  auto rows = storeKeysAndDependents(batch, rowContainer);
  SelectivityVector allRows(kNumRows);

  raw_vector<uint64_t> expectedHashes(kNumRows);
  for (auto column = 0; column < keys.size(); ++column) {
    VectorHasher hasher(keys[column], column);
    hasher.decode(*batch->childAt(column), allRows);
    hasher.hash(allRows, /*mix=*/column > 0, expectedHashes);
  }
  std::vector<uint64_t> rowHashes(kNumRows);
  rowContainer.hashKeys(rows, rowHashes);
  for (auto i = 0; i < kNumRows; ++i) {
    EXPECT_EQ(expectedHashes[i], rowHashes[i]);
  }

  auto copy =
      BaseVector::create<RowVector>(batch->type(), batch->size(), pool_.get());
  for (auto column = 0; column < batch->childrenSize(); ++column) {
    testExtractColumn(rowContainer, rows, column, batch->childAt(column));

    auto extracted = copy->childAt(column);
    extracted->resize(kNumRows);
    rowContainer.extractColumn(rows.data(), rows.size(), column, extracted);
    raw_vector<uint64_t> hashes(kNumRows);
    auto source = batch->childAt(column);
    auto columnType = batch->type()->as<TypeKind::ROW>().childAt(column);
    VectorHasher hasher(columnType, column);
    hasher.decode(*source, allRows);
    hasher.hash(allRows, false, hashes);
    DecodedVector decoded(*extracted, allRows);

    // The RowContainer has corresponding key and dependent columns
    // of each type.
    // This pairing of columns can be used to test the compare API
    // of 2 rows with 2 different columns.
    column_index_t dependantColumn =
        column < keys.size() ? column + keys.size() : column - keys.size();
    for (auto i = 0; i < kNumRows; ++i) {
      EXPECT_TRUE(source->equalValueAt(extracted.get(), i, i));
      EXPECT_EQ(source->compare(extracted.get(), i, i), 0);
      EXPECT_EQ(source->hashValueAt(i), hashes[i]);
      // Test non-null and nullable variants of equals.
      if (column < keys.size()) {
        EXPECT_EQ(
            rowContainer.compare<false>(
                rows[i], rowContainer.columnAt(column), decoded, i),
            0);
      } else if (!columnType->isMap()) {
        EXPECT_EQ(
            rowContainer.compare<true>(
                rows[i], rowContainer.columnAt(column), decoded, i),
            0);
      }
      // Non-key map columns are not comparable, as the map keys are not sorted.
      if (columnType->isMap() && column >= keys.size()) {
        continue;
      }
      if (i > 0) {
        // We compare the values on consecutive rows.
        auto result = sign(source->compare(source.get(), i, i - 1));
        EXPECT_EQ(
            result,
            sign(rowContainer.compare(
                rows[i], rowContainer.columnAt(column), decoded, i - 1)));
        EXPECT_EQ(
            result, sign(rowContainer.compare(rows[i], rows[i - 1], column)));
        EXPECT_EQ(
            result,
            sign(rowContainer.compare(rows[i], rows[i - 1], column, column)));

        // This variation of the API compares the values on consecutive rows
        // between the same column type (used as a key and as a dependent) in
        // the row structure.
        EXPECT_EQ(
            sign(rowContainer.compare(
                rows[i], rows[i - 1], column, dependantColumn)),
            -sign(rowContainer.compare(
                rows[i - 1], rows[i], dependantColumn, column)));
      }
    }
  }
  EXPECT_LT(0, rowContainer.usedVariableWidthBytes());
  EXPECT_LT(
      rowContainer.usedVariableWidthBytes(), rowContainer.allocatedBytes());
}

TEST_P(RowContainerTest, extractNulls) {
  constexpr int32_t kNumRows = 100;
  auto batch = makeRowVector({
      makeFlatVector<bool>(
          kNumRows, [](auto row) { return row % 5 == 0; }, nullEvery(2)),
      makeFlatVector<int8_t>(
          kNumRows, [](auto row) { return row % 5; }, nullEvery(3)),
      makeFlatVector<int16_t>(
          kNumRows, [](auto row) { return row % 5; }, nullEvery(4)),
      makeFlatVector<int32_t>(
          kNumRows, [](auto row) { return row % 5; }, nullEvery(5)),
      makeFlatVector<int64_t>(
          kNumRows, [](auto row) { return row % 5; }, nullEvery(6)),
      makeFlatVector<float>(
          kNumRows, [](auto row) { return row % 5; }, nullEvery(7)),
      makeFlatVector<double>(
          kNumRows, [](auto row) { return row % 5; }, nullEvery(8)),
      makeFlatVector<StringView>(
          kNumRows,
          [](auto /* row */) { return StringView("abcd"); },
          nullEvery(9)),
      makeArrayVector<int32_t>(
          kNumRows,
          [](auto i) { return i % 5; },
          [](auto i) { return i % 10; },
          nullEvery(10)),
      makeMapVector<int32_t, int32_t>(
          kNumRows,
          [](auto i) { return i % 5; },
          [](auto i) { return i % 10; },
          [](auto i) { return i % 3; },
          nullEvery(11)),
      makeRowVector(
          {"c0", "c1"},
          {makeFlatVector<int32_t>(kNumRows, [](auto i) { return i % 5; }),
           makeFlatVector<int32_t>(kNumRows, [](auto i) { return i % 7; })},
          nullEvery(12)),
  });

  RowContainer rowContainer(
      {},
      {},
      batch->rowType()->children(),
      pool(),
      {.useListRowIndex = useListRowIndex()});
  const auto rows = storeColumns(rowContainer, batch);

  auto nulls = allocateNulls(kNumRows, pool());
  auto rawNulls = nulls->as<uint64_t>();
  for (int i = 0; i < 11; i++) {
    const auto numNulls =
        rowContainer.extractNulls(rows.data(), kNumRows, i, nulls);

    auto nullEvery = i + 2;
    vector_size_t expectedNumNulls = 0;
    for (int j = 0; j < kNumRows; j += 1) {
      EXPECT_EQ(bits::isBitNull(rawNulls, j), j % nullEvery == 0);
      expectedNumNulls += j % nullEvery == 0;
    }
    EXPECT_EQ(numNulls, expectedNumNulls);
  }
}

TEST_F(RowContainerTest, erase) {
  constexpr int32_t kNumRows = 100;
  RowContainer rowContainer(
      {SMALLINT()},
      {},
      {SMALLINT()},
      pool(),
      {.nullableKeys = false, .hasNext = true});

  // The layout is expected to be smallint - 6 bytes of padding - 1 byte of bits
  // - smallint - next pointer. The bits are a null flag for the second
  // smallint and a free flag.
  EXPECT_EQ(rowContainer.nextOffset(), 11);
  std::unordered_set<char*> rowSet;
  std::vector<char*> rows;
  for (int i = 0; i < kNumRows; ++i) {
    rows.push_back(rowContainer.newRow());
    rowSet.insert(rows.back());
  }
  EXPECT_EQ(kNumRows, rowContainer.numRows());
  std::vector<char*> rowsFromContainer(kNumRows);
  RowContainerIterator iter;
  EXPECT_EQ(rowContainer.listRows(&iter, std::span<char*>{}), 0);
  EXPECT_EQ(
      rowContainer.listRows(&iter, std::span(rowsFromContainer)), kNumRows);
  EXPECT_EQ(0, rowContainer.listRows(&iter, std::span(rows)));
  EXPECT_EQ(rows, rowsFromContainer);

  std::vector<char*> erased;
  for (auto i = 0; i < rows.size(); i += 2) {
    erased.push_back(rows[i]);
  }
  rowContainer.eraseRows(folly::Range<char**>(erased.data(), erased.size()));
  rowContainer.checkConsistency();

  std::vector<char*> remaining(rowContainer.numRows());
  iter.reset();
  EXPECT_EQ(
      remaining.size(), rowContainer.listRows(&iter, std::span(remaining)));
  // We check that the next new rows reuse the erased rows.
  for (auto i = 0; i < erased.size(); ++i) {
    auto reused = rowContainer.newRow();
    EXPECT_NE(rowSet.end(), rowSet.find(reused));
  }
  // The next row will be a new one.
  auto newRow = rowContainer.newRow();
  EXPECT_EQ(rowSet.end(), rowSet.find(newRow));
  rowContainer.checkConsistency();
  rowContainer.clear();
  EXPECT_EQ(0, rowContainer.numRows());
  EXPECT_EQ(0, rowContainer.usedVariableWidthBytes());
  rowContainer.checkConsistency();
}

TEST_F(RowContainerTest, eraseRowTwice) {
  auto data = makeRowVector({
      makeFlatVector<std::string>({"non-inline string 1"}),
  });
  RowContainer rowContainer({VARCHAR()}, pool_.get());
  DecodedVector decoded(*data->childAt(0));
  char* row = rowContainer.newRow();
  rowContainer.store(decoded, 0, row, 0);

  std::vector<char*> rows{row, row};
  VELOX_ASSERT_THROW(
      rowContainer.eraseRows(folly::Range<char**>(rows.data(), rows.size())),
      "Double free of row");
}

TEST_P(RowContainerTest, initialNulls) {
  auto isNullAt =
      [](const RowContainer& rowContainer, const char* row, int32_t i) {
        return RowContainer::isNullAt(row, rowContainer.columnAt(i));
      };

  // Non-nullable keys, as in a join build.
  {
    RowContainer rowContainer(
        {INTEGER()},
        {},
        {INTEGER()},
        pool(),
        {.nullableKeys = false, .useListRowIndex = useListRowIndex()});
    auto* row = rowContainer.newRow();
    EXPECT_FALSE(isNullAt(rowContainer, row, 0));
    EXPECT_FALSE(isNullAt(rowContainer, row, 1));
  }

  // Nullable keys.
  {
    RowContainer rowContainer(
        {INTEGER()},
        {},
        {INTEGER()},
        pool(),
        {.useListRowIndex = useListRowIndex()});
    auto* row = rowContainer.newRow();
    EXPECT_FALSE(isNullAt(rowContainer, row, 0));
    EXPECT_FALSE(isNullAt(rowContainer, row, 1));
  }
}

TEST_F(RowContainerTest, rowSize) {
  constexpr int32_t kNumRows = 100;
  RowContainer rowContainer(
      {SMALLINT()}, {}, {VARCHAR()}, pool(), {.hasNext = true});

  // The layout is expected to be smallint - 6 bytes of padding - 1 byte of bits
  // - StringView - rowSize - next pointer. The bits are null flags for the
  // smallint and the StringView and a free flag.
  EXPECT_EQ(25, rowContainer.rowSizeOffset());
  EXPECT_EQ(29, rowContainer.nextOffset());

  std::vector<char*> rows;
  rows.reserve(kNumRows);
  for (int i = 0; i < kNumRows; ++i) {
    rows.push_back(rowContainer.newRow());
  }
  EXPECT_EQ(kNumRows, rowContainer.numRows());
  for (auto i = 0; i < rows.size(); ++i) {
    rowContainer.testingIncrementRowSize(rows[i], i * 1000);
  }
  // The first size overflows 32 bits.
  rowContainer.testingIncrementRowSize(rows[0], 2000000000);
  rowContainer.testingIncrementRowSize(rows[0], 2000000000);
  rowContainer.testingIncrementRowSize(rows[0], 2000000000);
  std::vector<char*> rowsFromContainer(kNumRows);
  RowContainerIterator iter;
  // The first row is returned alone.
  auto numRows =
      rowContainer.listRows(&iter, std::span(rowsFromContainer), 100000);
  EXPECT_EQ(1, numRows);
  while (numRows < kNumRows) {
    numRows += rowContainer.listRows(
        &iter, std::span(rowsFromContainer).subspan(numRows), 100000);
  }
  EXPECT_EQ(0, rowContainer.listRows(&iter, std::span(rows)));

  EXPECT_EQ(rows, rowsFromContainer);
}

TEST_P(RowContainerTest, columnSize) {
  const uint64_t kNumRows = 1000;
  RowContainer rowContainer(
      {BIGINT(), VARCHAR()},
      {},
      {BIGINT(), VARCHAR()},
      pool(),
      {.useListRowIndex = useListRowIndex()});

  VectorFuzzer fuzzer(
      {
          .vectorSize = kNumRows,
          .stringLength = 100,
          .stringVariableLength = true,
      },
      pool());

  auto rowVector =
      fuzzer.fuzzInputFlatRow(ROW({BIGINT(), VARCHAR(), BIGINT(), VARCHAR()}));

  ASSERT_EQ(rowContainer.numRows(), 0);
  const auto rows = store(rowContainer, rowVector);
  ASSERT_EQ(rowContainer.fixedSizeAt(0), 8);
  ASSERT_EQ(rowContainer.fixedSizeAt(2), 8);
  const auto key2Vector = rowVector->childAt(1)->asFlatVector<StringView>();
  const auto dep2Vector = rowVector->childAt(3)->asFlatVector<StringView>();
  for (size_t i = 0; i < kNumRows; i++) {
    ASSERT_EQ(
        rowContainer.variableSizeAt(rows[i], 1), key2Vector->valueAt(i).size());
    ASSERT_EQ(
        rowContainer.variableSizeAt(rows[i], 3), dep2Vector->valueAt(i).size());
  }
}

TEST_P(RowContainerTest, rowSizeWithNormalizedKey) {
  RowContainer rowContainer(
      {SMALLINT()},
      {},
      {VARCHAR()},
      pool(),
      {.hasNormalizedKey = true, .useListRowIndex = useListRowIndex()});
  rowContainer.newRow();
  rowContainer.disableNormalizedKeys();
  rowContainer.newRow();
  auto rowSize = rowContainer.fixedRowSize() + sizeof(normalized_key_t);
  RowContainerIterator iter;
  char* rows[2];
  auto numRows = rowContainer.listRows(&iter, std::span(rows), rowSize - 1);
  ASSERT_EQ(numRows, 1);
}

TEST_F(RowContainerTest, estimateRowSize) {
  auto numRows = 200'000;

  // Make a RowContainer with a fixed-length key column and a variable-length
  // dependent column.
  RowContainer rowContainer(
      {BIGINT()},
      {},
      {VARCHAR()},
      pool(),
      {.nullableKeys = false, .hasNext = true, .hasNormalizedKey = true});
  EXPECT_FALSE(rowContainer.estimateRowSize().has_value());

  // Store rows to the container.
  auto key = makeFlatIdentityVector<int64_t>(numRows);
  auto dependent = makeFlatVector<std::string>(
      numRows, [&](auto row) { return fmt::format("string - {}", row); });

  store(rowContainer, makeRowVector({key, dependent}));
  EXPECT_EQ(37, rowContainer.fixedRowSize());
  EXPECT_EQ(64, rowContainer.estimateRowSize());
  // 2*2MB huge page size blocks + 4 * 64K small allocations.
  EXPECT_EQ(0x440000, rowContainer.stringAllocator().retainedSize());
}

TEST_F(RowContainerTest, maxGrowthBytes) {
  {
    RowContainer rowContainer({BIGINT()}, pool());
    EXPECT_EQ(
        rowContainer.maxGrowthBytes(1, 0),
        memory::AllocationTraits::kHugePageSize);

    // The first row allocates a run with room for more rows.
    rowContainer.newRow();
    EXPECT_EQ(rowContainer.maxGrowthBytes(100, 0), 0);
    EXPECT_GT(rowContainer.maxGrowthBytes(10'000'000, 0), 0);
  }

  // A row with a normalized key also takes the key's bytes. Adding as many
  // rows as maxGrowthBytes() says fit allocates no memory.
  {
    RowContainer rowContainer(
        {BIGINT()}, {}, {}, pool(), {.hasNormalizedKey = true});
    rowContainer.newRow();
    vector_size_t numRows = 0;
    while (rowContainer.maxGrowthBytes(numRows + 1, 0) == 0) {
      ++numRows;
    }
    const auto allocatedBytes = rowContainer.allocatedBytes();
    for (auto i = 0; i < numRows; ++i) {
      rowContainer.newRow();
    }
    EXPECT_EQ(rowContainer.allocatedBytes(), allocatedBytes);
  }
}

TEST_F(RowContainerTest, alignment) {
  std::vector<Accumulator> accumulators{Accumulator(
      true, // isFixedSize
      42, // fixedSize
      false, // usesExternalMemory
      64, // alignment
      nullptr, // spillType
      [](auto, auto) { VELOX_UNREACHABLE(); },
      [](auto) {})};
  RowContainer rowContainer(
      {SMALLINT()},
      accumulators,
      {},
      pool_.get(),
      {.hasProbedFlag = true, .hasNormalizedKey = true});
  constexpr int kNumRows = 100;
  char* rows[kNumRows];
  for (int i = 0; i < kNumRows; ++i) {
    rows[i] = rowContainer.newRow();
    ASSERT_EQ(reinterpret_cast<uintptr_t>(rows[i]) % 64, 0);
    if (i > 0 && rows[i - 1] < rows[i]) {
      ASSERT_LE(rows[i - 1] + rowContainer.fixedRowSize(), rows[i]);
    }
    if (i == 50) {
      rowContainer.disableNormalizedKeys();
    }
  }
  RowContainerIterator iter;
  char* result[kNumRows];
  ASSERT_EQ(rowContainer.listRows(&iter, std::span(result), 1e8), kNumRows);
  for (int i = 0; i < kNumRows; ++i) {
    ASSERT_EQ(result[i], rows[i]);
  }
}

// The raw dependent holds what the caller writes, aligned for 8-byte values,
// without overlapping the keys and dependents.
TEST_F(RowContainerTest, rawDependent) {
  RowContainer rowContainer(
      {SMALLINT()},
      {},
      {VARCHAR()},
      pool_.get(),
      {.hasNext = true, .rawDependentSize = 24});
  ASSERT_EQ(rowContainer.rawDependentSize(), 24);

  auto input = makeRowVector({
      makeFlatVector<int16_t>({1, 2, 3}),
      makeFlatVector<std::string>({"a", "a long string, not inlined", "c"}),
  });
  const auto rows = store(rowContainer, input);
  for (vector_size_t i = 0; i < input->size(); ++i) {
    const auto bytes = rowContainer.rawDependent(rows[i]);
    ASSERT_EQ(bytes.size(), 24);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(bytes.data()) % alignof(int64_t), 0);
    std::fill_n(reinterpret_cast<int64_t*>(bytes.data()), 3, i + 100);
  }

  for (vector_size_t i = 0; i < input->size(); ++i) {
    const auto* bytes = reinterpret_cast<const int64_t*>(
        rowContainer.rawDependent(rows[i]).data());
    EXPECT_THAT(
        std::vector<int64_t>(bytes, bytes + 3),
        testing::Each(testing::Eq(i + 100)));
  }
  auto extracted = BaseVector::create(VARCHAR(), input->size(), pool());
  rowContainer.extractColumn(rows.data(), input->size(), 1, extracted);
  assertEqualVectors(input->childAt(1), extracted);

  common::SpillConfig spillConfig;
  SpillStats spillStats;
  VELOX_ASSERT_THROW(
      SortOutputSpiller(
          &rowContainer, input->rowType(), &spillConfig, &spillStats),
      "A row container with a raw dependent cannot be spilled");
}

TEST_F(RowContainerTest, accumulatorsWithDependents) {
  const Accumulator accumulator{/*isFixedSize=*/true,
                                /*fixedSize=*/8,
                                /*usesExternalMemory=*/false,
                                /*alignment=*/8,
                                /*spillType=*/nullptr,
                                /*spillExtractFunction=*/nullptr,
                                /*destroyFunction=*/nullptr};
  VELOX_ASSERT_THROW(
      RowContainer({BIGINT()}, {accumulator}, {VARCHAR()}, pool_.get(), {}),
      "A row container cannot have both accumulators and dependent columns");
}

TEST_F(RowContainerTest, externalMemoryAccumulatorRequiresDestroyFunction) {
  VELOX_ASSERT_THROW(
      Accumulator(
          true, // isFixedSize
          8, // fixedSize
          true, // usesExternalMemory
          1, // alignment
          nullptr, // spillType
          [](auto, auto) { VELOX_UNREACHABLE(); },
          nullptr),
      "An accumulator that uses external memory requires a destroy function.");
}

// Verify comparison of fringe float values
TEST_F(RowContainerTest, equalAndCompareFloat) {
  testEqualAndCompareRowContainerTypeFloat<float>(REAL());
}

// Verify comparison of fringe double values
TEST_F(RowContainerTest, equalAndCompareDouble) {
  testEqualAndCompareRowContainerTypeFloat<double>(DOUBLE());
}

TEST_F(RowContainerTest, equalAndCompareArrayTypeFloat) {
  testEqualAndCompareRowContainerTypeFloat<float>(ARRAY(REAL()));
}

TEST_F(RowContainerTest, equalAndCompareArrayTypeDouble) {
  testEqualAndCompareRowContainerTypeFloat<double>(ARRAY(DOUBLE()));
}

TEST_F(RowContainerTest, equalAndCompareMapTypeFloat) {
  testEqualAndCompareRowContainerTypeFloat<float>(MAP(REAL(), INTEGER()));
}

TEST_F(RowContainerTest, equalAndCompareMapTypeDouble) {
  testEqualAndCompareRowContainerTypeFloat<double>(MAP(DOUBLE(), INTEGER()));
}

TEST_F(RowContainerTest, equalAndCompareRowTypeFloat) {
  static const std::string colName{"floatCol"};
  testEqualAndCompareRowContainerTypeFloat<float>(ROW({colName}, {REAL()}));
}

TEST_F(RowContainerTest, equalAndCompareRowTypeDouble) {
  static const std::string colName{"doubleCol"};
  testEqualAndCompareRowContainerTypeFloat<double>(ROW({colName}, {DOUBLE()}));
}

TEST_F(RowContainerTest, compareStrings) {
  RowContainer rowContainer({VARCHAR()}, pool());
  auto testCompare = [&](std::string_view left, std::string_view right) {
    SCOPED_TRACE(testing::Message() << left << " vs " << right);
    const auto vector =
        makeFlatVector<StringView>({StringView(left), StringView(right)});
    const DecodedVector decoded(*vector);
    auto* leftRow = rowContainer.newRow();
    rowContainer.store(decoded, 0, leftRow, 0);
    auto* rightRow = rowContainer.newRow();
    rowContainer.store(decoded, 1, rightRow, 0);

    const auto expected = sign(left.compare(right));
    EXPECT_EQ(sign(rowContainer.compare(leftRow, rightRow, 0)), expected);
    EXPECT_EQ(
        sign(rowContainer.compare(
            leftRow, rowContainer.columnAt(0), decoded, 1)),
        expected);
  };

  const std::vector<std::pair<std::string, std::string>> pairs = {
      {"", "a"},
      {"abc", "abd"},
      {"ab", "abc"},
      {"abcdef", "abcdeg"},
      {"abcdefghijkl", "abcdefghijklm"},
      {"abcdXfghijklmnop", "abcdYfghijklmnop"},
      {"abcdefghijklmnop", "abcdefghijklmnoq"},
      {"abcdefghijklmnop", "abcdefghijklmnopq"},
  };
  for (const auto& [left, right] : pairs) {
    testCompare(left, right);
    testCompare(right, left);
    testCompare(left, left);
  }
}

TEST_F(RowContainerTest, partition) {
  // We assign an arbitrary partition number to each row and iterate over the
  // rows a partition at a time.
  constexpr int32_t kNumRows = 100019;
  constexpr uint8_t kNumPartitions = 16;
  auto batch = makeDataset(
      ROW(
          {{"int_val", INTEGER()},
           {"long_val", BIGINT()},
           {"string_val", VARCHAR()}}),
      kNumRows,
      [](RowVectorPtr /*rows*/) {});

  auto data = roundTrip(batch);
  std::vector<char*> rows(kNumRows);
  RowContainerIterator iter;
  data->listRows(&iter, std::span(rows), RowContainer::kUnlimited);

  // Test random skipping of RowContainerIterator.
  for (auto count = 0; count < 100; ++count) {
    auto index = (count * 121) % kNumRows;
    iter.reset();
    data->testingSkip(iter, index);
    EXPECT_EQ(iter.currentRow(), rows[index]);
    if (index + count < kNumRows) {
      data->testingSkip(iter, count);
      EXPECT_EQ(iter.currentRow(), rows[index + count]);
    }
  }

  // Expect throws before we get row partitions from this row container.
  for (auto partition = 0; partition < kNumPartitions; ++partition) {
    char* dummyBuffer;
    RowPartitions dummyRowPartitions(data->numRows(), *pool_);
    VELOX_ASSERT_THROW(
        data->listPartitionRows(
            iter, partition, dummyRowPartitions, std::span(&dummyBuffer, 1)),
        "Can't list partition rows from a mutable row container");
  }

  auto partitions = data->createRowPartitions(*pool_);
  ASSERT_FALSE(data->testingMutable());
  // Verify we can only get row partitions once from a row container.
  VELOX_ASSERT_THROW(
      data->createRowPartitions(*pool_),
      "Can only create RowPartitions once from a row container");
  // Verify we can't insert new row into a immutable row container.
#ifndef NDEBUG
  VELOX_ASSERT_THROW(
      data->newRow(), "Can't add row into an immutable row container")
#endif

  std::vector<uint8_t> rowPartitions(kNumRows);
  // Assign a partition to each row based on  modulo of first column.
  std::vector<std::vector<char*>> partitionRows(kNumPartitions);
  auto column = batch->childAt(0)->as<FlatVector<int32_t>>();
  for (auto i = 0; i < kNumRows; ++i) {
    uint8_t partition =
        static_cast<uint32_t>(column->valueAt(i)) % kNumPartitions;
    rowPartitions[i] = partition;
    partitionRows[partition].push_back(rows[i]);
  }
  partitions->appendPartitions(rowPartitions);
  for (auto partition = 0; partition < kNumPartitions; ++partition) {
    std::vector<char*> result(partitionRows[partition].size() + 10);
    iter.reset();
    int32_t numFound = 0;
    int32_t resultBatch = 1;
    // Read the rows in multiple batches.
    while (auto numResults = data->listPartitionRows(
               iter,
               partition,
               *partitions,
               std::span(result).subspan(
                   numFound,
                   std::min<size_t>(resultBatch, result.size() - numFound)))) {
      numFound += numResults;
      resultBatch += 13;
    }
    EXPECT_EQ(numFound, partitionRows[partition].size());
    result.resize(numFound);
    EXPECT_EQ(partitionRows[partition], result);
  }
}

TEST_F(RowContainerTest, partitionWithEmptyRowContainer) {
  auto rowType = ROW(
      {{"int_val", INTEGER()},
       {"long_val", BIGINT()},
       {"string_val", VARCHAR()}});
  RowContainer rowContainer(rowType->children(), pool());
  auto partitions = rowContainer.createRowPartitions(*pool_);
  ASSERT_EQ(partitions->size(), 0);
}

TEST_F(RowContainerTest, probedFlag) {
  RowContainer rowContainer(
      {BIGINT()},
      {},
      {BIGINT()},
      pool(),
      {.hasNext = true, .hasProbedFlag = true});

  auto input = makeRowVector({
      makeNullableFlatVector<int64_t>({1, 2, 3, 4, std::nullopt, 5}),
      makeFlatVector<int64_t>({10, 20, 30, 40, -1, 50}),
  });

  const auto size = input->size();
  const auto rows = store(rowContainer, input);

  // No 'probed' flags set. Verify all false, except for null key row.
  auto result = BaseVector::create<FlatVector<bool>>(BOOLEAN(), 1, pool());
  rowContainer.extractProbedFlags(
      rows,
      true /*setNullForNullKeysRow*/,
      false /*setNullForNonProbedRow*/,
      result);

  auto expected = makeNullableFlatVector<bool>(
      {false, false, false, false, std::nullopt, false});
  assertEqualVectors(expected, result);

  rowContainer.extractProbedFlags(
      rows,
      false /*setNullForNullKeysRow*/,
      false /*setNullForNonProbedRow*/,
      result);
  assertEqualVectors(makeConstant(false, size), result);

  rowContainer.extractProbedFlags(
      rows,
      true /*setNullForNullKeysRow*/,
      true /*setNullForNonProbedRow*/,
      result);
  assertEqualVectors(makeNullConstant(TypeKind::BOOLEAN, size), result);

  // Set 'probed' flags for every other row.
  for (auto i = 0; i < size; i += 2) {
    rowContainer.setProbedFlag({rows.data() + i, 1});
  }

  rowContainer.extractProbedFlags(
      rows,
      true /*setNullForNullKeysRow*/,
      false /*setNullForNonProbedRow*/,
      result);

  assertEqualVectors(
      makeNullableFlatVector<bool>(
          {true, false, true, false, std::nullopt, false}),
      result);

  rowContainer.extractProbedFlags(
      rows,
      true /*setNullForNullKeysRow*/,
      true /*setNullForNonProbedRow*/,
      result);

  assertEqualVectors(
      makeNullableFlatVector<bool>(
          {true, std::nullopt, true, std::nullopt, std::nullopt, std::nullopt}),
      result);

  // Set 'probed' flags for all rows.
  rowContainer.setProbedFlag(rows);

  rowContainer.extractProbedFlags(
      rows,
      true /*setNullForNullKeysRow*/,
      false /*setNullForNonProbedRow*/,
      result);

  assertEqualVectors(
      makeNullableFlatVector<bool>(
          {true, true, true, true, std::nullopt, true}),
      result);

  rowContainer.extractProbedFlags(
      rows,
      true /*setNullForNullKeysRow*/,
      true /*setNullForNonProbedRow*/,
      result);

  assertEqualVectors(
      makeNullableFlatVector<bool>(
          {true, true, true, true, std::nullopt, true}),
      result);
}

TEST_F(RowContainerTest, countFlag) {
  RowContainer rowContainer(
      {INTEGER()}, {}, {}, pool(), {.hasCountFlag = true});

  ASSERT_TRUE(rowContainer.hasCount());

  auto* row = rowContainer.newRow();
  EXPECT_EQ(rowContainer.count(row), 1);
  rowContainer.incrementCount(row);
  EXPECT_EQ(rowContainer.count(row), 2);
  rowContainer.decrementCount(row);
  EXPECT_EQ(rowContainer.count(row), 1);
  rowContainer.decrementCount(row);
  EXPECT_EQ(rowContainer.count(row), 0);
}

TEST_F(RowContainerTest, mixedFree) {
  constexpr int32_t kNumRows = 100'000;
  constexpr int32_t kNumBad = 100;
  const std::vector<TypePtr> dependent(11, TIMESTAMP());
  RowContainer data1({SMALLINT()}, dependent, pool());
  RowContainer data2({SMALLINT()}, dependent, pool());
  std::vector<char*> rows;

  // We put every second row in one container and every second in the other.
  for (auto i = 0; i < kNumRows; ++i) {
    rows.push_back(data1.newRow());
    rows.push_back(data2.newRow());
  }

  // We add some bad rows that are in neither container.
  for (auto i = 0; i < kNumBad; ++i) {
    rows.push_back(reinterpret_cast<char*>(i));
    rows.push_back(reinterpret_cast<char*>((1UL << 48) + i));
  }

  // We check that the containers correctly identify their own rows.
  std::vector<char*> result1(rows.size());
  std::vector<char*> result2(rows.size());
  ASSERT_EQ(kNumRows, data1.filterOwnedRows(rows, result1));
  for (auto i = 0; i < kNumRows * 2; i += 2) {
    ASSERT_EQ(rows[i], result1[i / 2]);
  }
  ASSERT_EQ(kNumRows, data2.filterOwnedRows(rows, result2));
  for (auto i = 1; i < kNumRows * 2; i += 2) {
    ASSERT_EQ(rows[i], result2[i / 2]);
  }

  // A container with no rows owns none of them.
  RowContainer empty({SMALLINT()}, dependent, pool());
  ASSERT_EQ(0, empty.filterOwnedRows(rows, result1));

  // We erase all rows from both containers and check there are no corruptions.
  data1.eraseRows(folly::Range<char**>(result1.data(), kNumRows));
  data2.eraseRows(folly::Range<char**>(result2.data(), kNumRows));
  EXPECT_EQ(0, data1.numRows());
  EXPECT_EQ(0, data2.numRows());
  data1.checkConsistency();
  data2.checkConsistency();
}

TEST_F(RowContainerTest, unknown) {
  std::vector<TypePtr> types = {UNKNOWN()};
  RowContainer rowContainer(types, pool());

  auto data = makeRowVector({
      makeAllNullFlatVector<UnknownValue>(5),
  });

  auto size = data->size();
  DecodedVector decoded(*data->childAt(0));
  auto rows = store(rowContainer, decoded, size);

  std::vector<uint64_t> hashes(size, 0);
  rowContainer.hashKeys(rows, hashes);
  for (auto hash : hashes) {
    ASSERT_EQ(BaseVector::kNullHash, hash);
  }

  RowContainer twoKeys({UNKNOWN(), UNKNOWN()}, pool_.get());
  auto twoKeyRows =
      store(twoKeys, makeRowVector({data->childAt(0), data->childAt(0)}));
  twoKeys.hashKeys(twoKeyRows, hashes);
  for (auto hash : hashes) {
    ASSERT_EQ(
        bits::hashMix(BaseVector::kNullHash, BaseVector::kNullHash), hash);
  }

  for (size_t row = 0; row < size; ++row) {
    ASSERT_EQ(
        rowContainer.compare</*mayHaveNulls=*/true>(
            rows[row], rowContainer.columnAt(0), decoded, row),
        0);
  }

  {
    // Verify compare method with two rows as input
    std::sort(rows.begin(), rows.end(), [&](const char* l, const char* r) {
      return rowContainer.compare(l, r, 0, {}) < 0;
    });
    VectorPtr result = BaseVector::create(UNKNOWN(), rows.size(), pool_.get());
    rowContainer.extractColumn(rows.data(), rows.size(), 0, result);
    // Since the Vector is just a null constant, it should be indistinguishable
    // from the original.
    assertEqualVectors(data->childAt(0), result);
  }

  std::vector<std::pair<int, char*>> indexedRows(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    indexedRows[i] = std::make_pair(i, rows[i]);
  }

  // Verify compare method with row and decoded vector as input
  // Sorting a NULL constant Vector doesn't change the Vector, so we just
  // validate that it runs without throwing an exception.
  EXPECT_NO_THROW(
      std::sort(
          indexedRows.begin(),
          indexedRows.end(),
          [&](const std::pair<int, char*>& l, const std::pair<int, char*>& r) {
            return rowContainer.compare(
                       l.second,
                       rowContainer.columnAt(0),
                       decoded,
                       r.first,
                       {}) < 0;
          }));

  // Verify compareRows method with row as input.
  // Sorting a NULL constant Vector doesn't change the Vector, so we just
  // validate that it runs without throwing an exception.
  EXPECT_NO_THROW(
      std::sort(
          indexedRows.begin(),
          indexedRows.end(),
          [&](const std::pair<int, char*>& l, const std::pair<int, char*>& r) {
            return rowContainer.compareRows(l.second, r.second) < 0;
          }));
}

TEST_F(RowContainerTest, nans) {
  const static auto kNaN = std::numeric_limits<double>::quiet_NaN();
  const static auto kSNaN = std::numeric_limits<double>::signaling_NaN();
  static const auto kNaNHash = folly::hasher<double>{}(kNaN);
  std::vector<TypePtr> types = {DOUBLE()};
  RowContainer rowContainer(types, pool());

  auto data = makeRowVector({
      makeFlatVector<double>({kNaN, kSNaN}),
  });

  auto size = data->size();
  DecodedVector decoded(*data->childAt(0));
  auto rows = store(rowContainer, decoded, size);

  // Verify that the hashes are equal.
  std::vector<uint64_t> hashes(size, 0);
  rowContainer.hashKeys(rows, hashes);
  for (auto hash : hashes) {
    ASSERT_EQ(kNaNHash, hash);
  }

  RowContainer twoKeys({DOUBLE(), DOUBLE()}, pool_.get());
  auto twoKeyRows =
      store(twoKeys, makeRowVector({data->childAt(0), data->childAt(0)}));
  twoKeys.hashKeys(twoKeyRows, hashes);
  for (auto hash : hashes) {
    ASSERT_EQ(bits::hashMix(kNaNHash, kNaNHash), hash);
  }

  // Verify that they are considered equal.
  for (size_t row = 0; row < size; ++row) {
    ASSERT_EQ(
        rowContainer.compare<false>(
            rows[row], rowContainer.columnAt(0), decoded, row),
        0);
  }
  ASSERT_EQ(rowContainer.compare(rows[0], rows[1], 0, {}), 0);
}

TEST_F(RowContainerTest, toString) {
  std::vector<TypePtr> keyTypes = {BIGINT(), VARCHAR()};
  std::vector<TypePtr> dependentTypes = {TINYINT(), REAL(), ARRAY(BIGINT())};
  RowContainer rowContainer(keyTypes, dependentTypes, pool());

  EXPECT_EQ(
      rowContainer.toString(),
      "Keys: BIGINT, VARCHAR Dependents: TINYINT, REAL, ARRAY<BIGINT> Num rows: 0");

  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<std::string>({"summer", "fall", "winter", "spring"}),
      makeFlatVector<int16_t>({11, 12, 13}),
      makeNullableFlatVector<float>({0.1, std::nullopt, 123.003}),
      makeArrayVectorFromJson<int64_t>({"[1, 2, 3]", "[4, 5]", "null"}),
  });

  const auto rows = store(rowContainer, data);

  EXPECT_EQ(
      rowContainer.toString(),
      "Keys: BIGINT, VARCHAR Dependents: TINYINT, REAL, ARRAY<BIGINT> Num rows: 3");

  EXPECT_EQ(
      rowContainer.toString(rows[0]),
      "{1, summer, 11, 0.10000000149011612, {1, 2, 3}}");
  EXPECT_EQ(rowContainer.toString(rows[1]), "{2, fall, 0, null, {4, 5}}");
  EXPECT_EQ(
      rowContainer.toString(rows[2]),
      "{3, winter, 12, 123.00299835205078, null}");
}

TEST_F(RowContainerTest, reuseRow) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2}),
      makeFlatVector<std::string>(
          {"non-inline string 1", "non-inline string number 2"}),
      makeArrayVector<int64_t>({{1, 2, 3}, {4, 5}}),
  });

  RowContainer rowContainer(
      {BIGINT()}, {VARCHAR(), ARRAY(BIGINT())}, pool_.get());

  std::vector<DecodedVector> decodedVectors;
  for (const auto& child : data->children()) {
    decodedVectors.emplace_back(*child);
  }

  auto* row = rowContainer.newRow();
  for (auto i = 0; i < decodedVectors.size(); ++i) {
    rowContainer.store(decodedVectors[i], 0, row, i);
  }

  row = rowContainer.initializeRow(row, /*reuse=*/true);
  for (auto i = 0; i < decodedVectors.size(); ++i) {
    rowContainer.store(decodedVectors[i], 1, row, i);
  }

  rowContainer.checkConsistency();
  ASSERT_EQ(rowContainer.numRows(), 1);
  for (auto i = 0; i < decodedVectors.size(); ++i) {
    ASSERT_EQ(rowContainer.columnStats(i).nullCount(), 0);
  }
  ASSERT_EQ(rowContainer.columnStats(1).sumBytes(), 26);
  ASSERT_EQ(
      rowContainer.toString(row), "{2, non-inline string number 2, {4, 5}}");
}

// A row may be erased or reused with only some of its columns stored. The
// columns not stored read as empty values.
TEST_F(RowContainerTest, partialWriteComplexTypedRow) {
  auto data = makeRowVector(
      {makeFlatVector<int64_t>(1, [](auto row) { return row; }),
       makeFlatVector<std::string>({"non-inline string"}),
       makeArrayVector<int64_t>({{1, 2, 3, 4, 5}}),
       makeMapVector<int64_t, int64_t>({{{4, 41}}}),
       makeRowVector({makeFlatVector<std::string>({"non-inline string"})})});

  RowContainer rowContainer(data->type()->asRow().children(), pool_.get());

  std::vector<DecodedVector> decodedVectors;
  for (const auto& child : data->children()) {
    decodedVectors.emplace_back(*child);
  }

  // No column stored.
  auto* row = rowContainer.newRow();
  rowContainer.checkConsistency();
  rowContainer.eraseRows(folly::Range<char**>(&row, 1));
  rowContainer.checkConsistency();

  // All columns stored, then reused with only the first three stored.
  row = rowContainer.newRow();
  for (auto i = 0; i < decodedVectors.size(); ++i) {
    rowContainer.store(decodedVectors[i], 0, row, i);
  }
  row = rowContainer.initializeRow(row, /*reuse=*/true);
  for (auto i = 0; i < 3; ++i) {
    rowContainer.store(decodedVectors[i], 0, row, i);
  }
  rowContainer.checkConsistency();

  row = rowContainer.initializeRow(row, /*reuse=*/true);
  rowContainer.checkConsistency();
  rowContainer.eraseRows(folly::Range<char**>(&row, 1));
  rowContainer.checkConsistency();
  for (auto i = 0; i < decodedVectors.size(); ++i) {
    ASSERT_EQ(rowContainer.columnStats(i).nullCount(), 0);
    ASSERT_EQ(rowContainer.columnStats(i).sumBytes(), 0);
  }
}

TEST_F(RowContainerTest, extractSerializedRowWithNonNullableKeys) {
  RowContainer rowContainer(
      {BIGINT()},
      {},
      {BIGINT()},
      pool(),
      {.nullableKeys = false, .hasProbedFlag = true});

  auto input = makeRowVector({
      makeFlatVector<int64_t>({123}),
      makeNullableFlatVector<int64_t>({std::nullopt}),
  });
  DecodedVector decodedKey(*input->childAt(0));
  DecodedVector decodedDependent(*input->childAt(1));

  auto* row = rowContainer.newRow();
  rowContainer.store(decodedKey, 0, row, 0);
  rowContainer.store(decodedDependent, 0, row, 1);
  rowContainer.setProbedFlag({&row, 1});

  auto serialized =
      BaseVector::create<FlatVector<StringView>>(VARBINARY(), 1, pool());
  rowContainer.extractSerializedRows(folly::Range<char**>(&row, 1), serialized);

  rowContainer.clear();
  row = rowContainer.newRow();
  rowContainer.storeSerializedRow(*serialized, 0, row);

  auto key = BaseVector::create<FlatVector<int64_t>>(BIGINT(), 1, pool());
  rowContainer.extractColumn(&row, 1, 0, key);
  EXPECT_EQ(key->valueAt(0), 123);

  auto dependent = BaseVector::create<FlatVector<int64_t>>(BIGINT(), 1, pool());
  rowContainer.extractColumn(&row, 1, 1, dependent);
  EXPECT_TRUE(dependent->isNullAt(0));

  auto probed = BaseVector::create<FlatVector<bool>>(BOOLEAN(), 1, pool());
  rowContainer.extractProbedFlags(
      {&row, 1},
      false, // setNullForNullKeysRow
      false, // setNullForNonProbedRow
      probed);
  EXPECT_TRUE(probed->valueAt(0));
}

TEST_F(RowContainerTest, extractSerializedRowUnsupportedParts) {
  auto result = BaseVector::create(VARBINARY(), 1, pool());
  {
    const Accumulator accumulator{/*isFixedSize=*/true,
                                  /*fixedSize=*/8,
                                  /*usesExternalMemory=*/false,
                                  /*alignment=*/8,
                                  /*spillType=*/nullptr,
                                  /*spillExtractFunction=*/nullptr,
                                  /*destroyFunction=*/nullptr};
    RowContainer rowContainer({BIGINT()}, {accumulator}, {}, pool(), {});
    char* row = rowContainer.newRow();
    VELOX_ASSERT_THROW(
        rowContainer.extractSerializedRows({&row, 1}, result),
        "Rows with accumulators cannot be serialized");
  }
  {
    RowContainer rowContainer(
        {BIGINT()}, {}, {}, pool(), {.rawDependentSize = 8});
    char* row = rowContainer.newRow();
    VELOX_ASSERT_THROW(
        rowContainer.extractSerializedRows({&row, 1}, result),
        "Rows with a raw dependent cannot be serialized");
  }
}

TEST_F(RowContainerTest, extractSerializedRow) {
  VectorFuzzer fuzzer(
      {
          .vectorSize = 100,
          .nullRatio = 0.1,
      },
      pool());

  for (auto i = 0; i < 100; ++i) {
    SCOPED_TRACE(fmt::format("Iteration #: {}", i));

    auto rowType = fuzzer.randRowType();
    auto data = fuzzer.fuzzInputRow(rowType);
    std::vector<bool> expectedColumnHasNulls(data->size(), false);
    for (int col = 0; col < rowType->size(); ++col) {
      const auto child = data->childAt(col);
      if (!child->mayHaveNulls()) {
        expectedColumnHasNulls[col] = false;
      } else {
        for (auto row = 0; row < child->size(); ++row) {
          if (child->isNullAt(row)) {
            expectedColumnHasNulls[col] = true;
            break;
          }
        }
      }
    }

    SCOPED_TRACE(data->toString());

    RowContainer rowContainer{rowType->children(), pool()};

    auto rows = store(rowContainer, data);

    for (int col = 0; col < rowType->size(); ++col) {
      ASSERT_EQ(rowContainer.columnHasNulls(col), expectedColumnHasNulls[col]);
    }

    // Extract serialized rows.
    auto serialized = BaseVector::create<FlatVector<StringView>>(
        VARBINARY(), data->size(), pool());
    rowContainer.extractSerializedRows(
        folly::Range(rows.data(), rows.size()), serialized);

    for (int col = 0; col < rowType->size(); ++col) {
      ASSERT_EQ(rowContainer.columnHasNulls(col), expectedColumnHasNulls[col]);
    }

    rowContainer.clear();
    for (int col = 0; col < rowType->size(); ++col) {
      ASSERT_EQ(rowContainer.columnHasNulls(col), false);
    }
    rows.clear();

    // Load serialized rows back.
    for (auto i = 0; i < data->size(); ++i) {
      rows.push_back(rowContainer.newRow());
      rowContainer.storeSerializedRow(*serialized, i, rows.back());
    }

    // Extract into regular vector.
    auto copy = BaseVector::create<RowVector>(rowType, data->size(), pool());
    for (auto i = 0; i < copy->childrenSize(); ++i) {
      rowContainer.extractColumn(
          rows.data(), copy->size(), i, copy->childAt(i));
    }

    assertEqualVectors(data, copy);
  }
}

DEBUG_ONLY_TEST_F(RowContainerTest, eraseAfterOomStoringString) {
  RowContainer rowContainer({VARCHAR()}, pool());

  auto vector = makeFlatVector({std::string(StringView::kInlineSize * 2, 'a')});

  DecodedVector decoded(*vector);
  auto row = rowContainer.newRow();

  const std::string exceptionMsg = "Expected simulated OOM.";
  SCOPED_TESTVALUE_SET(
      "facebook::velox::common::memory::MemoryPoolImpl::allocateNonContiguous",
      std::function<void(memory::MemoryPool*)>(
          [&](memory::MemoryPool*) { VELOX_FAIL(exceptionMsg); }));
  VELOX_ASSERT_THROW(rowContainer.store(decoded, 0, row, 0), exceptionMsg);

  std::vector<char*> rows(1);
  RowContainerIterator iter;
  auto numRows = rowContainer.listRows(&iter, std::span(rows.data(), 1));
  VELOX_CHECK_EQ(numRows, 1);
  // If the StringView was placed in the row before the memory was
  // allocated in the container's string allocator, this will fail
  // attempting to free a string the allocator doesn't own.
  rowContainer.eraseRows(folly::Range<char**>(rows.data(), numRows));
}

TEST_P(RowContainerTest, hugeIntStoreWithNulls) {
  constexpr int32_t kNumRows = 100;
  constexpr int32_t kColumnIndex = 0;

  auto isNullAt = [](vector_size_t row) { return row % 10 == 0; };

  // The indices of null rows are out of range, so storing must not read them.
  auto source = BaseVector::wrapInDictionary(
      makeNulls(kNumRows, isNullAt),
      makeIndices(
          kNumRows,
          [&](auto row) {
            return isNullAt(row) ? std::numeric_limits<vector_size_t>::max()
                                 : row;
          }),
      kNumRows,
      makeFlatVector<int128_t>(
          kNumRows, [](auto row) { return HugeInt::build(row, row); }));

  RowContainer rowContainer(
      {HUGEINT()}, {}, {}, pool(), {.useListRowIndex = useListRowIndex()});
  const auto rows = storeColumns(rowContainer, makeRowVector({source}));
  auto extracted = BaseVector::copy(*source);
  rowContainer.extractColumn(rows.data(), kNumRows, kColumnIndex, extracted);
  assertEqualVectors(source, extracted);
}

TEST_P(RowContainerTest, columnHasNulls) {
  RowContainer rowContainer(
      {BIGINT(), BIGINT()},
      {},
      {BIGINT(), BIGINT()},
      pool(),
      {.useListRowIndex = useListRowIndex()});
  for (int i = 0; i < rowContainer.columnTypes().size(); ++i) {
    ASSERT_FALSE(rowContainer.columnHasNulls(i));
  }

  const uint64_t kNumRows = 1000;
  auto rowVector = makeRowVector(
      {makeFlatVector<int64_t>(kNumRows, [](auto row) { return row % 5; }),
       makeFlatVector<int64_t>(
           kNumRows, [](auto row) { return row % 5; }, nullEvery(3)),
       makeFlatVector<int64_t>(kNumRows, [](auto row) { return row % 7; }),
       makeFlatVector<int64_t>(
           kNumRows, [](auto row) { return row % 7; }, nullEvery(999))});

  const auto rows = storeColumns(rowContainer, rowVector);
  for (int i = 0; i < rowContainer.columnTypes().size(); ++i) {
    if (i % 2 == 0) {
      ASSERT_TRUE(!rowContainer.columnHasNulls(i));
    } else {
      ASSERT_TRUE(rowContainer.columnHasNulls(i));
    }
  }
  // If the column's mayHaveNulls is false, the extracted vector's mayHaveNulls
  // should be false.
  for (int i = 0; i < rowContainer.columnTypes().size(); ++i) {
    auto vector =
        BaseVector::create(rowVector->childAt(i)->type(), kNumRows, pool());
    rowContainer.extractColumn(rows.data(), kNumRows, i, vector);
    if (i % 2 == 0) {
      ASSERT_TRUE(!vector->mayHaveNulls());
    } else {
      ASSERT_TRUE(vector->mayHaveNulls());
    }
  }
}

TEST_P(RowContainerTest, store) {
  const uint64_t kNumRows = 1000;
  auto rowVectorWithNulls = makeRowVector({
      makeFlatVector<int64_t>(
          kNumRows, [](auto row) { return row % 5; }, nullEvery(6)),
      makeFlatVector<std::string>(
          kNumRows,
          [](auto row) { return fmt::format("abcdefg123_{}", row); },
          nullEvery(7)),
      makeFlatVector<int64_t>(
          kNumRows, [](auto row) { return row; }, nullEvery(8)),
      makeArrayVector<int32_t>(
          kNumRows,
          [](auto i) { return i % 5; },
          [](auto i) { return i % 10; },
          nullEvery(10)),
  });

  auto rowVectorNoNulls = makeRowVector({
      makeFlatVector<int64_t>(kNumRows, [](auto row) { return row % 5; }),
      makeFlatVector<std::string>(
          kNumRows, [](auto row) { return fmt::format("abcdefg12_{}", row); }),
      makeFlatVector<int64_t>(kNumRows, [](auto row) { return row; }),
      makeArrayVector<int64_t>(
          kNumRows,
          [](auto i) { return i % 3; },
          [](auto i) { return i % 10; }),
  });
  for (auto& rowVector : {rowVectorWithNulls, rowVectorNoNulls}) {
    RowContainer rowContainer(
        {BIGINT(), VARCHAR()},
        {},
        {BIGINT(), ARRAY(BIGINT())},
        pool(),
        {.useListRowIndex = useListRowIndex()});
    ASSERT_EQ(rowContainer.numRows(), 0);
    const auto rows = storeColumns(rowContainer, rowVector);
    ASSERT_EQ(rowContainer.numRows(), kNumRows);
    for (int i = 0; i < rowContainer.columnTypes().size(); ++i) {
      auto vector =
          BaseVector::create(rowVector->childAt(i)->type(), kNumRows, pool());
      rowContainer.extractColumn(rows.data(), kNumRows, i, vector);
      assertEqualVectors(rowVector->childAt(i), vector);
    }
  }
}

TEST_F(RowContainerTest, customComparison) {
  auto values = makeNullableFlatVector<int64_t>(
      {std::nullopt,
       256 * 4 + 3,
       256 * 2 + 4,
       256 + 2,
       5,
       256 * 5,
       256 * 2 + 1},
      BIGINT_TYPE_WITH_CUSTOM_COMPARISON());

  // The custom comparison compares the values mod 256.
  std::vector<std::optional<int64_t>> ascNullsFirstOrder = {
      std::nullopt, 256 * 5, 256 * 2 + 1, 256 + 2, 256 * 4 + 3, 256 * 2 + 4, 5};

  testOrderAndEqualsWithNullsFirstVariations<int64_t>(
      BIGINT_TYPE_WITH_CUSTOM_COMPARISON(),
      values,
      ascNullsFirstOrder,
      [&](const auto& expectedOrder) {
        return makeNullableFlatVector<int64_t>(
            expectedOrder, BIGINT_TYPE_WITH_CUSTOM_COMPARISON());
      });
}

TEST_F(RowContainerTest, customComparisonArray) {
  auto values = makeNullableArrayVector<int64_t>(
      {std::nullopt,
       {{std::nullopt}},
       {{256 * 4 + 3}},
       {{256 * 2 + 4}},
       {{256 + 2}},
       {{5}},
       {{256 * 5}},
       {{256 * 2 + 1}}},
      ARRAY(BIGINT_TYPE_WITH_CUSTOM_COMPARISON()));

  // The custom comparison compares the values mod 256.
  std::vector<std::optional<std::vector<std::optional<int64_t>>>>
      ascNullsFirstOrder = {
          std::nullopt,
          {{std::nullopt}},
          {{256 * 5}},
          {{256 * 2 + 1}},
          {{256 + 2}},
          {{256 * 4 + 3}},
          {{256 * 2 + 4}},
          {{5}}};

  testOrderAndEqualsWithNullsFirstVariations<
      std::vector<std::optional<int64_t>>>(
      ARRAY(BIGINT_TYPE_WITH_CUSTOM_COMPARISON()),
      values,
      ascNullsFirstOrder,
      [&](const auto& expectedOrder) {
        return makeNullableArrayVector<int64_t>(
            expectedOrder, ARRAY(BIGINT_TYPE_WITH_CUSTOM_COMPARISON()));
      });
}

TEST_F(RowContainerTest, customComparisonMap) {
  auto values = makeNullableMapVector<int64_t, int32_t>(
      {{{std::nullopt}},
       {{{256 * 4 + 3, 1}}},
       {{{256 * 2 + 4, 2}}},
       {{{256 + 2, 3}}},
       {{{5, 4}}},
       {{{256 * 5, 5}}},
       {{{256 * 2 + 1, 6}}}},
      MAP(BIGINT_TYPE_WITH_CUSTOM_COMPARISON(), INTEGER()));

  // The custom comparison compares the values mod 256.
  std::vector<
      std::optional<std::vector<std::pair<int64_t, std::optional<int32_t>>>>>
      ascNullsFirstOrder = {
          {{std::nullopt}},
          {{{256 * 5, 5}}},
          {{{256 * 2 + 1, 6}}},
          {{{256 + 2, 3}}},
          {{{256 * 4 + 3, 1}}},
          {{{256 * 2 + 4, 2}}},
          {{{5, 4}}}};

  testOrderAndEqualsWithNullsFirstVariations<
      std::vector<std::pair<int64_t, std::optional<int32_t>>>>(
      MAP(BIGINT_TYPE_WITH_CUSTOM_COMPARISON(), INTEGER()),
      values,
      ascNullsFirstOrder,
      [&](const auto& expectedOrder) {
        return makeNullableMapVector<int64_t, int32_t>(
            expectedOrder,
            MAP(BIGINT_TYPE_WITH_CUSTOM_COMPARISON(), INTEGER()));
      });
}

TEST_F(RowContainerTest, customComparisonRow) {
  auto values = makeRowVector({makeNullableFlatVector<int64_t>(
      {std::nullopt,
       256 * 4 + 3,
       256 * 2 + 4,
       256 + 2,
       5,
       256 * 5,
       256 * 2 + 1},
      BIGINT_TYPE_WITH_CUSTOM_COMPARISON())});

  // The custom comparison compares the values mod 256.
  std::vector<std::optional<int64_t>> ascNullsFirstOrder = {
      std::nullopt, 256 * 5, 256 * 2 + 1, 256 + 2, 256 * 4 + 3, 256 * 2 + 4, 5};

  testOrderAndEqualsWithNullsFirstVariations<int64_t>(
      ROW({BIGINT_TYPE_WITH_CUSTOM_COMPARISON()}),
      values,
      ascNullsFirstOrder,
      [&](const auto& expectedOrder) {
        return makeRowVector({makeNullableFlatVector<int64_t>(
            expectedOrder, BIGINT_TYPE_WITH_CUSTOM_COMPARISON())});
      });
}

TEST_P(RowContainerTest, isNanAt) {
  const auto kNan = std::numeric_limits<double>::quiet_NaN();
  const auto kNanF = std::numeric_limits<float>::quiet_NaN();
  auto rowVector = makeRowVector({
      makeFlatVector<float>({kNanF, 0.2, 0.3, 0.4}),
      makeFlatVector<double>({0.1, kNan, 0.3, 0.4}),
      makeFlatVector<float>({0.1, 0.2, kNanF, 0.4}),
      makeFlatVector<double>({0.1, 0.2, 0.3, kNan}),
  });
  const auto kNumRows = rowVector->size();

  RowContainer rowContainer(
      {REAL(), DOUBLE()},
      {},
      {REAL(), DOUBLE()},
      pool(),
      {.useListRowIndex = useListRowIndex()});
  const auto rows = storeColumns(rowContainer, rowVector);

  for (int i = 0; i < kNumRows; ++i) {
    const auto* row = rows[i];
    for (int j = 0; j < 4; ++j) {
      const auto rowColumn = rowContainer.columnAt(j);
      if (i % 2 == 0) {
        if (i == j) {
          EXPECT_TRUE(rowContainer.isNanAt<float>(row, rowColumn));
        } else {
          EXPECT_FALSE(rowContainer.isNanAt<float>(row, rowColumn));
        }
      } else {
        if (i == j) {
          EXPECT_TRUE(rowContainer.isNanAt<double>(row, rowColumn));
        } else {
          EXPECT_FALSE(rowContainer.isNanAt<double>(row, rowColumn));
        }
      }
    }
  }
}

TEST_F(RowContainerTest, invalidatedColumnStats) {
  constexpr int32_t kNumRows = 100;
  auto batch = makeBatchWithLargeStrings(
      ROW({
          {"bool_val", BOOLEAN()},
          {"int_val", INTEGER()},
          {"string_val", VARCHAR()},
          {"array_val", ARRAY(VARCHAR())},
          {"struct_val2",
           ROW({{"s_int", INTEGER()}, {"s_array", ARRAY(REAL())}})},
      }),
      kNumRows);
  // The first two columns are keys, the rest dependents.
  const auto& types = batch->rowType()->children();
  const std::vector<TypePtr> keys(types.begin(), types.begin() + 2);
  const std::vector<TypePtr> dependents(types.begin() + 2, types.end());

  // Adds rows, which have no nulls and no variable-width values until stored.
  const auto addRows = [&](RowContainer& rowContainer) {
    for (auto i = 0; i < rowContainer.columnTypes().size(); ++i) {
      EXPECT_EQ(rowContainer.columnStats(i).nullCount(), 0);
      EXPECT_EQ(rowContainer.columnStats(i).sumBytes(), 0);
    }
    for (int i = 0; i < kNumRows; ++i) {
      rowContainer.newRow();
    }
    auto rows = storeKeysAndDependents(batch, rowContainer);
    EXPECT_EQ(rowContainer.numRows(), kNumRows);
    return rows;
  };

  {
    RowContainer rowContainer(
        keys, {}, dependents, pool(), {.nullableKeys = false});
    auto rows = addRows(rowContainer);
    rowContainer.setAllNull(rows[0]);
    rowContainer.checkConsistency();
  }

  {
    RowContainer rowContainer(
        keys, {}, dependents, pool(), {.nullableKeys = false});
    auto rows = addRows(rowContainer);
    rowContainer.eraseRows(rows);
    rowContainer.checkConsistency();
  }
}

TEST_F(RowContainerTest, rowColumnStats) {
  RowColumnStats stats;
  EXPECT_EQ(stats.nullCount(), 0);
  EXPECT_EQ(stats.sumBytes(), 0);
  EXPECT_EQ(stats.maxBytes(), 0);
  EXPECT_EQ(stats.avgBytes(/*numRows=*/0), 0);

  stats.addValue(10);
  stats.addValue(20);
  stats.addValue(5);
  EXPECT_EQ(stats.nullCount(), 0);
  EXPECT_EQ(stats.sumBytes(), 35);
  EXPECT_EQ(stats.maxBytes(), 20);
  EXPECT_EQ(stats.avgBytes(/*numRows=*/3), 11);

  stats.addNull();
  stats.addNull();
  EXPECT_EQ(stats.nullCount(), 2);
  EXPECT_EQ(stats.sumBytes(), 35);
  EXPECT_EQ(stats.avgBytes(/*numRows=*/5), 11);

  // The maximum stays as a bound.
  stats.removeValue(20);
  EXPECT_EQ(stats.sumBytes(), 15);
  EXPECT_EQ(stats.maxBytes(), 20);
  EXPECT_EQ(stats.avgBytes(/*numRows=*/4), 7);

  stats.removeNull();
  EXPECT_EQ(stats.nullCount(), 1);
  EXPECT_EQ(stats.avgBytes(/*numRows=*/3), 7);

  // Only nulls left.
  stats.removeValue(10);
  stats.removeValue(5);
  EXPECT_EQ(stats.sumBytes(), 0);
  EXPECT_EQ(stats.avgBytes(/*numRows=*/1), 0);
}

TEST_P(RowContainerTest, storeAndCollectColumnStats) {
  const uint64_t kNumRows = 1000;
  auto rowVector = makeRowVector({
      makeFlatVector<int64_t>(
          kNumRows, [](auto row) { return row % 5; }, nullEvery(7)),
      makeFlatVector<std::string>(
          kNumRows,
          [](auto row) { return fmt::format("abcdefg123_{}", row); },
          nullEvery(7)),
  });

  RowContainer rowContainer(
      {BIGINT(), VARCHAR()},
      {},
      {},
      pool(),
      {.useListRowIndex = useListRowIndex()});
  auto rows = storeColumns(rowContainer, rowVector);

  ASSERT_EQ(rowContainer.numRows(), kNumRows);
  for (int i = 0; i < rowContainer.columnTypes().size(); ++i) {
    const auto& stats = rowContainer.columnStats(i);
    EXPECT_EQ(stats.nullCount(), 143);
    if (rowVector->childAt(i)->typeKind() == TypeKind::VARCHAR) {
      EXPECT_EQ(stats.maxBytes(), 14);
      EXPECT_EQ(stats.sumBytes(), 11905);
      EXPECT_EQ(stats.avgBytes(kNumRows), 13);
    } else {
      EXPECT_EQ(stats.maxBytes(), 0);
      EXPECT_EQ(stats.sumBytes(), 0);
    }
  }

  rowContainer.eraseRows(folly::Range(rows.data(), 10)); // there are 2 nulls
  for (int i = 0; i < rowContainer.columnTypes().size(); ++i) {
    const auto& stats = rowContainer.columnStats(i);
    EXPECT_EQ(stats.nullCount(), 141);
    if (rowVector->childAt(i)->typeKind() == TypeKind::VARCHAR) {
      EXPECT_EQ(stats.sumBytes(), 11809);
      EXPECT_EQ(stats.avgBytes(kNumRows - 10), 13);
    }
  }
  rowContainer.clear();
  for (int i = 0; i < rowContainer.columnTypes().size(); ++i) {
    EXPECT_EQ(rowContainer.columnStats(i).nullCount(), 0);
    EXPECT_EQ(rowContainer.columnStats(i).sumBytes(), 0);
  }
}

// Fixed-width columns count only nulls. Variable-width columns also sum their
// sizes.
TEST_P(RowContainerTest, columnStatsMixedTypes) {
  const vector_size_t kNumRows = 1000;
  const auto nulls = nullEvery(7);
  auto rowVector = makeRowVector({
      makeFlatVector<bool>(
          kNumRows, [](auto row) { return row % 2 == 0; }, nulls),
      makeFlatVector<int8_t>(
          kNumRows, [](auto row) { return row; }, nulls),
      makeFlatVector<int16_t>(
          kNumRows, [](auto row) { return row; }, nulls),
      makeFlatVector<int32_t>(
          kNumRows, [](auto row) { return row; }, nulls),
      makeFlatVector<int64_t>(
          kNumRows, [](auto row) { return row; }, nulls),
      makeFlatVector<int128_t>(
          kNumRows, [](auto row) { return row; }, nulls),
      makeFlatVector<float>(
          kNumRows, [](auto row) { return row; }, nulls),
      makeFlatVector<double>(
          kNumRows, [](auto row) { return row; }, nulls),
      makeFlatVector<Timestamp>(
          kNumRows, [](auto row) { return Timestamp(row, 0); }, nulls),
      makeFlatVector<std::string>(
          kNumRows,
          [](auto row) { return fmt::format("abcdefg123_{}", row); },
          nulls),
      makeArrayVector<int64_t>(
          kNumRows,
          [](auto row) { return row % 4; },
          [](auto row) { return row; },
          nulls),
  });
  const auto numCols = rowVector->childrenSize();
  const auto numFixed = numCols - 2;

  // Split the columns between keys and dependents to cover both layouts.
  const auto& types = rowVector->rowType()->children();
  const std::vector<TypePtr> keyTypes(types.begin(), types.begin() + 5);
  const std::vector<TypePtr> dependentTypes(types.begin() + 5, types.end());
  RowContainer rowContainer(
      keyTypes,
      {},
      dependentTypes,
      pool(),
      {.useListRowIndex = useListRowIndex()});
  auto rows = storeColumns(rowContainer, rowVector);

  auto checkStats = [&](int32_t numRows, int32_t numNonNull) {
    for (auto i = 0; i < numCols; ++i) {
      SCOPED_TRACE(fmt::format("column {}", i));
      const auto& stats = rowContainer.columnStats(i);
      EXPECT_EQ(stats.nullCount(), numRows - numNonNull);
      if (i < numFixed) {
        EXPECT_EQ(stats.sumBytes(), 0);
        EXPECT_EQ(stats.maxBytes(), 0);
      } else {
        EXPECT_GT(stats.sumBytes(), 0);
        EXPECT_GT(stats.avgBytes(numRows), 0);
        EXPECT_GE(stats.maxBytes(), stats.avgBytes(numRows));
      }
    }
  };
  checkStats(kNumRows, 857);

  // The first 10 rows have 2 nulls.
  rowContainer.eraseRows(folly::Range(rows.data(), 10));
  checkStats(kNumRows - 10, 849);
  rowContainer.checkConsistency();

  // Erasing the remaining rows leaves no values.
  rowContainer.eraseRows(folly::Range(rows.data() + 10, kNumRows - 10));
  for (auto i = 0; i < numCols; ++i) {
    SCOPED_TRACE(fmt::format("column {}", i));
    const auto& stats = rowContainer.columnStats(i);
    EXPECT_EQ(stats.nullCount(), 0);
    EXPECT_EQ(stats.sumBytes(), 0);
  }

  // clear() resets the stats.
  rowContainer.clear();
  for (auto i = 0; i < numCols; ++i) {
    EXPECT_EQ(rowContainer.columnStats(i).nullCount(), 0);
    EXPECT_EQ(rowContainer.columnStats(i).maxBytes(), 0);
  }
  rows = storeColumns(rowContainer, rowVector);
  checkStats(kNumRows, 857);
}

TEST_F(RowContainerTest, mergeColumnStats) {
  RowColumnStats first;
  first.addValue(10);
  first.addValue(20);

  RowColumnStats nullsOnly;
  nullsOnly.addNull();

  RowColumnStats last;
  last.addValue(15);

  {
    const auto merged = RowColumnStats::merge({first, nullsOnly, last});
    EXPECT_EQ(merged.nullCount(), 1);
    EXPECT_EQ(merged.sumBytes(), 45);
    EXPECT_EQ(merged.maxBytes(), 20);
    EXPECT_EQ(merged.avgBytes(/*numRows=*/4), 15);
  }

  {
    const auto merged = RowColumnStats::merge({});
    EXPECT_EQ(merged.nullCount(), 0);
    EXPECT_EQ(merged.sumBytes(), 0);
    EXPECT_EQ(merged.maxBytes(), 0);
  }
}

TEST_F(RowContainerTest, setAllNull) {
  RowContainer rowContainer(
      {INTEGER()},
      {Accumulator(
          true, 8, false, 8, INTEGER(), [](auto, auto) {}, [](auto) {})},
      {},
      pool(),
      {});

  auto row = rowContainer.newRow();

  auto keyColumn = rowContainer.columnAt(0);
  auto accumulator = rowContainer.columnAt(1);
  const auto isInitialized = [&] {
    return (row[accumulator.initializedByte()] &
            accumulator.initializedMask()) != 0;
  };
  ASSERT_FALSE(RowContainer::isNullAt(row, keyColumn));
  ASSERT_FALSE(RowContainer::isNullAt(row, accumulator));
  ASSERT_FALSE(isInitialized());

  rowContainer.setAllNull(row);

  ASSERT_TRUE(RowContainer::isNullAt(row, keyColumn));
  ASSERT_TRUE(RowContainer::isNullAt(row, accumulator));
  ASSERT_FALSE(isInitialized());
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    RowContainerTest,
    RowContainerTest,
    testing::ValuesIn({false, true}));

} // namespace
} // namespace facebook::velox::exec::test
