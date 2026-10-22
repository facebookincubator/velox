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

#include "velox/exec/FlatRowCompare.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include "velox/exec/FlatRowSerde.h"
#include "velox/exec/tests/utils/FlatRowTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

class FlatRowCompareTest : public FlatRowTestBase {
 public:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
  }

 protected:
  // Asserts that compareRow produces the same sort order as the
  // RowContainer::compare oracle for the given column flags.
  void verifyCompareSort(
      const RowVectorPtr& input,
      const std::vector<CompareFlags>& flags) {
    const int32_t numCols = static_cast<int32_t>(input->childrenSize());
    VELOX_CHECK_EQ(
        static_cast<int32_t>(flags.size()),
        numCols,
        "flags.size() must equal the number of columns");

    std::vector<TypePtr> keyTypes;
    for (int32_t i = 0; i < numCols; ++i) {
      keyTypes.push_back(input->childAt(i)->type());
    }

    auto container = makeRowContainer(
        keyTypes, /*dependentTypes=*/{}, /*nullableKeys=*/true);
    storeRows(container.get(), input, numCols);
    std::vector<char*> liveRows = listRows(container.get());

    RowContainerLayout layout(container.get());
    auto [buffer, serializedRows] = serializeRows(liveRows, layout);
    for (char* ser : serializedRows) {
      char* mutableRow = ser;
      deserializeRow(mutableRow, layout);
    }

    const int32_t numRows = static_cast<int32_t>(liveRows.size());

    std::vector<int32_t> refOrder(numRows);
    std::iota(refOrder.begin(), refOrder.end(), 0);
    std::stable_sort(
        refOrder.begin(), refOrder.end(), [&](int32_t lhsIdx, int32_t rhsIdx) {
          for (int32_t col = 0; col < numCols; ++col) {
            const int32_t compareResult = container->compare(
                liveRows[lhsIdx], liveRows[rhsIdx], col, flags[col]);
            if (compareResult != 0) {
              return compareResult < 0;
            }
          }
          // Tiebreak by insertion index for stability.
          return lhsIdx < rhsIdx;
        });

    std::vector<int32_t> actualOrder(numRows);
    std::iota(actualOrder.begin(), actualOrder.end(), 0);
    std::stable_sort(
        actualOrder.begin(),
        actualOrder.end(),
        [&](int32_t lhsIdx, int32_t rhsIdx) {
          for (int32_t col = 0; col < numCols; ++col) {
            const int32_t compareResult = compareRow(
                serializedRows[lhsIdx],
                serializedRows[rhsIdx],
                layout.rowColumns[col],
                container->columnTypes()[col],
                flags[col]);
            if (compareResult != 0) {
              return compareResult < 0;
            }
          }
          // Tiebreak by insertion index for stability.
          return lhsIdx < rhsIdx;
        });

    EXPECT_EQ(refOrder, actualOrder)
        << "compareRow sort order does not match RowContainer::compare";
  }
};

// Exercises compareRow against all scalar and complex types across all four
// ASC/DESC × nullsFirst/nullsLast flag combinations with ~10% nulls.
TEST_F(FlatRowCompareTest, fuzzedCompareSort) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 1'024;
  opts.nullRatio = 0.1;
  opts.stringLength = 20;
  opts.stringVariableLength = true;
  VectorFuzzer fuzzer(opts, pool(), /*seed=*/42);

  const std::vector<VectorPtr> cols = {
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
  };

  for (bool ascending : {true, false}) {
    for (bool nullsFirst : {true, false}) {
      const CompareFlags flags{nullsFirst, ascending};
      for (const auto& col : cols) {
        verifyCompareSort(makeRowVector({col}), {flags});
      }
    }
  }
}

// Verifies Velox float sort order (lowest < 0 < min < max < inf < NaN) for
// REAL and DOUBLE. VectorFuzzer never generates NaN/inf so tested explicitly.
TEST_F(FlatRowCompareTest, floatSpecialValuesCompare) {
  auto checkAllFlags = [&](VectorPtr col) {
    auto input = makeRowVector({std::move(col)});
    for (bool ascending : {true, false}) {
      for (bool nullsFirst : {true, false}) {
        verifyCompareSort(input, {CompareFlags{nullsFirst, ascending}});
      }
    }
  };

  // REAL: lowest < 0 < min < max < inf < NaN.
  checkAllFlags(
      makeFlatVector<float>({
          std::numeric_limits<float>::lowest(),
          0.0f,
          std::numeric_limits<float>::min(),
          std::numeric_limits<float>::max(),
          std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::quiet_NaN(),
      }));

  // DOUBLE: same ordering.
  checkAllFlags(
      makeFlatVector<double>({
          std::numeric_limits<double>::lowest(),
          0.0,
          std::numeric_limits<double>::min(),
          std::numeric_limits<double>::max(),
          std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::quiet_NaN(),
      }));
}
