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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::functions::test;

namespace {

class RowFunctionTest : public FunctionBaseTest {
 protected:
  auto evalRow(
      vector_size_t numRows,
      const std::string colNames,
      const std::vector<VectorPtr>& inputs) {
    auto rows = SelectivityVector(numRows);
    return evaluate(
        fmt::format("row_constructor({})", colNames),
        makeRowVector(inputs),
        rows);
  }
};

TEST_F(RowFunctionTest, childSizeLessThanRowsEnd) {
  // Throw when any child not of right size
  VELOX_ASSERT_THROW(
      evalRow(
          5,
          "C0, C1",
          {makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
           makeFlatVector<int64_t>({5, 6, 7, 8})}),
      "Child size less than Row size");

  // No throw with children of same size.
  evalRow(
      4,
      "C0, C1",
      {makeFlatVector<int32_t>({1, 2, 3, 4}),
       makeFlatVector<int64_t>({5, 6, 7, 8})});
}

} // namespace
