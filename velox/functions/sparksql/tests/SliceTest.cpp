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
#include "velox/functions/lib/Slice.h"
#include "velox/functions/lib/tests/SliceTestBase.h"

namespace facebook::velox::functions::sparksql::test {

namespace {
class SliceTest : public SliceTestBase {
 protected:
  void SetUp() override {
    FunctionBaseTest::SetUp();
    // Parses integer literals as INTEGER, not BIGINT.
    options_.parseIntegerAsBigint = false;
    registerIntegerSliceFunction("spark_");
  }

  void testSlice(
      const std::string& expression,
      const std::vector<VectorPtr>& parameters,
      const ArrayVectorPtr& expectedArrayVector) override {
    auto result =
        evaluate<ArrayVector>("spark_" + expression, makeRowVector(parameters));
    assertEqualVectors(expectedArrayVector, result);
    EXPECT_NO_THROW(expectedArrayVector->checkRanges());
  }
};

TEST_F(SliceTest, basic) {
  basicTestCases();
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
