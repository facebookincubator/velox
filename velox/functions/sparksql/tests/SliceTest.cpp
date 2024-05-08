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
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
class SliceTest : public SparkFunctionBaseTest {
 public:
  void SetUp() override {
    SparkFunctionBaseTest::SetUp();
    // Spark Slice() needs to parse integer literals as INTEGER, not BIGINT.
    options_.parseIntegerAsBigint = false;
  }

 protected:
  void testSlice(
      const std::string& expression,
      const std::vector<VectorPtr>& parameters,
      const ArrayVectorPtr& expectedArrayVector) {
    auto result = evaluate<ArrayVector>(expression, makeRowVector(parameters));
    assertEqualVectors(expectedArrayVector, result);
    EXPECT_NO_THROW(expectedArrayVector->checkRanges());
  }
};

TEST_F(SliceTest, sparkTestCases) {
  auto arrayVector = makeArrayVector<int64_t>({{1, 2, 3, 4, 5, 6}});
  auto expectedArrayVector = makeArrayVector<int64_t>({{1, 2, 3, 4}});
  testSlice("slice(C0, 1, 4)", {arrayVector}, expectedArrayVector);

  // Slice with negative start index.
  expectedArrayVector = makeArrayVector<int64_t>({{4, 5}});
  testSlice("slice(C0, -3, 2)", {arrayVector}, expectedArrayVector);

  // Slice with the 0 start index.
  VELOX_ASSERT_THROW(
      testSlice("slice(C0, 0, 2)", {arrayVector}, expectedArrayVector),
      "SQL array indices start at 1");

  // Slice with string input.
  auto stringArrayVector = makeArrayVector<StringView>({{"a", "b", "c", "d"}});
  auto expectedStringArrayVector = makeArrayVector<StringView>({{"b", "c"}});
  testSlice("slice(C0, 2, 2)", {stringArrayVector}, expectedStringArrayVector);
}

} // namespace facebook::velox::functions::sparksql::test
