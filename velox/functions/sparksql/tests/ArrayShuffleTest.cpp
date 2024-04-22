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

#include "velox/expression/VectorReaders.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

using namespace facebook::velox::test;

class ArrayShuffleTest : public SparkFunctionBaseTest {
 protected:
  template <typename T>
  VectorPtr
  testShuffle(const VectorPtr& input, int64_t seed, int32_t partitionId = 0) {
    setSparkPartitionId(partitionId);
    return evaluate(
        fmt::format("shuffle(c0, {})", seed), makeRowVector({input}));
  }

  template <typename T>
  void compareResult(
      const VectorPtr& result,
      const VectorPtr& expected,
      bool equal) {
    DecodedVector decodedResult(*result.get());
    exec::VectorReader<Array<T>> readerResult(&decodedResult);
    DecodedVector decodedExpected(*expected.get());
    exec::VectorReader<Array<T>> readerExpected(&decodedExpected);
    for (auto i = 0; i < decodedExpected.size(); i++) {
      auto resultArray = readerResult[i].materialize();
      auto expectedArray = readerExpected[i].materialize();
      if (equal) {
        ASSERT_TRUE(std::equal(
            resultArray.begin(), resultArray.end(), expectedArray.begin()));
      } else {
        if (resultArray.size() > 2) {
          ASSERT_FALSE(std::equal(
              resultArray.begin(), resultArray.end(), expectedArray.begin()));
        }
        ASSERT_TRUE(std::is_permutation(
            resultArray.begin(), resultArray.end(), expectedArray.begin()));
      }
    }
  }
};

TEST_F(ArrayShuffleTest, basic) {
  auto input = makeArrayVector<int64_t>({{1, 2, 3, 4, 5}});
  auto result = makeArrayVector<int64_t>({{3, 5, 4, 1, 2}});
  auto stringInput = makeArrayVector<std::string>({{"a", "b", "c", "d"}});
  auto stringResult = makeArrayVector<std::string>({{"a", "c", "b", "d"}});
  compareResult<int64_t>(testShuffle<int64_t>(input, 0), result, true);
  compareResult<std::string>(
      testShuffle<std::string>(stringInput, 0), stringResult, true);

  // Assert results are different with different seeds / partition ids.
  compareResult<int64_t>(
      testShuffle<int64_t>(input, 0, 0),
      testShuffle<int64_t>(input, 0, 1),
      false);
  compareResult<int64_t>(
      testShuffle<int64_t>(input, 1, 0),
      testShuffle<int64_t>(input, 0, 0),
      false);
}

TEST_F(ArrayShuffleTest, nestedArrays) {
  using innerArrayType = std::vector<std::optional<int64_t>>;
  using outerArrayType =
      std::vector<std::optional<std::vector<std::optional<int64_t>>>>;
  innerArrayType a{1, 2, 3, 4};
  innerArrayType b{5, 6};
  innerArrayType c{6, 7, 8};
  outerArrayType row1{{a}, {b}};
  outerArrayType row2{std::nullopt, std::nullopt, {a}, {b}, {c}};
  outerArrayType row3{{}};
  outerArrayType row4{{{std::nullopt}}};
  auto input =
      makeNullableNestedArrayVector<int64_t>({{row1}, {row2}, {row3}, {row4}});
  compareResult<Array<int64_t>>(
      testShuffle<Array<int64_t>>(input, 0, 0),
      testShuffle<int64_t>(input, 0, 0),
      true);
}

TEST_F(ArrayShuffleTest, constantEncoding) {
  vector_size_t size = 2;
  // Test empty array, array with null element,
  // array with duplicate elements, and array with distinct values.
  auto valueVector = makeNullableArrayVector<int64_t>(
      {{}, {std::nullopt, 0}, {5, 5}, {1, 2, 3}});

  for (auto i = 0; i < valueVector->size(); i++) {
    auto input = BaseVector::wrapInConstant(size, i, valueVector);
    compareResult<int64_t>(
        testShuffle<int64_t>(input, 0), testShuffle<int64_t>(input, 0), true);
    compareResult<int64_t>(
        testShuffle<int64_t>(input, 0), testShuffle<int64_t>(input, 1), false);
  }
}

TEST_F(ArrayShuffleTest, dictEncoding) {
  // Test dict with repeated elements: {1,2,3} x 3, {4,5} x 2.
  auto base = makeNullableArrayVector<int64_t>(
      {{0},
       {1, 2, 3},
       {4, 5, std::nullopt},
       {1, 2, 3},
       {1, 2, 3},
       {4, 5, std::nullopt}});
  // Test repeated index elements and indices filtering (filter out element at
  // index 0).
  auto indices = makeIndices({3, 3, 4, 2, 2, 1, 1, 1});
  auto input = wrapInDictionary(indices, base);

  compareResult<int64_t>(
      testShuffle<int64_t>(input, 0), testShuffle<int64_t>(input, 0), true);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
