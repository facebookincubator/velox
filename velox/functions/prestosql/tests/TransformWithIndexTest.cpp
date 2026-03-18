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

#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

namespace facebook::velox::functions {
namespace {

class TransformWithIndexTest : public test::FunctionBaseTest {};

TEST_F(TransformWithIndexTest, basic) {
  auto data = makeRowVector({
      makeArrayVector<int64_t>({
          {1, 2, 3},
          {4, 5},
          {6, 7, 8, 9},
          {},
      }),
  });

  // Transform each element to elem * index
  auto result = evaluate("transform_with_index(c0, (x, i) -> x * i)", data);

  // Expected: [[1*1, 2*2, 3*3], [4*1, 5*2], [6*1, 7*2, 8*3, 9*4], []]
  auto expected = makeArrayVector<int64_t>({
      {1, 4, 9},
      {4, 10},
      {6, 14, 24, 36},
      {},
  });

  assertEqualVectors(expected, result);
}

TEST_F(TransformWithIndexTest, useOnlyIndex) {
  auto data = makeRowVector({
      makeArrayVector<int64_t>({
          {100, 200, 300},
          {10, 20},
          {},
      }),
  });

  // Transform each element to just the index
  auto result = evaluate("transform_with_index(c0, (x, i) -> i)", data);

  // Expected: indices are 1-based
  auto expected = makeArrayVector<int64_t>({
      {1, 2, 3},
      {1, 2},
      {},
  });

  assertEqualVectors(expected, result);
}

TEST_F(TransformWithIndexTest, useOnlyElement) {
  auto data = makeRowVector({
      makeArrayVector<int64_t>({
          {1, 2, 3},
          {4, 5},
          {},
      }),
  });

  // Transform each element using only the element (ignoring index)
  auto result = evaluate("transform_with_index(c0, (x, i) -> x + 100)", data);

  auto expected = makeArrayVector<int64_t>({
      {101, 102, 103},
      {104, 105},
      {},
  });

  assertEqualVectors(expected, result);
}

TEST_F(TransformWithIndexTest, stringArray) {
  auto data = makeRowVector({
      makeArrayVector<StringView>({
          {"apple", "banana", "cherry"},
          {"hello", "world"},
          {},
      }),
  });

  // Concatenate element with its index
  auto result = evaluate(
      "transform_with_index(c0, (x, i) -> concat(x, cast(i as varchar)))",
      data);

  auto expected = makeArrayVector<StringView>({
      {"apple1", "banana2", "cherry3"},
      {"hello1", "world2"},
      {},
  });

  assertEqualVectors(expected, result);
}

TEST_F(TransformWithIndexTest, nullElements) {
  auto data = makeRowVector({
      makeNullableArrayVector<int64_t>({
          {1, std::nullopt, 3},
          {std::nullopt, 5},
          {6, 7, std::nullopt, 9},
      }),
  });

  // When element is null, index is still available
  auto result = evaluate("transform_with_index(c0, (x, i) -> i)", data);

  // Indices are still generated for null elements
  auto expected = makeArrayVector<int64_t>({
      {1, 2, 3},
      {1, 2},
      {1, 2, 3, 4},
  });

  assertEqualVectors(expected, result);
}

TEST_F(TransformWithIndexTest, nullElementsInComputation) {
  auto data = makeRowVector({
      makeNullableArrayVector<int64_t>({
          {1, std::nullopt, 3},
          {std::nullopt, 5},
      }),
  });

  // When computing x + i, null element results in null
  auto result = evaluate("transform_with_index(c0, (x, i) -> x + i)", data);

  auto expected = makeNullableArrayVector<int64_t>({
      {2, std::nullopt, 6},
      {std::nullopt, 7},
  });

  assertEqualVectors(expected, result);
}

TEST_F(TransformWithIndexTest, nullArray) {
  auto data = makeRowVector({
      makeNullableArrayVector<int64_t>({
          {{1, 2, 3}},
          std::nullopt,
          {{4, 5}},
      }),
  });

  auto result = evaluate("transform_with_index(c0, (x, i) -> x + i)", data);

  auto expected = makeNullableArrayVector<int64_t>({
      {{2, 4, 6}},
      std::nullopt,
      {{5, 7}},
  });

  assertEqualVectors(expected, result);
}

TEST_F(TransformWithIndexTest, floatArray) {
  auto data = makeRowVector({
      makeArrayVector<double>({
          {1.5, 2.5, 3.5},
          {0.5, 0.25},
      }),
  });

  // elem * index - need to cast index to double for multiplication
  auto result = evaluate(
      "transform_with_index(c0, (x, i) -> x * cast(i as double))", data);

  auto expected = makeArrayVector<double>({
      {1.5, 5.0, 10.5},
      {0.5, 0.5},
  });

  assertEqualVectors(expected, result);
}

TEST_F(TransformWithIndexTest, conditional) {
  auto data = makeRowVector({
      makeArrayVector<int64_t>({
          {1, 2, 3, 4, 5},
          {10, 20, 30},
      }),
  });

  // Return element for odd indices, index for even indices
  auto result =
      evaluate("transform_with_index(c0, (x, i) -> if(i % 2 = 1, x, i))", data);

  auto expected = makeArrayVector<int64_t>({
      {1, 2, 3, 4, 5}, // indices 1, 2, 3, 4, 5: odd (1, 3, 5) -> elem, even (2,
                       // 4) -> index
      {10, 2, 30}, // indices 1, 2, 3: odd (1, 3) -> elem, even (2) -> index
  });

  assertEqualVectors(expected, result);
}

TEST_F(TransformWithIndexTest, emptyArray) {
  auto data = makeRowVector({
      makeArrayVector<int64_t>({
          {},
          {},
      }),
  });

  auto result = evaluate("transform_with_index(c0, (x, i) -> x * i)", data);

  auto expected = makeArrayVector<int64_t>({
      {},
      {},
  });

  assertEqualVectors(expected, result);
}

TEST_F(TransformWithIndexTest, singleElement) {
  auto data = makeRowVector({
      makeArrayVector<int64_t>({
          {42},
          {100},
      }),
  });

  auto result = evaluate("transform_with_index(c0, (x, i) -> x + i)", data);

  // index is 1-based, so single element gets index 1
  auto expected = makeArrayVector<int64_t>({
      {43},
      {101},
  });

  assertEqualVectors(expected, result);
}

TEST_F(TransformWithIndexTest, largeArray) {
  std::vector<int64_t> largeInput;
  std::vector<int64_t> largeExpected;
  for (int64_t i = 0; i < 1000; ++i) {
    largeInput.push_back(i);
    largeExpected.push_back(i * (i + 1)); // elem * (1-based index)
  }

  auto data = makeRowVector({
      makeArrayVector<int64_t>({largeInput}),
  });

  auto result = evaluate("transform_with_index(c0, (x, i) -> x * i)", data);

  auto expected = makeArrayVector<int64_t>({largeExpected});

  assertEqualVectors(expected, result);
}

TEST_F(TransformWithIndexTest, typeChange) {
  auto data = makeRowVector({
      makeArrayVector<int32_t>({
          {1, 2, 3},
          {4, 5},
      }),
  });

  // Transform to double
  auto result = evaluate(
      "transform_with_index(c0, (x, i) -> cast(x as double) / cast(i as double))",
      data);

  auto expected = makeArrayVector<double>({
      {1.0, 1.0, 1.0},
      {4.0, 2.5},
  });

  assertEqualVectors(expected, result);
}

TEST_F(TransformWithIndexTest, captureClosure) {
  auto data = makeRowVector({
      makeArrayVector<int64_t>({
          {1, 2, 3},
          {4, 5, 6},
      }),
      makeFlatVector<int64_t>({10, 100}),
  });

  // Use captured variable c1 in the lambda
  auto result =
      evaluate("transform_with_index(c0, (x, i) -> x + i + c1)", data);

  auto expected = makeArrayVector<int64_t>({
      {12, 14, 16}, // 1+1+10, 2+2+10, 3+3+10
      {105, 107, 109}, // 4+1+100, 5+2+100, 6+3+100
  });

  assertEqualVectors(expected, result);
}

TEST_F(TransformWithIndexTest, booleanArray) {
  auto data = makeRowVector({
      makeArrayVector<bool>({
          {true, false, true, false},
          {false, true},
      }),
  });

  // Return true for odd indices, false for even
  auto result = evaluate("transform_with_index(c0, (x, i) -> i % 2 = 1)", data);

  auto expected = makeArrayVector<bool>({
      {true, false, true, false},
      {true, false},
  });

  assertEqualVectors(expected, result);
}

TEST_F(TransformWithIndexTest, nestedArray) {
  // Test with array of arrays using a simpler construction
  auto data = makeRowVector({
      makeNestedArrayVectorFromJson<int64_t>({
          "[[1, 2], [3, 4, 5]]",
          "[[6], [7, 8], [9, 10, 11, 12]]",
      }),
  });

  // Return the size of each inner array multiplied by index
  auto result =
      evaluate("transform_with_index(c0, (x, i) -> cardinality(x) * i)", data);

  // Row 0: [[1,2], [3,4,5]] -> [size(2)*1, size(3)*2] = [2, 6]
  // Row 1: [[6], [7,8], [9,10,11,12]] -> [size(1)*1, size(2)*2, size(4)*3] =
  // [1, 4, 12]
  auto expected = makeArrayVector<int64_t>({
      {2, 6},
      {1, 4, 12},
  });

  assertEqualVectors(expected, result);
}

} // namespace

// ============================================================================
// CUSTOM FUZZER TESTS
// These tests use VectorFuzzer to generate random inputs and verify
// properties of transform_with_index:
// 1. Output array size equals input array size
// 2. When using only index, values are 1-based sequential (1, 2, 3, ...)
// 3. Function handles nulls correctly
// ============================================================================

class TransformWithIndexFuzzerTest : public test::FunctionBaseTest {
 protected:
  static SelectivityVector getNonNullRows(const RowVectorPtr& data) {
    SelectivityVector nonNullRows(data->size());
    for (vector_size_t i = 0; i < data->size(); ++i) {
      bool hasNull = false;
      for (vector_size_t j = 0; j < data->childrenSize(); ++j) {
        if (data->childAt(j)->isNullAt(i)) {
          hasNull = true;
          break;
        }
      }
      if (hasNull) {
        nonNullRows.setValid(i, false);
      }
    }
    nonNullRows.updateBounds();
    return nonNullRows;
  }

  void testSizePreservation(const RowVectorPtr& data) {
    auto nonNullRows = getNonNullRows(data);
    if (nonNullRows.countSelected() == 0) {
      return;
    }

    VectorPtr result;
    try {
      result = evaluate("transform_with_index(c0, (x, i) -> i)", data);
    } catch (...) {
      return;
    }

    if (!result) {
      return;
    }

    auto inputArray = data->childAt(0)->as<ArrayVector>();
    auto resultArray = result->as<ArrayVector>();
    if (!inputArray || !resultArray) {
      return;
    }

    for (auto i = 0; i < data->size(); ++i) {
      if (!nonNullRows.isValid(i) || result->isNullAt(i)) {
        continue;
      }

      if (data->childAt(0)->isNullAt(i)) {
        continue;
      }

      auto inputSize = inputArray->sizeAt(i);
      auto resultSize = resultArray->sizeAt(i);
      ASSERT_EQ(resultSize, inputSize)
          << "Result array size should equal input array size at row " << i;
    }
  }

  void testIndexValues(const RowVectorPtr& data) {
    auto nonNullRows = getNonNullRows(data);
    if (nonNullRows.countSelected() == 0) {
      return;
    }

    VectorPtr result;
    try {
      result = evaluate("transform_with_index(c0, (x, i) -> i)", data);
    } catch (...) {
      return;
    }

    if (!result) {
      return;
    }

    auto inputArray = data->childAt(0)->as<ArrayVector>();
    auto resultArray = result->as<ArrayVector>();
    if (!inputArray || !resultArray) {
      return;
    }

    auto resultElements = resultArray->elements()->as<FlatVector<int64_t>>();
    if (!resultElements) {
      return;
    }

    for (auto i = 0; i < data->size(); ++i) {
      if (!nonNullRows.isValid(i) || result->isNullAt(i)) {
        continue;
      }

      if (data->childAt(0)->isNullAt(i)) {
        continue;
      }

      auto offset = resultArray->offsetAt(i);
      auto size = resultArray->sizeAt(i);
      for (auto j = 0; j < size; ++j) {
        auto expectedIndex = j + 1;
        auto actualIndex = resultElements->valueAt(offset + j);
        ASSERT_EQ(actualIndex, expectedIndex)
            << "Index should be 1-based at row " << i << ", element " << j;
      }
    }
  }

  template <typename T>
  void runFuzzerTest(
      const TypePtr& elementType,
      vector_size_t vectorSize,
      double nullRatio) {
    VectorFuzzer::Options opts;
    opts.vectorSize = vectorSize;
    opts.nullRatio = nullRatio;
    opts.containerLength = 10;
    opts.containerVariableLength = true;
    VectorFuzzer fuzzer(opts, pool());

    auto inputArray = fuzzer.fuzz(ARRAY(elementType));
    auto data = makeRowVector({inputArray});
    testSizePreservation(data);
    testIndexValues(data);
  }
};

TEST_F(TransformWithIndexFuzzerTest, fuzzInteger) {
  runFuzzerTest<int64_t>(BIGINT(), 100, 0.1);
}

TEST_F(TransformWithIndexFuzzerTest, fuzzSmallint) {
  runFuzzerTest<int16_t>(SMALLINT(), 100, 0.1);
}

TEST_F(TransformWithIndexFuzzerTest, fuzzDouble) {
  runFuzzerTest<double>(DOUBLE(), 100, 0.1);
}

TEST_F(TransformWithIndexFuzzerTest, fuzzVarchar) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  opts.stringLength = 20;
  opts.stringVariableLength = true;
  VectorFuzzer fuzzer(opts, pool());

  auto inputArray = fuzzer.fuzz(ARRAY(VARCHAR()));
  auto data = makeRowVector({inputArray});
  testSizePreservation(data);
}

TEST_F(TransformWithIndexFuzzerTest, fuzzHighNullRatio) {
  runFuzzerTest<int64_t>(BIGINT(), 100, 0.5);
}

TEST_F(TransformWithIndexFuzzerTest, fuzzLargeVectors) {
  runFuzzerTest<int64_t>(BIGINT(), 500, 0.1);
}

TEST_F(TransformWithIndexFuzzerTest, fuzzEmptyContainers) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 2;
  opts.containerVariableLength = true;
  VectorFuzzer fuzzer(opts, pool());

  auto inputArray = fuzzer.fuzz(ARRAY(INTEGER()));
  auto data = makeRowVector({inputArray});
  testSizePreservation(data);
  testIndexValues(data);
}

TEST_F(TransformWithIndexFuzzerTest, fuzzWithElementTransformation) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  VectorFuzzer fuzzer(opts, pool());

  for (int iter = 0; iter < 10; ++iter) {
    auto inputArray = fuzzer.fuzz(ARRAY(BIGINT()));
    auto data = makeRowVector({inputArray});

    VectorPtr result;
    try {
      result = evaluate("transform_with_index(c0, (x, i) -> x + i)", data);
    } catch (...) {
      continue;
    }

    if (!result) {
      continue;
    }

    auto inputArrayVec = data->childAt(0)->as<ArrayVector>();
    auto resultArray = result->as<ArrayVector>();
    if (!inputArrayVec || !resultArray) {
      continue;
    }

    for (auto i = 0; i < data->size(); ++i) {
      if (data->childAt(0)->isNullAt(i) || result->isNullAt(i)) {
        continue;
      }

      ASSERT_EQ(resultArray->sizeAt(i), inputArrayVec->sizeAt(i))
          << "Result array size should equal input array size at row " << i;
    }
  }
}

TEST_F(TransformWithIndexFuzzerTest, fuzzDictionaryEncoded) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  VectorFuzzer fuzzer(opts, pool());

  for (int iter = 0; iter < 10; ++iter) {
    auto baseArray = fuzzer.fuzz(ARRAY(BIGINT()));
    auto dictArray = fuzzer.fuzzDictionary(baseArray, opts.vectorSize);

    auto data = makeRowVector({dictArray});

    VectorPtr result;
    try {
      result = evaluate("transform_with_index(c0, (x, i) -> i)", data);
    } catch (...) {
      continue;
    }

    if (!result) {
      continue;
    }

    ASSERT_NE(result, nullptr);
  }
}

TEST_F(TransformWithIndexFuzzerTest, fuzzConditionalTransformation) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  VectorFuzzer fuzzer(opts, pool());

  for (int iter = 0; iter < 10; ++iter) {
    auto inputArray = fuzzer.fuzz(ARRAY(BIGINT()));
    auto data = makeRowVector({inputArray});

    VectorPtr result;
    try {
      result = evaluate(
          "transform_with_index(c0, (x, i) -> if(i % 2 = 1, x, i))", data);
    } catch (...) {
      continue;
    }

    if (!result) {
      continue;
    }

    auto inputArrayVec = data->childAt(0)->as<ArrayVector>();
    auto resultArray = result->as<ArrayVector>();
    if (!inputArrayVec || !resultArray) {
      continue;
    }

    for (auto i = 0; i < data->size(); ++i) {
      if (data->childAt(0)->isNullAt(i) || result->isNullAt(i)) {
        continue;
      }

      ASSERT_EQ(resultArray->sizeAt(i), inputArrayVec->sizeAt(i))
          << "Result array size should equal input array size at row " << i;
    }
  }
}

} // namespace facebook::velox::functions
