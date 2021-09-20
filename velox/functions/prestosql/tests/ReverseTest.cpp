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

#include "velox/functions/prestosql/tests/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::functions::test;

namespace {
class ReverseTest : public FunctionBaseTest {
 protected:
  template <typename T>
  void testExpr(
      const VectorPtr& expected,
      const std::string& expression,
      const VectorPtr& input) {
    auto result = evaluate<T>(expression, makeRowVector({input}));
    assertEqualVectors(expected, result);
  }
};

TEST_F(ReverseTest, intArrays) {
  auto array1 = makeNullableArrayVector<int64_t>({
      {1, -2, 3, std::nullopt, 4, 5, 6, std::nullopt},
      {1, 2, -2, 1},
      {3, 8, std::nullopt},
      {1, 8},
  });

  auto expected = makeNullableArrayVector<int64_t>({
      {std::nullopt, 6, 5, 4, std::nullopt, 3, -2, 1},
      {1, -2, 2, 1},
      {std::nullopt, 8, 3},
      {8, 1},
  });
  testExpr<ArrayVector>(expected, "reverse(C0)", array1);
}

TEST_F(ReverseTest, doubleArrays) {
  auto array1 = makeNullableArrayVector<double>({
      {1, 2, -2, 1},
      {3, 8, std::nullopt},
      {1, 8},
  });

  auto expected = makeNullableArrayVector<double>({
      {1, -2, 2, 1},
      {std::nullopt, 8, 3},
      {8, 1},
  });
  testExpr<ArrayVector>(expected, "reverse(C0)", array1);
}

TEST_F(ReverseTest, stringArrays) {
  using S = StringView;
  auto array1 = makeNullableArrayVector<StringView>(
      {{S("a"), S("b"), S("c")}, {S("mno"), S("xyz"), std::nullopt}, {}});

  auto expected = makeNullableArrayVector<StringView>(
      {{S("c"), S("b"), S("a")}, {std::nullopt, S("xyz"), S("mno")}, {}});

  testExpr<ArrayVector>(expected, "reverse(C0)", array1);
}

TEST_F(ReverseTest, constantArray) {
  auto array1 = makeNullableArrayVector<int32_t>({
      {1, 2, 3},
  });

  auto constant = BaseVector::wrapInConstant(1, 0, array1);
  ASSERT_TRUE(constant->isConstantEncoding());

  auto expected = makeNullableArrayVector<int32_t>({
      {3, 2, 1},
  });

  testExpr<SimpleVector<ComplexType>>(expected, "reverse(C0)", constant);
}

TEST_F(ReverseTest, complexArray) {
  auto createArrayOfArrays =
      [&](std::vector<std::vector<std::optional<int32_t>>> data) {
        auto baseArray = makeNullableArrayVector<int32_t>(data);

        vector_size_t size = 2;
        BufferPtr nulls = AlignedBuffer::allocate<bool>(size, pool());
        BufferPtr offsets =
            AlignedBuffer::allocate<vector_size_t>(size, pool());
        BufferPtr sizes = AlignedBuffer::allocate<vector_size_t>(size, pool());

        auto rawOffsets = offsets->asMutable<vector_size_t>();
        auto rawSizes = sizes->asMutable<vector_size_t>();
        auto rawNulls = nulls->asMutable<uint64_t>();

        for (int i = 0; i < 2; i++) {
          bits::setNull(rawNulls, i, false);
          rawOffsets[i] = 2 * i;
          rawSizes[i] = 2;
        }

        return std::make_shared<ArrayVector>(
            pool(),
            ARRAY(CppToType<Array<int32_t>>::create()),
            nulls,
            size,
            offsets,
            sizes,
            baseArray,
            0);
      };

  auto arrayOfArrays =
      createArrayOfArrays({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}});
  auto expected =
      createArrayOfArrays({{4, 5, 6}, {1, 2, 3}, {10, 11, 12}, {7, 8, 9}});

  testExpr<ArrayVector>(expected, "reverse(C0)", arrayOfArrays);
}
} // namespace
