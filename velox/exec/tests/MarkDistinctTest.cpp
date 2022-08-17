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

#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;

class MarkDistinctTest : public OperatorTestBase {
 protected:
  template <class NativeType>
  std::function<VectorPtr()> baseFunctorTemplate() {
    return [&]() {
      return makeFlatVector<NativeType>(
          2, [&](vector_size_t row) { return (NativeType)(row % 2); }, nullptr);
    };
  }

  template <class NativeType>
  void markDistinctTest(
      std::function<VectorPtr()> baseFunctor,
      std::function<TypePtr()> exprTypeFunctor = []() {
        using implType = typename CppToType<NativeType>::ImplType;
        return std::make_shared<implType>();
      }) {
    std::vector<RowVectorPtr> vectors;

    VectorPtr base = baseFunctor();

    const vector_size_t baseSize = base->size();
    const vector_size_t size = baseSize * 2;
    auto indices = AlignedBuffer::allocate<vector_size_t>(size, pool());
    auto rawIndices = indices->asMutable<vector_size_t>();
    for (auto i = 0; i < size; ++i) {
      rawIndices[i] = i % (baseSize);
    }
    auto baseEncoded =
        BaseVector::wrapInDictionary(nullptr, indices, size, base);

    vectors.push_back(makeRowVector({baseEncoded}));

    auto field1 =
        std::make_shared<core::FieldAccessTypedExpr>(exprTypeFunctor(), "c0");

    auto distinctCol = makeFlatVector<bool>(
        size, [&](vector_size_t row) { return row < baseSize; }, nullptr);

    RowVectorPtr expectedResults = makeRowVector({baseEncoded, distinctCol});

    auto op = PlanBuilder().values(vectors).markDistinct({field1}).planNode();

    CursorParameters params;
    params.planNode = op;

    auto result = readCursor(params, [](auto) {});
    auto res = result.second[0]->childAt(1);

    assertEqualVectors(distinctCol, res);
  }
};

TEST_F(MarkDistinctTest, basicTinyIntTest) {
  using cppColType = int8_t;
  markDistinctTest<cppColType>(baseFunctorTemplate<cppColType>());
}

TEST_F(MarkDistinctTest, basicSmallIntTest) {
  using cppColType = int16_t;
  markDistinctTest<cppColType>(baseFunctorTemplate<cppColType>());
}

TEST_F(MarkDistinctTest, basicIntTest) {
  using cppColType = int32_t;
  markDistinctTest<cppColType>(baseFunctorTemplate<cppColType>());
}

TEST_F(MarkDistinctTest, basicBigIntTest) {
  using cppColType = int64_t;
  markDistinctTest<cppColType>(baseFunctorTemplate<cppColType>());
}

TEST_F(MarkDistinctTest, basicRealTest) {
  using cppColType = float;
  markDistinctTest<cppColType>(baseFunctorTemplate<cppColType>());
}

TEST_F(MarkDistinctTest, basicDoubleTest) {
  using cppColType = double;
  markDistinctTest<cppColType>(baseFunctorTemplate<cppColType>());
}

TEST_F(MarkDistinctTest, basicBooleanTest) {
  using cppColType = bool;
  markDistinctTest<cppColType>(baseFunctorTemplate<cppColType>());
}

TEST_F(MarkDistinctTest, basicArrayTest) {
  using cppColType = Array<int64_t>;
  auto base = makeArrayVector<int64_t>({
      {1, 2, 3, 4, 5},
      {1, 2, 3},
  });
  markDistinctTest<cppColType>(
      [&]() { return base; }, []() { return CppToType<cppColType>::create(); });
}

TEST_F(MarkDistinctTest, basicMapTest) {
  using cppColType = Map<int8_t, int32_t>;
  auto base = makeMapVector<int8_t, int32_t>(
      {{{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}}, {{1, 1}, {1, 1}, {1, 1}}});
  markDistinctTest<cppColType>(
      [&]() { return base; }, []() { return CppToType<cppColType>::create(); });
}

TEST_F(MarkDistinctTest, basicVarcharTest) {
  using cppColType = StringView;
  auto base = makeFlatVector<StringView>({
      "{1, 2, 3, 4, 5}",
      "{1, 2, 3}",
  });
  markDistinctTest<cppColType>(
      [&]() { return base; }, []() { return CppToType<cppColType>::create(); });
}

TEST_F(MarkDistinctTest, basicRowTest) {
  using cppColType = Row<Array<int64_t>, Map<int8_t, int32_t>>;
  auto base = makeRowVector(
      {makeArrayVector<int64_t>({
           {1, 2, 3, 4, 5},
           {1, 2, 3},
       }),
       makeMapVector<int8_t, int32_t>(
           {{{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
            {{1, 1}, {1, 1}, {1, 1}}})});
  markDistinctTest<cppColType>(
      [&]() { return base; }, []() { return CppToType<cppColType>::create(); });
}