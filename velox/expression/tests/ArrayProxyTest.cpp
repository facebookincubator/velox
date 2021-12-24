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

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "velox/expression/VectorUdfTypeSystem.h"
#include "velox/functions/Udf.h"
#include "velox/functions/prestosql/tests/FunctionBaseTest.h"

namespace facebook::velox {
namespace {
// Function that creates array with values 0...n-1.
// Used all possible functions in the array proxy interface.
template <typename T>
struct Func {
  template <typename TOut>
  bool call(TOut& out, const int64_t& n) {
    for (int i = 0; i < n; i++) {
      switch (i % 4) {
        case 0:
          out.add_item() = i;
          break;
        case 1:
          out.push_back(i);
          break;
        case 2:
          out.add_null();
          break;
        case 3:
          out.resize(out.size() + 1);
          out[out.size() - 1] = i;
          break;
      }
    }
    return true;
  }
};

class ArrayProxyTest : public functions::test::FunctionBaseTest {
 public:
  VectorPtr prepareResult(
      const std::shared_ptr<Type>& arrayType,
      vector_size_t size_ = 1) {
    VectorPtr result;
    BaseVector::ensureWritable(
        SelectivityVector(size_), arrayType, this->execCtx_.pool(), &result);
    return result;
  }

  template <typename T>
  void testE2E() {
    registerFunction<Func, ArrayProxyT<T>, int64_t>({"func"});

    auto result = evaluate(
        "func(c0)",
        makeRowVector(
            {makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10})}));

    std::vector<std::vector<std::optional<T>>> expected;
    for (auto i = 1; i <= 10; i++) {
      expected.push_back({});
      auto& currentExpected = expected[expected.size() - 1];
      for (auto j = 0; j < i; j++) {
        switch (j % 4) {
          case 0:
            currentExpected.push_back(j);
            break;
          case 1:
            currentExpected.push_back(j);
            break;
          case 2:
            currentExpected.push_back(std::nullopt);
            break;
          case 3:
            currentExpected.push_back(j);
            break;
        }
      }
    }
    assertEqualVectors(result, makeNullableArrayVector(expected));
  };
};

TEST_F(ArrayProxyTest, ArrayAddNull) {
  auto result = prepareResult(std::make_shared<ArrayType>(ArrayType(BIGINT())));

  exec::VectorWriter<ArrayProxyT<int64_t>> writer;
  writer.init(*result.get()->as<ArrayVector>());
  writer.setOffset(0);

  auto& proxy = writer.current();
  proxy.add_null();
  proxy.add_null();
  proxy.add_null();
  writer.commit();

  auto expected = std::vector<std::vector<std::optional<int64_t>>>{
      {std::nullopt, std::nullopt, std::nullopt}};
  assertEqualVectors(result, makeNullableArrayVector(expected));
}

TEST_F(ArrayProxyTest, intArrayPushBackNull) {
  auto result = prepareResult(std::make_shared<ArrayType>(ArrayType(BIGINT())));

  exec::VectorWriter<ArrayProxyT<int64_t>> writer;
  writer.init(*result.get()->as<ArrayVector>());
  writer.setOffset(0);

  auto& proxy = writer.current();
  proxy.push_back(std::nullopt);
  proxy.push_back(std::optional<int64_t>{std::nullopt});
  proxy.push_back(std::nullopt);
  writer.commit();

  auto expected = std::vector<std::vector<std::optional<int64_t>>>{
      {std::nullopt, std::nullopt, std::nullopt}};
  assertEqualVectors(result, makeNullableArrayVector(expected));
}

TEST_F(ArrayProxyTest, EmptyArray) {
  auto result = prepareResult(std::make_shared<ArrayType>(ArrayType(BIGINT())));

  exec::VectorWriter<ArrayProxyT<int64_t>> writer;
  writer.init(*result.get()->as<ArrayVector>());
  writer.setOffset(0);

  auto& proxy = writer.current();
  writer.commit();

  auto expected = std::vector<std::vector<std::optional<int64_t>>>{{}};
  assertEqualVectors(result, makeNullableArrayVector(expected));
}

TEST_F(ArrayProxyTest, PushBack) {
  auto result = prepareResult(std::make_shared<ArrayType>(ArrayType(BIGINT())));

  exec::VectorWriter<ArrayProxyT<int64_t>> writer;
  writer.init(*result.get()->as<ArrayVector>());
  writer.setOffset(0);

  auto& proxy = writer.current();
  proxy.push_back(1);
  proxy.push_back(2);
  proxy.push_back(std::optional<int64_t>{3});
  writer.commit();

  auto expected = std::vector<std::vector<std::optional<int64_t>>>{{1, 2, 3}};
  assertEqualVectors(result, makeNullableArrayVector(expected));
}

TEST_F(ArrayProxyTest, AddItem) {
  auto result = prepareResult(std::make_shared<ArrayType>(ArrayType(BIGINT())));

  exec::VectorWriter<ArrayProxyT<int64_t>> writer;
  writer.init(*result.get()->as<ArrayVector>());
  writer.setOffset(0);
  auto& arrayProxy = writer.current();
  {
    auto& intProxy = arrayProxy.add_item();
    intProxy = 1;
  }

  {
    auto& intProxy = arrayProxy.add_item();
    intProxy = 2;
  }

  {
    auto& intProxy = arrayProxy.add_item();
    intProxy = 3;
  }

  writer.commit();

  auto expected = std::vector<std::vector<std::optional<int64_t>>>{{1, 2, 3}};
  assertEqualVectors(result, makeNullableArrayVector(expected));
}

TEST_F(ArrayProxyTest, Subscript) {
  auto result = prepareResult(std::make_shared<ArrayType>(ArrayType(BIGINT())));

  exec::VectorWriter<ArrayProxyT<int64_t>> writer;
  writer.init(*result.get()->as<ArrayVector>());
  writer.setOffset(0);
  auto& arrayProxy = writer.current();
  arrayProxy.resize(3);
  arrayProxy[0] = std::nullopt;
  arrayProxy[1] = 2;
  arrayProxy[2] = 3;

  writer.commit();

  auto expected =
      std::vector<std::vector<std::optional<int64_t>>>{{std::nullopt, 2, 3}};
  assertEqualVectors(result, makeNullableArrayVector(expected));
}

TEST_F(ArrayProxyTest, MultipleRows) {
  auto expected = std::vector<std::vector<std::optional<int64_t>>>{
      {1, 2, 3},
      {},
      {1, 2, 3, 4, 5, 6, 7},
      {std::nullopt, std::nullopt, 1, 2},
      {},
      {}};
  auto result = prepareResult(
      std::make_shared<ArrayType>(ArrayType(BIGINT())), expected.size());

  exec::VectorWriter<ArrayProxyT<int64_t>> writer;
  writer.init(*result.get()->as<ArrayVector>());

  for (auto i = 0; i < expected.size(); i++) {
    writer.setOffset(i);
    auto& proxy = writer.current();
    // The simple function interface will receive a proxy.
    for (auto j = 0; j < expected[i].size(); j++) {
      proxy.push_back(expected[i][j]);
    }
    // This commit is called by the vector function adapter.
    writer.commit(true);
  }

  assertEqualVectors(result, makeNullableArrayVector(expected));
}

TEST_F(ArrayProxyTest, E2E) {
  testE2E<int8_t>();
  testE2E<int16_t>();
  testE2E<int32_t>();
  testE2E<int64_t>();
  testE2E<float>();
  testE2E<double>();
}
} // namespace
} // namespace facebook::velox
