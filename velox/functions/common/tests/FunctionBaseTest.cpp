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
#include "velox/functions/common/tests/FunctionBaseTest.h"
#include "velox/exec/tests/utils/FunctionUtils.h"
#include "velox/functions/common/CoreFunctions.h"
#include "velox/functions/common/VectorFunctions.h"

namespace facebook::velox::functions::test {
void FunctionBaseTest::SetUpTestCase() {
  exec::test::registerTypeResolver();
  functions::registerFunctions();
  functions::registerVectorFunctions();
}

BufferPtr FunctionBaseTest::makeOddIndices(vector_size_t size) {
  return makeIndices(
      size, [](vector_size_t i) { return 2 * i + 1; }, execCtx_.pool());
}

BufferPtr FunctionBaseTest::makeEvenIndices(vector_size_t size) {
  return makeIndices(
      size, [](vector_size_t i) { return 2 * i; }, execCtx_.pool());
}

// static
VectorPtr FunctionBaseTest::wrapInDictionary(
    BufferPtr indices,
    vector_size_t size,
    VectorPtr vector) {
  return BaseVector::wrapInDictionary(
      BufferPtr(nullptr), indices, size, vector);
}

// static
BufferPtr FunctionBaseTest::makeIndices(
    vector_size_t size,
    std::function<vector_size_t(vector_size_t)> indexAt,
    memory::MemoryPool* pool) {
  BufferPtr indices = AlignedBuffer::allocate<vector_size_t>(size, pool);
  auto rawIndices = indices->asMutable<vector_size_t>();

  for (vector_size_t i = 0; i < size; i++) {
    rawIndices[i] = indexAt(i);
  }

  return indices;
}

template <typename T>
ArrayVectorPtr FunctionBaseTest::makeSample1NullableArrayVector() {
  return nullptr;
}

template <typename T>
ArrayVectorPtr FunctionBaseTest::makeSample1ArrayVector() {
  return nullptr;
}

template <>
ArrayVectorPtr FunctionBaseTest::makeSample1NullableArrayVector<int8_t>() {
  return makeNullableArrayVector<int8_t>(
      {{-1, 0, 1, 2, 3, 4},
       {4, 3, 2, 1, 0, -1, -2},
       {-5, -4, -3, -2, -1},
       {101, 102, 103, 104, std::nullopt},
       {std::nullopt, -1, -2, -3, -4},
       {},
       {std::nullopt}});
}

template <>
ArrayVectorPtr FunctionBaseTest::makeSample1ArrayVector<int8_t>() {
  return makeArrayVector<int8_t>(
      {{-1, 0, 1, 2, 3, 4},
       {4, 3, 2, 1, 0, -1, -2},
       {-5, -4, -3, -2, -1},
       {101, 102, 103, 104, 105},
       {}});
}

template <>
ArrayVectorPtr FunctionBaseTest::makeSample1NullableArrayVector<int16_t>() {
  return makeNullableArrayVector<int16_t>(
      {{-1, 0, 1, 2, 3, 4},
       {4, 3, 2, 1, 0, -1, -2},
       {-5, -4, -3, -2, -1},
       {101, 102, 103, 104, std::nullopt},
       {std::nullopt, -1, -2, -3, -4},
       {},
       {std::nullopt}});
}

template <>
ArrayVectorPtr FunctionBaseTest::makeSample1ArrayVector<int16_t>() {
  return makeArrayVector<int16_t>(
      {{-1, 0, 1, 2, 3, 4},
       {4, 3, 2, 1, 0, -1, -2},
       {-5, -4, -3, -2, -1},
       {101, 102, 103, 104, 105},
       {}});
}

template <>
ArrayVectorPtr FunctionBaseTest::makeSample1NullableArrayVector<int32_t>() {
  return makeNullableArrayVector<int32_t>(
      {{-1, 0, 1, 2, 3, 4},
       {4, 3, 2, 1, 0, -1, -2},
       {-5, -4, -3, -2, -1},
       {101, 102, 103, 104, std::nullopt},
       {std::nullopt, -1, -2, -3, -4},
       {},
       {std::nullopt}});
}

template <>
ArrayVectorPtr FunctionBaseTest::makeSample1ArrayVector<int32_t>() {
  return makeArrayVector<int32_t>(
      {{-1, 0, 1, 2, 3, 4},
       {4, 3, 2, 1, 0, -1, -2},
       {-5, -4, -3, -2, -1},
       {101, 102, 103, 104, 105},
       {}});
}

template <>
ArrayVectorPtr FunctionBaseTest::makeSample1NullableArrayVector<int64_t>() {
  return makeNullableArrayVector<int64_t>(
      {{-1, 0, 1, 2, 3, 4},
       {4, 3, 2, 1, 0, -1, -2},
       {-5, -4, -3, -2, -1},
       {101, 102, 103, 104, std::nullopt},
       {std::nullopt, -1, -2, -3, -4},
       {},
       {std::nullopt}});
}

template <>
ArrayVectorPtr FunctionBaseTest::makeSample1ArrayVector<int64_t>() {
  return makeArrayVector<int64_t>(
      {{-1, 0, 1, 2, 3, 4},
       {4, 3, 2, 1, 0, -1, -2},
       {-5, -4, -3, -2, -1},
       {101, 102, 103, 104, 105},
       {}});
}

template <>
ArrayVectorPtr FunctionBaseTest::makeSample1NullableArrayVector<StringView>() {
  using S = StringView;
  return makeNullableArrayVector<S>({
      {S("red"), S("blue")},
      {std::nullopt, S("blue"), S("yellow"), S("orange")},
      {},
      {S("red"), S("purple"), S("green")},
  });
}

template <>
ArrayVectorPtr FunctionBaseTest::makeSample1ArrayVector<StringView>() {
  using S = StringView;
  return makeArrayVector<S>({
      {S("red"), S("blue")},
      {S("blue"), S("yellow"), S("orange")},
      {},
      {S("red"), S("purple"), S("green")},
  });
}

template <>
ArrayVectorPtr FunctionBaseTest::makeSample1NullableArrayVector<bool>() {
  return makeNullableArrayVector<bool>(
      {{true, false},
       {true},
       {false},
       {},
       {true, false, true, std::nullopt},
       {std::nullopt, true, false, true},
       {false, false, false},
       {true, true, true}});
}

template <>
ArrayVectorPtr FunctionBaseTest::makeSample1ArrayVector<bool>() {
  return makeArrayVector<bool>(
      {{true, false},
       {true},
       {false},
       {},
       {false, false, false},
       {true, true, true}});
}

template <>
ArrayVectorPtr FunctionBaseTest::makeSample1NullableArrayVector<float>() {
  return makeNullableArrayVector<float>(
      {{0.0000, 0.00001},
       {std::nullopt, 1.1, 1.11, -2.2, -1.0},
       {-0.0001, -0.0002, -0.0003},
       {},
       {1.1, 1.22222, 1.33, std::nullopt},
       {-0.00001, -0.0002, 0.0001}});
}

template <>
ArrayVectorPtr FunctionBaseTest::makeSample1ArrayVector<float>() {
  return makeArrayVector<float>(
      {{0.0000, 0.00001},
       {1.1, 1.11, -2.2, -1.0},
       {-0.0001, -0.0002, -0.0003},
       {},
       {1.1, 1.22222, 1.33},
       {-0.00001, -0.0002, 0.0001}});
}

template <>
ArrayVectorPtr FunctionBaseTest::makeSample1NullableArrayVector<double>() {
  return makeNullableArrayVector<double>(
      {{0.0000, 0.00001},
       {std::nullopt, 1.1, 1.11, -2.2, -1.0},
       {-0.0001, -0.0002, -0.0003},
       {},
       {1.1, 1.22222, 1.33, std::nullopt},
       {-0.00001, -0.0002, 0.0001}});
}

template <>
ArrayVectorPtr FunctionBaseTest::makeSample1ArrayVector<double>() {
  return makeArrayVector<double>(
      {{0.0000, 0.00001},
       {1.1, 1.11, -2.2, -1.0},
       {-0.0001, -0.0002, -0.0003},
       {},
       {1.1, 1.22222, 1.33},
       {-0.00001, -0.0002, 0.0001}});
}

ArrayVectorPtr FunctionBaseTest::makeSample1NullableLongVarcharArrayVector() {
  using S = StringView;
  // use > 12 length string to avoid inlining
  return makeNullableArrayVector<S>({
      {S("red shiny car ahead"), S("blue clear sky above")},
      {std::nullopt,
       S("blue clear sky above"),
       S("yellow rose flowers"),
       S("orange beautiful sunset")},
      {},
      {S("red shiny car ahead"),
       S("purple is an elegant color"),
       S("green plants make us happy")},
  });
}

ArrayVectorPtr FunctionBaseTest::makeSample1LongVarcharArrayVector() {
  using S = StringView;
  // use > 12 length string to avoid inlining
  return makeArrayVector<S>({
      {S("red shiny car ahead"), S("blue clear sky above")},
      {S("blue clear sky above"),
       S("yellow rose flowers"),
       S("orange beautiful sunset")},
      {},
      {S("red shiny car ahead"),
       S("purple is an elegant color"),
       S("green plants make us happy")},
  });
}

} // namespace facebook::velox::functions::test
