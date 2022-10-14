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

#include <gtest/gtest.h>

#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

class LazyVectorTest : public testing::Test, public VectorTestBase {};

TEST_F(LazyVectorTest, lazyInDictionary) {
  // We have a dictionary over LazyVector. We load for some indices in
  // the dictionary. We check that the loads on the wrapped lazy
  // vector are properly translated and deduplicated.
  static constexpr int32_t kInnerSize = 100;
  static constexpr int32_t kOuterSize = 1000;
  std::vector<vector_size_t> loadedRows;
  auto lazy = std::make_shared<LazyVector>(
      pool_.get(),
      INTEGER(),
      kInnerSize,
      std::make_unique<test::SimpleVectorLoader>([&](auto rows) {
        for (auto row : rows) {
          loadedRows.push_back(row);
        }
        return makeFlatVector<int32_t>(
            rows.back() + 1, [](auto row) { return row; });
      }));
  auto wrapped = BaseVector::wrapInDictionary(
      nullptr,
      makeIndices(kOuterSize, [](auto row) { return row / 10; }),
      kOuterSize,
      lazy);

  // We expect a single level of dictionary and rows loaded for the selected
  // indices in rows.

  SelectivityVector rows(kOuterSize, false);
  // We select 3 rows, the 2 first fall on 0 and the last on 5 in 'base'.
  rows.setValid(1, true);
  rows.setValid(9, true);
  rows.setValid(55, true);
  rows.updateBounds();
  LazyVector::ensureLoadedRows(wrapped, rows);
  EXPECT_EQ(wrapped->encoding(), VectorEncoding::Simple::DICTIONARY);
  EXPECT_EQ(wrapped->valueVector()->encoding(), VectorEncoding::Simple::FLAT);
  EXPECT_EQ(loadedRows, (std::vector<vector_size_t>{0, 5}));
  assertCopyableVector(wrapped);
}

TEST_F(LazyVectorTest, lazyInCostant) {
  // Wrap Lazy vector in a Constant, load some indices and verify that the
  // results.
  static constexpr int32_t kInnerSize = 100;
  static constexpr int32_t kOuterSize = 1000;
  auto base = makeFlatVector<int32_t>(kInnerSize, [](auto row) { return row; });
  std::vector<vector_size_t> loadedRows;
  auto lazy = std::make_shared<LazyVector>(
      pool_.get(),
      INTEGER(),
      kInnerSize,
      std::make_unique<test::SimpleVectorLoader>([&](auto rows) {
        for (auto row : rows) {
          loadedRows.push_back(row);
        }
        return base;
      }));
  VectorPtr wrapped =
      std::make_shared<ConstantVector<int32_t>>(pool_.get(), 10, 7, lazy);

  SelectivityVector rows(kOuterSize, false);
  rows.setValid(1, true);
  rows.setValid(9, true);
  rows.setValid(55, true);
  rows.updateBounds();
  LazyVector::ensureLoadedRows(wrapped, rows);
  EXPECT_EQ(wrapped->encoding(), VectorEncoding::Simple::CONSTANT);
  EXPECT_EQ(loadedRows, (std::vector<vector_size_t>{7}));

  EXPECT_EQ(wrapped->as<SimpleVector<int32_t>>()->valueAt(1), 7);
  EXPECT_EQ(wrapped->as<SimpleVector<int32_t>>()->valueAt(9), 7);
  EXPECT_EQ(wrapped->as<SimpleVector<int32_t>>()->valueAt(55), 7);
}

TEST_F(LazyVectorTest, lazyInDoubleDictionary) {
  // We have dictionaries over LazyVector. We load for some indices in
  // the top dictionary. The intermediate dictionaries refer to
  // non-loaded items in the base of the LazyVector, including indices
  // past its end. We check that we end up with one level of
  // dictionary and have no dictionaries that are invalid by
  // referring to uninitialized/nonexistent positions.
  static constexpr int32_t kInnerSize = 100;
  static constexpr int32_t kOuterSize = 1000;

  VectorPtr lazy;
  vector_size_t loadEnd = 0;

  auto makeWrapped = [&](BufferPtr nulls) {
    loadEnd = 0;
    lazy = std::make_shared<LazyVector>(
        pool_.get(),
        INTEGER(),
        kOuterSize,
        std::make_unique<test::SimpleVectorLoader>([&](auto rows) {
          loadEnd = rows.back() + 1;
          return makeFlatVector<int32_t>(loadEnd, [](auto row) { return row; });
        }));

    return BaseVector::wrapInDictionary(
        std::move(nulls),
        makeIndices(kInnerSize, [](auto row) { return row; }),
        kInnerSize,
        BaseVector::wrapInDictionary(
            nullptr,
            makeIndices(kOuterSize, [](auto row) { return row; }),
            kOuterSize,
            lazy));
  };

  // We expect a single level of dictionary and rows loaded for kInnerSize first
  // elements of 'lazy'.

  // No nulls.
  auto wrapped = makeWrapped(nullptr);

  SelectivityVector rows(kInnerSize);
  LazyVector::ensureLoadedRows(wrapped, rows);
  EXPECT_EQ(wrapped->encoding(), VectorEncoding::Simple::DICTIONARY);
  EXPECT_EQ(wrapped->valueVector()->encoding(), VectorEncoding::Simple::FLAT);
  EXPECT_EQ(kInnerSize, loadEnd);

  auto expected =
      makeFlatVector<int32_t>(kInnerSize, [](auto row) { return row; });
  assertEqualVectors(wrapped, expected);

  // With nulls.
  wrapped = makeWrapped(makeNulls(kInnerSize, nullEvery(7)));
  LazyVector::ensureLoadedRows(wrapped, rows);
  EXPECT_EQ(wrapped->encoding(), VectorEncoding::Simple::DICTIONARY);
  EXPECT_EQ(wrapped->valueVector()->encoding(), VectorEncoding::Simple::FLAT);
  EXPECT_EQ(kInnerSize, loadEnd);

  expected = makeFlatVector<int32_t>(
      kInnerSize, [](auto row) { return row; }, nullEvery(7));
  assertEqualVectors(wrapped, expected);

  // With nulls at the end.
  wrapped = makeWrapped(makeNulls(kInnerSize, nullEvery(3)));
  LazyVector::ensureLoadedRows(wrapped, rows);
  EXPECT_EQ(wrapped->encoding(), VectorEncoding::Simple::DICTIONARY);
  EXPECT_EQ(wrapped->valueVector()->encoding(), VectorEncoding::Simple::FLAT);
  EXPECT_EQ(kInnerSize - 1, loadEnd);

  expected = makeFlatVector<int32_t>(
      kInnerSize, [](auto row) { return row; }, nullEvery(3));
  assertEqualVectors(wrapped, expected);
}

TEST_F(LazyVectorTest, lazySlice) {
  auto lazy = std::make_shared<LazyVector>(
      pool_.get(),
      INTEGER(),
      100,
      std::make_unique<test::SimpleVectorLoader>([&](auto rows) {
        return makeFlatVector<int32_t>(
            rows.back() + 1, [](auto row) { return row; });
      }));
  EXPECT_THROW(lazy->slice(0, 10), VeloxRuntimeError);
  lazy->loadedVector();
  auto slice = lazy->slice(0, 10);
  for (int i = 0; i < slice->size(); ++i) {
    EXPECT_TRUE(slice->equalValueAt(lazy.get(), i, i));
  }
}

TEST_F(LazyVectorTest, lazyInMultipleDictionaryAllResultantNullRows) {
  // Verifies that lazy loading works for a lazy vector that is wrapped in
  // multiple layers of dictionary encoding such that the rows that it needs to
  // be loaded for all end up pointing to nulls. This results in a zero sized
  // base vector which when wrapped in a dictionary layer can run into invalid
  // internal state for row indices that were not asked to be loaded.
  static constexpr int32_t kVectorSize = 10;
  auto lazy = std::make_shared<LazyVector>(
      pool_.get(),
      INTEGER(),
      kVectorSize,
      std::make_unique<test::SimpleVectorLoader>([&](auto rows) {
        return makeFlatVector<int32_t>(
            rows.back() + 1, [](auto row) { return row; });
      }));
  auto wrapped = BaseVector::wrapInDictionary(
      makeNulls(kVectorSize, [](vector_size_t /*row*/) { return true; }),
      makeIndices(kVectorSize, [](auto row) { return row; }),
      kVectorSize,
      lazy);
  wrapped = BaseVector::wrapInDictionary(
      nullptr,
      makeIndices(kVectorSize, [](auto row) { return row; }),
      kVectorSize,
      wrapped);
  SelectivityVector rows(kVectorSize, true);
  rows.setValid(1, false);
  LazyVector::ensureLoadedRows(wrapped, rows);
  auto expected =
      BaseVector::createNullConstant(lazy->type(), wrapped->size(), pool());
  assertEqualVectors(expected, wrapped);
}

TEST_F(LazyVectorTest, lazyInDictionaryNoRowsToLoad) {
  // Verifies that lazy loading works for a lazy vector that is wrapped a
  // dictionary with no extra nulls when loading for 0 selected rows.
  static constexpr int32_t kVectorSize = 10;
  auto lazy = std::make_shared<LazyVector>(
      pool_.get(),
      INTEGER(),
      kVectorSize,
      std::make_unique<test::SimpleVectorLoader>([&](auto rows) {
        return makeFlatVector<int32_t>(
            rows.back() + 1, [](auto row) { return row; });
      }));
  auto wrapped = BaseVector::wrapInDictionary(
      nullptr,
      makeIndices(kVectorSize, [](auto row) { return row; }),
      kVectorSize,
      lazy);
  SelectivityVector rows(kVectorSize, false);
  LazyVector::ensureLoadedRows(wrapped, rows);
  auto expected = makeFlatVector<int32_t>(0);
  assertEqualVectors(expected, wrapped);
}
