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
#include <bit>

#include <gtest/gtest.h>

#include "velox/functions/lib/NormalizeFloatingPoint.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using facebook::velox::test::assertEqualVectors;

namespace facebook::velox::functions::test {
namespace {

class NormalizeFloatingPointTest : public testing::Test,
                                   public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  static inline const double kOtherNaN =
      std::bit_cast<double>(0x7ff8000000000001ULL);

  static uint64_t bits(double value) {
    return std::bit_cast<uint64_t>(value);
  }

  // Returns the bits of the double at 'row' of 'vector'.
  static uint64_t bitsAt(const VectorPtr& vector, vector_size_t row) {
    return bits(vector->as<SimpleVector<double>>()->valueAt(row));
  }
};

TEST_F(NormalizeFloatingPointTest, scalar) {
  EXPECT_EQ(bits(normalizeFloatingPoint(-0.0)), bits(0.0));
  EXPECT_EQ(bits(normalizeFloatingPoint(0.0)), bits(0.0));
  EXPECT_EQ(bits(normalizeFloatingPoint(kOtherNaN)), bits(kNaN));
  EXPECT_EQ(bits(normalizeFloatingPoint(-kNaN)), bits(kNaN));
  EXPECT_EQ(normalizeFloatingPoint(-1.5), -1.5);
  EXPECT_EQ(
      std::bit_cast<uint32_t>(normalizeFloatingPoint(-0.0f)),
      std::bit_cast<uint32_t>(0.0f));
}

TEST_F(NormalizeFloatingPointTest, containsFloatingPoint) {
  EXPECT_TRUE(containsFloatingPoint(*DOUBLE()));
  EXPECT_TRUE(containsFloatingPoint(*REAL()));
  EXPECT_TRUE(containsFloatingPoint(*ARRAY(ROW({BIGINT(), DOUBLE()}))));
  EXPECT_TRUE(containsFloatingPoint(*MAP(REAL(), BIGINT())));
  EXPECT_FALSE(containsFloatingPoint(*BIGINT()));
  EXPECT_FALSE(containsFloatingPoint(*ARRAY(ROW({BIGINT(), VARCHAR()}))));
}

TEST_F(NormalizeFloatingPointTest, flat) {
  auto vector =
      makeNullableFlatVector<double>({1.0, -0.0, std::nullopt, kOtherNaN, 0.0});
  auto result = normalizeFloatingPoint(vector, pool());
  ASSERT_NE(result, vector);
  assertEqualVectors(vector, result);
  EXPECT_EQ(bitsAt(result, 1), bits(0.0));
  EXPECT_TRUE(result->isNullAt(2));
  EXPECT_EQ(bitsAt(result, 3), bits(kNaN));

  // Nothing to normalize: the same vector is returned.
  vector = makeNullableFlatVector<double>({1.0, 0.0, std::nullopt, kNaN});
  EXPECT_EQ(normalizeFloatingPoint(vector, pool()), vector);

  auto bigints = makeFlatVector<int64_t>({1, 2});
  EXPECT_EQ(normalizeFloatingPoint(bigints, pool()), bigints);
}

TEST_F(NormalizeFloatingPointTest, dictionaryAndConstant) {
  auto dictionary = wrapInDictionary(
      makeIndices({1, 0}), makeFlatVector<double>({1.0, -0.0}));
  auto result = normalizeFloatingPoint(dictionary, pool());
  ASSERT_NE(result, dictionary);
  assertEqualVectors(dictionary, result);
  EXPECT_EQ(bitsAt(result, 0), bits(0.0));

  auto constant = makeConstant<double>(-0.0, 3);
  result = normalizeFloatingPoint(constant, pool());
  ASSERT_TRUE(result->isConstantEncoding());
  EXPECT_EQ(result->size(), 3);
  EXPECT_EQ(bitsAt(result, 2), bits(0.0));

  auto nullConstant = makeNullConstant(TypeKind::DOUBLE, 3);
  EXPECT_EQ(normalizeFloatingPoint(nullConstant, pool()), nullConstant);

  auto constantArray = BaseVector::wrapInConstant(
      2, 1, makeArrayVector<double>({{1.0}, {-0.0}}));
  result = normalizeFloatingPoint(constantArray, pool());
  ASSERT_TRUE(result->isConstantEncoding());
  assertEqualVectors(constantArray, result);
  auto elements = result->wrappedVector()->as<ArrayVector>()->elements();
  EXPECT_EQ(bitsAt(elements, 1), bits(0.0));
}

TEST_F(NormalizeFloatingPointTest, complexTypes) {
  auto array = makeNullableArrayVector<double>({{{1.0, -0.0}}, std::nullopt});
  auto result = normalizeFloatingPoint(array, pool());
  ASSERT_NE(result, array);
  assertEqualVectors(array, result);
  EXPECT_TRUE(result->isNullAt(1));
  EXPECT_EQ(bitsAt(result->as<ArrayVector>()->elements(), 1), bits(0.0));

  auto map = makeMapVector<double, double>({{{-0.0, kOtherNaN}}});
  result = normalizeFloatingPoint(map, pool());
  assertEqualVectors(map, result);
  EXPECT_EQ(bitsAt(result->as<MapVector>()->mapKeys(), 0), bits(0.0));
  EXPECT_EQ(bitsAt(result->as<MapVector>()->mapValues(), 0), bits(kNaN));

  auto ids = makeFlatVector<int64_t>({1, 2});
  auto row = makeRowVector({ids, makeFlatVector<double>({-0.0, 1.0})});
  result = normalizeFloatingPoint(row, pool());
  assertEqualVectors(row, result);
  // Children without floating-point values are reused.
  EXPECT_EQ(result->as<RowVector>()->childAt(0), ids);
  EXPECT_EQ(bitsAt(result->as<RowVector>()->childAt(1), 0), bits(0.0));

  row = makeRowVector({ids, makeFlatVector<double>({0.0, 1.0})});
  EXPECT_EQ(normalizeFloatingPoint(row, pool()), row);
}

} // namespace
} // namespace facebook::velox::functions::test
