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
#include "velox/vector/tests/VectorTestUtils.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook;
using namespace facebook::velox;

class VectorPrepareForReuseTest : public testing::Test,
                                  public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  VectorPrepareForReuseTest() = default;
};

class MemoryAllocationChecker {
 public:
  explicit MemoryAllocationChecker(memory::MemoryPool* pool)
      : pool_{pool}, numAllocations_{pool_->stats().numAllocs} {}

  bool assertOne() {
    bool ok = numAllocations_ + 1 == pool_->stats().numAllocs;
    numAllocations_ = pool_->stats().numAllocs;
    return ok;
  }

  bool assertAtLeastOne() {
    bool ok = numAllocations_ < pool_->stats().numAllocs;
    numAllocations_ = pool_->stats().numAllocs;
    return ok;
  }

  ~MemoryAllocationChecker() {
    EXPECT_EQ(numAllocations_, pool_->stats().numAllocs);
  }

 private:
  memory::MemoryPool* const pool_;
  uint64_t numAllocations_;
};

TEST_F(VectorPrepareForReuseTest, strings) {
  std::vector<std::string> largeStrings = {
      std::string(20, '.'),
      std::string(30, '-'),
      std::string(40, '='),
  };

  auto stringAt = [&](auto row) {
    return row % 3 == 0 ? StringView(largeStrings[(row / 3) % 3]) : ""_sv;
  };

  VectorPtr vector = makeFlatVector<StringView>(1'000, stringAt);
  auto originalBytes = vector->retainedSize();
  BaseVector* originalVector = vector.get();

  // Verify that string buffers get reused rather than appended to.
  {
    MemoryAllocationChecker allocationChecker(pool());

    BaseVector::prepareForReuse(vector, vector->size());
    ASSERT_EQ(originalVector, vector.get());
    ASSERT_EQ(originalBytes, vector->retainedSize());

    // Verify that StringViews are reset to empty strings.
    for (auto i = 0; i < vector->size(); i++) {
      ASSERT_EQ("", vector->asFlatVector<StringView>()->valueAt(i).str());
    }

    for (auto i = 0; i < vector->size(); i++) {
      vector->asFlatVector<StringView>()->set(i, stringAt(i));
    }
    ASSERT_EQ(originalBytes, vector->retainedSize());
  }

  // Verify that string buffers get dropped if not singly referenced.
  auto getStringBuffers = [](const VectorPtr& vector) {
    return vector->asFlatVector<StringView>()->stringBuffers();
  };

  {
    MemoryAllocationChecker allocationChecker(pool());

    auto stringBuffers = getStringBuffers(vector);
    ASSERT_FALSE(stringBuffers.empty());

    BaseVector::prepareForReuse(vector, vector->size());
    ASSERT_EQ(originalVector, vector.get());
    ASSERT_GT(originalBytes, vector->retainedSize());
    ASSERT_TRUE(getStringBuffers(vector).empty());

    for (auto i = 0; i < vector->size(); i++) {
      vector->asFlatVector<StringView>()->set(i, stringAt(i));
    }
    ASSERT_EQ(originalBytes, vector->retainedSize());

    ASSERT_TRUE(allocationChecker.assertAtLeastOne());
  }

  // Verify that only one string buffer is kept for re-use.
  {
    std::vector<std::string> extraLargeStrings = {
        std::string(200, '.'),
        std::string(300, '-'),
        std::string(400, '='),
    };

    VectorPtr extraLargeVector = makeFlatVector<StringView>(
        1'000,
        [&](auto row) { return StringView(extraLargeStrings[row % 3]); });
    ASSERT_LT(1, getStringBuffers(extraLargeVector).size());

    auto originalExtraLargeBytes = extraLargeVector->retainedSize();
    BaseVector* originalExtraLargeVector = extraLargeVector.get();

    MemoryAllocationChecker allocationChecker(pool());

    BaseVector::prepareForReuse(extraLargeVector, extraLargeVector->size());
    ASSERT_EQ(originalExtraLargeVector, extraLargeVector.get());
    ASSERT_GT(originalExtraLargeBytes, extraLargeVector->retainedSize());
    ASSERT_EQ(1, getStringBuffers(extraLargeVector).size());

    for (auto i = 0; i < extraLargeVector->size(); i++) {
      extraLargeVector->asFlatVector<StringView>()->set(i, stringAt(i));
    }
    ASSERT_EQ(originalBytes, extraLargeVector->retainedSize());
  }
}

TEST_F(VectorPrepareForReuseTest, nulls) {
  VectorPtr vector = makeFlatVector<int32_t>(
      1'000, [](auto row) { return row; }, nullEvery(7));
  auto originalBytes = vector->retainedSize();
  BaseVector* originalVector = vector.get();

  // Verify that nulls buffer is reused.
  {
    MemoryAllocationChecker allocationChecker(pool());

    ASSERT_TRUE(vector->nulls() != nullptr);

    BaseVector::prepareForReuse(vector, vector->size());
    ASSERT_EQ(originalVector, vector.get());
    ASSERT_EQ(originalBytes, vector->retainedSize());
  }

  // Verify that nulls buffer is freed if there are no nulls.
  {
    MemoryAllocationChecker allocationChecker(pool());

    for (auto i = 0; i < vector->size(); i++) {
      vector->setNull(i, false);
    }
    ASSERT_TRUE(vector->nulls() != nullptr);
    ASSERT_EQ(originalBytes, vector->retainedSize());

    BaseVector::prepareForReuse(vector, vector->size());
    ASSERT_EQ(originalVector, vector.get());
    ASSERT_TRUE(vector->nulls() == nullptr);
    ASSERT_GT(originalBytes, vector->retainedSize());

    vector->setNull(12, true);
    ASSERT_EQ(originalBytes, vector->retainedSize());

    ASSERT_TRUE(allocationChecker.assertOne());
  }

  // Verify that nulls buffer is dropped if not singly-referenced.
  {
    MemoryAllocationChecker allocationChecker(pool());

    ASSERT_TRUE(vector->nulls() != nullptr);
    ASSERT_EQ(originalBytes, vector->retainedSize());

    auto nulls = vector->nulls();
    BaseVector::prepareForReuse(vector, vector->size());
    ASSERT_EQ(originalVector, vector.get());
    ASSERT_TRUE(vector->nulls() == nullptr);
    ASSERT_GT(originalBytes, vector->retainedSize());

    vector->setNull(12, true);
    ASSERT_EQ(originalBytes, vector->retainedSize());

    ASSERT_TRUE(allocationChecker.assertOne());
  }
}

TEST_F(VectorPrepareForReuseTest, arrays) {
  VectorPtr vector = makeArrayVector<int32_t>(
      1'000,
      [](auto row) { return 1; },
      [](auto row, auto index) { return row + index; });
  auto originalSize = vector->retainedSize();
  BaseVector* originalVector = vector.get();

  auto otherVector = makeArrayVector<int32_t>(
      1'000,
      [](auto row) { return 1; },
      [](auto row, auto index) { return 2 * row + index; });

  MemoryAllocationChecker allocationChecker(pool());
  BaseVector::prepareForReuse(vector, vector->size());
  ASSERT_EQ(originalVector, vector.get());
  ASSERT_EQ(originalSize, vector->retainedSize());

  for (auto i = 0; i < 1'000; i++) {
    ASSERT_EQ(0, vector->as<ArrayVector>()->sizeAt(i));
    ASSERT_EQ(0, vector->as<ArrayVector>()->offsetAt(i));
  }

  vector->copy(otherVector.get(), 0, 0, 1'000);
  ASSERT_EQ(originalSize, vector->retainedSize());
}

TEST_F(VectorPrepareForReuseTest, arrayOfStrings) {
  VectorPtr vector = makeArrayVector<std::string>(
      1'000,
      [](auto /*row*/) { return 1; },
      [](auto row, auto index) {
        return std::string(20 + index, 'a' + row % 5);
      });
  auto originalSize = vector->retainedSize();
  BaseVector* originalVector = vector.get();

  MemoryAllocationChecker allocationChecker(pool());
  BaseVector::prepareForReuse(vector, vector->size());
  ASSERT_EQ(originalVector, vector.get());
  ASSERT_EQ(originalSize, vector->retainedSize());

  auto* arrayVector = vector->as<ArrayVector>();
  for (auto i = 0; i < 1'000; i++) {
    ASSERT_EQ(0, arrayVector->sizeAt(i));
    ASSERT_EQ(0, arrayVector->offsetAt(i));
  }

  // Cannot use BaseVector::copy because it is too smart and acquired string
  // buffers instead of copying the strings.
  auto* elementsVector = arrayVector->elements()->as<FlatVector<StringView>>();
  elementsVector->resize(1'000);
  for (auto i = 0; i < 1'000; i++) {
    arrayVector->setOffsetAndSize(i, i, 1);
    std::string newValue(21, 'b' + i % 7);
    elementsVector->set(i, StringView(newValue));
  }

  ASSERT_EQ(originalSize, vector->retainedSize());
}

TEST_F(VectorPrepareForReuseTest, dataDependentFlags) {
  auto size = 10;

  auto prepareForReuseStatic = [](VectorPtr& vector) {
    BaseVector::prepareForReuse(vector, vector->size());
  };
  auto prepareForReuseInstance = [](VectorPtr& vector) {
    vector->prepareForReuse();
  };

  // Primitive flat vector.
  {
    SCOPED_TRACE("Flat");
    auto createVector = [&]() {
      return test::makeFlatVectorWithFlags<TypeKind::VARCHAR>(size, pool());
    };

    test::checkVectorFlagsReset(
        createVector, prepareForReuseInstance, SelectivityVector{size});
    test::checkVectorFlagsReset(
        createVector, prepareForReuseStatic, SelectivityVector{size});
  }

  // Constant vector.
  {
    SCOPED_TRACE("Constant");
    auto createVector = [&]() {
      return test::makeConstantVectorWithFlags<TypeKind::VARCHAR>(size, pool());
    };

    test::checkVectorFlagsReset(
        createVector, prepareForReuseStatic, SelectivityVector{size});
  }

  // Dictionary vector.
  {
    SCOPED_TRACE("Dictionary");
    auto createVector = [&]() {
      return test::makeDictionaryVectorWithFlags<TypeKind::VARCHAR>(
          size, pool());
    };

    test::checkVectorFlagsReset(
        createVector, prepareForReuseStatic, SelectivityVector{size});
  }

  // Map vector.
  {
    SCOPED_TRACE("Map");
    auto createVector = [&]() {
      return test::makeMapVectorWithFlags<TypeKind::VARCHAR, TypeKind::VARCHAR>(
          size, pool());
    };

    test::checkVectorFlagsReset(
        createVector, prepareForReuseInstance, SelectivityVector{size});
    test::checkVectorFlagsReset(
        createVector, prepareForReuseStatic, SelectivityVector{size});
  }
}

TEST_F(VectorPrepareForReuseTest, recursivelyReusableFlatVector) {
  // Single reference flat vector should be reusable.
  VectorPtr vector = makeFlatVector<int32_t>(100, [](auto row) { return row; });
  ASSERT_TRUE(BaseVector::recursivelyReusable(vector));

  // Multiple references make it non-reusable.
  VectorPtr copy = vector;
  ASSERT_FALSE(BaseVector::recursivelyReusable(vector));
  ASSERT_FALSE(BaseVector::recursivelyReusable(copy));

  // Release the extra reference.
  copy.reset();
  ASSERT_TRUE(BaseVector::recursivelyReusable(vector));
}

TEST_F(VectorPrepareForReuseTest, recursivelyReusableNullVector) {
  // Null vector should return false (not reusable because there's nothing to
  // reuse).
  VectorPtr nullVector = nullptr;
  ASSERT_TRUE(BaseVector::recursivelyReusable(nullVector));
}

TEST_F(VectorPrepareForReuseTest, recursivelyReusableArrayVector) {
  // Single reference array vector with single reference elements.
  VectorPtr vector = makeArrayVector<int32_t>(
      100,
      [](auto row) { return 1; },
      [](auto row, auto index) { return row + index; });
  ASSERT_TRUE(BaseVector::recursivelyReusable(vector));

  // Share the elements - should make it non-reusable.
  auto* arrayVector = vector->as<ArrayVector>();
  VectorPtr elementsCopy = arrayVector->elements();
  ASSERT_FALSE(BaseVector::recursivelyReusable(vector));

  // Release the elements copy.
  elementsCopy.reset();
  ASSERT_TRUE(BaseVector::recursivelyReusable(vector));
}

TEST_F(VectorPrepareForReuseTest, recursivelyReusableRowVector) {
  // Create children vectors first
  auto child0 = makeFlatVector<int32_t>(100, [](auto row) { return row; });
  auto child1 = makeFlatVector<int64_t>(100, [](auto row) { return row * 2; });

  // Create row vector - children are moved in, so row vector owns them
  VectorPtr vector = std::make_shared<RowVector>(
      pool(),
      ROW({{"a", INTEGER()}, {"b", BIGINT()}}),
      nullptr,
      100,
      std::vector<VectorPtr>{child0, child1});

  // At this point, child0 and child1 still hold references
  // so the row vector is NOT reusable
  ASSERT_FALSE(BaseVector::recursivelyReusable(vector));

  // Release the external references to children
  child0.reset();
  child1.reset();
  ASSERT_TRUE(BaseVector::recursivelyReusable(vector));

  // Share a child - should make it non-reusable.
  VectorPtr childCopy = vector->as<RowVector>()->childAt(0);
  ASSERT_FALSE(BaseVector::recursivelyReusable(vector));

  // Release the child copy.
  childCopy.reset();
  ASSERT_TRUE(BaseVector::recursivelyReusable(vector));
}

TEST_F(VectorPrepareForReuseTest, recursivelyReusableMapVector) {
  // Use makeMapVector helper which creates a fully owned structure
  VectorPtr vector =
      makeMapVector<int32_t, int32_t>({{{{1, 10}, {2, 20}}}, {{{3, 30}}}});
  ASSERT_TRUE(BaseVector::recursivelyReusable(vector));

  // Share the keys - should make it non-reusable.
  auto* mapVector = vector->as<MapVector>();
  VectorPtr keysCopy = mapVector->mapKeys();
  ASSERT_FALSE(BaseVector::recursivelyReusable(vector));

  // Release the keys copy.
  keysCopy.reset();
  ASSERT_TRUE(BaseVector::recursivelyReusable(vector));

  // Share values instead.
  VectorPtr valuesCopy = mapVector->mapValues();
  ASSERT_FALSE(BaseVector::recursivelyReusable(vector));

  valuesCopy.reset();
  ASSERT_TRUE(BaseVector::recursivelyReusable(vector));
}

TEST_F(VectorPrepareForReuseTest, recursivelyReusableNestedArrayOfRow) {
  // Test nested structure: Array<Row<int32, int64>>
  // Sharing deeply nested children makes the whole structure non-reusable.
  const auto rowType = ROW({{"a", INTEGER()}, {"b", BIGINT()}});
  constexpr int kNumArrays = 10;
  constexpr int kElementsPerArray = 5;
  constexpr int kTotalElements = kNumArrays * kElementsPerArray;

  // Create the rows RowVector as elements for the ArrayVector.
  auto child0 = makeFlatVector<int32_t>(kTotalElements, [](auto idx) {
    return (idx / kElementsPerArray) * 10 + (idx % kElementsPerArray);
  });
  auto child1 = makeFlatVector<int64_t>(kTotalElements, [](auto idx) {
    return (idx / kElementsPerArray) * 100 + (idx % kElementsPerArray);
  });
  auto rows =
      makeRowVector(rowType->names(), {std::move(child0), std::move(child1)});

  // Build the ArrayVector on top of rows.
  VectorPtr vector =
      makeArrayVector({0, 5, 10, 15, 20, 25, 30, 35, 40, 45}, rows);
  ASSERT_FALSE(BaseVector::recursivelyReusable(vector));

  rows.reset();
  ASSERT_TRUE(BaseVector::recursivelyReusable(vector));

  // Share the nested row's child - should make the whole array non-reusable.
  auto* arrayVector = vector->as<ArrayVector>();
  auto* rowElements = arrayVector->elements()->as<RowVector>();
  VectorPtr nestedChildCopy = rowElements->childAt(0);
  ASSERT_FALSE(BaseVector::recursivelyReusable(vector));

  nestedChildCopy.reset();
  ASSERT_TRUE(BaseVector::recursivelyReusable(vector));
}

TEST_F(VectorPrepareForReuseTest, recursivelyReusableDictionaryVector) {
  // Dictionary vectors are not considered reusable encoding.
  auto flat = makeFlatVector<int32_t>(100, [](auto row) { return row; });
  auto indices = makeIndices(100, [](auto row) { return row; });
  auto dictionary = BaseVector::wrapInDictionary(nullptr, indices, 100, flat);

  ASSERT_FALSE(BaseVector::recursivelyReusable(dictionary));

  flat.reset();
  ASSERT_FALSE(BaseVector::recursivelyReusable(dictionary));
  indices.reset();
  ASSERT_FALSE(BaseVector::recursivelyReusable(dictionary));
}
