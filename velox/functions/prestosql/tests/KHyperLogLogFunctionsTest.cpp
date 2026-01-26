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

#include "velox/common/memory/HashStringAllocator.h"
#include "velox/functions/lib/KHyperLogLog.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/prestosql/types/KHyperLogLogType.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::common::hll;
using namespace facebook::velox::functions::test;

namespace facebook::velox::functions {
namespace {

class KHyperLogLogFunctionsTest : public functions::test::FunctionBaseTest {
 protected:
  void SetUp() override {
    FunctionBaseTest::SetUp();
    pool_ = memory::memoryManager()->addLeafPool();
    allocator_ = std::make_unique<HashStringAllocator>(pool_.get());
  }

  // Helper to create a KHLL with specific value-UII pairs.
  std::string createKHLL(
      const std::vector<std::pair<int64_t, int64_t>>& valueUiiPairs) {
    auto khll = std::make_unique<
        common::hll::KHyperLogLog<int64_t, HashStringAllocator>>(
        allocator_.get());

    for (const auto& [value, uii] : valueUiiPairs) {
      khll->add(value, uii);
    }

    size_t totalSize = khll->estimatedSerializedSize();
    std::string outputBuffer(totalSize, '\0');
    khll->serialize(outputBuffer.data());
    return outputBuffer;
  }

  // Helper to create array of KHLLs for merge_khll SQL function.
  VectorPtr createKHLLArray(const std::vector<std::string>& serializedKHLLs) {
    return makeArrayVector<std::string>({serializedKHLLs}, KHYPERLOGLOG());
  }

  // Build a map from the result vector for easier verification.
  std::map<int64_t, double> createResultMap(MapVector* mapVector) {
    auto mapSize = mapVector->sizeAt(0);

    auto offset = mapVector->offsetAt(0);
    auto keys = mapVector->mapKeys()->as<SimpleVector<int64_t>>();
    auto values = mapVector->mapValues()->as<SimpleVector<double>>();

    std::map<int64_t, double> resultMap;
    for (vector_size_t i = 0; i < mapSize; ++i) {
      resultMap[keys->valueAt(offset + i)] = values->valueAt(offset + i);
    }
    return resultMap;
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::unique_ptr<HashStringAllocator> allocator_;
};

TEST_F(KHyperLogLogFunctionsTest, cardinalityBasic) {
  auto serialized = createKHLL({{0, 10}, {1, 11}, {2, 12}});
  auto input =
      makeFlatVector<StringView>({StringView(serialized)}, KHYPERLOGLOG());
  auto result = evaluate("cardinality(c0)", makeRowVector({input}));
  auto expected = makeFlatVector<int64_t>({3});

  assertEqualVectors(expected, result);
}

TEST_F(KHyperLogLogFunctionsTest, mergeKhll) {
  // Merge KHLLs with completely disjoint key sets.
  {
    auto khll1 = createKHLL({{0, 10}, {1, 11}, {2, 12}});
    auto khll2 = createKHLL({{3, 13}, {4, 14}, {5, 15}});
    auto khll3 = createKHLL({{6, 16}, {7, 17}, {8, 18}});

    auto arrayInput = createKHLLArray({khll1, khll2, khll3});
    auto result = evaluate("merge_khll(c0)", makeRowVector({arrayInput}));
    EXPECT_FALSE(result->isNullAt(0));
    EXPECT_EQ(result->type(), KHYPERLOGLOG());

    auto cardinalityResult =
        evaluate("cardinality(c0)", makeRowVector({result}));
    auto expected = makeFlatVector<int64_t>({9});
    assertEqualVectors(expected, cardinalityResult);
  }

  // Merge KHLLs with overlapping key sets.
  {
    auto khll1 = createKHLL({{0, 10}, {1, 11}, {2, 12}, {3, 13}, {4, 14}});
    auto khll2 = createKHLL({{3, 13}, {4, 14}, {5, 15}, {6, 16}, {7, 17}});

    auto arrayInput = createKHLLArray({khll1, khll2});
    auto result = evaluate("merge_khll(c0)", makeRowVector({arrayInput}));

    auto cardinalityResult =
        evaluate("cardinality(c0)", makeRowVector({result}));
    auto expected = makeFlatVector<int64_t>({8});
    assertEqualVectors(expected, cardinalityResult);
  }

  // Merge KHLL with single element.
  {
    auto khll = createKHLL({{0, 10}, {1, 11}, {2, 12}});
    auto arrayInput = createKHLLArray({khll});
    auto result = evaluate("merge_khll(c0)", makeRowVector({arrayInput}));

    auto cardinalityResult =
        evaluate("cardinality(c0)", makeRowVector({result}));
    auto expected = makeFlatVector<int64_t>({3});
    assertEqualVectors(expected, cardinalityResult);
  }

  // Merge same KHLL multiple times.
  {
    auto khll = createKHLL({{0, 10}, {1, 11}, {2, 12}});
    auto arrayInput = createKHLLArray({khll, khll, khll});
    auto result = evaluate("merge_khll(c0)", makeRowVector({arrayInput}));

    // Cardinality should still be 3.
    auto cardinalityResult =
        evaluate("cardinality(c0)", makeRowVector({result}));
    auto expected = makeFlatVector<int64_t>({3});
    assertEqualVectors(expected, cardinalityResult);
  }

  // Empty array
  {
    auto elements = makeFlatVector<StringView>({}, KHYPERLOGLOG());
    auto arrayInput = makeArrayVector({0}, elements);
    auto result = evaluate("merge_khll(c0)", makeRowVector({arrayInput}));

    EXPECT_TRUE(result->isNullAt(0));
  }

  // Array with nulls
  {
    auto khll = createKHLL({{0, 10}, {1, 11}, {2, 12}});

    std::vector<std::string> serializedData;
    serializedData.push_back(khll);

    auto elements = makeNullableFlatVector<StringView>(
        {StringView(serializedData[0]),
         std::nullopt,
         StringView(serializedData[0])},
        KHYPERLOGLOG());
    auto arrayInput = makeArrayVector({0}, elements, {3});

    auto result = evaluate("merge_khll(c0)", makeRowVector({arrayInput}));

    EXPECT_FALSE(result->isNullAt(0));

    auto cardinalityResult =
        evaluate("cardinality(c0)", makeRowVector({result}));
    auto expected = makeFlatVector<int64_t>({3});
    assertEqualVectors(expected, cardinalityResult);
  }

  // All nulls in array
  {
    auto elements = makeNullableFlatVector<StringView>(
        {std::nullopt, std::nullopt, std::nullopt}, KHYPERLOGLOG());
    auto arrayInput = makeArrayVector({0}, elements, {3});
    auto result = evaluate("merge_khll(c0)", makeRowVector({arrayInput}));

    EXPECT_TRUE(result->isNullAt(0));
  }

  // Null array
  {
    auto nullArray =
        makeNullableFlatVector<int64_t>({std::nullopt}, ARRAY(KHYPERLOGLOG()));
    auto result = evaluate("merge_khll(c0)", makeRowVector({nullArray}));

    EXPECT_TRUE(result->isNullAt(0));
  }
}

TEST_F(KHyperLogLogFunctionsTest, intersectionCardinalityExact) {
  {
    auto khll1 = createKHLL({{0, 10}, {1, 11}, {2, 12}, {3, 13}, {4, 14}});
    auto khll2 = createKHLL({{3, 13}, {4, 14}, {5, 15}, {6, 16}, {7, 17}});

    auto input1 =
        makeFlatVector<StringView>({StringView(khll1)}, KHYPERLOGLOG());
    auto input2 =
        makeFlatVector<StringView>({StringView(khll2)}, KHYPERLOGLOG());

    auto result = evaluate(
        "intersection_cardinality(c0, c1)", makeRowVector({input1, input2}));
    auto expected = makeFlatVector<int64_t>({2});
    assertEqualVectors(expected, result);
  }

  // Intersection cardinality of completely overlapping KHLLs.
  {
    auto khll = createKHLL({{0, 10}, {1, 11}, {2, 12}, {3, 13}, {4, 14}});

    auto input1 =
        makeFlatVector<StringView>({StringView(khll)}, KHYPERLOGLOG());
    auto input2 =
        makeFlatVector<StringView>({StringView(khll)}, KHYPERLOGLOG());

    auto result = evaluate(
        "intersection_cardinality(c0, c1)", makeRowVector({input1, input2}));
    auto expected = makeFlatVector<int64_t>({5});
    assertEqualVectors(expected, result);
  }

  // Intersection cardinality of empty sets.
  {
    auto emptyKhll = createKHLL({});
    auto nonEmptyKhll = createKHLL({{0, 10}, {1, 11}, {2, 12}});

    auto input1 =
        makeFlatVector<StringView>({StringView(emptyKhll)}, KHYPERLOGLOG());
    auto input2 =
        makeFlatVector<StringView>({StringView(nonEmptyKhll)}, KHYPERLOGLOG());

    auto result = evaluate(
        "intersection_cardinality(c0, c1)", makeRowVector({input1, input2}));
    auto expected = makeFlatVector<int64_t>({0});
    assertEqualVectors(expected, result);

    // Test in reverse order.
    result = evaluate(
        "intersection_cardinality(c0, c1)", makeRowVector({input2, input1}));
    assertEqualVectors(expected, result);
  }

  {
    auto emptyKhll1 = createKHLL({});
    auto emptyKhll2 = createKHLL({});

    auto input1 =
        makeFlatVector<StringView>({StringView(emptyKhll1)}, KHYPERLOGLOG());
    auto input2 =
        makeFlatVector<StringView>({StringView(emptyKhll2)}, KHYPERLOGLOG());

    auto result = evaluate(
        "intersection_cardinality(c0, c1)", makeRowVector({input1, input2}));
    auto expected = makeFlatVector<int64_t>({0});
    assertEqualVectors(expected, result);
  }

  // Intersection cardinality of completely disjoint sets
  {
    auto khll1 = createKHLL({{0, 10}, {1, 11}, {2, 12}});
    auto khll2 = createKHLL({{100, 110}, {101, 111}, {102, 112}});

    auto input1 =
        makeFlatVector<StringView>({StringView(khll1)}, KHYPERLOGLOG());
    auto input2 =
        makeFlatVector<StringView>({StringView(khll2)}, KHYPERLOGLOG());

    auto result = evaluate(
        "intersection_cardinality(c0, c1)", makeRowVector({input1, input2}));
    auto expected = makeFlatVector<int64_t>({0});
    assertEqualVectors(expected, result);
  }

  // Intersection where one is a subset of the other.
  {
    auto khll1 = createKHLL(
        {{0, 10},
         {1, 11},
         {2, 12},
         {3, 13},
         {4, 14},
         {5, 15},
         {6, 16},
         {7, 17},
         {8, 18},
         {9, 19}});
    auto khll2 = createKHLL({{2, 12}, {3, 13}, {4, 14}});

    auto input1 =
        makeFlatVector<StringView>({StringView(khll1)}, KHYPERLOGLOG());
    auto input2 =
        makeFlatVector<StringView>({StringView(khll2)}, KHYPERLOGLOG());

    auto result = evaluate(
        "intersection_cardinality(c0, c1)", makeRowVector({input1, input2}));
    auto expected = makeFlatVector<int64_t>({3});
    assertEqualVectors(expected, result);
  }
}

TEST_F(KHyperLogLogFunctionsTest, intersectionCardinalityApproximate) {
  // KHLL1: 0-4999 (5000 values), KHLL2: 2500-5499 (3000 values).
  // Expected intersection: 2500-4999 (2500 elements).
  {
    auto khll1 = std::make_unique<
        common::hll::KHyperLogLog<int64_t, HashStringAllocator>>(
        allocator_.get());
    auto khll2 = std::make_unique<
        common::hll::KHyperLogLog<int64_t, HashStringAllocator>>(
        allocator_.get());

    for (int64_t i = 0; i < 5000; ++i) {
      khll1->add(i, 100);
    }

    for (int64_t i = 2500; i < 5500; ++i) {
      khll2->add(i, 100);
    }

    std::string serialized1(khll1->estimatedSerializedSize(), '\0');
    khll1->serialize(serialized1.data());

    std::string serialized2(khll2->estimatedSerializedSize(), '\0');
    khll2->serialize(serialized2.data());

    auto input1 =
        makeFlatVector<StringView>({StringView(serialized1)}, KHYPERLOGLOG());
    auto input2 =
        makeFlatVector<StringView>({StringView(serialized2)}, KHYPERLOGLOG());

    auto result = evaluate(
        "intersection_cardinality(c0, c1)", makeRowVector({input1, input2}));

    // Allow 5% error for approximation
    auto actualResult = result->as<FlatVector<int64_t>>()->valueAt(0);
    EXPECT_NEAR(actualResult, 2500, 2500 * 0.05);
  }

  {
    auto khll = std::make_unique<
        common::hll::KHyperLogLog<int64_t, HashStringAllocator>>(
        allocator_.get());

    for (int64_t i = 0; i < 5000; ++i) {
      khll->add(i, 100);
    }

    std::string serialized(khll->estimatedSerializedSize(), '\0');
    khll->serialize(serialized.data());

    auto input1 =
        makeFlatVector<StringView>({StringView(serialized)}, KHYPERLOGLOG());
    auto input2 =
        makeFlatVector<StringView>({StringView(serialized)}, KHYPERLOGLOG());

    auto result = evaluate(
        "intersection_cardinality(c0, c1)", makeRowVector({input1, input2}));

    auto actualResult = result->as<FlatVector<int64_t>>()->valueAt(0);
    EXPECT_NEAR(actualResult, 5000, 5000 * 0.05);
  }
}

TEST_F(KHyperLogLogFunctionsTest, uniquenessDistribution) {
  auto serialized = createKHLL(
      {{0, 0}, {1, 0}, {1, 1}, {2, 0}, {2, 1}, {3, 0}, {3, 1}, {3, 2}});
  auto input =
      makeFlatVector<StringView>({StringView(serialized)}, KHYPERLOGLOG());

  // Uniqueness distribution with default histogram size of 4.
  {
    auto result =
        evaluate("uniqueness_distribution(c0)", makeRowVector({input}));

    EXPECT_FALSE(result->isNullAt(0));
    EXPECT_EQ(result->type()->kind(), TypeKind::MAP);

    auto mapVector = result->as<MapVector>();
    auto mapSize = mapVector->sizeAt(0);
    EXPECT_EQ(mapSize, 4);

    auto resultMap = createResultMap(mapVector);

    std::map<int64_t, double> expected;
    for (int i = 1; i <= mapSize; ++i) {
      expected[i] = 0;
    }
    expected[1] = 0.25;
    expected[2] = 0.5;
    expected[3] = 0.25;

    EXPECT_EQ(resultMap, expected);
  }

  // Uniqueness distribution with custom histogram size.
  {
    auto histSize = makeFlatVector<int64_t>({128});
    auto result = evaluate(
        "uniqueness_distribution(c0, c1)", makeRowVector({input, histSize}));

    EXPECT_FALSE(result->isNullAt(0));
    EXPECT_EQ(result->type()->kind(), TypeKind::MAP);

    auto mapVector = result->as<MapVector>();
    auto mapSize = mapVector->sizeAt(0);
    EXPECT_EQ(mapSize, 128);

    auto resultMap = createResultMap(mapVector);

    std::map<int64_t, double> expected;
    for (int i = 1; i <= mapSize; ++i) {
      expected[i] = 0;
    }
    expected[1] = 0.25;
    expected[2] = 0.5;
    expected[3] = 0.25;

    EXPECT_EQ(resultMap, expected);
  }

  // Uniqueness distribution with empty KHLL. Since minhashSize is 0, the
  // histogram size is 0 and the returned map is empty.
  {
    auto emptyKhll = createKHLL({});
    auto emptyInput =
        makeFlatVector<StringView>({StringView(emptyKhll)}, KHYPERLOGLOG());
    auto result =
        evaluate("uniqueness_distribution(c0)", makeRowVector({emptyInput}));

    EXPECT_FALSE(result->isNullAt(0));
    auto mapVector = result->as<MapVector>();
    auto mapSize = mapVector->sizeAt(0);
    EXPECT_EQ(mapSize, 0);
  }

  // Null inputs
  {
    auto nullKhll =
        makeNullableFlatVector<StringView>({std::nullopt}, KHYPERLOGLOG());
    auto result =
        evaluate("uniqueness_distribution(c0)", makeRowVector({nullKhll}));
    EXPECT_TRUE(result->isNullAt(0));
    auto nullHistSize = makeNullableFlatVector<int64_t>({std::nullopt});
    result = evaluate(
        "uniqueness_distribution(c0, c1)",
        makeRowVector({input, nullHistSize}));
    EXPECT_TRUE(result->isNullAt(0));
  }
}

TEST_F(KHyperLogLogFunctionsTest, jaccardIndex) {
  // Jaccard index with identical sets.
  {
    auto khll = createKHLL({{0, 10}, {1, 11}, {2, 12}, {3, 13}, {4, 14}});
    auto input1 =
        makeFlatVector<StringView>({StringView(khll)}, KHYPERLOGLOG());
    auto input2 =
        makeFlatVector<StringView>({StringView(khll)}, KHYPERLOGLOG());

    auto result =
        evaluate("jaccard_index(c0, c1)", makeRowVector({input1, input2}));
    auto expected = makeFlatVector<double>({1.0});
    assertEqualVectors(expected, result);
  }

  // Jaccard index with completely disjoint sets.
  {
    auto khll1 = createKHLL({{0, 10}, {1, 11}, {2, 12}});
    auto khll2 = createKHLL({{100, 110}, {101, 111}, {102, 112}});

    auto input1 =
        makeFlatVector<StringView>({StringView(khll1)}, KHYPERLOGLOG());
    auto input2 =
        makeFlatVector<StringView>({StringView(khll2)}, KHYPERLOGLOG());

    auto result =
        evaluate("jaccard_index(c0, c1)", makeRowVector({input1, input2}));
    auto expected = makeFlatVector<double>({0.0});
    assertEqualVectors(expected, result);
  }

  // Jaccard index with some overlapping elements.
  {
    auto khll1 = createKHLL({{0, 10}, {1, 11}, {2, 12}, {3, 13}, {4, 14}});
    auto khll2 = createKHLL({{3, 13}, {4, 14}, {5, 15}, {6, 16}, {7, 17}});

    auto input1 =
        makeFlatVector<StringView>({StringView(khll1)}, KHYPERLOGLOG());
    auto input2 =
        makeFlatVector<StringView>({StringView(khll2)}, KHYPERLOGLOG());

    auto result =
        evaluate("jaccard_index(c0, c1)", makeRowVector({input1, input2}));
    auto expected = makeFlatVector<double>({0.2});
    assertEqualVectors(expected, result);
  }

  {
    auto khll1 = createKHLL(
        {{0, 10},
         {1, 11},
         {2, 12},
         {3, 13},
         {4, 14},
         {5, 15},
         {6, 16},
         {7, 17},
         {8, 18},
         {9, 19}});
    auto khll2 = createKHLL({{2, 12}, {3, 13}, {4, 14}});

    auto input1 =
        makeFlatVector<StringView>({StringView(khll1)}, KHYPERLOGLOG());
    auto input2 =
        makeFlatVector<StringView>({StringView(khll2)}, KHYPERLOGLOG());

    auto result =
        evaluate("jaccard_index(c0, c1)", makeRowVector({input1, input2}));
    auto expected = makeFlatVector<double>({0.6666666666666666});
    assertEqualVectors(expected, result);
  }

  // Jaccard index with both sets empty. Jaccard index of two empty sets is
  // defined as 1.0
  {
    auto emptyKhll1 = createKHLL({});
    auto emptyKhll2 = createKHLL({});

    auto input1 =
        makeFlatVector<StringView>({StringView(emptyKhll1)}, KHYPERLOGLOG());
    auto input2 =
        makeFlatVector<StringView>({StringView(emptyKhll2)}, KHYPERLOGLOG());

    auto result =
        evaluate("jaccard_index(c0, c1)", makeRowVector({input1, input2}));
    auto expected = makeFlatVector<double>({1.0});
    assertEqualVectors(expected, result);
  }

  // Jaccard index with an empty and non empty set.
  {
    auto emptyKhll = createKHLL({});
    auto nonEmptyKhll = createKHLL({{0, 10}, {1, 11}, {2, 12}});

    auto input1 =
        makeFlatVector<StringView>({StringView(emptyKhll)}, KHYPERLOGLOG());
    auto input2 =
        makeFlatVector<StringView>({StringView(nonEmptyKhll)}, KHYPERLOGLOG());

    auto result =
        evaluate("jaccard_index(c0, c1)", makeRowVector({input1, input2}));
    auto expected = makeFlatVector<double>({0.0});
    assertEqualVectors(expected, result);

    // Test in reverse order
    result = evaluate("jaccard_index(c0, c1)", makeRowVector({input2, input1}));
    assertEqualVectors(expected, result);
  }
}

TEST_F(KHyperLogLogFunctionsTest, reidentificationPotential) {
  // Different reidentification potential cases
  {
    auto khll = createKHLL({{0, 0}, {1, 0}, {2, 0}});
    auto input = makeFlatVector<StringView>({StringView(khll)}, KHYPERLOGLOG());
    auto threshold = makeFlatVector<int64_t>({2});

    auto result = evaluate(
        "reidentification_potential(c0, c1)",
        makeRowVector({input, threshold}));
    auto expected = makeFlatVector<double>({1.0});
    assertEqualVectors(expected, result);
  }

  {
    auto khll = createKHLL({
        {0, 0},
        {0, 1},
        {0, 2},
        {0, 3},
        {0, 4}, // Value 0: 5 UIIs
        {1, 0},
        {1, 1},
        {1, 2},
        {1, 3}, // Value 1: 4 UIIs
        {2, 0},
        {2, 1},
        {2, 2} // Value 2: 3 UIIs
    });
    auto input = makeFlatVector<StringView>({StringView(khll)}, KHYPERLOGLOG());
    auto threshold = makeFlatVector<int64_t>({2});

    auto result = evaluate(
        "reidentification_potential(c0, c1)",
        makeRowVector({input, threshold}));
    auto expected = makeFlatVector<double>({0.0});
    assertEqualVectors(expected, result);
  }

  // 2 out of 3 values at risk.
  {
    auto khll = createKHLL({
        {0, 0},
        {0, 1}, // Value 0: 2 UIIs
        {1, 0},
        {1, 1},
        {1, 2}, // Value 1: 3 UIIs
        {2, 0} // Value 2: 1 UII
    });
    auto input = makeFlatVector<StringView>({StringView(khll)}, KHYPERLOGLOG());
    auto threshold = makeFlatVector<int64_t>({2});

    auto result = evaluate(
        "reidentification_potential(c0, c1)",
        makeRowVector({input, threshold}));
    auto expected = makeFlatVector<double>({0.6666666666666666});
    assertEqualVectors(expected, result);
  }

  // Reidentification potential with threshold of 0.
  {
    auto khll = createKHLL({{0, 0}, {1, 0}, {2, 0}});
    auto input = makeFlatVector<StringView>({StringView(khll)}, KHYPERLOGLOG());
    auto threshold = makeFlatVector<int64_t>({0});

    auto result = evaluate(
        "reidentification_potential(c0, c1)",
        makeRowVector({input, threshold}));
    auto expected = makeFlatVector<double>({0.0});
    assertEqualVectors(expected, result);
  }

  // Reidentification potential with empty KHLL
  {
    auto emptyKhll = createKHLL({});
    auto input =
        makeFlatVector<StringView>({StringView(emptyKhll)}, KHYPERLOGLOG());
    auto threshold = makeFlatVector<int64_t>({2});

    auto result = evaluate(
        "reidentification_potential(c0, c1)",
        makeRowVector({input, threshold}));
    auto expected = makeFlatVector<double>({0.0});
    assertEqualVectors(expected, result);
  }

  // Reidentification potential with single value with multiple UIIs.
  {
    auto khll = createKHLL({{0, 0}, {0, 1}, {0, 2}});
    auto input = makeFlatVector<StringView>({StringView(khll)}, KHYPERLOGLOG());
    auto threshold = makeFlatVector<int64_t>({2});

    auto result = evaluate(
        "reidentification_potential(c0, c1)",
        makeRowVector({input, threshold}));
    auto expected = makeFlatVector<double>({0.0});
    assertEqualVectors(expected, result);
  }

  // Null inputs
  {
    auto nullKhll =
        makeNullableFlatVector<StringView>({std::nullopt}, KHYPERLOGLOG());
    auto threshold = makeFlatVector<int64_t>({2});

    auto result = evaluate(
        "reidentification_potential(c0, c1)",
        makeRowVector({nullKhll, threshold}));
    EXPECT_TRUE(result->isNullAt(0));

    auto khll = createKHLL({{0, 0}, {1, 0}});
    auto input = makeFlatVector<StringView>({StringView(khll)}, KHYPERLOGLOG());
    auto nullThreshold = makeNullableFlatVector<int64_t>({std::nullopt});
    result = evaluate(
        "reidentification_potential(c0, c1)",
        makeRowVector({input, nullThreshold}));
    EXPECT_TRUE(result->isNullAt(0));
  }
}

} // namespace
} // namespace facebook::velox::functions
