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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <optional>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/SelectivityVector.h"
#include "velox/vector/SimpleVector.h"
#include "velox/vector/TypeAliases.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox {
namespace {

// For testing Opaque types.
struct NonPOD {
  static int alive;

  int x;
  NonPOD(int x = 123) : x(x) {
    ++alive;
  }
  ~NonPOD() {
    --alive;
  }
  bool operator==(const NonPOD& other) const {
    return x == other.x;
  }
};

int NonPOD::alive = 0;

class MapVectorToRowVectorTest : public testing::Test,
                                 public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  // Generates a set of test cases for MapVector::toRowVector. Each test case
  // is a MapVector with the same keys but different value types.
  std::vector<MapVectorPtr> makeTestCases() {
    std::vector<MapVectorPtr> testCases;
    auto map = makeMapVectorFromJson<int64_t, int64_t>(
        {"{1:1, 2:2, 3:3}", // All valid values
         "{2:20, 4:40, 7:3}", // Only one projected key
         "{1:100, 2:null, 6:600}", // Null Value
         "{4:4000, 6:6000}", // No projected keys
         "{1:10000}", // single element, projected key
         "{}", // empty map
         "null"}); // null map
    auto keyType = map->type()->asMap().keyType();
    auto intValues = map->mapValues();
    VELOX_CHECK_EQ(intValues->size(), 12);
    // testCases.push_back(map);

    auto makeCopyWithValues = [&](VectorPtr& values) {
      return std::make_shared<MapVector>(
          pool(),
          MAP(keyType, values->type()),
          map->nulls(),
          map->size(),
          map->offsets(),
          map->sizes(),
          map->mapKeys(),
          values);
    };

    // With String Values.
    VectorFuzzer::Options opts{.nullRatio = 0};
    VectorFuzzer fuzzer(opts, pool());
    auto stringValues = fuzzer.fuzzFlat(VARCHAR(), intValues->size());
    stringValues->setNulls(intValues->nulls());
    testCases.push_back(makeCopyWithValues(stringValues));

    // With Array Values.
    auto arrayValues = fuzzer.fuzzFlat(ARRAY(BIGINT()), intValues->size());
    arrayValues->setNulls(intValues->nulls());
    testCases.push_back(makeCopyWithValues(arrayValues));

    // With Row Values.
    auto rowValues = fuzzer.fuzzFlat(ROW({BIGINT()}), intValues->size());
    rowValues->setNulls(intValues->nulls());
    testCases.push_back(makeCopyWithValues(rowValues));

    // With Map Values.
    auto mapValues =
        fuzzer.fuzzFlat(MAP(BIGINT(), BIGINT()), intValues->size());
    mapValues->setNulls(intValues->nulls());
    testCases.push_back(makeCopyWithValues(mapValues));

    auto opaqueValues =
        BaseVector::create(OPAQUE<int>(), intValues->size(), pool());
    auto flat = opaqueValues->asFlatVector<std::shared_ptr<void>>();
    for (int i = 0; i < intValues->size(); i++) {
      flat->set(i, std::make_shared<NonPOD>(i));
    }
    testCases.push_back(makeCopyWithValues(opaqueValues));

    return testCases;
  }

  // Helper function that verifies the result of MapVector::toRowVector by
  // iterating through all rows of the result and looking up its corresponding
  // value in the original MapVector (by looking up the key for that row), if it
  // exists. If key is not found, ensures that its set to null or the default
  // value depending on whether replaceNulls is set.
  void verifyResult(
      const VectorPtr& result,
      const VectorPtr& input,
      SelectivityVector& rows,
      bool replaceNulls) {
    ASSERT_TRUE(result != nullptr);
    // Assert expected types and field names
    ASSERT_EQ(result->type()->asRow().children().size(), 2);
    ASSERT_EQ(result->type()->asRow().nameOf(0), "key1");
    ASSERT_EQ(result->type()->asRow().nameOf(1), "key2");
    ASSERT_EQ(
        result->type()->asRow().childAt(0), input->type()->asMap().valueType());
    ASSERT_EQ(
        result->type()->asRow().childAt(1), input->type()->asMap().valueType());

    // Assert expected number of rows
    ASSERT_EQ(result->size(), rows.end());

    DecodedVector decodedInput(*input);
    auto map = decodedInput.base()->as<MapVector>();
    auto mapKeys = map->mapKeys()->as<SimpleVector<int64_t>>();
    auto mapValues = map->mapValues();
    auto rawOffsets = map->rawOffsets();
    auto rawSizes = map->rawSizes();
    VectorPtr defaultValue = replaceNulls
        ? BaseVector::create(mapValues->type(), 1, pool())
        : nullptr;
    auto verifyDefault = [&](const VectorPtr vec, vector_size_t row) {
      ASSERT_EQ(vec->compare(defaultValue.get(), row, 0), 0)
          << "Expected Default Value: " << defaultValue->toString(0, 1)
          << " Actual Value: " << vec->toString(row, row + 1);
    };

    auto findValueIdx = [&](vector_size_t row,
                            int64_t key) -> std::optional<vector_size_t> {
      auto offset = rawOffsets[decodedInput.index(row)];
      auto size = rawSizes[decodedInput.index(row)];
      for (auto i = offset; i < offset + size; ++i) {
        if (mapKeys->valueAt(i) == key) {
          return i;
        }
      }
      return std::nullopt;
    };
    // Verify expected values
    rows.applyToSelected([&](auto row) {
      for (int64_t key : {1, 2}) {
        auto valueIdx = findValueIdx(row, key);
        auto child = result->as<RowVector>()->childAt(key - 1);
        if (valueIdx.has_value() && !mapValues->isNullAt(valueIdx.value())) {
          ASSERT_TRUE(!child->isNullAt(row));
          if (child->type()->kind() == TypeKind::OPAQUE) {
            auto opaqueValue =
                child->asFlatVector<std::shared_ptr<void>>()->valueAt(row);
            auto expectedOpaqueValue =
                mapValues->as<SimpleVector<std::shared_ptr<void>>>()->valueAt(
                    valueIdx.value());
            EXPECT_EQ(
                *std::static_pointer_cast<NonPOD>(opaqueValue),
                *std::static_pointer_cast<NonPOD>(expectedOpaqueValue));
          } else {
            child->compare(mapValues.get(), row, valueIdx.value());
          }
        } else if (replaceNulls) {
          verifyDefault(child, row);
        } else {
          ASSERT_TRUE(child->isNullAt(row));
        }
      }
    });
  }

  // Helper to create SelectivityVector for different selected patterns
  SelectivityVector createSelectivityVector(
      vector_size_t size,
      const std::string& pattern) {
    SelectivityVector rows(size);
    if (pattern == "all") {
      // All rows selected (default)
    } else if (pattern == "start") {
      // First half selected
      rows.setValidRange(size / 2, size, false);
    } else if (pattern == "end") {
      // Second half selected
      rows.setValidRange(0, size / 2, false);
    } else if (pattern == "mid") {
      // Middle quarter selected
      auto start = size / 4;
      auto end = 3 * size / 4;
      rows.setValidRange(0, start, false);
      rows.setValidRange(end, size, false);
    } else if (pattern == "scattered") {
      // Every 3rd row selected
      rows.clearAll();
      for (auto i = 0; i < size; i += 3) {
        rows.setValid(i, true);
      }
    }
    rows.updateBounds();
    return rows;
  }

  // Helper functions to execute a single test case with different parameters
  // like selected rows and replaceNulls verify the result.
  void executeTestCase(MapVectorPtr& testCase) {
    auto keysToProject = std::vector<Variant>{
        Variant::create<int64_t>(1), Variant::create<int64_t>(2)};
    auto outputFieldNames = std::vector<std::string>{"key1", "key2"};
    MapVector::ToRowVectorOptions options{
        .keysToProject = keysToProject,
        .outputFieldNames = outputFieldNames,
        .replaceNulls = false,
        .throwOnDuplicateKeys = false,
        .allowTopLevelNulls = false};
    for (bool replaceNulls : {false, true}) {
      for (std::string selectedRowsStr :
           {"all", "start", "end", "mid", "scattered"}) {
        if (replaceNulls &&
            testCase->type()->asMap().valueType()->kind() == TypeKind::ROW) {
          continue;
        }
        SCOPED_TRACE(
            "Selected Rows: " + selectedRowsStr +
            "replaceNulls: " + std::to_string(replaceNulls) +
            " MapType: " + testCase->type()->toString());
        auto rows = createSelectivityVector(testCase->size(), selectedRowsStr);
        options.replaceNulls = replaceNulls;
        auto result = MapVector::toRowVector(*testCase, options, rows);
        verifyResult(result, testCase, rows, replaceNulls);
      }
    }
  }
};

TEST_F(MapVectorToRowVectorTest, basic) {
  EXPECT_EQ(NonPOD::alive, 0);
  auto testCases = makeTestCases();
  for (auto& testCase : testCases) {
    executeTestCase(testCase);
  }
  testCases.clear();
  EXPECT_EQ(NonPOD::alive, 0);
}

TEST_F(MapVectorToRowVectorTest, unknownTypeValues) {
  auto testCase = makeMapVectorFromJson<int64_t, int64_t>(
      {"{1:1, 2:2, 3:3}", // All valid values
       "{2:20, 4:40, 7:3}", // Only one projected key
       "{1:100, 2:null, 6:600}", // Null Value
       "{4:4000, 6:6000}", // No projected keys
       "{1:10000}", // single element, projected key
       "{}", // empty map
       "null"}); // null map
  auto unknownTypeValues =
      BaseVector::create(UNKNOWN(), testCase->mapValues()->size(), pool());

  testCase = std::make_shared<MapVector>(
      pool(),
      MAP(BIGINT(), UNKNOWN()),
      testCase->nulls(),
      testCase->size(),
      testCase->offsets(),
      testCase->sizes(),
      testCase->mapKeys(),
      unknownTypeValues);

  SelectivityVector rows(testCase->size());
  auto keysToProject = std::vector<Variant>{
      Variant::create<int64_t>(1), Variant::create<int64_t>(2)};
  auto outputFieldNames = std::vector<std::string>{"key1", "key2"};
  MapVector::ToRowVectorOptions options{
      .keysToProject = keysToProject,
      .outputFieldNames = outputFieldNames,
      .replaceNulls = false,
      .throwOnDuplicateKeys = false,
      .allowTopLevelNulls = false};
  auto result = MapVector::toRowVector(*testCase, options, rows);

  auto expectedType = ROW(outputFieldNames, {UNKNOWN(), UNKNOWN()});
  ASSERT_EQ(*result->type(), *expectedType);
  ASSERT_EQ(result->size(), testCase->size());

  options.replaceNulls = true;
  VELOX_ASSERT_THROW(
      MapVector::toRowVector(*testCase, options, rows),
      "Unsupported type for replacing nulls: UNKNOWN");
}

TEST_F(MapVectorToRowVectorTest, encoded) {
  EXPECT_EQ(NonPOD::alive, 0);
  auto testCases = makeTestCases();
  auto reverseIndices = makeIndicesInReverse(testCases[0]->size());
  auto reverseElementIndices =
      makeIndicesInReverse(testCases[0]->mapKeys()->size());

  // Wrap the top level map with a dictionary and a constant encoding.
  for (auto& testCase : testCases) {
    auto dictOnMap = BaseVector::wrapInDictionary(
        nullptr, reverseIndices, testCase->size(), testCase);
    executeTestCase(testCase);

    auto constOnMap =
        BaseVector::wrapInConstant(testCase->size(), 0, dictOnMap);
    executeTestCase(testCase);
  }

  // Wrap the keys and values with a dictionary encoding.
  for (auto& testCase : testCases) {
    auto elementsSizes = testCase->mapValues()->size();
    auto dictKeys = BaseVector::wrapInDictionary(
        nullptr, reverseElementIndices, elementsSizes, testCase->mapKeys());
    auto dictValues = BaseVector::wrapInDictionary(
        nullptr, reverseElementIndices, elementsSizes, testCase->mapValues());
    testCase->setKeysAndValues(dictKeys, dictValues);
    executeTestCase(testCase);
  }
  testCases.clear();
  EXPECT_EQ(NonPOD::alive, 0);
}

TEST_F(MapVectorToRowVectorTest, errorCases) {
  SelectivityVector rows(1);
  auto validMap = makeMapVectorFromJson<int64_t, int64_t>({"{1:1, 2:2, 3:3}"});

  // Input of typeKind other than Map used
  MapVector::ToRowVectorOptions options{
      .keysToProject = {},
      .outputFieldNames = {},
      .replaceNulls = false,
      .throwOnDuplicateKeys = false,
      .allowTopLevelNulls = false};
  ASSERT_ANY_THROW(
      MapVector::toRowVector(*(validMap->mapKeys()), options, rows));

  // Invalid key type
  auto mapWithStringKeys =
      makeMapVectorFromJson<std::string, int32_t>({"{\"x\": 99, \"y\": 100}"});
  options.keysToProject = {"x", "y"};
  options.outputFieldNames = {"x", "y"};
  VELOX_ASSERT_THROW(
      MapVector::toRowVector(*mapWithStringKeys, options, rows),
      "Only SMALLINT, INTEGER, BIGINT keys are currently supported, instead got VARCHAR");

  // Mismatched Type between keysToProject and map keys
  VELOX_ASSERT_THROW(
      MapVector::toRowVector(*validMap, options, rows),
      "(BIGINT vs. VARCHAR) Key type and the type of keys to project are not the same");

  // Keys to project are not empty
  // Mismatched Type between keysToProject and map keys
  options.keysToProject = {};
  options.outputFieldNames = {"key1", "key2"};
  ASSERT_ANY_THROW(MapVector::toRowVector(*validMap, options, rows));

  // Mismatched size between keysToProject and outputFieldNames
  options.keysToProject = {
      Variant::create<int64_t>(1), Variant::create<int64_t>(2)};
  options.outputFieldNames = {"key1"};
  ASSERT_ANY_THROW(MapVector::toRowVector(*validMap, options, rows));

  // Keys to project cannot contain null
  options.keysToProject = {
      Variant::create<int64_t>(1), Variant::null(TypeKind::BIGINT)};
  options.outputFieldNames = {"key1", "key2"};
  VELOX_ASSERT_THROW(
      MapVector::toRowVector(*validMap, options, rows),
      "Keys to project cannot contain null");

  // Duplicate keys not allowed
  options.keysToProject = {
      Variant::create<int64_t>(1), Variant::create<int64_t>(1)};
  VELOX_ASSERT_THROW(
      MapVector::toRowVector(*validMap, options, rows),
      "Duplicate keys not allowed");

  // Replace nulls is true, but map values are of ROW type
  auto rowValues = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  options.keysToProject = {
      Variant::create<int64_t>(1), Variant::create<int64_t>(2)};
  options.replaceNulls = true;
  auto mapWithRowValues = std::make_shared<MapVector>(
      pool(),
      MAP(validMap->mapKeys()->type(), rowValues->type()),
      validMap->nulls(),
      validMap->size(),
      validMap->offsets(),
      validMap->sizes(),
      validMap->mapKeys(),
      rowValues);
  VELOX_ASSERT_THROW(
      MapVector::toRowVector(*mapWithRowValues, options, rows),
      "Unsupported type for replacing nulls: ROW");
}

} // namespace
} // namespace facebook::velox
