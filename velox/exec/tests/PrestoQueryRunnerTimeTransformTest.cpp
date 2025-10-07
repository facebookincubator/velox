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

#include "velox/exec/fuzzer/PrestoQueryRunnerIntermediateTypeTransforms.h"
#include "velox/exec/tests/PrestoQueryRunnerIntermediateTypeTransformTestBase.h"

namespace facebook::velox::exec::test {
namespace {

class PrestoQueryRunnerTimeTransformTest
    : public PrestoQueryRunnerIntermediateTypeTransformTestBase {
 public:
  VectorPtr createTimeVector(const vector_size_t size) {
    std::vector<std::optional<int64_t>> values;
    values.reserve(size);

    for (vector_size_t i = 0; i < size; ++i) {
      if (i % 10 == 0) {
        values.push_back(std::nullopt); // Some nulls for testing
      } else {
        // Create realistic TIME values: 0 to 86399999 (23:59:59.999)
        values.push_back((i * 3661000 + 123) % 86400000);
      }
    }

    return makeNullableFlatVector(values, TIME());
  }
};

// Test that TIME is recognized as an intermediate type that needs
// transformation
TEST_F(PrestoQueryRunnerTimeTransformTest, isIntermediateOnlyType) {
  // Core test: TIME should be an intermediate type
  ASSERT_TRUE(isIntermediateOnlyType(TIME()));

  // Complex types containing TIME should also be intermediate types
  ASSERT_TRUE(isIntermediateOnlyType(ARRAY(TIME())));
  ASSERT_TRUE(isIntermediateOnlyType(MAP(TIME(), BIGINT())));
  ASSERT_TRUE(isIntermediateOnlyType(MAP(BIGINT(), TIME())));
  ASSERT_TRUE(isIntermediateOnlyType(ROW({TIME(), BIGINT()})));
}

// Test that TIME vectors can be transformed for fuzzer persistence/testing
TEST_F(PrestoQueryRunnerTimeTransformTest, transformTimeVector) {
  // Core test: verify TIME vector can be transformed (TIME -> BIGINT)
  auto timeVector = createTimeVector(100);
  test(timeVector); // This tests the TIME->BIGINT transform works
}

// Test TIME with all encodings: flat, dictionary, and constant
// Dictionary encodings: 3 variants (no nulls, some nulls, all nulls)
// Constant encodings: 2 variants (non-null constant, null constant)
TEST_F(PrestoQueryRunnerTimeTransformTest, transformTimeEncodings) {
  test(TIME());
}

// Test TIME in arrays with dictionary and constant encodings
// Dictionary encoded arrays: wraps array in dict with various null patterns
// Constant encoded arrays: wraps array as constant vector
TEST_F(PrestoQueryRunnerTimeTransformTest, transformTimeArray) {
  auto timeVector = createTimeVector(100);
  testArray(timeVector); // internally calls testDictionary() and testConstant()
}

// Test TIME in maps with dictionary and constant encodings
// Dictionary encoded maps: wraps map in dict with various null patterns
// Constant encoded maps: wraps map as constant vector
TEST_F(PrestoQueryRunnerTimeTransformTest, transformTimeMap) {
  // create TIME keys without nulls since map keys can't be null
  std::vector<std::optional<int64_t>> keyValues;
  keyValues.reserve(100);
  for (vector_size_t i = 0; i < 100; ++i) {
    keyValues.push_back((i * 3661000 + 123) % 86400000);
  }
  auto keys = makeNullableFlatVector(keyValues, TIME());

  std::vector<std::optional<int64_t>> valueValues;
  valueValues.reserve(100);
  for (vector_size_t i = 0; i < 100; ++i) {
    if (i % 7 == 2) {
      valueValues.push_back(std::nullopt); // Some nulls for testing
    } else {
      valueValues.push_back(1000 + i * 101 + 1);
    }
  }
  auto values = makeNullableFlatVector(valueValues, BIGINT());
  testMap(keys, values); // internally calls testDictionary() and testConstant()
}

// Test TIME in rows with dictionary and constant encodings
// Dictionary encoded rows: wraps row in dict with various null patterns
// Constant encoded rows: wraps row as constant vector
TEST_F(PrestoQueryRunnerTimeTransformTest, transformTimeRow) {
  auto timeVector = createTimeVector(100);
  testRow(
      {timeVector},
      {"time_field"}); // internally calls testDictionary() and testConstant()
}

} // namespace
} // namespace facebook::velox::exec::test
