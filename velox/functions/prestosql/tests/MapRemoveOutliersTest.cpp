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

#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

namespace facebook::velox::functions {
namespace {

class MapRemoveOutliersTest : public test::FunctionBaseTest {};

TEST_F(MapRemoveOutliersTest, basicIntegerValues) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int32_t, int32_t>({
          "{1:10, 2:20, 3:30, 4:40, 5:50}",
      }),
      makeFlatVector<double>({3.0}),
  });

  auto result = evaluate("map_remove_outliers(c0, c1)", data);

  auto expected = makeMapVectorFromJson<int32_t, int32_t>({
      "{1:10, 2:20, 3:30, 4:40, 5:50}",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapRemoveOutliersTest, emptyMap) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int32_t, int32_t>({
          "{}",
      }),
      makeFlatVector<double>({2.0}),
  });

  auto result = evaluate("map_remove_outliers(c0, c1)", data);

  auto expected = makeMapVectorFromJson<int32_t, int32_t>({
      "{}",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapRemoveOutliersTest, singleValue) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int32_t, int32_t>({
          "{1:100}",
      }),
      makeFlatVector<double>({2.0}),
  });

  auto result = evaluate("map_remove_outliers(c0, c1)", data);

  auto expected = makeMapVectorFromJson<int32_t, int32_t>({
      "{1:100}",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapRemoveOutliersTest, allSameValues) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int32_t, int32_t>({
          "{1:50, 2:50, 3:50, 4:50}",
      }),
      makeFlatVector<double>({1.0}),
  });

  auto result = evaluate("map_remove_outliers(c0, c1)", data);

  auto expected = makeMapVectorFromJson<int32_t, int32_t>({
      "{1:50, 2:50, 3:50, 4:50}",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapRemoveOutliersTest, nullValues) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int32_t, int32_t>({
          "{1:10, 2:null, 3:30, 4:40}",
      }),
      makeFlatVector<double>({2.0}),
  });

  auto result = evaluate("map_remove_outliers(c0, c1)", data);

  auto expected = makeMapVectorFromJson<int32_t, int32_t>({
      "{1:10, 2:null, 3:30, 4:40}",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapRemoveOutliersTest, varcharValues) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int32_t, std::string>({
          "{1:\"apple\", 2:\"banana\", 3:\"cherry\"}",
      }),
      makeFlatVector<double>({2.0}),
  });

  auto result = evaluate("map_remove_outliers(c0, c1)", data);

  auto expected = makeMapVectorFromJson<int32_t, std::string>({
      "{1:\"apple\", 2:\"banana\", 3:\"cherry\"}",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapRemoveOutliersTest, zeroThreshold) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int32_t, int32_t>({
          "{1:10, 2:20, 3:30}",
      }),
      makeFlatVector<double>({0.0}),
  });

  auto result = evaluate("map_remove_outliers(c0, c1)", data);

  auto expected = makeMapVectorFromJson<int32_t, int32_t>({
      "{2:20}",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapRemoveOutliersTest, largeThreshold) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int32_t, int32_t>({
          "{1:10, 2:20, 3:30, 4:1000}",
      }),
      makeFlatVector<double>({10.0}),
  });

  auto result = evaluate("map_remove_outliers(c0, c1)", data);

  auto expected = makeMapVectorFromJson<int32_t, int32_t>({
      "{1:10, 2:20, 3:30, 4:1000}",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapRemoveOutliersTest, allNullValues) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int32_t, int32_t>({
          "{1:null, 2:null, 3:null}",
      }),
      makeFlatVector<double>({2.0}),
  });

  auto result = evaluate("map_remove_outliers(c0, c1)", data);

  auto expected = makeMapVectorFromJson<int32_t, int32_t>({
      "{1:null, 2:null, 3:null}",
  });

  assertEqualVectors(expected, result);
}

} // namespace
} // namespace facebook::velox::functions
