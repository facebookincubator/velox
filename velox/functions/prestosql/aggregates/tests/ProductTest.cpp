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
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/functions/lib/aggregates/tests/AggregationTestBase.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::aggregate::test {

namespace {

class ProductTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    allowInputShuffle();
  }
};

TEST_F(ProductTest, globalEmpty) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>(std::vector<int32_t>{}),
  });

  testAggregations({data}, {}, {"product(c0)"}, "SELECT NULL");
}

TEST_F(ProductTest, globalNulls) {
  auto data = makeRowVector({
      makeAllNullFlatVector<int32_t>(100),
      makeFlatVector<int32_t>(
          100, [](auto row) { return row; }, nullEvery(2)),
  });

  // All nulls.
  testAggregations({data}, {}, {"product(c0)"}, "SELECT NULL");

  // Every other is null.
  int64_t product = 1;
  for (auto i = 1; i < 100; i += 2) {
    product *= i;
  }

  auto expected = makeRowVector({
      makeConstant(product, 1),
  });

  testAggregations({data}, {}, {"product(c1)"}, {expected});
}

TEST_F(ProductTest, globalIntegers) {
  auto data = makeRowVector({
      makeFlatVector<int8_t>(100, [](auto row) { return row / 7; }),
      makeFlatVector<int16_t>(100, [](auto row) { return row / 7; }),
      makeFlatVector<int32_t>(100, [](auto row) { return row / 7; }),
      makeFlatVector<int64_t>(100, [](auto row) { return row / 7; }),
  });

  int64_t product = 1;
  for (int32_t i = 0; i < 100; ++i) {
    product *= (i / 7);
  }

  auto expected = makeRowVector({
      makeFlatVector(std::vector<int64_t>{
          product,
      }),
  });

  testAggregations({data}, {}, {"product(c0)"}, {expected});
  testAggregations({data}, {}, {"product(c1)"}, {expected});
  testAggregations({data}, {}, {"product(c2)"}, {expected});
  testAggregations({data}, {}, {"product(c3)"}, {expected});
}

TEST_F(ProductTest, groupByNulls) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>(100, [](auto row) { return row / 10; }),
      makeFlatVector<int32_t>(
          100,
          [](auto row) { return row; },
          [](auto row) {
            // All values in group 3 are null.
            if (row / 10 == 3) {
              return true;
            }

            // Every other value is null.
            return row % 2 == 0;
          }),
  });

  auto expected = makeRowVector({
      makeFlatVector<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      makeFlatVector<int64_t>(
          10,
          [](auto row) {
            int64_t product = 1;
            for (int32_t i = 1; i < 10; i += 2) {
              product *= (row * 10 + i);
            }
            return product;
          },
          [](auto row) { return row == 3; }),
  });

  testAggregations({data}, {"c0"}, {"product(c1)"}, {expected});
}

TEST_F(ProductTest, groupByIntegers) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>(100, [](auto row) { return row / 10; }),
      makeFlatVector<int8_t>(100, [](auto row) { return row; }),
      makeFlatVector<int16_t>(100, [](auto row) { return row; }),
      makeFlatVector<int32_t>(100, [](auto row) { return row; }),
      makeFlatVector<int64_t>(100, [](auto row) { return row; }),
  });

  auto expected = makeRowVector({
      makeFlatVector<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      makeFlatVector<int64_t>(
          10,
          [](auto row) {
            int64_t product = 1;
            for (int32_t i = 0; i < 10; ++i) {
              product *= (row * 10 + i);
            }
            return product;
          }),
  });

  testAggregations({data}, {"c0"}, {"product(c1)"}, {expected});
  testAggregations({data}, {"c0"}, {"product(c2)"}, {expected});
  testAggregations({data}, {"c0"}, {"product(c3)"}, {expected});
  testAggregations({data}, {"c0"}, {"product(c4)"}, {expected});
}

TEST_F(ProductTest, globalDoubles) {
  auto data = makeRowVector({
      makeFlatVector<float>(100, [](auto row) { return row * 0.1 / 7; }),
      makeFlatVector<double>(100, [](auto row) { return row * 0.1 / 7; }),
  });

  double product = 1.0;
  for (int32_t i = 0; i < 100; ++i) {
    product *= (i * 0.1 / 7);
  }

  auto expected = makeRowVector({
      makeFlatVector(std::vector<double>{
          product,
      }),
  });

  testAggregations({data}, {}, {"product(c0)"}, {expected});
  testAggregations({data}, {}, {"product(c1)"}, {expected});
}

TEST_F(ProductTest, groupByDoubles) {
  auto data = makeRowVector({
      makeFlatVector<int32_t>(100, [](auto row) { return row / 10; }),
      makeFlatVector<float>(100, [](auto row) { return row * 0.1; }),
      makeFlatVector<double>(100, [](auto row) { return row * 0.1; }),
  });

  auto expected = makeRowVector({
      makeFlatVector<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      makeFlatVector<double>(
          10,
          [](auto row) {
            double product = 1.0;
            for (int32_t i = 0; i < 10; ++i) {
              product *= (row + i * 0.1);
            }
            return product;
          }),
  });

  testAggregations({data}, {"c0"}, {"product(c1)"}, {expected});
  testAggregations({data}, {"c0"}, {"product(c2)"}, {expected});
}

} // namespace
} // namespace facebook::velox::aggregate::test
