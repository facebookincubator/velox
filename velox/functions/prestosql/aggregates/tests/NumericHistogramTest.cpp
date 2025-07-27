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
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::aggregate::test {

namespace {

class NumericHistogramTest : public AggregationTestBase {};

TEST_F(NumericHistogramTest, DoubleThreeArgs) {
  AggregationTestBase::disableTestIncremental();
  auto input = makeRowVector({
      makeNullableFlatVector<double>({10, 20, 30, 80, 90, 100}),
      makeNullableFlatVector<double>({2, 3, 1, 2, 1, 4}),
  });

  auto expected = makeRowVector({
      makeMapVector<double, double>({
          {{10, 2}, {20, 3}, {30, 1}, {100, 4}, {83.33333333333333, 3}},
      }),
  });

  auto expected2 = makeRowVector({
      makeMapVector<double, double>({
          {{18.333333333333332, 6}, {92.85714285714286, 7}},
      }),
  });
  auto expected4 = makeRowVector({
      makeMapVector<double, double>({
          {{10, 2}, {22.5, 4}, {83.33333333333333, 3}, {100, 4}},
      }),
  });

  testAggregations({input}, {}, {"numeric_histogram(5, c0, c1)"}, {expected});
  testAggregations({input}, {}, {"numeric_histogram(2, c0, c1)"}, {expected2});
  testAggregations({input}, {}, {"numeric_histogram(4, c0, c1)"}, {expected4});

  auto input22 = makeRowVector({
      makeNullableFlatVector<double>({10, 20, 30, 40, 50, 60, 70, 80}),
      makeNullableFlatVector<double>({2, 3, 1, 4, 2, 1, 3, 2}),
  });

  auto expected22 = makeRowVector({
      makeMapVector<double, double>({
          {{10.0, 2}, {40.0, 4}, {53.333333333333336, 3}, {22.5, 4}, {74.0, 5}},
      }),
  });
  testAggregations(
      {input22}, {}, {"numeric_histogram(5, c0, c1)"}, {expected22});

  auto input33 = makeRowVector({
      makeNullableFlatVector<double>({10, 50, 60, 70, 80, 90, 100}),
      makeNullableFlatVector<double>({2, 2, 1, 3, 2, 1, 4}),
  });
  auto expected33 = makeRowVector({
      makeMapVector<double, double>({
          {{10.0, 2},
           {53.333333333333336, 3},
           {100.0, 4},
           {70.0, 3},
           {83.33333333333333, 3}},
      }),
  });
  testAggregations(
      {input33}, {}, {"numeric_histogram(5, c0, c1)"}, {expected33});

  auto input1 = makeRowVector({
      makeNullableFlatVector<double>({10}),
      makeNullableFlatVector<double>({2}),
  });

  auto expected1 = makeRowVector({
      makeMapVector<double, double>({
          {{10, 2}},
      }),
  });
  testAggregations({input1}, {}, {"numeric_histogram(5, c0, c1)"}, {expected1});

  auto input44 = makeRowVector({
      makeNullableFlatVector<double>({10, 20, 30, 40, 50, 60, 70, 80, 90}),
      makeNullableFlatVector<double>({2, 3, 1, 4, 2, 1, 3, 2, 1}),
  });

  auto expected44 = makeRowVector({
      makeMapVector<double, double>({
          {{18.333333333333332, 6},
           {40.0, 4},
           {53.333333333333336, 3},
           {70, 3},
           {83.33333333333333, 3}},
      }),
  });
  testAggregations(
      {input44}, {}, {"numeric_histogram(5, c0, c1)"}, {expected44});

  auto secondinput = makeRowVector({
      makeNullableFlatVector<double>(
          {2.9, 3.1, 3.4, 3.5, 3.1, 2.9, 3,   3.8, 3.6, 3.1, 3.6, 2.5, 2.8, 2.3,
           3,   3,   3.2, 3.8, 2.6, 2.9, 3.9, 3.5, 2.2, 2.9, 3,   3.2, 3.4, 3.2,
           3,   4.2, 3,   2.9, 4.4, 2.4, 3.9, 2.8, 3.1, 3.2, 3,   3,   3.7, 2.9,
           3,   3.1, 2.5, 3.3, 2.7, 3,   3.1, 3,   2.8, 3,   3.5, 2.6, 3.2, 2.6,
           2.9, 4,   2.8, 3.2, 2.8, 3.2, 3.4, 2.8, 3.7, 3.8, 2.7, 3.3, 2.5, 2.8,
           2.4, 2.9, 3,   2.7, 3.8, 2.8, 3,   3,   3.5, 3.4, 3.4, 3.4, 3,   3.8,
           2.9, 3,   3.6, 3.1, 3.4, 2.7, 2.5, 2.5, 3.2, 2.7, 2.3, 3.3, 2.2, 3.7,
           3.5, 2.7, 2.8, 3.4, 2.9, 3.4, 3,   2.8, 2.7, 3.1, 3.5, 3.3, 3.2, 3.1,
           3.2, 3.6, 3,   3.2, 3,   2.5, 3.1, 3,   3,   2.7, 2.7, 2.6, 3,   2.3,
           3.3, 2.8, 3.2, 3.4, 2.8, 3,   3.1, 2,   3,   2.5, 2.4, 3.3, 2.3, 3,
           2.8, 2.8, 2.6, 3.8, 3.2, 3.4, 2.5, 4.1, 2.2, 3.4}),
      makeNullableFlatVector<double>(
          {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}),
  });

  auto secondexpected = makeRowVector({
      makeMapVector<double, double>({
          {{2.3904761904761904, 21.0},
           {2.898550724637681, 69.0},
           {3.314285714285714, 42.0},
           {3.8444444444444454, 18.0}},
      }),
  });

  testAggregations(
      {secondinput}, {}, {"numeric_histogram(4, c0, c1)"}, {secondexpected});

  auto inputSameValues = makeRowVector({
      makeNullableFlatVector<double>(
          {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0}),
      makeNullableFlatVector<double>({3, 5, 3, 5, 3, 5, 3, 5, 3, 5}),
  });

  auto expectedSameValues = makeRowVector({
      makeMapVector<double, double>({
          {{1.0, 15}, {2.0, 25}},
      }),
  });
  testAggregations(
      {inputSameValues},
      {},
      {"numeric_histogram(4, c0, c1)"},
      {expectedSameValues});
}

TEST_F(NumericHistogramTest, DoubleTwoArgs) {
  AggregationTestBase::disableTestIncremental();
  auto input = makeRowVector({
      makeNullableFlatVector<double>({10, 20, 30, 80, 90, 100}),
  });

  auto expected = makeRowVector({
      makeMapVector<double, double>({
          {{15, 2}, {30, 1}, {80, 1}, {90, 1}, {100, 1}},
      }),
  });

  auto expected2 = makeRowVector({
      makeMapVector<double, double>({
          {{20, 3}, {90, 3}},
      }),
  });
  auto expected4 = makeRowVector({
      makeMapVector<double, double>({
          {{15, 2}, {30, 1}, {85, 2}, {100, 1}},
      }),
  });

  testAggregations({input}, {}, {"numeric_histogram(5, c0)"}, {expected});
  testAggregations({input}, {}, {"numeric_histogram(2, c0)"}, {expected2});
  testAggregations({input}, {}, {"numeric_histogram(4, c0)"}, {expected4});

  auto input22 = makeRowVector({
      makeNullableFlatVector<double>({10, 20, 30, 40, 50, 60, 70, 80}),
  });

  auto expected22 = makeRowVector({
      makeMapVector<double, double>({
          {{15.0, 2}, {35.0, 2}, {55.0, 2}, {70.0, 1}, {80.0, 1}},
      }),
  });
  testAggregations({input22}, {}, {"numeric_histogram(5, c0)"}, {expected22});

  auto input33 = makeRowVector({
      makeNullableFlatVector<double>({10, 50, 60, 70, 80, 90, 100}),
  });
  auto expected33 = makeRowVector({
      makeMapVector<double, double>({
          {{10.0, 1}, {55.0, 2}, {100.0, 1}, {75.0, 2}, {90.0, 1}},
      }),
  });
  testAggregations({input33}, {}, {"numeric_histogram(5, c0)"}, {expected33});

  auto input1 = makeRowVector({
      makeNullableFlatVector<double>({10}),
  });

  auto expected1 = makeRowVector({
      makeMapVector<double, double>({
          {{10, 1}},
      }),
  });
  testAggregations({input1}, {}, {"numeric_histogram(5, c0)"}, {expected1});

  auto input44 = makeRowVector({
      makeNullableFlatVector<double>({10, 20, 30, 40, 50, 60, 70, 80, 90}),
  });

  auto expected44 = makeRowVector({
      makeMapVector<double, double>({
          {{15, 2}, {35.0, 2}, {55, 2}, {75, 2}, {90.0, 1}},
      }),
  });
  testAggregations({input44}, {}, {"numeric_histogram(5, c0)"}, {expected44});

  auto secondinput = makeRowVector({makeNullableFlatVector<double>(
      {2.9, 3.1, 3.4, 3.5, 3.1, 2.9, 3,   3.8, 3.6, 3.1, 3.6, 2.5, 2.8, 2.3,
       3,   3,   3.2, 3.8, 2.6, 2.9, 3.9, 3.5, 2.2, 2.9, 3,   3.2, 3.4, 3.2,
       3,   4.2, 3,   2.9, 4.4, 2.4, 3.9, 2.8, 3.1, 3.2, 3,   3,   3.7, 2.9,
       3,   3.1, 2.5, 3.3, 2.7, 3,   3.1, 3,   2.8, 3,   3.5, 2.6, 3.2, 2.6,
       2.9, 4,   2.8, 3.2, 2.8, 3.2, 3.4, 2.8, 3.7, 3.8, 2.7, 3.3, 2.5, 2.8,
       2.4, 2.9, 3,   2.7, 3.8, 2.8, 3,   3,   3.5, 3.4, 3.4, 3.4, 3,   3.8,
       2.9, 3,   3.6, 3.1, 3.4, 2.7, 2.5, 2.5, 3.2, 2.7, 2.3, 3.3, 2.2, 3.7,
       3.5, 2.7, 2.8, 3.4, 2.9, 3.4, 3,   2.8, 2.7, 3.1, 3.5, 3.3, 3.2, 3.1,
       3.2, 3.6, 3,   3.2, 3,   2.5, 3.1, 3,   3,   2.7, 2.7, 2.6, 3,   2.3,
       3.3, 2.8, 3.2, 3.4, 2.8, 3,   3.1, 2,   3,   2.5, 2.4, 3.3, 2.3, 3,
       2.8, 2.8, 2.6, 3.8, 3.2, 3.4, 2.5, 4.1, 2.2, 3.4})});

  auto secondexpected = makeRowVector({
      makeMapVector<double, double>({
          {{2.3904761904761904, 21.0},
           {2.898550724637681, 69.0},
           {3.314285714285714, 42.0},
           {3.8444444444444454, 18.0}},
      }),
  });

  testAggregations(
      {secondinput}, {}, {"numeric_histogram(4, c0)"}, {secondexpected});

  auto inputSameValues = makeRowVector({makeNullableFlatVector<double>(
      {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0})});

  auto expectedSameValues = makeRowVector({
      makeMapVector<double, double>({
          {{1.0, 5}, {2.0, 5}},
      }),
  });
  testAggregations(
      {inputSameValues},
      {},
      {"numeric_histogram(4, c0)"},
      {expectedSameValues});

  auto input5 = makeRowVector({
      makeNullableFlatVector<double>(
          {0, 0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
           0, 1, 0, 0, 0, 0, 2, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}),
  });

  auto expected5 = makeRowVector({
      makeMapVector<double, double>({
          {{0, 285}, {1, 31}, {2, 5}},
      }),
  });

  testAggregations({input5}, {}, {"numeric_histogram(5, c0)"}, {expected5});
}

TEST_F(NumericHistogramTest, floatTwoArgs) {
  AggregationTestBase::disableTestIncremental();
  auto input = makeRowVector({
      makeNullableFlatVector<float>({10, 20, 30, 80, 90, 100}),
  });

  auto expected = makeRowVector({
      makeMapVector<float, float>({
          {{15, 2}, {30, 1}, {80, 1}, {90, 1}, {100, 1}},
      }),
  });

  auto expected2 = makeRowVector({
      makeMapVector<float, float>({
          {{20, 3}, {90, 3}},
      }),
  });
  auto expected4 = makeRowVector({
      makeMapVector<float, float>({
          {{15, 2}, {30, 1}, {85, 2}, {100, 1}},
      }),
  });

  testAggregations({input}, {}, {"numeric_histogram(5, c0)"}, {expected});
  testAggregations({input}, {}, {"numeric_histogram(2, c0)"}, {expected2});
  testAggregations({input}, {}, {"numeric_histogram(4, c0)"}, {expected4});

  auto input22 = makeRowVector({
      makeNullableFlatVector<float>({10, 20, 30, 40, 50, 60, 70, 80}),
  });

  auto expected22 = makeRowVector({
      makeMapVector<float, float>({
          {{15.0, 2}, {35.0, 2}, {55.0, 2}, {70.0, 1}, {80.0, 1}},
      }),
  });
  testAggregations({input22}, {}, {"numeric_histogram(5, c0)"}, {expected22});

  auto input33 = makeRowVector({
      makeNullableFlatVector<float>({10, 50, 60, 70, 80, 90, 100}),
  });
  auto expected33 = makeRowVector({
      makeMapVector<float, float>({
          {{10.0, 1}, {55.0, 2}, {100.0, 1}, {75.0, 2}, {90.0, 1}},
      }),
  });
  testAggregations({input33}, {}, {"numeric_histogram(5, c0)"}, {expected33});

  auto input1 = makeRowVector({
      makeNullableFlatVector<float>({10}),
  });

  auto expected1 = makeRowVector({
      makeMapVector<float, float>({
          {{10, 1}},
      }),
  });
  testAggregations({input1}, {}, {"numeric_histogram(5, c0)"}, {expected1});

  auto input44 = makeRowVector({
      makeNullableFlatVector<float>({10, 20, 30, 40, 50, 60, 70, 80, 90}),
  });

  auto expected44 = makeRowVector({
      makeMapVector<float, float>({
          {{15, 2}, {35.0, 2}, {55, 2}, {75, 2}, {90.0, 1}},
      }),
  });
  testAggregations({input44}, {}, {"numeric_histogram(5, c0)"}, {expected44});

  auto secondinput = makeRowVector({makeNullableFlatVector<float>(
      {2.9, 3.1, 3.4, 3.5, 3.1, 2.9, 3,   3.8, 3.6, 3.1, 3.6, 2.5, 2.8, 2.3,
       3,   3,   3.2, 3.8, 2.6, 2.9, 3.9, 3.5, 2.2, 2.9, 3,   3.2, 3.4, 3.2,
       3,   4.2, 3,   2.9, 4.4, 2.4, 3.9, 2.8, 3.1, 3.2, 3,   3,   3.7, 2.9,
       3,   3.1, 2.5, 3.3, 2.7, 3,   3.1, 3,   2.8, 3,   3.5, 2.6, 3.2, 2.6,
       2.9, 4,   2.8, 3.2, 2.8, 3.2, 3.4, 2.8, 3.7, 3.8, 2.7, 3.3, 2.5, 2.8,
       2.4, 2.9, 3,   2.7, 3.8, 2.8, 3,   3,   3.5, 3.4, 3.4, 3.4, 3,   3.8,
       2.9, 3,   3.6, 3.1, 3.4, 2.7, 2.5, 2.5, 3.2, 2.7, 2.3, 3.3, 2.2, 3.7,
       3.5, 2.7, 2.8, 3.4, 2.9, 3.4, 3,   2.8, 2.7, 3.1, 3.5, 3.3, 3.2, 3.1,
       3.2, 3.6, 3,   3.2, 3,   2.5, 3.1, 3,   3,   2.7, 2.7, 2.6, 3,   2.3,
       3.3, 2.8, 3.2, 3.4, 2.8, 3,   3.1, 2,   3,   2.5, 2.4, 3.3, 2.3, 3,
       2.8, 2.8, 2.6, 3.8, 3.2, 3.4, 2.5, 4.1, 2.2, 3.4})});

  auto secondexpected = makeRowVector({
      makeMapVector<float, float>({
          {{2.3904761904761904, 21.0},
           {2.898550724637681, 69.0},
           {3.314285714285714, 42.0},
           {3.8444444444444454, 18.0}},
      }),
  });

  testAggregations(
      {secondinput}, {}, {"numeric_histogram(4, c0)"}, {secondexpected});

  auto inputSameValues = makeRowVector({makeNullableFlatVector<float>(
      {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0})});

  auto expectedSameValues = makeRowVector({
      makeMapVector<float, float>({
          {{1.0, 5}, {2.0, 5}},
      }),
  });
  testAggregations(
      {inputSameValues},
      {},
      {"numeric_histogram(4, c0)"},
      {expectedSameValues});

  auto input5 = makeRowVector({
      makeNullableFlatVector<float>(
          {0, 0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
           0, 1, 0, 0, 0, 0, 2, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}),
  });

  auto expected5 = makeRowVector({
      makeMapVector<float, float>({
          {{0, 285}, {1, 31}, {2, 5}},
      }),
  });

  testAggregations({input5}, {}, {"numeric_histogram(5, c0)"}, {expected5});
}

} // namespace
} // namespace facebook::velox::aggregate::test
