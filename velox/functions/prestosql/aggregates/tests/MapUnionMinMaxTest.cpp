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
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::aggregate::test {

namespace {

class MapUnionMinMaxTest : public AggregationTestBase {};

TEST_F(MapUnionMinMaxTest, global) {
  SCOPED_TRACE("global");
  auto data = makeRowVector({
      makeNullableMapVector<int64_t, int64_t>({
          {{}}, // empty map
          std::nullopt, // null map
          {{{1, 10}, {2, 20}}},
          {{{1, 11}, {3, 30}, {4, 40}}},
          {{{3, 30}, {5, 50}, {1, 12}}},
      }),
  });

  {
    // map_union_min
    auto expected = makeRowVector({
        makeMapVector<int64_t, int64_t>({
            {{1, 10}, {2, 20}, {3, 30}, {4, 40}, {5, 50}},
        }),
    });

    testAggregations({data}, {}, {"map_union_min(c0)"}, {expected});
  }
  {
    // map_union_max
    auto expected = makeRowVector({
        makeMapVector<int64_t, int64_t>({
            {{1, 12}, {2, 20}, {3, 30}, {4, 40}, {5, 50}},
        }),
    });

    testAggregations({data}, {}, {"map_union_max(c0)"}, {expected});
  }
}

TEST_F(MapUnionMinMaxTest, globalVarcharKey) {
  SCOPED_TRACE("globalVarcharKey");
  std::vector<std::string> keyStrings = {
      "Tall mountains",
      "Wide rivers",
      "Deep oceans",
      "Thick dark forests",
      "Expansive vistas",
  };
  std::vector<StringView> keys;
  for (const auto& key : keyStrings) {
    keys.push_back(StringView(key));
  }

  auto data = makeRowVector({
      makeNullableMapVector<StringView, int64_t>({
          {{}}, // empty map
          std::nullopt, // null map
          {{{keys[0], 10}, {keys[1], 20}}},
          {{{keys[0], 11}, {keys[2], 30}, {keys[3], 40}}},
          {{{keys[2], 30}, {keys[4], 50}, {keys[0], 12}}},
      }),
  });

  {
    // map_union_min
    auto expected = makeRowVector({
        makeMapVector<StringView, int64_t>({
            {{keys[0], 10},
             {keys[1], 20},
             {keys[2], 30},
             {keys[3], 40},
             {keys[4], 50}},
        }),
    });

    testAggregations({data}, {}, {"map_union_min(c0)"}, {expected});
  }
  {
    // map_union_max
    auto expected = makeRowVector({
        makeMapVector<StringView, int64_t>({
            {{keys[0], 12},
             {keys[1], 20},
             {keys[2], 30},
             {keys[3], 40},
             {keys[4], 50}},
        }),
    });

    testAggregations({data}, {}, {"map_union_max(c0)"}, {expected});
  }
}

TEST_F(MapUnionMinMaxTest, nullAndEmptyMaps) {
  SCOPED_TRACE("nullAndEmptyMaps");
  auto allEmptyMaps = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {},
          {},
          {},
      }),
  });

  auto expectedEmpty = makeRowVector({
      makeMapVector<int64_t, int64_t>({
          {},
      }),
  });

  testAggregations({allEmptyMaps}, {}, {"map_union_min(c0)"}, {expectedEmpty});
  testAggregations({allEmptyMaps}, {}, {"map_union_max(c0)"}, {expectedEmpty});

  auto allNullMaps = makeRowVector({
      makeNullableMapVector<int64_t, int64_t>({
          std::nullopt,
          std::nullopt,
          std::nullopt,
      }),
  });

  auto expectedNull = makeRowVector({
      makeNullableMapVector<int64_t, int64_t>({
          std::nullopt,
      }),
  });

  testAggregations({allNullMaps}, {}, {"map_union_min(c0)"}, {expectedNull});
  testAggregations({allNullMaps}, {}, {"map_union_max(c0)"}, {expectedNull});

  auto emptyAndNullMaps = makeRowVector({
      makeNullableMapVector<int64_t, int64_t>({
          std::nullopt,
          {{}},
          std::nullopt,
          {{}},
      }),
  });

  testAggregations(
      {emptyAndNullMaps}, {}, {"map_union_min(c0)"}, {expectedEmpty});
  testAggregations(
      {emptyAndNullMaps}, {}, {"map_union_max(c0)"}, {expectedEmpty});
}

TEST_F(MapUnionMinMaxTest, integerMaxMinValue) {
  SCOPED_TRACE("integerMaxMinValue");
  const int32_t maxValue = std::numeric_limits<int32_t>::max();
  const int32_t minValue = std::numeric_limits<int32_t>::lowest();
  auto data = makeRowVector({
      makeNullableMapVector<int64_t, int32_t>({
          {{{1, 10}, {2, 20}}},
          {{{1, maxValue}, {3, minValue}, {4, 40}}},
          {{{3, 30}, {5, 50}, {1, 30}}},
      }),
  });

  {
    // map_union_min
    auto expected = makeRowVector({
        makeMapVector<int64_t, int32_t>({
            {{1, 10}, {2, 20}, {3, minValue}, {4, 40}, {5, 50}},
        }),
    });

    testAggregations({data}, {}, {"map_union_min(c0)"}, {expected});
  }
  {
    // map_union_max
    auto expected = makeRowVector({
        makeMapVector<int64_t, int32_t>({
            {{1, maxValue}, {2, 20}, {3, 30}, {4, 40}, {5, 50}},
        }),
    });

    testAggregations({data}, {}, {"map_union_max(c0)"}, {expected});
  }
}

TEST_F(MapUnionMinMaxTest, floatNan) {
  SCOPED_TRACE("floatNan");
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNan = std::numeric_limits<float>::quiet_NaN();

  auto data = makeRowVector({
      makeNullableMapVector<int64_t, float>({
          {{{1, 10}, {2, 20}}},
          {{{1, kNan}, {3, 30}, {5, 50}}},
          {{{3, 30}, {5, kInf}, {1, 30}}},
      }),
  });

  {
    // map_union_min
    auto expected = makeRowVector({
        makeMapVector<int64_t, float>({
            {{1, 10}, {2, 20}, {3, 30}, {5, 50}},
        }),
    });

    testAggregations({data}, {}, {"map_union_min(c0)"}, {expected});
  }
  {
    // map_union_max
    auto expected = makeRowVector({
        makeMapVector<int64_t, float>({
            {{1, 30}, {2, 20}, {3, 30}, {5, kInf}},
        }),
    });

    testAggregations({data}, {}, {"map_union_max(c0)"}, {expected});
  }
}

TEST_F(MapUnionMinMaxTest, doubleNan) {
  SCOPED_TRACE("doubleNan");
  constexpr double kInf = std::numeric_limits<double>::infinity();
  constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

  auto data = makeRowVector({
      makeNullableMapVector<int64_t, double>({
          {{{1, 10}, {2, 20}}},
          {{{1, kNan}, {3, 30}, {5, 50}}},
          {{{3, 30}, {5, kInf}, {1, 30}}},
      }),
  });

  {
    auto expected = makeRowVector({
        makeMapVector<int64_t, double>({
            {{1, 10}, {2, 20}, {3, 30}, {5, 50}},
        }),
    });

    testAggregations({data}, {}, {"map_union_min(c0)"}, {expected});
  }
  {
    auto expected = makeRowVector({
        makeMapVector<int64_t, double>({
            {{1, 30}, {2, 20}, {3, 30}, {5, kInf}},
        }),
    });

    testAggregations({data}, {}, {"map_union_max(c0)"}, {expected});
  }
}

TEST_F(MapUnionMinMaxTest, groupBy) {
  SCOPED_TRACE("groupBy");
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 1, 2, 1}),
      makeNullableMapVector<int64_t, int64_t>({
          {}, // empty map
          std::nullopt, // null map
          {{{1, 10}, {2, 20}}},
          {{{1, 11}, {3, 30}, {4, 40}}},
          {{{3, 30}, {5, 50}, {1, 12}}},
      }),
  });

  {
    auto expected = makeRowVector({
        makeFlatVector<int64_t>({1, 2}),
        makeMapVector<int64_t, int64_t>({
            {{1, 10}, {2, 20}, {3, 30}, {5, 50}},
            {{1, 11}, {3, 30}, {4, 40}},
        }),
    });

    testAggregations({data}, {"c0"}, {"map_union_min(c1)"}, {expected});
  }
  {
    auto expected = makeRowVector({
        makeFlatVector<int64_t>({1, 2}),
        makeMapVector<int64_t, int64_t>({
            {{1, 12}, {2, 20}, {3, 30}, {5, 50}},
            {{1, 11}, {3, 30}, {4, 40}},
        }),
    });

    testAggregations({data}, {"c0"}, {"map_union_max(c1)"}, {expected});
  }
}

TEST_F(MapUnionMinMaxTest, groupByVarcharKey) {
  SCOPED_TRACE("groupByVarcharKey");
  std::vector<std::string> keyStrings = {
      "Tall mountains",
      "Wide rivers",
      "Deep oceans",
      "Thick dark forests",
      "Expansive vistas",
  };
  std::vector<StringView> keys;
  for (const auto& key : keyStrings) {
    keys.push_back(StringView(key));
  }

  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 1, 2, 1}),
      makeNullableMapVector<StringView, int64_t>({
          {}, // empty map
          std::nullopt, // null map
          {{{keys[0], 10}, {keys[1], 20}}},
          {{{keys[0], 11}, {keys[2], 30}, {keys[3], 40}}},
          {{{keys[2], 30}, {keys[4], 50}, {keys[0], 12}}},
      }),
  });

  {
    auto expected = makeRowVector({
        makeFlatVector<int64_t>({1, 2}),
        makeMapVector<StringView, int64_t>({
            {{keys[0], 10}, {keys[1], 20}, {keys[2], 30}, {keys[4], 50}},
            {{keys[0], 11}, {keys[2], 30}, {keys[3], 40}},
        }),
    });

    testAggregations({data}, {"c0"}, {"map_union_min(c1)"}, {expected});
  }
  {
    auto expected = makeRowVector({
        makeFlatVector<int64_t>({1, 2}),
        makeMapVector<StringView, int64_t>({
            {{keys[0], 12}, {keys[1], 20}, {keys[2], 30}, {keys[4], 50}},
            {{keys[0], 11}, {keys[2], 30}, {keys[3], 40}},
        }),
    });

    testAggregations({data}, {"c0"}, {"map_union_max(c1)"}, {expected});
  }
}

TEST_F(MapUnionMinMaxTest, floatingPointKeys) {
  SCOPED_TRACE("floatingPointKeys");
  auto data = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 1, 2, 1, 1, 2, 2}),
      makeMapVectorFromJson<float, int64_t>({
          "{1.1: 10, 1.2: 20, 1.3: 30}",
          "{2.1: 10, 1.2: 20, 2.3: 30}",
          "{3.1: 10, 1.2: 20, 2.3: 30}",
          "{}",
          "null",
          "{4.1: 10, 4.2: 20, 2.3: 30}",
          "{5.1: 10, 4.2: 20, 2.3: 30}",
          "{6.1: 10, 6.2: 20, 6.3: 30}",
      }),
  });
  auto expected = makeRowVector({
      makeMapVectorFromJson<float, int64_t>({
          "{1.1: 10, 1.2: 20, 1.3: 30, 2.1: 10, 2.3: 30, 3.1: 10, 4.1: 10, "
          "4.2: 20, 5.1: 10, 6.1: 10, 6.2: 20, 6.3: 30}",
      }),
  });

  testAggregations({data}, {}, {"map_union_min(c1)"}, {expected});
  testAggregations({data}, {}, {"map_union_max(c1)"}, {expected});
}

} // namespace
} // namespace facebook::velox::aggregate::test
