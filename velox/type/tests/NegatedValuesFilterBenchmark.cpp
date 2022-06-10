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

#include <limits>
#include "folly/Benchmark.h"
#include "folly/Portability.h"
#include "folly/Random.h"
#include "folly/Varint.h"
#include "folly/init/Init.h"
#include "folly/lang/Bits.h"
#include "velox/dwio/common/exception/Exception.h"

#include "velox/type/Filter.h"

using namespace facebook::velox;
using namespace facebook::velox::common;

std::vector<int64_t> verySparseValues;
std::vector<int64_t> sparseValues;
std::vector<int64_t> mediumValues;
std::vector<int64_t> denseValues;
std::vector<int64_t> veryDenseValues;
std::vector<int64_t> extremelyDenseValues; // used for testing != only
std::vector<int64_t> randomValues;
std::vector<std::unique_ptr<Filter>> newFilters;
std::vector<std::unique_ptr<Filter>> oldFilters;

int32_t filterOld(int filterNum, const std::vector<int64_t>& data) {
  int32_t count = 0;
  for (auto i = 0; i < data.size(); ++i) {
    if (oldFilters[filterNum]->testInt64(data[i]))
      ++count;
  }
  return count;
}

int32_t filterNew(int filterNum, const std::vector<int64_t>& data) {
  int32_t count = 0;
  for (auto i = 0; i < data.size(); ++i) {
    if (newFilters[filterNum]->testInt64(data[i]))
      ++count;
  }
  return count;
}

BENCHMARK(verySparseOldNotEqual) {
  // expect to remove about 1 out of every 1,000,000 values
  folly::doNotOptimizeAway(filterOld(0, verySparseValues));
}

BENCHMARK_RELATIVE(verySparseNewNotEqual) {
  folly::doNotOptimizeAway(filterNew(0, verySparseValues));
}

BENCHMARK(sparseOldNotEqual) {
  // expect to remove about 1 out of every 100,000 values
  folly::doNotOptimizeAway(filterOld(0, sparseValues));
}

BENCHMARK_RELATIVE(sparseNewNotEqual) {
  folly::doNotOptimizeAway(filterNew(0, sparseValues));
}

BENCHMARK(mediumOldNotEqual) {
  // expect to remove about 1 out of every 15,000 values (< 0.1%)
  folly::doNotOptimizeAway(filterOld(0, mediumValues));
}

BENCHMARK_RELATIVE(mediumNewNotEqual) {
  folly::doNotOptimizeAway(filterNew(0, mediumValues));
}

BENCHMARK(denseOldNotEqual) {
  // expect to remove about 1 out of every 3,000 values
  folly::doNotOptimizeAway(filterOld(0, denseValues));
}

BENCHMARK_RELATIVE(denseNewNotEqual) {
  folly::doNotOptimizeAway(filterNew(0, denseValues));
}

BENCHMARK(veryDenseOldNotEqual) {
  // expect to remove about 1 out of every 1,050 values
  folly::doNotOptimizeAway(filterOld(0, veryDenseValues));
}

BENCHMARK_RELATIVE(veryDenseNewNotEqual) {
  folly::doNotOptimizeAway(filterNew(0, veryDenseValues));
}

BENCHMARK(extremelyDenseOldNotEqual) {
  // expect to remove about 1 out of every 20 values (5%)
  folly::doNotOptimizeAway(filterOld(0, extremelyDenseValues));
}

BENCHMARK_RELATIVE(extremelyDenseNewNotEqual) {
  folly::doNotOptimizeAway(filterNew(0, extremelyDenseValues));
}

BENCHMARK(verySparseOld10) {
  // expect to remove about 1 out of every 100,000 values
  folly::doNotOptimizeAway(filterOld(1, verySparseValues));
}

BENCHMARK_RELATIVE(verySparseNew10) {
  folly::doNotOptimizeAway(filterNew(1, verySparseValues));
}

BENCHMARK(sparseOld10) {
  // expect to remove about 1 out of every 10,000 values
  folly::doNotOptimizeAway(filterOld(1, sparseValues));
}

BENCHMARK_RELATIVE(sparseNew10) {
  folly::doNotOptimizeAway(filterNew(1, sparseValues));
}

BENCHMARK(mediumOld10) {
  // expect to remove about 1 out of every 1,500 values
  folly::doNotOptimizeAway(filterOld(1, mediumValues));
}

BENCHMARK_RELATIVE(mediumNew10) {
  folly::doNotOptimizeAway(filterNew(1, mediumValues));
}

BENCHMARK(denseOld10) {
  // expect to remove about 1 out of every 300 values
  folly::doNotOptimizeAway(filterOld(1, denseValues));
}

BENCHMARK_RELATIVE(denseNew10) {
  folly::doNotOptimizeAway(filterNew(1, denseValues));
}

BENCHMARK(veryDenseOld10) {
  // expect to remove about 1 out of every 105 values or about 1%
  folly::doNotOptimizeAway(filterOld(1, veryDenseValues));
}

BENCHMARK_RELATIVE(veryDenseNew10) {
  folly::doNotOptimizeAway(filterNew(1, veryDenseValues));
}

BENCHMARK(extremelyDenseOld10) {
  // expect to remove about half the values (50%)
  folly::doNotOptimizeAway(filterOld(1, extremelyDenseValues));
}

BENCHMARK_RELATIVE(extremelyDenseNew10) {
  folly::doNotOptimizeAway(filterNew(1, extremelyDenseValues));
}

BENCHMARK(verySparseOld100) {
  // expect to remove about 1 out of every 10,000 values
  folly::doNotOptimizeAway(filterOld(2, verySparseValues));
}

BENCHMARK_RELATIVE(verySparseNew100) {
  folly::doNotOptimizeAway(filterNew(2, verySparseValues));
}

BENCHMARK(sparseOld100) {
  // expect to remove about 1 out of every 1,000 values
  folly::doNotOptimizeAway(filterOld(2, sparseValues));
}

BENCHMARK_RELATIVE(sparseNew100) {
  folly::doNotOptimizeAway(filterNew(2, sparseValues));
}

BENCHMARK(mediumOld100) {
  // expect to remove about 1 out of every 150 values
  folly::doNotOptimizeAway(filterOld(2, mediumValues));
}

BENCHMARK_RELATIVE(mediumNew100) {
  folly::doNotOptimizeAway(filterNew(2, mediumValues));
}

BENCHMARK(denseOld100) {
  // expect to remove about 1 out of every 30 values (3%)
  folly::doNotOptimizeAway(filterOld(2, denseValues));
}

BENCHMARK_RELATIVE(denseNew100) {
  folly::doNotOptimizeAway(filterNew(2, denseValues));
}

BENCHMARK(veryDenseOld100) {
  // expect to remove about 1 out of every 10.5 values (9%)
  folly::doNotOptimizeAway(filterOld(2, veryDenseValues));
}

BENCHMARK_RELATIVE(veryDenseNew100) {
  folly::doNotOptimizeAway(filterNew(2, veryDenseValues));
}

BENCHMARK(verySparseOld1000) {
  // expect to remove about 1 out of every 1,000 values
  folly::doNotOptimizeAway(filterOld(3, verySparseValues));
}

BENCHMARK_RELATIVE(verySparseNew1000) {
  folly::doNotOptimizeAway(filterNew(3, verySparseValues));
}

BENCHMARK(sparseOld1000) {
  // expect to remove about 1 out of every 100 values (1%)
  folly::doNotOptimizeAway(filterOld(3, sparseValues));
}

BENCHMARK_RELATIVE(sparseNew1000) {
  folly::doNotOptimizeAway(filterNew(3, sparseValues));
}

BENCHMARK(mediumOld1000) {
  // expect to remove about 1 out of every 15 values (7%)
  folly::doNotOptimizeAway(filterOld(3, mediumValues));
}

BENCHMARK_RELATIVE(mediumNew1000) {
  folly::doNotOptimizeAway(filterNew(3, mediumValues));
}

BENCHMARK(denseOld1000) {
  // expect to remove about 1 out of every 3 values (33%)
  folly::doNotOptimizeAway(filterOld(3, denseValues));
}

BENCHMARK_RELATIVE(denseNew1000) {
  folly::doNotOptimizeAway(filterNew(3, denseValues));
}

BENCHMARK(veryDenseOld1000) {
  // expect to remove about 1 out of every 1.05 values (95%)
  folly::doNotOptimizeAway(filterOld(3, veryDenseValues));
}

BENCHMARK_RELATIVE(veryDenseNew1000) {
  folly::doNotOptimizeAway(filterNew(3, veryDenseValues));
}

BENCHMARK(verySparseOld10000) {
  // expect to remove about 1 out of every 100 values
  folly::doNotOptimizeAway(filterOld(4, verySparseValues));
}

BENCHMARK_RELATIVE(verySparseNew10000) {
  folly::doNotOptimizeAway(filterNew(4, verySparseValues));
}

BENCHMARK(sparseOld10000) {
  // expect to remove about 1 out of every 10 values (10%)
  folly::doNotOptimizeAway(filterOld(4, sparseValues));
}

BENCHMARK_RELATIVE(sparseNew10000) {
  folly::doNotOptimizeAway(filterNew(4, sparseValues));
}

BENCHMARK(mediumOld10000) {
  // expect to remove about two-thirds of the values (66%)
  folly::doNotOptimizeAway(filterOld(4, mediumValues));
}

BENCHMARK_RELATIVE(mediumNew10000) {
  folly::doNotOptimizeAway(filterNew(4, mediumValues));
}

BENCHMARK(denseOld10000) {
  // removes all the values
  folly::doNotOptimizeAway(filterOld(4, denseValues));
}

BENCHMARK_RELATIVE(denseNew10000) {
  folly::doNotOptimizeAway(filterNew(4, denseValues));
}

// various tests to see if other factors impact the tests

BENCHMARK(greaterSpreadOld) {
  // removes about 10% of the values, but they're spread apart more
  // checking to see if a greater spread in the ranges affects anything
  folly::doNotOptimizeAway(filterOld(5, sparseValues));
}

BENCHMARK_RELATIVE(greaterSpreadNew) {
  folly::doNotOptimizeAway(filterNew(5, sparseValues));
}

BENCHMARK(bitmaskOld) {
  // removes about 10% of the values
  folly::doNotOptimizeAway(filterOld(6, veryDenseValues));
}

BENCHMARK_RELATIVE(bitmaskNew) {
  folly::doNotOptimizeAway(filterNew(6, veryDenseValues));
}

BENCHMARK(randomOld3Val) {
  // removes about 3 out of a million values
  folly::doNotOptimizeAway(filterOld(7, randomValues));
}

BENCHMARK_RELATIVE(randomNew3Val) {
  folly::doNotOptimizeAway(filterNew(7, randomValues));
}

BENCHMARK(randomOld10Val) {
  // removes about 10 out of a million values
  folly::doNotOptimizeAway(filterOld(8, randomValues));
}

BENCHMARK_RELATIVE(randomNew10Val) {
  folly::doNotOptimizeAway(filterNew(8, randomValues));
}

BENCHMARK(randomOld100Val) {
  // removes about 100 out of a million values
  folly::doNotOptimizeAway(filterOld(9, randomValues));
}

BENCHMARK_RELATIVE(randomNew100Val) {
  folly::doNotOptimizeAway(filterNew(9, randomValues));
}

BENCHMARK(randomOld1000Val) {
  // removes about 1000 out of a million values
  folly::doNotOptimizeAway(filterOld(10, randomValues));
}

BENCHMARK_RELATIVE(randomNew1000Val) {
  folly::doNotOptimizeAway(filterNew(10, randomValues));
}

BENCHMARK(randomOld10000Val) {
  // removes about 1% of the million values (10,000)
  folly::doNotOptimizeAway(filterOld(11, randomValues));
}

BENCHMARK_RELATIVE(randomNew10000Val) {
  folly::doNotOptimizeAway(filterNew(11, randomValues));
}

int32_t main(int32_t argc, char* argv[]) {
  constexpr int32_t kNumValues = 1000000;
  // first one represents a != filter, the rest are "NOT IN" filters
  std::vector<int32_t> kFilterSizes = {1, 5, 100, 1000, 10000, 10, 100};
  std::vector<int32_t> kFilterIntervals = {
      1000, 1000, 1000, 1000, 1000, 10000, 10};
  folly::init(&argc, &argv);

  for (auto j = 0; j < kFilterSizes.size(); ++j) {
    std::vector<int64_t> filterValues;
    std::vector<std::unique_ptr<common::BigintRange>> subfilters;
    filterValues.reserve(kFilterSizes[j]);
    subfilters.reserve(kFilterSizes[j]);
    int64_t start = std::numeric_limits<int64_t>::min();
    for (auto i = 0; i < kFilterSizes[j]; ++i) {
      filterValues.emplace_back(i * kFilterIntervals[j]);
      subfilters.emplace_back(std::make_unique<common::BigintRange>(
          start, i * kFilterIntervals[j] - 1, false));
      start = i * kFilterIntervals[j] + 1;
    }
    subfilters.emplace_back(std::make_unique<common::BigintRange>(
        start, std::numeric_limits<int64_t>::max(), false));
    newFilters.emplace_back(createNegatedBigintValues(filterValues, false));
    oldFilters.emplace_back(std::make_unique<common::BigintMultiRange>(
        std::move(subfilters), false));
  }

  std::vector<int32_t> kRandSizes = {3, 10, 100, 1000, 10000};
  // generate some random filters of numbers between 1 and 1,000,000
  for (auto size : kRandSizes) {
    std::unordered_set<int64_t> rejects;
    while (rejects.size() < size) {
      rejects.insert(folly::Random::rand64() % 1000000);
    }
    std::vector<int64_t> rejectedValues;
    rejectedValues.reserve(rejects.size());
    for (auto it = rejects.begin(); it != rejects.end(); ++it) {
      rejectedValues.emplace_back(*it);
    }
    sort(rejectedValues.begin(), rejectedValues.end());
    newFilters.emplace_back(createNegatedBigintValues(rejectedValues, false));
    std::vector<std::unique_ptr<common::BigintRange>> subfilters;
    int64_t start = std::numeric_limits<int64_t>::min();

    for (auto i = 0; i < size; ++i) {
      if (rejectedValues[i] != start) {
        subfilters.emplace_back(std::make_unique<common::BigintRange>(
            start, rejectedValues[i] - 1, false));
      }
      start = rejectedValues[i] + 1;
    }
    subfilters.emplace_back(std::make_unique<common::BigintRange>(
        start, std::numeric_limits<int64_t>::max(), false));
    oldFilters.emplace_back(std::make_unique<common::BigintMultiRange>(
        std::move(subfilters), false));
  }

  veryDenseValues.resize(kNumValues);
  denseValues.resize(kNumValues);
  mediumValues.resize(kNumValues);
  sparseValues.resize(kNumValues);
  verySparseValues.resize(kNumValues);
  extremelyDenseValues.resize(kNumValues);
  randomValues.resize(kNumValues);

  for (auto i = 0; i < kNumValues; ++i) {
    extremelyDenseValues[i] = (folly::Random::rand32() % 20) * 1000;
    veryDenseValues[i] = (folly::Random::rand32() % 1050) * 1000;
    denseValues[i] = (folly::Random::rand32() % 3000) * 1000;
    mediumValues[i] = (folly::Random::rand32() % 15000) * 1000;
    sparseValues[i] = (folly::Random::rand32() % 100000) * 1000;
    verySparseValues[i] = (folly::Random::rand32() % 1000000) * 1000;
    randomValues[i] = (folly::Random::rand32() % 1000000);
  }

  // comment out this section to speed up testing + skip verifying correctness

  VELOX_CHECK_EQ(
      filterOld(0, extremelyDenseValues), filterNew(0, extremelyDenseValues));
  for (int i = 0; i < newFilters.size(); ++i) {
    VELOX_CHECK_EQ(
        filterOld(i, verySparseValues), filterNew(i, verySparseValues));
    VELOX_CHECK_EQ(filterOld(i, sparseValues), filterNew(i, sparseValues));
    VELOX_CHECK_EQ(filterOld(i, mediumValues), filterNew(i, mediumValues));
    VELOX_CHECK_EQ(filterOld(i, denseValues), filterNew(i, denseValues));
    VELOX_CHECK_EQ(
        filterOld(i, veryDenseValues), filterNew(i, veryDenseValues));
    VELOX_CHECK_EQ(filterOld(i, randomValues), filterNew(i, randomValues));
  }

  folly::runBenchmarks();
  return 0;
}
