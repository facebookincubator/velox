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

// Measures fixedWidthScan with an IsNotNull filter, the filter a scan pushes
// down for 'x IS NOT NULL' and for join keys.

#include <folly/Benchmark.h>
#include <folly/Random.h>
#include <folly/init/Init.h>

#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "velox/dwio/common/DecoderUtil.h"
#include "velox/dwio/common/SeekableInputStream.h"
#include "velox/dwio/common/SelectiveColumnReader.h"
#include "velox/type/Filter.h"

using namespace facebook::velox;
using namespace facebook::velox::dwio::common;

namespace {

// Rows per scan call, matching the default batch size of the selective readers.
constexpr int32_t kNumRows{10'000};

// Slack past the last output slot. The SIMD filter path stores whole batches.
constexpr int32_t kPadding{64};

enum class RowPattern {
  // Every row in the range, as when nothing filtered rows out earlier.
  kDense,
  // Every other row.
  kEveryOther,
  // About half of the rows, chosen at random.
  kRandomHalf,
};

std::string_view patternName(RowPattern pattern) {
  switch (pattern) {
    case RowPattern::kDense:
      return "dense";
    case RowPattern::kEveryOther:
      return "everyOther";
    case RowPattern::kRandomHalf:
      return "randomHalf";
  }
  VELOX_UNREACHABLE();
}

// Returns kNumRows increasing positions in the value stream.
std::vector<int32_t> makeRows(RowPattern pattern) {
  std::vector<int32_t> rows;
  rows.reserve(kNumRows);
  switch (pattern) {
    case RowPattern::kDense:
      for (int32_t i = 0; i < kNumRows; ++i) {
        rows.push_back(i);
      }
      break;
    case RowPattern::kEveryOther:
      for (int32_t i = 0; i < kNumRows; ++i) {
        rows.push_back(2 * i);
      }
      break;
    case RowPattern::kRandomHalf: {
      folly::Random::DefaultGenerator rng(1);
      for (int32_t row = 0; rows.size() < kNumRows; ++row) {
        if (folly::Random::oneIn(2, rng)) {
          rows.push_back(row);
        }
      }
      break;
    }
  }
  return rows;
}

// Inputs and outputs of one benchmark, built once outside the timed loop.
template <typename T>
struct ScanInput {
  explicit ScanInput(RowPattern pattern) : rows(makeRows(pattern)) {
    data.resize(rows.back() + 1 + kPadding);
    std::iota(data.begin(), data.end(), T{1});
    // With nulls, the reader maps each non-null value to its output row,
    // skipping the null rows in between.
    scatterRows.resize(rows.size());
    for (int32_t i = 0; i < rows.size(); ++i) {
      scatterRows[i] = i + i / 8;
    }
    values.resize(kNumRows + kPadding);
    hits.resize(kNumRows + kPadding);
  }

  std::vector<int32_t> rows;
  std::vector<int32_t> scatterRows;
  std::vector<T> data;
  std::vector<T> values;
  std::vector<int32_t> hits;
};

template <typename T, bool kFilterOnly, bool kScatter>
unsigned scan(ScanInput<T>& input, unsigned iterations) {
  const facebook::velox::common::IsNotNull filter;
  NoHook noHook;
  for (unsigned i = 0; i < iterations; ++i) {
    SeekableArrayInputStream stream(
        reinterpret_cast<const char*>(input.data.data()),
        input.data.size() * sizeof(T));
    const char* bufferStart{nullptr};
    const char* bufferEnd{nullptr};
    int32_t numValues{0};
    fixedWidthScan<T, kFilterOnly, kScatter>(
        folly::Range<const int32_t*>(input.rows.data(), input.rows.size()),
        kScatter ? input.scatterRows.data() : nullptr,
        input.values.data(),
        input.hits.data(),
        numValues,
        stream,
        bufferStart,
        bufferEnd,
        filter,
        noHook);
    VELOX_CHECK_EQ(numValues, kNumRows);
    folly::doNotOptimizeAway(input.hits.data());
    folly::doNotOptimizeAway(input.values.data());
  }
  return iterations;
}

template <typename T, bool kFilterOnly, bool kScatter>
void addScanBenchmarks(std::string_view typeName) {
  const std::string mode = kFilterOnly ? "filterOnly" : "values";
  for (auto pattern :
       {RowPattern::kDense, RowPattern::kEveryOther, RowPattern::kRandomHalf}) {
    auto input = std::make_shared<ScanInput<T>>(pattern);
    folly::addBenchmark(
        __FILE__,
        fmt::format(
            "{}_{}_{}{}",
            typeName,
            mode,
            patternName(pattern),
            kScatter ? "_scatter" : ""),
        [input](unsigned iterations) {
          return scan<T, kFilterOnly, kScatter>(*input, iterations);
        });
  }
}

template <typename T>
void addTypeBenchmarks(std::string_view typeName) {
  addScanBenchmarks<T, false, false>(typeName);
  addScanBenchmarks<T, true, false>(typeName);
  addScanBenchmarks<T, false, true>(typeName);
  addScanBenchmarks<T, true, true>(typeName);
  folly::addBenchmark(__FILE__, "-", []() -> unsigned { return 0; });
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  addTypeBenchmarks<int16_t>("smallint");
  addTypeBenchmarks<int32_t>("integer");
  addTypeBenchmarks<int64_t>("bigint");
  addTypeBenchmarks<float>("real");
  addTypeBenchmarks<double>("double");
  folly::runBenchmarks();
  return 0;
}
