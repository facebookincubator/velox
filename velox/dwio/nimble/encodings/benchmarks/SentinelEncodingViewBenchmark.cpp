/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "folly/Benchmark.h"
#include "folly/init/Init.h"
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/SimdUtil.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

using namespace facebook::nimble;
using namespace facebook::nimble::benchmarks;
namespace bits = facebook::velox::bits;

namespace {

// Keeps compiler optimizations enabled for the scalar reference loop.
template <typename T>
uint32_t normalizeScalar(
    T* output,
    uint32_t numRows,
    T sentinel,
    const std::function<void(uint32_t)>* setNull) {
  uint32_t numNulls{0};
  for (uint32_t row{0}; row < numRows; ++row) {
    if (output[row] == sentinel) {
      ++numNulls;
      if (setNull != nullptr) {
        (*setNull)(row);
      }
      output[row] = T{};
    }
  }
  return numRows - numNulls;
}

// Retains the SIMD alternative for comparison with the scalar reference loop.
template <typename T>
uint32_t normalizeSimd(
    T* output,
    uint32_t numRows,
    T sentinel,
    const std::function<void(uint32_t)>* setNull) {
  uint32_t numNulls{0};
  uint32_t row{0};
  using Batch = xsimd::batch<T>;
  static_assert(Batch::size <= 64);
  constexpr auto kLaneMask =
      std::numeric_limits<uint64_t>::max() >> (64 - Batch::size);
  const Batch sentinelValues{sentinel};
  const Batch zero{T{}};
  const auto simdRows = numRows / Batch::size * Batch::size;
  for (; row < simdRows; row += Batch::size) {
    const auto values = Batch::load_unaligned(output + row);
    const auto nulls = values == sentinelValues;
    auto nullMask =
        static_cast<uint64_t>(facebook::velox::simd::toBitMask(nulls)) &
        kLaneMask;
    numNulls += static_cast<uint32_t>(std::popcount(nullMask));
    if (setNull != nullptr) {
      while (nullMask != 0) {
        const auto nullRow = row + std::countr_zero(nullMask);
        (*setNull)(static_cast<uint32_t>(nullRow));
        nullMask &= nullMask - 1;
      }
    }
    xsimd::select(nulls, zero, values).store_unaligned(output + row);
  }
  for (; row < numRows; ++row) {
    if (output[row] == sentinel) {
      ++numNulls;
      if (setNull != nullptr) {
        (*setNull)(row);
      }
      output[row] = T{};
    }
  }
  return numRows - numNulls;
}

// Compares scalar and SIMD loops after the same child read. Both loops are
// checked against the production view before timing.
template <typename T>
class SentinelBenchmark {
 public:
  SentinelBenchmark(uint32_t numRows, uint32_t nullPct)
      : numRows_{numRows},
        output_(numRows),
        nulls_(bits::nwords(numRows)),
        setNull_{[this](uint32_t row) { bits::setBit(nulls_.data(), row); }} {
    Vector<T> values{benchmarkPool().get(), numRows};
    std::vector<T> expectedValues(numRows);
    std::vector<uint64_t> expectedNulls(bits::nwords(numRows));
    std::mt19937 generator{1};
    std::uniform_int_distribution<uint32_t> distribution{0, 99};
    uint32_t numNulls{0};
    for (uint32_t row{0}; row < numRows; ++row) {
      const bool isNull = distribution(generator) < nullPct;
      values[row] = isNull ? kSentinel : static_cast<T>(row + 1);
      expectedValues[row] = isNull ? T{} : values[row];
      if (isNull) {
        bits::setBit(expectedNulls.data(), row);
        ++numNulls;
      }
    }

    const std::span<const T> physicalValues{values.data(), values.size()};
    EncodingSelection<T> selection{
        EncodingSelectionResult{.encodingType = EncodingType::Trivial},
        Statistics<T>::create(physicalValues),
        makeDefaultPolicy(TypeTraits<T>::dataType),
    };
    Buffer buffer{*benchmarkPool()};
    const auto child =
        TrivialEncoding<T>::encode(selection, physicalValues, buffer);
    const auto childOffset =
        EncodingPrefix::serializedSize(numRows, /*useVarintRowCount=*/false) +
        2 * sizeof(uint32_t);
    encoded_.resize(childOffset + child.size() + sizeof(T));
    char* position = encoded_.data();
    EncodingPrefix::serialize(
        EncodingType::Sentinel,
        TypeTraits<T>::dataType,
        numRows,
        /*useVarintRowCount=*/false,
        position);
    encoding::writeUint32(numNulls, position);
    encoding::writeUint32(static_cast<uint32_t>(child.size()), position);
    encoding::writeBytes(child, position);
    encoding::write<T>(kSentinel, position);
    NIMBLE_CHECK_EQ(position, encoded_.data() + encoded_.size());
    auto productionView =
        createEncodingView(encoded_, benchmarkPool().get(), {});
    childView_ = createEncodingView(
        std::string_view{encoded_}.substr(childOffset, child.size()),
        benchmarkPool().get(),
        {});
    NIMBLE_CHECK_EQ(productionView->encodingType(), EncodingType::Sentinel);
    const std::array<RowRange, 1> ranges{RowRange{0, numRows_}};
    const auto productionNonNullRows =
        productionView->read(ranges, setNull_, output_.data());
    NIMBLE_CHECK_EQ(productionNonNullRows, numRows - numNulls);
    NIMBLE_CHECK(output_ == expectedValues);
    NIMBLE_CHECK(nulls_ == expectedNulls);
    productionView->read(0, numRows_, output_.data());
    NIMBLE_CHECK(output_ == expectedValues);

    for (const bool useSimd : {false, true}) {
      const auto nonNullRows =
          useSimd ? read<true, true>() : read<false, true>();
      NIMBLE_CHECK_EQ(nonNullRows, numRows - numNulls);
      NIMBLE_CHECK(output_ == expectedValues);
      NIMBLE_CHECK(nulls_ == expectedNulls);
      if (useSimd) {
        read<true, false>();
      } else {
        read<false, false>();
      }
      NIMBLE_CHECK(output_ == expectedValues);
    }
  }

  template <bool useSimd, bool reportNulls>
  uint32_t run(uint32_t iterations) {
    for (uint32_t iteration{0}; iteration < iterations; ++iteration) {
      const auto nonNullRows = read<useSimd, reportNulls>();
      if constexpr (reportNulls) {
        folly::doNotOptimizeAway(nonNullRows);
        folly::doNotOptimizeAway(nulls_.data());
      }
      folly::doNotOptimizeAway(output_.data());
    }
    return iterations;
  }

 private:
  template <bool useSimd, bool reportNulls>
  uint32_t read() {
    if constexpr (reportNulls) {
      std::fill(nulls_.begin(), nulls_.end(), 0);
    }
    childView_->read(0, numRows_, output_.data());
    uint32_t nonNullRows;
    if constexpr (useSimd) {
      nonNullRows = normalizeSimd(
          output_.data(),
          numRows_,
          kSentinel,
          reportNulls ? &setNull_ : nullptr);
    } else {
      nonNullRows = normalizeScalar(
          output_.data(),
          numRows_,
          kSentinel,
          reportNulls ? &setNull_ : nullptr);
    }
    if constexpr (reportNulls) {
      return nonNullRows;
    } else {
      return 0;
    }
  }

  static constexpr T kSentinel{std::numeric_limits<T>::max()};
  // Number of rows materialized by each timed iteration.
  const uint32_t numRows_;
  // Owns the identical child payload used by both read paths.
  std::string encoded_;
  // Runs the identical child read before either normalization loop.
  std::unique_ptr<EncodingView> childView_;
  // Reuses decoded values and null bits across timed iterations.
  std::vector<T> output_;
  std::vector<uint64_t> nulls_;
  // Marks dense output positions using the production callback convention.
  std::function<void(uint32_t)> setNull_;
};

template <typename T, bool reportNulls>
void addBenchmarks(std::string_view typeName) {
  for (const auto numRows : {32U, 1'024U, 4'096U}) {
    for (const auto nullPct : {0U, 1U, 10U, 50U, 100U}) {
      auto fixture = std::make_shared<SentinelBenchmark<T>>(numRows, nullPct);
      const auto suffix = fmt::format(
          "{}_{}Rows_{}PctNull_{}",
          typeName,
          numRows,
          nullPct,
          reportNulls ? "WithNulls" : "ValuesOnly");
      folly::addBenchmark(
          __FILE__,
          fmt::format("Scalar_{}", suffix),
          [fixture](unsigned iterations) {
            return fixture->template run<false, reportNulls>(iterations);
          });
      folly::addBenchmark(
          __FILE__,
          fmt::format("%SIMD_{}", suffix),
          [fixture](unsigned iterations) {
            return fixture->template run<true, reportNulls>(iterations);
          });
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  const folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize({});
  addBenchmarks<uint32_t, false>("uint32");
  addBenchmarks<uint32_t, true>("uint32");
  addBenchmarks<uint64_t, false>("uint64");
  addBenchmarks<uint64_t, true>("uint64");
  folly::runBenchmarks();
}
