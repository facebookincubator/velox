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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

DEFINE_uint32(
    rows,
    65'536,
    "Values per dataset; use --bm_regex to select cases.");

namespace {

using facebook::nimble::ALPEncoding;
using facebook::nimble::Buffer;
using facebook::nimble::EncodingFactory;
using facebook::nimble::EncodingType;
using facebook::nimble::ManualEncodingSelectionPolicy;
using facebook::nimble::TypeTraits;
using facebook::nimble::benchmarks::benchmarkPool;
using facebook::nimble::benchmarks::nullFactory;
namespace alp = facebook::nimble::detail::alp;

constexpr uint64_t kSeed = 0x51ED0FF1CEULL;

template <typename FloatType, typename Generator>
std::vector<FloatType> makeValues(Generator generator) {
  std::mt19937_64 random(kSeed);
  std::vector<FloatType> values;
  values.reserve(FLAGS_rows);
  for (uint32_t row = 0; row < FLAGS_rows; ++row) {
    values.push_back(generator(random));
  }
  return values;
}

template <typename FloatType>
std::vector<FloatType> makeSensorValues(double outlierRate) {
  std::uniform_int_distribution<int32_t> clean(0, 99'999);
  std::uniform_int_distribution<int64_t> outlier(
      5'000'000'000LL, 9'999'999'999LL);
  std::uniform_real_distribution<double> probability(0.0, 1.0);
  return makeValues<FloatType>([&](auto& random) {
    return probability(random) < outlierRate
        ? static_cast<FloatType>(outlier(random)) /
            static_cast<FloatType>(1'000'000)
        : static_cast<FloatType>(clean(random)) / static_cast<FloatType>(1'000);
  });
}

struct GridResult {
  uint8_t exponent{0};
  uint8_t factor{0};
  uint32_t representable{0};

  bool operator==(const GridResult&) const = default;
};

template <typename FloatType>
class AlpBenchmarkFixture {
 public:
  using Alp = ALPEncoding<FloatType>;
  using PhysicalType = typename TypeTraits<FloatType>::physicalType;

  AlpBenchmarkFixture(
      const std::string& name,
      std::vector<FloatType> values,
      uint8_t exponent)
      : values_{std::move(values)},
        physicals_(values_.size()),
        zigZag_(values_.size()),
        mask_(values_.size()),
        output_(values_.size()),
        exponent_{exponent} {
    for (size_t row = 0; row < values_.size(); ++row) {
      physicals_[row] = alp::toPhysical<FloatType>(values_[row]);
    }

    const auto scalarCount = runTransform<false>();
    const auto scalarZigZag = zigZag_;
    const auto scalarMask = mask_;
    CHECK_EQ(runTransform<true>(), scalarCount);
    CHECK(scalarZigZag == zigZag_);
    CHECK(scalarMask == mask_);

    const auto scalarGrid = runGrid<false>();
    const auto batchGrid = runGrid<true>();
    CHECK(scalarGrid == batchGrid);

    Buffer buffer{*benchmarkPool()};
    encoded_ = std::string{encode(buffer)};
    auto encoding =
        EncodingFactory{}.create(*benchmarkPool(), encoded_, nullFactory());
    if (encoding->encodingType() == EncodingType::ALP) {
      const char* position = encoded_.data() + encoding->dataOffset();
      const auto header = alp::readHeader(position);
      CHECK_EQ(header.exponent, batchGrid.exponent);
      CHECK_EQ(header.factor, batchGrid.factor);
    }

    decode();
    for (size_t row = 0; row < values_.size(); ++row) {
      CHECK_EQ(alp::toPhysical<FloatType>(output_[row]), physicals_[row]);
    }

    fmt::print(
        "{}: rows={} transform=({},0) exceptions={:.2f}% "
        "grid=({},{}) encoding={} bytes={} encoded/raw={:.4f}\n",
        name,
        values_.size(),
        exponent_,
        100.0 * (values_.size() - scalarCount) / values_.size(),
        batchGrid.exponent,
        batchGrid.factor,
        facebook::nimble::toString(encoding->encodingType()),
        encoded_.size(),
        static_cast<double>(encoded_.size()) /
            (values_.size() * sizeof(FloatType)));
  }

  template <bool useBatch>
  uint32_t runTransform() {
    const auto count = transform<useBatch, true>(values_.size(), exponent_, 0);
    folly::doNotOptimizeAway(zigZag_.data());
    folly::doNotOptimizeAway(mask_.data());
    return count;
  }

  template <bool useBatch>
  GridResult runGrid() {
    folly::doNotOptimizeAway(values_.data());
    const auto sampleSize = Alp::estimateSampleSize(values_.size());
    GridResult best;
    for (uint8_t exponent = 0; exponent < Alp::kPow10Double.size();
         ++exponent) {
      for (uint8_t factor = 0; factor <= exponent; ++factor) {
        const auto count =
            transform<useBatch, false>(sampleSize, exponent, factor);
        if (count > best.representable) {
          best = {exponent, factor, count};
        }
        if (best.representable == sampleSize) {
          return best;
        }
      }
    }
    return best;
  }

  std::string_view encode(Buffer& buffer) const {
    auto policy = std::make_unique<ManualEncodingSelectionPolicy<FloatType>>(
        std::vector<std::pair<EncodingType, float>>{
            {EncodingType::ALP, 1.0},
            {EncodingType::Trivial, 1.0},
            {EncodingType::FixedBitWidth, 1.0}},
        std::nullopt,
        std::nullopt);
    return EncodingFactory::encode<FloatType>(
        std::move(policy), std::span<const FloatType>{values_}, buffer);
  }

  void decode() {
    auto encoding =
        EncodingFactory{}.create(*benchmarkPool(), encoded_, nullFactory());
    encoding->materialize(values_.size(), output_.data());
    folly::doNotOptimizeAway(output_.data());
  }

 private:
  template <bool useBatch, bool materialize>
  FOLLY_NOINLINE uint32_t
  transform(size_t size, uint8_t exponent, uint8_t factor) {
    const double exponentMultiplier = Alp::kPow10Double[exponent];
    const double factorMultiplier = Alp::kPow10Double[factor];
    uint32_t representableCount = 0;
    auto record = [&](size_t row, bool representable, uint64_t zigZag) {
      representableCount += representable;
      if constexpr (materialize) {
        mask_[row] = representable;
        zigZag_[row] = zigZag;
      }
    };

    size_t row = 0;
    if constexpr (useBatch) {
      constexpr auto kBatchSize = Alp::kBatchSize;
      alignas(64) uint64_t zigZagLanes[kBatchSize];
      alignas(64) bool maskLanes[kBatchSize];
      for (; row + kBatchSize <= size; row += kBatchSize) {
        Alp::batchTransform(
            values_.data() + row,
            physicals_.data() + row,
            exponentMultiplier,
            factorMultiplier,
            zigZagLanes,
            maskLanes);
        for (size_t lane = 0; lane < kBatchSize; ++lane) {
          record(
              row + lane,
              maskLanes[lane],
              maskLanes[lane] ? zigZagLanes[lane] : 0);
        }
      }
    }
    for (; row < size; ++row) {
      uint64_t zigZag = 0;
      const bool representable = Alp::scalarTransformOne(
          values_[row],
          physicals_[row],
          exponentMultiplier,
          factorMultiplier,
          zigZag);
      record(row, representable, zigZag);
    }
    return representableCount;
  }

  std::vector<FloatType> values_;
  std::vector<PhysicalType> physicals_;
  std::vector<uint64_t> zigZag_;
  std::vector<uint8_t> mask_;
  std::vector<FloatType> output_;
  uint8_t exponent_;
  std::string encoded_;
};

template <typename FloatType>
void registerDataset(
    const std::string& dataset,
    std::vector<FloatType> values,
    uint8_t exponent) {
  const auto name = fmt::format(
      "{}_{}", std::is_same_v<FloatType, float> ? "Float" : "Double", dataset);
  auto fixture = std::make_shared<AlpBenchmarkFixture<FloatType>>(
      name, std::move(values), exponent);

  folly::addBenchmark(
      __FILE__, fmt::format("TransformScalar_{}", name), [fixture] {
        folly::doNotOptimizeAway(fixture->template runTransform<false>());
        return 1;
      });
  folly::addBenchmark(
      __FILE__, fmt::format("%TransformBatch_{}", name), [fixture] {
        folly::doNotOptimizeAway(fixture->template runTransform<true>());
        return 1;
      });
  folly::addBenchmark(__FILE__, fmt::format("GridScalar_{}", name), [fixture] {
    folly::doNotOptimizeAway(fixture->template runGrid<false>());
    return 1;
  });
  folly::addBenchmark(__FILE__, fmt::format("%GridBatch_{}", name), [fixture] {
    folly::doNotOptimizeAway(fixture->template runGrid<true>());
    return 1;
  });
  folly::addBenchmark(__FILE__, fmt::format("Encode_{}", name), [fixture] {
    Buffer buffer{*benchmarkPool()};
    folly::doNotOptimizeAway(fixture->encode(buffer));
    return 1;
  });
  folly::addBenchmark(__FILE__, fmt::format("Decode_{}", name), [fixture] {
    fixture->decode();
    return 1;
  });
}

template <typename FloatType>
void registerDatasets() {
  std::uniform_int_distribution<int32_t> integers(-1'000'000, 1'000'000);
  registerDataset(
      "Integers",
      makeValues<FloatType>([&](auto& random) {
        return static_cast<FloatType>(integers(random));
      }),
      0);

  std::uniform_int_distribution<int32_t> cents(0, 999'999);
  registerDataset(
      "TwoDecimal",
      makeValues<FloatType>([&](auto& random) {
        return static_cast<FloatType>(cents(random)) /
            static_cast<FloatType>(100);
      }),
      2);

  std::normal_distribution<double> prices(500.0, 120.0);
  registerDataset(
      "Prices",
      makeValues<FloatType>([&](auto& random) {
        const auto rounded = std::llround(std::abs(prices(random)) * 100.0);
        return static_cast<FloatType>(rounded) / static_cast<FloatType>(100);
      }),
      2);

  std::uniform_int_distribution<int64_t> seconds(
      1'700'000'000LL, 1'800'000'000LL);
  std::uniform_int_distribution<int32_t> milliseconds(0, 999);
  registerDataset(
      "Timestamps",
      makeValues<FloatType>([&](auto& random) {
        const auto value = seconds(random) * 1'000LL + milliseconds(random);
        return static_cast<FloatType>(value) / static_cast<FloatType>(1'000);
      }),
      3);

  const std::array<double, 8> palette{
      0.1234, 1.5000, 2.7182, 3.1415, 42.0000, 100.9999, 999.0001, 12.3456};
  std::uniform_int_distribution<uint32_t> pick(0, palette.size() - 1);
  registerDataset(
      "LowCardinality",
      makeValues<FloatType>([&](auto& random) {
        return static_cast<FloatType>(palette[pick(random)]);
      }),
      4);

  registerDataset("Sensor2Percent", makeSensorValues<FloatType>(0.02), 3);
  registerDataset(
      "MixedPrecision30Percent", makeSensorValues<FloatType>(0.30), 3);

  std::uniform_real_distribution<double> chaotic(-1e6, 1e6);
  registerDataset(
      "Chaotic",
      makeValues<FloatType>([&](auto& random) {
        return static_cast<FloatType>(chaotic(random));
      }),
      2);
}

} // namespace

int main(int argc, char** argv) {
  const folly::Init init{&argc, &argv};
  CHECK_GT(FLAGS_rows, 0);
  facebook::velox::memory::MemoryManager::initialize({});
  fmt::print(
      "Transform/encode/decode times are per {} rows; grid times are per "
      "{}-value sample. Relative rows compare batch with scalar.\n",
      FLAGS_rows,
      ALPEncoding<double>::estimateSampleSize(FLAGS_rows));
  registerDatasets<double>();
  registerDatasets<float>();
  folly::runBenchmarks();
}
