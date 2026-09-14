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
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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
DEFINE_bool(
    profile,
    false,
    "Report repeated Release timings and encoded sizes.");
DEFINE_string(
    profile_filter,
    "",
    "Only profile dataset names containing this text.");
DEFINE_uint32(
    profile_trials,
    5,
    "Number of timed trials per profile operation.");
DEFINE_uint32(
    profile_min_ms,
    10,
    "Minimum calibrated duration per profile trial.");

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
constexpr uint32_t kSamplingChunks = 32;

template <typename Function>
void profileOperation(
    const std::string& name,
    std::string_view operation,
    Function&& function) {
  const auto timeIterations = [&](uint64_t iterations) {
    const auto start = std::chrono::steady_clock::now();
    for (uint64_t iteration = 0; iteration < iterations; ++iteration) {
      function();
    }
    return std::chrono::duration<double, std::nano>(
               std::chrono::steady_clock::now() - start)
        .count();
  };

  uint64_t iterations = 1;
  while (timeIterations(iterations) < FLAGS_profile_min_ms * 1'000'000.0) {
    CHECK_LE(iterations, std::numeric_limits<uint64_t>::max() / 2);
    iterations *= 2;
  }
  std::vector<double> timings;
  for (uint32_t trial = 0; trial < FLAGS_profile_trials; ++trial) {
    timings.push_back(timeIterations(iterations) / iterations);
  }
  std::sort(timings.begin(), timings.end());
  const auto middle = timings.size() / 2;
  const double median = timings.size() % 2 == 0
      ? (timings[middle - 1] + timings[middle]) / 2
      : timings[middle];
  fmt::print(
      "PROFILE,{},{},{},{:.2f},{:.2f},{:.2f}\n",
      name,
      operation,
      iterations,
      median,
      timings.front(),
      timings.back());
}

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
        sample_(Alp::estimateSampleSize(values_.size())),
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
    int selectedExponent = -1;
    int selectedFactor = -1;
    if (encoding->encodingType() == EncodingType::ALP) {
      const char* position = encoded_.data() + encoding->dataOffset();
      const auto header = alp::readHeader(position);
      selectedExponent = header.exponent;
      selectedFactor = header.factor;
    }

    decode();
    for (size_t row = 0; row < values_.size(); ++row) {
      CHECK_EQ(alp::toPhysical<FloatType>(output_[row]), physicals_[row]);
    }

    uint64_t encodedHash = 14695981039346656037ULL;
    for (const unsigned char byte : encoded_) {
      encodedHash = (encodedHash ^ byte) * 1099511628211ULL;
    }
    const auto estimatedBytes = Alp::estimateSize(physicals_);
    CHECK(estimatedBytes.has_value());
    fmt::print(
        "{}: rows={} transform=({},0) exceptions={:.2f}% "
        "head-grid=({},{}) selected=({},{}) encoding={} bytes={} "
        "estimated={} encoded/raw={:.4f} hash={:016x}\n",
        name,
        values_.size(),
        exponent_,
        100.0 * (values_.size() - scalarCount) / values_.size(),
        batchGrid.exponent,
        batchGrid.factor,
        selectedExponent,
        selectedFactor,
        facebook::nimble::toString(encoding->encodingType()),
        encoded_.size(),
        *estimatedBytes,
        static_cast<double>(encoded_.size()) /
            (values_.size() * sizeof(FloatType)),
        encodedHash);
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

  template <bool chunked>
  void gatherSample() {
    const uint64_t rowCount = values_.size();
    const uint32_t sampleSize = sample_.size();
    if constexpr (chunked) {
      if (sampleSize == rowCount) {
        std::copy(physicals_.begin(), physicals_.end(), sample_.begin());
        folly::doNotOptimizeAway(sample_.data());
        return;
      }
      if (rowCount > kSamplingChunks && sampleSize > kSamplingChunks) {
        const uint32_t chunkSize = sampleSize / kSamplingChunks;
        uint32_t written = 0;
        for (uint32_t chunk = 0; chunk < kSamplingChunks; ++chunk) {
          const auto start = chunk * rowCount / kSamplingChunks;
          std::copy_n(
              physicals_.data() + start, chunkSize, sample_.data() + written);
          written += chunkSize;
        }
        while (written < sampleSize) {
          sample_[written++] = physicals_.back();
        }
        folly::doNotOptimizeAway(sample_.data());
        return;
      }
    }
    for (uint32_t index = 0; index < sampleSize; ++index) {
      sample_[index] = physicals_[index * rowCount / sampleSize];
    }
    folly::doNotOptimizeAway(sample_.data());
  }

  void estimate() const {
    folly::doNotOptimizeAway(Alp::estimateSize(physicals_));
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
  std::vector<PhysicalType> sample_;
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
  if (FLAGS_profile && name.find(FLAGS_profile_filter) == std::string::npos) {
    return;
  }
  auto fixture = std::make_shared<AlpBenchmarkFixture<FloatType>>(
      name, std::move(values), exponent);

  if (FLAGS_profile) {
    profileOperation(name, "TransformScalar", [&] {
      folly::doNotOptimizeAway(fixture->template runTransform<false>());
    });
    profileOperation(name, "TransformBatch", [&] {
      folly::doNotOptimizeAway(fixture->template runTransform<true>());
    });
    profileOperation(name, "GridScalar", [&] {
      folly::doNotOptimizeAway(fixture->template runGrid<false>());
    });
    profileOperation(name, "GridBatch", [&] {
      folly::doNotOptimizeAway(fixture->template runGrid<true>());
    });
    profileOperation(name, "SamplingLegacy", [&] {
      fixture->template gatherSample<false>();
    });
    profileOperation(name, "SamplingChunked", [&] {
      fixture->template gatherSample<true>();
    });
    profileOperation(name, "Estimate", [&] { fixture->estimate(); });
    profileOperation(name, "Encode", [&] {
      Buffer buffer{*benchmarkPool()};
      folly::doNotOptimizeAway(fixture->encode(buffer));
    });
    profileOperation(name, "Decode", [&] { fixture->decode(); });
    return;
  }

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
  folly::addBenchmark(
      __FILE__, fmt::format("SamplingLegacy_{}", name), [fixture] {
        fixture->template gatherSample<false>();
        return 1;
      });
  folly::addBenchmark(
      __FILE__, fmt::format("%SamplingChunked_{}", name), [fixture] {
        fixture->template gatherSample<true>();
        return 1;
      });
  folly::addBenchmark(__FILE__, fmt::format("Estimate_{}", name), [fixture] {
    fixture->estimate();
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

  auto headBiased = makeValues<FloatType>([&](auto& random) {
    return static_cast<FloatType>(cents(random)) / static_cast<FloatType>(100);
  });
  const auto headSize = std::min<size_t>(1024, headBiased.size() / 4);
  std::fill_n(headBiased.begin(), headSize, FloatType{1});
  registerDataset("HeadBiased", std::move(headBiased), 2);

  uint32_t row = 0;
  registerDataset(
      "SparseExceptions",
      makeValues<FloatType>([&](auto& random) {
        return row++ % 50 == 0
            ? std::numeric_limits<FloatType>::infinity()
            : FloatType{1'000'000} + static_cast<FloatType>(cents(random) % 16);
      }),
      0);
}

} // namespace

int main(int argc, char** argv) {
  const folly::Init init{&argc, &argv};
  CHECK_GT(FLAGS_rows, 0);
  CHECK_GT(FLAGS_profile_trials, 0);
  CHECK_GT(FLAGS_profile_min_ms, 0);
  facebook::velox::memory::MemoryManager::initialize({});
  fmt::print(
      "Transform/encode/decode times are per {} rows; grid times are per "
      "{}-value prefix sample. Relative rows compare batch with scalar "
      "or chunked with legacy sampling.\n",
      FLAGS_rows,
      ALPEncoding<double>::estimateSampleSize(FLAGS_rows));
  registerDatasets<double>();
  registerDatasets<float>();
  if (!FLAGS_profile) {
    folly::runBenchmarks();
  }
}
