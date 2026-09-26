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

#include <time.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "fmt/format.h"
#include "folly/Benchmark.h"
#include "folly/init/Init.h"
#include "folly/lang/Bits.h"
#include "gflags/gflags.h"
#include "velox/dwio/nimble/encodings/ALPRDEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"

DEFINE_bool(
    profile,
    false,
    "Report ALP_RD size and repeated wall/CPU timings.");
DEFINE_bool(
    selection_profile,
    false,
    "Compare the configured candidates with and without ALPRD.");
DEFINE_string(
    selection_read_factors,
    "Constant=1;Trivial=1;FixedBitWidth=1;MainlyConstant=1;SparseBool=1;"
    "Dictionary=1;RLE=1;Varint=1;ALP=1",
    "Baseline candidates and weights for selection_profile; excludes ALPRD. "
    "The default uses equal weights to compare estimated sizes.");
DEFINE_double(alprd_read_factor, 1.0, "ALPRD weight in selection_profile.");
DEFINE_uint32(
    rows,
    65'536,
    "Generated values, or maximum values read from input_file.");
DEFINE_uint32(profile_trials, 7, "Timed trials per operation.");
DEFINE_uint32(profile_min_ms, 50, "Minimum calibrated CPU time per trial.");
DEFINE_string(profile_type, "all", "Value type: float, double, or all.");
DEFINE_string(profile_dataset, "", "Synthetic dataset; empty selects all.");
DEFINE_string(
    input_file,
    "",
    "Raw little-endian IEEE values; requires profile and one profile_type.");
DEFINE_bool(
    exact_bits,
    false,
    "Use Nimble's existing exact-bit child encoding option.");

using namespace facebook::nimble;
using namespace facebook::nimble::benchmarks;

namespace {

constexpr uint64_t kSeed = 0xA1F0;
constexpr std::array<std::string_view, 12> kDatasets{
    "shared_prefix",
    "eight_prefixes",
    "mixed_sign",
    "exceptions_1pct",
    "exceptions_10pct",
    "duckdb_best_case",
    "wide_exponents",
    "random_bits",
    "decimals",
    "dictionary",
    "runs",
    "mostly_constant",
};

Encoding::Options encodingOptions() {
  Encoding::Options options;
  options.fixedBitWidthUseExactBits = FLAGS_exact_bits;
  return options;
}

template <typename T>
Vector<T> makeInput(std::string_view dataset) {
  using P = typename TypeTraits<T>::physicalType;
  constexpr auto kMantissaBits = std::numeric_limits<T>::digits - 1;
  constexpr auto kRightBits = sizeof(T) * 8 - 16;
  constexpr P kSign = P{1} << (sizeof(T) * 8 - 1);
  constexpr P kMantissaMask = (P{1} << kMantissaBits) - 1;
  constexpr P kLowMask = (P{1} << kRightBits) - 1;
  constexpr P kCommonHigh = sizeof(T) == 8 ? 0x3ff1 : 0x3f81;
  constexpr auto kExponentBias = std::numeric_limits<T>::max_exponent - 1;
  const uint32_t exceptionThreshold = dataset == "exceptions_1pct" ? 100
      : dataset == "exceptions_10pct"                              ? 1'000
                                                                   : 0;
  std::mt19937_64 random(kSeed);
  Vector<T> values{benchmarkPool().get()};
  values.resize(FLAGS_rows);
  for (auto& value : values) {
    P bits;
    if (dataset == "decimals") {
      value = static_cast<T>(random() % 10'000) / 100;
      continue;
    } else if (dataset == "random_bits") {
      bits = static_cast<P>(random());
    } else if (dataset == "wide_exponents") {
      const P exponent = random() % 201 + kExponentBias - 100;
      bits = (exponent << kMantissaBits) | (random() & kMantissaMask);
      if (random() & 1) {
        bits |= kSign;
      }
    } else if (dataset == "duckdb_best_case") {
      // Follow DuckDB's ALP_RD best-case workload: random() + 10.
      value =
          static_cast<T>(10.0 + std::generate_canonical<double, 53>(random));
      continue;
    } else {
      P high = kCommonHigh;
      if (dataset == "eight_prefixes") {
        high += random() % 8;
      } else if (
          (dataset == "mixed_sign" || dataset == "dictionary" ||
           dataset == "runs" || dataset == "mostly_constant") &&
          (random() & 1)) {
        high |= 0x8000;
      } else if (exceptionThreshold && random() % 10'000 < exceptionThreshold) {
        // Random positions avoid aliasing evenly spaced training samples.
        high = 0xc000 + random() % 256;
      }
      bits = (high << kRightBits) | (random() & kLowMask);
    }
    value = std::bit_cast<T>(bits);
  }
  if (dataset == "dictionary") {
    for (size_t i = 512; i < values.size(); ++i) {
      values[i] = values[i % 512];
    }
  } else if (dataset == "runs") {
    for (size_t i = values.size(); i > 0; --i) {
      values[i - 1] = values[(i - 1) / 8];
    }
  } else if (dataset == "mostly_constant") {
    for (size_t i = 0; i < values.size(); ++i) {
      if (i % 8 != 0) {
        values[i] = T{0};
      }
    }
  }
  if (dataset == "random_bits") {
    const P infinity = std::bit_cast<P>(std::numeric_limits<T>::infinity());
    const std::array<P, 10> special{
        0,
        kSign,
        infinity,
        kSign | infinity,
        infinity | 1,
        infinity | (P{1} << (kMantissaBits - 1)) | 0x1234,
        1,
        kSign | 1,
        infinity - 1,
        kSign | (infinity - 1)};
    for (size_t i = 0; i < std::min(values.size(), special.size()); ++i) {
      values[i] = std::bit_cast<T>(special[i]);
    }
  }
  return values;
}

template <typename T>
Vector<T> readInput() {
  using P = typename TypeTraits<T>::physicalType;
  std::ifstream file(FLAGS_input_file, std::ios::binary | std::ios::ate);
  NIMBLE_USER_CHECK(
      file.is_open(), "Cannot open input file: {}", FLAGS_input_file);
  const auto bytes = static_cast<std::streamoff>(file.tellg());
  NIMBLE_USER_CHECK_GT(bytes, 0, "Input file must contain at least one value.");
  NIMBLE_USER_CHECK_EQ(
      bytes % sizeof(T), 0, "Input file has a partial IEEE value.");
  Vector<T> values{benchmarkPool().get()};
  values.resize(std::min<uint64_t>(FLAGS_rows, bytes / sizeof(T)));
  const auto readBytes =
      static_cast<std::streamsize>(values.size() * sizeof(T));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(values.data()), readBytes);
  NIMBLE_USER_CHECK_EQ(
      file.gcount(), readBytes, "Could not read the input values.");
  for (auto& value : values) {
    value = std::bit_cast<T>(folly::Endian::little(std::bit_cast<P>(value)));
  }
  return values;
}

template <typename T>
void validate(
    const Vector<T>& values,
    const std::vector<typename TypeTraits<T>::physicalType>& output) {
  using P = typename TypeTraits<T>::physicalType;
  for (size_t i = 0; i < values.size(); ++i) {
    NIMBLE_CHECK_EQ(
        output[i], std::bit_cast<P>(values[i]), "Mismatch at row {}", i);
  }
}

struct Timings {
  double wallNs;
  double cpuNs;
};

int64_t threadCpuNs() {
  timespec time{};
  NIMBLE_CHECK_EQ(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time), 0);
  return time.tv_sec * 1'000'000'000LL + time.tv_nsec;
}

struct Operation {
  std::string_view name;
  std::function<void(uint32_t)> run;
  uint32_t iterations{1};
  std::vector<double> wallSamples{};
  std::vector<double> cpuSamples{};

  Timings time() const {
    const auto cpuStart = threadCpuNs();
    const auto start = std::chrono::steady_clock::now();
    run(iterations);
    const auto end = std::chrono::steady_clock::now();
    const auto cpuEnd = threadCpuNs();
    return {
        std::chrono::duration<double, std::nano>(end - start).count(),
        static_cast<double>(cpuEnd - cpuStart)};
  }
};

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const auto middle = samples.size() / 2;
  return samples.size() % 2 ? samples[middle]
                            : (samples[middle - 1] + samples[middle]) / 2;
}

// Calibrates and interleaves every comparison so ordering does not favor one
// candidate set. All input generation, validation and reporting stay untimed.
template <typename Report>
void measureOperations(std::span<Operation> operations, Report report) {
  for (auto& operation : operations) {
    while (operation.time().cpuNs < FLAGS_profile_min_ms * 1'000'000.0) {
      NIMBLE_CHECK_LE(
          operation.iterations, std::numeric_limits<uint32_t>::max() / 2);
      operation.iterations *= 2;
    }
  }
  std::vector<size_t> order(operations.size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937_64 random(kSeed);
  for (uint32_t trial = 0; trial < FLAGS_profile_trials; ++trial) {
    std::shuffle(order.begin(), order.end(), random);
    for (const auto index : order) {
      auto& operation = operations[index];
      auto timings = operation.time();
      timings.wallNs /= operation.iterations;
      timings.cpuNs /= operation.iterations;
      operation.wallSamples.push_back(timings.wallNs);
      operation.cpuSamples.push_back(timings.cpuNs);
      report(index, trial, timings);
    }
  }
  for (size_t i = 0; i < operations.size(); ++i) {
    report(
        i,
        -1,
        Timings{
            median(operations[i].wallSamples),
            median(operations[i].cpuSamples)});
  }
  std::fflush(stdout);
}

std::string formatLayout(const EncodingLayout& layout) {
  auto result = toString(layout.encodingType());
  if (layout.childrenCount() != 0) {
    result += "[";
    for (uint8_t i = 0; i < layout.childrenCount(); ++i) {
      if (i != 0) {
        result += ";";
      }
      result += layout.child(i) ? formatLayout(*layout.child(i)) : "-";
    }
    result += "]";
  }
  return result;
}

template <typename T>
void profileSelection(std::string_view dataset) {
  using P = typename TypeTraits<T>::physicalType;
  const auto values =
      dataset == "input" ? readInput<T>() : makeInput<T>(dataset);
  const auto physicals = EncodingPhysicalType<T>::asEncodingPhysicalTypeSpan(
      std::span<const T>{values.data(), values.size()});
  const auto options = encodingOptions();
  auto factors = ManualEncodingSelectionPolicyFactory::parseEncodingReadFactors(
      FLAGS_selection_read_factors);
  NIMBLE_USER_CHECK(
      std::none_of(
          factors.begin(),
          factors.end(),
          [](const auto& entry) { return entry.first == EncodingType::ALPRD; }),
      "selection_read_factors must exclude ALPRD.");
  const ManualEncodingSelectionPolicyFactory baseline(factors, std::nullopt);
  factors.emplace_back(EncodingType::ALPRD, FLAGS_alprd_read_factor);
  const ManualEncodingSelectionPolicyFactory candidate(factors, std::nullopt);
  const std::array<const ManualEncodingSelectionPolicyFactory*, 2> factories{
      &baseline, &candidate};
  const auto makePolicy = [&](size_t variant) {
    return std::unique_ptr<
        EncodingSelectionPolicy<T>>(static_cast<EncodingSelectionPolicy<T>*>(
        factories[variant]->createPolicy(TypeTraits<T>::dataType).release()));
  };
  std::array<std::string, 2> encoded;
  std::array<std::string, 2> layouts;
  std::array<uint64_t, 2> estimates{};
  std::vector<P> output(values.size());
  const EncodingFactory factory(options);
  std::vector<Operation> operations;
  for (size_t variant = 0; variant < factories.size(); ++variant) {
    Buffer buffer{*benchmarkPool()};
    encoded[variant] = EncodingFactory::encode<T>(
        makePolicy(variant), {values.data(), values.size()}, buffer, options);
    layouts[variant] =
        formatLayout(EncodingLayoutCapture::capture(encoded[variant], options));
    estimates[variant] =
        makePolicy(variant)
            ->select(physicals, Statistics<P>::create(physicals), options)
            .estimatedSize.value_or(0);
    auto decoder =
        factory.create(*benchmarkPool(), encoded[variant], nullFactory());
    decoder->materialize(output.size(), output.data());
    validate(values, output);
    operations.push_back(
        {"select", [&, variant](uint32_t iterations) {
           while (iterations--) {
             // Rebuild lazy statistics as encode() does; do not reuse warmed
             // stats.
             const auto result = makePolicy(variant)->select(
                 physicals, Statistics<P>::create(physicals), options);
             folly::doNotOptimizeAway(result);
           }
         }});
    operations.push_back({"encode", [&, variant](uint32_t iterations) {
                            Buffer scratch{*benchmarkPool()};
                            while (iterations--) {
                              scratch.reset();
                              const auto result = EncodingFactory::encode<T>(
                                  makePolicy(variant),
                                  {values.data(), values.size()},
                                  scratch,
                                  options);
                              folly::doNotOptimizeAway(result);
                            }
                          }});
    operations.push_back(
        {"construct_decode", [&, variant](uint32_t iterations) {
           while (iterations--) {
             auto reader = factory.create(
                 *benchmarkPool(), encoded[variant], nullFactory());
             reader->materialize(output.size(), output.data());
             folly::doNotOptimizeAway(output.data());
           }
         }});
  }
  measureOperations(
      operations, [&](size_t index, int32_t trial, Timings timings) {
        const auto variant = index / 3;
        const auto& operation = operations[index];
        fmt::print(
            "{},{},{},{},{},{},{},{},{},{},{},{},{:.2f},{:.2f}\n",
            trial < 0 ? "median" : "trial",
            sizeof(T) == 4 ? "float" : "double",
            dataset,
            values.size(),
            variant == 0 ? "baseline" : "with_alprd",
            operation.name,
            trial,
            operation.iterations,
            encoded[variant].size(),
            estimates[variant],
            layouts[variant],
            FLAGS_exact_bits ? 1 : 0,
            timings.wallNs,
            timings.cpuNs);
      });
  validate(values, output);
}

template <typename T>
void profileDataset(std::string_view dataset) {
  if (FLAGS_selection_profile) {
    profileSelection<T>(dataset);
    return;
  }

  using P = typename TypeTraits<T>::physicalType;
  const auto values =
      dataset == "input" ? readInput<T>() : makeInput<T>(dataset);
  const auto options = encodingOptions();
  const auto encoded =
      encodeData<ALPRDEncoding<T>>(EncodingType::ALPRD, values, options);
  const auto metadata = ALPRDEncodingBase::readMetadata(encoded, options);
  std::vector<P> output(values.size());
  const auto physicals = std::span<const P>(
      reinterpret_cast<const P*>(values.data()), values.size());
  const EncodingFactory factory(options);
  auto reader = factory.create(*benchmarkPool(), encoded, nullFactory());
  auto constructDecode = [&](uint32_t iterations) {
    while (iterations--) {
      auto decoder = factory.create(*benchmarkPool(), encoded, nullFactory());
      decoder->materialize(output.size(), output.data());
      folly::doNotOptimizeAway(output.data());
    }
  };
  auto resetDecode = [&](uint32_t iterations) {
    while (iterations--) {
      reader->reset();
      reader->materialize(output.size(), output.data());
      folly::doNotOptimizeAway(output.data());
    }
  };
  // Validate both decoder lifecycles outside the timed regions.
  constructDecode(1);
  validate(values, output);
  resetDecode(1);
  validate(values, output);
  std::array<Operation, 4> operations{
      {{"train",
        [&](uint32_t iterations) {
          while (iterations--) {
            const auto parameters = ALPRDEncodingBase::selectParameters(
                physicals, options, nullptr);
            folly::doNotOptimizeAway(parameters);
          }
        }},
       {"encode",
        [&](uint32_t iterations) {
          // Includes training, statistics, child selection and serialization.
          // The setup-only string copy in encodeData is not part of this loop.
          encodeBenchmark<ALPRDEncoding<T>>(
              EncodingType::ALPRD, values, iterations, options);
        }},
       {"construct_decode", constructDecode},
       {"reset_decode", resetDecode}}};
  const auto print = [&](std::string_view record,
                         const Operation& operation,
                         int32_t trial,
                         Timings timings) {
    fmt::print(
        "{},{},{},{},{},{},{},{},{:.4f},{},{},{},{},{:.2f},{:.2f}\n",
        record,
        sizeof(T) == 4 ? "float" : "double",
        dataset,
        values.size(),
        operation.name,
        trial,
        operation.iterations,
        encoded.size(),
        8.0 * encoded.size() / values.size(),
        metadata.exceptionCount,
        metadata.parameters.rightBitWidth,
        metadata.parameters.dictionarySize,
        FLAGS_exact_bits ? 1 : 0,
        timings.wallNs,
        timings.cpuNs);
  };
  measureOperations(
      operations, [&](size_t index, int32_t trial, Timings timings) {
        print(
            trial < 0 ? "median" : "trial", operations[index], trial, timings);
      });
  validate(values, output);
}

void runProfile() {
  NIMBLE_USER_CHECK_GT(FLAGS_profile_trials, 0);
  NIMBLE_USER_CHECK_GT(FLAGS_profile_min_ms, 0);
  NIMBLE_USER_CHECK(
      FLAGS_profile_type == "all" || FLAGS_profile_type == "float" ||
      FLAGS_profile_type == "double");
  NIMBLE_USER_CHECK(
      FLAGS_profile_dataset.empty() ||
          std::find(
              kDatasets.begin(), kDatasets.end(), FLAGS_profile_dataset) !=
              kDatasets.end(),
      "Unknown profile_dataset: {}",
      FLAGS_profile_dataset);
  NIMBLE_USER_CHECK(
      FLAGS_input_file.empty() ||
          (FLAGS_profile_type != "all" && FLAGS_profile_dataset.empty()),
      "input_file requires one profile_type and no synthetic dataset filter.");
  if (FLAGS_selection_profile) {
    fmt::print(
        "record,type,dataset,rows,variant,operation,trial,iterations,"
        "encoded_bytes,estimated_bytes,layout,exact_bits,wall_ns,cpu_ns\n");
  } else {
    fmt::print(
        "record,type,dataset,rows,operation,trial,iterations,encoded_bytes,"
        "bits_per_value,exceptions,right_bit_width,dictionary_size,exact_bits,wall_ns,cpu_ns\n");
  }
  if (!FLAGS_input_file.empty()) {
    if (FLAGS_profile_type == "float") {
      profileDataset<float>("input");
    } else {
      profileDataset<double>("input");
    }
    return;
  }
  for (const auto dataset : kDatasets) {
    if (!FLAGS_profile_dataset.empty() && FLAGS_profile_dataset != dataset) {
      continue;
    }
    if (FLAGS_profile_type != "double") {
      profileDataset<float>(dataset);
    }
    if (FLAGS_profile_type != "float") {
      profileDataset<double>(dataset);
    }
  }
}

template <typename T>
void encode(uint32_t iterations, bool withExceptions) {
  Vector<T> values{benchmarkPool().get()};
  BENCHMARK_SUSPEND {
    values = makeInput<T>(withExceptions ? "exceptions_1pct" : "shared_prefix");
  }
  encodeBenchmark<ALPRDEncoding<T>>(
      EncodingType::ALPRD, values, iterations, encodingOptions());
}

template <typename T>
void decode(uint32_t iterations, bool withExceptions) {
  std::string encoded;
  std::vector<typename TypeTraits<T>::physicalType> output;
  const auto options = encodingOptions();
  const EncodingFactory factory(options);
  BENCHMARK_SUSPEND {
    const auto values =
        makeInput<T>(withExceptions ? "exceptions_1pct" : "shared_prefix");
    encoded =
        encodeData<ALPRDEncoding<T>>(EncodingType::ALPRD, values, options);
    output.resize(values.size());
    auto reader = factory.create(*benchmarkPool(), encoded, nullFactory());
    reader->materialize(output.size(), output.data());
    validate(values, output);
  }
  while (iterations--) {
    auto reader = factory.create(*benchmarkPool(), encoded, nullFactory());
    reader->materialize(output.size(), output.data());
    folly::doNotOptimizeAway(output.data());
  }
}

BENCHMARK(ALPRD_Encode_Float_CommonPrefix, iterations) {
  encode<float>(iterations, false);
}

BENCHMARK(ALPRD_ConstructAndDecode_Float_CommonPrefix, iterations) {
  decode<float>(iterations, false);
}

BENCHMARK(ALPRD_Encode_Float_Exceptions, iterations) {
  encode<float>(iterations, true);
}

BENCHMARK(ALPRD_ConstructAndDecode_Float_Exceptions, iterations) {
  decode<float>(iterations, true);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(ALPRD_Encode_Double_CommonPrefix, iterations) {
  encode<double>(iterations, false);
}

BENCHMARK(ALPRD_ConstructAndDecode_Double_CommonPrefix, iterations) {
  decode<double>(iterations, false);
}

BENCHMARK(ALPRD_Encode_Double_Exceptions, iterations) {
  encode<double>(iterations, true);
}

BENCHMARK(ALPRD_ConstructAndDecode_Double_Exceptions, iterations) {
  decode<double>(iterations, true);
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  facebook::velox::memory::MemoryManager::initialize({});
  NIMBLE_USER_CHECK_GT(FLAGS_rows, 0);
  NIMBLE_USER_CHECK(
      FLAGS_profile || FLAGS_selection_profile || FLAGS_input_file.empty(),
      "input_file requires --profile.");
  if (FLAGS_profile || FLAGS_selection_profile) {
    runProfile();
  } else {
    folly::runBenchmarks();
  }
}
