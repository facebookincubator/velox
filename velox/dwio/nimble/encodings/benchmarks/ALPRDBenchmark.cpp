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

#include <bit>
#include <random>

#include "folly/Benchmark.h"
#include "folly/init/Init.h"
#include "velox/dwio/nimble/encodings/ALPRDEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"

using namespace facebook::nimble;
using namespace facebook::nimble::benchmarks;

namespace {

template <typename T>
Vector<T> makeInput(bool withExceptions) {
  using PhysicalType = typename TypeTraits<T>::physicalType;
  constexpr uint8_t kRightBitWidth = sizeof(T) * 8 - 16;
  constexpr PhysicalType kMask = (PhysicalType{1} << kRightBitWidth) - 1;
  std::mt19937_64 random(0xA1F0);
  Vector<T> values{benchmarkPool().get()};
  values.resize(kNumElements);
  for (uint32_t i = 0; i < values.size(); ++i) {
    // Use one common high part and a one-percent long tail of other prefixes.
    // The tail has more than eight keys, so the dictionary needs exceptions.
    const PhysicalType high = withExceptions && i % 100 == 99
        ? 0xc000 + (i / 100) % 256
        : (sizeof(T) == 8 ? 0x3ff1 : 0x3f81);
    const auto bits = static_cast<PhysicalType>(
        (high << kRightBitWidth) | (random() & kMask));
    values[i] = std::bit_cast<T>(bits);
  }
  return values;
}

template <typename T>
void encode(uint32_t iterations, bool withExceptions) {
  Vector<T> values{benchmarkPool().get()};
  BENCHMARK_SUSPEND {
    values = makeInput<T>(withExceptions);
  }
  // Include parameter training, child selection, allocations and serialization.
  encodeBenchmark<ALPRDEncoding<T>>(EncodingType::ALPRD, values, iterations);
}

template <typename T>
void decode(uint32_t iterations, bool withExceptions) {
  std::string encoded;
  std::vector<typename TypeTraits<T>::physicalType> output;
  BENCHMARK_SUSPEND {
    const auto values = makeInput<T>(withExceptions);
    encoded = encodeData<ALPRDEncoding<T>>(EncodingType::ALPRD, values);
    output.resize(values.size());
    const auto metadata = detail::alprd::readMetadata(encoded, {});
    NIMBLE_CHECK_EQ(metadata.exceptionCount != 0, withExceptions);
    // Validate the complete result before measuring the decode loop.
    auto reader =
        EncodingFactory{}.create(*benchmarkPool(), encoded, nullFactory());
    reader->materialize(output.size(), output.data());
    for (uint32_t i = 0; i < values.size(); ++i) {
      NIMBLE_CHECK_EQ(
          output[i],
          std::bit_cast<typename TypeTraits<T>::physicalType>(values[i]));
    }
  }
  while (iterations--) {
    // Include decoder construction, exception materialization and destruction.
    auto reader =
        EncodingFactory{}.create(*benchmarkPool(), encoded, nullFactory());
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
  folly::runBenchmarks();
}
