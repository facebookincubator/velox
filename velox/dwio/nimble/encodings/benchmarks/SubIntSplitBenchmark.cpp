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

#include "folly/Benchmark.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"

using namespace facebook::nimble;
using namespace facebook::nimble::benchmarks;

namespace {

// Values with a constant high prefix and a varying low half -- the canonical
// case SubIntSplit targets (e.g. small counters in a wide field, or timestamps
// whose high bits rarely change). The selector should split off the constant
// high section and bit-pack the low section.
template <typename T>
Vector<T> makeBitStructured(int lowBits, uint32_t n = kNumElements) {
  auto& pool = benchmarkPool();
  Vector<T> data{pool.get()};
  data.resize(n);
  using U = std::make_unsigned_t<T>;
  const U prefix = (sizeof(T) == 4) ? static_cast<U>(0x12340000u)
                                    : static_cast<U>(0x1234567800000000ULL);
  const U lowMask = (static_cast<U>(1) << lowBits) - 1;
  for (uint32_t i = 0; i < n; ++i) {
    const U bits =
        prefix | (static_cast<U>(folly::Random::secureRand64()) & lowMask);
    data[i] = std::bit_cast<T>(bits);
  }
  return data;
}

} // namespace

// Encode + decode benchmark pair for a given 64-bit data pattern.
#define SUBINT_BENCH(Pattern, DataExpr)                      \
  BENCHMARK(SubIntSplit_Encode_##Pattern, iters) {           \
    Vector<uint64_t> data{benchmarkPool().get()};            \
    BENCHMARK_SUSPEND {                                      \
      data = DataExpr;                                       \
    }                                                        \
    encodeBenchmark<SubIntSplitEncoding<uint64_t>>(          \
        EncodingType::SubIntSplit, data, iters);             \
  }                                                          \
  BENCHMARK(SubIntSplit_Decode_##Pattern, iters) {           \
    std::string encoded;                                     \
    BENCHMARK_SUSPEND {                                      \
      auto data = DataExpr;                                  \
      encoded = encodeData<SubIntSplitEncoding<uint64_t>>(   \
          EncodingType::SubIntSplit, data);                  \
    }                                                        \
    decodeBenchmark<uint64_t>(encoded, kNumElements, iters); \
  }                                                          \
  BENCHMARK_DRAW_LINE()

SUBINT_BENCH(BitStructured16, makeBitStructured<uint64_t>(16));
SUBINT_BENCH(BitStructured32, makeBitStructured<uint64_t>(32));
SUBINT_BENCH(Random, makeRandom<uint64_t>());
SUBINT_BENCH(Narrow16bit, makeNarrow<uint64_t>(16));
SUBINT_BENCH(Constant, makeConstant<uint64_t>(0x1234567800000000ULL));
SUBINT_BENCH(MainlyConstant, makeMainlyConstant<uint64_t>(42));
SUBINT_BENCH(LowCardinality1024, makeLowCardinality<uint64_t>(1024));

#undef SUBINT_BENCH

int main() {
  facebook::velox::memory::MemoryManager::initialize({});
  folly::runBenchmarks();
}
