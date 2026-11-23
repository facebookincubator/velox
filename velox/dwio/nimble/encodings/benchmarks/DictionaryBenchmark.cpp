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

#include "folly/Benchmark.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"

using namespace facebook::nimble;
using namespace facebook::nimble::benchmarks;

#define DICT_BENCH(Pattern, DataExpr)                        \
  BENCHMARK(Dict_Encode_##Pattern, iters) {                  \
    Vector<uint32_t> data{benchmarkPool().get()};            \
    BENCHMARK_SUSPEND {                                      \
      data = DataExpr;                                       \
    }                                                        \
    encodeBenchmark<DictionaryEncoding<uint32_t>>(           \
        EncodingType::Dictionary, data, iters);              \
  }                                                          \
  BENCHMARK(Dict_Decode_##Pattern, iters) {                  \
    std::string encoded;                                     \
    BENCHMARK_SUSPEND {                                      \
      auto data = DataExpr;                                  \
      encoded = encodeData<DictionaryEncoding<uint32_t>>(    \
          EncodingType::Dictionary, data);                   \
    }                                                        \
    decodeBenchmark<uint32_t>(encoded, kNumElements, iters); \
  }                                                          \
  BENCHMARK_DRAW_LINE()

DICT_BENCH(Random, makeRandom<uint32_t>());
DICT_BENCH(Narrow8bit, makeNarrow<uint32_t>(8));
DICT_BENCH(Constant, makeConstant<uint32_t>(42));
DICT_BENCH(MainlyConstant, makeMainlyConstant<uint32_t>(42));
DICT_BENCH(RunLength, makeRunLength<uint32_t>());
DICT_BENCH(Increasing, makeIncreasing<uint32_t>());
DICT_BENCH(LowCardinality64, makeLowCardinality<uint32_t>(64));
DICT_BENCH(LowCardinality1024, makeLowCardinality<uint32_t>(1024));

#undef DICT_BENCH

int main() {
  facebook::velox::memory::MemoryManager::initialize({});
  folly::runBenchmarks();
}
