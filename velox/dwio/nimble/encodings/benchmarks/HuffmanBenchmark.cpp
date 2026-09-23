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
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"

using namespace facebook::nimble;
using namespace facebook::nimble::benchmarks;

namespace {

Vector<uint32_t> makeSkewedLowCardinality() {
  Vector<uint32_t> data{benchmarkPool().get()};
  data.resize(kNumElements);
  for (uint32_t i = 0; i < data.size(); ++i) {
    if (i % 100 < 90) {
      data[i] = 0;
    } else if (i % 100 < 96) {
      data[i] = 1;
    } else {
      data[i] = 2 + i % 14;
    }
  }
  return data;
}

Vector<uint32_t> makeUniformMaxCardinality() {
  Vector<uint32_t> data{benchmarkPool().get()};
  data.resize(kNumElements);
  for (uint32_t i = 0; i < data.size(); ++i) {
    data[i] = i % HuffmanEncoding<uint32_t>::kMaxSymbols;
  }
  return data;
}

#define HUFFMAN_BENCH(Pattern, DataExpr)                                      \
  BENCHMARK(Huffman_Encode_##Pattern, iters) {                                \
    Vector<uint32_t> data{benchmarkPool().get()};                             \
    BENCHMARK_SUSPEND {                                                       \
      data = DataExpr;                                                        \
    }                                                                         \
    encodeBenchmark<HuffmanEncoding<uint32_t>>(                               \
        EncodingType::Huffman, data, iters);                                  \
  }                                                                           \
  BENCHMARK(Huffman_Decode_##Pattern, iters) {                                \
    std::string encoded;                                                      \
    BENCHMARK_SUSPEND {                                                       \
      auto data = DataExpr;                                                   \
      encoded =                                                               \
          encodeData<HuffmanEncoding<uint32_t>>(EncodingType::Huffman, data); \
    }                                                                         \
    decodeBenchmark<uint32_t>(encoded, kNumElements, iters);                  \
  }                                                                           \
  BENCHMARK_DRAW_LINE()

HUFFMAN_BENCH(SkewedLowCardinality, makeSkewedLowCardinality());
HUFFMAN_BENCH(UniformMaxCardinality, makeUniformMaxCardinality());

#undef HUFFMAN_BENCH

} // namespace

int main() {
  facebook::velox::memory::MemoryManager::initialize({});
  folly::runBenchmarks();
}
