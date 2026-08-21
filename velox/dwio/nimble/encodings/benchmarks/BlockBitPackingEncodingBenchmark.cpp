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
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"

using namespace facebook::nimble;
using namespace facebook::nimble::benchmarks;

namespace {

constexpr uint32_t kRowCount = kNumElements;

Vector<uint32_t> makeNarrowBlocks() {
  auto& pool = benchmarkPool();
  Vector<uint32_t> data{pool.get()};
  data.resize(kRowCount);
  for (uint32_t i = 0; i < kRowCount; ++i) {
    const auto block = i / kBlockBitPackingBlockSize;
    data[i] = block * 1000 + (i % 16);
  }
  return data;
}

#define BBP_BENCH(Pattern, DataExpr)                           \
  BENCHMARK(BBP_Encode_##Pattern, iters) {                     \
    Vector<uint32_t> data{benchmarkPool().get()};              \
    BENCHMARK_SUSPEND {                                        \
      data = DataExpr;                                         \
    }                                                          \
    encodeBenchmark<BlockBitPackingEncoding<uint32_t>>(        \
        EncodingType::BlockBitPacking, data, iters);           \
  }                                                            \
  BENCHMARK(BBP_Decode_##Pattern, iters) {                     \
    std::string encoded;                                       \
    BENCHMARK_SUSPEND {                                        \
      auto data = DataExpr;                                    \
      encoded = encodeData<BlockBitPackingEncoding<uint32_t>>( \
          EncodingType::BlockBitPacking, data);                \
    }                                                          \
    decodeBenchmark<uint32_t>(encoded, kRowCount, iters);      \
  }                                                            \
  BENCHMARK_DRAW_LINE()

BBP_BENCH(NarrowBlocks, makeNarrowBlocks());
BBP_BENCH(Narrow8bit, makeNarrow<uint32_t>(8, kRowCount));
BBP_BENCH(Uniform20bit, makeNarrow<uint32_t>(20, kRowCount));
BBP_BENCH(Constant, makeConstant<uint32_t>(42, kRowCount));
BBP_BENCH(Increasing, makeIncreasing<uint32_t>(kRowCount));

#undef BBP_BENCH

} // namespace

int main() {
  facebook::velox::memory::MemoryManager::initialize({});
  folly::runBenchmarks();
}
