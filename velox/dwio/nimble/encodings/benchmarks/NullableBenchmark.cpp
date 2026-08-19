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
#include "velox/dwio/nimble/encodings/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"

using namespace facebook::nimble;
using namespace facebook::nimble::benchmarks;

namespace {

struct NullableData {
  Vector<uint32_t> values;
  Vector<bool> nulls;
};

NullableData makeNullable(Vector<uint32_t> values, uint32_t nonNullPct) {
  auto& pool = benchmarkPool();
  Vector<bool> nulls{pool.get()};
  nulls.resize(values.size());
  for (uint32_t i = 0; i < values.size(); ++i) {
    nulls[i] = (folly::Random::secureRand32() % 100 < nonNullPct);
  }
  return {std::move(values), std::move(nulls)};
}

std::string encodeNullable(const NullableData& nd) {
  using P = typename TypeTraits<uint32_t>::physicalType;
  auto& pool = benchmarkPool();
  Buffer buffer{*pool};
  auto values = std::span<const P>(
      reinterpret_cast<const P*>(nd.values.data()), nd.values.size());
  auto nulls = std::span<const bool>(nd.nulls.data(), nd.nulls.size());
  EncodingSelectionResult result{.encodingType = EncodingType::Nullable};
  auto selection = EncodingSelection<P>{
      std::move(result), Statistics<P>::create(values), nullptr};
  auto enc = NullableEncoding<uint32_t>::encodeNullable(
      selection, values, nulls, buffer);
  return std::string{enc.data(), enc.size()};
}

void encodeNullableBench(const NullableData& nd, uint32_t iters) {
  using P = typename TypeTraits<uint32_t>::physicalType;
  auto& pool = benchmarkPool();
  auto values = std::span<const P>(
      reinterpret_cast<const P*>(nd.values.data()), nd.values.size());
  auto nulls = std::span<const bool>(nd.nulls.data(), nd.nulls.size());
  while (iters--) {
    Buffer buffer{*pool};
    EncodingSelectionResult result{.encodingType = EncodingType::Nullable};
    auto selection = EncodingSelection<P>{
        std::move(result), Statistics<P>::create(values), nullptr};
    NullableEncoding<uint32_t>::encodeNullable(
        selection, values, nulls, buffer);
  }
}

} // namespace

#define NULLABLE_BENCH(Pattern, DataExpr, NullPct)                      \
  BENCHMARK(Nullable_Encode_##Pattern##_##NullPct##pctNonNull, iters) { \
    NullableData nd{                                                    \
        Vector<uint32_t>(benchmarkPool().get()),                        \
        Vector<bool>(benchmarkPool().get())};                           \
    BENCHMARK_SUSPEND {                                                 \
      nd = makeNullable(DataExpr, NullPct);                             \
    }                                                                   \
    encodeNullableBench(nd, iters);                                     \
  }                                                                     \
  BENCHMARK(Nullable_Decode_##Pattern##_##NullPct##pctNonNull, iters) { \
    std::string encoded;                                                \
    BENCHMARK_SUSPEND {                                                 \
      auto nd = makeNullable(DataExpr, NullPct);                        \
      encoded = encodeNullable(nd);                                     \
    }                                                                   \
    decodeBenchmark<uint32_t>(encoded, kNumElements, iters);            \
  }                                                                     \
  BENCHMARK_DRAW_LINE()

NULLABLE_BENCH(Random, makeRandom<uint32_t>(), 80);
NULLABLE_BENCH(Random, makeRandom<uint32_t>(), 50);
NULLABLE_BENCH(Random, makeRandom<uint32_t>(), 10);
NULLABLE_BENCH(Narrow8bit, makeNarrow<uint32_t>(8), 80);
NULLABLE_BENCH(Constant, makeConstant<uint32_t>(42), 80);
NULLABLE_BENCH(RunLength, makeRunLength<uint32_t>(), 80);
NULLABLE_BENCH(Increasing, makeIncreasing<uint32_t>(), 80);

#undef NULLABLE_BENCH

int main() {
  facebook::velox::memory::MemoryManager::initialize({});
  folly::runBenchmarks();
}
