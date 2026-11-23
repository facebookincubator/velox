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

#include <stdexcept>
#include "folly/Benchmark.h"
#include "folly/init/Init.h"
#include "velox/common/base/BitUtil.h"
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

NullableData makeMaterializeNullable(uint32_t nonNullPct) {
  auto& pool = benchmarkPool();
  Vector<uint32_t> values{pool.get()};
  Vector<bool> nulls{pool.get()};
  values.resize(kNumElements);
  nulls.resize(kNumElements);
  for (uint32_t i = 0; i < kNumElements; ++i) {
    values[i] = i * 2'654'435'761U;
    nulls[i] = (i * 37U) % 100 < nonNullPct;
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
      std::move(result),
      Statistics<P>::create(values),
      makeDefaultPolicy(TypeTraits<uint32_t>::dataType),
  };
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
        std::move(result),
        Statistics<P>::create(values),
        makeDefaultPolicy(TypeTraits<uint32_t>::dataType),
    };
    NullableEncoding<uint32_t>::encodeNullable(
        selection, values, nulls, buffer);
  }
}

void materializeNullableBench(
    const std::string& encoded,
    const facebook::velox::bits::Bitmap* scatterOutputBitmap,
    uint32_t iters) {
  auto& pool = benchmarkPool();
  std::vector<uint32_t> output(kNumElements);
  std::vector<uint64_t> outputNulls(
      facebook::velox::bits::nwords(kNumElements));
  auto encodingOwner = EncodingFactory{}.create(*pool, encoded, nullFactory());
  auto* encoding = encodingOwner.get();
  if (encoding == nullptr) {
    throw std::runtime_error("EncodingFactory returned null.");
  }
  while (iters--) {
    encoding->reset();
    const auto nonNullCount = encoding->materializeNullable(
        kNumElements,
        output.data(),
        [&outputNulls]() -> void* { return outputNulls.data(); },
        scatterOutputBitmap);
    folly::doNotOptimizeAway(nonNullCount);
    folly::doNotOptimizeAway(output.data());
    folly::doNotOptimizeAway(outputNulls.data());
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

#define NULLABLE_MATERIALIZE_BENCH(NonNullPct)                              \
  BENCHMARK(                                                                \
      Nullable_Materialize_ByteReference_##NonNullPct##pctNonNull, iters) { \
    std::string encoded;                                                    \
    std::vector<uint64_t> identityScatterBits;                              \
    BENCHMARK_SUSPEND {                                                     \
      encoded = encodeNullable(makeMaterializeNullable(NonNullPct));        \
      identityScatterBits.assign(                                           \
          facebook::velox::bits::nwords(kNumElements), ~uint64_t{0});       \
    }                                                                       \
    const facebook::velox::bits::Bitmap identityScatterBitmap{              \
        identityScatterBits.data(), kNumElements};                          \
    materializeNullableBench(encoded, &identityScatterBitmap, iters);       \
  }                                                                         \
  BENCHMARK_RELATIVE(                                                       \
      Nullable_Materialize_Packed_##NonNullPct##pctNonNull, iters) {        \
    std::string encoded;                                                    \
    BENCHMARK_SUSPEND {                                                     \
      encoded = encodeNullable(makeMaterializeNullable(NonNullPct));        \
    }                                                                       \
    materializeNullableBench(encoded, nullptr, iters);                      \
  }                                                                         \
  BENCHMARK_DRAW_LINE()

// An identity scatter bitmap selects the retained byte-per-row path without
// changing output positions, which makes it a reference for contiguous reads.
NULLABLE_MATERIALIZE_BENCH(1);
NULLABLE_MATERIALIZE_BENCH(10);
NULLABLE_MATERIALIZE_BENCH(50);
NULLABLE_MATERIALIZE_BENCH(100);

#undef NULLABLE_MATERIALIZE_BENCH

int main(int argc, char** argv) {
  const folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize({});
  folly::runBenchmarks();
}
