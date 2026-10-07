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

#include "fmt/core.h"
#include "folly/Benchmark.h"
#include "folly/init/Init.h"
#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/ForEncoding.h"
#include "velox/dwio/nimble/encodings/FsstEncoding.h"
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/PFOREncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"

#include <iostream>
#include <limits>

using namespace facebook::nimble;
using namespace facebook::nimble::benchmarks;

namespace {

constexpr uint32_t kSliceOffset = 12345;
constexpr uint32_t kSliceLength = 4096;
constexpr uint32_t kPartialBlockSliceOffset = 12345;
constexpr uint32_t kPartialBlockSliceLength = 256;

void sliceBenchmark(
    const std::string& encoded,
    uint32_t offset,
    uint32_t length,
    uint32_t iters) {
  Buffer buffer{*benchmarkPool()};
  while (iters--) {
    buffer.reset();
    const auto sliced = EncodingFactory::slice(encoded, offset, length, buffer);
    folly::doNotOptimizeAway(sliced.data());
    folly::doNotOptimizeAway(sliced.size());
  }
}

void sliceBenchmark(const std::string& encoded, uint32_t iters) {
  sliceBenchmark(encoded, kSliceOffset, kSliceLength, iters);
}

template <typename EncodingT, typename T>
void materializeEncodeBenchmark(
    const std::string& encoded,
    EncodingType encodingType,
    uint32_t offset,
    uint32_t length,
    uint32_t iters) {
  Buffer stringBuffer{*benchmarkPool()};
  while (iters--) {
    stringBuffer.reset();
    auto encoding = EncodingFactory{}.create(
        *benchmarkPool(), encoded, [&](uint32_t size) -> void* {
          return stringBuffer.reserve(size);
        });
    encoding->skip(offset);

    Vector<T> values{benchmarkPool().get(), length};
    encoding->materialize(length, values.data());

    const auto materialized =
        encodeData<EncodingT>(encodingType, values, Encoding::Options{});
    folly::doNotOptimizeAway(materialized.data());
    folly::doNotOptimizeAway(materialized.size());
  }
}

template <typename EncodingT, typename T>
void materializeEncodeBenchmark(
    const std::string& encoded,
    EncodingType encodingType,
    uint32_t iters) {
  materializeEncodeBenchmark<EncodingT, T>(
      encoded, encodingType, kSliceOffset, kSliceLength, iters);
}

template <typename T>
void materializeBenchmark(const std::string& encoded, uint32_t iters) {
  Vector<T> values{benchmarkPool().get(), kSliceLength};
  while (iters--) {
    auto encoding =
        EncodingFactory{}.create(*benchmarkPool(), encoded, nullFactory());
    encoding->skip(kSliceOffset);
    encoding->materialize(kSliceLength, values.data());
    folly::doNotOptimizeAway(values.data());
  }
}

template <typename EncodingT, typename T>
void printSize(
    std::string_view name,
    EncodingType encodingType,
    const Vector<T>& data) {
  const auto encoded = encodeData<EncodingT>(encodingType, data);

  Buffer sliceBuffer{*benchmarkPool()};
  const auto sliced =
      EncodingFactory::slice(encoded, kSliceOffset, kSliceLength, sliceBuffer);

  Buffer stringBuffer{*benchmarkPool()};
  auto encoding = EncodingFactory{}.create(
      *benchmarkPool(), encoded, [&](uint32_t size) -> void* {
        return stringBuffer.reserve(size);
      });
  encoding->skip(kSliceOffset);

  Vector<T> values{benchmarkPool().get(), kSliceLength};
  encoding->materialize(kSliceLength, values.data());

  const auto materialized =
      encodeData<EncodingT>(encodingType, values, Encoding::Options{});
  const auto proportionalTarget =
      encoded.size() * static_cast<double>(kSliceLength) / data.size();

  std::cout << fmt::format(
      "{:<28} source={} target={:.0f} slice={} materialize_encode={}\n",
      name,
      encoded.size(),
      proportionalTarget,
      sliced.size(),
      materialized.size());
}

Vector<double> makeAlpDouble(uint32_t n = kNumElements) {
  auto& pool = benchmarkPool();
  Vector<double> data{pool.get()};
  data.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    data[i] = static_cast<double>(static_cast<int32_t>(i % 2048) - 1024) / 100;
  }
  return data;
}

Vector<uint32_t> makePforUint32(uint32_t n = kNumElements) {
  auto& pool = benchmarkPool();
  Vector<uint32_t> data{pool.get()};
  data.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    data[i] = i % 10 == 7 ? 100000 + i : 50 + (i % 64);
  }
  return data;
}

Vector<std::string_view> makeFsstStrings(uint32_t n = kNumElements) {
  static std::vector<std::string> storage;
  if (storage.size() != n) {
    storage.clear();
    storage.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
      storage.emplace_back(
          fmt::format(
              "common/prefix/path/{}/segment/{}?query=value", i % 2048, i));
    }
  }

  auto& pool = benchmarkPool();
  Vector<std::string_view> data{pool.get()};
  data.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    data[i] = storage[i];
  }
  return data;
}

Encoding::Options fsstOptions() {
  Encoding::Options options;
  options.fsstCompressionTargetRatio = std::numeric_limits<double>::max();
  return options;
}

void materializeEncodeFsstBenchmark(
    const std::string& encoded,
    uint32_t iters) {
  auto& pool = benchmarkPool();
  while (iters--) {
    std::vector<facebook::velox::BufferPtr> stringBuffers;
    auto stringBufferFactory = [&](uint32_t totalLength) {
      auto& buffer = stringBuffers.emplace_back(
          facebook::velox::AlignedBuffer::allocate<char>(
              totalLength, benchmarkPool().get()));
      return buffer->asMutable<void>();
    };
    auto encoding =
        EncodingFactory{}.create(*pool, encoded, stringBufferFactory);
    encoding->skip(kSliceOffset);

    Vector<std::string_view> values{pool.get(), kSliceLength};
    encoding->materialize(kSliceLength, values.data());

    const auto materialized =
        encodeData<FsstEncoding>(EncodingType::Fsst, values, fsstOptions());
    folly::doNotOptimizeAway(materialized.data());
    folly::doNotOptimizeAway(materialized.size());
  }
}

} // namespace

#define SLICE_BENCHMARKS(Name, EncodingT, ValueT, EncodingTypeValue, DataExpr) \
  BENCHMARK(Slice_##Name, iters) {                                             \
    std::string encoded;                                                       \
    BENCHMARK_SUSPEND {                                                        \
      const auto data = DataExpr;                                              \
      encoded = encodeData<EncodingT>(EncodingTypeValue, data);                \
    }                                                                          \
    sliceBenchmark(encoded, iters);                                            \
  }                                                                            \
  BENCHMARK_RELATIVE(MaterializeEncode_##Name, iters) {                        \
    std::string encoded;                                                       \
    BENCHMARK_SUSPEND {                                                        \
      const auto data = DataExpr;                                              \
      encoded = encodeData<EncodingT>(EncodingTypeValue, data);                \
    }                                                                          \
    materializeEncodeBenchmark<EncodingT, ValueT>(                             \
        encoded, EncodingTypeValue, iters);                                    \
  }                                                                            \
  BENCHMARK_DRAW_LINE()

#define SLICE_MATERIALIZE_BENCHMARKS(                           \
    Name, EncodingT, EncodingTypeValue, DataExpr)               \
  BENCHMARK(SliceCreate_##Name, iters) {                        \
    std::string encoded;                                        \
    BENCHMARK_SUSPEND {                                         \
      const auto data = DataExpr;                               \
      encoded = encodeData<EncodingT>(EncodingTypeValue, data); \
    }                                                           \
    sliceBenchmark(encoded, iters);                             \
  }                                                             \
  BENCHMARK_RELATIVE(MaterializeRange_##Name, iters) {          \
    std::string encoded;                                        \
    BENCHMARK_SUSPEND {                                         \
      const auto data = DataExpr;                               \
      encoded = encodeData<EncodingT>(EncodingTypeValue, data); \
    }                                                           \
    materializeBenchmark<uint32_t>(encoded, iters);             \
  }                                                             \
  BENCHMARK_DRAW_LINE()

SLICE_BENCHMARKS(
    ConstantUint32,
    ConstantEncoding<uint32_t>,
    uint32_t,
    EncodingType::Constant,
    makeConstant<uint32_t>(42));
SLICE_BENCHMARKS(
    TrivialUint32,
    TrivialEncoding<uint32_t>,
    uint32_t,
    EncodingType::Trivial,
    makeRandom<uint32_t>());
SLICE_BENCHMARKS(
    RLEUint32,
    RLEEncoding<uint32_t>,
    uint32_t,
    EncodingType::RLE,
    makeRunLength<uint32_t>());
SLICE_BENCHMARKS(
    DictionaryUint32,
    DictionaryEncoding<uint32_t>,
    uint32_t,
    EncodingType::Dictionary,
    makeLowCardinality<uint32_t>(1024));
SLICE_BENCHMARKS(
    FixedBitWidthUint32,
    FixedBitWidthEncoding<uint32_t>,
    uint32_t,
    EncodingType::FixedBitWidth,
    makeNarrow<uint32_t>(12));
SLICE_BENCHMARKS(
    BlockBitPackingUint32,
    BlockBitPackingEncoding<uint32_t>,
    uint32_t,
    EncodingType::BlockBitPacking,
    makeIncreasing<uint32_t>());
SLICE_BENCHMARKS(
    PFORUint32,
    PFOREncoding<uint32_t>,
    uint32_t,
    EncodingType::PFOR,
    makePforUint32());
SLICE_BENCHMARKS(
    SimdForBitpackUint32,
    SimdForBitpackEncoding<uint32_t>,
    uint32_t,
    EncodingType::SimdForBitpack,
    makeNarrow<uint32_t>(16));
SLICE_BENCHMARKS(
    HuffmanUint32,
    HuffmanEncoding<uint32_t>,
    uint32_t,
    EncodingType::Huffman,
    makeLowCardinality<uint32_t>(16));
SLICE_BENCHMARKS(
    FORUint32,
    ForEncoding<uint32_t>,
    uint32_t,
    EncodingType::FOR,
    makeIncreasing<uint32_t>());
SLICE_BENCHMARKS(
    ALPDouble,
    ALPEncoding<double>,
    double,
    EncodingType::ALP,
    makeAlpDouble());

BENCHMARK(Slice_FsstString, iters) {
  std::string encoded;
  BENCHMARK_SUSPEND {
    const auto data = makeFsstStrings();
    encoded = encodeData<FsstEncoding>(EncodingType::Fsst, data, fsstOptions());
  }
  sliceBenchmark(encoded, iters);
}
BENCHMARK_RELATIVE(MaterializeEncode_FsstString, iters) {
  std::string encoded;
  BENCHMARK_SUSPEND {
    const auto data = makeFsstStrings();
    encoded = encodeData<FsstEncoding>(EncodingType::Fsst, data, fsstOptions());
  }
  materializeEncodeFsstBenchmark(encoded, iters);
}
BENCHMARK_DRAW_LINE();

BENCHMARK(Slice_BlockBitPackingUint32PartialBlock, iters) {
  std::string encoded;
  BENCHMARK_SUSPEND {
    const auto data = makeIncreasing<uint32_t>();
    encoded = encodeData<BlockBitPackingEncoding<uint32_t>>(
        EncodingType::BlockBitPacking, data);
  }
  sliceBenchmark(
      encoded, kPartialBlockSliceOffset, kPartialBlockSliceLength, iters);
}
BENCHMARK_RELATIVE(MaterializeEncode_BlockBitPackingUint32PartialBlock, iters) {
  std::string encoded;
  BENCHMARK_SUSPEND {
    const auto data = makeIncreasing<uint32_t>();
    encoded = encodeData<BlockBitPackingEncoding<uint32_t>>(
        EncodingType::BlockBitPacking, data);
  }
  materializeEncodeBenchmark<BlockBitPackingEncoding<uint32_t>, uint32_t>(
      encoded,
      EncodingType::BlockBitPacking,
      kPartialBlockSliceOffset,
      kPartialBlockSliceLength,
      iters);
}
BENCHMARK_DRAW_LINE();

SLICE_MATERIALIZE_BENCHMARKS(
    TrivialUint32,
    TrivialEncoding<uint32_t>,
    EncodingType::Trivial,
    makeRandom<uint32_t>());
SLICE_MATERIALIZE_BENCHMARKS(
    RLEUint32,
    RLEEncoding<uint32_t>,
    EncodingType::RLE,
    makeRunLength<uint32_t>());

#undef SLICE_MATERIALIZE_BENCHMARKS
#undef SLICE_BENCHMARKS

int main(int argc, char** argv) {
  bool printSliceSizes{false};
  int write{1};
  for (int read = 1; read < argc; ++read) {
    if (std::string_view{argv[read]} == "--print_slice_sizes") {
      printSliceSizes = true;
      continue;
    }
    argv[write++] = argv[read];
  }
  argc = write;
  folly::Init init(&argc, &argv);
  facebook::velox::memory::MemoryManager::initialize({});
  if (printSliceSizes) {
    printSize<ConstantEncoding<uint32_t>>(
        "ConstantUint32", EncodingType::Constant, makeConstant<uint32_t>(42));
    printSize<TrivialEncoding<uint32_t>>(
        "TrivialUint32", EncodingType::Trivial, makeRandom<uint32_t>());
    printSize<RLEEncoding<uint32_t>>(
        "RLEUint32", EncodingType::RLE, makeRunLength<uint32_t>());
    printSize<DictionaryEncoding<uint32_t>>(
        "DictionaryUint32",
        EncodingType::Dictionary,
        makeLowCardinality<uint32_t>(1024));
    printSize<FixedBitWidthEncoding<uint32_t>>(
        "FixedBitWidthUint32",
        EncodingType::FixedBitWidth,
        makeNarrow<uint32_t>(12));
    printSize<BlockBitPackingEncoding<uint32_t>>(
        "BlockBitPackingUint32",
        EncodingType::BlockBitPacking,
        makeIncreasing<uint32_t>());
    printSize<PFOREncoding<uint32_t>>(
        "PFORUint32", EncodingType::PFOR, makePforUint32());
    printSize<SimdForBitpackEncoding<uint32_t>>(
        "SimdForBitpackUint32",
        EncodingType::SimdForBitpack,
        makeNarrow<uint32_t>(16));
    printSize<HuffmanEncoding<uint32_t>>(
        "HuffmanUint32",
        EncodingType::Huffman,
        makeLowCardinality<uint32_t>(16));
    printSize<ForEncoding<uint32_t>>(
        "FORUint32", EncodingType::FOR, makeIncreasing<uint32_t>());
    printSize<ALPEncoding<double>>(
        "ALPDouble", EncodingType::ALP, makeAlpDouble());
    printSize<FsstEncoding>(
        "FSSTString", EncodingType::Fsst, makeFsstStrings());
    return 0;
  }
  folly::runBenchmarks();
}
