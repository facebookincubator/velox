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
#include "velox/dwio/nimble/encodings/SparseBoolEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"

using namespace facebook::nimble;
using namespace facebook::nimble::benchmarks;

namespace {

std::string encodeBool(const Vector<bool>& data) {
  auto& pool = benchmarkPool();
  Buffer buffer{*pool};
  auto values = std::span<const bool>(data.data(), data.size());
  EncodingSelectionResult result{.encodingType = EncodingType::SparseBool};
  auto selection = EncodingSelection<bool>{
      std::move(result), Statistics<bool>::create(values), nullptr};
  auto enc = SparseBoolEncoding::encode(selection, values, buffer);
  return std::string{enc.data(), enc.size()};
}

void encodeBoolBenchmark(const Vector<bool>& data, uint32_t iters) {
  auto& pool = benchmarkPool();
  auto values = std::span<const bool>(data.data(), data.size());
  while (iters--) {
    Buffer buffer{*pool};
    EncodingSelectionResult result{.encodingType = EncodingType::SparseBool};
    auto selection = EncodingSelection<bool>{
        std::move(result), Statistics<bool>::create(values), nullptr};
    SparseBoolEncoding::encode(selection, values, buffer);
  }
}

void decodeBoolBenchmark(const std::string& encoded, uint32_t iters) {
  auto& pool = benchmarkPool();
  Vector<bool> output{pool.get()};
  output.resize(kNumElements);
  while (iters--) {
    auto encoding = EncodingFactory{}.create(*pool, encoded, nullFactory());
    encoding->materialize(kNumElements, output.data());
  }
}

} // namespace

BENCHMARK(SparseBool_Encode_Sparse5pct, iters) {
  Vector<bool> data{benchmarkPool().get()};
  BENCHMARK_SUSPEND {
    data = makeSparseBool();
  }
  encodeBoolBenchmark(data, iters);
}

BENCHMARK(SparseBool_Decode_Sparse5pct, iters) {
  std::string encoded;
  BENCHMARK_SUSPEND {
    auto data = makeSparseBool();
    encoded = encodeBool(data);
  }
  decodeBoolBenchmark(encoded, iters);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(SparseBool_Encode_Dense50pct, iters) {
  Vector<bool> data{benchmarkPool().get()};
  BENCHMARK_SUSPEND {
    data = makeDenseBool();
  }
  encodeBoolBenchmark(data, iters);
}

BENCHMARK(SparseBool_Decode_Dense50pct, iters) {
  std::string encoded;
  BENCHMARK_SUSPEND {
    auto data = makeDenseBool();
    encoded = encodeBool(data);
  }
  decodeBoolBenchmark(encoded, iters);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(SparseBool_Encode_AllFalse, iters) {
  Vector<bool> data{benchmarkPool().get()};
  BENCHMARK_SUSPEND {
    data.resize(kNumElements);
    for (uint32_t i = 0; i < kNumElements; ++i) {
      data[i] = false;
    }
  }
  encodeBoolBenchmark(data, iters);
}

BENCHMARK(SparseBool_Decode_AllFalse, iters) {
  std::string encoded;
  BENCHMARK_SUSPEND {
    Vector<bool> data{benchmarkPool().get()};
    data.resize(kNumElements);
    for (uint32_t i = 0; i < kNumElements; ++i) {
      data[i] = false;
    }
    encoded = encodeBool(data);
  }
  decodeBoolBenchmark(encoded, iters);
}

int main() {
  facebook::velox::memory::MemoryManager::initialize({});
  folly::runBenchmarks();
}
