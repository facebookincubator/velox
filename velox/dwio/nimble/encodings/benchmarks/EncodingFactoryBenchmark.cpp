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

// Measures EncodingFactory::create plus destruction of the small encodings a
// stream decoder builds per segment: one per stream, plus one per nested
// child. "withOptions" is the four-argument create that StreamData and nested
// encodings call; "fromFactory" is the three-argument create reading the
// factory's own options.

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"

namespace facebook::nimble::benchmarks {
namespace {

// One row, as most streams of a small payload hold.
const std::string& constantUint8() {
  static const std::string data = [] {
    Vector<uint8_t> values{benchmarkPool().get()};
    values.push_back(7);
    return encodeData<ConstantEncoding<uint8_t>>(
        EncodingType::Constant, values);
  }();
  return data;
}

const std::string& trivialInt64() {
  static const std::string data = [] {
    Vector<int64_t> values{benchmarkPool().get()};
    for (int64_t i = 0; i < 8; ++i) {
      values.push_back(i * 1'000'003);
    }
    return encodeData<TrivialEncoding<int64_t>>(EncodingType::Trivial, values);
  }();
  return data;
}

// A parent with two nested children (alphabet and indices).
const std::string& dictionaryUint32() {
  static const std::string data = [] {
    Vector<uint32_t> values{benchmarkPool().get()};
    for (uint32_t i = 0; i < 16; ++i) {
      values.push_back(i % 3);
    }
    return encodeData<DictionaryEncoding<uint32_t>>(
        EncodingType::Dictionary, values);
  }();
  return data;
}

enum class CreateCall { kWithOptions, kFromFactory };

void createAndDestroy(
    uint32_t iters,
    const std::string& data,
    CreateCall call) {
  const Encoding::Options options{};
  const auto noStringBuffers = [](uint32_t) -> void* { return nullptr; };
  while (iters--) {
    auto encoding = call == CreateCall::kWithOptions
        ? EncodingFactory().create(
              *benchmarkPool(), data, noStringBuffers, options)
        : EncodingFactory(options).create(
              *benchmarkPool(), data, noStringBuffers);
    folly::doNotOptimizeAway(encoding.get());
  }
}

} // namespace

BENCHMARK(constantUint8WithOptions, iters) {
  createAndDestroy(iters, constantUint8(), CreateCall::kWithOptions);
}

BENCHMARK(constantUint8FromFactory, iters) {
  createAndDestroy(iters, constantUint8(), CreateCall::kFromFactory);
}

BENCHMARK(trivialInt64WithOptions, iters) {
  createAndDestroy(iters, trivialInt64(), CreateCall::kWithOptions);
}

BENCHMARK(dictionaryUint32WithOptions, iters) {
  createAndDestroy(iters, dictionaryUint32(), CreateCall::kWithOptions);
}

} // namespace facebook::nimble::benchmarks

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize({});
  folly::runBenchmarks();
  return 0;
}
