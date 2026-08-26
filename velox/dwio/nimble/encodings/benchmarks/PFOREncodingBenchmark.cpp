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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include <folly/Benchmark.h>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"
#include "velox/dwio/nimble/encodings/benchmarks/PFOREncodingBenchmarkData.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

using namespace facebook::nimble;
using namespace facebook::nimble::benchmarks;

namespace {

Vector<uint32_t> makePforData(
    uint32_t rowCount = kNumElements,
    uint64_t seed = kPforBenchmarkDefaultSeed) {
  Vector<uint32_t> values{benchmarkPool().get()};
  values.reserve(rowCount);
  for (uint32_t row = 0; row < rowCount; ++row) {
    values.push_back(pforBenchmarkValue(row, seed));
  }
  return values;
}

EncodingSelectionPolicyCreator fallbackPolicyCreator() {
  return [](DataType dataType) {
    return ManualEncodingSelectionPolicyFactory{
        {{EncodingType::Trivial, 1.0}},
        /*compressionOptions=*/std::nullopt}
        .createPolicy(dataType);
  };
}

std::unique_ptr<EncodingSelectionPolicy<uint32_t>> pforPolicy() {
  EncodingLayout fixedBitWidth{
      EncodingType::FixedBitWidth, {}, CompressionType::Uncompressed};
  EncodingLayout pfor{
      EncodingType::PFOR,
      {},
      CompressionType::Uncompressed,
      {fixedBitWidth, fixedBitWidth}};
  return std::make_unique<ReplayedEncodingSelectionPolicy<uint32_t>>(
      std::move(pfor),
      /*compressionOptions=*/std::nullopt,
      fallbackPolicyCreator());
}

std::string_view encodePforToBuffer(
    const Vector<uint32_t>& values,
    Buffer& buffer) {
  return EncodingFactory::encode<uint32_t>(
      pforPolicy(),
      std::span<const uint32_t>{values.data(), values.size()},
      buffer);
}

std::string encodePforFixture(const Vector<uint32_t>& values) {
  Buffer buffer{*benchmarkPool()};
  return std::string{encodePforToBuffer(values, buffer)};
}

BENCHMARK(PFOR_Encode_Outliers10Pct, iterations) {
  Vector<uint32_t> values{benchmarkPool().get()};
  std::unique_ptr<Buffer> buffer;
  BENCHMARK_SUSPEND {
    values = makePforData();
    buffer = std::make_unique<Buffer>(*benchmarkPool());
  }
  while (iterations--) {
    buffer->reset();
    const auto encoded = encodePforToBuffer(values, *buffer);
    folly::doNotOptimizeAway(encoded);
  }
}

BENCHMARK(PFOR_DecodeDense_Outliers10Pct, iterations) {
  std::string encoded;
  std::unique_ptr<Encoding> encoding;
  Vector<uint32_t> output{benchmarkPool().get()};
  BENCHMARK_SUSPEND {
    encoded = encodePforFixture(makePforData());
    encoding = EncodingFactory{}.create(
        *benchmarkPool(), encoded, [](uint32_t) -> void* { return nullptr; });
    output.resize(kNumElements);
  }
  while (iterations--) {
    encoding->reset();
    encoding->materialize(kNumElements, output.data());
    folly::doNotOptimizeAway(output.back());
  }
}

BENCHMARK(PFOR_SkipSeek_Outliers10Pct, iterations) {
  std::string encoded;
  std::unique_ptr<Encoding> encoding;
  Vector<uint32_t> output{benchmarkPool().get()};
  BENCHMARK_SUSPEND {
    encoded = encodePforFixture(makePforData());
    encoding = EncodingFactory{}.create(
        *benchmarkPool(), encoded, [](uint32_t) -> void* { return nullptr; });
    output.resize(kNumElements);
  }
  while (iterations--) {
    encoding->reset();
    uint32_t cursor{0};
    uint32_t outputRows{0};
    while (cursor < kNumElements) {
      const uint32_t skip = std::min<uint32_t>(31, kNumElements - cursor);
      encoding->skip(skip);
      cursor += skip;
      if (cursor == kNumElements) {
        break;
      }
      const uint32_t read = std::min<uint32_t>(3, kNumElements - cursor);
      encoding->materialize(read, output.data() + outputRows);
      outputRows += read;
      cursor += read;
    }
    folly::doNotOptimizeAway(output[outputRows - 1]);
  }
}

} // namespace

int main() {
  facebook::velox::memory::MemoryManager::initialize({});
  folly::runBenchmarks();
}
