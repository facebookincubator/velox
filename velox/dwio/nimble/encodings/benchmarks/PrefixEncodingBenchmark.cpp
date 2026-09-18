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
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/buffer/Buffer.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/PrefixEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"
#include "velox/dwio/nimble/encodings/benchmarks/PrefixBenchmarkData.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

using namespace facebook::nimble;
using namespace facebook::nimble::benchmarks;

namespace {

constexpr size_t kStringPageSize = 256 * 1024;

EncodingSelectionPolicyCreator fallbackPolicyCreator() {
  return [](DataType dataType) {
    return ManualEncodingSelectionPolicyFactory{
        {{EncodingType::Trivial, 1.0}},
        /*compressionOptions=*/std::nullopt}
        .createPolicy(dataType);
  };
}

std::unique_ptr<EncodingSelectionPolicy<std::string_view>> prefixPolicy() {
  EncodingLayout layout{
      EncodingType::Prefix, {}, CompressionType::Uncompressed};
  return std::make_unique<ReplayedEncodingSelectionPolicy<std::string_view>>(
      std::move(layout),
      /*compressionOptions=*/std::nullopt,
      fallbackPolicyCreator());
}

std::string_view encodePrefixToBuffer(
    std::span<const std::string_view> values,
    Buffer& buffer) {
  return EncodingFactory::encode<std::string_view>(
      prefixPolicy(), values, buffer);
}

std::string encodePrefixFixture(std::span<const std::string_view> values) {
  Buffer buffer{*benchmarkPool()};
  return std::string{encodePrefixToBuffer(values, buffer)};
}

std::unique_ptr<Encoding> createPrefixDecoder(
    std::string_view encoded,
    std::vector<facebook::velox::BufferPtr>& stringPages) {
  auto stringBufferFactory = [&stringPages](uint32_t bytes) -> void* {
    auto& page = stringPages.emplace_back(
        facebook::velox::AlignedBuffer::allocate<char>(
            bytes, benchmarkPool().get()));
    return page->asMutable<void>();
  };
  return EncodingFactory{}.create(
      *benchmarkPool(), encoded, std::move(stringBufferFactory));
}

void validatePrefixFixture(
    const StringBenchmarkCorpus& corpus,
    std::string_view encoded,
    Encoding& encoding,
    Vector<std::string_view>& output) {
  if (corpus.rawBytes <= 2 * kStringPageSize) {
    throw std::runtime_error{
        "Prefix benchmark corpus must span multiple string pages"};
  }
  if (encoded.empty() || encoded.size() >= corpus.rawBytes) {
    throw std::runtime_error{
        "Prefix benchmark fixture did not compress its logical strings"};
  }
  if (encoding.encodingType() != EncodingType::Prefix ||
      encoding.rowCount() != corpus.values.size()) {
    throw std::runtime_error{"Prefix benchmark fixture metadata mismatch"};
  }

  encoding.reset();
  encoding.materialize(corpus.values.size(), output.data());
  if (!std::equal(output.begin(), output.end(), corpus.values.begin())) {
    throw std::runtime_error{"Prefix benchmark fixture failed round trip"};
  }
  encoding.reset();
}

uint32_t runPrefixSkipSeek(
    Encoding& encoding,
    uint32_t rowCount,
    std::string_view* output) {
  encoding.reset();
  uint32_t cursor{0};
  uint32_t outputRows{0};
  while (cursor < rowCount) {
    const uint32_t skip = std::min<uint32_t>(31, rowCount - cursor);
    encoding.skip(skip);
    cursor += skip;
    if (cursor == rowCount) {
      break;
    }
    const uint32_t read = std::min<uint32_t>(3, rowCount - cursor);
    encoding.materialize(read, output + outputRows);
    outputRows += read;
    cursor += read;
  }
  return outputRows;
}

void validatePrefixSkipSeek(
    Encoding& encoding,
    const StringBenchmarkCorpus& corpus,
    Vector<std::string_view>& output) {
  const uint32_t outputRows =
      runPrefixSkipSeek(encoding, corpus.values.size(), output.data());
  uint32_t cursor{0};
  uint32_t outputRow{0};
  while (cursor < corpus.values.size()) {
    cursor += std::min<uint32_t>(31, corpus.values.size() - cursor);
    if (cursor == corpus.values.size()) {
      break;
    }
    const uint32_t read = std::min<uint32_t>(3, corpus.values.size() - cursor);
    if (!std::equal(
            output.begin() + outputRow,
            output.begin() + outputRow + read,
            corpus.values.begin() + cursor)) {
      throw std::runtime_error{"Prefix benchmark skip trace mismatch"};
    }
    outputRow += read;
    cursor += read;
  }
  if (outputRow != outputRows) {
    throw std::runtime_error{"Prefix benchmark skip trace row mismatch"};
  }
  encoding.reset();
}

BENCHMARK(Prefix_Encode_String_SortedPathPrefixMixedLengths, iterations) {
  StringBenchmarkCorpus corpus;
  std::unique_ptr<Buffer> buffer;
  BENCHMARK_SUSPEND {
    corpus = makePrefixBenchmarkCorpus();
    buffer = std::make_unique<Buffer>(*benchmarkPool());
    const auto encoded = encodePrefixToBuffer(corpus.values, *buffer);
    std::vector<facebook::velox::BufferPtr> stringPages;
    auto encoding = createPrefixDecoder(encoded, stringPages);
    Vector<std::string_view> output{
        benchmarkPool().get(), corpus.values.size()};
    validatePrefixFixture(corpus, encoded, *encoding, output);
  }
  while (iterations--) {
    buffer->reset();
    const auto encoded = encodePrefixToBuffer(corpus.values, *buffer);
    folly::doNotOptimizeAway(encoded);
  }
}

BENCHMARK(Prefix_DecodeDense_String_SortedPathPrefixMixedLengths, iterations) {
  StringBenchmarkCorpus corpus;
  std::string encoded;
  std::vector<facebook::velox::BufferPtr> stringPages;
  std::unique_ptr<Encoding> encoding;
  Vector<std::string_view> output{benchmarkPool().get()};
  size_t stablePageCount{0};
  BENCHMARK_SUSPEND {
    corpus = makePrefixBenchmarkCorpus();
    encoded = encodePrefixFixture(corpus.values);
    encoding = createPrefixDecoder(encoded, stringPages);
    output.resize(corpus.values.size());
    validatePrefixFixture(corpus, encoded, *encoding, output);
    stablePageCount = stringPages.size();
  }
  while (iterations--) {
    encoding->reset();
    encoding->materialize(corpus.values.size(), output.data());
    folly::doNotOptimizeAway(output.back());
  }
  BENCHMARK_SUSPEND {
    if (stringPages.size() != stablePageCount) {
      throw std::runtime_error{"Prefix dense decode allocated during timing"};
    }
  }
}

BENCHMARK(Prefix_SkipSeek_String_SortedPathPrefixMixedLengths, iterations) {
  StringBenchmarkCorpus corpus;
  std::string encoded;
  std::vector<facebook::velox::BufferPtr> stringPages;
  std::unique_ptr<Encoding> encoding;
  Vector<std::string_view> output{benchmarkPool().get()};
  BENCHMARK_SUSPEND {
    corpus = makePrefixBenchmarkCorpus();
    encoded = encodePrefixFixture(corpus.values);
    encoding = createPrefixDecoder(encoded, stringPages);
    output.resize(corpus.values.size());
    validatePrefixFixture(corpus, encoded, *encoding, output);
    validatePrefixSkipSeek(*encoding, corpus, output);
  }
  while (iterations--) {
    const uint32_t outputRows =
        runPrefixSkipSeek(*encoding, corpus.values.size(), output.data());
    folly::doNotOptimizeAway(output[outputRows - 1]);
  }
}

} // namespace

int main(int argc, char** argv) {
  const folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize({});
  folly::runBenchmarks();
}
