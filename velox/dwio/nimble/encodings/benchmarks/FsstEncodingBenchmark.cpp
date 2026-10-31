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
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <folly/Benchmark.h>
#include <folly/CppAttributes.h>
#include <folly/init/Init.h>

#include "velox/buffer/Buffer.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/FsstEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"
#include "velox/dwio/nimble/encodings/benchmarks/FsstBenchmarkData.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

using namespace facebook::nimble;
using namespace facebook::nimble::benchmarks;

namespace {

constexpr size_t kStringPageSize = 256 * 1024;
constexpr uint32_t kMaxStringPageBytes = 16 * 1024 * 1024;
constexpr uint64_t kMaxTotalStringPageBytes = 256 * 1024 * 1024;
constexpr size_t kMaxStringPageCount = 2048;

EncodingSelectionPolicyCreator fallbackPolicyCreator() {
  return [](DataType dataType) {
    return ManualEncodingSelectionPolicyFactory{
        {{EncodingType::Trivial, 1.0}},
        /*compressionOptions=*/std::nullopt}
        .createPolicy(dataType);
  };
}

std::unique_ptr<EncodingSelectionPolicy<std::string_view>> fsstPolicy() {
  std::vector<std::optional<const EncodingLayout>> children;
  children.emplace_back(
      EncodingLayout{
          EncodingType::FixedBitWidth, {}, CompressionType::Uncompressed});
  EncodingLayout layout{
      EncodingType::Fsst,
      {},
      CompressionType::Uncompressed,
      std::move(children)};
  return std::make_unique<ReplayedEncodingSelectionPolicy<std::string_view>>(
      std::move(layout),
      /*compressionOptions=*/std::nullopt,
      fallbackPolicyCreator());
}

Encoding::Options fsstOptions() {
  Encoding::Options options;
  options.fsstCompressionTargetRatio = std::numeric_limits<double>::max();
  return options;
}

std::string_view encodeFsstToBuffer(
    std::span<const std::string_view> values,
    Buffer& buffer) {
  return EncodingFactory::encode<std::string_view>(
      fsstPolicy(), values, buffer, fsstOptions());
}

std::string encodeFsstFixture(std::span<const std::string_view> values) {
  Buffer buffer{*benchmarkPool()};
  return std::string{encodeFsstToBuffer(values, buffer)};
}

class StringPageArena {
 public:
  void* allocate(uint32_t bytes) {
    if (bytes == 0) {
      throw std::runtime_error{"FSST requested an empty string page"};
    }
    if (bytes > kMaxStringPageBytes ||
        allocatedBytes_ > kMaxTotalStringPageBytes - bytes ||
        pages_.size() >= kMaxStringPageCount) {
      throw std::runtime_error{"FSST benchmark string page limit exceeded"};
    }
    allocatedBytes_ += bytes;
    auto& page = pages_.emplace_back(
        facebook::velox::AlignedBuffer::allocate<char>(
            bytes, benchmarkPool().get()));
    return page->asMutable<void>();
  }

  size_t pageCount() const {
    return pages_.size();
  }

  uint64_t allocatedBytes() const {
    return allocatedBytes_;
  }

 private:
  std::vector<facebook::velox::BufferPtr> pages_;
  uint64_t allocatedBytes_{0};
};

std::unique_ptr<Encoding> createFsstDecoder(
    std::string_view encoded,
    StringPageArena& stringPages) {
  auto stringBufferFactory = [&stringPages](uint32_t bytes) -> void* {
    return stringPages.allocate(bytes);
  };
  return EncodingFactory{}.create(
      *benchmarkPool(), encoded, std::move(stringBufferFactory));
}

uint32_t runFsstSkipSeek(
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

class FsstBenchmarkFixture {
 public:
  FsstBenchmarkFixture()
      : corpus_{makeFsstBenchmarkCorpus()},
        encoded_{encodeFsstFixture(corpus_.values)},
        decoder_{createFsstDecoder(encoded_, stringPages_)},
        output_{benchmarkPool().get(), corpus_.values.size()} {
    validateMetadata();
    decodeDense();
    validateDenseResult();
    const uint32_t outputRows = skipSeek();
    validateSkipResult(outputRows);
    if (stringPages_.pageCount() < 3) {
      throw std::runtime_error{
          "FSST benchmark decode must span at least three string pages"};
    }
    decoder_->reset();
    stablePageCount_ = stringPages_.pageCount();
    stablePageBytes_ = stringPages_.allocatedBytes();
  }

  const StringBenchmarkCorpus& corpus() const {
    return corpus_;
  }

  std::string_view encoded() const {
    return encoded_;
  }

  void decodeDense() {
    decoder_->reset();
    decoder_->materialize(rowCount(), output_.data());
  }

  uint32_t skipSeek() {
    return runFsstSkipSeek(*decoder_, rowCount(), output_.data());
  }

  std::string_view lastOutput() const {
    return output_.back();
  }

  std::string_view skipOutput(uint32_t outputRows) const {
    return output_[outputRows - 1];
  }

  void validateDenseResult() const {
    if (!std::equal(output_.begin(), output_.end(), corpus_.values.begin())) {
      throw std::runtime_error{"FSST benchmark dense decode mismatch"};
    }
  }

  void validateSkipResult(uint32_t outputRows) const {
    uint32_t cursor{0};
    uint32_t outputRow{0};
    while (cursor < rowCount()) {
      cursor += std::min<uint32_t>(31, rowCount() - cursor);
      if (cursor == rowCount()) {
        break;
      }
      const uint32_t read = std::min<uint32_t>(3, rowCount() - cursor);
      if (!std::equal(
              output_.begin() + outputRow,
              output_.begin() + outputRow + read,
              corpus_.values.begin() + cursor)) {
        throw std::runtime_error{"FSST benchmark skip trace mismatch"};
      }
      outputRow += read;
      cursor += read;
    }
    if (outputRow != outputRows) {
      throw std::runtime_error{"FSST benchmark skip trace row mismatch"};
    }
  }

  void validateStablePages() const {
    if (stringPages_.pageCount() != stablePageCount_ ||
        stringPages_.allocatedBytes() != stablePageBytes_) {
      throw std::runtime_error{
          "FSST benchmark decode allocated string pages during timing"};
    }
  }

 private:
  uint32_t rowCount() const {
    return static_cast<uint32_t>(corpus_.values.size());
  }

  void validateMetadata() const {
    if (corpus_.rawBytes <= 2 * kStringPageSize) {
      throw std::runtime_error{
          "FSST benchmark corpus must span multiple string pages"};
    }
    if (encoded_.empty() || encoded_.size() >= corpus_.rawBytes) {
      throw std::runtime_error{
          "FSST benchmark full artifact did not compress the corpus"};
    }
    if (decoder_->encodingType() != EncodingType::Fsst ||
        decoder_->dataType() != DataType::String ||
        decoder_->rowCount() != corpus_.values.size()) {
      throw std::runtime_error{"FSST benchmark fixture metadata mismatch"};
    }
  }

  StringBenchmarkCorpus corpus_;
  std::string encoded_;
  StringPageArena stringPages_;
  std::unique_ptr<Encoding> decoder_;
  Vector<std::string_view> output_;
  size_t stablePageCount_{0};
  uint64_t stablePageBytes_{0};
};

BENCHMARK(FSST_Encode_String_SortedStructuredTextMixedLengths, iterations) {
  std::unique_ptr<FsstBenchmarkFixture> fixture;
  std::unique_ptr<Buffer> buffer;
  std::string_view timedArtifact;
  BENCHMARK_SUSPEND {
    fixture = std::make_unique<FsstBenchmarkFixture>();
    buffer = std::make_unique<Buffer>(*benchmarkPool());
    timedArtifact = fixture->encoded();
  }
  while (iterations--) {
    buffer->reset();
    timedArtifact = encodeFsstToBuffer(fixture->corpus().values, *buffer);
    folly::doNotOptimizeAway(timedArtifact);
  }
  BENCHMARK_SUSPEND {
    if (timedArtifact != fixture->encoded()) {
      throw std::runtime_error{"FSST benchmark timed encode mismatch"};
    }
  }
}

void benchmarkEncodeBuffers(
    uint32_t iterations,
    uint32_t rowCount,
    bool useBufferPool,
    bool fallback) {
  folly::BenchmarkSuspender suspender;
  const auto corpus = makeFsstBenchmarkCorpus(rowCount);
  Buffer buffer{*benchmarkPool()};
  EncodingBufferPool bufferPool{benchmarkPool().get()};
  auto options = fsstOptions();
  if (fallback) {
    options.fsstCompressionTargetRatio = 0;
  }
  const auto encode = [&] {
    buffer.reset();
    return EncodingFactory::encode<std::string_view>(
        fsstPolicy(), corpus.values, buffer, options);
  };
  const std::string expected{encode()};
  if (useBufferPool) {
    options.encodingBufferPool = &bufferPool;
  }
  // Exclude the initial output and scratch allocations from the steady-state
  // measurement. The baseline uses the same setup without a scratch pool.
  auto encoded = encode();
  suspender.dismiss();
  while (iterations--) {
    encoded = encode();
    folly::doNotOptimizeAway(encoded);
  }
  suspender.rehire();
  if (encoded != expected) {
    throw std::runtime_error{"FSST buffer benchmark timed encode mismatch"};
  }
}

BENCHMARK(FSST_Encode_SmallChunk, iterations) {
  benchmarkEncodeBuffers(iterations, 64, false, false);
}

BENCHMARK_RELATIVE(FSST_Encode_SmallChunk_Pooled, iterations) {
  benchmarkEncodeBuffers(iterations, 64, true, false);
}

BENCHMARK(FSST_Encode_LargeChunk, iterations) {
  benchmarkEncodeBuffers(
      iterations, kFsstBenchmarkDefaultRowCount, false, false);
}

BENCHMARK_RELATIVE(FSST_Encode_LargeChunk_Pooled, iterations) {
  benchmarkEncodeBuffers(
      iterations, kFsstBenchmarkDefaultRowCount, true, false);
}

BENCHMARK(FSST_Encode_TrivialFallback, iterations) {
  benchmarkEncodeBuffers(
      iterations, kFsstBenchmarkDefaultRowCount, false, true);
}

BENCHMARK_RELATIVE(FSST_Encode_TrivialFallback_Pooled, iterations) {
  benchmarkEncodeBuffers(iterations, kFsstBenchmarkDefaultRowCount, true, true);
}

BENCHMARK(
    FSST_CreateDestroy_String_SortedStructuredTextMixedLengths,
    iterations) {
  std::optional<FsstBenchmarkFixture> fixture;
  std::optional<StringPageArena> stringPages;
  BENCHMARK_SUSPEND {
    fixture.emplace();
    stringPages.emplace();
  }
  // Time the complete decoder lifecycle. Pausing and resuming the benchmark
  // around every destruction would distort this sub-microsecond operation.
  while (iterations--) {
    auto decoder = createFsstDecoder(fixture->encoded(), *stringPages);
    folly::doNotOptimizeAway(decoder.get());
  }
}

BENCHMARK_DRAW_LINE();

BENCHMARK(
    FSST_DecodeDense_String_SortedStructuredTextMixedLengths,
    iterations) {
  std::unique_ptr<FsstBenchmarkFixture> fixture;
  BENCHMARK_SUSPEND {
    fixture = std::make_unique<FsstBenchmarkFixture>();
  }
  while (iterations--) {
    fixture->decodeDense();
    folly::doNotOptimizeAway(fixture->lastOutput());
  }
  BENCHMARK_SUSPEND {
    fixture->validateDenseResult();
    fixture->validateStablePages();
  }
}

BENCHMARK(FSST_SkipSeek_String_SortedStructuredTextMixedLengths, iterations) {
  std::unique_ptr<FsstBenchmarkFixture> fixture;
  uint32_t outputRows{0};
  BENCHMARK_SUSPEND {
    fixture = std::make_unique<FsstBenchmarkFixture>();
  }
  while (iterations--) {
    outputRows = fixture->skipSeek();
    folly::doNotOptimizeAway(fixture->skipOutput(outputRows));
  }
  BENCHMARK_SUSPEND {
    fixture->validateSkipResult(outputRows);
    fixture->validateStablePages();
  }
}

// Fast and decoded controls share an artifact, selection and range. The
// control deliberately uses a different FilterType to retain the previous
// decompress-then-filter path while applying the same BytesRange predicate.
class RangeBenchmarkReader {
 public:
  const facebook::velox::BufferPtr& nullsInReadRange() const {
    return nulls_;
  }
  const uint64_t* FOLLY_NULLABLE rawNullsInReadRange() const {
    return nullptr;
  }
  bool returnReaderNulls() const {
    return false;
  }

 private:
  facebook::velox::BufferPtr nulls_;
};

template <bool compressed>
class RangeBenchmarkVisitor {
 public:
  using DataType = std::string_view;
  using FilterType =
      std::conditional_t<compressed, facebook::velox::common::BytesRange, void>;
  using Extract = facebook::velox::dwio::common::DropValues;

  RangeBenchmarkVisitor(
      const std::vector<facebook::velox::vector_size_t>& rows,
      const facebook::velox::common::BytesRange& filter,
      std::vector<facebook::velox::vector_size_t>& passingRows)
      : rows_{rows}, filter_{filter}, passingRows_{passingRows} {
    passingRows_.clear();
  }
  RangeBenchmarkReader& reader() {
    return reader_;
  }
  const facebook::velox::common::BytesRange& filter() const {
    return filter_;
  }
  auto numRows() const {
    return rows_.size();
  }
  auto rowAt(size_t index) const {
    return rows_[index];
  }
  auto currentRow() const {
    return rows_[rowIndex_];
  }
  bool allowNulls() const {
    return false;
  }
  void filterPassed(std::string_view) {
    passingRows_.push_back(currentRow());
  }
  void filterFailed() {}
  void addRowIndex(size_t count) {
    rowIndex_ += count;
  }
  void addNumValues(size_t) {}
  bool atEnd() const {
    return rowIndex_ == rows_.size();
  }
  void process(std::string_view value, bool& atEnd) {
    if (filter_.testBytes(value.data(), static_cast<int32_t>(value.size()))) {
      filterPassed(value);
    }
    addRowIndex(1);
    atEnd = this->atEnd();
  }
  [[noreturn]] void processNull(bool&) {
    throw std::runtime_error{"Unexpected null in range benchmark"};
  }

 private:
  const std::vector<facebook::velox::vector_size_t>& rows_;
  const facebook::velox::common::BytesRange& filter_;
  std::vector<facebook::velox::vector_size_t>& passingRows_;
  RangeBenchmarkReader reader_;
  size_t rowIndex_{0};
};

template <bool compressed>
void benchmarkRange(
    uint32_t iterations,
    uint32_t stride,
    int selectivity,
    bool shortValues) {
  constexpr uint32_t kRows = 12'000;
  std::vector<std::string> storage;
  std::vector<std::string_view> values;
  std::vector<facebook::velox::vector_size_t> rows;
  std::vector<facebook::velox::vector_size_t> expected;
  std::vector<facebook::velox::vector_size_t> passing;
  std::string encoded;
  StringPageArena pages;
  std::unique_ptr<FsstEncoding> decoder;
  const std::string prefix =
      shortValues ? "key/" : "https://www.example.com/api/v2/";
  const std::string lower = selectivity == 0 ? prefix + "missing/"
      : selectivity == 100                   ? prefix
                                             : prefix + "match/";
  facebook::velox::common::BytesRange filter{
      lower, false, false, lower + "\xff", false, true, false};
  BENCHMARK_SUSPEND {
    storage.reserve(kRows);
    values.reserve(kRows);
    rows.reserve(kRows);
    passing.reserve(kRows);
    for (uint32_t row = 0; row < kRows; ++row) {
      storage.push_back(
          shortValues
              ? fmt::format(
                    "{}{}{}", prefix, row % 10 == 0 ? "match/" : "other/", row)
              : fmt::format(
                    "{}{}{}?payload={}",
                    prefix,
                    row % 10 == 0 ? "match/" : "other/",
                    row,
                    std::string(
                        128 + row % 64, static_cast<char>('a' + row % 26))));
    }
    for (uint32_t row = 0; row < kRows; ++row) {
      values.emplace_back(storage.at(row));
      if (row % stride == 0) {
        rows.push_back(static_cast<vector_size_t>(row));
        if (filter.testBytes(
                values.at(row).data(),
                static_cast<int32_t>(values.at(row).size()))) {
          expected.push_back(static_cast<vector_size_t>(row));
        }
      }
    }
    encoded = encodeFsstFixture(values);
    decoder = std::make_unique<FsstEncoding>(
        *benchmarkPool(), encoded, [&](uint32_t bytes) {
          return pages.allocate(bytes);
        });
    if (decoder->encodingType() != EncodingType::Fsst ||
        decoder->rowCount() != kRows) {
      throw std::runtime_error{"Range benchmark must use FSST"};
    }
    RangeBenchmarkVisitor<compressed> visitor{rows, filter, passing};
    ReadWithVisitorParams params{};
    decoder->readWithVisitor(visitor, params);
    if (passing != expected || (compressed && pages.pageCount() != 0)) {
      throw std::runtime_error{"FSST range benchmark validation failed"};
    }
  }
  while (iterations--) {
    decoder->reset();
    RangeBenchmarkVisitor<compressed> visitor{rows, filter, passing};
    ReadWithVisitorParams params{};
    decoder->readWithVisitor(visitor, params);
    folly::doNotOptimizeAway(passing.size());
  }
  BENCHMARK_SUSPEND {
    if (passing != expected || (compressed && pages.pageCount() != 0)) {
      throw std::runtime_error{"FSST range benchmark result mismatch"};
    }
  }
}

BENCHMARK(FSST_Range_Dense_NoHit_Decoded, iters) {
  benchmarkRange<false>(iters, 1, 0, false);
}
BENCHMARK_RELATIVE(FSST_Range_Dense_NoHit_Compressed, iters) {
  benchmarkRange<true>(iters, 1, 0, false);
}
BENCHMARK(FSST_Range_Dense_TenPercent_Decoded, iters) {
  benchmarkRange<false>(iters, 1, 10, false);
}
BENCHMARK_RELATIVE(FSST_Range_Dense_TenPercent_Compressed, iters) {
  benchmarkRange<true>(iters, 1, 10, false);
}
BENCHMARK(FSST_Range_Dense_All_Decoded, iters) {
  benchmarkRange<false>(iters, 1, 100, false);
}
BENCHMARK_RELATIVE(FSST_Range_Dense_All_Compressed, iters) {
  benchmarkRange<true>(iters, 1, 100, false);
}
BENCHMARK(FSST_Range_Sparse_NoHit_Decoded, iters) {
  benchmarkRange<false>(iters, 31, 0, false);
}
BENCHMARK_RELATIVE(FSST_Range_Sparse_NoHit_Compressed, iters) {
  benchmarkRange<true>(iters, 31, 0, false);
}
BENCHMARK(FSST_Range_Sparse_TenPercent_Decoded, iters) {
  benchmarkRange<false>(iters, 31, 10, false);
}
BENCHMARK_RELATIVE(FSST_Range_Sparse_TenPercent_Compressed, iters) {
  benchmarkRange<true>(iters, 31, 10, false);
}
BENCHMARK(FSST_Range_Sparse_All_Decoded, iters) {
  benchmarkRange<false>(iters, 31, 100, false);
}
BENCHMARK_RELATIVE(FSST_Range_Sparse_All_Compressed, iters) {
  benchmarkRange<true>(iters, 31, 100, false);
}

void printArtifactRatio() {
  // Keep shared setup encode-only. Decode benchmarks validate their fixtures.
  const auto corpus = makeFsstBenchmarkCorpus();
  const auto encoded = encodeFsstFixture(corpus.values);
  fmt::print(
      "FSST full artifact: raw={} encoded={} encoded_to_raw={:.6f} "
      "raw_to_encoded={:.6f}\n",
      corpus.rawBytes,
      encoded.size(),
      static_cast<double>(encoded.size()) / corpus.rawBytes,
      static_cast<double>(corpus.rawBytes) / encoded.size());
}

BENCHMARK(FSST_Range_Short_Dense_NoHit_Decoded, iterations) {
  benchmarkRange<false>(iterations, 1, 0, true);
}
BENCHMARK_RELATIVE(FSST_Range_Short_Dense_NoHit_Compressed, iterations) {
  benchmarkRange<true>(iterations, 1, 0, true);
}
BENCHMARK(FSST_Range_Short_Dense_TenPercent_Decoded, iterations) {
  benchmarkRange<false>(iterations, 1, 10, true);
}
BENCHMARK_RELATIVE(FSST_Range_Short_Dense_TenPercent_Compressed, iterations) {
  benchmarkRange<true>(iterations, 1, 10, true);
}
BENCHMARK(FSST_Range_Short_Dense_All_Decoded, iterations) {
  benchmarkRange<false>(iterations, 1, 100, true);
}
BENCHMARK_RELATIVE(FSST_Range_Short_Dense_All_Compressed, iterations) {
  benchmarkRange<true>(iterations, 1, 100, true);
}
BENCHMARK(FSST_Range_Short_Sparse_TenPercent_Decoded, iterations) {
  benchmarkRange<false>(iterations, 31, 10, true);
}
BENCHMARK_RELATIVE(FSST_Range_Short_Sparse_TenPercent_Compressed, iterations) {
  benchmarkRange<true>(iterations, 31, 10, true);
}

} // namespace

int main(int argc, char** argv) {
  const folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize({});
  printArtifactRatio();
  folly::runBenchmarks();
}
