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
    std::string_view* FOLLY_NONNULL output) {
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
  folly::BenchmarkSuspender suspender;
  FsstBenchmarkFixture fixture;
  Buffer buffer{*benchmarkPool()};
  std::string_view timedArtifact = fixture.encoded();
  suspender.dismiss();
  while (iterations--) {
    buffer.reset();
    timedArtifact = encodeFsstToBuffer(fixture.corpus().values, buffer);
    folly::doNotOptimizeAway(timedArtifact);
  }
  suspender.rehire();
  if (timedArtifact != fixture.encoded()) {
    throw std::runtime_error{"FSST benchmark timed encode mismatch"};
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
  folly::BenchmarkSuspender suspender;
  FsstBenchmarkFixture fixture;
  suspender.dismiss();
  while (iterations--) {
    fixture.decodeDense();
    folly::doNotOptimizeAway(fixture.lastOutput());
  }
  suspender.rehire();
  fixture.validateDenseResult();
  fixture.validateStablePages();
}

BENCHMARK(FSST_SkipSeek_String_SortedStructuredTextMixedLengths, iterations) {
  folly::BenchmarkSuspender suspender;
  FsstBenchmarkFixture fixture;
  uint32_t outputRows{0};
  suspender.dismiss();
  while (iterations--) {
    outputRows = fixture.skipSeek();
    folly::doNotOptimizeAway(fixture.skipOutput(outputRows));
  }
  suspender.rehire();
  fixture.validateSkipResult(outputRows);
  fixture.validateStablePages();
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

template <
    bool compressed,
    typename Filter = facebook::velox::common::BytesRange>
class RangeBenchmarkVisitor {
 public:
  using DataType = std::string_view;
  using FilterType = std::conditional_t<compressed, Filter, void>;
  using Extract = facebook::velox::dwio::common::DropValues;

  RangeBenchmarkVisitor(
      const std::vector<facebook::velox::vector_size_t>& rows,
      const Filter& filter,
      std::vector<facebook::velox::vector_size_t>& passingRows)
      : rows_{rows}, filter_{filter}, passingRows_{passingRows} {
    passingRows_.clear();
  }
  RangeBenchmarkReader& reader() {
    return reader_;
  }
  const Filter& filter() const {
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
  const Filter& filter_;
  std::vector<facebook::velox::vector_size_t>& passingRows_;
  RangeBenchmarkReader reader_;
  size_t rowIndex_{0};
};

// Configures the corpus shared by decoded and compressed range benchmarks.
struct RangeBenchmarkParams {
  // Distance between selected input rows.
  uint32_t stride;
  // Whether to omit the long payload suffix from corpus values.
  bool shortValues;
  // Whether matching values all equal the lower bound.
  bool singleValue;
  // Percentage of matching values in a single-value corpus.
  int selectivity;
  // Common prefix for generated corpus values.
  std::string_view prefix;
  // Value used for matches in a single-value corpus.
  std::string_view lower;
  // Predicate used to compute the expected passing rows.
  const facebook::velox::common::Filter& filter;
  std::optional<size_t> mismatchOffset{};
};

// Shares untimed corpus setup across the decoded and compressed filter
// variants.
class RangeBenchmarkFixture {
 public:
  // Builds one encoded workload for decoded and compressed filtering.
  explicit RangeBenchmarkFixture(const RangeBenchmarkParams& params) {
    constexpr uint32_t kRows = 12'000;
    std::vector<std::string> storage;
    std::vector<std::string_view> values;
    storage.reserve(kRows);
    values.reserve(kRows);
    rows_.reserve(kRows);
    passingRows_.reserve(kRows);
    for (uint32_t row = 0; row < kRows; ++row) {
      storage.push_back(
          params.shortValues
              ? fmt::format(
                    "{}{}{}",
                    params.prefix,
                    row % 10 == 0 ? "match/" : "other/",
                    row)
              : fmt::format(
                    "{}{}{}?payload={}",
                    params.prefix,
                    row % 10 == 0 ? "match/" : "other/",
                    row,
                    std::string(
                        128 + row % 64, static_cast<char>('a' + row % 26))));
    }
    if (params.singleValue) {
      for (uint32_t row = 0; row < kRows; ++row) {
        auto& value = storage.at(row);
        if (params.selectivity == 10
                ? row % 10 == 0
                : static_cast<int>(row % 100) < params.selectivity) {
          value = params.lower;
        } else {
          if (params.mismatchOffset) {
            value = params.lower;
            value.at(*params.mismatchOffset) = 'y';
          } else {
            value = std::string(params.prefix) + "other/";
            if (!params.shortValues) {
              value.append(160, 'x');
            }
          }
        }
      }
    }
    for (uint32_t row = 0; row < kRows; ++row) {
      values.emplace_back(storage.at(row));
      if (row % params.stride == 0) {
        rows_.push_back(static_cast<vector_size_t>(row));
        if (params.filter.testBytes(
                values.at(row).data(),
                static_cast<int32_t>(values.at(row).size()))) {
          expectedRows_.push_back(static_cast<vector_size_t>(row));
        }
      }
    }
    encoded_ = encodeFsstFixture(values);
    decoder_ = std::make_unique<FsstEncoding>(
        *benchmarkPool(), encoded_, [&](uint32_t bytes) {
          return pages_.allocate(bytes);
        });
    if (decoder_->encodingType() != EncodingType::Fsst ||
        decoder_->rowCount() != kRows) {
      throw std::runtime_error{"Range benchmark must use FSST"};
    }
  }

  const std::vector<vector_size_t>& rows() const {
    return rows_;
  }

  std::vector<vector_size_t>& passingRows() {
    return passingRows_;
  }

  FsstEncoding& decoder() {
    return *decoder_;
  }

  // Checks matching rows and verifies the expected string page usage.
  void validate(bool expectNoPages) const {
    if (passingRows_ != expectedRows_ ||
        (expectNoPages && pages_.pageCount() != 0)) {
      throw std::runtime_error{"FSST range benchmark result mismatch"};
    }
  }

 private:
  // Selects logical rows visited by both benchmark variants.
  std::vector<vector_size_t> rows_;
  // Stores matching row numbers computed before timing.
  std::vector<vector_size_t> expectedRows_;
  // Collects matching row numbers during each measured read.
  std::vector<vector_size_t> passingRows_;
  // Owns the serialized FSST input shared by both variants.
  std::string encoded_;
  // Keeps decoded strings alive during filter evaluation.
  StringPageArena pages_;
  // Reuses the same FSST symbol table and buffers between iterations.
  std::unique_ptr<FsstEncoding> decoder_;
};

// Measures filtering with fixture setup excluded from timing.
template <bool compressed, bool negated>
void benchmarkRange(
    uint32_t iterations,
    uint32_t stride,
    int selectivity,
    bool shortValues,
    bool singleValue) {
  folly::BenchmarkSuspender suspender;
  const std::string prefix =
      shortValues ? "key/" : "https://www.example.com/api/v2/";
  const std::string lower = (selectivity == 0         ? prefix + "missing/"
                                 : selectivity == 100 ? prefix
                                                      : prefix + "match/") +
      (singleValue && !shortValues ? std::string(160, 'x') : "");
  using Filter = std::conditional_t<
      negated,
      facebook::velox::common::NegatedBytesRange,
      facebook::velox::common::BytesRange>;
  Filter filter{
      lower,
      false,
      false,
      singleValue ? lower : lower + "\xff",
      false,
      !singleValue,
      false,
  };
  RangeBenchmarkFixture fixture{RangeBenchmarkParams{
      .stride = stride,
      .shortValues = shortValues,
      .singleValue = singleValue,
      .selectivity = selectivity,
      .prefix = prefix,
      .lower = lower,
      .filter = filter,
  }};
  {
    RangeBenchmarkVisitor<compressed, Filter> visitor{
        fixture.rows(), filter, fixture.passingRows()};
    ReadWithVisitorParams params{};
    fixture.decoder().readWithVisitor(visitor, params);
    fixture.validate(compressed);
  }
  suspender.dismiss();
  while (iterations--) {
    fixture.decoder().reset();
    RangeBenchmarkVisitor<compressed, Filter> visitor{
        fixture.rows(), filter, fixture.passingRows()};
    ReadWithVisitorParams params{};
    fixture.decoder().readWithVisitor(visitor, params);
    folly::doNotOptimizeAway(fixture.passingRows().size());
  }
  suspender.rehire();
  fixture.validate(compressed);
}

BENCHMARK(FSST_Range_Dense_NoHit_Decoded, iters) {
  benchmarkRange<false, false>(iters, 1, 0, false, false);
}
BENCHMARK_RELATIVE(FSST_Range_Dense_NoHit_Compressed, iters) {
  benchmarkRange<true, false>(iters, 1, 0, false, false);
}
BENCHMARK(FSST_Range_Dense_TenPercent_Decoded, iters) {
  benchmarkRange<false, false>(iters, 1, 10, false, false);
}
BENCHMARK_RELATIVE(FSST_Range_Dense_TenPercent_Compressed, iters) {
  benchmarkRange<true, false>(iters, 1, 10, false, false);
}
BENCHMARK(FSST_Range_Dense_All_Decoded, iters) {
  benchmarkRange<false, false>(iters, 1, 100, false, false);
}
BENCHMARK_RELATIVE(FSST_Range_Dense_All_Compressed, iters) {
  benchmarkRange<true, false>(iters, 1, 100, false, false);
}
BENCHMARK(FSST_Range_Sparse_NoHit_Decoded, iters) {
  benchmarkRange<false, false>(iters, 31, 0, false, false);
}
BENCHMARK_RELATIVE(FSST_Range_Sparse_NoHit_Compressed, iters) {
  benchmarkRange<true, false>(iters, 31, 0, false, false);
}
BENCHMARK(FSST_Range_Sparse_TenPercent_Decoded, iters) {
  benchmarkRange<false, false>(iters, 31, 10, false, false);
}
BENCHMARK_RELATIVE(FSST_Range_Sparse_TenPercent_Compressed, iters) {
  benchmarkRange<true, false>(iters, 31, 10, false, false);
}
BENCHMARK(FSST_Range_Sparse_All_Decoded, iters) {
  benchmarkRange<false, false>(iters, 31, 100, false, false);
}
BENCHMARK_RELATIVE(FSST_Range_Sparse_All_Compressed, iters) {
  benchmarkRange<true, false>(iters, 31, 100, false, false);
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
      static_cast<double>(encoded.size()) /
          static_cast<double>(corpus.rawBytes),
      static_cast<double>(corpus.rawBytes) /
          static_cast<double>(encoded.size()));
}

BENCHMARK(FSST_Range_Short_Dense_NoHit_Decoded, iterations) {
  benchmarkRange<false, false>(iterations, 1, 0, true, false);
}
BENCHMARK_RELATIVE(FSST_Range_Short_Dense_NoHit_Compressed, iterations) {
  benchmarkRange<true, false>(iterations, 1, 0, true, false);
}
BENCHMARK(FSST_Range_Short_Dense_TenPercent_Decoded, iterations) {
  benchmarkRange<false, false>(iterations, 1, 10, true, false);
}
BENCHMARK_RELATIVE(FSST_Range_Short_Dense_TenPercent_Compressed, iterations) {
  benchmarkRange<true, false>(iterations, 1, 10, true, false);
}
BENCHMARK(FSST_Range_Short_Dense_All_Decoded, iterations) {
  benchmarkRange<false, false>(iterations, 1, 100, true, false);
}
BENCHMARK_RELATIVE(FSST_Range_Short_Dense_All_Compressed, iterations) {
  benchmarkRange<true, false>(iterations, 1, 100, true, false);
}
BENCHMARK(FSST_Range_Short_Sparse_TenPercent_Decoded, iterations) {
  benchmarkRange<false, false>(iterations, 31, 10, true, false);
}
BENCHMARK_RELATIVE(FSST_Range_Short_Sparse_TenPercent_Compressed, iterations) {
  benchmarkRange<true, false>(iterations, 31, 10, true, false);
}

BENCHMARK(FSST_NegatedRange_Dense_TenPercent_Decoded, iterations) {
  benchmarkRange<false, true>(iterations, 1, 10, false, false);
}
BENCHMARK_RELATIVE(FSST_NegatedRange_Dense_TenPercent_Compressed, iterations) {
  benchmarkRange<true, true>(iterations, 1, 10, false, false);
}
BENCHMARK(FSST_NegatedRange_Sparse_TenPercent_Decoded, iterations) {
  benchmarkRange<false, true>(iterations, 31, 10, false, false);
}
BENCHMARK_RELATIVE(FSST_NegatedRange_Sparse_TenPercent_Compressed, iterations) {
  benchmarkRange<true, true>(iterations, 31, 10, false, false);
}
BENCHMARK(FSST_NegatedRange_Short_Dense_TenPercent_Decoded, iterations) {
  benchmarkRange<false, true>(iterations, 1, 10, true, false);
}
BENCHMARK_RELATIVE(
    FSST_NegatedRange_Short_Dense_TenPercent_Compressed,
    iterations) {
  benchmarkRange<true, true>(iterations, 1, 10, true, false);
}
BENCHMARK(FSST_NegatedRange_Short_Sparse_TenPercent_Decoded, iterations) {
  benchmarkRange<false, true>(iterations, 31, 10, true, false);
}
BENCHMARK_RELATIVE(
    FSST_NegatedRange_Short_Sparse_TenPercent_Compressed,
    iterations) {
  benchmarkRange<true, true>(iterations, 31, 10, true, false);
}
BENCHMARK(FSST_Equals_Dense_TenPercent_Decoded, iterations) {
  benchmarkRange<false, false>(iterations, 1, 10, false, true);
}
BENCHMARK_RELATIVE(FSST_Equals_Dense_TenPercent_Compressed, iterations) {
  benchmarkRange<true, false>(iterations, 1, 10, false, true);
}
BENCHMARK(FSST_Equals_Sparse_TenPercent_Decoded, iterations) {
  benchmarkRange<false, false>(iterations, 31, 10, false, true);
}
BENCHMARK_RELATIVE(FSST_Equals_Sparse_TenPercent_Compressed, iterations) {
  benchmarkRange<true, false>(iterations, 31, 10, false, true);
}
BENCHMARK(FSST_Equals_Dense_All_Decoded, iterations) {
  benchmarkRange<false, false>(iterations, 1, 100, false, true);
}
BENCHMARK_RELATIVE(FSST_Equals_Dense_All_Compressed, iterations) {
  benchmarkRange<true, false>(iterations, 1, 100, false, true);
}
BENCHMARK(FSST_Equals_Sparse_All_Decoded, iterations) {
  benchmarkRange<false, false>(iterations, 31, 100, false, true);
}
BENCHMARK_RELATIVE(FSST_Equals_Sparse_All_Compressed, iterations) {
  benchmarkRange<true, false>(iterations, 31, 100, false, true);
}
BENCHMARK(FSST_NotEquals_Dense_TenPercent_Decoded, iterations) {
  benchmarkRange<false, true>(iterations, 1, 10, false, true);
}
BENCHMARK_RELATIVE(FSST_NotEquals_Dense_TenPercent_Compressed, iterations) {
  benchmarkRange<true, true>(iterations, 1, 10, false, true);
}
BENCHMARK(FSST_NotEquals_Sparse_TenPercent_Decoded, iterations) {
  benchmarkRange<false, true>(iterations, 31, 10, false, true);
}
BENCHMARK_RELATIVE(FSST_NotEquals_Sparse_TenPercent_Compressed, iterations) {
  benchmarkRange<true, true>(iterations, 31, 10, false, true);
}
BENCHMARK(FSST_NotEquals_Short_Dense_TenPercent_Decoded, iterations) {
  benchmarkRange<false, true>(iterations, 1, 10, true, true);
}
BENCHMARK_RELATIVE(
    FSST_NotEquals_Short_Dense_TenPercent_Compressed,
    iterations) {
  benchmarkRange<true, true>(iterations, 1, 10, true, true);
}
BENCHMARK(FSST_NotEquals_Short_Sparse_TenPercent_Decoded, iterations) {
  benchmarkRange<false, true>(iterations, 31, 10, true, true);
}
BENCHMARK_RELATIVE(
    FSST_NotEquals_Short_Sparse_TenPercent_Compressed,
    iterations) {
  benchmarkRange<true, true>(iterations, 31, 10, true, true);
}

BENCHMARK(FSST_NotEquals_Dense_All_Decoded, iterations) {
  benchmarkRange<false, true>(iterations, 1, 100, false, true);
}
BENCHMARK_RELATIVE(FSST_NotEquals_Dense_All_Compressed, iterations) {
  benchmarkRange<true, true>(iterations, 1, 100, false, true);
}
BENCHMARK(FSST_NotEquals_Sparse_All_Decoded, iterations) {
  benchmarkRange<false, true>(iterations, 31, 100, false, true);
}
BENCHMARK_RELATIVE(FSST_NotEquals_Sparse_All_Compressed, iterations) {
  benchmarkRange<true, true>(iterations, 31, 100, false, true);
}

template <bool compressed, bool negated>
void benchmarkEqualityShape(
    uint32_t iterations,
    size_t valueBytes,
    size_t mismatchOffset,
    int equalPercent,
    uint32_t stride = 1) {
  folly::BenchmarkSuspender suspender;
  const std::string bound(valueBytes, 'x');
  using Filter = std::conditional_t<
      negated,
      facebook::velox::common::NegatedBytesRange,
      facebook::velox::common::BytesRange>;
  Filter filter{bound, false, false, bound, false, false, false};
  RangeBenchmarkFixture fixture{RangeBenchmarkParams{
      .stride = stride,
      .shortValues = false,
      .singleValue = true,
      .selectivity = equalPercent,
      .prefix = "",
      .lower = bound,
      .filter = filter,
      .mismatchOffset = mismatchOffset,
  }};
  {
    RangeBenchmarkVisitor<compressed, Filter> visitor{
        fixture.rows(), filter, fixture.passingRows()};
    ReadWithVisitorParams params{};
    fixture.decoder().readWithVisitor(visitor, params);
    fixture.validate(compressed);
  }
  suspender.dismiss();
  while (iterations--) {
    fixture.decoder().reset();
    RangeBenchmarkVisitor<compressed, Filter> visitor{
        fixture.rows(), filter, fixture.passingRows()};
    ReadWithVisitorParams params{};
    fixture.decoder().readWithVisitor(visitor, params);
    folly::doNotOptimizeAway(fixture.passingRows().size());
  }
  suspender.rehire();
  fixture.validate(compressed);
}

BENCHMARK(FSST_EqualityShape_B8_Early_Eq_Decoded, iterations) {
  benchmarkEqualityShape<false, false>(iterations, 8, 0, 0, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B8_Early_Eq_Compressed, iterations) {
  benchmarkEqualityShape<true, false>(iterations, 8, 0, 0, 1);
}

BENCHMARK(FSST_EqualityShape_B8_Early_Ne_Decoded, iterations) {
  benchmarkEqualityShape<false, true>(iterations, 8, 0, 0, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B8_Early_Ne_Compressed, iterations) {
  benchmarkEqualityShape<true, true>(iterations, 8, 0, 0, 1);
}

BENCHMARK(FSST_EqualityShape_B8_Late_Eq_Decoded, iterations) {
  benchmarkEqualityShape<false, false>(iterations, 8, 7, 0, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B8_Late_Eq_Compressed, iterations) {
  benchmarkEqualityShape<true, false>(iterations, 8, 7, 0, 1);
}

BENCHMARK(FSST_EqualityShape_B8_Late_Ne_Decoded, iterations) {
  benchmarkEqualityShape<false, true>(iterations, 8, 7, 0, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B8_Late_Ne_Compressed, iterations) {
  benchmarkEqualityShape<true, true>(iterations, 8, 7, 0, 1);
}

BENCHMARK(FSST_EqualityShape_B8_All_Eq_Decoded, iterations) {
  benchmarkEqualityShape<false, false>(iterations, 8, 0, 100, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B8_All_Eq_Compressed, iterations) {
  benchmarkEqualityShape<true, false>(iterations, 8, 0, 100, 1);
}

BENCHMARK(FSST_EqualityShape_B8_All_Ne_Decoded, iterations) {
  benchmarkEqualityShape<false, true>(iterations, 8, 0, 100, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B8_All_Ne_Compressed, iterations) {
  benchmarkEqualityShape<true, true>(iterations, 8, 0, 100, 1);
}

BENCHMARK(FSST_EqualityShape_B32_Early_Eq_Decoded, iterations) {
  benchmarkEqualityShape<false, false>(iterations, 32, 0, 0, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B32_Early_Eq_Compressed, iterations) {
  benchmarkEqualityShape<true, false>(iterations, 32, 0, 0, 1);
}

BENCHMARK(FSST_EqualityShape_B32_Early_Ne_Decoded, iterations) {
  benchmarkEqualityShape<false, true>(iterations, 32, 0, 0, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B32_Early_Ne_Compressed, iterations) {
  benchmarkEqualityShape<true, true>(iterations, 32, 0, 0, 1);
}

BENCHMARK(FSST_EqualityShape_B32_Late_Eq_Decoded, iterations) {
  benchmarkEqualityShape<false, false>(iterations, 32, 31, 0, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B32_Late_Eq_Compressed, iterations) {
  benchmarkEqualityShape<true, false>(iterations, 32, 31, 0, 1);
}

BENCHMARK(FSST_EqualityShape_B32_Late_Ne_Decoded, iterations) {
  benchmarkEqualityShape<false, true>(iterations, 32, 31, 0, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B32_Late_Ne_Compressed, iterations) {
  benchmarkEqualityShape<true, true>(iterations, 32, 31, 0, 1);
}

BENCHMARK(FSST_EqualityShape_B32_All_Eq_Decoded, iterations) {
  benchmarkEqualityShape<false, false>(iterations, 32, 0, 100, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B32_All_Eq_Compressed, iterations) {
  benchmarkEqualityShape<true, false>(iterations, 32, 0, 100, 1);
}

BENCHMARK(FSST_EqualityShape_B32_All_Ne_Decoded, iterations) {
  benchmarkEqualityShape<false, true>(iterations, 32, 0, 100, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B32_All_Ne_Compressed, iterations) {
  benchmarkEqualityShape<true, true>(iterations, 32, 0, 100, 1);
}

BENCHMARK(FSST_EqualityShape_B192_Early_Eq_Decoded, iterations) {
  benchmarkEqualityShape<false, false>(iterations, 192, 0, 0, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B192_Early_Eq_Compressed, iterations) {
  benchmarkEqualityShape<true, false>(iterations, 192, 0, 0, 1);
}

BENCHMARK(FSST_EqualityShape_B192_Early_Ne_Decoded, iterations) {
  benchmarkEqualityShape<false, true>(iterations, 192, 0, 0, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B192_Early_Ne_Compressed, iterations) {
  benchmarkEqualityShape<true, true>(iterations, 192, 0, 0, 1);
}

BENCHMARK(FSST_EqualityShape_B192_Late_Eq_Decoded, iterations) {
  benchmarkEqualityShape<false, false>(iterations, 192, 191, 0, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B192_Late_Eq_Compressed, iterations) {
  benchmarkEqualityShape<true, false>(iterations, 192, 191, 0, 1);
}

BENCHMARK(FSST_EqualityShape_B192_Late_Ne_Decoded, iterations) {
  benchmarkEqualityShape<false, true>(iterations, 192, 191, 0, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B192_Late_Ne_Compressed, iterations) {
  benchmarkEqualityShape<true, true>(iterations, 192, 191, 0, 1);
}

BENCHMARK(FSST_EqualityShape_B192_All_Eq_Decoded, iterations) {
  benchmarkEqualityShape<false, false>(iterations, 192, 0, 100, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B192_All_Eq_Compressed, iterations) {
  benchmarkEqualityShape<true, false>(iterations, 192, 0, 100, 1);
}

BENCHMARK(FSST_EqualityShape_B192_All_Ne_Decoded, iterations) {
  benchmarkEqualityShape<false, true>(iterations, 192, 0, 100, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B192_All_Ne_Compressed, iterations) {
  benchmarkEqualityShape<true, true>(iterations, 192, 0, 100, 1);
}

BENCHMARK(FSST_EqualityShape_B1024_Early_Eq_Decoded, iterations) {
  benchmarkEqualityShape<false, false>(iterations, 1'024, 0, 0, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B1024_Early_Eq_Compressed, iterations) {
  benchmarkEqualityShape<true, false>(iterations, 1'024, 0, 0, 1);
}

BENCHMARK(FSST_EqualityShape_B1024_Early_Ne_Decoded, iterations) {
  benchmarkEqualityShape<false, true>(iterations, 1'024, 0, 0, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B1024_Early_Ne_Compressed, iterations) {
  benchmarkEqualityShape<true, true>(iterations, 1'024, 0, 0, 1);
}

BENCHMARK(FSST_EqualityShape_B1024_Late_Eq_Decoded, iterations) {
  benchmarkEqualityShape<false, false>(iterations, 1'024, 1'023, 0, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B1024_Late_Eq_Compressed, iterations) {
  benchmarkEqualityShape<true, false>(iterations, 1'024, 1'023, 0, 1);
}

BENCHMARK(FSST_EqualityShape_B1024_Late_Ne_Decoded, iterations) {
  benchmarkEqualityShape<false, true>(iterations, 1'024, 1'023, 0, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B1024_Late_Ne_Compressed, iterations) {
  benchmarkEqualityShape<true, true>(iterations, 1'024, 1'023, 0, 1);
}

BENCHMARK(FSST_EqualityShape_B1024_All_Eq_Decoded, iterations) {
  benchmarkEqualityShape<false, false>(iterations, 1'024, 0, 100, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B1024_All_Eq_Compressed, iterations) {
  benchmarkEqualityShape<true, false>(iterations, 1'024, 0, 100, 1);
}

BENCHMARK(FSST_EqualityShape_B1024_All_Ne_Decoded, iterations) {
  benchmarkEqualityShape<false, true>(iterations, 1'024, 0, 100, 1);
}
BENCHMARK_RELATIVE(FSST_EqualityShape_B1024_All_Ne_Compressed, iterations) {
  benchmarkEqualityShape<true, true>(iterations, 1'024, 0, 100, 1);
}

BENCHMARK(FSST_EqualityShape_B192_MixedEarly_Eq_Decoded, iterations) {
  benchmarkEqualityShape<false, false>(iterations, 192, 0, 10, 1);
}
BENCHMARK_RELATIVE(
    FSST_EqualityShape_B192_MixedEarly_Eq_Compressed,
    iterations) {
  benchmarkEqualityShape<true, false>(iterations, 192, 0, 10, 1);
}

BENCHMARK(FSST_EqualityShape_B192_MixedEarly_Ne_Decoded, iterations) {
  benchmarkEqualityShape<false, true>(iterations, 192, 0, 10, 1);
}
BENCHMARK_RELATIVE(
    FSST_EqualityShape_B192_MixedEarly_Ne_Compressed,
    iterations) {
  benchmarkEqualityShape<true, true>(iterations, 192, 0, 10, 1);
}

BENCHMARK(FSST_EqualityShape_B192_MixedLate_Eq_Decoded, iterations) {
  benchmarkEqualityShape<false, false>(iterations, 192, 191, 10, 1);
}
BENCHMARK_RELATIVE(
    FSST_EqualityShape_B192_MixedLate_Eq_Compressed,
    iterations) {
  benchmarkEqualityShape<true, false>(iterations, 192, 191, 10, 1);
}

BENCHMARK(FSST_EqualityShape_B192_MixedLate_Ne_Decoded, iterations) {
  benchmarkEqualityShape<false, true>(iterations, 192, 191, 10, 1);
}
BENCHMARK_RELATIVE(
    FSST_EqualityShape_B192_MixedLate_Ne_Compressed,
    iterations) {
  benchmarkEqualityShape<true, true>(iterations, 192, 191, 10, 1);
}

BENCHMARK(FSST_EqualityShape_B192_SparseLate_Eq_Decoded, iterations) {
  benchmarkEqualityShape<false, false>(iterations, 192, 191, 0, 31);
}
BENCHMARK_RELATIVE(
    FSST_EqualityShape_B192_SparseLate_Eq_Compressed,
    iterations) {
  benchmarkEqualityShape<true, false>(iterations, 192, 191, 0, 31);
}

BENCHMARK(FSST_EqualityShape_B192_SparseLate_Ne_Decoded, iterations) {
  benchmarkEqualityShape<false, true>(iterations, 192, 191, 0, 31);
}
BENCHMARK_RELATIVE(
    FSST_EqualityShape_B192_SparseLate_Ne_Compressed,
    iterations) {
  benchmarkEqualityShape<true, true>(iterations, 192, 191, 0, 31);
}

BENCHMARK(FSST_EqualityShape_B192_SparseAll_Eq_Decoded, iterations) {
  benchmarkEqualityShape<false, false>(iterations, 192, 0, 100, 31);
}
BENCHMARK_RELATIVE(
    FSST_EqualityShape_B192_SparseAll_Eq_Compressed,
    iterations) {
  benchmarkEqualityShape<true, false>(iterations, 192, 0, 100, 31);
}

BENCHMARK(FSST_EqualityShape_B192_SparseAll_Ne_Decoded, iterations) {
  benchmarkEqualityShape<false, true>(iterations, 192, 0, 100, 31);
}
BENCHMARK_RELATIVE(
    FSST_EqualityShape_B192_SparseAll_Ne_Compressed,
    iterations) {
  benchmarkEqualityShape<true, true>(iterations, 192, 0, 100, 31);
}

template <bool compressed, bool negated>
void benchmarkRangeShape(uint32_t iterations, size_t valueBytes, int shape) {
  folly::BenchmarkSuspender suspender;
  const std::string value(valueBytes, 'x');
  std::string lower = shape == 0 ? std::string(valueBytes, 'm') : value;
  std::string upper = shape == 0 ? std::string(valueBytes, 'z') : value;
  if (shape != 0) {
    lower.back() = shape == 2 ? 'z' : 'w';
    upper.back() = shape == 2 ? '{' : 'z';
  }
  using Filter = std::conditional_t<
      negated,
      facebook::velox::common::NegatedBytesRange,
      facebook::velox::common::BytesRange>;
  Filter filter{lower, false, false, upper, shape == 3, false, false};
  RangeBenchmarkFixture fixture{RangeBenchmarkParams{
      .stride = 1,
      .shortValues = false,
      .singleValue = true,
      .selectivity = 100,
      .prefix = "",
      .lower = value,
      .filter = filter,
  }};
  {
    RangeBenchmarkVisitor<compressed, Filter> visitor{
        fixture.rows(), filter, fixture.passingRows()};
    ReadWithVisitorParams params{};
    fixture.decoder().readWithVisitor(visitor, params);
    fixture.validate(compressed);
  }
  suspender.dismiss();
  while (iterations--) {
    fixture.decoder().reset();
    RangeBenchmarkVisitor<compressed, Filter> visitor{
        fixture.rows(), filter, fixture.passingRows()};
    ReadWithVisitorParams params{};
    fixture.decoder().readWithVisitor(visitor, params);
    folly::doNotOptimizeAway(fixture.passingRows().size());
  }
  suspender.rehire();
  fixture.validate(compressed);
}

BENCHMARK(FSST_RangeShape_B192_EarlyInside_Positive_Decoded, iterations) {
  benchmarkRangeShape<false, false>(iterations, 192, 0);
}
BENCHMARK_RELATIVE(
    FSST_RangeShape_B192_EarlyInside_Positive_Compressed,
    iterations) {
  benchmarkRangeShape<true, false>(iterations, 192, 0);
}

BENCHMARK(FSST_RangeShape_B192_EarlyInside_Negated_Decoded, iterations) {
  benchmarkRangeShape<false, true>(iterations, 192, 0);
}
BENCHMARK_RELATIVE(
    FSST_RangeShape_B192_EarlyInside_Negated_Compressed,
    iterations) {
  benchmarkRangeShape<true, true>(iterations, 192, 0);
}

BENCHMARK(FSST_RangeShape_B192_LateInside_Positive_Decoded, iterations) {
  benchmarkRangeShape<false, false>(iterations, 192, 1);
}
BENCHMARK_RELATIVE(
    FSST_RangeShape_B192_LateInside_Positive_Compressed,
    iterations) {
  benchmarkRangeShape<true, false>(iterations, 192, 1);
}

BENCHMARK(FSST_RangeShape_B192_LateInside_Negated_Decoded, iterations) {
  benchmarkRangeShape<false, true>(iterations, 192, 1);
}
BENCHMARK_RELATIVE(
    FSST_RangeShape_B192_LateInside_Negated_Compressed,
    iterations) {
  benchmarkRangeShape<true, true>(iterations, 192, 1);
}

BENCHMARK(FSST_RangeShape_B192_LateBelow_Positive_Decoded, iterations) {
  benchmarkRangeShape<false, false>(iterations, 192, 2);
}
BENCHMARK_RELATIVE(
    FSST_RangeShape_B192_LateBelow_Positive_Compressed,
    iterations) {
  benchmarkRangeShape<true, false>(iterations, 192, 2);
}

BENCHMARK(FSST_RangeShape_B192_LateBelow_Negated_Decoded, iterations) {
  benchmarkRangeShape<false, true>(iterations, 192, 2);
}
BENCHMARK_RELATIVE(
    FSST_RangeShape_B192_LateBelow_Negated_Compressed,
    iterations) {
  benchmarkRangeShape<true, true>(iterations, 192, 2);
}

BENCHMARK(FSST_RangeShape_B192_OneSidedLate_Positive_Decoded, iterations) {
  benchmarkRangeShape<false, false>(iterations, 192, 3);
}
BENCHMARK_RELATIVE(
    FSST_RangeShape_B192_OneSidedLate_Positive_Compressed,
    iterations) {
  benchmarkRangeShape<true, false>(iterations, 192, 3);
}

BENCHMARK(FSST_RangeShape_B192_OneSidedLate_Negated_Decoded, iterations) {
  benchmarkRangeShape<false, true>(iterations, 192, 3);
}
BENCHMARK_RELATIVE(
    FSST_RangeShape_B192_OneSidedLate_Negated_Compressed,
    iterations) {
  benchmarkRangeShape<true, true>(iterations, 192, 3);
}

BENCHMARK(FSST_RangeShape_B1024_EarlyInside_Positive_Decoded, iterations) {
  benchmarkRangeShape<false, false>(iterations, 1'024, 0);
}
BENCHMARK_RELATIVE(
    FSST_RangeShape_B1024_EarlyInside_Positive_Compressed,
    iterations) {
  benchmarkRangeShape<true, false>(iterations, 1'024, 0);
}

BENCHMARK(FSST_RangeShape_B1024_EarlyInside_Negated_Decoded, iterations) {
  benchmarkRangeShape<false, true>(iterations, 1'024, 0);
}
BENCHMARK_RELATIVE(
    FSST_RangeShape_B1024_EarlyInside_Negated_Compressed,
    iterations) {
  benchmarkRangeShape<true, true>(iterations, 1'024, 0);
}

BENCHMARK(FSST_RangeShape_B1024_LateInside_Positive_Decoded, iterations) {
  benchmarkRangeShape<false, false>(iterations, 1'024, 1);
}
BENCHMARK_RELATIVE(
    FSST_RangeShape_B1024_LateInside_Positive_Compressed,
    iterations) {
  benchmarkRangeShape<true, false>(iterations, 1'024, 1);
}

BENCHMARK(FSST_RangeShape_B1024_LateInside_Negated_Decoded, iterations) {
  benchmarkRangeShape<false, true>(iterations, 1'024, 1);
}
BENCHMARK_RELATIVE(
    FSST_RangeShape_B1024_LateInside_Negated_Compressed,
    iterations) {
  benchmarkRangeShape<true, true>(iterations, 1'024, 1);
}

BENCHMARK(FSST_RangeShape_B1024_LateBelow_Positive_Decoded, iterations) {
  benchmarkRangeShape<false, false>(iterations, 1'024, 2);
}
BENCHMARK_RELATIVE(
    FSST_RangeShape_B1024_LateBelow_Positive_Compressed,
    iterations) {
  benchmarkRangeShape<true, false>(iterations, 1'024, 2);
}

BENCHMARK(FSST_RangeShape_B1024_LateBelow_Negated_Decoded, iterations) {
  benchmarkRangeShape<false, true>(iterations, 1'024, 2);
}
BENCHMARK_RELATIVE(
    FSST_RangeShape_B1024_LateBelow_Negated_Compressed,
    iterations) {
  benchmarkRangeShape<true, true>(iterations, 1'024, 2);
}

BENCHMARK(FSST_RangeShape_B1024_OneSidedLate_Positive_Decoded, iterations) {
  benchmarkRangeShape<false, false>(iterations, 1'024, 3);
}
BENCHMARK_RELATIVE(
    FSST_RangeShape_B1024_OneSidedLate_Positive_Compressed,
    iterations) {
  benchmarkRangeShape<true, false>(iterations, 1'024, 3);
}

BENCHMARK(FSST_RangeShape_B1024_OneSidedLate_Negated_Decoded, iterations) {
  benchmarkRangeShape<false, true>(iterations, 1'024, 3);
}
BENCHMARK_RELATIVE(
    FSST_RangeShape_B1024_OneSidedLate_Negated_Compressed,
    iterations) {
  benchmarkRangeShape<true, true>(iterations, 1'024, 3);
}

} // namespace

int main(int argc, char** argv) {
  const folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize({});
  printArtifactRatio();
  folly::runBenchmarks();
}
