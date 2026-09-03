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

void printArtifactRatio() {
  const FsstBenchmarkFixture fixture;
  fmt::print(
      "FSST full artifact: raw={} encoded={} encoded_to_raw={:.6f} "
      "raw_to_encoded={:.6f}\n",
      fixture.corpus().rawBytes,
      fixture.encoded().size(),
      static_cast<double>(fixture.encoded().size()) / fixture.corpus().rawBytes,
      static_cast<double>(fixture.corpus().rawBytes) /
          fixture.encoded().size());
}

} // namespace

int main(int argc, char** argv) {
  const folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize({});
  printArtifactRatio();
  folly::runBenchmarks();
}
