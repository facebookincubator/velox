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

#include "velox/dwio/nimble/encodings/benchmarks/NimbleEncodingRunner.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <folly/Benchmark.h>
#include <folly/String.h>
#include <folly/dynamic.h>
#include <folly/json/json.h>
#include <folly/ssl/OpenSSLHash.h>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble::benchmarks {
namespace {

using Clock = std::chrono::steady_clock;

enum class RunnerEncoding {
  Dictionary,
  Nullable,
  ALP,
  DeltaBlock,
};

enum class RunnerLane {
  Encode,
  DecodeConstruct,
  DecodeDense,
  DecodeRange50,
  DecodeScatter10,
  DecodeScatter1,
  SkipSeek,
  ViewRandom,
  Slice,
  SelectionE2E,
};

struct TaskSpec {
  RunnerEncoding encoding;
  RunnerLane lane;
  std::string_view encodingSlug;
  std::string_view laneName;
  EncodingType encodingType;
  std::string_view dataType;
};

constexpr std::array<std::pair<std::string_view, RunnerLane>, 10> kLanes{{
    {"encode", RunnerLane::Encode},
    {"decode_construct", RunnerLane::DecodeConstruct},
    {"decode_dense", RunnerLane::DecodeDense},
    {"decode_range50", RunnerLane::DecodeRange50},
    {"decode_scatter10", RunnerLane::DecodeScatter10},
    {"decode_scatter1", RunnerLane::DecodeScatter1},
    {"skip_seek", RunnerLane::SkipSeek},
    {"view_random", RunnerLane::ViewRandom},
    {"slice", RunnerLane::Slice},
    {"selection_e2e", RunnerLane::SelectionE2E},
}};

struct EncodingSpec {
  std::string_view slug;
  RunnerEncoding encoding;
  EncodingType encodingType;
  std::string_view dataType;
  bool supportsView;
};

constexpr std::array<EncodingSpec, 4> kEncodings{{
    {"dictionary",
     RunnerEncoding::Dictionary,
     EncodingType::Dictionary,
     "int64",
     true},
    {"nullable",
     RunnerEncoding::Nullable,
     EncodingType::Nullable,
     "int64",
     false},
    {"alp", RunnerEncoding::ALP, EncodingType::ALP, "double", true},
    {"delta_block",
     RunnerEncoding::DeltaBlock,
     EncodingType::DeltaBlock,
     "int64",
     true},
}};

constexpr uint32_t kMaxRowCount = 1'048'576;
constexpr uint32_t kMaxWarmups = 100;
constexpr uint32_t kMaxSamples = 100;
constexpr uint32_t kMaxMinSampleTimeMicros = 10'000'000;
constexpr uint32_t kMaxCalibratedIterations = 1'000'000'000;

[[noreturn]] void fail(std::string message) {
  throw std::runtime_error{std::move(message)};
}

void require(bool condition, std::string message) {
  if (!condition) {
    fail(std::move(message));
  }
}

TaskSpec parseTaskId(std::string_view taskId) {
  constexpr std::string_view kPrefix{"nimble."};
  constexpr std::string_view kSuffix{".v1"};
  if (!taskId.starts_with(kPrefix) || !taskId.ends_with(kSuffix)) {
    throw std::invalid_argument(
        "task_id must match nimble.<encoding>.<lane>.v1");
  }

  const auto body = taskId.substr(
      kPrefix.size(), taskId.size() - kPrefix.size() - kSuffix.size());
  for (const auto& encoding : kEncodings) {
    const auto encodingPrefix = fmt::format("{}.", encoding.slug);
    if (!body.starts_with(encodingPrefix)) {
      continue;
    }
    const auto laneName = body.substr(encodingPrefix.size());
    for (const auto& [candidateName, lane] : kLanes) {
      if (laneName != candidateName) {
        continue;
      }
      if (lane == RunnerLane::ViewRandom && !encoding.supportsView) {
        throw std::invalid_argument(
            fmt::format("{} does not support view_random", encoding.slug));
      }
      return TaskSpec{
          .encoding = encoding.encoding,
          .lane = lane,
          .encodingSlug = encoding.slug,
          .laneName = candidateName,
          .encodingType = encoding.encodingType,
          .dataType = encoding.dataType,
      };
    }
    throw std::invalid_argument(
        fmt::format("unsupported Stage-0 lane: {}", laneName));
  }
  throw std::invalid_argument(
      "encoding runner currently supports Dictionary, Nullable, ALP, and "
      "DeltaBlock");
}

TaskSpec validateConfig(const EncodingRunnerConfig& config) {
  if (config.rowCount < 100) {
    throw std::invalid_argument("row_count must be at least 100");
  }
  if (config.rowCount > kMaxRowCount) {
    throw std::invalid_argument("row_count exceeds the runner limit");
  }
  if (config.samples < 5) {
    throw std::invalid_argument("samples must be at least 5");
  }
  if (config.samples > kMaxSamples) {
    throw std::invalid_argument("samples exceeds the runner limit");
  }
  if (config.warmups > kMaxWarmups) {
    throw std::invalid_argument("warmups exceeds the runner limit");
  }
  if (config.minSampleTimeMicros > kMaxMinSampleTimeMicros) {
    throw std::invalid_argument(
        "min_sample_time_micros exceeds the runner limit");
  }
  if (config.innerIterations == 0) {
    throw std::invalid_argument("inner_iterations must be positive");
  }
  if (config.innerIterations > kMaxCalibratedIterations) {
    throw std::invalid_argument("inner_iterations exceeds the runner limit");
  }
  if (config.seed >
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    throw std::invalid_argument(
        "seed must fit in a signed 64-bit JSON integer");
  }
  return parseTaskId(config.taskId);
}

std::string digest(std::span<const std::byte> bytes) {
  std::array<uint8_t, 32> result{};
  folly::ssl::OpenSSLHash::sha256(
      folly::MutableByteRange{result.data(), result.size()},
      folly::ByteRange{
          reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size()});
  return folly::hexlify(folly::ByteRange{result.data(), result.size()});
}

std::string digest(std::string_view value) {
  return digest(std::as_bytes(std::span{value.data(), value.size()}));
}

template <typename T>
std::string semanticDigest(
    const std::vector<T>& values,
    std::span<const bool> nonNulls,
    std::string_view dataType) {
  const auto valueBytes =
      std::as_bytes(std::span{values.data(), values.size()});
  std::string canonical;
  canonical.reserve(dataType.size() + 1 + nonNulls.size() + valueBytes.size());
  canonical.append(dataType);
  canonical.push_back('\0');
  for (const bool nonNull : nonNulls) {
    canonical.push_back(nonNull ? '\1' : '\0');
  }
  canonical.append(
      reinterpret_cast<const char*>(valueBytes.data()), valueBytes.size());
  return digest(canonical);
}

template <typename T>
typename TypeTraits<T>::physicalType toPhysical(T value) {
  using PhysicalType = typename TypeTraits<T>::physicalType;
  static_assert(sizeof(T) == sizeof(PhysicalType));
  return std::bit_cast<PhysicalType>(value);
}

template <typename T>
struct Corpus {
  using PhysicalType = typename TypeTraits<T>::physicalType;

  explicit Corpus(velox::memory::MemoryPool& pool, uint32_t rowCount)
      : nonNulls{&pool, rowCount, true} {
    logicalValues.reserve(rowCount);
    expectedPhysical.reserve(rowCount);
  }

  std::vector<T> logicalValues;
  std::vector<PhysicalType> expectedPhysical;
  Vector<bool> nonNulls;
};

Corpus<int64_t> makeIntegerCorpus(
    const TaskSpec& spec,
    const EncodingRunnerConfig& config,
    velox::memory::MemoryPool& pool) {
  Corpus<int64_t> corpus{pool, config.rowCount};
  std::mt19937_64 rng{config.seed};

  if (spec.encoding == RunnerEncoding::DeltaBlock) {
    int64_t value{-1'000'000};
    for (uint32_t row = 0; row < config.rowCount; ++row) {
      value += static_cast<int64_t>(rng() % 17);
      corpus.logicalValues.push_back(value);
      corpus.expectedPhysical.push_back(toPhysical(value));
    }
    return corpus;
  }

  for (uint32_t row = 0; row < config.rowCount; ++row) {
    const bool isNonNull = spec.encoding != RunnerEncoding::Nullable ||
        (row != 0 && rng() % 5 != 0);
    corpus.nonNulls[row] = isNonNull;
    const int64_t value = static_cast<int64_t>(rng() % 64) - 32;
    if (isNonNull) {
      corpus.logicalValues.push_back(value);
      corpus.expectedPhysical.push_back(toPhysical(value));
    } else {
      corpus.expectedPhysical.push_back(0);
    }
  }
  return corpus;
}

Corpus<double> makeDoubleCorpus(
    const EncodingRunnerConfig& config,
    velox::memory::MemoryPool& pool) {
  Corpus<double> corpus{pool, config.rowCount};
  std::mt19937_64 rng{config.seed};
  for (uint32_t row = 0; row < config.rowCount; ++row) {
    double value =
        static_cast<double>(static_cast<int64_t>(rng() % 100'001) - 50'000) /
        100.0;
    if (row % 257 == 0) {
      value = -0.0;
    } else if (row % 1021 == 0) {
      value = std::numeric_limits<double>::infinity();
    }
    corpus.logicalValues.push_back(value);
    corpus.expectedPhysical.push_back(toPhysical(value));
  }
  return corpus;
}

EncodingLayout trivialLayout() {
  return EncodingLayout{
      EncodingType::Trivial, {}, CompressionType::Uncompressed};
}

EncodingLayout fixedBitWidthLayout() {
  return EncodingLayout{
      EncodingType::FixedBitWidth, {}, CompressionType::Uncompressed};
}

EncodingLayout varintLayout() {
  return EncodingLayout{
      EncodingType::Varint, {}, CompressionType::Uncompressed};
}

EncodingLayout replayLayout(RunnerEncoding encoding) {
  switch (encoding) {
    case RunnerEncoding::Dictionary:
      return EncodingLayout{
          EncodingType::Dictionary,
          {},
          CompressionType::Uncompressed,
          {trivialLayout(), fixedBitWidthLayout()}};
    case RunnerEncoding::Nullable:
      return trivialLayout();
    case RunnerEncoding::ALP:
      return EncodingLayout{
          EncodingType::ALP,
          {},
          CompressionType::Uncompressed,
          {fixedBitWidthLayout(), varintLayout(), trivialLayout()}};
    case RunnerEncoding::DeltaBlock:
      return EncodingLayout{
          EncodingType::DeltaBlock, {}, CompressionType::Uncompressed};
  }
  fail("unknown runner encoding");
}

EncodingSelectionPolicyCreator fallbackPolicyCreator() {
  return [](DataType dataType) {
    return ManualEncodingSelectionPolicyFactory{
        {{EncodingType::Trivial, 1.0}},
        /*compressionOptions=*/std::nullopt}
        .createPolicy(dataType);
  };
}

template <typename T>
std::unique_ptr<EncodingSelectionPolicy<T>> replayPolicy(
    RunnerEncoding encoding) {
  return std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
      replayLayout(encoding),
      /*compressionOptions=*/std::nullopt,
      fallbackPolicyCreator());
}

template <typename T>
std::unique_ptr<EncodingSelectionPolicy<T>> selectionPolicy(
    EncodingType encodingType) {
  ManualEncodingSelectionPolicyFactory factory{
      {{encodingType, 1.0}},
      /*compressionOptions=*/std::nullopt};
  auto policy = factory.createPolicy(TypeTraits<T>::dataType);
  return std::unique_ptr<EncodingSelectionPolicy<T>>{
      static_cast<EncodingSelectionPolicy<T>*>(policy.release())};
}

std::vector<uint32_t>
scatterPositions(uint32_t rowCount, uint32_t percentage, uint64_t seed) {
  const uint32_t count = std::max<uint32_t>(
      1,
      static_cast<uint32_t>(
          static_cast<uint64_t>(rowCount) * percentage / 100));
  const uint32_t rotation = static_cast<uint32_t>(seed % rowCount);
  std::vector<uint32_t> positions;
  positions.reserve(count);
  for (uint32_t index = 0; index < count; ++index) {
    positions.push_back(
        (static_cast<uint64_t>(index) * rowCount / count + rotation) %
        rowCount);
  }
  std::sort(positions.begin(), positions.end());
  require(
      std::adjacent_find(positions.begin(), positions.end()) == positions.end(),
      "scatter position generation produced a duplicate");
  return positions;
}

std::vector<uint32_t> randomViewPositions(uint32_t rowCount, uint64_t seed) {
  auto positions = scatterPositions(rowCount, 10, seed);
  std::mt19937_64 rng{seed ^ 0x9E3779B97F4A7C15ULL};
  for (size_t remaining = positions.size(); remaining > 1; --remaining) {
    std::swap(positions[remaining - 1], positions[rng() % remaining]);
  }
  if (std::is_sorted(positions.begin(), positions.end())) {
    std::rotate(positions.begin(), positions.begin() + 1, positions.end());
  }
  return positions;
}

template <typename T>
class TypedRunner {
 public:
  using PhysicalType = typename TypeTraits<T>::physicalType;

  TypedRunner(
      const EncodingRunnerConfig& config,
      TaskSpec spec,
      velox::memory::MemoryPool& pool,
      std::optional<std::string_view> artifact = std::nullopt)
      : config_{config},
        spec_{spec},
        pool_{pool},
        corpus_{makeCorpus()},
        encodedArtifact_{
            artifact.has_value() ? std::string{*artifact} : encodeArtifact()},
        timingBuffer_{pool_},
        output_(config.rowCount),
        rangeOutput_(config.rowCount / 2),
        scatter10_{scatterPositions(config.rowCount, 10, config.seed)},
        scatter1_{scatterPositions(config.rowCount, 1, config.seed)},
        viewPositions_{randomViewPositions(config.rowCount, config.seed)},
        scatterOutput_(scatter10_.size()),
        skipSeekOutput_(config.rowCount),
        viewOutput_(viewPositions_.size()) {
    validateArtifact();
    decoder_ = createEncoding(encodedArtifact_);
    if (spec_.lane == RunnerLane::ViewRandom) {
      view_ = createEncodingView(encodedArtifact_, &pool_, options_);
    }
  }

  EncodingRunnerMeasurement run() {
    if (spec_.lane == RunnerLane::SelectionE2E) {
      validateSelection();
    }
    const uint32_t iterations = calibrateIterations();
    for (uint32_t warmup = 0; warmup < config_.warmups; ++warmup) {
      runIterations(iterations);
      validateTimedResult();
    }

    std::vector<double> samples;
    samples.reserve(config_.samples);
    for (uint32_t sample = 0; sample < config_.samples; ++sample) {
      const double elapsed = runIterations(iterations) / iterations;
      require(
          std::isfinite(elapsed) && elapsed > 0.0,
          "timing sample must be finite and positive");
      samples.push_back(elapsed);
      validateTimedResult();
    }

    return EncodingRunnerMeasurement{
        .taskId = config_.taskId,
        .encoding = std::string{spec_.encodingSlug},
        .lane = std::string{spec_.laneName},
        .dataType = std::string{spec_.dataType},
        .seed = config_.seed,
        .rowCount = config_.rowCount,
        .encodedBytes = static_cast<uint32_t>(encodedArtifact_.size()),
        .inputDigest = inputDigest(),
        .outputDigest = outputDigest_,
        .artifactDigest = digest(encodedArtifact_),
        .samplesSeconds = std::move(samples),
        .encodedArtifact = encodedArtifact_,
    };
  }

  EncodingArtifactVerification verification() const {
    return EncodingArtifactVerification{
        .taskId = config_.taskId,
        .encoding = std::string{spec_.encodingSlug},
        .dataType = std::string{spec_.dataType},
        .seed = config_.seed,
        .rowCount = config_.rowCount,
        .encodedBytes = static_cast<uint32_t>(encodedArtifact_.size()),
        .inputDigest = inputDigest(),
        .outputDigest = outputDigest_,
        .artifactDigest = digest(encodedArtifact_),
    };
  }

 private:
  std::span<const bool> nonNulls() const {
    return {corpus_.nonNulls.data(), corpus_.nonNulls.size()};
  }

  std::string inputDigest() const {
    return semanticDigest(corpus_.expectedPhysical, nonNulls(), spec_.dataType);
  }

  Corpus<T> makeCorpus() {
    if constexpr (std::is_same_v<T, double>) {
      return makeDoubleCorpus(config_, pool_);
    } else {
      return makeIntegerCorpus(spec_, config_, pool_);
    }
  }

  std::string_view encodeToBuffer(Buffer& buffer, bool runSelection) const {
    auto policy = runSelection ? selectionPolicy<T>(spec_.encodingType)
                               : replayPolicy<T>(spec_.encoding);
    std::string_view encoded;
    if (spec_.encoding == RunnerEncoding::Nullable) {
      encoded = EncodingFactory::encodeNullable<T>(
          std::move(policy),
          corpus_.logicalValues,
          std::span<const bool>{
              corpus_.nonNulls.data(), corpus_.nonNulls.size()},
          buffer,
          options_);
    } else {
      encoded = EncodingFactory::encode<T>(
          std::move(policy), corpus_.logicalValues, buffer, options_);
    }
    return encoded;
  }

  std::string encodeArtifact() {
    Buffer buffer{pool_};
    return std::string{
        encodeToBuffer(buffer, spec_.lane == RunnerLane::SelectionE2E)};
  }

  std::unique_ptr<Encoding> createEncoding(std::string_view artifact) const {
    return EncodingFactory{options_}.create(
        pool_, artifact, [](uint32_t) -> void* { return nullptr; });
  }

  void validateArtifact() {
    require(!encodedArtifact_.empty(), "encoded artifact must not be empty");
    require(
        encodedArtifact_.size() <= kMaxEncodingArtifactBytes,
        "encoded artifact exceeds the runner size limit");
    auto encoding = createEncoding(encodedArtifact_);
    require(
        encoding->encodingType() == spec_.encodingType,
        "artifact root encoding does not match task");
    require(
        encoding->rowCount() == config_.rowCount,
        "artifact row count does not match corpus");

    validateDense(*encoding);
    validateFragmented(*encoding);
    validateRange(*encoding);
    validateScatter(*encoding, scatter10_);
    validateScatter(*encoding, scatter1_);
    validateNullable(*encoding);
    validateSlice();
    validateView();
  }

  void validateDense(Encoding& encoding) {
    std::vector<PhysicalType> actual(config_.rowCount);
    encoding.reset();
    encoding.materialize(config_.rowCount, actual.data());
    require(actual == corpus_.expectedPhysical, "dense round trip failed");
    outputDigest_ = semanticDigest(actual, nonNulls(), spec_.dataType);
  }

  void validateFragmented(Encoding& encoding) const {
    std::vector<PhysicalType> actual(config_.rowCount);
    encoding.reset();
    uint32_t offset{0};
    while (offset < config_.rowCount) {
      const uint32_t count =
          std::min<uint32_t>(1 + offset % 31, config_.rowCount - offset);
      encoding.materialize(count, actual.data() + offset);
      offset += count;
    }
    require(
        actual == corpus_.expectedPhysical,
        "fragmented materialization failed");
  }

  void validateRange(Encoding& encoding) const {
    const uint32_t offset = config_.rowCount / 4;
    const uint32_t count = config_.rowCount / 2;
    std::vector<PhysicalType> actual(count);
    encoding.reset();
    encoding.skip(offset);
    encoding.materialize(count, actual.data());
    require(
        std::equal(
            actual.begin(),
            actual.end(),
            corpus_.expectedPhysical.begin() + offset),
        "range materialization failed");
  }

  void validateScatter(
      Encoding& encoding,
      const std::vector<uint32_t>& positions) const {
    std::vector<PhysicalType> actual(positions.size());
    encoding.reset();
    uint32_t cursor{0};
    for (uint32_t index = 0; index < positions.size(); ++index) {
      const uint32_t position = positions[index];
      encoding.skip(position - cursor);
      encoding.materialize(1, actual.data() + index);
      cursor = position + 1;
    }
    for (uint32_t index = 0; index < positions.size(); ++index) {
      require(
          actual[index] == corpus_.expectedPhysical[positions[index]],
          "scatter materialization failed");
    }
  }

  void validateNullable(Encoding& encoding) const {
    if (spec_.encoding != RunnerEncoding::Nullable) {
      return;
    }
    std::vector<PhysicalType> actual(config_.rowCount);
    std::vector<uint64_t> nonNullBitmap((config_.rowCount + 63) / 64, 0);
    encoding.reset();
    const auto actualNonNullCount = encoding.materializeNullable(
        config_.rowCount, actual.data(), [&nonNullBitmap]() -> void* {
          return nonNullBitmap.data();
        });
    require(
        actualNonNullCount == corpus_.logicalValues.size(),
        "nullable non-null count mismatch");
    for (uint32_t row = 0; row < config_.rowCount; ++row) {
      const bool actualNonNull =
          (nonNullBitmap[row / 64] & (uint64_t{1} << (row % 64))) != 0;
      require(
          actualNonNull == corpus_.nonNulls[row], "nullable bitmap mismatch");
      if (actualNonNull) {
        require(
            actual[row] == corpus_.expectedPhysical[row],
            "nullable value mismatch");
      }
    }
  }

  void validateSlice() const {
    const uint32_t offset = config_.rowCount / 4;
    const uint32_t count = config_.rowCount / 2;
    Buffer buffer{pool_};
    const auto sliced = EncodingFactory::slice(
        encodedArtifact_, offset, count, buffer, options_);
    validateSliceArtifact(sliced);
  }

  void validateSliceArtifact(std::string_view sliced) const {
    const uint32_t offset = config_.rowCount / 4;
    const uint32_t count = config_.rowCount / 2;
    auto encoding = createEncoding(sliced);
    require(
        encoding->rowCount() == count, "sliced artifact row count mismatch");
    std::vector<PhysicalType> actual(count);
    encoding->materialize(count, actual.data());
    require(
        std::equal(
            actual.begin(),
            actual.end(),
            corpus_.expectedPhysical.begin() + offset),
        "sliced artifact materialization failed");
  }

  void validateView() const {
    if (!supportsEncodingView(spec_.encodingType)) {
      return;
    }
    auto view = createEncodingView(encodedArtifact_, &pool_, options_);
    for (const auto position : viewPositions_) {
      PhysicalType actual{};
      view->readAt(position, &actual);
      require(
          actual == corpus_.expectedPhysical[position],
          "encoding view read failed");
    }
  }

  void validateSelection() const {
    Buffer buffer{pool_};
    const auto selected = encodeToBuffer(buffer, true);
    require(
        selected == encodedArtifact_,
        "selection_e2e did not reproduce the exported artifact");
    auto encoding = createEncoding(selected);
    require(
        encoding->encodingType() == spec_.encodingType,
        "selection_e2e selected a different root encoding");
    std::vector<PhysicalType> actual(config_.rowCount);
    encoding->materialize(config_.rowCount, actual.data());
    require(
        actual == corpus_.expectedPhysical, "selection_e2e round trip failed");
  }

  void validateTimedEncode(bool runSelection) const {
    require(
        timedArtifact_ == encodedArtifact_,
        "timed encode did not reproduce the validated artifact");
    if (runSelection) {
      require(
          output_ == corpus_.expectedPhysical,
          "timed selection_e2e decode failed");
    }
  }

  void validateTimedConstruct() const {
    require(
        constructedRowCount_ == config_.rowCount,
        "timed construction returned the wrong row count");
    require(
        constructedEncodingType_ == spec_.encodingType,
        "timed construction returned the wrong encoding type");
  }

  void validateTimedScatter(const std::vector<uint32_t>& positions) const {
    for (uint32_t index = 0; index < positions.size(); ++index) {
      require(
          scatterOutput_[index] == corpus_.expectedPhysical[positions[index]],
          "timed scatter decode failed");
    }
  }

  void validateTimedSkipSeek() const {
    require(skipSeekReadCount_ > 0, "timed skip_seek did not read a value");
    uint32_t cursor{0};
    uint32_t outputIndex{0};
    while (cursor < config_.rowCount) {
      const uint32_t skip = std::min<uint32_t>(31, config_.rowCount - cursor);
      cursor += skip;
      if (cursor == config_.rowCount) {
        break;
      }
      const uint32_t read = std::min<uint32_t>(3, config_.rowCount - cursor);
      require(
          std::equal(
              skipSeekOutput_.begin() + outputIndex,
              skipSeekOutput_.begin() + outputIndex + read,
              corpus_.expectedPhysical.begin() + cursor),
          "timed skip_seek decode failed");
      outputIndex += read;
      cursor += read;
    }
    require(
        outputIndex == skipSeekReadCount_,
        "timed skip_seek returned the wrong value count");
  }

  void validateTimedResult() const {
    switch (spec_.lane) {
      case RunnerLane::Encode:
        validateTimedEncode(false);
        return;
      case RunnerLane::DecodeConstruct:
        validateTimedConstruct();
        return;
      case RunnerLane::DecodeDense:
        require(
            output_ == corpus_.expectedPhysical, "timed dense decode failed");
        return;
      case RunnerLane::DecodeRange50:
        require(
            std::equal(
                rangeOutput_.begin(),
                rangeOutput_.end(),
                corpus_.expectedPhysical.begin() + config_.rowCount / 4),
            "timed range decode failed");
        return;
      case RunnerLane::DecodeScatter10:
        validateTimedScatter(scatter10_);
        return;
      case RunnerLane::DecodeScatter1:
        validateTimedScatter(scatter1_);
        return;
      case RunnerLane::SkipSeek:
        validateTimedSkipSeek();
        return;
      case RunnerLane::ViewRandom:
        for (uint32_t index = 0; index < viewPositions_.size(); ++index) {
          require(
              viewOutput_[index] ==
                  corpus_.expectedPhysical[viewPositions_[index]],
              "timed view read failed");
        }
        return;
      case RunnerLane::Slice:
        validateSliceArtifact(timedArtifact_);
        return;
      case RunnerLane::SelectionE2E:
        validateTimedEncode(true);
        return;
    }
    fail("unknown runner lane");
  }

  void runOnce() {
    switch (spec_.lane) {
      case RunnerLane::Encode:
        runEncode(false);
        return;
      case RunnerLane::DecodeConstruct:
        runConstruct();
        return;
      case RunnerLane::DecodeDense:
        runDense();
        return;
      case RunnerLane::DecodeRange50:
        runRange();
        return;
      case RunnerLane::DecodeScatter10:
        runScatter(scatter10_);
        return;
      case RunnerLane::DecodeScatter1:
        runScatter(scatter1_);
        return;
      case RunnerLane::SkipSeek:
        runSkipSeek();
        return;
      case RunnerLane::ViewRandom:
        runView();
        return;
      case RunnerLane::Slice:
        runSlice();
        return;
      case RunnerLane::SelectionE2E:
        runEncode(true);
        return;
    }
    fail("unknown runner lane");
  }

  void runEncode(bool runSelection) {
    timingBuffer_.reset();
    const auto encoded = encodeToBuffer(timingBuffer_, runSelection);
    timedArtifact_ = encoded;
    if (runSelection) {
      auto encoding = createEncoding(encoded);
      require(
          encoding->encodingType() == spec_.encodingType,
          "selection_e2e selected a different root encoding");
      encoding->materialize(config_.rowCount, output_.data());
      folly::doNotOptimizeAway(output_.back());
    } else {
      folly::doNotOptimizeAway(encoded.size());
    }
  }

  void runConstruct() {
    auto encoding = createEncoding(encodedArtifact_);
    constructedRowCount_ = encoding->rowCount();
    constructedEncodingType_ = encoding->encodingType();
    folly::doNotOptimizeAway(*constructedRowCount_);
  }

  void runDense() {
    decoder_->reset();
    decoder_->materialize(config_.rowCount, output_.data());
    folly::doNotOptimizeAway(output_.back());
  }

  void runRange() {
    decoder_->reset();
    decoder_->skip(config_.rowCount / 4);
    decoder_->materialize(config_.rowCount / 2, rangeOutput_.data());
    folly::doNotOptimizeAway(rangeOutput_.back());
  }

  void runScatter(const std::vector<uint32_t>& positions) {
    decoder_->reset();
    uint32_t cursor{0};
    for (uint32_t index = 0; index < positions.size(); ++index) {
      decoder_->skip(positions[index] - cursor);
      decoder_->materialize(1, scatterOutput_.data() + index);
      cursor = positions[index] + 1;
    }
    folly::doNotOptimizeAway(scatterOutput_[positions.size() - 1]);
  }

  void runSkipSeek() {
    decoder_->reset();
    uint32_t cursor{0};
    skipSeekReadCount_ = 0;
    while (cursor < config_.rowCount) {
      const uint32_t skip = std::min<uint32_t>(31, config_.rowCount - cursor);
      decoder_->skip(skip);
      cursor += skip;
      if (cursor == config_.rowCount) {
        break;
      }
      const uint32_t read = std::min<uint32_t>(3, config_.rowCount - cursor);
      decoder_->materialize(read, skipSeekOutput_.data() + skipSeekReadCount_);
      skipSeekReadCount_ += read;
      cursor += read;
    }
    folly::doNotOptimizeAway(skipSeekOutput_[skipSeekReadCount_ - 1]);
  }

  void runView() {
    require(view_ != nullptr, "view_random requires a supported encoding view");
    for (uint32_t index = 0; index < viewPositions_.size(); ++index) {
      view_->readAt(viewPositions_[index], viewOutput_.data() + index);
    }
    folly::doNotOptimizeAway(viewOutput_.back());
  }

  void runSlice() {
    timingBuffer_.reset();
    timedArtifact_ = EncodingFactory::slice(
        encodedArtifact_,
        config_.rowCount / 4,
        config_.rowCount / 2,
        timingBuffer_,
        options_);
    folly::doNotOptimizeAway(timedArtifact_.size());
  }

  double runIterations(uint32_t iterations) {
    const auto start = Clock::now();
    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
      runOnce();
    }
    const auto elapsed = Clock::now() - start;
    return std::chrono::duration<double>{elapsed}.count();
  }

  uint32_t calibrateIterations() {
    uint32_t iterations = config_.innerIterations;
    if (config_.minSampleTimeMicros == 0) {
      return iterations;
    }
    const double targetSeconds =
        static_cast<double>(config_.minSampleTimeMicros) / 1'000'000.0;
    for (uint32_t attempt = 0; attempt < 16; ++attempt) {
      const double elapsed = runIterations(iterations);
      if (elapsed >= targetSeconds) {
        return iterations;
      }
      const double safeElapsed = std::max(elapsed, 1e-9);
      const auto multiplier = static_cast<uint32_t>(
          std::clamp(std::ceil(targetSeconds / safeElapsed), 2.0, 16.0));
      if (iterations > kMaxCalibratedIterations / multiplier) {
        fail("timing iteration calibration overflowed");
      }
      iterations *= multiplier;
    }
    fail("timing iteration calibration did not reach the requested duration");
  }

  const EncodingRunnerConfig& config_;
  TaskSpec spec_;
  velox::memory::MemoryPool& pool_;
  Corpus<T> corpus_;
  Encoding::Options options_{};
  std::string encodedArtifact_;
  std::string outputDigest_;
  Buffer timingBuffer_;
  std::unique_ptr<Encoding> decoder_;
  std::unique_ptr<EncodingView> view_;
  std::optional<uint32_t> constructedRowCount_;
  std::optional<EncodingType> constructedEncodingType_;
  std::string_view timedArtifact_;
  std::vector<PhysicalType> output_;
  std::vector<PhysicalType> rangeOutput_;
  std::vector<uint32_t> scatter10_;
  std::vector<uint32_t> scatter1_;
  std::vector<uint32_t> viewPositions_;
  std::vector<PhysicalType> scatterOutput_;
  std::vector<PhysicalType> skipSeekOutput_;
  std::vector<PhysicalType> viewOutput_;
  uint32_t skipSeekReadCount_{0};
};

template <typename Result>
folly::dynamic commonJson(const Result& result, std::string_view kind) {
  return folly::dynamic::object("schema_version", 1)("kind", kind)(
      "task_id", result.taskId)("encoding", result.encoding)(
      "data_type", result.dataType)("seed", static_cast<int64_t>(result.seed))(
      "row_count", result.rowCount)("encoded_bytes", result.encodedBytes)(
      "input_digest", result.inputDigest)("output_digest", result.outputDigest)(
      "artifact_digest", result.artifactDigest)("correctness", true);
}

folly::dynamic samplesJson(const std::vector<double>& samples) {
  folly::dynamic result = folly::dynamic::array;
  for (const auto sample : samples) {
    result.push_back(sample);
  }
  return result;
}

bool isSha256(std::string_view value) {
  constexpr std::string_view kHexDigits{"0123456789abcdef"};
  return value.size() == 64 &&
      value.find_first_not_of(kHexDigits) == std::string_view::npos;
}

template <typename Result>
void validateCommonResult(const Result& result) {
  const auto spec = parseTaskId(result.taskId);
  require(result.encoding == spec.encodingSlug, "result encoding mismatch");
  require(result.dataType == spec.dataType, "result data type mismatch");
  require(
      result.seed <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "result seed exceeds the JSON integer limit");
  require(
      result.rowCount >= 100 && result.rowCount <= kMaxRowCount,
      "result row count is outside runner limits");
  require(
      result.encodedBytes > 0 &&
          result.encodedBytes <= kMaxEncodingArtifactBytes,
      "result encoded size is outside runner limits");
  require(
      result.inputDigest == result.outputDigest,
      "result output digest does not match its input digest");
  require(isSha256(result.inputDigest), "result input digest is not SHA-256");
  require(
      isSha256(result.artifactDigest), "result artifact digest is not SHA-256");
}

void validateMeasurementResult(const EncodingRunnerMeasurement& measurement) {
  validateCommonResult(measurement);
  const auto spec = parseTaskId(measurement.taskId);
  require(measurement.lane == spec.laneName, "measurement lane mismatch");
  require(
      measurement.encodedBytes == measurement.encodedArtifact.size(),
      "measurement artifact size mismatch");
  require(
      measurement.samplesSeconds.size() >= 5 &&
          measurement.samplesSeconds.size() <= kMaxSamples,
      "measurement sample count is outside runner limits");
  for (const double sample : measurement.samplesSeconds) {
    require(
        std::isfinite(sample) && sample > 0.0,
        "measurement samples must be finite and positive");
  }
}

bool consumesCanonicalArtifact(RunnerLane lane) {
  return lane != RunnerLane::Encode && lane != RunnerLane::SelectionE2E;
}

} // namespace

EncodingRunnerMeasurement runEncodingBenchmark(
    const EncodingRunnerConfig& config,
    velox::memory::MemoryPool& pool,
    std::optional<std::string_view> benchmarkArtifact) {
  const auto spec = validateConfig(config);
  if (benchmarkArtifact.has_value() && !consumesCanonicalArtifact(spec.lane)) {
    throw std::invalid_argument(
        "encode and selection_e2e cannot consume a benchmark artifact");
  }
  if (benchmarkArtifact.has_value() &&
      benchmarkArtifact->size() > kMaxEncodingArtifactBytes) {
    throw std::invalid_argument("benchmark artifact exceeds the runner limit");
  }
  if (spec.encoding == RunnerEncoding::ALP) {
    return TypedRunner<double>{config, spec, pool, benchmarkArtifact}.run();
  }
  return TypedRunner<int64_t>{config, spec, pool, benchmarkArtifact}.run();
}

EncodingArtifactVerification verifyEncodingArtifact(
    const EncodingRunnerConfig& config,
    std::string_view encodedArtifact,
    velox::memory::MemoryPool& pool) {
  const auto spec = validateConfig(config);
  if (encodedArtifact.size() > kMaxEncodingArtifactBytes) {
    throw std::invalid_argument(
        "verification artifact exceeds the runner limit");
  }
  if (spec.encoding == RunnerEncoding::ALP) {
    return TypedRunner<double>{
        config, spec, pool, std::optional{encodedArtifact}}
        .verification();
  }
  return TypedRunner<int64_t>{
      config, spec, pool, std::optional{encodedArtifact}}
      .verification();
}

std::string measurementToJson(const EncodingRunnerMeasurement& measurement) {
  validateMeasurementResult(measurement);
  auto result = commonJson(measurement, "nimble_encoding_measurement");
  result["lane"] = measurement.lane;
  result["samples_seconds"] = samplesJson(measurement.samplesSeconds);
  return folly::toJson(result);
}

std::string verificationToJson(
    const EncodingArtifactVerification& verification) {
  validateCommonResult(verification);
  return folly::toJson(
      commonJson(verification, "nimble_encoding_verification"));
}

} // namespace facebook::nimble::benchmarks
