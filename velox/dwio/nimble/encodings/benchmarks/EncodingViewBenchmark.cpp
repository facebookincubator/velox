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
#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "folly/Benchmark.h"
#include "folly/Random.h"
#include "folly/init/Init.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/ALPRDEncoding.h"
#include "velox/dwio/nimble/encodings/BlockBitPackingV2Encoding.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaBlockEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/EliasFanoEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/ForEncoding.h"
#include "velox/dwio/nimble/encodings/FsstEncoding.h"
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/PFOREncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SharedDictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"
#include "velox/dwio/nimble/encodings/SparseBoolEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/VarintEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

using namespace facebook::nimble;
using namespace facebook::nimble::benchmarks;

namespace {

constexpr uint32_t kRows = 4096;
constexpr uint32_t kPositions = 130;
constexpr uint32_t kRangeSize = 1024;

template <typename T>
std::span<const typename TypeTraits<T>::physicalType> physicalSpan(
    const Vector<T>& data) {
  using P = typename TypeTraits<T>::physicalType;
  return {reinterpret_cast<const P*>(data.data()), data.size()};
}

template <typename EncodingT, typename T>
std::string encodeWithSelection(
    EncodingType encodingType,
    const Vector<T>& data,
    const Encoding::Options& options = {}) {
  using P = typename TypeTraits<T>::physicalType;
  auto values = physicalSpan(data);
  auto selection = EncodingSelection<P>{
      EncodingSelectionResult{.encodingType = encodingType},
      Statistics<P>::create(values),
      makeDefaultPolicy(TypeTraits<T>::dataType)};
  Buffer buffer{*benchmarkPool()};
  auto encoded = EncodingT::encode(selection, values, buffer, options);
  return {encoded.data(), encoded.size()};
}

std::string encodeFsstString(const Vector<std::string_view>& data) {
  auto values = std::span<const std::string_view>{data.data(), data.size()};
  auto selection = EncodingSelection<std::string_view>{
      EncodingSelectionResult{.encodingType = EncodingType::Fsst},
      Statistics<std::string_view>::create(values),
      makeDefaultPolicy(DataType::String)};
  Buffer buffer{*benchmarkPool()};
  const Encoding::Options options{
      .fsstCompressionTargetRatio = std::numeric_limits<double>::max()};
  const auto encoded = FsstEncoding::encode(selection, values, buffer, options);
  return {encoded.data(), encoded.size()};
}

std::string encodeTrivialString(
    const Vector<std::string_view>& data,
    const Encoding::Options& options = {}) {
  auto values = std::span<const std::string_view>{data.data(), data.size()};
  auto selection = EncodingSelection<std::string_view>{
      EncodingSelectionResult{.encodingType = EncodingType::Trivial},
      Statistics<std::string_view>::create(values),
      makeDefaultPolicy(DataType::String)};
  Buffer buffer{*benchmarkPool()};
  auto encoded = TrivialEncoding<std::string_view>::encode(
      selection, values, buffer, options);
  return {encoded.data(), encoded.size()};
}

std::vector<uint32_t> randomPositions(uint32_t rows) {
  std::vector<uint32_t> positions(kPositions);
  for (auto& position : positions) {
    position = folly::Random::secureRand32() % rows;
  }
  return positions;
}

std::vector<uint32_t> sortedRandomPositions(uint32_t rows) {
  auto positions = randomPositions(rows);
  std::sort(positions.begin(), positions.end());
  return positions;
}

std::vector<uint32_t> clusteredPositions(uint32_t rows) {
  std::vector<uint32_t> positions(kPositions);
  const auto start = folly::Random::secureRand32() % (rows - kPositions);
  std::iota(positions.begin(), positions.end(), start);
  return positions;
}

std::vector<uint32_t> sharedDictionaryRandomIndices() {
  std::mt19937 generator{1};
  std::uniform_int_distribution<uint32_t> distribution{0, kRows - 1};
  std::vector<uint32_t> indices(kPositions);
  for (auto& index : indices) {
    index = distribution(generator);
  }
  return indices;
}

std::vector<uint32_t> sharedDictionarySortedIndices() {
  auto indices = sharedDictionaryRandomIndices();
  std::sort(indices.begin(), indices.end());
  return indices;
}

std::vector<uint32_t> sharedDictionaryClusteredIndices() {
  std::vector<uint32_t> indices(kPositions);
  std::iota(indices.begin(), indices.end(), (kRows - kPositions) / 2);
  return indices;
}

template <typename T>
void readPositionsWithView(
    const EncodingView& view,
    const std::vector<uint32_t>& positions,
    uint32_t iters) {
  typename TypeTraits<T>::physicalType value;
  while (iters--) {
    for (const auto position : positions) {
      view.readAt(position, &value);
      folly::doNotOptimizeAway(value);
    }
  }
}

template <typename T>
void readPositionsBatchWithView(
    const EncodingView& view,
    const std::vector<uint32_t>& positions,
    uint32_t iters) {
  using PhysicalType = typename TypeTraits<T>::physicalType;
  std::vector<PhysicalType> output(positions.size());
  while (iters--) {
    view.readAt(positions, output.data());
    folly::doNotOptimizeAway(output.data());
  }
}

template <typename T>
std::shared_ptr<const SharedDictionaryAlphabet> createSharedDictionaryAlphabet(
    const Vector<T>& values) {
  Buffer buffer{*benchmarkPool()};
  const auto encoded = SharedDictionaryAlphabet::encode<T>(
      {values.data(), values.size()},
      std::array{EncodingType::BlockBitPackingV2},
      buffer);
  auto encodedOwner = std::make_shared<const std::string>(encoded);
  const std::string_view encodedAlphabet{*encodedOwner};
  return SharedDictionaryAlphabet::create(
      encodedAlphabet, std::move(encodedOwner), benchmarkPool().get());
}

template <typename T>
void readSharedDictionaryAlphabetScalar(
    const SharedDictionaryAlphabet& alphabet,
    const std::vector<uint32_t>& indices,
    uint32_t iters) {
  typename TypeTraits<T>::physicalType value;
  while (iters--) {
    for (const auto index : indices) {
      value = alphabet.physicalValueAt<T>(index);
      folly::doNotOptimizeAway(value);
    }
  }
}

template <typename T>
void readSharedDictionaryAlphabetBatch(
    const SharedDictionaryAlphabet& alphabet,
    const std::vector<uint32_t>& indices,
    uint32_t iters) {
  using PhysicalType = typename TypeTraits<T>::physicalType;
  std::vector<PhysicalType> output(indices.size());
  while (iters--) {
    alphabet.materialize<T>(indices, output.data());
    folly::doNotOptimizeAway(output.data());
  }
}

template <typename T>
void readRangesWithView(const EncodingView& view, uint32_t iters) {
  auto output =
      std::make_unique<typename TypeTraits<T>::physicalType[]>(kRangeSize);
  while (iters--) {
    for (uint32_t offset = 0; offset < kRows; offset += kRangeSize) {
      view.read(
          offset, std::min<uint32_t>(kRangeSize, kRows - offset), output.get());
    }
    folly::doNotOptimizeAway(output[0]);
  }
}

template <typename T>
void readUnalignedRangesWithView(const EncodingView& view, uint32_t iters) {
  auto output =
      std::make_unique<typename TypeTraits<T>::physicalType[]>(kRangeSize);
  while (iters--) {
    for (uint32_t offset = 17; offset < kRows; offset += kRangeSize) {
      view.read(
          offset, std::min<uint32_t>(kRangeSize, kRows - offset), output.get());
    }
    folly::doNotOptimizeAway(output[0]);
  }
}

template <typename T>
void readPositionsWithMaterialization(
    Encoding& encoding,
    const std::vector<uint32_t>& positions,
    uint32_t iters) {
  T value;
  while (iters--) {
    for (const auto position : positions) {
      encoding.reset();
      encoding.skip(position);
      encoding.materialize(1, &value);
      folly::doNotOptimizeAway(value);
    }
  }
}

std::unique_ptr<Encoding> createRegularEncoding(
    std::string_view encoded,
    std::vector<facebook::velox::BufferPtr>& stringBuffers) {
  auto stringBufferFactory = [&](uint32_t bytes) -> void* {
    auto& buffer = stringBuffers.emplace_back(
        facebook::velox::AlignedBuffer::allocate<char>(
            bytes, benchmarkPool().get()));
    return buffer->asMutable<void>();
  };
  return EncodingFactory{}.create(
      *benchmarkPool(), encoded, stringBufferFactory);
}

// Creates the view of one chunk and reads a single row, as a point lookup does.
template <typename T>
void createAndReadWithView(
    std::string_view encoded,
    const std::vector<uint32_t>& positions,
    uint32_t iters,
    const Encoding::Options& options) {
  typename TypeTraits<T>::physicalType value;
  while (iters--) {
    const auto view =
        createEncodingView(encoded, benchmarkPool().get(), options);
    view->readAt(positions[iters % positions.size()], &value);
    folly::doNotOptimizeAway(value);
  }
}

// Creates the encoding of one chunk, skips to a row and reads it.
template <typename T>
void createAndReadWithSkip(
    std::string_view encoded,
    const std::vector<uint32_t>& positions,
    uint32_t iters) {
  std::vector<facebook::velox::BufferPtr> stringBuffers;
  T value;
  while (iters--) {
    // Drops the string buffers of the previous encoding, which no longer
    // exists.
    stringBuffers.clear();
    const auto encoding = createRegularEncoding(encoded, stringBuffers);
    encoding->skip(positions[iters % positions.size()]);
    encoding->materialize(1, &value);
    folly::doNotOptimizeAway(value);
  }
}

// Creates the view of one chunk and reads one row that may be null. A nullable
// view reads through row ranges because readAt() requires a non-null row.
template <typename T>
void createAndReadNullableWithView(
    std::string_view encoded,
    const std::vector<uint32_t>& positions,
    uint32_t iters,
    const Encoding::Options& options) {
  typename TypeTraits<T>::physicalType value;
  const std::function<void(uint32_t)> setNull = [](uint32_t) {};
  while (iters--) {
    const auto view =
        createEncodingView(encoded, benchmarkPool().get(), options);
    const auto row = positions[iters % positions.size()];
    const RowRange range{row, row + 1};
    folly::doNotOptimizeAway(
        view->read(std::span<const RowRange>{&range, 1}, setNull, &value));
    folly::doNotOptimizeAway(value);
  }
}

// Creates the encoding of one chunk, skips to a row and reads it with its null
// flag.
template <typename T>
void createAndReadNullableWithSkip(
    std::string_view encoded,
    const std::vector<uint32_t>& positions,
    uint32_t iters) {
  std::vector<facebook::velox::BufferPtr> stringBuffers;
  T value;
  uint64_t nulls{0};
  while (iters--) {
    stringBuffers.clear();
    const auto encoding = createRegularEncoding(encoded, stringBuffers);
    encoding->skip(positions[iters % positions.size()]);
    folly::doNotOptimizeAway(encoding->materializeNullable(
        1, &value, [&nulls]() -> void* { return &nulls; }));
    folly::doNotOptimizeAway(value);
  }
}

// Reads one row per iteration from a view that was created once.
template <typename T>
void readWithReusedView(
    const EncodingView& view,
    const std::vector<uint32_t>& positions,
    uint32_t iters) {
  typename TypeTraits<T>::physicalType value;
  while (iters--) {
    view.readAt(positions[iters % positions.size()], &value);
    folly::doNotOptimizeAway(value);
  }
}

// Reads one row that may be null per iteration from a view that was created
// once.
template <typename T>
void readNullableWithReusedView(
    const EncodingView& view,
    const std::vector<uint32_t>& positions,
    uint32_t iters) {
  typename TypeTraits<T>::physicalType value;
  const std::function<void(uint32_t)> setNull = [](uint32_t) {};
  while (iters--) {
    const auto row = positions[iters % positions.size()];
    const RowRange range{row, row + 1};
    folly::doNotOptimizeAway(
        view.read(std::span<const RowRange>{&range, 1}, setNull, &value));
    folly::doNotOptimizeAway(value);
  }
}

// Reads one row per iteration from an encoding that was created once, by
// resetting it and skipping to the row.
template <typename T>
void readWithReusedSkip(
    Encoding& encoding,
    const std::vector<uint32_t>& positions,
    uint32_t iters) {
  T value;
  while (iters--) {
    encoding.reset();
    encoding.skip(positions[iters % positions.size()]);
    encoding.materialize(1, &value);
    folly::doNotOptimizeAway(value);
  }
}

template <typename T>
void readNullableWithReusedSkip(
    Encoding& encoding,
    const std::vector<uint32_t>& positions,
    uint32_t iters) {
  T value;
  uint64_t nulls{0};
  while (iters--) {
    encoding.reset();
    encoding.skip(positions[iters % positions.size()]);
    folly::doNotOptimizeAway(encoding.materializeNullable(
        1, &value, [&nulls]() -> void* { return &nulls; }));
    folly::doNotOptimizeAway(value);
  }
}

class ReadAtOffsetsStringTrivialView {
 public:
  ReadAtOffsetsStringTrivialView(
      std::string_view data,
      facebook::velox::memory::MemoryPool* pool,
      const Encoding::Options& options) {
    rowCount_ = EncodingPrefix::readRowCount(data, options.useVarintRowCount);
    const auto dataOffset =
        EncodingPrefix::prefixSize(data, options.useVarintRowCount);
    const char* pos = data.data() + dataOffset;
    const auto compressionType =
        static_cast<CompressionType>(encoding::readChar(pos));
    NIMBLE_CHECK_EQ(compressionType, CompressionType::Uncompressed);
    const auto lengthsSize = encoding::readUint32(pos);
    auto lengths = detail::createTypedEncodingView<uint32_t>(
        {pos, lengthsSize}, pool, options);
    NIMBLE_CHECK_NOT_NULL(lengths);

    offsets_.reserve(rowCount_ + 1);
    offsets_.push_back(0);
    uint32_t offset = 0;
    for (uint32_t i = 0; i < rowCount_; ++i) {
      offset += lengths->readAt(i);
      offsets_.push_back(offset);
    }
    blob_ = pos + lengthsSize;
  }

  std::string_view readAt(uint32_t index) const {
    return {blob_ + offsets_[index], offsets_[index + 1] - offsets_[index]};
  }

 private:
  uint32_t rowCount_{0};
  const char* blob_{nullptr};
  std::vector<uint32_t> offsets_;
};

class MaterializedOffsetsStringTrivialView {
 public:
  MaterializedOffsetsStringTrivialView(
      std::string_view data,
      facebook::velox::memory::MemoryPool* pool,
      const Encoding::Options& options) {
    rowCount_ = EncodingPrefix::readRowCount(data, options.useVarintRowCount);
    const auto dataOffset =
        EncodingPrefix::prefixSize(data, options.useVarintRowCount);
    const char* pos = data.data() + dataOffset;
    const auto compressionType =
        static_cast<CompressionType>(encoding::readChar(pos));
    NIMBLE_CHECK_EQ(compressionType, CompressionType::Uncompressed);
    const auto lengthsSize = encoding::readUint32(pos);

    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    auto lengths = EncodingFactory().create(
        *pool, {pos, lengthsSize}, noStringBufferFactory, options);
    NIMBLE_CHECK_NOT_NULL(lengths);
    offsets_.resize(rowCount_ + 1);
    offsets_[0] = 0;
    lengths->materialize(rowCount_, offsets_.data() + 1);
    for (uint32_t i = 0; i < rowCount_; ++i) {
      offsets_[i + 1] += offsets_[i];
    }
    blob_ = pos + lengthsSize;
  }

  std::string_view readAt(uint32_t index) const {
    return {blob_ + offsets_[index], offsets_[index + 1] - offsets_[index]};
  }

 private:
  uint32_t rowCount_{0};
  const char* blob_{nullptr};
  std::vector<uint32_t> offsets_;
};

class ReadAtRunEndsRLEView {
 public:
  ReadAtRunEndsRLEView(
      std::string_view data,
      facebook::velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : runEnds_{pool} {
    rowCount_ = EncodingPrefix::readRowCount(data, options.useVarintRowCount);
    const auto dataOffset =
        EncodingPrefix::prefixSize(data, options.useVarintRowCount);
    const char* pos = data.data() + dataOffset;
    const auto runLengthsSize = encoding::readUint32(pos);
    auto runLengths = detail::createTypedEncodingView<uint32_t>(
        {pos, runLengthsSize}, pool, options);
    NIMBLE_CHECK_NOT_NULL(runLengths);

    runEnds_.reserve(runLengths->rowCount());
    uint32_t end = 0;
    for (uint32_t i = 0; i < runLengths->rowCount(); ++i) {
      end += runLengths->readAt(i);
      runEnds_.push_back(end);
    }
    NIMBLE_CHECK_EQ(end, rowCount_);
  }

  uint32_t lastRunEnd() const {
    return runEnds_.back();
  }

 private:
  uint32_t rowCount_{0};
  Vector<uint32_t> runEnds_;
};

class MaterializedRunEndsRLEView {
 public:
  MaterializedRunEndsRLEView(
      std::string_view data,
      facebook::velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : runEnds_{pool} {
    rowCount_ = EncodingPrefix::readRowCount(data, options.useVarintRowCount);
    const auto dataOffset =
        EncodingPrefix::prefixSize(data, options.useVarintRowCount);
    const char* pos = data.data() + dataOffset;
    const auto runLengthsSize = encoding::readUint32(pos);
    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    auto runLengths = EncodingFactory().create(
        *pool, {pos, runLengthsSize}, noStringBufferFactory, options);
    NIMBLE_CHECK_NOT_NULL(runLengths);

    runEnds_.resize(runLengths->rowCount());
    runLengths->materialize(runLengths->rowCount(), runEnds_.data());
    uint32_t end = 0;
    for (auto& runEnd : runEnds_) {
      end += runEnd;
      runEnd = end;
    }
    NIMBLE_CHECK_EQ(end, rowCount_);
  }

  uint32_t lastRunEnd() const {
    return runEnds_.back();
  }

 private:
  uint32_t rowCount_{0};
  Vector<uint32_t> runEnds_;
};

Vector<uint32_t> dictionaryData(uint32_t rows) {
  Vector<uint32_t> data{benchmarkPool().get()};
  data.resize(rows);
  for (uint32_t i = 0; i < rows; ++i) {
    data[i] = (i * 17) % 53;
  }
  return data;
}

template <typename T>
Vector<T> floatingDictionaryData() {
  Vector<T> data{benchmarkPool().get()};
  data.resize(kRows);
  for (uint32_t i = 0; i < kRows; ++i) {
    data[i] = static_cast<T>(static_cast<int32_t>((i * 17) % 53) - 26) /
        static_cast<T>(4);
  }
  return data;
}

template <typename T>
Vector<T> alpData(uint32_t rows) {
  Vector<T> data{benchmarkPool().get()};
  data.resize(rows);
  for (uint32_t i = 0; i < rows; ++i) {
    data[i] = static_cast<T>(static_cast<int32_t>((i * 37) % 2001) - 1000) /
        static_cast<T>(100);
  }
  return data;
}

Vector<uint32_t> rleData(uint32_t rows) {
  Vector<uint32_t> data{benchmarkPool().get()};
  data.resize(rows);
  for (uint32_t i = 0; i < rows; ++i) {
    data[i] = i / 16;
  }
  return data;
}

Vector<uint32_t> pforData(uint32_t rows) {
  Vector<uint32_t> data{benchmarkPool().get()};
  data.resize(rows);
  for (uint32_t i = 0; i < rows; ++i) {
    data[i] = i % 127;
    if (i % 97 == 0) {
      data[i] += 10000;
    }
  }
  return data;
}

Vector<uint32_t> huffmanSkewedData(uint32_t rows) {
  Vector<uint32_t> data{benchmarkPool().get()};
  data.resize(rows);
  for (uint32_t row = 0; row < rows; ++row) {
    data[row] = row % 10 == 0 ? 1 + row % 15 : 0;
  }
  return data;
}

Vector<uint32_t> huffmanUniformData() {
  Vector<uint32_t> data{benchmarkPool().get()};
  data.resize(kRows);
  std::iota(data.begin(), data.end(), 0);
  return data;
}

Vector<uint32_t> mainlyConstantData(uint32_t rows) {
  Vector<uint32_t> data{benchmarkPool().get()};
  data.resize(rows, 7);
  for (uint32_t i = 0; i < rows; i += 17) {
    data[i] = i;
  }
  return data;
}

Vector<bool> boolData(uint32_t rows) {
  Vector<bool> data{benchmarkPool().get()};
  data.resize(rows);
  for (uint32_t i = 0; i < rows; ++i) {
    data[i] = (i % 7) == 0;
  }
  return data;
}

Vector<std::string_view> stringData(
    std::vector<std::string>& backing,
    uint32_t rows) {
  Vector<std::string_view> data{benchmarkPool().get()};
  data.reserve(rows);
  backing.reserve(rows);
  for (uint32_t i = 0; i < rows; ++i) {
    backing.push_back(fmt::format("feature-value-{}", i % 251));
    data.push_back(backing.back());
  }
  return data;
}

Vector<std::string_view> dictionaryStringData(
    std::vector<std::string>& backing,
    uint32_t rows) {
  Vector<std::string_view> data{benchmarkPool().get()};
  data.reserve(rows);
  backing.reserve(128);
  for (uint32_t i = 0; i < 128; ++i) {
    backing.push_back(fmt::format("dict-value-{}", i));
  }
  for (uint32_t i = 0; i < rows; ++i) {
    data.push_back(backing[(i * 17) % backing.size()]);
  }
  return data;
}

std::string encodeNullableView(
    uint32_t nullsPerThousand,
    uint32_t rows,
    const std::optional<CompressionOptions>& compression) {
  Vector<int64_t> nonNullValues{benchmarkPool().get()};
  Vector<bool> isNonNull{benchmarkPool().get(), rows};
  std::mt19937 generator{7'204'113};
  for (uint32_t row{0}; row < rows; ++row) {
    isNonNull[row] = generator() % 1'000 >= nullsPerThousand;
    if (isNonNull[row]) {
      nonNullValues.push_back(static_cast<int64_t>(row) * 17 + 3);
    }
  }
  auto policy = std::make_unique<ManualEncodingSelectionPolicy<int64_t>>(
      std::vector<std::pair<EncodingType, float>>{{EncodingType::Trivial, 1.0}},
      compression,
      std::nullopt);
  Buffer buffer{*benchmarkPool()};
  const auto encoded = EncodingFactory::encodeNullable<int64_t>(
      std::move(policy), nonNullValues, isNonNull, buffer);
  return {encoded.data(), encoded.size()};
}

unsigned readNullableRanges(
    uint32_t nullsPerThousand,
    uint32_t rangeSize,
    unsigned iterations) {
  folly::BenchmarkSuspender suspender;
  const auto encoded =
      encodeNullableView(nullsPerThousand, kRows, CompressionOptions{});
  const auto view = createEncodingView(encoded, benchmarkPool().get());
  std::vector<RowRange> ranges;
  uint32_t numRows{0};
  for (uint32_t row{0}; row < kRows; row += 2 * rangeSize) {
    ranges.emplace_back(row, std::min(row + rangeSize, kRows));
    numRows += ranges.back().numRows();
  }
  Vector<int64_t> output{benchmarkPool().get(), numRows};
  suspender.dismiss();
  for (unsigned iteration{0}; iteration < iterations; ++iteration) {
    const auto numNonNulls = view->read(ranges, [](uint32_t) {}, output.data());
    folly::doNotOptimizeAway(numNonNulls);
    folly::doNotOptimizeAway(output.data());
  }
  suspender.rehire();
  return iterations;
}

// Reads rows spread evenly over a stream through a view without
// cacheEncodingView. A few rows are cheaper to read one by one, and many rows
// are cheaper to read through one pass over the stream.
unsigned readNullableSparseIndices(
    uint32_t numRows,
    uint32_t numSelected,
    unsigned iterations) {
  folly::BenchmarkSuspender suspender;
  const auto encoded = encodeNullableView(
      /*nullsPerThousand=*/100, numRows, CompressionOptions{});
  const auto view = createEncodingView(encoded, benchmarkPool().get());
  const auto stride = numRows / numSelected;
  std::vector<uint32_t> indices(numSelected);
  for (uint32_t i{0}; i < numSelected; ++i) {
    indices[i] = i * stride + (i * 7) % stride;
  }
  Vector<int64_t> output{benchmarkPool().get(), numSelected};
  suspender.dismiss();
  for (unsigned iteration{0}; iteration < iterations; ++iteration) {
    const auto numNonNulls =
        view->read(indices, [](uint32_t) {}, output.data());
    folly::doNotOptimizeAway(numNonNulls);
    folly::doNotOptimizeAway(output.data());
  }
  suspender.rehire();
  return iterations;
}

// Reads ranges of 'rangeSize' rows spread evenly over a stream, as above.
unsigned readNullableSparseRanges(
    uint32_t numRows,
    uint32_t numRanges,
    uint32_t rangeSize,
    unsigned iterations) {
  folly::BenchmarkSuspender suspender;
  const auto encoded = encodeNullableView(
      /*nullsPerThousand=*/100, numRows, CompressionOptions{});
  const auto view = createEncodingView(encoded, benchmarkPool().get());
  const auto stride = numRows / numRanges;
  std::vector<RowRange> ranges;
  for (uint32_t i{0}; i < numRanges; ++i) {
    ranges.emplace_back(i * stride, i * stride + rangeSize);
  }
  Vector<int64_t> output{benchmarkPool().get(), numRanges * rangeSize};
  suspender.dismiss();
  for (unsigned iteration{0}; iteration < iterations; ++iteration) {
    const auto numNonNulls = view->read(ranges, [](uint32_t) {}, output.data());
    folly::doNotOptimizeAway(numNonNulls);
    folly::doNotOptimizeAway(output.data());
  }
  suspender.rehire();
  return iterations;
}

unsigned createNullableViews(uint32_t nullsPerThousand, unsigned iterations) {
  folly::BenchmarkSuspender suspender;
  const auto encoded =
      encodeNullableView(nullsPerThousand, kRows, CompressionOptions{});
  suspender.dismiss();
  for (unsigned iteration{0}; iteration < iterations; ++iteration) {
    const auto view = createEncodingView(encoded, benchmarkPool().get());
    folly::doNotOptimizeAway(view.get());
  }
  suspender.rehire();
  return iterations;
}

void registerNullableViewBenchmarks() {
  for (const uint32_t numRows : {4'096, 65'536}) {
    for (const uint32_t numSelected :
         {4, 8, 16, 32, 64, 96, 128, 192, 256, 512, 1'024}) {
      folly::addBenchmark(
          __FILE__,
          fmt::format("nullableSparseIndices_rows{}_n{}", numRows, numSelected),
          [numRows, numSelected](unsigned iterations) {
            return readNullableSparseIndices(numRows, numSelected, iterations);
          });
    }
    for (const uint32_t rangeSize : {16, 128, 1'024}) {
      for (const uint32_t numRanges : {2, 4, 8, 16, 32, 64, 128, 256}) {
        if (numRanges * rangeSize > numRows) {
          continue;
        }
        folly::addBenchmark(
            __FILE__,
            fmt::format(
                "nullableSparseRanges_rows{}_n{}_len{}",
                numRows,
                numRanges,
                rangeSize),
            [numRows, numRanges, rangeSize](unsigned iterations) {
              return readNullableSparseRanges(
                  numRows, numRanges, rangeSize, iterations);
            });
      }
    }
  }
  for (const uint32_t nullsPerThousand :
       {0, 2, 10, 40, 120, 250, 330, 500, 950, 1'000}) {
    folly::addBenchmark(
        __FILE__,
        fmt::format("nullableViewCreate_nulls{}", nullsPerThousand),
        [nullsPerThousand](unsigned iterations) {
          return createNullableViews(nullsPerThousand, iterations);
        });
    for (const uint32_t rangeSize : {16, 1'024}) {
      folly::addBenchmark(
          __FILE__,
          fmt::format(
              "nullableViewRange{}_nulls{}", rangeSize, nullsPerThousand),
          [nullsPerThousand, rangeSize](unsigned iterations) {
            return readNullableRanges(nullsPerThousand, rangeSize, iterations);
          });
    }
  }
}

#define VIEW_BENCHMARK(Name, Type, EncodedExpr, PositionsExpr)   \
  BENCHMARK(Name, iters) {                                       \
    std::string encoded;                                         \
    std::vector<uint32_t> positions;                             \
    std::unique_ptr<EncodingView> view;                          \
    BENCHMARK_SUSPEND {                                          \
      encoded = EncodedExpr;                                     \
      positions = PositionsExpr;                                 \
      view = createEncodingView(encoded, benchmarkPool().get()); \
    }                                                            \
    readPositionsWithView<Type>(*view, positions, iters);        \
  }

#define RANGE_VIEW_BENCHMARK(Name, Type, EncodedExpr)            \
  BENCHMARK(Name, iters) {                                       \
    std::string encoded;                                         \
    std::unique_ptr<EncodingView> view;                          \
    BENCHMARK_SUSPEND {                                          \
      encoded = EncodedExpr;                                     \
      view = createEncodingView(encoded, benchmarkPool().get()); \
    }                                                            \
    readRangesWithView<Type>(*view, iters);                      \
  }

#define BATCH_VIEW_BENCHMARK(Name, Type, EncodedExpr, PositionsExpr) \
  BENCHMARK(Name, iters) {                                           \
    std::string encoded;                                             \
    std::vector<uint32_t> positions;                                 \
    std::unique_ptr<EncodingView> view;                              \
    BENCHMARK_SUSPEND {                                              \
      encoded = EncodedExpr;                                         \
      positions = PositionsExpr;                                     \
      view = createEncodingView(encoded, benchmarkPool().get());     \
    }                                                                \
    readPositionsBatchWithView<Type>(*view, positions, iters);       \
  }

#define SHARED_DICTIONARY_ALPHABET_BENCHMARK(                      \
    Name, Type, ValuesExpr, IndicesExpr, ReadFunction)             \
  BENCHMARK(Name, iters) {                                         \
    std::shared_ptr<const SharedDictionaryAlphabet> alphabet;      \
    std::vector<uint32_t> indices;                                 \
    BENCHMARK_SUSPEND {                                            \
      alphabet = createSharedDictionaryAlphabet<Type>(ValuesExpr); \
      indices = IndicesExpr;                                       \
    }                                                              \
    ReadFunction<Type>(*alphabet, indices, iters);                 \
  }

#define UNALIGNED_RANGE_VIEW_BENCHMARK(Name, Type, EncodedExpr)  \
  BENCHMARK(Name, iters) {                                       \
    std::string encoded;                                         \
    std::unique_ptr<EncodingView> view;                          \
    BENCHMARK_SUSPEND {                                          \
      encoded = EncodedExpr;                                     \
      view = createEncodingView(encoded, benchmarkPool().get()); \
    }                                                            \
    readUnalignedRangesWithView<Type>(*view, iters);             \
  }

#define MATERIALIZE_BENCHMARK(Name, Type, EncodedExpr, PositionsExpr)    \
  BENCHMARK(Name, iters) {                                               \
    std::string encoded;                                                 \
    std::vector<uint32_t> positions;                                     \
    std::vector<facebook::velox::BufferPtr> stringBuffers;               \
    std::unique_ptr<Encoding> encoding;                                  \
    BENCHMARK_SUSPEND {                                                  \
      encoded = EncodedExpr;                                             \
      positions = PositionsExpr;                                         \
      encoding = createRegularEncoding(encoded, stringBuffers);          \
    }                                                                    \
    readPositionsWithMaterialization<Type>(*encoding, positions, iters); \
  }

VIEW_BENCHMARK(
    View_TrivialUint32_Random130,
    uint32_t,
    encodeWithSelection<TrivialEncoding<uint32_t>>(
        EncodingType::Trivial,
        makeRandom<uint32_t>(kRows)),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_TrivialUint32_Range1024,
    uint32_t,
    encodeWithSelection<TrivialEncoding<uint32_t>>(
        EncodingType::Trivial,
        makeRandom<uint32_t>(kRows)))
VIEW_BENCHMARK(
    View_TrivialBool_Random130,
    bool,
    encodeWithSelection<TrivialEncoding<bool>>(
        EncodingType::Trivial,
        boolData(kRows)),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_TrivialBool_Range1024,
    bool,
    encodeWithSelection<TrivialEncoding<bool>>(
        EncodingType::Trivial,
        boolData(kRows)))
MATERIALIZE_BENCHMARK(
    Materialize_TrivialUint32_Random130,
    uint32_t,
    encodeWithSelection<TrivialEncoding<uint32_t>>(
        EncodingType::Trivial,
        makeRandom<uint32_t>(kRows)),
    randomPositions(kRows))
MATERIALIZE_BENCHMARK(
    Materialize_TrivialBool_Random130,
    bool,
    encodeWithSelection<TrivialEncoding<bool>>(
        EncodingType::Trivial,
        boolData(kRows)),
    randomPositions(kRows))
VIEW_BENCHMARK(
    View_SparseBool_Random130,
    bool,
    encodeWithSelection<SparseBoolEncoding>(
        EncodingType::SparseBool,
        boolData(kRows)),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_SparseBool_Range1024,
    bool,
    encodeWithSelection<SparseBoolEncoding>(
        EncodingType::SparseBool,
        boolData(kRows)))
MATERIALIZE_BENCHMARK(
    Materialize_SparseBool_Random130,
    bool,
    encodeWithSelection<SparseBoolEncoding>(
        EncodingType::SparseBool,
        boolData(kRows)),
    randomPositions(kRows))

BENCHMARK(View_TrivialStringEager_Random130, iters) {
  std::vector<std::string> backing;
  std::string encoded;
  std::vector<uint32_t> positions;
  std::unique_ptr<EncodingView> view;
  BENCHMARK_SUSPEND {
    encoded = encodeTrivialString(stringData(backing, kRows));
    positions = randomPositions(kRows);
    view = createEncodingView(encoded, benchmarkPool().get());
  }
  readPositionsWithView<std::string_view>(*view, positions, iters);
}

BENCHMARK(RangeView_TrivialStringEager_Range1024, iters) {
  std::vector<std::string> backing;
  std::string encoded;
  std::unique_ptr<EncodingView> view;
  BENCHMARK_SUSPEND {
    encoded = encodeTrivialString(stringData(backing, kRows));
    view = createEncodingView(encoded, benchmarkPool().get());
  }
  readRangesWithView<std::string_view>(*view, iters);
}

BENCHMARK(Construct_TrivialStringOffsets_ReadAt, iters) {
  std::vector<std::string> backing;
  std::string encoded;
  BENCHMARK_SUSPEND {
    encoded = encodeTrivialString(stringData(backing, kRows));
  }
  while (iters--) {
    ReadAtOffsetsStringTrivialView view{
        encoded, benchmarkPool().get(), Encoding::Options{}};
    folly::doNotOptimizeAway(view.readAt(kRows / 2));
  }
}

BENCHMARK(Construct_TrivialStringOffsets_Materialize, iters) {
  std::vector<std::string> backing;
  std::string encoded;
  BENCHMARK_SUSPEND {
    encoded = encodeTrivialString(stringData(backing, kRows));
  }
  while (iters--) {
    MaterializedOffsetsStringTrivialView view{
        encoded, benchmarkPool().get(), Encoding::Options{}};
    folly::doNotOptimizeAway(view.readAt(kRows / 2));
  }
}

BENCHMARK(Materialize_TrivialString_Random130, iters) {
  std::vector<std::string> backing;
  std::string encoded;
  std::vector<uint32_t> positions;
  std::vector<facebook::velox::BufferPtr> stringBuffers;
  std::unique_ptr<Encoding> encoding;
  BENCHMARK_SUSPEND {
    encoded = encodeTrivialString(stringData(backing, kRows));
    positions = randomPositions(kRows);
    encoding = createRegularEncoding(encoded, stringBuffers);
  }
  readPositionsWithMaterialization<std::string_view>(
      *encoding, positions, iters);
}

BENCHMARK(View_TrivialStringEager_Sorted130, iters) {
  std::vector<std::string> backing;
  std::string encoded;
  std::vector<uint32_t> positions;
  std::unique_ptr<EncodingView> view;
  BENCHMARK_SUSPEND {
    encoded = encodeTrivialString(stringData(backing, kRows));
    positions = sortedRandomPositions(kRows);
    view = createEncodingView(encoded, benchmarkPool().get());
  }
  readPositionsWithView<std::string_view>(*view, positions, iters);
}

BENCHMARK(View_TrivialStringEager_Clustered130, iters) {
  std::vector<std::string> backing;
  std::string encoded;
  std::vector<uint32_t> positions;
  std::unique_ptr<EncodingView> view;
  BENCHMARK_SUSPEND {
    encoded = encodeTrivialString(stringData(backing, kRows));
    positions = clusteredPositions(kRows);
    view = createEncodingView(encoded, benchmarkPool().get());
  }
  readPositionsWithView<std::string_view>(*view, positions, iters);
}

BENCHMARK(View_FsstString_Sorted130, iters) {
  std::vector<std::string> backing;
  std::string encoded;
  std::vector<uint32_t> positions;
  std::unique_ptr<EncodingView> view;
  BENCHMARK_SUSPEND {
    encoded = encodeFsstString(stringData(backing, kRows));
    positions = sortedRandomPositions(kRows);
    view = createEncodingView(encoded, benchmarkPool().get());
  }
  readPositionsWithView<std::string_view>(*view, positions, iters);
}

BENCHMARK(BatchView_FsstString_Sorted130, iters) {
  std::vector<std::string> backing;
  std::string encoded;
  std::vector<uint32_t> positions;
  std::unique_ptr<EncodingView> view;
  BENCHMARK_SUSPEND {
    encoded = encodeFsstString(stringData(backing, kRows));
    positions = sortedRandomPositions(kRows);
    view = createEncodingView(encoded, benchmarkPool().get());
  }
  readPositionsBatchWithView<std::string_view>(*view, positions, iters);
}

BENCHMARK(RangeView_FsstString_Range1024, iters) {
  std::vector<std::string> backing;
  std::string encoded;
  std::unique_ptr<EncodingView> view;
  BENCHMARK_SUSPEND {
    encoded = encodeFsstString(stringData(backing, kRows));
    view = createEncodingView(encoded, benchmarkPool().get());
  }
  readRangesWithView<std::string_view>(*view, iters);
}

VIEW_BENCHMARK(
    View_ConstantUint32_Random130,
    uint32_t,
    encodeWithSelection<ConstantEncoding<uint32_t>>(
        EncodingType::Constant,
        makeConstant<uint32_t>(42, kRows)),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_ConstantUint32_Range1024,
    uint32_t,
    encodeWithSelection<ConstantEncoding<uint32_t>>(
        EncodingType::Constant,
        makeConstant<uint32_t>(42, kRows)))
MATERIALIZE_BENCHMARK(
    Materialize_ConstantUint32_Random130,
    uint32_t,
    encodeWithSelection<ConstantEncoding<uint32_t>>(
        EncodingType::Constant,
        makeConstant<uint32_t>(42, kRows)),
    randomPositions(kRows))
VIEW_BENCHMARK(
    View_FixedBitWidthUint32_Random130,
    uint32_t,
    encodeWithSelection<FixedBitWidthEncoding<uint32_t>>(
        EncodingType::FixedBitWidth,
        makeNarrow<uint32_t>(10, kRows)),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_FixedBitWidthUint32_Range1024,
    uint32_t,
    encodeWithSelection<FixedBitWidthEncoding<uint32_t>>(
        EncodingType::FixedBitWidth,
        makeNarrow<uint32_t>(10, kRows)))
MATERIALIZE_BENCHMARK(
    Materialize_FixedBitWidthUint32_Random130,
    uint32_t,
    encodeWithSelection<FixedBitWidthEncoding<uint32_t>>(
        EncodingType::FixedBitWidth,
        makeNarrow<uint32_t>(10, kRows)),
    randomPositions(kRows))
VIEW_BENCHMARK(
    View_MainlyConstantUint32_Random130,
    uint32_t,
    encodeWithSelection<MainlyConstantEncoding<uint32_t>>(
        EncodingType::MainlyConstant,
        mainlyConstantData(kRows)),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_MainlyConstantUint32_Range1024,
    uint32_t,
    encodeWithSelection<MainlyConstantEncoding<uint32_t>>(
        EncodingType::MainlyConstant,
        mainlyConstantData(kRows)))
MATERIALIZE_BENCHMARK(
    Materialize_MainlyConstantUint32_Random130,
    uint32_t,
    encodeWithSelection<MainlyConstantEncoding<uint32_t>>(
        EncodingType::MainlyConstant,
        mainlyConstantData(kRows)),
    randomPositions(kRows))
VIEW_BENCHMARK(
    View_DictionaryUint32_Random130,
    uint32_t,
    encodeWithSelection<DictionaryEncoding<uint32_t>>(
        EncodingType::Dictionary,
        dictionaryData(kRows)),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_DictionaryUint32_Range1024,
    uint32_t,
    encodeWithSelection<DictionaryEncoding<uint32_t>>(
        EncodingType::Dictionary,
        dictionaryData(kRows)))
VIEW_BENCHMARK(
    View_DictionaryFloat_Random130,
    float,
    encodeWithSelection<DictionaryEncoding<float>>(
        EncodingType::Dictionary,
        floatingDictionaryData<float>()),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_DictionaryFloat_Range1024,
    float,
    encodeWithSelection<DictionaryEncoding<float>>(
        EncodingType::Dictionary,
        floatingDictionaryData<float>()))
VIEW_BENCHMARK(
    View_DictionaryDouble_Random130,
    double,
    encodeWithSelection<DictionaryEncoding<double>>(
        EncodingType::Dictionary,
        floatingDictionaryData<double>()),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_DictionaryDouble_Range1024,
    double,
    encodeWithSelection<DictionaryEncoding<double>>(
        EncodingType::Dictionary,
        floatingDictionaryData<double>()))
BENCHMARK(View_DictionaryString_Random130, iters) {
  std::vector<std::string> backing;
  std::string encoded;
  std::vector<uint32_t> positions;
  std::unique_ptr<EncodingView> view;
  BENCHMARK_SUSPEND {
    encoded = encodeWithSelection<DictionaryEncoding<std::string_view>>(
        EncodingType::Dictionary, dictionaryStringData(backing, kRows));
    positions = randomPositions(kRows);
    view = createEncodingView(encoded, benchmarkPool().get());
  }
  readPositionsWithView<std::string_view>(*view, positions, iters);
}
BENCHMARK(RangeView_DictionaryString_Range1024, iters) {
  std::vector<std::string> backing;
  std::string encoded;
  std::unique_ptr<EncodingView> view;
  BENCHMARK_SUSPEND {
    encoded = encodeWithSelection<DictionaryEncoding<std::string_view>>(
        EncodingType::Dictionary, dictionaryStringData(backing, kRows));
    view = createEncodingView(encoded, benchmarkPool().get());
  }
  readRangesWithView<std::string_view>(*view, iters);
}
MATERIALIZE_BENCHMARK(
    Materialize_DictionaryUint32_Random130,
    uint32_t,
    encodeWithSelection<DictionaryEncoding<uint32_t>>(
        EncodingType::Dictionary,
        dictionaryData(kRows)),
    randomPositions(kRows))
MATERIALIZE_BENCHMARK(
    Materialize_DictionaryFloat_Random130,
    float,
    encodeWithSelection<DictionaryEncoding<float>>(
        EncodingType::Dictionary,
        floatingDictionaryData<float>()),
    randomPositions(kRows))
MATERIALIZE_BENCHMARK(
    Materialize_DictionaryDouble_Random130,
    double,
    encodeWithSelection<DictionaryEncoding<double>>(
        EncodingType::Dictionary,
        floatingDictionaryData<double>()),
    randomPositions(kRows))
BENCHMARK(Materialize_DictionaryString_Random130, iters) {
  std::vector<std::string> backing;
  std::string encoded;
  std::vector<uint32_t> positions;
  std::vector<facebook::velox::BufferPtr> stringBuffers;
  std::unique_ptr<Encoding> encoding;
  BENCHMARK_SUSPEND {
    encoded = encodeWithSelection<DictionaryEncoding<std::string_view>>(
        EncodingType::Dictionary, dictionaryStringData(backing, kRows));
    positions = randomPositions(kRows);
    encoding = createRegularEncoding(encoded, stringBuffers);
  }
  readPositionsWithMaterialization<std::string_view>(
      *encoding, positions, iters);
}
VIEW_BENCHMARK(
    View_RLEUint32_Random130,
    uint32_t,
    encodeWithSelection<RLEEncoding<uint32_t>>(
        EncodingType::RLE,
        rleData(kRows)),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_RLEUint32_Range1024,
    uint32_t,
    encodeWithSelection<RLEEncoding<uint32_t>>(
        EncodingType::RLE,
        rleData(kRows)))
MATERIALIZE_BENCHMARK(
    Materialize_RLEUint32_Random130,
    uint32_t,
    encodeWithSelection<RLEEncoding<uint32_t>>(
        EncodingType::RLE,
        rleData(kRows)),
    randomPositions(kRows))
BENCHMARK(Construct_RLERunEnds_ReadAt, iters) {
  std::string encoded;
  BENCHMARK_SUSPEND {
    encoded = encodeWithSelection<RLEEncoding<uint32_t>>(
        EncodingType::RLE, rleData(kRows));
  }
  while (iters--) {
    ReadAtRunEndsRLEView view{
        encoded, benchmarkPool().get(), Encoding::Options{}};
    folly::doNotOptimizeAway(view.lastRunEnd());
  }
}
BENCHMARK(Construct_RLERunEnds_Materialize, iters) {
  std::string encoded;
  BENCHMARK_SUSPEND {
    encoded = encodeWithSelection<RLEEncoding<uint32_t>>(
        EncodingType::RLE, rleData(kRows));
  }
  while (iters--) {
    MaterializedRunEndsRLEView view{
        encoded, benchmarkPool().get(), Encoding::Options{}};
    folly::doNotOptimizeAway(view.lastRunEnd());
  }
}
VIEW_BENCHMARK(
    View_ALPFloat_Random130,
    float,
    encodeWithSelection<ALPEncoding<float>>(
        EncodingType::ALP,
        alpData<float>(kRows)),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_ALPFloat_Range1024,
    float,
    encodeWithSelection<ALPEncoding<float>>(
        EncodingType::ALP,
        alpData<float>(kRows)))
VIEW_BENCHMARK(
    View_ALPDouble_Random130,
    double,
    encodeWithSelection<ALPEncoding<double>>(
        EncodingType::ALP,
        alpData<double>(kRows)),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_ALPDouble_Range1024,
    double,
    encodeWithSelection<ALPEncoding<double>>(
        EncodingType::ALP,
        alpData<double>(kRows)))
MATERIALIZE_BENCHMARK(
    Materialize_ALPFloat_Random130,
    float,
    encodeWithSelection<ALPEncoding<float>>(
        EncodingType::ALP,
        alpData<float>(kRows)),
    randomPositions(kRows))
MATERIALIZE_BENCHMARK(
    Materialize_ALPDouble_Random130,
    double,
    encodeWithSelection<ALPEncoding<double>>(
        EncodingType::ALP,
        alpData<double>(kRows)),
    randomPositions(kRows))
VIEW_BENCHMARK(
    View_FORUint32_Random130,
    uint32_t,
    encodeWithSelection<ForEncoding<uint32_t>>(
        EncodingType::FOR,
        pforData(kRows)),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_FORUint32_Range1024,
    uint32_t,
    encodeWithSelection<ForEncoding<uint32_t>>(
        EncodingType::FOR,
        pforData(kRows)))
BENCHMARK(Materialize_FORUint32_Random130, iters) {
  std::string encoded;
  std::vector<uint32_t> positions;
  std::unique_ptr<ForEncoding<uint32_t>> encoding;
  BENCHMARK_SUSPEND {
    encoded = encodeWithSelection<ForEncoding<uint32_t>>(
        EncodingType::FOR, pforData(kRows));
    positions = randomPositions(kRows);
    encoding = std::make_unique<ForEncoding<uint32_t>>(
        *benchmarkPool(), encoded, nullptr, Encoding::Options{});
  }
  readPositionsWithMaterialization<uint32_t>(*encoding, positions, iters);
}
VIEW_BENCHMARK(
    View_PFORUint32_Random130,
    uint32_t,
    encodeWithSelection<PFOREncoding<uint32_t>>(
        EncodingType::PFOR,
        pforData(kRows)),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_PFORUint32_Range1024,
    uint32_t,
    encodeWithSelection<PFOREncoding<uint32_t>>(
        EncodingType::PFOR,
        pforData(kRows)))
MATERIALIZE_BENCHMARK(
    Materialize_PFORUint32_Random130,
    uint32_t,
    encodeWithSelection<PFOREncoding<uint32_t>>(
        EncodingType::PFOR,
        pforData(kRows)),
    randomPositions(kRows))
VIEW_BENCHMARK(
    View_HuffmanUint32_SkewedRandom130,
    uint32_t,
    encodeWithSelection<HuffmanEncoding<uint32_t>>(
        EncodingType::Huffman,
        huffmanSkewedData(kRows)),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_HuffmanUint32_SkewedRange1024,
    uint32_t,
    encodeWithSelection<HuffmanEncoding<uint32_t>>(
        EncodingType::Huffman,
        huffmanSkewedData(kRows)))
MATERIALIZE_BENCHMARK(
    Materialize_HuffmanUint32_SkewedRandom130,
    uint32_t,
    encodeWithSelection<HuffmanEncoding<uint32_t>>(
        EncodingType::Huffman,
        huffmanSkewedData(kRows)),
    randomPositions(kRows))
VIEW_BENCHMARK(
    View_HuffmanUint32_UniformRandom130,
    uint32_t,
    encodeWithSelection<HuffmanEncoding<uint32_t>>(
        EncodingType::Huffman,
        huffmanUniformData()),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_HuffmanUint32_UniformRange1024,
    uint32_t,
    encodeWithSelection<HuffmanEncoding<uint32_t>>(
        EncodingType::Huffman,
        huffmanUniformData()))
MATERIALIZE_BENCHMARK(
    Materialize_HuffmanUint32_UniformRandom130,
    uint32_t,
    encodeWithSelection<HuffmanEncoding<uint32_t>>(
        EncodingType::Huffman,
        huffmanUniformData()),
    randomPositions(kRows))
VIEW_BENCHMARK(
    View_SimdForBitpackUint32_Random130,
    uint32_t,
    encodeWithSelection<SimdForBitpackEncoding<uint32_t>>(
        EncodingType::SimdForBitpack,
        makeNarrow<uint32_t>(10, kRows)),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_SimdForBitpackUint32_Range1024,
    uint32_t,
    encodeWithSelection<SimdForBitpackEncoding<uint32_t>>(
        EncodingType::SimdForBitpack,
        makeNarrow<uint32_t>(10, kRows)))
MATERIALIZE_BENCHMARK(
    Materialize_SimdForBitpackUint32_Random130,
    uint32_t,
    encodeWithSelection<SimdForBitpackEncoding<uint32_t>>(
        EncodingType::SimdForBitpack,
        makeNarrow<uint32_t>(10, kRows)),
    randomPositions(kRows))
VIEW_BENCHMARK(
    View_SimdForBitpackUint16_Random130,
    uint16_t,
    encodeWithSelection<SimdForBitpackEncoding<uint16_t>>(
        EncodingType::SimdForBitpack,
        makeNarrow<uint16_t>(10, kRows)),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_SimdForBitpackUint16_Range1024,
    uint16_t,
    encodeWithSelection<SimdForBitpackEncoding<uint16_t>>(
        EncodingType::SimdForBitpack,
        makeNarrow<uint16_t>(10, kRows)))
MATERIALIZE_BENCHMARK(
    Materialize_SimdForBitpackUint16_Random130,
    uint16_t,
    encodeWithSelection<SimdForBitpackEncoding<uint16_t>>(
        EncodingType::SimdForBitpack,
        makeNarrow<uint16_t>(10, kRows)),
    randomPositions(kRows))
VIEW_BENCHMARK(
    View_SimdForBitpackUint8_Random130,
    uint8_t,
    encodeWithSelection<SimdForBitpackEncoding<uint8_t>>(
        EncodingType::SimdForBitpack,
        makeNarrow<uint8_t>(10, kRows)),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_SimdForBitpackUint8_Range1024,
    uint8_t,
    encodeWithSelection<SimdForBitpackEncoding<uint8_t>>(
        EncodingType::SimdForBitpack,
        makeNarrow<uint8_t>(10, kRows)))
MATERIALIZE_BENCHMARK(
    Materialize_SimdForBitpackUint8_Random130,
    uint8_t,
    encodeWithSelection<SimdForBitpackEncoding<uint8_t>>(
        EncodingType::SimdForBitpack,
        makeNarrow<uint8_t>(10, kRows)),
    randomPositions(kRows))
VIEW_BENCHMARK(
    View_BlockBitPackingUint32_Random130,
    uint32_t,
    encodeWithSelection<BlockBitPackingV2Encoding<uint32_t>>(
        EncodingType::BlockBitPackingV2,
        pforData(kRows)),
    randomPositions(kRows))
VIEW_BENCHMARK(
    View_BlockBitPackingUint32_Sorted130,
    uint32_t,
    encodeWithSelection<BlockBitPackingV2Encoding<uint32_t>>(
        EncodingType::BlockBitPackingV2,
        pforData(kRows)),
    sortedRandomPositions(kRows))
VIEW_BENCHMARK(
    View_BlockBitPackingUint32_Clustered130,
    uint32_t,
    encodeWithSelection<BlockBitPackingV2Encoding<uint32_t>>(
        EncodingType::BlockBitPackingV2,
        pforData(kRows)),
    clusteredPositions(kRows))
BATCH_VIEW_BENCHMARK(
    BatchView_BlockBitPackingUint32_Random130,
    uint32_t,
    encodeWithSelection<BlockBitPackingV2Encoding<uint32_t>>(
        EncodingType::BlockBitPackingV2,
        pforData(kRows)),
    randomPositions(kRows))
BATCH_VIEW_BENCHMARK(
    BatchView_BlockBitPackingUint32_Sorted130,
    uint32_t,
    encodeWithSelection<BlockBitPackingV2Encoding<uint32_t>>(
        EncodingType::BlockBitPackingV2,
        pforData(kRows)),
    sortedRandomPositions(kRows))
BATCH_VIEW_BENCHMARK(
    BatchView_BlockBitPackingUint32_Clustered130,
    uint32_t,
    encodeWithSelection<BlockBitPackingV2Encoding<uint32_t>>(
        EncodingType::BlockBitPackingV2,
        pforData(kRows)),
    clusteredPositions(kRows))
SHARED_DICTIONARY_ALPHABET_BENCHMARK(
    SharedDictionary_BBP_Uint32_ScalarRandom130,
    uint32_t,
    huffmanUniformData(),
    sharedDictionaryRandomIndices(),
    readSharedDictionaryAlphabetScalar)
SHARED_DICTIONARY_ALPHABET_BENCHMARK(
    SharedDictionary_BBP_Uint32_BatchRandom130,
    uint32_t,
    huffmanUniformData(),
    sharedDictionaryRandomIndices(),
    readSharedDictionaryAlphabetBatch)
SHARED_DICTIONARY_ALPHABET_BENCHMARK(
    SharedDictionary_BBP_Uint32_ScalarSorted130,
    uint32_t,
    huffmanUniformData(),
    sharedDictionarySortedIndices(),
    readSharedDictionaryAlphabetScalar)
SHARED_DICTIONARY_ALPHABET_BENCHMARK(
    SharedDictionary_BBP_Uint32_BatchSorted130,
    uint32_t,
    huffmanUniformData(),
    sharedDictionarySortedIndices(),
    readSharedDictionaryAlphabetBatch)
SHARED_DICTIONARY_ALPHABET_BENCHMARK(
    SharedDictionary_BBP_Uint32_ScalarClustered130,
    uint32_t,
    huffmanUniformData(),
    sharedDictionaryClusteredIndices(),
    readSharedDictionaryAlphabetScalar)
SHARED_DICTIONARY_ALPHABET_BENCHMARK(
    SharedDictionary_BBP_Uint32_BatchClustered130,
    uint32_t,
    huffmanUniformData(),
    sharedDictionaryClusteredIndices(),
    readSharedDictionaryAlphabetBatch)
RANGE_VIEW_BENCHMARK(
    RangeView_BlockBitPackingUint32_Range1024,
    uint32_t,
    encodeWithSelection<BlockBitPackingV2Encoding<uint32_t>>(
        EncodingType::BlockBitPackingV2,
        pforData(kRows)))
UNALIGNED_RANGE_VIEW_BENCHMARK(
    RangeView_BlockBitPackingUint32_UnalignedRange1024,
    uint32_t,
    encodeWithSelection<BlockBitPackingV2Encoding<uint32_t>>(
        EncodingType::BlockBitPackingV2,
        pforData(kRows)))
UNALIGNED_RANGE_VIEW_BENCHMARK(
    RangeView_BlockBitPackingUint16_UnalignedRange1024,
    uint16_t,
    encodeWithSelection<BlockBitPackingV2Encoding<uint16_t>>(
        EncodingType::BlockBitPackingV2,
        makeNarrow<uint16_t>(10, kRows)))
MATERIALIZE_BENCHMARK(
    Materialize_BlockBitPackingUint32_Random130,
    uint32_t,
    encodeWithSelection<BlockBitPackingV2Encoding<uint32_t>>(
        EncodingType::BlockBitPackingV2,
        pforData(kRows)),
    randomPositions(kRows))

constexpr uint32_t kLargeRows = 16'384;

using CreateReadFunction = std::function<void(
    std::string_view encoded,
    const std::vector<uint32_t>& positions,
    uint32_t iterations)>;

using CreateReadViewFunction = std::function<void(
    std::string_view encoded,
    const std::vector<uint32_t>& positions,
    uint32_t iterations,
    const Encoding::Options& options)>;

// One encoding for the lifecycle of a point lookup: create the reader of an
// encoded chunk and read one row, once with the encoding view and once with the
// encoding's skip() and materialize(). The skip variant is empty for an
// encoding that EncodingFactory cannot create.
struct CreateReadCase {
  std::string name;
  std::function<std::string(uint32_t rows)> encode;
  CreateReadViewFunction createAndReadWithView;
  CreateReadFunction createAndReadWithSkip;
  // Read one row per iteration from a view or an encoding that was created
  // once, which shows the cost of a read without the cost of creating the
  // reader.
  std::function<
      void(const EncodingView&, const std::vector<uint32_t>&, uint32_t)>
      readWithReusedView;
  std::function<void(Encoding&, const std::vector<uint32_t>&, uint32_t)>
      readWithReusedSkip;
};

template <typename T>
CreateReadCase makeCreateReadCase(
    std::string name,
    std::function<std::string(uint32_t rows)> encode) {
  return {
      std::move(name),
      std::move(encode),
      createAndReadWithView<T>,
      createAndReadWithSkip<T>,
      readWithReusedView<T>,
      readWithReusedSkip<T>};
}

template <typename T>
CreateReadCase makeNullableCreateReadCase(
    std::string name,
    std::function<std::string(uint32_t rows)> encode) {
  return {
      std::move(name),
      std::move(encode),
      createAndReadNullableWithView<T>,
      createAndReadNullableWithSkip<T>,
      readNullableWithReusedView<T>,
      readNullableWithReusedSkip<T>};
}

// Encodes a chunk whose reserved value marks 10% of the rows as null.
std::string encodeSentinelChunk(uint32_t rows) {
  constexpr uint32_t kSentinel{0xFFFF'FFFFU};
  constexpr uint32_t kNullPercent{10};
  Vector<uint32_t> values{benchmarkPool().get(), rows};
  std::mt19937 generator{1};
  uint32_t numNulls{0};
  for (uint32_t row{0}; row < rows; ++row) {
    const bool isNull = generator() % 100 < kNullPercent;
    values[row] = isNull ? kSentinel : row + 1;
    numNulls += isNull ? 1 : 0;
  }

  const std::span<const uint32_t> physicalValues{values.data(), values.size()};
  EncodingSelection<uint32_t> selection{
      EncodingSelectionResult{.encodingType = EncodingType::Trivial},
      Statistics<uint32_t>::create(physicalValues),
      makeDefaultPolicy(DataType::Uint32),
  };
  Buffer buffer{*benchmarkPool()};
  const auto child =
      TrivialEncoding<uint32_t>::encode(selection, physicalValues, buffer);
  std::string encoded;
  encoded.resize(
      EncodingPrefix::serializedSize(rows, /*useVarint=*/false) +
      2 * sizeof(uint32_t) + child.size() + sizeof(uint32_t));
  char* position = encoded.data();
  EncodingPrefix::serialize(
      EncodingType::Sentinel,
      DataType::Uint32,
      rows,
      /*useVarint=*/false,
      position);
  encoding::writeUint32(numNulls, position);
  encoding::writeUint32(static_cast<uint32_t>(child.size()), position);
  encoding::writeBytes(child, position);
  encoding::write<uint32_t>(kSentinel, position);
  NIMBLE_CHECK_EQ(position, encoded.data() + encoded.size());
  return encoded;
}

// Lists every encoding that has an encoding view, with data that suits it.
const std::vector<CreateReadCase>& createReadCases() {
  static const std::vector<CreateReadCase> cases = [] {
    std::vector<CreateReadCase> result;
    result.push_back(
        makeCreateReadCase<uint32_t>("Trivial_uint32", [](uint32_t rows) {
          return encodeWithSelection<TrivialEncoding<uint32_t>>(
              EncodingType::Trivial, makeRandom<uint32_t>(rows));
        }));
    result.push_back(
        makeCreateReadCase<bool>("Trivial_bool", [](uint32_t rows) {
          return encodeWithSelection<TrivialEncoding<bool>>(
              EncodingType::Trivial, boolData(rows));
        }));
    result.push_back(
        makeCreateReadCase<std::string_view>(
            "Trivial_string", [](uint32_t rows) {
              std::vector<std::string> backing;
              return encodeTrivialString(stringData(backing, rows));
            }));
    result.push_back(
        makeCreateReadCase<uint32_t>("Constant_uint32", [](uint32_t rows) {
          return encodeWithSelection<ConstantEncoding<uint32_t>>(
              EncodingType::Constant, makeConstant<uint32_t>(42, rows));
        }));
    result.push_back(
        makeCreateReadCase<uint32_t>("FixedBitWidth_uint32", [](uint32_t rows) {
          return encodeWithSelection<FixedBitWidthEncoding<uint32_t>>(
              EncodingType::FixedBitWidth, makeNarrow<uint32_t>(10, rows));
        }));
    result.push_back(
        makeCreateReadCase<uint32_t>(
            "MainlyConstant_uint32", [](uint32_t rows) {
              return encodeWithSelection<MainlyConstantEncoding<uint32_t>>(
                  EncodingType::MainlyConstant, mainlyConstantData(rows));
            }));
    result.push_back(
        makeCreateReadCase<uint32_t>("Dictionary_uint32", [](uint32_t rows) {
          return encodeWithSelection<DictionaryEncoding<uint32_t>>(
              EncodingType::Dictionary, dictionaryData(rows));
        }));
    result.push_back(
        makeCreateReadCase<std::string_view>(
            "Dictionary_string", [](uint32_t rows) {
              std::vector<std::string> backing;
              return encodeWithSelection<DictionaryEncoding<std::string_view>>(
                  EncodingType::Dictionary,
                  dictionaryStringData(backing, rows));
            }));
    result.push_back(
        makeCreateReadCase<uint32_t>("RLE_uint32", [](uint32_t rows) {
          return encodeWithSelection<RLEEncoding<uint32_t>>(
              EncodingType::RLE, rleData(rows));
        }));
    result.push_back(makeCreateReadCase<bool>("SparseBool", [](uint32_t rows) {
      return encodeWithSelection<SparseBoolEncoding>(
          EncodingType::SparseBool, boolData(rows));
    }));
    result.push_back(
        makeCreateReadCase<uint32_t>("FOR_uint32", [](uint32_t rows) {
          return encodeWithSelection<ForEncoding<uint32_t>>(
              EncodingType::FOR, pforData(rows));
        }));
    result.push_back(
        makeCreateReadCase<uint32_t>("PFOR_uint32", [](uint32_t rows) {
          return encodeWithSelection<PFOREncoding<uint32_t>>(
              EncodingType::PFOR, pforData(rows));
        }));
    result.push_back(
        makeCreateReadCase<uint32_t>("Huffman_uint32", [](uint32_t rows) {
          return encodeWithSelection<HuffmanEncoding<uint32_t>>(
              EncodingType::Huffman, huffmanSkewedData(rows));
        }));
    result.push_back(
        makeCreateReadCase<uint32_t>(
            "SimdForBitpack_uint32", [](uint32_t rows) {
              return encodeWithSelection<SimdForBitpackEncoding<uint32_t>>(
                  EncodingType::SimdForBitpack, makeNarrow<uint32_t>(10, rows));
            }));
    result.push_back(
        makeCreateReadCase<uint32_t>(
            "BlockBitPacking_uint32", [](uint32_t rows) {
              return encodeWithSelection<BlockBitPackingV2Encoding<uint32_t>>(
                  EncodingType::BlockBitPackingV2, pforData(rows));
            }));
    result.push_back(
        makeCreateReadCase<uint32_t>("DeltaBlock_uint32", [](uint32_t rows) {
          return encodeWithSelection<DeltaBlockEncoding<uint32_t>>(
              EncodingType::DeltaBlock, makeIncreasing<uint32_t>(rows));
        }));
    result.push_back(
        makeCreateReadCase<uint32_t>("EliasFano_uint32", [](uint32_t rows) {
          return encodeWithSelection<EliasFanoEncoding<uint32_t>>(
              EncodingType::EliasFano, makeIncreasing<uint32_t>(rows));
        }));
    result.push_back(
        makeCreateReadCase<uint64_t>("SubIntSplit_uint64", [](uint32_t rows) {
          return encodeWithSelection<SubIntSplitEncoding<uint64_t>>(
              EncodingType::SubIntSplit, makeNarrow<uint64_t>(40, rows));
        }));
    result.push_back(
        makeCreateReadCase<uint32_t>("Varint_uint32", [](uint32_t rows) {
          return encodeWithSelection<VarintEncoding<uint32_t>>(
              EncodingType::Varint, makeNarrow<uint32_t>(14, rows));
        }));
    result.push_back(
        makeCreateReadCase<uint64_t>("Varint_uint64", [](uint32_t rows) {
          return encodeWithSelection<VarintEncoding<uint64_t>>(
              EncodingType::Varint, makeNarrow<uint64_t>(30, rows));
        }));
    result.push_back(makeCreateReadCase<float>("ALP_float", [](uint32_t rows) {
      return encodeWithSelection<ALPEncoding<float>>(
          EncodingType::ALP, alpData<float>(rows));
    }));
    result.push_back(
        makeCreateReadCase<double>("ALP_double", [](uint32_t rows) {
          return encodeWithSelection<ALPEncoding<double>>(
              EncodingType::ALP, alpData<double>(rows));
        }));
    result.push_back(
        makeCreateReadCase<double>("ALPRD_double", [](uint32_t rows) {
          Vector<double> data{benchmarkPool().get(), rows};
          for (uint32_t row{0}; row < rows; ++row) {
            data[row] = 1.0 + folly::Random::randDouble01();
          }
          return encodeWithSelection<ALPRDEncoding<double>>(
              EncodingType::ALPRD, data);
        }));
    result.push_back(
        makeCreateReadCase<std::string_view>("Fsst_string", [](uint32_t rows) {
          std::vector<std::string> backing;
          return encodeFsstString(stringData(backing, rows));
        }));
    result.push_back(
        makeNullableCreateReadCase<int64_t>(
            "Nullable_int64", [](uint32_t rows) {
              // Compression would measure decompressing the values instead of
              // building the view.
              return encodeNullableView(
                  /*nullsPerThousand=*/100, rows, /*compression=*/std::nullopt);
            }));
    // EncodingFactory has no Sentinel encoding, so only the view reads it.
    auto sentinelCase = makeNullableCreateReadCase<uint32_t>(
        "Sentinel_uint32", encodeSentinelChunk);
    sentinelCase.createAndReadWithSkip = nullptr;
    sentinelCase.readWithReusedSkip = nullptr;
    result.push_back(std::move(sentinelCase));
    return result;
  }();
  return cases;
}

unsigned runCreateRead(
    const CreateReadCase& testCase,
    const CreateReadFunction& createAndRead,
    uint32_t rows,
    unsigned iterations) {
  folly::BenchmarkSuspender suspender;
  const auto encoded = testCase.encode(rows);
  const auto positions = randomPositions(rows);
  suspender.dismiss();
  createAndRead(encoded, positions, iterations);
  suspender.rehire();
  return iterations;
}

// Registers the view with cached encoding views, the view without caching and
// the skip variant of every case for small and large chunks.
void registerCreateReadBenchmarks() {
  for (const uint32_t rows : {kRows, kLargeRows}) {
    for (const auto& testCase : createReadCases()) {
      for (const auto& variant :
           {std::pair{"Cached", true}, std::pair{"Default", false}}) {
        const auto cacheEncodingView = variant.second;
        folly::addBenchmark(
            __FILE__,
            fmt::format(
                "CreateRead_{}_{}_{}", variant.first, testCase.name, rows),
            [&testCase, rows, cacheEncodingView](unsigned iterations) {
              Encoding::Options options;
              options.cacheEncodingView = cacheEncodingView;
              return runCreateRead(
                  testCase,
                  [&testCase, &options](
                      std::string_view encoded,
                      const std::vector<uint32_t>& positions,
                      uint32_t numIterations) {
                    testCase.createAndReadWithView(
                        encoded, positions, numIterations, options);
                  },
                  rows,
                  iterations);
            });
      }
      if (testCase.createAndReadWithSkip == nullptr) {
        continue;
      }
      folly::addBenchmark(
          __FILE__,
          fmt::format("CreateRead_Skip_{}_{}", testCase.name, rows),
          [&testCase, rows](unsigned iterations) {
            return runCreateRead(
                testCase, testCase.createAndReadWithSkip, rows, iterations);
          });
    }
  }
}

// Registers reads from a view or an encoding that is created once and reused
// for every read, with cached encoding views, without caching and with skip.
// The creation is not timed.
void registerReuseReadBenchmarks() {
  for (const uint32_t rows : {kRows, kLargeRows}) {
    for (const auto& testCase : createReadCases()) {
      for (const auto& variant :
           {std::pair{"Cached", true}, std::pair{"Default", false}}) {
        const auto cacheEncodingView = variant.second;
        folly::addBenchmark(
            __FILE__,
            fmt::format("Reuse_{}_{}_{}", variant.first, testCase.name, rows),
            [&testCase, rows, cacheEncodingView](unsigned iterations) {
              folly::BenchmarkSuspender suspender;
              const auto encoded = testCase.encode(rows);
              const auto positions = randomPositions(rows);
              Encoding::Options options;
              options.cacheEncodingView = cacheEncodingView;
              const auto view =
                  createEncodingView(encoded, benchmarkPool().get(), options);
              suspender.dismiss();
              testCase.readWithReusedView(*view, positions, iterations);
              suspender.rehire();
              return iterations;
            });
      }
      if (testCase.readWithReusedSkip == nullptr) {
        continue;
      }
      folly::addBenchmark(
          __FILE__,
          fmt::format("Reuse_Skip_{}_{}", testCase.name, rows),
          [&testCase, rows](unsigned iterations) {
            folly::BenchmarkSuspender suspender;
            const auto encoded = testCase.encode(rows);
            const auto positions = randomPositions(rows);
            std::vector<facebook::velox::BufferPtr> stringBuffers;
            const auto encoding = createRegularEncoding(encoded, stringBuffers);
            suspender.dismiss();
            testCase.readWithReusedSkip(*encoding, positions, iterations);
            suspender.rehire();
            return iterations;
          });
    }
  }
}

#undef VIEW_BENCHMARK
#undef BATCH_VIEW_BENCHMARK
#undef SHARED_DICTIONARY_ALPHABET_BENCHMARK
#undef RANGE_VIEW_BENCHMARK
#undef UNALIGNED_RANGE_VIEW_BENCHMARK
#undef MATERIALIZE_BENCHMARK

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  facebook::velox::memory::MemoryManager::initialize({});
  registerNullableViewBenchmarks();
  registerCreateReadBenchmarks();
  registerReuseReadBenchmarks();
  folly::runBenchmarks();
}
