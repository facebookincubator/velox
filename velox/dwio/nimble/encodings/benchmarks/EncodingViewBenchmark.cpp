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
#include <memory>
#include <numeric>
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
#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/ForEncoding.h"
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/PFOREncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"
#include "velox/dwio/nimble/encodings/SparseBoolEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
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
    auto lengths = EncodingFactory(options).create(
        *pool, {pos, lengthsSize}, noStringBufferFactory);
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
    auto runLengths = EncodingFactory(options).create(
        *pool, {pos, runLengthsSize}, noStringBufferFactory);
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

Vector<uint32_t> dictionaryData() {
  Vector<uint32_t> data{benchmarkPool().get()};
  data.resize(kRows);
  for (uint32_t i = 0; i < kRows; ++i) {
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
Vector<T> alpData() {
  Vector<T> data{benchmarkPool().get()};
  data.resize(kRows);
  for (uint32_t i = 0; i < kRows; ++i) {
    data[i] = static_cast<T>(static_cast<int32_t>((i * 37) % 2001) - 1000) /
        static_cast<T>(100);
  }
  return data;
}

Vector<uint32_t> rleData() {
  Vector<uint32_t> data{benchmarkPool().get()};
  data.resize(kRows);
  for (uint32_t i = 0; i < kRows; ++i) {
    data[i] = i / 16;
  }
  return data;
}

Vector<uint32_t> pforData() {
  Vector<uint32_t> data{benchmarkPool().get()};
  data.resize(kRows);
  for (uint32_t i = 0; i < kRows; ++i) {
    data[i] = i % 127;
    if (i % 97 == 0) {
      data[i] += 10000;
    }
  }
  return data;
}

Vector<uint32_t> huffmanSkewedData() {
  Vector<uint32_t> data{benchmarkPool().get()};
  data.resize(kRows);
  for (uint32_t row = 0; row < kRows; ++row) {
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

Vector<uint32_t> mainlyConstantData() {
  Vector<uint32_t> data{benchmarkPool().get()};
  data.resize(kRows, 7);
  for (uint32_t i = 0; i < kRows; i += 17) {
    data[i] = i;
  }
  return data;
}

Vector<bool> boolData() {
  Vector<bool> data{benchmarkPool().get()};
  data.resize(kRows);
  for (uint32_t i = 0; i < kRows; ++i) {
    data[i] = (i % 7) == 0;
  }
  return data;
}

Vector<std::string_view> stringData(std::vector<std::string>& backing) {
  Vector<std::string_view> data{benchmarkPool().get()};
  data.reserve(kRows);
  backing.reserve(kRows);
  for (uint32_t i = 0; i < kRows; ++i) {
    backing.push_back(fmt::format("feature-value-{}", i % 251));
    data.push_back(backing.back());
  }
  return data;
}

Vector<std::string_view> dictionaryStringData(
    std::vector<std::string>& backing) {
  Vector<std::string_view> data{benchmarkPool().get()};
  data.reserve(kRows);
  backing.reserve(128);
  for (uint32_t i = 0; i < 128; ++i) {
    backing.push_back(fmt::format("dict-value-{}", i));
  }
  for (uint32_t i = 0; i < kRows; ++i) {
    data.push_back(backing[(i * 17) % backing.size()]);
  }
  return data;
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
        boolData()),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_TrivialBool_Range1024,
    bool,
    encodeWithSelection<TrivialEncoding<bool>>(
        EncodingType::Trivial,
        boolData()))
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
        boolData()),
    randomPositions(kRows))
VIEW_BENCHMARK(
    View_SparseBool_Random130,
    bool,
    encodeWithSelection<SparseBoolEncoding>(
        EncodingType::SparseBool,
        boolData()),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_SparseBool_Range1024,
    bool,
    encodeWithSelection<SparseBoolEncoding>(
        EncodingType::SparseBool,
        boolData()))
MATERIALIZE_BENCHMARK(
    Materialize_SparseBool_Random130,
    bool,
    encodeWithSelection<SparseBoolEncoding>(
        EncodingType::SparseBool,
        boolData()),
    randomPositions(kRows))

BENCHMARK(View_TrivialStringEager_Random130, iters) {
  std::vector<std::string> backing;
  std::string encoded;
  std::vector<uint32_t> positions;
  std::unique_ptr<EncodingView> view;
  BENCHMARK_SUSPEND {
    encoded = encodeTrivialString(stringData(backing));
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
    encoded = encodeTrivialString(stringData(backing));
    view = createEncodingView(encoded, benchmarkPool().get());
  }
  readRangesWithView<std::string_view>(*view, iters);
}

BENCHMARK(Construct_TrivialStringOffsets_ReadAt, iters) {
  std::vector<std::string> backing;
  std::string encoded;
  BENCHMARK_SUSPEND {
    encoded = encodeTrivialString(stringData(backing));
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
    encoded = encodeTrivialString(stringData(backing));
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
    encoded = encodeTrivialString(stringData(backing));
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
    encoded = encodeTrivialString(stringData(backing));
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
    encoded = encodeTrivialString(stringData(backing));
    positions = clusteredPositions(kRows);
    view = createEncodingView(encoded, benchmarkPool().get());
  }
  readPositionsWithView<std::string_view>(*view, positions, iters);
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
        mainlyConstantData()),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_MainlyConstantUint32_Range1024,
    uint32_t,
    encodeWithSelection<MainlyConstantEncoding<uint32_t>>(
        EncodingType::MainlyConstant,
        mainlyConstantData()))
MATERIALIZE_BENCHMARK(
    Materialize_MainlyConstantUint32_Random130,
    uint32_t,
    encodeWithSelection<MainlyConstantEncoding<uint32_t>>(
        EncodingType::MainlyConstant,
        mainlyConstantData()),
    randomPositions(kRows))
VIEW_BENCHMARK(
    View_DictionaryUint32_Random130,
    uint32_t,
    encodeWithSelection<DictionaryEncoding<uint32_t>>(
        EncodingType::Dictionary,
        dictionaryData()),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_DictionaryUint32_Range1024,
    uint32_t,
    encodeWithSelection<DictionaryEncoding<uint32_t>>(
        EncodingType::Dictionary,
        dictionaryData()))
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
        EncodingType::Dictionary, dictionaryStringData(backing));
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
        EncodingType::Dictionary, dictionaryStringData(backing));
    view = createEncodingView(encoded, benchmarkPool().get());
  }
  readRangesWithView<std::string_view>(*view, iters);
}
MATERIALIZE_BENCHMARK(
    Materialize_DictionaryUint32_Random130,
    uint32_t,
    encodeWithSelection<DictionaryEncoding<uint32_t>>(
        EncodingType::Dictionary,
        dictionaryData()),
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
        EncodingType::Dictionary, dictionaryStringData(backing));
    positions = randomPositions(kRows);
    encoding = createRegularEncoding(encoded, stringBuffers);
  }
  readPositionsWithMaterialization<std::string_view>(
      *encoding, positions, iters);
}
VIEW_BENCHMARK(
    View_RLEUint32_Random130,
    uint32_t,
    encodeWithSelection<RLEEncoding<uint32_t>>(EncodingType::RLE, rleData()),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_RLEUint32_Range1024,
    uint32_t,
    encodeWithSelection<RLEEncoding<uint32_t>>(EncodingType::RLE, rleData()))
MATERIALIZE_BENCHMARK(
    Materialize_RLEUint32_Random130,
    uint32_t,
    encodeWithSelection<RLEEncoding<uint32_t>>(EncodingType::RLE, rleData()),
    randomPositions(kRows))
BENCHMARK(Construct_RLERunEnds_ReadAt, iters) {
  std::string encoded;
  BENCHMARK_SUSPEND {
    encoded = encodeWithSelection<RLEEncoding<uint32_t>>(
        EncodingType::RLE, rleData());
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
        EncodingType::RLE, rleData());
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
        alpData<float>()),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_ALPFloat_Range1024,
    float,
    encodeWithSelection<ALPEncoding<float>>(
        EncodingType::ALP,
        alpData<float>()))
VIEW_BENCHMARK(
    View_ALPDouble_Random130,
    double,
    encodeWithSelection<ALPEncoding<double>>(
        EncodingType::ALP,
        alpData<double>()),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_ALPDouble_Range1024,
    double,
    encodeWithSelection<ALPEncoding<double>>(
        EncodingType::ALP,
        alpData<double>()))
MATERIALIZE_BENCHMARK(
    Materialize_ALPFloat_Random130,
    float,
    encodeWithSelection<ALPEncoding<float>>(
        EncodingType::ALP,
        alpData<float>()),
    randomPositions(kRows))
MATERIALIZE_BENCHMARK(
    Materialize_ALPDouble_Random130,
    double,
    encodeWithSelection<ALPEncoding<double>>(
        EncodingType::ALP,
        alpData<double>()),
    randomPositions(kRows))
VIEW_BENCHMARK(
    View_FORUint32_Random130,
    uint32_t,
    encodeWithSelection<ForEncoding<uint32_t>>(EncodingType::FOR, pforData()),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_FORUint32_Range1024,
    uint32_t,
    encodeWithSelection<ForEncoding<uint32_t>>(EncodingType::FOR, pforData()))
BENCHMARK(Materialize_FORUint32_Random130, iters) {
  std::string encoded;
  std::vector<uint32_t> positions;
  std::unique_ptr<ForEncoding<uint32_t>> encoding;
  BENCHMARK_SUSPEND {
    encoded = encodeWithSelection<ForEncoding<uint32_t>>(
        EncodingType::FOR, pforData());
    positions = randomPositions(kRows);
    encoding = std::make_unique<ForEncoding<uint32_t>>(
        *benchmarkPool(), encoded, nullptr, Encoding::Options{});
  }
  readPositionsWithMaterialization<uint32_t>(*encoding, positions, iters);
}
VIEW_BENCHMARK(
    View_PFORUint32_Random130,
    uint32_t,
    encodeWithSelection<PFOREncoding<uint32_t>>(EncodingType::PFOR, pforData()),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_PFORUint32_Range1024,
    uint32_t,
    encodeWithSelection<PFOREncoding<uint32_t>>(EncodingType::PFOR, pforData()))
MATERIALIZE_BENCHMARK(
    Materialize_PFORUint32_Random130,
    uint32_t,
    encodeWithSelection<PFOREncoding<uint32_t>>(EncodingType::PFOR, pforData()),
    randomPositions(kRows))
VIEW_BENCHMARK(
    View_HuffmanUint32_SkewedRandom130,
    uint32_t,
    encodeWithSelection<HuffmanEncoding<uint32_t>>(
        EncodingType::Huffman,
        huffmanSkewedData()),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_HuffmanUint32_SkewedRange1024,
    uint32_t,
    encodeWithSelection<HuffmanEncoding<uint32_t>>(
        EncodingType::Huffman,
        huffmanSkewedData()))
MATERIALIZE_BENCHMARK(
    Materialize_HuffmanUint32_SkewedRandom130,
    uint32_t,
    encodeWithSelection<HuffmanEncoding<uint32_t>>(
        EncodingType::Huffman,
        huffmanSkewedData()),
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
    encodeWithSelection<BlockBitPackingEncoding<uint32_t>>(
        EncodingType::BlockBitPacking,
        pforData()),
    randomPositions(kRows))
RANGE_VIEW_BENCHMARK(
    RangeView_BlockBitPackingUint32_Range1024,
    uint32_t,
    encodeWithSelection<BlockBitPackingEncoding<uint32_t>>(
        EncodingType::BlockBitPacking,
        pforData()))
UNALIGNED_RANGE_VIEW_BENCHMARK(
    RangeView_BlockBitPackingUint32_UnalignedRange1024,
    uint32_t,
    encodeWithSelection<BlockBitPackingEncoding<uint32_t>>(
        EncodingType::BlockBitPacking,
        pforData()))
UNALIGNED_RANGE_VIEW_BENCHMARK(
    RangeView_BlockBitPackingUint16_UnalignedRange1024,
    uint16_t,
    encodeWithSelection<BlockBitPackingEncoding<uint16_t>>(
        EncodingType::BlockBitPacking,
        makeNarrow<uint16_t>(10, kRows)))
MATERIALIZE_BENCHMARK(
    Materialize_BlockBitPackingUint32_Random130,
    uint32_t,
    encodeWithSelection<BlockBitPackingEncoding<uint32_t>>(
        EncodingType::BlockBitPacking,
        pforData()),
    randomPositions(kRows))

#undef VIEW_BENCHMARK
#undef RANGE_VIEW_BENCHMARK
#undef UNALIGNED_RANGE_VIEW_BENCHMARK
#undef MATERIALIZE_BENCHMARK

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  facebook::velox::memory::MemoryManager::initialize({});
  folly::runBenchmarks();
}
