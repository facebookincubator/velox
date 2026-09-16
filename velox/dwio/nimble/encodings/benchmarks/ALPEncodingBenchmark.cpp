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
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <folly/Benchmark.h>
#include <folly/hash/SpookyHashV2.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "velox/common/io/IoStatistics.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/FormatData.h"
#include "velox/dwio/common/SelectiveColumnReader.h"
#include "velox/dwio/common/SelectiveColumnReaderInternal.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/common/TypeWithId.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/BenchmarkUtils.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingUtils.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/velox/ChunkedStream.h"
#include "velox/dwio/nimble/velox/selective/SelectiveNimbleReader.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/vector/FlatVector.h"

DEFINE_uint32(
    rows,
    65'536,
    "Values per dataset; use --bm_regex to select cases.");
DEFINE_bool(profile, false, "Report repeated Release decode timings.");
DEFINE_bool(e2e, false, "Benchmark Nimble file writing and selective reading.");
DEFINE_bool(
    force_alp,
    false,
    "Force ALP value streams in non-nullable E2E benchmarks.");
DEFINE_uint32(read_batch_size, 4096, "Rows per selective reader batch.");
DEFINE_uint32(
    visitor_start,
    0,
    "Physical start row for microbenchmark visitors.");
DEFINE_uint32(
    visitor_rows,
    0,
    "Visitor row range; zero uses the remaining stream.");
DEFINE_uint32(visitor_stride, 4, "Sparse visitor row stride.");
DEFINE_uint32(
    visitor_batch_size,
    0,
    "Selected rows per visitor call; zero reads all.");
DEFINE_string(
    exception_pattern,
    "",
    "Synthetic exceptions: none, one, four, five, all, prefix, suffix, periodic.");
DEFINE_string(
    profile_filter,
    "",
    "Only profile dataset names containing this text.");
DEFINE_string(profile_operation, "", "Only profile this operation.");
DEFINE_bool(
    profile_skip_unselected_setup,
    true,
    "Skip constructing unselected microbenchmark fixtures.");
DEFINE_bool(
    profile_scalar_visitor_setup,
    false,
    "Use the scalar visitor for fixture validation.");
DEFINE_uint32(profile_trials, 7, "Number of timed trials per operation.");
DEFINE_uint32(
    profile_min_ms,
    20,
    "Minimum calibrated duration per timed trial.");

namespace {

using facebook::nimble::ALPEncoding;
using facebook::nimble::Buffer;
using facebook::nimble::EncodingFactory;
using facebook::nimble::EncodingType;
using facebook::nimble::ManualEncodingSelectionPolicy;
using facebook::nimble::TypeTraits;
using facebook::nimble::benchmarks::benchmarkPool;
using facebook::nimble::benchmarks::nullFactory;
namespace alp = facebook::nimble::detail::alp;
namespace common = facebook::velox::dwio::common;
namespace velox = facebook::velox;
namespace nimble = facebook::nimble;

constexpr uint64_t kSeed = 0x51ED0FF1CEULL;

template <typename Function>
void profileOperation(
    const std::string& name,
    std::string_view operation,
    Function&& function) {
  if (!FLAGS_profile_operation.empty() &&
      operation != FLAGS_profile_operation) {
    return;
  }
  const auto timeIterations = [&](uint64_t iterations) {
    const auto start = std::chrono::steady_clock::now();
    for (uint64_t iteration = 0; iteration < iterations; ++iteration) {
      function();
    }
    return std::chrono::duration<double, std::nano>(
               std::chrono::steady_clock::now() - start)
        .count();
  };

  uint64_t iterations = 1;
  while (timeIterations(iterations) < FLAGS_profile_min_ms * 1'000'000.0) {
    CHECK_LE(iterations, std::numeric_limits<uint64_t>::max() / 2);
    iterations *= 2;
  }
  std::vector<double> timings;
  timings.reserve(FLAGS_profile_trials);
  for (uint32_t trial = 0; trial < FLAGS_profile_trials; ++trial) {
    timings.push_back(timeIterations(iterations) / iterations);
  }
  for (size_t trial = 0; trial < timings.size(); ++trial) {
    fmt::print(
        "TRIAL,{},{},{},{:.2f}\n", name, operation, trial, timings[trial]);
  }
  std::sort(timings.begin(), timings.end());
  const auto middle = timings.size() / 2;
  const double median = timings.size() % 2 == 0
      ? (timings[middle - 1] + timings[middle]) / 2
      : timings[middle];
  fmt::print(
      "PROFILE,{},{},{},{:.2f},{:.2f},{:.2f}\n",
      name,
      operation,
      iterations,
      median,
      timings.front(),
      timings.back());
}

class BenchmarkFormatData final : public common::FormatData {
 public:
  void readNulls(
      facebook::velox::vector_size_t,
      const uint64_t*,
      facebook::velox::BufferPtr& nulls,
      bool) final {
    nulls = nullptr;
  }

  uint64_t skipNulls(uint64_t numValues, bool) final {
    return numValues;
  }

  uint64_t skip(uint64_t numValues) final {
    return numValues;
  }

  bool hasNulls() const final {
    return false;
  }

  common::PositionProvider seekToRowGroup(int64_t) final {
    static std::vector<uint64_t> positions;
    return common::PositionProvider(positions);
  }

  void filterRowGroups(
      const common::ScanSpec&,
      uint64_t,
      const common::StatsContext&,
      FilterRowGroupsResult&) final {}
};

class BenchmarkFormatParams final : public common::FormatParams {
 public:
  BenchmarkFormatParams(
      facebook::velox::memory::MemoryPool& pool,
      common::SplitStats& stats)
      : FormatParams(pool, stats) {}

  std::unique_ptr<common::FormatData> toFormatData(
      const std::shared_ptr<const common::TypeWithId>&,
      const facebook::velox::common::ScanSpec&) final {
    return std::make_unique<BenchmarkFormatData>();
  }
};

struct VisitorInfrastructure {
  common::SplitStats stats{common::FileFormat::NIMBLE};
  BenchmarkFormatParams params{*benchmarkPool(), stats};
  facebook::velox::common::ScanSpec scanSpec{"value"};

  VisitorInfrastructure() {
    scanSpec.setProjectOut(true);
  }
};

template <typename FloatType>
class BenchmarkColumnReader final : public common::SelectiveColumnReader {
 public:
  BenchmarkColumnReader(
      const facebook::velox::TypePtr& type,
      BenchmarkFormatParams& params,
      facebook::velox::common::ScanSpec& scanSpec)
      : SelectiveColumnReader(
            type,
            common::TypeWithId::create(type),
            params,
            scanSpec) {}

  void prepare(const facebook::velox::RowSet& rows) {
    this->template prepareRead<FloatType>(0, rows, nullptr);
  }

  void read(int64_t, const facebook::velox::RowSet&, const uint64_t*) final {}

  void getValues(const facebook::velox::RowSet&, facebook::velox::VectorPtr*)
      final {}
};

template <typename Visitor>
facebook::nimble::ReadWithVisitorParams makeVisitorParams(
    Visitor& visitor,
    const facebook::velox::RowSet& rows) {
  facebook::nimble::ReadWithVisitorParams params{};
  params.prepareResultNulls = [&visitor, rows] {
    visitor.reader().prepareNulls(rows, false, 8);
  };
  params.setReturnNullsMode = [&visitor, rows] {
    visitor.reader().setReturnNullsMode(rows);
  };
  params.numScanned = 0;
  return params;
}

template <typename FloatType>
facebook::velox::TypePtr floatingPointType() {
  if constexpr (std::is_same_v<FloatType, float>) {
    return facebook::velox::REAL();
  }
  return facebook::velox::DOUBLE();
}

template <typename FloatType, typename Generator>
std::vector<FloatType> makeValues(Generator generator) {
  std::mt19937_64 random(kSeed);
  std::vector<FloatType> values;
  values.reserve(FLAGS_rows);
  for (uint32_t row = 0; row < FLAGS_rows; ++row) {
    values.push_back(generator(random));
  }
  return values;
}

template <typename FloatType>
std::vector<FloatType> makeSensorValues(double outlierRate) {
  std::uniform_int_distribution<int32_t> clean(0, 99'999);
  std::uniform_int_distribution<int64_t> outlier(
      5'000'000'000LL, 9'999'999'999LL);
  std::uniform_real_distribution<double> probability(0.0, 1.0);
  return makeValues<FloatType>([&](auto& random) {
    return probability(random) < outlierRate
        ? static_cast<FloatType>(outlier(random)) /
            static_cast<FloatType>(1'000'000)
        : static_cast<FloatType>(clean(random)) / static_cast<FloatType>(1'000);
  });
}

struct GridResult {
  uint8_t exponent{0};
  uint8_t factor{0};
  uint32_t representable{0};

  bool operator==(const GridResult&) const = default;
};

enum class FileShape { Flat, Nullable, Nested, Sparse };

template <typename FloatType>
class AlpFileBenchmarkFixture {
 public:
  AlpFileBenchmarkFixture(
      const std::string& name,
      const std::vector<FloatType>& values,
      FileShape shape)
      : rootPool_(velox::memory::memoryManager()->addRootPool()),
        pool_(rootPool_->addLeafChild("alp_file_benchmark")),
        values_(values),
        shape_(shape) {
    auto floating = velox::BaseVector::create(
        floatingPointType<FloatType>(), values.size(), pool_.get());
    auto* flat = floating->template asFlatVector<FloatType>();
    std::copy(values.begin(), values.end(), flat->mutableRawValues());
    if (shape_ == FileShape::Nullable) {
      for (uint32_t row = 0; row < values.size(); row += 10) {
        flat->setNull(row, true);
      }
    }
    if (shape_ == FileShape::Nested) {
      floating = std::make_shared<velox::RowVector>(
          pool_.get(),
          velox::ROW({"value"}, {floating->type()}),
          nullptr,
          values.size(),
          std::vector<velox::VectorPtr>{floating});
    }
    std::vector<std::string> names{"c0"};
    std::vector<velox::VectorPtr> children{floating};
    if (shape_ == FileShape::Sparse) {
      auto selector = velox::BaseVector::create(
          velox::BIGINT(), values.size(), pool_.get());
      auto* raw =
          selector->template asFlatVector<int64_t>()->mutableRawValues();
      for (uint32_t row = 0; row < values.size(); ++row) {
        raw[row] = row % 4;
      }
      names.insert(names.begin(), "selector");
      children.insert(children.begin(), selector);
    }
    std::vector<velox::TypePtr> types;
    for (const auto& child : children) {
      types.push_back(child->type());
    }
    input_ = std::make_shared<velox::RowVector>(
        pool_.get(),
        velox::ROW(std::move(names), std::move(types)),
        nullptr,
        values.size(),
        std::move(children));
    auto ordered = values;
    auto middle = ordered.begin() + ordered.size() / 2;
    std::nth_element(ordered.begin(), middle, ordered.end());
    filterUpper_ = *middle;
    file_ = write();
    read(false, true);
    read(true, true);
    printFileInfo(name);
  }

  std::string write() const {
    nimble::WriterOptions options;
    options.compressionOptions.compressionType =
        nimble::CompressionType::Uncompressed;
    options.encodingSelectionPolicyCreator = [](nimble::DataType dataType) {
      auto candidates = nimble::ManualEncodingSelectionPolicyFactory::
          defaultEncodingReadFactors();
      if (dataType == nimble::DataType::Float ||
          dataType == nimble::DataType::Double) {
        candidates = {
            {EncodingType::ALP, 1.0},
            {EncodingType::Trivial, 1.0},
            {EncodingType::FixedBitWidth, 1.0}};
      }
      return nimble::ManualEncodingSelectionPolicyFactory{
          std::move(candidates), std::nullopt}
          .createPolicy(dataType);
    };
    if (FLAGS_force_alp) {
      using StreamLayouts = std::unordered_map<
          nimble::EncodingLayoutTree::StreamIdentifier,
          nimble::EncodingLayout>;
      const nimble::EncodingLayout trivialLayout{
          EncodingType::Trivial, {}, nimble::CompressionType::Uncompressed};
      const nimble::EncodingLayout alpLayout{
          EncodingType::ALP,
          {},
          nimble::CompressionType::Uncompressed,
          {nimble::EncodingLayout{
               EncodingType::FixedBitWidth,
               {},
               nimble::CompressionType::Uncompressed},
           trivialLayout,
           trivialLayout}};
      const StreamLayouts valueLayout{
          {nimble::EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
           alpLayout}};
      std::vector<nimble::EncodingLayoutTree> children;
      if (shape_ == FileShape::Sparse) {
        children.emplace_back(
            nimble::Kind::Scalar, StreamLayouts{}, "selector");
      }
      if (shape_ == FileShape::Nested) {
        std::vector<nimble::EncodingLayoutTree> nested;
        nested.emplace_back(nimble::Kind::Scalar, valueLayout, "value");
        children.emplace_back(
            nimble::Kind::Row, StreamLayouts{}, "c0", std::move(nested));
      } else {
        children.emplace_back(nimble::Kind::Scalar, valueLayout, "c0");
      }
      options.encodingLayoutTree.emplace(
          nimble::Kind::Row, StreamLayouts{}, "", std::move(children));
    }
    std::string file;
    nimble::Writer writer(
        input_->type(),
        std::make_unique<velox::InMemoryWriteFile>(&file),
        *rootPool_,
        std::move(options));
    writer.write(input_);
    writer.close();
    return file;
  }

  uint64_t read(bool filtered, bool verify = false) const {
    auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input_->type());
    if (shape_ == FileShape::Sparse) {
      scanSpec->childByName("selector")
          ->setFilter(
              std::make_unique<velox::common::BigintRange>(0, 0, false));
    } else if (filtered) {
      auto* valueSpec = scanSpec->childByName("c0");
      if (shape_ == FileShape::Nested) {
        valueSpec = valueSpec->childByName("value");
      }
      valueSpec->setFilter(
          std::make_unique<velox::common::FloatingPointRange<FloatType>>(
              FloatType{0}, true, false, filterUpper_, false, false, false));
    }
    auto options = readerOptions();
    options.setScanSpec(scanSpec);
    nimble::SelectiveNimbleReaderFactory factory;
    auto reader = factory.createReader(
        std::make_unique<common::BufferedInput>(
            std::make_shared<velox::InMemoryReadFile>(file_), *pool_),
        options);
    common::RowReaderOptions rowOptions;
    rowOptions.setScanSpec(scanSpec);
    rowOptions.setRequestedType(velox::asRowType(input_->type()));
    auto rowReader = reader->createRowReader(rowOptions);
    velox::VectorPtr output =
        velox::BaseVector::create(input_->type(), 0, pool_.get());
    uint64_t scannedRows = 0;
    uint64_t outputRows = 0;
    uint64_t expectedRow = 0;
    while (const auto scanned =
               rowReader->next(FLAGS_read_batch_size, output)) {
      scannedRows += scanned;
      auto* leaf = output->as<velox::RowVector>()
                       ->childAt(shape_ == FileShape::Sparse ? 1 : 0)
                       ->loadedVector();
      if (shape_ == FileShape::Nested) {
        leaf = leaf->as<velox::RowVector>()->childAt(0)->loadedVector();
      }
      folly::doNotOptimizeAway(leaf);
      outputRows += output->size();
      if (verify) {
        const auto* decoded =
            leaf->template as<velox::SimpleVector<FloatType>>();
        CHECK_NOTNULL(decoded);
        for (velox::vector_size_t row = 0; row < output->size(); ++row) {
          while (expectedRow < values_.size() &&
                 !selected(expectedRow, filtered)) {
            ++expectedRow;
          }
          CHECK_LT(expectedRow, values_.size());
          CHECK_EQ(decoded->isNullAt(row), isNull(expectedRow));
          if (!isNull(expectedRow)) {
            CHECK_EQ(
                alp::toPhysical<FloatType>(decoded->valueAt(row)),
                alp::toPhysical<FloatType>(values_[expectedRow]));
          }
          ++expectedRow;
        }
      }
    }
    CHECK_EQ(scannedRows, values_.size());
    if (verify) {
      uint64_t expectedCount = 0;
      for (uint32_t row = 0; row < values_.size(); ++row) {
        expectedCount += selected(row, filtered);
      }
      CHECK_EQ(outputRows, expectedCount);
    }
    return outputRows;
  }

 private:
  bool isNull(uint32_t row) const {
    return shape_ == FileShape::Nullable && row % 10 == 0;
  }

  bool selected(uint32_t row, bool filtered) const {
    if (shape_ == FileShape::Sparse) {
      return row % 4 == 0;
    }
    return !filtered || (!isNull(row) && values_[row] <= filterUpper_);
  }

  common::ReaderOptions readerOptions() const {
    common::ReaderOptions options(pool_.get());
    options.setDataIoStats(std::make_shared<velox::io::IoStatistics>());
    options.setMetadataIoStats(std::make_shared<velox::io::IoStatistics>());
    return options;
  }

  void printFileInfo(const std::string& name) const {
    auto options = readerOptions();
    auto tablet = nimble::TabletReader::create(
        std::make_shared<velox::InMemoryReadFile>(file_),
        pool_.get(),
        nimble::TabletReader::configureOptions(options));
    uint32_t alpChunks = 0;
    uint32_t trivialChunks = 0;
    uint32_t otherChunks = 0;
    for (uint32_t stripeIndex = 0; stripeIndex < tablet->stripeCount();
         ++stripeIndex) {
      const auto stripe = tablet->stripeIdentifier(stripeIndex);
      std::vector<uint32_t> streamIds(tablet->streamCount(stripe));
      std::iota(streamIds.begin(), streamIds.end(), 0);
      auto streams = tablet->load(stripe, streamIds);
      for (auto& stream : streams) {
        if (!stream) {
          continue;
        }
        nimble::InMemoryChunkedStream chunks(*pool_, std::move(stream));
        while (chunks.hasNext()) {
          const auto chunk = chunks.nextChunk();
          const auto dataType = nimble::EncodingPrefix::dataType(chunk);
          if (dataType != nimble::DataType::Float &&
              dataType != nimble::DataType::Double) {
            continue;
          }
          const auto layout = nimble::EncodingLayoutCapture::capture(chunk, {});
          const auto& valueLayout =
              layout.encodingType() == EncodingType::Nullable
              ? layout.child(nimble::EncodingIdentifiers::Nullable::Data)
                    .value()
              : layout;
          const auto encoding = valueLayout.encodingType();
          alpChunks += encoding == EncodingType::ALP;
          trivialChunks += encoding == EncodingType::Trivial;
          otherChunks += encoding != EncodingType::ALP &&
              encoding != EncodingType::Trivial;
        }
      }
    }
    if (FLAGS_force_alp) {
      CHECK_GT(alpChunks, 0);
      CHECK_EQ(trivialChunks + otherChunks, 0);
    }
    fmt::print(
        "FILE,{},{},{},{},{},{},{},{}\n",
        name,
        values_.size(),
        file_.size(),
        folly::hash::SpookyHashV2::Hash64(file_.data(), file_.size(), 0),
        tablet->stripeCount(),
        alpChunks,
        trivialChunks,
        otherChunks);
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::vector<FloatType> values_;
  FileShape shape_;
  FloatType filterUpper_;
  velox::RowVectorPtr input_;
  std::string file_;
};

template <typename FloatType>
void registerFileDataset(
    const std::string& name,
    const std::vector<FloatType>& values) {
  for (const auto& [shape, shapeName] :
       {std::pair{FileShape::Flat, "Flat"},
        std::pair{FileShape::Nullable, "Nullable"},
        std::pair{FileShape::Nested, "Nested"},
        std::pair{FileShape::Sparse, "Sparse"}}) {
    const auto fileName = fmt::format("{}_{}", name, shapeName);
    if (FLAGS_force_alp && shape == FileShape::Nullable) {
      continue;
    }
    if (!FLAGS_profile_filter.empty() &&
        fileName.find(FLAGS_profile_filter) == std::string::npos) {
      continue;
    }
    auto fixture = std::make_shared<AlpFileBenchmarkFixture<FloatType>>(
        fileName, values, shape);
    const auto addOperation = [&](std::string_view operation, auto function) {
      if (FLAGS_profile) {
        profileOperation(fileName, operation, function);
      } else {
        folly::addBenchmark(
            __FILE__, fmt::format("{}_{}", operation, fileName), [function] {
              function();
              return 1;
            });
      }
    };
    addOperation(
        "FileWrite", [fixture] { folly::doNotOptimizeAway(fixture->write()); });
    addOperation("FileRead", [fixture] {
      folly::doNotOptimizeAway(fixture->read(false));
    });
    if (shape != FileShape::Sparse) {
      addOperation("FileReadFilter", [fixture] {
        folly::doNotOptimizeAway(fixture->read(true));
      });
    }
  }
}

template <typename FloatType>
class AlpBenchmarkFixture {
 public:
  using Alp = ALPEncoding<FloatType>;
  using PhysicalType = typename TypeTraits<FloatType>::physicalType;

  AlpBenchmarkFixture(
      const std::string& name,
      std::vector<FloatType> values,
      uint8_t exponent)
      : values_{std::move(values)},
        physicals_(values_.size()),
        zigZag_(values_.size()),
        mask_(values_.size()),
        output_(values_.size()),
        exponent_{exponent},
        visitorReader_{
            floatingPointType<FloatType>(),
            visitorInfrastructure_.params,
            visitorInfrastructure_.scanSpec},
        denseRows_(
            FLAGS_visitor_rows == 0 ? values_.size() - FLAGS_visitor_start
                                    : FLAGS_visitor_rows),
        sparseRows_{} {
    std::iota(denseRows_.begin(), denseRows_.end(), 0);
    sparseRows_.reserve((denseRows_.size() - 1) / FLAGS_visitor_stride + 1);
    for (uint64_t row = 0; row < denseRows_.size();
         row += FLAGS_visitor_stride) {
      sparseRows_.push_back(row);
    }
    for (size_t row = 0; row < values_.size(); ++row) {
      physicals_[row] = alp::toPhysical<FloatType>(values_[row]);
    }

    const auto scalarCount = runTransform<false>();
    const auto scalarZigZag = zigZag_;
    const auto scalarMask = mask_;
    CHECK_EQ(runTransform<true>(), scalarCount);
    CHECK(scalarZigZag == zigZag_);
    CHECK(scalarMask == mask_);

    const auto scalarGrid = runGrid<false>();
    const auto batchGrid = runGrid<true>();
    CHECK(scalarGrid == batchGrid);

    Buffer buffer{*benchmarkPool()};
    encoded_ = std::string{encode(buffer)};
    Buffer alpBuffer{*benchmarkPool()};
    alpEncoded_ = std::string{encodeAlp(alpBuffer)};
    auto encoding =
        EncodingFactory{}.create(*benchmarkPool(), encoded_, nullFactory());
    visitorEncoding_ =
        EncodingFactory{}.create(*benchmarkPool(), alpEncoded_, nullFactory());
    CHECK_EQ(visitorEncoding_->encodingType(), EncodingType::ALP);
    if (encoding->encodingType() == EncodingType::ALP) {
      const char* position = encoded_.data() + encoding->dataOffset();
      const auto header = alp::readHeader(position);
      CHECK_EQ(header.exponent, batchGrid.exponent);
      CHECK_EQ(header.factor, batchGrid.factor);
    }

    decode();
    for (size_t row = 0; row < values_.size(); ++row) {
      CHECK_EQ(alp::toPhysical<FloatType>(output_[row]), physicals_[row]);
    }

    if (FLAGS_profile_scalar_visitor_setup) {
      decodeVisitorDense<false>(true);
    } else {
      decodeVisitorDense<true>(true);
    }
    decodeVisitorSparse(true);

    fmt::print(
        "ALP,{},{},{},{},{}\n",
        name,
        values_.size(),
        alpEncoded_.size(),
        folly::hash::SpookyHashV2::Hash64(
            alpEncoded_.data(), alpEncoded_.size(), 0),
        visitorEncoding_->debugString());

    fmt::print(
        "{}: rows={} transform=({},0) exceptions={:.2f}% "
        "grid=({},{}) encoding={} bytes={} encoded/raw={:.4f}\n",
        name,
        values_.size(),
        exponent_,
        100.0 * (values_.size() - scalarCount) / values_.size(),
        batchGrid.exponent,
        batchGrid.factor,
        facebook::nimble::toString(encoding->encodingType()),
        encoded_.size(),
        static_cast<double>(encoded_.size()) /
            (values_.size() * sizeof(FloatType)));
  }

  template <bool useBatch>
  uint32_t runTransform() {
    const auto count = transform<useBatch, true>(values_.size(), exponent_, 0);
    folly::doNotOptimizeAway(zigZag_.data());
    folly::doNotOptimizeAway(mask_.data());
    return count;
  }

  template <bool useBatch>
  GridResult runGrid() {
    folly::doNotOptimizeAway(values_.data());
    const auto sampleSize = Alp::estimateSampleSize(values_.size());
    GridResult best;
    for (uint8_t exponent = 0; exponent < Alp::kPow10Double.size();
         ++exponent) {
      for (uint8_t factor = 0; factor <= exponent; ++factor) {
        const auto count =
            transform<useBatch, false>(sampleSize, exponent, factor);
        if (count > best.representable) {
          best = {exponent, factor, count};
        }
        if (best.representable == sampleSize) {
          return best;
        }
      }
    }
    return best;
  }

  std::string_view encode(Buffer& buffer) const {
    auto policy = std::make_unique<ManualEncodingSelectionPolicy<FloatType>>(
        std::vector<std::pair<EncodingType, float>>{
            {EncodingType::ALP, 1.0},
            {EncodingType::Trivial, 1.0},
            {EncodingType::FixedBitWidth, 1.0}},
        std::nullopt,
        std::nullopt);
    return EncodingFactory::encode<FloatType>(
        std::move(policy), std::span<const FloatType>{values_}, buffer);
  }

  std::string_view encodeAlp(Buffer& buffer) const {
    auto policy = std::make_unique<ManualEncodingSelectionPolicy<FloatType>>(
        std::vector<std::pair<EncodingType, float>>{{EncodingType::ALP, 1.0}},
        std::nullopt,
        std::nullopt);
    return EncodingFactory::encode<FloatType>(
        std::move(policy), std::span<const FloatType>{values_}, buffer);
  }

  void decode() {
    auto encoding =
        EncodingFactory{}.create(*benchmarkPool(), encoded_, nullFactory());
    encoding->materialize(values_.size(), output_.data());
    folly::doNotOptimizeAway(output_.data());
  }

  void decodeReuse() {
    visitorEncoding_->reset();
    visitorEncoding_->materialize(values_.size(), output_.data());
    folly::doNotOptimizeAway(output_.data());
  }

  template <bool hasBulkPath>
  void decodeVisitorDense(bool verify = false) {
    auto& filter = common::alwaysTrue();
    decodeVisitor<true, hasBulkPath>(denseRows_, filter, verify);
  }

  template <bool hasBulkPath>
  void decodeVisitorDenseFiltered() {
    facebook::velox::common::FloatingPointRange<FloatType> filter(
        static_cast<FloatType>(0),
        false,
        false,
        static_cast<FloatType>(1'000),
        false,
        false,
        false);
    decodeVisitor<true, hasBulkPath>(denseRows_, filter, false);
  }

  void decodeVisitorSparse(bool verify = false) {
    auto& filter = common::alwaysTrue();
    decodeVisitor<false, true>(sparseRows_, filter, verify);
  }

 private:
  template <bool dense, bool hasBulkPath, typename Filter>
  void decodeVisitor(
      const std::vector<velox::vector_size_t>& selectedRows,
      Filter& filter,
      bool verify) {
    visitorEncoding_->reset();
    visitorEncoding_->skip(FLAGS_visitor_start);
    const auto batchSize = FLAGS_visitor_batch_size == 0
        ? selectedRows.size()
        : FLAGS_visitor_batch_size;
    for (size_t start = 0; start < selectedRows.size(); start += batchSize) {
      const velox::RowSet rows{
          selectedRows.data() + start,
          std::min(batchSize, selectedRows.size() - start)};
      visitorReader_.prepare(rows);
      common::ExtractToReader extractValues(&visitorReader_);
      common::ColumnVisitor<
          FloatType,
          Filter,
          common::ExtractToReader,
          dense,
          hasBulkPath>
          visitor(filter, &visitorReader_, rows, extractValues);
      auto params = makeVisitorParams(visitor, rows);
      params.numScanned = start == 0 ? 0 : selectedRows[start - 1] + 1;
      nimble::callReadWithVisitor(*visitorEncoding_, visitor, params);
      folly::doNotOptimizeAway(visitorReader_.rawValues());
      if (verify) {
        CHECK_EQ(visitorReader_.numValues(), rows.size());
        const auto* output =
            static_cast<const FloatType*>(visitorReader_.rawValues());
        for (size_t index = 0; index < rows.size(); ++index) {
          CHECK_EQ(
              alp::toPhysical<FloatType>(output[index]),
              physicals_[FLAGS_visitor_start + rows[index]]);
        }
      }
    }
  }

  template <bool useBatch, bool materialize>
  FOLLY_NOINLINE uint32_t
  transform(size_t size, uint8_t exponent, uint8_t factor) {
    const double exponentMultiplier = Alp::kPow10Double[exponent];
    const double factorMultiplier = Alp::kPow10Double[factor];
    uint32_t representableCount = 0;
    auto record = [&](size_t row, bool representable, uint64_t zigZag) {
      representableCount += representable;
      if constexpr (materialize) {
        mask_[row] = representable;
        zigZag_[row] = zigZag;
      }
    };

    size_t row = 0;
    if constexpr (useBatch) {
      constexpr auto kBatchSize = Alp::kBatchSize;
      alignas(64) uint64_t zigZagLanes[kBatchSize];
      alignas(64) bool maskLanes[kBatchSize];
      for (; row + kBatchSize <= size; row += kBatchSize) {
        Alp::batchTransform(
            values_.data() + row,
            physicals_.data() + row,
            exponentMultiplier,
            factorMultiplier,
            zigZagLanes,
            maskLanes);
        for (size_t lane = 0; lane < kBatchSize; ++lane) {
          record(
              row + lane,
              maskLanes[lane],
              maskLanes[lane] ? zigZagLanes[lane] : 0);
        }
      }
    }
    for (; row < size; ++row) {
      uint64_t zigZag = 0;
      const bool representable = Alp::scalarTransformOne(
          values_[row],
          physicals_[row],
          exponentMultiplier,
          factorMultiplier,
          zigZag);
      record(row, representable, zigZag);
    }
    return representableCount;
  }

  std::vector<FloatType> values_;
  std::vector<PhysicalType> physicals_;
  std::vector<uint64_t> zigZag_;
  std::vector<uint8_t> mask_;
  std::vector<FloatType> output_;
  uint8_t exponent_;
  std::string encoded_;
  std::string alpEncoded_;
  VisitorInfrastructure visitorInfrastructure_;
  BenchmarkColumnReader<FloatType> visitorReader_;
  std::unique_ptr<facebook::nimble::Encoding> visitorEncoding_;
  std::vector<facebook::velox::vector_size_t> denseRows_;
  std::vector<facebook::velox::vector_size_t> sparseRows_;
};

template <typename FloatType>
void registerDataset(
    const std::string& dataset,
    std::vector<FloatType> values,
    uint8_t exponent) {
  const auto name = fmt::format(
      "{}_{}", std::is_same_v<FloatType, float> ? "Float" : "Double", dataset);
  if (FLAGS_e2e) {
    registerFileDataset(name, values);
    return;
  }
  if (FLAGS_profile && FLAGS_profile_skip_unselected_setup &&
      !FLAGS_profile_filter.empty() &&
      name.find(FLAGS_profile_filter) == std::string::npos) {
    return;
  }
  auto fixture = std::make_shared<AlpBenchmarkFixture<FloatType>>(
      name, std::move(values), exponent);

  if (FLAGS_profile) {
    if (FLAGS_profile_filter.empty() ||
        name.find(FLAGS_profile_filter) != std::string::npos) {
      profileOperation(name, "DecodeConstruct", [&] { fixture->decode(); });
      profileOperation(name, "Encode", [&] {
        Buffer buffer{*benchmarkPool()};
        folly::doNotOptimizeAway(fixture->encode(buffer));
      });
      profileOperation(name, "DecodeReuse", [&] { fixture->decodeReuse(); });
      profileOperation(name, "VisitorDenseSlow", [&] {
        fixture->template decodeVisitorDense<false>();
      });
      profileOperation(name, "VisitorDenseAuto", [&] {
        fixture->template decodeVisitorDense<true>();
      });
      profileOperation(name, "VisitorFilterSlow", [&] {
        fixture->template decodeVisitorDenseFiltered<false>();
      });
      profileOperation(name, "VisitorFilterAuto", [&] {
        fixture->template decodeVisitorDenseFiltered<true>();
      });
      profileOperation(
          name, "VisitorSparse", [&] { fixture->decodeVisitorSparse(); });
    }
    return;
  }

  auto* fixturePtr = fixture.get();
  static std::vector<std::shared_ptr<AlpBenchmarkFixture<FloatType>>> fixtures;
  fixtures.push_back(std::move(fixture));

  folly::addBenchmark(
      __FILE__, fmt::format("TransformScalar_{}", name), [fixturePtr] {
        folly::doNotOptimizeAway(fixturePtr->template runTransform<false>());
        return 1;
      });
  folly::addBenchmark(
      __FILE__, fmt::format("%TransformBatch_{}", name), [fixturePtr] {
        folly::doNotOptimizeAway(fixturePtr->template runTransform<true>());
        return 1;
      });
  folly::addBenchmark(
      __FILE__, fmt::format("GridScalar_{}", name), [fixturePtr] {
        folly::doNotOptimizeAway(fixturePtr->template runGrid<false>());
        return 1;
      });
  folly::addBenchmark(
      __FILE__, fmt::format("%GridBatch_{}", name), [fixturePtr] {
        folly::doNotOptimizeAway(fixturePtr->template runGrid<true>());
        return 1;
      });
  folly::addBenchmark(__FILE__, fmt::format("Encode_{}", name), [fixturePtr] {
    Buffer buffer{*benchmarkPool()};
    folly::doNotOptimizeAway(fixturePtr->encode(buffer));
    return 1;
  });
  folly::addBenchmark(__FILE__, fmt::format("Decode_{}", name), [fixturePtr] {
    fixturePtr->decode();
    return 1;
  });
  folly::addBenchmark(
      __FILE__, fmt::format("DecodeReuse_{}", name), [fixturePtr] {
        fixturePtr->decodeReuse();
        return 1;
      });
  folly::addBenchmark(
      __FILE__, fmt::format("VisitorDenseSlow_{}", name), [fixturePtr] {
        fixturePtr->template decodeVisitorDense<false>();
        return 1;
      });
  folly::addBenchmark(
      __FILE__, fmt::format("VisitorDenseAuto_{}", name), [fixturePtr] {
        fixturePtr->template decodeVisitorDense<true>();
        return 1;
      });
  folly::addBenchmark(
      __FILE__, fmt::format("VisitorFilterSlow_{}", name), [fixturePtr] {
        fixturePtr->template decodeVisitorDenseFiltered<false>();
        return 1;
      });
  folly::addBenchmark(
      __FILE__, fmt::format("VisitorFilterAuto_{}", name), [fixturePtr] {
        fixturePtr->template decodeVisitorDenseFiltered<true>();
        return 1;
      });
  folly::addBenchmark(
      __FILE__, fmt::format("VisitorSparse_{}", name), [fixturePtr] {
        fixturePtr->decodeVisitorSparse();
        return 1;
      });
}

template <typename FloatType>
void registerDatasets() {
  if (!FLAGS_exception_pattern.empty()) {
    std::vector<FloatType> values(FLAGS_rows);
    for (uint32_t row = 0; row < FLAGS_rows; ++row) {
      const auto& pattern = FLAGS_exception_pattern;
      const bool exception = pattern == "all" ||
          (pattern == "one" && row == FLAGS_rows / 2) ||
          (pattern == "four" && row < 4) || (pattern == "five" && row < 5) ||
          (pattern == "prefix" && row < FLAGS_rows / 2) ||
          (pattern == "suffix" && row >= FLAGS_rows / 2) ||
          (pattern == "periodic" && row % 16 == 0);
      values[row] = static_cast<FloatType>(row % 1024);
      if (exception) {
        const std::array<FloatType, 4> special{
            -FloatType{0},
            std::numeric_limits<FloatType>::infinity(),
            -std::numeric_limits<FloatType>::infinity(),
            alp::toLogical<FloatType>(
                alp::toPhysical<FloatType>(
                    std::numeric_limits<FloatType>::quiet_NaN()) |
                (row % 256))};
        values[row] = special[row % special.size()];
      }
    }
    registerDataset(
        "Exceptions_" + FLAGS_exception_pattern, std::move(values), 0);
    return;
  }
  std::uniform_int_distribution<int32_t> integers(-1'000'000, 1'000'000);
  registerDataset(
      "Integers",
      makeValues<FloatType>([&](auto& random) {
        return static_cast<FloatType>(integers(random));
      }),
      0);

  std::uniform_int_distribution<int32_t> cents(0, 999'999);
  registerDataset(
      "TwoDecimal",
      makeValues<FloatType>([&](auto& random) {
        return static_cast<FloatType>(cents(random)) /
            static_cast<FloatType>(100);
      }),
      2);

  std::normal_distribution<double> prices(500.0, 120.0);
  registerDataset(
      "Prices",
      makeValues<FloatType>([&](auto& random) {
        const auto rounded = std::llround(std::abs(prices(random)) * 100.0);
        return static_cast<FloatType>(rounded) / static_cast<FloatType>(100);
      }),
      2);

  std::uniform_int_distribution<int64_t> seconds(
      1'700'000'000LL, 1'800'000'000LL);
  std::uniform_int_distribution<int32_t> milliseconds(0, 999);
  registerDataset(
      "Timestamps",
      makeValues<FloatType>([&](auto& random) {
        const auto value = seconds(random) * 1'000LL + milliseconds(random);
        return static_cast<FloatType>(value) / static_cast<FloatType>(1'000);
      }),
      3);

  const std::array<double, 8> palette{
      0.1234, 1.5000, 2.7182, 3.1415, 42.0000, 100.9999, 999.0001, 12.3456};
  std::uniform_int_distribution<uint32_t> pick(0, palette.size() - 1);
  registerDataset(
      "LowCardinality",
      makeValues<FloatType>([&](auto& random) {
        return static_cast<FloatType>(palette[pick(random)]);
      }),
      4);

  registerDataset("Sensor2Percent", makeSensorValues<FloatType>(0.02), 3);
  registerDataset(
      "MixedPrecision30Percent", makeSensorValues<FloatType>(0.30), 3);

  std::uniform_real_distribution<double> chaotic(-1e6, 1e6);
  registerDataset(
      "Chaotic",
      makeValues<FloatType>([&](auto& random) {
        return static_cast<FloatType>(chaotic(random));
      }),
      2);
}

} // namespace

int main(int argc, char** argv) {
  const folly::Init init{&argc, &argv};
  CHECK_GT(FLAGS_rows, 0);
  CHECK_LE(FLAGS_rows, std::numeric_limits<velox::vector_size_t>::max());
  CHECK_GT(FLAGS_read_batch_size, 0);
  CHECK_LT(FLAGS_visitor_start, FLAGS_rows);
  CHECK_LE(FLAGS_visitor_rows, FLAGS_rows - FLAGS_visitor_start);
  CHECK_GT(FLAGS_visitor_stride, 0);
  const std::array<std::string_view, 9> exceptionPatterns{
      "", "none", "one", "four", "five", "all", "prefix", "suffix", "periodic"};
  CHECK(
      std::find(
          exceptionPatterns.begin(),
          exceptionPatterns.end(),
          FLAGS_exception_pattern) != exceptionPatterns.end());
  CHECK_LE(
      FLAGS_read_batch_size, std::numeric_limits<velox::vector_size_t>::max());
  CHECK_GT(FLAGS_profile_trials, 0);
  CHECK_GT(FLAGS_profile_min_ms, 0);
  facebook::velox::memory::MemoryManager::initialize({});
  fmt::print(
      "Transform/encode/decode times are per {} rows; grid times are per "
      "{}-value sample. Relative rows compare batch with scalar.\n",
      FLAGS_rows,
      ALPEncoding<double>::estimateSampleSize(FLAGS_rows));
  registerDatasets<double>();
  registerDatasets<float>();
  if (!FLAGS_profile) {
    folly::runBenchmarks();
  }
}
