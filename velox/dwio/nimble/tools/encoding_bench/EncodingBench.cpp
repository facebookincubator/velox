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

// Nimble encoding benchmark: compare encoding size and decode CPU for int64
// columns in different encoding choices.
//
// Steps:
// 1. Loads int64 values from a CSV file (one value per line)
// 2. Encodes with specified encoding layout using MetaInternal compression
//    Goal: measure the output size
// 3. Measures create() cost separately (decompression + deserialization)
// 4. Decodes end-to-end (create + readWithVisitor) with 3 row patterns:
//    - dense:   create + read all rows
//    - range:   create + read middle 50%
//    - scatter: create + read random 10%
//    Goal: measure total decode time (us) including decompression
//    Single pass per pattern — no averaging, matches production behavior.
//
// Decode uses callReadWithVisitor() with AlwaysTrue filter — this is the
// same code path as the production selective reader (SelectiveColumnReader
// → IntegerColumnReader → readWithVisitor → bulkScan).
//
// Usage:
//   buck2 run fbcode//velox/dwio/nimble/tools/encoding_bench:encoding_bench --
//   \
//     --file=values.csv --encodings="Trivial,FBW,Dict[FBW]"
//
// Encoding syntax:
//   Leaf:  Trivial, FBW, RLE, Constant
//   Dict:  Dict[<indices>]  (alphabet is always Trivial)
//
// Extract values from Presto (skip header + row count):
//   meta presto.query execute \
//     --sql="SELECT col FROM table WHERE ds='2026-01-01'" \
//     --namespace=ns --output=csv | tail -n +3 > values.csv

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/ColumnVisitors.h"
#include "velox/dwio/common/SelectiveColumnReader.h"
#include "velox/dwio/common/SelectiveStructColumnReader.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingUtils.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/EncodingLayoutTestHelper.h"
#include "velox/dwio/nimble/velox/selective/ColumnReader.h"
#include "velox/dwio/nimble/velox/selective/IntegerColumnReader.h"
#include "velox/dwio/nimble/velox/selective/NimbleData.h"
#include "velox/dwio/nimble/velox/selective/ReaderBase.h"
#include "velox/dwio/nimble/velox/selective/RowSizeTracker.h"
#include "velox/vector/FlatVector.h"

#include <folly/init/Init.h>
#include <gflags/gflags.h>

DEFINE_string(file, "", "CSV file with one int64 per line");
DEFINE_string(
    encodings,
    "",
    "Comma-separated encoding specs to test. "
    "Leaf: Trivial, FBW, Varint, RLE, Constant. "
    "Dict: Dict[<indices>] (alphabet is always Trivial). "
    "Example: Trivial,FBW,Dict[FBW]. "
    "If empty, tests a default set.");

using namespace facebook;
using namespace facebook::velox;
using Clock = std::chrono::high_resolution_clock;
using RowSet = folly::Range<const vector_size_t*>;

// Expose prepareRead so we can call readWithVisitor with a custom encoding.
class IntegerColumnReaderAccessor : public nimble::IntegerColumnReader {
 public:
  using IntegerColumnReader::IntegerColumnReader;

  template <typename T>
  void
  doPrepareRead(int64_t offset, const RowSet& rows, const uint64_t* nulls) {
    this->template prepareRead<T>(offset, rows, nulls);
  }

  void advanceReadOffset(const RowSet& rows) {
    readOffset_ += rows.back() + 1;
  }
};

std::vector<int64_t> loadValues(const std::string& path) {
  std::vector<int64_t> values;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty()) {
      values.push_back(std::stoll(line));
    }
  }
  return values;
}

std::string_view encodeWithLayout(
    velox::memory::MemoryPool& pool,
    std::span<const int64_t> data,
    nimble::EncodingLayout layout,
    const std::optional<nimble::CompressionOptions>& compressionOpts,
    nimble::Buffer& buffer) {
  nimble::ManualEncodingSelectionPolicyFactory fallbackFactory{
      nimble::ManualEncodingSelectionPolicyFactory::
          defaultEncodingReadFactors(),
      compressionOpts};

  auto policy =
      std::make_unique<nimble::ReplayedEncodingSelectionPolicy<int64_t>>(
          std::move(layout),
          compressionOpts,
          [&fallbackFactory](nimble::DataType dataType) {
            return fallbackFactory.createPolicy(dataType);
          });

  return nimble::EncodingFactory::encode<int64_t>(
      std::move(policy), data, buffer);
}

std::string_view encodeAutoSelect(
    velox::memory::MemoryPool& pool,
    std::span<const int64_t> data,
    const std::optional<nimble::CompressionOptions>& compressionOpts,
    nimble::Buffer& buffer) {
  auto policy =
      std::make_unique<nimble::ManualEncodingSelectionPolicy<int64_t>>(
          nimble::ManualEncodingSelectionPolicyFactory::
              defaultEncodingReadFactors(),
          compressionOpts,
          std::nullopt);

  return nimble::EncodingFactory::encode<int64_t>(
      std::move(policy), data, buffer);
}

std::string encodingTypeName(nimble::EncodingType type) {
  switch (type) {
    case nimble::EncodingType::Trivial:
      return "Trivial";
    case nimble::EncodingType::FixedBitWidth:
      return "FBW";
    case nimble::EncodingType::Dictionary:
      return "Dict";
    case nimble::EncodingType::RLE:
      return "RLE";
    case nimble::EncodingType::Varint:
      return "Varint";
    case nimble::EncodingType::Constant:
      return "Constant";
    case nimble::EncodingType::MainlyConstant:
      return "MainlyConstant";
    default:
      return "Unknown";
  }
}

std::optional<nimble::EncodingType> parseEncodingType(const std::string& name) {
  if (name == "Trivial")
    return nimble::EncodingType::Trivial;
  if (name == "FBW")
    return nimble::EncodingType::FixedBitWidth;
  if (name == "Dict" || name == "Dictionary")
    return nimble::EncodingType::Dictionary;
  if (name == "RLE")
    return nimble::EncodingType::RLE;
  if (name == "Varint")
    return nimble::EncodingType::Varint;
  if (name == "Constant")
    return nimble::EncodingType::Constant;
  if (name == "MainlyConstant")
    return nimble::EncodingType::MainlyConstant;
  return std::nullopt;
}

struct EncodingSpec {
  std::string name;
  std::function<nimble::EncodingLayout(nimble::CompressionType)> makeLayout;
};

std::optional<EncodingSpec> parseEncodingSpec(const std::string& spec) {
  if (spec.substr(0, 4) == "Dict" && spec.find('[') != std::string::npos) {
    auto bracketStart = spec.find('[');
    auto bracketEnd = spec.find(']');
    if (bracketEnd == std::string::npos) {
      return std::nullopt;
    }
    auto indicesName =
        spec.substr(bracketStart + 1, bracketEnd - bracketStart - 1);

    auto indicesType = parseEncodingType(indicesName);
    if (!indicesType) {
      return std::nullopt;
    }

    auto idxType = *indicesType;
    std::string displayName = "Dict[Trivial{C}[" + indicesName + "{C}]]";

    return EncodingSpec{
        .name = displayName,
        .makeLayout = [idxType](nimble::CompressionType comp) {
          return nimble::EncodingLayout{
              nimble::EncodingType::Dictionary,
              {},
              nimble::CompressionType::Uncompressed,
              {
                  nimble::EncodingLayout{
                      nimble::EncodingType::Trivial, {}, comp},
                  nimble::EncodingLayout{idxType, {}, comp},
              }};
        }};
  }

  auto type = parseEncodingType(spec);
  if (!type) {
    return std::nullopt;
  }

  auto encType = *type;
  std::string displayName = spec + "{C}";

  return EncodingSpec{
      .name = displayName,
      .makeLayout = [encType](nimble::CompressionType comp) {
        return nimble::EncodingLayout{encType, {}, comp};
      }};
}

std::vector<EncodingSpec> parseEncodingSpecs(const std::string& input) {
  std::vector<EncodingSpec> specs;
  std::istringstream stream(input);
  std::string token;
  while (std::getline(stream, token, ',')) {
    auto start = token.find_first_not_of(" \t");
    auto end = token.find_last_not_of(" \t");
    if (start == std::string::npos) {
      continue;
    }
    token = token.substr(start, end - start + 1);

    auto spec = parseEncodingSpec(token);
    if (spec) {
      specs.push_back(std::move(*spec));
    } else {
      std::cerr << "Warning: unknown encoding spec '" << token << "', skipping"
                << std::endl;
    }
  }
  return specs;
}

std::vector<EncodingSpec> defaultEncodingSpecs() {
  return {
      *parseEncodingSpec("Trivial"),
      *parseEncodingSpec("FBW"),
      *parseEncodingSpec("Dict[FBW]"),
  };
}

// Holds the SelectiveColumnReader infrastructure needed for readWithVisitor.
struct ReaderContext {
  std::string fileData;
  std::shared_ptr<nimble::ReaderBase> readerBase;
  std::unique_ptr<nimble::StripeStreams> streams;
  dwio::common::SplitStats stats{dwio::common::FileFormat::NIMBLE};
  std::unique_ptr<nimble::RowSizeTracker> rowSizeTracker;
  nimble::EncodingFactory encodingFactory;
  std::shared_ptr<common::ScanSpec> scanSpec;
  std::unique_ptr<dwio::common::SelectiveColumnReader> root;
  IntegerColumnReaderAccessor* intReader = nullptr;
};

std::unique_ptr<ReaderContext> makeReaderContext(
    velox::memory::MemoryPool& pool,
    const std::vector<int64_t>& values) {
  auto ctx = std::make_unique<ReaderContext>();

  auto vector = BaseVector::create(BIGINT(), values.size(), &pool);
  auto* flat = vector->asFlatVector<int64_t>();
  for (size_t i = 0; i < values.size(); ++i) {
    flat->set(i, values[i]);
  }
  auto rowVector = std::make_shared<RowVector>(
      &pool,
      ROW({"c0"}, {BIGINT()}),
      nullptr,
      values.size(),
      std::vector<VectorPtr>{vector});

  ctx->fileData = nimble::test::createNimbleFile(pool, rowVector);

  auto readFile = std::make_shared<InMemoryReadFile>(ctx->fileData);
  dwio::common::ReaderOptions readerOpts(&pool);
  auto ioStats = std::make_shared<velox::io::IoStatistics>();
  readerOpts.setDataIoStats(ioStats);
  readerOpts.setMetadataIoStats(ioStats);
  ctx->readerBase = nimble::ReaderBase::create(
      std::make_unique<dwio::common::BufferedInput>(readFile, pool),
      readerOpts);

  ctx->streams = std::make_unique<nimble::StripeStreams>(ctx->readerBase);
  ctx->streams->setStripe(0);

  ctx->rowSizeTracker = std::make_unique<nimble::RowSizeTracker>(
      ctx->readerBase->fileSchemaWithId());

  auto rowType = ROW({"c0"}, {BIGINT()});
  ctx->scanSpec = std::make_shared<common::ScanSpec>("root");
  auto& scanSpec = ctx->scanSpec;
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());

  nimble::NimbleParams params(
      pool,
      ctx->stats,
      ctx->readerBase->nimbleSchema(),
      *ctx->streams,
      ctx->rowSizeTracker.get(),
      ctx->encodingFactory);

  ctx->root = nimble::buildColumnReader(
      rowType, ctx->readerBase->fileSchemaWithId(), params, *scanSpec, true);
  ctx->root->setIsTopLevel();
  ctx->streams->load();

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(
          ctx->root.get());
  ctx->intReader = static_cast<IntegerColumnReaderAccessor*>(
      dynamic_cast<nimble::IntegerColumnReader*>(structReader->children()[0]));

  return ctx;
}

template <typename V>
nimble::ReadWithVisitorParams makeReadWithVisitorParams(
    V& visitor,
    const RowSet& rows,
    velox::memory::MemoryPool* memPool) {
  nimble::ReadWithVisitorParams params{};
  params.prepareResultNulls = [&visitor, &rows] {
    visitor.reader().prepareNulls(rows, true, 8);
  };
  params.setReturnNullsMode = [&visitor, &rows] {
    visitor.reader().setReturnNullsMode(rows);
  };
  const auto numRows = visitor.numRows();
  params.makeReaderNulls = [memPool, numRows, &visitor] {
    auto& nulls = visitor.reader().nullsInReadRange();
    if (!nulls) {
      nulls = velox::AlignedBuffer::allocate<bool>(
          visitor.rowAt(numRows - 1) + 1, memPool, velox::bits::kNotNull);
    }
    return nulls->template asMutable<uint64_t>();
  };
  params.numScanned = 0;
  return params;
}

// End-to-end: create + readWithVisitor in one timed pass
double runEndToEnd(
    velox::memory::MemoryPool& pool,
    std::string_view serialized,
    ReaderContext& ctx,
    const std::vector<vector_size_t>& rowIndices) {
  RowSet rows(rowIndices.data(), rowIndices.size());
  ctx.intReader->doPrepareRead<int64_t>(0, rows, nullptr);

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(ctx.intReader);
  constexpr bool kIsDense = true;
  dwio::common::ColumnVisitor<
      int64_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, ctx.intReader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, &pool);

  auto start = Clock::now();
  nimble::EncodingFactory factory{{}};
  auto encoding = factory.create(pool, serialized, nullptr);
  nimble::callReadWithVisitor(*encoding, visitor, params);
  auto end = Clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
      .count();
}

struct DecodeResult {
  double createUs;
  double denseUs;
  double rangeUs;
  double scatterUs;
};

DecodeResult benchmarkDecode(
    velox::memory::MemoryPool& pool,
    std::string_view serialized,
    ReaderContext& ctx,
    size_t totalRows) {
  std::vector<vector_size_t> denseRows(totalRows);
  std::iota(denseRows.begin(), denseRows.end(), 0);

  vector_size_t rangeStart = totalRows / 4;
  vector_size_t rangeCount = totalRows / 2;
  std::vector<vector_size_t> rangeRows(rangeCount);
  std::iota(rangeRows.begin(), rangeRows.end(), rangeStart);

  size_t scatterCount = totalRows / 10;
  std::vector<vector_size_t> scatterRows(scatterCount);
  std::mt19937 rng(42);
  for (auto& r : scatterRows) {
    r = rng() % totalRows;
  }
  std::sort(scatterRows.begin(), scatterRows.end());
  scatterRows.erase(
      std::unique(scatterRows.begin(), scatterRows.end()), scatterRows.end());

  nimble::EncodingFactory factory{{}};
  auto createStart = Clock::now();
  auto encoding = factory.create(pool, serialized, nullptr);
  auto createEnd = Clock::now();
  double createNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        createEnd - createStart)
                        .count();

  return DecodeResult{
      .createUs = createNs / 1000,
      .denseUs = runEndToEnd(pool, serialized, ctx, denseRows) / 1000,
      .rangeUs = runEndToEnd(pool, serialized, ctx, rangeRows) / 1000,
      .scatterUs = runEndToEnd(pool, serialized, ctx, scatterRows) / 1000,
  };
}

void printResult(
    const std::string& label,
    std::string_view serialized,
    size_t rowCount,
    velox::memory::MemoryPool& pool,
    ReaderContext& ctx) {
  auto type =
      nimble::encoding::peek<uint8_t, nimble::EncodingType>(serialized.data());
  auto result = benchmarkDecode(pool, serialized, ctx, rowCount);
  std::cout << "  " << label << ": " << serialized.size() << " bytes ("
            << static_cast<double>(serialized.size()) / rowCount << " B/row)"
            << " | create=" << std::fixed << std::setprecision(0)
            << result.createUs << " us"
            << " | e2e: dense=" << result.denseUs << " us"
            << ", range=" << result.rangeUs << " us"
            << ", scatter=" << result.scatterUs << " us"
            << " [" << encodingTypeName(type) << "]" << std::endl;
}

void runBenchmark(
    velox::memory::MemoryPool& pool,
    std::span<const int64_t> values,
    const std::optional<nimble::CompressionOptions>& compressionOpts,
    nimble::CompressionType compType,
    const std::vector<EncodingSpec>& specs,
    ReaderContext& ctx) {
  std::cout << "\n=== MetaInternal (Zstrong) Compression ===" << std::endl;

  for (const auto& spec : specs) {
    nimble::Buffer buffer(pool);
    try {
      auto layout = spec.makeLayout(compType);
      auto serialized = encodeWithLayout(
          pool, values, std::move(layout), compressionOpts, buffer);
      printResult(spec.name, serialized, values.size(), pool, ctx);
    } catch (const std::exception& e) {
      std::cout << "  " << spec.name << ": FAILED - " << e.what() << std::endl;
    }
  }

  nimble::Buffer buffer(pool);
  auto serialized = encodeAutoSelect(pool, values, compressionOpts, buffer);
  printResult("AutoSelect", serialized, values.size(), pool, ctx);
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);

  if (FLAGS_file.empty()) {
    std::cerr << "Usage: encoding_bench --file=<csv> "
              << "[--encodings=Trivial,Dict[FBW],...]" << std::endl;
    return 1;
  }

  auto values = loadValues(FLAGS_file);
  std::cout << "Loaded " << values.size() << " values" << std::endl;
  std::cout << "Unique: "
            << std::set<int64_t>(values.begin(), values.end()).size()
            << std::endl;

  auto specs = FLAGS_encodings.empty() ? defaultEncodingSpecs()
                                       : parseEncodingSpecs(FLAGS_encodings);
  std::cout << "Encodings: ";
  for (size_t i = 0; i < specs.size(); ++i) {
    if (i > 0)
      std::cout << ", ";
    std::cout << specs[i].name;
  }
  std::cout << ", AutoSelect" << std::endl;
  std::cout << "Decode via readWithVisitor (selective reader path)"
            << std::endl;
  std::cout << "  dense = create+read all, range = create+read 50%,"
            << " scatter = create+read 10%" << std::endl;

  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();

  auto ctx = makeReaderContext(*pool, values);

  nimble::CompressionOptions compOpts;
  compOpts.compressionAcceptRatio = 1.0f;

  runBenchmark(
      *pool,
      std::span(values),
      compOpts,
      nimble::CompressionType::MetaInternal,
      specs,
      *ctx);

  return 0;
}
