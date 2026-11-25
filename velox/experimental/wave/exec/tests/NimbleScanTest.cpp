/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include <cuda_runtime.h> // @manual
#include <unistd.h>
#include <atomic>
#include <fstream>
#include <set>
#include <thread>

#include "velox/common/file/File.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/nimble/writer/FlushPolicy.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/experimental/wave/dwio/nimble/NimbleSplitReader.h"
#include "velox/experimental/wave/dwio/nimble/NimbleStream.h"
#include "velox/experimental/wave/exec/ToWave.h"
#include "velox/experimental/wave/exec/WaveHiveDataSource.h"

DECLARE_int32(wave_max_reader_batch_rows);
DECLARE_int32(wave_reader_rows_per_tb);

DEFINE_int64(
    nimble_scan_test_max_rss_mb,
    8192,
    "The test aborts if its resident memory exceeds this many MB");

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
namespace nimble = facebook::nimble;
using facebook::nimble::CompressionType;
using facebook::nimble::EncodingLayout;
using facebook::nimble::EncodingLayoutTree;
using facebook::nimble::EncodingType;
using facebook::nimble::Kind;

namespace {

// Aborts the process when its resident memory exceeds
// --nimble_scan_test_max_rss_mb. The tests need little memory. A runaway
// allocation, e.g. from a corrupt size read from device, fails the test
// instead of exhausting the host.
class RssWatchdog {
 public:
  static void start() {
    static RssWatchdog watchdog;
  }

 private:
  RssWatchdog() : thread_([this]() { run(); }) {}

  ~RssWatchdog() {
    stop_ = true;
    thread_.join();
  }

  void run() {
    const int64_t pageSize = sysconf(_SC_PAGESIZE);
    const int64_t limit = FLAGS_nimble_scan_test_max_rss_mb << 20;
    while (!stop_) {
      std::ifstream statm("/proc/self/statm");
      int64_t size = 0;
      int64_t resident = 0;
      statm >> size >> resident;
      if (resident * pageSize > limit) {
        fprintf(
            stderr,
            "RssWatchdog: resident memory %ld MB exceeds %ld MB, aborting\n",
            (resident * pageSize) >> 20,
            limit >> 20);
        std::abort();
      }
      // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  std::atomic<bool> stop_{false};
  std::thread thread_;
};

EncodingLayout leaf(EncodingType type) {
  return EncodingLayout{type, {}, CompressionType::Uncompressed};
}

// Dictionary with a Trivial alphabet and FixedBitWidth indices.
EncodingLayout dictionary() {
  return EncodingLayout{
      EncodingType::Dictionary,
      {},
      CompressionType::Uncompressed,
      {leaf(EncodingType::Trivial), leaf(EncodingType::FixedBitWidth)}};
}

EncodingLayout mainlyConstant() {
  return EncodingLayout{
      EncodingType::MainlyConstant,
      {},
      CompressionType::Uncompressed,
      {std::nullopt, leaf(EncodingType::FixedBitWidth)}};
}

EncodingLayout rle() {
  return EncodingLayout{
      EncodingType::RLE,
      {},
      CompressionType::Uncompressed,
      {leaf(EncodingType::FixedBitWidth), leaf(EncodingType::Trivial)}};
}

// Chooses the encodings of streams and children that the layout tree leaves
// open among the encodings of the GPU reader, without compression.
nimble::ManualEncodingSelectionPolicyFactory gpuEncodingsPolicyFactory(
    std::optional<nimble::CompressionOptions> compression) {
  const std::set<EncodingType> supported = {
      EncodingType::Constant,
      EncodingType::Trivial,
      EncodingType::FixedBitWidth,
      EncodingType::MainlyConstant,
      EncodingType::SparseBool,
      EncodingType::Dictionary,
      EncodingType::RLE,
  };
  std::vector<std::pair<EncodingType, float>> readFactors;
  for (const auto& [encoding, factor] : nimble::
           ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors()) {
    if (supported.contains(encoding)) {
      readFactors.emplace_back(encoding, factor);
    }
  }
  return nimble::ManualEncodingSelectionPolicyFactory(
      readFactors, std::move(compression));
}

// Returns 'layout' with 'compression' for all its leaves.
EncodingLayout withCompression(
    const EncodingLayout& layout,
    CompressionType compression) {
  std::vector<std::optional<const EncodingLayout>> children;
  for (uint8_t i = 0; i < layout.childrenCount(); ++i) {
    const auto& child = layout.child(i);
    if (child.has_value()) {
      children.emplace_back(withCompression(*child, compression));
    } else {
      children.emplace_back(std::nullopt);
    }
  }
  return EncodingLayout{
      layout.encodingType(), layout.config(), compression, std::move(children)};
}

// Options that compress every leaf with 'compression', or no compression for
// Uncompressed.
std::optional<nimble::CompressionOptions> compressionOptions(
    CompressionType compression) {
  if (compression == CompressionType::Uncompressed) {
    return std::nullopt;
  }
  nimble::CompressionOptions options;
  options.compressionAcceptRatio = 100;
  options.compressionType = compression;
  options.zstdMinCompressionSize = 0;
  options.internalMinCompressionSize = 0;
  options.openzlMinCompressionSize = 0;
  return options;
}

// Returns a layout tree for 'type' that leaves all encodings to the
// encoding selection policy.
EncodingLayoutTree openLayout(const TypePtr& type) {
  std::vector<EncodingLayoutTree> children;
  children.reserve(type->size());
  for (auto i = 0; i < type->size(); ++i) {
    children.push_back(openLayout(type->childAt(i)));
  }
  switch (type->kind()) { // NOLINT(clang-diagnostic-switch-enum)
    case TypeKind::ROW:
      return EncodingLayoutTree{Kind::Row, {}, "", std::move(children)};
    case TypeKind::ARRAY:
      return EncodingLayoutTree{Kind::Array, {}, "", std::move(children)};
    case TypeKind::MAP:
      return EncodingLayoutTree{Kind::Map, {}, "", std::move(children)};
    default:
      return EncodingLayoutTree{Kind::Scalar, {}, ""};
  }
}

EncodingLayoutTree scalarLayout(const EncodingLayout& layout) {
  return EncodingLayoutTree{
      Kind::Scalar,
      {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream, layout}},
      ""};
}

class NimbleScanTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    RssWatchdog::start();
    HiveConnectorTestBase::SetUp();
    if (int device; cudaGetDevice(&device) != cudaSuccess) {
      GTEST_SKIP() << "No CUDA detected, skipping all tests";
    }
    wave::registerWave();
    wave::WaveHiveDataSource::registerConnector();
    wave::NimbleSplitReader::registerFactory();
    FLAGS_wave_max_reader_batch_rows = 5'000;
    FLAGS_wave_reader_rows_per_tb = 1024;
  }

  void TearDown() override {
    vectors_.clear();
    HiveConnectorTestBase::TearDown();
  }

  // Writes 'vectors' to a Nimble file with 'compression_'. Each vector is a
  // chunk of every
  // stream. A stripe has 'vectorsPerStripe' vectors. 'layouts' gives the
  // encoding of each column. The encoding of a column without a layout is
  // chosen by gpuEncodingsPolicyFactory() for each chunk.
  std::shared_ptr<connector::ConnectorSplit> writeFile(
      const std::vector<RowVectorPtr>& vectors,
      const std::vector<std::optional<EncodingLayout>>& layouts,
      int32_t vectorsPerStripe) {
    std::vector<EncodingLayoutTree> columns;
    columns.reserve(layouts.size());
    for (const auto& layout : layouts) {
      columns.push_back(
          layout.has_value()
              ? scalarLayout(withCompression(*layout, compression_))
              : EncodingLayoutTree{Kind::Scalar, {}, ""});
    }
    return writeFile(vectors, std::move(columns), vectorsPerStripe);
  }

  // Writes 'vectors' with the layout tree of each top level column in
  // 'columns'. The columns in 'flatMapColumns' are written as flat maps.
  std::shared_ptr<connector::ConnectorSplit> writeFile(
      const std::vector<RowVectorPtr>& vectors,
      std::vector<EncodingLayoutTree> columns,
      int32_t vectorsPerStripe,
      const std::vector<std::string>& flatMapColumns = {}) {
    auto compression = compressionOptions(compression_);
    nimble::WriterOptions options{
        .encodingLayoutTree =
            EncodingLayoutTree{Kind::Row, {}, "", std::move(columns)},
        .compressionOptions = compression.value_or(
            nimble::CompressionOptions{
                .compressionAcceptRatio = 0,
                .compressionType = CompressionType::Uncompressed}),
        .minStreamChunkRawSize = 0,
        .encodingSelectionPolicyCreator =
            [factory = gpuEncodingsPolicyFactory(compression)](
                nimble::DataType type) { return factory.createPolicy(type); },
        .flushPolicyFactory =
            []() {
              return std::make_unique<nimble::LambdaFlushPolicy>(
                  [](const auto&) { return false; },
                  [](const auto&) { return true; });
            },
    };
    for (const auto& name : flatMapColumns) {
      options.flatMapColumns[name] = {};
    }
    auto file = common::testutil::TempFilePath::create();
    nimble::Writer writer(
        vectors[0]->type(),
        std::make_unique<LocalWriteFile>(file->getPath(), false, false),
        *rootPool_,
        std::move(options));
    for (auto i = 0; i < vectors.size(); ++i) {
      writer.write(vectors[i]);
      if ((i + 1) % vectorsPerStripe == 0) {
        writer.flush();
      }
    }
    writer.close();
    files_.push_back(file);
    return connector::hive::HiveConnectorSplitBuilder(file->getPath())
        .connectorId(kHiveConnectorId)
        .fileFormat(dwio::common::FileFormat::NIMBLE)
        .build();
  }

  // Makes 'numVectors' vectors of 'vectorSize' rows with one column per
  // encoding in 'scalarEncodings()'.
  std::vector<RowVectorPtr> makeVectors(
      int32_t numVectors,
      int32_t vectorSize) {
    std::vector<RowVectorPtr> vectors;
    int64_t row = 0;
    for (auto i = 0; i < numVectors; ++i) {
      const int64_t first = row;
      vectors.push_back(makeRowVector(
          {"trivial64",
           "fbw64",
           "trivial32",
           "fbw32",
           "dbl",
           "flt",
           "dict",
           "constant",
           "rn"},
          {
              makeFlatVector<int64_t>(
                  vectorSize,
                  [&](auto j) { return (first + j) * 7'919 - 50'000'000; }),
              makeFlatVector<int64_t>(
                  vectorSize,
                  [&](auto j) { return 1'000 * i + (first + j) % (100 << i); }),
              makeFlatVector<int32_t>(
                  vectorSize,
                  [&](auto j) { return (first + j) * 31 - 1'000'000; }),
              makeFlatVector<int32_t>(
                  vectorSize, [&](auto j) { return (first + j) % 1'000; }),
              makeFlatVector<double>(
                  vectorSize, [&](auto j) { return (first + j) * 1.5; }),
              makeFlatVector<float>(
                  vectorSize, [&](auto j) { return (first + j) * 0.25f; }),
              makeFlatVector<int64_t>(
                  vectorSize,
                  [&](auto j) {
                    return ((first + j) % 13) * 1'000'000'007LL + i;
                  }),
              makeFlatVector<int64_t>(
                  vectorSize, [&](auto /*j*/) { return 77 + i; }),
              makeFlatVector<int64_t>(
                  vectorSize, [&](auto j) { return first + j; }),
          }));
      row += vectorSize;
    }
    return vectors;
  }

  static std::vector<std::optional<EncodingLayout>> scalarEncodings() {
    return {
        leaf(EncodingType::Trivial),
        leaf(EncodingType::FixedBitWidth),
        leaf(EncodingType::Trivial),
        leaf(EncodingType::FixedBitWidth),
        leaf(EncodingType::Trivial),
        leaf(EncodingType::Trivial),
        dictionary(),
        leaf(EncodingType::Constant),
        leaf(EncodingType::Trivial),
    };
  }

  std::shared_ptr<connector::ConnectorSplit> makeScalarFile() {
    vectors_ = makeVectors(8, 2'500);
    createDuckDbTable(vectors_);
    return writeFile(vectors_, scalarEncodings(), 3);
  }

  // Makes vectors with nulls at different densities and runs, a mainly
  // constant column and a column of runs. The writer wraps the encoding of a
  // column with nulls in a Nullable encoding and chooses the encoding of the
  // nulls.
  std::shared_ptr<connector::ConnectorSplit> makeNullsAndRunsFile() {
    constexpr int32_t kVectorSize = 2'500;
    int64_t row = 0;
    for (auto i = 0; i < 8; ++i) {
      const int64_t first = row;
      vectors_.push_back(makeRowVector(
          {"nulls7", "nulls300", "null_runs", "mostly5", "runs", "rn"},
          {
              makeFlatVector<int64_t>(
                  kVectorSize,
                  [&](auto j) { return (first + j) * 3; },
                  [&](auto j) { return (first + j) % 7 == 0; }),
              makeFlatVector<int32_t>(
                  kVectorSize,
                  [&](auto j) { return (first + j) % 1'000; },
                  [&](auto j) { return (first + j) % 300 == 0; }),
              makeFlatVector<int64_t>(
                  kVectorSize,
                  [&](auto j) { return ((first + j) % 9) * 1'000'000'007LL; },
                  [&](auto j) { return ((first + j) / 100) % 3 == 0; }),
              makeFlatVector<int64_t>(
                  kVectorSize,
                  [&](auto j) {
                    return (first + j) % 11 == 0 ? first + j : 5;
                  }),
              makeFlatVector<int64_t>(
                  kVectorSize, [&](auto j) { return ((first + j) / 37) * 3; }),
              makeFlatVector<int64_t>(
                  kVectorSize, [&](auto j) { return first + j; }),
          }));
      row += kVectorSize;
    }
    createDuckDbTable(vectors_);
    return writeFile(
        vectors_,
        {leaf(EncodingType::Trivial),
         leaf(EncodingType::FixedBitWidth),
         dictionary(),
         mainlyConstant(),
         rle(),
         leaf(EncodingType::Trivial)},
        3);
  }

  // Makes a column whose chunks have different encodings: constant, few
  // distinct values, runs, mostly one value and a wide range.
  std::shared_ptr<connector::ConnectorSplit> makeMixedFile() {
    constexpr int32_t kVectorSize = 2'500;
    int64_t row = 0;
    for (auto i = 0; i < 10; ++i) {
      const int64_t first = row;
      vectors_.push_back(makeRowVector(
          {"mixed", "rn"},
          {
              makeFlatVector<int64_t>(
                  kVectorSize,
                  [&](auto j) -> int64_t {
                    const int64_t n = first + j;
                    switch (i % 5) {
                      case 0:
                        return 1'000 + i;
                      case 1:
                        return (n * 7'919 % 5) * 1'000'000'007LL;
                      case 2:
                        return n / 300;
                      case 3:
                        return n % 97 == 0 ? n : 42;
                      default:
                        return n * 1'000'003;
                    }
                  }),
              makeFlatVector<int64_t>(
                  kVectorSize, [&](auto j) { return first + j; }),
          }));
      row += kVectorSize;
    }
    createDuckDbTable(vectors_);
    return writeFile(vectors_, {std::nullopt, leaf(EncodingType::Trivial)}, 5);
  }

  // Makes a nullable struct with a nullable nested struct:
  // s ROW(a BIGINT, b INTEGER, t ROW(c BIGINT)).
  std::shared_ptr<connector::ConnectorSplit> makeStructFile() {
    constexpr int32_t kVectorSize = 2'500;
    int64_t row = 0;
    for (auto i = 0; i < 8; ++i) {
      const int64_t first = row;
      auto c = makeFlatVector<int64_t>(
          kVectorSize,
          [&](auto j) { return (first + j) * 11; },
          [&](auto j) { return (first + j) % 13 == 0; });
      auto t = makeRowVector(
          {"c"}, {c}, [&](auto j) { return (first + j) % 3 == 0; });
      auto s = makeRowVector(
          {"a", "b", "t"},
          {makeFlatVector<int64_t>(
               kVectorSize, [&](auto j) { return (first + j) * 7; }),
           makeFlatVector<int32_t>(
               kVectorSize,
               [&](auto j) { return (first + j) % 1'000; },
               [&](auto j) { return (first + j) % 17 == 0; }),
           t},
          [&](auto j) { return (first + j) % 5 == 0; });
      vectors_.push_back(makeRowVector(
          {"s", "rn"}, {s, makeFlatVector<int64_t>(kVectorSize, [&](auto j) {
                          return first + j;
                        })}));
      row += kVectorSize;
    }
    createDuckDbTable(vectors_);
    auto scalar = []() { return EncodingLayoutTree{Kind::Scalar, {}, ""}; };
    std::vector<EncodingLayoutTree> columns;
    columns.push_back(
        EncodingLayoutTree{
            Kind::Row,
            {},
            "",
            {scalar(),
             scalar(),
             EncodingLayoutTree{Kind::Row, {}, "", {scalar()}}}});
    columns.push_back(scalar());
    return writeFile(vectors_, std::move(columns), 3);
  }

  // Makes m MAP(INTEGER, BIGINT), written as a flat map, with nulls. Key k
  // in 1..5 is in a row if the row number is not divisible by k + 1.
  std::shared_ptr<connector::ConnectorSplit> makeFlatMapFile() {
    constexpr int32_t kVectorSize = 2'500;
    constexpr int32_t kNumKeys = 5;
    int64_t row = 0;
    for (auto i = 0; i < 6; ++i) {
      const int64_t first = row;
      std::vector<vector_size_t> offsets;
      std::vector<int32_t> keys;
      std::vector<int64_t> values;
      for (auto j = 0; j < kVectorSize; ++j) {
        offsets.push_back(static_cast<int32_t>(keys.size()));
        for (int32_t key = 1; key <= kNumKeys; ++key) {
          if ((first + j) % (key + 1) != 0) {
            keys.push_back(key);
            values.push_back((first + j) * 10 + key);
          }
        }
      }
      auto map = makeMapVector(
          offsets,
          makeFlatVector<int32_t>(keys),
          makeFlatVector<int64_t>(values),
          {});
      for (auto j = 0; j < kVectorSize; ++j) {
        if ((first + j) % 7 == 3) {
          map->setNull(j, true);
        }
      }
      vectors_.push_back(makeRowVector(
          {"m", "rn"}, {map, makeFlatVector<int64_t>(kVectorSize, [&](auto j) {
                          return first + j;
                        })}));
      row += kVectorSize;
    }
    std::vector<EncodingLayoutTree> columns;
    columns.push_back(EncodingLayoutTree{Kind::FlatMap, {}, ""});
    columns.push_back(EncodingLayoutTree{Kind::Scalar, {}, ""});
    return writeFile(vectors_, std::move(columns), 3, {"m"});
  }

  // Returns element_at(m, key) for each row of 'vectors_' from the definition
  // of the data in makeFlatMapFile().
  VectorPtr expectedElements(int32_t key) {
    std::vector<std::optional<int64_t>> result;
    int64_t row = 0;
    for (const auto& vector : vectors_) {
      for (auto j = 0; j < vector->size(); ++j, ++row) {
        if (row % 7 == 3 || row % (key + 1) == 0) {
          result.emplace_back(std::nullopt);
        } else {
          result.emplace_back(row * 10 + key);
        }
      }
    }
    return makeNullableFlatVector<int64_t>(result);
  }

  void assertQuery(
      const core::PlanNodePtr& plan,
      const std::shared_ptr<connector::ConnectorSplit>& split,
      const std::string& sql) {
    AssertQueryBuilder(plan, duckDbQueryRunner_)
        .split(split)
        .assertResults(sql);
  }

  // Opens the last written file.
  std::shared_ptr<nimble::ReaderBase> openReader() {
    return nimble::ReaderBase::create(
        std::make_unique<dwio::common::BufferedInput>(
            std::make_shared<LocalReadFile>(files_.back()->getPath()), *pool_),
        dwio::common::ReaderOptions(pool_.get()));
  }

  // Returns true if stripe 'stripe' of the last written file has stream
  // 'streamId'.
  bool hasStream(uint32_t stripe, uint32_t streamId) {
    auto readerBase = openReader();
    const auto& tablet = readerBase->tablet();
    return tablet.streamLocation(tablet.stripeIdentifier(stripe), streamId)
               .size > 0;
  }

  // Returns the parsed data streams of the top level columns of each stripe
  // of the last written file.
  std::vector<std::vector<wave::NimbleStream>> parseStripes() {
    auto file = std::make_shared<LocalReadFile>(files_.back()->getPath());
    auto readerBase = openReader();
    const auto& tablet = readerBase->tablet();
    const auto& row = readerBase->nimbleSchema()->asRow();
    std::vector<std::vector<wave::NimbleStream>> stripes;
    for (uint32_t stripe = 0; stripe < tablet.stripeCount(); ++stripe) {
      auto& columns = stripes.emplace_back();
      for (const auto& child : row.children()) {
        auto location = tablet.streamLocation(
            tablet.stripeIdentifier(stripe),
            child->asScalar().scalarDescriptor().offset());
        std::string bytes(location.size, 0);
        file->pread(
            tablet.stripeOffset(stripe) + location.offset,
            location.size,
            bytes.data());
        columns.push_back(wave::NimbleStream::parse(bytes, pool_.get()));
      }
    }
    return stripes;
  }

  // Returns a string of 'row'. Strings are up to 4 bytes, 7 to 12 bytes
  // (inline in a StringView), longer, and for a few rows over 64KB.
  static std::string stringValue(int64_t row) {
    if (row % 997 == 5) {
      return std::string(70'000 + row % 13, static_cast<char>('a' + row % 26));
    }
    if (row % 4 == 0) {
      return fmt::format("s{}", row % 1'000);
    }
    if (row % 4 == 1) {
      return fmt::format("inl{:09}", row % 1'000'000'000)
          .substr(0, 7 + row % 6);
    }
    return fmt::format("string {} with some padding {}", row, row % 7);
  }

  // Makes a vector of 'size' strings from 'valueAt' with nulls at 'isNullAt'.
  VectorPtr makeStrings(
      vector_size_t size,
      const std::function<std::string(vector_size_t)>& valueAt,
      const std::function<bool(vector_size_t)>& isNullAt = nullptr) {
    std::vector<std::string> values;
    values.reserve(size);
    for (auto i = 0; i < size; ++i) {
      values.push_back(valueAt(i));
    }
    return makeFlatVector<StringView>(
        size,
        [&](auto i) { return StringView(values[i]); },
        isNullAt ? std::function<bool(vector_size_t)>(isNullAt) : nullptr);
  }

  // Makes 'numRows' arrays. Row i is null if 'isNullAt(i)' and has
  // 'sizeAt(i)' elements otherwise. 'makeElements(n)' makes the n elements of
  // all rows.
  VectorPtr makeArrays(
      vector_size_t numRows,
      const std::function<vector_size_t(vector_size_t)>& sizeAt,
      const std::function<bool(vector_size_t)>& isNullAt,
      const std::function<VectorPtr(vector_size_t)>& makeElements) {
    std::vector<vector_size_t> offsets;
    std::vector<vector_size_t> nulls;
    vector_size_t numElements = 0;
    for (auto i = 0; i < numRows; ++i) {
      offsets.push_back(numElements);
      if (isNullAt(i)) {
        nulls.push_back(i);
      } else {
        numElements += sizeAt(i);
      }
    }
    return makeArrayVector(offsets, makeElements(numElements), nulls);
  }

  // Like makeArrays() for maps. Keys are distinct in each map.
  VectorPtr makeMaps(
      vector_size_t numRows,
      const std::function<vector_size_t(vector_size_t)>& sizeAt,
      const std::function<bool(vector_size_t)>& isNullAt,
      const std::function<VectorPtr(vector_size_t)>& makeKeys,
      const std::function<VectorPtr(vector_size_t)>& makeValues) {
    std::vector<vector_size_t> offsets;
    std::vector<vector_size_t> nulls;
    vector_size_t numElements = 0;
    for (auto i = 0; i < numRows; ++i) {
      offsets.push_back(numElements);
      if (isNullAt(i)) {
        nulls.push_back(i);
      } else {
        numElements += sizeAt(i);
      }
    }
    return makeMapVector(
        offsets, makeKeys(numElements), makeValues(numElements), nulls);
  }

  // Makes strings with Dictionary, Trivial, Constant and RLE encodings, with
  // and without nulls.
  std::shared_ptr<connector::ConnectorSplit> makeStringFile() {
    constexpr int32_t kVectorSize = 2'500;
    int64_t row = 0;
    for (auto i = 0; i < 8; ++i) {
      const int64_t first = row;
      vectors_.push_back(makeRowVector(
          {"dict", "uniq", "nulls", "constant", "runs", "rn"},
          {
              makeStrings(
                  kVectorSize,
                  [&](auto j) {
                    const auto k = (first + j) % 23;
                    return fmt::format("value {}", k * k * k);
                  }),
              makeStrings(
                  kVectorSize, [&](auto j) { return stringValue(first + j); }),
              makeStrings(
                  kVectorSize,
                  [&](auto j) { return stringValue((first + j) * 3); },
                  [&](auto j) { return (first + j) % 5 == 0; }),
              makeStrings(
                  kVectorSize,
                  [&](auto /*j*/) {
                    return fmt::format("constant string of vector {}", i);
                  }),
              makeStrings(
                  kVectorSize,
                  [&](auto j) {
                    return fmt::format("run {}", (first + j) / 50);
                  }),
              makeFlatVector<int64_t>(
                  kVectorSize, [&](auto j) { return first + j; }),
          }));
      row += kVectorSize;
    }
    createDuckDbTable(vectors_);
    return writeFile(
        vectors_, std::vector<std::optional<EncodingLayout>>(6), 3);
  }

  // Makes arrays, arrays of arrays, arrays of structs and maps with nulls at
  // all levels and empty arrays.
  std::shared_ptr<connector::ConnectorSplit> makeArrayFile() {
    constexpr int32_t kVectorSize = 2'500;
    int64_t row = 0;
    for (auto i = 0; i < 8; ++i) {
      const int64_t first = row;
      auto bigints = [&](int64_t scale, int32_t nullEvery) {
        return [this, scale, nullEvery, first](vector_size_t n) {
          return makeFlatVector<int64_t>(
              n,
              [&](auto k) { return (first + k) * scale; },
              [&](auto k) { return (first + k) % nullEvery == 0; });
        };
      };
      // array(array(integer)): null outer rows, null and empty inner arrays
      // and null elements.
      auto arrayOfArrays = makeArrays(
          kVectorSize,
          [&](auto j) { return (first + j) % 4; },
          [&](auto j) { return (first + j) % 9 == 0; },
          [&](vector_size_t n) {
            return makeArrays(
                n,
                [&](auto k) { return k % 3; },
                [&](auto k) { return k % 5 == 0; },
                [&](vector_size_t m) {
                  return makeFlatVector<int32_t>(
                      m,
                      [&](auto e) { return e * 7 + i; },
                      [&](auto e) { return e % 13 == 0; });
                });
          });
      // array(array(array(bigint))) with nulls at each level.
      auto threeLevels = makeArrays(
          kVectorSize,
          [&](auto j) { return (first + j) % 3; },
          [&](auto j) { return (first + j) % 10 == 0; },
          [&](vector_size_t n) {
            return makeArrays(
                n,
                [&](auto k) { return (k + 1) % 3; },
                [&](auto k) { return k % 7 == 3; },
                [&](vector_size_t m) {
                  return makeArrays(
                      m,
                      [&](auto e) { return e % 4; },
                      [&](auto e) { return e % 6 == 5; },
                      bigints(11, 17));
                });
          });
      // array(row(x bigint, y array(varchar), z varchar)) with null structs.
      auto arrayOfStructs = makeArrays(
          kVectorSize,
          [&](auto j) { return (first + j) % 5; },
          [&](auto j) { return (first + j) % 8 == 0; },
          [&](vector_size_t n) {
            return makeRowVector(
                {"x", "y", "z"},
                {bigints(3, 7)(n),
                 makeArrays(
                     n,
                     [&](auto k) { return k % 3; },
                     [&](auto k) { return k % 4 == 1; },
                     [&](vector_size_t m) {
                       return makeStrings(
                           m,
                           [&](auto e) { return stringValue(e + first); },
                           [&](auto e) { return e % 9 == 0; });
                     }),
                 makeStrings(
                     n,
                     [&](auto k) { return stringValue(k * 5); },
                     [&](auto k) { return k % 11 == 0; })},
                [&](auto k) { return k % 6 == 2; });
          });
      vectors_.push_back(makeRowVector(
          {"a", "aa", "a3", "as", "m", "ms", "rn"},
          {
              makeArrays(
                  kVectorSize,
                  [&](auto j) { return (first + j) % 5; },
                  [&](auto j) { return (first + j) % 7 == 0; },
                  bigints(5, 11)),
              arrayOfArrays,
              threeLevels,
              arrayOfStructs,
              makeMaps(
                  kVectorSize,
                  [&](auto j) { return (first + j) % 4; },
                  [&](auto j) { return (first + j) % 6 == 0; },
                  [&](vector_size_t n) {
                    return makeFlatVector<int32_t>(
                        n, [&](auto k) { return k; });
                  },
                  bigints(13, 9)),
              makeMaps(
                  kVectorSize,
                  [&](auto j) { return (first + j) % 3; },
                  [&](auto j) { return (first + j) % 10 == 3; },
                  [&](vector_size_t n) {
                    return makeStrings(
                        n, [&](auto k) { return fmt::format("key {}", k); });
                  },
                  [&](vector_size_t n) {
                    return makeArrays(
                        n,
                        [&](auto k) { return k % 4; },
                        [&](auto k) { return k % 5 == 4; },
                        bigints(17, 23));
                  }),
              makeFlatVector<int64_t>(
                  kVectorSize, [&](auto j) { return first + j; }),
          }));
      row += kVectorSize;
    }
    std::vector<EncodingLayoutTree> columns;
    for (const auto& child : vectors_[0]->type()->asRow().children()) {
      columns.push_back(openLayout(child));
    }
    return writeFile(vectors_, std::move(columns), 3);
  }

  // Returns the rows of 'vectors_' in one vector.
  // Rows of stripe 0 in makeMissingFile(). The file has 2 stripes.
  static constexpr int32_t kStripe0Rows = 7'500;

  // Makes columns that have no stream in one of the 2 stripes, i.e. no
  // non-null value: sparse BIGINT (null in stripe 0), sstr VARCHAR (null in
  // stripe 1), arr ARRAY(BIGINT) (only null elements in stripe 0, null in
  // stripe 1), st ROW(x BIGINT, y VARCHAR) (null in stripe 0, y null in
  // stripe 1), and m MAP(INTEGER, BIGINT) written as a flat map, with key 1 in
  // both stripes, key 3 in stripe 0 and key 2 in stripe 1.
  std::shared_ptr<connector::ConnectorSplit> makeMissingFile() {
    constexpr int32_t kVectorSize = 2'500;
    int64_t row = 0;
    for (auto i = 0; i < 6; ++i) {
      const int64_t first = row;
      const bool isStripe0 = first < kStripe0Rows;
      auto x = makeFlatVector<int64_t>(
          kVectorSize, [&](auto j) { return first + j; });
      auto y = makeStrings(
          kVectorSize,
          [&](auto j) { return stringValue(first + j); },
          [&](auto /*j*/) { return true; });
      auto st = makeRowVector(
          {"x", "y"}, {x, y}, [&](auto /*j*/) { return isStripe0; });
      std::vector<vector_size_t> offsets;
      std::vector<int32_t> keys;
      std::vector<int64_t> values;
      for (auto j = 0; j < kVectorSize; ++j) {
        offsets.push_back(static_cast<int32_t>(keys.size()));
        for (int32_t key = 1; key <= 3; ++key) {
          if (flatMapValue(first + j, key).has_value()) {
            keys.push_back(key);
            values.push_back(*flatMapValue(first + j, key));
          }
        }
      }
      auto map = makeMapVector(
          offsets,
          makeFlatVector<int32_t>(keys),
          makeFlatVector<int64_t>(values),
          {});
      for (auto j = 0; j < kVectorSize; ++j) {
        if ((first + j) % 7 == 3) {
          map->setNull(j, true);
        }
      }
      vectors_.push_back(makeRowVector(
          {"sparse", "sstr", "arr", "st", "m", "rn"},
          {makeFlatVector<int64_t>(
               kVectorSize,
               [&](auto j) { return (first + j) * 3; },
               [&](auto /*j*/) { return isStripe0; }),
           makeStrings(
               kVectorSize,
               [&](auto j) { return stringValue(first + j); },
               [&](auto /*j*/) { return !isStripe0; }),
           makeArrays(
               kVectorSize,
               [&](auto j) { return (first + j) % 3; },
               [&](auto /*j*/) { return !isStripe0; },
               [&](auto n) {
                 return makeFlatVector<int64_t>(
                     n,
                     [](auto k) { return k; },
                     [](auto /*k*/) { return true; });
               }),
           st,
           map,
           makeFlatVector<int64_t>(
               kVectorSize, [&](auto j) { return first + j; })}));
      row += kVectorSize;
    }
    std::vector<EncodingLayoutTree> columns;
    for (const auto& child : vectors_[0]->type()->asRow().children()) {
      columns.push_back(
          child->isMap() ? EncodingLayoutTree{Kind::FlatMap, {}, ""}
                         : openLayout(child));
    }
    return writeFile(vectors_, std::move(columns), 3, {"m"});
  }

  // Value of key 'key' of the map of 'row' in makeMissingFile(), nullopt if
  // the map has no 'key'.
  static std::optional<int64_t> flatMapValue(int64_t row, int32_t key) {
    if (row % 7 == 3 || key > 3) {
      return std::nullopt;
    }
    const bool isStripe0 = row < kStripe0Rows;
    if ((key == 2 && isStripe0) || (key == 3 && (!isStripe0 || row % 2))) {
      return std::nullopt;
    }
    return row * 10 + key;
  }

  // Returns the m."key" of all rows of makeMissingFile().
  VectorPtr flatMapField(int32_t key) {
    std::vector<std::optional<int64_t>> result;
    result.reserve(2 * kStripe0Rows);
    for (int64_t row = 0; row < 2 * kStripe0Rows; ++row) {
      result.push_back(flatMapValue(row, key));
    }
    return makeNullableFlatVector<int64_t>(result);
  }

  RowVectorPtr allRows() {
    vector_size_t numRows = 0;
    for (const auto& vector : vectors_) {
      numRows += vector->size();
    }
    auto result = std::static_pointer_cast<RowVector>(
        BaseVector::create(vectors_[0]->type(), numRows, pool_.get()));
    vector_size_t offset = 0;
    for (const auto& vector : vectors_) {
      result->copy(vector.get(), offset, 0, vector->size());
      offset += vector->size();
    }
    return result;
  }

  // Compression of the leaves of the written files.
  CompressionType compression_{CompressionType::Uncompressed};
  std::vector<RowVectorPtr> vectors_;
  std::vector<std::shared_ptr<common::testutil::TempFilePath>> files_;
};

TEST_F(NimbleScanTest, streamLayout) {
  makeScalarFile();
  const std::vector<std::string> expected = {
      "NFbw<uint64_t>",
      "NFbw<uint64_t>",
      "NFbw<uint32_t>",
      "NFbw<uint32_t>",
      "NFbw<uint64_t>",
      "NFbw<uint32_t>",
      "NDict<NFbw<uint32_t>, NFbw<uint64_t>>",
      "NConstant",
      "NFbw<uint64_t>",
  };
  auto stripes = parseStripes();
  EXPECT_GT(stripes.size(), 1);
  int32_t numRows = 0;
  for (const auto& columns : stripes) {
    std::vector<std::string> accessors;
    for (const auto& column : columns) {
      EXPECT_GT(column.chunks.size(), 1);
      EXPECT_EQ(column.numRows, columns[0].numRows);
      accessors.push_back(column.accessorList());
    }
    EXPECT_EQ(accessors, expected);
    numRows += columns[0].numRows;
  }
  EXPECT_EQ(numRows, 8 * 2'500);
}

TEST_F(NimbleScanTest, scalarEncodings) {
  auto split = makeScalarFile();
  auto type = asRowType(vectors_[0]->type());
  auto plan = PlanBuilder(pool_.get()).tableScan(type).planNode();
  assertQuery(plan, split, "SELECT * FROM tmp");
}

TEST_F(NimbleScanTest, filters) {
  auto split = makeScalarFile();
  auto type = asRowType(vectors_[0]->type());
  auto plan = PlanBuilder(pool_.get())
                  .tableScan(type, {"fbw64 < 3500", "dict > 3000000000"})
                  .planNode();
  assertQuery(
      plan,
      split,
      "SELECT * FROM tmp WHERE fbw64 < 3500 AND dict > 3000000000");
}

TEST_F(NimbleScanTest, filterOnlyColumn) {
  auto split = makeScalarFile();
  auto type = asRowType(vectors_[0]->type());
  auto plan = PlanBuilder(pool_.get())
                  .tableScan(
                      ROW(
                          {{"trivial32", INTEGER()},
                           {"dbl", DOUBLE()},
                           {"rn", BIGINT()}}),
                      {"constant >= 80", "fbw32 < 500"},
                      "",
                      type)
                  .planNode();
  assertQuery(
      plan,
      split,
      "SELECT trivial32, dbl, rn FROM tmp WHERE constant >= 80 AND fbw32 < 500");
}

TEST_F(NimbleScanTest, nullsAndRunsLayout) {
  makeNullsAndRunsFile();
  const std::vector<std::string> expected = {
      "NNullable<NFbw<uint64_t>>",
      "NNullable<NFbw<uint32_t>>",
      "NNullable<NDict<NFbw<uint32_t>, NFbw<uint64_t>>>",
      "NMainlyConstant<NFbw<uint64_t>>",
      "NRle<NFbw<uint64_t>>",
      "NFbw<uint64_t>",
  };
  std::set<int32_t> nullsEncodings;
  for (const auto& columns : parseStripes()) {
    std::vector<std::string> accessors;
    for (const auto& column : columns) {
      accessors.push_back(column.accessorList());
      for (const auto& node : column.nodes) {
        if (node.dataType == static_cast<uint8_t>(nimble::DataType::Bool)) {
          nullsEncodings.insert(node.encoding);
        }
      }
    }
    EXPECT_EQ(accessors, expected);
  }
  // Trivial, SparseBool and RLE.
  EXPECT_EQ(nullsEncodings, std::set<int32_t>({0, 1, 6}));
}

TEST_F(NimbleScanTest, nullsAndRuns) {
  auto split = makeNullsAndRunsFile();
  auto type = asRowType(vectors_[0]->type());
  auto plan = PlanBuilder(pool_.get()).tableScan(type).planNode();
  assertQuery(plan, split, "SELECT * FROM tmp");
}

TEST_F(NimbleScanTest, nullsAndRunsFilters) {
  auto split = makeNullsAndRunsFile();
  auto type = asRowType(vectors_[0]->type());
  {
    auto plan = PlanBuilder(pool_.get())
                    .tableScan(type, {"nulls7 > 20000", "runs < 1000"})
                    .planNode();
    assertQuery(
        plan, split, "SELECT * FROM tmp WHERE nulls7 > 20000 AND runs < 1000");
  }
  {
    auto plan =
        PlanBuilder(pool_.get())
            .tableScan(
                type,
                {"mostly5 = 5", "null_runs between 2000000000 and 6000000000"})
            .planNode();
    assertQuery(
        plan,
        split,
        "SELECT * FROM tmp WHERE mostly5 = 5 AND null_runs BETWEEN 2000000000 AND 6000000000");
  }
  {
    auto plan = PlanBuilder(pool_.get())
                    .tableScan(type, {"mostly5 > 5", "nulls300 < 500"})
                    .planNode();
    assertQuery(
        plan, split, "SELECT * FROM tmp WHERE mostly5 > 5 AND nulls300 < 500");
  }
}

TEST_F(NimbleScanTest, mixedEncodings) {
  auto split = makeMixedFile();
  for (const auto& columns : parseStripes()) {
    EXPECT_GE(columns[0].accessors.size(), 3) << columns[0].accessorList();
  }
  auto type = asRowType(vectors_[0]->type());
  assertQuery(
      PlanBuilder(pool_.get()).tableScan(type).planNode(),
      split,
      "SELECT * FROM tmp");
  assertQuery(
      PlanBuilder(pool_.get())
          .tableScan(type, {"mixed between 40 and 2000000000"})
          .planNode(),
      split,
      "SELECT * FROM tmp WHERE mixed BETWEEN 40 AND 2000000000");
}

// Scans files whose leaves are compressed. The payloads are decompressed on
// the host.
TEST_F(NimbleScanTest, compressed) {
  for (auto compression :
       {CompressionType::Zstd,
        CompressionType::MetaInternal,
        CompressionType::OpenZL}) {
    SCOPED_TRACE(nimble::toString(compression));
    vectors_.clear();
    compression_ = compression;
    auto split = makeNullsAndRunsFile();
    int64_t decompressedBytes = 0;
    for (const auto& columns : parseStripes()) {
      for (const auto& column : columns) {
        decompressedBytes += static_cast<int64_t>(column.decompressed.size());
      }
    }
    EXPECT_GT(decompressedBytes, 0);
    auto type = asRowType(vectors_[0]->type());
    assertQuery(
        PlanBuilder(pool_.get()).tableScan(type).planNode(),
        split,
        "SELECT * FROM tmp");
    assertQuery(
        PlanBuilder(pool_.get())
            .tableScan(type, {"nulls7 > 20000", "mostly5 = 5"})
            .planNode(),
        split,
        "SELECT * FROM tmp WHERE nulls7 > 20000 AND mostly5 = 5");
  }
}

TEST_F(NimbleScanTest, nestedStructs) {
  auto split = makeStructFile();
  auto type = asRowType(vectors_[0]->type());
  assertQuery(
      PlanBuilder(pool_.get())
          .tableScan(type)
          .project({"s.a", "s.b", "(s.t).c", "rn"})
          .planNode(),
      split,
      "SELECT s.a, s.b, s.t.c, rn FROM tmp");
  assertQuery(
      PlanBuilder(pool_.get())
          .tableScan(type, {"s.a > 50000", "(s.t).c < 150000"})
          .project({"s.a", "s.b", "(s.t).c", "rn"})
          .planNode(),
      split,
      "SELECT s.a, s.b, s.t.c, rn FROM tmp WHERE s.a > 50000 AND s.t.c < 150000");
}

TEST_F(NimbleScanTest, flatMapKeys) {
  auto split = makeFlatMapFile();
  auto type = asRowType(vectors_[0]->type());
  std::vector<VectorPtr> rn;
  rn.reserve(vectors_.size());
  for (const auto& vector : vectors_) {
    rn.push_back(vector->childAt(1));
  }
  auto expected = makeRowVector(
      {expectedElements(1),
       expectedElements(3),
       expectedElements(5),
       makeFlatVector<int64_t>(
           static_cast<vector_size_t>(vectors_.size() * vectors_[0]->size()),
           [](auto row) { return row; })});
  auto plan = PlanBuilder(pool_.get())
                  .tableScan(type)
                  .project(
                      {"element_at(m, 1::INTEGER)",
                       "element_at(m, 3::INTEGER)",
                       "element_at(m, 5::INTEGER)",
                       "rn"})
                  .planNode();
  AssertQueryBuilder(plan).split(split).assertResults(expected);
}

TEST_F(NimbleScanTest, strings) {
  auto split = makeStringFile();
  std::string accessors;
  for (const auto& columns : parseStripes()) {
    for (const auto& column : columns) {
      accessors += column.accessorList() + ";";
    }
  }
  EXPECT_NE(accessors.find("NString"), std::string::npos) << accessors;
  EXPECT_NE(accessors.find("NDict<"), std::string::npos) << accessors;
  auto type = asRowType(vectors_[0]->type());
  assertQuery(
      PlanBuilder(pool_.get()).tableScan(type).planNode(),
      split,
      "SELECT * FROM tmp");
  assertQuery(
      PlanBuilder(pool_.get())
          .tableScan(type, {"rn between 1000 and 9000"})
          .planNode(),
      split,
      "SELECT * FROM tmp WHERE rn BETWEEN 1000 AND 9000");
}

TEST_F(NimbleScanTest, arraysAndMaps) {
  auto split = makeArrayFile();
  auto type = asRowType(vectors_[0]->type());
  auto expected = allRows();
  AssertQueryBuilder(PlanBuilder(pool_.get()).tableScan(type).planNode())
      .split(split)
      .assertResults(expected);
  // A filter on a top level column selects the arrays of the passing rows.
  AssertQueryBuilder(PlanBuilder(pool_.get())
                         .tableScan(type, {"rn between 3001 and 12000"})
                         .planNode())
      .split(split)
      .assertResults(
          std::static_pointer_cast<RowVector>(expected->slice(3001, 9000)));
}

// Reads columns that are not in the file and columns without a stream in a
// stripe. These are null.
TEST_F(NimbleScanTest, missingColumns) {
  auto split = makeMissingFile();
  auto fileType = asRowType(vectors_[0]->type());
  {
    // The streams without a non-null value are not in the file.
    auto readerBase = openReader();
    const auto& row = readerBase->nimbleSchema()->asRow();
    auto scalarId = [](const nimble::Type& type) {
      return type.asScalar().scalarDescriptor().offset();
    };
    const auto& arr = row.childAt(2)->asArray();
    const auto& st = row.childAt(3)->asRow();
    const auto& m = row.childAt(4)->asFlatMap();
    auto key = m.findChild("2");
    ASSERT_TRUE(key.has_value());
    EXPECT_FALSE(hasStream(0, scalarId(*row.childAt(0))));
    EXPECT_TRUE(hasStream(1, scalarId(*row.childAt(0))));
    EXPECT_FALSE(hasStream(1, scalarId(*row.childAt(1))));
    EXPECT_FALSE(hasStream(0, scalarId(*arr.elements())));
    EXPECT_FALSE(hasStream(1, arr.lengthsDescriptor().offset()));
    EXPECT_FALSE(hasStream(0, scalarId(*st.childAt(0))));
    EXPECT_FALSE(hasStream(1, st.nullsDescriptor().offset()));
    EXPECT_FALSE(hasStream(0, scalarId(*m.childAt(key.value()))));
    EXPECT_FALSE(hasStream(0, m.inMapDescriptorAt(key.value()).offset()));
  }
  std::vector<std::string> names = fileType->names();
  std::vector<TypePtr> types = fileType->children();
  const std::vector<std::string> missingNames = {
      "extra", "extras", "ms", "ma", "mm"};
  const std::vector<TypePtr> missingTypes = {
      BIGINT(),
      VARCHAR(),
      ROW({{"a", BIGINT()}, {"b", VARCHAR()}}),
      ARRAY(BIGINT()),
      MAP(INTEGER(), BIGINT())};
  names.insert(names.end(), missingNames.begin(), missingNames.end());
  types.insert(types.end(), missingTypes.begin(), missingTypes.end());
  auto type = ROW(std::move(names), std::move(types));

  const vector_size_t numRows = 2 * kStripe0Rows;
  auto all = allRows();
  auto nulls = [&](const TypePtr& type) {
    return BaseVector::createNullConstant(type, numRows, pool_.get());
  };
  auto st = all->childAt(3)->as<RowVector>();
  auto stField = [&](int32_t idx) {
    // The fields of the null structs are null.
    auto field = BaseVector::copy(*st->childAt(idx));
    for (auto row = 0; row < numRows; ++row) {
      if (st->isNullAt(row)) {
        field->setNull(row, true);
      }
    }
    return field;
  };
  const std::vector<std::string> projections = {
      "rn",
      "sparse",
      "sstr",
      "arr",
      "st.x",
      "st.y",
      "extra",
      "extras",
      "ms.a",
      "ms.b",
      "ma",
      "mm"};
  auto expected = makeRowVector(
      {all->childAt(5),
       all->childAt(0),
       all->childAt(1),
       all->childAt(2),
       stField(0),
       stField(1),
       nulls(BIGINT()),
       nulls(VARCHAR()),
       nulls(BIGINT()),
       nulls(VARCHAR()),
       nulls(ARRAY(BIGINT())),
       nulls(MAP(INTEGER(), BIGINT()))});
  auto scan = [&](const std::vector<std::string>& filters) {
    return PlanBuilder(pool_.get())
        .tableScan(type, filters, "", type)
        .project(projections)
        .planNode();
  };
  AssertQueryBuilder(scan({})).split(split).assertResults(expected);
  // Filters on columns that are null in a stripe or not in the file.
  AssertQueryBuilder(scan({"sparse > 30000"}))
      .split(split)
      .assertResults(
          std::static_pointer_cast<RowVector>(
              expected->slice(10'001, numRows - 10'001)));
  AssertQueryBuilder(scan({"rn < 5000", "st.x > 10"}))
      .split(split)
      .assertEmptyResults();
  AssertQueryBuilder(scan({"extra > 5"})).split(split).assertEmptyResults();
}

// Reads a flat map as a struct of some of its keys. Key 9 is not in the file
// and keys 2 and 3 have no streams in one of the stripes.
TEST_F(NimbleScanTest, flatMapAsStruct) {
  auto split = makeMissingFile();
  auto fileType = asRowType(vectors_[0]->type());
  auto structType = ROW({"1", "2", "3", "9"}, BIGINT());
  auto outputType = ROW({{"m", structType}, {"rn", BIGINT()}});
  auto all = allRows();
  auto plan = PlanBuilder(pool_.get())
                  .tableScan(outputType, {}, "", fileType)
                  .project({"m.\"3\"", "m.\"1\"", "m.\"9\"", "m.\"2\"", "rn"})
                  .planNode();
  AssertQueryBuilder(plan).split(split).assertResults(makeRowVector(
      {flatMapField(3),
       flatMapField(1),
       flatMapField(9),
       flatMapField(2),
       all->childAt(5)}));

  // Only the streams of the keys that are fields of the struct are read.
  auto readerBase = openReader();
  const auto& m = *readerBase->nimbleSchema()->asRow().childAt(4);
  common::ScanSpec spec("m");
  spec.setFlatMapAsStruct(true);
  spec.addField("1", 0);
  spec.addField("3", 1);
  std::vector<uint32_t> streamIds;
  wave::NimbleFormatReader::addStreamIds(m, &spec, streamIds);
  const auto& flatMap = m.asFlatMap();
  std::vector<uint32_t> expectedIds = {flatMap.nullsDescriptor().offset()};
  for (auto i = 0; i < flatMap.childrenCount(); ++i) {
    if (flatMap.nameAt(i) == "1" || flatMap.nameAt(i) == "3") {
      expectedIds.push_back(flatMap.inMapDescriptorAt(i).offset());
      expectedIds.push_back(
          flatMap.childAt(i)->asScalar().scalarDescriptor().offset());
    }
  }
  EXPECT_EQ(streamIds, expectedIds);
}

} // namespace
