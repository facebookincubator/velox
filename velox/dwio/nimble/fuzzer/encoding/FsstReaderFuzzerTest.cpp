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

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <folly/Random.h>
#include <algorithm>
#include <limits>
#include <optional>
#include <random>
#include <string_view>

#include "velox/common/file/File.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/nimble/common/ChunkedStream.h"
#include "velox/dwio/nimble/common/SchemaSerialization.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/compression/Compression.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/fuzzer/encoding/EncodingFuzzer.h"
#include "velox/dwio/nimble/selective/SelectiveNimbleReader.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/tests/TabletTestUtils.h"
#include "velox/dwio/nimble/writer/WriterOptions.h"
#include "velox/type/Filter.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/tests/utils/VectorMaker.h"

DEFINE_uint32(fsst_reader_iterations, 24, "FSST reader fuzzer iterations");
DEFINE_uint32(fsst_reader_max_rows, 512, "Maximum rows per FSST reader chunk");
DEFINE_uint32(fsst_reader_seed, 42, "FSST reader fuzzer seed (0 = random)");

namespace facebook::nimble::test {
namespace {

using namespace facebook::velox;

std::string dataFromSource(
    const std::vector<std::optional<std::string>>& source,
    std::mt19937& rng) {
  auto value = source[rng() % source.size()].value_or("");
  if (!value.empty() && rng() % 2) {
    value.resize(1 + rng() % value.size());
  }
  return value;
}

// Describes a byte range and evaluates original rows independently of the
// reader.
struct Predicate {
  // Lower byte bound of the range before negation.
  std::string lower;
  // Upper byte bound of the range before negation.
  std::string upper;
  // Ignores the lower bound when true.
  bool lowerUnbounded{false};
  // Ignores the upper bound when true.
  bool upperUnbounded{false};
  // Excludes values equal to the lower bound from the non-negated range.
  bool lowerExclusive{false};
  // Excludes values equal to the upper bound from the non-negated range.
  bool upperExclusive{false};
  // Selects non-null values outside the range when true.
  bool negated{false};
  // Controls null matches independently of range negation.
  bool nullAllowed{false};

  std::unique_ptr<common::Filter> filter() const {
    if (negated) {
      return std::make_unique<common::NegatedBytesRange>(
          lower,
          lowerUnbounded,
          lowerExclusive,
          upper,
          upperUnbounded,
          upperExclusive,
          nullAllowed);
    }
    return std::make_unique<common::BytesRange>(
        lower,
        lowerUnbounded,
        lowerExclusive,
        upper,
        upperUnbounded,
        upperExclusive,
        nullAllowed);
  }

  bool matches(const std::optional<std::string>& value) const {
    if (!value) {
      return nullAllowed;
    }
    const auto lowerCompare = value->compare(lower);
    const auto upperCompare = value->compare(upper);
    const bool inRange = (lowerUnbounded || lowerCompare > 0 ||
                          (lowerCompare == 0 && !lowerExclusive)) &&
        (upperUnbounded || upperCompare < 0 ||
         (upperCompare == 0 && !upperExclusive));
    return negated ? !inRange : inRange;
  }
};

// Verifies filtered FSST and dictionary reads against the original input rows.
class FsstReaderFuzzerTest : public ::testing::TestWithParam<bool> {
 protected:
  static void SetUpTestSuite() {
    if (!memory::MemoryManager::testInstance()) {
      memory::MemoryManager::initialize({});
    }
    registerSelectiveNimbleReaderFactory();
  }

  static void TearDownTestSuite() {
    unregisterSelectiveNimbleReaderFactory();
  }

  EncodingLayoutTree layout(bool ordinary, CompressionType compressionType) {
    const EncodingLayout trivial{
        EncodingType::Trivial, {}, CompressionType::Uncompressed};
    EncodingLayout strings{
        ordinary ? EncodingType::Trivial : EncodingType::Fsst,
        {},
        compressionType,
        {trivial}};
    const auto valueLayout = [&] {
      if (!ordinary && GetParam()) {
        return EncodingLayout{
            EncodingType::Dictionary,
            {},
            CompressionType::Uncompressed,
            {strings, trivial}};
      }
      return std::move(strings);
    }();
    return EncodingLayoutTree{
        Kind::Row,
        {},
        "",
        {EncodingLayoutTree{Kind::Scalar, {}, "id"},
         EncodingLayoutTree{
             Kind::Scalar,
             {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
               valueLayout}},
             "value"},
         EncodingLayoutTree{Kind::Scalar, {}, "gate"}}};
  }

  std::string write(
      const std::vector<VectorPtr>& batches,
      bool ordinary,
      CompressionType compressionType) {
    WriterOptions options{};
    options.fsstCompressionTargetRatio = std::numeric_limits<double>::max();
    options.encodingLayoutTree.emplace(layout(ordinary, compressionType));
    options.compressionOptions.compressionAcceptRatio = 1.0;
    options.compressionOptions.zstdMinCompressionSize = 0;
    options.compressionOptions.lz4MinCompressionSize = 0;
    options.enableChunking = true;
    options.minStreamChunkRawSize = 0;
    options.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          [](const StripeProgress&) { return false; },
          [](const StripeProgress&) { return true; });
    };
    return createNimbleFile(*rootPool_, batches, std::move(options), false);
  }

  void collectSymbolBounds(std::string_view encoded) {
    const auto header = FsstEncoding::parseHeader(
        encoded, EncodingPrefix::prefixSize(encoded, false));
    fsst_decoder_t decoder{};
    ASSERT_GT(
        nimble_fsst_import(
            &decoder,
            reinterpret_cast<unsigned char*>(
                const_cast<char*>(header.symbolTable.data()))),
        0);
    auto lengths = EncodingFactory().create(*pool_, header.lengths, nullptr);
    std::vector<uint32_t> sizes(lengths->rowCount());
    lengths->materialize(lengths->rowCount(), sizes.data());
    BufferPtr uncompressedBlob;
    auto blob = header.blob;
    if (header.compressionType != CompressionType::Uncompressed) {
      uncompressedBlob = Compression::uncompress(
          *pool_, header.compressionType, DataType::String, blob, nullptr);
      blob = {uncompressedBlob->as<char>(), uncompressedBlob->size()};
    }
    size_t offset{0};
    for (const auto size : sizes) {
      std::string prefix;
      const auto row = blob.substr(offset, size);
      offset += size;
      for (size_t i = 0; i < row.size(); ++i) {
        const auto code = static_cast<uint8_t>(row[i]);
        if (code == 255) {
          ++i;
          ASSERT_LT(i, row.size()) << "Incomplete FSST escape code";
          prefix.push_back(row[i]);
        } else {
          const auto length = decoder.len[code];
          const auto* symbol =
              reinterpret_cast<const char*>(&decoder.symbol[code]);
          if (length > 1) {
            symbolBounds_.push_back(prefix + std::string(symbol, length - 1));
            return;
          }
          prefix.append(symbol, length);
        }
      }
    }
  }

  // Verifies at least three FSST chunks, optional dictionary wrapping, and the
  // requested secondary compression, then collects bounds inside FSST symbols.
  void verifyLayout(std::string_view file, CompressionType compressionType) {
    symbolBounds_.clear();
    auto tablet = TabletReader::create(
        std::make_shared<InMemoryReadFile>(file),
        pool_.get(),
        makeTestTabletOptions(pool_.get()));
    auto section = tablet->loadOptionalSection(std::string(kSchemaSection));
    ASSERT_TRUE(section.has_value());
    auto schema = SchemaDeserializer::deserialize(section->content().data());
    const auto offset =
        schema->asRow().childAt(1)->asScalar().scalarDescriptor().offset();
    uint32_t chunks{0};
    for (uint32_t stripe = 0; stripe < tablet->stripeCount(); ++stripe) {
      auto streams =
          tablet->load(tablet->stripeIdentifier(stripe), std::span{&offset, 1});
      InMemoryChunkedStream stream{*pool_, std::move(streams[0])};
      while (stream.hasNext()) {
        auto encoded = stream.nextChunk();
        const auto firstChild = [&] {
          const auto* pos =
              encoded.data() + EncodingPrefix::prefixSize(encoded, false);
          const auto size = encoding::readUint32(pos);
          return std::string_view(pos, size);
        };
        if (EncodingPrefix::encodingType(encoded) == EncodingType::Nullable) {
          encoded = firstChild();
        }
        auto capture =
            EncodingLayoutCapture::capture(encoded, Encoding::Options{});
        const auto* value = &capture;
        if (GetParam()) {
          ASSERT_EQ(value->encodingType(), EncodingType::Dictionary);
          ASSERT_TRUE(value->child(0).has_value());
          value = &value->child(0).value();
          encoded = firstChild();
        }
        ASSERT_EQ(value->encodingType(), EncodingType::Fsst);
        ASSERT_EQ(value->compressionType(), compressionType);
        const auto header = FsstEncoding::parseHeader(
            encoded, EncodingPrefix::prefixSize(encoded, false));
        ASSERT_EQ(header.compressionType, compressionType);
        collectSymbolBounds(encoded);
        ASSERT_FALSE(HasFatalFailure());
        ++chunks;
      }
    }
    ASSERT_GE(chunks, 3);
    ASSERT_FALSE(symbolBounds_.empty());
  }

  using Result = std::vector<std::pair<int64_t, std::optional<std::string>>>;

  Result read(
      std::string_view file,
      const RowTypePtr& type,
      const Predicate& predicate,
      bool project,
      bool sparse,
      bool cache,
      bool preserve,
      bool fastPath,
      uint32_t batchSize) {
    auto spec = std::make_shared<dwio::common::ScanSpec>("root");
    spec->addField("id", 0);
    auto* value = spec->getOrCreateChild("value");
    value->setFilter(predicate.filter());
    value->setProjectOut(project);
    if (project) {
      value->setChannel(1);
    }
    if (sparse) {
      spec->getOrCreateChild("gate")->setFilter(
          std::make_unique<common::BigintRange>(1, 1, false));
    }
    dwio::common::ReaderOptions options(pool_.get());
    options.setScanSpec(spec);
    auto reader =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE)
            ->createReader(
                std::make_unique<dwio::common::BufferedInput>(
                    std::make_shared<InMemoryReadFile>(file), *pool_),
                options);
    dwio::common::RowReaderOptions rowOptions;
    rowOptions.setScanSpec(spec);
    rowOptions.setRequestedType(type);
    rowOptions.setStringDecoderZeroCopy(true);
    rowOptions.setNimbleFsstDecodeRepeatedPayloads(cache);
    rowOptions.setNimblePreserveDictionaryEncoding(preserve);
    rowOptions.setNimbleScanFastPath(fastPath);
    auto rows = reader->createRowReader(rowOptions);
    const auto resultType = project
        ? ROW({{"id", BIGINT()}, {"value", VARCHAR()}})
        : ROW("id", BIGINT());
    auto batch = BaseVector::create(resultType, 0, pool_.get());
    Result result;
    while (rows->next(batchSize, batch) > 0) {
      auto* row = batch->as<RowVector>();
      DecodedVector ids(*row->childAt(0)->loadedVector());
      DecodedVector strings;
      if (project) {
        strings.decode(*row->childAt(1)->loadedVector());
      }
      for (vector_size_t i = 0; i < row->size(); ++i) {
        std::optional<std::string> text;
        if (project && !strings.isNullAt(i)) {
          text = strings.valueAt<StringView>(i).str();
        }
        result.emplace_back(ids.valueAt<int64_t>(i), std::move(text));
      }
    }
    return result;
  }

  void runFilteredReads(CompressionType compressionType);

  // Owns memory used to write and read the test files.
  std::shared_ptr<memory::MemoryPool> rootPool_ =
      memory::memoryManager()->addRootPool("fsst_reader_fuzzer");
  // Allocates reader vectors and temporary decoding buffers.
  std::shared_ptr<memory::MemoryPool> pool_ = rootPool_->addLeafChild("reader");
  // Stores decoded prefixes ending inside multi-byte FSST symbols for filters.
  std::vector<std::string> symbolBounds_;
};

void FsstReaderFuzzerTest::runFilteredReads(CompressionType compressionType) {
  ASSERT_GE(FLAGS_fsst_reader_max_rows, 32);
  const auto seed = FLAGS_fsst_reader_seed == 0 ? folly::Random::rand32()
                                                : FLAGS_fsst_reader_seed;
  LOG(INFO) << "FSST reader fuzzer seed=" << seed
            << " dictionary=" << GetParam()
            << " compression=" << static_cast<int>(compressionType);
  std::mt19937 rng(seed);
  velox::test::VectorMaker maker{pool_.get()};
  for (uint32_t iteration = 0; iteration < FLAGS_fsst_reader_iterations;
       ++iteration) {
    const bool project = iteration % 2;
    const bool sparse = rng() % 2;
    const bool preserve = (iteration / 2) % 2;
    const bool fastPath = (iteration / 4) % 2;
    SCOPED_TRACE(
        ::testing::Message()
        << "seed=" << seed << " iteration=" << iteration << " dictionary="
        << GetParam() << " project=" << project << " sparse=" << sparse
        << " preserve=" << preserve << " fastPath=" << fastPath);
    Buffer stringsBuffer(*pool_);
    std::vector<VectorPtr> batches;
    std::vector<std::optional<std::string>> source;
    const auto chunkRows =
        std::max<uint32_t>(32, 1 + rng() % FLAGS_fsst_reader_max_rows);
    for (uint32_t chunk = 0; chunk < 3; ++chunk) {
      auto data = makeFsstData(
          *pool_, rng, chunkRows, &stringsBuffer, iteration % 3 != 2);
      std::vector<int64_t> ids;
      std::vector<int64_t> gates;
      std::vector<std::optional<std::string>> values;
      for (uint32_t i = 0; i < chunkRows; ++i) {
        ids.push_back(static_cast<int64_t>(source.size()));
        gates.push_back(source.size() % 3 == 0);
        if (compressionType == CompressionType::Uncompressed) {
          values.emplace_back(
              i > 0 && rng() % 7 == 0 ? std::nullopt
                                      : std::optional<std::string>(data[i]));
        } else if (i == 7 || (i > 7 && rng() % 7 == 0)) {
          values.emplace_back(std::nullopt);
        } else {
          std::string text{data[i < 8 ? i : i / 2]};
          if (i >= 8) {
            // Repeated symbol sequences make secondary compression worthwhile
            // even for the smallest dictionary chunk. Pairs also retain hits
            // for the direct FSST decode cache.
            for (uint32_t repeat = 0; repeat < 8; ++repeat) {
              text.append("/secondary/compression/repeated/prefix");
            }
            text.append("/" + std::to_string(i / 2));
          }
          values.emplace_back(std::move(text));
        }
        source.push_back(values.at(i));
      }
      batches.push_back(maker.rowVector(
          {"id", "value", "gate"},
          {maker.flatVector(ids),
           maker.flatVectorNullable(values),
           maker.flatVector(gates)}));
    }
    auto type = asRowType(batches[0]->type());
    const auto fsstFile = write(batches, false, compressionType);
    verifyLayout(fsstFile, compressionType);
    ASSERT_FALSE(HasFatalFailure());
    const auto ordinaryFile =
        write(batches, true, CompressionType::Uncompressed);

    Predicate predicate;
    const auto kind = (iteration / 2) % 6;
    predicate.lower = "common/prefix/alpha";
    predicate.upper = "common/prefix/alphabet";
    if (kind < 2) {
      predicate.upper = predicate.lower;
      predicate.negated = kind == 1;
    } else {
      predicate.negated = kind == 3;
      predicate.lowerExclusive = rng() % 2;
      predicate.upperExclusive = rng() % 2;
      predicate.lowerUnbounded = kind == 4;
      predicate.upperUnbounded = kind == 5;
      if (iteration >= 12) {
        auto lower = iteration % 4 < 2 || kind == 3
            ? symbolBounds_[rng() % symbolBounds_.size()]
            : dataFromSource(source, rng);
        auto upper = dataFromSource(source, rng);
        if (upper < lower) {
          std::swap(lower, upper);
        }
        if (lower == upper) {
          upper.push_back('\xff');
        }
        predicate.lower = std::move(lower);
        predicate.upper = std::move(upper);
      }
    }
    predicate.nullAllowed = rng() % 2;
    Result expected;
    for (size_t row = 0; row < source.size(); ++row) {
      if ((!sparse || row % 3 == 0) && predicate.matches(source[row])) {
        expected.emplace_back(row, project ? source[row] : std::nullopt);
      }
    }
    const uint32_t batchSize = iteration % 3 == 0 ? 1
        : iteration % 3 == 1                      ? 7
                                                  : chunkRows * 2 + 1;
    SCOPED_TRACE(
        ::testing::Message()
        << "lower=" << ::testing::PrintToString(predicate.lower)
        << " upper=" << ::testing::PrintToString(predicate.upper)
        << " lowerUnbounded=" << predicate.lowerUnbounded
        << " upperUnbounded=" << predicate.upperUnbounded
        << " lowerExclusive=" << predicate.lowerExclusive << " upperExclusive="
        << predicate.upperExclusive << " negated=" << predicate.negated
        << " nullAllowed=" << predicate.nullAllowed
        << " batchSize=" << batchSize);
    for (const bool cache : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "cache=" << cache);
      EXPECT_EQ(
          read(
              fsstFile,
              type,
              predicate,
              project,
              sparse,
              cache,
              preserve,
              fastPath,
              batchSize),
          expected);
    }
    EXPECT_EQ(
        read(
            ordinaryFile,
            type,
            predicate,
            project,
            sparse,
            false,
            false,
            false,
            batchSize),
        expected);
  }
}

TEST_P(FsstReaderFuzzerTest, filteredReadsMatchOriginalRows) {
  runFilteredReads(CompressionType::Uncompressed);
}

TEST_P(
    FsstReaderFuzzerTest,
    filteredReadsWithSecondaryCompressionMatchOriginalRows) {
  for (const auto compressionType :
       {CompressionType::Zstd, CompressionType::Lz4}) {
    SCOPED_TRACE(
        ::testing::Message()
        << "compression=" << static_cast<int>(compressionType));
    runFilteredReads(compressionType);
    ASSERT_FALSE(HasFatalFailure());
  }
}

INSTANTIATE_TEST_SUITE_P(FsstLayouts, FsstReaderFuzzerTest, ::testing::Bool());

} // namespace
} // namespace facebook::nimble::test
