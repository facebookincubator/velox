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

// Tests that exercise readWithVisitor code paths by constructing real
// SelectiveColumnReader instances via buildColumnReader and calling read()
// directly.  This bypasses the RowReader::next() pipeline while still using
// the full encoding → ChunkedDecoder → readWithVisitor chain.
//
// Data patterns are chosen to exercise specific encoding types:
//   constant → ConstantEncoding
//   sequential/random → FixedBitWidthEncoding / TrivialEncoding
//   repeated runs → RLEEncoding
//   nullable wrappers → NullableEncoding

#include <gtest/gtest.h>

#include "folly/Random.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/dwio/common/ColumnVisitors.h"
#include "velox/dwio/common/SelectiveColumnReader.h"
#include "velox/dwio/common/SelectiveStructColumnReader.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingUtils.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingUtils.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/tests/EncodingLayoutTestHelper.h"
#include "velox/dwio/nimble/encodings/tests/SharedDictionaryEncodingTestUtils.h"
#include "velox/dwio/nimble/velox/ChunkedStream.h"
#include "velox/dwio/nimble/velox/selective/ByteColumnReader.h"
#include "velox/dwio/nimble/velox/selective/ColumnReader.h"
#include "velox/dwio/nimble/velox/selective/FloatingPointColumnReader.h"
#include "velox/dwio/nimble/velox/selective/IntegerColumnReader.h"
#include "velox/dwio/nimble/velox/selective/NimbleData.h"
#include "velox/dwio/nimble/velox/selective/ReaderBase.h"
#include "velox/dwio/nimble/velox/selective/RowSizeTracker.h"
#include "velox/dwio/nimble/velox/selective/StringColumnReader.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <random>

namespace facebook::nimble {
namespace {

using namespace facebook::velox;

template <typename V>
ReadWithVisitorParams makeReadWithVisitorParams(
    V& visitor,
    const RowSet& rows,
    velox::memory::MemoryPool* memPool);

// ---------------------------------------------------------------------------
// Test-only accessor: exposes protected prepareRead and readOffset_ so that
// tests can call readWithVisitor explicitly with a hand-built ColumnVisitor.
// No new data members — static_cast from the real IntegerColumnReader* is
// layout-safe.
// ---------------------------------------------------------------------------
class IntegerColumnReaderTestAccessor : public IntegerColumnReader {
 public:
  using IntegerColumnReader::IntegerColumnReader;

  template <typename T>
  void
  doPrepareRead(int64_t offset, const RowSet& rows, const uint64_t* nulls) {
    this->template prepareRead<T>(offset, rows, nulls);
  }

  const int32_t* rawIndices() const {
    return reinterpret_cast<const int32_t*>(rawValues_);
  }

  void advanceReadOffset(const RowSet& rows) {
    readOffset_ += rows.back() + 1;
  }
};

// ---------------------------------------------------------------------------
// Test-only accessor for FloatingPointColumnReader: exposes protected
// prepareRead so tests can call readWithVisitor explicitly with a hand-built
// ColumnVisitor.
// No new data members — static_cast from the real FloatingPointColumnReader*
// is layout-safe.
// ---------------------------------------------------------------------------
template <typename T>
class FloatingPointColumnReaderTestAccessor
    : public FloatingPointColumnReader<T, T> {
 public:
  using FloatingPointColumnReader<T, T>::FloatingPointColumnReader;

  void
  doPrepareRead(int64_t offset, const RowSet& rows, const uint64_t* nulls) {
    this->template prepareRead<T>(offset, rows, nulls);
  }
};

// ---------------------------------------------------------------------------
// Test-only accessor for StringColumnReader: exposes prepareRead<int32_t>
// and rawValues_ so tests can call readIndicesWithVisitor and verify the
// indices stored in the reader's buffer.
// No new data members — static_cast from StringColumnReader* is layout-safe.
// ---------------------------------------------------------------------------
class StringColumnReaderTestAccessor : public StringColumnReader {
 public:
  using StringColumnReader::StringColumnReader;

  void doPrepareReadAsIndices(
      int64_t offset,
      const RowSet& rows,
      const uint64_t* nulls) {
    this->template prepareRead<int32_t>(offset, rows, nulls);
  }

  ChunkedDecoder& chunkedDecoder() {
    return decoder_;
  }

  const int32_t* rawIndices() const {
    return reinterpret_cast<const int32_t*>(rawValues_);
  }

  void advanceReadOffset(const RowSet& rows) {
    readOffset_ += rows.back() + 1;
  }
};

// ---------------------------------------------------------------------------
// SubIntSplit test support.
//
// SubIntSplit is not yet wired into the EncodingFactory dispatch, so we build
// the encoding directly: encode via the static encode() with a policy that
// never recurses back into SubIntSplit for the section sub-streams, then
// construct the typed encoding (its constructor decodes the sections via the
// regular factory) and call readWithVisitor on it directly.
// ---------------------------------------------------------------------------
template <typename T>
class NonRecursiveSubIntSplitPolicy final : public EncodingSelectionPolicy<T> {
  using physicalType = typename TypeTraits<T>::physicalType;

 public:
  EncodingSelectionResult select(
      std::span<const physicalType> /* values */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) override {
    return {.encodingType = EncodingType::SubIntSplit};
  }

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) override {
    return {.encodingType = EncodingType::Nullable};
  }

  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType /* encodingType */,
      NestedEncodingIdentifier /* identifier */,
      DataType type) override {
    auto readFactors =
        ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();
    readFactors.erase(
        std::remove_if(
            readFactors.begin(),
            readFactors.end(),
            [](const auto& factor) {
              return factor.first == EncodingType::SubIntSplit;
            }),
        readFactors.end());
    ManualEncodingSelectionPolicyFactory factory{
        std::move(readFactors), std::nullopt};
    return factory.createPolicy(type);
  }
};

template <typename T>
std::unique_ptr<SubIntSplitEncoding<T>> makeSubIntSplitEncoding(
    const std::vector<T>& data,
    Buffer& buffer,
    velox::memory::MemoryPool& memPool) {
  using PhysicalType = typename TypeTraits<T>::physicalType;
  auto span = std::span<const PhysicalType>(
      reinterpret_cast<const PhysicalType*>(data.data()), data.size());
  EncodingSelection<PhysicalType> selection{
      {.encodingType = EncodingType::SubIntSplit},
      Statistics<PhysicalType>::create(span),
      std::make_unique<NonRecursiveSubIntSplitPolicy<T>>()};
  auto encoded = SubIntSplitEncoding<T>::encode(selection, span, buffer);
  return std::make_unique<SubIntSplitEncoding<T>>(
      memPool, encoded, [](uint32_t) { return nullptr; });
}

EncodingLayout makeAlpEncodingLayout(EncodingType encodedValuesEncodingType) {
  const auto encodedValuesLayout = [&] {
    switch (encodedValuesEncodingType) {
      case EncodingType::Trivial:
        return EncodingLayout{
            EncodingType::Trivial, {}, CompressionType::Uncompressed};
      case EncodingType::Dictionary:
        return EncodingLayout{
            EncodingType::Dictionary,
            {},
            CompressionType::Uncompressed,
            {std::nullopt, std::nullopt}};
      default:
        NIMBLE_UNREACHABLE(
            "Unsupported ALP encoded values encoding type: {}",
            encodedValuesEncodingType);
    }
  }();

  return EncodingLayout{
      EncodingType::ALP,
      {},
      CompressionType::Uncompressed,
      {encodedValuesLayout}};
}

WriterOptions makeSingleColumnWriterOptions(
    const EncodingLayout& encodingLayout) {
  using StreamLayouts =
      std::unordered_map<EncodingLayoutTree::StreamIdentifier, EncodingLayout>;

  WriterOptions options;
  std::vector<EncodingLayoutTree> children;
  children.emplace_back(
      Kind::Scalar,
      StreamLayouts{
          {EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
           encodingLayout}},
      "c0");
  options.encodingLayoutTree.emplace(
      Kind::Row, StreamLayouts{}, "", std::move(children));
  return options;
}

template <typename T>
std::vector<T> makeAlpFloatingPointValues() {
  constexpr int kRows = 120;
  std::vector<T> data(kRows);
  for (int i = 0; i < kRows; ++i) {
    data[i] = static_cast<T>((i % 21) - 10) / static_cast<T>(4);
  }
  return data;
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
// Parameter: true = non-legacy (stringDecoderZeroCopy=true),
//            false = legacy (stringDecoderZeroCopy=false).
class ReadWithVisitorTest : public ::testing::TestWithParam<bool>,
                            public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::initializeMemoryManager(memory::MemoryManager::Options{});
  }

  bool useNonLegacy() const {
    return GetParam();
  }

  // Holds the infrastructure needed to build column readers for a single file.
  struct FileContext {
    std::string fileData;
    std::shared_ptr<ReaderBase> readerBase;
    std::unique_ptr<StripeStreams> streams;
    dwio::common::SplitStats stats{dwio::common::FileFormat::NIMBLE};
    std::unique_ptr<RowSizeTracker> rowSizeTracker;
    EncodingFactory encodingFactory;
  };

  // Write |input| to an in-memory nimble file and prepare the stripe for
  // reading.  Returns a heap-allocated context to avoid copy/move issues.
  std::unique_ptr<FileContext> makeFileContext(
      const RowVectorPtr& input,
      WriterOptions writerOptions = {}) {
    auto ctx = std::make_unique<FileContext>();
    ctx->fileData =
        test::createNimbleFile(*rootPool_, input, std::move(writerOptions));

    auto readFile = std::make_shared<InMemoryReadFile>(ctx->fileData);
    dwio::common::ReaderOptions readerOpts(pool());
    readerOpts.setDataIoStats(dataIoStats_);
    readerOpts.setMetadataIoStats(metadataIoStats_);
    ctx->readerBase = ReaderBase::create(
        std::make_unique<dwio::common::BufferedInput>(readFile, *pool()),
        readerOpts);

    ctx->streams = std::make_unique<StripeStreams>(ctx->readerBase);
    ctx->streams->setStripe(0);

    ctx->rowSizeTracker =
        std::make_unique<RowSizeTracker>(ctx->readerBase->fileSchemaWithId());

    return ctx;
  }

  std::optional<EncodingLayout> captureFirstColumnEncoding(FileContext& ctx) {
    const auto& childNode =
        ctx.readerBase->nimbleSchema()->asRow().childAt(0)->asScalar();
    auto streams = ctx.readerBase->tablet().load(
        ctx.readerBase->tablet().stripeIdentifier(0),
        std::vector<uint32_t>{childNode.scalarDescriptor().offset()});

    InMemoryChunkedStream chunkedStream{*pool(), std::move(streams[0])};
    if (!chunkedStream.hasNext()) {
      return std::nullopt;
    }
    return EncodingLayoutCapture::capture(
        chunkedStream.nextChunk(), Encoding::Options{});
  }

  template <typename T>
  void testColumnReaderAlpFloatingPointRange(
      EncodingType encodedValuesEncodingType) {
    SCOPED_TRACE(fmt::format("dataType={}", toString(TypeTraits<T>::dataType)));
    constexpr T kLower = static_cast<T>(-1.25);
    constexpr T kUpper = static_cast<T>(1.25);

    const auto data = makeAlpFloatingPointValues<T>();
    auto input = makeRowVector(
        {makeFlatVector<T>(data.size(), [&](auto row) { return data[row]; })});
    auto rowType = asRowType(input->type());
    auto ctx = makeFileContext(
        input,
        makeSingleColumnWriterOptions(
            makeAlpEncodingLayout(encodedValuesEncodingType)));

    auto captured = captureFirstColumnEncoding(*ctx);
    ASSERT_TRUE(captured.has_value());
    ASSERT_EQ(captured->encodingType(), EncodingType::ALP);
    const auto& encodedValues =
        captured->child(EncodingIdentifiers::ALP::EncodedValues);
    ASSERT_TRUE(encodedValues.has_value());
    EXPECT_EQ(encodedValues->encodingType(), encodedValuesEncodingType);
    if (encodedValuesEncodingType == EncodingType::Dictionary) {
      const auto& alphabet =
          encodedValues->child(EncodingIdentifiers::Dictionary::Alphabet);
      const auto& indices =
          encodedValues->child(EncodingIdentifiers::Dictionary::Indices);
      EXPECT_TRUE(alphabet.has_value());
      EXPECT_TRUE(indices.has_value());
    }

    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*rowType);
    scanSpec->childByName("c0")->setFilter(
        std::make_unique<common::FloatingPointRange<T>>(
            kLower, false, false, kUpper, false, false, false));

    auto root =
        buildReader(*ctx, rowType, *scanSpec, /*stringDecoderZeroCopy=*/true);
    auto* child = readColumn(root.get(), data.size());

    std::vector<vector_size_t> expectedRows;
    std::vector<T> expectedValues;
    for (int i = 0; i < static_cast<int>(data.size()); ++i) {
      if (data[i] >= kLower && data[i] <= kUpper) {
        expectedRows.push_back(i);
        expectedValues.push_back(data[i]);
      }
    }

    EXPECT_EQ(child->numValues(), expectedValues.size());

    const auto outputRows = child->outputRows();
    ASSERT_EQ(outputRows.size(), expectedRows.size());
    for (int i = 0; i < static_cast<int>(expectedRows.size()); ++i) {
      SCOPED_TRACE(fmt::format("rowIndex={}", i));
      EXPECT_EQ(outputRows[i], expectedRows[i]);
    }

    const auto actualValues = getValues<T>(child);
    ASSERT_EQ(actualValues.size(), expectedValues.size());
    for (int i = 0; i < static_cast<int>(expectedValues.size()); ++i) {
      SCOPED_TRACE(fmt::format("valueIndex={}", i));
      EXPECT_EQ(actualValues[i], expectedValues[i]);
    }
  }

  template <typename T>
  void testEncodingLevelAlpRandomizedSparse(
      EncodingType encodedValuesEncodingType) {
    SCOPED_TRACE(
        fmt::format(
            "dataType={} encodedValuesEncodingType={}",
            toString(TypeTraits<T>::dataType),
            toString(encodedValuesEncodingType)));

    constexpr int kRows = 512;
    constexpr T kLower = static_cast<T>(-25);
    constexpr T kUpper = static_cast<T>(25);

    std::mt19937 rng(0xA117);
    std::uniform_int_distribution<int32_t> valueDistribution{-5'000, 5'000};
    std::uniform_int_distribution<int32_t> rowDistribution{0, 3};

    std::vector<T> data(kRows);
    for (int i = 0; i < kRows; ++i) {
      data[i] = static_cast<T>(valueDistribution(rng)) / static_cast<T>(100);
    }

    std::vector<vector_size_t> rowVec;
    for (int i = 0; i < kRows; ++i) {
      if (rowDistribution(rng) == 0) {
        rowVec.push_back(i);
      }
    }
    ASSERT_FALSE(rowVec.empty());

    auto input = makeRowVector(
        {makeFlatVector<T>(kRows, [&](auto i) { return data[i]; })});
    auto rowType = asRowType(input->type());
    auto ctx = makeFileContext(input);
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*rowType);
    scanSpec->childByName("c0")->setFilter(
        std::make_unique<common::FloatingPointRange<T>>(
            kLower, false, false, kUpper, false, false, false));
    auto root = buildReader(*ctx, rowType, *scanSpec);

    auto* structReader =
        dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(
            root.get());
    auto* reader = static_cast<FloatingPointColumnReaderTestAccessor<T>*>(
        dynamic_cast<FloatingPointColumnReader<T, T>*>(
            structReader->children()[0]));
    ASSERT_NE(reader, nullptr);

    RowSet rows(rowVec.data(), rowVec.size());
    reader->doPrepareRead(0, rows, nullptr);

    Buffer buffer(*pool());
    auto encoding = createFromCustomLayout<T>(
        makeAlpEncodingLayout(encodedValuesEncodingType),
        data,
        *pool(),
        buffer);
    ASSERT_EQ(encoding->encodingType(), EncodingType::ALP);

    common::FloatingPointRange<T> filter(
        kLower, false, false, kUpper, false, false, false);
    dwio::common::ExtractToReader extractValues(reader);
    constexpr bool kIsDense = false;
    DecoderVisitor<
        T,
        common::FloatingPointRange<T>,
        dwio::common::ExtractToReader,
        kIsDense>
        visitor(filter, reader, rows, extractValues);
    auto params = makeReadWithVisitorParams(visitor, rows, pool());

    dispatchCallReadWithVisitor(*encoding, visitor, params);

    std::vector<T> expectedValues;
    for (const auto row : rowVec) {
      if (data[row] >= kLower && data[row] <= kUpper) {
        expectedValues.push_back(data[row]);
      }
    }

    EXPECT_EQ(reader->numValues(), expectedValues.size());
    const auto actualValues = getValues<T>(reader);
    ASSERT_EQ(actualValues.size(), expectedValues.size());
    for (int i = 0; i < static_cast<int>(expectedValues.size()); ++i) {
      SCOPED_TRACE(fmt::format("valueIndex={}", i));
      EXPECT_EQ(actualValues[i], expectedValues[i]);
    }
  }

  std::unique_ptr<dwio::common::SelectiveColumnReader> buildReader(
      FileContext& ctx,
      const RowTypePtr& rowType,
      common::ScanSpec& scanSpec,
      bool stringDecoderZeroCopy = false) {
    NimbleParams params(
        *pool(),
        ctx.stats,
        ctx.readerBase->nimbleSchema(),
        *ctx.streams,
        ctx.rowSizeTracker.get(),
        ctx.encodingFactory,
        stringDecoderZeroCopy);

    auto reader = buildColumnReader(
        rowType,
        ctx.readerBase->fileSchemaWithId(),
        params,
        scanSpec,
        /*isRoot=*/true);
    reader->setIsTopLevel();
    ctx.streams->load();
    return reader;
  }

  // Decode an encoded buffer with the appropriate factory.
  std::unique_ptr<Encoding> decodeEncoding(
      std::string_view encoded,
      velox::memory::MemoryPool& memPool) {
    if (useNonLegacy()) {
      return EncodingFactory().create(
          memPool, encoded, nullptr, Encoding::Options{});
    }
    return legacy::EncodingFactory().create(
        memPool, encoded, nullptr, Encoding::Options{});
  }

  // Dispatch callReadWithVisitor to the appropriate family.
  template <typename V>
  void dispatchCallReadWithVisitor(
      Encoding& encoding,
      V& visitor,
      ReadWithVisitorParams& params) {
    if (useNonLegacy()) {
      nimble::callReadWithVisitor(encoding, visitor, params);
    } else {
      legacy::callReadWithVisitor(encoding, visitor, params);
    }
  }

  // Encode data with a custom layout, then decode with the appropriate factory.
  template <typename T>
  std::unique_ptr<Encoding> createFromCustomLayout(
      const EncodingLayout& layout,
      const std::vector<T>& data,
      velox::memory::MemoryPool& memPool,
      Buffer& buffer) {
    auto policy = std::make_unique<ReplayedEncodingSelectionPolicy<T>>(
        layout,
        CompressionOptions{},
        [](DataType type) -> std::unique_ptr<EncodingSelectionPolicyBase> {
          UNIQUE_PTR_FACTORY(type, TrivialNestedPolicy);
        });
    auto encoded = EncodingFactory::encode<T>(
        std::move(policy),
        std::span<const T>(data.data(), data.size()),
        buffer);
    return decodeEncoding(encoded, memPool);
  }

  /// Manually encode data as MainlyConstant wrapping Dictionary.
  /// Works for all types including string_view (unlike createFromCustomLayout
  /// which fails for string_view due to nested sub-encoding depth).
  template <typename T>
  std::unique_ptr<Encoding> encodeMainlyConstantDictionary(
      const std::vector<T>& values) {
    using PhysicalType = typename TypeTraits<T>::physicalType;
    NIMBLE_CHECK(!values.empty());
    const uint32_t rowCount = values.size();

    std::unordered_map<PhysicalType, uint32_t> counts;
    for (const auto& v : values) {
      ++counts[reinterpret_cast<const PhysicalType&>(v)];
    }
    PhysicalType commonValue{};
    uint32_t maxCount = 0;
    for (const auto& [val, cnt] : counts) {
      if (cnt > maxCount) {
        maxCount = cnt;
        commonValue = val;
      }
    }

    nimble::Vector<bool> isCommon{pool(), rowCount, false};
    nimble::Vector<T> otherVec{pool()};
    for (size_t i = 0; i < values.size(); ++i) {
      if (reinterpret_cast<const PhysicalType&>(values[i]) == commonValue) {
        isCommon[i] = true;
      } else {
        otherVec.push_back(values[i]);
      }
    }

    auto isCommonSpan = std::span<const bool>(isCommon.data(), isCommon.size());
    Buffer isCommonBuf{*pool()};
    EncodingSelection<bool> isCommonSel{
        {.encodingType = EncodingType::Trivial},
        Statistics<bool>::create(isCommonSpan),
        std::make_unique<TrivialNestedPolicy<bool>>()};
    auto serializedIsCommon =
        TrivialEncoding<bool>::encode(isCommonSel, isCommonSpan, isCommonBuf);

    auto otherSpan = std::span<const PhysicalType>(
        reinterpret_cast<const PhysicalType*>(otherVec.data()),
        otherVec.size());
    Buffer otherBuf{*pool()};
    EncodingSelection<PhysicalType> otherSel{
        {.encodingType = EncodingType::Dictionary},
        Statistics<PhysicalType>::create(otherSpan),
        std::make_unique<TrivialNestedPolicy<T>>()};
    auto serializedOther =
        DictionaryEncoding<T>::encode(otherSel, otherSpan, otherBuf);

    uint32_t encodingSize = Encoding::kPrefixSize + 8 +
        serializedIsCommon.size() + serializedOther.size();
    if constexpr (isNumericType<PhysicalType>()) {
      encodingSize += sizeof(PhysicalType);
    } else {
      encodingSize += 4 + commonValue.size();
    }
    auto& buf = mcDictBuffer_;
    buf = std::make_unique<Buffer>(*pool());
    char* reserved = buf->reserve(encodingSize);
    char* pos = reserved;
    encoding::writeChar(static_cast<char>(EncodingType::MainlyConstant), pos);
    encoding::writeChar(static_cast<char>(TypeTraits<T>::dataType), pos);
    encoding::writeUint32(rowCount, pos);
    encoding::writeString(serializedIsCommon, pos);
    encoding::writeString(serializedOther, pos);
    encoding::write<PhysicalType>(commonValue, pos);

    return EncodingFactory().create(
        *pool(),
        std::string_view(reserved, encodingSize),
        [this](uint32_t size) {
          mcDictStringBuffers_.push_back(
              velox::AlignedBuffer::allocate<char>(size, pool()));
          return mcDictStringBuffers_.back()->asMutable<void>();
        },
        Encoding::Options{});
  }

  // Build root reader, get its first child, call read(), return child.
  // Caller inspects outputRows / numValues / rawValues / hasNulls on it.
  dwio::common::SelectiveColumnReader* readColumn(
      dwio::common::SelectiveColumnReader* root,
      int numRows) {
    // RowSet: sequential [0 .. numRows-1]
    std::vector<vector_size_t> rowVec(numRows);
    std::iota(rowVec.begin(), rowVec.end(), 0);
    RowSet rows(rowVec.data(), rowVec.size());

    // Trigger read on the root struct reader which calls read() on children.
    root->read(0, rows, nullptr);

    auto* structReader =
        dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root);
    EXPECT_NE(structReader, nullptr);
    EXPECT_FALSE(structReader->children().empty());
    return structReader->children()[0];
  }

  // Helper: get typed values from a column reader after read().
  template <typename T>
  std::vector<T> getValues(dwio::common::SelectiveColumnReader* reader) {
    auto* raw = reinterpret_cast<const T*>(reader->rawValues());
    return std::vector<T>(raw, raw + reader->numValues());
  }

  const std::shared_ptr<io::IoStatistics> dataIoStats_{
      std::make_shared<io::IoStatistics>()};
  const std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};
  std::unique_ptr<Buffer> mcDictBuffer_;
  std::vector<velox::BufferPtr> mcDictStringBuffers_;
};

class ReadWithVisitorNonLegacyTest : public ReadWithVisitorTest {};

// ===========================================================================
// Test: Dense read, no filter, sequential int64 data (no nulls)
// Exercises TrivialEncoding / FixedBitWidthEncoding readWithVisitor with
// AlwaysTrue filter.
// ===========================================================================
TEST_P(ReadWithVisitorTest, denseNoFilterNoNulls) {
  constexpr int kRows = 200;
  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kRows, [](auto i) { return i * 7; })});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  // AlwaysTrue filter forces read() instead of lazy-loading the child.
  // This exercises the same readWithVisitor<AlwaysTrue> code path.
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* child = readColumn(root.get(), kRows);

  EXPECT_EQ(child->numValues(), kRows);
  auto values = getValues<int64_t>(child);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(values[i], i * 7) << "row " << i;
  }
}

// ===========================================================================
// Test: Dense read, no filter, nullable int64 (exercises NullableEncoding)
// ===========================================================================
TEST_P(ReadWithVisitorTest, denseNoFilterWithNulls) {
  constexpr int kRows = 200;
  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kRows, folly::identity, nullEvery(7))});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* child = readColumn(root.get(), kRows);

  // Every row is "output" since there is no filter.
  EXPECT_EQ(child->numValues(), kRows);
  EXPECT_TRUE(child->hasNulls());
}

// ===========================================================================
// Test: BigintRange filter, sequential data — sparse visitor output
// ===========================================================================
TEST_P(ReadWithVisitorTest, sparseWithBigintRange) {
  constexpr int kRows = 500;
  auto input = makeRowVector({makeFlatVector<int64_t>(kRows, folly::identity)});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(100, 200, false));

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* child = readColumn(root.get(), kRows);

  // Exactly rows [100..200] pass the filter → 101 values.
  EXPECT_EQ(child->numValues(), 101);
  auto rows = child->outputRows();
  EXPECT_EQ(rows[0], 100);
  EXPECT_EQ(rows[rows.size() - 1], 200);
}

// ===========================================================================
// Test: BigintRange filter + nullable data — filter skips nulls
// ===========================================================================
TEST_P(ReadWithVisitorTest, rangeFilterWithNulls) {
  constexpr int kRows = 500;
  auto c0 = makeFlatVector<int64_t>(kRows, folly::identity, nullEvery(7));
  auto input = makeRowVector({c0});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(50, 300, false));

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* child = readColumn(root.get(), kRows);

  // Count expected: value in [50,300] AND not null.
  int expected = 0;
  for (int i = 0; i < kRows; ++i) {
    if (!c0->isNullAt(i) && c0->valueAt(i) >= 50 && c0->valueAt(i) <= 300) {
      ++expected;
    }
  }
  EXPECT_EQ(child->numValues(), expected);
}

// ===========================================================================
// Test: IsNotNull filter on nullable data
// ===========================================================================
TEST_P(ReadWithVisitorTest, isNotNullFilter) {
  constexpr int kRows = 300;
  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kRows, folly::identity, nullEvery(5))});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(std::make_unique<common::IsNotNull>());

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* child = readColumn(root.get(), kRows);

  int expectedNonNull = 0;
  for (int i = 0; i < kRows; ++i) {
    expectedNonNull += (i % 5 != 0) ? 1 : 0;
  }
  EXPECT_EQ(child->numValues(), expectedNonNull);
  // All output values are non-null; hasNulls should be false (or true only
  // if the encoding delivers reader-nulls).
}

// ===========================================================================
// Test: Constant data → ConstantEncoding::readWithVisitor
// ===========================================================================
TEST_P(ReadWithVisitorTest, constantEncoding) {
  constexpr int kRows = 300;
  auto input =
      makeRowVector({makeFlatVector<int64_t>(kRows, [](auto) { return 42; })});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* child = readColumn(root.get(), kRows);

  EXPECT_EQ(child->numValues(), kRows);
  auto values = getValues<int64_t>(child);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(values[i], 42);
  }
}

// ===========================================================================
// Test: Constant data + non-matching filter → zero output rows
// ===========================================================================
TEST_P(ReadWithVisitorTest, constantEncodingFilterMiss) {
  constexpr int kRows = 300;
  auto input =
      makeRowVector({makeFlatVector<int64_t>(kRows, [](auto) { return 42; })});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(100, 200, false));

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* child = readColumn(root.get(), kRows);

  EXPECT_EQ(child->numValues(), 0);
  EXPECT_TRUE(child->outputRows().empty());
}

// ===========================================================================
// Test: Boolean data with BoolValue filter (SparseBoolEncoding)
// ===========================================================================
TEST_P(ReadWithVisitorTest, boolWithFilter) {
  constexpr int kRows = 300;
  auto c0 = makeFlatVector<bool>(kRows, [](auto i) { return i % 3 != 0; });
  auto input = makeRowVector({c0});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BoolValue>(true, false));

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* child = readColumn(root.get(), kRows);

  int expectedTrue = 0;
  for (int i = 0; i < kRows; ++i) {
    expectedTrue += (i % 3 != 0) ? 1 : 0;
  }
  EXPECT_EQ(child->numValues(), expectedTrue);
}

// ===========================================================================
// Test: Boolean data with nulls + filter
// ===========================================================================
TEST_P(ReadWithVisitorTest, boolWithNullsAndFilter) {
  constexpr int kRows = 300;
  auto c0 = makeFlatVector<bool>(
      kRows, [](auto i) { return i % 3 != 0; }, nullEvery(7));
  auto input = makeRowVector({c0});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BoolValue>(true, false));

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* child = readColumn(root.get(), kRows);

  int expected = 0;
  for (int i = 0; i < kRows; ++i) {
    if (!c0->isNullAt(i) && c0->valueAt(i)) {
      ++expected;
    }
  }
  EXPECT_EQ(child->numValues(), expected);
}

// ===========================================================================
// Test: String data, no filter (TrivialEncoding<string_view>)
// ===========================================================================
TEST_P(ReadWithVisitorTest, stringNoFilter) {
  constexpr int kRows = 200;
  auto input = makeRowVector({makeFlatVector<StringView>(kRows, [](auto i) {
    return StringView::makeInline(fmt::format("s{}", i));
  })});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* child = readColumn(root.get(), kRows);

  EXPECT_EQ(child->numValues(), kRows);
}

// ===========================================================================
// Test: String data with nulls + IsNotNull filter
// ===========================================================================
TEST_P(ReadWithVisitorTest, stringWithNullsIsNotNull) {
  constexpr int kRows = 200;
  auto c0 = makeFlatVector<StringView>(
      kRows,
      [](auto i) { return StringView::makeInline(fmt::format("s{}", i)); },
      nullEvery(5));
  auto input = makeRowVector({c0});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(std::make_unique<common::IsNotNull>());

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* child = readColumn(root.get(), kRows);

  int expectedNonNull = 0;
  for (int i = 0; i < kRows; ++i) {
    expectedNonNull += (i % 5 != 0) ? 1 : 0;
  }
  EXPECT_EQ(child->numValues(), expectedNonNull);
}

// ===========================================================================
// Test: All-null data → NullableEncoding with all-null inner
// ===========================================================================
TEST_P(ReadWithVisitorTest, allNullData) {
  constexpr int kRows = 100;
  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kRows, [](auto) { return 0; }, nullEvery(1))});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  // Use IsNotNull to verify null handling in readWithVisitor — all rows are
  // null so zero rows should pass.
  scanSpec->childByName("c0")->setFilter(std::make_unique<common::IsNotNull>());

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* child = readColumn(root.get(), kRows);

  EXPECT_EQ(child->numValues(), 0);
  EXPECT_TRUE(child->outputRows().empty());
}

// ===========================================================================
// Test: All-null data + IsNotNull filter → zero output
// ===========================================================================
TEST_P(ReadWithVisitorTest, allNullWithIsNotNull) {
  constexpr int kRows = 100;
  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kRows, [](auto) { return 0; }, nullEvery(1))});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(std::make_unique<common::IsNotNull>());

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* child = readColumn(root.get(), kRows);

  EXPECT_EQ(child->numValues(), 0);
}

// ===========================================================================
// Test: Large sequential data + range filter (multi-chunk / stripe)
// ===========================================================================
TEST_P(ReadWithVisitorTest, largeDataRangeFilter) {
  constexpr int kRows = 10000;
  auto c0 = makeFlatVector<int64_t>(kRows, folly::identity, nullEvery(13));
  auto input = makeRowVector({c0});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(1000, 5000, false));

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* child = readColumn(root.get(), kRows);

  int expected = 0;
  for (int i = 0; i < kRows; ++i) {
    if (!c0->isNullAt(i) && c0->valueAt(i) >= 1000 && c0->valueAt(i) <= 5000) {
      ++expected;
    }
  }
  EXPECT_EQ(child->numValues(), expected);

  // Verify outputRows are sorted and in range.
  auto rows = child->outputRows();
  for (int i = 0; i < rows.size(); ++i) {
    EXPECT_GE(rows[i], 1000);
    EXPECT_LE(rows[i], 5000);
    if (i > 0) {
      EXPECT_GT(rows[i], rows[i - 1]);
    }
  }
}

// ===========================================================================
// Test: Multiple types (int32, double) — exercises different column readers
// ===========================================================================
TEST_P(ReadWithVisitorTest, multipleColumnTypesWithFilters) {
  constexpr int kRows = 300;
  auto c0 =
      makeFlatVector<int32_t>(kRows, [](auto i) { return i; }, nullEvery(11));
  auto c1 = makeFlatVector<double>(
      kRows, [](auto i) { return i * 0.5; }, nullEvery(7));
  auto input = makeRowVector({c0, c1});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  // Filter on int column — exercises int readWithVisitor with sparse output.
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(50, 150, false));

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* child = readColumn(root.get(), kRows);

  int expected = 0;
  for (int i = 0; i < kRows; ++i) {
    if (!c0->isNullAt(i) && c0->valueAt(i) >= 50 && c0->valueAt(i) <= 150) {
      ++expected;
    }
  }
  EXPECT_EQ(child->numValues(), expected);
  EXPECT_FALSE(child->outputRows().empty());
}

// ===========================================================================
// Test: Filter on one column of multi-column row — exercises sparse rows
// propagation to second column
// ===========================================================================
TEST_P(ReadWithVisitorTest, filterOnOneColumnInspectState) {
  constexpr int kRows = 300;
  auto c0 = makeFlatVector<int64_t>(kRows, folly::identity);
  auto input = makeRowVector({c0});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(50, 100, false));

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* child = readColumn(root.get(), kRows);

  // c0 filter passes rows [50..100] → 51 values.
  EXPECT_EQ(child->numValues(), 51);

  // Verify outputRows are exactly [50..100].
  auto rows = child->outputRows();
  EXPECT_EQ(rows.size(), 51);
  for (int i = 0; i < rows.size(); ++i) {
    EXPECT_EQ(rows[i], 50 + i);
  }

  // Verify values.
  auto values = getValues<int64_t>(child);
  for (int i = 0; i < values.size(); ++i) {
    EXPECT_EQ(values[i], 50 + i);
  }
}

// ===========================================================================
// EXPLICIT readWithVisitor TESTS
//
// These tests manually construct a ColumnVisitor with explicit template
// parameters and call readWithVisitor on the concrete column reader.
// This bypasses readCommon / processFilter entirely.
// ===========================================================================

// Explicit readWithVisitor: BigintRange filter, dense rows, ExtractToReader.
TEST_P(ReadWithVisitorTest, explicitReadWithVisitorBigintRangeDense) {
  constexpr int kRows = 500;
  auto input = makeRowVector({makeFlatVector<int64_t>(kRows, folly::identity)});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(100, 200, false));

  auto root = buildReader(*ctx, rowType, *scanSpec);

  // Extract the child IntegerColumnReader.
  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  ASSERT_NE(structReader, nullptr);
  auto* child = structReader->children()[0];

  // Cast to our test accessor to call prepareRead.
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(child));
  ASSERT_NE(reader, nullptr);

  // Build dense RowSet [0..kRows-1].
  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  // Step 1: prepareRead — sets up nulls, output buffers, values capacity.
  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  // Step 2: Construct ColumnVisitor with explicit template parameters.
  common::BigintRange filter(100, 200, false);
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int64_t,
      common::BigintRange,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);

  // Step 3: Explicit readWithVisitor call.
  reader->readWithVisitor(rows, std::move(visitor));
  reader->advanceReadOffset(rows);

  // Step 4: Inspect reader state.
  EXPECT_EQ(reader->numValues(), 101);
  auto outRows = reader->outputRows();
  EXPECT_EQ(outRows.size(), 101);
  EXPECT_EQ(outRows[0], 100);
  EXPECT_EQ(outRows[100], 200);

  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < 101; ++i) {
    EXPECT_EQ(values[i], 100 + i);
  }
}

// Explicit readWithVisitor: AlwaysTrue filter (no filtering), dense rows.
TEST_P(ReadWithVisitorTest, explicitReadWithVisitorAlwaysTrueDense) {
  constexpr int kRows = 200;
  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kRows, [](auto i) { return i * 3; })});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int64_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);

  reader->readWithVisitor(rows, std::move(visitor));
  reader->advanceReadOffset(rows);

  EXPECT_EQ(reader->numValues(), kRows);
  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(values[i], i * 3);
  }
}

// Explicit readWithVisitor: BigintRange filter, sparse rows (non-dense).
TEST_P(ReadWithVisitorTest, explicitReadWithVisitorBigintRangeSparse) {
  constexpr int kRows = 500;
  auto input = makeRowVector({makeFlatVector<int64_t>(kRows, folly::identity)});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(0, 499, false));

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  // Sparse RowSet: every other row → non-dense (rows.back() != rows.size()-1).
  std::vector<vector_size_t> rowVec;
  for (int i = 0; i < kRows; i += 2) {
    rowVec.push_back(i);
  }
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  common::BigintRange filter(0, 499, false);
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = false;
  DecoderVisitor<
      int64_t,
      common::BigintRange,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);

  reader->readWithVisitor(rows, std::move(visitor));
  reader->advanceReadOffset(rows);

  // All even-indexed rows pass the filter.
  EXPECT_EQ(reader->numValues(), static_cast<int>(rowVec.size()));
  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < values.size(); ++i) {
    EXPECT_EQ(values[i], i * 2);
  }
}

// Explicit readWithVisitor: IsNotNull filter on nullable data.
TEST_P(ReadWithVisitorTest, explicitReadWithVisitorIsNotNullNullable) {
  constexpr int kRows = 300;
  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kRows, folly::identity, nullEvery(5))});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(std::make_unique<common::IsNotNull>());

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  common::IsNotNull filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int64_t,
      common::IsNotNull,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);

  reader->readWithVisitor(rows, std::move(visitor));
  reader->advanceReadOffset(rows);

  int expectedNonNull = 0;
  for (int i = 0; i < kRows; ++i) {
    expectedNonNull += (i % 5 != 0) ? 1 : 0;
  }
  EXPECT_EQ(reader->numValues(), expectedNonNull);
}

TEST_P(ReadWithVisitorNonLegacyTest, columnReaderAlpFloatAndDouble) {
  for (const auto encodedValuesEncodingType :
       {EncodingType::Trivial, EncodingType::Dictionary}) {
    SCOPED_TRACE(
        fmt::format(
            "encodedValuesEncodingType={}",
            toString(encodedValuesEncodingType)));
    testColumnReaderAlpFloatingPointRange<float>(encodedValuesEncodingType);
    testColumnReaderAlpFloatingPointRange<double>(encodedValuesEncodingType);
  }
}

TEST_P(ReadWithVisitorNonLegacyTest, encodingLevelAlpRandomizedSparse) {
  for (const auto encodedValuesEncodingType :
       {EncodingType::Trivial, EncodingType::Dictionary}) {
    SCOPED_TRACE(
        fmt::format(
            "encodedValuesEncodingType={}",
            toString(encodedValuesEncodingType)));
    testEncodingLevelAlpRandomizedSparse<float>(encodedValuesEncodingType);
    testEncodingLevelAlpRandomizedSparse<double>(encodedValuesEncodingType);
  }
}

// ===========================================================================
// ENCODING-LEVEL readWithVisitor TESTS
//
// These tests create Encoding* instances directly with forced encoding types
// and call callReadWithVisitor(*encoding, visitor, params) on them, fully
// isolating the encoding under test from file I/O.  The nimble file is only
// used to set up the SelectiveColumnReader infrastructure needed by the
// ColumnVisitor.
// ===========================================================================

// Construct ReadWithVisitorParams for the non-null path, modeled after
// ChunkedDecoder::readWithVisitor.
template <typename V>
ReadWithVisitorParams makeReadWithVisitorParams(
    V& visitor,
    const RowSet& rows,
    velox::memory::MemoryPool* memPool) {
  ReadWithVisitorParams params{};
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

// ---------------------------------------------------------------------------
// 1. TrivialEncoding<int64_t> + BigintRange + dense
// ---------------------------------------------------------------------------
TEST_P(ReadWithVisitorTest, encodingLevelTrivialBigintRangeDense) {
  constexpr int kRows = 500;

  // Sequential data [0..499].
  std::vector<int64_t> data(kRows);
  std::iota(data.begin(), data.end(), 0);

  // Write nimble file with the same data (for reader infrastructure).
  auto input = makeRowVector({makeFlatVector<int64_t>(kRows, folly::identity)});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(100, 200, false));
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  // Create the encoding independently with forced TrivialEncoding.
  Buffer buffer(*pool());
  auto encoding =
      createFromCustomLayout<int64_t>(TrivialEnc{}, data, *pool(), buffer);

  // Construct visitor and params, then call callReadWithVisitor directly.
  common::BigintRange filter(100, 200, false);
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int64_t,
      common::BigintRange,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  EXPECT_EQ(reader->numValues(), 101);
  auto outRows = reader->outputRows();
  EXPECT_EQ(outRows[0], 100);
  EXPECT_EQ(outRows[100], 200);
  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < 101; ++i) {
    EXPECT_EQ(values[i], 100 + i);
  }
}

// ---------------------------------------------------------------------------
// 2. ConstantEncoding<int64_t> + AlwaysTrue + dense
// ---------------------------------------------------------------------------
TEST_P(ReadWithVisitorTest, encodingLevelConstantAlwaysTrueDense) {
  constexpr int kRows = 300;

  // All-same data.
  std::vector<int64_t> data(kRows, 42);

  auto input =
      makeRowVector({makeFlatVector<int64_t>(kRows, [](auto) { return 42; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding =
      createFromCustomLayout<int64_t>(ConstantEnc{}, data, *pool(), buffer);

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int64_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  EXPECT_EQ(reader->numValues(), kRows);
  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(values[i], 42);
  }
}

// ---------------------------------------------------------------------------
// 3. FixedBitWidthEncoding<int64_t> + BigintRange + sparse
// ---------------------------------------------------------------------------
TEST_P(ReadWithVisitorTest, encodingLevelFixedBitWidthBigintRangeSparse) {
  constexpr int kRows = 500;

  // Small-range data [0..15] that fits in fixed-bit-width encoding.
  std::vector<int64_t> data(kRows);
  for (int i = 0; i < kRows; ++i) {
    data[i] = i % 16;
  }

  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kRows, [](auto i) { return i % 16; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(5, 10, false));
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  // Sparse RowSet: every other row.
  std::vector<vector_size_t> rowVec;
  for (int i = 0; i < kRows; i += 2) {
    rowVec.push_back(i);
  }
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int64_t>(
      FixedBitWidthEnc{}, data, *pool(), buffer);

  common::BigintRange filter(5, 10, false);
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = false;
  DecoderVisitor<
      int64_t,
      common::BigintRange,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  // Count expected: even-indexed rows where data[i] in [5..10].
  int expected = 0;
  for (int i = 0; i < kRows; i += 2) {
    if (data[i] >= 5 && data[i] <= 10) {
      ++expected;
    }
  }
  EXPECT_EQ(reader->numValues(), expected);

  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < reader->numValues(); ++i) {
    EXPECT_GE(values[i], 5);
    EXPECT_LE(values[i], 10);
  }
}

TEST_P(ReadWithVisitorNonLegacyTest, encodingLevelAlpFloatingPointRangeSparse) {
  constexpr int kRows = 120;
  constexpr double kLower = -1.25;
  constexpr double kUpper = 1.25;

  std::vector<double> data(kRows);
  for (int i = 0; i < kRows; ++i) {
    data[i] = static_cast<double>((i % 21) - 10) / 4;
  }

  auto input = makeRowVector(
      {makeFlatVector<double>(kRows, [&](auto i) { return data[i]; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::FloatingPointRange<double>>(
          kLower, false, false, kUpper, false, false, false));
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<FloatingPointColumnReaderTestAccessor<double>*>(
      dynamic_cast<FloatingPointColumnReader<double, double>*>(
          structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec;
  for (int i = 0; i < kRows; i += 3) {
    rowVec.push_back(i);
  }
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead(0, rows, nullptr);

  std::vector<std::optional<const EncodingLayout>> children;
  children.emplace_back(TrivialEnc{});
  const EncodingLayout alpLayout(
      EncodingType::ALP,
      {},
      CompressionType::Uncompressed,
      std::move(children));
  Buffer buffer(*pool());
  auto encoding =
      createFromCustomLayout<double>(alpLayout, data, *pool(), buffer);
  ASSERT_EQ(encoding->encodingType(), EncodingType::ALP);

  common::FloatingPointRange<double> filter(
      kLower, false, false, kUpper, false, false, false);
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = false;
  DecoderVisitor<
      double,
      common::FloatingPointRange<double>,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  int expected = 0;
  for (const auto row : rowVec) {
    if (data[row] >= kLower && data[row] <= kUpper) {
      ++expected;
    }
  }
  EXPECT_EQ(reader->numValues(), expected);

  auto values = getValues<double>(reader);
  int outputIndex = 0;
  for (const auto row : rowVec) {
    if (data[row] >= kLower && data[row] <= kUpper) {
      EXPECT_DOUBLE_EQ(values[outputIndex], data[row]);
      ++outputIndex;
    }
  }
}

// ---------------------------------------------------------------------------
// 4. RLEEncoding<int64_t> + AlwaysTrue + dense
// ---------------------------------------------------------------------------
TEST_P(ReadWithVisitorTest, encodingLevelRleAlwaysTrueDense) {
  constexpr int kRows = 500;

  // Repeated-run data: 50 copies of each value [0..9].
  std::vector<int64_t> data(kRows);
  for (int i = 0; i < kRows; ++i) {
    data[i] = i / 50;
  }

  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kRows, [](auto i) { return i / 50; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding =
      createFromCustomLayout<int64_t>(RLEEnc{}, data, *pool(), buffer);

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int64_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  EXPECT_EQ(reader->numValues(), kRows);
  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(values[i], i / 50) << "row " << i;
  }
}

// ---------------------------------------------------------------------------
// 5. TrivialEncoding
// 5. TrivialEncoding<int64_t> + AlwaysTrue + dense + SLOW PATH
//    hasBulkPath=false forces readWithVisitorSlow regardless of CPU features.
// ---------------------------------------------------------------------------
TEST_P(ReadWithVisitorTest, encodingLevelTrivialAlwaysTrueDenseSlowPath) {
  constexpr int kRows = 500;

  std::vector<int64_t> data(kRows);
  std::iota(data.begin(), data.end(), 0);

  auto input = makeRowVector({makeFlatVector<int64_t>(kRows, folly::identity)});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding =
      createFromCustomLayout<int64_t>(TrivialEnc{}, data, *pool(), buffer);

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  constexpr bool kHasBulkPath = false;
  velox::dwio::common::ColumnVisitor<
      int64_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense,
      kHasBulkPath>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  EXPECT_EQ(reader->numValues(), kRows);
  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(values[i], i);
  }
}

// ---------------------------------------------------------------------------
// 6. RLEEncoding<int64_t> + AlwaysTrue + dense + SLOW PATH
// ---------------------------------------------------------------------------
TEST_P(ReadWithVisitorTest, encodingLevelRleAlwaysTrueDenseSlowPath) {
  constexpr int kRows = 500;

  std::vector<int64_t> data(kRows);
  for (int i = 0; i < kRows; ++i) {
    data[i] = i / 50;
  }

  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kRows, [](auto i) { return i / 50; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding =
      createFromCustomLayout<int64_t>(RLEEnc{}, data, *pool(), buffer);

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  constexpr bool kHasBulkPath = false;
  velox::dwio::common::ColumnVisitor<
      int64_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense,
      kHasBulkPath>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  EXPECT_EQ(reader->numValues(), kRows);
  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(values[i], i / 50) << "row " << i;
  }
}

// ---------------------------------------------------------------------------
// 7. MainlyConstantEncoding<int64_t> + AlwaysTrue + dense (FAST path)
// ---------------------------------------------------------------------------
TEST_P(ReadWithVisitorTest, encodingLevelMainlyConstantAlwaysTrueDense) {
  constexpr int kRows = 500;

  std::vector<int64_t> data(kRows, 42);
  data[100] = 7;
  data[200] = 13;
  data[300] = 19;
  data[400] = 23;
  data[450] = 29;

  auto input = makeRowVector({makeFlatVector<int64_t>(data)});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int64_t>(
      MainlyConstantEnc{}, data, *pool(), buffer);

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int64_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  EXPECT_EQ(reader->numValues(), kRows);
  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(values[i], data[i]) << "row " << i;
  }
}

// ---------------------------------------------------------------------------
// 8. MainlyConstantEncoding
// 8. MainlyConstantEncoding<int64_t> + AlwaysTrue + dense + SLOW PATH
// ---------------------------------------------------------------------------
TEST_P(
    ReadWithVisitorTest,
    encodingLevelMainlyConstantAlwaysTrueDenseSlowPath) {
  constexpr int kRows = 500;

  std::vector<int64_t> data(kRows, 42);
  data[100] = 7;
  data[200] = 13;
  data[300] = 19;
  data[400] = 23;
  data[450] = 29;

  auto input = makeRowVector({makeFlatVector<int64_t>(data)});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int64_t>(
      MainlyConstantEnc{}, data, *pool(), buffer);

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  constexpr bool kHasBulkPath = false;
  velox::dwio::common::ColumnVisitor<
      int64_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense,
      kHasBulkPath>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  EXPECT_EQ(reader->numValues(), kRows);
  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(values[i], data[i]) << "row " << i;
  }
}

// ---------------------------------------------------------------------------
// 9. DictionaryEncoding
// 9. DictionaryEncoding<int64_t> + AlwaysTrue + dense
//    Dictionary always uses slow path (no bulkScan).
// ---------------------------------------------------------------------------
TEST_P(ReadWithVisitorTest, encodingLevelDictionaryAlwaysTrueDense) {
  constexpr int kRows = 500;

  std::vector<int64_t> data(kRows);
  for (int i = 0; i < kRows; ++i) {
    data[i] = (i % 5) * 10;
  }

  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kRows, [](auto i) { return (i % 5) * 10; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding =
      createFromCustomLayout<int64_t>(DictionaryEnc{}, data, *pool(), buffer);

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int64_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  EXPECT_EQ(reader->numValues(), kRows);
  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(values[i], (i % 5) * 10) << "row " << i;
  }
}

TEST_P(ReadWithVisitorTest, encodingLevelSharedDictionaryAlwaysTrueDense) {
  if (!useNonLegacy()) {
    GTEST_SKIP() << "SharedDictionaryEncoding requires non-legacy encoding";
  }

  const std::vector<int32_t> alphabet{10, 20, 30, 40};
  const std::vector<uint32_t> indices{2, 0, 3, 1, 2};
  const std::vector<int32_t> expected{30, 10, 40, 20, 30};
  const auto kRows = static_cast<int>(indices.size());

  auto input = makeRowVector({makeFlatVector<int32_t>(
      kRows, [&](auto row) { return expected[row]; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int32_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  const auto encoded = test::encodeSharedDictionary(buffer, indices);
  Encoding::Options options;
  options.sharedDictionaryAlphabet =
      test::createSharedDictionaryAlphabet<int32_t>(
          alphabet, /*candidateEncodings=*/{}, pool());
  SharedDictionaryEncoding<int32_t> encoding{
      *pool(), encoded, [](uint32_t) { return nullptr; }, options};

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int32_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  callReadWithVisitor(encoding, visitor, params);

  EXPECT_EQ(reader->numValues(), kRows);
  EXPECT_EQ(getValues<int32_t>(reader), expected);
}

// ---------------------------------------------------------------------------
// 10. DictionaryEncoding<int64_t> + BigintRange + sparse
// ---------------------------------------------------------------------------
TEST_P(ReadWithVisitorTest, encodingLevelDictionaryBigintRangeSparse) {
  constexpr int kRows = 500;

  std::vector<int64_t> data(kRows);
  for (int i = 0; i < kRows; ++i) {
    data[i] = (i % 5) * 10;
  }

  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kRows, [](auto i) { return (i % 5) * 10; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(15, 35, false));
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec;
  for (int i = 0; i < kRows; i += 2) {
    rowVec.push_back(i);
  }
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding =
      createFromCustomLayout<int64_t>(DictionaryEnc{}, data, *pool(), buffer);

  common::BigintRange filter(15, 35, false);
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = false;
  DecoderVisitor<
      int64_t,
      common::BigintRange,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  int expected = 0;
  for (int i = 0; i < kRows; i += 2) {
    if (data[i] >= 15 && data[i] <= 35) {
      ++expected;
    }
  }
  EXPECT_EQ(reader->numValues(), expected);

  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < reader->numValues(); ++i) {
    EXPECT_GE(values[i], 15);
    EXPECT_LE(values[i], 35);
  }
}

// ---------------------------------------------------------------------------
// 11. MainlyConstant<Dictionary> + AlwaysTrue + dense (FAST path, composite)
// ---------------------------------------------------------------------------
TEST_P(
    ReadWithVisitorTest,
    encodingLevelMainlyConstantDictionaryAlwaysTrueDense) {
  constexpr int kRows = 500;

  // Mostly 42; non-common values cycle through {100, 101, 102}.
  std::vector<int64_t> data(kRows, 42);
  for (int i = 0; i < kRows; i += 10) {
    data[i] = 100 + (i / 10) % 3;
  }

  auto input = makeRowVector({makeFlatVector<int64_t>(data)});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int64_t>(
      MainlyConstantEnc{.otherValues = DictionaryEnc{}}, data, *pool(), buffer);

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int64_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  EXPECT_EQ(reader->numValues(), kRows);
  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(values[i], data[i]) << "row " << i;
  }
}

// ---------------------------------------------------------------------------
// 12. MainlyConstant<Dictionary> + AlwaysTrue + dense + SLOW PATH (composite)
// ---------------------------------------------------------------------------
TEST_P(
    ReadWithVisitorTest,
    encodingLevelMainlyConstantDictionaryAlwaysTrueDenseSlowPath) {
  constexpr int kRows = 500;

  std::vector<int64_t> data(kRows, 42);
  for (int i = 0; i < kRows; i += 10) {
    data[i] = 100 + (i / 10) % 3;
  }

  auto input = makeRowVector({makeFlatVector<int64_t>(data)});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int64_t>(
      MainlyConstantEnc{.otherValues = DictionaryEnc{}}, data, *pool(), buffer);

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  constexpr bool kHasBulkPath = false;
  velox::dwio::common::ColumnVisitor<
      int64_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense,
      kHasBulkPath>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  EXPECT_EQ(reader->numValues(), kRows);
  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(values[i], data[i]) << "row " << i;
  }
}

// ---------------------------------------------------------------------------
// 13. FixedBitWidthEncoding<int32_t> + AlwaysTrue + dense + FAST PATH
//     The FixedBitWidth fast path (bulkScan) requires:
//       - 4-byte or 8-byte integral physical type
//       - No filter (!kHasFilter → AlwaysTrue)
//       - No hook (!kHasHook → ExtractToReader)
//       - kExtractToReader && (kSameType || kIsWidening)
//     This test uses int32_t data to exercise the 4-byte path.
// ---------------------------------------------------------------------------
TEST_P(
    ReadWithVisitorTest,
    encodingLevelFixedBitWidthAlwaysTrueDenseInt32FastPath) {
  constexpr int kRows = 500;

  // Small-range int32 data that fits in fixed-bit-width encoding.
  std::vector<int32_t> data(kRows);
  for (int i = 0; i < kRows; ++i) {
    data[i] = i % 16;
  }

  // Write a nimble file with an int32 column (INTEGER type).
  auto input = makeRowVector({makeFlatVector<int32_t>(
      kRows, [](auto i) { return static_cast<int32_t>(i % 16); })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  // prepareRead with int32_t — must match the column's physical type.
  reader->doPrepareRead<int32_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int32_t>(
      FixedBitWidthEnc{}, data, *pool(), buffer);

  // Visitor DataType = int32_t → dispatch matches
  // FixedBitWidthEncoding<int32_t>. AlwaysTrue + ExtractToReader +
  // hasBulkPath=true (default) → fast path.
  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int32_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  EXPECT_EQ(reader->numValues(), kRows);
  auto values = getValues<int32_t>(reader);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(values[i], i % 16) << "row " << i;
  }
}

// ---------------------------------------------------------------------------
// 13a. FixedBitWidthEncoding<int64_t> + AlwaysTrue + dense + FAST PATH
//      Exercises the 8-byte bulkScan path, which decodes through
//      FixedBitArray::bulkGet64WithBaseline.
// ---------------------------------------------------------------------------
TEST_P(
    ReadWithVisitorTest,
    encodingLevelFixedBitWidthAlwaysTrueDenseInt64FastPath) {
  constexpr int kRows = 500;
  constexpr int64_t kBaseline = 1000;
  constexpr int64_t kLargeResidual = (int64_t{1} << 60) - 1;

  std::vector<int64_t> data(kRows);
  for (int i = 0; i < kRows; ++i) {
    int64_t residual = 0;
    switch (i % 4) {
      case 0:
        residual = 0;
        break;
      case 1:
        residual = 17;
        break;
      case 2:
        residual = int64_t{1} << 59;
        break;
      default:
        residual = kLargeResidual - (i % 17);
        break;
    }
    data[i] = kBaseline + residual;
  }

  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kRows, [&data](auto i) { return data[i]; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int64_t>(
      FixedBitWidthEnc{}, data, *pool(), buffer);

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int64_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  EXPECT_EQ(reader->numValues(), kRows);
  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(values[i], data[i]) << "row " << i;
  }
}

// ---------------------------------------------------------------------------
// 13b. FixedBitWidthEncoding<int32_t> + AlwaysTrue + dense + FAST PATH + NULLS
//      Tests the scatter path in bulkScan when encoding-level nulls are present
//      (as set by NullableEncoding). Non-null values must land at their correct
//      scattered positions in the output buffer.
// ---------------------------------------------------------------------------
TEST_P(
    ReadWithVisitorTest,
    encodingLevelFixedBitWidthAlwaysTrueDenseInt32FastPathWithNulls) {
  constexpr int kRows = 500;

  // Build null bitmap: every 5th row is null.
  auto nulls = velox::allocateNulls(kRows, pool(), velox::bits::kNotNull);
  auto* rawNulls = nulls->asMutable<uint64_t>();
  int expectedNonNull = 0;
  for (int i = 0; i < kRows; ++i) {
    if (i % 5 == 0) {
      velox::bits::setNull(rawNulls, i);
    } else {
      ++expectedNonNull;
    }
  }

  // Small-range int32 data for all rows (non-null values use i % 16).
  std::vector<int32_t> nonNullData(expectedNonNull);
  {
    int j = 0;
    for (int i = 0; i < kRows; ++i) {
      if (i % 5 != 0) {
        nonNullData[j++] = i % 16;
      }
    }
  }

  auto input = makeRowVector({makeFlatVector<int32_t>(
      kRows, [](auto i) { return static_cast<int32_t>(i % 16); })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int32_t>(0, rows, nullptr);

  // Set encoding-level nulls (simulates NullableEncoding wrapping FBW).
  reader->nullsInReadRange() = nulls;

  // Create FBW encoding with only the non-null values.
  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int32_t>(
      FixedBitWidthEnc{}, nonNullData, *pool(), buffer);

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int32_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  // With nulls, numValues should be kRows (non-null values scattered to their
  // correct positions, null positions left untouched).
  EXPECT_EQ(reader->numValues(), kRows);
  auto values = getValues<int32_t>(reader);
  for (int i = 0; i < kRows; ++i) {
    if (i % 5 != 0) {
      EXPECT_EQ(values[i], i % 16) << "non-null row " << i;
    }
  }
}

// ---------------------------------------------------------------------------
// 13c. FixedBitWidthEncoding<int32_t> + AlwaysTrue + sparse + FAST PATH
//      Exercises the sparse branch of bulkScan (individual value reads at
//      specified positions) followed by processFixedWidthRun. A subset of rows
//      is selected to make the read non-dense.
// ---------------------------------------------------------------------------
TEST_P(
    ReadWithVisitorTest,
    encodingLevelFixedBitWidthAlwaysTrueSparseInt32FastPath) {
  constexpr int kTotalRows = 500;

  // Small-range int32 data.
  std::vector<int32_t> data(kTotalRows);
  for (int i = 0; i < kTotalRows; ++i) {
    data[i] = i % 16;
  }

  auto input = makeRowVector({makeFlatVector<int32_t>(
      kTotalRows, [](auto i) { return static_cast<int32_t>(i % 16); })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  // Select every other row to make the read sparse.
  std::vector<vector_size_t> rowVec;
  for (int i = 0; i < kTotalRows; i += 2) {
    rowVec.push_back(i);
  }
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int32_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int32_t>(
      FixedBitWidthEnc{}, data, *pool(), buffer);

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = false;
  DecoderVisitor<
      int32_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  auto numSelected = static_cast<int>(rowVec.size());
  EXPECT_EQ(reader->numValues(), numSelected);
  auto values = getValues<int32_t>(reader);
  for (int i = 0; i < numSelected; ++i) {
    EXPECT_EQ(values[i], rowVec[i] % 16) << "selected row index " << i;
  }
}

// ---------------------------------------------------------------------------
// 13d. FixedBitWidthEncoding<int32_t> + BigintRange + dense + FAST PATH
//      Exercises the filter path in bulkScan: only values passing BigintRange
//      [4, 12] should appear in the output.
// ---------------------------------------------------------------------------
TEST_P(
    ReadWithVisitorTest,
    encodingLevelFixedBitWidthBigintRangeDenseInt32FastPath) {
  constexpr int kRows = 500;

  std::vector<int32_t> data(kRows);
  for (int i = 0; i < kRows; ++i) {
    data[i] = i % 16;
  }

  auto input = makeRowVector({makeFlatVector<int32_t>(
      kRows, [](auto i) { return static_cast<int32_t>(i % 16); })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(4, 12, false));
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int32_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int32_t>(
      FixedBitWidthEnc{}, data, *pool(), buffer);

  common::BigintRange filter(4, 12, false);
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int32_t,
      common::BigintRange,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  int expected = 0;
  for (int i = 0; i < kRows; ++i) {
    if (data[i] >= 4 && data[i] <= 12) {
      ++expected;
    }
  }
  EXPECT_EQ(reader->numValues(), expected);

  auto values = getValues<int32_t>(reader);
  for (int i = 0; i < reader->numValues(); ++i) {
    EXPECT_GE(values[i], 4);
    EXPECT_LE(values[i], 12);
  }
}

// ---------------------------------------------------------------------------
// 13e. FixedBitWidthEncoding<int32_t> + BigintRange + sparse + FAST PATH
//      Exercises the filter path in bulkScan with sparse (non-dense) row
//      selection: every other row is selected, then filtered by [4, 12].
// ---------------------------------------------------------------------------
TEST_P(
    ReadWithVisitorTest,
    encodingLevelFixedBitWidthBigintRangeSparseInt32FastPath) {
  constexpr int kTotalRows = 500;

  std::vector<int32_t> data(kTotalRows);
  for (int i = 0; i < kTotalRows; ++i) {
    data[i] = i % 16;
  }

  auto input = makeRowVector({makeFlatVector<int32_t>(
      kTotalRows, [](auto i) { return static_cast<int32_t>(i % 16); })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(4, 12, false));
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec;
  for (int i = 0; i < kTotalRows; i += 2) {
    rowVec.push_back(i);
  }
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int32_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int32_t>(
      FixedBitWidthEnc{}, data, *pool(), buffer);

  common::BigintRange filter(4, 12, false);
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = false;
  DecoderVisitor<
      int32_t,
      common::BigintRange,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  int expected = 0;
  for (int i = 0; i < kTotalRows; i += 2) {
    if (data[i] >= 4 && data[i] <= 12) {
      ++expected;
    }
  }
  EXPECT_EQ(reader->numValues(), expected);

  auto values = getValues<int32_t>(reader);
  for (int i = 0; i < reader->numValues(); ++i) {
    EXPECT_GE(values[i], 4);
    EXPECT_LE(values[i], 12);
  }
}

// ---------------------------------------------------------------------------
// 13f. FixedBitWidthEncoding<int32_t> + BigintRange + dense + FAST PATH + NULLS
//      Exercises the filter + scatter combination in bulkScan: encoding-level
//      nulls trigger scatter (kScatter=true), and the BigintRange filter
//      selects only values in [4, 12]. Both processFixedWidthRun concerns
//      are exercised simultaneously.
// ---------------------------------------------------------------------------
TEST_P(
    ReadWithVisitorTest,
    encodingLevelFixedBitWidthBigintRangeDenseInt32FastPathWithNulls) {
  constexpr int kRows = 500;

  // Build null bitmap: every 5th row is null.
  auto nulls = velox::allocateNulls(kRows, pool(), velox::bits::kNotNull);
  auto* rawNulls = nulls->asMutable<uint64_t>();
  int expectedNonNull = 0;
  for (int i = 0; i < kRows; ++i) {
    if (i % 5 == 0) {
      velox::bits::setNull(rawNulls, i);
    } else {
      ++expectedNonNull;
    }
  }

  // Non-null values only (encoding stores only non-null values).
  std::vector<int32_t> nonNullData(expectedNonNull);
  {
    int j = 0;
    for (int i = 0; i < kRows; ++i) {
      if (i % 5 != 0) {
        nonNullData[j++] = i % 16;
      }
    }
  }

  auto input = makeRowVector({makeFlatVector<int32_t>(
      kRows, [](auto i) { return static_cast<int32_t>(i % 16); })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(4, 12, false));
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int32_t>(0, rows, nullptr);

  // Set encoding-level nulls (simulates NullableEncoding wrapping FBW).
  reader->nullsInReadRange() = nulls;

  // Create FBW encoding with only the non-null values.
  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int32_t>(
      FixedBitWidthEnc{}, nonNullData, *pool(), buffer);

  common::BigintRange filter(4, 12, false);
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int32_t,
      common::BigintRange,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  // Count expected: non-null rows whose value is in [4, 12].
  int expected = 0;
  for (int i = 0; i < kRows; ++i) {
    if (i % 5 != 0 && (i % 16) >= 4 && (i % 16) <= 12) {
      ++expected;
    }
  }
  EXPECT_EQ(reader->numValues(), expected);

  auto values = getValues<int32_t>(reader);
  for (int i = 0; i < reader->numValues(); ++i) {
    EXPECT_GE(values[i], 4);
    EXPECT_LE(values[i], 12);
  }
}

// ---------------------------------------------------------------------------
// 14. DeltaEncoding<int64_t> + AlwaysTrue + dense
//     Delta always uses slow path (no bulkScan).
// ---------------------------------------------------------------------------
TEST_P(ReadWithVisitorTest, encodingLevelDeltaAlwaysTrueDense) {
  constexpr int kRows = 500;

  // Monotonically increasing data suitable for delta encoding.
  std::vector<int64_t> data(kRows);
  for (int i = 0; i < kRows; ++i) {
    data[i] = i * 3;
  }

  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kRows, [](auto i) { return i * 3; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding =
      createFromCustomLayout<int64_t>(DeltaEnc{}, data, *pool(), buffer);

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int64_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  EXPECT_EQ(reader->numValues(), kRows);
  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(values[i], i * 3) << "row " << i;
  }
}

// ---------------------------------------------------------------------------
// 15. DeltaEncoding<int64_t> + BigintRange + sparse
// ---------------------------------------------------------------------------
TEST_P(ReadWithVisitorTest, encodingLevelDeltaBigintRangeSparse) {
  constexpr int kRows = 500;

  // Monotonically increasing data with occasional restatements.
  std::vector<int64_t> data(kRows);
  for (int i = 0; i < kRows; ++i) {
    // Every 50 rows, reset to a lower value to trigger restatements.
    data[i] = (i % 50) * 2 + (i / 50) * 10;
  }

  auto input = makeRowVector({makeFlatVector<int64_t>(data)});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(20, 60, false));
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  // Sparse RowSet: every other row.
  std::vector<vector_size_t> rowVec;
  for (int i = 0; i < kRows; i += 2) {
    rowVec.push_back(i);
  }
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding =
      createFromCustomLayout<int64_t>(DeltaEnc{}, data, *pool(), buffer);

  common::BigintRange filter(20, 60, false);
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = false;
  DecoderVisitor<
      int64_t,
      common::BigintRange,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  int expected = 0;
  for (int i = 0; i < kRows; i += 2) {
    if (data[i] >= 20 && data[i] <= 60) {
      ++expected;
    }
  }
  EXPECT_EQ(reader->numValues(), expected);

  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < reader->numValues(); ++i) {
    EXPECT_GE(values[i], 20);
    EXPECT_LE(values[i], 60);
  }
}

// ---------------------------------------------------------------------------
// 16. DeltaEncoding<int64_t> + AlwaysTrue + dense + SLOW PATH
// ---------------------------------------------------------------------------
TEST_P(ReadWithVisitorTest, encodingLevelDeltaAlwaysTrueDenseSlowPath) {
  constexpr int kRows = 500;

  std::vector<int64_t> data(kRows);
  for (int i = 0; i < kRows; ++i) {
    data[i] = i * 3;
  }

  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kRows, [](auto i) { return i * 3; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding =
      createFromCustomLayout<int64_t>(DeltaEnc{}, data, *pool(), buffer);

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  constexpr bool kHasBulkPath = false;
  velox::dwio::common::ColumnVisitor<
      int64_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense,
      kHasBulkPath>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  EXPECT_EQ(reader->numValues(), kRows);
  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(values[i], i * 3) << "row " << i;
  }
}

// ===========================================================================
// Test: readIndicesWithVisitor at the encoding level.
TEST_P(ReadWithVisitorNonLegacyTest, readIndicesWithVisitorString) {
  constexpr int kRows = 50;
  constexpr std::string_view kAlphabet[] = {"alpha", "bravo", "charlie"};

  // Write the same data to get reader infrastructure.
  auto input = makeRowVector({makeFlatVector<StringView>(
      kRows, [&](auto i) { return StringView(kAlphabet[i % 3]); })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader =
      static_cast<StringColumnReaderTestAccessor*>(structReader->children()[0]);
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());
  reader->doPrepareReadAsIndices(0, rows, nullptr);

  // Encode the same data as DictionaryEncoding independently.
  nimble::Vector<std::string_view> vec{pool()};
  for (int i = 0; i < kRows; ++i) {
    vec.push_back(kAlphabet[i % 3]);
  }
  auto span = std::span<const std::string_view>(vec.data(), vec.size());
  nimble::EncodingSelection<std::string_view> selection{
      {.encodingType = nimble::EncodingType::Dictionary},
      nimble::Statistics<std::string_view>::create(span),
      std::make_unique<TrivialNestedPolicy<std::string_view>>()};
  Buffer encBuffer(*pool());
  auto encoded = nimble::DictionaryEncoding<std::string_view>::encode(
      selection, span, encBuffer);
  std::vector<velox::BufferPtr> stringBuffers;
  auto encoding =
      std::make_unique<nimble::DictionaryEncoding<std::string_view>>(
          *pool(), encoded, [&](uint32_t size) {
            stringBuffers.push_back(
                velox::AlignedBuffer::allocate<char>(size, pool()));
            return stringBuffers.back()->asMutable<void>();
          });

  ASSERT_TRUE(encoding->dictionaryEnabled());
  const auto alphabetSize = encoding->dictionarySize();
  const auto* alphabet =
      static_cast<const std::string_view*>(encoding->dictionaryEntries());

  // Build visitor that writes int32 indices to the reader's rawValues_.
  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  DecoderVisitor<
      int32_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      /*isDense=*/true>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  encoding->readIndicesWithVisitor(visitor, params);

  ASSERT_EQ(reader->numValues(), kRows);
  const auto* indices = reader->rawIndices();
  for (int i = 0; i < kRows; ++i) {
    ASSERT_GE(indices[i], 0) << "row " << i;
    ASSERT_LT(indices[i], static_cast<int32_t>(alphabetSize)) << "row " << i;
    EXPECT_EQ(alphabet[indices[i]], kAlphabet[i % 3]) << "row " << i;
  }
}

// Test: Fuzz readIndicesWithVisitor at the encoding level.
// Random string data, verifies indices map to correct alphabet entries.
TEST_P(ReadWithVisitorNonLegacyTest, fuzzReadIndicesWithVisitorString) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  constexpr int kNumRuns = 10;

  for (int run = 0; run < kNumRuns; ++run) {
    const int numRows = 20 + rng() % 100;
    const int cardinality = 2 + rng() % 8;

    SCOPED_TRACE(fmt::format("run={} seed={} numRows={}", run, seed, numRows));

    // Generate random alphabet.
    std::vector<std::string> pool;
    for (int j = 0; j < cardinality; ++j) {
      int len = rng() % 20;
      std::string s = fmt::format("{}:", j);
      for (int k = 0; k < len; ++k) {
        s += static_cast<char>('a' + rng() % 26);
      }
      pool.push_back(std::move(s));
    }

    // Generate data sampling from the pool.
    std::vector<std::string_view> data;
    data.reserve(numRows);
    for (int i = 0; i < numRows; ++i) {
      data.push_back(pool[rng() % pool.size()]);
    }

    // Build reader infrastructure from the same data.
    auto input = makeRowVector({makeFlatVector<StringView>(
        numRows, [&](auto i) { return StringView(data[i]); })});
    auto rowType = asRowType(input->type());
    auto ctx = makeFileContext(input);
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*rowType);
    auto root = buildReader(*ctx, rowType, *scanSpec);

    auto* structReader =
        dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(
            root.get());
    auto* reader = static_cast<StringColumnReaderTestAccessor*>(
        structReader->children()[0]);

    std::vector<vector_size_t> rowVec(numRows);
    std::iota(rowVec.begin(), rowVec.end(), 0);
    RowSet rows(rowVec.data(), rowVec.size());
    reader->doPrepareReadAsIndices(0, rows, nullptr);

    // Encode independently as DictionaryEncoding.
    nimble::Vector<std::string_view> vec{this->pool()};
    for (auto& d : data) {
      vec.push_back(d);
    }
    auto span = std::span<const std::string_view>(vec.data(), vec.size());
    nimble::EncodingSelection<std::string_view> selection{
        {.encodingType = nimble::EncodingType::Dictionary},
        nimble::Statistics<std::string_view>::create(span),
        std::make_unique<TrivialNestedPolicy<std::string_view>>()};
    Buffer encBuffer(*this->pool());
    auto encoded = nimble::DictionaryEncoding<std::string_view>::encode(
        selection, span, encBuffer);
    std::vector<velox::BufferPtr> stringBuffers;
    auto encoding =
        std::make_unique<nimble::DictionaryEncoding<std::string_view>>(
            *this->pool(), encoded, [&](uint32_t size) {
              stringBuffers.push_back(
                  velox::AlignedBuffer::allocate<char>(size, this->pool()));
              return stringBuffers.back()->asMutable<void>();
            });

    const auto alphabetSize = encoding->dictionarySize();
    const auto* alphabet =
        static_cast<const std::string_view*>(encoding->dictionaryEntries());

    common::AlwaysTrue filter;
    dwio::common::ExtractToReader extractValues(reader);
    DecoderVisitor<
        int32_t,
        common::AlwaysTrue,
        dwio::common::ExtractToReader,
        /*isDense=*/true>
        visitor(filter, reader, rows, extractValues);
    auto params = makeReadWithVisitorParams(visitor, rows, this->pool());

    encoding->readIndicesWithVisitor(visitor, params);

    ASSERT_EQ(reader->numValues(), numRows);
    const auto* indices = reader->rawIndices();
    for (int i = 0; i < numRows; ++i) {
      ASSERT_GE(indices[i], 0) << "row " << i;
      ASSERT_LT(indices[i], static_cast<int32_t>(alphabetSize)) << "row " << i;
      EXPECT_EQ(alphabet[indices[i]], data[i]) << "row " << i;
    }
  }
}

// ===========================================================================
// Test: readIndicesWithVisitor on integer DictionaryEncoding (multiple types).
// ===========================================================================
TEST_P(ReadWithVisitorNonLegacyTest, readIndicesWithVisitorInteger) {
  auto runForType = [&]<typename T>(T) {
    SCOPED_TRACE(fmt::format("type size = {}", sizeof(T)));
    constexpr int kRows = 50;
    const T kAlphabet[] = {T(10), T(20), T(30)};

    auto input = makeRowVector(
        {makeFlatVector<T>(kRows, [&](auto i) { return kAlphabet[i % 3]; })});
    auto rowType = asRowType(input->type());
    auto ctx = makeFileContext(input);
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*rowType);
    auto root = buildReader(*ctx, rowType, *scanSpec);

    auto* structReader =
        dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(
            root.get());
    auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
        dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
    ASSERT_NE(reader, nullptr);

    std::vector<vector_size_t> rowVec(kRows);
    std::iota(rowVec.begin(), rowVec.end(), 0);
    RowSet rows(rowVec.data(), rowVec.size());
    reader->doPrepareRead<int32_t>(0, rows, nullptr);

    using PhysicalType = nimble::TypeTraits<T>::physicalType;
    nimble::Vector<T> vec{pool()};
    for (int i = 0; i < kRows; ++i) {
      vec.push_back(kAlphabet[i % 3]);
    }
    auto span = std::span<const PhysicalType>(
        reinterpret_cast<const PhysicalType*>(vec.data()), vec.size());
    nimble::EncodingSelection<PhysicalType> selection{
        {.encodingType = nimble::EncodingType::Dictionary},
        nimble::Statistics<PhysicalType>::create(span),
        std::make_unique<TrivialNestedPolicy<T>>()};
    Buffer encBuffer(*pool());
    auto encoded =
        nimble::DictionaryEncoding<T>::encode(selection, span, encBuffer);
    auto encoding = std::make_unique<nimble::DictionaryEncoding<T>>(
        *pool(), encoded, nullptr);

    ASSERT_TRUE(encoding->dictionaryEnabled());
    const auto alphabetSize = encoding->dictionarySize();
    const auto* alphabet = static_cast<const T*>(encoding->dictionaryEntries());

    common::AlwaysTrue filter;
    dwio::common::ExtractToReader extractValues(reader);
    DecoderVisitor<
        int32_t,
        common::AlwaysTrue,
        dwio::common::ExtractToReader,
        /*isDense=*/true>
        visitor(filter, reader, rows, extractValues);
    auto params = makeReadWithVisitorParams(visitor, rows, pool());

    encoding->readIndicesWithVisitor(visitor, params);

    ASSERT_EQ(reader->numValues(), kRows);
    const auto* indices = reader->rawIndices();
    for (int i = 0; i < kRows; ++i) {
      ASSERT_GE(indices[i], 0) << "row " << i;
      ASSERT_LT(indices[i], static_cast<int32_t>(alphabetSize)) << "row " << i;
      EXPECT_EQ(alphabet[indices[i]], kAlphabet[i % 3]) << "row " << i;
    }
  };

  runForType(int16_t{});
  runForType(int32_t{});
  runForType(int64_t{});
}

// ===========================================================================
// Test: Fuzz readIndicesWithVisitor on integer DictionaryEncoding.
// ===========================================================================
TEST_P(ReadWithVisitorNonLegacyTest, fuzzReadIndicesWithVisitorInteger) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  constexpr int kNumRuns = 10;

  auto runForType = [&]<typename T>(T, int run) {
    const int numRows = 20 + rng() % 100;
    const int cardinality = 2 + rng() % 8;

    SCOPED_TRACE(
        fmt::format(
            "run={} seed={} numRows={} typeSize={}",
            run,
            seed,
            numRows,
            sizeof(T)));

    std::vector<T> alphabet;
    for (int j = 0; j < cardinality; ++j) {
      alphabet.push_back(static_cast<T>(j * 7 + 1));
    }

    std::vector<T> data;
    data.reserve(numRows);
    for (int i = 0; i < numRows; ++i) {
      data.push_back(alphabet[rng() % alphabet.size()]);
    }

    auto input = makeRowVector(
        {makeFlatVector<T>(numRows, [&](auto i) { return data[i]; })});
    auto rowType = asRowType(input->type());
    auto ctx = makeFileContext(input);
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*rowType);
    auto root = buildReader(*ctx, rowType, *scanSpec);

    auto* structReader =
        dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(
            root.get());
    auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
        dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
    ASSERT_NE(reader, nullptr);

    std::vector<vector_size_t> rowVec(numRows);
    std::iota(rowVec.begin(), rowVec.end(), 0);
    RowSet rows(rowVec.data(), rowVec.size());
    reader->doPrepareRead<int32_t>(0, rows, nullptr);

    using PhysicalType = nimble::TypeTraits<T>::physicalType;
    nimble::Vector<T> vec{this->pool()};
    for (auto& d : data) {
      vec.push_back(d);
    }
    auto span = std::span<const PhysicalType>(
        reinterpret_cast<const PhysicalType*>(vec.data()), vec.size());
    nimble::EncodingSelection<PhysicalType> selection{
        {.encodingType = nimble::EncodingType::Dictionary},
        nimble::Statistics<PhysicalType>::create(span),
        std::make_unique<TrivialNestedPolicy<T>>()};
    Buffer encBuffer(*this->pool());
    auto encoded =
        nimble::DictionaryEncoding<T>::encode(selection, span, encBuffer);
    auto encoding = std::make_unique<nimble::DictionaryEncoding<T>>(
        *this->pool(), encoded, nullptr);

    ASSERT_TRUE(encoding->dictionaryEnabled());
    const auto alphabetSize = encoding->dictionarySize();
    const auto* entries = static_cast<const T*>(encoding->dictionaryEntries());

    common::AlwaysTrue filter;
    dwio::common::ExtractToReader extractValues(reader);
    DecoderVisitor<
        int32_t,
        common::AlwaysTrue,
        dwio::common::ExtractToReader,
        /*isDense=*/true>
        visitor(filter, reader, rows, extractValues);
    auto params = makeReadWithVisitorParams(visitor, rows, this->pool());

    encoding->readIndicesWithVisitor(visitor, params);

    ASSERT_EQ(reader->numValues(), numRows);
    const auto* indices = reader->rawIndices();
    for (int i = 0; i < numRows; ++i) {
      ASSERT_GE(indices[i], 0) << "row " << i;
      ASSERT_LT(indices[i], static_cast<int32_t>(alphabetSize)) << "row " << i;
      EXPECT_EQ(entries[indices[i]], data[i]) << "row " << i;
    }
  };

  for (int run = 0; run < kNumRuns; ++run) {
    switch (run % 3) {
      case 0:
        runForType(int16_t{}, run);
        break;
      case 1:
        runForType(int32_t{}, run);
        break;
      case 2:
        runForType(int64_t{}, run);
        break;
    }
  }
}

// ===========================================================================
// Test: readIndicesWithVisitor on NullableEncoding<DictionaryEncoding<string>>.
// ===========================================================================
TEST_P(ReadWithVisitorNonLegacyTest, readIndicesWithVisitorNullable) {
  constexpr int kRows = 50;
  constexpr std::string_view kAlphabet[] = {"alpha", "bravo", "charlie"};

  auto input = makeRowVector({makeFlatVector<StringView>(
      kRows,
      [&](auto i) { return StringView(kAlphabet[i % 3]); },
      nullEvery(5))});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader =
      static_cast<StringColumnReaderTestAccessor*>(structReader->children()[0]);
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());
  reader->doPrepareReadAsIndices(0, rows, nullptr);

  nimble::Vector<std::string_view> nonNullVec{pool()};
  for (int i = 0; i < kRows; ++i) {
    if (i % 5 != 0) {
      nonNullVec.push_back(kAlphabet[i % 3]);
    }
  }
  auto nonNullSpan =
      std::span<const std::string_view>(nonNullVec.data(), nonNullVec.size());
  nimble::EncodingSelection<std::string_view> valSelection{
      {.encodingType = nimble::EncodingType::Dictionary},
      nimble::Statistics<std::string_view>::create(nonNullSpan),
      std::make_unique<TrivialNestedPolicy<std::string_view>>()};
  Buffer valBuffer(*pool());
  auto serializedValues = nimble::DictionaryEncoding<std::string_view>::encode(
      valSelection, nonNullSpan, valBuffer);

  nimble::Vector<bool> nullBits{pool()};
  for (int i = 0; i < kRows; ++i) {
    nullBits.push_back(i % 5 != 0);
  }
  auto nullSpan = std::span<const bool>(nullBits.data(), nullBits.size());
  nimble::EncodingSelection<bool> nullSelection{
      {.encodingType = nimble::EncodingType::Trivial},
      nimble::Statistics<bool>::create(nullSpan),
      std::make_unique<TrivialNestedPolicy<bool>>()};
  Buffer nullBuffer(*pool());
  auto serializedNulls = nimble::TrivialEncoding<bool>::encode(
      nullSelection, nullSpan, nullBuffer);

  const uint32_t encodingSize = nimble::EncodingPrefix::kFixedPrefixSize + 4 +
      serializedValues.size() + serializedNulls.size();
  Buffer assemblyBuffer(*pool());
  char* reserved = assemblyBuffer.reserve(encodingSize);
  char* pos = reserved;
  nimble::encoding::writeChar(
      static_cast<char>(nimble::EncodingType::Nullable), pos);
  nimble::encoding::writeChar(static_cast<char>(nimble::DataType::String), pos);
  nimble::encoding::writeUint32(kRows, pos);
  nimble::encoding::writeString(serializedValues, pos);
  nimble::encoding::writeBytes(serializedNulls, pos);

  std::vector<velox::BufferPtr> stringBuffers;
  auto encoding = nimble::EncodingFactory().create(
      *pool(),
      std::string_view(reserved, encodingSize),
      [&](uint32_t size) {
        stringBuffers.push_back(
            velox::AlignedBuffer::allocate<char>(size, pool()));
        return stringBuffers.back()->asMutable<void>();
      },
      nimble::Encoding::Options{});

  ASSERT_TRUE(encoding->isNullable());
  ASSERT_TRUE(encoding->dictionaryEnabled());
  const auto alphabetSize = encoding->dictionarySize();
  const auto* alphabet =
      static_cast<const std::string_view*>(encoding->dictionaryEntries());

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  DecoderVisitor<
      int32_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      /*isDense=*/true>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  callReadIndicesWithVisitor(*encoding, visitor, params);

  ASSERT_EQ(reader->numValues(), kRows);
  const auto* indices = reader->rawIndices();
  for (int i = 0; i < kRows; ++i) {
    if (i % 5 == 0) {
      continue;
    }
    ASSERT_GE(indices[i], 0) << "row " << i;
    ASSERT_LT(indices[i], static_cast<int32_t>(alphabetSize)) << "row " << i;
    EXPECT_EQ(alphabet[indices[i]], kAlphabet[i % 3]) << "row " << i;
  }
}

// ===========================================================================
// Test: readIndicesWithVisitor with a value exclusively in null-masked rows.
// Verifies that the dictionary excludes "delta" which only appears at null
// positions (rows 0, 4, 8, ...) and that indices correctly map to the
// 3-entry non-null alphabet.
// ===========================================================================
TEST_P(
    ReadWithVisitorNonLegacyTest,
    readIndicesWithVisitorNullableAlphabetValue) {
  constexpr int kRows = 20;
  // "delta" appears only at rows 0,4,8,12,16 which are all null.
  // Non-null rows see only "alpha", "bravo", "charlie".
  constexpr std::string_view kValues[] = {"delta", "alpha", "bravo", "charlie"};
  auto isNull = [](auto i) { return i % 4 == 0; };

  auto input = makeRowVector({makeFlatVector<StringView>(
      kRows, [&](auto i) { return StringView(kValues[i % 4]); }, isNull)});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader =
      static_cast<StringColumnReaderTestAccessor*>(structReader->children()[0]);
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());
  reader->doPrepareReadAsIndices(0, rows, nullptr);

  // Build encoding manually with only non-null values.
  nimble::Vector<std::string_view> nonNullVec{pool()};
  for (int i = 0; i < kRows; ++i) {
    if (!isNull(i)) {
      nonNullVec.push_back(kValues[i % 4]);
    }
  }
  auto nonNullSpan =
      std::span<const std::string_view>(nonNullVec.data(), nonNullVec.size());
  nimble::EncodingSelection<std::string_view> valSelection{
      {.encodingType = nimble::EncodingType::Dictionary},
      nimble::Statistics<std::string_view>::create(nonNullSpan),
      std::make_unique<TrivialNestedPolicy<std::string_view>>()};
  Buffer valBuffer(*pool());
  auto serializedValues = nimble::DictionaryEncoding<std::string_view>::encode(
      valSelection, nonNullSpan, valBuffer);

  nimble::Vector<bool> nullBits{pool()};
  for (int i = 0; i < kRows; ++i) {
    nullBits.push_back(!isNull(i));
  }
  auto nullSpan = std::span<const bool>(nullBits.data(), nullBits.size());
  nimble::EncodingSelection<bool> nullSelection{
      {.encodingType = nimble::EncodingType::Trivial},
      nimble::Statistics<bool>::create(nullSpan),
      std::make_unique<TrivialNestedPolicy<bool>>()};
  Buffer nullBuffer(*pool());
  auto serializedNulls = nimble::TrivialEncoding<bool>::encode(
      nullSelection, nullSpan, nullBuffer);

  const uint32_t encodingSize = nimble::EncodingPrefix::kFixedPrefixSize + 4 +
      serializedValues.size() + serializedNulls.size();
  Buffer assemblyBuffer(*pool());
  char* reserved = assemblyBuffer.reserve(encodingSize);
  char* pos = reserved;
  nimble::encoding::writeChar(
      static_cast<char>(nimble::EncodingType::Nullable), pos);
  nimble::encoding::writeChar(static_cast<char>(nimble::DataType::String), pos);
  nimble::encoding::writeUint32(kRows, pos);
  nimble::encoding::writeString(serializedValues, pos);
  nimble::encoding::writeBytes(serializedNulls, pos);

  std::vector<velox::BufferPtr> stringBuffers;
  auto encoding = nimble::EncodingFactory().create(
      *pool(),
      std::string_view(reserved, encodingSize),
      [&](uint32_t size) {
        stringBuffers.push_back(
            velox::AlignedBuffer::allocate<char>(size, pool()));
        return stringBuffers.back()->asMutable<void>();
      },
      nimble::Encoding::Options{});

  ASSERT_TRUE(encoding->isNullable());
  ASSERT_TRUE(encoding->dictionaryEnabled());
  // Dictionary should have 3 entries (alpha, bravo, charlie), NOT 4.
  // "delta" only appeared in null rows and was excluded.
  EXPECT_EQ(encoding->dictionarySize(), 3);

  const auto* alphabet =
      static_cast<const std::string_view*>(encoding->dictionaryEntries());
  std::set<std::string_view> alphabetSet(
      alphabet, alphabet + encoding->dictionarySize());
  EXPECT_FALSE(alphabetSet.count("delta"))
      << "Dictionary should not contain null-masked-only value";
  EXPECT_TRUE(alphabetSet.count("alpha"));
  EXPECT_TRUE(alphabetSet.count("bravo"));
  EXPECT_TRUE(alphabetSet.count("charlie"));

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  DecoderVisitor<
      int32_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      /*isDense=*/true>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  callReadIndicesWithVisitor(*encoding, visitor, params);

  ASSERT_EQ(reader->numValues(), kRows);
  const auto* indices = reader->rawIndices();
  for (int i = 0; i < kRows; ++i) {
    if (isNull(i)) {
      continue;
    }
    ASSERT_GE(indices[i], 0) << "row " << i;
    ASSERT_LT(indices[i], 3) << "row " << i;
    EXPECT_EQ(alphabet[indices[i]], kValues[i % 4]) << "row " << i;
  }
}

// ===========================================================================
// Test: Fuzz readIndicesWithVisitor on NullableEncoding<DictionaryEncoding>.
// ===========================================================================
TEST_P(ReadWithVisitorNonLegacyTest, fuzzReadIndicesWithVisitorNullable) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  constexpr int kNumRuns = 10;

  for (int run = 0; run < kNumRuns; ++run) {
    const int numRows = 20 + rng() % 100;
    const int cardinality = 2 + rng() % 8;

    SCOPED_TRACE(fmt::format("run={} seed={} numRows={}", run, seed, numRows));

    std::vector<std::string> stringPool;
    for (int j = 0; j < cardinality; ++j) {
      int len = rng() % 20;
      std::string s = fmt::format("{}:", j);
      for (int k = 0; k < len; ++k) {
        s += static_cast<char>('a' + rng() % 26);
      }
      stringPool.push_back(std::move(s));
    }

    std::vector<std::string_view> data;
    nimble::Vector<bool> nullBits{pool()};
    data.reserve(numRows);
    for (int i = 0; i < numRows; ++i) {
      bool isNull = (rng() % 5 == 0);
      nullBits.push_back(!isNull);
      data.push_back(stringPool[rng() % stringPool.size()]);
    }

    nimble::Vector<std::string_view> nonNullVec{pool()};
    for (int i = 0; i < numRows; ++i) {
      if (nullBits[i]) {
        nonNullVec.push_back(data[i]);
      }
    }

    if (nonNullVec.empty()) {
      continue;
    }

    auto input = makeRowVector({makeFlatVector<StringView>(
        numRows,
        [&](auto i) { return StringView(data[i]); },
        [&](auto i) { return !nullBits[i]; })});
    auto rowType = asRowType(input->type());
    auto ctx = makeFileContext(input);
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*rowType);
    auto root = buildReader(*ctx, rowType, *scanSpec);

    auto* structReader =
        dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(
            root.get());
    auto* reader = static_cast<StringColumnReaderTestAccessor*>(
        structReader->children()[0]);

    std::vector<vector_size_t> rowVec(numRows);
    std::iota(rowVec.begin(), rowVec.end(), 0);
    RowSet rows(rowVec.data(), rowVec.size());
    reader->doPrepareReadAsIndices(0, rows, nullptr);

    auto nonNullSpan =
        std::span<const std::string_view>(nonNullVec.data(), nonNullVec.size());
    nimble::EncodingSelection<std::string_view> valSelection{
        {.encodingType = nimble::EncodingType::Dictionary},
        nimble::Statistics<std::string_view>::create(nonNullSpan),
        std::make_unique<TrivialNestedPolicy<std::string_view>>()};
    Buffer valBuffer(*this->pool());
    auto serializedValues =
        nimble::DictionaryEncoding<std::string_view>::encode(
            valSelection, nonNullSpan, valBuffer);

    auto nullSpan = std::span<const bool>(nullBits.data(), nullBits.size());
    nimble::EncodingSelection<bool> nullSelection{
        {.encodingType = nimble::EncodingType::Trivial},
        nimble::Statistics<bool>::create(nullSpan),
        std::make_unique<TrivialNestedPolicy<bool>>()};
    Buffer nullBuffer(*this->pool());
    auto serializedNulls = nimble::TrivialEncoding<bool>::encode(
        nullSelection, nullSpan, nullBuffer);

    const uint32_t encodingSize = nimble::EncodingPrefix::kFixedPrefixSize + 4 +
        serializedValues.size() + serializedNulls.size();
    Buffer assemblyBuffer(*this->pool());
    char* reserved = assemblyBuffer.reserve(encodingSize);
    char* pos = reserved;
    nimble::encoding::writeChar(
        static_cast<char>(nimble::EncodingType::Nullable), pos);
    nimble::encoding::writeChar(
        static_cast<char>(nimble::DataType::String), pos);
    nimble::encoding::writeUint32(numRows, pos);
    nimble::encoding::writeString(serializedValues, pos);
    nimble::encoding::writeBytes(serializedNulls, pos);

    std::vector<velox::BufferPtr> stringBuffers;
    auto encoding = nimble::EncodingFactory().create(
        *this->pool(),
        std::string_view(reserved, encodingSize),
        [&](uint32_t size) {
          stringBuffers.push_back(
              velox::AlignedBuffer::allocate<char>(size, this->pool()));
          return stringBuffers.back()->asMutable<void>();
        },
        nimble::Encoding::Options{});

    ASSERT_TRUE(encoding->isNullable());
    ASSERT_TRUE(encoding->dictionaryEnabled());
    const auto alphabetSize = encoding->dictionarySize();
    const auto* alphabet =
        static_cast<const std::string_view*>(encoding->dictionaryEntries());

    common::AlwaysTrue filter;
    dwio::common::ExtractToReader extractValues(reader);
    DecoderVisitor<
        int32_t,
        common::AlwaysTrue,
        dwio::common::ExtractToReader,
        /*isDense=*/true>
        visitor(filter, reader, rows, extractValues);
    auto params = makeReadWithVisitorParams(visitor, rows, this->pool());

    callReadIndicesWithVisitor(*encoding, visitor, params);

    ASSERT_EQ(reader->numValues(), numRows);
    const auto* indices = reader->rawIndices();
    for (int i = 0; i < numRows; ++i) {
      if (!nullBits[i]) {
        continue;
      }
      ASSERT_GE(indices[i], 0) << "row " << i;
      ASSERT_LT(indices[i], static_cast<int32_t>(alphabetSize)) << "row " << i;
      EXPECT_EQ(alphabet[indices[i]], data[i]) << "row " << i;
    }
  }
}

// ===========================================================================
// Test: Dictionary-encoded string data read through the flat path.
// Verifies that DictionaryEncoding::readWithVisitor correctly decodes
// dictionary indices → alphabet lookup → StringView output.
// ===========================================================================
TEST_P(ReadWithVisitorTest, stringDictionaryEncoded) {
  constexpr int kRows = 100;
  constexpr std::string_view kExpected[] = {"alpha", "bravo", "charlie"};
  auto input = makeRowVector({makeFlatVector<StringView>(
      kRows, [&](auto i) { return StringView(kExpected[i % 3]); })});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* child = readColumn(root.get(), kRows);

  ASSERT_EQ(child->numValues(), kRows);
  auto values = getValues<StringView>(child);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(
        std::string_view(values[i].data(), values[i].size()), kExpected[i % 3])
        << "row " << i;
  }
}

// ===========================================================================
// Test: Nullable dictionary-encoded string data.
// Verifies correct value output when NullableEncoding wraps DictionaryEncoding.
// ===========================================================================
TEST_P(ReadWithVisitorTest, stringDictionaryEncodedNullable) {
  constexpr int kRows = 100;
  constexpr std::string_view kExpected[] = {"x", "y", "z"};
  auto input = makeRowVector({makeFlatVector<StringView>(
      kRows,
      [&](auto i) { return StringView(kExpected[i % 3]); },
      nullEvery(7))});
  auto rowType = asRowType(input->type());

  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());

  auto root = buildReader(*ctx, rowType, *scanSpec);
  auto* child = readColumn(root.get(), kRows);

  ASSERT_EQ(child->numValues(), kRows);
  auto values = getValues<StringView>(child);
  for (int i = 0; i < kRows; ++i) {
    if (i % 7 == 0) {
      continue;
    }
    EXPECT_EQ(
        std::string_view(values[i].data(), values[i].size()), kExpected[i % 3])
        << "row " << i;
  }
}

// ===========================================================================
// Test: Fuzz string data with VectorFuzzer — exercises DictionaryEncoding
// readWithVisitor with randomized alphabet content and null patterns.
// ===========================================================================
TEST_P(ReadWithVisitorTest, fuzzStringDictionaryEncoded) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;

  constexpr int kNumRuns = 10;

  for (int run = 0; run < kNumRuns; ++run) {
    auto runSeed = seed + run;
    velox::VectorFuzzer::Options opts;
    opts.vectorSize = 50 + runSeed % 200;
    opts.nullRatio = (runSeed % 3 == 0) ? 0.0 : 0.2;
    opts.stringLength = 1 + runSeed % 20;
    opts.stringVariableLength = true;
    velox::VectorFuzzer fuzzer(opts, pool(), runSeed);

    SCOPED_TRACE(
        fmt::format(
            "run={} seed={} vectorSize={} nullRatio={}",
            run,
            runSeed,
            opts.vectorSize,
            opts.nullRatio));

    auto c0 = fuzzer.fuzzFlat(velox::VARCHAR());
    auto input = makeRowVector({c0});
    auto rowType = asRowType(input->type());

    auto ctx = makeFileContext(input);
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*rowType);
    scanSpec->childByName("c0")->setFilter(
        std::make_unique<common::AlwaysTrue>());

    auto root = buildReader(*ctx, rowType, *scanSpec);
    auto* child = readColumn(root.get(), opts.vectorSize);

    // Build expected values from the input vector (AlwaysTrue passes all rows).
    auto* flat = c0->asFlatVector<StringView>();
    int expectedCount = opts.vectorSize;
    ASSERT_EQ(child->numValues(), expectedCount);

    auto values = getValues<StringView>(child);
    for (int i = 0; i < expectedCount; ++i) {
      if (flat->isNullAt(i)) {
        continue;
      }
      auto expected = flat->valueAt(i);
      EXPECT_EQ(
          std::string_view(values[i].data(), values[i].size()),
          std::string_view(expected.data(), expected.size()))
          << "row " << i;
    }
  }
}

// ===========================================================================
// MainlyConstant dictionary API tests
// ===========================================================================

// Verifies dictionaryEnabled/dictionarySize/dictionaryEntry on
// MC<Dict<string>>. The combined alphabet is [innerAlphabet..., commonValue].
TEST_P(ReadWithVisitorNonLegacyTest, mainlyConstantDictionaryApiString) {
  constexpr int kRows = 100;
  const std::string_view commonValue = "common_value_long_string";
  const std::string_view otherValues[] = {"alpha", "bravo", "charlie"};

  std::vector<std::string_view> data(kRows, commonValue);
  for (int i = 0; i < kRows; i += 10) {
    data[i] = otherValues[(i / 10) % 3];
  }

  auto encoding = encodeMainlyConstantDictionary<std::string_view>(data);

  ASSERT_EQ(encoding->encodingType(), EncodingType::MainlyConstant);
  ASSERT_TRUE(encoding->dictionaryEnabled());

  const auto alphabetSize = encoding->dictionarySize();
  EXPECT_EQ(alphabetSize, 4);

  std::set<std::string> entries;
  for (uint32_t i = 0; i < alphabetSize; ++i) {
    const auto* entry =
        static_cast<const std::string_view*>(encoding->dictionaryEntry(i));
    entries.insert(std::string(*entry));
  }
  EXPECT_TRUE(entries.count("alpha"));
  EXPECT_TRUE(entries.count("bravo"));
  EXPECT_TRUE(entries.count("charlie"));
  EXPECT_TRUE(entries.count(std::string(commonValue)));
}

// Verifies dictionaryEnabled/dictionarySize/dictionaryEntry on MC<Dict<int64>>.
TEST_P(ReadWithVisitorNonLegacyTest, mainlyConstantDictionaryApiInteger) {
  constexpr int kRows = 100;
  constexpr int64_t commonValue = 42;

  std::vector<int64_t> data(kRows, commonValue);
  for (int i = 0; i < kRows; i += 10) {
    data[i] = 100 + (i / 10) % 3;
  }

  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int64_t>(
      MainlyConstantEnc{.otherValues = DictionaryEnc{}}, data, *pool(), buffer);

  ASSERT_EQ(encoding->encodingType(), EncodingType::MainlyConstant);
  ASSERT_TRUE(encoding->dictionaryEnabled());

  const auto alphabetSize = encoding->dictionarySize();
  EXPECT_EQ(alphabetSize, 4);

  std::set<int64_t> entries;
  for (uint32_t i = 0; i < alphabetSize; ++i) {
    const auto* entry =
        static_cast<const int64_t*>(encoding->dictionaryEntry(i));
    entries.insert(*entry);
  }
  EXPECT_TRUE(entries.count(100));
  EXPECT_TRUE(entries.count(101));
  EXPECT_TRUE(entries.count(102));
  EXPECT_TRUE(entries.count(commonValue));
}

// Verifies that MC without Dict inner encoding returns dictionaryEnabled=false.
TEST_P(ReadWithVisitorTest, mainlyConstantNoDictionaryApi) {
  constexpr int kRows = 100;
  std::vector<int64_t> data(kRows, 42);
  for (int i = 0; i < kRows; i += 10) {
    data[i] = 100 + i;
  }

  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int64_t>(
      MainlyConstantEnc{}, data, *pool(), buffer);

  ASSERT_EQ(encoding->encodingType(), EncodingType::MainlyConstant);
  EXPECT_FALSE(encoding->dictionaryEnabled());
}

// Fuzz readIndicesWithVisitor for MC→Dict string encoding.
TEST_P(
    ReadWithVisitorNonLegacyTest,
    fuzzReadIndicesWithVisitorMainlyConstantString) {
  auto seed = folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng(seed);

  constexpr int kNumRuns = 10;

  for (int run = 0; run < kNumRuns; ++run) {
    const int numRows = 20 + rng() % 200;
    const int cardinality = 2 + rng() % 6;
    const int commonPct = 70 + rng() % 25;

    SCOPED_TRACE(
        fmt::format(
            "run={} seed={} numRows={} cardinality={} commonPct={}",
            run,
            seed,
            numRows,
            cardinality,
            commonPct));

    std::vector<std::string> otherPool;
    for (int j = 0; j < cardinality; ++j) {
      int len = 1 + rng() % 20;
      std::string s;
      for (int k = 0; k < len; ++k) {
        s += static_cast<char>('a' + rng() % 26);
      }
      otherPool.push_back(std::move(s));
    }
    int commonLen = 10 + rng() % 40;
    std::string commonValue(commonLen, 'x' + rng() % 3);

    std::vector<std::string_view> data;
    std::vector<std::string> storage;
    storage.reserve(numRows);
    data.reserve(numRows);
    for (int i = 0; i < numRows; ++i) {
      if (static_cast<int>(rng() % 100) < commonPct) {
        storage.push_back(commonValue);
      } else {
        storage.push_back(otherPool[rng() % otherPool.size()]);
      }
      data.push_back(storage.back());
    }

    auto input = makeRowVector({makeFlatVector<StringView>(
        numRows, [&](auto i) { return StringView(data[i]); })});
    auto rowType = asRowType(input->type());
    auto ctx = makeFileContext(input);
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*rowType);
    auto root = buildReader(*ctx, rowType, *scanSpec);

    auto* structReader =
        dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(
            root.get());
    auto* reader = static_cast<StringColumnReaderTestAccessor*>(
        structReader->children()[0]);

    std::vector<vector_size_t> rowVec(numRows);
    std::iota(rowVec.begin(), rowVec.end(), 0);
    RowSet rows(rowVec.data(), rowVec.size());
    reader->doPrepareReadAsIndices(0, rows, nullptr);

    auto encoding = encodeMainlyConstantDictionary<std::string_view>(
        std::vector<std::string_view>(data.begin(), data.end()));

    ASSERT_TRUE(encoding->dictionaryEnabled());
    auto alphabet =
        buildEncodingDictionaryAlphabet<std::string_view>(encoding.get());

    common::AlwaysTrue filter;
    dwio::common::ExtractToReader extractValues(reader);
    DecoderVisitor<
        int32_t,
        common::AlwaysTrue,
        dwio::common::ExtractToReader,
        /*isDense=*/true>
        visitor(filter, reader, rows, extractValues);
    auto params = makeReadWithVisitorParams(visitor, rows, pool());

    callReadIndicesWithVisitor(*encoding, visitor, params);

    ASSERT_EQ(reader->numValues(), numRows);
    const auto* indices = reader->rawIndices();
    for (int i = 0; i < numRows; ++i) {
      ASSERT_GE(indices[i], 0) << "row " << i;
      ASSERT_LT(indices[i], static_cast<int32_t>(alphabet.size()))
          << "row " << i;
      EXPECT_EQ(alphabet[indices[i]], data[i]) << "row " << i;
    }
  }
}

// Verifies that numValues() after readIndicesWithVisitor on MC→Dict equals
// the total row count, not just the inner non-common count. The inner
// encoding's bulkScan calls addNumValues with the non-common count; MC must
// adjust to the full count.
TEST_P(ReadWithVisitorNonLegacyTest, numValuesAfterMainlyConstantReadIndices) {
  constexpr int kRows = 50;
  const std::string_view commonValue = "COMMON_VALUE_STRING";
  const std::string_view otherValues[] = {"alpha", "bravo", "charlie"};

  std::vector<std::string_view> data;
  data.reserve(kRows);
  int numCommon = 0;
  for (int i = 0; i < kRows; ++i) {
    if (i % 5 != 0) {
      data.push_back(commonValue);
      ++numCommon;
    } else {
      data.push_back(otherValues[i % 3]);
    }
  }
  const int numNonCommon = kRows - numCommon;
  ASSERT_GT(numCommon, 0);
  ASSERT_GT(numNonCommon, 0);

  auto input = makeRowVector({makeFlatVector<StringView>(
      kRows, [&](auto i) { return StringView(data[i]); })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader =
      static_cast<StringColumnReaderTestAccessor*>(structReader->children()[0]);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());
  reader->doPrepareReadAsIndices(0, rows, nullptr);

  auto encoding = encodeMainlyConstantDictionary<std::string_view>(
      std::vector<std::string_view>(data.begin(), data.end()));
  ASSERT_TRUE(encoding->dictionaryEnabled());

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  DecoderVisitor<
      int32_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      /*isDense=*/true>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  ASSERT_EQ(reader->numValues(), 0);
  callReadIndicesWithVisitor(*encoding, visitor, params);

  // numValues must equal kRows (all rows), not just numNonCommon
  // (what the inner Dict encoding's bulkScan wrote via addNumValues).
  EXPECT_EQ(reader->numValues(), kRows)
      << "numValues should be total rows (" << kRows
      << "), not just non-common rows (" << numNonCommon << ")";
}

// ===========================================================================
// Tests for detail::readDenseMaterializedIndices and
// detail::readSparseMaterializedIndices helper functions.
// ===========================================================================

// Verifies that readDenseMaterializedIndices writes correct dictionary indices
// to rawValues_ and updates visitor state (numValues, rowIndex).
TEST_P(ReadWithVisitorNonLegacyTest, readDenseMaterializedIndicesNoNulls) {
  constexpr int kRows = 100;
  constexpr int64_t kDistinct[] = {10, 20, 30};

  // Build data with 3 distinct values cycling.
  std::vector<int64_t> data(kRows);
  for (int i = 0; i < kRows; ++i) {
    data[i] = kDistinct[i % 3];
  }

  // Create reader infrastructure.
  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kRows, [&](auto i) { return data[i]; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());
  reader->doPrepareRead<int32_t>(0, rows, nullptr);

  // Encode as DictionaryEncoding independently.
  Buffer buffer(*pool());
  auto encoding =
      createFromCustomLayout<int64_t>(DictionaryEnc{}, data, *pool(), buffer);
  ASSERT_TRUE(encoding->dictionaryEnabled());
  const auto* alphabet =
      static_cast<const int64_t*>(encoding->dictionaryEntries());
  const auto alphabetSize = encoding->dictionarySize();

  // Build visitor.
  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int32_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);

  // Verify numValues is 0 before the call.
  ASSERT_EQ(reader->numValues(), 0);

  // Call the helper with no nulls.
  ReadWithVisitorParams params{.numScanned = 0};
  params.prepareResultNulls = [] {};
  detail::readDenseMaterializedIndices(
      *encoding,
      visitor,
      params,
      /*rawNulls=*/nullptr,
      /*numReadRows=*/kRows,
      /*numNonNulls=*/kRows);

  // Verify numValues was updated by the helper.
  EXPECT_EQ(reader->numValues(), kRows);

  // Verify indices in rawValues_.
  const auto* indices = reader->rawIndices();
  for (int i = 0; i < kRows; ++i) {
    SCOPED_TRACE(fmt::format("row {}", i));
    ASSERT_GE(indices[i], 0);
    ASSERT_LT(indices[i], static_cast<int32_t>(alphabetSize));
    EXPECT_EQ(alphabet[indices[i]], data[i]);
  }
}

// Verifies that readDenseMaterializedIndices correctly scatters indices when
// a null bitmap is present (every 5th row null). Non-null positions should
// have correct indices; null positions are gaps.
TEST_P(ReadWithVisitorNonLegacyTest, readDenseMaterializedIndicesWithNulls) {
  constexpr int kRows = 100;
  constexpr int64_t kDistinct[] = {10, 20, 30};

  // Build data with 3 distinct values cycling.
  std::vector<int64_t> data(kRows);
  for (int i = 0; i < kRows; ++i) {
    data[i] = kDistinct[i % 3];
  }

  // Build null bitmap: every 5th row is null.
  auto nulls = velox::allocateNulls(kRows, pool(), velox::bits::kNotNull);
  auto* rawNulls = nulls->asMutable<uint64_t>();
  int numNonNull = 0;
  for (int i = 0; i < kRows; ++i) {
    if (i % 5 == 0) {
      velox::bits::setNull(rawNulls, i);
    } else {
      ++numNonNull;
    }
  }

  // Build non-null-only data for encoding.
  std::vector<int64_t> nonNullData;
  nonNullData.reserve(numNonNull);
  for (int i = 0; i < kRows; ++i) {
    if (i % 5 != 0) {
      nonNullData.push_back(data[i]);
    }
  }

  // Create reader infrastructure.
  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kRows, [&](auto i) { return data[i]; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());
  reader->doPrepareRead<int32_t>(0, rows, nullptr);

  // Encode only non-null values as DictionaryEncoding.
  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int64_t>(
      DictionaryEnc{}, nonNullData, *pool(), buffer);
  ASSERT_TRUE(encoding->dictionaryEnabled());
  const auto* alphabet =
      static_cast<const int64_t*>(encoding->dictionaryEntries());
  const auto alphabetSize = encoding->dictionarySize();

  // Build visitor.
  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int32_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);

  ASSERT_EQ(reader->numValues(), 0);

  // Call the helper with nulls. prepareResultNulls must mirror what
  // ChunkedDecoder wires up in production: the dense index path materializes
  // nulls into the read-range bitmap only, so it needs an allocated,
  // output-indexed result-nulls buffer to copy them into whenever
  // returnReaderNulls_ is false -- as it is here, the scan spec carries a
  // filter.
  ReadWithVisitorParams params{.numScanned = 0};
  params.prepareResultNulls = [&] {
    reader->prepareNulls(rows, /*hasNulls=*/true, /*extraRows=*/8);
  };
  detail::readDenseMaterializedIndices(
      *encoding,
      visitor,
      params,
      rawNulls,
      /*numReadRows=*/kRows,
      /*numNonNulls=*/numNonNull);

  // Verify numValues was updated by the helper.
  EXPECT_EQ(reader->numValues(), kRows);

  // Verify non-null positions have correct indices.
  const auto* indices = reader->rawIndices();
  for (int i = 0; i < kRows; ++i) {
    if (i % 5 == 0) {
      continue;
    }
    SCOPED_TRACE(fmt::format("row {}", i));
    ASSERT_GE(indices[i], 0);
    ASSERT_LT(indices[i], static_cast<int32_t>(alphabetSize));
    EXPECT_EQ(alphabet[indices[i]], data[i]);
  }
}

// Verifies that readSparseMaterializedIndices materializes and gathers the
// correct indices for a sparse (non-dense) row set, and updates visitor state.
TEST_P(ReadWithVisitorNonLegacyTest, readSparseMaterializedIndices) {
  constexpr int kTotalRows = 30;
  constexpr int64_t kDistinct[] = {10, 20, 30};

  // Build data with 3 distinct values cycling.
  std::vector<int64_t> data(kTotalRows);
  for (int i = 0; i < kTotalRows; ++i) {
    data[i] = kDistinct[i % 3];
  }

  // Create reader infrastructure.
  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kTotalRows, [&](auto i) { return data[i]; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  // Sparse row set: rows [2, 5, 8].
  std::vector<vector_size_t> sparseRows = {2, 5, 8};
  RowSet rows(sparseRows.data(), sparseRows.size());
  reader->doPrepareRead<int32_t>(0, rows, nullptr);

  // Encode as DictionaryEncoding.
  Buffer buffer(*pool());
  auto encoding =
      createFromCustomLayout<int64_t>(DictionaryEnc{}, data, *pool(), buffer);
  ASSERT_TRUE(encoding->dictionaryEnabled());
  const auto* alphabet =
      static_cast<const int64_t*>(encoding->dictionaryEntries());
  const auto alphabetSize = encoding->dictionarySize();

  // Build visitor with sparse (non-dense) rows.
  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = false;
  DecoderVisitor<
      int32_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  ASSERT_EQ(reader->numValues(), 0);

  // Call the helper.
  std::vector<uint32_t> scratchBuffer(kTotalRows);
  detail::readSparseMaterializedIndices(
      *encoding,
      visitor,
      params.numScanned,
      params.prepareResultNulls,
      /*rawNulls=*/nullptr,
      /*numReadRows=*/kTotalRows,
      /*numNonNulls=*/kTotalRows,
      scratchBuffer.data());

  // Verify numValues was updated by the helper.
  EXPECT_EQ(reader->numValues(), static_cast<int32_t>(sparseRows.size()));

  // Verify gathered indices match the selected rows.
  const auto* indices = reader->rawIndices();
  for (int i = 0; i < static_cast<int>(sparseRows.size()); ++i) {
    SCOPED_TRACE(fmt::format("sparse index {} (row {})", i, sparseRows[i]));
    ASSERT_GE(indices[i], 0);
    ASSERT_LT(indices[i], static_cast<int32_t>(alphabetSize));
    EXPECT_EQ(alphabet[indices[i]], data[sparseRows[i]]);
  }
}

// Verifies that readSparseMaterializedIndices correctly skips null rows in a
// sparse row set. Null rows should not produce output values.
TEST_P(ReadWithVisitorNonLegacyTest, readSparseMaterializedIndicesWithNulls) {
  constexpr int kTotalRows = 30;
  constexpr int64_t kDistinct[] = {10, 20, 30};

  // Build data with 3 distinct values cycling.
  std::vector<int64_t> data(kTotalRows);
  for (int i = 0; i < kTotalRows; ++i) {
    data[i] = kDistinct[i % 3];
  }

  // Build null bitmap: rows 0, 5, 10, 15, 20, 25 are null.
  auto nulls = velox::allocateNulls(kTotalRows, pool(), velox::bits::kNotNull);
  auto* rawNulls = nulls->asMutable<uint64_t>();
  int numNonNull = 0;
  for (int i = 0; i < kTotalRows; ++i) {
    if (i % 5 == 0) {
      velox::bits::setNull(rawNulls, i);
    } else {
      ++numNonNull;
    }
  }

  // Build non-null-only data for encoding.
  std::vector<int64_t> nonNullData;
  nonNullData.reserve(numNonNull);
  for (int i = 0; i < kTotalRows; ++i) {
    if (i % 5 != 0) {
      nonNullData.push_back(data[i]);
    }
  }

  // Create reader infrastructure.
  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kTotalRows, [&](auto i) { return data[i]; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  // Sparse row set: rows [2, 5, 8, 10, 13].
  // Row 5 and 10 are null; rows 2, 8, 13 are non-null.
  std::vector<vector_size_t> sparseRows = {2, 5, 8, 10, 13};
  RowSet rows(sparseRows.data(), sparseRows.size());
  reader->doPrepareRead<int32_t>(0, rows, nullptr);

  // Encode only non-null values as DictionaryEncoding.
  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int64_t>(
      DictionaryEnc{}, nonNullData, *pool(), buffer);
  ASSERT_TRUE(encoding->dictionaryEnabled());
  const auto* alphabet =
      static_cast<const int64_t*>(encoding->dictionaryEntries());
  const auto alphabetSize = encoding->dictionarySize();

  // Build visitor with sparse (non-dense) rows.
  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = false;
  DecoderVisitor<
      int32_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  ASSERT_EQ(reader->numValues(), 0);

  // Call the helper.
  std::vector<uint32_t> scratchBuffer(numNonNull);
  detail::readSparseMaterializedIndices(
      *encoding,
      visitor,
      params.numScanned,
      params.prepareResultNulls,
      rawNulls,
      /*numReadRows=*/kTotalRows,
      /*numNonNulls=*/numNonNull,
      scratchBuffer.data());

  // Verify numValues was updated by the helper.
  EXPECT_EQ(reader->numValues(), static_cast<int32_t>(sparseRows.size()));

  // Verify gathered indices for non-null rows.
  const auto* indices = reader->rawIndices();
  for (int i = 0; i < static_cast<int>(sparseRows.size()); ++i) {
    const auto row = sparseRows[i];
    if (row % 5 == 0) {
      // Null row — skipped by readSparseMaterializedIndices.
      continue;
    }
    SCOPED_TRACE(fmt::format("sparse index {} (row {})", i, row));
    ASSERT_GE(indices[i], 0);
    ASSERT_LT(indices[i], static_cast<int32_t>(alphabetSize));
    EXPECT_EQ(alphabet[indices[i]], data[row]);
  }
}

// ===========================================================================
// SimdForBitpack column-reader coverage.
// ===========================================================================
TEST_P(ReadWithVisitorTest, encodingLevelSimdForBitpackBigintRangeSparse) {
  auto runForType = [&]<typename T>(T) {
    SCOPED_TRACE(fmt::format("type size = {}", sizeof(T)));
    constexpr int kRows = 500;
    constexpr int64_t kMinFilter = 20;
    constexpr int64_t kMaxFilter = 40;

    std::vector<T> data(kRows);
    for (int i = 0; i < kRows; ++i) {
      data[i] = static_cast<T>(10 + i % 64);
    }

    auto input = makeRowVector({makeFlatVector<T>(
        kRows, [](auto i) { return static_cast<T>(10 + i % 64); })});
    auto rowType = asRowType(input->type());
    auto ctx = makeFileContext(input);
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*rowType);
    scanSpec->childByName("c0")->setFilter(
        std::make_unique<common::BigintRange>(kMinFilter, kMaxFilter, false));
    auto root = buildReader(*ctx, rowType, *scanSpec);

    auto* structReader =
        dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(
            root.get());
    auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
        dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
    ASSERT_NE(reader, nullptr);

    std::vector<vector_size_t> rowVec;
    for (int i = 0; i < kRows; i += 2) {
      rowVec.push_back(i);
    }
    RowSet rows(rowVec.data(), rowVec.size());

    reader->doPrepareRead<T>(0, rows, nullptr);

    Buffer buffer(*pool());
    const EncodingLayout layout{
        EncodingType::SimdForBitpack, {}, CompressionType::Uncompressed};
    auto encoding = createFromCustomLayout<T>(layout, data, *pool(), buffer);

    common::BigintRange filter(kMinFilter, kMaxFilter, false);
    dwio::common::ExtractToReader extractValues(reader);
    constexpr bool kIsDense = false;
    DecoderVisitor<
        T,
        common::BigintRange,
        dwio::common::ExtractToReader,
        kIsDense>
        visitor(filter, reader, rows, extractValues);
    auto params = makeReadWithVisitorParams(visitor, rows, pool());

    dispatchCallReadWithVisitor(*encoding, visitor, params);

    int expected = 0;
    for (int i = 0; i < kRows; i += 2) {
      if (data[i] >= kMinFilter && data[i] <= kMaxFilter) {
        ++expected;
      }
    }
    EXPECT_EQ(reader->numValues(), expected);

    auto values = getValues<T>(reader);
    for (int i = 0; i < reader->numValues(); ++i) {
      EXPECT_GE(values[i], static_cast<T>(kMinFilter));
      EXPECT_LE(values[i], static_cast<T>(kMaxFilter));
    }
  };

  runForType(int16_t{});
  runForType(int32_t{});
  runForType(int64_t{});
}

// ---------------------------------------------------------------------------
// BlockBitPackingEncoding<int32_t> + AlwaysTrue + dense + FAST PATH
// Exercises the dense bulkScan path that materializes directly into the
// output buffer (no temp buffer copy). Uses multi-chunk data (>1024 rows)
// with variable bit widths per chunk.
// ---------------------------------------------------------------------------
TEST_P(
    ReadWithVisitorTest,
    encodingLevel_BlockBitPacking_AlwaysTrue_Dense_Int32_FastPath) {
  constexpr int kChunkSize = 1024;
  constexpr int kRows = kChunkSize * 3 + 500;

  std::vector<int32_t> data(kRows);
  // Chunk 0: narrow range [100, 103]
  for (int i = 0; i < kChunkSize; ++i) {
    data[i] = 100 + (i % 4);
  }
  // Chunk 1: wider range [0, 255]
  for (int i = kChunkSize; i < kChunkSize * 2; ++i) {
    data[i] = i % 256;
  }
  // Chunk 2: constant 42 (bitWidth=0)
  for (int i = kChunkSize * 2; i < kChunkSize * 3; ++i) {
    data[i] = 42;
  }
  // Chunk 3 (partial): range [0, 15]
  for (int i = kChunkSize * 3; i < kRows; ++i) {
    data[i] = i % 16;
  }

  auto input = makeRowVector(
      {makeFlatVector<int32_t>(kRows, [&](auto i) { return data[i]; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int32_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int32_t>(
      BlockBitPackingEnc{}, data, *pool(), buffer);

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int32_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  EXPECT_EQ(reader->numValues(), kRows);
  auto values = getValues<int32_t>(reader);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(values[i], data[i]) << "row " << i;
  }
}

// ---------------------------------------------------------------------------
// BlockBitPackingEncoding<int32_t> + AlwaysTrue + sparse + FAST PATH
// Exercises the sparse bulkScan path that batches selected rows by chunk,
// materializes each touched chunk range once, then scatters values.
// Selects every 3rd row across multiple chunks.
// ---------------------------------------------------------------------------
TEST_P(
    ReadWithVisitorTest,
    encodingLevel_BlockBitPacking_AlwaysTrue_Sparse_Int32_FastPath) {
  constexpr int kChunkSize = 1024;
  constexpr int kTotalRows = kChunkSize * 3;

  std::vector<int32_t> data(kTotalRows);
  // Chunk 0: narrow range
  for (int i = 0; i < kChunkSize; ++i) {
    data[i] = 1000 + (i % 8);
  }
  // Chunk 1: constant (bitWidth=0 path)
  for (int i = kChunkSize; i < kChunkSize * 2; ++i) {
    data[i] = 77;
  }
  // Chunk 2: wider range
  for (int i = kChunkSize * 2; i < kTotalRows; ++i) {
    data[i] = i % 512;
  }

  auto input = makeRowVector(
      {makeFlatVector<int32_t>(kTotalRows, [&](auto i) { return data[i]; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  // Every 3rd row — sparse and crosses chunk boundaries.
  std::vector<vector_size_t> rowVec;
  for (int i = 0; i < kTotalRows; i += 3) {
    rowVec.push_back(i);
  }
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int32_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int32_t>(
      BlockBitPackingEnc{}, data, *pool(), buffer);

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = false;
  DecoderVisitor<
      int32_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  auto numSelected = static_cast<int>(rowVec.size());
  EXPECT_EQ(reader->numValues(), numSelected);
  auto values = getValues<int32_t>(reader);
  for (int i = 0; i < numSelected; ++i) {
    EXPECT_EQ(values[i], data[rowVec[i]]) << "selected row index " << i;
  }
}

// ---------------------------------------------------------------------------
// BlockBitPackingEncoding<int32_t> + BigintRange + dense + FAST PATH
// Exercises the filter path in bulkScan: only values in [100, 103] pass.
// Data spans multiple chunks with different ranges.
// ---------------------------------------------------------------------------
TEST_P(
    ReadWithVisitorTest,
    encodingLevel_BlockBitPacking_BigintRange_Dense_Int32_FastPath) {
  constexpr int kChunkSize = 1024;
  constexpr int kRows = kChunkSize * 2;

  std::vector<int32_t> data(kRows);
  // Chunk 0: values [100, 103] — all pass filter
  for (int i = 0; i < kChunkSize; ++i) {
    data[i] = 100 + (i % 4);
  }
  // Chunk 1: values [200, 207] — none pass filter
  for (int i = kChunkSize; i < kRows; ++i) {
    data[i] = 200 + (i % 8);
  }

  auto input = makeRowVector(
      {makeFlatVector<int32_t>(kRows, [&](auto i) { return data[i]; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(100, 103, false));
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int32_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int32_t>(
      BlockBitPackingEnc{}, data, *pool(), buffer);

  common::BigintRange filter(100, 103, false);
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int32_t,
      common::BigintRange,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  // Only chunk 0 values pass the filter [100, 103].
  EXPECT_EQ(reader->numValues(), kChunkSize);
  auto values = getValues<int32_t>(reader);
  for (int i = 0; i < kChunkSize; ++i) {
    EXPECT_GE(values[i], 100);
    EXPECT_LE(values[i], 103);
    EXPECT_EQ(values[i], data[i]) << "row " << i;
  }
}

// ---------------------------------------------------------------------------
// BlockBitPackingEncoding<int32_t> + AlwaysTrue + sparse + FAST PATH
// Very sparse selection: only a few rows per chunk, some chunks skipped
// entirely. Exercises the chunk-batching optimization in the sparse path.
// ---------------------------------------------------------------------------
TEST_P(
    ReadWithVisitorTest,
    encodingLevel_BlockBitPacking_AlwaysTrue_VerySparse_Int32_FastPath) {
  constexpr int kChunkSize = 1024;
  constexpr int kTotalRows = kChunkSize * 4;

  std::vector<int32_t> data(kTotalRows);
  for (int i = 0; i < kTotalRows; ++i) {
    data[i] = i;
  }

  auto input = makeRowVector(
      {makeFlatVector<int32_t>(kTotalRows, [&](auto i) { return data[i]; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  // Pick specific rows: first, last, chunk boundaries, and mid-chunk.
  std::vector<vector_size_t> rowVec = {
      0, // chunk 0 first
      512, // chunk 0 mid
      kChunkSize - 1, // chunk 0 last
      kChunkSize, // chunk 1 first
      kChunkSize * 2 + 100, // chunk 2 mid (skip chunk 1 end)
      kChunkSize * 3 - 1, // chunk 2 last
      kChunkSize * 4 - 1, // chunk 3 last (overall last)
  };
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int32_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding = createFromCustomLayout<int32_t>(
      BlockBitPackingEnc{}, data, *pool(), buffer);

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = false;
  DecoderVisitor<
      int32_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  dispatchCallReadWithVisitor(*encoding, visitor, params);

  auto numSelected = static_cast<int>(rowVec.size());
  EXPECT_EQ(reader->numValues(), numSelected);
  auto values = getValues<int32_t>(reader);
  for (int i = 0; i < numSelected; ++i) {
    EXPECT_EQ(values[i], data[rowVec[i]])
        << "selected row " << rowVec[i] << " at index " << i;
  }
}

// ---------------------------------------------------------------------------
// SubIntSplitEncoding<int64_t> + AlwaysTrue + dense
// Exercises the no-filter dense fast path for the 8-byte physical type.
// ---------------------------------------------------------------------------
TEST_P(ReadWithVisitorTest, encodingLevelSubIntSplitAlwaysTrueDense) {
  constexpr int kRows = 500;

  // Values share a high prefix so the splitter produces multiple sections.
  std::vector<int64_t> data(kRows);
  for (int i = 0; i < kRows; ++i) {
    data[i] = static_cast<int64_t>(0x1234560000000000LL + i);
  }

  auto input = makeRowVector({makeFlatVector<int64_t>(
      kRows, [](auto i) { return 0x1234560000000000LL + i; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding = makeSubIntSplitEncoding<int64_t>(data, buffer, *pool());

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int64_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  encoding->readWithVisitor(visitor, params);

  EXPECT_EQ(reader->numValues(), kRows);
  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(values[i], data[i]) << "row " << i;
  }
}

// ---------------------------------------------------------------------------
// SubIntSplitEncoding<int64_t> + BigintRange + sparse
// Exercises the sparse gather + filter fast path.
// ---------------------------------------------------------------------------
TEST_P(ReadWithVisitorTest, encodingLevelSubIntSplitBigintRangeSparse) {
  constexpr int kRows = 500;

  // Small-range data so a BigintRange filter selects a meaningful subset.
  std::vector<int64_t> data(kRows);
  for (int i = 0; i < kRows; ++i) {
    data[i] = i % 16;
  }

  auto input = makeRowVector(
      {makeFlatVector<int64_t>(kRows, [](auto i) { return i % 16; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(5, 10, false));
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  // Sparse RowSet: every other row.
  std::vector<vector_size_t> rowVec;
  for (int i = 0; i < kRows; i += 2) {
    rowVec.push_back(i);
  }
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int64_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding = makeSubIntSplitEncoding<int64_t>(data, buffer, *pool());

  common::BigintRange filter(5, 10, false);
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = false;
  DecoderVisitor<
      int64_t,
      common::BigintRange,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  encoding->readWithVisitor(visitor, params);

  int expected = 0;
  for (int i = 0; i < kRows; i += 2) {
    if (data[i] >= 5 && data[i] <= 10) {
      ++expected;
    }
  }
  EXPECT_EQ(reader->numValues(), expected);
  auto values = getValues<int64_t>(reader);
  for (int i = 0; i < reader->numValues(); ++i) {
    EXPECT_GE(values[i], 5);
    EXPECT_LE(values[i], 10);
  }
}

// ---------------------------------------------------------------------------
// SubIntSplitEncoding<int32_t> + AlwaysTrue + dense
// Exercises the dense fast path for the 4-byte physical type.
// ---------------------------------------------------------------------------
TEST_P(ReadWithVisitorTest, encodingLevelSubIntSplitInt32AlwaysTrueDense) {
  constexpr int kRows = 500;

  std::vector<int32_t> data(kRows);
  for (int i = 0; i < kRows; ++i) {
    data[i] = static_cast<int32_t>(0x12340000 + i);
  }

  auto input = makeRowVector(
      {makeFlatVector<int32_t>(kRows, [](auto i) { return 0x12340000 + i; })});
  auto rowType = asRowType(input->type());
  auto ctx = makeFileContext(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::AlwaysTrue>());
  auto root = buildReader(*ctx, rowType, *scanSpec);

  auto* structReader =
      dynamic_cast<dwio::common::SelectiveStructColumnReaderBase*>(root.get());
  auto* reader = static_cast<IntegerColumnReaderTestAccessor*>(
      dynamic_cast<IntegerColumnReader*>(structReader->children()[0]));
  ASSERT_NE(reader, nullptr);

  std::vector<vector_size_t> rowVec(kRows);
  std::iota(rowVec.begin(), rowVec.end(), 0);
  RowSet rows(rowVec.data(), rowVec.size());

  reader->doPrepareRead<int32_t>(0, rows, nullptr);

  Buffer buffer(*pool());
  auto encoding = makeSubIntSplitEncoding<int32_t>(data, buffer, *pool());

  common::AlwaysTrue filter;
  dwio::common::ExtractToReader extractValues(reader);
  constexpr bool kIsDense = true;
  DecoderVisitor<
      int32_t,
      common::AlwaysTrue,
      dwio::common::ExtractToReader,
      kIsDense>
      visitor(filter, reader, rows, extractValues);
  auto params = makeReadWithVisitorParams(visitor, rows, pool());

  encoding->readWithVisitor(visitor, params);

  EXPECT_EQ(reader->numValues(), kRows);
  auto values = getValues<int32_t>(reader);
  for (int i = 0; i < kRows; ++i) {
    EXPECT_EQ(values[i], data[i]) << "row " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(
    NonLegacyAndLegacy,
    ReadWithVisitorTest,
    ::testing::Values(true, false),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "NonLegacy" : "Legacy";
    });

INSTANTIATE_TEST_SUITE_P(
    NonLegacyOnly,
    ReadWithVisitorNonLegacyTest,
    ::testing::Values(true),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "NonLegacy" : "Legacy";
    });

} // namespace
} // namespace facebook::nimble
