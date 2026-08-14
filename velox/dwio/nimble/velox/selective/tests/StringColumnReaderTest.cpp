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

#include "velox/dwio/nimble/velox/selective/SelectiveNimbleReader.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/File.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/testutil/RandomSeed.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/tests/TabletTestUtils.h"
#include "velox/dwio/nimble/velox/ChunkedStream.h"
#include "velox/dwio/nimble/velox/SchemaSerialization.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"
#include "velox/dwio/nimble/writer/FlushPolicy.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/vector/DictionaryVector.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>

namespace facebook::nimble {
namespace {

using namespace facebook::velox;

// Tests for StringColumnReader covering VARCHAR and VARBINARY edge cases.
class StringColumnReaderTest : public ::testing::Test,
                               public velox::test::VectorTestBase,
                               public ::testing::WithParamInterface<bool> {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(velox::memory::MemoryManager::Options{});
    common::testutil::TestValue::enable();
    registerSelectiveNimbleReaderFactory();
  }

  static void TearDownTestCase() {
    unregisterSelectiveNimbleReaderFactory();
  }

  memory::MemoryPool* rootPool() {
    return rootPool_.get();
  }

  struct Readers {
    std::unique_ptr<dwio::common::Reader> reader;
    std::unique_ptr<dwio::common::RowReader> rowReader;
  };

  Readers makeReaders(
      const RowVectorPtr& expected,
      const std::string& file,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      bool stringDecoderZeroCopy) {
    auto readFile = std::make_shared<InMemoryReadFile>(file);
    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
    dwio::common::ReaderOptions options(pool());
    options.setDataIoStats(dataIoStats_);
    options.setMetadataIoStats(metadataIoStats_);
    options.setScanSpec(scanSpec);
    Readers readers;
    readers.reader = factory->createReader(
        std::make_unique<dwio::common::BufferedInput>(readFile, *pool()),
        options);
    auto type = asRowType(expected->type());
    dwio::common::RowReaderOptions rowOptions;
    rowOptions.setScanSpec(scanSpec);
    rowOptions.setRequestedType(type);
    rowOptions.setStringDecoderZeroCopy(stringDecoderZeroCopy);
    rowOptions.setNimblePreserveDictionaryEncoding(stringDecoderZeroCopy);
    readers.rowReader = readers.reader->createRowReader(rowOptions);
    return readers;
  }

  Readers makeReaders(
      const RowVectorPtr& input,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      bool stringDecoderZeroCopy) {
    return makeReaders(
        input,
        test::createNimbleFile(*rootPool(), input),
        scanSpec,
        stringDecoderZeroCopy);
  }

  void validate(
      const RowVector& expected,
      dwio::common::RowReader& rowReader,
      int batchSize) {
    auto result = BaseVector::create(asRowType(expected.type()), 0, pool());
    int numScanned = 0;
    int offset = 0;
    while (numScanned < expected.size()) {
      numScanned += rowReader.next(batchSize, result);
      result->validate();
      for (int j = 0; j < result->size(); ++j) {
        ASSERT_TRUE(result->equalValueAt(&expected, j, offset + j))
            << "Mismatch at row " << (offset + j);
      }
      offset += result->size();
    }
    ASSERT_EQ(numScanned, expected.size());
    ASSERT_EQ(0, rowReader.next(1, result));
  }

  void validateWithFilter(
      const RowVector& input,
      dwio::common::RowReader& rowReader,
      int batchSize,
      const std::function<bool(int)>& filter,
      std::optional<VectorEncoding::Simple> expectedChildEncoding =
          std::nullopt,
      int childIndex = 0) {
    // The RowReader already has the actual ScanSpec filter. This predicate
    // identifies the input rows expected to pass that filter.
    auto result = BaseVector::create(asRowType(input.type()), 0, pool());
    int numScanned = 0;
    int i = 0;
    while (numScanned < input.size()) {
      numScanned += rowReader.next(batchSize, result);
      result->validate();
      // When set, assert the column under test was materialized with the
      // expected encoding (e.g. DICTIONARY for the dict-index path, FLAT for
      // the flat path), so a silent encoding fallback cannot mask the read
      // path.
      if (expectedChildEncoding.has_value() && result->size() > 0) {
        EXPECT_EQ(
            BaseVector::loadedVectorShared(
                result->asUnchecked<RowVector>()->childAt(childIndex))
                ->encoding(),
            *expectedChildEncoding)
            << "Unexpected encoding for child " << childIndex;
      }
      for (int j = 0; j < result->size(); ++j) {
        for (;;) {
          ASSERT_LT(i, input.size());
          if (filter(i)) {
            break;
          }
          ++i;
        }
        ASSERT_TRUE(result->equalValueAt(&input, j, i))
            << "Mismatch at input row " << i;
        ++i;
      }
    }
    while (i < input.size()) {
      ASSERT_FALSE(filter(i));
      ++i;
    }
    ASSERT_EQ(numScanned, input.size());
    ASSERT_EQ(0, rowReader.next(1, result));
  }

  const std::shared_ptr<io::IoStatistics> dataIoStats_{
      std::make_shared<io::IoStatistics>()};
  const std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};
};

// Returns the encoding of a vector, loading through lazy wrappers if present.
VectorEncoding::Simple getEncoding(const VectorPtr& vector) {
  return BaseVector::loadedVectorShared(vector)->encoding();
}

// The on-disk encoding of one chunk of a scalar stream.
struct ChunkEncoding {
  // Top-level encoding of the chunk. Any Nullable/Sentinel wrapper is already
  // stripped by EncodingLayoutCapture, so this is the data encoding directly.
  EncodingType encodingType;
  // Run-values sub-encoding, populated only for RLE chunks.
  std::optional<EncodingType> runValuesType;
};

// Reads the on-disk encoding of every chunk of the scalar stream for the column
// at 'columnIndex' from an in-memory Nimble file. Used to assert that crafted
// data produced the intended per-chunk encodings (e.g. RLE<Dictionary> vs
// Trivial), since natural encoding selection is data-driven, not guaranteed.
std::vector<ChunkEncoding> getStreamChunkEncodings(
    const std::string& file,
    uint32_t columnIndex,
    velox::memory::MemoryPool* pool) {
  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tablet =
      TabletReader::create(readFile, pool, test::makeTestTabletOptions(pool));

  auto section = tablet->loadOptionalSection(std::string(kSchemaSection));
  NIMBLE_CHECK(section.has_value(), "Schema not found.");
  auto schema = SchemaDeserializer::deserialize(section->content().data());
  const auto& scalarNode = schema->asRow().childAt(columnIndex)->asScalar();
  const std::vector<uint32_t> identifiers{
      scalarNode.scalarDescriptor().offset()};

  std::vector<ChunkEncoding> result;
  for (uint32_t stripe = 0; stripe < tablet->stripeCount(); ++stripe) {
    auto streams = tablet->load(tablet->stripeIdentifier(stripe), identifiers);
    if (streams.empty() || streams[0] == nullptr) {
      continue;
    }
    InMemoryChunkedStream chunkedStream{*pool, std::move(streams[0])};
    while (chunkedStream.hasNext()) {
      auto capture = EncodingLayoutCapture::capture(
          chunkedStream.nextChunk(), Encoding::Options{});
      ChunkEncoding chunk{capture.encodingType(), std::nullopt};
      if (capture.encodingType() == EncodingType::RLE) {
        const auto& runValues =
            capture.child(EncodingIdentifiers::RunLength::RunValues);
        if (runValues.has_value()) {
          chunk.runValuesType = runValues->encodingType();
        }
      }
      result.push_back(chunk);
    }
  }
  return result;
}

EncodingLayout makeFsstEncodingLayout() {
  return EncodingLayout{
      EncodingType::Fsst,
      {},
      CompressionType::Uncompressed,
      {EncodingLayout{
          EncodingType::Trivial, {}, CompressionType::Uncompressed}}};
}

// Validates correctness and checks the expected encoding per batch.
// expectedEncodings[i] is the expected encoding for the i-th non-empty batch.
// totalRows is the total number of rows in the file (used to drive the read
// loop). expected is the expected string column values (only qualifying rows).
// childIndex selects which child of the result RowVector to check.
void validateWithEncodingChecks(
    const BaseVector& expected,
    dwio::common::RowReader& rowReader,
    int batchSize,
    int totalRows,
    const std::vector<VectorEncoding::Simple>& expectedEncodings,
    const RowTypePtr& resultType,
    velox::memory::MemoryPool* pool,
    int childIndex = 0) {
  auto result = BaseVector::create(resultType, 0, pool);
  int numScanned = 0;
  int outputOffset = 0;
  int batchIndex = 0;
  while (numScanned < totalRows) {
    numScanned += rowReader.next(batchSize, result);
    result->validate();
    if (result->size() == 0) {
      continue;
    }
    auto* row = result->asUnchecked<RowVector>();
    auto stringChild = BaseVector::loadedVectorShared(row->childAt(childIndex));
    ASSERT_LT(batchIndex, expectedEncodings.size());
    EXPECT_EQ(stringChild->encoding(), expectedEncodings[batchIndex])
        << "Non-empty batch " << batchIndex << " at output offset "
        << outputOffset << " has unexpected encoding";
    for (int j = 0; j < result->size(); ++j) {
      ASSERT_TRUE(stringChild->equalValueAt(&expected, j, outputOffset + j))
          << "Mismatch at row " << (outputOffset + j);
    }
    outputOffset += result->size();
    ++batchIndex;
  }
  ASSERT_EQ(outputOffset, expected.size());
  ASSERT_EQ(0, rowReader.next(1, result));
}

// Strings shorter than StringView::kInlineSize (12 bytes).
TEST_P(StringColumnReaderTest, shortStrings) {
  const bool stringDecoderZeroCopy = GetParam();
  auto input = makeRowVector({
      makeFlatVector<std::string>(
          100, [](auto i) { return std::to_string(i); }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 23);
}

// Strings longer than inline size.
TEST_P(StringColumnReaderTest, longStrings) {
  const bool stringDecoderZeroCopy = GetParam();
  const std::string prefix(20, 'x');
  auto input = makeRowVector({
      makeFlatVector<std::string>(
          50, [&](auto i) { return prefix + std::to_string(i); }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 13);
}

// Alternating short and long strings.
TEST_P(StringColumnReaderTest, mixedLengthStrings) {
  const bool stringDecoderZeroCopy = GetParam();
  const std::string longPrefix(20, 'y');
  auto input = makeRowVector({
      makeFlatVector<std::string>(
          100,
          [&](auto i) {
            return i % 2 == 0 ? std::to_string(i)
                              : longPrefix + std::to_string(i);
          }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 17);
}

// Strings with null values.
TEST_P(StringColumnReaderTest, stringsWithNulls) {
  const bool stringDecoderZeroCopy = GetParam();
  auto input = makeRowVector({
      makeFlatVector<std::string>(
          100, [](auto i) { return "str_" + std::to_string(i); }, nullEvery(5)),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 11);
}

// All-null string column.
TEST_P(StringColumnReaderTest, allNullStrings) {
  const bool stringDecoderZeroCopy = GetParam();
  auto input = makeRowVector({
      makeConstant<StringView>(std::nullopt, 50),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  auto result = BaseVector::create(input->type(), 0, pool());
  ASSERT_EQ(readers.rowReader->next(50, result), 50);
  auto* row = result->asUnchecked<RowVector>();
  for (int i = 0; i < 50; ++i) {
    ASSERT_TRUE(row->childAt(0)->isNullAt(i));
  }
}

// All empty "" strings.
TEST_P(StringColumnReaderTest, emptyStrings) {
  const bool stringDecoderZeroCopy = GetParam();
  auto input = makeRowVector({
      makeFlatVector<std::string>(50, [](auto) { return ""; }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 50);
}

// VARBINARY type (shares reader with VARCHAR).
TEST_P(StringColumnReaderTest, varbinaryColumn) {
  const bool stringDecoderZeroCopy = GetParam();
  auto input = makeRowVector({
      makeFlatVector<std::string>(
          50,
          [](auto i) {
            std::string s(4, '\0');
            s[0] = static_cast<char>(i);
            s[1] = static_cast<char>(i + 1);
            s[2] = static_cast<char>(i + 2);
            s[3] = static_cast<char>(i + 3);
            return s;
          }),
  });
  // We write as VARCHAR but the byte content round-trips the same way.
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 50);
}

// BytesValues filter on string column.
TEST_P(StringColumnReaderTest, stringBytesFilter) {
  const bool stringDecoderZeroCopy = GetParam();
  auto c0 = makeFlatVector<std::string>(
      100, [](auto i) { return "val_" + std::to_string(i % 10); });
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BytesValues>(
          std::vector<std::string>{"val_3", "val_7"}, false));
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validateWithFilter(*input, *readers.rowReader, 23, [](auto i) {
    return i % 10 == 3 || i % 10 == 7;
  });
}

// BytesRange filter on string column.
TEST_P(StringColumnReaderTest, stringBytesRangeFilter) {
  const bool stringDecoderZeroCopy = GetParam();
  auto c0 = makeFlatVector<std::string>(
      100, [](auto i) { return "val_" + std::to_string(i % 10); });
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BytesRange>(
          "val_3", false, false, "val_5", false, false, false));
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validateWithFilter(*input, *readers.rowReader, 23, [](auto i) {
    auto val = "val_" + std::to_string(i % 10);
    return val >= "val_3" && val <= "val_5";
  });
}

// Filter on string column with a sibling integer column. Both columns are
// projected. Verifies that the dict path correctly produces outputRows_ for
// the struct reader and that the sibling column's values are correctly
// filtered.
TEST_P(StringColumnReaderTest, stringFilterWithSiblingColumn) {
  const bool stringDecoderZeroCopy = GetParam();
  const int numRows = 100;
  auto filterCol = makeFlatVector<std::string>(
      numRows, [](auto i) { return "val_" + std::to_string(i % 10); });
  auto dataCol =
      makeFlatVector<int64_t>(numRows, [](auto i) { return i * 100; });
  auto input = makeRowVector({filterCol, dataCol});

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BytesValues>(
          std::vector<std::string>{"val_3", "val_7"}, false));
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);

  auto resultType = asRowType(input->type());
  auto result = BaseVector::create(resultType, 0, pool());
  int numScanned = 0;
  int expectedRow = 0;
  while (numScanned < numRows) {
    numScanned += readers.rowReader->next(23, result);
    auto* rowResult = result->as<RowVector>();
    for (int j = 0; j < result->size(); ++j) {
      while (expectedRow < numRows &&
             !(expectedRow % 10 == 3 || expectedRow % 10 == 7)) {
        ++expectedRow;
      }
      ASSERT_LT(expectedRow, numRows);
      // c1 (data column) should match the corresponding input row.
      ASSERT_TRUE(
          rowResult->childAt(1)->equalValueAt(dataCol.get(), j, expectedRow))
          << "c1 mismatch at filtered position " << j << " (input row "
          << expectedRow << ")";
      ++expectedRow;
    }
  }
}

TEST_P(StringColumnReaderTest, fsstWithSiblingFilter) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "FSST is only available in the current encoding factory.";
  }

  constexpr int kRows = 2'000;
  auto filterCol = makeFlatVector<int64_t>(kRows, [](auto i) { return i; });
  auto dataCol = makeFlatVector<std::string>(kRows, [](auto i) {
    return fmt::format("common/prefix/for/fsst/selective/{}", i % 32);
  });
  auto input = makeRowVector({filterCol, dataCol});

  WriterOptions writerOptions;
  writerOptions.fsstCompressionTargetRatio = 10.0;
  writerOptions.encodingLayoutTree.emplace(
      Kind::Row,
      std::
          unordered_map<EncodingLayoutTree::StreamIdentifier, EncodingLayout>{},
      "",
      std::vector<EncodingLayoutTree>{
          EncodingLayoutTree{Kind::Scalar, {}, "c0"},
          EncodingLayoutTree{
              Kind::Scalar,
              {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
                makeFsstEncodingLayout()}},
              "c1"}});
  auto file = test::createNimbleFile(*rootPool(), input, writerOptions);

  {
    SCOPED_TRACE("sibling filter");
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    scanSpec->childByName("c0")->setFilter(
        std::make_unique<common::BigintRange>(
            250, 1'249, /*nullAllowed=*/false));
    auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);

    validateWithFilter(*input, *readers.rowReader, 137, [](auto i) {
      return i >= 250 && i <= 1'249;
    });
  }

  {
    SCOPED_TRACE("FSST string filter");
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    scanSpec->childByName("c1")->setFilter(
        std::make_unique<common::BytesValues>(
            std::vector<std::string>{
                "common/prefix/for/fsst/selective/3",
                "common/prefix/for/fsst/selective/17"},
            false));
    auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);

    validateWithFilter(*input, *readers.rowReader, 137, [](auto i) {
      return i % 32 == 3 || i % 32 == 17;
    });
  }
}

// Filter that passes all rows — verifies no-op filtering doesn't corrupt.
TEST_P(StringColumnReaderTest, stringFilterPassesAll) {
  const bool stringDecoderZeroCopy = GetParam();
  auto c0 = makeFlatVector<std::string>(
      100, [](auto i) { return "val_" + std::to_string(i % 5); });
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BytesValues>(
          std::vector<std::string>{"val_0", "val_1", "val_2", "val_3", "val_4"},
          false));
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validateWithFilter(*input, *readers.rowReader, 23, [](auto) { return true; });
}

// Filter that passes no rows — verifies empty output is handled correctly.
TEST_P(StringColumnReaderTest, stringFilterPassesNone) {
  const bool stringDecoderZeroCopy = GetParam();
  auto c0 = makeFlatVector<std::string>(
      100, [](auto i) { return "val_" + std::to_string(i % 5); });
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BytesValues>(
          std::vector<std::string>{"nonexistent"}, false));
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  auto result = BaseVector::create(asRowType(input->type()), 0, pool());
  int numScanned = 0;
  while (numScanned < input->size()) {
    numScanned += readers.rowReader->next(23, result);
    ASSERT_EQ(result->size(), 0) << "Expected no rows to pass filter";
  }
}

// Filter with nulls — nulls should not pass BytesValues filter.
TEST_P(StringColumnReaderTest, stringFilterWithNulls) {
  const bool stringDecoderZeroCopy = GetParam();
  std::vector<std::optional<std::string>> data(100);
  for (int i = 0; i < 100; ++i) {
    if (i % 5 == 0) {
      data[i] = std::nullopt;
    } else {
      data[i] = "val_" + std::to_string(i % 10);
    }
  }
  auto c0 = makeNullableFlatVector<std::string>(data);
  auto input = makeRowVector({c0});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BytesValues>(
          std::vector<std::string>{"val_3", "val_7"}, false));
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validateWithFilter(*input, *readers.rowReader, 23, [](auto i) {
    if (i % 5 == 0) {
      return false;
    }
    return i % 10 == 3 || i % 10 == 7;
  });
}

// Fuzz test: filter on dict-encoded string column with random data shapes,
// null rates, filter selectivities, and batch sizes.
TEST_P(StringColumnReaderTest, fuzzFilterDictionary) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary filter path requires stringDecoderZeroCopy";
  }

  for (int run = 0; run < 5; ++run) {
    // Deterministic by default; set env VELOX_TEST_USE_RANDOM_SEED=1 for a
    // fresh, logged seed per run (idiomatic with stress runs).
    auto seed = common::testutil::getRandomSeed(/*fixedValue=*/1 + run);
    std::mt19937 rng(seed);
    const int numRows = 100 + rng() % 900;
    const int batchSize = 10 + rng() % numRows;
    const int cardinality = 3 + rng() % 10;
    const bool hasNulls = rng() % 3 == 0;
    const int numAccepted = 1 + rng() % cardinality;

    SCOPED_TRACE(
        fmt::format(
            "run={} seed={} numRows={} batchSize={} cardinality={} hasNulls={} numAccepted={}",
            run,
            seed,
            numRows,
            batchSize,
            cardinality,
            hasNulls,
            numAccepted));

    // Build accepted set.
    std::vector<std::string> accepted;
    for (int i = 0; i < numAccepted; ++i) {
      accepted.push_back(fmt::format("val_{}", i));
    }
    std::set<std::string> acceptedSet(accepted.begin(), accepted.end());

    // Build data.
    std::vector<std::optional<std::string>> data(numRows);
    for (int i = 0; i < numRows; ++i) {
      if (hasNulls && rng() % 5 == 0) {
        data[i] = std::nullopt;
      } else {
        data[i] = fmt::format("val_{}", rng() % cardinality);
      }
    }
    auto c0 = makeNullableFlatVector<std::string>(data);
    auto input = makeRowVector({c0});
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    scanSpec->childByName("c0")->setFilter(
        std::make_unique<common::BytesValues>(accepted, false));
    auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);

    // Validate: read all batches and compare against reference filter.
    auto result = BaseVector::create(asRowType(input->type()), 0, pool());
    int numScanned = 0;
    int inputRow = 0;
    while (numScanned < numRows) {
      numScanned += readers.rowReader->next(batchSize, result);
      for (int j = 0; j < result->size(); ++j) {
        while (inputRow < numRows) {
          if (c0->isNullAt(inputRow)) {
            ++inputRow;
            continue;
          }
          auto sv = c0->valueAt(inputRow).str();
          if (acceptedSet.count(sv)) {
            break;
          }
          ++inputRow;
        }
        ASSERT_LT(inputRow, numRows)
            << "Ran out of input rows at result position " << j;
        ASSERT_TRUE(result->equalValueAt(input.get(), j, inputRow))
            << "Mismatch at input row " << inputRow;
        ++inputRow;
      }
    }
  }
}

// Small batch sizes to exercise skip positioning.
TEST_P(StringColumnReaderTest, stringSkipAndRead) {
  const bool stringDecoderZeroCopy = GetParam();
  const std::string longPrefix(15, 'z');
  auto input = makeRowVector({
      makeFlatVector<std::string>(
          100,
          [&](auto i) {
            return i % 3 == 0 ? std::to_string(i)
                              : longPrefix + std::to_string(i);
          }),
  });
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, 7);
}

// Verifies that dictionary state is correctly invalidated when skip crosses
// a chunk boundary. Without the onChunkLoad callback, alphabet_ would retain
// stale string_views from the previous chunk, producing corrupted output.
TEST_P(StringColumnReaderTest, skipAcrossChunkBoundary) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    // Dict vector path requires stringDecoderZeroCopy.
    return;
  }

  // Chunk 1: 500 rows with alphabet {"aaa", "bbb", "ccc"}.
  auto chunk1 = makeRowVector({makeFlatVector<std::string>(500, [](auto i) {
    return std::vector<std::string>{"aaa", "bbb", "ccc"}[i % 3];
  })});
  // Chunk 2: 500 rows with a completely different alphabet {"xxx", "yyy"}.
  auto chunk2 = makeRowVector({makeFlatVector<std::string>(500, [](auto i) {
    return std::vector<std::string>{"xxx", "yyy"}[i % 2];
  })});

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file = test::createNimbleFile(
      *rootPool(), {chunk1, chunk2}, writerOptions, /*flushAfterWrite=*/false);

  // Build expected vector by concatenating both chunks.
  auto expected = makeRowVector({makeFlatVector<std::string>(1000, [](auto i) {
    if (i < 500) {
      return std::vector<std::string>{"aaa", "bbb", "ccc"}[i % 3];
    }
    return std::vector<std::string>{"xxx", "yyy"}[(i - 500) % 2];
  })});

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*expected->type());
  auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);

  // Use a batch size that forces the reader to skip across the chunk boundary.
  // Read batch 1 (rows 0-199) from chunk 1, then skip to batch 3 (rows 400+)
  // which crosses the chunk boundary at row 500.
  validate(*expected, *readers.rowReader, 200);
}

// Verifies that dictionary state is invalidated when the remainingValues gate
// forces a flat-path fallback mid-file. With 2 chunks of 300 rows and batch
// size 200:
//   Read 1 (rows 0-199): chunk 1 has 300 remaining >= 200, dict path taken,
//     alphabet built from chunk 1.
//   Read 2 (rows 200-399): chunk 1 has only 100 remaining < 200, falls to flat
//     path which must clear dictionary state.
//   Read 3 (rows 400-599): chunk 2 has enough rows, dict path re-entered.
//     Must rebuild alphabet from chunk 2, not reuse stale chunk 1 alphabet.
TEST_P(StringColumnReaderTest, flatFallbackClearsDictionaryState) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    return;
  }

  auto chunk1 = makeRowVector({makeFlatVector<std::string>(300, [](auto i) {
    return std::vector<std::string>{"aaa", "bbb"}[i % 2];
  })});
  auto chunk2 = makeRowVector({makeFlatVector<std::string>(300, [](auto i) {
    return std::vector<std::string>{"xxx", "yyy"}[i % 2];
  })});

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file = test::createNimbleFile(
      *rootPool(), {chunk1, chunk2}, writerOptions, /*flushAfterWrite=*/false);

  auto expected = makeRowVector({makeFlatVector<std::string>(600, [](auto i) {
    if (i < 300) {
      return std::vector<std::string>{"aaa", "bbb"}[i % 2];
    }
    return std::vector<std::string>{"xxx", "yyy"}[(i - 300) % 2];
  })});

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*expected->type());
  auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
  using E = VectorEncoding::Simple;
  validateWithEncodingChecks(
      *expected->childAt(0),
      *readers.rowReader,
      200,
      /*totalRows=*/600,
      {E::DICTIONARY, E::DICTIONARY, E::DICTIONARY},
      asRowType(expected->type()),
      pool());
}

// When the skip in prepareRead crosses a chunk boundary mid-file, the
// decoder loads a new chunk. The dictionary state must be rebuilt from
// the new chunk's encoding.
TEST_P(StringColumnReaderTest, skipAcrossChunkBoundaryMidFile) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kChunkRows = 500;
  constexpr int kBatchSize = 250;

  auto chunk1 = makeRowVector({
      makeFlatVector<int64_t>(kChunkRows, [](auto i) { return i; }),
      makeFlatVector<std::string>(
          kChunkRows,
          [](auto i) {
            return std::vector<std::string>{"aaa", "bbb", "ccc"}[i % 3];
          }),
  });
  auto chunk2 = makeRowVector({
      makeFlatVector<int64_t>(
          kChunkRows, [](auto i) { return kChunkRows + i; }),
      makeFlatVector<std::string>(
          kChunkRows,
          [](auto i) { return std::vector<std::string>{"xxx", "yyy"}[i % 2]; }),
  });

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file = test::createNimbleFile(
      *rootPool(), {chunk1, chunk2}, writerOptions, /*flushAfterWrite=*/false);

  // Filter on c0: accept [0, 249] ∪ [600, 900]. c1 has no filter.
  auto readType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  std::vector<std::unique_ptr<common::BigintRange>> ranges;
  ranges.push_back(
      std::make_unique<common::BigintRange>(
          0, kBatchSize - 1, /*nullAllowed=*/false));
  ranges.push_back(
      std::make_unique<common::BigintRange>(600, 900, /*nullAllowed=*/false));
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintMultiRange>(
          std::move(ranges), /*nullAllowed=*/false));

  auto expectedStrings = makeFlatVector<std::string>(551, [](auto i) {
    if (i < 250) {
      return std::vector<std::string>{"aaa", "bbb", "ccc"}[i % 3];
    }
    int chunk2Offset = (i - 250) + 100;
    return std::vector<std::string>{"xxx", "yyy"}[chunk2Offset % 2];
  });

  auto readers = makeReaders(chunk1, file, scanSpec, stringDecoderZeroCopy);

  using E = VectorEncoding::Simple;
  validateWithEncodingChecks(
      *expectedStrings,
      *readers.rowReader,
      kBatchSize,
      /*totalRows=*/2 * kChunkRows,
      {E::DICTIONARY, E::DICTIONARY, E::DICTIONARY},
      readType,
      pool(),
      /*childIndex=*/1);
}

// Within-chunk dict read with a filter on a sibling column that creates
// row gaps. The string column takes the dict path (no filter on its own
// scanSpec), and the visitor's gap-skipping logic is exercised.
TEST_P(StringColumnReaderTest, skipWithinChunkReturnsDictionary) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kRows = 500;
  auto input = makeRowVector({
      makeFlatVector<int64_t>(kRows, [](auto i) { return i; }),
      makeFlatVector<std::string>(
          kRows,
          [](auto i) {
            return std::vector<std::string>{"aaa", "bbb", "ccc"}[i % 3];
          }),
  });

  auto file = test::createNimbleFile(*rootPool(), input);

  // Filter on c0: accept first 300 rows only (creating a gap at end).
  auto readType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(0, 299, /*nullAllowed=*/false));

  auto expectedStrings = makeFlatVector<std::string>(300, [](auto i) {
    return std::vector<std::string>{"aaa", "bbb", "ccc"}[i % 3];
  });

  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);

  using E = VectorEncoding::Simple;
  validateWithEncodingChecks(
      *expectedStrings,
      *readers.rowReader,
      /*batchSize=*/100,
      /*totalRows=*/kRows,
      {E::DICTIONARY, E::DICTIONARY, E::DICTIONARY},
      readType,
      pool(),
      /*childIndex=*/1);
}

// Dict recovery after skip crosses a chunk boundary via filter gaps.
// Two scenarios: (1) skip within a batch crosses the boundary, (2) entire
// batch rejected by filter causes a between-batch skip across the boundary.
TEST_P(StringColumnReaderTest, skipAcrossChunkBoundaryDictRecovery) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  // Scenario 1: mid-batch filter gap crosses chunk boundary.
  // 2 chunks of 400 rows. Filter accepts [0,349] + [600,799].
  // Batch 2 (rows 250-499): accepts 250-349, then skip from 350 to 600
  // crosses chunk boundary → flat. Batch 3 (rows 600+): dict recovery.
  {
    constexpr int kChunkRows = 400;
    auto chunk1 = makeRowVector({
        makeFlatVector<int64_t>(kChunkRows, [](auto i) { return i; }),
        makeFlatVector<std::string>(
            kChunkRows,
            [](auto i) {
              return std::vector<std::string>{"aaa", "bbb", "ccc"}[i % 3];
            }),
    });
    auto chunk2 = makeRowVector({
        makeFlatVector<int64_t>(
            kChunkRows, [](auto i) { return kChunkRows + i; }),
        makeFlatVector<std::string>(
            kChunkRows,
            [](auto i) {
              return std::vector<std::string>{"xxx", "yyy"}[i % 2];
            }),
    });

    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.minStreamChunkRawSize = 0;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    auto file = test::createNimbleFile(
        *rootPool(),
        {chunk1, chunk2},
        writerOptions,
        /*flushAfterWrite=*/false);

    auto readType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*readType);
    std::vector<std::unique_ptr<common::BigintRange>> ranges;
    ranges.push_back(
        std::make_unique<common::BigintRange>(0, 349, /*nullAllowed=*/false));
    ranges.push_back(
        std::make_unique<common::BigintRange>(
            600, 2 * kChunkRows - 1, /*nullAllowed=*/false));
    scanSpec->childByName("c0")->setFilter(
        std::make_unique<common::BigintMultiRange>(
            std::move(ranges), /*nullAllowed=*/false));

    using E = VectorEncoding::Simple;
    auto expectedStrings = makeFlatVector<std::string>(550, [](auto i) {
      if (i < 350) {
        return std::vector<std::string>{"aaa", "bbb", "ccc"}[i % 3];
      }
      int chunk2Offset = (i - 350) + 200;
      return std::vector<std::string>{"xxx", "yyy"}[chunk2Offset % 2];
    });

    auto readers = makeReaders(chunk1, file, scanSpec, stringDecoderZeroCopy);
    validateWithEncodingChecks(
        *expectedStrings,
        *readers.rowReader,
        /*batchSize=*/250,
        /*totalRows=*/2 * kChunkRows,
        {E::DICTIONARY, E::DICTIONARY, E::DICTIONARY, E::DICTIONARY},
        readType,
        pool(),
        /*childIndex=*/1);
  }

  // Scenario 2: entire batch rejected by filter, between-batch skip crosses
  // chunk boundary. 2 chunks of 500 rows. Filter accepts [0,499] + [750,999].
  // Batch 3 (rows 500-749): all rejected → c1 not loaded. Batch 4 (rows
  // 750+): ensureLoaded loads chunk 2, skip 250 within chunk 2 → dict.
  {
    constexpr int kChunkRows = 500;
    auto chunk1 = makeRowVector({
        makeFlatVector<int64_t>(kChunkRows, [](auto i) { return i; }),
        makeFlatVector<std::string>(
            kChunkRows,
            [](auto i) {
              return std::vector<std::string>{"aaa", "bbb", "ccc"}[i % 3];
            }),
    });
    auto chunk2 = makeRowVector({
        makeFlatVector<int64_t>(
            kChunkRows, [](auto i) { return kChunkRows + i; }),
        makeFlatVector<std::string>(
            kChunkRows,
            [](auto i) {
              return std::vector<std::string>{"xxx", "yyy"}[i % 2];
            }),
    });

    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.minStreamChunkRawSize = 0;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    auto file = test::createNimbleFile(
        *rootPool(),
        {chunk1, chunk2},
        writerOptions,
        /*flushAfterWrite=*/false);

    auto readType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*readType);
    std::vector<std::unique_ptr<common::BigintRange>> ranges;
    ranges.push_back(
        std::make_unique<common::BigintRange>(
            0, kChunkRows - 1, /*nullAllowed=*/false));
    ranges.push_back(
        std::make_unique<common::BigintRange>(
            750, 2 * kChunkRows - 1, /*nullAllowed=*/false));
    scanSpec->childByName("c0")->setFilter(
        std::make_unique<common::BigintMultiRange>(
            std::move(ranges), /*nullAllowed=*/false));

    using E = VectorEncoding::Simple;
    auto expectedStrings = makeFlatVector<std::string>(750, [](auto i) {
      if (i < 500) {
        return std::vector<std::string>{"aaa", "bbb", "ccc"}[i % 3];
      }
      int chunk2Offset = (i - 500) + 250;
      return std::vector<std::string>{"xxx", "yyy"}[chunk2Offset % 2];
    });

    auto readers = makeReaders(chunk1, file, scanSpec, stringDecoderZeroCopy);
    validateWithEncodingChecks(
        *expectedStrings,
        *readers.rowReader,
        /*batchSize=*/250,
        /*totalRows=*/2 * kChunkRows,
        {E::DICTIONARY, E::DICTIONARY, E::DICTIONARY},
        readType,
        pool(),
        /*childIndex=*/1);
  }
}

// Regression for T278234148: a between-batch skip in prepareRead crosses a
// chunk boundary and lands on a chunk whose encoding is NOT dictionary-enabled.
//
// readWithDictionary checks dictionaryConvertible() on the chunk that is
// current at entry (chunk1, Dictionary → passes), but then prepareRead<int32_t>
// runs seekTo → skip, which loads a new chunk. When that chunk is a non-dict
// encoding
// (here Trivial, forced by high-cardinality unique strings), the stale guard no
// longer matches the current encoding, and ensureDictionaryState() calls
// buildEncodingDictionaryAlphabet() on it → NIMBLE_CHECK(dictionaryEnabled())
// fires. This mirrors the production crash on
// ad_delivery.ads_ai_eval_judge_results.original_request_id (all-varchar table,
// per-chunk encoding varies between Dictionary and Trivial). The fix re-checks
// dictionaryConvertible() after prepareRead and falls back to the flat path
// (FLAT output encoding).
TEST_P(StringColumnReaderTest, abandonDictionaryAfterSkip) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kChunkRows = 400;
  // Chunk 1: low-cardinality strings → Dictionary (dictionary-enabled).
  auto chunk1 = makeRowVector({
      makeFlatVector<int64_t>(kChunkRows, [](auto i) { return i; }),
      makeFlatVector<std::string>(
          kChunkRows,
          [](auto i) {
            return std::vector<std::string>{"aaa", "bbb", "ccc"}[i % 3];
          }),
  });
  // Chunk 2: all-unique strings → Trivial (NOT dictionary-enabled).
  auto chunk2 = makeRowVector({
      makeFlatVector<int64_t>(
          kChunkRows, [](auto i) { return kChunkRows + i; }),
      makeFlatVector<std::string>(
          kChunkRows,
          [](auto i) { return fmt::format("unique_string_value_{}", i); }),
  });

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file = test::createNimbleFile(
      *rootPool(), {chunk1, chunk2}, writerOptions, /*flushAfterWrite=*/false);

  // Filter on c0: accept [0,349] then [600,799]. Rejecting [350,599] forces a
  // between-batch skip (350 → 600) that crosses the chunk boundary at 400 and
  // lands inside chunk 2 (Trivial), while the decoder is still parked mid
  // chunk 1 at entry (so the dictionaryConvertible() guard sees chunk 1).
  auto readType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  std::vector<std::unique_ptr<common::BigintRange>> ranges;
  ranges.push_back(
      std::make_unique<common::BigintRange>(0, 349, /*nullAllowed=*/false));
  ranges.push_back(
      std::make_unique<common::BigintRange>(
          600, 2 * kChunkRows - 1, /*nullAllowed=*/false));
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintMultiRange>(
          std::move(ranges), /*nullAllowed=*/false));

  // Accepted output: chunk 1 rows 0-349 (350), then chunk 2 rows 600-799 which
  // are chunk-local indices 200-399 (200) → 550 rows total.
  auto expectedStrings = makeFlatVector<std::string>(550, [](auto i) {
    if (i < 350) {
      return std::vector<std::string>{"aaa", "bbb", "ccc"}[i % 3];
    }
    const int chunk2Index = (i - 350) + 200;
    return fmt::format("unique_string_value_{}", chunk2Index);
  });

  auto readers = makeReaders(chunk1, file, scanSpec, stringDecoderZeroCopy);

  using E = VectorEncoding::Simple;
  // Batches: [0,249] dict, [250,349] dict (chunk 1), [600,749] flat,
  // [750,799] flat (chunk 2 via the abandon-to-flat fallback).
  validateWithEncodingChecks(
      *expectedStrings,
      *readers.rowReader,
      /*batchSize=*/250,
      /*totalRows=*/2 * kChunkRows,
      {E::DICTIONARY, E::DICTIONARY, E::FLAT, E::FLAT},
      readType,
      pool(),
      /*childIndex=*/1);
}

// Same as abandonDictionaryAfterSkip but with a NULLABLE string column. The
// between-batch skip that lands in the non-dict chunk drives the abandon path
// (readWithDictionary runs prepareRead<int32_t>, the landed chunk is not
// convertible, so it abandons and read() re-runs
// prepareRead<std::string_view>). Because NimbleData::readNulls advances the
// nulls decoder (nextBools), that double prepareRead reads the null stream
// twice and misplaces the nulls.
TEST_P(StringColumnReaderTest, abandonDictionaryAfterSkipWithNulls) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kChunkRows = 400;
  auto chunk1String = [](auto i) -> std::optional<std::string> {
    if (i % 7 == 0) {
      return std::nullopt;
    }
    return std::vector<std::string>{"aaa", "bbb", "ccc"}[i % 3];
  };
  auto chunk2String = [](auto i) -> std::optional<std::string> {
    if (i % 7 == 0) {
      return std::nullopt;
    }
    return fmt::format("unique_string_value_{}", i);
  };

  std::vector<std::optional<std::string>> chunk1Data(kChunkRows);
  std::vector<std::optional<std::string>> chunk2Data(kChunkRows);
  for (int i = 0; i < kChunkRows; ++i) {
    chunk1Data[i] = chunk1String(i);
    chunk2Data[i] = chunk2String(i);
  }
  // Chunk 1: low-cardinality → nullable Dictionary. Chunk 2: unique → Trivial.
  auto chunk1 = makeRowVector({
      makeFlatVector<int64_t>(kChunkRows, [](auto i) { return i; }),
      makeNullableFlatVector<std::string>(chunk1Data),
  });
  auto chunk2 = makeRowVector({
      makeFlatVector<int64_t>(
          kChunkRows, [](auto i) { return kChunkRows + i; }),
      makeNullableFlatVector<std::string>(chunk2Data),
  });

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file = test::createNimbleFile(
      *rootPool(), {chunk1, chunk2}, writerOptions, /*flushAfterWrite=*/false);

  auto readType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  std::vector<std::unique_ptr<common::BigintRange>> ranges;
  ranges.push_back(
      std::make_unique<common::BigintRange>(0, 349, /*nullAllowed=*/false));
  ranges.push_back(
      std::make_unique<common::BigintRange>(
          600, 2 * kChunkRows - 1, /*nullAllowed=*/false));
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintMultiRange>(
          std::move(ranges), /*nullAllowed=*/false));

  // Accepted output: chunk-1 rows 0-349, then chunk-2 rows 600-799 (chunk-local
  // indices 200-399) → 550 rows total, carrying their nulls.
  std::vector<std::optional<std::string>> expectedData(550);
  for (int i = 0; i < 550; ++i) {
    expectedData[i] = i < 350 ? chunk1String(i) : chunk2String((i - 350) + 200);
  }
  auto expectedStrings = makeNullableFlatVector<std::string>(expectedData);

  auto readers = makeReaders(chunk1, file, scanSpec, stringDecoderZeroCopy);
  using E = VectorEncoding::Simple;
  validateWithEncodingChecks(
      *expectedStrings,
      *readers.rowReader,
      /*batchSize=*/250,
      /*totalRows=*/2 * kChunkRows,
      {E::DICTIONARY, E::DICTIONARY, E::FLAT, E::FLAT},
      readType,
      pool(),
      /*childIndex=*/1);
}

// Non-convertible ENTRY chunk with NO cross-chunk skip: a single Trivial
// (all-unique) nullable string column read contiguously. readWithDictionary
// abandons at entry. Isolates whether the no-skip abandon (prepareRead then
// abandon, no between-batch skip) corrupts nulls.
TEST_P(StringColumnReaderTest, abandonDictionaryNoSkipWithNulls) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kRows = 400;
  std::vector<std::optional<std::string>> data(kRows);
  for (int i = 0; i < kRows; ++i) {
    data[i] = (i % 7 == 0)
        ? std::nullopt
        : std::optional<std::string>(fmt::format("unique_value_{}", i));
  }
  auto chunk = makeRowVector({
      makeFlatVector<int64_t>(kRows, [](auto i) { return i; }),
      makeNullableFlatVector<std::string>(data),
  });

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  auto file = test::createNimbleFile(*rootPool(), chunk, writerOptions);

  auto readType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);

  auto expected = makeNullableFlatVector<std::string>(data);
  auto readers = makeReaders(chunk, file, scanSpec, stringDecoderZeroCopy);
  using E = VectorEncoding::Simple;
  validateWithEncodingChecks(
      *expected,
      *readers.rowReader,
      /*batchSize=*/kRows,
      /*totalRows=*/kRows,
      {E::FLAT},
      readType,
      pool(),
      /*childIndex=*/1);
}

// Latent check #2 bug for a FLATMAP feature: the feature's value stream is
// Dictionary in chunk 1 and Trivial in chunk 2, and a sibling-column filter
// forces a between-batch skip that crosses the chunk boundary, driving the
// feature's child StringColumnReader to abandon the dictionary after
// prepareRead. read() then re-runs prepareRead (numReadRows == 0), so
// NimbleData::readNulls advances the feature's in-map decoder twice and the
// output map gets phantom/missing entries. Plain-null columns survive this, so
// only the flatmap in-map path exposes it.
TEST_P(StringColumnReaderTest, abandonDictionaryAfterSkipFlatMap) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kChunkRows = 400;
  // Feature key 1 is present on ~2/3 of rows (absent when g % 3 == 0), so the
  // in-map bitmap is non-trivial. Its values are low-cardinality in chunk 1
  // (→ Dictionary) and unique in chunk 2 (→ Trivial), all inline (< 12 bytes)
  // so makeMapVector copies them and no source lifetime is needed.
  auto present = [](int g) { return g % 3 != 0; };
  auto valueStr = [](int g) {
    return g < kChunkRows ? fmt::format("v{}", g % 5) : fmt::format("u{}", g);
  };
  std::vector<std::string> keep;
  keep.reserve(4 * kChunkRows);
  auto makeChunk = [&](int base, int n) {
    std::vector<std::vector<std::pair<int32_t, std::optional<StringView>>>>
        maps(n);
    for (int r = 0; r < n; ++r) {
      const int g = base + r;
      if (present(g)) {
        keep.push_back(valueStr(g));
        maps[r] = {{1, StringView(keep.back())}};
      }
    }
    return makeRowVector(
        {"c0", "c1"},
        {makeFlatVector<int64_t>(n, [base](auto r) { return base + r; }),
         makeMapVector<int32_t, StringView>(maps)});
  };
  auto chunk1 = makeChunk(0, kChunkRows);
  auto chunk2 = makeChunk(kChunkRows, kChunkRows);
  auto combined = makeChunk(0, 2 * kChunkRows);

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  writerOptions.flatMapColumns = {{"c1", {}}};
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file = test::createNimbleFile(
      *rootPool(), {chunk1, chunk2}, writerOptions, /*flushAfterWrite=*/false);

  // Filter on sibling c0: accept [0,349] then [600,799]. Rejecting [350,599]
  // forces a between-batch skip (350 → 600) crossing the chunk boundary at 400.
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*combined->type());
  std::vector<std::unique_ptr<common::BigintRange>> ranges;
  ranges.push_back(
      std::make_unique<common::BigintRange>(0, 349, /*nullAllowed=*/false));
  ranges.push_back(
      std::make_unique<common::BigintRange>(
          600, 2 * kChunkRows - 1, /*nullAllowed=*/false));
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintMultiRange>(
          std::move(ranges), /*nullAllowed=*/false));

  auto readers = makeReaders(combined, file, scanSpec, stringDecoderZeroCopy);
  validateWithFilter(
      *combined, *readers.rowReader, /*batchSize=*/250, [](int i) {
        return i <= 349 || i >= 600;
      });
}

// Forces the string column "c1" (child index 1) to encode as RLE wrapping
// Dictionary, while the bigint sibling "c0" (child index 0) keeps its default
// encoding. Row-schema children are matched to the layout tree by position, so
// both children must be present in schema order.
EncodingLayoutTree makeSecondChildRleDictionaryLayoutTree() {
  EncodingLayout dictLayout{
      EncodingType::Dictionary,
      {},
      CompressionType::Uncompressed,
      {std::nullopt, std::nullopt}};
  EncodingLayout rleLayout{
      EncodingType::RLE,
      {},
      CompressionType::Uncompressed,
      {std::nullopt, std::move(dictLayout)}};
  return EncodingLayoutTree{
      Kind::Row,
      std::
          unordered_map<EncodingLayoutTree::StreamIdentifier, EncodingLayout>{},
      "",
      std::vector<EncodingLayoutTree>{
          EncodingLayoutTree{
              Kind::Scalar,
              std::unordered_map<
                  EncodingLayoutTree::StreamIdentifier,
                  EncodingLayout>{},
              "c0"},
          EncodingLayoutTree{Kind::Scalar, {{0, std::move(rleLayout)}}, "c1"}}};
}

// A between-batch skip (inside prepareRead's seekTo) advances the decoder into
// a second RLE<Dictionary> chunk that is dictionary-convertible. Because the
// value decoder carries the dictionary-preserve flag, the skip loads that chunk
// in dictionary mode (dictValues_ set, dictionaryEnabled()==true), so
// readWithDictionary's post-prepareRead dictionaryConvertible() check passes
// and the chunk is read in dictionary mode across the skip.
//
// With lazy dict/flat mode in RLE, the mode is determined by the caller's API
// choice, not a flag at construction time. skipWithoutIndex/seekToChunk load
// chunks without committing to a mode; the subsequent read determines it.
//
// Mirrors skipAcrossChunkBoundaryDictRecovery scenario 1 (kChunkRows=400,
// accept [0,349] u [600,799], batchSize=250), but forces BOTH chunks of the
// string column to RLE<Dictionary> via a forced encoding layout so the second
// chunk is dict-capable and must stay in dictionary mode across the skip. (A
// preserve-gated RLE<Dictionary> is the shape that flips modes with the load
// flag; a plain Dictionary chunk is dict-convertible regardless.)
TEST_P(StringColumnReaderTest, skipAcrossChunkBoundaryRleDictionaryPreserved) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kChunkRows = 400;
  constexpr int kRunLength = 100;
  const std::string chunk1Values[] = {"alpha", "bravo", "charlie"};
  const std::string chunk2Values[] = {"delta", "echo", "foxtrot"};
  // Long runs (100 rows each) so the forced RLE<Dictionary> layout produces
  // real RLE runs of dictionary indices.
  auto chunk1 = makeRowVector({
      makeFlatVector<int64_t>(kChunkRows, [](auto i) { return i; }),
      makeFlatVector<std::string>(
          kChunkRows,
          [&](auto i) { return chunk1Values[(i / kRunLength) % 3]; }),
  });
  auto chunk2 = makeRowVector({
      makeFlatVector<int64_t>(
          kChunkRows, [](auto i) { return kChunkRows + i; }),
      makeFlatVector<std::string>(
          kChunkRows,
          [&](auto i) { return chunk2Values[(i / kRunLength) % 3]; }),
  });

  WriterOptions writerOptions;
  writerOptions.encodingLayoutTree.emplace(
      makeSecondChildRleDictionaryLayoutTree());
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file = test::createNimbleFile(
      *rootPool(), {chunk1, chunk2}, writerOptions, /*flushAfterWrite=*/false);

  // Filter on c0: accept [0, 349] u [600, 799]. c1 has no filter, so it takes
  // the dictionary index path.
  auto readType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  std::vector<std::unique_ptr<common::BigintRange>> ranges;
  ranges.push_back(
      std::make_unique<common::BigintRange>(0, 349, /*nullAllowed=*/false));
  ranges.push_back(
      std::make_unique<common::BigintRange>(
          600, 2 * kChunkRows - 1, /*nullAllowed=*/false));
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintMultiRange>(
          std::move(ranges), /*nullAllowed=*/false));

  // Qualifying rows only, in output order: chunk-1 rows [0, 349] then chunk-2
  // rows [600, 799] (chunk-local [200, 399]).
  auto expectedStrings = makeFlatVector<std::string>(550, [&](auto i) {
    if (i < 350) {
      return chunk1Values[(i / kRunLength) % 3];
    }
    const int fileRow = 600 + (i - 350);
    return chunk2Values[((fileRow - kChunkRows) / kRunLength) % 3];
  });

  auto readers = makeReaders(chunk1, file, scanSpec, stringDecoderZeroCopy);
  using E = VectorEncoding::Simple;
  // Batches: [0,249] dict, [250,349] dict (chunk 1), [600,749] dict,
  // [750,799] dict (chunk 2 — read in dictionary mode across the skip).
  validateWithEncodingChecks(
      *expectedStrings,
      *readers.rowReader,
      /*batchSize=*/250,
      /*totalRows=*/2 * kChunkRows,
      {E::DICTIONARY, E::DICTIONARY, E::DICTIONARY, E::DICTIONARY},
      readType,
      pool(),
      /*childIndex=*/1);
}

// Exercises dictionary preservation (the skip-time preserve flag) together with
// the abandon path across a four-chunk string column whose on-disk chunk
// encodings are RLE<Dictionary>, RLE<Dictionary>, Trivial, RLE<Dictionary>.
//
// A forced encoding layout cannot produce this mix: a layout is bound per
// stream and applied to every chunk, and RLE<Dictionary> never falls back for
// all-unique data (RLE/Dictionary accept any input), so a forced
// RLE<Dictionary> would make the "Trivial" chunk dictionary-convertible too.
// The mix is therefore produced by natural cost-based selection with crafted
// per-chunk data -- many short runs of a few distinct strings pick RLE with a
// Dictionary run-value sub-encoding, while all-unique data picks Trivial -- and
// the on-disk encodings are asserted below (natural selection is data-driven,
// not guaranteed).
//
// The Trivial chunk is not dictionary-convertible, so the dictionary path
// abandons on it; the RLE<Dictionary> chunks are preserve-gated, so a skip that
// lands in one must load it in dictionary mode rather than flatten it. Four
// read patterns are exercised, each with a fresh reader:
//   1) Full range in one batch: abandons at the third chunk -> whole batch
//   FLAT. 2) First read stops mid the second chunk (dictionary); a later read
//   skips
//      across the Trivial third chunk into the fourth chunk, which stays
//      DICTIONARY (preserved across the skip).
//   3) Skip the first chunk entirely and read from the second chunk to the end:
//      the second chunk is loaded by the skip in dictionary mode, then the read
//      abandons at the Trivial third chunk -> FLAT. Differs from pattern 1 in
//      that the first dictionary chunk is reached via a skip.
//   4) Read through the Trivial third chunk (abandoning, which turns the
//      preserve flag off), then skip into the fourth chunk, which is still
//      DICTIONARY because readWithDictionary turns the flag back on before the
//      skip.
TEST_P(
    StringColumnReaderTest,
    abandonAndPreserveAcrossRleDictionaryAndTrivialChunks) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kChunkRows = 400;
  constexpr int kRunLength = 10;
  constexpr int kNumChunks = 4;
  constexpr int kTotalRows = kChunkRows * kNumChunks;
  const std::string kChunk0[3] = {"alpha", "bravo", "charlie"};
  const std::string kChunk1[3] = {"delta", "echo", "foxtrot"};
  const std::string kChunk3[3] = {"golf", "hotel", "india"};

  // Value for global row g. Chunks 0/1/3 cycle three distinct strings in
  // kRunLength-row runs (many runs, few distinct -> RLE<Dictionary>); chunk 2
  // is all-unique (-> Trivial).
  auto valueAt = [&](int g) -> std::string {
    const int chunk = g / kChunkRows;
    const int run = (g % kChunkRows / kRunLength) % 3;
    switch (chunk) {
      case 0:
        return kChunk0[run];
      case 1:
        return kChunk1[run];
      case 2:
        return fmt::format("uniq_{}", g);
      default:
        return kChunk3[run];
    }
  };

  RowVectorPtr schemaSample;
  std::vector<VectorPtr> chunkVectors;
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    const int base = chunk * kChunkRows;
    auto rowVector = makeRowVector({
        makeFlatVector<int64_t>(kChunkRows, [&](auto i) { return base + i; }),
        makeFlatVector<std::string>(
            kChunkRows, [&](auto i) { return valueAt(base + i); }),
    });
    if (chunk == 0) {
      schemaSample = rowVector;
    }
    chunkVectors.push_back(rowVector);
  }

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file = test::createNimbleFile(
      *rootPool(), chunkVectors, writerOptions, /*flushAfterWrite=*/false);

  // Assert the crafted data produced the intended on-disk encodings, since
  // natural selection is data-driven: chunks 0/1/3 = RLE<Dictionary>, chunk 2 =
  // Trivial. This guards the preserve-gating patterns below from silently
  // degrading (e.g. a chunk coming out plain Dictionary, which is always
  // convertible). c1 is column index 1.
  const auto chunkEncodings =
      getStreamChunkEncodings(file, /*columnIndex=*/1, pool());
  ASSERT_EQ(chunkEncodings.size(), static_cast<size_t>(kNumChunks));
  for (int c : {0, 1, 3}) {
    EXPECT_EQ(chunkEncodings[c].encodingType, EncodingType::RLE)
        << "chunk " << c;
    ASSERT_TRUE(chunkEncodings[c].runValuesType.has_value()) << "chunk " << c;
    EXPECT_EQ(*chunkEncodings[c].runValuesType, EncodingType::Dictionary)
        << "chunk " << c;
  }
  EXPECT_EQ(chunkEncodings[2].encodingType, EncodingType::Trivial);

  auto readType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  using E = VectorEncoding::Simple;

  // Expected qualifying string values (in output order) for a row predicate.
  auto expectedFor = [&](const std::function<bool(int)>& accept) {
    std::vector<std::string> values;
    for (int g = 0; g < kTotalRows; ++g) {
      if (accept(g)) {
        values.push_back(valueAt(g));
      }
    }
    return makeFlatVector<std::string>(
        values.size(), [&](auto i) { return values[i]; });
  };

  // Builds a ScanSpec whose c0 filter accepts the given inclusive row ranges,
  // forcing the string column to skip over the rejected rows.
  auto makeC0RangeFilter =
      [&](std::vector<std::pair<int64_t, int64_t>> accepted) {
        auto scanSpec = std::make_shared<common::ScanSpec>("root");
        scanSpec->addAllChildFields(*readType);
        std::vector<std::unique_ptr<common::BigintRange>> ranges;
        for (const auto& [lo, hi] : accepted) {
          ranges.push_back(
              std::make_unique<common::BigintRange>(
                  lo, hi, /*nullAllowed=*/false));
        }
        if (ranges.size() == 1) {
          scanSpec->childByName("c0")->setFilter(std::move(ranges[0]));
        } else {
          scanSpec->childByName("c0")->setFilter(
              std::make_unique<common::BigintMultiRange>(
                  std::move(ranges), /*nullAllowed=*/false));
        }
        return scanSpec;
      };

  // Control: contiguous per-chunk read. The Trivial chunk abandons to FLAT; the
  // RLE<Dictionary> chunks read as DICTIONARY (no skip involved).
  {
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*readType);
    auto readers =
        makeReaders(schemaSample, file, scanSpec, stringDecoderZeroCopy);
    validateWithEncodingChecks(
        *expectedFor([](int) { return true; }),
        *readers.rowReader,
        /*batchSize=*/kChunkRows,
        /*totalRows=*/kTotalRows,
        {E::DICTIONARY, E::DICTIONARY, E::FLAT, E::DICTIONARY},
        readType,
        pool(),
        /*childIndex=*/1);
  }

  // Pattern 1: full range in a single batch -> abandons mid-read at the Trivial
  // chunk -> the whole batch is FLAT.
  {
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*readType);
    auto readers =
        makeReaders(schemaSample, file, scanSpec, stringDecoderZeroCopy);
    validateWithEncodingChecks(
        *expectedFor([](int) { return true; }),
        *readers.rowReader,
        /*batchSize=*/kTotalRows,
        /*totalRows=*/kTotalRows,
        {E::FLAT},
        readType,
        pool(),
        /*childIndex=*/1);
  }

  // Pattern 2: first read stops mid the second chunk (rows [0,599],
  // dictionary), then a skip crosses the Trivial third chunk and lands in the
  // fourth chunk (rows [1200,1599]), which is read in DICTIONARY mode
  // (preserved across the skip).
  {
    auto accept = [](int g) { return g <= 599 || g >= 1200; };
    auto scanSpec = makeC0RangeFilter({{0, 599}, {1200, kTotalRows - 1}});
    auto readers =
        makeReaders(schemaSample, file, scanSpec, stringDecoderZeroCopy);
    validateWithEncodingChecks(
        *expectedFor(accept),
        *readers.rowReader,
        /*batchSize=*/600,
        /*totalRows=*/kTotalRows,
        {E::DICTIONARY, E::DICTIONARY},
        readType,
        pool(),
        /*childIndex=*/1);
  }

  // Pattern 3: skip the first chunk entirely and read the second chunk to the
  // end in one batch. The second chunk is loaded by the skip (dictionary mode),
  // then the read abandons at the Trivial third chunk -> whole batch FLAT.
  {
    auto accept = [](int g) { return g >= 400; };
    auto scanSpec = makeC0RangeFilter({{400, kTotalRows - 1}});
    auto readers =
        makeReaders(schemaSample, file, scanSpec, stringDecoderZeroCopy);
    validateWithEncodingChecks(
        *expectedFor(accept),
        *readers.rowReader,
        /*batchSize=*/kTotalRows,
        /*totalRows=*/kTotalRows,
        {E::FLAT},
        readType,
        pool(),
        /*childIndex=*/1);
  }

  // Pattern 4: first read runs through the Trivial third chunk (rows [0,899])
  // and abandons there (FLAT), turning the preserve flag off; the next read
  // skips into the fourth chunk (rows [1200,1599]), which must still be read as
  // DICTIONARY because readWithDictionary turns the flag back on before the
  // skip.
  //
  // TODO(T280373333): the fourth chunk should be DICTIONARY, but currently
  // reads FLAT. read1 ends mid the Trivial third chunk (row 900), so this
  // batch's leading rejected rows [900,1199] make the framework call
  // read(offset=900, rows=[300..699]): offset lands in the Trivial chunk while
  // the first row actually read is in the fourth chunk. The convertibility gate
  // inspects the chunk at offset (Trivial) instead of at offset + rows[0] (the
  // fourth chunk), so it spuriously abandons. This is a pre-existing gap,
  // independent of the preserve flag. Expect FLAT here until that gap is fixed,
  // then flip the second batch to DICTIONARY.
  {
    auto accept = [](int g) { return g <= 899 || g >= 1200; };
    auto scanSpec = makeC0RangeFilter({{0, 899}, {1200, kTotalRows - 1}});
    auto readers =
        makeReaders(schemaSample, file, scanSpec, stringDecoderZeroCopy);
    validateWithEncodingChecks(
        *expectedFor(accept),
        *readers.rowReader,
        /*batchSize=*/900,
        /*totalRows=*/kTotalRows,
        {E::FLAT, E::FLAT},
        readType,
        pool(),
        /*childIndex=*/1);
  }
}

// Flatmap string column read via dictionary path.
// When a string column is inside a flatmap, rows where the key is absent have
// inMap=0 (null). The encoding only has data for in-map rows. The dictionary
// path must account for these nulls when reading indices. If kHasNulls is
// ignored in readWithVisitorImpl, the encoding will try to read more
// values than it contains.
// Read batch spans chunk boundary. With multi-chunk dictionary support,
// the batch crosses the boundary and merges alphabets from both chunks.
TEST_P(StringColumnReaderTest, readAcrossChunkBoundary) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  auto chunk1 = makeRowVector({makeFlatVector<std::string>(300, [](auto i) {
    return std::vector<std::string>{"aaa", "bbb"}[i % 2];
  })});
  auto chunk2 = makeRowVector({makeFlatVector<std::string>(300, [](auto i) {
    return std::vector<std::string>{"xxx", "yyy"}[i % 2];
  })});

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file = test::createNimbleFile(
      *rootPool(), {chunk1, chunk2}, writerOptions, /*flushAfterWrite=*/false);

  auto expected = makeRowVector({makeFlatVector<std::string>(600, [](auto i) {
    if (i < 300) {
      return std::vector<std::string>{"aaa", "bbb"}[i % 2];
    }
    return std::vector<std::string>{"xxx", "yyy"}[(i - 300) % 2];
  })});

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*expected->type());
  auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
  using E = VectorEncoding::Simple;
  validateWithEncodingChecks(
      *expected->childAt(0),
      *readers.rowReader,
      200,
      /*totalRows=*/600,
      {E::DICTIONARY, E::DICTIONARY, E::DICTIONARY},
      asRowType(expected->type()),
      pool());
}

// Read across stripe boundary with pure Dictionary encoding (MC excluded).
// Each stripe has a different alphabet. The reader must rebuild dictionary
// state when crossing stripes.
TEST_P(StringColumnReaderTest, readAcrossStripeBoundary) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kStripeRows = 150;
  auto stripe1 =
      makeRowVector({makeFlatVector<std::string>(kStripeRows, [](auto i) {
        return std::vector<std::string>{"alpha", "bravo", "charlie"}[i % 3];
      })});
  auto stripe2 =
      makeRowVector({makeFlatVector<std::string>(kStripeRows, [](auto i) {
        return std::vector<std::string>{"delta", "echo"}[i % 2];
      })});

  auto readFactors =
      ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();
  std::erase_if(readFactors, [](const auto& pair) {
    return pair.first == EncodingType::MainlyConstant;
  });
  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.encodingSelectionPolicyCreator =
      [readFactors](DataType dataType) {
        return ManualEncodingSelectionPolicyFactory(readFactors)
            .createPolicy(dataType);
      };
  auto file = test::createNimbleFile(
      *rootPool(), {stripe1, stripe2}, writerOptions, /*flushAfterWrite=*/true);

  auto expected =
      makeRowVector({makeFlatVector<std::string>(kStripeRows * 2, [](auto i) {
        if (i < kStripeRows) {
          return std::vector<std::string>{"alpha", "bravo", "charlie"}[i % 3];
        }
        return std::vector<std::string>{"delta", "echo"}[(i - kStripeRows) % 2];
      })});

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*expected->type());
  auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
  using E = VectorEncoding::Simple;
  validateWithEncodingChecks(
      *expected->childAt(0),
      *readers.rowReader,
      100,
      /*totalRows=*/2 * kStripeRows,
      {E::DICTIONARY, E::DICTIONARY, E::DICTIONARY, E::DICTIONARY},
      asRowType(expected->type()),
      pool());
}

TEST_P(StringColumnReaderTest, flatMapStringDictionaryPath) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  // Write map<int8_t, varchar> as a flatmap. Key 1 is present in all rows,
  // key 2 is present in only some rows (creating inMap nulls for key 2).
  constexpr int kRows = 100;
  std::vector<int8_t> allKeys;
  std::vector<std::string> allVals;
  std::vector<vector_size_t> mapOffsets;
  std::vector<vector_size_t> mapSizes;
  for (int i = 0; i < kRows; ++i) {
    mapOffsets.push_back(allKeys.size());
    allKeys.push_back(1);
    allVals.push_back("val_" + std::to_string(i % 5));
    if (i % 3 != 0) {
      allKeys.push_back(2);
      allVals.push_back("key2_" + std::to_string(i % 4));
    }
    mapSizes.push_back(allKeys.size() - mapOffsets.back());
  }
  auto keysFv = makeFlatVector<int8_t>(allKeys);
  auto valsFv = makeFlatVector<std::string>(allVals);
  auto offBuf = allocateOffsets(kRows, pool());
  auto sizBuf = allocateSizes(kRows, pool());
  auto* rawOff = offBuf->asMutable<vector_size_t>();
  auto* rawSiz = sizBuf->asMutable<vector_size_t>();
  for (int i = 0; i < kRows; ++i) {
    rawOff[i] = mapOffsets[i];
    rawSiz[i] = mapSizes[i];
  }
  auto input = makeRowVector({std::make_shared<MapVector>(
      pool(),
      MAP(TINYINT(), VARCHAR()),
      nullptr,
      kRows,
      offBuf,
      sizBuf,
      keysFv,
      valsFv)});

  WriterOptions writerOptions;
  writerOptions.flatMapColumns = {{"c0", {}}};
  auto file = test::createNimbleFile(*rootPool(), input, writerOptions);

  // Read back as struct with flatmap-as-struct projection.
  auto outType = ROW({"c0"}, {ROW({"1", "2"}, {VARCHAR(), VARCHAR()})});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outType);
  scanSpec->childByName("c0")->setFlatMapAsStruct(true);

  auto readFile = std::make_shared<InMemoryReadFile>(file);
  auto factory =
      dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
  dwio::common::ReaderOptions options(pool());
  options.setDataIoStats(dataIoStats_);
  options.setMetadataIoStats(metadataIoStats_);
  options.setScanSpec(scanSpec);
  auto reader = factory->createReader(
      std::make_unique<dwio::common::BufferedInput>(readFile, *pool()),
      options);
  dwio::common::RowReaderOptions rowOptions;
  rowOptions.setScanSpec(scanSpec);
  rowOptions.setRequestedType(asRowType(input->type()));
  rowOptions.setStringDecoderZeroCopy(true);
  rowOptions.setNimblePreserveDictionaryEncoding(true);
  auto rowReader = reader->createRowReader(rowOptions);

  // Build expected output: key 1 always present, key 2 null when i%3==0.
  auto expected = makeRowVector({makeRowVector(
      {"1", "2"},
      {
          makeFlatVector<std::string>(
              kRows, [](auto i) { return "val_" + std::to_string(i % 5); }),
          makeFlatVector<std::string>(
              kRows,
              [](auto i) { return "key2_" + std::to_string(i % 4); },
              [](auto i) { return i % 3 == 0; }),
      })});

  auto result = BaseVector::create(outType, 0, pool());
  int numScanned = 0;
  int offset = 0;
  while (numScanned < kRows) {
    numScanned += rowReader->next(23, result);
    result->validate();
    for (int j = 0; j < result->size(); ++j) {
      ASSERT_TRUE(result->equalValueAt(expected.get(), j, offset + j))
          << "Mismatch at row " << (offset + j) << ": expected "
          << expected->toString(offset + j) << " got " << result->toString(j);
    }
    offset += result->size();
  }
  ASSERT_EQ(numScanned, kRows);
  ASSERT_EQ(0, rowReader->next(1, result));
}

// Flatmap string column read through a BARE DictionaryEncoding with EXTERNAL
// (inMap) nulls. This is the genuine regression guard for
// ChunkedDecoder::readDictionaryIndicesImpl<kHasNulls=true> handling nulls that
// originate from the flatmap inMap bitmap rather than from the encoding itself.
//
// Why a dedicated test is needed: flatMapStringDictionaryPath above uses data
// whose value stream is encoded as MainlyConstant<Dictionary>. MC's own
// readIndicesWithVisitor maps rows to values internally, so that test passes
// with OR without the kHasNulls fix and never exercises the external-null
// batching in readDictionaryIndicesImpl. Here MainlyConstant is removed from
// the writer's read factors so the value stream is encoded as a bare
// DictionaryEncoding (dictionaryEnabled() == true, no inner null handling). The
// decode then flows read() -> readWithDictionary() ->
// decoder_.readDictionaryIndices() -> readDictionaryIndicesImpl<true>, which
// must use computeNextRowIndex<kHasNulls> to skip the inMap-null rows when
// batching indices. If kHasNulls were ignored, the decoder would over-read the
// index stream and land values at the wrong (null) row positions.
//
// Key 1 is present in every row (dense, no inMap nulls). Key 2 is absent when
// i % 3 == 0, so its child column receives external inMap nulls at exactly
// those rows. The alphabets are small with no dominant value, so Dictionary is
// the natural choice for the sub-encoding once MainlyConstant is excluded.
TEST_P(
    StringColumnReaderTest,
    flatMapStringDictionaryPathExternalNullsBareDict) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  // Write map<int8_t, varchar> as a flatmap. Key 1 is present in all rows;
  // key 2 is present only when i % 3 != 0, creating inMap nulls for key 2.
  constexpr int kRows = 120;
  std::vector<int8_t> allKeys;
  std::vector<std::string> allVals;
  std::vector<vector_size_t> mapOffsets;
  std::vector<vector_size_t> mapSizes;
  // Cycle through small alphabets with no single dominant value so the value
  // streams are encoded as a bare Dictionary (not Trivial or MainlyConstant).
  const std::vector<std::string> key1Alphabet = {
      "alpha", "bravo", "charlie", "delta"};
  const std::vector<std::string> key2Alphabet = {"echo", "foxtrot", "golf"};
  for (int i = 0; i < kRows; ++i) {
    mapOffsets.push_back(static_cast<vector_size_t>(allKeys.size()));
    allKeys.push_back(1);
    allVals.push_back(key1Alphabet[i % key1Alphabet.size()]);
    if (i % 3 != 0) {
      allKeys.push_back(2);
      allVals.push_back(key2Alphabet[i % key2Alphabet.size()]);
    }
    mapSizes.push_back(
        static_cast<vector_size_t>(allKeys.size() - mapOffsets.back()));
  }
  auto keysFv = makeFlatVector<int8_t>(allKeys);
  auto valsFv = makeFlatVector<std::string>(allVals);
  auto offBuf = allocateOffsets(kRows, pool());
  auto sizBuf = allocateSizes(kRows, pool());
  auto* rawOff = offBuf->asMutable<vector_size_t>();
  auto* rawSiz = sizBuf->asMutable<vector_size_t>();
  for (int i = 0; i < kRows; ++i) {
    rawOff[i] = mapOffsets[i];
    rawSiz[i] = mapSizes[i];
  }
  auto input = makeRowVector({std::make_shared<MapVector>(
      pool(),
      MAP(TINYINT(), VARCHAR()),
      nullptr,
      kRows,
      offBuf,
      sizBuf,
      keysFv,
      valsFv)});

  // Exclude MainlyConstant so the flatmap value streams pick a bare Dictionary
  // encoding, forcing the read through readDictionaryIndicesImpl rather than
  // MC's internal row->value mapping.
  auto readFactors =
      ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();
  std::erase_if(readFactors, [](const auto& pair) {
    return pair.first == EncodingType::MainlyConstant;
  });
  WriterOptions writerOptions;
  writerOptions.flatMapColumns = {{"c0", {}}};
  writerOptions.encodingSelectionPolicyCreator =
      [readFactors](DataType dataType) {
        return ManualEncodingSelectionPolicyFactory(readFactors)
            .createPolicy(dataType);
      };
  auto file = test::createNimbleFile(*rootPool(), input, writerOptions);

  // Read back as struct with flatmap-as-struct projection.
  auto outType = ROW({"c0"}, {ROW({"1", "2"}, {VARCHAR(), VARCHAR()})});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outType);
  scanSpec->childByName("c0")->setFlatMapAsStruct(true);

  auto readFile = std::make_shared<InMemoryReadFile>(file);
  auto factory =
      dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
  dwio::common::ReaderOptions options(pool());
  options.setDataIoStats(dataIoStats_);
  options.setMetadataIoStats(metadataIoStats_);
  options.setScanSpec(scanSpec);
  auto reader = factory->createReader(
      std::make_unique<dwio::common::BufferedInput>(readFile, *pool()),
      options);
  dwio::common::RowReaderOptions rowOptions;
  rowOptions.setScanSpec(scanSpec);
  rowOptions.setRequestedType(asRowType(input->type()));
  rowOptions.setStringDecoderZeroCopy(true);
  rowOptions.setNimblePreserveDictionaryEncoding(true);
  auto rowReader = reader->createRowReader(rowOptions);

  // Build expected output: key 1 always present, key 2 null when i % 3 == 0.
  auto expected = makeRowVector({makeRowVector(
      {"1", "2"},
      {
          makeFlatVector<std::string>(
              kRows,
              [&](auto i) { return key1Alphabet[i % key1Alphabet.size()]; }),
          makeFlatVector<std::string>(
              kRows,
              [&](auto i) { return key2Alphabet[i % key2Alphabet.size()]; },
              [](auto i) { return i % 3 == 0; }),
      })});

  using E = VectorEncoding::Simple;
  auto result = BaseVector::create(outType, 0, pool());
  int numScanned = 0;
  int offset = 0;
  while (numScanned < kRows) {
    // Batch size 23 forces several batches so the external-null batching runs
    // repeatedly within the chunk.
    numScanned += rowReader->next(23, result);
    result->validate();
    if (result->size() == 0) {
      continue;
    }
    // Assert the key-2 child (the one with external inMap nulls) decoded into a
    // DictionaryVector, proving the bare-dictionary index path was taken.
    // Load through any lazy/wrapping layers before reaching the nested struct.
    auto c0 = BaseVector::loadedVectorShared(
        result->asUnchecked<RowVector>()->childAt(0));
    auto key2Child = BaseVector::loadedVectorShared(
        c0->asUnchecked<RowVector>()->childAt(1));
    EXPECT_EQ(key2Child->encoding(), E::DICTIONARY)
        << "key 2 child should be a bare DictionaryVector at output offset "
        << offset;
    for (int j = 0; j < result->size(); ++j) {
      ASSERT_TRUE(result->equalValueAt(expected.get(), j, offset + j))
          << "Mismatch at row " << (offset + j) << ": expected "
          << expected->toString(offset + j) << " got " << result->toString(j);
    }
    offset += result->size();
  }
  ASSERT_EQ(numScanned, kRows);
  ASSERT_EQ(0, rowReader->next(1, result));
}

// Multi-chunk flatmap string dictionary path. Tests kHasNulls=true dict
// index reading across chunk boundaries with inMap nulls. Key 2 is absent
// in some rows (inMap=0), and the batch spans two chunks.
TEST_P(StringColumnReaderTest, flatMapStringDictionaryPathMultiChunk) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kRowsPerChunk = 200;
  constexpr int kTotalRows = 2 * kRowsPerChunk;

  auto buildMapVector = [&](int startRow, int numRows) {
    std::vector<int8_t> keys;
    std::vector<std::string> vals;
    std::vector<vector_size_t> offsets;
    std::vector<vector_size_t> sizes;
    for (int i = 0; i < numRows; ++i) {
      int globalRow = startRow + i;
      offsets.push_back(keys.size());
      keys.push_back(1);
      vals.push_back("val_" + std::to_string(globalRow % 5));
      if (globalRow % 3 != 0) {
        keys.push_back(2);
        vals.push_back("key2_" + std::to_string(globalRow % 4));
      }
      sizes.push_back(keys.size() - offsets.back());
    }
    auto keysFv = makeFlatVector<int8_t>(keys);
    auto valsFv = makeFlatVector<std::string>(vals);
    auto offBuf = allocateOffsets(numRows, pool());
    auto sizBuf = allocateSizes(numRows, pool());
    auto* rawOff = offBuf->asMutable<vector_size_t>();
    auto* rawSiz = sizBuf->asMutable<vector_size_t>();
    for (int i = 0; i < numRows; ++i) {
      rawOff[i] = offsets[i];
      rawSiz[i] = sizes[i];
    }
    return makeRowVector({std::make_shared<MapVector>(
        pool(),
        MAP(TINYINT(), VARCHAR()),
        nullptr,
        numRows,
        offBuf,
        sizBuf,
        keysFv,
        valsFv)});
  };

  auto chunk1 = buildMapVector(0, kRowsPerChunk);
  auto chunk2 = buildMapVector(kRowsPerChunk, kRowsPerChunk);

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  writerOptions.flatMapColumns = {{"c0", {}}};
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file = test::createNimbleFile(
      *rootPool(), {chunk1, chunk2}, writerOptions, /*flushAfterWrite=*/false);

  auto outType = ROW({"c0"}, {ROW({"1", "2"}, {VARCHAR(), VARCHAR()})});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outType);
  scanSpec->childByName("c0")->setFlatMapAsStruct(true);

  auto readFile = std::make_shared<InMemoryReadFile>(file);
  auto factory =
      dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
  dwio::common::ReaderOptions options(pool());
  options.setDataIoStats(dataIoStats_);
  options.setMetadataIoStats(metadataIoStats_);
  options.setScanSpec(scanSpec);
  auto reader = factory->createReader(
      std::make_unique<dwio::common::BufferedInput>(readFile, *pool()),
      options);
  dwio::common::RowReaderOptions rowOptions;
  rowOptions.setScanSpec(scanSpec);
  rowOptions.setRequestedType(asRowType(chunk1->type()));
  rowOptions.setStringDecoderZeroCopy(true);
  rowOptions.setNimblePreserveDictionaryEncoding(true);
  auto rowReader = reader->createRowReader(rowOptions);

  auto expected = makeRowVector({makeRowVector(
      {"1", "2"},
      {
          makeFlatVector<std::string>(
              kTotalRows,
              [](auto i) { return "val_" + std::to_string(i % 5); }),
          makeFlatVector<std::string>(
              kTotalRows,
              [](auto i) { return "key2_" + std::to_string(i % 4); },
              [](auto i) { return i % 3 == 0; }),
      })});

  // Batch size 300 spans the chunk boundary at row 200.
  auto result = BaseVector::create(outType, 0, pool());
  int numScanned = 0;
  int offset = 0;
  while (numScanned < kTotalRows) {
    numScanned += rowReader->next(300, result);
    result->validate();
    for (int j = 0; j < result->size(); ++j) {
      ASSERT_TRUE(result->equalValueAt(expected.get(), j, offset + j))
          << "Mismatch at row " << (offset + j);
    }
    offset += result->size();
  }
  ASSERT_EQ(numScanned, kTotalRows);
  ASSERT_EQ(0, rowReader->next(1, result));
}

// Mainly-constant string column with chunking. The data shape naturally
// triggers MainlyConstant→Dictionary encoding: a long dominant common value
// with a small repeating "other" alphabet. The read batch spans multiple
// chunks, exercising readIndicesWithVisitor across chunk boundaries in
// ChunkedDecoder's loop.
TEST_P(StringColumnReaderTest, mainlyConstantDictionaryMultiChunk) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }
  const int numRows = 10000;
  // ~95% common value, ~5% cycling through 3 other values.
  // The long common string makes MC encoding strongly preferred over plain
  // Dictionary. The repeating small "other" alphabet ensures Dictionary is
  // selected for the sub-encoding.
  const std::string commonValue(50, 'x');
  const std::string otherValues[] = {"alpha", "bravo", "charlie"};
  auto input = makeRowVector({
      makeFlatVector<std::string>(
          numRows,
          [&](auto i) {
            return i % 20 == 0 ? otherValues[i % 3] : commonValue;
          }),
  });

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  // Chunk every other write: 1000 rows per chunk.
  auto writeCount = std::make_shared<int>(0);
  writerOptions.flushPolicyFactory = [writeCount] {
    return std::make_unique<LambdaFlushPolicy>(
        [](const StripeProgress&) { return false; },
        [writeCount](const StripeProgress&) {
          return ++(*writeCount) % 2 == 0;
        });
  };

  // Write 500 rows per batch → chunks of 1000 rows (2 writes per chunk).
  // Each chunk has ~50 "other" values (3 distinct, ~17 repeats each),
  // enough for Dictionary to be selected for the sub-encoding.
  std::string file;
  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    Writer writer(
        input->type(),
        std::move(writeFile),
        *rootPool(),
        std::move(writerOptions));
    const int batchWriteSize = 500;
    for (int i = 0; i < numRows; i += batchWriteSize) {
      writer.write(input->slice(i, std::min(batchWriteSize, numRows - i)));
    }
    writer.close();
  }

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
  using E = VectorEncoding::Simple;
  validateWithEncodingChecks(
      *input->childAt(0),
      *readers.rowReader,
      /*batchSize=*/5000,
      /*totalRows=*/numRows,
      {E::DICTIONARY, E::DICTIONARY},
      asRowType(input->type()),
      pool());
}

// Same as above but with nulls.
TEST_P(StringColumnReaderTest, mainlyConstantDictionaryMultiChunkWithNulls) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }
  const int numRows = 10000;
  const std::string commonValue(50, 'x');
  const std::string otherValues[] = {"alpha", "bravo", "charlie"};
  auto input = makeRowVector({
      makeFlatVector<std::string>(
          numRows,
          [&](auto i) {
            return i % 20 == 0 ? otherValues[i % 3] : commonValue;
          },
          nullEvery(7)),
  });

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  auto writeCount = std::make_shared<int>(0);
  writerOptions.flushPolicyFactory = [writeCount] {
    return std::make_unique<LambdaFlushPolicy>(
        [](const StripeProgress&) { return false; },
        [writeCount](const StripeProgress&) {
          return ++(*writeCount) % 2 == 0;
        });
  };

  std::string file;
  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    Writer writer(
        input->type(),
        std::move(writeFile),
        *rootPool(),
        std::move(writerOptions));
    const int batchWriteSize = 500;
    for (int i = 0; i < numRows; i += batchWriteSize) {
      writer.write(input->slice(i, std::min(batchWriteSize, numRows - i)));
    }
    writer.close();
  }

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
  using E = VectorEncoding::Simple;
  validateWithEncodingChecks(
      *input->childAt(0),
      *readers.rowReader,
      /*batchSize=*/5000,
      /*totalRows=*/numRows,
      {E::DICTIONARY, E::DICTIONARY},
      asRowType(input->type()),
      pool());
}

// Basic mainly-constant string read: ~95% common value with a small other
// alphabet. Naturally triggers Nullable<MainlyConstant<Dictionary>> encoding.
TEST_P(StringColumnReaderTest, mainlyConstantDictionary) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kRows = 500;
  const std::string commonValue(50, 'x');
  const std::string otherValues[] = {"alpha", "bravo", "charlie"};
  auto input = makeRowVector({
      makeFlatVector<std::string>(
          kRows,
          [&](auto i) {
            return i % 20 == 0 ? otherValues[i % 3] : commonValue;
          }),
  });

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validateWithEncodingChecks(
      *input->childAt(0),
      *readers.rowReader,
      kRows,
      /*totalRows=*/kRows,
      {VectorEncoding::Simple::DICTIONARY},
      asRowType(input->type()),
      pool());
}

// Mainly-constant string with nulls.
TEST_P(StringColumnReaderTest, mainlyConstantDictionaryWithNulls) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kRows = 500;
  const std::string commonValue(50, 'x');
  const std::string otherValues[] = {"alpha", "bravo", "charlie"};
  auto input = makeRowVector({
      makeFlatVector<std::string>(
          kRows,
          [&](auto i) {
            return i % 20 == 0 ? otherValues[i % 3] : commonValue;
          },
          nullEvery(7)),
  });

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validateWithEncodingChecks(
      *input->childAt(0),
      *readers.rowReader,
      kRows,
      /*totalRows=*/kRows,
      {VectorEncoding::Simple::DICTIONARY},
      asRowType(input->type()),
      pool());
}

// Mainly-constant across stripe boundaries: two stripes with different
// common values and different "other" alphabets.
TEST_P(StringColumnReaderTest, mainlyConstantDictionaryAcrossStripes) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kStripeRows = 200;
  const std::string common1(50, 'x');
  const std::string common2(50, 'y');
  const std::string others1[] = {"alpha", "bravo", "charlie"};
  const std::string others2[] = {"delta", "echo", "foxtrot"};
  auto stripe1 = makeRowVector({
      makeFlatVector<std::string>(
          kStripeRows,
          [&](auto i) { return i % 20 == 0 ? others1[i % 3] : common1; }),
  });
  auto stripe2 = makeRowVector({
      makeFlatVector<std::string>(
          kStripeRows,
          [&](auto i) { return i % 20 == 0 ? others2[i % 3] : common2; }),
  });

  // Force MC→Dict via encoding layout tree. The writer wraps in
  // Nullable automatically for nullable columns.
  EncodingLayout dictLayout{
      EncodingType::Dictionary,
      {},
      CompressionType::Uncompressed,
      {std::nullopt, std::nullopt}};
  EncodingLayout mainlyConstLayout{
      EncodingType::MainlyConstant,
      {},
      CompressionType::Uncompressed,
      {std::nullopt, std::move(dictLayout)}};
  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.encodingLayoutTree.emplace(
      Kind::Row,
      std::
          unordered_map<EncodingLayoutTree::StreamIdentifier, EncodingLayout>{},
      "",
      std::vector<EncodingLayoutTree>{EncodingLayoutTree{
          Kind::Scalar, {{0, std::move(mainlyConstLayout)}}, "c0"}});
  auto file = test::createNimbleFile(
      *rootPool(),
      {stripe1, stripe2},
      writerOptions,
      /*flushAfterWrite=*/true);

  auto expected = makeRowVector({
      makeFlatVector<std::string>(
          kStripeRows * 2,
          [&](auto i) {
            if (i < kStripeRows) {
              return i % 20 == 0 ? others1[i % 3] : common1;
            }
            auto j = i - kStripeRows;
            return j % 20 == 0 ? others2[j % 3] : common2;
          }),
  });

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*expected->type());
  auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
  using E = VectorEncoding::Simple;
  validateWithEncodingChecks(
      *expected->childAt(0),
      *readers.rowReader,
      100,
      /*totalRows=*/2 * kStripeRows,
      {E::DICTIONARY, E::DICTIONARY, E::DICTIONARY, E::DICTIONARY},
      asRowType(expected->type()),
      pool());
}

// Regression for an out-of-bounds write in MainlyConstant::materializeIndices
// when a read range has zero dense non-null rows (an all-null batch). nwords(0)
// is 0, so the isCommon[numWords - 1] tail write underflowed to isCommon[-1].
// The column is forced to MainlyConstant; the first half is non-null (mainly a
// common value) and the second half is all null, so the second read batch has
// no non-null rows and calls materializeIndices(0).
TEST_P(StringColumnReaderTest, mainlyConstantAllNullReadRange) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kRows = 200;
  constexpr int kBatchSize = 100;
  const std::string commonValue(20, 'c');
  const std::string otherValue(20, 'o');

  // Rows [0, kBatchSize): non-null, mainly the common value (a few others) so
  // MainlyConstant applies. Rows [kBatchSize, kRows): all null, making the
  // second read batch a zero-non-null (all-null) range.
  auto input = makeRowVector({makeFlatVector<std::string>(
      kRows,
      [&](auto i) { return i % 10 == 0 ? otherValue : commonValue; },
      [&](auto i) { return i >= kBatchSize; })});

  // Force MainlyConstant for the string column; the writer wraps it in Nullable
  // for the null rows.
  EncodingLayout dictLayout{
      EncodingType::Dictionary,
      {},
      CompressionType::Uncompressed,
      {std::nullopt, std::nullopt}};
  EncodingLayout mainlyConstLayout{
      EncodingType::MainlyConstant,
      {},
      CompressionType::Uncompressed,
      {std::nullopt, std::move(dictLayout)}};
  WriterOptions writerOptions;
  writerOptions.encodingLayoutTree.emplace(
      Kind::Row,
      std::
          unordered_map<EncodingLayoutTree::StreamIdentifier, EncodingLayout>{},
      "",
      std::vector<EncodingLayoutTree>{EncodingLayoutTree{
          Kind::Scalar, {{0, std::move(mainlyConstLayout)}}, "c0"}});
  auto file = test::createNimbleFile(*rootPool(), input, writerOptions);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
  validate(*input, *readers.rowReader, kBatchSize);
}

// Repro for Bug #3: an IsNull filter on a dictionary-encoded string column that
// is ALSO projected (keepValues=true). Null rows pass the IsNull filter and
// must be emitted as null values. The dictionary filter+extract path must
// produce one output row per null row, not drop them.
TEST_P(StringColumnReaderTest, dictionaryIsNullFilterProjected) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  const int numRows = 100;
  // Small alphabet so dictionary encoding applies; null every 3rd row.
  auto input = makeRowVector({
      makeFlatVector<std::string>(
          numRows,
          [](auto i) { return "v_" + std::to_string(i % 5); },
          nullEvery(3)),
  });

  // Force plain Dictionary encoding for the string column; the writer wraps it
  // in Nullable for the null rows (Nullable -> Dictionary).
  EncodingLayout dictLayout{
      EncodingType::Dictionary,
      {},
      CompressionType::Uncompressed,
      {std::nullopt, std::nullopt}};
  WriterOptions writerOptions;
  writerOptions.encodingLayoutTree.emplace(
      Kind::Row,
      std::
          unordered_map<EncodingLayoutTree::StreamIdentifier, EncodingLayout>{},
      "",
      std::vector<EncodingLayoutTree>{EncodingLayoutTree{
          Kind::Scalar, {{0, std::move(dictLayout)}}, "c0"}});
  auto file = test::createNimbleFile(*rootPool(), input, writerOptions);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(std::make_unique<common::IsNull>());

  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
  // IsNull passes exactly the null rows (every 3rd).
  validateWithFilter(
      *input, *readers.rowReader, 23, [](auto i) { return i % 3 == 0; });
}

// Regression for the stale-resultNulls_ bug in filterDictionaryIndices ("3c").
// A no-null read at a SPARSE (non-contiguous) row set must not inspect a
// resultNulls_ left dirty by an earlier null-containing batch. resultNulls_ for
// non-dense reads is populated lazily (only when a null is actually hit) and is
// not reset per read, so an earlier null batch can leave it stale. If
// filterDictionaryIndices reads it unconditionally, extractNonNullEntries
// wrongly drops rows and the wrong null-handling path runs.
//
// Setup: a dict-encoded string column c1 with a filter on it, plus a sibling
// integer column c0 whose filter narrows the row set to a sparse set within
// each batch. The first batch covers rows [0,99] where c1 has nulls (dirtying
// resultNulls_); the second batch covers rows [100,199] where c1 has no nulls.
// The second batch is the no-null sparse read whose filterDictionaryIndices
// must not be misled by the stale buffer.
TEST_P(StringColumnReaderTest, dictionaryFilterSparseNoNullsAfterNullBatch) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kRows = 200;
  // c1: small alphabet so Dictionary applies. Nulls only in the first half
  // (rows [0,99], every 7th); the second half (rows [100,199]) is all non-null.
  auto c0 = makeFlatVector<int64_t>(kRows, [](auto i) { return i; });
  auto c1 = makeFlatVector<std::string>(
      kRows,
      [](auto i) { return "v_" + std::to_string(i % 4); },
      [](auto i) { return i < 100 && i % 7 == 0; });
  auto input = makeRowVector({c0, c1});

  // Force plain Dictionary encoding for c1; the writer wraps it in Nullable for
  // the first-half null rows (Nullable -> Dictionary).
  EncodingLayout dictLayout{
      EncodingType::Dictionary,
      {},
      CompressionType::Uncompressed,
      {std::nullopt, std::nullopt}};
  WriterOptions writerOptions;
  writerOptions.encodingLayoutTree.emplace(
      Kind::Row,
      std::
          unordered_map<EncodingLayoutTree::StreamIdentifier, EncodingLayout>{},
      "",
      std::vector<EncodingLayoutTree>{
          EncodingLayoutTree{Kind::Scalar, {}, "c0"},
          EncodingLayoutTree{
              Kind::Scalar, {{0, std::move(dictLayout)}}, "c1"}});
  auto file = test::createNimbleFile(*rootPool(), input, writerOptions);

  // Sibling filter on c0 keeps only even rows -> a sparse (non-contiguous) row
  // set reaches c1 in every batch. Filter on c1 accepts a subset of the
  // alphabet so filterDictionaryIndices runs on the sparse rows.
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintValuesUsingBitmask>(
          0,
          kRows,
          [] {
            std::vector<int64_t> evens;
            for (int i = 0; i < kRows; i += 2) {
              evens.push_back(i);
            }
            return evens;
          }(),
          /*nullAllowed=*/false));
  scanSpec->childByName("c1")->setFilter(
      std::make_unique<common::BytesValues>(
          std::vector<std::string>{"v_1", "v_3"}, false));

  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
  // Batch size 100 splits into batch 1 = rows [0,99] (c1 has nulls) and
  // batch 2 = rows [100,199] (c1 no nulls, sparse). A row survives only when
  // c0 is even, c1 is non-null, and c1 in {"v_1","v_3"}.
  validateWithFilter(*input, *readers.rowReader, 100, [](int i) {
    if (i % 2 != 0) {
      return false;
    }
    if (i < 100 && i % 7 == 0) {
      return false;
    }
    auto v = i % 4;
    return v == 1 || v == 3;
  });
}

// IS NULL filter on a nullable dict-encoded string column read at a SPARSE row
// set. dictionaryIsNullFilterProjected covers the DENSE case; this exercises
// filterDictionaryIndices' null-accepting row-ordered merge (path 3) under a
// non-contiguous row set produced by a sibling column's filter. The surviving
// (null) rows and their row mapping must be correct.
TEST_P(StringColumnReaderTest, dictionaryIsNullFilterSparseRows) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kRows = 200;
  auto c0 = makeFlatVector<int64_t>(kRows, [](auto i) { return i; });
  // c1: small alphabet (Dictionary applies); null every 3rd row.
  auto c1 = makeFlatVector<std::string>(
      kRows, [](auto i) { return "v_" + std::to_string(i % 5); }, nullEvery(3));
  auto input = makeRowVector({c0, c1});

  // Force plain Dictionary encoding for c1 (Nullable -> Dictionary).
  EncodingLayout dictLayout{
      EncodingType::Dictionary,
      {},
      CompressionType::Uncompressed,
      {std::nullopt, std::nullopt}};
  WriterOptions writerOptions;
  writerOptions.encodingLayoutTree.emplace(
      Kind::Row,
      std::
          unordered_map<EncodingLayoutTree::StreamIdentifier, EncodingLayout>{},
      "",
      std::vector<EncodingLayoutTree>{
          EncodingLayoutTree{Kind::Scalar, {}, "c0"},
          EncodingLayoutTree{
              Kind::Scalar, {{0, std::move(dictLayout)}}, "c1"}});
  auto file = test::createNimbleFile(*rootPool(), input, writerOptions);

  // Sibling filter on c0 keeps only even rows -> a sparse row set reaches c1.
  // IS NULL on c1 then passes exactly the null rows within that sparse set.
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintValuesUsingBitmask>(
          0,
          kRows,
          [] {
            std::vector<int64_t> evens;
            for (int i = 0; i < kRows; i += 2) {
              evens.push_back(i);
            }
            return evens;
          }(),
          /*nullAllowed=*/false));
  scanSpec->childByName("c1")->setFilter(std::make_unique<common::IsNull>());

  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
  // A row survives only when c0 is even and c1 is null (every 3rd row).
  validateWithFilter(*input, *readers.rowReader, 100, [](int i) {
    return i % 2 == 0 && i % 3 == 0;
  });
}

// Mainly-constant string column with chunking where read batches align with
// chunk boundaries. This exercises the non-fallback dictionary fast path for
// chunked layouts: each batch fits within a single chunk, so the dictionary
// state (alphabet, indices) remains valid for the entire read.
TEST_P(StringColumnReaderTest, mainlyConstantDictionaryAlignedChunks) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  const int numRows = 10000;
  const std::string commonValue(50, 'x');
  const std::string otherValues[] = {"alpha", "bravo", "charlie"};
  auto input = makeRowVector({
      makeFlatVector<std::string>(
          numRows,
          [&](auto i) {
            return i % 20 == 0 ? otherValues[i % 3] : commonValue;
          }),
  });

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  auto writeCount = std::make_shared<int>(0);
  writerOptions.flushPolicyFactory = [writeCount] {
    return std::make_unique<LambdaFlushPolicy>(
        [](const StripeProgress&) { return false; },
        [writeCount](const StripeProgress&) {
          return ++(*writeCount) % 2 == 0;
        });
  };

  std::string file;
  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    Writer writer(
        input->type(),
        std::move(writeFile),
        *rootPool(),
        std::move(writerOptions));
    const int batchWriteSize = 500;
    for (int i = 0; i < numRows; i += batchWriteSize) {
      writer.write(input->slice(i, std::min(batchWriteSize, numRows - i)));
    }
    writer.close();
  }

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
  // Batch size fits within a single chunk (1000 rows per chunk), so all
  // batches use the dictionary path. With 10 chunks of 1000 rows and
  // batch size 500, we get 20 batches — each within a single chunk.
  std::vector<VectorEncoding::Simple> expectedEncodings(
      20, VectorEncoding::Simple::DICTIONARY);
  validateWithEncodingChecks(
      *input->childAt(0),
      *readers.rowReader,
      /*batchSize=*/500,
      /*totalRows=*/numRows,
      expectedEncodings,
      asRowType(input->type()),
      pool());
}

// Same as mainlyConstantDictionaryAlignedChunks but with nulls.
TEST_P(StringColumnReaderTest, mainlyConstantDictionaryAlignedChunksWithNulls) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  const int numRows = 10000;
  const std::string commonValue(50, 'x');
  const std::string otherValues[] = {"alpha", "bravo", "charlie"};
  auto input = makeRowVector({
      makeFlatVector<std::string>(
          numRows,
          [&](auto i) {
            return i % 20 == 0 ? otherValues[i % 3] : commonValue;
          },
          nullEvery(7)),
  });

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  auto writeCount = std::make_shared<int>(0);
  writerOptions.flushPolicyFactory = [writeCount] {
    return std::make_unique<LambdaFlushPolicy>(
        [](const StripeProgress&) { return false; },
        [writeCount](const StripeProgress&) {
          return ++(*writeCount) % 2 == 0;
        });
  };

  std::string file;
  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    Writer writer(
        input->type(),
        std::move(writeFile),
        *rootPool(),
        std::move(writerOptions));
    const int batchWriteSize = 500;
    for (int i = 0; i < numRows; i += batchWriteSize) {
      writer.write(input->slice(i, std::min(batchWriteSize, numRows - i)));
    }
    writer.close();
  }

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
  std::vector<VectorEncoding::Simple> expectedEncodings(
      20, VectorEncoding::Simple::DICTIONARY);
  validateWithEncodingChecks(
      *input->childAt(0),
      *readers.rowReader,
      /*batchSize=*/500,
      /*totalRows=*/numRows,
      expectedEncodings,
      asRowType(input->type()),
      pool());
}

// RLE→Dictionary string column returns DictionaryVector. Uses forced
// encoding layout to create RLE wrapping Dictionary for string values
// with repeated patterns (natural RLE runs).
TEST_P(StringColumnReaderTest, rleDictionaryVector) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kStripeRows = 300;
  const std::string values[] = {"alpha", "bravo", "charlie"};
  // Create data with repeated runs to produce natural RLE encoding.
  auto data = makeRowVector({
      makeFlatVector<std::string>(
          kStripeRows, [&](auto i) { return values[(i / 10) % 3]; }),
  });

  // Force RLE→Dictionary encoding layout.
  EncodingLayout dictLayout{
      EncodingType::Dictionary,
      {},
      CompressionType::Uncompressed,
      {std::nullopt, std::nullopt}};
  EncodingLayout rleLayout{
      EncodingType::RLE,
      {},
      CompressionType::Uncompressed,
      {std::nullopt, std::move(dictLayout)}};
  WriterOptions writerOptions;
  writerOptions.encodingLayoutTree.emplace(
      Kind::Row,
      std::
          unordered_map<EncodingLayoutTree::StreamIdentifier, EncodingLayout>{},
      "",
      std::vector<EncodingLayoutTree>{
          EncodingLayoutTree{Kind::Scalar, {{0, std::move(rleLayout)}}, "c0"}});
  auto file = test::createNimbleFile(
      *rootPool(), std::vector<VectorPtr>{data}, writerOptions);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*data->type());
  auto readers = makeReaders(data, file, scanSpec, stringDecoderZeroCopy);
  using E = VectorEncoding::Simple;
  validateWithEncodingChecks(
      *data->childAt(0),
      *readers.rowReader,
      100,
      /*totalRows=*/kStripeRows,
      {E::DICTIONARY, E::DICTIONARY, E::DICTIONARY},
      asRowType(data->type()),
      pool());
}

// Forces column "c0" to encode as RLE wrapping Dictionary, so repeated-value
// runs become RLE runs of dictionary indices (mirrors rleDictionaryVector).
EncodingLayoutTree makeRleDictionaryLayoutTree() {
  EncodingLayout dictLayout{
      EncodingType::Dictionary,
      {},
      CompressionType::Uncompressed,
      {std::nullopt, std::nullopt}};
  EncodingLayout rleLayout{
      EncodingType::RLE,
      {},
      CompressionType::Uncompressed,
      {std::nullopt, std::move(dictLayout)}};
  return EncodingLayoutTree{
      Kind::Row,
      std::
          unordered_map<EncodingLayoutTree::StreamIdentifier, EncodingLayout>{},
      "",
      std::vector<EncodingLayoutTree>{
          EncodingLayoutTree{Kind::Scalar, {{0, std::move(rleLayout)}}, "c0"}}};
}

// Regression for the RLE dict-index mid-run resume bug: when a chunk read is
// split across next() batches and the split lands inside an RLE run,
// RLEEncoding::materializeIndices is re-entered with copiesRemaining_ > 0 and
// must reuse the in-progress run's dictionary index (member currentIndex_)
// rather than restart at 0. The run length (50) does not divide the batch size
// (75), so a run straddles a next() boundary and forces the resume path.
TEST_P(StringColumnReaderTest, rleDictionaryMidRunBatchResume) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kRows = 300;
  constexpr int kRunLength = 50;
  const std::string values[] = {"alpha", "bravo", "charlie"};
  auto data = makeRowVector({makeFlatVector<std::string>(
      kRows, [&](auto i) { return values[(i / kRunLength) % 3]; })});

  WriterOptions writerOptions;
  writerOptions.encodingLayoutTree.emplace(makeRleDictionaryLayoutTree());
  auto file = test::createNimbleFile(
      *rootPool(), std::vector<VectorPtr>{data}, writerOptions);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*data->type());
  auto readers = makeReaders(data, file, scanSpec, stringDecoderZeroCopy);
  validate(*data, *readers.rowReader, /*batchSize=*/75);
}

// Same mid-run resume as above but with nulls interspersed — the shape of the
// original residue-dependent failure (seed 1031254216). The inner RLE runs
// cover the dense non-null values, which must still resume correctly mid-run.
TEST_P(StringColumnReaderTest, rleDictionaryMidRunBatchResumeWithNulls) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kRows = 300;
  constexpr int kRunLength = 50;
  const std::string values[] = {"alpha", "bravo", "charlie"};
  std::vector<std::optional<std::string>> raw(kRows);
  for (int i = 0; i < kRows; ++i) {
    raw[i] = (i % 7 == 0)
        ? std::nullopt
        : std::optional<std::string>(values[(i / kRunLength) % 3]);
  }
  auto data = makeRowVector({makeNullableFlatVector<std::string>(raw)});

  WriterOptions writerOptions;
  writerOptions.encodingLayoutTree.emplace(makeRleDictionaryLayoutTree());
  auto file = test::createNimbleFile(
      *rootPool(), std::vector<VectorPtr>{data}, writerOptions);

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*data->type());
  auto readers = makeReaders(data, file, scanSpec, stringDecoderZeroCopy);
  validate(*data, *readers.rowReader, /*batchSize=*/75);
}

// Mid-run resume in the production multi-chunk shape: two RLE→Dictionary chunks
// within a stripe, read with a batch size (70) that both splits runs mid-run
// inside a chunk and produces a batch spanning the chunk boundary.
TEST_P(StringColumnReaderTest, rleDictionaryMidRunResumeAcrossChunk) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kChunkRows = 300;
  constexpr int kRunLength = 50;
  const std::string chunk1Values[] = {"alpha", "bravo", "charlie"};
  const std::string chunk2Values[] = {"delta", "echo", "foxtrot"};
  auto chunk1 = makeRowVector({makeFlatVector<std::string>(
      kChunkRows, [&](auto i) { return chunk1Values[(i / kRunLength) % 3]; })});
  auto chunk2 = makeRowVector({makeFlatVector<std::string>(
      kChunkRows, [&](auto i) { return chunk2Values[(i / kRunLength) % 3]; })});

  WriterOptions writerOptions;
  writerOptions.encodingLayoutTree.emplace(makeRleDictionaryLayoutTree());
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file = test::createNimbleFile(
      *rootPool(), {chunk1, chunk2}, writerOptions, /*flushAfterWrite=*/false);

  auto expected =
      makeRowVector({makeFlatVector<std::string>(2 * kChunkRows, [&](auto i) {
        return i < kChunkRows
            ? chunk1Values[(i / kRunLength) % 3]
            : chunk2Values[((i - kChunkRows) / kRunLength) % 3];
      })});

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*expected->type());
  auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
  validate(*expected, *readers.rowReader, /*batchSize=*/70);
}

// Constant string encoding — all values identical.
TEST_P(StringColumnReaderTest, constantDictionary) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kRows = 200;
  const std::string constantValue = "always_the_same";
  auto input = makeRowVector({
      makeFlatVector<std::string>(
          kRows, [&](auto /*i*/) { return constantValue; }),
  });

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validateWithEncodingChecks(
      *input->childAt(0),
      *readers.rowReader,
      kRows,
      /*totalRows=*/kRows,
      {VectorEncoding::Simple::DICTIONARY},
      asRowType(input->type()),
      pool());
}

// Constant string with nulls — constant non-null value, some rows null.
TEST_P(StringColumnReaderTest, constantDictionaryWithNulls) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kRows = 200;
  const std::string constantValue = "always_the_same";
  auto input = makeRowVector({
      makeFlatVector<std::string>(
          kRows,
          [&](auto /*i*/) { return constantValue; },
          velox::test::VectorMaker::nullEvery(7)),
  });

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);
  validateWithEncodingChecks(
      *input->childAt(0),
      *readers.rowReader,
      kRows,
      /*totalRows=*/kRows,
      {VectorEncoding::Simple::DICTIONARY},
      asRowType(input->type()),
      pool());
}

// Regression test for filter-only (DropValues) columns spanning multiple
// chunks where the first chunk's encoding does not call addNumValues.
//
// The "key" column is filtered with projectOut=false (DropValues). The first
// chunk is forced to Constant encoding, whose readWithVisitorSlow does not
// update numValues for DropValues. The second chunk has mixed values, so the
// Constant layout does not apply and the writer falls back to an encoding
// whose bulkScan calls addNumValues -> setNumRows. Without the fix, the stale
// numValues from the Constant chunk causes setNumRows to truncate outputRows,
// silently dropping rows that passed the filter in the first chunk.
TEST_P(StringColumnReaderTest, filterOnlyMultiChunkNumValuesSync) {
  const bool stringDecoderZeroCopy = GetParam();
  const std::string targetValue = "MATCH";

  // Chunk 1: all rows match the filter -> Constant encoding.
  constexpr int kChunk1Rows = 300;
  auto chunk1 = makeRowVector(
      {"key", "value"},
      {makeFlatVector<StringView>(
           kChunk1Rows, [&](auto) { return StringView(targetValue); }),
       makeFlatVector<int64_t>(kChunk1Rows, [](auto i) { return i; })});

  // Chunk 2: runs of matching / non-matching values -> RLE encoding (has
  // bulkScan). Half the rows match.
  constexpr int kChunk2Rows = 300;
  constexpr int kRunLength = 10;
  const StringView matchView(targetValue);
  const StringView otherView("OTHER");
  auto chunk2 = makeRowVector(
      {"key", "value"},
      {makeFlatVector<StringView>(
           kChunk2Rows,
           [&](auto i) {
             return (i / kRunLength) % 2 == 0 ? matchView : otherView;
           }),
       makeFlatVector<int64_t>(
           kChunk2Rows, [](auto i) { return kChunk1Rows + i; })});

  // Force the key column to Constant encoding. Chunk 1 (all identical) uses it;
  // chunk 2 (mixed) cannot, so the writer falls back to a bulkScan encoding.
  EncodingLayout constantLayout{
      EncodingType::Constant, {}, CompressionType::Uncompressed};
  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  // Force one chunk per write() so the two vectors land in separate chunks
  // within a single stripe. A single filter-only read then spans the chunk
  // boundary, which is what triggers the numValues desync bug.
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  writerOptions.encodingLayoutTree.emplace(
      Kind::Row,
      std::
          unordered_map<EncodingLayoutTree::StreamIdentifier, EncodingLayout>{},
      "",
      std::vector<EncodingLayoutTree>{
          EncodingLayoutTree{
              Kind::Scalar, {{0, std::move(constantLayout)}}, "key"},
          EncodingLayoutTree{Kind::Scalar, {}, "value"}});

  auto file = test::createNimbleFile(
      *rootPool(),
      {chunk1, chunk2},
      writerOptions,
      /*flushAfterWrite=*/false);

  auto type = asRowType(chunk1->type());

  auto readRows = [&](bool projectOut) -> int64_t {
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*type);
    auto* keySpec = scanSpec->childByName("key");
    keySpec->setFilter(
        std::make_unique<common::BytesValues>(
            std::vector<std::string>{targetValue}, false));
    keySpec->setProjectOut(projectOut);
    auto readers = makeReaders(chunk1, file, scanSpec, stringDecoderZeroCopy);
    auto result = BaseVector::create(type, 0, pool());
    int64_t totalRows = 0;
    // Large batch so a single next() spans the chunk boundary.
    while (readers.rowReader->next(100000, result) > 0) {
      totalRows += result->size();
    }
    return totalRows;
  };

  // All of chunk 1 matches plus half of chunk 2.
  const int64_t expected = kChunk1Rows + kChunk2Rows / 2;
  ASSERT_EQ(readRows(/*projectOut=*/true), expected);
  ASSERT_EQ(readRows(/*projectOut=*/false), expected);
}

// Verifies that the dictionary alphabet is rebuilt when crossing chunk
// boundaries. Uses TestValue injection to inspect the alphabet at the
// readWithDictionary entry point.
DEBUG_ONLY_TEST_P(
    StringColumnReaderTest,
    chunkTransitionInvalidatesDictionaryState) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kBatchSize = 500;
  auto chunk1 =
      makeRowVector({makeFlatVector<std::string>(kBatchSize, [](auto i) {
        return std::vector<std::string>{"aaa", "bbb", "ccc"}[i % 3];
      })});
  auto chunk2 =
      makeRowVector({makeFlatVector<std::string>(kBatchSize, [](auto i) {
        return std::vector<std::string>{"pp", "qq", "rr", "ss", "tt"}[i % 5];
      })});

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  std::string file;
  {
    auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
    Writer writer(
        chunk1->type(),
        std::move(writeFile),
        *rootPool(),
        std::move(writerOptions));
    writer.write(chunk1);
    writer.write(chunk2);
    writer.close();
  }

  auto expected =
      makeRowVector({makeFlatVector<std::string>(kBatchSize * 2, [](auto i) {
        if (i < kBatchSize) {
          return std::vector<std::string>{"aaa", "bbb", "ccc"}[i % 3];
        }
        return std::vector<std::string>{
            "pp", "qq", "rr", "ss", "tt"}[(i - kBatchSize) % 5];
      })});

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*expected->type());
  auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);

  int dictPathEntries = 0;
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::StringColumnReader::readWithDictionary",
      std::function<void(const std::vector<std::string_view>*)>(
          [&](const std::vector<std::string_view>* alphabet) {
            std::set<std::string> entries;
            for (const auto& sv : *alphabet) {
              entries.insert(std::string(sv));
            }
            if (dictPathEntries == 0) {
              EXPECT_TRUE(entries.count("aaa"));
              EXPECT_FALSE(entries.count("pp"))
                  << "Chunk 1 alphabet contains stale chunk 2 values";
            } else {
              EXPECT_TRUE(entries.count("pp"));
              EXPECT_FALSE(entries.count("aaa"))
                  << "Chunk 2 alphabet contains stale chunk 1 values";
            }
            ++dictPathEntries;
          }));

  using E = VectorEncoding::Simple;
  validateWithEncodingChecks(
      *expected->childAt(0),
      *readers.rowReader,
      kBatchSize,
      /*totalRows=*/2 * kBatchSize,
      {E::DICTIONARY, E::DICTIONARY},
      asRowType(expected->type()),
      pool());
  EXPECT_EQ(dictPathEntries, 2);
}

// Verifies the dictionary alphabet size via TestValue injection.
DEBUG_ONLY_TEST_P(StringColumnReaderTest, dictionaryPathAlphabetSize) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kRows = 500;
  auto input = makeRowVector({makeFlatVector<std::string>(kRows, [](auto i) {
    return std::vector<std::string>{"alpha", "bravo", "charlie"}[i % 3];
  })});

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  auto readers = makeReaders(input, scanSpec, stringDecoderZeroCopy);

  int dictPathEntries = 0;
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::StringColumnReader::readWithDictionary",
      std::function<void(const std::vector<std::string_view>*)>(
          [&](const std::vector<std::string_view>* alphabet) {
            ++dictPathEntries;
            std::set<std::string> entries;
            for (const auto& sv : *alphabet) {
              entries.insert(std::string(sv));
            }
            EXPECT_TRUE(entries.count("alpha"));
            EXPECT_TRUE(entries.count("bravo"));
            EXPECT_TRUE(entries.count("charlie"));
          }));

  using E = VectorEncoding::Simple;
  validateWithEncodingChecks(
      *input->childAt(0),
      *readers.rowReader,
      kRows,
      /*totalRows=*/kRows,
      {E::DICTIONARY},
      asRowType(input->type()),
      pool());
  EXPECT_EQ(dictPathEntries, 1);
}

// Reactive fallback: chunk 1 is dictionary-encoded, chunk 2 is trivial.
// A batch spanning both chunks starts on the dict path, encounters the
// non-dict chunk at the boundary, and must expand already-read dict
// indices to flat StringViews before reading the rest as flat.
TEST_P(StringColumnReaderTest, flatEncodingFallback) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  // Chunk 1: low-cardinality → dictionary encoding.
  auto chunk1 = makeRowVector({makeFlatVector<std::string>(500, [](auto i) {
    return std::vector<std::string>{"aaa", "bbb", "ccc"}[i % 3];
  })});
  // Chunk 2: all unique → trivial encoding.
  auto chunk2 = makeRowVector({makeFlatVector<std::string>(500, [](auto i) {
    return fmt::format("unique_value_row_{}_padding_to_avoid_inline", i);
  })});

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file = test::createNimbleFile(
      *rootPool(), {chunk1, chunk2}, writerOptions, /*flushAfterWrite=*/false);

  auto expected = makeRowVector({makeFlatVector<std::string>(1000, [](auto i) {
    if (i < 500) {
      return std::vector<std::string>{"aaa", "bbb", "ccc"}[i % 3];
    }
    return fmt::format("unique_value_row_{}_padding_to_avoid_inline", i - 500);
  })});

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*expected->type());
  auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
  // Batch size 1000 spans both chunks. The reader starts with dict,
  // hits the trivial chunk, and must fall back to flat for the whole batch.
  validate(*expected, *readers.rowReader, 1000);
}

// Verifies that the reactive fallback from dict to flat preserves nulls
// correctly. When nullsInReadRange_ covers the full batch, the flat
// continuation must read from the correct offset without corrupting the
// bitmap. Regression test for a bug where shifting nullsInReadRange_ in
// place would corrupt the result nulls returned by getValues.
TEST_P(StringColumnReaderTest, flatEncodingFallbackWithNulls) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  // Chunk 1: low-cardinality with nulls → dictionary encoding.
  auto chunk1 = makeRowVector({makeFlatVector<std::string>(
      500,
      [](auto i) {
        return std::vector<std::string>{"aaa", "bbb", "ccc"}[i % 3];
      },
      velox::test::VectorMaker::nullEvery(7))});

  // Chunk 2: all unique → trivial encoding, with nulls.
  auto chunk2 = makeRowVector({makeFlatVector<std::string>(
      500,
      [](auto i) {
        return fmt::format("unique_value_row_{}_padding_to_avoid_inline", i);
      },
      velox::test::VectorMaker::nullEvery(11))});

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file = test::createNimbleFile(
      *rootPool(), {chunk1, chunk2}, writerOptions, /*flushAfterWrite=*/false);

  auto expected = makeRowVector({makeFlatVector<std::string>(
      1000,
      [](auto i) {
        if (i < 500) {
          return std::vector<std::string>{"aaa", "bbb", "ccc"}[i % 3];
        }
        return fmt::format(
            "unique_value_row_{}_padding_to_avoid_inline", i - 500);
      },
      [](auto i) -> bool {
        return i < 500 ? (i % 7 == 0) : ((i - 500) % 11 == 0);
      })});

  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*expected->type());
  auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
  validate(*expected, *readers.rowReader, 1000);
}

// Variable read range scenarios testing transitions between multi-chunk
// dict reads, mid-chunk dict reads, and flat reads.
TEST_P(StringColumnReaderTest, variableReadRangeTransitions) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  using E = VectorEncoding::Simple;

  // Scenario 1: multi-chunk dict → mid-chunk dict.
  // 3 chunks of 500 rows (MC pattern: ~95% common, ~5% other), batch 750.
  // Batch 1 (0-749): spans chunk 1 (500) + chunk 2 (250) → multi-chunk dict.
  // Batch 2 (750-1499): mid-chunk 2 (250) + chunk 3 (500) → multi-chunk dict.
  {
    const std::string commonValues[] = {
        std::string(50, 'a'), std::string(50, 'x'), std::string(50, 'm')};
    const std::string otherValues[] = {"alpha", "bravo", "charlie"};

    std::vector<velox::VectorPtr> chunks;
    chunks.reserve(3);
    for (int c = 0; c < 3; ++c) {
      chunks.push_back(
          makeRowVector({makeFlatVector<std::string>(500, [&](auto i) {
            return i % 20 == 0 ? otherValues[i % 3] : commonValues[c];
          })}));
    }

    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.minStreamChunkRawSize = 0;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    auto file = test::createNimbleFile(
        *rootPool(), chunks, writerOptions, /*flushAfterWrite=*/false);

    auto expected =
        makeRowVector({makeFlatVector<std::string>(1500, [&](auto i) {
          int c = i / 500;
          int r = i % 500;
          return r % 20 == 0 ? otherValues[r % 3] : commonValues[c];
        })});

    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*expected->type());
    auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
    validateWithEncodingChecks(
        *expected->childAt(0),
        *readers.rowReader,
        /*batchSize=*/750,
        /*totalRows=*/1500,
        {E::DICTIONARY, E::DICTIONARY},
        asRowType(expected->type()),
        pool());
  }

  // Scenario 2: single-chunk dict → mid-chunk dict (no boundary crossing).
  // 1 chunk of 1000 rows (MC pattern), batch 200.
  // All 5 batches stay within the same chunk → dict, alphabet reused.
  {
    const std::string commonValue(50, 'x');
    const std::string otherValues[] = {"alpha", "bravo", "charlie"};
    auto input = makeRowVector({makeFlatVector<std::string>(1000, [&](auto i) {
      return i % 20 == 0 ? otherValues[i % 3] : commonValue;
    })});

    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.minStreamChunkRawSize = 0;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    auto file = test::createNimbleFile(
        *rootPool(), input, writerOptions, /*flushAfterWrite=*/false);

    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
    validateWithEncodingChecks(
        *input->childAt(0),
        *readers.rowReader,
        /*batchSize=*/200,
        /*totalRows=*/1000,
        {E::DICTIONARY,
         E::DICTIONARY,
         E::DICTIONARY,
         E::DICTIONARY,
         E::DICTIONARY},
        asRowType(input->type()),
        pool());
  }

  // Scenario 3: flat multi-chunk read (non-dict chunks) → mid-chunk dict read.
  // Chunk 1: unique strings (trivial encoding). Chunk 2: low-cardinality
  // (dict). Batch 1 (0-399): spans chunk 1 (300) + chunk 2 (100) — chunk 1 is
  // non-dict,
  //   so the dict path fails at the start and falls to flat for the whole
  //   batch.
  // Batch 2 (400-599): mid-chunk 2, dict-compatible → dict path.
  {
    auto chunk1 = makeRowVector({makeFlatVector<std::string>(300, [](auto i) {
      return fmt::format("unique_row_{}_padding_to_avoid_inline", i);
    })});
    const std::string commonValue(50, 'x');
    const std::string otherValues[] = {"alpha", "bravo", "charlie"};
    auto chunk2 = makeRowVector({makeFlatVector<std::string>(300, [&](auto i) {
      return i % 20 == 0 ? otherValues[i % 3] : commonValue;
    })});

    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.minStreamChunkRawSize = 0;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    auto file = test::createNimbleFile(
        *rootPool(),
        {chunk1, chunk2},
        writerOptions,
        /*flushAfterWrite=*/false);

    auto expected =
        makeRowVector({makeFlatVector<std::string>(600, [&](auto i) {
          if (i < 300) {
            return fmt::format("unique_row_{}_padding_to_avoid_inline", i);
          }
          auto r = i - 300;
          return r % 20 == 0 ? otherValues[r % 3] : commonValue;
        })});

    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*expected->type());
    auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
    // Batch 1: flat (chunk 1 is trivial). Batch 2: dict (mid-chunk 2).
    validateWithEncodingChecks(
        *expected->childAt(0),
        *readers.rowReader,
        /*batchSize=*/400,
        /*totalRows=*/600,
        {E::FLAT, E::DICTIONARY},
        asRowType(expected->type()),
        pool());
  }

  // Scenario 4: multi-chunk dict → reactive fallback flat → dict recovery.
  // Chunk 1: dict (MC). Chunk 2: unique (trivial). Chunk 3: dict (MC).
  // Batch 1 (0-399): spans chunk 1 (400 of 500) → dict.
  // Batch 2 (400-799): chunk 1 (100) → chunk 2 (300). Chunk 2 non-dict
  //   → reactive fallback, batch is flat.
  // Batch 3 (800-1199): chunk 2 (200) → chunk 3 (200). Chunk 2 non-dict
  //   at start → flat.
  // Batch 4 (1200-1499): mid-chunk 3, dict → dict recovery.
  {
    const std::string commonValue1(50, 'a');
    const std::string commonValue3(50, 'm');
    const std::string otherValues[] = {"alpha", "bravo", "charlie"};

    auto chunk1 = makeRowVector({makeFlatVector<std::string>(500, [&](auto i) {
      return i % 20 == 0 ? otherValues[i % 3] : commonValue1;
    })});
    auto chunk2 = makeRowVector({makeFlatVector<std::string>(500, [](auto i) {
      return fmt::format("unique_chunk2_row_{}_pad_to_avoid_inline", i);
    })});
    auto chunk3 = makeRowVector({makeFlatVector<std::string>(500, [&](auto i) {
      return i % 20 == 0 ? otherValues[i % 3] : commonValue3;
    })});

    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.minStreamChunkRawSize = 0;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    auto file = test::createNimbleFile(
        *rootPool(),
        {chunk1, chunk2, chunk3},
        writerOptions,
        /*flushAfterWrite=*/false);

    auto expected =
        makeRowVector({makeFlatVector<std::string>(1500, [&](auto i) {
          if (i < 500) {
            return i % 20 == 0 ? otherValues[i % 3] : commonValue1;
          }
          if (i < 1000) {
            return fmt::format(
                "unique_chunk2_row_{}_pad_to_avoid_inline", i - 500);
          }
          auto r = i - 1000;
          return r % 20 == 0 ? otherValues[r % 3] : commonValue3;
        })});

    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*expected->type());
    auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
    validateWithEncodingChecks(
        *expected->childAt(0),
        *readers.rowReader,
        /*batchSize=*/400,
        /*totalRows=*/1500,
        {E::DICTIONARY, E::FLAT, E::FLAT, E::DICTIONARY},
        asRowType(expected->type()),
        pool());
  }
}

// Repro for the sparse-row dict-abandon bug. When a projected dict-encoded
// string column receives a non-dense row set (from a sibling column's
// filter) and the dictionary read is abandoned mid-batch at a non-dict
// chunk, the flat encoding fallback must preserve the dict portion's nulls
// and produce the correct remaining rows. The fallback continuation
// (readOffset_ advance and the continuation row set) assumed dense rows,
// and the flat fallback's prepareNulls cleared the dict portion's nulls,
// which the sparse path writes into resultNulls_.
TEST_P(StringColumnReaderTest, sparseRowsAcrossChunkAbandonPreservesNulls) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  // chunk1: low cardinality with nulls → Nullable→Dictionary (dict path).
  auto chunk1 = makeRowVector({
      makeFlatVector<int64_t>(300, [](auto i) { return i; }),
      makeFlatVector<std::string>(
          300,
          [](auto i) { return std::vector<std::string>{"aaa", "bbb"}[i % 2]; },
          nullEvery(7)),
  });
  // chunk2: all-unique long strings → Trivial (non-dict) → forces abandon.
  auto chunk2 = makeRowVector({
      makeFlatVector<int64_t>(300, [](auto i) { return 300 + i; }),
      makeFlatVector<std::string>(
          300,
          [](auto i) {
            return fmt::format("unique_chunk2_row_{}_pad_to_avoid_inline", i);
          }),
  });

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file = test::createNimbleFile(
      *rootPool(), {chunk1, chunk2}, writerOptions, /*flushAfterWrite=*/false);

  // Sibling filter on c0 selects a non-dense set spanning both chunks:
  // [50,199] in chunk1 (dict, with nulls) and [400,549] in chunk2 (non-dict).
  auto readType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  std::vector<std::unique_ptr<common::BigintRange>> ranges;
  ranges.push_back(
      std::make_unique<common::BigintRange>(50, 199, /*nullAllowed=*/false));
  ranges.push_back(
      std::make_unique<common::BigintRange>(400, 549, /*nullAllowed=*/false));
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintMultiRange>(
          std::move(ranges), /*nullAllowed=*/false));

  // Full 600-row content for comparison (nulls only in chunk1's c1).
  auto expected = makeRowVector({
      makeFlatVector<int64_t>(600, [](auto i) { return i; }),
      makeFlatVector<std::string>(
          600,
          [](auto i) {
            if (i < 300) {
              return std::vector<std::string>{"aaa", "bbb"}[i % 2];
            }
            return fmt::format(
                "unique_chunk2_row_{}_pad_to_avoid_inline", i - 300);
          },
          [](auto i) { return i < 300 && i % 7 == 0; }),
  });

  auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
  // Single batch spans both chunks → the dict read is abandoned mid-batch on
  // a sparse row set.
  validateWithFilter(*expected, *readers.rowReader, 600, [](int i) {
    return (i >= 50 && i <= 199) || (i >= 400 && i <= 549);
  });
}

// Like sparseRowsAcrossChunkAbandonPreservesNulls, but the non-dict chunk
// (the flat encoding fallback portion) ALSO has nulls. This exercises the
// flat fallback emitting nulls into the shared result null buffer at the
// output offset left by the dict portion (numValues_), on top of preserving
// the dict portion's nulls.
TEST_P(StringColumnReaderTest, sparseRowsAcrossChunkAbandonNullsBothPortions) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  // chunk1: low cardinality with nulls → Nullable→Dictionary (dict path).
  auto chunk1 = makeRowVector({
      makeFlatVector<int64_t>(300, [](auto i) { return i; }),
      makeFlatVector<std::string>(
          300,
          [](auto i) { return std::vector<std::string>{"aaa", "bbb"}[i % 2]; },
          nullEvery(7)),
  });
  // chunk2: all-unique long strings (non-dict → forces abandon) WITH nulls →
  // Nullable→Trivial. The flat encoding fallback must emit these nulls.
  auto chunk2 = makeRowVector({
      makeFlatVector<int64_t>(300, [](auto i) { return 300 + i; }),
      makeFlatVector<std::string>(
          300,
          [](auto i) {
            return fmt::format("unique_chunk2_row_{}_pad_to_avoid_inline", i);
          },
          nullEvery(5)),
  });

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file = test::createNimbleFile(
      *rootPool(), {chunk1, chunk2}, writerOptions, /*flushAfterWrite=*/false);

  // Sibling filter on c0 selects a non-dense set spanning both chunks:
  // [50,199] in chunk1 (dict, with nulls) and [400,549] in chunk2 (non-dict,
  // with nulls).
  auto readType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  std::vector<std::unique_ptr<common::BigintRange>> ranges;
  ranges.push_back(
      std::make_unique<common::BigintRange>(50, 199, /*nullAllowed=*/false));
  ranges.push_back(
      std::make_unique<common::BigintRange>(400, 549, /*nullAllowed=*/false));
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintMultiRange>(
          std::move(ranges), /*nullAllowed=*/false));

  // Full 600-row content. Nulls in chunk1 (i % 7 == 0) and chunk2
  // (chunk2-local index (i - 300) % 5 == 0).
  auto expected = makeRowVector({
      makeFlatVector<int64_t>(600, [](auto i) { return i; }),
      makeFlatVector<std::string>(
          600,
          [](auto i) {
            if (i < 300) {
              return std::vector<std::string>{"aaa", "bbb"}[i % 2];
            }
            return fmt::format(
                "unique_chunk2_row_{}_pad_to_avoid_inline", i - 300);
          },
          [](auto i) {
            return (i < 300 && i % 7 == 0) || (i >= 300 && (i - 300) % 5 == 0);
          }),
  });

  auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
  validateWithFilter(*expected, *readers.rowReader, 600, [](int i) {
    return (i >= 50 && i <= 199) || (i >= 400 && i <= 549);
  });
}

// Like sparseRowsAcrossChunkAbandonNullsBothPortions, but the dict portion
// (chunk1) is ALL non-null while only the flat-fallback portion (chunk2) has
// nulls. An all-non-null dict chunk never allocates/prepares the result null
// buffer during the dict phase (rawNullsInReadRange stays null), so the
// readOffset>0 flat continuation is the first to discover nulls and must
// prepare the buffer over the complete output itself rather than assuming the
// dict phase already did. Regression for the resume branch no-op'ing
// prepareResultNulls/setReturnNullsMode and emitting nulls into an unprepared
// resultNulls_.
TEST_P(
    StringColumnReaderTest,
    sparseRowsAcrossChunkAbandonNullsOnlyInFlatPortion) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  // chunk1: low cardinality, NO nulls → Dictionary (dict path, no null buffer).
  auto chunk1 = makeRowVector({
      makeFlatVector<int64_t>(300, [](auto i) { return i; }),
      makeFlatVector<std::string>(
          300,
          [](auto i) { return std::vector<std::string>{"aaa", "bbb"}[i % 2]; }),
  });
  // chunk2: all-unique long strings (non-dict → forces abandon) WITH nulls →
  // Nullable→Trivial. The flat fallback emits these nulls into a result null
  // buffer the all-non-null dict phase never prepared.
  auto chunk2 = makeRowVector({
      makeFlatVector<int64_t>(300, [](auto i) { return 300 + i; }),
      makeFlatVector<std::string>(
          300,
          [](auto i) {
            return fmt::format("unique_chunk2_row_{}_pad_to_avoid_inline", i);
          },
          nullEvery(5)),
  });

  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file = test::createNimbleFile(
      *rootPool(), {chunk1, chunk2}, writerOptions, /*flushAfterWrite=*/false);

  // Sibling filter on c0 selects a non-dense set spanning both chunks:
  // [50,199] in chunk1 (dict, no nulls) and [400,549] in chunk2 (non-dict,
  // with nulls).
  auto readType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  std::vector<std::unique_ptr<common::BigintRange>> ranges;
  ranges.push_back(
      std::make_unique<common::BigintRange>(50, 199, /*nullAllowed=*/false));
  ranges.push_back(
      std::make_unique<common::BigintRange>(400, 549, /*nullAllowed=*/false));
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintMultiRange>(
          std::move(ranges), /*nullAllowed=*/false));

  // Full 600-row content. Nulls only in chunk2 (chunk2-local (i-300) % 5 == 0).
  auto expected = makeRowVector({
      makeFlatVector<int64_t>(600, [](auto i) { return i; }),
      makeFlatVector<std::string>(
          600,
          [](auto i) {
            if (i < 300) {
              return std::vector<std::string>{"aaa", "bbb"}[i % 2];
            }
            return fmt::format(
                "unique_chunk2_row_{}_pad_to_avoid_inline", i - 300);
          },
          [](auto i) { return i >= 300 && (i - 300) % 5 == 0; }),
  });

  auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
  validateWithFilter(*expected, *readers.rowReader, 600, [](int i) {
    return (i >= 50 && i <= 199) || (i >= 400 && i <= 549);
  });
}

// Fuzz test: randomizes the number of chunks, rows per chunk, batch size,
// and null probability. Exercises multi-chunk dict reads with merged
// alphabets across chunk boundaries.
// TODO: Add mixed dict/non-dict chunks once the reactive fallback is
// validated (seed 1373641041).
TEST_P(StringColumnReaderTest, fuzzMultiChunkDictionary) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  // Runs 0..N-1 replay previously-failing seeds as fixed regressions; the last
  // run is a single random draw (use --stress-runs to exercise more seeds):
  //   - 2770505665: MainlyConstant::materializeIndices rowCount==0 OOB write
  //     (an all-null read range across a chunk boundary).
  //   - 968259391: RLE dict-index mid-run resume bug — a chunk read split
  //     across two batches re-read the in-progress run's index as 0.
  constexpr uint32_t kRegressionSeeds[] = {2770505665u, 968259391u};
  constexpr int kNumRegressionSeeds =
      static_cast<int>(sizeof(kRegressionSeeds) / sizeof(kRegressionSeeds[0]));
  for (int run = 0; run < kNumRegressionSeeds + 1; ++run) {
    // Random run is deterministic by default; set VELOX_TEST_USE_RANDOM_SEED=1
    // (idiomatic with --stress-runs) to draw a fresh, logged seed per process.
    const auto seed = run < kNumRegressionSeeds
        ? kRegressionSeeds[run]
        : common::testutil::getRandomSeed(/*fixedValue=*/2);
    std::mt19937 rng(seed);
    const int numChunks = 2 + rng() % 6;
    const int rowsPerChunk = 100 + rng() % 900;
    const int batchSize = 50 + rng() % (numChunks * rowsPerChunk);
    const bool hasNulls = rng() % 3 == 0;

    SCOPED_TRACE(
        fmt::format(
            "run={} seed={} numChunks={} rowsPerChunk={} batchSize={} hasNulls={}",
            run,
            seed,
            numChunks,
            rowsPerChunk,
            batchSize,
            hasNulls));

    int totalRows = 0;
    std::vector<std::vector<std::string>> chunkValues;

    for (int c = 0; c < numChunks; ++c) {
      std::vector<std::string> values;
      values.reserve(rowsPerChunk);
      for (int r = 0; r < rowsPerChunk; ++r) {
        // All chunks use dict-compatible data (low cardinality) so the
        // multi-chunk merged-alphabet path is exercised. Per-chunk alphabet
        // varies to test alphabet merging with different entries.
        // FIXME: Add mixed dict/non-dict chunks once the reactive fallback
        // bug (seed 1373641041) is fixed.
        const std::string dictValues[] = {
            fmt::format("dict_alpha_chunk{}_pad", c),
            fmt::format("dict_bravo_chunk{}_pad", c),
            fmt::format("dict_charlie_chunk{}_pad", c)};
        values.push_back(dictValues[rng() % 3]);
      }
      chunkValues.push_back(std::move(values));
      totalRows += rowsPerChunk;
    }

    // Build the expected flat vector from all chunks.
    auto expected = makeRowVector({makeFlatVector<std::string>(
        totalRows,
        [&](auto i) {
          int chunkIdx = i / rowsPerChunk;
          int rowIdx = i % rowsPerChunk;
          return chunkValues[chunkIdx][rowIdx];
        },
        hasNulls ? nullEvery(7) : nullptr)});

    // Build chunk vectors for writing. Use global null indices so the
    // null pattern matches the expected vector.
    std::vector<velox::VectorPtr> chunkVectors;
    for (int c = 0; c < numChunks; ++c) {
      const auto& vals = chunkValues[c];
      const auto chunkStart = c * rowsPerChunk;
      auto nullFunc = [&](auto i) { return (chunkStart + i) % 7 == 0; };
      chunkVectors.push_back(makeRowVector({makeFlatVector<std::string>(
          rowsPerChunk,
          [&](auto i) { return vals[i]; },
          hasNulls ? std::function<bool(velox::vector_size_t)>(nullFunc)
                   : nullptr)}));
    }

    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.minStreamChunkRawSize = 0;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    auto file = test::createNimbleFile(
        *rootPool(), chunkVectors, writerOptions, /*flushAfterWrite=*/false);

    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*expected->type());
    auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
    validate(*expected, *readers.rowReader, batchSize);
  }
}

// Fuzz test for both legacy and zero-copy paths. Verifies data correctness
// with randomized chunk shapes, batch sizes, and null patterns. Runs for
// both param values (legacy does flat reads, zero-copy may take dict path).
TEST_P(StringColumnReaderTest, fuzzMultiChunkReadCorrectness) {
  const bool stringDecoderZeroCopy = GetParam();

  for (int run = 0; run < 5; ++run) {
    // Deterministic by default; set env VELOX_TEST_USE_RANDOM_SEED=1 for a
    // fresh, logged seed per run (idiomatic with stress runs).
    auto seed = common::testutil::getRandomSeed(/*fixedValue=*/3 + run);
    std::mt19937 rng(seed);
    const int numChunks = 2 + rng() % 5;
    const int rowsPerChunk = 100 + rng() % 500;
    const int batchSize = 50 + rng() % (numChunks * rowsPerChunk);
    const bool hasNulls = rng() % 3 == 0;

    SCOPED_TRACE(
        fmt::format(
            "run={} seed={} numChunks={} rowsPerChunk={} batchSize={} "
            "hasNulls={} zeroCopy={}",
            run,
            seed,
            numChunks,
            rowsPerChunk,
            batchSize,
            hasNulls,
            stringDecoderZeroCopy));

    int totalRows = 0;
    std::vector<std::vector<std::string>> chunkValues;

    for (int c = 0; c < numChunks; ++c) {
      std::vector<std::string> values;
      values.reserve(rowsPerChunk);
      for (int r = 0; r < rowsPerChunk; ++r) {
        // Unique strings per row so both legacy and zero-copy paths use
        // the flat read path. This verifies chunking correctness
        // independent of the dictionary optimization.
        values.push_back(fmt::format("val_c{}_r{}_pad_to_avoid_inline", c, r));
      }
      chunkValues.push_back(std::move(values));
      totalRows += rowsPerChunk;
    }

    auto expected = makeRowVector({makeFlatVector<std::string>(
        totalRows,
        [&](auto i) {
          int chunkIdx = i / rowsPerChunk;
          int rowIdx = i % rowsPerChunk;
          return chunkValues[chunkIdx][rowIdx];
        },
        hasNulls ? nullEvery(7) : nullptr)});

    std::vector<velox::VectorPtr> chunkVectors;
    for (int c = 0; c < numChunks; ++c) {
      const auto& vals = chunkValues[c];
      const auto chunkStart = c * rowsPerChunk;
      auto nullFunc = [&](auto i) { return (chunkStart + i) % 7 == 0; };
      chunkVectors.push_back(makeRowVector({makeFlatVector<std::string>(
          rowsPerChunk,
          [&](auto i) { return vals[i]; },
          hasNulls ? std::function<bool(velox::vector_size_t)>(nullFunc)
                   : nullptr)}));
    }

    WriterOptions writerOptions;
    writerOptions.enableChunking = true;
    writerOptions.minStreamChunkRawSize = 0;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          /*flushLambda=*/[](const StripeProgress&) { return false; },
          /*chunkLambda=*/[](const StripeProgress&) { return true; });
    };
    auto file = test::createNimbleFile(
        *rootPool(), chunkVectors, writerOptions, /*flushAfterWrite=*/false);

    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*expected->type());
    auto readers = makeReaders(expected, file, scanSpec, stringDecoderZeroCopy);
    validate(*expected, *readers.rowReader, batchSize);
  }
}

// Regression for the prepareResultNulls guard bug. A sibling filter on c0
// forces ONE sparse readWithVisitor over c1 that spans both chunks: chunk 1's
// c1 values are non-null and chunk 2's are nullable. The null-free chunk-1
// portion calls prepareResultNulls with hasNulls=false (prepareNulls no-ops and
// allocates nothing); if the shared resultNullsPrepared guard is flipped
// regardless, the nullable chunk-2 portion skips the real prepare and writes
// nulls into an unallocated resultNulls_ -> crash. Exercises the
// dictionary-index read path (readDictionaryIndices).
TEST_P(StringColumnReaderTest, multiChunkLateNullPreparationDictPath) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Dictionary path requires stringDecoderZeroCopy";
  }

  constexpr int kRowsPerChunk = 200;
  // c1 low-cardinality -> Dictionary. chunk 1 non-null, chunk 2 nullable.
  auto chunk1 = makeRowVector({
      makeFlatVector<int64_t>(kRowsPerChunk, [](auto i) { return i; }),
      makeFlatVector<std::string>(
          kRowsPerChunk,
          [](auto i) {
            return std::vector<std::string>{"aa", "bb", "cc"}[i % 3];
          }),
  });
  auto chunk2 = makeRowVector({
      makeFlatVector<int64_t>(
          kRowsPerChunk, [](auto i) { return kRowsPerChunk + i; }),
      makeFlatVector<std::string>(
          kRowsPerChunk,
          [](auto i) { return std::vector<std::string>{"xx", "yy"}[i % 2]; },
          nullEvery(7)),
  });

  // Force c1 to a bare Dictionary via the encoding layout tree, driving the
  // dictionary-index read path. The writer wraps it in Nullable automatically
  // for the nullable chunk.
  EncodingLayout c1DictLayout{
      EncodingType::Dictionary,
      {},
      CompressionType::Uncompressed,
      {std::nullopt, std::nullopt}};
  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  writerOptions.encodingLayoutTree.emplace(
      Kind::Row,
      std::
          unordered_map<EncodingLayoutTree::StreamIdentifier, EncodingLayout>{},
      "",
      std::vector<EncodingLayoutTree>{
          EncodingLayoutTree{Kind::Scalar, {}, "c0"},
          EncodingLayoutTree{
              Kind::Scalar, {{0, std::move(c1DictLayout)}}, "c1"}});
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file = test::createNimbleFile(
      *rootPool(), {chunk1, chunk2}, writerOptions, /*flushAfterWrite=*/false);

  // Filter c0 to [100, 300]: a row set spanning the chunk boundary at 200, so
  // c1 (no filter) is read sparsely across both chunks in a single
  // readWithVisitor.
  auto readType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(100, 300, /*nullAllowed=*/false));

  auto input = makeRowVector({
      makeFlatVector<int64_t>(2 * kRowsPerChunk, [](auto i) { return i; }),
      makeFlatVector<std::string>(
          2 * kRowsPerChunk,
          [](auto i) {
            return i < kRowsPerChunk
                ? std::vector<std::string>{"aa", "bb", "cc"}[i % 3]
                : std::vector<std::string>{"xx", "yy"}[(i - kRowsPerChunk) % 2];
          },
          [](auto i) {
            return i >= kRowsPerChunk && ((i - kRowsPerChunk) % 7 == 0);
          }),
  });

  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
  validateWithFilter(
      *input,
      *readers.rowReader,
      2 * kRowsPerChunk,
      [](int i) { return i >= 100 && i <= 300; },
      VectorEncoding::Simple::DICTIONARY,
      /*childIndex=*/1);
}

// Same regression for the non-dictionary (flat) read path (readWithVisitor):
// unique values force Trivial, so c1 goes through readWithVisitorFast rather
// than the dictionary-index path.
TEST_P(StringColumnReaderTest, multiChunkLateNullPreparationFlatPath) {
  const bool stringDecoderZeroCopy = GetParam();
  if (!stringDecoderZeroCopy) {
    GTEST_SKIP() << "Repro requires the dict-vector flags";
  }

  constexpr int kRowsPerChunk = 200;
  // c1 unique -> Trivial (flat). chunk 1 non-null, chunk 2 nullable.
  auto chunk1 = makeRowVector({
      makeFlatVector<int64_t>(kRowsPerChunk, [](auto i) { return i; }),
      makeFlatVector<std::string>(
          kRowsPerChunk, [](auto i) { return "u_" + std::to_string(i); }),
  });
  auto chunk2 = makeRowVector({
      makeFlatVector<int64_t>(
          kRowsPerChunk, [](auto i) { return kRowsPerChunk + i; }),
      makeFlatVector<std::string>(
          kRowsPerChunk,
          [](auto i) { return "u_" + std::to_string(kRowsPerChunk + i); },
          nullEvery(7)),
  });

  // Force c1 to Trivial via the encoding layout tree, driving the flat
  // readWithVisitorFast path. The writer wraps it in Nullable automatically for
  // the nullable chunk.
  EncodingLayout c1TrivialLayout{
      EncodingType::Trivial, {}, CompressionType::Uncompressed, {std::nullopt}};
  WriterOptions writerOptions;
  writerOptions.enableChunking = true;
  writerOptions.minStreamChunkRawSize = 0;
  writerOptions.encodingLayoutTree.emplace(
      Kind::Row,
      std::
          unordered_map<EncodingLayoutTree::StreamIdentifier, EncodingLayout>{},
      "",
      std::vector<EncodingLayoutTree>{
          EncodingLayoutTree{Kind::Scalar, {}, "c0"},
          EncodingLayoutTree{
              Kind::Scalar, {{0, std::move(c1TrivialLayout)}}, "c1"}});
  writerOptions.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  auto file = test::createNimbleFile(
      *rootPool(), {chunk1, chunk2}, writerOptions, /*flushAfterWrite=*/false);

  auto readType = ROW({"c0", "c1"}, {BIGINT(), VARCHAR()});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*readType);
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(100, 300, /*nullAllowed=*/false));

  auto input = makeRowVector({
      makeFlatVector<int64_t>(2 * kRowsPerChunk, [](auto i) { return i; }),
      makeFlatVector<std::string>(
          2 * kRowsPerChunk,
          [](auto i) { return "u_" + std::to_string(i); },
          [](auto i) {
            return i >= kRowsPerChunk && ((i - kRowsPerChunk) % 7 == 0);
          }),
  });

  auto readers = makeReaders(input, file, scanSpec, stringDecoderZeroCopy);
  validateWithFilter(
      *input,
      *readers.rowReader,
      2 * kRowsPerChunk,
      [](int i) { return i >= 100 && i <= 300; },
      VectorEncoding::Simple::FLAT,
      /*childIndex=*/1);
}

INSTANTIATE_TEST_CASE_P(
    StringColumnReaderTestSuite,
    StringColumnReaderTest,
    testing::Values(false, true),
    [](const testing::TestParamInfo<bool>& info) {
      return info.param ? "zeroCopy" : "legacy";
    });

} // namespace
} // namespace facebook::nimble
