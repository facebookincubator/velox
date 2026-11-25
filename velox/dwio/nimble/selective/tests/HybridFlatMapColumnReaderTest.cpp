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

#include "velox/dwio/nimble/selective/SelectiveNimbleReader.h"

#include "velox/common/file/File.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/common/Mutation.h"
#include "velox/dwio/nimble/common/HybridFlatMap.h"
#include "velox/dwio/nimble/common/SchemaUtils.h"
#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/reader/BatchReader.h"
#include "velox/dwio/nimble/tests/HybridFlatMapGroupings.h"
#include "velox/dwio/nimble/writer/BufferPolicy.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <algorithm>
#include <array>
#include <functional>
#include <numeric>
#include <random>

#include <fmt/format.h>
#include <folly/Random.h>
#include <folly/container/F14Set.h>
#include <gtest/gtest.h>

namespace facebook::nimble {
namespace {

using namespace facebook::velox;

// Builds a scan spec over 'rowType' carrying a map-key filter on 'column',
// the shape a query engine produces for subfield (map key) pruning.
std::shared_ptr<common::ScanSpec> makeKeyFilterScanSpec(
    const TypePtr& rowType,
    std::unique_ptr<common::Filter> keyFilter,
    const std::string& column = "c0") {
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*rowType);
  scanSpec->childByName(column)
      ->childByName(common::ScanSpec::kMapKeysFieldName)
      ->setFilter(std::move(keyFilter));
  return scanSpec;
}

struct HybridFlatMapColumnReaderTestParam {
  bool skipConstantFlatMapInMapStreams;
  // Puts every key in the Default group instead of a grouping with explicit
  // groups.
  bool defaultOnlyGrouping;
  // How each stripe records its Default keys.
  HybridFlatMap::KeyPresenceMode keyPresenceMode;
};

std::string testName(const HybridFlatMapColumnReaderTestParam& param) {
  return fmt::format(
      "{}_{}",
      param.defaultOnlyGrouping ? "defaultOnly" : "explicitGroups",
      toString(param.keyPresenceMode));
}

using MapRow =
    std::optional<std::vector<std::pair<int64_t, std::optional<int64_t>>>>;

// Returns 'numRows' maps over 'keys'. A non-null, non-empty map in row 'r'
// holds key 'k' unless (r + k) % 4 is 0, with value r * 100 + k, or null when
// (r + k) % 13 is 0.
std::vector<MapRow> makeMapRows(
    vector_size_t numRows,
    const std::vector<int64_t>& keys,
    const std::function<bool(vector_size_t)>& isNullAt,
    const std::function<bool(vector_size_t)>& isEmptyAt) {
  std::vector<MapRow> rows;
  rows.reserve(numRows);
  for (vector_size_t row = 0; row < numRows; ++row) {
    if (isNullAt(row)) {
      rows.push_back(std::nullopt);
      continue;
    }
    std::vector<std::pair<int64_t, std::optional<int64_t>>> entries;
    if (!isEmptyAt(row)) {
      for (auto key : keys) {
        if ((row + key) % 4 == 0) {
          continue;
        }
        entries.emplace_back(
            key,
            (row + key) % 13 == 0 ? std::nullopt
                                  : std::optional<int64_t>(row * 100 + key));
      }
    }
    rows.push_back(std::move(entries));
  }
  return rows;
}

// One fuzzed map: null, or its entries as (key index, value) pairs. The key
// index picks a key from the fuzzed column's key list.
using FuzzMapRow =
    std::optional<std::vector<std::pair<size_t, std::optional<int64_t>>>>;

// Returns 'numRows' fuzzed maps over 'numKeys' keys. A map is null or empty
// with probability 1/10 each. Otherwise it holds each key with a probability
// drawn per key for the call, so some keys are rare or absent, and each value
// is null with probability 1/10.
std::vector<FuzzMapRow>
makeFuzzMapRows(std::mt19937& rng, vector_size_t numRows, size_t numKeys) {
  std::uniform_real_distribution<double> uniform{0.0, 1.0};
  std::vector<double> keyProbabilities(numKeys);
  for (auto& probability : keyProbabilities) {
    probability = uniform(rng);
  }
  std::vector<FuzzMapRow> rows;
  rows.reserve(numRows);
  for (vector_size_t row = 0; row < numRows; ++row) {
    if (uniform(rng) < 0.1) {
      rows.emplace_back(std::nullopt);
      continue;
    }
    auto& entries = rows.emplace_back(std::in_place).value();
    if (uniform(rng) < 0.1) {
      continue;
    }
    for (size_t key = 0; key < numKeys; ++key) {
      if (uniform(rng) >= keyProbabilities[key]) {
        continue;
      }
      entries.emplace_back(
          key,
          uniform(rng) < 0.1
              ? std::nullopt
              : std::optional<int64_t>(static_cast<int64_t>(rng() % 1'000)));
    }
  }
  return rows;
}

// Tests for HybridFlatMapColumnReader. Each test runs with explicit groups and
// with every key in the Default group, each with a key-presence bitmap and
// with a Default key list.
class HybridFlatMapColumnReaderTest
    : public ::testing::TestWithParam<HybridFlatMapColumnReaderTestParam>,
      public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
    registerSelectiveNimbleReaderFactory();
    common::testutil::TestValue::enable();
  }

  static void TearDownTestCase() {
    common::testutil::TestValue::disable();
    unregisterSelectiveNimbleReaderFactory();
  }

  memory::MemoryPool* rootPool() {
    return rootPool_.get();
  }

  struct Readers {
    std::unique_ptr<dwio::common::Reader> reader;
    std::unique_ptr<dwio::common::RowReader> rowReader;
  };

  // Returns writer options that write 'mapColumns' as Hybrid FlatMaps. The
  // explicit groups hold keys that many tests write, other keys go to
  // Default, and no test writes key -1, so every stripe also has an explicit
  // group without data. Each write is its own stripe.
  WriterOptions hybridFlatMapWriterOptions(
      const velox::RowType& type,
      const std::vector<std::string>& mapColumns) const {
    WriterOptions writerOptions;
    writerOptions.skipConstantFlatMapInMapStreams =
        GetParam().skipConstantFlatMapInMapStreams;
    for (const auto& column : mapColumns) {
      const auto keyKind =
          hybridFlatMapKeyKind(*type.findChild(column)->childAt(0));
      writerOptions.hybridFlatMapColumns.emplace(
          column,
          test::makeHybridFlatMap(
              keyKind,
              GetParam().defaultOnlyGrouping
                  ? std::vector<std::vector<std::string>>{}
                  : std::vector<std::vector<
                        std::string>>{{"-1"}, {"1", "5"}, {"2", "3"}},
              GetParam().keyPresenceMode));
    }
    writerOptions.bufferPolicyFactory =
        test::stripePerWriteBufferPolicyFactory();
    return writerOptions;
  }

  std::string writeHybridFlatMapFile(
      const VectorPtr& input,
      const std::string& column = "c0") {
    return test::createNimbleFile(
        *rootPool(),
        input,
        hybridFlatMapWriterOptions(*asRowType(input->type()), {column}));
  }

  // Writes one stripe per vector, so later stripes can hold key sets that
  // differ from the first.
  std::string writeMultiStripeHybridFlatMapFile(
      const std::vector<VectorPtr>& stripes,
      const std::string& column = "c0") {
    return test::createNimbleFile(
        *rootPool(),
        stripes,
        hybridFlatMapWriterOptions(
            *asRowType(stripes.front()->type()), {column}),
        /*flushAfterWrite=*/false);
  }

  // Uses the type of 'expected' as the requested type.
  Readers makeReaders(
      const RowVectorPtr& expected,
      const std::string& file,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      bool preserveFlatMapsInMemory = false,
      bool lazyColumnIo = false,
      const folly::F14FastSet<std::string>& remainingFilterColumns = {}) {
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
    dwio::common::RowReaderOptions rowOptions;
    rowOptions.setScanSpec(scanSpec);
    rowOptions.setRequestedType(asRowType(expected->type()));
    rowOptions.setPreserveFlatMapsInMemory(preserveFlatMapsInMemory);
    rowOptions.setLazyColumnIo(lazyColumnIo);
    if (!remainingFilterColumns.empty()) {
      rowOptions.setRemainingFilterColumns(remainingFilterColumns);
    }
    readers.rowReader = readers.reader->createRowReader(rowOptions);
    return readers;
  }

  // Reads every batch of 'batchSize' rows and checks that the rows match
  // 'expected'.
  void validate(
      const RowVector& expected,
      dwio::common::RowReader& rowReader,
      int batchSize) {
    auto result = BaseVector::create(asRowType(expected.type()), 0, pool());
    vector_size_t numScanned{0};
    vector_size_t offset{0};
    while (numScanned < expected.size()) {
      numScanned += rowReader.next(batchSize, result);
      result->validate();
      for (vector_size_t i = 0; i < result->size(); ++i) {
        ASSERT_TRUE(result->equalValueAt(&expected, i, offset + i))
            << "Mismatch at row " << offset + i << ": expected "
            << expected.toString(offset + i) << " got " << result->toString(i);
      }
      offset += result->size();
    }
    ASSERT_EQ(numScanned, expected.size());
    ASSERT_EQ(rowReader.next(1, result), 0);
  }

  // Returns 'rows' over 'keys' as a MAP(K, BIGINT), or with 'arrayValues' as a
  // MAP(K, ARRAY(BIGINT)) whose arrays hold value % 4 elements.
  template <typename K>
  VectorPtr makeFuzzMapVector(
      const std::vector<FuzzMapRow>& rows,
      const std::vector<K>& keys,
      bool arrayValues) {
    std::vector<vector_size_t> offsets;
    std::vector<vector_size_t> nullRows;
    std::vector<K> mapKeys;
    std::vector<std::optional<int64_t>> values;
    for (vector_size_t row = 0; row < static_cast<vector_size_t>(rows.size());
         ++row) {
      offsets.push_back(static_cast<vector_size_t>(mapKeys.size()));
      if (!rows[row].has_value()) {
        nullRows.push_back(row);
        continue;
      }
      for (const auto& [key, value] : *rows[row]) {
        mapKeys.push_back(keys[key]);
        values.push_back(value);
      }
    }
    if (!arrayValues) {
      return makeMapVector(
          offsets,
          makeFlatVector<K>(mapKeys),
          makeNullableFlatVector<int64_t>(values),
          nullRows);
    }
    // makeArrayVector() needs at least one array.
    if (values.empty()) {
      return makeMapVector(
          offsets,
          makeFlatVector<K>(mapKeys),
          BaseVector::create(ARRAY(BIGINT()), 0, pool()),
          nullRows);
    }
    std::vector<vector_size_t> arrayOffsets;
    std::vector<vector_size_t> nullArrays;
    std::vector<int64_t> elements;
    for (vector_size_t i = 0; i < static_cast<vector_size_t>(values.size());
         ++i) {
      arrayOffsets.push_back(static_cast<vector_size_t>(elements.size()));
      if (!values[i].has_value()) {
        nullArrays.push_back(i);
        continue;
      }
      for (int64_t element = 0; element < *values[i] % 4; ++element) {
        elements.push_back(*values[i] * 10 + element);
      }
    }
    return makeMapVector(
        offsets,
        makeFlatVector<K>(mapKeys),
        makeArrayVector(
            arrayOffsets, makeFlatVector<int64_t>(elements), nullArrays),
        nullRows);
  }

  // Returns a keys filter that passes exactly 'values'.
  template <typename K>
  static std::unique_ptr<common::Filter> makeKeyValuesFilter(
      const std::vector<K>& values) {
    if constexpr (std::is_same_v<K, std::string>) {
      return std::make_unique<common::BytesValues>(values, false);
    } else {
      return common::createBigintValues(
          std::vector<int64_t>(values.begin(), values.end()), false);
    }
  }

  // Checks that the batch reader reads 'written' back from 'file'.
  void verifyBatchReader(const RowVector& written, const std::string& file) {
    InMemoryReadFile readFile{file};
    BatchReader reader{&readFile, *pool()};
    VectorPtr result;
    vector_size_t offset{0};
    while (reader.next(37, result)) {
      for (vector_size_t i = 0; i < result->size(); ++i) {
        ASSERT_LT(offset + i, written.size());
        ASSERT_TRUE(result->equalValueAt(&written, i, offset + i))
            << "Batch reader mismatch at row " << offset + i << ": expected "
            << written.toString(offset + i) << " got " << result->toString(i);
      }
      offset += result->size();
    }
    EXPECT_EQ(offset, written.size());
  }

  // Writes fuzzed maps over 'keys' in several write batches under a random
  // raw-size stripe target, so stripes merge batches and cut through them.
  // Then reads the file with random key, map null and id filters, batch sizes
  // and read options. Key filters list some of 'keys' and one of 'absentKeys',
  // which no row holds.
  template <typename K>
  void fuzzReads(
      std::mt19937& rng,
      const std::vector<K>& keys,
      const std::vector<K>& absentKeys) {
    const bool arrayValues = rng() % 2 == 0;
    std::vector<FuzzMapRow> rows;
    std::vector<VectorPtr> batches;
    const auto numBatches = 2 + rng() % 5;
    for (size_t batch = 0; batch < numBatches; ++batch) {
      const auto numRows = static_cast<vector_size_t>(1 + rng() % 80);
      const auto firstRow = static_cast<int64_t>(rows.size());
      auto batchRows = makeFuzzMapRows(rng, numRows, keys.size());
      batches.push_back(makeRowVector({
          makeFuzzMapVector(batchRows, keys, arrayValues),
          makeFlatVector<int64_t>(
              numRows, [&](auto row) { return firstRow + row; }),
      }));
      rows.insert(rows.end(), batchRows.begin(), batchRows.end());
    }
    const auto& rowType = asRowType(batches.front()->type());
    auto writerOptions = hybridFlatMapWriterOptions(*rowType, {"c0"});
    constexpr std::array<uint64_t, 5> kStripeRawSizes{
        1, 64, 512, 4'096, 1 << 20};
    const auto stripeRawSize = kStripeRawSizes[rng() % kStripeRawSizes.size()];
    writerOptions.bufferPolicyFactory = [stripeRawSize] {
      return std::make_unique<StripeRawSizeBufferPolicy>(stripeRawSize);
    };
    const bool chunked = rng() % 2 == 0;
    if (chunked) {
      writerOptions.minStreamChunkRawSize = 1;
      writerOptions.maxStreamChunkRawSize = 64;
    }
    SCOPED_TRACE(
        fmt::format(
            "arrayValues {}, stripeRawSize {}, chunked {}, rows {}",
            arrayValues,
            stripeRawSize,
            chunked,
            rows.size()));
    const auto file = test::createNimbleFile(
        *rootPool(), batches, writerOptions, /*flushAfterWrite=*/false);

    const auto ids = [](const std::vector<vector_size_t>& rowNumbers) {
      return std::vector<int64_t>(rowNumbers.begin(), rowNumbers.end());
    };
    std::vector<vector_size_t> allRows(rows.size());
    std::iota(allRows.begin(), allRows.end(), 0);
    const auto written = makeRowVector({
        makeFuzzMapVector(rows, keys, arrayValues),
        makeFlatVector<int64_t>(ids(allRows)),
    });
    verifyBatchReader(*written, file);

    constexpr int kNumReads{4};
    for (int read = 0; read < kNumReads; ++read) {
      SCOPED_TRACE(fmt::format("read {}", read));
      std::vector<bool> keepKey(keys.size(), true);
      std::unique_ptr<common::Filter> keyFilter;
      const auto keyFilterKind = rng() % 3;
      if (keyFilterKind == 1 ||
          (keyFilterKind == 2 && std::is_same_v<K, std::string>)) {
        std::vector<K> values{absentKeys[rng() % absentKeys.size()]};
        for (size_t key = 0; key < keys.size(); ++key) {
          keepKey[key] = rng() % 2 == 0;
          if (keepKey[key]) {
            values.push_back(keys[key]);
          }
        }
        keyFilter = makeKeyValuesFilter(values);
      } else if (keyFilterKind == 2) {
        if constexpr (!std::is_same_v<K, std::string>) {
          auto lower = keys[rng() % keys.size()];
          auto upper = keys[rng() % keys.size()];
          if (lower > upper) {
            std::swap(lower, upper);
          }
          for (size_t key = 0; key < keys.size(); ++key) {
            keepKey[key] = lower <= keys[key] && keys[key] <= upper;
          }
          keyFilter =
              std::make_unique<common::BigintRange>(lower, upper, false);
        }
      }
      const auto mapNullFilter = rng() % 3;
      const bool filterIds = rng() % 2 == 0;
      std::vector<bool> keepId(rows.size(), true);
      std::vector<int64_t> passingIds;
      if (filterIds) {
        for (size_t row = 0; row < rows.size(); ++row) {
          keepId[row] = rng() % 2 == 0;
          if (keepId[row]) {
            passingIds.push_back(static_cast<int64_t>(row));
          }
        }
      }
      const bool preserveFlatMapsInMemory = rng() % 2 == 0;
      const bool lazyColumnIo = rng() % 2 == 0;
      const auto batchSize = static_cast<int>(1 + rng() % 64);
      SCOPED_TRACE(
          fmt::format(
              "keyFilter {}, mapNullFilter {}, filterIds {}, "
              "preserveFlatMapsInMemory {}, lazyColumnIo {}, batchSize {}",
              keyFilter == nullptr ? "none" : keyFilter->toString(),
              mapNullFilter,
              filterIds,
              preserveFlatMapsInMemory,
              lazyColumnIo,
              batchSize));

      std::vector<FuzzMapRow> expectedRows;
      std::vector<vector_size_t> expectedRowNumbers;
      for (size_t row = 0; row < rows.size(); ++row) {
        const bool isNull = !rows[row].has_value();
        if ((mapNullFilter == 1 && !isNull) || (mapNullFilter == 2 && isNull) ||
            !keepId[row]) {
          continue;
        }
        expectedRowNumbers.push_back(static_cast<vector_size_t>(row));
        auto& expectedRow = expectedRows.emplace_back(rows[row]);
        if (expectedRow.has_value()) {
          std::erase_if(*expectedRow, [&](const auto& entry) {
            return !keepKey[entry.first];
          });
        }
      }
      if (expectedRows.empty()) {
        continue;
      }
      const auto expected = makeRowVector({
          makeFuzzMapVector(expectedRows, keys, arrayValues),
          makeFlatVector<int64_t>(ids(expectedRowNumbers)),
      });

      auto scanSpec = keyFilter == nullptr
          ? std::make_shared<common::ScanSpec>("root")
          : makeKeyFilterScanSpec(rowType, std::move(keyFilter));
      if (scanSpec->children().empty()) {
        scanSpec->addAllChildFields(*rowType);
      }
      if (mapNullFilter == 1) {
        scanSpec->childByName("c0")->setFilter(
            std::make_unique<common::IsNull>());
      } else if (mapNullFilter == 2) {
        scanSpec->childByName("c0")->setFilter(
            std::make_unique<common::IsNotNull>());
      }
      if (filterIds) {
        scanSpec->childByName("c1")->setFilter(
            common::createBigintValues(passingIds, false));
      }
      auto readers = makeReaders(
          written, file, scanSpec, preserveFlatMapsInMemory, lazyColumnIo);
      auto result = BaseVector::create(rowType, 0, pool());
      vector_size_t numRead{0};
      while (readers.rowReader->next(batchSize, result) > 0) {
        for (vector_size_t i = 0; i < result->size(); ++i) {
          ASSERT_LT(numRead, expected->size());
          ASSERT_TRUE(result->equalValueAt(expected.get(), i, numRead))
              << "Mismatch at expected row " << numRead << ": expected "
              << expected->toString(numRead) << " got " << result->toString(i);
          ++numRead;
        }
      }
      EXPECT_EQ(numRead, expected->size());
    }
  }

  // Reads every batch and checks that the returned rows are the rows of
  // 'input' for which 'keep' is true, in order.
  void validateFiltered(
      const RowVector& input,
      dwio::common::RowReader& rowReader,
      int batchSize,
      const std::function<bool(vector_size_t)>& keep) {
    auto result = BaseVector::create(asRowType(input.type()), 0, pool());
    vector_size_t inputRow = 0;
    while (rowReader.next(batchSize, result) > 0) {
      result->validate();
      for (vector_size_t i = 0; i < result->size(); ++i) {
        while (inputRow < input.size() && !keep(inputRow)) {
          ++inputRow;
        }
        ASSERT_LT(inputRow, input.size());
        ASSERT_TRUE(result->equalValueAt(&input, i, inputRow))
            << "Mismatch at row " << inputRow << ": expected "
            << input.toString(inputRow) << " got " << result->toString(i);
        ++inputRow;
      }
    }
    for (; inputRow < input.size(); ++inputRow) {
      ASSERT_FALSE(keep(inputRow)) << "Missing row " << inputRow;
    }
  }

  const std::shared_ptr<io::IoStatistics> dataIoStats_{
      std::make_shared<io::IoStatistics>()};
  const std::shared_ptr<io::IoStatistics> metadataIoStats_{
      std::make_shared<io::IoStatistics>()};
};

// Stripes whose maps are all empty have no streams for the column, stripes
// whose maps are all null have only the nulls stream, and explicit groups
// can be absent from a stripe and present in a later one. Default keys that
// do not fit inline in a StringView are first seen in different stripes.
TEST_P(HybridFlatMapColumnReaderTest, emptyAndNullStripes) {
  using StringMapRow =
      std::optional<std::vector<std::pair<StringView, std::optional<int64_t>>>>;
  const StringMapRow emptyMap{std::in_place};
  const std::vector<std::vector<StringMapRow>> stripeRows{
      {emptyMap, emptyMap, emptyMap},
      {std::nullopt, std::nullopt},
      {
          StringMapRow{{{"1", 10}, {"default_key_seven", 70}}},
          std::nullopt,
          emptyMap,
          StringMapRow{{{"2", 20}, {"default_key_nine", 90}}},
      },
      {
          StringMapRow{{{"default_key_seven", 71}}},
          StringMapRow{{{"default_key_eleven", 111}}},
          std::nullopt,
      },
      {
          StringMapRow{{{"1", 14}, {"5", 54}}},
          emptyMap,
          StringMapRow{{{"default_key_thirteen", 134}, {"3", 34}}},
      },
  };
  std::vector<VectorPtr> stripes;
  std::vector<StringMapRow> allRows;
  std::vector<StringMapRow> prunedRows;
  for (const auto& rows : stripeRows) {
    stripes.push_back(
        makeRowVector({makeNullableMapVector<StringView, int64_t>(rows)}));
    for (const auto& row : rows) {
      allRows.push_back(row);
      prunedRows.push_back(row.has_value() ? emptyMap : std::nullopt);
    }
  }
  auto file = writeMultiStripeHybridFlatMapFile(stripes);

  {
    SCOPED_TRACE("all keys");
    auto expected =
        makeRowVector({makeNullableMapVector<StringView, int64_t>(allRows)});
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*expected->type());
    // The keys must stay valid after the readers and the file schema are gone.
    std::vector<VectorPtr> batches;
    {
      auto readers = makeReaders(expected, file, scanSpec);
      auto batch = BaseVector::create(expected->type(), 0, pool());
      while (readers.rowReader->next(2, batch) > 0) {
        batch->loadedVector();
        batches.push_back(std::move(batch));
        batch = BaseVector::create(expected->type(), 0, pool());
      }
    }
    vector_size_t offset = 0;
    for (const auto& batch : batches) {
      velox::test::assertEqualVectors(
          expected->slice(offset, batch->size()), batch);
      offset += batch->size();
    }
    EXPECT_EQ(offset, expected->size());
  }
  {
    SCOPED_TRACE("no key passes the filter");
    auto expected =
        makeRowVector({makeNullableMapVector<StringView, int64_t>(prunedRows)});
    auto scanSpec = makeKeyFilterScanSpec(
        expected->type(),
        std::make_unique<common::BytesValues>(
            std::vector<std::string>{"no_such_key"}, false));
    auto readers = makeReaders(expected, file, scanSpec);
    validate(*expected, *readers.rowReader, 2);
  }
}

// Null values under present keys. A group whose values are all null in a
// stripe has no value streams there, and ROW values can be null. A non-null
// ROW value whose fields are all null has no streams either.
TEST_P(HybridFlatMapColumnReaderTest, nullValues) {
  {
    SCOPED_TRACE("scalar values");
    // In the grouping with explicit groups, group {2, 3} holds only null values
    // in the first stripe. In both groupings, Default holds only null values in
    // the second stripe.
    const MapRow emptyMap{std::in_place};
    const std::vector<std::vector<MapRow>> stripeRows{
        {
            MapRow{{{1, std::nullopt}, {7, 70}}},
            MapRow{{{2, std::nullopt}, {3, std::nullopt}}},
            std::nullopt,
            MapRow{{{5, 50}, {2, std::nullopt}}},
            emptyMap,
        },
        {
            MapRow{{{7, std::nullopt}}},
            MapRow{{{9, std::nullopt}}},
        },
    };
    std::vector<VectorPtr> stripes;
    std::vector<MapRow> allRows;
    for (const auto& rows : stripeRows) {
      stripes.push_back(
          makeRowVector({makeNullableMapVector<int64_t, int64_t>(rows)}));
      allRows.insert(allRows.end(), rows.begin(), rows.end());
    }
    auto file = writeMultiStripeHybridFlatMapFile(stripes);
    auto expected =
        makeRowVector({makeNullableMapVector<int64_t, int64_t>(allRows)});
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*expected->type());
    auto readers = makeReaders(expected, file, scanSpec);
    validate(*expected, *readers.rowReader, 3);
  }
  {
    SCOPED_TRACE("ROW values");
    // Rows: {1: null, 7: {70}}, {1: {10}, 2: {20}, 3: {null}}, null,
    // {7: null}. Every present group keeps a non-null leaf value.
    auto values = makeRowVector(
        {"val"},
        {makeNullableFlatVector<int64_t>(
            {std::nullopt, 70, 10, 20, std::nullopt, std::nullopt})},
        [](auto row) { return row == 0 || row == 5; });
    auto map = std::make_shared<MapVector>(
        pool(),
        MAP(INTEGER(), ROW("val", BIGINT())),
        makeNulls(4, [](auto row) { return row == 2; }),
        4,
        makeIndices({0, 2, 5, 5}),
        makeIndices({2, 3, 0, 1}),
        makeFlatVector<int32_t>({1, 7, 1, 2, 3, 7}),
        values);
    auto input = makeRowVector({map});
    auto file = writeHybridFlatMapFile(input);
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    auto readers = makeReaders(input, file, scanSpec);
    validate(*input, *readers.rowReader, 10);
  }
  {
    SCOPED_TRACE("ROW values with only null fields");
    // Rows: {1: {null}}, {}.
    auto input = makeRowVector({makeMapVector(
        {0, 1},
        makeFlatVector<int32_t>({1}),
        makeRowVector(
            {"val"}, {makeNullableFlatVector<int64_t>({std::nullopt})}))});
    auto file = writeHybridFlatMapFile(input);
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    auto readers = makeReaders(input, file, scanSpec);
    validate(*input, *readers.rowReader, 10);
  }
}

// Map IS NULL and IS NOT NULL filters. When another filter drops a whole
// batch after the map column was read, the next batch must still find the
// right values. When only another column has a filter, the map column skips
// the rows that filter drops.
TEST_P(HybridFlatMapColumnReaderTest, mapNullFilters) {
  constexpr vector_size_t kSize = 40;
  const auto isNullAt = [](vector_size_t row) { return row % 4 == 3; };
  auto input = makeRowVector({
      makeNullableMapVector<int64_t, int64_t>(makeMapRows(
          kSize,
          {-1, 1, 2, 3, 5, 7, 9},
          isNullAt,
          [](vector_size_t row) { return row % 5 == 0; })),
      makeFlatIdentityVector<int64_t>(kSize),
  });
  auto file = writeHybridFlatMapFile(input);

  for (const bool isNull : {true, false}) {
    SCOPED_TRACE(isNull ? "IS NULL" : "IS NOT NULL");
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    if (isNull) {
      scanSpec->childByName("c0")->setFilter(
          std::make_unique<common::IsNull>());
    } else {
      scanSpec->childByName("c0")->setFilter(
          std::make_unique<common::IsNotNull>());
    }
    auto readers = makeReaders(input, file, scanSpec);
    validateFiltered(*input, *readers.rowReader, 7, [&](vector_size_t row) {
      return isNullAt(row) == isNull;
    });
  }
  {
    SCOPED_TRACE("other filter drops a batch");
    // The map filter sorts before the c1 filter, so c0 reads rows 10-19,
    // including the null maps 11, 15 and 19, and never returns them.
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    scanSpec->childByName("c0")->setFilter(
        std::make_unique<common::IsNotNull>());
    scanSpec->childByName("c1")->setFilter(
        std::make_unique<common::NegatedBigintRange>(10, 19, false));
    scanSpec->disableStatsBasedFilterReorder();
    auto readers = makeReaders(input, file, scanSpec);
    validateFiltered(*input, *readers.rowReader, 10, [&](vector_size_t row) {
      return !isNullAt(row) && (row < 10 || row > 19);
    });
  }
  {
    SCOPED_TRACE("filter on another column only");
    // c0 is read after the c1 filter, up to the last row it passes in each
    // batch of 7. Each later read first skips the rest of the previous batch,
    // such as rows 18 to 20, where map 19 is null and map 20 is empty.
    const auto passes = [](vector_size_t row) {
      return row % 4 == 1 || row % 7 == 3;
    };
    std::vector<int64_t> passingIds;
    for (vector_size_t row = 0; row < kSize; ++row) {
      if (passes(row)) {
        passingIds.push_back(row);
      }
    }
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    scanSpec->childByName("c1")->setFilter(
        common::createBigintValues(passingIds, false));
    auto readers = makeReaders(input, file, scanSpec);
    validateFiltered(*input, *readers.rowReader, 7, passes);
  }
  {
    SCOPED_TRACE("other map filters");
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    scanSpec->childByName("c0")->setFilter(
        std::make_unique<common::BigintRange>(1, 2, false));
    NIMBLE_ASSERT_USER_THROW(
        ([&] {
          auto readers = makeReaders(input, file, scanSpec);
          auto result = BaseVector::create(input->type(), 0, pool());
          readers.rowReader->next(10, result);
        })(),
        "Hybrid FlatMap supports only IS NULL and IS NOT NULL map filters");
  }
}

// Two Hybrid FlatMap columns after a scalar column, loaded lazily at the rows
// the scalar column's filter passes.
TEST_P(HybridFlatMapColumnReaderTest, multipleMapColumns) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      makeMapVector<int32_t, int64_t>({
          {{1, 100}, {2, 200}},
          {{1, 300}},
          {{2, 400}, {3, 500}},
          {{1, 600}, {2, 700}, {3, 800}},
          {},
      }),
      makeMapVector<int32_t, int32_t>({
          {{10, 1}},
          {{20, 2}, {30, 3}},
          {{10, 4}},
          {{20, 5}},
          {{10, 6}, {30, 7}},
      }),
  });
  auto file = test::createNimbleFile(
      *rootPool(),
      input,
      hybridFlatMapWriterOptions(*asRowType(input->type()), {"c1", "c2"}));
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")->setFilter(
      std::make_unique<common::BigintRange>(30, 100, false));
  auto readers = makeReaders(
      input,
      file,
      scanSpec,
      /*preserveFlatMapsInMemory=*/false,
      /*lazyColumnIo=*/true);
  VectorPtr result = BaseVector::create(asRowType(input->type()), 0, pool());
  readers.rowReader->next(100, result);
  ASSERT_EQ(result->size(), 3);
  auto expected = makeRowVector({
      makeFlatVector<int64_t>({30, 40, 50}),
      makeMapVector<int32_t, int64_t>({
          {{2, 400}, {3, 500}},
          {{1, 600}, {2, 700}, {3, 800}},
          {},
      }),
      makeMapVector<int32_t, int32_t>({
          {{10, 4}},
          {{20, 5}},
          {{10, 6}, {30, 7}},
      }),
  });
  velox::test::assertEqualVectors(expected, result);
}

// Value and nulls streams split into many chunks, while the key-presence and
// in-map streams stay in one chunk each.
TEST_P(HybridFlatMapColumnReaderTest, chunkedValueStreams) {
  constexpr vector_size_t kSize = 600;
  const auto isNullAt = [](vector_size_t row) { return row % 11 == 0; };
  const auto isEmptyAt = [](vector_size_t row) { return row % 7 == 0; };
  const std::vector<int64_t> keys{1, 2, 3, 5, 7, 9, 11};
  auto input = makeRowVector({makeNullableMapVector<int64_t, int64_t>(
      makeMapRows(kSize, keys, isNullAt, isEmptyAt))});
  auto writerOptions =
      hybridFlatMapWriterOptions(*asRowType(input->type()), {"c0"});
  writerOptions.minStreamChunkRawSize = 1;
  writerOptions.maxStreamChunkRawSize = 256;
  auto file = test::createNimbleFile(*rootPool(), input, writerOptions);

  {
    SCOPED_TRACE("all keys");
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    auto readers = makeReaders(input, file, scanSpec);
    validate(*input, *readers.rowReader, 64);
  }
  {
    SCOPED_TRACE("one key");
    // Key 9 shares its group with keys 7 and 11, so its values start in a
    // later chunk of the value stream.
    auto expected = makeRowVector({makeNullableMapVector<int64_t, int64_t>(
        makeMapRows(kSize, {9}, isNullAt, isEmptyAt))});
    auto scanSpec = makeKeyFilterScanSpec(
        input->type(), std::make_unique<common::BigintRange>(9, 9, false));
    auto readers = makeReaders(expected, file, scanSpec);
    validate(*expected, *readers.rowReader, 64);
  }
}

// A map read with a wider integer key type than the file's widens the keys,
// also when a Default group lists them.
TEST_P(HybridFlatMapColumnReaderTest, widenedIntegerKeys) {
  auto input = makeRowVector({makeMapVector<int32_t, int64_t>({
      {{1, 10}, {7, 70}},
      {{2, 20}, {9, 90}},
  })});
  auto file = writeHybridFlatMapFile(input);
  auto expected = makeRowVector({makeMapVector<int64_t, int64_t>({
      {{1, 10}, {7, 70}},
      {{2, 20}, {9, 90}},
  })});
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*expected->type());
  auto readers = makeReaders(expected, file, scanSpec);
  validate(*expected, *readers.rowReader, 10);
}

// Every key in no explicit group goes to Default, so a keys filter that passes
// only keys of explicit groups skips the Default group, also when it lists its
// keys per stripe. A filter that passes a Default key reads it.
DEBUG_ONLY_TEST_P(
    HybridFlatMapColumnReaderTest,
    defaultGroupReadOnlyForKeysInNoExplicitGroup) {
  std::vector<uint32_t> builtGroupIds;
  SCOPED_TESTVALUE_SET(
      "facebook::nimble::makeHybridFlatMapGroups",
      std::function<void(uint32_t*)>(
          [&](uint32_t* groupId) { builtGroupIds.push_back(*groupId); }));

  // Keys 1, 5, 2 and 3 are in explicit groups, and keys 7 and 9 in Default.
  auto input = makeRowVector({makeMapVector<int64_t, int64_t>({
      {{1, 10}, {2, 20}, {7, 70}},
      {{3, 30}, {5, 50}, {9, 90}},
  })});
  auto file = writeHybridFlatMapFile(input);
  const auto readsDefault = [&](const std::vector<int64_t>& keys,
                                const VectorPtr& expectedMaps) {
    builtGroupIds.clear();
    auto expected = makeRowVector({expectedMaps});
    auto scanSpec = makeKeyFilterScanSpec(
        input->type(), common::createBigintValues(keys, false));
    auto readers = makeReaders(expected, file, scanSpec);
    validate(*expected, *readers.rowReader, 10);
    return std::find(
               builtGroupIds.begin(),
               builtGroupIds.end(),
               HybridFlatMap::kDefaultGroupId) != builtGroupIds.end();
  };
  {
    SCOPED_TRACE("explicit keys, BigintValues filter");
    // Without explicit groups, every key is in Default.
    EXPECT_EQ(
        readsDefault(
            {1, 5}, makeMapVector<int64_t, int64_t>({{{1, 10}}, {{5, 50}}})),
        GetParam().defaultOnlyGrouping);
  }
  {
    SCOPED_TRACE("explicit keys, BigintRange filter");
    EXPECT_EQ(
        readsDefault(
            {2, 3}, makeMapVector<int64_t, int64_t>({{{2, 20}}, {{3, 30}}})),
        GetParam().defaultOnlyGrouping);
  }
  {
    SCOPED_TRACE("explicit and Default keys");
    EXPECT_TRUE(readsDefault(
        {1, 9}, makeMapVector<int64_t, int64_t>({{{1, 10}}, {{9, 90}}})));
  }
}

// Fuzzes Hybrid FlatMap files and reads for INTEGER, BIGINT and VARCHAR keys.
// Keys in explicit groups, in Default and in no group are written; -1 is a
// grouped key no row holds. The test logs its seed.
TEST_P(HybridFlatMapColumnReaderTest, readFuzz) {
  const auto seed = folly::Random::rand32();
  LOG(INFO) << "readFuzz seed: " << seed;
  SCOPED_TRACE(fmt::format("seed {}", seed));
  std::mt19937 rng{seed};
  constexpr int kIterations{9};
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    SCOPED_TRACE(fmt::format("iteration {}", iteration));
    switch (iteration % 3) {
      case 0:
        fuzzReads<int32_t>(rng, {1, 2, 3, 5, 7, 9, 11, 13, 17}, {-1, 99});
        break;
      case 1:
        fuzzReads<int64_t>(rng, {1, 2, 3, 5, 7, 9, 1'000'000'007}, {-1, 99});
        break;
      default:
        fuzzReads<std::string>(
            rng,
            {"1",
             "2",
             "3",
             "5",
             "7",
             "default_key_long_nine",
             "default_key_long_eleven"},
            {"-1", "absent_key_long"});
        break;
    }
  }
}

TEST_P(HybridFlatMapColumnReaderTest, asStructRejected) {
  auto input = makeRowVector({makeMapVector<int32_t, int64_t>({
      {{1, 10}, {2, 20}},
      {{2, 30}},
  })});
  auto file = writeHybridFlatMapFile(input);
  auto outType = ROW("c0", ROW({"1", "2"}, BIGINT()));
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*outType);
  scanSpec->childByName("c0")->setFlatMapAsStruct(true);
  NIMBLE_ASSERT_USER_THROW(
      ([&] {
        auto readers = makeReaders(input, file, scanSpec);
        VectorPtr batch = BaseVector::create(outType, 0, pool());
        readers.rowReader->next(10, batch);
      })(),
      "Hybrid FlatMap does not support struct output.");
}

TEST_P(HybridFlatMapColumnReaderTest, valueFilterRejected) {
  auto input = makeRowVector({makeMapVector<int32_t, int64_t>({
      {{1, 10}, {2, 20}},
      {{2, 30}},
  })});
  auto file = writeHybridFlatMapFile(input);
  auto scanSpec = std::make_shared<common::ScanSpec>("root");
  scanSpec->addAllChildFields(*input->type());
  scanSpec->childByName("c0")
      ->childByName(common::ScanSpec::kMapValuesFieldName)
      ->setFilter(std::make_unique<common::BigintRange>(10, 20, false));
  NIMBLE_ASSERT_USER_THROW(
      ([&] {
        auto readers = makeReaders(input, file, scanSpec);
        auto result = BaseVector::create(input->type(), 0, pool());
        readers.rowReader->next(10, result);
      })(),
      "Hybrid FlatMap does not support filters on map values.");
}

// Extraction pushdown is rejected. With a delta update the reader returns the
// whole map, and the extraction applies after the update.
TEST_P(HybridFlatMapColumnReaderTest, extractionRejectedUnlessDeltaUpdate) {
  auto input = makeRowVector({makeMapVector<int32_t, int64_t>({
      {{1, 10}, {2, 20}},
      {{2, 30}, {7, 70}},
  })});
  auto file = writeHybridFlatMapFile(input);
  for (const auto extractionType :
       {common::ScanSpec::ExtractionType::kKeys,
        common::ScanSpec::ExtractionType::kSize}) {
    SCOPED_TRACE(static_cast<int>(extractionType));
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    scanSpec->childByName("c0")->setExtractionType(extractionType);
    NIMBLE_ASSERT_USER_THROW(
        ([&] {
          auto readers = makeReaders(input, file, scanSpec);
          auto result = BaseVector::create(input->type(), 0, pool());
          readers.rowReader->next(10, result);
          result->as<RowVector>()->childAt(0)->loadedVector();
        })(),
        "Hybrid FlatMap does not support extraction pushdown.");
  }
  {
    SCOPED_TRACE("delta update");
    class NoOpDeltaColumnUpdater : public dwio::common::DeltaColumnUpdater {
     public:
      void update(const RowSet& /*baseRows*/, VectorPtr& /*result*/) final {}
    };
    NoOpDeltaColumnUpdater deltaUpdate;
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    auto* c0 = scanSpec->childByName("c0");
    c0->setExtractionType(common::ScanSpec::ExtractionType::kKeys);
    c0->setDeltaUpdate(&deltaUpdate);
    auto readers = makeReaders(input, file, scanSpec);
    validate(*input, *readers.rowReader, 10);
  }
  const auto expectChildExtractionThrows = [&](std::string_view childName,
                                               std::string_view message) {
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*input->type());
    scanSpec->childByName("c0")
        ->childByName(std::string(childName))
        ->setExtractionType(common::ScanSpec::ExtractionType::kSize);
    NIMBLE_ASSERT_USER_THROW(
        ([&] {
          auto readers = makeReaders(input, file, scanSpec);
          auto result = BaseVector::create(input->type(), 0, pool());
          readers.rowReader->next(10, result);
          result->as<RowVector>()->childAt(0)->loadedVector();
        })(),
        message);
  };
  {
    SCOPED_TRACE("map keys");
    expectChildExtractionThrows(
        common::ScanSpec::kMapKeysFieldName,
        "Hybrid FlatMap does not support extraction pushdown on map keys.");
  }
  {
    SCOPED_TRACE("map values");
    expectChildExtractionThrows(
        common::ScanSpec::kMapValuesFieldName,
        "Hybrid FlatMap does not support extraction pushdown on map values.");
  }
}

INSTANTIATE_TEST_SUITE_P(
    HybridFlatMapColumnReaderTests,
    HybridFlatMapColumnReaderTest,
    ::testing::Values(
        HybridFlatMapColumnReaderTestParam{
            false,
            false,
            HybridFlatMap::KeyPresenceMode::kBitmap},
        HybridFlatMapColumnReaderTestParam{
            true,
            true,
            HybridFlatMap::KeyPresenceMode::kBitmap},
        HybridFlatMapColumnReaderTestParam{
            false,
            false,
            HybridFlatMap::KeyPresenceMode::kKeyList},
        HybridFlatMapColumnReaderTestParam{
            false,
            true,
            HybridFlatMap::KeyPresenceMode::kKeyList}),
    [](const ::testing::TestParamInfo<HybridFlatMapColumnReaderTestParam>&
           info) { return testName(info.param); });

} // namespace
} // namespace facebook::nimble
