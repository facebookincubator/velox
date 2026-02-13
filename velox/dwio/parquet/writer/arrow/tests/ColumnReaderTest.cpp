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

// Adapted from Apache Arrow.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/array/array_binary.h"
#include "arrow/util/macros.h"

#include "velox/dwio/parquet/writer/arrow/ColumnPage.h"
#include "velox/dwio/parquet/writer/arrow/Schema.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"
#include "velox/dwio/parquet/writer/arrow/tests/ColumnReader.h"
#include "velox/dwio/parquet/writer/arrow/tests/TestUtil.h"

namespace facebook::velox::parquet::arrow {

using ParquetType = parquet::Type;

using internal::BinaryRecordReader;
using schema::GroupNode;
using schema::NodePtr;
using schema::PrimitiveNode;
using testing::ElementsAre;

namespace test {

template <typename T>
static inline bool vectorEqualWithDefLevels(
    const std::vector<T>& left,
    const std::vector<int16_t>& defLevels,
    int16_t maxDefLevels,
    int16_t maxRepLevels,
    const std::vector<T>& right) {
  size_t iLeft = 0;
  size_t iRight = 0;
  for (size_t i = 0; i < defLevels.size(); i++) {
    if (defLevels[i] == maxDefLevels) {
      // Compare.
      if (left[iLeft] != right[iRight]) {
        std::cerr << "index " << i << " left was " << left[iLeft]
                  << " right was " << right[i] << std::endl;
        return false;
      }
      iLeft++;
      iRight++;
    } else if (defLevels[i] == (maxDefLevels - 1)) {
      // Null entry on the lowest nested level.
      iRight++;
    } else if (defLevels[i] < (maxDefLevels - 1)) {
      // Null entry on a higher nesting level, only supported for non-repeating.
      // Data.
      if (maxRepLevels == 0) {
        iRight++;
      }
    }
  }

  return true;
}

class TestPrimitiveReader : public ::testing::Test {
 public:
  void initReader(const ColumnDescriptor* d) {
    auto pager = std::make_unique<MockPageReader>(pages_);
    reader_ = ColumnReader::make(d, std::move(pager));
  }

  void checkResults() {
    std::vector<int32_t> vresult(numValues_, -1);
    std::vector<int16_t> dresult(numLevels_, -1);
    std::vector<int16_t> rresult(numLevels_, -1);
    int64_t valuesRead = 0;
    int totalValuesRead = 0;
    int batchActual = 0;

    Int32Reader* reader = static_cast<Int32Reader*>(reader_.get());
    int32_t batchSize = 8;
    int batch = 0;
    // This will cover both the cases.
    // 1) batch_size < page_size (multiple ReadBatch from a single page)
    // 2) batch_size > page_size (BatchRead limits to a single page)
    do {
      batch = static_cast<int>(reader->readBatch(
          batchSize,
          &dresult[0] + batchActual,
          &rresult[0] + batchActual,
          &vresult[0] + totalValuesRead,
          &valuesRead));
      totalValuesRead += static_cast<int>(valuesRead);
      batchActual += batch;
      batchSize = std::min(1 << 24, std::max(batchSize * 2, 4096));
    } while (batch > 0);

    ASSERT_EQ(numLevels_, batchActual);
    ASSERT_EQ(numValues_, totalValuesRead);
    ASSERT_TRUE(vectorEqual(values_, vresult));
    if (maxDefLevel_ > 0) {
      ASSERT_TRUE(vectorEqual(defLevels_, dresult));
    }
    if (maxRepLevel_ > 0) {
      ASSERT_TRUE(vectorEqual(repLevels_, rresult));
    }
    // Catch improper writes at EOS.
    batchActual = static_cast<int>(
        reader->readBatch(5, nullptr, nullptr, nullptr, &valuesRead));
    ASSERT_EQ(0, batchActual);
    ASSERT_EQ(0, valuesRead);
  }
  void checkResultsSpaced() {
    std::vector<int32_t> vresult(numLevels_, -1);
    std::vector<int16_t> dresult(numLevels_, -1);
    std::vector<int16_t> rresult(numLevels_, -1);
    std::vector<uint8_t> validBits(numLevels_, 255);
    int totalValuesRead = 0;
    int batchActual = 0;
    int levelsActual = 0;
    int64_t nullCount = -1;
    int64_t levelsRead = 0;
    int64_t valuesRead;

    Int32Reader* reader = static_cast<Int32Reader*>(reader_.get());
    int32_t batchSize = 8;
    int batch = 0;
    // This will cover both the cases.
    // 1) batch_size < page_size (multiple ReadBatch from a single page)
    // 2) batch_size > page_size (BatchRead limits to a single page)
    do {
      ARROW_SUPPRESS_DEPRECATION_WARNING
      batch = static_cast<int>(reader->readBatchSpaced(
          batchSize,
          dresult.data() + levelsActual,
          rresult.data() + levelsActual,
          vresult.data() + batchActual,
          validBits.data() + batchActual,
          0,
          &levelsRead,
          &valuesRead,
          &nullCount));
      ARROW_UNSUPPRESS_DEPRECATION_WARNING
      totalValuesRead += batch - static_cast<int>(nullCount);
      batchActual += batch;
      levelsActual += static_cast<int>(levelsRead);
      batchSize = std::min(1 << 24, std::max(batchSize * 2, 4096));
    } while ((batch > 0) || (levelsRead > 0));

    ASSERT_EQ(numLevels_, levelsActual);
    ASSERT_EQ(numValues_, totalValuesRead);
    if (maxDefLevel_ > 0) {
      ASSERT_TRUE(vectorEqual(defLevels_, dresult));
      ASSERT_TRUE(vectorEqualWithDefLevels(
          values_, dresult, maxDefLevel_, maxRepLevel_, vresult));
    } else {
      ASSERT_TRUE(vectorEqual(values_, vresult));
    }
    if (maxRepLevel_ > 0) {
      ASSERT_TRUE(vectorEqual(repLevels_, rresult));
    }
    // Catch improper writes at EOS.
    ARROW_SUPPRESS_DEPRECATION_WARNING
    batchActual = static_cast<int>(reader->readBatchSpaced(
        5,
        nullptr,
        nullptr,
        nullptr,
        validBits.data(),
        0,
        &levelsRead,
        &valuesRead,
        &nullCount));
    ARROW_UNSUPPRESS_DEPRECATION_WARNING
    ASSERT_EQ(0, batchActual);
    ASSERT_EQ(0, nullCount);
  }

  void clear() {
    values_.clear();
    defLevels_.clear();
    repLevels_.clear();
    pages_.clear();
    reader_.reset();
  }

  void
  executePlain(int numPages, int levelsPerPage, const ColumnDescriptor* d) {
    numValues_ = makePages<Int32Type>(
        d,
        numPages,
        levelsPerPage,
        defLevels_,
        repLevels_,
        values_,
        dataBuffer_,
        pages_,
        Encoding::kPlain);
    numLevels_ = numPages * levelsPerPage;
    initReader(d);
    checkResults();
    clear();

    numValues_ = makePages<Int32Type>(
        d,
        numPages,
        levelsPerPage,
        defLevels_,
        repLevels_,
        values_,
        dataBuffer_,
        pages_,
        Encoding::kPlain);
    numLevels_ = numPages * levelsPerPage;
    initReader(d);
    checkResultsSpaced();
    clear();
  }

  void executeDict(int numPages, int levelsPerPage, const ColumnDescriptor* d) {
    numValues_ = makePages<Int32Type>(
        d,
        numPages,
        levelsPerPage,
        defLevels_,
        repLevels_,
        values_,
        dataBuffer_,
        pages_,
        Encoding::kRleDictionary);
    numLevels_ = numPages * levelsPerPage;
    initReader(d);
    checkResults();
    clear();

    numValues_ = makePages<Int32Type>(
        d,
        numPages,
        levelsPerPage,
        defLevels_,
        repLevels_,
        values_,
        dataBuffer_,
        pages_,
        Encoding::kRleDictionary);
    numLevels_ = numPages * levelsPerPage;
    initReader(d);
    checkResultsSpaced();
    clear();
  }

 protected:
  int numLevels_;
  int numValues_;
  int16_t maxDefLevel_;
  int16_t maxRepLevel_;
  std::vector<std::shared_ptr<Page>> pages_;
  std::shared_ptr<ColumnReader> reader_;
  std::vector<int32_t> values_;
  std::vector<int16_t> defLevels_;
  std::vector<int16_t> repLevels_;
  std::vector<uint8_t> dataBuffer_; // For BA and FLBA
};

TEST_F(TestPrimitiveReader, TestInt32FlatRequired) {
  int levelsPerPage = 100;
  int numPages = 50;
  maxDefLevel_ = 0;
  maxRepLevel_ = 0;
  NodePtr type = schema::int32("a", Repetition::kRequired);
  const ColumnDescriptor descr(type, maxDefLevel_, maxRepLevel_);
  ASSERT_NO_FATAL_FAILURE(executePlain(numPages, levelsPerPage, &descr));
  ASSERT_NO_FATAL_FAILURE(executeDict(numPages, levelsPerPage, &descr));
}

TEST_F(TestPrimitiveReader, TestInt32FlatOptional) {
  int levelsPerPage = 100;
  int numPages = 50;
  maxDefLevel_ = 4;
  maxRepLevel_ = 0;
  NodePtr type = schema::int32("b", Repetition::kOptional);
  const ColumnDescriptor descr(type, maxDefLevel_, maxRepLevel_);
  ASSERT_NO_FATAL_FAILURE(executePlain(numPages, levelsPerPage, &descr));
  ASSERT_NO_FATAL_FAILURE(executeDict(numPages, levelsPerPage, &descr));
}

TEST_F(TestPrimitiveReader, TestInt32FlatRepeated) {
  int levelsPerPage = 100;
  int numPages = 50;
  maxDefLevel_ = 4;
  maxRepLevel_ = 2;
  NodePtr type = schema::int32("c", Repetition::kRepeated);
  const ColumnDescriptor descr(type, maxDefLevel_, maxRepLevel_);
  ASSERT_NO_FATAL_FAILURE(executePlain(numPages, levelsPerPage, &descr));
  ASSERT_NO_FATAL_FAILURE(executeDict(numPages, levelsPerPage, &descr));
}

// Tests skipping around page boundaries.
TEST_F(TestPrimitiveReader, TestSkipAroundPageBoundries) {
  int levelsPerPage = 100;
  int numPages = 7;
  maxDefLevel_ = 0;
  maxRepLevel_ = 0;
  NodePtr type = schema::int32("b", Repetition::kRequired);
  const ColumnDescriptor descr(type, maxDefLevel_, maxRepLevel_);
  makePages<Int32Type>(
      &descr,
      numPages,
      levelsPerPage,
      defLevels_,
      repLevels_,
      values_,
      dataBuffer_,
      pages_,
      Encoding::kPlain);
  initReader(&descr);
  std::vector<int32_t> vresult(levelsPerPage / 2, -1);
  std::vector<int16_t> dresult(levelsPerPage / 2, -1);
  std::vector<int16_t> rresult(levelsPerPage / 2, -1);

  Int32Reader* reader = static_cast<Int32Reader*>(reader_.get());
  int64_t valuesRead = 0;

  // 1) skip_size > page_size (multiple pages skipped)
  // Skip first 2 pages.
  int64_t levelsSkipped = reader->skip(2 * levelsPerPage);
  ASSERT_EQ(2 * levelsPerPage, levelsSkipped);
  // Read half a page.
  reader->readBatch(
      levelsPerPage / 2,
      dresult.data(),
      rresult.data(),
      vresult.data(),
      &valuesRead);
  std::vector<int32_t> subValues(
      values_.begin() + 2 * levelsPerPage,
      values_.begin() + static_cast<int>(2.5 * levelsPerPage));
  ASSERT_TRUE(vectorEqual(subValues, vresult));

  // 2) skip_size == page_size (skip across two pages from page 2.5 to 3.5)
  levelsSkipped = reader->skip(levelsPerPage);
  ASSERT_EQ(levelsPerPage, levelsSkipped);
  // Read half a page (page 3.5 to 4)
  reader->readBatch(
      levelsPerPage / 2,
      dresult.data(),
      rresult.data(),
      vresult.data(),
      &valuesRead);
  subValues.clear();
  subValues.insert(
      subValues.end(),
      values_.begin() + static_cast<int>(3.5 * levelsPerPage),
      values_.begin() + 4 * levelsPerPage);
  ASSERT_TRUE(vectorEqual(subValues, vresult));

  // 3) skip_size == page_size (skip page 4 from start of the page to the end)
  levelsSkipped = reader->skip(levelsPerPage);
  ASSERT_EQ(levelsPerPage, levelsSkipped);
  // Read half a page (page 5 to 5.5)
  reader->readBatch(
      levelsPerPage / 2,
      dresult.data(),
      rresult.data(),
      vresult.data(),
      &valuesRead);
  subValues.clear();
  subValues.insert(
      subValues.end(),
      values_.begin() + static_cast<int>(5.0 * levelsPerPage),
      values_.begin() + static_cast<int>(5.5 * levelsPerPage));
  ASSERT_TRUE(vectorEqual(subValues, vresult));

  // 4) skip_size < page_size (skip limited to a single page)
  // Skip half a page (page 5.5 to 6)
  levelsSkipped = reader->skip(levelsPerPage / 2);
  ASSERT_EQ(0.5 * levelsPerPage, levelsSkipped);
  // Read half a page (6 to 6.5)
  reader->readBatch(
      levelsPerPage / 2,
      dresult.data(),
      rresult.data(),
      vresult.data(),
      &valuesRead);
  subValues.clear();
  subValues.insert(
      subValues.end(),
      values_.begin() + static_cast<int>(6.0 * levelsPerPage),
      values_.begin() + static_cast<int>(6.5 * levelsPerPage));
  ASSERT_TRUE(vectorEqual(subValues, vresult));

  // 5) Skip_size = 0.
  levelsSkipped = reader->skip(0);
  ASSERT_EQ(0, levelsSkipped);

  // 6) Skip past the end page.
  levelsSkipped = reader->skip(levelsPerPage / 2 + 10);
  ASSERT_EQ(levelsPerPage / 2, levelsSkipped);

  values_.clear();
  defLevels_.clear();
  repLevels_.clear();
  pages_.clear();
  reader_.reset();
}

// Skip with repeated field. This test makes it clear that we are skipping.
// Values and not records.
TEST_F(TestPrimitiveReader, TestSkipRepeatedField) {
  // Example schema: message M { repeated int32 b = 1 }
  maxDefLevel_ = 1;
  maxRepLevel_ = 1;
  NodePtr type = schema::int32("b", Repetition::kRepeated);
  const ColumnDescriptor descr(type, maxDefLevel_, maxRepLevel_);
  // Example rows: {}, {[10, 10]}, {[20, 20, 20]}
  std::vector<int32_t> values = {10, 10, 20, 20, 20};
  std::vector<int16_t> defLevels = {0, 1, 1, 1, 1, 1};
  std::vector<int16_t> repLevels = {0, 0, 1, 0, 1, 1};
  numValues_ = static_cast<int>(defLevels.size());
  std::shared_ptr<DataPageV1> page = makeDataPage<Int32Type>(
      &descr,
      values,
      numValues_,
      Encoding::kPlain,
      {},
      0,
      defLevels,
      maxDefLevel_,
      repLevels,
      maxRepLevel_);

  pages_.push_back(std::move(page));

  initReader(&descr);
  Int32Reader* reader = static_cast<Int32Reader*>(reader_.get());

  // Vecotrs to hold read values, definition levels, and repetition levels.
  std::vector<int32_t> readVals(4, -1);
  std::vector<int16_t> readDefs(4, -1);
  std::vector<int16_t> readReps(4, -1);

  // Skip two levels.
  int64_t levelsSkipped = reader->skip(2);
  ASSERT_EQ(2, levelsSkipped);

  int64_t numReadValues = 0;
  // Read the next set of values.
  reader->readBatch(
      10, readDefs.data(), readReps.data(), readVals.data(), &numReadValues);
  ASSERT_EQ(numReadValues, 4);
  // Note that we end up in the record with {[10, 10]}
  ASSERT_TRUE(vectorEqual({10, 20, 20, 20}, readVals));
  ASSERT_TRUE(vectorEqual({1, 1, 1, 1}, readDefs));
  ASSERT_TRUE(vectorEqual({1, 0, 1, 1}, readReps));

  // No values remain in data page.
  levelsSkipped = reader->skip(2);
  ASSERT_EQ(0, levelsSkipped);
  reader->readBatch(
      10, readDefs.data(), readReps.data(), readVals.data(), &numReadValues);
  ASSERT_EQ(numReadValues, 0);
}

// Page claims to have two values but only 1 is present.
TEST_F(TestPrimitiveReader, TestReadValuesMissing) {
  maxDefLevel_ = 1;
  maxRepLevel_ = 0;
  constexpr int batchSize = 1;
  std::vector<bool> values(1, false);
  std::vector<int16_t> inputDefLevels(1, 1);
  NodePtr type = schema::boolean("a", Repetition::kOptional);
  const ColumnDescriptor descr(type, maxDefLevel_, maxRepLevel_);

  // The data page falls back to plain encoding.
  std::shared_ptr<ResizableBuffer> dummy = allocateBuffer();
  std::shared_ptr<DataPageV1> dataPage = makeDataPage<BooleanType>(
      &descr,
      values,
      2,
      Encoding::kPlain,
      {},
      0,
      inputDefLevels,
      maxDefLevel_,
      {},
      0);
  pages_.push_back(dataPage);
  initReader(&descr);
  auto reader = static_cast<BoolReader*>(reader_.get());
  ASSERT_TRUE(reader->hasNext());
  std::vector<int16_t> defLevels(batchSize, 0);
  std::vector<int16_t> repLevels(batchSize, 0);
  bool valuesOut[batchSize];
  int64_t valuesRead;
  EXPECT_EQ(
      1,
      reader->readBatch(
          batchSize,
          defLevels.data(),
          repLevels.data(),
          valuesOut,
          &valuesRead));
  ASSERT_THROW(
      reader->readBatch(
          batchSize,
          defLevels.data(),
          repLevels.data(),
          valuesOut,
          &valuesRead),
      ParquetException);
}

// Repetition level byte length reported in Page but Max Repetition level.
// Is zero for the column.
TEST_F(TestPrimitiveReader, TestRepetitionLvlBytesWithMaxRepetitionZero) {
  constexpr int batchSize = 4;
  maxDefLevel_ = 1;
  maxRepLevel_ = 0;
  NodePtr type = schema::int32("a", Repetition::kOptional);
  const ColumnDescriptor descr(type, maxDefLevel_, maxRepLevel_);
  // Bytes here came from the example parquet file in ARROW-17453's int32.
  // Column which was delta bit-packed. The key part is the first three.
  // Bytes: the page header reports 1 byte for repetition levels even.
  // Though the max rep level is 0. If that byte isn't skipped then.
  // We get def levels of [1, 1, 0, 0] instead of the correct [1, 1, 1, 0].
  const std::vector<uint8_t> pageData{0x3,  0x3, 0x7, 0x80, 0x1, 0x4, 0x3,
                                      0x18, 0x1, 0x2, 0x0,  0x0, 0x0, 0xc,
                                      0x0,  0x0, 0x0, 0x0,  0x0, 0x0, 0x0};

  std::shared_ptr<DataPageV2> dataPage = std::make_shared<DataPageV2>(
      Buffer::Wrap(pageData.data(), pageData.size()),
      4,
      1,
      4,
      Encoding::kDeltaBinaryPacked,
      2,
      1,
      21);

  pages_.push_back(dataPage);
  initReader(&descr);
  auto reader = static_cast<Int32Reader*>(reader_.get());
  int16_t defLevelsOut[batchSize];
  int32_t values[batchSize];
  int64_t valuesRead;
  ASSERT_TRUE(reader->hasNext());
  EXPECT_EQ(
      4,
      reader->readBatch(batchSize, defLevelsOut, nullptr, values, &valuesRead));
  EXPECT_EQ(3, valuesRead);
}

// Page claims to have two values but only 1 is present.
TEST_F(TestPrimitiveReader, TestReadValuesMissingWithDictionary) {
  constexpr int batchSize = 1;
  maxDefLevel_ = 1;
  maxRepLevel_ = 0;
  NodePtr type = schema::int32("a", Repetition::kOptional);
  const ColumnDescriptor descr(type, maxDefLevel_, maxRepLevel_);
  std::shared_ptr<ResizableBuffer> dummy = allocateBuffer();

  std::shared_ptr<DictionaryPage> dictPage =
      std::make_shared<DictionaryPage>(dummy, 0, Encoding::kPlain);
  std::vector<int16_t> inputDefLevels(1, 0);
  std::shared_ptr<DataPageV1> dataPage = makeDataPage<Int32Type>(
      &descr,
      {},
      2,
      Encoding::kRleDictionary,
      {},
      0,
      inputDefLevels,
      maxDefLevel_,
      {},
      0);
  pages_.push_back(dictPage);
  pages_.push_back(dataPage);
  initReader(&descr);
  auto reader = static_cast<ByteArrayReader*>(reader_.get());
  const ByteArray* dict = nullptr;
  int32_t dictLen = 0;
  int64_t indicesRead = 0;
  int32_t indices[batchSize];
  int16_t defLevelsOut[batchSize];
  ASSERT_TRUE(reader->hasNext());
  EXPECT_EQ(
      1,
      reader->readBatchWithDictionary(
          batchSize,
          defLevelsOut,
          nullptr,
          indices,
          &indicesRead,
          &dict,
          &dictLen));
  ASSERT_THROW(
      reader->readBatchWithDictionary(
          batchSize,
          defLevelsOut,
          nullptr,
          indices,
          &indicesRead,
          &dict,
          &dictLen),
      ParquetException);
}

TEST_F(TestPrimitiveReader, TestDictionaryEncodedPages) {
  maxDefLevel_ = 0;
  maxRepLevel_ = 0;
  NodePtr type = schema::int32("a", Repetition::kRequired);
  const ColumnDescriptor descr(type, maxDefLevel_, maxRepLevel_);
  std::shared_ptr<ResizableBuffer> dummy = allocateBuffer();

  std::shared_ptr<DictionaryPage> dictPage =
      std::make_shared<DictionaryPage>(dummy, 0, Encoding::kPlain);
  std::shared_ptr<DataPageV1> dataPage = makeDataPage<Int32Type>(
      &descr, {}, 0, Encoding::kRleDictionary, {}, 0, {}, 0, {}, 0);
  pages_.push_back(dictPage);
  pages_.push_back(dataPage);
  initReader(&descr);
  // Tests Dict : PLAIN, Data : RLE_DICTIONARY.
  ASSERT_NO_THROW(reader_->hasNext());
  pages_.clear();

  dictPage =
      std::make_shared<DictionaryPage>(dummy, 0, Encoding::kPlainDictionary);
  dataPage = makeDataPage<Int32Type>(
      &descr, {}, 0, Encoding::kPlainDictionary, {}, 0, {}, 0, {}, 0);
  pages_.push_back(dictPage);
  pages_.push_back(dataPage);
  initReader(&descr);
  // Tests Dict : PLAIN_DICTIONARY, Data : PLAIN_DICTIONARY.
  ASSERT_NO_THROW(reader_->hasNext());
  pages_.clear();

  dataPage = makeDataPage<Int32Type>(
      &descr, {}, 0, Encoding::kRleDictionary, {}, 0, {}, 0, {}, 0);
  pages_.push_back(dataPage);
  initReader(&descr);
  // Tests dictionary page must occur before data page.
  ASSERT_THROW(reader_->hasNext(), ParquetException);
  pages_.clear();

  dictPage =
      std::make_shared<DictionaryPage>(dummy, 0, Encoding::kDeltaByteArray);
  pages_.push_back(dictPage);
  initReader(&descr);
  // Tests only RLE_DICTIONARY is supported.
  ASSERT_THROW(reader_->hasNext(), ParquetException);
  pages_.clear();

  std::shared_ptr<DictionaryPage> dictPage1 =
      std::make_shared<DictionaryPage>(dummy, 0, Encoding::kPlainDictionary);
  std::shared_ptr<DictionaryPage> dictPage2 =
      std::make_shared<DictionaryPage>(dummy, 0, Encoding::kPlain);
  pages_.push_back(dictPage1);
  pages_.push_back(dictPage2);
  initReader(&descr);
  // Column cannot have more than one dictionary.
  ASSERT_THROW(reader_->hasNext(), ParquetException);
  pages_.clear();

  dataPage = makeDataPage<Int32Type>(
      &descr, {}, 0, Encoding::kDeltaByteArray, {}, 0, {}, 0, {}, 0);
  pages_.push_back(dataPage);
  initReader(&descr);
  // Unsupported encoding.
  ASSERT_THROW(reader_->hasNext(), ParquetException);
  pages_.clear();
}

TEST_F(TestPrimitiveReader, TestDictionaryEncodedPagesWithExposeEncoding) {
  maxDefLevel_ = 0;
  maxRepLevel_ = 0;
  int levelsPerPage = 100;
  int numPages = 5;
  std::vector<int16_t> defLevels;
  std::vector<int16_t> repLevels;
  std::vector<ByteArray> values;
  std::vector<uint8_t> buffer;
  NodePtr type = schema::byteArray("a", Repetition::kRequired);
  const ColumnDescriptor descr(type, maxDefLevel_, maxRepLevel_);

  // Fully dictionary encoded.
  makePages<ByteArrayType>(
      &descr,
      numPages,
      levelsPerPage,
      defLevels,
      repLevels,
      values,
      buffer,
      pages_,
      Encoding::kRleDictionary);
  initReader(&descr);

  auto reader = static_cast<ByteArrayReader*>(reader_.get());
  const ByteArray* dict = nullptr;
  int32_t dictLen = 0;
  int64_t totalIndices = 0;
  int64_t indicesRead = 0;
  int64_t valueSize = values.size();
  auto indices = std::make_unique<int32_t[]>(valueSize);
  while (totalIndices < valueSize && reader->hasNext()) {
    const ByteArray* tmpDict = nullptr;
    int32_t tmpDictLen = 0;
    EXPECT_NO_THROW(reader->readBatchWithDictionary(
        valueSize,
        nullptr,
        nullptr,
        indices.get() + totalIndices,
        &indicesRead,
        &tmpDict,
        &tmpDictLen));
    if (tmpDict != nullptr) {
      // Dictionary is read along with data.
      EXPECT_GT(indicesRead, 0);
      dict = tmpDict;
      dictLen = tmpDictLen;
    } else {
      // Dictionary is not read when there's no data.
      EXPECT_EQ(indicesRead, 0);
    }
    totalIndices += indicesRead;
  }

  EXPECT_EQ(totalIndices, valueSize);
  for (int64_t i = 0; i < totalIndices; ++i) {
    EXPECT_LT(indices[i], dictLen);
    EXPECT_EQ(dict[indices[i]].len, values[i].len);
    EXPECT_EQ(memcmp(dict[indices[i]].ptr, values[i].ptr, values[i].len), 0);
  }
  pages_.clear();
}

TEST_F(TestPrimitiveReader, TestNonDictionaryEncodedPagesWithExposeEncoding) {
  maxDefLevel_ = 0;
  maxRepLevel_ = 0;
  int64_t valueSize = 100;
  std::vector<int32_t> values(valueSize, 0);
  NodePtr type = schema::int32("a", Repetition::kRequired);
  const ColumnDescriptor descr(type, maxDefLevel_, maxRepLevel_);

  // The data page falls back to plain encoding.
  std::shared_ptr<ResizableBuffer> dummy = allocateBuffer();
  std::shared_ptr<DictionaryPage> dictPage =
      std::make_shared<DictionaryPage>(dummy, 0, Encoding::kPlain);
  std::shared_ptr<DataPageV1> dataPage = makeDataPage<Int32Type>(
      &descr,
      values,
      static_cast<int>(valueSize),
      Encoding::kPlain,
      {},
      0,
      {},
      0,
      {},
      0);
  pages_.push_back(dictPage);
  pages_.push_back(dataPage);
  initReader(&descr);

  auto reader = static_cast<ByteArrayReader*>(reader_.get());
  const ByteArray* dict = nullptr;
  int32_t dictLen = 0;
  int64_t indicesRead = 0;
  auto indices = std::make_unique<int32_t[]>(valueSize);
  // Dictionary cannot be exposed when it's not fully dictionary encoded.
  EXPECT_THROW(
      reader->readBatchWithDictionary(
          valueSize,
          nullptr,
          nullptr,
          indices.get(),
          &indicesRead,
          &dict,
          &dictLen),
      ParquetException);
  pages_.clear();
}

namespace {

LevelInfo computeLevelInfo(const ColumnDescriptor* descr) {
  LevelInfo levelInfo;
  levelInfo.defLevel = descr->maxDefinitionLevel();
  levelInfo.repLevel = descr->maxRepetitionLevel();

  int16_t minSpacedDefLevel = descr->maxDefinitionLevel();
  const schema::Node* Node = descr->schemaNode().get();
  while (Node != nullptr && !Node->isRepeated()) {
    if (Node->isOptional()) {
      minSpacedDefLevel--;
    }
    Node = Node->parent();
  }
  levelInfo.repeatedAncestorDefLevel = minSpacedDefLevel;
  return levelInfo;
}

} // namespace

using ReadDenseForNullable = bool;
class RecordReaderPrimitiveTypeTest
    : public ::testing::TestWithParam<ReadDenseForNullable> {
 public:
  const int32_t kNullValue = -1;

  void init(NodePtr column) {
    NodePtr root = GroupNode::make("root", Repetition::kRequired, {column});
    schemaDescriptor_.init(root);
    descr_ = schemaDescriptor_.column(0);
    recordReader_ = internal::RecordReader::make(
        descr_,
        computeLevelInfo(descr_),
        ::arrow::default_memory_pool(),
        false,
        GetParam());
  }

  void checkReadValues(
      std::vector<int32_t> expectedValues,
      std::vector<int16_t> expectedDefs,
      std::vector<int16_t> expectedReps) {
    const auto readValues =
        reinterpret_cast<const int32_t*>(recordReader_->values());
    std::vector<int32_t> readVals(
        readValues, readValues + recordReader_->valuesWritten());
    ASSERT_EQ(readVals.size(), expectedValues.size());
    for (size_t i = 0; i < expectedValues.size(); ++i) {
      if (expectedValues[i] != kNullValue) {
        ASSERT_EQ(expectedValues[i], readValues[i]);
      }
    }

    if (!descr_->schemaNode()->isRequired()) {
      std::vector<int16_t> readDefs(
          recordReader_->defLevels(),
          recordReader_->defLevels() + recordReader_->levelsPosition());
      ASSERT_TRUE(vectorEqual(expectedDefs, readDefs));
    }

    if (descr_->schemaNode()->isRepeated()) {
      std::vector<int16_t> readReps(
          recordReader_->repLevels(),
          recordReader_->repLevels() + recordReader_->levelsPosition());
      ASSERT_TRUE(vectorEqual(expectedReps, readReps));
    }
  }

  void checkState(
      int64_t valuesWritten,
      int64_t nullCount,
      int64_t levelsWritten,
      int64_t levelsPosition) {
    ASSERT_EQ(recordReader_->valuesWritten(), valuesWritten);
    ASSERT_EQ(recordReader_->nullCount(), nullCount);
    ASSERT_EQ(recordReader_->levelsWritten(), levelsWritten);
    ASSERT_EQ(recordReader_->levelsPosition(), levelsPosition);
  }

 protected:
  SchemaDescriptor schemaDescriptor_;
  std::shared_ptr<internal::RecordReader> recordReader_;
  const ColumnDescriptor* descr_;
};

// Tests reading a required field. The expected results are the same for.
// Reading dense and spaced.
TEST_P(RecordReaderPrimitiveTypeTest, ReadRequired) {
  init(schema::int32("b", Repetition::kRequired));

  // Records look like: {10, 20, 20, 30, 30, 30}
  std::vector<std::shared_ptr<Page>> pages;
  std::vector<int32_t> values = {10, 20, 20, 30, 30, 30};
  std::vector<int16_t> defLevels = {};
  std::vector<int16_t> repLevels = {};

  std::shared_ptr<DataPageV1> page = makeDataPage<Int32Type>(
      descr_,
      values,
      static_cast<int>(defLevels.size()),
      Encoding::kPlain,
      {},
      0,
      defLevels,
      descr_->maxDefinitionLevel(),
      repLevels,
      descr_->maxRepetitionLevel());
  pages.push_back(std::move(page));
  auto pager = std::make_unique<MockPageReader>(pages);
  recordReader_->setPageReader(std::move(pager));

  // Read [10].
  int64_t recordsRead = recordReader_->readRecords(1);
  ASSERT_EQ(recordsRead, 1);
  checkState(1, 0, 0, 0);
  checkReadValues({10}, {}, {});
  recordReader_->reset();
  checkState(0, 0, 0, 0);

  // Read 20, 20, 30, 30, 30.
  recordsRead = recordReader_->readRecords(10);
  ASSERT_EQ(recordsRead, 5);
  checkState(5, 0, 0, 0);
  checkReadValues({20, 20, 30, 30, 30}, {}, {});
  recordReader_->reset();
  checkState(0, 0, 0, 0);
}

// Tests reading an optional field.
// Use a max definition field > 1 to test both cases where parent is present or.
// Parent is missing.
TEST_P(RecordReaderPrimitiveTypeTest, ReadOptional) {
  NodePtr column = GroupNode::make(
      "a",
      Repetition::kOptional,
      {PrimitiveNode::make(
          "element", Repetition::kOptional, ParquetType::kInt32)});
  init(column);

  // Records look like: {10, null, 20, 20, null, 30, 30, 30, null}
  std::vector<std::shared_ptr<Page>> pages;
  std::vector<int32_t> values = {10, 20, 20, 30, 30, 30};
  std::vector<int16_t> defLevels = {2, 0, 2, 2, 1, 2, 2, 2, 0};

  std::shared_ptr<DataPageV1> page = makeDataPage<Int32Type>(
      descr_,
      values,
      static_cast<int>(defLevels.size()),
      Encoding::kPlain,
      {},
      0,
      defLevels,
      descr_->maxDefinitionLevel(),
      {},
      descr_->maxRepetitionLevel());
  pages.push_back(std::move(page));
  auto pager = std::make_unique<MockPageReader>(pages);
  recordReader_->setPageReader(std::move(pager));

  // Read 10, null.
  int64_t recordsRead = recordReader_->readRecords(2);
  ASSERT_EQ(recordsRead, 2);
  if (GetParam() == true) {
    checkState(1, 0, 9, 2);
    checkReadValues({10}, {2, 0}, {});
  } else {
    checkState(2, 1, 9, 2);
    checkReadValues({10, kNullValue}, {2, 0}, {});
  }
  recordReader_->reset();
  checkState(0, 0, 7, 0);

  // Read 20, 20, null (parent present), 30, 30, 30.
  recordsRead = recordReader_->readRecords(6);
  ASSERT_EQ(recordsRead, 6);
  if (GetParam() == true) {
    checkState(5, 0, 7, 6);
    checkReadValues({20, 20, 30, 30, 30}, {2, 2, 1, 2, 2, 2}, {});
  } else {
    checkState(6, 1, 7, 6);
    checkReadValues({20, 20, kNullValue, 30, 30, 30}, {2, 2, 1, 2, 2, 2}, {});
  }
  recordReader_->reset();
  checkState(0, 0, 1, 0);

  // Read the last null value and read past the end.
  recordsRead = recordReader_->readRecords(3);
  ASSERT_EQ(recordsRead, 1);
  if (GetParam() == true) {
    checkState(0, 0, 1, 1);
    checkReadValues({}, {0}, {});
  } else {
    checkState(1, 1, 1, 1);
    checkReadValues({kNullValue}, {0}, {});
  }
  recordReader_->reset();
  checkState(0, 0, 0, 0);
}

// Tests reading a required repeated field. The results are the same for
// reading. Dense or spaced.
TEST_P(RecordReaderPrimitiveTypeTest, ReadRequiredRepeated) {
  NodePtr column = GroupNode::make(
      "p",
      Repetition::kRequired,
      {GroupNode::make(
          "list",
          Repetition::kRepeated,
          {PrimitiveNode::make(
              "element", Repetition::kRequired, ParquetType::kInt32)})});
  init(column);

  // Records look like: {[10], [20, 20], [30, 30, 30]}
  std::vector<std::shared_ptr<Page>> pages;
  std::vector<int32_t> values = {10, 20, 20, 30, 30, 30};
  std::vector<int16_t> defLevels = {1, 1, 1, 1, 1, 1};
  std::vector<int16_t> repLevels = {0, 0, 1, 0, 1, 1};

  std::shared_ptr<DataPageV1> page = makeDataPage<Int32Type>(
      descr_,
      values,
      static_cast<int>(defLevels.size()),
      Encoding::kPlain,
      {},
      0,
      defLevels,
      descr_->maxDefinitionLevel(),
      repLevels,
      descr_->maxRepetitionLevel());
  pages.push_back(std::move(page));
  auto pager = std::make_unique<MockPageReader>(pages);
  recordReader_->setPageReader(std::move(pager));

  // Read [10].
  int64_t recordsRead = recordReader_->readRecords(1);
  ASSERT_EQ(recordsRead, 1);
  checkState(1, 0, 6, 1);
  checkReadValues({10}, {1}, {0});
  recordReader_->reset();
  checkState(0, 0, 5, 0);

  // Read [20, 20], [30, 30, 30].
  recordsRead = recordReader_->readRecords(3);
  ASSERT_EQ(recordsRead, 2);
  checkState(5, 0, 5, 5);
  checkReadValues({20, 20, 30, 30, 30}, {1, 1, 1, 1, 1}, {0, 1, 0, 1, 1});
  recordReader_->reset();
  checkState(0, 0, 0, 0);
}

// Tests reading a nullable repeated field. Tests reading null values at.
// Differnet levels and reading an empty list.
TEST_P(RecordReaderPrimitiveTypeTest, ReadNullableRepeated) {
  NodePtr column = GroupNode::make(
      "p",
      Repetition::kOptional,
      {GroupNode::make(
          "list",
          Repetition::kRepeated,
          {PrimitiveNode::make(
              "element", Repetition::kOptional, ParquetType::kInt32)})});
  init(column);

  // Records look like: {[10], null, [20, 20], [], [30, 30, null, 30]}
  // Some explanation regarding the behavior. When reading spaced, for an empty.
  // List or for a top-level null, we do not leave a space and we do not count.
  // It towards null_count.  For a leaf-level null, we leave a space for it and.
  // We count it towards null_count. When reading dense, null_count is always
  // 0,. And we do not leave any space for values.
  std::vector<std::shared_ptr<Page>> pages;
  std::vector<int32_t> values = {10, 20, 20, 30, 30, 30};
  std::vector<int16_t> defLevels = {3, 0, 3, 3, 1, 3, 3, 2, 3};
  std::vector<int16_t> repLevels = {0, 0, 0, 1, 0, 0, 1, 1, 1};

  std::shared_ptr<DataPageV1> page = makeDataPage<Int32Type>(
      descr_,
      values,
      static_cast<int>(defLevels.size()),
      Encoding::kPlain,
      {},
      0,
      defLevels,
      descr_->maxDefinitionLevel(),
      repLevels,
      descr_->maxRepetitionLevel());
  pages.push_back(std::move(page));
  auto pager = std::make_unique<MockPageReader>(pages);
  recordReader_->setPageReader(std::move(pager));

  // Test reading 0 records.
  int64_t recordsRead = recordReader_->readRecords(0);
  ASSERT_EQ(recordsRead, 0);

  // Test the descr() accessor.
  ASSERT_EQ(recordReader_->descr()->maxDefinitionLevel(), 3);

  // Read [10], null.
  // We do not read this null for both reading dense and spaced.
  recordsRead = recordReader_->readRecords(2);
  ASSERT_EQ(recordsRead, 2);
  if (GetParam() == true) {
    checkState(1, 0, 9, 2);
    checkReadValues({10}, {3, 0}, {0, 0});
  } else {
    checkState(1, 0, 9, 2);
    checkReadValues({10}, {3, 0}, {0, 0});
  }
  recordReader_->reset();
  checkState(0, 0, 7, 0);

  // Read [20, 20], [].
  // We do not read any value for this, it will be counted towards null count.
  // When reading spaced.
  recordsRead = recordReader_->readRecords(2);
  ASSERT_EQ(recordsRead, 2);
  if (GetParam() == true) {
    checkState(2, 0, 7, 3);
    checkReadValues({20, 20}, {3, 3, 1}, {0, 1, 0});
  } else {
    checkState(2, 0, 7, 3);
    checkReadValues({20, 20}, {3, 3, 1}, {0, 1, 0});
  }
  recordReader_->reset();
  checkState(0, 0, 4, 0);

  // Test reading 0 records.
  recordsRead = recordReader_->readRecords(0);
  ASSERT_EQ(recordsRead, 0);

  // Read the last record.
  recordsRead = recordReader_->readRecords(1);
  ASSERT_EQ(recordsRead, 1);
  if (GetParam() == true) {
    checkState(3, 0, 4, 4);
    checkReadValues({30, 30, 30}, {3, 3, 2, 3}, {0, 1, 1, 1});
  } else {
    checkState(4, 1, 4, 4);
    checkReadValues({30, 30, kNullValue, 30}, {3, 3, 2, 3}, {0, 1, 1, 1});
  }
  recordReader_->reset();
  checkState(0, 0, 0, 0);
}

// Test that we can skip required top level field.
TEST_P(RecordReaderPrimitiveTypeTest, SkipRequiredTopLevel) {
  init(schema::int32("b", Repetition::kRequired));

  std::vector<std::shared_ptr<Page>> pages;
  std::vector<int32_t> values = {10, 20, 20, 30, 30, 30};
  std::shared_ptr<DataPageV1> page = makeDataPage<Int32Type>(
      descr_,
      values,
      static_cast<int>(values.size()),
      Encoding::kPlain,
      {},
      0,
      {},
      descr_->maxDefinitionLevel(),
      {},
      descr_->maxRepetitionLevel());
  pages.push_back(std::move(page));
  auto pager = std::make_unique<MockPageReader>(pages);
  recordReader_->setPageReader(std::move(pager));

  int64_t recordsSkipped = recordReader_->skipRecords(3);
  ASSERT_EQ(recordsSkipped, 3);
  checkState(0, 0, 0, 0);

  int64_t recordsRead = recordReader_->readRecords(2);
  ASSERT_EQ(recordsRead, 2);
  checkState(2, 0, 0, 0);
  checkReadValues({30, 30}, {}, {});
  recordReader_->reset();
  checkState(0, 0, 0, 0);
}

// Skip an optional field. Intentionally included some null values.
TEST_P(RecordReaderPrimitiveTypeTest, SkipOptional) {
  init(schema::int32("b", Repetition::kOptional));

  // Records look like {null, 10, 20, 30, null, 40, 50, 60}
  std::vector<std::shared_ptr<Page>> pages;
  std::vector<int32_t> values = {10, 20, 30, 40, 50, 60};
  std::vector<int16_t> defLevels = {0, 1, 1, 0, 1, 1, 1, 1};

  std::shared_ptr<DataPageV1> page = makeDataPage<Int32Type>(
      descr_,
      values,
      static_cast<int>(values.size()),
      Encoding::kPlain,
      {},
      0,
      defLevels,
      descr_->maxDefinitionLevel(),
      {},
      descr_->maxRepetitionLevel());
  pages.push_back(std::move(page));
  auto pager = std::make_unique<MockPageReader>(pages);
  recordReader_->setPageReader(std::move(pager));

  {
    // Skip {null, 10}
    // This also tests when we start with a Skip.
    int64_t recordsSkipped = recordReader_->skipRecords(2);
    ASSERT_EQ(recordsSkipped, 2);
    checkState(0, 0, 0, 0);
  }

  {
    // Read 3 records: {20, null, 30}
    int64_t recordsRead = recordReader_->readRecords(3);

    ASSERT_EQ(recordsRead, 3);
    if (GetParam() == true) {
      // We had skipped 2 of the levels above. So there is only 6 left in total.
      // To read, and we read 3 of them here.
      checkState(2, 0, 6, 3);
      checkReadValues({20, 30}, {1, 0, 1}, {});
    } else {
      checkState(3, 1, 6, 3);
      checkReadValues({20, kNullValue, 30}, {1, 0, 1}, {});
    }

    recordReader_->reset();
  }

  {
    // Skip {40, 50}.
    int64_t recordsSkipped = recordReader_->skipRecords(2);
    ASSERT_EQ(recordsSkipped, 2);
    checkState(0, 0, 1, 0);
    checkReadValues({}, {}, {});
    // Try reset after a Skip.
    recordReader_->reset();
    checkState(0, 0, 1, 0);
  }

  {
    // Read to the end of the column. Read {60}
    // This test checks that ReadAndThrowAwayValues works, since if it.
    // Does not we would read the wrong values.
    int64_t recordsRead = recordReader_->readRecords(1);

    ASSERT_EQ(recordsRead, 1);
    checkState(1, 0, 1, 1);
    checkReadValues({60}, {1}, {});
  }

  // We have exhausted all the records.
  ASSERT_EQ(recordReader_->readRecords(3), 0);
  ASSERT_EQ(recordReader_->skipRecords(3), 0);
}

// Test skipping for repeated fields.
TEST_P(RecordReaderPrimitiveTypeTest, SkipRepeated) {
  init(schema::int32("b", Repetition::kRepeated));

  // Records look like {null, [20, 20, 20], null, [30, 30], [40]}
  std::vector<std::shared_ptr<Page>> pages;
  std::vector<int32_t> values = {20, 20, 20, 30, 30, 40};
  std::vector<int16_t> defLevels = {0, 1, 1, 1, 0, 1, 1, 1};
  std::vector<int16_t> repLevels = {0, 0, 1, 1, 0, 0, 1, 0};

  std::shared_ptr<DataPageV1> page = makeDataPage<Int32Type>(
      descr_,
      values,
      static_cast<int>(values.size()),
      Encoding::kPlain,
      {},
      0,
      defLevels,
      descr_->maxDefinitionLevel(),
      repLevels,
      descr_->maxRepetitionLevel());
  pages.push_back(std::move(page));
  auto pager = std::make_unique<MockPageReader>(pages);
  recordReader_->setPageReader(std::move(pager));

  {
    // Skip 0 records.
    int64_t recordsSkipped = recordReader_->skipRecords(0);
    ASSERT_EQ(recordsSkipped, 0);
  }

  {
    // This should skip the first null record.
    int64_t recordsSkipped = recordReader_->skipRecords(1);
    ASSERT_EQ(recordsSkipped, 1);
    ASSERT_EQ(recordReader_->valuesWritten(), 0);
    ASSERT_EQ(recordReader_->nullCount(), 0);
    // For repeated fields, we need to read the levels to find the record.
    // Boundaries and skip. So some levels are read, however, the skipped.
    // level should not be there after the skip. That's why levels_position()
    // Is 0.
    checkState(0, 0, 7, 0);
    checkReadValues({}, {}, {});
  }

  {
    // Read [20, 20, 20].
    int64_t recordsRead = recordReader_->readRecords(1);
    ASSERT_EQ(recordsRead, 1);
    checkState(3, 0, 7, 3);
    checkReadValues({20, 20, 20}, {1, 1, 1}, {0, 1, 1});
  }

  {
    // Skip 0 records.
    int64_t recordsSkipped = recordReader_->skipRecords(0);
    ASSERT_EQ(recordsSkipped, 0);
  }

  {
    // Skip the null record and also skip [30, 30].
    int64_t recordsSkipped = recordReader_->skipRecords(2);
    ASSERT_EQ(recordsSkipped, 2);
    // We remove the skipped levels from the buffer.
    checkState(3, 0, 4, 3);
    checkReadValues({20, 20, 20}, {1, 1, 1}, {0, 1, 1});
  }

  {
    // Read [40].
    int64_t recordsRead = recordReader_->readRecords(1);
    ASSERT_EQ(recordsRead, 1);
    checkState(4, 0, 4, 4);
    checkReadValues({20, 20, 20, 40}, {1, 1, 1, 1}, {0, 1, 1, 0});
  }
}

// Tests that for repeated fields, we first consume what is in the buffer.
// Before reading more levels.
TEST_P(RecordReaderPrimitiveTypeTest, SkipRepeatedConsumeBufferFirst) {
  init(schema::int32("b", Repetition::kRepeated));

  std::vector<std::shared_ptr<Page>> pages;
  std::vector<int32_t> values(2048, 10);
  std::vector<int16_t> defLevels(2048, 1);
  std::vector<int16_t> repLevels(2048, 0);

  std::shared_ptr<DataPageV1> page = makeDataPage<Int32Type>(
      descr_,
      values,
      static_cast<int>(values.size()),
      Encoding::kPlain,
      {},
      0,
      defLevels,
      descr_->maxDefinitionLevel(),
      repLevels,
      descr_->maxRepetitionLevel());
  pages.push_back(std::move(page));
  auto pager = std::make_unique<MockPageReader>(pages);
  recordReader_->setPageReader(std::move(pager));
  {
    // Read 1000 records. We will read 1024 levels because that is the minimum.
    // Number of levels to read.
    int64_t recordsRead = recordReader_->readRecords(1000);
    ASSERT_EQ(recordsRead, 1000);
    checkState(1000, 0, 1024, 1000);
    std::vector<int32_t> expectedValues(1000, 10);
    std::vector<int16_t> expectedDefLevels(1000, 1);
    std::vector<int16_t> expectedRepLevels(1000, 0);
    checkReadValues(expectedValues, expectedDefLevels, expectedRepLevels);
    // Reset removes the already consumed values and levels.
    recordReader_->reset();
  }

  { // Skip 12 records. Since we already have 24 in the buffer, we should not be
    // Reading any more levels into the buffer, we will just consume 12 of it.
    int64_t recordsSkipped = recordReader_->skipRecords(12);
    ASSERT_EQ(recordsSkipped, 12);
    checkState(0, 0, 12, 0);
    // Everthing is empty because we reset the reader before this skip.
    checkReadValues({}, {}, {});
  }
}

// Test reading when one record spans multiple pages for a repeated field.
TEST_P(RecordReaderPrimitiveTypeTest, ReadPartialRecord) {
  init(schema::int32("b", Repetition::kRepeated));

  std::vector<std::shared_ptr<Page>> pages;

  // Page 1: {[10], [20, 20, 20 ... } continues to next page.
  {
    std::shared_ptr<DataPageV1> page = makeDataPage<Int32Type>(
        descr_,
        {10, 20, 20, 20},
        4,
        Encoding::kPlain,
        {},
        0,
        {1, 1, 1, 1},
        descr_->maxDefinitionLevel(),
        {0, 0, 1, 1},
        descr_->maxRepetitionLevel());
    pages.push_back(std::move(page));
  }

  // Page 2: {... 20, 20, ...} continues from previous page and to next page.
  {
    std::shared_ptr<DataPageV1> page = makeDataPage<Int32Type>(
        descr_,
        {20, 20},
        2,
        Encoding::kPlain,
        {},
        0,
        {1, 1},
        descr_->maxDefinitionLevel(),
        {1, 1},
        descr_->maxRepetitionLevel());
    pages.push_back(std::move(page));
  }

  // Page 3: { ... 20], [30]} continues from previous page.
  {
    std::shared_ptr<DataPageV1> page = makeDataPage<Int32Type>(
        descr_,
        {20, 30},
        2,
        Encoding::kPlain,
        {},
        0,
        {1, 1},
        descr_->maxDefinitionLevel(),
        {1, 0},
        descr_->maxRepetitionLevel());
    pages.push_back(std::move(page));
  }

  auto pager = std::make_unique<MockPageReader>(pages);
  recordReader_->setPageReader(std::move(pager));

  {
    // Read [10].
    int64_t recordsRead = recordReader_->readRecords(1);
    ASSERT_EQ(recordsRead, 1);
    checkState(1, 0, 4, 1);
    checkReadValues({10}, {1}, {0});
  }

  {
    // Read [20, 20, 20, 20, 20, 20] that spans multiple pages.
    int64_t recordsRead = recordReader_->readRecords(1);
    ASSERT_EQ(recordsRead, 1);
    checkState(7, 0, 8, 7);
    checkReadValues(
        {10, 20, 20, 20, 20, 20, 20},
        {1, 1, 1, 1, 1, 1, 1},
        {0, 0, 1, 1, 1, 1, 1});
  }

  {
    // Read [30].
    int64_t recordsRead = recordReader_->readRecords(1);
    ASSERT_EQ(recordsRead, 1);
    checkState(8, 0, 8, 8);
    checkReadValues(
        {10, 20, 20, 20, 20, 20, 20, 30},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {0, 0, 1, 1, 1, 1, 1, 0});
  }
}

// Test skipping for repeated fields for the case when one record spans
// multiple. Pages.
TEST_P(RecordReaderPrimitiveTypeTest, SkipPartialRecord) {
  init(schema::int32("b", Repetition::kRepeated));

  std::vector<std::shared_ptr<Page>> pages;

  // Page 1: {[10], [20, 20, 20 ... } continues to next page.
  {
    std::shared_ptr<DataPageV1> page = makeDataPage<Int32Type>(
        descr_,
        {10, 20, 20, 20},
        4,
        Encoding::kPlain,
        {},
        0,
        {1, 1, 1, 1},
        descr_->maxDefinitionLevel(),
        {0, 0, 1, 1},
        descr_->maxRepetitionLevel());
    pages.push_back(std::move(page));
  }

  // Page 2: {... 20, 20, ...} continues from previous page and to next page.
  {
    std::shared_ptr<DataPageV1> page = makeDataPage<Int32Type>(
        descr_,
        {20, 20},
        2,
        Encoding::kPlain,
        {},
        0,
        {1, 1},
        descr_->maxDefinitionLevel(),
        {1, 1},
        descr_->maxRepetitionLevel());
    pages.push_back(std::move(page));
  }

  // Page 3: { ... 20, [30]} continues from previous page.
  {
    std::shared_ptr<DataPageV1> page = makeDataPage<Int32Type>(
        descr_,
        {20, 30},
        2,
        Encoding::kPlain,
        {},
        0,
        {1, 1},
        descr_->maxDefinitionLevel(),
        {1, 0},
        descr_->maxRepetitionLevel());
    pages.push_back(std::move(page));
  }

  auto pager = std::make_unique<MockPageReader>(pages);
  recordReader_->setPageReader(std::move(pager));

  {
    // Read [10].
    int64_t recordsRead = recordReader_->readRecords(1);
    ASSERT_EQ(recordsRead, 1);
    // There are 4 levels in the first page.
    checkState(1, 0, 4, 1);
    checkReadValues({10}, {1}, {0});
  }

  {
    // Skip the record that goes across pages.
    int64_t recordsSkipped = recordReader_->skipRecords(1);
    ASSERT_EQ(recordsSkipped, 1);
    checkState(1, 0, 2, 1);
    checkReadValues({10}, {1}, {0});
  }

  {
    // Read [30].
    int64_t recordsRead = recordReader_->readRecords(1);

    ASSERT_EQ(recordsRead, 1);
    checkState(2, 0, 2, 2);
    checkReadValues({10, 30}, {1, 1}, {0, 0});
  }
}

INSTANTIATE_TEST_SUITE_P(
    RecordReaderPrimitveTypeTests,
    RecordReaderPrimitiveTypeTest,
    ::testing::Values(true, false),
    testing::PrintToStringParamName());

// Parameterized test for FLBA record reader.
class FLBARecordReaderTest : public ::testing::TestWithParam<bool> {
 public:
  bool readDenseForNullable() {
    return GetParam();
  }

  void makeRecordReader(int levelsPerPage, int numPages, int flbaTypeLength) {
    levelsPerPage_ = levelsPerPage;
    flbaTypeLength_ = flbaTypeLength;
    LevelInfo levelInfo;
    levelInfo.defLevel = 1;
    levelInfo.repLevel = 0;
    NodePtr type = schema::PrimitiveNode::make(
        "b",
        Repetition::kOptional,
        Type::kFixedLenByteArray,
        ConvertedType::kNone,
        flbaTypeLength_);
    descr_ = std::make_unique<ColumnDescriptor>(
        type, levelInfo.defLevel, levelInfo.repLevel);
    makePages<FLBAType>(
        descr_.get(),
        numPages,
        levelsPerPage,
        defLevels_,
        repLevels_,
        values_,
        buffer_,
        pages_,
        Encoding::kPlain);
    auto pager = std::make_unique<MockPageReader>(pages_);
    recordReader_ = internal::RecordReader::make(
        descr_.get(),
        levelInfo,
        ::arrow::default_memory_pool(),
        false,
        readDenseForNullable());
    recordReader_->setPageReader(std::move(pager));
  }

  // Returns expected values in row range.
  // We need this since some values are null.
  std::vector<std::string_view> expectedValues(int start, int end) {
    std::vector<std::string_view> result;
    // Find out where in the values_ vector we start from.
    size_t valuesIndex = 0;
    for (int i = 0; i < start; ++i) {
      if (defLevels_[i] != 0) {
        ++valuesIndex;
      }
    }

    for (int i = start; i < end; ++i) {
      if (defLevels_[i] == 0) {
        if (!readDenseForNullable()) {
          result.emplace_back();
        }
        continue;
      }
      result.emplace_back(
          reinterpret_cast<const char*>(values_[valuesIndex].ptr),
          flbaTypeLength_);
      ++valuesIndex;
    }
    return result;
  }

  void checkReadValues(int start, int end) {
    auto binaryReader = dynamic_cast<BinaryRecordReader*>(recordReader_.get());
    ASSERT_NE(binaryReader, nullptr);
    // Chunks are reset after this call.
    ::arrow::ArrayVector arrayVector = binaryReader->getBuilderChunks();
    ASSERT_EQ(arrayVector.size(), 1);
    auto binaryArray =
        dynamic_cast<::arrow::FixedSizeBinaryArray*>(arrayVector[0].get());

    ASSERT_NE(binaryArray, nullptr);
    ASSERT_EQ(binaryArray->length(), recordReader_->valuesWritten());
    if (readDenseForNullable()) {
      ASSERT_EQ(binaryArray->null_count(), 0);
      ASSERT_EQ(recordReader_->nullCount(), 0);
    } else {
      ASSERT_EQ(binaryArray->null_count(), recordReader_->nullCount());
    }

    std::vector<std::string_view> expected = expectedValues(start, end);
    for (size_t i = 0; i < expected.size(); ++i) {
      if (defLevels_[i + start] == 0) {
        ASSERT_EQ(!readDenseForNullable(), binaryArray->IsNull(i));
      } else {
        ASSERT_EQ(expected[i].compare(binaryArray->GetView(i)), 0);
        ASSERT_FALSE(binaryArray->IsNull(i));
      }
    }
  }

 protected:
  std::shared_ptr<internal::RecordReader> recordReader_;

 private:
  int levelsPerPage_;
  int flbaTypeLength_;
  std::vector<std::shared_ptr<Page>> pages_;
  std::vector<int16_t> defLevels_;
  std::vector<int16_t> repLevels_;
  std::vector<FixedLenByteArray> values_;
  std::vector<uint8_t> buffer_;
  std::unique_ptr<ColumnDescriptor> descr_;
};

// Similar to above, except for Byte arrays. FLBA and Byte arrays are.
// Sufficiently different to warrant a separate class for readability.
class ByteArrayRecordReaderTest : public ::testing::TestWithParam<bool> {
 public:
  bool readDenseForNullable() {
    return GetParam();
  }

  void makeRecordReader(int levelsPerPage, int numPages) {
    levelsPerPage_ = levelsPerPage;
    LevelInfo levelInfo;
    levelInfo.defLevel = 1;
    levelInfo.repLevel = 0;
    NodePtr type = schema::byteArray("b", Repetition::kOptional);
    descr_ = std::make_unique<ColumnDescriptor>(
        type, levelInfo.defLevel, levelInfo.repLevel);
    makePages<ByteArrayType>(
        descr_.get(),
        numPages,
        levelsPerPage,
        defLevels_,
        repLevels_,
        values_,
        buffer_,
        pages_,
        Encoding::kPlain);

    auto pager = std::make_unique<MockPageReader>(pages_);

    recordReader_ = internal::RecordReader::make(
        descr_.get(),
        levelInfo,
        ::arrow::default_memory_pool(),
        false,
        readDenseForNullable());
    recordReader_->setPageReader(std::move(pager));
  }

  // Returns expected values in row range.
  // We need this since some values are null.
  std::vector<std::string_view> expectedValues(int start, int end) {
    std::vector<std::string_view> result;
    // Find out where in the values_ vector we start from.
    size_t valuesIndex = 0;
    for (int i = 0; i < start; ++i) {
      if (defLevels_[i] != 0) {
        ++valuesIndex;
      }
    }

    for (int i = start; i < end; ++i) {
      if (defLevels_[i] == 0) {
        if (!readDenseForNullable()) {
          result.emplace_back();
        }
        continue;
      }
      result.emplace_back(
          reinterpret_cast<const char*>(values_[valuesIndex].ptr),
          values_[valuesIndex].len);
      ++valuesIndex;
    }
    return result;
  }

  void checkReadValues(int start, int end) {
    auto binaryReader = dynamic_cast<BinaryRecordReader*>(recordReader_.get());
    ASSERT_NE(binaryReader, nullptr);
    // Chunks are reset after this call.
    ::arrow::ArrayVector arrayVector = binaryReader->getBuilderChunks();
    ASSERT_EQ(arrayVector.size(), 1);
    ::arrow::BinaryArray* binaryArray =
        dynamic_cast<::arrow::BinaryArray*>(arrayVector[0].get());

    ASSERT_NE(binaryArray, nullptr);
    ASSERT_EQ(binaryArray->length(), recordReader_->valuesWritten());
    if (readDenseForNullable()) {
      ASSERT_EQ(binaryArray->null_count(), 0);
      ASSERT_EQ(recordReader_->nullCount(), 0);
    } else {
      ASSERT_EQ(binaryArray->null_count(), recordReader_->nullCount());
    }

    std::vector<std::string_view> expected = expectedValues(start, end);
    for (size_t i = 0; i < expected.size(); ++i) {
      if (defLevels_[i + start] == 0) {
        ASSERT_EQ(!readDenseForNullable(), binaryArray->IsNull(i));
      } else {
        ASSERT_EQ(expected[i].compare(binaryArray->GetView(i)), 0);
        ASSERT_FALSE(binaryArray->IsNull(i));
      }
    }
  }

 protected:
  std::shared_ptr<internal::RecordReader> recordReader_;

 private:
  int levelsPerPage_;
  std::vector<std::shared_ptr<Page>> pages_;
  std::vector<int16_t> defLevels_;
  std::vector<int16_t> repLevels_;
  std::vector<ByteArray> values_;
  std::vector<uint8_t> buffer_;
  std::unique_ptr<ColumnDescriptor> descr_;
};

// Tests reading and skipping a ByteArray field.
// The binary readers only differ in DeocdeDense and DecodeSpaced functions, so.
// Testing optional is sufficient in excercising those code paths.
TEST_P(ByteArrayRecordReaderTest, ReadAndSkipOptional) {
  makeRecordReader(90, 1);

  // Read one-third of the page.
  ASSERT_EQ(recordReader_->readRecords(30), 30);
  checkReadValues(0, 30);
  recordReader_->reset();

  // Skip 30 records.
  ASSERT_EQ(recordReader_->skipRecords(30), 30);

  // Read 60 more records. Only 30 will be read, since we read 30 and skipped.
  // 30, So only 30 is left.
  ASSERT_EQ(recordReader_->readRecords(60), 30);
  checkReadValues(60, 90);
  recordReader_->reset();
}

// Tests reading and skipping an optional FLBA field.
// The binary readers only differ in DeocdeDense and DecodeSpaced functions, so.
// Testing optional is sufficient in excercising those code paths.
TEST_P(FLBARecordReaderTest, ReadAndSkipOptional) {
  makeRecordReader(90, 1, 4);

  // Read one-third of the page.
  ASSERT_EQ(recordReader_->readRecords(30), 30);
  checkReadValues(0, 30);
  recordReader_->reset();

  // Skip 30 records.
  ASSERT_EQ(recordReader_->skipRecords(30), 30);

  // Read 60 more records. Only 30 will be read, since we read 30 and skipped.
  // 30, So only 30 is left.
  ASSERT_EQ(recordReader_->readRecords(60), 30);
  checkReadValues(60, 90);
  recordReader_->reset();
}

INSTANTIATE_TEST_SUITE_P(
    ByteArrayRecordReaderTests,
    ByteArrayRecordReaderTest,
    testing::Bool());

INSTANTIATE_TEST_SUITE_P(
    FLBARecordReaderTests,
    FLBARecordReaderTest,
    testing::Bool());

// Test random combination of ReadRecords and SkipRecords.
class RecordReaderStressTest
    : public ::testing::TestWithParam<Repetition::type> {};

TEST_P(RecordReaderStressTest, StressTest) {
  LevelInfo levelInfo;
  // Define these boolean variables for improving readability below.
  bool repeated = false, required = false;
  if (GetParam() == Repetition::kRequired) {
    levelInfo.defLevel = 0;
    levelInfo.repLevel = 0;
    required = true;
  } else if (GetParam() == Repetition::kOptional) {
    levelInfo.defLevel = 1;
    levelInfo.repLevel = 0;
  } else {
    levelInfo.defLevel = 1;
    levelInfo.repLevel = 1;
    repeated = true;
  }

  NodePtr type = schema::int32("b", GetParam());
  const ColumnDescriptor descr(type, levelInfo.defLevel, levelInfo.repLevel);

  auto seed1 = static_cast<uint32_t>(time(0));
  std::default_random_engine gen(seed1);
  // Generate random number of pages with random number of values per page.
  std::uniform_int_distribution<int> d(0, 2000);
  const int numPages = d(gen);
  const int levelsPerPage = d(gen);
  std::vector<int32_t> values;
  std::vector<int16_t> defLevels;
  std::vector<int16_t> repLevels;
  std::vector<uint8_t> dataBuffer;
  std::vector<std::shared_ptr<Page>> pages;
  auto seed2 = static_cast<uint32_t>(time(0));
  // Uses time(0) as seed so it would run a different test every time it is.
  // Run.
  makePages<Int32Type>(
      &descr,
      numPages,
      levelsPerPage,
      defLevels,
      repLevels,
      values,
      dataBuffer,
      pages,
      Encoding::kPlain,
      seed2);
  std::unique_ptr<PageReader> pager;
  pager.reset(new test::MockPageReader(pages));

  // Set up the RecordReader.
  std::shared_ptr<internal::RecordReader> recordReader =
      internal::RecordReader::make(&descr, levelInfo);
  recordReader->setPageReader(std::move(pager));

  // Figure out how many total records.
  int totalRecords = 0;
  if (repeated) {
    for (int16_t rep : repLevels) {
      if (rep == 0) {
        ++totalRecords;
      }
    }
  } else {
    totalRecords = static_cast<int>(defLevels.size());
  }

  // Generate a sequence of reads and skips.
  int recordsLeft = totalRecords;
  // The first element of the pair is 1 if SkipRecords and 0 if ReadRecords.
  // The second element indicates the number of records for reading or.
  // Skipping.
  std::vector<std::pair<bool, int>> sequence;
  while (recordsLeft > 0) {
    std::uniform_int_distribution<int> d(0, recordsLeft);
    // Generate a number to decide if this is a skip or read.
    bool isSkip = d(gen) < recordsLeft / 2;
    int numRecords = d(gen);

    sequence.emplace_back(isSkip, numRecords);
    recordsLeft -= numRecords;
  }

  // The levels_index and values_index are over the original vectors that have.
  // All the rep/def values for all the records. In the following loop, we will.
  // Read/skip a numebr of records and Reset the reader after each iteration.
  // This is on-par with how the record reader is used.
  size_t levelsIndex = 0;
  size_t valuesIndex = 0;
  for (const auto& [isSkip, numRecords] : sequence) {
    // Reset the reader before the next round of read/skip.
    recordReader->reset();

    // Prepare the expected result and do the SkipRecords and ReadRecords.
    std::vector<int32_t> expectedValues;
    std::vector<int16_t> expectedDefLevels;
    std::vector<int16_t> expectedRepLevels;
    bool insideRepeatedField = false;

    int readRecords = 0;
    while (readRecords < numRecords || insideRepeatedField) {
      if (!repeated || (repeated && repLevels[levelsIndex] == 0)) {
        ++readRecords;
      }

      bool hasValue = required ||
          (!required && defLevels[levelsIndex] == levelInfo.defLevel);

      // If we are not skipping, we need to update the expected values and.
      // Rep/defs. If we are skipping, we just keep going.
      if (!isSkip) {
        if (!required) {
          expectedDefLevels.push_back(defLevels[levelsIndex]);
          if (!hasValue) {
            expectedValues.push_back(-1);
          }
        }
        if (repeated) {
          expectedRepLevels.push_back(repLevels[levelsIndex]);
        }
        if (hasValue) {
          expectedValues.push_back(values[valuesIndex]);
        }
      }

      if (hasValue) {
        ++valuesIndex;
      }

      // If we are in the middle of a repeated field, we should keep going.
      // Until we consume it all.
      if (repeated && levelsIndex + 1 < repLevels.size() &&
          repLevels[levelsIndex + 1] == 1) {
        insideRepeatedField = true;
      } else {
        insideRepeatedField = false;
      }

      ++levelsIndex;
    }

    // Print out the seeds with each failing ASSERT to easily reproduce the bug.
    std::string seeds =
        "seeds: " + std::to_string(seed1) + " " + std::to_string(seed2);

    // Perform the actual read/skip.
    if (isSkip) {
      int64_t skippedRecords = recordReader->skipRecords(numRecords);
      ASSERT_EQ(skippedRecords, numRecords) << seeds;
    } else {
      int64_t readRecords = recordReader->readRecords(numRecords);
      ASSERT_EQ(readRecords, numRecords) << seeds;
    }

    const auto readValues =
        reinterpret_cast<const int32_t*>(recordReader->values());
    if (required) {
      ASSERT_EQ(recordReader->nullCount(), 0) << seeds;
    }
    std::vector<int32_t> readVals(
        readValues, readValues + recordReader->valuesWritten());
    for (size_t i = 0; i < expectedValues.size(); ++i) {
      if (expectedValues[i] != -1) {
        ASSERT_EQ(readVals[i], expectedValues[i]) << seeds;
      }
    }

    if (!required) {
      std::vector<int16_t> readDefLevels(
          recordReader->defLevels(),
          recordReader->defLevels() + recordReader->levelsPosition());
      ASSERT_TRUE(vectorEqual(readDefLevels, expectedDefLevels)) << seeds;
    }

    if (repeated) {
      std::vector<int16_t> readRepLevels(
          recordReader->repLevels(),
          recordReader->repLevels() + recordReader->levelsPosition());
      ASSERT_TRUE(vectorEqual(readRepLevels, expectedRepLevels)) << seeds;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    repetitionType,
    RecordReaderStressTest,
    ::testing::Values(
        Repetition::kRequired,
        Repetition::kOptional,
        Repetition::kRepeated));

} // namespace test
} // namespace facebook::velox::parquet::arrow
