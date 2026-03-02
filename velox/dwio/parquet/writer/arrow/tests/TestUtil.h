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

#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "arrow/io/memory.h"
#include "arrow/testing/util.h"

#include "velox/dwio/parquet/writer/arrow/ColumnPage.h"
#include "velox/dwio/parquet/writer/arrow/ColumnWriter.h"
#include "velox/dwio/parquet/writer/arrow/Encoding.h"
#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/tests/ColumnReader.h"
#include "velox/exec/tests/utils/TempFilePath.h"

// https://github.com/google/googletest/pull/2904 might not be available
// In our version of gtest/gmock.
#define EXPECT_THROW_THAT(callable, exType, property)   \
  EXPECT_THROW(                                         \
      try { (callable)(); } catch (const exType& err) { \
        EXPECT_THAT(err, (property));                   \
        throw;                                          \
      },                                                \
      exType)

namespace facebook::velox::parquet::arrow {

static constexpr int FLBA_LENGTH = 12;

inline bool operator==(const FixedLenByteArray& a, const FixedLenByteArray& b) {
  return 0 == memcmp(a.ptr, b.ptr, FLBA_LENGTH);
}

namespace test {

typedef ::testing::Types<
    BooleanType,
    Int32Type,
    Int64Type,
    Int96Type,
    FloatType,
    DoubleType,
    ByteArrayType,
    FLBAType>
    ParquetTypes;

class ParquetTestException : public ParquetException {
  using ParquetException::ParquetException;
};

const char* getDataDir();
std::string getBadDataDir();

std::string getDataFile(const std::string& filename, bool isGood = true);

// Write an Arrow buffer to a temporary file path
void writeToFile(
    std::shared_ptr<exec::test::TempFilePath> filePath,
    std::shared_ptr<::arrow::Buffer> buffer);

template <typename T>
static inline void assertVectorEqual(
    const std::vector<T>& left,
    const std::vector<T>& right) {
  ASSERT_EQ(left.size(), right.size());

  for (size_t i = 0; i < left.size(); ++i) {
    ASSERT_EQ(left[i], right[i]) << i;
  }
}

template <typename T>
static inline bool vectorEqual(
    const std::vector<T>& left,
    const std::vector<T>& right) {
  if (left.size() != right.size()) {
    return false;
  }

  for (size_t i = 0; i < left.size(); ++i) {
    if (left[i] != right[i]) {
      std::cerr << "index " << i << " left was " << left[i] << " right was "
                << right[i] << std::endl;
      return false;
    }
  }

  return true;
}

template <typename T>
static std::vector<T> slice(const std::vector<T>& values, int start, int end) {
  if (end < start) {
    return std::vector<T>(0);
  }

  std::vector<T> out(end - start);
  for (int i = start; i < end; ++i) {
    out[i - start] = values[i];
  }
  return out;
}

void randomBytes(int n, uint32_t seed, std::vector<uint8_t>* out);
void randomBools(int n, double p, uint32_t seed, bool* out);

template <typename T>
inline void
randomNumbers(int n, uint32_t seed, T minValue, T maxValue, T* out) {
  std::default_random_engine gen(seed);
  std::uniform_int_distribution<T> d(minValue, maxValue);
  for (int i = 0; i < n; ++i) {
    out[i] = d(gen);
  }
}

template <>
inline void randomNumbers(
    int n,
    uint32_t seed,
    float minValue,
    float maxValue,
    float* out) {
  std::default_random_engine gen(seed);
  std::uniform_real_distribution<float> d(minValue, maxValue);
  for (int i = 0; i < n; ++i) {
    out[i] = d(gen);
  }
}

template <>
inline void randomNumbers(
    int n,
    uint32_t seed,
    double minValue,
    double maxValue,
    double* out) {
  std::default_random_engine gen(seed);
  std::uniform_real_distribution<double> d(minValue, maxValue);
  for (int i = 0; i < n; ++i) {
    out[i] = d(gen);
  }
}

void randomInt96Numbers(
    int n,
    uint32_t seed,
    int32_t minValue,
    int32_t maxValue,
    Int96* out);

void randomFixedByteArray(
    int n,
    uint32_t seed,
    uint8_t* buf,
    int len,
    FLBA* out);

void randomByteArray(
    int n,
    uint32_t seed,
    uint8_t* buf,
    ByteArray* out,
    int minSize,
    int maxSize);

void randomByteArray(
    int n,
    uint32_t seed,
    uint8_t* buf,
    ByteArray* out,
    int maxSize);

template <typename Type, typename Sequence>
std::shared_ptr<Buffer> encodeValues(
    Encoding::type encoding,
    bool useDictionary,
    const Sequence& values,
    int length,
    const ColumnDescriptor* descr) {
  auto encoder = makeTypedEncoder<Type>(encoding, useDictionary, descr);
  encoder->put(values, length);
  return encoder->flushValues();
}

template <typename T>
static void initValues(
    int numValues,
    uint32_t seed,
    std::vector<T>& values,
    std::vector<uint8_t>& buffer) {
  randomNumbers(
      numValues,
      seed,
      std::numeric_limits<T>::min(),
      std::numeric_limits<T>::max(),
      values.data());
}

template <typename T>
static void initValues(
    int numValues,
    std::vector<T>& values,
    std::vector<uint8_t>& buffer) {
  initValues(numValues, 0, values, buffer);
}

template <typename T>
static void initDictValues(
    int numValues,
    int numDicts,
    std::vector<T>& values,
    std::vector<uint8_t>& buffer) {
  int repeatFactor = numValues / numDicts;
  initValues<T>(numDicts, values, buffer);
  // Add some repeated values.
  for (int j = 1; j < repeatFactor; ++j) {
    for (int i = 0; i < numDicts; ++i) {
      std::memcpy(&values[numDicts * j + i], &values[i], sizeof(T));
    }
  }
  // Computed only dict_per_page * repeat_factor - 1 values < num_values.
  // Compute remaining.
  for (int i = numDicts * repeatFactor; i < numValues; ++i) {
    std::memcpy(&values[i], &values[i - numDicts * repeatFactor], sizeof(T));
  }
}

template <>
inline void initDictValues<bool>(
    int numValues,
    int numDicts,
    std::vector<bool>& values,
    std::vector<uint8_t>& buffer) {
  // No op for bool.
}

class MockPageReader : public PageReader {
 public:
  explicit MockPageReader(const std::vector<std::shared_ptr<Page>>& pages)
      : pages_(pages), pageIndex_(0) {}

  std::shared_ptr<Page> nextPage() override {
    if (pageIndex_ == static_cast<int>(pages_.size())) {
      // EOS to consumer.
      return std::shared_ptr<Page>(nullptr);
    }
    return pages_[pageIndex_++];
  }

  // No-op.
  void setMaxPageHeaderSize(uint32_t size) override {}

 private:
  std::vector<std::shared_ptr<Page>> pages_;
  int pageIndex_;
};

// TODO(wesm): this is only used for testing for now. Refactor to form part of.
// Primary file write path.
template <typename Type>
class DataPageBuilder {
 public:
  using CType = typename Type::CType;

  // This class writes data and metadata to the passed inputs.
  explicit DataPageBuilder(ArrowOutputStream* sink)
      : sink_(sink),
        numValues_(0),
        encoding_(Encoding::kPlain),
        definitionLevelEncoding_(Encoding::kRle),
        repetitionLevelEncoding_(Encoding::kRle),
        haveDefLevels_(false),
        haveRepLevels_(false),
        haveValues_(false) {}

  void appendDefLevels(
      const std::vector<int16_t>& levels,
      int16_t maxLevel,
      Encoding::type encoding = Encoding::kRle) {
    appendLevels(levels, maxLevel, encoding);

    numValues_ = std::max(static_cast<int32_t>(levels.size()), numValues_);
    definitionLevelEncoding_ = encoding;
    haveDefLevels_ = true;
  }

  void appendRepLevels(
      const std::vector<int16_t>& levels,
      int16_t maxLevel,
      Encoding::type encoding = Encoding::kRle) {
    appendLevels(levels, maxLevel, encoding);

    numValues_ = std::max(static_cast<int32_t>(levels.size()), numValues_);
    repetitionLevelEncoding_ = encoding;
    haveRepLevels_ = true;
  }

  void appendValues(
      const ColumnDescriptor* d,
      const std::vector<CType>& values,
      Encoding::type encoding = Encoding::kPlain) {
    std::shared_ptr<Buffer> valuesSink = encodeValues<Type>(
        encoding, false, values.data(), static_cast<int>(values.size()), d);
    PARQUET_THROW_NOT_OK(sink_->Write(valuesSink->data(), valuesSink->size()));

    numValues_ = std::max(static_cast<int32_t>(values.size()), numValues_);
    encoding_ = encoding;
    haveValues_ = true;
  }

  int32_t numValues() const {
    return numValues_;
  }

  Encoding::type encoding() const {
    return encoding_;
  }

  Encoding::type repLevelEncoding() const {
    return repetitionLevelEncoding_;
  }

  Encoding::type defLevelEncoding() const {
    return definitionLevelEncoding_;
  }

 private:
  ArrowOutputStream* sink_;

  int32_t numValues_;
  Encoding::type encoding_;
  Encoding::type definitionLevelEncoding_;
  Encoding::type repetitionLevelEncoding_;

  bool haveDefLevels_;
  bool haveRepLevels_;
  bool haveValues_;

  // Used internally for both repetition and definition levels.
  void appendLevels(
      const std::vector<int16_t>& levels,
      int16_t maxLevel,
      Encoding::type encoding) {
    if (encoding != Encoding::kRle) {
      ParquetException::NYI("only rle encoding currently implemented");
    }

    std::vector<uint8_t> encodeBuffer(
        LevelEncoder::maxBufferSize(
            Encoding::kRle, maxLevel, static_cast<int>(levels.size())));

    // We encode into separate memory from the output stream because the.
    // RLE-encoded bytes have to be preceded in the stream by their absolute.
    // Size.
    LevelEncoder encoder;
    encoder.init(
        encoding,
        maxLevel,
        static_cast<int>(levels.size()),
        encodeBuffer.data(),
        static_cast<int>(encodeBuffer.size()));

    encoder.encode(static_cast<int>(levels.size()), levels.data());

    int32_t rleBytes = encoder.len();
    PARQUET_THROW_NOT_OK(sink_->Write(
        reinterpret_cast<const uint8_t*>(&rleBytes), sizeof(int32_t)));
    PARQUET_THROW_NOT_OK(sink_->Write(encodeBuffer.data(), rleBytes));
  }
};

template <>
inline void DataPageBuilder<BooleanType>::appendValues(
    const ColumnDescriptor* d,
    const std::vector<bool>& values,
    Encoding::type encoding) {
  if (encoding != Encoding::kPlain) {
    ParquetException::NYI("only plain encoding currently implemented");
  }

  auto encoder = makeTypedEncoder<BooleanType>(Encoding::kPlain, false, d);
  dynamic_cast<BooleanEncoder*>(encoder.get())
      ->put(values, static_cast<int>(values.size()));
  std::shared_ptr<Buffer> buffer = encoder->flushValues();
  PARQUET_THROW_NOT_OK(sink_->Write(buffer->data(), buffer->size()));

  numValues_ = std::max(static_cast<int32_t>(values.size()), numValues_);
  encoding_ = encoding;
  haveValues_ = true;
}

template <typename Type>
static std::shared_ptr<DataPageV1> makeDataPage(
    const ColumnDescriptor* d,
    const std::vector<typename Type::CType>& values,
    int numVals,
    Encoding::type encoding,
    const uint8_t* indices,
    int indicesSize,
    const std::vector<int16_t>& defLevels,
    int16_t maxDefLevel,
    const std::vector<int16_t>& repLevels,
    int16_t maxRepLevel) {
  int numValues = 0;

  auto pageStream = createOutputStream();
  test::DataPageBuilder<Type> pageBuilder(pageStream.get());

  if (!repLevels.empty()) {
    pageBuilder.appendRepLevels(repLevels, maxRepLevel);
  }
  if (!defLevels.empty()) {
    pageBuilder.appendDefLevels(defLevels, maxDefLevel);
  }

  if (encoding == Encoding::kPlain) {
    pageBuilder.appendValues(d, values, encoding);
    numValues = std::max(pageBuilder.numValues(), numVals);
  } else { // DICTIONARY PAGES
    PARQUET_THROW_NOT_OK(pageStream->Write(indices, indicesSize));
    numValues = std::max(pageBuilder.numValues(), numVals);
  }

  PARQUET_ASSIGN_OR_THROW(auto buffer, pageStream->Finish());

  return std::make_shared<DataPageV1>(
      buffer,
      numValues,
      encoding,
      pageBuilder.defLevelEncoding(),
      pageBuilder.repLevelEncoding(),
      buffer->size());
}

template <typename TYPE>
class DictionaryPageBuilder {
 public:
  typedef typename TYPE::CType TC;
  static constexpr int TN = TYPE::typeNum;
  using SpecializedEncoder = typename EncodingTraits<TYPE>::Encoder;

  // This class writes data and metadata to the passed inputs.
  explicit DictionaryPageBuilder(const ColumnDescriptor* d)
      : numDictValues_(0), haveValues_(false) {
    auto encoder = makeTypedEncoder<TYPE>(Encoding::kPlain, true, d);
    dictTraits_ = dynamic_cast<DictEncoder<TYPE>*>(encoder.get());
    encoder_.reset(dynamic_cast<SpecializedEncoder*>(encoder.release()));
  }

  ~DictionaryPageBuilder() {}

  std::shared_ptr<Buffer> appendValues(const std::vector<TC>& values) {
    int numValues = static_cast<int>(values.size());
    // Dictionary encoding.
    encoder_->put(values.data(), numValues);
    numDictValues_ = dictTraits_->numEntries();
    haveValues_ = true;
    return encoder_->flushValues();
  }

  std::shared_ptr<Buffer> writeDict() {
    std::shared_ptr<Buffer> dictBuffer = allocateBuffer(
        ::arrow::default_memory_pool(), dictTraits_->dictEncodedSize());
    dictTraits_->writeDict(dictBuffer->mutable_data());
    return dictBuffer;
  }

  int32_t numValues() const {
    return numDictValues_;
  }

 private:
  DictEncoder<TYPE>* dictTraits_;
  std::unique_ptr<SpecializedEncoder> encoder_;
  int32_t numDictValues_;
  bool haveValues_;
};

template <>
inline DictionaryPageBuilder<BooleanType>::DictionaryPageBuilder(
    const ColumnDescriptor* d) {
  ParquetException::NYI(
      "only plain encoding currently implemented for boolean");
}

template <>
inline std::shared_ptr<Buffer> DictionaryPageBuilder<BooleanType>::writeDict() {
  ParquetException::NYI(
      "only plain encoding currently implemented for boolean");
  return nullptr;
}

template <>
inline std::shared_ptr<Buffer> DictionaryPageBuilder<BooleanType>::appendValues(
    const std::vector<TC>& values) {
  ParquetException::NYI(
      "only plain encoding currently implemented for boolean");
  return nullptr;
}

template <typename Type>
inline static std::shared_ptr<DictionaryPage> makeDictPage(
    const ColumnDescriptor* d,
    const std::vector<typename Type::CType>& values,
    const std::vector<int>& valuesPerPage,
    Encoding::type encoding,
    std::vector<std::shared_ptr<Buffer>>& rleIndices) {
  test::DictionaryPageBuilder<Type> pageBuilder(d);
  int numPages = static_cast<int>(valuesPerPage.size());
  int valueStart = 0;

  for (int i = 0; i < numPages; i++) {
    rleIndices.push_back(pageBuilder.appendValues(
        slice(values, valueStart, valueStart + valuesPerPage[i])));
    valueStart += valuesPerPage[i];
  }

  auto buffer = pageBuilder.writeDict();

  return std::make_shared<DictionaryPage>(
      buffer, pageBuilder.numValues(), Encoding::kPlain);
}

// Given def/rep levels and values create multiple dict pages.
template <typename Type>
inline static void paginateDict(
    const ColumnDescriptor* d,
    const std::vector<typename Type::CType>& values,
    const std::vector<int16_t>& defLevels,
    int16_t maxDefLevel,
    const std::vector<int16_t>& repLevels,
    int16_t maxRepLevel,
    int numLevelsPerPage,
    const std::vector<int>& valuesPerPage,
    std::vector<std::shared_ptr<Page>>& pages,
    Encoding::type encoding = Encoding::kRleDictionary) {
  int numPages = static_cast<int>(valuesPerPage.size());
  std::vector<std::shared_ptr<Buffer>> rleIndices;
  std::shared_ptr<DictionaryPage> dictPage =
      makeDictPage<Type>(d, values, valuesPerPage, encoding, rleIndices);
  pages.push_back(dictPage);
  int defLevelStart = 0;
  int defLevelEnd = 0;
  int repLevelStart = 0;
  int repLevelEnd = 0;
  for (int i = 0; i < numPages; i++) {
    if (maxDefLevel > 0) {
      defLevelStart = i * numLevelsPerPage;
      defLevelEnd = (i + 1) * numLevelsPerPage;
    }
    if (maxRepLevel > 0) {
      repLevelStart = i * numLevelsPerPage;
      repLevelEnd = (i + 1) * numLevelsPerPage;
    }
    std::shared_ptr<DataPageV1> dataPage = makeDataPage<Int32Type>(
        d,
        {},
        valuesPerPage[i],
        encoding,
        rleIndices[i]->data(),
        static_cast<int>(rleIndices[i]->size()),
        slice(defLevels, defLevelStart, defLevelEnd),
        maxDefLevel,
        slice(repLevels, repLevelStart, repLevelEnd),
        maxRepLevel);
    pages.push_back(dataPage);
  }
}

// Given def/rep levels and values create multiple plain pages.
template <typename Type>
static inline void paginatePlain(
    const ColumnDescriptor* d,
    const std::vector<typename Type::CType>& values,
    const std::vector<int16_t>& defLevels,
    int16_t maxDefLevel,
    const std::vector<int16_t>& repLevels,
    int16_t maxRepLevel,
    int numLevelsPerPage,
    const std::vector<int>& valuesPerPage,
    std::vector<std::shared_ptr<Page>>& pages,
    Encoding::type encoding = Encoding::kPlain) {
  int numPages = static_cast<int>(valuesPerPage.size());
  int defLevelStart = 0;
  int defLevelEnd = 0;
  int repLevelStart = 0;
  int repLevelEnd = 0;
  int valueStart = 0;
  for (int i = 0; i < numPages; i++) {
    if (maxDefLevel > 0) {
      defLevelStart = i * numLevelsPerPage;
      defLevelEnd = (i + 1) * numLevelsPerPage;
    }
    if (maxRepLevel > 0) {
      repLevelStart = i * numLevelsPerPage;
      repLevelEnd = (i + 1) * numLevelsPerPage;
    }
    std::shared_ptr<DataPage> page = makeDataPage<Type>(
        d,
        slice(values, valueStart, valueStart + valuesPerPage[i]),
        valuesPerPage[i],
        encoding,
        nullptr,
        0,
        slice(defLevels, defLevelStart, defLevelEnd),
        maxDefLevel,
        slice(repLevels, repLevelStart, repLevelEnd),
        maxRepLevel);
    pages.push_back(page);
    valueStart += valuesPerPage[i];
  }
}

// Generates pages from randomly generated data.
template <typename Type>
static inline int makePages(
    const ColumnDescriptor* d,
    int numPages,
    int levelsPerPage,
    std::vector<int16_t>& defLevels,
    std::vector<int16_t>& repLevels,
    std::vector<typename Type::CType>& values,
    std::vector<uint8_t>& buffer,
    std::vector<std::shared_ptr<Page>>& pages,
    Encoding::type encoding = Encoding::kPlain,
    uint32_t seed = 0) {
  int numLevels = levelsPerPage * numPages;
  int numValues = 0;
  int16_t zero = 0;
  int16_t maxDefLevel = d->maxDefinitionLevel();
  int16_t maxRepLevel = d->maxRepetitionLevel();
  std::vector<int> valuesPerPage(numPages, levelsPerPage);
  // Create definition levels.
  if (maxDefLevel > 0 && numLevels != 0) {
    defLevels.resize(numLevels);
    randomNumbers(numLevels, seed, zero, maxDefLevel, defLevels.data());
    for (int p = 0; p < numPages; p++) {
      int numValuesPerPage = 0;
      for (int i = 0; i < levelsPerPage; i++) {
        if (defLevels[i + p * levelsPerPage] == maxDefLevel) {
          numValuesPerPage++;
          numValues++;
        }
      }
      valuesPerPage[p] = numValuesPerPage;
    }
  } else {
    numValues = numLevels;
  }
  // Create repitition levels.
  if (maxRepLevel > 0 && numLevels != 0) {
    repLevels.resize(numLevels);
    // Using a different seed so that def_levels and rep_levels are different.
    randomNumbers(numLevels, seed + 789, zero, maxRepLevel, repLevels.data());
    // The generated levels are random. Force the very first page to start with.
    // A new record.
    repLevels[0] = 0;
    // For a null value, rep_levels and def_levels are both 0.
    // If we have a repeated value right after this, it needs to start with.
    // Rep_level = 0 to indicate a new record.
    for (int i = 0; i < numLevels - 1; ++i) {
      if (repLevels[i] == 0 && defLevels[i] == 0) {
        repLevels[i + 1] = 0;
      }
    }
  }
  // Create values.
  values.resize(numValues);
  if (encoding == Encoding::kPlain) {
    initValues<typename Type::CType>(numValues, values, buffer);
    paginatePlain<Type>(
        d,
        values,
        defLevels,
        maxDefLevel,
        repLevels,
        maxRepLevel,
        levelsPerPage,
        valuesPerPage,
        pages);
  } else if (
      encoding == Encoding::kRleDictionary ||
      encoding == Encoding::kPlainDictionary) {
    // Calls InitValues and repeats the data.
    initDictValues<typename Type::CType>(
        numValues, levelsPerPage, values, buffer);
    paginateDict<Type>(
        d,
        values,
        defLevels,
        maxDefLevel,
        repLevels,
        maxRepLevel,
        levelsPerPage,
        valuesPerPage,
        pages);
  }

  return numValues;
}

// ----------------------------------------------------------------------.
// Test data generation.

template <>
void inline initValues<bool>(
    int numValues,
    uint32_t seed,
    std::vector<bool>& values,
    std::vector<uint8_t>& buffer) {
  values = {};
  if (seed == 0) {
    seed = static_cast<uint32_t>(::arrow::random_seed());
  }
  ::arrow::random_is_valid(numValues, 0.5, &values, static_cast<int>(seed));
}

template <>
inline void initValues<ByteArray>(
    int numValues,
    uint32_t seed,
    std::vector<ByteArray>& values,
    std::vector<uint8_t>& buffer) {
  int maxByteArrayLen = 12;
  int numBytes = static_cast<int>(maxByteArrayLen + sizeof(uint32_t));
  size_t nbytes = numValues * numBytes;
  buffer.resize(nbytes);
  randomByteArray(
      numValues, seed, buffer.data(), values.data(), maxByteArrayLen);
}

inline void initWideByteArrayValues(
    int numValues,
    std::vector<ByteArray>& values,
    std::vector<uint8_t>& buffer,
    int minLen,
    int maxLen) {
  int numBytes = static_cast<int>(maxLen + sizeof(uint32_t));
  size_t nbytes = numValues * numBytes;
  buffer.resize(nbytes);
  randomByteArray(numValues, 0, buffer.data(), values.data(), minLen, maxLen);
}

template <>
inline void initValues<FLBA>(
    int numValues,
    uint32_t seed,
    std::vector<FLBA>& values,
    std::vector<uint8_t>& buffer) {
  size_t nbytes = numValues * FLBA_LENGTH;
  buffer.resize(nbytes);
  randomFixedByteArray(
      numValues, seed, buffer.data(), FLBA_LENGTH, values.data());
}

template <>
inline void initValues<Int96>(
    int numValues,
    uint32_t seed,
    std::vector<Int96>& values,
    std::vector<uint8_t>& buffer) {
  randomInt96Numbers(
      numValues,
      seed,
      std::numeric_limits<int32_t>::min(),
      std::numeric_limits<int32_t>::max(),
      values.data());
}

inline std::string testColumnName(int i) {
  std::stringstream colName;
  colName << "column_" << i;
  return colName.str();
}

// This class lives here because of its dependency on the InitValues.
// Specializations.
template <typename TestType>
class PrimitiveTypedTest : public ::testing::Test {
 public:
  using CType = typename TestType::CType;

  void setUpSchema(Repetition::type repetition, int numColumns = 1) {
    std::vector<schema::NodePtr> fields;

    for (int i = 0; i < numColumns; ++i) {
      std::string name = testColumnName(i);
      fields.push_back(
          schema::PrimitiveNode::make(
              name,
              repetition,
              TestType::typeNum,
              ConvertedType::kNone,
              FLBA_LENGTH));
    }
    node_ = schema::GroupNode::make("schema", Repetition::kRequired, fields);
    schema_.init(node_);
  }

  void generateData(int64_t numValues, uint32_t seed = 0);
  void setupValuesOut(int64_t numValues);
  void syncValuesOut();

 protected:
  schema::NodePtr node_;
  SchemaDescriptor schema_;

  // Input buffers.
  std::vector<CType> values_;

  std::vector<int16_t> defLevels_;

  std::vector<uint8_t> buffer_;
  // Pointer to the values, needed as we cannot use std::vector<bool>::data()
  CType* valuesPtr_;
  std::vector<uint8_t> boolBuffer_;

  // Output buffers.
  std::vector<CType> valuesOut_;
  std::vector<uint8_t> boolBufferOut_;
  CType* valuesOutPtr_;
};

template <typename TestType>
inline void PrimitiveTypedTest<TestType>::syncValuesOut() {}

template <>
inline void PrimitiveTypedTest<BooleanType>::syncValuesOut() {
  std::vector<uint8_t>::const_iterator sourceIterator = boolBufferOut_.begin();
  std::vector<CType>::iterator destinationIterator = valuesOut_.begin();
  while (sourceIterator != boolBufferOut_.end()) {
    *destinationIterator++ = *sourceIterator++ != 0;
  }
}

template <typename TestType>
inline void PrimitiveTypedTest<TestType>::setupValuesOut(int64_t numValues) {
  valuesOut_.clear();
  valuesOut_.resize(numValues);
  valuesOutPtr_ = valuesOut_.data();
}

template <>
inline void PrimitiveTypedTest<BooleanType>::setupValuesOut(int64_t numValues) {
  valuesOut_.clear();
  valuesOut_.resize(numValues);

  boolBufferOut_.clear();
  boolBufferOut_.resize(numValues);
  // Write once to all values so we can copy it without getting Valgrind errors.
  // About uninitialised values.
  std::fill(boolBufferOut_.begin(), boolBufferOut_.end(), true);
  valuesOutPtr_ = reinterpret_cast<bool*>(boolBufferOut_.data());
}

template <typename TestType>
inline void PrimitiveTypedTest<TestType>::generateData(
    int64_t numValues,
    uint32_t seed) {
  defLevels_.resize(numValues);
  values_.resize(numValues);

  initValues<CType>(static_cast<int>(numValues), seed, values_, buffer_);
  valuesPtr_ = values_.data();

  std::fill(defLevels_.begin(), defLevels_.end(), 1);
}

template <>
inline void PrimitiveTypedTest<BooleanType>::generateData(
    int64_t numValues,
    uint32_t seed) {
  defLevels_.resize(numValues);
  values_.resize(numValues);

  initValues<CType>(static_cast<int>(numValues), seed, values_, buffer_);
  boolBuffer_.resize(numValues);
  std::copy(values_.begin(), values_.end(), boolBuffer_.begin());
  valuesPtr_ = reinterpret_cast<bool*>(boolBuffer_.data());

  std::fill(defLevels_.begin(), defLevels_.end(), 1);
}

// ----------------------------------------------------------------------.
// Test data generation.

template <typename T>
inline void generateData(int numValues, T* out, std::vector<uint8_t>* heap) {
  // Seed the prng so failure is deterministic.
  randomNumbers(
      numValues,
      0,
      std::numeric_limits<T>::min(),
      std::numeric_limits<T>::max(),
      out);
}

template <typename T>
inline void generateBoundData(
    int numValues,
    T* out,
    T min,
    T max,
    std::vector<uint8_t>* heap) {
  // Seed the prng so failure is deterministic.
  randomNumbers(numValues, 0, min, max, out);
}

template <>
inline void
generateData<bool>(int numValues, bool* out, std::vector<uint8_t>* heap) {
  // Seed the prng so failure is deterministic.
  randomBools(numValues, 0.5, 0, out);
}

template <>
inline void
generateData<Int96>(int numValues, Int96* out, std::vector<uint8_t>* heap) {
  // Seed the prng so failure is deterministic.
  randomInt96Numbers(
      numValues,
      0,
      std::numeric_limits<int32_t>::min(),
      std::numeric_limits<int32_t>::max(),
      out);
}

template <>
inline void generateData<ByteArray>(
    int numValues,
    ByteArray* out,
    std::vector<uint8_t>* heap) {
  // Seed the prng so failure is deterministic.
  int maxByteArrayLen = 12;
  heap->resize(numValues * maxByteArrayLen);
  randomByteArray(numValues, 0, heap->data(), out, 2, maxByteArrayLen);
}

static constexpr int kGenerateDataFLBALength = 8;

template <>
inline void
generateData<FLBA>(int numValues, FLBA* out, std::vector<uint8_t>* heap) {
  // Seed the prng so failure is deterministic.
  heap->resize(numValues * kGenerateDataFLBALength);
  randomFixedByteArray(
      numValues, 0, heap->data(), kGenerateDataFLBALength, out);
}

} // namespace test
} // namespace facebook::velox::parquet::arrow
