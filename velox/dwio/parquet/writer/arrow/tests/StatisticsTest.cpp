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

// Adapted from Apache Arrow.

#include <gtest/gtest.h>

#include "arrow/testing/builder.h"

#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/dwio/parquet/writer/arrow/FileWriter.h"
#include "velox/dwio/parquet/writer/arrow/tests/TestUtil.h"
#include "velox/exec/tests/utils/TempFilePath.h"

using arrow::default_memory_pool;
using arrow::MemoryPool;
using arrow::util::SafeCopy;

namespace bit_util = arrow::bit_util;

namespace facebook::velox::parquet::arrow {

using schema::GroupNode;
using schema::NodePtr;
using schema::PrimitiveNode;

namespace test {
namespace {
void writeToFile(
    std::shared_ptr<exec::test::TempFilePath> filePath,
    std::shared_ptr<::arrow::Buffer> buffer) {
  auto localWriteFile =
      std::make_unique<LocalWriteFile>(filePath->getPath(), false, false);
  auto bufferReader = std::make_shared<::arrow::io::BufferReader>(buffer);
  auto bufferToString = bufferReader->buffer()->ToString();
  localWriteFile->append(bufferToString);
  localWriteFile->close();
}
} // namespace

// ----------------------------------------------------------------------.
// Test Comparators.

static ByteArray byteArrayFromString(const std::string& s) {
  auto ptr = reinterpret_cast<const uint8_t*>(s.data());
  return ByteArray(static_cast<uint32_t>(s.size()), ptr);
}

static FLBA fLBAFromString(const std::string& s) {
  auto ptr = reinterpret_cast<const uint8_t*>(s.data());
  return FLBA(ptr);
}

TEST(Comparison, SignedByteArray) {
  // Signed byte array comparison is only used for Decimal comparison. When
  // decimals are encoded as byte arrays they use twos complement big-endian
  // encoded values. Comparisons of byte arrays of unequal types need to handle
  // sign extension.
  auto Comparator =
      makeComparator<ByteArrayType>(Type::kByteArray, SortOrder::kSigned);
  struct Case {
    std::vector<uint8_t> bytes;
    int order;
    ByteArray toByteArray() const {
      return ByteArray(static_cast<int>(bytes.size()), bytes.data());
    }
  };

  // Test a mix of big-endian comparison values that are both equal and
  // unequal after sign extension.
  std::vector<Case> cases = {
      {{0x80, 0x80, 0, 0}, 0},
      {{/*0xFF,*/ 0x80, 0, 0}, 1},
      {{0xFF, 0x80, 0, 0}, 1},
      {{/*0xFF,*/ 0xFF, 0x01, 0}, 2},
      {{/*0xFF,  0xFF,*/ 0x80, 0}, 3},
      {{/*0xFF,*/ 0xFF, 0x80, 0}, 3},
      {{0xFF, 0xFF, 0x80, 0}, 3},
      {{/*0xFF,0xFF,0xFF,*/ 0x80}, 4},
      {{/*0xFF, 0xFF, 0xFF,*/ 0xFF}, 5},
      {{/*0, 0,*/ 0x01, 0x01}, 6},
      {{/*0,*/ 0, 0x01, 0x01}, 6},
      {{0, 0, 0x01, 0x01}, 6},
      {{/*0,*/ 0x01, 0x01, 0}, 7},
      {{0x01, 0x01, 0, 0}, 8}};

  for (size_t x = 0; x < cases.size(); x++) {
    const auto& case1 = cases[x];
    // Empty array is always the smallest values.
    EXPECT_TRUE(Comparator->compare(ByteArray(), case1.toByteArray())) << x;
    EXPECT_FALSE(Comparator->compare(case1.toByteArray(), ByteArray())) << x;
    // Equals is always false.
    EXPECT_FALSE(Comparator->compare(case1.toByteArray(), case1.toByteArray()))
        << x;

    for (size_t y = 0; y < cases.size(); y++) {
      const auto& case2 = cases[y];
      if (case1.order < case2.order) {
        EXPECT_TRUE(
            Comparator->compare(case1.toByteArray(), case2.toByteArray()))
            << x << " (order: " << case1.order << ") " << y
            << " (order: " << case2.order << ")";
      } else {
        EXPECT_FALSE(
            Comparator->compare(case1.toByteArray(), case2.toByteArray()))
            << x << " (order: " << case1.order << ") " << y
            << " (order: " << case2.order << ")";
      }
    }
  }
}

TEST(Comparison, UnsignedByteArray) {
  // Check if UTF-8 is compared using unsigned correctly.
  auto Comparator =
      makeComparator<ByteArrayType>(Type::kByteArray, SortOrder::kUnsigned);

  std::string s1 = "arrange";
  std::string s2 = "arrangement";
  ByteArray s1ba = byteArrayFromString(s1);
  ByteArray s2ba = byteArrayFromString(s2);
  ASSERT_TRUE(Comparator->compare(s1ba, s2ba));

  // Multi-byte UTF-8 characters.
  s1 = "braten";
  s2 = "bügeln";
  s1ba = byteArrayFromString(s1);
  s2ba = byteArrayFromString(s2);
  ASSERT_TRUE(Comparator->compare(s1ba, s2ba));

  s1 = "ünk123456"; // ü = 252
  s2 = "ănk123456"; // ă = 259
  s1ba = byteArrayFromString(s1);
  s2ba = byteArrayFromString(s2);
  ASSERT_TRUE(Comparator->compare(s1ba, s2ba));
}

TEST(Comparison, SignedFLBA) {
  int size = 4;
  auto Comparator = makeComparator<FLBAType>(
      Type::kFixedLenByteArray, SortOrder::kSigned, size);

  std::vector<uint8_t> byteValues[] = {
      {0x80, 0, 0, 0},
      {0xFF, 0xFF, 0x01, 0},
      {0xFF, 0xFF, 0x80, 0},
      {0xFF, 0xFF, 0xFF, 0x80},
      {0xFF, 0xFF, 0xFF, 0xFF},
      {0, 0, 0x01, 0x01},
      {0, 0x01, 0x01, 0},
      {0x01, 0x01, 0, 0}};
  std::vector<FLBA> valuesToCompare;
  for (auto& bytes : byteValues) {
    valuesToCompare.emplace_back(FLBA(bytes.data()));
  }

  for (size_t x = 0; x < valuesToCompare.size(); x++) {
    EXPECT_FALSE(Comparator->compare(valuesToCompare[x], valuesToCompare[x]))
        << x;
    for (size_t y = x + 1; y < valuesToCompare.size(); y++) {
      EXPECT_TRUE(Comparator->compare(valuesToCompare[x], valuesToCompare[y]))
          << x << " " << y;
      EXPECT_FALSE(Comparator->compare(valuesToCompare[y], valuesToCompare[x]))
          << y << " " << x;
    }
  }
}

TEST(Comparison, UnsignedFLBA) {
  int size = 10;
  auto Comparator = makeComparator<FLBAType>(
      Type::kFixedLenByteArray, SortOrder::kUnsigned, size);

  std::string s1 = "Anti123456";
  std::string s2 = "Bunkd123456";
  FLBA s1flba = fLBAFromString(s1);
  FLBA s2flba = fLBAFromString(s2);
  ASSERT_TRUE(Comparator->compare(s1flba, s2flba));

  s1 = "Bunk123456";
  s2 = "Bünk123456";
  s1flba = fLBAFromString(s1);
  s2flba = fLBAFromString(s2);
  ASSERT_TRUE(Comparator->compare(s1flba, s2flba));
}

TEST(Comparison, SignedInt96) {
  Int96 a{{1, 41, 14}}, b{{1, 41, 42}};
  Int96 aa{{1, 41, 14}}, bb{{1, 41, 14}};
  Int96 aaa{{1, 41, static_cast<uint32_t>(-14)}}, bbb{{1, 41, 42}};

  auto Comparator = makeComparator<Int96Type>(Type::kInt96, SortOrder::kSigned);

  ASSERT_TRUE(Comparator->compare(a, b));
  ASSERT_TRUE(!Comparator->compare(aa, bb) && !Comparator->compare(bb, aa));
  ASSERT_TRUE(Comparator->compare(aaa, bbb));
}

TEST(Comparison, UnsignedInt96) {
  Int96 a{{1, 41, 14}}, b{{1, static_cast<uint32_t>(-41), 42}};
  Int96 aa{{1, 41, 14}}, bb{{1, 41, static_cast<uint32_t>(-14)}};
  Int96 aaa, bbb;

  auto Comparator =
      makeComparator<Int96Type>(Type::kInt96, SortOrder::kUnsigned);

  ASSERT_TRUE(Comparator->compare(a, b));
  ASSERT_TRUE(Comparator->compare(aa, bb));

  // INT96 Timestamp.
  aaa.value[2] = 2451545; // 2000-01-01
  bbb.value[2] = 2451546; // 2000-01-02
  // 12 Hours + 34 minutes + 56 seconds.
  int96SetNanoSeconds(aaa, 45296000000000);
  // 12 Hours + 34 minutes + 50 seconds.
  int96SetNanoSeconds(bbb, 45290000000000);
  ASSERT_TRUE(Comparator->compare(aaa, bbb));

  aaa.value[2] = 2451545; // 2000-01-01
  bbb.value[2] = 2451545; // 2000-01-01
  // 11 Hours + 34 minutes + 56 seconds.
  int96SetNanoSeconds(aaa, 41696000000000);
  // 12 Hours + 34 minutes + 50 seconds.
  int96SetNanoSeconds(bbb, 45290000000000);
  ASSERT_TRUE(Comparator->compare(aaa, bbb));

  aaa.value[2] = 2451545; // 2000-01-01
  bbb.value[2] = 2451545; // 2000-01-01
  // 12 Hours + 34 minutes + 55 seconds.
  int96SetNanoSeconds(aaa, 45295000000000);
  // 12 Hours + 34 minutes + 56 seconds.
  int96SetNanoSeconds(bbb, 45296000000000);
  ASSERT_TRUE(Comparator->compare(aaa, bbb));
}

TEST(Comparison, SignedInt64) {
  int64_t a = 1, b = 4;
  int64_t aa = 1, bb = 1;
  int64_t aaa = -1, bbb = 1;

  NodePtr Node =
      PrimitiveNode::make("SignedInt64", Repetition::kRequired, Type::kInt64);
  ColumnDescriptor descr(Node, 0, 0);

  auto Comparator = makeComparator<Int64Type>(&descr);

  ASSERT_TRUE(Comparator->compare(a, b));
  ASSERT_TRUE(!Comparator->compare(aa, bb) && !Comparator->compare(bb, aa));
  ASSERT_TRUE(Comparator->compare(aaa, bbb));
}

TEST(Comparison, UnsignedInt64) {
  uint64_t a = 1, b = 4;
  uint64_t aa = 1, bb = 1;
  uint64_t aaa = 1, bbb = -1;

  NodePtr Node = PrimitiveNode::make(
      "UnsignedInt64",
      Repetition::kRequired,
      Type::kInt64,
      ConvertedType::kUint64);
  ColumnDescriptor descr(Node, 0, 0);

  ASSERT_EQ(SortOrder::kUnsigned, descr.sortOrder());
  auto Comparator = makeComparator<Int64Type>(&descr);

  ASSERT_TRUE(Comparator->compare(a, b));
  ASSERT_TRUE(!Comparator->compare(aa, bb) && !Comparator->compare(bb, aa));
  ASSERT_TRUE(Comparator->compare(aaa, bbb));
}

TEST(Comparison, UnsignedInt32) {
  uint32_t a = 1, b = 4;
  uint32_t aa = 1, bb = 1;
  uint32_t aaa = 1, bbb = -1;

  NodePtr Node = PrimitiveNode::make(
      "UnsignedInt32",
      Repetition::kRequired,
      Type::kInt32,
      ConvertedType::kUint32);
  ColumnDescriptor descr(Node, 0, 0);

  ASSERT_EQ(SortOrder::kUnsigned, descr.sortOrder());
  auto Comparator = makeComparator<Int32Type>(&descr);

  ASSERT_TRUE(Comparator->compare(a, b));
  ASSERT_TRUE(!Comparator->compare(aa, bb) && !Comparator->compare(bb, aa));
  ASSERT_TRUE(Comparator->compare(aaa, bbb));
}

TEST(Comparison, UnknownSortOrder) {
  NodePtr Node = PrimitiveNode::make(
      "Unknown",
      Repetition::kRequired,
      Type::kFixedLenByteArray,
      ConvertedType::kInterval,
      12);
  ColumnDescriptor descr(Node, 0, 0);

  ASSERT_THROW(Comparator::make(&descr), ParquetException);
}

// ----------------------------------------------------------------------.

template <typename TestType>
class TestStatistics : public PrimitiveTypedTest<TestType> {
 public:
  using CType = typename TestType::CType;

  std::vector<CType> getDeepCopy(
      const std::vector<CType>&); // allocates new memory for FLBA/ByteArray

  CType* getValuesPointer(std::vector<CType>&);
  void deepFree(std::vector<CType>&);

  void testMinMaxEncode() {
    this->generateData(1000);

    auto statistics1 = makeStatistics<TestType>(this->schema_.column(0));
    statistics1->update(this->valuesPtr_, this->values_.size(), 0);
    std::string encodedMin = statistics1->encodeMin();
    std::string encodedMax = statistics1->encodeMax();

    auto statistics2 = makeStatistics<TestType>(
        this->schema_.column(0),
        encodedMin,
        encodedMax,
        this->values_.size(),
        0, // nullCount.
        0, // distinctCount.
        true, // hasMinMax.
        true, // hasNullCount.
        true, // hasDistinctCount.
        false, // hasNaNCount.
        0); // nanCount.

    auto statistics3 = makeStatistics<TestType>(this->schema_.column(0));
    std::vector<uint8_t> validBits(
        ::arrow::bit_util::BytesForBits(
            static_cast<uint32_t>(this->values_.size())) +
            1,
        255);
    statistics3->updateSpaced(
        this->valuesPtr_,
        validBits.data(),
        0,
        this->values_.size(),
        this->values_.size(),
        0);
    std::string encodedMinSpaced = statistics3->encodeMin();
    std::string encodedMaxSpaced = statistics3->encodeMax();

    ASSERT_EQ(encodedMin, statistics2->encodeMin());
    ASSERT_EQ(encodedMax, statistics2->encodeMax());
    ASSERT_EQ(statistics1->min(), statistics2->min());
    ASSERT_EQ(statistics1->max(), statistics2->max());
    ASSERT_EQ(encodedMinSpaced, statistics2->encodeMin());
    ASSERT_EQ(encodedMaxSpaced, statistics2->encodeMax());
    ASSERT_EQ(statistics3->min(), statistics2->min());
    ASSERT_EQ(statistics3->max(), statistics2->max());
  }

  void testReset() {
    this->generateData(1000);

    auto Statistics = makeStatistics<TestType>(this->schema_.column(0));
    Statistics->update(this->valuesPtr_, this->values_.size(), 0);
    ASSERT_EQ(this->values_.size(), Statistics->numValues());

    Statistics->reset();
    ASSERT_TRUE(Statistics->hasNullCount());
    ASSERT_FALSE(Statistics->hasMinMax());
    ASSERT_FALSE(Statistics->hasDistinctCount());
    ASSERT_EQ(0, Statistics->nullCount());
    ASSERT_EQ(0, Statistics->numValues());
    ASSERT_EQ(0, Statistics->distinctCount());
    ASSERT_EQ("", Statistics->encodeMin());
    ASSERT_EQ("", Statistics->encodeMax());
  }

  void testMerge() {
    int numNull[2];
    randomNumbers(2, 42, 0, 100, numNull);

    auto statistics1 = makeStatistics<TestType>(this->schema_.column(0));
    this->generateData(1000);
    statistics1->update(
        this->valuesPtr_, this->values_.size() - numNull[0], numNull[0]);

    auto statistics2 = makeStatistics<TestType>(this->schema_.column(0));
    this->generateData(1000);
    statistics2->update(
        this->valuesPtr_, this->values_.size() - numNull[1], numNull[1]);

    auto total = makeStatistics<TestType>(this->schema_.column(0));
    total->merge(*statistics1);
    total->merge(*statistics2);

    ASSERT_EQ(numNull[0] + numNull[1], total->nullCount());
    ASSERT_EQ(
        this->values_.size() * 2 - numNull[0] - numNull[1], total->numValues());
    ASSERT_EQ(total->min(), std::min(statistics1->min(), statistics2->min()));
    ASSERT_EQ(total->max(), std::max(statistics1->max(), statistics2->max()));
  }

  void testEquals() {
    const auto nValues = 1;
    auto statisticsHaveMinmax1 =
        makeStatistics<TestType>(this->schema_.column(0));
    const auto seed1 = 1;
    this->generateData(nValues, seed1);
    statisticsHaveMinmax1->update(this->valuesPtr_, this->values_.size(), 0);
    auto statisticsHaveMinmax2 =
        makeStatistics<TestType>(this->schema_.column(0));
    const auto seed2 = 9999;
    this->generateData(nValues, seed2);
    statisticsHaveMinmax2->update(this->valuesPtr_, this->values_.size(), 0);
    auto statisticsNoMinmax = makeStatistics<TestType>(this->schema_.column(0));

    ASSERT_EQ(true, statisticsHaveMinmax1->equals(*statisticsHaveMinmax1));
    ASSERT_EQ(true, statisticsNoMinmax->equals(*statisticsNoMinmax));
    ASSERT_EQ(false, statisticsHaveMinmax1->equals(*statisticsHaveMinmax2));
    ASSERT_EQ(false, statisticsHaveMinmax1->equals(*statisticsNoMinmax));
  }

  void testFullRoundtrip(int64_t numValues, int64_t nullCount) {
    this->generateData(numValues);

    // Compute statistics for the whole batch.
    auto expectedStats = makeStatistics<TestType>(this->schema_.column(0));
    expectedStats->update(this->valuesPtr_, numValues - nullCount, nullCount);

    auto sink = createOutputStream();
    auto gnode = std::static_pointer_cast<GroupNode>(this->node_);
    std::shared_ptr<WriterProperties> WriterProperties =
        WriterProperties::Builder().enableStatistics("column")->build();
    auto fileWriter = ParquetFileWriter::open(sink, gnode, WriterProperties);
    auto rowGroupWriter = fileWriter->appendRowGroup();
    auto columnWriter =
        static_cast<TypedColumnWriter<TestType>*>(rowGroupWriter->nextColumn());

    // Simulate the case when data comes from multiple buffers,
    // in which case special care is necessary for FLBA/ByteArray types.
    for (int i = 0; i < 2; i++) {
      int64_t batchNumValues = i ? numValues - numValues / 2 : numValues / 2;
      int64_t batchNullCount = i ? nullCount : 0;
      VELOX_DCHECK(nullCount <= numValues); // avoid too much headache
      std::vector<int16_t> definitionLevels(batchNullCount, 0);
      definitionLevels.insert(
          definitionLevels.end(), batchNumValues - batchNullCount, 1);
      auto beg = this->values_.cbegin() + i * numValues / 2;
      auto end = beg + batchNumValues;
      std::vector<CType> batch = getDeepCopy(std::vector<CType>(beg, end));
      CType* batchValuesPtr = getValuesPointer(batch);
      columnWriter->writeBatch(
          batchNumValues, definitionLevels.data(), nullptr, batchValuesPtr);
      deepFree(batch);
    }
    columnWriter->close();
    rowGroupWriter->close();
    fileWriter->close();

    ASSERT_OK_AND_ASSIGN(auto buffer, sink->Finish());

    // Write the buffer to a temp file.
    auto filePath = exec::test::TempFilePath::create();
    writeToFile(filePath, buffer);
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    std::shared_ptr<facebook::velox::memory::MemoryPool> rootPool =
        memory::memoryManager()->addRootPool("StatisticsTest");
    std::shared_ptr<facebook::velox::memory::MemoryPool> leafPool =
        rootPool->addLeafChild("StatisticsTest");
    dwio::common::ReaderOptions readerOptions{leafPool.get()};
    auto input = std::make_unique<dwio::common::BufferedInput>(
        std::make_shared<LocalReadFile>(filePath->getPath()),
        readerOptions.memoryPool());
    auto reader =
        std::make_unique<ParquetReader>(std::move(input), readerOptions);
    auto rowGroup = reader->fileMetaData().rowGroup(0);
    auto columnChunk = rowGroup.columnChunk(0);
    EXPECT_EQ(nullCount, columnChunk.getColumnMetadataStatsNullCount());
    EXPECT_TRUE(expectedStats->hasMinMax());
    EXPECT_EQ(
        expectedStats->encodeMin(),
        columnChunk.getColumnMetadataStatsMinValue());
    EXPECT_EQ(
        expectedStats->encodeMax(),
        columnChunk.getColumnMetadataStatsMaxValue());
    auto columnStats =
        columnChunk.getColumnStatistics(INTEGER(), rowGroup.numRows());
    EXPECT_EQ(numValues - nullCount, columnStats->getNumberOfValues());
  }
};

template <typename TestType>
typename TestType::CType* TestStatistics<TestType>::getValuesPointer(
    std::vector<typename TestType::CType>& values) {
  return values.data();
}

template <>
bool* TestStatistics<BooleanType>::getValuesPointer(std::vector<bool>& values) {
  static std::vector<uint8_t> boolBuffer;
  boolBuffer.clear();
  boolBuffer.resize(values.size());
  std::copy(values.begin(), values.end(), boolBuffer.begin());
  return reinterpret_cast<bool*>(boolBuffer.data());
}

template <typename TestType>
typename std::vector<typename TestType::CType>
TestStatistics<TestType>::getDeepCopy(
    const std::vector<typename TestType::CType>& values) {
  return values;
}

template <>
std::vector<FLBA> TestStatistics<FLBAType>::getDeepCopy(
    const std::vector<FLBA>& values) {
  std::vector<FLBA> copy;
  MemoryPool* pool = ::arrow::default_memory_pool();
  for (const FLBA& flba : values) {
    uint8_t* ptr;
    PARQUET_THROW_NOT_OK(pool->Allocate(FLBA_LENGTH, &ptr));
    memcpy(ptr, flba.ptr, FLBA_LENGTH);
    copy.emplace_back(ptr);
  }
  return copy;
}

template <>
std::vector<ByteArray> TestStatistics<ByteArrayType>::getDeepCopy(
    const std::vector<ByteArray>& values) {
  std::vector<ByteArray> copy;
  MemoryPool* pool = default_memory_pool();
  for (const ByteArray& ba : values) {
    uint8_t* ptr;
    PARQUET_THROW_NOT_OK(pool->Allocate(ba.len, &ptr));
    memcpy(ptr, ba.ptr, ba.len);
    copy.emplace_back(ba.len, ptr);
  }
  return copy;
}

template <typename TestType>
void TestStatistics<TestType>::deepFree(
    std::vector<typename TestType::CType>& values) {}

template <>
void TestStatistics<FLBAType>::deepFree(std::vector<FLBA>& values) {
  MemoryPool* pool = default_memory_pool();
  for (FLBA& flba : values) {
    auto ptr = const_cast<uint8_t*>(flba.ptr);
    memset(ptr, 0, FLBA_LENGTH);
    pool->Free(ptr, FLBA_LENGTH);
  }
}

template <>
void TestStatistics<ByteArrayType>::deepFree(std::vector<ByteArray>& values) {
  MemoryPool* pool = default_memory_pool();
  for (ByteArray& ba : values) {
    auto ptr = const_cast<uint8_t*>(ba.ptr);
    memset(ptr, 0, ba.len);
    pool->Free(ptr, ba.len);
  }
}

template <>
void TestStatistics<ByteArrayType>::testMinMaxEncode() {
  this->generateData(1000);
  // Test that we encode min max strings correctly.
  auto statistics1 = makeStatistics<ByteArrayType>(this->schema_.column(0));
  statistics1->update(this->valuesPtr_, this->values_.size(), 0);
  std::string encodedMin = statistics1->encodeMin();
  std::string encodedMax = statistics1->encodeMax();

  // Encoded is same as unencoded.
  ASSERT_EQ(
      encodedMin,
      std::string(
          reinterpret_cast<const char*>(statistics1->min().ptr),
          statistics1->min().len));
  ASSERT_EQ(
      encodedMax,
      std::string(
          reinterpret_cast<const char*>(statistics1->max().ptr),
          statistics1->max().len));

  auto statistics2 = makeStatistics<ByteArrayType>(
      this->schema_.column(0),
      encodedMin,
      encodedMax,
      this->values_.size(),
      0, // nullCount
      0, // distinctCount
      true, // hasMinMax
      true, // hasNullCount
      true, // hasDistinctCount
      false, // hasNaNCount
      0); // nanCount

  ASSERT_EQ(encodedMin, statistics2->encodeMin());
  ASSERT_EQ(encodedMax, statistics2->encodeMax());
  ASSERT_EQ(statistics1->min(), statistics2->min());
  ASSERT_EQ(statistics1->max(), statistics2->max());
}

using Types = ::testing::Types<
    Int32Type,
    Int64Type,
    FloatType,
    DoubleType,
    ByteArrayType,
    FLBAType,
    BooleanType>;

TYPED_TEST_SUITE(TestStatistics, Types);

TYPED_TEST(TestStatistics, MinMaxEncode) {
  this->setUpSchema(Repetition::kRequired);
  ASSERT_NO_FATAL_FAILURE(this->testMinMaxEncode());
}

TYPED_TEST(TestStatistics, reset) {
  this->setUpSchema(Repetition::kOptional);
  ASSERT_NO_FATAL_FAILURE(this->testReset());
}

TYPED_TEST(TestStatistics, equals) {
  this->setUpSchema(Repetition::kOptional);
  ASSERT_NO_FATAL_FAILURE(this->testEquals());
}

TYPED_TEST(TestStatistics, FullRoundtrip) {
  this->setUpSchema(Repetition::kOptional);
  ASSERT_NO_FATAL_FAILURE(this->testFullRoundtrip(100, 31));
  ASSERT_NO_FATAL_FAILURE(this->testFullRoundtrip(1000, 415));
  ASSERT_NO_FATAL_FAILURE(this->testFullRoundtrip(10000, 926));
}

template <typename TestType>
class TestNumericStatistics : public TestStatistics<TestType> {};

using NumericTypes =
    ::testing::Types<Int32Type, Int64Type, FloatType, DoubleType>;

TYPED_TEST_SUITE(TestNumericStatistics, NumericTypes);

TYPED_TEST(TestNumericStatistics, merge) {
  this->setUpSchema(Repetition::kOptional);
  ASSERT_NO_FATAL_FAILURE(this->testMerge());
}

TYPED_TEST(TestNumericStatistics, equals) {
  this->setUpSchema(Repetition::kOptional);
  ASSERT_NO_FATAL_FAILURE(this->testEquals());
}

template <typename TestType>
class TestStatisticsHasFlag : public TestStatistics<TestType> {
 public:
  void SetUp() override {
    this->setUpSchema(Repetition::kOptional);
  }

  std::shared_ptr<TypedStatistics<TestType>> mergedStatistics(
      const TypedStatistics<TestType>& stats1,
      const TypedStatistics<TestType>& stats2) {
    auto chunkStatistics = makeStatistics<TestType>(this->schema_.column(0));
    chunkStatistics->merge(stats1);
    chunkStatistics->merge(stats2);
    return chunkStatistics;
  }

  void verifyMergedStatistics(
      const TypedStatistics<TestType>& stats1,
      const TypedStatistics<TestType>& stats2,
      const std::function<void(TypedStatistics<TestType>*)>& testFn) {
    ASSERT_NO_FATAL_FAILURE(testFn(mergedStatistics(stats1, stats2).get()));
    ASSERT_NO_FATAL_FAILURE(testFn(mergedStatistics(stats2, stats1).get()));
  }

  // Distinct count should set to false when Merge is called.
  void testMergeDistinctCount() {
    // Create a statistics object with distinct count.
    std::shared_ptr<TypedStatistics<TestType>> statistics1;
    {
      EncodedStatistics encodedStatistics1;
      statistics1 = std::dynamic_pointer_cast<TypedStatistics<TestType>>(
          Statistics::make(this->schema_.column(0), &encodedStatistics1, 1000));
      EXPECT_FALSE(statistics1->hasDistinctCount());
    }

    // Create a statistics object with distinct count.
    std::shared_ptr<TypedStatistics<TestType>> statistics2;
    {
      EncodedStatistics encodedStatistics2;
      encodedStatistics2.hasDistinctCount = true;
      encodedStatistics2.distinctCount = 500;
      statistics2 = std::dynamic_pointer_cast<TypedStatistics<TestType>>(
          Statistics::make(this->schema_.column(0), &encodedStatistics2, 1000));
      EXPECT_TRUE(statistics2->hasDistinctCount());
    }

    verifyMergedStatistics(
        *statistics1,
        *statistics2,
        [](TypedStatistics<TestType>* mergedStatistics) {
          EXPECT_FALSE(mergedStatistics->hasDistinctCount());
          EXPECT_FALSE(mergedStatistics->encode().hasDistinctCount);
        });
  }

  // If all values in a page are null or nan, its stats should not set min-max.
  // Merging its stats with another page having good min-max stats should not
  // drop the valid min-max from the latter page.
  void testMergeMinMax() {
    this->generateData(1000);
    // Create a statistics object without min-max.
    std::shared_ptr<TypedStatistics<TestType>> statistics1;
    {
      statistics1 = makeStatistics<TestType>(this->schema_.column(0));
      statistics1->update(this->valuesPtr_, 0, this->values_.size());
      auto encodedStats1 = statistics1->encode();
      EXPECT_FALSE(statistics1->hasMinMax());
      EXPECT_FALSE(encodedStats1.hasMin);
      EXPECT_FALSE(encodedStats1.hasMax);
    }
    // Create a statistics object with min-max.
    std::shared_ptr<TypedStatistics<TestType>> statistics2;
    {
      statistics2 = makeStatistics<TestType>(this->schema_.column(0));
      statistics2->update(this->valuesPtr_, this->values_.size(), 0);
      auto encodedStats2 = statistics2->encode();
      EXPECT_TRUE(statistics2->hasMinMax());
      EXPECT_TRUE(encodedStats2.hasMin);
      EXPECT_TRUE(encodedStats2.hasMax);
    }
    verifyMergedStatistics(
        *statistics1,
        *statistics2,
        [](TypedStatistics<TestType>* mergedStatistics) {
          EXPECT_TRUE(mergedStatistics->hasMinMax());
          EXPECT_TRUE(mergedStatistics->encode().hasMin);
          EXPECT_TRUE(mergedStatistics->encode().hasMax);
        });
  }

  // Default statistics should have null_count even if no nulls is written.
  // However, if statistics is created from thrift message, it might not
  // have null_count. Merging statistics from such page will result in an
  // invalid null_count as well.
  void testMergeNullCount() {
    this->generateData(1000);

    // Page should have null-count even if no nulls.
    std::shared_ptr<TypedStatistics<TestType>> statistics1;
    {
      statistics1 = makeStatistics<TestType>(this->schema_.column(0));
      statistics1->update(this->valuesPtr_, this->values_.size(), 0);
      auto encodedStats1 = statistics1->encode();
      EXPECT_TRUE(statistics1->hasNullCount());
      EXPECT_EQ(0, statistics1->nullCount());
      EXPECT_TRUE(statistics1->encode().hasNullCount);
    }
    // Merge with null-count should also have null count.
    verifyMergedStatistics(
        *statistics1,
        *statistics1,
        [](TypedStatistics<TestType>* mergedStatistics) {
          EXPECT_TRUE(mergedStatistics->hasNullCount());
          EXPECT_EQ(0, mergedStatistics->nullCount());
          auto encoded = mergedStatistics->encode();
          EXPECT_TRUE(encoded.hasNullCount);
          EXPECT_EQ(0, encoded.nullCount);
        });

    // When loaded from thrift, might not have null count.
    std::shared_ptr<TypedStatistics<TestType>> statistics2;
    {
      EncodedStatistics encodedStatistics2;
      encodedStatistics2.hasNullCount = false;
      statistics2 = std::dynamic_pointer_cast<TypedStatistics<TestType>>(
          Statistics::make(this->schema_.column(0), &encodedStatistics2, 1000));
      EXPECT_FALSE(statistics2->encode().hasNullCount);
      EXPECT_FALSE(statistics2->hasNullCount());
    }

    // Merge without null-count should not have null count.
    verifyMergedStatistics(
        *statistics1,
        *statistics2,
        [](TypedStatistics<TestType>* mergedStatistics) {
          EXPECT_FALSE(mergedStatistics->hasNullCount());
          EXPECT_FALSE(mergedStatistics->encode().hasNullCount);
        });
  }

  // Statistics.all_null_value is used to build the page index.
  // If statistics doesn't have null count, all_null_value should be false.
  void testMissingNullCount() {
    EncodedStatistics EncodedStatistics;
    EncodedStatistics.hasNullCount = false;
    auto Statistics =
        Statistics::make(this->schema_.column(0), &EncodedStatistics, 1000);
    auto typedStats =
        std::dynamic_pointer_cast<TypedStatistics<TestType>>(Statistics);
    EXPECT_FALSE(typedStats->hasNullCount());
    auto encoded = typedStats->encode();
    EXPECT_FALSE(encoded.allNullValue);
    EXPECT_FALSE(encoded.hasNullCount);
    EXPECT_FALSE(encoded.hasDistinctCount);
    EXPECT_FALSE(encoded.hasMin);
    EXPECT_FALSE(encoded.hasMax);
  }
};

TYPED_TEST_SUITE(TestStatisticsHasFlag, Types);

TYPED_TEST(TestStatisticsHasFlag, MergeDistinctCount) {
  ASSERT_NO_FATAL_FAILURE(this->testMergeDistinctCount());
}

TYPED_TEST(TestStatisticsHasFlag, MergeNullCount) {
  ASSERT_NO_FATAL_FAILURE(this->testMergeNullCount());
}

TYPED_TEST(TestStatisticsHasFlag, MergeMinMax) {
  ASSERT_NO_FATAL_FAILURE(this->testMergeMinMax());
}

TYPED_TEST(TestStatisticsHasFlag, MissingNullCount) {
  ASSERT_NO_FATAL_FAILURE(this->testMissingNullCount());
}

// Helper for basic statistics tests below.
void assertStatsSet(
    const ApplicationVersion& version,
    std::shared_ptr<WriterProperties> props,
    const ColumnDescriptor* column,
    bool expectedIsSet) {
  auto metadataBuilder = ColumnChunkMetaDataBuilder::make(props, column);
  auto columnChunk = ColumnChunkMetaData::make(
      metadataBuilder->Contents(), column, defaultReaderProperties(), &version);
  EncodedStatistics stats;
  stats.setIsSigned(false);
  metadataBuilder->setStatistics(stats);
  ASSERT_EQ(columnChunk->isStatsSet(), expectedIsSet);
}

// Statistics are restricted for few types in older parquet version.
TEST(CorruptStatistics, Basics) {
  std::string createdBy = "parquet-mr version 1.8.0";
  ApplicationVersion version(createdBy);
  SchemaDescriptor schema;
  schema::NodePtr Node;
  std::vector<schema::NodePtr> fields;
  // Test Physical Types.
  fields.push_back(
      schema::PrimitiveNode::make(
          "col1", Repetition::kOptional, Type::kInt32, ConvertedType::kNone));
  fields.push_back(
      schema::PrimitiveNode::make(
          "col2",
          Repetition::kOptional,
          Type::kByteArray,
          ConvertedType::kNone));
  // Test Logical Types.
  fields.push_back(
      schema::PrimitiveNode::make(
          "col3", Repetition::kOptional, Type::kInt32, ConvertedType::kDate));
  fields.push_back(
      schema::PrimitiveNode::make(
          "col4", Repetition::kOptional, Type::kInt32, ConvertedType::kUint32));
  fields.push_back(
      schema::PrimitiveNode::make(
          "col5",
          Repetition::kOptional,
          Type::kFixedLenByteArray,
          ConvertedType::kInterval,
          12));
  fields.push_back(
      schema::PrimitiveNode::make(
          "col6",
          Repetition::kOptional,
          Type::kByteArray,
          ConvertedType::kUtf8));
  Node = schema::GroupNode::make("schema", Repetition::kRequired, fields);
  schema.init(Node);

  WriterProperties::Builder Builder;
  Builder.createdBy(createdBy);
  std::shared_ptr<WriterProperties> props = Builder.build();

  assertStatsSet(version, props, schema.column(0), true);
  assertStatsSet(version, props, schema.column(1), false);
  assertStatsSet(version, props, schema.column(2), true);
  assertStatsSet(version, props, schema.column(3), false);
  assertStatsSet(version, props, schema.column(4), false);
  assertStatsSet(version, props, schema.column(5), false);
}

// Statistics for all types have no restrictions in newer parquet version.
TEST(CorrectStatistics, Basics) {
  std::string createdBy = "parquet-cpp version 1.3.0";
  ApplicationVersion version(createdBy);
  SchemaDescriptor schema;
  schema::NodePtr Node;
  std::vector<schema::NodePtr> fields;
  // Test Physical Types.
  fields.push_back(
      schema::PrimitiveNode::make(
          "col1", Repetition::kOptional, Type::kInt32, ConvertedType::kNone));
  fields.push_back(
      schema::PrimitiveNode::make(
          "col2",
          Repetition::kOptional,
          Type::kByteArray,
          ConvertedType::kNone));
  // Test Logical Types.
  fields.push_back(
      schema::PrimitiveNode::make(
          "col3", Repetition::kOptional, Type::kInt32, ConvertedType::kDate));
  fields.push_back(
      schema::PrimitiveNode::make(
          "col4", Repetition::kOptional, Type::kInt32, ConvertedType::kUint32));
  fields.push_back(
      schema::PrimitiveNode::make(
          "col5",
          Repetition::kOptional,
          Type::kFixedLenByteArray,
          ConvertedType::kInterval,
          12));
  fields.push_back(
      schema::PrimitiveNode::make(
          "col6",
          Repetition::kOptional,
          Type::kByteArray,
          ConvertedType::kUtf8));
  Node = schema::GroupNode::make("schema", Repetition::kRequired, fields);
  schema.init(Node);

  WriterProperties::Builder Builder;
  Builder.createdBy(createdBy);
  std::shared_ptr<WriterProperties> props = Builder.build();

  assertStatsSet(version, props, schema.column(0), true);
  assertStatsSet(version, props, schema.column(1), true);
  assertStatsSet(version, props, schema.column(2), true);
  assertStatsSet(version, props, schema.column(3), true);
  assertStatsSet(version, props, schema.column(4), false);
  assertStatsSet(version, props, schema.column(5), true);
}

// Test SortOrder class.
static const int NUM_VALUES = 10;

template <typename TestType>
class TestStatisticsSortOrder : public ::testing::Test {
 public:
  using CType = typename TestType::CType;

  void addNodes(std::string name) {
    fields_.push_back(
        schema::PrimitiveNode::make(
            name,
            Repetition::kRequired,
            TestType::typeNum,
            ConvertedType::kNone));
  }

  void setUpSchema() {
    stats_.resize(fields_.size());
    values_.resize(NUM_VALUES);
    schema_ = std::static_pointer_cast<GroupNode>(
        GroupNode::make("Schema", Repetition::kRequired, fields_));

    parquetSink_ = createOutputStream();
  }

  void setValues();

  void writeParquet() {
    // Add writer properties.
    WriterProperties::Builder Builder;
    Builder.compression(Compression::SNAPPY);
    Builder.createdBy("parquet-cpp version 1.3.0");
    std::shared_ptr<WriterProperties> props = Builder.build();

    // Create a ParquetFileWriter instance.
    auto fileWriter = ParquetFileWriter::open(parquetSink_, schema_, props);

    // Append a RowGroup with a specific number of rows.
    auto rgWriter = fileWriter->appendRowGroup();

    this->setValues();

    // Insert Values.
    for (int i = 0; i < static_cast<int>(fields_.size()); i++) {
      auto columnWriter =
          static_cast<TypedColumnWriter<TestType>*>(rgWriter->nextColumn());
      columnWriter->writeBatch(NUM_VALUES, nullptr, nullptr, values_.data());
    }
  }

  void verifyParquetStats() {
    ASSERT_OK_AND_ASSIGN(auto pbuffer, parquetSink_->Finish());

    // Write the pbuffer to a temp file.
    auto filePath = exec::test::TempFilePath::create();
    writeToFile(filePath, pbuffer);
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    std::shared_ptr<facebook::velox::memory::MemoryPool> rootPool =
        memory::memoryManager()->addRootPool("StatisticsTest");
    std::shared_ptr<facebook::velox::memory::MemoryPool> leafPool =
        rootPool->addLeafChild("StatisticsTest");
    dwio::common::ReaderOptions readerOptions{leafPool.get()};
    auto input = std::make_unique<dwio::common::BufferedInput>(
        std::make_shared<LocalReadFile>(filePath->getPath()),
        readerOptions.memoryPool());
    auto reader =
        std::make_unique<ParquetReader>(std::move(input), readerOptions);
    auto rowGroup = reader->fileMetaData().rowGroup(0);
    for (int i = 0; i < static_cast<int>(fields_.size()); i++) {
      ARROW_SCOPED_TRACE("Statistics for field #", i);
      auto columnChunk = rowGroup.columnChunk(i);
      EXPECT_EQ(stats_[i].min(), columnChunk.getColumnMetadataStatsMinValue());
      EXPECT_EQ(stats_[i].max(), columnChunk.getColumnMetadataStatsMaxValue());
    }
  }

 protected:
  std::vector<CType> values_;
  std::vector<uint8_t> valuesBuf_;
  std::vector<schema::NodePtr> fields_;
  std::shared_ptr<schema::GroupNode> schema_;
  std::shared_ptr<::arrow::io::BufferOutputStream> parquetSink_;
  std::vector<EncodedStatistics> stats_;
};

using CompareTestTypes = ::testing::
    Types<Int32Type, Int64Type, FloatType, DoubleType, ByteArrayType, FLBAType>;

// TYPE::INT32.
template <>
void TestStatisticsSortOrder<Int32Type>::addNodes(std::string name) {
  // UINT_32 logical type to set Unsigned Statistics.
  fields_.push_back(
      schema::PrimitiveNode::make(
          name, Repetition::kRequired, Type::kInt32, ConvertedType::kUint32));
  // INT_32 logical type to set Signed Statistics.
  fields_.push_back(
      schema::PrimitiveNode::make(
          name, Repetition::kRequired, Type::kInt32, ConvertedType::kInt32));
}

template <>
void TestStatisticsSortOrder<Int32Type>::setValues() {
  for (int i = 0; i < NUM_VALUES; i++) {
    values_[i] = i - 5; // {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
  }

  // Write UINT32 min/max values.
  stats_[0]
      .setMin(
          std::string(
              reinterpret_cast<const char*>(&values_[5]), sizeof(CType)))
      .setMax(
          std::string(
              reinterpret_cast<const char*>(&values_[4]), sizeof(CType)));

  // Write INT32 min/max values.
  stats_[1]
      .setMin(
          std::string(
              reinterpret_cast<const char*>(&values_[0]), sizeof(CType)))
      .setMax(
          std::string(
              reinterpret_cast<const char*>(&values_[9]), sizeof(CType)));
}

// TYPE::INT64.
template <>
void TestStatisticsSortOrder<Int64Type>::addNodes(std::string name) {
  // UINT_64 logical type to set Unsigned Statistics.
  fields_.push_back(
      schema::PrimitiveNode::make(
          name, Repetition::kRequired, Type::kInt64, ConvertedType::kUint64));
  // INT_64 logical type to set Signed Statistics.
  fields_.push_back(
      schema::PrimitiveNode::make(
          name, Repetition::kRequired, Type::kInt64, ConvertedType::kInt64));
}

template <>
void TestStatisticsSortOrder<Int64Type>::setValues() {
  for (int i = 0; i < NUM_VALUES; i++) {
    values_[i] = i - 5; // {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
  }

  // Write UINT64 min/max values.
  stats_[0]
      .setMin(
          std::string(
              reinterpret_cast<const char*>(&values_[5]), sizeof(CType)))
      .setMax(
          std::string(
              reinterpret_cast<const char*>(&values_[4]), sizeof(CType)));

  // Write INT64 min/max values.
  stats_[1]
      .setMin(
          std::string(
              reinterpret_cast<const char*>(&values_[0]), sizeof(CType)))
      .setMax(
          std::string(
              reinterpret_cast<const char*>(&values_[9]), sizeof(CType)));
}

// TYPE::FLOAT.
template <>
void TestStatisticsSortOrder<FloatType>::setValues() {
  for (int i = 0; i < NUM_VALUES; i++) {
    values_[i] = static_cast<float>(i) -
        5; // {-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0};
  }

  // Write Float min/max values.
  stats_[0]
      .setMin(
          std::string(
              reinterpret_cast<const char*>(&values_[0]), sizeof(CType)))
      .setMax(
          std::string(
              reinterpret_cast<const char*>(&values_[9]), sizeof(CType)));
}

// TYPE::DOUBLE.
template <>
void TestStatisticsSortOrder<DoubleType>::setValues() {
  for (int i = 0; i < NUM_VALUES; i++) {
    values_[i] = static_cast<float>(i) -
        5; // {-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0};
  }

  // Write Double min/max values.
  stats_[0]
      .setMin(
          std::string(
              reinterpret_cast<const char*>(&values_[0]), sizeof(CType)))
      .setMax(
          std::string(
              reinterpret_cast<const char*>(&values_[9]), sizeof(CType)));
}

// TYPE::ByteArray.
template <>
void TestStatisticsSortOrder<ByteArrayType>::addNodes(std::string name) {
  // UTF8 logical type to set Unsigned Statistics.
  fields_.push_back(
      schema::PrimitiveNode::make(
          name, Repetition::kRequired, Type::kByteArray, ConvertedType::kUtf8));
}

template <>
void TestStatisticsSortOrder<ByteArrayType>::setValues() {
  int maxByteArrayLen = 10;
  size_t nbytes = NUM_VALUES * maxByteArrayLen;
  valuesBuf_.resize(nbytes);
  std::vector<std::string> vals = {
      "c123",
      "b123",
      "a123",
      "d123",
      "e123",
      "f123",
      "g123",
      "h123",
      "i123",
      "ü123"};

  uint8_t* base = &valuesBuf_.data()[0];
  for (int i = 0; i < NUM_VALUES; i++) {
    memcpy(base, vals[i].c_str(), vals[i].length());
    values_[i].ptr = base;
    values_[i].len = static_cast<uint32_t>(vals[i].length());
    base += vals[i].length();
  }

  // Write String min/max values.
  stats_[0]
      .setMin(
          std::string(
              reinterpret_cast<const char*>(vals[2].c_str()), vals[2].length()))
      .setMax(
          std::string(
              reinterpret_cast<const char*>(vals[9].c_str()),
              vals[9].length()));
}

// TYPE::FLBAArray.
template <>
void TestStatisticsSortOrder<FLBAType>::addNodes(std::string name) {
  // FLBA has only Unsigned Statistics.
  fields_.push_back(
      schema::PrimitiveNode::make(
          name,
          Repetition::kRequired,
          Type::kFixedLenByteArray,
          ConvertedType::kNone,
          FLBA_LENGTH));
}

template <>
void TestStatisticsSortOrder<FLBAType>::setValues() {
  size_t nbytes = NUM_VALUES * FLBA_LENGTH;
  valuesBuf_.resize(nbytes);
  char vals[NUM_VALUES][FLBA_LENGTH] = {
      "b12345",
      "a12345",
      "c12345",
      "d12345",
      "e12345",
      "f12345",
      "g12345",
      "h12345",
      "z12345",
      "a12345"};

  uint8_t* base = &valuesBuf_.data()[0];
  for (int i = 0; i < NUM_VALUES; i++) {
    memcpy(base, &vals[i][0], FLBA_LENGTH);
    values_[i].ptr = base;
    base += FLBA_LENGTH;
  }

  // Write FLBA min,max values.
  stats_[0]
      .setMin(
          std::string(reinterpret_cast<const char*>(&vals[1][0]), FLBA_LENGTH))
      .setMax(
          std::string(reinterpret_cast<const char*>(&vals[8][0]), FLBA_LENGTH));
}

TYPED_TEST_SUITE(TestStatisticsSortOrder, CompareTestTypes);

TYPED_TEST(TestStatisticsSortOrder, MinMax) {
  this->addNodes("Column ");
  this->setUpSchema();
  this->writeParquet();
  ASSERT_NO_FATAL_FAILURE(this->verifyParquetStats());
}

template <typename ArrowType>
void testByteArrayStatisticsFromArrow() {
  using TypeTraits = ::arrow::TypeTraits<ArrowType>;
  using ArrayType = typename TypeTraits::ArrayType;

  auto values = ::arrow::ArrayFromJSON(
      TypeTraits::type_singleton(),
      "[\"c123\", \"b123\", \"a123\", null, "
      "null, \"f123\", \"g123\", \"h123\", \"i123\", \"ü123\"]");

  const auto& typedValues = static_cast<const ArrayType&>(*values);

  NodePtr Node = PrimitiveNode::make(
      "field", Repetition::kRequired, Type::kByteArray, ConvertedType::kUtf8);
  ColumnDescriptor descr(Node, 0, 0);
  auto stats = makeStatistics<ByteArrayType>(&descr);
  ASSERT_NO_FATAL_FAILURE(stats->update(*values));

  ASSERT_EQ(ByteArray(typedValues.GetView(2)), stats->min());
  ASSERT_EQ(ByteArray(typedValues.GetView(9)), stats->max());
  ASSERT_EQ(2, stats->nullCount());
}

TEST(testByteArrayStatisticsFromArrow, StringType) {
  // Part of ARROW-3246. Replicating TestStatisticsSortOrder test but via Arrow.
  testByteArrayStatisticsFromArrow<::arrow::StringType>();
}

TEST(testByteArrayStatisticsFromArrow, LargeStringType) {
  testByteArrayStatisticsFromArrow<::arrow::LargeStringType>();
}

// Ensure Decimal sort order is handled properly.
using TestStatisticsSortOrderFLBA = TestStatisticsSortOrder<FLBAType>;

TEST_F(TestStatisticsSortOrderFLBA, decimalSortOrder) {
  this->fields_.push_back(
      schema::PrimitiveNode::make(
          "Column 0",
          Repetition::kRequired,
          Type::kFixedLenByteArray,
          ConvertedType::kDecimal,
          FLBA_LENGTH,
          12,
          2));
  this->setUpSchema();
  this->writeParquet();

  ASSERT_OK_AND_ASSIGN(auto pbuffer, parquetSink_->Finish());

  // Write the pbuffer to a temp file.
  auto filePath = exec::test::TempFilePath::create();
  writeToFile(filePath, pbuffer);
  memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  std::shared_ptr<facebook::velox::memory::MemoryPool> rootPool =
      memory::memoryManager()->addRootPool("StatisticsTest");
  std::shared_ptr<facebook::velox::memory::MemoryPool> leafPool =
      rootPool->addLeafChild("StatisticsTest");
  dwio::common::ReaderOptions readerOptions{leafPool.get()};
  auto input = std::make_unique<dwio::common::BufferedInput>(
      std::make_shared<LocalReadFile>(filePath->getPath()),
      readerOptions.memoryPool());
  auto reader =
      std::make_unique<ParquetReader>(std::move(input), readerOptions);
  auto rowGroup = reader->fileMetaData().rowGroup(0);
  auto columnChunk = rowGroup.columnChunk(0);
  ASSERT_TRUE(columnChunk.hasStatistics());
}

template <
    typename Stats,
    typename Array,
    typename T = typename Array::value_type>
void assertMinMaxAre(
    Stats stats,
    const Array& values,
    T expectedMin,
    T expectedMax) {
  stats->update(values.data(), values.size(), 0);
  ASSERT_TRUE(stats->hasMinMax());
  EXPECT_EQ(stats->min(), expectedMin);
  EXPECT_EQ(stats->max(), expectedMax);
}

template <typename Stats, typename Array, typename T = typename Stats::T>
void assertMinMaxAre(
    Stats stats,
    const Array& values,
    const uint8_t* validBitmap,
    T expectedMin,
    T expectedMax) {
  auto nValues = values.size();
  auto nullCount = ::arrow::internal::CountSetBits(validBitmap, nValues, 0);
  auto nonNullCount = nValues - nullCount;
  stats->updateSpaced(
      values.data(),
      validBitmap,
      0,
      nonNullCount + nullCount,
      nonNullCount,
      nullCount);
  ASSERT_TRUE(stats->hasMinMax());
  EXPECT_EQ(stats->min(), expectedMin);
  EXPECT_EQ(stats->max(), expectedMax);
}

template <typename Stats, typename Array>
void assertUnsetMinMax(Stats stats, const Array& values) {
  stats->update(values.data(), values.size(), 0);
  ASSERT_FALSE(stats->hasMinMax());
}

template <typename Stats, typename Array>
void assertUnsetMinMax(
    Stats stats,
    const Array& values,
    const uint8_t* validBitmap) {
  auto nValues = values.size();
  auto nullCount = ::arrow::internal::CountSetBits(validBitmap, nValues, 0);
  auto nonNullCount = nValues - nullCount;
  stats->updateSpaced(
      values.data(),
      validBitmap,
      0,
      nonNullCount + nullCount,
      nonNullCount,
      nullCount);
  ASSERT_FALSE(stats->hasMinMax());
}

template <typename ParquetType, typename T = typename ParquetType::CType>
void checkExtrema() {
  using UT = typename std::make_unsigned<T>::type;

  const T smin = std::numeric_limits<T>::min();
  const T smax = std::numeric_limits<T>::max();
  const T umin = SafeCopy<T>(std::numeric_limits<UT>::min());
  const T umax = SafeCopy<T>(std::numeric_limits<UT>::max());

  constexpr int kNumValues = 8;
  std::array<T, kNumValues> values{
      0, smin, smax, umin, umax, smin + 1, smax - 1, umax - 1};

  NodePtr unsignedNode = PrimitiveNode::make(
      "uint",
      Repetition::kOptional,
      LogicalType::intType(sizeof(T) * CHAR_BIT, false /*signed*/),
      ParquetType::typeNum);
  ColumnDescriptor unsignedDescr(unsignedNode, 1, 1);
  NodePtr signedNode = PrimitiveNode::make(
      "int",
      Repetition::kOptional,
      LogicalType::intType(sizeof(T) * CHAR_BIT, true /*signed*/),
      ParquetType::typeNum);
  ColumnDescriptor signedDescr(signedNode, 1, 1);

  {
    ARROW_SCOPED_TRACE(
        "unsigned statistics: umin = ",
        umin,
        ", umax = ",
        umax,
        ", node type = ",
        unsignedNode->logicalType()->toString(),
        ", physical type = ",
        unsignedDescr.physicalType(),
        ", sort order = ",
        unsignedDescr.sortOrder());
    auto unsignedStats = makeStatistics<ParquetType>(&unsignedDescr);
    assertMinMaxAre(unsignedStats, values, umin, umax);
  }
  {
    ARROW_SCOPED_TRACE(
        "signed statistics: smin = ",
        smin,
        ", smax = ",
        smax,
        ", node type = ",
        signedNode->logicalType()->toString(),
        ", physical type = ",
        signedDescr.physicalType(),
        ", sort order = ",
        signedDescr.sortOrder());
    auto signedStats = makeStatistics<ParquetType>(&signedDescr);
    assertMinMaxAre(signedStats, values, smin, smax);
  }

  // With validity bitmap.
  std::vector<bool> isValid = {
      true, false, false, false, false, true, true, true};
  std::shared_ptr<Buffer> validBitmap;
  ::arrow::BitmapFromVector(isValid, &validBitmap);
  {
    ARROW_SCOPED_TRACE(
        "spaced unsigned statistics: umin = ",
        umin,
        ", umax = ",
        umax,
        ", node type = ",
        unsignedNode->logicalType()->toString(),
        ", physical type = ",
        unsignedDescr.physicalType(),
        ", sort order = ",
        unsignedDescr.sortOrder());
    auto unsignedStats = makeStatistics<ParquetType>(&unsignedDescr);
    assertMinMaxAre(unsignedStats, values, validBitmap->data(), T{0}, umax - 1);
  }
  {
    ARROW_SCOPED_TRACE(
        "spaced signed statistics: smin = ",
        smin,
        ", smax = ",
        smax,
        ", node type = ",
        signedNode->logicalType()->toString(),
        ", physical type = ",
        signedDescr.physicalType(),
        ", sort order = ",
        signedDescr.sortOrder());
    auto signedStats = makeStatistics<ParquetType>(&signedDescr);
    assertMinMaxAre(
        signedStats, values, validBitmap->data(), smin + 1, smax - 1);
  }
}

TEST(TestStatistic, Int32Extrema) {
  checkExtrema<Int32Type>();
}
TEST(TestStatistic, Int64Extrema) {
  checkExtrema<Int64Type>();
}

// PARQUET-1225: Float NaN values may lead to incorrect min-max.
template <typename ParquetType>
void checkNaNs() {
  using T = typename ParquetType::CType;

  constexpr int kNumValues = 8;
  NodePtr Node =
      PrimitiveNode::make("f", Repetition::kOptional, ParquetType::typeNum);
  ColumnDescriptor descr(Node, 1, 1);

  constexpr T nan = std::numeric_limits<T>::quiet_NaN();
  constexpr T min = -4.0f;
  constexpr T max = 3.0f;

  std::array<T, kNumValues> allNans{nan, nan, nan, nan, nan, nan, nan, nan};
  std::array<T, kNumValues> someNans{
      nan, max, -3.0f, -1.0f, nan, 2.0f, min, nan};
  uint8_t validBitmap = 0x7F; // 0b01111111
  // NaNs excluded.
  uint8_t validBitmapNoNans = 0x6E; // 0b01101110

  // Test values.
  auto someNanStats = makeStatistics<ParquetType>(&descr);
  // Ingesting only nans should not yield valid min max.
  assertUnsetMinMax(someNanStats, allNans);
  EXPECT_EQ(someNanStats->nanCount(), allNans.size());
  // Ingesting a mix of NaNs and non-NaNs should not yield valid min max.
  assertMinMaxAre(someNanStats, someNans, min, max);
  // Ingesting only nans after a valid min/max, should have not effect.
  assertMinMaxAre(someNanStats, allNans, min, max);

  someNanStats = makeStatistics<ParquetType>(&descr);
  assertUnsetMinMax(someNanStats, allNans, &validBitmap);
  // NaNs should not pollute min max when excluded via null bitmap.
  assertMinMaxAre(someNanStats, someNans, &validBitmapNoNans, min, max);
  // Ingesting NaNs with a null bitmap should not change the result.
  assertMinMaxAre(someNanStats, someNans, &validBitmap, min, max);

  // An array that doesn't start with NaN.
  std::array<T, kNumValues> otherNans{
      1.5f, max, -3.0f, -1.0f, nan, 2.0f, min, nan};
  auto otherStats = makeStatistics<ParquetType>(&descr);
  assertMinMaxAre(otherStats, otherNans, min, max);
  EXPECT_EQ(otherStats->nanCount(), 2);
}

TEST(TestStatistic, NaNFloatValues) {
  checkNaNs<FloatType>();
}

TEST(TestStatistic, NaNDoubleValues) {
  checkNaNs<DoubleType>();
}

// ARROW-7376.
TEST(TestStatisticsSortOrderFloatNaN, NaNAndNullsInfiniteLoop) {
  constexpr int kNumValues = 8;
  NodePtr Node =
      PrimitiveNode::make("nan_float", Repetition::kOptional, Type::kFloat);
  ColumnDescriptor descr(Node, 1, 1);

  constexpr float nan = std::numeric_limits<float>::quiet_NaN();
  std::array<float, kNumValues> nansButLast{
      nan, nan, nan, nan, nan, nan, nan, 0.0f};

  uint8_t allButLastValid = 0x7F; // 0b01111111
  auto stats = makeStatistics<FloatType>(&descr);
  assertUnsetMinMax(stats, nansButLast, &allButLastValid);
  EXPECT_EQ(stats->nanCount(), kNumValues - 1);
}

template <
    typename Stats,
    typename Array,
    typename T = typename Array::value_type>
void assertMinMaxZeroesSign(Stats stats, const Array& values) {
  stats->update(values.data(), values.size(), 0);
  ASSERT_TRUE(stats->hasMinMax());

  T zero{};
  ASSERT_EQ(stats->min(), zero);
  ASSERT_TRUE(std::signbit(stats->min()));

  ASSERT_EQ(stats->max(), zero);
  ASSERT_FALSE(std::signbit(stats->max()));
}

// ARROW-5562: Ensure that -0.0f and 0.0f values are properly handled like in.
// Parquet-mr.
template <typename ParquetType>
void checkNegativeZeroStats() {
  using T = typename ParquetType::CType;

  NodePtr Node =
      PrimitiveNode::make("f", Repetition::kOptional, ParquetType::typeNum);
  ColumnDescriptor descr(Node, 1, 1);
  T zero{};

  {
    std::array<T, 2> values{-zero, zero};
    auto stats = makeStatistics<ParquetType>(&descr);
    assertMinMaxZeroesSign(stats, values);
  }

  {
    std::array<T, 2> values{zero, -zero};
    auto stats = makeStatistics<ParquetType>(&descr);
    assertMinMaxZeroesSign(stats, values);
  }

  {
    std::array<T, 2> values{-zero, -zero};
    auto stats = makeStatistics<ParquetType>(&descr);
    assertMinMaxZeroesSign(stats, values);
  }

  {
    std::array<T, 2> values{zero, zero};
    auto stats = makeStatistics<ParquetType>(&descr);
    assertMinMaxZeroesSign(stats, values);
  }
}

TEST(TestStatistics, FloatNegativeZero) {
  checkNegativeZeroStats<FloatType>();
}

TEST(TestStatistics, DoubleNegativeZero) {
  checkNegativeZeroStats<DoubleType>();
}

// Test infinity handling in statistics.
template <typename ParquetType>
void checkInfinityStats() {
  using T = typename ParquetType::CType;

  constexpr int32_t kNumValues = 8;
  NodePtr Node = PrimitiveNode::make(
      "infinity_test", Repetition::kOptional, ParquetType::typeNum);
  ColumnDescriptor descr(Node, 1, 1);

  constexpr T posInf = std::numeric_limits<T>::infinity();
  constexpr T negInf = -std::numeric_limits<T>::infinity();
  constexpr T min = -1.0f;
  constexpr T max = 1.0f;

  {
    std::array<T, kNumValues> allPosInf{
        posInf, posInf, posInf, posInf, posInf, posInf, posInf, posInf};
    auto stats = makeStatistics<ParquetType>(&descr);
    assertMinMaxAre(stats, allPosInf, posInf, posInf);
  }

  {
    std::array<T, kNumValues> allNegInf{
        negInf, negInf, negInf, negInf, negInf, negInf, negInf, negInf};
    auto stats = makeStatistics<ParquetType>(&descr);
    assertMinMaxAre(stats, allNegInf, negInf, negInf);
  }

  {
    std::array<T, kNumValues> mixedInf{
        posInf, negInf, posInf, negInf, posInf, negInf, posInf, negInf};
    auto stats = makeStatistics<ParquetType>(&descr);
    assertMinMaxAre(stats, mixedInf, negInf, posInf);
  }

  {
    std::array<T, kNumValues> mixedValues{
        posInf, max, min, min, negInf, max, min, posInf};
    auto stats = makeStatistics<ParquetType>(&descr);
    assertMinMaxAre(stats, mixedValues, negInf, posInf);
  }

  {
    constexpr T nan = std::numeric_limits<T>::quiet_NaN();
    std::array<T, kNumValues> mixedWithNan{
        posInf, nan, max, negInf, nan, min, posInf, nan};
    auto stats = makeStatistics<ParquetType>(&descr);
    assertMinMaxAre(stats, mixedWithNan, negInf, posInf);
  }
}

TEST(TestStatistics, FloatInfinityValues) {
  checkInfinityStats<FloatType>();
}

TEST(TestStatistics, DoubleInfinityValues) {
  checkInfinityStats<DoubleType>();
}

// Test infinity values with validity bitmap.
TEST(TestStatistics, InfinityWithNullBitmap) {
  constexpr int kNumValues = 8;
  NodePtr Node = PrimitiveNode::make(
      "infinity_null_test", Repetition::kOptional, Type::kFloat);
  ColumnDescriptor descr(Node, 1, 1);

  constexpr float posInf = std::numeric_limits<float>::infinity();
  constexpr float negInf = -std::numeric_limits<float>::infinity();

  // Test with some infinity values marked as null.
  std::array<float, kNumValues> valuesWithNulls{
      posInf, negInf, 1.0f, 2.0f, posInf, -1.0f, 3.0f, negInf};

  // Bitmap: exclude first posInf and last negInf (01111110 = 0x7E).
  uint8_t validBitmap = 0x7E;

  auto stats = makeStatistics<FloatType>(&descr);
  assertMinMaxAre(stats, valuesWithNulls, &validBitmap, negInf, posInf);
  valuesWithNulls = {posInf, 0.0f, 1.0f, 2.0f, -2.0f, -1.0f, 3.0f, negInf};

  stats = makeStatistics<FloatType>(&descr);
  assertMinMaxAre(stats, valuesWithNulls, &validBitmap, -2.0f, 3.0f);
}

// Test merging statistics with infinity values.
TEST(TestStatistics, MergeInfinityStatistics) {
  NodePtr Node = PrimitiveNode::make(
      "merge_infinity", Repetition::kOptional, Type::kDouble);
  ColumnDescriptor descr(Node, 1, 1);

  constexpr double posInf = std::numeric_limits<double>::infinity();
  constexpr double negInf = -std::numeric_limits<double>::infinity();

  auto stats1 = makeStatistics<DoubleType>(&descr);
  std::array<double, 3> normalValues{-1.0f, 0.0f, 1.0f};
  assertMinMaxAre(stats1, normalValues, -1.0f, 1.0f);

  auto stats2 = makeStatistics<DoubleType>(&descr);
  std::array<double, 2> infinityValues{negInf, posInf};
  assertMinMaxAre(stats2, infinityValues, negInf, posInf);

  auto mergedStats = makeStatistics<DoubleType>(&descr);
  mergedStats->merge(*stats1);
  mergedStats->merge(*stats2);

  // Result should have infinity bounds.
  ASSERT_TRUE(mergedStats->hasMinMax());
  ASSERT_EQ(negInf, mergedStats->min());
  ASSERT_EQ(posInf, mergedStats->max());
}

TEST(TestStatistics, CleanInfinityStatistics) {
  constexpr int kNumValues = 4;
  NodePtr Node = PrimitiveNode::make(
      "clean_stat_nullopt", Repetition::kOptional, Type::kFloat);
  ColumnDescriptor descr(Node, 1, 1);

  constexpr float nan = std::numeric_limits<float>::quiet_NaN();

  {
    std::array<float, kNumValues> allNans{nan, nan, nan, nan};
    auto stats = makeStatistics<FloatType>(&descr);
    assertUnsetMinMax(stats, allNans);
  }

  {
    std::array<float, kNumValues> values{1.0f, 2.0f, 3.0f, 4.0f};
    uint8_t allNullBitmap = 0x00;

    auto stats = makeStatistics<FloatType>(&descr);
    assertUnsetMinMax(stats, values, &allNullBitmap);
  }

  {
    std::array<float, kNumValues> mixedNans{nan, 1.0f, nan, 2.0f};
    uint8_t partialNullBitmap = 0x05;

    auto stats = makeStatistics<FloatType>(&descr);
    assertUnsetMinMax(stats, mixedNans, &partialNullBitmap);
  }
}

TEST(TestStatistics, InfinityCleanStatisticValid) {
  constexpr int kNumValues = 4;
  NodePtr Node = PrimitiveNode::make(
      "clean_stat_valid", Repetition::kOptional, Type::kDouble);
  ColumnDescriptor descr(Node, 1, 1);

  constexpr double posInf = std::numeric_limits<double>::infinity();
  constexpr double negInf = -std::numeric_limits<double>::infinity();
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();

  {
    std::array<double, kNumValues> mixedValues{posInf, nan, negInf, nan};
    auto stats = makeStatistics<DoubleType>(&descr);
    assertMinMaxAre(stats, mixedValues, negInf, posInf);
  }

  {
    std::array<double, 1> singleInf{negInf};
    auto stats = makeStatistics<DoubleType>(&descr);
    assertMinMaxAre(stats, singleInf, negInf, negInf);
  }
}

// TODO: disabled as it requires Arrow parquet data dir.
// Test statistics for binary column with UNSIGNED sort order.
/*
TEST(TestStatisticsSortOrderMinMax, Unsigned) {
  std::string dir_string(test::get_data_dir());
  std::stringstream ss;
  ss << dir_string << "/binary.parquet";
  auto path = ss.str();

  // The file is generated by parquet-mr 1.10.0, the first version that.
  // Supports correct statistics for binary data (see PARQUET-1025). It.
  // Contains a single column of binary type. Data is just single byte values.
  // From 0x00 to 0x0B.
  auto file_reader = ParquetFileReader::OpenFile(path);
  auto rg_reader = file_reader->RowGroup(0);
  auto metadata = rg_reader->metadata();
  auto column_schema = metadata->schema()->Column(0);
  ASSERT_EQ(SortOrder::kUnsigned, column_schema->sort_order());

  auto column_chunk = metadata->ColumnChunk(0);
  ASSERT_TRUE(column_chunk->is_stats_set());

  std::shared_ptr<Statistics> stats = column_chunk->statistics();
  ASSERT_TRUE(stats != NULL);
  ASSERT_EQ(0, stats->null_count());
  ASSERT_EQ(12, stats->num_values());
  ASSERT_EQ(0x00, stats->EncodeMin()[0]);
  ASSERT_EQ(0x0b, stats->EncodeMax()[0]);
}

TEST(TestEncodedStatistics, CopySafe) {
  EncodedStatistics encoded_statistics;
  encoded_statistics.set_max("abc");
  encoded_statistics.has_max = true;

  encoded_statistics.set_min("abc");
  encoded_statistics.has_min = true;

  EncodedStatistics copy_statistics = encoded_statistics;
  copy_statistics.set_max("abcd");
  copy_statistics.set_min("a");

  EXPECT_EQ("abc", encoded_statistics.min());
  EXPECT_EQ("abc", encoded_statistics.max());
}
*/

namespace {

constexpr int32_t kTruncLen = 16;

template <typename ParquetType, typename T>
std::shared_ptr<TypedStatistics<ParquetType>> makeStats(
    const ColumnDescriptor* descr,
    std::initializer_list<T> values) {
  auto stats = makeStatistics<ParquetType>(descr);
  std::vector<T> v(values);
  stats->update(v.data(), v.size(), 0);
  return stats;
}

std::shared_ptr<TypedStatistics<ByteArrayType>> makeStats(
    const ColumnDescriptor* descr,
    std::initializer_list<std::string> values) {
  auto stats = makeStatistics<ByteArrayType>(descr);
  std::vector<std::string> strings(values);
  std::vector<ByteArray> byteArrays;
  byteArrays.reserve(strings.size());
  for (const auto& s : strings) {
    byteArrays.push_back(byteArrayFromString(s));
  }
  stats->update(byteArrays.data(), byteArrays.size(), 0);
  return stats;
}

} // namespace

TEST(IcebergStatistics, decimalMinMaxValue) {
  const NodePtr Node = PrimitiveNode::make(
      "decimal_col",
      Repetition::kRequired,
      LogicalType::decimal(10, 2),
      Type::kInt64);
  ColumnDescriptor descr(Node, 0, 0);

  auto stats =
      makeStats<Int64Type, int64_t>(&descr, {12345, -67890, 100, 50000});

  ASSERT_TRUE(stats->hasMinMax());
  EXPECT_EQ(stats->min(), -67890);
  EXPECT_EQ(stats->max(), 50000);

  const auto lowerBound = stats->icebergLowerBoundInclusive(kTruncLen);
  const auto upperBound = stats->icebergUpperBoundExclusive(kTruncLen);

  ASSERT_TRUE(upperBound.has_value());

  // Verify the encoding is big-endian.
  int64_t decodedMin = ::arrow::bit_util::FromBigEndian(
      *reinterpret_cast<const int64_t*>(lowerBound.data()));
  int64_t decodedMax = ::arrow::bit_util::FromBigEndian(
      *reinterpret_cast<const int64_t*>(upperBound->data()));

  EXPECT_EQ(decodedMin, -67890);
  EXPECT_EQ(decodedMax, 50000);
}

TEST(IcebergStatistics, nonDecimalBounds) {
  const NodePtr Node =
      PrimitiveNode::make("int_col", Repetition::kRequired, Type::kInt64);
  ColumnDescriptor descr(Node, 0, 0);

  auto stats = makeStats<Int64Type, int64_t>(&descr, {100, -200, 300, 50});

  ASSERT_TRUE(stats->hasMinMax());

  // For non-decimal INT64, IcebergLowerBound should equal EncodeMin (plain.
  // Encoding).
  EXPECT_EQ(stats->icebergLowerBoundInclusive(kTruncLen), stats->encodeMin());
  const auto upperBound = stats->icebergUpperBoundExclusive(kTruncLen);
  ASSERT_TRUE(upperBound.has_value());
  EXPECT_EQ(*upperBound, stats->encodeMax());
}

TEST(IcebergStatistics, byteArrayBounds) {
  NodePtr Node = PrimitiveNode::make(
      "string_col",
      Repetition::kRequired,
      Type::kByteArray,
      ConvertedType::kUtf8);
  ColumnDescriptor descr(Node, 0, 0);

  auto stats = makeStats(
      &descr,
      {"AAAAAAAAAAAAAAAAAAAAAAAAA", "ZZZZZZZZZZZZZZZZZZZZZZZZZ", "Hello"});

  ASSERT_TRUE(stats->hasMinMax());

  // IcebergLowerBound should be truncated to 16 characters.
  const auto lowerBound = stats->icebergLowerBoundInclusive(kTruncLen);
  EXPECT_EQ(lowerBound, "AAAAAAAAAAAAAAAA");

  // IcebergUpperBound should be truncated and incremented.
  const auto upperBound = stats->icebergUpperBoundExclusive(kTruncLen);
  ASSERT_TRUE(upperBound.has_value());
  // 'Z' (0x5A) incremented becomes '[' (0x5B).
  EXPECT_EQ(*upperBound, "ZZZZZZZZZZZZZZZ[");
}

TEST(IcebergStatistics, byteArrayBoundsNoTruncation) {
  NodePtr Node = PrimitiveNode::make(
      "string_col",
      Repetition::kRequired,
      Type::kByteArray,
      ConvertedType::kUtf8);
  ColumnDescriptor descr(Node, 0, 0);

  // Create strings shorter than 16 characters.
  auto stats = makeStats(&descr, {"apple", "zebra", "banana"});

  ASSERT_TRUE(stats->hasMinMax());

  // For short strings, IcebergLowerBound/IcebergUpperBound should be the same.
  // As EncodeMin/EncodeMax.
  EXPECT_EQ(stats->icebergLowerBoundInclusive(kTruncLen), stats->encodeMin());
  const auto upperBound = stats->icebergUpperBoundExclusive(kTruncLen);
  ASSERT_TRUE(upperBound.has_value());
  EXPECT_EQ(*upperBound, stats->encodeMax());
}

TEST(IcebergStatistics, byteArrayBoundsUnicode) {
  NodePtr Node = PrimitiveNode::make(
      "string_col",
      Repetition::kRequired,
      Type::kByteArray,
      ConvertedType::kUtf8);
  ColumnDescriptor descr(Node, 0, 0);

  // Create Unicode strings longer than 16 characters to trigger truncation.
  // Truncation is based on character count (16 chars), not byte count.
  // "世" is U+4E16 (3 bytes in UTF-8: E4 B8 96)
  // "界" is U+754C (3 bytes in UTF-8: E7 95 8C)
  // "你" is U+4F60 (3 bytes in UTF-8: E4 BD A0)
  // "好" is U+597D (3 bytes in UTF-8: E5 A5 BD)
  auto stats = makeStats(
      &descr,
      {"AAAA世界世界世界世界世界世界世",
       "ZZZZ你好你好你好你好你好你好你",
       "Hello, 世界!"});

  ASSERT_TRUE(stats->hasMinMax());

  const auto lowerBound = stats->icebergLowerBoundInclusive(kTruncLen);
  const auto upperBound = stats->icebergUpperBoundExclusive(kTruncLen);

  ASSERT_TRUE(upperBound.has_value());

  // Str1 has 18 characters (4 ASCII + 14 Chinese).
  // Truncated to 16 chars: "AAAA" (4) + "世界世界世界世界世界世界 (12)".
  const std::string expectedLower = "AAAA世界世界世界世界世界世界";
  EXPECT_EQ(lowerBound, expectedLower);

  // Str2 has 18 characters (4 ASCII + 14 Chinese).
  // For upperBound, the last character is incremented after truncation.
  // "好" (U+597D) incremented becomes U+597E "奾".
  const std::string expectedUpper = "ZZZZ你好你好你好你好你好你奾";
  EXPECT_EQ(*upperBound, expectedUpper);
}

TEST(IcebergStatistics, floatBounds) {
  NodePtr Node =
      PrimitiveNode::make("float_col", Repetition::kRequired, Type::kFloat);
  ColumnDescriptor descr(Node, 0, 0);

  auto stats = makeStats<FloatType, float>(&descr, {1.5f, -2.5f, 3.5f, 0.5f});

  ASSERT_TRUE(stats->hasMinMax());
  EXPECT_EQ(stats->icebergLowerBoundInclusive(kTruncLen), stats->encodeMin());
  const auto upperBound = stats->icebergUpperBoundExclusive(kTruncLen);
  ASSERT_TRUE(upperBound.has_value());
  EXPECT_EQ(*upperBound, stats->encodeMax());
}

TEST(IcebergStatistics, doubleBounds) {
  NodePtr Node =
      PrimitiveNode::make("double_col", Repetition::kRequired, Type::kDouble);
  ColumnDescriptor descr(Node, 0, 0);

  auto stats = makeStats<DoubleType, double>(&descr, {1.5, -2.5, 3.5, 0.5});

  ASSERT_TRUE(stats->hasMinMax());
  EXPECT_EQ(stats->icebergLowerBoundInclusive(kTruncLen), stats->encodeMin());
  const auto upperBound = stats->icebergUpperBoundExclusive(kTruncLen);
  ASSERT_TRUE(upperBound.has_value());
  EXPECT_EQ(*upperBound, stats->encodeMax());
}

TEST(IcebergStatistics, emptyStringBounds) {
  NodePtr Node = PrimitiveNode::make(
      "string_col",
      Repetition::kRequired,
      Type::kByteArray,
      ConvertedType::kUtf8);
  ColumnDescriptor descr(Node, 0, 0);

  {
    auto stats = makeStats(&descr, {"", "hello", ""});

    ASSERT_TRUE(stats->hasMinMax());
    EXPECT_EQ(stats->icebergLowerBoundInclusive(kTruncLen), "");

    const auto upperBound = stats->icebergUpperBoundExclusive(kTruncLen);
    ASSERT_TRUE(upperBound.has_value());
    EXPECT_EQ(*upperBound, "hello");
  }

  {
    auto stats = makeStats(&descr, {"", ""});

    ASSERT_TRUE(stats->hasMinMax());
    EXPECT_EQ(stats->min(), ByteArray(""));
    EXPECT_EQ(stats->max(), ByteArray(""));

    EXPECT_EQ(stats->icebergLowerBoundInclusive(kTruncLen), "");

    const auto upperBound = stats->icebergUpperBoundExclusive(kTruncLen);
    ASSERT_TRUE(upperBound.has_value());
    EXPECT_EQ(*upperBound, "");
  }
}

TEST(IcebergStatistics, unboundedUpperBound) {
  NodePtr Node = PrimitiveNode::make(
      "string_col",
      Repetition::kRequired,
      Type::kByteArray,
      ConvertedType::kUtf8);
  ColumnDescriptor descr(Node, 0, 0);

  {
    std::string allMaxAscii(20, '\x7F');
    auto stats = makeStats(&descr, {"hello", allMaxAscii});

    ASSERT_TRUE(stats->hasMinMax());

    const auto lowerBound = stats->icebergLowerBoundInclusive(kTruncLen);
    const auto upperBound = stats->icebergUpperBoundExclusive(kTruncLen);

    EXPECT_EQ(lowerBound, stats->encodeMin());
    EXPECT_FALSE(upperBound.has_value());
  }

  {
    std::string allMaxUnicode;
    allMaxUnicode.reserve(17 * 4);
    for (int i = 0; i < 17; ++i) {
      allMaxUnicode += "\U0010FFFF";
    }
    auto stats = makeStats(&descr, {"hello", allMaxUnicode});

    ASSERT_TRUE(stats->hasMinMax());

    const auto lowerBound = stats->icebergLowerBoundInclusive(kTruncLen);
    const auto upperBound = stats->icebergUpperBoundExclusive(kTruncLen);
    EXPECT_EQ(lowerBound, stats->encodeMin());
    EXPECT_FALSE(upperBound.has_value());
  }

  {
    std::string allMaxAscii(20, '\x7F');
    auto stats = makeStats(&descr, {allMaxAscii, allMaxAscii});

    ASSERT_TRUE(stats->hasMinMax());
    EXPECT_EQ(stats->min(), ByteArray(allMaxAscii));
    EXPECT_EQ(stats->max(), ByteArray(allMaxAscii));

    const auto lowerBound = stats->icebergLowerBoundInclusive(kTruncLen);
    const auto upperBound = stats->icebergUpperBoundExclusive(kTruncLen);
    EXPECT_TRUE(stats->encodeMin().starts_with(lowerBound));
    EXPECT_FALSE(upperBound.has_value());
  }

  {
    std::string allMaxUnicode;
    allMaxUnicode.reserve(17 * 4);
    for (int i = 0; i < 17; ++i) {
      allMaxUnicode += "\U0010FFFF";
    }
    auto stats = makeStats(&descr, {allMaxUnicode, allMaxUnicode});

    ASSERT_TRUE(stats->hasMinMax());
    EXPECT_EQ(stats->min(), ByteArray(allMaxUnicode));
    EXPECT_EQ(stats->max(), ByteArray(allMaxUnicode));

    const auto lowerBound = stats->icebergLowerBoundInclusive(kTruncLen);
    const auto upperBound = stats->icebergUpperBoundExclusive(kTruncLen);
    EXPECT_TRUE(stats->encodeMin().starts_with(lowerBound));
    EXPECT_FALSE(upperBound.has_value());
  }
}

TEST(StatisticsComparison, withInt64) {
  NodePtr Node =
      PrimitiveNode::make("int_col", Repetition::kRequired, Type::kInt64);
  ColumnDescriptor descr(Node, 0, 0);

  auto stats1 = makeStats<Int64Type, int64_t>(&descr, {10, 20, 30});
  auto stats2 = makeStats<Int64Type, int64_t>(&descr, {5, 15, 25});
  auto stats3 = makeStats<Int64Type, int64_t>(&descr, {10, 20, 30});

  ASSERT_TRUE(stats1->hasMinMax());
  ASSERT_TRUE(stats2->hasMinMax());
  ASSERT_TRUE(stats3->hasMinMax());

  EXPECT_TRUE(stats1->maxGreaterThan(*stats2));
  EXPECT_FALSE(stats2->maxGreaterThan(*stats1));
  EXPECT_TRUE(stats1->maxGreaterThan(*stats3));
  EXPECT_TRUE(stats3->maxGreaterThan(*stats1));

  EXPECT_FALSE(stats1->minLessThan(*stats2));
  EXPECT_TRUE(stats2->minLessThan(*stats1));
  EXPECT_FALSE(stats1->minLessThan(*stats3));
  EXPECT_FALSE(stats3->minLessThan(*stats1));
}

TEST(StatisticsComparison, withDouble) {
  NodePtr Node =
      PrimitiveNode::make("double_col", Repetition::kRequired, Type::kDouble);
  ColumnDescriptor descr(Node, 0, 0);

  auto stats1 = makeStats<DoubleType, double>(&descr, {1.0, 2.0, 3.0});
  auto stats2 = makeStats<DoubleType, double>(&descr, {0.5, 1.5, 2.5});

  ASSERT_TRUE(stats1->hasMinMax());
  ASSERT_TRUE(stats2->hasMinMax());

  EXPECT_TRUE(stats1->maxGreaterThan(*stats2));
  EXPECT_FALSE(stats2->maxGreaterThan(*stats1));

  EXPECT_FALSE(stats1->minLessThan(*stats2));
  EXPECT_TRUE(stats2->minLessThan(*stats1));
}

TEST(StatisticsComparison, withByteArray) {
  NodePtr Node = PrimitiveNode::make(
      "string_col",
      Repetition::kRequired,
      Type::kByteArray,
      ConvertedType::kUtf8);
  ColumnDescriptor descr(Node, 0, 0);

  auto stats1 = makeStats(&descr, {"apple", "zebra"});
  auto stats2 = makeStats(&descr, {"banana", "mangomango"});

  ASSERT_TRUE(stats1->hasMinMax());
  ASSERT_TRUE(stats2->hasMinMax());

  EXPECT_TRUE(stats1->maxGreaterThan(*stats2));
  EXPECT_FALSE(stats2->maxGreaterThan(*stats1));

  EXPECT_TRUE(stats1->minLessThan(*stats2));
  EXPECT_FALSE(stats2->minLessThan(*stats1));
}

} // namespace test
} // namespace facebook::velox::parquet::arrow
