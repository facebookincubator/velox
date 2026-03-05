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

#include "velox/dwio/common/SelectiveColumnReader.h"
#include "velox/dwio/common/SelectiveColumnReaderInternal.h"

#include <gtest/gtest.h>

#include <numeric>

#include "velox/common/memory/MemoryPool.h"
#include "velox/dwio/common/FormatData.h"
#include "velox/dwio/common/TypeWithId.h"

namespace facebook::velox::dwio::common {
namespace {

TEST(IsDenseTest, empty) {
  const RowSet rows(nullptr, nullptr);
  EXPECT_TRUE(isDense(rows));
}

TEST(IsDenseTest, singleElement) {
  const std::vector<int32_t> data{0};
  const RowSet rows(data.data(), data.size());
  EXPECT_TRUE(isDense(rows));
}

TEST(IsDenseTest, contiguousFromZero) {
  const std::vector<int32_t> data{0, 1, 2, 3, 4};
  const RowSet rows(data.data(), data.size());
  EXPECT_TRUE(isDense(rows));
}

TEST(IsDenseTest, sparseRows) {
  const std::vector<int32_t> data{0, 2, 4};
  const RowSet rows(data.data(), data.size());
  EXPECT_FALSE(isDense(rows));
}

TEST(IsDenseTest, startingFromNonZero) {
  const std::vector<int32_t> data{1, 2, 3};
  const RowSet rows(data.data(), data.size());
  EXPECT_FALSE(isDense(rows));
}

TEST(IsDenseTest, singleNonZeroElement) {
  const std::vector<int32_t> data{5};
  const RowSet rows(data.data(), data.size());
  EXPECT_FALSE(isDense(rows));
}

/// Minimal FormatData stub for testing SelectiveColumnReader in isolation.
class StubFormatData : public FormatData {
 public:
  void readNulls(
      vector_size_t /*numValues*/,
      const uint64_t* /*incomingNulls*/,
      BufferPtr& nulls,
      bool /*nullsOnly*/) override {
    nulls = nullptr;
  }
  uint64_t skipNulls(uint64_t numValues, bool /*nullsOnly*/) override {
    return numValues;
  }
  uint64_t skip(uint64_t numValues) override {
    return numValues;
  }
  bool hasNulls() const override {
    return false;
  }
  dwio::common::PositionProvider seekToRowGroup(int64_t /*index*/) override {
    static std::vector<uint64_t> empty;
    return dwio::common::PositionProvider(empty);
  }
  void filterRowGroups(
      const velox::common::ScanSpec& /*scanSpec*/,
      uint64_t /*rowsPerRowGroup*/,
      const StatsContext& /*writerContext*/,
      FilterRowGroupsResult& /*result*/) override {}
};

/// Minimal FormatParams stub that produces a StubFormatData.
class StubFormatParams : public FormatParams {
 public:
  StubFormatParams(memory::MemoryPool& pool, ColumnReaderStatistics& stats)
      : FormatParams(pool, stats) {}

  std::unique_ptr<FormatData> toFormatData(
      const std::shared_ptr<const dwio::common::TypeWithId>& /*type*/,
      const velox::common::ScanSpec& /*scanSpec*/) override {
    return std::make_unique<StubFormatData>();
  }
};

/// Concrete subclass that exposes getFlatValues and internal state for testing.
class TestColumnReader : public SelectiveColumnReader {
 public:
  TestColumnReader(
      const TypePtr& requestedType,
      std::shared_ptr<const dwio::common::TypeWithId> fileType,
      FormatParams& params,
      velox::common::ScanSpec& scanSpec)
      : SelectiveColumnReader(
            requestedType,
            std::move(fileType),
            params,
            scanSpec) {}

  void read(
      int64_t /*offset*/,
      const RowSet& /*rows*/,
      const uint64_t* /*incomingNulls*/) override {}

  void getValues(const RowSet& /*rows*/, VectorPtr* /*result*/) override {}

  /// Populate the internal values buffer with source data of type T.
  /// Sets valueSize_, numValues_, mayGetValues_, and inputRows_.
  template <typename T>
  void setupValues(
      const std::vector<T>& data,
      const std::vector<int32_t>& rowNumbers) {
    const auto n = static_cast<vector_size_t>(data.size());
    ensureValuesCapacity<T>(n);
    std::memcpy(rawValues_, data.data(), n * sizeof(T));
    numValues_ = n;
    valueSize_ = sizeof(T);
    mayGetValues_ = true;
    allNull_ = false;
    inputRows_ = RowSet(rowNumbers.data(), rowNumbers.size());
  }

  /// Populate internal values and mark elements null according to the mask.
  template <typename T>
  void setupValuesWithNulls(
      const std::vector<T>& data,
      const std::vector<bool>& nulls,
      const std::vector<int32_t>& rowNumbers) {
    VELOX_CHECK_EQ(data.size(), nulls.size());
    setupValues(data, rowNumbers);
    const auto n = static_cast<vector_size_t>(data.size());
    anyNulls_ = true;
    resultNulls_ = AlignedBuffer::allocate<bool>(
        n + simd::kPadding * 8, memoryPool_, bits::kNotNull);
    rawResultNulls_ = resultNulls_->asMutable<uint64_t>();
    for (int32_t i = 0; i < n; ++i) {
      bits::setBit(rawResultNulls_, i, !nulls[i]);
    }
  }

  using SelectiveColumnReader::getFlatValues;
};

class GetFlatValuesTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool("GetFlatValuesTest");
    stats_ = std::make_unique<ColumnReaderStatistics>();
    params_ = std::make_unique<StubFormatParams>(*pool_, *stats_);
    scanSpec_ = std::make_unique<velox::common::ScanSpec>("test");
    scanSpec_->setProjectOut(true);
  }

  std::unique_ptr<TestColumnReader> makeReader(
      const TypePtr& requestedType) const {
    auto fileType = TypeWithId::create(requestedType);
    return std::make_unique<TestColumnReader>(
        requestedType, std::move(fileType), *params_, *scanSpec_);
  }

  /// Run getFlatValues and verify each element equals static_cast<Target>(src).
  template <typename Source, typename Target>
  void testConversion(const TypePtr& type, const std::vector<Source>& data) {
    auto reader = makeReader(type);
    std::vector<int32_t> rowNums(data.size());
    std::iota(rowNums.begin(), rowNums.end(), 0);
    reader->setupValues(data, rowNums);

    RowSet const rows(rowNums.data(), rowNums.size());
    VectorPtr result;
    reader->getFlatValues<Source, Target>(rows, &result, type, true);

    auto* flat = result->as<FlatVector<Target>>();
    ASSERT_NE(flat, nullptr);
    ASSERT_EQ(flat->size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      EXPECT_EQ(flat->valueAt(i), static_cast<Target>(data[i]))
          << "Mismatch at index " << i;
    }
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::unique_ptr<ColumnReaderStatistics> stats_;
  std::unique_ptr<StubFormatParams> params_;
  std::unique_ptr<velox::common::ScanSpec> scanSpec_;
};

// Cross-domain upcast conversions (different sizes, valid via
// upcastScalarValues).

TEST_F(GetFlatValuesTest, int16ToFloat) {
  testConversion<int16_t, float>(REAL(), {1, -32'000, 0, 127, 32'767});
}

TEST_F(GetFlatValuesTest, int32ToDouble) {
  testConversion<int32_t, double>(
      DOUBLE(), {1, -999'999, 2'147'483'647, 0, -2'147'483'647});
}

TEST_F(GetFlatValuesTest, int16ToDouble) {
  testConversion<int16_t, double>(DOUBLE(), {0, 1, -1, 32'767, -32'768});
}

// Same-domain conversions as regression guards.

TEST_F(GetFlatValuesTest, int32ToInt32) {
  testConversion<int32_t, int32_t>(INTEGER(), {0, 1, -1, 42, 2'147'483'647});
}

TEST_F(GetFlatValuesTest, int16ToInt32) {
  testConversion<int16_t, int32_t>(INTEGER(), {0, 1, -1, 32'767, -32'768});
}

TEST_F(GetFlatValuesTest, int32ToInt16) {
  testConversion<int32_t, int16_t>(SMALLINT(), {0, 1, -1, 127, -128});
}

TEST_F(GetFlatValuesTest, int64ToInt16) {
  testConversion<int64_t, int16_t>(SMALLINT(), {0, 1, -1, 127, -128});
}

TEST_F(GetFlatValuesTest, int32ToInt8) {
  testConversion<int32_t, int8_t>(TINYINT(), {0, 1, -1, 42, -100});
}

TEST_F(GetFlatValuesTest, int16ToInt8) {
  testConversion<int16_t, int8_t>(TINYINT(), {0, 1, -1, 42, -100});
}

TEST_F(GetFlatValuesTest, int32ToInt64) {
  testConversion<int32_t, int64_t>(
      BIGINT(), {0, 1, -1, 2'147'483'647, -2'147'483'648});
}

TEST_F(GetFlatValuesTest, int16ToInt64) {
  testConversion<int16_t, int64_t>(BIGINT(), {0, 1, -1, 32'767, -32'768});
}

// HUGEINT widening — int-to-int128_t conversions added for Parquet type
// widening.

TEST_F(GetFlatValuesTest, int128ToInt128) {
  testConversion<int128_t, int128_t>(DECIMAL(38, 0), {0, 1, -1, 1'000'000});
}

TEST_F(GetFlatValuesTest, int64ToInt128) {
  testConversion<int64_t, int128_t>(
      DECIMAL(38, 0),
      {0, 1, -1, 9'223'372'036'854'775'807LL, -9'223'372'036'854'775'807LL});
}

TEST_F(GetFlatValuesTest, int32ToInt128) {
  testConversion<int32_t, int128_t>(
      DECIMAL(38, 0), {0, 1, -1, 2'147'483'647, -2'147'483'648});
}

TEST_F(GetFlatValuesTest, int16ToInt128) {
  testConversion<int16_t, int128_t>(
      DECIMAL(38, 0), {0, 1, -1, 32'767, -32'768});
}

// Regression guards for existing same-domain paths affected by template
// changes.

TEST_F(GetFlatValuesTest, int64ToInt64) {
  testConversion<int64_t, int64_t>(
      BIGINT(),
      {0, 1, -1, 9'223'372'036'854'775'807LL, -9'223'372'036'854'775'807LL});
}

TEST_F(GetFlatValuesTest, int64ToInt32) {
  testConversion<int64_t, int32_t>(INTEGER(), {0, 1, -1, 42, -100});
}

TEST_F(GetFlatValuesTest, int16ToInt16) {
  testConversion<int16_t, int16_t>(SMALLINT(), {0, 1, -1, 32'767, -32'768});
}

// ByteRle conversions (int8_t source).

TEST_F(GetFlatValuesTest, int8ToInt8) {
  testConversion<int8_t, int8_t>(TINYINT(), {0, 1, -1, 127, -128});
}

TEST_F(GetFlatValuesTest, int8ToInt16) {
  testConversion<int8_t, int16_t>(SMALLINT(), {0, 1, -1, 127, -128});
}

TEST_F(GetFlatValuesTest, int8ToInt32) {
  testConversion<int8_t, int32_t>(INTEGER(), {0, 1, -1, 127, -128});
}

TEST_F(GetFlatValuesTest, int8ToInt64) {
  testConversion<int8_t, int64_t>(BIGINT(), {0, 1, -1, 127, -128});
}

// Unsigned integer conversions.

TEST_F(GetFlatValuesTest, uint8ToUint8) {
  testConversion<uint8_t, uint8_t>(TINYINT(), {0, 1, 42, 255});
}

TEST_F(GetFlatValuesTest, uint16ToUint16) {
  testConversion<uint16_t, uint16_t>(SMALLINT(), {0, 1, 42, 65'535});
}

TEST_F(GetFlatValuesTest, uint32ToUint8) {
  testConversion<uint32_t, uint8_t>(TINYINT(), {0, 1, 42, 200});
}

TEST_F(GetFlatValuesTest, uint32ToUint16) {
  testConversion<uint32_t, uint16_t>(SMALLINT(), {0, 1, 42, 60'000});
}

TEST_F(GetFlatValuesTest, uint32ToUint32) {
  testConversion<uint32_t, uint32_t>(INTEGER(), {0, 1, 42, 4'294'967'295U});
}

TEST_F(GetFlatValuesTest, uint32ToUint64) {
  testConversion<uint32_t, uint64_t>(BIGINT(), {0, 1, 42, 4'294'967'295U});
}

TEST_F(GetFlatValuesTest, uint64ToUint64) {
  testConversion<uint64_t, uint64_t>(BIGINT(), {0, 1, 42, 1'000'000'000ULL});
}

TEST_F(GetFlatValuesTest, uint64ToUint128) {
  testConversion<uint64_t, uint128_t>(
      DECIMAL(38, 0), {0, 1, 42, 1'000'000'000ULL});
}

TEST_F(GetFlatValuesTest, uint128ToUint128) {
  testConversion<uint128_t, uint128_t>(DECIMAL(38, 0), {0, 1, 42, 1'000'000});
}

// Floating-point conversions.

TEST_F(GetFlatValuesTest, floatToFloat) {
  testConversion<float, float>(REAL(), {0.0f, 1.5f, -3.14f, 1e10f});
}

TEST_F(GetFlatValuesTest, floatToDouble) {
  testConversion<float, double>(DOUBLE(), {0.0f, 1.5f, -3.14f, 1e10f});
}

TEST_F(GetFlatValuesTest, doubleToDouble) {
  testConversion<double, double>(DOUBLE(), {0.0, 1.5, -3.14, 1e100});
}

// Cross-domain upcast conversion with nulls.

TEST_F(GetFlatValuesTest, int32ToDoubleWithNulls) {
  auto reader = makeReader(DOUBLE());
  std::vector<int32_t> const data = {10, 20, 0, 40, 0};
  std::vector<bool> const nulls = {false, false, true, false, true};
  std::vector<int32_t> rowNums(data.size());
  std::iota(rowNums.begin(), rowNums.end(), 0);
  reader->setupValuesWithNulls(data, nulls, rowNums);

  RowSet const rows(rowNums.data(), rowNums.size());
  VectorPtr result;
  reader->getFlatValues<int32_t, double>(rows, &result, DOUBLE(), true);

  auto* flat = result->as<FlatVector<double>>();
  ASSERT_NE(flat, nullptr);
  ASSERT_EQ(flat->size(), data.size());
  EXPECT_DOUBLE_EQ(flat->valueAt(0), 10.0);
  EXPECT_DOUBLE_EQ(flat->valueAt(1), 20.0);
  EXPECT_TRUE(flat->isNullAt(2));
  EXPECT_DOUBLE_EQ(flat->valueAt(3), 40.0);
  EXPECT_TRUE(flat->isNullAt(4));
}

} // namespace
} // namespace facebook::velox::dwio::common
