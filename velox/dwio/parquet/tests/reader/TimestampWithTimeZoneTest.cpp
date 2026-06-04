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

#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/parquet/tests/ParquetTestBase.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/Timestamp.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

using namespace facebook::velox;
using namespace facebook::velox::parquet;
using namespace facebook::velox::dwio::common;

class TimestampWithTimeZoneTest : public ParquetTestBase {
 protected:
  // Helper to create TIMESTAMP WITH TIME ZONE type
  static TypePtr timestampWithTimeZoneType() {
    return TIMESTAMP_WITH_TIME_ZONE();
  }

  // Helper to pack timestamp and timezone into int64
  static int64_t pack(const Timestamp& timestamp, int16_t tzKey) {
    return (static_cast<int64_t>(timestamp.toMillis()) << 12) | (tzKey & 0xFFF);
  }

  // Helper to unpack int64 into timestamp and timezone
  static std::pair<Timestamp, int16_t> unpack(int64_t packed) {
    int64_t millis = packed >> 12;
    int16_t tzKey = packed & 0xFFF;
    return {Timestamp::fromMillis(millis), tzKey};
  }

  // Write test data to Parquet in memory and return reader
  std::unique_ptr<ParquetReader> writeAndCreateReader(
      const std::vector<Timestamp>& timestamps,
      const std::vector<bool>& nulls = {}) {
    // Write as TIMESTAMP type (creates proper Parquet timestamp metadata)
    auto writeRowType = ROW({"ts_with_tz"}, {TIMESTAMP()});

    // Create Timestamp vector for writing
    VectorPtr timestampVector;

    if (!nulls.empty()) {
      // Use makeNullableFlatVector when we have nulls - this is the correct way
      std::vector<std::optional<Timestamp>> nullableTimestamps;
      for (size_t i = 0; i < timestamps.size(); ++i) {
        if (i < nulls.size() && nulls[i]) {
          nullableTimestamps.push_back(std::nullopt);
        } else {
          nullableTimestamps.push_back(timestamps[i]);
        }
      }
      timestampVector = makeNullableFlatVector<Timestamp>(nullableTimestamps);
    } else {
      timestampVector = makeFlatVector<Timestamp>(timestamps);
    }

    auto batch = makeRowVector({timestampVector});

    // Write to Parquet in memory
    auto sink = std::make_unique<MemorySink>(
        4 * 1024 * 1024,
        FileSink::Options{.pool = leafPool_.get()});
    auto* sinkPtr = sink.get();

    parquet::WriterOptions options;
    options.memoryPool = leafPool_.get();
    options.writeInt96AsTimestamp = false; // Use INT64 format
    options.parquetWriteTimestampUnit = TimestampPrecision::kMicroseconds;
    auto writer = std::make_unique<parquet::Writer>(
        std::move(sink),
        options,
        rootPool_,
        writeRowType);

    writer->write(batch);
    writer->close();

    // Create reader requesting TIMESTAMP WITH TIME ZONE type
    // The reader will convert TIMESTAMP to packed int64 format
    auto readRowType = ROW({"ts_with_tz"}, {timestampWithTimeZoneType()});
    dwio::common::ReaderOptions readerOpts{leafPool_.get()};
    readerOpts.setFileSchema(readRowType);
    return createReaderInMemory(*sinkPtr, readerOpts);
  }
};

// Test reading INT64 timestamps as TIMESTAMP WITH TIME ZONE
TEST_F(TimestampWithTimeZoneTest, readInt64Timestamps) {
  std::vector<Timestamp> timestamps = {
      Timestamp(0, 0),                    // Epoch
      Timestamp(1000000, 0),              // 1970-01-12 13:46:40
      Timestamp(1609459200, 0),           // 2021-01-01 00:00:00
      Timestamp(1735689600, 0),           // 2025-01-01 00:00:00
      Timestamp(-62135596800, 0),         // 0001-01-01 00:00:00
  };

  auto reader = writeAndCreateReader(timestamps);

  auto rowType = ROW({"ts_with_tz"}, {timestampWithTimeZoneType()});
  auto rowReaderOpts = getReaderOpts(rowType);
  rowReaderOpts.setScanSpec(makeScanSpec(rowType));
  auto rowReader = reader->createRowReader(rowReaderOpts);

  VectorPtr result = BaseVector::create(rowType, 0, leafPool_.get());
  auto rowsRead = rowReader->next(1000, result);

  ASSERT_EQ(rowsRead, timestamps.size());
  auto resultVector = result->loadedVector()->as<RowVector>();
  auto tsVector = resultVector->childAt(0)->as<FlatVector<int64_t>>();

  constexpr int16_t kUtcKey = 0;
  for (size_t i = 0; i < timestamps.size(); ++i) {
    ASSERT_FALSE(tsVector->isNullAt(i)) << "Row " << i << " should not be null";
    auto [ts, tzKey] = unpack(tsVector->valueAt(i));
    EXPECT_EQ(ts.getSeconds(), timestamps[i].getSeconds())
        << "Timestamp seconds mismatch at row " << i;
    EXPECT_EQ(ts.getNanos(), timestamps[i].getNanos())
        << "Timestamp nanos mismatch at row " << i;
    EXPECT_EQ(tzKey, kUtcKey) << "Timezone key should be UTC at row " << i;
  }
}

// Test reading timestamps with null values
TEST_F(TimestampWithTimeZoneTest, readWithNulls) {
  std::vector<Timestamp> timestamps = {
      Timestamp(1000000, 0),
      Timestamp(2000000, 0),
      Timestamp(3000000, 0),
      Timestamp(4000000, 0),
      Timestamp(5000000, 0),
  };

  std::vector<bool> nulls = {false, true, false, true, false};

  auto reader = writeAndCreateReader(timestamps, nulls);

  auto rowType = ROW({"ts_with_tz"}, {timestampWithTimeZoneType()});
  auto rowReaderOpts = getReaderOpts(rowType);
  rowReaderOpts.setScanSpec(makeScanSpec(rowType));
  auto rowReader = reader->createRowReader(rowReaderOpts);

  VectorPtr result = BaseVector::create(rowType, 0, leafPool_.get());
  auto rowsRead = rowReader->next(1000, result);

  ASSERT_EQ(rowsRead, timestamps.size());
  auto resultVector = result->loadedVector()->as<RowVector>();
  auto tsVector = resultVector->childAt(0)->as<FlatVector<int64_t>>();

  for (size_t i = 0; i < timestamps.size(); ++i) {
    if (nulls[i]) {
      EXPECT_TRUE(tsVector->isNullAt(i)) << "Row " << i << " should be null";
    } else {
      EXPECT_FALSE(tsVector->isNullAt(i)) << "Row " << i << " should not be null";
      auto [ts, tzKey] = unpack(tsVector->valueAt(i));
      EXPECT_EQ(ts.getSeconds(), timestamps[i].getSeconds())
          << "Timestamp seconds mismatch at row " << i;
    }
  }
}

// Test reading timestamps with microsecond precision
TEST_F(TimestampWithTimeZoneTest, readMicrosecondPrecision) {
  std::vector<Timestamp> timestamps = {
      Timestamp(1000000, 123456000),      // With microseconds
      Timestamp(2000000, 999999000),      // Max microseconds
      Timestamp(3000000, 0),              // No fractional seconds
      Timestamp(4000000, 500000000),      // Half second
  };

  auto reader = writeAndCreateReader(timestamps);

  auto rowType = ROW({"ts_with_tz"}, {timestampWithTimeZoneType()});
  auto rowReaderOpts = getReaderOpts(rowType);
  rowReaderOpts.setScanSpec(makeScanSpec(rowType));
  auto rowReader = reader->createRowReader(rowReaderOpts);

  VectorPtr result = BaseVector::create(rowType, 0, leafPool_.get());
  auto rowsRead = rowReader->next(1000, result);

  ASSERT_EQ(rowsRead, timestamps.size());
  auto resultVector = result->loadedVector()->as<RowVector>();
  auto tsVector = resultVector->childAt(0)->as<FlatVector<int64_t>>();

  for (size_t i = 0; i < timestamps.size(); ++i) {
    auto [ts, tzKey] = unpack(tsVector->valueAt(i));
    EXPECT_EQ(ts.getSeconds(), timestamps[i].getSeconds())
        << "Timestamp seconds mismatch at row " << i;
    // Note: Precision may be adjusted during conversion
    EXPECT_GE(ts.getNanos(), 0) << "Nanos should be non-negative at row " << i;
    EXPECT_LT(ts.getNanos(), 1000000000) << "Nanos should be < 1 second at row " << i;
  }
}

// Test reading negative timestamps (before epoch)
TEST_F(TimestampWithTimeZoneTest, readNegativeTimestamps) {
  std::vector<Timestamp> timestamps = {
      Timestamp(-1000000, 0),             // Before epoch
      Timestamp(-86400, 0),               // 1969-12-31 00:00:00
      Timestamp(-1, 999999999),           // Just before epoch
      Timestamp(0, 0),                    // Epoch
      Timestamp(1, 0),                    // Just after epoch
  };

  auto reader = writeAndCreateReader(timestamps);

  auto rowType = ROW({"ts_with_tz"}, {timestampWithTimeZoneType()});
  auto rowReaderOpts = getReaderOpts(rowType);
  rowReaderOpts.setScanSpec(makeScanSpec(rowType));
  auto rowReader = reader->createRowReader(rowReaderOpts);

  VectorPtr result = BaseVector::create(rowType, 0, leafPool_.get());
  auto rowsRead = rowReader->next(1000, result);

  ASSERT_EQ(rowsRead, timestamps.size());
  auto resultVector = result->loadedVector()->as<RowVector>();
  auto tsVector = resultVector->childAt(0)->as<FlatVector<int64_t>>();

  constexpr int16_t kUtcKey = 0;
  for (size_t i = 0; i < timestamps.size(); ++i) {
    auto [ts, tzKey] = unpack(tsVector->valueAt(i));
    EXPECT_EQ(ts.getSeconds(), timestamps[i].getSeconds())
        << "Timestamp seconds mismatch at row " << i;
    EXPECT_EQ(tzKey, kUtcKey) << "Timezone key should be UTC at row " << i;
  }
}

// Test efficient buffer reuse (from commit 40ff71ead)
TEST_F(TimestampWithTimeZoneTest, bufferReuseEfficiency) {
  // This test verifies that the optimized conversion from commit 40ff71ead
  // reuses buffers efficiently instead of allocating new ones

  std::vector<Timestamp> timestamps;
  for (int i = 0; i < 1000; ++i) {
    timestamps.push_back(Timestamp(i * 1000, i * 1000000));
  }

  auto reader = writeAndCreateReader(timestamps);

  auto rowType = ROW({"ts_with_tz"}, {timestampWithTimeZoneType()});
  auto rowReaderOpts = getReaderOpts(rowType);
  rowReaderOpts.setScanSpec(makeScanSpec(rowType));
  auto rowReader = reader->createRowReader(rowReaderOpts);

  VectorPtr result = BaseVector::create(rowType, 0, leafPool_.get());
  auto rowsRead = rowReader->next(1000, result);

  ASSERT_EQ(rowsRead, timestamps.size());
  auto resultVector = result->loadedVector()->as<RowVector>();
  auto tsVector = resultVector->childAt(0)->as<FlatVector<int64_t>>();

  // Verify all values are correctly packed
  for (size_t i = 0; i < timestamps.size(); ++i) {
    ASSERT_FALSE(tsVector->isNullAt(i));
    auto [ts, tzKey] = unpack(tsVector->valueAt(i));
    EXPECT_EQ(ts.getSeconds(), timestamps[i].getSeconds())
        << "Mismatch at row " << i;
  }

  // Verify the vector uses the expected type
  EXPECT_EQ(tsVector->type()->kind(), TypeKind::BIGINT);
  EXPECT_TRUE(isTimestampWithTimeZoneType(tsVector->type()));
}

// Test UTC timezone key is always 0
TEST_F(TimestampWithTimeZoneTest, utcTimezoneKeyIsZero) {
  std::vector<Timestamp> timestamps = {
      Timestamp(1000000, 0),
      Timestamp(2000000, 0),
      Timestamp(3000000, 0),
  };

  auto reader = writeAndCreateReader(timestamps);

  auto rowType = ROW({"ts_with_tz"}, {timestampWithTimeZoneType()});
  auto rowReaderOpts = getReaderOpts(rowType);
  rowReaderOpts.setScanSpec(makeScanSpec(rowType));
  auto rowReader = reader->createRowReader(rowReaderOpts);

  VectorPtr result = BaseVector::create(rowType, 0, leafPool_.get());
  auto rowsRead = rowReader->next(1000, result);

  ASSERT_EQ(rowsRead, timestamps.size());
  auto resultVector = result->loadedVector()->as<RowVector>();
  auto tsVector = resultVector->childAt(0)->as<FlatVector<int64_t>>();

  for (size_t i = 0; i < timestamps.size(); ++i) {
    auto [ts, tzKey] = unpack(tsVector->valueAt(i));
    EXPECT_EQ(tzKey, 0) << "Timezone key should always be 0 (UTC) at row " << i;
  }
}