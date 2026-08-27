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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/parquet/tests/ParquetTestBase.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/Filter.h"
#include "velox/type/tz/TimeZoneMap.h"

using namespace facebook::velox;
using namespace facebook::velox::parquet;

namespace {

class TimestampWithTimeZoneTest : public ParquetTestBase {
 protected:
  // Writes 'data' to an in-memory Parquet file and returns a reader over it
  // that requests 'readType'.
  std::unique_ptr<ParquetReader> writeAndCreateReader(
      const RowVectorPtr& data,
      const RowTypePtr& readType,
      TimestampPrecision writePrecision = TimestampPrecision::kMicroseconds,
      bool writeInt96 = false,
      bool utcNormalized = true) {
    ParquetWriterOptions parquetOptions;
    parquetOptions.writeInt96AsTimestamp = writeInt96;
    parquetOptions.parquetWriteTimestampUnit = writePrecision;
    // The writer derives the logical type's isAdjustedToUTC flag from whether a
    // time zone is set here, and only a UTC-normalized column can be read as
    // TIMESTAMP WITH TIME ZONE.
    if (utcNormalized) {
      parquetOptions.parquetWriteTimestampTimeZone = "UTC";
    }
    dwio::common::WriterOptions options;
    options.memoryPool = rootPool_.get();
    options.formatSpecificOptions =
        std::make_shared<ParquetWriterOptions>(parquetOptions);

    auto sink = std::make_unique<dwio::common::MemorySink>(
        4 << 20, dwio::common::FileSink::Options{.pool = leafPool_.get()});
    auto* sinkPtr = sink.get();
    auto writer = std::make_unique<parquet::Writer>(
        std::move(sink), options, asRowType(data->type()));
    writer->write(data);
    writer->close();

    auto readerOptions = makeDefaultReaderOptions();
    readerOptions.setFileSchema(readType);
    return createReaderInMemory(*sinkPtr, readerOptions);
  }

  // Reads the whole file as 'readType' and returns the single child vector.
  VectorPtr read(
      ParquetReader& reader,
      const RowTypePtr& readType,
      std::optional<TimestampPrecision> requestedPrecision = std::nullopt,
      std::unique_ptr<common::Filter> filter = nullptr) {
    auto rowReaderOpts = makeRowReaderOpts(readType);
    auto scanSpec = makeScanSpec(readType);
    if (filter != nullptr) {
      scanSpec->childByName(readType->nameOf(0))->setFilter(std::move(filter));
    }
    rowReaderOpts.setScanSpec(scanSpec);
    if (requestedPrecision.has_value()) {
      rowReaderOpts.setTimestampPrecision(*requestedPrecision);
    }
    auto rowReader = reader.createRowReader(rowReaderOpts);

    VectorPtr result = BaseVector::create(readType, 0, leafPool_.get());
    rowReader->next(1'000, result);
    return result->loadedVector()->as<RowVector>()->childAt(0);
  }

  // Splits a packed value with the canonical helpers, so the test never
  // reimplements the layout it is verifying.
  static std::pair<Timestamp, TimeZoneKey> unpackValue(int64_t packed) {
    return {unpackTimestampUtc(packed), unpackZoneKeyId(packed)};
  }

  // Asserts that 'actual' holds the UTC instants in 'expected', each stamped
  // with the UTC zone key. A nullopt expects a null row.
  static void assertPackedUtc(
      const VectorPtr& actual,
      const std::vector<std::optional<Timestamp>>& expected) {
    ASSERT_TRUE(isTimestampWithTimeZoneType(actual->type()));
    auto* values = actual->as<FlatVector<int64_t>>();
    ASSERT_NE(values, nullptr);
    ASSERT_EQ(values->size(), expected.size());

    const auto utcKey = tz::getTimeZoneID("UTC");
    for (auto i = 0; i < expected.size(); ++i) {
      SCOPED_TRACE(fmt::format("Row {}", i));
      if (!expected[i].has_value()) {
        EXPECT_TRUE(values->isNullAt(i));
        continue;
      }
      ASSERT_FALSE(values->isNullAt(i));
      const auto [timestamp, zoneKey] = unpackValue(values->valueAt(i));
      EXPECT_EQ(timestamp, *expected[i]);
      EXPECT_EQ(zoneKey, utcKey);
    }
  }

  const RowTypePtr readType_ = ROW({"ts"}, {TIMESTAMP_WITH_TIME_ZONE()});
};

TEST_F(TimestampWithTimeZoneTest, int64Micros) {
  const std::vector<Timestamp> timestamps = {
      Timestamp(0, 0),
      Timestamp(1'609'459'200, 0), // 2021-01-01 00:00:00.
      Timestamp(1'735'689'600, 123'000'000), // 2025-01-01 00:00:00.123.
      Timestamp(-86'400, 0), // 1969-12-31 00:00:00.
      Timestamp(-62'135'596'800, 0), // 0001-01-01 00:00:00.
  };

  auto reader = writeAndCreateReader(
      makeRowVector({makeFlatVector<Timestamp>(timestamps)}), readType_);
  assertPackedUtc(
      read(*reader, readType_), {timestamps.begin(), timestamps.end()});
}

TEST_F(TimestampWithTimeZoneTest, int64MillisAndNanos) {
  const std::vector<Timestamp> timestamps = {
      Timestamp(1'609'459'200, 456'000'000),
      Timestamp(-1, 999'000'000),
  };
  const auto data = makeRowVector({makeFlatVector<Timestamp>(timestamps)});

  for (auto precision :
       {TimestampPrecision::kMilliseconds, TimestampPrecision::kNanoseconds}) {
    SCOPED_TRACE(
        fmt::format("Write precision {}", static_cast<int>(precision)));
    auto reader = writeAndCreateReader(data, readType_, precision);
    assertPackedUtc(
        read(*reader, readType_), {timestamps.begin(), timestamps.end()});
  }
}

TEST_F(TimestampWithTimeZoneTest, int96) {
  const std::vector<Timestamp> timestamps = {
      Timestamp(0, 0),
      Timestamp(1'609'459'200, 789'000'000),
      Timestamp(-86'400, 0),
  };

  auto reader = writeAndCreateReader(
      makeRowVector({makeFlatVector<Timestamp>(timestamps)}),
      readType_,
      TimestampPrecision::kNanoseconds,
      /*writeInt96=*/true);
  assertPackedUtc(
      read(*reader, readType_), {timestamps.begin(), timestamps.end()});
}

TEST_F(TimestampWithTimeZoneTest, nulls) {
  const std::vector<std::optional<Timestamp>> timestamps = {
      Timestamp(1'000'000, 0),
      std::nullopt,
      Timestamp(2'000'000, 0),
      std::nullopt,
      std::nullopt,
      Timestamp(3'000'000, 500'000'000),
  };

  auto reader = writeAndCreateReader(
      makeRowVector({makeNullableFlatVector<Timestamp>(timestamps)}),
      readType_);
  assertPackedUtc(read(*reader, readType_), timestamps);
}

TEST_F(TimestampWithTimeZoneTest, allNulls) {
  const std::vector<std::optional<Timestamp>> timestamps(4, std::nullopt);

  auto reader = writeAndCreateReader(
      makeRowVector({makeNullableFlatVector<Timestamp>(timestamps)}),
      readType_);
  auto values = read(*reader, readType_);
  ASSERT_TRUE(isTimestampWithTimeZoneType(values->type()));
  ASSERT_EQ(values->size(), timestamps.size());
  for (auto i = 0; i < values->size(); ++i) {
    EXPECT_TRUE(values->isNullAt(i)) << "Row " << i;
  }
}

// TIMESTAMP WITH TIME ZONE holds milliseconds, so anything finer in the file
// is truncated toward negative infinity rather than rounded.
TEST_F(TimestampWithTimeZoneTest, subMillisecondTruncation) {
  const std::vector<Timestamp> timestamps = {
      Timestamp(1, 123'456'000), // 1.123456 -> 1.123.
      Timestamp(1, 123'999'000), // 1.123999 -> 1.123.
      Timestamp(-1, 999'999'000), // -0.000001 -> -0.001.
      Timestamp(-2, 999'999'000), // -1.000001 -> -1.001.
  };
  const std::vector<std::optional<Timestamp>> expected = {
      Timestamp(1, 123'000'000),
      Timestamp(1, 123'000'000),
      Timestamp(-1, 999'000'000),
      Timestamp(-2, 999'000'000),
  };

  auto reader = writeAndCreateReader(
      makeRowVector({makeFlatVector<Timestamp>(timestamps)}), readType_);
  assertPackedUtc(read(*reader, readType_), expected);
}

// The output layout is fixed at milliseconds, so it must not vary with the
// session's requested timestamp precision.
TEST_F(TimestampWithTimeZoneTest, ignoresRequestedPrecision) {
  const std::vector<Timestamp> timestamps = {Timestamp(1, 123'456'000)};
  const std::vector<std::optional<Timestamp>> expected = {
      Timestamp(1, 123'000'000)};
  const auto data = makeRowVector({makeFlatVector<Timestamp>(timestamps)});

  for (auto precision :
       {TimestampPrecision::kMilliseconds,
        TimestampPrecision::kMicroseconds,
        TimestampPrecision::kNanoseconds}) {
    SCOPED_TRACE(
        fmt::format("Requested precision {}", static_cast<int>(precision)));
    auto reader = writeAndCreateReader(data, readType_);
    assertPackedUtc(read(*reader, readType_, precision), expected);
  }
}

// A filter on a TIMESTAMP WITH TIME ZONE column is built from its physical
// BIGINT kind, so its bounds are in the packed domain and cannot be applied to
// raw file values. The reader must say so rather than produce wrong rows.
TEST_F(TimestampWithTimeZoneTest, valueFilterIsRejected) {
  const auto data =
      makeRowVector({makeFlatVector<Timestamp>({Timestamp(0, 0)})});
  auto reader = writeAndCreateReader(data, readType_);

  VELOX_ASSERT_THROW(
      read(
          *reader,
          readType_,
          std::nullopt,
          std::make_unique<common::BigintRange>(0, 100, false)),
      "Filter pushdown on TIMESTAMP WITH TIME ZONE is not supported");
}

// A filter installed on the ScanSpec after the reader was built, as
// FileDataSource::addDynamicFilter does for join-pushed filters, must be
// rejected too.
TEST_F(TimestampWithTimeZoneTest, dynamicValueFilterIsRejected) {
  const auto data =
      makeRowVector({makeFlatVector<Timestamp>({Timestamp(0, 0)})});
  auto reader = writeAndCreateReader(data, readType_);

  auto rowReaderOpts = makeRowReaderOpts(readType_);
  auto scanSpec = makeScanSpec(readType_);
  rowReaderOpts.setScanSpec(scanSpec);
  auto rowReader = reader->createRowReader(rowReaderOpts);

  // The reader is already constructed at this point.
  scanSpec->childByName(readType_->nameOf(0))
      ->setFilter(std::make_unique<common::BigintRange>(0, 100, false));
  scanSpec->resetCachedValues(true);

  VectorPtr result = BaseVector::create(readType_, 0, leafPool_.get());
  VELOX_ASSERT_THROW(
      rowReader->next(1'000, result),
      "Filter pushdown on TIMESTAMP WITH TIME ZONE is not supported");
}

// A value beyond what the 52-bit millis field can hold is reported, naming the
// offending value, rather than silently wrapping.
TEST_F(TimestampWithTimeZoneTest, outOfRangeMillisIsRejected) {
  // 3e12 seconds is 3e18 micros, which fits an int64 file value, but 3e15
  // millis exceeds kMaxMillisUtc.
  const auto data = makeRowVector(
      {makeFlatVector<Timestamp>({Timestamp(3'000'000'000'000, 0)})});
  auto reader = writeAndCreateReader(data, readType_);

  VELOX_ASSERT_USER_THROW(
      read(*reader, readType_),
      "Timestamp is out of the range TIMESTAMP WITH TIME ZONE can represent: 3000000000000000 ms");
}

TEST_F(TimestampWithTimeZoneTest, nullFiltersArePushedDown) {
  const std::vector<std::optional<Timestamp>> timestamps = {
      Timestamp(1'000'000, 0), std::nullopt, Timestamp(2'000'000, 0)};
  const auto data =
      makeRowVector({makeNullableFlatVector<Timestamp>(timestamps)});

  auto notNullReader = writeAndCreateReader(data, readType_);
  assertPackedUtc(
      read(
          *notNullReader,
          readType_,
          std::nullopt,
          std::make_unique<common::IsNotNull>()),
      {Timestamp(1'000'000, 0), Timestamp(2'000'000, 0)});

  auto nullReader = writeAndCreateReader(data, readType_);
  assertPackedUtc(
      read(
          *nullReader,
          readType_,
          std::nullopt,
          std::make_unique<common::IsNull>()),
      {std::nullopt});
}

// A column the file does not declare UTC normalized holds wall clock readings
// in a zone the file does not record, so there is no instant to pack.
TEST_F(TimestampWithTimeZoneTest, notUtcNormalizedIsRejected) {
  const auto data =
      makeRowVector({makeFlatVector<Timestamp>({Timestamp(0, 0)})});

  auto reader = writeAndCreateReader(
      data,
      readType_,
      TimestampPrecision::kMicroseconds,
      /*writeInt96=*/false,
      /*utcNormalized=*/false);
  VELOX_ASSERT_USER_THROW(
      read(*reader, readType_),
      "Cannot read a Parquet timestamp that is not UTC-normalized as TIMESTAMP WITH TIME ZONE");

  // The same column still reads as plain TIMESTAMP.
  const auto timestampType = ROW({"ts"}, {TIMESTAMP()});
  auto timestampReader = writeAndCreateReader(
      data,
      timestampType,
      TimestampPrecision::kMicroseconds,
      /*writeInt96=*/false,
      /*utcNormalized=*/false);
  auto values = read(*timestampReader, timestampType);
  ASSERT_EQ(values->type()->kind(), TypeKind::TIMESTAMP);
  EXPECT_EQ(values->as<FlatVector<Timestamp>>()->valueAt(0), Timestamp(0, 0));
}

// A plain, unannotated INT64 column read as TIMESTAMP WITH TIME ZONE keeps
// passing its values through as already-packed values. Guards against routing
// this case into the timestamp reader, which has no file precision to work
// with.
TEST_F(TimestampWithTimeZoneTest, unannotatedBigintPassesThrough) {
  const auto utcKey = tz::getTimeZoneID("UTC");
  const std::vector<int64_t> packed = {
      pack(0, utcKey),
      pack(1'609'459'200'000, utcKey),
      pack(-86'400'000, tz::getTimeZoneID("America/New_York")),
  };

  auto reader = writeAndCreateReader(
      makeRowVector({makeFlatVector<int64_t>(packed)}), readType_);
  auto values = read(*reader, readType_);

  ASSERT_TRUE(isTimestampWithTimeZoneType(values->type()));
  auto* flat = values->as<FlatVector<int64_t>>();
  ASSERT_NE(flat, nullptr);
  EXPECT_THAT(
      std::vector<int64_t>(flat->rawValues(), flat->rawValues() + flat->size()),
      ::testing::ElementsAreArray(packed));
}

} // namespace
