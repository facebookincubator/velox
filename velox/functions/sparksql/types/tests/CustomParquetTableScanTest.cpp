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

#include <folly/init/Init.h>

#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/dwio/parquet/RegisterParquetReader.h" // @manual
#include "velox/dwio/parquet/writer/Writer.h" // @manual
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h" // @manual
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/sparksql/types/TimestampNTZRegistration.h"
#include "velox/functions/sparksql/types/TimestampNTZType.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

class CustomParquetTableScanTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    parquet::registerParquetReaderFactory();
    registerTimestampNTZType();
  }

  void loadData(RowVectorPtr data) {
    createDuckDbTable({data});
  }

  std::shared_ptr<connector::hive::HiveConnectorSplit> makeSplit(
      const std::string& filePath,
      const std::optional<
          std::unordered_map<std::string, std::optional<std::string>>>&
          partitionKeys = std::nullopt,
      const std::optional<std::unordered_map<std::string, std::string>>&
          infoColumns = std::nullopt) {
    return makeHiveConnectorSplits(
        filePath,
        1,
        dwio::common::FileFormat::PARQUET,
        partitionKeys,
        infoColumns)[0];
  }

  // Write data to a parquet file on specified path.
  void writeToParquetFile(
      const std::string& path,
      const std::vector<RowVectorPtr>& data,
      parquet::WriterOptions options) {
    VELOX_CHECK_GT(data.size(), 0);

    auto writeFile = std::make_unique<LocalWriteFile>(path, true, false);
    auto sink = std::make_unique<dwio::common::WriteFileSink>(
        std::move(writeFile), path);
    auto childPool =
        rootPool_->addAggregateChild("ParquetTableScanTest.Writer");
    options.memoryPool = childPool.get();

    auto writer = std::make_unique<parquet::Writer>(
        std::move(sink), options, asRowType(data[0]->type()));

    for (const auto& vector : data) {
      writer->write(vector);
    }
    writer->close();
  }

  void testTimestampNTZRead(const parquet::WriterOptions& options) {
    auto stringToTimestamp = [](std::string_view view) {
      return util::fromTimestampString(
                 view.data(),
                 view.size(),
                 util::TimestampParseMode::kPrestoCast)
          .thenOrThrow(folly::identity, [&](const Status& status) {
            VELOX_USER_FAIL("{}", status.message());
          });
    };
    std::vector<std::string_view> views = {
        "2015-06-01 19:34:56.007",
        "2015-06-02 19:34:56.12306",
        "2001-02-03 03:34:06.056",
        "1998-03-01 08:01:06.996669",
        "2022-12-23 03:56:01",
        "1980-01-24 00:23:07",
        "1999-12-08 13:39:26.123456",
        "2023-04-21 09:09:34.5",
        "2000-09-12 22:36:29",
        "2007-12-12 04:27:56.999",
    };

    std::vector<Timestamp> values;
    std::vector<int64_t> bigints;
    values.reserve(views.size());
    bigints.reserve(views.size());
    for (auto view : views) {
      auto timestamp = stringToTimestamp(view);
      values.emplace_back(timestamp);
      bigints.push_back(
          options.parquetWriteTimestampUnit == TimestampPrecision::kMicroseconds
              ? timestamp.toMicros()
              : timestamp.toMillis() * 1'000);
    }

    auto vector = makeRowVector({"t"}, {makeFlatVector<Timestamp>(values)});
    auto file = common::testutil::TempFilePath::create();
    writeToParquetFile(file->getPath(), {vector}, options);
    loadData(makeRowVector({"t"}, {makeFlatVector<int64_t>(bigints)}));

    // The values are stored as TIMESTAMPs in Parquet, and the reader will
    // convert them to TIMESTAMP_NTZ type.
    const auto plan = PlanBuilder()
                          .tableScan(
                              ROW({"t"}, {TIMESTAMP_NTZ()}),
                              {},
                              "",
                              ROW({"t"}, {TIMESTAMP()}))
                          .planNode();
    AssertQueryBuilder(plan, duckDbQueryRunner_)
        .split(makeSplit(file->getPath()))
        .assertResults("SELECT t from tmp");
  }
};

TEST_F(CustomParquetTableScanTest, timestampNTZPlainMicro) {
  parquet::WriterOptions options;
  options.writeInt96AsTimestamp = false;
  options.enableDictionary = false;
  options.parquetWriteTimestampUnit = TimestampPrecision::kMicroseconds;
  testTimestampNTZRead(options);
}

TEST_F(CustomParquetTableScanTest, timestampNTZDictionaryMicro) {
  parquet::WriterOptions options;
  options.writeInt96AsTimestamp = false;
  options.enableDictionary = true;
  options.parquetWriteTimestampUnit = TimestampPrecision::kMicroseconds;
  testTimestampNTZRead(options);
}

TEST_F(CustomParquetTableScanTest, timestampNTZPlainMilli) {
  parquet::WriterOptions options;
  options.writeInt96AsTimestamp = false;
  options.enableDictionary = false;
  options.parquetWriteTimestampUnit = TimestampPrecision::kMilliseconds;
  testTimestampNTZRead(options);
}

TEST_F(CustomParquetTableScanTest, timestampNTZDictionaryMilli) {
  parquet::WriterOptions options;
  options.writeInt96AsTimestamp = false;
  options.enableDictionary = true;
  options.parquetWriteTimestampUnit = TimestampPrecision::kMilliseconds;
  testTimestampNTZRead(options);
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv, false};
  return RUN_ALL_TESTS();
}
