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

// Runtime behavior of an Iceberg geometry or geography read as a function of
// the build flag. Unlike IcebergGeospatialReadTest this file is compiled in
// *both* configurations, because the point is to pin what a
// VELOX_ENABLE_GEO=OFF binary does when it meets a geospatial column: it must
// fail with an explicit message naming the flag, and it must never hand raw WKB
// back inside a GEOMETRY or SPHERICAL_GEOGRAPHY vector.
//
// Nothing here needs GEOS: the WKB is a hard-coded byte string and
// GeometryType and SphericalGeographyType are header-only, so the test links in
// a geospatial-free build.

#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

#include <folly/Singleton.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/iceberg/EqualityDeleteFileReader.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/functions/prestosql/types/GeometryType.h"
#include "velox/functions/prestosql/types/SphericalGeographyType.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

// Little-endian ISO WKB for POINT (10 20), written out by hand so that this
// file needs no WKB writer.
const std::string kPointWkb = {
    '\x01', // little endian
    '\x01', '\x00', '\x00', '\x00', // type 1 = Point
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x24', '\x40', // x = 10
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x34', '\x40' // y = 20
};

// Little-endian ISO WKB for POINT (30 40), a second value an equality delete
// does not match.
const std::string kOtherPointWkb = {
    '\x01', // little endian
    '\x01', '\x00', '\x00', '\x00', // type 1 = Point
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x3e', '\x40', // x = 30
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x44', '\x40' // y = 40
};

// Substring shared by every geospatial-free-build error.
constexpr std::string_view kBuildFlagMessage{
    "requires a build with geospatial support (VELOX_ENABLE_GEO=ON)"};

class IcebergGeometryBuildGateTest : public test::IcebergTestBase {
 protected:
  void SetUp() override {
    test::IcebergTestBase::SetUp();
    folly::SingletonVault::singleton()->registrationComplete();
    fileFormat_ = dwio::common::FileFormat::PARQUET;
  }

  VectorPtr makeWkbVector() {
    return makeFlatVector<StringView>(
        1,
        [&](vector_size_t /*i*/) { return StringView(kPointWkb); },
        nullptr,
        VARBINARY());
  }

  // Writes 'data' to a new Parquet file under 'directory' and returns its
  // path. The caller keeps 'directory' alive for as long as the file is read.
  // The sink is released before returning: each sink adds a writer
  // sub-pool named "part[0]" under the shared connector pool, so two live sinks
  // would collide on that name.
  std::string writeParquetFile(
      const RowVectorPtr& data,
      const std::shared_ptr<test::TempDirectoryPath>& directory) {
    auto sink = createDataSinkAndAppendData({data}, directory->getPath());
    sink->close();
    sink.reset();
    auto splits = createSplitsForDirectory(directory->getPath());
    VELOX_CHECK_EQ(splits.size(), 1);
    return std::dynamic_pointer_cast<HiveConnectorSplit>(splits[0])->filePath;
  }
};

// Runs each build-flag test with the column read as GEOMETRY() and as
// SPHERICAL_GEOGRAPHY(), which share the WKB storage and the build gate.
class IcebergGeospatialBuildGateTest
    : public IcebergGeometryBuildGateTest,
      public testing::WithParamInterface<TypePtr> {};

#ifdef VELOX_ENABLE_PARQUET

TEST_P(IcebergGeospatialBuildGateTest, readDependsOnBuildFlag) {
  const auto& type = GetParam();
  const auto outputDirectory = test::TempDirectoryPath::create();
  const auto dataPath = outputDirectory->getPath();
  auto data = makeRowVector({"geom"}, {makeWkbVector()});
  auto dataSink = createDataSinkAndAppendData({data}, dataPath);
  dataSink->close();

  auto plan = makeIcebergTableScanPlan(ROW({"geom"}, {type}));
  auto splits = createSplitsForDirectory(dataPath);

#ifdef VELOX_ENABLE_GEO
  // With geospatial support the value is re-encoded, so the output is *not* the
  // WKB that is on disk.
  auto result =
      exec::test::AssertQueryBuilder(plan).splits(splits).copyResults(pool());
  ASSERT_EQ(result->size(), 1);
  auto column = result->as<RowVector>()->childAt(0);
  ASSERT_TRUE(column->type()->equivalent(*type)) << column->type()->toString();
  DecodedVector decoded(*column);
  EXPECT_NE(decoded.valueAt<StringView>(0).str(), kPointWkb)
      << "the " << type->toString() << " vector still holds raw WKB";
#else
  // Without it, the read must fail and name the flag. Any successful read here
  // would mean raw WKB was exposed as a geospatial vector.
  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(plan).splits(splits).copyResults(pool()),
      kBuildFlagMessage);
#endif
}

TEST_P(IcebergGeospatialBuildGateTest, equalityDeleteDependsOnBuildFlag) {
  const auto& type = GetParam();
  const auto baseDirectory = test::TempDirectoryPath::create();
  const auto deleteDirectory = test::TempDirectoryPath::create();
  const auto baseFilePath = writeParquetFile(
      makeRowVector(
          {"id", "geom"},
          {makeFlatVector<int64_t>({1, 2}),
           makeFlatVector<StringView>(
               {StringView(kPointWkb), StringView(kOtherPointWkb)},
               VARBINARY())}),
      baseDirectory);
  const auto deleteFilePath = writeParquetFile(
      makeRowVector({"geom"}, {makeWkbVector()}), deleteDirectory);

  // Field id 2 is the second top-level column, "geom".
  IcebergDeleteFile equalityDelete(
      FileContent::kEqualityDeletes,
      deleteFilePath,
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(deleteFilePath),
      /*equalityFieldIds=*/{2});

  const auto tableSchema = ROW({"id", "geom"}, {BIGINT(), type});
  auto plan = makeIcebergTableScanPlan(ROW({"id"}, {BIGINT()}), tableSchema);
  auto splits = makeIcebergSplits(baseFilePath, {equalityDelete});

#ifdef VELOX_ENABLE_GEO
  // The delete key is re-encoded like the base rows, so it matches row 1.
  exec::test::AssertQueryBuilder(plan).splits(splits).assertResults(
      makeRowVector({"id"}, {makeFlatVector<int64_t>({2})}));
#else
  // The equality column is read even though it is not projected, so the read
  // must fail and name the flag. Any successful read would mean the delete was
  // applied, or silently skipped, on raw WKB.
  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(plan).splits(splits).copyResults(pool()),
      kBuildFlagMessage);
#endif
}

// A scan never reaches EqualityDeleteFileReader with a geospatial column in a
// geospatial-free build, because IcebergSplitReader::prepareSplit() rejects the
// equality column first. The reader's own guard is therefore exercised
// directly.
TEST_P(IcebergGeospatialBuildGateTest, equalityDeleteReaderDependsOnBuildFlag) {
  const auto& type = GetParam();
  const auto deleteDirectory = test::TempDirectoryPath::create();
  const auto deleteFilePath = writeParquetFile(
      makeRowVector({"geom"}, {makeWkbVector()}), deleteDirectory);
  IcebergDeleteFile equalityDelete(
      FileContent::kEqualityDeletes,
      deleteFilePath,
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(deleteFilePath),
      /*equalityFieldIds=*/{1});

  auto sessionProperties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>());
  auto hiveConfig = std::make_shared<HiveConfig>(sessionProperties);
  FileHandleFactory fileHandleFactory(
      std::make_unique<SimpleLRUCache<FileHandleKey, FileHandle>>(
          hiveConfig->numCacheFileHandles()),
      std::make_unique<FileHandleGenerator>(sessionProperties));
  auto ioStatistics = std::make_shared<io::IoStatistics>();
  auto ioStats = std::make_shared<IoStats>();
  dwio::common::RuntimeStats runtimeStats;

  auto makeReader = [&]() {
    return std::make_unique<EqualityDeleteFileReader>(
        equalityDelete,
        std::vector<std::string>{"geom"},
        std::vector<TypePtr>{type},
        /*baseFilePath=*/"",
        &fileHandleFactory,
        connectorQueryCtx_.get(),
        /*executor=*/nullptr,
        hiveConfig,
        ioStatistics,
        ioStats,
        runtimeStats,
        test::kIcebergConnectorId);
  };

#ifdef VELOX_ENABLE_GEO
  EXPECT_EQ(makeReader()->numDeleteKeys(), 1);
#else
  // Hashing raw WKB delete keys against re-encoded base rows would silently
  // drop every delete, so the reader must refuse and name the flag.
  VELOX_ASSERT_THROW(makeReader(), kBuildFlagMessage);
#endif
}

TEST_F(IcebergGeometryBuildGateTest, plainBinaryReadWorksInEitherBuild) {
  // The same file read as VARBINARY is unaffected by the flag: the bytes come
  // back verbatim.
  const auto outputDirectory = test::TempDirectoryPath::create();
  const auto dataPath = outputDirectory->getPath();
  auto data = makeRowVector({"geom"}, {makeWkbVector()});
  auto dataSink = createDataSinkAndAppendData({data}, dataPath);
  dataSink->close();

  auto plan = makeIcebergTableScanPlan(ROW({"geom"}, {VARBINARY()}));
  auto result = exec::test::AssertQueryBuilder(plan)
                    .splits(createSplitsForDirectory(dataPath))
                    .copyResults(pool());
  ASSERT_EQ(result->size(), 1);
  DecodedVector decoded(*result->as<RowVector>()->childAt(0));
  EXPECT_EQ(decoded.valueAt<StringView>(0).str(), kPointWkb);
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    IcebergGeospatialBuildGateTest,
    IcebergGeospatialBuildGateTest,
    testing::Values(GEOMETRY(), SPHERICAL_GEOGRAPHY()),
    [](const testing::TestParamInfo<TypePtr>& info) {
      return isGeometryType(info.param) ? "geometry" : "geography";
    });

#endif // VELOX_ENABLE_PARQUET

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
