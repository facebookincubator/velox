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

// Runtime behavior of an Iceberg geometry read as a function of the build flag. Unlike
// IcebergGeometryReadTest this file is compiled in *both* configurations, because the point is to
// pin what a VELOX_ENABLE_GEO=OFF binary does when it meets a geometry column: it must fail with an
// explicit message naming the flag, and it must never hand raw WKB back inside a GEOMETRY vector.
//
// Nothing here needs GEOS: the WKB is a hard-coded byte string and GeometryType is header-only, so
// the test links in a geospatial-free build.

#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

#include <folly/Singleton.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/types/GeometryType.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::connector::hive::iceberg {
namespace {

// Little-endian ISO WKB for POINT (10 20), written out by hand so that this file needs no WKB writer.
const std::string kPointWkb = {
    '\x01',                                                 // little endian
    '\x01', '\x00', '\x00', '\x00',                         // type 1 = Point
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x24', '\x40', // x = 10
    '\x00', '\x00', '\x00', '\x00', '\x00', '\x00', '\x34', '\x40'  // y = 20
};

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
};

#ifdef VELOX_ENABLE_PARQUET

TEST_F(IcebergGeometryBuildGateTest, geometryReadDependsOnBuildFlag) {
  const auto outputDirectory = test::TempDirectoryPath::create();
  const auto dataPath = outputDirectory->getPath();
  auto data = makeRowVector({"geom"}, {makeWkbVector()});
  auto dataSink = createDataSinkAndAppendData({data}, dataPath);
  dataSink->close();

  auto plan = exec::test::PlanBuilder()
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(ROW({"geom"}, {GEOMETRY()}))
                  .endTableScan()
                  .planNode();
  auto splits = createSplitsForDirectory(dataPath);

#ifdef VELOX_ENABLE_GEO
  // With geospatial support the value is re-encoded, so the output is *not* the WKB that is on disk.
  auto result =
      exec::test::AssertQueryBuilder(plan).splits(splits).copyResults(pool());
  ASSERT_EQ(result->size(), 1);
  auto column = result->as<RowVector>()->childAt(0);
  ASSERT_TRUE(isGeometryType(column->type()));
  DecodedVector decoded(*column);
  EXPECT_NE(decoded.valueAt<StringView>(0).str(), kPointWkb)
      << "the GEOMETRY vector still holds raw WKB";
#else
  // Without it, the read must fail and name the flag. Any successful read here would mean raw WKB
  // was exposed as a GEOMETRY vector.
  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(plan).splits(splits).copyResults(pool()),
      "requires a build with geospatial support (VELOX_ENABLE_GEO=ON)");
#endif
}

TEST_F(IcebergGeometryBuildGateTest, plainBinaryReadWorksInEitherBuild) {
  // The same file read as VARBINARY is unaffected by the flag: the bytes come back verbatim.
  const auto outputDirectory = test::TempDirectoryPath::create();
  const auto dataPath = outputDirectory->getPath();
  auto data = makeRowVector({"geom"}, {makeWkbVector()});
  auto dataSink = createDataSinkAndAppendData({data}, dataPath);
  dataSink->close();

  auto plan = exec::test::PlanBuilder()
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(ROW({"geom"}, {VARBINARY()}))
                  .endTableScan()
                  .planNode();
  auto result = exec::test::AssertQueryBuilder(plan)
                    .splits(createSplitsForDirectory(dataPath))
                    .copyResults(pool());
  ASSERT_EQ(result->size(), 1);
  DecodedVector decoded(*result->as<RowVector>()->childAt(0));
  EXPECT_EQ(decoded.valueAt<StringView>(0).str(), kPointWkb);
}

#endif // VELOX_ENABLE_PARQUET

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
