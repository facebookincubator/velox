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

#include "velox/connectors/hive/iceberg/tests/IcebergTestBase.h"

#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>

#include <folly/Singleton.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/geospatial/GeometrySerde.h"
#include "velox/connectors/hive/iceberg/IcebergGeospatialConverter.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/types/GeometryRegistration.h"
#include "velox/functions/prestosql/types/GeometryType.h"
#include "velox/functions/prestosql/types/SphericalGeographyRegistration.h"
#include "velox/functions/prestosql/types/SphericalGeographyType.h"
#include "velox/vector/DecodedVector.h"

#define USE_UNSTABLE_GEOS_CPP_API 1
#include <geos/io/WKBWriter.h>
#include <geos/io/WKTReader.h>

namespace facebook::velox::connector::hive::iceberg {
namespace {

using TempFilePath = common::testutil::TempFilePath;

// Little-endian WKB scalar appenders, used to hand-build payloads whose nested
// headers no writer would emit.
void appendUint32Le(std::string& out, uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    out.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
  }
}

void appendDoubleLe(std::string& out, double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<char>((bits >> (8 * i)) & 0xFF));
  }
}

// Every primary geometry kind plus a collection.
const std::vector<std::string> kAllKinds = {
    "POINT (10 20)",
    "MULTIPOINT ((0 0), (10 20), (30 40))",
    "LINESTRING (0 0, 10 10, 20 20)",
    "MULTILINESTRING ((0 0, 5 5), (10 10, 20 20))",
    "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
    "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((5 5, 9 5, 9 9, 5 9, 5 5)))",
    "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (0 0, 1 1))"};

const std::vector<std::string> kEmptyKinds = {
    "POINT EMPTY",
    "LINESTRING EMPTY",
    "POLYGON EMPTY",
    "MULTIPOINT EMPTY",
    "MULTILINESTRING EMPTY",
    "MULTIPOLYGON EMPTY",
    "GEOMETRYCOLLECTION EMPTY"};

class IcebergGeospatialReadTest : public test::IcebergTestBase {
 protected:
  void SetUp() override {
    test::IcebergTestBase::SetUp();
    folly::SingletonVault::singleton()->registrationComplete();
    registerGeometryType();
    registerSphericalGeographyType();
    fileFormat_ = dwio::common::FileFormat::PARQUET;
  }

  // ISO WKB, as an Iceberg `geometry` column stores it on disk.
  static std::string toWkb(const std::string& wkt) {
    geos::io::WKTReader wktReader;
    geos::io::WKBWriter wkbWriter;
    std::ostringstream out;
    wkbWriter.write(*wktReader.read(wkt), out);
    return out.str();
  }

  // Velox's internal geospatial encoding, i.e. what a GEOMETRY or
  // SPHERICAL_GEOGRAPHY vector must hold.
  static std::string toVeloxGeospatial(const std::string& wkt) {
    geos::io::WKTReader wktReader;
    std::string out;
    common::geospatial::GeometrySerializer::serialize(
        *wktReader.read(wkt), out);
    return out;
  }

  // A five-byte WKB prefix carrying only the type word, enough to exercise the
  // header validation that runs before any coordinate is read.
  static std::string typeCodeOnlyWkb(uint32_t typeCode) {
    std::string bytes(5, '\0');
    bytes[0] = 1;
    bytes[1] = static_cast<char>(typeCode & 0xFF);
    bytes[2] = static_cast<char>((typeCode >> 8) & 0xFF);
    bytes[3] = static_cast<char>((typeCode >> 16) & 0xFF);
    bytes[4] = static_cast<char>((typeCode >> 24) & 0xFF);
    return bytes;
  }

  // ISO WKB for a little-endian XY point with arbitrary ordinates.
  static std::string pointWkb(double x, double y) {
    std::string bytes;
    bytes.push_back(1);
    appendUint32Le(bytes, 1);
    appendDoubleLe(bytes, x);
    appendDoubleLe(bytes, y);
    return bytes;
  }

  VectorPtr makeVarbinaryVector(
      const std::vector<std::optional<std::string>>& values) {
    return makeFlatVector<StringView>(
        values.size(),
        [&](vector_size_t i) {
          return values[i].has_value() ? StringView(*values[i]) : StringView();
        },
        [&](vector_size_t i) { return !values[i].has_value(); },
        VARBINARY());
  }

  VectorPtr makeGeometryVector(
      const std::vector<std::optional<std::string>>& values) {
    return makeGeospatialVector(values, GEOMETRY());
  }

  VectorPtr makeGeographyVector(
      const std::vector<std::optional<std::string>>& values) {
    return makeGeospatialVector(values, SPHERICAL_GEOGRAPHY());
  }

  VectorPtr makeGeospatialVector(
      const std::vector<std::optional<std::string>>& values,
      const TypePtr& type) {
    return makeFlatVector<StringView>(
        values.size(),
        [&](vector_size_t i) {
          return values[i].has_value() ? StringView(*values[i]) : StringView();
        },
        [&](vector_size_t i) { return !values[i].has_value(); },
        type);
  }

  // Writes 'vectors' as Iceberg Parquet data files and scans them back with
  // 'outputType'.
  void assertScan(
      const std::vector<RowVectorPtr>& vectors,
      const RowTypePtr& outputType,
      const std::vector<RowVectorPtr>& expected) {
    const auto outputDirectory = test::TempDirectoryPath::create();
    const auto dataPath = outputDirectory->getPath();
    const auto dataSink = createDataSinkAndAppendData(vectors, dataPath);
    dataSink->close();

    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(outputType)
                    .endTableScan()
                    .planNode();
    exec::test::AssertQueryBuilder(plan)
        .splits(createSplitsForDirectory(dataPath))
        .assertResults(expected);
  }

  // Applies an equality delete on a 'type' column, GEOMETRY or
  // SPHERICALGEOGRAPHY, whose delete file holds one of two base values as ISO
  // WKB, and asserts that exactly the matching base row is deleted.
  void assertEqualityDeleteOnGeospatialColumn(const TypePtr& type) {
    const std::string matchWkt = "POINT (10 20)";
    const std::string keepWkt = "LINESTRING (0 0, 1 1)";

    // Base file: written as binary WKB and read back as 'type', which is how a
    // real Iceberg geometry or geography data file looks on disk.
    auto baseDirectory = test::TempDirectoryPath::create();
    auto baseData = makeRowVector(
        {"id", "geom"},
        {makeFlatVector<int64_t>({1, 2}),
         makeVarbinaryVector({toWkb(matchWkt), toWkb(keepWkt)})});
    auto baseSink =
        createDataSinkAndAppendData({baseData}, baseDirectory->getPath());
    baseSink->close();
    // Release the sink before creating the next one: each sink adds a writer
    // sub-pool named "part[0]" under the shared connector pool, so two live
    // sinks would collide on that name.
    baseSink.reset();
    auto baseSplits = createSplitsForDirectory(baseDirectory->getPath());
    ASSERT_EQ(baseSplits.size(), 1);
    const auto baseFilePath =
        std::dynamic_pointer_cast<HiveConnectorSplit>(baseSplits[0])->filePath;

    // Equality-delete file: one row carrying G_match as ISO WKB, exactly as
    // another engine would have written it. Written as Parquet like the base
    // file, so this test stays inside the PR's Parquet-only geometry scope.
    auto deleteDirectory = test::TempDirectoryPath::create();
    auto deleteData =
        makeRowVector({"geom"}, {makeVarbinaryVector({toWkb(matchWkt)})});
    auto deleteSink =
        createDataSinkAndAppendData({deleteData}, deleteDirectory->getPath());
    deleteSink->close();
    deleteSink.reset();
    auto deleteSplits = createSplitsForDirectory(deleteDirectory->getPath());
    ASSERT_EQ(deleteSplits.size(), 1);
    const auto deleteFilePath =
        std::dynamic_pointer_cast<HiveConnectorSplit>(deleteSplits[0])
            ->filePath;

    // Field id 2 is the second top-level column, "geom".
    IcebergDeleteFile equalityDelete(
        FileContent::kEqualityDeletes,
        deleteFilePath,
        dwio::common::FileFormat::PARQUET,
        1,
        getFileSize(deleteFilePath),
        /*equalityFieldIds=*/{2});

    const auto tableSchema = ROW({"id", "geom"}, {BIGINT(), type});
    auto plan = exec::test::PlanBuilder()
                    .startTableScan(test::kIcebergConnectorId)
                    .outputType(tableSchema)
                    .dataColumns(tableSchema)
                    .endTableScan()
                    .planNode();

    // G_match is deleted; G_keep survives. Without the delete-side conversion
    // the delete key hashes as raw WKB while the base row hashes as internal
    // bytes, nothing matches, and both rows come back.
    auto expected = makeRowVector(
        {"id", "geom"},
        {makeFlatVector<int64_t>({2}),
         makeGeospatialVector({toVeloxGeospatial(keepWkt)}, type)});

    exec::test::AssertQueryBuilder(plan)
        .splits(makeIcebergSplits(baseFilePath, {equalityDelete}))
        .assertResults(expected);
  }
};

// ---------------------------------------------------------------------------
// Vector-level tests for the converter the Iceberg connector owns.
// ---------------------------------------------------------------------------

TEST_F(IcebergGeospatialReadTest, containsGeospatialDetection) {
  // Nothing but an actual GEOMETRY or SPHERICALGEOGRAPHY may switch the
  // conversion on.
  EXPECT_FALSE(containsGeospatial(VARBINARY()));
  EXPECT_FALSE(containsGeospatial(VARCHAR()));
  EXPECT_FALSE(containsGeospatial(ROW({"a", "b"}, {BIGINT(), VARBINARY()})));
  EXPECT_FALSE(containsGeospatial(ARRAY(VARBINARY())));
  EXPECT_FALSE(containsGeospatial(MAP(VARCHAR(), VARBINARY())));

  for (const auto& type :
       {TypePtr(GEOMETRY()), TypePtr(SPHERICAL_GEOGRAPHY())}) {
    EXPECT_TRUE(containsGeospatial(type)) << type->toString();
    EXPECT_TRUE(containsGeospatial(ROW({"a", "g"}, {BIGINT(), type})));
    EXPECT_TRUE(containsGeospatial(ARRAY(type)));
    EXPECT_TRUE(containsGeospatial(MAP(VARCHAR(), type)));
    EXPECT_TRUE(containsGeospatial(ARRAY(ROW({"g"}, {type}))));
  }
}

TEST_F(IcebergGeospatialReadTest, flatVectorAllGeometryKinds) {
  std::vector<std::optional<std::string>> wkb;
  std::vector<std::optional<std::string>> expected;
  for (const auto& wkt : kAllKinds) {
    wkb.emplace_back(toWkb(wkt));
    expected.emplace_back(toVeloxGeospatial(wkt));
  }
  for (const auto& wkt : kEmptyKinds) {
    wkb.emplace_back(toWkb(wkt));
    expected.emplace_back(toVeloxGeospatial(wkt));
  }
  // Nulls interleaved at both ends.
  wkb.emplace_back(std::nullopt);
  expected.emplace_back(std::nullopt);

  auto input = makeVarbinaryVector(wkb);
  auto converted = convertIcebergGeospatial(input, GEOMETRY(), pool(), "geom");

  ASSERT_TRUE(isGeometryType(converted->type()));
  velox::test::assertEqualVectors(makeGeometryVector(expected), converted);
}

// Geography shares the geometry encoding, so every kind converts to the same
// bytes, in a SPHERICALGEOGRAPHY vector. All the shapes lie within longitude
// and latitude ranges.
TEST_F(IcebergGeospatialReadTest, flatVectorAllGeographyKinds) {
  std::vector<std::optional<std::string>> wkb;
  std::vector<std::optional<std::string>> expected;
  for (const auto& wkt : kAllKinds) {
    wkb.emplace_back(toWkb(wkt));
    expected.emplace_back(toVeloxGeospatial(wkt));
  }
  for (const auto& wkt : kEmptyKinds) {
    wkb.emplace_back(toWkb(wkt));
    expected.emplace_back(toVeloxGeospatial(wkt));
  }
  wkb.emplace_back(std::nullopt);
  expected.emplace_back(std::nullopt);

  auto input = makeVarbinaryVector(wkb);
  auto converted =
      convertIcebergGeospatial(input, SPHERICAL_GEOGRAPHY(), pool(), "geog");

  ASSERT_TRUE(isSphericalGeographyType(converted->type()));
  velox::test::assertEqualVectors(makeGeographyVector(expected), converted);
}

TEST_F(IcebergGeospatialReadTest, dictionaryVectorIsConvertedOncePerEntry) {
  const std::vector<std::string> distinctWkt = {
      "POINT (1 2)", "LINESTRING (0 0, 1 1)", "POLYGON ((0 0, 1 0, 1 1, 0 0))"};
  std::vector<std::optional<std::string>> distinctWkb;
  for (const auto& wkt : distinctWkt) {
    distinctWkb.emplace_back(toWkb(wkt));
  }
  auto base = makeVarbinaryVector(distinctWkb);
  // Keep a copy of the dictionary bytes so we can prove they were not mutated
  // in place.
  std::vector<std::string> baseBytesBefore;
  for (vector_size_t i = 0; i < base->size(); ++i) {
    baseBytesBefore.emplace_back(
        base->asFlatVector<StringView>()->valueAt(i).str());
  }

  constexpr vector_size_t kSize = 10;
  auto indices = makeIndices(kSize, [](vector_size_t i) { return i % 3; });
  auto nulls = makeNulls(kSize, [](vector_size_t i) { return i % 5 == 4; });
  auto dictionary = BaseVector::wrapInDictionary(nulls, indices, kSize, base);
  ASSERT_EQ(dictionary->encoding(), VectorEncoding::Simple::DICTIONARY);

  auto converted =
      convertIcebergGeospatial(dictionary, GEOMETRY(), pool(), "geom");

  // The dictionary wrapping is preserved, so a repeated value is parsed once
  // per dictionary entry rather than once per row.
  ASSERT_EQ(converted->encoding(), VectorEncoding::Simple::DICTIONARY);
  ASSERT_EQ(converted->valueVector()->size(), base->size());
  ASSERT_TRUE(isGeometryType(converted->type()));

  // The shared source dictionary is untouched.
  for (vector_size_t i = 0; i < base->size(); ++i) {
    EXPECT_EQ(
        base->asFlatVector<StringView>()->valueAt(i).str(), baseBytesBefore[i]);
  }

  std::vector<std::optional<std::string>> expected;
  for (vector_size_t i = 0; i < kSize; ++i) {
    if (i % 5 == 4) {
      expected.emplace_back(std::nullopt);
    } else {
      expected.emplace_back(toVeloxGeospatial(distinctWkt[i % 3]));
    }
  }
  velox::test::assertEqualVectors(makeGeometryVector(expected), converted);
}

TEST_F(IcebergGeospatialReadTest, nullConstantVector) {
  auto input = BaseVector::createNullConstant(VARBINARY(), 5, pool());
  auto converted = convertIcebergGeospatial(input, GEOMETRY(), pool(), "geom");
  ASSERT_TRUE(isGeometryType(converted->type()));
  ASSERT_EQ(converted->size(), 5);
  for (vector_size_t i = 0; i < 5; ++i) {
    EXPECT_TRUE(converted->isNullAt(i));
  }
}

// A non-null constant keeps its encoding: the value is parsed once and
// re-wrapped, rather than flattened and re-parsed per row. Such a vector does
// not arise from a scan today, but preserving the encoding keeps the converter
// correct and O(1) if a scan later emits CONSTANT for a uniform-value column.
TEST_F(IcebergGeospatialReadTest, nonNullConstantVectorPreservesEncoding) {
  const std::string wkt = "POINT (10 20)";
  auto value = makeVarbinaryVector({toWkb(wkt)});
  auto input = BaseVector::wrapInConstant(5, 0, value);
  ASSERT_EQ(input->encoding(), VectorEncoding::Simple::CONSTANT);

  auto converted = convertIcebergGeospatial(input, GEOMETRY(), pool(), "geom");

  EXPECT_EQ(converted->encoding(), VectorEncoding::Simple::CONSTANT);
  EXPECT_TRUE(isGeometryType(converted->type()));
  ASSERT_EQ(converted->size(), 5);
  // Constant encoding is itself the "parsed once" guarantee: the vector stores
  // a single value and every position resolves to it. For a scalar geometry,
  // ConstantVector copies that value into its own buffer, so there is no
  // backing vector left to inspect.
  for (vector_size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(converted->wrappedIndex(i), 0);
  }

  velox::test::assertEqualVectors(
      BaseVector::wrapInConstant(
          5, 0, makeGeometryVector({toVeloxGeospatial(wkt)})),
      converted);
}

// The converter must not parse a value that no selected row can reach.
TEST_F(IcebergGeospatialReadTest, constantVectorWithEmptySelectionIsNotParsed) {
  // Deliberately invalid WKB: if the value were parsed, this would throw.
  auto value = makeVarbinaryVector({std::string("\x01\x02\x03", 3)});
  auto input = BaseVector::wrapInConstant(4, 0, value);

  SelectivityVector noRows(4, false);
  auto converted =
      convertIcebergGeospatial(input, GEOMETRY(), noRows, pool(), "geom");

  EXPECT_TRUE(isGeometryType(converted->type()));
  ASSERT_EQ(converted->size(), 4);
  for (vector_size_t i = 0; i < 4; ++i) {
    EXPECT_TRUE(converted->isNullAt(i));
  }
}

// A partially selected constant still converts its single value once; the
// constant result carries that value at every position, which is what CONSTANT
// encoding means.
TEST_F(IcebergGeospatialReadTest, constantVectorWithPartialSelection) {
  const std::string wkt = "LINESTRING (0 0, 10 10, 20 20)";
  auto value = makeVarbinaryVector({toWkb(wkt)});
  auto input = BaseVector::wrapInConstant(4, 0, value);

  SelectivityVector someRows(4, false);
  someRows.setValid(1, true);
  someRows.setValid(2, true);
  someRows.updateBounds();
  auto converted =
      convertIcebergGeospatial(input, GEOMETRY(), someRows, pool(), "geom");

  EXPECT_EQ(converted->encoding(), VectorEncoding::Simple::CONSTANT);
  EXPECT_TRUE(isGeometryType(converted->type()));
  EXPECT_EQ(converted->wrappedIndex(3), 0);
  velox::test::assertEqualVectors(
      BaseVector::wrapInConstant(
          4, 0, makeGeometryVector({toVeloxGeospatial(wkt)})),
      converted);
}

// A constant complex value goes through the same ROW recursion as a flat one,
// so the geometry leaf is converted and the outer CONSTANT encoding is
// preserved.
TEST_F(IcebergGeospatialReadTest, constantRowWithGeometryField) {
  const std::string wkt = "POINT (3 4)";
  auto row = makeRowVector(
      {"id", "geom"},
      {makeFlatVector<int64_t>({7}), makeVarbinaryVector({toWkb(wkt)})});
  auto input = BaseVector::wrapInConstant(3, 0, row);
  auto targetType = ROW({"id", "geom"}, {BIGINT(), GEOMETRY()});

  auto converted =
      convertIcebergGeospatial(input, targetType, pool(), "nested");

  EXPECT_EQ(converted->encoding(), VectorEncoding::Simple::CONSTANT);
  ASSERT_TRUE(converted->type()->equivalent(*targetType));
  ASSERT_EQ(converted->size(), 3);
  // A complex constant retains its one-row base, so the single conversion is
  // directly observable here.
  ASSERT_EQ(converted->valueVector()->size(), 1);
  velox::test::assertEqualVectors(
      BaseVector::wrapInConstant(
          3,
          0,
          makeRowVector(
              {"id", "geom"},
              {makeFlatVector<int64_t>({7}),
               makeGeometryVector({toVeloxGeospatial(wkt)})})),
      converted);
}

// An Iceberg equality-delete file stores its geometry as the ISO WKB the spec
// mandates, while the base rows it is probed against have already been
// re-encoded into Velox's internal geometry encoding. Both sides have to be
// hashed in the same logical encoding or the delete silently never matches.
TEST_F(IcebergGeospatialReadTest, equalityDeleteOnGeometryColumn) {
  assertEqualityDeleteOnGeospatialColumn(GEOMETRY());
}

// Geography values take the same WKB -> internal re-encoding on both sides, so
// the delete key and the base row hash alike.
TEST_F(IcebergGeospatialReadTest, equalityDeleteOnGeographyColumn) {
  assertEqualityDeleteOnGeospatialColumn(SPHERICAL_GEOGRAPHY());
}

// A hash join on a GEOMETRY key must return the matching Iceberg row even with
// string/binary dynamic-filter pushdown enabled. The build side holds Velox
// internal geometry bytes while the Iceberg file holds ISO WKB, so a
// BytesValues filter built from the build side and evaluated by the scan
// against the file bytes would drop the row. No filter may be produced for
// this custom VARBINARY-backed type, leaving the join to do the matching.
TEST_F(IcebergGeospatialReadTest, geometryHashJoinWithDynamicFilterPushdown) {
  const std::string matchWkt = "POINT (10 20)";
  const std::string otherWkt = "LINESTRING (0 0, 1 1)";

  auto directory = test::TempDirectoryPath::create();
  auto data = makeRowVector(
      {"id", "geom"},
      {makeFlatVector<int64_t>({1, 2}),
       makeVarbinaryVector({toWkb(matchWkt), toWkb(otherWkt)})});
  auto sink = createDataSinkAndAppendData({data}, directory->getPath());
  sink->close();

  // Build side: the same shape, already in Velox's internal encoding, which is
  // what a GEOMETRY vector anywhere in the plan carries.
  auto buildData = makeRowVector(
      {"bgeom"}, {makeGeometryVector({toVeloxGeospatial(matchWkt)})});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId scanId;
  auto plan = exec::test::PlanBuilder(planNodeIdGenerator)
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(ROW({"id", "geom"}, {BIGINT(), GEOMETRY()}))
                  .endTableScan()
                  .capturePlanNodeId(scanId)
                  .hashJoin(
                      {"geom"},
                      {"bgeom"},
                      exec::test::PlanBuilder(planNodeIdGenerator)
                          .values({buildData})
                          .planNode(),
                      /*filter=*/"",
                      {"id"})
                  .planNode();

  auto expected = makeRowVector({"id"}, {makeFlatVector<int64_t>({1})});

  std::shared_ptr<exec::Task> task;
  auto result =
      exec::test::AssertQueryBuilder(plan)
          .config(
              core::QueryConfig::kHashProbeStringDynamicFilterPushdownEnabled,
              "true")
          .splits(scanId, createSplitsForDirectory(directory->getPath()))
          .copyResults(pool(), task);

  velox::test::assertEqualVectors(expected, result);

  // No dynamic filter may have been produced for the GEOMETRY key: the scan
  // would have evaluated it against the file's WKB, and if the join were also
  // replaced by that filter the matching row would be dropped outright.
  for (const auto& pipeline : task->taskStats().pipelineStats) {
    for (const auto& op : pipeline.operatorStats) {
      EXPECT_EQ(op.runtimeStats.count("dynamicFiltersProduced"), 0)
          << op.operatorType;
      EXPECT_EQ(op.runtimeStats.count("replacedWithDynamicFilterRows"), 0)
          << op.operatorType;
    }
  }
}

TEST_F(IcebergGeospatialReadTest, invalidWkbErrorNamesColumnPath) {
  auto shortValue = makeVarbinaryVector({std::string("\x01\x02\x03", 3)});
  VELOX_ASSERT_THROW(
      convertIcebergGeospatial(shortValue, GEOMETRY(), pool(), "shapes.geom"),
      "Iceberg geometry column 'shapes.geom'");

  auto badByteOrder =
      makeVarbinaryVector({typeCodeOnlyWkb(1).replace(0, 1, "\x07")});
  VELOX_ASSERT_THROW(
      convertIcebergGeospatial(badByteOrder, GEOMETRY(), pool(), "geom"),
      "unknown byte order marker");

  // A valid header with a truncated body: GEOS rejects it and the message still
  // names the column.
  auto truncated =
      makeVarbinaryVector({std::string("\x01\x01\x00\x00\x00\x00\x00", 7)});
  VELOX_ASSERT_THROW(
      convertIcebergGeospatial(truncated, GEOMETRY(), pool(), "geom"),
      "Iceberg geometry column 'geom'");
}

TEST_F(IcebergGeospatialReadTest, zmAndEwkbAreRejectedNotFlattened) {
  auto expectFailure = [&](uint32_t typeCode, const std::string& message) {
    auto input = makeVarbinaryVector({typeCodeOnlyWkb(typeCode)});
    VELOX_ASSERT_THROW(
        convertIcebergGeospatial(input, GEOMETRY(), pool(), "geom"), message);
  };

  expectFailure(1001, "contains Z coordinates");
  expectFailure(2001, "contains M coordinates");
  expectFailure(3001, "contains Z and M coordinates");
  expectFailure(0x2000'0001, "extended WKB (EWKB)");
  expectFailure(0x8000'0001, "extended WKB (EWKB)");
  // ISO code 15 is PolyhedralSurface, which Velox GEOMETRY cannot represent.
  expectFailure(15, "unsupported WKB geometry type code 15");
}

// Iceberg geography coordinates are WGS84 longitude/latitude. A file written by
// another engine may hold values Presto's to_spherical_geography would reject,
// and those must fail the read rather than surface as a SPHERICALGEOGRAPHY.
TEST_F(IcebergGeospatialReadTest, geographyCoordinatesAreRangeChecked) {
  auto expectFailure = [&](const std::string& wkb, const std::string& message) {
    auto input = makeVarbinaryVector({wkb});
    VELOX_ASSERT_THROW(
        convertIcebergGeospatial(input, SPHERICAL_GEOGRAPHY(), pool(), "geog"),
        message);
  };
  const std::string kLongitude =
      "Invalid value in Iceberg geography column 'geog': Longitude must be between -180 and 180";
  const std::string kLatitude =
      "Invalid value in Iceberg geography column 'geog': Latitude must be between -90 and 90";

  expectFailure(toWkb("POINT (180.5 0)"), kLongitude);
  expectFailure(toWkb("POINT (-181 0)"), kLongitude);
  expectFailure(toWkb("POINT (0 90.5)"), kLatitude);
  expectFailure(toWkb("POINT (0 -91)"), kLatitude);
  // Latitude/longitude swapped, a common writer bug.
  expectFailure(toWkb("POINT (37.7749 -122.4194)"), kLatitude);

  // Every coordinate is checked, not only the first or the outer shape.
  expectFailure(toWkb("LINESTRING (0 0, 10 10, 200 10)"), kLongitude);
  expectFailure(
      toWkb("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 2 1, 2 95, 1 1))"),
      kLatitude);
  expectFailure(
      toWkb("GEOMETRYCOLLECTION (POINT (1 2), MULTIPOINT ((0 0), (0 100)))"),
      kLatitude);

  // NaN and infinity are out of range. POINT EMPTY is also NaN, NaN on disk,
  // so only a single NaN ordinate is invalid.
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  const auto infinity = std::numeric_limits<double>::infinity();
  expectFailure(pointWkb(nan, 0), kLongitude);
  expectFailure(pointWkb(0, nan), kLatitude);
  expectFailure(pointWkb(infinity, 0), kLongitude);
  expectFailure(pointWkb(0, -infinity), kLatitude);

  // Geography is two-dimensional like geometry; the error names the type.
  expectFailure(
      typeCodeOnlyWkb(1001),
      "Iceberg geography column 'geog' contains Z coordinates; Velox SPHERICALGEOGRAPHY supports only two-dimensional (XY) geometries");
}

TEST_F(IcebergGeospatialReadTest, geographyRangeBoundariesAreAccepted) {
  const std::vector<std::string> wkts = {
      "POINT (180 90)",
      "POINT (-180 -90)",
      "LINESTRING (-180 0, 180 0)",
      "POLYGON ((-180 -90, 180 -90, 180 90, -180 90, -180 -90))",
      "POINT EMPTY",
      "MULTIPOINT EMPTY",
  };
  std::vector<std::optional<std::string>> wkb;
  std::vector<std::optional<std::string>> expected;
  for (const auto& wkt : wkts) {
    wkb.emplace_back(toWkb(wkt));
    expected.emplace_back(toVeloxGeospatial(wkt));
  }
  auto converted = convertIcebergGeospatial(
      makeVarbinaryVector(wkb), SPHERICAL_GEOGRAPHY(), pool(), "geog");
  velox::test::assertEqualVectors(makeGeographyVector(expected), converted);
}

// Only geography carries a coordinate system. A geometry is planar with no
// bounds, so the same out-of-range values must still read as GEOMETRY.
TEST_F(IcebergGeospatialReadTest, geometryCoordinatesAreNotRangeChecked) {
  const std::string wkt = "LINESTRING (500 -500, 1000 1000)";
  auto converted = convertIcebergGeospatial(
      makeVarbinaryVector({toWkb(wkt)}), GEOMETRY(), pool(), "geom");
  velox::test::assertEqualVectors(
      makeGeometryVector({toVeloxGeospatial(wkt)}), converted);
}

// Validation runs on live positions only, like parsing: a row a filter or
// delete has already removed must not fail the read.
TEST_F(IcebergGeospatialReadTest, unselectedGeographyValuesAreNotValidated) {
  const std::string valid = "POINT (1 2)";
  auto input = makeVarbinaryVector(
      {toWkb(valid), toWkb("POINT (500 500)"), std::nullopt});
  SelectivityVector rows(input->size());
  rows.setValid(1, false);
  rows.updateBounds();

  auto converted = convertIcebergGeospatial(
      input, SPHERICAL_GEOGRAPHY(), rows, pool(), "geog");
  velox::test::assertEqualVectors(
      makeGeographyVector(
          {toVeloxGeospatial(valid), std::nullopt, std::nullopt}),
      converted);
}

// The children of a MULTIPOINT/MULTILINESTRING/MULTIPOLYGON/GEOMETRYCOLLECTION
// each carry their own WKB header, so a collection whose own type word says XY
// can still contain a Z/M/ZM/EWKB child. GEOS parses such a payload and
// GeometrySerializer would then write X and Y only, silently discarding the
// extra ordinate. Validation therefore has to walk every nested header.
TEST_F(IcebergGeospatialReadTest, nestedWkbHeadersAreValidated) {
  // A child geometry with an arbitrary type word and 'numOrdinates' doubles.
  auto child = [](uint32_t typeCode, int numOrdinates, bool hasSrid = false) {
    std::string out;
    out.push_back(1); // little endian
    appendUint32Le(out, typeCode);
    if (hasSrid) {
      appendUint32Le(out, 4326);
    }
    for (int i = 0; i < numOrdinates; ++i) {
      appendDoubleLe(out, 1.0 + i);
    }
    return out;
  };
  // A container of 'typeCode' wrapping the given complete child geometries.
  auto container = [](uint32_t typeCode,
                      const std::vector<std::string>& children) {
    std::string out;
    out.push_back(1);
    appendUint32Le(out, typeCode);
    appendUint32Le(out, static_cast<uint32_t>(children.size()));
    for (const auto& c : children) {
      out += c;
    }
    return out;
  };
  auto expectFailure = [&](const std::string& wkb, const std::string& message) {
    auto input = makeVarbinaryVector({wkb});
    VELOX_ASSERT_THROW(
        convertIcebergGeospatial(input, GEOMETRY(), pool(), "geom"), message);
  };

  const std::string xyPoint = child(1, 2);

  // A 2D GEOMETRYCOLLECTION whose child declares extra ordinates.
  expectFailure(
      container(7, {xyPoint, child(1001, 3)}), "contains Z coordinates");
  expectFailure(
      container(7, {xyPoint, child(2001, 3)}), "contains M coordinates");
  expectFailure(
      container(7, {xyPoint, child(3001, 4)}), "contains Z and M coordinates");
  // A child using PostGIS EWKB with an embedded SRID.
  expectFailure(
      container(7, {xyPoint, child(0x2000'0001, 2, /*hasSrid=*/true)}),
      "extended WKB (EWKB)");

  // The same, one level deeper: GEOMETRYCOLLECTION(GEOMETRYCOLLECTION(Z)).
  expectFailure(
      container(7, {container(7, {child(1001, 3)})}), "contains Z coordinates");

  // MULTIPOINT/MULTILINESTRING/MULTIPOLYGON use the same embedded-WKB
  // mechanism, so the recursive walk has to cover them too.
  expectFailure(
      container(4, {xyPoint, child(1001, 3)}), "contains Z coordinates");
  expectFailure(container(5, {child(1002, 0)}), "contains Z coordinates");
  expectFailure(container(6, {child(1003, 0)}), "contains Z coordinates");

  // An unsupported ISO code nested inside a valid collection.
  expectFailure(
      container(7, {xyPoint, child(15, 0)}),
      "unsupported WKB geometry type code 15");
}

// The validator must accept every legal nested XY shape, including empties and
// both byte orders, so the recursive check does not over-reject.
TEST_F(IcebergGeospatialReadTest, nestedXyWkbIsAccepted) {
  // Held as std::string rather than iterated straight off a braced list of
  // string literals: binding 'const std::string&' to a 'const char*' element
  // would construct a temporary per iteration, which
  // -Werror=range-loop-construct rejects.
  const std::vector<std::string> nestedXyWkts = {
      "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (0 0, 1 1))",
      "GEOMETRYCOLLECTION (MULTIPOINT ((1 2), (3 4)))",
      "GEOMETRYCOLLECTION (GEOMETRYCOLLECTION (POINT (1 2)))",
      "GEOMETRYCOLLECTION (POLYGON ((0 0, 1 0, 1 1, 0 0)))",
      "GEOMETRYCOLLECTION (MULTIPOINT EMPTY, POINT (1 2))",
      "GEOMETRYCOLLECTION EMPTY",
      "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((5 5, 9 5, 9 9, 5 9, 5 5)))",
      "MULTILINESTRING ((0 0, 5 5), (10 10, 20 20))",
  };
  for (const auto& wkt : nestedXyWkts) {
    auto input = makeVarbinaryVector({toWkb(wkt)});
    auto converted =
        convertIcebergGeospatial(input, GEOMETRY(), pool(), "geom");
    ASSERT_TRUE(isGeometryType(converted->type())) << wkt;
    velox::test::assertEqualVectors(
        makeGeometryVector({toVeloxGeospatial(wkt)}), converted);
  }
}

// Nesting is bounded so a deeply nested payload cannot exhaust the stack. Each
// GEOMETRYCOLLECTION level costs about nine bytes on disk but one frame in both
// this validator and in geos::io::WKBReader::read, and the GEOS version Velox
// pins has no depth cap of its own, so the limit has to be enforced here.
TEST_F(IcebergGeospatialReadTest, wkbNestingDepthIsBounded) {
  // A chain of 'collectionLevels' nested GEOMETRYCOLLECTIONs wrapping one
  // POINT. Total nesting depth is collectionLevels + 1 (the point itself).
  auto nestedCollections = [](int collectionLevels) {
    std::string out;
    out.push_back(1); // little endian
    appendUint32Le(out, 1); // POINT
    appendDoubleLe(out, 1.0);
    appendDoubleLe(out, 2.0);
    for (int i = 0; i < collectionLevels; ++i) {
      std::string wrapped;
      wrapped.push_back(1);
      appendUint32Le(wrapped, 7); // GEOMETRYCOLLECTION
      appendUint32Le(wrapped, 1); // exactly one child
      wrapped += out;
      out = std::move(wrapped);
    }
    return out;
  };

  // Depth exactly at the limit (99 collections + the innermost point) is
  // accepted, so the cap does not reject legitimately nested data.
  {
    auto input = makeVarbinaryVector({nestedCollections(99)});
    auto converted =
        convertIcebergGeospatial(input, GEOMETRY(), pool(), "geom");
    ASSERT_TRUE(isGeometryType(converted->type()));
    ASSERT_EQ(converted->size(), 1);
    EXPECT_FALSE(converted->isNullAt(0));
  }

  // One level beyond the limit is rejected, naming the column and the limit.
  {
    auto input = makeVarbinaryVector({nestedCollections(100)});
    VELOX_ASSERT_THROW(
        convertIcebergGeospatial(input, GEOMETRY(), pool(), "geom"),
        "Iceberg geometry column 'geom' is nested more than 100 levels deep");
  }

  // Far beyond the limit: rejected by the same check rather than recursing,
  // which is the case that would otherwise reach GEOS and overflow the stack.
  {
    auto input = makeVarbinaryVector({nestedCollections(50'000)});
    VELOX_ASSERT_THROW(
        convertIcebergGeospatial(input, GEOMETRY(), pool(), "geom"),
        "nested more than 100 levels deep");
  }
}

// A malformed or truncated payload must produce a user error rather than an
// out-of-bounds read while walking nested headers.
TEST_F(IcebergGeospatialReadTest, malformedNestedWkbIsRejectedSafely) {
  auto expectFailure = [&](const std::string& wkb) {
    auto input = makeVarbinaryVector({wkb});
    VELOX_ASSERT_THROW(
        convertIcebergGeospatial(input, GEOMETRY(), pool(), "geom"), "geom");
  };

  // A collection claiming two children but carrying none.
  std::string missingChildren;
  missingChildren.push_back(1);
  appendUint32Le(missingChildren, 7);
  appendUint32Le(missingChildren, 2);
  expectFailure(missingChildren);

  // A LINESTRING claiming a huge point count; the size computation must not
  // overflow into a value that passes the bounds check.
  std::string hugeCount;
  hugeCount.push_back(1);
  appendUint32Le(hugeCount, 2);
  appendUint32Le(hugeCount, 0xFFFF'FFFFu);
  expectFailure(hugeCount);

  // A child header truncated mid-type-word.
  std::string truncatedChild;
  truncatedChild.push_back(1);
  appendUint32Le(truncatedChild, 7);
  appendUint32Le(truncatedChild, 1);
  truncatedChild.push_back(1);
  truncatedChild.push_back(1);
  expectFailure(truncatedChild);

  // Trailing bytes after a complete geometry.
  std::string trailing = toWkb("POINT (1 2)");
  trailing.push_back('\x00');
  expectFailure(trailing);
}

TEST_F(IcebergGeospatialReadTest, nestedRowArrayAndMap) {
  const std::string wkt = "POINT (3 4)";
  const auto wkb = toWkb(wkt);
  const auto expectedBytes = toVeloxGeospatial(wkt);

  // ROW(BIGINT, GEOMETRY)
  {
    auto input = makeRowVector(
        {"id", "geom"},
        {makeFlatVector<int64_t>({1, 2}),
         makeVarbinaryVector({wkb, std::nullopt})});
    auto targetType = ROW({"id", "geom"}, {BIGINT(), GEOMETRY()});
    auto converted =
        convertIcebergGeospatial(input, targetType, pool(), "nested");
    ASSERT_TRUE(converted->type()->equivalent(*targetType));
    velox::test::assertEqualVectors(
        makeRowVector(
            {"id", "geom"},
            {makeFlatVector<int64_t>({1, 2}),
             makeGeometryVector({expectedBytes, std::nullopt})}),
        converted);
  }

  // ARRAY(GEOMETRY)
  {
    auto input = makeArrayVector<StringView>(
        {{StringView(wkb)}, {StringView(wkb), StringView(wkb)}}, VARBINARY());
    auto converted =
        convertIcebergGeospatial(input, ARRAY(GEOMETRY()), pool(), "shapes");
    ASSERT_TRUE(converted->type()->equivalent(*ARRAY(GEOMETRY())));
    auto* array = converted->as<ArrayVector>();
    ASSERT_EQ(array->size(), 2);
    ASSERT_TRUE(isGeometryType(array->elements()->type()));
    auto* elements = array->elements()->asFlatVector<StringView>();
    for (vector_size_t i = 0; i < 3; ++i) {
      EXPECT_EQ(elements->valueAt(i).str(), expectedBytes);
    }
  }

  // MAP(VARCHAR, GEOMETRY)
  {
    auto keys = makeFlatVector<StringView>({"a", "b"});
    auto values = makeVarbinaryVector({wkb, std::nullopt});
    auto offsets = makeIndices({0});
    auto sizes = makeIndices({2});
    auto input = std::make_shared<MapVector>(
        pool(),
        MAP(VARCHAR(), VARBINARY()),
        nullptr,
        1,
        offsets,
        sizes,
        keys,
        values);
    auto targetType = MAP(VARCHAR(), GEOMETRY());
    auto converted = convertIcebergGeospatial(input, targetType, pool(), "m");
    ASSERT_TRUE(converted->type()->equivalent(*targetType));
    auto* map = converted->as<MapVector>();
    ASSERT_TRUE(isGeometryType(map->mapValues()->type()));
    auto* mapValues = map->mapValues()->asFlatVector<StringView>();
    EXPECT_EQ(mapValues->valueAt(0).str(), expectedBytes);
    EXPECT_TRUE(mapValues->isNullAt(1));
    // Keys are shared with the input, not rebuilt.
    EXPECT_EQ(map->mapKeys().get(), keys.get());
  }
}

TEST_F(IcebergGeospatialReadTest, nestedGeography) {
  const std::string wkt = "POINT (3 4)";
  const auto wkb = toWkb(wkt);
  const auto expectedBytes = toVeloxGeospatial(wkt);
  const auto outOfRange = toWkb("POINT (3 400)");

  auto targetType =
      ROW({"id", "arr", "m"},
          {BIGINT(),
           ARRAY(SPHERICAL_GEOGRAPHY()),
           MAP(VARCHAR(), SPHERICAL_GEOGRAPHY())});
  auto makeInput = [&](const std::string& mapValue) {
    return makeRowVector(
        {"id", "arr", "m"},
        {makeFlatVector<int64_t>({1}),
         makeArrayVector<StringView>({{StringView(wkb)}}, VARBINARY()),
         makeMapVector(
             {0},
             makeFlatVector<StringView>({"k"}),
             makeFlatVector<StringView>({StringView(mapValue)}, VARBINARY()))});
  };

  auto converted =
      convertIcebergGeospatial(makeInput(wkb), targetType, pool(), "nested");
  ASSERT_TRUE(converted->type()->equivalent(*targetType));
  auto* row = converted->as<RowVector>();
  auto* array = row->childAt(1)->as<ArrayVector>();
  ASSERT_TRUE(isSphericalGeographyType(array->elements()->type()));
  EXPECT_EQ(
      array->elements()->asFlatVector<StringView>()->valueAt(0).str(),
      expectedBytes);
  auto* map = row->childAt(2)->as<MapVector>();
  ASSERT_TRUE(isSphericalGeographyType(map->mapValues()->type()));
  EXPECT_EQ(
      map->mapValues()->asFlatVector<StringView>()->valueAt(0).str(),
      expectedBytes);

  // A nested out-of-range value fails, naming the path to it.
  VELOX_ASSERT_THROW(
      convertIcebergGeospatial(
          makeInput(outOfRange), targetType, pool(), "nested"),
      "Invalid value in Iceberg geography column 'nested.m[value]': Latitude must be between -90 and 90");
}

TEST_F(IcebergGeospatialReadTest, arrayWithGapsAndNonZeroOffsets) {
  // The elements vector is deliberately *not* packed: unreferenced positions
  // hold bytes that are not WKB at all, and the first row starts at a non-zero
  // offset. Conversion must read only the positions the offsets/sizes reach, so
  // nothing here can be parsed by accident.
  const auto wkb = toWkb("POINT (1 2)");
  const auto expectedBytes = toVeloxGeospatial("POINT (1 2)");
  const std::string garbage = "definitely not wkb";

  auto elements = makeVarbinaryVector(
      {garbage, // unreferenced (before the first offset)
       wkb,
       wkb,
       garbage, // unreferenced (gap between rows)
       std::nullopt,
       wkb,
       garbage}); // unreferenced (past the last row)

  auto offsets = makeIndices({1, 4});
  auto sizes = makeIndices({2, 2});
  auto input = std::make_shared<ArrayVector>(
      pool(), ARRAY(VARBINARY()), nullptr, 2, offsets, sizes, elements);

  auto converted =
      convertIcebergGeospatial(input, ARRAY(GEOMETRY()), pool(), "shapes");

  auto* array = converted->as<ArrayVector>();
  ASSERT_EQ(array->size(), 2);
  // Offsets and sizes are preserved exactly.
  EXPECT_EQ(array->offsetAt(0), 1);
  EXPECT_EQ(array->sizeAt(0), 2);
  EXPECT_EQ(array->offsetAt(1), 4);
  EXPECT_EQ(array->sizeAt(1), 2);
  ASSERT_TRUE(isGeometryType(array->elements()->type()));

  auto* out = array->elements()->asFlatVector<StringView>();
  ASSERT_EQ(out->size(), elements->size());
  // Referenced, non-null.
  EXPECT_EQ(out->valueAt(1).str(), expectedBytes);
  EXPECT_EQ(out->valueAt(2).str(), expectedBytes);
  EXPECT_EQ(out->valueAt(5).str(), expectedBytes);
  // Referenced but null in the input.
  EXPECT_TRUE(out->isNullAt(4));
  // Unreferenced: never parsed, and carries no bytes.
  EXPECT_TRUE(out->isNullAt(0));
  EXPECT_TRUE(out->isNullAt(3));
  EXPECT_TRUE(out->isNullAt(6));
}

TEST_F(IcebergGeospatialReadTest, nullArrayRowsDoNotReachTheirElements) {
  // A null array row must not cause its element range to be parsed.
  const auto wkb = toWkb("POINT (1 2)");
  auto elements = makeVarbinaryVector({std::string("not wkb"), wkb});
  auto offsets = makeIndices({0, 1});
  auto sizes = makeIndices({1, 1});
  auto nulls = makeNulls(2, [](vector_size_t i) { return i == 0; });
  auto input = std::make_shared<ArrayVector>(
      pool(), ARRAY(VARBINARY()), nulls, 2, offsets, sizes, elements);

  auto converted =
      convertIcebergGeospatial(input, ARRAY(GEOMETRY()), pool(), "shapes");
  auto* array = converted->as<ArrayVector>();
  EXPECT_TRUE(array->isNullAt(0));
  EXPECT_TRUE(array->elements()->isNullAt(0));
  EXPECT_EQ(
      array->elements()->asFlatVector<StringView>()->valueAt(1).str(),
      toVeloxGeospatial("POINT (1 2)"));
}

TEST_F(IcebergGeospatialReadTest, dictionaryWrappedArray) {
  // The ARRAY itself is dictionary-wrapped: only the base rows the indices
  // reach may be converted.
  const auto wkb = toWkb("POINT (7 8)");
  const auto expectedBytes = toVeloxGeospatial("POINT (7 8)");
  auto elements =
      makeVarbinaryVector({wkb, std::string("not wkb at all"), wkb});
  auto offsets = makeIndices({0, 1, 2});
  auto sizes = makeIndices({1, 1, 1});
  auto base = std::make_shared<ArrayVector>(
      pool(), ARRAY(VARBINARY()), nullptr, 3, offsets, sizes, elements);

  // Row 1 of the base, which holds the garbage element, is never referenced.
  auto indices = makeIndices({0, 2, 2, 0});
  auto dictionary = BaseVector::wrapInDictionary(nullptr, indices, 4, base);

  auto converted =
      convertIcebergGeospatial(dictionary, ARRAY(GEOMETRY()), pool(), "shapes");
  ASSERT_EQ(converted->encoding(), VectorEncoding::Simple::DICTIONARY);
  ASSERT_EQ(converted->size(), 4);
  auto* convertedBase = converted->valueVector()->as<ArrayVector>();
  ASSERT_TRUE(isGeometryType(convertedBase->elements()->type()));
  auto* out = convertedBase->elements()->asFlatVector<StringView>();
  EXPECT_EQ(out->valueAt(0).str(), expectedBytes);
  EXPECT_TRUE(out->isNullAt(1));
  EXPECT_EQ(out->valueAt(2).str(), expectedBytes);
}

TEST_F(
    IcebergGeospatialReadTest,
    dictionaryOfGeometryLeafIgnoresUnreferencedEntries) {
  // Only the referenced dictionary entries are parsed, so a dictionary that
  // also holds non-WKB entries for other columns/batches does not break the
  // read.
  const auto wkb = toWkb("POINT (3 4)");
  auto base = makeVarbinaryVector(
      {wkb, std::string("garbage"), wkb, std::string("more garbage")});
  auto indices = makeIndices({0, 2, 0, 2, 2});
  auto dictionary = BaseVector::wrapInDictionary(nullptr, indices, 5, base);

  auto converted =
      convertIcebergGeospatial(dictionary, GEOMETRY(), pool(), "geom");
  ASSERT_EQ(converted->encoding(), VectorEncoding::Simple::DICTIONARY);
  auto* out = converted->valueVector()->asFlatVector<StringView>();
  EXPECT_EQ(out->valueAt(0).str(), toVeloxGeospatial("POINT (3 4)"));
  EXPECT_TRUE(out->isNullAt(1));
  EXPECT_EQ(out->valueAt(2).str(), toVeloxGeospatial("POINT (3 4)"));
  EXPECT_TRUE(out->isNullAt(3));
}

TEST_F(IcebergGeospatialReadTest, slicedGeometryVector) {
  // A sliced (non-zero offset) input: only the slice's own positions are live.
  const auto wkb = toWkb("POINT (5 6)");
  auto full = makeVarbinaryVector(
      {std::string("garbage"), wkb, wkb, std::string("garbage")});
  auto sliced = full->slice(1, 2);

  auto converted = convertIcebergGeospatial(sliced, GEOMETRY(), pool(), "geom");
  ASSERT_EQ(converted->size(), 2);
  DecodedVector decoded(*converted);
  EXPECT_EQ(
      decoded.valueAt<StringView>(0).str(), toVeloxGeospatial("POINT (5 6)"));
  EXPECT_EQ(
      decoded.valueAt<StringView>(1).str(), toVeloxGeospatial("POINT (5 6)"));
}

TEST_F(IcebergGeospatialReadTest, mapWithGapsAndNestedNulls) {
  // MAP keys and values are indexed by the same offsets/sizes; unreferenced
  // value slots must not be parsed and nested nulls must survive.
  const auto wkb = toWkb("POINT (9 9)");
  auto keys = makeFlatVector<StringView>({"skip", "a", "b", "skip"});
  auto values = makeVarbinaryVector(
      {std::string("not wkb"), wkb, std::nullopt, std::string("not wkb")});
  auto offsets = makeIndices({1});
  auto sizes = makeIndices({2});
  auto input = std::make_shared<MapVector>(
      pool(),
      MAP(VARCHAR(), VARBINARY()),
      nullptr,
      1,
      offsets,
      sizes,
      keys,
      values);

  auto converted =
      convertIcebergGeospatial(input, MAP(VARCHAR(), GEOMETRY()), pool(), "m");
  auto* map = converted->as<MapVector>();
  EXPECT_EQ(map->offsetAt(0), 1);
  EXPECT_EQ(map->sizeAt(0), 2);
  auto* out = map->mapValues()->asFlatVector<StringView>();
  EXPECT_TRUE(out->isNullAt(0));
  EXPECT_EQ(out->valueAt(1).str(), toVeloxGeospatial("POINT (9 9)"));
  EXPECT_TRUE(out->isNullAt(2));
  EXPECT_TRUE(out->isNullAt(3));
  // Keys are shared with the input, not rebuilt.
  EXPECT_EQ(map->mapKeys().get(), keys.get());
}

TEST_F(IcebergGeospatialReadTest, rowWithNullsInsideArray) {
  // ARRAY(ROW(..., GEOMETRY)): the element rows are positional, and the array's
  // offsets still decide which of them are live.
  const auto wkb = toWkb("POINT (2 3)");
  auto rowElements = makeRowVector(
      {"geom", "label"},
      {makeVarbinaryVector({std::string("not wkb"), wkb, std::nullopt}),
       makeFlatVector<StringView>({"x", "y", "z"})});
  auto offsets = makeIndices({1});
  auto sizes = makeIndices({2});
  auto input = std::make_shared<ArrayVector>(
      pool(),
      ARRAY(ROW({"geom", "label"}, {VARBINARY(), VARCHAR()})),
      nullptr,
      1,
      offsets,
      sizes,
      rowElements);

  auto targetType = ARRAY(ROW({"geom", "label"}, {GEOMETRY(), VARCHAR()}));
  auto converted =
      convertIcebergGeospatial(input, targetType, pool(), "shapes");
  auto* array = converted->as<ArrayVector>();
  auto* rows = array->elements()->as<RowVector>();
  auto* geom = rows->childAt(0)->asFlatVector<StringView>();
  EXPECT_TRUE(geom->isNullAt(0)); // unreferenced
  EXPECT_EQ(geom->valueAt(1).str(), toVeloxGeospatial("POINT (2 3)"));
  EXPECT_TRUE(geom->isNullAt(2)); // referenced but null
}

// ---------------------------------------------------------------------------
// End-to-end tests through the Iceberg connector and the Parquet reader.
// ---------------------------------------------------------------------------

#ifdef VELOX_ENABLE_PARQUET

TEST_F(IcebergGeospatialReadTest, parquetGeometryColumn) {
  std::vector<std::optional<std::string>> wkb;
  std::vector<std::optional<std::string>> expected;
  for (const auto& wkt : kAllKinds) {
    wkb.emplace_back(toWkb(wkt));
    expected.emplace_back(toVeloxGeospatial(wkt));
  }
  wkb.emplace_back(std::nullopt);
  expected.emplace_back(std::nullopt);

  std::vector<int64_t> ids(wkb.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    ids[i] = static_cast<int64_t>(i);
  }
  // A sibling Iceberg `binary` column must come back untouched. The length is
  // derived from the literal rather than hand-counted: an over-long count here
  // reads past the string literal, which ASAN reports as a global-buffer
  // overflow.
  static constexpr char kRawPayload[] = "\x01\x02\x03 raw bytes";
  std::vector<std::optional<std::string>> payload(
      wkb.size(), std::string(kRawPayload, sizeof(kRawPayload) - 1));

  auto data = makeRowVector(
      {"id", "geom", "payload"},
      {makeFlatVector<int64_t>(ids),
       makeVarbinaryVector(wkb),
       makeVarbinaryVector(payload)});

  auto expectedVector = makeRowVector(
      {"id", "geom", "payload"},
      {makeFlatVector<int64_t>(ids),
       makeGeometryVector(expected),
       makeVarbinaryVector(payload)});

  assertScan(
      {data},
      ROW({"id", "geom", "payload"}, {BIGINT(), GEOMETRY(), VARBINARY()}),
      {expectedVector});
}

TEST_F(IcebergGeospatialReadTest, parquetGeometryAcrossBatchesAndDictionaries) {
  // Few distinct values repeated many times over several batches: the Parquet
  // writer dictionary encodes the column and the reader hands the same
  // dictionary to consecutive batches.
  constexpr int32_t kNumBatches = 3;
  constexpr vector_size_t kRowsPerBatch = 500;

  std::vector<std::optional<std::string>> wkbCycle;
  std::vector<std::optional<std::string>> expectedCycle;
  for (const auto& wkt : kAllKinds) {
    wkbCycle.emplace_back(toWkb(wkt));
    expectedCycle.emplace_back(toVeloxGeospatial(wkt));
  }

  std::vector<RowVectorPtr> data;
  std::vector<RowVectorPtr> expected;
  for (int32_t batch = 0; batch < kNumBatches; ++batch) {
    std::vector<std::optional<std::string>> wkb;
    std::vector<std::optional<std::string>> expectedBytes;
    std::vector<int64_t> ids;
    for (vector_size_t row = 0; row < kRowsPerBatch; ++row) {
      const auto index = (batch * kRowsPerBatch + row);
      ids.push_back(index);
      if (index % 11 == 10) {
        wkb.emplace_back(std::nullopt);
        expectedBytes.emplace_back(std::nullopt);
      } else {
        wkb.emplace_back(wkbCycle[index % wkbCycle.size()]);
        expectedBytes.emplace_back(expectedCycle[index % expectedCycle.size()]);
      }
    }
    data.push_back(makeRowVector(
        {"id", "geom"},
        {makeFlatVector<int64_t>(ids), makeVarbinaryVector(wkb)}));
    expected.push_back(makeRowVector(
        {"id", "geom"},
        {makeFlatVector<int64_t>(ids), makeGeometryVector(expectedBytes)}));
  }

  assertScan(data, ROW({"id", "geom"}, {BIGINT(), GEOMETRY()}), expected);
}

TEST_F(IcebergGeospatialReadTest, parquetGeographyColumn) {
  std::vector<std::optional<std::string>> wkb;
  std::vector<std::optional<std::string>> expected;
  for (const auto& wkt : kAllKinds) {
    wkb.emplace_back(toWkb(wkt));
    expected.emplace_back(toVeloxGeospatial(wkt));
  }
  wkb.emplace_back(std::nullopt);
  expected.emplace_back(std::nullopt);
  std::vector<int64_t> ids(wkb.size());
  std::iota(ids.begin(), ids.end(), 0);

  auto data = makeRowVector(
      {"id", "geog"}, {makeFlatVector<int64_t>(ids), makeVarbinaryVector(wkb)});
  auto expectedVector = makeRowVector(
      {"id", "geog"},
      {makeFlatVector<int64_t>(ids), makeGeographyVector(expected)});
  assertScan(
      {data},
      ROW({"id", "geog"}, {BIGINT(), SPHERICAL_GEOGRAPHY()}),
      {expectedVector});
}

// An out-of-range value in a data file fails the scan with the column and the
// offending coordinate, rather than surfacing as a SPHERICALGEOGRAPHY.
TEST_F(
    IcebergGeospatialReadTest,
    parquetGeographyWithInvalidCoordinatesFailsScan) {
  auto data = makeRowVector(
      {"id", "geog"},
      {makeFlatVector<int64_t>({1, 2}),
       makeVarbinaryVector({toWkb("POINT (1 2)"), toWkb("POINT (200 2)")})});
  const auto outputDirectory = test::TempDirectoryPath::create();
  const auto dataSink =
      createDataSinkAndAppendData({data}, outputDirectory->getPath());
  dataSink->close();

  auto plan =
      exec::test::PlanBuilder()
          .startTableScan(test::kIcebergConnectorId)
          .outputType(ROW({"id", "geog"}, {BIGINT(), SPHERICAL_GEOGRAPHY()}))
          .endTableScan()
          .planNode();
  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(plan)
          .splits(createSplitsForDirectory(outputDirectory->getPath()))
          .copyResults(pool()),
      "Invalid value in Iceberg geography column 'geog': Longitude must be between -180 and 180; found 200");
}

TEST_F(IcebergGeospatialReadTest, plainEncodedGeometryColumn) {
  // Distinct, long values so the Parquet writer falls back to plain encoding.
  std::vector<std::optional<std::string>> wkb;
  std::vector<std::optional<std::string>> expected;
  for (int32_t i = 0; i < 200; ++i) {
    std::ostringstream wkt;
    wkt << "LINESTRING (";
    for (int32_t point = 0; point < 40; ++point) {
      if (point > 0) {
        wkt << ", ";
      }
      wkt << (i + point) << " " << (i * 2 + point);
    }
    wkt << ")";
    wkb.emplace_back(toWkb(wkt.str()));
    expected.emplace_back(toVeloxGeospatial(wkt.str()));
  }

  auto data = makeRowVector({"geom"}, {makeVarbinaryVector(wkb)});
  auto expectedVector = makeRowVector({"geom"}, {makeGeometryVector(expected)});
  assertScan({data}, ROW({"geom"}, {GEOMETRY()}), {expectedVector});
}

TEST_F(IcebergGeospatialReadTest, genericBinaryColumnIsNotDecoded) {
  // Same bytes, but the query asks for VARBINARY: the Iceberg schema does not
  // say geometry, so nothing is parsed and the bytes are returned verbatim.
  std::vector<std::optional<std::string>> wkb;
  for (const auto& wkt : kAllKinds) {
    wkb.emplace_back(toWkb(wkt));
  }
  auto data = makeRowVector({"payload"}, {makeVarbinaryVector(wkb)});
  assertScan({data}, ROW({"payload"}, {VARBINARY()}), {data});
}

TEST_F(IcebergGeospatialReadTest, veloxInternalGeometryBytesAreNotReparsed) {
  // A column holding Velox's *internal* geometry encoding (as a file written
  // from an existing GEOMETRY vector would) is not WKB. Read as VARBINARY it
  // must come back byte-identical; nothing may attempt to parse it.
  std::vector<std::optional<std::string>> internalBytes;
  for (const auto& wkt : kAllKinds) {
    internalBytes.emplace_back(toVeloxGeospatial(wkt));
  }
  auto data = makeRowVector({"payload"}, {makeVarbinaryVector(internalBytes)});
  assertScan({data}, ROW({"payload"}, {VARBINARY()}), {data});
}

TEST_F(IcebergGeospatialReadTest, hiveConnectorDoesNotConvert) {
  // The critical negative case: the generic Parquet reader must not decode WKB
  // just because the requested type is GEOMETRY. Only the Iceberg connector
  // converts, so the same file scanned through the Hive connector returns the
  // raw bytes.
  std::vector<std::optional<std::string>> wkb;
  for (const auto& wkt : kAllKinds) {
    wkb.emplace_back(toWkb(wkt));
  }
  auto data = makeRowVector({"geom"}, {makeVarbinaryVector(wkb)});

  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), {data});

  auto plan = exec::test::PlanBuilder()
                  .tableScan(ROW({"geom"}, {GEOMETRY()}))
                  .planNode();
  auto result =
      exec::test::AssertQueryBuilder(plan)
          .split(
              exec::test::HiveConnectorTestBase::makeHiveConnectorSplit(
                  filePath->getPath()))
          .copyResults(pool());

  ASSERT_EQ(result->size(), kAllKinds.size());
  // Compare bytes directly rather than with assertEqualVectors: the point of
  // the test is that the payload was not transformed, whatever type label the
  // scan carries.
  DecodedVector decoded(*result->as<RowVector>()->childAt(0));
  for (vector_size_t i = 0; i < result->size(); ++i) {
    ASSERT_FALSE(decoded.isNullAt(i));
    EXPECT_EQ(decoded.valueAt<StringView>(i).str(), *wkb[i])
        << "row " << i << " was modified by the generic reader";
  }
}

TEST_F(IcebergGeospatialReadTest, geometryParquetFileWrittenByAnotherEngine) {
  // A real Iceberg v3 data file produced outside Velox: a single `binary`
  // column carrying the GEOMETRY logical annotation and ISO WKB payloads.
  // Reading it as GEOMETRY must yield Velox's internal encoding for each shape.
  auto path = facebook::velox::test::getDataFilePath(
      "velox/connectors/hive/iceberg/tests", "examples/geometry.parquet");

  const std::vector<std::string> fileContents = {
      "POINT (10 20)",
      "LINESTRING (0 0, 10 10, 20 20)",
      "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
      "MULTIPOINT ((0 0), (10 20), (30 40))",
      "MULTILINESTRING ((0 0, 5 5), (10 10, 20 20))",
      "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((5 5, 9 5, 9 9, 5 9, 5 5)))"};

  std::vector<std::optional<std::string>> expected;
  for (const auto& wkt : fileContents) {
    expected.emplace_back(toVeloxGeospatial(wkt));
  }

  auto plan = exec::test::PlanBuilder()
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(ROW({"geom"}, {GEOMETRY()}))
                  .endTableScan()
                  .planNode();
  exec::test::AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(path))
      .assertResults(makeRowVector({"geom"}, {makeGeometryVector(expected)}));
}

TEST_F(IcebergGeospatialReadTest, geographyParquetFileWrittenByAnotherEngine) {
  // An Iceberg v3 geography data file written by parquet-java 1.16.0 as
  // Presto's Java writer does: a `binary` column with field id 1 carrying the
  // GEOGRAPHY logical annotation (default CRS and algorithm) and ISO WKB
  // payloads. Velox's Parquet thrift schema does not define that annotation, so
  // this also covers a reader that must skip it.
  auto path = facebook::velox::test::getDataFilePath(
      "velox/connectors/hive/iceberg/tests", "examples/geography.parquet");

  const std::vector<std::optional<std::string>> fileContents = {
      "POINT (-122.4194 37.7749)",
      "LINESTRING (-122.4194 37.7749, -118.2437 34.0522)",
      "POLYGON ((-10 -10, 10 -10, 10 10, -10 10, -10 -10))",
      "MULTIPOINT ((0 0), (180 90), (-180 -90))",
      "MULTILINESTRING ((0 0, 5 5), (10 10, 20 20))",
      "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((5 5, 9 5, 9 9, 5 9, 5 5)))",
      "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (0 0, 1 1))",
      "POINT EMPTY",
      std::nullopt,
  };

  std::vector<std::optional<std::string>> expected;
  for (const auto& wkt : fileContents) {
    expected.emplace_back(
        wkt.has_value() ? std::optional(toVeloxGeospatial(*wkt))
                        : std::nullopt);
  }

  auto plan = exec::test::PlanBuilder()
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(ROW({"geog"}, {SPHERICAL_GEOGRAPHY()}))
                  .endTableScan()
                  .planNode();
  exec::test::AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(path))
      .assertResults(makeRowVector({"geog"}, {makeGeographyVector(expected)}));
}

TEST_F(IcebergGeospatialReadTest, tableWithoutGeospatialIsUntouched) {
  auto data = makeRowVector(
      {"id", "name", "payload"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<std::string>({"a", "b", "c"}),
       makeVarbinaryVector(
           {std::string("\x00\x01", 2), std::nullopt, std::string("zzz")})});
  assertScan(
      {data},
      ROW({"id", "name", "payload"}, {BIGINT(), VARCHAR(), VARBINARY()}),
      {data});
}

TEST_F(IcebergGeospatialReadTest, nonParquetGeometryIsRejected) {
  // Iceberg also maps geometry onto ORC/DWRF binary, and the converter is
  // format-agnostic, but only the Parquet path has a fixture. Refuse the rest
  // instead of returning unverified values.
  fileFormat_ = dwio::common::FileFormat::DWRF;
  const auto wkb = toWkb("POINT (1 2)");
  auto data = makeRowVector({"geom"}, {makeVarbinaryVector({wkb})});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), {data});

  auto plan = exec::test::PlanBuilder()
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(ROW({"geom"}, {GEOMETRY()}))
                  .endTableScan()
                  .planNode();
  VELOX_ASSERT_THROW(
      exec::test::AssertQueryBuilder(plan)
          .splits(makeIcebergSplits(filePath->getPath()))
          .copyResults(pool()),
      "Reading Iceberg geometry or geography columns is only supported for Parquet files");
}

// Geometry support is read-only. The reader re-encodes the file's ISO WKB into
// Velox's internal encoding, but the writer does not perform the inverse
// conversion, and GEOMETRY is VARBINARY-backed -- so writing a GEOMETRY vector
// would put internal bytes on disk where the Iceberg spec requires WKB,
// producing a file neither this reader nor any other Iceberg engine can read.
// The sink rejects the write instead. Once the writer converts internal -> WKB,
// this test should be replaced by a GEOMETRY vector -> write -> read round
// trip.
TEST_F(IcebergGeospatialReadTest, writingGeospatialIsRejected) {
  const auto outputDirectory = test::TempDirectoryPath::create();

  VELOX_ASSERT_THROW(
      createDataSink(ROW({"geom"}, {GEOMETRY()}), outputDirectory->getPath()),
      "Writing an Iceberg geometry or geography column is not supported");

  // Nested geometry is rejected on the same grounds.
  VELOX_ASSERT_THROW(
      createDataSink(
          ROW({"r"}, {ROW({"geom"}, {GEOMETRY()})}),
          outputDirectory->getPath()),
      "Writing an Iceberg geometry or geography column is not supported");
  VELOX_ASSERT_THROW(
      createDataSink(
          ROW({"a"}, {ARRAY(GEOMETRY())}), outputDirectory->getPath()),
      "Writing an Iceberg geometry or geography column is not supported");

  // Geography is rejected on the same grounds, at any depth.
  VELOX_ASSERT_THROW(
      createDataSink(
          ROW({"geog"}, {SPHERICAL_GEOGRAPHY()}), outputDirectory->getPath()),
      "Writing an Iceberg geometry or geography column is not supported");
  VELOX_ASSERT_THROW(
      createDataSink(
          ROW({"m"}, {MAP(VARCHAR(), SPHERICAL_GEOGRAPHY())}),
          outputDirectory->getPath()),
      "Writing an Iceberg geometry or geography column is not supported");

  // A geometry-free schema still writes.
  EXPECT_NO_THROW(createDataSink(
      ROW({"id", "b"}, {BIGINT(), VARBINARY()}), outputDirectory->getPath()));
}

#endif // VELOX_ENABLE_PARQUET

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
