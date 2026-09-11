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
#include <sstream>

#include <folly/Singleton.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/geospatial/GeometrySerde.h"
#include "velox/connectors/hive/iceberg/IcebergGeometryConverter.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/types/GeometryRegistration.h"
#include "velox/functions/prestosql/types/GeometryType.h"
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

class IcebergGeometryReadTest : public test::IcebergTestBase {
 protected:
  void SetUp() override {
    test::IcebergTestBase::SetUp();
    folly::SingletonVault::singleton()->registrationComplete();
    registerGeometryType();
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

  // Velox's internal geometry encoding, i.e. what a GEOMETRY vector must hold.
  static std::string toVeloxGeometry(const std::string& wkt) {
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
    return makeFlatVector<StringView>(
        values.size(),
        [&](vector_size_t i) {
          return values[i].has_value() ? StringView(*values[i]) : StringView();
        },
        [&](vector_size_t i) { return !values[i].has_value(); },
        GEOMETRY());
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
};

// ---------------------------------------------------------------------------
// Vector-level tests for the converter the Iceberg connector owns.
// ---------------------------------------------------------------------------

TEST_F(IcebergGeometryReadTest, containsGeometryDetection) {
  // Nothing but an actual GEOMETRY may switch the conversion on.
  EXPECT_FALSE(containsGeometry(VARBINARY()));
  EXPECT_FALSE(containsGeometry(VARCHAR()));
  EXPECT_FALSE(containsGeometry(ROW({"a", "b"}, {BIGINT(), VARBINARY()})));
  EXPECT_FALSE(containsGeometry(ARRAY(VARBINARY())));
  EXPECT_FALSE(containsGeometry(MAP(VARCHAR(), VARBINARY())));

  EXPECT_TRUE(containsGeometry(GEOMETRY()));
  EXPECT_TRUE(containsGeometry(ROW({"a", "g"}, {BIGINT(), GEOMETRY()})));
  EXPECT_TRUE(containsGeometry(ARRAY(GEOMETRY())));
  EXPECT_TRUE(containsGeometry(MAP(VARCHAR(), GEOMETRY())));
  EXPECT_TRUE(containsGeometry(ARRAY(ROW({"g"}, {GEOMETRY()}))));
}

TEST_F(IcebergGeometryReadTest, flatVectorAllGeometryKinds) {
  std::vector<std::optional<std::string>> wkb;
  std::vector<std::optional<std::string>> expected;
  for (const auto& wkt : kAllKinds) {
    wkb.emplace_back(toWkb(wkt));
    expected.emplace_back(toVeloxGeometry(wkt));
  }
  for (const auto& wkt : kEmptyKinds) {
    wkb.emplace_back(toWkb(wkt));
    expected.emplace_back(toVeloxGeometry(wkt));
  }
  // Nulls interleaved at both ends.
  wkb.emplace_back(std::nullopt);
  expected.emplace_back(std::nullopt);

  auto input = makeVarbinaryVector(wkb);
  auto converted = convertIcebergGeometry(input, GEOMETRY(), pool(), "geom");

  ASSERT_TRUE(isGeometryType(converted->type()));
  velox::test::assertEqualVectors(makeGeometryVector(expected), converted);
}

TEST_F(IcebergGeometryReadTest, dictionaryVectorIsConvertedOncePerEntry) {
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
      convertIcebergGeometry(dictionary, GEOMETRY(), pool(), "geom");

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
      expected.emplace_back(toVeloxGeometry(distinctWkt[i % 3]));
    }
  }
  velox::test::assertEqualVectors(makeGeometryVector(expected), converted);
}

TEST_F(IcebergGeometryReadTest, nullConstantVector) {
  auto input = BaseVector::createNullConstant(VARBINARY(), 5, pool());
  auto converted = convertIcebergGeometry(input, GEOMETRY(), pool(), "geom");
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
TEST_F(IcebergGeometryReadTest, nonNullConstantVectorPreservesEncoding) {
  const std::string wkt = "POINT (10 20)";
  auto value = makeVarbinaryVector({toWkb(wkt)});
  auto input = BaseVector::wrapInConstant(5, 0, value);
  ASSERT_EQ(input->encoding(), VectorEncoding::Simple::CONSTANT);

  auto converted = convertIcebergGeometry(input, GEOMETRY(), pool(), "geom");

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
          5, 0, makeGeometryVector({toVeloxGeometry(wkt)})),
      converted);
}

// The converter must not parse a value that no selected row can reach.
TEST_F(IcebergGeometryReadTest, constantVectorWithEmptySelectionIsNotParsed) {
  // Deliberately invalid WKB: if the value were parsed, this would throw.
  auto value = makeVarbinaryVector({std::string("\x01\x02\x03", 3)});
  auto input = BaseVector::wrapInConstant(4, 0, value);

  SelectivityVector noRows(4, false);
  auto converted =
      convertIcebergGeometry(input, GEOMETRY(), noRows, pool(), "geom");

  EXPECT_TRUE(isGeometryType(converted->type()));
  ASSERT_EQ(converted->size(), 4);
  for (vector_size_t i = 0; i < 4; ++i) {
    EXPECT_TRUE(converted->isNullAt(i));
  }
}

// A partially selected constant still converts its single value once; the
// constant result carries that value at every position, which is what CONSTANT
// encoding means.
TEST_F(IcebergGeometryReadTest, constantVectorWithPartialSelection) {
  const std::string wkt = "LINESTRING (0 0, 10 10, 20 20)";
  auto value = makeVarbinaryVector({toWkb(wkt)});
  auto input = BaseVector::wrapInConstant(4, 0, value);

  SelectivityVector someRows(4, false);
  someRows.setValid(1, true);
  someRows.setValid(2, true);
  someRows.updateBounds();
  auto converted =
      convertIcebergGeometry(input, GEOMETRY(), someRows, pool(), "geom");

  EXPECT_EQ(converted->encoding(), VectorEncoding::Simple::CONSTANT);
  EXPECT_TRUE(isGeometryType(converted->type()));
  EXPECT_EQ(converted->wrappedIndex(3), 0);
  velox::test::assertEqualVectors(
      BaseVector::wrapInConstant(
          4, 0, makeGeometryVector({toVeloxGeometry(wkt)})),
      converted);
}

// A constant complex value goes through the same ROW recursion as a flat one,
// so the geometry leaf is converted and the outer CONSTANT encoding is
// preserved.
TEST_F(IcebergGeometryReadTest, constantRowWithGeometryField) {
  const std::string wkt = "POINT (3 4)";
  auto row = makeRowVector(
      {"id", "geom"},
      {makeFlatVector<int64_t>({7}), makeVarbinaryVector({toWkb(wkt)})});
  auto input = BaseVector::wrapInConstant(3, 0, row);
  auto targetType = ROW({"id", "geom"}, {BIGINT(), GEOMETRY()});

  auto converted = convertIcebergGeometry(input, targetType, pool(), "nested");

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
               makeGeometryVector({toVeloxGeometry(wkt)})})),
      converted);
}

// An Iceberg equality-delete file stores its geometry as the ISO WKB the spec
// mandates, while the base rows it is probed against have already been
// re-encoded into Velox's internal geometry encoding. Both sides have to be
// hashed in the same logical encoding or the delete silently never matches.
TEST_F(IcebergGeometryReadTest, equalityDeleteOnGeometryColumn) {
  const std::string matchWkt = "POINT (10 20)";
  const std::string keepWkt = "LINESTRING (0 0, 1 1)";

  // Base file: written as binary WKB and read back as GEOMETRY, which is how a
  // real Iceberg geometry data file looks on disk.
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
      std::dynamic_pointer_cast<HiveConnectorSplit>(deleteSplits[0])->filePath;

  // Field id 2 is the second top-level column, "geom".
  IcebergDeleteFile equalityDelete(
      FileContent::kEqualityDeletes,
      deleteFilePath,
      dwio::common::FileFormat::PARQUET,
      1,
      getFileSize(deleteFilePath),
      /*equalityFieldIds=*/{2});

  const auto tableSchema = ROW({"id", "geom"}, {BIGINT(), GEOMETRY()});
  auto plan = exec::test::PlanBuilder()
                  .startTableScan(test::kIcebergConnectorId)
                  .outputType(tableSchema)
                  .dataColumns(tableSchema)
                  .endTableScan()
                  .planNode();

  // G_match is deleted; G_keep survives. Without the delete-side conversion the
  // delete key hashes as raw WKB while the base row hashes as internal bytes,
  // nothing matches, and both rows come back.
  auto expected = makeRowVector(
      {"id", "geom"},
      {makeFlatVector<int64_t>({2}),
       makeGeometryVector({toVeloxGeometry(keepWkt)})});

  exec::test::AssertQueryBuilder(plan)
      .splits(makeIcebergSplits(baseFilePath, {equalityDelete}))
      .assertResults(expected);
}

// A hash join on a GEOMETRY key must return the matching Iceberg row even with
// string/binary dynamic-filter pushdown enabled. The build side holds Velox
// internal geometry bytes while the Iceberg file holds ISO WKB, so a
// BytesValues filter built from the build side and evaluated by the scan
// against the file bytes would drop the row. No filter may be produced for
// this custom VARBINARY-backed type, leaving the join to do the matching.
TEST_F(IcebergGeometryReadTest, geometryHashJoinWithDynamicFilterPushdown) {
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
      {"bgeom"}, {makeGeometryVector({toVeloxGeometry(matchWkt)})});

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

TEST_F(IcebergGeometryReadTest, invalidWkbErrorNamesColumnPath) {
  auto shortValue = makeVarbinaryVector({std::string("\x01\x02\x03", 3)});
  VELOX_ASSERT_THROW(
      convertIcebergGeometry(shortValue, GEOMETRY(), pool(), "shapes.geom"),
      "Iceberg geometry column 'shapes.geom'");

  auto badByteOrder =
      makeVarbinaryVector({typeCodeOnlyWkb(1).replace(0, 1, "\x07")});
  VELOX_ASSERT_THROW(
      convertIcebergGeometry(badByteOrder, GEOMETRY(), pool(), "geom"),
      "unknown byte order marker");

  // A valid header with a truncated body: GEOS rejects it and the message still
  // names the column.
  auto truncated =
      makeVarbinaryVector({std::string("\x01\x01\x00\x00\x00\x00\x00", 7)});
  VELOX_ASSERT_THROW(
      convertIcebergGeometry(truncated, GEOMETRY(), pool(), "geom"),
      "Iceberg geometry column 'geom'");
}

TEST_F(IcebergGeometryReadTest, zmAndEwkbAreRejectedNotFlattened) {
  auto expectFailure = [&](uint32_t typeCode, const std::string& message) {
    auto input = makeVarbinaryVector({typeCodeOnlyWkb(typeCode)});
    VELOX_ASSERT_THROW(
        convertIcebergGeometry(input, GEOMETRY(), pool(), "geom"), message);
  };

  expectFailure(1001, "contains Z coordinates");
  expectFailure(2001, "contains M coordinates");
  expectFailure(3001, "contains Z and M coordinates");
  expectFailure(0x2000'0001, "extended WKB (EWKB)");
  expectFailure(0x8000'0001, "extended WKB (EWKB)");
  // ISO code 15 is PolyhedralSurface, which Velox GEOMETRY cannot represent.
  expectFailure(15, "unsupported WKB geometry type code 15");
}

// The children of a MULTIPOINT/MULTILINESTRING/MULTIPOLYGON/GEOMETRYCOLLECTION
// each carry their own WKB header, so a collection whose own type word says XY
// can still contain a Z/M/ZM/EWKB child. GEOS parses such a payload and
// GeometrySerializer would then write X and Y only, silently discarding the
// extra ordinate. Validation therefore has to walk every nested header.
TEST_F(IcebergGeometryReadTest, nestedWkbHeadersAreValidated) {
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
        convertIcebergGeometry(input, GEOMETRY(), pool(), "geom"), message);
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
TEST_F(IcebergGeometryReadTest, nestedXyWkbIsAccepted) {
  for (
      const std::string& wkt : {
          "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (0 0, 1 1))",
          "GEOMETRYCOLLECTION (MULTIPOINT ((1 2), (3 4)))",
          "GEOMETRYCOLLECTION (GEOMETRYCOLLECTION (POINT (1 2)))",
          "GEOMETRYCOLLECTION (POLYGON ((0 0, 1 0, 1 1, 0 0)))",
          "GEOMETRYCOLLECTION (MULTIPOINT EMPTY, POINT (1 2))",
          "GEOMETRYCOLLECTION EMPTY",
          "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((5 5, 9 5, 9 9, 5 9, 5 5)))",
          "MULTILINESTRING ((0 0, 5 5), (10 10, 20 20))",
      }) {
    auto input = makeVarbinaryVector({toWkb(wkt)});
    auto converted = convertIcebergGeometry(input, GEOMETRY(), pool(), "geom");
    ASSERT_TRUE(isGeometryType(converted->type())) << wkt;
    velox::test::assertEqualVectors(
        makeGeometryVector({toVeloxGeometry(wkt)}), converted);
  }
}

// A malformed or truncated payload must produce a user error rather than an
// out-of-bounds read while walking nested headers.
TEST_F(IcebergGeometryReadTest, malformedNestedWkbIsRejectedSafely) {
  auto expectFailure = [&](const std::string& wkb) {
    auto input = makeVarbinaryVector({wkb});
    VELOX_ASSERT_THROW(
        convertIcebergGeometry(input, GEOMETRY(), pool(), "geom"), "geom");
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

TEST_F(IcebergGeometryReadTest, nestedRowArrayAndMap) {
  const std::string wkt = "POINT (3 4)";
  const auto wkb = toWkb(wkt);
  const auto expectedBytes = toVeloxGeometry(wkt);

  // ROW(BIGINT, GEOMETRY)
  {
    auto input = makeRowVector(
        {"id", "geom"},
        {makeFlatVector<int64_t>({1, 2}),
         makeVarbinaryVector({wkb, std::nullopt})});
    auto targetType = ROW({"id", "geom"}, {BIGINT(), GEOMETRY()});
    auto converted =
        convertIcebergGeometry(input, targetType, pool(), "nested");
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
        convertIcebergGeometry(input, ARRAY(GEOMETRY()), pool(), "shapes");
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
    auto converted = convertIcebergGeometry(input, targetType, pool(), "m");
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

TEST_F(IcebergGeometryReadTest, arrayWithGapsAndNonZeroOffsets) {
  // The elements vector is deliberately *not* packed: unreferenced positions
  // hold bytes that are not WKB at all, and the first row starts at a non-zero
  // offset. Conversion must read only the positions the offsets/sizes reach, so
  // nothing here can be parsed by accident.
  const auto wkb = toWkb("POINT (1 2)");
  const auto expectedBytes = toVeloxGeometry("POINT (1 2)");
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
      convertIcebergGeometry(input, ARRAY(GEOMETRY()), pool(), "shapes");

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

TEST_F(IcebergGeometryReadTest, nullArrayRowsDoNotReachTheirElements) {
  // A null array row must not cause its element range to be parsed.
  const auto wkb = toWkb("POINT (1 2)");
  auto elements = makeVarbinaryVector({std::string("not wkb"), wkb});
  auto offsets = makeIndices({0, 1});
  auto sizes = makeIndices({1, 1});
  auto nulls = makeNulls(2, [](vector_size_t i) { return i == 0; });
  auto input = std::make_shared<ArrayVector>(
      pool(), ARRAY(VARBINARY()), nulls, 2, offsets, sizes, elements);

  auto converted =
      convertIcebergGeometry(input, ARRAY(GEOMETRY()), pool(), "shapes");
  auto* array = converted->as<ArrayVector>();
  EXPECT_TRUE(array->isNullAt(0));
  EXPECT_TRUE(array->elements()->isNullAt(0));
  EXPECT_EQ(
      array->elements()->asFlatVector<StringView>()->valueAt(1).str(),
      toVeloxGeometry("POINT (1 2)"));
}

TEST_F(IcebergGeometryReadTest, dictionaryWrappedArray) {
  // The ARRAY itself is dictionary-wrapped: only the base rows the indices
  // reach may be converted.
  const auto wkb = toWkb("POINT (7 8)");
  const auto expectedBytes = toVeloxGeometry("POINT (7 8)");
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
      convertIcebergGeometry(dictionary, ARRAY(GEOMETRY()), pool(), "shapes");
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
    IcebergGeometryReadTest,
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
      convertIcebergGeometry(dictionary, GEOMETRY(), pool(), "geom");
  ASSERT_EQ(converted->encoding(), VectorEncoding::Simple::DICTIONARY);
  auto* out = converted->valueVector()->asFlatVector<StringView>();
  EXPECT_EQ(out->valueAt(0).str(), toVeloxGeometry("POINT (3 4)"));
  EXPECT_TRUE(out->isNullAt(1));
  EXPECT_EQ(out->valueAt(2).str(), toVeloxGeometry("POINT (3 4)"));
  EXPECT_TRUE(out->isNullAt(3));
}

TEST_F(IcebergGeometryReadTest, slicedGeometryVector) {
  // A sliced (non-zero offset) input: only the slice's own positions are live.
  const auto wkb = toWkb("POINT (5 6)");
  auto full = makeVarbinaryVector(
      {std::string("garbage"), wkb, wkb, std::string("garbage")});
  auto sliced = full->slice(1, 2);

  auto converted = convertIcebergGeometry(sliced, GEOMETRY(), pool(), "geom");
  ASSERT_EQ(converted->size(), 2);
  DecodedVector decoded(*converted);
  EXPECT_EQ(
      decoded.valueAt<StringView>(0).str(), toVeloxGeometry("POINT (5 6)"));
  EXPECT_EQ(
      decoded.valueAt<StringView>(1).str(), toVeloxGeometry("POINT (5 6)"));
}

TEST_F(IcebergGeometryReadTest, mapWithGapsAndNestedNulls) {
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
      convertIcebergGeometry(input, MAP(VARCHAR(), GEOMETRY()), pool(), "m");
  auto* map = converted->as<MapVector>();
  EXPECT_EQ(map->offsetAt(0), 1);
  EXPECT_EQ(map->sizeAt(0), 2);
  auto* out = map->mapValues()->asFlatVector<StringView>();
  EXPECT_TRUE(out->isNullAt(0));
  EXPECT_EQ(out->valueAt(1).str(), toVeloxGeometry("POINT (9 9)"));
  EXPECT_TRUE(out->isNullAt(2));
  EXPECT_TRUE(out->isNullAt(3));
  // Keys are shared with the input, not rebuilt.
  EXPECT_EQ(map->mapKeys().get(), keys.get());
}

TEST_F(IcebergGeometryReadTest, rowWithNullsInsideArray) {
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
  auto converted = convertIcebergGeometry(input, targetType, pool(), "shapes");
  auto* array = converted->as<ArrayVector>();
  auto* rows = array->elements()->as<RowVector>();
  auto* geom = rows->childAt(0)->asFlatVector<StringView>();
  EXPECT_TRUE(geom->isNullAt(0)); // unreferenced
  EXPECT_EQ(geom->valueAt(1).str(), toVeloxGeometry("POINT (2 3)"));
  EXPECT_TRUE(geom->isNullAt(2)); // referenced but null
}

// ---------------------------------------------------------------------------
// End-to-end tests through the Iceberg connector and the Parquet reader.
// ---------------------------------------------------------------------------

#ifdef VELOX_ENABLE_PARQUET

TEST_F(IcebergGeometryReadTest, parquetGeometryColumn) {
  std::vector<std::optional<std::string>> wkb;
  std::vector<std::optional<std::string>> expected;
  for (const auto& wkt : kAllKinds) {
    wkb.emplace_back(toWkb(wkt));
    expected.emplace_back(toVeloxGeometry(wkt));
  }
  wkb.emplace_back(std::nullopt);
  expected.emplace_back(std::nullopt);

  std::vector<int64_t> ids(wkb.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    ids[i] = static_cast<int64_t>(i);
  }
  // A sibling Iceberg `binary` column must come back untouched.
  std::vector<std::optional<std::string>> payload(
      wkb.size(), std::string("\x01\x02\x03 raw bytes", 15));

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

TEST_F(IcebergGeometryReadTest, parquetGeometryAcrossBatchesAndDictionaries) {
  // Few distinct values repeated many times over several batches: the Parquet
  // writer dictionary encodes the column and the reader hands the same
  // dictionary to consecutive batches.
  constexpr int32_t kNumBatches = 3;
  constexpr vector_size_t kRowsPerBatch = 500;

  std::vector<std::optional<std::string>> wkbCycle;
  std::vector<std::optional<std::string>> expectedCycle;
  for (const auto& wkt : kAllKinds) {
    wkbCycle.emplace_back(toWkb(wkt));
    expectedCycle.emplace_back(toVeloxGeometry(wkt));
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

TEST_F(IcebergGeometryReadTest, plainEncodedGeometryColumn) {
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
    expected.emplace_back(toVeloxGeometry(wkt.str()));
  }

  auto data = makeRowVector({"geom"}, {makeVarbinaryVector(wkb)});
  auto expectedVector = makeRowVector({"geom"}, {makeGeometryVector(expected)});
  assertScan({data}, ROW({"geom"}, {GEOMETRY()}), {expectedVector});
}

TEST_F(IcebergGeometryReadTest, genericBinaryColumnIsNotDecoded) {
  // Same bytes, but the query asks for VARBINARY: the Iceberg schema does not
  // say geometry, so nothing is parsed and the bytes are returned verbatim.
  std::vector<std::optional<std::string>> wkb;
  for (const auto& wkt : kAllKinds) {
    wkb.emplace_back(toWkb(wkt));
  }
  auto data = makeRowVector({"payload"}, {makeVarbinaryVector(wkb)});
  assertScan({data}, ROW({"payload"}, {VARBINARY()}), {data});
}

TEST_F(IcebergGeometryReadTest, veloxInternalGeometryBytesAreNotReparsed) {
  // A column holding Velox's *internal* geometry encoding (as a file written
  // from an existing GEOMETRY vector would) is not WKB. Read as VARBINARY it
  // must come back byte-identical; nothing may attempt to parse it.
  std::vector<std::optional<std::string>> internalBytes;
  for (const auto& wkt : kAllKinds) {
    internalBytes.emplace_back(toVeloxGeometry(wkt));
  }
  auto data = makeRowVector({"payload"}, {makeVarbinaryVector(internalBytes)});
  assertScan({data}, ROW({"payload"}, {VARBINARY()}), {data});
}

TEST_F(IcebergGeometryReadTest, hiveConnectorDoesNotConvert) {
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

TEST_F(IcebergGeometryReadTest, geometryParquetFileWrittenByAnotherEngine) {
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
    expected.emplace_back(toVeloxGeometry(wkt));
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

TEST_F(IcebergGeometryReadTest, tableWithoutGeometryIsUntouched) {
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

TEST_F(IcebergGeometryReadTest, nonParquetGeometryIsRejected) {
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
      "Reading Iceberg geometry columns is only supported for Parquet files");
}

// Geometry support is read-only. The reader re-encodes the file's ISO WKB into
// Velox's internal encoding, but the writer does not perform the inverse
// conversion, and GEOMETRY is VARBINARY-backed -- so writing a GEOMETRY vector
// would put internal bytes on disk where the Iceberg spec requires WKB,
// producing a file neither this reader nor any other Iceberg engine can read.
// The sink rejects the write instead. Once the writer converts internal -> WKB,
// this test should be replaced by a GEOMETRY vector -> write -> read round
// trip.
TEST_F(IcebergGeometryReadTest, writingGeometryIsRejected) {
  const auto outputDirectory = test::TempDirectoryPath::create();

  VELOX_ASSERT_THROW(
      createDataSink(ROW({"geom"}, {GEOMETRY()}), outputDirectory->getPath()),
      "Writing an Iceberg geometry column is not supported");

  // Nested geometry is rejected on the same grounds.
  VELOX_ASSERT_THROW(
      createDataSink(
          ROW({"r"}, {ROW({"geom"}, {GEOMETRY()})}),
          outputDirectory->getPath()),
      "Writing an Iceberg geometry column is not supported");
  VELOX_ASSERT_THROW(
      createDataSink(
          ROW({"a"}, {ARRAY(GEOMETRY())}), outputDirectory->getPath()),
      "Writing an Iceberg geometry column is not supported");

  // A geometry-free schema still writes.
  EXPECT_NO_THROW(createDataSink(
      ROW({"id", "b"}, {BIGINT(), VARBINARY()}), outputDirectory->getPath()));
}

#endif // VELOX_ENABLE_PARQUET

} // namespace
} // namespace facebook::velox::connector::hive::iceberg
