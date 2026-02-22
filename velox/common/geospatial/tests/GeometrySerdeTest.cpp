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

#include "velox/common/geospatial/GeometrySerde.h"
#include <geos/geom/Geometry.h>
#include <geos/io/WKTReader.h>
#include <geos/io/WKTWriter.h>
#include <gtest/gtest.h>
#include "velox/type/StringView.h"

using namespace ::testing;

using namespace facebook::velox::common::geospatial;

void assertRoundtrip(const std::string& wkt) {
  geos::io::WKTReader reader;
  geos::io::WKTWriter writer;
  std::unique_ptr<geos::geom::Geometry> geometry = reader.read(wkt);

  std::string buffer;
  GeometrySerializer::serialize(*geometry, buffer);
  facebook::velox::StringView readBuffer(buffer);
  auto deserialized = GeometryDeserializer::deserialize(readBuffer);

  EXPECT_TRUE(geometry->equals(deserialized.get()))
      << std::endl
      << "Input:" << std::endl
      << wkt << std::endl
      << "Output:" << std::endl
      << writer.write(deserialized.get());
}

TEST(GeometrySerdeTest, testBasicSerde) {
  assertRoundtrip("POINT EMPTY");
  assertRoundtrip("POINT (1 2)");
  assertRoundtrip("MULTIPOINT EMPTY");
  assertRoundtrip("MULTIPOINT (1 2)");
  assertRoundtrip("MULTIPOINT (1 2, 1 0)");

  assertRoundtrip("LINESTRING EMPTY");
  assertRoundtrip("LINESTRING (1 2, 1 0)");
  assertRoundtrip("LINESTRING (1 0, 2 0, 2 1, 1 1, 1 0)");
  assertRoundtrip("MULTILINESTRING EMPTY");
  assertRoundtrip("MULTILINESTRING ((1 2, 1 0))");
  assertRoundtrip("MULTILINESTRING ((1 2, 1 0), (10 11, 12 13))");

  assertRoundtrip("POLYGON EMPTY");
  assertRoundtrip("POLYGON ((1 0, 2 0, 2 1, 1 1, 1 0))");
  assertRoundtrip(
      "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 2 1, 2 2, 1 2, 1 1))");
  assertRoundtrip("MULTIPOLYGON EMPTY");
  assertRoundtrip("MULTIPOLYGON (((1 0, 2 0, 2 1, 1 1, 1 0)))");
  assertRoundtrip(
      "MULTIPOLYGON ( ((10 0, 20 0, 20 10, 10 10, 10 0)),  ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 2 1, 2 2, 1 2, 1 1)) )");
}

TEST(GeometrySerdeTest, testGeometryCollectionSerde) {
  assertRoundtrip("GEOMETRYCOLLECTION EMPTY");
  assertRoundtrip("GEOMETRYCOLLECTION (POINT EMPTY)");
  assertRoundtrip("GEOMETRYCOLLECTION (POINT (0 0))");
  assertRoundtrip("GEOMETRYCOLLECTION (POINT (0 0), POINT EMPTY)");
  assertRoundtrip("GEOMETRYCOLLECTION (POINT (0 0), POINT (0 0))");
  assertRoundtrip("GEOMETRYCOLLECTION (POINT (0 0), POINT (1 1))");
  assertRoundtrip("GEOMETRYCOLLECTION (MULTIPOINT EMPTY)");
  assertRoundtrip(
      "GEOMETRYCOLLECTION (MULTIPOINT (0 0, 1 2), POINT (1 1), MULTIPOINT EMPTY)");

  assertRoundtrip("GEOMETRYCOLLECTION (LINESTRING EMPTY)");
  assertRoundtrip("GEOMETRYCOLLECTION (MULTILINESTRING EMPTY)");
  assertRoundtrip(
      "GEOMETRYCOLLECTION (MULTILINESTRING ((0 1, 2 3, 0 3, 0 1), (10 10, 10 12, 12 10)), POINT EMPTY, LINESTRING (0 0, -1 -1, 2 0))");

  assertRoundtrip("GEOMETRYCOLLECTION (POLYGON EMPTY)");
  assertRoundtrip("GEOMETRYCOLLECTION (MULTIPOLYGON EMPTY)");
  assertRoundtrip("GEOMETRYCOLLECTION (GEOMETRYCOLLECTION EMPTY)");
  assertRoundtrip(
      "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (8 4, 5 7), POLYGON EMPTY)");
  assertRoundtrip(
      "GEOMETRYCOLLECTION (GEOMETRYCOLLECTION ( MULTIPOINT (1 2) ))");
}

TEST(GeometrySerdeTest, testComplexSerde) {
  assertRoundtrip("GEOMETRYCOLLECTION ( MULTIPOINT EMPTY, MULTIPOINT (1 1) )");
  assertRoundtrip("GEOMETRYCOLLECTION (POLYGON EMPTY, POINT (1 2))");
  assertRoundtrip(
      "GEOMETRYCOLLECTION (POLYGON EMPTY, MULTIPOINT (1 2), GEOMETRYCOLLECTION ( MULTIPOINT (3 4) ))");
  assertRoundtrip(
      "GEOMETRYCOLLECTION (POLYGON EMPTY, GEOMETRYCOLLECTION ( POINT (1 2), POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 2 1, 2 2, 1 2, 1 1)), GEOMETRYCOLLECTION EMPTY, MULTIPOLYGON ( ((10 10, 14 10, 14 14, 10 14, 10 10), (11 11, 12 11, 12 12, 11 12, 11 11)), ((-1 -1, -2 -2, -1 -2, -1 -1)) ) ))");
}

TEST(GeometrySerdeTest, testSmallAreaRing) {
  assertRoundtrip(
      "MULTIPOLYGON (((18.6317421 49.9605785, 18.6318832 49.9607979, 18.6324683 49.9607312, 18.6332842 49.9605658, 18.6332003 49.9603557, 18.6339711 49.9602283, 18.6341994 49.9601905, 18.6343455 49.96016, 18.6344167 49.9601452, 18.6346696 49.9600919, 18.6349643 49.9600567, 18.6352271 49.9601455, 18.6354493 49.9600501, 18.6358024 49.9601071, 18.6358911 49.9600263, 18.6336542 49.9592453, 18.6334794 49.9591838, 18.6337483 49.9581339, 18.6335303 49.9580562, 18.6331284 49.9579122, 18.6324931 49.9576885, 18.6322503 49.9575998, 18.6321381 49.9581593, 18.6321172 49.9582692, 18.6324683 49.9583852, 18.6325255 49.9584004, 18.6327588 49.958489, 18.6324792 49.9588351, 18.6323941 49.9588049, 18.6323261 49.9587807, 18.6320354 49.9586789, 18.6319443 49.9592903, 18.6326731 49.9595648, 18.6331388 49.9594836, 18.6335981 49.959673, 18.6333065 49.9597934, 18.6328096 49.9600844, 18.6330209 49.9601348, 18.633424 49.9602597, 18.6332263 49.960317, 18.6315633 49.9597642, 18.6309331 49.9600741, 18.6317421 49.9605785)), ((18.6298591 49.9606201, 18.6298592 49.96062, 18.6298589 49.9606193, 18.6298591 49.9606201)))");
}
