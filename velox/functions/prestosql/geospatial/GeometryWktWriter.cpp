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

#include "velox/functions/prestosql/geospatial/GeometryWktWriter.h"

#include <geos/io/WKTWriter.h>

namespace facebook::velox::functions::geospatial {

namespace {

// GEOS's WKTWriter::appendMultiPointText is not virtual, so the MULTIPOINT form
// cannot be corrected by subclassing. Reimplement only MULTIPOINT (and
// GEOMETRYCOLLECTION, to reach a nested MULTIPOINT) and delegate everything
// else, so coordinate number formatting stays identical to the rest of the
// output.

std::string writeWkt(
    geos::io::WKTWriter& writer,
    const geos::geom::Geometry& geometry) {
  return writer.write(&geometry);
}

/// Return the "x y" text of a single point by asking GEOS to write it and
/// taking what is between the outermost parentheses of "POINT (x y)". The
/// input is a Point this function just wrote itself, and coordinate text never
/// contains parentheses, so the span is unambiguous. Callers must handle empty
/// points, for which GEOS writes "POINT EMPTY" and there is nothing to
/// extract.
std::string pointCoordinateText(
    geos::io::WKTWriter& writer,
    const geos::geom::Geometry& point) {
  const std::string text = writeWkt(writer, point);
  const auto open = text.find('(');
  const auto close = text.rfind(')');
  if (open == std::string::npos || close == std::string::npos ||
      close <= open) {
    // Not the expected shape; fall back to GEOS's text rather than corrupt it.
    return text;
  }
  return text.substr(open + 1, close - open - 1);
}

std::string writeWktJtsCompatible(
    geos::io::WKTWriter& writer,
    const geos::geom::Geometry& geometry);

std::string writeMultiPoint(
    geos::io::WKTWriter& writer,
    const geos::geom::Geometry& geometry) {
  // Emptiness is decided by child count, not isEmpty(). GEOS reports isEmpty()
  // for a MULTIPOINT whose every child is empty, but JTS still writes those
  // children: "MULTIPOINT (EMPTY)", not "MULTIPOINT EMPTY".
  if (geometry.getNumGeometries() == 0) {
    return "MULTIPOINT EMPTY";
  }

  std::string result = "MULTIPOINT (";
  for (std::size_t i = 0; i < geometry.getNumGeometries(); ++i) {
    if (i > 0) {
      result += ", ";
    }
    const geos::geom::Geometry* child = geometry.getGeometryN(i);
    if (child->isEmpty()) {
      // JTS writes an empty child as a bare EMPTY, not "(EMPTY)".
      result += "EMPTY";
    } else {
      result += "(";
      result += pointCoordinateText(writer, *child);
      result += ")";
    }
  }
  result += ")";
  return result;
}

std::string writeGeometryCollection(
    geos::io::WKTWriter& writer,
    const geos::geom::Geometry& geometry) {
  // Same reasoning as writeMultiPoint: "GEOMETRYCOLLECTION (MULTIPOINT EMPTY)"
  // is isEmpty() but must keep its child.
  if (geometry.getNumGeometries() == 0) {
    return "GEOMETRYCOLLECTION EMPTY";
  }

  std::string result = "GEOMETRYCOLLECTION (";
  for (std::size_t i = 0; i < geometry.getNumGeometries(); ++i) {
    if (i > 0) {
      result += ", ";
    }
    result += writeWktJtsCompatible(writer, *geometry.getGeometryN(i));
  }
  result += ")";
  return result;
}

std::string writeWktJtsCompatible(
    geos::io::WKTWriter& writer,
    const geos::geom::Geometry& geometry) {
  switch (geometry.getGeometryTypeId()) {
    case geos::geom::GEOS_MULTIPOINT:
      return writeMultiPoint(writer, geometry);
    case geos::geom::GEOS_GEOMETRYCOLLECTION:
      return writeGeometryCollection(writer, geometry);
    default:
      return writeWkt(writer, geometry);
  }
}

} // namespace

std::string writeWktPrestoCompatible(const geos::geom::Geometry& geometry) {
  geos::io::WKTWriter writer;
  writer.setTrim(true);
  return writeWktJtsCompatible(writer, geometry);
}

} // namespace facebook::velox::functions::geospatial
