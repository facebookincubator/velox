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

#pragma once

#include <geos/geom/Geometry.h>

#include <string>

namespace facebook::velox::functions::geospatial {

/// Write a geometry as WKT in the form Presto's Java engine produces.
///
/// Presto's Java engine writes WKT with org.locationtech.jts.io.WKTWriter,
/// which parenthesizes each child of a MULTIPOINT:
///
///   MULTIPOINT ((0 0), (10 20), (30 40))
///
/// GEOS writes the children bare:
///
///   MULTIPOINT (0 0, 10 20, 30 40)
///
/// Both are legal OGC WKT -- the spec permits either -- but ST_AsText is a
/// user-visible contract, so a query must not change its output when a cluster
/// moves to Prestissimo. This writer follows JTS.
///
/// MULTIPOINT is the only geometry type where GEOS and JTS disagree; every
/// other type is delegated to geos::io::WKTWriter unchanged, including all
/// coordinate number formatting. GEOMETRYCOLLECTION is walked so that a
/// MULTIPOINT nested inside one is also written in the JTS form.
std::string writeWktPrestoCompatible(const geos::geom::Geometry& geometry);

} // namespace facebook::velox::functions::geospatial
