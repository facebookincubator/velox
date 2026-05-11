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

// Build isolation layer for s2geometry. s2geometry's malloc_extension.h
// and folly's MallocImpl.h both define `nallocx` with incompatible types
// (function declaration vs function pointer). This header exposes S2
// operations through primitive types so that no translation unit needs
// to include both s2geometry and folly headers.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace geos::geom {
class Geometry;
} // namespace geos::geom

namespace facebook::velox::functions {

/// Result of an S2 covering operation. Empty 'error' indicates success.
struct S2Covering {
  std::vector<int64_t> cellIds;
  std::string error;
};

/// Wraps s2geometry operations, isolated from folly headers.
/// All methods are static — no instance state. Cell IDs are int64_t
/// (reinterpreted from unsigned S2 cell IDs) to match the SQL BIGINT type.
///
/// Unless otherwise noted, methods that accept a cell ID require it to be
/// valid (as determined by isValid()). Passing an invalid cell ID results
/// in undefined behavior. Callers should validate via isValid() first.
struct S2CellOp {
  /// Returns true if 'cellId' is a valid S2 cell identifier.
  /// Safe to call with any int64_t value.
  static bool isValid(int64_t cellId);

  /// Returns the area of the cell in square kilometers.
  static double areaSqKm(int64_t cellId);

  /// Returns the level (0–30) of the cell.
  static int level(int64_t cellId);

  /// Returns the ancestor cell at the given level.
  static int64_t parent(int64_t cellId, int level);

  /// Returns true if 'parent' contains 'child'.
  static bool contains(int64_t parent, int64_t child);

  /// Converts a hex token string to a cell identifier. The returned cell ID
  /// may be invalid if the token is malformed; callers should check with
  /// isValid().
  static int64_t fromToken(std::string_view token);

  /// Converts a cell identifier to a hex token string.
  static std::string toToken(int64_t cellId);

  /// Constructs a valid cell ID by starting from a face cell (0–5) and
  /// descending to the given level (0–30). At each level, the next child
  /// (0–3) is selected from successive 2-bit groups of 'position'
  /// (bits [1:0] for level 1, bits [3:2] for level 2, etc.). Any uint64_t
  /// value is safe — extra bits beyond 2*level are ignored.
  static int64_t
  cellIdFromFacePositionLevel(int face, uint64_t position, int level);

  /// Converts a GEOS geometry to S2 objects and returns the cell covering.
  /// GEOS uses planar (straight-line) edges; S2 uses geodesic (great-circle)
  /// edges. For small geometries the difference is negligible; for large ones
  /// (country-sized) the S2 covering follows Earth's curvature.
  static S2Covering tryCovering(
      const geos::geom::Geometry& geom,
      int32_t level);

  /// Returns a compact mixed-level covering, similar to
  /// geometry_to_dissolved_bing_tiles. Uses large cells for interiors and
  /// small cells for boundaries.
  static S2Covering tryDissolvedCovering(
      const geos::geom::Geometry& geom,
      int32_t minLevel,
      int32_t maxLevel,
      int32_t maxCells);
};

} // namespace facebook::velox::functions
