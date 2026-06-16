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

#include <string_view>

#include <folly/CPortability.h>
#include "velox/common/base/Status.h"
#include "velox/common/geospatial/GeometrySerde.h"
#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/S2CellOperations.h"
#include "velox/functions/prestosql/types/GeometryType.h"

namespace facebook::velox::functions {

namespace detail {

template <typename... Args>
Status userError(std::string_view format, Args&&... args) {
  if (threadSkipErrorDetails()) {
    return Status::UserError();
  }
  return Status::UserError(format, std::forward<Args>(args)...);
}

inline Status validateCellId(int64_t cellId, std::string_view functionName) {
  if (!S2CellOp::isValid(cellId)) {
    return userError("{}: Invalid cell ID: {}", functionName, cellId);
  }
  return Status::OK();
}

inline Status validateLevel(int32_t level, std::string_view functionName) {
  if (level < 0 || level > 30) {
    return userError(
        "{}: Level must be in [0, 30] range, got {}", functionName, level);
  }
  return Status::OK();
}

} // namespace detail

/// Implements the s2_cell_area_sq_km SQL function.
template <typename TExec>
struct S2CellAreaSqKmFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE Status
  call(double& result, const arg_type<int64_t>& cellId) {
    VELOX_RETURN_NOT_OK(detail::validateCellId(cellId, "s2_cell_area_sq_km"));
    result = S2CellOp::areaSqKm(cellId);
    return Status::OK();
  }
};

/// Implements the s2_cell_contains SQL function.
template <typename TExec>
struct S2CellContainsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE Status call(
      bool& result,
      const arg_type<int64_t>& parent,
      const arg_type<int64_t>& child) {
    VELOX_RETURN_NOT_OK(detail::validateCellId(parent, "s2_cell_contains"));
    VELOX_RETURN_NOT_OK(detail::validateCellId(child, "s2_cell_contains"));
    result = S2CellOp::contains(parent, child);
    return Status::OK();
  }
};

/// Implements the s2_cell_from_token SQL function.
template <typename TExec>
struct S2CellFromTokenFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE Status
  call(int64_t& result, const arg_type<Varchar>& token) {
    if (token.empty()) {
      return detail::userError("s2_cell_from_token: Empty cell token");
    }
    result = S2CellOp::fromToken({token.data(), token.size()});
    if (!S2CellOp::isValid(result)) {
      return detail::userError(
          "s2_cell_from_token: Invalid cell token: {}",
          std::string_view(token.data(), token.size()));
    }
    return Status::OK();
  }
};

/// Implements the s2_cell_level SQL function.
template <typename TExec>
struct S2CellLevelFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE Status
  call(int32_t& result, const arg_type<int64_t>& cellId) {
    VELOX_RETURN_NOT_OK(detail::validateCellId(cellId, "s2_cell_level"));
    result = S2CellOp::level(cellId);
    return Status::OK();
  }
};

/// Implements the s2_cell_parent SQL function.
template <typename TExec>
struct S2CellParentFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE Status call(
      int64_t& result,
      const arg_type<int64_t>& cellId,
      const arg_type<int32_t>& level) {
    VELOX_RETURN_NOT_OK(detail::validateLevel(level, "s2_cell_parent"));
    VELOX_RETURN_NOT_OK(detail::validateCellId(cellId, "s2_cell_parent"));

    if (S2CellOp::level(cellId) <= level) {
      result = cellId;
      return Status::OK();
    }
    result = S2CellOp::parent(cellId, level);
    return Status::OK();
  }
};

/// Implements the s2_cell_to_token SQL function.
template <typename TExec>
struct S2CellToTokenFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Varchar>& result, const arg_type<int64_t>& cellId) {
    VELOX_RETURN_NOT_OK(detail::validateCellId(cellId, "s2_cell_to_token"));
    result = S2CellOp::toToken(cellId);
    return Status::OK();
  }
};

/// Implements the fixed-level s2_cells(geometry, level) SQL function.
template <typename TExec>
struct S2CellsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Array<int64_t>>& result,
      const arg_type<Geometry>& geometry,
      const arg_type<int32_t>& level) {
    VELOX_RETURN_NOT_OK(detail::validateLevel(level, "s2_cells"));

    auto geom =
        common::geospatial::GeometryDeserializer::deserializeNonEmpty(geometry);
    if (!geom) {
      return Status::OK();
    }
    auto covering = S2CellOp::tryCovering(*geom, level);
    if (!covering.error.empty()) {
      return detail::userError("s2_cells: {}", covering.error);
    }
    for (auto cellId : covering.cellIds) {
      result.add_item() = cellId;
    }
    return Status::OK();
  }
};

/// Implements the dissolved s2_cells(geometry, minLevel, maxLevel, maxCells)
/// SQL function.
template <typename TExec>
struct S2CellsDissolvedFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Array<int64_t>>& result,
      const arg_type<Geometry>& geometry,
      const arg_type<int32_t>& minLevel,
      const arg_type<int32_t>& maxLevel,
      const arg_type<int32_t>& maxCells) {
    VELOX_RETURN_NOT_OK(detail::validateLevel(minLevel, "s2_cells"));
    VELOX_RETURN_NOT_OK(detail::validateLevel(maxLevel, "s2_cells"));
    if (minLevel > maxLevel) {
      return detail::userError(
          "s2_cells: min_level ({}) must be <= max_level ({})",
          minLevel,
          maxLevel);
    }
    if (maxCells < 1) {
      return detail::userError(
          "s2_cells: max_cells must be >= 1, got {}", maxCells);
    }

    auto geom =
        common::geospatial::GeometryDeserializer::deserializeNonEmpty(geometry);
    if (!geom) {
      return Status::OK();
    }
    auto covering =
        S2CellOp::tryDissolvedCovering(*geom, minLevel, maxLevel, maxCells);
    if (!covering.error.empty()) {
      return detail::userError("s2_cells: {}", covering.error);
    }
    for (auto cellId : covering.cellIds) {
      result.add_item() = cellId;
    }
    return Status::OK();
  }
};

} // namespace facebook::velox::functions
