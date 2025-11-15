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

#include "velox/functions/prestosql/S2Functions.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"

// Include S2 geometry library
#include <s2/s2cell.h>
#include <s2/s2cell_id.h>
#include <s2/s2earth.h>

namespace facebook::velox::functions {
namespace {

template <typename TExec>
struct S2CellIDAreaSqKmFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename R, typename S>
  void call(R& out, const S& cellIdStr) {
    if (cellIdStr.empty()) {
      // Return null if the cell ID is empty
      out = 0.0;
      return;
    }

    try {
      // Parse the S2CellId from the string
      uint64_t cellIdValue = std::stoull(cellIdStr);
      S2CellId cellId(cellIdValue);

      // Check if the cell ID is valid
      if (!cellId.is_valid()) {
        VELOX_USER_FAIL("S2_CELL_AREA_SQ_KM: Invalid cell ID: {}", cellIdStr);
      }

      // Create an S2Cell from the cell ID
      S2Cell cell(cellId);

      // Calculate the area in square kilometers
      double areaSqKm = S2Earth::SteradiansToSquareKm(cell.ExactArea());

      out = areaSqKm;
    } catch (const std::exception& e) {
      VELOX_USER_FAIL(
          "S2_CELL_AREA_SQ_KM: Error processing cell ID {}: {}",
          cellIdStr,
          e.what());
    }
  }
};

template <typename TExec>
struct S2CellTokenParentFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename R, typename S, typename I>
  void call(R& out, const S& cellToken, const I& level) {
    if (cellToken.empty()) {
      // Return null if the cell token is empty
      out = "";
      return;
    }

    if (level < 0 || level > 30) {
      VELOX_USER_FAIL(
          "S2_CELL_TOKEN_PARENT: Expected level {} to be in [0,30] range",
          level);
    }

    try {
      // Parse the S2CellId from the token
      std::string cellTokenStr(cellToken.data(), cellToken.size());
      S2CellId cellId = S2CellId::FromToken(cellTokenStr);

      // Check if the cell ID is valid
      if (!cellId.is_valid()) {
        VELOX_USER_FAIL(
            "S2_CELL_TOKEN_PARENT: Invalid cell token: {}", cellToken);
      }

      if (cellId.level() <= level) {
        // Return the original token if the level is too high
        out = cellToken;
        return;
      }

      // Get the parent at the specified level
      S2CellId parentCellId = cellId.parent(level);

      // Convert the parent cell ID back to a token
      out = parentCellId.ToToken();
    } catch (const std::exception& e) {
      VELOX_USER_FAIL(
          "S2_CELL_TOKEN_PARENT: Error processing cell token {}: {}",
          cellToken,
          e.what());
    }
  }
};

template <typename TExec>
struct S2CellIdParentFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename R, typename S, typename I>
  void call(R& out, const S& cellIdStr, const I& level) {
    if (cellIdStr.empty()) {
      // Return null if the cell ID is empty
      out = "";
      return;
    }

    if (level < 0 || level > 30) {
      VELOX_USER_FAIL(
          "S2_CELL_ID_PARENT: Expected level {} to be in [0,30] range", level);
    }

    try {
      // Parse the S2CellId from the string
      uint64_t cellIdValue = std::stoull(cellIdStr);
      S2CellId cellId(cellIdValue);

      // Check if the cell ID is valid
      if (!cellId.is_valid()) {
        VELOX_USER_FAIL("S2_CELL_ID_PARENT: Invalid cell ID: {}", cellIdStr);
      }

      if (cellId.level() <= level) {
        // Return the original token if the level is too high
        out = cellIdStr;
        return;
      }

      // Get the parent at the specified level
      S2CellId parentCellId = cellId.parent(level);

      // Convert the parent cell ID back to a string
      out = std::to_string(parentCellId.id());
    } catch (const std::exception& e) {
      VELOX_USER_FAIL(
          "S2_CELL_ID_PARENT: Error processing cell ID {}: {}",
          cellIdStr,
          e.what());
    }
  }
};

} // namespace

void registerS2CellIdParent(const std::string& prefix) {
  // Register the function with the signature (varchar, integer) -> varchar
  registerFunction<S2CellIdParentFunction, Varchar, Varchar, int32_t>(
      {prefix + "s2_cell_id_parent"});
}

void registerS2CellAreaSqKm(const std::string& prefix) {
  // Register the function with the signature (varchar) -> double
  registerFunction<S2CellIDAreaSqKmFunction, double, Varchar>(
      {prefix + "s2_cell_id_area_sq_km"});
}

void registerS2CellTokenParent(const std::string& prefix) {
  // Register the function with the signature (varchar, integer) -> varchar
  registerFunction<S2CellTokenParentFunction, Varchar, Varchar, int32_t>(
      {prefix + "s2_cell_token_parent"});
}

} // namespace facebook::velox::functions
