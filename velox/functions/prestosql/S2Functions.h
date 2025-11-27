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

#include <s2/s2cell.h>
#include <s2/s2cell_id.h>
#include <s2/s2earth.h>
#include <string>
#include "velox/functions/Macros.h"

namespace facebook::velox::functions {

template <typename TExec>
struct S2CellIDAreaSqKmFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename R, typename S>
  void call(R& out, const S& cellIdStr) {
    if (cellIdStr.empty()) {
      out = 0.0;
      return;
    }

    try {
      uint64_t cellIdValue = std::stoull(cellIdStr);
      S2CellId cellId(cellIdValue);

      if (!cellId.is_valid()) {
        VELOX_USER_FAIL("S2_CELL_AREA_SQ_KM: Invalid cell ID: {}", cellIdStr);
      }

      S2Cell cell(cellId);

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
      out = "";
      return;
    }

    if (level < 0 || level > 30) {
      VELOX_USER_FAIL(
          "S2_CELL_TOKEN_PARENT: Expected level {} to be in [0,30] range",
          level);
    }

    try {
      std::string cellTokenStr(cellToken.data(), cellToken.size());
      S2CellId cellId = S2CellId::FromToken(cellTokenStr);

      if (!cellId.is_valid()) {
        VELOX_USER_FAIL(
            "S2_CELL_TOKEN_PARENT: Invalid cell token: {}", cellToken);
      }

      if (cellId.level() <= level) {
        out = cellToken;
        return;
      }

      S2CellId parentCellId = cellId.parent(level);

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
      out = "";
      return;
    }

    if (level < 0 || level > 30) {
      VELOX_USER_FAIL(
          "S2_CELL_ID_PARENT: Expected level {} to be in [0,30] range", level);
    }

    try {
      uint64_t cellIdValue = std::stoull(cellIdStr);
      S2CellId cellId(cellIdValue);

      if (!cellId.is_valid()) {
        VELOX_USER_FAIL("S2_CELL_ID_PARENT: Invalid cell ID: {}", cellIdStr);
      }

      if (cellId.level() <= level) {
        out = cellIdStr;
        return;
      }

      S2CellId parentCellId = cellId.parent(level);

      out = std::to_string(parentCellId.id());
    } catch (const std::exception& e) {
      VELOX_USER_FAIL(
          "S2_CELL_ID_PARENT: Error processing cell ID {}: {}",
          cellIdStr,
          e.what());
    }
  }
};

} // namespace facebook::velox::functions
