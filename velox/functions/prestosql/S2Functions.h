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

#include <string>
#include <string_view>

#include <folly/CPortability.h>
#include <folly/Conv.h>
#include <s2/s2cell.h>
#include <s2/s2cell_id.h>
#include <s2/s2earth.h>
#include "velox/common/base/Status.h"
#include "velox/functions/Macros.h"

namespace facebook::velox::functions {

template <typename TExec>
struct S2CellIDAreaSqKmFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<double>& result, const arg_type<Varchar>& cellIdStr) {
    if (cellIdStr.empty()) {
      if (threadSkipErrorDetails()) {
        return Status::UserError();
      }
      return Status::UserError("S2_CELL_AREA_SQ_KM: Empty cell ID");
    }

    std::string_view cellIdView(cellIdStr.data(), cellIdStr.size());

    auto cellIdResult = folly::tryTo<uint64_t>(cellIdView);
    if (cellIdResult.hasError()) {
      if (threadSkipErrorDetails()) {
        return Status::UserError();
      }
      return Status::UserError(
          "S2_CELL_AREA_SQ_KM: Invalid cell ID: {}", cellIdView);
    }
    uint64_t cellIdValue = cellIdResult.value();

    S2CellId cellId(cellIdValue);

    if (!cellId.is_valid()) {
      if (threadSkipErrorDetails()) {
        return Status::UserError();
      }
      return Status::UserError(
          "S2_CELL_AREA_SQ_KM: Invalid cell ID: {}", cellIdView);
    }

    S2Cell cell(cellId);
    result = S2Earth::SteradiansToSquareKm(cell.ExactArea());
    return Status::OK();
  }
};

template <typename TExec>
struct S2CellTokenParentFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Varchar>& result,
      const arg_type<Varchar>& cellToken,
      int level) {
    if (cellToken.empty()) {
      if (threadSkipErrorDetails()) {
        return Status::UserError();
      }
      return Status::UserError("S2_CELL_TOKEN_PARENT: Empty cell token");
    }

    if (level < 0 || level > 30) {
      if (threadSkipErrorDetails()) {
        return Status::UserError();
      }
      return Status::UserError(
          "S2_CELL_TOKEN_PARENT: Expected level {} to be in [0,30] range",
          level);
    }

    std::string_view cellTokenView(cellToken.data(), cellToken.size());
    S2CellId cellId = S2CellId::FromToken(cellTokenView);

    if (!cellId.is_valid()) {
      if (threadSkipErrorDetails()) {
        return Status::UserError();
      }
      return Status::UserError(
          "S2_CELL_TOKEN_PARENT: Invalid cell token: {}", cellTokenView);
    }

    if (cellId.level() <= level) {
      result = cellToken;
      return Status::OK();
    }

    S2CellId parentCellId = cellId.parent(level);
    result = parentCellId.ToToken();
    return Status::OK();
  }
};

template <typename TExec>
struct S2CellIdParentFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<int64_t>& result,
      const arg_type<Varchar>& cellIdStr,
      int level) {
    if (cellIdStr.empty()) {
      if (threadSkipErrorDetails()) {
        return Status::UserError();
      }
      return Status::UserError("S2_CELL_ID_PARENT: Empty cell ID");
    }

    if (level < 0 || level > 30) {
      if (threadSkipErrorDetails()) {
        return Status::UserError();
      }
      return Status::UserError(
          "S2_CELL_ID_PARENT: Expected level {} to be in [0,30] range", level);
    }

    std::string_view cellIdView(cellIdStr.data(), cellIdStr.size());

    auto cellIdResult = folly::tryTo<uint64_t>(cellIdView);
    if (cellIdResult.hasError()) {
      if (threadSkipErrorDetails()) {
        return Status::UserError();
      }
      return Status::UserError(
          "S2_CELL_ID_PARENT: Invalid cell ID: {}", cellIdView);
    }
    uint64_t cellIdValue = cellIdResult.value();

    S2CellId cellId(cellIdValue);

    if (!cellId.is_valid()) {
      if (threadSkipErrorDetails()) {
        return Status::UserError();
      }
      return Status::UserError(
          "S2_CELL_ID_PARENT: Invalid cell ID: {}", cellIdView);
    }

    if (cellId.level() <= level) {
      result = static_cast<int64_t>(cellIdValue);
      return Status::OK();
    }

    S2CellId parentCellId = cellId.parent(level);
    result = static_cast<int64_t>(parentCellId.id());
    return Status::OK();
  }
};

} // namespace facebook::velox::functions
