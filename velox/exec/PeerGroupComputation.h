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

#include "velox/common/base/Exceptions.h"
#include "velox/vector/TypeAliases.h"

#include <algorithm>
#include <utility>

namespace facebook::velox::exec {

/// Computes peer group bounds over storage-specific window partition rows.
class PeerGroupComputation {
 public:
  struct Result {
    vector_size_t peerStart;
    vector_size_t peerEnd;
    bool previousRowConsumed;
  };

  /// Computes peer start and end bounds for rows in [start, end).
  ///
  /// RowAccessor contract:
  /// - partitionEnd() returns the absolute end row of the available partition.
  /// - hasPreviousRow() returns true if there is a retained row immediately
  ///   before 'start' for cross-batch peer comparison.
  /// - previousRowEquals(row) compares the retained previous row with 'row'.
  /// - rowsEqual(lhs, rhs) compares two absolute partition rows.
  ///
  /// The accessor is read-only. If the previous row is consumed, the result
  /// sets previousRowConsumed=true and the caller remains responsible for
  /// clearing or releasing that state.
  template <typename RowAccessor>
  static Result compute(
      const RowAccessor& rows,
      vector_size_t start,
      vector_size_t end,
      vector_size_t prevPeerStart,
      vector_size_t prevPeerEnd,
      vector_size_t* rawPeerStarts,
      vector_size_t* rawPeerEnds) {
    VELOX_CHECK_LE(start, rows.partitionEnd());
    VELOX_CHECK_LE(end, rows.partitionEnd());

    auto peerStart = prevPeerStart;
    auto peerEnd = prevPeerEnd;
    vector_size_t next = start;
    vector_size_t index = 0;
    bool previousRowConsumed = false;

    if (rows.hasPreviousRow() && start < end) {
      const auto samePeer = rows.previousRowEquals(start);
      previousRowConsumed = true;
      if (samePeer) {
        peerEnd = findEnd(rows, start, rows.partitionEnd());
        for (; next < std::min(end, peerEnd); ++next, ++index) {
          rawPeerStarts[index] = peerStart;
          rawPeerEnds[index] = peerEnd - 1;
        }
      }
    }

    for (; next < end; ++next, ++index) {
      if (next == 0 || next >= peerEnd) {
        peerStart = next;
        peerEnd = findEnd(rows, peerStart, rows.partitionEnd());
      }

      rawPeerStarts[index] = peerStart;
      rawPeerEnds[index] = peerEnd - 1;
    }

    VELOX_CHECK_EQ(index, end - start);
    return {peerStart, peerEnd, previousRowConsumed};
  }

 private:
  template <typename RowAccessor>
  static vector_size_t findEnd(
      const RowAccessor& rows,
      vector_size_t peerStart,
      vector_size_t partitionEnd) {
    auto peerEnd = peerStart + 1;
    while (peerEnd < partitionEnd && rows.rowsEqual(peerStart, peerEnd)) {
      ++peerEnd;
    }
    return peerEnd;
  }
};

} // namespace facebook::velox::exec
