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

#include <cstdint>
#include <string>
#include <vector>

#include "velox/vector/ComplexVector.h"

namespace facebook::velox::connector::hive::iceberg {

/// Stateless page-level transform that fans MERGE / UPDATE input rows out into
/// delete + insert rows for connectors using the
/// `DELETE_ROW_AND_INSERT_ROW` row-change paradigm (Iceberg). Mirrors the
/// behavior of the OSS Java reference
/// `presto-main-base/.../operator/DeleteAndInsertMergeProcessor.java`.
///
/// Input contract (channels in this fixed order, mirroring
/// `MergeRowChangeProcessor.transformPage`):
///   0. unique_id        BIGINT       (not consumed by this transform)
///   1. target_row_id    ROW          (Iceberg row id — opaque here; copied
///                                      verbatim into the output)
///   2. merge_row        ROW          (target column 0..N-1, then operation
///                                      TINYINT, then case_number INTEGER)
///   3. case_number      INTEGER      (not consumed)
///   4. is_distinct      BOOLEAN      (not consumed)
///
/// The channel indices for target_row_id and merge_row are configurable via
/// the constructor (the unused channels are not consulted at all, so callers
/// may pass any layout that places these two channels at the configured
/// positions).
///
/// Output contract:
///   columns 0..N-1     : target column values, with `targetColumnTypes`
///                        in order. Set to NULL on DELETE rows; copied from
///                        the merge_row's leading fields on INSERT rows.
///   column N           : operation TINYINT — INSERT (1) or DELETE (2).
///                        DEFAULT (-1) rows are dropped before output;
///                        UPDATE (3) is fanned out to a DELETE row + an
///                        INSERT row, so the output operation is always
///                        INSERT or DELETE.
///   column N+1         : rowId, same type as the input row id. Copied
///                        verbatim on DELETE; NULL on INSERT.
///   column N+2         : insert_from_update TINYINT — 1 if this row was
///                        produced as the INSERT half of an UPDATE,
///                        0 otherwise. Matches Java which uses TINYINT
///                        rather than BOOLEAN.
///
/// Total output positions equal `numInsert + numDelete + 2 * numUpdate`
/// where the counts are over the input operation column. Inputs with
/// DEFAULT_CASE are skipped (no output).
class IcebergMergeProcessor {
 public:
  /// Operation byte values defined by `ConnectorMergeSink.java:22-38`. Kept
  /// in sync with the Java side so the per-row dispatch stays compatible
  /// with the coordinator-emitted merge_row payload.
  static constexpr int8_t kInsertOperationNumber = 1;
  static constexpr int8_t kDeleteOperationNumber = 2;
  static constexpr int8_t kUpdateOperationNumber = 3;

  /// DEFAULT_CASE marker defined by
  /// `MergeRowChangeProcessor.java:DEFAULT_CASE_OPERATION_NUMBER = -1`. Rows
  /// with this operation byte are dropped from the output (they correspond
  /// to MERGE WHEN cases whose source row matched no WHEN clause).
  static constexpr int8_t kDefaultCaseOperationNumber = -1;

  /// @param targetColumnTypes Types of the target table data columns, in the
  /// same order as they appear at the leading fields of the merge_row ROW.
  /// The number of entries determines how many leading fields of merge_row
  /// will be consumed for INSERT rows and how many leading columns the
  /// output will carry.
  /// @param outputColumnNames Names of every output column, in order:
  ///   [target columns ...] ++ [operation, row_id, insert_from_update]
  /// Size must equal `targetColumnTypes.size() + 3`. The trailing three
  /// names are taken from the planner's row-id / insert-from-update
  /// variables rather than hardcoded so downstream Velox nodes that bind
  /// by name (TableWriter::setTypeMappings) see the iceberg/planner names
  /// rather than synthetic positional placeholders.
  /// @param rowIdType Type of the Iceberg target row id. Used both to type
  /// the corresponding output column and as the source/destination type when
  /// copying row ids on DELETE rows. Treated as opaque — the inner struct
  /// shape is not inspected.
  /// @param targetRowIdChannel Index in the input RowVector that holds the
  /// row id column. Typically 1, but configurable to keep this transform
  /// independent of the exact upstream channel layout.
  /// @param mergeRowChannel Index in the input RowVector that holds the
  /// merge_row column (a ROW with target column values + operation +
  /// case_number). Typically 2.
  IcebergMergeProcessor(
      std::vector<TypePtr> targetColumnTypes,
      std::vector<std::string> outputColumnNames,
      TypePtr rowIdType,
      column_index_t targetRowIdChannel,
      column_index_t mergeRowChannel);

  /// Returns the output RowType produced by `transform`. Useful for
  /// constructing the consuming operator's expected output type.
  const RowTypePtr& outputType() const {
    return outputType_;
  }

  /// Returns the per-row fan-out of `input`. The returned RowVector is
  /// allocated from `pool` and has type `outputType()`. A zero-position
  /// input returns a zero-position output. `pool` must outlive the returned
  /// vector.
  RowVectorPtr transform(const RowVectorPtr& input, memory::MemoryPool* pool)
      const;

 private:
  // Counts INSERT / DELETE / UPDATE / DEFAULT_CASE positions in
  // `operationVector` over [0, numRows). Throws VELOX_USER_FAIL on any
  // other operation value.
  struct OperationCounts {
    uint64_t numInsert{0};
    uint64_t numDelete{0};
    uint64_t numUpdate{0};
  };

  OperationCounts countOperations(
      const FlatVector<int8_t>& operationVector,
      vector_size_t numRows) const;

  // Allocates the output children for a page of `numRows` positions: the
  // target columns followed by the trailing operation / rowId /
  // insertFromUpdate columns, each sized to `numRows`.
  std::vector<VectorPtr> allocateOutputChildren(
      vector_size_t numRows,
      memory::MemoryPool* pool) const;

  // Builds the output RowType once at construction time:
  //   targetColumnTypes_ ... ++ [TINYINT, rowIdType_, TINYINT].
  // Output names come from `outputColumnNames` (size N+3) in the same
  // order; the trailing three positions still carry the operation /
  // rowId / insertFromUpdate types respectively.
  static RowTypePtr buildOutputType(
      const std::vector<TypePtr>& targetColumnTypes,
      const std::vector<std::string>& outputColumnNames,
      const TypePtr& rowIdType);

  const std::vector<TypePtr> targetColumnTypes_;
  const std::vector<std::string> outputColumnNames_;
  const TypePtr rowIdType_;
  const column_index_t targetRowIdChannel_;
  const column_index_t mergeRowChannel_;
  const size_t numTargetColumns_;
  const RowTypePtr outputType_;
};

} // namespace facebook::velox::connector::hive::iceberg
