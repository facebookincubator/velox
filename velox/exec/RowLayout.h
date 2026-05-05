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

#include <vector>

#include "velox/exec/RowContainer.h"

namespace facebook::velox::exec {

/// Optional flags that control how RowLayout::compute() lays out a row.
///
/// Defaults match a RowContainer constructed with only key types and a
/// memory pool: keys are nullable; no next-row pointer, probed flag,
/// counting-join counter, or normalized-key prefix.
struct RowLayoutOptions {
  /// True if each key column has a null flag bit.
  bool nullableKeys{true};

  /// True if the row carries an inline next-row pointer used to thread
  /// duplicate keys in a hash-join build side.
  bool hasNext{false};

  /// True if the row carries a probed-state bit for full or right outer
  /// join.
  bool hasProbedFlag{false};

  /// True if the row carries a per-row int32_t count, used by counting
  /// joins.
  bool hasCountFlag{false};

  /// True if the row reserves space for a normalized 64-bit key digest in
  /// the slot immediately below the row pointer.
  bool hasNormalizedKeys{false};
};

/// Computed memory layout of a single row inside a RowContainer.
///
/// A row carries its keys, then null flags for keys, accumulators and
/// dependents, then an optional probed bit, then the free bit, then the
/// accumulator payloads, then the dependent fields, then optional uint32_t
/// row size and next-row pointer and counting-join counter slots. Variable
/// width key, dependent and accumulator data is stored out of line through
/// HashStringAllocator and referenced from the row by StringView (for
/// VARCHAR/VARBINARY) or std::string_view (for ARRAY/MAP/ROW). When
/// 'hasNormalizedKeys' is true the row is additionally prefixed with a
/// 64-bit normalized key digest in the slot immediately below the row
/// pointer.
///
/// All fields describe positions within a single row:
///  - Byte offsets are relative to the start of the row (the row pointer
///    returned by RowContainer::newRow()).
///  - Bit offsets ('nullOffsets', 'probedFlagOffset', 'freeFlagOffset') are
///    relative to the start of the row, in bits.
///
/// This is a pure value object. It carries no allocator or runtime state.
/// Use RowLayout::compute() to derive a layout from container parameters
/// without instantiating a RowContainer.
struct RowLayout {
  /// Byte offset of each non-aggregate field. Order is keys, accumulators,
  /// dependents. Size equals the total column count
  /// (keyTypes.size() + accumulators.size() + dependentTypes.size()).
  std::vector<int32_t> offsets;

  /// Bit offset of the null flag for each non-aggregate field. Same order and
  /// size as 'offsets'. When 'options.nullableKeys' is false the entries for
  /// keys are populated for indexing convenience but the bits are unused;
  /// the corresponding RowColumn in 'rowColumns' carries kNotNullOffset.
  std::vector<int32_t> nullOffsets;

  /// Packed (offset, nullByte, nullMask) view per column. Same order and size
  /// as 'offsets'.
  std::vector<RowColumn> rowColumns;

  /// Bit offset of the probed flag for full or right outer join. 0 when not
  /// applicable.
  int32_t probedFlagOffset{0};

  /// Bit offset of the free-row flag.
  int32_t freeFlagOffset{0};

  /// Byte offset of the per-row count for counting joins. 0 when not
  /// applicable.
  int32_t countOffset{0};

  /// Byte offset of the next-row pointer for hash join chains. 0 when keys
  /// are guaranteed unique.
  int32_t nextOffset{0};

  /// Byte offset of the uint32_t variable-row-size counter. 0 when the row
  /// has no variable-width fields and no accumulators.
  int32_t rowSizeOffset{0};

  /// Number of bytes occupied by the flag region (per-column null bits, the
  /// optional probed bit, and the free bit).
  int32_t flagBytes{0};

  /// Total fixed row size in bytes, rounded up to 'alignment'.
  int32_t fixedRowSize{0};

  /// Bytes reserved before each row for the optional normalized key. 0 when
  /// normalized keys are not requested.
  int32_t originalNormalizedKeySize{0};

  /// Row alignment in bytes. Combined from accumulator alignments. Always a
  /// power of two.
  int32_t alignment{1};

  /// True if any accumulator reports usesExternalMemory.
  bool usesExternalMemory{false};

  /// Computes the row layout from container parameters. Pure function: makes
  /// no allocations beyond the returned vectors and reads no global state.
  /// 'keyTypes', 'accumulators' and 'dependentTypes' must outlive the call.
  /// Each accumulator's alignment must be a power of two.
  static RowLayout compute(
      const std::vector<TypePtr>& keyTypes,
      const std::vector<Accumulator>& accumulators,
      const std::vector<TypePtr>& dependentTypes,
      const RowLayoutOptions& options = {});

  /// Returns the in-row byte size of a fixed-width value of the given type
  /// kind.
  static int32_t typeKindSize(TypeKind kind);
};

} // namespace facebook::velox::exec
