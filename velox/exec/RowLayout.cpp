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

#include "velox/exec/RowLayout.h"

#include "velox/common/base/BitUtil.h"

namespace facebook::velox::exec {
namespace {

template <TypeKind Kind>
int32_t kindSize() {
  return sizeof(typename KindToFlatVector<Kind>::HashRowType);
}

} // namespace

// static
int32_t RowLayout::typeKindSize(TypeKind kind) {
  if (kind == TypeKind::UNKNOWN) {
    return sizeof(UnknownValue);
  }
  return VELOX_DYNAMIC_TYPE_DISPATCH(kindSize, kind);
}

// static
RowLayout RowLayout::compute(
    const std::vector<TypePtr>& keyTypes,
    const std::vector<Accumulator>& accumulators,
    const std::vector<TypePtr>& dependentTypes,
    const RowLayoutOptions& options) {
  RowLayout layout;

  int32_t offset = 0;
  int32_t flagOffset = 0;
  bool isVariableWidth = false;
  for (const auto& type : keyTypes) {
    layout.offsets.push_back(offset);
    offset += typeKindSize(type->kind());
    layout.nullOffsets.push_back(flagOffset);
    isVariableWidth |= !type->isFixedWidth();
    if (options.nullableKeys) {
      ++flagOffset;
    }
  }
  // Make offset at least sizeof pointer so that there is space for a free
  // list next pointer below the bit at 'freeFlagOffset'.
  offset = std::max<int32_t>(offset, sizeof(void*));
  const int32_t firstAggregateOffset = offset;
  if (!accumulators.empty()) {
    // Move flagOffset to the start of the next byte. This guarantees the
    // null and initialized bits for an aggregate always appear in the same
    // byte.
    flagOffset = (flagOffset + 7) & -8;
  }
  for (const auto& accumulator : accumulators) {
    // Null bit.
    layout.nullOffsets.push_back(flagOffset);
    // Increment for two bits: null bit and following initialized bit.
    flagOffset += RowContainer::kNumAccumulatorFlags;
    isVariableWidth |= !accumulator.isFixedSize();
    layout.usesExternalMemory |= accumulator.usesExternalMemory();
    layout.alignment = RowContainer::combineAlignments(
        accumulator.alignment(), layout.alignment);
  }
  for (const auto& type : dependentTypes) {
    layout.nullOffsets.push_back(flagOffset);
    ++flagOffset;
    isVariableWidth |= !type->isFixedWidth();
  }
  if (options.hasProbedFlag) {
    layout.probedFlagOffset = flagOffset + firstAggregateOffset * 8;
    ++flagOffset;
  }
  // Free flag.
  layout.freeFlagOffset = flagOffset + firstAggregateOffset * 8;
  ++flagOffset;
  // 'flagOffset' is now the total number of flag bits; convert to bytes.
  layout.flagBytes = bits::nbytes(flagOffset);
  // Until this point each entry in 'nullOffsets' is a bit offset relative to
  // the start of the flag region. Shift them to be bit offsets from the row
  // start so RowColumn callers can read them directly.
  for (int32_t i = 0; i < layout.nullOffsets.size(); ++i) {
    layout.nullOffsets[i] += firstAggregateOffset * 8;
  }
  offset += layout.flagBytes;
  for (const auto& accumulator : accumulators) {
    // Each accumulator's payload is aligned to its declared alignment.
    offset = bits::roundUp(offset, accumulator.alignment());
    layout.offsets.push_back(offset);
    offset += accumulator.fixedWidthSize();
  }
  for (const auto& type : dependentTypes) {
    layout.offsets.push_back(offset);
    offset += typeKindSize(type->kind());
  }
  if (isVariableWidth) {
    layout.rowSizeOffset = offset;
    offset += sizeof(uint32_t);
  }
  if (options.hasNext) {
    layout.nextOffset = offset;
    offset += sizeof(void*);
  }
  if (options.hasCountFlag) {
    layout.countOffset = offset;
    offset += sizeof(int32_t);
  }
  layout.fixedRowSize = bits::roundUp(offset, layout.alignment);
  layout.originalNormalizedKeySize = options.hasNormalizedKeys
      ? bits::roundUp(sizeof(normalized_key_t), layout.alignment)
      : 0;

  size_t nullOffsetsPos = 0;
  for (size_t i = 0; i < layout.offsets.size(); ++i) {
    layout.rowColumns.emplace_back(
        layout.offsets[i],
        (options.nullableKeys || i >= keyTypes.size())
            ? layout.nullOffsets[nullOffsetsPos]
            : RowColumn::kNotNullOffset);
    ++nullOffsetsPos;
  }
  return layout;
}

} // namespace facebook::velox::exec
