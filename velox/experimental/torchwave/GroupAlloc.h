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

#include "velox/experimental/torchwave/AllocGroup.h"
#include "velox/experimental/torchwave/Cat.h"
#include "velox/experimental/torchwave/Executor.h"

namespace torch::wave {

/// Adds 'buffer' to the state's allocation records and returns its index,
/// with no live values yet.
int32_t addAllocation(ExecutionState& state, at::Tensor buffer);

/// Drops the hold the table entry of 'id' has on its allocation; the buffer
/// goes when its last value is released.
void releaseOwner(ExecutionState& state, nativert::ValueId id);

/// Drops every allocation record. For the end of an execution, after the
/// streams are drained.
void releaseAllAllocations(ExecutionState& state);

/// Writes the entry of 'id' as a contiguous tensor of 'dims' and 'dtype' at
/// 'storage', held by allocation 'owner'.
void writeMemberEntry(
    ExecutionState& state,
    nativert::ValueId id,
    void* storage,
    const std::vector<int64_t>& dims,
    c10::ScalarType dtype,
    int32_t owner);

/// Allocates a tensor of 'dims' and 'dtype' for 'id' as an allocation of its
/// own and writes only its table entry.
void allocateTableOnly(
    ExecutionState& state,
    nativert::ValueId id,
    c10::IntArrayRef dims,
    c10::ScalarType dtype);

/// Writes a table entry with 'dims' and 'dtype' and no storage, for a group
/// member sized before its group is carved.
void writeShapeEntry(
    ExecutionState& state,
    nativert::ValueId id,
    c10::IntArrayRef dims,
    c10::ScalarType dtype);

/// Carves a lifetime group like materializeAllocGroup, except that a member
/// in ExecutionState::tableOnly gets only its table entry. Its frame slot
/// keeps the shape placeholder the sizing left there.
AllocGroupBuffer materializeAllocGroupTable(
    const std::vector<AllocRequest>& requests,
    nativert::ExecutionFrame& frame,
    ExecutionState& state);

/// Whether this execution already gave 'id' storage of 'dims', as a launch of
/// an earlier step that also outputs it does.
bool holdsTensorOfDims(
    const ExecutionState& state,
    nativert::ValueId id,
    c10::IntArrayRef dims);

/// Whether 'id' lives in the table alone this execution.
bool isTableOnly(const ExecutionState& state, nativert::ValueId id);

/// Gives the frame a tensor (or integer) for the table entry of 'id'. A
/// tensor's storage is in its owner's buffer, or else in the frame tensor of
/// 'baseId'.
void materializeEntry(
    ExecutionState& state,
    nativert::ValueId id,
    nativert::ValueId baseId);

/// materializeEntry for 'id' unless it lives in the table alone. 'id' must
/// have an owner.
void materializeIfHostRead(ExecutionState& state, nativert::ValueId id);

/// The allocation holding the concat result 'id' of 'dims': the one this
/// execution already gave it at that shape, else a new one.
int32_t concatResultOwner(
    ExecutionState& state,
    nativert::ValueId id,
    c10::IntArrayRef dims,
    c10::ScalarType dtype);

/// Writes the entry of 'id' as the band of the concat result 'resultId' that
/// starts at 'start' along the join axis and spans 'extent' (a stack band
/// drops the axis), held by the result's allocation.
void writeBandEntry(
    ExecutionState& state,
    nativert::ValueId id,
    nativert::ValueId resultId,
    const ConcatSpec& spec,
    int64_t start,
    int64_t extent);

/// Lays out a concat group over the table: allocates the result and gives the
/// carved and copied operands the table entries of their bands, with frame
/// tensors only for values host code reads. An operand the group does not
/// carve is measured from its size expression or, if its own reserve sizes it,
/// from its table entry. Adds the carved
/// operands to 'numCarved'.
AllocGroupBuffer materializeConcatGroupTable(
    const ConcatGroupLayout& layout,
    const std::vector<AllocRequest>& requests,
    const std::vector<bool>& sized,
    ExecutionState& state,
    int32_t& numCarved);

/// True when every operand of 'layout' can be measured now: a carved member
/// the step has sized, or an operand whose reserve result or size-expression
/// inputs are already in the table this execution. An operand written by a
/// launch that runs after the step's wait is not, until that launch is sized.
bool concatGroupMeasurable(
    const ConcatGroupLayout& layout,
    const std::vector<bool>& sized,
    const ExecutionState& state);

} // namespace torch::wave
