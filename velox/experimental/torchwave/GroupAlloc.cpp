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

// The Group allocation op of the bound execution path: one buffer for a set
// of values, each member's table entry written in place, a frame tensor only
// for a member host code reads.

#include "velox/experimental/torchwave/GroupAlloc.h"

#include "velox/experimental/torchwave/Cat.h"
#include "velox/experimental/torchwave/Utils.h"

namespace torch::wave {

int32_t addAllocation(ExecutionState& state, at::Tensor buffer) {
  int32_t index;
  if (!state.freeAllocations.empty()) {
    index = state.freeAllocations.back();
    state.freeAllocations.pop_back();
  } else {
    index = static_cast<int32_t>(state.allocations.size());
    state.allocations.emplace_back();
  }
  state.allocations[index] = {.buffer = std::move(buffer), .liveValues = 0};
  return index;
}

void releaseOwner(ExecutionState& state, nativert::ValueId id) {
  if (id < 0 || static_cast<size_t>(id) >= state.valueEntries.size()) {
    return;
  }
  auto& entry = state.valueCold[id];
  if (entry.owner < 0) {
    return;
  }
  auto& allocation = state.allocations[entry.owner];
  if (--allocation.liveValues == 0) {
    allocation.buffer = at::Tensor();
    state.freeAllocations.push_back(entry.owner);
  }
  entry.owner = -1;
}

void releaseAllAllocations(ExecutionState& state) {
  for (auto& entry : state.valueCold) {
    entry.owner = -1;
  }
  state.allocations.clear();
  state.freeAllocations.clear();
}

void writeMemberEntry(
    ExecutionState& state,
    nativert::ValueId id,
    void* storage,
    const std::vector<int64_t>& dims,
    c10::ScalarType dtype,
    int32_t owner) {
  auto& entry = state.valueEntries[id];
  auto& t = entry.t;
  const auto rank = static_cast<int32_t>(dims.size());
  t.storage = storage;
  t.rank = static_cast<int8_t>(rank);
  t.elementSize = static_cast<uint8_t>(c10::elementSize(dtype));
  t.elementType = static_cast<uint8_t>(dtype);
  int64_t stride = 1;
  for (int32_t d = kMaxDims - 1; d >= 0; --d) {
    if (d >= rank) {
      t.dims[d] = 0;
      t.strides[d] = 0;
      continue;
    }
    t.dims[d] = static_cast<int32_t>(dims[d]);
    t.strides[d] = dims[d] == 1 ? 0 : static_cast<int32_t>(stride);
    stride *= dims[d];
  }
  t.numEl = static_cast<uint32_t>(stride);
  entry.kind = ValueEntry::kTensor;
  entry.generation = state.tableGeneration;
  state.valueCold[id].owner = owner;
  ++state.allocations[owner].liveValues;
}

void allocateTableOnly(
    ExecutionState& state,
    nativert::ValueId id,
    c10::IntArrayRef dims,
    c10::ScalarType dtype) {
  at::Tensor tensor;
  {
    ScopedAllocCall timed;
    tensor =
        at::empty(dims, at::TensorOptions().dtype(dtype).device(at::kCUDA));
  }
  auto* storage = tensor.data_ptr();
  const auto owner = addAllocation(state, std::move(tensor));
  writeMemberEntry(
      state,
      id,
      storage,
      std::vector<int64_t>(dims.begin(), dims.end()),
      dtype,
      owner);
}

void writeShapeEntry(
    ExecutionState& state,
    nativert::ValueId id,
    c10::IntArrayRef dims,
    c10::ScalarType dtype) {
  auto& entry = state.valueEntries[id];
  auto& t = entry.t;
  const auto rank = static_cast<int32_t>(dims.size());
  t.storage = nullptr;
  t.rank = static_cast<int8_t>(rank);
  t.elementSize = static_cast<uint8_t>(c10::elementSize(dtype));
  t.elementType = static_cast<uint8_t>(dtype);
  int64_t stride = 1;
  for (int32_t d = kMaxDims - 1; d >= 0; --d) {
    if (d >= rank) {
      t.dims[d] = 0;
      t.strides[d] = 0;
      continue;
    }
    t.dims[d] = static_cast<int32_t>(dims[d]);
    t.strides[d] = dims[d] == 1 ? 0 : static_cast<int32_t>(stride);
    stride *= dims[d];
  }
  t.numEl = static_cast<uint32_t>(stride);
  entry.kind = ValueEntry::kTensor;
  entry.generation = state.tableGeneration;
  state.valueCold[id].owner = -1;
}

AllocGroupBuffer materializeAllocGroupTable(
    const std::vector<AllocRequest>& requests,
    nativert::ExecutionFrame& frame,
    ExecutionState& state) {
  AllocGroupBuffer result;
  result.offsets.reserve(requests.size());
  int64_t cursor = 0;
  for (const auto& request : requests) {
    result.offsets.push_back(cursor);
    int64_t numel = 1;
    for (auto d : request.dims) {
      numel *= d;
    }
    cursor += alignAllocSize(
        numel * static_cast<int64_t>(c10::elementSize(request.dtype)));
  }
  result.totalBytes = cursor;
  {
    ScopedAllocCall timed;
    result.base = at::empty(
        {result.totalBytes},
        at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
  }
  result.slots.assign(requests.size(), at::Tensor());
  auto* base = static_cast<uint8_t*>(result.base.data_ptr());
  int32_t owner = -1;
  std::vector<int64_t> strides;
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto& request = requests[i];
    const auto id = request.actualId;
    const auto rank = static_cast<int32_t>(request.dims.size());
    if (id >= 0 && static_cast<size_t>(id) < state.tableOnly.size() &&
        state.tableOnly[id] != 0 && rank <= kMaxDims) {
      if (owner < 0) {
        owner = addAllocation(state, result.base);
      }
      writeMemberEntry(
          state,
          id,
          base + result.offsets[i],
          request.dims,
          request.dtype,
          owner);
      continue;
    }
    strides.resize(rank);
    int64_t stride = 1;
    for (int32_t d = rank; d-- > 0;) {
      strides[d] = stride;
      stride *= request.dims[d];
    }
    const auto elementSize =
        static_cast<int64_t>(c10::elementSize(request.dtype));
    auto slot = aliasTensor(
        result.base,
        request.dims,
        strides,
        result.offsets[i] / elementSize,
        request.dtype);
    writeFrame(frame, id, slot);
    result.slots[i] = std::move(slot);
  }
  return result;
}

bool holdsTensorOfDims(
    const ExecutionState& state,
    nativert::ValueId id,
    c10::IntArrayRef dims) {
  const auto& table = state.valueTable;
  if (!table.valid(id) || !table.isTensor(id)) {
    return false;
  }
  const auto& t = table.tensor(id);
  if (t.storage == nullptr || t.rank != static_cast<int32_t>(dims.size())) {
    return false;
  }
  for (int32_t i = 0; i < t.rank; ++i) {
    if (t.dims[i] != dims[i]) {
      return false;
    }
  }
  return true;
}

bool isTableOnly(const ExecutionState& state, nativert::ValueId id) {
  return state.tableOnlyActive && id >= 0 &&
      static_cast<size_t>(id) < state.tableOnly.size() &&
      state.tableOnly[id] != 0;
}

void materializeEntry(
    ExecutionState& state,
    nativert::ValueId id,
    nativert::ValueId baseId) {
  auto& frame = *state.frame;
  const auto& entry = state.valueEntries[id];
  if (entry.kind == ValueEntry::kInt) {
    writeFrame(frame, id, c10::IValue(entry.scalar));
    return;
  }
  const auto owner = state.valueCold[id].owner;
  const at::Tensor base = owner >= 0 ? state.allocations[owner].buffer
                                     : frame.getIValue(baseId).toTensor();
  const auto& t = entry.t;
  int64_t sizes[kMaxDims];
  int64_t strides[kMaxDims];
  for (int32_t i = 0; i < t.rank; ++i) {
    sizes[i] = t.dims[i];
    strides[i] = t.strides[i];
  }
  const auto* start = static_cast<const char*>(base.storage().data());
  const auto offset =
      (static_cast<const char*>(t.storage) - start) / t.elementSize;
  writeFrame(
      frame,
      id,
      aliasTensor(
          base,
          c10::IntArrayRef(sizes, t.rank),
          c10::IntArrayRef(strides, t.rank),
          offset,
          static_cast<c10::ScalarType>(t.elementType)));
}

int32_t concatResultOwner(
    ExecutionState& state,
    nativert::ValueId id,
    c10::IntArrayRef dims,
    c10::ScalarType dtype) {
  if (holdsTensorOfDims(state, id, dims)) {
    auto& entry = state.valueCold[id];
    if (entry.owner < 0) {
      const auto& existing = state.frame->getIValue(id);
      TORCH_CHECK(
          existing.isTensor(),
          "Concat result %",
          id,
          " is in the table with no owner and no frame tensor");
      entry.owner = addAllocation(state, existing.toTensor());
      ++state.allocations[entry.owner].liveValues;
    }
    return entry.owner;
  }
  releaseOwner(state, id);
  allocateTableOnly(state, id, dims, dtype);
  return state.valueCold[id].owner;
}

void writeBandEntry(
    ExecutionState& state,
    nativert::ValueId id,
    nativert::ValueId resultId,
    const ConcatSpec& spec,
    int64_t start,
    int64_t extent) {
  const auto& result = state.valueEntries[resultId];
  const auto& r = result.t;
  int64_t joinStride = 1;
  for (int32_t d = r.rank - 1; d > spec.dim; --d) {
    joinStride *= r.dims[d];
  }
  const auto owner = state.valueCold[resultId].owner;
  if (owner >= 0) {
    ++state.allocations[owner].liveValues;
  }
  releaseOwner(state, id);
  auto& entry = state.valueEntries[id];
  auto& t = entry.t;
  t = r;
  t.storage =
      static_cast<char*>(r.storage) + start * joinStride * r.elementSize;
  int32_t rank = 0;
  int64_t numEl = 1;
  for (int32_t d = 0; d < r.rank; ++d) {
    if (spec.isStack && d == spec.dim) {
      continue;
    }
    const int64_t size = d == spec.dim ? extent : r.dims[d];
    t.dims[rank] = static_cast<int32_t>(size);
    t.strides[rank] = size == 1
        ? 0
        : static_cast<int32_t>(d == spec.dim ? joinStride : r.strides[d]);
    numEl *= size;
    ++rank;
  }
  for (int32_t d = rank; d < kMaxDims; ++d) {
    t.dims[d] = 0;
    t.strides[d] = 0;
  }
  t.rank = static_cast<int8_t>(rank);
  t.numEl = static_cast<uint32_t>(numEl);
  entry.kind = ValueEntry::kTensor;
  entry.generation = state.tableGeneration;
  state.valueCold[id].owner = owner;
}

void materializeIfHostRead(ExecutionState& state, nativert::ValueId id) {
  if (!isTableOnly(state, id)) {
    materializeEntry(state, id, -1);
  }
}

namespace {

// True when every value 'expr' reads has an entry this execution.
bool sizeExprInTable(const SizeExpr& expr, const ValueTable& table) {
  for (const auto id : expr.values) {
    if (!table.valid(id)) {
      return false;
    }
  }
  for (const auto& arg : expr.args) {
    if (!sizeExprInTable(arg, table)) {
      return false;
    }
  }
  return true;
}

} // namespace

bool concatGroupMeasurable(
    const ConcatGroupLayout& layout,
    const std::vector<bool>& sized,
    const ExecutionState& state) {
  const auto& table = state.valueTable;
  for (size_t i = 0; i < layout.operands.size(); ++i) {
    const auto& operand = layout.operands[i];
    const auto member = layout.memberOfOperand[i];
    if (member >= 0 && sized.at(member)) {
      continue;
    }
    if (operand.hasReserve) {
      if (!table.valid(operand.valueId) || !table.isTensor(operand.valueId)) {
        return false;
      }
    } else if (
        operand.sizeExpr.op != SizeShortcut::kNone &&
        !sizeExprInTable(operand.sizeExpr, table)) {
      return false;
    }
  }
  return true;
}

AllocGroupBuffer materializeConcatGroupTable(
    const ConcatGroupLayout& layout,
    const std::vector<AllocRequest>& requests,
    const std::vector<bool>& sized,
    ExecutionState& state,
    int32_t& numCarved) {
  ScopedSetupPhase phase(SetupPhase::kCarve);
  const ConcatSpec spec{
      .isStack = layout.isStack, .dim = layout.dim, .outRank = layout.outRank};
  const auto elementRank = spec.elementRank();
  std::vector<std::vector<Dim>> operandShapes;
  operandShapes.reserve(layout.operands.size());
  std::vector<bool> carve(layout.operands.size(), false);
  std::vector<Dim> shape;
  for (size_t i = 0; i < layout.operands.size(); ++i) {
    const auto& operand = layout.operands[i];
    const auto member = layout.memberOfOperand[i];
    if (member >= 0 && sized.at(member)) {
      const auto& dims = requests.at(member).dims;
      operandShapes.emplace_back(dims.begin(), dims.end());
      carve[i] = true;
      continue;
    }
    if (operand.hasReserve) {
      TORCH_CHECK(
          state.valueTable.valid(operand.valueId) &&
              state.valueTable.isTensor(operand.valueId),
          "Concat operand %",
          operand.valueId,
          " is sized by its reserve after its group is laid out");
      const auto& t = state.valueTable.tensor(operand.valueId);
      shape.assign(t.dims, t.dims + t.rank);
    } else if (operand.sizeExpr.op != SizeShortcut::kNone) {
      operand.sizeExpr.dims(state.valueTable, shape);
    } else {
      shape.clear();
    }
    operandShapes.push_back(coerceConcatShape(operand, shape, elementRank));
  }

  const auto outShape = concatResultShape(spec, operandShapes);
  const std::vector<int64_t> outSizes(outShape.begin(), outShape.end());
  const auto owner =
      concatResultOwner(state, layout.resultId, outSizes, layout.dtype);

  AllocGroupBuffer buffer;
  buffer.base = state.allocations[owner].buffer;
  buffer.totalBytes = buffer.base.numel() * buffer.base.element_size();
  buffer.slots.assign(requests.size(), at::Tensor());
  buffer.offsets.assign(requests.size(), 0);
  int64_t joinStride = 1;
  for (int32_t d = spec.outRank - 1; d > spec.dim; --d) {
    joinStride *= outShape[d];
  }
  const auto elementSize = static_cast<int64_t>(c10::elementSize(layout.dtype));
  int64_t offset = 0;
  for (size_t i = 0; i < layout.operands.size(); ++i) {
    const auto& operand = layout.operands[i];
    const int64_t extent =
        spec.isStack ? 1 : static_cast<int64_t>(operandShapes[i].at(spec.dim));
    const int64_t start = spec.isStack ? static_cast<int64_t>(i) : offset;
    offset += extent;
    TORCH_CHECK(
        !(carve[i] && operand.copyDestId >= 0),
        "Concat operand %",
        operand.valueId,
        " of result %",
        layout.resultId,
        " is both carved into the result and copied into it");
    const auto id = carve[i] ? operand.valueId : operand.copyDestId;
    if (id < 0) {
      continue;
    }
    writeBandEntry(state, id, layout.resultId, spec, start, extent);
    materializeIfHostRead(state, id);
    if (carve[i]) {
      buffer.offsets[layout.memberOfOperand[i]] =
          start * joinStride * elementSize;
      ++numCarved;
    }
  }
  materializeIfHostRead(state, layout.resultId);
  return buffer;
}

} // namespace torch::wave
