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

#include "velox/experimental/torchwave/ValueTable.h"

#include <cstddef>
#include <cstring>
#include <sstream>

#include <ATen/ATen.h>

#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"

namespace torch::wave {

void writeTensorFields(c10::TensorImpl* impl, TableTensor* t) {
  const auto sizes = impl->sizes();
  const auto strides = impl->strides();
  const auto rank = static_cast<int32_t>(sizes.size());
  TORCH_CHECK(
      rank <= kMaxDims,
      "Tensors with more than ",
      kMaxDims,
      " dims not supported, got ",
      rank);
  t->storage = impl->mutable_data();
  t->rank = static_cast<int8_t>(rank);
  t->elementSize = static_cast<uint8_t>(impl->itemsize());
  t->elementType = static_cast<uint8_t>(impl->dtype().toScalarType());
  for (int i = 0; i < kMaxDims; ++i) {
    t->dims[i] = i < rank ? static_cast<int32_t>(sizes[i]) : 0;
    t->strides[i] =
        i < rank ? (sizes[i] == 1 ? 0 : static_cast<int32_t>(strides[i])) : 0;
  }
  t->numEl = static_cast<uint32_t>(impl->numel());
}

std::string tensorToString(const Tensor& t) {
  std::stringstream ss;
  ss << "Tensor{storage=" << t.storage << " rank=" << static_cast<int>(t.rank)
     << " dims=[";
  for (int i = 0; i < t.rank; ++i) {
    if (i > 0) {
      ss << ",";
    }
    ss << t.dims[i];
  }
  ss << "] strides=[";
  for (int i = 0; i < t.rank; ++i) {
    if (i > 0) {
      ss << ",";
    }
    ss << t.strides[i];
  }
  ss << "] numEl=" << t.numEl << " status=" << t.status << "}";
  return ss.str();
}

std::string tensorToString(const TableTensor& t) {
  Tensor full;
  std::memcpy(&full, &t, sizeof(TableTensor));
  return tensorToString(full);
}

void ensureTableSize(ExecutionState& state, size_t size) {
  if (size <= state.valueEntries.size()) {
    return;
  }
  state.valueEntries.resize(size);
  state.valueCold.resize(size);
  state.valueTable = ValueTable{
      .entries = state.valueEntries.data(),
      .cold = state.valueCold.data(),
      .size = static_cast<int32_t>(size),
      .generation = state.tableGeneration};
}

namespace {
void publishValue(nativert::ValueId id, const c10::IValue& value) {
  auto* state = currentExecutionState();
  if (state == nullptr || !state->tableActive || id < 0) {
    return;
  }
  ensureTableSize(*state, static_cast<size_t>(id) + 1);
  auto& entry = state->valueEntries[id];
  if (value.isTensor() && value.toTensor().defined()) {
    state->valueCold[id].impl = value.toTensor().unsafeGetTensorImpl();
    entry.kind = ValueEntry::kTensorPending;
  } else if (value.isInt()) {
    entry.scalar = value.toInt();
    entry.kind = ValueEntry::kInt;
  } else if (value.isSymInt() && value.toSymInt().maybe_as_int().has_value()) {
    entry.scalar = *value.toSymInt().maybe_as_int();
    entry.kind = ValueEntry::kInt;
  } else {
    entry.kind = ValueEntry::kOther;
  }
  entry.generation = state->tableGeneration;
}
} // namespace

void writeFrame(
    nativert::ExecutionFrame& frame,
    nativert::ValueId id,
    c10::IValue value) {
  ++threadSetupCounters().frameWrites;
  publishValue(id, value);
  frame.setIValue(id, std::move(value));
}

c10::IValue debugValue(
    const nativert::ExecutionFrame& frame,
    nativert::ValueId id) {
  const auto& value = frame.getIValue(id);
  const auto* state = currentExecutionState();
  if (!value.isNone() || state == nullptr || !state->tableActive ||
      !state->valueTable.valid(id)) {
    return value;
  }
  const auto& entry = state->valueEntries[id];
  if (entry.kind == ValueEntry::kInt) {
    return c10::IValue(entry.scalar);
  }
  const auto owner = state->valueCold[id].owner;
  if (entry.kind != ValueEntry::kTensor || owner < 0 ||
      entry.t.storage == nullptr) {
    return value;
  }
  const auto& base = state->allocations[owner].buffer;
  const auto& t = entry.t;
  int64_t sizes[kMaxDims];
  int64_t strides[kMaxDims];
  for (int32_t i = 0; i < t.rank; ++i) {
    sizes[i] = t.dims[i];
    strides[i] = t.strides[i];
  }
  const auto* start = static_cast<const char*>(base.storage().data());
  return aliasTensor(
      base,
      c10::IntArrayRef(sizes, t.rank),
      c10::IntArrayRef(strides, t.rank),
      (static_cast<const char*>(t.storage) - start) / t.elementSize,
      static_cast<c10::ScalarType>(t.elementType));
}

void publishFromFrame(nativert::ExecutionFrame& frame, nativert::ValueId id) {
  const auto* state = currentExecutionState();
  if (state == nullptr || !state->tableActive) {
    return;
  }
  publishValue(id, frame.getIValue(id));
}

void unpublish(nativert::ValueId id) {
  auto* state = currentExecutionState();
  if (state != nullptr && id >= 0 &&
      static_cast<size_t>(id) < state->valueEntries.size()) {
    state->valueEntries[id].generation = 0;
  }
}

void materializeTableTensor(const ValueTable& table, int32_t id) {
  auto& entry = table.entries[id];
  writeTensorFields(table.cold[id].impl, &entry.t);
  entry.kind = ValueEntry::kTensor;
}

} // namespace torch::wave
