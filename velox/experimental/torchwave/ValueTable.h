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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "velox/experimental/torchwave/KernelParams.h"

namespace c10 {
struct TensorImpl;
} // namespace c10

namespace torch::wave {

/// The host fields of a device Tensor, laid out as its first bytes so a fill
/// copies them with one memcpy. The table does not keep 'status', which it
/// always leaves uninitialized, or 'sizes[]', which the device builds.
struct TableTensor {
  void* storage{nullptr};
  int8_t rank{0};
  uint8_t elementSize{0};
  uint8_t elementType{0};
  int32_t dims[kMaxDims]{};
  int32_t strides[kMaxDims]{};
  uint32_t numEl{0};
};

static_assert(offsetof(TableTensor, dims) == offsetof(Tensor, dims));
static_assert(offsetof(TableTensor, numEl) == offsetof(Tensor, numEl));
static_assert(sizeof(TableTensor) == offsetof(Tensor, status));

/// The metadata of one value in one execution: the single copy of its dims,
/// strides, element type and storage, and its integer if it is a scalar. Valid
/// when 'generation' equals the execution's. One cache line; what the sizing
/// and the fills do not read is in ValueCold.
struct alignas(64) ValueEntry {
  enum Kind : uint8_t {
    /// None, a list, a float: nothing the sizing or the fill reads.
    kOther = 0,
    kTensor = 1,
    kInt = 2,
    /// A tensor whose fields are built on first read from the frame's
    /// TensorImpl (ValueCold::impl).
    kTensorPending = 3,
  };

  enum Flags : uint8_t {
    /// The device sets the dims; they are final after the step's transfer.
    kDeviceShape = 1,
    /// Host code reads the value, so the frame holds an at::Tensor for it.
    kHostRead = 2,
  };

  TableTensor t;
  uint32_t generation{0};
  uint8_t kind{kOther};
  uint8_t flags{0};
  int64_t scalar{0};
};

static_assert(sizeof(ValueEntry) == 64);

/// The rest of a value's entry, parallel to ValueEntry.
struct ValueCold {
  /// The frame's TensorImpl of a kTensorPending entry.
  c10::TensorImpl* impl{nullptr};
  /// The host copy of a tensor's elements, in pinned memory, for a value some
  /// sizing reads the contents of (ReserveInputs::hostTensors). Valid when
  /// 'hostGeneration' equals the table's generation.
  const void* hostData{nullptr};
  uint32_t hostGeneration{0};
  /// Index of the Allocation holding the storage, or -1.
  int32_t owner{-1};
};

/// Writes the descriptor fields of the tensor 'impl' into 't': storage, rank,
/// element size and type, dims, strides (0 for size-1 dims) and numEl.
void writeTensorFields(c10::TensorImpl* impl, TableTensor* t);

/// Renders the host-visible fields of 't' for messages.
std::string tensorToString(const Tensor& t);
std::string tensorToString(const TableTensor& t);

struct ValueTable;

/// Builds the Tensor of a kTensorPending entry from its TensorImpl.
void materializeTableTensor(const ValueTable& table, int32_t id);

/// View of the value table of one execution, indexed by value id. Ids are
/// dense per graph, so the id is the slot.
struct ValueTable {
  ValueEntry* entries{nullptr};
  ValueCold* cold{nullptr};
  int32_t size{0};
  uint32_t generation{0};

  bool valid(int32_t id) const {
    return id >= 0 && id < size && entries[id].generation == generation;
  }

  /// For a valid entry.
  bool isTensor(int32_t id) const {
    return entries[id].kind == ValueEntry::kTensor ||
        entries[id].kind == ValueEntry::kTensorPending;
  }

  /// For a valid entry.
  bool isInt(int32_t id) const {
    return entries[id].kind == ValueEntry::kInt;
  }

  /// For a valid tensor entry.
  const TableTensor& tensor(int32_t id) const {
    if (entries[id].kind == ValueEntry::kTensorPending) {
      materializeTableTensor(*this, id);
    }
    return entries[id].t;
  }

  /// For a valid tensor entry: the entry with its fields built.
  const ValueEntry& entry(int32_t id) const {
    tensor(id);
    return entries[id];
  }

  int64_t scalar(int32_t id) const {
    return entries[id].scalar;
  }
};

} // namespace torch::wave
