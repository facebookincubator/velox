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
#include "velox/experimental/wave/common/Cuda.h"

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <limits>
#include "velox/experimental/torchwave/Cat.h"
#include "velox/experimental/torchwave/KernelParams.h"
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

void noteGroupBase(
    ExecutionState& state,
    const AllocGroup* group,
    const at::Tensor& buffer) {
  auto it = state.groupBaseSlot.find(group);
  if (it == state.groupBaseSlot.end()) {
    return;
  }
  setAddress(state, it->second, reinterpret_cast<int64_t>(buffer.data_ptr()));
  const int64_t bytes = buffer.numel() * buffer.element_size();
  setAddress(state, state.groupCapacityBase + it->second, bytes);
  if (static_cast<size_t>(it->second) < state.groupHostBytes.size()) {
    state.groupHostBytes[it->second] = bytes;
  }
}

void poisonGroupBytes(const at::Tensor& buffer) {
  if (!WaveConfig::get().poisonGroups || !buffer.defined() ||
      buffer.numel() == 0) {
    return;
  }
  auto err = cudaMemsetAsync(buffer.data_ptr(), 0xff, buffer.numel(), nullptr);
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
  err = cudaStreamSynchronize(nullptr);
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

namespace {

at::Tensor allocateGroupBytes(int64_t bytes) {
  ScopedAllocCall timed;
  auto buffer = at::empty(
      {bytes}, at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
  poisonGroupBytes(buffer);
  return buffer;
}

// True when no thread outside the execution still has work queued on the
// slot of 'entry'. Only a release during the execution can still be pending
// here: beginGroupArena orders the execution after the earlier ones.
bool releaseSettled(GroupArena::Entry& entry) {
  if (!entry.releasePending) {
    return true;
  }
  if (!entry.releaseEvent->query()) {
    return false;
  }
  entry.releasePending = false;
  return true;
}

// Frees the slot of 'entry' for the next carve. A release outside the
// execution on the arena records the releasing thread's torch stream first.
// Returns false, leaving the slot held, when the record fails, e.g. on a
// thread without a device at process exit.
bool releaseSlot(GroupArena& arena, GroupArena::Entry& entry) {
  if (arena.executing && std::this_thread::get_id() == arena.executingThread) {
    entry.inUse = false;
    return true;
  }
  try {
    if (entry.releaseEvent == nullptr) {
      entry.releaseEvent = std::make_unique<facebook::velox::wave::Event>();
    }
    entry.releaseEvent->record(torchStream());
  } catch (const std::exception& e) {
    LOG(WARNING) << "Group arena: slot stays held, no release event: "
                 << e.what();
    return false;
  }
  entry.releasePending = true;
  entry.inUse = false;
  return true;
}

// The buffer for one carve of 'group': a slot of the arena when one is free,
// else an allocation of its own. Either way its release is recorded.
at::Tensor
groupBuffer(ExecutionState& state, const AllocGroup* group, int64_t bytes) {
  if (group == nullptr || !WaveConfig::get().groupArena || bytes == 0 ||
      bytes >= WaveConfig::get().groupArenaMaxBytes) {
    auto buffer = allocateGroupBytes(bytes);
    noteGroupBase(state, group, buffer);
    return buffer;
  }
  if (state.groupArena == nullptr) {
    state.groupArena = std::make_shared<GroupArena>();
  }
  auto arena = state.groupArena;
  std::lock_guard<std::mutex> lock(arena->mutex);
  auto [it, inserted] = arena->index.try_emplace(
      group, static_cast<int32_t>(arena->entries.size()));
  const auto idx = it->second;
  if (inserted) {
    arena->entries.emplace_back();
    arena->changed = true;
  }
  auto& entry = arena->entries[idx];
  if (bytes > entry.bytes) {
    entry.bytes = bytes;
    arena->changed = true;
  }
  const int64_t carve = ++arena->sequence;
  entry.start = carve;
  entry.end = -1;
  entry.generation = arena->generation;
  bool fromArena = entry.offset >= 0 && !entry.inUse &&
      bytes <= entry.capacity && arena->buffer.defined() &&
      releaseSettled(entry);
  for (size_t i = 0; fromArena && i < entry.conflicts.size(); ++i) {
    auto& conflict = arena->entries.at(entry.conflicts[i]);
    fromArena = !conflict.inUse && releaseSettled(conflict);
  }
  at::Tensor own;
  void* data = nullptr;
  if (fromArena) {
    entry.inUse = true;
    entry.inUseCarve = carve;
    data = static_cast<uint8_t*>(arena->buffer.data_ptr()) + entry.offset;
    ++arena->hits;
  } else {
    own = allocateGroupBytes(bytes);
    data = own.data_ptr();
    ++arena->misses;
  }
  // The deleter keeps the arena's buffer, or the group's own one, alive until
  // the last reference to the group goes.
  auto keep = fromArena ? arena->buffer : own;
  auto buffer = at::from_blob(
      data,
      {bytes},
      [arena, idx, keep, fromArena, carve](void*) {
        // The entry may describe a later carve by now, or another group at a
        // recycled address after a recompile. Only this carve's own state
        // changes.
        std::lock_guard<std::mutex> lock(arena->mutex);
        auto& released = arena->entries[idx];
        if (fromArena && released.inUse && released.inUseCarve == carve) {
          releaseSlot(*arena, released);
        }
        if (released.start == carve) {
          released.end = ++arena->sequence;
          released.endGeneration = arena->generation;
        }
      },
      at::TensorOptions().dtype(at::kByte).device(keep.device()));
  noteGroupBase(state, group, buffer);
  return buffer;
}

} // namespace

void allocateDeviceGroup(
    ExecutionState& state,
    const AllocGroup& group,
    int32_t sequenceNumber,
    int32_t stepIdx) {
  const auto slot = state.groupBaseSlot.at(&group);
  const auto demand = state.recordedDemand.at(slot);
  TORCH_CHECK(demand >= 0, "No recorded demand to size group slot: ", slot);
  // An empty group still gets memory: a kernel sized by a broadcast-free
  // count can read an empty operand's base, which must not be null.
  const auto bytes = std::max<int64_t>(
      alignAllocSize(1),
      alignAllocSize(
          demand + demand * WaveConfig::get().deviceAllocMarginPct / 100));
  auto buffer = groupBuffer(state, &group, bytes);
  // Not a host size, so not checked against the demand.
  state.groupHostBytes.at(slot) = -1;
  if (group.concat != nullptr) {
    setAddress(
        state,
        state.valueAddressBase + group.concat->resultId,
        reinterpret_cast<int64_t>(buffer.data_ptr()));
  }
  const auto owner = addAllocation(state, std::move(buffer));
  auto own = [&](nativert::ValueId id) {
    if (id < 0) {
      return;
    }
    releaseOwner(state, id);
    state.valueCold[id].owner = owner;
    ++state.allocations[owner].liveValues;
  };
  for (auto id : group.actualIds) {
    own(id);
  }
  if (group.concat != nullptr) {
    own(group.concat->resultId);
  }
  const auto numSlots = state.recordedDemand.size();
  if (state.deviceGroupStep.size() < numSlots) {
    state.deviceGroupStep.resize(numSlots, {-1, -1});
    state.deviceGroupOwner.resize(numSlots, -1);
    state.deviceGroupOf.resize(numSlots, nullptr);
  }
  state.deviceGroupStep.at(slot) = {sequenceNumber, stepIdx};
  state.deviceGroupOwner.at(slot) = owner;
  state.deviceGroupOf.at(slot) = &group;
}

void regrowDeviceGroup(ExecutionState& state, int32_t slot, int64_t demand) {
  const auto owner = state.deviceGroupOwner.at(slot);
  const auto* group = state.deviceGroupOf.at(slot);
  TORCH_CHECK(
      owner >= 0 && group != nullptr &&
          state.allocations.at(owner).buffer.defined(),
      "Group slot ",
      slot,
      " failed its capacity check but holds no allocation");
  const auto bytes = std::max<int64_t>(
      alignAllocSize(1),
      alignAllocSize(
          demand + demand * WaveConfig::get().deviceAllocMarginPct / 100));
  // A slot of the packed arena cannot grow in place, since its neighbors are
  // live: the group gets a buffer of its own for the rest of the execution,
  // and the next execution lays the arena out for the new demand.
  auto buffer = allocateGroupBytes(bytes);
  if (state.groupArena != nullptr) {
    auto& arena = *state.groupArena;
    auto it = arena.index.find(group);
    if (it != arena.index.end() && arena.entries[it->second].bytes < bytes) {
      arena.entries[it->second].bytes = bytes;
      arena.changed = true;
    }
  }
  noteGroupBase(state, group, buffer);
  state.groupHostBytes.at(slot) = -1;
  if (group->concat != nullptr) {
    setAddress(
        state,
        state.valueAddressBase + group->concat->resultId,
        reinterpret_cast<int64_t>(buffer.data_ptr()));
  }
  auto& recorded = state.recordedDemand.at(slot);
  recorded = std::max(recorded, demand);
  // The values that own the old buffer now own this one. The old one held
  // only what the failed setup placed, which nothing reads.
  state.allocations.at(owner).buffer = std::move(buffer);
}

void setAddress(ExecutionState& state, int32_t slot, int64_t address) {
  TORCH_CHECK(
      slot >= 0 && slot < state.addressPinned.numel(),
      "Address table slot out of range: ",
      slot);
  state.addressPinned.data_ptr<int64_t>()[slot] = address;
  state.addressDirtyBegin = std::min(state.addressDirtyBegin, slot);
  state.addressDirtyEnd = std::max(state.addressDirtyEnd, slot + 1);
}

void copyAddresses(
    ExecutionState& state,
    facebook::velox::wave::Stream& stream) {
  if (state.addressDirtyBegin < state.addressDirtyEnd) {
    const auto begin = state.addressDirtyBegin;
    stream.hostToDeviceAsync(
        state.addressTable.data_ptr<int64_t>() + begin,
        state.addressPinned.data_ptr<int64_t>() + begin,
        (state.addressDirtyEnd - begin) * sizeof(int64_t));
    state.addressDirtyBegin = std::numeric_limits<int32_t>::max();
    state.addressDirtyEnd = 0;
  }
  if (state.shapeDirtyBegin < state.shapeDirtyEnd) {
    const auto first = state.shapeDirtyBegin;
    stream.hostToDeviceAsync(
        state.shapeTable.data_ptr<int64_t>() + first,
        state.shapePinned.data_ptr<int64_t>() + first,
        (state.shapeDirtyEnd - first) * sizeof(int64_t));
    state.shapeDirtyBegin = std::numeric_limits<int32_t>::max();
    state.shapeDirtyEnd = 0;
  }
}

void setValueShape(
    ExecutionState& state,
    int32_t valueId,
    const uint8_t* param) {
  if (valueId < 0 ||
      valueId >= static_cast<int32_t>(state.valueShapeSlot.size()) ||
      state.valueShapeSlot[valueId] < 0) {
    return;
  }
  const auto first = state.valueShapeSlot[valueId] * 2 * kMaxDims;
  const auto* tensor = reinterpret_cast<const Tensor*>(param);
  auto* words = state.shapePinned.data_ptr<int64_t>() + first;
  for (int32_t d = 0; d < kMaxDims; ++d) {
    words[d] = tensor->dims[d];
    words[kMaxDims + d] = tensor->strides[d];
  }
  state.shapeDirtyBegin = std::min(state.shapeDirtyBegin, first);
  state.shapeDirtyEnd = std::max(state.shapeDirtyEnd, first + 2 * kMaxDims);
}

void beginGroupArena(ExecutionState& state) {
  auto* arena = state.groupArena.get();
  if (arena == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(arena->mutex);
  // Slots released by other threads or streams since the last execution may
  // still be read by work queued there. Every step of this execution is
  // ordered after 'state.stream'.
  for (auto& entry : arena->entries) {
    if (!entry.releasePending) {
      continue;
    }
    if (state.stream != nullptr) {
      entry.releaseEvent->wait(*state.stream);
    } else {
      entry.releaseEvent->wait();
    }
    entry.releasePending = false;
  }
  arena->executing = true;
  arena->executingThread = std::this_thread::get_id();
  // A group released in the generation it was carved in, after its carve, has
  // an interval; the others get no slot.
  struct Item {
    int32_t entry;
    int64_t start;
    int64_t end;
    int64_t bytes;
  };
  std::vector<Item> items;
  bool eligibilityChanged = false;
  for (int32_t i = 0; i < static_cast<int32_t>(arena->entries.size()); ++i) {
    const auto& entry = arena->entries[i];
    const bool eligible = entry.generation == arena->generation &&
        entry.endGeneration == arena->generation && entry.end > entry.start;
    if (eligible != (entry.offset >= 0)) {
      eligibilityChanged = true;
    }
    if (eligible) {
      items.push_back({i, entry.start, entry.end, alignAllocSize(entry.bytes)});
    }
  }
  ++arena->generation;
  arena->hits = 0;
  arena->misses = 0;
  if (!arena->changed && !eligibilityChanged) {
    return;
  }
  arena->changed = false;
  // Largest first, each at the lowest offset clear of every placed item whose
  // interval overlaps its own.
  std::sort(items.begin(), items.end(), [](const Item& lhs, const Item& rhs) {
    return lhs.bytes != rhs.bytes ? lhs.bytes > rhs.bytes
                                  : lhs.start < rhs.start;
  });
  std::vector<int64_t> offsets(items.size(), 0);
  std::vector<std::pair<int64_t, int64_t>> busy;
  int64_t total = 0;
  for (size_t i = 0; i < items.size(); ++i) {
    busy.clear();
    for (size_t j = 0; j < i; ++j) {
      const auto& placed = items.at(j);
      if (placed.start < items.at(i).end && items.at(i).start < placed.end) {
        busy.emplace_back(offsets.at(j), offsets.at(j) + placed.bytes);
      }
    }
    std::sort(busy.begin(), busy.end());
    int64_t offset = 0;
    for (const auto& [begin, end] : busy) {
      if (offset + items[i].bytes <= begin) {
        break;
      }
      offset = std::max(offset, end);
    }
    offsets[i] = offset;
    total = std::max(total, offset + items[i].bytes);
  }
  for (auto& entry : arena->entries) {
    entry.offset = -1;
    entry.capacity = 0;
    entry.conflicts.clear();
  }
  for (size_t i = 0; i < items.size(); ++i) {
    auto& entry = arena->entries[items[i].entry];
    entry.offset = offsets[i];
    entry.capacity = items[i].bytes;
    for (size_t j = 0; j < items.size(); ++j) {
      if (j != i && offsets[j] < offsets[i] + items[i].bytes &&
          offsets[i] < offsets[j] + items[j].bytes) {
        entry.conflicts.push_back(items[j].entry);
      }
    }
  }
  // A slot still held from the old layout may overlap any new one, so the new
  // layout gets a buffer of its own; the deleters keep the old one alive.
  bool held = false;
  for (const auto& entry : arena->entries) {
    held = held || entry.inUse;
  }
  if (held || !arena->buffer.defined() || arena->buffer.numel() < total) {
    arena->buffer = at::Tensor();
    if (total > 0) {
      arena->buffer = allocateGroupBytes(total);
    }
  }
}

void endGroupArena(ExecutionState& state) {
  auto* arena = state.groupArena.get();
  if (arena == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(arena->mutex);
  arena->executing = false;
}

namespace {

// A group larger than the free device memory may be sized from values a step
// on the device path never made: a step that finds a group over its capacity
// fails, and the steps after it do not run, but the host still sizes the steps
// off the path from what they would have written. Names the failed step
// instead of running out of memory. A restart (recoverFailedSetup) issues the
// device launches again but cannot redo host sizing that already read their
// results, so a group sized this way stays an error.
// cudaMemGetInfo takes a driver lock that other processes hold for tens of
// milliseconds at a time, so it is asked only for a group that is a sizable
// part of the device.
void checkFailedBeforeGroup(ExecutionState& state, int64_t bytes) {
  static const int64_t kDeviceBytes = [] {
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    return cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess
        ? static_cast<int64_t>(totalBytes)
        : std::numeric_limits<int64_t>::max();
  }();
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  if (!state.failureWords.defined() || bytes <= kDeviceBytes / 8 ||
      cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess ||
      bytes <= static_cast<int64_t>(freeBytes)) {
    return;
  }
  int32_t failed{kNoFailedStep};
  state.stream->deviceToHostAsync(
      &failed, state.failureWords.data_ptr<int32_t>(), sizeof(failed));
  state.stream->wait();
  TORCH_CHECK(
      failed == kNoFailedStep,
      "Step ",
      failed,
      " of the execution needs more memory than its allocation group was "
      "given, so the sizes of the steps after it are not valid; a restart "
      "cannot redo host sizing that read them. The host was sizing a ",
      bytes,
      " byte group at step ",
      state.stepOrdinal);
}

} // namespace

AllocGroupBuffer materializeAllocGroupTable(
    const std::vector<AllocRequest>& requests,
    nativert::ExecutionFrame& frame,
    ExecutionState& state,
    const AllocGroup* group) {
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
  checkFailedBeforeGroup(state, result.totalBytes);
  result.base = groupBuffer(state, group, result.totalBytes);
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

void takeOwner(
    ExecutionState& state,
    nativert::ValueId id,
    nativert::ValueId source) {
  // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
  auto& out = state.valueCold[id];
  out.owner = source >= 0 ? state.valueCold[source].owner : -1;
  if (out.owner < 0 && source >= 0) {
    const auto& base = state.frame->getIValue(source);
    if (base.isTensor()) {
      out.owner = addAllocation(state, base.toTensor());
    }
  }
  if (out.owner >= 0) {
    ++state.allocations[out.owner].liveValues;
  }
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
  TORCH_CHECK(
      owner >= 0 || frame.getIValue(baseId).isTensor(),
      "Materializing %",
      id,
      " over %",
      baseId,
      ", which has no owner and no frame tensor");
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
  // The setup places the result, and the bands in it, from its address.
  if (layout.resultId < static_cast<int32_t>(state.hasValueAddress.size()) &&
      state.hasValueAddress[layout.resultId] != 0) {
    setAddress(
        state,
        state.valueAddressBase + layout.resultId,
        reinterpret_cast<int64_t>(buffer.base.data_ptr()));
  }
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
