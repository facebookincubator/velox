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

#include "velox/experimental/torchwave/Derive.h"

#include <algorithm>
#include <limits>
#include <optional>

namespace torch::wave {

namespace {

// Dims and strides in elements, editable before they become the output entry.
struct Geometry {
  int32_t rank{0};
  int64_t sizes[kMaxDims]{};
  int64_t strides[kMaxDims]{};
};

Geometry geometryOf(const TableTensor& t) {
  Geometry g;
  g.rank = t.rank;
  for (int32_t i = 0; i < t.rank; ++i) {
    g.sizes[i] = t.dims[i];
    g.strides[i] = t.strides[i];
  }
  return g;
}

void erase(Geometry& g, int32_t dim) {
  for (int32_t i = dim; i + 1 < g.rank; ++i) {
    g.sizes[i] = g.sizes[i + 1];
    g.strides[i] = g.strides[i + 1];
  }
  --g.rank;
}

int32_t wrapDim(int64_t dim, int32_t rank) {
  return static_cast<int32_t>(dim < 0 ? dim + rank : dim);
}

// Reads the operands of one op. A dynamic integer must be an int in the table;
// None reads as nullopt through 'optionalInt'.
class Operands {
 public:
  Operands(const DeriveView& op, const ValueTable& table)
      : op_(op), table_(table) {}

  const TableTensor* tensor(size_t i) const {
    const auto id = op_.operands[i];
    return id >= 0 && table_.valid(id) && table_.isTensor(id)
        ? &table_.tensor(id)
        : nullptr;
  }

  bool integer(size_t i, int64_t& value) const {
    const auto id = op_.operands[i];
    if (id < 0) {
      value = op_.constants[i];
      return true;
    }
    if (!table_.valid(id) || !table_.isInt(id)) {
      return false;
    }
    value = table_.scalar(id);
    return true;
  }

  // False when the operand is not in the table; 'value' empty for None.
  bool optionalInt(size_t i, std::optional<int64_t>& value) const {
    const auto id = op_.operands[i];
    if (id >= 0 && table_.valid(id) &&
        table_.entries[id].kind == ValueEntry::kOther) {
      value.reset();
      return true;
    }
    int64_t result;
    if (!integer(i, result)) {
      return false;
    }
    value = result;
    return true;
  }

 private:
  const DeriveView& op_;
  const ValueTable& table_;
};

void writeTensorEntry(
    const ValueTable& table,
    int32_t output,
    const TableTensor& self,
    const Geometry& g,
    int64_t elementOffset) {
  auto& entry = table.entries[output];
  auto& t = entry.t;
  t.storage =
      static_cast<char*>(self.storage) + elementOffset * self.elementSize;
  t.rank = static_cast<int8_t>(g.rank);
  t.elementSize = self.elementSize;
  t.elementType = self.elementType;
  int64_t numel = 1;
  for (int32_t i = 0; i < kMaxDims; ++i) {
    const bool inRank = i < g.rank;
    t.dims[i] = inRank ? static_cast<int32_t>(g.sizes[i]) : 0;
    t.strides[i] =
        inRank && g.sizes[i] != 1 ? static_cast<int32_t>(g.strides[i]) : 0;
    numel *= inRank ? g.sizes[i] : 1;
  }
  t.numEl = static_cast<uint32_t>(numel);
  entry.kind = ValueEntry::kTensor;
  entry.generation = table.generation;
}

void writeIntEntry(const ValueTable& table, int32_t output, int64_t value) {
  auto& entry = table.entries[output];
  entry.scalar = value;
  entry.kind = ValueEntry::kInt;
  entry.generation = table.generation;
}

bool isContiguous(const Geometry& g) {
  int64_t expected = 1;
  for (int32_t i = g.rank - 1; i >= 0; --i) {
    if (g.sizes[i] != 1 && g.strides[i] != expected) {
      return false;
    }
    expected *= g.sizes[i];
  }
  return true;
}

bool slice(
    const DeriveView& op,
    const ValueTable& table,
    const Operands& args) {
  const auto* self = args.tensor(0);
  int64_t dimArg;
  int64_t step;
  std::optional<int64_t> startArg;
  std::optional<int64_t> endArg;
  if (self == nullptr || self->rank == 0 || !args.integer(1, dimArg) ||
      !args.optionalInt(2, startArg) || !args.optionalInt(3, endArg) ||
      !args.integer(4, step) || step <= 0) {
    return false;
  }
  auto g = geometryOf(*self);
  const auto dim = wrapDim(dimArg, g.rank);
  const auto extent = g.sizes[dim];
  auto start = startArg.value_or(0);
  auto end = endArg.value_or(std::numeric_limits<int64_t>::max());
  if (start < 0) {
    start += extent;
  }
  if (end < 0) {
    end += extent;
  }
  start = std::clamp<int64_t>(start, 0, extent);
  end = std::clamp<int64_t>(end, start, extent);
  const auto offset = start * g.strides[dim];
  g.sizes[dim] = (end - start + step - 1) / step;
  g.strides[dim] *= step;
  writeTensorEntry(table, op.output, *self, g, offset);
  return true;
}

bool select(
    const DeriveView& op,
    const ValueTable& table,
    const Operands& args) {
  const auto* self = args.tensor(0);
  int64_t dimArg;
  int64_t index;
  if (self == nullptr || self->rank == 0 || !args.integer(1, dimArg) ||
      !args.integer(2, index)) {
    return false;
  }
  auto g = geometryOf(*self);
  const auto dim = wrapDim(dimArg, g.rank);
  const auto extent = g.sizes[dim];
  if (index < 0) {
    index += extent;
  }
  TORCH_CHECK_INDEX(
      index >= 0 && index < extent, "select(): index out of range: ", index);
  const auto offset = index * g.strides[dim];
  erase(g, dim);
  writeTensorEntry(table, op.output, *self, g, offset);
  return true;
}

bool narrow(
    const DeriveView& op,
    const ValueTable& table,
    const Operands& args) {
  const auto* self = args.tensor(0);
  int64_t dimArg;
  int64_t start;
  int64_t length;
  if (self == nullptr || self->rank == 0 || !args.integer(1, dimArg) ||
      !args.integer(2, start) || !args.integer(3, length)) {
    return false;
  }
  auto g = geometryOf(*self);
  const auto dim = wrapDim(dimArg, g.rank);
  const auto extent = g.sizes[dim];
  if (start < 0) {
    start += extent;
  }
  TORCH_CHECK(
      length >= 0 && start >= 0 && start <= extent - length,
      "narrow(): range out of bounds: start ",
      start,
      " length ",
      length,
      " size ",
      extent);
  const auto offset = start * g.strides[dim];
  g.sizes[dim] = length;
  writeTensorEntry(table, op.output, *self, g, offset);
  return true;
}

bool squeezeDim(
    const DeriveView& op,
    const ValueTable& table,
    const Operands& args) {
  const auto* self = args.tensor(0);
  int64_t dimArg;
  if (self == nullptr || !args.integer(1, dimArg)) {
    return false;
  }
  auto g = geometryOf(*self);
  if (g.rank > 0) {
    const auto dim = wrapDim(dimArg, g.rank);
    if (g.sizes[dim] == 1) {
      erase(g, dim);
    }
  }
  writeTensorEntry(table, op.output, *self, g, 0);
  return true;
}

bool unsqueeze(
    const DeriveView& op,
    const ValueTable& table,
    const Operands& args) {
  const auto* self = args.tensor(0);
  int64_t dimArg;
  if (self == nullptr || !args.integer(1, dimArg) || self->rank >= kMaxDims) {
    return false;
  }
  auto g = geometryOf(*self);
  const auto dim = wrapDim(dimArg, g.rank + 1);
  for (int32_t i = g.rank; i > dim; --i) {
    g.sizes[i] = g.sizes[i - 1];
    g.strides[i] = g.strides[i - 1];
  }
  g.sizes[dim] = 1;
  g.strides[dim] = 0;
  ++g.rank;
  writeTensorEntry(table, op.output, *self, g, 0);
  return true;
}

bool transpose(
    const DeriveView& op,
    const ValueTable& table,
    const Operands& args) {
  const auto* self = args.tensor(0);
  int64_t first;
  int64_t second;
  if (self == nullptr || !args.integer(1, first) || !args.integer(2, second)) {
    return false;
  }
  auto g = geometryOf(*self);
  if (g.rank > 0) {
    const auto a = wrapDim(first, g.rank);
    const auto b = wrapDim(second, g.rank);
    std::swap(g.sizes[a], g.sizes[b]);
    std::swap(g.strides[a], g.strides[b]);
  }
  writeTensorEntry(table, op.output, *self, g, 0);
  return true;
}

bool expand(
    const DeriveView& op,
    const ValueTable& table,
    const Operands& args) {
  const auto* self = args.tensor(0);
  const auto& target = op.intList;
  if (self == nullptr || target.empty() ||
      static_cast<int32_t>(target.size()) < self->rank ||
      target.size() > kMaxDims) {
    return false;
  }
  const auto source = geometryOf(*self);
  Geometry g;
  g.rank = static_cast<int32_t>(target.size());
  const auto lead = g.rank - source.rank;
  for (int32_t d = 0; d < g.rank; ++d) {
    const auto sourceDim = d - lead;
    if (sourceDim < 0) {
      g.sizes[d] = target[d];
      g.strides[d] = 0;
      continue;
    }
    const auto extent = source.sizes[sourceDim];
    if (target[d] == -1 || target[d] == extent) {
      g.sizes[d] = extent;
      g.strides[d] = source.strides[sourceDim];
      continue;
    }
    TORCH_CHECK(
        extent == 1,
        "expand(): the expanded size ",
        target[d],
        " must match the existing size ",
        extent);
    g.sizes[d] = target[d];
    g.strides[d] = 0;
  }
  writeTensorEntry(table, op.output, *self, g, 0);
  return true;
}

bool view(const DeriveView& op, const ValueTable& table, const Operands& args) {
  const auto* self = args.tensor(0);
  const auto& target = op.intList;
  if (self == nullptr || target.empty() || target.size() > kMaxDims) {
    return false;
  }
  const auto source = geometryOf(*self);
  if (!isContiguous(source)) {
    return false;
  }
  Geometry g;
  g.rank = static_cast<int32_t>(target.size());
  int64_t known = 1;
  int32_t inferred = -1;
  for (int32_t d = 0; d < g.rank; ++d) {
    if (target[d] == -1) {
      inferred = d;
    } else {
      known *= target[d];
      g.sizes[d] = target[d];
    }
  }
  if (inferred >= 0) {
    g.sizes[inferred] = known == 0 ? 0 : self->numEl / known;
  }
  int64_t stride = 1;
  for (int32_t d = g.rank - 1; d >= 0; --d) {
    g.strides[d] = stride;
    stride *= g.sizes[d];
  }
  TORCH_CHECK(
      stride == self->numEl,
      "view(): shape does not match the input's ",
      self->numEl,
      " elements");
  writeTensorEntry(table, op.output, *self, g, 0);
  return true;
}

} // namespace

// unbind (self, dim): element i is select(self, dim, i).
bool unbind(
    const DeriveView& op,
    const ValueTable& table,
    const Operands& args) {
  const auto* self = args.tensor(0);
  int64_t dimArg;
  if (self == nullptr || self->rank == 0 || !args.integer(1, dimArg)) {
    return false;
  }
  auto g = geometryOf(*self);
  const auto dim = wrapDim(dimArg, g.rank);
  if (g.sizes[dim] != static_cast<int64_t>(op.outputs.size())) {
    return false;
  }
  const auto stride = g.strides[dim];
  erase(g, dim);
  for (size_t i = 0; i < op.outputs.size(); ++i) {
    writeTensorEntry(
        table, op.outputs[i], *self, g, static_cast<int64_t>(i) * stride);
  }
  return true;
}

// split_with_sizes (self, sizes, dim): consecutive narrows along 'dim'.
bool splitWithSizes(
    const DeriveView& op,
    const ValueTable& table,
    const Operands& args) {
  const auto* self = args.tensor(0);
  int64_t dimArg;
  if (self == nullptr || self->rank == 0 || op.intList.empty() ||
      op.intList.size() != op.outputs.size() || !args.integer(2, dimArg)) {
    return false;
  }
  auto g = geometryOf(*self);
  const auto dim = wrapDim(dimArg, g.rank);
  const auto extent = g.sizes[dim];
  const auto stride = g.strides[dim];
  int64_t start = 0;
  for (size_t i = 0; i < op.outputs.size(); ++i) {
    const auto length = op.intList[i];
    TORCH_CHECK(
        length >= 0 && start + length <= extent,
        "split_with_sizes(): split sizes overrun dimension ",
        dim,
        " of size ",
        extent);
    g.sizes[dim] = length;
    writeTensorEntry(table, op.outputs[i], *self, g, start * stride);
    start += length;
  }
  return true;
}

// A list unpack: each output is its element's entry.
bool listUnpack(const DeriveView& op, const ValueTable& table) {
  if (op.elements.size() != op.outputs.size()) {
    return false;
  }
  for (auto id : op.elements) {
    if (!table.valid(id)) {
      return false;
    }
  }
  for (size_t i = 0; i < op.outputs.size(); ++i) {
    const auto& from = table.entries[op.elements[i]];
    auto& to = table.entries[op.outputs[i]];
    if (from.kind == ValueEntry::kTensorPending) {
      table.tensor(op.elements[i]);
    }
    to.t = from.t;
    to.scalar = from.scalar;
    to.kind = from.kind;
    to.generation = table.generation;
  }
  return true;
}

bool derivable(StandaloneShortcut kind) {
  switch (kind) {
    case StandaloneShortcut::kSymSize:
    case StandaloneShortcut::kSymNumel:
    case StandaloneShortcut::kSlice:
    case StandaloneShortcut::kSelectInt:
    case StandaloneShortcut::kNarrow:
    case StandaloneShortcut::kSqueezeDim:
    case StandaloneShortcut::kUnsqueeze:
    case StandaloneShortcut::kTranspose:
    case StandaloneShortcut::kExpand:
    case StandaloneShortcut::kView:
      return true;
    case StandaloneShortcut::kNone:
    case StandaloneShortcut::kListPack:
    case StandaloneShortcut::kListUnpack:
    case StandaloneShortcut::kUnbind:
    case StandaloneShortcut::kSplitWithSizes:
    case StandaloneShortcut::kEmptyFactory:
      return false;
  }
  return false;
}

bool runDerive(const DeriveView& op, const ValueTable& table) {
  const Operands args(op, table);
  switch (op.kind) {
    case StandaloneShortcut::kSymSize: {
      const auto* self = args.tensor(0);
      int64_t dim;
      if (self == nullptr || !args.integer(1, dim)) {
        return false;
      }
      writeIntEntry(table, op.output, self->dims[wrapDim(dim, self->rank)]);
      return true;
    }
    case StandaloneShortcut::kSymNumel: {
      const auto* self = args.tensor(0);
      if (self == nullptr) {
        return false;
      }
      writeIntEntry(table, op.output, self->numEl);
      return true;
    }
    case StandaloneShortcut::kSlice:
      return slice(op, table, args);
    case StandaloneShortcut::kSelectInt:
      return select(op, table, args);
    case StandaloneShortcut::kNarrow:
      return narrow(op, table, args);
    case StandaloneShortcut::kSqueezeDim:
      return squeezeDim(op, table, args);
    case StandaloneShortcut::kUnsqueeze:
      return unsqueeze(op, table, args);
    case StandaloneShortcut::kTranspose:
      return transpose(op, table, args);
    case StandaloneShortcut::kExpand:
      return expand(op, table, args);
    case StandaloneShortcut::kView:
      return view(op, table, args);
    case StandaloneShortcut::kUnbind:
      return unbind(op, table, args);
    case StandaloneShortcut::kSplitWithSizes:
      return splitWithSizes(op, table, args);
    case StandaloneShortcut::kListUnpack:
      return listUnpack(op, table);
    case StandaloneShortcut::kListPack: {
      auto& entry = table.entries[op.output];
      entry.kind = ValueEntry::kOther;
      entry.generation = table.generation;
      return true;
    }
    case StandaloneShortcut::kNone:
    case StandaloneShortcut::kEmptyFactory:
      return false;
  }
  return false;
}

} // namespace torch::wave
