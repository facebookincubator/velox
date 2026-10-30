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

#include "velox/experimental/torchwave/StaticDims.h"

#include <algorithm>
#include <string_view>

#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/Utils.h"

namespace torch::wave {

std::optional<StaticDims> staticDimsOf(const ValueTypes& types, ValueCP value) {
  if (value == nullptr ||
      value->type().kind() != nativert::Type::Kind::Tensor) {
    return std::nullopt;
  }
  const auto id = value->id();
  if (id < 0 || static_cast<size_t>(id) >= types.constraints.size()) {
    return std::nullopt;
  }
  const auto& constraint = types.constraints[id];
  if (constraint.rank < 0) {
    return std::nullopt;
  }
  if (constraint.dims.size() == static_cast<size_t>(constraint.rank)) {
    return constraint.dims;
  }
  return StaticDims(constraint.rank, -1);
}

bool isFullyStatic(const ValueTypes& types, ValueCP value) {
  auto dims = staticDimsOf(types, value);
  return dims.has_value() &&
      std::all_of(
             dims->begin(), dims->end(), [](auto dim) { return dim >= 0; });
}

namespace {

using MaybeDims = std::optional<StaticDims>;

std::optional<int64_t> intAttr(NodeCP node, const char* name) {
  const auto* attr = node->tryGetAttribute(name);
  if (attr != nullptr && std::holds_alternative<int64_t>(attr->value)) {
    return std::get<int64_t>(attr->value);
  }
  return std::nullopt;
}

const std::vector<int64_t>* intListAttr(NodeCP node, const char* name) {
  const auto* attr = node->tryGetAttribute(name);
  if (attr != nullptr &&
      std::holds_alternative<std::vector<int64_t>>(attr->value)) {
    return &std::get<std::vector<int64_t>>(attr->value);
  }
  return nullptr;
}

ValueCP namedInput(NodeCP node, const char* name) {
  const auto* input = node->tryGetInput(name);
  return input != nullptr ? input->value : nullptr;
}

std::optional<int64_t> numel(const StaticDims& dims) {
  int64_t result = 1;
  for (auto dim : dims) {
    if (dim < 0) {
      return std::nullopt;
    }
    result *= dim;
  }
  return result;
}

// Right-aligned broadcast. A known non-1 extent wins; a dim stays unknown
// when some operand's extent there is unknown and none is a known non-1.
MaybeDims broadcast(const std::vector<StaticDims>& operands) {
  size_t rank = 0;
  for (const auto& dims : operands) {
    rank = std::max(rank, dims.size());
  }
  StaticDims result(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    bool unknown = false;
    for (const auto& dims : operands) {
      if (i >= dims.size()) {
        continue;
      }
      const auto dim = dims[dims.size() - 1 - i];
      if (dim < 0) {
        unknown = true;
      } else if (dim != 1) {
        result[rank - 1 - i] = dim;
      }
    }
    if (unknown && result[rank - 1 - i] == 1) {
      result[rank - 1 - i] = -1;
    }
  }
  return result;
}

MaybeDims reshaped(const StaticDims& input, const std::vector<int64_t>& size) {
  StaticDims result(size.begin(), size.end());
  auto inferred = std::find(result.begin(), result.end(), -1);
  if (inferred == result.end()) {
    return result;
  }
  auto total = numel(input);
  int64_t known = 1;
  for (auto it = result.begin(); it != result.end(); ++it) {
    if (it != inferred) {
      known *= *it;
    }
  }
  if (total.has_value() && known > 0) {
    *inferred = *total / known;
  }
  return result;
}

int64_t normalizeDim(int64_t dim, size_t rank) {
  return dim < 0 ? dim + static_cast<int64_t>(rank) : dim;
}

MaybeDims unsqueezed(StaticDims input, int64_t dim) {
  dim = normalizeDim(dim, input.size() + 1);
  if (dim < 0 || dim > static_cast<int64_t>(input.size())) {
    return std::nullopt;
  }
  input.insert(input.begin() + dim, 1);
  return input;
}

MaybeDims expanded(const StaticDims& input, const std::vector<int64_t>& size) {
  if (size.size() < input.size()) {
    return std::nullopt;
  }
  StaticDims result(size.begin(), size.end());
  const auto lead = size.size() - input.size();
  for (size_t i = lead; i < size.size(); ++i) {
    if (result[i] == -1) {
      result[i] = input[i - lead];
    }
  }
  return result;
}

MaybeDims sliced(StaticDims input, NodeCP node) {
  const auto dim = normalizeDim(intAttr(node, "dim").value_or(0), input.size());
  if (dim < 0 || dim >= static_cast<int64_t>(input.size())) {
    return std::nullopt;
  }
  const auto extent = input[dim];
  const auto step = intAttr(node, "step").value_or(1);
  const auto start = intAttr(node, "start");
  const auto end = intAttr(node, "end");
  const bool startDynamic = !start && namedInput(node, "start") != nullptr;
  const bool endDynamic = !end && namedInput(node, "end") != nullptr;
  if (extent < 0 || step <= 0 || startDynamic || endDynamic) {
    input[dim] = -1;
    return input;
  }
  auto clampIndex = [&](int64_t index) {
    index = index < 0 ? index + extent : index;
    return std::clamp<int64_t>(index, 0, extent);
  };
  const auto from = clampIndex(start.value_or(0));
  const auto to = clampIndex(end.value_or(extent));
  input[dim] = to > from ? (to - from + step - 1) / step : 0;
  return input;
}

MaybeDims reduced(StaticDims input, NodeCP node) {
  std::vector<int64_t> reducedDims;
  if (const auto* list = intListAttr(node, "dim")) {
    reducedDims = *list;
  } else if (auto dim = intAttr(node, "dim")) {
    reducedDims = {*dim};
  } else {
    for (size_t i = 0; i < input.size(); ++i) {
      reducedDims.push_back(static_cast<int64_t>(i));
    }
  }
  const auto* keepdimAttr = node->tryGetAttribute("keepdim");
  const auto* keepdimFlag =
      keepdimAttr != nullptr ? std::get_if<bool>(&keepdimAttr->value) : nullptr;
  const bool keepdim = keepdimFlag != nullptr && *keepdimFlag;
  std::vector<bool> drop(input.size(), false);
  for (auto dim : reducedDims) {
    dim = normalizeDim(dim, input.size());
    if (dim < 0 || dim >= static_cast<int64_t>(input.size())) {
      return std::nullopt;
    }
    drop[dim] = true;
  }
  StaticDims result;
  for (size_t i = 0; i < input.size(); ++i) {
    if (!drop[i]) {
      result.push_back(input[i]);
    } else if (keepdim) {
      result.push_back(1);
    }
  }
  return result;
}

std::optional<int64_t> arangeExtent(NodeCP node) {
  const auto end = intAttr(node, "end");
  const auto start = intAttr(node, "start").value_or(0);
  const auto step = intAttr(node, "step").value_or(1);
  if (!end || step == 0 || namedInput(node, "start") != nullptr ||
      namedInput(node, "step") != nullptr) {
    return std::nullopt;
  }
  const auto span = *end - start;
  const auto size =
      step > 0 ? (span + step - 1) / step : (span + step + 1) / step;
  return std::max<int64_t>(size, 0);
}

// Dims of the tensors packed into the list argument 'name'.
std::optional<std::vector<StaticDims>>
listElementDims(NodeCP node, const char* name, const ValueTypes& types) {
  auto* list = namedInput(node, name);
  auto* pack = list != nullptr ? list->producer() : nullptr;
  if (pack == nullptr || pack->target() != "prim.ListPack") {
    return std::nullopt;
  }
  std::vector<StaticDims> result;
  for (const auto& element : pack->inputs()) {
    auto dims = staticDimsOf(types, element.value);
    if (!dims) {
      return std::nullopt;
    }
    result.push_back(std::move(*dims));
  }
  return result;
}

MaybeDims concatenated(NodeCP node, const ValueTypes& types) {
  auto operands = listElementDims(node, "tensors", types);
  if (!operands || operands->empty()) {
    return std::nullopt;
  }
  auto result = operands->front();
  const auto dim =
      normalizeDim(intAttr(node, "dim").value_or(0), result.size());
  if (dim < 0 || dim >= static_cast<int64_t>(result.size())) {
    return std::nullopt;
  }
  for (size_t i = 1; i < operands->size(); ++i) {
    const auto& operand = (*operands)[i];
    if (operand.size() != result.size()) {
      return std::nullopt;
    }
    for (size_t d = 0; d < result.size(); ++d) {
      if (static_cast<int64_t>(d) == dim) {
        result[d] =
            result[d] < 0 || operand[d] < 0 ? -1 : result[d] + operand[d];
      } else if (result[d] < 0) {
        result[d] = operand[d];
      }
    }
  }
  return result;
}

// A pointwise op whose output is the broadcast of its tensor operands: an
// elementwise op that reads no operand as a whole tensor and takes no shape
// from an output or a reserve.
bool isPointwise(const Metadata& meta) {
  if (!meta.elementwise || meta.sizeFromOutput ||
      meta.elementwise->hasOutputArg || meta.elementwise->hasOwnOutputArg) {
    return false;
  }
  for (const auto& arg : meta.argumentMeta) {
    if (arg.wholeTensor) {
      return false;
    }
  }
  for (const auto& ret : meta.returnMeta) {
    if (ret.reserve != nullptr) {
      return false;
    }
  }
  return true;
}

MaybeDims pointwise(NodeCP node, const ValueTypes& types) {
  std::vector<StaticDims> operands;
  for (const auto& input : node->inputs()) {
    if (input.value->type().kind() != nativert::Type::Kind::Tensor) {
      continue;
    }
    auto dims = staticDimsOf(types, input.value);
    if (!dims) {
      return std::nullopt;
    }
    operands.push_back(std::move(*dims));
  }
  return broadcast(operands);
}

bool startsWith(std::string_view text, std::string_view prefix) {
  return text.substr(0, prefix.size()) == prefix;
}

MaybeDims inferFirstOutput(NodeCP node, const ValueTypes& types) {
  const std::string_view target = node->target();
  auto input = [&](size_t i) -> MaybeDims {
    return i < node->inputs().size()
        ? staticDimsOf(types, node->inputs()[i].value)
        : std::nullopt;
  };
  auto named = [&](const char* name) -> MaybeDims {
    return staticDimsOf(types, namedInput(node, name));
  };

  if (target == "torch.ops.aten.view.default" ||
      target == "torch.ops.aten.reshape.default" ||
      target == "torch.ops.aten._unsafe_view.default") {
    const auto* size = intListAttr(node, "size");
    size = size != nullptr ? size : intListAttr(node, "shape");
    auto self = input(0);
    if (size == nullptr || !self) {
      return std::nullopt;
    }
    return reshaped(*self, *size);
  }
  if (target == "torch.ops.aten.unsqueeze.default") {
    auto self = input(0);
    auto dim = intAttr(node, "dim");
    return self && dim ? unsqueezed(*self, *dim) : std::nullopt;
  }
  if (target == "torch.ops.aten.squeeze.dim") {
    auto self = input(0);
    auto dim = intAttr(node, "dim");
    if (!self || !dim) {
      return std::nullopt;
    }
    const auto d = normalizeDim(*dim, self->size());
    if (d < 0 || d >= static_cast<int64_t>(self->size()) || (*self)[d] < 0) {
      return std::nullopt;
    }
    if ((*self)[d] == 1) {
      self->erase(self->begin() + d);
    }
    return self;
  }
  if (target == "torch.ops.aten.permute.default") {
    auto self = input(0);
    const auto* order = intListAttr(node, "dims");
    if (!self || order == nullptr || order->size() != self->size()) {
      return std::nullopt;
    }
    StaticDims result;
    for (auto d : *order) {
      result.push_back((*self)[normalizeDim(d, self->size())]);
    }
    return result;
  }
  if (target == "torch.ops.aten.t.default") {
    auto self = input(0);
    if (self && self->size() == 2) {
      std::swap((*self)[0], (*self)[1]);
    }
    return self;
  }
  if (target == "torch.ops.aten.transpose.int") {
    auto self = input(0);
    auto dim0 = intAttr(node, "dim0");
    auto dim1 = intAttr(node, "dim1");
    if (!self || !dim0 || !dim1) {
      return std::nullopt;
    }
    std::swap(
        (*self)[normalizeDim(*dim0, self->size())],
        (*self)[normalizeDim(*dim1, self->size())]);
    return self;
  }
  if (target == "torch.ops.aten.expand.default") {
    auto self = input(0);
    const auto* size = intListAttr(node, "size");
    return self && size != nullptr ? expanded(*self, *size) : std::nullopt;
  }
  if (target == "torch.ops.aten.slice.Tensor") {
    auto self = input(0);
    return self ? sliced(*self, node) : std::nullopt;
  }
  if (target == "torch.ops.aten.select.int") {
    auto self = input(0);
    auto dim = intAttr(node, "dim");
    if (!self || !dim) {
      return std::nullopt;
    }
    const auto d = normalizeDim(*dim, self->size());
    if (d < 0 || d >= static_cast<int64_t>(self->size())) {
      return std::nullopt;
    }
    self->erase(self->begin() + d);
    return self;
  }
  if (target == "torch.ops.aten.cat.default") {
    return concatenated(node, types);
  }
  if (startsWith(target, "torch.ops.aten.arange.")) {
    auto extent = arangeExtent(node);
    return extent ? MaybeDims(StaticDims{*extent}) : std::nullopt;
  }
  static const std::vector<std::string_view> kFactories = {
      "torch.ops.aten.zeros.default",
      "torch.ops.aten.ones.default",
      "torch.ops.aten.full.default",
      "torch.ops.aten.empty.memory_format",
      "torch.ops.aten.new_zeros.default",
      "torch.ops.aten.new_ones.default",
      "torch.ops.aten.new_full.default",
      "torch.ops.aten.new_empty.default",
  };
  if (std::find(kFactories.begin(), kFactories.end(), target) !=
      kFactories.end()) {
    const auto* size = intListAttr(node, "size");
    return size != nullptr ? MaybeDims(StaticDims(size->begin(), size->end()))
                           : std::nullopt;
  }
  if (target == "torch.ops.aten.cumsum.default" ||
      target == "torch.ops.aten.clone.default" ||
      target == "torch.ops.aten.contiguous.default" ||
      target == "torch.ops.aten.alias.default" ||
      target == "torch.ops.aten.detach.default") {
    return input(0);
  }
  if (target == "tw.gather" || target == "torch.ops.aten.gather.default") {
    return named("index");
  }
  if (target == "tw.searchsorted" ||
      target == "torch.ops.aten.searchsorted.Tensor" ||
      target == "torch.ops.aten.bucketize.Tensor") {
    return named("self");
  }
  if (startsWith(target, "tw.index_elt_")) {
    auto indices = listElementDims(node, "indices", types);
    return indices ? broadcast(*indices) : std::nullopt;
  }
  if (target == "tw.any_rows") {
    auto self = input(0);
    auto keep = intAttr(node, "keep");
    if (!self || !keep || *keep > static_cast<int64_t>(self->size())) {
      return std::nullopt;
    }
    self->resize(*keep);
    return self;
  }
  static const std::vector<std::string_view> kReductions = {
      "torch.ops.aten.any.dim",
      "torch.ops.aten.any.dims",
      "torch.ops.aten.all.dim",
      "torch.ops.aten.sum.dim_IntList",
      "torch.ops.aten.amax.default",
      "torch.ops.aten.amin.default",
      "torch.ops.aten.mean.dim",
  };
  if (std::find(kReductions.begin(), kReductions.end(), target) !=
      kReductions.end()) {
    auto self = input(0);
    return self ? reduced(*self, node) : std::nullopt;
  }
  // Not nodeMeta: that caches the metadata of the current target, and the
  // optimizer may still retarget this node.
  const auto* meta = Registry::metadata(node->target());
  if (meta != nullptr && isPointwise(*meta)) {
    return pointwise(node, types);
  }
  return std::nullopt;
}

} // namespace

std::vector<std::optional<StaticDims>> inferStaticDims(
    NodeCP node,
    const ValueTypes& types) {
  std::vector<std::optional<StaticDims>> result(node->outputs().size());
  if (!result.empty() &&
      node->outputs()[0]->type().kind() == nativert::Type::Kind::Tensor) {
    result[0] = inferFirstOutput(node, types);
  }
  return result;
}

} // namespace torch::wave
