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

// Shape rules for the aten view, shape and reduction ops: each gives an op's
// output shape from its inputs' shapes and its integer operands, so that
// shapes follow from the graph inputs' dims rather than from the dims the
// export was traced with.

#include "velox/experimental/torchwave/SymShape.h"

#include <folly/CppAttributes.h>
#include <torch/nativert/graph/Graph.h>

namespace torch::wave {

namespace {

using nativert::Node;
using nativert::Value;

constexpr int64_t kMaxSliceEnd = int64_t{1} << 62;

const SymShape* FOLLY_NULLABLE
shapeOfValue(const SymShapes& shapes, const Value* value) {
  if (value == nullptr) {
    return nullptr;
  }
  auto it = shapes.byName.find(std::string(value->name()));
  return it != shapes.byName.end() ? &it->second : nullptr;
}

const SymShape* FOLLY_NULLABLE
inputShape(const SymShapes& shapes, const Node& node, std::string_view name) {
  const auto* input = node.tryGetInput(name);
  return input != nullptr ? shapeOfValue(shapes, input->value) : nullptr;
}

std::optional<SymExpr> scalarOfValue(
    const SymShapes& shapes,
    const Value* value) {
  if (value == nullptr) {
    return std::nullopt;
  }
  auto it = shapes.scalarByName.find(std::string(value->name()));
  if (it == shapes.scalarByName.end()) {
    return std::nullopt;
  }
  return it->second;
}

bool isAbsent(const Node& node, std::string_view name) {
  const auto* attr = node.tryGetAttribute(name);
  if (attr != nullptr) {
    return std::holds_alternative<nativert::None>(attr->value);
  }
  const auto* input = node.tryGetInput(name);
  return input == nullptr || input->value == nullptr ||
      input->value->type().kind() == nativert::Type::Kind::None;
}

// An integer operand: an attribute or a SymInt input.
std::optional<SymExpr>
intOperand(const SymShapes& shapes, const Node& node, std::string_view name) {
  if (const auto* attr = node.tryGetAttribute(name)) {
    if (const auto* value = std::get_if<int64_t>(&attr->value)) {
      return SymExpr::constant(*value);
    }
    return std::nullopt;
  }
  const auto* input = node.tryGetInput(name);
  return input != nullptr ? scalarOfValue(shapes, input->value) : std::nullopt;
}

std::optional<int64_t> intAttr(const Node& node, std::string_view name) {
  const auto* attr = node.tryGetAttribute(name);
  if (attr == nullptr) {
    return std::nullopt;
  }
  if (const auto* value = std::get_if<int64_t>(&attr->value)) {
    return *value;
  }
  return std::nullopt;
}

// An int list operand: an attribute or a ListPack of SymInts.
std::optional<std::vector<SymExpr>> intListOperand(
    const SymShapes& shapes,
    const Node& node,
    std::string_view name) {
  std::vector<SymExpr> result;
  if (const auto* attr = node.tryGetAttribute(name)) {
    if (const auto* values = std::get_if<std::vector<int64_t>>(&attr->value)) {
      for (auto value : *values) {
        result.push_back(SymExpr::constant(value));
      }
      return result;
    }
    if (const auto* value = std::get_if<int64_t>(&attr->value)) {
      result.push_back(SymExpr::constant(*value));
      return result;
    }
    return std::nullopt;
  }
  const auto* input = node.tryGetInput(name);
  const auto* pack = input != nullptr && input->value != nullptr
      ? input->value->producer()
      : nullptr;
  if (pack == nullptr || pack->target() != "prim.ListPack") {
    return std::nullopt;
  }
  for (const auto& element : pack->inputs()) {
    auto value = scalarOfValue(shapes, element.value);
    if (!value) {
      return std::nullopt;
    }
    result.push_back(std::move(*value));
  }
  return result;
}

std::optional<size_t> normalizeDim(int64_t dim, size_t rank) {
  if (dim < 0) {
    dim += static_cast<int64_t>(rank);
  }
  if (dim < 0 || dim >= static_cast<int64_t>(rank)) {
    return std::nullopt;
  }
  return static_cast<size_t>(dim);
}

std::optional<size_t> dimOperand(
    const Node& node,
    std::string_view name,
    size_t rank,
    int64_t defaultDim = 0) {
  return normalizeDim(intAttr(node, name).value_or(defaultDim), rank);
}

bool isOne(const SymExpr& expr) {
  return expr.isConstant() && expr.value == 1;
}

SymExpr symSub(const SymExpr& lhs, const SymExpr& rhs) {
  return symAdd(lhs, symMul(SymExpr::constant(-1), rhs));
}

SymExpr symMin(const SymExpr& lhs, const SymExpr& rhs) {
  if (lhs.isConstant() && rhs.isConstant()) {
    return SymExpr::constant(std::min(lhs.value, rhs.value));
  }
  return SymExpr{.kind = SymExpr::Kind::kMin, .args = {lhs, rhs}};
}

int32_t addValueShape(SymShapes& shapes, const Value* value, SymShape&& shape) {
  return shapes.byName.emplace(std::string(value->name()), std::move(shape))
             .second
      ? 1
      : 0;
}

int32_t setOutput(
    SymShapes& shapes,
    const Node& node,
    std::vector<SymExpr> dims,
    size_t output = 0) {
  if (node.outputs().size() <= output) {
    return 0;
  }
  return addValueShape(
      shapes, node.outputs()[output], denseShape(std::move(dims)));
}

// The storage 'self' of 'node' views and where 'self' starts in it, moved
// 'delta' elements on.
std::pair<std::string, SymExpr>
viewOfSelf(const SymShapes& shapes, const Node& node, const SymExpr& delta) {
  const auto* input = node.tryGetInput("self");
  if (input == nullptr) {
    return {std::string(), SymExpr::constant(0)};
  }
  auto it = shapes.byName.find(std::string(input->value->name()));
  if (it == shapes.byName.end() || it->second.base.empty()) {
    return {std::string(input->value->name()), delta};
  }
  return {it->second.base, symAdd(it->second.offset, delta)};
}

// Sets a view's output: its dims over the strides of its base, starting
// 'delta' elements after the start of 'self'.
int32_t setViewOutput(
    SymShapes& shapes,
    const Node& node,
    std::vector<SymExpr> dims,
    std::vector<SymExpr> strides,
    const SymExpr& delta = SymExpr::constant(0)) {
  if (node.outputs().empty()) {
    return 0;
  }
  if (strides.size() != dims.size()) {
    return setOutput(shapes, node, std::move(dims));
  }
  auto [base, offset] = viewOfSelf(shapes, node, delta);
  return addValueShape(
      shapes,
      node.outputs()[0],
      SymShape{
          .dims = std::move(dims),
          .strides = std::move(strides),
          .base = std::move(base),
          .offset = std::move(offset)});
}

// The strides of 'self', or dense strides when it has none for every dim.
std::vector<SymExpr> stridesOf(const SymShape& self) {
  return self.strides.size() == self.dims.size()
      ? self.strides
      : denseShape(self.dims).strides;
}

// The number of elements of a list output that its ListUnpack takes apart.
std::optional<size_t> unpackedCount(const Node& node) {
  if (node.outputs().empty()) {
    return std::nullopt;
  }
  for (const auto* user : node.outputs()[0]->users()) {
    if (user->target() == "prim.ListUnpack") {
      return user->outputs().size();
    }
  }
  return std::nullopt;
}

// Sets the elements of a list output, dense, or as views over 'strides'
// when given.
int32_t setListElements(
    SymShapes& shapes,
    const Node& node,
    const std::vector<std::vector<SymExpr>>& elements,
    const std::vector<SymExpr>* strides = nullptr,
    const std::vector<SymExpr>* starts = nullptr) {
  const auto list = node.outputs()[0]->name();
  int32_t numAdded = 0;
  for (size_t i = 0; i < elements.size(); ++i) {
    auto shape = strides != nullptr && strides->size() == elements[i].size()
        ? SymShape{.dims = elements[i], .strides = *strides}
        : denseShape(elements[i]);
    if (starts != nullptr && i < starts->size() && strides != nullptr &&
        strides->size() == elements[i].size()) {
      std::tie(shape.base, shape.offset) =
          viewOfSelf(shapes, node, (*starts)[i]);
    }
    numAdded +=
        shapes.byName.emplace(listElementName(list, i), std::move(shape)).second
        ? 1
        : 0;
  }
  return numAdded;
}

int32_t sliceRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  if (self == nullptr) {
    return 0;
  }
  auto dim = dimOperand(node, "dim", self->dims.size());
  if (!dim) {
    return 0;
  }
  const auto& size = self->dims[*dim];
  SymExpr start = SymExpr::constant(0);
  if (!isAbsent(node, "start")) {
    auto value = intOperand(shapes, node, "start");
    if (!value) {
      return 0;
    }
    start =
        value->isConstant() && value->value < 0 ? symAdd(size, *value) : *value;
  }
  SymExpr end = size;
  if (!isAbsent(node, "end")) {
    auto value = intOperand(shapes, node, "end");
    if (!value) {
      return 0;
    }
    // An end the export traced at or past the dim's extent is the extent.
    const auto tracedSize = tracedValue(size);
    if (!value->isConstant()) {
      end = *value;
    } else if (value->value < 0) {
      end = symAdd(size, *value);
    } else if (
        value->value < kMaxSliceEnd &&
        !(tracedSize.has_value() && value->value >= *tracedSize)) {
      end = symMin(*value, size);
    }
  }
  const auto step = intAttr(node, "step").value_or(1);
  auto length = symSub(end, start);
  if (step > 1) {
    length = symFloorDiv(
        symAdd(length, SymExpr::constant(step - 1)), SymExpr::constant(step));
  }
  auto dims = self->dims;
  dims[*dim] = length;
  auto strides = stridesOf(*self);
  auto& stride = strides.at(*dim);
  const auto delta = symMul(start, stride);
  stride = symMul(stride, SymExpr::constant(step));
  return setViewOutput(
      shapes, node, std::move(dims), std::move(strides), delta);
}

int32_t selectRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  if (self == nullptr) {
    return 0;
  }
  auto dim = dimOperand(node, "dim", self->dims.size());
  if (!dim) {
    return 0;
  }
  auto index = intOperand(shapes, node, "index");
  if (index && index->isConstant() && index->value < 0) {
    index = symAdd(self->dims[*dim], *index);
  } else if (index && !index->isConstant()) {
    // A symbolic index may be negative, counting from the end.
    index.reset();
  }
  auto dims = self->dims;
  dims.erase(dims.begin() + static_cast<std::ptrdiff_t>(*dim));
  auto strides = stridesOf(*self);
  // Without the index the view's start is unknown: it gets no base.
  const auto delta =
      index ? symMul(*index, strides.at(*dim)) : SymExpr::constant(0);
  strides.erase(strides.begin() + static_cast<std::ptrdiff_t>(*dim));
  if (!index) {
    const auto added =
        setViewOutput(shapes, node, std::move(dims), std::move(strides));
    if (added != 0) {
      shapes.byName.at(std::string(node.outputs()[0]->name())).base.clear();
    }
    return added;
  }
  return setViewOutput(
      shapes, node, std::move(dims), std::move(strides), delta);
}

int32_t unsqueezeRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  if (self == nullptr) {
    return 0;
  }
  auto dim = dimOperand(node, "dim", self->dims.size() + 1);
  if (!dim) {
    return 0;
  }
  auto dims = self->dims;
  auto strides = stridesOf(*self);
  // A size-1 dim's stride is never stepped; the one a contiguous base would
  // give keeps the result contiguous.
  const auto stride = *dim < strides.size() ? symMul(strides[*dim], dims[*dim])
                                            : SymExpr::constant(1);
  dims.insert(
      dims.begin() + static_cast<std::ptrdiff_t>(*dim), SymExpr::constant(1));
  strides.insert(strides.begin() + static_cast<std::ptrdiff_t>(*dim), stride);
  return setViewOutput(shapes, node, std::move(dims), std::move(strides));
}

// Drops the given dims, or every dim, that are 1.
int32_t squeezeRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  if (self == nullptr) {
    return 0;
  }
  const auto rank = self->dims.size();
  // A dim is 1 if its expression is, or the export traced it at 1.
  const SymShape* exported = shapes.exported != nullptr
      ? shapeOfValue(*shapes.exported, node.tryGetInput("self")->value)
      : nullptr;
  auto isUnit = [&](size_t i) {
    return isOne(self->dims[i]) || tracedValue(self->dims[i]) == 1 ||
        (exported != nullptr && exported->dims.size() == rank &&
         isOne(exported->dims[i]));
  };
  std::vector<bool> drop(rank, false);
  auto squeezable = intListOperand(shapes, node, "dim");
  if (squeezable) {
    for (const auto& dim : *squeezable) {
      auto index =
          dim.isConstant() ? normalizeDim(dim.value, rank) : std::nullopt;
      if (!index) {
        return 0;
      }
      drop[*index] = isUnit(*index);
    }
  } else {
    for (size_t i = 0; i < rank; ++i) {
      drop[i] = isUnit(i);
    }
  }
  const auto selfStrides = stridesOf(*self);
  std::vector<SymExpr> dims;
  std::vector<SymExpr> strides;
  for (size_t i = 0; i < rank; ++i) {
    if (!drop[i]) {
      dims.push_back(self->dims[i]);
      strides.push_back(selfStrides[i]);
    }
  }
  return setViewOutput(shapes, node, std::move(dims), std::move(strides));
}

int32_t catRule(SymShapes& shapes, const Node& node) {
  const auto* tensors = node.tryGetInput("tensors");
  if (tensors == nullptr) {
    return 0;
  }
  const auto elements = listElementShapes(shapes, tensors->value);
  const SymShape* first = nullptr;
  for (const auto* element : elements) {
    if (element == nullptr) {
      return 0;
    }
    // A 1-D empty operand is skipped, as torch.cat does.
    const bool empty = element->dims.size() == 1 &&
        element->dims[0].isConstant() && element->dims[0].value == 0;
    if (first == nullptr && !empty) {
      first = element;
    }
  }
  if (first == nullptr) {
    return 0;
  }
  auto dim = dimOperand(node, "dim", first->dims.size());
  if (!dim) {
    return 0;
  }
  SymExpr total = SymExpr::constant(0);
  for (const auto* element : elements) {
    if (element->dims.size() != first->dims.size()) {
      continue;
    }
    total = symAdd(total, element->dims[*dim]);
  }
  auto dims = first->dims;
  dims[*dim] = total;
  return setOutput(shapes, node, std::move(dims));
}

int32_t stackRule(SymShapes& shapes, const Node& node) {
  const auto* tensors = node.tryGetInput("tensors");
  if (tensors == nullptr) {
    return 0;
  }
  const auto elements = listElementShapes(shapes, tensors->value);
  if (elements.empty() || elements[0] == nullptr) {
    return 0;
  }
  auto dim = dimOperand(node, "dim", elements[0]->dims.size() + 1);
  if (!dim) {
    return 0;
  }
  auto dims = elements[0]->dims;
  dims.insert(
      dims.begin() + static_cast<std::ptrdiff_t>(*dim),
      SymExpr::constant(static_cast<int64_t>(elements.size())));
  return setOutput(shapes, node, std::move(dims));
}

// Matches the constant targets of a view, from one end, to the input dims the
// export traced: one equal traced dim, or a run whose product is equal, gives
// that dim or product of the actual dims. A 1 the input does not have takes
// no dim. Stops at the first target it cannot match. Fills 'out' and returns
// how many targets it matched; 'consumed' is how many input dims it used.
size_t alignFromEnd(
    const std::vector<SymExpr>& targets,
    const std::vector<SymExpr>& dims,
    const std::vector<std::optional<int64_t>>& traced,
    bool fromBack,
    size_t maxTargets,
    size_t maxDims,
    std::vector<std::optional<SymExpr>>& out,
    size_t& consumed) {
  const auto numTargets = targets.size();
  const auto rank = dims.size();
  auto targetAt = [&](size_t k) { return fromBack ? numTargets - 1 - k : k; };
  auto dimAt = [&](size_t k) { return fromBack ? rank - 1 - k : k; };
  consumed = 0;
  size_t matched = 0;
  for (; matched < maxTargets; ++matched) {
    const auto& target = targets[targetAt(matched)];
    if (!target.isConstant() || target.value < 1) {
      break;
    }
    if (target.value == 1 &&
        (consumed >= maxDims || traced[dimAt(consumed)] != 1)) {
      out.at(targetAt(matched)) = target;
      continue;
    }
    std::optional<size_t> length;
    int64_t product = 1;
    for (size_t k = consumed; k < maxDims && traced[dimAt(k)].has_value();
         ++k) {
      product *= *traced[dimAt(k)];
      if (product == target.value) {
        length = k + 1 - consumed;
        break;
      }
      if (product > target.value) {
        break;
      }
    }
    if (!length) {
      break;
    }
    SymExpr dim = SymExpr::constant(1);
    for (size_t k = consumed; k < consumed + *length; ++k) {
      dim = symMul(dim, dims[dimAt(k)]);
    }
    out.at(targetAt(matched)) = dim;
    consumed += *length;
  }
  return matched;
}

// Lines the target sizes of a view up with the input's dims from both ends;
// see alignFromEnd. A -1 is what the other dims leave.
// The strides of a view of 'self' with 'dims'. A view that only adds or
// drops size-1 dims keeps the strides of the dims it keeps, and reads a
// size-1 dim through stride 0, as the host does. Any other view is dense. A
// target dim the export traced as 1 but that is not 1 for every input makes
// the view dense: its stride must not be 0 when the dim is larger, and the
// setup gives a dim that is 1 at run time stride 0 anyway.
std::vector<SymExpr> viewStrides(
    const SymShape& self,
    const std::vector<SymExpr>& dims) {
  auto tracedOnly = [](const SymExpr& dim) {
    return !isOne(dim) && tracedValue(dim) == 1;
  };
  if (std::any_of(dims.begin(), dims.end(), tracedOnly)) {
    return denseShape(dims).strides;
  }
  const auto selfStrides = stridesOf(self);
  std::vector<size_t> kept;
  for (size_t i = 0; i < self.dims.size(); ++i) {
    if (!isOne(self.dims[i]) && tracedValue(self.dims[i]) != 1) {
      kept.push_back(i);
    }
  }
  std::vector<SymExpr> strides;
  size_t next = 0;
  for (const auto& dim : dims) {
    if (isOne(dim) || tracedValue(dim) == 1) {
      strides.push_back(SymExpr::constant(0));
      continue;
    }
    if (next >= kept.size() || !sameDim(dim, self.dims.at(kept[next]))) {
      return denseShape(dims).strides;
    }
    strides.push_back(selfStrides[kept[next++]]);
  }
  return next == kept.size() ? strides : denseShape(dims).strides;
}

int32_t viewRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  auto targets = intListOperand(shapes, node, "size");
  if (self == nullptr || !targets) {
    return 0;
  }
  const auto rank = self->dims.size();
  const SymShape* exported = shapes.exported != nullptr
      ? shapeOfValue(*shapes.exported, node.tryGetInput("self")->value)
      : nullptr;
  std::vector<std::optional<int64_t>> traced;
  traced.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    traced.push_back(
        exported != nullptr && exported->dims.size() == rank &&
                exported->dims[i].isConstant()
            ? exported->dims[i].value
            : tracedValue(self->dims[i]));
  }
  const auto numTargets = targets->size();
  std::vector<std::optional<SymExpr>> matched(numTargets);
  size_t frontDims = 0;
  const auto front = alignFromEnd(
      *targets,
      self->dims,
      traced,
      false,
      numTargets,
      rank,
      matched,
      frontDims);
  size_t backDims = 0;
  alignFromEnd(
      *targets,
      self->dims,
      traced,
      true,
      numTargets - front,
      rank - frontDims,
      matched,
      backDims);
  std::vector<SymExpr> dims;
  std::optional<size_t> inferred;
  for (size_t i = 0; i < numTargets; ++i) {
    const auto& target = (*targets)[i];
    if (matched[i].has_value()) {
      dims.push_back(*matched[i]);
    } else if (target.isConstant() && target.value == -1) {
      inferred = i;
      dims.push_back(SymExpr::constant(1));
    } else {
      dims.push_back(target);
    }
  }
  if (inferred) {
    SymExpr others = SymExpr::constant(1);
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i != *inferred) {
        others = symMul(others, dims[i]);
      }
    }
    dims.at(*inferred) = symFloorDiv(numelOf(*self), others);
  }
  return setViewOutput(shapes, node, dims, viewStrides(*self, dims));
}

int32_t expandRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  auto targets = intListOperand(shapes, node, "size");
  if (self == nullptr || !targets || targets->size() < self->dims.size()) {
    return 0;
  }
  const auto offset = targets->size() - self->dims.size();
  const auto selfStrides = stridesOf(*self);
  std::vector<SymExpr> dims;
  std::vector<SymExpr> strides;
  for (size_t i = 0; i < targets->size(); ++i) {
    const auto& target = (*targets)[i];
    const SymExpr* own = i >= offset ? &self->dims[i - offset] : nullptr;
    const bool keep = own != nullptr &&
        ((target.isConstant() && target.value == -1) || !isOne(*own));
    dims.push_back(keep ? *own : target);
    // An expanded dim reads its one element through stride 0.
    if (keep) {
      strides.push_back(selfStrides[i - offset]);
    } else {
      strides.push_back(SymExpr::constant(0));
    }
  }
  return setViewOutput(shapes, node, std::move(dims), std::move(strides));
}

int32_t sizeListRule(SymShapes& shapes, const Node& node) {
  auto dims = intListOperand(shapes, node, "size");
  if (!dims) {
    return 0;
  }
  return setOutput(shapes, node, std::move(*dims));
}

int32_t scalarTensorRule(SymShapes& shapes, const Node& node) {
  return setOutput(shapes, node, {});
}

int32_t indexSelectRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  const auto* index = inputShape(shapes, node, "index");
  if (self == nullptr || index == nullptr) {
    return 0;
  }
  auto dim = dimOperand(node, "dim", self->dims.size());
  if (!dim) {
    return 0;
  }
  auto dims = self->dims;
  dims[*dim] = numelOf(*index);
  return setOutput(shapes, node, std::move(dims));
}

int32_t permuteRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  auto order = intListOperand(shapes, node, "dims");
  if (self == nullptr || !order || order->size() != self->dims.size()) {
    return 0;
  }
  const auto selfStrides = stridesOf(*self);
  std::vector<SymExpr> dims;
  std::vector<SymExpr> strides;
  for (const auto& dim : *order) {
    auto index = dim.isConstant() ? normalizeDim(dim.value, self->dims.size())
                                  : std::nullopt;
    if (!index) {
      return 0;
    }
    dims.push_back(self->dims[*index]);
    strides.push_back(selfStrides[*index]);
  }
  return setViewOutput(shapes, node, std::move(dims), std::move(strides));
}

int32_t transposeRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  if (self == nullptr) {
    return 0;
  }
  auto dims = self->dims;
  auto strides = stridesOf(*self);
  if (node.target() == "torch.ops.aten.t.default") {
    std::reverse(dims.begin(), dims.end());
    std::reverse(strides.begin(), strides.end());
    return setViewOutput(shapes, node, std::move(dims), std::move(strides));
  }
  auto first = dimOperand(node, "dim0", dims.size());
  auto second = dimOperand(node, "dim1", dims.size());
  if (!first || !second) {
    return 0;
  }
  std::swap(dims[*first], dims[*second]);
  std::swap(strides.at(*first), strides.at(*second));
  return setViewOutput(shapes, node, std::move(dims), std::move(strides));
}

int32_t unbindRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  if (self == nullptr) {
    return 0;
  }
  auto dim = dimOperand(node, "dim", self->dims.size());
  if (!dim) {
    return 0;
  }
  const auto& size = self->dims[*dim];
  auto count = size.isConstant() ? std::optional<size_t>(size.value)
                                 : unpackedCount(node);
  if (!count) {
    return 0;
  }
  auto dims = self->dims;
  dims.erase(dims.begin() + static_cast<std::ptrdiff_t>(*dim));
  auto strides = stridesOf(*self);
  std::vector<SymExpr> starts;
  starts.reserve(*count);
  for (size_t i = 0; i < *count; ++i) {
    starts.push_back(
        symMul(SymExpr::constant(static_cast<int64_t>(i)), strides.at(*dim)));
  }
  strides.erase(strides.begin() + static_cast<std::ptrdiff_t>(*dim));
  return setListElements(
      shapes,
      node,
      std::vector<std::vector<SymExpr>>(*count, dims),
      &strides,
      &starts);
}

int32_t splitWithSizesRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  auto sizes = intListOperand(shapes, node, "split_sizes");
  if (self == nullptr || !sizes) {
    return 0;
  }
  auto dim = dimOperand(node, "dim", self->dims.size());
  if (!dim) {
    return 0;
  }
  const auto strides = stridesOf(*self);
  std::vector<std::vector<SymExpr>> elements;
  std::vector<SymExpr> starts;
  SymExpr start = SymExpr::constant(0);
  for (const auto& size : *sizes) {
    auto& dims = elements.emplace_back(self->dims);
    dims.at(*dim) = size;
    starts.push_back(symMul(start, strides[*dim]));
    start = symAdd(start, size);
  }
  return setListElements(shapes, node, elements, &strides, &starts);
}

// Chunks of 'split_size', the last one what remains.
int32_t splitRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  auto chunk = intOperand(shapes, node, "split_size");
  auto count = unpackedCount(node);
  if (self == nullptr || !chunk || !count || *count == 0) {
    return 0;
  }
  auto dim = dimOperand(node, "dim", self->dims.size());
  if (!dim) {
    return 0;
  }
  std::vector<std::vector<SymExpr>> elements(*count, self->dims);
  for (size_t i = 0; i + 1 < *count; ++i) {
    elements[i][*dim] = *chunk;
  }
  elements.at(*count - 1).at(*dim) = symSub(
      self->dims[*dim],
      symMul(SymExpr::constant(static_cast<int64_t>(*count - 1)), *chunk));
  const auto strides = stridesOf(*self);
  std::vector<SymExpr> starts;
  starts.reserve(*count);
  for (size_t i = 0; i < *count; ++i) {
    starts.push_back(symMul(
        symMul(SymExpr::constant(static_cast<int64_t>(i)), *chunk),
        strides[*dim]));
  }
  return setListElements(shapes, node, elements, &strides, &starts);
}

// Leading dims indexed by tensors give their broadcast shape; the rest stay.
int32_t indexRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  const auto* indices = node.tryGetInput("indices");
  const auto* pack = indices != nullptr && indices->value != nullptr
      ? indices->value->producer()
      : nullptr;
  if (self == nullptr || pack == nullptr ||
      (pack->target() != "prim.ListPack" &&
       pack->target() != "prim.ListPackOptional")) {
    return 0;
  }
  // Leading None entries keep their dims; the indexed dims must follow them
  // and be consecutive.
  std::vector<const SymShape*> operands;
  size_t leading = 0;
  for (const auto& input : pack->inputs()) {
    if (input.value == nullptr ||
        input.value->type().kind() == nativert::Type::Kind::None) {
      if (!operands.empty()) {
        return 0;
      }
      ++leading;
      continue;
    }
    const auto* shape = shapeOfValue(shapes, input.value);
    if (shape == nullptr) {
      return 0;
    }
    operands.push_back(shape);
  }
  if (operands.empty() || leading + operands.size() > self->dims.size()) {
    return 0;
  }
  size_t rank = 0;
  for (const auto* shape : operands) {
    rank = std::max(rank, shape->dims.size());
  }
  std::vector<SymExpr> dims(rank, SymExpr::constant(1));
  for (size_t i = 0; i < rank; ++i) {
    auto& out = dims[rank - 1 - i];
    for (const auto* shape : operands) {
      if (i < shape->dims.size() &&
          !isOne(shape->dims[shape->dims.size() - 1 - i])) {
        out = shape->dims[shape->dims.size() - 1 - i];
      }
    }
  }
  dims.insert(
      dims.begin(),
      self->dims.begin(),
      self->dims.begin() + static_cast<std::ptrdiff_t>(leading));
  dims.insert(
      dims.end(),
      self->dims.begin() +
          static_cast<std::ptrdiff_t>(leading + operands.size()),
      self->dims.end());
  return setOutput(shapes, node, std::move(dims));
}

// One more than the input's elements, as the offsets of its lengths.
int32_t completeCumsumRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  if (self == nullptr) {
    self = inputShape(shapes, node, "input");
  }
  if (self == nullptr) {
    return 0;
  }
  return setOutput(
      shapes, node, {symAdd(numelOf(*self), SymExpr::constant(1))});
}

int32_t searchsortedRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  if (self == nullptr) {
    return 0;
  }
  return setOutput(shapes, node, self->dims);
}

int32_t indexShapedRule(SymShapes& shapes, const Node& node) {
  const auto* index = inputShape(shapes, node, "index");
  if (index == nullptr) {
    return 0;
  }
  return setOutput(shapes, node, index->dims);
}

// A vector as long as an integer operand: 'total' or 'output_size'. Without
// either, only the data decides the length: a symbol of its own.
int32_t lengthOperandRule(SymShapes& shapes, const Node& node) {
  auto length = intOperand(shapes, node, "total");
  if (!length && !isAbsent(node, "output_size")) {
    length = intOperand(shapes, node, "output_size");
  }
  if (!length && node.tryGetInput("total") == nullptr &&
      isAbsent(node, "output_size") && !node.outputs().empty()) {
    length = SymExpr{
        .kind = SymExpr::Kind::kSymbol,
        .symbol = "u_" + std::string(node.outputs()[0]->name())};
  }
  if (!length) {
    return 0;
  }
  return setOutput(shapes, node, {*length});
}

// A vector as long as the 'size' operand: tw.unpack_slot, which decodes one
// slot of a packed input.
int32_t sizeOperandRule(SymShapes& shapes, const Node& node) {
  auto length = intOperand(shapes, node, "size");
  if (!length) {
    return 0;
  }
  return setOutput(shapes, node, {*length});
}

// A count only the data decides: masked_select's selected elements, and
// nonzero's rows of indices, one column per dim of 'self'.
int32_t selectedCountRule(SymShapes& shapes, const Node& node) {
  if (node.outputs().empty()) {
    return 0;
  }
  const SymExpr count{
      .kind = SymExpr::Kind::kSymbol,
      .symbol = "u_" + std::string(node.outputs()[0]->name())};
  if (node.target() == "torch.ops.aten.nonzero.default") {
    const auto* self = inputShape(shapes, node, "self");
    if (self == nullptr) {
      return 0;
    }
    return setOutput(
        shapes,
        node,
        {count, SymExpr::constant(static_cast<int64_t>(self->dims.size()))});
  }
  // An operand with a constant 0 dim selects nothing.
  for (const char* name : {"self", "mask"}) {
    const auto* operand = inputShape(shapes, node, name);
    if (operand != nullptr &&
        std::any_of(
            operand->dims.begin(), operand->dims.end(), [](const auto& dim) {
              return dim.isConstant() && dim.value == 0;
            })) {
      return setOutput(shapes, node, {SymExpr::constant(0)});
    }
  }
  return setOutput(shapes, node, {count});
}

// Every output has the shape of 'self': sort's values and indices.
int32_t allLikeSelfRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  if (self == nullptr) {
    return 0;
  }
  const auto dims = self->dims;
  int32_t numAdded = 0;
  for (size_t i = 0; i < node.outputs().size(); ++i) {
    numAdded += setOutput(shapes, node, dims, i);
  }
  return numAdded;
}

// Every output drops the last dim of 'self': min_rows' values and indices.
int32_t allButLastLikeSelfRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  if (self == nullptr || self->dims.empty()) {
    return 0;
  }
  const std::vector<SymExpr> dims(self->dims.begin(), self->dims.end() - 1);
  int32_t numAdded = 0;
  for (size_t i = 0; i < node.outputs().size(); ++i) {
    numAdded += setOutput(shapes, node, dims, i);
  }
  return numAdded;
}

int32_t exponentShapedRule(SymShapes& shapes, const Node& node) {
  const auto* exponent = inputShape(shapes, node, "exponent");
  if (exponent == nullptr) {
    return 0;
  }
  return setOutput(shapes, node, exponent->dims);
}

// Removes dim1 and dim2 and appends the diagonal's length.
int32_t diagonalRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  if (self == nullptr || intAttr(node, "offset").value_or(0) != 0) {
    return 0;
  }
  const auto rank = self->dims.size();
  auto first = dimOperand(node, "dim1", rank, 0);
  auto second = dimOperand(node, "dim2", rank, 1);
  if (!first || !second || *first == *second) {
    return 0;
  }
  const auto selfStrides = stridesOf(*self);
  std::vector<SymExpr> dims;
  std::vector<SymExpr> strides;
  for (size_t i = 0; i < rank; ++i) {
    if (i != *first && i != *second) {
      dims.push_back(self->dims[i]);
      strides.push_back(selfStrides[i]);
    }
  }
  const auto& lhs = self->dims[*first];
  const auto& rhs = self->dims[*second];
  if (sameDim(lhs, rhs)) {
    dims.push_back(lhs);
  } else {
    dims.push_back(symMin(lhs, rhs));
  }
  // A step along the diagonal steps both dims.
  strides.push_back(symAdd(selfStrides[*first], selfStrides[*second]));
  return setViewOutput(shapes, node, std::move(dims), std::move(strides));
}

// The open-addressing table isin builds: the smallest power of two at least
// twice the test elements, plus one. Known only for a constant count.
int32_t isinHeadRule(SymShapes& shapes, const Node& node) {
  const auto* elements = inputShape(shapes, node, "test_elements");
  if (elements == nullptr) {
    return 0;
  }
  const auto numel = numelOf(*elements);
  if (!numel.isConstant()) {
    return 0;
  }
  int64_t tableSize = 1;
  while (tableSize < numel.value * 2) {
    tableSize *= 2;
  }
  return setOutput(shapes, node, {SymExpr::constant(tableSize + 1)});
}

int32_t elementsShapedRule(SymShapes& shapes, const Node& node) {
  const auto* elements = inputShape(shapes, node, "elements");
  if (elements == nullptr) {
    return 0;
  }
  return setOutput(shapes, node, elements->dims);
}

int32_t repeatInterleaveHeadRule(SymShapes& shapes, const Node& node) {
  const auto* repeats = inputShape(shapes, node, "repeats");
  if (repeats == nullptr) {
    return 0;
  }
  return setOutput(shapes, node, repeats->dims);
}

// 'total' elements along 'dim', or in all when 'dim' is absent.
int32_t repeatInterleaveFinalRule(SymShapes& shapes, const Node& node) {
  const bool isAten =
      node.target() == "torch.ops.aten.repeat_interleave.self_Tensor";
  const auto* input = inputShape(shapes, node, isAten ? "self" : "input");
  auto total = intOperand(shapes, node, isAten ? "output_size" : "total");
  if (input == nullptr || !total) {
    return 0;
  }
  if (isAbsent(node, "dim")) {
    return setOutput(shapes, node, {*total});
  }
  auto dim = dimOperand(node, "dim", input->dims.size());
  if (!dim) {
    return 0;
  }
  auto dims = input->dims;
  dims[*dim] = *total;
  return setOutput(shapes, node, std::move(dims));
}

// Pads the last dims by the pairs in 'pad', the last dim first.
int32_t padRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  auto pad = intListOperand(shapes, node, "pad");
  if (self == nullptr || !pad || pad->size() % 2 != 0 ||
      pad->size() / 2 > self->dims.size()) {
    return 0;
  }
  auto dims = self->dims;
  if (node.target() == "tw.pad_last_dim") {
    dims.back() = symAdd(dims.back(), symAdd((*pad)[0], (*pad)[1]));
    return setOutput(shapes, node, std::move(dims));
  }
  for (size_t i = 0; i < pad->size() / 2; ++i) {
    auto& dim = dims[dims.size() - 1 - i];
    dim = symAdd(dim, symAdd((*pad)[2 * i], (*pad)[2 * i + 1]));
  }
  return setOutput(shapes, node, std::move(dims));
}

int32_t arangeRule(SymShapes& shapes, const Node& node) {
  auto end = intOperand(shapes, node, "end");
  if (!end) {
    return 0;
  }
  SymExpr start = SymExpr::constant(0);
  if (!isAbsent(node, "start")) {
    auto value = intOperand(shapes, node, "start");
    if (!value) {
      return 0;
    }
    start = *value;
  }
  const auto step = intAttr(node, "step").value_or(1);
  if (step < 1) {
    return 0;
  }
  auto length = symSub(*end, start);
  if (step > 1) {
    length = symFloorDiv(
        symAdd(length, SymExpr::constant(step - 1)), SymExpr::constant(step));
  }
  return setOutput(shapes, node, {length});
}

int32_t repeatRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  auto repeats = intListOperand(shapes, node, "repeats");
  if (self == nullptr || !repeats || repeats->size() < self->dims.size()) {
    return 0;
  }
  const auto offset = repeats->size() - self->dims.size();
  std::vector<SymExpr> dims;
  dims.reserve(repeats->size());
  for (size_t i = 0; i < repeats->size(); ++i) {
    if (i < offset) {
      dims.push_back((*repeats)[i]);
    } else {
      dims.push_back(symMul(self->dims[i - offset], (*repeats)[i]));
    }
  }
  return setOutput(shapes, node, std::move(dims));
}

// Reductions over a dim list, or over one dim with values and indices out.
int32_t reduceRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  auto reduced = intListOperand(shapes, node, "dim");
  if (self == nullptr || !reduced) {
    return 0;
  }
  const auto rank = self->dims.size();
  const bool keep = node.tryGetAttribute("keepdim") != nullptr &&
      std::holds_alternative<bool>(node.tryGetAttribute("keepdim")->value) &&
      std::get<bool>(node.tryGetAttribute("keepdim")->value);
  std::vector<bool> isReduced(rank, reduced->empty());
  for (const auto& dim : *reduced) {
    auto index =
        dim.isConstant() ? normalizeDim(dim.value, rank) : std::nullopt;
    if (!index) {
      return 0;
    }
    isReduced[*index] = true;
  }
  std::vector<SymExpr> dims;
  for (size_t i = 0; i < rank; ++i) {
    if (!isReduced[i]) {
      dims.push_back(self->dims[i]);
    } else if (keep) {
      dims.push_back(SymExpr::constant(1));
    }
  }
  int32_t numAdded = 0;
  for (size_t i = 0; i < node.outputs().size(); ++i) {
    numAdded += setOutput(shapes, node, dims, i);
  }
  return numAdded;
}

// sym_size gives the dim of its input; it replaces the export's expression,
// which holds the traced value when the export specialized the dim.
int32_t symSizeRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  if (self == nullptr || node.outputs().empty()) {
    return 0;
  }
  auto dim = dimOperand(node, "dim", self->dims.size());
  if (!dim) {
    return 0;
  }
  const auto& value = self->dims[*dim];
  auto [it, added] = shapes.scalarByName.try_emplace(
      std::string(node.outputs()[0]->name()), value);
  // The export's own expression stays when these are the export's shapes.
  if (!added && shapes.exported != nullptr &&
      it->second.toString() != value.toString()) {
    it->second = value;
    return 1;
  }
  return added ? 1 : 0;
}

// Elementwise ops outside keepsSelfShape: the output has the shape of 'self'
// or 'input'.
int32_t sameShapeRule(SymShapes& shapes, const Node& node) {
  const auto* self = inputShape(shapes, node, "self");
  if (self == nullptr) {
    self = inputShape(shapes, node, "input");
  }
  if (self == nullptr) {
    return 0;
  }
  return setOutput(shapes, node, self->dims);
}

// Elementwise ops outside broadcastsOperands: the right-aligned broadcast of
// the tensor operands. Two extents that are not 1 must be the same dim.
int32_t broadcastRule(SymShapes& shapes, const Node& node) {
  std::vector<const SymShape*> operands;
  for (const auto& input : node.inputs()) {
    if (input.value == nullptr ||
        input.value->type().kind() != nativert::Type::Kind::Tensor) {
      continue;
    }
    const auto* shape = shapeOfValue(shapes, input.value);
    if (shape == nullptr) {
      return 0;
    }
    operands.push_back(shape);
  }
  size_t rank = 0;
  for (const auto* shape : operands) {
    rank = std::max(rank, shape->dims.size());
  }
  std::vector<SymExpr> dims(rank, SymExpr::constant(1));
  for (size_t i = 0; i < rank; ++i) {
    auto& out = dims[rank - 1 - i];
    for (const auto* shape : operands) {
      if (i >= shape->dims.size()) {
        continue;
      }
      const auto& dim = shape->dims[shape->dims.size() - 1 - i];
      if (isOne(dim)) {
        continue;
      }
      if (isOne(out)) {
        out = dim;
      } else if (!sameDim(out, dim)) {
        return 0;
      } else if (out.isConstant()) {
        out = dim;
      }
    }
  }
  return setOutput(shapes, node, std::move(dims));
}

} // namespace

void registerAtenShapeRules() {
  for (const auto* target :
       {"torch.ops.aten.__lshift__.Scalar",
        "torch.ops.aten.__rshift__.Scalar",
        "torch.ops.aten.bitwise_and.Scalar",
        "torch.ops.aten.bitwise_or.Scalar",
        "torch.ops.aten.bitwise_xor.Scalar",
        "torch.ops.aten.eq.Scalar",
        "torch.ops.aten.ne.Scalar",
        "torch.ops.aten.lt.Scalar",
        "torch.ops.aten.le.Scalar",
        "torch.ops.aten.gt.Scalar",
        "torch.ops.aten.ge.Scalar",
        "torch.ops.aten.floor_divide.Scalar",
        "torch.ops.aten.fmod.Scalar",
        "torch.ops.aten.pow.Tensor_Scalar",
        "torch.ops.aten.clamp_min.default",
        "torch.ops.aten.clamp_max.default",
        "torch.ops.aten.sigmoid.default",
        "torch.ops.aten.relu.default",
        "torch.ops.aten.tanh.default",
        "torch.ops.aten.log.default",
        "torch.ops.aten.log1p.default",
        "torch.ops.aten.exp.default",
        "torch.ops.aten.sqrt.default",
        "torch.ops.aten.rsqrt.default",
        "torch.ops.aten.reciprocal.default",
        "torch.ops.aten.sign.default",
        "torch.ops.aten.floor.default",
        "torch.ops.aten.ceil.default",
        "torch.ops.aten.round.default",
        "torch.ops.aten.isnan.default",
        "torch.ops.aten.logical_not.default",
        "torch.ops.aten.masked_fill.Scalar",
        "torch.ops.aten.fill.Scalar",
        "torch.ops.aten.alias.default",
        "torch.ops.aten.detach.default",
        "torch.ops.aten.contiguous.default",
        "torch.ops.aten.lift_fresh_copy.default",
        "torch.ops.aten.copy.default",
        "torch.ops.aten.logit.default",
        "torch.ops.fb.batch_box_cox.default",
        "tw.select_scatter",
        "tw.slice_scatter",
        "torch.ops.aten.nan_to_num.default",
        "torch.ops.fb.sigrid_hash.default"}) {
    registerSymShapeRule(target, sameShapeRule);
  }
  for (const auto* target :
       {"torch.ops.aten.bitwise_xor.Tensor",
        "torch.ops.aten.__rshift__.Tensor",
        "torch.ops.aten.__or__.Tensor",
        "torch.ops.aten.__and__.Tensor",
        "torch.ops.aten.__xor__.Tensor",
        "torch.ops.aten.__lshift__.Tensor",
        "torch.ops.aten.maximum.default",
        "torch.ops.aten.minimum.default",
        "torch.ops.aten.floor_divide.default",
        "torch.ops.aten.div.Tensor",
        "torch.ops.aten.div.Tensor_mode",
        "torch.ops.aten.remainder.Tensor",
        "torch.ops.aten.fmod.Tensor",
        "torch.ops.aten.pow.Tensor_Tensor",
        "torch.ops.aten.logical_and.default",
        "torch.ops.aten.logical_or.default",
        "torch.ops.aten.masked_fill.Tensor",
        "torch.ops.aten.where.ScalarOther",
        "torch.ops.aten.where.ScalarSelf"}) {
    registerSymShapeRule(target, broadcastRule);
  }
  registerSymShapeRule("torch.ops.aten.slice.Tensor", sliceRule);
  registerSymShapeRule("torch.ops.aten.select.int", selectRule);
  registerSymShapeRule("torch.ops.aten.unsqueeze.default", unsqueezeRule);
  for (const auto* target :
       {"torch.ops.aten.squeeze.default",
        "torch.ops.aten.squeeze.dim",
        "torch.ops.aten.squeeze.dims"}) {
    registerSymShapeRule(target, squeezeRule);
  }
  registerSymShapeRule("torch.ops.aten.cat.default", catRule);
  registerSymShapeRule("torch.ops.aten.stack.default", stackRule);
  for (const auto* target :
       {"torch.ops.aten.view.default",
        "torch.ops.aten.reshape.default",
        "torch.ops.aten._unsafe_view.default"}) {
    registerSymShapeRule(target, viewRule);
  }
  registerSymShapeRule("torch.ops.aten.expand.default", expandRule);
  for (const auto* target :
       {"torch.ops.aten.new_full.default",
        "torch.ops.aten.new_ones.default",
        "torch.ops.aten.new_zeros.default",
        "torch.ops.aten.new_empty.default",
        "torch.ops.aten.full.default",
        "torch.ops.aten.zeros.default",
        "torch.ops.aten.ones.default",
        "torch.ops.aten.empty.memory_format"}) {
    registerSymShapeRule(target, sizeListRule);
  }
  for (const auto* target :
       {"torch.ops.aten.scalar_tensor.default",
        "torch.ops.aten.max.default",
        "torch.ops.aten.min.default",
        "torch.ops.aten.sum.default",
        "torch.ops.aten.any.default",
        "torch.ops.aten.all.default"}) {
    registerSymShapeRule(target, scalarTensorRule);
  }
  registerSymShapeRule("torch.ops.aten.index_select.default", indexSelectRule);
  registerSymShapeRule("tw.index_select", indexSelectRule);
  registerSymShapeRule("torch.ops.aten.permute.default", permuteRule);
  registerSymShapeRule("torch.ops.aten.transpose.int", transposeRule);
  registerSymShapeRule("torch.ops.aten.t.default", transposeRule);
  registerSymShapeRule("torch.ops.aten.unbind.int", unbindRule);
  registerSymShapeRule(
      "torch.ops.aten.split_with_sizes.default", splitWithSizesRule);
  registerSymShapeRule("torch.ops.aten.split.Tensor", splitRule);
  registerSymShapeRule("torch.ops.aten.index.Tensor", indexRule);
  registerSymShapeRule("torch.ops.aten.searchsorted.Tensor", searchsortedRule);
  registerSymShapeRule("tw.searchsorted", searchsortedRule);
  registerSymShapeRule("tw.bucketize", sameShapeRule);
  registerSymShapeRule("tw.cumsum_rows", sameShapeRule);
  registerSymShapeRule("tw.min_rows", allButLastLikeSelfRule);
  registerSymShapeRule("tw.gather", indexShapedRule);
  registerSymShapeRule("torch.ops.aten.gather.default", indexShapedRule);
  registerSymShapeRule("tw.repeat_interleave_index_final", lengthOperandRule);
  registerSymShapeRule("tw.unpack_slot", sizeOperandRule);
  registerSymShapeRule("tw.isin_head", isinHeadRule);
  registerSymShapeRule(
      "torch.ops.aten.exclusive_sum.default", completeCumsumRule);
  registerSymShapeRule("torch.ops.aten.sort.default", allLikeSelfRule);
  registerSymShapeRule(
      "torch.ops.aten.masked_select.default", selectedCountRule);
  registerSymShapeRule("torch.ops.aten.nonzero.default", selectedCountRule);
  registerSymShapeRule("torch.ops.aten.sort.stable", allLikeSelfRule);
  registerSymShapeRule("torch.ops.aten.isin.Tensor_Tensor", elementsShapedRule);
  registerSymShapeRule("tw.scatter", sameShapeRule);
  registerSymShapeRule("tw.scatter_add", sameShapeRule);
  registerSymShapeRule("torch.ops.aten.pow.Scalar", exponentShapedRule);
  registerSymShapeRule("torch.ops.aten.diagonal.default", diagonalRule);
  registerSymShapeRule("tw.isin_final", elementsShapedRule);
  registerSymShapeRule("tw.repeat_interleave_head", repeatInterleaveHeadRule);
  registerSymShapeRule("tw.repeat_interleave_final", repeatInterleaveFinalRule);
  registerSymShapeRule(
      "torch.ops.aten.repeat_interleave.self_Tensor",
      repeatInterleaveFinalRule);
  registerSymShapeRule("torch.ops.aten.pad.default", padRule);
  registerSymShapeRule("tw.pad_last_dim", padRule);
  for (const auto* target :
       {"torch.ops.aten.arange.default",
        "torch.ops.aten.arange.start",
        "torch.ops.aten.arange.start_step"}) {
    registerSymShapeRule(target, arangeRule);
  }
  registerSymShapeRule(
      "torch.ops.aten.repeat_interleave.Tensor", lengthOperandRule);
  registerSymShapeRule("torch.ops.aten.repeat.default", repeatRule);
  for (const auto* target :
       {"torch.ops.aten.sum.dim_IntList",
        "torch.ops.aten.mean.dim",
        "torch.ops.aten.amax.default",
        "torch.ops.aten.amin.default",
        "torch.ops.aten.any.dim",
        "torch.ops.aten.any.dims",
        "torch.ops.aten.max.dim",
        "torch.ops.aten.min.dim"}) {
    registerSymShapeRule(target, reduceRule);
  }
  registerSymShapeRule("torch.ops.aten.sym_size.int", symSizeRule);
}

} // namespace torch::wave
