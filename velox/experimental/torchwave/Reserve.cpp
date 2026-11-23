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

#include "velox/experimental/torchwave/Reserve.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <variant>

#include "velox/experimental/torchwave/KernelOperation.h"
#include "velox/experimental/torchwave/WaveGraph.h"

namespace torch::wave {

namespace {

int32_t actualIdOf(ValueCP value, const FormalToActual& bindings) {
  auto it = bindings.find(value->id());
  return it != bindings.end() ? it->second : value->id();
}

ValueCP namedInput(NodeCP node, const OperandName& name) {
  if (name.ordinal >= 0) {
    return static_cast<size_t>(name.ordinal) < node->inputs().size()
        ? node->inputs()[name.ordinal].value
        : nullptr;
  }
  const auto* input = node->tryGetInput(name.name);
  return input != nullptr ? input->value : nullptr;
}

// An integer attribute of the invoked node.
std::optional<int64_t> intAttribute(NodeCP node, const char* name) {
  const auto* attr = node->tryGetAttribute(name);
  if (attr == nullptr) {
    return std::nullopt;
  }
  if (std::holds_alternative<int64_t>(attr->value)) {
    return std::get<int64_t>(attr->value);
  }
  if (std::holds_alternative<double>(attr->value)) {
    return static_cast<int64_t>(std::ceil(std::get<double>(attr->value)));
  }
  if (std::holds_alternative<bool>(attr->value)) {
    return std::get<bool>(attr->value) ? 1 : 0;
  }
  return std::nullopt;
}

} // namespace

std::vector<int32_t> hostTensorIds(
    const OutputDesc& desc,
    const FormalToActual& bindings) {
  std::vector<int32_t> ids;
  if (desc.reserveFn == nullptr || desc.reserveInputs == nullptr ||
      desc.reserveNode == nullptr) {
    return ids;
  }
  for (const auto& name : desc.reserveInputs->hostTensors) {
    auto* value = namedInput(desc.reserveNode, name);
    if (value != nullptr) {
      ids.push_back(actualIdOf(value, bindings));
    }
  }
  return ids;
}

bool bindReserve(
    const OutputDesc& desc,
    const FormalToActual& bindings,
    const NodeMap& nodeMap,
    ReserveBinding& args) {
  const auto& inputs = *desc.reserveInputs;
  auto* node = desc.reserveNode;
  auto* attrNode = desc.reserveAttrNode;
  auto it = nodeMap.find(attrNode);
  auto* invoked = it != nodeMap.end() ? it->second : attrNode;
  args.tensors.clear();
  args.scalars.clear();
  args.intLists.clear();
  args.hostTensors.clear();
  for (const auto& name : inputs.hostTensors) {
    auto* value = namedInput(node, name);
    if (value == nullptr ||
        value->type().kind() != nativert::Type::Kind::Tensor) {
      return false;
    }
    args.hostTensors.push_back(actualIdOf(value, bindings));
  }
  for (const auto& name : inputs.tensors) {
    auto* value = namedInput(node, name);
    if (value == nullptr) {
      return false;
    }
    auto& group = args.tensors.emplace_back();
    if (value->type().kind() == nativert::Type::Kind::TensorList) {
      for (auto* element : value->getListElements()) {
        group.push_back(actualIdOf(element, bindings));
      }
    } else {
      group.push_back(actualIdOf(value, bindings));
    }
  }
  for (const auto& name : inputs.scalars) {
    auto* value = namedInput(name.ordinal >= 0 ? node : attrNode, name);
    if (value != nullptr) {
      args.scalars.push_back(
          {.kind = ScalarRef::kValue, .valueId = actualIdOf(value, bindings)});
      continue;
    }
    auto constant =
        name.name != nullptr ? intAttribute(invoked, name.name) : std::nullopt;
    if (!constant.has_value()) {
      return false;
    }
    args.scalars.push_back(
        {.kind = ScalarRef::kConstant, .constant = *constant});
  }
  for (const auto& name : inputs.optionalScalars) {
    auto* value = namedInput(name.ordinal >= 0 ? node : attrNode, name);
    if (value != nullptr) {
      args.scalars.push_back(
          {.kind = ScalarRef::kValue, .valueId = actualIdOf(value, bindings)});
      continue;
    }
    auto constant =
        name.name != nullptr ? intAttribute(invoked, name.name) : std::nullopt;
    args.scalars.push_back(
        constant.has_value()
            ? ScalarRef{.kind = ScalarRef::kConstant, .constant = *constant}
            : ScalarRef{.kind = ScalarRef::kAbsent});
  }
  if (inputs.elementwiseLeaves >= 0) {
    if (static_cast<size_t>(inputs.elementwiseLeaves) >=
        node->inputs().size()) {
      return false;
    }
    auto& leaves = args.tensors.emplace_back();
    std::function<void(ValueCP)> trace = [&](ValueCP value) {
      if (bindings.count(value->id()) != 0 &&
          value->type().kind() == nativert::Type::Kind::Tensor) {
        leaves.push_back(bindings.at(value->id()));
        return;
      }
      auto* producer = value->producer();
      if (producer == nullptr) {
        return;
      }
      const auto* meta = Registry::metadata(producer->target());
      if (meta != nullptr && meta->elementwise) {
        for (const auto& input : producer->inputs()) {
          trace(input.value);
        }
      }
    };
    trace(node->inputs()[inputs.elementwiseLeaves].value);
  }
  for (const auto& name : inputs.intLists) {
    auto& list = args.intLists.emplace_back();
    auto* value = namedInput(name.ordinal >= 0 ? node : attrNode, name);
    if (value != nullptr) {
      // Only a list packed from scalars has a table form; its elements are
      // actual values already.
      const auto& idToValue = waveGraph()->idToValue();
      auto valueIt = idToValue.find(actualIdOf(value, bindings));
      if (valueIt == idToValue.end()) {
        return false;
      }
      auto* producer = valueIt->second->producer();
      if (producer == nullptr || producer->target() != "prim.ListPack") {
        return false;
      }
      for (const auto& element : producer->inputs()) {
        list.push_back(
            {.kind = ScalarRef::kValue,
             .valueId = actualIdOf(element.value, bindings)});
      }
      continue;
    }
    if (name.name == nullptr) {
      return false;
    }
    const auto* attr = invoked->tryGetAttribute(name.name);
    if (attr == nullptr ||
        !std::holds_alternative<std::vector<int64_t>>(attr->value)) {
      return false;
    }
    for (auto element : std::get<std::vector<int64_t>>(attr->value)) {
      list.push_back({.kind = ScalarRef::kConstant, .constant = element});
    }
  }
  return true;
}

namespace {

// The value's tensor if the table holds one this execution, with its fields
// built.
const TableTensor* tensorOperand(const ReserveArgs& args, int32_t id) {
  const auto& table = *args.table;
  return table.valid(id) && table.isTensor(id) ? &table.tensor(id) : nullptr;
}

bool intOperand(const ReserveArgs& args, int32_t id) {
  const auto& table = *args.table;
  return table.valid(id) && table.isInt(id);
}

// The ids of tensor group 0, which holds a built-in's values.
ArrayView<int32_t> values(const ReserveArgs& args) {
  return {args.ids + args.groups[0].begin, args.groups[0].size};
}

// Folds 'dims' of 'rank' into the right-aligned broadcast in 'result'. A size-1
// dim takes the other side's extent, including 0.
void combineBroadcast(Shape& result, const Dim* dims, int32_t rank) {
  if (rank > result.rank) {
    const int32_t shift = rank - result.rank;
    for (int32_t i = result.rank - 1; i >= 0; --i) {
      result.dims[i + shift] = result.dims[i];
    }
    for (int32_t i = 0; i < shift; ++i) {
      result.dims[i] = 1;
    }
    result.rank = static_cast<int8_t>(rank);
  }
  for (int32_t i = 0; i < rank; ++i) {
    auto& resultDim = result.dims[result.rank - rank + i];
    if (resultDim == 1) {
      resultDim = dims[i];
    } else if (dims[i] != 1) {
      resultDim = std::max<Dim>(resultDim, dims[i]);
    }
  }
}

void broadcastShape(const ReserveArgs& args, Shape& result) {
  result.rank = 0;
  Dim dims[kMaxDims];
  for (auto id : values(args)) {
    if (const auto* t = tensorOperand(args, id)) {
      for (int32_t i = 0; i < t->rank; ++i) {
        dims[i] = static_cast<Dim>(t->dims[i]);
      }
      combineBroadcast(result, dims, t->rank);
    }
  }
  for (int32_t l = 0; l < args.numIntLists(); ++l) {
    const auto constShape = args.intList(l);
    for (size_t i = 0; i < constShape.size(); ++i) {
      dims[i] = static_cast<Dim>(constShape[i].constant);
    }
    combineBroadcast(result, dims, static_cast<int32_t>(constShape.size()));
  }
  if (result.rank == 0) {
    result.setVector(1);
  }
}

bool noOperands(const ReserveArgs& args) {
  return args.groupSize(0) == 0 && args.numIntLists() == 0;
}

} // namespace

bool maxCountReserve(const ReserveArgs& args, Shape* out, int32_t /*numOut*/) {
  if (noOperands(args)) {
    out[0].setVector(1);
    return true;
  }
  int64_t result = 0;
  for (auto id : values(args)) {
    if (const auto* t = tensorOperand(args, id)) {
      result = std::max<int64_t>(result, t->numEl);
    } else if (intOperand(args, id)) {
      result = std::max<int64_t>(result, args.table->scalar(id));
    }
  }
  out[0].setVector(static_cast<Dim>(result));
  return true;
}

bool sumCountReserve(const ReserveArgs& args, Shape* out, int32_t /*numOut*/) {
  if (noOperands(args)) {
    out[0].setVector(1);
    return true;
  }
  int64_t result = 0;
  for (auto id : values(args)) {
    if (const auto* t = tensorOperand(args, id)) {
      result += t->numEl;
    } else if (intOperand(args, id)) {
      result += args.table->scalar(id);
    }
  }
  out[0].setVector(static_cast<Dim>(result));
  return true;
}

bool broadcastCountReserve(
    const ReserveArgs& args,
    Shape* out,
    int32_t /*numOut*/) {
  if (noOperands(args)) {
    out[0].setVector(1);
    return true;
  }
  Shape shape;
  broadcastShape(args, shape);
  out[0].setVector(static_cast<Dim>(shape.numElements()));
  return true;
}

bool maxShapeReserve(const ReserveArgs& args, Shape* out, int32_t /*numOut*/) {
  if (noOperands(args)) {
    out[0].setVector(1);
    return true;
  }
  int64_t bestNumel = -1;
  out[0].rank = -1;
  for (auto id : values(args)) {
    if (const auto* t = tensorOperand(args, id)) {
      if (static_cast<int64_t>(t->numEl) > bestNumel) {
        bestNumel = t->numEl;
        out[0].setLike(*t);
      }
    }
  }
  return true;
}

bool largestInputReserve(
    const ReserveArgs& args,
    Shape* out,
    int32_t /*numOut*/) {
  out[0].rank = -1;
  if (noOperands(args) || args.numIntLists() != 0) {
    return true;
  }
  int64_t result = 0;
  for (auto id : values(args)) {
    if (const auto* t = tensorOperand(args, id)) {
      if (static_cast<int64_t>(t->numEl) > result) {
        result = t->numEl;
        out[0].setLike(*t);
      }
    } else if (intOperand(args, id) && args.table->scalar(id) > result) {
      result = args.table->scalar(id);
    }
  }
  return true;
}

bool broadcastShapeReserve(
    const ReserveArgs& args,
    Shape* out,
    int32_t /*numOut*/) {
  if (noOperands(args)) {
    out[0].setVector(1);
    return true;
  }
  broadcastShape(args, out[0]);
  return true;
}

namespace {
bool scalarReady(const ScalarRef& ref, const ValueTable& table) {
  return ref.kind != ScalarRef::kValue ||
      (table.valid(ref.valueId) && table.isInt(ref.valueId));
}

} // namespace

bool reserveOperandsReady(const ReserveArgs& args) {
  const auto& table = *args.table;
  for (int32_t g = 0; g < args.numGroups; ++g) {
    for (size_t e = 0; e < args.groupSize(g); ++e) {
      const auto id = args.id(g, e);
      if (!table.valid(id) || !table.isTensor(id)) {
        return false;
      }
      table.tensor(id);
    }
  }
  for (int32_t i = 0; i < args.numScalars; ++i) {
    if (!scalarReady(args.scalars[i], table)) {
      return false;
    }
  }
  for (int32_t l = 0; l < args.numLists; ++l) {
    for (const auto& ref : args.intList(l)) {
      if (!scalarReady(ref, table)) {
        return false;
      }
    }
  }
  for (int32_t i = 0; i < args.numHost; ++i) {
    const auto& cold = table.cold[args.hostIds[i]];
    if (cold.hostData == nullptr || cold.hostGeneration != table.generation) {
      return false;
    }
  }
  return true;
}

} // namespace torch::wave
