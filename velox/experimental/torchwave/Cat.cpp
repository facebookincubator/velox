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

#include "velox/experimental/torchwave/Cat.h"
#include "velox/experimental/torchwave/Compile.h"
#include "velox/experimental/torchwave/KernelOperation.h"
#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"
#include "velox/experimental/torchwave/WaveGraph.h"

#include <ATen/ATen.h>
#include <fmt/format.h>
#include <folly/CppAttributes.h>
#include <gflags/gflags.h>
#include <algorithm>
#include <iostream>

// elt_trace is now WaveConfig::kernelDebugOutput

namespace torch::wave {

namespace {

constexpr std::string_view kCatTarget = "torch.ops.aten.cat.default";
constexpr std::string_view kStackTarget = "torch.ops.aten.stack.default";

SizeExpr translateSizeExpr(const SizeExpr& expr, const FormalToActual& map) {
  SizeExpr result;
  result.op = expr.op;
  // Preserve broadcast and constShapes: a pure factory operand (e.g.
  // zeros(size=[1000])) carries its extent in constShapes with no tensor
  // leaves, so dropping them collapses its size to the scalar default.
  result.broadcast = expr.broadcast;
  result.constShapes = expr.constShapes;
  for (auto id : expr.values) {
    auto it = map.find(id);
    result.values.push_back(it != map.end() ? it->second : id);
  }
  for (const auto& child : expr.args) {
    result.args.push_back(translateSizeExpr(child, map));
  }
  return result;
}

// The 'dim' attribute of a concat, still in torch's (possibly negative) form.
// Defaults to 0, matching both schemas.
int64_t concatDimAttribute(NodeCP node) {
  const auto* attr = node->tryGetAttribute("dim");
  if (attr && std::holds_alternative<int64_t>(attr->value)) {
    return std::get<int64_t>(attr->value);
  }
  return 0;
}

ConcatSpec concatSpec(NodeCP node, const ValueTypes& types) {
  ConcatSpec spec;
  spec.isStack = node->target() == kStackTarget;
  int64_t dim = concatDimAttribute(node);
  const auto& inputs = node->inputs();
  if (!inputs.empty() &&
      inputs[0].value->type().kind() == nativert::Type::Kind::TensorList) {
    auto elements = inputs[0].value->getListElements();
    if (!elements.empty()) {
      auto elementRank = types.rank(elements[0]);
      if (elementRank >= 0) {
        spec.outRank =
            spec.isStack ? static_cast<int8_t>(elementRank + 1) : elementRank;
      }
    }
  }
  if (dim < 0 && spec.outRank > 0) {
    dim += spec.outRank;
  }
  spec.dim = static_cast<int32_t>(dim);
  return spec;
}

// Rewrites a negative 'dim' attribute to its non-negative form. Subgraph
// deduplication compares the raw attribute value, and the generated code bakes
// the axis in, so a dim of -1 on a 2-d concat and on a 3-d concat must not look
// alike. Called from maybeReplace, once the operand ranks are known.
void normalizeConcatDim(NodeCP node, const ValueTypes& types) {
  auto dim = concatDimAttribute(node);
  if (dim >= 0) {
    return;
  }
  auto outRank = concatSpec(node, types).outRank;
  if (outRank <= 0) {
    return;
  }
  auto* attr = const_cast<nativert::Attribute*>(node->tryGetAttribute("dim"));
  if (attr) {
    attr->value = dim + outRank;
  }
}

// Falls back to the eager op whenever the fused form cannot lay the result out:
// an unknown or too-large rank, a join axis outside the result, a dim that is
// only known at run time, or operands of differing rank (torch's legacy
// empty-operand cat).
bool concatIsStandalone(NodeCP node, const ValueTypes& types) {
  const auto& inputs = node->inputs();
  if (inputs.empty() ||
      inputs[0].value->type().kind() != nativert::Type::Kind::TensorList) {
    return true;
  }
  auto elements = inputs[0].value->getListElements();
  if (elements.empty()) {
    return true;
  }
  // The generated code bakes in the axis, so it must be a constant attribute.
  if (node->tryGetInput("dim") != nullptr) {
    return true;
  }
  auto spec = concatSpec(node, types);
  if (spec.outRank < 1 || spec.outRank > kMaxDims) {
    return true;
  }
  if (spec.dim < 0 || spec.dim >= spec.outRank) {
    return true;
  }
  auto elementRank = spec.elementRank();
  // A stack over 0-d operands is well formed -- N scalars become a 1-d,
  // N-element result -- but a cat needs an existing axis to join along.
  if (elementRank < 0 || (elementRank == 0 && !spec.isStack)) {
    return true;
  }
  for (auto* element : elements) {
    if (types.rank(element) != elementRank) {
      return true;
    }
  }
  return false;
}

std::vector<ValueConstraint> concatOutputConstraints(
    NodeCP node,
    const ValueTypes& types) {
  const auto& inputs = node->inputs();
  if (inputs.empty()) {
    return {};
  }
  auto elements = inputs[0].value->getListElements();
  if (elements.empty()) {
    return {};
  }
  auto rank = types.rank(elements[0]);
  if (node->target() == kStackTarget && rank >= 0) {
    // A stack concatenates along a new dim, so rank = element rank + 1.
    rank = static_cast<int8_t>(rank + 1);
  }
  // Both ops materialize a fresh, densely-laid-out output.
  return {{.rank = rank, .contiguity = Contiguity::kContiguous}};
}

// Returns true if any node in the producer chain of 'node' (stopping at
// subgraphInputs) has a shapeSetOnDevice return.
bool hasShapeOnDeviceInChain(
    NodeCP node,
    const std::unordered_set<ValueCP>& subgraphInputs,
    std::unordered_set<NodeCP>& visited) {
  if (!visited.insert(node).second) {
    return false;
  }
  auto* meta = Registry::metadata(node->target());
  if (meta) {
    for (const auto& rm : meta->returnMeta) {
      if (rm.shapeSetOnDevice) {
        return true;
      }
    }
  }
  for (const auto& input : node->inputs()) {
    if (subgraphInputs.count(input.value)) {
      continue;
    }
    auto* producer = input.value->producer();
    if (producer &&
        hasShapeOnDeviceInChain(producer, subgraphInputs, visited)) {
      return true;
    }
  }
  return false;
}

// Returns true if any node in the producer chain of 'node' (stopping at
// subgraphInputs) sizes a return with a reserve function. See
// ConcatInputInfo::hasReserveInChain. The walk stops at subgraph inputs on
// purpose: an earlier kernel's output is in the frame by the time the group is
// carved, however its shape was arrived at.
bool hasReserveShapeInChain(
    NodeCP node,
    const std::unordered_set<ValueCP>& subgraphInputs,
    std::unordered_set<NodeCP>& visited) {
  if (!visited.insert(node).second) {
    return false;
  }
  auto* meta = Registry::metadata(node->target());
  if (meta) {
    for (const auto& rm : meta->returnMeta) {
      if (rm.reserveShape != nullptr) {
        return true;
      }
    }
  }
  for (const auto& input : node->inputs()) {
    if (subgraphInputs.count(input.value)) {
      continue;
    }
    auto* producer = input.value->producer();
    if (producer && hasReserveShapeInChain(producer, subgraphInputs, visited)) {
      return true;
    }
  }
  return false;
}

// The launch-time shape of one operand, coerced to 'rank' dimensions.
std::vector<Dim> concatInputShape(
    const ConcatInputInfo& info,
    int8_t rank,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map) {
  std::vector<Dim> shape;
  if (info.reserveShape) {
    static const NodeMap emptyNodeMap;
    auto shapes = info.reserveShape(nullptr, frame, map, nullptr, emptyNodeMap);
    TORCH_CHECK(
        !shapes.empty(), "reserveShape returned no shape for a concat operand");
    shape = std::move(shapes[0]);
  } else if (info.sizeExpr.op != SizeShortcut::kNone) {
    shape = translateSizeExpr(info.sizeExpr, map).dims(&frame);
  }
  if (shape.empty()) {
    // No size expression resolved: the operand contributes nothing, as for the
    // undefined tensor a zero-length cat operand arrives as.
    return std::vector<Dim>(rank, 0);
  }
  auto shapeRank = static_cast<int8_t>(shape.size());
  if (shapeRank == rank) {
    return shape;
  }
  if (rank == 1) {
    Dim numElements = 1;
    for (auto extent : shape) {
      numElements *= extent;
    }
    return {numElements};
  }
  TORCH_CHECK(
      shapeRank < rank,
      "Concat operand %",
      info.valueId,
      " resolved to a rank-",
      static_cast<int>(shapeRank),
      " shape, expected rank ",
      static_cast<int>(rank));
  // A broadcast size expression drops leading 1-dims; restore them.
  shape.insert(shape.begin(), rank - shapeRank, 1);
  return shape;
}

// Computes the operand shapes, allocates or resizes the concat output, and
// hands every operand this kernel computes a view of the output region it
// occupies, so the producing expression writes its result in place. Operands
// the kernel copies (boundary inputs and views) get no view; __concatCopy moves
// them at kernel time.
std::vector<std::vector<Dim>> reserveConcatOutput(
    const std::vector<ConcatInputInfo>& inputInfos,
    const ConcatSpec& spec,
    nativert::ValueId concatFormalId,
    c10::ScalarType dtype,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map) {
  const auto elementRank = spec.elementRank();
  std::vector<std::vector<Dim>> shapes;
  shapes.reserve(inputInfos.size());
  for (const auto& info : inputInfos) {
    shapes.push_back(concatInputShape(info, elementRank, frame, map));
  }
  auto outShape = concatResultShape(spec, shapes);

  auto concatActualId = concatFormalId;
  if (auto it = map.find(concatFormalId); it != map.end()) {
    concatActualId = it->second;
  }
  const std::vector<int64_t> outSizes(outShape.begin(), outShape.end());
  auto& existing = frame.getIValue(concatActualId);
  at::Tensor concatTensor;
  if (existing.isTensor() && existing.toTensor().is_cuda() &&
      (existing.toTensor().sizes() == c10::IntArrayRef(outSizes) ||
       existing.toTensor().storage().use_count() == 1)) {
    concatTensor = existing.toTensor();
    if (concatTensor.sizes() != c10::IntArrayRef(outSizes)) {
      concatTensor.resize_(outSizes);
    }
  } else {
    // Either there is nothing to keep, or the result is a different shape than
    // the tensor already there AND something else holds its storage -- views of
    // it a concat allocation group carved for the operands, which resize_ would
    // reallocate out from under. Those views stay valid on their own reference,
    // so starting again leaves the operands to be copied in rather than losing
    // what they wrote.
    concatTensor =
        at::empty(outSizes, at::TensorOptions().dtype(dtype).device(at::kCUDA));
    frame.setIValue(concatActualId, concatTensor);
  }

  bool canComputeOffset = true;
  int64_t offset = 0;
  for (size_t i = 0; i < inputInfos.size(); ++i) {
    // 'shapes' is filled one entry per inputInfos above, so .at(i) is in range;
    // spelled as a checked access so the invariant is enforced rather than
    // assumed.
    const int64_t extent =
        spec.isStack ? 1 : static_cast<int64_t>(shapes.at(i).at(spec.dim));
    if (inputInfos[i].isSubgraphInput || inputInfos[i].isView) {
      offset += extent;
      continue;
    }
    auto inputActualId = inputInfos[i].valueId;
    if (auto it = map.find(inputInfos[i].valueId); it != map.end()) {
      inputActualId = it->second;
    }
    // An earlier operand whose length the kernel itself computes leaves this
    // one's offset unknown on the host; the kernel then patches the view's base
    // (see the view fixup in concatSpecialForm). Only a 1-d cat gets here -- a
    // wider concat keeps device-sized operands out of its kernel entirely.
    const bool pending = !canComputeOffset || inputInfos[i].hasShapeOnDevice;
    // A cat operand spans 'extent' positions along the join axis; a stack
    // operand occupies the single position 'i', which drops that axis.
    const int64_t start =
        spec.isStack ? static_cast<int64_t>(i) : (pending ? 0 : offset);
    auto view = concatOperandView(concatTensor, spec, start, extent);
    if (WaveConfig::get().trace & WaveConfig::kTensors) {
      std::cout << "  concat view v" << inputActualId << " of v"
                << concatActualId << " dim=" << spec.dim << " offset=";
      if (pending) {
        std::cout << "pending";
      } else {
        std::cout << offset;
      }
      std::cout << " size=" << extent << " " << traceIValue(c10::IValue(view))
                << std::endl;
    }
    frame.setIValue(inputActualId, std::move(view));
    if (pending) {
      canComputeOffset = false;
    } else {
      offset += extent;
    }
  }

  return {std::move(outShape)};
}

void concatSetOutputs(
    KernelOperation* op,
    NodeCP node,
    const std::unordered_set<ValueCP>& subgraphInputs,
    std::vector<ValueCP>& outputValues,
    std::vector<OutputDesc>& outputDescs,
    bool /*inMemory*/,
    bool /*callerIsElementwise*/) {
  auto elements = node->inputs()[0].value->getListElements();
  auto& types = waveGraph()->types();
  auto spec = concatSpec(node, types);

  // Call setOutputs for each element's producer, forcing to memory.
  for (auto* elem : elements) {
    auto* producer = elem->producer();
    if (producer && !subgraphInputs.count(elem)) {
      op->setOutputs(
          producer, subgraphInputs, outputValues, outputDescs, true, false);
    }
  }

  std::vector<ConcatInputInfo> inputInfos;
  inputInfos.reserve(elements.size());
  for (auto* elem : elements) {
    if (subgraphInputs.count(elem)) {
      SizeExpr sizeExpr;
      sizeExpr.op = SizeShortcut::kMax;
      sizeExpr.values.push_back(elem->id());
      inputInfos.push_back(
          {.valueId = elem->id(),
           .sizeExpr = std::move(sizeExpr),
           .isSubgraphInput = true});
      continue;
    }
    int32_t descIdx = -1;
    for (size_t i = 0; i < outputValues.size(); ++i) {
      if (outputValues[i] == elem) {
        descIdx = static_cast<int32_t>(i);
        break;
      }
    }
    TORCH_CHECK(
        descIdx >= 0 && static_cast<size_t>(descIdx) < outputDescs.size(),
        "No OutputDesc created for cat input ",
        elem->id());
    auto& desc = outputDescs.at(descIdx);
    desc.delegated = true;

    std::unordered_set<NodeCP> visited;
    bool hasSod = desc.shapeSetOnDevice;
    if (!hasSod && elem->producer()) {
      hasSod =
          hasShapeOnDeviceInChain(elem->producer(), subgraphInputs, visited);
    }

    std::unordered_set<NodeCP> reserveVisited;
    bool hasReserveInChain = false;
    if (elem->producer() != nullptr) {
      hasReserveInChain = hasReserveShapeInChain(
          elem->producer(), subgraphInputs, reserveVisited);
    }

    bool elemIsView = desc.viewNode != nullptr;
    if (!elemIsView && elem->producer()) {
      auto* producerMeta = Registry::metadata(elem->producer()->target());
      if (producerMeta && producerMeta->isView()) {
        elemIsView = true;
      }
    }
    auto inputSizeExpr = desc.sizeExpr;
    if (elemIsView && inputSizeExpr.op == SizeShortcut::kNone) {
      inputSizeExpr.op = SizeShortcut::kMax;
      inputSizeExpr.values.push_back(elem->id());
    }
    OutputReserveFunc catReserve;
    if (desc.reserveShape) {
      auto inner = desc.reserveShape;
      catReserve = [inner](
                       NodeCP /*unused*/,
                       nativert::ExecutionFrame& frame,
                       const FormalToActual& map,
                       NodeCP /*originalFormalNode*/,
                       const NodeMap& nodeMap) {
        return inner(frame, map, nodeMap);
      };
    }
    inputInfos.push_back(
        {.valueId = elem->id(),
         .sizeExpr = std::move(inputSizeExpr),
         .reserveShape = std::move(catReserve),
         .hasShapeOnDevice = hasSod,
         .hasReserveInChain = hasReserveInChain,
         .mayWriteStrided = producerMayWriteStrided(elem),
         .isSubgraphInput = false,
         .isView = elemIsView});
  }

  // Create the concat output desc.
  auto concatOutputValue = node->outputs()[0];
  auto concatFormalId = concatOutputValue->id();

  TORCH_CHECK(
      concatFormalId >= 0 &&
          static_cast<size_t>(concatFormalId) < types.types.size() &&
          types.types[concatFormalId],
      "No TensorMeta for cat output value ",
      concatFormalId);
  auto dtype = types.types[concatFormalId]->dtype();

  OutputDesc concatDesc;
  concatDesc.sizeExpr.op = SizeShortcut::kSum;
  for (auto* elem : elements) {
    concatDesc.sizeExpr.values.push_back(elem->id());
  }
  for (const auto& info : inputInfos) {
    if (info.hasShapeOnDevice) {
      concatDesc.shapeSetOnDevice = true;
      break;
    }
  }
  TORCH_CHECK(
      spec.outRank == 1 || !concatDesc.shapeSetOnDevice,
      node->target(),
      ": an operand's extent is computed on device inside the concat's own "
      "kernel, which a rank-",
      static_cast<int>(spec.outRank),
      " result cannot lay out");

  // Subgraph deduplication matches on structure, dtype and the 'dim'
  // attribute, but not on rank, so one KernelOperation can serve both a 2-d and
  // a 3-d concat on the same axis. The generated code is rank-agnostic (it
  // reads dims[] at run time); the host-side layout below is not, so it takes
  // the rank from the node this invocation actually stands for. nodeMap is
  // keyed by the ProjectOperation's formal subgraph, so a grid-variant copy of
  // 'node' has to be mapped back to its original first.
  const auto* valueTypes = &types;
  auto* originalNode = waveGraph()->compileCtx()
      ? waveGraph()->compileCtx()->originalFromVariant(node)
      : node;
  // originalFromVariant returns null for a node that is not a variant copy
  // (it never went through the grid-variant rewrite), in which case the node
  // already is its own original.
  if (originalNode == nullptr) {
    originalNode = node;
  }
  // Also handed to the allocation-group pass, which recognizes the concat from
  // it and can then place the whole result before any operand is produced. The
  // reserve below is one of its two readers, so the two cannot describe
  // different operands.
  auto layout = std::make_shared<ConcatLayout>(ConcatLayout{
      .spec = spec,
      .dtype = dtype,
      .inputs = std::move(inputInfos),
      .outputFormalId = concatFormalId,
      .originalNode = originalNode,
      .types = valueTypes});

  concatDesc.reserveShape =
      [layout, concatFormalId](
          nativert::ExecutionFrame& frame,
          const FormalToActual& map,
          const NodeMap& nodeMap) -> std::vector<std::vector<Dim>> {
    auto [actualSpec, actualDtype] = layout->resolve(nodeMap);
    return reserveConcatOutput(
        layout->inputs, actualSpec, concatFormalId, actualDtype, frame, map);
  };
  concatDesc.concatLayout = std::move(layout);

  addOrUpdateOutput(
      outputValues, outputDescs, concatOutputValue, std::move(concatDesc));
}

// Recursively collects the leaf elements of cats nested on the same dim.
void flattenCatElements(
    ValueCP value,
    int64_t dim,
    std::vector<nativert::Value*>& result) {
  if (!value) {
    return;
  }
  auto* producer = value->producer();
  if (producer && producer->target() == kCatTarget &&
      concatDimAttribute(producer) == dim) {
    auto elements = producer->inputs()[0].value->getListElements();
    for (auto* elem : elements) {
      flattenCatElements(elem, dim, result);
    }
  } else {
    result.push_back(const_cast<nativert::Value*>(value));
  }
}

// Returns the cumsum node if 'node' is cat(zeros(size=[1]), cumsum(...)).
NodeCP FOLLY_NULLABLE isExclusiveSumPattern(NodeCP node) {
  auto elements = node->inputs()[0].value->getListElements();
  if (elements.size() != 2) {
    return nullptr;
  }
  auto* zerosProducer = elements[0]->producer();
  if (!zerosProducer ||
      zerosProducer->target() != "torch.ops.aten.zeros.default") {
    return nullptr;
  }
  const auto* sizeAttr = zerosProducer->tryGetAttribute("size");
  if (!sizeAttr ||
      !std::holds_alternative<std::vector<int64_t>>(sizeAttr->value)) {
    return nullptr;
  }
  const auto& size = std::get<std::vector<int64_t>>(sizeAttr->value);
  if (size.size() != 1 || size[0] != 1) {
    return nullptr;
  }
  auto* cumsumProducer = elements[1]->producer();
  if (!cumsumProducer ||
      cumsumProducer->target() != "torch.ops.aten.cumsum.default") {
    return nullptr;
  }
  return cumsumProducer;
}

// True if 'operand' cannot itself fill the region of the result it occupies, so
// something has to move its bytes there. A value with no producer, or one whose
// producer only makes a view, has no write of its own to redirect; a value of
// another dtype cannot be written through a view that would have to convert it;
// and a pitched band needs a producer that indexes its output through strides.
bool concatOperandNeedsCopy(
    ValueCP operand,
    int64_t dim,
    c10::ScalarType resultDtype,
    const ValueTypes& types) {
  auto* producer = operand->producer();
  if (producer == nullptr) {
    return true;
  }
  const auto* producerMeta = Registry::metadata(producer->target());
  if (producerMeta == nullptr || producerMeta->isView()) {
    return true;
  }
  // An operand whose extent is settled on device must not be copied. A clone of
  // it reserves a static shape, which launders the shapeSetOnDevice marking the
  // group relies on to refuse a layout it cannot compute: the host would then
  // lay the result out from a stale extent and the regions would overlap, so
  // one operand's write lands inside another's. Left uncopied, the concat
  // refuses the group instead, which is merely slower.
  for (const auto& returnMeta : producerMeta->returnMeta) {
    if (returnMeta.shapeSetOnDevice) {
      return false;
    }
  }
  const auto operandId = operand->id();
  if (operandId >= 0 && static_cast<size_t>(operandId) < types.types.size() &&
      types.types[operandId] &&
      types.types[operandId]->dtype() != resultDtype) {
    return true;
  }
  return dim != 0 && !producerMayWriteStrided(operand);
}

// Gives every operand that cannot fill its own region of the result a clone of
// its own to fill it with. The clone is a value a kernel writes, so the
// allocation group carves it into the region and the ordinary machinery makes
// it an op of its own, sized by that operand and scheduled beside the rest.
// Without this the concat's kernel would walk those operands one after another
// through a running offset, which is the serialization a wide concat must not
// have. Every occurrence gets its own clone, which also settles cat([x, y, x]):
// the two regions are filled by two different values.
void insertConcatOperandCopies(
    NodeCP node,
    ValueTypes& types,
    WaveGraph& waveGraph) {
  if (!WaveConfig::get().parallelConcatFill) {
    return;
  }
  auto* listValue = node->inputs()[0].value;
  auto* listPack = listValue->producer();
  if (listPack == nullptr || listPack->target() != "prim.ListPack" ||
      listPack->inputs().size() <= 2) {
    return;
  }
  const auto resultId = node->outputs()[0]->id();
  if (resultId < 0 || static_cast<size_t>(resultId) >= types.types.size() ||
      !types.types[resultId]) {
    return;
  }
  const auto resultDtype = types.types[resultId]->dtype();
  const int64_t dim = concatDimAttribute(node);

  auto* graph = waveGraph.graph();
  auto* mutableListPack = const_cast<nativert::Node*>(listPack);
  auto& inputs = mutableListPack->inputs();

  // Collected first: rewiring every occurrence of a value before dropping the
  // user record keeps the two consistent, since eraseUser removes the node from
  // the list outright rather than one use of it.
  std::vector<ValueCP> toCopy;
  for (const auto& input : inputs) {
    auto* operand = input.value;
    if (std::find(toCopy.begin(), toCopy.end(), operand) != toCopy.end()) {
      continue;
    }
    if (concatOperandNeedsCopy(operand, dim, resultDtype, types)) {
      toCopy.push_back(operand);
    }
  }

  for (auto* operand : toCopy) {
    for (auto& input : inputs) {
      if (input.value != operand) {
        continue;
      }
      auto* clone = graph->createNode(
          "torch.ops.aten.clone.default",
          {{"self", const_cast<nativert::Value*>(operand)}});
      graph->insertBefore(clone, mutableListPack);
      auto* copied = waveGraph.newTensorValue(clone, "cat_copy", resultDtype);
      input.value = copied;
      copied->addUser(mutableListPack);
    }
    const_cast<nativert::Value*>(operand)->eraseUser(mutableListPack);
  }
}

std::vector<std::pair<ValueCP, ValueCP>>
concatMaybeReplace(NodeCP node, ValueTypes& types, WaveGraph& waveGraph) {
  normalizeConcatDim(node, types);
  insertConcatOperandCopies(node, types, waveGraph);
  if (node->target() == kStackTarget) {
    return {};
  }
  auto* graph = waveGraph.graph();
  const int64_t dim = concatDimAttribute(node);

  if (auto* cumsumNode = dim == 0 ? isExclusiveSumPattern(node) : nullptr) {
    auto* cumsumInput = cumsumNode->inputs()[0].value;
    if (types.rank(cumsumInput) != 1) {
      return {};
    }
    auto inputId = cumsumInput->id();
    auto dtype = c10::ScalarType::Long;
    const auto* dtypeAttr = cumsumNode->tryGetAttribute("dtype");
    if (dtypeAttr &&
        std::holds_alternative<c10::ScalarType>(dtypeAttr->value)) {
      dtype = std::get<c10::ScalarType>(dtypeAttr->value);
    } else if (
        dtypeAttr && std::holds_alternative<std::string>(dtypeAttr->value)) {
      auto dtypeStr = std::get<std::string>(dtypeAttr->value);
      static const std::unordered_map<std::string, c10::ScalarType>
          kNameToType = {
              {"Float", c10::ScalarType::Float},
              {"Double", c10::ScalarType::Double},
              {"Half", c10::ScalarType::Half},
              {"BFloat16", c10::ScalarType::BFloat16},
              {"Long", c10::ScalarType::Long},
              {"Int", c10::ScalarType::Int},
              {"Short", c10::ScalarType::Short},
              {"Byte", c10::ScalarType::Byte},
              {"Bool", c10::ScalarType::Bool}};
      auto it = kNameToType.find(dtypeStr);
      if (it != kNameToType.end()) {
        dtype = it->second;
      }
    } else if (
        inputId < static_cast<int>(types.types.size()) &&
        types.types[inputId]) {
      auto inputDtype = types.types[inputId]->dtype();
      dtype = c10::isIntegralType(inputDtype, true) ? c10::ScalarType::Long
                                                    : inputDtype;
    }
    auto* exclusiveSum = graph->createNode(
        "torch.ops.aten.exclusive_sum.default", {{"input", cumsumInput}});
    exclusiveSum->addAttribute({"dim", static_cast<int64_t>(0)});
    exclusiveSum->addAttribute({"dtype", dtype});
    graph->insertBefore(exclusiveSum, const_cast<nativert::Node*>(node));
    auto* newOutput =
        waveGraph.newTensorValue(exclusiveSum, "exclusive_sum", dtype);
    return {{node->outputs()[0], newOutput}};
  }

  auto elements = node->inputs()[0].value->getListElements();
  bool hasNestedCat = false;
  for (auto* elem : elements) {
    auto* producer = elem->producer();
    // Only a cat on the same axis flattens: cat(cat(a, b, dim=1), c, dim=0) is
    // not cat(a, b, c) on either axis.
    if (producer && producer->target() == kCatTarget &&
        concatDimAttribute(producer) == dim) {
      hasNestedCat = true;
      break;
    }
  }
  if (!hasNestedCat) {
    return {};
  }

  std::vector<nativert::Value*> flatElements;
  for (auto* elem : elements) {
    flattenCatElements(elem, dim, flatElements);
  }
  TORCH_CHECK(!flatElements.empty(), "flattenCatElements produced no elements");

  auto firstId = flatElements.at(0)->id();
  auto dtype =
      (firstId < static_cast<int>(types.types.size()) && types.types[firstId])
      ? types.types[firstId]->dtype()
      : c10::ScalarType::Long;

  auto* listPack = graph->createListPack(
      std::move(flatElements), nativert::Type::Kind::Tensor);
  graph->insertBefore(listPack, const_cast<nativert::Node*>(node));

  auto* newCat = graph->createNode(
      std::string(kCatTarget), {{"tensors", listPack->outputs()[0]}});
  newCat->addAttribute({"dim", dim});
  graph->insertBefore(newCat, const_cast<nativert::Node*>(node));
  auto* newOutput = waveGraph.newTensorValue(newCat, "cat_result", dtype);

  return {{node->outputs()[0], newOutput}};
}

void concatSpecialForm(
    NodeCP node,
    const std::vector<ResultSpec>& /*resultSpecs*/,
    CompileCtx* ctx) {
  auto* listPackProducer = node->inputs()[0].value->producer();
  if (listPackProducer) {
    ctx->generatingOp()->allNodes().insert(listPackProducer);
  }
  auto elements = node->inputs()[0].value->getListElements();
  TORCH_CHECK(!elements.empty(), "cat requires at least one input tensor");
  auto* concatOutput = node->outputs()[0];

  auto& types = ctx->waveGraph().types();
  auto spec = concatSpec(node, types);

  // The destination type is the result's, not the first operand's: torch
  // promotes a mixed-dtype concat, and __concatCopy value-converts each element
  // into the buffer reserveConcatOutput allocated with that dtype.
  auto outputId = concatOutput->id();
  TORCH_CHECK(
      outputId >= 0 && static_cast<size_t>(outputId) < types.types.size() &&
          types.types[outputId],
      "No TensorMeta for cat output ",
      outputId);
  const auto typeName = cudaTypeString(types.types[outputId]->dtype());
  const auto elementSize =
      static_cast<int32_t>(c10::elementSize(types.types[outputId]->dtype()));

  // Operands whose extent this same kernel computes (a fused masked_select /
  // nonzero). Only a 1-d cat can absorb one: it patches the following views'
  // bases on device, which a strided layout cannot express.
  std::vector<bool> sizeSetInOp(elements.size(), false);
  for (size_t i = 0; i < elements.size(); ++i) {
    std::unordered_set<ValueCP> visited;
    sizeSetInOp.at(i) = ctx->isSizeSetInThisOp(elements.at(i), visited);
    TORCH_CHECK(
        !sizeSetInOp[i] || spec.outRank == 1,
        node->target(),
        ": operand ",
        i,
        " has its extent computed on device inside the concat's own kernel, "
        "which a rank-",
        static_cast<int>(spec.outRank),
        " result cannot lay out");
  }

  auto& op = *ctx->generatingOp();

  bool needsViewFixup = false;
  // int64_t: the accumulator sums the operands' join-axis extents, which for a
  // large 1-D cat can pass INT32_MAX before __concatCopy widens it.
  ctx->emitCode("  {\n  int64_t offset = 0;\n");
  int64_t lastAccumulated = -1;
  // Advances 'offset' past the operands in (lastAccumulated, upTo]. Extents are
  // read from the operand params rather than baked in: one KernelOperation
  // serves every invocation of a matching subgraph, whose shapes differ.
  auto accumulate = [&](int64_t upTo) {
    std::string incrExpr;
    for (auto j = lastAccumulated + 1; j <= upTo; ++j) {
      if (!incrExpr.empty()) {
        incrExpr += " + ";
      }
      incrExpr += ctx->param(elements[j], op) + "->dims[" +
          std::to_string(spec.dim) + "]";
    }
    if (!incrExpr.empty()) {
      ctx->emitCode("  offset += " + incrExpr + ";\n");
    }
    lastAccumulated = std::max(lastAccumulated, upTo);
  };

  for (size_t i = 0; i < elements.size(); ++i) {
    auto* elem = elements[i];
    auto* producer = elem->producer();
    auto* producerMeta =
        producer ? Registry::metadata(producer->target()) : nullptr;
    bool isSubgraphInput = !producer || ctx->generatingOp()->isInput(elem);
    bool producerIsView = producerMeta && producerMeta->isView();
    bool isCopyInput = isSubgraphInput || producerIsView;
    if (isCopyInput) {
      // A view element (e.g. slice(cumsum(...)) in an exclusive-prefix
      // cat([zeros, cumsum[:-1]])) is metadata-only, but its producer chain
      // holds interior fused compute that must still run -- otherwise the copy
      // below reads the buffer the view aliases before anything wrote it.
      // fusedCode recurses through the view into that producer (the cumsum) and
      // is idempotent on already-placed nodes.
      if (producerIsView && !isSubgraphInput && !ctx->isPlaced(producer)) {
        auto viewSpecs = outputSpecs(producer);
        ctx->fusedCode(producer, viewSpecs);
      }
      std::string offsetExpr;
      if (spec.isStack) {
        // Every stack operand occupies exactly one position along the new dim.
        offsetExpr = std::to_string(i);
      } else {
        accumulate(static_cast<int64_t>(i) - 1);
        offsetExpr = "offset";
      }
      ctx->emitCode(
          fmt::format(
              "  __concatCopy<{}, {}, {}>({}, {}, {}, {}, blockInfo);\n",
              ctx->cudaType(elem),
              typeName,
              spec.isStack ? "true" : "false",
              ctx->param(elem, op),
              ctx->param(concatOutput, op),
              spec.dim,
              offsetExpr));
    } else {
      std::vector<ResultSpec> prodSpecs;
      for (auto* output : producer->outputs()) {
        prodSpecs.push_back({output, {}});
      }
      ctx->fusedCode(producer, prodSpecs);
    }

    if (sizeSetInOp.at(i)) {
      needsViewFixup = true;
    }

    if (needsViewFixup && i + 1 < elements.size()) {
      accumulate(static_cast<int64_t>(i));
      auto* nextElem = elements[i + 1];
      auto* nextProducer = nextElem->producer();
      auto* nextMeta =
          nextProducer ? Registry::metadata(nextProducer->target()) : nullptr;
      bool nextIsCopy = !nextProducer ||
          ctx->generatingOp()->isInput(nextElem) ||
          (nextMeta && nextMeta->isView());
      if (!nextIsCopy) {
        ctx->callView(concatOutput, nextElem, "offset", elementSize);
      }
      if (WaveConfig::get().kernelDebugOutput) {
        ctx->emitCode(
            "  TRACE0(printf(\"cat view[" + std::to_string(i + 1) +
            "] offset=%ld\\n\", (long)offset));\n");
      }
    }
  }

  // A cat element's copy (__concatCopy) partitions blocks by SOURCE index, but
  // an in-kernel consumer (e.g. a fused elementwise add reading the cat output)
  // partitions by DESTINATION index. When a copy shifts data by a nonzero
  // offset (e.g. the exclusive-prefix cat([zeros[1], cumsum[:-1]]) writing
  // offsets[1+i] = cumsum[i]), the element that lands on one block's
  // destination range was written by a different block, so the consumer's
  // cross-block read is ordered only by intra-block __syncthreads() -- stale
  // across non-co-resident blocks in the multi-block (non-cooperative) path.
  // Fence the cat's writes with a grid-wide opBarrier before any consumer
  // reads. Only needed in multi-block non-CG mode (single-block is intra-block
  // safe; CG blocks are co-resident and already fenced), matching
  // isKernelBreak's grid-mode gating. Gated by the runtime
  // WaveConfig::scanOutputReturnBarrier toggle.
  if (WaveConfig::get().scanOutputReturnBarrier && !ctx->isSingleBlock() &&
      !ctx->isCgGrid()) {
    ctx->emitBarrier();
  }
  if (needsViewFixup) {
    accumulate(static_cast<int64_t>(elements.size()) - 1);
    if (WaveConfig::get().kernelDebugOutput) {
      ctx->emitCode(
          "  TRACE0(printf(\"cat final offset=%ld\\n\", (long)offset));\n");
    }
    auto concatParam = ctx->param(concatOutput, op);
    ctx->emitCode(
        "  if (threadIdx.x == 0) { " + concatParam +
        "->dims[0] = offset; }\n"
        "  __syncthreads();\n");
  }
  ctx->emitCode("  }\n");
  ctx->markPlaced(node);
}

} // namespace

std::vector<Dim> concatResultShape(
    const ConcatSpec& spec,
    const std::vector<std::vector<Dim>>& operandShapes) {
  // The result takes its non-joined extents from the operands (which agree on
  // them) and its joined extent from their sum, or from their count for a
  // stack, where each operand occupies one position along a new dimension.
  const auto elementRank = spec.elementRank();
  std::vector<Dim> outShape(spec.outRank, 0);
  for (const auto& shape : operandShapes) {
    for (int8_t d = 0; d < elementRank; ++d) {
      auto outDim = spec.isStack && d >= spec.dim ? d + 1 : d;
      if (!spec.isStack && d == spec.dim) {
        outShape[outDim] += shape[d];
      } else {
        outShape[outDim] = std::max(outShape[outDim], shape[d]);
      }
    }
  }
  if (spec.isStack) {
    outShape[spec.dim] = static_cast<Dim>(operandShapes.size());
  }
  return outShape;
}

at::Tensor concatOperandView(
    const at::Tensor& result,
    const ConcatSpec& spec,
    int64_t start,
    int64_t extent) {
  const auto baseSizes = result.sizes();
  const auto baseStrides = result.strides();
  c10::SmallVector<int64_t, kMaxDims> viewSizes;
  c10::SmallVector<int64_t, kMaxDims> viewStrides;
  for (int32_t d = 0; d < static_cast<int32_t>(baseSizes.size()); ++d) {
    if (spec.isStack && d == spec.dim) {
      continue;
    }
    viewSizes.push_back(d == spec.dim ? extent : baseSizes[d]);
    viewStrides.push_back(baseStrides[d]);
  }
  return aliasTensor(
      result,
      viewSizes,
      viewStrides,
      result.storage_offset() + start * baseStrides[spec.dim]);
}

std::pair<ConcatSpec, c10::ScalarType> ConcatLayout::resolve(
    const NodeMap& nodeMap) const {
  auto actual = nodeMap.find(originalNode);
  if (actual == nodeMap.end() || actual->second == originalNode ||
      types == nullptr) {
    return {spec, dtype};
  }
  auto actualId = actual->second->outputs()[0]->id();
  return {
      concatSpec(actual->second, *types), types->types.at(actualId)->dtype()};
}

bool concatNeedsHostShapes(NodeCP node, const ValueTypes& types) {
  if (node->target() != kCatTarget && node->target() != kStackTarget) {
    return false;
  }
  if (concatIsStandalone(node, types)) {
    return false;
  }
  if (concatSpec(node, types).outRank > 1) {
    return true;
  }
  // More than two operands is the allocation group's path, which lays the
  // result out on the host and hands every operand the region it fills. There
  // is no serial fallback that walks the operands incrementing an offset, so an
  // operand whose extent is only settled inside the concat's own kernel has to
  // end that kernel first and be read back as a host-side shape.
  auto* listPack = node->inputs()[0].value->producer();
  return listPack != nullptr && listPack->inputs().size() > 2;
}

bool concatFillsInParallel(NodeCP node, const ValueTypes& types) {
  if (!WaveConfig::get().parallelConcatFill) {
    return false;
  }
  if (node->target() != kCatTarget && node->target() != kStackTarget) {
    return false;
  }
  if (concatIsStandalone(node, types)) {
    return false;
  }
  // Two operands are not worth a step of their own: the pushdown costs a
  // kernel boundary, which only pays once there are enough operands for the
  // serial chain of copies to be the problem. Matches the threshold the concat
  // allocation group uses.
  auto* listPack = node->inputs()[0].value->producer();
  return listPack && listPack->inputs().size() > 2;
}

void registerConcatMetadata() {
  // 'dim' is a template attribute rather than a plain constant: the generated
  // code bakes the axis in, so two structurally identical concats that differ
  // only in 'dim' must not deduplicate onto one KernelOperation.
  MetadataBuilder(kCatTarget)
      .sizeShortcut(SizeShortcut::kSum)
      .sizeOrdinal({0})
      .sizeArgsList({true})
      .templateAttrs({"dim"})
      .isStandaloneFunc(concatIsStandalone)
      // __concatCopy decomposes the index per element and visits each element
      // once, so a densifying clone of a concat operand only moves that
      // addressing into the copy.
      .layoutAgnostic()
      .outputConstraints(concatOutputConstraints)
      .maybeReplace(concatMaybeReplace)
      .setOutputs(concatSetOutputs)
      .specialForm(concatSpecialForm)
      .registerOp();

  MetadataBuilder(kStackTarget)
      .sizeShortcut(SizeShortcut::kSum)
      .sizeOrdinal({0})
      .sizeArgsList({true})
      .templateAttrs({"dim"})
      .isStandaloneFunc(concatIsStandalone)
      .layoutAgnostic()
      .outputConstraints(concatOutputConstraints)
      .maybeReplace(concatMaybeReplace)
      .setOutputs(concatSetOutputs)
      .specialForm(concatSpecialForm)
      .registerOp();
}

} // namespace torch::wave
