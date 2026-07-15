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

#include "velox/experimental/torchwave/KernelOperation.h"
#include "velox/experimental/torchwave/Compile.h"
#include "velox/experimental/torchwave/NodePrinter.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveGraph.h"

#include <algorithm>
#include <sstream>

namespace torch::wave {

// --- SizeExpr ---

int64_t SizeExpr::numElements(FrameP frame, nativert::ValueId* largestOut)
    const {
  // A pure factory expression (e.g. zeros(size=[1000])) has no tensor leaves to
  // size from -- its extent lives entirely in constShapes.  Fall through to the
  // constShapes path below rather than returning the scalar default.
  if (values.empty() && args.empty() && constShapes.empty()) {
    return 1;
  }
  // Broadcasting elementwise: the element count is the product of the broadcast
  // shape across all operands, which can exceed the largest single operand
  // (e.g. [20,1] and [100] broadcast to [20,100]).  Also used when a fused
  // factory op contributes a constant shape (constShapes) the leaves lack.
  if (broadcast || !constShapes.empty()) {
    int64_t n = 1;
    for (auto dim : dims(frame)) {
      n *= dim;
    }
    return n;
  }
  int64_t result = 0;
  for (auto valueId : values) {
    auto& ivalue = frame->getIValue(valueId);
    // A size operand must be a tensor (sized by numel) or a scalar int.
    // Anything else (a list / float / None that slipped into the size
    // expression, e.g. from a list-producing op's subgraph) does not drive the
    // grid, so treat it as 0 rather than asserting in IValue::toInt().
    if (!ivalue.isTensor() && !ivalue.isInt()) {
      continue;
    }
    int64_t n;
    if (ivalue.isTensor()) {
      n = ivalue.toTensor().numel();
    } else if (ivalue.isInt()) {
      n = ivalue.toInt();
    } else {
      LOG(WARNING) << "SizeExpr: value " << valueId << " is "
                   << ivalue.tagKind() << ", expected Tensor or Int";
      n = 1;
    }
    if (op == SizeShortcut::kMax) {
      if (n > result) {
        result = n;
        if (largestOut && ivalue.isTensor()) {
          *largestOut = valueId;
        }
      }
    } else {
      result += n;
    }
  }
  for (auto& child : args) {
    int64_t n = child.numElements(frame, largestOut);
    if (op == SizeShortcut::kMax) {
      if (n > result) {
        result = n;
      }
    } else {
      result += n;
    }
  }
  return result;
}

std::vector<Dim> SizeExpr::dims(FrameP frame) const {
  TORCH_CHECK(
      op == SizeShortcut::kMax,
      "SizeExpr::dims only supported for kMax, got ",
      static_cast<int>(op));
  if (values.empty() && args.empty() && constShapes.empty()) {
    return {1};
  }
  // Broadcasting: combine all operand shapes right-aligned, taking the max
  // (non-1) extent per dimension. Operands are assumed broadcast-compatible.
  // constShapes folds in factory-op extents (zeros/ones/full) fused into the
  // subtree whose size comes from a `size` attribute, not a frame value.
  if (broadcast || !constShapes.empty()) {
    std::vector<Dim> result;
    auto combine = [&result](const std::vector<Dim>& shape) {
      // Right-align: grow result to the larger rank, then broadcast each dim
      // from the right.  Broadcast rule: a size-1 dim takes the other operand's
      // extent -- which may be 0 for an empty tensor (1 broadcasts to 0, so
      // [N,1] vs [N,0] -> [N,0]).  Using max here would wrongly turn an empty
      // result into a non-empty one and read past the empty operand.
      if (shape.size() > result.size()) {
        result.insert(result.begin(), shape.size() - result.size(), 1);
      }
      for (size_t i = 0; i < shape.size(); ++i) {
        auto& resultDim = result[result.size() - shape.size() + i];
        if (resultDim == 1) {
          resultDim = shape[i];
        } else if (shape[i] != 1) {
          // Both extents are non-1; in a valid broadcast they are equal.
          resultDim = std::max<Dim>(resultDim, shape[i]);
        }
      }
    };
    for (auto valueId : values) {
      auto& ivalue = frame->getIValue(valueId);
      if (ivalue.isTensor()) {
        auto sizes = ivalue.toTensor().sizes();
        combine(std::vector<Dim>(sizes.begin(), sizes.end()));
      }
    }
    for (auto& child : args) {
      combine(child.dims(frame));
    }
    for (const auto& shape : constShapes) {
      combine(shape);
    }
    if (result.empty()) {
      result.push_back(1);
    }
    return result;
  }
  int64_t bestNumel = -1;
  std::vector<Dim> bestDims;
  for (auto valueId : values) {
    auto& ivalue = frame->getIValue(valueId);
    if (ivalue.isTensor()) {
      auto numel = ivalue.toTensor().numel();
      if (numel > bestNumel) {
        bestNumel = numel;
        auto sizes = ivalue.toTensor().sizes();
        bestDims.assign(sizes.begin(), sizes.end());
      }
    }
  }
  for (auto& child : args) {
    auto childDims = child.dims(frame);
    int64_t numel = 1;
    for (auto dim : childDims) {
      numel *= dim;
    }
    if (numel > bestNumel) {
      bestNumel = numel;
      bestDims = std::move(childDims);
    }
  }
  return bestDims;
}

SizeExpr SizeExpr::toActual(
    const FormalToActual& bindings,
    const IdToValueMap& idToValue) const {
  SizeExpr result;
  result.op = op;
  result.broadcast = broadcast;
  result.constShapes = constShapes;
  result.values.reserve(values.size());
  for (auto valueId : values) {
    auto it = bindings.find(valueId);
    auto actualId = it != bindings.end() ? it->second : valueId;
    TORCH_CHECK(
        idToValue.count(actualId),
        "Actual value id ",
        actualId,
        " not found in graph");
    result.values.push_back(actualId);
  }
  result.args.reserve(args.size());
  for (auto& child : args) {
    result.args.push_back(child.toActual(bindings, idToValue));
  }
  return result;
}

// --- Subgraph helpers ---

namespace {

void makeConstantIndicesImpl(
    NodeCP node,
    const std::unordered_set<ValueCP>& inputs,
    std::unordered_set<NodeCP>& visited,
    int32_t& ordinal,
    int32_t& numAttrsSeen,
    std::unordered_map<int32_t, int32_t>& result) {
  if (!visited.insert(node).second) {
    return;
  }
  auto myOrdinal = ordinal++;
  for (const auto& input : node->inputs()) {
    if (inputs.count(input.value)) {
      continue;
    }
    auto* producer = input.value->producer();
    if (producer) {
      makeConstantIndicesImpl(
          producer, inputs, visited, ordinal, numAttrsSeen, result);
    }
  }
  int32_t numAttrs = 0;
  forEachSortedAttribute(node, [&](NodeCP, const nativert::Attribute& attr) {
    // Match listConstants: an attribute whose value is None does not occupy a
    // constant slot, so it must not be counted here either.  Otherwise a None
    // attribute (e.g. searchsorted's `side`/`sorter`, repeat_interleave's
    // `output_size`) on a producer node shifts every later node's constant
    // index past the end of the constants vector.
    if (!nativert::constantToIValue(attr.value).isNone()) {
      ++numAttrs;
    }
  });
  if (numAttrs > 0) {
    result[myOrdinal] = numAttrsSeen;
    numAttrsSeen += numAttrs;
  }
}

int32_t nodeOrdinalImpl(
    NodeCP node,
    NodeCP target,
    const std::unordered_set<ValueCP>& inputs,
    std::unordered_set<NodeCP>& visited,
    int32_t& ordinal) {
  if (!visited.insert(node).second) {
    return -1;
  }
  auto myOrdinal = ordinal++;
  if (node == target) {
    return myOrdinal;
  }
  for (const auto& input : node->inputs()) {
    if (inputs.count(input.value)) {
      continue;
    }
    auto* producer = input.value->producer();
    if (producer) {
      auto result = nodeOrdinalImpl(producer, target, inputs, visited, ordinal);
      if (result >= 0) {
        return result;
      }
    }
  }
  return -1;
}

bool isAllElementwise(
    NodeCP node,
    const std::unordered_set<ValueCP>& subgraphInputs,
    std::unordered_set<NodeCP>& visited) {
  if (!visited.insert(node).second) {
    return true;
  }
  auto* meta = Registry::metadata(node->target());
  if (!meta || !meta->elementwise) {
    return false;
  }
  for (auto& input : node->inputs()) {
    auto* value = input.value;
    if (subgraphInputs.count(value)) {
      continue;
    }
    auto* producer = value->producer();
    if (!producer) {
      continue;
    }
    if (!isAllElementwise(producer, subgraphInputs, visited)) {
      return false;
    }
  }
  return true;
}

void collectAttrOffsets(
    NodeCP node,
    const std::unordered_set<ValueCP>& inputs,
    std::unordered_set<NodeCP>& visited,
    int32_t& offset,
    std::unordered_map<
        std::pair<NodeCP, std::string>,
        int32_t,
        KernelOperation::NodeAttrHash>& attrOffsets) {
  forEachSortedAttribute(
      node, inputs, visited, [&](NodeCP n, const nativert::Attribute& attr) {
        attrOffsets[{n, attr.name}] = offset;
        offset += 8;
      });
}

bool hasAlwaysSingleBlock(
    NodeCP node,
    const std::unordered_set<ValueCP>& inputs,
    std::unordered_set<NodeCP>& visited) {
  if (!visited.insert(node).second) {
    return false;
  }
  auto* meta = Registry::metadata(node->target());
  if (meta && meta->alwaysSingleBlock) {
    return true;
  }
  for (const auto& input : node->inputs()) {
    if (inputs.count(input.value)) {
      continue;
    }
    auto* producer = input.value->producer();
    if (producer && hasAlwaysSingleBlock(producer, inputs, visited)) {
      return true;
    }
  }
  return false;
}

float sumNodeCosts(
    NodeCP node,
    const std::unordered_set<ValueCP>& inputs,
    std::unordered_set<NodeCP>& visited) {
  if (!visited.insert(node).second) {
    return 0;
  }
  float cost = 0;
  auto* meta = Registry::metadata(node->target());
  if (meta) {
    cost += meta->unitCost(node);
  }
  for (const auto& input : node->inputs()) {
    if (inputs.count(input.value)) {
      continue;
    }
    auto* producer = input.value->producer();
    if (producer) {
      cost += sumNodeCosts(producer, inputs, visited);
    }
  }
  return cost;
}

void makeBindingsImpl(
    NodeCP formalNode,
    NodeCP actualNode,
    const std::unordered_set<ValueCP>& formalInputs,
    const std::unordered_set<ValueCP>& outputValues,
    std::unordered_set<NodeCP>& visited,
    FormalToActual& bindings) {
  if (!visited.insert(formalNode).second) {
    return;
  }
  const auto& formalIns = formalNode->inputs();
  const auto& actualIns = actualNode->inputs();
  for (size_t i = 0; i < formalIns.size() && i < actualIns.size(); ++i) {
    auto* formalValue = formalIns[i].value;
    if (formalInputs.count(formalValue)) {
      continue;
    }
    auto* formalProducer = formalValue->producer();
    auto* actualProducer = actualIns[i].value->producer();
    if (formalProducer && actualProducer) {
      makeBindingsImpl(
          formalProducer,
          actualProducer,
          formalInputs,
          outputValues,
          visited,
          bindings);
    }
  }
  const auto& formalOutputs = formalNode->outputs();
  const auto& actualOutputs = actualNode->outputs();
  for (size_t i = 0; i < formalOutputs.size() && i < actualOutputs.size();
       ++i) {
    if (outputValues.count(formalOutputs[i])) {
      bindings[formalOutputs[i]->id()] = actualOutputs[i]->id();
    }
  }
}

} // namespace

// --- Subgraph ---

std::unordered_map<int32_t, int32_t> Subgraph::makeConstantIndices() const {
  std::unordered_set<ValueCP> inputSet(inputs.begin(), inputs.end());
  std::unordered_set<NodeCP> visited;
  int32_t ordinal = 0;
  int32_t numAttrsSeen = 0;
  std::unordered_map<int32_t, int32_t> result;
  makeConstantIndicesImpl(
      root, inputSet, visited, ordinal, numAttrsSeen, result);
  return result;
}

int32_t Subgraph::nodeOrdinal(NodeCP node) const {
  std::unordered_set<ValueCP> inputSet(inputs.begin(), inputs.end());
  std::unordered_set<NodeCP> visited;
  int32_t ordinal = 0;
  return nodeOrdinalImpl(root, node, inputSet, visited, ordinal);
}

std::string Subgraph::toString(Listing /*mode*/) const {
  std::unordered_set<ValueCP> inputSet(inputs.begin(), inputs.end());
  PrintOptions opts = NodePrinter::defaults();
  opts.inlineIntermediates = true;
  opts.showOutputIds = false;
  opts.boundaryValues = &inputSet;
  return NodePrinter(opts).print(root);
}

// --- KernelOperation ---

namespace {

void flattenScalarListInputs(
    std::unordered_set<ValueCP>& inputs,
    std::vector<ValueCP>& orderedInputs) {
  std::vector<ValueCP> flattened;
  for (auto* value : orderedInputs) {
    if (value->type().kind() == nativert::Type::Kind::SymIntList) {
      auto* producer = value->producer();
      if (producer && producer->target() == "prim.ListPack") {
        inputs.erase(value);
        for (const auto& input : producer->inputs()) {
          if (!inputs.count(input.value)) {
            inputs.insert(input.value);
            flattened.push_back(input.value);
          }
        }
        continue;
      }
    }
    flattened.push_back(value);
  }
  orderedInputs = std::move(flattened);
}

} // namespace

void KernelOperation::assignParamOffset(ValueCP value, int32_t& offset) {
  if (value->type().kind() == nativert::Type::Kind::TensorList) {
    auto elements = value->getListElements();
    for (auto* elem : elements) {
      if (paramOffsets_.find(elem) == paramOffsets_.end()) {
        paramOffsets_[elem] = offset;
        offset += static_cast<int32_t>(sizeof(Tensor));
      }
    }
    paramOffsets_[value] = offset;
    offset += static_cast<int32_t>(sizeof(TensorList) + 8 * elements.size());
  } else if (value->type().kind() == nativert::Type::Kind::Tensor) {
    paramOffsets_[value] = offset;
    offset += static_cast<int32_t>(sizeof(Tensor));
  } else {
    paramOffsets_[value] = offset;
    offset += 8;
  }
}

KernelOperation::KernelOperation(
    const Subgraph& sg,
    int32_t opCode,
    CompileCtx& compileCtx)
    : opCode_{opCode},
      expr_{sg.root},
      compileCtx_{compileCtx},
      waveGraph_{&compileCtx.waveGraph()},
      inputs_{sg.inputs.begin(), sg.inputs.end()},
      orderedInputs_{sg.inputs},
      inputTypes_{sg.inputTypes},
      numInputs_(static_cast<int32_t>(sg.inputs.size())) {
  flattenScalarListInputs(inputs_, orderedInputs_);
  numInputs_ = static_cast<int32_t>(orderedInputs_.size());

  int32_t offset = 0;
  for (const auto* value : orderedInputs_) {
    assignParamOffset(value, offset);
  }

  if (!sg.root) {
    constantAreaOffset_ = offset;
    return;
  }

  // Collect and assign offsets to outputs.
  std::vector<ValueCP> outputValues;
  setOutputs(sg.root, inputs_, outputValues, outputDescs_, true);

  // Compute unit cost: per-tensor I/O cost (scaled by element size) + sum of
  // node costs.
  auto tensorCost = [this](ValueCP value) -> float {
    auto& types = waveGraph_->types();
    auto id = value->id();
    if (id >= 0 && static_cast<size_t>(id) < types.types.size() &&
        types.types[id]) {
      auto dtype = types.types[id]->dtype();
      auto elemSize = c10::elementSize(dtype);
      if (elemSize >= 8) {
        return 18.0f;
      }
      if (elemSize >= 4) {
        return 10.0f;
      }
      if (elemSize >= 2) {
        return 6.0f;
      }
      return 3.0f;
    }
    return 10.0f;
  };
  float tensorCostSum = 0;
  for (int32_t i = 0; i < numInputs_; ++i) {
    if (orderedInputs_[i]->type().kind() == nativert::Type::Kind::TensorList) {
      for (auto* elem : orderedInputs_[i]->getListElements()) {
        tensorCostSum += tensorCost(elem);
      }
    } else {
      tensorCostSum += tensorCost(orderedInputs_[i]);
    }
  }
  for (auto* value : outputValues) {
    tensorCostSum += tensorCost(value);
  }
  std::unordered_set<NodeCP> costVisited;
  unitCost_ = tensorCostSum;
  unitCost_ += sumNodeCosts(sg.root, inputs_, costVisited);

  // Check if any node in the subgraph has alwaysSingleBlock set.
  std::unordered_set<NodeCP> asbVisited;
  alwaysSingleBlock_ = hasAlwaysSingleBlock(sg.root, inputs_, asbVisited);

  for (size_t oi = 0; oi < outputValues.size(); ++oi) {
    auto* value = outputValues[oi];
    if (paramOffsets_.find(value) != paramOffsets_.end()) {
      // The value already has a param slot because it is also a (leaf) input of
      // this kernel. If it is produced by a view node (e.g. a slice fed into a
      // prim.ListPack), the kernel emits no device code to compute it, so it
      // must be materialized on the host before the kernel reads it. Keep the
      // output desc, set its viewNode, and reuse the existing input offset (do
      // not re-assign) so allocateLaunchOutputs executes the view before
      // launch. Without this the value stays None in the frame.
      auto* producer = value->producer();
      auto* producerMeta =
          producer ? Registry::metadata(producer->target()) : nullptr;
      if (producerMeta && producerMeta->isView() &&
          !outputDescs_[oi].viewNode) {
        outputDescs_[oi].viewNode = compileCtx_.originalFromVariant(producer);
        orderedInputs_.push_back(value);
        continue;
      }
      outputDescs_.erase(outputDescs_.begin() + static_cast<ptrdiff_t>(oi));
      outputValues.erase(outputValues.begin() + static_cast<ptrdiff_t>(oi));
      --oi;
      continue;
    }
    orderedInputs_.push_back(value);
    assignParamOffset(value, offset);
  }

  constantAreaOffset_ = offset;

  // Assign offsets to attributes.
  std::unordered_set<NodeCP> visited;
  collectAttrOffsets(sg.root, inputs_, visited, offset, attrOffsets_);
  numConstants_ = static_cast<int32_t>(attrOffsets_.size());
  altParamOffset_ = offset;

  // For all-elementwise kernel ops, mark the output as byLargestInput,
  // unless any input is wholeTensor (its size should not affect the output).
  std::unordered_set<NodeCP> ewVisited;
  if (!outputValues.empty() && isAllElementwise(sg.root, inputs_, ewVisited)) {
    TORCH_CHECK(
        outputDescs_.size() == 1,
        "All-elementwise op should have exactly one output");
    auto* meta = Registry::metadata(sg.root->target());
    bool hasWholeTensor = false;
    if (meta) {
      for (size_t i = 0; i < meta->argumentMeta.size(); ++i) {
        if (meta->argumentMeta[i].wholeTensor) {
          hasWholeTensor = true;
          break;
        }
      }
    }
    if (!hasWholeTensor) {
      outputDescs_[0].byLargestInput = true;
    }
  }

  sizeExpr_ = makeDeepSizeExpr();
}

std::vector<int32_t> KernelOperation::tensorParamOffsets() const {
  std::vector<int32_t> offsets;
  for (const auto* value : orderedInputs_) {
    if (value->type().kind() == nativert::Type::Kind::Tensor) {
      offsets.push_back(paramOffsets_.at(value));
    }
  }
  return offsets;
}

int32_t KernelOperation::paramOffset(ValueCP value) const {
  auto it = paramOffsets_.find(value);
  TORCH_CHECK(it != paramOffsets_.end(), "Value not found in paramOffsets");
  return it->second;
}

int32_t KernelOperation::attrOffset(NodeCP node, std::string_view attr) const {
  auto it = attrOffsets_.find({node, std::string(attr)});
  TORCH_CHECK(
      it != attrOffsets_.end(),
      "Attribute '",
      attr,
      "' not found for node ",
      standaloneToString(node));
  return it->second;
}

int64_t KernelOperation::largestInput(
    nativert::ExecutionFrame& frame,
    const FormalToActual& map) const {
  int64_t largest = 0;
  for (int32_t i = 0; i < numInputs_; ++i) {
    auto* formalValue = orderedInputs_[i];
    auto it = map.find(formalValue->id());
    auto actualId = it != map.end() ? it->second : formalValue->id();
    const auto& ivalue = frame.getIValue(actualId);
    if (ivalue.isTensor()) {
      largest = std::max(largest, ivalue.toTensor().numel());
    } else {
      largest = std::max(largest, static_cast<int64_t>(1));
    }
  }
  return largest;
}

void KernelOperation::addSharedDeclaration(const std::string& decl) {
  if (std::find(sharedDeclarations_.begin(), sharedDeclarations_.end(), decl) ==
      sharedDeclarations_.end()) {
    sharedDeclarations_.push_back(decl);
  }
}

namespace {

bool hasShapeOnDeviceInChain(
    NodeCP node,
    const std::unordered_set<NodeCP>& allNodes,
    std::unordered_set<NodeCP>& visited) {
  if (!allNodes.count(node) || !visited.insert(node).second) {
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
    auto* producer = input.value->producer();
    if (producer && hasShapeOnDeviceInChain(producer, allNodes, visited)) {
      return true;
    }
  }
  return false;
}

} // namespace

void KernelOperation::setCode(std::stringstream& code) {
  text_ = code.str();
  code.str("");
  code.clear();

  orderingInputs_.clear();
  orderingOutputs_.clear();
  for (auto* node : allNodes_) {
    for (const auto& input : node->inputs()) {
      auto* value = input.value;
      if (value->type().kind() == nativert::Type::Kind::SymIntList) {
        auto* producer = value->producer();
        if (producer && producer->target() == "prim.ListPack") {
          for (const auto& listInput : producer->inputs()) {
            orderingInputs_.insert(listInput.value->id());
          }
          continue;
        }
      }
      orderingInputs_.insert(value->id());
    }
    for (auto* output : node->outputs()) {
      orderingOutputs_.insert(output->id());
    }
  }

  // A view that is run on the host before launch (a viewNode output desc, e.g.
  // a leaf-input slice) has computed shape/offset operands such as a slice end
  // = item(...). Those operands are produced by other launches but are not
  // inputs of any fused node here, so they are missing from orderingInputs_.
  // Add them, otherwise this launch can be scheduled into a step before the
  // producer of the operand and the host view reads a stale (None) bound.
  for (const auto& desc : outputDescs_) {
    if (!desc.viewNode) {
      continue;
    }
    auto* viewMeta = Registry::metadata(desc.viewNode->target());
    int32_t baseOrdinal = (viewMeta && viewMeta->viewOfArg.has_value())
        ? *viewMeta->viewOfArg
        : -1;
    const auto& viewInputs = desc.viewNode->inputs();
    for (size_t k = 0; k < viewInputs.size(); ++k) {
      if (static_cast<int32_t>(k) == baseOrdinal) {
        continue;
      }
      orderingInputs_.insert(viewInputs[k].value->id());
    }
  }

  // The loop above collects ordering inputs from this op's (possibly lowered)
  // variant nodes.  When an op is lowered to tw.* helpers, a boundary tensor
  // input can be read only via reserveShape / sizeExpr at gatherLaunches time
  // rather than as a tracked variant node input, so its id is missing here.
  // Add the op's actual boundary inputs (orderedInputs_) so placeKernelLaunch
  // orders this op after their producers -- otherwise a standalone producer can
  // be co-scheduled in the same step and its output read during gatherLaunches
  // before the standalone has run.
  for (int32_t i = 0; i < numInputs_; ++i) {
    orderingInputs_.insert(orderedInputs_[i]->id());
  }

  for (size_t i = 0; i < outputDescs_.size(); ++i) {
    if (outputDescs_[i].shapeSetOnDevice) {
      continue;
    }
    auto* value = orderedInputs_[numInputs_ + i];
    auto* producer = value->producer();
    if (producer) {
      std::unordered_set<NodeCP> visited;
      if (hasShapeOnDeviceInChain(producer, allNodes_, visited)) {
        outputDescs_[i].shapeSetOnDevice = true;
      }
    }
  }
}

namespace {

// Recurses through elementwise producers, collecting leaf inputs (kernel op
// inputs or non-elementwise producers) into 'leafIds'. Skips inputs whose
// argumentMeta has wholeTensor set.
void collectElementwiseLeaves(
    NodeCP node,
    const std::unordered_set<ValueCP>& subgraphInputs,
    std::unordered_set<ValueCP>& seen,
    std::vector<nativert::ValueId>& leafIds) {
  auto* meta = Registry::metadata(node->target());
  const auto& inputs = node->inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto* value = inputs[i].value;
    if (meta && i < meta->argumentMeta.size() &&
        meta->argumentMeta[i].wholeTensor) {
      continue;
    }
    if (!seen.insert(value).second) {
      continue;
    }
    auto* producer = value->producer();
    if (producer && producer->target() == "prim.ListPack") {
      bool isRegister = meta && i < meta->argumentMeta.size() &&
          meta->argumentMeta[i].isRegister;
      if (isRegister) {
        for (const auto& listInput : producer->inputs()) {
          auto* lv = listInput.value;
          if (!seen.insert(lv).second) {
            continue;
          }
          if (subgraphInputs.count(lv) || !lv->producer()) {
            leafIds.push_back(lv->id());
          } else {
            auto* lp = lv->producer();
            auto* lpMeta = Registry::metadata(lp->target());
            if (lpMeta && lpMeta->elementwise) {
              collectElementwiseLeaves(lp, subgraphInputs, seen, leafIds);
            } else {
              leafIds.push_back(lv->id());
            }
          }
        }
        continue;
      }
    }
    if (subgraphInputs.count(value) || !producer) {
      leafIds.push_back(value->id());
      continue;
    }
    auto* producerMeta = Registry::metadata(producer->target());
    if (producerMeta && producerMeta->elementwise) {
      collectElementwiseLeaves(producer, subgraphInputs, seen, leafIds);
    } else {
      leafIds.push_back(value->id());
    }
  }
}

// Walks the elementwise subtree rooted at 'node' and records the `size`
// attribute of any factory op (zeros/ones/full) in it.  Such ops have no
// tensor inputs, so collectElementwiseLeaves finds no leaf to size the fused
// subtree from; their extent lives in the `size` attribute instead and must be
// broadcast into the output shape.
void collectFactorySizes(
    NodeCP node,
    const std::unordered_set<ValueCP>& subgraphInputs,
    std::unordered_set<NodeCP>& visited,
    std::vector<std::vector<Dim>>& constShapes) {
  if (!visited.insert(node).second) {
    return;
  }
  if (const auto* sizeAttr = node->tryGetAttribute("size")) {
    if (std::holds_alternative<std::vector<int64_t>>(sizeAttr->value)) {
      const auto& sz = std::get<std::vector<int64_t>>(sizeAttr->value);
      constShapes.emplace_back(sz.begin(), sz.end());
    }
  }
  for (const auto& input : node->inputs()) {
    auto* value = input.value;
    if (subgraphInputs.count(value)) {
      continue;
    }
    auto* producer = value->producer();
    if (!producer) {
      continue;
    }
    auto* producerMeta = Registry::metadata(producer->target());
    if (producerMeta && producerMeta->elementwise) {
      collectFactorySizes(producer, subgraphInputs, visited, constShapes);
    }
  }
}

} // namespace

SizeExpr KernelOperation::makeSizeExpr(
    NodeCP node,
    const std::unordered_set<ValueCP>& subgraphInputs,
    int32_t outputIdx) {
  auto* meta = Registry::metadata(node->target());
  if (meta && meta->elementwise) {
    std::unordered_set<ValueCP> seen;
    std::vector<nativert::ValueId> leafIds;
    collectElementwiseLeaves(node, subgraphInputs, seen, leafIds);
    std::unordered_set<NodeCP> factoryVisited;
    std::vector<std::vector<Dim>> constShapes;
    collectFactorySizes(node, subgraphInputs, factoryVisited, constShapes);
    // With more than one operand the result is their broadcast: the size is the
    // product of the broadcast shape, which can exceed the largest single
    // operand (e.g. [20,1] + [100] -> [20,100]). With one operand the size is
    // just that operand, so the broadcast path is unnecessary.
    SizeExpr expr{SizeShortcut::kMax, std::move(leafIds), {}};
    expr.broadcast = expr.values.size() > 1;
    expr.constShapes = std::move(constShapes);
    return expr;
  }

  TORCH_CHECK(meta, "No metadata for: ", node->target());
  TORCH_CHECK(
      outputIdx >= 0 &&
          static_cast<size_t>(outputIdx) < meta->returnMeta.size(),
      "outputIdx ",
      outputIdx,
      " out of range for ",
      node->target());

  const auto& retMeta = meta->returnMeta[outputIdx];
  SizeExpr result;
  result.op = retMeta.sizeShortcut;

  for (size_t i = 0; i < retMeta.sizeArgs.ordinal.size(); ++i) {
    auto ordinal = retMeta.sizeArgs.ordinal[i];
    auto* value = node->inputs()[ordinal].value;
    bool isList =
        i < retMeta.sizeArgs.isList.size() && retMeta.sizeArgs.isList[i];

    auto* producer = value->producer();
    if (producer && producer->target() == "prim.ListPack") {
      for (const auto& listInput : producer->inputs()) {
        result.values.push_back(listInput.value->id());
      }
    } else if (isList) {
      for (auto* elem : value->getListElements()) {
        result.values.push_back(elem->id());
      }
    } else {
      result.values.push_back(value->id());
    }
  }

  return result;
}

SizeExpr KernelOperation::makeDeepSizeExpr() {
  if (expr_) {
    std::unordered_set<NodeCP> ewVisited;
    if (isAllElementwise(expr_, inputs_, ewVisited)) {
      std::unordered_set<ValueCP> seen;
      std::vector<nativert::ValueId> leafIds;
      collectElementwiseLeaves(expr_, inputs_, seen, leafIds);
      std::unordered_set<NodeCP> factoryVisited;
      std::vector<std::vector<Dim>> constShapes;
      collectFactorySizes(expr_, inputs_, factoryVisited, constShapes);
      // Multiple operands broadcast against each other; size by the broadcast
      // shape (see makeSizeExpr).
      SizeExpr expr{SizeShortcut::kMax, std::move(leafIds), {}};
      expr.broadcast = expr.values.size() > 1;
      expr.constShapes = std::move(constShapes);
      return expr;
    }
  }
  // kMax over the kernel's inputs. Skip inputs that are neither a tensor, a
  // tensor list, nor a scalar int -- SizeExpr::numElements can only size by
  // those, and a list / float / None input does not drive the grid (and would
  // assert in IValue::toInt(); this is what an orphaned list-producing op such
  // as group_length_guard_final, generated with a SymIntList-bearing subgraph,
  // would otherwise hit).
  std::vector<nativert::ValueId> leafIds;
  for (int32_t i = 0; i < numInputs_; ++i) {
    auto* value = orderedInputs_[i];
    auto kind = value->type().kind();
    if (kind == nativert::Type::Kind::TensorList) {
      for (auto* elem : value->getListElements()) {
        leafIds.push_back(elem->id());
      }
    } else if (
        kind == nativert::Type::Kind::Tensor ||
        kind == nativert::Type::Kind::SymInt) {
      leafIds.push_back(value->id());
    }
  }
  return SizeExpr{SizeShortcut::kMax, std::move(leafIds), {}};
}

void mergeOutputDesc(OutputDesc& dst, OutputDesc&& src) {
  if (src.reserveShape) {
    dst.reserveShape = std::move(src.reserveShape);
  }
  if (src.sizeExpr.op != SizeShortcut::kNone) {
    dst.sizeExpr = std::move(src.sizeExpr);
  }
  if (src.shapeSetOnDevice) {
    dst.shapeSetOnDevice = true;
  }
  if (src.neededOnHost) {
    dst.neededOnHost = true;
  }
  if (src.isList) {
    dst.isList = true;
  }
  if (src.nonRootOutput) {
    dst.nonRootOutput = true;
  }
  if (src.byLargestInput) {
    dst.byLargestInput = true;
  }
  if (!src.shapeOnly && dst.shapeOnly) {
    dst.shapeOnly = false;
  }
  if (src.viewNode) {
    dst.viewNode = src.viewNode;
  }
}

bool addOrUpdateOutput(
    std::vector<ValueCP>& outputValues,
    std::vector<OutputDesc>& outputDescs,
    ValueCP value,
    OutputDesc desc) {
  for (size_t i = 0; i < outputValues.size(); ++i) {
    if (outputValues[i] == value) {
      TORCH_CHECK(i < outputDescs.size());
      mergeOutputDesc(outputDescs.at(i), std::move(desc));
      return false;
    }
  }
  if (value->type().kind() == nativert::Type::Kind::TensorList) {
    for (auto* elem : value->getListElements()) {
      bool found = false;
      for (size_t i = 0; i < outputValues.size(); ++i) {
        if (outputValues[i] == elem) {
          found = true;
          break;
        }
      }
      if (!found) {
        OutputDesc elemDesc;
        elemDesc.delegated = true;
        elemDesc.shapeSetOnDevice = desc.shapeSetOnDevice;
        // Propagate nonRootOutput so a list result of a split root op (e.g.
        // tw.group_length_guard_head) keeps its unpacked elements out of the
        // freeable intermediates list as well, not just the list value.
        elemDesc.nonRootOutput = desc.nonRootOutput;
        outputValues.push_back(elem);
        outputDescs.push_back(std::move(elemDesc));
      }
    }
  }
  outputValues.push_back(value);
  outputDescs.push_back(std::move(desc));
  return true;
}

OutputDesc KernelOperation::makeOutputDesc(
    const ArgumentMeta& returnMeta,
    NodeCP node,
    const std::unordered_set<ValueCP>& /*subgraphInputs*/) {
  OutputDesc desc;
  desc.shapeSetOnDevice = returnMeta.shapeSetOnDevice;
  desc.neededOnHost = returnMeta.neededOnHost;
  desc.nonRootOutput = returnMeta.nonRootOutput;
  desc.sizeExpr.op = returnMeta.sizeShortcut;

  // Expand sizeArgs ordinals to ValueIds from the node's inputs.
  // Skip when reserveShape is set — it handles sizing independently.
  if (returnMeta.sizeShortcut != SizeShortcut::kNone &&
      !returnMeta.reserveShape) {
    const auto& sizeArgs = returnMeta.sizeArgs;
    for (size_t j = 0; j < sizeArgs.ordinal.size(); ++j) {
      auto ordinal = sizeArgs.ordinal[j];
      TORCH_CHECK(
          ordinal >= 0 && static_cast<size_t>(ordinal) < node->inputs().size());
      auto* value = node->inputs().at(ordinal).value;
      bool isList = j < sizeArgs.isList.size() && sizeArgs.isList.at(j);
      if (isList) {
        for (auto* elem : value->getListElements()) {
          desc.sizeExpr.values.push_back(elem->id());
        }
      } else {
        desc.sizeExpr.values.push_back(value->id());
      }
    }
  }

  if (returnMeta.reserveShape) {
    auto reserveShape = returnMeta.reserveShape;
    auto* originalNode = compileCtx_.originalFromVariant(node);
    desc.reserveShape = [node, originalNode, reserveShape](
                            nativert::ExecutionFrame& frame,
                            const FormalToActual& map,
                            const NodeMap& nodeMap) {
      return reserveShape(node, frame, map, originalNode, nodeMap);
    };
  }

  return desc;
}

bool KernelOperation::isMultiUseInput(nativert::ValueId id) const {
  // Read from the WaveGraph (persists to execution), not compileCtx_, which is
  // a stack temporary destroyed after compilation. LaunchData calls this while
  // gathering launches at execution time.
  return waveGraph_ != nullptr && waveGraph_->isMultiUseInput(id);
}

bool KernelOperation::isGraphOutput(nativert::ValueId id) const {
  return waveGraph_ != nullptr && waveGraph_->isGraphOutput(id);
}

void KernelOperation::setOutputs(
    NodeCP node,
    const std::unordered_set<ValueCP>& subgraphInputs,
    std::vector<ValueCP>& outputValues,
    std::vector<OutputDesc>& outputDescs,
    bool inMemory,
    bool callerIsElementwise) {
  auto* meta = Registry::metadata(node->target());
  bool isEw = meta && meta->elementwise;

  // Recurse into producers. Pass inMemory based on whether the corresponding
  // argument expects a memory-backed value.
  const auto& inputs = node->inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto* value = inputs[i].value;
    if (subgraphInputs.count(value)) {
      continue;
    }
    auto* producer = value->producer();
    if (producer) {
      bool inputInMemory = meta && i < meta->argumentMeta.size() &&
          !meta->argumentMeta[i].isRegister;
      setOutputs(
          producer,
          subgraphInputs,
          outputValues,
          outputDescs,
          inputInMemory,
          isEw);
    }
  }

  if (!meta) {
    if (node->target() == "prim.ListPack" && !node->outputs().empty() &&
        node->outputs()[0]->type().kind() == nativert::Type::Kind::TensorList) {
      for (const auto& input : node->inputs()) {
        auto* value = input.value;
        if (subgraphInputs.count(value)) {
          continue;
        }
        auto* producer = value->producer();
        if (producer) {
          setOutputs(
              producer,
              subgraphInputs,
              outputValues,
              outputDescs,
              inMemory,
              callerIsElementwise);
        }
      }
      auto* listValue = node->outputs()[0];
      bool consumerHasSpecialSetOutputs = false;
      for (auto* user : listValue->users()) {
        auto* userMeta = Registry::metadata(user->target());
        if (userMeta && userMeta->setOutputs) {
          consumerHasSpecialSetOutputs = true;
          break;
        }
      }
      if (!consumerHasSpecialSetOutputs) {
        OutputDesc desc;
        desc.isList = true;
        addOrUpdateOutput(
            outputValues, outputDescs, listValue, std::move(desc));
      }
    }
    return;
  }

  if (meta->setOutputs) {
    meta->setOutputs(
        this,
        node,
        subgraphInputs,
        outputValues,
        outputDescs,
        inMemory,
        callerIsElementwise);
    return;
  }

  // If this is a view node, add its output with viewNode set so the view
  // is computed by executing the nativert node at launch time.
  if (meta->isView()) {
    const auto& outputs = node->outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
      OutputDesc desc;
      desc.viewNode = compileCtx_.originalFromVariant(node);
      addOrUpdateOutput(outputValues, outputDescs, outputs[i], std::move(desc));
    }
    return;
  }

  // Add outputs that need memory allocation: either because the return
  // metadata says so, or because the caller requires memory-backed results.
  // Also add shapeOnly outputs when an elementwise producer feeds a register
  // argument of a non-elementwise consumer.
  const auto& outputs = node->outputs();
  for (size_t i = 0; i < meta->returnMeta.size() && i < outputs.size(); ++i) {
    bool isListOutput =
        outputs[i]->type().kind() == nativert::Type::Kind::TensorList;
    bool needsShapeOnly = !inMemory && isEw && !callerIsElementwise;
    if (!meta->returnMeta[i].isRegister || inMemory || needsShapeOnly) {
      OutputDesc desc;
      if (needsShapeOnly && meta->returnMeta[i].isRegister) {
        desc.shapeOnly = true;
        desc.sizeExpr =
            makeSizeExpr(node, subgraphInputs, static_cast<int32_t>(i));
      } else if (meta->returnMeta[i].reserveShape) {
        desc = makeOutputDesc(meta->returnMeta[i], node, subgraphInputs);
      } else {
        desc.shapeSetOnDevice = meta->returnMeta[i].shapeSetOnDevice;
        desc.neededOnHost = meta->returnMeta[i].neededOnHost;
        desc.sizeExpr =
            makeSizeExpr(node, subgraphInputs, static_cast<int32_t>(i));
        // In-place elementwise (Tensor(a!)): the materialized output aliases
        // the mutated self argument, so reserve it as a view of self instead of
        // a fresh buffer (see reserveOutputs). This makes the returned tensor
        // reflect self, including later in-place mutations of self.
        if (meta->elementwise && i == 0) {
          auto mutated = dataMutatedInputs(node);
          if (!mutated.empty()) {
            desc.aliasSelfId = mutated[0]->id();
          }
        }
      }
      // If the kernel's top node returns a naked scalar (e.g. sym_size / numel
      // used as the kernel output rather than fused in-register), it must be
      // read back into the execution frame so a consuming kernel can pick it up
      // as a host scalar param. Fused uses don't reach here: the producer is
      // then an interior node whose register output is never materialized.
      auto outKind = outputs[i]->type().kind();
      if (node == expr_ && outKind != nativert::Type::Kind::Tensor &&
          outKind != nativert::Type::Kind::TensorList) {
        desc.neededOnHost = true;
      }
      if (isListOutput) {
        desc.isList = true;
      }
      addOrUpdateOutput(outputValues, outputDescs, outputs[i], std::move(desc));
    }
  }
}

std::string KernelOperation::toString(const OpInvocation* invocation) const {
  std::unordered_set<ValueCP> inputSet(
      orderedInputs_.begin(), orderedInputs_.begin() + numInputs_);
  std::unordered_set<ValueCP> outputSet(
      orderedInputs_.begin() + numInputs_, orderedInputs_.end());

  PrintOptions opts = NodePrinter::defaults();
  opts.boundaryValues = &inputSet;
  opts.breakoutValues = &outputSet;
  opts.showOutputIds = true;
  if (invocation) {
    opts.formalToActual = &invocation->bindings();
  }
  if (waveGraph_) {
    opts.valueTypes = &waveGraph_->types();
    opts.graph = waveGraph_->graph();
  }
  NodePrinter printer(opts);

  std::stringstream ss;
  ss << printer.print(expr_);

  std::stringstream special;
  for (size_t i = 0; i < outputDescs_.size(); ++i) {
    const auto& desc = outputDescs_[i];
    if (!desc.shapeOnly && !desc.shapeSetOnDevice && !desc.neededOnHost) {
      continue;
    }
    if (special.tellp() > 0) {
      special << " ";
    }
    special << "%" << orderedInputs_[numInputs_ + i]->id() << "(";
    bool first = true;
    auto flag = [&](const char* name) {
      if (!first) {
        special << "|";
      }
      special << name;
      first = false;
    };
    if (desc.shapeOnly) {
      flag("shape");
    }
    if (desc.shapeSetOnDevice) {
      flag("shapeondev");
    }
    if (desc.neededOnHost) {
      flag("reqh");
    }
    special << ")";
  }
  if (special.tellp() > 0) {
    ss << "  outputs: " << special.str() << "\n";
  }

  return ss.str();
}

// --- makeBindings ---

FormalToActual makeBindings(
    const Subgraph& formalSg,
    const Subgraph& actualSg,
    const KernelOperation& op) {
  FormalToActual bindings;
  const auto& orderedInputs = op.orderedInputs();
  auto numInputs = op.numInputs();
  std::unordered_set<ValueCP> formalInputs(
      formalSg.inputs.begin(), formalSg.inputs.end());
  // Map formal input values to actual input values.
  for (size_t i = 0; i < formalSg.inputs.size() && i < actualSg.inputs.size();
       ++i) {
    TORCH_CHECK(i < formalSg.inputs.size() && i < actualSg.inputs.size());
    bindings[formalSg.inputs.at(i)->id()] = actualSg.inputs.at(i)->id();
  }
  std::unordered_set<ValueCP> outputValues(
      orderedInputs.begin() + numInputs, orderedInputs.end());
  if (!outputValues.empty()) {
    std::unordered_set<NodeCP> visited;
    makeBindingsImpl(
        formalSg.root,
        actualSg.root,
        formalInputs,
        outputValues,
        visited,
        bindings);
  }
  return bindings;
}

} // namespace torch::wave
