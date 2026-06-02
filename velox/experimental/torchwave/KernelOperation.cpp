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
  if (values.empty() && args.empty()) {
    return 1;
  }
  int64_t result = 0;
  for (auto valueId : values) {
    auto& ivalue = frame->getIValue(valueId);
    int64_t n = ivalue.isTensor() ? ivalue.toTensor().numel() : ivalue.toInt();
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
  if (values.empty() && args.empty()) {
    return {1};
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
  forEachSortedAttribute(
      node, [&](NodeCP, const nativert::Attribute&) { ++numAttrs; });
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
    cost += meta->cost;
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

  // Compute unit cost: 10 per input/output tensor + sum of node costs.
  // Tensor lists count as one per element.
  int32_t numTensors = 0;
  for (int32_t i = 0; i < numInputs_; ++i) {
    if (orderedInputs_[i]->type().kind() == nativert::Type::Kind::TensorList) {
      numTensors +=
          static_cast<int32_t>(orderedInputs_[i]->getListElements().size());
    } else {
      ++numTensors;
    }
  }
  numTensors += static_cast<int32_t>(outputValues.size());
  std::unordered_set<NodeCP> costVisited;
  unitCost_ = 10.0f * static_cast<float>(numTensors);
  unitCost_ += sumNodeCosts(sg.root, inputs_, costVisited);

  // Check if any node in the subgraph has alwaysSingleBlock set.
  std::unordered_set<NodeCP> asbVisited;
  alwaysSingleBlock_ = hasAlwaysSingleBlock(sg.root, inputs_, asbVisited);

  for (size_t oi = 0; oi < outputValues.size(); ++oi) {
    auto* value = outputValues[oi];
    if (paramOffsets_.find(value) != paramOffsets_.end()) {
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
      node->toString());
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
    return SizeExpr{SizeShortcut::kMax, std::move(leafIds), {}};
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
      return SizeExpr{SizeShortcut::kMax, std::move(leafIds), {}};
    }
  }
  std::vector<nativert::ValueId> leafIds;
  for (int32_t i = 0; i < numInputs_; ++i) {
    auto* value = orderedInputs_[i];
    if (value->type().kind() == nativert::Type::Kind::TensorList) {
      for (auto* elem : value->getListElements()) {
        leafIds.push_back(elem->id());
      }
    } else {
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
        desc.sizeExpr = makeSizeExpr(node, subgraphInputs, i);
      } else if (meta->returnMeta[i].reserveShape) {
        desc = makeOutputDesc(meta->returnMeta[i], node, subgraphInputs);
      } else {
        desc.shapeSetOnDevice = meta->returnMeta[i].shapeSetOnDevice;
        desc.neededOnHost = meta->returnMeta[i].neededOnHost;
        desc.sizeExpr = makeSizeExpr(node, subgraphInputs, i);
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
