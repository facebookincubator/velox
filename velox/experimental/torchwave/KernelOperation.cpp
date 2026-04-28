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
#include "velox/experimental/torchwave/Utils.h"

#include <algorithm>
#include <sstream>

namespace torch::wave {

// --- SizeExpr ---

int64_t SizeExpr::numElements(FrameP frame, nativert::ValueId* largestOut)
    const {
  int64_t result = 0;
  // Combine direct values.
  for (auto valueId : values) {
    auto& ivalue = frame->getIValue(valueId);
    int64_t n = ivalue.isTensor() ? ivalue.toTensor().numel() : ivalue.toInt();
    if (op == SizeShortcut::kMax) {
      if (n > result) {
        result = n;
        if (largestOut) {
          *largestOut = valueId;
        }
      }
    } else {
      result += n;
    }
  }
  // Combine recursive sub-expressions.
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

void collectDeduppedInputs(
    NodeCP node,
    const std::unordered_set<ValueCP>& subgraphInputs,
    std::unordered_set<ValueCP>& seen,
    std::vector<ValueCP>& result) {
  for (auto& input : node->inputs()) {
    auto* value = input.value;
    if (subgraphInputs.count(value) || !value->producer()) {
      if (seen.insert(value).second) {
        result.push_back(value);
      }
    } else {
      collectDeduppedInputs(value->producer(), subgraphInputs, seen, result);
    }
  }
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

void kernelExprImpl(
    std::stringstream& ss,
    NodeCP node,
    const std::unordered_set<ValueCP>& inputSet,
    const std::unordered_set<ValueCP>& outputSet) {
  if (node->inputs().empty() && node->attributes().empty()) {
    ss << node->target();
    return;
  }
  ss << node->target() << "(";
  bool first = true;
  for (const auto& input : node->inputs()) {
    if (!first) {
      ss << ", ";
    }
    first = false;
    auto* value = input.value;
    auto* producer = value->producer();
    if (producer == nullptr || inputSet.count(value) ||
        outputSet.count(value)) {
      ss << "%v" << value->id();
    } else {
      kernelExprImpl(ss, producer, inputSet, outputSet);
    }
  }
  for (const auto& attr : node->attributes()) {
    if (!first) {
      ss << ", ";
    }
    first = false;
    ss << attr.name << "=" << constantToString(attr.value);
  }
  ss << ")";
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

void subgraphToStringImpl(
    std::stringstream& ss,
    NodeCP node,
    const std::unordered_set<ValueCP>& inputSet) {
  ss << node->target();
  const auto& nodeInputs = node->inputs();
  if (nodeInputs.empty()) {
    return;
  }
  ss << "(";
  for (size_t i = 0; i < nodeInputs.size(); ++i) {
    if (i > 0) {
      ss << ", ";
    }
    auto* value = nodeInputs[i].value;
    auto* producer = value->producer();
    if (!producer || inputSet.count(value)) {
      ss << "%" << value->id();
    } else {
      subgraphToStringImpl(ss, producer, inputSet);
    }
  }
  ss << ")";
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

std::string Subgraph::toString(Listing mode) const {
  std::unordered_set<ValueCP> inputSet(inputs.begin(), inputs.end());
  std::stringstream ss;
  subgraphToStringImpl(ss, root, inputSet);
  return ss.str();
}

// --- KernelOperation ---

KernelOperation::KernelOperation(
    const Subgraph& sg,
    int32_t opCode,
    CompileCtx& compileCtx)
    : opCode_{opCode},
      expr_{sg.root},
      compileCtx_{compileCtx},
      inputs_{sg.inputs.begin(), sg.inputs.end()},
      orderedInputs_{sg.inputs},
      numInputs_(sg.inputs.size()) {
  // Assign offsets to inputs.
  int32_t offset = 0;
  for (const auto* value : orderedInputs_) {
    paramOffsets_[value] = offset;
    if (value->type().kind() == nativert::Type::Kind::Tensor) {
      offset += sizeof(Tensor);
    } else {
      offset += 8;
    }
  }

  if (!sg.root) {
    constantAreaOffset_ = offset;
    return;
  }

  // Collect and assign offsets to outputs.
  std::vector<ValueCP> outputValues;
  setOutputs(sg.root, inputs_, outputValues, outputDescs_, true);

  // Compute unit cost before adding outputs to inputs_: 10 per input/output +
  // sum of node costs from registry.
  std::unordered_set<NodeCP> costVisited;
  unitCost_ = 10.0f * (numInputs_ + outputValues.size());
  unitCost_ += sumNodeCosts(sg.root, inputs_, costVisited);

  // Check if any node in the subgraph has alwaysSingleBlock set.
  std::unordered_set<NodeCP> asbVisited;
  alwaysSingleBlock_ = hasAlwaysSingleBlock(sg.root, inputs_, asbVisited);

  for (auto* value : outputValues) {
    orderedInputs_.push_back(value);
    paramOffsets_[value] = offset;
    if (value->type().kind() == nativert::Type::Kind::Tensor) {
      offset += sizeof(Tensor);
    } else {
      offset += 8;
    }
  }

  constantAreaOffset_ = offset;

  // Assign offsets to attributes.
  std::unordered_set<NodeCP> visited;
  collectAttrOffsets(sg.root, inputs_, visited, offset, attrOffsets_);
  numConstants_ = static_cast<int32_t>(attrOffsets_.size());
  altParamStart_ = offset;
  altParamOffset_ = offset;

  // For all-elementwise kernel ops, mark the output as byLargestInput.
  std::unordered_set<NodeCP> ewVisited;
  if (!outputValues.empty() && isAllElementwise(sg.root, inputs_, ewVisited)) {
    TORCH_CHECK(
        outputDescs_.size() == 1,
        "All-elementwise op should have exactly one output");
    outputDescs_[0].byLargestInput = true;
  }

  sizeExpr_ = makeSizeExpr(sg.root, inputs_);
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

void KernelOperation::setCode(std::stringstream& code) {
  text_ = code.str();
  code.str("");
  code.clear();
}

SizeExpr KernelOperation::makeSizeExpr(
    NodeCP node,
    const std::unordered_set<ValueCP>& subgraphInputs) {
  // Check if the entire subtree rooted here is elementwise.
  std::unordered_set<NodeCP> ewVisited;
  if (isAllElementwise(node, subgraphInputs, ewVisited)) {
    // Collect all distinct leaf Values reachable from this node.
    std::unordered_set<ValueCP> seen;
    std::vector<ValueCP> leaves;
    collectDeduppedInputs(node, subgraphInputs, seen, leaves);
    std::vector<nativert::ValueId> leafIds;
    leafIds.reserve(leaves.size());
    for (auto* v : leaves) {
      leafIds.push_back(v->id());
    }
    return SizeExpr{SizeShortcut::kMax, std::move(leafIds), {}};
  }

  auto* meta = Registry::metadata(node->target());
  TORCH_CHECK(meta, "No metadata for: ", node->target());

  SizeExpr result;
  result.op = meta->sizeShortcut;

  // Recurse on the inputs indicated by meta->sizeArgs.
  const auto& sizeArgs = meta->sizeArgs;
  for (size_t i = 0; i < sizeArgs.ordinal.size(); ++i) {
    auto ordinal = sizeArgs.ordinal[i];
    auto* value = node->inputs()[ordinal].value;

    // If this input is a subgraph leaf or has no producer, it's a terminal
    // Value.
    if (subgraphInputs.count(value) || !value->producer()) {
      bool isList = i < sizeArgs.isList.size() && sizeArgs.isList[i];
      if (isList) {
        for (auto* elem : value->getListElements()) {
          result.values.push_back(elem->id());
        }
      } else {
        result.values.push_back(value->id());
      }
    } else {
      // Recurse into the producer subtree.
      auto child = makeSizeExpr(value->producer(), subgraphInputs);
      // Flatten if child has the same op.
      if (child.op == result.op) {
        result.values.insert(
            result.values.end(), child.values.begin(), child.values.end());
        result.args.insert(
            result.args.end(),
            std::make_move_iterator(child.args.begin()),
            std::make_move_iterator(child.args.end()));
      } else {
        result.args.push_back(std::move(child));
      }
    }
  }

  return result;
}

OutputDesc KernelOperation::makeOutputDesc(
    const ArgumentMeta& returnMeta,
    NodeCP node,
    const std::unordered_set<ValueCP>& subgraphInputs) {
  OutputDesc desc;
  desc.shapeSetOnDevice = returnMeta.shapeSetOnDevice;
  desc.neededOnHost = returnMeta.neededOnHost;
  desc.sizeExpr.op = returnMeta.sizeShortcut;

  // Expand sizeArgs ordinals to ValueIds from the node's inputs.
  if (returnMeta.sizeShortcut != SizeShortcut::kNone) {
    const auto& sizeArgs = returnMeta.sizeArgs;
    for (size_t j = 0; j < sizeArgs.ordinal.size(); ++j) {
      auto ordinal = sizeArgs.ordinal[j];
      auto* value = node->inputs()[ordinal].value;
      bool isList = j < sizeArgs.isList.size() && sizeArgs.isList[j];
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
    desc.reserveShape = [node, reserveShape](
                            NodeCP /*unused*/,
                            nativert::ExecutionFrame& frame,
                            FormalToActual map) {
      return reserveShape(node, frame, map);
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
  node = compileCtx_.executableNode(node);
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
    return;
  }

  // Add outputs that need memory allocation: either because the return
  // metadata says so, or because the caller requires memory-backed results.
  // Also add shapeOnly outputs when an elementwise producer feeds a register
  // argument of a non-elementwise consumer.
  const auto& outputs = compileCtx_.outputs(node);
  for (size_t i = 0; i < meta->returnMeta.size() && i < outputs.size(); ++i) {
    bool needsShapeOnly = !inMemory && isEw && !callerIsElementwise;
    if (!meta->returnMeta[i].isRegister || inMemory || needsShapeOnly) {
      outputValues.push_back(outputs[i]);
      if (needsShapeOnly && meta->returnMeta[i].isRegister) {
        std::unordered_set<ValueCP> seen;
        std::vector<ValueCP> deduppedInputs;
        collectDeduppedInputs(node, subgraphInputs, seen, deduppedInputs);
        OutputDesc desc;
        desc.shapeOnly = true;
        desc.sizeExpr.op = SizeShortcut::kMax;
        for (auto* v : deduppedInputs) {
          desc.sizeExpr.values.push_back(v->id());
        }
        outputDescs.push_back(std::move(desc));
      } else if (
          meta->returnMeta[i].reserveShape ||
          meta->returnMeta[i].sizeShortcut != SizeShortcut::kNone) {
        outputDescs.push_back(
            makeOutputDesc(meta->returnMeta[i], node, subgraphInputs));
      } else {
        // Register-typed output forced to memory (e.g. elementwise root).
        // Reserve shape from the largest input.
        std::unordered_set<ValueCP> seen;
        std::vector<ValueCP> deduppedInputs;
        collectDeduppedInputs(node, subgraphInputs, seen, deduppedInputs);
        OutputDesc desc;
        desc.shapeSetOnDevice = meta->returnMeta[i].shapeSetOnDevice;
        desc.neededOnHost = meta->returnMeta[i].neededOnHost;
        desc.sizeExpr.op = SizeShortcut::kMax;
        for (auto* v : deduppedInputs) {
          desc.sizeExpr.values.push_back(v->id());
        }
        outputDescs.push_back(std::move(desc));
      }
    }
  }
}

std::string KernelOperation::toString(Listing mode) const {
  std::stringstream ss;
  std::unordered_set<ValueCP> inputSet(
      orderedInputs_.begin(), orderedInputs_.begin() + numInputs_);
  std::unordered_set<ValueCP> outputSet(
      orderedInputs_.begin() + numInputs_, orderedInputs_.end());

  // Print output-producing nodes as separate assignment lines, dependencies
  // first.
  std::unordered_set<NodeCP> printed;
  std::function<void(NodeCP)> printNode = [&](NodeCP node) {
    if (!printed.insert(node).second) {
      return;
    }
    // Recurse into producers that themselves have outputs in outputSet.
    for (const auto& input : node->inputs()) {
      auto* producer = input.value->producer();
      if (producer && !inputSet.count(input.value) &&
          outputSet.count(input.value)) {
        printNode(producer);
      }
    }

    // Collect this node's outputs that are in the output set.
    std::vector<ValueCP> nodeOutputs;
    for (const auto* v : node->outputs()) {
      if (outputSet.count(v)) {
        nodeOutputs.push_back(v);
      }
    }
    if (!nodeOutputs.empty()) {
      ss << "(";
      for (size_t i = 0; i < nodeOutputs.size(); ++i) {
        if (i > 0) {
          ss << ", ";
        }
        ss << "%v" << nodeOutputs[i]->id();
      }
      ss << ") = ";
    }
    kernelExprImpl(ss, node, inputSet, outputSet);
    ss << "\n";
  };

  printNode(expr_);

  {
    std::stringstream special;
    for (size_t i = 0; i < outputDescs_.size(); ++i) {
      const auto& desc = outputDescs_[i];
      if (!desc.shapeOnly && !desc.shapeSetOnDevice && !desc.neededOnHost) {
        continue;
      }
      if (special.tellp() > 0) {
        special << " ";
      }
      special << "%v" << orderedInputs_[numInputs_ + i]->id() << "(";
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
  }

  if (mode == kCode) {
    ss << "opCode = " << opCode_ << ":\n";
    ss << text_ << "\n";
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
    bindings[formalSg.inputs[i]->id()] = actualSg.inputs[i]->id();
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
