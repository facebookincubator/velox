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

#include "velox/experimental/torchwave/CompiledOp.h"
#include "velox/experimental/torchwave/Compile.h"
#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/GraphOptimizer.h"
#include "velox/experimental/torchwave/ParallelExpr.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"

#include <ATen/ATen.h>
#include <c10/util/StringUtil.h>
#include <folly/ScopeGuard.h>
#include <gflags/gflags.h>
#include <algorithm>
#include <fstream>
#include <sstream>

#include "velox/experimental/wave/common/GpuArena.h"

DECLARE_bool(print_timing);
DEFINE_bool(
    debug_single_ops,
    false,
    "Launch kernel once per block for debugging, waiting after each launch");

namespace torch::wave {

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
  const auto& attrs = node->attributes();
  if (!attrs.empty()) {
    result[myOrdinal] = numAttrsSeen;
    numAttrsSeen += attrs.size();
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

void fillTensorParam(const at::Tensor& tensor, void* dest) {
  TORCH_CHECK(
      tensor.dim() <= 3,
      "Tensors with more than 3 dims not supported, got ",
      tensor.dim());
  auto* t = reinterpret_cast<Tensor*>(dest);
  t->storage = tensor.data_ptr();
  t->rank = tensor.dim();
  for (int i = 0; i < 3; ++i) {
    t->dims[i] = i < tensor.dim() ? tensor.size(i) : 0;
    t->strides[i] =
        i < tensor.dim() ? (tensor.size(i) == 1 ? 0 : tensor.stride(i)) : 0;
  }
  t->numEl = tensor.numel();
  t->status = Tensor::kUninited;
}

/// Converts a c10::IValue default value to a nativert::Constant.
nativert::Constant iValueToConstant(const c10::IValue& iv) {
  if (iv.isNone()) {
    return nativert::None{};
  }
  if (iv.isBool()) {
    return iv.toBool();
  }
  if (iv.isInt()) {
    return iv.toInt();
  }
  if (iv.isDouble()) {
    return iv.toDouble();
  }
  if (iv.isString()) {
    return iv.toStringRef();
  }
  if (iv.isIntList()) {
    return iv.toIntVector();
  }
  if (iv.isDoubleList()) {
    return iv.toDoubleVector();
  }
  if (iv.isBoolList()) {
    auto list = iv.toBoolList();
    std::vector<bool> result;
    result.reserve(list.size());
    for (bool b : list) {
      result.push_back(b);
    }
    return result;
  }
  if (iv.isDevice()) {
    return iv.toDevice();
  }
  TORCH_CHECK(
      false, "iValueToConstant: unsupported IValue type: ", iv.tagKind());
}

/// Looks up the FunctionSchema for an aten node target like
/// "torch.ops.aten.add.Tensor". Returns nullptr for non-aten ops.
const c10::FunctionSchema* findSchema(std::string_view target) {
  auto atoms = c10::split(target, '.');
  if (atoms.size() < 3 || atoms[atoms.size() - 3] != "aten") {
    return nullptr;
  }
  return findFunctionSchema(target);
}

void fillScalarParam(const c10::IValue& ivalue, void* dest) {
  if (ivalue.isInt()) {
    *reinterpret_cast<int64_t*>(dest) = ivalue.toInt();
  } else if (ivalue.isDouble()) {
    *reinterpret_cast<double*>(dest) = ivalue.toDouble();
  } else if (ivalue.isBool()) {
    *reinterpret_cast<int64_t*>(dest) = ivalue.toBool() ? 1 : 0;
  } else {
    TORCH_CHECK(false, "Unsupported IValue type for kernel param");
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

at::Tensor paramTensor(
    ValueCP value,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map) {
  auto it = map.find(value->id());
  TORCH_CHECK(
      it != map.end(),
      "Input value %v",
      value->id(),
      " not found in FormalToActual map");
  return frame.getTensor(it->second);
}

int64_t paramSymInt(
    ValueCP value,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map) {
  auto it = map.find(value->id());
  TORCH_CHECK(
      it != map.end(),
      "Input value %v",
      value->id(),
      " not found in FormalToActual map");
  return frame.getSymInt(it->second);
}

std::vector<std::vector<Dim>> elementwiseInputShape(
    NodeCP node,
    nativert::ExecutionFrame& frame,
    FormalToActual map,
    int32_t ordinal) {
  std::vector<int64_t> bestShape;
  int64_t bestNumel = -1;

  std::function<void(ValueCP)> trace = [&](ValueCP v) {
    auto it = map.find(v->id());
    if (it != map.end()) {
      auto& ivalue = frame.getIValue(it->second);
      if (ivalue.isTensor()) {
        auto numel = ivalue.toTensor().numel();
        if (numel > bestNumel) {
          bestNumel = numel;
          auto sizes = ivalue.toTensor().sizes();
          bestShape.assign(sizes.begin(), sizes.end());
        }
        return;
      }
    }
    auto* producer = v->producer();
    if (!producer) {
      return;
    }
    auto* meta = Registry::metadata(producer->target());
    if (meta && meta->elementwise) {
      for (const auto& input : producer->inputs()) {
        trace(input.value);
      }
    }
  };

  trace(node->inputs()[ordinal].value);

  TORCH_CHECK(
      bestNumel >= 0, "elementwiseInputShape: no tensor found in frame");
  return {std::vector<Dim>(bestShape.begin(), bestShape.end())};
}

int32_t makeGrid(
    const std::vector<LaunchData>& launches,
    StepVectors& sv,
    int32_t maxBlocksPerSM) {
  const int32_t blockSize = 256;

  // Compute cost per launch: numElements * unitCost.
  sv.costs.resize(launches.size());
  float totalCost = 0;
  for (size_t i = 0; i < launches.size(); ++i) {
    sv.costs[i] = static_cast<float>(launches[i].numElements) *
        launches[i].launch->op->unitCost();
    totalCost += sv.costs[i];
  }

  // Max blocks each launch could use.
  sv.maxBlocks.resize(launches.size());
  const int32_t elementsPerBlock = blockSize * 4;
  for (size_t i = 0; i < launches.size(); ++i) {
    sv.maxBlocks[i] = static_cast<int32_t>(
        (launches[i].numElements + elementsPerBlock - 1) / elementsPerBlock);
    if (sv.maxBlocks[i] < 1) {
      sv.maxBlocks[i] = 1;
    }
  }

  // Target blocks from device SM count and kernel occupancy.
  int32_t numSMs = WaveConfig::get().numSms;
  if (numSMs == 0) {
    numSMs = 100;
    auto* device = facebook::velox::wave::currentDevice();
    if (device) {
      numSMs = device->numSM;
    }
  }
  int32_t blocksPerSM = maxBlocksPerSM > 0 ? maxBlocksPerSM : 4;
  int32_t targetBlocks = numSMs * blocksPerSM;

  // Assign blocks pro rata by cost, at least 1 per launch, capped by
  // maxBlocks.
  sv.numBlocksPerLaunch.resize(launches.size());
  int32_t totalAssigned = 0;
  for (size_t i = 0; i < launches.size(); ++i) {
    // Operations marked alwaysSingleBlock always get exactly one block.
    if (launches[i].launch->op && launches[i].launch->op->alwaysSingleBlock()) {
      sv.numBlocksPerLaunch[i] = 1;
      totalAssigned += 1;
      continue;
    }
    float fraction =
        (totalCost > 0) ? sv.costs[i] / totalCost : 1.0f / launches.size();
    int32_t assigned =
        std::max(1, static_cast<int32_t>(fraction * targetBlocks + 0.5f));
    assigned = std::min(assigned, sv.maxBlocks[i]);
    // Round down blocks so all but the last process a multiple of blockSize
    // elements.
    if (assigned > 1) {
      auto elemsPerBlock = (launches[i].numElements + assigned - 1) / assigned;
      auto alignedElems = roundUp(elemsPerBlock, elementsPerBlock);
      assigned = std::max(
          1,
          static_cast<int32_t>(
              (launches[i].numElements + alignedElems - 1) / alignedElems));
    }
    sv.numBlocksPerLaunch[i] = assigned;
    totalAssigned += assigned;
  }

  // Fill blocks and launchIndices.
  sv.blocks.resize(totalAssigned);
  sv.launchIndices.resize(totalAssigned);
  int32_t blockIdx = 0;
  for (size_t i = 0; i < launches.size(); ++i) {
    auto opCode = launches[i].launch->op->opCode();
    auto nBlocks = sv.numBlocksPerLaunch[i];
    for (int32_t b = 0; b < nBlocks; ++b) {
      auto& info = sv.blocks[blockIdx];
      info.op = opCode;
      info.blockInOp = b;
      info.numBlocksInOp = nBlocks;
      info.params = nullptr;
      sv.launchIndices[blockIdx] = static_cast<int32_t>(i);
      ++blockIdx;
    }
  }
  return blockSize;
}

// --- Launch ---

std::string Launch::toString(Listing mode) const {
  if (op) {
    return "kernel: " + op->toString(mode);
  }
  if (standalone) {
    Subgraph sg;
    sg.root = standalone;
    for (const auto& input : standalone->inputs()) {
      sg.inputs.push_back(input.value);
    }
    return "standalone " + sg.toString(mode);
  }
  return "";
}

// --- ProjectOperation ---

ProjectOperation::ProjectOperation(const Subgraph& sg, CompileCtx& ctx)
    : subgraph_(sg) {}

namespace {

void printLaunchGrid(
    std::stringstream& ss,
    const LaunchGrid& grid,
    const char* heading,
    Listing mode) {
  if (heading) {
    ss << heading << "\n";
  }
  for (size_t step = 0; step < grid.size(); ++step) {
    ss << "Step" << step << "\n";
    for (size_t lane = 0; lane < grid[step].size(); ++lane) {
      ss << "  Lane " << lane << ": ";
      const auto& launch = grid[step][lane];
      if (launch.op) {
        ss << launch.op->toString(mode);
        ss << "    Params: (";
        for (size_t i = 0; i < launch.values.size(); ++i) {
          if (i > 0) {
            ss << ", ";
          }
          ss << "%v" << launch.values[i]->id();
        }
        ss << ")\n";
        if (!launch.constantIndices.empty()) {
          ss << "    ConstantIndices: (";
          for (size_t i = 0; i < launch.constantIndices.size(); ++i) {
            if (i > 0) {
              ss << ", ";
            }
            ss << launch.constantIndices[i];
          }
          ss << ")\n";
        }
      } else if (launch.standalone) {
        ss << "Standalone: " << launch.standalone->toString() << "\n";
      }
    }
  }
}

} // namespace

std::string ProjectOperation::toString(Listing mode) const {
  std::stringstream ss;
  printLaunchGrid(ss, grid_, nullptr, mode);
  if (!singleBlockGrid_.empty()) {
    printLaunchGrid(ss, singleBlockGrid_, "Single Block Variant", mode);
  }
  return ss.str();
}

// --- KernelOperation ---

KernelOperation::KernelOperation(
    const Subgraph& sg,
    int32_t opCode,
    const CompileCtx& compileCtx)
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
    bool inMemory) {
  auto* meta = Registry::metadata(node->target());

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
          producer, subgraphInputs, outputValues, outputDescs, inputInMemory);
    }
  }

  if (!meta) {
    return;
  }

  // Add outputs that need memory allocation: either because the return
  // metadata says so, or because the caller requires memory-backed results.
  const auto& outputs = compileCtx_.outputs(node);
  for (size_t i = 0; i < meta->returnMeta.size() && i < outputs.size(); ++i) {
    if (!meta->returnMeta[i].isRegister || inMemory) {
      outputValues.push_back(outputs[i]);
      if (meta->returnMeta[i].reserveShape ||
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

  if (mode == kCode) {
    ss << "opCode = " << opCode_ << ":\n";
    ss << text_ << "\n";
  }

  return ss.str();
}

// --- OpInvocation ---

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

namespace {

/// Builds a node-to-node map by walking formal and actual subgraphs in
/// parallel. Recursion stops at subgraph inputs.
void makeNodeMap(
    const Subgraph& formalSg,
    const Subgraph& actualSg,
    std::unordered_map<NodeCP, NodeCP>& nodeMap) {
  std::unordered_set<ValueCP> formalInputSet(
      formalSg.inputs.begin(), formalSg.inputs.end());
  std::function<void(NodeCP, NodeCP)> walk = [&](NodeCP formalNode,
                                                 NodeCP actualNode) {
    if (!nodeMap.emplace(formalNode, actualNode).second) {
      return;
    }
    const auto& fi = formalNode->inputs();
    const auto& ai = actualNode->inputs();
    for (size_t i = 0; i < fi.size(); ++i) {
      if (formalInputSet.count(fi[i].value)) {
        continue;
      }
      auto* fp = fi[i].value->producer();
      auto* ap = ai[i].value->producer();
      if (fp && ap) {
        walk(fp, ap);
      }
    }
  };
  if (formalSg.root && actualSg.root) {
    walk(formalSg.root, actualSg.root);
  }
}

/// Builds bindings by walking formal and actual subgraphs in parallel.
/// Maps all inputs positionally and all node outputs by parallel tree walk.
FormalToActual makeSubgraphBindings(
    const Subgraph& formalSg,
    const Subgraph& actualSg) {
  TORCH_CHECK(
      formalSg.inputs.size() == actualSg.inputs.size(),
      "Input count mismatch: formal=",
      formalSg.inputs.size(),
      " actual=",
      actualSg.inputs.size());
  FormalToActual bindings;
  for (size_t i = 0; i < formalSg.inputs.size(); ++i) {
    bindings[formalSg.inputs[i]->id()] = actualSg.inputs[i]->id();
  }
  std::unordered_set<ValueCP> formalInputSet(
      formalSg.inputs.begin(), formalSg.inputs.end());
  std::unordered_set<NodeCP> visited;
  std::function<void(NodeCP, NodeCP)> walk = [&](NodeCP formalNode,
                                                 NodeCP actualNode) {
    if (!visited.insert(formalNode).second) {
      return;
    }
    const auto& fo = formalNode->outputs();
    const auto& ao = actualNode->outputs();
    TORCH_CHECK(
        fo.size() == ao.size(),
        "Output count mismatch at node ",
        formalNode->target());
    for (size_t i = 0; i < fo.size(); ++i) {
      bindings[fo[i]->id()] = ao[i]->id();
    }
    const auto& fi = formalNode->inputs();
    const auto& ai = actualNode->inputs();
    TORCH_CHECK(
        fi.size() == ai.size(),
        "Input count mismatch at node ",
        formalNode->target());
    for (size_t i = 0; i < fi.size(); ++i) {
      if (formalInputSet.count(fi[i].value)) {
        continue;
      }
      auto* fp = fi[i].value->producer();
      auto* ap = ai[i].value->producer();
      if (fp && ap) {
        walk(fp, ap);
      }
    }
  };
  if (formalSg.root && actualSg.root) {
    walk(formalSg.root, actualSg.root);
  }
  return bindings;
}

} // namespace

OpInvocation::OpInvocation(
    ProjectOperation* projectOp,
    const Subgraph& sg,
    std::deque<c10::IValue>& storage)
    : projectOp_{projectOp} {
  const auto& formalSg = projectOp->subgraph();
  bindings_ = makeSubgraphBindings(formalSg, sg);
  constants_ = listConstants(sg, storage);
  makeNodeMap(formalSg, sg, nodeMap_);
}

// --- LaunchData ---

LaunchData::LaunchData(
    const Launch& launch,
    OpInvocation& op,
    const IdToValueMap& idToValue)
    : launch(&launch), op(&op), numElements(0) {
  const auto& bindings = op.bindings();

  auto translateId = [&](ValueCP formal) -> nativert::ValueId {
    auto it = bindings.find(formal->id());
    auto actualId = it != bindings.end() ? it->second : formal->id();
    return actualId;
  };

  if (!launch.op) {
    // Standalone: translate node via nodeMap, inputs and outputs via bindings.
    auto nodeIt = op.nodeMap().find(launch.standalone);
    TORCH_CHECK(
        nodeIt != op.nodeMap().end(), "Standalone node not found in nodeMap");
    standalone = nodeIt->second;
    for (auto& input : launch.standalone->inputs()) {
      actualInputs.push_back(translateId(input.value));
    }
    for (auto* output : launch.standalone->outputs()) {
      actualOutputs.push_back(translateId(output));
    }
  } else {
    // Kernel op: translate sizeExpr, inputs, outputs, and output descs.
    auto* kernelOp = launch.op;
    sizeExpr = kernelOp->sizeExpr().toActual(bindings, idToValue);

    const auto& orderedInputs = kernelOp->orderedInputs();
    auto nInputs = kernelOp->numInputs();
    for (int32_t i = 0; i < nInputs; ++i) {
      actualInputs.push_back(translateId(orderedInputs[i]));
    }
    for (size_t i = nInputs; i < orderedInputs.size(); ++i) {
      actualOutputs.push_back(translateId(orderedInputs[i]));
    }

    const auto& outputDescs = kernelOp->outputDescs();
    for (size_t i = 0; i < outputDescs.size(); ++i) {
      const auto& desc = outputDescs[i];
      OutputDesc actualDesc = desc;
      actualDesc.sizeExpr = desc.sizeExpr.toActual(bindings, idToValue);
      if (desc.storageFrom) {
        actualDesc.storageFrom = idToValue.at(translateId(desc.storageFrom));
      }
      // Non-tensor outputs (scalars, SymInt, etc.) must be read back to host.
      auto outputValueId = actualOutputs[i];
      auto outputValueIt = idToValue.find(outputValueId);
      TORCH_CHECK(
          outputValueIt != idToValue.end(),
          "Output value id not found in idToValue: ",
          outputValueId);
      if (outputValueIt->second->type().kind() !=
          nativert::Type::Kind::Tensor) {
        actualDesc.neededOnHost = true;
      }
      if (actualDesc.shapeSetOnDevice || actualDesc.neededOnHost) {
        returnValues.push_back(actualOutputs[i]);
        returnTypes.push_back(outputValueIt->second->type().kind());
      }
      actualOutputDescs.push_back(std::move(actualDesc));
    }

    // Record the type kind from each output Value.
    for (size_t i = 0; i < actualOutputs.size(); ++i) {
      auto outputId = actualOutputs[i];
      auto it = idToValue.find(outputId);
      TORCH_CHECK(
          it != idToValue.end(),
          "Output value id not found in idToValue: ",
          outputId);
      actualOutputTypes.push_back(it->second->type().kind());
    }
  }
}

// --- CompositeKernel ---

CompositeKernel::CompositeKernel(
    std::vector<std::unique_ptr<ProjectOperation>>&& ops,
    std::vector<std::unique_ptr<KernelOperation>>&& kernelOps,
    const std::unordered_set<std::string>& includes)
    : ops_(std::move(ops)), kernelOpStorage_(std::move(kernelOps)) {
  auto kernelId = CompileCtx::nextKernelId();
  auto kernelName = "torchwave" + std::to_string(kernelId);
  auto entryPoint = "torch::wave::" + kernelName;

  std::stringstream ss;
  ss << "#include \"velox/experimental/torchwave/Core.cuh\"\n";
  for (const auto& inc : includes) {
    ss << "#include \"" << inc << "\"\n";
  }
  ss << "\nnamespace torch::wave {\n\n"
     << "__global__ void " << kernelName << "(TorchWaveParams params) {\n"
     << "  ENTRY;\n";
  eltTrace(
      ss,
      "\"entry blockIdx %d op %d blockInOp %d\\n\", blockIdx.x, blockInfo.op, blockInfo.blockInOp");
  std::unordered_set<std::string> emittedDecls;
  for (const auto& kop : kernelOpStorage_) {
    for (const auto& decl : kop->sharedDeclarations()) {
      if (emittedDecls.insert(decl).second) {
        ss << decl;
      }
    }
  }
  ss << "  switch (blockInfo.op) {\n";
  for (const auto& kop : kernelOpStorage_) {
    ss << "    case " << kop->opCode() << ": {\n";
    auto offsets = kop->tensorParamOffsets();
    if (!offsets.empty()) {
      ss << "  {\n    static int32_t paramOffsets[] = {";
      for (size_t i = 0; i < offsets.size(); ++i) {
        if (i > 0) {
          ss << ", ";
        }
        ss << offsets[i];
      }
      ss << "};\n"
         << "    for (auto i = threadIdx.x; i < sizeof(paramOffsets) / sizeof(paramOffsets[0]); i += blockDim.x) {\n"
         << "      param<Tensor>(blockInfo, paramOffsets[i])->init<true>();\n"
         << "    }\n"
         << "  }\n"
         << "  __syncthreads();\n";
    }
    ss << kop->code() << "      break;\n"
       << "    }\n";
  }
  ss << "  }\n"
     << "  LEAVE();\n"
     << "}\n\n"
     << "} // namespace torch::wave\n";

  auto code = ss.str();

  // Save to /tmp for debugging.
  auto filePath = "/tmp/kernel" + std::to_string(kernelId) + ".cu";
  {
    std::ofstream out(filePath);
    out << code;
  }

  // Only compile the kernel if a GPU is available.
  if (facebook::velox::wave::currentDevice()) {
    kernel_ = facebook::velox::wave::CompiledKernel::getKernel(
        entryPoint,
        [code = std::move(code),
         entryPoint,
         filePath]() -> facebook::velox::wave::KernelSpec {
          facebook::velox::wave::KernelSpec spec;
          spec.code = code;
          spec.entryPoints = {entryPoint};
          spec.filePath = filePath;
          spec.numHeaders = 0;
          spec.headers = nullptr;
          return spec;
        });

    LOG(INFO) << "Kernel " << kernelName << ": " << kernel_->info(0).toString();

    // Warmup launch: run one block of 1 thread with op=-1 (no-op) to absorb
    // first-launch overhead from the CUDA runtime.
    {
      Timer warmup("warmup", FLAGS_print_timing);
      TorchWaveParams params;
      memset(&params, 0, sizeof(params));
      params.info = nullptr;
      params.debugInfo = nullptr;
      params.inlineInfo[0].op = -1;
      void* args[] = {&params};
      facebook::velox::wave::Stream stream;
      kernel_->launch(0, 1, 1, 0, &stream, args);
      stream.wait();
    }
  }
}

facebook::velox::wave::KernelInfo CompositeKernel::kernelInfo() const {
  if (kernel_) {
    return kernel_->info(0);
  }
  return {};
}

void CompositeKernel::launch(
    int32_t numBlocks,
    int32_t numThreads,
    int32_t sharedMemory,
    facebook::velox::wave::Stream* stream,
    void** args) {
  kernel_->launch(0, numBlocks, numThreads, sharedMemory, stream, args);
}

std::string CompositeKernel::toString(Listing mode) const {
  std::stringstream ss;
  for (const auto& kop : kernelOpStorage_) {
    ss << kop->toString(mode);
  }
  auto info = kernelInfo();
  if (info.numRegs > 0) {
    ss << info.toString() << "\n";
  }
  return ss.str();
}

namespace {

void fillLaunchParams(
    LaunchData& launch,
    nativert::ExecutionFrame& frame,
    uint8_t* paramBase,
    int32_t& returnBegin,
    int32_t& returnEnd) {
  if (!launch.tensorsInFrame.empty() || !launch.scalarsInFrame.empty()) {
    // Cached path: fill only variable tensors and scalars, skip constants.
    for (size_t i = 0; i < launch.tensorsInFrame.size(); ++i) {
      fillTensorParam(
          frame.getIValue(launch.tensorsInFrame[i]).toTensor(),
          paramBase + launch.tensorOffsets[i]);
    }
    for (size_t i = 0; i < launch.scalarsInFrame.size(); ++i) {
      fillScalarParam(
          frame.getIValue(launch.scalarsInFrame[i]),
          paramBase + launch.scalarOffsets[i]);
    }
    if (!launch.returnValues.empty()) {
      if (returnBegin == -1) {
        returnBegin = launch.returnOffsets.front();
      }
      auto lastType = launch.returnTypes.back();
      int32_t lastSize =
          lastType == nativert::Type::Kind::Tensor ? sizeof(Tensor) : 8;
      returnEnd = launch.returnOffsets.back() + lastSize;
    }
    return;
  }

  auto* kernelOp = launch.launch->op;
  const auto& orderedInputs = kernelOp->orderedInputs();
  auto numInputs = kernelOp->numInputs();

  // Fill input params, recording tensor/scalar values and their offsets.
  int32_t returnCounter = 0;
  for (int32_t i = 0; i < numInputs; ++i) {
    auto offset = kernelOp->paramOffset(orderedInputs[i]);
    auto* dest = paramBase + offset;
    auto actualId = launch.actualInputs[i];
    const auto& ivalue = frame.getIValue(actualId);
    if (ivalue.isTensor()) {
      fillTensorParam(ivalue.toTensor(), dest);
      launch.tensorsInFrame.push_back(actualId);
      launch.tensorOffsets.push_back(offset);
    } else {
      fillScalarParam(ivalue, dest);
      launch.scalarsInFrame.push_back(actualId);
      launch.scalarOffsets.push_back(offset);
    }
    if (returnCounter < static_cast<int32_t>(launch.returnValues.size()) &&
        actualId == launch.returnValues[returnCounter]) {
      launch.returnOffsets.push_back(offset);
      if (returnBegin == -1) {
        returnBegin = offset;
      }
      returnEnd = offset +
          (ivalue.isTensor() ? static_cast<int32_t>(sizeof(Tensor)) : 8);
      ++returnCounter;
    }
  }

  // Fill output params, recording values and offsets.
  for (size_t i = 0; i < launch.actualOutputs.size(); ++i) {
    auto offset = kernelOp->paramOffset(orderedInputs[numInputs + i]);
    auto* dest = paramBase + offset;
    auto actualId = launch.actualOutputs[i];
    bool isTensorOutput = i < launch.actualOutputTypes.size() &&
        launch.actualOutputTypes[i] == nativert::Type::Kind::Tensor;
    if (isTensorOutput) {
      const auto& ivalue = frame.getIValue(actualId);
      TORCH_CHECK(ivalue.isTensor(), "Expected tensor for output param");
      fillTensorParam(ivalue.toTensor(), dest);
      launch.tensorsInFrame.push_back(actualId);
      launch.tensorOffsets.push_back(offset);
    } else {
      // Non-tensor output: write a 64-bit zero placeholder.
      *reinterpret_cast<int64_t*>(dest) = 0;
      launch.scalarsInFrame.push_back(actualId);
      launch.scalarOffsets.push_back(offset);
    }
    if (returnCounter < static_cast<int32_t>(launch.returnValues.size()) &&
        actualId == launch.returnValues[returnCounter]) {
      launch.returnOffsets.push_back(offset);
      if (returnBegin == -1) {
        returnBegin = offset;
      }
      returnEnd =
          offset + (isTensorOutput ? static_cast<int32_t>(sizeof(Tensor)) : 8);
      ++returnCounter;
    }
  }

  // Fill constant params (first time only, constants don't change).
  auto constantOffset = kernelOp->constantAreaOffset();
  const auto& opConstants = launch.op->constants();
  for (auto idx : launch.launch->constantIndices) {
    auto* dest = paramBase + constantOffset;
    fillScalarParam(*opConstants[idx], dest);
    constantOffset += 8;
  }
}

void allocateLaunchOutputs(
    const LaunchData& launch,
    nativert::ExecutionFrame& frame,
    const ValueTypes& types,
    nativert::ValueId largestId) {
  const auto& descs = launch.actualOutputDescs;
  const auto& actualOutputs = launch.actualOutputs;
  const auto& outputTypes = launch.actualOutputTypes;

  // Shortcut: if largestId is set, resize tensor outputs to match it.
  if (largestId >= 0) {
    auto dims = frame.getIValue(largestId).toTensor().sizes();
    for (size_t i = 0; i < descs.size(); ++i) {
      // Skip non-tensor outputs.
      if (i < outputTypes.size() &&
          outputTypes[i] != nativert::Type::Kind::Tensor) {
        continue;
      }
      auto actualId = actualOutputs[i];
      auto& existing = frame.getIValue(actualId);
      if (existing.isTensor() && existing.toTensor().is_cuda()) {
        auto& tensor = existing.toTensor();
        if (tensor.sizes() != dims) {
          tensor.resize_(dims);
        }
      } else {
        auto* meta = types.types[actualId];
        if (!meta) {
          continue;
        }
        auto tensor = at::empty(
            dims, at::TensorOptions().dtype(meta->dtype()).device(at::kCUDA));
        frame.setIValue(actualId, std::move(tensor));
      }
    }
    return;
  }

  const auto& bindings = launch.op->bindings();
  for (size_t i = 0; i < descs.size(); ++i) {
    // Skip non-tensor outputs.
    if (i < outputTypes.size() &&
        outputTypes[i] != nativert::Type::Kind::Tensor) {
      continue;
    }
    auto actualId = actualOutputs[i];
    std::vector<int64_t> dims;
    if (descs[i].reserveShape) {
      auto shapes = descs[i].reserveShape(nullptr, frame, bindings);
      TORCH_CHECK(
          !shapes.empty(),
          "OutputReserveFunc returned empty shapes for output ",
          i);
      dims.assign(shapes[0].begin(), shapes[0].end());
    } else if (descs[i].sizeExpr.op != SizeShortcut::kNone) {
      dims = {descs[i].sizeExpr.numElements(&frame)};
    } else {
      continue;
    }
    auto& existing = frame.getIValue(actualId);
    if (existing.isTensor() && existing.toTensor().is_cuda()) {
      auto& tensor = existing.toTensor();
      if (tensor.sizes() != dims) {
        tensor.resize_(dims);
      }
    } else {
      auto* meta = types.types[actualId];
      if (!meta) {
        continue;
      }
      auto tensor = at::empty(
          dims, at::TensorOptions().dtype(meta->dtype()).device(at::kCUDA));
      frame.setIValue(actualId, std::move(tensor));
    }
  }
}

int32_t launchParamSize(const LaunchData& launch) {
  return launch.launch->op->constantAreaOffset() +
      static_cast<int32_t>(launch.launch->constantIndices.size()) * 8;
}

facebook::velox::wave::WaveBufferPtr& getOrAllocateBuffer(
    std::vector<std::vector<facebook::velox::wave::WaveBufferPtr>>& buffers,
    int32_t sequenceNumber,
    int32_t stepIdx,
    int64_t requiredBytes,
    facebook::velox::wave::GpuArena* arena) {
  if (static_cast<int32_t>(buffers.size()) <= sequenceNumber) {
    buffers.resize(sequenceNumber + 1);
  }
  auto& steps = buffers[sequenceNumber];
  if (static_cast<int32_t>(steps.size()) <= stepIdx) {
    steps.resize(stepIdx + 1);
  }
  auto& buffer = steps[stepIdx];
  if (!buffer || buffer->capacity() < static_cast<size_t>(requiredBytes)) {
    buffer = arena->allocateBytes(requiredBytes);
  }
  return buffer;
}

StepVectors& getStepVectors(
    std::vector<std::vector<StepVectors>>& allSteps,
    int32_t sequenceNumber,
    int32_t stepIdx) {
  if (static_cast<int32_t>(allSteps.size()) <= sequenceNumber) {
    allSteps.resize(sequenceNumber + 1);
  }
  auto& steps = allSteps[sequenceNumber];
  if (static_cast<int32_t>(steps.size()) <= stepIdx) {
    steps.resize(stepIdx + 1);
  }
  return steps[stepIdx];
}

// Returns true if every launch's numElements falls within the cached
// [sizesLower, sizesUpper] bounds.
bool gridSizesMatch(
    const std::vector<LaunchData>& launches,
    const StepVectors& sv) {
  if (!sv.hasGridCache || launches.size() != sv.sizesLower.size()) {
    return false;
  }
  for (size_t i = 0; i < launches.size(); ++i) {
    auto n = launches[i].numElements;
    if (n < sv.sizesLower[i] || n > sv.sizesUpper[i]) {
      return false;
    }
  }
  return true;
}

// Updates the cached size bounds to [size - size/8, size + size/8] for each
// launch.
void updateGridSizeBounds(
    const std::vector<LaunchData>& launches,
    StepVectors& sv) {
  sv.sizesLower.resize(launches.size());
  sv.sizesUpper.resize(launches.size());
  for (size_t i = 0; i < launches.size(); ++i) {
    auto n = launches[i].numElements;
    auto margin = n / 8;
    sv.sizesLower[i] = n - margin;
    sv.sizesUpper[i] = n + margin;
  }
  sv.hasGridCache = true;
}

} // namespace

// --- CompositeInvocation ---

void CompositeInvocation::gatherLaunches(
    const ExecutionState& state,
    std::vector<GridChoice>& grids,
    int32_t stepIdx,
    StepVectors& sv) {
  const auto& idToValue = state.waveGraph->idToValue();
  int32_t kernelIdx = 0;
  int32_t standaloneIdx = 0;
  sv.gridChanged = false;
  for (size_t i = 0; i < ops.size(); ++i) {
    auto* grid = grids[i].grid;
    if (stepIdx >= static_cast<int32_t>(grid->size())) {
      continue;
    }
    auto& step = (*grid)[stepIdx];
    for (size_t j = 0; j < step.size(); ++j) {
      auto& launch = step[j];
      if (launch.op != nullptr) {
        bool isNew = kernelIdx >= static_cast<int32_t>(sv.kernels.size());
        if (isNew) {
          sv.kernels.emplace_back(launch, ops[i], idToValue);
        }
        auto& data = sv.kernels[kernelIdx];
        bool hasByLargestInput = !data.launch->op->outputDescs().empty() &&
            data.launch->op->outputDescs()[0].byLargestInput;
        nativert::ValueId largestId = -1;
        data.numElements = data.sizeExpr.numElements(
            state.frame, hasByLargestInput ? &largestId : nullptr);

        if (launch.op->isGridChoice()) {
          auto* projectOp = ops[i].projectOp();
          bool wantSingleBlock;
          if (WaveConfig::get().useSingleBlock.has_value()) {
            wantSingleBlock = *WaveConfig::get().useSingleBlock;
          } else {
            wantSingleBlock =
                data.numElements <= projectOp->singleBlockMaxSize();
          }
          if (wantSingleBlock != grids[i].singleBlock) {
            if (wantSingleBlock) {
              grids[i].grid = &projectOp->singleBlockGrid();
            } else {
              grids[i].grid = &projectOp->grid();
            }
            grids[i].singleBlock = wantSingleBlock;
            if (!isNew) {
              sv.gridChanged = true;
            }
            // Reassign LaunchData from the new grid's launch.
            auto& newLaunch = (*grids[i].grid)[stepIdx][j];
            data = LaunchData(newLaunch, ops[i], idToValue);
            largestId = -1;
            data.numElements = data.sizeExpr.numElements(
                state.frame, hasByLargestInput ? &largestId : nullptr);
          }
        }

        allocateLaunchOutputs(data, *state.frame, *state.valueTypes, largestId);
        ++kernelIdx;
      } else {
        if (standaloneIdx >= static_cast<int32_t>(sv.standalones.size())) {
          sv.standalones.emplace_back(launch, ops[i], idToValue);
        }
        ++standaloneIdx;
      }
    }
  }
}

void invalidateStepVectors(std::vector<StepVectors>& steps, int32_t stepIdx) {
  for (auto i = stepIdx; i < static_cast<int32_t>(steps.size()); ++i) {
    auto& sv = steps[i];
    sv.hasGridCache = false;
    sv.hasLaunchCache = false;
    for (auto& data : sv.kernels) {
      data.tensorsInFrame.clear();
      data.tensorOffsets.clear();
      data.scalarsInFrame.clear();
      data.scalarOffsets.clear();
    }
  }
}

void CompositeInvocation::processReturnData(
    StepVectors& sv,
    nativert::ExecutionFrame& frame,
    uint8_t* pinnedBase) {
  for (size_t i = 0; i < sv.kernels.size(); ++i) {
    auto& data = sv.kernels[i];
    if (data.returnValues.empty()) {
      continue;
    }
    for (size_t j = 0; j < data.returnValues.size(); ++j) {
      auto actualId = data.returnValues[j];
      auto absOffset = sv.paramOffsets[i] + data.returnOffsets[j];
      auto typeKind = data.returnTypes[j];
      if (typeKind == nativert::Type::Kind::Tensor) {
        auto* t = reinterpret_cast<Tensor*>(pinnedBase + absOffset);
        auto& tensor = frame.getIValue(actualId).toTensor();
        std::vector<int64_t> newDims(t->rank);
        for (int d = 0; d < t->rank; ++d) {
          newDims[d] = t->dims[d];
        }
        tensor.resize_(newDims);
      } else if (typeKind == nativert::Type::Kind::SymFloat) {
        frame.setIValue(
            actualId,
            c10::IValue(*reinterpret_cast<double*>(pinnedBase + absOffset)));
      } else if (typeKind == nativert::Type::Kind::SymBool) {
        frame.setIValue(
            actualId,
            c10::IValue(
                *reinterpret_cast<int64_t*>(pinnedBase + absOffset) != 0));
      } else {
        // SymInt and other non-tensor types: read as int64.
        frame.setIValue(
            actualId,
            c10::IValue(*reinterpret_cast<int64_t*>(pinnedBase + absOffset)));
      }
    }
  }
}

void CompositeInvocation::execute(ExecutionState& state) {
  Timer ex("comp inv execute", FLAGS_print_timing);
  auto& frame = *state.frame;

  auto& sv0 = getStepVectors(state.stepVectors, sequenceNumber, 0);
  auto& gridChoices = sv0.gridChoices;
  if (gridChoices.empty()) {
    for (auto& op : ops) {
      gridChoices.push_back({0, false, &op.projectOp()->grid()});
    }
  }

  int32_t blockSize;
  for (int32_t stepIdx = 0;; ++stepIdx) {
    auto& sv = getStepVectors(state.stepVectors, sequenceNumber, stepIdx);
    // Re-fetch since the resize above may have invalidated the reference.
    auto& gridChoices = state.stepVectors[sequenceNumber][0].gridChoices;

    {
      Timer t("gather", FLAGS_print_timing);
      gatherLaunches(state, gridChoices, stepIdx, sv);
    }
    if (sv.gridChanged) {
      invalidateStepVectors(state.stepVectors[sequenceNumber], stepIdx);
    }
    if (sv.kernels.empty() && sv.standalones.empty()) {
      break;
    }

    if (!sv.standalones.empty()) {
      runStandalones(
          sv.standalones,
          state,
          *state.kernelMap,
          *state.standaloneIndices,
          *state.standaloneStats);
    }

    if (sv.kernels.empty()) {
      continue;
    }

    {
      Timer t("grid", FLAGS_print_timing);
      // Reuse previous makeGrid result if sizes are within cached bounds.
      if (gridSizesMatch(sv.kernels, sv)) {
        blockSize = sv.cachedBlockSize;
      } else {
        blockSize = makeGrid(sv.kernels, sv, kernelInfo.maxOccupancy0);
        TORCH_CHECK(
            (blockSize & (blockSize - 1)) == 0,
            "Block size must be a power of two, got ",
            blockSize);
        sv.cachedBlockSize = blockSize;
        updateGridSizeBounds(sv.kernels, sv);
      }
    }

    // Compute total param bytes and track per-launch param offsets.
    auto numBlocks = sv.blocks.size();
    auto blockInfoBytes = static_cast<int64_t>(numBlocks) * sizeof(BlockInfo);

    int64_t totalPinnedBytes;
    int64_t totalAllocBytes;
    uint8_t* pinnedBase;
    uint8_t* deviceBase;
    {
      Timer t("alloc outputs", FLAGS_print_timing);
      sv.paramOffsets.resize(sv.kernels.size());
      int64_t paramCursor = blockInfoBytes;
      for (size_t i = 0; i < sv.kernels.size(); ++i) {
        sv.paramOffsets[i] = paramCursor;
        paramCursor += launchParamSize(sv.kernels[i]);
      }
      totalPinnedBytes = paramCursor;

      // Allocate extra space for per-block DebugInfo at the end of the buffer.
      auto debugInfoBytes = static_cast<int64_t>(numBlocks) * sizeof(DebugInfo);
      totalAllocBytes = totalPinnedBytes + debugInfoBytes;

      // Get or allocate pinned and device buffers.
      auto& pinnedBuffer = getOrAllocateBuffer(
          state.pinnedBuffers,
          sequenceNumber,
          stepIdx,
          totalAllocBytes,
          state.pinnedArena);
      auto& deviceBuffer = getOrAllocateBuffer(
          state.deviceBuffers,
          sequenceNumber,
          stepIdx,
          totalAllocBytes,
          state.deviceArena);
      pinnedBase = pinnedBuffer->as<uint8_t>();
      deviceBase = deviceBuffer->as<uint8_t>();
    }

    auto* deviceDebugBase =
        reinterpret_cast<DebugInfo*>(deviceBase + totalPinnedBytes);
    int32_t returnBegin = -1;
    int32_t returnEnd = -1;
    {
      Timer t("fill params", FLAGS_print_timing);
      // Copy BlockInfos to beginning of pinned buffer.
      if (!sv.blocks.empty()) {
        memcpy(pinnedBase, sv.blocks.data(), blockInfoBytes);
      }

      // Fill params for each kernel launch.
      for (size_t i = 0; i < sv.kernels.size(); ++i) {
        int32_t rb = -1;
        int32_t re = -1;
        fillLaunchParams(
            sv.kernels[i], frame, pinnedBase + sv.paramOffsets[i], rb, re);
        if (rb >= 0) {
          if (returnBegin == -1) {
            returnBegin = sv.paramOffsets[i] + rb;
          }
          returnEnd = sv.paramOffsets[i] + re;
        }
      }

      // Patch BlockInfo params and debugInfo pointers to device-side addresses.
      auto* pinnedBlocks = reinterpret_cast<BlockInfo*>(pinnedBase);
      for (size_t b = 0; b < numBlocks; ++b) {
        auto idx = sv.launchIndices[b];
        pinnedBlocks[b].params = deviceBase + sv.paramOffsets[idx];
        pinnedBlocks[b].debugInfo = deviceDebugBase + b;
      }
    }

    // Record debug info locations for later D2H transfer.
    state.launchDebugInfos->push_back(
        {reinterpret_cast<DebugInfo*>(pinnedBase + totalPinnedBytes),
         deviceDebugBase,
         static_cast<int32_t>(numBlocks)});

    {
      Timer t("launch1", FLAGS_print_timing);
      launch(
          static_cast<int32_t>(numBlocks),
          blockSize,
          pinnedBase,
          deviceBase,
          totalPinnedBytes,
          returnBegin,
          returnEnd,
          deviceDebugBase,
          state.stream.get(),
          sv,
          stepIdx);
    }

    if (returnBegin >= 0) {
      processReturnData(sv, frame, pinnedBase);
    }
  }
}

void CompositeInvocation::launch(
    int32_t numBlocks,
    int32_t blockSize,
    uint8_t* pinnedBase,
    uint8_t* deviceBase,
    int64_t totalPinnedBytes,
    int32_t returnBegin,
    int32_t returnEnd,
    DebugInfo* deviceDebugBase,
    facebook::velox::wave::Stream* stream,
    const StepVectors& sv,
    int32_t stepIdx) {
  TorchWaveParams params;
  params.info = reinterpret_cast<BlockInfo*>(deviceBase);
  params.debugInfo = deviceDebugBase;
  void* args[] = {&params};

  auto* pinnedBlocks = reinterpret_cast<BlockInfo*>(pinnedBase);

  if (FLAGS_debug_single_ops) {
    // Save original opcodes.
    std::vector<int32_t> originalOps(numBlocks);
    for (int32_t b = 0; b < numBlocks; ++b) {
      originalOps[b] = pinnedBlocks[b].op;
    }

    for (int32_t active = 0; active < numBlocks; ++active) {
      // Set all blocks to no-op except the active one.
      for (int32_t b = 0; b < numBlocks; ++b) {
        pinnedBlocks[b].op = (b == active) ? originalOps[b] : -1;
      }
      try {
        stream->hostToDeviceAsync(deviceBase, pinnedBase, totalPinnedBytes);
        kernel->launch(numBlocks, blockSize, 0, stream, args);
        if (returnBegin >= 0) {
          stream->deviceToHostAsync(
              pinnedBase + returnBegin,
              deviceBase + returnBegin,
              returnEnd - returnBegin);
        }
        stream->wait();
      } catch (const std::exception& e) {
        auto opCode = originalOps[active];
        auto launchIdx = sv.launchIndices[active];
        std::string opText;
        if (launchIdx < static_cast<int32_t>(sv.kernels.size()) &&
            sv.kernels[launchIdx].launch && sv.kernels[launchIdx].launch->op) {
          opText = sv.kernels[launchIdx].launch->op->toString();
        }
        LOG(ERROR) << "debug_single_ops: block " << active << " opCode "
                   << opCode << " blockInOp "
                   << pinnedBlocks[active].blockInOp << " stepIdx " << stepIdx
                   << " op: " << opText << " error: " << e.what();
        throw;
      }
    }

    // Restore original opcodes.
    for (int32_t b = 0; b < numBlocks; ++b) {
      pinnedBlocks[b].op = originalOps[b];
    }
  } else {
    stream->hostToDeviceAsync(deviceBase, pinnedBase, totalPinnedBytes);
    kernel->launch(numBlocks, blockSize, 0, stream, args);
    if (returnBegin >= 0) {
      stream->deviceToHostAsync(
          pinnedBase + returnBegin,
          deviceBase + returnBegin,
          returnEnd - returnBegin);
      stream->wait();
    }
  }
}

std::string CompositeInvocation::toString(Listing mode, int32_t ordinal) const {
  std::stringstream ss;

  // Collect distinct ProjectOperations.
  std::vector<ProjectOperation*> projectOps;
  std::unordered_map<ProjectOperation*, int32_t> projectOpIndex;
  for (const auto& op : ops) {
    auto* po = op.projectOp();
    if (projectOpIndex.find(po) == projectOpIndex.end()) {
      projectOpIndex[po] = static_cast<int32_t>(projectOps.size());
      projectOps.push_back(po);
    }
  }

  // Print OpInvocations with their ProjectOperation ordinal and bindings.
  for (size_t i = 0; i < ops.size(); ++i) {
    auto it = projectOpIndex.find(ops[i].projectOp());
    ss << ordinal << "." << i << ": ProjectOp " << it->second;
    const auto& bindings = ops[i].bindings();
    for (const auto& [formalId, actualId] : bindings) {
      ss << " %" << formalId << " = %" << actualId;
    }
    ss << "\n";
  }

  // Print distinct ops with their grids.
  ss << "\nDistinct Ops\n";
  for (size_t i = 0; i < projectOps.size(); ++i) {
    ss << "Op " << ordinal << "." << i << "\n";
    ss << projectOps[i]->toString(mode);
  }

  return ss.str();
}

// --- CompiledNode ---

void CompiledNode::execute(ExecutionState& state) {
  kernels_->execute(state);
}

std::string CompiledNode::toString(Listing mode, int32_t ordinal) const {
  std::stringstream ss;
  if (mode == kExprs) {
    // Collect distinct ProjectOperations and count invocations.
    std::vector<ProjectOperation*> projectOps;
    std::unordered_map<ProjectOperation*, int32_t> invocationCount;
    for (const auto& op : kernels_->ops) {
      auto* po = op.projectOp();
      if (invocationCount[po]++ == 0) {
        projectOps.push_back(po);
      }
    }
    for (auto* po : projectOps) {
      ss << invocationCount[po] << "x "
         << po->subgraph().toString(mode) << "\n";
    }
  } else {
    ss << kernels_->toString(mode, ordinal);
  }
  return ss.str();
}

// --- WaveGraph ---

void WaveGraph::normalizeAndAnnotateGraph() {
  for (auto& node : graph_->nodes()) {
    const auto* schema = findSchema(node.target());
    bool resolveDtype =
        (node.target() == "torch.ops.aten.sum.default" ||
         node.target() == "torch.ops.aten.cumsum.default") &&
        !node.inputs().empty();
    if (schema) {
      for (const auto& schemaArg : schema->arguments()) {
        if (!schemaArg.default_value()) {
          continue;
        }
        if (node.tryGetInput(schemaArg.name())) {
          continue;
        }
        if (node.tryGetAttribute(schemaArg.name())) {
          continue;
        }
        auto defaultVal = iValueToConstant(*schemaArg.default_value());
        // For sum/cumsum, resolve dtype=None to the concrete output type.
        // PyTorch promotes integer inputs to int64 when dtype is unspecified.
        // The kernels handle the cast inline via separate TIn/TOut template
        // parameters.
        if (resolveDtype && schemaArg.name() == "dtype" &&
            std::holds_alternative<nativert::None>(defaultVal)) {
          auto* selfInput = node.inputs()[0].value;
          auto inputId = selfInput->id();
          if (inputId < static_cast<int>(types_.types.size()) &&
              types_.types[inputId]) {
            auto inputDtype = types_.types[inputId]->dtype();
            auto outDtype =
                c10::isIntegralType(inputDtype, /*includeBool=*/true)
                ? c10::ScalarType::Long
                : inputDtype;
            defaultVal = std::string(c10::toString(outDtype));
          }
        }
        node.addAttribute(
            nativert::Attribute{
                std::string(schemaArg.name()), std::move(defaultVal)});
      }
    }

    const auto* md = Registry::metadata(node.target());
    if (md && md->makeMultiKernelVariant) {
      auto* lastNode = md->makeMultiKernelVariant(&node, this);
      std::vector<ValueCP> inputs;
      inputs.reserve(node.inputs().size());
      for (const auto& input : node.inputs()) {
        inputs.push_back(input.value);
      }
      std::vector<const nativert::TensorMeta*> inputTypes;
      inputTypes.reserve(inputs.size());
      for (const auto* value : inputs) {
        auto id = value->id();
        inputTypes.push_back(
            id < static_cast<int>(types_.types.size()) ? types_.types[id]
                                                       : nullptr);
      }
      multiKernelVariants_[&node] = Subgraph{
          .root = lastNode,
          .inputs = std::move(inputs),
          .inputTypes = std::move(inputTypes)};
    }
  }
}

static thread_local WaveGraph* threadWaveGraph{nullptr};

WaveGraph*& waveGraph() {
  return threadWaveGraph;
}

const Metadata* nodeMeta(NodeCP node) {
  auto* graph = waveGraph();
  TORCH_CHECK(graph, "No WaveGraph on this thread");
  auto& infos = graph->nodeInfos();
  auto it = infos.find(node);
  if (it != infos.end()) {
    return it->second.metadata;
  }
  auto* meta = Registry::metadata(node->target());
  infos[node] = NodeInfo{meta};
  return meta;
}

WaveGraph::WaveGraph(nativert::Graph& graph, ValueTypes types)
    : types_(std::move(types)), graph_(&graph) {
  waveGraph() = this;
  SCOPE_EXIT {
    waveGraph() = nullptr;
  };
  normalizeAndAnnotateGraph();
  for (auto* v : graph.values()) {
    idToValue_[v->id()] = v;
  }
  optimizer_ = std::make_unique<Optimizer>(types_);
  optimizer_->optimizeGraph(graph_);
  ParallelNodes parallelNodes;
  auto* lastProjectNode = parallelNodes.makeParallelNodes(graph);

  CompileCtx ctx(*this);

  // Collect nodes from last to first, then reverse to get leafmost first.
  std::vector<ProjectNode*> ordered;
  for (auto* pn = lastProjectNode; pn != nullptr; pn = pn->input()) {
    ordered.push_back(pn);
  }
  std::reverse(ordered.begin(), ordered.end());

  for (auto* pn : ordered) {
    auto compiled = ctx.compileNode(*pn);
    if (compiled) {
      nodes_.push_back(std::move(compiled));
    }
  }

  // Build standaloneIndices_ by walking all launches across all compiled nodes.
  for (const auto& node : nodes_) {
    const auto* inv = node->kernels();
    if (!inv) {
      continue;
    }
    for (const auto& op : inv->ops) {
      auto* projectOp = op.projectOp();
      auto scanGrid = [&](LaunchGrid& grid) {
        for (const auto& step : grid) {
          for (const auto& launch : step) {
            if (launch.standalone) {
              auto it = op.nodeMap().find(launch.standalone);
              if (it != op.nodeMap().end()) {
                auto* actualNode = it->second;
                if (standaloneIndices_.find(actualNode) ==
                    standaloneIndices_.end()) {
                  standaloneIndices_[actualNode] =
                      static_cast<int32_t>(standaloneIndices_.size());
                }
              }
            }
          }
        }
      };
      scanGrid(projectOp->grid());
      scanGrid(projectOp->singleBlockGrid());
    }
  }
  standaloneStats_.resize(standaloneIndices_.size());
  optimizer_.reset();
}

void WaveGraph::optimizeNode(const nativert::Node* node) {
  TORCH_CHECK(optimizer_, "optimizeNode called outside WaveGraph construction");
  optimizer_->optimizeNode(node);
}

WaveGraph::~WaveGraph() = default;

std::unique_ptr<ExecutionState> WaveGraph::getState() {
  std::lock_guard<std::mutex> lock(statePoolMutex_);
  if (!statePool_.empty()) {
    auto state = std::move(statePool_.back());
    statePool_.pop_back();
    return state;
  }
  return std::make_unique<ExecutionState>();
}

void WaveGraph::returnState(std::unique_ptr<ExecutionState> state) {
  std::lock_guard<std::mutex> lock(statePoolMutex_);
  statePool_.push_back(std::move(state));
}

std::string WaveGraph::toString(Listing mode) const {
  std::stringstream ss;
  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (i > 0) {
      ss << "\n";
    }
    ss << "Node " << i << ":\n";
    ss << nodes_[i]->toString(mode, i);
  }
  return ss.str();
}

namespace {

torch::_export::ScalarType toExportScalarType(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Byte:
      return torch::_export::ScalarType::BYTE;
    case c10::ScalarType::Char:
      return torch::_export::ScalarType::CHAR;
    case c10::ScalarType::Short:
      return torch::_export::ScalarType::SHORT;
    case c10::ScalarType::Int:
      return torch::_export::ScalarType::INT;
    case c10::ScalarType::Long:
      return torch::_export::ScalarType::LONG;
    case c10::ScalarType::Half:
      return torch::_export::ScalarType::HALF;
    case c10::ScalarType::Float:
      return torch::_export::ScalarType::FLOAT;
    case c10::ScalarType::Double:
      return torch::_export::ScalarType::DOUBLE;
    case c10::ScalarType::ComplexHalf:
      return torch::_export::ScalarType::COMPLEXHALF;
    case c10::ScalarType::ComplexFloat:
      return torch::_export::ScalarType::COMPLEXFLOAT;
    case c10::ScalarType::ComplexDouble:
      return torch::_export::ScalarType::COMPLEXDOUBLE;
    case c10::ScalarType::Bool:
      return torch::_export::ScalarType::BOOL;
    case c10::ScalarType::BFloat16:
      return torch::_export::ScalarType::BFLOAT16;
    case c10::ScalarType::UInt16:
      return torch::_export::ScalarType::UINT16;
    case c10::ScalarType::Float8_e4m3fn:
      return torch::_export::ScalarType::FLOAT8E4M3FN;
    case c10::ScalarType::Float8_e5m2:
      return torch::_export::ScalarType::FLOAT8E5M2;
    case c10::ScalarType::Float8_e4m3fnuz:
      return torch::_export::ScalarType::FLOAT8E4M3FNUZ;
    case c10::ScalarType::Float8_e5m2fnuz:
      return torch::_export::ScalarType::FLOAT8E5M2FNUZ;
    default:
      TORCH_CHECK(false, "unsupported scalar type ", static_cast<int>(dtype));
  }
}

} // namespace

void WaveGraph::registerTensorMeta(ValueCP value, c10::ScalarType dtype) {
  torch::_export::TensorMeta exportMeta;
  exportMeta.set_dtype(toExportScalarType(dtype));
  exportMeta.set_layout(torch::_export::Layout::Strided);
  exportMeta.set_requires_grad(false);
  torch::_export::Device device;
  device.set_type("cuda");
  device.set_index(0);
  exportMeta.set_device(std::move(device));
  torch::_export::SymInt zero;
  zero.set_as_int(0);
  exportMeta.set_storage_offset(std::move(zero));

  auto meta = std::make_unique<nativert::TensorMeta>(exportMeta);
  auto* metaPtr = meta.get();
  metaStorage_.push_back(std::move(meta));

  auto id = value->id();
  if (id >= static_cast<int>(types_.types.size())) {
    types_.types.resize(id + 1, nullptr);
  }
  types_.types[id] = metaPtr;
}

nativert::Value* WaveGraph::newTensorValue(
    nativert::Node* node,
    std::string_view name,
    c10::ScalarType dtype) {
  auto uname = uniqueName(name);
  auto* value =
      node->addOutput(uname, nativert::Type(nativert::Type::Kind::Tensor));
  registerTensorMeta(value, dtype);
  createdValueDtypes_[value] = dtype;
  return value;
}

nativert::Value* WaveGraph::newScalarValue(
    nativert::Node* node,
    std::string_view name,
    c10::ScalarType dtype) {
  nativert::Type::Kind kind;
  switch (dtype) {
    case c10::ScalarType::Float:
    case c10::ScalarType::Double:
    case c10::ScalarType::Half:
    case c10::ScalarType::BFloat16:
      kind = nativert::Type::Kind::SymFloat;
      break;
    case c10::ScalarType::Bool:
      kind = nativert::Type::Kind::SymBool;
      break;
    default:
      kind = nativert::Type::Kind::SymInt;
      break;
  }
  auto uname = uniqueName(name);
  auto* value = node->addOutput(uname, nativert::Type(kind));
  auto id = value->id();
  if (id >= static_cast<int>(types_.types.size())) {
    types_.types.resize(id + 1, nullptr);
  }
  createdValueDtypes_[value] = dtype;
  return value;
}

bool WaveGraph::isCreatedValue(ValueCP value) const {
  return createdValueDtypes_.count(value) > 0;
}

nativert::Value* WaveGraph::duplicateValue(ValueCP original) {
  if (!placeholderNode_) {
    placeholderNode_ = graph_->createNode("tw.placeholder", {});
  }
  auto it = createdValueDtypes_.find(original);
  TORCH_CHECK(
      it != createdValueDtypes_.end(),
      "Cannot duplicate: value not tracked by newTensorValue/newScalarValue");
  auto dtype = it->second;
  nativert::Value* result;
  if (original->type().kind() == nativert::Type::Kind::Tensor) {
    result = newTensorValue(placeholderNode_, original->name(), dtype);
  } else {
    result = newScalarValue(placeholderNode_, original->name(), dtype);
  }
  idToValue_[result->id()] = result;
  return result;
}

void initValueTypes(
    const nativert::Graph& graph,
    ValueTypes& types,
    std::vector<std::unique_ptr<nativert::TensorMeta>>& metaStore) {
  const auto& tensorValuesMeta = graph.tensorValuesMeta();
  auto numValues = graph.values().size();
  types.types.resize(numValues, nullptr);
  types.constraints.resize(numValues);
  for (const auto* value : graph.values()) {
    if (value == nullptr) {
      continue;
    }
    auto it = tensorValuesMeta.find(std::string{value->name()});
    if (it != tensorValuesMeta.end()) {
      auto meta = std::make_unique<nativert::TensorMeta>(it->second);
      types.constraints[value->id()].rank = meta->dim();
      types.types[value->id()] = meta.get();
      metaStore.push_back(std::move(meta));
    }
  }
}

} // namespace torch::wave
