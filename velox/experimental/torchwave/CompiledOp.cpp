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
#include "velox/experimental/torchwave/ParallelExpr.h"
#include "velox/experimental/torchwave/Utils.h"

#include <ATen/ATen.h>
#include <algorithm>
#include <fstream>
#include <sstream>

#include "velox/experimental/wave/common/GpuArena.h"

namespace torch::wave {

namespace {

bool isAllElementwise(
    const nativert::Node* node,
    const std::unordered_set<const nativert::Value*>& subgraphInputs,
    std::unordered_set<const nativert::Node*>& visited) {
  if (!visited.insert(node).second) {
    return true;
  }
  auto* meta = Registry::metadata(node->target());
  if (!meta || !meta->elementWise) {
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
    const nativert::Node* node,
    const std::unordered_set<const nativert::Value*>& subgraphInputs,
    std::unordered_set<const nativert::Value*>& seen,
    std::vector<const nativert::Value*>& result) {
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
    const nativert::Node* node,
    const std::unordered_set<const nativert::Value*>& inputs,
    std::unordered_set<const nativert::Node*>& visited,
    int32_t& offset,
    std::unordered_map<
        std::pair<const nativert::Node*, std::string>,
        int32_t,
        KernelOperation::NodeAttrHash>& attrOffsets) {
  forEachSortedAttribute(
      node,
      inputs,
      visited,
      [&](const nativert::Node* n, const nativert::Attribute& attr) {
        attrOffsets[{n, attr.name}] = offset;
        offset += 8;
      });
}

float sumNodeCosts(
    const nativert::Node* node,
    const std::unordered_set<const nativert::Value*>& inputs,
    std::unordered_set<const nativert::Node*>& visited) {
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
    const nativert::Node* node,
    const std::unordered_set<const nativert::Value*>& inputSet,
    const std::unordered_set<const nativert::Value*>& outputSet) {
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
    const nativert::Node* formalNode,
    const nativert::Node* actualNode,
    const std::unordered_set<const nativert::Value*>& formalInputs,
    const std::unordered_set<const nativert::Value*>& outputValues,
    std::unordered_set<const nativert::Node*>& visited,
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
    t->strides[i] = i < tensor.dim() ? tensor.stride(i) : 0;
  }
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

} // namespace

std::vector<std::vector<Dim>> elementwiseOutputShape(
    const std::vector<const nativert::Value*>& inputs,
    const nativert::Node* /*node*/,
    nativert::ExecutionFrame& frame,
    FormalToActual map) {
  int64_t maxNumel = -1;
  std::vector<Dim> bestShape;
  for (const auto* input : inputs) {
    auto it = map.find(input->id());
    TORCH_CHECK(
        it != map.end(),
        "Input value %v",
        input->id(),
        " not found in FormalToActual map");
    auto actualId = it->second;
    auto tensor = frame.getTensor(actualId);
    auto numel = tensor.numel();
    if (numel > maxNumel) {
      maxNumel = numel;
      auto sizes = tensor.sizes();
      bestShape.assign(sizes.begin(), sizes.end());
    }
  }
  return {bestShape};
}

void makeGrid(
    const std::vector<OpInvocation>& ops,
    const std::vector<int64_t>& numElements,
    std::vector<BlockInfo>& blocks,
    std::vector<int32_t>& opIndices,
    std::vector<OpInvocation*>& nextOps) {
  const int32_t blockSize = 256;

  // Compute cost per op: numElements * unitCost.
  std::vector<float> costs(ops.size());
  float totalCost = 0;
  for (size_t i = 0; i < ops.size(); ++i) {
    costs[i] = static_cast<float>(numElements[i]) * ops[i].op()->unitCost();
    totalCost += costs[i];
  }

  // Max blocks each op could use.
  std::vector<int32_t> maxBlocks(ops.size());
  for (size_t i = 0; i < ops.size(); ++i) {
    maxBlocks[i] =
        static_cast<int32_t>((numElements[i] + blockSize - 1) / blockSize);
    if (maxBlocks[i] < 1) {
      maxBlocks[i] = 1;
    }
  }

  // Target blocks from device SM count.
  int32_t numSMs = 100;
  auto* device = facebook::velox::wave::currentDevice();
  if (device) {
    numSMs = device->numSM;
  }
  int32_t targetBlocks = numSMs * 4;

  // Assign blocks pro rata by cost, at least 1 per op, capped by maxBlocks.
  std::vector<int32_t> numBlocksPerOp(ops.size());
  int32_t totalAssigned = 0;
  for (size_t i = 0; i < ops.size(); ++i) {
    float fraction = (totalCost > 0) ? costs[i] / totalCost : 1.0f / ops.size();
    int32_t assigned =
        std::max(1, static_cast<int32_t>(fraction * targetBlocks + 0.5f));
    assigned = std::min(assigned, maxBlocks[i]);
    // Round down blocks so all but the last process a multiple of blockSize
    // elements.
    if (assigned > 1) {
      auto elemsPerBlock = (numElements[i] + assigned - 1) / assigned;
      auto alignedElems = roundUp(elemsPerBlock, blockSize);
      assigned = std::max(
          1,
          static_cast<int32_t>(
              (numElements[i] + alignedElems - 1) / alignedElems));
    }
    numBlocksPerOp[i] = assigned;
    totalAssigned += assigned;
  }

  // Fill blocks and opIndices.
  blocks.resize(totalAssigned);
  opIndices.resize(totalAssigned);
  int32_t blockIdx = 0;
  for (size_t i = 0; i < ops.size(); ++i) {
    auto opCode = ops[i].op()->opCodes()[0];
    auto nBlocks = numBlocksPerOp[i];
    auto elements = numElements[i];
    // Align elements per block to a multiple of blockSize so all but the last
    // block process a blockSize-aligned number of elements.
    auto rawElemsPerBlock = (elements + nBlocks - 1) / nBlocks;
    auto alignedElems = roundUp(rawElemsPerBlock, blockSize);
    for (int32_t b = 0; b < nBlocks; ++b) {
      auto& info = blocks[blockIdx];
      info.op = opCode;
      info.blockInOp = b;
      info.numBlocksInOp = nBlocks;
      auto startRow = static_cast<int64_t>(b) * alignedElems;
      auto endRow = std::min(startRow + alignedElems, elements);
      info.rowsForBlock = static_cast<int32_t>(endRow - startRow);
      info.rowIdx = 0;
      info.params = nullptr;
      info.returnData = nullptr;
      info.debug = nullptr;
      opIndices[blockIdx] = static_cast<int32_t>(i);
      ++blockIdx;
    }
  }

  nextOps.clear();
}

// --- KernelOperation ---

KernelOperation::KernelOperation(const Subgraph& sg, int32_t opCode)
    : opCode_{opCode},
      expr_{sg.root},
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

  // Collect and assign offsets to outputs.
  std::vector<const nativert::Value*> outputValues;
  setOutputs(sg.root, inputs_, outputValues, outputReserve_);

  // Compute unit cost before adding outputs to inputs_: 10 per input/output +
  // sum of node costs from registry.
  std::unordered_set<const nativert::Node*> costVisited;
  unitCost_ = 10.0f * (numInputs_ + outputValues.size());
  unitCost_ += sumNodeCosts(sg.root, inputs_, costVisited);

  for (auto* value : outputValues) {
    inputs_.insert(value);
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
  std::unordered_set<const nativert::Node*> visited;
  collectAttrOffsets(sg.root, inputs_, visited, offset, attrOffsets_);

  // For all-elementwise kernel ops, set numElements_ to get the numel of the
  // output tensor.
  std::unordered_set<const nativert::Node*> ewVisited;
  if (!outputValues.empty() && isAllElementwise(sg.root, inputs_, ewVisited)) {
    auto outputId = outputValues[0]->id();
    numElements_ = [outputId](
                       nativert::ExecutionFrame& frame,
                       const FormalToActual& map) -> int64_t {
      auto it = map.find(outputId);
      TORCH_CHECK(
          it != map.end(),
          "Output value %v",
          outputId,
          " not found in FormalToActual map");
      return frame.getTensor(it->second).numel();
    };
  }
}

int32_t KernelOperation::paramOffset(const nativert::Value* value) const {
  auto it = paramOffsets_.find(value);
  TORCH_CHECK(it != paramOffsets_.end(), "Value not found in paramOffsets");
  return it->second;
}

int32_t KernelOperation::attrOffset(
    const nativert::Node* node,
    std::string_view attr) const {
  auto it = attrOffsets_.find({node, std::string(attr)});
  TORCH_CHECK(
      it != attrOffsets_.end(),
      "Attribute '",
      attr,
      "' not found for node ",
      node->toString());
  return it->second;
}

int64_t KernelOperation::numElements(
    nativert::ExecutionFrame& frame,
    const FormalToActual& map) const {
  TORCH_CHECK(
      numElements_ != nullptr, "numElements_ not set for KernelOperation");
  return numElements_(frame, map);
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

void KernelOperation::setOutputs(
    const nativert::Node* node,
    const std::unordered_set<const nativert::Value*>& subgraphInputs,
    std::vector<const nativert::Value*>& outputValues,
    std::vector<OutputReserveFunc>& outputReserves) {
  std::unordered_set<const nativert::Node*> visited;
  if (isAllElementwise(node, subgraphInputs, visited)) {
    outputValues.push_back(node->outputs()[0]);
    std::unordered_set<const nativert::Value*> seen;
    std::vector<const nativert::Value*> deduppedInputs;
    collectDeduppedInputs(node, subgraphInputs, seen, deduppedInputs);
    outputReserves.push_back([deduppedInputs = std::move(deduppedInputs)](
                                 const nativert::Node* n,
                                 nativert::ExecutionFrame& frame,
                                 FormalToActual map) {
      return elementwiseOutputShape(deduppedInputs, n, frame, map);
    });
    return;
  }

  for (auto& input : node->inputs()) {
    auto* value = input.value;
    if (subgraphInputs.count(value)) {
      continue;
    }
    auto* producer = value->producer();
    if (producer) {
      setOutputs(producer, subgraphInputs, outputValues, outputReserves);
    }
  }

  auto* meta = Registry::metadata(node->target());
  if (!meta) {
    return;
  }
  const auto& outputs = node->outputs();
  for (size_t i = 0; i < meta->returnMeta.size() && i < outputs.size(); ++i) {
    if (!meta->returnMeta[i].isRegister) {
      outputValues.push_back(outputs[i]);
      auto reserveShape = meta->returnMeta[i].reserveShape;
      outputReserves.push_back(
          [node, reserveShape](
              const nativert::Node* /*unused*/,
              nativert::ExecutionFrame& frame,
              FormalToActual map) { return reserveShape(node, frame, map); });
    }
  }
}

std::string KernelOperation::toString() const {
  std::stringstream ss;
  std::unordered_set<const nativert::Value*> inputSet(
      orderedInputs_.begin(), orderedInputs_.begin() + numInputs_);
  std::unordered_set<const nativert::Value*> outputSet(
      orderedInputs_.begin() + numInputs_, orderedInputs_.end());

  // Print output-producing nodes as separate assignment lines, dependencies
  // first.
  std::unordered_set<const nativert::Node*> printed;
  std::function<void(const nativert::Node*)> printNode =
      [&](const nativert::Node* node) {
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
        std::vector<const nativert::Value*> nodeOutputs;
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

  for (auto code : opCode_) {
    ss << "opCode = " << code << ":\n";
  }
  ss << text_ << "\n";

  return ss.str();
}

// --- OpInvocation ---

OpInvocation::OpInvocation(
    KernelOperation* op,
    const Subgraph& sg,
    std::vector<const c10::IValue*> constants,
    int32_t startOffset)
    : op_{op},
      values_{op->orderedInputs()},
      constants_{std::move(constants)},
      paramOffset_{startOffset},
      paramSize_{
          op->constantAreaOffset() +
          static_cast<int32_t>(constants_.size()) * 8} {
  Subgraph formalSg;
  formalSg.root = op->expr();
  formalSg.inputs.assign(values_.begin(), values_.begin() + op->numInputs());
  makeBindings(formalSg, sg);
}

void OpInvocation::makeBindings(
    const Subgraph& formalSg,
    const Subgraph& actualSg) {
  const auto& orderedInputs = op_->orderedInputs();
  auto numInputs = op_->numInputs();
  std::unordered_set<const nativert::Value*> formalInputs(
      formalSg.inputs.begin(), formalSg.inputs.end());
  // Map formal input values to actual input values.
  for (size_t i = 0; i < formalSg.inputs.size() && i < actualSg.inputs.size();
       ++i) {
    bindings_[formalSg.inputs[i]->id()] = actualSg.inputs[i]->id();
  }
  std::unordered_set<const nativert::Value*> outputValues(
      orderedInputs.begin() + numInputs, orderedInputs.end());
  if (outputValues.empty()) {
    return;
  }
  std::unordered_set<const nativert::Node*> visited;
  makeBindingsImpl(
      formalSg.root,
      actualSg.root,
      formalInputs,
      outputValues,
      visited,
      bindings_);
}

int64_t OpInvocation::allocateOutput(
    nativert::ExecutionFrame& frame,
    const ValueTypes& types) {
  const auto& reserves = op_->outputReserves();
  const auto& orderedInputs = op_->orderedInputs();
  auto numInputs = op_->numInputs();
  for (size_t i = 0; i < reserves.size(); ++i) {
    auto* formalValue = orderedInputs[numInputs + i];
    auto shapes = reserves[i](nullptr, frame, bindings_);
    TORCH_CHECK(
        !shapes.empty(),
        "OutputReserveFunc returned empty shapes for output ",
        i);
    auto* meta = types.types[formalValue->id()];
    TORCH_CHECK(
        meta != nullptr,
        "No TensorMeta for output value %v",
        formalValue->id());
    auto dtype = meta->dtype();
    auto it = bindings_.find(formalValue->id());
    TORCH_CHECK(
        it != bindings_.end(),
        "Output value %v",
        formalValue->id(),
        " not found in bindings");
    auto actualId = it->second;
    std::vector<int64_t> dims(shapes[0].begin(), shapes[0].end());
    const auto& existing = frame.getIValue(actualId);
    if (existing.isTensor() && existing.toTensor().is_cuda()) {
      auto tensor = existing.toTensor();
      tensor.resize_(dims);
    } else {
      auto tensor =
          at::empty(dims, at::TensorOptions().dtype(dtype).device(at::kCUDA));
      frame.setIValue(actualId, std::move(tensor));
    }
  }
  return op_->numElements(frame, bindings_);
}

void OpInvocation::fillParams(
    nativert::ExecutionFrame& frame,
    uint8_t* paramBase) {
  const auto& orderedInputs = op_->orderedInputs();
  auto numInputs = op_->numInputs();

  // Fill input params.
  for (int32_t i = 0; i < numInputs; ++i) {
    auto* formalValue = orderedInputs[i];
    auto offset = op_->paramOffset(formalValue);
    auto* dest = paramBase + offset;
    auto it = bindings_.find(formalValue->id());
    auto actualId = it != bindings_.end() ? it->second : formalValue->id();
    const auto& ivalue = frame.getIValue(actualId);
    if (ivalue.isTensor()) {
      fillTensorParam(ivalue.toTensor(), dest);
    } else {
      fillScalarParam(ivalue, dest);
    }
  }

  // Fill output params.
  for (size_t i = numInputs; i < orderedInputs.size(); ++i) {
    auto* formalValue = orderedInputs[i];
    auto offset = op_->paramOffset(formalValue);
    auto* dest = paramBase + offset;
    auto it = bindings_.find(formalValue->id());
    TORCH_CHECK(
        it != bindings_.end(),
        "Output value %v",
        formalValue->id(),
        " not found in bindings");
    auto actualId = it->second;
    const auto& ivalue = frame.getIValue(actualId);
    TORCH_CHECK(ivalue.isTensor(), "Expected tensor for output param");
    fillTensorParam(ivalue.toTensor(), dest);
  }

  // Fill constant params.
  auto constantOffset = op_->constantAreaOffset();
  for (const auto* c : constants_) {
    auto* dest = paramBase + constantOffset;
    fillScalarParam(*c, dest);
    constantOffset += 8;
  }
}

std::string OpInvocation::toString() const {
  std::stringstream ss;
  for (auto code : op_->opCodes()) {
    ss << "opCode=" << code;
  }
  ss << " inputs=(";
  for (size_t i = 0; i < values_.size(); ++i) {
    if (i > 0) {
      ss << ", ";
    }
    ss << "%v" << values_[i]->id();
  }
  ss << ")";
  // Print outputs from bindings.
  const auto& orderedInputs = op_->orderedInputs();
  auto numInputs = op_->numInputs();
  if (static_cast<size_t>(numInputs) < orderedInputs.size()) {
    ss << " outputs=(";
    bool first = true;
    for (size_t i = numInputs; i < orderedInputs.size(); ++i) {
      if (!first) {
        ss << ", ";
      }
      first = false;
      auto it = bindings_.find(orderedInputs[i]->id());
      if (it != bindings_.end()) {
        ss << "%v" << it->second;
      } else {
        ss << "%v" << orderedInputs[i]->id();
      }
    }
    ss << ")";
  }
  if (!constants_.empty()) {
    ss << " constants=(";
    for (size_t i = 0; i < constants_.size(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << ivalueToString(*constants_[i]);
    }
    ss << ")";
  }
  ss << "\n";
  return ss.str();
}

// --- CompositeKernel ---

CompositeKernel::CompositeKernel(
    std::vector<std::unique_ptr<KernelOperation>>&& ops,
    const std::unordered_set<std::string>& includes)
    : ops_(std::move(ops)) {
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
  std::unordered_set<std::string> emittedDecls;
  for (const auto& op : ops_) {
    for (const auto& decl : op->sharedDeclarations()) {
      if (emittedDecls.insert(decl).second) {
        ss << decl;
      }
    }
  }
  ss << "  switch (blockInfo.op) {\n";
  for (const auto& op : ops_) {
    for (auto opCode : op->opCodes()) {
      ss << "    case " << opCode << ": {\n"
         << op->code() << "      break;\n"
         << "    }\n";
    }
  }
  ss << "  }\n"
     << "  __syncthreads();\n"
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

std::string CompositeKernel::toString() const {
  std::stringstream ss;
  for (const auto& op : ops_) {
    ss << op->toString();
  }
  auto info = kernelInfo();
  if (info.numRegs > 0) {
    ss << info.toString() << "\n";
  }
  return ss.str();
}

// --- CompositeInvocation ---

void CompositeInvocation::execute(ExecutionState& state) {
  auto& frame = state.frame;
  const auto& types = *state.valueTypes;

  // 1. Allocate outputs and collect numElements per op.
  std::vector<int64_t> numElements;
  numElements.reserve(ops.size());
  for (auto& op : ops) {
    numElements.push_back(op.allocateOutput(frame, types));
  }

  // 2. Build the grid.
  std::vector<BlockInfo> blocks;
  std::vector<int32_t> opIndices;
  std::vector<OpInvocation*> nextOps;
  makeGrid(ops, numElements, blocks, opIndices, nextOps);

  // 3. Compute total param bytes and track per-op param offsets relative to
  //    the start of the pinned buffer.
  auto numBlocks = blocks.size();
  auto blockInfoBytes = static_cast<int64_t>(numBlocks) * sizeof(BlockInfo);
  std::vector<int64_t> opParamOffsets(ops.size());
  int64_t paramCursor = blockInfoBytes;
  for (size_t i = 0; i < ops.size(); ++i) {
    opParamOffsets[i] = paramCursor;
    paramCursor += ops[i].paramSize();
  }
  auto totalPinnedBytes = paramCursor;

  // 4. Allocate pinned host buffer: BlockInfos + params.
  auto pinnedBuffer = state.pinnedArena->allocateBytes(totalPinnedBytes);
  auto* pinnedBase = pinnedBuffer->as<uint8_t>();

  // 5. Copy BlockInfos to beginning of pinned buffer.
  if (!blocks.empty()) {
    memcpy(pinnedBase, blocks.data(), blockInfoBytes);
  }

  // 6. Fill params from frame values after the BlockInfos.
  for (size_t opIdx = 0; opIdx < ops.size(); ++opIdx) {
    ops[opIdx].fillParams(frame, pinnedBase + opParamOffsets[opIdx]);
  }

  // 7. Allocate device memory.
  auto deviceBuffer = state.deviceArena->allocateBytes(totalPinnedBytes);
  auto* deviceBase = deviceBuffer->as<uint8_t>();

  // 8. Patch BlockInfo params pointers to point to the device-side param
  //    area for the corresponding op (will be valid after H2D copy).
  auto* pinnedBlocks = reinterpret_cast<BlockInfo*>(pinnedBase);
  for (size_t b = 0; b < numBlocks; ++b) {
    auto opIdx = opIndices[b];
    pinnedBlocks[b].params = deviceBase + opParamOffsets[opIdx];
  }

  // 9. Get a stream and enqueue H2D transfer.
  auto stream = state.streamPool->get();
  stream->hostToDeviceAsync(deviceBase, pinnedBase, totalPinnedBytes);

  // 10. Launch the composite kernel on the same stream.
  const int32_t blockSize = 256;
  TorchWaveParams params;
  params.info = reinterpret_cast<BlockInfo*>(deviceBase);
  void* args[] = {&params};
  kernel->launch(
      static_cast<int32_t>(numBlocks), blockSize, 0, stream.get(), args);

  // 11. Wait for the kernel to complete and return the stream to the pool.
  stream->wait();
  state.streamPool->put(std::move(stream));
}

std::string CompositeInvocation::toString() const {
  std::stringstream ss;
  ss << "CompositeKernel:\n" << kernel->toString();
  ss << "Invocations:\n";
  for (const auto& op : ops) {
    ss << "  " << op.toString();
  }
  return ss.str();
}

// --- CompiledNode ---

void CompiledNode::execute(ExecutionState& state) {
  for (auto& group : kernels_) {
    for (auto& invocation : group) {
      invocation->execute(state);
    }
  }
}

std::string CompiledNode::toString() const {
  std::stringstream ss;
  // Find the length of the longest inner vector.
  size_t maxWaves = 0;
  for (const auto& inner : kernels_) {
    maxWaves = std::max(maxWaves, inner.size());
  }

  for (size_t wave = 0; wave < maxWaves; ++wave) {
    if (wave > 0) {
      ss << "\n";
    }
    ss << "Wave " << wave << ":\n";
    for (size_t g = 0; g < kernels_.size(); ++g) {
      if (wave < kernels_[g].size()) {
        if (g > 0) {
          ss << "\n";
        }
        ss << "Group " << g << ":\n";
        ss << kernels_[g][wave]->toString();
      }
    }
  }
  return ss.str();
}

// --- WaveGraph ---

WaveGraph::WaveGraph(nativert::Graph& graph, const ValueTypes& types) {
  normalizeDefaults(graph);
  ParallelNodes parallelNodes;
  auto* lastProjectNode = parallelNodes.makeParallelNodes(graph);

  CompileCtx ctx(types);

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
}

std::string WaveGraph::toString() const {
  std::stringstream ss;
  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (i > 0) {
      ss << "\n";
    }
    ss << "Node " << i << ":\n";
    ss << nodes_[i]->toString();
  }
  return ss.str();
}

} // namespace torch::wave
