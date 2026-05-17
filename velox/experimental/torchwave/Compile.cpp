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

#include "velox/experimental/torchwave/Compile.h"
#include "velox/experimental/torchwave/Executor.h"

#include <ATen/ATen.h>
#include <algorithm>
#include <deque>
#include <fstream>
#include <sstream>
#include <vector>

namespace torch::wave {

std::atomic<int32_t> CompileCtx::kernelCounter_{0};

int32_t CompileCtx::nextKernelId() {
  return kernelCounter_++;
}

CompositeKernel::CompositeKernel(
    std::vector<std::unique_ptr<KernelOperation>>&& ops)
    : ops_(std::move(ops)) {
  auto kernelId = CompileCtx::nextKernelId();
  auto entryPoint = "torchwave" + std::to_string(kernelId);

  std::stringstream ss;
  ss << "#include \"velox/experimental/torchwave/Core.cuh\"\n\n"
     << "namespace torch::wave {\n\n"
     << "__global__ void " << entryPoint << "(TorchWaveParams params) {\n"
     << "  ENTRY;\n"
     << "  switch (blockInfo.op) {\n";
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

template <typename Func>
bool CompileCtx::allReachable(
    const nativert::Node& node,
    const NodeSet& placed,
    Func&& predicate,
    NodeSet& visited) const {
  if (!visited.insert(&node).second) {
    return true;
  }
  auto* meta = Registry::metadata(node.target());
  if (!meta || !predicate(*meta)) {
    return false;
  }
  for (auto& input : node.inputs()) {
    auto* producer = input.value->producer();
    if (!producer || (inputs_ && inputs_->count(producer)) ||
        placed.count(producer)) {
      continue;
    }
    if (!allReachable(*producer, placed, predicate, visited)) {
      return false;
    }
  }
  return true;
}

template <typename Func>
bool CompileCtx::anyReachable(
    const nativert::Node& node,
    const NodeSet& placed,
    Func&& predicate,
    NodeSet& visited) const {
  if (!visited.insert(&node).second) {
    return false;
  }
  auto* meta = Registry::metadata(node.target());
  if (meta && predicate(*meta)) {
    return true;
  }
  for (auto& input : node.inputs()) {
    auto* producer = input.value->producer();
    if (!producer || (inputs_ && inputs_->count(producer)) ||
        placed.count(producer)) {
      continue;
    }
    if (anyReachable(*producer, placed, predicate, visited)) {
      return true;
    }
  }
  return false;
}

namespace {

void extractSubgraphInputs(
    const nativert::Node* node,
    const CompileCtx::NodeSet& inputs,
    const CompileCtx::NodeSet& placed,
    std::unordered_set<const nativert::Value*>& seen,
    std::vector<const nativert::Value*>& result) {
  for (auto& input : node->inputs()) {
    auto* value = input.value;
    auto* producer = value->producer();
    if (!producer || inputs.count(producer) || placed.count(producer)) {
      if (seen.insert(value).second) {
        result.push_back(value);
      }
    } else {
      extractSubgraphInputs(producer, inputs, placed, seen, result);
    }
  }
}

void listConstantsImpl(
    const nativert::Node* node,
    const std::unordered_set<const nativert::Value*>& inputs,
    std::unordered_set<const nativert::Node*>& visited,
    std::deque<c10::IValue>& storage) {
  if (!visited.insert(node).second) {
    return;
  }
  for (const auto& input : node->inputs()) {
    if (inputs.count(input.value)) {
      continue;
    }
    auto* producer = input.value->producer();
    if (producer) {
      listConstantsImpl(producer, inputs, visited, storage);
    }
  }
  const auto& attrs = node->attributes();
  if (attrs.empty()) {
    return;
  }
  std::vector<const nativert::Attribute*> sorted;
  sorted.reserve(attrs.size());
  for (const auto& attr : attrs) {
    sorted.push_back(&attr);
  }
  std::sort(sorted.begin(), sorted.end(), [](const auto* a, const auto* b) {
    return a->name < b->name;
  });
  for (const auto* attr : sorted) {
    storage.push_back(nativert::constantToIValue(attr->value));
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
  if (!visited.insert(node).second) {
    return;
  }
  for (const auto& input : node->inputs()) {
    if (inputs.count(input.value)) {
      continue;
    }
    auto* producer = input.value->producer();
    if (producer) {
      collectAttrOffsets(producer, inputs, visited, offset, attrOffsets);
    }
  }
  const auto& attrs = node->attributes();
  if (attrs.empty()) {
    return;
  }
  std::vector<const nativert::Attribute*> sorted;
  sorted.reserve(attrs.size());
  for (const auto& attr : attrs) {
    sorted.push_back(&attr);
  }
  std::sort(sorted.begin(), sorted.end(), [](const auto* a, const auto* b) {
    return a->name < b->name;
  });
  for (const auto* attr : sorted) {
    attrOffsets[{node, attr->name}] = offset;
    offset += 8;
  }
}

std::vector<const c10::IValue*> listConstants(
    const Subgraph& sg,
    std::deque<c10::IValue>& storage) {
  std::unordered_set<const nativert::Value*> inputSet(
      sg.inputs.begin(), sg.inputs.end());
  std::unordered_set<const nativert::Node*> visited;
  auto startSize = storage.size();
  listConstantsImpl(sg.root, inputSet, visited, storage);
  std::vector<const c10::IValue*> result;
  for (auto it = storage.begin() + startSize; it != storage.end(); ++it) {
    result.push_back(&*it);
  }
  return result;
}

bool subgraphNodesMatch(
    const nativert::Node* left,
    const nativert::Node* right,
    const std::unordered_set<const nativert::Value*>& leftInputs,
    const std::unordered_set<const nativert::Value*>& rightInputs,
    const Subgraph& leftSg,
    const Subgraph& rightSg) {
  if (left->target() != right->target()) {
    return false;
  }
  auto& li = left->inputs();
  auto& ri = right->inputs();
  if (li.size() != ri.size()) {
    return false;
  }
  for (size_t i = 0; i < li.size(); ++i) {
    bool leftIsInput = leftInputs.count(li[i].value);
    bool rightIsInput = rightInputs.count(ri[i].value);
    if (leftIsInput != rightIsInput) {
      return false;
    }
    if (leftIsInput) {
      // Both are subgraph inputs. Must be at the same position and same type.
      auto leftIt =
          std::find(leftSg.inputs.begin(), leftSg.inputs.end(), li[i].value);
      auto rightIt =
          std::find(rightSg.inputs.begin(), rightSg.inputs.end(), ri[i].value);
      auto leftPos = leftIt - leftSg.inputs.begin();
      auto rightPos = rightIt - rightSg.inputs.begin();
      if (leftPos != rightPos) {
        return false;
      }
      if (leftSg.inputTypes[leftPos] != rightSg.inputTypes[rightPos]) {
        return false;
      }
    } else {
      auto* lp = li[i].value->producer();
      auto* rp = ri[i].value->producer();
      if (!lp || !rp) {
        if (lp != rp) {
          return false;
        }
        continue;
      }
      if (!subgraphNodesMatch(
              lp, rp, leftInputs, rightInputs, leftSg, rightSg)) {
        return false;
      }
    }
  }
  return true;
}

void hashSubgraphNode(
    const nativert::Node* node,
    const std::unordered_set<const nativert::Value*>& inputs,
    size_t& hash) {
  auto h = std::hash<std::string_view>{}(node->target());
  hash ^= h + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  for (auto& input : node->inputs()) {
    if (inputs.count(input.value)) {
      continue;
    }
    auto* producer = input.value->producer();
    if (producer) {
      hashSubgraphNode(producer, inputs, hash);
    }
  }
}

int32_t fillParamOffsets(
    const std::vector<const nativert::Value*>& values,
    const std::vector<const c10::IValue*>& constants,
    int32_t startOffset,
    std::vector<int32_t>& offsets) {
  offsets.reserve(values.size() + constants.size());
  int32_t offset = startOffset;
  for (const auto* value : values) {
    offsets.push_back(offset);
    if (value->type().kind() == nativert::Type::Kind::Tensor) {
      offset += sizeof(Tensor);
    } else {
      offset += 8;
    }
  }
  for (size_t i = 0; i < constants.size(); ++i) {
    offsets.push_back(offset);
    offset += 8;
  }
  return offset - startOffset;
}

} // namespace

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
      node->target());
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

} // namespace

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

namespace {

/// Prints the expression rooted at 'node'. Inputs in 'inputSet' and values
/// without a producer print as %v<id>. Nodes with outputs in 'outputSet' also
/// print as %v<id> (they get their own assignment line).
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

} // namespace

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

namespace {

std::string ivalueToString(const c10::IValue& value) {
  if (value.isInt()) {
    return std::to_string(value.toInt());
  }
  if (value.isDouble()) {
    return std::to_string(value.toDouble());
  }
  if (value.isBool()) {
    return value.toBool() ? "true" : "false";
  }
  if (value.isString()) {
    return "\"" + value.toStringRef() + "\"";
  }
  if (value.isNone()) {
    return "None";
  }
  return "<IValue>";
}

} // namespace

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
    auto tensor =
        at::empty(dims, at::TensorOptions().dtype(dtype).device(at::kCUDA));
    frame.setIValue(actualId, std::move(tensor));
  }
  return op_->numElements(frame, bindings_);
}

std::string CompositeKernel::toString() const {
  std::stringstream ss;
  for (const auto& op : ops_) {
    ss << op->toString();
  }
  return ss.str();
}

void CompositeKernel::launch(
    int32_t numBlocks,
    int32_t numThreads,
    int32_t sharedMemory,
    facebook::velox::wave::Stream* stream,
    void** args) {
  kernel_->launch(0, numBlocks, numThreads, sharedMemory, stream, args);
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

namespace {

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

} // namespace

KernelOperation::KernelOperation(
    const Subgraph& sg,
    int32_t opCode,
    bool useSingleBlock)
    : opCode_{opCode},
      expr_{sg.root},
      inputs_{sg.inputs.begin(), sg.inputs.end()},
      orderedInputs_{sg.inputs},
      numInputs_(sg.inputs.size()),
      isSingleBlock_{useSingleBlock} {
  int32_t offset = 0;
  for (const auto* value : orderedInputs_) {
    paramOffsets_[value] = offset;
    if (value->type().kind() == nativert::Type::Kind::Tensor) {
      offset += sizeof(Tensor);
    } else {
      offset += 8;
    }
  }
  std::unordered_set<const nativert::Node*> visited;
  collectAttrOffsets(sg.root, inputs_, visited, offset, attrOffsets_);

  std::vector<const nativert::Value*> outputValues;
  setOutputs(sg.root, inputs_, outputValues, outputReserve_);

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
}

OpInvocation::OpInvocation(
    KernelOperation* op,
    const Subgraph& sg,
    std::vector<const c10::IValue*> constants,
    int32_t startOffset)
    : op_{op},
      values_{sg.inputs},
      constants_{std::move(constants)},
      paramOffset_{startOffset} {
  paramSize_ = fillParamOffsets(values_, constants_, startOffset, offsets_);
  Subgraph formalSg;
  formalSg.root = op->expr();
  formalSg.inputs = op->orderedInputs();
  formalSg.inputs.resize(op->numInputs());
  makeBindings(formalSg, sg);
}

namespace {

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

} // namespace

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

Subgraph CompileCtx::extractSubgraph(
    const nativert::Node* node,
    const NodeSet& inputs,
    const NodeSet& placed) {
  Subgraph sg;
  sg.root = node;
  std::unordered_set<const nativert::Value*> seen;
  extractSubgraphInputs(node, inputs, placed, seen, sg.inputs);
  sg.inputTypes.reserve(sg.inputs.size());
  for (auto* value : sg.inputs) {
    sg.inputTypes.push_back(types_.types[value->id()]);
  }
  return sg;
}

bool subgraphsMatch(const Subgraph& left, const Subgraph& right) {
  if (left.inputs.size() != right.inputs.size()) {
    return false;
  }
  if (left.root->outputs().size() != right.root->outputs().size()) {
    return false;
  }
  std::unordered_set<const nativert::Value*> leftInputs(
      left.inputs.begin(), left.inputs.end());
  std::unordered_set<const nativert::Value*> rightInputs(
      right.inputs.begin(), right.inputs.end());
  return subgraphNodesMatch(
      left.root, right.root, leftInputs, rightInputs, left, right);
}

size_t SubgraphHash::operator()(const Subgraph& sg) const {
  std::unordered_set<const nativert::Value*> inputSet(
      sg.inputs.begin(), sg.inputs.end());
  size_t hash = 0;
  hashSubgraphNode(sg.root, inputSet, hash);
  return hash;
}

bool CompileCtx::isElementWise(
    const nativert::Node& node,
    const NodeSet& placed) const {
  NodeSet visited;
  return allReachable(
      node,
      placed,
      [](const Metadata& m) { return m.elementWise != nullptr; },
      visited);
}

bool CompileCtx::hasBarrier(const nativert::Node& node, const NodeSet& placed)
    const {
  NodeSet visited;
  return anyReachable(
      node, placed, [](const Metadata& m) { return m.hasBarrier; }, visited);
}

bool CompileCtx::isSingleBlock(
    const nativert::Node& node,
    const NodeSet& placed) const {
  NodeSet visited;
  return allReachable(
      node,
      placed,
      [](const Metadata& m) {
        return m.elementWise != nullptr || m.singleBlockVariant != nullptr;
      },
      visited);
}

bool CompileCtx::hasStandalone(
    const nativert::Node& node,
    const NodeSet& placed) const {
  NodeSet visited;
  return anyReachable(
      node, placed, [](const Metadata& m) { return m.isStandalone; }, visited);
}

bool CompileCtx::isMultikernel(
    const nativert::Node& node,
    const NodeSet& placed) const {
  NodeSet visited;
  return anyReachable(
      node,
      placed,
      [](const Metadata& m) { return m.nextKernel != nullptr; },
      visited);
}

KernelOperation* CompileCtx::makeKernelOperation(const Subgraph& sg) {
  auto opCode = nextOpCode_++;
  opStorage_.push_back(std::make_unique<KernelOperation>(sg, opCode));
  auto* op = opStorage_.back().get();
  if (isElementWise(*sg.root, placed_)) {
    ResultSpec resultSpec;
    resultSpec.value = sg.root->outputs()[0];
    generateElementwise(sg, resultSpec);
    std::stringstream combined;
    combined << declarations_.str() << code_.str();
    declarations_.str("");
    declarations_.clear();
    op->setCode(combined);
    code_.str("");
    code_.clear();
    return op;
  }
  return nullptr;
}

void CompileCtx::generateElementwise(
    const Subgraph& sg,
    const ResultSpec& resultSpec) {
  auto inputs = sg.inputs;
  auto* root = sg.root;
  if (resultSpec.value) {
    inputs.push_back(resultSpec.value);
    auto tp = cudaType(resultSpec.value);
    auto id = inputs.size() - 1;
    elementWiseBody(
        root,
        *opStorage_.back(),
        inputs,
        tp + " result",
        "r" + std::to_string(id) + "[idx] = result;",
        false);
  } else {
    auto tp = cudaType(inputs[0]);
    elementWiseBody(
        root,
        *opStorage_.back(),
        inputs,
        tp + " result",
        " " + resultSpec.variable + " = result;",
        false);
  }
}

std::vector<const nativert::Value*> CompileCtx::elementwiseHead(
    const nativert::Node* node,
    KernelOperation* op) {
  std::unordered_set<const nativert::Value*> seen;
  std::vector<const nativert::Value*> leafInputs;
  extractSubgraphInputs(node, *inputs_, placed_, seen, leafInputs);

  op->addSharedDeclaration("  __shared__ uint32_t size;\n");
  auto numFastPathVars = (leafInputs.size() + 31) / 32;
  for (size_t i = 0; i < numFastPathVars; ++i) {
    op->addSharedDeclaration(
        "  __shared__ uint32_t isFastPath" + std::to_string(i) + ";\n");
  }

  code_ << "  if (threadIdx.x == 0) {\n";
  for (size_t i = 0; i < numFastPathVars; ++i) {
    code_ << "    isFastPath" << i << " = 0;\n";
  }
  code_ << "    Tensor* temp = param(blockInfo, "
        << op->paramOffset(leafInputs[0]) << ");  size = numEl(temp);\n"
        << "    isFastPath0 |= isFastPath(temp);\n";
  if (leafInputs.size() > 1) {
    code_ << "    uint32_t size2;\n";
  }
  for (size_t valueIdx = 1; valueIdx < leafInputs.size(); ++valueIdx) {
    auto W = valueIdx / 32;
    auto B = valueIdx % 32;
    code_ << "    temp = param<tensor>(blockInfo, "
          << op->paramOffset(leafInputs[valueIdx]) << ");\n"
          << "    size2 = numEl(temp);\n"
          << "    isFastPath" << W << " |= isFastPath(temp) << " << B << ";\n"
          << "    if (size2 != size) {\n"
          << "      if (size2 > size) {\n";
    for (size_t I = 0; (I + 1) * 32 <= valueIdx; ++I) {
      code_ << "        isFastPath" << I << " = 0;\n";
    }
    code_ << "        isFastPath" << W << " &= ~((1 << " << B << ") - 1);\n"
          << "      size = size2;\n"
          << "      } else {\n"
          << "    fastPath" << W << " &= ~(1 << " << B << ");\n"
          << "}"
          << "    }\n";
  }
  code_ << "  }\n"
        << "  __syncthreads();\n";
  return leafInputs;
}

namespace {

std::string cudaAttrType(const nativert::Constant& c) {
  return std::visit(
      [](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, bool>) {
          return "bool";
        } else if constexpr (std::is_same_v<T, int64_t>) {
          return "int64_t";
        } else if constexpr (std::is_same_v<T, double>) {
          return "double";
        } else {
          TORCH_CHECK(false, "Unsupported attribute type for CUDA");
        }
      },
      c);
}

void declareAttributesImpl(
    const nativert::Node* node,
    const KernelOperation& op,
    const std::unordered_set<const nativert::Value*>& inputs,
    std::unordered_set<const nativert::Node*>& visited,
    std::stringstream& ss) {
  if (!visited.insert(node).second) {
    return;
  }
  for (const auto& input : node->inputs()) {
    if (inputs.count(input.value)) {
      continue;
    }
    auto* producer = input.value->producer();
    if (producer) {
      declareAttributesImpl(producer, op, inputs, visited, ss);
    }
  }
  const auto& attrs = node->attributes();
  if (attrs.empty()) {
    return;
  }
  std::vector<const nativert::Attribute*> sorted;
  sorted.reserve(attrs.size());
  for (const auto& attr : attrs) {
    sorted.push_back(&attr);
  }
  std::sort(sorted.begin(), sorted.end(), [](const auto* a, const auto* b) {
    return a->name < b->name;
  });
  for (const auto* attr : sorted) {
    auto off = op.attrOffset(node, attr->name);
    auto tp = cudaAttrType(attr->value);
    ss << "  " << tp << " attr" << off << " = getAttr<" << tp << ">(blockInfo, "
       << off << ");\n";
  }
}

} // namespace

std::string CompileCtx::declareAttributes(
    const nativert::Node* node,
    const KernelOperation& op,
    const std::vector<const nativert::Value*>& inputs) {
  std::unordered_set<const nativert::Value*> inputSet(
      inputs.begin(), inputs.end());
  std::unordered_set<const nativert::Node*> visited;
  std::stringstream ss;
  declareAttributesImpl(node, op, inputSet, visited, ss);
  return ss.str();
}

void CompileCtx::elementWiseBody(
    const nativert::Node* node,
    const KernelOperation& op,
    const std::vector<const nativert::Value*>& inputs,
    std::string resultName,
    std::string resultStmt,
    bool fullBlockResult) {
  // Assign tensor storage for each input to a register.
  for (size_t valueIdx = 0; valueIdx < inputs.size(); ++valueIdx) {
    auto tp = cudaType(inputs[valueIdx]);
    code_ << "  " << tp << " b" << valueIdx << " = storage<" << tp
          << ">(param(blockInfo, " << op.paramOffset(inputs[valueIdx])
          << "));\n";
  }

  // Declare attribute variables.
  code_ << declareAttributes(node, op, inputs);

  // Generate fastPath test for all inputs.
  auto numFastPathVars = (inputs.size() + 31) / 32;
  code_ << "  if (";
  for (size_t i = 0; i < numFastPathVars; ++i) {
    if (i > 0) {
      code_ << " && ";
    }
    uint32_t mask;
    if (i < numFastPathVars - 1) {
      mask = 0xffffffff;
    } else {
      auto bitsInLast = inputs.size() - i * 32;
      if (bitsInLast >= 32) {
        mask = 0xffffffff;
      } else {
        mask = (1u << bitsInLast) - 1;
      }
    }
    code_ << "isFastPath" << i << " == 0x" << std::hex << mask << std::dec;
  }
  code_ << ") {\n";

  auto expr = elementWiseExpr(node, op, inputs);

  // Fast path body.
  if (fullBlockResult) {
    code_
        << "    uint32_t rounded = roundUp(size, blockDim.x);\n"
        << "    for (uint32_t idx = blockInfo.blockInOp * blockDim.x + threadIdx.x; idx < rounded; idx += blockInfo.numBlocksInOp * blockDim.x) {\n"
        << "      " << resultName << ";\n"
        << "      if (idx < size) {\n"
        << "        result = " << expr << ";\n"
        << "      }\n"
        << "      " << resultStmt << "\n"
        << "    }\n";
  } else {
    code_
        << "    for (uint32_t idx = blockInfo.blockInOp * blockDim.x + threadIdx.x; idx < size; idx += blockInfo.numBlocksInOp * blockDim.x) {\n"
        << "      " << resultName << " = " << expr << ";\n"
        << "      " << resultStmt << "\n"
        << "    }\n";
  }

  code_ << "  } else {\n";

  // Slow path body (same structure, but would handle broadcast etc.).
  if (fullBlockResult) {
    code_
        << "    uint32_t rounded = roundUp(size, blockDim.x);\n"
        << "    for (uint32_t idx = blockInfo.blockInOp * blockDim.x + threadIdx.x; idx < rounded; idx += blockInfo.numBlocksInOp * blockDim.x) {\n"
        << "      " << resultName << ";\n"
        << "      if (idx < size) {\n"
        << "        result = " << expr << ";\n"
        << "      }\n"
        << "      " << resultStmt << "\n"
        << "    }\n";
  } else {
    code_
        << "    for (uint32_t idx = blockInfo.blockInOp * blockDim.x + threadIdx.x; idx < size; idx += blockInfo.numBlocksInOp * blockDim.x) {\n"
        << "      " << resultName << " = " << expr << ";\n"
        << "      " << resultStmt << "\n"
        << "    }\n";
  }

  code_ << "  }\n";
}

void CompileCtx::addInclude(std::string_view header) {
  includes_.insert(std::string(header));
}

namespace {

void elementWiseExprImpl(
    const nativert::Node* node,
    const std::unordered_set<const nativert::Value*>& inputSet,
    const std::vector<const nativert::Value*>& inputs,
    const KernelOperation& op,
    std::stringstream& ss) {
  auto* meta = Registry::metadata(node->target());
  TORCH_CHECK(
      meta && meta->elementWise, "Not an elementwise op: ", node->target());
  const auto& ew = *meta->elementWise;
  // Convert "--func" to "__func".
  std::string funcName = ew.functionName;
  TORCH_CHECK(
      funcName.size() >= 3 && funcName[0] == '-' && funcName[1] == '-',
      "Invalid elementwise function name: ",
      funcName);
  funcName[0] = '_';
  funcName[1] = '_';
  ss << funcName << "(";
  bool first = true;
  for (const auto& input : node->inputs()) {
    auto* value = input.value;
    if (!first) {
      ss << ", ";
    }
    first = false;
    if (inputSet.count(value)) {
      auto it = std::find(inputs.begin(), inputs.end(), value);
      TORCH_CHECK(it != inputs.end(), "Input value not found in inputs vector");
      auto valueIdx = it - inputs.begin();
      ss << "b" << valueIdx << "[idx]";
    } else {
      auto* producer = value->producer();
      TORCH_CHECK(producer, "Non-input value has no producer");
      elementWiseExprImpl(producer, inputSet, inputs, op, ss);
    }
  }
  for (const auto& attrName : ew.attributeArgs) {
    if (!first) {
      ss << ", ";
    }
    first = false;
    auto off = op.attrOffset(node, attrName);
    ss << "attr" << off;
  }
  ss << ")";
}

} // namespace

std::string CompileCtx::elementWiseExpr(
    const nativert::Node* node,
    const KernelOperation& op,
    const std::vector<const nativert::Value*>& inputs) {
  addInclude("Elementwise.cuh");
  std::unordered_set<const nativert::Value*> inputSet(
      inputs.begin(), inputs.end());
  std::stringstream ss;
  elementWiseExprImpl(node, inputSet, inputs, op, ss);
  return ss.str();
}

std::string CompileCtx::cudaType(const nativert::Value* value) const {
  TORCH_CHECK(
      value->id() < types_.types.size() && types_.types[value->id()],
      "No TensorMeta for value ",
      value->name());
  auto dtype = types_.types[value->id()]->dtype();
  switch (dtype) {
    case c10::ScalarType::Float:
      return "float";
    case c10::ScalarType::Double:
      return "double";
    case c10::ScalarType::Half:
      return "__half";
    case c10::ScalarType::BFloat16:
      return "__nv_bfloat16";
    case c10::ScalarType::Int:
      return "int32_t";
    case c10::ScalarType::Long:
      return "int64_t";
    case c10::ScalarType::Short:
      return "int16_t";
    case c10::ScalarType::Char:
      return "int8_t";
    case c10::ScalarType::Byte:
      return "uint8_t";
    case c10::ScalarType::Bool:
      return "bool";
    default:
      TORCH_CHECK(false, "Unsupported dtype ", c10::toString(dtype));
  }
}

namespace {
std::string cudaTypeString(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Float:
      return "float";
    case c10::ScalarType::Double:
      return "double";
    case c10::ScalarType::Half:
      return "__half";
    case c10::ScalarType::BFloat16:
      return "__nv_bfloat16";
    case c10::ScalarType::Int:
      return "int32_t";
    case c10::ScalarType::Long:
      return "int64_t";
    case c10::ScalarType::Short:
      return "int16_t";
    case c10::ScalarType::Char:
      return "int8_t";
    case c10::ScalarType::Byte:
      return "uint8_t";
    case c10::ScalarType::Bool:
      return "bool";
    default:
      TORCH_CHECK(false, "Unsupported dtype ", c10::toString(dtype));
  }
}
} // namespace

std::string CompileCtx::declare(c10::ScalarType scalarType) {
  auto tp = cudaTypeString(scalarType);
  auto name = "temp" + std::to_string(declareCounter_++);
  declarations_ << "  " << tp << " " << name << ";\n";
  return name;
}

std::unique_ptr<CompiledNode> CompileCtx::compileNode(ProjectNode& project) {
  inputs_ = &project.inputs();
  auto& nodes = project.nodes();
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto* node = nodes[i];
    auto sg = extractSubgraph(node, project.inputs(), placed_);
    auto it = kernelOps_.find(sg);
    if (it != kernelOps_.end()) {
      auto constants = listConstants(sg, ivalueStorage_);
      ops_.emplace_back(it->second, sg, std::move(constants), paramOffset_);
      paramOffset_ += ops_.back().paramSize();
    } else {
      auto* op = makeKernelOperation(sg);
      if (op) {
        kernelOps_[sg] = op;
        auto constants = listConstants(sg, ivalueStorage_);
        ops_.emplace_back(op, sg, std::move(constants), paramOffset_);
        paramOffset_ += ops_.back().paramSize();
      }
    }
  }
  auto compositeKernel =
      std::make_unique<CompositeKernel>(std::move(opStorage_));
  auto invocation = std::make_unique<CompositeInvocation>();
  invocation->kernel = std::move(compositeKernel);
  invocation->ops = std::move(ops_);

  std::vector<std::unique_ptr<CompositeInvocation>> inner;
  inner.push_back(std::move(invocation));
  std::vector<std::vector<std::unique_ptr<CompositeInvocation>>> outer;
  outer.push_back(std::move(inner));

  return std::make_unique<CompiledNode>(std::move(outer));
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

WaveGraph::WaveGraph(const nativert::Graph& graph, const ValueTypes& types) {
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
    nodes_.push_back(ctx.compileNode(*pn));
  }
}

} // namespace torch::wave
