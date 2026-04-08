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
A * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/experimental/torchwave/Compile.h"
#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/Headers.h"
#include "velox/experimental/torchwave/Utils.h"

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <algorithm>
#include <deque>
#include <fstream>
#include <sstream>
#include <vector>

namespace torch::wave {

namespace {

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

} // namespace

void normalizeDefaults(nativert::Graph& graph) {
  for (auto& node : graph.nodes()) {
    const auto* schema = findSchema(node.target());
    if (!schema) {
      continue;
    }
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
      node.addAttribute(
          nativert::Attribute{
              std::string(schemaArg.name()),
              iValueToConstant(*schemaArg.default_value())});
    }
  }
}

std::atomic<int32_t> CompileCtx::kernelCounter_{0};

int32_t CompileCtx::nextKernelId() {
  return kernelCounter_++;
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
  forEachSortedAttribute(
      node,
      inputs,
      visited,
      [&](const nativert::Node*, const nativert::Attribute& attr) {
        storage.push_back(nativert::constantToIValue(attr.value));
      });
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

bool tensorMetaCompatible(
    const nativert::TensorMeta& l,
    const nativert::TensorMeta& r) {
  return l.dtype() == r.dtype() && l.layout() == r.layout() &&
      l.requires_grad() == r.requires_grad();
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
      if (!tensorMetaCompatible(
              *leftSg.inputTypes[leftPos], *rightSg.inputTypes[rightPos])) {
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

} // namespace

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
    elementwiseBody(
        root,
        *opStorage_.back(),
        inputs,
        tp + " result",
        "b" + std::to_string(id) + "[idx] = result;",
        false);
  } else {
    auto tp = cudaType(inputs[0]);
    elementwiseBody(
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
  code_ << "    Tensor* temp = " << param(leafInputs[0], *op)
        << ";\n    size = numEl(*temp);\n"
        << "    isFastPath0 |= isFastPathTensor(*temp);\n";
  if (leafInputs.size() > 1) {
    code_ << "    uint32_t size2;\n";
  }
  for (size_t valueIdx = 1; valueIdx < leafInputs.size(); ++valueIdx) {
    auto W = valueIdx / 32;
    auto B = valueIdx % 32;
    code_ << "    temp = " << param(leafInputs[valueIdx], *op) << ";\n"
          << "    size2 = numEl(*temp);\n"
          << "    isFastPath" << W << " |= isFastPathTensor(*temp) << " << B
          << ";\n"
          << "    if (size2 != size) {\n"
          << "      if (size2 > size) {\n";
    for (size_t I = 0; (I + 1) * 32 <= valueIdx; ++I) {
      code_ << "        isFastPath" << I << " = 0;\n";
    }
    code_ << "        isFastPath" << W << " &= ~((1 << " << B << ") - 1);\n"
          << "      size = size2;\n"
          << "      } else {\n"
          << "    isFastPath" << W << " &= ~(1 << " << B << ");\n"
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
  forEachSortedAttribute(
      node,
      inputs,
      visited,
      [&](const nativert::Node* n, const nativert::Attribute& attr) {
        auto off = op.attrOffset(n, attr.name);
        auto tp = cudaAttrType(attr.value);
        ss << "  " << tp << " attr" << off << " = *param<" << tp
           << ">(blockInfo, " << off << ");\n";
      });
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

void CompileCtx::elementwiseBody(
    const nativert::Node* node,
    KernelOperation& op,
    const std::vector<const nativert::Value*>& inputs,
    std::string resultName,
    std::string resultStmt,
    bool fullBlockResult) {
  auto leafInputs = elementwiseHead(node, &op);
  addInclude("velox/experimental/torchwave/Elementwise.cuh");

  // Assign tensor storage for each input to a register.
  for (size_t valueIdx = 0; valueIdx < inputs.size(); ++valueIdx) {
    auto tp = cudaType(inputs[valueIdx]);
    code_ << "  " << tp << "* b" << valueIdx << " = storage<" << tp << ">("
          << param(inputs[valueIdx], op) << ");\n";
  }

  // Declare attribute variables.
  code_ << declareAttributes(node, op, inputs);

  // Generate fastPath test for all leaf inputs.
  auto numLeafInputs = leafInputs.size();
  auto numFastPathVars = (numLeafInputs + 31) / 32;
  code_ << "  if (";
  for (size_t i = 0; i < numFastPathVars; ++i) {
    if (i > 0) {
      code_ << " && ";
    }
    uint32_t mask;
    if (i < numFastPathVars - 1) {
      mask = 0xffffffff;
    } else {
      auto bitsInLast = numLeafInputs - i * 32;
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

  code_
      << "  } else {\n"
      << "    printf(\"Unimplemented slow path %d isFastPath0=%u\\n\", __LINE__, isFastPath0);\n"
      << "    __trap();\n"
      << "  }\n";
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
  addInclude("velox/experimental/torchwave/Elementwise.cuh");
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
  return cudaTypeString(types_.types[value->id()]->dtype());
}

std::string CompileCtx::declare(c10::ScalarType scalarType) {
  auto tp = cudaTypeString(scalarType);
  auto name = "temp" + std::to_string(declareCounter_++);
  declarations_ << "  " << tp << " " << name << ";\n";
  return name;
}

std::string CompileCtx::param(
    const nativert::Value* value,
    const KernelOperation& op) const {
  auto off = op.paramOffset(value);
  if (value->type().kind() == nativert::Type::Kind::Tensor) {
    return fmt::format("param<Tensor>(blockInfo, {})", off);
  }
  return fmt::format("param<{}>(blockInfo, {})", cudaType(value), off);
}

std::unique_ptr<CompiledNode> CompileCtx::compileNode(ProjectNode& project) {
  inputs_ = &project.inputs();
  auto& nodes = project.nodes();
  bool allInput = std::all_of(nodes.begin(), nodes.end(), [](const auto* node) {
    return node->target() == "prim.Input";
  });
  if (allInput) {
    return nullptr;
  }
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
      std::make_unique<CompositeKernel>(std::move(opStorage_), includes_);
  auto invocation = std::make_unique<CompositeInvocation>();
  invocation->kernel = std::move(compositeKernel);
  invocation->ops = std::move(ops_);
  invocation->ivalueStorage = std::move(ivalueStorage_);

  std::vector<std::unique_ptr<CompositeInvocation>> inner;
  inner.push_back(std::move(invocation));
  std::vector<std::vector<std::unique_ptr<CompositeInvocation>>> outer;
  outer.push_back(std::move(inner));

  return std::make_unique<CompiledNode>(std::move(outer));
}

} // namespace torch::wave
