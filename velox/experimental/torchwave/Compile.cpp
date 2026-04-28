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
#include "velox/experimental/torchwave/WaveConfig.h"

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <gflags/gflags.h>
#include <algorithm>
#include <deque>
#include <fstream>
#include <sstream>
#include <vector>

DEFINE_bool(elt_trace, false, "Trace elementwise kernel results");

namespace torch::wave {

void eltTrace(std::stringstream& ss, std::string_view printf) {
  if (FLAGS_elt_trace) {
    ss << "  if (threadIdx.x == 0) {printf(" << printf << ");}\n";
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
    NodeCP node,
    const CompileCtx::NodeSet& inputs,
    const CompileCtx::NodeSet& placed,
    std::unordered_set<ValueCP>& seen,
    std::vector<ValueCP>& result) {
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
    NodeCP node,
    const std::unordered_set<ValueCP>& inputs,
    std::unordered_set<NodeCP>& visited,
    std::deque<c10::IValue>& storage) {
  forEachSortedAttribute(
      node, inputs, visited, [&](NodeCP, const nativert::Attribute& attr) {
        storage.push_back(nativert::constantToIValue(attr.value));
      });
}

// Walk formal and actual trees in parallel, collecting the actual Value*
// for each formal Value* that is in formalOutputs.
void mapActualOutputs(
    NodeCP formalNode,
    NodeCP actualNode,
    const std::unordered_set<ValueCP>& formalInputs,
    const std::unordered_set<ValueCP>& formalOutputs,
    std::unordered_set<NodeCP>& visited,
    std::unordered_map<ValueCP, ValueCP>& formalToActual) {
  if (!visited.insert(formalNode).second) {
    return;
  }
  const auto& fo = formalNode->outputs();
  const auto& ao = actualNode->outputs();
  for (size_t i = 0; i < fo.size() && i < ao.size(); ++i) {
    if (formalOutputs.count(fo[i])) {
      formalToActual[fo[i]] = ao[i];
    }
  }
  const auto& fi = formalNode->inputs();
  const auto& ai = actualNode->inputs();
  for (size_t i = 0; i < fi.size() && i < ai.size(); ++i) {
    if (formalInputs.count(fi[i].value)) {
      continue;
    }
    auto* fp = fi[i].value->producer();
    auto* ap = ai[i].value->producer();
    if (fp && ap) {
      mapActualOutputs(
          fp, ap, formalInputs, formalOutputs, visited, formalToActual);
    }
  }
}

} // namespace

std::vector<const c10::IValue*> listConstants(
    const Subgraph& sg,
    std::deque<c10::IValue>& storage) {
  std::unordered_set<ValueCP> inputSet(sg.inputs.begin(), sg.inputs.end());
  std::unordered_set<NodeCP> visited;
  auto startSize = storage.size();
  listConstantsImpl(sg.root, inputSet, visited, storage);
  std::vector<const c10::IValue*> result;
  for (auto it = storage.begin() + startSize; it != storage.end(); ++it) {
    result.push_back(&*it);
  }
  return result;
}

namespace {

bool tensorMetaCompatible(
    const nativert::TensorMeta& l,
    const nativert::TensorMeta& r) {
  return l.dtype() == r.dtype() && l.layout() == r.layout() &&
      l.requires_grad() == r.requires_grad();
}

bool subgraphNodesMatch(
    NodeCP left,
    NodeCP right,
    const std::unordered_set<ValueCP>& leftInputs,
    const std::unordered_set<ValueCP>& rightInputs,
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
    NodeCP node,
    const std::unordered_set<ValueCP>& inputs,
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
    NodeCP node,
    const NodeSet& inputs,
    const NodeSet& placed) {
  Subgraph sg;
  sg.root = node;
  std::unordered_set<ValueCP> seen;
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
  std::unordered_set<ValueCP> leftInputs(
      left.inputs.begin(), left.inputs.end());
  std::unordered_set<ValueCP> rightInputs(
      right.inputs.begin(), right.inputs.end());
  return subgraphNodesMatch(
      left.root, right.root, leftInputs, rightInputs, left, right);
}

size_t SubgraphHash::operator()(const Subgraph& sg) const {
  std::unordered_set<ValueCP> inputSet(sg.inputs.begin(), sg.inputs.end());
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
      [](const Metadata& m) { return m.elementwise != nullptr; },
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
        return m.elementwise != nullptr || m.singleBlockIfFused;
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
      [](const Metadata& m) { return m.makeMultiKernelVariant != nullptr; },
      visited);
}

ProjectOperation* CompileCtx::makeProjectionOperation(const Subgraph& sg) {
  projectOpSubgraph_ = &sg;
  constantMap_ = sg.makeConstantIndices();
  opStorage_.push_back(std::make_unique<ProjectOperation>(sg, *this));
  auto* projectOp = opStorage_.back().get();

  // Check if any node in the subgraph has singleBlockIfFused.
  NodeSet visited;
  bool hasSingleBlock = anyReachable(
      *sg.root,
      placed_,
      [](const Metadata& m) { return m.singleBlockIfFused; },
      visited);
  if (hasSingleBlock) {
    setIsSingleBlock(true);
    projectOp->singleBlockGrid_ = makeGrid(sg.root);
    setIsSingleBlock(false);
  }
  projectOp->grid_ = makeGrid(sg.root);
  setGridChoice(projectOp);
  collectExtraValues(projectOp);
  projectOpSubgraph_ = nullptr;
  constantMap_.clear();
  return projectOp;
}

void CompileCtx::setGridChoice(ProjectOperation* projectOp) {
  if (projectOp->grid_.empty() || projectOp->singleBlockGrid_.empty()) {
    return;
  }
  auto& grid = projectOp->grid_;
  auto& sbGrid = projectOp->singleBlockGrid_;
  for (size_t i = 0; i < grid.size() && i < sbGrid.size(); ++i) {
    for (size_t j = 0; j < grid[i].size() && j < sbGrid[i].size(); ++j) {
      auto& gl = grid[i][j];
      auto& sl = sbGrid[i][j];
      if (gl.standalone && sl.standalone) {
        TORCH_CHECK(
            gl.standalone == sl.standalone,
            "Grid and singleBlockGrid have standalone launches with different nodes");
      } else if (gl.standalone || sl.standalone) {
        TORCH_CHECK(
            false,
            "Grid and singleBlockGrid mismatch: one has a standalone and the other has a kernel");
      } else if (gl.op && sl.op) {
        gl.op->setIsGridChoice(true);
        sl.op->setIsGridChoice(true);
        return;
      }
    }
  }
}

void CompileCtx::collectExtraValues(ProjectOperation* projectOp) {
  std::unordered_set<ValueCP> seen;
  auto addIfCreated = [&](ValueCP value) {
    if (waveGraph_.isCreatedValue(value) && seen.insert(value).second) {
      projectOp->extraValues_.push_back(value);
    }
  };
  auto scanGrid = [&](const LaunchGrid& grid) {
    for (const auto& step : grid) {
      for (const auto& launch : step) {
        if (launch.standalone) {
          for (const auto& input : launch.standalone->inputs()) {
            addIfCreated(input.value);
          }
          for (auto* output : launch.standalone->outputs()) {
            addIfCreated(output);
          }
        } else if (launch.op) {
          for (auto* value : launch.op->orderedInputs()) {
            addIfCreated(value);
          }
        }
      }
    }
  };
  scanGrid(projectOp->grid_);
  scanGrid(projectOp->singleBlockGrid_);
}

const std::vector<nativert::Value*>& CompileCtx::outputs(NodeCP node) const {
  auto it = originalNode_.find(node);
  if (it != originalNode_.end()) {
    return it->second->outputs();
  }
  return node->outputs();
}

NodeCP CompileCtx::getMultiBlockVariant(NodeCP node, WaveGraph* waveGraph) {
  auto* variant = waveGraph->multiKernelVariant(node);
  if (!variant) {
    return nullptr;
  }
  originalNode_[variant->root] = node;
  return variant->root;
}

void CompileCtx::newGrid() {
  placed_.clear();
  grid_.clear();
}

LaunchGrid CompileCtx::makeGrid(NodeCP node) {
  newGrid();
  if (!isSingleBlock_) {
    if (auto* variantRoot = getMultiBlockVariant(node, &waveGraph_)) {
      node = variantRoot;
    }
  }
  auto result = placeKernels(node, Context::kTop);
  if (result == Context::kFused) {
    pushdownFused(node);
  }
  return std::move(grid_);
}

Context CompileCtx::placeKernels(NodeCP node, Context context) {
  if (!isSingleBlock_) {
    if (auto* variantRoot = getMultiBlockVariant(node, &waveGraph_)) {
      node = variantRoot;
    }
  }
  auto* meta = Registry::metadata(node->target());
  auto thisContext =
      (!meta || meta->isStandalone || WaveConfig::get().allStandalone)
      ? Context::kStandalone
      : Context::kFused;
  std::vector<NodeCP> standaloneInputs;
  std::vector<NodeCP> fusedInputs;

  for (auto i = 0; i < node->inputs().size(); ++i) {
    if (meta && meta->inputFromPreviousKernel.has_value() &&
        i != meta->inputFromPreviousKernel.value()) {
      continue;
    }
    auto* producer = node->inputs()[i].value->producer();
    if (!producer || placed_.count(producer) ||
        (inputs_ && inputs_->count(producer))) {
      continue;
    }
    auto inputContext = placeKernels(producer, thisContext);
    if (inputContext == Context::kStandalone) {
      continue;
    } else if (inputContext == Context::kFusedBreak) {
      continue;
    } else {
      fusedInputs.push_back(producer);
    }
  }

  if (thisContext == Context::kStandalone) {
    for (auto* fused : fusedInputs) {
      pushdownFused(fused);
    }
    pushdownStandalone(node);
    return thisContext;
  }
  if (meta->isKernelBreak(isSingleBlock_)) {
    pushdownFused(node);
    return Context::kFusedBreak;
  }
  return Context::kFused;
}

void CompileCtx::pushdownStandalone(NodeCP node) {
  Launch launch;
  launch.standalone = node;
  placeKernelLaunch(std::move(launch));
  placed_.insert(node);
}

void CompileCtx::fillConstantIndices(const Subgraph& sg, Launch& launch) {
  std::unordered_set<ValueCP> sgInputSet(sg.inputs.begin(), sg.inputs.end());
  std::unordered_set<NodeCP> attrVisited;
  NodeCP prevNode = nullptr;
  int32_t attrCount = 0;
  forEachSortedAttribute(
      sg.root,
      sgInputSet,
      attrVisited,
      [&](NodeCP n, const nativert::Attribute&) {
        if (n != prevNode) {
          prevNode = n;
          attrCount = 0;
        }
        auto ordinal = projectOpSubgraph_->nodeOrdinal(n);
        auto mapIt = constantMap_.find(ordinal);
        TORCH_CHECK(
            mapIt != constantMap_.end(),
            "Node ordinal not found in constantMap");
        launch.constantIndices.push_back(mapIt->second + attrCount);
        ++attrCount;
      });
}

void CompileCtx::pushdownFused(NodeCP node) {
  auto sg = extractSubgraph(node, *inputs_, placed_);
  Launch launch;

  auto it = projectKernelOps_.find(sg);
  if (it != projectKernelOps_.end()) {
    auto* kernelOp = it->second;
    launch.op = kernelOp;

    const auto& formalSg = it->first;
    auto bindings = makeBindings(formalSg, sg, *kernelOp);

    // Translate orderedInputs to actual values using bindings.
    const auto& idToValue = waveGraph_.idToValue();
    const auto& orderedInputs = kernelOp->orderedInputs();
    launch.values.reserve(orderedInputs.size());
    for (auto* v : orderedInputs) {
      launch.values.push_back(idToValue.at(bindings.at(v->id())));
    }
  } else {
    auto kernelOp = generateFused(sg);
    kernelOpStorage_.push_back(std::move(kernelOp));
    launch.op = kernelOpStorage_.back().get();
    launch.values.assign(
        launch.op->orderedInputs().begin(), launch.op->orderedInputs().end());
  }

  fillConstantIndices(sg, launch);
  placeKernelLaunch(std::move(launch));
  placed_.insert(node);
}

std::unique_ptr<KernelOperation> CompileCtx::generateFused(const Subgraph& sg) {
  auto op = std::make_unique<KernelOperation>(sg, nextOpCode(), *this);
  generatingOp_ = op.get();
  generateFusedInner(sg);
  std::stringstream combined;
  combined << declarations_.str() << code_.str();
  declarations_.str("");
  declarations_.clear();
  op->setCode(combined);
  code_.str("");
  code_.clear();
  return op;
}

void CompileCtx::generateFusedInner(const Subgraph& sg) {
  std::vector<ResultSpec> resultSpecs;
  for (auto* output : outputs(sg.root)) {
    ResultSpec rs;
    rs.value = output;
    resultSpecs.push_back(rs);
  }
  fusedCode(sg.root, resultSpecs);
}

void CompileCtx::placeKernelLaunch(Launch launch) {
  int32_t latestLevel = -1;
  // Collect the input values of this launch.
  std::vector<ValueCP> inputValues;
  if (launch.standalone) {
    for (const auto& input : launch.standalone->inputs()) {
      inputValues.push_back(input.value);
    }
  } else if (launch.op) {
    auto numInputs = launch.op->numInputs();
    for (int32_t i = 0;
         i < numInputs && i < static_cast<int32_t>(launch.values.size());
         ++i) {
      inputValues.push_back(launch.values[i]);
    }
  }

  // Find the latest level in grid_ containing a Launch that produces
  // any of these input values.
  for (auto* value : inputValues) {
    for (int32_t level = 0; level < static_cast<int32_t>(grid_.size());
         ++level) {
      for (auto& existing : grid_[level]) {
        bool produces = false;
        if (existing.standalone) {
          for (auto* output : existing.standalone->outputs()) {
            if (output == value) {
              produces = true;
              break;
            }
          }
        } else if (existing.op) {
          const auto& ordered = existing.op->orderedInputs();
          for (int32_t i = existing.op->numInputs();
               i < static_cast<int32_t>(ordered.size());
               ++i) {
            if (ordered[i] == value) {
              produces = true;
              break;
            }
          }
        }
        if (produces) {
          latestLevel = std::max(latestLevel, level);
        }
      }
    }
  }

  int32_t targetLevel = latestLevel + 1;
  if (targetLevel >= static_cast<int32_t>(grid_.size())) {
    grid_.emplace_back();
  }
  grid_[targetLevel].push_back(std::move(launch));
}

void CompileCtx::collectSubgraphInputs(
    NodeCP node,
    const std::unordered_set<ValueCP>& sgInputs,
    std::unordered_set<ValueCP>& seen,
    std::vector<ValueCP>& result) const {
  for (auto& input : node->inputs()) {
    auto* value = input.value;
    auto* producer = value->producer();
    if (sgInputs.count(value) || (producer && placed_.count(producer))) {
      if (seen.insert(value).second) {
        result.push_back(value);
      }
    } else if (producer) {
      collectSubgraphInputs(producer, sgInputs, seen, result);
    }
  }
}

std::vector<ValueCP> CompileCtx::subgraphInputs(
    const std::vector<Subgraph>& subgraphs) const {
  std::unordered_set<ValueCP> seen;
  std::vector<ValueCP> result;
  for (auto& sg : subgraphs) {
    std::unordered_set<ValueCP> sgInputSet(sg.inputs.begin(), sg.inputs.end());
    collectSubgraphInputs(sg.root, sgInputSet, seen, result);
  }
  return result;
}

void CompileCtx::generateElementwiseBorderImpl(
    NodeCP node,
    const std::unordered_set<ValueCP>& opInputs,
    NodeSet& visited) {
  for (auto& input : node->inputs()) {
    auto* value = input.value;
    auto* producer = value->producer();
    if (!producer || placed_.count(producer) || opInputs.count(value)) {
      continue;
    }
    if (!visited.insert(producer).second) {
      continue;
    }
    auto* meta = Registry::metadata(producer->target());
    if (meta && meta->elementwise) {
      generateElementwiseBorderImpl(producer, opInputs, visited);
    } else {
      std::vector<ResultSpec> resultSpecs;
      for (auto* output : outputs(producer)) {
        ResultSpec rs;
        rs.value = output;
        resultSpecs.push_back(rs);
      }
      fusedCode(producer, resultSpecs);
    }
  }
}

void CompileCtx::generateElementwiseBorder(NodeCP node) {
  const auto& ordered = generatingOp_->orderedInputs();
  std::unordered_set<ValueCP> opInputs(
      ordered.begin(), ordered.begin() + generatingOp_->numInputs());
  NodeSet visited;
  generateElementwiseBorderImpl(node, opInputs, visited);
}

void CompileCtx::functionLoop(NodeCP node) {
  auto& op = *generatingOp_;
  auto* meta = Registry::metadata(node->target());
  TORCH_CHECK(meta, "No metadata for: ", node->target());

  // Find the size argument.
  TORCH_CHECK(
      !meta->sizeArgs.ordinal.empty(), "functionLoop requires sizeArgs");
  auto sizeArgIdx = meta->sizeArgs.ordinal[0];
  auto* sizeValue = node->inputs()[sizeArgIdx].value;

  // Add shared declaration for size.
  op.addSharedDeclaration("  __shared__ uint32_t size;\n");

  // Compute size from the tensor.
  code_ << "  size = numEl(*" << param(sizeValue, op) << ");\n";
  code_ << "  __syncthreads();\n";

  // Set up input and output specs for makeCall.
  const auto& inputs = node->inputs();
  std::vector<ResultSpec> inputSpecs;
  for (const auto& input : inputs) {
    ResultSpec rs;
    rs.value = input.value;
    inputSpecs.push_back(rs);
  }

  std::vector<ResultSpec> resultSpecs;
  for (auto* output : outputs(node)) {
    ResultSpec rs;
    rs.value = output;
    resultSpecs.push_back(rs);
  }

  // Generate the loop with the call inside.
  code_
      << "    for (uint32_t idx = blockInfo.blockInOp * blockDim.x + threadIdx.x; idx < size; idx += blockInfo.numBlocksInOp * blockDim.x) {\n";
  code_ << "  " << makeCall(node, inputSpecs, resultSpecs) << "\n";
  code_ << "    }\n";
}

void CompileCtx::fusedCode(NodeCP node, std::vector<ResultSpec>& resultSpecs) {
  if (placed_.count(node)) {
    return;
  }
  auto* meta = Registry::metadata(node->target());
  TORCH_CHECK(meta, "No metadata for node: ", node->target());

  if (meta->elementwise) {
    generateElementwiseBorder(node);
    Subgraph sg = extractSubgraph(node, *inputs_, placed_);
    generateElementwise({sg}, {resultSpecs[0]});
    return;
  }

  // Not elementwise - recurse on inputs backed by memory (outputs of
  // generatingOp_).
  const auto& ordered = generatingOp_->orderedInputs();
  std::unordered_set<ValueCP> memOutputs(ordered.begin(), ordered.end());

  for (const auto& input : node->inputs()) {
    auto* value = input.value;
    if (memOutputs.count(value)) {
      auto* producer = value->producer();
      if (producer && !placed_.count(producer)) {
        std::vector<ResultSpec> prodSpecs;
        for (auto* output : outputs(producer)) {
          ResultSpec rs;
          rs.value = output;
          prodSpecs.push_back(rs);
        }
        fusedCode(producer, prodSpecs);
      }
    }
  }

  const auto& inputs = node->inputs();

  if (!meta->hasRegisterInputs()) {
    // No register inputs - generate plain call.
    std::vector<ResultSpec> inputSpecs;
    for (const auto& input : inputs) {
      ResultSpec rs;
      rs.value = input.value;
      inputSpecs.push_back(rs);
    }
    code_ << "  " << makeCall(node, inputSpecs, resultSpecs) << "\n";
    placed_.insert(node);
    return;
  }

  // Has register inputs - check if all inputs are backed by memory.
  bool allInMemory = true;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (i < meta->argumentMeta.size() && meta->argumentMeta[i].isRegister) {
      if (!memOutputs.count(inputs[i].value)) {
        allInMemory = false;
        break;
      }
    }
  }

  if (allInMemory) {
    functionLoop(node);
    placed_.insert(node);
    return;
  }

  // Some register inputs need elementwise computation.
  std::vector<Subgraph> subgraphs;
  std::vector<ResultSpec> ewResultSpecs;
  std::vector<ResultSpec> callInputSpecs;

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto* value = inputs[i].value;
    bool isReg =
        i < meta->argumentMeta.size() && meta->argumentMeta[i].isRegister;
    if (isReg && !memOutputs.count(value)) {
      auto tempName = declareTemp(value);
      ResultSpec ewRs;
      ewRs.variable = tempName;
      ewResultSpecs.push_back(ewRs);

      ResultSpec callRs;
      callRs.variable = tempName;
      callInputSpecs.push_back(callRs);

      auto* producer = value->producer();
      TORCH_CHECK(producer, "Register input has no producer");
      if (!placed_.count(producer)) {
        generateElementwiseBorder(producer);
        subgraphs.push_back(extractSubgraph(producer, *inputs_, placed_));
      }
    } else {
      ResultSpec rs;
      rs.value = value;
      callInputSpecs.push_back(rs);
    }
  }

  auto callStmt = makeCall(node, callInputSpecs, resultSpecs);
  if (!subgraphs.empty()) {
    generateElementwise(subgraphs, ewResultSpecs, callStmt, true);
  }
  placed_.insert(node);
}

void CompileCtx::generateElementwise(
    const std::vector<Subgraph>& subgraphs,
    const std::vector<ResultSpec>& resultSpecs,
    std::string resultStmt,
    bool fullBlockResult) {
  auto& op = *generatingOp_;

  // Get unique leaf inputs by walking subgraph roots.
  auto leafInputs = subgraphInputs(subgraphs);

  // Build allInputs = leafInputs + result values for storage declarations.
  std::unordered_set<ValueCP> seen(leafInputs.begin(), leafInputs.end());
  auto allInputs = leafInputs;
  for (auto& rs : resultSpecs) {
    if (rs.value && seen.insert(rs.value).second) {
      allInputs.push_back(rs.value);
    }
  }

  code_ << "  {\n";
  // Generate shared declarations for size and fast path flags.
  op.addSharedDeclaration("  __shared__ uint32_t size;\n");
  auto numFastPathVars = (leafInputs.size() + 31) / 32;
  for (size_t i = 0; i < numFastPathVars; ++i) {
    op.addSharedDeclaration(
        "  __shared__ uint32_t isFastPath" + std::to_string(i) + ";\n");
  }

  // Generate the head: compute size and fast path flags from all leaf inputs.
  code_ << "  if (threadIdx.x == 0) {\n";
  for (size_t i = 0; i < numFastPathVars; ++i) {
    code_ << "    isFastPath" << i << " = 0;\n";
  }
  code_ << "    Tensor* temp = " << param(leafInputs[0], op)
        << ";\n    size = numEl(*temp);\n"
        << "    isFastPath0 |= isFastPathTensor(*temp);\n";
  if (leafInputs.size() > 1) {
    code_ << "    uint32_t size2;\n";
  }
  for (size_t valueIdx = 1; valueIdx < leafInputs.size(); ++valueIdx) {
    auto W = valueIdx / 32;
    auto B = valueIdx % 32;
    code_ << "    temp = " << param(leafInputs[valueIdx], op) << ";\n"
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

  addInclude("velox/experimental/torchwave/Elementwise.cuh");

  // Declare tensor storage for all unique inputs.
  for (size_t i = 0; i < allInputs.size(); ++i) {
    auto tp = cudaType(allInputs[i]);
    code_ << "  " << tp << "* b" << i << " = storage<" << tp << ">("
          << param(allInputs[i], op) << ");\n";
  }

  // Declare attributes for all subgraph roots.
  for (auto& sg : subgraphs) {
    code_ << declareAttributes(sg.root, op, allInputs);
  }

  // Generate fast path test for all leaf inputs.
  auto numLeafInputs = leafInputs.size();
  auto numFPVars = (numLeafInputs + 31) / 32;
  code_ << "  if (";
  for (size_t i = 0; i < numFPVars; ++i) {
    if (i > 0) {
      code_ << " && ";
    }
    uint32_t mask;
    if (i < numFPVars - 1) {
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

  // Fast path body: loop over all elements, computing all expressions.
  if (fullBlockResult) {
    code_
        << "    uint32_t rounded = roundUpPwr2(size, blockDim.x);\n"
        << "    for (uint32_t idx = blockInfo.blockInOp * blockDim.x + threadIdx.x; idx < rounded; idx += blockInfo.numBlocksInOp * blockDim.x) {\n";
    eltTrace(
        code_,
        "\"%d %d idx %d blockIdx %d\\n\", blockInfo.op, blockInfo.blockInOp, idx, blockIdx.x");
    for (size_t s = 0; s < subgraphs.size(); ++s) {
      auto tp = resultSpecs[s].value ? cudaType(resultSpecs[s].value)
                                     : cudaType(allInputs[0]);
      code_ << "      " << tp << " result" << s << ";\n";
    }
    code_ << "      if (idx < size) {\n";
    for (size_t s = 0; s < subgraphs.size(); ++s) {
      auto expr = elementwiseExpr(subgraphs[s].root, op, allInputs);
      code_ << "        result" << s << " = " << expr << ";\n";
      if (resultSpecs[s].value) {
        auto it =
            std::find(allInputs.begin(), allInputs.end(), resultSpecs[s].value);
        auto id = it - allInputs.begin();
        code_ << "        b" << id << "[idx] = result" << s << ";\n";
      } else {
        code_ << "        " << resultSpecs[s].variable << " = result" << s
              << ";\n";
      }
    }
    code_ << "      }\n";
    if (!resultStmt.empty()) {
      code_ << "      " << resultStmt << "\n";
    }
    code_ << "    }\n";
  } else {
    code_
        << "    for (uint32_t idx = blockInfo.blockInOp * blockDim.x + threadIdx.x; idx < size; idx += blockInfo.numBlocksInOp * blockDim.x) {\n";
    eltTrace(
        code_,
        "\"%d %d idx %d blockIdx %d\\n\", blockInfo.op, blockInfo.blockInOp, idx, blockIdx.x");
    for (size_t s = 0; s < subgraphs.size(); ++s) {
      auto expr = elementwiseExpr(subgraphs[s].root, op, allInputs);
      if (resultSpecs[s].value) {
        auto it =
            std::find(allInputs.begin(), allInputs.end(), resultSpecs[s].value);
        auto id = it - allInputs.begin();
        auto tp = cudaType(resultSpecs[s].value);
        code_ << "      " << tp << " result" << s << " = " << expr << ";\n"
              << "      b" << id << "[idx] = result" << s << ";\n";
      } else {
        auto tp = cudaType(allInputs[0]);
        code_ << "      " << tp << " result" << s << " = " << expr << ";\n"
              << "       " << resultSpecs[s].variable << " = result" << s
              << ";\n";
      }
    }
    if (!resultStmt.empty()) {
      code_ << "      " << resultStmt << "\n";
    }
    code_ << "    }\n";
  }

  code_
      << "  } else {\n"
      << "    printf(\"Unimplemented slow path %d isFastPath0=%u\\n\", __LINE__, isFastPath0);\n"
      << "    __trap();\n"
      << "  }\n";
  code_ << "  }\n";
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
    NodeCP node,
    const KernelOperation& op,
    const std::unordered_set<ValueCP>& inputs,
    std::unordered_set<NodeCP>& visited,
    std::stringstream& ss) {
  forEachSortedAttribute(
      node, inputs, visited, [&](NodeCP n, const nativert::Attribute& attr) {
        auto off = op.attrOffset(n, attr.name);
        auto tp = cudaAttrType(attr.value);
        ss << "  " << tp << " attr" << off << " = *param<" << tp
           << ">(blockInfo, " << off << ");\n";
      });
}

} // namespace

std::string CompileCtx::declareAttributes(
    NodeCP node,
    const KernelOperation& op,
    const std::vector<ValueCP>& inputs) {
  std::unordered_set<ValueCP> inputSet(inputs.begin(), inputs.end());
  std::unordered_set<NodeCP> visited;
  std::stringstream ss;
  declareAttributesImpl(node, op, inputSet, visited, ss);
  return ss.str();
}

std::string CompileCtx::makeCall(
    NodeCP node,
    std::vector<ResultSpec> inputs,
    std::vector<ResultSpec> outputs) {
  auto& op = *generatingOp_;
  auto* meta = Registry::metadata(node->target());
  TORCH_CHECK(meta, "No metadata for: ", node->target());

  if (!meta->headerFile.empty()) {
    addInclude(meta->headerFile);
  }

  std::stringstream ss;

  // Function name.
  ss << meta->deviceFunc;

  // Type template parameters from dtypes of node inputs at specified indices.
  if (meta->hasBlockSizeTemplateParam || !meta->typeTemplateParams.empty()) {
    const auto& nodeInputs = node->inputs();
    ss << "<";
    bool firstTp = true;
    if (meta->hasBlockSizeTemplateParam) {
      ss << WaveConfig::get().blockSize;
      firstTp = false;
    }
    for (size_t i = 0; i < meta->typeTemplateParams.size(); ++i) {
      if (!firstTp) {
        ss << ", ";
      }
      firstTp = false;
      auto idx = meta->typeTemplateParams[i];
      ss << cudaType(nodeInputs[idx].value);
    }
    ss << ">";
  }

  // Argument list.
  ss << "(";
  bool first = true;
  auto comma = [&] {
    if (!first) {
      ss << ", ";
    }
    first = false;
  };

  // Inputs.
  for (size_t i = 0; i < inputs.size(); ++i) {
    comma();
    if (inputs[i].value && inputs[i].variable.empty() &&
        i < meta->argumentMeta.size() && meta->argumentMeta[i].isRegister) {
      ss << makeElementRef(inputs[i].value, op);
    } else if (inputs[i].value) {
      ss << param(inputs[i].value, op);
    } else {
      ss << inputs[i].variable;
    }
  }

  // Outputs.
  for (size_t i = 0; i < outputs.size(); ++i) {
    comma();
    if (outputs[i].value && outputs[i].variable.empty() &&
        i < meta->returnMeta.size() && meta->returnMeta[i].isRegister) {
      ss << makeElementRef(outputs[i].value, op);
    } else if (outputs[i].value) {
      ss << param(outputs[i].value, op);
    } else {
      ss << outputs[i].variable;
    }
    if (outputs[i].value && i < meta->returnMeta.size() &&
        (meta->returnMeta[i].shapeSetOnDevice ||
         meta->returnMeta[i].neededOnHost)) {
      waveGraph_.addSyncableValueId(outputs[i].value->id());
    }
  }

  // Attributes in alphabetic order.
  const auto& attrs = node->attributes();
  if (!attrs.empty()) {
    std::vector<const nativert::Attribute*> sorted;
    sorted.reserve(attrs.size());
    for (const auto& attr : attrs) {
      sorted.push_back(&attr);
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto* a, const auto* b) {
      return a->name < b->name;
    });
    for (const auto* attr : sorted) {
      comma();
      auto off = op.attrOffset(node, attr->name);
      auto tp = cudaAttrType(attr->value);
      ss << "*param<" << tp << ">(blockInfo, " << off << ")";
    }
  }

  // Shared declarations: declare in the kernel and pass as arguments.
  for (const auto& [type, name] : meta->sharedDecls) {
    op.addSharedDeclaration("  __shared__ " + type + " " + name + ";\n");
    comma();
    ss << name;
  }

  // If not elementwise and has register inputs, pass idx and size before
  // blockInfo.
  if (!meta->elementwise) {
    bool hasRegister = false;
    for (size_t i = 0; i < meta->argumentMeta.size(); ++i) {
      if (meta->argumentMeta[i].isRegister) {
        hasRegister = true;
        break;
      }
    }
    if (hasRegister) {
      comma();
      ss << "idx, size";
    }
  }

  // blockInfo is always the last argument.
  comma();
  ss << "blockInfo);";

  return ss.str();
}

void CompileCtx::addInclude(std::string_view header) {
  includes_.insert(std::string(header));
}

void CompileCtx::elementwiseExprImpl(
    NodeCP node,
    const std::unordered_set<ValueCP>& inputSet,
    const std::vector<ValueCP>& inputs,
    const KernelOperation& op,
    std::stringstream& ss) {
  placed_.insert(node);
  auto* meta = Registry::metadata(node->target());
  TORCH_CHECK(
      meta && meta->elementwise, "Not an elementwise op: ", node->target());
  const auto& ew = *meta->elementwise;
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
      elementwiseExprImpl(producer, inputSet, inputs, op, ss);
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

std::string CompileCtx::elementwiseExpr(
    NodeCP node,
    const KernelOperation& op,
    const std::vector<ValueCP>& inputs) {
  addInclude("velox/experimental/torchwave/Elementwise.cuh");
  std::unordered_set<ValueCP> inputSet(inputs.begin(), inputs.end());
  std::stringstream ss;
  elementwiseExprImpl(node, inputSet, inputs, op, ss);
  return ss.str();
}

std::string CompileCtx::cudaType(ValueCP value) const {
  auto kind = value->type().kind();
  if (kind == nativert::Type::Kind::Tensor) {
    TORCH_CHECK(
        value->id() < types_.types.size() && types_.types[value->id()],
        "No TensorMeta for value ",
        value->name());
    return cudaTypeString(types_.types[value->id()]->dtype());
  }
  switch (kind) {
    case nativert::Type::Kind::SymInt:
      return "int32_t";
    case nativert::Type::Kind::SymFloat:
      return "float";
    case nativert::Type::Kind::SymBool:
      return "bool";
    default:
      TORCH_CHECK(false, "Unsupported type kind for value ", value->name());
  }
}

std::string CompileCtx::declare(c10::ScalarType scalarType) {
  auto tp = cudaTypeString(scalarType);
  auto name = "temp" + std::to_string(declareCounter_++);
  declarations_ << "  " << tp << " " << name << ";\n";
  return name;
}

std::string CompileCtx::declareTemp(ValueCP value) {
  auto tp = cudaType(value);
  auto name = "temp" + std::to_string(declareCounter_++);
  declarations_ << "  " << tp << " " << name << ";\n";
  return name;
}

std::string CompileCtx::param(ValueCP value, const KernelOperation& op) const {
  auto off = op.paramOffset(value);
  if (value->type().kind() == nativert::Type::Kind::Tensor) {
    return fmt::format("param<Tensor>(blockInfo, {})", off);
  }
  return fmt::format("param<{}>(blockInfo, {})", cudaType(value), off);
}

std::string CompileCtx::makeElementRef(ValueCP value, const KernelOperation& op)
    const {
  auto off = op.paramOffset(value);
  if (value->type().kind() == nativert::Type::Kind::Tensor) {
    return fmt::format(
        "elementRef<{}>(param<Tensor>(blockInfo, {}), idx)",
        cudaType(value),
        off);
  }
  return fmt::format("*param<{}>(blockInfo, {})", cudaType(value), off);
}

void addSelfExtraBindings(
    OpInvocation& op,
    const std::vector<ValueCP>& extraValues) {
  for (auto* value : extraValues) {
    op.addBinding(value->id(), value->id());
  }
}

void addDuplicateExtraBindings(
    OpInvocation& op,
    const std::vector<ValueCP>& formalExtras,
    WaveGraph& waveGraph) {
  for (auto* formal : formalExtras) {
    auto* actual = waveGraph.duplicateValue(formal);
    op.addBinding(formal->id(), actual->id());
  }
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
    auto it = projectOps_.find(sg);
    if (it != projectOps_.end()) {
      ops_.emplace_back(it->second, sg, ivalueStorage_);
      addDuplicateExtraBindings(
          ops_.back(), it->second->extraValues(), waveGraph_);
      // Map formal syncable value ids to actual ids.
      const auto& bindings = ops_.back().bindings();
      for (const auto& [formalId, actualId] : bindings) {
        if (waveGraph_.syncableValueIds().count(formalId)) {
          waveGraph_.addSyncableValueId(actualId);
        }
      }
    } else {
      auto* projectOp = makeProjectionOperation(sg);
      if (projectOp) {
        projectOps_[sg] = projectOp;
        ops_.emplace_back(projectOp, sg, ivalueStorage_);
        addSelfExtraBindings(ops_.back(), projectOp->extraValues());
      }
    }
  }
  auto compositeKernel = std::make_unique<CompositeKernel>(
      std::move(opStorage_), std::move(kernelOpStorage_), includes_);
  auto invocation = std::make_unique<CompositeInvocation>();
  invocation->kernelInfo = compositeKernel->kernelInfo();
  invocation->kernel = std::move(compositeKernel);
  invocation->ops = std::move(ops_);
  invocation->ivalueStorage = std::move(ivalueStorage_);
  invocation->sequenceNumber = waveGraph_.nextCompositeInvocationId();
  return std::make_unique<CompiledNode>(std::move(invocation));
}

} // namespace torch::wave
