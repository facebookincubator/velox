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
#include <folly/ScopeGuard.h>
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
  auto* meta = nodeMeta(&node);
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
  auto* meta = nodeMeta(&node);
  if (meta && predicate(*meta, &node)) {
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
    CompileCtx::NodeSet& placed,
    std::unordered_set<ValueCP>& seen,
    std::vector<ValueCP>& result) {
  for (auto& input : node->inputs()) {
    auto* value = input.value;
    auto* producer = value->producer();
    if (producer && producer->target() == "prim.Input") {
      placed.insert(producer);
    }
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
  // dtype attributes must match when present.
  const auto* lDtype = left->tryGetAttribute("dtype");
  const auto* rDtype = right->tryGetAttribute("dtype");
  if ((lDtype != nullptr) != (rDtype != nullptr)) {
    return false;
  }
  if (lDtype && lDtype->value != rDtype->value) {
    return false;
  }
  auto* meta = nodeMeta(left);
  if (meta) {
    for (const auto& attrName : meta->templateAttrs) {
      const auto* lAttr = left->tryGetAttribute(attrName);
      const auto* rAttr = right->tryGetAttribute(attrName);
      if ((lAttr != nullptr) != (rAttr != nullptr)) {
        return false;
      }
      if (lAttr && lAttr->value != rAttr->value) {
        return false;
      }
    }
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
      auto leftKind = li[i].value->type().kind();
      auto rightKind = ri[i].value->type().kind();
      if (leftKind != rightKind) {
        return false;
      }
      auto* lp = li[i].value->producer();
      auto* rp = ri[i].value->producer();
      auto* lMeta = lp ? Registry::metadata(lp->target()) : nullptr;
      auto* rMeta = rp ? Registry::metadata(rp->target()) : nullptr;
      bool leftIsView = lMeta && lMeta->isView();
      bool rightIsView = rMeta && rMeta->isView();
      if (leftIsView != rightIsView) {
        return false;
      }
      if (leftKind == nativert::Type::Kind::Tensor) {
        if (!tensorMetaCompatible(
                *leftSg.inputTypes[leftPos], *rightSg.inputTypes[rightPos])) {
          return false;
        }
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
  // Include dtype attribute in hash if present.
  const auto* dtypeAttr = node->tryGetAttribute("dtype");
  if (dtypeAttr && std::holds_alternative<std::string>(dtypeAttr->value)) {
    auto dh = std::hash<std::string>{}(std::get<std::string>(dtypeAttr->value));
    hash ^= dh + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  }
  auto* meta = nodeMeta(node);
  if (meta) {
    for (const auto& attrName : meta->templateAttrs) {
      const auto* attr = node->tryGetAttribute(attrName);
      if (attr) {
        auto ah = std::hash<std::string>{}(constantToString(attr->value));
        hash ^= ah + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
    }
  }
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

void copyAttributes(const nativert::Node* source, nativert::Node* dest) {
  for (const auto& attr : source->attributes()) {
    std::visit(
        [&](auto&& val) {
          using T = std::decay_t<decltype(val)>;
          if constexpr (!std::is_same_v<T, std::unique_ptr<nativert::Graph>>) {
            nativert::Attribute newAttr;
            newAttr.name = attr.name;
            newAttr.value = val;
            dest->addAttribute(std::move(newAttr));
          }
        },
        attr.value);
  }
}

// Walks the variant chain from 'variantRoot' and maps every reachable node
// to 'original' in nodeMap, stopping at values already in valueMap (boundary).
void mapVariantChain(
    NodeCP variantRoot,
    NodeCP original,
    const std::unordered_map<ValueCP, nativert::Value*>& valueMap,
    std::unordered_map<NodeCP, NodeCP>& nodeMap) {
  std::unordered_set<NodeCP> visited;
  std::function<void(NodeCP)> walk = [&](NodeCP node) {
    if (!visited.insert(node).second) {
      return;
    }
    nodeMap[node] = original;
    for (const auto& input : node->inputs()) {
      if (valueMap.count(input.value)) {
        continue;
      }
      auto* producer = input.value->producer();
      if (producer) {
        walk(producer);
      }
    }
  };
  walk(variantRoot);
}

// Recursively deep-copies a node and its producers into 'target',
// stopping at values already in valueMap (boundary inputs).
// In kMulti/kCG modes, expands nodes that have a matching variant.
// Records copy→original mapping in nodeMap.
nativert::Node* copyVariantNode(
    NodeCP node,
    std::unordered_map<ValueCP, nativert::Value*>& valueMap,
    std::unordered_set<NodeCP>& visited,
    std::unordered_map<NodeCP, NodeCP>& nodeMap,
    nativert::Graph* target,
    VariantMode mode,
    WaveGraph& waveGraph) {
  if (visited.count(node)) {
    return nullptr;
  }

  for (const auto& input : node->inputs()) {
    if (valueMap.count(input.value)) {
      continue;
    }
    auto* producer = input.value->producer();
    if (producer) {
      copyVariantNode(
          producer, valueMap, visited, nodeMap, target, mode, waveGraph);
    }
  }

  if (mode != VariantMode::kSingle) {
    auto* meta = Registry::metadata(node->target());
    if (meta) {
      nativert::Node* variantRoot = nullptr;
      if (mode == VariantMode::kMulti && meta->makeMultiKernelVariant) {
        variantRoot = meta->makeMultiKernelVariant(node, &waveGraph);
      } else if (mode == VariantMode::kCG && meta->cgVariant) {
        variantRoot = meta->cgVariant(node, &waveGraph);
      }
      if (variantRoot) {
        waveGraph.optimizeNode(variantRoot);
        visited.insert(node);
        if (waveGraph.currentVariantGraph()) {
          // Variant function created nodes directly in the variant graph.
          // Map all variant chain nodes to the original.
          mapVariantChain(variantRoot, node, valueMap, nodeMap);
          for (auto* origOut : node->outputs()) {
            for (auto* vrOut : variantRoot->outputs()) {
              if (vrOut->id() == origOut->id()) {
                valueMap[origOut] = vrOut;
                break;
              }
            }
          }
        } else {
          // Variant function created nodes in the main graph. Deep-copy
          // the variant chain into the target graph.
          auto* newRoot = copyVariantNode(
              variantRoot, valueMap, visited, nodeMap, target, mode, waveGraph);
          if (newRoot) {
            nodeMap[newRoot] = node;
          }
          for (auto* origOut : node->outputs()) {
            if (!valueMap.count(origOut) && newRoot) {
              auto* newOut = newRoot->addOutput(
                  std::string(origOut->name()), origOut->type());
              newOut->setId(origOut->id());
              valueMap[origOut] = newOut;
            }
          }
        }
        return variantRoot;
      }
    }
  }

  std::vector<nativert::NamedArgument> newInputs;
  for (const auto& input : node->inputs()) {
    auto it = valueMap.find(input.value);
    TORCH_CHECK(
        it != valueMap.end(),
        "Missing input value in variant subgraph copy: ",
        input.value->name());
    newInputs.push_back({std::string(input.name), it->second});
  }
  auto* newNode = target->insertNode(
      std::string(node->target()), std::move(newInputs), node->metadata());
  copyAttributes(node, newNode);
  for (const auto* outVal : node->outputs()) {
    auto* newOut =
        newNode->addOutput(std::string(outVal->name()), outVal->type());
    newOut->setId(outVal->id());
    valueMap[outVal] = newOut;
  }
  for (const auto* outVal : node->outputs()) {
    if (outVal->type().kind() != nativert::Type::Kind::TensorList) {
      continue;
    }
    if (node->target() == "prim.ListPack") {
      continue;
    }
    for (auto* user : outVal->users()) {
      if (user->target() != "prim.ListUnpack" || visited.count(user)) {
        continue;
      }
      std::vector<nativert::NamedArgument> unpackInputs;
      for (const auto& inp : user->inputs()) {
        auto it = valueMap.find(inp.value);
        TORCH_CHECK(it != valueMap.end());
        unpackInputs.push_back({std::string(inp.name), it->second});
      }
      auto* newUnpack = target->insertNode(
          "prim.ListUnpack", std::move(unpackInputs), user->metadata());
      for (const auto* unpackOut : user->outputs()) {
        auto* newUnpackOut = newUnpack->addOutput(
            std::string(unpackOut->name()), unpackOut->type());
        newUnpackOut->setId(unpackOut->id());
        valueMap[unpackOut] = newUnpackOut;
      }
      visited.insert(user);
      nodeMap[newUnpack] = user;
      break;
    }
  }
  visited.insert(node);
  nodeMap[newNode] = node;
  return newNode;
}

} // namespace

Subgraph CompileCtx::extractSubgraph(
    NodeCP node,
    const NodeSet& inputs,
    NodeSet& placed) {
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

Subgraph CompileCtx::variantSubgraph(const Subgraph& sg, VariantMode mode) {
  auto graph = nativert::Graph::createGraph();
  waveGraph_.setCurrentVariantGraph(graph.get());

  Subgraph result;
  result.inputTypes = sg.inputTypes;
  std::unordered_map<ValueCP, nativert::Value*> valueMap;
  result.inputs.reserve(sg.inputs.size());
  for (size_t i = 0; i < sg.inputs.size(); ++i) {
    auto* inputValue = sg.inputs[i];
    auto* newValue = graph->addValue(
        std::string(inputValue->name()), inputValue->type(), nullptr);
    newValue->setId(inputValue->id());
    valueMap[inputValue] = newValue;
    result.inputs.push_back(newValue);
  }

  std::unordered_set<NodeCP> visited;
  std::unordered_map<NodeCP, NodeCP> nodeMap;
  result.root = copyVariantNode(
      sg.root, valueMap, visited, nodeMap, graph.get(), mode, waveGraph_);
  variantToOriginal_.insert(nodeMap.begin(), nodeMap.end());

  waveGraph_.setCurrentVariantGraph(nullptr);
  waveGraph_.addVariantGraph(std::move(graph));
  return result;
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

bool CompileCtx::isMultikernel(
    const nativert::Node& node,
    const NodeSet& placed) const {
  NodeSet visited;
  return anyReachable(
      node,
      placed,
      [](const Metadata& m, NodeCP) {
        return m.makeMultiKernelVariant != nullptr;
      },
      visited);
}

ProjectOperation* CompileCtx::makeProjectionOperation(const Subgraph& sg) {
  projectOpSubgraph_ = &sg;
  constantMap_ = sg.makeConstantIndices();
  opStorage_.push_back(std::make_unique<ProjectOperation>(sg, *this));
  auto* projectOp = opStorage_.back().get();

  // Check if any node in the subgraph has singleBlockIfFused.
  NodeSet visited;
  auto& types = waveGraph_.types();
  bool hasSingleBlock = anyReachable(
      *sg.root,
      placed_,
      [&types](const Metadata& m, NodeCP node) {
        return m.singleBlockIfFused && !m.isStandalone(node, types);
      },
      visited);
  if (hasSingleBlock) {
    auto singleSg = variantSubgraph(sg, VariantMode::kSingle);
    setIsSingleBlock(true);
    projectOp->singleBlockGrid_ = makeGrid(singleSg.root);
    setIsSingleBlock(false);

    // Check if any node has a cgVariant.
    NodeSet cgVisited;
    bool hasCgVariant = anyReachable(
        *sg.root,
        placed_,
        [&types](const Metadata& m, NodeCP node) {
          return m.cgVariant && !m.isStandalone(node, types);
        },
        cgVisited);
    if (hasCgVariant) {
      auto cgSg = variantSubgraph(sg, VariantMode::kCG);
      setIsCgGrid(true);
      projectOp->cgGrid_ = makeGrid(cgSg.root);
      setIsCgGrid(false);
    }
  }
  auto multiSg = variantSubgraph(sg, VariantMode::kMulti);
  projectOp->grid_ = makeGrid(multiSg.root);
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
  scanGrid(projectOp->cgGrid_);
}

void CompileCtx::newGrid() {
  placed_ = placedBeforeNode_;
  grid_.clear();
}

LaunchGrid CompileCtx::makeGrid(NodeCP node) {
  newGrid();
  auto result = placeKernels(node, Context::kTop);
  if (result == Context::kFused) {
    pushdownFused(node);
  }
  return std::move(grid_);
}

// Returns true if the input name maps to a sizeArgs ordinal in the metadata.
static bool isSizeArg(
    std::string_view inputName,
    const Metadata* meta) {
  if (!meta || !meta->functionSchema) {
    return false;
  }
  const auto& schemaArgs = meta->functionSchema->arguments();
  for (size_t i = 0; i < schemaArgs.size(); ++i) {
    if (schemaArgs[i].name() == inputName) {
      auto ordinal = static_cast<int32_t>(i);
      for (auto sizeOrd : meta->sizeArgs.ordinal) {
        if (sizeOrd == ordinal) {
          return true;
        }
      }
      return false;
    }
  }
  return false;
}

Context CompileCtx::placeKernels(NodeCP node, Context context) {
  auto* meta = nodeMeta(node);
  auto thisContext = (!meta || meta->isStandalone(node, types_) ||
                      WaveConfig::get().allStandalone)
      ? Context::kStandalone
      : Context::kFused;
  std::vector<NodeCP> standaloneInputs;
  std::vector<NodeCP> fusedInputs;

  auto placeInput = [&](ValueCP value, bool isSize) {
    auto* producer = value->producer();
    if (!producer || placed_.count(producer) ||
        (inputs_ && inputs_->count(producer))) {
      return;
    }
    auto inputContext = placeKernels(producer, thisContext);
    if (inputContext == Context::kFused) {
      if (isSize) {
        pushdownFused(producer);
      } else {
        fusedInputs.push_back(producer);
      }
    }
  };

  for (auto i = 0; i < node->inputs().size(); ++i) {
    if (meta && meta->inputFromPreviousKernel.has_value() &&
        i != meta->inputFromPreviousKernel.value()) {
      continue;
    }
    auto* inputValue = node->inputs()[i].value;
    auto* producer = inputValue->producer();
    auto inputKind = inputValue->type().kind();
    bool isSize = isSizeArg(node->inputs()[i].name, meta) &&
        inputKind != nativert::Type::Kind::Tensor &&
        inputKind != nativert::Type::Kind::TensorList;
    if (thisContext == Context::kFused && producer &&
        producer->target() == "prim.ListPack") {
      for (const auto& listInput : producer->inputs()) {
        placeInput(listInput.value, isSize);
      }
    } else {
      placeInput(inputValue, isSize);
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
  launch.standalone = originalFromVariant(node);
  placeKernelLaunch(std::move(launch));
  placed_.insert(node);
  standaloneNodes_.insert(originalFromVariant(node));
}

void CompileCtx::fillConstantIndices(const Subgraph& sg, Launch& launch) {
  std::unordered_set<ValueCP> sgInputSet(sg.inputs.begin(), sg.inputs.end());
  std::unordered_set<NodeCP> attrVisited;
  forEachSortedAttribute(
      sg.root,
      sgInputSet,
      attrVisited,
      [&](NodeCP n, const nativert::Attribute& attr) {
        NodeCP original = originalFromVariant(n);

        auto ordinal = projectOpSubgraph_->nodeOrdinal(original);
        auto mapIt = constantMap_.find(ordinal);
        TORCH_CHECK(
            mapIt != constantMap_.end(),
            "Node ordinal not found in constantMap: ",
            original->target());

        // Find the attribute's offset in the original node's sorted
        // non-skipped attributes.
        int32_t attrOffset = 0;
        bool found = false;
        forEachSortedAttribute(
            original, [&](NodeCP, const nativert::Attribute& origAttr) {
              if (origAttr.name == attr.name) {
                found = true;
              } else if (!found) {
                ++attrOffset;
              }
            });
        TORCH_CHECK(
            found,
            "Attribute '",
            attr.name,
            "' not found in original node: ",
            original->target());
        launch.constantIndices.push_back(mapIt->second + attrOffset);
      });
  TORCH_CHECK(
      static_cast<int32_t>(launch.constantIndices.size()) ==
          launch.op->numConstants(),
      "Launch constant count (",
      launch.constantIndices.size(),
      ") does not match KernelOperation numConstants (",
      launch.op->numConstants(),
      ")");
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
  for (auto* output : sg.root->outputs()) {
    ResultSpec rs;
    rs.value = output;
    resultSpecs.push_back(rs);
  }
  fusedCode(sg.root, resultSpecs);
}

void CompileCtx::placeKernelLaunch(Launch launch) {
  int32_t latestLevel = -1;

  // Collect input value ids of this launch.
  std::unordered_set<nativert::ValueId> inputIds;
  if (launch.standalone) {
    for (const auto& input : launch.standalone->inputs()) {
      inputIds.insert(input.value->id());
    }
  } else if (launch.op) {
    inputIds = launch.op->orderingInputs();
  }

  // Find the latest level in grid_ containing a Launch that produces
  // any of these input values.
  for (int32_t level = 0; level < static_cast<int32_t>(grid_.size()); ++level) {
    for (auto& existing : grid_[level]) {
      bool produces = false;
      if (existing.standalone) {
        for (auto* output : existing.standalone->outputs()) {
          if (inputIds.count(output->id())) {
            produces = true;
            break;
          }
        }
      } else if (existing.op) {
        for (auto outputId : existing.op->orderingOutputs()) {
          if (inputIds.count(outputId)) {
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
    if (producer->target() == "prim.ListPack") {
      for (auto& listInput : producer->inputs()) {
        auto* listValue = listInput.value;
        auto* listProducer = listValue->producer();
        if (!listProducer || placed_.count(listProducer) ||
            opInputs.count(listValue)) {
          continue;
        }
        if (!visited.insert(listProducer).second) {
          continue;
        }
        std::vector<ResultSpec> resultSpecs;
        for (auto* output : listProducer->outputs()) {
          ResultSpec rs;
          rs.value = output;
          resultSpecs.push_back(rs);
        }
        fusedCode(listProducer, resultSpecs);
      }
      continue;
    }
    auto* meta = nodeMeta(producer);
    if (meta && meta->elementwise) {
      generateElementwiseBorderImpl(producer, opInputs, visited);
    } else {
      std::vector<ResultSpec> resultSpecs;
      for (auto* output : producer->outputs()) {
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

void CompileCtx::generateIndexToOffset(
    const ElementExpr& ee,
    const std::vector<ValueCP>& allInputs) {
  auto& op = *generatingOp_;
  std::vector<int32_t> paramOffs;
  std::vector<int32_t> outputOffs;
  std::vector<int32_t> altOffs;

  int32_t outputOff = op.paramOffset(ee.output);
  for (auto* v : ee.inputs) {
    if (v->type().kind() != nativert::Type::Kind::Tensor) {
      continue;
    }
    paramOffs.push_back(op.paramOffset(v));
    outputOffs.push_back(outputOff);
    auto ait = ee.altParamOffset.find(v);
    altOffs.push_back(ait != ee.altParamOffset.end() ? ait->second : -1);
  }
  if (ee.output->type().kind() == nativert::Type::Kind::Tensor) {
    paramOffs.push_back(outputOff);
    outputOffs.push_back(outputOff);
    altOffs.push_back(-1);
  }

  if (paramOffs.empty()) {
    return;
  }

  auto emitArray = [&](const char* name, const std::vector<int32_t>& arr) {
    code_ << "    static int32_t " << name << "[] = {";
    for (size_t i = 0; i < arr.size(); ++i) {
      if (i > 0) {
        code_ << ", ";
      }
      code_ << arr[i];
    }
    code_ << "};\n";
  };
  code_ << "  {\n";
  emitArray("paramOffsets", paramOffs);
  emitArray("outputOffsets", outputOffs);
  emitArray("altOffsets", altOffs);
  code_
      << "    for (auto i = threadIdx.x; i < sizeof(paramOffsets) / sizeof(paramOffsets[0]); i += blockDim.x) {\n"
      << "      if (altOffsets[i] != -1) {\n"
      << "        copyTensorHead(param<Tensor>(blockInfo, paramOffsets[i]), param<Tensor>(blockInfo, altOffsets[i]));\n"
      << "        param<Tensor>(blockInfo, altOffsets[i])->init<true>(param<Tensor>(blockInfo, outputOffsets[i]));\n"
      << "      } else {\n"
      << "        param<Tensor>(blockInfo, paramOffsets[i])->init<true>(outputOffsets[i] != paramOffsets[i] ? param<Tensor>(blockInfo, outputOffsets[i]) : nullptr);\n"
      << "      }\n"
      << "    }\n"
      << "  }\n"
      << "  __syncthreads();\n";
}

void CompileCtx::functionLoop(NodeCP node) {
  auto& op = *generatingOp_;
  auto* meta = nodeMeta(node);
  TORCH_CHECK(meta, "No metadata for: ", node->target());

  // Find the size argument.
  TORCH_CHECK(
      !meta->sizeArgs.ordinal.empty(), "functionLoop requires sizeArgs");
  auto sizeArgIdx = meta->sizeArgs.ordinal[0];
  auto* sizeValue = node->inputs()[sizeArgIdx].value;

  // Add shared declaration for size.
  op.addSharedDeclaration("  __shared__ uint32_t size;\n");

  // Set up input and output specs for makeCall.
  const auto& inputs = node->inputs();
  std::vector<ResultSpec> inputSpecs;
  for (const auto& input : inputs) {
    ResultSpec rs;
    rs.value = input.value;
    inputSpecs.push_back(rs);
  }

  std::vector<ResultSpec> resultSpecs;
  for (auto* output : node->outputs()) {
    ResultSpec rs;
    rs.value = output;
    resultSpecs.push_back(rs);
  }

  // Compute size (and optionally rounded) on thread 0, then sync.
  if (meta->hasBarrier) {
    op.addSharedDeclaration("  __shared__ uint32_t rounded;\n");
    code_
        << "  if (threadIdx.x == 0) {\n"
        << "    size = numEl(*" << param(sizeValue, op) << ");\n"
        << "    rounded = roundUpPwr2(size, blockDim.x);\n"
        << "  }\n"
        << "  __syncthreads();\n"
        << "    for (uint32_t idx = blockInfo.blockInOp * blockDim.x + threadIdx.x; idx < rounded; idx += blockInfo.numBlocksInOp * blockDim.x) {\n";
  } else {
    code_
        << "  if (threadIdx.x == 0) {\n"
        << "    size = numEl(*" << param(sizeValue, op) << ");\n"
        << "  }\n"
        << "  __syncthreads();\n"
        << "    for (uint32_t idx = blockInfo.blockInOp * blockDim.x + threadIdx.x; idx < size; idx += blockInfo.numBlocksInOp * blockDim.x) {\n";
  }
  code_ << "  " << makeCall(node, inputSpecs, resultSpecs) << "\n";
  code_ << "    }\n";
}

bool CompileCtx::isSizeSetInThisOp(
    ValueCP value,
    std::unordered_set<ValueCP>& visited) {
  if (!visited.insert(value).second) {
    return false;
  }
  const auto& opInputs = generatingOp_->orderedInputs();
  auto numInputs = generatingOp_->numInputs();
  for (int32_t i = 0; i < numInputs; ++i) {
    if (opInputs[i] == value) {
      return false;
    }
  }
  auto* producer = value->producer();
  if (!producer) {
    return false;
  }
  auto* meta = nodeMeta(producer);
  if (meta) {
    for (size_t i = 0; i < meta->returnMeta.size(); ++i) {
      if (meta->returnMeta[i].shapeSetOnDevice) {
        return true;
      }
    }
  }
  for (const auto& input : producer->inputs()) {
    if (isSizeSetInThisOp(input.value, visited)) {
      return true;
    }
  }
  return false;
}

void CompileCtx::fusedCode(NodeCP node, std::vector<ResultSpec>& resultSpecs) {
  if (placed_.count(node)) {
    return;
  }
  auto* meta = nodeMeta(node);
  TORCH_CHECK(meta, "No metadata for node: ", node->target());

  if (meta->specialForm) {
    meta->specialForm(node, resultSpecs, this);
    return;
  }

  if (meta->isView()) {
    auto viewArgOrdinal = *meta->viewOfArg;
    auto* viewInput = node->inputs()[viewArgOrdinal].value;
    auto* producer = viewInput->producer();
    if (producer && !placed_.count(producer)) {
      std::vector<ResultSpec> prodSpecs;
      for (auto* output : producer->outputs()) {
        ResultSpec rs;
        rs.value = output;
        prodSpecs.push_back(rs);
      }
      fusedCode(producer, prodSpecs);
    }
    std::unordered_set<ValueCP> visited;
    if (isSizeSetInThisOp(viewInput, visited) && !meta->deviceFunc.empty()) {
      std::vector<ResultSpec> inputSpecs;
      for (const auto& input : node->inputs()) {
        ResultSpec rs;
        rs.value = input.value;
        inputSpecs.push_back(rs);
      }
      code_ << "  " << makeCall(node, inputSpecs, resultSpecs) << "\n";
    }
    placed_.insert(node);
    generatingOp_->allNodes().insert(node);
    return;
  }

  if (meta->elementwise) {
    generateElementwiseBorder(node);
    Subgraph sg = extractSubgraph(node, *inputs_, placed_);
    generateElementwise({sg}, {resultSpecs[0]});
    return;
  }

  // Not elementwise - recurse on inputs backed by memory (outputs of
  // generatingOp_).
  auto memOutputs = generatingOp_->memOutputs();

  for (const auto& input : node->inputs()) {
    auto* value = input.value;
    if (memOutputs.count(value)) {
      auto* producer = value->producer();
      if (producer && !placed_.count(producer)) {
        std::vector<ResultSpec> prodSpecs;
        for (auto* output : producer->outputs()) {
          ResultSpec rs;
          rs.value = output;
          prodSpecs.push_back(rs);
        }
        fusedCode(producer, prodSpecs);
      }
    }
  }

  const auto& inputs = node->inputs();

  bool cgSingleBlock = isCgGrid_ && meta->singleBlockIfFused;

  if (!meta->hasRegisterInputs()) {
    // No register inputs - generate plain call.
    std::vector<ResultSpec> inputSpecs;
    for (const auto& input : inputs) {
      ResultSpec rs;
      rs.value = input.value;
      inputSpecs.push_back(rs);
    }
    if (cgSingleBlock) {
      emitBarrier();
      code_ << "  if (blockInfo.blockInOp == 0) {\n";
    }
    code_ << "  " << makeCall(node, inputSpecs, resultSpecs) << "\n";
    if (cgSingleBlock) {
      code_ << "  }\n";
      emitBarrier();
    }
    placed_.insert(node);
    generatingOp_->allNodes().insert(node);
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
    if (cgSingleBlock) {
      emitBarrier();
      code_ << "  if (blockInfo.blockInOp == 0) {\n";
    }
    functionLoop(node);
    if (cgSingleBlock) {
      code_ << "  }\n";
      emitBarrier();
    }
    placed_.insert(node);
    generatingOp_->allNodes().insert(node);
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
  if (cgSingleBlock) {
    emitBarrier();
    code_ << "  if (blockInfo.blockInOp == 0) {\n";
  }
  if (!subgraphs.empty()) {
    generateElementwise(subgraphs, ewResultSpecs, callStmt, meta->hasBarrier);
  }
  if (cgSingleBlock) {
    code_ << "  }\n";
    emitBarrier();
  }
  placed_.insert(node);
  generatingOp_->allNodes().insert(node);
}

void CompileCtx::generateElementwise(
    const std::vector<Subgraph>& subgraphs,
    const std::vector<ResultSpec>& resultSpecs,
    std::string resultStmt,
    bool fullBlockResult) {
  auto& op = *generatingOp_;

  // Get unique leaf inputs by walking subgraph roots.
  auto leafInputs = subgraphInputs(subgraphs);

  // Build ElementExprs early so altParamOffset is available during code gen.
  std::vector<ElementExpr> newExprs;
  for (size_t s = 0; s < subgraphs.size(); ++s) {
    ElementExpr ee;
    // There is an output Value (shape only) also when generating an element
    // wise expr that produces values in registers.
    ee.output = subgraphs[s].root->outputs()[0];
    ee.inputs = leafInputs;
    {
      auto* producer = ee.output->producer();
      if (producer && op.allNodes().count(producer)) {
        for (const auto& desc : op.outputDescs()) {
          if (desc.shapeSetOnDevice) {
            ee.shapeFromThisOp = true;
            break;
          }
        }
      }
    }
    std::unordered_set<ValueCP> existingInputs;
    for (const auto& prev : op.elementExprs()) {
      for (auto* v : prev.inputs) {
        existingInputs.insert(v);
      }
    }
    for (size_t p = 0; p < s; ++p) {
      for (auto* v : newExprs[p].inputs) {
        existingInputs.insert(v);
      }
    }
    for (auto* v : ee.inputs) {
      if (v->type().kind() == nativert::Type::Kind::Tensor &&
          existingInputs.count(v)) {
        ee.altParamOffset[v] = op.allocAltParam();
      }
    }
    newExprs.push_back(std::move(ee));
  }

  for (auto& rs : resultSpecs) {
    if (rs.value &&
        std::find(leafInputs.begin(), leafInputs.end(), rs.value) ==
            leafInputs.end()) {
      leafInputs.push_back(rs.value);
    }
  }

  // Set currentElementExpr_ for the duration of code generation.
  const ElementExpr* prevElementExpr = currentElementExpr_;
  currentElementExpr_ = newExprs.empty() ? nullptr : &newExprs[0];
  SCOPE_EXIT {
    currentElementExpr_ = prevElementExpr;
  };

  // Build allInputs = leafInputs + result values for storage declarations.
  std::unordered_set<ValueCP> seen(leafInputs.begin(), leafInputs.end());
  auto allInputs = leafInputs;
  for (auto& rs : resultSpecs) {
    if (rs.value && seen.insert(rs.value).second) {
      allInputs.push_back(rs.value);
    }
  }

  code_ << "  {\n";

  // Collect values marked as wholeTensor in any subgraph node's argumentMeta.
  std::unordered_set<ValueCP> wholeTensorValues;
  for (const auto& sg : subgraphs) {
    auto* meta = nodeMeta(sg.root);
    if (meta) {
      const auto& inputs = sg.root->inputs();
      for (size_t i = 0; i < inputs.size() && i < meta->argumentMeta.size();
           ++i) {
        if (meta->argumentMeta[i].wholeTensor) {
          wholeTensorValues.insert(inputs[i].value);
        }
      }
    }
  }

  // Build tensor-only bit index for fast path processing.
  int32_t tensorCount = 0;
  fastPathBitIndex_.assign(leafInputs.size(), -1);
  for (size_t i = 0; i < leafInputs.size(); ++i) {
    if (leafInputs[i]->type().kind() == nativert::Type::Kind::Tensor) {
      fastPathBitIndex_[i] = tensorCount++;
    }
  }
  // Generate shared declarations for size and fast path flags.
  op.addSharedDeclaration("  __shared__ uint32_t size;\n");
  auto numFastPathVars = (tensorCount + 31) / 32;
  for (size_t i = 0; i < numFastPathVars; ++i) {
    op.addSharedDeclaration(
        "  __shared__ uint32_t isFastPath" + std::to_string(i) + ";\n");
  }

  // Generate the head: compute size and fast path flags from tensor leaf
  // inputs only.
  code_ << "  if (threadIdx.x == 0) {\n";
  for (size_t i = 0; i < numFastPathVars; ++i) {
    code_ << "    isFastPath" << i << " = 0;\n";
  }
  bool firstTensor = true;
  bool declaredSize2 = false;
  for (size_t valueIdx = 0; valueIdx < leafInputs.size(); ++valueIdx) {
    if (leafInputs[valueIdx]->type().kind() != nativert::Type::Kind::Tensor) {
      continue;
    }
    if (wholeTensorValues.count(leafInputs[valueIdx])) {
      continue;
    }
    auto bitIdx = fastPathBitIndex_[valueIdx];
    auto W = bitIdx / 32;
    auto B = bitIdx % 32;
    if (firstTensor) {
      code_ << "    Tensor* temp = " << param(leafInputs[valueIdx], op)
            << ";\n    size = temp->numEl;\n"
            << "    isFastPath0 |= temp->contiguous;\n";
      firstTensor = false;
    } else {
      if (!declaredSize2) {
        code_ << "    uint32_t size2;\n";
        declaredSize2 = true;
      }
      code_ << "    temp = " << param(leafInputs[valueIdx], op) << ";\n"
            << "    size2 = temp->numEl;\n"
            << "    isFastPath" << W << " |= (uint32_t)temp->contiguous << "
            << B << ";\n"
            << "    if (size2 != size) {\n"
            << "      if (size2 > size) {\n";
      for (int32_t I = 0; (I + 1) * 32 <= bitIdx; ++I) {
        code_ << "        isFastPath" << I << " = 0;\n";
      }
      code_ << "        isFastPath" << W << " &= ~((1 << " << B
            << ") - 1);\n"
            << "      size = size2;\n"
            << "      } else {\n"
            << "    isFastPath" << W << " &= ~(1 << " << B << ");\n"
            << "}"
            << "    }\n";
    }
  }
  code_ << "  }\n"
        << "  __syncthreads();\n";

  addInclude("velox/experimental/torchwave/Elementwise.cuh");

  // Declare tensor storage for all unique inputs.
  for (size_t i = 0; i < allInputs.size(); ++i) {
    auto tp = cudaType(allInputs[i]);
    if (allInputs[i]->type().kind() == nativert::Type::Kind::Tensor) {
      code_ << "  " << tp << "* b" << i << " = storage<" << tp << ">("
            << param(allInputs[i], op) << ");\n";
    } else {
      code_ << "  " << tp << "* b" << i << " = " << param(allInputs[i], op)
            << ";\n";
    }
  }

  // Declare attributes for all subgraph roots.
  for (auto& sg : subgraphs) {
    code_ << declareAttributes(sg.root, op, allInputs);
  }

  for (const auto& ee : newExprs) {
    if (ee.shapeFromThisOp) {
      generateIndexToOffset(ee, allInputs);
    }
  }

  // Generate fast path test for tensor leaf inputs.
  code_ << "  if (";
  for (size_t i = 0; i < numFastPathVars; ++i) {
    if (i > 0) {
      code_ << " && ";
    }
    uint32_t mask;
    if (i < numFastPathVars - 1) {
      mask = 0xffffffff;
    } else {
      auto bitsInLast = tensorCount - i * 32;
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
        if (resultSpecs[s].value->type().kind() ==
            nativert::Type::Kind::Tensor) {
          code_ << "        b" << id << "[idx] = result" << s << ";\n";
        } else {
          code_ << "        b" << id << "[0] = result" << s << ";\n";
        }
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
        code_ << "      " << tp << " result" << s << " = " << expr << ";\n";
        if (resultSpecs[s].value->type().kind() ==
            nativert::Type::Kind::Tensor) {
          code_ << "      b" << id << "[idx] = result" << s << ";\n";
        } else {
          code_ << "      b" << id << "[0] = result" << s << ";\n";
        }
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

  code_ << "  } else {\n";
  if (fullBlockResult) {
    code_
        << "    uint32_t rounded = roundUpPwr2(size, blockDim.x);\n"
        << "    for (uint32_t idx = blockInfo.blockInOp * blockDim.x + threadIdx.x; idx < rounded; idx += blockInfo.numBlocksInOp * blockDim.x) {\n";
    for (size_t s = 0; s < subgraphs.size(); ++s) {
      auto tp = resultSpecs[s].value ? cudaType(resultSpecs[s].value)
                                     : cudaType(allInputs[0]);
      code_ << "      " << tp << " result" << s << ";\n";
    }
    code_ << "      if (idx < size) {\n";
    for (size_t s = 0; s < subgraphs.size(); ++s) {
      auto expr = elementwiseExpr(subgraphs[s].root, op, allInputs, true);
      code_ << "        result" << s << " = " << expr << ";\n";
      if (resultSpecs[s].value) {
        auto it =
            std::find(allInputs.begin(), allInputs.end(), resultSpecs[s].value);
        auto id = it - allInputs.begin();
        if (resultSpecs[s].value->type().kind() ==
            nativert::Type::Kind::Tensor) {
          auto bitIdx = fastPathBitIndex_[id];
          code_ << "        b" << id << "[complexIdx(isFastPath" << bitIdx / 32
                << " & (1 << " << bitIdx % 32 << "), "
                << param(resultSpecs[s].value, op) << ", idx)] = result" << s
                << ";\n";
        } else {
          code_ << "        b" << id << "[0] = result" << s << ";\n";
        }
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
    for (size_t s = 0; s < subgraphs.size(); ++s) {
      auto expr = elementwiseExpr(subgraphs[s].root, op, allInputs, true);
      if (resultSpecs[s].value) {
        auto it =
            std::find(allInputs.begin(), allInputs.end(), resultSpecs[s].value);
        auto id = it - allInputs.begin();
        auto tp = cudaType(resultSpecs[s].value);
        code_ << "      " << tp << " result" << s << " = " << expr << ";\n";
        if (resultSpecs[s].value->type().kind() ==
            nativert::Type::Kind::Tensor) {
          auto bitIdx = fastPathBitIndex_[id];
          code_ << "      b" << id << "[complexIdx(isFastPath" << bitIdx / 32
                << " & (1 << " << bitIdx % 32 << "), "
                << param(resultSpecs[s].value, op) << ", idx)] = result" << s
                << ";\n";
        } else {
          code_ << "      b" << id << "[0] = result" << s << ";\n";
        }
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
  code_ << "  }\n";
  code_ << "  }\n";

  for (auto& ee : newExprs) {
    op.elementExprs().push_back(std::move(ee));
  }
}

namespace {

std::string cudaAttrType(const nativert::Constant& c) {
  return std::visit(
      [&c](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, bool>) {
          return "bool";
        } else if constexpr (std::is_same_v<T, int64_t>) {
          return "int64_t";
        } else if constexpr (std::is_same_v<T, double>) {
          return "double";
        } else if constexpr (
            std::is_same_v<T, c10::ScalarType> ||
            std::is_same_v<T, c10::MemoryFormat> ||
            std::is_same_v<T, c10::Layout>) {
          return "int64_t";
        } else {
          TORCH_CHECK(
              false,
              "Unsupported attribute type for CUDA: ",
              constantToString(c));
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
        if (std::holds_alternative<nativert::None>(attr.value)) {
          return;
        }
        auto off = op.attrOffset(n, attr.name);
        auto tp = cudaAttrType(attr.value);
        ss << "  " << tp << " attr" << off << " = *param<" << tp
           << ">(blockInfo, " << off << ");\n";
      });
}

} // namespace

std::string CompileCtx::emitScalarListSetup(
    size_t argOrdinal,
    ValueCP value,
    const nativert::Attribute* attr,
    NodeCP node) {
  auto& op = *generatingOp_;
  std::vector<std::string> elements;
  if (value) {
    auto* producer = value->producer();
    TORCH_CHECK(
        producer && producer->target() == "prim.ListPack",
        "SymIntList argument must come from prim.ListPack: ",
        node->target());
    for (const auto& listInput : producer->inputs()) {
      elements.push_back(
          "*" + param(listInput.value, op));
    }
  } else {
    TORCH_CHECK(attr, "ScalarList argument has no value or attribute");
    auto* vec = std::get_if<std::vector<int64_t>>(&attr->value);
    TORCH_CHECK(
        vec,
        "ScalarList attribute must be vector<int64_t>: ",
        node->target());
    for (auto v : *vec) {
      elements.push_back(std::to_string(v));
    }
  }
  auto numElements = elements.size();
  auto allocSize =
      sizeof(ScalarList) + sizeof(int64_t) * numElements;
  auto off = op.allocAltParam(allocSize);
  auto varName = "l" + std::to_string(argOrdinal);
  std::stringstream setup;
  setup << "  ScalarList* " << varName << " = param<ScalarList>(blockInfo, "
        << off << ");\n"
        << "  if (threadIdx.x == 0) {\n"
        << "    " << varName << "->size = " << numElements << ";\n"
        << "    " << varName
        << "->data = reinterpret_cast<int64_t*>(" << varName << " + 1);\n";
  for (size_t i = 0; i < elements.size(); ++i) {
    setup << "    " << varName << "->data[" << i << "] = " << elements[i]
          << ";\n";
  }
  setup << "  }\n"
        << "  __syncthreads();\n";
  return setup.str();
}

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

namespace {

std::string presentTemplateParams(const Metadata& meta, NodeCP node) {
  std::string result;
  const auto& schemaArgs = meta.functionSchema->arguments();
  const auto& nodeInputs = node->inputs();
  for (size_t i = 0; i < schemaArgs.size(); ++i) {
    if (i >= meta.argumentMeta.size() ||
        !meta.argumentMeta[i].hasPresentTemplateParam) {
      continue;
    }
    if (!result.empty()) {
      result += ", ";
    }
    bool present = false;
    const auto& argName = schemaArgs[i].name();
    for (const auto& input : nodeInputs) {
      if (input.name == argName) {
        present = true;
        break;
      }
    }
    if (!present) {
      const auto* attr = node->tryGetAttribute(argName);
      if (attr && !std::holds_alternative<nativert::None>(attr->value)) {
        present = true;
      }
    }
    result += present ? "true" : "false";
  }
  return result;
}

} // namespace

std::string CompileCtx::makeCall(
    NodeCP node,
    std::vector<ResultSpec> inputs,
    std::vector<ResultSpec> outputs) {
  auto& op = *generatingOp_;
  auto* meta = nodeMeta(node);
  TORCH_CHECK(meta, "No metadata for: ", node->target());

  if (!meta->headerFile.empty()) {
    addInclude(meta->headerFile);
  }

  std::stringstream ss;

  // Function name.
  ss << meta->deviceFunc;

  // Type template parameters from dtypes of node inputs at specified indices.
  auto presenceParams = meta->hasPresentTemplateParams()
      ? presentTemplateParams(*meta, node)
      : std::string();
  if (meta->hasBlockSizeTemplateParam || !meta->typeTemplateParams.empty() ||
      meta->hasDtypeTemplateParam || !meta->templateAttrs.empty() ||
      !presenceParams.empty()) {
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
    if (meta->hasDtypeTemplateParam) {
      if (!firstTp) {
        ss << ", ";
      }
      firstTp = false;
      const auto* dtypeAttr = node->tryGetAttribute("dtype");
      TORCH_CHECK(dtypeAttr, node->target(), ": missing dtype attribute");
      ss << cudaTypeFromDtype(*dtypeAttr);
    }
    for (const auto& attrName : meta->templateAttrs) {
      if (!firstTp) {
        ss << ", ";
      }
      firstTp = false;
      const auto* attr = node->tryGetAttribute(attrName);
      TORCH_CHECK(
          attr, node->target(), ": missing template attribute ", attrName);
      ss << constantToString(attr->value);
    }
    if (!presenceParams.empty()) {
      if (!firstTp) {
        ss << ", ";
      }
      ss << presenceParams;
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

  // Build a map from input name to index in the inputs ResultSpec vector.
  std::unordered_map<std::string_view, size_t> inputNameToIdx;
  for (size_t i = 0; i < node->inputs().size(); ++i) {
    inputNameToIdx[node->inputs()[i].name] = i;
  }

  // Setup code for ScalarList arguments emitted before the call.
  std::stringstream setupSs;

  // Inputs and attributes in schema argument order.
  forArguments(
      *meta,
      node,
      [&](size_t schemaIdx,
          ValueCP value,
          const nativert::Attribute* attr) {
        // Check for SymInt list: Value with SymIntList type or attribute with
        // vector<int64_t>.
        bool isSymIntList =
            (value &&
             value->type().kind() == nativert::Type::Kind::SymIntList) ||
            (attr &&
             std::holds_alternative<std::vector<int64_t>>(attr->value));
        if (isSymIntList) {
          setupSs << emitScalarListSetup(schemaIdx, value, attr, node);
          comma();
          ss << "*l" << schemaIdx;
          return;
        }
        if (value) {
          auto it = inputNameToIdx.find(
              meta->functionSchema->arguments()[schemaIdx].name());
          TORCH_CHECK(it != inputNameToIdx.end());
          auto i = it->second;
          if (i < meta->argumentMeta.size() &&
              meta->argumentMeta[schemaIdx].linkOnly) {
            return;
          }
          comma();
          if (inputs[i].value && inputs[i].variable.empty() &&
              schemaIdx < meta->argumentMeta.size() &&
              meta->argumentMeta[schemaIdx].isRegister) {
            ss << makeElementRef(inputs[i].value, op);
          } else if (inputs[i].value) {
            ss << param(inputs[i].value, op);
          } else {
            ss << inputs[i].variable;
          }
        } else if (attr) {
          comma();
          if (std::holds_alternative<nativert::None>(attr->value)) {
            ss << "0";
          } else {
            auto off = op.attrOffset(node, attr->name);
            auto tp = cudaAttrType(attr->value);
            ss << "*param<" << tp << ">(blockInfo, " << off << ")";
          }
        }
      });

  // Outputs.
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (i < meta->returnMeta.size() && meta->returnMeta[i].linkOnly) {
      continue;
    }
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

  // Shared declarations: declare in the kernel and pass as arguments.
  for (const auto& [type, name] : meta->sharedDecls) {
    op.addSharedDeclaration("  __shared__ " + type + " " + name + ";\n");
    comma();
    ss << name;
  }

  // Dynamic shared declarations: type from input dtype, name suffixed by type.
  // Ordinal -1 means use the resolved dtype attribute instead of an input.
  for (const auto& [ordinal, baseName] : meta->dynamicSharedDecls) {
    std::string tp;
    std::string suffix;
    if (ordinal >= 0) {
      auto* value = node->inputs()[ordinal].value;
      tp = cudaType(value);
      suffix = cudaTypeIdSuffix(types_.types[value->id()]->dtype());
    } else {
      const auto* dtypeAttr = node->tryGetAttribute("dtype");
      TORCH_CHECK(dtypeAttr, node->target(), ": missing dtype attribute");
      tp = cudaTypeFromDtype(*dtypeAttr);
      suffix = dtypeName(*dtypeAttr);
    }
    auto varName = baseName + suffix;
    op.addSharedDeclaration("  __shared__ " + tp + " " + varName + ";\n");
    comma();
    ss << varName;
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

  for (int32_t b = 0; b < meta->numBarriers; ++b) {
    comma();
    ss << op.allocateBarrier();
  }

  // blockInfo is always the last argument.
  comma();
  ss << "blockInfo);";

  auto setup = setupSs.str();
  if (setup.empty()) {
    return ss.str();
  }
  return "{\n" + setup + "  " + ss.str() + "\n  }";
}

void CompileCtx::callView(
    ValueCP src,
    ValueCP dest,
    const std::string& offsetExpr,
    int32_t elementSize) {
  auto& op = *generatingOp_;
  addInclude("velox/experimental/torchwave/Views.cuh");
  emitBarrier();
  if (isSingleBlock_) {
    code_ << "  if (threadIdx.x == 0) {\n";
  } else {
    code_ << "  if (blockInfo.blockInOp == 0 && threadIdx.x == 0) {\n";
  }
  code_ << "    __view(*" << param(src, op) << ", " << offsetExpr << ", "
        << elementSize << ", *" << param(dest, op) << ");\n"
        << "  }\n";
  emitBarrier();
}

void CompileCtx::emitCopy(
    ValueCP source,
    ValueCP dest,
    const std::string& destOffsetExpr,
    const std::string& cudaTypeName) {
  auto& op = *generatingOp_;
  code_ << "  __copy<" << cudaTypeName << ">(" << param(source, op) << ", "
        << "storage<" << cudaTypeName << ">(" << param(dest, op) << ")";
  if (!destOffsetExpr.empty()) {
    code_ << " + " << destOffsetExpr;
  }
  code_ << ", blockInfo);\n";
}

void CompileCtx::emitCode(std::string_view text) {
  code_ << text;
}

void CompileCtx::emitBarrier() {
  if (isSingleBlock_) {
    code_ << "  __syncthreads();\n";
  } else {
    auto barrierOffset = generatingOp_->allocateBarrier();
    code_ << "  opBarrier(blockInfo, " << barrierOffset << ");\n";
  }
}

void CompileCtx::addInclude(std::string_view header) {
  includes_.insert(std::string(header));
}

void CompileCtx::elementwiseExprImpl(
    NodeCP node,
    const std::unordered_set<ValueCP>& inputSet,
    const std::vector<ValueCP>& inputs,
    const KernelOperation& op,
    std::stringstream& ss,
    bool slowPath) {
  placed_.insert(node);
  generatingOp_->allNodes().insert(node);
  auto* meta = nodeMeta(node);
  TORCH_CHECK(
      meta && meta->elementwise, "Not an elementwise op: ", node->target());
  const auto& ew = *meta->elementwise;

  if (meta->generateCall) {
    std::vector<std::string> args;
    for (const auto& input : node->inputs()) {
      auto* value = input.value;
      if (inputSet.count(value)) {
        auto it = std::find(inputs.begin(), inputs.end(), value);
        TORCH_CHECK(
            it != inputs.end(), "Input value not found in inputs vector");
        auto valueIdx = it - inputs.begin();
        std::stringstream argSs;
        if (value->type().kind() != nativert::Type::Kind::Tensor) {
          argSs << "*b" << valueIdx;
        } else if (slowPath) {
          auto bitIdx = fastPathBitIndex_[valueIdx];
          argSs << "b" << valueIdx << "[complexIdx(isFastPath" << bitIdx / 32
                << " & (1 << " << bitIdx % 32 << "), " << param(value, op)
                << ", idx)]";
        } else {
          argSs << "b" << valueIdx << "[idx]";
        }
        args.push_back(argSs.str());
      } else {
        auto* producer = value->producer();
        TORCH_CHECK(producer, "Non-input value has no producer");
        if (producer->target() == "prim.ListPack") {
          placed_.insert(producer);
          generatingOp_->allNodes().insert(producer);
          for (const auto& listInput : producer->inputs()) {
            std::stringstream argSs;
            elementwiseExprImpl(
                listInput.value->producer(),
                inputSet,
                inputs,
                op,
                argSs,
                slowPath);
            args.push_back(argSs.str());
          }
        } else {
          std::stringstream argSs;
          elementwiseExprImpl(producer, inputSet, inputs, op, argSs, slowPath);
          args.push_back(argSs.str());
        }
      }
    }
    meta->generateCall(ss, node, std::move(args));
    return;
  }

  // Convert "--func" to "__func".
  std::string funcName = ew.functionName;
  TORCH_CHECK(
      funcName.size() >= 3 && funcName[0] == '-' && funcName[1] == '-',
      "Invalid elementwise function name: ",
      funcName);
  funcName[0] = '_';
  funcName[1] = '_';
  ss << funcName;
  auto ewPresenceParams = meta->hasPresentTemplateParams()
      ? presentTemplateParams(*meta, node)
      : std::string();
  if (!meta->typeTemplateParams.empty() || meta->hasDtypeTemplateParam ||
      !ewPresenceParams.empty()) {
    ss << "<";
    const auto& nodeInputs = node->inputs();
    bool firstTp = true;
    for (size_t i = 0; i < meta->typeTemplateParams.size(); ++i) {
      if (!firstTp) {
        ss << ", ";
      }
      firstTp = false;
      auto idx = meta->typeTemplateParams[i];
      ss << cudaType(nodeInputs[idx].value);
    }
    if (meta->hasDtypeTemplateParam) {
      if (!firstTp) {
        ss << ", ";
      }
      const auto* dtypeAttr = node->tryGetAttribute("dtype");
      TORCH_CHECK(dtypeAttr, node->target(), ": missing dtype attribute");
      ss << cudaTypeFromDtype(*dtypeAttr);
    }
    if (!ewPresenceParams.empty()) {
      if (!firstTp) {
        ss << ", ";
      }
      ss << ewPresenceParams;
    }
    ss << ">";
  }
  ss << "(";
  bool first = true;
  if (ew.hasIdxArg) {
    ss << "idx";
    first = false;
  }
  if (ew.hasSizeArg) {
    if (!first) {
      ss << ", ";
    }
    ss << "size";
    first = false;
  }
  forArguments(
      *meta,
      node,
      [&](size_t schemaIdx,
          ValueCP value,
          const nativert::Attribute* attr) {
        if (value) {
          if (!first) {
            ss << ", ";
          }
          first = false;
          bool isWhole = schemaIdx < meta->argumentMeta.size() &&
              meta->argumentMeta[schemaIdx].wholeTensor;
          if (isWhole) {
            ss << param(value, op);
          } else if (inputSet.count(value)) {
            auto it = std::find(inputs.begin(), inputs.end(), value);
            TORCH_CHECK(
                it != inputs.end(), "Input value not found in inputs vector");
            auto valueIdx = it - inputs.begin();
            if (value->type().kind() != nativert::Type::Kind::Tensor) {
              ss << "*b" << valueIdx;
            } else if (slowPath) {
              auto bitIdx = fastPathBitIndex_[valueIdx];
              ss << "b" << valueIdx << "[complexIdx(isFastPath"
                 << bitIdx / 32 << " & (1 << " << bitIdx % 32 << "), "
                 << param(value, op) << ", idx)]";
            } else {
              ss << "b" << valueIdx << "[idx]";
            }
          } else {
            auto* producer = value->producer();
            TORCH_CHECK(producer, "Non-input value has no producer");
            if (producer->target() == "prim.ListPack") {
              placed_.insert(producer);
              generatingOp_->allNodes().insert(producer);
              for (const auto& listInput : producer->inputs()) {
                elementwiseExprImpl(
                    listInput.value->producer(),
                    inputSet,
                    inputs,
                    op,
                    ss,
                    slowPath);
              }
            } else {
              elementwiseExprImpl(
                  producer, inputSet, inputs, op, ss, slowPath);
            }
          }
        } else if (attr) {
          if (!first) {
            ss << ", ";
          }
          first = false;
          if (std::holds_alternative<nativert::None>(attr->value)) {
            ss << "0";
          } else {
            auto off = op.attrOffset(node, attr->name);
            ss << "attr" << off;
          }
        }
      });
  ss << ")";
}

std::string CompileCtx::elementwiseExpr(
    NodeCP node,
    const KernelOperation& op,
    const std::vector<ValueCP>& inputs,
    bool slowPath) {
  addInclude("velox/experimental/torchwave/Elementwise.cuh");
  std::unordered_set<ValueCP> inputSet(inputs.begin(), inputs.end());
  std::stringstream ss;
  elementwiseExprImpl(node, inputSet, inputs, op, ss, slowPath);
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
  if (currentElementExpr_) {
    auto it = currentElementExpr_->altParamOffset.find(value);
    if (it != currentElementExpr_->altParamOffset.end()) {
      if (value->type().kind() == nativert::Type::Kind::Tensor) {
        return fmt::format("param<Tensor>(blockInfo, {})", it->second);
      }
      if (value->type().kind() == nativert::Type::Kind::TensorList) {
        return fmt::format("param<TensorList>(blockInfo, {})", it->second);
      }
      return fmt::format(
          "param<{}>(blockInfo, {})", cudaType(value), it->second);
    }
  }
  auto off = op.paramOffset(value);
  if (value->type().kind() == nativert::Type::Kind::Tensor) {
    return fmt::format("param<Tensor>(blockInfo, {})", off);
  }
  if (value->type().kind() == nativert::Type::Kind::TensorList) {
    return fmt::format("param<TensorList>(blockInfo, {})", off);
  }
  return fmt::format("param<{}>(blockInfo, {})", cudaType(value), off);
}

std::string CompileCtx::makeElementRef(ValueCP value, const KernelOperation& op)
    const {
  auto off = op.paramOffset(value);
  if (value->type().kind() == nativert::Type::Kind::Tensor) {
    return fmt::format(
        "elementRef<{}>(param<Tensor>(blockInfo, {}), idx, size)",
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

bool isAllViews(
    NodeCP node,
    const std::unordered_set<NodeCP>& placed,
    const std::unordered_set<NodeCP>& projectInputs,
    std::unordered_set<NodeCP>& visited) {
  if (!visited.insert(node).second) {
    return true;
  }
  auto* meta = Registry::metadata(node->target());
  bool isViewNode =
      (meta && meta->isView()) || node->target() == "torch.ops.aten.sym_size";
  if (!isViewNode) {
    return false;
  }
  for (const auto& input : node->inputs()) {
    auto* producer = input.value->producer();
    if (!producer || placed.count(producer) || projectInputs.count(producer)) {
      continue;
    }
    if (!isAllViews(producer, placed, projectInputs, visited)) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<CompiledNode> CompileCtx::compileNode(ProjectNode& project) {
  placedBeforeNode_ = placed_;
  inputs_ = &project.inputs();
  auto& nodes = project.nodes();
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto* node = nodes[i];
    if (node->target() == "prim.Input" || placed_.count(node)) {
      continue;
    }
    if (node->target() == "prim.ListUnpack") {
      auto* listValue = node->inputs()[0].value;
      auto* producer = listValue->producer();
      if (!producer || !standaloneNodes_.count(producer)) {
        placed_.insert(node);
        continue;
      }
    }
    auto sg = extractSubgraph(node, project.inputs(), placed_);
    auto it = projectOps_.find(sg);
    if (it != projectOps_.end()) {
      ops_.emplace_back(it->second, sg, ivalueStorage_);
      addDuplicateExtraBindings(
          ops_.back(), it->second->extraValues(), waveGraph_);
      // Map formal syncable and standalone value ids to actual ids.
      const auto& bindings = ops_.back().bindings();
      for (const auto& [formalId, actualId] : bindings) {
        if (waveGraph_.syncableValueIds().count(formalId)) {
          waveGraph_.addSyncableValueId(actualId);
        }
      }
      const auto& nodeMap = ops_.back().nodeMap();
      for (const auto& [formal, actual] : nodeMap) {
        if (standaloneNodes_.count(formal)) {
          standaloneNodes_.insert(actual);
        }
      }
    } else {
      NodeSet viewVisited;
      bool allViews =
          isAllViews(sg.root, placed_, project.inputs(), viewVisited);
      auto& config = WaveConfig::get();
      bool savedAllStandalone = config.allStandalone;
      if (allViews) {
        config.allStandalone = true;
      }
      auto* projectOp = makeProjectionOperation(sg);
      config.allStandalone = savedAllStandalone;
      if (projectOp) {
        projectOps_[sg] = projectOp;
        ops_.emplace_back(projectOp, sg, ivalueStorage_);
        addSelfExtraBindings(ops_.back(), projectOp->extraValues());
      }
    }
  }
  if (ops_.empty()) {
    return nullptr;
  }
  auto compositeKernel = std::make_unique<CompositeKernel>(
      std::move(opStorage_), std::move(kernelOpStorage_), includes_);
  auto invocation = std::make_unique<CompositeInvocation>();
  invocation->kernelInfo = compositeKernel->kernelInfo();
  invocation->kernel = std::move(compositeKernel);
  invocation->ops = std::move(ops_);
  invocation->ivalueStorage = std::move(ivalueStorage_);
  invocation->sequenceNumber = waveGraph_.nextCompositeInvocationId();
  placed_.insert(nodes.begin(), nodes.end());
  return std::make_unique<CompiledNode>(std::move(invocation));
}

} // namespace torch::wave
