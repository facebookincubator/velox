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
#include <fmt/format.h>
#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/Headers.h" // @manual: registers JIT headers via static init
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <gflags/gflags.h>
#include <algorithm>
#include <deque>
#include <sstream>
#include <vector>

// elt_trace is now WaveConfig::kernelDebugOutput

namespace torch::wave {

void eltTrace(std::stringstream& ss, std::string_view printf) {
  if (WaveConfig::get().kernelDebugOutput) {
    ss << "  if (threadIdx.x == 0) {printf(" << printf << ");}\n";
  }
}

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
      node, inputs, visited, [&](NodeCP n, const nativert::Attribute& attr) {
        auto iv = nativert::constantToIValue(attr.value);
        if (iv.isNone()) {
          return;
        }
        storage.push_back(std::move(iv));
      });
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
  for (auto it = storage.begin() + static_cast<std::ptrdiff_t>(startSize);
       it != storage.end();
       ++it) {
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

bool isParamPresent(NodeCP node, std::string_view name) {
  for (const auto& input : node->inputs()) {
    if (input.name == name) {
      return true;
    }
  }
  const auto* attr = node->tryGetAttribute(std::string(name));
  return attr && !std::holds_alternative<nativert::None>(attr->value);
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
    if (meta->functionSchema) {
      const auto& args = meta->functionSchema->arguments();
      for (size_t i = 0; i < args.size() && i < meta->argumentMeta.size();
           ++i) {
        if (meta->argumentMeta[i].hasPresentTemplateParam) {
          if (isParamPresent(left, args[i].name()) !=
              isParamPresent(right, args[i].name())) {
            return false;
          }
        }
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
        TORCH_CHECK(
            leftPos >= 0 &&
                static_cast<size_t>(leftPos) < leftSg.inputTypes.size() &&
                rightPos >= 0 &&
                static_cast<size_t>(rightPos) < rightSg.inputTypes.size(),
            "Input position out of range");
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

// Golden-ratio hash combiner
constexpr uint32_t kGoldenRatioHash = 0x9e3779b9;

void hashSubgraphNode(
    NodeCP node,
    const std::unordered_set<ValueCP>& inputs,
    size_t& hash) {
  auto h = std::hash<std::string_view>{}(node->target());
  hash ^= h + kGoldenRatioHash + (hash << 6) + (hash >> 2);
  // Include dtype attribute in hash if present.
  const auto* dtypeAttr = node->tryGetAttribute("dtype");
  if (dtypeAttr && std::holds_alternative<std::string>(dtypeAttr->value)) {
    auto dh = std::hash<std::string>{}(std::get<std::string>(dtypeAttr->value));
    hash ^= dh + kGoldenRatioHash + (hash << 6) + (hash >> 2);
  }
  auto* meta = nodeMeta(node);
  if (meta) {
    for (const auto& attrName : meta->templateAttrs) {
      const auto* attr = node->tryGetAttribute(attrName);
      if (attr) {
        auto ah = std::hash<std::string>{}(constantToString(attr->value));
        hash ^= ah + kGoldenRatioHash + (hash << 6) + (hash >> 2);
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

// Copies a TensorList output and its component element values from 'original'
// to 'node', preserving ids. Creates a prim.ListUnpack user so that
// getListElements() works on the copy. Adds mappings for the list and all
// elements to 'valueMap'.
nativert::Value* copyTensorList(
    const nativert::Value* original,
    nativert::Node* node,
    nativert::Graph* graph,
    std::unordered_map<ValueCP, nativert::Value*>& valueMap) {
  auto* newList =
      node->addOutput(std::string(original->name()), original->type());
  newList->setId(original->id());
  valueMap[original] = newList;
  auto* unpack = graph->insertNode(
      "prim.ListUnpack", {{"input", newList}}, node->metadata());
  auto elements = original->getListElements();
  for (auto* elem : elements) {
    auto* newElem = unpack->addOutput(std::string(elem->name()), elem->type());
    newElem->setId(elem->id());
    valueMap[elem] = newElem;
  }
  return newList;
}

// Copies a subgraph node into the compilation context, handling three cases:
// (1) variant node lives in the variant graph - copy and map its values,
// (2) variant node references the main graph - reuse existing mappings,
// (3) plain node - direct copy with fresh value allocation.
// The resulting subgraph shares execution frame slots (value IDs) with the
// original so that variant and non-variant paths are interchangeable at
// runtime.
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
                if (origOut->type().kind() ==
                    nativert::Type::Kind::TensorList) {
                  auto origElems = origOut->getListElements();
                  auto vrElems = vrOut->getListElements();
                  for (size_t j = 0; j < origElems.size() && j < vrElems.size();
                       ++j) {
                    valueMap[origElems.at(j)] =
                        const_cast<nativert::Value*>(vrElems.at(j));
                  }
                }
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
              if (origOut->type().kind() == nativert::Type::Kind::TensorList) {
                copyTensorList(origOut, newRoot, target, valueMap);
              } else {
                auto* newOut = newRoot->addOutput(
                    std::string(origOut->name()), origOut->type());
                newOut->setId(origOut->id());
                valueMap[origOut] = newOut;
              }
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
  TORCH_CHECK(result.root, "copyVariantNode returned null for subgraph root");

  // Copy prim.ListUnpack users of the root's TensorList outputs, unless the
  // copied value in the new graph already has a ListUnpack user.
  for (const auto* outVal : sg.root->outputs()) {
    if (outVal->type().kind() != nativert::Type::Kind::TensorList ||
        sg.root->target() == "prim.ListPack") {
      continue;
    }
    auto mappedIt = valueMap.find(outVal);
    if (mappedIt != valueMap.end()) {
      bool alreadyUnpacked = false;
      for (auto* u : mappedIt->second->users()) {
        if (u->target() == "prim.ListUnpack") {
          alreadyUnpacked = true;
          break;
        }
      }
      if (alreadyUnpacked) {
        continue;
      }
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
      auto* newUnpack = graph->insertNode(
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
  opStorage_.push_back(std::make_unique<ProjectOperation>(sg));
  auto* projectOp = opStorage_.back().get();

  auto& config = WaveConfig::get();
  auto& types = waveGraph_.types();
  bool forceCg = config.isCg.has_value() && *config.isCg;
  bool forceSingleBlock =
      config.useSingleBlock.has_value() && *config.useSingleBlock;

  if (forceCg) {
    // Only generate the CG grid variant.
    NodeSet cgVisited;
    bool hasCgVariant = anyReachable(
        *sg.root,
        placed_,
        [&types](const Metadata& m, NodeCP node) {
          return (m.cgVariant || m.multiBlockReturnBarrier) &&
              !m.isStandalone(node, types);
        },
        cgVisited);
    if (hasCgVariant) {
      auto cgSg = variantSubgraph(sg, VariantMode::kCG);
      setIsCgGrid(true);
      projectOp->grid_ = makeGrid(cgSg.root);
      setIsCgGrid(false);
    } else {
      auto multiSg = variantSubgraph(sg, VariantMode::kMulti);
      projectOp->grid_ = makeGrid(multiSg.root);
    }
  } else if (forceSingleBlock) {
    // Only generate the single-block grid variant.
    auto singleSg = variantSubgraph(sg, VariantMode::kSingle);
    setIsSingleBlock(true);
    projectOp->grid_ = makeGrid(singleSg.root);
    setIsSingleBlock(false);
  } else {
    // Generate all applicable variants and let the runtime choose.
    NodeSet visited;
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

      NodeSet cgVisited;
      bool hasCgVariant = anyReachable(
          *sg.root,
          placed_,
          [&types](const Metadata& m, NodeCP node) {
            return (m.cgVariant || m.multiBlockReturnBarrier) &&
                !m.isStandalone(node, types);
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
  }
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
static bool isSizeArg(std::string_view inputName, const Metadata* meta) {
  if (!meta || !meta->functionSchema) {
    return false;
  }
  const auto& schemaArgs = meta->functionSchema->arguments();
  for (size_t i = 0; i < schemaArgs.size(); ++i) {
    if (schemaArgs[i].name() == inputName) {
      auto ordinal = static_cast<int32_t>(i);
      for (const auto& ret : meta->returnMeta) {
        for (auto sizeOrd : ret.sizeArgs.ordinal) {
          if (sizeOrd == ordinal) {
            return true;
          }
        }
      }
      return false;
    }
  }
  return false;
}

Context CompileCtx::placeKernels(NodeCP node, Context /*context*/) {
  if (node->target() == "prim.ListUnpack" && !node->inputs().empty()) {
    auto* inputValue = node->inputs()[0].value;
    auto* producer = inputValue->producer();
    if (producer && !placed_.count(producer)) {
      auto* producerMeta = nodeMeta(producer);
      bool producerIsStandalone = !producerMeta ||
          producerMeta->isStandalone(producer, types_) || allStandalone_;
      auto result = placeKernels(producer, Context::kStandalone);
      if (!producerIsStandalone) {
        return result;
      }
    }
    pushdownStandalone(node);
    return Context::kStandalone;
  }

  auto* meta = nodeMeta(node);
  auto thisContext =
      (!meta || meta->isStandalone(node, types_) || allStandalone_)
      ? Context::kStandalone
      : Context::kFused;
  std::vector<NodeCP> standaloneInputs;
  std::vector<NodeCP> fusedInputs;

  auto placeInput = [&](ValueCP value, bool isScalarSize) {
    auto* producer = value->producer();
    if (!producer || placed_.count(producer) ||
        (inputs_ && inputs_->count(producer))) {
      return;
    }
    auto inputContext = placeKernels(producer, thisContext);
    if (inputContext == Context::kFused) {
      if (isScalarSize) {
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
    bool isScalarSize = isSizeArg(node->inputs()[i].name, meta) &&
        inputKind != nativert::Type::Kind::Tensor &&
        inputKind != nativert::Type::Kind::TensorList;
    if (thisContext == Context::kFused && producer &&
        producer->target() == "prim.ListPack") {
      for (const auto& listInput : producer->inputs()) {
        placeInput(listInput.value, isScalarSize);
      }
    } else {
      placeInput(inputValue, isScalarSize);
    }
  }

  if (thisContext == Context::kStandalone) {
    for (auto* fused : fusedInputs) {
      pushdownFused(fused);
    }
    pushdownStandalone(node);
    return thisContext;
  }
  if (meta->isKernelBreak(isSingleBlock_, isCgGrid_)) {
    pushdownFused(node);
    return Context::kFusedBreak;
  }
  return Context::kFused;
}

void CompileCtx::pushdownStandalone(NodeCP node) {
  auto* actualNode = originalFromVariant(node);
  Launch launch(actualNode, types_, waveGraph_);
  placeKernelLaunch(std::move(launch));
  placed_.insert(node);
  standaloneNodes_.insert(actualNode);
}

void CompileCtx::fillConstantIndices(const Subgraph& sg, Launch& launch) {
  std::unordered_set<ValueCP> sgInputSet(sg.inputs.begin(), sg.inputs.end());
  std::unordered_set<NodeCP> attrVisited;
  forEachSortedAttribute(
      sg.root,
      sgInputSet,
      attrVisited,
      [&](NodeCP n, const nativert::Attribute& attr) {
        if (std::holds_alternative<nativert::None>(attr.value)) {
          auto* meta = Registry::metadata(n->target());
          if (meta && meta->isPresenceTemplateParam(attr.name)) {
            return;
          }
          TORCH_CHECK(
              false,
              "Constant attribute '",
              attr.name,
              "' is None in node: ",
              n->toString());
        }

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

  auto kernelOp = generateFused(sg);
  kernelOpStorage_.push_back(std::move(kernelOp));
  launch.op = kernelOpStorage_.back().get();
  launch.values.assign(
      launch.op->orderedInputs().begin(), launch.op->orderedInputs().end());

  fillConstantIndices(sg, launch);
  placeKernelLaunch(std::move(launch));
  placed_.insert(node);
}

std::unique_ptr<KernelOperation> CompileCtx::generateFused(const Subgraph& sg) {
  auto op = std::make_unique<KernelOperation>(sg, nextOpCode(), *this);
  generatingOp_ = op.get();
  generateFusedInner(sg);
  for (auto& [type, count] : typeTemps_) {
    for (int32_t i = 0; i < count; ++i) {
      declarations_ << "  " << type << " temp_" << type << "_" << i << ";\n";
    }
  }
  typeTemps_.clear();
  tempNames_.clear();
  tempUseLog_.clear();
  helperVarDeps_.clear();
  std::stringstream combined;
  combined << declarations_.str() << code_.str();
  declarations_.str("");
  declarations_.clear();
  op->setCode(combined);
  code_.str("");
  code_.clear();
  auto helpers = outOfLineFunctions_.str();
  outOfLineFunctions_.str("");
  outOfLineFunctions_.clear();
  if (!helpers.empty()) {
    op->setHelperCode(std::move(helpers));
  }
  return op;
}

void CompileCtx::generateFusedInner(const Subgraph& sg) {
  auto resultSpecs = outputSpecs(sg.root);
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
  if (launch.op) {
    const char* mode = isCgGrid_ ? "cg" : (isSingleBlock_ ? "single" : "multi");
    launch.op->setLabel(
        fmt::format(
            "{}.{}.{}.{} {} distinct",
            currentNodeId_,
            currentExprOrdinal_,
            mode,
            targetLevel,
            numDistinctOps_));
  }
  grid_[targetLevel].push_back(std::move(launch));
}

void CompileCtx::generateElementwiseBorderImpl(
    NodeCP node,
    const std::unordered_set<ValueCP>& opInputs,
    NodeSet& visited) {
  auto* consumerMeta = nodeMeta(node);

  auto getArgMeta = [&](size_t idx) -> const ArgumentMeta* {
    if (consumerMeta && idx < consumerMeta->argumentMeta.size()) {
      return &consumerMeta->argumentMeta[idx];
    }
    return nullptr;
  };

  auto shouldSkip = [&](ValueCP value, NodeCP producer) {
    return !producer || placed_.count(producer) || opInputs.count(value) ||
        !visited.insert(producer).second;
  };

  auto emitBorder = [&](NodeCP producer) {
    auto specs = outputSpecs(producer);
    fusedCode(producer, specs);
  };

  // An elementwise producer can be part of the elementwise tree only if
  // its result is in register. Ops with non-register return (like
  // index_put_elt_*) write to a whole tensor as a side effect and must
  // be treated as border — they produce a value in memory, not a
  // register scalar.
  auto isElementwiseProducer = [&](NodeCP producer) {
    auto* meta = nodeMeta(producer);
    if (!meta || !meta->elementwise) {
      return false;
    }
    if (!meta->returnMeta.empty() && !meta->returnMeta[0].isRegister) {
      return false;
    }
    return true;
  };

  for (size_t inputIdx = 0; inputIdx < node->inputs().size(); ++inputIdx) {
    auto* value = node->inputs()[inputIdx].value;
    auto* producer = value->producer();
    if (shouldSkip(value, producer)) {
      continue;
    }
    if (producer->target() == "prim.ListPack") {
      auto* am = getArgMeta(inputIdx);
      bool isRegister = am && am->isRegister;
      for (auto& listInput : producer->inputs()) {
        auto* listProducer = listInput.value->producer();
        if (shouldSkip(listInput.value, listProducer)) {
          continue;
        }
        if (isRegister && isElementwiseProducer(listProducer)) {
          generateElementwiseBorderImpl(listProducer, opInputs, visited);
        } else {
          emitBorder(listProducer);
        }
      }
      continue;
    }
    auto* am = getArgMeta(inputIdx);
    if (isElementwiseProducer(producer) && !(am && am->wholeTensor)) {
      generateElementwiseBorderImpl(producer, opInputs, visited);
    } else {
      emitBorder(producer);
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
  auto* meta = nodeMeta(node);
  TORCH_CHECK(meta, "No metadata for: ", node->target());

  // Find the size argument.
  TORCH_CHECK(
      !meta->returnMeta.empty() &&
          !meta->returnMeta[0].sizeArgs.ordinal.empty(),
      "functionLoop requires sizeArgs");
  auto sizeArgIdx = meta->returnMeta[0].sizeArgs.ordinal[0];
  auto* sizeValue = node->inputs()[sizeArgIdx].value;

  // Add shared declaration for size.
  op.addSharedDeclaration("  __shared__ uint32_t size;\n");

  auto inSpecs = inputSpecs(node);
  auto resultSpecs = outputSpecs(node);

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
  code_ << "  " << makeCall(node, inSpecs, resultSpecs) << "\n";
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

// Generates device code for a subgraph that compiles to a single kernel.
// Dispatch order: special-form ops, view ops (no-op if shapes known at
// launch, otherwise codegen), elementwise with register inputs (merged into
// the elementwise loop to avoid passing through memory), non-elementwise
// ops that accept registers, and finally plain device-function calls per
// metadata.
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
      auto prodSpecs = outputSpecs(producer);
      fusedCode(producer, prodSpecs);
    }
    std::unordered_set<ValueCP> visited;
    if (isSizeSetInThisOp(viewInput, visited) && !meta->deviceFunc.empty()) {
      auto inSpecs = inputSpecs(node);
      code_ << "  " << makeCall(node, inSpecs, resultSpecs) << "\n";
    }
    placed_.insert(node);
    generatingOp_->allNodes().insert(node);
    return;
  }

  if (meta->elementwise) {
    generateElementwiseBorder(node);
    Subgraph sg = extractSubgraph(node, *inputs_, placed_);
    TORCH_CHECK(
        !resultSpecs.empty(), "resultSpecs is empty for elementwise node");
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
        if (producer->target() == "prim.ListPack") {
          placed_.insert(producer);
          for (const auto& lpInput : producer->inputs()) {
            auto* lpValue = lpInput.value;
            auto* lpProducer = lpValue->producer();
            if (lpProducer && !placed_.count(lpProducer)) {
              std::vector<ResultSpec> lpSpecs;
              ResultSpec rs;
              rs.value = lpValue;
              lpSpecs.push_back(rs);
              fusedCode(lpProducer, lpSpecs);
            }
          }
        } else {
          auto prodSpecs = outputSpecs(producer);
          fusedCode(producer, prodSpecs);
        }
      }
    }
  }

  const auto& inputs = node->inputs();

  bool multiBlockSingleOp = !isSingleBlock_ && meta->singleBlockIfFused;

  if (!meta->hasRegisterInputs()) {
    auto inSpecs = inputSpecs(node);
    if (callNeedsBarrier(node)) {
      emitBarrier();
    }
    if (multiBlockSingleOp) {
      emitBarrier();
      code_ << "  if (blockInfo.blockInOp == 0) {\n";
    }
    code_ << "  " << makeCall(node, inSpecs, resultSpecs) << "\n";
    if (multiBlockSingleOp) {
      code_ << "  }\n";
      emitBarrier();
    }
    if ((isCgGrid_ || isSingleBlock_) && meta->multiBlockReturnBarrier) {
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
    if (callNeedsBarrier(node)) {
      emitBarrier();
    }
    if (multiBlockSingleOp) {
      emitBarrier();
      code_ << "  if (blockInfo.blockInOp == 0) {\n";
    }
    functionLoop(node);
    if (multiBlockSingleOp) {
      code_ << "  }\n";
      emitBarrier();
    }
    if ((isCgGrid_ || isSingleBlock_) && meta->multiBlockReturnBarrier) {
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
  if (callNeedsBarrier(node)) {
    emitBarrier();
  }
  if (multiBlockSingleOp) {
    emitBarrier();
    code_ << "  if (blockInfo.blockInOp == 0) {\n";
  }
  if (!subgraphs.empty()) {
    generateElementwise(subgraphs, ewResultSpecs, callStmt, meta->hasBarrier);
  }
  if (multiBlockSingleOp) {
    code_ << "  }\n";
    emitBarrier();
  }
  if (isCgGrid_ && meta->multiBlockReturnBarrier) {
    emitBarrier();
  }
  placed_.insert(node);
  generatingOp_->allNodes().insert(node);
}

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

namespace {

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
      elements.push_back("*" + param(listInput.value, op));
    }
  } else {
    TORCH_CHECK(attr, "ScalarList argument has no value or attribute");
    auto* vec = std::get_if<std::vector<int64_t>>(&attr->value);
    TORCH_CHECK(
        vec, "ScalarList attribute must be vector<int64_t>: ", node->target());
    for (auto v : *vec) {
      elements.push_back(std::to_string(v));
    }
  }
  auto numElements = elements.size();
  auto allocSize = sizeof(ScalarList) + sizeof(int64_t) * numElements;
  auto off = op.allocAltParam(static_cast<int32_t>(allocSize));
  auto varName = "l" + std::to_string(argOrdinal);
  std::stringstream setup;
  setup << "  ScalarList* " << varName << " = param<ScalarList>(blockInfo, "
        << off << ");\n"
        << "  if (threadIdx.x == 0) {\n"
        << "    " << varName << "->size = " << numElements << ";\n"
        << "    " << varName << "->data = reinterpret_cast<int64_t*>("
        << varName << " + 1);\n";
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
      [&](size_t schemaIdx, ValueCP value, const nativert::Attribute* attr) {
        // Check for SymInt list: Value with SymIntList type or attribute with
        // vector<int64_t>.
        bool isSymIntList =
            (value &&
             value->type().kind() == nativert::Type::Kind::SymIntList) ||
            (attr && std::holds_alternative<std::vector<int64_t>>(attr->value));
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
    std::string decl = "  __shared__ ";
    decl += type;
    decl += " ";
    decl += name;
    decl += ";\n";
    op.addSharedDeclaration(decl);
    comma();
    ss << name;
  }

  // Dynamic shared declarations: type from input dtype, name suffixed by type.
  // kTypeFromDtype means use the resolved dtype attribute instead of an input.
  for (const auto& [ordinal, baseName] : meta->dynamicSharedDecls) {
    std::string tp;
    std::string suffix;
    if (ordinal != Metadata::kTypeFromDtype) {
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
    std::string dynDecl = "  __shared__ ";
    dynDecl += tp;
    dynDecl += " ";
    dynDecl += varName;
    dynDecl += ";\n";
    op.addSharedDeclaration(dynDecl);
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
    generatingOp_->setAlwaysSingleBlock(true);
  } else {
    auto barrierOffset = generatingOp_->allocateBarrier();
    code_ << "  opBarrier(blockInfo, " << barrierOffset << ");\n";
  }
  for (auto* node : generatingOp_->allNodes()) {
    for (auto* output : node->outputs()) {
      preBarrierValues_.insert(output);
    }
  }
}

bool CompileCtx::callNeedsBarrier(NodeCP node) {
  auto* meta = nodeMeta(node);
  if (!meta) {
    return false;
  }
  const auto& inputs = node->inputs();
  for (size_t i = 0; i < inputs.size() && i < meta->argumentMeta.size(); ++i) {
    if (!meta->argumentMeta[i].randomAccess) {
      continue;
    }
    auto* value = inputs[i].value;
    auto* producer = value->producer();
    if (producer && generatingOp_->allNodes().count(producer) &&
        !preBarrierValues_.count(value)) {
      return true;
    }
  }
  return false;
}

void CompileCtx::addInclude(std::string_view header) {
  includes_.insert(std::string(header));
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
    case nativert::Type::Kind::None:
      // Type not set during export.  Recover from TensorMeta if available,
      // otherwise fall back to int32_t (covers integer scalar attributes
      // like dim, index whose Kind was not annotated).
      if (value->id() < types_.types.size() && types_.types[value->id()]) {
        return cudaTypeString(types_.types[value->id()]->dtype());
      }
      LOG(WARNING) << "Value " << value->name()
                   << " has Kind::None with no TensorMeta, defaulting to "
                      "int32_t";
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

std::string CompileCtx::useTemp(ValueCP value) {
  auto type = cudaType(value);
  auto& names = tempNames_[type];
  std::string name;
  if (!names.empty()) {
    name = std::move(names.back());
    names.pop_back();
  } else {
    auto& counter = typeTemps_[type];
    name = fmt::format("temp_{}_{}", type, counter);
    ++counter;
  }
  tempUseLog_.emplace_back(type, name);
  return name;
}

void CompileCtx::tempDone(ValueCP value, const std::string& name) {
  tempNames_[cudaType(value)].push_back(name);
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
  currentNodeId_ = project.id();
  numDistinctOps_ = 0;
  auto& nodes = project.nodes();
  for (size_t i = 0; i < nodes.size(); ++i) {
    currentExprOrdinal_ = static_cast<int32_t>(i);
    auto* node = nodes[i];
    if (node->target() == "prim.Input" || placed_.count(node)) {
      continue;
    }
    if (node->target() == "prim.ListUnpack") {
      auto* listValue = node->inputs()[0].value;
      auto* producer = listValue->producer();
      // A prim.listunpack over a placed fused op is a no-op. The fused code
      // assigns the tensors in the list directly.
      if (producer) {
        auto* producerMeta = Registry::metadata(producer->target());
        if (producerMeta && !producerMeta->isStandalone(producer, types_)) {
          placed_.insert(node);
          continue;
        }
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
      bool savedAllStandalone = allStandalone_;
      if (allViews) {
        allStandalone_ = true;
      }
      auto* projectOp = makeProjectionOperation(sg);
      allStandalone_ = savedAllStandalone;
      ++numDistinctOps_;
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
  auto invocation = std::make_unique<CompositeInvocation>(
      std::move(compositeKernel),
      std::move(ops_),
      std::move(ivalueStorage_),
      waveGraph_.nextCompositeInvocationId());
  placed_.insert(nodes.begin(), nodes.end());
  return std::make_unique<CompiledNode>(std::move(invocation));
}

} // namespace torch::wave
