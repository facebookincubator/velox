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
#include <iostream>
#include "velox/experimental/torchwave/AllocGroup.h"
#include "velox/experimental/torchwave/Cat.h"
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
          auto* meta = Registry::metadata(n->target());
          if (meta && meta->isPresenceTemplateParam(attr.name)) {
            return;
          }
          // A None constant attribute does not occupy a constant slot (e.g.
          // searchsorted's side/sorter, repeat_interleave's output_size); skip
          // it rather than erroring, so later nodes' constant indices are not
          // shifted past the end of the constants vector.
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
      // A None-typed input slot represents an absent optional argument (e.g.
      // clamp(self, min=None, max=5)).  It must NOT count as present, otherwise
      // the op selects the wrong presence template (e.g. __clamp<true, true>
      // reading a None->0 min) and dedup merges it with present-arg ops.
      return input.value->type().kind() != nativert::Type::Kind::None;
    }
  }
  const auto* attr = node->tryGetAttribute(std::string(name));
  return attr && !std::holds_alternative<nativert::None>(attr->value);
}

// Counts the SymIntList (vector<int64_t>) attributes of 'node'.
size_t numIntListAttributes(NodeCP node) {
  size_t count = 0;
  for (const auto& attr : node->attributes()) {
    if (std::holds_alternative<std::vector<int64_t>>(attr.value)) {
      ++count;
    }
  }
  return count;
}

// True if 'left' and 'right' carry the same SymIntList attributes. Unlike
// scalar constants, these never reach the constant area (forEachSortedAttribute
// skips them): a fused op materializes their literal values inline, and a
// factory op's 'size' also fixes the enclosing expression's shape (SizeExpr::
// constShapes). Both are baked into the ProjectOperation, so two nodes that
// differ in one are not interchangeable.
bool intListAttributesMatch(NodeCP left, NodeCP right) {
  if (numIntListAttributes(left) != numIntListAttributes(right)) {
    return false;
  }
  for (const auto& attr : left->attributes()) {
    const auto* values = std::get_if<std::vector<int64_t>>(&attr.value);
    if (!values) {
      continue;
    }
    const auto* rightAttr = right->tryGetAttribute(attr.name);
    if (!rightAttr) {
      return false;
    }
    const auto* rightValues =
        std::get_if<std::vector<int64_t>>(&rightAttr->value);
    if (!rightValues || *rightValues != *values) {
      return false;
    }
  }
  return true;
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
  if (!intListAttributesMatch(left, right)) {
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
  // SymIntList attributes are baked into the generated code as literals (see
  // emitScalarListSetup) and into the subgraph's shape expressions, so they are
  // part of a node's identity (see intListAttributesMatch): two otherwise
  // identical ops differing in one must not share a deduplicated kernel. Scalar
  // attributes go through the param area and need no hashing.
  for (const auto& attr : node->attributes()) {
    if (!std::holds_alternative<std::vector<int64_t>>(attr.value)) {
      continue;
    }
    auto ah = std::hash<std::string>{}(
        attr.name + "=" + constantToString(attr.value));
    hash ^= ah + kGoldenRatioHash + (hash << 6) + (hash >> 2);
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
          // An original output can be produced by any node of the variant
          // chain, not just variantRoot: a multi-stage expansion (e.g.
          // masked_select_jagged -> head/scatter/lengths) yields the values
          // list from the scatter stage and new_sizes from the lengths stage.
          // Collect every chain-produced output (walking from variantRoot via
          // producers, stopping at already-mapped external inputs, as
          // mapVariantChain does) and match original outputs by id against all
          // of them.
          std::vector<nativert::Value*> chainOutputs;
          {
            std::unordered_set<NodeCP> seen;
            std::function<void(NodeCP)> collect = [&](NodeCP chainNode) {
              if (!seen.insert(chainNode).second) {
                return;
              }
              for (const auto* out : chainNode->outputs()) {
                chainOutputs.push_back(const_cast<nativert::Value*>(out));
              }
              for (const auto& input : chainNode->inputs()) {
                if (valueMap.count(input.value)) {
                  continue;
                }
                if (auto* producer = input.value->producer()) {
                  collect(producer);
                }
              }
            };
            collect(variantRoot);
          }
          for (auto* origOut : node->outputs()) {
            for (auto* vrOut : chainOutputs) {
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
  opCarvesAConcat_ = false;
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
  // grid_ and singleBlockGrid_ are two complete alternative plans for the same
  // ProjectOp; at runtime the executor picks one wholesale based on the
  // grid-choice kernel's element count (see CompiledOp.cpp). Launches within a
  // parallel step are data-independent, so their relative order can differ
  // between the two grids (e.g. fusion can reorder a standalone view relative
  // to sibling kernels). Match the grid-choice kernel across the two grids by
  // node identity -- the original root node each kernel op was built from --
  // rather than by position, which would spuriously flag a standalone in one
  // grid against a kernel in the other when the orders diverge.
  std::unordered_map<NodeCP, KernelOperation*> sbKernelByRoot;
  for (auto& step : projectOp->singleBlockGrid_) {
    for (auto& launch : step) {
      if (launch.op && launch.op->expr()) {
        sbKernelByRoot[originalFromVariant(launch.op->expr())] = launch.op;
      }
    }
  }
  for (auto& step : projectOp->grid_) {
    for (auto& launch : step) {
      if (!launch.op || !launch.op->expr()) {
        continue;
      }
      auto it = sbKernelByRoot.find(originalFromVariant(launch.op->expr()));
      if (it != sbKernelByRoot.end()) {
        launch.op->setIsGridChoice(true);
        it->second->setIsGridChoice(true);
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
          // A launch that carries its own parameters -- one of the copies that
          // fill a wide concat's bands -- names values the op itself does not,
          // since they all share one op. Missing them here leaves them
          // unduplicated, so every instance of a deduplicated project op would
          // write the same band and the last one would win.
          for (auto* value : launch.values) {
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
  gridWrittenAt_.clear();
  gridRealizedAt_.clear();
}

LaunchGrid CompileCtx::makeGrid(NodeCP node) {
  newGrid();
  concatPushdownSkips_.clear();
  auto result = placeKernels(node, Context::kTop);
  if (result == Context::kFused) {
    pushdownFused(node);
  }
  // Only the reasons are gathered under kTiming, so without it every line here
  // would be a bare concat id. Now that every wide concat takes this path that
  // is a line per concat on every compile.
  if ((WaveConfig::get().trace & WaveConfig::kTiming) != 0) {
    for (const auto& [concatId, reasons] : concatPushdownSkips_) {
      std::cout << "  concat fill pushdown %" << concatId << ":";
      for (const auto& [reason, howMany] : reasons) {
        std::cout << " [" << howMany << " x " << reason << "]";
      }
      std::cout << std::endl;
    }
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

// True if 'producer' is a non-elementwise op that computes (part of) its output
// size on device (shapeSetOnDevice). Such an op cannot fuse into an elementwise
// consumer whose shape depends on that size without a kernel boundary in
// between: the elementwise loop bound would read the size before the producer's
// kernel writes it, giving a wrong (too-long) result. The multi-kernel
// masked_select_final reserves its size from a host scalar (shapeSetOnDevice is
// false there), so it is not caught here and still fuses.
static bool setsSizeOnDevice(NodeCP producer) {
  const auto* meta = nodeMeta(producer);
  if (!meta || meta->elementwise) {
    return false;
  }
  for (const auto& ret : meta->returnMeta) {
    if (ret.shapeSetOnDevice) {
      return true;
    }
  }
  return false;
}

// True when the node's extent is only known once its reserve function has run,
// so nothing upstream of that can compute it. A reserve that just returns an
// input's shape (ArgumentMeta::shapeFromInput) does not count -- that extent is
// the input's, and whoever needs it can read the input instead.
static bool sizeNeedsReserve(NodeCP producer) {
  const auto* meta = nodeMeta(producer);
  // An elementwise output has the extent of its inputs whatever its reserve
  // does with it, so there is never anything to wait for. Same exclusion
  // setsSizeOnDevice makes, and for the same reason.
  if (!meta || meta->elementwise) {
    return false;
  }
  // More than one output means the op is half of a split (a head feeding a
  // final), whose two parts are placed as a pair. Ending the kernel between
  // them is not this pass's to do -- inputFromPreviousKernel already arranges
  // where they break -- and doing it anyway produces wrong values.
  if (meta->returnMeta.size() != 1) {
    return false;
  }
  const auto& ret = meta->returnMeta.front();
  return ret.reserveShape != nullptr && ret.shapeFromInput < 0;
}

// True when a node other than 'reader' overwrites 'value' in place (it is that
// node's mutatesArg "self"). Such a write is invisible to a producer walk: the
// writer's own output is a different value, so 'value' still names whoever
// created the storage -- an earlier kernel, or nothing at all for a graph
// input.
static bool isWrittenInPlace(ValueCP value, NodeCP reader) {
  if (!value) {
    return false;
  }
  for (auto* user : value->users()) {
    if (user == reader) {
      continue;
    }
    const auto* meta = nodeMeta(user);
    if (!meta || !meta->mutatesArg.has_value()) {
      continue;
    }
    auto ordinal = static_cast<size_t>(*meta->mutatesArg);
    const auto& userInputs = user->inputs();
    if (ordinal < userInputs.size() && userInputs[ordinal].value == value) {
      return true;
    }
  }
  return false;
}

// Follows 'value' back through view producers to the value whose storage it
// aliases. A view writes nothing of its own -- what a consumer reading through
// it actually reads is the base -- so both ordering questions below ("does this
// come from memory written in this kernel" and "has that write been
// synchronized") have to be asked about the base rather than about the view.
//
// The walk stops short of a base that something overwrites in place. Looking
// through to such a base answers the wrong question: its producer is whoever
// created the storage, which says nothing about the in-place write that may be
// landing in this very kernel. Stopping leaves the view itself as the answer,
// which keeps the conservative pre-view behaviour for that edge.
static ValueCP viewBase(ValueCP value) {
  // Deeper than any real view chain; a bound rather than trusting the graph to
  // be acyclic here.
  constexpr int32_t kMaxViewDepth = 16;
  for (int32_t depth = 0; value != nullptr && depth < kMaxViewDepth; ++depth) {
    auto* producer = value->producer();
    if (!producer) {
      return value;
    }
    const auto* meta = nodeMeta(producer);
    if (!meta || !meta->isView()) {
      return value;
    }
    auto ordinal = static_cast<size_t>(*meta->viewOfArg);
    if (ordinal >= producer->inputs().size()) {
      return value;
    }
    auto* base = producer->inputs()[ordinal].value;
    if (isWrittenInPlace(base, producer)) {
      return value;
    }
    value = base;
  }
  return value;
}

// True when the consumer described by 'consumerMeta' reads its input
// 'inputIdx' from the memory output of a 'producer' that would fuse into the
// consumer's kernel. Any such edge needs a barrier: the producer's writes come
// from every block, so a consumer block reading them without one can see a
// partially written tensor. Codegen orders the edge with an intra-kernel
// opBarrier (see callNeedsBarrier), which in multi-kernel mode forces a
// cooperative, whole-grid-resident launch; ending the producer's kernel avoids
// that. What decides it is how the value reaches the consumer, not what kind of
// op produced it: through memory when the input is a whole tensor
// (argumentMeta wholeTensor) or when the producer's output for this value is
// not a register. A value handed over in a register never leaves the block and
// needs no barrier. Producers that already run standalone, or that carry
// multiBlockReturnBarrier and so end their own kernel via isKernelBreak, are
// ordered by their own launch boundary. 'inputValue'/'producer' must already be
// resolved through any views (see viewBase): a view of a value written before
// this kernel needs no ordering at all, and a view of one written inside it has
// to end the kernel at the writer, not at the view, which materializes nothing.
static bool readsFusedElementwiseProducerFromMemory(
    int32_t inputIdx,
    NodeCP consumer,
    const Metadata* consumerMeta,
    ValueCP inputValue,
    NodeCP producer,
    const ValueTypes& types,
    bool allStandalone) {
  if (!producer) {
    return false;
  }
  const auto* producerMeta = nodeMeta(producer);
  if (!producerMeta || producerMeta->isStandalone(producer, types) ||
      allStandalone) {
    return false;
  }
  const auto* argMeta =
      argMetaForInput(consumerMeta, consumer, static_cast<size_t>(inputIdx));
  const bool wholeTensorArg = argMeta && argMeta->wholeTensor;
  // Ask about the output the consumer actually reads, not the producer's
  // first one. A producer with several outputs can pass some in registers and
  // materialize others, so keying on returnMeta[0] would break the producer's
  // kernel for a register-passed input (or fuse through a materialized one).
  // Elementwise ops are usually single-output, in which case this resolves to
  // index 0 anyway.
  size_t outputIdx = 0;
  const auto& producerOutputs = producer->outputs();
  for (size_t i = 0; i < producerOutputs.size(); ++i) {
    if (producerOutputs[i] == inputValue) {
      outputIdx = i;
      break;
    }
  }
  // Only an explicit isRegister says the value reaches the consumer without
  // going through memory. An op that declares no returnMeta for this output
  // materializes it, so default to memory rather than to the register case.
  const bool nonRegisterProducer =
      outputIdx >= producerMeta->returnMeta.size() ||
      !producerMeta->returnMeta[outputIdx].isRegister;
  return wholeTensorArg || nonRegisterProducer;
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
      // A non-elementwise producer that sets its output size on device (e.g.
      // masked_select's single-block/cg form) must not fuse into an elementwise
      // consumer whose loop bound is that size: the elementwise would read the
      // size before the producer's kernel writes it, giving a too-long result.
      // Force a kernel boundary so the size is materialized first. The
      // multi-kernel masked_select_final reserves its size from a host scalar
      // (shapeSetOnDevice is false), so it still fuses with the elementwise.
      bool consumerIsElementwise = meta && meta->elementwise != nullptr;
      if (isScalarSize ||
          (consumerIsElementwise && setsSizeOnDevice(producer))) {
        pushdownFused(producer);
      } else {
        fusedInputs.push_back(producer);
      }
    }
  };

  // Place the previous-kernel (ordering) input first, if any. Its producer is
  // its own kernel stage, separated from this op by a kernel break; placing it
  // first materializes any inputs it shares with this op (e.g. a scan head that
  // reads the same data tensor as its final stage), so the loop below then
  // finds those producers already placed. When the producer's isKernelBreak is
  // false (e.g. a single-block scan head) it returns kFused and would be
  // collected into this op's fusedInputs -- which the kFusedBreak path below
  // discards -- so push it directly instead, otherwise this op reads a buffer
  // its producer never wrote (an orphaned scan stage).
  if (meta && meta->inputFromPreviousKernel.has_value()) {
    breakProducerIntoOwnKernel(
        node->inputs()[meta->inputFromPreviousKernel.value()]
            .value->producer());
  }

  // Concat operand copies, held back until the concat itself has a step. A
  // copy fills a band of the concat result, and that band exists only once the
  // allocation group has carved it -- which cannot happen before the step whose
  // head lays the result out. Placed as they are found, they would go one step
  // after their own source instead, which for an operand that was ready early
  // is steps ahead of the group. See the emission after the pushdown below.
  std::vector<std::pair<ValueCP, ValueCP>> pendingConcatCopies;

  for (auto i = 0; i < node->inputs().size(); ++i) {
    // The ordering input was placed above. Its non-ordering siblings still need
    // their producers placed if the previous-kernel stage did not already
    // materialize them (placeInput is a no-op for producers already placed).
    // Skipping them entirely -- as this loop used to -- orphans a producer
    // reachable only through this op, e.g. a standalone op feeding
    // repeat_interleave's data input: placeKernels never materializes it, yet
    // extractSubgraph still pulls it in as an interior node, so codegen aborts.
    if (meta && meta->inputFromPreviousKernel.has_value() &&
        i == meta->inputFromPreviousKernel.value()) {
      continue;
    }
    auto* inputValue = node->inputs()[i].value;
    auto* producer = inputValue->producer();
    // In multi-kernel mode, a fused producer whose output this op reads from
    // memory would be ordered by an intra-kernel opBarrier, which forces a
    // cooperative (whole-grid-resident) launch. End the producer's kernel
    // instead so its output is materialized by a prior stream-ordered launch;
    // the launch boundary is the barrier. cg and single-block keep the
    // in-kernel barrier on purpose.
    //
    // The decision is about the base of any view chain, since a view writes
    // nothing of its own: reading through a view of a value this kernel does
    // not write needs no ordering at all. The break itself still ends the
    // kernel at the direct producer -- for a view that pulls the base's writer
    // into the earlier kernel, which is what the launch boundary then orders.
    auto* dataValue = viewBase(inputValue);
    if (thisContext == Context::kFused && !isCgGrid_ && !isSingleBlock_ &&
        readsFusedElementwiseProducerFromMemory(
            i,
            node,
            meta,
            dataValue,
            dataValue ? dataValue->producer() : nullptr,
            types_,
            allStandalone_)) {
      breakProducerIntoOwnKernel(producer);
      continue;
    }
    auto inputKind = inputValue->type().kind();
    bool isScalarSize = isSizeArg(node->inputs()[i].name, meta) &&
        inputKind != nativert::Type::Kind::Tensor &&
        inputKind != nativert::Type::Kind::TensorList;
    if (thisContext == Context::kFused && producer &&
        producer->target() == "prim.ListPack") {
      const bool hostShapes = concatNeedsHostShapes(node, types_);
      const bool parallelFill = concatFillsInParallel(node, types_);
      const auto* dimAttr = node->tryGetAttribute("dim");
      const int64_t concatDim =
          dimAttr != nullptr && std::holds_alternative<int64_t>(dimAttr->value)
          ? std::get<int64_t>(dimAttr->value)
          : 0;
      const auto resultId = node->outputs()[0]->id();
      const auto resultDtype = resultId >= 0 &&
              static_cast<size_t>(resultId) < types_.types.size() &&
              types_.types[resultId]
          ? types_.types[resultId]->dtype()
          : c10::ScalarType::Float;
      // Every operand is placed first. Where the result can be laid out is the
      // latest point at which any operand's dimensions become known, and none
      // of those points exists until the launches that write them have landed
      // -- placeInput is what lands them. So the carve decision cannot be
      // taken in the same pass that places the operands.
      std::vector<ValueCP> operands;
      operands.reserve(producer->inputs().size());
      for (const auto& listArg : producer->inputs()) {
        auto* listInput = listArg.value;
        operands.push_back(listInput);
        if (hostShapes) {
          breakUnmeasurableProducers(listInput);
        }
        // Pushing the operand into a kernel of its own is what makes it
        // "already placed" when the carve is decided, and an already-placed
        // operand cannot be given a band -- it is copied instead. When its
        // producer could write the band itself, leave it here: placeInput then
        // fuses it into the concat's own kernel, reserveConcatOutput binds it
        // to its band, and the copy disappears.
        const bool inPlace =
            concatOperandFusesInPlace(listInput, node, concatDim, resultDtype);
        if (parallelFill && !inPlace) {
          breakConcatOperandIntoOwnKernel(listInput, node->outputs()[0]);
        }
        placeInput(listInput, isScalarSize);
      }
      // Decided whenever the allocation-group pass will look at this concat,
      // not only when the operands are filled in parallel: the group carves
      // from these verdicts either way, and the flag decides only whether a
      // copy op fills the bands the group does not carve or the concat's own
      // kernel does.
      if (concatAllocGroupEnabled()) {
        decideConcatCarve(node, operands, concatDim, resultDtype);
      }
      // An operand the concat allocation group will carve writes its band
      // itself; anything else needs a copy to move it there. One decision
      // answers both, so the group cannot later carve a band a copy is already
      // filling, nor leave a copy writing through a slot nothing bound.
      if (parallelFill) {
        for (size_t i = 0; i < operands.size(); ++i) {
          const auto occurrence = static_cast<int32_t>(i);
          if (!concatOperandNeedsCopyOp(node, occurrence)) {
            continue;
          }
          // The destination has to be minted now: the concat's own setOutputs
          // reads concatCopyDest_ when its code is generated, just below, to
          // know which operands its kernel must not move. Only the launch
          // waits.
          if (auto* destination =
                  makeConcatCopyDestination(node, occurrence, operands[i])) {
            pendingConcatCopies.emplace_back(operands[i], destination);
          }
        }
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
  // A concat whose operands fill it from ops of their own ends its kernel. The
  // operands write the result in the previous step, so a reader of the result
  // has to be in a later kernel; keeping it fused would instead need an
  // opBarrier on top of every operand's op, which the fusion path has no way to
  // express.
  if (concatFillsInParallel(node, types_)) {
    const auto concatLevel = pushdownFused(node);
    // Same step as the concat, not before it. A step sizes every launch, then
    // carves its groups -- binding each copy's band -- then fills the parameter
    // blocks, then launches; so a copy sharing the concat's step still finds
    // its band bound, while one placed earlier does not. The concat's own
    // kernel emits nothing for a copied operand, so the two never touch the
    // same bytes.
    for (const auto& [operand, destination] : pendingConcatCopies) {
      emitConcatOperandCopy(
          operand, destination, node->outputs()[0], concatLevel);
    }
    return Context::kFusedBreak;
  }
  if (meta->isKernelBreak(
          isSingleBlock_,
          isCgGrid_,
          WaveConfig::get().scanOutputReturnBarrier)) {
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
              standaloneToString(n));
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
              // None-valued attributes do not occupy a constant slot (see
              // listConstants / makeConstantIndices), so skip them when
              // computing the offset of 'attr' among the real constants.
              if (nativert::constantToIValue(origAttr.value).isNone()) {
                return;
              }
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

int32_t CompileCtx::pushdownFused(NodeCP node, int32_t minLevel) {
  auto sg = extractSubgraph(node, *inputs_, placed_);
  Launch launch;

  auto kernelOp = generateFused(sg);
  kernelOpStorage_.push_back(std::move(kernelOp));
  launch.op = kernelOpStorage_.back().get();
  launch.values.assign(
      launch.op->orderedInputs().begin(), launch.op->orderedInputs().end());
  launch.minLevel = minLevel;

  fillConstantIndices(sg, launch);
  const auto level = placeKernelLaunch(std::move(launch));
  placed_.insert(node);
  return level;
}

void CompileCtx::breakUnmeasurableProducers(ValueCP value) {
  auto* producer = value->producer();
  if (!producer || placed_.count(producer) ||
      (inputs_ && inputs_->count(producer))) {
    return;
  }
  // Post-order: the innermost one ends its kernel first, so the ops above it
  // read a host tensor that already carries the real extent and can stay fused
  // with the concat.
  for (const auto& input : producer->inputs()) {
    breakUnmeasurableProducers(input.value);
  }
  // Two ways an operand's extent can be out of reach when the concat lays its
  // result out: the device settles it, or only the producer's own reserve
  // knows it. Either way the concat cannot place a single operand until this
  // one is measurable, because the layout needs every extent at one point --
  // so the producer ends its kernel here and the value is materialized in an
  // earlier step, where it becomes an ordinary frame tensor the host can
  // measure and the concat can hand a view of its band.
  if (setsSizeOnDevice(producer) || sizeNeedsReserve(producer)) {
    breakProducerIntoOwnKernel(producer);
  }
}

bool CompileCtx::computedByThisKernel(ValueCP operand) const {
  auto* producer = operand->producer();
  return producer != nullptr && placed_.count(producer) == 0 &&
      (inputs_ == nullptr || inputs_->count(producer) == 0);
}

namespace {

// Follows a value back through prim.ListUnpack / prim.ListPack pairs to the
// value that was packed, which is the one a launch writes. The unpack does not
// copy -- it moves the packed element into its own frame slot -- so the two
// name one tensor, and every question about who fills its buffer has to be
// asked of the packed value. A concat operand reaching the list through more
// than one pack/unpack pair is why this loops rather than taking one hop.
ValueCP throughListPlumbing(ValueCP value) {
  for (int32_t guard = 0; guard < 8; ++guard) {
    auto* unpack = value->producer();
    if (unpack == nullptr || unpack->target() != "prim.ListUnpack" ||
        unpack->inputs().empty()) {
      return value;
    }
    const auto& outs = unpack->outputs();
    size_t index = outs.size();
    for (size_t i = 0; i < outs.size(); ++i) {
      if (outs[i] == value) {
        index = i;
        break;
      }
    }
    auto* pack = unpack->inputs()[0].value->producer();
    if (index == outs.size() || pack == nullptr ||
        pack->target() != "prim.ListPack" || index >= pack->inputs().size()) {
      return value;
    }
    value = pack->inputs()[index].value;
  }
  return value;
}

// The copy cause of a value already resolved through list plumbing.
//
// kNoMetadata is overruled when a launch writes the value. That verdict says
// only that the producing op is unregistered, which is what a chain of
// prim.ListUnpack looks like from here -- and it is exactly the question
// 'written' has already answered: a launch filling the buffer means there is a
// real kernel output to hand a band to, not a view and not a value the graph
// merely named. Every other cause still refuses, because each of those says
// something the write does not contradict.
ConcatCopyCause concatCopyCauseAfterPlumbing(
    ValueCP writer,
    const WaveGraph::SchedulePoint* written,
    int64_t dim,
    c10::ScalarType resultDtype,
    const ValueTypes& types) {
  const auto cause = concatOperandCopyCause(writer, dim, resultDtype, types);
  if (cause == ConcatCopyCause::kNoMetadata && written != nullptr) {
    return ConcatCopyCause::kNone;
  }
  return cause;
}

} // namespace

// True when 'operand' should stay for the concat's own kernel to compute rather
// than be pushed into a kernel of its own a step earlier.
//
// The pushdown exists so the host can measure the operand before the result is
// laid out. That is the right answer for an operand whose extent is only known
// once it has run; it is the wrong one for an ordinary size-preserving producer
// the concat is the only reader of, which can just as well write its band from
// inside the concat's kernel and save the copy entirely.
bool CompileCtx::concatOperandFusesInPlace(
    ValueCP operand,
    NodeCP concat,
    int64_t dim,
    c10::ScalarType resultDtype) const {
  if (!WaveConfig::get().concatOperandsInPlace) {
    return false;
  }
  auto* producer = operand->producer();
  // Nothing here to fuse: the pushdown would be a no-op and the operand is
  // copied either way.
  if (producer == nullptr || placed_.count(producer) != 0 ||
      (inputs_ != nullptr && inputs_->count(producer) != 0)) {
    return false;
  }
  // The band has to be writable by this producer at all -- a view, a promoted
  // dtype or a device-settled extent is refused here for the same reasons the
  // carve refuses it.
  if (concatOperandCopyCause(operand, dim, resultDtype, types_) !=
      ConcatCopyCause::kNone) {
    return false;
  }
  // The result is laid out before the kernel runs, so the host has to be able
  // to size this operand without it. An extent only its own reserve or the
  // device knows is exactly what the pushdown was for.
  //
  // This has to ask the question the ALLOCATION GROUP will ask, not a similar
  // one. The group refuses a whole concat when any operand answers
  // hasReserveShapeInChain (Cat.cpp), and refusing it after placement has
  // already minted copies for the other operands is a hard error, not a missed
  // optimization -- those copies would write bands nothing bound.
  // sizeNeedsReserve is NOT that question: it excludes an elementwise producer
  // and a multi-output one before it ever looks at the reserve, while the
  // group's walk scans every returnMeta of every node with no such exclusion.
  // An elementwise producer carrying a real reserve therefore passed here and
  // trapped there, which is what fusing in place by default first hit.
  //
  // Asking it of this producer alone is enough because of the sole-absorbed-
  // level rule below: every one of its inputs is already materialized, so once
  // the subgraph is extracted they are its inputs and the group's walk stops on
  // them immediately.
  auto reserveDefeatsGroup = [](NodeCP node) {
    const auto* meta = nodeMeta(node);
    if (meta == nullptr) {
      return false;
    }
    for (const auto& ret : meta->returnMeta) {
      if (ret.reserveShape != nullptr && ret.shapeFromInput < 0) {
        return true;
      }
    }
    return false;
  };
  if (setsSizeOnDevice(producer) || reserveDefeatsGroup(producer)) {
    return false;
  }
  // And the producer must be the ONLY thing this kernel absorbs for it: every
  // one of its inputs is already materialized. Walking further was tried and
  // is not sound here -- 'placed_' at placement time is not the boundary
  // ConcatInputInfo::hasReserveInChain uses once the subgraph is extracted, so
  // a chain that looked clear still handed the operand a reserve and cost the
  // whole concat its allocation group. One level, with everything behind it
  // already measured, is the case this is for anyway: an elementwise producer
  // reading values an earlier step wrote.
  for (const auto& input : producer->inputs()) {
    auto* inputProducer =
        input.value == nullptr ? nullptr : input.value->producer();
    if (inputProducer == nullptr || placed_.count(inputProducer) != 0 ||
        (inputs_ != nullptr && inputs_->count(inputProducer) != 0)) {
      continue;
    }
    return false;
  }
  // Sole consumer. A value read by anything else is a partitioner CSE border
  // and becomes a top-level output of its ProjectNode, so it is not this
  // concat's to reschedule. aten.sym_size.int reads the shape and no data, so
  // it does not make the operand shared.
  const auto mainIt = waveGraph_.idToValue().find(operand->id());
  auto* asRewritten =
      mainIt != waveGraph_.idToValue().end() ? mainIt->second : operand;
  for (auto* user : asRewritten->users()) {
    if (user == concat || user->target() == "prim.ListPack" ||
        user->target() == "torch.ops.aten.sym_size.int") {
      continue;
    }
    return false;
  }
  return true;
}

void CompileCtx::decideConcatCarve(
    NodeCP concat,
    const std::vector<ValueCP>& operands,
    int64_t dim,
    c10::ScalarType resultDtype) {
  auto* original = originalFromVariant(concat);
  if (original == nullptr) {
    original = concat;
  }

  // Where the result's layout can first be computed: the latest point at which
  // any operand's dimensions become known. An operand no launch produces -- a
  // graph input, an eager op's output -- carries its extent from the start and
  // puts no floor under it.
  // The earliest point the host can know a value's EXTENT, which is not the
  // point its buffer is filled. An ordinary kernel output is a shape function
  // of its inputs, so its extent is knowable as soon as theirs are -- measuring
  // it at the write is what pinned a wide concat's layout to the last launch of
  // its node, and refused a band to every operand written in an earlier one.
  //
  // The conservative case is recognised from the recording rather than from
  // metadata: recordSchedulePoints gives a standalone's output, and one the
  // device sizes, a realized point one step AFTER its write, precisely because
  // nothing on the host knows the extent until it has run. That is the
  // merge_and_dedup case, and it must keep its late point -- a cat fed by it
  // genuinely cannot be laid out until it returns.
  // 'known' false means the recursion could not answer and the caller keeps the
  // operand's own conservative point. That is NOT the same as a null point,
  // which means the extent puts no floor under the layout at all -- a value the
  // graph handed us carries its extent from the start. Conflating the two is
  // wrong in a way a bad layout does not reveal: schedule points are filled AS
  // placement progresses, so an ancestor not yet placed has no record and would
  // read as "known from the start", moving the layout too early and giving a
  // different answer depending on when this runs.
  // An op that declares its whole output size as a literal -- view/reshape with
  // no -1, zeros/ones/full with an explicit size -- knows its extent at compile
  // time, whatever its input does. Following its input would inherit a floor
  // that the shape does not actually depend on.
  auto constantShaped = [](NodeCP producer) {
    const auto* attr =
        const_cast<nativert::Node*>(producer)->tryGetAttribute("size");
    const auto* dims =
        attr ? std::get_if<std::vector<int64_t>>(&attr->value) : nullptr;
    if (dims == nullptr || dims->empty()) {
      return false;
    }
    return std::all_of(
        dims->begin(), dims->end(), [](int64_t d) { return d >= 0; });
  };
  struct ShapePoint {
    bool known{false};
    std::optional<WaveGraph::SchedulePoint> point;
    // Which branch settled it, for the diagnostic below.
    const char* why{"recursed to inputs"};
  };
  std::unordered_map<nativert::ValueId, ShapePoint> shapeMemo;
  std::function<ShapePoint(ValueCP, int32_t)> shapeRealized =
      [&](ValueCP value, int32_t depth) -> ShapePoint {
    if (value == nullptr) {
      return {false, std::nullopt};
    }
    const auto valueId = value->id();
    if (auto it = shapeMemo.find(valueId); it != shapeMemo.end()) {
      return it->second;
    }
    const auto* realized = realizedPoint(valueId);
    auto* producer = value->producer();
    const ShapePoint recorded{
        true,
        realized != nullptr ? std::optional<WaveGraph::SchedulePoint>{*realized}
                            : std::nullopt};
    // A value no node produces carries its extent from the start. Reported as
    // the graph's first point rather than as "no floor": by the time a concat
    // is placed the whole grid is laid out, so every operand's extent IS known
    // somewhere definite, and an absent answer only hides which. It also gives
    // the layout a point it can actually use -- the old null fell through to
    // the concat's own launch, the LATEST legal point, which is the opposite of
    // what "known from the start" means.
    if (producer == nullptr) {
      shapeMemo[valueId] =
          ShapePoint{true, WaveGraph::SchedulePoint{0, 0}, "known from start"};
      return shapeMemo[valueId];
    }
    // Constant output shape: no floor, and no reason to look at the input.
    if (constantShaped(producer)) {
      shapeMemo[valueId] =
          ShapePoint{true, WaveGraph::SchedulePoint{0, 0}, "constant size"};
      return shapeMemo[valueId];
    }
    // Produced but never recorded. By the time a concat is placed the whole
    // grid is laid out, so this is not a value waiting to be placed -- it is
    // one that is never a launch output: an intermediate fused into a kernel's
    // expression, which allocates nothing and so puts no floor of its own. Its
    // extent is known when its inputs' are, so walk through it rather than
    // giving up, which would inherit the CONSUMER's point instead.
    const bool fusedIntermediate = realized == nullptr;
    // Guard the recursion before it can cycle, and memoize so a wide operand
    // list does not walk one shared ancestor once per operand.
    if (depth >= 32) {
      auto tooDeep = recorded;
      tooDeep.why = "recursion depth";
      shapeMemo[valueId] = tooDeep;
      return tooDeep;
    }
    const auto* written = writtenPoint(valueId);
    // Realized later than written: the recording already says the host cannot
    // size this until it has run. Believe it. That is merge_and_dedup, and any
    // other standalone.
    // A reserve function is NOT a reason to stop. It is the host-side mechanism
    // that computes the extent from the inputs, so it can run as soon as they
    // are known -- which for a chain gather is once the flip or masked-select
    // HEAD it reads has produced its offsets, not once the gather itself has
    // run. Stopping here charged the gather's own step to every cat that reads
    // it. What genuinely has to wait is an extent the DEVICE settles, and a
    // standalone, both of which show up as realized after written.
    if (!fusedIntermediate &&
        (written == nullptr || !(*realized == *written) ||
         setsSizeOnDevice(producer))) {
      auto stop = recorded;
      stop.why = written == nullptr ? "never written"
          : !(*realized == *written)
          ? "realized after written (standalone or device-sized)"
          : "sets size on device";
      shapeMemo[valueId] = stop;
      return stop;
    }
    std::optional<WaveGraph::SchedulePoint> latest;
    for (const auto& input : producer->inputs()) {
      const auto point = shapeRealized(input.value, depth + 1);
      if (!point.known) {
        auto blocked = recorded;
        blocked.why = "an input is genuinely unanswerable";
        shapeMemo[valueId] = blocked;
        return blocked;
      }
      if (point.point.has_value() &&
          (!latest.has_value() || *latest < *point.point)) {
        latest = point.point;
      }
    }
    // Answered, and nothing under it imposes a floor: the extent is settled
    // from the start. Reported as the first point for the same reason as a
    // producer-less value -- an absent answer is not one.
    shapeMemo[valueId] = ShapePoint{
        true,
        latest.has_value() ? latest : WaveGraph::SchedulePoint{0, 0},
        latest.has_value() ? "recursed to inputs" : "no input imposes a floor"};
    return shapeMemo[valueId];
  };

  // Two points per operand, deliberately kept apart.
  //
  // The SHAPE point is when the host can know the extent; the DATA point is
  // when a launch fills the buffer. The layout only needs the first, but
  // CARVING needs the second, because the allocation-group collector is built
  // per step (CompiledOp.cpp, AllocGroupCollector(stepGroups)) and can only
  // intercept a member whose launch is sized in the step the group belongs to.
  //
  // So the layout still runs off the data points, which is what keeps a group
  // where its members are sized. The shape points are recorded beside them and
  // are what a collector able to span steps would use instead: they are
  // strictly earlier, and the gap between the two is the carving this cannot
  // reach yet.
  std::optional<WaveGraph::SchedulePoint> shapeAt;
  ValueCP shapeSetter = nullptr;
  const char* shapeWhy = "";
  std::optional<WaveGraph::SchedulePoint> layoutAt;
  ValueCP layoutSetter = nullptr;
  std::unordered_map<nativert::ValueId, std::optional<WaveGraph::SchedulePoint>>
      operandShapePoint;
  for (auto* operand : operands) {
    const auto shape = shapeRealized(operand, 0);
    const auto* dataRealized = realizedPoint(operand->id());
    const auto dataPoint = dataRealized != nullptr
        ? std::optional<WaveGraph::SchedulePoint>{*dataRealized}
        : std::nullopt;
    const auto shapePoint = shape.known ? shape.point : dataPoint;
    operandShapePoint[operand->id()] = shapePoint;
    if (shapePoint.has_value() &&
        (!shapeAt.has_value() || *shapeAt < *shapePoint)) {
      shapeAt = shapePoint;
      shapeSetter = operand;
      shapeWhy = shape.why;
    }
    if (dataPoint.has_value() &&
        (!layoutAt.has_value() || *layoutAt < *dataPoint)) {
      layoutAt = dataPoint;
      layoutSetter = operand;
    }
  }
  concatShapePoint_[original] = shapeAt;
  const auto dataAtForDiag = layoutAt;
  // Lay the result out where its SHAPE is knowable, not where the last
  // operand's data lands. Operands written after that point find their band
  // already there and fill it; the allocation-group pass gives each its own
  // group at the step its launch is sized, so moving this earlier no longer
  // costs the members written later.
  // NOT switched to shapeAt yet, though it is strictly earlier -- see
  // concatShapePoint. Two things still stand in the way, both found by trying
  // it: an operand whose extent has no floor at all would want the layout at
  // the earliest WRITE, and writtenPoint disagrees with the allocation group's
  // 'produced' map for a value an eager op writes (%2842 on the ROO graph),
  // which trips the carved-but-nothing-writes-it check. The per-step group
  // split below is in place and waiting for it.
  const WaveGraph::SchedulePoint* layoutPoint =
      layoutAt.has_value() ? &*layoutAt : nullptr;
  // DIAGNOSTIC: the one operand whose realization the whole layout waits on,
  // and why it is late. Every other operand written before this point is
  // refused a band only because of this one.
  if (layoutSetter != nullptr && operands.size() > 2 &&
      getenv("TW_CARVE_DIAG") != nullptr) {
    auto* setterProducer = layoutSetter->producer();
    const auto* meta = setterProducer == nullptr
        ? nullptr
        : Registry::metadata(setterProducer->target());
    bool sod = false;
    if (meta != nullptr) {
      for (const auto& returnMeta : meta->returnMeta) {
        sod = sod || returnMeta.shapeSetOnDevice;
      }
    }
    const auto* setterWritten = writtenPoint(layoutSetter->id());
    fprintf(
        stderr,
        "TW_CARVE_DIAG concat %%%d of %d: SHAPE=%s(%%%d %s / %s) DATA=%s | layout at node %d step %d set by"
        " %%%d producer=%s shapeOnDevice=%d registered=%d writtenAt=%s\n",
        static_cast<int>(concat->outputs()[0]->id()),
        static_cast<int>(operands.size()),
        shapeAt.has_value()
            ? fmt::format("{}/{}", shapeAt->node, shapeAt->step).c_str()
            : "none",
        shapeSetter != nullptr ? static_cast<int>(shapeSetter->id()) : -1,
        (shapeSetter != nullptr && shapeSetter->producer() != nullptr)
            ? std::string(shapeSetter->producer()->target()).c_str()
            : "<none>",
        shapeWhy,
        dataAtForDiag.has_value()
            ? fmt::format("{}/{}", dataAtForDiag->node, dataAtForDiag->step)
                  .c_str()
            : "none",
        layoutPoint->node,
        layoutPoint->step,
        static_cast<int>(layoutSetter->id()),
        setterProducer == nullptr
            ? "<none>"
            : std::string(setterProducer->target()).c_str(),
        sod ? 1 : 0,
        meta != nullptr ? 1 : 0,
        setterWritten == nullptr
            ? "<none>"
            : fmt::format("{}/{}", setterWritten->node, setterWritten->step)
                  .c_str());
  }

  // A concat may join one value at more than one position -- cat([x, y, x]) --
  // and one buffer cannot be two bands of the result.
  std::unordered_map<nativert::ValueId, int32_t> occurrences;
  for (auto* operand : operands) {
    ++occurrences[operand->id()];
  }

  // Whatever it decides, this op is now tied to the schedule around it and
  // must not be deduplicated onto an isomorphic concat elsewhere.
  opCarvesAConcat_ = true;

  // Where the group will be created. With no operand fixing an earlier point
  // the layout happens at the concat's own launch, which is the next step this
  // grid will place -- every operand of it is written there or in place.
  concatLayoutPoint_[original] = layoutPoint != nullptr
      ? *layoutPoint
      : WaveGraph::SchedulePoint{
            compileNodeIndex_, static_cast<int32_t>(grid_.size())};

  int32_t occurrence = -1;
  for (auto* operand : operands) {
    ++occurrence;
    const auto operandId = operand->id();
    // Asked of the value in the graph the rewrites left, not of this one: a
    // variant graph gives every boundary value a stand-in with no producer, so
    // asking here would answer "needs a copy" for every operand that merely
    // crosses into this kernel.
    const auto mainIt = waveGraph_.idToValue().find(operandId);
    auto* asRewritten =
        mainIt != waveGraph_.idToValue().end() ? mainIt->second : operand;
    // A prim.ListUnpack names an element of somebody else's list without
    // copying it, so an operand reached that way has no producer of its own to
    // ask and no launch writes its id. Every question below -- can the producer
    // write a band, does a launch write this, which slot does the group bind --
    // is about the value that was PACKED, so resolve to it once and ask there.
    auto* writer = throughListPlumbing(asRewritten);
    const auto writerId = writer->id();
    const bool throughPlumbing = writer != asRewritten;
    const auto* written = writtenPoint(throughPlumbing ? writerId : operandId);

    ConcatCarve decision;
    decision.needsCopy = true;
    if (throughPlumbing) {
      decision.writerId = writerId;
    }
    if (occurrences[operandId] != 1) {
      decision.reason = "joined at more than one position";
    } else if (
        carvedOperands_.count(throughPlumbing ? writerId : operandId) != 0) {
      decision.reason = "already carved for an earlier concat";
    } else if (const auto cause = concatCopyCauseAfterPlumbing(
                   writer, written, dim, resultDtype, types_);
               cause != ConcatCopyCause::kNone) {
      // Naming the producer's op, not just the cause: whether a copy is
      // avoidable depends on WHAT cannot write the band. An operand resolved
      // through plumbing blames the value that was packed, which is the one
      // that would have to write it.
      auto* blame = writer->producer();
      decision.reason = blame == nullptr
          ? std::string(concatCopyCauseText(cause))
          : fmt::format("{} ({})", concatCopyCauseText(cause), blame->target());
    } else if (computedByThisKernel(operand)) {
      // The concat's own kernel computes it, so reserveConcatOutput binds it
      // to its band and the expression writes the result in place. Neither a
      // group member -- the group allocates no buffer for it -- nor a copy.
      decision.needsCopy = false;
      decision.reason = "written in place by the concat's own kernel";
    } else if (written == nullptr) {
      // The seam this closes: in the main graph the value has a producer, so
      // the test above says its band is writable, but no wave kernel launch
      // writes it -- an eager op does. Nothing would fill the band.
      decision.reason = "no kernel launch writes it";
    } else if (layoutPoint == nullptr) {
      // Nothing measures earlier than the concat's own launch, so that is
      // where the result is laid out -- after this operand was written.
      decision.reason = "written before the concat's own launch lays it out";
    } else if (*written < *layoutPoint) {
      // DIAGNOSTIC: whether this operand could be MOVED past the point the
      // cat's dims become known. That is only safe if the cat is its sole
      // consumer -- a value with another user is a partitioner CSE border and
      // becomes a top-level output of its ProjectNode, so it cannot be
      // rescheduled into the concat's own kernel.
      if (getenv("TW_CARVE_DIAG") != nullptr) {
        int32_t users = 0;
        std::string others;
        for (auto* user : asRewritten->users()) {
          ++users;
          const bool isConcatSide = user == concat || user == original ||
              user->target() == "prim.ListPack";
          if (!isConcatSide) {
            others += " ";
            others += user->target();
          }
        }
        auto* blameOp = writer->producer();
        // Why it is not already in the concat's own kernel:
        // computedByThisKernel is false when the producer is in placed_
        // (something got there first) or in inputs_ (it crosses into this
        // kernel from outside).
        auto* prod = operand->producer();
        const char* placedWhy = prod == nullptr ? "no-producer"
            : placed_.count(prod) != 0          ? "already-placed"
            : (inputs_ != nullptr && inputs_->count(prod) != 0) ? "kernel-input"
                                                                : "would-fuse";
        fprintf(
            stderr,
            "TW_CARVE_DIAG movable concat=%%%d operand=%%%d by=%s users=%d"
            " placed=%s otherUsers=[%s]\n",
            static_cast<int>(concat->outputs()[0]->id()),
            static_cast<int>(operand->id()),
            blameOp == nullptr ? "<none>"
                               : std::string(blameOp->target()).c_str(),
            users,
            placedWhy,
            others.empty() ? "" : others.c_str() + 1);
      }
      // Strictly before, not merely different. The layout point now comes from
      // when extents are KNOWABLE, so it can precede the writes: an operand
      // written after the result is laid out finds its band already there and
      // fills it, which is the whole point. Only one written before the band
      // exists has to be copied.
      auto* blame = writer->producer();
      decision.reason = fmt::format(
          "written at node {} step {} by {}, before the result is laid out at "
          "node {} step {}",
          written->node,
          written->step,
          blame == nullptr ? "<none>" : std::string(blame->target()),
          layoutPoint->node,
          layoutPoint->step);
    } else {
      decision.groupCarves = true;
      decision.needsCopy = false;
      decision.reason = "written where the result is laid out";
      // Keyed on what the group binds: two concats reaching one packed value
      // through different unpacks would otherwise each carve it a band.
      carvedOperands_.insert(throughPlumbing ? writerId : operandId);
    }
    concatCarve_[{original, occurrence}] = std::move(decision);
  }

  // Recorded, not printed. This runs while the graph compiles, and tracing is
  // normally turned on around a later run -- printing here would reach a clear
  // trace bit and the reasons would be lost, which is what made a concat that
  // carves nothing impossible to account for from outside.
  std::map<std::string, std::vector<nativert::ValueId>> byReason;
  for (size_t i = 0; i < operands.size(); ++i) {
    const auto& decision = concatCarve_[{original, static_cast<int32_t>(i)}];
    byReason[decision.reason].push_back(operands[i]->id());
  }
  auto line = fmt::format(
      "  concat carve %{} of {} operands:",
      concat->outputs()[0]->id(),
      operands.size());
  for (const auto& [reason, ids] : byReason) {
    line += fmt::format(" [{} x {}:", ids.size(), reason);
    for (size_t i = 0; i < ids.size() && i < 6; ++i) {
      line += fmt::format(" %{}", ids[i]);
    }
    line += "]";
  }
  waveGraph_.addConcatCarveReport(line + "\n");
}

bool CompileCtx::breakConcatOperandIntoOwnKernel(
    ValueCP operand,
    ValueCP concatOutput) {
  auto* producer = operand->producer();
  const bool trace = (WaveConfig::get().trace & WaveConfig::kTiming) != 0;
  auto& skips = concatPushdownSkips_[concatOutput->id()];
  if (!producer || placed_.count(producer) ||
      (inputs_ && inputs_->count(producer))) {
    // Nothing of this operand is computed here: it is a graph input or a value
    // an earlier step already materialized, so there is no producing expression
    // to push down and no write of its own to redirect. It is copied instead.
    if (trace) {
      ++skips
          [!producer ? std::string("copied: no producer")
               : placed_.count(producer)
               ? std::string("copied: already placed: ") +
                   std::string(producer->target())
               : std::string("copied: a subgraph input: ") +
                   std::string(producer->target())];
    }
    return false;
  }
  if (trace) {
    ++skips[std::string("pushed down: ") + std::string(producer->target())];
  }
  const size_t before = kernelOpStorage_.size();
  breakProducerIntoOwnKernel(producer);
  for (size_t i = before; i < kernelOpStorage_.size(); ++i) {
    kernelOpStorage_[i]->addOrderingOutput(concatOutput->id());
  }
  return true;
}

nativert::Value* CompileCtx::makeConcatCopyDestination(
    NodeCP concat,
    int32_t occurrence,
    ValueCP operand) {
  const auto resultId = concat->outputs()[0]->id();
  if (resultId < 0 || static_cast<size_t>(resultId) >= types_.types.size() ||
      !types_.types[resultId]) {
    return nullptr;
  }
  const auto dtype = types_.types[resultId]->dtype();

  auto* graph = const_cast<nativert::Node*>(concat)->owningGraph();
  const auto known = concatCopyDest(originalFromVariant(concat), occurrence);

  // Normally each grid variant is built into a graph of its own, so the
  // destination is minted into a graph that has never seen it. A concat outside
  // every variant chain is the exception: it stays on the main graph, so all
  // three variants ask for the destination on the SAME graph. Naming it apart
  // from the minted value is what lets both cases use one key -- the minted
  // value is "cat_copy_<id>" on the main graph, and colliding with it is how
  // this crashed once the multi-kernel grid started carving.
  const auto destName =
      known >= 0 ? fmt::format("cat_copy_out_{}", known) : std::string{};
  if (!destName.empty()) {
    if (auto* existing = graph->tryGetValue(destName)) {
      // One copy node per occurrence per graph is exactly what is wanted: it
      // reads the same source and writes the same band either way.
      return existing;
    }
  }

  // A node of its own so the copy has an expression to generate from, holding
  // the source it reads. Its output is not consumed by anything: the concat's
  // operand list still names the original operand, which is what the layout
  // measures, and the copy only fills the band that operand occupies.
  auto* copyNode = graph->createNode(
      "torch.ops.aten.clone.default",
      {{"self", const_cast<nativert::Value*>(operand)}});
  graph->insertBefore(copyNode, const_cast<nativert::Node*>(concat));

  if (known >= 0) {
    auto* copied = copyNode->addOutput(
        destName, nativert::Type(nativert::Type::Kind::Tensor));
    copied->setId(known);
    return copied;
  }
  // The id is minted on the main graph and mirrored here, never handed out by
  // this graph: a variant is built by giving fresh values the ids of the ones
  // they stand for, so its own id counter is still near zero and an id it hands
  // out lands on a frame slot some live value already holds. The band view then
  // goes over that value, and its reader sees the wrong tensor.
  auto* minted = waveGraph_.newTensorValue(
      waveGraph_.placeholderNode(), "cat_copy", dtype);
  auto* copied = copyNode->addOutput(
      fmt::format("cat_copy_out_{}", minted->id()), minted->type());
  copied->setId(minted->id());
  setConcatCopyDest(originalFromVariant(concat), occurrence, copied->id());
  // The destination is a band of the result, which off the outermost axis is
  // pitched. Saying contiguous there would let a reader take a fast path the
  // layout does not support; __copyTensor goes through the strides either way.
  auto& constraint = waveGraph_.types().constraints.at(copied->id());
  constraint.rank = types_.rank(operand);
  const auto* dimAttr = concat->tryGetAttribute("dim");
  const bool joinsOutermost = dimAttr == nullptr ||
      !std::holds_alternative<int64_t>(dimAttr->value) ||
      std::get<int64_t>(dimAttr->value) == 0;
  constraint.contiguity =
      joinsOutermost ? Contiguity::kContiguous : Contiguity::kUnknown;
  return copied;
}

bool CompileCtx::emitConcatOperandCopy(
    ValueCP operand,
    ValueCP destination,
    ValueCP concatOutput,
    int32_t minLevel) {
  if (destination == nullptr) {
    return false;
  }
  const auto destId = destination->id();
  if (destId < 0 || static_cast<size_t>(destId) >= types_.types.size() ||
      !types_.types[destId]) {
    return false;
  }
  const auto dtype = types_.types[destId]->dtype();

  auto* copyNode = destination->producer();
  TORCH_CHECK(
      copyNode != nullptr,
      "Concat operand copy destination %",
      destId,
      " has no copy node");

  auto it = concatCopyOp_.find(dtype);
  const bool traceCopy = (WaveConfig::get().trace & WaveConfig::kTiming) != 0;
  auto traceLevel = [&](int32_t level, const char* how) {
    if (traceCopy) {
      std::cout << "  concat copy %" << operand->id() << " -> %" << destId
                << " dtype=" << c10::toString(dtype) << " minLevel=" << minLevel
                << " landed at " << level << " " << how << std::endl;
    }
  };
  if (it == concatCopyOp_.end()) {
    // The first of its type carries the code. pushdownFused builds the op from
    // the copy node's own subgraph, whose only input is the source, so the
    // parameter block is the two descriptors this copy needs rather than every
    // operand of the concat.
    const size_t before = kernelOpStorage_.size();
    traceLevel(pushdownFused(copyNode, minLevel), "(new op)");
    TORCH_CHECK(
        kernelOpStorage_.size() > before,
        "pushdownFused produced no kernel op for a concat operand copy");
    for (size_t i = before; i < kernelOpStorage_.size(); ++i) {
      kernelOpStorage_[i]->addOrderingOutput(concatOutput->id());
    }
    // The band is the concat group's, put in the frame before this runs, so
    // the copy must not have a buffer reserved for it.
    kernelOpStorage_.back()->delegateOutputs();
    concatCopyOp_[dtype] = kernelOpStorage_.back().get();
    return true;
  }

  // Every later copy is the same op with different parameters. Launch::values
  // is positional against op->orderedInputs(), so the substitution is by
  // position: the op's own source and destination stand for this one's.
  auto* op = it->second;
  Launch launch;
  launch.op = op;
  launch.values.reserve(op->orderedInputs().size());
  auto* formalDest = op->expr() != nullptr && !op->expr()->outputs().empty()
      ? op->expr()->outputs()[0]
      : nullptr;
  auto* formalSource = op->expr() != nullptr && !op->expr()->inputs().empty()
      ? op->expr()->inputs()[0].value
      : nullptr;
  for (auto* value : op->orderedInputs()) {
    if (value == formalSource) {
      launch.values.push_back(operand);
    } else if (value == formalDest) {
      launch.values.push_back(destination);
    } else {
      launch.values.push_back(value);
    }
  }
  launch.minLevel = minLevel;
  traceLevel(placeKernelLaunch(std::move(launch)), "(reuses the op)");
  placed_.insert(copyNode);
  return true;
}

void CompileCtx::breakProducerIntoOwnKernel(NodeCP producer) {
  if (!producer || placed_.count(producer) ||
      (inputs_ && inputs_->count(producer))) {
    return;
  }
  placeKernels(producer, Context::kFused);
  if (!placed_.count(producer)) {
    pushdownFused(producer);
  }
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

int32_t CompileCtx::placeKernelLaunch(Launch launch) {
  int32_t latestLevel = -1;

  // Collect input value ids of this launch.
  std::unordered_set<nativert::ValueId> inputIds;
  if (launch.standalone) {
    for (const auto& input : launch.standalone->inputs()) {
      inputIds.insert(input.value->id());
    }
  } else if (launch.op) {
    inputIds = launch.op->orderingInputs();
    // A launch carrying its own parameters is ordered against those, not
    // against the op's. The copies that fill a wide concat's bands all run one
    // op, so taking the op's inputs would schedule every one of them against
    // the first copy's source: a copy whose own source is written later would
    // be placed in a step before the value it reads exists.
    if (launch.values.size() == launch.op->orderedInputs().size()) {
      const auto numInputs = launch.op->numInputs();
      for (int32_t i = 0; i < numInputs; ++i) {
        inputIds.insert(launch.values[i]->id());
      }
    }
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

  int32_t targetLevel = std::max(latestLevel + 1, launch.minLevel);
  while (targetLevel >= static_cast<int32_t>(grid_.size())) {
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
  recordSchedulePoints(launch, targetLevel, {}, /*intoGrid=*/true);
  grid_[targetLevel].push_back(std::move(launch));
  return targetLevel;
}

void CompileCtx::recordInvocationSchedulePoints(OpInvocation& op) {
  // The grid the allocation-group mode will run, which is the one its step
  // indices name.
  auto& grid = allocGroupGrid(op);
  for (size_t level = 0; level < grid.size(); ++level) {
    for (const auto& launch : grid[level]) {
      recordSchedulePoints(
          launch,
          static_cast<int32_t>(level),
          op.bindings(),
          /*intoGrid=*/false);
    }
  }
}

const WaveGraph::SchedulePoint* CompileCtx::writtenPoint(
    nativert::ValueId id) const {
  const auto it = gridWrittenAt_.find(id);
  return it != gridWrittenAt_.end() ? &it->second : waveGraph_.writtenAt(id);
}

const WaveGraph::SchedulePoint* CompileCtx::realizedPoint(
    nativert::ValueId id) const {
  const auto it = gridRealizedAt_.find(id);
  return it != gridRealizedAt_.end() ? &it->second : waveGraph_.realizedAt(id);
}

void CompileCtx::recordSchedulePoints(
    const Launch& launch,
    int32_t step,
    const FormalToActual& bindings,
    bool intoGrid) {
  // A step index only names one launch when the config fixes a single grid.
  // Otherwise the same op is placed once per variant and the first variant
  // built would decide every point, which is not the grid that runs.
  //
  // Which grid that is does not have to be the cooperative one: the
  // multi-kernel grid is settled by the same compilation. What matters is that
  // the choice is made, so the points name the launches that will actually run.
  // Left at auto, either could run and a point names nothing. The per-grid maps
  // are cleared by newGrid() for every variant, so recording while a variant
  // that will not run is being built only affects that variant's own placement
  // decisions; the graph-wide map is written from one grid, by
  // recordInvocationSchedulePoints.
  const auto& config = WaveConfig::get();
  if (!config.isCg.has_value()) {
    return;
  }
  const WaveGraph::SchedulePoint atStep{compileNodeIndex_, step};
  const WaveGraph::SchedulePoint afterStep{compileNodeIndex_, step + 1};
  // Translated to this invocation's own values. A grid is built once per
  // project op, over the formal subgraph, and every invocation of it binds its
  // own actuals -- which is what the frame, and the allocation group reading
  // it, are expressed in. Recording the formals would name values no execution
  // ever writes.
  auto record = [&](nativert::ValueId id,
                    const WaveGraph::SchedulePoint& realized) {
    if (id < 0) {
      return;
    }
    if (intoGrid) {
      gridWrittenAt_.try_emplace(id, atStep);
      gridRealizedAt_.try_emplace(id, realized);
      return;
    }
    const auto it = bindings.find(id);
    waveGraph_.addSchedulePoint(
        it != bindings.end() ? it->second : id, atStep, realized);
  };

  if (launch.standalone != nullptr) {
    // A standalone sizes its own output, so nothing on the host knows the
    // extent until it has run.
    for (auto* output : launch.standalone->outputs()) {
      record(output->id(), afterStep);
    }
    return;
  }
  if (launch.op == nullptr) {
    return;
  }
  // A launch carrying its own parameters -- one of the copies filling a wide
  // concat's bands -- names values the shared op does not, so its outputs are
  // its own rather than the op's formals.
  const auto& formals = launch.op->orderedInputs();
  const auto& values =
      launch.values.size() == formals.size() ? launch.values : formals;
  const auto numInputs = static_cast<size_t>(launch.op->numInputs());
  const auto& descs = launch.op->outputDescs();
  for (size_t i = 0; i < descs.size(); ++i) {
    const size_t slot = numInputs + i;
    if (slot >= values.size() || values[slot] == nullptr) {
      break;
    }
    const auto& realized = descs[i].shapeSetOnDevice ? afterStep : atStep;
    // A tensor list output is a header, not a buffer: each element is reserved
    // and written on its own, and it is the elements a concat joins. Recording
    // only the header would leave every element looking like a value no launch
    // writes.
    if (values[slot]->type().kind() == nativert::Type::Kind::TensorList) {
      for (const auto* element : values[slot]->getListElements()) {
        if (element != nullptr) {
          record(element->id(), realized);
        }
      }
      continue;
    }
    record(values[slot]->id(), realized);
  }
}

void CompileCtx::generateElementwiseBorderImpl(
    NodeCP node,
    const std::unordered_set<ValueCP>& opInputs,
    NodeSet& visited) {
  auto* consumerMeta = nodeMeta(node);

  auto getArgMeta = [&](size_t idx) -> const ArgumentMeta* {
    return argMetaForInput(consumerMeta, node, idx);
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
    return meta && meta->elementwise && !meta->isElementwiseBorder();
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
    if (!value) {
      continue;
    }
    auto* producer = value->producer();
    // A tensor list read by this (non-elementwise) op is consumed from memory,
    // so every element must be materialized to its own Value buffer -- never
    // left in a register -- in all modes. Do this whether or not the list is a
    // boundary memOutput: an internal ListPack fed by fused elementwise
    // producers (e.g. values built by an add) is not a memOutput and would
    // otherwise never be written. Element producers already placed (a boundary
    // list from a prior op) are skipped by the placed_ guard.
    if (producer && producer->target() == "prim.ListPack" &&
        !placed_.count(producer)) {
      placed_.insert(producer);
      for (const auto& lpInput : producer->inputs()) {
        auto* lpValue = lpInput.value;
        auto* lpProducer = lpValue ? lpValue->producer() : nullptr;
        if (lpProducer && !placed_.count(lpProducer)) {
          std::vector<ResultSpec> lpSpecs;
          ResultSpec rs;
          rs.value = lpValue;
          lpSpecs.push_back(rs);
          fusedCode(lpProducer, lpSpecs);
        }
      }
      continue;
    }
    if (memOutputs.count(value)) {
      if (producer && !placed_.count(producer)) {
        auto prodSpecs = outputSpecs(producer);
        fusedCode(producer, prodSpecs);
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
    const auto* am = argMetaForInput(meta, node, i);
    if (am && am->isRegister) {
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
    const auto* am = argMetaForInput(meta, node, i);
    bool isReg = am && am->isRegister;
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
    // For a data-dependent scan op (masked_select / nonzero) whose size is set
    // on device, pass its output so generateElementwise can zero the length for
    // an empty input before a fused consumer reads it.
    ValueCP shapeSetOnDeviceResult = nullptr;
    if (meta && !meta->returnMeta.empty() &&
        meta->returnMeta[0].shapeSetOnDevice && !node->outputs().empty()) {
      shapeSetOnDeviceResult = node->outputs()[0];
    }
    generateElementwise(
        subgraphs,
        ewResultSpecs,
        callStmt,
        meta->hasBarrier,
        shapeSetOnDeviceResult);
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

// A C++ type spelling reduced to something usable inside an identifier, so a
// shared declaration's variable can be made unique per type. Anything that is
// not alphanumeric becomes an underscore, which keeps distinct types distinct
// without needing to understand the spelling.
std::string identifierSuffix(std::string_view type) {
  std::string out;
  out.reserve(type.size());
  for (char c : type) {
    out += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
  }
  return out;
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
  // Schema-less scalar ops (isScalarElementwise) have no presence template
  // params.
  if (!meta.functionSchema) {
    return result;
  }
  const auto& schemaArgs = meta.functionSchema->arguments();
  for (size_t i = 0; i < schemaArgs.size(); ++i) {
    if (i >= meta.argumentMeta.size() ||
        !meta.argumentMeta[i].hasPresentTemplateParam) {
      continue;
    }
    if (!result.empty()) {
      result += ", ";
    }
    // Use the same presence rule as dedup (isParamPresent), which treats a
    // None-typed input slot as absent.
    result += isParamPresent(node, schemaArgs[i].name()) ? "true" : "false";
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
  // Before the list is opened, because it may declare what it names at
  // translation-unit scope and so must run exactly once per call.
  auto generatedParam = meta->generateTemplateArg
      ? meta->generateTemplateArg(node, this)
      : std::string();
  if (meta->hasBlockSizeTemplateParam || !meta->typeTemplateParams.empty() ||
      meta->hasDtypeTemplateParam || !meta->templateAttrs.empty() ||
      !generatedParam.empty() || !presenceParams.empty()) {
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
    if (!generatedParam.empty()) {
      if (!firstTp) {
        ss << ", ";
      }
      firstTp = false;
      ss << generatedParam;
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
    // A naked scalar that is this kernel's top-level output (e.g. sym_size /
    // numel returned to host or consumed by another kernel as a host param)
    // must be read back into the frame between launches. Mirrors the dynamic
    // neededOnHost decision in KernelOperation::setOutputs (node == expr_).
    // Fused interior uses emit to a register and do not reach this branch.
    bool nakedScalarOutput = outputs[i].value && op.expr() == node &&
        outputs[i].value->type().kind() != nativert::Type::Kind::Tensor &&
        outputs[i].value->type().kind() != nativert::Type::Kind::TensorList;
    if (outputs[i].value &&
        ((i < meta->returnMeta.size() &&
          (meta->returnMeta[i].shapeSetOnDevice ||
           meta->returnMeta[i].neededOnHost)) ||
         nakedScalarOutput)) {
      waveGraph_.addSyncableValueId(outputs[i].value->id());
    }
  }

  // Shared declarations: declare in the kernel and pass as arguments. The name
  // is suffixed by the type, as the dynamic form below already does, because
  // two ops fused into one kernel can ask for the same name at different
  // types: masked_select_jagged wants a uint32_t 'counter' and
  // group_length_guard_head an int64_t one. Emitting both unsuffixed put two
  // conflicting declarations of 'counter' in one scope. nvcc calls that a
  // redeclaration; NVRTC segfaults inside nvrtcCompileProgram, before the
  // program log is readable, so it surfaces as a broken promise for the
  // compiled module rather than as a compile error.
  //
  // Deduplication cannot fix this on its own: addSharedDeclaration matches on
  // the whole declaration string, and even matching on the name would be wrong,
  // since the two ops need storage of different types rather than one shared
  // variable.
  for (const auto& [type, name] : meta->sharedDecls) {
    auto varName = name + "_" + identifierSuffix(type);
    std::string decl = "  __shared__ ";
    decl += type;
    decl += " ";
    decl += varName;
    decl += ";\n";
    op.addSharedDeclaration(decl);
    comma();
    ss << varName;
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
  auto srcTypeName = cudaType(source);
  if (srcTypeName != cudaTypeName) {
    // Mixed-dtype cat element (e.g. an int64 cumsum slice copied into a float
    // cat output, where torch type-promotes the element): value-convert rather
    // than bit-copy, which would reinterpret the source bytes.
    code_ << "  __copyConvert<" << srcTypeName << ", " << cudaTypeName << ">("
          << param(source, op) << ", "
          << "storage<" << cudaTypeName << ">(" << param(dest, op) << ")";
  } else {
    code_ << "  __copy<" << cudaTypeName << ">(" << param(source, op) << ", "
          << "storage<" << cudaTypeName << ">(" << param(dest, op) << ")";
  }
  if (!destOffsetExpr.empty()) {
    code_ << " + " << destOffsetExpr;
  }
  code_ << ", blockInfo);\n";
}

void CompileCtx::emitCode(std::string_view text) {
  code_ << text;
}

void CompileCtx::emitHelperCode(std::string_view text) {
  outOfLineFunctions_ << text;
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

// Resolves the ordering question for one operand. An input read through a
// view is the base's storage: a view of something this kernel fills still
// needs the barrier, a view of anything else does not, and the view node
// itself never writes and so never justifies one. An in-place writer is not
// the base's producer either -- the storage was created elsewhere, often in
// an earlier kernel -- so the producer test alone cannot see it; ask the
// users too, since a writer running unsynchronized in this kernel is the same
// hazard as a producer running in it.
bool CompileCtx::valueNeedsBarrier(ValueCP operand, NodeCP consumer) {
  auto* value = viewBase(operand);
  if (!value) {
    return false;
  }
  auto* producer = value->producer();
  if (producer && generatingOp_->allNodes().count(producer) &&
      !preBarrierValues_.count(value)) {
    return true;
  }
  for (auto* user : value->users()) {
    if (user == consumer || !generatingOp_->allNodes().count(user)) {
      continue;
    }
    const auto* userMeta = nodeMeta(user);
    if (!userMeta || !userMeta->mutatesArg.has_value()) {
      continue;
    }
    auto ordinal = static_cast<size_t>(*userMeta->mutatesArg);
    const auto& userInputs = user->inputs();
    if (ordinal >= userInputs.size() || userInputs[ordinal].value != value) {
      continue;
    }
    bool synchronized = true;
    for (auto* output : user->outputs()) {
      if (!preBarrierValues_.count(output)) {
        synchronized = false;
        break;
      }
    }
    if (!synchronized) {
      return true;
    }
  }
  return false;
}

bool CompileCtx::callNeedsBarrier(NodeCP node) {
  // A call reads its inputs from memory, so if a producer ran earlier in this
  // same kernel with no barrier since, that producer's writes from other blocks
  // may not yet be visible. randomAccess is not consulted -- an aligned read is
  // still unsafe across blocks, and the preBarrierValues_ check below avoids
  // emitting a redundant barrier when one already separates them.
  auto* meta = nodeMeta(node);
  if (!meta) {
    return false;
  }
  auto needsBarrierFor = [&](ValueCP operand) {
    return valueNeedsBarrier(operand, node);
  };
  const auto& inputs = node->inputs();
  for (size_t i = 0; i < inputs.size() && i < meta->argumentMeta.size(); ++i) {
    // Register inputs flow inline as fused values, not through memory, so they
    // never need a barrier.
    if (meta->argumentMeta[i].isRegister) {
      continue;
    }
    auto* value = inputs[i].value;
    if (!value) {
      continue;
    }
    // For a tensor list the list node itself is metadata-only; the real
    // producers are the element nodes, so check each element.
    if (value->type().kind() == nativert::Type::Kind::TensorList) {
      for (auto* elem : value->getListElements()) {
        if (elem && needsBarrierFor(elem)) {
          return true;
        }
      }
    } else if (needsBarrierFor(value)) {
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
  // A TensorList's element type (all elements share a dtype) lets an op
  // template on a list input, e.g. group_length_guard_final templates on the
  // head_list (offset) and values element types.
  if (kind == nativert::Type::Kind::TensorList) {
    auto elements = value->getListElements();
    TORCH_CHECK(
        !elements.empty(), "Empty TensorList for cudaType: ", value->name());
    auto elemId = elements[0]->id();
    TORCH_CHECK(
        elemId < types_.types.size() && types_.types[elemId],
        "No TensorMeta for TensorList element of ",
        value->name());
    return cudaTypeString(types_.types[elemId]->dtype());
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

// Returns every actual frame value 'op' may read, over all of its grid
// variants. Used to decide at which step a last-use value can be released, so
// this has to err on the side of listing too much: a reader missed here is a
// buffer freed while a later step still reads it. orderingInputs is the right
// source for a kernel launch -- it is the set the scheduler orders the launch
// against, and so already includes the host-side view operands that are inputs
// of no fused node. The subgraph leaves cover an op whose grid is empty.
std::unordered_set<nativert::ValueId> opReadSet(const OpInvocation& op) {
  const auto& bindings = op.bindings();
  auto toActual = [&](nativert::ValueId formalId) {
    auto it = bindings.find(formalId);
    return it != bindings.end() ? it->second : formalId;
  };
  std::unordered_set<nativert::ValueId> ids;
  auto* projectOp = op.projectOp();
  for (const auto* grid :
       {&projectOp->grid(),
        &projectOp->singleBlockGrid(),
        &projectOp->cgGrid()}) {
    for (const auto& step : *grid) {
      for (const auto& launch : step) {
        if (launch.op != nullptr) {
          for (auto id : launch.op->orderingInputs()) {
            ids.insert(toActual(id));
          }
        } else if (launch.standalone != nullptr) {
          for (const auto& input : launch.standalone->inputs()) {
            if (input.value != nullptr) {
              ids.insert(toActual(input.value->id()));
            }
          }
        }
      }
    }
  }
  for (const auto* input : projectOp->subgraph().inputs) {
    ids.insert(toActual(input->id()));
  }
  return ids;
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
  // ProjectOperations are deduplicated only within a single compiled node.
  // opStorage_ (which owns the ProjectOperations) is moved into this node's
  // CompositeKernel at the end of this function, so any ProjectOperation*
  // recorded in projectOps_ from a previous node points into a kernel that no
  // longer belongs to the node being compiled. Reusing it here would launch an
  // opcode whose case is absent from this node's kernel. Clear the map so dedup
  // never crosses node boundaries.
  projectOps_.clear();
  // The KernelOperation a concat operand copy reuses is owned by
  // kernelOpStorage_, which is moved into this node's CompositeKernel the same
  // way, so it cannot cross a node boundary either. Kept until the second node
  // with a wide concat this was a launch of the previous node's code: the band
  // stayed at whatever the allocation held.
  concatCopyOp_.clear();
  placedBeforeNode_ = placed_;
  inputs_ = &project.inputs();
  currentNodeId_ = project.id();
  numDistinctOps_ = 0;
  auto& nodes = project.nodes();

  // Generates the kernel / standalone for a single node: extract its subgraph
  // and either reuse a duplicate ProjectOperation or build a new one.
  auto generateNode = [&](NodeCP genNode) {
    auto sg = extractSubgraph(genNode, project.inputs(), placed_);
    auto it = projectOps_.find(sg);
    if (it != projectOps_.end()) {
      ops_.emplace_back(it->second, sg, ivalueStorage_);
      addDuplicateExtraBindings(
          ops_.back(), it->second->extraValues(), waveGraph_);
      recordInvocationSchedulePoints(ops_.back());
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
        // Left out of the dedup map when it carves a concat. Carving is an
        // agreement between this occurrence and the schedule around it: which
        // operands the kernel writes in place and which a copy fills depends on
        // the step each operand's producer landed on, and the generated code
        // bakes that in -- concatSpecialForm emits a move for exactly the
        // operands placement said to copy. A second, isomorphic concat later in
        // the graph joins different values at different steps, so one kernel
        // cannot serve both. Registering it would also make the decision's
        // value ids formal rather than actual, which is what the allocation
        // group reads them as.
        if (!opCarvesAConcat_) {
          projectOps_[sg] = projectOp;
        }
        ops_.emplace_back(projectOp, sg, ivalueStorage_);
        addSelfExtraBindings(ops_.back(), projectOp->extraValues());
        recordInvocationSchedulePoints(ops_.back());
      }
    }
  };

  // Fused ops whose only consumer is a no-op prim.ListUnpack (the fused code
  // assigns the list tensors directly). They are skipped during the main pass;
  // any that a downstream consumer pulls into its own kernel become placed, and
  // the genuinely orphaned ones (e.g. a single-use list result feeding graph
  // outputs, like group_length_guard_final) are generated in the cleanup pass.
  std::vector<NodeCP> deferredFusedProducers;

  for (size_t i = 0; i < nodes.size(); ++i) {
    currentExprOrdinal_ = static_cast<int32_t>(i);
    auto* node = nodes[i];
    if (node->target() == "prim.Input" || placed_.count(node)) {
      continue;
    }
    if (node->target() == "prim.ListUnpack") {
      auto* listValue = node->inputs()[0].value;
      auto* producer = listValue->producer();
      // A prim.listunpack over a fused op is a no-op. Mark the unpack placed
      // and defer its producer: it is generated on its own only if nothing else
      // pulls it into a kernel (see the cleanup pass below).
      if (producer) {
        auto* producerMeta = Registry::metadata(producer->target());
        if (producerMeta && !producerMeta->isStandalone(producer, types_)) {
          placed_.insert(node);
          if (!placed_.count(producer)) {
            deferredFusedProducers.push_back(producer);
          }
          continue;
        }
      }
    }
    generateNode(node);
  }

  // Cleanup pass: generate any deferred fused producer that no consumer pulled
  // into a kernel, rooted at the producer rather than the (no-op) unpack.
  for (size_t k = 0; k < deferredFusedProducers.size(); ++k) {
    auto* producer = deferredFusedProducers[k];
    if (placed_.count(producer)) {
      continue;
    }
    currentExprOrdinal_ = static_cast<int32_t>(nodes.size() + k);
    generateNode(producer);
  }
  if (ops_.empty()) {
    return nullptr;
  }
  auto compositeKernel = std::make_unique<CompositeKernel>(
      std::move(opStorage_),
      std::move(kernelOpStorage_),
      includes_,
      nextKernelId());
  // Values whose last use is in this node (graph outputs already excluded);
  // WaveConfig::freeIntermediates releases their frame tensors after execute().
  std::vector<nativert::ValueId> lastUseIds;
  lastUseIds.reserve(project.lastUse.size());
  // For each of them, the ops_ indices that read it, which is what lets the
  // executor release the value after the last step those ops occupy instead of
  // after the node's last step. Left empty -- meaning "release at the node's
  // last step" -- for a value produced inside this node: its producer writes
  // the frame slot at a step this reader-based bound knows nothing about.
  std::vector<std::vector<int32_t>> lastUseReaderOps;
  lastUseReaderOps.reserve(project.lastUse.size());
  std::vector<std::unordered_set<nativert::ValueId>> readSets;
  readSets.reserve(ops_.size());
  for (const auto& op : ops_) {
    readSets.push_back(opReadSet(op));
  }
  const std::unordered_set<NodeCP> layerNodes(nodes.begin(), nodes.end());
  for (auto* value : project.lastUse) {
    lastUseIds.push_back(value->id());
    std::vector<int32_t> readers;
    const auto* producer = value->producer();
    if (producer == nullptr || layerNodes.count(producer) == 0) {
      for (size_t i = 0; i < readSets.size(); ++i) {
        if (readSets[i].count(value->id()) != 0) {
          readers.push_back(static_cast<int32_t>(i));
        }
      }
    }
    lastUseReaderOps.push_back(std::move(readers));
  }
  // Values whose buffer an elementwise op may reuse in place for its output:
  // reusable last-use boundary inputs and expr-local overwritable temps. Only
  // eligible when EVERY consumer is a pointwise op
  // (Metadata::inPlaceIfLastUse): such ops read the operand at the identity
  // index, so writing the output into its buffer is a per-element
  // read-before-write and cannot clobber a value the kernel still needs. A
  // non-pointwise consumer (broadcast/gather/view/scatter, which carry
  // inPlaceIfLastUse=false) reads at other indices, where in-place reuse would
  // corrupt results.
  auto reuseSafe = [](ValueCP value) {
    // Exactly one consumer: a forked value (>1 user) feeds several ops, each of
    // which could reuse its buffer for its own output and clobber the others'
    // reads. Require a single consumer so at most one output reuses the buffer.
    if (value->users().size() != 1) {
      return false;
    }
    for (auto* user : value->users()) {
      const Metadata* m = Registry::metadata(user->target());
      if (m == nullptr || !m->inPlaceIfLastUse) {
        return false;
      }
    }
    return true;
  };
  std::vector<nativert::ValueId> reusableIds;
  for (const auto& perExpr : project.reusableValues_) {
    for (auto* value : perExpr) {
      if (reuseSafe(value)) {
        reusableIds.push_back(value->id());
      }
    }
  }
  for (const auto& perExpr : project.overwritableTemps_) {
    for (auto* value : perExpr) {
      if (reuseSafe(value)) {
        reusableIds.push_back(value->id());
      }
    }
  }
  // Inputs of the clones elided in this node, with the number of copies of
  // each saved. Reported at runtime as the bytes not copied; also registered
  // graph-wide so the reference-frame checks skip these deliberately
  // overwritten buffers.
  std::vector<std::pair<nativert::ValueId, int32_t>> elidedCloneInputs(
      project.elidedCloneCounts.begin(), project.elidedCloneCounts.end());
  for (const auto& [valueId, count] : elidedCloneInputs) {
    waveGraph_.addElidedCloneInput(valueId);
  }
  auto invocation = std::make_unique<CompositeInvocation>(
      std::move(compositeKernel),
      std::move(ops_),
      std::move(ivalueStorage_),
      waveGraph_.nextCompositeInvocationId(),
      std::move(lastUseIds),
      std::move(lastUseReaderOps),
      std::move(reusableIds),
      std::vector<Launch>{},
      std::move(elidedCloneInputs));
  placed_.insert(nodes.begin(), nodes.end());
  // Only a node that produces one takes an index, so the count stays in step
  // with the position this node will have among the graph's CompiledNodes --
  // which is what the schedule points above are expressed against.
  ++compileNodeIndex_;
  return std::make_unique<CompiledNode>(std::move(invocation));
}

} // namespace torch::wave
