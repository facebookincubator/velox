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

#include "velox/experimental/torchwave/WaveGraph.h"
#include "velox/experimental/torchwave/Compile.h"
#include "velox/experimental/torchwave/CompiledOp.h"
#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/GraphOptimizer.h"
#include "velox/experimental/torchwave/ParallelExpr.h"
#include "velox/experimental/torchwave/Utils.h"

#include <torch/nativert/executor/ConstantFolder.h>

#include <c10/util/StringUtil.h>
#include <folly/ScopeGuard.h>
#include <sstream>

#include "velox/experimental/wave/common/Cuda.h"

namespace torch::wave {

namespace {

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

const c10::FunctionSchema* findSchema(std::string_view target) {
  auto atoms = c10::split(target, '.');
  if (atoms.size() < 3 || atoms[atoms.size() - 3] != "aten") {
    return nullptr;
  }
  const auto* schema = findFunctionSchema(target);
  return schema;
}

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

// --- Thread-local WaveGraph ---

WaveGraph*& waveGraph() {
  static thread_local WaveGraph* threadWaveGraph{nullptr};
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

// --- WaveGraph ---

WaveGraph::WaveGraph(nativert::Graph& graph, ValueTypes types, OptimizeOnlyTag)
    : types_(std::move(types)), graph_(&graph) {}

std::unique_ptr<WaveGraph> WaveGraph::optimizeOnly(
    nativert::Graph& graph,
    const ValueTypes& types) {
  auto wg = std::unique_ptr<WaveGraph>(
      new WaveGraph(graph, types, OptimizeOnlyTag{}));
  auto* prevWaveGraph = waveGraph();
  waveGraph() = wg.get();
  SCOPE_EXIT {
    waveGraph() = prevWaveGraph;
  };
  wg->normalizeAndAnnotateGraph();
  for (auto* v : graph.values()) {
    wg->idToValue_[v->id()] = v;
  }
  wg->optimizer_ = std::make_unique<Optimizer>(*wg);
  wg->optimizer_->optimizeGraph(wg->graph_);
  return wg;
}

WaveGraph::WaveGraph(std::shared_ptr<ModelContext> modelContext)
    : graph_(modelContext->graph), modelContext_(std::move(modelContext)) {
  waveGraph() = this;
  SCOPE_EXIT {
    waveGraph() = nullptr;
  };

  initValueTypes(*graph_, types_, metaStorage_);

  normalizeAndAnnotateGraph();
  for (auto* v : graph_->values()) {
    idToValue_[v->id()] = v;
  }
  optimizer_ = std::make_unique<Optimizer>(*this);
  optimizer_->optimizeGraph(graph_);
  createdValueDtypes_.clear();
  ParallelNodes parallelNodes;
  auto* lastProjectNode = parallelNodes.makeParallelNodes(*graph_);

  CompileCtx ctx(*this);
  compileCtx_ = &ctx;
  SCOPE_EXIT {
    compileCtx_ = nullptr;
  };

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

  for (const auto& node : nodes_) {
    auto* inv = node->kernels();
    if (!inv || !inv->kernel()) {
      continue;
    }
    auto* ck = inv->kernel();
    ck->warmup();
    auto info = ck->kernelInfo();
    LOG(INFO) << "Kernel " << ck->entryPoint() << ": " << info.toString();
  }

  // Build standaloneIndices_ by walking all launches across all compiled nodes.
  for (const auto& node : nodes_) {
    const auto* inv = node->kernels();
    if (!inv) {
      continue;
    }
    for (const auto& op : inv->ops()) {
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
            if (launch.op) {
              for (const auto& desc : launch.op->outputDescs()) {
                if (desc.viewNode) {
                  auto it = op.nodeMap().find(desc.viewNode);
                  auto* actualNode =
                      it != op.nodeMap().end() ? it->second : desc.viewNode;
                  if (standaloneIndices_.find(actualNode) ==
                      standaloneIndices_.end()) {
                    standaloneIndices_[actualNode] =
                        static_cast<int32_t>(standaloneIndices_.size());
                  }
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

void WaveGraph::normalizeAndAnnotateGraph() {
  for (auto& node : graph_->nodes()) {
    const auto* md = Registry::metadata(node.target());
    if (md && md->normalize) {
      md->normalize(&node, types_);
    }
    const auto* schema = findSchema(node.target());
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
        node.addAttribute(
            nativert::Attribute{
                std::string(schemaArg.name()), std::move(defaultVal)});
      }
    }

    if (md && md->makeMultiKernelVariant) {
      auto* lastNode = md->makeMultiKernelVariant(&node, this);
      auto inputs = inputValues(&node);
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

void WaveGraph::optimizeNode(const nativert::Node* node) {
  TORCH_CHECK(optimizer_, "optimizeNode called outside WaveGraph construction");
  optimizer_->optimizeNode(node);
}

std::string WaveGraph::toString(Listing mode) const {
  std::stringstream ss;
  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (i > 0) {
      ss << "\n";
    }
    ss << "Node " << i << ":\n";
    ss << nodes_[i]->toString(mode, static_cast<int32_t>(i));
  }
  return ss.str();
}

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
    types_.constraints.resize(id + 1);
  }
  types_.types[id] = metaPtr;
  idToValue_[id] = value;
}

nativert::Value* WaveGraph::newTensorValue(
    nativert::Node* node,
    std::string_view name,
    c10::ScalarType dtype) {
  auto uname = uniqueName(name);
  auto* value =
      node->addOutput(uname, nativert::Type(nativert::Type::Kind::Tensor));
  registerTensorMeta(value, dtype);
  createdValueDtypes_[value->id()] = dtype;
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
    types_.constraints.resize(id + 1);
  }
  idToValue_[id] = value;
  createdValueDtypes_[value->id()] = dtype;
  return value;
}

bool WaveGraph::isCreatedValue(ValueCP value) const {
  return createdValueDtypes_.count(value->id()) > 0;
}

nativert::Value* WaveGraph::duplicateValue(ValueCP original) {
  if (!placeholderNode_) {
    placeholderNode_ = graph_->createNode("tw.placeholder", {});
  }
  auto it = createdValueDtypes_.find(original->id());
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
  return result;
}

nativert::Graph* variantNodeGraph(WaveGraph* waveGraph) {
  auto* variantGraph = waveGraph->currentVariantGraph();
  return variantGraph ? variantGraph : waveGraph->graph();
}

nativert::Value* newVariantTensorValue(
    nativert::Node* node,
    WaveGraph* waveGraph,
    std::string_view name,
    c10::ScalarType dtype) {
  if (!waveGraph->currentVariantGraph()) {
    return waveGraph->newTensorValue(node, name, dtype);
  }
  auto* mainValue =
      waveGraph->newTensorValue(waveGraph->placeholderNode(), name, dtype);
  auto* copy =
      node->addOutput(std::string(mainValue->name()), mainValue->type());
  copy->setId(mainValue->id());
  return copy;
}

nativert::Value* newVariantScalarValue(
    nativert::Node* node,
    WaveGraph* waveGraph,
    std::string_view name,
    c10::ScalarType dtype) {
  if (!waveGraph->currentVariantGraph()) {
    return waveGraph->newScalarValue(node, name, dtype);
  }
  auto* mainValue =
      waveGraph->newScalarValue(waveGraph->placeholderNode(), name, dtype);
  auto* copy =
      node->addOutput(std::string(mainValue->name()), mainValue->type());
  copy->setId(mainValue->id());
  return copy;
}

void copyOriginalOutputs(
    nativert::Node* node,
    NodeCP original,
    WaveGraph* waveGraph) {
  if (!waveGraph->currentVariantGraph()) {
    return;
  }
  auto* graph = waveGraph->currentVariantGraph();
  for (const auto* output : original->outputs()) {
    auto* newValue =
        node->addOutput(std::string(output->name()), output->type());
    newValue->setId(output->id());
    if (output->type().kind() == nativert::Type::Kind::TensorList) {
      auto* unpack = graph->insertNode(
          "prim.ListUnpack", {{"input", newValue}}, node->metadata());
      for (auto* elem : output->getListElements()) {
        auto* newElem =
            unpack->addOutput(std::string(elem->name()), elem->type());
        newElem->setId(elem->id());
      }
    }
  }
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
      auto id = value->id();
      TORCH_CHECK(
          id >= 0 && static_cast<size_t>(id) < types.types.size(),
          "Value id out of range: ",
          id);
      auto meta = std::make_unique<nativert::TensorMeta>(it->second);
      types.constraints[id].rank = static_cast<int8_t>(meta->dim());
      types.types.at(id) = meta.get();
      metaStore.push_back(std::move(meta));
    }
  }
}

} // namespace torch::wave
