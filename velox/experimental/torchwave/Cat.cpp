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

#include "velox/experimental/torchwave/Cat.h"
#include "velox/experimental/torchwave/Compile.h"
#include "velox/experimental/torchwave/KernelOperation.h"
#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"
#include "velox/experimental/torchwave/WaveGraph.h"

#include <ATen/ATen.h>
#include <iostream>

namespace torch::wave {

namespace {

SizeExpr translateSizeExpr(const SizeExpr& expr, const FormalToActual& map) {
  SizeExpr result;
  result.op = expr.op;
  for (auto id : expr.values) {
    auto it = map.find(id);
    result.values.push_back(it != map.end() ? it->second : id);
  }
  for (const auto& child : expr.args) {
    result.args.push_back(translateSizeExpr(child, map));
  }
  return result;
}

// Returns true if any node in the producer chain of 'node' (stopping at
// subgraphInputs) has a shapeSetOnDevice return.
bool hasShapeOnDeviceInChain(
    NodeCP node,
    const std::unordered_set<ValueCP>& subgraphInputs,
    std::unordered_set<NodeCP>& visited) {
  if (!visited.insert(node).second) {
    return false;
  }
  auto* meta = Registry::metadata(node->target());
  if (meta) {
    for (const auto& rm : meta->returnMeta) {
      if (rm.shapeSetOnDevice) {
        return true;
      }
    }
  }
  for (const auto& input : node->inputs()) {
    if (subgraphInputs.count(input.value)) {
      continue;
    }
    auto* producer = input.value->producer();
    if (producer &&
        hasShapeOnDeviceInChain(producer, subgraphInputs, visited)) {
      return true;
    }
  }
  return false;
}

struct CatInputInfo {
  nativert::ValueId formalId;
  SizeExpr sizeExpr;
  OutputReserveFunc reserveShape;
  bool hasShapeOnDevice;
  bool isSubgraphInput{false};
};

void catSetOutputs(
    KernelOperation* op,
    NodeCP node,
    const std::unordered_set<ValueCP>& subgraphInputs,
    std::vector<ValueCP>& outputValues,
    std::vector<OutputDesc>& outputDescs,
    bool /*inMemory*/,
    bool /*callerIsElementwise*/) {
  auto elements = node->inputs()[0].value->getListElements();

  // Call setOutputs for each element's producer, forcing to memory.
  for (auto* elem : elements) {
    auto* producer = elem->producer();
    if (producer && !subgraphInputs.count(elem)) {
      producer = op->executableNode(producer);
      op->setOutputs(
          producer, subgraphInputs, outputValues, outputDescs, true, false);
    }
  }

  std::vector<CatInputInfo> inputInfos;
  inputInfos.reserve(elements.size());
  for (auto* elem : elements) {
    if (subgraphInputs.count(elem)) {
      SizeExpr sizeExpr;
      sizeExpr.op = SizeShortcut::kMax;
      sizeExpr.values.push_back(elem->id());
      inputInfos.push_back(
          {elem->id(), std::move(sizeExpr), nullptr, false, true});
      continue;
    }
    int32_t descIdx = -1;
    for (size_t i = 0; i < outputValues.size(); ++i) {
      if (outputValues[i] == elem) {
        descIdx = i;
        break;
      }
    }
    TORCH_CHECK(
        descIdx >= 0, "No OutputDesc created for cat input %v", elem->id());
    auto& desc = outputDescs[descIdx];
    desc.delegated = true;

    std::unordered_set<NodeCP> visited;
    bool hasSod = desc.shapeSetOnDevice;
    if (!hasSod && elem->producer()) {
      auto* prodNode = op->executableNode(elem->producer());
      hasSod = hasShapeOnDeviceInChain(prodNode, subgraphInputs, visited);
    }

    inputInfos.push_back(
        {elem->id(), desc.sizeExpr, desc.reserveShape, hasSod, false});
  }

  // Create the cat output desc.
  auto catOutputValue = node->outputs()[0];
  auto catFormalId = catOutputValue->id();

  auto& types = waveGraph()->types();
  auto catId = catOutputValue->id();
  TORCH_CHECK(
      catId >= 0 && static_cast<size_t>(catId) < types.types.size() &&
          types.types[catId],
      "No TensorMeta for cat output value ",
      catId);
  auto dtype = types.types[catId]->dtype();

  OutputDesc catDesc;
  catDesc.sizeExpr.op = SizeShortcut::kSum;
  for (auto* elem : elements) {
    catDesc.sizeExpr.values.push_back(elem->id());
  }

  catDesc.reserveShape =
      [inputInfos, catFormalId, dtype](
          NodeCP /*unused*/,
          nativert::ExecutionFrame& frame,
          FormalToActual map) -> std::vector<std::vector<Dim>> {
    std::vector<int64_t> sizes(inputInfos.size());
    int64_t totalSize = 0;
    for (size_t i = 0; i < inputInfos.size(); ++i) {
      const auto& info = inputInfos[i];
      if (info.reserveShape) {
        auto shapes = info.reserveShape(nullptr, frame, map);
        int64_t s = 1;
        for (auto d : shapes[0]) {
          s *= d;
        }
        sizes[i] = s;
      } else if (info.sizeExpr.op != SizeShortcut::kNone) {
        auto actual = translateSizeExpr(info.sizeExpr, map);
        sizes[i] = actual.numElements(&frame);
      }
      totalSize += sizes[i];
    }

    // Allocate or resize the cat output tensor.
    auto catActualId = catFormalId;
    if (auto it = map.find(catFormalId); it != map.end()) {
      catActualId = it->second;
    }
    auto& existing = frame.getIValue(catActualId);
    at::Tensor catTensor;
    if (existing.isTensor() && existing.toTensor().is_cuda()) {
      catTensor = existing.toTensor();
      if (catTensor.numel() != totalSize) {
        catTensor.resize_({totalSize});
      }
    } else {
      catTensor = at::empty(
          {totalSize}, at::TensorOptions().dtype(dtype).device(at::kCUDA));
      frame.setIValue(catActualId, catTensor);
    }

    // Set up views for computed inputs. Subgraph inputs are not given
    // views — they are copied into the cat output by __copy at kernel time.
    bool canComputeOffset = true;
    int64_t offset = 0;
    for (size_t i = 0; i < inputInfos.size(); ++i) {
      if (inputInfos[i].isSubgraphInput) {
        offset += sizes[i];
        continue;
      }
      auto inputActualId = inputInfos[i].formalId;
      if (auto it = map.find(inputInfos[i].formalId); it != map.end()) {
        inputActualId = it->second;
      }
      if (canComputeOffset && !inputInfos[i].hasShapeOnDevice) {
        auto view = catTensor.narrow(0, offset, sizes[i]);
        if (WaveConfig::get().trace & WaveConfig::kTensors) {
          std::cout << "  cat view v" << inputActualId << " of v" << catActualId
                    << " " << traceIValue(c10::IValue(view)) << std::endl;
        }
        frame.setIValue(inputActualId, std::move(view));
        offset += sizes[i];
      } else {
        canComputeOffset = false;
        if (WaveConfig::get().trace & WaveConfig::kTensors) {
          std::cout << "  cat view v" << inputActualId << " of v" << catActualId
                    << " " << traceIValue(c10::IValue(catTensor)) << std::endl;
        }
        frame.setIValue(inputActualId, catTensor);
      }
    }

    return {{static_cast<Dim>(totalSize)}};
  };

  addOrUpdateOutput(
      outputValues, outputDescs, catOutputValue, std::move(catDesc));
}

// Recursively collects the leaf elements of nested cats.
void flattenCatElements(ValueCP value, std::vector<nativert::Value*>& result) {
  auto* producer = value->producer();
  if (producer && producer->target() == "torch.ops.aten.cat.default") {
    auto elements = producer->inputs()[0].value->getListElements();
    for (auto* elem : elements) {
      flattenCatElements(elem, result);
    }
  } else {
    result.push_back(const_cast<nativert::Value*>(value));
  }
}

// Returns the cumsum node if 'node' is cat(zeros(size=[1]), cumsum(...)).
NodeCP isExclusiveSumPattern(NodeCP node) {
  auto elements = node->inputs()[0].value->getListElements();
  if (elements.size() != 2) {
    return nullptr;
  }
  auto* zerosProducer = elements[0]->producer();
  if (!zerosProducer ||
      zerosProducer->target() != "torch.ops.aten.zeros.default") {
    return nullptr;
  }
  const auto* sizeAttr = zerosProducer->tryGetAttribute("size");
  if (!sizeAttr ||
      !std::holds_alternative<std::vector<int64_t>>(sizeAttr->value)) {
    return nullptr;
  }
  const auto& size = std::get<std::vector<int64_t>>(sizeAttr->value);
  if (size.size() != 1 || size[0] != 1) {
    return nullptr;
  }
  auto* cumsumProducer = elements[1]->producer();
  if (!cumsumProducer ||
      cumsumProducer->target() != "torch.ops.aten.cumsum.default") {
    return nullptr;
  }
  return cumsumProducer;
}

std::vector<std::pair<ValueCP, ValueCP>>
catMaybeReplace(NodeCP node, ValueTypes& /*types*/, WaveGraph& waveGraph) {
  auto* graph = waveGraph.graph();

  if (auto* cumsumNode = isExclusiveSumPattern(node)) {
    auto* cumsumInput = cumsumNode->inputs()[0].value;
    auto& types = waveGraph.types();
    if (types.rank(cumsumInput) != 1) {
      return {};
    }
    auto inputId = cumsumInput->id();
    auto dtype = c10::ScalarType::Long;
    const auto* dtypeAttr = cumsumNode->tryGetAttribute("dtype");
    if (dtypeAttr &&
        std::holds_alternative<c10::ScalarType>(dtypeAttr->value)) {
      dtype = std::get<c10::ScalarType>(dtypeAttr->value);
    } else if (
        dtypeAttr && std::holds_alternative<std::string>(dtypeAttr->value)) {
      auto dtypeStr = std::get<std::string>(dtypeAttr->value);
      static const std::unordered_map<std::string, c10::ScalarType>
          kNameToType = {
              {"Float", c10::ScalarType::Float},
              {"Double", c10::ScalarType::Double},
              {"Half", c10::ScalarType::Half},
              {"BFloat16", c10::ScalarType::BFloat16},
              {"Long", c10::ScalarType::Long},
              {"Int", c10::ScalarType::Int},
              {"Short", c10::ScalarType::Short},
              {"Byte", c10::ScalarType::Byte},
              {"Bool", c10::ScalarType::Bool}};
      auto it = kNameToType.find(dtypeStr);
      if (it != kNameToType.end()) {
        dtype = it->second;
      }
    } else if (
        inputId < static_cast<int>(types.types.size()) &&
        types.types[inputId]) {
      auto inputDtype = types.types[inputId]->dtype();
      dtype = c10::isIntegralType(inputDtype, true) ? c10::ScalarType::Long
                                                    : inputDtype;
    }
    auto* exclusiveSum = graph->createNode(
        "torch.ops.aten.exclusive_sum.default", {{"input", cumsumInput}});
    exclusiveSum->addAttribute({"dim", static_cast<int64_t>(0)});
    exclusiveSum->addAttribute({"dtype", std::string(c10::toString(dtype))});
    graph->insertBefore(exclusiveSum, const_cast<nativert::Node*>(node));
    auto* newOutput =
        waveGraph.newTensorValue(exclusiveSum, "exclusive_sum", dtype);
    return {{node->outputs()[0], newOutput}};
  }

  auto elements = node->inputs()[0].value->getListElements();
  bool hasNestedCat = false;
  for (auto* elem : elements) {
    auto* producer = elem->producer();
    if (producer && producer->target() == "torch.ops.aten.cat.default") {
      hasNestedCat = true;
      break;
    }
  }
  if (!hasNestedCat) {
    return {};
  }

  std::vector<nativert::Value*> flatElements;
  for (auto* elem : elements) {
    flattenCatElements(elem, flatElements);
  }

  auto& types = waveGraph.types();
  auto firstId = flatElements[0]->id();
  auto dtype =
      (firstId < static_cast<int>(types.types.size()) && types.types[firstId])
      ? types.types[firstId]->dtype()
      : c10::ScalarType::Long;

  auto* listPack = graph->createListPack(
      std::move(flatElements), nativert::Type::Kind::Tensor);
  graph->insertBefore(listPack, const_cast<nativert::Node*>(node));

  auto* newCat = graph->createNode(
      "torch.ops.aten.cat.default", {{"tensors", listPack->outputs()[0]}});
  for (const auto& attr : node->attributes()) {
    newCat->addAttribute({attr.name, constantToIValue(attr.value).toInt()});
  }
  graph->insertBefore(newCat, const_cast<nativert::Node*>(node));
  auto* newOutput = waveGraph.newTensorValue(newCat, "cat_result", dtype);

  return {{node->outputs()[0], newOutput}};
}

void catSpecialForm(
    NodeCP node,
    const std::vector<ResultSpec>& resultSpecs,
    CompileCtx* ctx) {
  auto* listPackProducer = node->inputs()[0].value->producer();
  if (listPackProducer) {
    ctx->generatingOp()->allNodes().insert(listPackProducer);
  }
  auto elements = node->inputs()[0].value->getListElements();
  auto* catOutput = node->outputs()[0];

  // Determine element size from the first element's dtype.
  auto& types = ctx->waveGraph().types();
  auto dtype = types.types[elements[0]->id()]->dtype();
  auto elemSize = static_cast<int32_t>(c10::elementSize(dtype));

  // Check which elements have shape set on device within this kernel op.
  std::vector<bool> sizeSetInOp(elements.size(), false);
  for (size_t i = 0; i < elements.size(); ++i) {
    std::unordered_set<ValueCP> visited;
    sizeSetInOp[i] = ctx->isSizeSetInThisOp(elements[i], visited);
  }

  auto typeName = cudaTypeString(dtype);

  bool needsViewFixup = false;
  ctx->emitCode("  {\n  int64_t offset = 0;\n");
  int64_t lastAccumulated = -1;
  for (size_t i = 0; i < elements.size(); ++i) {
    auto* elem = elements[i];
    auto* producer = elem->producer();
    if (!producer || ctx->generatingOp()->isInput(elem)) {
      auto& op = *ctx->generatingOp();
      std::string incrExpr;
      for (size_t j = lastAccumulated + 1; j < i; ++j) {
        auto p = ctx->param(elements[j], op);
        if (!incrExpr.empty()) {
          incrExpr += " + ";
        }
        incrExpr += p + "->numEl";
      }
      if (!incrExpr.empty()) {
        ctx->emitCode("  offset += " + incrExpr + ";\n");
      }
      lastAccumulated = static_cast<int64_t>(i) - 1;
      ctx->emitCopy(elem, catOutput, "offset", typeName);
    } else {
      std::vector<ResultSpec> prodSpecs;
      for (auto* output : producer->outputs()) {
        prodSpecs.push_back({output, {}});
      }
      ctx->fusedCode(producer, prodSpecs);
    }

    if (sizeSetInOp[i]) {
      needsViewFixup = true;
    }

    if (needsViewFixup && i + 1 < elements.size()) {
      auto& op = *ctx->generatingOp();
      for (size_t j = lastAccumulated + 1; j <= i; ++j) {
        auto p = ctx->param(elements[j], op);
        ctx->emitCode("  offset += " + p + "->dims[0];\n");
      }
      lastAccumulated = static_cast<int64_t>(i);
      ctx->callView(catOutput, elements[i + 1], "offset", elemSize);
      ctx->emitCode(
          "  TRACE0(printf(\"cat view[" + std::to_string(i + 1) +
          "] offset=%ld\\n\", (long)offset));\n");
    }
  }
  if (needsViewFixup) {
    auto& op = *ctx->generatingOp();
    for (auto j = lastAccumulated + 1;
         j < static_cast<int64_t>(elements.size());
         ++j) {
      auto p = ctx->param(elements[j], op);
      ctx->emitCode("  offset += " + p + "->dims[0];\n");
    }
    ctx->emitCode(
        "  TRACE0(printf(\"cat final offset=%ld\\n\", (long)offset));\n");
    auto catP = ctx->param(catOutput, op);
    ctx->emitCode(
        "  if (threadIdx.x == 0) { " + catP +
        "->dims[0] = offset; }\n"
        "  __syncthreads();\n");
  }
  ctx->emitCode("  }\n");
  ctx->markPlaced(node);
}

} // namespace

void registerCatMetadata() {
  MetadataBuilder("torch.ops.aten.cat.default")
      .sizeShortcut(SizeShortcut::kSum)
      .sizeOrdinal({0})
      .sizeArgsList({true})
      .only1d()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            const auto& inputs = node->inputs();
            if (inputs.empty()) {
              return {};
            }
            auto elements = inputs[0].value->getListElements();
            if (elements.empty()) {
              return {};
            }
            return {{.rank = types.rank(elements[0])}};
          })
      .maybeReplace(catMaybeReplace)
      .setOutputs(catSetOutputs)
      .specialForm(catSpecialForm)
      .registerOp();
}

} // namespace torch::wave
