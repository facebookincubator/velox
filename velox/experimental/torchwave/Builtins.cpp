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
#include "velox/experimental/torchwave/CompiledOp.h"
#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"

#include <cmath>
#include <limits>

namespace torch::wave {

/// Resolves the output ScalarType and dtype string from a node's dtype
/// attribute and input tensor type. Returns {outDtype, dtypeStr}.
std::pair<c10::ScalarType, std::string> resolveOutDtype(
    NodeCP single,
    WaveGraph* waveGraph) {
  auto inputId = single->inputs()[0].value->id();
  auto inputDtype = waveGraph->types().types[inputId]->dtype();
  auto outDtype = inputDtype;
  std::string dtypeStr;
  const auto* dtypeAttr = single->tryGetAttribute("dtype");
  if (dtypeAttr) {
    if (std::holds_alternative<c10::ScalarType>(dtypeAttr->value)) {
      outDtype = std::get<c10::ScalarType>(dtypeAttr->value);
      dtypeStr = c10::toString(outDtype);
    } else if (std::holds_alternative<std::string>(dtypeAttr->value)) {
      dtypeStr = std::get<std::string>(dtypeAttr->value);
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
        outDtype = it->second;
      }
    }
  } else {
    dtypeStr = c10::toString(outDtype);
  }
  return {outDtype, dtypeStr};
}

nativert::Node* makeSumVariant(NodeCP single, WaveGraph* waveGraph) {
  auto* graph = variantNodeGraph(waveGraph);
  auto [outDtype, dtypeStr] = resolveOutDtype(single, waveGraph);

  auto* headNode =
      graph->createNode("tw.sum_head", {{"input", single->inputs()[0].value}});
  headNode->addAttribute({"dtype", dtypeStr});
  auto* headOutput =
      newVariantTensorValue(headNode, waveGraph, "sum_blocks", outDtype);

  auto* finalNode = graph->createNode("tw.sum_final", {{"input", headOutput}});
  finalNode->addAttribute({"dtype", dtypeStr});
  copyOriginalOutputs(finalNode, single, waveGraph);
  return finalNode;
}

nativert::Node* makeCumsumVariant(NodeCP single, WaveGraph* waveGraph) {
  auto* graph = variantNodeGraph(waveGraph);
  auto [outDtype, dtypeStr] = resolveOutDtype(single, waveGraph);

  auto* headNode = graph->createNode(
      "tw.cumsum_head", {{"input", single->inputs()[0].value}});
  headNode->addAttribute({"dtype", dtypeStr});
  auto* headOutput =
      newVariantTensorValue(headNode, waveGraph, "cumsum_counts", outDtype);

  auto* addSizesNode =
      graph->createNode("tw.cumsum_add_sizes", {{"input", headOutput}});
  addSizesNode->addAttribute({"dtype", dtypeStr});
  auto* addSizesOutput = newVariantScalarValue(
      addSizesNode, waveGraph, "cumsum_link", c10::ScalarType::Int);

  std::vector<nativert::NamedArgument> finalInputs = {
      {"input", single->inputs()[0].value},
      {"counts", headOutput},
      {"link", addSizesOutput}};
  auto* finalNode =
      graph->createNode("tw.cumsum_final", std::move(finalInputs));
  finalNode->addAttribute({"dtype", dtypeStr});
  const auto* dimAttr = single->tryGetAttribute("dim");
  if (dimAttr) {
    finalNode->addAttribute({dimAttr->name, std::get<int64_t>(dimAttr->value)});
  }
  copyOriginalOutputs(finalNode, single, waveGraph);
  return finalNode;
}

nativert::Node* makeExclusiveSumVariant(NodeCP single, WaveGraph* waveGraph) {
  auto* graph = variantNodeGraph(waveGraph);
  auto [outDtype, dtypeStr] = resolveOutDtype(single, waveGraph);

  auto* headNode = graph->createNode(
      "tw.exclusive_sum_head", {{"input", single->inputs()[0].value}});
  headNode->addAttribute({"dtype", dtypeStr});
  auto* headOutput =
      newVariantTensorValue(headNode, waveGraph, "exsum_counts", outDtype);

  auto* addSizesNode =
      graph->createNode("tw.exclusive_sum_add_sizes", {{"input", headOutput}});
  addSizesNode->addAttribute({"dtype", dtypeStr});
  auto* addSizesOutput = newVariantScalarValue(
      addSizesNode, waveGraph, "exsum_link", c10::ScalarType::Int);

  auto* finalNode = graph->createNode(
      "tw.exclusive_sum_final",
      {{"input", single->inputs()[0].value},
       {"counts", headOutput},
       {"link", addSizesOutput}});
  finalNode->addAttribute({"dtype", dtypeStr});
  copyOriginalOutputs(finalNode, single, waveGraph);
  return finalNode;
}

nativert::Node* makeMaskedSelectVariant(NodeCP single, WaveGraph* waveGraph) {
  auto* graph = variantNodeGraph(waveGraph);

  auto* headNode = graph->createNode(
      "tw.masked_select_head",
      {{"input", single->inputs()[0].value},
       {"mask", single->inputs()[1].value}});
  auto* headOutput = newVariantTensorValue(
      headNode, waveGraph, "masked_select_counts", c10::ScalarType::Int);

  auto* addSizesNode =
      graph->createNode("tw.add_sizes", {{"input", headOutput}});
  auto* addSizesOutput = newVariantScalarValue(
      addSizesNode, waveGraph, "masked_select_total", c10::ScalarType::Int);

  auto* finalNode = graph->createNode(
      "tw.masked_select_final",
      {{"input", single->inputs()[0].value},
       {"mask", single->inputs()[1].value},
       {"counts", headOutput},
       {"total", addSizesOutput}});
  copyOriginalOutputs(finalNode, single, waveGraph);
  return finalNode;
}

nativert::Node* makeCumsumCgVariant(NodeCP single, WaveGraph* waveGraph) {
  auto* graph = variantNodeGraph(waveGraph);
  auto [outDtype, dtypeStr] = resolveOutDtype(single, waveGraph);
  auto* cgNode =
      graph->createNode("tw.cumsum_cg", {{"input", single->inputs()[0].value}});
  cgNode->addAttribute({"dtype", dtypeStr});
  const auto* dimAttr = single->tryGetAttribute("dim");
  if (dimAttr) {
    cgNode->addAttribute({dimAttr->name, std::get<int64_t>(dimAttr->value)});
  }
  copyOriginalOutputs(cgNode, single, waveGraph);
  newVariantTensorValue(cgNode, waveGraph, "cumsum_counts", outDtype);
  return cgNode;
}

nativert::Node* makeExclusiveSumCgVariant(NodeCP single, WaveGraph* waveGraph) {
  auto* graph = variantNodeGraph(waveGraph);
  auto* cgNode = graph->createNode(
      "tw.exclusive_sum_cg", {{"input", single->inputs()[0].value}});
  auto [outDtype, dtypeStr] = resolveOutDtype(single, waveGraph);
  cgNode->addAttribute({"dtype", dtypeStr});
  copyOriginalOutputs(cgNode, single, waveGraph);
  newVariantTensorValue(cgNode, waveGraph, "exsum_counts", outDtype);
  return cgNode;
}

nativert::Node* makeMaskedSelectCgVariant(NodeCP single, WaveGraph* waveGraph) {
  auto* graph = variantNodeGraph(waveGraph);
  auto* cgNode = graph->createNode(
      "tw.masked_select_cg",
      {{"input", single->inputs()[0].value},
       {"mask", single->inputs()[1].value}});
  copyOriginalOutputs(cgNode, single, waveGraph);
  newVariantTensorValue(
      cgNode, waveGraph, "masked_select_counts", c10::ScalarType::Int);
  return cgNode;
}

nativert::Node* makeSumCgVariant(NodeCP single, WaveGraph* waveGraph) {
  auto* graph = variantNodeGraph(waveGraph);
  auto [outDtype, dtypeStr] = resolveOutDtype(single, waveGraph);
  auto* cgNode =
      graph->createNode("tw.sum_cg", {{"input", single->inputs()[0].value}});
  cgNode->addAttribute({"dtype", dtypeStr});
  copyOriginalOutputs(cgNode, single, waveGraph);
  newVariantTensorValue(cgNode, waveGraph, "sum_partials", outDtype);
  return cgNode;
}

namespace {

void resolveDtypeFromInput(nativert::Node* node, const ValueTypes& types) {
  if (node->inputs().empty()) {
    return;
  }
  if (node->tryGetAttribute("dtype") || node->tryGetInput("dtype")) {
    return;
  }
  auto inputId = node->inputs()[0].value->id();
  if (inputId < static_cast<int>(types.types.size()) && types.types[inputId]) {
    auto inputDtype = types.types[inputId]->dtype();
    auto outDtype = c10::isIntegralType(inputDtype, /*includeBool=*/true)
        ? c10::ScalarType::Long
        : inputDtype;
    node->addAttribute({"dtype", std::string(c10::toString(outDtype))});
  }
}

std::vector<ValueConstraint> rank1Constraint(
    NodeCP /*node*/,
    const ValueTypes& /*types*/) {
  return {{.rank = 1}};
}

// Returns a scalar value from an attribute (int64_t or double) or a symint
// input named 'name'. For double attributes, returns ceil.
int64_t integerParamByName(
    NodeCP node,
    const char* name,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map) {
  const auto* attr = node->tryGetAttribute(name);
  if (attr) {
    if (std::holds_alternative<int64_t>(attr->value)) {
      return std::get<int64_t>(attr->value);
    }
    return static_cast<int64_t>(std::ceil(std::get<double>(attr->value)));
  }
  const auto* input = node->tryGetInput(name);
  TORCH_CHECK(
      input, node->target(), ": missing '", name, "' attribute or input");
  return paramSymInt(input->value, frame, map);
}

void resolveDefaultDtype(nativert::Node* node, const ValueTypes& /*types*/) {
  if (node->tryGetAttribute("dtype") || node->tryGetInput("dtype")) {
    return;
  }
  node->addAttribute({"dtype", std::string("Float")});
}

void resolveArangeDtype(nativert::Node* node, const ValueTypes& /*types*/) {
  if (node->tryGetAttribute("dtype") || node->tryGetInput("dtype")) {
    return;
  }
  bool hasFloat = false;
  for (const auto& input : node->inputs()) {
    auto kind = input.value->type().kind();
    if (kind == nativert::Type::Kind::SymFloat) {
      hasFloat = true;
      break;
    }
  }
  node->addAttribute({"dtype", std::string(hasFloat ? "Float" : "Long")});
}

void resolveNanToNumDefaults(nativert::Node* node, const ValueTypes& types) {
  double posinfDefault = std::numeric_limits<double>::max();
  double neginfDefault = std::numeric_limits<double>::lowest();
  if (!node->inputs().empty()) {
    auto inputId = node->inputs()[0].value->id();
    if (inputId < static_cast<int>(types.types.size()) &&
        types.types[inputId]) {
      switch (types.types[inputId]->dtype()) {
        case c10::ScalarType::Half:
          posinfDefault = 65504.0;
          neginfDefault = -65504.0;
          break;
        case c10::ScalarType::Float:
          posinfDefault = std::numeric_limits<float>::max();
          neginfDefault = std::numeric_limits<float>::lowest();
          break;
        case c10::ScalarType::BFloat16:
          // BFloat16 max value
          posinfDefault = 0x1.FEp+127;
          neginfDefault = -0x1.FEp+127;
          break;
        default:
          break;
      }
    }
  }
  auto resolveAttr = [&](const char* name, double defaultVal) {
    if (node->tryGetInput(name)) {
      return;
    }
    const auto* attr = node->tryGetAttribute(name);
    if (!attr) {
      node->addAttribute({name, defaultVal});
    } else if (std::holds_alternative<nativert::None>(attr->value)) {
      const_cast<nativert::Attribute*>(attr)->value = defaultVal;
    }
  };
  resolveAttr("nan", 0.0);
  resolveAttr("posinf", posinfDefault);
  resolveAttr("neginf", neginfDefault);
}

void resolveDtypeFromInputExact(nativert::Node* node, const ValueTypes& types) {
  if (node->inputs().empty()) {
    return;
  }
  if (node->tryGetAttribute("dtype") || node->tryGetInput("dtype")) {
    return;
  }
  auto inputId = node->inputs()[0].value->id();
  if (inputId < static_cast<int>(types.types.size()) && types.types[inputId]) {
    auto inputDtype = types.types[inputId]->dtype();
    node->addAttribute({"dtype", std::string(c10::toString(inputDtype))});
  }
}

std::vector<ValueConstraint> sizeAttrRankConstraint(
    NodeCP node,
    const ValueTypes& /*types*/) {
  const auto* sizeAttr = node->tryGetAttribute("size");
  if (sizeAttr) {
    const auto& size = std::get<std::vector<int64_t>>(sizeAttr->value);
    return {{.rank = static_cast<int8_t>(size.size())}};
  }
  return {};
}

std::vector<std::vector<Dim>> sizeAttrReserveShape(
    NodeCP node,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map) {
  const auto* sizeAttr = node->tryGetAttribute("size");
  if (sizeAttr) {
    const auto& size = std::get<std::vector<int64_t>>(sizeAttr->value);
    return {{size.begin(), size.end()}};
  }
  auto* sizeInput = node->tryGetInput("size");
  TORCH_CHECK(sizeInput, node->target(), ": missing size attribute or input");
  auto it = map.find(sizeInput->value->id());
  TORCH_CHECK(it != map.end(), node->target(), ": size not in FormalToActual");
  auto& ivalue = frame.getIValue(it->second);
  if (!ivalue.isNone()) {
    auto size = ivalue.toIntVector();
    return {{size.begin(), size.end()}};
  }
  auto& idToValue = waveGraph()->idToValue();
  auto valueIt = idToValue.find(it->second);
  TORCH_CHECK(
      valueIt != idToValue.end(),
      node->target(),
      ": size value id not in idToValue");
  auto* producer = valueIt->second->producer();
  TORCH_CHECK(
      producer && producer->target() == "prim.ListPack",
      node->target(),
      ": expected prim.ListPack producer for size");
  std::vector<Dim> size;
  for (const auto& input : producer->inputs()) {
    auto elemIt = map.find(input.value->id());
    TORCH_CHECK(
        elemIt != map.end(),
        node->target(),
        ": ListPack element not in FormalToActual");
    size.push_back(static_cast<Dim>(frame.getIValue(elemIt->second).toInt()));
  }
  return {size};
}

std::vector<std::vector<Dim>> numBlocksShape(
    NodeCP node,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map) {
  auto tensor = paramTensor(node->inputs()[0].value, frame, map);
  auto blockSize = WaveConfig::get().blockSize;
  auto numBlocks =
      static_cast<Dim>((tensor.numel() + blockSize - 1) / blockSize);
  return {{numBlocks}};
}

std::vector<std::vector<Dim>> inputShape(
    NodeCP node,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map) {
  auto tensor = paramTensor(node->inputs()[0].value, frame, map);
  auto sizes = tensor.sizes();
  return {{sizes.begin(), sizes.end()}};
}

std::vector<std::vector<Dim>> inputShapePlusOne(
    NodeCP node,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map) {
  auto tensor = paramTensor(node->inputs()[0].value, frame, map);
  return {{static_cast<Dim>(tensor.numel() + 1)}};
}

void registerRankPreservingStandalone(
    const char* opName,
    std::optional<int32_t> viewOfArgOrdinal = std::nullopt) {
  MetadataBuilder builder(opName);
  builder.sizeOrdinal({0})
      .isStandalone()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            const auto& inputs = node->inputs();
            if (inputs.empty()) {
              return {};
            }
            return {{.rank = types.rank(inputs[0].value)}};
          })
      .maybeReplace(
          [](NodeCP node,
             ValueTypes& types,
             WaveGraph&) -> std::vector<std::pair<ValueCP, ValueCP>> {
            const auto& inputs = node->inputs();
            const auto& outputs = node->outputs();
            if (inputs.empty() || outputs.empty()) {
              return {};
            }
            if (types.rank(inputs[0].value) == 1) {
              return {{outputs[0], inputs[0].value}};
            }
            return {};
          });
  if (viewOfArgOrdinal.has_value()) {
    builder.viewOfArg(*viewOfArgOrdinal);
  }
  builder.registerOp();
}

void registerReshapeLikeOp(const char* opName, const char* shapeAttrName) {
  MetadataBuilder(opName)
      .sizeOrdinal({0})
      .isStandalone()
      .viewOfArg(0)
      .shapeAttr(shapeAttrName)
      .outputConstraints(
          [shapeAttrName](
              NodeCP node,
              const ValueTypes& /*types*/) -> std::vector<ValueConstraint> {
            const auto* attr = node->tryGetAttribute(shapeAttrName);
            if (!attr) {
              return {};
            }
            const auto& shape = std::get<std::vector<int64_t>>(attr->value);
            ValueConstraint constraint;
            if (shape.size() == 1 && shape[0] == -1) {
              constraint.rank = 1;
            } else {
              constraint.rank = static_cast<int8_t>(shape.size());
            }
            return {constraint};
          })
      .maybeReplace(
          [shapeAttrName](NodeCP node, ValueTypes& types, WaveGraph&)
              -> std::vector<std::pair<ValueCP, ValueCP>> {
            const auto& inputs = node->inputs();
            const auto& outputs = node->outputs();
            if (inputs.empty() || outputs.empty()) {
              return {};
            }
            const auto* attr = node->tryGetAttribute(shapeAttrName);
            if (!attr) {
              return {};
            }
            const auto& shape = std::get<std::vector<int64_t>>(attr->value);
            if (shape.size() == 1 && shape[0] == -1 &&
                types.rank(inputs[0].value) == 1) {
              return {{outputs[0], inputs[0].value}};
            }
            return {};
          })
      .registerOp();
}

} // namespace

void registerBuiltins() {
  // Binary arithmetic.
  MetadataBuilder("torch.ops.aten.add.Tensor")
      .elementwise()
      .arithmeticPromotion()
      .registerOp();
  MetadataBuilder("torch.ops.aten.sub.Tensor")
      .elementwise()
      .arithmeticPromotion()
      .registerOp();
  MetadataBuilder("torch.ops.aten.mul.Tensor")
      .elementwise()
      .arithmeticPromotion()
      .registerOp();
  MetadataBuilder("torch.ops.aten.div.Tensor")
      .elementwise()
      .arithmeticPromotion()
      .registerOp();
  MetadataBuilder("torch.ops.aten.remainder.Tensor")
      .elementwise()
      .arithmeticPromotion()
      .registerOp();
  MetadataBuilder("torch.ops.aten.fmod.Tensor")
      .elementwise()
      .arithmeticPromotion()
      .registerOp();
  MetadataBuilder("torch.ops.aten.pow.Tensor_Tensor")
      .elementwise()
      .arithmeticPromotion()
      .registerOp();

  // Binary arithmetic (Tensor, Scalar).
  MetadataBuilder("torch.ops.aten.add.Scalar")
      .elementwiseFunc("__add")
      .arithmeticPromotion()
      .registerOp();
  MetadataBuilder("torch.ops.aten.sub.Scalar")
      .elementwiseFunc("__sub")
      .arithmeticPromotion()
      .registerOp();
  MetadataBuilder("torch.ops.aten.mul.Scalar")
      .elementwiseFunc("__mul")
      .arithmeticPromotion()
      .registerOp();
  MetadataBuilder("torch.ops.aten.div.Scalar")
      .elementwiseFunc("__div")
      .arithmeticPromotion()
      .registerOp();
  MetadataBuilder("torch.ops.aten.remainder.Scalar")
      .elementwiseFunc("__remainder")
      .arithmeticPromotion()
      .registerOp();
  MetadataBuilder("torch.ops.aten.fmod.Scalar")
      .elementwiseFunc("__fmod")
      .arithmeticPromotion()
      .registerOp();
  MetadataBuilder("torch.ops.aten.pow.Tensor_Scalar")
      .elementwiseFunc("__pow")
      .arithmeticPromotion()
      .registerOp();

  // Comparison.
  MetadataBuilder("torch.ops.aten.eq.Tensor").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.ne.Tensor").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.lt.Tensor").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.le.Tensor").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.gt.Tensor").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.ge.Tensor").elementwise().registerOp();

  // Comparison (Tensor, Scalar).
  MetadataBuilder("torch.ops.aten.eq.Scalar")
      .elementwiseFunc("__eq")
      .registerOp();
  MetadataBuilder("torch.ops.aten.ne.Scalar")
      .elementwiseFunc("__ne")
      .registerOp();
  MetadataBuilder("torch.ops.aten.lt.Scalar")
      .elementwiseFunc("__lt")
      .registerOp();
  MetadataBuilder("torch.ops.aten.le.Scalar")
      .elementwiseFunc("__le")
      .registerOp();
  MetadataBuilder("torch.ops.aten.gt.Scalar")
      .elementwiseFunc("__gt")
      .registerOp();
  MetadataBuilder("torch.ops.aten.ge.Scalar")
      .elementwiseFunc("__ge")
      .registerOp();

  // Bitwise.
  MetadataBuilder("torch.ops.aten.bitwise_and.Tensor")
      .elementwise()
      .registerOp();
  MetadataBuilder("torch.ops.aten.bitwise_or.Tensor")
      .elementwise()
      .registerOp();
  MetadataBuilder("torch.ops.aten.bitwise_xor.Tensor")
      .elementwise()
      .registerOp();
  MetadataBuilder("torch.ops.aten.__and__.Tensor")
      .elementwiseFunc("__bitwise_and")
      .registerOp();
  MetadataBuilder("torch.ops.aten.__or__.Tensor")
      .elementwiseFunc("__bitwise_or")
      .registerOp();

  // Bitwise (Tensor, Scalar).
  MetadataBuilder("torch.ops.aten.bitwise_and.Scalar")
      .elementwiseFunc("__bitwise_and")
      .registerOp();
  MetadataBuilder("torch.ops.aten.bitwise_or.Scalar")
      .elementwiseFunc("__bitwise_or")
      .registerOp();
  MetadataBuilder("torch.ops.aten.bitwise_xor.Scalar")
      .elementwiseFunc("__bitwise_xor")
      .registerOp();
  MetadataBuilder("torch.ops.aten.bitwise_not.default")
      .elementwise()
      .registerOp();

  // Logical.
  MetadataBuilder("torch.ops.aten.logical_and.default")
      .elementwise()
      .registerOp();
  MetadataBuilder("torch.ops.aten.logical_or.default")
      .elementwise()
      .registerOp();
  MetadataBuilder("torch.ops.aten.logical_xor.default")
      .elementwise()
      .registerOp();
  MetadataBuilder("torch.ops.aten.logical_not.default")
      .elementwise()
      .registerOp();

  // Unary math.
  MetadataBuilder("torch.ops.aten.abs.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.neg.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.ceil.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.floor.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.round.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.trunc.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.sign.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.sqrt.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.rsqrt.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.reciprocal.default")
      .elementwise()
      .registerOp();
  MetadataBuilder("torch.ops.aten.exp.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.log.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.log2.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.log10.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.log1p.default").elementwise().registerOp();

  // Trigonometric.
  MetadataBuilder("torch.ops.aten.sin.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.cos.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.tan.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.asin.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.acos.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.atan.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.atan2.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.sinh.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.cosh.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.tanh.default").elementwise().registerOp();

  // Item.
  MetadataBuilder("torch.ops.aten.item.default")
      .elementwiseFunc("__item")
      .inPlaceIfLastUse(false)
      .numArgs(1)
      .argumentMeta({{.isRegister = false, .wholeTensor = true}})
      .returnMeta({ArgumentMeta{.isRegister = true}})
      .typeTemplateParams({0})
      .registerOp();

  // Where.
  MetadataBuilder("torch.ops.aten.where.self")
      .elementwiseFunc("__where")
      .registerOp();

  // Activation functions.
  MetadataBuilder("torch.ops.aten.relu.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.sigmoid.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.clamp.default")
      .elementwise()
      .argumentMeta(
          {{.isRegister = true},
           {.hasPresentTemplateParam = true},
           {.hasPresentTemplateParam = true}})
      .registerOp();
  MetadataBuilder("torch.ops.aten.nan_to_num.default")
      .elementwise()
      .normalize(resolveNanToNumDefaults)
      .registerOp();

  // Type cast.
  MetadataBuilder("torch.ops.aten.to.dtype")
      .elementwiseFunc("__to")
      .generateCall([](std::stringstream& ss,
                       NodeCP node,
                       std::vector<std::string> args) {
        TORCH_CHECK(!args.empty(), "to.dtype requires at least one input");
        const auto* dtypeAttr = node->tryGetAttribute("dtype");
        TORCH_CHECK(dtypeAttr, "to.dtype missing dtype attribute");
        ss << "static_cast<" << cudaTypeFromDtype(*dtypeAttr) << ">(" << args[0]
           << ")";
      })
      .ignoreAttrs({"memory_format", "copy", "non_blocking"})
      .registerOp();

  // Zeros.
  MetadataBuilder("torch.ops.aten.zeros.default")
      .elementwiseFunc("__zero")
      .numArgs(0)
      .returnMeta({{.isRegister = true, .reserveShape = sizeAttrReserveShape}})
      .normalize(resolveDefaultDtype)
      .outputConstraints(sizeAttrRankConstraint)
      .hasDtypeTemplateParam()
      .shapeAttr("size")
      .registerOp();

  // Ones.
  MetadataBuilder("torch.ops.aten.ones.default")
      .elementwiseFunc("__one")
      .numArgs(0)
      .returnMeta({{.isRegister = true, .reserveShape = sizeAttrReserveShape}})
      .normalize(resolveDefaultDtype)
      .outputConstraints(sizeAttrRankConstraint)
      .hasDtypeTemplateParam()
      .shapeAttr("size")
      .registerOp();

  // Zeros_like.
  MetadataBuilder("torch.ops.aten.zeros_like.default")
      .elementwiseFunc("__zero")
      .numArgs(0)
      .defaultInputMeta()
      .returnMeta(
          {{.isRegister = true,
            .reserveShape =
                [](NodeCP node,
                   nativert::ExecutionFrame& frame,
                   const FormalToActual& map) -> std::vector<std::vector<Dim>> {
              return elementwiseInputShape(node, frame, map, 0);
            }}})
      .normalize(resolveDtypeFromInputExact)
      .rankArgument(0)
      .hasDtypeTemplateParam()
      .registerOp();

  // Ones_like.
  MetadataBuilder("torch.ops.aten.ones_like.default")
      .elementwiseFunc("__one")
      .numArgs(0)
      .defaultInputMeta()
      .returnMeta(
          {{.isRegister = true,
            .reserveShape =
                [](NodeCP node,
                   nativert::ExecutionFrame& frame,
                   const FormalToActual& map) -> std::vector<std::vector<Dim>> {
              return elementwiseInputShape(node, frame, map, 0);
            }}})
      .normalize(resolveDtypeFromInputExact)
      .rankArgument(0)
      .hasDtypeTemplateParam()
      .registerOp();

  // New_zeros -> zeros replacement.
  MetadataBuilder("torch.ops.aten.new_zeros.default")
      .sizeOrdinal({0})
      .maybeReplace(
          [](NodeCP node,
             ValueTypes& /*types*/,
             WaveGraph& waveGraph) -> std::vector<std::pair<ValueCP, ValueCP>> {
            const auto* sizeAttr = node->tryGetAttribute("size");
            if (!sizeAttr) {
              return {};
            }
            auto [outDtype, dtypeStr] = resolveOutDtype(node, &waveGraph);
            auto* graph = waveGraph.graph();
            auto* zerosNode =
                graph->createNode("torch.ops.aten.zeros.default", {});
            zerosNode->addAttribute(
                {sizeAttr->name,
                 std::get<std::vector<int64_t>>(sizeAttr->value)});
            zerosNode->addAttribute({"dtype", dtypeStr});
            graph->insertBefore(zerosNode, const_cast<nativert::Node*>(node));
            auto* newOutput =
                waveGraph.newTensorValue(zerosNode, "zeros", outDtype);
            return {{node->outputs()[0], newOutput}};
          })
      .registerOp();

  // Arange.
  MetadataBuilder("torch.ops.aten.arange.default")
      .elementwiseFunc("__arange")
      .hasIdxArg()
      .returnMeta(
          {{.isRegister = true,
            .reserveShape =
                [](NodeCP node,
                   nativert::ExecutionFrame& frame,
                   const FormalToActual& map) -> std::vector<std::vector<Dim>> {
              auto end = integerParamByName(node, "end", frame, map);
              return {{static_cast<Dim>(end)}};
            }}})
      .normalize(resolveArangeDtype)
      .outputConstraints(rank1Constraint)
      .hasDtypeTemplateParam()
      .shapeAttr("end")
      .registerOp();

  MetadataBuilder("torch.ops.aten.arange.start")
      .elementwiseFunc("__arange_start")
      .hasIdxArg()
      .returnMeta(
          {{.isRegister = true,
            .reserveShape =
                [](NodeCP node,
                   nativert::ExecutionFrame& frame,
                   const FormalToActual& map) -> std::vector<std::vector<Dim>> {
              auto start = integerParamByName(node, "start", frame, map);
              auto end = integerParamByName(node, "end", frame, map);
              return {{static_cast<Dim>(end - start)}};
            }}})
      .normalize(resolveArangeDtype)
      .outputConstraints(rank1Constraint)
      .hasDtypeTemplateParam()
      .shapeAttr("end")
      .registerOp();

  // Min/max.
  MetadataBuilder("torch.ops.aten.minimum.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.maximum.default").elementwise().registerOp();

  // Shape query.
  MetadataBuilder("torch.ops.aten.sym_size.int")
      .elementwiseFunc("__sym_size")
      .argumentMeta(
          {{.isRegister = false, .wholeTensor = true}, {.isRegister = true}})
      .metadataGetter()
      .registerOp();

  // Identity-like ops: output replaces first input.
  for (const auto* opName :
       {"torch.ops.aten.detach_.default",
        "torch.ops.aten.lift_fresh_copy.default"}) {
    MetadataBuilder builder(opName);
    builder.sizeOrdinal({0}).rankArgument(0).maybeReplace(
        [](NodeCP node,
           ValueTypes& /*types*/,
           WaveGraph&) -> std::vector<std::pair<ValueCP, ValueCP>> {
          const auto& inputs = node->inputs();
          const auto& outputs = node->outputs();
          if (inputs.empty() || outputs.empty()) {
            return {};
          }
          return {{outputs[0], inputs[0].value}};
        });
    if (std::string_view(opName) == "torch.ops.aten.detach_.default") {
      builder.viewOfArg(0);
    }
    builder.registerOp();
  }

  // Clone: eliminate unless a user mutates the output in place.
  MetadataBuilder("torch.ops.aten.clone.default")
      .sizeOrdinal({0})
      .rankArgument(0)
      .isStandalone()
      .maybeReplace(
          [](NodeCP node,
             ValueTypes& /*types*/,
             WaveGraph&) -> std::vector<std::pair<ValueCP, ValueCP>> {
            const auto& inputs = node->inputs();
            const auto& outputs = node->outputs();
            if (inputs.empty() || outputs.empty()) {
              return {};
            }
            for (auto* user : outputs[0]->users()) {
              if (isInPlaceMutation(user, outputs[0])) {
                return {};
              }
            }
            return {{outputs[0], inputs[0].value}};
          })
      .registerOp();

  // Transpose.
  registerRankPreservingStandalone("torch.ops.aten.transpose.int", 0);

  // Contiguous.
  registerRankPreservingStandalone("torch.ops.aten.contiguous.default");

  // Slice.
  MetadataBuilder("torch.ops.aten.slice.Tensor")
      .sizeOrdinal({0})
      .isStandalone()
      .viewOfArg(0)
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            const auto& inputs = node->inputs();
            if (inputs.empty()) {
              return {};
            }
            ValueConstraint constraint;
            constraint.rank = types.rank(inputs[0].value);
            return {constraint};
          })
      .registerOp();

  // Select.
  MetadataBuilder("torch.ops.aten.select.int")
      .sizeOrdinal({0})
      .viewOfArg(0)
      .headerFile("velox/experimental/torchwave/Views.cuh")
      .deviceFunc("tw_select")
      .typeTemplateParams({0})
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            const auto& inputs = node->inputs();
            if (inputs.empty()) {
              return {};
            }
            ValueConstraint constraint;
            constraint.rank =
                static_cast<int8_t>(types.rank(inputs[0].value) - 1);
            return {constraint};
          })
      .registerOp();

  // Narrow.
  MetadataBuilder("torch.ops.aten.narrow.default")
      .sizeOrdinal({0})
      .isStandalone()
      .viewOfArg(0)
      .rankArgument(0)
      .registerOp();

  // Unsqueeze.
  MetadataBuilder("torch.ops.aten.unsqueeze.default")
      .sizeOrdinal({0})
      .isStandalone()
      .viewOfArg(0)
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            const auto& inputs = node->inputs();
            if (inputs.empty()) {
              return {};
            }
            ValueConstraint constraint;
            constraint.rank =
                static_cast<int8_t>(types.rank(inputs[0].value) + 1);
            return {constraint};
          })
      .registerOp();

  // Flatten.
  MetadataBuilder("torch.ops.aten.flatten.using_ints")
      .sizeOrdinal({0})
      .isStandalone()
      .viewOfArg(0)
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            const auto& inputs = node->inputs();
            if (inputs.empty()) {
              return {};
            }
            auto inputRank = types.rank(inputs[0].value);
            auto getInt = [&](const char* name, int64_t defaultVal) -> int64_t {
              const auto* attr = node->tryGetAttribute(name);
              if (!attr) {
                return defaultVal;
              }
              return std::get<int64_t>(attr->value);
            };
            auto startDim = getInt("start_dim", 0);
            auto endDim = getInt("end_dim", -1);
            if (startDim < 0) {
              startDim += inputRank;
            }
            if (endDim < 0) {
              endDim += inputRank;
            }
            ValueConstraint constraint;
            constraint.rank =
                static_cast<int8_t>(inputRank - (endDim - startDim));
            return {constraint};
          })
      .registerOp();

  // Unbind.
  MetadataBuilder("torch.ops.aten.unbind.int")
      .sizeOrdinal({0})
      .isStandalone()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            const auto& inputs = node->inputs();
            if (inputs.empty()) {
              return {};
            }
            int8_t outputRank =
                static_cast<int8_t>(types.rank(inputs[0].value) - 1);
            std::vector<ValueConstraint> constraints;
            constraints.reserve(node->outputs().size());
            for (size_t i = 0; i < node->outputs().size(); ++i) {
              constraints.push_back({outputRank});
            }
            return constraints;
          })
      .registerOp();

  // Concat.
  MetadataBuilder("torch.ops.aten.concat.default")
      .sizeOrdinal({0})
      .isStandalone()
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
            return {ValueConstraint{.rank = types.rank(elements[0])}};
          })
      .registerOp();

  // Tensor split.
  MetadataBuilder("torch.ops.aten.tensor_split.tensor_indices_or_sections")
      .sizeOrdinal({0})
      .isStandalone()
      .argumentMeta({{}, {.cpuOnly = true}, {}})
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            const auto& inputs = node->inputs();
            if (inputs.empty()) {
              return {};
            }
            auto rank = types.rank(inputs[0].value);
            std::vector<ValueConstraint> constraints;
            constraints.reserve(node->outputs().size());
            for (size_t i = 0; i < node->outputs().size(); ++i) {
              constraints.push_back({rank});
            }
            return constraints;
          })
      .registerOp();

  // Reshape.
  registerReshapeLikeOp("torch.ops.aten.reshape.default", "shape");

  // View.
  registerReshapeLikeOp("torch.ops.aten.view.default", "size");

  static const std::string kScanHeader =
      "velox/experimental/torchwave/Scan.cuh";

  // Sum reduction.
  MetadataBuilder("torch.ops.aten.sum.default")
      .sizeOrdinal({0})
      .sizeShortcut(SizeShortcut::kMax)
      .hasBarrier()
      .singleBlockIfFused()
      .argumentMeta({{.isRegister = true}, {}})
      .returnMeta(
          {{.isRegister = false,
            .reserveShape = [](NodeCP /*node*/,
                               nativert::ExecutionFrame& /*frame*/,
                               const FormalToActual& /*map*/)
                -> std::vector<std::vector<Dim>> { return {{}}; }}})
      .normalize(resolveDtypeFromInput)
      .makeMultiKernelVariant(makeSumVariant)
      .cgVariant(makeSumCgVariant)
      .headerFile(kScanHeader)
      .deviceFunc("tw_sum")
      .sharedDecls({{"Int32X32", "warpSums"}})
      .dynamicSharedDecls({{Metadata::kTypeFromDtype, "counter"}})
      .typeTemplateParams({0})
      .hasDtypeTemplateParam()
      .hasBlockSizeTemplateParam()
      .alwaysSingleBlock()
      .registerOp();

  // --- Multi-kernel sum intrinsics ---

  // tw.sum_head: (Tensor) -> Tensor
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.sum_head",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .hasBarrier()
      .returnMeta({{.isRegister = false, .reserveShape = numBlocksShape}})
      .headerFile(kScanHeader)
      .deviceFunc("tw_sum_head")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .typeTemplateParams({0})
      .hasDtypeTemplateParam()
      .hasBlockSizeTemplateParam()
      .kernelBreakForMultiblock()
      .outputConstraints(rank1Constraint)
      .registerOp();

  // tw.sum_final: (Tensor) -> Tensor
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.sum_final",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .hasBarrier()
      .returnMeta(
          {{.isRegister = false,
            .reserveShape = [](NodeCP /*node*/,
                               nativert::ExecutionFrame& /*frame*/,
                               const FormalToActual& /*map*/)
                -> std::vector<std::vector<Dim>> { return {{}}; }}})
      .inputFromPreviousKernel(0)
      .headerFile(kScanHeader)
      .deviceFunc("tw_sum_tensor")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .dynamicSharedDecls({{0, "counter"}})
      .typeTemplateParams({0})
      .hasDtypeTemplateParam()
      .hasBlockSizeTemplateParam()
      .singleBlockIfFused()
      .outputConstraints(rank1Constraint)
      .registerOp();

  // Stream compaction.
  MetadataBuilder("torch.ops.aten.masked_select.default")
      .sizeOrdinal({0})
      .hasBarrier()
      .singleBlockIfFused()
      .argumentMeta({{.isRegister = true}, {.isRegister = true}})
      .returnMeta(
          {{.isRegister = false,
            .reserveShape =
                [](NodeCP node,
                   nativert::ExecutionFrame& frame,
                   const FormalToActual& map) -> std::vector<std::vector<Dim>> {
              return elementwiseInputShape(node, frame, map, 0);
            },
            .shapeSetOnDevice = true}})
      .makeMultiKernelVariant(makeMaskedSelectVariant)
      .cgVariant(makeMaskedSelectCgVariant)
      .headerFile(kScanHeader)
      .deviceFunc("masked_select")
      .sharedDecls({{"Int32X32", "warpSums"}, {"uint32_t", "counter"}})
      .typeTemplateParams({0})
      .hasBlockSizeTemplateParam()
      .alwaysSingleBlock()
      .registerOp();

  // --- Torchwave intrinsics for multi-kernel masked_select ---

  // tw.masked_select_head: (Tensor, Tensor) -> Tensor
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.masked_select_head",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get()),
              c10::Argument("mask", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .hasBarrier()
      .returnMeta({{.isRegister = false, .reserveShape = numBlocksShape}})
      .headerFile(kScanHeader)
      .deviceFunc("masked_select_head")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .typeTemplateParams({0})
      .hasBlockSizeTemplateParam()
      .kernelBreakForMultiblock()
      .outputConstraints(rank1Constraint)
      .registerOp();

  // tw.add_sizes: (Tensor) -> int64
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.add_sizes",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::IntType::get())}))
      .sizeOrdinal({0})
      .hasBarrier()
      .returnMeta({{.neededOnHost = true}})
      .headerFile(kScanHeader)
      .deviceFunc("add_sizes")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"},
           {"uint32_t", "counter"}})
      .typeTemplateParams({})
      .hasBlockSizeTemplateParam()
      .inputFromPreviousKernel(0)
      .kernelBreakForMultiblock()
      .alwaysSingleBlock()
      .registerOp();

  // tw.masked_select_final: (Tensor, Tensor, Tensor, int) -> Tensor
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.masked_select_final",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get()),
              c10::Argument("mask", c10::TensorType::get()),
              c10::Argument("counts", c10::TensorType::get()),
              c10::Argument("total", c10::IntType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .returnMeta(
          {{.isRegister = false,
            .reserveShape =
                [](NodeCP node,
                   nativert::ExecutionFrame& frame,
                   const FormalToActual& map) -> std::vector<std::vector<Dim>> {
              auto total = paramSymInt(node->inputs()[3].value, frame, map);
              return {{static_cast<Dim>(total)}};
            }}})
      .inputFromPreviousKernel(3)
      .headerFile(kScanHeader)
      .deviceFunc("masked_select_final")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .typeTemplateParams({0})
      .hasBarrier()
      .hasBlockSizeTemplateParam()
      .outputConstraints(rank1Constraint)
      .registerOp();

  // --- Cumulative sum ---

  MetadataBuilder("torch.ops.aten.cumsum.default")
      .sizeOrdinal({0})
      .sizeShortcut(SizeShortcut::kMax)
      .hasBarrier()
      .singleBlockIfFused()
      .returnMeta(
          {{.isRegister = false,
            .reserveShape =
                [](NodeCP node,
                   nativert::ExecutionFrame& frame,
                   const FormalToActual& map) -> std::vector<std::vector<Dim>> {
              return elementwiseInputShape(node, frame, map, 0);
            }}})
      .normalize(resolveDtypeFromInput)
      .makeMultiKernelVariant(makeCumsumVariant)
      .cgVariant(makeCumsumCgVariant)
      .headerFile(kScanHeader)
      .deviceFunc("cumsum")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .dynamicSharedDecls({{Metadata::kTypeFromDtype, "counter"}})
      .typeTemplateParams({0})
      .hasDtypeTemplateParam()
      .hasBlockSizeTemplateParam()
      .alwaysSingleBlock()
      .only1d()
      .templateAttrs({"dim"})
      .registerOp();

  // tw.cumsum_head: (Tensor) -> Tensor
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.cumsum_head",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .hasBarrier()
      .returnMeta({{.isRegister = false, .reserveShape = numBlocksShape}})
      .headerFile(kScanHeader)
      .deviceFunc("cumsum_head")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .typeTemplateParams({0})
      .hasDtypeTemplateParam()
      .hasBlockSizeTemplateParam()
      .kernelBreakForMultiblock()
      .only1d()
      .outputConstraints(rank1Constraint)
      .registerOp();

  // tw.cumsum_add_sizes: (Tensor) -> int (link-only output)
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.cumsum_add_sizes",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::IntType::get())}))
      .sizeOrdinal({0})
      .hasBarrier()
      .returnMeta({{.linkOnly = true}})
      .headerFile(kScanHeader)
      .deviceFunc("add_sizes")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .dynamicSharedDecls({{0, "counter"}})
      .hasDtypeTemplateParam()
      .hasBlockSizeTemplateParam()
      .inputFromPreviousKernel(0)
      .kernelBreakForMultiblock()
      .alwaysSingleBlock()
      .only1d()
      .registerOp();

  // tw.cumsum_final: (Tensor, Tensor, int) -> Tensor
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.cumsum_final",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get()),
              c10::Argument("counts", c10::TensorType::get()),
              c10::Argument("link", c10::IntType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .argumentMeta(
          {{.isRegister = false}, {.isRegister = false}, {.linkOnly = true}})
      .returnMeta({{.isRegister = false, .reserveShape = inputShape}})
      .inputFromPreviousKernel(2)
      .headerFile(kScanHeader)
      .deviceFunc("cumsum_final")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .typeTemplateParams({0})
      .hasBarrier()
      .hasDtypeTemplateParam()
      .hasBlockSizeTemplateParam()
      .only1d()
      .templateAttrs({"dim"})
      .outputConstraints(rank1Constraint)
      .registerOp();

  // --- Exclusive sum ---

  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "torch.ops.aten.exclusive_sum",
          "default",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .sizeShortcut(SizeShortcut::kMax)
      .hasBarrier()
      .singleBlockIfFused()
      .returnMeta({{.isRegister = false, .reserveShape = inputShapePlusOne}})
      .normalize(resolveDtypeFromInput)
      .makeMultiKernelVariant(makeExclusiveSumVariant)
      .cgVariant(makeExclusiveSumCgVariant)
      .headerFile(kScanHeader)
      .deviceFunc("exclusive_sum")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .dynamicSharedDecls({{Metadata::kTypeFromDtype, "counter"}})
      .typeTemplateParams({0})
      .hasDtypeTemplateParam()
      .hasBlockSizeTemplateParam()
      .alwaysSingleBlock()
      .only1d()
      .templateAttrs({"dim"})
      .outputConstraints(rank1Constraint)
      .registerOp();

  // tw.exclusive_sum_head: (Tensor) -> Tensor
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.exclusive_sum_head",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .hasBarrier()
      .returnMeta({{.isRegister = false, .reserveShape = numBlocksShape}})
      .headerFile(kScanHeader)
      .deviceFunc("exclusive_sum_head")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .typeTemplateParams({0})
      .hasDtypeTemplateParam()
      .hasBlockSizeTemplateParam()
      .kernelBreakForMultiblock()
      .only1d()
      .outputConstraints(rank1Constraint)
      .registerOp();

  // tw.exclusive_sum_add_sizes: (Tensor) -> int (link-only output)
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.exclusive_sum_add_sizes",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::IntType::get())}))
      .sizeOrdinal({0})
      .hasBarrier()
      .returnMeta({{.linkOnly = true}})
      .headerFile(kScanHeader)
      .deviceFunc("add_sizes")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .dynamicSharedDecls({{0, "counter"}})
      .hasDtypeTemplateParam()
      .hasBlockSizeTemplateParam()
      .inputFromPreviousKernel(0)
      .kernelBreakForMultiblock()
      .alwaysSingleBlock()
      .only1d()
      .registerOp();

  // tw.exclusive_sum_final: (Tensor, Tensor, int) -> Tensor
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.exclusive_sum_final",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get()),
              c10::Argument("counts", c10::TensorType::get()),
              c10::Argument("link", c10::IntType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .argumentMeta(
          {{.isRegister = false}, {.isRegister = false}, {.linkOnly = true}})
      .returnMeta({{.isRegister = false, .reserveShape = inputShapePlusOne}})
      .inputFromPreviousKernel(2)
      .headerFile(kScanHeader)
      .deviceFunc("exclusive_sum_final")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .typeTemplateParams({0})
      .hasBarrier()
      .hasDtypeTemplateParam()
      .hasBlockSizeTemplateParam()
      .only1d()
      .outputConstraints(rank1Constraint)
      .registerOp();

  // --- Cooperative grid variants ---

  // tw.sum_cg: (Tensor) -> (Tensor, Tensor[partials])
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.sum_cg",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get()),
              c10::Argument("partials", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .sizeShortcut(SizeShortcut::kMax)
      .hasBarrier()
      .returnMeta(
          {{.isRegister = false,
            .reserveShape = [](NodeCP /*node*/,
                               nativert::ExecutionFrame& /*frame*/,
                               const FormalToActual& /*map*/)
                -> std::vector<std::vector<Dim>> { return {{}}; }},
           {.isRegister = false, .reserveShape = numBlocksShape}})
      .normalize(resolveDtypeFromInput)
      .headerFile(kScanHeader)
      .deviceFunc("tw_sum_cg")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .dynamicSharedDecls({{Metadata::kTypeFromDtype, "counter"}})
      .typeTemplateParams({0})
      .hasDtypeTemplateParam()
      .hasBlockSizeTemplateParam()
      .numBarriers(2)
      .registerOp();

  // tw.cumsum_cg: (Tensor) -> (Tensor, Tensor[counts])
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.cumsum_cg",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get()),
              c10::Argument("counts", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .sizeShortcut(SizeShortcut::kMax)
      .hasBarrier()
      .returnMeta(
          {{.isRegister = false, .reserveShape = inputShape},
           {.isRegister = false, .reserveShape = numBlocksShape}})
      .normalize(resolveDtypeFromInput)
      .headerFile(kScanHeader)
      .deviceFunc("cumsum_cg")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .dynamicSharedDecls({{Metadata::kTypeFromDtype, "counter"}})
      .typeTemplateParams({0})
      .hasDtypeTemplateParam()
      .hasBlockSizeTemplateParam()
      .numBarriers(2)
      .only1d()
      .templateAttrs({"dim"})
      .outputConstraints(rank1Constraint)
      .registerOp();

  // tw.exclusive_sum_cg: (Tensor) -> (Tensor, Tensor[counts])
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.exclusive_sum_cg",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get()),
              c10::Argument("counts", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .sizeShortcut(SizeShortcut::kMax)
      .hasBarrier()
      .returnMeta(
          {{.isRegister = false, .reserveShape = inputShapePlusOne},
           {.isRegister = false, .reserveShape = numBlocksShape}})
      .normalize(resolveDtypeFromInput)
      .headerFile(kScanHeader)
      .deviceFunc("exclusive_sum_cg")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .dynamicSharedDecls({{Metadata::kTypeFromDtype, "counter"}})
      .typeTemplateParams({0})
      .hasDtypeTemplateParam()
      .hasBlockSizeTemplateParam()
      .numBarriers(2)
      .only1d()
      .outputConstraints(rank1Constraint)
      .registerOp();

  // tw.masked_select_cg: (Tensor, Tensor) -> (Tensor, Tensor[counts])
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.masked_select_cg",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get()),
              c10::Argument("mask", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get()),
              c10::Argument("counts", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .hasBarrier()
      .returnMeta(
          {{.isRegister = false,
            .reserveShape = inputShape,
            .shapeSetOnDevice = true},
           {.isRegister = false, .reserveShape = numBlocksShape}})
      .headerFile(kScanHeader)
      .deviceFunc("masked_select_cg")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"},
           {"uint32_t", "counter"}})
      .typeTemplateParams({0})
      .hasBlockSizeTemplateParam()
      .numBarriers(3)
      .outputConstraints(rank1Constraint)
      .registerOp();

  // Cat — registered in Cat.cpp.
  registerCatMetadata();

  // --- repeat_interleave ---

  MetadataBuilder("torch.ops.aten.repeat_interleave.self_Tensor")
      .sizeOrdinal({1})
      .outputConstraints(rank1Constraint)
      .maybeReplace(
          [](NodeCP node,
             ValueTypes& types,
             WaveGraph& waveGraph) -> std::vector<std::pair<ValueCP, ValueCP>> {
            auto* graph = waveGraph.graph();
            auto inputId = node->inputs()[0].value->id();
            auto inputDtype = types.types[inputId]->dtype();
            auto* headNode = graph->createNode(
                "tw.repeat_interleave_head",
                {{"repeats", node->inputs()[1].value}});
            auto* prefixOutput = waveGraph.newTensorValue(
                headNode, "repeat_prefix", c10::ScalarType::Int);
            auto* totalOutput = waveGraph.newScalarValue(
                headNode, "repeat_total", c10::ScalarType::Int);
            auto* finalNode = graph->createNode(
                "tw.repeat_interleave_final",
                {{"input", node->inputs()[0].value},
                 {"prefix", prefixOutput},
                 {"total", totalOutput}});
            auto* resultOutput = waveGraph.newTensorValue(
                finalNode, "repeat_result", inputDtype);
            return {{node->outputs()[0], resultOutput}};
          })
      .registerOp();

  // tw.repeat_interleave_head: (Tensor) -> (Tensor, int)
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.repeat_interleave_head",
          "",
          std::vector<c10::Argument>{
              c10::Argument("repeats", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("prefix", c10::TensorType::get()),
              c10::Argument("total", c10::IntType::get())}))
      .sizeOrdinal({0})
      .hasBarrier()
      .returnMeta(
          {{.isRegister = false, .reserveShape = inputShape},
           {.neededOnHost = true}})
      .headerFile(kScanHeader)
      .deviceFunc("repeat_interleave_head")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"},
           {"uint32_t", "counter"}})
      .typeTemplateParams({0})
      .hasBlockSizeTemplateParam()
      .kernelBreakForMultiblock()
      .singleBlockIfFused()
      .outputConstraints(
          [](NodeCP /*node*/,
             const ValueTypes& /*types*/) -> std::vector<ValueConstraint> {
            return {{.rank = 1}, {}};
          })
      .registerOp();

  auto repeatInterleaveFinalReserve =
      [](NodeCP node,
         nativert::ExecutionFrame& frame,
         const FormalToActual& map) -> std::vector<std::vector<Dim>> {
    auto total = paramSymInt(node->inputs()[2].value, frame, map);
    return {{static_cast<Dim>(total)}};
  };

  auto makeRepeatInterleaveFinalVariant = [](NodeCP single,
                                             WaveGraph* waveGraph) {
    auto* graph = variantNodeGraph(waveGraph);
    auto* cgNode = graph->createNode(
        "tw.repeat_interleave_final_cg",
        {{"input", single->inputs()[0].value},
         {"prefix", single->inputs()[1].value},
         {"total", single->inputs()[2].value}});
    copyOriginalOutputs(cgNode, single, waveGraph);
    return cgNode;
  };

  // tw.repeat_interleave_final: (Tensor, Tensor, int) -> Tensor
  // Intrinsically single-block: uses prefix sums from the head stage.
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.repeat_interleave_final",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get()),
              c10::Argument("prefix", c10::TensorType::get()),
              c10::Argument("total", c10::IntType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .returnMeta(
          {{.isRegister = false, .reserveShape = repeatInterleaveFinalReserve}})
      .inputFromPreviousKernel(2)
      .headerFile(kScanHeader)
      .deviceFunc("repeat_interleave_final")
      .sharedDecls({{"uint32_t", "size"}, {"uint32_t", "rounded"}})
      .typeTemplateParams({0})
      .hasBarrier()
      .hasBlockSizeTemplateParam()
      .alwaysSingleBlock()
      .singleBlockIfFused()
      .makeMultiKernelVariant(makeRepeatInterleaveFinalVariant)
      .cgVariant(makeRepeatInterleaveFinalVariant)
      .outputConstraints(rank1Constraint)
      .registerOp();

  // tw.repeat_interleave_final_cg: multi-block variant with op barrier.
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.repeat_interleave_final_cg",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get()),
              c10::Argument("prefix", c10::TensorType::get()),
              c10::Argument("total", c10::IntType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .returnMeta(
          {{.isRegister = false, .reserveShape = repeatInterleaveFinalReserve}})
      .inputFromPreviousKernel(2)
      .headerFile(kScanHeader)
      .deviceFunc("repeat_interleave_final_cg")
      .sharedDecls({{"uint32_t", "size"}, {"uint32_t", "rounded"}})
      .typeTemplateParams({0})
      .hasBarrier()
      .hasBlockSizeTemplateParam()
      .numBarriers(1)
      .outputConstraints(rank1Constraint)
      .registerOp();

  static const std::string kHashHeader =
      "velox/experimental/torchwave/Hash.cuh";

  // isin.
  MetadataBuilder("torch.ops.aten.isin.Tensor_Tensor")
      .sizeOrdinal({0})
      .outputConstraints(rank1Constraint)
      .ignoreAttrs({"assume_unique"})
      .maybeReplace(
          [](NodeCP node,
             ValueTypes& types,
             WaveGraph& waveGraph) -> std::vector<std::pair<ValueCP, ValueCP>> {
            auto* graph = waveGraph.graph();
            auto inputId = node->inputs()[0].value->id();
            auto inputDtype = types.types[inputId]->dtype();

            auto* headNode = graph->createNode(
                "tw.isin_head", {{"test_elements", node->inputs()[1].value}});
            auto* headOutput =
                waveGraph.newTensorValue(headNode, "isin_table", inputDtype);

            std::vector<nativert::NamedArgument> finalInputs = {
                {"elements", node->inputs()[0].value}, {"table", headOutput}};
            const auto* invertInput = node->tryGetInput("invert");
            if (invertInput) {
              finalInputs.push_back({"invert", invertInput->value});
            }
            auto* finalNode =
                graph->createNode("tw.isin_final", std::move(finalInputs));
            if (!invertInput) {
              const auto* invertAttr = node->tryGetAttribute("invert");
              bool invertVal = false;
              if (invertAttr) {
                if (std::holds_alternative<bool>(invertAttr->value)) {
                  invertVal = std::get<bool>(invertAttr->value);
                } else if (std::holds_alternative<int64_t>(invertAttr->value)) {
                  invertVal = std::get<int64_t>(invertAttr->value) != 0;
                }
              }
              finalNode->addAttribute({"invert", invertVal});
            }
            auto* resultOutput = waveGraph.newTensorValue(
                finalNode, "isin_result", c10::ScalarType::Bool);
            return {{node->outputs()[0], resultOutput}};
          })
      .registerOp();

  // tw.isin_head: (Tensor) -> Tensor
  // Builds the hash table in shared memory; runs one block wide when fused.
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.isin_head",
          "",
          std::vector<c10::Argument>{
              c10::Argument("test_elements", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .singleBlockIfFused()
      .returnMeta(
          {{.isRegister = false,
            .reserveShape =
                [](NodeCP node,
                   nativert::ExecutionFrame& frame,
                   const FormalToActual& map) -> std::vector<std::vector<Dim>> {
              auto tensor = paramTensor(node->inputs()[0].value, frame, map);
              auto n = tensor.numel();
              int64_t tableSize = 1;
              while (tableSize < n * 2) {
                tableSize *= 2;
              }
              return {{static_cast<Dim>(tableSize + 1)}};
            }}})
      .headerFile(kHashHeader)
      .deviceFunc("tw_isin_head")
      .typeTemplateParams({0})
      .outputConstraints(rank1Constraint)
      .registerOp();

  // tw.isin_final: (Tensor, Tensor, bool) -> Tensor(bool)
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.isin_final",
          "",
          std::vector<c10::Argument>{
              c10::Argument("elements", c10::TensorType::get()),
              c10::Argument("table", c10::TensorType::get()),
              c10::Argument("invert", c10::BoolType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .elementwiseFunc("__isin_final")
      .hasIdxArg()
      .hasSizeArg()
      .argumentMeta(
          {{.isRegister = true},
           {.isRegister = false, .wholeTensor = true},
           {.isRegister = true}})
      .headerFile(kHashHeader)
      .typeTemplateParams({0})
      .outputConstraints(rank1Constraint)
      .registerOp();

  MetadataBuilder("torch.ops.aten.pad.default")
      .isStandalone()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            const auto& inputs = node->inputs();
            if (inputs.empty()) {
              return {};
            }
            return {{types.rank(inputs[0].value)}};
          })
      .maybeReplace(
          [](NodeCP node,
             ValueTypes&,
             WaveGraph&) -> std::vector<std::pair<ValueCP, ValueCP>> {
            const auto* attr = node->tryGetAttribute("value");
            if (!attr) {
              const_cast<nativert::Node*>(node)->addAttribute(
                  {"value", static_cast<int64_t>(0)});
            } else if (std::holds_alternative<nativert::None>(attr->value)) {
              const_cast<nativert::Attribute*>(attr)->value =
                  static_cast<int64_t>(0);
            }
            return {};
          })
      .registerOp();
}

} // namespace torch::wave
