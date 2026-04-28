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
#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/Utils.h"

#include <cmath>

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
  auto* graph = waveGraph->graph();
  auto [outDtype, dtypeStr] = resolveOutDtype(single, waveGraph);

  // Node 1: tw.sum_head. Per-block reduction. Output is one TOut value per
  // block.
  auto* headNode =
      graph->createNode("tw.sum_head", {{"input", single->inputs()[0].value}});
  headNode->addAttribute({"dtype", dtypeStr});
  auto* headOutput =
      waveGraph->newTensorValue(headNode, "sum_blocks", outDtype);

  // Node 2: tw.sum_final. Single-block reduction of per-block sums to a
  // scalar.
  auto* finalNode = graph->createNode("tw.sum_final", {{"input", headOutput}});
  finalNode->addAttribute({"dtype", dtypeStr});
  return finalNode;
}

nativert::Node* makeCumsumVariant(NodeCP single, WaveGraph* waveGraph) {
  auto* graph = waveGraph->graph();
  auto [outDtype, dtypeStr] = resolveOutDtype(single, waveGraph);

  // Node 1: tw.cumsum_head. Input is the original input. Output is per-block
  // sums tensor with size = ceil(inputNumel / 256).
  auto* headNode = graph->createNode(
      "tw.cumsum_head", {{"input", single->inputs()[0].value}});
  headNode->addAttribute({"dtype", dtypeStr});
  auto* headOutput =
      waveGraph->newTensorValue(headNode, "cumsum_counts", outDtype);

  // Node 2: tw.cumsum_add_sizes. Input is the head output. Output is a
  // link-only scalar (not passed at runtime, only for ordering).
  auto* addSizesNode =
      graph->createNode("tw.cumsum_add_sizes", {{"input", headOutput}});
  addSizesNode->addAttribute({"dtype", dtypeStr});
  auto* addSizesOutput = waveGraph->newScalarValue(
      addSizesNode, "cumsum_link", c10::ScalarType::Int);

  // Node 3: tw.cumsum_final. Inputs are the original input, the head output
  // (prefix-summed counts), and the link from add_sizes.
  std::vector<nativert::NamedArgument> finalInputs = {
      {"input", single->inputs()[0].value},
      {"counts", headOutput},
      {"link", addSizesOutput}};
  auto* finalNode =
      graph->createNode("tw.cumsum_final", std::move(finalInputs));
  finalNode->addAttribute({"dtype", dtypeStr});
  // Copy the dim attribute (int64_t) from the original node.
  const auto* dimAttr = single->tryGetAttribute("dim");
  if (dimAttr) {
    finalNode->addAttribute({dimAttr->name, std::get<int64_t>(dimAttr->value)});
  }
  return finalNode;
}

nativert::Node* makeExclusiveSumVariant(NodeCP single, WaveGraph* waveGraph) {
  auto* graph = waveGraph->graph();
  auto inputId = single->inputs()[0].value->id();
  auto inputDtype = waveGraph->types().types[inputId]->dtype();
  auto outDtype = c10::isIntegralType(inputDtype, true) ? c10::ScalarType::Long
                                                        : inputDtype;
  auto dtypeStr = c10::toString(outDtype);

  auto* headNode = graph->createNode(
      "tw.exclusive_sum_head", {{"input", single->inputs()[0].value}});
  headNode->addAttribute({"dtype", dtypeStr});
  auto* headOutput =
      waveGraph->newTensorValue(headNode, "exsum_counts", outDtype);

  auto* addSizesNode =
      graph->createNode("tw.exclusive_sum_add_sizes", {{"input", headOutput}});
  addSizesNode->addAttribute({"dtype", dtypeStr});
  auto* addSizesOutput = waveGraph->newScalarValue(
      addSizesNode, "exsum_link", c10::ScalarType::Int);

  auto* finalNode = graph->createNode(
      "tw.exclusive_sum_final",
      {{"input", single->inputs()[0].value},
       {"counts", headOutput},
       {"link", addSizesOutput}});
  finalNode->addAttribute({"dtype", dtypeStr});
  return finalNode;
}

nativert::Node* makeMaskedSelectVariant(NodeCP single, WaveGraph* waveGraph) {
  auto* graph = waveGraph->graph();

  // Node 1: tw.masked_select_head. Same inputs as original. Output is a
  // per-block counts tensor of int with size = ceil(firstInputNumel / 256).
  std::vector<nativert::NamedArgument> headInputs(
      single->inputs().begin(), single->inputs().end());
  auto* headNode =
      graph->createNode("tw.masked_select_head", std::move(headInputs));
  auto* headOutput = waveGraph->newTensorValue(
      headNode, "masked_select_counts", c10::ScalarType::Int);

  // Node 2: tw.add_sizes. Input is the head output. Output is a scalar
  // int64 giving the total selected count.
  auto* addSizesNode =
      graph->createNode("tw.add_sizes", {{"input", headOutput}});
  auto* addSizesOutput = waveGraph->newScalarValue(
      addSizesNode, "masked_select_total", c10::ScalarType::Int);

  // Node 3: tw.masked_select_final. Inputs are the original first input,
  // the head output (per-block counts), and the total from add_sizes.
  // Its outputs must match the original node's outputs.
  auto* finalNode = graph->createNode(
      "tw.masked_select_final",
      {{"input", single->inputs()[0].value},
       {"mask", single->inputs()[1].value},
       {"counts", headOutput},
       {"total", addSizesOutput}});
  return finalNode;
}

namespace {

void resolveDtypeFromInput(nativert::Node* node, const ValueTypes& types) {
  if (node->inputs().empty()) {
    return;
  }
  if (node->tryGetAttribute("dtype")) {
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
int64_t scalarParamByName(
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
  if (node->tryGetAttribute("dtype")) {
    return;
  }
  node->addAttribute({"dtype", std::string("Float")});
}

void resolveArangeDtype(nativert::Node* node, const ValueTypes& /*types*/) {
  if (node->tryGetAttribute("dtype")) {
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

void resolveDtypeFromInputExact(nativert::Node* node, const ValueTypes& types) {
  if (node->inputs().empty()) {
    return;
  }
  if (node->tryGetAttribute("dtype")) {
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
    FormalToActual map) {
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
  auto size = ivalue.toIntVector();
  return {{size.begin(), size.end()}};
}

std::vector<std::vector<Dim>> numBlocksShape(
    NodeCP node,
    nativert::ExecutionFrame& frame,
    FormalToActual map) {
  auto tensor = paramTensor(node->inputs()[0].value, frame, map);
  auto numBlocks = static_cast<Dim>((tensor.numel() + 255) / 256);
  return {{numBlocks}};
}

std::vector<std::vector<Dim>>
inputShape(NodeCP node, nativert::ExecutionFrame& frame, FormalToActual map) {
  auto tensor = paramTensor(node->inputs()[0].value, frame, map);
  auto sizes = tensor.sizes();
  return {{sizes.begin(), sizes.end()}};
}

std::vector<std::vector<Dim>> inputShapePlusOne(
    NodeCP node,
    nativert::ExecutionFrame& frame,
    FormalToActual map) {
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
             nativert::Graph*) -> std::vector<std::pair<ValueCP, ValueCP>> {
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
          [](NodeCP node,
             ValueTypes& types,
             nativert::Graph*) -> std::vector<std::pair<ValueCP, ValueCP>> {
            const auto& inputs = node->inputs();
            const auto& outputs = node->outputs();
            if (inputs.empty() || outputs.empty()) {
              return {};
            }
            if (types.rank(inputs[0].value) == 1 &&
                types.rank(outputs[0]) == 1) {
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
      .attributeArgs({"alpha"})
      .registerOp();
  MetadataBuilder("torch.ops.aten.sub.Tensor")
      .elementwise()
      .attributeArgs({"alpha"})
      .registerOp();
  MetadataBuilder("torch.ops.aten.mul.Tensor").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.div.Tensor").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.remainder.Tensor").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.fmod.Tensor").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.pow.Tensor_Tensor")
      .elementwise()
      .registerOp();

  // Binary arithmetic (Tensor, Scalar).
  MetadataBuilder("torch.ops.aten.add.Scalar")
      .elementwiseFunc("add")
      .attributeArgs({"other", "alpha"})
      .registerOp();
  MetadataBuilder("torch.ops.aten.sub.Scalar")
      .elementwiseFunc("sub")
      .attributeArgs({"other", "alpha"})
      .registerOp();
  MetadataBuilder("torch.ops.aten.mul.Scalar")
      .elementwiseFunc("mul")
      .attributeArgs({"other"})
      .registerOp();
  MetadataBuilder("torch.ops.aten.div.Scalar")
      .elementwiseFunc("div")
      .attributeArgs({"other"})
      .registerOp();
  MetadataBuilder("torch.ops.aten.remainder.Scalar")
      .elementwiseFunc("remainder")
      .attributeArgs({"other"})
      .registerOp();
  MetadataBuilder("torch.ops.aten.fmod.Scalar")
      .elementwiseFunc("fmod")
      .attributeArgs({"other"})
      .registerOp();
  MetadataBuilder("torch.ops.aten.pow.Tensor_Scalar")
      .elementwiseFunc("pow")
      .attributeArgs({"other"})
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
      .elementwiseFunc("eq")
      .attributeArgs({"other"})
      .registerOp();
  MetadataBuilder("torch.ops.aten.ne.Scalar")
      .elementwiseFunc("ne")
      .attributeArgs({"other"})
      .registerOp();
  MetadataBuilder("torch.ops.aten.lt.Scalar")
      .elementwiseFunc("lt")
      .attributeArgs({"other"})
      .registerOp();
  MetadataBuilder("torch.ops.aten.le.Scalar")
      .elementwiseFunc("le")
      .attributeArgs({"other"})
      .registerOp();
  MetadataBuilder("torch.ops.aten.gt.Scalar")
      .elementwiseFunc("gt")
      .attributeArgs({"other"})
      .registerOp();
  MetadataBuilder("torch.ops.aten.ge.Scalar")
      .elementwiseFunc("ge")
      .attributeArgs({"other"})
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
      .elementwiseFunc("bitwise_and")
      .registerOp();
  MetadataBuilder("torch.ops.aten.__or__.Tensor")
      .elementwiseFunc("bitwise_or")
      .registerOp();

  // Bitwise (Tensor, Scalar).
  MetadataBuilder("torch.ops.aten.bitwise_and.Scalar")
      .elementwiseFunc("bitwise_and")
      .attributeArgs({"other"})
      .registerOp();
  MetadataBuilder("torch.ops.aten.bitwise_or.Scalar")
      .elementwiseFunc("bitwise_or")
      .attributeArgs({"other"})
      .registerOp();
  MetadataBuilder("torch.ops.aten.bitwise_xor.Scalar")
      .elementwiseFunc("bitwise_xor")
      .attributeArgs({"other"})
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
      .elementwiseFunc("item")
      .inPlaceIfLastUse(false)
      .numArgs(1)
      .argumentMeta({{.isRegister = false, .wholeTensor = true}})
      .returnMeta({ArgumentMeta{.isRegister = true}})
      .typeTemplateParams({0})
      .registerOp();

  // Where.
  MetadataBuilder("torch.ops.aten.where.self")
      .elementwiseFunc("where")
      .registerOp();

  // Activation functions.
  MetadataBuilder("torch.ops.aten.relu.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.sigmoid.default").elementwise().registerOp();
  MetadataBuilder("torch.ops.aten.clamp.default")
      .elementwise()
      .attributeArgs({"min", "max"})
      .registerOp();

  // Type cast.
  MetadataBuilder("torch.ops.aten.to.dtype")
      .elementwiseFunc("to")
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
      .elementwiseFunc("zero")
      .numArgs(0)
      .returnMeta({{.isRegister = true, .reserveShape = sizeAttrReserveShape}})
      .normalize(resolveDefaultDtype)
      .outputConstraints(sizeAttrRankConstraint)
      .hasDtypeTemplateParam()
      .shapeAttr("size")
      .registerOp();

  // Ones.
  MetadataBuilder("torch.ops.aten.ones.default")
      .elementwiseFunc("one")
      .numArgs(0)
      .returnMeta({{.isRegister = true, .reserveShape = sizeAttrReserveShape}})
      .normalize(resolveDefaultDtype)
      .outputConstraints(sizeAttrRankConstraint)
      .hasDtypeTemplateParam()
      .shapeAttr("size")
      .registerOp();

  // Zeros_like.
  MetadataBuilder("torch.ops.aten.zeros_like.default")
      .elementwiseFunc("zero")
      .numArgs(0)
      .defaultInputMeta()
      .returnMeta(
          {{.isRegister = true,
            .reserveShape =
                [](NodeCP node,
                   nativert::ExecutionFrame& frame,
                   FormalToActual map) -> std::vector<std::vector<Dim>> {
              return elementwiseInputShape(node, frame, map, 0);
            }}})
      .normalize(resolveDtypeFromInputExact)
      .rankArgument(0)
      .hasDtypeTemplateParam()
      .registerOp();

  // Ones_like.
  MetadataBuilder("torch.ops.aten.ones_like.default")
      .elementwiseFunc("one")
      .numArgs(0)
      .defaultInputMeta()
      .returnMeta(
          {{.isRegister = true,
            .reserveShape =
                [](NodeCP node,
                   nativert::ExecutionFrame& frame,
                   FormalToActual map) -> std::vector<std::vector<Dim>> {
              return elementwiseInputShape(node, frame, map, 0);
            }}})
      .normalize(resolveDtypeFromInputExact)
      .rankArgument(0)
      .hasDtypeTemplateParam()
      .registerOp();

  // Arange.
  MetadataBuilder("torch.ops.aten.arange.default")
      .elementwiseFunc("arange")
      .hasIdxArg()
      .returnMeta(
          {{.isRegister = true,
            .reserveShape =
                [](NodeCP node,
                   nativert::ExecutionFrame& frame,
                   FormalToActual map) -> std::vector<std::vector<Dim>> {
              auto end = scalarParamByName(node, "end", frame, map);
              return {{static_cast<Dim>(end)}};
            }}})
      .normalize(resolveArangeDtype)
      .outputConstraints(rank1Constraint)
      .hasDtypeTemplateParam()
      .shapeAttr("end")
      .registerOp();

  MetadataBuilder("torch.ops.aten.arange.start")
      .elementwiseFunc("arange_start")
      .attributeArgs({"start"})
      .hasIdxArg()
      .returnMeta(
          {{.isRegister = true,
            .reserveShape =
                [](NodeCP node,
                   nativert::ExecutionFrame& frame,
                   FormalToActual map) -> std::vector<std::vector<Dim>> {
              auto start = scalarParamByName(node, "start", frame, map);
              auto end = scalarParamByName(node, "end", frame, map);
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
      .elementwiseFunc("sym_size")
      .attributeArgs({"dim"})
      .registerOp();

  // Identity-like ops: output replaces first input.
  for (const auto* opName :
       {"torch.ops.aten.detach_.default",
        "torch.ops.aten.clone.default",
        "torch.ops.aten.lift_fresh_copy.default"}) {
    MetadataBuilder builder(opName);
    builder.sizeOrdinal({0}).rankArgument(0).maybeReplace(
        [](NodeCP node,
           ValueTypes& /*types*/,
           nativert::Graph*) -> std::vector<std::pair<ValueCP, ValueCP>> {
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
      .maybeReplace(
          [](NodeCP node,
             ValueTypes& types,
             nativert::Graph*) -> std::vector<std::pair<ValueCP, ValueCP>> {
            const auto& inputs = node->inputs();
            const auto& outputs = node->outputs();
            if (inputs.empty() || outputs.empty()) {
              return {};
            }
            auto getInt = [&](const char* name, int64_t defaultVal) -> int64_t {
              const auto* attr = node->tryGetAttribute(name);
              if (!attr) {
                return defaultVal;
              }
              return std::get<int64_t>(attr->value);
            };
            auto dim = getInt("dim", 0);
            auto start = getInt("start", 0);
            auto end = getInt("end", std::numeric_limits<int64_t>::max());
            auto step = getInt("step", 1);
            if (dim == 0 && start == 0 &&
                (end == -1 || end == std::numeric_limits<int64_t>::max()) &&
                step == 1) {
              return {{outputs[0], inputs[0].value}};
            }
            return {};
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
                               FormalToActual /*map*/)
                -> std::vector<std::vector<Dim>> { return {{}}; }}})
      .normalize(resolveDtypeFromInput)
      .makeMultiKernelVariant(makeSumVariant)
      .headerFile(kScanHeader)
      .deviceFunc("tw_sum")
      .sharedDecls({{"Int32X32", "warpSums"}})
      .dynamicSharedDecls({{-1, "counter"}})
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
                               FormalToActual /*map*/)
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
      .alwaysSingleBlock()
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
                   FormalToActual map) -> std::vector<std::vector<Dim>> {
              return elementwiseInputShape(node, frame, map, 0);
            },
            .shapeSetOnDevice = true}})
      .makeMultiKernelVariant(makeMaskedSelectVariant)
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
            .reserveShape = inputShape,
            .shapeSetOnDevice = true}})
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
                   FormalToActual map) -> std::vector<std::vector<Dim>> {
              return elementwiseInputShape(node, frame, map, 0);
            }}})
      .normalize(resolveDtypeFromInput)
      .makeMultiKernelVariant(makeCumsumVariant)
      .headerFile(kScanHeader)
      .deviceFunc("cumsum")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .dynamicSharedDecls({{-1, "counter"}})
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
           {"uint32_t", "rounded"},
           {"uint32_t", "counter"}})
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
      .makeMultiKernelVariant(makeExclusiveSumVariant)
      .headerFile(kScanHeader)
      .deviceFunc("exclusive_sum")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .dynamicSharedDecls({{-1, "counter"}})
      .typeTemplateParams({0})
      .hasDtypeTemplateParam()
      .hasBlockSizeTemplateParam()
      .alwaysSingleBlock()
      .only1d()
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
           {"uint32_t", "rounded"},
           {"uint32_t", "counter"}})
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

  // Cat.
  MetadataBuilder("torch.ops.aten.cat.default")
      .sizeShortcut(SizeShortcut::kSum)
      .sizeOrdinal({0})
      .sizeArgsList({true})
      .only1d()
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
            return {{.rank = types.rank(elements[0])}};
          })
      .registerOp();

  // --- repeat_interleave ---

  MetadataBuilder("torch.ops.aten.repeat_interleave.self_Tensor")
      .sizeOrdinal({1})
      .maybeReplace(
          [](NodeCP node, ValueTypes& /*types*/, nativert::Graph* graph)
              -> std::vector<std::pair<ValueCP, ValueCP>> {
            auto* headNode = graph->createNode(
                "tw.repeat_interleave_head",
                {{"repeats", node->inputs()[1].value}});
            auto* prefixOutput = headNode->addOutput(
                "repeat_prefix", nativert::Type(nativert::Type::Kind::Tensor));
            auto* totalOutput = headNode->addOutput(
                "repeat_total", nativert::Type(nativert::Type::Kind::SymInt));
            auto* finalNode = graph->createNode(
                "tw.repeat_interleave_final",
                {{"input", node->inputs()[0].value},
                 {"prefix", prefixOutput},
                 {"total", totalOutput}});
            auto* resultOutput = finalNode->addOutput(
                "repeat_result", nativert::Type(nativert::Type::Kind::Tensor));
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
      .deviceFunc("add_sizes")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"},
           {"uint32_t", "counter"}})
      .hasBlockSizeTemplateParam()
      .kernelBreakForMultiblock()
      .alwaysSingleBlock()
      .outputConstraints(
          [](NodeCP /*node*/,
             const ValueTypes& /*types*/) -> std::vector<ValueConstraint> {
            return {{.rank = 1}, {}};
          })
      .registerOp();

  // tw.repeat_interleave_final: (Tensor, Tensor, int) -> Tensor
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
          {{.isRegister = false,
            .reserveShape =
                [](NodeCP node,
                   nativert::ExecutionFrame& frame,
                   FormalToActual map) -> std::vector<std::vector<Dim>> {
              auto total = paramSymInt(node->inputs()[2].value, frame, map);
              return {{static_cast<Dim>(total)}};
            }}})
      .inputFromPreviousKernel(2)
      .headerFile(kScanHeader)
      .deviceFunc("repeat_interleave_final")
      .sharedDecls({{"uint32_t", "size"}, {"uint32_t", "rounded"}})
      .typeTemplateParams({0})
      .hasBarrier()
      .hasBlockSizeTemplateParam()
      .outputConstraints(rank1Constraint)
      .registerOp();
}

} // namespace torch::wave
