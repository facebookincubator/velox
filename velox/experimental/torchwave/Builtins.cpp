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

namespace torch::wave {

nativert::Node* makeRepeatInterleaveVariant(
    NodeCP single,
    WaveGraph* waveGraph) {
  auto* graph = waveGraph->graph();

  // Node 1: tw.repeat_interleave_head. Takes the repeats tensor (second input).
  // Output 1: prefix-summed repeats (int tensor, same size as repeats).
  // Output 2: scalar total count (needed on host for allocation).
  auto* headNode = graph->createNode(
      "tw.repeat_interleave_head",
      {{"repeats", single->inputs()[1].value}});
  auto* prefixOutput = waveGraph->newTensorValue(
      headNode, "repeat_prefix", c10::ScalarType::Int);
  auto* totalOutput = waveGraph->newScalarValue(
      headNode, "repeat_total", c10::ScalarType::Int);

  // Node 2: tw.repeat_interleave_final. Takes self and prefix sums.
  // The total is passed as an input for ordering and size allocation.
  auto* finalNode = graph->createNode(
      "tw.repeat_interleave_final",
      {{"input", single->inputs()[0].value},
       {"prefix", prefixOutput},
       {"total", totalOutput}});
  return finalNode;
}

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
  auto* headNode = graph->createNode(
      "tw.sum_head", {{"input", single->inputs()[0].value}});
  headNode->addAttribute({"dtype", dtypeStr});
  auto* headOutput =
      waveGraph->newTensorValue(headNode, "sum_blocks", outDtype);

  // Node 2: tw.sum_final. Single-block reduction of per-block sums to a
  // scalar.
  auto* finalNode =
      graph->createNode("tw.sum_final", {{"input", headOutput}});
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
  auto* headOutput = waveGraph->newTensorValue(
      headNode, "cumsum_counts", outDtype);

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
    finalNode->addAttribute(
        {dimAttr->name, std::get<int64_t>(dimAttr->value)});
  }
  return finalNode;
}

nativert::Node* makeExclusiveSumVariant(NodeCP single, WaveGraph* waveGraph) {
  auto* graph = waveGraph->graph();
  auto inputId = single->inputs()[0].value->id();
  auto inputDtype = waveGraph->types().types[inputId]->dtype();
  auto outDtype = c10::isIntegralType(inputDtype, true)
      ? c10::ScalarType::Long
      : inputDtype;
  auto dtypeStr = c10::toString(outDtype);

  auto* headNode = graph->createNode(
      "tw.exclusive_sum_head", {{"input", single->inputs()[0].value}});
  headNode->addAttribute({"dtype", dtypeStr});
  auto* headOutput = waveGraph->newTensorValue(
      headNode, "exsum_counts", outDtype);

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

std::vector<ValueConstraint> rank1Constraint(
    NodeCP /*node*/,
    const ValueTypes& /*types*/) {
  return {{.rank = 1}};
}

std::vector<std::vector<Dim>> numBlocksShape(
    NodeCP node,
    nativert::ExecutionFrame& frame,
    FormalToActual map) {
  auto tensor = paramTensor(node->inputs()[0].value, frame, map);
  auto numBlocks = static_cast<Dim>((tensor.numel() + 255) / 256);
  return {{numBlocks}};
}

std::vector<std::vector<Dim>> inputShape(
    NodeCP node,
    nativert::ExecutionFrame& frame,
    FormalToActual map) {
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
    std::optional<int32_t> viewOfArg = std::nullopt) {
  const auto* schema = findFunctionSchema(opName);
  TORCH_CHECK(schema, "FunctionSchema not found for: ", opName);
  Metadata md;
  md.functionSchema = schema;
  md.defaultInputMeta();
  md.defaultOutputMeta();
  md.sizeArgs.ordinal = {0};
  md.isStandalone_ = true;
  md.viewOfArg = viewOfArg;
  md.outputConstraints = [](NodeCP node, const ValueTypes& types)
      -> std::vector<ValueConstraint> {
    const auto& inputs = node->inputs();
    if (inputs.empty()) {
      return {};
    }
    return {{.rank = types.rank(inputs[0].value)}};
  };
  md.maybeReplace = [](NodeCP node, ValueTypes& types)
      -> std::vector<std::pair<ValueCP, ValueCP>> {
    const auto& inputs = node->inputs();
    const auto& outputs = node->outputs();
    if (inputs.empty() || outputs.empty()) {
      return {};
    }
    if (types.rank(inputs[0].value) == 1) {
      return {{outputs[0], inputs[0].value}};
    }
    return {};
  };
  Registry::registerMetadata(opName, std::move(md));
}

void registerReshapeLikeOp(const char* opName, const char* shapeAttrName) {
  const auto* schema = findFunctionSchema(opName);
  TORCH_CHECK(schema, "FunctionSchema not found for: ", opName);
  Metadata md;
  md.functionSchema = schema;
  md.defaultInputMeta();
  md.defaultOutputMeta();
  md.sizeArgs.ordinal = {0};
  md.isStandalone_ = true;
  md.viewOfArg = 0;
  md.outputConstraints = [shapeAttrName](
                              NodeCP node, const ValueTypes& /*types*/)
      -> std::vector<ValueConstraint> {
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
  };
  md.maybeReplace = [](NodeCP node, ValueTypes& types)
      -> std::vector<std::pair<ValueCP, ValueCP>> {
    const auto& inputs = node->inputs();
    const auto& outputs = node->outputs();
    if (inputs.empty() || outputs.empty()) {
      return {};
    }
    if (types.rank(inputs[0].value) == 1 && types.rank(outputs[0]) == 1) {
      return {{outputs[0], inputs[0].value}};
    }
    return {};
  };
  Registry::registerMetadata(opName, std::move(md));
}

} // namespace

void registerBuiltins() {
  // Binary arithmetic.
  Registry::registerElementwise("torch.ops.aten.add.Tensor", {"alpha"});
  Registry::registerElementwise("torch.ops.aten.sub.Tensor", {"alpha"});
  Registry::registerElementwise("torch.ops.aten.mul.Tensor");
  Registry::registerElementwise("torch.ops.aten.div.Tensor");
  Registry::registerElementwise("torch.ops.aten.remainder.Tensor");
  Registry::registerElementwise("torch.ops.aten.fmod.Tensor");
  Registry::registerElementwise("torch.ops.aten.pow.Tensor_Tensor");

  // Binary arithmetic (Tensor, Scalar).
  Registry::registerElementwiseOp(
      "torch.ops.aten.add.Scalar", "add", false, {"other", "alpha"});
  Registry::registerElementwiseOp(
      "torch.ops.aten.sub.Scalar", "sub", false, {"other", "alpha"});
  Registry::registerElementwiseOp(
      "torch.ops.aten.mul.Scalar", "mul", false, {"other"});
  Registry::registerElementwiseOp(
      "torch.ops.aten.div.Scalar", "div", false, {"other"});
  Registry::registerElementwiseOp(
      "torch.ops.aten.remainder.Scalar", "remainder", false, {"other"});
  Registry::registerElementwiseOp(
      "torch.ops.aten.fmod.Scalar", "fmod", false, {"other"});
  Registry::registerElementwiseOp(
      "torch.ops.aten.pow.Tensor_Scalar", "pow", false, {"other"});

  // Comparison.
  Registry::registerElementwise("torch.ops.aten.eq.Tensor");
  Registry::registerElementwise("torch.ops.aten.ne.Tensor");
  Registry::registerElementwise("torch.ops.aten.lt.Tensor");
  Registry::registerElementwise("torch.ops.aten.le.Tensor");
  Registry::registerElementwise("torch.ops.aten.gt.Tensor");
  Registry::registerElementwise("torch.ops.aten.ge.Tensor");

  // Comparison (Tensor, Scalar).
  Registry::registerElementwiseOp(
      "torch.ops.aten.eq.Scalar", "eq", false, {"other"});
  Registry::registerElementwiseOp(
      "torch.ops.aten.ne.Scalar", "ne", false, {"other"});
  Registry::registerElementwiseOp(
      "torch.ops.aten.lt.Scalar", "lt", false, {"other"});
  Registry::registerElementwiseOp(
      "torch.ops.aten.le.Scalar", "le", false, {"other"});
  Registry::registerElementwiseOp(
      "torch.ops.aten.gt.Scalar", "gt", false, {"other"});
  Registry::registerElementwiseOp(
      "torch.ops.aten.ge.Scalar", "ge", false, {"other"});

  // Bitwise.
  Registry::registerElementwise("torch.ops.aten.bitwise_and.Tensor");
  Registry::registerElementwise("torch.ops.aten.bitwise_or.Tensor");
  Registry::registerElementwise("torch.ops.aten.bitwise_xor.Tensor");
  Registry::registerElementwiseOp(
      "torch.ops.aten.__and__.Tensor", "bitwise_and", false);
  Registry::registerElementwiseOp(
      "torch.ops.aten.__or__.Tensor", "bitwise_or", false);

  // Bitwise (Tensor, Scalar).
  Registry::registerElementwiseOp(
      "torch.ops.aten.bitwise_and.Scalar", "bitwise_and", false, {"other"});
  Registry::registerElementwiseOp(
      "torch.ops.aten.bitwise_or.Scalar", "bitwise_or", false, {"other"});
  Registry::registerElementwiseOp(
      "torch.ops.aten.bitwise_xor.Scalar", "bitwise_xor", false, {"other"});
  Registry::registerElementwise("torch.ops.aten.bitwise_not.default");

  // Logical.
  Registry::registerElementwise("torch.ops.aten.logical_and.default");
  Registry::registerElementwise("torch.ops.aten.logical_or.default");
  Registry::registerElementwise("torch.ops.aten.logical_xor.default");
  Registry::registerElementwise("torch.ops.aten.logical_not.default");

  // Unary math.
  Registry::registerElementwise("torch.ops.aten.abs.default");
  Registry::registerElementwise("torch.ops.aten.neg.default");
  Registry::registerElementwise("torch.ops.aten.ceil.default");
  Registry::registerElementwise("torch.ops.aten.floor.default");
  Registry::registerElementwise("torch.ops.aten.round.default");
  Registry::registerElementwise("torch.ops.aten.trunc.default");
  Registry::registerElementwise("torch.ops.aten.sign.default");
  Registry::registerElementwise("torch.ops.aten.sqrt.default");
  Registry::registerElementwise("torch.ops.aten.rsqrt.default");
  Registry::registerElementwise("torch.ops.aten.reciprocal.default");
  Registry::registerElementwise("torch.ops.aten.exp.default");
  Registry::registerElementwise("torch.ops.aten.log.default");
  Registry::registerElementwise("torch.ops.aten.log2.default");
  Registry::registerElementwise("torch.ops.aten.log10.default");
  Registry::registerElementwise("torch.ops.aten.log1p.default");

  // Trigonometric.
  Registry::registerElementwise("torch.ops.aten.sin.default");
  Registry::registerElementwise("torch.ops.aten.cos.default");
  Registry::registerElementwise("torch.ops.aten.tan.default");
  Registry::registerElementwise("torch.ops.aten.asin.default");
  Registry::registerElementwise("torch.ops.aten.acos.default");
  Registry::registerElementwise("torch.ops.aten.atan.default");
  Registry::registerElementwise("torch.ops.aten.atan2.default");
  Registry::registerElementwise("torch.ops.aten.sinh.default");
  Registry::registerElementwise("torch.ops.aten.cosh.default");
  Registry::registerElementwise("torch.ops.aten.tanh.default");

  // Item.
  {
    const auto* schema = findFunctionSchema("torch.ops.aten.item.default");
    TORCH_CHECK(schema, "FunctionSchema not found for item.default");
    Metadata md;
    md.functionSchema = schema;
    md.sizeArgs.ordinal = {0};
    md.argumentMeta = {
        {.isRegister = false, .wholeTensor = true},
    };
    md.returnMeta = {ArgumentMeta{.isRegister = true}};
    md.elementwise = std::make_unique<ElementwiseOp>();
    md.elementwise->functionName = "--item";
    md.elementwise->numArgs = 1;
    md.typeTemplateParams = {0};
    Registry::registerMetadata("torch.ops.aten.item.default", std::move(md));
  }

  // Where.
  Registry::registerElementwiseOp(
      "torch.ops.aten.where.self", "where", false, {});

  // Activation functions.
  Registry::registerElementwise("torch.ops.aten.relu.default");
  Registry::registerElementwise("torch.ops.aten.sigmoid.default");
  Registry::registerElementwise("torch.ops.aten.clamp.default", {"min", "max"});

  // Type cast.
  {
    const auto* schema = findFunctionSchema("torch.ops.aten.to.dtype");
    TORCH_CHECK(schema, "FunctionSchema not found for to.dtype");
    Metadata md;
    md.functionSchema = schema;
    md.sizeArgs.ordinal = {0};
    md.inPlaceIfLastUse = true;
    md.argumentMeta.resize(
        schema->arguments().size(), ArgumentMeta{.isRegister = true});
    md.returnMeta = {ArgumentMeta{.isRegister = true}};
    md.elementwise = std::make_unique<ElementwiseOp>();
    md.elementwise->functionName = "--to";
    md.generateCall = [](std::stringstream& ss,
                         NodeCP node,
                         std::vector<std::string> args) {
      TORCH_CHECK(!args.empty(), "to.dtype requires at least one input");
      const auto* dtypeAttr = node->tryGetAttribute("dtype");
      TORCH_CHECK(dtypeAttr, "to.dtype missing dtype attribute");
      ss << "static_cast<" << cudaTypeFromDtype(*dtypeAttr) << ">("
         << args[0] << ")";
    };
    Registry::registerMetadata("torch.ops.aten.to.dtype", std::move(md));
  }

  // Zeros.
  {
    const auto* schema = findFunctionSchema("torch.ops.aten.zeros.default");
    TORCH_CHECK(schema, "FunctionSchema not found for zeros.default");
    Metadata md;
    md.functionSchema = schema;
    md.sizeArgs.ordinal = {0};
    md.inPlaceIfLastUse = true;
    md.argumentMeta.resize(
        schema->arguments().size(), ArgumentMeta{.isRegister = true});
    md.returnMeta = {
        {.isRegister = false,
         .reserveShape =
             [](NodeCP node,
                nativert::ExecutionFrame& /*frame*/,
                FormalToActual /*map*/) -> std::vector<std::vector<Dim>> {
           const auto* sizeAttr = node->tryGetAttribute("size");
           TORCH_CHECK(sizeAttr, "zeros.default missing size attribute");
           const auto& size = std::get<std::vector<int64_t>>(sizeAttr->value);
           std::vector<Dim> shape(size.begin(), size.end());
           return {shape};
         }}};
    md.elementwise = std::make_unique<ElementwiseOp>();
    md.elementwise->functionName = "--zero";
    md.elementwise->numArgs = 0;
    Registry::registerMetadata("torch.ops.aten.zeros.default", std::move(md));
  }

  // Arange.
  {
    const auto* schema = findFunctionSchema("torch.ops.aten.arange.default");
    TORCH_CHECK(schema, "FunctionSchema not found for arange.default");
    Metadata md;
    md.functionSchema = schema;
    md.sizeArgs.ordinal = {0};
    md.inPlaceIfLastUse = true;
    md.argumentMeta.resize(
        schema->arguments().size(), ArgumentMeta{.isRegister = true});
    md.returnMeta = {ArgumentMeta{.isRegister = true}};
    md.elementwise = std::make_unique<ElementwiseOp>();
    md.elementwise->functionName = "--arange";
    md.elementwise->hasIdxArg = true;
    Registry::registerMetadata("torch.ops.aten.arange.default", std::move(md));
  }
  {
    const auto* schema = findFunctionSchema("torch.ops.aten.arange.start");
    TORCH_CHECK(schema, "FunctionSchema not found for arange.start");
    Metadata md;
    md.functionSchema = schema;
    md.sizeArgs.ordinal = {0};
    md.inPlaceIfLastUse = true;
    md.argumentMeta.resize(
        schema->arguments().size(), ArgumentMeta{.isRegister = true});
    md.returnMeta = {ArgumentMeta{.isRegister = true}};
    md.elementwise = std::make_unique<ElementwiseOp>();
    md.elementwise->functionName = "--arange_start";
    md.elementwise->attributeArgs = {"start"};
    md.elementwise->hasIdxArg = true;
    Registry::registerMetadata("torch.ops.aten.arange.start", std::move(md));
  }

  // Min/max.
  Registry::registerElementwise("torch.ops.aten.minimum.default");
  Registry::registerElementwise("torch.ops.aten.maximum.default");

  // Shape query.
  Registry::registerElementwiseOp(
      "torch.ops.aten.sym_size.int", "sym_size", false, {"dim"});

  // Identity-like ops: output replaces first input.
  for (const auto* opName : {
           "torch.ops.aten.detach_.default",
           "torch.ops.aten.clone.default",
           "torch.ops.aten.lift_fresh_copy.default"}) {
    const auto* schema = findFunctionSchema(opName);
    TORCH_CHECK(schema, "FunctionSchema not found for: ", opName);
    Metadata md;
    md.functionSchema = schema;
    md.defaultInputMeta();
    md.defaultOutputMeta();
    md.sizeArgs.ordinal = {0};
    md.rankArgument = 0;
    if (std::string_view(opName) == "torch.ops.aten.detach_.default") {
      md.viewOfArg = 0;
    }
    md.maybeReplace = [](NodeCP node, ValueTypes& /*types*/)
        -> std::vector<std::pair<ValueCP, ValueCP>> {
      const auto& inputs = node->inputs();
      const auto& outputs = node->outputs();
      if (inputs.empty() || outputs.empty()) {
        return {};
      }
      return {{outputs[0], inputs[0].value}};
    };
    Registry::registerMetadata(opName, std::move(md));
  }

  // Transpose.
  registerRankPreservingStandalone("torch.ops.aten.transpose.int", 0);

  // Contiguous.
  registerRankPreservingStandalone("torch.ops.aten.contiguous.default");

  // Slice.
  {
    const auto* schema =
        findFunctionSchema("torch.ops.aten.slice.Tensor");
    TORCH_CHECK(schema, "FunctionSchema not found for slice.Tensor");
    Metadata md;
    md.functionSchema = schema;
    md.defaultInputMeta();
    md.defaultOutputMeta();
    md.sizeArgs.ordinal = {0};
    md.isStandalone_ = true;
    md.viewOfArg = 0;
    md.outputConstraints = [](NodeCP node,
                              const ValueTypes& types)
        -> std::vector<ValueConstraint> {
      const auto& inputs = node->inputs();
      if (inputs.empty()) {
        return {};
      }
      ValueConstraint constraint;
      constraint.rank = types.rank(inputs[0].value);
      return {constraint};
    };
    md.maybeReplace = [](NodeCP node,
                         ValueTypes& types)
        -> std::vector<std::pair<ValueCP, ValueCP>> {
      const auto& inputs = node->inputs();
      const auto& outputs = node->outputs();
      if (inputs.empty() || outputs.empty()) {
        return {};
      }
      auto getInt = [&](const char* name,
                        int64_t defaultVal) -> int64_t {
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
    };
    Registry::registerMetadata(
        "torch.ops.aten.slice.Tensor", std::move(md));
  }

  // Reshape.
  registerReshapeLikeOp("torch.ops.aten.reshape.default", "shape");

  // View.
  registerReshapeLikeOp("torch.ops.aten.view.default", "size");

  static const std::string kScanHeader =
      "velox/experimental/torchwave/Scan.cuh";

  // Sum reduction.
  {
    const auto* schema = findFunctionSchema("torch.ops.aten.sum.default");
    TORCH_CHECK(schema, "FunctionSchema not found for sum.default");
    Metadata md;
    md.functionSchema = schema;
    md.sizeArgs.ordinal = {0};
    md.sizeShortcut = SizeShortcut::kMax;
    md.hasBarrier = true;
    md.singleBlockIfFused = true;
    md.defaultInputMeta();
    md.argumentMeta[0].isRegister = true;
    md.returnMeta = {
        {.isRegister = false,
         .reserveShape =
             [](NodeCP /*node*/,
                nativert::ExecutionFrame& /*frame*/,
                FormalToActual /*map*/) -> std::vector<std::vector<Dim>> {
           return {{}}; // Scalar (rank-0) output.
         }},
    };
    md.makeMultiKernelVariant = makeSumVariant;
    md.headerFile = kScanHeader;
    md.deviceFunc = "tw_sum";
    md.sharedDecls = {{"Int32X32", "warpSums"}};
    md.dynamicSharedDecls = {{-1, "counter"}};
    md.typeTemplateParams = {0};
    md.hasDtypeTemplateParam = true;
    md.hasBlockSizeTemplateParam = true;
    md.alwaysSingleBlock = true;
    Registry::registerMetadata("torch.ops.aten.sum.default", std::move(md));
  }

  // --- Multi-kernel sum intrinsics ---

  // tw.sum_head: (Tensor) -> Tensor
  // Per-block reduction. Output size = ceil(inputNumel / blockSize).
  {
    auto schema = std::make_unique<c10::FunctionSchema>(
        "tw.sum_head",
        "",
        std::vector<c10::Argument>{
            c10::Argument("input", c10::TensorType::get())},
        std::vector<c10::Argument>{
            c10::Argument("output", c10::TensorType::get())});
    Metadata md;
    md.functionSchema = Registry::ownSchema(std::move(schema));
    md.sizeArgs.ordinal = {0};
    md.hasBarrier = true;
    md.argumentMeta = {{.isRegister = false}};
    md.returnMeta = {{.isRegister = false, .reserveShape = numBlocksShape}};
    md.headerFile = kScanHeader;
    md.deviceFunc = "tw_sum_head";
    md.sharedDecls = {
        {"Int32X32", "warpSums"},
        {"uint32_t", "size"},
        {"uint32_t", "rounded"}};
    md.typeTemplateParams = {0};
    md.hasDtypeTemplateParam = true;
    md.hasBlockSizeTemplateParam = true;
    md.kernelBreakForMultiblock = true;
    md.outputConstraints = rank1Constraint;
    Registry::registerMetadata("tw.sum_head", std::move(md));
  }

  // tw.sum_final: (Tensor) -> Tensor
  // Single-block reduction of per-block sums to a scalar (rank-0) tensor.
  {
    auto schema = std::make_unique<c10::FunctionSchema>(
        "tw.sum_final",
        "",
        std::vector<c10::Argument>{
            c10::Argument("input", c10::TensorType::get())},
        std::vector<c10::Argument>{
            c10::Argument("output", c10::TensorType::get())});
    Metadata md;
    md.functionSchema = Registry::ownSchema(std::move(schema));
    md.sizeArgs.ordinal = {0};
    md.hasBarrier = true;
    md.argumentMeta = {{.isRegister = false}};
    md.returnMeta = {
        {.isRegister = false,
         .reserveShape =
             [](NodeCP /*node*/,
                nativert::ExecutionFrame& /*frame*/,
                FormalToActual /*map*/) -> std::vector<std::vector<Dim>> {
           return {{}}; // Scalar (rank-0) output.
         }},
    };
    md.inputFromPreviousKernel = 0;
    md.headerFile = kScanHeader;
    md.deviceFunc = "tw_sum_tensor";
    md.sharedDecls = {
        {"Int32X32", "warpSums"},
        {"uint32_t", "size"},
        {"uint32_t", "rounded"}};
    md.dynamicSharedDecls = {{0, "counter"}};
    md.typeTemplateParams = {0};
    md.hasDtypeTemplateParam = true;
    md.hasBlockSizeTemplateParam = true;
    md.alwaysSingleBlock = true;
    md.outputConstraints = rank1Constraint;
    Registry::registerMetadata("tw.sum_final", std::move(md));
  }

  // Stream compaction.
  {
    const auto* schema =
        findFunctionSchema("torch.ops.aten.masked_select.default");
    TORCH_CHECK(schema, "FunctionSchema not found for masked_select.default");
    Metadata md;

    md.functionSchema = schema;
    md.sizeArgs.ordinal = {0};
    md.hasBarrier = true;
    md.singleBlockIfFused = true;
    md.argumentMeta = {
        {.isRegister = true},
        {.isRegister = true},
    };
    md.returnMeta = {
        {.isRegister = false,
         .reserveShape =
             [](NodeCP node,
                nativert::ExecutionFrame& frame,
                FormalToActual map) -> std::vector<std::vector<Dim>> {
           return elementwiseInputShape(node, frame, map, 0);
         },
         .shapeSetOnDevice = true},
    };
    md.makeMultiKernelVariant = makeMaskedSelectVariant;
    md.headerFile = kScanHeader;
    md.deviceFunc = "masked_select";
    md.sharedDecls = {{"Int32X32", "warpSums"}, {"uint32_t", "counter"}};
    md.typeTemplateParams = {0};
    md.hasBlockSizeTemplateParam = true;
    md.alwaysSingleBlock = true;
    Registry::registerMetadata(
        "torch.ops.aten.masked_select.default", std::move(md));
  }

  // --- Torchwave intrinsics for multi-kernel masked_select ---

  // tw.masked_select_head: (Tensor, Tensor) -> Tensor
  // Per-block counts. Output size = ceil(input0_numel / 256).
  {
    auto schema = std::make_unique<c10::FunctionSchema>(
        "tw.masked_select_head",
        "",
        std::vector<c10::Argument>{
            c10::Argument("input", c10::TensorType::get()),
            c10::Argument("mask", c10::TensorType::get())},
        std::vector<c10::Argument>{
            c10::Argument("output", c10::TensorType::get())});
    Metadata md;

    md.functionSchema = Registry::ownSchema(std::move(schema));
    md.sizeArgs.ordinal = {0};
    md.hasBarrier = true;
    md.argumentMeta = {
        {.isRegister = false},
        {.isRegister = false},
    };
    md.returnMeta = {{.isRegister = false, .reserveShape = numBlocksShape}};
    md.headerFile = kScanHeader;
    md.deviceFunc = "masked_select_head";
    md.sharedDecls = {
        {"Int32X32", "warpSums"},
        {"uint32_t", "size"},
        {"uint32_t", "rounded"}};
    md.typeTemplateParams = {0};
    md.hasBlockSizeTemplateParam = true;
    md.kernelBreakForMultiblock = true;
    md.outputConstraints = rank1Constraint;
    Registry::registerMetadata("tw.masked_select_head", std::move(md));
  }

  // tw.add_sizes: (Tensor) -> int64
  // Sums per-block counts into a total. Result needed on host.
  {
    auto schema = std::make_unique<c10::FunctionSchema>(
        "tw.add_sizes",
        "",
        std::vector<c10::Argument>{
            c10::Argument("input", c10::TensorType::get())},
        std::vector<c10::Argument>{
            c10::Argument("output", c10::IntType::get())});
    Metadata md;

    md.functionSchema = Registry::ownSchema(std::move(schema));
    md.sizeArgs.ordinal = {0};
    md.hasBarrier = true;
    md.argumentMeta = {
        {.isRegister = false},
    };
    md.returnMeta = {
        {.neededOnHost = true},
    };
    md.headerFile = kScanHeader;
    md.deviceFunc = "add_sizes";
    md.sharedDecls = {
        {"Int32X32", "warpSums"},
        {"uint32_t", "size"},
        {"uint32_t", "rounded"},
        {"uint32_t", "counter"}};

    md.typeTemplateParams = {};
    md.hasBlockSizeTemplateParam = true;
    md.inputFromPreviousKernel = 0;
    md.kernelBreakForMultiblock = true;
    md.alwaysSingleBlock = true;
    Registry::registerMetadata("tw.add_sizes", std::move(md));
  }

  // tw.masked_select_final: (Tensor, Tensor, Tensor, int) -> Tensor
  // Final scatter. Output size comes from counts.
  {
    auto schema = std::make_unique<c10::FunctionSchema>(
        "tw.masked_select_final",
        "",
        std::vector<c10::Argument>{
            c10::Argument("input", c10::TensorType::get()),
            c10::Argument("mask", c10::TensorType::get()),
            c10::Argument("counts", c10::TensorType::get()),
            c10::Argument("total", c10::IntType::get())},
        std::vector<c10::Argument>{
            c10::Argument("output", c10::TensorType::get())});
    Metadata md;

    md.functionSchema = Registry::ownSchema(std::move(schema));
    md.sizeArgs.ordinal = {0};
    md.argumentMeta = {
        {.isRegister = false},
        {.isRegister = false},
        {.isRegister = false},
        {.isRegister = false},
    };
    md.returnMeta = {
        {.isRegister = false,
         .reserveShape = inputShape,
         .shapeSetOnDevice = true},
    };
    md.inputFromPreviousKernel = 3;
    md.headerFile = kScanHeader;
    md.deviceFunc = "masked_select_final";
    md.sharedDecls = {
        {"Int32X32", "warpSums"},
        {"uint32_t", "size"},
        {"uint32_t", "rounded"}};
    md.typeTemplateParams = {0};
    md.hasBarrier = true;
    md.hasBlockSizeTemplateParam = true;
    md.outputConstraints = rank1Constraint;
    Registry::registerMetadata("tw.masked_select_final", std::move(md));
  }

  // --- Cumulative sum ---

  // torch.ops.aten.cumsum.default: (Tensor, int, *, ScalarType? dtype=None)
  // -> Tensor.  Single-block cumsum with multi-kernel variant for large inputs.
  // The dim argument is ignored (always reduces over the flat tensor).
  {
    const auto* schema =
        findFunctionSchema("torch.ops.aten.cumsum.default");
    TORCH_CHECK(schema, "FunctionSchema not found for cumsum.default");
    Metadata md;
    md.functionSchema = schema;
    md.sizeArgs.ordinal = {0};
    md.sizeShortcut = SizeShortcut::kMax;
    md.hasBarrier = true;
    md.singleBlockIfFused = true;
    md.defaultInputMeta();
    md.returnMeta = {
        {.isRegister = false,
         .reserveShape =
             [](NodeCP node,
                nativert::ExecutionFrame& frame,
                FormalToActual map) -> std::vector<std::vector<Dim>> {
           return elementwiseInputShape(node, frame, map, 0);
         }},
    };
    md.makeMultiKernelVariant = makeCumsumVariant;
    md.headerFile = kScanHeader;
    md.deviceFunc = "cumsum";
    md.sharedDecls = {
        {"Int32X32", "warpSums"},
        {"uint32_t", "size"},
        {"uint32_t", "rounded"}};
    md.dynamicSharedDecls = {{-1, "counter"}};
    md.typeTemplateParams = {0};
    md.hasDtypeTemplateParam = true;
    md.hasBlockSizeTemplateParam = true;
    md.alwaysSingleBlock = true;
    md.only1d = true;
    md.templateAttrs = {"dim"};
    Registry::registerMetadata(
        "torch.ops.aten.cumsum.default", std::move(md));
  }

  // tw.cumsum_head: (Tensor) -> Tensor
  // Per-block sums. Output size = ceil(inputNumel / blockSize).
  {
    auto schema = std::make_unique<c10::FunctionSchema>(
        "tw.cumsum_head",
        "",
        std::vector<c10::Argument>{
            c10::Argument("input", c10::TensorType::get())},
        std::vector<c10::Argument>{
            c10::Argument("output", c10::TensorType::get())});
    Metadata md;
    md.functionSchema = Registry::ownSchema(std::move(schema));
    md.sizeArgs.ordinal = {0};
    md.hasBarrier = true;
    md.argumentMeta = {
        {.isRegister = false},
    };
    md.returnMeta = {{.isRegister = false, .reserveShape = numBlocksShape}};
    md.headerFile = kScanHeader;
    md.deviceFunc = "cumsum_head";
    md.sharedDecls = {
        {"Int32X32", "warpSums"},
        {"uint32_t", "size"},
        {"uint32_t", "rounded"}};
    md.typeTemplateParams = {0};
    md.hasDtypeTemplateParam = true;
    md.hasBlockSizeTemplateParam = true;
    md.kernelBreakForMultiblock = true;
    md.only1d = true;
    md.outputConstraints = rank1Constraint;
    Registry::registerMetadata("tw.cumsum_head", std::move(md));
  }

  // tw.cumsum_add_sizes: (Tensor) -> int (link-only output)
  // Prefix-sums the per-block counts in place. Output is link-only.
  {
    auto schema = std::make_unique<c10::FunctionSchema>(
        "tw.cumsum_add_sizes",
        "",
        std::vector<c10::Argument>{
            c10::Argument("input", c10::TensorType::get())},
        std::vector<c10::Argument>{
            c10::Argument("output", c10::IntType::get())});
    Metadata md;
    md.functionSchema = Registry::ownSchema(std::move(schema));
    md.sizeArgs.ordinal = {0};
    md.hasBarrier = true;
    md.argumentMeta = {
        {.isRegister = false},
    };
    md.returnMeta = {
        {.linkOnly = true},
    };
    md.headerFile = kScanHeader;
    md.deviceFunc = "add_sizes";
    md.sharedDecls = {
        {"Int32X32", "warpSums"},
        {"uint32_t", "size"},
        {"uint32_t", "rounded"},
        {"uint32_t", "counter"}};
    md.hasDtypeTemplateParam = true;
    md.hasBlockSizeTemplateParam = true;
    md.inputFromPreviousKernel = 0;
    md.kernelBreakForMultiblock = true;
    md.alwaysSingleBlock = true;
    md.only1d = true;
    Registry::registerMetadata("tw.cumsum_add_sizes", std::move(md));
  }

  // tw.cumsum_final: (Tensor, Tensor, int) -> Tensor
  // Final prefix sum using per-block prefix sums from add_sizes.
  // The third input (link) is link-only for ordering.
  {
    auto schema = std::make_unique<c10::FunctionSchema>(
        "tw.cumsum_final",
        "",
        std::vector<c10::Argument>{
            c10::Argument("input", c10::TensorType::get()),
            c10::Argument("counts", c10::TensorType::get()),
            c10::Argument("link", c10::IntType::get())},
        std::vector<c10::Argument>{
            c10::Argument("output", c10::TensorType::get())});
    Metadata md;
    md.functionSchema = Registry::ownSchema(std::move(schema));
    md.sizeArgs.ordinal = {0};
    md.argumentMeta = {
        {.isRegister = false},
        {.isRegister = false},
        {.linkOnly = true},
    };
    md.returnMeta = {
        {.isRegister = false,
         .reserveShape = inputShape},
    };
    md.inputFromPreviousKernel = 2;
    md.headerFile = kScanHeader;
    md.deviceFunc = "cumsum_final";
    md.sharedDecls = {
        {"Int32X32", "warpSums"},
        {"uint32_t", "size"},
        {"uint32_t", "rounded"}};
    md.typeTemplateParams = {0};
    md.hasBarrier = true;
    md.hasDtypeTemplateParam = true;
    md.hasBlockSizeTemplateParam = true;
    md.only1d = true;
    md.templateAttrs = {"dim"};
    md.outputConstraints = rank1Constraint;
    Registry::registerMetadata("tw.cumsum_final", std::move(md));
  }

  // --- Exclusive sum ---

  // torch.ops.aten.exclusive_sum.default: (Tensor) -> Tensor
  // Like cumsum but exclusive: out[0]=0, out[i+1]=sum(in[0..i]),
  // output has numel+1 elements. Integer inputs promote to Long.
  {
    auto schema = std::make_unique<c10::FunctionSchema>(
        "torch.ops.aten.exclusive_sum",
        "default",
        std::vector<c10::Argument>{
            c10::Argument("input", c10::TensorType::get())},
        std::vector<c10::Argument>{
            c10::Argument("output", c10::TensorType::get())});
    Metadata md;
    md.functionSchema = Registry::ownSchema(std::move(schema));
    md.sizeArgs.ordinal = {0};
    md.sizeShortcut = SizeShortcut::kMax;
    md.hasBarrier = true;
    md.singleBlockIfFused = true;
    md.argumentMeta = {{.isRegister = false}};
    md.returnMeta = {
        {.isRegister = false, .reserveShape = inputShapePlusOne}};
    md.makeMultiKernelVariant = makeExclusiveSumVariant;
    md.headerFile = kScanHeader;
    md.deviceFunc = "exclusive_sum";
    md.sharedDecls = {
        {"Int32X32", "warpSums"},
        {"uint32_t", "size"},
        {"uint32_t", "rounded"}};
    md.dynamicSharedDecls = {{-1, "counter"}};
    md.typeTemplateParams = {0};
    md.hasDtypeTemplateParam = true;
    md.hasBlockSizeTemplateParam = true;
    md.alwaysSingleBlock = true;
    md.only1d = true;
    md.outputConstraints = rank1Constraint;
    Registry::registerMetadata(
        "torch.ops.aten.exclusive_sum.default", std::move(md));
  }

  // tw.exclusive_sum_head: (Tensor) -> Tensor
  // Per-block sums (same computation as cumsum_head).
  {
    auto schema = std::make_unique<c10::FunctionSchema>(
        "tw.exclusive_sum_head",
        "",
        std::vector<c10::Argument>{
            c10::Argument("input", c10::TensorType::get())},
        std::vector<c10::Argument>{
            c10::Argument("output", c10::TensorType::get())});
    Metadata md;
    md.functionSchema = Registry::ownSchema(std::move(schema));
    md.sizeArgs.ordinal = {0};
    md.hasBarrier = true;
    md.argumentMeta = {{.isRegister = false}};
    md.returnMeta = {{.isRegister = false, .reserveShape = numBlocksShape}};
    md.headerFile = kScanHeader;
    md.deviceFunc = "exclusive_sum_head";
    md.sharedDecls = {
        {"Int32X32", "warpSums"},
        {"uint32_t", "size"},
        {"uint32_t", "rounded"}};
    md.typeTemplateParams = {0};
    md.hasDtypeTemplateParam = true;
    md.hasBlockSizeTemplateParam = true;
    md.kernelBreakForMultiblock = true;
    md.only1d = true;
    md.outputConstraints = rank1Constraint;
    Registry::registerMetadata("tw.exclusive_sum_head", std::move(md));
  }

  // tw.exclusive_sum_add_sizes: (Tensor) -> int (link-only output)
  {
    auto schema = std::make_unique<c10::FunctionSchema>(
        "tw.exclusive_sum_add_sizes",
        "",
        std::vector<c10::Argument>{
            c10::Argument("input", c10::TensorType::get())},
        std::vector<c10::Argument>{
            c10::Argument("output", c10::IntType::get())});
    Metadata md;
    md.functionSchema = Registry::ownSchema(std::move(schema));
    md.sizeArgs.ordinal = {0};
    md.hasBarrier = true;
    md.argumentMeta = {{.isRegister = false}};
    md.returnMeta = {{.linkOnly = true}};
    md.headerFile = kScanHeader;
    md.deviceFunc = "add_sizes";
    md.sharedDecls = {
        {"Int32X32", "warpSums"},
        {"uint32_t", "size"},
        {"uint32_t", "rounded"},
        {"uint32_t", "counter"}};
    md.hasDtypeTemplateParam = true;
    md.hasBlockSizeTemplateParam = true;
    md.inputFromPreviousKernel = 0;
    md.kernelBreakForMultiblock = true;
    md.alwaysSingleBlock = true;
    md.only1d = true;
    Registry::registerMetadata("tw.exclusive_sum_add_sizes", std::move(md));
  }

  // tw.exclusive_sum_final: (Tensor, Tensor, int) -> Tensor
  // Output has numel+1 elements.
  {
    auto schema = std::make_unique<c10::FunctionSchema>(
        "tw.exclusive_sum_final",
        "",
        std::vector<c10::Argument>{
            c10::Argument("input", c10::TensorType::get()),
            c10::Argument("counts", c10::TensorType::get()),
            c10::Argument("link", c10::IntType::get())},
        std::vector<c10::Argument>{
            c10::Argument("output", c10::TensorType::get())});
    Metadata md;
    md.functionSchema = Registry::ownSchema(std::move(schema));
    md.sizeArgs.ordinal = {0};
    md.argumentMeta = {
        {.isRegister = false},
        {.isRegister = false},
        {.linkOnly = true},
    };
    md.returnMeta = {
        {.isRegister = false, .reserveShape = inputShapePlusOne}};
    md.inputFromPreviousKernel = 2;
    md.headerFile = kScanHeader;
    md.deviceFunc = "exclusive_sum_final";
    md.sharedDecls = {
        {"Int32X32", "warpSums"},
        {"uint32_t", "size"},
        {"uint32_t", "rounded"}};
    md.typeTemplateParams = {0};
    md.hasBarrier = true;
    md.hasDtypeTemplateParam = true;
    md.hasBlockSizeTemplateParam = true;
    md.only1d = true;
    md.outputConstraints = rank1Constraint;
    Registry::registerMetadata("tw.exclusive_sum_final", std::move(md));
  }

  // --- repeat_interleave ---

  // torch.ops.aten.repeat_interleave.self_Tensor: (Tensor, Tensor, int?,
  // int?) -> Tensor. Always split into head + final via makeMultiKernelVariant.
  {
    const auto* schema =
        findFunctionSchema("torch.ops.aten.repeat_interleave.self_Tensor");
    TORCH_CHECK(
        schema, "FunctionSchema not found for repeat_interleave.self_Tensor");
    Metadata md;
    md.functionSchema = schema;
    md.defaultInputMeta();
    md.defaultOutputMeta();
    md.sizeArgs.ordinal = {1};
    md.makeMultiKernelVariant = makeRepeatInterleaveVariant;
    Registry::registerMetadata(
        "torch.ops.aten.repeat_interleave.self_Tensor", std::move(md));
  }

  // tw.repeat_interleave_head: (Tensor) -> (Tensor, int)
  // Prefix-sums the repeats tensor in place. Returns total as scalar int
  // (needed on host for allocation of final output).
  {
    auto schema = std::make_unique<c10::FunctionSchema>(
        "tw.repeat_interleave_head",
        "",
        std::vector<c10::Argument>{
            c10::Argument("repeats", c10::TensorType::get())},
        std::vector<c10::Argument>{
            c10::Argument("prefix", c10::TensorType::get()),
            c10::Argument("total", c10::IntType::get())});
    Metadata md;
    md.functionSchema = Registry::ownSchema(std::move(schema));
    md.sizeArgs.ordinal = {0};
    md.hasBarrier = true;
    md.argumentMeta = {
        {.isRegister = false},
    };
    md.returnMeta = {
        {.isRegister = false,
         .reserveShape = inputShape},
        {.neededOnHost = true},
    };
    md.headerFile = kScanHeader;
    md.deviceFunc = "add_sizes";
    md.sharedDecls = {
        {"Int32X32", "warpSums"},
        {"uint32_t", "size"},
        {"uint32_t", "rounded"},
        {"uint32_t", "counter"}};
    md.hasBlockSizeTemplateParam = true;
    md.kernelBreakForMultiblock = true;
    md.alwaysSingleBlock = true;
    md.outputConstraints = [](NodeCP /*node*/, const ValueTypes& /*types*/)
        -> std::vector<ValueConstraint> { return {{.rank = 1}, {}}; };
    Registry::registerMetadata("tw.repeat_interleave_head", std::move(md));
  }

  // tw.repeat_interleave_final: (Tensor, Tensor, int) -> Tensor
  // Scatters self using prefix sums. Output size = total from head.
  {
    auto schema = std::make_unique<c10::FunctionSchema>(
        "tw.repeat_interleave_final",
        "",
        std::vector<c10::Argument>{
            c10::Argument("input", c10::TensorType::get()),
            c10::Argument("prefix", c10::TensorType::get()),
            c10::Argument("total", c10::IntType::get())},
        std::vector<c10::Argument>{
            c10::Argument("output", c10::TensorType::get())});
    Metadata md;
    md.functionSchema = Registry::ownSchema(std::move(schema));
    md.sizeArgs.ordinal = {0};
    md.argumentMeta = {
        {.isRegister = false},
        {.isRegister = false},
        {.isRegister = false},
    };
    md.returnMeta = {
        {.isRegister = false,
         .reserveShape =
             [](NodeCP node,
                nativert::ExecutionFrame& frame,
                FormalToActual map) -> std::vector<std::vector<Dim>> {
           auto total = paramSymInt(node->inputs()[2].value, frame, map);
           return {{static_cast<Dim>(total)}};
         }},
    };
    md.inputFromPreviousKernel = 2;
    md.headerFile = kScanHeader;
    md.deviceFunc = "repeat_interleave_final";
    md.sharedDecls = {
        {"uint32_t", "size"},
        {"uint32_t", "rounded"}};
    md.typeTemplateParams = {0};
    md.hasBarrier = true;
    md.hasBlockSizeTemplateParam = true;
    md.outputConstraints = rank1Constraint;
    Registry::registerMetadata("tw.repeat_interleave_final", std::move(md));
  }

}

} // namespace torch::wave
