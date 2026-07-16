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
  // An empty or unrecognized dtype string (e.g. dtype="" serialized for a
  // factory op that infers its type from the input) must fall back to the
  // resolved dtype name; otherwise downstream codegen defaults it to Float.
  if (dtypeStr.empty()) {
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
  // The op input feeds both head and final, and the head counts feed both
  // add_sizes and final: each is consumed by more than one part of this
  // expansion, so it must not be freed as a per-op intermediate.
  waveGraph->declareMultiplyReferencedInput(single->inputs()[0].value);
  waveGraph->declareMultiplyReferencedInput(headOutput);
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
  // Consumed by more than one part of this expansion (input by head+final,
  // counts by add_sizes+final); keep out of the freeable intermediates.
  waveGraph->declareMultiplyReferencedInput(single->inputs()[0].value);
  waveGraph->declareMultiplyReferencedInput(headOutput);
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
  // input, mask and counts are each consumed by more than one part of this
  // expansion (input/mask by head+final, counts by add_sizes+final); keep them
  // out of the freeable intermediates.
  waveGraph->declareMultiplyReferencedInput(single->inputs()[0].value);
  waveGraph->declareMultiplyReferencedInput(single->inputs()[1].value);
  waveGraph->declareMultiplyReferencedInput(headOutput);
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

// Returns node's input value at 'idx', erroring if it is absent. These ops
// require the input, so a missing one is a fatal error rather than a case to
// silently skip.
nativert::Value* inputAt(NodeCP node, size_t idx) {
  const auto& inputs = node->inputs();
  TORCH_CHECK(
      idx < inputs.size() && inputs[idx].value,
      node->target(),
      ": required input ",
      idx,
      " is missing");
  return inputs[idx].value;
}

void resolveDtypeFromInput(nativert::Node* node, const ValueTypes& types) {
  if (node->tryGetAttribute("dtype") || node->tryGetInput("dtype")) {
    return;
  }
  auto inputId = inputAt(node, 0)->id();
  if (inputId < static_cast<int>(types.types.size()) && types.types[inputId]) {
    auto inputDtype = types.types[inputId]->dtype();
    auto outDtype = c10::isIntegralType(inputDtype, /*includeBool=*/true)
        ? c10::ScalarType::Long
        : inputDtype;
    node->addAttribute({"dtype", outDtype});
  }
}

std::vector<ValueConstraint> rank1Constraint(
    NodeCP /*node*/,
    const ValueTypes& /*types*/) {
  // Used by ops that materialize a fresh dense 1-D output (cumsum, exclusive
  // sum, masked_select, repeat_interleave and their multi-kernel intrinsics),
  // so the result is contiguous.
  return {{.rank = 1, .contiguous = true}};
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

// Casts int64 scalar attributes to double when the first input is a
// floating-point tensor. Without this, an attribute like other=2 (int64)
// gets copied as raw bytes to a param slot the kernel reads as double.
void castScalarAttrsToInputDtype(
    nativert::Node* node,
    const ValueTypes& types) {
  auto inputId = inputAt(node, 0)->id();
  if (inputId >= static_cast<int>(types.types.size()) ||
      !types.types[inputId]) {
    return;
  }
  auto dtype = types.types[inputId]->dtype();
  if (!at::isFloatingType(dtype)) {
    return;
  }
  // PyTorch performs Tensor-Scalar ops -- including the comparison chains that
  // build selection masks (ge/lt/gt/le.Scalar) -- in the tensor's dtype: the
  // scalar is first converted to that dtype. For a float32 tensor that means
  // rounding the scalar to float32 before the op. Without this, a threshold
  // like 0.1 stays a double and the generated `x >= threshold` promotes the
  // float `x` to double, comparing in double; at a boundary value that flips
  // the mask bit vs eager and the difference propagates into downstream
  // jagged/masked-select counts. The float Constant kind is double, so we keep
  // a double whose value is exactly representable as float32 -- a subsequent
  // double comparison/arith then yields the same result as float.
  const bool toFloat = (dtype == c10::ScalarType::Float);
  for (auto& attr : node->attributes()) {
    auto& value = const_cast<nativert::Attribute&>(attr).value;
    if (std::holds_alternative<int64_t>(value)) {
      double d = static_cast<double>(std::get<int64_t>(value));
      value = toFloat ? static_cast<double>(static_cast<float>(d)) : d;
    } else if (toFloat && std::holds_alternative<double>(value)) {
      value = static_cast<double>(static_cast<float>(std::get<double>(value)));
    }
  }
}

void resolveDefaultDtype(nativert::Node* node, const ValueTypes& /*types*/) {
  if (node->tryGetAttribute("dtype") || node->tryGetInput("dtype")) {
    return;
  }
  node->addAttribute({"dtype", c10::ScalarType::Float});
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
  node->addAttribute(
      {"dtype", hasFloat ? c10::ScalarType::Float : c10::ScalarType::Long});
}

void resolveNanToNumDefaults(nativert::Node* node, const ValueTypes& types) {
  double posinfDefault = std::numeric_limits<double>::max();
  double neginfDefault = std::numeric_limits<double>::lowest();
  auto inputId = inputAt(node, 0)->id();
  if (inputId < static_cast<int>(types.types.size()) && types.types[inputId]) {
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

// aten.logit(self, eps=None). __logit takes eps as a double and only clamps
// when eps >= 0, so resolve a missing/None eps to -1.0 (the "no clamp"
// sentinel). A dynamic eps input is left as-is and passed through as a value.
void resolveLogitDefault(nativert::Node* node, const ValueTypes& /*types*/) {
  if (node->tryGetInput("eps")) {
    return;
  }
  const auto* attr = node->tryGetAttribute("eps");
  if (!attr) {
    node->addAttribute({"eps", -1.0});
  } else if (std::holds_alternative<nativert::None>(attr->value)) {
    const_cast<nativert::Attribute*>(attr)->value = -1.0;
  }
}

void resolveDtypeFromInputExact(nativert::Node* node, const ValueTypes& types) {
  if (node->tryGetAttribute("dtype") || node->tryGetInput("dtype")) {
    return;
  }
  auto inputId = inputAt(node, 0)->id();
  if (inputId < static_cast<int>(types.types.size()) && types.types[inputId]) {
    auto inputDtype = types.types[inputId]->dtype();
    node->addAttribute({"dtype", inputDtype});
  }
}

std::vector<ValueConstraint> sizeAttrRankConstraint(
    NodeCP node,
    const ValueTypes& /*types*/) {
  const auto* sizeAttr = node->tryGetAttribute("size");
  if (sizeAttr) {
    const auto& size = std::get<std::vector<int64_t>>(sizeAttr->value);
    // Factory ops (zeros/ones/full/empty/...) materialize a fresh dense tensor.
    return {{.rank = static_cast<int8_t>(size.size()), .contiguous = true}};
  }
  return {};
}

// Reads a statically-known integer argument 'name' from 'node' (a constant
// attribute). Returns nullopt when the argument is absent, None, or supplied as
// a dynamic symint input -- in which case the value is not known at compile
// time.
std::optional<int64_t> constIntArg(NodeCP node, const char* name) {
  const auto* attr = node->tryGetAttribute(name);
  if (attr && std::holds_alternative<int64_t>(attr->value)) {
    return std::get<int64_t>(attr->value);
  }
  return std::nullopt;
}

// Whether argument 'name' is supplied as a dynamic (symint) input rather than a
// constant attribute.
bool hasDynamicArg(NodeCP node, const char* name) {
  return node->tryGetInput(name) != nullptr;
}

// A view-like op (view/slice/...) is materialized as a before-launch host view
// (OutputDesc::viewNode), which requires every shape argument to be available
// in the frame when the kernel's outputs are set up. If any argument other than
// the viewed tensor is a dynamic Value (e.g. slice end = sym_size/item, or a
// dynamic size list), that bound may be produced by the same wave and is not
// yet computed at setup time, so the op must run as a standalone (after its
// inputs are ready) instead. The viewed tensor (viewOfArg) is the only expected
// Value operand; any additional Value input is such a dynamic shape argument.
bool viewHasDynamicShapeArgs(NodeCP node, const ValueTypes& /*types*/) {
  return node->inputs().size() > 1;
}

std::vector<std::vector<Dim>> sizeAttrReserveShape(
    NodeCP /*node*/,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map,
    NodeCP originalFormalNode,
    const NodeMap& nodeMap) {
  auto size =
      paramIntListByName(originalFormalNode, "size", frame, map, nodeMap);
  return {{size.begin(), size.end()}};
}

std::vector<std::vector<Dim>> numBlocksShape(
    NodeCP node,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map,
    NodeCP /*originalFormalNode*/,
    const NodeMap& /*nodeMap*/) {
  auto tensor = paramTensor(node->inputs()[0].value, frame, map);
  auto blockSize = WaveConfig::get().blockSize;
  // At least one block: makeGrid always launches >=1 block (even for an empty
  // input), and that block's head kernel unconditionally writes one per-block
  // partial sum (out[blockInOp]).  A zero-length counts buffer would make that
  // write dereference null storage.
  auto numBlocks = std::max<Dim>(
      1, static_cast<Dim>((tensor.numel() + blockSize - 1) / blockSize));
  return {{numBlocks}};
}

std::vector<std::vector<Dim>> inputShape(
    NodeCP node,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map,
    NodeCP /*originalFormalNode*/,
    const NodeMap& /*nodeMap*/) {
  auto tensor = paramTensor(node->inputs()[0].value, frame, map);
  auto sizes = tensor.sizes();
  return {{sizes.begin(), sizes.end()}};
}

std::vector<std::vector<Dim>> inputShapePlusOne(
    NodeCP node,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map,
    NodeCP /*originalFormalNode*/,
    const NodeMap& /*nodeMap*/) {
  auto tensor = paramTensor(node->inputs()[0].value, frame, map);
  return {{static_cast<Dim>(tensor.numel() + 1)}};
}

void registerRankPreservingStandalone(
    const char* opName,
    std::optional<int32_t> viewOfArgOrdinal = std::nullopt,
    bool contiguousOutput = false) {
  MetadataBuilder builder(opName);
  builder.sizeOrdinal({0})
      .isStandalone()
      .outputConstraints(
          [contiguousOutput](
              NodeCP node,
              const ValueTypes& types) -> std::vector<ValueConstraint> {
            // contiguousOutput is true only for ops that produce a dense result
            // (e.g. aten.contiguous); a view-like op such as transpose leaves
            // it false.
            return {
                {.rank = types.rank(inputAt(node, 0)),
                 .contiguous = contiguousOutput}};
          })
      .maybeReplace(
          [](NodeCP node,
             ValueTypes& types,
             WaveGraph&) -> std::vector<std::pair<ValueCP, ValueCP>> {
            const auto& outputs = node->outputs();
            if (outputs.empty()) {
              return {};
            }
            auto* self = inputAt(node, 0);
            if (types.rank(self) == 1) {
              return {{outputs[0], self}};
            }
            return {};
          });
  if (viewOfArgOrdinal.has_value()) {
    builder.viewOfArg(*viewOfArgOrdinal);
    // A view-like rank-preserving op (e.g. transpose) only rearranges tensor
    // metadata; a materializing op (contiguousOutput) does real compute.
    if (!contiguousOutput) {
      builder.metadataOnly();
    }
  }
  builder.registerOp();
}

void registerReshapeLikeOp(
    const char* opName,
    const char* shapeAttrName,
    bool isView) {
  MetadataBuilder builder(opName);
  builder.sizeOrdinal({0})
      .viewOfArg(0)
      .metadataOnly(isView)
      .shapeAttr(shapeAttrName)
      .outputConstraints(
          [shapeAttrName](
              NodeCP node,
              const ValueTypes& types) -> std::vector<ValueConstraint> {
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
            // view/reshape return a view that reinterprets the same storage, so
            // the result is contiguous exactly when the input is.
            constraint.contiguous = types.contiguous(inputAt(node, 0));
            return {constraint};
          })
      .maybeReplace(
          [shapeAttrName, isView](NodeCP node, ValueTypes& types, WaveGraph&)
              -> std::vector<std::pair<ValueCP, ValueCP>> {
            const auto& outputs = node->outputs();
            if (outputs.empty()) {
              return {};
            }
            auto* selfVal = inputAt(node, 0);
            const auto* attr = node->tryGetAttribute(shapeAttrName);

            // Identity: a rank-1 tensor reshaped/viewed to [-1] is its input.
            if (attr) {
              const auto& shape = std::get<std::vector<int64_t>>(attr->value);
              if (shape.size() == 1 && shape[0] == -1 &&
                  types.rank(selfVal) == 1) {
                return {{outputs[0], selfVal}};
              }
            }

            // reshape with a contiguous input never copies, so it is exactly a
            // view. Retarget it to aten.view (copying the shape list under
            // view's "size" attribute name) so the view-collapse below treats
            // reshape and view uniformly. Returning {} with a changed target
            // makes the optimizer re-visit the node as a view.
            if (!isView && types.contiguous(selfVal)) {
              auto* mutableNode = const_cast<nativert::Node*>(node);
              if (attr) {
                // Copy the list before addAttribute, which invalidates the
                // attribute spans (the stale "shape" attribute is then ignored
                // by view's name-based argument lookup).
                auto shapeList = std::get<std::vector<int64_t>>(attr->value);
                mutableNode->addAttribute({"size", std::move(shapeList)});
              }
              mutableNode->setTarget("torch.ops.aten.view.default");
              return {};
            }

            // Collapse a view whose only consumer is another view (or a reshape
            // that will itself become a view) when this view's input is
            // contiguous: replace this view's output with its input so the
            // consumer views the input directly. Safe because a view of a
            // contiguous tensor to any compatible shape is always legal, so the
            // consumer stays valid. This chains: once an inner view folds away,
            // the next outer view sees the same contiguous input and folds too,
            // leaving one view.
            //
            // The consumer may still read as reshape.default here: the
            // optimizer visits inputs first, so this (inner) view runs before
            // the outer reshape is retargeted to view by the branch above. A
            // reshape of a contiguous input is a view, so accept it too.
            if (isView && types.contiguous(selfVal) &&
                outputs[0]->users().size() == 1) {
              NodeCP user = *outputs[0]->users().begin();
              if (user != nullptr &&
                  (user->target() == "torch.ops.aten.view.default" ||
                   user->target() == "torch.ops.aten.reshape.default")) {
                return {{outputs[0], selfVal}};
              }
            }
            return {};
          });
  if (isView) {
    // view only reinterprets storage metadata, so give it a device function so
    // it can run fused inside a kernel (like select.int). It still runs as a
    // metadata-only standalone shortcut when it is not fused. A view with a
    // dynamic size list must stay a standalone (its bounds are not known when
    // the kernel's outputs are set up before launch).
    builder.headerFile("velox/experimental/torchwave/Views.cuh")
        .deviceFunc("tw_view")
        .isStandaloneFunc(viewHasDynamicShapeArgs);
  } else {
    // reshape may materialize a copy when its input is non-contiguous, so it
    // has no device function and runs as a standalone. A contiguous reshape is
    // retargeted to view by maybeReplace above and then picks up tw_view.
    builder.isStandalone();
  }
  builder.registerOp();
}

float elementwiseCostFromDtype(c10::ScalarType dtype, float baseCost) {
  auto elemSize = c10::elementSize(dtype);
  if (c10::isFloatingType(dtype)) {
    if (elemSize <= 4) {
      return baseCost;
    }
    auto* device = facebook::velox::wave::currentDevice();
    float ratio = device ? static_cast<float>(device->float32To64Ratio) : 2.0f;
    return baseCost * ratio;
  }
  return elemSize <= 4 ? baseCost : baseCost * 2.0f;
}

c10::ScalarType nodeOutputDtype(NodeCP node) {
  auto* wg = waveGraph();
  if (!wg) {
    return c10::ScalarType::Float;
  }
  auto& types = wg->types();
  for (auto* output : node->outputs()) {
    auto id = output->id();
    if (id >= 0 && static_cast<size_t>(id) < types.types.size() &&
        types.types[id]) {
      return types.types[id]->dtype();
    }
  }
  if (!node->inputs().empty()) {
    auto id = node->inputs()[0].value->id();
    if (id >= 0 && static_cast<size_t>(id) < types.types.size() &&
        types.types[id]) {
      return types.types[id]->dtype();
    }
  }
  return c10::ScalarType::Float;
}

float arithmeticCost(NodeCP node, const Metadata& /*meta*/) {
  return elementwiseCostFromDtype(nodeOutputDtype(node), 1.0f);
}

float mulCost(NodeCP node, const Metadata& /*meta*/) {
  auto dtype = nodeOutputDtype(node);
  if (c10::isIntegralType(dtype, true) && c10::elementSize(dtype) > 4) {
    return 4.0f;
  }
  return elementwiseCostFromDtype(dtype, 1.0f);
}

float divCost(NodeCP node, const Metadata& /*meta*/) {
  auto dtype = nodeOutputDtype(node);
  if (c10::isIntegralType(dtype, true)) {
    return c10::elementSize(dtype) > 4 ? 16.0f : 8.0f;
  }
  return elementwiseCostFromDtype(dtype, 2.0f);
}

float remainderCost(NodeCP node, const Metadata& /*meta*/) {
  auto dtype = nodeOutputDtype(node);
  if (c10::isIntegralType(dtype, true)) {
    return c10::elementSize(dtype) > 4 ? 20.0f : 10.0f;
  }
  return elementwiseCostFromDtype(dtype, 3.0f);
}

float powCost(NodeCP node, const Metadata& /*meta*/) {
  return elementwiseCostFromDtype(nodeOutputDtype(node), 8.0f);
}

float transcendentalCost(NodeCP node, const Metadata& /*meta*/) {
  return elementwiseCostFromDtype(nodeOutputDtype(node), 4.0f);
}

} // namespace

void registerBuiltins() {
  // Binary arithmetic.
  MetadataBuilder("torch.ops.aten.add.Tensor")
      .elementwise()
      .arithmeticPromotion()
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.sub.Tensor")
      .elementwise()
      .arithmeticPromotion()
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.mul.Tensor")
      .elementwise()
      .arithmeticPromotion()
      .costFunction(mulCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.div.Tensor")
      .elementwise()
      .arithmeticPromotion()
      .costFunction(divCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.remainder.Tensor")
      .elementwise()
      .arithmeticPromotion()
      .costFunction(remainderCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.fmod.Tensor")
      .elementwise()
      .arithmeticPromotion()
      .costFunction(remainderCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.pow.Tensor_Tensor")
      .elementwise()
      .arithmeticPromotion()
      .costFunction(powCost)
      .registerOp();

  // In-place binary arithmetic. PyTorch keeps these in non-functionalized
  // graphs (e.g. dense-feature normalization x.add_(mean).mul_(inv_std)).
  // Registered as elementwise; the device functions take the mutated arg by
  // reference so the write lands in self's storage. No arithmeticPromotion:
  // in-place ops keep self's dtype rather than promoting the result.
  MetadataBuilder("torch.ops.aten.add_.Tensor")
      .elementwise()
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.sub_.Tensor")
      .elementwise()
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.mul_.Tensor")
      .elementwise()
      .costFunction(mulCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.div_.Tensor")
      .elementwise()
      .costFunction(divCost)
      .registerOp();

  // In-place (Tensor, Scalar). Same __*_ device functions; the scalar operand
  // is cast to self's dtype via normalize, and no arithmeticPromotion so the
  // result keeps self's dtype.
  MetadataBuilder("torch.ops.aten.add_.Scalar")
      .elementwiseFunc("__add_")
      .normalize(castScalarAttrsToInputDtype)
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.sub_.Scalar")
      .elementwiseFunc("__sub_")
      .normalize(castScalarAttrsToInputDtype)
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.mul_.Scalar")
      .elementwiseFunc("__mul_")
      .normalize(castScalarAttrsToInputDtype)
      .costFunction(mulCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.div_.Scalar")
      .elementwiseFunc("__div_")
      .normalize(castScalarAttrsToInputDtype)
      .costFunction(divCost)
      .registerOp();

  // Binary arithmetic (Tensor, Scalar).
  MetadataBuilder("torch.ops.aten.add.Scalar")
      .elementwiseFunc("__add")
      .arithmeticPromotion()
      .normalize(castScalarAttrsToInputDtype)
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.sub.Scalar")
      .elementwiseFunc("__sub")
      .arithmeticPromotion()
      .normalize(castScalarAttrsToInputDtype)
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.mul.Scalar")
      .elementwiseFunc("__mul")
      .arithmeticPromotion()
      .normalize(castScalarAttrsToInputDtype)
      .costFunction(mulCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.div.Scalar")
      .elementwiseFunc("__div")
      .arithmeticPromotion()
      .normalize(castScalarAttrsToInputDtype)
      .costFunction(divCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.remainder.Scalar")
      .elementwiseFunc("__remainder")
      .arithmeticPromotion()
      .normalize(castScalarAttrsToInputDtype)
      .costFunction(remainderCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.fmod.Scalar")
      .elementwiseFunc("__fmod")
      .arithmeticPromotion()
      .normalize(castScalarAttrsToInputDtype)
      .costFunction(remainderCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.pow.Tensor_Scalar")
      .elementwiseFunc("__pow")
      .arithmeticPromotion()
      .normalize(castScalarAttrsToInputDtype)
      .costFunction(powCost)
      .registerOp();

  // Comparison.
  MetadataBuilder("torch.ops.aten.eq.Tensor")
      .elementwise()
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.ne.Tensor")
      .elementwise()
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.lt.Tensor")
      .elementwise()
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.le.Tensor")
      .elementwise()
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.gt.Tensor")
      .elementwise()
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.ge.Tensor")
      .elementwise()
      .costFunction(arithmeticCost)
      .registerOp();

  // Comparison (Tensor, Scalar).
  MetadataBuilder("torch.ops.aten.eq.Scalar")
      .elementwiseFunc("__eq")
      .normalize(castScalarAttrsToInputDtype)
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.ne.Scalar")
      .elementwiseFunc("__ne")
      .normalize(castScalarAttrsToInputDtype)
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.lt.Scalar")
      .elementwiseFunc("__lt")
      .normalize(castScalarAttrsToInputDtype)
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.le.Scalar")
      .elementwiseFunc("__le")
      .normalize(castScalarAttrsToInputDtype)
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.gt.Scalar")
      .elementwiseFunc("__gt")
      .normalize(castScalarAttrsToInputDtype)
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.ge.Scalar")
      .elementwiseFunc("__ge")
      .normalize(castScalarAttrsToInputDtype)
      .costFunction(arithmeticCost)
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

  // Unary math - cheap ops (single cycle).
  MetadataBuilder("torch.ops.aten.abs.default")
      .elementwise()
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.neg.default")
      .elementwise()
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.ceil.default")
      .elementwise()
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.floor.default")
      .elementwise()
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.round.default")
      .elementwise()
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.trunc.default")
      .elementwise()
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.sign.default")
      .elementwise()
      .costFunction(arithmeticCost)
      .registerOp();
  // Unary math - SFU ops (2-4 cycles).
  MetadataBuilder("torch.ops.aten.sqrt.default")
      .elementwise()
      .cost(2.0f)
      .registerOp();
  MetadataBuilder("torch.ops.aten.rsqrt.default")
      .elementwise()
      .cost(2.0f)
      .registerOp();
  MetadataBuilder("torch.ops.aten.reciprocal.default")
      .elementwise()
      .cost(2.0f)
      .registerOp();
  MetadataBuilder("torch.ops.aten.exp.default")
      .elementwise()
      .costFunction(transcendentalCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.log.default")
      .elementwise()
      .costFunction(transcendentalCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.log2.default")
      .elementwise()
      .costFunction(transcendentalCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.log10.default")
      .elementwise()
      .costFunction(transcendentalCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.log1p.default")
      .elementwise()
      .costFunction(transcendentalCost)
      .registerOp();

  // Trigonometric (SFU, ~4 cycles).
  MetadataBuilder("torch.ops.aten.sin.default")
      .elementwise()
      .costFunction(transcendentalCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.cos.default")
      .elementwise()
      .costFunction(transcendentalCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.tan.default")
      .elementwise()
      .costFunction(transcendentalCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.asin.default")
      .elementwise()
      .costFunction(transcendentalCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.acos.default")
      .elementwise()
      .costFunction(transcendentalCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.atan.default")
      .elementwise()
      .costFunction(transcendentalCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.atan2.default")
      .elementwise()
      .costFunction(transcendentalCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.sinh.default")
      .elementwise()
      .costFunction(transcendentalCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.cosh.default")
      .elementwise()
      .costFunction(transcendentalCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.tanh.default")
      .elementwise()
      .costFunction(transcendentalCost)
      .registerOp();

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
  MetadataBuilder("torch.ops.aten.relu.default")
      .elementwise()
      .costFunction(arithmeticCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.sigmoid.default")
      .elementwise()
      .costFunction(transcendentalCost)
      .registerOp();
  MetadataBuilder("torch.ops.aten.clamp.default")
      .elementwise()
      .argumentMeta(
          {{.isRegister = true},
           {.hasPresentTemplateParam = true},
           {.hasPresentTemplateParam = true}})
      .normalize(castScalarAttrsToInputDtype)
      .registerOp();
  // In-place clamp. Mirrors clamp.default; __clamp_(T&, lo, hi) writes self.
  MetadataBuilder("torch.ops.aten.clamp_.default")
      .elementwise()
      .argumentMeta(
          {{.isRegister = true},
           {.hasPresentTemplateParam = true},
           {.hasPresentTemplateParam = true}})
      .normalize(castScalarAttrsToInputDtype)
      .registerOp();
  MetadataBuilder("torch.ops.aten.nan_to_num.default")
      .elementwise()
      .normalize(resolveNanToNumDefaults)
      .registerOp();

  // logit (inverse sigmoid): log(self / (1 - self)), with an optional eps that
  // clamps self to [eps, 1 - eps]. Elementwise; resolveLogitDefault turns a
  // None eps into the -1 "no clamp" sentinel that __logit understands.
  MetadataBuilder("torch.ops.aten.logit.default")
      .elementwise()
      .normalize(resolveLogitDefault)
      .costFunction(transcendentalCost)
      .registerOp();

  // Type cast.
  MetadataBuilder("torch.ops.aten.to.dtype")
      .elementwiseFunc("__to")
      .numArgs(1)
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

  // _to_copy: functional copy/cast. Same as to.dtype -- the cast is driven by
  // the (here optional) dtype attribute, named "dtype" exactly as in to.dtype,
  // so no attribute rename is needed. When dtype is absent (a pure device/copy
  // with no type change) the cast is the identity. The other kwargs only affect
  // placement/layout, which wave handles separately, so they are ignored.
  MetadataBuilder("torch.ops.aten._to_copy.default")
      .elementwiseFunc("__to")
      .numArgs(1)
      .generateCall([](std::stringstream& ss,
                       NodeCP node,
                       std::vector<std::string> args) {
        TORCH_CHECK(!args.empty(), "_to_copy requires at least one input");
        const auto* dtypeAttr = node->tryGetAttribute("dtype");
        if (!dtypeAttr ||
            std::holds_alternative<nativert::None>(dtypeAttr->value)) {
          ss << args[0];
          return;
        }
        ss << "static_cast<" << cudaTypeFromDtype(*dtypeAttr) << ">(" << args[0]
           << ")";
      })
      .ignoreAttrs(
          {"layout", "device", "pin_memory", "non_blocking", "memory_format"})
      .registerOp();

  // scalar_tensor: materialize a Scalar as a 0-d (single-element) tensor.
  // Elementwise with one (register) scalar input; the output is always one
  // element and its value is, like to.dtype, a static_cast of the scalar to the
  // type named by the dtype attribute (the literal's own type when dtype is
  // absent/None). The layout/device/pin_memory kwargs only affect placement, so
  // they are ignored.
  MetadataBuilder("torch.ops.aten.scalar_tensor.default")
      .elementwiseFunc("__scalar_tensor")
      .numArgs(1)
      // The schema has 5 args (s, dtype, layout, device, pin_memory); only the
      // scalar s is a real (register) input, the rest arrive as attributes.
      // argumentMeta must still cover every schema argument.
      .argumentMeta({{.isRegister = true}, {}, {}, {}, {}})
      .returnMeta(
          {{.isRegister = true,
            .reserveShape = [](NodeCP /*node*/,
                               nativert::ExecutionFrame& /*frame*/,
                               const FormalToActual& /*map*/,
                               NodeCP /*originalFormalNode*/,
                               const NodeMap& /*nodeMap*/)
                -> std::vector<std::vector<Dim>> {
              // 0-d (rank-0) output: one element, matching PyTorch's
              // scalar_tensor shape.
              return {{}};
            }}})
      .generateCall([](std::stringstream& ss,
                       NodeCP node,
                       std::vector<std::string> args) {
        TORCH_CHECK(!args.empty(), "scalar_tensor requires at least one input");
        const auto* dtypeAttr = node->tryGetAttribute("dtype");
        if (!dtypeAttr ||
            std::holds_alternative<nativert::None>(dtypeAttr->value)) {
          ss << args[0];
          return;
        }
        ss << "static_cast<" << cudaTypeFromDtype(*dtypeAttr) << ">(" << args[0]
           << ")";
      })
      .ignoreAttrs({"layout", "device", "pin_memory"})
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
            .reserveShape = [](NodeCP node,
                               nativert::ExecutionFrame& frame,
                               const FormalToActual& map,
                               NodeCP /*originalFormalNode*/,
                               const NodeMap& /*nodeMap*/)
                -> std::vector<std::vector<Dim>> {
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
            .reserveShape = [](NodeCP node,
                               nativert::ExecutionFrame& frame,
                               const FormalToActual& map,
                               NodeCP /*originalFormalNode*/,
                               const NodeMap& /*nodeMap*/)
                -> std::vector<std::vector<Dim>> {
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
            auto outDtype = resolveOutDtype(node, &waveGraph).first;
            auto* graph = waveGraph.graph();
            auto* zerosNode =
                graph->createNode("torch.ops.aten.zeros.default", {});
            zerosNode->addAttribute(
                {sizeAttr->name,
                 std::get<std::vector<int64_t>>(sizeAttr->value)});
            // aten.zeros runs as a standalone via nativert C10Kernel, whose
            // boxed schema expects an int ScalarType for `dtype`.  Emit a typed
            // ScalarType (not the string name) so it unboxes natively, with no
            // string-reinterpret workaround needed in
            // prefillStackWithStaticArgs.
            zerosNode->addAttribute({"dtype", outDtype});
            // new_zeros inherits its input's device; the rewritten zeros has no
            // such input, so pin it to the wave (GPU) device.  Without this the
            // C10 zeros falls back to CPU and trips device-mismatch checks.
            if (auto* dev = facebook::velox::wave::currentDevice()) {
              zerosNode->addAttribute(
                  {"device",
                   c10::Device(
                       c10::kCUDA,
                       static_cast<c10::DeviceIndex>(dev->deviceId))});
            }
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
            .reserveShape = [](NodeCP node,
                               nativert::ExecutionFrame& frame,
                               const FormalToActual& map,
                               NodeCP /*originalFormalNode*/,
                               const NodeMap& /*nodeMap*/)
                -> std::vector<std::vector<Dim>> {
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
            .reserveShape = [](NodeCP node,
                               nativert::ExecutionFrame& frame,
                               const FormalToActual& map,
                               NodeCP /*originalFormalNode*/,
                               const NodeMap& /*nodeMap*/)
                -> std::vector<std::vector<Dim>> {
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
      .returnMeta({{.isRegister = true}})
      .metadataGetter()
      .isScalarElementwise()
      .registerOp();

  // Element count. Like sym_size, reads tensor metadata; returns the numEl
  // field of wave::Tensor. x.numel() on a dynamically-shaped tensor lowers to
  // aten.sym_numel.default. Takes the whole tensor; output is a single int.
  MetadataBuilder("torch.ops.aten.sym_numel.default")
      .elementwiseFunc("__numel")
      .argumentMeta({{.isRegister = false, .wholeTensor = true}})
      .returnMeta({{.isRegister = true}})
      .metadataGetter()
      .isScalarElementwise()
      .registerOp();

  // _operator.* Python scalar ops (sym arithmetic / comparisons), e.g.
  // sym_size // constant, size - 1, or a guard like sym_size >= 0. Operands and
  // result are naked scalars (SymInt / SymFloat / SymBool), not tensors, and
  // these ops are not in the PyTorch dispatcher, so they have no
  // FunctionSchema. Register each schema-less with isScalarElementwise: the
  // elementwise codegen then runs a single iteration and writes the result to a
  // scalar parameter. The device functions are templated on the operand types,
  // so one registration serves int and float operands.
  auto registerScalarOp =
      [](const char* opName, const char* deviceFunc, int32_t arity) {
        std::vector<ArgumentMeta> args(arity, ArgumentMeta{.isRegister = true});
        // The _operator builtins name their positional params with single
        // letters from the Python signature: add(a, b), neg(a), etc. Bind by
        // these names rather than by container order (see forArguments); a
        // future op whose serialized names differ then errors instead of
        // silently miscomputing.
        std::vector<std::string> names;
        names.reserve(arity);
        for (int32_t i = 0; i < arity; ++i) {
          names.emplace_back(1, static_cast<char>('a' + i));
        }
        MetadataBuilder(opName, MetadataBuilder::NoSchema{})
            .elementwiseFunc(deviceFunc)
            .numArgs(arity)
            .isScalarElementwise()
            .argumentMeta(std::move(args))
            .argumentNames(std::move(names))
            .returnMeta({{.isRegister = true}})
            .registerOp();
      };
  // Binary arithmetic (result type follows operand promotion).
  registerScalarOp("_operator.add", "__opadd", 2);
  registerScalarOp("_operator.sub", "__opsub", 2);
  registerScalarOp("_operator.mul", "__mul", 2);
  registerScalarOp("_operator.floordiv", "__floordiv", 2);
  registerScalarOp("_operator.mod", "__remainder", 2);
  registerScalarOp("_operator.truediv", "__div", 2);
  registerScalarOp("_operator.pow", "__pow", 2);
  // Unary arithmetic.
  registerScalarOp("_operator.neg", "__neg", 1);
  registerScalarOp("_operator.abs", "__abs", 1);
  // Comparisons (result is a bool).
  registerScalarOp("_operator.eq", "__eq", 2);
  registerScalarOp("_operator.ne", "__ne", 2);
  registerScalarOp("_operator.lt", "__lt", 2);
  registerScalarOp("_operator.le", "__le", 2);
  registerScalarOp("_operator.gt", "__gt", 2);
  registerScalarOp("_operator.ge", "__ge", 2);

  // Identity-like ops: output replaces first input.
  for (const auto* opName :
       {"torch.ops.aten.detach_.default",
        "torch.ops.aten.lift_fresh_copy.default"}) {
    MetadataBuilder builder(opName);
    builder.sizeOrdinal({0}).rankArgument(0).maybeReplace(
        [](NodeCP node,
           ValueTypes& /*types*/,
           WaveGraph&) -> std::vector<std::pair<ValueCP, ValueCP>> {
          const auto& outputs = node->outputs();
          if (outputs.empty()) {
            return {};
          }
          return {{outputs[0], inputAt(node, 0)}};
        });
    if (std::string_view(opName) == "torch.ops.aten.detach_.default") {
      builder.viewOfArg(0);
    }
    builder.registerOp();
  }

  // --- Helpers for index ops ---

  auto isSingleBoolIndices =
      [](ValueTypes& types, ValueCP tensor, ValueCP indices) -> bool {
    if (!tensor || !indices) {
      return false;
    }
    auto tensorId = tensor->id();
    if (tensorId < 0 || static_cast<size_t>(tensorId) >= types.types.size() ||
        !types.types[tensorId] || types.types[tensorId]->dim() != 1) {
      return false;
    }
    auto* listPack = indices->producer();
    if (!listPack || listPack->target() != "prim.ListPack" ||
        listPack->inputs().size() != 1 || !listPack->inputs()[0].value) {
      return false;
    }
    auto* idxVal = listPack->inputs()[0].value;
    auto idxId = idxVal->id();
    return idxId >= 0 && static_cast<size_t>(idxId) < types.types.size() &&
        types.types[idxId] &&
        types.types[idxId]->dtype() == c10::ScalarType::Bool;
  };

  auto isIntegerIndices =
      [](ValueTypes& types, ValueCP tensor, ValueCP indices) -> int {
    if (!tensor || !indices) {
      return 0;
    }
    auto tensorId = tensor->id();
    if (tensorId < 0 || static_cast<size_t>(tensorId) >= types.types.size() ||
        !types.types[tensorId]) {
      return 0;
    }
    auto rank = types.types[tensorId]->dim();
    if (rank < 1 || rank > 3) {
      return 0;
    }
    auto* listPack = indices->producer();
    if (!listPack || listPack->target() != "prim.ListPack" ||
        static_cast<int64_t>(listPack->inputs().size()) != rank) {
      return 0;
    }
    for (size_t i = 0; i < listPack->inputs().size(); ++i) {
      auto* idxVal = listPack->inputs()[i].value;
      if (!idxVal) {
        return 0;
      }
      auto idxId = idxVal->id();
      if (idxId < 0 || static_cast<size_t>(idxId) >= types.types.size() ||
          !types.types[idxId]) {
        return 0;
      }
      auto dt = types.types[idxId]->dtype();
      if (dt != c10::ScalarType::Int && dt != c10::ScalarType::Long) {
        return 0;
      }
    }
    return static_cast<int>(rank);
  };

  // Shared output constraint for in-place scatter ops (index_put_,
  // index_put_elt_*, masked_put_): the result aliases the first argument
  // (self), so it has self's rank and is contiguous exactly when self is.
  auto viewOfFirstArgConstraint =
      [](NodeCP node, const ValueTypes& types) -> std::vector<ValueConstraint> {
    auto* self = inputAt(node, 0);
    return {{.rank = types.rank(self), .contiguous = types.contiguous(self)}};
  };

  // Clone: eliminate unless a user mutates the output in place.
  MetadataBuilder("torch.ops.aten.clone.default")
      .sizeOrdinal({0})
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            // clone makes a fresh, densely-laid-out copy regardless of the
            // input's layout.
            return {{.rank = types.rank(inputAt(node, 0)), .contiguous = true}};
          })
      .deviceFunc("__copyTensor")
      .typeTemplateParams({0})
      .maybeReplace(
          [](NodeCP node,
             ValueTypes& /*types*/,
             WaveGraph& waveGraph) -> std::vector<std::pair<ValueCP, ValueCP>> {
            const auto& outputs = node->outputs();
            if (outputs.empty()) {
              return {};
            }
            auto* source = inputAt(node, 0);
            auto* outputNode = waveGraph.graph()->outputNode();
            // clone(memory_format=contiguous_format) is not an identity: it
            // produces a contiguous copy that a following view() relies on.
            // Keep the clone (its __copyTensor writes a contiguous register)
            // rather than eliding it.
            const auto* mfAttr = node->tryGetAttribute("memory_format");
            if (mfAttr) {
              bool isContiguous = false;
              if (std::holds_alternative<c10::MemoryFormat>(mfAttr->value)) {
                isContiguous = std::get<c10::MemoryFormat>(mfAttr->value) ==
                    c10::MemoryFormat::Contiguous;
              } else if (std::holds_alternative<std::string>(mfAttr->value)) {
                const auto& s = std::get<std::string>(mfAttr->value);
                isContiguous = s == "contiguous_format" || s == "Contiguous";
              }
              if (isContiguous) {
                return {};
              }
            }
            for (auto* user : outputs[0]->users()) {
              if (isInPlaceMutation(user, outputs[0])) {
                return {};
              }
              // A returned clone must stay a real copy: it is a distinct output
              // tensor, and aliasing it to its source can collide with another
              // output that is the same value.
              if (user == outputNode) {
                return {};
              }
            }
            // Do not eliminate the clone if its source storage is mutated in
            // place later: the clone is a required snapshot of the pre-mutation
            // value, so it cannot be aliased to its (later-overwritten) source.
            if (baseMutatedAfter(*waveGraph.graph(), node, source)) {
              return {};
            }
            // Otherwise eliminate the clone (alias its output to the source)
            // only when the source is produced by an elementwise node whose
            // result has no other user -- a fresh value read nowhere else.  In
            // any other case (graph input, view, standalone, or shared) keep
            // the clone as a real copy.
            const auto* producer = source->producer();
            const auto* producerMeta = producer ? nodeMeta(producer) : nullptr;
            if (!producerMeta || !producerMeta->elementwise ||
                source->users().size() != 1) {
              return {};
            }
            return {{outputs[0], source}};
          })
      .ignoreAttrs({"memory_format"})
      .multiBlockReturnBarrier()
      .registerOp();

  // index.Tensor: gather elements by index. Rename to tw.idx_gather for
  // any rank when all indices are int/long.
  MetadataBuilder("torch.ops.aten.index.Tensor")
      .sizeOrdinal({1, 0})
      .isStandalone()
      .maybeReplace(
          [&isIntegerIndices](
              NodeCP node, ValueTypes& types, WaveGraph& /*waveGraph*/)
              -> std::vector<std::pair<ValueCP, ValueCP>> {
            auto n =
                isIntegerIndices(types, inputAt(node, 0), inputAt(node, 1));
            if (n > 0) {
              static const char* kNames[] = {
                  "tw.index_elt_one", "tw.index_elt_two", "tw.index_elt_three"};
              const_cast<nativert::Node*>(node)->setTarget(kNames[n - 1]);
            }
            return {};
          })
      .registerOp();

  // tw.index_elt_one/two/three: elementwise index gather with scalar
  // indices from a TensorList. Args: (Tensor self, TensorList indices).
  {
    static const char* kNames[] = {
        "tw.index_elt_one", "tw.index_elt_two", "tw.index_elt_three"};
    static const char* kFuncs[] = {
        "__index_elt_one", "__index_elt_two", "__index_elt_three"};
    auto indexEltReserve =
        [](NodeCP node,
           nativert::ExecutionFrame& frame,
           const FormalToActual& map,
           NodeCP /*originalFormalNode*/,
           const NodeMap& /*nodeMap*/) -> std::vector<std::vector<Dim>> {
      auto* indicesValue = node->inputs()[1].value;
      auto* listPack = indicesValue->producer();
      TORCH_CHECK(
          listPack && listPack->target() == "prim.ListPack",
          "index_elt_* expects prim.ListPack for indices");
      auto* firstIdx = listPack->inputs()[0].value;
      auto firstId = firstIdx->id();
      auto it = map.find(firstId);
      auto actualId = it != map.end() ? it->second : firstId;
      auto& idxTensor = frame.getIValue(actualId).toTensor();
      auto sizes = idxTensor.sizes();
      return {{sizes.begin(), sizes.end()}};
    };
    for (int numIdx = 1; numIdx <= 3; ++numIdx) {
      (void)numIdx;
      int i = numIdx - 1;
      MetadataBuilder(
          std::make_unique<c10::FunctionSchema>(
              kNames[i],
              "",
              std::vector<c10::Argument>{
                  c10::Argument("self", c10::TensorType::get()),
                  c10::Argument("indices", c10::ListType::ofTensors())},
              std::vector<c10::Argument>{
                  c10::Argument("output", c10::TensorType::get())}))
          .elementwiseFunc(kFuncs[i])
          .argumentMeta(
              {{.isRegister = false, .wholeTensor = true, .randomAccess = true},
               {.isRegister = true}})
          .returnMeta({{.isRegister = true, .reserveShape = indexEltReserve}})
          .typeTemplateParams({0})
          .hasBlockInfo()
          .registerOp();
    }
  }

  // index_put (functional, out-of-place): route through the in-place path.
  // Copy the first input with an aten.clone (unless it is already a clone with
  // a single user, which is then safe to mutate directly) and retarget the node
  // to aten.index_put_. The optimizer re-visits a node whose target changed, so
  // index_put_'s maybeReplace then performs the elementwise / masked_put
  // rewrites on the in-place form.
  MetadataBuilder("torch.ops.aten.index_put.default")
      .sizeOrdinal({0})
      .isStandalone()
      .ignoreAttrs({"accumulate"})
      .maybeReplace(
          [](NodeCP node,
             ValueTypes& types,
             WaveGraph& waveGraph) -> std::vector<std::pair<ValueCP, ValueCP>> {
            auto* mutableNode = const_cast<nativert::Node*>(node);
            auto* selfVal = inputAt(node, 0);
            auto* selfProducer = selfVal->producer();
            const bool alreadyClone = selfProducer &&
                selfProducer->target() == "torch.ops.aten.clone.default" &&
                selfVal->users().size() == 1;
            if (!alreadyClone) {
              auto selfId = selfVal->id();
              if (selfId >= static_cast<int>(types.types.size()) ||
                  !types.types[selfId]) {
                // Type not resolved at optimization time; leave index_put as a
                // standalone rather than risk a null dereference.
                return {};
              }
              auto selfDtype = types.types[selfId]->dtype();
              auto* graph = waveGraph.graph();
              auto* cloneNode = graph->createNode(
                  "torch.ops.aten.clone.default", {{"self", selfVal}});
              graph->insertBefore(cloneNode, mutableNode);
              auto* cloneOutput = waveGraph.newTensorValue(
                  cloneNode, "index_put_clone", selfDtype);
              // newTensorValue records only the dtype, so the clone output's
              // TensorMeta has no sizes (dim()==0). clone preserves self's
              // shape, so carry self's full type onto it; otherwise the
              // index_put_ rewrite below cannot read the rank and the fused
              // form (masked_put_ / index_put_elt_*) is silently skipped,
              // leaving a standalone index_put_.
              auto cloneId = cloneOutput->id();
              if (cloneId >= 0) {
                if (static_cast<size_t>(cloneId) >= types.types.size()) {
                  types.types.resize(cloneId + 1, nullptr);
                }
                types.types[cloneId] = types.types[selfId];
              }
              mutableNode->inputs()[0].value = cloneOutput;
            }
            mutableNode->setTarget("torch.ops.aten.index_put_.default");
            return {};
          })
      .registerOp();

  // masked_put_: in-place scatter with bool mask. Shortcut for index_put_
  // with a single 1D bool index tensor. Registered before index_put_ so
  // the FunctionSchema is available when createNode is called.
  {
    auto maskedPutReserve =
        [](NodeCP node,
           nativert::ExecutionFrame& frame,
           const FormalToActual& map,
           NodeCP /*originalFormalNode*/,
           const NodeMap& /*nodeMap*/) -> std::vector<std::vector<Dim>> {
      auto* selfValue = node->inputs()[0].value;
      auto selfId = selfValue->id();
      auto it = map.find(selfId);
      auto actualId = it != map.end() ? it->second : selfId;
      auto& selfTensor = frame.getIValue(actualId).toTensor();
      auto* outputValue = node->outputs()[0];
      auto outputId = outputValue->id();
      auto outputIt = map.find(outputId);
      auto actualOutputId = outputIt != map.end() ? outputIt->second : outputId;
      frame.setIValue(actualOutputId, selfTensor);
      auto sizes = selfTensor.sizes();
      return {{sizes.begin(), sizes.end()}};
    };

    MetadataBuilder(
        std::make_unique<c10::FunctionSchema>(
            "tw.masked_put_",
            "",
            std::vector<c10::Argument>{
                c10::Argument("self", c10::TensorType::get()),
                c10::Argument("mask", c10::TensorType::get()),
                c10::Argument("values", c10::TensorType::get())},
            std::vector<c10::Argument>{
                c10::Argument("output", c10::TensorType::get())}))
        .sizeOrdinal({0})
        .outputConstraints(viewOfFirstArgConstraint)
        .deviceFunc("__masked_put")
        .typeTemplateParams({0})
        .returnMeta({{.reserveShape = maskedPutReserve}})
        .sharedDecls({{"uint32_t", "size"}})
        .multiBlockReturnBarrier()
        .registerOp();
  }

  // tw.index_put_elt_one/two/three: elementwise index_put with scalar
  // indices from a TensorList. Args: (Tensor self, TensorList indices,
  // Tensor values, bool accumulate).
  {
    static const char* kNames[] = {
        "tw.index_put_elt_one",
        "tw.index_put_elt_two",
        "tw.index_put_elt_three"};
    static const char* kFuncs[] = {
        "__index_put_elt_one", "__index_put_elt_two", "__index_put_elt_three"};
    for (int numIdx = 1; numIdx <= 3; ++numIdx) {
      auto indexPutEltReserve =
          [](NodeCP node,
             nativert::ExecutionFrame& frame,
             const FormalToActual& map,
             NodeCP /*originalFormalNode*/,
             const NodeMap& /*nodeMap*/) -> std::vector<std::vector<Dim>> {
        auto* selfValue = node->inputs()[0].value;
        auto selfId = selfValue->id();
        auto it = map.find(selfId);
        auto actualId = it != map.end() ? it->second : selfId;
        auto& selfTensor = frame.getIValue(actualId).toTensor();
        auto* outputValue = node->outputs()[0];
        auto outputId = outputValue->id();
        auto outputIt = map.find(outputId);
        auto actualOutputId =
            outputIt != map.end() ? outputIt->second : outputId;
        frame.setIValue(actualOutputId, selfTensor);
        auto sizes = selfTensor.sizes();
        return {{sizes.begin(), sizes.end()}};
      };
      int i = numIdx - 1;
      MetadataBuilder(
          std::make_unique<c10::FunctionSchema>(
              kNames[i],
              "",
              std::vector<c10::Argument>{
                  c10::Argument("self", c10::TensorType::get()),
                  c10::Argument("indices", c10::ListType::ofTensors()),
                  c10::Argument("values", c10::TensorType::get()),
                  c10::Argument(
                      "accumulate",
                      c10::BoolType::get(),
                      std::nullopt,
                      c10::IValue(false))},
              std::vector<c10::Argument>{
                  c10::Argument("output", c10::TensorType::get())}))
          .elementwiseFunc(kFuncs[i])
          .outputConstraints(viewOfFirstArgConstraint)
          .argumentMeta(
              {{.isRegister = false, .wholeTensor = true, .randomAccess = true},
               {.isRegister = true},
               {.isRegister = true},
               {.isRegister = true}})
          .returnMeta(
              {{.isRegister = false,
                .reserveShape = indexPutEltReserve,
                .wholeTensor = true,
                .randomAccess = true}})
          .typeTemplateParams({0})
          .hasBlockInfo()
          .multiBlockReturnBarrier()
          .registerOp();
    }
  }

  // index_put_: in-place scatter.
  {
    MetadataBuilder("torch.ops.aten.index_put_.default")
        .isStandalone()
        .outputConstraints(viewOfFirstArgConstraint)
        .maybeReplace(
            [&isIntegerIndices, &isSingleBoolIndices](
                NodeCP node, ValueTypes& types, WaveGraph& waveGraph)
                -> std::vector<std::pair<ValueCP, ValueCP>> {
              auto* selfVal = inputAt(node, 0);
              auto* indicesVal = inputAt(node, 1);
              auto* valuesVal = inputAt(node, 2);
              auto n = isIntegerIndices(types, selfVal, indicesVal);
              if (n > 0) {
                static const char* kEltNames[] = {
                    "tw.index_put_elt_one",
                    "tw.index_put_elt_two",
                    "tw.index_put_elt_three"};
                const_cast<nativert::Node*>(node)->setTarget(kEltNames[n - 1]);
                return {};
              }
              if (isSingleBoolIndices(types, selfVal, indicesVal)) {
                auto selfId = selfVal->id();
                auto selfDtype = types.types[selfId]->dtype();
                auto* listPack = indicesVal->producer();
                auto* graph = waveGraph.graph();
                auto* maskedPutNode = graph->createNode(
                    "tw.masked_put_",
                    {{"self", selfVal},
                     {"mask",
                      const_cast<nativert::Value*>(
                          listPack->inputs()[0].value)},
                     {"values", valuesVal}});
                // Place the replacement at the original op's program position.
                // createNode inserts at the graph's current insertion point,
                // which is not the index_put_ site; leaving it there would put
                // this in-place mutation out of program order and invert the
                // memory-dependency edges computed in ParallelExpr (a read of
                // self before the mutation could be scheduled after it).
                graph->insertBefore(
                    maskedPutNode, const_cast<nativert::Node*>(node));
                auto* resultValue = waveGraph.newTensorValue(
                    maskedPutNode, "masked_put_result", selfDtype);
                return {{node->outputs()[0], resultValue}};
              }
              return {};
            })
        .registerOp();
  }

  // Transpose -- a view that permutes dims, generally not contiguous.
  registerRankPreservingStandalone("torch.ops.aten.transpose.int", 0);

  // Contiguous -- always materializes a dense result.
  registerRankPreservingStandalone(
      "torch.ops.aten.contiguous.default",
      /*viewOfArgOrdinal=*/std::nullopt,
      /*contiguousOutput=*/true);

  // Slice.
  MetadataBuilder("torch.ops.aten.slice.Tensor")
      .sizeOrdinal({0})
      .viewOfArg(0)
      .metadataOnly()
      .isStandaloneFunc(viewHasDynamicShapeArgs)
      .headerFile("velox/experimental/torchwave/Views.cuh")
      .deviceFunc("tw_slice")
      .typeTemplateParams({0})
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            auto* self = inputAt(node, 0);
            ValueConstraint constraint;
            int8_t rank = types.rank(self);
            constraint.rank = rank;
            // aten.slice returns a view. A step-1 slice preserves the input's
            // contiguity only when it slices the outermost dim, or is a
            // full-extent (no-op) slice; slicing an inner dim, or step > 1,
            // introduces a stride gap. If any parameter is a dynamic symint we
            // cannot prove contiguity, so leave it false (conservative).
            constexpr int64_t kSliceEndMax = int64_t{1} << 62;
            bool stepOne = !hasDynamicArg(node, "step") &&
                constIntArg(node, "step").value_or(1) == 1;
            auto dimOpt = constIntArg(node, "dim");
            bool dimKnown = dimOpt.has_value() && !hasDynamicArg(node, "dim");
            int64_t dim = dimOpt.value_or(0);
            if (dim < 0 && rank > 0) {
              dim += rank;
            }
            // start defaults to None (== 0); end defaults to None (== to end).
            bool startZero = !hasDynamicArg(node, "start") &&
                constIntArg(node, "start").value_or(0) == 0;
            bool endFull = !hasDynamicArg(node, "end") &&
                constIntArg(node, "end").value_or(kSliceEndMax) >= kSliceEndMax;
            bool fullSlice = startZero && endFull;
            bool dimZero = dimKnown && dim == 0;
            constraint.contiguous =
                types.contiguous(self) && stepOne && (dimZero || fullSlice);
            return {constraint};
          })
      .registerOp();

  // Select.
  MetadataBuilder("torch.ops.aten.select.int")
      .sizeOrdinal({0})
      .viewOfArg(0)
      .metadataOnly()
      .headerFile("velox/experimental/torchwave/Views.cuh")
      .deviceFunc("tw_select")
      .typeTemplateParams({0})
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            auto* self = inputAt(node, 0);
            ValueConstraint constraint;
            int8_t rank = types.rank(self);
            constraint.rank = static_cast<int8_t>(rank - 1);
            // aten.select removes 'dim'. The result is contiguous only when the
            // input is contiguous and the outermost dim is removed; removing an
            // inner dim leaves the outer strides with a gap.
            auto dimOpt = constIntArg(node, "dim");
            bool dimKnown = dimOpt.has_value() && !hasDynamicArg(node, "dim");
            int64_t dim = dimOpt.value_or(0);
            if (dim < 0 && rank > 0) {
              dim += rank;
            }
            constraint.contiguous =
                types.contiguous(self) && dimKnown && dim == 0;
            return {constraint};
          })
      .registerOp();

  // Narrow.
  MetadataBuilder("torch.ops.aten.narrow.default")
      .sizeOrdinal({0})
      .isStandalone()
      .viewOfArg(0)
      .metadataOnly()
      .rankArgument(0)
      .registerOp();

  // tw.slice_scatter: fused functional slice_scatter along dim 0 or dim 1. Args
  // mirror aten.slice_scatter (self, src, dim, start, end, step) so the rewrite
  // below can retarget the node directly. The loop iterates every output
  // element (sized by self, ordinal 0); __slice_scatter returns each element,
  // taking it from 'src' inside the slice and passing 'self' through elsewhere.
  // Both 'self' and 'src' are whole tensors read at computed offsets (so they
  // carry randomAccess, forcing them to be materialized before this op), not
  // per-loop register reads. Being functional, it needs no clone of self.
  {
    auto sliceScatterReserve =
        [](NodeCP node,
           nativert::ExecutionFrame& frame,
           const FormalToActual& map,
           NodeCP /*originalFormalNode*/,
           const NodeMap& /*nodeMap*/) -> std::vector<std::vector<Dim>> {
      // Functional slice_scatter: the output is a fresh tensor with self's
      // shape (the framework allocates it; do not alias self).
      auto* selfValue = node->inputs()[0].value;
      auto selfId = selfValue->id();
      auto it = map.find(selfId);
      auto actualId = it != map.end() ? it->second : selfId;
      auto& selfTensor = frame.getIValue(actualId).toTensor();
      auto sizes = selfTensor.sizes();
      return {{sizes.begin(), sizes.end()}};
    };

    MetadataBuilder(
        std::make_unique<c10::FunctionSchema>(
            "tw.slice_scatter",
            "",
            std::vector<c10::Argument>{
                c10::Argument("self", c10::TensorType::get()),
                c10::Argument("src", c10::TensorType::get()),
                c10::Argument(
                    "dim", c10::IntType::get(), std::nullopt, c10::IValue(0)),
                c10::Argument(
                    "start",
                    c10::OptionalType::create(c10::IntType::get()),
                    std::nullopt,
                    c10::IValue()),
                c10::Argument(
                    "end",
                    c10::OptionalType::create(c10::IntType::get()),
                    std::nullopt,
                    c10::IValue()),
                c10::Argument(
                    "step", c10::IntType::get(), std::nullopt, c10::IValue(1))},
            std::vector<c10::Argument>{
                c10::Argument("output", c10::TensorType::get())}))
        .elementwiseFunc("__slice_scatter")
        // Size the loop by self (== the output shape), so every output element
        // is visited regardless of how few elements the slice writes. The
        // runtime start/end scalars must not drive the grid.
        .sizeOrdinal({0})
        .hasIdxArg()
        .hasBlockInfo()
        .outputConstraints(
            [](NodeCP node,
               const ValueTypes& types) -> std::vector<ValueConstraint> {
              // Functional output: self's rank, freshly allocated (contiguous).
              return {
                  {.rank = types.rank(inputAt(node, 0)), .contiguous = true}};
            })
        .argumentMeta(
            {{.isRegister = false, .wholeTensor = true, .randomAccess = true},
             {.isRegister = false, .wholeTensor = true, .randomAccess = true},
             {.isRegister = true},
             {.isRegister = true},
             {.isRegister = true},
             {.isRegister = true}})
        .returnMeta({{.isRegister = true, .reserveShape = sliceScatterReserve}})
        .typeTemplateParams({0})
        .registerOp();
  }

  // slice_scatter (functional, out-of-place): returns a fresh copy of self with
  // a slice overwritten by src. For dim 0 and dim 1 with a statically-known
  // dim, retarget to the fused functional tw.slice_scatter (no clone needed: it
  // reads self and produces a fresh output). Other dims (or a dynamic dim) fall
  // back to the standalone op.
  MetadataBuilder("torch.ops.aten.slice_scatter.default")
      .sizeOrdinal({0})
      .isStandalone()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            auto* self = inputAt(node, 0);
            return {{.rank = types.rank(self), .contiguous = true}};
          })
      .maybeReplace(
          [](NodeCP node, ValueTypes& /*types*/, WaveGraph& /*waveGraph*/)
              -> std::vector<std::pair<ValueCP, ValueCP>> {
            // Fuse scatter along dim 0 or dim 1 with a statically-known dim;
            // other cases fall back to the standalone op.
            int64_t scatterDim = constIntArg(node, "dim").value_or(0);
            if (hasDynamicArg(node, "dim") ||
                (scatterDim != 0 && scatterDim != 1)) {
              return {};
            }
            const_cast<nativert::Node*>(node)->setTarget("tw.slice_scatter");
            return {};
          })
      .registerOp();

  // Unsqueeze.
  MetadataBuilder("torch.ops.aten.unsqueeze.default")
      .sizeOrdinal({0})
      .isStandalone()
      .viewOfArg(0)
      .metadataOnly()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            auto* self = inputAt(node, 0);
            ValueConstraint constraint;
            constraint.rank = static_cast<int8_t>(types.rank(self) + 1);
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
            auto* self = inputAt(node, 0);
            auto inputRank = types.rank(self);
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
            auto* self = inputAt(node, 0);
            int8_t outputRank = static_cast<int8_t>(types.rank(self) - 1);
            std::vector<ValueConstraint> constraints;
            constraints.reserve(node->outputs().size());
            for (size_t i = 0; i < node->outputs().size(); ++i) {
              constraints.push_back({outputRank});
            }
            return constraints;
          })
      .registerOp();

  // Concat.
  // concat is an alias of cat with the identical (tensors, dim) signature.
  // Retarget it to aten.cat so it shares cat's full handling (size shortcut,
  // setOutputs, special form, and the contiguous output constraint) instead of
  // duplicating a subset. The optimizer re-visits the node as cat after the
  // target changes, recomputing its constraints from cat's metadata.
  MetadataBuilder("torch.ops.aten.concat.default")
      .sizeOrdinal({0})
      .isStandalone()
      .maybeReplace(
          [](NodeCP node,
             ValueTypes& /*types*/,
             WaveGraph&) -> std::vector<std::pair<ValueCP, ValueCP>> {
            auto* mutableNode = const_cast<nativert::Node*>(node);
            // cat reads its concat dimension from a "dim" attribute; supply the
            // schema default (0) only if the node carries neither a "dim"
            // attribute nor a "dim" input.
            if (node->tryGetAttribute("dim") == nullptr &&
                node->tryGetInput("dim") == nullptr) {
              mutableNode->addAttribute({"dim", static_cast<int64_t>(0)});
            }
            mutableNode->setTarget("torch.ops.aten.cat.default");
            return {};
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
            auto* self = inputAt(node, 0);
            auto rank = types.rank(self);
            std::vector<ValueConstraint> constraints;
            constraints.reserve(node->outputs().size());
            for (size_t i = 0; i < node->outputs().size(); ++i) {
              constraints.push_back({rank});
            }
            return constraints;
          })
      .registerOp();

  // Reshape.
  registerReshapeLikeOp(
      "torch.ops.aten.reshape.default", "shape", /*isView=*/false);

  // View.
  registerReshapeLikeOp("torch.ops.aten.view.default", "size", /*isView=*/true);

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
                               const FormalToActual& /*map*/,
                               NodeCP /*originalFormalNode*/,
                               const NodeMap& /*nodeMap*/)
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
      .multiBlockReturnBarrier()
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
                               const FormalToActual& /*map*/,
                               NodeCP /*originalFormalNode*/,
                               const NodeMap& /*nodeMap*/)
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
            .reserveShape = [](NodeCP node,
                               nativert::ExecutionFrame& frame,
                               const FormalToActual& map,
                               NodeCP /*originalFormalNode*/,
                               const NodeMap& /*nodeMap*/)
                -> std::vector<std::vector<Dim>> {
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
      .outputConstraints(rank1Constraint)
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
      .multiBlockReturnBarrier()
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
      .multiBlockReturnBarrier()
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
            .reserveShape = [](NodeCP node,
                               nativert::ExecutionFrame& frame,
                               const FormalToActual& map,
                               NodeCP /*originalFormalNode*/,
                               const NodeMap& /*nodeMap*/)
                -> std::vector<std::vector<Dim>> {
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
            .reserveShape = [](NodeCP node,
                               nativert::ExecutionFrame& frame,
                               const FormalToActual& map,
                               NodeCP /*originalFormalNode*/,
                               const NodeMap& /*nodeMap*/)
                -> std::vector<std::vector<Dim>> {
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
      .outputConstraints(rank1Constraint)
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
      .multiBlockReturnBarrier()
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
      .multiBlockReturnBarrier()
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
      // The multi-block final stage's output is read cross-block by fused cat
      // consumers (e.g. the exclusive-prefix cat([zeros[1], cumsum[:-1]]) and
      // the outer cat of per-chain offsets). Without a launch boundary those
      // reads are ordered only by intra-block __syncthreads(), which is
      // insufficient across non-co-resident blocks. Gated by the runtime
      // WaveConfig::scanOutputReturnBarrier toggle so the fix can be A/B'd.
      .scanOutputReturnBarrier()
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
      .multiBlockReturnBarrier()
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
      .multiBlockReturnBarrier()
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
      // See cumsum_final: the multi-block final stage's output is read
      // cross-block by fused cat consumers (the outer cat of per-chain offsets
      // reads each chain's exclusive-prefix output), so it must end its launch.
      // Gated by the WaveConfig::scanOutputReturnBarrier toggle.
      .scanOutputReturnBarrier()
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
                               const FormalToActual& /*map*/,
                               NodeCP /*originalFormalNode*/,
                               const NodeMap& /*nodeMap*/)
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
      .cost(25.0f)
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
      .cost(25.0f)
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
      .cost(30.0f)
      .registerOp();

  // --- nonzero (1D) ---

  auto nonzeroReserve =
      [](NodeCP node,
         nativert::ExecutionFrame& frame,
         const FormalToActual& map,
         NodeCP /*originalFormalNode*/,
         const NodeMap& /*nodeMap*/) -> std::vector<std::vector<Dim>> {
    auto shapes = elementwiseInputShape(node, frame, map, 0);
    if (!shapes.empty() && !shapes[0].empty()) {
      shapes[0].push_back(1);
    }
    return shapes;
  };

  MetadataBuilder("torch.ops.aten.nonzero.default")
      .sizeOrdinal({0})
      .hasBarrier()
      .singleBlockIfFused()
      .returnMeta(
          {{.isRegister = false,
            .reserveShape = nonzeroReserve,
            .shapeSetOnDevice = true}})
      .headerFile(kScanHeader)
      .deviceFunc("nonzero1d")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "counter"}})
      .typeTemplateParams({0})
      .hasBlockSizeTemplateParam()
      .alwaysSingleBlock()
      .multiBlockReturnBarrier()
      .registerOp();

  // tw.nonzero1d_head: (Tensor) -> Tensor
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.nonzero1d_head",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .hasBarrier()
      .returnMeta({{.isRegister = false, .reserveShape = numBlocksShape}})
      .headerFile(kScanHeader)
      .deviceFunc("nonzero1d_head")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .typeTemplateParams({0})
      .hasBlockSizeTemplateParam()
      .multiBlockReturnBarrier()
      .outputConstraints(rank1Constraint)
      .registerOp();

  // tw.nonzero1d_final: (Tensor, Tensor, int) -> Tensor
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.nonzero1d_final",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get()),
              c10::Argument("counts", c10::TensorType::get()),
              c10::Argument("total", c10::IntType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .returnMeta(
          {{.isRegister = false,
            .reserveShape = [](NodeCP node,
                               nativert::ExecutionFrame& frame,
                               const FormalToActual& map,
                               NodeCP /*originalFormalNode*/,
                               const NodeMap& /*nodeMap*/)
                -> std::vector<std::vector<Dim>> {
              auto total = paramSymInt(node->inputs()[2].value, frame, map);
              return {{static_cast<Dim>(total)}};
            },
            .shapeSetOnDevice = true}})
      .inputFromPreviousKernel(2)
      .headerFile(kScanHeader)
      .deviceFunc("nonzero1d_final")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .typeTemplateParams({0})
      .hasBarrier()
      .hasBlockSizeTemplateParam()
      .alwaysSingleBlock()
      .singleBlockIfFused()
      .multiBlockReturnBarrier()
      .outputConstraints(rank1Constraint)
      .registerOp();

  // tw.nonzero1d_cg: (Tensor) -> (Tensor, Tensor[counts])
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.nonzero1d_cg",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get()),
              c10::Argument("counts", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .hasBarrier()
      .returnMeta(
          {{.isRegister = false,
            .reserveShape = nonzeroReserve,
            .shapeSetOnDevice = true},
           {.isRegister = false, .reserveShape = numBlocksShape}})
      .headerFile(kScanHeader)
      .deviceFunc("nonzero1d_cg")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"},
           {"uint32_t", "counter"}})
      .typeTemplateParams({0})
      .hasBlockSizeTemplateParam()
      .numBarriers(3)
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
      .multiBlockReturnBarrier()
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
         const FormalToActual& map,
         NodeCP /*originalFormalNode*/,
         const NodeMap& /*nodeMap*/) -> std::vector<std::vector<Dim>> {
    auto total = paramSymInt(node->inputs()[2].value, frame, map);
    return {{static_cast<Dim>(total)}};
  };

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
          {{.isRegister = false, .reserveShape = repeatInterleaveFinalReserve}})
      .inputFromPreviousKernel(2)
      .headerFile(kScanHeader)
      .deviceFunc("repeat_interleave_final")
      .sharedDecls({{"uint32_t", "size"}, {"uint32_t", "rounded"}})
      .typeTemplateParams({0})
      .hasBarrier()
      .hasBlockSizeTemplateParam()
      .multiBlockReturnBarrier()
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
            .reserveShape = [](NodeCP node,
                               nativert::ExecutionFrame& frame,
                               const FormalToActual& map,
                               NodeCP /*originalFormalNode*/,
                               const NodeMap& /*nodeMap*/)
                -> std::vector<std::vector<Dim>> {
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
            auto* self = inputAt(node, 0);
            return {{types.rank(self)}};
          })
      .normalize([](nativert::Node* node, const ValueTypes& types) {
        const auto* attr = node->tryGetAttribute("value");
        bool needsDefault =
            !attr || std::holds_alternative<nativert::None>(attr->value);
        if (!needsDefault) {
          return;
        }
        bool isFloat = false;
        if (!node->inputs().empty()) {
          auto inputId = node->inputs()[0].value->id();
          if (inputId < static_cast<int>(types.types.size()) &&
              types.types[inputId]) {
            auto dt = types.types[inputId]->dtype();
            isFloat = at::isFloatingType(dt);
          }
        }
        nativert::Constant value =
            isFloat ? nativert::Constant(0.0) : nativert::Constant(int64_t(0));
        if (!attr) {
          node->addAttribute({"value", std::move(value)});
        } else {
          const_cast<nativert::Attribute*>(attr)->value = std::move(value);
        }
      })
      .registerOp();
}

} // namespace torch::wave
