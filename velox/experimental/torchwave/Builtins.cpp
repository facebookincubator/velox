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

#include <fmt/format.h>

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

// Expands a single-block reduction (sum/max/min) into its multi-kernel form:
// tw.<prefix>_head (per-block partials) -> tw.<prefix>_final (reduce partials).
// 'prefix' names the reduction so one builder serves all of them.
nativert::Node* makeReduceVariant(
    NodeCP single,
    WaveGraph* waveGraph,
    const std::string& prefix) {
  auto* graph = variantNodeGraph(waveGraph);
  auto [outDtype, dtypeStr] = resolveOutDtype(single, waveGraph);

  auto* headNode = graph->createNode(
      "tw." + prefix + "_head", {{"input", single->inputs()[0].value}});
  headNode->addAttribute({"dtype", dtypeStr});
  auto* headOutput =
      newVariantTensorValue(headNode, waveGraph, prefix + "_blocks", outDtype);

  auto* finalNode =
      graph->createNode("tw." + prefix + "_final", {{"input", headOutput}});
  finalNode->addAttribute({"dtype", dtypeStr});
  copyOriginalOutputs(finalNode, single, waveGraph);
  return finalNode;
}

nativert::Node* makeSumVariant(NodeCP single, WaveGraph* waveGraph) {
  return makeReduceVariant(single, waveGraph, "sum");
}

nativert::Node* makeMaxVariant(NodeCP single, WaveGraph* waveGraph) {
  return makeReduceVariant(single, waveGraph, "max");
}

nativert::Node* makeMinVariant(NodeCP single, WaveGraph* waveGraph) {
  return makeReduceVariant(single, waveGraph, "min");
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

// Expands a scan into its single-pass form tw.<prefix>_1pass, whose extra
// output is the decoupled look-back state.
nativert::Node* makeScanSinglePassVariant(
    NodeCP single,
    WaveGraph* waveGraph,
    const std::string& prefix,
    bool copyDim) {
  auto* graph = variantNodeGraph(waveGraph);
  auto [outDtype, dtypeStr] = resolveOutDtype(single, waveGraph);
  auto* node = graph->createNode(
      "tw." + prefix + "_1pass", {{"input", single->inputs()[0].value}});
  node->addAttribute({"dtype", dtypeStr});
  // Only where the target declares 'dim' as a template attribute; copying it
  // to an op that does not would make it a constant parameter of a node the
  // constant map has no entry for.
  const auto* dimAttr = copyDim ? single->tryGetAttribute("dim") : nullptr;
  if (dimAttr) {
    node->addAttribute({dimAttr->name, std::get<int64_t>(dimAttr->value)});
  }
  copyOriginalOutputs(node, single, waveGraph);
  newVariantTensorValue(
      node, waveGraph, prefix + "_lookback", c10::ScalarType::Int);
  return node;
}

nativert::Node* makeCumsumSinglePassVariant(
    NodeCP single,
    WaveGraph* waveGraph) {
  return makeScanSinglePassVariant(
      single, waveGraph, "cumsum", /*copyDim=*/true);
}

nativert::Node* makeExclusiveSumSinglePassVariant(
    NodeCP single,
    WaveGraph* waveGraph) {
  return makeScanSinglePassVariant(
      single, waveGraph, "exclusive_sum", /*copyDim=*/false);
}

nativert::Node* makeMaskedSelectSinglePassVariant(
    NodeCP single,
    WaveGraph* waveGraph) {
  auto* graph = variantNodeGraph(waveGraph);
  auto* node = graph->createNode(
      "tw.masked_select_1pass",
      {{"input", single->inputs()[0].value},
       {"mask", single->inputs()[1].value}});
  copyOriginalOutputs(node, single, waveGraph);
  newVariantTensorValue(
      node, waveGraph, "masked_select_lookback", c10::ScalarType::Int);
  return node;
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

// Expands a single-block reduction into its cooperative-grid form
// tw.<prefix>_cg (one launch: per-block partials + inter-block barrier + reduce
// of partials).
nativert::Node* makeReduceCgVariant(
    NodeCP single,
    WaveGraph* waveGraph,
    const std::string& prefix) {
  auto* graph = variantNodeGraph(waveGraph);
  auto [outDtype, dtypeStr] = resolveOutDtype(single, waveGraph);
  auto* cgNode = graph->createNode(
      "tw." + prefix + "_cg", {{"input", single->inputs()[0].value}});
  cgNode->addAttribute({"dtype", dtypeStr});
  copyOriginalOutputs(cgNode, single, waveGraph);
  newVariantTensorValue(cgNode, waveGraph, prefix + "_partials", outDtype);
  return cgNode;
}

nativert::Node* makeSumCgVariant(NodeCP single, WaveGraph* waveGraph) {
  return makeReduceCgVariant(single, waveGraph, "sum");
}

nativert::Node* makeMaxCgVariant(NodeCP single, WaveGraph* waveGraph) {
  return makeReduceCgVariant(single, waveGraph, "max");
}

nativert::Node* makeMinCgVariant(NodeCP single, WaveGraph* waveGraph) {
  return makeReduceCgVariant(single, waveGraph, "min");
}

// Cooperative-grid variant of tw.bincount_final: same inputs/attributes/output,
// retargeted to the multi-block tw.bincount_final_cg (opBarrier between the
// output-clear and the atomic-add scatter).
nativert::Node* makeBincountFinalCgVariant(
    NodeCP single,
    WaveGraph* waveGraph) {
  auto* graph = variantNodeGraph(waveGraph);
  std::vector<nativert::NamedArgument> inputs;
  for (const auto& in : single->inputs()) {
    inputs.push_back({in.name, in.value});
  }
  auto* cgNode = graph->createNode("tw.bincount_final_cg", std::move(inputs));
  // nativert::Constant is move-only (it can hold a unique_ptr<Graph>), so copy
  // the specific host-side attributes bincount_final carries rather than the
  // whole variant.
  if (const auto* m = single->tryGetAttribute("minlength")) {
    if (std::holds_alternative<int64_t>(m->value)) {
      cgNode->addAttribute({"minlength", std::get<int64_t>(m->value)});
    }
  }
  if (const auto* d = single->tryGetAttribute("dtype")) {
    if (std::holds_alternative<c10::ScalarType>(d->value)) {
      cgNode->addAttribute({"dtype", std::get<c10::ScalarType>(d->value)});
    }
  }
  copyOriginalOutputs(cgNode, single, waveGraph);
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

// If 'value' is produced by a constant-fill factory (zeros/ones/full and their
// _like / new_ variants), returns the scalar fill it produces, for use as the
// out-of-range default of tw.index_elt_one_default. Returns nullopt otherwise.
std::optional<nativert::Constant> factoryDefault(nativert::Value* value) {
  auto* producer = value ? value->producer() : nullptr;
  if (!producer) {
    return std::nullopt;
  }
  const auto target = producer->target();
  if (target == "torch.ops.aten.zeros.default" ||
      target == "torch.ops.aten.zeros_like.default" ||
      target == "torch.ops.aten.new_zeros.default") {
    return nativert::Constant{int64_t{0}};
  }
  if (target == "torch.ops.aten.ones.default" ||
      target == "torch.ops.aten.ones_like.default" ||
      target == "torch.ops.aten.new_ones.default") {
    return nativert::Constant{int64_t{1}};
  }
  if (target == "torch.ops.aten.full.default" ||
      target == "torch.ops.aten.full_like.default" ||
      target == "torch.ops.aten.new_full.default") {
    const auto* fill = producer->tryGetAttribute("fill_value");
    if (fill) {
      if (std::holds_alternative<int64_t>(fill->value)) {
        return nativert::Constant{std::get<int64_t>(fill->value)};
      }
      if (std::holds_alternative<double>(fill->value)) {
        return nativert::Constant{std::get<double>(fill->value)};
      }
      if (std::holds_alternative<bool>(fill->value)) {
        return nativert::Constant{std::get<bool>(fill->value)};
      }
    }
  }
  return std::nullopt;
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
  return {{.rank = 1, .contiguity = Contiguity::kContiguous}};
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

// Canonicalizes searchsorted/bucketize's attributes for the fused kernel:
//  - Folds 'right' and searchsorted's 'side' into a single bool 'right'
//    attribute (matches ATen: side="right" means right=true;
//    searchsorted_pre_check forbids the contradictory (right=true,
//    side="left")). Emitted as a runtime *param<bool>.
//  - Ensures a bool 'out_int32' attribute exists (default false). It is a
//    compile-time template argument (see templateAttrs) that selects the output
//    integer width; making it a template arg also keeps the int32 and int64
//    variants as distinct kernels, which otherwise dedup (they share dtypes)
//    and store the wrong width.
// Both attributes are always present afterwards so codegen never sees a missing
// template/runtime attribute.
void normalizeSearchAttrs(nativert::Node* node) {
  auto setAttr = [node](std::string_view name, nativert::Constant value) {
    for (auto& attr : node->attributes()) {
      if (attr.name == name) {
        const_cast<nativert::Attribute&>(attr).value = std::move(value);
        return;
      }
    }
    node->addAttribute({std::string(name), std::move(value)});
  };

  bool right = false;
  if (const auto* attr = node->tryGetAttribute("right")) {
    if (std::holds_alternative<bool>(attr->value)) {
      right = std::get<bool>(attr->value);
    } else if (std::holds_alternative<int64_t>(attr->value)) {
      right = std::get<int64_t>(attr->value) != 0;
    }
  }
  if (const auto* side = node->tryGetAttribute("side")) {
    if (std::holds_alternative<std::string>(side->value) &&
        std::get<std::string>(side->value) == "right") {
      right = true;
    }
  }
  setAttr("right", right);

  bool outInt32 = false;
  if (const auto* attr = node->tryGetAttribute("out_int32")) {
    if (std::holds_alternative<bool>(attr->value)) {
      outInt32 = std::get<bool>(attr->value);
    } else if (std::holds_alternative<int64_t>(attr->value)) {
      outInt32 = std::get<int64_t>(attr->value) != 0;
    }
  }
  // Stored as int (0/1), not bool: it is emitted as a template argument and a
  // bool constant stringifies to "True"/"False", which is not valid C++.
  setAttr("out_int32", static_cast<int64_t>(outInt32 ? 1 : 0));
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

// Resolves 'value' to the buffer it will actually read once clone elision has
// run. Follows storage aliasing like viewStorageBase, and additionally looks
// through aten.clone: a clone taken only to snapshot a value for reading is
// elidable, and once elision drops it the clone's output IS its source's
// buffer. Comparing post-elision roots errs in the safe direction -- a clone
// that survives costs one needless materialization, a clone that is dropped
// after we assumed it would not would be a wrong answer.
ValueCP elidableStorageRoot(ValueCP value) {
  while (value != nullptr) {
    auto* base = viewStorageBase(value);
    auto* producer = base->producer();
    if (!producer || producer->target() != "torch.ops.aten.clone.default" ||
        producer->inputs().empty()) {
      return base;
    }
    value = producer->inputs()[0].value;
  }
  return value;
}

// True if an aten.copy node may read the buffer its result is written back
// into. The copy's result always lands in self's storage: the enclosing
// slice_scatter / select_scatter writes it there, in place once clone elision
// drops the defensive copy of self. So self and src sharing a storage root
// means the read and the write can overlap -- and at a shift, since they are
// different views of it. Identical views are exempt: self[i] = self[i] needs no
// ordering.
bool copyMayOverlap(NodeCP node) {
  auto* self = inputAt(node, 0);
  auto* src = inputAt(node, 1);
  if (!self || !src || self == src) {
    return false;
  }
  auto* selfRoot = elidableStorageRoot(self);
  auto* srcRoot = elidableStorageRoot(src);
  // A root can come back null when a clone chain has an unset source. Two
  // nulls comparing equal would be the right answer only by accident, and one
  // null would report "no overlap" on no evidence. Report possible overlap
  // instead: that is the same safe direction the root comparison itself errs
  // in -- a needless materialization costs one step, a missed hazard is a
  // wrong answer.
  if (!selfRoot || !srcRoot) {
    return true;
  }
  return selfRoot == srcRoot;
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
    return {
        {.rank = static_cast<int8_t>(size.size()),
         .contiguity = Contiguity::kContiguous}};
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

// A metadata getter (sym_size / sym_numel) is worth running on the host exactly
// when it would otherwise be an expression of its own. The host already holds
// every tensor, so reading a size there is a field access; a kernel op has to
// pass the tensor's metadata in, compute one scalar, and read it back out. When
// the getter instead feeds a single fusable consumer it is a subexpression of
// that consumer's kernel, where the value is wanted on the device anyway, so it
// stays fused.
//
// Consulting the consumer's isStandalone cannot recurse back here: a getter
// takes a tensor and yields a scalar, so a getter is never the user of a
// getter.
//
// A getter left fused when it is alone costs a whole thread block that reads
// one field and exits. That block is charged against the launch's slowest
// block, so a handful of them drag a step's reported balance down and the
// packer has nothing it can do about it -- there is no block count at which a
// no-op op balances.
bool metadataGetterIsAlone(NodeCP node, const ValueTypes& types) {
  if (!WaveConfig::get().metadataGetterStandalone) {
    return false;
  }
  if (node->outputs().empty() || node->outputs()[0] == nullptr) {
    return false;
  }
  ValueCP out = node->outputs()[0];
  if (types.graphOutput(out)) {
    return true;
  }
  const auto& users = out->users();
  if (users.size() != 1) {
    // Unused, or shared -- either way it is materialized on its own.
    return true;
  }
  const Metadata* userMeta = Registry::metadata(users[0]->target());
  return userMeta == nullptr || userMeta->isStandalone(users[0], types);
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

// Elements a thread of a single-pass scan handles per tile. Mirrors
// kScanItemsPerThread in Scan.cuh; the two must change together, since the
// buffer sized here is indexed by a tile number the device computes.
constexpr int32_t kSinglePassItemsPerThread = 8;

// Shape of the decoupled look-back state of a single-pass scan: an int32 tensor
// holding a status flag, an aggregate and a prefix per tile. Mirrors
// LookbackState::numWords in Scan.cuh, but sized for the widest accumulator
// rather than the op's own, because the accumulator type is not resolved here
// and the device lays the three arrays out from its own sizeof. Over-sizing
// just leaves a tail of the buffer unused.
std::vector<std::vector<Dim>> lookbackStateShape(
    NodeCP node,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map,
    NodeCP /*originalFormalNode*/,
    const NodeMap& /*nodeMap*/) {
  constexpr int64_t kWidestAccumulatorWords = 2;
  auto tensor = paramTensor(node->inputs()[0].value, frame, map);
  const int64_t tileSize = static_cast<int64_t>(WaveConfig::get().blockSize) *
      kSinglePassItemsPerThread;
  // At least one tile: an empty input processes none, but a zero-length buffer
  // would make the reserve produce a tensor with null storage.
  const int64_t numTiles =
      std::max<int64_t>(1, (tensor.numel() + tileSize - 1) / tileSize);
  const int64_t flagWords = (numTiles + 1) & ~int64_t(1);
  return {
      {static_cast<Dim>(flagWords + 2 * numTiles * kWidestAccumulatorWords)}};
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
                 .contiguity = contiguousOutput ? Contiguity::kContiguous
                                                : Contiguity::kUnknown}};
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
            constraint.contiguity = types.contiguity(inputAt(node, 0));
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

// Registers a full-tensor reduction ('prefix' in {sum, max, min}) across all
// three execution forms, mirroring the aten.sum.default family: the
// single-block op (also the fused register form) plus its tw.<prefix>_head /
// _final / _cg intrinsics reached via the multi-kernel and cooperative-grid
// variants. The device functions (tw_<prefix>, tw_<prefix>_head,
// tw_<prefix>_tensor, tw_<prefix>_cg) share one templated implementation in
// Scan.cuh, differing only by reduction Op. 'normalize' sets the output dtype
// (sum promotes ints to Long; max/min keep the input dtype exactly).
void registerReduction(
    const std::string& scanHeader,
    const std::string& atenName,
    const std::string& prefix,
    void (*normalize)(nativert::Node*, const ValueTypes&),
    nativert::Node* (*variantFn)(NodeCP, WaveGraph*),
    nativert::Node* (*cgVariantFn)(NodeCP, WaveGraph*)) {
  auto scalarShape = [](NodeCP,
                        nativert::ExecutionFrame&,
                        const FormalToActual&,
                        NodeCP,
                        const NodeMap&) -> std::vector<std::vector<Dim>> {
    return {{}};
  };
  auto tensorInOut = [](const std::string& name) {
    return std::make_unique<c10::FunctionSchema>(
        name,
        "",
        std::vector<c10::Argument>{
            c10::Argument("input", c10::TensorType::get())},
        std::vector<c10::Argument>{
            c10::Argument("output", c10::TensorType::get())});
  };

  // Single-block op; also the fused register form. Placement expands it to the
  // multi-kernel or cooperative-grid variant for large inputs.
  MetadataBuilder(atenName)
      .sizeOrdinal({0})
      .sizeShortcut(SizeShortcut::kMax)
      .hasBarrier()
      .singleBlockIfFused()
      .argumentMeta({{.isRegister = true}})
      .returnMeta({{.isRegister = false, .reserveShape = scalarShape}})
      .normalize(normalize)
      .makeMultiKernelVariant(variantFn)
      .cgVariant(cgVariantFn)
      .headerFile(scanHeader)
      .deviceFunc("tw_" + prefix)
      .sharedDecls({{"Int32X32", "warpSums"}})
      .dynamicSharedDecls({{Metadata::kTypeFromDtype, "counter"}})
      .typeTemplateParams({0})
      .hasDtypeTemplateParam()
      .hasBlockSizeTemplateParam()
      .alwaysSingleBlock()
      .registerOp();

  // Multi-kernel stage 1: per-block partials.
  MetadataBuilder(tensorInOut("tw." + prefix + "_head"))
      .sizeOrdinal({0})
      .hasBarrier()
      .returnMeta({{.isRegister = false, .reserveShape = numBlocksShape}})
      .headerFile(scanHeader)
      .deviceFunc("tw_" + prefix + "_head")
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

  // Multi-kernel stage 2: reduce the partials to a scalar.
  MetadataBuilder(tensorInOut("tw." + prefix + "_final"))
      .sizeOrdinal({0})
      .hasBarrier()
      .returnMeta({{.isRegister = false, .reserveShape = scalarShape}})
      .inputFromPreviousKernel(0)
      .headerFile(scanHeader)
      .deviceFunc("tw_" + prefix + "_tensor")
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

  // Cooperative-grid variant: head + reduce-of-partials in one launch.
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw." + prefix + "_cg",
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
          {{.isRegister = false, .reserveShape = scalarShape},
           {.isRegister = false, .reserveShape = numBlocksShape}})
      .normalize(normalize)
      .headerFile(scanHeader)
      .deviceFunc("tw_" + prefix + "_cg")
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
}

// Inserts clone(self) before an in-place-writer node and rewires the node's
// self (arg 0, == mutatesArg) to read the clone, carrying self's full
// TensorMeta (sizes) onto the clone output. A later clone-elision drops the
// clone when self is a dead intermediate -- turning the write in place -- and
// keeps it as a defensive copy when self is still live (a graph output, a
// weight/input, or otherwise externally owned). Mirrors the index_put ->
// index_put_ rewrite. No-op (returns true) if self already comes from a
// single-user clone; returns false (leaving the node functional/standalone)
// if self's type is not resolved at optimization time.
bool insertSelfCloneForInPlace(
    NodeCP node,
    ValueTypes& types,
    WaveGraph& waveGraph,
    const char* cloneName) {
  auto* mutableNode = const_cast<nativert::Node*>(node);
  auto* selfVal = inputAt(node, 0);
  auto* selfProducer = selfVal->producer();
  if (selfProducer &&
      selfProducer->target() == "torch.ops.aten.clone.default" &&
      selfVal->users().size() == 1) {
    return true;
  }
  auto selfId = selfVal->id();
  if (selfId >= static_cast<int>(types.types.size()) || !types.types[selfId]) {
    return false;
  }
  auto selfDtype = types.types[selfId]->dtype();
  auto* graph = waveGraph.graph();
  auto* cloneNode =
      graph->createNode("torch.ops.aten.clone.default", {{"self", selfVal}});
  graph->insertBefore(cloneNode, mutableNode);
  auto* cloneOutput = waveGraph.newTensorValue(cloneNode, cloneName, selfDtype);
  // newTensorValue records only the dtype (dim()==0); clone preserves self's
  // shape, so carry self's full type onto the clone output -- otherwise the
  // in-place op cannot read the rank/sizes (see index_put_inserted_clone_type).
  auto cloneId = cloneOutput->id();
  if (cloneId >= 0) {
    if (static_cast<size_t>(cloneId) >= types.types.size()) {
      types.types.resize(cloneId + 1, nullptr);
    }
    types.types[cloneId] = types.types[selfId];
  }
  mutableNode->inputs()[0].value = cloneOutput;
  // Maintain Value::users() so last-use / reuse / clone-elision (which key off
  // users()) see the node reading the clone, not the original self.
  cloneOutput->addUser(mutableNode);
  bool stillReadsSelf = false;
  for (const auto& in : mutableNode->inputs()) {
    if (in.value == selfVal) {
      stillReadsSelf = true;
      break;
    }
  }
  if (!stillReadsSelf) {
    const_cast<nativert::Value*>(selfVal)->eraseUser(mutableNode);
  }
  return true;
}

// Rewrites a functional aten.scatter.src / aten.scatter_add.default to the
// fused in-place tw.scatter / tw.scatter_add. Fuses only the supported cases: a
// statically known dim in {0, 1}, an int or long index tensor with the same
// shape as src, and -- for scatter_add, which accumulates with an atomic add --
// a float/double/int32/int64 self dtype. Inserts a clone of self and
// retargets; clone-elision drops the clone when self is dead and keeps it when
// src shares a base with self (accumulating in place would read partially
// updated values). Returns false to leave the node as the eager standalone op.
bool tryFuseScatter(
    NodeCP node,
    ValueTypes& types,
    WaveGraph& waveGraph,
    bool accumulate) {
  int64_t scatterDim = constIntArg(node, "dim").value_or(0);
  if (hasDynamicArg(node, "dim") || (scatterDim != 0 && scatterDim != 1)) {
    return false;
  }
  auto tensorMeta = [&](ValueCP value) -> const nativert::TensorMeta* {
    auto id = value->id();
    if (id < 0 || static_cast<size_t>(id) >= types.types.size()) {
      return nullptr;
    }
    return types.types[id];
  };
  const auto* indexMeta = tensorMeta(inputAt(node, 1));
  const auto* srcMeta = tensorMeta(inputAt(node, 2));
  if (!indexMeta || !srcMeta) {
    return false;
  }
  // Rank comes from ValueTypes, not TensorMeta::dim(): a value produced by an
  // earlier wave rewrite carries a dtype-only TensorMeta (no sizes, so dim() is
  // 0 and hasSymbolicShape() is false) while its rank is still tracked in the
  // constraints. Reading dim() here would reject such an index outright.
  auto indexRank = types.rank(inputAt(node, 1));
  auto srcRank = types.rank(inputAt(node, 2));
  if (indexRank < 0 || indexRank != srcRank) {
    return false;
  }
  // Only the ranks have to agree at compile time: scatterDestOffset decomposes
  // the src-sized loop counter against index's runtime dims, so the extents are
  // never baked into the kernel. Compare extents only when both metas carry
  // real static sizes; a symbolic extent cannot be compared at all
  // (TensorMeta::sizes() throws on one and nativert exposes no symbolic sizes),
  // so equal shapes are taken on the aten contract, as the slice/select_scatter
  // rewrites already do.
  if (!indexMeta->hasSymbolicShape() && !srcMeta->hasSymbolicShape() &&
      indexMeta->dim() == indexRank && srcMeta->dim() == srcRank &&
      indexMeta->sizes() != srcMeta->sizes()) {
    return false;
  }
  auto indexDtype = indexMeta->dtype();
  if (indexDtype != c10::ScalarType::Long &&
      indexDtype != c10::ScalarType::Int) {
    return false;
  }
  if (accumulate) {
    const auto* selfMeta = tensorMeta(inputAt(node, 0));
    if (!selfMeta) {
      return false;
    }
    auto dtype = selfMeta->dtype();
    if (dtype != c10::ScalarType::Float && dtype != c10::ScalarType::Double &&
        dtype != c10::ScalarType::Int && dtype != c10::ScalarType::Long) {
      return false;
    }
  }
  if (!insertSelfCloneForInPlace(
          node,
          types,
          waveGraph,
          accumulate ? "scatter_add_clone" : "scatter_clone")) {
    return false;
  }
  const_cast<nativert::Node*>(node)->setTarget(
      accumulate ? "tw.scatter_add" : "tw.scatter");
  return true;
}

} // namespace

void registerBuiltins() {
  // With WaveConfig::singlePass the two scans and the stream compaction get one
  // implementation, the decoupled look-back form, in place of the multi-kernel
  // and the cooperative-grid expansions. It replaces both so the choice does
  // not also depend on whether the step happens to run a cooperative grid; the
  // op declares a barrier either way, which is what gives it the resident grid
  // its look-back needs.
  const bool singlePass = WaveConfig::get().singlePass;

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
  MetadataBuilder("torch.ops.aten.__and__.Scalar")
      .elementwiseFunc("__bitwise_and")
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

  // copy(self, src, non_blocking) -> Tensor: the functionalized form of copy_.
  // The result carries self's shape, dtype, device and layout but src's values,
  // so it is `_to_copy(src.expand_as(self), dtype=self.dtype)` -- one register
  // input and a cast. self contributes metadata only: it is skipped during
  // argument iteration (ignoreAttrs) and read through rankArgument /
  // reserveShape, the same way full_like reads the tensor it takes its shape
  // from. Skipping it matters beyond the saved load: reading the destination
  // would make the copy depend on its contents and force an ordering edge that
  // the op does not actually need.
  //
  // The elementwise loop is sized by the output (self's shape) and reads src
  // per lane, so a size-1 dim or an aten.expand source broadcasts through the
  // ordinary stride-0 leaf path with no separate expand step.
  //
  // A copy whose src shares storage with self is retargeted to tw.copy_out --
  // see copyMayOverlap.
  {
    // self's shape, for both the register and the materializing form.
    auto selfShape =
        [](NodeCP node,
           nativert::ExecutionFrame& frame,
           const FormalToActual& map,
           NodeCP /*originalFormalNode*/,
           const NodeMap& /*nodeMap*/) -> std::vector<std::vector<Dim>> {
      return elementwiseInputShape(node, frame, map, 0);
    };
    // The result has self's rank but is a fresh dense buffer, not self's
    // (possibly strided) layout.
    auto selfRankContiguous =
        [](NodeCP node,
           const ValueTypes& types) -> std::vector<ValueConstraint> {
      return {
          {.rank = types.rank(inputAt(node, 0)),
           .contiguity = Contiguity::kContiguous}};
    };
    // resolveDtypeFromInputExact writes self's dtype onto the node, so the cast
    // is driven by the same "dtype" attribute _to_copy and to.dtype use.
    auto castToDtype =
        [](std::stringstream& ss, NodeCP node, std::vector<std::string> args) {
          TORCH_CHECK(!args.empty(), "copy requires a src input");
          const auto* dtypeAttr = node->tryGetAttribute("dtype");
          if (!dtypeAttr ||
              std::holds_alternative<nativert::None>(dtypeAttr->value)) {
            ss << args[0];
            return;
          }
          ss << "static_cast<" << cudaTypeFromDtype(*dtypeAttr) << ">("
             << args[0] << ")";
        };

    MetadataBuilder("torch.ops.aten.copy.default")
        .elementwiseFunc("__to")
        .numArgs(1)
        .ignoreAttrs({"self", "non_blocking"})
        .argumentMeta({{}, {.isRegister = true}, {}})
        .returnMeta(
            {{.isRegister = true,
              .reserveShape = selfShape,
              .shapeFromInput = 0}})
        .normalize(resolveDtypeFromInputExact)
        .rankArgument(0)
        .outputConstraints(selfRankContiguous)
        .generateCall(castToDtype)
        .layoutAgnostic()
        .maybeReplace(
            [](NodeCP node, ValueTypes& /*types*/, WaveGraph& /*waveGraph*/)
                -> std::vector<std::pair<ValueCP, ValueCP>> {
              // Retargeted in place rather than returned as a replacement:
              // the node keeps its identity, inputs and users, and only the op
              // it resolves to changes, so there is nothing for the caller to
              // rewire -- an empty substitution list is the accurate answer.
              // Guarded on the target so a repeat call is a no-op: once the
              // node is tw.copy_out this hook is not reached for it again.
              if (node->target() == "torch.ops.aten.copy.default" &&
                  copyMayOverlap(node)) {
                const_cast<nativert::Node*>(node)->setTarget("tw.copy_out");
              }
              return {};
            })
        .registerOp();

    // tw.copy_out: aten.copy with its result in memory instead of a register.
    // Used when the copy reads the buffer the enclosing scatter writes back
    // into. A register result is inlined into the scatter's write, so the read
    // of lane i and the write of lane j land in one loop with no ordering
    // between them; when the two views overlap at a shift that is a data race.
    // Materializing splits the copy into its own elementwise expression, and
    // because the consumer then reads a value a producer wrote earlier in the
    // same kernel, callNeedsBarrier emits the barrier that orders the whole
    // read before the write.
    MetadataBuilder(
        std::make_unique<c10::FunctionSchema>(
            "tw.copy_out",
            "",
            std::vector<c10::Argument>{
                c10::Argument("self", c10::TensorType::get()),
                c10::Argument("src", c10::TensorType::get())},
            std::vector<c10::Argument>{
                c10::Argument("output", c10::TensorType::get())}))
        .elementwiseFunc("__copy_out")
        .numArgs(1)
        .hasIdxArg()
        .hasOutputArg()
        .hasDtypeTemplateParam()
        .ignoreAttrs({"self", "non_blocking"})
        .argumentMeta({{}, {.isRegister = true}})
        .returnMeta(
            {{.isRegister = false,
              .reserveShape = selfShape,
              .shapeFromInput = 0}})
        .normalize(resolveDtypeFromInputExact)
        .rankArgument(0)
        .outputConstraints(selfRankContiguous)
        .layoutAgnostic()
        .registerOp();
  }

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
      // The optimizer's elementwise fallback derives an output rank as the max
      // over the input ranks; this op's only input is a rankless scalar, so the
      // 0-d rank has to be stated here or the output is left rank-unknown.
      .outputConstraints([](NodeCP /*node*/, const ValueTypes& /*types*/) {
        return std::vector<ValueConstraint>{
            {.rank = 0, .contiguity = Contiguity::kContiguous}};
      })
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

  // Full: size-shaped tensor filled with a scalar. Like ones, but the fill
  // value arrives as the "fill_value" scalar attribute param consumed by
  // __full ("size" is the shape attribute, so it is skipped as an argument).
  MetadataBuilder("torch.ops.aten.full.default")
      .elementwiseFunc("__full")
      .numArgs(1)
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
            },
            .shapeFromInput = 0}})
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
            },
            .shapeFromInput = 0}})
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
      .metadataOnly()
      .isStandaloneFunc(metadataGetterIsAlone)
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
      .metadataOnly()
      .isStandaloneFunc(metadataGetterIsAlone)
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
    // Use the propagated constraint rank, not the archive TensorMeta dim(): the
    // latter can be a stale rank-0 placeholder for data-dependent sources (the
    // archive under-specifies their shape), while the constraint rank is
    // inferred input-first and is correct by the time this rewrite runs.
    int rank = types.rank(tensor);
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
    return {{.rank = types.rank(self), .contiguity = types.contiguity(self)}};
  };

  // Clone: eliminate unless a user mutates the output in place.
  MetadataBuilder("torch.ops.aten.clone.default")
      .sizeOrdinal({0})
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            // clone makes a fresh, densely-laid-out copy regardless of the
            // input's layout.
            return {
                {.rank = types.rank(inputAt(node, 0)),
                 .contiguity = Contiguity::kContiguous}};
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
              // A consumer that writes this clone in place via mutatesArg (a
              // fused scatter like tw.slice_scatter / tw.select_scatter, whose
              // schema carries no Tensor(a!) alias annotation for
              // isInPlaceMutation to see) needs the clone kept: it is both the
              // write buffer and a materialization barrier for self.
              // Reuse-gated clone-elision drops it later when self is a dead
              // intermediate.
              const auto* userMeta = Registry::metadata(user->target());
              if (userMeta != nullptr && userMeta->mutatesArg.has_value()) {
                auto ord = *userMeta->mutatesArg;
                if (ord >= 0 &&
                    static_cast<size_t>(ord) < user->inputs().size() &&
                    user->inputs()[ord].value == outputs[0]) {
                  return {};
                }
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

  // masked_select: gathers the elements where the boolean mask is true into a
  // fresh 1-D tensor (data-dependent length). Runs eager (standalone); it is
  // the target of x[bool_mask] on a 1-D source (see the index.Tensor rewrite
  // below).
  MetadataBuilder("torch.ops.aten.masked_select.default")
      .sizeOrdinal({0})
      .isStandalone()
      .outputConstraints(
          [](NodeCP /*node*/,
             const ValueTypes& /*types*/) -> std::vector<ValueConstraint> {
            return {{.rank = 1, .contiguity = Contiguity::kContiguous}};
          })
      .registerOp();

  // index.Tensor: gather elements by index.
  //
  // A single 1-D integer index that selects one dimension of a rank>1 source
  // (x[idx], x[:, idx], x[:, :, idx]) is exactly index_select along that
  // dimension, so rewrite it to the fused tw.index_select. When every dimension
  // is indexed by an integer tensor (x[i, j, k] on a rank-3 source) the result
  // is a scalar-per-index gather handled by tw.index_elt_{one,two,three}.
  // Everything else -- several but not all dimensions indexed (x[i, :, k]), a
  // multi-dimensional index, or a non-integer index -- falls through to the
  // eager standalone op.
  MetadataBuilder("torch.ops.aten.index.Tensor")
      .sizeOrdinal({1, 0})
      .isStandalone()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            // Advanced indexing materializes a fresh dense tensor, so the eager
            // output is contiguous. Its rank follows the numpy/torch rule:
            // broadcastRank(index tensors) + (sourceRank - numIndexed), since
            // each integer index consumes one source dim and the indices
            // broadcast together, while None gaps and trailing dims are kept.
            // Rank is left unknown when the source rank is unavailable or a
            // non-integer (boolean-mask) index makes the count data-dependent;
            // the output is contiguous either way.
            auto* self = inputAt(node, 0);
            int8_t sourceRank = types.rank(self);
            auto* indices =
                node->inputs().size() > 1 ? node->inputs()[1].value : nullptr;
            auto* listPack = indices ? indices->producer() : nullptr;
            if (sourceRank < 0 || !listPack ||
                listPack->target() != "prim.ListPack") {
              return {{.rank = -1, .contiguity = Contiguity::kContiguous}};
            }
            int32_t numIndexed = 0;
            int32_t broadcastRank = 0;
            for (const auto& entry : listPack->inputs()) {
              auto* indexValue = entry.value;
              if (!indexValue ||
                  indexValue->type().kind() == nativert::Type::Kind::None) {
                continue;
              }
              auto indexValueId = indexValue->id();
              bool intIndex = indexValueId >= 0 &&
                  static_cast<size_t>(indexValueId) < types.types.size() &&
                  types.types[indexValueId] &&
                  (types.types[indexValueId]->dtype() == c10::ScalarType::Int ||
                   types.types[indexValueId]->dtype() == c10::ScalarType::Long);
              if (!intIndex) {
                return {{.rank = -1, .contiguity = Contiguity::kContiguous}};
              }
              ++numIndexed;
              broadcastRank = std::max(
                  broadcastRank,
                  static_cast<int32_t>(types.types[indexValueId]->dim()));
            }
            int32_t rank = numIndexed == 0 ? sourceRank
                                           : broadcastRank +
                    (static_cast<int32_t>(sourceRank) - numIndexed);
            return {
                {.rank = static_cast<int8_t>(rank),
                 .contiguity = Contiguity::kContiguous}};
          })
      .maybeReplace(
          [&isIntegerIndices](
              NodeCP node, ValueTypes& types, WaveGraph& waveGraph)
              -> std::vector<std::pair<ValueCP, ValueCP>> {
            auto* self = inputAt(node, 0);
            auto* indices = inputAt(node, 1);
            auto* listPack = indices ? indices->producer() : nullptr;
            if (self && listPack && listPack->target() == "prim.ListPack") {
              auto selfId = self->id();
              // selfOk gates access to the source dtype (needed to type the new
              // index_select value); the rank comes from the propagated
              // constraint, which is correct even when the archive TensorMeta
              // carries a stale rank-0 placeholder.
              bool selfOk = selfId >= 0 &&
                  static_cast<size_t>(selfId) < types.types.size() &&
                  types.types[selfId];
              int32_t sourceRank = types.rank(self);
              // Scan the index list for exactly one 1-D integer index tensor,
              // with every other entry a None gap.
              const auto& idxInputs = listPack->inputs();
              int32_t numIndices = 0;
              int32_t dim = -1;
              nativert::Value* idxVal = nullptr;
              bool convertible = true;
              for (size_t i = 0; i < idxInputs.size(); ++i) {
                auto* indexValue = idxInputs[i].value;
                if (!indexValue ||
                    indexValue->type().kind() == nativert::Type::Kind::None) {
                  continue;
                }
                auto indexValueId = indexValue->id();
                bool is1dIntTensor =
                    indexValue->type().kind() == nativert::Type::Kind::Tensor &&
                    types.rank(indexValue) == 1 && indexValueId >= 0 &&
                    static_cast<size_t>(indexValueId) < types.types.size() &&
                    types.types[indexValueId] &&
                    (types.types[indexValueId]->dtype() ==
                         c10::ScalarType::Int ||
                     types.types[indexValueId]->dtype() ==
                         c10::ScalarType::Long);
                if (!is1dIntTensor) {
                  convertible = false;
                  break;
                }
                ++numIndices;
                dim = static_cast<int32_t>(i);
                idxVal = indexValue;
              }
              if (selfOk && convertible && numIndices == 1 && sourceRank > 1 &&
                  sourceRank <= 3) {
                auto* graph = waveGraph.graph();
                auto* newNode = graph->createNode(
                    "tw.index_select", {{"self", self}, {"index", idxVal}});
                newNode->addAttribute({"dim", static_cast<int64_t>(dim)});
                graph->insertBefore(newNode, const_cast<nativert::Node*>(node));
                auto* newOutput = waveGraph.newTensorValue(
                    newNode, "index_select", types.types[selfId]->dtype());
                return {{node->outputs()[0], newOutput}};
              }
              // x[bool_mask] with a 1-D source and a single 1-D boolean index
              // is masked_select: a data-dependent 1-D gather of the positions
              // where the mask is true. Retarget to the standalone
              // masked_select op (a boolean mask is not a fusable gather).
              if (selfOk && sourceRank == 1) {
                nativert::Value* maskVal = nullptr;
                int32_t numMask = 0;
                for (const auto& idxEntry : idxInputs) {
                  auto* maskValue = idxEntry.value;
                  if (!maskValue ||
                      maskValue->type().kind() == nativert::Type::Kind::None) {
                    continue;
                  }
                  ++numMask;
                  auto maskValueId = maskValue->id();
                  if (maskValue->type().kind() ==
                          nativert::Type::Kind::Tensor &&
                      types.rank(maskValue) == 1 && maskValueId >= 0 &&
                      static_cast<size_t>(maskValueId) < types.types.size() &&
                      types.types[maskValueId] &&
                      types.types[maskValueId]->dtype() ==
                          c10::ScalarType::Bool) {
                    maskVal = maskValue;
                  }
                }
                if (numMask == 1 && maskVal) {
                  auto* graph = waveGraph.graph();
                  auto* newNode = graph->createNode(
                      "torch.ops.aten.masked_select.default",
                      {{"self", self}, {"mask", maskVal}});
                  graph->insertBefore(
                      newNode, const_cast<nativert::Node*>(node));
                  auto* newOutput = waveGraph.newTensorValue(
                      newNode, "masked_select", types.types[selfId]->dtype());
                  return {{node->outputs()[0], newOutput}};
                }
              }
            }
            auto n = isIntegerIndices(types, self, indices);
            // A single integer index over cat(var, factory) with exactly two
            // cat operands, where the second is a constant-fill factory
            // (zeros/ones/ full), is an index of 'var' with that constant as
            // the out-of-range default. Rewrite to the fused
            // tw.index_elt_one_default reading 'var' directly, so the cat and
            // its padding never materialize.
            if (n == 1) {
              auto* catNode = self->producer();
              if (catNode &&
                  catNode->target() == "torch.ops.aten.cat.default" &&
                  !catNode->inputs().empty()) {
                auto* catList = catNode->inputs()[0].value;
                auto* catListPack = catList ? catList->producer() : nullptr;
                if (catListPack && catListPack->target() == "prim.ListPack" &&
                    catListPack->inputs().size() == 2) {
                  auto* dataArg = catListPack->inputs()[0].value;
                  auto* padArg = catListPack->inputs()[1].value;
                  auto deflt = factoryDefault(padArg);
                  auto dataId = dataArg ? dataArg->id() : -1;
                  bool dataOk = dataId >= 0 &&
                      static_cast<size_t>(dataId) < types.types.size() &&
                      types.types[dataId];
                  if (deflt && dataOk) {
                    auto* graph = waveGraph.graph();
                    auto* newNode = graph->createNode(
                        "tw.index_elt_one_default",
                        {{"self", dataArg}, {"indices", indices}});
                    newNode->addAttribute({"deflt", std::move(*deflt)});
                    graph->insertBefore(
                        newNode, const_cast<nativert::Node*>(node));
                    auto* newOutput = waveGraph.newTensorValue(
                        newNode,
                        "index_elt_one_default",
                        types.types[dataId]->dtype());
                    return {{node->outputs()[0], newOutput}};
                  }
                }
              }
            }
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

    // tw.index_elt_one_default: like tw.index_elt_one but returns a scalar
    // default for an out-of-range index instead of erroring. The default's type
    // is deduced as a second type template parameter after the source element
    // type, so a scalar constant of a possibly different type converts to the
    // element type. No BlockInfo: there is no error path.
    MetadataBuilder(
        std::make_unique<c10::FunctionSchema>(
            "tw.index_elt_one_default",
            "",
            std::vector<c10::Argument>{
                c10::Argument("self", c10::TensorType::get()),
                c10::Argument("indices", c10::ListType::ofTensors()),
                c10::Argument("deflt", c10::NumberType::get())},
            std::vector<c10::Argument>{
                c10::Argument("output", c10::TensorType::get())}))
        .elementwiseFunc("__index_elt_one_default")
        .argumentMeta(
            {{.isRegister = false, .wholeTensor = true, .randomAccess = true},
             {.isRegister = true},
             {.isRegister = true}})
        .returnMeta({{.isRegister = true, .reserveShape = indexEltReserve}})
        .typeTemplateParams({0})
        .registerOp();
  }

  // tw.index_select: fused elementwise index_select along 'dim'. The rewrite
  // below retargets aten.index_select (which names its operands self / dim /
  // index) onto this op; forArguments binds by name, so the schema lists the
  // two tensor operands before the scalar 'dim'. That order matters: the
  // wholeTensor classification in codegen indexes argumentMeta by node input
  // position, and a constant 'dim' is an attribute (not an input), so putting
  // 'dim' last keeps self and index aligned with their argumentMeta entries.
  // The loop iterates every element of the enclosing expression's output;
  // __index_select maps each output element to a source element, taking the
  // coordinate along 'dim' from 'index'. 'self' and 'index' are whole tensors
  // read at computed offsets (randomAccess, forcing materialization before this
  // op). The output shape is self's shape with 'dim' resized to the index
  // length, computed at launch by indexSelectReserve. hasOutputArg passes the
  // enclosing expression's output tensor so the device function knows the shape
  // it iterates over, distinct from index_select's own (broadcast) shape.
  {
    auto indexSelectReserve =
        [](NodeCP node,
           nativert::ExecutionFrame& frame,
           const FormalToActual& map,
           NodeCP /*originalFormalNode*/,
           const NodeMap& /*nodeMap*/) -> std::vector<std::vector<Dim>> {
      auto* selfValue = node->inputs()[0].value;
      auto selfId = selfValue->id();
      auto selfActual = map.find(selfId);
      auto& selfTensor =
          frame.getIValue(selfActual != map.end() ? selfActual->second : selfId)
              .toTensor();
      auto* indexValue = node->inputs()[1].value;
      auto indexId = indexValue->id();
      auto indexActual = map.find(indexId);
      auto& indexTensor =
          frame
              .getIValue(
                  indexActual != map.end() ? indexActual->second : indexId)
              .toTensor();
      auto dimOpt = constIntArg(node, "dim");
      TORCH_CHECK(
          dimOpt.has_value(), "tw.index_select requires a constant dim");
      int64_t rank = selfTensor.dim();
      int64_t dim = *dimOpt < 0 ? *dimOpt + rank : *dimOpt;
      std::vector<Dim> shape(
          selfTensor.sizes().begin(), selfTensor.sizes().end());
      shape.at(dim) = static_cast<Dim>(indexTensor.numel());
      return {shape};
    };

    MetadataBuilder(
        std::make_unique<c10::FunctionSchema>(
            "tw.index_select",
            "",
            std::vector<c10::Argument>{
                c10::Argument("self", c10::TensorType::get()),
                c10::Argument("index", c10::TensorType::get()),
                c10::Argument("dim", c10::IntType::get())},
            std::vector<c10::Argument>{
                c10::Argument("output", c10::TensorType::get())}))
        .elementwiseFunc("__index_select")
        // index_select's output has a different element count than any input,
        // so it must not reuse an input buffer as its output.
        .inPlaceIfLastUse(false)
        .sizeOrdinal({0})
        .hasIdxArg()
        .hasOutputArg()
        .hasBlockInfo()
        .outputConstraints(
            [](NodeCP node,
               const ValueTypes& types) -> std::vector<ValueConstraint> {
              return {
                  {.rank = types.rank(inputAt(node, 0)),
                   .contiguity = Contiguity::kContiguous}};
            })
        .argumentMeta(
            {{.isRegister = false, .wholeTensor = true, .randomAccess = true},
             {.isRegister = false, .wholeTensor = true, .randomAccess = true},
             {.isRegister = true}})
        .returnMeta({{.isRegister = true, .reserveShape = indexSelectReserve}})
        .typeTemplateParams({0})
        // The output shape (source shape with 'dim' resized to the index
        // length) is not derivable from the operands, so when fused the output
        // is kept as a shape-only tensor and the enclosing expression is sized
        // from it.
        .sizeFromOutput()
        // 'dim' is a compile-time template parameter: it selects different
        // device code per axis and, because subgraphNodesMatch keys on
        // templateAttrs, keeps index_selects along different dims as distinct
        // ops. That prevents an op for one dim from being reused for another,
        // which would make indexSelectReserve read the wrong (formal) dim.
        .templateAttrs({"dim"})
        .registerOp();
  }

  // index_select (functional, out-of-place): gather rows along 'dim' by index.
  // Retarget to the fused elementwise tw.index_select when 'dim' is a
  // statically-known constant; a dynamic dim falls back to the standalone op.
  MetadataBuilder("torch.ops.aten.index_select.default")
      .sizeOrdinal({0})
      .isStandalone()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            return {
                {.rank = types.rank(inputAt(node, 0)),
                 .contiguity = Contiguity::kContiguous}};
          })
      .maybeReplace(
          [](NodeCP node, ValueTypes& /*types*/, WaveGraph& /*waveGraph*/)
              -> std::vector<std::pair<ValueCP, ValueCP>> {
            if (hasDynamicArg(node, "dim") ||
                !constIntArg(node, "dim").has_value()) {
              return {};
            }
            const_cast<nativert::Node*>(node)->setTarget("tw.index_select");
            return {};
          })
      .registerOp();

  // tw.searchsorted / tw.bucketize: fused elementwise binary search. The query
  // 'self' is a per-element register input; the searched array
  // (sorted_sequence / boundaries) is a whole tensor read at random. 'right' is
  // a runtime bool (side folded into it by normalizeSearchAttrs). Output shape
  // == the query shape: the wholeTensor searched array is excluded from the
  // output sizing, so the output is sized by the (query) inputs. The output
  // dtype (int64, or int32 for out_int32) rides on the output value's
  // TensorMeta and is selected in the kernel by the out_int32 template arg, so
  // no arithmeticPromotion
  // -- that would force the int output dtype onto the float inputs, like it
  // must not for comparisons. searchsorted supports an N-D sorted sequence
  // (per-row search) so it takes the loop index and the enclosing output
  // tensor; bucketize's boundaries are always 1-D, so it needs neither.
  {
    MetadataBuilder(
        std::make_unique<c10::FunctionSchema>(
            "tw.searchsorted",
            "",
            std::vector<c10::Argument>{
                c10::Argument("sorted_sequence", c10::TensorType::get()),
                c10::Argument("self", c10::TensorType::get()),
                c10::Argument(
                    "right",
                    c10::BoolType::get(),
                    std::nullopt,
                    c10::IValue(false))},
            std::vector<c10::Argument>{
                c10::Argument("output", c10::TensorType::get())}))
        .elementwiseFunc("__searchsorted")
        // Output dtype (int) differs from the value dtype, so it must not reuse
        // an input buffer as its output.
        .inPlaceIfLastUse(false)
        .hasIdxArg()
        .hasOutputArg()
        .ignoreAttrs({"side", "sorter"})
        .outputConstraints(
            [](NodeCP node,
               const ValueTypes& types) -> std::vector<ValueConstraint> {
              return {
                  {.rank = types.rank(inputAt(node, 1)),
                   .contiguity = Contiguity::kContiguous}};
            })
        .argumentMeta(
            {{.isRegister = false, .wholeTensor = true, .randomAccess = true},
             {.isRegister = true},
             {.isRegister = true}})
        .returnMeta({{.isRegister = true}})
        // T is the shared value dtype of sorted_sequence and self; the kernel
        // reads storage<T>(sorted_sequence).
        .typeTemplateParams({0})
        // out_int32 is a compile-time template arg (emitted after T) selecting
        // the output integer width. It also distinguishes the int32 and int64
        // kernels so they are not deduped (they share dtypes) into one wrongly
        // typed store.
        .templateAttrs({"out_int32"})
        .registerOp();

    MetadataBuilder(
        std::make_unique<c10::FunctionSchema>(
            "tw.bucketize",
            "",
            std::vector<c10::Argument>{
                c10::Argument("self", c10::TensorType::get()),
                c10::Argument("boundaries", c10::TensorType::get()),
                c10::Argument(
                    "right",
                    c10::BoolType::get(),
                    std::nullopt,
                    c10::IValue(false))},
            std::vector<c10::Argument>{
                c10::Argument("output", c10::TensorType::get())}))
        .elementwiseFunc("__bucketize")
        .inPlaceIfLastUse(false)
        .ignoreAttrs({"side", "sorter"})
        .outputConstraints(
            [](NodeCP node,
               const ValueTypes& types) -> std::vector<ValueConstraint> {
              return {
                  {.rank = types.rank(inputAt(node, 0)),
                   .contiguity = Contiguity::kContiguous}};
            })
        .argumentMeta(
            {{.isRegister = true},
             {.isRegister = false, .wholeTensor = true, .randomAccess = true},
             {.isRegister = true}})
        .returnMeta({{.isRegister = true}})
        // T is the shared value dtype of self and boundaries; the kernel reads
        // storage<T>(boundaries).
        .typeTemplateParams({1})
        // out_int32 is a compile-time template arg (emitted after T) selecting
        // the output integer width. It also distinguishes the int32 and int64
        // kernels so they are not deduped (they share dtypes) into one wrongly
        // typed store.
        .templateAttrs({"out_int32"})
        .registerOp();
  }

  // aten.searchsorted.Tensor: retarget to the fused tw.searchsorted unless a
  // sorter is given (unsupported -> eager) or the query and sorted sequence do
  // not share a dtype (ATen's fast path requires equality; the fused kernel
  // reads both as one template type).
  MetadataBuilder("torch.ops.aten.searchsorted.Tensor")
      .sizeOrdinal({1})
      .isStandalone()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            return {
                {.rank = types.rank(inputAt(node, 1)),
                 .contiguity = Contiguity::kContiguous}};
          })
      .maybeReplace(
          [](NodeCP node, ValueTypes& types, WaveGraph& /*waveGraph*/)
              -> std::vector<std::pair<ValueCP, ValueCP>> {
            // A sorter (optional) is not supported; leave the eager op when one
            // is given as an input value or a non-None attribute.
            if (const auto* sorterInput = node->tryGetInput("sorter")) {
              if (sorterInput->value &&
                  sorterInput->value->type().kind() !=
                      nativert::Type::Kind::None) {
                return {};
              }
            }
            if (const auto* sorter = node->tryGetAttribute("sorter")) {
              if (!std::holds_alternative<nativert::None>(sorter->value)) {
                return {};
              }
            }
            auto sortedId = inputAt(node, 0)->id();
            auto selfId = inputAt(node, 1)->id();
            if (sortedId >= static_cast<int>(types.types.size()) ||
                selfId >= static_cast<int>(types.types.size()) ||
                !types.types[sortedId] || !types.types[selfId] ||
                types.types[sortedId]->dtype() !=
                    types.types[selfId]->dtype()) {
              return {};
            }
            auto* mutableNode = const_cast<nativert::Node*>(node);
            normalizeSearchAttrs(mutableNode);
            mutableNode->setTarget("tw.searchsorted");
            return {};
          })
      .registerOp();

  // aten.bucketize.Tensor: retarget to the fused tw.bucketize unless the query
  // and boundaries do not share a dtype. boundaries is always 1-D (ATen).
  MetadataBuilder("torch.ops.aten.bucketize.Tensor")
      .sizeOrdinal({0})
      .isStandalone()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            return {
                {.rank = types.rank(inputAt(node, 0)),
                 .contiguity = Contiguity::kContiguous}};
          })
      .maybeReplace(
          [](NodeCP node, ValueTypes& types, WaveGraph& /*waveGraph*/)
              -> std::vector<std::pair<ValueCP, ValueCP>> {
            auto selfId = inputAt(node, 0)->id();
            auto boundsId = inputAt(node, 1)->id();
            if (selfId >= static_cast<int>(types.types.size()) ||
                boundsId >= static_cast<int>(types.types.size()) ||
                !types.types[selfId] || !types.types[boundsId] ||
                types.types[selfId]->dtype() !=
                    types.types[boundsId]->dtype()) {
              return {};
            }
            auto* mutableNode = const_cast<nativert::Node*>(node);
            normalizeSearchAttrs(mutableNode);
            mutableNode->setTarget("tw.bucketize");
            return {};
          })
      .registerOp();

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
      .mutatesArg(0)
      .indicesArg(1)
      .valuesArg(2)
      .accumulateAttr("accumulate")
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
              // Maintain Value::users(): the node now reads cloneOutput, not
              // selfVal, in the self slot. A raw slot assignment alone leaves
              // cloneOutput with no recorded user and selfVal with a stale one,
              // which breaks the last-use / reuse analysis and the
              // clone-elision pass (which key off users()).
              cloneOutput->addUser(mutableNode);
              bool stillReadsSelf = false;
              for (const auto& in : mutableNode->inputs()) {
                if (in.value == selfVal) {
                  stillReadsSelf = true;
                  break;
                }
              }
              if (!stillReadsSelf) {
                const_cast<nativert::Value*>(selfVal)->eraseUser(mutableNode);
              }
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
        .mutatesArg(0)
        .indicesArg(1)
        .valuesArg(2)
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
          .mutatesArg(0)
          .indicesArg(1)
          .valuesArg(2)
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
        .mutatesArg(0)
        .indicesArg(1)
        .valuesArg(2)
        .accumulateAttr("accumulate")
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
                // The original index_put_ node is superseded by maskedPutNode
                // (its output is replaced below), so it no longer consumes
                // self. Drop it from self's users() so the storage's user set
                // lists only the live consumer, maskedPutNode.
                const_cast<nativert::Value*>(selfVal)->eraseUser(
                    const_cast<nativert::Node*>(node));
                return {{node->outputs()[0], resultValue}};
              }
              return {};
            })
        .registerOp();
  }

  // Transpose -- a view that permutes dims, generally not contiguous.
  // EXPERIMENT: transpose as a fusable view rather than an unconditional
  // standalone, so setOutputs reaches its meta->isView() branch and the view
  // becomes an OutputDesc (viewNode) of the enclosing kernel op.
  MetadataBuilder("torch.ops.aten.transpose.int")
      .sizeOrdinal({0})
      .viewOfArg(0)
      .metadataOnly()
      .isStandaloneFunc(viewHasDynamicShapeArgs)
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            return {
                {.rank = types.rank(inputAt(node, 0)),
                 .contiguity = Contiguity::kUnknown}};
          })
      .registerOp();

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
      .normalizeDimAttr()
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
            constraint.contiguity =
                (types.contiguous(self) && stepOne && (dimZero || fullSlice))
                ? Contiguity::kContiguous
                : Contiguity::kUnknown;
            return {constraint};
          })
      .registerOp();

  // Select.
  MetadataBuilder("torch.ops.aten.select.int")
      .sizeOrdinal({0})
      .viewOfArg(0)
      .metadataOnly()
      .normalizeDimAttr()
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
            constraint.contiguity =
                (types.contiguous(self) && dimKnown && dim == 0)
                ? Contiguity::kContiguous
                : Contiguity::kUnknown;
            return {constraint};
          })
      .registerOp();

  // Narrow.
  MetadataBuilder("torch.ops.aten.narrow.default")
      .sizeOrdinal({0})
      .isStandalone()
      .viewOfArg(0)
      .metadataOnly()
      .normalizeDimAttr()
      .rankArgument(0)
      .registerOp();

  // tw.slice_scatter: fused in-place slice_scatter along dim 0 or dim 1. Args
  // mirror aten.slice_scatter (self, src, dim, start, end, step) so the rewrite
  // below can retarget the node directly. The loop is sized by 'src' (the
  // values written, an isRegister leaf), so it runs one iteration per src
  // element and scatters each into the output at its strided slice position;
  // the pass-through elements are supplied by the clone/in-place base and are
  // never touched. The output aliases self in place (mutatesArg -> aliasSelfId,
  // no reserveShape); the aten.slice_scatter rewrite inserts a clone of self
  // that clone-elision drops when self is dead. hasOutputArg passes the output
  // so the destination offset is decomposed with its magic dividers.
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
      .hasIdxArg()
      .hasOutputArg()
      .hasBlockInfo()
      .outputConstraints(viewOfFirstArgConstraint)
      .argumentMeta(
          {{.isRegister = false, .wholeTensor = true, .randomAccess = true},
           {.isRegister = true},
           {.isRegister = true},
           {.isRegister = true},
           {.isRegister = true},
           {.isRegister = true}})
      .returnMeta(
          {{.isRegister = false, .wholeTensor = true, .randomAccess = true}})
      .typeTemplateParams({0})
      .mutatesArg(0)
      .valuesArg(1)
      .registerOp();

  // tw.select_scatter: fused in-place select_scatter along dim 0 or dim 1.
  // Args mirror aten.select_scatter (self, src, dim, index) so the rewrite
  // below can retarget the node directly. The loop is sized by 'src' (the
  // values written, an isRegister leaf), so it runs one iteration per src
  // element and writes each to its destination in the output (src's shape is
  // self's with 'dim' removed; the destination inserts 'index' at 'dim'); the
  // pass-through elements are supplied by the clone/in-place base and are never
  // touched. The output aliases self in place (mutatesArg -> aliasSelfId, no
  // reserveShape); the aten.select_scatter rewrite inserts a clone of self that
  // clone-elision drops when self is dead. hasOutputArg passes the output so
  // the destination offset is decomposed with its magic dividers.
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.select_scatter",
          "",
          std::vector<c10::Argument>{
              c10::Argument("self", c10::TensorType::get()),
              c10::Argument("src", c10::TensorType::get()),
              c10::Argument(
                  "dim", c10::IntType::get(), std::nullopt, c10::IValue(0)),
              c10::Argument(
                  "index", c10::IntType::get(), std::nullopt, c10::IValue(0))},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .elementwiseFunc("__select_scatter")
      .hasIdxArg()
      .hasOutputArg()
      .hasBlockInfo()
      .outputConstraints(viewOfFirstArgConstraint)
      .argumentMeta(
          {{.isRegister = false, .wholeTensor = true, .randomAccess = true},
           {.isRegister = true},
           {.isRegister = true},
           {.isRegister = true}})
      .returnMeta(
          {{.isRegister = false, .wholeTensor = true, .randomAccess = true}})
      .typeTemplateParams({0})
      .mutatesArg(0)
      .valuesArg(1)
      .registerOp();

  // tw.scatter / tw.scatter_add: fused in-place scatter along 'dim'. Args
  // mirror aten.scatter.src / aten.scatter_add.default (self, dim, index, src)
  // so the rewrite retargets the node directly. The loop is sized by 'src' (the
  // values written, an isRegister leaf), one iteration per src element; 'index'
  // (same shape as src, checked by the rewrite when both shapes are static) is
  // read whole to supply the destination coordinate along 'dim'. scatter
  // overwrites; scatter_add accumulates atomically so duplicate indices sum.
  // The output aliases self in place (mutatesArg -> aliasSelfId, no
  // reserveShape); the functional rewrite inserts a clone of self that
  // clone-elision drops when self is dead.
  for (const bool accumulate : {false, true}) {
    MetadataBuilder(
        std::make_unique<c10::FunctionSchema>(
            accumulate ? "tw.scatter_add" : "tw.scatter",
            "",
            std::vector<c10::Argument>{
                c10::Argument("self", c10::TensorType::get()),
                c10::Argument("dim", c10::IntType::get()),
                c10::Argument("index", c10::TensorType::get()),
                c10::Argument("src", c10::TensorType::get())},
            std::vector<c10::Argument>{
                c10::Argument("output", c10::TensorType::get())}))
        .elementwiseFunc(accumulate ? "__scatter_add" : "__scatter")
        .hasIdxArg()
        .hasOutputArg()
        .hasBlockInfo()
        .outputConstraints(viewOfFirstArgConstraint)
        .argumentMeta(
            {{.isRegister = false, .wholeTensor = true, .randomAccess = true},
             {.isRegister = true},
             {.isRegister = false, .wholeTensor = true, .randomAccess = true},
             {.isRegister = true}})
        .returnMeta(
            {{.isRegister = false, .wholeTensor = true, .randomAccess = true}})
        .typeTemplateParams({0})
        .mutatesArg(0)
        .valuesArg(2)
        .registerOp();
  }

  // slice_scatter (functional, out-of-place): returns a fresh copy of self with
  // a slice overwritten by src. For dim 0 and dim 1 with a statically-known
  // dim, insert a clone of self and retarget to the in-place tw.slice_scatter
  // (which writes the clone); clone-elision then drops the clone when self is a
  // dead intermediate, realizing the write in place, and keeps it as a
  // defensive copy when self is still live. Other dims (or a dynamic dim) fall
  // back to the standalone op.
  MetadataBuilder("torch.ops.aten.slice_scatter.default")
      .sizeOrdinal({0})
      .isStandalone()
      .mutatesArg(0)
      .valuesArg(1)
      .dimAttr("dim")
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            auto* self = inputAt(node, 0);
            return {
                {.rank = types.rank(self),
                 .contiguity = Contiguity::kContiguous}};
          })
      .maybeReplace(
          [](NodeCP node,
             ValueTypes& types,
             WaveGraph& waveGraph) -> std::vector<std::pair<ValueCP, ValueCP>> {
            // Fuse scatter along dim 0 or dim 1 with a statically-known dim;
            // other cases fall back to the standalone op.
            int64_t scatterDim = constIntArg(node, "dim").value_or(0);
            if (hasDynamicArg(node, "dim") ||
                (scatterDim != 0 && scatterDim != 1)) {
              return {};
            }
            if (!insertSelfCloneForInPlace(
                    node, types, waveGraph, "slice_scatter_clone")) {
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
  // Not viewOfArg: flatten only returns a view when the flattened range is
  // contiguous in memory. On any other layout aten falls back to reshape, which
  // copies, so the output does not alias self and the op does real device work.
  // Nothing here constrains the input's layout, so claiming the view would let
  // the clone-elision and fusion paths (Metadata::isView) read through to a
  // buffer the output does not actually share. viewStorageBase is unaffected --
  // it takes the alias from the c10 schema's Tensor(a), which is authoritative.
  MetadataBuilder("torch.ops.aten.flatten.using_ints")
      .sizeOrdinal({0})
      .isStandalone()
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

  // Unbind. Each output is a select() view of self, so this enqueues nothing on
  // the device: metadataOnly keeps it from forcing a wave-stream sync before
  // its step's kernel, and from taking a timing sync that would charge it the
  // drain of every outstanding wave kernel.
  MetadataBuilder("torch.ops.aten.unbind.int")
      .sizeOrdinal({0})
      .isStandalone()
      .metadataOnly()
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

  // split_with_sizes: splits self along a dim into chunks whose sizes are given
  // by split_sizes. Each chunk is a view keeping self's rank (only the split
  // dim's extent changes), so it is not necessarily contiguous.
  MetadataBuilder("torch.ops.aten.split_with_sizes.default")
      .sizeOrdinal({0})
      .isStandalone()
      .metadataOnly()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            int8_t rank = types.rank(inputAt(node, 0));
            std::vector<ValueConstraint> constraints;
            constraints.reserve(node->outputs().size());
            for (size_t i = 0; i < node->outputs().size(); ++i) {
              constraints.push_back({rank});
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
      .makeMultiKernelVariant(
          singlePass ? makeMaskedSelectSinglePassVariant
                     : makeMaskedSelectVariant)
      .cgVariant(
          singlePass ? makeMaskedSelectSinglePassVariant
                     : makeMaskedSelectCgVariant)
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
            },
            .shapeFromInput = 0}})
      .normalize(resolveDtypeFromInput)
      .makeMultiKernelVariant(
          singlePass ? makeCumsumSinglePassVariant : makeCumsumVariant)
      .cgVariant(singlePass ? makeCumsumSinglePassVariant : makeCumsumCgVariant)
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
      .returnMeta(
          {{.isRegister = false,
            .reserveShape = inputShape,
            .shapeFromInput = 0}})
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
      .returnMeta(
          {{.isRegister = false,
            .reserveShape = inputShapePlusOne,
            .shapeFromInput = 0}})
      .normalize(resolveDtypeFromInput)
      .makeMultiKernelVariant(
          singlePass ? makeExclusiveSumSinglePassVariant
                     : makeExclusiveSumVariant)
      .cgVariant(
          singlePass ? makeExclusiveSumSinglePassVariant
                     : makeExclusiveSumCgVariant)
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
      .returnMeta(
          {{.isRegister = false,
            .reserveShape = inputShapePlusOne,
            .shapeFromInput = 0}})
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

  // tw.lengths_to_offsets: single-block exclusive prefix sum truncated to the
  // input length (fb.lengths_to_offsets with include_last_offset=False). Like
  // exclusive_sum but the output has N (not N+1) elements: it drops the
  // trailing total. Always single block: the lengths tensor is only a few
  // thousand elements, so the multi-kernel / cg scan is not worth it.
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.lengths_to_offsets",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .sizeShortcut(SizeShortcut::kMax)
      .hasBarrier()
      .singleBlockIfFused()
      .returnMeta(
          {{.isRegister = false,
            .reserveShape = inputShape,
            .shapeFromInput = 0}})
      .normalize(resolveDtypeFromInputExact)
      .headerFile(kScanHeader)
      .deviceFunc("lengths_to_offsets")
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
          {{.isRegister = false,
            .reserveShape = inputShape,
            .shapeFromInput = 0},
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
          {{.isRegister = false,
            .reserveShape = inputShapePlusOne,
            .shapeFromInput = 0},
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
            .shapeFromInput = 0,
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

  // --- Single-pass (decoupled look-back) variants ---
  //
  // One launch that reads the input once, where the cg forms above read it
  // twice with an opBarrier between. Each carries a second output, the
  // per-tile look-back state. They are always registered; only the expansions
  // that produce them are behind WaveConfig::singlePass. Like the cg forms they
  // need every block of the op resident, which the declared barrier (used once,
  // to publish the cleared descriptors) makes the launch provide.

  // tw.cumsum_1pass: (Tensor) -> (Tensor, Tensor[lookback])
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.cumsum_1pass",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get()),
              c10::Argument("lookback", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .sizeShortcut(SizeShortcut::kMax)
      .hasBarrier()
      .returnMeta(
          {{.isRegister = false,
            .reserveShape = inputShape,
            .shapeFromInput = 0},
           {.isRegister = false, .reserveShape = lookbackStateShape}})
      .normalize(resolveDtypeFromInput)
      .headerFile(kScanHeader)
      .deviceFunc("cumsum_1pass")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "numTiles"}})
      .dynamicSharedDecls(
          {{Metadata::kTypeFromDtype, "runTotal"},
           {Metadata::kTypeFromDtype, "tileExclusive"}})
      .typeTemplateParams({0})
      .hasDtypeTemplateParam()
      .hasBlockSizeTemplateParam()
      .numBarriers(2)
      // The look-back holds kScanItemsPerThread partials per thread in
      // registers, which under a cooperative grid would cost every op sharing
      // the kernel a block per SM. Cap the budget at the occupancy the rest of
      // a scan-bearing kernel runs at.
      .minBlocksPerSm(4)
      // Like the multi-kernel final stages, the output is read cross-block by
      // fused consumers, so end the launch after it in multi-block mode. In
      // cooperative-grid mode the launch does not end and the trailing
      // opBarrier of the device function is what fences the output.
      .scanOutputReturnBarrier()
      .only1d()
      .templateAttrs({"dim"})
      .outputConstraints(rank1Constraint)
      .cost(25.0f)
      .registerOp();

  // tw.exclusive_sum_1pass: (Tensor) -> (Tensor, Tensor[lookback])
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.exclusive_sum_1pass",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get()),
              c10::Argument("lookback", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .sizeShortcut(SizeShortcut::kMax)
      .hasBarrier()
      .returnMeta(
          {{.isRegister = false,
            .reserveShape = inputShapePlusOne,
            .shapeFromInput = 0},
           {.isRegister = false, .reserveShape = lookbackStateShape}})
      .normalize(resolveDtypeFromInput)
      .headerFile(kScanHeader)
      .deviceFunc("exclusive_sum_1pass")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "numTiles"}})
      .dynamicSharedDecls(
          {{Metadata::kTypeFromDtype, "runTotal"},
           {Metadata::kTypeFromDtype, "tileExclusive"}})
      .typeTemplateParams({0})
      .hasDtypeTemplateParam()
      .hasBlockSizeTemplateParam()
      .numBarriers(2)
      // The look-back holds kScanItemsPerThread partials per thread in
      // registers, which under a cooperative grid would cost every op sharing
      // the kernel a block per SM. Cap the budget at the occupancy the rest of
      // a scan-bearing kernel runs at.
      .minBlocksPerSm(4)
      // Like the multi-kernel final stages, the output is read cross-block by
      // fused consumers, so end the launch after it in multi-block mode. In
      // cooperative-grid mode the launch does not end and the trailing
      // opBarrier of the device function is what fences the output.
      .scanOutputReturnBarrier()
      .only1d()
      .outputConstraints(rank1Constraint)
      .cost(25.0f)
      .registerOp();

  // tw.masked_select_1pass: (Tensor, Tensor) -> (Tensor, Tensor[lookback]).
  // The selected elements of a tile are staged through a __shared__ buffer so
  // they leave in contiguous, fully coalesced stores; the buffer is sized for
  // the block size in force when the op is registered.
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.masked_select_1pass",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get()),
              c10::Argument("mask", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get()),
              c10::Argument("lookback", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .hasBarrier()
      .returnMeta(
          {{.isRegister = false,
            .reserveShape = inputShape,
            .shapeFromInput = 0,
            .shapeSetOnDevice = true},
           {.isRegister = false, .reserveShape = lookbackStateShape}})
      .headerFile(kScanHeader)
      .deviceFunc("masked_select_1pass")
      .multiBlockReturnBarrier()
      .sharedDecls(
          {{fmt::format("SelectStaging<{}>", WaveConfig::get().blockSize),
            "staging"},
           {"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "numTiles"},
           {"int32_t", "tileTotal"},
           {"int32_t", "tileExclusive"}})
      .typeTemplateParams({0})
      .hasBlockSizeTemplateParam()
      .numBarriers(2)
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
      // One row per non-zero element, one column per input dim (nonzeroReserve
      // and nonzero1d both produce that shape), so the result is one rank wider
      // than the input. Without this, consumers see an unknown rank and fall
      // back to their eager form.
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            auto rank = types.rank(inputAt(node, 0));
            return {
                {.rank = rank >= 0 ? static_cast<int8_t>(rank + 1)
                                   : static_cast<int8_t>(-1),
                 .contiguity = Contiguity::kContiguous}};
          })
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

  // Cat and stack — registered together in Cat.cpp.
  registerConcatMetadata();

  // --- repeat_interleave ---

  MetadataBuilder("torch.ops.aten.repeat_interleave.self_Tensor")
      .sizeOrdinal({1})
      .outputConstraints(rank1Constraint)
      .maybeReplace(
          [](NodeCP node,
             ValueTypes& types,
             WaveGraph& waveGraph) -> std::vector<std::pair<ValueCP, ValueCP>> {
            auto* graph = waveGraph.graph();
            auto* input = node->inputs()[0].value;
            auto* repeats = node->inputs()[1].value;
            auto inputDtype = types.types[input->id()]->dtype();
            // The interleave axis is resolved once, here, and handed to both
            // nodes already normalized: nullopt means dim=None (flatten to
            // 1-D), otherwise a non-negative axis. Normalizing in one place
            // keeps the head and the final from disagreeing about what a
            // negative 'dim' means.
            std::optional<int64_t> dim;
            if (auto dimOpt = constIntArg(node, "dim")) {
              int64_t rank = static_cast<int64_t>(types.rank(input));
              int64_t resolved = *dimOpt < 0 ? *dimOpt + rank : *dimOpt;
              TORCH_CHECK(
                  resolved >= 0 && resolved < rank,
                  "repeat_interleave: dim ",
                  *dimOpt,
                  " out of range for rank ",
                  rank);
              dim = resolved;
            }
            // A 0-dim repeat count broadcasts to every segment; the broadcast
            // head reads the segment count from its (host-sized) prefix, so it
            // takes 'input' and the 'dim' attr. A 1-D repeats is per-segment.
            //
            // torch.repeat_interleave also broadcasts a 1-D repeats of length
            // 1, but that cannot be detected here: an input's TensorMeta
            // carries placeholder sizes (only its rank is meaningful), so
            // numel() would misroute every 1-D repeats to the broadcast head.
            // The length is first known at launch, by which point the head
            // kernel is already chosen -- so instead of guessing,
            // repeatInterleaveFinalReserve rejects a prefix that does not cover
            // the segment count, turning what would be an out-of-bounds device
            // read into a host-side error.
            nativert::Node* headNode = nullptr;
            if (types.rank(repeats) == 0) {
              headNode = graph->createNode(
                  "tw.repeat_interleave_bcast_head",
                  {{"repeats", repeats}, {"input", input}});
              if (dim.has_value()) {
                headNode->addAttribute({"dim", *dim});
              }
            } else {
              headNode = graph->createNode(
                  "tw.repeat_interleave_head", {{"repeats", repeats}});
            }
            auto* prefixOutput = waveGraph.newTensorValue(
                headNode, "repeat_prefix", c10::ScalarType::Int);
            auto* totalOutput = waveGraph.newScalarValue(
                headNode, "repeat_total", c10::ScalarType::Int);
            auto* finalNode = graph->createNode(
                "tw.repeat_interleave_final",
                {{"input", input},
                 {"prefix", prefixOutput},
                 {"total", totalOutput}});
            // Propagate the interleave axis so the final's reserveShape and
            // rank constraint reconstruct the full (multi-dim) output shape,
            // and so the kernel knows the axis without re-deriving it from the
            // shapes (which is ambiguous when the repeats sum back to
            // size(input, dim)). It is a template parameter of the device func,
            // so the flatten-to-1D case (dim=None) is encoded as -1.
            finalNode->addAttribute({"dim", dim.value_or(-1)});
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
          {{.isRegister = false,
            .reserveShape = inputShape,
            .shapeFromInput = 0},
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

  auto repeatInterleaveBcastReserve =
      [](NodeCP node,
         nativert::ExecutionFrame& frame,
         const FormalToActual& map,
         NodeCP /*originalFormalNode*/,
         const NodeMap& /*nodeMap*/) -> std::vector<std::vector<Dim>> {
    // prefix length = number of source segments = size(input, dim), or
    // numEl(input) when flattening (dim=None).
    auto inputTensor = paramTensor(node->inputs()[1].value, frame, map);
    auto dimOpt = constIntArg(node, "dim");
    if (!dimOpt.has_value()) {
      return {{static_cast<Dim>(inputTensor.numel())}};
    }
    // 'dim' arrives already normalized to a non-negative axis -- the rewrite
    // resolves it once for both this head and the final -- so this only
    // re-checks it against the runtime rank.
    int64_t rank = inputTensor.dim();
    int64_t dim = *dimOpt;
    TORCH_CHECK(
        dim >= 0 && dim < rank,
        "repeat_interleave: dim ",
        dim,
        " out of range for rank ",
        rank);
    return {{static_cast<Dim>(inputTensor.size(dim))}};
  };

  // Broadcast (0-dim repeat count) head: prefix[i] = (i+1)*r, sized to the
  // segment count. Reused for both dim-given and flatten (dim=None) cases.
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.repeat_interleave_bcast_head",
          "",
          std::vector<c10::Argument>{
              c10::Argument("repeats", c10::TensorType::get()),
              c10::Argument("input", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("prefix", c10::TensorType::get()),
              c10::Argument("total", c10::IntType::get())}))
      .sizeOrdinal({0})
      .hasBarrier()
      .returnMeta(
          {{.isRegister = false, .reserveShape = repeatInterleaveBcastReserve},
           {.neededOnHost = true}})
      .headerFile(kScanHeader)
      .deviceFunc("repeat_interleave_bcast_head")
      .typeTemplateParams({0})
      .hasBlockSizeTemplateParam()
      .multiBlockReturnBarrier()
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
    // The whole scan pipeline indexes prefixes with int32, so a result at or
    // past INT32_MAX cannot be represented no matter how it arose. The
    // broadcast head saturates to INT32_MAX on an overflowing segments *
    // repeatCount and reports it through debugInfo, but that report is
    // diagnostic only -- errorString() is gated on keepStatsOnThread and never
    // throws -- so without this the gather would run on a clamped prefix and
    // return wrong data. 'total' is already neededOnHost, so gating here costs
    // no extra device round trip.
    TORCH_CHECK(
        total < std::numeric_limits<int32_t>::max(),
        "repeat_interleave: result of ",
        total,
        " elements along the interleave axis is not representable in the "
        "int32 prefix-sum pipeline");
    auto dimOpt = constIntArg(node, "dim");
    // The gather walks prefix[0 .. segments), where segments is size(input,
    // dim) or numEl(input) when flattening. A per-segment 'repeats' sizes the
    // prefix by its own length, so a mismatch (notably torch's length-1
    // broadcast form, which the rewrite cannot recognize -- see the comment
    // there) would read past the end of the prefix on device. Checked here,
    // where the real extents are known, so it surfaces as a host-side error.
    // The index_final overload shares this reserve, carries no 'dim' at all,
    // and sizes its prefix from the same repeats it iterates, so it is exempt.
    auto checkPrefixCoversSegments = [&](int64_t segments) {
      auto prefixTensor = paramTensor(node->inputs()[1].value, frame, map);
      TORCH_CHECK(
          prefixTensor.numel() == segments,
          "repeat_interleave: 'repeats' has ",
          prefixTensor.numel(),
          " entries but the input has ",
          segments,
          " segments along the interleave axis; a length-1 broadcast 'repeats'"
          " is not supported on the wave path");
    };
    // dim=None: repeat_interleave flattens to a 1-D output of 'total'
    // elements. The index_final overload shares this reserve and never carries
    // a 'dim'; the gather final always does, using -1 for the flatten case.
    if (!dimOpt.has_value()) {
      return {{static_cast<Dim>(total)}};
    }
    auto inputTensor = paramTensor(node->inputs()[0].value, frame, map);
    if (*dimOpt < 0) {
      checkPrefixCoversSegments(inputTensor.numel());
      return {{static_cast<Dim>(total)}};
    }
    int64_t rank = inputTensor.dim();
    int64_t dim = *dimOpt;
    TORCH_CHECK(
        dim < rank,
        "repeat_interleave: dim ",
        dim,
        " out of range for rank ",
        rank);
    checkPrefixCoversSegments(inputTensor.size(dim));
    std::vector<Dim> shape(
        inputTensor.sizes().begin(), inputTensor.sizes().end());
    shape.at(dim) = static_cast<Dim>(total);
    return {shape};
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
      // The gather decomposes the output index and maps it through the
      // output's strides, so it fills a pitched concat band correctly.
      .mayWriteStrided()
      .inputFromPreviousKernel(2)
      .headerFile(kScanHeader)
      .deviceFunc("repeat_interleave_final")
      .sharedDecls({{"uint32_t", "size"}})
      .typeTemplateParams({0})
      // The interleave axis is a compile-time parameter of the kernel: it
      // cannot be recovered on device from the shapes alone.
      .templateAttrs({"dim"})
      .hasBarrier()
      .hasBlockSizeTemplateParam()
      .multiBlockReturnBarrier()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            // dim=None (encoded as -1) flattens to rank 1; otherwise the
            // output keeps the input's rank (only the interleave axis changes
            // length).
            auto dimOpt = constIntArg(node, "dim");
            if (!dimOpt.has_value() || *dimOpt < 0) {
              return {{.rank = 1, .contiguity = Contiguity::kContiguous}};
            }
            return {
                {.rank = types.rank(inputAt(node, 0)),
                 .contiguity = Contiguity::kContiguous}};
          })
      .registerOp();

  // Index-generating overload: given a 1-D counts tensor, returns a 1-D tensor
  // where index i appears repeats[i] times (repeats=[2,3,1] -> [0,0,1,1,1,2]).
  // Same computation as self_Tensor except the output element is the segment
  // index i, not a gathered self[i]. Eager repeat_interleave(repeats) returns
  // the same dtype as repeats, so the output follows the repeats dtype. Reuses
  // the head, then runs the index-emitting final stage.
  MetadataBuilder("torch.ops.aten.repeat_interleave.Tensor")
      .sizeOrdinal({0})
      .ignoreAttrs({"output_size"})
      .outputConstraints(rank1Constraint)
      .maybeReplace(
          [](NodeCP node,
             ValueTypes& types,
             WaveGraph& waveGraph) -> std::vector<std::pair<ValueCP, ValueCP>> {
            auto* graph = waveGraph.graph();
            auto* repeats = node->inputs()[0].value;
            auto repeatsDtype = types.types[repeats->id()]->dtype();
            auto* headNode = graph->createNode(
                "tw.repeat_interleave_head", {{"repeats", repeats}});
            auto* prefixOutput = waveGraph.newTensorValue(
                headNode, "repeat_prefix", c10::ScalarType::Int);
            auto* totalOutput = waveGraph.newScalarValue(
                headNode, "repeat_total", c10::ScalarType::Int);
            auto* finalNode = graph->createNode(
                "tw.repeat_interleave_index_final",
                {{"repeats", repeats},
                 {"prefix", prefixOutput},
                 {"total", totalOutput}});
            auto* resultOutput = waveGraph.newTensorValue(
                finalNode, "repeat_result", repeatsDtype);
            return {{node->outputs()[0], resultOutput}};
          })
      .registerOp();

  // tw.repeat_interleave_index_final: (Tensor, Tensor, int) -> Tensor
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.repeat_interleave_index_final",
          "",
          std::vector<c10::Argument>{
              c10::Argument("repeats", c10::TensorType::get()),
              c10::Argument("prefix", c10::TensorType::get()),
              c10::Argument("total", c10::IntType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .returnMeta(
          {{.isRegister = false, .reserveShape = repeatInterleaveFinalReserve}})
      // No mayWriteStrided: unlike repeat_interleave_final, this one writes the
      // run of repeated indices as out[j] for j in [start, end), a linear walk
      // that ignores the output's strides. A pitched concat band would land in
      // the wrong places, silently, since a placed operand is not copied in.
      .inputFromPreviousKernel(2)
      .headerFile(kScanHeader)
      .deviceFunc("repeat_interleave_index_final")
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

  // tw.pad_last_dim: the fused form of aten.pad below when constant mode pads
  // only the last dimension, where the op is just a gather with a default (see
  // __pad_last_dim). Elementwise, so it folds into the consumer's kernel
  // instead of taking a launch and a wave-stream sync of its own. 'self' is
  // read at computed offsets, so it is a whole-tensor random-access operand.
  // 'pad' carries the whole list for the host-side output width only and is
  // kept out of the kernel arguments (ignoreAttrs); the kernel takes the two
  // amounts as 'left'/'right' registers. Reading the list rather than the two
  // elements matters: paramIntListByName resolves the packed list from the
  // frame, which a dynamic pad amount populates before an element-wise read of
  // the prim.ListPack operands would see it.
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.pad_last_dim",
          "",
          std::vector<c10::Argument>{
              c10::Argument("self", c10::TensorType::get()),
              c10::Argument("pad", c10::ListType::create(c10::IntType::get())),
              c10::Argument("left", c10::IntType::get()),
              c10::Argument("right", c10::IntType::get()),
              c10::Argument("value", c10::NumberType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .elementwiseFunc("__pad_last_dim")
      .hasIdxArg()
      .ignoreAttrs({"pad"})
      .argumentMeta(
          {{.isRegister = false, .wholeTensor = true, .randomAccess = true},
           {},
           {.isRegister = true},
           {.isRegister = true},
           {.isRegister = true}})
      .returnMeta(
          {{.isRegister = true,
            .reserveShape =
                [](NodeCP node,
                   nativert::ExecutionFrame& frame,
                   const FormalToActual& map,
                   NodeCP originalFormalNode,
                   const NodeMap& nodeMap) -> std::vector<std::vector<Dim>> {
              auto self = paramTensor(node->inputs()[0].value, frame, map);
              auto pad = paramIntListByName(
                  originalFormalNode, "pad", frame, map, nodeMap);
              auto sizes = self.sizes();
              std::vector<Dim> shape(sizes.begin(), sizes.end());
              shape.back() += static_cast<Dim>(pad[0] + pad[1]);
              return {std::move(shape)};
            }}})
      .typeTemplateParams({0})
      // The output extent (self's, widened by the pad amounts) is not derivable
      // from the operands: self is a whole-tensor operand, which the
      // elementwise size leaves drop, so a fused pad would leave its consumer's
      // expression with no size leaf at all. sizeFromOutput keeps the output as
      // a shape-only tensor and sizes the enclosing expression from it, as for
      // index_select.
      .sizeFromOutput()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            return {
                {.rank = types.rank(inputAt(node, 0)),
                 .contiguity = Contiguity::kContiguous}};
          })
      .registerOp();

  MetadataBuilder("torch.ops.aten.pad.default")
      .isStandalone()
      .maybeReplace(
          [](NodeCP node,
             ValueTypes& types,
             WaveGraph& waveGraph) -> std::vector<std::pair<ValueCP, ValueCP>> {
            // Only constant mode is an index-with-default: reflect/replicate
            // mirror indices back into self, and circular wraps them. A pad
            // list longer than 2 pads interior dims, which would need those
            // dims decomposed out of the lane index too.
            const auto* mode = node->tryGetAttribute("mode");
            if (mode && std::holds_alternative<std::string>(mode->value) &&
                std::get<std::string>(mode->value) != "constant") {
              return {};
            }
            auto* self = inputAt(node, 0);
            if (!self || types.rank(self) < 1) {
              return {};
            }
            auto selfId = self->id();
            if (selfId < 0 ||
                static_cast<size_t>(selfId) >= types.types.size() ||
                !types.types[selfId]) {
              return {};
            }
            // normalize() above fills a missing/None 'value' with a typed
            // zero, so anything else here is a form this rewrite does not
            // model.
            const auto* value = node->tryGetAttribute("value");
            if (!value) {
              return {};
            }
            nativert::Constant deflt;
            if (std::holds_alternative<int64_t>(value->value)) {
              deflt = nativert::Constant{std::get<int64_t>(value->value)};
            } else if (std::holds_alternative<double>(value->value)) {
              deflt = nativert::Constant{std::get<double>(value->value)};
            } else {
              return {};
            }
            // The pad amounts are either a constant list attribute or a
            // prim.ListPack of two symints; both give the kernel its two
            // registers, and the list itself gives the reserve the width.
            auto* graph = waveGraph.graph();
            std::vector<nativert::NamedArgument> inputs{{"self", self}};
            nativert::Node* fused = nullptr;
            if (const auto* padValue = node->tryGetInput("pad")) {
              auto* pack = padValue->value->producer();
              if (!pack || pack->target() != "prim.ListPack" ||
                  pack->inputs().size() != 2) {
                return {};
              }
              inputs.push_back({"pad", padValue->value});
              inputs.push_back({"left", pack->inputs()[0].value});
              inputs.push_back({"right", pack->inputs()[1].value});
              fused = graph->createNode("tw.pad_last_dim", inputs);
            } else {
              const auto* padAttr = node->tryGetAttribute("pad");
              if (!padAttr ||
                  !std::holds_alternative<std::vector<int64_t>>(
                      padAttr->value)) {
                return {};
              }
              const auto& pad = std::get<std::vector<int64_t>>(padAttr->value);
              if (pad.size() != 2) {
                return {};
              }
              fused = graph->createNode("tw.pad_last_dim", inputs);
              fused->addAttribute({"pad", pad});
              fused->addAttribute({"left", pad[0]});
              fused->addAttribute({"right", pad[1]});
            }
            fused->addAttribute({"value", std::move(deflt)});
            graph->insertBefore(fused, const_cast<nativert::Node*>(node));
            auto* fusedOutput = waveGraph.newTensorValue(
                fused, "pad_last_dim", types.types[selfId]->dtype());
            return {{node->outputs()[0], fusedOutput}};
          })
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

  // ==========================================================================
  // Ops from the ads-preproc graph (model 830857007) that had no wave builtin.
  // The clearly-elementwise ones fuse via a device function; the rest are
  // registered as standalone (run eager via nativert) but carry rank /
  // contiguity metadata so the optimizer can propagate shapes through them.
  // ==========================================================================

  // Small helpers for inferring output constraints from the op docs.
  auto constAttrList = [](NodeCP node,
                          const char* name) -> const std::vector<int64_t>* {
    const auto* attr = node->tryGetAttribute(name);
    if (attr && std::holds_alternative<std::vector<int64_t>>(attr->value)) {
      return &std::get<std::vector<int64_t>>(attr->value);
    }
    return nullptr;
  };
  auto keepdimSet = [](NodeCP node) -> bool {
    const auto* attr = node->tryGetAttribute("keepdim");
    return attr && std::holds_alternative<bool>(attr->value) &&
        std::get<bool>(attr->value);
  };

  // --- Elementwise arithmetic (fused via device functions) ---
  MetadataBuilder("torch.ops.aten.floor_divide.default")
      .elementwiseFunc("__floor_divide")
      .arithmeticPromotion()
      .registerOp();
  // floor_divide.Scalar: divisor is a scalar attribute; same device function.
  MetadataBuilder("torch.ops.aten.floor_divide.Scalar")
      .elementwiseFunc("__floor_divide")
      .arithmeticPromotion()
      .normalize(castScalarAttrsToInputDtype)
      .registerOp();
  MetadataBuilder("torch.ops.aten.__rshift__.Tensor")
      .elementwiseFunc("__rshift")
      .registerOp();
  MetadataBuilder("torch.ops.aten.__rshift__.Scalar")
      .elementwiseFunc("__rshift")
      .normalize(castScalarAttrsToInputDtype)
      .registerOp();
  MetadataBuilder("torch.ops.aten.__lshift__.Scalar")
      .elementwiseFunc("__lshift")
      .normalize(castScalarAttrsToInputDtype)
      .registerOp();
  MetadataBuilder("torch.ops.aten.rsub.Scalar")
      .elementwiseFunc("__rsub")
      .arithmeticPromotion()
      .normalize(castScalarAttrsToInputDtype)
      .registerOp();

  // tw.div_trunc: elementwise truncating division, the target of
  // div.Tensor_mode with rounding_mode="trunc".
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.div_trunc",
          "",
          std::vector<c10::Argument>{
              c10::Argument("self", c10::TensorType::get()),
              c10::Argument("other", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .elementwiseFunc("__div_trunc")
      .arithmeticPromotion()
      .registerOp();

  // div.Tensor_mode: dispatch on the (constant) rounding mode to an existing
  // elementwise op -- None -> true division, "floor" -> floor_divide, "trunc"
  // -> div_trunc. A dynamic mode falls back to the standalone op.
  MetadataBuilder("torch.ops.aten.div.Tensor_mode")
      .sizeOrdinal({0})
      .isStandalone()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            // other may be a scalar attribute rather than a tensor input; when
            // it is a tensor (input 1) the output rank broadcasts over both.
            int8_t r = types.rank(inputAt(node, 0));
            const auto& inputs = node->inputs();
            if (inputs.size() > 1 && inputs[1].value) {
              r = std::max(r, types.rank(inputAt(node, 1)));
            }
            return {{.rank = r, .contiguity = Contiguity::kContiguous}};
          })
      .maybeReplace(
          [](NodeCP node, ValueTypes& /*types*/, WaveGraph& /*waveGraph*/)
              -> std::vector<std::pair<ValueCP, ValueCP>> {
            const auto* attr = node->tryGetAttribute("rounding_mode");
            if (!attr) {
              return {};
            }
            auto* mutableNode = const_cast<nativert::Node*>(node);
            if (std::holds_alternative<nativert::None>(attr->value)) {
              mutableNode->setTarget("torch.ops.aten.div.Tensor");
            } else if (std::holds_alternative<std::string>(attr->value)) {
              const auto& mode = std::get<std::string>(attr->value);
              if (mode == "floor") {
                mutableNode->setTarget("torch.ops.aten.floor_divide.default");
              } else if (mode == "trunc") {
                mutableNode->setTarget("tw.div_trunc");
              }
            }
            return {};
          })
      .registerOp();

  // div.Scalar_mode: like div.Tensor_mode but the divisor is a scalar
  // attribute. Dispatch the constant rounding mode to the matching scalar
  // elementwise op -- None -> div.Scalar, "floor" -> floor_divide.Scalar.
  // "trunc" has no scalar elementwise variant, and a dynamic mode is not
  // constant-foldable, so both keep the eager standalone.
  MetadataBuilder("torch.ops.aten.div.Scalar_mode")
      .sizeOrdinal({0})
      .isStandalone()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            return {
                {.rank = types.rank(inputAt(node, 0)),
                 .contiguity = Contiguity::kContiguous}};
          })
      .maybeReplace(
          [](NodeCP node, ValueTypes& /*types*/, WaveGraph& /*waveGraph*/)
              -> std::vector<std::pair<ValueCP, ValueCP>> {
            const auto* attr = node->tryGetAttribute("rounding_mode");
            if (!attr) {
              return {};
            }
            auto* mutableNode = const_cast<nativert::Node*>(node);
            if (std::holds_alternative<nativert::None>(attr->value)) {
              mutableNode->setTarget("torch.ops.aten.div.Scalar");
            } else if (std::holds_alternative<std::string>(attr->value)) {
              const auto& mode = std::get<std::string>(attr->value);
              if (mode == "floor") {
                mutableNode->setTarget("torch.ops.aten.floor_divide.Scalar");
              }
            }
            return {};
          })
      .registerOp();

  // --- Views / metadata-only (run as a host-side view or eager) ---

  // t: transpose a <=2-D tensor (rank preserved, view).
  registerRankPreservingStandalone("torch.ops.aten.t.default", 0);

  // detach: identity alias in inference; eliminate it.
  MetadataBuilder("torch.ops.aten.detach.default")
      .sizeOrdinal({0})
      .isStandalone()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            auto* self = inputAt(node, 0);
            return {
                {.rank = types.rank(self),
                 .contiguity = types.contiguity(self)}};
          })
      .maybeReplace(
          [](NodeCP node, ValueTypes& /*types*/, WaveGraph& /*waveGraph*/)
              -> std::vector<std::pair<ValueCP, ValueCP>> {
            const auto& outputs = node->outputs();
            if (outputs.empty()) {
              return {};
            }
            return {{outputs[0], inputAt(node, 0)}};
          })
      .registerOp();

  // diagonal: strided view; removes dim1/dim2 and appends the diagonal, so rank
  // drops by one and the result is not contiguous.
  MetadataBuilder("torch.ops.aten.diagonal.default")
      .sizeOrdinal({0})
      .isStandalone()
      .viewOfArg(0)
      .metadataOnly()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            auto r = types.rank(inputAt(node, 0));
            return {
                {.rank = r > 0 ? static_cast<int8_t>(r - 1)
                               : static_cast<int8_t>(-1),
                 .contiguity = Contiguity::kNoncontiguous}};
          })
      .registerOp();

  // expand: broadcast view (0 strides on expanded dims), rank = size length.
  //
  // Two lowerings. A consumer that reads the result from memory wants the view:
  // 0 strides cost nothing, while a fused broadcast would materialize every
  // repeat. A consumer that takes the value in a register wants the fused form:
  // the elementwise identity below inlines into the consumer's expression and
  // produces nothing at all, where the view costs a standalone launch and a
  // real tensor. isStandaloneFunc picks the view unless every consumer is an
  // elementwise op -- the aten.copy case, where expand is how a functionalized
  // `dst.copy_(src)` spells its broadcast.
  //
  // viewOfArg stays set either way: the fused form reads the source's storage,
  // so resolving the result to that base is what alias analysis needs (and what
  // copyMayOverlap relies on to see through an expand into the real buffer).
  MetadataBuilder("torch.ops.aten.expand.default")
      .sizeOrdinal({0})
      .isStandaloneFunc(
          [constAttrList](NodeCP node, const ValueTypes& /*types*/) {
            // A dynamic size arrives as a SymIntList operand that is not
            // resolved when the op runs, so only a compile-time size can be
            // fused; the ROO graph has both forms.
            if (!constAttrList(node, "size")) {
              return true;
            }
            const auto& outputs = node->outputs();
            if (outputs.empty() || outputs[0]->users().empty()) {
              return true;
            }
            for (auto* user : outputs[0]->users()) {
              const auto* userMeta = Registry::metadata(user->target());
              if (!userMeta || !userMeta->elementwise) {
                return true;
              }
            }
            return false;
          })
      .elementwiseFunc("__to")
      .numArgs(1)
      .generateCall([](std::stringstream& ss,
                       NodeCP /*node*/,
                       std::vector<std::string> args) {
        TORCH_CHECK(!args.empty(), "expand requires a self input");
        ss << args[0];
      })
      .viewOfArg(0)
      .metadataOnly()
      .shapeAttr("size")
      .outputConstraints(
          [constAttrList](
              NodeCP node,
              const ValueTypes& /*types*/) -> std::vector<ValueConstraint> {
            const auto* size = constAttrList(node, "size");
            if (!size) {
              return {};
            }
            return {
                {.rank = static_cast<int8_t>(size->size()),
                 .contiguity = Contiguity::kNoncontiguous,
                 .zeroStrides = true}};
          })
      .registerOp();

  // squeeze (all size-1 dims): rank is data-dependent, so leave it unknown;
  // it is a view over the same storage.
  MetadataBuilder("torch.ops.aten.squeeze.default")
      .sizeOrdinal({0})
      .isStandalone()
      .viewOfArg(0)
      .metadataOnly()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            return {
                {.rank = -1, .contiguity = types.contiguity(inputAt(node, 0))}};
          })
      .registerOp();

  // squeeze.dim: removes one dim if it is size 1 (a no-op otherwise), so the
  // output rank is data-dependent; leave it unknown. It is a view.
  MetadataBuilder("torch.ops.aten.squeeze.dim")
      .sizeOrdinal({0})
      .isStandalone()
      .viewOfArg(0)
      .metadataOnly()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            return {
                {.rank = -1, .contiguity = types.contiguity(inputAt(node, 0))}};
          })
      .registerOp();

  // --- Reductions (eager standalone, fresh contiguous output) ---

  // any.dim / sum.dim_IntList / min.dim: rank drops by the reduced dims unless
  // keepdim keeps them as size 1.
  auto singleDimReduceConstraint =
      [keepdimSet](
          NodeCP node,
          const ValueTypes& types) -> std::vector<ValueConstraint> {
    auto r = types.rank(inputAt(node, 0));
    if (r < 0) {
      return {{.rank = -1, .contiguity = Contiguity::kContiguous}};
    }
    int8_t out = keepdimSet(node) ? r : static_cast<int8_t>(r - 1);
    return {{.rank = out, .contiguity = Contiguity::kContiguous}};
  };
  auto listDimReduceConstraint =
      [keepdimSet, constAttrList](
          NodeCP node,
          const ValueTypes& types) -> std::vector<ValueConstraint> {
    auto r = types.rank(inputAt(node, 0));
    if (r < 0) {
      return {{.rank = -1, .contiguity = Contiguity::kContiguous}};
    }
    if (keepdimSet(node)) {
      return {{.rank = r, .contiguity = Contiguity::kContiguous}};
    }
    const auto* dims = constAttrList(node, "dim");
    int8_t reduced = dims ? static_cast<int8_t>(dims->size()) : r;
    return {
        {.rank = static_cast<int8_t>(r - reduced),
         .contiguity = Contiguity::kContiguous}};
  };

  MetadataBuilder("torch.ops.aten.any.dim")
      .sizeOrdinal({0})
      .isStandalone()
      .outputConstraints(singleDimReduceConstraint)
      .registerOp();
  MetadataBuilder("torch.ops.aten.any.dims")
      .sizeOrdinal({0})
      .isStandalone()
      .outputConstraints(listDimReduceConstraint)
      .registerOp();
  MetadataBuilder("torch.ops.aten.sum.dim_IntList")
      .sizeOrdinal({0})
      .isStandalone()
      .outputConstraints(listDimReduceConstraint)
      .registerOp();
  // min.dim returns (values, indices); both share the reduced shape.
  MetadataBuilder("torch.ops.aten.min.dim")
      .sizeOrdinal({0})
      .isStandalone()
      .outputConstraints(
          [singleDimReduceConstraint](
              NodeCP node,
              const ValueTypes& types) -> std::vector<ValueConstraint> {
            auto one = singleDimReduceConstraint(node, types);
            return {one[0], one[0]};
          })
      .registerOp();
  // max / min (global): fused single-scalar reductions across all three exec
  // forms, mirroring aten.sum.default. resolveDtypeFromInputExact keeps the
  // input dtype (unlike sum, which promotes ints to Long).
  registerReduction(
      kScanHeader,
      "torch.ops.aten.max.default",
      "max",
      resolveDtypeFromInputExact,
      makeMaxVariant,
      makeMaxCgVariant);
  registerReduction(
      kScanHeader,
      "torch.ops.aten.min.default",
      "min",
      resolveDtypeFromInputExact,
      makeMinVariant,
      makeMinCgVariant);

  // --- Gather / scatter / sort / search (eager standalone) ---

  // gather: output has the index tensor's shape. dim is a constant attribute,
  // so the node's tensor inputs are [self, index] and index is at position 1.
  MetadataBuilder("torch.ops.aten.gather.default")
      .sizeOrdinal({1})
      .isStandalone()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            return {
                {.rank = types.rank(inputAt(node, 1)),
                 .contiguity = Contiguity::kContiguous}};
          })
      .registerOp();
  // scatter.src(self, dim, index, src): overwrite self along dim at index. For
  // a statically-known dim 0 or 1 with matching index/src shapes, insert a
  // clone of self and retarget to the in-place tw.scatter; clone-elision drops
  // the clone when self is dead. Other cases fall back to the eager standalone
  // op.
  MetadataBuilder("torch.ops.aten.scatter.src")
      .sizeOrdinal({0})
      .isStandalone()
      .mutatesArg(0)
      .indicesArg(1)
      .valuesArg(2)
      .dimAttr("dim")
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            return {
                {.rank = types.rank(inputAt(node, 0)),
                 .contiguity = Contiguity::kContiguous}};
          })
      .maybeReplace(
          [](NodeCP node,
             ValueTypes& types,
             WaveGraph& waveGraph) -> std::vector<std::pair<ValueCP, ValueCP>> {
            tryFuseScatter(node, types, waveGraph, /*accumulate=*/false);
            return {};
          })
      .registerOp();
  // aten.copy.default is registered as a fused elementwise next to _to_copy,
  // which it decomposes to. The registry keeps the last registration for a
  // name, so it must not be re-registered as a standalone here.

  // Additional tensor-writing ops, registered as eager standalone (functional:
  // output has self's shape, freshly contiguous). The in-place rewrite pass
  // targets them via mutatesArg (self = tensor-input 0); even without a fused
  // kernel, rewriting the functional op to its aten in-place variant elides the
  // implicit clone of self. dim / offset / scalar-value args are constant
  // attributes, so the tensor inputs are self, then index/mask, then
  // source/values.
  {
    auto selfShapeContiguous =
        [](NodeCP node,
           const ValueTypes& types) -> std::vector<ValueConstraint> {
      return {
          {.rank = types.rank(inputAt(node, 0)),
           .contiguity = Contiguity::kContiguous}};
    };
    // index_copy(self, dim, index, source): overwrite along dim at index.
    MetadataBuilder("torch.ops.aten.index_copy.default")
        .sizeOrdinal({0})
        .isStandalone()
        .mutatesArg(0)
        .indicesArg(1)
        .valuesArg(2)
        .dimAttr("dim")
        .outputConstraints(selfShapeContiguous)
        .registerOp();
    // index_add(self, dim, index, source, alpha): accumulate along dim (always
    // accumulates; a fused in-place form would need atomics).
    MetadataBuilder("torch.ops.aten.index_add.default")
        .sizeOrdinal({0})
        .isStandalone()
        .mutatesArg(0)
        .indicesArg(1)
        .valuesArg(2)
        .dimAttr("dim")
        .outputConstraints(selfShapeContiguous)
        .registerOp();
    // index_fill.int_Scalar(self, dim, index, value): scalar fill along dim.
    MetadataBuilder("torch.ops.aten.index_fill.int_Scalar")
        .sizeOrdinal({0})
        .isStandalone()
        .mutatesArg(0)
        .indicesArg(1)
        .dimAttr("dim")
        .outputConstraints(selfShapeContiguous)
        .registerOp();
    // masked_fill.Scalar(self, mask, value): scalar fill where mask is true.
    MetadataBuilder("torch.ops.aten.masked_fill.Scalar")
        .sizeOrdinal({0})
        .isStandalone()
        .mutatesArg(0)
        .indicesArg(1)
        .outputConstraints(selfShapeContiguous)
        .registerOp();
    // masked_fill.Tensor(self, mask, value): tensor fill where mask is true.
    MetadataBuilder("torch.ops.aten.masked_fill.Tensor")
        .sizeOrdinal({0})
        .isStandalone()
        .mutatesArg(0)
        .indicesArg(1)
        .valuesArg(2)
        .outputConstraints(selfShapeContiguous)
        .registerOp();
    // masked_scatter(self, mask, source): write source where mask is true.
    MetadataBuilder("torch.ops.aten.masked_scatter.default")
        .sizeOrdinal({0})
        .isStandalone()
        .mutatesArg(0)
        .indicesArg(1)
        .valuesArg(2)
        .outputConstraints(selfShapeContiguous)
        .registerOp();
    // select_scatter(self, src, dim, index): write src into self.select(dim,
    // index). For a statically-known dim 0 or 1, insert a clone of self and
    // retarget to the in-place tw.select_scatter (which writes the clone);
    // clone-elision drops the clone when self is dead (write in place) and
    // keeps it as a defensive copy when self is still live. Other dims / a
    // dynamic dim fall back to the eager standalone.
    MetadataBuilder("torch.ops.aten.select_scatter.default")
        .sizeOrdinal({0})
        .isStandalone()
        .mutatesArg(0)
        .valuesArg(1)
        .dimAttr("dim")
        .outputConstraints(selfShapeContiguous)
        .maybeReplace(
            [](NodeCP node, ValueTypes& types, WaveGraph& waveGraph)
                -> std::vector<std::pair<ValueCP, ValueCP>> {
              int64_t scatterDim = constIntArg(node, "dim").value_or(0);
              if (hasDynamicArg(node, "dim") ||
                  (scatterDim != 0 && scatterDim != 1)) {
                return {};
              }
              if (!insertSelfCloneForInPlace(
                      node, types, waveGraph, "select_scatter_clone")) {
                return {};
              }
              const_cast<nativert::Node*>(node)->setTarget("tw.select_scatter");
              return {};
            })
        .registerOp();
    // diagonal_scatter(self, src, offset, dim1, dim2): write src into self's
    // diagonal.
    MetadataBuilder("torch.ops.aten.diagonal_scatter.default")
        .sizeOrdinal({0})
        .isStandalone()
        .mutatesArg(0)
        .valuesArg(1)
        .outputConstraints(selfShapeContiguous)
        .registerOp();
    // scatter_add(self, dim, index, src): accumulate src along dim at index.
    // For a statically-known dim 0 or 1 with matching index/src shapes and an
    // atomic-add-able dtype, insert a clone of self and retarget to the
    // in-place tw.scatter_add; clone-elision drops the clone when self is dead
    // but keeps it when src shares a base with self (accumulating in place
    // would read partially updated values). Other cases fall back to the eager
    // standalone.
    MetadataBuilder("torch.ops.aten.scatter_add.default")
        .sizeOrdinal({0})
        .isStandalone()
        .mutatesArg(0)
        .indicesArg(1)
        .valuesArg(2)
        .dimAttr("dim")
        .outputConstraints(selfShapeContiguous)
        .maybeReplace(
            [](NodeCP node, ValueTypes& types, WaveGraph& waveGraph)
                -> std::vector<std::pair<ValueCP, ValueCP>> {
              tryFuseScatter(node, types, waveGraph, /*accumulate=*/true);
              return {};
            })
        .registerOp();
    // scatter.value(self, dim, index, value): scalar scatter along dim.
    MetadataBuilder("torch.ops.aten.scatter.value")
        .sizeOrdinal({0})
        .isStandalone()
        .mutatesArg(0)
        .indicesArg(1)
        .dimAttr("dim")
        .outputConstraints(selfShapeContiguous)
        .registerOp();
  }
  // aten.bucketize.Tensor and aten.searchsorted.Tensor are registered earlier
  // with a maybeReplace that retargets them to the fused tw.bucketize /
  // tw.searchsorted (falling back to the standalone eager op for the
  // unsupported cases). The registry keys on the op name and keeps the last
  // registration, so they must not be re-registered as plain standalone here.
  // sort: (values, indices), both with self's shape.
  MetadataBuilder("torch.ops.aten.sort.default")
      .sizeOrdinal({0})
      .isStandalone()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            ValueConstraint c{
                .rank = types.rank(inputAt(node, 0)),
                .contiguity = Contiguity::kContiguous};
            return {c, c};
          })
      .registerOp();
  // sort.stable: same (values, indices) shapes as sort.default.
  MetadataBuilder("torch.ops.aten.sort.stable")
      .sizeOrdinal({0})
      .isStandalone()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            ValueConstraint c{
                .rank = types.rank(inputAt(node, 0)),
                .contiguity = Contiguity::kContiguous};
            return {c, c};
          })
      .registerOp();
  // bincount: output length is data-dependent (max(input) + 1, at least
  // minlength), so it splits into a max head -- whose result sizes the output
  // on the host -- and a final that clears the bins and scatters one atomic add
  // per input element. Weighted bincount (a weights tensor) is left eager for
  // now; the fused path counts.
  MetadataBuilder("torch.ops.aten.bincount.default")
      .sizeOrdinal({0})
      .outputConstraints(rank1Constraint)
      .maybeReplace(
          [](NodeCP node,
             ValueTypes& /*types*/,
             WaveGraph& waveGraph) -> std::vector<std::pair<ValueCP, ValueCP>> {
            if (const auto* w = node->tryGetInput("weights")) {
              if (w->value &&
                  w->value->type().kind() != nativert::Type::Kind::None) {
                return {};
              }
            }
            auto* graph = waveGraph.graph();
            auto* input = node->inputs()[0].value;
            int64_t minlength = 0;
            if (const auto* m = node->tryGetAttribute("minlength")) {
              if (std::holds_alternative<int64_t>(m->value)) {
                minlength = std::get<int64_t>(m->value);
              }
            }
            auto* headNode =
                graph->createNode("tw.bincount_head", {{"input", input}});
            auto* maxOut = waveGraph.newScalarValue(
                headNode, "bincount_max", c10::ScalarType::Int);
            auto* finalNode = graph->createNode(
                "tw.bincount_final", {{"input", input}, {"maxval", maxOut}});
            finalNode->addAttribute({"minlength", minlength});
            finalNode->addAttribute({"dtype", c10::ScalarType::Long});
            waveGraph.declareMultiplyReferencedInput(input);
            auto* resultOutput = waveGraph.newTensorValue(
                finalNode, "bincount", c10::ScalarType::Long);
            return {{node->outputs()[0], resultOutput}};
          })
      .registerOp();

  // tw.bincount_head: max over the input; the result (an int, read on the host)
  // sizes the output. Single block.
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.bincount_head",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get())},
          std::vector<c10::Argument>{
              c10::Argument("maxval", c10::IntType::get())}))
      .sizeOrdinal({0})
      .hasBarrier()
      .returnMeta({{.neededOnHost = true}})
      .headerFile(kScanHeader)
      .deviceFunc("bincount_head")
      .sharedDecls(
          {{"Int32X32", "warpSums"},
           {"uint32_t", "size"},
           {"uint32_t", "rounded"}})
      .typeTemplateParams({0})
      .hasBlockSizeTemplateParam()
      .alwaysSingleBlock()
      .registerOp();

  // Output length = max(maxval + 1, minlength). An empty input reduces to
  // INT_MIN in the head, so maxval + 1 is negative and minlength wins.
  // minlength is read from the actual node (via nodeMap): bincount_final nodes
  // over the same input dedup to one kernel, so reading the captured formal
  // node's minlength would size every deduped instance from the first one's.
  auto bincountFinalReserve =
      [](NodeCP node,
         nativert::ExecutionFrame& frame,
         const FormalToActual& map,
         NodeCP originalFormalNode,
         const NodeMap& nodeMap) -> std::vector<std::vector<Dim>> {
    int64_t maxval =
        paramSymInt(node->tryGetInput("maxval")->value, frame, map);
    int64_t minlength =
        paramIntByName(originalFormalNode, "minlength", frame, map, nodeMap);
    int64_t outSize = std::max<int64_t>(maxval + 1, minlength);
    if (outSize < 0) {
      outSize = 0;
    }
    return {{static_cast<Dim>(outSize)}};
  };

  // tw.bincount_final: single-block clear + atomic-add scatter into the bins.
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.bincount_final",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get()),
              c10::Argument("maxval", c10::IntType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .ignoreAttrs({"minlength"})
      .returnMeta({{.isRegister = false, .reserveShape = bincountFinalReserve}})
      .inputFromPreviousKernel(1)
      .headerFile(kScanHeader)
      .deviceFunc("bincount_final")
      .sharedDecls({{"uint32_t", "size"}})
      .typeTemplateParams({0})
      .hasDtypeTemplateParam()
      .hasBarrier()
      .hasBlockSizeTemplateParam()
      // Single block: the device fn clears and scatters over the whole input
      // via threadIdx, so multiple blocks would each redo the full scatter and
      // overcount. Multi-block parallelism is the cg variant.
      .alwaysSingleBlock()
      .cgVariant(makeBincountFinalCgVariant)
      .outputConstraints(rank1Constraint)
      .registerOp();

  // tw.bincount_final_cg: cooperative-grid variant (opBarrier between clear and
  // scatter).
  MetadataBuilder(
      std::make_unique<c10::FunctionSchema>(
          "tw.bincount_final_cg",
          "",
          std::vector<c10::Argument>{
              c10::Argument("input", c10::TensorType::get()),
              c10::Argument("maxval", c10::IntType::get())},
          std::vector<c10::Argument>{
              c10::Argument("output", c10::TensorType::get())}))
      .sizeOrdinal({0})
      .ignoreAttrs({"minlength"})
      .returnMeta({{.isRegister = false, .reserveShape = bincountFinalReserve}})
      .inputFromPreviousKernel(1)
      .headerFile(kScanHeader)
      .deviceFunc("bincount_final_cg")
      .sharedDecls({{"uint32_t", "size"}})
      .typeTemplateParams({0})
      .hasDtypeTemplateParam()
      .hasBarrier()
      .hasBlockSizeTemplateParam()
      .numBarriers(2)
      .outputConstraints(rank1Constraint)
      .registerOp();

  // --- Shape-building / factory (eager standalone) ---

  // repeat (aten.repeat): tiles self along each dim. Fused as an elementwise
  // gather by MODULO of the output coordinate (like tw.index_select, but no
  // index tensor -- see __repeat). The output dim d is repeats[d] *
  // (rank-aligned self dim), with len(repeats) - self.rank leading 1-dims when
  // repeats is longer than self's rank. That shape is not derivable from self
  // alone (it depends on the constant 'repeats'), so it is computed at launch
  // by repeatReserve; sizeFromOutput + hasOutputArg pass the enclosing
  // expression's output tensor so __repeat knows the shape it iterates over.
  // 'self' is read as a whole tensor at computed offsets (randomAccess, so it
  // is materialized before this op). 'repeats' is a constant list attribute
  // used only by the host-side shape math, so it is skipped as a kernel
  // argument.
  {
    // 'repeats' is read via paramIntListByName from the actual node (resolved
    // through nodeMap), not the captured formal node: repeat ops that share an
    // input value dedup to one kernel (correctly -- __repeat uses out->dims,
    // not repeats), so reading the formal node's repeats would size every
    // deduped instance from the first one's repeats. self's runtime dims come
    // from the frame via 'map', as in indexSelectReserve.
    auto repeatReserve =
        [](NodeCP node,
           nativert::ExecutionFrame& frame,
           const FormalToActual& map,
           NodeCP originalFormalNode,
           const NodeMap& nodeMap) -> std::vector<std::vector<Dim>> {
      auto* selfValue = node->inputs()[0].value;
      auto selfId = selfValue->id();
      auto selfActual = map.find(selfId);
      auto& selfTensor =
          frame.getIValue(selfActual != map.end() ? selfActual->second : selfId)
              .toTensor();
      auto repeats = paramIntListByName(
          originalFormalNode, "repeats", frame, map, nodeMap);
      int64_t n = static_cast<int64_t>(repeats.size());
      int64_t rank = selfTensor.dim();
      int64_t dimOffset = n - rank;
      std::vector<Dim> shape(n);
      for (int64_t d = 0; d < n; ++d) {
        int64_t selfDim = d < dimOffset ? 1 : selfTensor.sizes()[d - dimOffset];
        shape[d] = static_cast<Dim>(repeats[d] * selfDim);
      }
      return {std::move(shape)};
    };

    MetadataBuilder("torch.ops.aten.repeat.default")
        .elementwiseFunc("__repeat")
        .ignoreAttrs({"repeats"})
        .hasIdxArg()
        .hasOutputArg()
        .hasBlockInfo()
        // repeat's output has a different element count than self, so it must
        // not reuse self's buffer as its output.
        .inPlaceIfLastUse(false)
        .sizeOrdinal({0})
        .outputConstraints(
            [constAttrList](
                NodeCP node,
                const ValueTypes& /*types*/) -> std::vector<ValueConstraint> {
              const auto* repeats = constAttrList(node, "repeats");
              if (!repeats) {
                return {};
              }
              return {
                  {.rank = static_cast<int8_t>(repeats->size()),
                   .contiguity = Contiguity::kContiguous}};
            })
        .argumentMeta(
            {{.isRegister = false, .wholeTensor = true, .randomAccess = true},
             {.isRegister = true}})
        .returnMeta({{.isRegister = true, .reserveShape = repeatReserve}})
        .typeTemplateParams({0})
        // The output shape (repeats * rank-aligned self dims) is not derivable
        // from the operands, so when fused the output is kept as a shape-only
        // tensor and the enclosing expression is sized from it.
        .sizeFromOutput()
        // A repeat of all ones whose length equals self's rank is a true
        // identity (an eager copy); replace it with self. A longer repeats list
        // (len > rank) adds leading 1-dims and reshapes, so it is not an
        // identity and falls through to the fused op.
        .maybeReplace(
            [constAttrList](
                NodeCP node, ValueTypes& types, WaveGraph& /*waveGraph*/)
                -> std::vector<std::pair<ValueCP, ValueCP>> {
              const auto* repeats = constAttrList(node, "repeats");
              if (!repeats) {
                return {};
              }
              if (static_cast<int64_t>(repeats->size()) !=
                  types.rank(inputAt(node, 0))) {
                return {};
              }
              for (auto r : *repeats) {
                if (r != 1) {
                  return {};
                }
              }
              return {{node->outputs()[0], node->inputs()[0].value}};
            })
        .registerOp();
  }
  // full_like: self-shaped tensor filled with a scalar. Like ones_like, but the
  // fill value arrives as the "fill_value" scalar attribute param consumed by
  // __full. self is used only for the output shape, so it is skipped during
  // argument iteration (ignoreAttrs) and read via rankArgument/reserveShape.
  MetadataBuilder("torch.ops.aten.full_like.default")
      .elementwiseFunc("__full")
      .numArgs(1)
      .ignoreAttrs({"self"})
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
            },
            .shapeFromInput = 0}})
      .normalize(resolveDtypeFromInputExact)
      .rankArgument(0)
      .hasDtypeTemplateParam()
      .registerOp();
  // new_full / new_ones: new tensor of the given size; rank = size length.
  auto newSizedConstraint =
      [constAttrList](
          NodeCP node,
          const ValueTypes& /*types*/) -> std::vector<ValueConstraint> {
    const auto* size = constAttrList(node, "size");
    if (!size) {
      return {};
    }
    return {
        {.rank = static_cast<int8_t>(size->size()),
         .contiguity = Contiguity::kContiguous}};
  };
  // new_full: when the size is a compile-time, non-empty attribute, rewrite to
  // the elementwise aten.full (mirrors new_ones -> ones). A dynamic-size,
  // empty, or dynamic-fill new_full keeps the eager standalone.
  MetadataBuilder("torch.ops.aten.new_full.default")
      .sizeOrdinal({0})
      .isStandalone()
      .outputConstraints(newSizedConstraint)
      .maybeReplace(
          [](NodeCP node,
             ValueTypes& /*types*/,
             WaveGraph& waveGraph) -> std::vector<std::pair<ValueCP, ValueCP>> {
            const auto* sizeAttr = node->tryGetAttribute("size");
            const auto* fillAttr = node->tryGetAttribute("fill_value");
            if (!sizeAttr || !fillAttr) {
              // Dynamic size or fill: leave the node for the eager standalone.
              return {};
            }
            // The Constant variant is move-only (it has a unique_ptr<Graph>
            // alternative), so the fill is copied by concrete type below; only
            // int/float/bool fills are supported.
            const bool intFill =
                std::holds_alternative<int64_t>(fillAttr->value);
            const bool doubleFill =
                std::holds_alternative<double>(fillAttr->value);
            const bool boolFill = std::holds_alternative<bool>(fillAttr->value);
            if (!intFill && !doubleFill && !boolFill) {
              return {};
            }
            const auto& sizeVec =
                std::get<std::vector<int64_t>>(sizeAttr->value);
            int64_t numel = 1;
            for (auto dim : sizeVec) {
              numel *= dim;
            }
            if (numel == 0) {
              // Empty result feeds the empty-input scan/masked_select fusion
              // path, which faults; keep the eager standalone.
              return {};
            }
            auto outDtype = resolveOutDtype(node, &waveGraph).first;
            auto* graph = waveGraph.graph();
            auto* fullNode =
                graph->createNode("torch.ops.aten.full.default", {});
            fullNode->addAttribute({sizeAttr->name, sizeVec});
            if (intFill) {
              fullNode->addAttribute(
                  {fillAttr->name, std::get<int64_t>(fillAttr->value)});
            } else if (doubleFill) {
              fullNode->addAttribute(
                  {fillAttr->name, std::get<double>(fillAttr->value)});
            } else {
              fullNode->addAttribute(
                  {fillAttr->name, std::get<bool>(fillAttr->value)});
            }
            fullNode->addAttribute({"dtype", outDtype});
            // new_full inherits its input's device; the rewritten full has no
            // such input, so pin it to the wave (GPU) device.
            if (auto* dev = facebook::velox::wave::currentDevice()) {
              fullNode->addAttribute(
                  {"device",
                   c10::Device(
                       c10::kCUDA,
                       static_cast<c10::DeviceIndex>(dev->deviceId))});
            }
            graph->insertBefore(fullNode, const_cast<nativert::Node*>(node));
            auto* newOutput =
                waveGraph.newTensorValue(fullNode, "full", outDtype);
            return {{node->outputs()[0], newOutput}};
          })
      .registerOp();
  // New_ones: when the size is a compile-time attribute, rewrite to the
  // elementwise aten.ones so the fill fuses (mirrors new_zeros -> zeros). Some
  // new_ones take the size as a dynamic value input instead; aten.ones cannot
  // represent that, so those keep the eager standalone (isStandalone +
  // newSizedConstraint) as the fallback.
  MetadataBuilder("torch.ops.aten.new_ones.default")
      .sizeOrdinal({0})
      .isStandalone()
      .outputConstraints(newSizedConstraint)
      .maybeReplace(
          [](NodeCP node,
             ValueTypes& /*types*/,
             WaveGraph& waveGraph) -> std::vector<std::pair<ValueCP, ValueCP>> {
            const auto* sizeAttr = node->tryGetAttribute("size");
            if (!sizeAttr) {
              // Dynamic size: leave the node for the eager standalone.
              return {};
            }
            const auto& sizeVec =
                std::get<std::vector<int64_t>>(sizeAttr->value);
            int64_t numel = 1;
            for (auto dim : sizeVec) {
              numel *= dim;
            }
            if (numel == 0) {
              // Empty result: an elementwise ones with zero elements feeds the
              // empty-input scan/masked_select fusion path, which faults. Keep
              // the eager standalone for empty factories.
              return {};
            }
            auto outDtype = resolveOutDtype(node, &waveGraph).first;
            auto* graph = waveGraph.graph();
            auto* onesNode =
                graph->createNode("torch.ops.aten.ones.default", {});
            onesNode->addAttribute({sizeAttr->name, sizeVec});
            onesNode->addAttribute({"dtype", outDtype});
            // new_ones inherits its input's device; the rewritten ones has no
            // such input, so pin it to the wave (GPU) device.
            if (auto* dev = facebook::velox::wave::currentDevice()) {
              onesNode->addAttribute(
                  {"device",
                   c10::Device(
                       c10::kCUDA,
                       static_cast<c10::DeviceIndex>(dev->deviceId))});
            }
            graph->insertBefore(onesNode, const_cast<nativert::Node*>(node));
            auto* newOutput =
                waveGraph.newTensorValue(onesNode, "ones", outDtype);
            return {{node->outputs()[0], newOutput}};
          })
      .registerOp();
  // pow.Scalar: scalar base raised to a tensor of exponents. self is a scalar
  // attribute, so the only tensor input (exponent) is at position 0.
  MetadataBuilder("torch.ops.aten.pow.Scalar")
      .sizeOrdinal({0})
      .isStandalone()
      .outputConstraints(
          [](NodeCP node,
             const ValueTypes& types) -> std::vector<ValueConstraint> {
            return {
                {.rank = types.rank(inputAt(node, 0)),
                 .contiguity = Contiguity::kContiguous}};
          })
      .registerOp();
  // _local_scalar_dense: Tensor.item(). torch.export emits this synonym of
  // aten.item, so register it fused exactly like aten.item -- it reads self[0]
  // (__item) into a scalar register, rather than a standalone's zero-dim
  // tensor, so it can feed scalar-size args (e.g. a dynamic arange end).
  MetadataBuilder("torch.ops.aten._local_scalar_dense.default")
      .elementwiseFunc("__item")
      .inPlaceIfLastUse(false)
      .numArgs(1)
      .argumentMeta({{.isRegister = false, .wholeTensor = true}})
      .returnMeta({ArgumentMeta{.isRegister = true}})
      .typeTemplateParams({0})
      .registerOp();
}

} // namespace torch::wave
