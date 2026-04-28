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

#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/torchwave/Utils.h"

namespace torch::wave {

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

  // Activation functions.
  Registry::registerElementwise("torch.ops.aten.relu.default");
  Registry::registerElementwise("torch.ops.aten.sigmoid.default");
  Registry::registerElementwise("torch.ops.aten.clamp.default", {"min", "max"});

  // Min/max.
  Registry::registerElementwise("torch.ops.aten.minimum.default");
  Registry::registerElementwise("torch.ops.aten.maximum.default");

  // Shape query.
  Registry::registerElementwiseOp(
      "torch.ops.aten.sym_size.int", "sym_size", false, {"dim"});

  static const std::string kScanHeader =
      "velox/experimental/torchwave/Scan.cuh";

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
    md.returnMeta = {
        {.isRegister = false,
         .reserveShape =
             [](NodeCP node,
                nativert::ExecutionFrame& frame,
                FormalToActual map) -> std::vector<std::vector<Dim>> {
           auto tensor = paramTensor(node->inputs()[0].value, frame, map);
           auto numel = tensor.numel();
           auto numBlocks = static_cast<Dim>((numel + 255) / 256);
           return {{numBlocks}};
         }},
    };
    md.headerFile = kScanHeader;
    md.deviceFunc = "masked_select_head";
    md.sharedDecls = {
        {"Int32X32", "warpSums"},
        {"uint32_t", "size"},
        {"uint32_t", "rounded"}};
    md.typeTemplateParams = {0};
    md.hasBlockSizeTemplateParam = true;
    md.kernelBreakForMultiblock = true;
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
         .reserveShape =
             [](NodeCP node,
                nativert::ExecutionFrame& frame,
                FormalToActual map) -> std::vector<std::vector<Dim>> {
           // Total is last element of prefix-summed counts tensor.
           auto tensor = paramTensor(node->inputs()[0].value, frame, map);
           auto sizes = tensor.sizes();
           return {{sizes.begin(), sizes.end()}};
         },
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
    md.hasBlockSizeTemplateParam = true;
    Registry::registerMetadata("tw.masked_select_final", std::move(md));
  }
}

} // namespace torch::wave
