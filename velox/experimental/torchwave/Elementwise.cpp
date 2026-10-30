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

#include <fmt/format.h>
#include "velox/experimental/torchwave/Compile.h"
#include "velox/experimental/torchwave/StaticDims.h"
#include "velox/experimental/torchwave/Utils.h"

#include "velox/experimental/torchwave/WaveConfig.h"

#include <folly/ScopeGuard.h>
#include <folly/container/F14Set.h>
#include <algorithm>
#include <limits>
#include <set>
#include <sstream>

namespace torch::wave {

void CompileCtx::collectSubgraphInputs(
    NodeCP node,
    const std::unordered_set<ValueCP>& sgInputs,
    std::unordered_set<ValueCP>& seen,
    std::vector<ValueCP>& result) const {
  auto* meta = nodeMeta(node);
  for (size_t i = 0; i < node->inputs().size(); ++i) {
    auto* value = node->inputs()[i].value;
    auto* producer = value->producer();
    // When a placed prim.ListPack has isRegister in the consumer's
    // argument meta, expand to its individual tensor elements so they
    // appear in the ElementExpr parameter set.
    const auto* listArgMeta = argMetaForInput(meta, node, i);
    if (producer && placed_.count(producer) &&
        producer->target() == "prim.ListPack" && listArgMeta &&
        listArgMeta->isRegister) {
      for (auto& listInput : producer->inputs()) {
        auto* lv = listInput.value;
        auto* lp = lv->producer();
        if (sgInputs.count(lv) || !lp || placed_.count(lp)) {
          if (seen.insert(lv).second) {
            result.push_back(lv);
          }
        } else {
          collectSubgraphInputs(lp, sgInputs, seen, result);
        }
      }
      continue;
    }
    // A producerless value is always a leaf of the subgraph (a graph input or
    // a formal boundary of the ProjectOperation's variant subgraph, e.g. an
    // index gather materialized in an earlier kernel and imported here through
    // a prim.ListPack). It has a kernel param slot, so it must appear in the
    // ElementExpr inputs; extractSubgraphInputs already treats it this way.
    // Without the !producer case it would be dropped, and elementwiseExprImpl
    // (which sees it as terminal) would fail to find it in the inputs vector.
    if (sgInputs.count(value) || !producer || placed_.count(producer)) {
      if (seen.insert(value).second) {
        result.push_back(value);
      }
    } else {
      collectSubgraphInputs(producer, sgInputs, seen, result);
    }
  }
}

std::vector<ValueCP> CompileCtx::subgraphInputs(
    const std::vector<Subgraph>& subgraphs) const {
  std::unordered_set<ValueCP> seen;
  std::vector<ValueCP> result;
  for (auto& sg : subgraphs) {
    std::unordered_set<ValueCP> sgInputSet(sg.inputs.begin(), sg.inputs.end());
    collectSubgraphInputs(sg.root, sgInputSet, seen, result);
  }
  return result;
}

void CompileCtx::generateIndexToOffset(
    const ElementExpr& ee,
    const std::vector<ValueCP>& /*allInputs*/) {
  auto& op = *generatingOp_;
  std::vector<int32_t> paramOffs;
  std::vector<int32_t> outputOffs;
  std::vector<int32_t> altOffs;

  int32_t outputOff = op.paramOffset(ee.output);
  for (auto* v : ee.inputs) {
    if (v->type().kind() != nativert::Type::Kind::Tensor) {
      continue;
    }
    paramOffs.push_back(op.paramOffset(v));
    outputOffs.push_back(outputOff);
    auto ait = ee.altParamOffset.find(v);
    altOffs.push_back(ait != ee.altParamOffset.end() ? ait->second : -1);
  }
  if (ee.output->type().kind() == nativert::Type::Kind::Tensor) {
    paramOffs.push_back(outputOff);
    outputOffs.push_back(outputOff);
    altOffs.push_back(-1);
  }

  if (paramOffs.empty()) {
    return;
  }

  auto emitArray = [&](const char* name, const std::vector<int32_t>& arr) {
    code_ << "    static int32_t " << name << "[] = {";
    for (size_t i = 0; i < arr.size(); ++i) {
      if (i > 0) {
        code_ << ", ";
      }
      code_ << arr[i];
    }
    code_ << "};\n";
  };
  code_ << "  {\n";
  emitArray("paramOffsets", paramOffs);
  emitArray("outputOffsets", outputOffs);
  emitArray("altOffsets", altOffs);
  code_
      << "    for (auto i = threadIdx.x; i < sizeof(paramOffsets) / sizeof(paramOffsets[0]); i += blockDim.x) {\n"
      << "      if (altOffsets[i] != -1) {\n"
      << "        copyTensorHead(param<Tensor>(blockInfo, paramOffsets[i]), param<Tensor>(blockInfo, altOffsets[i]));\n"
      << "        param<Tensor>(blockInfo, altOffsets[i])->init<true>(param<Tensor>(blockInfo, outputOffsets[i]));\n"
      << "      } else {\n"
      << "        param<Tensor>(blockInfo, paramOffsets[i])->init<true>(outputOffsets[i] != paramOffsets[i] ? param<Tensor>(blockInfo, outputOffsets[i]) : nullptr);\n"
      << "      }\n"
      << "    }\n"
      << "  }\n"
      << "  __syncthreads();\n";
  noteBlockBarrier();
}

// Generates CUDA code for elementwise operations (result[i] = f(a[i], b[i],
// ...)). Fast path: all operands are contiguous 1D - a single loop combines
// elements at each index. Slow path: some operands have broadcast dimensions or
// non-contiguous strides. A per-operand bitmask (isFastPath) marks which
// operands need index translation from the linear index to per-array offsets.
// The full-block variant calls the consumer on blockDim.x-aligned iterations
// (needed when the consumer has barriers). The non-full-block variant skips
// alignment padding.
namespace {

// True for a view inside an elementwise expression, which the expression
// computes through (see CompileCtx::fusesThroughView). A view that is itself
// elementwise, like expand, is generated as an elementwise op.
bool isFusedThroughView(const Metadata* meta) {
  return meta != nullptr && meta->isView() && meta->elementwise == nullptr;
}

void addViewLeafRefs(
    ValueCP value,
    ValueCP ref,
    const std::unordered_set<ValueCP>& leaves,
    const std::unordered_set<ValueCP>& wholeTensors,
    folly::F14FastMap<ValueCP, ValueCP>& refs,
    folly::F14FastSet<ValueCP>& rootReads) {
  if (leaves.count(value) || value->producer() == nullptr) {
    if (value->type().kind() != nativert::Type::Kind::Tensor ||
        wholeTensors.count(value)) {
      return;
    }
    if (ref == nullptr) {
      rootReads.insert(value);
      return;
    }
    auto [it, inserted] = refs.emplace(value, ref);
    TORCH_CHECK(
        inserted || it->second == ref,
        "Leaf read below two views of one expression: %",
        value->id());
    return;
  }
  auto* node = value->producer();
  const auto* meta = nodeMeta(node);
  if (isFusedThroughView(meta)) {
    auto* viewInput = node->inputs()[*meta->viewOfArg].value;
    addViewLeafRefs(
        viewInput, viewInput, leaves, wholeTensors, refs, rootReads);
    return;
  }
  for (const auto& input : node->inputs()) {
    addViewLeafRefs(input.value, ref, leaves, wholeTensors, refs, rootReads);
  }
}

// The test that every per-element tensor leaf is contiguous and has the loop's
// element count, so that each reads at idx.
std::string fastPathCondition(int32_t numFastPathVars, int32_t tensorCount) {
  if (numFastPathVars == 0) {
    return "true";
  }
  std::string condition;
  for (int32_t i = 0; i < numFastPathVars; ++i) {
    uint32_t mask{0xffffffff};
    const auto bitsInLast = tensorCount - i * kBitsPerWord;
    if (i == numFastPathVars - 1 && bitsInLast < kBitsPerWord) {
      mask = (1u << bitsInLast) - 1;
    }
    if (i > 0) {
      condition += " && ";
    }
    condition += fmt::format("isFastPath{} == 0x{:x}", i, mask);
  }
  return condition;
}

// The leaves of the expression rooted at 'root' that are read below a view it
// computes through, each mapped to that view's input.
folly::F14FastMap<ValueCP, ValueCP> viewLeafRefs(
    ValueCP root,
    const std::unordered_set<ValueCP>& leaves,
    const std::unordered_set<ValueCP>& wholeTensors) {
  folly::F14FastMap<ValueCP, ValueCP> refs;
  folly::F14FastSet<ValueCP> rootReads;
  // 'root' is itself among 'leaves', as the value the loop stores.
  for (const auto& input : root->producer()->inputs()) {
    addViewLeafRefs(
        input.value, nullptr, leaves, wholeTensors, refs, rootReads);
  }
  for (const auto& [leaf, ref] : refs) {
    TORCH_CHECK(
        !rootReads.count(leaf),
        "Leaf read both below and beside a view of one expression: %",
        leaf->id());
  }
  return refs;
}

} // namespace

void CompileCtx::generateElementwise(
    const std::vector<Subgraph>& subgraphs,
    const std::vector<ResultSpec>& resultSpecs,
    const std::string& resultStmt,
    bool fullBlockResult,
    ValueCP FOLLY_NULLABLE shapeSetOnDeviceResult,
    NodeCP FOLLY_NULLABLE rowReduction) {
  auto& op = *generatingOp_;

  // Scalar-elementwise ops (isScalarElementwise, e.g. _operator.* / sym_size /
  // sym_numel) produce a naked scalar, not a tensor: the loop runs a single
  // element (the kernel has no whole-tensor operand, so the size head computes
  // size == 1) and the result is written to a scalar parameter. Verify the
  // result Value really is a scalar so a misregistration is caught at compile
  // time rather than corrupting memory.
  for (const auto& sg : subgraphs) {
    auto* rootMeta = nodeMeta(sg.root);
    if (rootMeta && rootMeta->isScalarElementwise &&
        !sg.root->outputs().empty()) {
      auto kind = sg.root->outputs()[0]->type().kind();
      TORCH_CHECK(
          kind != nativert::Type::Kind::Tensor &&
              kind != nativert::Type::Kind::TensorList,
          "isScalarElementwise op ",
          sg.root->target(),
          " must produce a scalar result, not a tensor");
    }
  }

  // Get unique leaf inputs by walking subgraph roots.
  auto leafInputs = subgraphInputs(subgraphs);

  // Remove values whose producer is an elementwise op with non-register
  // first return (e.g. index_put_elt_*) when the value is computed inline in
  // this kernel: its output is sized by the source tensor, not the elementwise
  // loop, and it is re-derived by walking its producer. A value that has a
  // param slot in this kernel is different: it is backed by memory, either as a
  // boundary input (materialized by an earlier kernel) or as a materialized
  // output of an earlier sub-expression in this same kernel (e.g. an index_put
  // result that a later __or__ then reads). Such a value must stay a leaf.
  // Dropping it would leave it out of inputSet, making elementwiseExprImpl
  // treat it as non-terminal and recurse into the already-placed producer --
  // reaching values (an index_put self clone, or a ListPack index element from
  // a previous kernel) that have no param slot in the enclosing sub-expression.
  leafInputs.erase(
      std::remove_if(
          leafInputs.begin(),
          leafInputs.end(),
          [&op](ValueCP v) {
            if (op.hasParamSlot(v)) {
              return false;
            }
            auto* producer = v->producer();
            if (!producer) {
              return false;
            }
            auto* meta = nodeMeta(producer);
            return meta && meta->elementwise && !meta->returnMeta.empty() &&
                !meta->returnMeta[0].isRegister;
          }),
      leafInputs.end());

  // Build ElementExprs early so altParamOffset is available during code gen.
  std::vector<ElementExpr> newExprs;
  for (size_t s = 0; s < subgraphs.size(); ++s) {
    ElementExpr ee;
    // There is an output Value (shape only) also when generating an element
    // wise expr that produces values in registers.
    ee.output = subgraphs[s].root->outputs()[0];
    ee.inputs = leafInputs;
    {
      auto* producer = ee.output->producer();
      if (producer && op.allNodes().count(producer)) {
        for (const auto& desc : op.outputDescs()) {
          if (desc.shapeSetOnDevice) {
            ee.shapeFromThisOp = true;
            break;
          }
        }
      }
    }
    std::unordered_set<ValueCP> existingInputs;
    for (const auto& prev : op.elementExprs()) {
      for (auto* v : prev.inputs) {
        existingInputs.insert(v);
      }
    }
    for (size_t p = 0; p < s; ++p) {
      for (auto* v : newExprs.at(p).inputs) {
        existingInputs.insert(v);
      }
    }
    for (auto* v : ee.inputs) {
      if (v->type().kind() == nativert::Type::Kind::Tensor &&
          existingInputs.count(v)) {
        ee.altParamOffset[v] = op.allocAltTensor();
      }
    }
    newExprs.push_back(std::move(ee));
  }

  // The leaves as they stand before the result values are appended below.
  // These, and only these, are the values the generated expressions read from
  // memory; everything else in the tree is a register temp. The barrier check
  // further down needs the distinction.
  const std::unordered_set<ValueCP> memoryLeaves(
      leafInputs.begin(), leafInputs.end());

  for (size_t s = 0; s < resultSpecs.size(); ++s) {
    auto& rs = resultSpecs[s];
    if (!rs.value) {
      continue;
    }
    // Skip output tensors for ops with non-register return (e.g.
    // index_put_elt_*). Their output is a view of the self tensor and
    // has a different size than the elementwise loop — including it
    // would make the loop iterate over the wrong number of elements.
    if (s < subgraphs.size()) {
      auto* rootMeta = nodeMeta(subgraphs[s].root);
      if (rootMeta && !rootMeta->returnMeta.empty() &&
          !rootMeta->returnMeta[0].isRegister) {
        continue;
      }
    }
    if (std::find(leafInputs.begin(), leafInputs.end(), rs.value) ==
        leafInputs.end()) {
      leafInputs.push_back(rs.value);
    }
  }

  // Set currentElementExpr_ for the duration of code generation.
  const ElementExpr* prevElementExpr = currentElementExpr_;
  currentElementExpr_ = newExprs.empty() ? nullptr : newExprs.data();
  SCOPE_EXIT {
    currentElementExpr_ = prevElementExpr;
  };

  // Build allInputs = leafInputs + result values for storage declarations.
  std::unordered_set<ValueCP> seen(leafInputs.begin(), leafInputs.end());
  auto allInputs = leafInputs;
  for (auto& rs : resultSpecs) {
    if (rs.value && seen.insert(rs.value).second) {
      allInputs.push_back(rs.value);
    }
  }

  code_ << "  {\n";

  // Emit a comment with the subgraph text and param offsets for inputs.
  for (const auto& sg : subgraphs) {
    auto text = sg.toString();
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
      code_ << "  // " << line << "\n";
    }
  }
  for (auto* v : leafInputs) {
    if (v->type().kind() == nativert::Type::Kind::Tensor) {
      code_ << "  // %" << v->id() << " offset = " << op.paramOffset(v) << "\n";
    }
  }

  // Collect values marked as wholeTensor in all elementwise nodes in
  // the subgraph tree, not just the root.  When index_elt_* is fused
  // with downstream arithmetic, the root is the arithmetic op and the
  // wholeTensor source lives in an inner node.
  std::unordered_set<ValueCP> wholeTensorValues;
  std::unordered_set<ValueCP> leafInputSet(
      leafInputs.begin(), leafInputs.end());
  std::function<void(NodeCP)> collectWholeTensor = [&](NodeCP node) {
    auto* meta = nodeMeta(node);
    if (isFusedThroughView(meta)) {
      auto* viewInput = node->inputs()[*meta->viewOfArg].value;
      if (!leafInputSet.count(viewInput) && viewInput->producer()) {
        collectWholeTensor(viewInput->producer());
      }
      return;
    }
    if (!meta || !meta->elementwise) {
      return;
    }
    const auto& inputs = node->inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto* v = inputs[i].value;
      const ArgumentMeta* am = argMetaForInput(meta, node, i);
      if (am && am->wholeTensor) {
        wholeTensorValues.insert(v);
      }
      auto* producer = v->producer();
      if (producer && producer->target() == "prim.ListPack") {
        if (am && am->isRegister) {
          for (const auto& listInput : producer->inputs()) {
            if (!leafInputSet.count(listInput.value) &&
                listInput.value->producer()) {
              auto* lp = listInput.value->producer();
              if (nodeMeta(lp) && nodeMeta(lp)->elementwise) {
                collectWholeTensor(lp);
              }
            }
          }
        }
      } else if (!leafInputSet.count(v) && producer) {
        collectWholeTensor(producer);
      }
    }
  };
  for (const auto& sg : subgraphs) {
    collectWholeTensor(sg.root);
  }
  for (size_t s = 0; s < subgraphs.size(); ++s) {
    newExprs[s].leafRefs = viewLeafRefs(
        subgraphs[s].root->outputs()[0], leafInputSet, wholeTensorValues);
  }

  // Exclude output values whose return meta has both wholeTensor and
  // randomAccess (e.g. index_put_elt_* where the output is a view of
  // the randomly-accessed source tensor, sized differently from the
  // elementwise loop).
  std::unordered_set<ValueCP> sizeExcludedValues = wholeTensorValues;
  for (size_t s = 0; s < subgraphs.size() && s < resultSpecs.size(); ++s) {
    if (!resultSpecs[s].value) {
      continue;
    }
    auto* rootMeta = nodeMeta(subgraphs[s].root);
    if (rootMeta && !rootMeta->returnMeta.empty() &&
        rootMeta->returnMeta[0].wholeTensor &&
        rootMeta->returnMeta[0].randomAccess) {
      sizeExcludedValues.insert(resultSpecs[s].value);
    }
  }

  // Build tensor-only bit index for fast path processing, skipping
  // wholeTensor values which are not accessed per-element.
  int32_t tensorCount = 0;
  fastPathBitIndex_.assign(leafInputs.size(), -1);
  for (size_t i = 0; i < leafInputs.size(); ++i) {
    if (leafInputs[i]->type().kind() == nativert::Type::Kind::Tensor &&
        !wholeTensorValues.count(leafInputs[i])) {
      fastPathBitIndex_[i] = tensorCount++;
    }
  }
  // Generate shared declarations for size and fast path flags.
  op.addSharedDeclaration("  __shared__ uint32_t size;\n");
  auto numFastPathVars = (tensorCount + kBitsPerWord - 1) / kBitsPerWord;
  // Always declare at least isFastPath0: an elementwise kernel whose only
  // tensor operand is a whole tensor (e.g. sym_numel/sym_size fused into a
  // kernel) has tensorCount == 0 but still emits `isFastPath0 = 1` in the
  // size-1 branch below.
  if (numFastPathVars < 1) {
    numFastPathVars = 1;
  }
  std::vector<std::string> headScratch{"size"};
  for (size_t i = 0; i < numFastPathVars; ++i) {
    op.addSharedDeclaration(
        "  __shared__ uint32_t isFastPath" + std::to_string(i) + ";\n");
    headScratch.push_back("isFastPath" + std::to_string(i));
  }

  separateSharedScratch(headScratch);

  // With every operand at a static index, the loop bound is a literal and
  // there is no fast path to choose.
  const auto staticSize = staticLoopSize(
      leafInputs,
      wholeTensorValues,
      sizeExcludedValues,
      subgraphs,
      resultSpecs,
      shapeSetOnDeviceResult,
      op);
  if (staticSize.has_value()) {
    code_ << "  if (threadIdx.x == 0) {\n"
          << "    size = " << *staticSize << "u;\n";
    for (size_t i = 0; i < numFastPathVars; ++i) {
      code_ << "    isFastPath" << i << " = 0;\n";
    }
    code_ << "  }\n";
  }
  // Generate the head: compute size and fast path flags from tensor leaf
  // inputs only.
  if (!staticSize.has_value()) {
    code_ << "  if (threadIdx.x == 0) {\n";
    for (size_t i = 0; i < numFastPathVars; ++i) {
      code_ << "    isFastPath" << i << " = 0;\n";
    }
    bool firstTensor = true;
    bool declaredSize2 = false;
    for (size_t valueIdx = 0; valueIdx < leafInputs.size(); ++valueIdx) {
      if (leafInputs[valueIdx]->type().kind() != nativert::Type::Kind::Tensor) {
        continue;
      }
      if (wholeTensorValues.count(leafInputs[valueIdx])) {
        continue;
      }
      auto bitIdx = fastPathBitIndex_[valueIdx];
      auto W = bitIdx / kBitsPerWord;
      auto B = bitIdx % kBitsPerWord;
      bool skipSize = sizeExcludedValues.count(leafInputs[valueIdx]);
      if (skipSize) {
        code_ << "    isFastPath" << W << " |= (uint32_t)"
              << param(leafInputs[valueIdx], op) << "->contiguous << " << B
              << ";\n";
      } else if (firstTensor) {
        code_ << "    Tensor* temp = " << param(leafInputs[valueIdx], op)
              << ";\n    size = temp->numEl;\n"
              << "    isFastPath0 |= temp->contiguous;\n";
        firstTensor = false;
      } else {
        if (!declaredSize2) {
          code_ << "    uint32_t size2;\n";
          declaredSize2 = true;
        }
        code_ << "    temp = " << param(leafInputs[valueIdx], op) << ";\n"
              << "    size2 = temp->numEl;\n"
              << "    isFastPath" << W << " |= (uint32_t)temp->contiguous << "
              << B << ";\n"
              << "    if (size2 != size) {\n"
              << "      if (size2 > size) {\n";
        for (int32_t I = 0; (I + 1) * kBitsPerWord <= bitIdx; ++I) {
          code_ << "        isFastPath" << I << " = 0;\n";
        }
        code_ << "        isFastPath" << W << " &= ~((1 << " << B << ") - 1);\n"
              << "      size = size2;\n"
              << "      } else {\n"
              << "    isFastPath" << W << " &= ~(1 << " << B << ");\n"
              << "}"
              << "    }\n";
      }
    }
    if (firstTensor) {
      code_ << "    size = 1;\n"
            << "    isFastPath0 = 1;\n";
    }
    // The elementwise loop must cover the full output.  For an outer-product
    // broadcast ([N,1] + [1,M] -> [N,M]) the output exceeds every input, so the
    // max-input size computed above is too small.  Fold each memory-backed
    // tensor output's element count into `size`; if an output is larger, clear
    // input fast-path bits so every operand uses broadcast index translation.
    // A row reduction stores no element of the expression; its shape-only
    // root stands in for the output.
    std::vector<ValueCP> sizedOutputs;
    for (const auto& spec : resultSpecs) {
      sizedOutputs.push_back(spec.value);
    }
    if (rowReduction != nullptr) {
      sizedOutputs.push_back(subgraphs.at(0).root->outputs()[0]);
    }
    for (auto* outVal : sizedOutputs) {
      if (!outVal || outVal->type().kind() != nativert::Type::Kind::Tensor ||
          sizeExcludedValues.count(outVal) || wholeTensorValues.count(outVal)) {
        continue;
      }
      if (!declaredSize2) {
        code_ << "    uint32_t size2;\n";
        declaredSize2 = true;
      }
      code_ << "    size2 = " << param(outVal, op) << "->numEl;\n"
            << "    if (size2 == 0) {\n"
            << "      size = 0;\n"
            << "    } else if (size2 > size) {\n";
      for (size_t W = 0; W < numFastPathVars; ++W) {
        code_ << "      isFastPath" << W << " = 0;\n";
      }
      code_ << "      size = size2;\n"
            << "    }\n";
    }
    code_ << "  }\n";
  }
  code_ << "  __syncthreads();\n";

  // If anything the subgraphs read was written earlier in this same kernel
  // without a barrier since, emit one after the size setup.
  {
    bool needsBarrier = false;
    std::function<void(ValueCP)> checkTree = [&](ValueCP value) {
      if (needsBarrier || leafInputSet.count(value) || !value->producer()) {
        return;
      }
      auto* node = value->producer();
      // A view the expression computes through reads nothing from memory.
      if (!isFusedThroughView(nodeMeta(node)) && callNeedsBarrier(node)) {
        needsBarrier = true;
        return;
      }
      for (const auto& input : node->inputs()) {
        checkTree(input.value);
      }
    };
    // A leaf is a memory read like any call operand, and callNeedsBarrier
    // cannot see it: every argument of an elementwise op is registered
    // isRegister, which is true of a value fused inline into the expression
    // and false of one that reached memory. Nothing else covers a consumer
    // that reads, at index i, an element another block wrote -- which is what
    // a leaf that is a transposed view of a value this kernel just filled is.
    auto leavesNeedBarrier = [&](const Subgraph& sg) {
      const std::unordered_set<ValueCP> sgInputSet(
          sg.inputs.begin(), sg.inputs.end());
      std::unordered_set<ValueCP> visited;
      std::vector<ValueCP> leaves;
      collectSubgraphInputs(sg.root, sgInputSet, visited, leaves);
      for (auto* leaf : leaves) {
        if (memoryLeaves.count(leaf) && valueNeedsBarrier(leaf, sg.root)) {
          return true;
        }
      }
      return false;
    };
    for (const auto& sg : subgraphs) {
      if (callNeedsBarrier(sg.root) || leavesNeedBarrier(sg)) {
        needsBarrier = true;
        break;
      }
      for (const auto& input : sg.root->inputs()) {
        checkTree(input.value);
        if (needsBarrier) {
          break;
        }
      }
      if (needsBarrier) {
        break;
      }
    }
    if (needsBarrier) {
      emitBarrier();
    }
  }

  addInclude("velox/experimental/torchwave/Elementwise.cuh");

  // Declare tensor storage pointer variables when the number of inputs is
  // within the limit. Otherwise, storage expressions are inlined at each use
  // site to reduce register pressure in the generated CUDA kernel.
  elementwiseVarNames_.clear();
  if (static_cast<int32_t>(allInputs.size()) <=
      WaveConfig::get().maxElementwiseVars) {
    for (size_t i = 0; i < allInputs.size(); ++i) {
      auto tp = cudaType(allInputs[i]);
      auto varName = fmt::format("b{}", i);
      elementwiseVarNames_[i] = varName;
      if (allInputs[i]->type().kind() == nativert::Type::Kind::Tensor) {
        code_ << "  " << tp << "* " << varName << " = storage<" << tp << ">("
              << param(allInputs[i], op) << ");\n";
      } else {
        code_ << "  " << tp << "* " << varName << " = "
              << param(allInputs[i], op) << ";\n";
      }
    }
  }

  auto storageRef = [&](size_t idx) -> std::string {
    auto varIt = elementwiseVarNames_.find(idx);
    if (varIt != elementwiseVarNames_.end()) {
      return varIt->second;
    }
    auto tp = cudaType(allInputs[idx]);
    if (allInputs[idx]->type().kind() == nativert::Type::Kind::Tensor) {
      return fmt::format("storage<{}>({})", tp, param(allInputs[idx], op));
    }
    return param(allInputs[idx], op);
  };

  for (const auto& ee : newExprs) {
    if (ee.shapeFromThisOp) {
      generateIndexToOffset(ee, allInputs);
    }
  }

  bool skipFastPath =
      WaveConfig::get().noElementwiseFastPath || staticSize.has_value();

  if (rowReduction != nullptr) {
    auto* root = subgraphs.at(0).root->outputs()[0];
    const auto rowCounter = op.allocateCounter();
    if (!skipFastPath) {
      code_ << "  if (" << fastPathCondition(numFastPathVars, tensorCount)
            << ") {\n";
      emitRowReduction(rowReduction, root, op, allInputs, rowCounter, false);
      code_ << "  } else {\n";
    }
    emitRowReduction(rowReduction, root, op, allInputs, rowCounter, true);
    if (!skipFastPath) {
      code_ << "  }\n";
    }
  } else {
    // Generate fast path test for tensor leaf inputs.
    if (!skipFastPath) {
      code_ << "  if (" << fastPathCondition(numFastPathVars, tensorCount)
            << ") {\n";

      // Fast path body: loop over all elements, computing all expressions.
      if (fullBlockResult) {
        code_
            << "    uint32_t rounded = roundUpPwr2(size, blockDim.x);\n"
            << "    for (uint32_t idx = blockInfo.blockInOp * blockDim.x + threadIdx.x; idx < rounded; idx += blockInfo.numBlocksInOp * blockDim.x) {\n";
        eltTrace(
            code_,
            "\"%d %d idx %d blockIdx %d\\n\", blockInfo.op, blockInfo.blockInOp, idx, blockIdx.x");
        for (size_t s = 0; s < subgraphs.size(); ++s) {
          auto tp = resultSpecs[s].value ? cudaType(resultSpecs[s].value)
                                         : cudaType(allInputs.at(0));
          code_ << "      " << tp << " result" << s << ";\n";
        }
        code_ << "      if (idx < size) {\n";
        for (size_t s = 0; s < subgraphs.size(); ++s) {
          auto resultVar = fmt::format("result{}", s);
          elementwiseExpr(
              subgraphs[s].root->outputs()[0], resultVar, op, allInputs);
          if (resultSpecs[s].value) {
            auto it = std::find(
                allInputs.begin(), allInputs.end(), resultSpecs[s].value);
            auto id = it - allInputs.begin();
            if (resultSpecs[s].value->type().kind() ==
                nativert::Type::Kind::Tensor) {
              code_ << "        " << storageRef(id) << "[idx] = " << resultVar
                    << ";\n";
            } else {
              code_ << "        " << storageRef(id) << "[0] = " << resultVar
                    << ";\n";
            }
          } else {
            code_ << "        " << resultSpecs[s].variable << " = " << resultVar
                  << ";\n";
          }
        }
        code_ << "      }\n";
        if (!resultStmt.empty()) {
          code_ << "      " << resultStmt << "\n";
        }
        code_ << "    }\n";
      } else {
        code_
            << "    for (uint32_t idx = blockInfo.blockInOp * blockDim.x + threadIdx.x; idx < size; idx += blockInfo.numBlocksInOp * blockDim.x) {\n";
        eltTrace(
            code_,
            "\"%d %d idx %d blockIdx %d\\n\", blockInfo.op, blockInfo.blockInOp, idx, blockIdx.x");
        for (size_t s = 0; s < subgraphs.size(); ++s) {
          auto* rootMeta = nodeMeta(subgraphs[s].root);
          bool resultInRegister = !rootMeta || rootMeta->returnMeta.empty() ||
              rootMeta->returnMeta[0].isRegister;
          auto tp = resultSpecs[s].value ? cudaType(resultSpecs[s].value)
                                         : cudaType(allInputs.at(0));
          auto resultVar = fmt::format("result{}", s);
          code_ << "      " << tp << " " << resultVar << ";\n";
          elementwiseExpr(
              subgraphs[s].root->outputs()[0], resultVar, op, allInputs);
          if (!resultInRegister) {
            continue;
          }
          if (resultSpecs[s].value) {
            auto it = std::find(
                allInputs.begin(), allInputs.end(), resultSpecs[s].value);
            auto id = it - allInputs.begin();
            if (resultSpecs[s].value->type().kind() ==
                nativert::Type::Kind::Tensor) {
              code_ << "      " << storageRef(id) << "[idx] = " << resultVar
                    << ";\n";
            } else {
              code_ << "      " << storageRef(id) << "[0] = " << resultVar
                    << ";\n";
            }
          } else {
            code_ << "       " << resultSpecs[s].variable << " = " << resultVar
                  << ";\n";
          }
        }
        if (!resultStmt.empty()) {
          code_ << "      " << resultStmt << "\n";
        }
        code_ << "    }\n";
      }

      code_ << "  } else {\n";
    }
    if (fullBlockResult) {
      code_
          << "    uint32_t rounded = roundUpPwr2(size, blockDim.x);\n"
          << "    for (uint32_t idx = blockInfo.blockInOp * blockDim.x + threadIdx.x; idx < rounded; idx += blockInfo.numBlocksInOp * blockDim.x) {\n";
      for (size_t s = 0; s < subgraphs.size(); ++s) {
        auto tp = resultSpecs[s].value ? cudaType(resultSpecs[s].value)
                                       : cudaType(allInputs.at(0));
        code_ << "      " << tp << " result" << s << ";\n";
      }
      code_ << "      if (idx < size) {\n";
      for (size_t s = 0; s < subgraphs.size(); ++s) {
        auto* rootMeta = nodeMeta(subgraphs[s].root);
        bool resultInRegister = !rootMeta || rootMeta->returnMeta.empty() ||
            rootMeta->returnMeta[0].isRegister;
        auto resultVar = fmt::format("result{}", s);
        elementwiseExpr(
            subgraphs[s].root->outputs()[0], resultVar, op, allInputs, true);
        if (!resultInRegister) {
          continue;
        }
        if (resultSpecs[s].value) {
          auto it = std::find(
              allInputs.begin(), allInputs.end(), resultSpecs[s].value);
          auto id = it - allInputs.begin();
          if (resultSpecs[s].value->type().kind() ==
              nativert::Type::Kind::Tensor) {
            // The output's own contiguity, read from its descriptor, not a
            // bit of isFastPath. That bitmask is indexed by position in
            // leafInputs, and an output sits past the end of it in allInputs --
            // so the bit taken here belonged to some unrelated input, or to no
            // tensor at all. A contiguous input then made a strided output be
            // written linearly, which fills the first numEl slots of a pitched
            // band and leaves the rest of it untouched.
            code_ << "        " << storageRef(id)
                  << rootStoreIndex(resultSpecs[s].value, op) << " = "
                  << resultVar << ";\n";
          } else {
            code_ << "        " << storageRef(id) << "[0] = " << resultVar
                  << ";\n";
          }
        } else {
          code_ << "        " << resultSpecs[s].variable << " = " << resultVar
                << ";\n";
        }
      }
      code_ << "      }\n";
      if (!resultStmt.empty()) {
        code_ << "      " << resultStmt << "\n";
      }
      code_ << "    }\n";
    } else {
      code_
          << "    for (uint32_t idx = blockInfo.blockInOp * blockDim.x + threadIdx.x; idx < size; idx += blockInfo.numBlocksInOp * blockDim.x) {\n";
      for (size_t s = 0; s < subgraphs.size(); ++s) {
        auto* rootMeta = nodeMeta(subgraphs[s].root);
        bool resultInRegister = !rootMeta || rootMeta->returnMeta.empty() ||
            rootMeta->returnMeta[0].isRegister;
        auto tp = resultSpecs[s].value ? cudaType(resultSpecs[s].value)
                                       : cudaType(allInputs.at(0));
        auto resultVar = fmt::format("result{}", s);
        code_ << "      " << tp << " " << resultVar << ";\n";
        elementwiseExpr(
            subgraphs[s].root->outputs()[0], resultVar, op, allInputs, true);
        if (!resultInRegister) {
          continue;
        }
        if (resultSpecs[s].value) {
          auto it = std::find(
              allInputs.begin(), allInputs.end(), resultSpecs[s].value);
          auto id = it - allInputs.begin();
          if (resultSpecs[s].value->type().kind() ==
              nativert::Type::Kind::Tensor) {
            // See the store above: the output is not a leaf input, so it has
            // no bit in isFastPath. Its own descriptor carries its contiguity.
            code_ << "      " << storageRef(id)
                  << rootStoreIndex(resultSpecs[s].value, op) << " = "
                  << resultVar << ";\n";
          } else {
            code_ << "      " << storageRef(id) << "[0] = " << resultVar
                  << ";\n";
          }
        } else {
          code_ << "       " << resultSpecs[s].variable << " = " << resultVar
                << ";\n";
        }
      }
      if (!resultStmt.empty()) {
        code_ << "      " << resultStmt << "\n";
      }
      code_ << "    }\n";
    }
    if (!skipFastPath) {
      code_ << "  }\n";
    }
  }
  // Wait for all warps to finish before size in shared memory is reused.
  code_ << "  __syncthreads();\n";
  noteBlockBarrier();

  // A data-dependent scan output (shapeSetOnDevice, e.g. masked_select) writes
  // its length inside the element loop, at the last element. When the input is
  // empty the loop runs zero iterations, so no thread writes the length and the
  // output keeps its reserved (upper-bound) size. Zero it here, still inside
  // this op's block (size is in scope) so a fused consumer reading the length
  // next (e.g. cat summing its inputs' dims[0]) sees 0.
  if (shapeSetOnDeviceResult != nullptr &&
      shapeSetOnDeviceResult->type().kind() == nativert::Type::Kind::Tensor) {
    code_
        << "  if (size == 0 && threadIdx.x == 0 && blockInfo.blockInOp == 0) {\n"
        << "    " << param(shapeSetOnDeviceResult, op) << "->dims[0] = 0;\n"
        << "  }\n";
  }
  code_ << "  }\n";

  for (auto& ee : newExprs) {
    op.elementExprs().push_back(std::move(ee));
  }
}

namespace {

// The offset of the element a contiguous 'leaf' contributes to element 'idx'
// of 'root', as an expression over 'idx' with literal divisors, when both
// shapes are fully static and 'leaf' broadcasts to 'root'. Adjacent dims the
// leaf does not broadcast over merge into one divide and modulo, so a
// [R, C, 1] leaf of an [R, C, C] root reads at idx / C.
std::optional<std::string> staticLeafIndex(
    const StaticDims& root,
    const StaticDims& leaf) {
  if (leaf.size() > root.size()) {
    return std::nullopt;
  }
  const auto rank = root.size();
  StaticDims padded(rank - leaf.size(), 1);
  padded.insert(padded.end(), leaf.begin(), leaf.end());
  std::vector<bool> read(rank);
  for (size_t d = 0; d < rank; ++d) {
    if (root[d] < 0 || padded[d] < 0 ||
        (padded[d] != root[d] && padded[d] != 1)) {
      return std::nullopt;
    }
    read[d] = padded[d] == root[d] && root[d] != 1;
  }
  if (std::all_of(read.begin(), read.end(), [](bool r) { return r; })) {
    return "idx";
  }
  std::string expr;
  int64_t inner = 1;
  int64_t leafStride = 1;
  for (int64_t d = static_cast<int64_t>(rank) - 1; d >= 0;) {
    if (!read[d]) {
      inner *= root[d];
      leafStride *= padded[d];
      --d;
      continue;
    }
    int64_t extent = 1;
    const auto runInner = inner;
    const auto runStride = leafStride;
    while (d >= 0 && read[d]) {
      extent *= root[d];
      inner *= root[d];
      leafStride *= padded[d];
      --d;
    }
    std::string term = "idx";
    if (runInner != 1) {
      term = fmt::format("(idx / {}u)", runInner);
    }
    const bool outermost =
        std::none_of(
            read.begin(), read.begin() + d + 1, [](bool r) { return r; }) &&
        std::all_of(root.begin(), root.begin() + d + 1, [](int64_t r) {
          return r == 1;
        });
    if (!outermost) {
      term = fmt::format("({} % {}u)", term, extent);
    }
    if (runStride != 1) {
      term = fmt::format("{} * {}u", term, runStride);
    }
    expr = expr.empty() ? term : fmt::format("{} + {}", term, expr);
  }
  return expr.empty() ? std::string("0u") : expr;
}

} // namespace

std::string CompileCtx::formatLeafAccess(
    ValueCP value,
    const std::vector<ValueCP>& inputs,
    const KernelOperation& op,
    bool slowPath) {
  auto it = std::find(inputs.begin(), inputs.end(), value);
  if (it == inputs.end()) {
    const auto* producer = value->producer();
    const std::string producerTarget =
        producer ? std::string(producer->target()) : "<none>";
    TORCH_CHECK(
        false,
        "Input value not found in inputs vector: value=",
        value->name(),
        " id=",
        value->id(),
        " kind=",
        static_cast<int>(value->type().kind()),
        " producer=",
        producerTarget);
  }
  auto valueIdx = it - inputs.begin();
  auto varIt = elementwiseVarNames_.find(valueIdx);
  std::string base;
  if (varIt != elementwiseVarNames_.end()) {
    base = varIt->second;
  } else {
    auto tp = cudaType(value);
    if (value->type().kind() == nativert::Type::Kind::Tensor) {
      base = fmt::format("storage<{}>({})", tp, param(value, op));
    } else {
      base = param(value, op);
    }
  }
  if (value->type().kind() != nativert::Type::Kind::Tensor) {
    return fmt::format("*{}", base);
  }
  if (slowPath) {
    if (auto index = staticLeafIndexOf(value, op)) {
      return fmt::format("{}[{}]", base, *index);
    }
    auto bitIdx = fastPathBitIndex_[valueIdx];
    return fmt::format(
        "{}[complexIdx(isFastPath{} & (1 << {}), {}, idx)]",
        base,
        bitIdx / kBitsPerWord,
        bitIdx % kBitsPerWord,
        param(value, op));
  }
  return fmt::format("{}[idx]", base);
}

std::optional<int64_t> CompileCtx::staticLoopSize(
    const std::vector<ValueCP>& leafInputs,
    const std::unordered_set<ValueCP>& wholeTensorValues,
    const std::unordered_set<ValueCP>& sizeExcludedValues,
    const std::vector<Subgraph>& subgraphs,
    const std::vector<ResultSpec>& resultSpecs,
    ValueCP shapeSetOnDeviceResult,
    const KernelOperation& op) {
  if (!WaveConfig::get().staticLeafIndex || subgraphs.size() != 1 ||
      shapeSetOnDeviceResult != nullptr) {
    return std::nullopt;
  }
  auto* root = subgraphs[0].root->outputs()[0];
  if (!isFullyStatic(types_, root)) {
    return std::nullopt;
  }
  const auto rootDims = *staticDimsOf(types_, root);
  for (const auto& spec : resultSpecs) {
    auto* value = spec.value;
    if (value == nullptr ||
        value->type().kind() != nativert::Type::Kind::Tensor) {
      continue;
    }
    if (sizeExcludedValues.count(value) || !types_.contiguous(value) ||
        staticDimsOf(types_, value) != std::optional<StaticDims>(rootDims)) {
      return std::nullopt;
    }
  }
  const auto& leafRefs = currentElementExpr_->leafRefs;
  for (auto* leaf : leafInputs) {
    if (leaf->type().kind() != nativert::Type::Kind::Tensor ||
        wholeTensorValues.count(leaf)) {
      continue;
    }
    // A leaf below a view indexes against the view's input.
    auto refIt = leafRefs.find(leaf);
    auto* indexRoot = refIt != leafRefs.end() ? refIt->second : root;
    if (!types_.contiguous(leaf) || !isFullyStatic(types_, leaf) ||
        !isFullyStatic(types_, indexRoot) ||
        generatingOp_->broadcastRefs().count(op.paramOffset(leaf)) != 0 ||
        !staticLeafIndex(
            *staticDimsOf(types_, indexRoot), *staticDimsOf(types_, leaf))) {
      return std::nullopt;
    }
  }
  for (const auto& spec : resultSpecs) {
    if (spec.value != nullptr &&
        spec.value->type().kind() == nativert::Type::Kind::Tensor) {
      generatingOp_->addStaticShape(
          op.paramOffset(spec.value), rootDims, /*contiguous=*/false);
    }
  }
  int64_t numel = 1;
  for (auto dim : rootDims) {
    numel *= dim;
  }
  return numel;
}

namespace {

// Lanes reducing one row in the warp form: the largest power of two up to 32
// that divides a static row length, so no lane idles at the end of a row.
// Under 8 lanes the reads of a row coalesce poorly, so the choice is then
// whichever of 8, 16 and 32 leaves the fewest lanes idle. 32 for a row
// length known only at run time.
int32_t laneGroupSize(std::optional<int64_t> rowLen) {
  if (!rowLen.has_value() || *rowLen <= 0) {
    return 32;
  }
  const auto lowestBit = *rowLen & -*rowLen;
  if (lowestBit >= 8) {
    return static_cast<int32_t>(std::min<int64_t>(lowestBit, 32));
  }
  int32_t best = 32;
  int64_t bestIdle = std::numeric_limits<int64_t>::max();
  for (int32_t lanes : {32, 16, 8}) {
    const auto idle = (*rowLen + lanes - 1) / lanes * lanes - *rowLen;
    if (idle < bestIdle) {
      best = lanes;
      bestIdle = idle;
    }
  }
  return best;
}

} // namespace

void CompileCtx::emitRowReduction(
    NodeCP reduction,
    ValueCP root,
    KernelOperation& op,
    const std::vector<ValueCP>& inputs,
    int32_t rowCounter,
    bool slowPath) {
  constexpr int64_t kWarpRowMax = 1024;
  constexpr int32_t kRowsPerClaim = 4;
  const auto& spec = *nodeMeta(reduction)->rowReduction;
  const auto keep = *rowReductionKeep(reduction, spec);
  auto* output = reduction->outputs()[0];
  std::optional<int64_t> staticRowLen;
  code_ << "  {\n";
  if (WaveConfig::get().staticLeafIndex && isFullyStatic(types_, root)) {
    auto dims = *staticDimsOf(types_, root);
    int64_t numRows = 1;
    int64_t rowLen = 1;
    for (size_t d = 0; d < dims.size(); ++d) {
      (static_cast<int64_t>(d) < keep ? numRows : rowLen) *= dims[d];
    }
    staticRowLen = rowLen;
    code_ << "    const uint32_t numRows = " << numRows << "u;\n"
          << "    const uint32_t rowLen = " << rowLen << "u;\n";
    generatingOp_->addStaticShape(
        op.paramOffset(root), std::move(dims), /*contiguous=*/false);
  } else {
    code_ << "    uint32_t numRows = 1;\n"
          << "    for (int32_t d = 0; d < " << keep << "; ++d) {\n"
          << "      numRows *= " << param(root, op) << "->dims[d];\n"
          << "    }\n"
          << "    const uint32_t rowLen = numRows == 0 ? 0 : size / numRows;\n";
  }
  const auto store = fmt::format(
      "storage<{0}>({1})[complexIdx({1}->contiguous, {1}, row)]",
      cudaType(output),
      param(output, op));
  const auto claim =
      fmt::format("atomicAdd(param<uint32_t>(blockInfo, {}), ", rowCounter);
  // Evaluates one row into rowAcc and stores it, its elements taken from
  // 'firstElement' in steps of 'elementStep' while 'guard' holds.
  auto emitRow = [&](std::string_view firstElement,
                     std::string_view elementStep,
                     std::string_view guard,
                     std::string_view combineAcross,
                     std::string_view storeIf) {
    code_ << "      " << spec.accumulatorType << " rowAcc = " << spec.init
          << ";\n"
          << "      for (uint32_t k = " << firstElement << "; " << guard
          << "k < rowLen; k += " << elementStep << ") {\n"
          << "      const uint32_t idx = row * rowLen + k;\n"
          << "      " << cudaType(root) << " result0;\n";
    elementwiseExpr(root, "result0", op, inputs, slowPath);
    code_ << "      "
          << fmt::format(fmt::runtime(spec.combine), "rowAcc", "result0")
          << ";\n"
          << "      }\n"
          << "      rowAcc = " << combineAcross << ";\n"
          << "      if (" << storeIf << ") {\n"
          << "        " << store << " = rowAcc;\n"
          << "      }\n";
  };
  // Rows are claimed from a counter rather than dealt out by position, so
  // the blocks that run faster take more of them.
  // A warp reduces 32 / groupSize rows at a time, groupSize lanes on each.
  const int32_t groupSize = laneGroupSize(staticRowLen);
  const int32_t rowsPerStep = 32 / groupSize;
  const int32_t rowsPerClaim = std::max(kRowsPerClaim, rowsPerStep);
  auto emitWarpLoop = [&]() {
    code_ << "    const uint32_t lane = threadIdx.x & 31;\n"
          << "    for (;;) {\n"
          << "      uint32_t firstRow = 0;\n"
          << "      if (lane == 0) {\n"
          << "        firstRow = " << claim << rowsPerClaim << "u);\n"
          << "      }\n"
          << "      firstRow = __shfl_sync(0xffffffff, firstRow, 0);\n"
          << "      if (firstRow >= numRows) {\n"
          << "        break;\n"
          << "      }\n"
          << "      const uint32_t endRow = min(firstRow + " << rowsPerClaim
          << "u, numRows);\n"
          << "      for (uint32_t rowBase = firstRow; rowBase < endRow; "
          << "rowBase += " << rowsPerStep << "u) {\n"
          << "      const uint32_t row = rowBase + lane / " << groupSize
          << "u;\n";
    emitRow(
        fmt::format("lane % {}u", groupSize),
        fmt::format("{}u", groupSize),
        "row < endRow && ",
        fmt::format(fmt::runtime(spec.groupCombine), "rowAcc", groupSize),
        fmt::format("lane % {}u == 0 && row < endRow", groupSize));
    code_ << "      }\n"
          << "    }\n";
  };
  auto emitBlockLoop = [&]() {
    op.addSharedDeclaration("  __shared__ uint32_t rowClaim;\n");
    separateSharedScratch({"rowClaim"});
    code_ << "    for (;;) {\n"
          << "      if (threadIdx.x == 0) {\n"
          << "        rowClaim = " << claim << "1u);\n"
          << "      }\n"
          << "      __syncthreads();\n"
          << "      const uint32_t row = rowClaim;\n"
          << "      __syncthreads();\n"
          << "      if (row >= numRows) {\n"
          << "        break;\n"
          << "      }\n";
    emitRow(
        "threadIdx.x",
        "blockDim.x",
        "",
        fmt::format(fmt::runtime(spec.blockCombine), "rowAcc"),
        "threadIdx.x == 0");
    code_ << "    }\n";
  };
  if (staticRowLen.has_value()) {
    if (*staticRowLen <= kWarpRowMax) {
      emitWarpLoop();
    } else {
      emitBlockLoop();
    }
  } else {
    code_ << "  if (rowLen <= " << kWarpRowMax << "u) {\n";
    emitWarpLoop();
    code_ << "  } else {\n";
    emitBlockLoop();
    code_ << "  }\n";
  }
  code_ << "  }\n";
}

std::string CompileCtx::rootStoreIndex(ValueCP value, const KernelOperation& op)
    const {
  return fmt::format(
      "[complexIdx({0}->contiguous, {0}, idx)]", param(value, op));
}

std::optional<std::string> CompileCtx::staticLeafIndexOf(
    ValueCP value,
    const KernelOperation& op) const {
  if (!WaveConfig::get().staticLeafIndex || currentRootOutput_ == nullptr ||
      !types_.contiguous(value) ||
      generatingOp_->broadcastRefs().count(op.paramOffset(value)) != 0 ||
      !isFullyStatic(types_, value) ||
      !isFullyStatic(types_, currentRootOutput_)) {
    return std::nullopt;
  }
  auto rootDims = *staticDimsOf(types_, currentRootOutput_);
  auto leafDims = *staticDimsOf(types_, value);
  auto index = staticLeafIndex(rootDims, leafDims);
  if (index.has_value()) {
    generatingOp_->addStaticShape(
        op.paramOffset(value), std::move(leafDims), /*contiguous=*/true);
    generatingOp_->addStaticShape(
        op.paramOffset(currentRootOutput_),
        std::move(rootDims),
        /*contiguous=*/false);
  }
  return index;
}

std::string CompileCtx::buildElementwiseCall(
    const Metadata& meta,
    NodeCP node,
    const KernelOperation& op,
    const std::vector<std::string>& argTexts) {
  const auto& ew = *meta.elementwise;
  std::stringstream ss;
  ss << ew.functionName;
  auto ewPresenceParams = meta.hasPresentTemplateParams()
      ? presentTemplateParams(meta, node)
      : std::string();
  if (!meta.typeTemplateParams.empty() || meta.hasDtypeTemplateParam ||
      !meta.templateAttrs.empty() || !ewPresenceParams.empty()) {
    ss << "<";
    const auto& nodeInputs = node->inputs();
    bool firstTp = true;
    for (size_t i = 0; i < meta.typeTemplateParams.size(); ++i) {
      if (!firstTp) {
        ss << ", ";
      }
      firstTp = false;
      ss << cudaType(nodeInputs[meta.typeTemplateParams[i]].value);
    }
    if (meta.hasDtypeTemplateParam) {
      if (!firstTp) {
        ss << ", ";
      }
      firstTp = false;
      const auto* dtypeAttr = node->tryGetAttribute("dtype");
      TORCH_CHECK(dtypeAttr, node->target(), ": missing dtype attribute");
      ss << cudaTypeFromDtype(*dtypeAttr);
    }
    for (const auto& attrName : meta.templateAttrs) {
      if (!firstTp) {
        ss << ", ";
      }
      firstTp = false;
      const auto* attr = node->tryGetAttribute(attrName);
      TORCH_CHECK(
          attr, node->target(), ": missing template attribute ", attrName);
      ss << constantToString(attr->value);
    }
    if (!ewPresenceParams.empty()) {
      if (!firstTp) {
        ss << ", ";
      }
      ss << ewPresenceParams;
    }
    ss << ">";
  }
  ss << "(";
  bool first = true;
  if (ew.hasIdxArg) {
    ss << "idx";
    first = false;
  }
  if (ew.hasSizeArg) {
    if (!first) {
      ss << ", ";
    }
    ss << "size";
    first = false;
  }
  for (const auto& arg : argTexts) {
    if (!first) {
      ss << ", ";
    }
    first = false;
    ss << arg;
  }
  if (ew.hasOutputArg) {
    TORCH_CHECK(
        currentRootOutput_,
        "hasOutputArg requires a root output; none set for ",
        node->target());
    if (!first) {
      ss << ", ";
    }
    first = false;
    ss << param(currentRootOutput_, op);
  }
  if (ew.hasOwnOutputArg) {
    if (!first) {
      ss << ", ";
    }
    first = false;
    ss << param(node->outputs()[0], op);
  }
  if (ew.hasBlockInfo) {
    if (!first) {
      ss << ", ";
    }
    ss << "blockInfo";
  }
  ss << ")";
  return ss.str();
}

void CompileCtx::maybeExtractOutOfLine(
    ValueCP value,
    const std::string& resultName,
    const std::vector<ValueCP>& inputs,
    size_t codeStart,
    size_t tempLogStart,
    bool slowPath) {
  auto codeEnd = static_cast<size_t>(code_.tellp());
  if (WaveConfig::get().outOfLineExprSize <= 0 ||
      static_cast<int32_t>(codeEnd - codeStart) <=
          WaveConfig::get().outOfLineExprSize) {
    return;
  }
  auto fullCode = code_.str();
  auto extractedCode = fullCode.substr(codeStart, codeEnd - codeStart);

  std::set<std::pair<std::string, std::string>> uniqueTemps;
  for (size_t i = tempLogStart; i < tempUseLog_.size(); ++i) {
    uniqueTemps.insert(tempUseLog_[i]);
  }

  std::set<size_t> usedIndices;
  for (auto& [idx, varName] : elementwiseVarNames_) {
    if (extractedCode.find(varName + "[") != std::string::npos ||
        extractedCode.find("*" + varName) != std::string::npos) {
      usedIndices.insert(idx);
    }
  }
  for (auto& [helperName, deps] : helperVarDeps_) {
    if (extractedCode.find(helperName + "(") != std::string::npos) {
      usedIndices.insert(deps.begin(), deps.end());
    }
  }
  std::vector<std::pair<size_t, std::string>> usedVars;
  for (auto idx : usedIndices) {
    auto it = elementwiseVarNames_.find(idx);
    if (it != elementwiseVarNames_.end()) {
      usedVars.emplace_back(idx, it->second);
    }
  }

  auto funcName = fmt::format("elementExpr{}", outOfLineCounter_++);
  auto returnType = cudaType(value);

  outOfLineFunctions_ << "__device__ __noinline__ " << returnType << " "
                      << funcName
                      << "(\n    uint32_t idx, uint32_t size, "
                         "BlockInfo& blockInfo";
  int32_t numFastPathUsed = 0;
  if (slowPath) {
    for (int32_t i = 0;; ++i) {
      if (extractedCode.find("isFastPath" + std::to_string(i)) ==
          std::string::npos) {
        numFastPathUsed = i;
        break;
      }
    }
    for (int32_t i = 0; i < numFastPathUsed; ++i) {
      outOfLineFunctions_ << ",\n    uint32_t isFastPath" << i;
    }
  }
  for (auto& [idx, varName] : usedVars) {
    outOfLineFunctions_ << ",\n    " << cudaType(inputs[idx]) << "* "
                        << varName;
  }
  outOfLineFunctions_ << ") {\n";

  std::set<std::string> declaredInHelper;
  for (auto& [type, name] : uniqueTemps) {
    if (declaredInHelper.insert(name).second) {
      outOfLineFunctions_ << "  " << type << " " << name << ";\n";
    }
  }
  if (!declaredInHelper.count(resultName)) {
    outOfLineFunctions_ << "  " << returnType << " " << resultName << ";\n";
  }

  outOfLineFunctions_ << extractedCode;
  outOfLineFunctions_ << "  return " << resultName << ";\n";
  outOfLineFunctions_ << "}\n\n";

  helperVarDeps_[funcName] = usedIndices;

  code_.str(fullCode.substr(0, codeStart));
  code_.seekp(0, std::ios::end);
  code_ << "      " << resultName << " = " << funcName
        << "(idx, size, blockInfo";
  for (int32_t i = 0; i < numFastPathUsed; ++i) {
    code_ << ", isFastPath" << i;
  }
  for (auto& [idx, varName] : usedVars) {
    code_ << ", " << varName;
  }
  code_ << ");\n";

  tempUseLog_.erase(
      tempUseLog_.begin() + static_cast<ptrdiff_t>(tempLogStart),
      tempUseLog_.end());
}

void CompileCtx::elementwiseExprImpl(
    ValueCP value,
    const std::string& resultName,
    const std::unordered_set<ValueCP>& inputSet,
    const std::vector<ValueCP>& inputs,
    const KernelOperation& op,
    bool slowPath) {
  auto codeStart = static_cast<size_t>(code_.tellp());
  auto tempLogStart = tempUseLog_.size();

  auto leaf = [&](ValueCP v) {
    return formatLeafAccess(v, inputs, op, slowPath);
  };

  auto isTerminal = [&](ValueCP v) {
    return inputSet.count(v) || !v->producer();
  };

  if (isTerminal(value)) {
    code_ << "      " << resultName << " = " << leaf(value) << ";\n";
    return;
  }

  auto* node = value->producer();
  placed_.insert(node);
  generatingOp_->allNodes().insert(node);

  auto* meta = nodeMeta(node);
  if (isFusedThroughView(meta)) {
    TORCH_CHECK(
        fusesThroughView(node),
        "Elementwise expression reaches a view it does not compute through: ",
        node->target());
    // The ops below index against the view's input, which has the view's
    // element order.
    auto* viewInput = node->inputs()[*meta->viewOfArg].value;
    auto* root = currentRootOutput_;
    currentRootOutput_ = viewInput;
    elementwiseExprImpl(viewInput, resultName, inputSet, inputs, op, slowPath);
    currentRootOutput_ = root;
    return;
  }
  TORCH_CHECK(
      meta && meta->elementwise, "Not an elementwise op: ", node->target());

  if (!meta->headerFile.empty()) {
    addInclude(meta->headerFile);
  }

  std::vector<std::string> argTexts;
  std::vector<std::pair<ValueCP, std::string>> tempsToRelease;
  bool firstValue = true;

  // Processes a single value input. wholeTensor args pass the tensor
  // pointer; ListPack args with isRegister flatten their elements as
  // individual scalar args (f(x, ListPack(a,b)) → f(x, a, b));
  // ListPack args without isRegister pass as TensorList*. Terminal
  // values read from storage; the first same-type sub-expression
  // reuses resultName to avoid a temp allocation.
  auto processValue = [&](ValueCP v, bool isWhole, bool isRegister) {
    if (isWhole) {
      argTexts.push_back(param(v, op));
      firstValue = false;
      return;
    }
    auto* producer = v->producer();
    if (producer && producer->target() == "prim.ListPack") {
      if (isRegister) {
        placed_.insert(producer);
        generatingOp_->allNodes().insert(producer);
        for (const auto& listInput : producer->inputs()) {
          auto* lv = listInput.value;
          if (isTerminal(lv)) {
            argTexts.push_back(leaf(lv));
          } else {
            auto tempName = useTemp(lv);
            elementwiseExprImpl(lv, tempName, inputSet, inputs, op, slowPath);
            argTexts.push_back(tempName);
            tempsToRelease.emplace_back(lv, tempName);
          }
        }
      } else {
        argTexts.push_back(param(v, op));
      }
    } else if (isTerminal(v)) {
      argTexts.push_back(leaf(v));
    } else if (firstValue && cudaType(v) == cudaType(value)) {
      elementwiseExprImpl(v, resultName, inputSet, inputs, op, slowPath);
      argTexts.push_back(resultName);
    } else {
      auto tempName = useTemp(v);
      elementwiseExprImpl(v, tempName, inputSet, inputs, op, slowPath);
      argTexts.push_back(tempName);
      tempsToRelease.emplace_back(v, tempName);
    }
    firstValue = false;
  };

  // Points every terminal tensor leaf reachable through 'v' at src's shape. A
  // fused in-place scatter (tw.slice_scatter / select_scatter / scatter /
  // scatter_add: hasOutputArg + a valuesArg written per element) sizes its
  // elementwise loop by 'src' (the valuesArg), not by the op's output tensor
  // (== self, which is excluded from sizing). So a src leaf's per-lane read
  // must decompose 'idx' by src's shape, not broadcast against self's
  // (larger) shape as the default prologue would. When the op has an 'index'
  // tensor, it has src's shape and the leaf broadcasts against it, which
  // also covers a leaf of fewer elements than src. Otherwise the leaf gets
  // an own-dims calculator (ensureIndexCalculator()), right for a leaf of
  // src's full shape.
  const auto* scatterIndex = node->tryGetInput("index");
  const bool indexIsRef = scatterIndex != nullptr &&
      scatterIndex->value != nullptr &&
      scatterIndex->value->type().kind() == nativert::Type::Kind::Tensor;
  std::function<void(ValueCP)> forceSrcLeafOwnDims = [&](ValueCP v) {
    if (isTerminal(v)) {
      if (v->type().kind() == nativert::Type::Kind::Tensor) {
        if (indexIsRef) {
          generatingOp_->setBroadcastRef(
              op.paramOffset(v), op.paramOffset(scatterIndex->value));
        } else {
          generatingOp_->addOwnDimsCalcOffset(op.paramOffset(v));
        }
      }
      return;
    }
    for (const auto& input : v->producer()->inputs()) {
      forceSrcLeafOwnDims(input.value);
    }
  };

  // Gather call args via forArguments for BOTH the generateCall and the default
  // path, so register args that arrive as constant ATTRIBUTES (not graph edges)
  // are emitted too.  The base's scalar support folds e.g. scalar_tensor's
  // scalar into a constant attribute; iterating node->inputs() alone would miss
  // it and hand generateCall empty args (the scalar_tensor/__to ops set
  // numArgs to bound this to their single register arg).
  int32_t emittedArgs = 0;
  int32_t numArgs = meta->elementwise->numArgs;
  forArguments(
      *meta,
      node,
      [&](size_t schemaIdx, ValueCP v, const nativert::Attribute* attr) {
        if (emittedArgs >= numArgs) {
          return;
        }
        if (v) {
          ++emittedArgs;
          bool isWhole = schemaIdx < meta->argumentMeta.size() &&
              meta->argumentMeta[schemaIdx].wholeTensor;
          bool isReg = schemaIdx < meta->argumentMeta.size() &&
              meta->argumentMeta[schemaIdx].isRegister;
          // A gather op (hasOutputArg) reads its whole-tensor operands via
          // computed offsets and, for repeat, wraps by their own dims. Force
          // an own-dims index calculator so those reads (and __repeat's
          // self->sizes[].mod) never see a broadcast or uninitialized sizes[].
          if (isWhole && meta->elementwise && meta->elementwise->hasOutputArg &&
              v->type().kind() == nativert::Type::Kind::Tensor) {
            generatingOp_->addOwnDimsCalcOffset(op.paramOffset(v));
          }
          // The register src of a fused in-place scatter (hasOutputArg +
          // valuesArg) is read per element while the loop is sized by src, not
          // by self; force own-dims calculators on its leaves so a
          // non-contiguous src view is read at its real strides.
          if (isReg && meta->valuesArg.has_value() && meta->elementwise &&
              meta->elementwise->hasOutputArg &&
              v->type().kind() == nativert::Type::Kind::Tensor) {
            forceSrcLeafOwnDims(v);
          }
          processValue(v, isWhole, isReg);
        } else if (attr) {
          ++emittedArgs;
          if (std::holds_alternative<nativert::None>(attr->value)) {
            argTexts.emplace_back("0");
          } else {
            auto off = op.attrOffset(node, attr->name);
            auto tp = cudaAttrType(attr->value);
            argTexts.push_back(
                fmt::format("*param<{}>(blockInfo, {})", tp, off));
          }
        }
      });
  // A gather op (hasOutputArg) decomposes the enclosing output's linear index
  // by the output's own dims (out->sizes[]). Force an own-dims calculator for
  // that output tensor; the contiguous fast path would otherwise leave its
  // sizes[] uninitialized.
  if (meta->elementwise && meta->elementwise->hasOutputArg &&
      currentRootOutput_ &&
      currentRootOutput_->type().kind() == nativert::Type::Kind::Tensor) {
    generatingOp_->addOwnDimsCalcOffset(op.paramOffset(currentRootOutput_));
  }
  // Likewise for an op that decomposes 'idx' by its own output's dims.
  if (meta->elementwise && meta->elementwise->hasOwnOutputArg) {
    generatingOp_->addOwnDimsCalcOffset(op.paramOffset(node->outputs()[0]));
  }
  if (meta->generateCall) {
    std::stringstream callSs;
    meta->generateCall(callSs, node, std::move(argTexts));
    code_ << "      " << resultName << " = " << callSs.str() << ";\n";
  } else {
    code_ << "      " << resultName << " = "
          << buildElementwiseCall(*meta, node, op, argTexts) << ";\n";
  }

  for (auto& [v, name] : tempsToRelease) {
    tempDone(v, name);
  }

  maybeExtractOutOfLine(
      value, resultName, inputs, codeStart, tempLogStart, slowPath);
}

void CompileCtx::elementwiseExpr(
    ValueCP value,
    const std::string& resultName,
    const KernelOperation& op,
    const std::vector<ValueCP>& inputs,
    bool slowPath) {
  addInclude("velox/experimental/torchwave/Elementwise.cuh");
  // 'value' is the subgraph root output the loop stores at 'idx'; record it so
  // buildElementwiseCall can pass it to ops with hasOutputArg (e.g.
  // index_select needs the enclosing expression's output shape).
  currentRootOutput_ = value;
  std::unordered_set<ValueCP> inputSet(inputs.begin(), inputs.end());
  inputSet.erase(value);
  elementwiseExprImpl(value, resultName, inputSet, inputs, op, slowPath);
}

} // namespace torch::wave
