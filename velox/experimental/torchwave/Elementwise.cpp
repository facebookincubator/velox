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
#include "velox/experimental/torchwave/Utils.h"

#include "velox/experimental/torchwave/WaveConfig.h"

#include <folly/ScopeGuard.h>
#include <algorithm>
#include <set>
#include <sstream>

namespace torch::wave {

void CompileCtx::collectSubgraphInputs(
    NodeCP node,
    const std::unordered_set<ValueCP>& sgInputs,
    std::unordered_set<ValueCP>& seen,
    std::vector<ValueCP>& result) const {
  for (auto& input : node->inputs()) {
    auto* value = input.value;
    auto* producer = value->producer();
    if (sgInputs.count(value) || (producer && placed_.count(producer))) {
      if (seen.insert(value).second) {
        result.push_back(value);
      }
    } else if (producer) {
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
}

// Generates CUDA code for elementwise operations (result[i] = f(a[i], b[i],
// ...)). Fast path: all operands are contiguous 1D - a single loop combines
// elements at each index. Slow path: some operands have broadcast dimensions or
// non-contiguous strides. A per-operand bitmask (isFastPath) marks which
// operands need index translation from the linear index to per-array offsets.
// The full-block variant calls the consumer on blockDim.x-aligned iterations
// (needed when the consumer has barriers). The non-full-block variant skips
// alignment padding.
void CompileCtx::generateElementwise(
    const std::vector<Subgraph>& subgraphs,
    const std::vector<ResultSpec>& resultSpecs,
    const std::string& resultStmt,
    bool fullBlockResult) {
  auto& op = *generatingOp_;

  // Get unique leaf inputs by walking subgraph roots.
  auto leafInputs = subgraphInputs(subgraphs);

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

  for (auto& rs : resultSpecs) {
    if (rs.value &&
        std::find(leafInputs.begin(), leafInputs.end(), rs.value) ==
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

  // Collect values marked as wholeTensor in any subgraph node's argumentMeta.
  std::unordered_set<ValueCP> wholeTensorValues;
  for (const auto& sg : subgraphs) {
    auto* meta = nodeMeta(sg.root);
    if (meta) {
      const auto& inputs = sg.root->inputs();
      for (size_t i = 0; i < inputs.size() && i < meta->argumentMeta.size();
           ++i) {
        if (meta->argumentMeta[i].wholeTensor) {
          wholeTensorValues.insert(inputs[i].value);
        }
      }
    }
  }

  // Build tensor-only bit index for fast path processing.
  int32_t tensorCount = 0;
  fastPathBitIndex_.assign(leafInputs.size(), -1);
  for (size_t i = 0; i < leafInputs.size(); ++i) {
    if (leafInputs[i]->type().kind() == nativert::Type::Kind::Tensor) {
      fastPathBitIndex_[i] = tensorCount++;
    }
  }
  // Generate shared declarations for size and fast path flags.
  op.addSharedDeclaration("  __shared__ uint32_t size;\n");
  auto numFastPathVars = (tensorCount + kBitsPerWord - 1) / kBitsPerWord;
  for (size_t i = 0; i < numFastPathVars; ++i) {
    op.addSharedDeclaration(
        "  __shared__ uint32_t isFastPath" + std::to_string(i) + ";\n");
  }

  // Generate the head: compute size and fast path flags from tensor leaf
  // inputs only.
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
    if (firstTensor) {
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
  code_ << "  }\n"
        << "  __syncthreads();\n";

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

  // Generate fast path test for tensor leaf inputs.
  code_ << "  if (";
  for (size_t i = 0; i < numFastPathVars; ++i) {
    if (i > 0) {
      code_ << " && ";
    }
    uint32_t mask;
    if (i < numFastPathVars - 1) {
      mask = 0xffffffff;
    } else {
      auto bitsInLast = tensorCount - i * kBitsPerWord;
      if (bitsInLast >= kBitsPerWord) {
        mask = 0xffffffff;
      } else {
        mask = (1u << bitsInLast) - 1;
      }
    }
    code_ << "isFastPath" << i << " == 0x" << std::hex << mask << std::dec;
  }
  code_ << ") {\n";

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
        auto it =
            std::find(allInputs.begin(), allInputs.end(), resultSpecs[s].value);
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
      auto tp = resultSpecs[s].value ? cudaType(resultSpecs[s].value)
                                     : cudaType(allInputs.at(0));
      auto resultVar = fmt::format("result{}", s);
      code_ << "      " << tp << " " << resultVar << ";\n";
      elementwiseExpr(
          subgraphs[s].root->outputs()[0], resultVar, op, allInputs);
      if (resultSpecs[s].value) {
        auto it =
            std::find(allInputs.begin(), allInputs.end(), resultSpecs[s].value);
        auto id = it - allInputs.begin();
        if (resultSpecs[s].value->type().kind() ==
            nativert::Type::Kind::Tensor) {
          code_ << "      " << storageRef(id) << "[idx] = " << resultVar
                << ";\n";
        } else {
          code_ << "      " << storageRef(id) << "[0] = " << resultVar << ";\n";
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
      auto resultVar = fmt::format("result{}", s);
      elementwiseExpr(
          subgraphs[s].root->outputs()[0], resultVar, op, allInputs, true);
      if (resultSpecs[s].value) {
        auto it =
            std::find(allInputs.begin(), allInputs.end(), resultSpecs[s].value);
        auto id = it - allInputs.begin();
        if (resultSpecs[s].value->type().kind() ==
            nativert::Type::Kind::Tensor) {
          auto bitIdx = fastPathBitIndex_[id];
          code_ << "        " << storageRef(id) << "[complexIdx(isFastPath"
                << bitIdx / kBitsPerWord << " & (1 << " << bitIdx % kBitsPerWord
                << "), " << param(resultSpecs[s].value, op)
                << ", idx)] = " << resultVar << ";\n";
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
      auto tp = resultSpecs[s].value ? cudaType(resultSpecs[s].value)
                                     : cudaType(allInputs.at(0));
      auto resultVar = fmt::format("result{}", s);
      code_ << "      " << tp << " " << resultVar << ";\n";
      elementwiseExpr(
          subgraphs[s].root->outputs()[0], resultVar, op, allInputs, true);
      if (resultSpecs[s].value) {
        auto it =
            std::find(allInputs.begin(), allInputs.end(), resultSpecs[s].value);
        auto id = it - allInputs.begin();
        if (resultSpecs[s].value->type().kind() ==
            nativert::Type::Kind::Tensor) {
          auto bitIdx = fastPathBitIndex_[id];
          code_ << "      " << storageRef(id) << "[complexIdx(isFastPath"
                << bitIdx / kBitsPerWord << " & (1 << " << bitIdx % kBitsPerWord
                << "), " << param(resultSpecs[s].value, op)
                << ", idx)] = " << resultVar << ";\n";
        } else {
          code_ << "      " << storageRef(id) << "[0] = " << resultVar << ";\n";
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
  code_ << "  }\n";
  code_ << "  }\n";

  for (auto& ee : newExprs) {
    op.elementExprs().push_back(std::move(ee));
  }
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

  auto leafText = [&](ValueCP v) -> std::string {
    auto it = std::find(inputs.begin(), inputs.end(), v);
    TORCH_CHECK(it != inputs.end(), "Input value not found in inputs vector");
    auto valueIdx = it - inputs.begin();
    auto varIt = elementwiseVarNames_.find(valueIdx);
    std::string base;
    if (varIt != elementwiseVarNames_.end()) {
      base = varIt->second;
    } else {
      auto tp = cudaType(v);
      if (v->type().kind() == nativert::Type::Kind::Tensor) {
        base = fmt::format("storage<{}>({})", tp, param(v, op));
      } else {
        base = param(v, op);
      }
    }
    if (v->type().kind() != nativert::Type::Kind::Tensor) {
      return fmt::format("*{}", base);
    } else if (slowPath) {
      auto bitIdx = fastPathBitIndex_[valueIdx];
      std::stringstream ls;
      ls << base << "[complexIdx(isFastPath" << bitIdx / kBitsPerWord
         << " & (1 << " << bitIdx % kBitsPerWord << "), " << param(v, op)
         << ", idx)]";
      return ls.str();
    } else {
      return fmt::format("{}[idx]", base);
    }
  };

  auto isTerminal = [&](ValueCP v) -> bool {
    return inputSet.count(v) || !v->producer();
  };

  if (isTerminal(value)) {
    code_ << "      " << resultName << " = " << leafText(value) << ";\n";
    return;
  }

  auto* node = value->producer();
  placed_.insert(node);
  generatingOp_->allNodes().insert(node);

  auto* meta = nodeMeta(node);
  TORCH_CHECK(
      meta && meta->elementwise, "Not an elementwise op: ", node->target());
  const auto& ew = *meta->elementwise;

  if (!meta->headerFile.empty()) {
    addInclude(meta->headerFile);
  }

  std::vector<std::string> argTexts;
  std::vector<std::pair<ValueCP, std::string>> tempsToRelease;
  bool firstValue = true;

  auto processValue = [&](ValueCP v, bool isWhole) {
    if (isWhole) {
      argTexts.push_back(param(v, op));
      firstValue = false;
      return;
    }

    auto* producer = v->producer();
    bool isListPack = producer && producer->target() == "prim.ListPack";

    if (isListPack) {
      placed_.insert(producer);
      generatingOp_->allNodes().insert(producer);
      for (const auto& listInput : producer->inputs()) {
        auto* lv = listInput.value;
        if (isTerminal(lv)) {
          argTexts.push_back(leafText(lv));
        } else {
          auto tempName = useTemp(lv);
          elementwiseExprImpl(lv, tempName, inputSet, inputs, op, slowPath);
          argTexts.push_back(tempName);
          tempsToRelease.emplace_back(lv, tempName);
        }
      }
    } else if (isTerminal(v)) {
      argTexts.push_back(leafText(v));
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

  if (meta->generateCall) {
    for (const auto& input : node->inputs()) {
      processValue(input.value, false);
    }
    std::stringstream callSs;
    meta->generateCall(callSs, node, std::move(argTexts));
    code_ << "      " << resultName << " = " << callSs.str() << ";\n";
    for (auto& [v, name] : tempsToRelease) {
      tempDone(v, name);
    }
    return;
  }

  int32_t emittedArgs = 0;
  forArguments(
      *meta,
      node,
      [&](size_t schemaIdx, ValueCP v, const nativert::Attribute* attr) {
        if (emittedArgs >= ew.numArgs) {
          return;
        }
        if (v) {
          ++emittedArgs;
          bool isWhole = schemaIdx < meta->argumentMeta.size() &&
              meta->argumentMeta[schemaIdx].wholeTensor;
          processValue(v, isWhole);
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

  std::stringstream callSs;
  callSs << ew.functionName;
  auto ewPresenceParams = meta->hasPresentTemplateParams()
      ? presentTemplateParams(*meta, node)
      : std::string();
  if (!meta->typeTemplateParams.empty() || meta->hasDtypeTemplateParam ||
      !ewPresenceParams.empty()) {
    callSs << "<";
    const auto& nodeInputs = node->inputs();
    bool firstTp = true;
    for (size_t i = 0; i < meta->typeTemplateParams.size(); ++i) {
      if (!firstTp) {
        callSs << ", ";
      }
      firstTp = false;
      auto idx = meta->typeTemplateParams[i];
      callSs << cudaType(nodeInputs[idx].value);
    }
    if (meta->hasDtypeTemplateParam) {
      if (!firstTp) {
        callSs << ", ";
      }
      const auto* dtypeAttr = node->tryGetAttribute("dtype");
      TORCH_CHECK(dtypeAttr, node->target(), ": missing dtype attribute");
      callSs << cudaTypeFromDtype(*dtypeAttr);
    }
    if (!ewPresenceParams.empty()) {
      if (!firstTp) {
        callSs << ", ";
      }
      callSs << ewPresenceParams;
    }
    callSs << ">";
  }
  callSs << "(";
  bool first = true;
  if (ew.hasIdxArg) {
    callSs << "idx";
    first = false;
  }
  if (ew.hasSizeArg) {
    if (!first) {
      callSs << ", ";
    }
    callSs << "size";
    first = false;
  }
  for (const auto& arg : argTexts) {
    if (!first) {
      callSs << ", ";
    }
    first = false;
    callSs << arg;
  }
  callSs << ")";
  code_ << "      " << resultName << " = " << callSs.str() << ";\n";

  for (auto& [v, name] : tempsToRelease) {
    tempDone(v, name);
  }

  auto codeEnd = static_cast<size_t>(code_.tellp());
  if (WaveConfig::get().outOfLineExprSize > 0 &&
      static_cast<int32_t>(codeEnd - codeStart) >
          WaveConfig::get().outOfLineExprSize) {
    auto fullCode = code_.str();
    auto extractedCode = fullCode.substr(codeStart, codeEnd - codeStart);

    // Collect unique temps used in the extracted range.
    std::set<std::pair<std::string, std::string>> uniqueTemps;
    for (size_t i = tempLogStart; i < tempUseLog_.size(); ++i) {
      uniqueTemps.insert(tempUseLog_[i]);
    }

    // Find which bN variables are referenced in the extracted code,
    // including transitively via any helper functions called from it.
    std::set<size_t> usedIndices;
    for (auto& [idx, varName] : elementwiseVarNames_) {
      if (extractedCode.find(varName + "[") != std::string::npos ||
          extractedCode.find("*" + varName) != std::string::npos) {
        usedIndices.insert(idx);
      }
    }
    // Add vars needed by called helpers (transitive deps).
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

    // Build the helper function.
    outOfLineFunctions_ << "__device__ __noinline__ " << returnType << " "
                        << funcName
                        << "(\n    uint32_t idx, uint32_t size, "
                           "BlockInfo& blockInfo";
    // Find how many isFastPath variables are referenced.
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
      auto tp = cudaType(inputs[idx]);
      outOfLineFunctions_ << ",\n    " << tp << "* " << varName;
    }
    outOfLineFunctions_ << ") {\n";

    // Declare local temps inside the helper. The resultName temp is also
    // declared in the main kernel (for the call site assignment).
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

    // Record this helper's variable dependencies for transitive propagation.
    helperVarDeps_[funcName] = usedIndices;

    // Replace extracted code with a function call.
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

    // Remove extracted temps from the log so they won't be double-counted.
    tempUseLog_.erase(
        tempUseLog_.begin() + static_cast<ptrdiff_t>(tempLogStart),
        tempUseLog_.end());
  }
}

void CompileCtx::elementwiseExpr(
    ValueCP value,
    const std::string& resultName,
    const KernelOperation& op,
    const std::vector<ValueCP>& inputs,
    bool slowPath) {
  addInclude("velox/experimental/torchwave/Elementwise.cuh");
  std::unordered_set<ValueCP> inputSet(inputs.begin(), inputs.end());
  inputSet.erase(value);
  elementwiseExprImpl(value, resultName, inputSet, inputs, op, slowPath);
}

} // namespace torch::wave
