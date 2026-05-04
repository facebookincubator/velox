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
#include "velox/experimental/torchwave/Compile.h"
#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/GraphOptimizer.h"
#include "velox/experimental/torchwave/ParallelExpr.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveConfig.h"

#include <ATen/ATen.h>
#include <c10/util/StringUtil.h>
#include <folly/ScopeGuard.h>
#include <gflags/gflags.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

#include "velox/experimental/wave/common/GpuArena.h"

DECLARE_bool(print_timing);
DEFINE_bool(
    debug_single_ops,
    false,
    "Launch kernel once per block for debugging, waiting after each launch");
DEFINE_string(
    debug_kernel_dir,
    "",
    "If non-empty, read kernel code from this directory instead of generating it");

namespace torch::wave {

namespace {

void fillShapeOnlyTensorParam(const at::Tensor& tensor, void* dest) {
  auto* t = reinterpret_cast<Tensor*>(dest);
  t->storage = nullptr;
  t->rank = tensor.dim();
  for (int i = 0; i < 3; ++i) {
    t->dims[i] = i < tensor.dim() ? tensor.size(i) : 0;
    t->strides[i] = 0;
  }
  t->numEl = tensor.numel();
  t->status = Tensor::kUninited;
}

void fillTensorParam(const at::Tensor& tensor, void* dest) {
  TORCH_CHECK(
      tensor.dim() <= 3,
      "Tensors with more than 3 dims not supported, got ",
      tensor.dim());
  auto* t = reinterpret_cast<Tensor*>(dest);
  t->storage = tensor.data_ptr();
  t->rank = tensor.dim();
  for (int i = 0; i < 3; ++i) {
    t->dims[i] = i < tensor.dim() ? tensor.size(i) : 0;
    t->strides[i] =
        i < tensor.dim() ? (tensor.size(i) == 1 ? 0 : tensor.stride(i)) : 0;
  }
  t->numEl = tensor.numel();
  t->status = Tensor::kUninited;
}

std::string tensorToString(const Tensor& t) {
  std::stringstream ss;
  ss << "Tensor{storage=" << t.storage << " rank=" << static_cast<int>(t.rank)
     << " dims=[";
  for (int i = 0; i < t.rank; ++i) {
    if (i > 0) {
      ss << ",";
    }
    ss << t.dims[i];
  }
  ss << "] strides=[";
  for (int i = 0; i < t.rank; ++i) {
    if (i > 0) {
      ss << ",";
    }
    ss << t.strides[i];
  }
  ss << "] numEl=" << t.numEl << " status=" << t.status << "}";
  return ss.str();
}

std::string dumpOpParams(const KernelOperation& op, uint8_t* paramBase) {
  std::stringstream ss;
  const auto& inputs = op.orderedInputs();
  auto numInputs = op.numInputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto offset = op.paramOffset(inputs[i]);
    bool isOutput = static_cast<int32_t>(i) >= numInputs;
    ss << "  " << (isOutput ? "output" : "input") << "[" << i
       << "] offset=" << offset;
    if (inputs[i]->type().kind() == nativert::Type::Kind::Tensor) {
      auto* t = reinterpret_cast<Tensor*>(paramBase + offset);
      ss << " " << tensorToString(*t);
    } else {
      ss << " scalar=" << *reinterpret_cast<int64_t*>(paramBase + offset);
    }
    ss << "\n";
  }
  auto constantOffset = op.constantAreaOffset();
  for (int32_t i = 0; i < op.numConstants(); ++i) {
    ss << "  const[" << i << "] offset=" << constantOffset
       << " value=" << *reinterpret_cast<int64_t*>(paramBase + constantOffset)
       << "\n";
    constantOffset += 8;
  }
  return ss.str();
}

void fillScalarParam(const c10::IValue& ivalue, void* dest) {
  if (ivalue.isInt()) {
    *reinterpret_cast<int64_t*>(dest) = ivalue.toInt();
  } else if (ivalue.isDouble()) {
    *reinterpret_cast<double*>(dest) = ivalue.toDouble();
  } else if (ivalue.isBool()) {
    *reinterpret_cast<int64_t*>(dest) = ivalue.toBool() ? 1 : 0;
  } else {
    TORCH_CHECK(false, "Unsupported IValue type for kernel param");
  }
}

} // namespace

at::Tensor paramTensor(
    ValueCP value,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map) {
  auto it = map.find(value->id());
  TORCH_CHECK(
      it != map.end(),
      "Input value %v",
      value->id(),
      " not found in FormalToActual map");
  return frame.getTensor(it->second);
}

int64_t paramSymInt(
    ValueCP value,
    nativert::ExecutionFrame& frame,
    const FormalToActual& map) {
  auto it = map.find(value->id());
  TORCH_CHECK(
      it != map.end(),
      "Input value %v",
      value->id(),
      " not found in FormalToActual map");
  return frame.getSymInt(it->second);
}

std::vector<std::vector<Dim>> elementwiseInputShape(
    NodeCP node,
    nativert::ExecutionFrame& frame,
    FormalToActual map,
    int32_t ordinal) {
  std::vector<int64_t> bestShape;
  int64_t bestNumel = -1;

  std::function<void(ValueCP)> trace = [&](ValueCP v) {
    auto it = map.find(v->id());
    if (it != map.end()) {
      auto& ivalue = frame.getIValue(it->second);
      if (ivalue.isTensor()) {
        auto numel = ivalue.toTensor().numel();
        if (numel > bestNumel) {
          bestNumel = numel;
          auto sizes = ivalue.toTensor().sizes();
          bestShape.assign(sizes.begin(), sizes.end());
        }
        return;
      }
    }
    auto* producer = v->producer();
    if (!producer) {
      return;
    }
    auto* meta = Registry::metadata(producer->target());
    if (meta && meta->elementwise) {
      for (const auto& input : producer->inputs()) {
        trace(input.value);
      }
    }
  };

  trace(node->inputs()[ordinal].value);

  TORCH_CHECK(
      bestNumel >= 0, "elementwiseInputShape: no tensor found in frame");
  return {std::vector<Dim>(bestShape.begin(), bestShape.end())};
}

int32_t makeGrid(
    const std::vector<LaunchData>& launches,
    StepVectors& sv,
    int32_t maxBlocksPerSM) {
  const int32_t blockSize = 256;

  // Compute cost per launch: numElements * unitCost.
  sv.costs.resize(launches.size());
  float totalCost = 0;
  for (size_t i = 0; i < launches.size(); ++i) {
    sv.costs[i] = static_cast<float>(launches[i].numElements) *
        launches[i].launch->op->unitCost();
    totalCost += sv.costs[i];
  }

  // Max blocks each launch could use.
  sv.maxBlocks.resize(launches.size());
  const int32_t elementsPerBlock = blockSize * 4;
  for (size_t i = 0; i < launches.size(); ++i) {
    sv.maxBlocks[i] = static_cast<int32_t>(
        (launches[i].numElements + elementsPerBlock - 1) / elementsPerBlock);
    if (sv.maxBlocks[i] < 1) {
      sv.maxBlocks[i] = 1;
    }
  }

  // Target blocks from device SM count and kernel occupancy.
  int32_t numSMs = WaveConfig::get().numSms;
  if (numSMs == 0) {
    numSMs = 100;
    auto* device = facebook::velox::wave::currentDevice();
    if (device) {
      numSMs = device->numSM;
    }
  }
  int32_t blocksPerSM = maxBlocksPerSM > 0 ? maxBlocksPerSM : 4;
  int32_t targetBlocks = numSMs * blocksPerSM;

  // Assign blocks pro rata by cost, at least 1 per launch, capped by
  // maxBlocks.
  sv.numBlocksPerLaunch.resize(launches.size());
  int32_t totalAssigned = 0;
  for (size_t i = 0; i < launches.size(); ++i) {
    // Operations marked alwaysSingleBlock always get exactly one block.
    if (launches[i].launch->op && launches[i].launch->op->alwaysSingleBlock()) {
      sv.numBlocksPerLaunch[i] = 1;
      totalAssigned += 1;
      continue;
    }
    float fraction =
        (totalCost > 0) ? sv.costs[i] / totalCost : 1.0f / launches.size();
    int32_t assigned =
        std::max(1, static_cast<int32_t>(fraction * targetBlocks + 0.5f));
    assigned = std::min(assigned, sv.maxBlocks[i]);
    // Round down blocks so all but the last process a multiple of blockSize
    // elements.
    if (assigned > 1) {
      auto elemsPerBlock = (launches[i].numElements + assigned - 1) / assigned;
      auto alignedElems = roundUp(elemsPerBlock, elementsPerBlock);
      assigned = std::max(
          1,
          static_cast<int32_t>(
              (launches[i].numElements + alignedElems - 1) / alignedElems));
    }
    sv.numBlocksPerLaunch[i] = assigned;
    totalAssigned += assigned;
  }

  // Fill blocks and launchIndices.
  sv.blocks.resize(totalAssigned);
  sv.launchIndices.resize(totalAssigned);
  int32_t blockIdx = 0;
  for (size_t i = 0; i < launches.size(); ++i) {
    auto opCode = launches[i].launch->op->opCode();
    auto nBlocks = sv.numBlocksPerLaunch[i];
    for (int32_t b = 0; b < nBlocks; ++b) {
      auto& info = sv.blocks[blockIdx];
      info.op = opCode;
      info.blockInOp = b;
      info.numBlocksInOp = nBlocks;
      info.params = nullptr;
      sv.launchIndices[blockIdx] = static_cast<int32_t>(i);
      ++blockIdx;
    }
  }
  return blockSize;
}

// --- Launch ---

std::string Launch::toString(Listing mode) const {
  if (op) {
    return "kernel: " + op->toString(mode);
  }
  if (standalone) {
    Subgraph sg;
    sg.root = standalone;
    for (const auto& input : standalone->inputs()) {
      sg.inputs.push_back(input.value);
    }
    return "standalone " + sg.toString(mode);
  }
  return "";
}

// --- ProjectOperation ---

ProjectOperation::ProjectOperation(const Subgraph& sg, CompileCtx& ctx)
    : subgraph_(sg) {}

namespace {

void printLaunchGrid(
    std::stringstream& ss,
    const LaunchGrid& grid,
    const char* heading,
    Listing mode) {
  if (heading) {
    ss << heading << "\n";
  }
  for (size_t step = 0; step < grid.size(); ++step) {
    ss << "Step" << step << "\n";
    for (size_t lane = 0; lane < grid[step].size(); ++lane) {
      ss << "  Lane " << lane << ": ";
      const auto& launch = grid[step][lane];
      if (launch.op) {
        ss << launch.op->toString(mode);
        ss << "    Params: (";
        for (size_t i = 0; i < launch.values.size(); ++i) {
          if (i > 0) {
            ss << ", ";
          }
          ss << "%v" << launch.values[i]->id();
        }
        ss << ")\n";
        if (!launch.constantIndices.empty()) {
          ss << "    ConstantIndices: (";
          for (size_t i = 0; i < launch.constantIndices.size(); ++i) {
            if (i > 0) {
              ss << ", ";
            }
            ss << launch.constantIndices[i];
          }
          ss << ")\n";
        }
      } else if (launch.standalone) {
        ss << "Standalone: " << launch.standalone->toString() << "\n";
      }
    }
  }
}

} // namespace

std::string ProjectOperation::toString(Listing mode) const {
  std::stringstream ss;
  printLaunchGrid(ss, grid_, nullptr, mode);
  if (!singleBlockGrid_.empty()) {
    printLaunchGrid(ss, singleBlockGrid_, "Single Block Variant", mode);
  }
  return ss.str();
}

namespace {

/// Builds a node-to-node map by walking formal and actual subgraphs in
/// parallel. Recursion stops at subgraph inputs.
void makeNodeMap(
    const Subgraph& formalSg,
    const Subgraph& actualSg,
    std::unordered_map<NodeCP, NodeCP>& nodeMap) {
  std::unordered_set<ValueCP> formalInputSet(
      formalSg.inputs.begin(), formalSg.inputs.end());
  std::function<void(NodeCP, NodeCP)> walk = [&](NodeCP formalNode,
                                                 NodeCP actualNode) {
    if (!nodeMap.emplace(formalNode, actualNode).second) {
      return;
    }
    const auto& fi = formalNode->inputs();
    const auto& ai = actualNode->inputs();
    for (size_t i = 0; i < fi.size(); ++i) {
      if (formalInputSet.count(fi[i].value)) {
        auto* fp = fi[i].value->producer();
        if (fp) {
          auto* meta = Registry::metadata(fp->target());
          if (meta && meta->isView()) {
            auto* ap = ai[i].value->producer();
            TORCH_CHECK(
                ap && Registry::metadata(ap->target()) &&
                    Registry::metadata(ap->target())->isView(),
                "Formal input has view producer but actual does not");
            nodeMap.emplace(fp, ap);
          }
        }
        continue;
      }
      auto* fp = fi[i].value->producer();
      auto* ap = ai[i].value->producer();
      if (fp && ap) {
        walk(fp, ap);
      }
    }
  };
  if (formalSg.root && actualSg.root) {
    walk(formalSg.root, actualSg.root);
  }
}

/// Builds bindings by walking formal and actual subgraphs in parallel.
/// Maps all inputs positionally and all node outputs by parallel tree walk.
FormalToActual makeSubgraphBindings(
    const Subgraph& formalSg,
    const Subgraph& actualSg) {
  TORCH_CHECK(
      formalSg.inputs.size() == actualSg.inputs.size(),
      "Input count mismatch: formal=",
      formalSg.inputs.size(),
      " actual=",
      actualSg.inputs.size());
  FormalToActual bindings;
  for (size_t i = 0; i < formalSg.inputs.size(); ++i) {
    bindings[formalSg.inputs[i]->id()] = actualSg.inputs[i]->id();
  }
  std::unordered_set<ValueCP> formalInputSet(
      formalSg.inputs.begin(), formalSg.inputs.end());
  std::unordered_set<NodeCP> visited;
  std::function<void(NodeCP, NodeCP)> walk = [&](NodeCP formalNode,
                                                 NodeCP actualNode) {
    if (!visited.insert(formalNode).second) {
      return;
    }
    const auto& fo = formalNode->outputs();
    const auto& ao = actualNode->outputs();
    TORCH_CHECK(
        fo.size() == ao.size(),
        "Output count mismatch at node ",
        formalNode->target());
    for (size_t i = 0; i < fo.size(); ++i) {
      bindings[fo[i]->id()] = ao[i]->id();
    }
    const auto& fi = formalNode->inputs();
    const auto& ai = actualNode->inputs();
    TORCH_CHECK(
        fi.size() == ai.size(),
        "Input count mismatch at node ",
        formalNode->target());
    for (size_t i = 0; i < fi.size(); ++i) {
      if (formalInputSet.count(fi[i].value)) {
        auto* fp = fi[i].value->producer();
        if (fp) {
          auto* meta = Registry::metadata(fp->target());
          if (meta && meta->isView()) {
            auto* ap = ai[i].value->producer();
            TORCH_CHECK(
                ap && Registry::metadata(ap->target()) &&
                    Registry::metadata(ap->target())->isView(),
                "Formal input has view producer but actual does not");
            const auto& fo = fp->outputs();
            const auto& ao = ap->outputs();
            for (size_t j = 0; j < fo.size(); ++j) {
              bindings[fo[j]->id()] = ao[j]->id();
            }
          }
        }
        continue;
      }
      auto* fp = fi[i].value->producer();
      auto* ap = ai[i].value->producer();
      if (fp && ap) {
        walk(fp, ap);
      }
    }
  };
  if (formalSg.root && actualSg.root) {
    walk(formalSg.root, actualSg.root);
  }
  return bindings;
}

} // namespace

OpInvocation::OpInvocation(
    ProjectOperation* projectOp,
    const Subgraph& sg,
    std::deque<c10::IValue>& storage)
    : projectOp_{projectOp} {
  const auto& formalSg = projectOp->subgraph();
  bindings_ = makeSubgraphBindings(formalSg, sg);
  constants_ = listConstants(sg, storage);
  makeNodeMap(formalSg, sg, nodeMap_);
}

// --- LaunchData ---

LaunchData::LaunchData(
    const Launch& launch,
    OpInvocation& op,
    const IdToValueMap& idToValue)
    : launch(&launch), invocation(&op), numElements(0) {
  const auto& bindings = op.bindings();

  auto translateId = [&](ValueCP formal) -> nativert::ValueId {
    auto it = bindings.find(formal->id());
    auto actualId = it != bindings.end() ? it->second : formal->id();
    return actualId;
  };

  if (!launch.op) {
    // Standalone: translate node via nodeMap, inputs and outputs via bindings.
    auto nodeIt = op.nodeMap().find(launch.standalone);
    TORCH_CHECK(
        nodeIt != op.nodeMap().end(), "Standalone node not found in nodeMap");
    standalone = nodeIt->second;
    for (auto& input : launch.standalone->inputs()) {
      actualInputs.push_back(translateId(input.value));
    }
    for (auto* output : launch.standalone->outputs()) {
      actualOutputs.push_back(translateId(output));
    }
  } else {
    // Kernel op: translate sizeExpr, inputs, outputs, and output descs.
    auto* kernelOp = launch.op;
    sizeExpr = kernelOp->sizeExpr().toActual(bindings, idToValue);

    const auto& orderedInputs = kernelOp->orderedInputs();
    auto nInputs = kernelOp->numInputs();
    for (int32_t i = 0; i < nInputs; ++i) {
      actualInputs.push_back(translateId(orderedInputs[i]));
    }
    for (size_t i = nInputs; i < orderedInputs.size(); ++i) {
      actualOutputs.push_back(translateId(orderedInputs[i]));
    }

    const auto& outputDescs = kernelOp->outputDescs();
    for (size_t i = 0; i < outputDescs.size(); ++i) {
      const auto& desc = outputDescs[i];
      OutputDesc actualDesc = desc;
      actualDesc.sizeExpr = desc.sizeExpr.toActual(bindings, idToValue);
      if (desc.storageFrom) {
        actualDesc.storageFrom = idToValue.at(translateId(desc.storageFrom));
      }
      if (desc.viewNode) {
        auto viewIt = op.nodeMap().find(desc.viewNode);
        TORCH_CHECK(
            viewIt != op.nodeMap().end(), "View node not found in nodeMap");
        actualDesc.viewNode = viewIt->second;
      }
      // Non-tensor outputs (scalars, SymInt, etc.) must be read back to host.
      auto outputValueId = actualOutputs[i];
      auto outputValueIt = idToValue.find(outputValueId);
      TORCH_CHECK(
          outputValueIt != idToValue.end(),
          "Output value id not found in idToValue: ",
          outputValueId);
      if (outputValueIt->second->type().kind() !=
          nativert::Type::Kind::Tensor) {
        actualDesc.neededOnHost = true;
      }
      if (actualDesc.shapeSetOnDevice || actualDesc.neededOnHost) {
        returnValues.push_back(actualOutputs[i]);
        returnTypes.push_back(outputValueIt->second->type().kind());
      }
      actualOutputDescs.push_back(std::move(actualDesc));
    }

    // Record the type kind from each output Value.
    for (size_t i = 0; i < actualOutputs.size(); ++i) {
      auto outputId = actualOutputs[i];
      auto it = idToValue.find(outputId);
      TORCH_CHECK(
          it != idToValue.end(),
          "Output value id not found in idToValue: ",
          outputId);
      actualOutputTypes.push_back(it->second->type().kind());
    }
  }
}

// --- CompositeKernel ---

CompositeKernel::CompositeKernel(
    std::vector<std::unique_ptr<ProjectOperation>>&& ops,
    std::vector<std::unique_ptr<KernelOperation>>&& kernelOps,
    const std::unordered_set<std::string>& includes)
    : ops_(std::move(ops)), kernelOpStorage_(std::move(kernelOps)) {
  auto kernelId = CompileCtx::nextKernelId();
  auto kernelName = "torchwave" + std::to_string(kernelId);
  auto entryPoint = "torch::wave::" + kernelName;

  std::stringstream ss;
  ss << "#include \"velox/experimental/torchwave/Core.cuh\"\n";
  for (const auto& inc : includes) {
    ss << "#include \"" << inc << "\"\n";
  }
  ss << "\nnamespace torch::wave {\n\n"
     << "__global__ void " << kernelName << "(TorchWaveParams params) {\n"
     << "  ENTRY;\n";
  eltTrace(
      ss,
      "\"entry blockIdx %d op %d blockInOp %d\\n\", blockIdx.x, blockInfo.op, blockInfo.blockInOp");
  std::unordered_set<std::string> emittedDecls;
  for (const auto& kop : kernelOpStorage_) {
    for (const auto& decl : kop->sharedDeclarations()) {
      if (emittedDecls.insert(decl).second) {
        ss << decl;
      }
    }
  }
  ss << "  switch (blockInfo.op) {\n";
  for (const auto& kop : kernelOpStorage_) {
    ss << "    case " << kop->opCode() << ": {\n";

    // Build parallel paramOffsets/outputOffsets/altOffsets arrays from
    // ElementExprs and remaining orderedInputs.
    std::vector<int32_t> paramOffs;
    std::vector<int32_t> outputOffs;
    std::vector<int32_t> altOffs;
    std::unordered_set<ValueCP> inElementExpr;

    for (const auto& ee : kop->elementExprs()) {
      int32_t outputOff = kop->paramOffset(ee.output);
      for (auto* v : ee.inputs) {
        if (v->type().kind() != nativert::Type::Kind::Tensor) {
          continue;
        }
        inElementExpr.insert(v);
        paramOffs.push_back(kop->paramOffset(v));
        outputOffs.push_back(outputOff);
        auto ait = ee.altParamOffset.find(v);
        altOffs.push_back(ait != ee.altParamOffset.end() ? ait->second : -1);
      }
      if (ee.output->type().kind() == nativert::Type::Kind::Tensor) {
        inElementExpr.insert(ee.output);
        paramOffs.push_back(outputOff);
        outputOffs.push_back(outputOff);
        altOffs.push_back(-1);
      }
    }

    for (const auto* value : kop->orderedInputs()) {
      if (value->type().kind() == nativert::Type::Kind::Tensor &&
          !inElementExpr.count(value)) {
        auto off = kop->paramOffset(value);
        paramOffs.push_back(off);
        outputOffs.push_back(off);
        altOffs.push_back(-1);
      }
    }

    if (!paramOffs.empty()) {
      auto emitArray = [&](const char* name, const std::vector<int32_t>& arr) {
        ss << "    static int32_t " << name << "[] = {";
        for (size_t i = 0; i < arr.size(); ++i) {
          if (i > 0) {
            ss << ", ";
          }
          ss << arr[i];
        }
        ss << "};\n";
      };
      ss << "  {\n";
      emitArray("paramOffsets", paramOffs);
      emitArray("outputOffsets", outputOffs);
      emitArray("altOffsets", altOffs);
      ss << "    for (auto i = threadIdx.x; i < sizeof(paramOffsets) / sizeof(paramOffsets[0]); i += blockDim.x) {\n"
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
    ss << kop->code() << "      break;\n"
       << "    }\n";
  }
  ss << "  }\n"
     << "  LEAVE();\n"
     << "}\n\n"
     << "} // namespace torch::wave\n";

  auto code = ss.str();

  // Save to /tmp for debugging.
  auto filePath = "/tmp/kernel" + std::to_string(kernelId) + ".cu";
  {
    std::ofstream out(filePath);
    out << code;
  }

  if (!FLAGS_debug_kernel_dir.empty()) {
    auto debugPath =
        FLAGS_debug_kernel_dir + "/kernel" + std::to_string(kernelId) + ".cu";
    std::ifstream in(debugPath);
    TORCH_CHECK(in.good(), "Could not open ", debugPath);
    code = std::string(
        std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    LOG(INFO) << "reading code of kernel" << kernelId << " from " << debugPath;
  }

  // Only compile the kernel if a GPU is available.
  if (facebook::velox::wave::currentDevice()) {
    kernel_ = facebook::velox::wave::CompiledKernel::getKernel(
        entryPoint,
        [code = std::move(code),
         entryPoint,
         filePath]() -> facebook::velox::wave::KernelSpec {
          facebook::velox::wave::KernelSpec spec;
          spec.code = code;
          spec.entryPoints = {entryPoint};
          spec.filePath = filePath;
          spec.numHeaders = 0;
          spec.headers = nullptr;
          return spec;
        });

    LOG(INFO) << "Kernel " << kernelName << ": " << kernel_->info(0).toString();

    // Warmup launch: run one block of 1 thread with op=-1 (no-op) to absorb
    // first-launch overhead from the CUDA runtime.
    {
      Timer warmup("warmup", FLAGS_print_timing);
      TorchWaveParams params;
      memset(&params, 0, sizeof(params));
      params.info = nullptr;
      params.debugInfo = nullptr;
      params.inlineInfo[0].op = -1;
      void* args[] = {&params};
      facebook::velox::wave::Stream stream;
      kernel_->launch(0, 1, 1, 0, &stream, args);
      stream.wait();
    }
  }
}

facebook::velox::wave::KernelInfo CompositeKernel::kernelInfo() const {
  if (kernel_) {
    return kernel_->info(0);
  }
  return {};
}

void CompositeKernel::launch(
    int32_t numBlocks,
    int32_t numThreads,
    int32_t sharedMemory,
    facebook::velox::wave::Stream* stream,
    void** args) {
  kernel_->launch(0, numBlocks, numThreads, sharedMemory, stream, args);
}

void CompositeKernel::launchCooperative(
    int32_t numBlocks,
    int32_t numThreads,
    int32_t sharedMemory,
    facebook::velox::wave::Stream* stream,
    void** args) {
  kernel_->launchCooperative(
      0, numBlocks, numThreads, sharedMemory, stream, args);
}

std::string CompositeKernel::toString(Listing mode) const {
  std::stringstream ss;
  for (const auto& kop : kernelOpStorage_) {
    ss << kop->toString(mode);
  }
  auto info = kernelInfo();
  if (info.numRegs > 0) {
    ss << info.toString() << "\n";
  }
  return ss.str();
}

namespace {

void fillTensorListParam(
    LaunchData& launch,
    nativert::ExecutionFrame& frame,
    uint8_t* paramBase,
    const KernelOperation& kernelOp,
    ValueCP listValue,
    const FormalToActual& bindings,
    std::unordered_set<int32_t>& filledOffsets) {
  auto elements = listValue->getListElements();
  TensorListParam tlp;
  tlp.listOffset = kernelOp.paramOffset(listValue);
  for (auto* elem : elements) {
    auto elemId = elem->id();
    auto it = bindings.find(elemId);
    auto actualId = it != bindings.end() ? it->second : elemId;
    auto elemOffset = kernelOp.paramOffset(elem);
    tlp.elementOffsets.push_back(elemOffset);
    tlp.elementIds.push_back(actualId);
    if (filledOffsets.insert(elemOffset).second) {
      fillTensorParam(
          frame.getIValue(actualId).toTensor(), paramBase + elemOffset);
      launch.tensorsInFrame.push_back(actualId);
      launch.tensorOffsets.push_back(elemOffset);
    }
  }
  // Write TensorList struct header. The tensors pointer array follows
  // the struct and is patched with device-side addresses later.
  auto* tl = reinterpret_cast<TensorList*>(paramBase + tlp.listOffset);
  tl->size = elements.size();
  tl->tensors = nullptr;
  launch.tensorLists.push_back(std::move(tlp));
}

void patchTensorListPointers(
    const LaunchData& launch,
    uint8_t* paramBase,
    uint8_t* deviceBase) {
  for (const auto& tlp : launch.tensorLists) {
    auto* tl = reinterpret_cast<TensorList*>(paramBase + tlp.listOffset);
    auto* ptrArray = reinterpret_cast<Tensor**>(
        paramBase + tlp.listOffset + sizeof(TensorList));
    tl->tensors = reinterpret_cast<Tensor**>(
        deviceBase + tlp.listOffset + sizeof(TensorList));
    for (size_t j = 0; j < tlp.elementOffsets.size(); ++j) {
      ptrArray[j] =
          reinterpret_cast<Tensor*>(deviceBase + tlp.elementOffsets[j]);
    }
  }
}

void fillLaunchParams(
    LaunchData& launch,
    nativert::ExecutionFrame& frame,
    uint8_t* paramBase,
    int32_t& returnBegin,
    int32_t& returnEnd) {
  if (!launch.tensorsInFrame.empty() || !launch.scalarsInFrame.empty()) {
    // Cached path: fill only variable tensors and scalars, skip constants.
    for (size_t i = 0; i < launch.tensorsInFrame.size(); ++i) {
      if (launch.shapeOnlyTensorIndices.count(i)) {
        fillShapeOnlyTensorParam(
            frame.getIValue(launch.tensorsInFrame[i]).toTensor(),
            paramBase + launch.tensorOffsets[i]);
      } else {
        fillTensorParam(
            frame.getIValue(launch.tensorsInFrame[i]).toTensor(),
            paramBase + launch.tensorOffsets[i]);
      }
    }
    for (size_t i = 0; i < launch.scalarsInFrame.size(); ++i) {
      fillScalarParam(
          frame.getIValue(launch.scalarsInFrame[i]),
          paramBase + launch.scalarOffsets[i]);
    }
    for (auto offset : launch.launch->op->barrierCounters()) {
      *reinterpret_cast<int32_t*>(paramBase + offset) = 0;
    }
    if (!launch.returnValues.empty()) {
      if (returnBegin == -1) {
        returnBegin = launch.returnOffsets.front();
      }
      auto lastType = launch.returnTypes.back();
      int32_t lastSize =
          lastType == nativert::Type::Kind::Tensor ? sizeof(Tensor) : 8;
      returnEnd = launch.returnOffsets.back() + lastSize;
    }
    if (WaveConfig::get().trace & WaveConfig::kFrame) {
      for (size_t i = 0; i < launch.tensorsInFrame.size(); ++i) {
        std::cout << "  v" << launch.tensorsInFrame[i] << " = "
                  << launch.tensorOffsets[i] << std::endl;
      }
      for (size_t i = 0; i < launch.scalarsInFrame.size(); ++i) {
        std::cout << "  v" << launch.scalarsInFrame[i] << " = "
                  << launch.scalarOffsets[i] << std::endl;
      }
    }
    return;
  }

  auto* kernelOp = launch.launch->op;
  const auto& orderedInputs = kernelOp->orderedInputs();
  auto numInputs = kernelOp->numInputs();
  const auto& bindings = launch.invocation->bindings();

  // Track which tensor offsets have been filled to avoid duplicates
  // when multiple TensorLists share elements.
  std::unordered_set<int32_t> filledOffsets;

  // Fill input params, recording tensor/scalar values and their offsets.
  int32_t returnCounter = 0;
  for (int32_t i = 0; i < numInputs; ++i) {
    auto* formalValue = orderedInputs[i];
    if (formalValue->type().kind() == nativert::Type::Kind::TensorList) {
      fillTensorListParam(
          launch,
          frame,
          paramBase,
          *kernelOp,
          formalValue,
          bindings,
          filledOffsets);
      continue;
    }
    auto offset = kernelOp->paramOffset(formalValue);
    auto* dest = paramBase + offset;
    auto actualId = launch.actualInputs[i];
    const auto& ivalue = frame.getIValue(actualId);
    if (ivalue.isTensor()) {
      fillTensorParam(ivalue.toTensor(), dest);
      launch.tensorsInFrame.push_back(actualId);
      launch.tensorOffsets.push_back(offset);
    } else {
      fillScalarParam(ivalue, dest);
      launch.scalarsInFrame.push_back(actualId);
      launch.scalarOffsets.push_back(offset);
    }
    if (returnCounter < static_cast<int32_t>(launch.returnValues.size()) &&
        actualId == launch.returnValues[returnCounter]) {
      launch.returnOffsets.push_back(offset);
      if (returnBegin == -1) {
        returnBegin = offset;
      }
      returnEnd = offset +
          (ivalue.isTensor() ? static_cast<int32_t>(sizeof(Tensor)) : 8);
      ++returnCounter;
    }
  }

  // Fill output params, recording values and offsets.
  for (size_t i = 0; i < launch.actualOutputs.size(); ++i) {
    auto* formalValue = orderedInputs[numInputs + i];
    if (formalValue->type().kind() == nativert::Type::Kind::TensorList) {
      fillTensorListParam(
          launch,
          frame,
          paramBase,
          *kernelOp,
          formalValue,
          bindings,
          filledOffsets);
      continue;
    }
    auto offset = kernelOp->paramOffset(formalValue);
    auto* dest = paramBase + offset;
    auto actualId = launch.actualOutputs[i];
    bool isTensorOutput = i < launch.actualOutputTypes.size() &&
        launch.actualOutputTypes[i] == nativert::Type::Kind::Tensor;
    if (isTensorOutput) {
      const auto& ivalue = frame.getIValue(actualId);
      TORCH_CHECK(ivalue.isTensor(), "Expected tensor for output param");
      bool isShapeOnly = i < launch.actualOutputDescs.size() &&
          launch.actualOutputDescs[i].shapeOnly;
      if (isShapeOnly) {
        fillShapeOnlyTensorParam(ivalue.toTensor(), dest);
        launch.shapeOnlyTensorIndices.insert(launch.tensorsInFrame.size());
      } else {
        fillTensorParam(ivalue.toTensor(), dest);
      }
      launch.tensorsInFrame.push_back(actualId);
      launch.tensorOffsets.push_back(offset);
    } else {
      // Non-tensor output: write a 64-bit zero placeholder.
      *reinterpret_cast<int64_t*>(dest) = 0;
      launch.scalarsInFrame.push_back(actualId);
      launch.scalarOffsets.push_back(offset);
    }
    if (returnCounter < static_cast<int32_t>(launch.returnValues.size()) &&
        actualId == launch.returnValues[returnCounter]) {
      launch.returnOffsets.push_back(offset);
      if (returnBegin == -1) {
        returnBegin = offset;
      }
      returnEnd =
          offset + (isTensorOutput ? static_cast<int32_t>(sizeof(Tensor)) : 8);
      ++returnCounter;
    }
  }

  // Fill constant params (first time only, constants don't change).
  auto constantOffset = kernelOp->constantAreaOffset();
  const auto& opConstants = launch.invocation->constants();
  for (auto idx : launch.launch->constantIndices) {
    auto* dest = paramBase + constantOffset;
    fillScalarParam(*opConstants[idx], dest);
    constantOffset += 8;
  }

  // Set alternate param tensors to kUninited.
  auto [altStart, numAlt] = kernelOp->altParams();
  for (int32_t i = 0; i < numAlt; ++i) {
    auto* t =
        reinterpret_cast<Tensor*>(paramBase + altStart + i * sizeof(Tensor));
    t->status = Tensor::kUninited;
  }

  for (auto offset : kernelOp->barrierCounters()) {
    *reinterpret_cast<int32_t*>(paramBase + offset) = 0;
  }

  if (WaveConfig::get().trace & WaveConfig::kFrame) {
    for (size_t i = 0; i < launch.tensorsInFrame.size(); ++i) {
      std::cout << "  v" << launch.tensorsInFrame[i] << " = "
                << launch.tensorOffsets[i] << std::endl;
    }
    for (size_t i = 0; i < launch.scalarsInFrame.size(); ++i) {
      std::cout << "  v" << launch.scalarsInFrame[i] << " = "
                << launch.scalarOffsets[i] << std::endl;
    }
  }
}

void traceTensor(
    nativert::ValueId actualId,
    c10::IntArrayRef dims,
    const char* action) {
  if (!(WaveConfig::get().trace & WaveConfig::kTensors)) {
    return;
  }
  auto t =
      at::empty(dims, at::TensorOptions().dtype(at::kFloat).device(at::kMeta));
  std::cout << "  tensor v" << actualId << " " << action << " "
            << traceIValue(c10::IValue(t)) << std::endl;
}

void traceView(
    nativert::ValueId baseId,
    nativert::ValueId viewId,
    const c10::IValue& viewValue) {
  if (!(WaveConfig::get().trace & WaveConfig::kTensors)) {
    return;
  }
  std::cout << "  view v" << viewId << " of v" << baseId << " "
            << traceIValue(viewValue) << std::endl;
}

void ensureCudaTensor(
    nativert::ExecutionFrame& frame,
    const ValueTypes& types,
    nativert::ValueId actualId,
    c10::IntArrayRef dims) {
  auto& existing = frame.getIValue(actualId);
  if (existing.isTensor() && existing.toTensor().is_cuda()) {
    auto& tensor = existing.toTensor();
    if (tensor.sizes() != dims) {
      traceTensor(actualId, dims, "resize");
      tensor.resize_(dims);
    } else {
      traceTensor(actualId, dims, "keep");
    }
  } else {
    auto* meta = types.types[actualId];
    if (!meta) {
      return;
    }
    traceTensor(actualId, dims, "alloc");
    auto tensor = at::empty(
        dims, at::TensorOptions().dtype(meta->dtype()).device(at::kCUDA));
    frame.setIValue(actualId, std::move(tensor));
  }
}

void allocateLaunchOutputs(
    const LaunchData& launch,
    nativert::ExecutionFrame& frame,
    const ValueTypes& types,
    nativert::ValueId largestId,
    const std::unordered_map<NodeCP, nativert::OpKernel*>* kernelMap,
    const IdToValueMap& idToValue) {
  const auto& descs = launch.actualOutputDescs;
  const auto& actualOutputs = launch.actualOutputs;
  const auto& outputTypes = launch.actualOutputTypes;

  // Shortcut: if largestId is set, resize tensor outputs to match it.
  if (largestId >= 0) {
    auto dims = frame.getIValue(largestId).toTensor().sizes();
    for (size_t i = 0; i < descs.size(); ++i) {
      if (i < outputTypes.size() &&
          outputTypes[i] != nativert::Type::Kind::Tensor) {
        continue;
      }
      if (descs[i].delegated) {
        continue;
      }
      auto actualId = actualOutputs[i];
      if (descs[i].viewNode && kernelMap) {
        auto it = kernelMap->find(descs[i].viewNode);
        TORCH_CHECK(
            it != kernelMap->end(),
            "No kernel for view node ",
            descs[i].viewNode->target());
        it->second->compute(frame);
        if (WaveConfig::get().trace & WaveConfig::kTensors) {
          auto baseId = descs[i].viewNode->inputs()[0].value->id();
          traceView(baseId, actualId, frame.getIValue(actualId));
        }
        continue;
      }
      ensureCudaTensor(frame, types, actualId, dims);
    }
    return;
  }

  const auto& bindings = launch.invocation->bindings();
  for (size_t i = 0; i < descs.size(); ++i) {
    // Skip non-tensor and non-tensor-list outputs.
    if (i < outputTypes.size() &&
        outputTypes[i] != nativert::Type::Kind::Tensor &&
        outputTypes[i] != nativert::Type::Kind::TensorList) {
      continue;
    }
    if (descs[i].delegated) {
      continue;
    }
    auto actualId = actualOutputs[i];
    if (descs[i].viewNode && kernelMap) {
      auto it = kernelMap->find(descs[i].viewNode);
      TORCH_CHECK(
          it != kernelMap->end(),
          "No kernel for view node ",
          descs[i].viewNode->target());
      it->second->compute(frame);
      if (WaveConfig::get().trace & WaveConfig::kTensors) {
        auto baseId = descs[i].viewNode->inputs()[0].value->id();
        traceView(baseId, actualId, frame.getIValue(actualId));
      }
      continue;
    }

    // TensorList output: expand to component Values and allocate each.
    if (i < outputTypes.size() &&
        outputTypes[i] == nativert::Type::Kind::TensorList) {
      if (!descs[i].reserveShape) {
        continue;
      }
      auto shapes = descs[i].reserveShape(nullptr, frame, bindings);
      auto valueIt = idToValue.find(actualId);
      TORCH_CHECK(
          valueIt != idToValue.end(),
          "TensorList output value not found: ",
          actualId);
      auto elements = valueIt->second->getListElements();
      TORCH_CHECK(
          shapes.size() == elements.size(),
          "reserveShape returned ",
          shapes.size(),
          " shapes but TensorList has ",
          elements.size(),
          " elements");
      for (size_t j = 0; j < elements.size(); ++j) {
        auto elemId = elements[j]->id();
        auto elemActualIt = bindings.find(elemId);
        auto elemActualId =
            elemActualIt != bindings.end() ? elemActualIt->second : elemId;
        std::vector<int64_t> dims(shapes[j].begin(), shapes[j].end());
        ensureCudaTensor(frame, types, elemActualId, dims);
      }
      continue;
    }

    std::vector<int64_t> dims;
    if (descs[i].reserveShape) {
      auto shapes = descs[i].reserveShape(nullptr, frame, bindings);
      TORCH_CHECK(
          !shapes.empty(),
          "OutputReserveFunc returned empty shapes for output ",
          i);
      dims.assign(shapes[0].begin(), shapes[0].end());
    } else if (descs[i].sizeExpr.op != SizeShortcut::kNone) {
      dims = {descs[i].sizeExpr.numElements(&frame)};
    } else {
      continue;
    }
    if (descs[i].shapeOnly) {
      auto& existing = frame.getIValue(actualId);
      if (existing.isTensor()) {
        auto& tensor = existing.toTensor();
        if (tensor.sizes() != dims) {
          tensor.resize_(dims);
        }
      } else {
        auto tensor = at::empty(
            dims, at::TensorOptions().dtype(at::kFloat).device(at::kMeta));
        frame.setIValue(actualId, std::move(tensor));
      }
      continue;
    }
    ensureCudaTensor(frame, types, actualId, dims);
  }
}

int32_t launchParamSize(const LaunchData& launch) {
  return launch.launch->op->altParamOffset();
}

facebook::velox::wave::WaveBufferPtr& getOrAllocateBuffer(
    std::vector<std::vector<facebook::velox::wave::WaveBufferPtr>>& buffers,
    int32_t sequenceNumber,
    int32_t stepIdx,
    int64_t requiredBytes,
    facebook::velox::wave::GpuArena* arena) {
  if (static_cast<int32_t>(buffers.size()) <= sequenceNumber) {
    buffers.resize(sequenceNumber + 1);
  }
  auto& steps = buffers[sequenceNumber];
  if (static_cast<int32_t>(steps.size()) <= stepIdx) {
    steps.resize(stepIdx + 1);
  }
  auto& buffer = steps[stepIdx];
  if (!buffer || buffer->capacity() < static_cast<size_t>(requiredBytes)) {
    buffer = arena->allocateBytes(requiredBytes);
  }
  return buffer;
}

StepVectors& getStepVectors(
    std::vector<std::vector<StepVectors>>& allSteps,
    int32_t sequenceNumber,
    int32_t stepIdx) {
  if (static_cast<int32_t>(allSteps.size()) <= sequenceNumber) {
    allSteps.resize(sequenceNumber + 1);
  }
  auto& steps = allSteps[sequenceNumber];
  if (static_cast<int32_t>(steps.size()) <= stepIdx) {
    steps.resize(stepIdx + 1);
  }
  return steps[stepIdx];
}

// Returns true if every launch's numElements falls within the cached
// [sizesLower, sizesUpper] bounds.
bool gridSizesMatch(
    const std::vector<LaunchData>& launches,
    const StepVectors& sv) {
  if (!sv.hasGridCache || launches.size() != sv.sizesLower.size()) {
    return false;
  }
  for (size_t i = 0; i < launches.size(); ++i) {
    auto n = launches[i].numElements;
    if (n < sv.sizesLower[i] || n > sv.sizesUpper[i]) {
      return false;
    }
  }
  return true;
}

// Updates the cached size bounds to [size - size/8, size + size/8] for each
// launch.
void updateGridSizeBounds(
    const std::vector<LaunchData>& launches,
    StepVectors& sv) {
  sv.sizesLower.resize(launches.size());
  sv.sizesUpper.resize(launches.size());
  for (size_t i = 0; i < launches.size(); ++i) {
    auto n = launches[i].numElements;
    auto margin = n / 8;
    sv.sizesLower[i] = n - margin;
    sv.sizesUpper[i] = n + margin;
  }
  sv.hasGridCache = true;
}

// When sizeExpr is kNone (0-arg elementwise), gets numElements from the
// single output desc's reserveShape.
int64_t numElementsFromReserve(
    LaunchData& data,
    nativert::ExecutionFrame& frame) {
  const auto& descs = data.actualOutputDescs;
  TORCH_CHECK(
      descs.size() == 1,
      "sizeExpr is kNone but kernel op has ",
      descs.size(),
      " output descs, expected 1");
  TORCH_CHECK(
      descs[0].reserveShape,
      "sizeExpr is kNone but output desc has no reserveShape");
  const auto& bindings = data.invocation->bindings();
  auto shapes = descs[0].reserveShape(nullptr, frame, bindings);
  TORCH_CHECK(!shapes.empty(), "reserveShape returned empty shapes");
  int64_t numElements = 1;
  for (auto dim : shapes[0]) {
    numElements *= dim;
  }
  return numElements;
}

} // namespace

// --- CompositeInvocation ---

void CompositeInvocation::gatherLaunches(
    const ExecutionState& state,
    std::vector<GridChoice>& grids,
    int32_t stepIdx,
    StepVectors& sv) {
  const auto& idToValue = state.waveGraph->idToValue();
  int32_t kernelIdx = 0;
  int32_t standaloneIdx = 0;
  sv.gridChanged = false;
  for (size_t i = 0; i < ops.size(); ++i) {
    auto* grid = grids[i].grid;
    if (stepIdx >= static_cast<int32_t>(grid->size())) {
      continue;
    }
    auto* step = &(*grid)[stepIdx];
    for (size_t j = 0; j < step->size(); ++j) {
      auto& launch = (*step)[j];
      if (launch.op != nullptr) {
        bool isNew = kernelIdx >= static_cast<int32_t>(sv.kernels.size());
        if (isNew) {
          sv.kernels.emplace_back(launch, ops[i], idToValue);
        }
        auto& data = sv.kernels[kernelIdx];
        bool hasByLargestInput = !data.launch->op->outputDescs().empty() &&
            data.launch->op->outputDescs()[0].byLargestInput;
        nativert::ValueId largestId = -1;
        if (data.sizeExpr.op == SizeShortcut::kNone) {
          data.numElements = numElementsFromReserve(data, *state.frame);
        } else {
          data.numElements = data.sizeExpr.numElements(
              state.frame, hasByLargestInput ? &largestId : nullptr);
        }

        if (launch.op->isGridChoice()) {
          auto* projectOp = ops[i].projectOp();
          bool wantSingleBlock;
          if (WaveConfig::get().useSingleBlock.has_value()) {
            wantSingleBlock = *WaveConfig::get().useSingleBlock;
          } else {
            wantSingleBlock =
                data.numElements <= projectOp->singleBlockMaxSize();
          }
          LaunchGrid* newGrid = nullptr;
          if (wantSingleBlock != grids[i].singleBlock) {
            if (wantSingleBlock) {
              newGrid = &projectOp->singleBlockGrid();
            } else {
              newGrid = &projectOp->grid();
            }
            grids[i].singleBlock = wantSingleBlock;
          }
          if (!wantSingleBlock && !projectOp->cgGrid().empty() &&
              WaveConfig::get().isCg.has_value() && *WaveConfig::get().isCg) {
            newGrid = &projectOp->cgGrid();
          }
          if (newGrid && newGrid != grids[i].grid) {
            grids[i].grid = newGrid;
            grid = newGrid;
            step = &(*grid)[stepIdx];
            if (!isNew) {
              sv.gridChanged = true;
            }
            auto& newLaunch = (*grids[i].grid)[stepIdx][j];
            data = LaunchData(newLaunch, ops[i], idToValue);
            largestId = -1;
            if (data.sizeExpr.op == SizeShortcut::kNone) {
              data.numElements = numElementsFromReserve(data, *state.frame);
            } else {
              data.numElements = data.sizeExpr.numElements(
                  state.frame, hasByLargestInput ? &largestId : nullptr);
            }
          }
        }

        allocateLaunchOutputs(
            data,
            *state.frame,
            *state.valueTypes,
            largestId,
            state.kernelMap,
            idToValue);
        ++kernelIdx;
      } else {
        if (standaloneIdx >= static_cast<int32_t>(sv.standalones.size())) {
          sv.standalones.emplace_back(launch, ops[i], idToValue);
        }
        ++standaloneIdx;
      }
    }
  }
}

void invalidateStepVectors(std::vector<StepVectors>& steps, int32_t stepIdx) {
  for (auto i = stepIdx; i < static_cast<int32_t>(steps.size()); ++i) {
    auto& sv = steps[i];
    sv.hasGridCache = false;
    sv.hasLaunchCache = false;
    for (auto& data : sv.kernels) {
      data.tensorsInFrame.clear();
      data.tensorOffsets.clear();
      data.scalarsInFrame.clear();
      data.scalarOffsets.clear();
    }
  }
}

void CompositeInvocation::processReturnData(
    StepVectors& sv,
    nativert::ExecutionFrame& frame,
    uint8_t* pinnedBase) {
  bool trace = WaveConfig::get().trace & WaveConfig::kTensors;
  for (size_t i = 0; i < sv.kernels.size(); ++i) {
    auto& data = sv.kernels[i];
    if (data.returnValues.empty()) {
      continue;
    }
    for (size_t j = 0; j < data.returnValues.size(); ++j) {
      auto actualId = data.returnValues[j];
      auto absOffset = sv.paramOffsets[i] + data.returnOffsets[j];
      auto typeKind = data.returnTypes[j];
      if (typeKind == nativert::Type::Kind::Tensor) {
        auto* t = reinterpret_cast<Tensor*>(pinnedBase + absOffset);
        auto& tensor = frame.getIValue(actualId).toTensor();
        std::vector<int64_t> newDims(t->rank);
        for (int d = 0; d < t->rank; ++d) {
          newDims[d] = t->dims[d];
        }
        tensor.resize_(newDims);
        if (trace) {
          std::cout << "  D2H: %" << actualId << " = "
                    << traceIValue(c10::IValue(tensor)) << std::endl;
        }
      } else if (typeKind == nativert::Type::Kind::SymFloat) {
        auto val = *reinterpret_cast<double*>(pinnedBase + absOffset);
        frame.setIValue(actualId, c10::IValue(val));
        if (trace) {
          std::cout << "  D2H: %" << actualId << " = " << val << std::endl;
        }
      } else if (typeKind == nativert::Type::Kind::SymBool) {
        auto val = *reinterpret_cast<int64_t*>(pinnedBase + absOffset) != 0;
        frame.setIValue(actualId, c10::IValue(val));
        if (trace) {
          std::cout << "  D2H: %" << actualId << " = "
                    << (val ? "true" : "false") << std::endl;
        }
      } else {
        auto val = *reinterpret_cast<int64_t*>(pinnedBase + absOffset);
        frame.setIValue(actualId, c10::IValue(val));
        if (trace) {
          std::cout << "  D2H: %" << actualId << " = " << val << std::endl;
        }
      }
    }
  }
}

void CompositeInvocation::execute(ExecutionState& state) {
  Timer ex("comp inv execute", FLAGS_print_timing);
  auto& frame = *state.frame;

  if (WaveConfig::get().trace) {
    std::cout << "==== Node " << sequenceNumber << std::endl;
  }

  auto& sv0 = getStepVectors(state.stepVectors, sequenceNumber, 0);
  auto& gridChoices = sv0.gridChoices;
  if (gridChoices.empty()) {
    for (auto& op : ops) {
      gridChoices.push_back({0, false, &op.projectOp()->grid()});
    }
  }

  int32_t blockSize;
  for (int32_t stepIdx = 0;; ++stepIdx) {
    auto& sv = getStepVectors(state.stepVectors, sequenceNumber, stepIdx);
    // Re-fetch since the resize above may have invalidated the reference.
    auto& gridChoices = state.stepVectors[sequenceNumber][0].gridChoices;

    {
      Timer t("gather", FLAGS_print_timing);
      gatherLaunches(state, gridChoices, stepIdx, sv);
    }
    if (sv.gridChanged) {
      invalidateStepVectors(state.stepVectors[sequenceNumber], stepIdx);
    }
    if (sv.kernels.empty() && sv.standalones.empty()) {
      break;
    }

    if (!sv.standalones.empty()) {
      runStandalones(
          sv.standalones,
          state,
          *state.kernelMap,
          *state.standaloneIndices,
          *state.standaloneStats);
    }

    if (sv.kernels.empty()) {
      continue;
    }

    {
      Timer t("grid", FLAGS_print_timing);
      // Reuse previous makeGrid result if sizes are within cached bounds.
      if (gridSizesMatch(sv.kernels, sv)) {
        blockSize = sv.cachedBlockSize;
      } else {
        blockSize = makeGrid(sv.kernels, sv, kernelInfo.maxOccupancy0);
        TORCH_CHECK(
            (blockSize & (blockSize - 1)) == 0,
            "Block size must be a power of two, got ",
            blockSize);
        sv.cachedBlockSize = blockSize;
        updateGridSizeBounds(sv.kernels, sv);
      }
    }

    // Compute total param bytes and track per-launch param offsets.
    auto numBlocks = sv.blocks.size();
    auto blockInfoBytes = static_cast<int64_t>(numBlocks) * sizeof(BlockInfo);

    int64_t totalPinnedBytes;
    int64_t totalAllocBytes;
    uint8_t* pinnedBase;
    uint8_t* deviceBase;
    {
      Timer t("alloc outputs", FLAGS_print_timing);
      sv.paramOffsets.resize(sv.kernels.size());
      int64_t paramCursor = blockInfoBytes;
      for (size_t i = 0; i < sv.kernels.size(); ++i) {
        sv.paramOffsets[i] = paramCursor;
        paramCursor += launchParamSize(sv.kernels[i]);
      }
      totalPinnedBytes = paramCursor;

      // Allocate extra space for per-block DebugInfo at the end of the buffer.
      auto debugInfoBytes = static_cast<int64_t>(numBlocks) * sizeof(DebugInfo);
      totalAllocBytes = totalPinnedBytes + debugInfoBytes;

      // Get or allocate pinned and device buffers.
      auto& pinnedBuffer = getOrAllocateBuffer(
          state.pinnedBuffers,
          sequenceNumber,
          stepIdx,
          totalAllocBytes,
          state.pinnedArena);
      auto& deviceBuffer = getOrAllocateBuffer(
          state.deviceBuffers,
          sequenceNumber,
          stepIdx,
          totalAllocBytes,
          state.deviceArena);
      pinnedBase = pinnedBuffer->as<uint8_t>();
      deviceBase = deviceBuffer->as<uint8_t>();
      if (FLAGS_debug_single_ops) {
        // init with bad data to detect missing params.
        memset(pinnedBase, 0xaa, totalAllocBytes);
      }
    }

    auto* deviceDebugBase =
        reinterpret_cast<DebugInfo*>(deviceBase + totalPinnedBytes);
    int32_t returnBegin = -1;
    int32_t returnEnd = -1;
    {
      Timer t("fill params", FLAGS_print_timing);
      // Copy BlockInfos to beginning of pinned buffer.
      if (!sv.blocks.empty()) {
        memcpy(pinnedBase, sv.blocks.data(), blockInfoBytes);
      }

      // Fill params for each kernel launch.
      for (size_t i = 0; i < sv.kernels.size(); ++i) {
        int32_t rb = -1;
        int32_t re = -1;
        fillLaunchParams(
            sv.kernels[i], frame, pinnedBase + sv.paramOffsets[i], rb, re);
        if (rb >= 0) {
          if (returnBegin == -1) {
            returnBegin = sv.paramOffsets[i] + rb;
          }
          returnEnd = sv.paramOffsets[i] + re;
        }
      }

      // Patch BlockInfo params and debugInfo pointers to device-side addresses.
      auto* pinnedBlocks = reinterpret_cast<BlockInfo*>(pinnedBase);
      for (size_t b = 0; b < numBlocks; ++b) {
        auto idx = sv.launchIndices[b];
        pinnedBlocks[b].params = deviceBase + sv.paramOffsets[idx];
        pinnedBlocks[b].debugInfo = deviceDebugBase + b;
      }

      // Patch TensorList pointers to device-side Tensor addresses.
      for (size_t i = 0; i < sv.kernels.size(); ++i) {
        patchTensorListPointers(
            sv.kernels[i],
            pinnedBase + sv.paramOffsets[i],
            deviceBase + sv.paramOffsets[i]);
      }
    }

    if (WaveConfig::get().trace) {
      traceStep(stepIdx, sv, gridChoices);
    }

    // Record debug info locations for later D2H transfer.
    state.launchDebugInfos->push_back(
        {reinterpret_cast<DebugInfo*>(pinnedBase + totalPinnedBytes),
         deviceDebugBase,
         static_cast<int32_t>(numBlocks)});

    {
      Timer t("launch1", FLAGS_print_timing);
      launch(
          static_cast<int32_t>(numBlocks),
          blockSize,
          pinnedBase,
          deviceBase,
          totalPinnedBytes,
          returnBegin,
          returnEnd,
          deviceDebugBase,
          state.stream.get(),
          sv,
          stepIdx);
    }

    if (returnBegin >= 0) {
      processReturnData(sv, frame, pinnedBase);
    }
  }
}

void CompositeInvocation::launch(
    int32_t numBlocks,
    int32_t blockSize,
    uint8_t* pinnedBase,
    uint8_t* deviceBase,
    int64_t totalPinnedBytes,
    int32_t returnBegin,
    int32_t returnEnd,
    DebugInfo* deviceDebugBase,
    facebook::velox::wave::Stream* stream,
    const StepVectors& sv,
    int32_t stepIdx) {
  TorchWaveParams params;
  params.info = reinterpret_cast<BlockInfo*>(deviceBase);
  params.debugInfo = deviceDebugBase;
  void* args[] = {&params};

  auto* pinnedBlocks = reinterpret_cast<BlockInfo*>(pinnedBase);

  if (FLAGS_debug_single_ops) {
    // Save original opcodes.
    std::vector<int32_t> originalOps(numBlocks);
    for (int32_t b = 0; b < numBlocks; ++b) {
      originalOps[b] = pinnedBlocks[b].op;
    }

    for (int32_t active = 0; active < numBlocks; ++active) {
      // Set all blocks to no-op except the active one.
      for (int32_t b = 0; b < numBlocks; ++b) {
        pinnedBlocks[b].op = (b == active) ? originalOps[b] : -1;
      }
      try {
        stream->hostToDeviceAsync(deviceBase, pinnedBase, totalPinnedBytes);
        kernel->launch(numBlocks, blockSize, 0, stream, args);
        if (returnBegin >= 0) {
          stream->deviceToHostAsync(
              pinnedBase + returnBegin,
              deviceBase + returnBegin,
              returnEnd - returnBegin);
        }
        stream->wait();
      } catch (const std::exception& e) {
        auto opCode = originalOps[active];
        auto launchIdx = sv.launchIndices[active];
        std::string opText;
        std::string paramText;
        if (launchIdx < static_cast<int32_t>(sv.kernels.size()) &&
            sv.kernels[launchIdx].launch && sv.kernels[launchIdx].launch->op) {
          auto* kernelOp = sv.kernels[launchIdx].launch->op;
          opText = kernelOp->toString();
          auto* opParams = pinnedBase + sv.paramOffsets[launchIdx];
          paramText = dumpOpParams(*kernelOp, opParams);
        }
        LOG(ERROR) << "debug_single_ops: block " << active << " opCode "
                   << opCode << " blockInOp " << pinnedBlocks[active].blockInOp
                   << " stepIdx " << stepIdx << " op: " << opText
                   << "\nparams:\n"
                   << paramText << "error: " << e.what();
        throw;
      }
    }

    // Restore original opcodes.
    for (int32_t b = 0; b < numBlocks; ++b) {
      pinnedBlocks[b].op = originalOps[b];
    }
  } else {
    stream->hostToDeviceAsync(deviceBase, pinnedBase, totalPinnedBytes);
    kernel->launch(numBlocks, blockSize, 0, stream, args);
    if (returnBegin >= 0) {
      stream->deviceToHostAsync(
          pinnedBase + returnBegin,
          deviceBase + returnBegin,
          returnEnd - returnBegin);
      stream->wait();
    }
  }
}

void CompositeInvocation::traceStep(
    int32_t stepIdx,
    const StepVectors& sv,
    const std::vector<GridChoice>& gridChoices) {
  std::cout << "== " << sequenceNumber << " step " << stepIdx << std::endl;

  // Build map from OpInvocation* to its index in ops.
  std::unordered_map<const OpInvocation*, int32_t> opInvocationIndex;
  for (size_t i = 0; i < ops.size(); ++i) {
    opInvocationIndex[&ops[i]] = static_cast<int32_t>(i);
  }

  // Build map from ProjectOperation* to its distinct ordinal.
  std::unordered_map<ProjectOperation*, int32_t> projectOpIndex;
  for (const auto& op : ops) {
    auto* projectOp = op.projectOp();
    if (projectOpIndex.find(projectOp) == projectOpIndex.end()) {
      projectOpIndex[projectOp] = static_cast<int32_t>(projectOpIndex.size());
    }
  }

  for (const auto& launch : sv.standalones) {
    auto opIdx = opInvocationIndex[launch.invocation];
    std::cout << sequenceNumber << "." << opIdx << " standalone" << std::endl;
  }

  for (size_t i = 0; i < sv.kernels.size(); ++i) {
    const auto& launch = sv.kernels[i];
    auto opIdx = opInvocationIndex[launch.invocation];
    auto distinctOpIdx = projectOpIndex[launch.invocation->projectOp()];
    auto* projectOp = launch.invocation->projectOp();
    const char* gridLabel = "M";
    if (gridChoices[opIdx].singleBlock) {
      gridLabel = "S";
    } else if (gridChoices[opIdx].grid == &projectOp->cgGrid()) {
      gridLabel = "CG";
    }
    std::cout << sequenceNumber << "." << opIdx << " " << gridLabel << " op "
              << distinctOpIdx << " " << launch.numElements
              << " blocks=" << sv.numBlocksPerLaunch[i]
              << " opcode=" << launch.launch->op->opCode() << " "
              << (launch.standalone ? "standalone"
                                    : launch.launch->op->toString(kExprs))
              << std::endl;
  }
}

std::string CompositeInvocation::toString(Listing mode, int32_t ordinal) const {
  std::stringstream ss;

  // Collect distinct ProjectOperations.
  std::vector<ProjectOperation*> projectOps;
  std::unordered_map<ProjectOperation*, int32_t> projectOpIndex;
  for (const auto& op : ops) {
    auto* po = op.projectOp();
    if (projectOpIndex.find(po) == projectOpIndex.end()) {
      projectOpIndex[po] = static_cast<int32_t>(projectOps.size());
      projectOps.push_back(po);
    }
  }

  // Print OpInvocations with their ProjectOperation ordinal and bindings.
  for (size_t i = 0; i < ops.size(); ++i) {
    auto it = projectOpIndex.find(ops[i].projectOp());
    ss << ordinal << "." << i << ": ProjectOp " << it->second;
    const auto& bindings = ops[i].bindings();
    for (const auto& [formalId, actualId] : bindings) {
      ss << " %" << formalId << " = %" << actualId;
    }
    ss << "\n";
  }

  // Print distinct ops with their grids.
  ss << "\nDistinct Ops\n";
  for (size_t i = 0; i < projectOps.size(); ++i) {
    ss << "Op " << ordinal << "." << i << "\n";
    ss << projectOps[i]->toString(mode);
  }

  return ss.str();
}

// --- CompiledNode ---

void CompiledNode::execute(ExecutionState& state) {
  kernels_->execute(state);
}

std::string CompiledNode::toString(Listing mode, int32_t ordinal) const {
  std::stringstream ss;
  if (mode == kExprs) {
    // Collect distinct ProjectOperations and count invocations.
    std::vector<ProjectOperation*> projectOps;
    std::unordered_map<ProjectOperation*, int32_t> invocationCount;
    for (const auto& op : kernels_->ops) {
      auto* po = op.projectOp();
      if (invocationCount[po]++ == 0) {
        projectOps.push_back(po);
      }
    }
    for (auto* po : projectOps) {
      ss << invocationCount[po] << "x " << po->subgraph().toString(mode)
         << "\n";
    }
  } else {
    ss << kernels_->toString(mode, ordinal);
  }
  return ss.str();
}

} // namespace torch::wave
