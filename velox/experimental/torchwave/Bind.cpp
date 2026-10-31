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

// The bind pass of the bound execution path: turns every step of a node into
// a StepProgram before the first execution, reading only the compiled grids
// and the graph.

#include <algorithm>
#include <memory>
#include <type_traits>
#include <unordered_set>

#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>

#include "velox/experimental/torchwave/AllocGroup.h"
#include "velox/experimental/torchwave/CompiledOp.h"
#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/StepProgram.h"
#include "velox/experimental/torchwave/StepSupport.h"
#include "velox/experimental/torchwave/WaveConfig.h"

namespace torch::wave {

namespace {

int32_t actualIdOf(ValueCP value, const FormalToActual& bindings) {
  auto it = bindings.find(value->id());
  return it != bindings.end() ? it->second : value->id();
}

// A Size calling a built-in over the values and constant shapes of 'expr'.
int32_t
addSizeExprSize(StepProgram& program, const SizeExpr& expr, ReserveFn fn) {
  BoundSize size;
  size.fn = fn;
  size.args.tensors.emplace_back(expr.values.begin(), expr.values.end());
  for (const auto& constShape : expr.constShapes) {
    auto& list = size.args.intLists.emplace_back();
    for (auto dim : constShape) {
      list.push_back({.kind = ScalarRef::kConstant, .constant = dim});
    }
  }
  program.sizes.push_back(std::move(size));
  return static_cast<int32_t>(program.sizes.size() - 1);
}

int32_t addCountSize(StepProgram& program, const SizeExpr& expr) {
  const bool broadcast = expr.broadcast || !expr.constShapes.empty();
  return addSizeExprSize(
      program,
      expr,
      broadcast                           ? broadcastCountReserve
          : expr.op == SizeShortcut::kSum ? sumCountReserve
                                          : maxCountReserve);
}

int32_t addDimsSize(StepProgram& program, const SizeExpr& expr) {
  const bool broadcast = expr.broadcast || !expr.constShapes.empty();
  return addSizeExprSize(
      program, expr, broadcast ? broadcastShapeReserve : maxShapeReserve);
}

// Output tensors of output 'outputIdx' of 'data': its list's size for a
// TensorList, else 1.
int32_t numTensorsOfOutput(
    const LaunchData& data,
    size_t outputIdx,
    const IdToValueMap& idToValue) {
  if (outputIdx >= data.actualOutputTypes.size() ||
      data.actualOutputTypes[outputIdx] != nativert::Type::Kind::TensorList) {
    return 1;
  }
  return static_cast<int32_t>(
      idToValue.at(data.actualOutputs[outputIdx])->getListElements().size());
}

// A Size running the ReserveFn of output 'outputIdx'.
int32_t addReserveSize(
    StepProgram& program,
    const LaunchData& data,
    size_t outputIdx,
    int32_t blockSize,
    const IdToValueMap& idToValue) {
  const auto& desc = data.actualOutputDescs[outputIdx];
  BoundSize size;
  size.data = &data;
  size.outputIdx = static_cast<int32_t>(outputIdx);
  TORCH_CHECK(
      desc.reserveFn != nullptr &&
          bindReserve(
              desc,
              data.invocation->bindings(),
              data.invocation->nodeMap(),
              size.args),
      "bind: the reserve of ",
      desc.reserveAttrNode != nullptr ? desc.reserveAttrNode->target()
                                      : std::string_view("an output"),
      " output ",
      outputIdx,
      " cannot bind its operands");
  size.fn = desc.reserveFn;
  size.blockSize = blockSize;
  size.numShapes = numTensorsOfOutput(data, outputIdx, idToValue);
  program.sizes.push_back(std::move(size));
  return static_cast<int32_t>(program.sizes.size() - 1);
}

// A Size laying out the concat result of output 'outputIdx' over the table.
int32_t addConcatSize(
    StepProgram& program,
    const LaunchData& data,
    size_t outputIdx,
    int32_t blockSize,
    const IdToValueMap& idToValue) {
  const auto& desc = data.actualOutputDescs[outputIdx];
  const auto& layout = *desc.concatLayout;
  const auto& bindings = data.invocation->bindings();
  const auto [spec, dtype] = layout.resolve(data.invocation->nodeMap());
  BoundConcat concat{
      .spec = spec, .dtype = dtype, .resultId = data.actualOutputs[outputIdx]};
  const auto sizeExprs = translateConcatSizeExprs(layout, bindings);
  for (size_t i = 0; i < layout.inputs.size(); ++i) {
    const auto& info = layout.inputs[i];
    auto it = bindings.find(info.valueId);
    const auto id = it != bindings.end() ? it->second : info.valueId;
    int32_t size = -1;
    if (info.hasReserve) {
      // The operand is an output of this launch the concat sizes for it.
      for (size_t j = 0; j < data.actualOutputs.size(); ++j) {
        if (data.actualOutputs[j] == id &&
            data.actualOutputDescs[j].reserveFn != nullptr) {
          size = addReserveSize(program, data, j, blockSize, idToValue);
          break;
        }
      }
      TORCH_CHECK(
          size >= 0,
          "bind: concat operand %",
          id,
          " is sized by a reserve outside the concat's launch");
    } else if (info.sizeExpr.op != SizeShortcut::kNone) {
      TORCH_CHECK(
          info.sizeExpr.op == SizeShortcut::kMax && sizeExprs[i].args.empty(),
          "bind: concat operand %",
          id,
          " has a size expression that is not a flat maximum");
      size = addDimsSize(program, sizeExprs[i]);
    }
    concat.operandIds.push_back(id);
    concat.operandSizes.push_back(size);
    concat.inPlace.push_back(!info.isSubgraphInput && !info.isView);
    concat.shapeOnDevice.push_back(info.hasShapeOnDevice);
  }
  program.concats.push_back(std::move(concat));
  BoundSize size;
  size.concat = static_cast<int32_t>(program.concats.size() - 1);
  program.sizes.push_back(std::move(size));
  return static_cast<int32_t>(program.sizes.size() - 1);
}

// The Size of output 'outputIdx': its concat layout or its reserve.
int32_t addOutputSize(
    StepProgram& program,
    const LaunchData& data,
    size_t outputIdx,
    int32_t blockSize,
    const IdToValueMap& idToValue) {
  return data.actualOutputDescs[outputIdx].concatLayout != nullptr
      ? addConcatSize(program, data, outputIdx, blockSize, idToValue)
      : addReserveSize(program, data, outputIdx, blockSize, idToValue);
}

bool isTensorOutput(const LaunchData& data, size_t i) {
  return i >= data.actualOutputTypes.size() ||
      data.actualOutputTypes[i] == nativert::Type::Kind::Tensor;
}

// A kView output: a DeriveOp over the table when a metadata shortcut covers the
// view, else the view over the frame.
BoundOutput viewOutput(
    StepProgram& program,
    const OutputDesc& desc,
    nativert::ValueId actualId) {
  BoundOutput output{
      .kind = BoundOutput::kView, .valueId = actualId, .desc = &desc};
  const auto recipe = makeViewRecipe(desc.viewNode);
  if (recipe == nullptr || !derivable(recipe->launch.standaloneShortcut) ||
      recipe->data.args.empty() || recipe->data.args[0] == nullptr) {
    return output;
  }
  const auto& data = recipe->data;
  DeriveOp op{
      .kind = recipe->launch.standaloneShortcut,
      .intList = data.intList,
      .output = actualId};
  for (size_t i = 0; i < data.args.size(); ++i) {
    op.operands.push_back(data.args[i] ? data.args[i]->id() : -1);
    op.constants.push_back(i < data.intArgs.size() ? data.intArgs[i] : 0);
  }
  output.derive = static_cast<int32_t>(program.viewDerives.size());
  program.viewDerives.push_back(std::move(op));
  return output;
}

// The outputs of a launch sized by its largest input, used when that input
// exists.
std::vector<BoundOutput> largestInputOutputs(
    StepProgram& program,
    const LaunchData& data,
    int32_t largestSize) {
  const auto& descs = data.actualOutputDescs;
  folly::F14FastSet<int32_t> aliasSelfTargets;
  for (const auto& desc : descs) {
    if (desc.aliasSelfId) {
      aliasSelfTargets.insert(*desc.aliasSelfId);
    }
  }
  std::vector<BoundOutput> outputs;
  for (size_t i = 0; i < descs.size(); ++i) {
    if (!isTensorOutput(data, i)) {
      continue;
    }
    const auto& desc = descs[i];
    const auto actualId = data.actualOutputs[i];
    if (desc.viewNode) {
      outputs.push_back(viewOutput(program, desc, actualId));
      continue;
    }
    if (desc.delegated) {
      continue;
    }
    if (aliasSelfTargets.count(actualId) &&
        desc.sizeExpr.op != SizeShortcut::kNone) {
      outputs.push_back(
          {.kind = BoundOutput::kTensor,
           .valueId = actualId,
           .size = addDimsSize(program, desc.sizeExpr)});
      continue;
    }
    outputs.push_back(
        {.kind = BoundOutput::kTensor,
         .valueId = actualId,
         .size = largestSize,
         .aliasOf = desc.aliasSelfId ? *desc.aliasSelfId : -1});
  }
  return outputs;
}

// The outputs of a launch that is not sized by its largest input. 'countSize'
// is the Size already made for output 0's reserve, if the launch's element
// count comes from it.
std::vector<BoundOutput> describedOutputs(
    StepProgram& program,
    const LaunchData& data,
    int32_t countSize,
    int32_t blockSize,
    const IdToValueMap& idToValue) {
  const auto& descs = data.actualOutputDescs;
  const auto& bindings = data.invocation->bindings();
  std::vector<BoundOutput> outputs;
  auto reserveSize = [&](size_t i) {
    return i == 0 && countSize >= 0
        ? countSize
        : addOutputSize(program, data, i, blockSize, idToValue);
  };
  for (size_t i = 0; i < descs.size(); ++i) {
    const bool isList = i < data.actualOutputTypes.size() &&
        data.actualOutputTypes[i] == nativert::Type::Kind::TensorList;
    if (!isTensorOutput(data, i) && !isList) {
      continue;
    }
    const auto& desc = descs[i];
    const auto actualId = data.actualOutputs[i];
    if (desc.viewNode) {
      outputs.push_back(viewOutput(program, desc, actualId));
      continue;
    }
    if (desc.delegated) {
      continue;
    }
    const int32_t aliasOf = desc.aliasSelfId ? *desc.aliasSelfId : -1;
    if (isList) {
      if (desc.reserveFn == nullptr) {
        continue;
      }
      const auto size = reserveSize(i);
      const auto elements = idToValue.at(actualId)->getListElements();
      if (aliasOf >= 0) {
        outputs.push_back(
            {.kind = BoundOutput::kAlias,
             .valueId = actualId,
             .aliasOf = aliasOf});
      }
      for (size_t j = 0; j < elements.size(); ++j) {
        outputs.push_back(
            {.kind = BoundOutput::kListElement,
             .valueId = actualIdOf(elements[j], bindings),
             .size = size,
             .element = static_cast<int32_t>(j)});
      }
      continue;
    }
    BoundOutput output{.valueId = actualId, .aliasOf = aliasOf};
    if (desc.concatLayout != nullptr) {
      output.kind = BoundOutput::kConcat;
      output.desc = &desc;
      output.size = reserveSize(i);
    } else if (desc.reserveFn != nullptr) {
      output.size = reserveSize(i);
    } else if (desc.sizeExpr.op != SizeShortcut::kNone) {
      output.size = addDimsSize(program, desc.sizeExpr);
    } else if (aliasOf >= 0) {
      output.kind = BoundOutput::kAlias;
    } else {
      continue;
    }
    if (desc.shapeOnly && output.kind == BoundOutput::kTensor) {
      output.kind = BoundOutput::kShapeOnly;
    }
    outputs.push_back(output);
  }
  return outputs;
}

// Records where the launch's tensors, scalars and returns live in its
// parameter block, and adds the
// fill ops that rewrite them every execution. 'base' is the launch's offset in
// the step's pinned region.
void bindLaunchFill(
    StepProgram& program,
    LaunchData& data,
    int32_t kernel,
    int32_t base) {
  auto* kernelOp = data.launch->op;
  const auto& orderedInputs = kernelOp->orderedInputs();
  const auto numInputs = kernelOp->numInputs();
  const auto& bindings = data.invocation->bindings();
  data.tensorsInFrame.clear();
  data.tensorOffsets.clear();
  data.scalarsInFrame.clear();
  data.scalarOffsets.clear();
  data.scalarOutputOffsets.clear();
  data.shapeOnlyTensorIndices.clear();
  data.tensorLists.clear();
  data.returnOffsets.clear();
  std::unordered_set<int32_t> filledOffsets;
  size_t returnCounter = 0;

  auto addTensor = [&](int32_t offset, int32_t id, FillOp::Kind kind) {
    program.fills.push_back(
        {.kind = kind,
         .offset = base + offset,
         .valueId = id,
         .kernel = kernel});
    if (kind == FillOp::kShapeOnly) {
      data.shapeOnlyTensorIndices.insert(data.tensorsInFrame.size());
    }
    data.tensorsInFrame.push_back(id);
    data.tensorOffsets.push_back(offset);
  };
  auto addList = [&](ValueCP list) {
    TensorListParam tlp;
    tlp.listOffset = kernelOp->paramOffset(list);
    for (auto* element : list->getListElements()) {
      const auto elementOffset = kernelOp->paramOffset(element);
      const auto id = actualIdOf(element, bindings);
      tlp.elementOffsets.push_back(elementOffset);
      tlp.elementIds.push_back(id);
      if (filledOffsets.insert(elementOffset).second) {
        addTensor(elementOffset, id, FillOp::kTensor);
      }
    }
    data.tensorLists.push_back(std::move(tlp));
  };
  auto trackReturn = [&](int32_t id, int32_t offset, int32_t size) {
    if (returnCounter < data.returnValues.size() &&
        id == data.returnValues[returnCounter]) {
      data.returnOffsets.push_back(offset);
      const auto begin = base + offset;
      program.returnBegin = program.returnBegin < 0
          ? begin
          : std::min(program.returnBegin, begin);
      program.returnEnd = std::max(program.returnEnd, begin + size);
      ++returnCounter;
    }
  };

  for (int32_t i = 0; i < numInputs; ++i) {
    auto* formal = orderedInputs[i];
    if (formal->type().kind() == nativert::Type::Kind::TensorList) {
      addList(formal);
      continue;
    }
    const auto offset = kernelOp->paramOffset(formal);
    const auto id = data.actualInputs.at(i);
    if (formal->type().kind() == nativert::Type::Kind::Tensor) {
      addTensor(offset, id, FillOp::kTensor);
    } else {
      program.fills.push_back(
          {.kind = FillOp::kScalar,
           .offset = base + offset,
           .valueId = id,
           .kernel = kernel});
      data.scalarsInFrame.push_back(id);
      data.scalarOffsets.push_back(offset);
    }
  }
  for (size_t i = 0; i < data.actualOutputs.size(); ++i) {
    auto* formal = orderedInputs[numInputs + i];
    const auto id = data.actualOutputs[i];
    if (formal->type().kind() == nativert::Type::Kind::TensorList) {
      addList(formal);
      if (returnCounter < data.returnValues.size() &&
          id == data.returnValues[returnCounter]) {
        data.returnOffsets.push_back(kernelOp->paramOffset(formal));
        for (auto elementOffset : data.tensorLists.back().elementOffsets) {
          const auto begin = base + elementOffset;
          program.returnBegin = program.returnBegin < 0
              ? begin
              : std::min(program.returnBegin, begin);
          program.returnEnd = std::max(
              program.returnEnd, begin + static_cast<int32_t>(sizeof(Tensor)));
        }
        ++returnCounter;
      }
      continue;
    }
    const auto offset = kernelOp->paramOffset(formal);
    if (isTensorOutput(data, i)) {
      const bool shapeOnly = i < data.actualOutputDescs.size() &&
          data.actualOutputDescs[i].shapeOnly;
      addTensor(offset, id, shapeOnly ? FillOp::kShapeOnly : FillOp::kTensor);
      trackReturn(id, offset, sizeof(Tensor));
    } else {
      program.fills.push_back(
          {.kind = FillOp::kZero64, .offset = base + offset, .kernel = kernel});
      data.scalarOutputOffsets.push_back(offset);
      trackReturn(id, offset, 8);
    }
  }
  for (auto offset : kernelOp->altTensorOffsets()) {
    program.fills.push_back(
        {.kind = FillOp::kUninited, .offset = base + offset, .kernel = kernel});
  }
  for (auto offset : kernelOp->barrierCounters()) {
    program.fills.push_back(
        {.kind = FillOp::kZero32, .offset = base + offset, .kernel = kernel});
  }
}

// Writes what does not change between executions into the step's pinned
// block: constants, TensorList headers and their element pointer arrays.
void writeStaticParams(
    const StepVectors& sv,
    uint8_t* pinnedBase,
    uint8_t* deviceBase) {
  for (size_t k = 0; k < sv.kernels.size(); ++k) {
    const auto& data = sv.kernels[k];
    auto* params = pinnedBase + sv.paramOffsets[k];
    auto* devParams = deviceBase + sv.paramOffsets[k];
    auto* kernelOp = data.launch->op;
    auto constantOffset = kernelOp->constantAreaOffset();
    const auto& opConstants = data.invocation->constants();
    for (auto idx : data.launch->constantIndices) {
      fillScalarParam(*opConstants[idx], params + constantOffset, -1);
      constantOffset += 8;
    }
    for (const auto& tlp : data.tensorLists) {
      auto* list = reinterpret_cast<TensorList*>(params + tlp.listOffset);
      list->size = static_cast<int64_t>(tlp.elementOffsets.size());
      list->tensors = reinterpret_cast<Tensor**>(
          devParams + tlp.listOffset + sizeof(TensorList));
      auto* pointers = reinterpret_cast<Tensor**>(
          params + tlp.listOffset + sizeof(TensorList));
      for (size_t j = 0; j < tlp.elementOffsets.size(); ++j) {
        pointers[j] =
            reinterpret_cast<Tensor*>(devParams + tlp.elementOffsets[j]);
      }
    }
  }
}

// The operand lists checkUnreadyOperands reads each execution.
void bindReadyCheck(BoundLaunch& launch, const LaunchData& data) {
  const auto& op = *data.launch->op;
  const auto& opInputs = op.orderedInputs();
  const auto& opNodes = op.allNodes();
  const auto numOpInputs = static_cast<size_t>(op.numInputs());
  for (size_t k = 0; k < numOpInputs; ++k) {
    launch.allInputs.push_back(data.actualInputs[k]);
    const auto* formal = opInputs[k];
    if (formal->type().kind() != nativert::Type::Kind::Tensor) {
      continue;
    }
    const auto* producer = formal->producer();
    if (producer == nullptr || opNodes.count(producer) == 0) {
      launch.externalInputs.push_back(data.actualInputs[k]);
    }
  }
  for (size_t i = 0; i < data.actualOutputs.size(); ++i) {
    if (i < data.actualOutputTypes.size() &&
        data.actualOutputTypes[i] == nativert::Type::Kind::Tensor) {
      launch.tensorOutputs.push_back(data.actualOutputs[i]);
    }
  }
  for (const auto* tensorMeta : op.inputTypes()) {
    if (tensorMeta != nullptr && !tensorMeta->hasSymbolicShape()) {
      int64_t numElements = 1;
      for (auto extent : tensorMeta->sizes()) {
        numElements *= extent;
      }
      launch.staticNumElements =
          std::max(launch.staticNumElements, numElements);
    }
  }
}

bool readsAny(
    const folly::F14FastSet<nativert::ValueId>& reads,
    const folly::F14FastSet<int32_t>& returned) {
  for (auto id : reads) {
    if (returned.count(id) != 0) {
      return true;
    }
  }
  return false;
}

} // namespace

void CompositeInvocation::bindLaunches(
    ExecutionState& state,
    folly::F14FastSet<int32_t>& returned) {
  const auto& idToValue = state.waveGraph->idToValue();
  std::vector<GridChoice> choices;
  int32_t numSteps = 0;
  for (auto& op : ops_) {
    auto& grid = allocGroupGrid(op);
    choices.push_back({0, &grid == &op.projectOp()->singleBlockGrid(), &grid});
    numSteps = std::max(numSteps, static_cast<int32_t>(grid.size()));
  }
  // One step past the grids: it launches nothing, but the allocation groups
  // planned at that step are carved there, as the legacy loop carves the groups
  // of the first empty step before it stops.
  ++numSteps;
  for (int32_t stepIdx = 0; stepIdx < numSteps; ++stepIdx) {
    auto& sv = getStepVectors(state.stepVectors, sequenceNumber_, stepIdx);
    sv.kernels.clear();
    sv.standalones.clear();
    sv.shortcutStandalones.clear();
    sv.paramOffsets.clear();
    sv.hasGpuStandalones = false;
    sv.isCgGrid = false;
    layoutParamSlots(stepIdx, sv);
    sv.blockSize = WaveConfig::get().blockSize;
    auto program = std::make_shared<StepProgram>();
    for (size_t i = 0; i < ops_.size(); ++i) {
      const auto& grid = *choices[i].grid;
      if (stepIdx >= static_cast<int32_t>(grid.size())) {
        continue;
      }
      int32_t opKernelSlot = 0;
      for (const auto& launch : grid[stepIdx]) {
        if (launch.op != nullptr) {
          const auto slot = sv.opSlotBegin.at(i) + opKernelSlot++;
          TORCH_CHECK(
              slot < sv.opSlotBegin.at(i + 1),
              "Kernel launch slot beyond the reservation for op ",
              i,
              " at step ",
              stepIdx);
          sv.paramOffsets.push_back(sv.slotOffsets.at(slot));
          sv.kernels.emplace_back(launch, ops_[i], idToValue);
          setViewReadIds(sv.kernels.back());
          program->launches.push_back(
              {.kernel = static_cast<int32_t>(sv.kernels.size() - 1)});
          program->launchOp.push_back(static_cast<int32_t>(i));
          returned.insert(
              sv.kernels.back().returnValues.begin(),
              sv.kernels.back().returnValues.end());
          sv.blockSize =
              std::max(sv.blockSize, launch.op->blockSizePreference());
          if (!launch.op->barrierCounters().empty()) {
            sv.isCgGrid = true;
          }
        } else if (launch.standaloneShortcut != StandaloneShortcut::kNone) {
          sv.shortcutStandalones.emplace_back(launch, ops_[i], idToValue);
          program->standaloneOps.push_back(static_cast<int32_t>(i));
        } else {
          sv.standalones.emplace_back(launch, ops_[i], idToValue);
          setViewReadIds(sv.standalones.back());
          program->standaloneOps.push_back(static_cast<int32_t>(i));
          if (!launch.metadataOnly) {
            sv.hasGpuStandalones = true;
          }
        }
      }
    }
    sv.entry = kernel_ && !sv.kernels.empty()
        ? kernel_->entryFor(sv.blockSize, stepIdx)
        : 0;
    sv.program = std::move(program);
  }
  bindLastUseSteps(choices, numSteps);
  auto& sv0 = getStepVectors(state.stepVectors, sequenceNumber_, 0);
  sv0.gridChoices = std::move(choices);
  sv0.boundSteps = numSteps;
}

void CompositeInvocation::bindPrograms(
    ExecutionState& state,
    const folly::F14FastSet<int32_t>& returned) {
  const auto& idToValue = state.waveGraph->idToValue();
  const int32_t numSteps =
      getStepVectors(state.stepVectors, sequenceNumber_, 0).boundSteps;
  for (int32_t stepIdx = 0; stepIdx < numSteps; ++stepIdx) {
    auto& sv = getStepVectors(state.stepVectors, sequenceNumber_, stepIdx);
    auto& program = *sv.program;
    std::vector<const AllocGroup*> stepGroups;
    if (const auto* plan = allocGroupPlan_.get(); plan != nullptr &&
        stepIdx < static_cast<int32_t>(plan->groupsByStep.size())) {
      for (auto g : plan->groupsByStep[stepIdx]) {
        stepGroups.push_back(&plan->groups[g]);
      }
    }
    program.collector =
        std::make_shared<AllocGroupCollector>(stepGroups, /*install=*/false);
    // Ops whose reads a transfer can bring back go after the step's wait.
    std::vector<bool> dependentOp(ops_.size(), false);
    for (size_t i = 0; i < ops_.size() && i < sv.opReadIds.size(); ++i) {
      dependentOp[i] = readsAny(sv.opReadIds[i], returned);
    }
    for (size_t i = 0; i < ops_.size(); ++i) {
      if (dependentOp[i]) {
        program.waitReads.insert(
            sv.opReadIds[i].begin(), sv.opReadIds[i].end());
      }
    }
    // A launch whose sizing reads the elements of a value waits for the host
    // copy of it.
    for (size_t l = 0; l < program.launches.size(); ++l) {
      const auto& data = sv.kernels[program.launches[l].kernel];
      for (const auto& desc : data.actualOutputDescs) {
        for (auto id : hostTensorIds(desc, data.invocation->bindings())) {
          dependentOp[program.launchOp[l]] = true;
          program.waitReads.insert(id);
        }
      }
    }
    std::vector<BoundLaunch> independent;
    std::vector<BoundLaunch> dependent;
    for (size_t l = 0; l < program.launches.size(); ++l) {
      auto launch = std::move(program.launches[l]);
      auto& data = sv.kernels[launch.kernel];
      const bool byLargestInput = !data.launch->op->outputDescs().empty() &&
          data.launch->op->outputDescs()[0].byLargestInput;
      if (data.sizeExpr.op == SizeShortcut::kNone) {
        TORCH_CHECK(
            data.actualOutputDescs.size() == 1 &&
                data.actualOutputDescs[0].sizedByReserve(),
            "A launch with no size expression needs exactly one output with a reserve");
        launch.countSize =
            addOutputSize(program, data, 0, sv.blockSize, idToValue);
        launch.countFromReserve = true;
      } else {
        launch.countSize = addCountSize(program, data.sizeExpr);
        if (byLargestInput && data.sizeExpr.op == SizeShortcut::kMax &&
            !data.sizeExpr.broadcast && data.sizeExpr.constShapes.empty()) {
          launch.largestShapeSize =
              addSizeExprSize(program, data.sizeExpr, largestInputReserve);
          launch.largestOutputs =
              largestInputOutputs(program, data, launch.largestShapeSize);
        }
      }
      launch.outputs = describedOutputs(
          program,
          data,
          launch.countFromReserve ? launch.countSize : -1,
          sv.blockSize,
          idToValue);
      bindReadyCheck(launch, data);
      bindLaunchFill(
          program,
          data,
          launch.kernel,
          static_cast<int32_t>(sv.paramOffsets[launch.kernel]));
      if (dependentOp[program.launchOp[l]]) {
        dependent.push_back(std::move(launch));
      } else {
        independent.push_back(std::move(launch));
      }
    }
    program.numIndependent = static_cast<int32_t>(independent.size());
    program.launches = std::move(independent);
    for (auto& launch : dependent) {
      program.launches.push_back(std::move(launch));
    }

    sv.blockInfoOffset = roundUp(sv.paramRegionBytes, int64_t{16});
    program.totalPinnedBytes = sv.blockInfoOffset +
        static_cast<int64_t>(sv.blockCapacity) *
            static_cast<int64_t>(sizeof(BlockInfo));
    const int64_t totalAllocBytes = program.totalPinnedBytes +
        static_cast<int64_t>(sv.blockCapacity) *
            static_cast<int64_t>(sizeof(DebugInfo));
    auto* pinnedBase = getOrAllocateBuffer(
                           state.pinnedBuffers,
                           sequenceNumber_,
                           stepIdx,
                           totalAllocBytes,
                           state.pinnedArena,
                           WaveConfig::get().debugSingleOps
                               ? std::function<void(void*, int64_t)>(
                                     [](void* ptr, int64_t bytes) {
                                       memset(ptr, 0xaa, bytes);
                                     })
                               : nullptr)
                           ->as<uint8_t>();
    auto* deviceBase = getOrAllocateBuffer(
                           state.deviceBuffers,
                           sequenceNumber_,
                           stepIdx,
                           totalAllocBytes,
                           state.deviceArena)
                           ->as<uint8_t>();
    writeStaticParams(sv, pinnedBase, deviceBase);
  }
}

void bindTableOnly(ExecutionState& state, WaveGraph& waveGraph) {
  const auto numValues = static_cast<size_t>(waveGraph.graph()->numValues());
  std::vector<uint8_t> candidate(numValues, 0);
  std::vector<uint8_t> hostRead(numValues, 0);
  auto markHostRead = [&](nativert::ValueId id) {
    if (id >= 0 && static_cast<size_t>(id) < numValues) {
      hostRead[id] = 1;
    }
  };
  auto markCandidate = [&](nativert::ValueId id) {
    if (id >= 0 && static_cast<size_t>(id) < numValues) {
      candidate[id] = 1;
    }
  };
  for (const auto& node : waveGraph.nodes()) {
    const auto* invocation = node->kernels();
    if (invocation == nullptr) {
      continue;
    }
    if (const auto* plan = invocation->allocGroupPlan()) {
      for (const auto& group : plan->groups) {
        for (auto id : group.actualIds) {
          markCandidate(id);
        }
        if (group.concat != nullptr) {
          markCandidate(group.concat->resultId);
          for (const auto& operand : group.concat->operands) {
            markCandidate(operand.copyDestId);
          }
        }
      }
    }
    for (const auto& launch : invocation->prePassStandalones()) {
      if (launch.standalone != nullptr) {
        for (const auto& input : launch.standalone->inputs()) {
          markHostRead(input.value->id());
        }
      }
    }
  }
  // Concats laid out over the table: the result and the operands' bands are
  // table entries.
  for (const auto& steps : state.stepVectors) {
    for (const auto& sv : steps) {
      if (sv.program == nullptr) {
        continue;
      }
      for (const auto& concat : sv.program->concats) {
        markCandidate(concat.resultId);
        for (auto id : concat.operandIds) {
          markCandidate(id);
        }
      }
    }
  }
  for (const auto& steps : state.stepVectors) {
    for (const auto& sv : steps) {
      for (const auto& data : sv.kernels) {
        // A returned integer can live in the table alone; so can a returned
        // tensor that already does, whose dims the transfer then updates.
        for (size_t r = 0; r < data.returnValues.size(); ++r) {
          if (data.returnTypes[r] == nativert::Type::Kind::Tensor ||
              data.returnTypes[r] == nativert::Type::Kind::TensorList ||
              data.returnTypes[r] == nativert::Type::Kind::SymInt) {
            markCandidate(data.returnValues[r]);
          } else {
            markHostRead(data.returnValues[r]);
          }
        }
        // An in-place output is a view of its self, made over the frame.
        for (size_t i = 0; i < data.actualOutputDescs.size(); ++i) {
          const auto& desc = data.actualOutputDescs[i];
          if (desc.aliasSelfId) {
            markHostRead(*desc.aliasSelfId);
            markHostRead(data.actualOutputs[i]);
          }
        }
      }
      for (const auto& data : sv.standalones) {
        for (auto id : data.actualInputs) {
          markHostRead(id);
        }
        for (auto id : data.actualOutputs) {
          markHostRead(id);
        }
      }
      for (const auto& data : sv.shortcutStandalones) {
        if (derivable(data.launch->standaloneShortcut) ||
            data.launch->standaloneShortcut ==
                StandaloneShortcut::kListUnpack) {
          for (auto id : data.actualOutputs) {
            markCandidate(id);
          }
        }
      }
    }
  }
  // Kernel outputs allocated on their own, and the views of outputs: over the
  // table when derived, else over the frame.
  for (const auto& steps : state.stepVectors) {
    for (const auto& sv : steps) {
      if (sv.program == nullptr) {
        continue;
      }
      for (const auto& launch : sv.program->launches) {
        for (const auto* outputs : {&launch.outputs, &launch.largestOutputs}) {
          for (const auto& output : *outputs) {
            if (output.kind == BoundOutput::kView) {
              if (output.derive >= 0) {
                markCandidate(output.valueId);
              } else {
                for (const auto& input : output.desc->viewNode->inputs()) {
                  markHostRead(input.value->id());
                }
                markHostRead(output.valueId);
              }
            }
            if ((output.kind == BoundOutput::kTensor ||
                 output.kind == BoundOutput::kListElement) &&
                output.aliasOf < 0) {
              markCandidate(output.valueId);
            }
          }
        }
      }
    }
  }
  for (size_t id = 0; id < numValues; ++id) {
    const auto valueId = static_cast<nativert::ValueId>(id);
    if (waveGraph.isGraphOutput(valueId)) {
      hostRead[id] = 1;
    }
  }
  // Which nodes run eagerly, and the shortcut launch of each shortcut node.
  const auto& idToValue = waveGraph.idToValue();
  folly::F14FastSet<NodeCP> eagerNodes;
  folly::F14FastMap<NodeCP, const LaunchData*> shortcutOf;
  for (const auto& steps : state.stepVectors) {
    for (const auto& sv : steps) {
      for (const auto& data : sv.standalones) {
        eagerNodes.insert(data.standalone);
      }
      for (const auto& data : sv.shortcutStandalones) {
        shortcutOf[data.standalone] = &data;
      }
    }
  }
  auto kindOf = [&](NodeCP node) {
    auto it = shortcutOf.find(node);
    return it == shortcutOf.end() ? StandaloneShortcut::kNone
                                  : it->second->launch->standaloneShortcut;
  };
  auto listUsers = [&](nativert::ValueId id) -> std::vector<nativert::Node*> {
    auto it = idToValue.find(id);
    return it == idToValue.end() ? std::vector<nativert::Node*>{}
                                 : it->second->users();
  };
  // An unbind or split whose list only a list unpack reads derives together
  // with the unpack, straight into the unpacked values.
  folly::F14FastMap<NodeCP, NodeCP> unpackOf;
  folly::F14FastSet<NodeCP> absorbedUnpacks;
  for (const auto& [node, data] : shortcutOf) {
    const auto kind = data->launch->standaloneShortcut;
    if ((kind != StandaloneShortcut::kUnbind &&
         kind != StandaloneShortcut::kSplitWithSizes) ||
        data->actualOutputs.size() != 1) {
      continue;
    }
    const auto list = data->actualOutputs[0];
    const auto users = listUsers(list);
    if (users.size() == 1 &&
        kindOf(users[0]) == StandaloneShortcut::kListUnpack &&
        !hostRead[list]) {
      unpackOf[node] = users[0];
      absorbedUnpacks.insert(users[0]);
    }
  }
  // A list pack derives, as nothing, when only kernels and unpacks read its
  // list: kernels read the elements, and an unpack derives from them.
  auto packDerives = [&](const LaunchData& data) {
    if (data.actualOutputs.size() != 1 || hostRead[data.actualOutputs[0]]) {
      return false;
    }
    for (auto* user : listUsers(data.actualOutputs[0])) {
      if (eagerNodes.count(user) != 0 ||
          (shortcutOf.count(user) != 0 &&
           kindOf(user) != StandaloneShortcut::kListUnpack)) {
        return false;
      }
    }
    return true;
  };
  // The element values a list unpack yields, or empty when unknown.
  auto elementsOf = [&](nativert::ValueId list) {
    std::vector<int32_t> elements;
    auto it = idToValue.find(list);
    if (it == idToValue.end()) {
      return elements;
    }
    for (auto* element : it->second->getListElements()) {
      elements.push_back(element->id());
    }
    return elements;
  };

  // A supported shortcut derives over the table whether or not its operand is
  // host-read; the rest run over the frame, which makes their operands and
  // outputs host-read.
  for (auto& steps : state.stepVectors) {
    for (auto& sv : steps) {
      if (sv.program == nullptr) {
        continue;
      }
      std::vector<LaunchData> frameShortcuts;
      for (auto& data : sv.shortcutStandalones) {
        const auto kind = data.launch->standaloneShortcut;
        if (absorbedUnpacks.count(data.standalone) != 0) {
          continue;
        }
        DeriveOp op{
            .kind = kind,
            .intList = data.intList,
            .shortcut =
                static_cast<int32_t>(sv.program->derivedShortcuts.size())};
        bool derive = false;
        if (kind == StandaloneShortcut::kListPack) {
          derive = packDerives(data);
          op.output = data.actualOutputs.empty() ? -1 : data.actualOutputs[0];
        } else if (kind == StandaloneShortcut::kListUnpack) {
          op.elements = elementsOf(
              data.args.empty() || data.args[0] == nullptr
                  ? -1
                  : data.args[0]->id());
          op.outputs = data.actualOutputs;
          derive =
              !op.elements.empty() && op.elements.size() == op.outputs.size();
        } else if (unpackOf.count(data.standalone) != 0) {
          const auto* unpack = shortcutOf.at(unpackOf.at(data.standalone));
          op.outputs = unpack->actualOutputs;
          op.secondShortcut = op.shortcut + 1;
          derive = true;
        } else {
          derive = derivable(kind) && !data.args.empty() &&
              data.args[0] != nullptr && data.actualOutputs.size() == 1;
          op.output = derive ? data.actualOutputs[0] : -1;
        }
        if (!derive) {
          for (auto* arg : data.args) {
            if (arg != nullptr) {
              markHostRead(arg->id());
            }
          }
          for (auto id : data.actualOutputs) {
            markHostRead(id);
          }
          frameShortcuts.push_back(std::move(data));
          continue;
        }
        for (size_t i = 0; i < data.args.size(); ++i) {
          op.operands.push_back(data.args[i] ? data.args[i]->id() : -1);
          op.constants.push_back(i < data.intArgs.size() ? data.intArgs[i] : 0);
        }
        if (op.secondShortcut >= 0) {
          const auto* unpack = shortcutOf.at(unpackOf.at(data.standalone));
          sv.program->derivedShortcuts.push_back(std::move(data));
          sv.program->derivedShortcuts.push_back(*unpack);
        } else {
          sv.program->derivedShortcuts.push_back(std::move(data));
        }
        sv.program->derives.push_back(std::move(op));
      }
      sv.shortcutStandalones = std::move(frameShortcuts);
    }
  }
  for (auto& steps : state.stepVectors) {
    for (auto& sv : steps) {
      if (sv.program == nullptr) {
        continue;
      }
      for (auto& op : sv.program->viewDerives) {
        op.materialize = hostRead[op.output] != 0;
      }
      for (auto& op : sv.program->derives) {
        op.materialize = op.output >= 0 && hostRead[op.output] != 0;
        op.materializeOutputs.clear();
        for (auto id : op.outputs) {
          op.materializeOutputs.push_back(hostRead[id] != 0);
        }
      }
    }
  }
  state.tableOnly.assign(numValues, 0);
  for (size_t id = 0; id < numValues; ++id) {
    state.tableOnly[id] = candidate[id] && !hostRead[id];
  }
}

namespace {

// Lays out arrays of trivially copyable records in one block, each aligned for
// its type.
class CodeLayout {
 public:
  template <typename T>
  size_t add(size_t count) {
    static_assert(std::is_trivially_copyable_v<T>);
    bytes_ = roundUp(bytes_, alignof(T));
    const auto offset = bytes_;
    bytes_ += count * sizeof(T);
    return offset;
  }

  size_t bytes() const {
    return bytes_;
  }

 private:
  static size_t roundUp(size_t n, size_t alignment) {
    return (n + alignment - 1) / alignment * alignment;
  }

  size_t bytes_{0};
};

template <typename T>
T* place(uint8_t* base, size_t offset, const std::vector<T>& from) {
  auto* to = reinterpret_cast<T*>(base + offset);
  std::copy(from.begin(), from.end(), to);
  return to;
}

} // namespace

void packStepCode(
    StepProgram& program,
    const std::vector<uint8_t>& tableOnly,
    const ValueTypes& types,
    const folly::F14FastMap<int32_t, std::vector<int64_t>>& producersOf,
    const folly::F14FastSet<int32_t>& readIds) {
  program.outputDescs.clear();
  std::vector<SizeOp> sizes;
  std::vector<LaunchOp> launches;
  std::vector<OutputOp> outputs;
  std::vector<Span> spans;
  std::vector<ScalarRef> scalars;
  std::vector<int32_t> ids;
  std::vector<int64_t> consts;
  int32_t numShapes = 0;
  auto addIds = [&](const std::vector<int32_t>& from) {
    Span span{
        static_cast<int32_t>(ids.size()), static_cast<int32_t>(from.size())};
    ids.insert(ids.end(), from.begin(), from.end());
    return span;
  };
  auto addConsts = [&](const std::vector<int64_t>& from) {
    Span span{
        static_cast<int32_t>(consts.size()), static_cast<int32_t>(from.size())};
    consts.insert(consts.end(), from.begin(), from.end());
    return span;
  };
  auto addScalars = [&](const std::vector<ScalarRef>& from) {
    Span span{
        static_cast<int32_t>(scalars.size()),
        static_cast<int32_t>(from.size())};
    scalars.insert(scalars.end(), from.begin(), from.end());
    return span;
  };
  for (size_t i = 0; i < program.sizes.size(); ++i) {
    const auto& size = program.sizes[i];
    SizeOp op{
        .fn = size.fn,
        .concat = size.concat,
        .blockSize = size.blockSize,
        .source = static_cast<int32_t>(i)};
    // A list's spans are contiguous, so the ops' spans of spans can be
    // recorded before their elements.
    std::vector<Span> groups;
    groups.reserve(size.args.tensors.size());
    for (const auto& group : size.args.tensors) {
      groups.push_back(addIds(group));
    }
    std::vector<Span> lists;
    lists.reserve(size.args.intLists.size());
    for (const auto& list : size.args.intLists) {
      lists.push_back(addScalars(list));
    }
    op.groups = {
        static_cast<int32_t>(spans.size()),
        static_cast<int32_t>(groups.size())};
    spans.insert(spans.end(), groups.begin(), groups.end());
    op.intLists = {
        static_cast<int32_t>(spans.size()), static_cast<int32_t>(lists.size())};
    spans.insert(spans.end(), lists.begin(), lists.end());
    op.scalars = addScalars(size.args.scalars);
    op.hostIds = addIds(size.args.hostTensors);
    op.shapes = {numShapes, size.numShapes};
    numShapes += size.numShapes;
    sizes.push_back(op);
  }
  folly::F14FastMap<const OutputDesc*, int32_t> descIndex;
  auto addOutputs = [&](const std::vector<BoundOutput>& from) {
    Span span{
        static_cast<int32_t>(outputs.size()),
        static_cast<int32_t>(from.size())};
    for (const auto& output : from) {
      OutputOp op{
          .kind = output.kind,
          .valueId = output.valueId,
          .size = output.size,
          .element = output.element,
          .aliasOf = output.aliasOf,
          .derive = output.derive};
      const auto id = output.valueId;
      if (id >= 0 && static_cast<size_t>(id) < tableOnly.size() &&
          tableOnly[id] != 0) {
        op.flags |= OutputOp::kTableOnly;
      }
      const auto* type = id >= 0 ? types.types.at(id) : nullptr;
      if (type != nullptr) {
        op.flags |= OutputOp::kTyped;
        op.dtype = type->dtype();
      }
      if (output.desc != nullptr) {
        auto [it, inserted] = descIndex.emplace(
            output.desc, static_cast<int32_t>(program.outputDescs.size()));
        if (inserted) {
          program.outputDescs.push_back(output.desc);
        }
        op.desc = it->second;
      }
      outputs.push_back(op);
    }
    return span;
  };
  for (const auto& launch : program.launches) {
    LaunchOp op{
        .kernel = launch.kernel,
        .countSize = launch.countSize,
        .largestShapeSize = launch.largestShapeSize,
        .countFromReserve = launch.countFromReserve,
        .staticNumElements = launch.staticNumElements};
    op.outputs = addOutputs(launch.outputs);
    op.largestOutputs = addOutputs(launch.largestOutputs);
    op.externalInputs = addIds(launch.externalInputs);
    op.tensorOutputs = addIds(launch.tensorOutputs);
    op.allInputs = addIds(launch.allInputs);
    launches.push_back(op);
  }
  auto addDerives = [&](const std::vector<DeriveOp>& from) {
    std::vector<DeriveRec> records;
    for (const auto& op : from) {
      std::vector<int32_t> materializeOutputs(
          op.materializeOutputs.begin(), op.materializeOutputs.end());
      records.push_back(
          {.kind = op.kind,
           .materialize = op.materialize,
           .output = op.output,
           .operands = addIds(op.operands),
           .outputs = addIds(op.outputs),
           .elements = addIds(op.elements),
           .materializeOutputs = addIds(materializeOutputs),
           .constants = addConsts(op.constants),
           .intList = addConsts(op.intList),
           .shortcut = op.shortcut,
           .secondShortcut = op.secondShortcut});
    }
    return records;
  };
  const auto derives = addDerives(program.derives);
  const auto viewDerives = addDerives(program.viewDerives);
  std::vector<ConcatRec> concats;
  concats.reserve(program.concats.size());
  for (const auto& concat : program.concats) {
    std::vector<int32_t> flags;
    flags.reserve(concat.operandIds.size());
    for (size_t i = 0; i < concat.operandIds.size(); ++i) {
      flags.push_back(
          (concat.inPlace[i] ? ConcatRec::kInPlace : 0) |
          (concat.shapeOnDevice[i] ? ConcatRec::kShapeOnDevice : 0));
    }
    concats.push_back(
        {.spec = concat.spec,
         .dtype = concat.dtype,
         .resultId = concat.resultId,
         .operandIds = addIds(concat.operandIds),
         .operandSizes = addIds(concat.operandSizes),
         .operandFlags = addIds(flags)});
  }
  auto addProducers = [&](const folly::F14FastSet<int32_t>& reads) {
    std::vector<int64_t> keys;
    for (auto id : reads) {
      auto it = producersOf.find(id);
      if (it != producersOf.end()) {
        keys.insert(keys.end(), it->second.begin(), it->second.end());
      }
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return addConsts(keys);
  };
  const auto waitProducers = addProducers(program.waitReads);
  const auto readProducers = addProducers(readIds);
  // By kind, so each kind is one tight loop. Fills write disjoint offsets.
  auto fills = program.fills;
  std::stable_sort(
      fills.begin(), fills.end(), [](const FillOp& a, const FillOp& b) {
        return a.kind < b.kind;
      });

  CodeLayout layout;
  const auto sizesAt = layout.add<SizeOp>(sizes.size());
  const auto launchesAt = layout.add<LaunchOp>(launches.size());
  const auto outputsAt = layout.add<OutputOp>(outputs.size());
  const auto fillsAt = layout.add<FillOp>(fills.size());
  const auto derivesAt = layout.add<DeriveRec>(derives.size());
  const auto viewDerivesAt = layout.add<DeriveRec>(viewDerives.size());
  const auto concatsAt = layout.add<ConcatRec>(concats.size());
  const auto spansAt = layout.add<Span>(spans.size());
  const auto scalarsAt = layout.add<ScalarRef>(scalars.size());
  const auto idsAt = layout.add<int32_t>(ids.size());
  const auto constsAt = layout.add<int64_t>(consts.size());
  const auto shapesAt = layout.add<Shape>(numShapes);
  auto& code = program.code;
  code.storage = std::make_unique<uint8_t[]>(layout.bytes());
  auto* base = code.storage.get();
  code.sizes = place(base, sizesAt, sizes);
  code.launches = place(base, launchesAt, launches);
  code.outputs = place(base, outputsAt, outputs);
  code.fills = place(base, fillsAt, fills);
  code.derives = place(base, derivesAt, derives);
  code.viewDerives = place(base, viewDerivesAt, viewDerives);
  code.concats = place(base, concatsAt, concats);
  code.spans = place(base, spansAt, spans);
  code.scalars = place(base, scalarsAt, scalars);
  code.ids = place(base, idsAt, ids);
  code.consts = place(base, constsAt, consts);
  code.shapes = reinterpret_cast<Shape*>(base + shapesAt);
  std::uninitialized_default_construct_n(code.shapes, numShapes);
  code.numSizes = static_cast<int32_t>(sizes.size());
  code.numLaunches = static_cast<int32_t>(launches.size());
  code.numFills = static_cast<int32_t>(fills.size());
  code.numDerives = static_cast<int32_t>(derives.size());
  code.waitProducers = waitProducers;
  code.readProducers = readProducers;
}

} // namespace torch::wave
