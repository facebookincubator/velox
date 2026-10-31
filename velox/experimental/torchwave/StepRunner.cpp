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

// The driver of the bound execution path: runs each step of a node from the
// StepProgram built before the first execution.

#include <cstddef>
#include <cstring>
#include <deque>

#include <ATen/ATen.h>
#include <folly/ScopeGuard.h>
#include <folly/chrono/Hardware.h>

#include "velox/experimental/torchwave/AllocGroup.h"
#include "velox/experimental/torchwave/CompiledOp.h"
#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/GroupAlloc.h"
#include "velox/experimental/torchwave/Standalones.h"
#include "velox/experimental/torchwave/StepProgram.h"
#include "velox/experimental/torchwave/StepSupport.h"
#include "velox/experimental/torchwave/WaveConfig.h"

namespace torch::wave {

namespace {

void evaluateSize(SizeOp& size, StepProgram& program, ExecutionState& state);

// Lays out 'concat' over the table: sizes the operands, gives the result an
// entry of the joined shape, which goes in 'resultShape', and gives every
// operand the kernel writes in place the entry of its band.
void runBoundConcat(
    const ConcatRec& concat,
    StepProgram& program,
    ExecutionState& state,
    Shape& resultShape) {
  ScopedSetupPhase phase(SetupPhase::kReserve);
  auto& code = program.code;
  const auto& spec = concat.spec;
  const auto elementRank = spec.elementRank();
  const auto numOperands = static_cast<size_t>(concat.operandIds.size);
  const auto* operandIds = code.idsOf(concat.operandIds);
  const auto* operandSizes = code.idsOf(concat.operandSizes);
  const auto* operandFlags = code.idsOf(concat.operandFlags);
  // Reused across calls, so a layout allocates only when it grows. One buffer
  // per nesting level: an operand's size can be another concat, which
  // evaluateSize lays out through this function while 'shapes' is in use. A
  // deque keeps an outer level's buffer in place when an inner level is added.
  static thread_local std::deque<std::vector<std::vector<Dim>>> buffers;
  static thread_local size_t depth{0};
  if (buffers.size() <= depth) {
    buffers.resize(depth + 1);
  }
  auto& shapes = buffers[depth];
  ++depth;
  auto popDepth = folly::makeGuard([] { --depth; });
  shapes.resize(numOperands);
  for (size_t i = 0; i < numOperands; ++i) {
    auto& shape = shapes[i];
    shape.assign(elementRank, 0);
    if (operandSizes[i] < 0) {
      continue;
    }
    auto& size = code.sizes[operandSizes[i]];
    evaluateSize(size, program, state);
    const auto& operandShape = code.shapes[size.shapes.begin];
    if (operandShape.rank <= 0) {
      continue;
    }
    if (operandShape.rank == elementRank) {
      std::copy(
          operandShape.dims, operandShape.dims + elementRank, shape.begin());
    } else if (elementRank == 1) {
      shape[0] = static_cast<Dim>(operandShape.numElements());
    } else {
      TORCH_CHECK(
          operandShape.rank < elementRank,
          "Concat operand %",
          operandIds[i],
          " resolved to a rank-",
          static_cast<int>(operandShape.rank),
          " shape, expected rank ",
          static_cast<int>(elementRank));
      // A broadcast size drops leading 1-dims; restore them.
      const auto pad = elementRank - operandShape.rank;
      std::fill(shape.begin(), shape.begin() + pad, 1);
      std::copy(
          operandShape.dims,
          operandShape.dims + operandShape.rank,
          shape.begin() + pad);
    }
  }
  const auto outShape = concatResultShape(spec, shapes);
  int64_t dimStorage[kMaxDims];
  for (size_t d = 0; d < outShape.size(); ++d) {
    dimStorage[d] = outShape[d];
  }
  const c10::IntArrayRef outDims(dimStorage, outShape.size());
  // A result already in the table this execution was placed by its
  // allocation group, whose members are carved into that buffer and whose
  // frame tensor may already be handed out. Reallocating here would orphan
  // both, so the two layouts have to agree.
  TORCH_CHECK(
      !state.valueTable.valid(concat.resultId) ||
          !state.valueTable.isTensor(concat.resultId) ||
          state.valueTable.tensor(concat.resultId).storage == nullptr ||
          holdsTensorOfDims(state, concat.resultId, outDims),
      "Concat result's launch layout differs from its group's layout: %",
      concat.resultId);
  concatResultOwner(state, concat.resultId, outDims, concat.dtype);
  // An operand after one whose extent the kernel sets has an offset the host
  // does not know; the kernel patches its band's base.
  bool canComputeOffset = true;
  int64_t offset = 0;
  for (size_t i = 0; i < numOperands; ++i) {
    const int64_t extent =
        spec.isStack ? 1 : static_cast<int64_t>(shapes[i][spec.dim]);
    if ((operandFlags[i] & ConcatRec::kInPlace) == 0) {
      offset += extent;
      continue;
    }
    const bool pending =
        !canComputeOffset || (operandFlags[i] & ConcatRec::kShapeOnDevice) != 0;
    const int64_t start =
        spec.isStack ? static_cast<int64_t>(i) : (pending ? 0 : offset);
    const auto id = operandIds[i];
    writeBandEntry(state, id, concat.resultId, spec, start, extent);
    materializeIfHostRead(state, id);
    if (pending) {
      canComputeOffset = false;
    } else {
      offset += extent;
    }
  }
  materializeIfHostRead(state, concat.resultId);
  resultShape.rank = static_cast<int8_t>(outShape.size());
  std::copy(outShape.begin(), outShape.end(), resultShape.dims);
}

// Runs 'size' unless it already ran this execution: its concat layout or its
// ReserveFn.
void evaluateSize(SizeOp& size, StepProgram& program, ExecutionState& state) {
  if (size.generation == state.tableGeneration) {
    return;
  }
  size.generation = state.tableGeneration;
  auto* shapes = program.code.shapes + size.shapes.begin;
  if (size.concat >= 0) {
    runBoundConcat(
        program.code.concats[size.concat], program, state, shapes[0]);
    return;
  }
  ScopedSetupPhase phase(SetupPhase::kReserve);
  const auto args = program.code.argsOf(size, state.valueTable);
  const auto& source = program.sizes[size.source];
  // The step program orders a Size after everything it reads, so an operand
  // missing here is a bind error, not a reason to size some other way.
  TORCH_CHECK(
      source.data == nullptr || reserveOperandsReady(args),
      "sizing: the reserve of ",
      source.data->actualOutputDescs[source.outputIdx]
          .reserveAttrNode->target(),
      " output ",
      source.outputIdx,
      " runs before its operands are in the table");
  TORCH_CHECK(
      size.fn(args, shapes, size.shapes.size),
      "sizing: a ReserveFn declined, which it cannot");
}

const Shape& firstShape(const StepProgram& program, const SizeOp& size) {
  return program.code.shapes[size.shapes.begin];
}

// 'shape' as the int64 dims at::empty and resize_ take. Rank -1, a largest
// input that does not exist, reads as a 0-d shape.
c10::IntArrayRef toDims(const Shape& shape, int64_t* storage) {
  const int32_t rank = std::max<int32_t>(shape.rank, 0);
  for (int32_t i = 0; i < rank; ++i) {
    storage[i] = shape.dims[i];
  }
  return c10::IntArrayRef(storage, rank);
}

void makeShapeOnly(
    nativert::ExecutionFrame& frame,
    nativert::ValueId id,
    c10::IntArrayRef dims) {
  auto& existing = frame.getIValue(id);
  if (existing.isTensor()) {
    auto& tensor = existing.toTensor();
    if (tensor.sizes() != dims) {
      tensor.resize_(dims);
    }
    publishFromFrame(frame, id);
    return;
  }
  writeFrame(
      frame,
      id,
      at::empty(dims, at::TensorOptions().dtype(at::kFloat).device(at::kMeta)));
}

// Copies the table's Tensor of 'id' into the parameter at 'dest'. Returns false
// when the table holds no tensor for it.
bool copyTensor(const ValueTable& table, int32_t id, uint8_t* dest) {
  if (!table.valid(id) || !table.isTensor(id)) {
    return false;
  }
  std::memcpy(dest, &table.tensor(id), sizeof(TableTensor));
  reinterpret_cast<Tensor*>(dest)->status = Tensor::kUninited;
  return true;
}

// The shape-only fields of the table's Tensor of 'id' at 'dest', as
// fillShapeOnlyTensorParam writes them.
bool copyShapeOnly(const ValueTable& table, int32_t id, uint8_t* dest) {
  if (!table.valid(id) || !table.isTensor(id)) {
    return false;
  }
  const auto& from = table.tensor(id);
  auto* to = reinterpret_cast<Tensor*>(dest);
  to->storage = nullptr;
  to->rank = from.rank;
  for (int32_t i = 0; i < kMaxDims; ++i) {
    to->dims[i] = from.dims[i];
    to->strides[i] = 0;
  }
  to->numEl = from.numEl;
  to->status = Tensor::kUninited;
  return true;
}

void runFills(
    const StepProgram& program,
    StepVectors& sv,
    ExecutionState& state,
    nativert::ExecutionFrame& frame,
    uint8_t* pinnedBase) {
  const auto& table = state.valueTable;
  auto& counters = threadSetupCounters();
  const auto* fills = program.code.fills;
  for (int32_t f = 0; f < program.code.numFills; ++f) {
    const auto& fill = fills[f];
    auto* dest = pinnedBase + fill.offset;
    switch (fill.kind) {
      case FillOp::kTensor:
        if (copyTensor(table, fill.valueId, dest)) {
          ++counters.fills;
        } else {
          fillEmptyTensorParam(dest);
          sv.kernels[fill.kernel].numElements = 0;
        }
        break;
      case FillOp::kShapeOnly:
        if (!copyShapeOnly(table, fill.valueId, dest)) {
          fillEmptyTensorParam(dest);
          sv.kernels[fill.kernel].numElements = 0;
        }
        break;
      case FillOp::kScalar:
        if (table.valid(fill.valueId) && table.isInt(fill.valueId)) {
          *reinterpret_cast<int64_t*>(dest) = table.scalar(fill.valueId);
        } else if (!copyTensor(table, fill.valueId, dest)) {
          fillScalarParam(frame.getIValue(fill.valueId), dest, fill.valueId);
        }
        break;
      case FillOp::kZero64:
        *reinterpret_cast<int64_t*>(dest) = 0;
        break;
      case FillOp::kZero32:
        *reinterpret_cast<int32_t*>(dest) = 0;
        break;
      case FillOp::kUninited:
        reinterpret_cast<Tensor*>(dest)->status = Tensor::kUninited;
        break;
    }
  }
}

const std::string kDynamicSizeKey{"dyn"};

// Gives the derived entry of 'id' the owner of 'source', whose storage it
// views: the source's allocation, or a new one holding the source's frame
// tensor so the derived value can be materialized and outlives the source.
void takeOwner(
    ExecutionState& state,
    nativert::ValueId id,
    nativert::ValueId source) {
  auto& out = state.valueCold[id];
  out.owner = source >= 0 ? state.valueCold[source].owner : -1;
  if (out.owner < 0 && source >= 0) {
    const auto& base = state.frame->getIValue(source);
    if (base.isTensor()) {
      out.owner = addAllocation(state, base.toTensor());
    }
  }
  if (out.owner >= 0) {
    ++state.allocations[out.owner].liveValues;
  }
}

// Runs the step's DeriveOps; one whose operands the table cannot answer runs
// as its shortcut over the frame.
void runDerives(StepProgram& program, ExecutionState& state) {
  auto& frame = *state.frame;
  const auto& code = program.code;
  for (int32_t d = 0; d < code.numDerives; ++d) {
    const auto& rec = code.derives[d];
    const auto op = code.viewOf(rec);
    const auto input = op.operands.empty() ? -1 : op.operands[0];
    if (runDerive(op, state.valueTable)) {
      if (op.output >= 0 && op.kind != StandaloneShortcut::kListPack) {
        takeOwner(state, op.output, input);
        if (rec.materialize || !state.tableOnlyActive) {
          materializeEntry(state, op.output, input);
        }
      }
      const auto* materializeOutputs = code.idsOf(rec.materializeOutputs);
      for (size_t i = 0; i < op.outputs.size(); ++i) {
        const auto source =
            op.kind == StandaloneShortcut::kListUnpack ? op.elements[i] : input;
        if (state.valueEntries[op.outputs[i]].kind == ValueEntry::kTensor) {
          takeOwner(state, op.outputs[i], source);
        }
        if (materializeOutputs[i] != 0 || !state.tableOnlyActive) {
          materializeEntry(state, op.outputs[i], source);
        }
      }
      continue;
    }
    if (input >= 0 && !frame.getIValue(input).isTensor() &&
        state.valueTable.valid(input) && state.valueTable.isTensor(input)) {
      materializeEntry(state, input, input);
    }
    runStandaloneShortcut(program.derivedShortcuts[rec.shortcut], frame);
    if (rec.secondShortcut >= 0) {
      runStandaloneShortcut(
          program.derivedShortcuts[rec.secondShortcut], frame);
    }
  }
}

} // namespace

void CompositeInvocation::checkUnreadyOperands(
    const LaunchOp& launch,
    const StepCode& code,
    LaunchData& data,
    ExecutionState& state) {
  const auto& table = state.valueTable;
  auto absent = [&](int32_t id) {
    return !table.valid(id) || !table.isTensor(id);
  };
  if (data.numElements > 0) {
    const auto* ids = code.idsOf(launch.externalInputs);
    for (int32_t i = 0; i < launch.externalInputs.size; ++i) {
      if (absent(ids[i])) {
        data.numElements = 0;
        break;
      }
    }
  }
  if (data.numElements > 0) {
    auto* collector = currentAllocCollector();
    const auto* ids = code.idsOf(launch.tensorOutputs);
    for (int32_t i = 0; i < launch.tensorOutputs.size; ++i) {
      if (collector != nullptr && collector->owns(ids[i])) {
        continue;
      }
      if (absent(ids[i])) {
        data.numElements = 0;
        break;
      }
    }
  }
  if (data.numElements == 0 && WaveConfig::get().isCg.value_or(false)) {
    int64_t staticNumElements = launch.staticNumElements;
    const auto* ids = code.idsOf(launch.allInputs);
    for (int32_t i = 0; i < launch.allInputs.size; ++i) {
      if (!absent(ids[i])) {
        staticNumElements =
            std::max<int64_t>(staticNumElements, table.tensor(ids[i]).numEl);
      }
    }
    data.numElements = staticNumElements;
  }
}

void CompositeInvocation::runBoundOutput(
    const OutputOp& output,
    StepProgram& program,
    ExecutionState& state) {
  auto& frame = *state.frame;
  if (output.aliasOf >= 0) {
    const auto& self = frame.getIValue(output.aliasOf);
    if (self.isTensor()) {
      writeFrame(frame, output.valueId, self.toTensor().alias());
      return;
    }
  }
  int64_t dimStorage[kMaxDims];
  switch (output.kind) {
    case BoundOutput::kTensor:
    case BoundOutput::kListElement: {
      auto& size = program.code.sizes[output.size];
      evaluateSize(size, program, state);
      const auto& shape =
          program.code.shapes[size.shapes.begin + output.element];
      if ((output.flags & OutputOp::kTableOnly) != 0 && state.tableOnlyActive) {
        // A member of one of the step's groups: the carve writes its entry,
        // so sizing records only the shape.
        auto* collector = currentAllocCollector();
        const auto dims = toDims(shape, dimStorage);
        if (collector != nullptr && collector->capture(output.valueId, dims)) {
          writeShapeEntry(state, output.valueId, dims, output.dtype);
          break;
        }
        if ((output.flags & OutputOp::kTyped) != 0) {
          if (!holdsTensorOfDims(state, output.valueId, dims)) {
            releaseOwner(state, output.valueId);
            allocateTableOnly(state, output.valueId, dims, output.dtype);
          }
          break;
        }
      }
      ensureCudaTensor(
          frame,
          *state.valueTypes,
          output.valueId,
          toDims(shape, dimStorage),
          kDynamicSizeKey);
      break;
    }
    case BoundOutput::kShapeOnly: {
      auto& size = program.code.sizes[output.size];
      evaluateSize(size, program, state);
      makeShapeOnly(
          frame, output.valueId, toDims(firstShape(program, size), dimStorage));
      break;
    }
    case BoundOutput::kAlias:
      break;
    case BoundOutput::kView: {
      if (output.derive >= 0) {
        const auto& rec = program.code.viewDerives[output.derive];
        const auto op = program.code.viewOf(rec);
        const auto base = op.operands[0];
        if (runDerive(op, state.valueTable)) {
          takeOwner(state, op.output, base);
          if (rec.materialize || !state.tableOnlyActive) {
            materializeEntry(state, op.output, base);
          }
          break;
        }
        if (!frame.getIValue(base).isTensor() && state.valueTable.valid(base) &&
            state.valueTable.isTensor(base)) {
          materializeEntry(state, base, base);
        }
      }
      const auto& desc = *program.outputDescs[output.desc];
      if (runViewRecipe(desc, frame)) {
        break;
      }
      auto it = state.kernelMap->find(desc.viewNode);
      TORCH_CHECK(
          it != state.kernelMap->end(),
          "No kernel for view node ",
          desc.viewNode->target());
      ScopedSetupPhase viewPhase(SetupPhase::kView);
      executeNode(desc.viewNode, it->second, frame);
      break;
    }
    case BoundOutput::kConcat: {
      auto& size = program.code.sizes[output.size];
      evaluateSize(size, program, state);
      if (size.concat >= 0) {
        break;
      }
      ensureCudaTensor(
          frame,
          *state.valueTypes,
          output.valueId,
          toDims(firstShape(program, size), dimStorage),
          kDynamicSizeKey);
      break;
    }
  }
}

namespace {

void sizeExprReads(const SizeExpr& expr, std::vector<nativert::ValueId>& out) {
  out.insert(out.end(), expr.values.begin(), expr.values.end());
  for (const auto& arg : expr.args) {
    sizeExprReads(arg, out);
  }
}

} // namespace

void CompositeInvocation::sizeConcatReadsAhead(
    const ConcatGroupLayout& concat,
    ExecutionState& state) {
  std::vector<nativert::ValueId> reads;
  for (const auto& operand : concat.operands) {
    if (operand.sizeExpr.op != SizeShortcut::kNone) {
      sizeExprReads(operand.sizeExpr, reads);
    }
  }
  const int32_t numSteps =
      getStepVectors(state.stepVectors, sequenceNumber_, 0).boundSteps;
  int64_t dimStorage[kMaxDims];
  for (const auto id : reads) {
    if (state.valueTable.valid(id)) {
      continue;
    }
    bool found = false;
    for (int32_t step = 0; step < numSteps && !found; ++step) {
      auto& program =
          *getStepVectors(state.stepVectors, sequenceNumber_, step).program;
      auto& code = program.code;
      for (int32_t l = 0; l < code.numLaunches && !found; ++l) {
        const auto& launch = code.launches[l];
        for (int32_t i = 0; i < launch.outputs.size; ++i) {
          const auto& output = code.outputs[launch.outputs.begin + i];
          if (output.valueId != id || output.kind != BoundOutput::kTensor) {
            continue;
          }
          auto& size = code.sizes[output.size];
          evaluateSize(size, program, state);
          const auto& shape = code.shapes[size.shapes.begin + output.element];
          writeShapeEntry(state, id, toDims(shape, dimStorage), output.dtype);
          found = true;
          break;
        }
      }
    }
  }
}

void CompositeInvocation::runBoundLaunch(
    const LaunchOp& launch,
    StepProgram& program,
    StepVectors& sv,
    ExecutionState& state) {
  auto& code = program.code;
  auto& data = sv.kernels[launch.kernel];
  auto& count = code.sizes[launch.countSize];
  evaluateSize(count, program, state);
  const auto& countShape = firstShape(program, count);
  data.numElements = launch.countFromReserve
      ? countShape.numElements()
      : static_cast<int64_t>(countShape.dims[0]);
  Span outputs = launch.outputs;
  if (launch.largestShapeSize >= 0) {
    auto& largest = code.sizes[launch.largestShapeSize];
    evaluateSize(largest, program, state);
    if (firstShape(program, largest).rank >= 0) {
      outputs = launch.largestOutputs;
    }
  }
  for (int32_t i = 0; i < outputs.size; ++i) {
    runBoundOutput(code.outputs[outputs.begin + i], program, state);
  }
  checkUnreadyOperands(launch, code, data, state);
}

void CompositeInvocation::executeBound(ExecutionState& state) {
  auto& frame = *state.frame;
  NodeRun run = startNodeRun(state);
  // The plan is settled while the graph compiles, before the trace bits are
  // typically set; printed once, by the first node that runs with the bit set.
  if ((WaveConfig::get().trace & WaveConfig::kTiming) != 0 &&
      !state.waveGraph->allocGroupReport().empty() &&
      state.waveGraph->takeAllocGroupReportUnprinted()) {
    std::cout << state.waveGraph->concatCarveReport()
              << state.waveGraph->allocGroupReport();
  }
  using Clock = std::chrono::high_resolution_clock;
  auto elapsed = [](Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               Clock::now() - start)
        .count();
  };
  if (WaveConfig::get().trace & (WaveConfig::kNodes | WaveConfig::kLaunches)) {
    std::cout << "==== Node " << sequenceNumber_ << " (bound)" << std::endl;
  }
  const int32_t numSteps =
      getStepVectors(state.stepVectors, sequenceNumber_, 0).boundSteps;
  for (int32_t stepIdx = 0; stepIdx < numSteps; ++stepIdx) {
    auto& sv = getStepVectors(state.stepVectors, sequenceNumber_, stepIdx);
    auto& program = *sv.program;
    drainBeforeStepIfRequested(state);
    enforceDelayedFreeLimit(state);
    auto& gridChoices = state.stepVectors.at(sequenceNumber_).at(0).gridChoices;
    const int64_t totalAllocBytes = program.totalPinnedBytes +
        static_cast<int64_t>(sv.blockCapacity) *
            static_cast<int64_t>(sizeof(DebugInfo));
    if (totalAllocBytes > sv.bufferBytes) {
      sv.pinnedBase = getOrAllocateBuffer(
                          state.pinnedBuffers,
                          sequenceNumber_,
                          stepIdx,
                          totalAllocBytes,
                          state.pinnedArena)
                          ->as<uint8_t>();
      sv.deviceBase = getOrAllocateBuffer(
                          state.deviceBuffers,
                          sequenceNumber_,
                          stepIdx,
                          totalAllocBytes,
                          state.deviceArena)
                          ->as<uint8_t>();
      sv.bufferBytes = totalAllocBytes;
    }
    auto* pinnedBase = sv.pinnedBase;
    auto* deviceBase = sv.deviceBase;
    const SetupCounters setupBefore = threadSetupCounters();
    {
      auto& collector = *program.collector;
      AllocGroupCollector::Scope collecting(collector);
      auto t0 = Clock::now();
      sv.allocUs = 0;
      sv.allocCallUs = 0;
      sv.fillUs = 0;
      sv.allocGroups = 0;
      sv.allocGroupTensors = 0;
      const auto* launches = program.code.launches;
      const int32_t numLaunches = program.code.numLaunches;
      for (int32_t l = 0; l < program.numIndependent; ++l) {
        runBoundLaunch(launches[l], program, sv, state);
      }
      materializeStepGroups(
          state, collector, sv, /*afterWait=*/false, run.doTiming);
      int64_t waitUs = 0;
      if (program.numIndependent < numLaunches) {
        // Blocks the host on the device: idle, not interpretation.
        const auto throughStep = neededPendingStep(
            state,
            program.code.consts + program.code.waitProducers.begin,
            program.code.waitProducers.size);
        if (throughStep >= 0) {
          const auto tWait = Clock::now();
          resolvePendingReturns(state, throughStep);
          waitUs = elapsed(tWait);
        }
        for (int32_t l = program.numIndependent; l < numLaunches; ++l) {
          runBoundLaunch(launches[l], program, sv, state);
        }
      }
      materializeStepGroups(
          state, collector, sv, /*afterWait=*/true, run.doTiming);
      if (run.doTiming) {
        sv.gatherUs = elapsed(t0) - sv.allocUs - waitUs;
      }
    }
    if (!sv.standalones.empty() || !sv.shortcutStandalones.empty() ||
        program.code.numDerives > 0) {
      // A standalone or derive reads its inputs when it runs, which is after
      // this step's transfers have been waited for.
      const auto throughStep = neededPendingStep(
          state,
          program.code.consts + program.code.readProducers.begin,
          program.code.readProducers.size);
      if (throughStep >= 0) {
        resolvePendingReturns(state, throughStep);
      }
    }
    runDerives(program, state);
    {
      const uint64_t tFill = run.doTiming ? folly::hardware_timestamp() : 0;
      runFills(program, sv, state, frame, pinnedBase);
      if (run.doTiming) {
        sv.fillUs += static_cast<int64_t>(
            static_cast<double>(folly::hardware_timestamp() - tFill) /
            tscTicksPerMicro());
      }
    }
    sv.refCheckUs = 0;
    sv.elidedCloneBytes = 0;
    if (sv.kernels.empty() && sv.standalones.empty() &&
        sv.shortcutStandalones.empty() && program.code.numDerives == 0) {
      // Nothing launches past here, but a later step can still hold groups: a
      // concat laid out after a value the node's host plumbing writes past its
      // grids is planned a step later still. Its bands are what a later node's
      // copies write into, so it is carved now rather than dropped.
      for (int32_t later = stepIdx + 1; later < numSteps; ++later) {
        auto& laterSv =
            getStepVectors(state.stepVectors, sequenceNumber_, later);
        auto& laterCollector = *laterSv.program->collector;
        AllocGroupCollector::Scope collecting(laterCollector);
        materializeStepGroups(
            state, laterCollector, laterSv, /*afterWait=*/true, run.doTiming);
      }
      break;
    }
    run.lastExecStep = stepIdx;
    sampleRunAhead(state);
    finishStep(
        run,
        stepIdx,
        sv,
        pinnedBase,
        deviceBase,
        program.totalPinnedBytes,
        program.returnBegin,
        program.returnEnd,
        setupBefore,
        gridChoices);
  }
  finishNode(run);
}

} // namespace torch::wave
