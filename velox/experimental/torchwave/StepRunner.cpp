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
#include <c10/util/StringUtil.h>
#include <folly/ScopeGuard.h>
#include <folly/chrono/Hardware.h>

#include "velox/experimental/torchwave/AllocGroup.h"
#include "velox/experimental/torchwave/CompiledOp.h"
#include "velox/experimental/torchwave/DeviceGrid.h"
#include "velox/experimental/torchwave/DeviceSymbols.h"
#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/GroupAlloc.h"
#include "velox/experimental/torchwave/Standalones.h"
#include "velox/experimental/torchwave/StepProgram.h"
#include "velox/experimental/torchwave/StepSupport.h"
#include "velox/experimental/torchwave/WaveConfig.h"

namespace torch::wave {

namespace {

void evaluateSize(SizeOp& size, StepProgram& program, ExecutionState& state);

// A concat whose result went into a slot of the step's collector: its bands
// are written again once the group is carved and the result has storage.
struct DeferredConcat {
  const ConcatRec* concat;
  StepProgram* program;
};

// The concats of the step being set up whose bands wait for a carve.
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
thread_local std::vector<DeferredConcat> tDeferredConcats;

// A host-side view whose base is a slot the step's groups have not carved: its
// entry is derived again once the base has storage.
struct DeferredView {
  const OutputOp* output;
  StepProgram* program;
};

// The views of the step being set up that wait for a carve, in sizing order, so
// a view of a deferred view is derived after it.
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
thread_local std::vector<DeferredView> tDeferredViews;

// An in-place output whose self is a member the step's groups have not carved:
// the output becomes an alias of the self once the carve gives it storage.
struct DeferredAlias {
  int32_t output;
  int32_t self;
};

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
thread_local std::vector<DeferredAlias> tDeferredAliases;

// Lays out 'concat' over the table: sizes the operands, gives the result an
// entry of the joined shape, which goes in 'resultShape', and gives every
// operand the kernel writes in place the entry of its band.
void runBoundConcat(
    const ConcatRec& concat,
    StepProgram& program,
    ExecutionState& state,
    Shape& resultShape,
    bool allowSlot = true) {
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
  // A result the plan groups is sized into its slot; the bands below then
  // point at a base the carve has not given yet and are written again after
  // it. They are valid entries meanwhile, so the step does not count its
  // operands absent.
  bool deferred = false;
  if (allowSlot && WaveConfig::get().concatResultSlots &&
      !holdsTensorOfDims(state, concat.resultId, outDims)) {
    auto* collector = currentAllocCollector();
    if (collector != nullptr && collector->capture(concat.resultId, outDims)) {
      writeShapeEntry(state, concat.resultId, outDims, concat.dtype);
      tDeferredConcats.push_back({&concat, &program});
      deferred = true;
    }
  }
  if (!deferred) {
    concatResultOwner(state, concat.resultId, outDims, concat.dtype);
  }
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
    if (!deferred) {
      materializeIfHostRead(state, id);
    }
    if (pending) {
      canComputeOffset = false;
    } else {
      offset += extent;
    }
  }
  if (!deferred) {
    materializeIfHostRead(state, concat.resultId);
  }
  resultShape.rank = static_cast<int8_t>(outShape.size());
  std::copy(outShape.begin(), outShape.end(), resultShape.dims);
}

// Writes the bands of the deferred concats whose results a carve has placed.
// With 'last', a result still without storage gets an allocation of its own.
void finishDeferredConcats(ExecutionState& state, bool last) {
  auto& pending = tDeferredConcats;
  size_t kept = 0;
  for (const auto& deferred : pending) {
    const auto id = deferred.concat->resultId;
    if (!last && state.valueTable.tensor(id).storage == nullptr) {
      pending[kept++] = deferred;
      continue;
    }
    Shape shape;
    runBoundConcat(
        *deferred.concat, *deferred.program, state, shape, /*allowSlot=*/false);
  }
  pending.resize(kept);
}

} // namespace

void verifyConstantSize(
    const SizeOp& size,
    const StepProgram& program,
    const ValueTable& table) {
  const auto count = size.shapes.size;
  std::vector<Shape> actual(count);
  const auto args = program.code.argsOf(size, table);
  // Builds pending operand entries, as evaluateSize does before a call.
  TORCH_CHECK(
      reserveOperandsReady(args),
      "sizing: a constant Size is verified before its operands are in the table");
  TORCH_CHECK(
      size.fn(args, actual.data(), count),
      "sizing: a ReserveFn declined, which it cannot");
  const auto* expected = program.code.shapes + size.shapes.begin;
  for (int32_t i = 0; i < count; ++i) {
    const auto& lhs = expected[i];
    const auto& rhs = actual[i];
    if (lhs.rank == rhs.rank &&
        std::equal(lhs.dims, lhs.dims + lhs.rank, rhs.dims)) {
      continue;
    }
    const auto& source = program.sizes.at(size.source);
    std::string operands;
    for (int32_t g = 0; g < args.numGroups; ++g) {
      for (size_t e = 0; e < args.groupSize(g); ++e) {
        const auto id = args.id(g, e);
        operands +=
            fmt::format(" %{}{}", id, table.valid(id) ? "" : "(absent)");
      }
    }
    TORCH_CHECK(
        false,
        "Static size differs from the reserve: ",
        source.data != nullptr
            ? std::string(source.data->actualOutputDescs[source.outputIdx]
                              .reserveAttrNode->target())
            : std::string("built-in"),
        " operands",
        operands,
        ", size ",
        size.source,
        " shape ",
        i,
        ", static [",
        c10::Join(", ", std::vector<int64_t>(lhs.dims, lhs.dims + lhs.rank)),
        "], reserve [",
        c10::Join(", ", std::vector<int64_t>(rhs.dims, rhs.dims + rhs.rank)),
        "]");
  }
}

namespace {

// Runs 'size' unless it already ran this execution: its concat layout or its
// ReserveFn.
void evaluateSize(SizeOp& size, StepProgram& program, ExecutionState& state) {
  if (size.constant) {
    if (WaveConfig::get().verifyStaticSizes &&
        size.generation != state.tableGeneration) {
      size.generation = state.tableGeneration;
      verifyConstantSize(size, program, state.valueTable);
    }
    return;
  }
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
  const auto& source = program.sizes.at(size.source);
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

} // namespace

void runShapeChecks(const StepProgram& program, const ValueTable& table) {
  for (const auto& check : program.shapeChecks) {
    if (!table.valid(check.valueId) || !table.isTensor(check.valueId)) {
      continue;
    }
    const auto& tensor = table.tensor(check.valueId);
    bool same = tensor.rank == static_cast<int8_t>(check.dims.size());
    for (int32_t d = 0; same && d < tensor.rank; ++d) {
      same = tensor.dims[d] == check.dims[d];
    }
    if (same && check.contiguous) {
      int64_t stride = 1;
      for (int32_t d = tensor.rank - 1; d >= 0; --d) {
        TORCH_CHECK(
            tensor.dims[d] == 1 || tensor.strides[d] == stride,
            "Tensor layout differs from the dense layout its kernel was "
            "compiled for: value %",
            check.valueId,
            ", dim ",
            d,
            " has stride ",
            tensor.strides[d],
            " instead of ",
            stride);
        stride *= tensor.dims[d];
      }
    }
    if (!same) {
      std::vector<int64_t> actual(tensor.dims, tensor.dims + tensor.rank);
      TORCH_CHECK(
          false,
          "Tensor shape differs from the static shape its kernel was "
          "compiled for: value %",
          check.valueId,
          ", compiled for [",
          c10::Join(", ", check.dims),
          "], actual [",
          c10::Join(", ", actual),
          "]");
    }
  }
}

namespace {

void runFills(
    const StepProgram& program,
    StepVectors& sv,
    ExecutionState& state,
    nativert::ExecutionFrame& frame,
    uint8_t* pinnedBase) {
  const auto& table = state.valueTable;
  auto& counters = threadSetupCounters();
  for (auto id : program.rootAddressIds) {
    TORCH_CHECK(
        table.valid(id) && table.isTensor(id),
        "The setup places a view of %",
        id,
        ", which the host has not made");
    setAddress(
        state,
        state.valueAddressBase + id,
        reinterpret_cast<int64_t>(table.tensor(id).storage));
  }
  const auto* fills = program.code.fills;
  for (int32_t f = 0; f < program.code.numFills; ++f) {
    const auto& fill = fills[f];
    auto* dest = pinnedBase + fill.offset;
    // Only the setup of a step on the path writes an export, over the header.
    const bool deviceOnly = fill.valueId >= 0 &&
        fill.valueId < static_cast<int32_t>(state.deviceOnlyValue.size()) &&
        state.deviceOnlyValue[fill.valueId] != 0;
    if (fill.kernel == FillOp::kExport &&
        !(state.deviceAllocActive && sv.deviceAllocStep && deviceOnly)) {
      continue;
    }
    if (state.deviceAllocActive && fill.valueId >= 0 &&
        fill.valueId < static_cast<int32_t>(state.deviceOnlyValue.size()) &&
        state.deviceOnlyValue[fill.valueId] != 0 &&
        fill.kind <= FillOp::kScalar) {
      // A step on the path has the setup write it over a static header:
      // rank, element type, and unit dims, which read as a dense layout.
      // A step off the path has fetched its entry.
      if (sv.deviceAllocStep) {
        if (fill.kind == FillOp::kScalar) {
          continue;
        }
        // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
        const auto& header = state.staticHeaders[fill.valueId];
        const auto rank = header.rank;
        const auto dtype = header.dtype;
        TORCH_CHECK(rank >= 0, "No static header for %", fill.valueId);
        auto* to = reinterpret_cast<Tensor*>(dest);
        std::memset(dest, 0, sizeof(TableTensor));
        to->rank = rank;
        if (fill.kind == FillOp::kTensor) {
          to->elementSize = static_cast<uint8_t>(
              c10::elementSize(static_cast<c10::ScalarType>(dtype)));
          to->elementType = static_cast<uint8_t>(dtype);
        }
        int64_t numEl = 1;
        for (int32_t d = 0; d < rank; ++d) {
          to->dims[d] = header.dims[d];
          numEl *= header.dims[d];
        }
        // Dense strides over these dims, so the host layout reads as dense.
        int64_t stride = 1;
        for (int32_t d = rank - 1; d >= 0; --d) {
          to->strides[d] =
              header.dims[d] == 1 ? 0 : static_cast<int32_t>(stride);
          stride *= header.dims[d];
        }
        to->numEl = static_cast<uint32_t>(numEl);
        to->status = Tensor::kUninited;
        continue;
      }
      TORCH_CHECK(
          table.valid(fill.valueId),
          "A step off the device path reads a value a step on the device "
          "path produces, without fetching it: %",
          fill.valueId);
    }
    switch (fill.kind) {
      case FillOp::kTensor:
        if (copyTensor(table, fill.valueId, dest)) {
          ++counters.fills;
        } else {
          fillEmptyTensorParam(dest);
          sv.kernels[fill.kernel].numElements = 0;
        }
        // The setup kernel places this value from the address table.
        if (fill.valueId >= 0 &&
            fill.valueId < static_cast<int32_t>(state.hasValueAddress.size()) &&
            state.hasValueAddress[fill.valueId] != 0) {
          setAddress(
              state,
              state.valueAddressBase + fill.valueId,
              *reinterpret_cast<const int64_t*>(dest));
        }
        setValueShape(state, fill.valueId, dest);
        break;
      case FillOp::kShapeOnly:
        if (!copyShapeOnly(table, fill.valueId, dest)) {
          fillEmptyTensorParam(dest);
          sv.kernels[fill.kernel].numElements = 0;
        }
        setValueShape(state, fill.valueId, dest);
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

// The node and the steps that make 'id', for error messages.
std::string describeProducer(const ExecutionState& state, int32_t id) {
  const auto& idToValue = state.waveGraph->idToValue();
  auto it = idToValue.find(id);
  const auto* node = it != idToValue.end() ? it->second->producer() : nullptr;
  std::string where = ", made by " +
      (node != nullptr ? std::string(node->target()) : std::string("no node"));
  if (static_cast<size_t>(id) < state.deviceOnlyValue.size() &&
      state.deviceOnlyValue[id] != 0) {
    where += ", made on the device path";
  }
  for (size_t n = 0; n < state.stepVectors.size(); ++n) {
    for (size_t k = 0; k < state.stepVectors[n].size(); ++k) {
      const auto& producer = state.stepVectors[n][k];
      for (const auto* launches :
           {&producer.kernels,
            &producer.standalones,
            &producer.shortcutStandalones}) {
        for (const auto& data : *launches) {
          const bool out =
              std::find(
                  data.actualOutputs.begin(), data.actualOutputs.end(), id) !=
              data.actualOutputs.end();
          const bool ret =
              std::find(
                  data.returnValues.begin(), data.returnValues.end(), id) !=
              data.returnValues.end();
          if (out || ret) {
            where += fmt::format(
                "; step {}.{}{}{}{}",
                n,
                k,
                out ? " output" : "",
                ret ? " returned" : "",
                producer.deviceAllocStep ? " on the path" : "");
          }
        }
      }
      if (producer.program != nullptr) {
        for (const auto& derive : producer.program->derives) {
          if (derive.output == id) {
            where += fmt::format(
                "; step {}.{} derive{}",
                n,
                k,
                producer.deviceAllocStep ? " on the path" : "");
          }
        }
      }
    }
  }
  return where;
}

// Writes the SymInt list 'list' to the frame from its elements, each an
// integer of the table or the frame.
void writeIntList(
    ExecutionState& state,
    int32_t list,
    ArrayView<int32_t> elements) {
  auto& frame = *state.frame;
  std::vector<int64_t> ints;
  ints.reserve(elements.size());
  for (auto id : elements) {
    if (state.valueTable.valid(id) && state.valueTable.isInt(id)) {
      ints.push_back(state.valueTable.scalar(id));
      continue;
    }
    const auto& value = frame.getIValue(id);
    TORCH_CHECK(
        value.isSymInt() || value.isInt(),
        "Element %",
        id,
        " of SymInt list %",
        list,
        " is in neither the table nor the frame: ",
        value.tagKind(),
        state.valueTable.valid(id) ? ", table entry not an int" : "",
        describeProducer(state, id));
    ints.push_back(
        value.isSymInt() ? value.toSymInt().guard_int(__FILE__, __LINE__)
                         : value.toInt());
  }
  writeFrame(frame, list, c10::IValue(std::move(ints)));
}

// Writes to the frame each operand that a prim.ListPack of SymInts makes and
// the frame does not hold: a pack that derives makes the frame's list only
// when host code reads it.
void writeIntListOperands(ExecutionState& state, ArrayView<int32_t> operands) {
  const auto& idToValue = state.waveGraph->idToValue();
  for (auto id : operands) {
    if (id < 0 || !state.frame->getIValue(id).isNone()) {
      continue;
    }
    auto it = idToValue.find(id);
    const auto* pack = it != idToValue.end() ? it->second->producer() : nullptr;
    if (pack == nullptr || pack->target() != "prim.ListPack" ||
        it->second->type().kind() != nativert::Type::Kind::SymIntList) {
      continue;
    }
    std::vector<int32_t> elements;
    bool allValues = true;
    for (const auto& input : pack->inputs()) {
      if (input.value == nullptr) {
        allValues = false;
        break;
      }
      elements.push_back(input.value->id());
    }
    // A pack with an element that is not a value is not one the host derives.
    if (!allValues) {
      continue;
    }
    writeIntList(
        state, id, {elements.data(), static_cast<int32_t>(elements.size())});
  }
}

// Runs derive 'd' of 'program'; one whose operands the table cannot answer
// runs as its shortcut over the frame. 'materialize' makes the frame's tensor
// of the output even when the derive does not.
void runDeriveRec(
    StepProgram& program,
    int32_t d,
    bool materialize,
    ExecutionState& state) {
  auto& frame = *state.frame;
  const auto& code = program.code;
  const auto& rec = code.derives[d];
  const auto op = code.viewOf(rec);
  const auto input = op.operands.empty() ? -1 : op.operands[0];
  materialize = materialize || rec.materialize || !state.tableOnlyActive;
  if (op.kind == StandaloneShortcut::kListPack && materialize) {
    const auto& pack = program.derivedShortcuts.at(rec.shortcut);
    if (!pack.standalone->outputs().empty() &&
        pack.standalone->outputs()[0]->type().kind() ==
            nativert::Type::Kind::SymIntList) {
      std::vector<int32_t> elements;
      elements.reserve(pack.args.size());
      for (const auto* arg : pack.args) {
        elements.push_back(arg->id());
      }
      writeIntList(
          state,
          op.output,
          {elements.data(), static_cast<int32_t>(elements.size())});
    }
  }
  if (runDerive(op, state.valueTable)) {
    if (op.output >= 0 && op.kind != StandaloneShortcut::kListPack) {
      // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
      if (state.valueCold[op.output].owner < 0) {
        takeOwner(state, op.output, input);
      }
      if (materialize) {
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
    return;
  }
  if (input >= 0 && !frame.getIValue(input).isTensor() &&
      state.valueTable.valid(input) && state.valueTable.isTensor(input)) {
    materializeEntry(state, input, input);
  }
  // A SymInt list the host packs from its elements, which the frame holds.
  writeIntListOperands(state, op.operands);
  // The frame does not hold values the device path makes.
  for (auto id : op.operands) {
    TORCH_CHECK(
        !state.deviceAllocActive || id < 0 ||
            static_cast<size_t>(id) >= state.deviceOnlyValue.size() ||
            state.deviceOnlyValue[id] == 0 || state.valueTable.valid(id) ||
            !frame.getIValue(id).isNone(),
        "A derive off the device path reads %",
        id,
        ", which the device path makes, with no entry for it",
        describeProducer(state, id),
        "; the derive makes %",
        op.output);
  }
  runStandaloneShortcut(program.derivedShortcuts.at(rec.shortcut), frame);
  if (rec.secondShortcut >= 0) {
    runStandaloneShortcut(
        program.derivedShortcuts.at(rec.secondShortcut), frame);
  }
}

// The in-place outputs and views a step on the device path makes keep their
// bases' allocations live, as a host derive's takeOwner does: the host does
// not run them, and a group released at its members' last direct reads would
// be carved again under a view still to be read.
void ownDerivedViews(
    const StepVectors& sv,
    StepProgram& program,
    ExecutionState& state) {
  const auto& code = program.code;
  auto own = [&](int32_t id, int32_t base) {
    if (id >= 0 && base >= 0 && state.valueCold[id].owner < 0) {
      takeOwner(state, id, base);
    }
  };
  // In-place outputs share their selves' buffers.
  for (const auto& data : sv.kernels) {
    for (size_t i = 0;
         i < data.actualOutputDescs.size() && i < data.actualOutputs.size();
         ++i) {
      if (data.actualOutputDescs[i].aliasSelfId.has_value()) {
        own(data.actualOutputs[i], *data.actualOutputDescs[i].aliasSelfId);
      }
    }
  }
  for (int32_t d = 0; d < code.numDerives; ++d) {
    const auto op = code.viewOf(code.derives[d]);
    if (op.kind == StandaloneShortcut::kListPack ||
        op.kind == StandaloneShortcut::kListUnpack ||
        op.kind == StandaloneShortcut::kSymSize ||
        op.kind == StandaloneShortcut::kSymNumel || op.operands.empty()) {
      continue;
    }
    own(op.output, op.operands[0]);
    for (auto id : op.outputs) {
      own(id, op.operands[0]);
    }
  }
}

// Runs the step's DeriveOps.
void runDerives(StepProgram& program, ExecutionState& state) {
  for (int32_t d = 0; d < program.code.numDerives; ++d) {
    runDeriveRec(program, d, false, state);
  }
}

// True when 'base' has a shape this step but no storage yet: a member of a
// group still to be carved, or a value whose entry waits for one.
bool baseUncarved(const ExecutionState& state, nativert::ValueId base) {
  for (const auto& alias : tDeferredAliases) {
    if (alias.output == base) {
      return true;
    }
  }
  for (const auto& view : tDeferredViews) {
    if (view.output->valueId == base) {
      return true;
    }
  }
  for (const auto& deferred : tDeferredConcats) {
    const auto& concat = *deferred.concat;
    if (concat.resultId == base) {
      return true;
    }
    const auto* operandIds = deferred.program->code.idsOf(concat.operandIds);
    const auto* operandFlags =
        deferred.program->code.idsOf(concat.operandFlags);
    for (int32_t i = 0; i < concat.operandIds.size; ++i) {
      if (operandIds[i] == base &&
          (operandFlags[i] & ConcatRec::kInPlace) != 0) {
        return true;
      }
    }
  }
  const auto* collector = currentAllocCollector();
  // A slot may be empty, so a carved base can still have a null storage.
  return collector != nullptr && collector->awaitsCarve(base) &&
      state.valueTable.valid(base) && state.valueTable.isTensor(base) &&
      state.valueTable.tensor(base).storage == nullptr;
}

// Runs 'desc''s view node, or its recipe, over the frame.
void runViewOverFrame(const OutputDesc& desc, ExecutionState& state) {
  auto& frame = *state.frame;
  if (runViewRecipe(desc, frame)) {
    return;
  }
  auto it = state.kernelMap->find(desc.viewNode);
  TORCH_CHECK(
      it != state.kernelMap->end(),
      "No kernel for view node ",
      desc.viewNode->target());
  ScopedSetupPhase viewPhase(SetupPhase::kView);
  executeNode(desc.viewNode, it->second, frame);
}

// Clears the frame entries of 'ids' that a deferral left as shape-only (meta)
// tensors.
void clearShapeOnlyEntries(
    nativert::ExecutionFrame& frame,
    std::initializer_list<int32_t> ids) {
  for (const auto id : ids) {
    const auto& value = frame.getIValue(id);
    if (value.isTensor() && value.toTensor().is_meta()) {
      writeFrame(frame, id, c10::IValue());
    }
  }
}

// Gives a kView output its entry. With 'allowDefer', a view whose base is not
// carved yet gets only its shape -- what the rest of the sizing reads -- and is
// run again by finishDeferredViews.
void runBoundView(
    const OutputOp& output,
    StepProgram& program,
    ExecutionState& state,
    bool allowDefer) {
  auto& frame = *state.frame;
  const auto& desc = *program.outputDescs.at(output.desc);
  if (output.derive >= 0) {
    const auto& rec = program.code.viewDerives[output.derive];
    const auto op = program.code.viewOf(rec);
    const auto base = op.operands[0];
    if (allowDefer && baseUncarved(state, base)) {
      if (!runDerive(op, state.valueTable)) {
        // The frame view over a shape-only base gives the same shape.
        const auto& t = state.valueTable.tensor(base);
        int64_t dims[kMaxDims];
        std::copy(t.dims, t.dims + t.rank, dims);
        writeFrame(
            frame,
            base,
            at::empty(
                c10::IntArrayRef(dims, t.rank),
                at::TensorOptions()
                    .dtype(static_cast<c10::ScalarType>(t.elementType))
                    .device(at::kMeta)));
        writeIntListOperands(state, op.operands);
        runViewOverFrame(desc, state);
      }
      tDeferredViews.push_back({&output, &program});
      return;
    }
    if (!allowDefer) {
      clearShapeOnlyEntries(frame, {base, op.output});
    }
    if (runDerive(op, state.valueTable)) {
      takeOwner(state, op.output, base);
      if (rec.materialize || !state.tableOnlyActive) {
        materializeEntry(state, op.output, base);
      }
      return;
    }
    if (!frame.getIValue(base).isTensor() && state.valueTable.valid(base) &&
        state.valueTable.isTensor(base)) {
      materializeEntry(state, base, base);
    }
    writeIntListOperands(state, op.operands);
  }
  runViewOverFrame(desc, state);
}

// Gives in-place output 'alias' its self's entry when the self has storage in
// the table, and makes the frame's view of it only for a host reader. False
// when the self is not in the table.
bool aliasOverTable(
    ExecutionState& state,
    nativert::ValueId alias,
    nativert::ValueId self) {
  if (!state.tableOnlyActive || !state.valueTable.valid(self) ||
      !state.valueTable.isTensor(self) ||
      state.valueTable.tensor(self).storage == nullptr) {
    return false;
  }
  // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
  state.valueEntries[alias] = state.valueEntries[self];
  state.valueEntries[alias].generation = state.tableGeneration;
  takeOwner(state, alias, self);
  if (!isTableOnly(state, alias)) {
    materializeEntry(state, alias, self);
  }
  return true;
}

// Makes the deferred in-place outputs aliases of their selves once those are
// carved; with 'last', all of them.
void finishDeferredAliases(ExecutionState& state, bool last) {
  auto pending = std::move(tDeferredAliases);
  tDeferredAliases.clear();
  auto& frame = *state.frame;
  for (const auto& deferred : pending) {
    if (!last && baseUncarved(state, deferred.self)) {
      tDeferredAliases.push_back(deferred);
      continue;
    }
    TORCH_CHECK(
        !baseUncarved(state, deferred.self),
        "In-place output %",
        deferred.output,
        " of %",
        deferred.self,
        " outlived its step's carve without storage");
    if (isTableOnly(state, deferred.output)) {
      writeFrame(frame, deferred.output, c10::IValue());
    }
    if (aliasOverTable(state, deferred.output, deferred.self)) {
      continue;
    }
    if (!frame.getIValue(deferred.self).isTensor() &&
        state.valueTable.valid(deferred.self) &&
        state.valueTable.isTensor(deferred.self)) {
      materializeEntry(state, deferred.self, deferred.self);
    }
    writeFrame(
        frame,
        deferred.output,
        frame.getIValue(deferred.self).toTensor().alias());
  }
}

// Derives the deferred views whose bases a carve has placed, in sizing order.
// With 'last', every one is derived: its base has been carved or allocated by
// now.
void finishDeferredViews(ExecutionState& state, bool last) {
  auto pending = std::move(tDeferredViews);
  tDeferredViews.clear();
  // A throw puts the views not derived yet back, so that dropDeferredViews
  // clears their placeholders.
  size_t next = 0;
  auto putBack = folly::makeGuard([&] {
    tDeferredViews.insert(
        tDeferredViews.end(),
        pending.begin() + static_cast<std::ptrdiff_t>(next),
        pending.end());
  });
  for (; next < pending.size(); ++next) {
    const auto& deferred = pending[next];
    const auto base =
        deferred.program->code
            .viewOf(deferred.program->code.viewDerives[deferred.output->derive])
            .operands[0];
    if (!last && baseUncarved(state, base)) {
      tDeferredViews.push_back(deferred);
      continue;
    }
    TORCH_CHECK(
        !last || !baseUncarved(state, base),
        "View %",
        deferred.output->valueId,
        " of %",
        base,
        " outlived its step's carve without storage");
    runBoundView(
        *deferred.output, *deferred.program, state, /*allowDefer=*/false);
  }
  putBack.dismiss();
}

// Forgets the deferred views of a step whose setup threw, and clears the
// shape-only entries their deferral left in the frame.
void dropDeferredViews(ExecutionState& state) {
  for (const auto& deferred : tDeferredViews) {
    const auto& code = deferred.program->code;
    const auto op = code.viewOf(code.viewDerives[deferred.output->derive]);
    clearShapeOnlyEntries(*state.frame, {op.operands[0], op.output});
  }
  tDeferredViews.clear();
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
  if (output.aliasOf >= 0 && baseUncarved(state, output.aliasOf)) {
    // The sizing after this reads only the shape, which the self has.
    const auto& t = state.valueTable.tensor(output.aliasOf);
    int64_t dims[kMaxDims];
    std::copy(t.dims, t.dims + t.rank, dims);
    writeFrame(
        frame,
        output.valueId,
        at::empty(
            c10::IntArrayRef(dims, t.rank),
            at::TensorOptions()
                .dtype(static_cast<c10::ScalarType>(t.elementType))
                .device(at::kMeta)));
    if (isTableOnly(state, output.valueId)) {
      // The shape is what sizing reads; the carve gives the storage.
      // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
      state.valueEntries[output.valueId] = state.valueEntries[output.aliasOf];
      state.valueEntries[output.valueId].generation = state.tableGeneration;
    }
    tDeferredAliases.push_back({output.valueId, output.aliasOf});
    return;
  }
  if (output.aliasOf >= 0 &&
      aliasOverTable(state, output.valueId, output.aliasOf)) {
    return;
  }
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
    case BoundOutput::kView:
      runBoundView(output, program, state, /*allowDefer=*/true);
      break;
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
  if (launch.recount && !count.constant) {
    --count.generation;
    evaluateSize(count, program, state);
    data.numElements = static_cast<int64_t>(firstShape(program, count).dims[0]);
  }
  if (data.numElements > 0 && countsOutputs(data.sizeExpr)) {
    const auto& table = state.valueTable;
    for (auto id : data.actualOutputs) {
      if (id >= 0 && table.valid(id) && table.isTensor(id)) {
        data.numElements =
            std::max<int64_t>(data.numElements, table.tensor(id).numEl);
      }
    }
  }
  checkUnreadyOperands(launch, code, data, state);
}

void waitForFetches(
    ExecutionState& state,
    const std::vector<StepVectors::DeviceFetch>& fetches) {
  for (const auto& fetch : fetches) {
    if (state.valueTable.valid(fetch.id)) {
      continue;
    }
    const auto& event =
        state.stepVectors.at(fetch.node).at(fetch.step).fetchDone;
    if (event != nullptr && !event->query()) {
      VLOG(1) << "Device alloc: wait for %" << fetch.id << " from step "
              << fetch.node << "." << fetch.step;
      event->wait();
      ++state.numFetchWaits;
    }
  }
}

namespace {

// Whether a step launches, runs a standalone or derives a view.
bool stepHasWork(const StepVectors& sv) {
  return !sv.kernels.empty() || !sv.standalones.empty() ||
      !sv.shortcutStandalones.empty() || sv.program->code.numDerives > 0;
}

// Writes the table entries of the path values the step reads from the
// descriptors their producers' setups wrote, after the copies back land.
void fetchDeviceValues(ExecutionState& state, const StepVectors& sv) {
  auto& table = state.valueTable;
  bool any = false;
  for (const auto& fetch : sv.deviceFetches) {
    any = any || !table.valid(fetch.id) || fetch.frame;
    // The setup placed the view; the host never took its owner.
    // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
    if (fetch.base >= 0 && state.valueCold[fetch.id].owner < 0) {
      takeOwner(state, fetch.id, fetch.base);
    }
  }
  if (!any) {
    return;
  }
  // The entries come from setups that may have failed, also those the host
  // computes from the capacity layout: check the producers first. This waits
  // only for a setup before them that can fail.
  int32_t through = -1;
  for (const auto& fetch : sv.deviceFetches) {
    if (!table.valid(fetch.id) || fetch.frame) {
      through = std::max(
          through, state.stepVectors[fetch.node][fetch.step].stepOrdinal);
    }
  }
  checkSetupFailure(state, through);
  // Under hostFetch, the entries the host computes from the capacity layout,
  // which need no wait. In check mode they are compared with the fetched
  // descriptors below.
  const auto hostFetch = WaveConfig::get().hostFetch;
  std::vector<std::pair<int32_t, TableTensor>> computed;
  if (hostFetch != 0) {
    auto& hostSymbols = state.hostSymbols;
    const auto* addresses = state.addressPinned.data_ptr<int64_t>();
    std::vector<std::string> names;
    for (const auto& fetch : sv.deviceFetches) {
      if (fetch.recipe < 0 || table.valid(fetch.id)) {
        continue;
      }
      const auto& recipe = state.hostFetchRecipes[fetch.recipe];
      names.clear();
      recipe.address.collectSymbols(names);
      for (const auto& name : names) {
        if (name[0] == 'a') {
          hostSymbols[name] = addresses[std::stoll(name.substr(1))];
        }
      }
      TableTensor t;
      const auto address = recipe.address.evaluate(hostSymbols);
      bool known = address.has_value();
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      t.storage = reinterpret_cast<void*>(address.value_or(0));
      t.rank = static_cast<int8_t>(recipe.dims.size());
      t.elementSize = recipe.elementSize;
      t.elementType = recipe.elementType;
      int64_t numEl = 1;
      for (size_t d = 0; known && d < recipe.dims.size(); ++d) {
        const auto dim = recipe.dims[d].evaluate(hostSymbols);
        const auto stride = recipe.strides[d].evaluate(hostSymbols);
        known = dim.has_value() && stride.has_value();
        if (known) {
          t.dims[d] = static_cast<int32_t>(*dim);
          t.strides[d] = static_cast<int32_t>(*stride);
          numEl *= *dim;
        }
      }
      t.numEl = static_cast<uint32_t>(numEl);
      if (!known) {
        continue;
      }
      if (hostFetch == 2) {
        computed.emplace_back(fetch.id, t);
        continue;
      }
      auto& entry = state.valueEntries[fetch.id];
      entry.t = t;
      entry.kind = ValueEntry::kTensor;
      entry.flags = 0;
      entry.generation = state.tableGeneration;
      ++state.numHostFetches;
      if (fetch.frame) {
        materializeEntry(state, fetch.id, fetch.id);
      }
    }
  }
  waitForFetches(state, sv.deviceFetches);
  const auto& symbols = state.hostSymbols;
  for (const auto& fetch : sv.deviceFetches) {
    if (table.valid(fetch.id)) {
      if (fetch.frame) {
        materializeEntry(state, fetch.id, fetch.id);
      }
      continue;
    }
    const auto& from = state.stepVectors[fetch.node][fetch.step];
    auto& entry = state.valueEntries[fetch.id];
    std::memcpy(&entry.t, from.pinnedBase + fetch.offset, sizeof(TableTensor));
    auto hostShape = state.fetchHostShapes.find(fetch.id);
    if (hostShape != state.fetchHostShapes.end()) {
      const auto& [dims, strides] = hostShape->second;
      TORCH_CHECK(
          static_cast<int32_t>(dims.size()) == entry.t.rank,
          "Device alloc: fetched %",
          fetch.id,
          " has rank ",
          static_cast<int32_t>(entry.t.rank),
          ", its derived shape ",
          dims.size());
      int64_t numEl = 1;
      for (size_t d = 0; d < dims.size(); ++d) {
        const auto dim = dims[d].evaluate(symbols);
        const auto stride = strides[d].evaluate(symbols);
        TORCH_CHECK(
            dim.has_value() && stride.has_value(),
            "Device alloc: no capacity shape for fetched %",
            fetch.id);
        entry.t.dims[d] = static_cast<int32_t>(*dim);
        entry.t.strides[d] = static_cast<int32_t>(*stride);
        numEl *= *dim;
      }
      entry.t.numEl = static_cast<uint32_t>(numEl);
    }
    entry.kind = ValueEntry::kTensor;
    entry.flags = 0;
    entry.generation = state.tableGeneration;
    if (fetch.frame) {
      materializeEntry(state, fetch.id, fetch.id);
    }
  }
  for (const auto& [id, t] : computed) {
    const auto& fetched = state.valueEntries[id].t;
    bool same = fetched.storage == t.storage && fetched.rank == t.rank &&
        fetched.elementType == t.elementType;
    for (int32_t d = 0; same && d < t.rank; ++d) {
      same = fetched.dims[d] == t.dims[d] &&
          (fetched.dims[d] == 1 || fetched.strides[d] == t.strides[d]);
    }
    ++state.numHostFetches;
    if (!same) {
      ++state.numHostFetchMismatches;
      LOG_EVERY_N(WARNING, 1)
          << "Host fetch of %" << id << " differs: host " << t.storage
          << " rank " << int(t.rank) << " dims " << t.dims[0] << ","
          << t.dims[1] << " strides " << t.strides[0] << "," << t.strides[1]
          << "; device " << fetched.storage << " rank " << int(fetched.rank)
          << " dims " << fetched.dims[0] << "," << fetched.dims[1]
          << " strides " << fetched.strides[0] << "," << fetched.strides[1];
    }
  }
}

} // namespace

void CompositeInvocation::runDeviceAllocStep(
    int32_t stepIdx,
    StepVectors& sv,
    ExecutionState& state) {
  const auto* plan = allocGroupPlan();
  const bool hasGroups = plan != nullptr &&
      static_cast<size_t>(stepIdx) < plan->groupsByStep.size();
  // A group with no recorded demand yet: run the setup once for its demands
  // and wait for them.
  bool probe = false;
  if (hasGroups) {
    for (auto g : plan->groupsByStep[stepIdx]) {
      probe = probe ||
          state.recordedDemand.at(state.groupBaseSlot.at(&plan->groups[g])) < 0;
    }
  }
  if (probe && sv.program->symbolFixups != nullptr) {
    state.stream->hostToDeviceAsync(
        sv.deviceBase, sv.pinnedBase, sv.blockInfoOffset);
    copyAddresses(state, *state.stream);
    state.waveGraph->deviceSymbols()->runFixups(
        state,
        *sv.program->symbolFixups,
        sv.deviceBase,
        state.stream.get(),
        /*checkFailure=*/false);
    state.stream->wait();
    ++state.numDemandProbes;
    const auto demands = state.demandTable.cpu();
    const auto* demand = demands.data_ptr<int64_t>();
    for (auto g : plan->groupsByStep[stepIdx]) {
      const auto slot = state.groupBaseSlot.at(&plan->groups[g]);
      auto& recorded = state.recordedDemand.at(slot);
      if (recorded < 0) {
        recorded = demand[slot];
      }
    }
  }
  sv.setupCanFail = false;
  if (hasGroups) {
    for (auto g : plan->groupsByStep[stepIdx]) {
      const auto slot = state.groupBaseSlot.at(&plan->groups[g]);
      sv.setupCanFail = sv.setupCanFail ||
          static_cast<size_t>(slot) >= state.hostDemand.size() ||
          state.hostDemand[slot] < 0;
      allocateDeviceGroup(state, plan->groups[g], sequenceNumber_, stepIdx);
    }
  }
  ++gDeviceAllocStepRuns;
  // The device grid counts the elements. The counts of the last execution
  // the host sized bound its blocks, as on the host path; more blocks slow a
  // cooperative kernel's grid barriers. None is marked empty.
  for (auto& data : sv.kernels) {
    data.numElements = std::max<int64_t>(data.numElements, 1);
  }
  sv.allocUs = 0;
  sv.allocCallUs = 0;
  sv.fillUs = 0;
  sv.allocGroups = 0;
  sv.allocGroupTensors = 0;
  sv.gatherUs = 0;
  sv.refCheckUs = 0;
  sv.elidedCloneBytes = 0;
}

bool CompositeInvocation::prepareStepEarly(
    ExecutionState& state,
    int32_t stepIdx) {
  if (static_cast<size_t>(sequenceNumber_) >= state.stepVectors.size() ||
      static_cast<size_t>(stepIdx) >=
          state.stepVectors[sequenceNumber_].size()) {
    return false;
  }
  auto& sv = getStepVectors(state.stepVectors, sequenceNumber_, stepIdx);
  if (!sv.deviceAllocStep || sv.program == nullptr ||
      sv.program->symbolFixups == nullptr ||
      sv.program->code.numLaunches == 0 || sv.preparedEarly) {
    return false;
  }
  const int64_t totalAllocBytes = sv.program->totalPinnedBytes +
      static_cast<int64_t>(sv.blockCapacity) *
          static_cast<int64_t>(sizeof(DebugInfo));
  if (totalAllocBytes > sv.bufferBytes) {
    return false;
  }
  // A group without a recorded demand waits for a probe of the setup.
  const auto* plan = allocGroupPlan();
  if (plan != nullptr &&
      static_cast<size_t>(stepIdx) < plan->groupsByStep.size()) {
    for (auto g : plan->groupsByStep[stepIdx]) {
      if (state.recordedDemand.at(state.groupBaseSlot.at(&plan->groups[g])) <
          0) {
        return false;
      }
    }
  }
  runFills(*sv.program, sv, state, *state.frame, sv.pinnedBase);
  runDeviceAllocStep(stepIdx, sv, state);
  sv.preparedEarly = true;
  return layDeviceGrid(sv, sv.pinnedBase);
}

void CompositeInvocation::prepareSetupBatch(
    ExecutionState& state,
    int32_t stepIdx) {
  state.activeSetupBatch = -1;
  state.numActiveSetupSteps = 0;
  if (!WaveConfig::get().setupBatches || WaveConfig::get().setupStream) {
    return;
  }
  auto it = state.setupBatchAt.find(
      (static_cast<int64_t>(sequenceNumber_) << 32) | stepIdx);
  auto& head = getStepVectors(state.stepVectors, sequenceNumber_, stepIdx);
  if (it == state.setupBatchAt.end() || head.program == nullptr ||
      head.program->symbolFixups == nullptr ||
      !layDeviceGrid(head, head.pinnedBase)) {
    return;
  }
  const auto& batch = state.setupBatches.at(it->second);
  int32_t numSteps = 1;
  for (; numSteps < static_cast<int32_t>(batch.steps.size()); ++numSteps) {
    const auto [node, step] = batch.steps[numSteps];
    CompositeInvocation* invocation = nullptr;
    for (const auto& compiled : state.waveGraph->nodes()) {
      auto* kernels = compiled->kernels();
      if (kernels != nullptr && kernels->sequenceNumber() == node) {
        invocation = kernels;
        break;
      }
    }
    if (invocation == nullptr || !invocation->prepareStepEarly(state, step)) {
      break;
    }
  }
  if (numSteps > 1) {
    state.activeSetupBatch = it->second;
    state.numActiveSetupSteps = numSteps;
  }
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
    if (state.deviceAllocActive && sv.deviceAllocStep) {
      // A step of derives alone sets nothing up: the setups of their readers
      // place them.
      const bool launches = program.code.numLaunches > 0;
      if (!sv.preparedEarly && launches) {
        runFills(program, sv, state, frame, pinnedBase);
        runDeviceAllocStep(stepIdx, sv, state);
      }
      sv.preparedEarly = false;
      if (launches) {
        prepareSetupBatch(state, stepIdx);
      }
      ownDerivedViews(sv, program, state);
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
      if (sv.fetchEnd > sv.fetchBegin) {
        if (sv.fetchDone == nullptr) {
          sv.fetchDone = std::make_shared<facebook::velox::wave::Event>(
              /*withTime=*/false);
        }
        issueLogged(state, [&state, &sv, pinnedBase, deviceBase]() {
          state.stream->deviceToHostAsync(
              pinnedBase + sv.fetchBegin,
              deviceBase + sv.fetchBegin,
              sv.fetchEnd - sv.fetchBegin);
          sv.fetchDone->record(*state.stream);
        });
      }
      // The counts the device grid read, which bound the next launch.
      if (sv.deviceGrid != nullptr && sv.onDeviceGrid) {
        const auto numLaunches = sv.deviceGrid->numLaunches;
        if (!sv.gridCountsPinned.defined() ||
            sv.gridCountsPinned.numel() < numLaunches) {
          sv.gridCountsPinned = at::empty(
              {numLaunches},
              at::TensorOptions().dtype(at::kLong).pinned_memory(true));
        }
        issueLogged(state, [&state, &sv, numLaunches]() {
          state.stream->deviceToHostAsync(
              sv.gridCountsPinned.data_ptr<int64_t>(),
              sv.deviceGrid->device.data_ptr<int32_t>() +
                  sv.deviceGrid->longsOffset,
              numLaunches * sizeof(int64_t));
        });
        sv.gridCountsPending = true;
      }
      continue;
    }
    if (state.deviceAllocActive && !sv.deviceFetches.empty()) {
      fetchDeviceValues(state, sv);
    }
    if (state.deviceAllocActive && !sv.hostDerives.empty()) {
      folly::F14FastSet<nativert::ValueId> reads;
      for (const auto& derive : sv.hostDerives) {
        reads.insert(derive.reads.begin(), derive.reads.end());
      }
      resolvePendingReturns(state, neededPendingStep(state, reads));
      for (const auto& derive : sv.hostDerives) {
        // An earlier step's derive made it this execution: its operands may
        // since have been released.
        const auto& code = derive.program->code;
        const auto output = code.derives[derive.derive].output;
        if (output >= 0 && state.valueTable.valid(output)) {
          if (derive.frame && !state.frame->getIValue(output).isTensor()) {
            materializeEntry(state, output, output);
          }
          continue;
        }
        runDeriveRec(*derive.program, derive.derive, derive.frame, state);
      }
    }
    {
      auto& collector = *program.collector;
      AllocGroupCollector::Scope collecting(collector);
      // A throw before the last finish must not leave entries for the next
      // execution on this thread.
      auto clearDeferred = folly::makeGuard([&state] {
        tDeferredConcats.clear();
        tDeferredAliases.clear();
        dropDeferredViews(state);
      });
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
      finishDeferredConcats(state, /*last=*/false);
      finishDeferredAliases(state, /*last=*/false);
      finishDeferredViews(state, /*last=*/false);
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
          ++state.numReturnWaits;
          state.returnWaitUs += waitUs;
        }
        for (int32_t l = program.numIndependent; l < numLaunches; ++l) {
          runBoundLaunch(launches[l], program, sv, state);
        }
      }
      materializeStepGroups(
          state, collector, sv, /*afterWait=*/true, run.doTiming);
      finishDeferredConcats(state, /*last=*/true);
      finishDeferredAliases(state, /*last=*/true);
      finishDeferredViews(state, /*last=*/true);
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
      runShapeChecks(program, state.valueTable);
      if (run.doTiming) {
        sv.fillUs += static_cast<int64_t>(
            static_cast<double>(folly::hardware_timestamp() - tFill) /
            tscTicksPerMicro());
      }
    }
    sv.refCheckUs = 0;
    sv.elidedCloneBytes = 0;
    if (!stepHasWork(sv)) {
      // A step emptied by host plumbing (an unpack folded into the shortcut
      // before it) can precede steps that still launch.
      bool laterWork = false;
      for (int32_t later = stepIdx + 1; later < numSteps && !laterWork;
           ++later) {
        laterWork = stepHasWork(
            getStepVectors(state.stepVectors, sequenceNumber_, later));
      }
      if (laterWork) {
        continue;
      }
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
