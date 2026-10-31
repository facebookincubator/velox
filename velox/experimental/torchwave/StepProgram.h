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

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>

#include "velox/experimental/torchwave/Cat.h"
#include "velox/experimental/torchwave/CompiledOp.h"
#include "velox/experimental/torchwave/Derive.h"
#include "velox/experimental/torchwave/Reserve.h"

// The step program of the bound execution path: what one step of one node
// does each execution, resolved before the first run. Plain data;
// StepRunner.cpp executes it.

namespace torch::wave {

class AllocGroupCollector;
struct OutputDesc;
struct ValueTypes;

/// One sizing call as bind builds it: a ReserveFn over bound operands, or a
/// concat layout. Packed into a SizeOp.
struct BoundSize {
  ReserveFn fn{nullptr};
  ReserveBinding args;
  int32_t blockSize{0};
  /// How many Shapes the call produces, one per output tensor.
  int32_t numShapes{1};
  /// The output whose reserve this is, for error messages; null for a
  /// built-in.
  const LaunchData* data{nullptr};
  int32_t outputIdx{-1};
  /// Index into StepProgram::concats of the concat this Size lays out, or -1.
  /// Its one Shape is the result's.
  int32_t concat{-1};
};

/// The layout of a fused cat or stack over the table: the result's entry and
/// the bands of the operands its kernel computes in place.
struct BoundConcat {
  ConcatSpec spec;
  c10::ScalarType dtype{c10::ScalarType::Float};
  int32_t resultId{-1};
  /// Per operand in join order: its value, the Size giving its shape (-1 for
  /// an empty operand), whether the kernel writes it in place (it gets a band
  /// here) and whether the kernel sets its extent (a band at the offset the
  /// kernel patches).
  std::vector<int32_t> operandIds;
  std::vector<int32_t> operandSizes;
  std::vector<uint8_t> inPlace;
  std::vector<uint8_t> shapeOnDevice;
};

/// How one output of a kernel launch gets its shape and memory.
struct BoundOutput {
  enum Kind : uint8_t {
    /// Shaped by 'size', memory from the step's allocation group or its own.
    kTensor,
    /// One per element of a TensorList output, shaped by Shape 'element' of
    /// 'size'.
    kListElement,
    /// A shape and no memory.
    kShapeOnly,
    /// An in-place output with nothing to fall back to: a view of 'aliasOf'
    /// when that is a tensor, else nothing.
    kAlias,
    /// A host-side view of another value, through the frame.
    kView,
    /// A concat result and its operands' bands: over the table when 'size'
    /// is a concat Size, else through the frame.
    kConcat,
  };

  Kind kind{kTensor};
  int32_t valueId{-1};
  int32_t size{-1};
  int32_t element{0};
  /// An in-place output's self: when it is a tensor the output is a view of it
  /// and the kind's own sizing does not run.
  int32_t aliasOf{-1};
  /// The descriptor of a kView or kConcat output.
  const OutputDesc* desc{nullptr};
  /// For a kView output: index into StepProgram::viewDerives of the DeriveOp
  /// that computes it over the table, or -1 to run the view over the frame.
  int32_t derive{-1};
};

/// One write into a launch's parameter block.
struct FillOp {
  enum Kind : uint8_t {
    /// A tensor input or output; None empties it and the launch.
    kTensor,
    /// A tensor output that carries a shape and no data.
    kShapeOnly,
    /// A scalar input, from the frame.
    kScalar,
    /// A scalar output: zeroed for the kernel to write.
    kZero64,
    /// An op barrier counter: zeroed every execution.
    kZero32,
    /// A tensor the kernel builds itself: status reset every execution.
    kUninited,
  };

  Kind kind{kTensor};
  /// From the start of the step's pinned parameter region.
  int32_t offset{0};
  int32_t valueId{-1};
  /// The launch whose element count a None tensor zeroes.
  int32_t kernel{-1};
};

/// One kernel launch of the step.
struct BoundLaunch {
  /// Index into StepVectors::kernels.
  int32_t kernel{0};
  /// The Size whose first Shape gives the launch's element count: a 1-d count,
  /// or with 'countFromReserve' the shape of output 0, whose element count it
  /// is.
  int32_t countSize{-1};
  bool countFromReserve{false};
  /// Outputs when the launch is sized by its largest input and one exists,
  /// and the outputs otherwise.
  int32_t largestShapeSize{-1};
  std::vector<BoundOutput> largestOutputs;
  std::vector<BoundOutput> outputs;
  /// The tensor inputs the launch's op does not produce itself and its tensor
  /// outputs: when one is absent from the table the launch runs on no blocks,
  /// unless an allocation group carves the output this step.
  std::vector<int32_t> externalInputs;
  std::vector<int32_t> tensorOutputs;
  /// The op's inputs and the element count of its largest statically shaped
  /// input: the cooperative grid sizes an emptied launch by these.
  std::vector<int32_t> allInputs;
  int64_t staticNumElements{0};
};

/// A BoundConcat packed into a StepCode.
struct ConcatRec {
  enum OperandFlags : int32_t { kInPlace = 1, kShapeOnDevice = 2 };

  ConcatSpec spec;
  c10::ScalarType dtype{c10::ScalarType::Float};
  int32_t resultId{-1};
  /// Into StepCode::ids, parallel: each operand's value, its Size (-1 for an
  /// empty operand) and its OperandFlags.
  Span operandIds;
  Span operandSizes;
  Span operandFlags;
};

/// One sizing call of a StepCode. Its operands and results are spans into the
/// code's arrays.
struct SizeOp {
  ReserveFn fn{nullptr};
  /// The table generation the shapes were computed in. A launch's count and its
  /// first output often share one Size; this keeps it to one call.
  uint32_t generation{0};
  /// Index into StepCode::concats of the concat this Size lays out, or -1.
  int32_t concat{-1};
  /// Into StepCode::spans, each a tensor group of StepCode::ids.
  Span groups;
  Span scalars;
  /// Into StepCode::spans, each a list in StepCode::scalars.
  Span intLists;
  /// Into StepCode::ids.
  Span hostIds;
  /// Into StepCode::shapes: the result, one Shape per output tensor.
  Span shapes;
  int32_t blockSize{0};
  /// Index into StepProgram::sizes, for messages.
  int32_t source{-1};
};

/// How one output of a kernel launch gets its shape and memory: a packed
/// BoundOutput.
struct OutputOp {
  enum Flags : uint8_t {
    /// The value lives in the table alone when table-only values are active.
    kTableOnly = 1,
    /// The value has a type, so 'dtype' is its element type.
    kTyped = 2,
  };

  BoundOutput::Kind kind{BoundOutput::kTensor};
  uint8_t flags{0};
  c10::ScalarType dtype{c10::ScalarType::Float};
  int32_t valueId{-1};
  int32_t size{-1};
  int32_t element{0};
  int32_t aliasOf{-1};
  int32_t derive{-1};
  /// Index into StepProgram::outputDescs of a kView or kConcat descriptor.
  int32_t desc{-1};
};

/// One kernel launch of a StepCode: a packed BoundLaunch.
struct LaunchOp {
  int32_t kernel{0};
  int32_t countSize{-1};
  int32_t largestShapeSize{-1};
  bool countFromReserve{false};
  /// Into StepCode::outputs.
  Span outputs;
  Span largestOutputs;
  /// Into StepCode::ids.
  Span externalInputs;
  Span tensorOutputs;
  Span allInputs;
  int64_t staticNumElements{0};
};

/// The part of a step program that runs every execution, in one allocation:
/// fixed-size records whose operands are spans into the same block. Built from
/// the StepProgram once bind is done.
struct StepCode {
  StepCode() = default;
  ~StepCode() = default;
  StepCode(const StepCode&) = delete;
  StepCode& operator=(const StepCode&) = delete;
  StepCode(StepCode&&) = delete;
  StepCode& operator=(StepCode&&) = delete;

  SizeOp* sizes{nullptr};
  LaunchOp* launches{nullptr};
  OutputOp* outputs{nullptr};
  FillOp* fills{nullptr};
  DeriveRec* derives{nullptr};
  DeriveRec* viewDerives{nullptr};
  ConcatRec* concats{nullptr};
  Span* spans{nullptr};
  ScalarRef* scalars{nullptr};
  int32_t* ids{nullptr};
  int64_t* consts{nullptr};
  /// Written by the sizing calls every execution.
  Shape* shapes{nullptr};
  int32_t numSizes{0};
  int32_t numLaunches{0};
  int32_t numFills{0};
  int32_t numDerives{0};
  /// The steps whose transfers bring back what the dependent launches read,
  /// and what the step's standalones and derives read, as sorted
  /// stepKey()s into 'consts'.
  Span waitProducers;
  Span readProducers;
  std::unique_ptr<uint8_t[]> storage;

  /// The operands of 'op' over 'table'.
  ReserveArgs argsOf(const SizeOp& op, const ValueTable& table) const {
    return {
        .table = &table,
        .groups = spans + op.groups.begin,
        .numGroups = op.groups.size,
        .ids = ids,
        .scalars = scalars + op.scalars.begin,
        .numScalars = op.scalars.size,
        .intLists = spans + op.intLists.begin,
        .numLists = op.intLists.size,
        .scalarPool = scalars,
        .hostIds = ids + op.hostIds.begin,
        .numHost = op.hostIds.size,
        .blockSize = op.blockSize};
  }

  const int32_t* idsOf(Span span) const {
    return ids + span.begin;
  }

  DeriveView viewOf(const DeriveRec& op) const {
    return {
        .kind = op.kind,
        .output = op.output,
        .operands = {ids + op.operands.begin, op.operands.size},
        .constants = {consts + op.constants.begin, op.constants.size},
        .intList = {consts + op.intList.begin, op.intList.size},
        .outputs = {ids + op.outputs.begin, op.outputs.size},
        .elements = {ids + op.elements.begin, op.elements.size}};
  }
};

/// One step of one node.
struct StepProgram {
  std::vector<BoundSize> sizes;
  /// Launches that read no value a transfer brings back, then the ones that
  /// do. The step waits for its transfers between the two.
  std::vector<BoundLaunch> launches;
  int32_t numIndependent{0};
  /// The invocation op index of each launch in bind order, and of each
  /// standalone and shortcut launch. Used while binding.
  std::vector<int32_t> launchOp;
  std::vector<int32_t> standaloneOps;
  /// What the dependent launches read, for the wait.
  folly::F14FastSet<int32_t> waitReads;
  std::vector<FillOp> fills;
  /// The step's metadata shortcuts that run over the table, and the launches
  /// they stand for: when an operand is not in the table, the shortcut runs
  /// over the frame instead.
  std::vector<DeriveOp> derives;
  std::vector<LaunchData> derivedShortcuts;
  /// The views of kView outputs, run where the output is placed.
  std::vector<DeriveOp> viewDerives;
  std::vector<BoundConcat> concats;
  int32_t returnBegin{-1};
  int32_t returnEnd{-1};
  int64_t totalPinnedBytes{0};
  /// The descriptors kView and kConcat outputs name.
  std::vector<const OutputDesc*> outputDescs;
  /// Collects the step's allocation groups. Built at bind, installed for each
  /// execution of the step.
  std::shared_ptr<AllocGroupCollector> collector;
  /// The sizes, launches and fills above, packed. 'sizes' and 'launches' are
  /// kept for messages and the bind passes that follow; the run reads 'code'.
  StepCode code;
};

/// Names step 'stepIdx' of node 'sequenceNumber'.
inline int64_t stepKey(int32_t sequenceNumber, int32_t stepIdx) {
  return (static_cast<int64_t>(sequenceNumber) << 32) |
      static_cast<uint32_t>(stepIdx);
}

/// Packs what 'program' runs every execution into 'program.code'. 'tableOnly'
/// and 'types' give each output's flags and element type; 'producersOf' maps
/// a value a transfer brings back to the stepKey()s of the steps that send it,
/// and 'readIds' is what the step's standalones and derives read.
void packStepCode(
    StepProgram& program,
    const std::vector<uint8_t>& tableOnly,
    const ValueTypes& types,
    const folly::F14FastMap<int32_t, std::vector<int64_t>>& producersOf,
    const folly::F14FastSet<int32_t>& readIds);

} // namespace torch::wave
