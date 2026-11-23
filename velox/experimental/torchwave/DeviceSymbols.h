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

#include <ATen/ATen.h>
#include <folly/CppAttributes.h>
#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <torch/nativert/graph/Graph.h>
#include "velox/experimental/torchwave/SymShape.h"

namespace facebook::velox::wave {
class Stream;
} // namespace facebook::velox::wave

namespace torch::nativert {
class ExecutionFrame;
} // namespace torch::nativert

namespace torch::wave {

namespace nativert = torch::nativert;

struct AllocGroup;
struct ExecutionState;
struct StepProgram;
class WaveGraph;

/// The parameter fixups of one step: for each tensor the step's kernels read
/// whose dims depend on a symbol the device holds, programs that compute its
/// real dims and strides; for each such symbol passed as a scalar, its slot.
struct SymbolFixups {
  /// Packed records, 12 words each: slot offset in the step's parameter
  /// region, kind (0 tensor, 1 scalar, 2 shape only, 3 band start only), rank,
  /// the code offset of a concat band's start program or -1, then the code
  /// offsets of the dim and stride programs.
  std::vector<int32_t> records;
  /// Postfix programs over constants and symbol values.
  std::vector<int64_t> code;
  int32_t numRecords{0};
  at::Tensor deviceRecords;
  at::Tensor deviceCode;
};

/// What the fixup kernel of one step reads besides the step's parameters.
struct FixupArgs {
  int32_t* records{nullptr};
  int64_t* code{nullptr};
  int32_t numRecords{0};
  /// The symbols' source addresses, then their capacities.
  int64_t* symbols{nullptr};
  int32_t numSymbols{0};
  /// Words of 'code'.
  int32_t numCode{0};
};

/// Lets a step that would wait for an integer the device returns -- an
/// item() whose value sizes what follows -- run without waiting.
///
/// The host sizes everything that depends on the value from a capacity: the
/// largest value seen in earlier executions. The device keeps the real value
/// where the producing kernel left it. Before each step that reads a tensor
/// shaped by it, a small kernel rewrites that tensor's dims and strides in the
/// step's parameters from the export's symbolic expressions, so the step's
/// kernels loop over the real extent of buffers sized for the capacity.
///
/// A value qualifies when its expression is a single data-dependent symbol
/// and everything shaped by it is read only by fused kernels: no eager op, no
/// host read, no graph output, and every expression involved uses only
/// qualifying symbols. Executions run the ordinary path until a value has been
/// observed once.
///
/// A real value above its capacity is reported at the end of the execution;
/// the kernels see empty tensors for that execution. Rerunning it is not
/// implemented yet.
class DeviceSymbols {
 public:
  DeviceSymbols(std::shared_ptr<SymShapes> shapes, WaveGraph& waveGraph);

  /// True once some candidate value has been observed, so an execution can
  /// run on capacities.
  bool ready() const;

  /// Picks the values 'state' runs on capacities and removes them from
  /// 'returned', so no step waits for them. Called while binding a state in
  /// the device-symbols mode.
  void selectSubstitutions(
      ExecutionState& state,
      folly::F14FastSet<int32_t>& returned);

  /// Turns the sources selectSubstitutions recorded into parameter offsets,
  /// once the steps' parameters are laid out.
  void resolveSources(ExecutionState& state);

  /// Builds the fixups of 'program' for the values selectSubstitutions picked.
  void bindFixups(const ExecutionState& state, StepProgram& program);

  /// Puts the capacities in the table, where the sizing reads them.
  void beginExecution(ExecutionState& state);

  /// Records 'value' returned for 'id'. Returns true when 'id' runs on its
  /// capacity, so the table must keep the capacity.
  bool
  observeReturn(ExecutionState& state, nativert::ValueId id, int64_t value);

  /// Records the count a returned masked_select brings back for 'id'.
  /// Returns true when 'id' runs on its bound, so its frame tensor keeps the
  /// reserved dims.
  bool
  observeReturnDims(ExecutionState& state, nativert::ValueId id, int64_t count);

  /// Folds the observed values into the capacities and gives the graph
  /// outputs sized from capacities their real dims. Returns false when a
  /// value exceeded the capacity the execution ran on.
  bool endExecution(ExecutionState& state);

  /// Makes the fixups of a step ready to run on the device.
  FixupArgs prepareFixups(ExecutionState& state, SymbolFixups& fixups);

  /// Launches the fixups of a step on 'stream', after its parameters have
  /// been copied to 'deviceBase'.
  /// With 'checkFailure' false the setup neither checks nor marks the
  /// failure words: a probe for its demands.
  void runFixups(
      ExecutionState& state,
      SymbolFixups& fixups,
      uint8_t* deviceBase,
      facebook::velox::wave::Stream* stream,
      bool checkFailure = true);

  /// Gives the outputs of the bound steps of 'state' the shapes the symbolic
  /// forms of their ReserveFns compute, where the export has none. An integer
  /// operand without an expression, such as a total the device returns, gets
  /// a new symbol named u_<value>. Returns how many shapes it added.
  int32_t inferReserveShapes(const ExecutionState& state);

  /// Logs and clears what checkSizeExprs has tallied over an execution, after
  /// comparing the outputs it deferred with the execution's 'symbols'.
  void reportSizeExprs(
      const folly::F14FastMap<std::string, int64_t>& symbols) const;

  /// Under WaveConfig::deviceSizeCheck: adds to each step's fixups a check
  /// record for every tensor parameter whose size derived from graph-input
  /// dims has all its symbols written to the device by then. Input dims are
  /// read from the tensor parameters of the graph inputs, data-decided
  /// symbols from integer parameters. The setup kernel evaluates the records
  /// and counts the parameters whose dims differ from the host's.
  void bindSizeChecks(ExecutionState& state);

  /// Logs and zeroes the counts the check records added over the previous
  /// execution of 'state'. The streams must be idle.
  void reportSizeChecks(ExecutionState& state) const;

  /// Under WaveConfig::deviceSizingPoison: zeroes the dims the host wrote at
  /// 'params' for the tensor parameters whose every dim 'fixups' writes.
  static void poisonSizedParams(const SymbolFixups& fixups, uint8_t* params);

  /// Copies the values of the input symbols bindSizeChecks placed from
  /// 'frame' to the device. Called at each execution start.
  void writeInputSymbols(
      const nativert::ExecutionFrame& frame,
      ExecutionState& state) const;

  /// True if the export gave the shape of 'valueId'; see hostShaped_.
  bool isHostShaped(int32_t valueId) const;

  /// True if the device sets the shape of 'valueId'; see setOnDevice_.
  bool isSetOnDevice(int32_t valueId) const;

  /// Under WaveConfig::sizeExprsFromInputs: shapes for every value derived
  /// from graph-input dims taken as symbols, kept apart from the export's.
  int32_t deriveShapesFromInputs(const ExecutionState& state);

  /// One pass of deriveShapesFromInputs, starting from the shapes in
  /// 'preset'. Sets 'numAdded' to how many shapes the rules added.
  std::shared_ptr<SymShapes> deriveOnce(
      const ExecutionState& state,
      const folly::F14FastMap<std::string, SymShape>& preset,
      int32_t& numAdded);

  /// Runs the rules of the nodes of the variant graphs, which produce the
  /// values grid generation makes. Returns how many main-graph values it
  /// shaped under another name.
  int32_t shapeVariantNodes(SymShapes& shapes);

  /// Shapes the duplicates of values made during grid generation, which a
  /// reused op gives each of its invocations: the original's rule over the
  /// invocation's operands. Returns how many shapes it added.
  int32_t shapeDuplicates(SymShapes& shapes);

  /// Applies the reserves' symbolic forms to the bound steps of 'state',
  /// adding to 'shapes'. Returns how many shapes it added.
  int32_t applySymReserves(const ExecutionState& state, SymShapes& shapes);

  /// Checks that the export's symbolic shapes give the dims the host sized
  /// the tensor parameters of 'program' with, at 'params'. A symbol is bound
  /// from the first dim that is that symbol alone; every other dim is
  /// evaluated and compared. Logs a summary now and then. For
  /// WaveConfig::checkSizeExprs.
  void checkSizeExprs(
      ExecutionState& state,
      const StepProgram& program,
      const uint8_t* params,
      int32_t sequenceNumber,
      int32_t stepIdx) const;

  /// The CUDA source of the fixups: __device__ fixupBody(params, records,
  /// code, numRecords, symbols, numSymbols), called by every thread of one
  /// block, and the fixupSymbols kernel that runs it.
  static const char* fixupSource();

  /// The dynamic shared memory fixupBody uses for 'numCode' words of
  /// programs: all of them, or none when they do not fit.
  static int32_t fixupSharedBytes(int32_t numCode);

  /// Logs, for each step that still waits, the returned values it waits for
  /// and why each was not picked. For TW_DEVICE_SYMBOLS_TRACE.
  void traceWaits(
      const ExecutionState& state,
      const folly::F14FastSet<int32_t>& returned) const;

  /// One line on what was picked, for the trace.
  std::string report(const ExecutionState& state) const;

  /// Changes when a symbol is banned, so states bound with it are dropped.
  int64_t generation() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return generation_;
  }

  static constexpr int32_t kMaxSymbols = 1024;

 private:
  /// One pass of bindSizeChecks. 'capacityRegion' says in which groups the
  /// setup places the members of device symbols' regions at their
  /// capacities: none (nullptr) or the groups in the set; 'allRegions'
  /// places them so in every group.
  void bindSizeChecksPass(
      ExecutionState& state,
      const folly::F14FastSet<const AllocGroup*>* capacityRegion,
      bool allRegions);

  int32_t symbolIndex(const std::string& symbol);

  // Renames the symbols a broadcast proves equal to one of them, then defines
  // the lengths of slices with symbolic bounds.
  void mergeEqualSymbols();

  void applyRenames(const folly::F14FastMap<std::string, std::string>& renames);

  // Defines each slice length symbol no item() returns, from the slice's
  // bounds.
  void defineSliceSymbols();

  // Appends the postfix program of 'expr' to 'code' and returns its offset.
  int32_t emitProgram(const SymExpr& expr, std::vector<int64_t>& code);

  const SymShape* FOLLY_NULLABLE shapeOf(nativert::ValueId id) const;

  // A concat whose bands the host places, with the values the bands are.
  struct HostConcat {
    nativert::ValueId resultId{-1};
    int64_t dim{0};
    bool stack{false};
    // Per operand in join order: the value whose extent it has, and the bands
    // the host places for it.
    std::vector<nativert::ValueId> operands;
    std::vector<std::vector<nativert::ValueId>> bands;
  };

  // The concats of 'state''s steps that place bands on the host.
  std::vector<HostConcat> hostConcats(const ExecutionState& state) const;

  // Empty when the fixups can move 'band', else why not: the host reads it,
  // hands it out, or makes a view of it.
  std::string bandMovable(
      nativert::ValueId band,
      const folly::F14FastSet<nativert::ValueId>& hostRead,
      const folly::F14FastSet<nativert::ValueId>& graphOutputs) const;

  const SymExpr* FOLLY_NULLABLE scalarOf(nativert::ValueId id) const;

  // Logs the producer of the value a rejection names, its operands and users.
  // For TW_DEVICE_SYMBOLS_TRACE.
  void traceRejection(const std::string& why) const;

  std::shared_ptr<SymShapes> shapes_;
  // See deriveShapesFromInputs; null unless WaveConfig::sizeExprsFromInputs.
  std::shared_ptr<SymShapes> derivedShapes_;
  // Outputs the device gives their actual shape. The producing step's frame
  // holds the reserve's capacity and later frames the actual shape, so the
  // host's dims for them depend on the step.
  folly::F14FastSet<std::string> setOnDevice_;
  // Outputs of ops no rule describes, shaped from the export: the host
  // decides them when the op returns.
  folly::F14FastSet<std::string> hostShaped_;
  // By value name: the shape the reserve's symbolic form gives an output, a
  // capacity where the data decides the extent. Set by applySymReserves.
  folly::F14FastMap<std::string, SymShape> capacityByName_;
  WaveGraph& waveGraph_;
  // For checkSizeExprs: parameters without an expression by producer.
  mutable std::mutex noShapeMutex_;
  mutable folly::F14FastMap<std::string, int64_t> noShapeByTarget_;
  // For checkSizeExprs: outputs compared at the end of the execution.
  struct PendingSizeCheck {
    int32_t valueId;
    std::vector<SymExpr> dims;
    std::vector<int64_t> hostDims;
  };
  mutable std::vector<PendingSizeCheck> pendingSizeChecks_;
  // For checkSizeExprs: parameters whose expression differs, by producer.
  mutable folly::F14FastMap<std::string, int64_t> differByTarget_;
  // checkSizeExprs tallies since the last reportSizeExprs.
  mutable std::atomic<int64_t> numSizeValues_{0};
  mutable std::atomic<int64_t> numSizeNoShape_{0};
  mutable std::atomic<int64_t> numSizeMatched_{0};
  mutable std::atomic<int64_t> numSizeDiffer_{0};
  mutable std::atomic<int64_t> numSizeUnbound_{0};
  mutable std::atomic<int64_t> numSizeSetOnDevice_{0};
  mutable std::atomic<int64_t> numSizeHostShaped_{0};
  mutable std::atomic<int32_t> numSizeReports_{0};
  mutable std::atomic<int32_t> numSizeUnboundReports_{0};
  // Integer values whose expression is a single data-dependent symbol.
  folly::F14FastMap<nativert::ValueId, std::string> candidateSymbol_;
  // Tensors whose one dim is a data-dependent symbol, the count the device
  // writes into their dims.
  folly::F14FastMap<nativert::ValueId, std::string> candidateTensor_;
  // The reserved size of such a symbol, its fixed capacity.
  folly::F14FastMap<std::string, int64_t> boundOf_;
  folly::F14FastMap<std::string, int32_t> symbolIndex_;
  std::vector<std::string> symbols_;
  mutable std::mutex mutex_;
  folly::F14FastMap<std::string, int64_t> capacity_;
  // Symbols whose merged carriers disagreed at run time.
  folly::F14FastSet<std::string> banned_;
  int64_t generation_{0};
  // Why each candidate symbol was not picked, from the last selection.
  folly::F14FastMap<std::string, std::string> reason_;
};

} // namespace torch::wave
