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

#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <caffe2/serialize/inline_container.h> // @manual=//caffe2/caffe2/serialize:inline_container
#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/executor/Weights.h>
#include <torch/nativert/kernels/KernelFactory.h>

#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/Pt2Load.h"
#include "velox/experimental/torchwave/tests/DataGen.h"

namespace torch::wave {

/// Timing results from running a model through an executor.
struct RunTiming {
  /// Time to transfer inputs to device (0 for CPU executors).
  int64_t dataTransferUs{0};
  /// Execution time excluding data transfer.
  int64_t executeUs{0};
  /// Per-launch debug info (outer = launches, inner = blocks).
  std::vector<std::vector<DebugInfo>> debugInfo;
};

/// Holds the loaded model and weights shared across test methods.
struct ModelFixture {
  std::string pt2Path;
  std::shared_ptr<caffe2::serialize::PyTorchStreamReader> reader;
  LoadedModel model;
  std::shared_ptr<nativert::Weights> weights;
  nativert::ExecutorConfig config;

  // Path to external inputs file, if available.
  std::string externalInputsPath;

  static std::unique_ptr<ModelFixture> load(const std::string& pt2Path);

  /// Applies graph normalization passes (e.g. selectScalarOverload) that are
  /// needed before any executor can run the graph. Called automatically by
  /// load(); call manually when constructing a ModelFixture from a sigmoid
  /// archive or other non-pt2 source.
  static void prepareGraph(nativert::Graph* graph);

  /// Creates fresh node kernels (each executor consumes them).
  std::vector<std::unique_ptr<nativert::OpKernel>> makeKernels() const;

  /// Creates a ModelContext for passing to WaveGraphExecutor, moving the
  /// graph out of this fixture. Can only be called once per fixture.
  std::unique_ptr<ModelContext> makeModelContext();
};

/// Returns the path to a test data file. In fbcode (Buck), the CWD ends with
/// "fbcode" and files are found via baseDir/filePath. In CMake builds the CWD
/// is set to the source directory containing the test, so just filePath works.
std::string getDataFilePath(
    const std::string& baseDir,
    const std::string& filePath);

/// Reads a file into a vector of chars.
std::vector<char> readFile(const std::string& path);

/// Loads reference values saved by Python via torch.save([v1, v2, ...], path).
/// Values may be tensors or scalars.
std::vector<c10::IValue> loadReferenceValues(const std::string& path);

/// Extracts tensors from IValues and transfers them to device.
/// Returns the device tensors and the transfer time in microseconds.
std::pair<std::vector<c10::IValue>, int64_t> inputsToDevice(
    std::vector<c10::IValue>& inputs);

/// Inserts an `aten._to_copy(self, device=cpu)` node before every input
/// argument flagged `cpuOnly` in its wave Metadata (e.g. the indices of
/// `aten.tensor_split.tensor_indices_or_sections`), repointing just that edge.
/// This lets the generic nativert executor run the graph on GPU: tensor_split
/// reads its indices on the host and returns views of `self`, so `self` and the
/// outputs stay on GPU and no move-back is needed. Mutates `graph` in place, so
/// call it on a clone reserved for the nativert-GPU run (wave handles cpuOnly
/// args itself at runtime and must keep its own copy-free graph). Returns the
/// number of nodes inserted.
int32_t insertCpuOnlyCopies(nativert::Graph& graph);

/// Rewrites ops that have no CUDA implementation to a CUDA-capable equivalent
/// so the generic nativert executor can run the graph on GPU. Currently
/// rewrites `fb.simple_1d_concat` (CUDA registration is a throwing dummy) to
/// `aten.cat.default(dim=0)`, mirroring wave's MoreBuiltins rewrite. Mutates
/// `graph`; call on the nativert-GPU clone. Returns the number of nodes
/// rewritten.
int32_t rewriteGpuIncompatibleOps(nativert::Graph& graph);

/// Snapshots a frame: returns a map from value id to shape string (e.g.
/// "[3,4]") for tensors, "scalar" for scalars. None slots are omitted.
std::unordered_map<int32_t, std::string> snapshotFrame(
    const nativert::ExecutionFrame& frame,
    int32_t numValues);

/// Compares two frame snapshots and logs mismatches (slots that are set in one
/// but not the other, or have different shapes).
void compareFrameSnapshots(
    const std::unordered_map<int32_t, std::string>& serial,
    const std::unordered_map<int32_t, std::string>& wave,
    int32_t numValues);

/// Extracts values from output IValues and transfers tensors to host.
/// Scalar IValues are passed through unchanged.
std::vector<c10::IValue> outputsToHost(
    const std::vector<c10::IValue>& outputs,
    const std::string& label);

/// Prints standalone execution times from a WaveGraphExecutor, sorted highest
/// time first. Each node is serialized as in Launch::toString.
void standaloneReport(WaveGraphExecutor& executor);

/// Copies 'reader's records into a fresh archive at 'outPath', dropping the
/// data sections (data/weights, data/constants, data/sample_inputs) so the
/// result carries only the graph serialization and archive metadata. Used by
/// --save_model to write a small, data-free graph archive (a .pt2 for a .pt2
/// source, or the sigmoid graph for a sigmoid source) that --run_synthetic
/// loads and pairs with synthetic weights and inputs from the spec.
void saveModelArchive(
    caffe2::serialize::PyTorchStreamReader& reader,
    const std::string& outPath);

/// Base test fixture for running .pt2 models through serial and wave executors.
class ExecutorTestBase : public ::testing::Test {
 protected:
  static void SetUpTestSuite();

  /// Runs a .pt2 model through the nativert SerialGraphExecutor on CPU.
  /// Verifies outputs match 'expected' and returns timing.
  RunTiming runSerial(
      ModelFixture& fixture,
      const std::vector<c10::IValue>& expected);

  RunTiming runSerialOnDevice(
      ModelFixture& fixture,
      const std::vector<c10::IValue>& expected);

  /// Fills device inputs into a WaveGraphExecutor frame.
  void fillWaveFrame(
      const nativert::Graph& graph,
      nativert::ExecutionFrame& frame,
      const std::vector<c10::IValue>& deviceInputs);

  /// Runs a .pt2 model through the WaveGraphExecutor on device.
  /// Verifies outputs match 'expected' and returns timing. If 'alterInputs'
  /// is set, it is called after filling the frame but before execution,
  /// allowing modification of input tensors (e.g. injecting bad indices).
  RunTiming runWave(
      ModelFixture& fixture,
      const std::vector<c10::IValue>& expected,
      const std::function<void(nativert::ExecutionFrame&)>& alterInputs =
          nullptr);

  /// Runs only the wave path on 'fixture' (no serial run, no output
  /// verification) for testing device-side error conditions. 'alterInputs' is
  /// called after the frame is filled and before execution, so it can corrupt
  /// an input (e.g. an out-of-range index) to trigger a device-side check.
  /// Returns the formatted device error string (waveThreadInfo().errors), which
  /// is empty if no block reported an error. throwOnError is forced off for the
  /// run and restored afterwards.
  std::string runWaveExpectError(
      ModelFixture& fixture,
      const std::function<void(nativert::ExecutionFrame&)>& alterInputs);

  /// Loads a .pt2 model and reference results, runs serial on CPU, serial on
  /// device and wave, and logs the run times for each.
  void runTest(
      const std::string& pt2File,
      const std::string& resultsFile,
      const std::string& label = "");

  /// Like runTest but takes a pre-built fixture. Runs the same serial/wave
  /// pipeline with tracing and reference frame support.
  void runTestWithFixture(
      std::unique_ptr<ModelFixture> fixture,
      const std::vector<c10::IValue>& expected,
      const std::string& label = "");

  /// Executes a pre-filled frame node by node using the given kernels,
  /// tracing values in trace_values. Returns the user outputs.
  std::vector<c10::IValue> executeSerialWithTrace(
      const nativert::Graph& graph,
      nativert::ExecutionFrame& frame,
      std::vector<std::unique_ptr<nativert::OpKernel>> nodeKernels);

  virtual std::string dataDir() const {
    return "velox/experimental/torchwave/tests";
  }

  /// Loads the graph archive for --run_synthetic. The base loads a standard
  /// .pt2 (JSON). The meta harness overrides this to also accept a sigmoid
  /// (thrift) archive.
  virtual std::unique_ptr<ModelFixture> loadSyntheticFixture(
      const std::string& pt2Path);

  /// Subclass hook applied to both the reference and wave graphs in
  /// runSynthetic, before device placement, so the two runs use the same
  /// kernels. The meta harness overrides this to rewrite
  /// fused_datafm_merge_and_dedup_by_reference to its _tw variant (the
  /// reference and _tw dedup implementations differ, so comparing them directly
  /// produces spurious mismatches). Default: no-op.
  virtual void applySyntheticGraphRewrites(nativert::Graph& /*graph*/) {}

  /// --save_model: writes 'path.spec' (a DatasetSpec analyzed from 'inputs' and
  /// the fixture's weights) and 'path.pt2' (a copy of the source archive).
  void saveSyntheticModel(
      ModelFixture& fixture,
      const std::vector<c10::IValue>& inputs,
      const std::string& path);

  /// --run_synthetic: loads 'path.pt2' and 'path.spec', generates synthetic
  /// data, runs the nativert GPU reference to produce reference outputs (and a
  /// reference frame), then runs wave on the same data and verifies it against
  /// both. 'seed' is optional; when unset the spec's seed is used.
  void runSynthetic(
      const std::string& path,
      std::optional<uint64_t> seed = std::nullopt);

  /// Runs 'fixture' through the nativert serial executor on GPU with explicit
  /// 'inputs' (not loadSampleInputs). Applies applySyntheticGraphRewrites, then
  /// GPU placement (setGraphDevice / rewriteGpuIncompatibleOps /
  /// insertCpuOnlyCopies) so the reference runs on the same device and kernels
  /// as wave. The fixture's weights must already be GPU-resident. Returns the
  /// host outputs and, if 'refFramePath' is non-empty, saves the post-run frame
  /// there as a reference frame.
  std::vector<c10::IValue> runNativertReferenceWithInputs(
      ModelFixture& fixture,
      std::vector<c10::IValue> inputs,
      const std::string& refFramePath);

  /// Runs 'fixture' through the wave executor with explicit 'inputs' and
  /// verifies against 'expected'. If 'refFramePath' is non-empty it is loaded
  /// as the reference frame so wave checks intermediates too.
  void runWaveWithInputs(
      ModelFixture& fixture,
      std::vector<c10::IValue> inputs,
      const std::vector<c10::IValue>& expected,
      const std::string& refFramePath);

  /// Counters copied from WaveGraphExecutor after runWave.
  int64_t lastRefTensorsChecked_{0};
  int64_t lastRefNodesChecked_{0};

  /// Display name for the current test, included in failure messages.
  std::string displayName_;

  // Snapshot of the serial frame's initial state (after filling user inputs,
  // before execution). Maps value id to shape string or "none"/"scalar".
  // Used to verify that the wave frame has the same slots populated.
  std::unordered_map<int32_t, std::string> serialFrameSnapshot_;

  // TorchWave debug: deep copies of every node output captured the instant the
  // node ran in the serial (CPU nativert) reference run. Saved as the reference
  // frame so that later in-place corruption of a value's storage cannot poison
  // the recorded reference. Populated only when saving a reference.
  std::unordered_map<int64_t, at::Tensor> capturedRefOutputs_;

  // TorchWave debug: CPU copies of the serial (CPU nativert) run's final model
  // outputs. Compared against the wave run's final outputs to check end-to-end
  // correctness independent of intermediate dead-value noise.
  std::vector<at::Tensor> nativertOutputs_;

  /// Loads sample inputs from the .pt2 package.
  std::vector<c10::IValue> loadSampleInputs(ModelFixture& fixture);

  /// Verifies that outputs match expected values. Both tensors and scalars
  /// are supported.
  void verifyOutputs(
      const std::vector<c10::IValue>& outputs,
      const std::vector<c10::IValue>& expected,
      const std::string& label);
};

} // namespace torch::wave
