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
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <caffe2/serialize/inline_container.h> // @manual=//caffe2/caffe2/serialize:inline_container
#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/executor/Weights.h>
#include <torch/nativert/kernels/KernelFactory.h>

#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/Pt2Load.h"

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

  /// Counters copied from WaveGraphExecutor after runWave.
  int64_t lastRefTensorsChecked_{0};
  int64_t lastRefNodesChecked_{0};

  /// Display name for the current test, included in failure messages.
  std::string displayName_;

  // Snapshot of the serial frame's initial state (after filling user inputs,
  // before execution). Maps value id to shape string or "none"/"scalar".
  // Used to verify that the wave frame has the same slots populated.
  std::unordered_map<int32_t, std::string> serialFrameSnapshot_;

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
