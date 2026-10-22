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

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <ATen/core/ivalue.h>
#include <caffe2/serialize/inline_container.h> // @manual=//caffe2/caffe2/serialize:inline_container
#include <torch/nativert/executor/ExecutorConfig.h>

namespace torch::wave {

class WaveGraphExecutor;

/// Holds a loaded, compiled TorchWave model: the whole-graph alternative to an
/// AOTInductor model container. load() reads a .pt2 archive, compiles the
/// model's graph to a WaveGraph, and run() executes it on the GPU and returns
/// the user outputs. This is the C++ serving entry point a caller uses instead
/// of torch::inductor::AOTIModelContainerRunner.
class TorchWaveModel {
 public:
  /// Loads the first model in the .pt2 archive at 'path' and compiles it.
  static std::unique_ptr<TorchWaveModel> load(std::string_view path);

  /// Loads 'modelName' from the .pt2 archive at 'path' and compiles it.
  /// 'config' is the nativert executor config used when building fallback
  /// (standalone) kernels.
  static std::unique_ptr<TorchWaveModel> load(
      std::string_view path,
      std::string_view modelName,
      const nativert::ExecutorConfig& config);

  /// Loads the first model from an already-open archive reader.
  static std::unique_ptr<TorchWaveModel> loadFromReader(
      std::shared_ptr<caffe2::serialize::PyTorchStreamReader> reader);

  /// Loads 'modelName' from an already-open archive reader; otherwise like
  /// load().
  static std::unique_ptr<TorchWaveModel> loadFromReader(
      std::shared_ptr<caffe2::serialize::PyTorchStreamReader> reader,
      std::string_view modelName,
      const nativert::ExecutorConfig& config);

  ~TorchWaveModel();

  /// Runs the model on positional 'inputs' (in graph user-input order) and
  /// returns the user outputs. Inputs may be on host or device; outputs are on
  /// device.
  std::vector<c10::IValue> run(std::vector<c10::IValue> inputs);

  /// Like run() but reuses a single held execution frame across calls: weights
  /// and constants stay resident (set once) and only the user inputs are
  /// refilled. Single-threaded fast path (one in-flight call); use run() for
  /// concurrent callers.
  std::vector<c10::IValue> runReuse(std::vector<c10::IValue> inputs);

  /// Tensor-only convenience over run(): accepts tensor inputs and returns the
  /// tensor outputs (non-tensor outputs become undefined tensors).
  std::vector<at::Tensor> runTensors(std::vector<at::Tensor> inputs);

  WaveGraphExecutor& executor() {
    return *executor_;
  }

 private:
  TorchWaveModel(
      std::shared_ptr<caffe2::serialize::PyTorchStreamReader> reader,
      std::unique_ptr<WaveGraphExecutor> executor);

  // Keeps the .pt2 archive reader alive for the model's lifetime; weights and
  // constants may be read from it lazily during execution.
  std::shared_ptr<caffe2::serialize::PyTorchStreamReader> reader_;

  // Owns the compiled whole-graph WaveGraph and executes it on each run().
  std::unique_ptr<WaveGraphExecutor> executor_;
};

/// Converts the outputs of run() to tensors, mapping a non-tensor output to an
/// undefined tensor so the result stays positional with the graph's outputs.
std::vector<at::Tensor> outputsToTensors(std::vector<c10::IValue> outputs);

} // namespace torch::wave
