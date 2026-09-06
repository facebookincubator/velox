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

#include "velox/experimental/torchwave/Model.h"

#include <algorithm>

#include <caffe2/caffe2/serialize/file_adapter.h>
#include <torch/csrc/export/pt2_archive_constants.h>
#include <torch/nativert/executor/Weights.h>
#include <torch/nativert/graph/GraphPasses.h>

#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/GraphPrep.h"
#include "velox/experimental/torchwave/Pt2Load.h"
#include "velox/experimental/torchwave/Utils.h"
#include "velox/experimental/torchwave/WaveGraph.h"

namespace torch::wave {

std::unique_ptr<TorchWaveModel> TorchWaveModel::load(std::string_view path) {
  return load(path, std::string_view{}, nativert::ExecutorConfig{});
}

std::unique_ptr<TorchWaveModel> TorchWaveModel::load(
    std::string_view path,
    std::string_view modelName,
    const nativert::ExecutorConfig& config) {
  auto reader = std::make_shared<caffe2::serialize::PyTorchStreamReader>(
      std::make_unique<caffe2::serialize::FileAdapter>(std::string(path)));
  return loadFromReader(std::move(reader), modelName, config);
}

std::unique_ptr<TorchWaveModel> TorchWaveModel::loadFromReader(
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> reader) {
  return loadFromReader(
      std::move(reader), std::string_view{}, nativert::ExecutorConfig{});
}

std::unique_ptr<TorchWaveModel> TorchWaveModel::loadFromReader(
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> reader,
    std::string_view modelName,
    const nativert::ExecutorConfig& config) {
  // Set up the Wave runtime (device, NVRTC/system-header init, global device
  // arena) before building any executor. Idempotent; a caller that already ran
  // it (e.g. a test harness) pays nothing. Without it the executor runs with an
  // uninitialized runtime and crashes on the first execution.
  initialize();

  auto names = getModelNames(*reader);
  TORCH_CHECK(!names.empty(), "No models found in archive");
  std::string name{modelName.empty() ? names.front() : std::string(modelName)};
  // Guard against a misspelled or stale model name producing a cryptic error
  // deep in the loading stack.
  TORCH_CHECK(
      std::find(names.begin(), names.end(), name) != names.end(),
      "Model not found in archive: ",
      name);

  auto loaded = loadPt2Model(reader, name);
  // Graph normalization required before any executor can run the graph.
  nativert::selectScalarOverload(loaded.graph.get());

  auto weights = std::make_shared<nativert::Weights>(
      loaded.graph.get(),
      reader,
      loaded.tensorPaths,
      torch::_export::archive_spec::WEIGHTS_DIR,
      loaded.constantPaths,
      torch::_export::archive_spec::CONSTANTS_DIR);

  // Prep the graph for whole-graph GPU execution, mirroring the wave test
  // harness. A raw exported graph otherwise reaches wave codegen with
  // side-effect asserts and CPU-only ops it cannot compile (e.g. a None-typed
  // assert operand trips elementwise codegen). Strip the data asserts, then
  // force the whole graph onto CUDA -- ops that TorchWave does not fuse run as
  // standalone nativert nodes; without this they inherit the exported graph's
  // (host) device while runInputs moves the tensors to device, so a standalone
  // node fails with cudaErrorInvalidKernelImage -- then rewrite CPU-only ops to
  // CUDA-capable equivalents and keep cpuOnly args on the host.
  stripDataAsserts(*loaded.graph);
  setGraphDevice(loaded.graph.get(), /*isCuda=*/true);
  rewriteGpuIncompatibleOps(*loaded.graph);
  insertCpuOnlyCopies(*loaded.graph);
  // Retarget merge-and-dedup to the fused TorchWave _tw CUDA ops when a build
  // has registered them (e.g. the _torchwave_meta extension); a no-op for the
  // base engine, where the sparsenn op runs as a standalone.
  rewriteMergeAndDedupToTw(*loaded.graph);

  auto context = std::make_unique<ModelContext>();
  context->graph = std::move(loaded.graph);
  context->weights = std::move(weights);
  context->config = config;

  auto executor = std::make_unique<WaveGraphExecutor>(std::move(context));
  return std::unique_ptr<TorchWaveModel>(
      new TorchWaveModel(std::move(reader), std::move(executor)));
}

TorchWaveModel::TorchWaveModel(
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> reader,
    std::unique_ptr<WaveGraphExecutor> executor)
    : reader_(std::move(reader)), executor_(std::move(executor)) {}

TorchWaveModel::~TorchWaveModel() = default;

std::vector<c10::IValue> TorchWaveModel::run(std::vector<c10::IValue> inputs) {
  return executor_->runInputs(std::move(inputs));
}

std::vector<c10::IValue> TorchWaveModel::runReuse(
    std::vector<c10::IValue> inputs) {
  return executor_->runInputsReuse(std::move(inputs));
}

std::vector<at::Tensor> TorchWaveModel::runTensors(
    std::vector<at::Tensor> inputs) {
  std::vector<c10::IValue> ivalues;
  ivalues.reserve(inputs.size());
  for (auto& tensor : inputs) {
    ivalues.emplace_back(std::move(tensor));
  }
  return outputsToTensors(run(std::move(ivalues)));
}

std::vector<at::Tensor> outputsToTensors(std::vector<c10::IValue> outputs) {
  std::vector<at::Tensor> tensors;
  tensors.reserve(outputs.size());
  for (auto& output : outputs) {
    tensors.push_back(output.isTensor() ? output.toTensor() : at::Tensor());
  }
  return tensors;
}

} // namespace torch::wave
