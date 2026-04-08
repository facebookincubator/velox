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

#include "velox/experimental/torchwave/NativertDelegateExecutor.h"

#include <torch/csrc/export/pt2_archive_constants.h>
#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/kernels/KernelFactory.h>

namespace torch::wave {

namespace {
constexpr std::string_view kTorchwaveSuffix = "-torchwave";
} // namespace

NativertDelegateExecutor::NativertDelegateExecutor(
    const nativert::Node& node,
    const std::shared_ptr<nativert::Weights>& /*weights*/,
    const nativert::ExecutorConfig& executorConfig,
    caffe2::serialize::PyTorchStreamReader* packageReader)
    : ETDelegateExecutor("", node), config_(executorConfig) {
  initialize();

  // get_delegate_dir() returns "" + node.attributes()[0], e.g.
  // "model-torchwave". Strip the suffix to recover the original model name.
  const auto& delegatePath = get_delegate_dir();
  TORCH_CHECK(
      delegatePath.size() > kTorchwaveSuffix.size() &&
          delegatePath.compare(
              delegatePath.size() - kTorchwaveSuffix.size(),
              kTorchwaveSuffix.size(),
              kTorchwaveSuffix) == 0,
      "Expected delegate path ending with '-torchwave', got: ",
      delegatePath);

  std::string originalModelName =
      delegatePath.substr(0, delegatePath.size() - kTorchwaveSuffix.size());

  LOG(INFO) << "Loading torchwave delegate for model: " << originalModelName;

  // Wrap the raw pointer in a non-owning shared_ptr so we can pass it to
  // APIs that expect shared ownership.
  auto readerPtr = std::shared_ptr<caffe2::serialize::PyTorchStreamReader>(
      packageReader, [](auto*) {});

  model_ = loadPt2Model(readerPtr, originalModelName);
  auto& graph = *model_.graph;

  waveWeights_ = std::make_shared<nativert::Weights>(
      &graph,
      readerPtr,
      model_.tensorPaths,
      torch::_export::archive_spec::WEIGHTS_DIR,
      model_.constantPaths,
      torch::_export::archive_spec::CONSTANTS_DIR);

  nativert::KernelFactory factory;
  auto execKernels = factory.initializeNodeKernels(
      graph, waveWeights_, executorConfig, readerPtr);

  executor_ = std::make_unique<WaveGraphExecutor>(
      graph,
      std::move(execKernels.nodeKernels),
      executorConfig,
      waveWeights_);

  dummyFrame_ = std::make_unique<nativert::ExecutionFrame>(
      graph, *waveWeights_, config_);
}

void NativertDelegateExecutor::processWeights(
    std::shared_ptr<nativert::Weights> /*weights*/) {}

void NativertDelegateExecutor::commitWeights() {}

void NativertDelegateExecutor::initWeights(
    std::shared_ptr<nativert::Weights> /*weights*/) {}

std::vector<at::Tensor> NativertDelegateExecutor::run(
    std::vector<at::Tensor>& inputs) {
  std::vector<c10::IValue> ivalueInputs;
  ivalueInputs.reserve(inputs.size());
  for (auto& t : inputs) {
    ivalueInputs.emplace_back(t);
  }

  // WaveGraphExecutor::execute ignores the frame parameter and uses an
  // internal pool, but the API requires a reference.
  auto outputs = executor_->execute(*dummyFrame_, std::move(ivalueInputs));

  std::vector<at::Tensor> result;
  result.reserve(outputs.size());
  for (auto& iv : outputs) {
    TORCH_CHECK(
        iv.isTensor(), "Expected tensor output from torchwave delegate");
    result.push_back(iv.toTensor());
  }
  return result;
}

} // namespace torch::wave
