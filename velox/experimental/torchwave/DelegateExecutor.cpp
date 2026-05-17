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

#include "velox/experimental/torchwave/DelegateExecutor.h"

#include <sstream>

#include <ATen/ATen.h>
#include <caffe2/caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h> // @manual=//caffe2/caffe2/serialize:inline_container
#include <torch/csrc/export/pt2_archive_constants.h>
#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/kernels/KernelFactory.h>

#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/Pt2Load.h"

namespace torch::wave {

struct DelegateExecutor::Impl {
  LoadedModel model;
  std::shared_ptr<nativert::Weights> weights;
  nativert::ExecutorConfig config;
  std::unique_ptr<WaveGraphExecutor> executor;
  /// Frame passed to execute(). The WaveGraphExecutor ignores this parameter
  /// and uses an internal pool, but the API requires a reference.
  std::unique_ptr<nativert::ExecutionFrame> dummyFrame;
};

DelegateExecutor::DelegateExecutor(const std::string& pt2Path)
    : impl_(std::make_unique<Impl>()) {
  initialize();

  auto reader = std::make_shared<caffe2::serialize::PyTorchStreamReader>(
      std::make_unique<caffe2::serialize::FileAdapter>(pt2Path));

  auto modelNames = getModelNames(*reader);
  TORCH_CHECK(!modelNames.empty(), "No models found in: ", pt2Path);

  impl_->model = loadPt2Model(reader, modelNames[0]);
  auto& graph = *impl_->model.graph;

  impl_->weights = std::make_shared<nativert::Weights>(
      &graph,
      reader,
      impl_->model.tensorPaths,
      torch::_export::archive_spec::WEIGHTS_DIR,
      impl_->model.constantPaths,
      torch::_export::archive_spec::CONSTANTS_DIR);

  nativert::KernelFactory factory;
  auto execKernels = factory.initializeNodeKernels(
      graph, impl_->weights, impl_->config, reader);

  impl_->executor = std::make_unique<WaveGraphExecutor>(
      graph,
      std::move(execKernels.nodeKernels),
      impl_->config,
      impl_->weights);

  impl_->dummyFrame = std::make_unique<nativert::ExecutionFrame>(
      graph, *impl_->weights, impl_->config);
}

DelegateExecutor::~DelegateExecutor() = default;

std::vector<at::Tensor> DelegateExecutor::run(
    const std::vector<at::Tensor>& inputs) {
  std::vector<c10::IValue> ivalueInputs;
  ivalueInputs.reserve(inputs.size());
  for (const auto& t : inputs) {
    ivalueInputs.emplace_back(t);
  }

  auto outputs =
      impl_->executor->execute(*impl_->dummyFrame, std::move(ivalueInputs));

  std::vector<at::Tensor> result;
  result.reserve(outputs.size());
  for (auto& iv : outputs) {
    TORCH_CHECK(iv.isTensor(), "Expected tensor output from delegate executor");
    result.push_back(iv.toTensor());
  }
  return result;
}

std::string DelegateExecutor::toString() const {
  std::stringstream ss;
  ss << "DelegateExecutor for model '" << impl_->model.modelName << "'";
  return ss.str();
}

} // namespace torch::wave
