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

#include <torch/nativert/executor/ETDelegateExecutor.h>
#include <torch/nativert/executor/ExecutorConfig.h>
#include "velox/experimental/torchwave/Executor.h"
#include "velox/experimental/torchwave/Pt2Load.h"

namespace torch::wave {

/// Nativert-integrated delegate executor that runs subgraphs through the
/// torchwave GPU backend. Loads the original (non-delegated) model from the
/// pt2 archive and builds a WaveGraphExecutor to run it.
class NativertDelegateExecutor : public nativert::ETDelegateExecutor {
 public:
  NativertDelegateExecutor(
      const nativert::Node& node,
      const std::shared_ptr<nativert::Weights>& weights,
      const nativert::ExecutorConfig& executorConfig,
      caffe2::serialize::PyTorchStreamReader* packageReader);

  ~NativertDelegateExecutor() override = default;

  void processWeights(std::shared_ptr<nativert::Weights> weights) override;
  void commitWeights() override;
  void initWeights(std::shared_ptr<nativert::Weights> weights) override;
  std::vector<at::Tensor> run(std::vector<at::Tensor>& inputs) override;

 private:
  LoadedModel model_;
  std::shared_ptr<nativert::Weights> waveWeights_;
  nativert::ExecutorConfig config_;
  std::unique_ptr<WaveGraphExecutor> executor_;
  /// Dummy frame passed to execute(). WaveGraphExecutor ignores this and
  /// uses an internal pool, but the API requires a reference.
  std::unique_ptr<nativert::ExecutionFrame> dummyFrame_;
};

} // namespace torch::wave
