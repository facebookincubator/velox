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
#include <vector>

#include <ATen/core/ivalue.h>

namespace at {
class Tensor;
}

namespace torch::wave {

/// Self-contained executor that loads a .pt2 archive and runs the model through
/// the torchwave GPU backend. Follows the AOTI delegate pattern: construction
/// compiles the model graph to fused CUDA kernels, run() executes them.
class DelegateExecutor {
 public:
  /// Loads a .pt2 model archive, builds the wave graph, and initializes GPU
  /// resources.
  explicit DelegateExecutor(const std::string& pt2Path);
  ~DelegateExecutor();

  /// Executes the model with tensor inputs and returns tensor outputs. Inputs
  /// are transferred to device, execution runs on GPU, and outputs are returned
  /// on host.
  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inputs);

  std::string toString() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace torch::wave
