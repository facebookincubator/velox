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
#include <unordered_map>

#include <caffe2/serialize/inline_container.h>
#include <torch/nativert/executor/ExecutionFrame.h>
#include <torch/nativert/graph/Graph.h>

namespace torch::wave {

/// Format-agnostic representation of a loaded model.  Produced by
/// loadSigmoidPackage() or loadPt2File() and consumed by executeGraph()
/// and the compile path so that neither needs to know the file format.
struct LoadedModel {
  std::shared_ptr<caffe2::serialize::PyTorchStreamReader> reader;
  std::unique_ptr<torch::nativert::Graph> graph;
  std::string modelName;
  std::unordered_map<std::string, std::string> tensorPaths;
  std::unordered_map<std::string, std::string> constantPaths;
};

// Builds an ExecutionFrame from the graph and the package weights,
// then executes all nodes via nativert kernel dispatch.
// Returns the frame after execution.
std::unique_ptr<torch::nativert::ExecutionFrame> executeGraph(
    const LoadedModel& model);

} // namespace torch::wave
