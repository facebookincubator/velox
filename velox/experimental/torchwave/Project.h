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

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <torch/nativert/graph/Graph.h>

namespace torch::wave {

class ProjectNode {
 public:
  ProjectNode(
      std::vector<const nativert::Node*> nodes,
      std::unordered_set<const nativert::Node*> inputs,
      std::unordered_set<const nativert::Node*> leafInputs,
      ProjectNode* input,
      int32_t id)
      : nodes_{std::move(nodes)},
        inputs_{std::move(inputs)},
        leafInputs_{std::move(leafInputs)},
        input_{input},
        id_{id} {}

  int32_t id() const {
    return id_;
  }

  std::string toString(
      const nativert::Graph& graph,
      std::unordered_set<const nativert::Node*>& border) const;

  const std::vector<const nativert::Node*>& nodes() const {
    return nodes_;
  }

  const std::unordered_set<const nativert::Node*>& inputs() const {
    return inputs_;
  }

  const std::unordered_set<const nativert::Node*>& leafInputs() const {
    return leafInputs_;
  }

  ProjectNode* input() const {
    return input_;
  }

  void distinctFunctions(
      std::unordered_map<std::string, int32_t>& counts) const;

 private:
  std::vector<const nativert::Node*> nodes_;
  std::unordered_set<const nativert::Node*> inputs_;
  std::unordered_set<const nativert::Node*> leafInputs_;
  ProjectNode* input_;
  int32_t id_;
};

} // namespace torch::wave
