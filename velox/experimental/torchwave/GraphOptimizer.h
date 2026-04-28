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

#include <unordered_set>

#include "velox/experimental/torchwave/CompiledOp.h"
#include "velox/experimental/torchwave/Registry.h"

namespace torch::wave {

/// Walks the graph backwards from outputs, attaching ValueConstraints to
/// values based on TensorMeta and Registry Metadata.
class Optimizer {
 public:
  explicit Optimizer(ValueTypes& types);

  /// Recurses from graph outputs through producers, attaching constraints.
  void optimizeGraph(nativert::Graph* graph);

  /// Visits outputs of a subgraph root, reusing already-visited state.
  void optimizeNode(const nativert::Node* node);

 private:
  void visitValue(const nativert::Value* value);
  void ensureConstraint(int32_t id);

  ValueTypes& types_;
  nativert::Graph* graph_{nullptr};
  std::unordered_set<const nativert::Node*> visited_;
};

} // namespace torch::wave
