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

#include "velox/experimental/torchwave/GraphOptimizer.h"

namespace torch::wave {

Optimizer::Optimizer(ValueTypes& types) : types_(types) {}

void Optimizer::ensureConstraint(int32_t id) {
  if (id >= 0 && static_cast<size_t>(id) >= types_.constraints.size()) {
    types_.constraints.resize(id + 1);
  }
}

void Optimizer::optimizeGraph(nativert::Graph* graph) {
  graph_ = graph;
  types_.constraints.resize(types_.types.size());
  auto outputs = graph->outputs();
  for (auto* output : outputs) {
    visitValue(output);
  }
}

void Optimizer::optimizeNode(const nativert::Node* node) {
  types_.constraints.resize(types_.types.size());
  if (node->outputs().empty()) {
    for (const auto& input : node->inputs()) {
      visitValue(input.value);
    }
  } else {
    for (auto* output : node->outputs()) {
      visitValue(output);
    }
  }
}

void Optimizer::visitValue(const nativert::Value* value) {
  auto* producer = value->producer();
  if (!producer) {
    auto id = value->id();
    if (id >= 0 && static_cast<size_t>(id) < types_.types.size() &&
        types_.types[id]) {
      ensureConstraint(id);
      types_.constraints[id].rank = types_.types[id]->dim();
    }
    return;
  }

  if (!visited_.insert(producer).second) {
    return;
  }

  // Recurse into producer's inputs first.
  for (const auto& input : producer->inputs()) {
    visitValue(input.value);
  }

  // On return, set output constraints from Metadata if available.
  auto* metadata = Registry::metadata(producer->target());
  if (!metadata) {
    return;
  }
  if (metadata->rankArgument.has_value()) {
    auto ordinal = metadata->rankArgument.value();
    const auto& inputs = producer->inputs();
    TORCH_CHECK(
        ordinal >= 0 && static_cast<size_t>(ordinal) < inputs.size(),
        "rankArgument ordinal out of range: ",
        ordinal);
    auto inputRank = types_.rank(inputs[ordinal].value);
    for (auto* output : producer->outputs()) {
      auto outputId = output->id();
      if (outputId >= 0) {
        ensureConstraint(outputId);
        types_.constraints[outputId].rank = inputRank;
      }
    }
  } else if (metadata->outputConstraints) {
    auto outputConstraints = metadata->outputConstraints(producer, types_);
    const auto& outputs = producer->outputs();
    for (size_t i = 0; i < outputs.size() && i < outputConstraints.size();
         ++i) {
      auto outputId = outputs[i]->id();
      if (outputId >= 0) {
        ensureConstraint(outputId);
        types_.constraints[outputId] = outputConstraints[i];
      }
    }
  } else if (metadata->elementwise) {
    int8_t maxRank = -1;
    for (const auto& input : producer->inputs()) {
      auto inputId = input.value->id();
      if (inputId >= 0 &&
          static_cast<size_t>(inputId) < types_.constraints.size()) {
        maxRank = std::max(maxRank, types_.constraints[inputId].rank);
      }
    }
    for (auto* output : producer->outputs()) {
      auto outputId = output->id();
      if (outputId >= 0) {
        ensureConstraint(outputId);
        types_.constraints[outputId].rank = maxRank;
      }
    }
  }

  if (metadata->maybeReplace) {
    auto oldTarget = producer->target();
    auto replacements = metadata->maybeReplace(producer, types_);
    for (auto& [oldValue, newValue] : replacements) {
      graph_->replaceAllUses(
          const_cast<nativert::Value*>(oldValue),
          const_cast<nativert::Value*>(newValue));
    }
    if (producer->target() != oldTarget) {
      visited_.erase(producer);
      visitValue(value);
      return;
    }
  }
}

} // namespace torch::wave
