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

#include <c10/core/ScalarType.h>

namespace torch::wave {

namespace {

// Promotes types for tensor-tensor arithmetic to match PyTorch.
c10::ScalarType arithmeticPromoteTypes(
    c10::ScalarType lhs,
    c10::ScalarType rhs) {
  // Match PyTorch's tensor-tensor type promotion exactly. PyTorch promotes a
  // mix of an integral and a floating tensor to the floating type (e.g.
  // int64 * float32 -> float32); it does NOT widen to a float that can
  // represent the integer exactly. An earlier heuristic upcast int*float to
  // double (via minFloatForIntegral), which diverged from eager: a threshold
  // like int64*float computed in double lands just above an integer and flips
  // a `>=` comparison that eager (float32) evaluates as equal.
  // c10::promoteTypes is the same function PyTorch uses, so use it directly.
  return c10::promoteTypes(lhs, rhs);
}

} // namespace

Optimizer::Optimizer(WaveGraph& waveGraph)
    : waveGraph_(waveGraph), types_(waveGraph.types()) {}

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
      types_.constraints[id].rank =
          static_cast<int8_t>(types_.types[id]->dim());
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
        types_.constraints.at(outputId) = outputConstraints.at(i);
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

  // Insert casts where PyTorch type promotion differs from C++ rules.
  if (metadata->arithmeticPromotion) {
    // Collect dtypes of tensor inputs.
    std::vector<std::pair<size_t, c10::ScalarType>> tensorInputs;
    for (size_t i = 0; i < producer->inputs().size(); ++i) {
      auto inputId = producer->inputs()[i].value->id();
      if (inputId >= 0 && static_cast<size_t>(inputId) < types_.types.size() &&
          types_.types[inputId]) {
        tensorInputs.push_back({i, types_.types[inputId]->dtype()});
      }
    }
    if (tensorInputs.size() >= 2) {
      auto pytorchType = arithmeticPromoteTypes(
          tensorInputs[0].second, tensorInputs[1].second);
      // Cast any operand whose dtype differs from PyTorch's promoted type so
      // the op is computed in exactly that type (e.g. int64 * float32 -> cast
      // the int64 to float32, then multiply in float32, matching eager). The
      // generated elementwise call would otherwise compute a mixed-type product
      // in the wrong precision.
      bool needsCast = false;
      for (const auto& [ordinal, inputDtype] : tensorInputs) {
        if (inputDtype != pytorchType) {
          needsCast = true;
          break;
        }
      }
      if (needsCast) {
        auto* mutableProducer = const_cast<nativert::Node*>(producer);
        for (auto& [ordinal, inputDtype] : tensorInputs) {
          if (inputDtype == pytorchType) {
            continue;
          }
          auto* originalValue = mutableProducer->inputs()[ordinal].value;
          auto* castNode = graph_->createNode(
              "torch.ops.aten.to.dtype", {{"self", originalValue}});
          castNode->addAttribute(
              {"dtype", std::string(c10::toString(pytorchType))});
          graph_->insertBefore(castNode, mutableProducer);
          auto* castOutput = waveGraph_.newTensorValue(
              castNode,
              std::string(originalValue->name()) + "_promoted",
              pytorchType);
          mutableProducer->inputs()[ordinal].value = castOutput;
        }
        // Set the output type of the arithmetic node to the promoted type.
        for (auto* output : producer->outputs()) {
          waveGraph_.registerTensorMeta(output, pytorchType);
        }
      }
    }
  }

  if (metadata->maybeReplace) {
    auto oldTarget = producer->target();
    auto replacements = metadata->maybeReplace(producer, types_, waveGraph_);
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
    for (auto& [oldValue, newValue] : replacements) {
      visitValue(newValue);
    }
  }
}

} // namespace torch::wave
