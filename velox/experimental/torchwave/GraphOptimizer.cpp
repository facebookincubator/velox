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

// Returns the dtype that C++ would produce for a binary operation between
// two types, following the standard arithmetic conversions.
c10::ScalarType cppPromoteTypes(c10::ScalarType lhs, c10::ScalarType rhs) {
  bool lhsFloat = c10::isFloatingType(lhs);
  bool rhsFloat = c10::isFloatingType(rhs);
  if (lhsFloat && rhsFloat) {
    return c10::elementSize(lhs) >= c10::elementSize(rhs) ? lhs : rhs;
  }
  if (lhsFloat) {
    return lhs;
  }
  if (rhsFloat) {
    return rhs;
  }
  return c10::elementSize(lhs) >= c10::elementSize(rhs) ? lhs : rhs;
}

// Returns the narrowest float type whose mantissa can represent the given
// integral type without precision loss.
c10::ScalarType minFloatForIntegral(c10::ScalarType intType) {
  switch (intType) {
    case c10::ScalarType::Bool:
    case c10::ScalarType::Byte:
    case c10::ScalarType::Char:
      // 10-bit mantissa (Half) covers 8-bit integers.
      return c10::ScalarType::Half;
    case c10::ScalarType::Short:
      // 23-bit mantissa (Float) covers 16-bit integers.
      return c10::ScalarType::Float;
    case c10::ScalarType::Int:
    case c10::ScalarType::Long:
      // 52-bit mantissa (Double) is needed for 32/64-bit integers.
      return c10::ScalarType::Double;
    default:
      return c10::ScalarType::Float;
  }
}

// Promotes types for arithmetic so that the result float type is wide enough
// to represent the integer operand without precision loss.  For example
// Long * Float promotes to Double because Float's 23-bit mantissa cannot
// represent all int64 values.
c10::ScalarType arithmeticPromoteTypes(
    c10::ScalarType lhs,
    c10::ScalarType rhs) {
  auto base = c10::promoteTypes(lhs, rhs);
  bool lhsInt = c10::isIntegralType(lhs, /*includeBool=*/true);
  bool rhsInt = c10::isIntegralType(rhs, /*includeBool=*/true);
  if (lhsInt && !rhsInt) {
    auto minFloat = minFloatForIntegral(lhs);
    if (c10::elementSize(minFloat) > c10::elementSize(base)) {
      return minFloat;
    }
  } else if (rhsInt && !lhsInt) {
    auto minFloat = minFloatForIntegral(rhs);
    if (c10::elementSize(minFloat) > c10::elementSize(base)) {
      return minFloat;
    }
  }
  return base;
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
      auto cppType =
          cppPromoteTypes(tensorInputs[0].second, tensorInputs[1].second);
      if (pytorchType != cppType) {
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
