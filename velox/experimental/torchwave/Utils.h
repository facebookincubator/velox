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

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <ATen/core/ivalue.h>
#include <aten/src/ATen/core/function_schema.h>
#include <torch/nativert/graph/Graph.h>

#include "velox/experimental/torchwave/Registry.h"

namespace torch::wave {

/// Rounds 'x' up to the next multiple of 'y'.
template <typename T, typename U>
inline T roundUp(T x, U y) {
  return ((x + y - 1) / y) * y;
}

/// Parses a qualified op name (e.g. "torch.ops.aten.add.Tensor") and looks up
/// its FunctionSchema from the dispatcher. Returns nullptr if not found.
const c10::FunctionSchema* findFunctionSchema(std::string_view qualifiedName);

/// Converts a nativert::Constant to a human-readable string.
std::string constantToString(const nativert::Constant& c);

/// Converts a c10::IValue to a human-readable string.
std::string ivalueToString(const c10::IValue& value);

/// Returns the CUDA type string for a c10::ScalarType.
std::string cudaTypeString(c10::ScalarType dtype);

/// Visits sorted attributes of each node reachable from 'node' in dependency
/// order. Skips values in 'inputs' and already-visited nodes. Calls
/// func(node, attr) for each attribute in alphabetical order per node.
template <typename Func>
void forEachSortedAttribute(
    NodeCP node,
    const std::unordered_set<ValueCP>& inputs,
    std::unordered_set<NodeCP>& visited,
    Func&& func) {
  if (!visited.insert(node).second) {
    return;
  }
  for (const auto& input : node->inputs()) {
    if (inputs.count(input.value)) {
      continue;
    }
    auto* producer = input.value->producer();
    if (producer) {
      forEachSortedAttribute(producer, inputs, visited, func);
    }
  }
  const auto& attrs = node->attributes();
  if (attrs.empty()) {
    return;
  }
  std::vector<const nativert::Attribute*> sorted;
  sorted.reserve(attrs.size());
  for (const auto& attr : attrs) {
    sorted.push_back(&attr);
  }
  std::sort(sorted.begin(), sorted.end(), [](const auto* lhs, const auto* rhs) {
    return lhs->name < rhs->name;
  });
  for (const auto* attr : sorted) {
    func(node, *attr);
  }
}

/// RAII timer that prints elapsed microseconds on destruction.
class Timer {
 public:
  explicit Timer(const char* label, bool enable = true)
      : label_(label), enable_(enable) {
    if (enable_) {
      start_ = std::chrono::high_resolution_clock::now();
    }
  }

  ~Timer() {
    if (enable_) {
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start_)
                    .count();
      printf("%s %ld us\n", label_, us);
    }
  }

  Timer(const Timer&) = delete;
  Timer& operator=(const Timer&) = delete;

 private:
  const char* label_;
  bool enable_;
  std::chrono::high_resolution_clock::time_point start_;
};

} // namespace torch::wave
