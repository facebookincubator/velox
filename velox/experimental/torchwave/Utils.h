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

#include "velox/experimental/wave/common/Cuda.h"

namespace torch::wave {

/// Rounds 'x' up to the next multiple of 'y'.
template <typename T, typename U>
inline T roundUp(T x, U y) {
  return ((x + y - 1) / y) * y;
}

/// Reusable pool of unique_ptr<T>. Thread-safe. Returns an existing item if
/// available, otherwise creates one via the make function passed at
/// construction.
template <typename T>
class Pool {
 public:
  using MakeFunc = std::function<std::unique_ptr<T>()>;

  explicit Pool(MakeFunc make) : make_(std::move(make)) {}

  std::unique_ptr<T> get() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!items_.empty()) {
      auto item = std::move(items_.back());
      items_.pop_back();
      return item;
    }
    return make_();
  }

  void put(std::unique_ptr<T> item) {
    std::lock_guard<std::mutex> lock(mutex_);
    items_.push_back(std::move(item));
  }

 private:
  std::mutex mutex_;
  std::vector<std::unique_ptr<T>> items_;
  MakeFunc make_;
};

using StreamPool = Pool<facebook::velox::wave::Stream>;
using EventPool = Pool<facebook::velox::wave::Event>;

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
    const nativert::Node* node,
    const std::unordered_set<const nativert::Value*>& inputs,
    std::unordered_set<const nativert::Node*>& visited,
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
  std::sort(sorted.begin(), sorted.end(), [](const auto* a, const auto* b) {
    return a->name < b->name;
  });
  for (const auto* attr : sorted) {
    func(node, *attr);
  }
}

} // namespace torch::wave
