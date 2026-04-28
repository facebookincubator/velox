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
#include "velox/experimental/wave/common/Cuda.h"

namespace torch::wave {

struct Metadata;
const Metadata* nodeMeta(NodeCP node);

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

/// Returns the CUDA type string for a ScalarType name string like
/// "ScalarType::Long".
std::string cudaTypeFromScalarTypeName(const std::string& name);

/// Returns the CUDA type string from a dtype attribute, which may hold a
/// std::string (e.g. "Float") or a c10::ScalarType enum.
std::string cudaTypeFromDtype(const nativert::Attribute& attr);

/// Returns the ScalarType name (e.g. "Float") from a dtype attribute, which
/// may hold a std::string or a c10::ScalarType enum.
std::string dtypeName(const nativert::Attribute& attr);

/// Returns an identifier-safe suffix for a c10::ScalarType, e.g.
/// Float → "Float", Int → "Int32", Long → "Int64".
std::string cudaTypeIdSuffix(c10::ScalarType dtype);

/// Visits sorted attributes of each node reachable from 'node' in dependency
/// order. Skips values in 'inputs' and already-visited nodes. Calls
/// func(node, attr) for each attribute in alphabetical order per node.
// Calls 'func(node, attr)' for each non-string, non-device attribute of
// 'node', sorted by name.
template <typename Func>
void forEachSortedAttribute(NodeCP node, Func&& func) {
  const auto& attrs = node->attributes();
  if (attrs.empty()) {
    return;
  }
  auto* meta = nodeMeta(node);
  std::vector<const nativert::Attribute*> sorted;
  sorted.reserve(attrs.size());
  for (const auto& attr : attrs) {
    if (std::holds_alternative<std::string>(attr.value) ||
        std::holds_alternative<c10::Device>(attr.value) ||
        attr.name == "dtype" || attr.name == "layout" ||
        attr.name == "pin_memory") {
      continue;
    }
    if (meta && !meta->shapeAttr.empty() && attr.name == meta->shapeAttr) {
      continue;
    }
    if (meta && !meta->ignoreAttrs.empty() &&
        std::find(
            meta->ignoreAttrs.begin(), meta->ignoreAttrs.end(), attr.name) !=
            meta->ignoreAttrs.end()) {
      continue;
    }
    if (meta && !meta->templateAttrs.empty() &&
        std::find(
            meta->templateAttrs.begin(),
            meta->templateAttrs.end(),
            attr.name) != meta->templateAttrs.end()) {
      continue;
    }
    sorted.push_back(&attr);
  }
  std::sort(sorted.begin(), sorted.end(), [](const auto* lhs, const auto* rhs) {
    return lhs->name < rhs->name;
  });
  for (const auto* attr : sorted) {
    func(node, *attr);
  }
}

// Recursive variant: walks the dependency graph depth-first (dependencies
// before the node itself), skipping values in 'inputs' and already-visited
// nodes.
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
  forEachSortedAttribute(node, func);
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

/// Sets device attributes on all nodes in 'graph' to CUDA (current device) or
/// CPU. Uses nativert's Placement API to rewrite c10::Device attributes.
void setGraphDevice(nativert::Graph* graph, bool isCuda);

} // namespace torch::wave
