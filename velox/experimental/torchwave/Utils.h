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
#include <utility>
#include <vector>

#include <ATen/core/ivalue.h>
#include <aten/src/ATen/core/function_schema.h>
#include <torch/nativert/graph/Graph.h>

#include "velox/experimental/torchwave/Registry.h"
#include "velox/experimental/wave/common/Cuda.h"

namespace torch::wave {

struct Metadata;
const Metadata* nodeMeta(NodeCP node);

std::vector<ValueCP> inputValues(NodeCP node);

std::unordered_set<ValueCP> inputValueSet(NodeCP node);

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

// Returns true if 'name' should be skipped during argument iteration based on
// the Metadata skip lists and well-known non-argument attributes.
inline bool isSkippedAttribute(const std::string& name, const Metadata* meta) {
  if (name == "dtype" || name == "layout" || name == "pin_memory" ||
      name == "device" || name == "memory_format") {
    return true;
  }
  if (meta && !meta->shapeAttr.empty() && name == meta->shapeAttr) {
    return true;
  }
  if (meta && !meta->ignoreAttrs.empty() &&
      std::find(meta->ignoreAttrs.begin(), meta->ignoreAttrs.end(), name) !=
          meta->ignoreAttrs.end()) {
    return true;
  }
  if (meta && !meta->templateAttrs.empty() &&
      std::find(meta->templateAttrs.begin(), meta->templateAttrs.end(), name) !=
          meta->templateAttrs.end()) {
    return true;
  }
  return false;
}

/// Iterates over the arguments of the node's function schema in declaration
/// order. For each argument, looks it up first in the node's inputs (by name),
/// then in the node's attributes. Skips arguments that match the
/// forEachSortedAttribute skip criteria. Calls func(schemaArgIndex, value,
/// attr) where value is non-null if the argument is a node input and attr is
/// non-null if it is an attribute.
template <typename Func>
void forArguments(const Metadata& meta, NodeCP node, Func&& func) {
  TORCH_CHECK(
      meta.functionSchema, "forArguments requires functionSchema on metadata");
  const auto& schemaArgs = meta.functionSchema->arguments();
  const auto& nodeInputs = node->inputs();
  for (const auto& input : nodeInputs) {
    bool found = false;
    for (const auto& arg : schemaArgs) {
      if (arg.name() == input.name) {
        found = true;
        break;
      }
    }
    TORCH_CHECK(
        found,
        "Node ",
        node->target(),
        " has input '",
        input.name,
        "' not in schema for ",
        meta.functionSchema->name());
  }
  for (size_t i = 0; i < schemaArgs.size(); ++i) {
    const auto& argName = schemaArgs[i].name();
    if (isSkippedAttribute(argName, &meta)) {
      continue;
    }
    ValueCP value = nullptr;
    for (const auto& input : nodeInputs) {
      if (input.name == argName) {
        value = input.value;
        break;
      }
    }
    if (value) {
      func(i, value, static_cast<const nativert::Attribute*>(nullptr));
      continue;
    }
    const auto* attr = node->tryGetAttribute(argName);
    if (attr) {
      if (std::holds_alternative<std::string>(attr->value) ||
          std::holds_alternative<c10::Device>(attr->value)) {
        continue;
      }
      func(i, static_cast<ValueCP>(nullptr), attr);
    }
  }
}

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
        std::holds_alternative<c10::Device>(attr.value)) {
      continue;
    }
    if (isSkippedAttribute(attr.name, meta)) {
      continue;
    }
    if (std::holds_alternative<nativert::None>(attr.value) && meta &&
        meta->isPresenceTemplateParam(attr.name)) {
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
  Timer(Timer&&) = delete;
  Timer& operator=(Timer&&) = delete;

 private:
  const char* label_;
  bool enable_;
  std::chrono::high_resolution_clock::time_point start_;
};

/// Sets device attributes on all nodes in 'graph' to CUDA (current device) or
/// CPU. Uses nativert's Placement API to rewrite c10::Device attributes.
void setGraphDevice(nativert::Graph* graph, bool isCuda);

/// Returns a trace string for a c10::IValue: "[d0, d1, ...]" for tensors,
/// the value for scalars, or "none" for None.
std::string traceIValue(const c10::IValue& value);

/// Formats a standalone node with output ids and input value refs, using
/// the current NodePrinter defaults.
std::string standaloneToString(NodeCP node);

/// Returns true if 'node' mutates 'value' in place. Checks the c10
/// FunctionSchema for write alias annotations on the argument that matches
/// 'value'. Falls back to checking for "_." in the target name.
bool isInPlaceMutation(NodeCP node, ValueCP value);

/// Returns a debug string showing up to 'maxElements' elements of a tensor
/// after flattening to 1-D, plus the shape and dtype. 0 means no limit.
std::string tensorDebugString(const at::Tensor& t, int32_t maxElements = 0);

/// Returns a string describing the first difference between two tensors,
/// or empty string if they match. Compares shapes first, then element values
/// with tolerance for floating point types.
std::string firstDifference(
    const at::Tensor& actual,
    const at::Tensor& expected);

/// Returns true if two tensors have the same dtype, shape, and values.
/// Floating point types use allclose with rtol=1e-4, atol=1e-5.
bool tensorsMatch(const at::Tensor& actual, const at::Tensor& expected);

/// Returns a full string representation of a tensor's contents.
std::string tensorToString(const at::Tensor& t);

/// Saves all non-empty tensor and scalar slots from an execution frame
/// as a map from ValueId to IValue. TensorList values are skipped but their
/// tensor contents are included.
void saveReferenceFrame(
    const nativert::ExecutionFrame& frame,
    const nativert::Graph& graph,
    const std::string& path);

/// Saves all non-empty tensor and scalar slots from an execution frame
/// by iterating slots 0..numValues-1.
void saveReferenceFrame(
    const nativert::ExecutionFrame& frame,
    int32_t numValues,
    const std::string& path);

/// Loads a reference frame from a .pt file into a map keyed by ValueId.
std::unordered_map<int32_t, c10::IValue> loadReferenceFrame(
    const std::string& path);

/// Like torch::jit::pickle_load but uses NamedTuple types registered via
/// registerNamedTuple for deserialization.
c10::IValue pickleLoadWithTypes(const std::vector<char>& data);

// Debug helpers callable from a debugger. Marked noinline/used to prevent
// the linker from stripping them.

/// Returns a trace string for the value at 'valueId' in 'frame'.
__attribute__((used, noinline)) std::string frameString(
    const nativert::ExecutionFrame& frame,
    int32_t valueId);

/// Returns a trace string for 'valueId' from the global reference frame
/// in WaveConfig, or "none" if no reference frame is set or the id is absent.
__attribute__((used, noinline)) std::string refFrameString(int32_t valueId);

/// Tracks which values have been traced during a single execution.
struct TraceState {
  // Value ids to trace, parsed from WaveConfig::traceValues.
  std::unordered_set<nativert::ValueId> valueIds;

  // Value ids that have already been printed in this execution.
  std::unordered_set<nativert::ValueId> traced;

  bool empty() const {
    return valueIds.empty();
  }

  bool shouldTrace(nativert::ValueId id) const {
    return valueIds.count(id) && !traced.count(id);
  }

  void markTraced(nativert::ValueId id) {
    traced.insert(id);
  }
};

/// Parses a comma-separated string of value ids into a TraceState.
TraceState parseTraceValues(const std::string& csv);

/// Prints tensor values for any ids in 'valueIds' that appear in the frame
/// and haven't been traced yet. Marks them as traced. Uses
/// WaveConfig::tensorPrintElementLimit.
void traceFrameValues(
    const std::string& label,
    const std::vector<nativert::ValueId>& valueIds,
    nativert::ExecutionFrame& frame,
    TraceState& traceState);

} // namespace torch::wave
