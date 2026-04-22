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

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox::gpu {

struct GpuFunctionKey {
  std::string name;
  cudf::type_id returnType;
  std::vector<cudf::type_id> argTypes;

  bool operator==(const GpuFunctionKey& other) const {
    return name == other.name && returnType == other.returnType &&
        argTypes == other.argTypes;
  }
};

class GpuVectorFunction;

// Mirrors Velox's VectorFunctionArg: describes one argument to a GPU function.
// For constant arguments (e.g. pattern in LIKE, values in IN), constantValue
// holds a cuDF column containing the constant data on device.
// For scalar string constants (e.g. LIKE pattern), constantString stores the
// host-side value to avoid unnecessary D-to-H transfers.
struct GpuFunctionArg {
  cudf::type_id type;
  std::shared_ptr<cudf::column> constantValue;
  std::optional<std::string> constantString;
};

// Mirrors Velox's VectorFunctionFactory: receives the function name and
// argument descriptors (including constants) at compile time, returns a
// GpuVectorFunction instance with constants baked in.
using GpuStatefulFunctionFactory = std::function<
    std::unique_ptr<GpuVectorFunction>(
        const std::string& name,
        const std::vector<GpuFunctionArg>& inputArgs)>;

struct GpuFunctionKeyHash {
  size_t operator()(const GpuFunctionKey& key) const {
    size_t h = std::hash<std::string>{}(key.name);
    h ^= std::hash<int>{}(static_cast<int>(key.returnType)) + 0x9e3779b9 +
        (h << 6) + (h >> 2);
    for (auto t : key.argTypes) {
      h ^= std::hash<int>{}(static_cast<int>(t)) + 0x9e3779b9 + (h << 6) +
          (h >> 2);
    }
    return h;
  }
};

using GpuFunctionFactory =
    std::function<std::unique_ptr<GpuVectorFunction>()>;

class GpuFunctionRegistry {
 public:
  static GpuFunctionRegistry& instance();

  void registerFunction(GpuFunctionKey key, GpuFunctionFactory factory);

  GpuVectorFunction* resolveFunction(const GpuFunctionKey& key) const;

  bool hasFunction(const GpuFunctionKey& key) const;

  // Stateful function support: register a factory keyed by name only.
  // The factory receives constant argument descriptors and produces a
  // per-expression function instance with constants baked in.
  void registerStatefulFunction(
      const std::string& name,
      GpuStatefulFunctionFactory factory);

  // Try to create a stateful function instance. Returns nullptr if no
  // stateful factory is registered for the given name.
  std::unique_ptr<GpuVectorFunction> resolveStatefulFunction(
      const std::string& name,
      const std::vector<GpuFunctionArg>& inputArgs) const;

  bool hasStatefulFunction(const std::string& name) const;

  size_t size() const;

  void clear();

 private:
  GpuFunctionRegistry() = default;

  mutable std::mutex mu_;
  std::unordered_map<
      GpuFunctionKey,
      std::unique_ptr<GpuVectorFunction>,
      GpuFunctionKeyHash>
      resolved_;
  std::unordered_map<GpuFunctionKey, GpuFunctionFactory, GpuFunctionKeyHash>
      factories_;
  std::unordered_map<std::string, GpuStatefulFunctionFactory>
      statefulFactories_;
};

} // namespace facebook::velox::gpu
