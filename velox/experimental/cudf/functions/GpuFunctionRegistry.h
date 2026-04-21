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

#include <cudf/types.hpp>

#include <functional>
#include <memory>
#include <mutex>
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

class GpuVectorFunction;

using GpuFunctionFactory =
    std::function<std::unique_ptr<GpuVectorFunction>()>;

class GpuFunctionRegistry {
 public:
  static GpuFunctionRegistry& instance();

  void registerFunction(GpuFunctionKey key, GpuFunctionFactory factory);

  GpuVectorFunction* resolveFunction(const GpuFunctionKey& key) const;

  bool hasFunction(const GpuFunctionKey& key) const;

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
};

} // namespace facebook::velox::gpu
