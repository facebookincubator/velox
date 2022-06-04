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

#include "velox/common/memory/HashStringAllocator.h"
#include "velox/core/PlanNode.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::exec {

class WindowFunction {
 public:
  explicit WindowFunction(TypePtr resultType) : resultType_(resultType) {}

  virtual ~WindowFunction(){};

  TypePtr resultType() const {
    return resultType_;
  }

  // rows is a buffer of the all the rows in this partition
  virtual void resetPartition(const std::vector<char*>& rows) = 0;

  // peerGroupStarts, peerGroupEnds, frameStarts and frameEnds are the indexes
  // of the peer and frame start and end rows within the partition block.
  virtual void apply(
      int32_t peerGroupStarts,
      int32_t peerGroupEnds,
      int32_t frameStarts,
      int32_t frameEnds,
      int32_t currentOutputRow,
      const std::vector<VectorPtr>& argVectors,
      const VectorPtr& result) = 0;

  static std::unique_ptr<WindowFunction> create(
      const std::string& name,
      const std::vector<TypePtr>& argTypes,
      const TypePtr& resultType);

 protected:
  const TypePtr resultType_;
};

using WindowFunctionFactory = std::function<std::unique_ptr<WindowFunction>(
    const std::vector<TypePtr>& argTypes,
    const TypePtr& resultType)>;

/// Register a window function with the specified name and signatures.
bool registerWindowFunction(
    const std::string& name,
    std::vector<std::shared_ptr<FunctionSignature>> signatures,
    WindowFunctionFactory factory);

/// Returns signatures of the window function with the specified name.
/// Returns empty std::optional if function with that name is not found.
std::optional<std::vector<std::shared_ptr<FunctionSignature>>>
getWindowFunctionSignatures(const std::string& name);

struct WindowFunctionEntry {
  std::vector<std::shared_ptr<FunctionSignature>> signatures;
  WindowFunctionFactory factory;
};

using WindowFunctionMap = std::unordered_map<std::string, WindowFunctionEntry>;

WindowFunctionMap& windowFunctions();
} // namespace facebook::velox::exec