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
#include "velox/exec/RowContainer.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::exec {

class WindowFunction {
 public:
  explicit WindowFunction(TypePtr resultType, velox::memory::MemoryPool* pool)
      : resultType_(std::move(resultType)), pool_(pool) {}

  virtual ~WindowFunction() = default;

  const TypePtr& resultType() const {
    return resultType_;
  }

  velox::memory::MemoryPool* pool() const {
    return pool_;
  }

  /// Rows is a list of pointers to individual rows in the partition.
  /// The rows are sorted as specified by the ORDER BY clause.
  virtual void resetPartition(const folly::Range<char**>& rows) = 0;

  /// This function is invoked by the Window Operator to compute
  /// the window function for a batch of rows.
  /// @peerGroupStarts : A buffer of the indexes of rows at which the
  /// peer group of the current row starts. Rows are peers if they
  /// have the same value in the partition ordering.
  /// @peerGroupEnds : A buffer of the indexes of rows at which the
  /// peer group fo the current row ends.
  /// @frameStarts : A buffer of the indexes of rows at which the
  /// frame for the current row starts.
  /// @frameEnds : A buffer of the indexes of rows at which the frame
  /// for the current row ends.
  /// @resultOffset
  /// @result : The vector of result values to be populated.
  virtual void apply(
      const BufferPtr& peerGroupStarts,
      const BufferPtr& peerGroupEnds,
      const BufferPtr& frameStarts,
      const BufferPtr& frameEnds,
      vector_size_t resultOffset,
      const VectorPtr& result) = 0;

  static std::unique_ptr<WindowFunction> create(
      const std::string& name,
      const std::vector<RowColumn>& argColumns,
      const std::vector<TypePtr>& argTypes,
      const TypePtr& resultType,
      velox::memory::MemoryPool* pool);

 protected:
  const TypePtr resultType_;
  velox::memory::MemoryPool* pool_;
};

using WindowFunctionFactory = std::function<std::unique_ptr<WindowFunction>(
    const std::vector<RowColumn>& argColumns,
    const std::vector<TypePtr>& argTypes,
    const TypePtr& resultType,
    velox::memory::MemoryPool* pool)>;

/// Register a window function with the specified name and signatures.
bool registerWindowFunction(
    const std::string& name,
    std::vector<FunctionSignaturePtr> signatures,
    WindowFunctionFactory factory);

/// Returns signatures of the window function with the specified name.
/// Returns empty std::optional if function with that name is not found.
std::optional<std::vector<FunctionSignaturePtr>> getWindowFunctionSignatures(
    const std::string& name);

struct WindowFunctionEntry {
  std::vector<FunctionSignaturePtr> signatures;
  WindowFunctionFactory factory;
};

using WindowFunctionMap = std::unordered_map<std::string, WindowFunctionEntry>;

WindowFunctionMap& windowFunctions();
} // namespace facebook::velox::exec