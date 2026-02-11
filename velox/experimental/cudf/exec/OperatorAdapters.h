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

#include "velox/core/PlanNode.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"

#include <memory>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Base interface for operator adapters that handle GPU operator
/// replacement.
///
/// Each adapter is responsible for:
/// - Determining if it can handle a specific operator type
/// - Checking if the operator is supported for GPU execution
/// - Creating GPU replacement operators
/// - Providing metadata about GPU input/output capabilities
class OperatorAdapter {
 public:
  explicit OperatorAdapter(std::string name) : name_(std::move(name)) {}
  virtual ~OperatorAdapter() = default;

  /// Check if this adapter can handle the given operator type. Returns true
  /// if this adapter handles this operator type.
  virtual bool canHandle(const exec::Operator* op) const = 0;

  /// Check if the operator is supported for GPU execution. Returns true if
  /// the operator can be executed on GPU.
  virtual bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx) const = 0;

  /// Check if this operator accepts GPU input. Returns true if the operator
  /// can accept GPU vectors as input.
  virtual bool acceptsGpuInput() const = 0;

  /// Check if this operator produces GPU output. Returns true if the operator
  /// produces GPU vectors as output.
  virtual bool producesGpuOutput() const = 0;

  /// Bundled GPU capability properties for an operator.
  struct Properties {
    bool canRunOnGPU = false;
    bool acceptsGpuInput = false;
    bool producesGpuOutput = false;
  };

  /// Query all GPU capability properties at once for the given operator.
  /// acceptsGpuInput and producesGpuOutput are AND-gated with canRunOnGPU.
  Properties properties(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx) const {
    Properties props;
    props.canRunOnGPU = canRunOnGPU(op, planNode, ctx);
    props.acceptsGpuInput = props.canRunOnGPU && acceptsGpuInput();
    props.producesGpuOutput = props.canRunOnGPU && producesGpuOutput();
    return props;
  }

  /// Create replacement GPU operator(s). Returns a vector of replacement
  /// operators (empty if operator should be kept).
  virtual std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const = 0;

  /// Check if the original operator should be kept (not replaced). Returns
  /// true if the original operator should be kept, false otherwise.
  virtual bool keepOperator() const {
    return false;
  }

  /// Get the name of this adapter for debugging.
  const std::string& name() const {
    return name_;
  }

 private:
  const std::string name_;
};

/// Registry for operator adapters.
///
/// Manages the collection of operator adapters and provides lookup
/// functionality to find the appropriate adapter for a given operator.
class OperatorAdapterRegistry {
 public:
  /// Get the singleton instance of the registry.
  static OperatorAdapterRegistry& getInstance();

  /// Register an adapter with the registry.
  void registerAdapter(std::unique_ptr<OperatorAdapter> adapter);

  /// Find an adapter that can handle the given operator. Returns a pointer
  /// to the adapter, or nullptr if none found.
  const OperatorAdapter* findAdapter(const exec::Operator* op) const;

  /// Get all registered adapters.
  const std::vector<std::unique_ptr<OperatorAdapter>>& getAdapters() const;

  /// Clear all registered adapters.
  void clear();

 private:
  OperatorAdapterRegistry() = default;
  std::vector<std::unique_ptr<OperatorAdapter>> adapters_;
};

/// Register all operator adapters.
///
/// This function should be called from registerCudf() to register all
/// operator adapters with the registry.
void registerAllOperatorAdapters();

} // namespace facebook::velox::cudf_velox
