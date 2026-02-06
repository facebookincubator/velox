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

/**
 * @brief Base interface for operator adapters that handle GPU operator
 * replacement.
 *
 * Each adapter is responsible for:
 * - Determining if it can handle a specific operator type
 * - Checking if the operator is supported for GPU execution
 * - Creating GPU replacement operators
 * - Providing metadata about GPU input/output capabilities
 */
class OperatorAdapter {
 public:
  virtual ~OperatorAdapter() = default;

  /**
   * @brief Check if this adapter can handle the given operator type.
   * @param op The operator to check
   * @return true if this adapter handles this operator type
   */
  virtual bool canHandle(const exec::Operator* op) const = 0;

  /**
   * @brief Check if the operator is supported for GPU execution.
   * @param op The operator to check
   * @param planNode The plan node for this operator
   * @param ctx The driver context
   * @return true if the operator can be executed on GPU
   */
  virtual bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx) const = 0;

  /**
   * @brief Check if this operator accepts GPU input.
   * @return true if the operator can accept GPU vectors as input
   */
  virtual bool acceptsGpuInput() const = 0;

  /**
   * @brief Check if this operator produces GPU output.
   * @return true if the operator produces GPU vectors as output
   */
  virtual bool producesGpuOutput() const = 0;

  /**
   * @brief Create replacement GPU operator(s).
   * @param op The original operator
   * @param planNode The plan node for this operator
   * @param ctx The driver context
   * @param operatorId The operator ID to use
   * @return Vector of replacement operators (empty if operator should be kept)
   */
  virtual std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const = 0;

  /**
   * @brief Check if the original operator should be kept (not replaced).
   * @return 1 if the original operator should be kept, 0 otherwise
   */
  virtual int keepOperator() const {
    return 0;
  }

  /**
   * @brief Get the name of this adapter for debugging.
   * @return Human-readable adapter name
   */
  virtual std::string name() const = 0;
};

/**
 * @brief Registry for operator adapters.
 *
 * Manages the collection of operator adapters and provides lookup
 * functionality to find the appropriate adapter for a given operator.
 */
class OperatorAdapterRegistry {
 public:
  /**
   * @brief Get the singleton instance of the registry.
   * @return Reference to the registry instance
   */
  static OperatorAdapterRegistry& getInstance();

  /**
   * @brief Register an adapter with the registry.
   * @param adapter The adapter to register
   */
  void registerAdapter(std::unique_ptr<OperatorAdapter> adapter);

  /**
   * @brief Find an adapter that can handle the given operator.
   * @param op The operator to find an adapter for
   * @return Pointer to the adapter, or nullptr if none found
   */
  const OperatorAdapter* findAdapter(const exec::Operator* op) const;

  /**
   * @brief Get all registered adapters.
   * @return Reference to the vector of adapters
   */
  const std::vector<std::unique_ptr<OperatorAdapter>>& getAdapters() const;

  /**
   * @brief Clear all registered adapters.
   */
  void clear();

 private:
  OperatorAdapterRegistry() = default;
  std::vector<std::unique_ptr<OperatorAdapter>> adapters_;
};

/**
 * @brief Register all operator adapters.
 *
 * This function should be called from registerCudf() to register all
 * operator adapters with the registry.
 */
void registerAllOperatorAdapters();

} // namespace facebook::velox::cudf_velox
