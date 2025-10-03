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

#include "velox/experimental/cudf-exchange/CudfExchangeClient.h"

#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"

#include <gflags/gflags.h>

DECLARE_bool(velox_cudf_enabled);
DECLARE_string(velox_cudf_memory_resource);
DECLARE_bool(velox_cudf_debug);
DECLARE_bool(velox_cudf_table_scan);
DECLARE_bool(velox_cudf_exchange);

namespace facebook::velox::cudf_velox {

static const std::string kCudfAdapterName = "cuDF";

class CompileState {
 public:
  CompileState(const exec::DriverFactory& driverFactory, exec::Driver& driver)
      : driverFactory_(driverFactory), driver_(driver) {}

  exec::Driver& driver() {
    return driver_;
  }

  // Replaces sequences of Operators in the Driver given at construction with
  // cuDF equivalents. Returns true if the Driver was changed.
  bool compile();

  std::shared_ptr<cudf_exchange::CudfExchangeClient> createCudfExchangeClient(
      const core::PlanNodeId& planNodeId,
      const std::string& taskId,
      const int destination,
      const int32_t numberOfConsumers,
      folly::Executor* executor);

  const exec::DriverFactory& driverFactory_;
  exec::Driver& driver_;
  std::unordered_map<
      core::PlanNodeId,
      std::shared_ptr<cudf_exchange::CudfExchangeClient>>
      cudfExchangeClientByPlanNode_;
};

class CudfOptions {
 public:
  static CudfOptions& getInstance() {
    static CudfOptions instance;
    return instance;
  }

  void setPrefix(const std::string& prefix) {
    prefix_ = prefix;
  }

  const std::string& prefix() const {
    return prefix_;
  }
  void setShouldTransformLastOutput(bool newValue) {
    transformLastOutput_ = newValue;
  }

  const bool shouldTransformLastOutput() const {
    return transformLastOutput_;
  }

  const bool cudfEnabled;
  const std::string cudfMemoryResource;
  const bool cudfTableScan;
  const bool cudfExchange;
  // The initial percent of GPU memory to allocate for memory resource for one
  // thread.
  int memoryPercent;

 private:
  CudfOptions()
      : cudfEnabled(FLAGS_velox_cudf_enabled),
        cudfMemoryResource(FLAGS_velox_cudf_memory_resource),
        cudfTableScan(FLAGS_velox_cudf_table_scan),
        cudfExchange(FLAGS_velox_cudf_exchange),
        memoryPercent(50),
        prefix_(""),
        transformLastOutput_(false) {}
  CudfOptions(const CudfOptions&) = delete;
  CudfOptions& operator=(const CudfOptions&) = delete;
  std::string prefix_;
  bool transformLastOutput_;
};

/// Registers adapter to add cuDF operators to Drivers.
void registerCudf(const CudfOptions& options = CudfOptions::getInstance());
void unregisterCudf();

/// Returns true if cuDF is registered.
bool cudfIsRegistered();

/**
 * @brief Returns true if the velox_cudf_debug flag is set to true.
 */
bool cudfDebugEnabled();

/**
 * @brief Returns true if the velox_cudf_table_scan flag is set to true.
 */
bool cudfTableScanEnabled();

} // namespace facebook::velox::cudf_velox
