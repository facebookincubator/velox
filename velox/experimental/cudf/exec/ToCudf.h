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

#include "velox/exec/Operator.h"

#include <gflags/gflags.h>

DECLARE_bool(velox_cudf_enabled);
DECLARE_string(velox_cudf_memory_resource);
DECLARE_bool(velox_cudf_debug);
DECLARE_bool(velox_cudf_table_scan);

namespace facebook::velox::cudf_velox {

static const std::string kCudfAdapterName = "cuDF";

class CompileState {
 public:
  CompileState(
      const velox::exec::DriverFactory& driverFactory,
      velox::exec::Driver& driver,
      std::vector<velox::core::PlanNodePtr>& planNodes)
      : driverFactory_(driverFactory), driver_(driver), planNodes_(planNodes) {}

  velox::exec::Driver& driver() {
    return driver_;
  }

  // Replaces sequences of Operators in the Driver given at construction with
  // cuDF equivalents. Returns true if the Driver was changed.
  bool compile();

  const velox::exec::DriverFactory& driverFactory_;
  velox::exec::Driver& driver_;
  const std::vector<velox::core::PlanNodePtr>& planNodes_;
};

struct CudfOptions {
  bool cudfEnabled = FLAGS_velox_cudf_enabled;
  std::string cudfMemoryResource = FLAGS_velox_cudf_memory_resource;
  static CudfOptions defaultOptions() {
    return CudfOptions();
  }
};

/// Registers adapter to add cuDF operators to Drivers.
void registerCudf(const CudfOptions& options = CudfOptions::defaultOptions());
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
