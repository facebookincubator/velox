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

#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::cudf_velox {

class CompileState {
 public:
  CompileState(
      const exec::DriverFactory& driverFactory,
      exec::Driver& driver,
      std::vector<std::shared_ptr<core::PlanNode const>>& planNodes)
      : driverFactory_(driverFactory), driver_(driver), planNodes_(planNodes) {}

  exec::Driver& driver() {
    return driver_;
  }

  // Replaces sequences of Operators in the Driver given at construction with
  // cuDF equivalents. Returns true if the Driver was changed.
  bool compile();

  const exec::DriverFactory& driverFactory_;
  exec::Driver& driver_;
  const std::vector<std::shared_ptr<core::PlanNode const>>& planNodes_;
};

/// Registers adapter to add cuDF operators to Drivers.
void registerCudf();
void unregisterCudf();

/// Returns true if cuDF is registered.
bool cudfIsRegistered();

} // namespace facebook::velox::cudf_velox
