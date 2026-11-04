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

namespace facebook::velox::connector {
class ConnectorQueryCtx;
} // namespace facebook::velox::connector

namespace facebook::velox::cudf_velox {

class CompileState {
 public:
  CompileState(const exec::DriverFactory& driverFactory, exec::Driver& driver)
      : driverFactory_(driverFactory), driver_(driver) {}

  exec::Driver& driver() {
    return driver_;
  }

  // Get plan node by id lookup (asserts the id exists).
  core::PlanNodePtr getPlanNode(const core::PlanNodeId& id) const;

  // Resolve the plan node for an operator. Falls back to
  // driverFactory_.consumerNode for operators like CallbackSink whose
  // planNodeId is "N/A". May return nullptr.
  core::PlanNodePtr resolveOperatorPlanNode(const exec::Operator* op) const;

  // Replaces sequences of Operators in the Driver given at construction with
  // cuDF equivalents. Returns true if the Driver was changed.
  bool compile(bool allow_cpu_fallback);

  const exec::DriverFactory& driverFactory_;
  exec::Driver& driver_;
};

/// Registers adapter to add cuDF operators to Drivers.
void registerCudf();
void unregisterCudf();

/// Returns true if cuDF is registered.
bool cudfIsRegistered();

/// Returns true when the query behind 'connectorQueryCtx' may use cuDF: cuDF
/// is registered in this process and the query did not turn it off through the
/// 'cudf.enabled' (CudfConfig::kCudfEnabled) session property of the catalog
/// being read. An absent value, which includes a null 'connectorQueryCtx',
/// means the process-wide CudfConfig default.
///
/// A cuDF connector has to ask this before handing out a GPU data source,
/// because the data source and the driver adapter must reach the same answer:
/// a query the adapter leaves on the CPU cannot be fed CudfVectors. The two
/// read the same switch, the adapter from the query config and the connector
/// from the catalog's session properties, which is where the engine copies it
/// to -- ConnectorQueryCtx has no access to the query config.
bool cudfIsEnabled(
    const ::facebook::velox::connector::ConnectorQueryCtx* connectorQueryCtx);

} // namespace facebook::velox::cudf_velox
