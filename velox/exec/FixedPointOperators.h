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

#include <memory>

#include "velox/core/FixedPointPlanNodes.h"

namespace facebook::velox::exec {

struct DriverCtx;
class Operator;

/// Constructs execution operators for fixed-point plan nodes.
class FixedPointOperators {
 public:
  /// Creates an operator that reads the final output state of 'node'.
  static std::unique_ptr<Operator> create(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const core::FixedPointNodePtr& node);

  /// Creates an operator that reads the vector state named by 'node'.
  static std::unique_ptr<Operator> create(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const core::StateSourceNodePtr& node);

  /// Creates an operator that joins against the hash-table state named by
  /// 'node'.
  static std::unique_ptr<Operator> create(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const core::StateHashJoinNodePtr& node);
};

} // namespace facebook::velox::exec
