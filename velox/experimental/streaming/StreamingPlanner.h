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
#include "velox/experimental/streaming/StreamingOperator.h"

namespace facebook::velox::core {
struct PlanFragment;
} // namespace facebook::velox::core

namespace facebook::velox::streaming {

class StreamingPlanner {

 public:
  // Create streaming operator chain according to plan.
  static StreamingOperatorPtr plan(
      const core::PlanFragment& planFragment,
      exec::DriverCtx* ctx);

 private:
  static std::unique_ptr<StreamingOperator> nodeToStreamingOperator(
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx);

  static std::unique_ptr<exec::Operator> nodeToOperator(
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx);
};
} // namespace facebook::velox::streaming
