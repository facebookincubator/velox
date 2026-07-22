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

#include "velox/experimental/cudf/exec/CudfPlanRewriter.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

namespace facebook::velox::cudf_velox::test {

inline core::PlanNodePtr rewriteToCudfPlan(
    const core::PlanNodePtr& plan,
    int gpuDriverCount = 32,
    std::shared_ptr<core::QueryCtx> queryCtx = nullptr) {
  VELOX_CHECK(
      cudfIsRegistered(),
      "cuDF must be registered before rewriting a plan for cuDF execution");
  CudfPlanRewriter::Config config;
  config.gpuDriverCount = gpuDriverCount;
  config.addLocalPartitionBoundaries = false;
  config.queryCtx = std::move(queryCtx);
  return CudfPlanRewriter::rewrite(plan, config);
}

} // namespace facebook::velox::cudf_velox::test
