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

// Unified dispatch: tries native GPU-compiled function first, then falls
// back to cuDF built-in, returning nullptr if neither is available.
#pragma once

#include "velox/experimental/cudf/functions/CudfFallbackFunction.h"
#include "velox/experimental/cudf/functions/GpuFunctionRegistry.h"
#include "velox/experimental/cudf/functions/GpuVectorFunction.h"

#include <memory>

namespace facebook::velox::gpu {

enum class GpuDispatchKind {
  kNativeGpu,
  kCudfFallback,
  kNotFound,
};

struct GpuDispatchResult {
  GpuVectorFunction* function{nullptr};
  GpuDispatchKind kind{GpuDispatchKind::kNotFound};
  std::unique_ptr<GpuVectorFunction> owned;
};

GpuDispatchResult dispatchGpuFunction(
    const std::string& name,
    cudf::type_id returnType,
    const std::vector<cudf::type_id>& argTypes);

} // namespace facebook::velox::gpu
