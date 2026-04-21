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
#include "velox/experimental/cudf/functions/GpuFunctionDispatch.h"

namespace facebook::velox::gpu {

GpuDispatchResult dispatchGpuFunction(
    const std::string& name,
    cudf::type_id returnType,
    const std::vector<cudf::type_id>& argTypes) {
  GpuFunctionKey key{name, returnType, argTypes};
  auto* fn = GpuFunctionRegistry::instance().resolveFunction(key);
  if (fn) {
    return {fn, GpuDispatchKind::kNativeGpu, nullptr};
  }

  auto& fallback = CudfFallbackRegistry::instance();

  if (argTypes.size() == 2) {
    auto op = fallback.findBinaryOp(name);
    if (op.has_value()) {
      auto owned = std::make_unique<CudfBinaryFunction>(
          *op, cudf::data_type{returnType});
      auto* ptr = owned.get();
      return {ptr, GpuDispatchKind::kCudfFallback, std::move(owned)};
    }
  }

  if (argTypes.size() == 1) {
    auto op = fallback.findUnaryOp(name);
    if (op.has_value()) {
      auto owned = std::make_unique<CudfUnaryFunction>(*op);
      auto* ptr = owned.get();
      return {ptr, GpuDispatchKind::kCudfFallback, std::move(owned)};
    }
  }

  return {nullptr, GpuDispatchKind::kNotFound, nullptr};
}

} // namespace facebook::velox::gpu
