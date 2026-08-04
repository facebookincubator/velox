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

// Host side of the GPU simple-function bridge. Compiled against real Velox,
// with no shadow include path, so it may name Velox types freely. It reaches
// device code only through GpuLaunchFn, a function pointer over cudf and rmm
// types, which is what keeps the shadow boundary intact.

#include "velox/experimental/cudf/functions/GpuFunctionBridge.h"

#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/functions/GpuFunctionRegistry.h"

#include "velox/expression/FunctionSignature.h"

namespace facebook::velox::cudf_velox::gpu_sfi {
namespace {

/// Adapts one GPU-compiled simple function to the interface the cuDF
/// expression evaluator already dispatches through.
class GpuSimpleFunction : public CudfFunction {
 public:
  GpuSimpleFunction(GpuLaunchFn launch, cudf::data_type outputType)
      : launch_(launch), outputType_(outputType) {}

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    std::vector<cudf::column_view> views;
    views.reserve(inputColumns.size());
    for (auto& column : inputColumns) {
      views.push_back(asView(column));
    }
    return launch_(views, outputType_, stream, mr);
  }

 private:
  const GpuLaunchFn launch_;
  const cudf::data_type outputType_;
};

exec::FunctionSignaturePtr toVeloxSignature(
    const GpuFunctionSignature& signature) {
  auto builder = exec::FunctionSignatureBuilder();
  builder.returnType(signature.returnType);
  for (const auto& argumentType : signature.argumentTypes) {
    builder.argumentType(argumentType);
  }
  return builder.build();
}

} // namespace

void publishGpuFunctionsToCudf() {
  for (const auto& [name, entries] : gpuFunctionRegistry()) {
    for (const auto& entry : entries) {
      const auto launch = entry.launch;
      registerCudfFunction(
          name,
          [launch](
              const std::string& /*name*/,
              const core::TypedExprPtr& expr,
              memory::MemoryPool* /*pool*/) -> std::shared_ptr<CudfFunction> {
            return std::make_shared<GpuSimpleFunction>(
                launch, veloxToCudfDataType(expr->type()));
          },
          {toVeloxSignature(entry.signature)});
    }
  }
}

} // namespace facebook::velox::cudf_velox::gpu_sfi
