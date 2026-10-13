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

// Test-only simple functions compiled for GPU, registered the way
// GpuPrestoFunctions.cu registers the production ones. Compiled with the
// gpu_shadows/ include path ahead of the Velox source root.

#include "velox/experimental/cudf/functions/GpuSimpleFunctionAdapter.cuh"
#include "velox/experimental/cudf/tests/GpuTestFunctions.h"

#include "velox/common/base/Macros.h"
#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"

namespace facebook::velox::cudf_velox::gpu_sfi {
namespace {

// Returns the UTC millis of a TIMESTAMP WITH TIME ZONE. The argument arrives in
// the custom-type view, as on the CPU, and the packed int64 the column holds
// is behind its operator*.
template <typename T>
struct GpuMillisUtcFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(
      int64_t& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimeZone) {
    result = unpackMillisUtc(*timestampWithTimeZone);
  }
};

// Returns what initialize() saw for the second argument: its value when it was
// a constant, -1 otherwise. The row's own arguments are ignored, so the result
// can only come from the instance.
template <typename T>
struct GpuInitializeConstantFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<int64_t>* /*first*/,
      const arg_type<int64_t>* second) {
    if (second != nullptr) {
      constant_ = *second;
    }
  }

  VELOX_GPU_COMPATIBLE void call(
      int64_t& result,
      const arg_type<int64_t>& /*first*/,
      const arg_type<int64_t>& /*second*/) {
    result = constant_;
  }

  int64_t constant_{-1};
};

// Returns the length of the constant string initialize() saw, -1 when it saw
// none. The string is declared Constant<Varchar>, so a column never binds.
template <typename T>
struct GpuConstantVarcharLengthFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<int64_t>* /*first*/,
      const arg_type<Varchar>* text) {
    if (text != nullptr) {
      length_ = text->size();
    }
  }

  VELOX_GPU_COMPATIBLE void call(
      int64_t& result,
      const arg_type<int64_t>& /*first*/,
      const arg_type<Varchar>& /*text*/) {
    result = length_;
  }

  int64_t length_{-1};
};

} // namespace

void registerGpuTestFunctions() {
  registerGpuFunction<GpuMillisUtcFunction, int64_t, TimestampWithTimezone>(
      {"test_millis_utc"});
  registerGpuFunction<GpuInitializeConstantFunction, int64_t, int64_t, int64_t>(
      {"test_initialize_constant"});
  registerGpuFunction<
      GpuConstantVarcharLengthFunction,
      int64_t,
      int64_t,
      Constant<Varchar>>({"test_constant_varchar_length"});
}

} // namespace facebook::velox::cudf_velox::gpu_sfi
