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

// Returns the UTC millis of a TIMESTAMP WITH TIME ZONE. The argument arrives as
// the packed int64 the column holds, which is what the resolver promises.
template <typename T>
struct GpuMillisUtcFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  VELOX_GPU_COMPATIBLE void call(
      int64_t& result,
      const arg_type<TimestampWithTimezone>& timestampWithTimeZone) {
    result = unpackMillisUtc(timestampWithTimeZone);
  }
};

} // namespace

void registerGpuTestFunctions() {
  registerGpuFunction<GpuMillisUtcFunction, int64_t, TimestampWithTimezone>(
      {"test_millis_utc"});
}

} // namespace facebook::velox::cudf_velox::gpu_sfi
