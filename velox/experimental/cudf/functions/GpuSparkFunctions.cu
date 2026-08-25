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

// SparkSQL simple functions compiled for GPU.
//
// The sibling of GpuPrestoFunctions.cu, and the reason dialect separation was
// worth building: the two files instantiate their own Fn<GpuExec>, so where the
// dialects disagree they produce different kernels under different names,
// exactly as their CPU registrations do.
//
// The disagreement that matters here is overflow. Spark's add, subtract and
// multiply are the plain structs and are defined to wrap, while Presto's
// integral overloads of plus, minus and multiply are the Checked* structs and
// are defined to throw. So these registrations are faithful to Spark today,
// with no dependency on the per-row error reporting that the Presto side is
// still waiting on.
//
// Compiled with the gpu_shadows/ include path ahead of the Velox source root.

#include "velox/experimental/cudf/functions/GpuRegistrationHelpers.cuh"

// Only prestosql/Arithmetic.h: every struct registered below is one Spark
// shares with Presto rather than one it defines. sparksql/Arithmetic.h is
// deliberately not included -- it reaches real folly, which does not parse
// behind the shadow -- so Spark's own structs (RemainderFunction,
// UnaryMinusFunction, the pmod family) are not yet registrable here.
#include "velox/functions/prestosql/Arithmetic.h"

namespace facebook::velox::cudf_velox::gpu_sfi {

using namespace facebook::velox::functions;

void registerSparkGpuFunctions(const std::string& prefix) {
  // --- Arithmetic ---------------------------------------------------------
  // Spark spells these add/subtract/multiply and binds them to the unchecked
  // structs; its checked_* names are separate functions and are not registered
  // here, since those do need the error path.
  registerGpuBinaryNumeric<PlusFunction>({prefix + "add"});
  registerGpuBinaryNumeric<MinusFunction>({prefix + "subtract"});
  registerGpuBinaryNumeric<MultiplyFunction>({prefix + "multiply"});
  registerGpuFunction<DivideFunction, double, double, double>(
      {prefix + "divide"});

  // --- Rounding -----------------------------------------------------------
  // Spark reuses Presto's RoundFunction rather than defining its own, so this
  // is a true alias: the same struct under the Spark name. Spark has no
  // truncate, so there is nothing to alias for it.
  registerGpuUnaryNumeric<RoundFunction>({prefix + "round"});
  registerGpuNumericWithDecimals<RoundFunction>({prefix + "round"});
}

} // namespace facebook::velox::cudf_velox::gpu_sfi
