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

// SparkSQL simple functions compiled for GPU. Each dialect instantiates its
// own structs, so where Spark and Presto disagree they get different kernels,
// as on the CPU: Spark's add, subtract and multiply wrap on overflow, while
// Presto's integral plus, minus and multiply use the Checked* structs.
// Compiled with the gpu_shadows/ include path ahead of the Velox source root.

#include "velox/experimental/cudf/functions/GpuRegistrationHelpers.cuh"

// Every struct registered below is one Spark shares with Presto.
// sparksql/Arithmetic.h does not parse behind the shadow, so Spark's own
// structs (RemainderFunction, UnaryMinusFunction, the pmod family) are not
// registered.
#include "velox/functions/prestosql/Arithmetic.h"

namespace facebook::velox::cudf_velox::gpu_sfi {

using namespace facebook::velox::functions;

void registerSparkGpuFunctions(const std::string& prefix) {
  // --- Arithmetic ---------------------------------------------------------
  // The checked_* variants are not registered: they need error reporting.
  registerGpuBinaryNumeric<PlusFunction>({prefix + "add"});
  registerGpuBinaryNumeric<MinusFunction>({prefix + "subtract"});
  registerGpuBinaryNumeric<MultiplyFunction>({prefix + "multiply"});
  registerGpuFunction<DivideFunction, double, double, double>(
      {prefix + "divide"});

  // --- Rounding -----------------------------------------------------------
  // Spark uses Presto's RoundFunction.
  registerGpuUnaryNumeric<RoundFunction>({prefix + "round"});
  registerGpuNumericWithDecimals<RoundFunction>({prefix + "round"});
}

} // namespace facebook::velox::cudf_velox::gpu_sfi
