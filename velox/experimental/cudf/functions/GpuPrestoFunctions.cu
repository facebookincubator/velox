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

// Registers Velox Presto simple functions for native GPU execution.
// Each registerGpuFunction instantiates the GpuSimpleFunction bridge
// for that specific function+type combo.

#include "velox/experimental/cudf/functions/GpuSimpleFunction.cuh"
#include "velox/functions/prestosql/Arithmetic.h"
#include "velox/functions/prestosql/Comparisons.h"

namespace facebook::velox::gpu {

void registerPrestoArithmetic() {
  using namespace facebook::velox::functions;

  // Plus: double, float, int64, int32
  registerGpuFunction<PlusFunction<GpuExec>, double, double, double>("plus");
  registerGpuFunction<PlusFunction<GpuExec>, float, float, float>("plus");
  registerGpuFunction<PlusFunction<GpuExec>, int64_t, int64_t, int64_t>("plus");
  registerGpuFunction<PlusFunction<GpuExec>, int32_t, int32_t, int32_t>("plus");

  // Minus
  registerGpuFunction<MinusFunction<GpuExec>, double, double, double>("minus");
  registerGpuFunction<MinusFunction<GpuExec>, float, float, float>("minus");
  registerGpuFunction<MinusFunction<GpuExec>, int64_t, int64_t, int64_t>(
      "minus");
  registerGpuFunction<MinusFunction<GpuExec>, int32_t, int32_t, int32_t>(
      "minus");

  // Multiply
  registerGpuFunction<MultiplyFunction<GpuExec>, double, double, double>(
      "multiply");
  registerGpuFunction<MultiplyFunction<GpuExec>, float, float, float>(
      "multiply");
  registerGpuFunction<MultiplyFunction<GpuExec>, int64_t, int64_t, int64_t>(
      "multiply");

  // Divide
  registerGpuFunction<DivideFunction<GpuExec>, double, double, double>(
      "divide");

  // Modulus
  registerGpuFunction<ModulusFunction<GpuExec>, double, double, double>(
      "modulus");

  // Negate
  registerGpuFunction<NegateFunction<GpuExec>, double, double>("negate");
  registerGpuFunction<NegateFunction<GpuExec>, int64_t, int64_t>("negate");

  // Abs
  registerGpuFunction<AbsFunction<GpuExec>, double, double>("abs");
  registerGpuFunction<AbsFunction<GpuExec>, int64_t, int64_t>("abs");

  // Ceil/Floor
  registerGpuFunction<CeilFunction<GpuExec>, double, double>("ceil");
  registerGpuFunction<FloorFunction<GpuExec>, double, double>("floor");

  // Math functions
  registerGpuFunction<ExpFunction<GpuExec>, double, double>("exp");
  registerGpuFunction<LnFunction<GpuExec>, double, double>("ln");
  registerGpuFunction<Log2Function<GpuExec>, double, double>("log2");
  registerGpuFunction<Log10Function<GpuExec>, double, double>("log10");
  registerGpuFunction<SqrtFunction<GpuExec>, double, double>("sqrt");
  registerGpuFunction<CbrtFunction<GpuExec>, double, double>("cbrt");

  // Trig
  registerGpuFunction<SinFunction<GpuExec>, double, double>("sin");
  registerGpuFunction<CosFunction<GpuExec>, double, double>("cos");
  registerGpuFunction<TanFunction<GpuExec>, double, double>("tan");
  registerGpuFunction<AsinFunction<GpuExec>, double, double>("asin");
  registerGpuFunction<AcosFunction<GpuExec>, double, double>("acos");
  registerGpuFunction<AtanFunction<GpuExec>, double, double>("atan");
  registerGpuFunction<Atan2Function<GpuExec>, double, double, double>("atan2");

  // Power
  registerGpuFunction<PowerFunction<GpuExec>, double, double, double>("power");

  // Sign
  registerGpuFunction<SignFunction<GpuExec>, double, double>("sign");
  registerGpuFunction<SignFunction<GpuExec>, int64_t, int64_t>("sign");
}

void registerPrestoComparisons() {
  using namespace facebook::velox::functions;

  // Lt, Gt, Lte, Gte
  registerGpuFunction<LtFunction<GpuExec>, bool, double, double>("lt");
  registerGpuFunction<LtFunction<GpuExec>, bool, int64_t, int64_t>("lt");
  registerGpuFunction<GtFunction<GpuExec>, bool, double, double>("gt");
  registerGpuFunction<GtFunction<GpuExec>, bool, int64_t, int64_t>("gt");
  registerGpuFunction<LteFunction<GpuExec>, bool, double, double>("lte");
  registerGpuFunction<LteFunction<GpuExec>, bool, int64_t, int64_t>("lte");
  registerGpuFunction<GteFunction<GpuExec>, bool, double, double>("gte");
  registerGpuFunction<GteFunction<GpuExec>, bool, int64_t, int64_t>("gte");

  // Between
  registerGpuFunction<BetweenFunction<GpuExec>, bool, double, double, double>(
      "between");
  registerGpuFunction<
      BetweenFunction<GpuExec>,
      bool,
      int64_t,
      int64_t,
      int64_t>("between");
}

void registerAllPrestoGpuFunctions() {
  registerPrestoArithmetic();
  registerPrestoComparisons();
}

} // namespace facebook::velox::gpu
