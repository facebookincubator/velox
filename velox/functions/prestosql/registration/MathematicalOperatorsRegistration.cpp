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
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/RegistrationHelpers.h"
#include "velox/functions/prestosql/Arithmetic.h"
#include "velox/functions/prestosql/DecimalFunctions.h"

namespace facebook::velox::functions {

namespace {
void registerMathOperators(
    const std::string& prefix = "",
    std::string_view defaultOwner = {}) {
  registerBinaryFloatingPoint<PlusFunction>({prefix + "plus"}, defaultOwner);
  registerFunction<
      PlusFunction,
      IntervalDayTime,
      IntervalDayTime,
      IntervalDayTime>({prefix + "plus"}, {}, true, defaultOwner);
  registerFunction<
      PlusFunction,
      IntervalYearMonth,
      IntervalYearMonth,
      IntervalYearMonth>({prefix + "plus"}, {}, true, defaultOwner);
  registerBinaryFloatingPoint<MinusFunction>({prefix + "minus"}, defaultOwner);
  registerFunction<
      MinusFunction,
      IntervalDayTime,
      IntervalDayTime,
      IntervalDayTime>({prefix + "minus"}, {}, true, defaultOwner);
  registerFunction<
      MinusFunction,
      IntervalYearMonth,
      IntervalYearMonth,
      IntervalYearMonth>({prefix + "minus"}, {}, true, defaultOwner);
  registerBinaryFloatingPoint<MultiplyFunction>(
      {prefix + "multiply"}, defaultOwner);
  registerFunction<MultiplyFunction, IntervalDayTime, IntervalDayTime, int64_t>(
      {prefix + "multiply"}, {}, true, defaultOwner);
  registerFunction<MultiplyFunction, IntervalDayTime, int64_t, IntervalDayTime>(
      {prefix + "multiply"}, {}, true, defaultOwner);
  registerFunction<
      IntervalMultiplyFunction,
      IntervalDayTime,
      IntervalDayTime,
      double>({prefix + "multiply"}, {}, true, defaultOwner);
  registerFunction<
      IntervalMultiplyFunction,
      IntervalDayTime,
      double,
      IntervalDayTime>({prefix + "multiply"}, {}, true, defaultOwner);
  registerFunction<
      MultiplyFunction,
      IntervalYearMonth,
      IntervalYearMonth,
      int32_t>({prefix + "multiply"}, {}, true, defaultOwner);
  registerFunction<
      MultiplyFunction,
      IntervalYearMonth,
      int32_t,
      IntervalYearMonth>({prefix + "multiply"}, {}, true, defaultOwner);
  registerFunction<
      IntervalMultiplyFunction,
      IntervalYearMonth,
      IntervalYearMonth,
      double>({prefix + "multiply"}, {}, true, defaultOwner);
  registerFunction<
      IntervalMultiplyFunction,
      IntervalYearMonth,
      double,
      IntervalYearMonth>({prefix + "multiply"}, {}, true, defaultOwner);
  registerBinaryFloatingPoint<DivideFunction>(
      {prefix + "divide"}, defaultOwner);
  registerFunction<
      IntervalDivideFunction,
      IntervalDayTime,
      IntervalDayTime,
      double>({prefix + "divide"}, {}, true, defaultOwner);
  registerFunction<
      IntervalDivideFunction,
      IntervalYearMonth,
      IntervalYearMonth,
      double>({prefix + "divide"}, {}, true, defaultOwner);
  registerBinaryFloatingPoint<ModulusFunction>({prefix + "mod"}, defaultOwner);
  registerBinaryIntegral<PModIntFunction>({prefix + "pmod"}, defaultOwner);
  registerBinaryFloatingPoint<PModFloatFunction>(
      {prefix + "pmod"}, defaultOwner);
}

} // namespace

void registerMathematicalOperators(
    const std::string& prefix = "",
    std::string_view defaultOwner = {}) {
  registerMathOperators(prefix, defaultOwner);

  registerDecimalPlus(prefix, defaultOwner);
  registerDecimalMinus(prefix, defaultOwner);
  registerDecimalMultiply(prefix, defaultOwner);
  registerDecimalDivide(prefix, defaultOwner);
  registerDecimalModulus(prefix, defaultOwner);
}

} // namespace facebook::velox::functions
