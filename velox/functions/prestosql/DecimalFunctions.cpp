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

#include "velox/expression/DecodedArgs.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/DecimalMathFunctions.h"

namespace facebook::velox::functions {
namespace {

template <template <class> typename Func>
void registerDecimalBinary(
    const std::string& name,
    const std::vector<exec::SignatureVariable>& constraints) {
  // (long, long) -> long
  registerFunction<
      Func,
      LongDecimal<P3, S3>,
      LongDecimal<P1, S1>,
      LongDecimal<P2, S2>>({name}, constraints);

  // (short, short) -> short
  registerFunction<
      Func,
      ShortDecimal<P3, S3>,
      ShortDecimal<P1, S1>,
      ShortDecimal<P2, S2>>({name}, constraints);

  // (short, short) -> long
  registerFunction<
      Func,
      LongDecimal<P3, S3>,
      ShortDecimal<P1, S1>,
      ShortDecimal<P2, S2>>({name}, constraints);

  // (short, long) -> long
  registerFunction<
      Func,
      LongDecimal<P3, S3>,
      ShortDecimal<P1, S1>,
      LongDecimal<P2, S2>>({name}, constraints);

  // (long, short) -> long
  registerFunction<
      Func,
      LongDecimal<P3, S3>,
      LongDecimal<P1, S1>,
      ShortDecimal<P2, S2>>({name}, constraints);
}

template <template <class> typename Func>
void registerDecimalPlusMinus(const std::string& name) {
  std::vector<exec::SignatureVariable> constraints = {
      exec::SignatureVariable(
          P3::name(),
          fmt::format(
              "min(38, max({a_precision} - {a_scale}, {b_precision} - {b_scale}) + max({a_scale}, {b_scale}) + 1)",
              fmt::arg("a_precision", P1::name()),
              fmt::arg("b_precision", P2::name()),
              fmt::arg("a_scale", S1::name()),
              fmt::arg("b_scale", S2::name())),
          exec::ParameterType::kIntegerParameter),
      exec::SignatureVariable(
          S3::name(),
          fmt::format(
              "max({a_scale}, {b_scale})",
              fmt::arg("a_scale", S1::name()),
              fmt::arg("b_scale", S2::name())),
          exec::ParameterType::kIntegerParameter),
  };

  registerDecimalBinary<Func>(name, constraints);
}

} // namespace

void registerDecimalPlus(const std::string& prefix) {
  registerDecimalPlusMinus<DecimalPlusFunction>(prefix + "plus");
}

void registerDecimalMinus(const std::string& prefix) {
  registerDecimalPlusMinus<DecimalMinusFunction>(prefix + "minus");
}

void registerDecimalMultiply(const std::string& prefix) {
  std::vector<exec::SignatureVariable> constraints = {
      exec::SignatureVariable(
          P3::name(),
          fmt::format(
              "min(38, {a_precision} + {b_precision})",
              fmt::arg("a_precision", P1::name()),
              fmt::arg("b_precision", P2::name())),
          exec::ParameterType::kIntegerParameter),
      exec::SignatureVariable(
          S3::name(),
          // Result type resolution fails if sum of input scales exceeds 38.
          fmt::format(
              "{a_scale} + {b_scale}",
              fmt::arg("a_scale", S1::name()),
              fmt::arg("b_scale", S2::name())),
          exec::ParameterType::kIntegerParameter),
  };

  registerDecimalBinary<DecimalMultiplyFunction>(
      prefix + "multiply", constraints);
}

void registerDecimalDivide(const std::string& prefix) {
  std::vector<exec::SignatureVariable> constraints = {
      exec::SignatureVariable(
          P3::name(),
          fmt::format(
              "min(38, {a_precision} + {b_scale} + max(0, {b_scale} - {a_scale}))",
              fmt::arg("a_precision", P1::name()),
              fmt::arg("a_scale", S1::name()),
              fmt::arg("b_scale", S2::name())),
          exec::ParameterType::kIntegerParameter),
      exec::SignatureVariable(
          S3::name(),
          fmt::format(
              "max({a_scale}, {b_scale})",
              fmt::arg("a_scale", S1::name()),
              fmt::arg("b_scale", S2::name())),
          exec::ParameterType::kIntegerParameter),
  };

  registerDecimalBinary<DecimalDivideFunction>(prefix + "divide", constraints);

  // (short, long) -> short
  registerFunction<
      DecimalDivideFunction,
      ShortDecimal<P3, S3>,
      ShortDecimal<P1, S1>,
      LongDecimal<P2, S2>>({prefix + "divide"}, constraints);

  // (long, short) -> short
  registerFunction<
      DecimalDivideFunction,
      ShortDecimal<P3, S3>,
      LongDecimal<P1, S1>,
      ShortDecimal<P2, S2>>({prefix + "divide"}, constraints);
}

void registerDecimalModulus(const std::string& prefix) {
  std::vector<exec::SignatureVariable> constraints = {
      exec::SignatureVariable(
          P3::name(),
          fmt::format(
              "min({b_precision} - {b_scale}, {a_precision} - {a_scale}) + max({a_scale}, {b_scale})",
              fmt::arg("a_precision", P1::name()),
              fmt::arg("a_scale", S1::name()),
              fmt::arg("b_precision", P2::name()),
              fmt::arg("b_scale", S2::name())),
          exec::ParameterType::kIntegerParameter),
      exec::SignatureVariable(
          S3::name(),
          fmt::format(
              "max({a_scale}, {b_scale})",
              fmt::arg("a_scale", S1::name()),
              fmt::arg("b_scale", S2::name())),
          exec::ParameterType::kIntegerParameter),
  };

  // (short, short) -> short
  registerFunction<
      DecimalModulusFunction,
      ShortDecimal<P3, S3>,
      ShortDecimal<P1, S1>,
      ShortDecimal<P2, S2>>({prefix + "mod"}, constraints);

  // (short, long) -> short
  registerFunction<
      DecimalModulusFunction,
      ShortDecimal<P3, S3>,
      ShortDecimal<P1, S1>,
      LongDecimal<P2, S2>>({prefix + "mod"}, constraints);

  // (long, short) -> short
  registerFunction<
      DecimalModulusFunction,
      ShortDecimal<P3, S3>,
      LongDecimal<P1, S1>,
      ShortDecimal<P2, S2>>({prefix + "mod"}, constraints);

  // (short, long) -> long
  registerFunction<
      DecimalModulusFunction,
      LongDecimal<P3, S3>,
      ShortDecimal<P1, S1>,
      LongDecimal<P2, S2>>({prefix + "mod"}, constraints);

  // (long, short) -> long
  registerFunction<
      DecimalModulusFunction,
      LongDecimal<P3, S3>,
      LongDecimal<P1, S1>,
      ShortDecimal<P2, S2>>({prefix + "mod"}, constraints);

  // (long, long) -> long
  registerFunction<
      DecimalModulusFunction,
      LongDecimal<P3, S3>,
      LongDecimal<P1, S1>,
      LongDecimal<P2, S2>>({prefix + "mod"}, constraints);
}

template <template <class> typename TFunc>
void registerDecimalFloorOrCeil(
    const std::string& prefix,
    const std::string& functionName) {
  std::vector<exec::SignatureVariable> constraints = {
      exec::SignatureVariable(
          P2::name(),
          fmt::format(
              "min(38, {p} - {s} + min({s}, 1))",
              fmt::arg("p", P1::name()),
              fmt::arg("s", S1::name())),
          exec::ParameterType::kIntegerParameter),
      exec::SignatureVariable(
          S2::name(), "0", exec::ParameterType::kIntegerParameter),
  };

  registerFunction<TFunc, LongDecimal<P2, S2>, LongDecimal<P1, S1>>(
      {prefix + functionName}, constraints);

  registerFunction<TFunc, ShortDecimal<P2, S2>, LongDecimal<P1, S1>>(
      {prefix + functionName}, constraints);

  registerFunction<TFunc, ShortDecimal<P2, S2>, ShortDecimal<P1, S1>>(
      {prefix + functionName}, constraints);
}

void registerDecimalFloor(const std::string& prefix) {
  registerDecimalFloorOrCeil<DecimalFloorFunction>(prefix, "floor");
}

void registerDecimalCeil(const std::string& prefix) {
  registerDecimalFloorOrCeil<DecimalCeilFunction>(prefix, "ceil");
}

void registerDecimalRound(const std::string& prefix) {
  // round(decimal) -> decimal
  {
    std::vector<exec::SignatureVariable> constraints = {
        exec::SignatureVariable(
            P2::name(),
            fmt::format(
                "min(38, {p} - {s} + min({s}, 1))",
                fmt::arg("p", P1::name()),
                fmt::arg("s", S1::name())),
            exec::ParameterType::kIntegerParameter),
        exec::SignatureVariable(
            S2::name(), "0", exec::ParameterType::kIntegerParameter),
    };

    registerFunction<
        DecimalRoundFunction,
        LongDecimal<P2, S2>,
        LongDecimal<P1, S1>>({prefix + "round"}, constraints);

    registerFunction<
        DecimalRoundFunction,
        ShortDecimal<P2, S2>,
        LongDecimal<P1, S1>>({prefix + "round"}, constraints);

    registerFunction<
        DecimalRoundFunction,
        ShortDecimal<P2, S2>,
        ShortDecimal<P1, S1>>({prefix + "round"}, constraints);
  }

  // round(decimal, n) -> decimal
  {
    std::vector<exec::SignatureVariable> constraints = {
        exec::SignatureVariable(
            P2::name(),
            fmt::format("min(38, {p} + 1)", fmt::arg("p", P1::name())),
            exec::ParameterType::kIntegerParameter),
    };

    registerFunction<
        DecimalRoundFunction,
        LongDecimal<P2, S1>,
        LongDecimal<P1, S1>,
        int32_t>({prefix + "round"}, constraints);

    registerFunction<
        DecimalRoundFunction,
        ShortDecimal<P2, S1>,
        ShortDecimal<P1, S1>,
        int32_t>({prefix + "round"}, constraints);

    registerFunction<
        DecimalRoundFunction,
        LongDecimal<P2, S1>,
        ShortDecimal<P1, S1>,
        int32_t>({prefix + "round"}, constraints);
  }
}

void registerDecimalTruncate(const std::string& prefix) {
  // truncate(decimal) -> decimal
  std::vector<exec::SignatureVariable> constraints = {
      exec::SignatureVariable(
          P2::name(),
          fmt::format(
              "max({p} - {s}, 1)",
              fmt::arg("p", P1::name()),
              fmt::arg("s", S1::name())),
          exec::ParameterType::kIntegerParameter),
      exec::SignatureVariable(
          S2::name(), "0", exec::ParameterType::kIntegerParameter),
  };

  registerFunction<
      DecimalTruncateFunction,
      ShortDecimal<P2, S2>,
      ShortDecimal<P1, S1>>({prefix + "truncate"}, constraints);

  registerFunction<
      DecimalTruncateFunction,
      LongDecimal<P2, S2>,
      LongDecimal<P1, S1>>({prefix + "truncate"}, constraints);

  registerFunction<
      DecimalTruncateFunction,
      ShortDecimal<P2, S2>,
      LongDecimal<P1, S1>>({prefix + "truncate"}, constraints);

  // truncate(decimal, n) -> decimal
  registerFunction<
      DecimalTruncateFunction,
      ShortDecimal<P1, S1>,
      ShortDecimal<P1, S1>,
      int32_t>({prefix + "truncate"});

  registerFunction<
      DecimalTruncateFunction,
      LongDecimal<P1, S1>,
      LongDecimal<P1, S1>,
      int32_t>({prefix + "truncate"});
}

} // namespace facebook::velox::functions
