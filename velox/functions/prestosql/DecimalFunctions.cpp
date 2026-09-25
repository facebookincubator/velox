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
#include "velox/functions/prestosql/detail/DecimalMathFunctions.h"

namespace facebook::velox::functions {
namespace {

// The function structs are implementation detail shared with the GPU path;
// only the registration entry points below are API.
using namespace detail;

template <template <class> typename Func>
void registerDecimalBinary(
    const std::string& name,
    const std::vector<exec::SignatureVariable>& constraints,
    std::string_view defaultOwner) {
  // (long, long) -> long
  registerFunction<
      Func,
      LongDecimal<P3, S3>,
      LongDecimal<P1, S1>,
      LongDecimal<P2, S2>>({name}, constraints, true, defaultOwner);

  // (short, short) -> short
  registerFunction<
      Func,
      ShortDecimal<P3, S3>,
      ShortDecimal<P1, S1>,
      ShortDecimal<P2, S2>>({name}, constraints, true, defaultOwner);

  // (short, short) -> long
  registerFunction<
      Func,
      LongDecimal<P3, S3>,
      ShortDecimal<P1, S1>,
      ShortDecimal<P2, S2>>({name}, constraints, true, defaultOwner);

  // (short, long) -> long
  registerFunction<
      Func,
      LongDecimal<P3, S3>,
      ShortDecimal<P1, S1>,
      LongDecimal<P2, S2>>({name}, constraints, true, defaultOwner);

  // (long, short) -> long
  registerFunction<
      Func,
      LongDecimal<P3, S3>,
      LongDecimal<P1, S1>,
      ShortDecimal<P2, S2>>({name}, constraints, true, defaultOwner);
}

template <template <class> typename Func>
void registerDecimalPlusMinus(
    const std::string& name,
    std::string_view defaultOwner) {
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

  registerDecimalBinary<Func>(name, constraints, defaultOwner);
}

} // namespace

void registerDecimalPlus(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerDecimalPlusMinus<DecimalPlusFunction>(prefix + "plus", defaultOwner);
}

void registerDecimalMinus(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerDecimalPlusMinus<DecimalMinusFunction>(
      prefix + "minus", defaultOwner);
}

void registerDecimalMultiply(
    const std::string& prefix,
    std::string_view defaultOwner) {
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
      prefix + "multiply", constraints, defaultOwner);
}

void registerDecimalDivide(
    const std::string& prefix,
    std::string_view defaultOwner) {
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

  registerDecimalBinary<DecimalDivideFunction>(
      prefix + "divide", constraints, defaultOwner);

  // (short, long) -> short
  registerFunction<
      DecimalDivideFunction,
      ShortDecimal<P3, S3>,
      ShortDecimal<P1, S1>,
      LongDecimal<P2, S2>>(
      {prefix + "divide"}, constraints, true, defaultOwner);

  // (long, short) -> short
  registerFunction<
      DecimalDivideFunction,
      ShortDecimal<P3, S3>,
      LongDecimal<P1, S1>,
      ShortDecimal<P2, S2>>(
      {prefix + "divide"}, constraints, true, defaultOwner);
}

void registerDecimalModulus(
    const std::string& prefix,
    std::string_view defaultOwner) {
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
      ShortDecimal<P2, S2>>({prefix + "mod"}, constraints, true, defaultOwner);

  // (short, long) -> short
  registerFunction<
      DecimalModulusFunction,
      ShortDecimal<P3, S3>,
      ShortDecimal<P1, S1>,
      LongDecimal<P2, S2>>({prefix + "mod"}, constraints, true, defaultOwner);

  // (long, short) -> short
  registerFunction<
      DecimalModulusFunction,
      ShortDecimal<P3, S3>,
      LongDecimal<P1, S1>,
      ShortDecimal<P2, S2>>({prefix + "mod"}, constraints, true, defaultOwner);

  // (short, long) -> long
  registerFunction<
      DecimalModulusFunction,
      LongDecimal<P3, S3>,
      ShortDecimal<P1, S1>,
      LongDecimal<P2, S2>>({prefix + "mod"}, constraints, true, defaultOwner);

  // (long, short) -> long
  registerFunction<
      DecimalModulusFunction,
      LongDecimal<P3, S3>,
      LongDecimal<P1, S1>,
      ShortDecimal<P2, S2>>({prefix + "mod"}, constraints, true, defaultOwner);

  // (long, long) -> long
  registerFunction<
      DecimalModulusFunction,
      LongDecimal<P3, S3>,
      LongDecimal<P1, S1>,
      LongDecimal<P2, S2>>({prefix + "mod"}, constraints, true, defaultOwner);
}

template <template <class> typename TFunc>
void registerDecimalFloorOrCeil(
    const std::string& prefix,
    const std::string& functionName,
    std::string_view defaultOwner) {
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
      {prefix + functionName}, constraints, true, defaultOwner);

  registerFunction<TFunc, ShortDecimal<P2, S2>, LongDecimal<P1, S1>>(
      {prefix + functionName}, constraints, true, defaultOwner);

  registerFunction<TFunc, ShortDecimal<P2, S2>, ShortDecimal<P1, S1>>(
      {prefix + functionName}, constraints, true, defaultOwner);
}

void registerDecimalFloor(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerDecimalFloorOrCeil<DecimalFloorFunction>(
      prefix, "floor", defaultOwner);
}

void registerDecimalCeil(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerDecimalFloorOrCeil<DecimalCeilFunction>(prefix, "ceil", defaultOwner);
}

void registerDecimalRound(
    const std::string& prefix,
    std::string_view defaultOwner) {
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
        LongDecimal<P1, S1>>(
        {prefix + "round"}, constraints, true, defaultOwner);

    registerFunction<
        DecimalRoundFunction,
        ShortDecimal<P2, S2>,
        LongDecimal<P1, S1>>(
        {prefix + "round"}, constraints, true, defaultOwner);

    registerFunction<
        DecimalRoundFunction,
        ShortDecimal<P2, S2>,
        ShortDecimal<P1, S1>>(
        {prefix + "round"}, constraints, true, defaultOwner);
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
        int32_t>({prefix + "round"}, constraints, true, defaultOwner);

    registerFunction<
        DecimalRoundFunction,
        ShortDecimal<P2, S1>,
        ShortDecimal<P1, S1>,
        int32_t>({prefix + "round"}, constraints, true, defaultOwner);

    registerFunction<
        DecimalRoundFunction,
        LongDecimal<P2, S1>,
        ShortDecimal<P1, S1>,
        int32_t>({prefix + "round"}, constraints, true, defaultOwner);
  }
}

void registerDecimalTruncate(
    const std::string& prefix,
    std::string_view defaultOwner) {
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
      ShortDecimal<P1, S1>>(
      {prefix + "truncate"}, constraints, true, defaultOwner);

  registerFunction<
      DecimalTruncateFunction,
      LongDecimal<P2, S2>,
      LongDecimal<P1, S1>>(
      {prefix + "truncate"}, constraints, true, defaultOwner);

  registerFunction<
      DecimalTruncateFunction,
      ShortDecimal<P2, S2>,
      LongDecimal<P1, S1>>(
      {prefix + "truncate"}, constraints, true, defaultOwner);

  // truncate(decimal, n) -> decimal
  registerFunction<
      DecimalTruncateFunction,
      ShortDecimal<P1, S1>,
      ShortDecimal<P1, S1>,
      int32_t>({prefix + "truncate"}, {}, true, defaultOwner);

  registerFunction<
      DecimalTruncateFunction,
      LongDecimal<P1, S1>,
      LongDecimal<P1, S1>,
      int32_t>({prefix + "truncate"}, {}, true, defaultOwner);
}

} // namespace facebook::velox::functions
