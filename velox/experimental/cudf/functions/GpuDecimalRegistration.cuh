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

// Decimal registration helpers, shaped after the ones in
// velox/functions/prestosql/DecimalFunctions.cpp. A decimal function registers
// five type combinations, and its result precision and scale are expressions
// over the argument ones; the constraint strings are copied from that file.
#pragma once

#include "velox/experimental/cudf/functions/GpuSimpleFunctionAdapter.cuh"

#include <string>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::gpu_sfi {

using Constraints = std::vector<std::pair<std::string, std::string>>;

/// The five combinations Velox registers for a binary decimal function. Short
/// and long decimals are int64_t and int128_t, so each needs its own kernel.
template <template <class> typename Func>
void registerGpuDecimalBinary(
    const std::vector<std::string>& aliases,
    const Constraints& constraints) {
  // (long, long) -> long
  registerGpuFunction<
      Func,
      LongDecimal<P3, S3>,
      LongDecimal<P1, S1>,
      LongDecimal<P2, S2>>(aliases, constraints);

  // (short, short) -> short
  registerGpuFunction<
      Func,
      ShortDecimal<P3, S3>,
      ShortDecimal<P1, S1>,
      ShortDecimal<P2, S2>>(aliases, constraints);

  // (short, short) -> long
  registerGpuFunction<
      Func,
      LongDecimal<P3, S3>,
      ShortDecimal<P1, S1>,
      ShortDecimal<P2, S2>>(aliases, constraints);

  // (short, long) -> long
  registerGpuFunction<
      Func,
      LongDecimal<P3, S3>,
      ShortDecimal<P1, S1>,
      LongDecimal<P2, S2>>(aliases, constraints);

  // (long, short) -> long
  registerGpuFunction<
      Func,
      LongDecimal<P3, S3>,
      LongDecimal<P1, S1>,
      ShortDecimal<P2, S2>>(aliases, constraints);
}

// The expressions below are Velox's, from DecimalFunctions.cpp, built from the
// P1/S1/P2/S2/P3/S3 tags so they follow any renumbering of IntegerVariable.
namespace decimal_detail {

inline std::string aPrecision() {
  return P1::name();
}
inline std::string bPrecision() {
  return P2::name();
}
inline std::string aScale() {
  return S1::name();
}
inline std::string bScale() {
  return S2::name();
}
inline std::string widerScale() {
  return "max(" + aScale() + ", " + bScale() + ")";
}

} // namespace decimal_detail

/// plus and minus: the wider integral part and the wider scale, plus a digit
/// for the carry.
inline Constraints plusMinusConstraints() {
  using namespace decimal_detail;
  return {
      {P3::name(),
       "min(38, max(" + aPrecision() + " - " + aScale() + ", " + bPrecision() +
           " - " + bScale() + ") + " + widerScale() + " + 1)"},
      {S3::name(), widerScale()}};
}

/// multiply: precisions and scales add.
inline Constraints multiplyConstraints() {
  using namespace decimal_detail;
  return {
      {P3::name(), "min(38, " + aPrecision() + " + " + bPrecision() + ")"},
      {S3::name(), aScale() + " + " + bScale()}};
}

/// divide: widens to keep the dividend's integral digits and the divisor's
/// scale.
inline Constraints divideConstraints() {
  using namespace decimal_detail;
  return {
      {P3::name(),
       "min(38, " + aPrecision() + " + " + bScale() + " + max(0, " + bScale() +
           " - " + aScale() + "))"},
      {S3::name(), widerScale()}};
}

/// modulus: cannot exceed either operand's integral part, and keeps the wider
/// scale.
inline Constraints modulusConstraints() {
  using namespace decimal_detail;
  return {
      {P3::name(),
       "min(" + bPrecision() + " - " + bScale() + ", " + aPrecision() + " - " +
           aScale() + ") + " + widerScale()},
      {S3::name(), widerScale()}};
}

/// floor, ceil and one-argument round: scale 0, and the integral digits plus
/// one for the rounding carry. Matches registerDecimalFloorOrCeil.
inline Constraints roundToIntegerConstraints() {
  using namespace decimal_detail;
  return {
      {P2::name(),
       "min(38, " + aPrecision() + " - " + aScale() + " + min(" + aScale() +
           ", 1))"},
      {S2::name(), "0"}};
}

/// truncate discards rather than rounds, so it needs no carry digit.
inline Constraints truncateToIntegerConstraints() {
  using namespace decimal_detail;
  return {
      {P2::name(), "max(" + aPrecision() + " - " + aScale() + ", 1)"},
      {S2::name(), "0"}};
}

/// The three combinations a decimal-to-integer function registers.
template <template <class> typename Func>
void registerGpuDecimalToInteger(
    const std::vector<std::string>& aliases,
    const Constraints& constraints) {
  registerGpuFunction<Func, LongDecimal<P2, S2>, LongDecimal<P1, S1>>(
      aliases, constraints);
  registerGpuFunction<Func, ShortDecimal<P2, S2>, LongDecimal<P1, S1>>(
      aliases, constraints);
  registerGpuFunction<Func, ShortDecimal<P2, S2>, ShortDecimal<P1, S1>>(
      aliases, constraints);
}

/// round(decimal, n): the scale is kept and the precision gains one digit.
template <template <class> typename Func>
void registerGpuDecimalRoundWithDigits(
    const std::vector<std::string>& aliases) {
  using namespace decimal_detail;
  const Constraints constraints{
      {P2::name(), "min(38, " + aPrecision() + " + 1)"}};

  registerGpuFunction<Func, LongDecimal<P2, S1>, LongDecimal<P1, S1>, int32_t>(
      aliases, constraints);
  registerGpuFunction<
      Func,
      ShortDecimal<P2, S1>,
      ShortDecimal<P1, S1>,
      int32_t>(aliases, constraints);
  registerGpuFunction<Func, LongDecimal<P2, S1>, ShortDecimal<P1, S1>, int32_t>(
      aliases, constraints);
}

/// truncate(decimal, n) keeps the input type exactly, so it declares no result
/// variables and needs no constraints.
template <template <class> typename Func>
void registerGpuDecimalTruncateWithDigits(
    const std::vector<std::string>& aliases) {
  registerGpuFunction<
      Func,
      ShortDecimal<P1, S1>,
      ShortDecimal<P1, S1>,
      int32_t>(aliases);
  registerGpuFunction<Func, LongDecimal<P1, S1>, LongDecimal<P1, S1>, int32_t>(
      aliases);
}

} // namespace facebook::velox::cudf_velox::gpu_sfi
