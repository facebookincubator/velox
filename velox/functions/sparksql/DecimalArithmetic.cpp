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
#include "velox/functions/sparksql/detail/DecimalArithmeticFunctions.h"

namespace facebook::velox::functions::sparksql {
namespace {

// The function structs are implementation detail shared with the GPU path;
// only the registration entry points below are API.
using namespace detail;

static constexpr const char* kDenyPrecisionLoss = "_deny_precision_loss";

// Computes Spark's decimal positive modulo after aligning operand scales.
// Correction can exceed the result precision when the dividend has fewer
// integral digits than the divisor.
template <typename TExec, bool checked>
struct DecimalPmodFunction {
  using CallResult = std::conditional_t<checked, Status, bool>;

  template <typename A, typename B>
  void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& /*config*/,
      const A* /*dividend*/,
      const B* /*divisor*/) {
    const auto [dividendPrecision, dividendScale] =
        getDecimalPrecisionScale(*inputTypes[0]);
    const auto [divisorPrecision, divisorScale] =
        getDecimalPrecisionScale(*inputTypes[1]);
    const auto resultScale = std::max(dividendScale, divisorScale);
    const auto resultPrecision = std::min(
                                     dividendPrecision - dividendScale,
                                     divisorPrecision - divisorScale) +
        resultScale;
    dividendFactor_ =
        velox::DecimalUtil::kPowersOfTen[resultScale - dividendScale];
    divisorFactor_ =
        velox::DecimalUtil::kPowersOfTen[resultScale - divisorScale];
    resultLimit_ = velox::DecimalUtil::kPowersOfTen[resultPrecision];
  }

  template <typename R, typename A, typename B>
  CallResult call(R& out, const A& dividend, const B& divisor) {
    if (UNLIKELY(divisor == 0)) {
      return error("Division by zero");
    }
    const auto result = compute(dividend, divisor);
    if (UNLIKELY(!result.has_value())) {
      return error("Decimal overflow in pmod");
    }
    out = static_cast<R>(result.value());
    if constexpr (checked) {
      return Status::OK();
    } else {
      return true;
    }
  }

 private:
  CallResult error(std::string_view errorMessage) const {
    if constexpr (checked) {
      return Status::UserError(errorMessage);
    } else {
      return false;
    }
  }

  std::optional<int128_t> compute(int128_t dividend, int128_t divisor) const {
    int128_t scaledDividend;
    int128_t scaledDivisor;
    if (!__builtin_mul_overflow(dividend, dividendFactor_, &scaledDividend) &&
        !__builtin_mul_overflow(divisor, divisorFactor_, &scaledDivisor)) {
      auto result = scaledDividend % scaledDivisor;
      if (result < 0 && scaledDivisor > 0) {
        result += scaledDivisor;
      }
      if (result <= -resultLimit_ || result >= resultLimit_) {
        return std::nullopt;
      }
      return result;
    }

    // A 38-digit decimal scaled by up to 10^38 needs at most 76 digits.
    // Keep the correction and precision check wide before narrowing.
    const int256_t wideDividend = int256_t(dividend) * dividendFactor_;
    const int256_t wideDivisor = int256_t(divisor) * divisorFactor_;
    int256_t result = wideDividend % wideDivisor;
    if (result < 0 && wideDivisor > 0) {
      result += wideDivisor;
    }
    if (result <= -resultLimit_ || result >= resultLimit_) {
      return std::nullopt;
    }
    return result.convert_to<int128_t>();
  }

  int128_t dividendFactor_;
  int128_t divisorFactor_;
  int128_t resultLimit_;
};

template <typename TExec>
using PmodFunction = DecimalPmodFunction<TExec, false>;

template <typename TExec>
using CheckedPmodFunction = DecimalPmodFunction<TExec, true>;

template <template <class> typename Func>
void registerDecimalBinary(
    const std::string& name,
    std::vector<exec::SignatureVariable> constraints) {
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

// Used in function registration to generate the string to cap value at 38.
std::string bounded(const std::string& value) {
  return fmt::format("({}) <= 38 ? ({}) : 38", value, value);
}

std::vector<exec::SignatureVariable> makeConstraints(
    const std::string& rPrecision,
    const std::string& rScale,
    bool allowPrecisionLoss) {
  std::string finalScale = allowPrecisionLoss
      ? fmt::format(
            "({}) <= 38 ? ({}) : max(({}) - ({}) + 38, min(({}), 6))",
            rPrecision,
            rScale,
            rScale,
            rPrecision,
            rScale)
      : bounded(rScale);
  return {
      exec::SignatureVariable(
          P3::name(),
          fmt::format(
              "min(38, {r_precision})", fmt::arg("r_precision", rPrecision)),
          exec::ParameterType::kIntegerParameter),
      exec::SignatureVariable(
          S3::name(), finalScale, exec::ParameterType::kIntegerParameter)};
}

std::pair<std::string, std::string> getAddSubtractResultPrecisionScale() {
  std::string rPrecision = fmt::format(
      "max({a_precision} - {a_scale}, {b_precision} - {b_scale}) + max({a_scale}, {b_scale}) + 1",
      fmt::arg("a_precision", P1::name()),
      fmt::arg("b_precision", P2::name()),
      fmt::arg("a_scale", S1::name()),
      fmt::arg("b_scale", S2::name()));
  std::string rScale = fmt::format(
      "max({a_scale}, {b_scale})",
      fmt::arg("a_scale", S1::name()),
      fmt::arg("b_scale", S2::name()));
  return {rPrecision, rScale};
}

template <typename TExec>
using AddFunctionAllowPrecisionLoss = DecimalAddFunction<TExec, true>;

template <typename TExec>
using AddFunctionDenyPrecisionLoss = DecimalAddFunction<TExec, false>;

template <typename TExec>
using SubtractFunctionAllowPrecisionLoss = DecimalSubtractFunction<TExec, true>;

template <typename TExec>
using SubtractFunctionDenyPrecisionLoss = DecimalSubtractFunction<TExec, false>;

template <typename TExec>
using MultiplyFunctionAllowPrecisionLoss = DecimalMultiplyFunction<TExec, true>;

template <typename TExec>
using MultiplyFunctionDenyPrecisionLoss = DecimalMultiplyFunction<TExec, false>;

template <typename TExec>
using DivideFunctionAllowPrecisionLoss = DecimalDivideFunction<TExec, true>;

template <typename TExec>
using DivideFunctionDenyPrecisionLoss = DecimalDivideFunction<TExec, false>;

template <typename TExec>
using CheckedAddFunctionAllowPrecisionLoss =
    CheckedDecimalAddFunction<TExec, true>;

template <typename TExec>
using CheckedAddFunctionDenyPrecisionLoss =
    CheckedDecimalAddFunction<TExec, false>;

template <typename TExec>
using CheckedSubtractFunctionAllowPrecisionLoss =
    CheckedDecimalSubtractFunction<TExec, true>;

template <typename TExec>
using CheckedSubtractFunctionDenyPrecisionLoss =
    CheckedDecimalSubtractFunction<TExec, false>;

template <typename TExec>
using CheckedMultiplyFunctionAllowPrecisionLoss =
    CheckedDecimalMultiplyFunction<TExec, true>;

template <typename TExec>
using CheckedMultiplyFunctionDenyPrecisionLoss =
    CheckedDecimalMultiplyFunction<TExec, false>;

std::vector<exec::SignatureVariable> getDivideConstraintsDenyPrecisionLoss() {
  std::string wholeDigits = fmt::format(
      "min(38, {a_precision} - {a_scale} + {b_scale})",
      fmt::arg("a_precision", P1::name()),
      fmt::arg("a_scale", S1::name()),
      fmt::arg("b_scale", S2::name()));
  std::string fractionDigits = fmt::format(
      "min(38, max(6, {a_scale} + {b_precision} + 1))",
      fmt::arg("a_scale", S1::name()),
      fmt::arg("b_precision", P2::name()));
  std::string diff = wholeDigits + " + " + fractionDigits + " - 38";
  std::string newFractionDigits =
      fmt::format("({}) - ({}) / 2 - 1", fractionDigits, diff);
  std::string newWholeDigits = fmt::format("38 - ({})", newFractionDigits);
  return {
      exec::SignatureVariable(
          P3::name(),
          fmt::format(
              "({}) > 0 ? ({}) : ({})",
              diff,
              bounded(newWholeDigits + " + " + newFractionDigits),
              bounded(wholeDigits + " + " + fractionDigits)),
          exec::ParameterType::kIntegerParameter),
      exec::SignatureVariable(
          S3::name(),
          fmt::format(
              "({}) > 0 ? ({}) : ({})",
              diff,
              bounded(newFractionDigits),
              bounded(fractionDigits)),
          exec::ParameterType::kIntegerParameter)};
}

std::vector<exec::SignatureVariable> getDivideConstraintsAllowPrecisionLoss() {
  std::string rPrecision = fmt::format(
      "{a_precision} - {a_scale} + {b_scale} + max(6, {a_scale} + {b_precision} + 1)",
      fmt::arg("a_precision", P1::name()),
      fmt::arg("b_precision", P2::name()),
      fmt::arg("a_scale", S1::name()),
      fmt::arg("b_scale", S2::name()));
  std::string rScale = fmt::format(
      "max(6, {a_scale} + {b_precision} + 1)",
      fmt::arg("a_scale", S1::name()),
      fmt::arg("b_precision", P2::name()));
  return makeConstraints(rPrecision, rScale, true);
}

template <template <class> typename Func>
void registerDecimalDivide(
    const std::string& functionName,
    std::vector<exec::SignatureVariable> constraints) {
  registerDecimalBinary<Func>(functionName, constraints);

  // (short, long) -> short
  registerFunction<
      Func,
      ShortDecimal<P3, S3>,
      ShortDecimal<P1, S1>,
      LongDecimal<P2, S2>>({functionName}, constraints);

  // (long, short) -> short
  registerFunction<
      Func,
      ShortDecimal<P3, S3>,
      LongDecimal<P1, S1>,
      ShortDecimal<P2, S2>>({functionName}, constraints);
}

template <template <class> typename Func>
void registerIntegralDecimalDivide(const std::string& functionName) {
  // (short, short) -> int64_t
  registerFunction<Func, int64_t, ShortDecimal<P1, S1>, ShortDecimal<P2, S2>>(
      {functionName});

  // (long, long) -> int64_t
  registerFunction<Func, int64_t, LongDecimal<P1, S1>, LongDecimal<P2, S2>>(
      {functionName});

  // (short, long) -> int64_t
  registerFunction<Func, int64_t, ShortDecimal<P1, S1>, LongDecimal<P2, S2>>(
      {functionName});

  // (long, short) -> int64_t
  registerFunction<Func, int64_t, LongDecimal<P1, S1>, ShortDecimal<P2, S2>>(
      {functionName});
}

template <template <class> typename Func>
void registerDecimalPmod(const std::string& name) {
  const auto resultScale = fmt::format("max({}, {})", S1::name(), S2::name());
  const auto resultPrecision = fmt::format(
      "min({} - {}, {} - {}) + {}",
      P1::name(),
      S1::name(),
      P2::name(),
      S2::name(),
      resultScale);
  const auto constraints = makeConstraints(resultPrecision, resultScale, false);
  registerDecimalBinary<Func>(name, constraints);
  // The smaller integral precision can produce a short result when one
  // operand is a long decimal.
  registerFunction<
      Func,
      ShortDecimal<P3, S3>,
      ShortDecimal<P1, S1>,
      LongDecimal<P2, S2>>({name}, constraints);
  registerFunction<
      Func,
      ShortDecimal<P3, S3>,
      LongDecimal<P1, S1>,
      ShortDecimal<P2, S2>>({name}, constraints);
}
} // namespace

void registerDecimalAdd(const std::string& prefix) {
  auto [rPrecision, rScale] = getAddSubtractResultPrecisionScale();
  registerDecimalBinary<AddFunctionAllowPrecisionLoss>(
      prefix + "add", makeConstraints(rPrecision, rScale, true));
  registerDecimalBinary<AddFunctionDenyPrecisionLoss>(
      prefix + "add" + kDenyPrecisionLoss,
      makeConstraints(rPrecision, rScale, false));
  registerDecimalBinary<CheckedAddFunctionAllowPrecisionLoss>(
      prefix + "checked_add", makeConstraints(rPrecision, rScale, true));
  registerDecimalBinary<CheckedAddFunctionDenyPrecisionLoss>(
      prefix + "checked_add" + kDenyPrecisionLoss,
      makeConstraints(rPrecision, rScale, false));
}

void registerDecimalSubtract(const std::string& prefix) {
  auto [rPrecision, rScale] = getAddSubtractResultPrecisionScale();
  registerDecimalBinary<SubtractFunctionAllowPrecisionLoss>(
      prefix + "subtract", makeConstraints(rPrecision, rScale, true));
  registerDecimalBinary<SubtractFunctionDenyPrecisionLoss>(
      prefix + "subtract" + kDenyPrecisionLoss,
      makeConstraints(rPrecision, rScale, false));
  registerDecimalBinary<CheckedSubtractFunctionAllowPrecisionLoss>(
      prefix + "checked_subtract", makeConstraints(rPrecision, rScale, true));
  registerDecimalBinary<CheckedSubtractFunctionDenyPrecisionLoss>(
      prefix + "checked_subtract" + kDenyPrecisionLoss,
      makeConstraints(rPrecision, rScale, false));
}

void registerDecimalMultiply(const std::string& prefix) {
  std::string rPrecision = fmt::format(
      "{a_precision} + {b_precision} + 1",
      fmt::arg("a_precision", P1::name()),
      fmt::arg("b_precision", P2::name()));
  std::string rScale = fmt::format(
      "{a_scale} + {b_scale}",
      fmt::arg("a_scale", S1::name()),
      fmt::arg("b_scale", S2::name()));
  registerDecimalBinary<MultiplyFunctionAllowPrecisionLoss>(
      prefix + "multiply", makeConstraints(rPrecision, rScale, true));
  registerDecimalBinary<MultiplyFunctionDenyPrecisionLoss>(
      prefix + "multiply" + kDenyPrecisionLoss,
      makeConstraints(rPrecision, rScale, false));
  registerDecimalBinary<CheckedMultiplyFunctionAllowPrecisionLoss>(
      prefix + "checked_multiply", makeConstraints(rPrecision, rScale, true));
  registerDecimalBinary<CheckedMultiplyFunctionDenyPrecisionLoss>(
      prefix + "checked_multiply" + kDenyPrecisionLoss,
      makeConstraints(rPrecision, rScale, false));
}

void registerDecimalDivide(const std::string& prefix) {
  registerDecimalDivide<DivideFunctionAllowPrecisionLoss>(
      prefix + "divide", getDivideConstraintsAllowPrecisionLoss());
  registerDecimalDivide<DivideFunctionDenyPrecisionLoss>(
      prefix + "divide" + kDenyPrecisionLoss,
      getDivideConstraintsDenyPrecisionLoss());
}

void registerDecimalIntegralDivide(const std::string& prefix) {
  registerIntegralDecimalDivide<DecimalIntegralDivideFunction>(prefix + "div");
  registerIntegralDecimalDivide<CheckedDecimalIntegralDivideFunction>(
      prefix + "checked_div");
}

void registerDecimalPmod(const std::string& prefix) {
  registerDecimalPmod<PmodFunction>(prefix + "pmod");
  registerDecimalPmod<PmodFunction>(prefix + "pmod" + kDenyPrecisionLoss);
  registerDecimalPmod<CheckedPmodFunction>(prefix + "checked_pmod");
  registerDecimalPmod<CheckedPmodFunction>(
      prefix + "checked_pmod" + kDenyPrecisionLoss);
}
} // namespace facebook::velox::functions::sparksql
