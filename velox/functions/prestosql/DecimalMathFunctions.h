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
#pragma once

#include <algorithm>
#include <vector>

#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/ArithmeticImpl.h"
#include "velox/type/DecimalArithmetic.h"

namespace facebook::velox::functions {

template <typename TExec>
struct DecimalPlusFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename A, typename B>
  void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& /*config*/,
      A* /*a*/,
      B* /*b*/) {
    auto aType = inputTypes[0];
    auto bType = inputTypes[1];
    auto aScale = getDecimalPrecisionScale(*aType).second;
    auto bScale = getDecimalPrecisionScale(*bType).second;
    aRescale_ = computeRescaleFactor(aScale, bScale);
    bRescale_ = computeRescaleFactor(bScale, aScale);
  }

  template <typename R, typename A, typename B>
  VELOX_GPU_COMPATIBLE void call(R& out, const A& a, const B& b)
#if defined(__has_feature)
#if __has_feature(__address_sanitizer__)
      __attribute__((__no_sanitize__("signed-integer-overflow")))
#endif
#endif
  {
    int128_t aRescaled;
    int128_t bRescaled;
    if (detail::mulOverflow(
            a, DecimalArithmetic::powerOfTen(aRescale_), &aRescaled) ||
        detail::mulOverflow(
            b, DecimalArithmetic::powerOfTen(bRescale_), &bRescaled)) {
      VELOX_ARITHMETIC_ERROR("Decimal overflow: {} + {}", a, b);
    }
    out = checkedPlus<R>(R(aRescaled), R(bRescaled));
    DecimalArithmetic::valueInRange(out);
  }

 private:
  inline static uint8_t computeRescaleFactor(
      uint8_t fromScale,
      uint8_t toScale) {
    return std::max(0, toScale - fromScale);
  }

  uint8_t aRescale_;
  uint8_t bRescale_;
};

template <typename TExec>
struct DecimalMinusFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename A, typename B>
  void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& /*config*/,
      A* /*a*/,
      B* /*b*/) {
    const auto& aType = inputTypes[0];
    const auto& bType = inputTypes[1];
    auto aScale = getDecimalPrecisionScale(*aType).second;
    auto bScale = getDecimalPrecisionScale(*bType).second;
    aRescale_ = computeRescaleFactor(aScale, bScale);
    bRescale_ = computeRescaleFactor(bScale, aScale);
  }

  template <typename R, typename A, typename B>
  VELOX_GPU_COMPATIBLE void call(R& out, const A& a, const B& b)
#if defined(__has_feature)
#if __has_feature(__address_sanitizer__)
      __attribute__((__no_sanitize__("signed-integer-overflow")))
#endif
#endif
  {
    int128_t aRescaled;
    int128_t bRescaled;
    if (detail::mulOverflow(
            a, DecimalArithmetic::powerOfTen(aRescale_), &aRescaled) ||
        detail::mulOverflow(
            b, DecimalArithmetic::powerOfTen(bRescale_), &bRescaled)) {
      VELOX_ARITHMETIC_ERROR("Decimal overflow: {} - {}", a, b);
    }
    out = checkedMinus<R>(R(aRescaled), R(bRescaled));
    DecimalArithmetic::valueInRange(out);
  }

 private:
  inline static uint8_t computeRescaleFactor(
      uint8_t fromScale,
      uint8_t toScale) {
    return std::max(0, toScale - fromScale);
  }

  uint8_t aRescale_;
  uint8_t bRescale_;
};

template <typename TExec>
struct DecimalMultiplyFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename R, typename A, typename B>
  VELOX_GPU_COMPATIBLE void call(R& out, const A& a, const B& b) {
    out = checkedMultiply<R>(checkedMultiply<R>(R(a), R(b)), R(1));
    DecimalArithmetic::valueInRange(out);
  }
};

template <typename TExec>
struct DecimalDivideFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename A, typename B>
  void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& /*config*/,
      A* /*a*/,
      B* /*b*/) {
    auto aType = inputTypes[0];
    auto bType = inputTypes[1];
    auto aScale = getDecimalPrecisionScale(*aType).second;
    auto bScale = getDecimalPrecisionScale(*bType).second;
    auto rScale = std::max(aScale, bScale);
    aRescale_ = rScale - aScale + bScale;
    VELOX_USER_CHECK_LE(
        aRescale_, DecimalArithmetic::kMaxLongPrecision, "Decimal overflow");
  }

  template <typename R, typename A, typename B>
  VELOX_GPU_COMPATIBLE void call(R& out, const A& a, const B& b) {
    DecimalArithmetic::divideWithRoundUp<R, A, B>(
        out, a, b, false, aRescale_, 0);
    DecimalArithmetic::valueInRange(out);
  }

 private:
  uint8_t aRescale_;
};

template <typename TExec>
struct DecimalModulusFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename A, typename B>
  void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& /*config*/,
      A* /*a*/,
      B* /*b*/) {
    const auto& aType = inputTypes[0];
    const auto& bType = inputTypes[1];
    auto [aPrecision, aScale] = getDecimalPrecisionScale(*aType);
    auto [bPrecision, bScale] = getDecimalPrecisionScale(*bType);
    aRescale_ = std::max(0, bScale - aScale);
    bRescale_ = std::max(0, aScale - bScale);
  }

  template <typename R, typename A, typename B>
  VELOX_GPU_COMPATIBLE void call(R& out, const A& a, const B& b) {
    VELOX_USER_CHECK_NE(b, 0, "Modulus by zero");
    int remainderSign = 1;
    R unsignedDividendRescaled(a);
    if (a < 0) {
      remainderSign *= -1;
      unsignedDividendRescaled *= -1;
    }
    unsignedDividendRescaled = checkedMultiply<R>(
        unsignedDividendRescaled,
        R(DecimalArithmetic::powerOfTen(aRescale_)),
        "Decimal");

    R unsignedDivisorRescaled(b);
    if (b < 0) {
      unsignedDivisorRescaled *= -1;
    }
    unsignedDivisorRescaled = checkedMultiply<B>(
        unsignedDivisorRescaled,
        R(DecimalArithmetic::powerOfTen(bRescale_)),
        "Decimal");

    R remainder = unsignedDividendRescaled % unsignedDivisorRescaled;
    out = remainder * remainderSign;
  }

 private:
  uint8_t aRescale_;
  uint8_t bRescale_;
};

template <typename TExec>
struct DecimalRoundFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename A>
  void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& config,
      A* a) {
    initialize(inputTypes, config, a, nullptr);
  }

  template <typename A>
  void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& /*config*/,
      A* /*a*/,
      const int32_t* /*n*/) {
    const auto [precision, scale] = getDecimalPrecisionScale(*inputTypes[0]);
    precision_ = precision;
    scale_ = scale;
  }

  template <typename R, typename A>
  VELOX_GPU_COMPATIBLE void call(R& out, const A& a) {
    DecimalArithmetic::divideWithRoundUp<R, A, int128_t>(
        out, a, DecimalArithmetic::powerOfTen(scale_), false, 0, 0);
  }

  template <typename R, typename A>
  VELOX_GPU_COMPATIBLE void call(R& out, const A& a, int32_t n) {
    if (a == 0 || precision_ - scale_ + n <= 0) {
      out = 0;
      return;
    }
    if (n >= scale_) {
      out = a;
      return;
    }
    auto reScaleFactor = DecimalArithmetic::powerOfTen(scale_ - n);
    DecimalArithmetic::divideWithRoundUp<R, A, int128_t>(
        out, a, reScaleFactor, false, 0, 0);
    out *= reScaleFactor;
  }

 private:
  uint8_t precision_;
  uint8_t scale_;
};

template <typename TExec>
struct DecimalFloorFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename A>
  void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& /*config*/,
      A* /*a*/) {
    scale_ = getDecimalPrecisionScale(*inputTypes[0]).second;
  }

  template <typename R, typename A>
  VELOX_GPU_COMPATIBLE void call(R& out, const A& a) {
    const auto rescaleFactor = DecimalArithmetic::powerOfTen(scale_);
    // Round rowards -INF.
    const auto increment = (a % rescaleFactor) < 0 ? -1 : 0;
    out = a / rescaleFactor + increment;
  }

 private:
  uint8_t scale_;
};

template <typename TExec>
struct DecimalCeilFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename A>
  void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& /*config*/,
      A* /*a*/) {
    scale_ = getDecimalPrecisionScale(*inputTypes[0]).second;
  }

  template <typename R, typename A>
  VELOX_GPU_COMPATIBLE void call(R& out, const A& a) {
    const auto rescaleFactor = DecimalArithmetic::powerOfTen(scale_);
    // Round towards +INF.
    const auto increment = (a % rescaleFactor) > 0 ? 1 : 0;
    out = a / rescaleFactor + increment;
  }

 private:
  uint8_t scale_;
};

template <typename TExec>
struct DecimalTruncateFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  template <typename A>
  void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& /*config*/,
      A* /*a*/) {
    const auto [precision, scale] = getDecimalPrecisionScale(*inputTypes[0]);
    precision_ = precision;
    scale_ = scale;
  }

  template <typename A>
  void initialize(
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& config,
      A* a,
      const int32_t* /*n*/) {
    initialize(inputTypes, config, a);
  }

  template <typename R, typename A>
  VELOX_GPU_COMPATIBLE void call(R& out, const A& a) {
    if UNLIKELY (scale_ == 0 || a == 0) {
      out = a;
    } else {
      out = a / DecimalArithmetic::powerOfTen(scale_);
    }
  }

  template <typename A>
  VELOX_GPU_COMPATIBLE void call(A& out, const A& a, int32_t n) {
    if UNLIKELY (a == 0 || (n + precision_ - scale_) <= 0) {
      out = 0;
    } else if UNLIKELY (scale_ <= n) {
      out = a;
    } else {
      out = a - (a % DecimalArithmetic::powerOfTen(scale_ - n));
    }
  }

 private:
  uint8_t precision_;
  uint8_t scale_;
};

} // namespace facebook::velox::functions
