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

#include <cstdint>

namespace torch::wave {

// Casts __half and __nv_bfloat16 to float for arithmetic so that mixed-type
// expressions (e.g. int32_t * __half) do not hit ambiguous operator overloads
// from cuda_fp16.h / cuda_bf16.h.
template <typename T>
__device__ inline T arithCast(T x) {
  return x;
}
__device__ inline float arithCast(__half x) {
  return __half2float(x);
}
__device__ inline float arithCast(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

// Binary arithmetic.

template <typename T, typename T2, typename TAlpha>
__device__ inline T __add(T a1, T2 a2, TAlpha alpha) {
  return arithCast(a1) + arithCast(a2) * arithCast(alpha);
}

template <typename T, typename T2, typename TAlpha>
__device__ inline T __sub(T a1, T2 a2, TAlpha alpha) {
  return arithCast(a1) - arithCast(a2) * arithCast(alpha);
}

template <typename T, typename T2 = T>
__device__ inline T __mul(T a1, T2 a2) {
  return arithCast(a1) * arithCast(a2);
}

template <>
__device__ inline bool __mul<bool, bool>(bool a1, bool a2) {
  return a1 & a2;
}

template <typename T, typename T2 = T>
__device__ inline T __div(T a1, T2 a2) {
  return arithCast(a1) / arithCast(a2);
}

// Integer true division matching PyTorch semantics: div(int, int) -> double.
__device__ inline double __div(int64_t a1, int64_t a2) {
  return static_cast<double>(a1) / static_cast<double>(a2);
}

__device__ inline double __div(int32_t a1, int32_t a2) {
  return static_cast<double>(a1) / static_cast<double>(a2);
}

template <typename T, typename T2 = T>
__device__ inline T __remainder(T a1, T2 a2) {
  return a1 % a2;
}

__device__ inline float __remainder(float a1, float a2) {
  return remainderf(a1, a2);
}

__device__ inline double __remainder(double a1, double a2) {
  return ::remainder(a1, a2);
}

template <typename T, typename T2 = T>
__device__ inline T __fmod(T a1, T2 a2) {
  return a1 % a2;
}

__device__ inline float __fmod(float a1, float a2) {
  return fmodf(a1, a2);
}

__device__ inline double __fmod(double a1, double a2) {
  return fmod(a1, a2);
}

template <typename T, typename T2 = T>
__device__ inline T __pow(T a1, T2 a2) {
  return powf(static_cast<float>(a1), static_cast<float>(a2));
}

__device__ inline float __pow(float a1, float a2) {
  return powf(a1, a2);
}

__device__ inline double __pow(double a1, double a2) {
  return pow(a1, a2);
}

// Comparison.

template <typename T, typename T2 = T>
__device__ inline bool __eq(T a1, T2 a2) {
  return a1 == a2;
}

template <typename T, typename T2 = T>
__device__ inline bool __ne(T a1, T2 a2) {
  return a1 != a2;
}

template <typename T, typename T2 = T>
__device__ inline bool __lt(T a1, T2 a2) {
  return a1 < a2;
}

template <typename T, typename T2 = T>
__device__ inline bool __le(T a1, T2 a2) {
  return a1 <= a2;
}

template <typename T, typename T2 = T>
__device__ inline bool __gt(T a1, T2 a2) {
  return a1 > a2;
}

template <typename T, typename T2 = T>
__device__ inline bool __ge(T a1, T2 a2) {
  return a1 >= a2;
}

// Bitwise.

template <typename T>
__device__ inline T __bitwise_and(T a1, T a2) {
  return a1 & a2;
}

template <typename T>
__device__ inline T __bitwise_or(T a1, T a2) {
  return a1 | a2;
}

template <typename T>
__device__ inline T __bitwise_xor(T a1, T a2) {
  return a1 ^ a2;
}

template <typename T>
__device__ inline T __bitwise_not(T a1) {
  return ~a1;
}

template <>
__device__ inline bool __bitwise_not<bool>(bool a1) {
  return !a1;
}

// Logical.

template <typename T>
__device__ inline bool __logical_and(T a1, T a2) {
  return static_cast<bool>(a1) && static_cast<bool>(a2);
}

template <typename T>
__device__ inline bool __logical_or(T a1, T a2) {
  return static_cast<bool>(a1) || static_cast<bool>(a2);
}

template <typename T>
__device__ inline bool __logical_xor(T a1, T a2) {
  return static_cast<bool>(a1) != static_cast<bool>(a2);
}

template <typename T>
__device__ inline bool __logical_not(T a1) {
  return !static_cast<bool>(a1);
}

// Unary math.

template <typename T>
__device__ inline T __abs(T a1) {
  return abs(a1);
}

__device__ inline float __abs(float a1) {
  return fabsf(a1);
}

__device__ inline double __abs(double a1) {
  return fabs(a1);
}

template <typename T>
__device__ inline T __neg(T a1) {
  return -a1;
}

__device__ inline float __ceil(float a1) {
  return ceilf(a1);
}

__device__ inline double __ceil(double a1) {
  return ceil(a1);
}

__device__ inline float __floor(float a1) {
  return floorf(a1);
}

__device__ inline double __floor(double a1) {
  return floor(a1);
}

__device__ inline float __round(float a1) {
  return roundf(a1);
}

__device__ inline double __round(double a1) {
  return round(a1);
}

__device__ inline float __trunc(float a1) {
  return truncf(a1);
}

__device__ inline double __trunc(double a1) {
  return trunc(a1);
}

template <typename T>
__device__ inline T __sign(T a1) {
  return (a1 > T(0)) - (a1 < T(0));
}

__device__ inline float __sqrt(float a1) {
  return sqrtf(a1);
}

__device__ inline double __sqrt(double a1) {
  return sqrt(a1);
}

__device__ inline float __rsqrt(float a1) {
  return rsqrtf(a1);
}

__device__ inline double __rsqrt(double a1) {
  return 1.0 / sqrt(a1);
}

__device__ inline float __reciprocal(float a1) {
  return 1.0f / a1;
}

__device__ inline double __reciprocal(double a1) {
  return 1.0 / a1;
}

__device__ inline float __exp(float a1) {
  return expf(a1);
}

__device__ inline double __exp(double a1) {
  return exp(a1);
}

__device__ inline float __log(float a1) {
  return logf(a1);
}

__device__ inline double __log(double a1) {
  return log(a1);
}

__device__ inline float __log2(float a1) {
  return log2f(a1);
}

__device__ inline double __log2(double a1) {
  return log2(a1);
}

__device__ inline float __log10(float a1) {
  return log10f(a1);
}

__device__ inline double __log10(double a1) {
  return log10(a1);
}

__device__ inline float __log1p(float a1) {
  return log1pf(a1);
}

__device__ inline double __log1p(double a1) {
  return log1p(a1);
}

// Trigonometric.

__device__ inline float __sin(float a1) {
  return sinf(a1);
}

__device__ inline double __sin(double a1) {
  return sin(a1);
}

__device__ inline float __cos(float a1) {
  return cosf(a1);
}

__device__ inline double __cos(double a1) {
  return cos(a1);
}

__device__ inline float __tan(float a1) {
  return tanf(a1);
}

__device__ inline double __tan(double a1) {
  return tan(a1);
}

__device__ inline float __asin(float a1) {
  return asinf(a1);
}

__device__ inline double __asin(double a1) {
  return asin(a1);
}

__device__ inline float __acos(float a1) {
  return acosf(a1);
}

__device__ inline double __acos(double a1) {
  return acos(a1);
}

__device__ inline float __atan(float a1) {
  return atanf(a1);
}

__device__ inline double __atan(double a1) {
  return atan(a1);
}

__device__ inline float __atan2(float a1, float a2) {
  return atan2f(a1, a2);
}

__device__ inline double __atan2(double a1, double a2) {
  return atan2(a1, a2);
}

__device__ inline float __sinh(float a1) {
  return sinhf(a1);
}

__device__ inline double __sinh(double a1) {
  return sinh(a1);
}

__device__ inline float __cosh(float a1) {
  return coshf(a1);
}

__device__ inline double __cosh(double a1) {
  return cosh(a1);
}

__device__ inline float __tanh(float a1) {
  return tanhf(a1);
}

__device__ inline double __tanh(double a1) {
  return tanh(a1);
}

// Activation functions.

template <typename T>
__device__ inline T __relu(T a1) {
  return a1 > T(0) ? a1 : T(0);
}

__device__ inline float __sigmoid(float a1) {
  return 1.0f / (1.0f + expf(-a1));
}

__device__ inline double __sigmoid(double a1) {
  return 1.0 / (1.0 + exp(-a1));
}

template <
    bool kHasMin = true,
    bool kHasMax = true,
    typename T,
    typename TLo = T,
    typename THi = T>
__device__ inline T __clamp(T a1, TLo lo, THi hi) {
  T result = a1;
  if constexpr (kHasMin) {
    result = result < static_cast<T>(lo) ? static_cast<T>(lo) : result;
  }
  if constexpr (kHasMax) {
    result = result > static_cast<T>(hi) ? static_cast<T>(hi) : result;
  }
  return result;
}

// NaN/Inf replacement.

template <typename T>
__device__ inline T
__nan_to_num(T a1, double nan_val, double posinf_val, double neginf_val) {
  return a1;
}

__device__ inline float
__nan_to_num(float a1, double nan_val, double posinf_val, double neginf_val) {
  if (isnan(a1)) {
    return static_cast<float>(nan_val);
  }
  if (isinf(a1)) {
    return a1 > 0 ? static_cast<float>(posinf_val)
                  : static_cast<float>(neginf_val);
  }
  return a1;
}

__device__ inline double
__nan_to_num(double a1, double nan_val, double posinf_val, double neginf_val) {
  if (isnan(a1)) {
    return nan_val;
  }
  if (isinf(a1)) {
    return a1 > 0 ? posinf_val : neginf_val;
  }
  return a1;
}

// Min/max.

template <typename T>
__device__ inline T __minimum(T a1, T a2) {
  return a1 < a2 ? a1 : a2;
}

template <typename T>
__device__ inline T __maximum(T a1, T a2) {
  return a1 > a2 ? a1 : a2;
}

// Item.

template <typename T>
__device__ inline T __item(Tensor* self) {
  return *storage<T>(self);
}

// Where.

template <typename T>
__device__ inline T __where(bool c, T x, T y) {
  return c ? x : y;
}

// Zero / One.

template <typename T>
__device__ inline T __zero() {
  return T(0);
}

template <typename T>
__device__ inline T __one() {
  return T(1);
}

template <typename T, typename U>
__device__ inline T __one(U /*ignore*/) {
  return T(1);
}

// Arange.

template <typename T>
__device__ inline T __arange_start(int64_t idx, T start) {
  return idx + start;
}

template <typename T>
__device__ inline T __arange(int64_t idx) {
  return __arange_start<T>(idx, T(0));
}

// Shape query.

__device__ inline int64_t __sym_size(Tensor* self, int64_t dim) {
  return self->dims[self->rank - 1 - dim];
}

} // namespace torch::wave
