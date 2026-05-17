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

// Binary arithmetic.

template <typename T, typename TAlpha>
__device__ inline T __add(T a1, T a2, TAlpha alpha) {
  return a1 + a2 * alpha;
}

template <typename T, typename TAlpha>
__device__ inline T __sub(T a1, T a2, TAlpha alpha) {
  return a1 - a2 * alpha;
}

template <typename T>
__device__ inline T __mul(T a1, T a2) {
  return a1 * a2;
}

template <typename T>
__device__ inline T __div(T a1, T a2) {
  return a1 / a2;
}

template <typename T>
__device__ inline T __remainder(T a1, T a2) {
  return a1 % a2;
}

__device__ inline float __remainder(float a1, float a2) {
  return remainderf(a1, a2);
}

__device__ inline double __remainder(double a1, double a2) {
  return ::remainder(a1, a2);
}

template <typename T>
__device__ inline T __fmod(T a1, T a2) {
  return a1 % a2;
}

__device__ inline float __fmod(float a1, float a2) {
  return fmodf(a1, a2);
}

__device__ inline double __fmod(double a1, double a2) {
  return fmod(a1, a2);
}

template <typename T>
__device__ inline T __pow(T a1, T a2) {
  return powf(static_cast<float>(a1), static_cast<float>(a2));
}

__device__ inline float __pow(float a1, float a2) {
  return powf(a1, a2);
}

__device__ inline double __pow(double a1, double a2) {
  return pow(a1, a2);
}

// Comparison.

template <typename T>
__device__ inline bool __eq(T a1, T a2) {
  return a1 == a2;
}

template <typename T>
__device__ inline bool __ne(T a1, T a2) {
  return a1 != a2;
}

template <typename T>
__device__ inline bool __lt(T a1, T a2) {
  return a1 < a2;
}

template <typename T>
__device__ inline bool __le(T a1, T a2) {
  return a1 <= a2;
}

template <typename T>
__device__ inline bool __gt(T a1, T a2) {
  return a1 > a2;
}

template <typename T>
__device__ inline bool __ge(T a1, T a2) {
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

template <typename T>
__device__ inline T __clamp(T a1, T lo, T hi) {
  return a1 < lo ? lo : (a1 > hi ? hi : a1);
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

// Zero.

__device__ inline int32_t __zero() {
  return 0;
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
