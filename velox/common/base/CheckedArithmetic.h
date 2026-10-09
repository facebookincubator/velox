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

#include <functional>
#include <limits>
#include "folly/Likely.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/base/Macros.h"

#ifdef __CUDACC__
// __builtin_{add,sub,mul}_overflow are host-only, so under nvcc the seam below
// uses CCCL's __host__ __device__ cuda::*_overflow. That needs CCCL 3.4 or
// newer, which not every CUDA toolkit bundles, so the swap is conditional:
// without it, a device instantiation fails on nvcc diagnostic 20011 rather than
// silently computing something else. The seam makes the checks evaluable on
// device but not reportable, since GPU SFI turns VELOX_ARITHMETIC_ERROR into a
// no-op; see TODO(gpu-sfi-checks).
#if __has_include(<cuda/std/version>)
#include <cuda/std/version> // CCCL_VERSION
#endif
#if defined(CCCL_VERSION) && CCCL_VERSION >= 3004000 && __has_include(<cuda/numeric>)
#include <cuda/numeric>
#define VELOX_HAS_DEVICE_OVERFLOW_INTRINSICS 1
#endif
#endif

namespace facebook::velox {

/// Portable __builtin_{add,sub,mul}_overflow, also callable from device code.
/// Returns true on overflow and always stores the wrapped result.
template <typename R, typename A, typename B>
VELOX_GPU_COMPATIBLE bool addWithOverflow(A a, B b, R* result) {
#ifdef VELOX_HAS_DEVICE_OVERFLOW_INTRINSICS
  return cuda::add_overflow(*result, a, b);
#else
  return __builtin_add_overflow(a, b, result);
#endif
}

template <typename R, typename A, typename B>
VELOX_GPU_COMPATIBLE bool subWithOverflow(A a, B b, R* result) {
#ifdef VELOX_HAS_DEVICE_OVERFLOW_INTRINSICS
  return cuda::sub_overflow(*result, a, b);
#else
  return __builtin_sub_overflow(a, b, result);
#endif
}

template <typename R, typename A, typename B>
VELOX_GPU_COMPATIBLE bool mulWithOverflow(A a, B b, R* result) {
#ifdef VELOX_HAS_DEVICE_OVERFLOW_INTRINSICS
  return cuda::mul_overflow(*result, a, b);
#else
  return __builtin_mul_overflow(a, b, result);
#endif
}

template <typename T>
VELOX_GPU_COMPATIBLE T checkedPlus(T a, T b, const char* typeName = "integer") {
  T result;
  bool overflow = addWithOverflow(a, b, &result);
  if (UNLIKELY(overflow)) {
    VELOX_ARITHMETIC_ERROR("{} overflow: {} + {}", typeName, a, b);
  }
  return result;
}

template <typename T>
VELOX_GPU_COMPATIBLE T
checkedMinus(T a, T b, const char* typeName = "integer") {
  T result;
  bool overflow = subWithOverflow(a, b, &result);
  if (UNLIKELY(overflow)) {
    VELOX_ARITHMETIC_ERROR("{} overflow: {} - {}", typeName, a, b);
  }
  return result;
}

template <typename T>
VELOX_GPU_COMPATIBLE T
checkedMultiply(T a, T b, const char* typeName = "integer") {
  T result;
  bool overflow = mulWithOverflow(a, b, &result);
  if (UNLIKELY(overflow)) {
    VELOX_ARITHMETIC_ERROR("{} overflow: {} * {}", typeName, a, b);
  }
  return result;
}

template <typename T>
VELOX_GPU_COMPATIBLE T checkedDivide(T a, T b) {
  if (b == 0) {
    VELOX_ARITHMETIC_ERROR("division by zero");
  }

  // Type T can not represent abs(std::numeric_limits<T>::min()).
  if constexpr (std::is_integral_v<T>) {
    if (UNLIKELY(a == std::numeric_limits<T>::min() && b == -1)) {
      VELOX_ARITHMETIC_ERROR("integer overflow: {} / {}", a, b);
    }
  }
  return a / b;
}

template <typename T>
VELOX_GPU_COMPATIBLE T checkedModulus(T a, T b) {
  if (UNLIKELY(b == 0)) {
    VELOX_ARITHMETIC_ERROR("Cannot divide by 0");
  }
  // std::numeric_limits<int64_t>::min() % -1 could crash the program since
  // abs(std::numeric_limits<int64_t>::min()) can not be represented in
  // int64_t.
  if (b == -1) {
    return 0;
  }
  return (a % b);
}

template <typename T>
VELOX_GPU_COMPATIBLE T checkedNegate(T a) {
  if (UNLIKELY(a == std::numeric_limits<T>::min())) {
    VELOX_ARITHMETIC_ERROR("Cannot negate minimum value");
  }
  return std::negate<std::remove_cv_t<T>>()(a);
}

} // namespace facebook::velox
