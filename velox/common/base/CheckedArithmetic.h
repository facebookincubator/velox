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
// __builtin_{add,sub,mul}_overflow are host-only: nvcc rejects them in device
// code at every width. CCCL's equivalents are __host__ __device__ and lower to
// PTX on device and to the same compiler intrinsics on host, so this swap
// costs nothing where it is taken and is invisible to non-CUDA builds.
#include <cuda/numeric>
#endif

namespace facebook::velox {

namespace detail {

// Thin seam over the overflow primitive so the checked* bodies below read the
// same on both compilers. Each returns true on overflow and always stores the
// wrapped result, matching __builtin_*_overflow exactly.
template <typename R, typename A, typename B>
VELOX_GPU_COMPATIBLE bool addOverflow(A a, B b, R* result) {
#ifdef __CUDACC__
  return cuda::add_overflow(*result, a, b);
#else
  return __builtin_add_overflow(a, b, result);
#endif
}

template <typename R, typename A, typename B>
VELOX_GPU_COMPATIBLE bool subOverflow(A a, B b, R* result) {
#ifdef __CUDACC__
  return cuda::sub_overflow(*result, a, b);
#else
  return __builtin_sub_overflow(a, b, result);
#endif
}

template <typename R, typename A, typename B>
VELOX_GPU_COMPATIBLE bool mulOverflow(A a, B b, R* result) {
#ifdef __CUDACC__
  return cuda::mul_overflow(*result, a, b);
#else
  return __builtin_mul_overflow(a, b, result);
#endif
}

} // namespace detail

template <typename T>
VELOX_GPU_COMPATIBLE T checkedPlus(T a, T b, const char* typeName = "integer") {
  T result;
  bool overflow = detail::addOverflow(a, b, &result);
  if (UNLIKELY(overflow)) {
    VELOX_ARITHMETIC_ERROR("{} overflow: {} + {}", typeName, a, b);
  }
  return result;
}

template <typename T>
VELOX_GPU_COMPATIBLE T
checkedMinus(T a, T b, const char* typeName = "integer") {
  T result;
  bool overflow = detail::subOverflow(a, b, &result);
  if (UNLIKELY(overflow)) {
    VELOX_ARITHMETIC_ERROR("{} overflow: {} - {}", typeName, a, b);
  }
  return result;
}

template <typename T>
VELOX_GPU_COMPATIBLE T
checkedMultiply(T a, T b, const char* typeName = "integer") {
  T result;
  bool overflow = detail::mulOverflow(a, b, &result);
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
