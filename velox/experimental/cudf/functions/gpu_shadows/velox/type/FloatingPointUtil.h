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

// GPU shadow for velox/type/FloatingPointUtil.h
// Provides the NaN-aware comparator/hash functors without Folly deps.
// Uses cuda::std::isnan for guaranteed device compatibility (CCCL).
#pragma once

#include <cuda/std/cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <type_traits>

namespace facebook::velox {

namespace util::floating_point {

template <
    typename FLOAT,
    std::enable_if_t<std::is_floating_point<FLOAT>::value, bool> = true>
struct NaNAwareEquals {
  __host__ __device__ bool operator()(const FLOAT& lhs, const FLOAT& rhs)
      const {
    if (cuda::std::isnan(lhs) && cuda::std::isnan(rhs)) {
      return true;
    }
    return lhs == rhs;
  }
};

template <
    typename FLOAT,
    std::enable_if_t<std::is_floating_point<FLOAT>::value, bool> = true>
struct NaNAwareLessThan {
  __host__ __device__ bool operator()(const FLOAT& lhs, const FLOAT& rhs)
      const {
    if (!cuda::std::isnan(lhs) && cuda::std::isnan(rhs)) {
      return true;
    }
    return lhs < rhs;
  }
};

template <
    typename FLOAT,
    std::enable_if_t<std::is_floating_point<FLOAT>::value, bool> = true>
struct NaNAwareLessThanEqual {
  __host__ __device__ bool operator()(const FLOAT& lhs, const FLOAT& rhs)
      const {
    if (cuda::std::isnan(rhs)) {
      return true;
    }
    return lhs <= rhs;
  }
};

template <
    typename FLOAT,
    std::enable_if_t<std::is_floating_point<FLOAT>::value, bool> = true>
struct NaNAwareGreaterThan {
  __host__ __device__ bool operator()(const FLOAT& lhs, const FLOAT& rhs)
      const {
    if (cuda::std::isnan(lhs) && !cuda::std::isnan(rhs)) {
      return true;
    }
    return lhs > rhs;
  }
};

template <
    typename FLOAT,
    std::enable_if_t<std::is_floating_point<FLOAT>::value, bool> = true>
struct NaNAwareGreaterThanEqual {
  __host__ __device__ bool operator()(const FLOAT& lhs, const FLOAT& rhs)
      const {
    if (cuda::std::isnan(lhs)) {
      return true;
    }
    return lhs >= rhs;
  }
};

template <
    typename FLOAT,
    std::enable_if_t<std::is_floating_point<FLOAT>::value, bool> = true>
struct NaNAwareHash {
  std::size_t operator()(const FLOAT& val) const noexcept {
    static const std::size_t kNanHash =
        std::hash<FLOAT>{}(std::numeric_limits<FLOAT>::quiet_NaN());
    if (cuda::std::isnan(val)) {
      return kNanHash;
    }
    return std::hash<FLOAT>{}(val);
  }
};

} // namespace util::floating_point

class DoubleUtil {
 public:
  static const std::array<double, 309> kPowersOfTen;
};

} // namespace facebook::velox
