/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <type_traits>

#include "velox/common/base/SimdUtil.h"

namespace facebook::nimble {

template <typename T>
struct MinMax {
  T min;
  T max;
};

template <typename T>
constexpr bool kIntegralMinMaxType = std::is_integral_v<std::remove_cv_t<T>> &&
    !std::is_same_v<std::remove_cv_t<T>, bool>;

template <typename T>
constexpr bool kFloatingPointMinMaxType =
    std::is_floating_point_v<std::remove_cv_t<T>>;

template <typename T>
std::enable_if_t<kIntegralMinMaxType<T>, MinMax<std::remove_cv_t<T>>>
findMinMax(std::span<T> values) {
  using Value = std::remove_cv_t<T>;

  Value minValue{values.front()};
  Value maxValue{values.front()};

  for (const Value value : values) {
    minValue = std::min(minValue, value);
    maxValue = std::max(maxValue, value);
  }

  return {
      .min = minValue,
      .max = maxValue,
  };
}

template <typename T>
std::enable_if_t<
    kFloatingPointMinMaxType<T>,
    std::optional<MinMax<std::remove_cv_t<T>>>>
findMinMax(std::span<T> values) {
  using Value = std::remove_cv_t<T>;
  using Batch = xsimd::batch<Value>;

  const auto* rawValues = values.data();
  size_t index{0};
  Value minValue{values.front()};
  Value maxValue{values.front()};

  if (std::isnan(minValue)) {
    return std::nullopt;
  }

  if (values.size() >= Batch::size) {
    auto minBatch = Batch::load_unaligned(rawValues);
    if (xsimd::any(minBatch != minBatch)) {
      return std::nullopt;
    }
    auto maxBatch = minBatch;
    index = Batch::size;
    for (; index + Batch::size <= values.size(); index += Batch::size) {
      const auto batch = Batch::load_unaligned(rawValues + index);
      if (xsimd::any(batch != batch)) {
        return std::nullopt;
      }
      minBatch = xsimd::min(minBatch, batch);
      maxBatch = xsimd::max(maxBatch, batch);
    }
    minValue = xsimd::reduce_min(minBatch);
    maxValue = xsimd::reduce_max(maxBatch);
  }

  for (; index < values.size(); ++index) {
    const auto value = values[index];
    if (std::isnan(value)) {
      return std::nullopt;
    }
    minValue = std::min(minValue, value);
    maxValue = std::max(maxValue, value);
  }

  return MinMax<Value>{
      .min = minValue,
      .max = maxValue,
  };
}

} // namespace facebook::nimble
