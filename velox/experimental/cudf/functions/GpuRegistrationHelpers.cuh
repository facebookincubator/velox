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

#include "velox/experimental/cudf/functions/GpuSimpleFunctionAdapter.cuh"

/// Type-set registration helpers, mirroring velox/functions/lib/
/// RegistrationHelpers.h name for name.
///
/// The point is not brevity, though they are shorter. It is that a GPU
/// registration file can now be read side by side with the Velox registration
/// file it mirrors: where Velox writes registerUnaryNumeric<CeilFunction>, so
/// does this, and a difference in type coverage becomes visible as a difference
/// in helper rather than something you have to reconstruct by counting
/// overloads.
///
/// Velox's versions live in an anonymous namespace inside the header; these do
/// not, because more than one dialect translation unit calls them.
namespace facebook::velox::cudf_velox::gpu_sfi {

template <template <class> typename Fn>
void registerGpuBinaryIntegral(const std::vector<std::string>& aliases) {
  registerGpuFunction<Fn, int8_t, int8_t, int8_t>(aliases);
  registerGpuFunction<Fn, int16_t, int16_t, int16_t>(aliases);
  registerGpuFunction<Fn, int32_t, int32_t, int32_t>(aliases);
  registerGpuFunction<Fn, int64_t, int64_t, int64_t>(aliases);
}

template <template <class> typename Fn>
void registerGpuBinaryFloatingPoint(const std::vector<std::string>& aliases) {
  registerGpuFunction<Fn, double, double, double>(aliases);
  registerGpuFunction<Fn, float, float, float>(aliases);
}

template <template <class> typename Fn>
void registerGpuBinaryNumeric(const std::vector<std::string>& aliases) {
  registerGpuBinaryIntegral<Fn>(aliases);
  registerGpuBinaryFloatingPoint<Fn>(aliases);
}

template <template <class> typename Fn, typename TReturn>
void registerGpuBinaryNumericWithTReturn(
    const std::vector<std::string>& aliases) {
  registerGpuFunction<Fn, TReturn, int8_t, int8_t>(aliases);
  registerGpuFunction<Fn, TReturn, int16_t, int16_t>(aliases);
  registerGpuFunction<Fn, TReturn, int32_t, int32_t>(aliases);
  registerGpuFunction<Fn, TReturn, int64_t, int64_t>(aliases);
  registerGpuFunction<Fn, TReturn, double, double>(aliases);
  registerGpuFunction<Fn, TReturn, float, float>(aliases);
}

template <template <class> typename Fn>
void registerGpuUnaryIntegral(const std::vector<std::string>& aliases) {
  registerGpuFunction<Fn, int8_t, int8_t>(aliases);
  registerGpuFunction<Fn, int16_t, int16_t>(aliases);
  registerGpuFunction<Fn, int32_t, int32_t>(aliases);
  registerGpuFunction<Fn, int64_t, int64_t>(aliases);
}

template <template <class> typename Fn>
void registerGpuUnaryFloatingPoint(const std::vector<std::string>& aliases) {
  registerGpuFunction<Fn, double, double>(aliases);
  registerGpuFunction<Fn, float, float>(aliases);
}

template <template <class> typename Fn>
void registerGpuUnaryNumeric(const std::vector<std::string>& aliases) {
  registerGpuUnaryIntegral<Fn>(aliases);
  registerGpuUnaryFloatingPoint<Fn>(aliases);
}

/// Three same-typed arguments, one result of that type -- clamp's shape.
template <template <class> typename Fn>
void registerGpuTernaryNumeric(const std::vector<std::string>& aliases) {
  registerGpuFunction<Fn, int8_t, int8_t, int8_t, int8_t>(aliases);
  registerGpuFunction<Fn, int16_t, int16_t, int16_t, int16_t>(aliases);
  registerGpuFunction<Fn, int32_t, int32_t, int32_t, int32_t>(aliases);
  registerGpuFunction<Fn, int64_t, int64_t, int64_t, int64_t>(aliases);
  registerGpuFunction<Fn, double, double, double, double>(aliases);
  registerGpuFunction<Fn, float, float, float, float>(aliases);
}

/// Three same-typed arguments, one result of a fixed type -- between's shape.
template <template <class> typename Fn, typename TReturn>
void registerGpuTernaryNumericWithTReturn(
    const std::vector<std::string>& aliases) {
  registerGpuFunction<Fn, TReturn, int8_t, int8_t, int8_t>(aliases);
  registerGpuFunction<Fn, TReturn, int16_t, int16_t, int16_t>(aliases);
  registerGpuFunction<Fn, TReturn, int32_t, int32_t, int32_t>(aliases);
  registerGpuFunction<Fn, TReturn, int64_t, int64_t, int64_t>(aliases);
  registerGpuFunction<Fn, TReturn, double, double, double>(aliases);
  registerGpuFunction<Fn, TReturn, float, float, float>(aliases);
}

/// A value plus an integer number of decimal places -- round and truncate.
/// Velox writes these out one per type rather than with a helper; collected
/// here because both functions need the identical set.
template <template <class> typename Fn>
void registerGpuNumericWithDecimals(const std::vector<std::string>& aliases) {
  registerGpuFunction<Fn, int8_t, int8_t, int32_t>(aliases);
  registerGpuFunction<Fn, int16_t, int16_t, int32_t>(aliases);
  registerGpuFunction<Fn, int32_t, int32_t, int32_t>(aliases);
  registerGpuFunction<Fn, int64_t, int64_t, int32_t>(aliases);
  registerGpuFunction<Fn, double, double, int32_t>(aliases);
  registerGpuFunction<Fn, float, float, int32_t>(aliases);
}

} // namespace facebook::velox::cudf_velox::gpu_sfi
