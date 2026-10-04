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

#include "velox/experimental/cudf/functions/GpuCheckFailure.h"

#include <fmt/format.h>

#include <cstdint>
#include <string>

/// Where a failed VELOX_CHECK in a device-compiled Velox function body records
/// itself. A check site sits several frames below the kernel and is passed
/// nothing, so it can reach only its thread index and memory it can name
/// without a pointer. It uses the launch's dynamic shared memory, which is per
/// launch and per block, so concurrent launches on different streams cannot
/// collide as they would through one __device__ pointer per translation unit.
namespace facebook::velox::cudf_velox::gpu_sfi {

/// One byte per thread of the block, in the launch's dynamic shared memory. A
/// kernel that runs a Velox body requests at least blockDim.x bytes and zeroes
/// its thread's byte on entry; the adapter does both. Declared outside any
/// __CUDA_ARCH__ guard because nvcc's host pass also parses the device
/// functions that name it.
extern __shared__ uint8_t gpuErrorBytes[];

namespace detail {

/// A message argument as fmt can take it. fmt leaves __int128 unsupported in
/// a translation unit nvcc compiles, so the decimal checks' arguments are
/// spelled out here.
template <typename T>
const T& formattable(const T& value) {
  return value;
}

inline std::string formattable(unsigned __int128 value) {
  std::string digits;
  do {
    digits.insert(digits.begin(), static_cast<char>('0' + value % 10));
    value /= 10;
  } while (value != 0);
  return digits;
}

inline std::string formattable(__int128 value) {
  if (value < 0) {
    // Negated in two steps, since the most negative value has no positive.
    return "-" + formattable(static_cast<unsigned __int128>(-(value + 1)) + 1);
  }
  return formattable(static_cast<unsigned __int128>(value));
}

template <typename Format, typename... Args>
std::string formatMapped(const Format& format, const Args&... args) {
  return fmt::vformat(fmt::string_view(format), fmt::make_format_args(args...));
}

/// The message of a failed check, formatted as the real macro formats it: the
/// first argument is the format string. A check with no message has none.
inline std::string formatCheckMessage() {
  return {};
}

template <typename Format, typename... Args>
std::string formatCheckMessage(const Format& format, const Args&... args) {
  return formatMapped(format, formattable(args)...);
}

} // namespace detail

/// Records that this thread's row failed a check. It does not return: the body
/// runs on with the rejected data, and the row is declined. The first failure
/// wins, so the kind names the earliest rejected precondition. Each byte
/// belongs to one thread, so no atomics are needed. The message arguments are
/// not formatted on the device; the host re-evaluates the row for them.
///
/// On the host, where initialize() runs real function bodies, a failed check
/// throws the Velox error with its message instead, as the real macro would.
template <typename... Args>
__host__ __device__ inline void gpuRaise(
    GpuErrorKind kind,
    const Args&... messageArguments) {
#ifdef __CUDA_ARCH__
  if (gpuErrorBytes[threadIdx.x] == static_cast<uint8_t>(GpuErrorKind::kNone)) {
    gpuErrorBytes[threadIdx.x] = static_cast<uint8_t>(kind);
  }
#else
  throwCheckFailure(kind, detail::formatCheckMessage(messageArguments...));
#endif
}

/// Whether this thread's row has already failed a check in the running body.
/// The body runs on after a failed check, so a helper that check was guarding
/// can ask before the rejected value sends it past its caller's memory.
__host__ __device__ inline bool gpuRowFailed() {
#ifdef __CUDA_ARCH__
  return gpuErrorBytes[threadIdx.x] !=
      static_cast<uint8_t>(GpuErrorKind::kNone);
#else
  return false;
#endif
}

} // namespace facebook::velox::cudf_velox::gpu_sfi
