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

/// Where a failed VELOX_CHECK in a device-compiled Velox function body records
/// itself. A check site sits several frames below the kernel and is passed
/// nothing, so it can reach only its thread index and memory it can name
/// without a pointer. It uses the launch's dynamic shared memory, which is per
/// launch and per block, so concurrent launches on different streams cannot
/// collide as they would through one __device__ pointer per translation unit.
namespace facebook::velox::cudf_velox::gpu_sfi {

/// The class of error a declined row hit. The host re-evaluates the row through
/// Velox for the message, so this carries only what decides policy: whether a
/// TRY may turn the row into a null, which EvalCtx::setStatus allows for user
/// errors and not for runtime errors.
enum class GpuErrorKind : uint8_t {
  kNone = 0,
  /// VELOX_USER_CHECK*, VELOX_USER_FAIL, VELOX_ARITHMETIC_ERROR and
  /// VELOX_SCHEMA_MISMATCH_ERROR: a VeloxUserError, which a TRY may swallow.
  kUserError = 1,
  /// VELOX_CHECK*, VELOX_FAIL, VELOX_UNREACHABLE, VELOX_NYI, VELOX_UNSUPPORTED
  /// and the uncatchable forms: a VeloxRuntimeError, which fails the query even
  /// under a TRY.
  kRuntimeError = 2,
};

/// One byte per thread of the block, in the launch's dynamic shared memory. A
/// kernel that runs a Velox body requests at least blockDim.x bytes and zeroes
/// its thread's byte on entry; the adapter does both. Declared outside any
/// __CUDA_ARCH__ guard because nvcc's host pass also parses the device
/// functions that name it.
extern __shared__ uint8_t gpuErrorBytes[];

/// Records that this thread's row failed a check. It does not return: the body
/// runs on with the rejected data, and the row is declined. The first failure
/// wins, so the kind names the earliest rejected precondition. Each byte
/// belongs to one thread, so no atomics are needed.
__host__ __device__ inline void gpuRaise(GpuErrorKind kind) {
#ifdef __CUDA_ARCH__
  if (gpuErrorBytes[threadIdx.x] == static_cast<uint8_t>(GpuErrorKind::kNone)) {
    gpuErrorBytes[threadIdx.x] = static_cast<uint8_t>(kind);
  }
#else
  // The host pass of a translation unit compiled behind the shadow never runs
  // a function body; only registration code is host code there.
  (void)kind;
#endif
}

} // namespace facebook::velox::cudf_velox::gpu_sfi
