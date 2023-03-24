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

#include <cuda_runtime.h>
#include <cstdio>
#include <memory>
#include <type_traits>

#define CUDA_CHECK_FATAL(err)                \
  ::facebook::velox::gpu::detail::cudaCheck( \
      (err),                                 \
      __FILE__,                              \
      __LINE__,                              \
      ::facebook::velox::gpu::detail::CudaErrorSeverity::kFatal)

#define CUDA_CHECK_LOG(err)                  \
  ::facebook::velox::gpu::detail::cudaCheck( \
      (err),                                 \
      __FILE__,                              \
      __LINE__,                              \
      ::facebook::velox::gpu::detail::CudaErrorSeverity::kLog)

namespace facebook::velox::gpu {

namespace detail {

enum class CudaErrorSeverity { kLog, kFatal };

inline void cudaCheck(
    cudaError_t err,
    const char* file,
    int line,
    CudaErrorSeverity severity) {
  if (err == cudaSuccess) {
    return;
  }
  fprintf(stderr, "%s:%d %s\n", file, line, cudaGetErrorString(err));
  if (severity >= CudaErrorSeverity::kFatal) {
    abort();
  }
}

template <typename T>
struct CudaFreeDeleter {
  void operator()(T* ptr) const {
    if (!ptr) {
      return;
    }
    std::destroy_at(ptr);
    CUDA_CHECK_LOG(cudaFree(ptr));
  }
};

template <typename T>
struct CudaFreeDeleter<T[]> {
  std::enable_if_t<std::is_trivially_destructible_v<T>, void> operator()(
      T* ptr) const {
    CUDA_CHECK_LOG(cudaFree(ptr));
  }
};

struct CudaEventDestroyDeleter {
  void operator()(cudaEvent_t ptr) const {
    CUDA_CHECK_LOG(cudaEventDestroy(ptr));
  }
};

struct CudaStreamDestroyDeleter {
  void operator()(cudaStream_t ptr) const {
    CUDA_CHECK_LOG(cudaStreamDestroy(ptr));
  }
};

} // namespace detail

/// A unique_ptr taking care of releasing for memory allocated by CUDA.
///
/// When `T[]` is allocated, `~T()` must be trivial because we do not know the
/// number of elements initialized thus cannot call destructor on each of them.
template <typename T>
using CudaPtr = std::unique_ptr<T, detail::CudaFreeDeleter<T>>;

using CudaEvent = std::unique_ptr<CUevent_st, detail::CudaEventDestroyDeleter>;

inline CudaEvent createCudaEvent() {
  cudaEvent_t event;
  CUDA_CHECK_FATAL(cudaEventCreate(&event));
  return CudaEvent(event);
}

using CudaStream =
    std::unique_ptr<CUstream_st, detail::CudaStreamDestroyDeleter>;

inline CudaStream createCudaStream() {
  cudaStream_t stream;
  CUDA_CHECK_FATAL(cudaStreamCreate(&stream));
  return CudaStream(stream);
}

} // namespace facebook::velox::gpu
