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

// GPU shadow for velox/common/base/Status.h. Lets a header that defines a
// Status-returning function, such as Spark's checked_add beside add, parse in a
// device translation unit. A failed Status carries only its error class; the
// macros below also raise it into the error sink (GpuErrorSink.cuh), and the
// host re-evaluates the row for Velox's message. GpuUDFHolder still refuses to
// register a Status-returning call().
#pragma once

#include "velox/experimental/cudf/functions/GpuErrorSink.cuh"
#include "velox/experimental/cudf/types/GpuProxyTypes.cuh"

#include "velox/common/base/Exceptions.h"

namespace facebook::velox {

class Status {
 public:
  GPU_HOST_DEVICE Status() = default;

  GPU_HOST_DEVICE static Status OK() {
    return Status{};
  }

  /// A failed Status carries its error class, which decides whether a TRY may
  /// swallow the row; the message comes from re-evaluating the row.
  GPU_HOST_DEVICE static Status UserError() {
    return Status{
        ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kUserError};
  }

  GPU_HOST_DEVICE static Status RuntimeError() {
    return Status{
        ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kRuntimeError};
  }

  GPU_HOST_DEVICE ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind
  errorKind() const {
    return kind_;
  }

  GPU_HOST_DEVICE bool ok() const {
    return ok_;
  }

 private:
  GPU_HOST_DEVICE explicit Status(
      ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind kind)
      : ok_(false), kind_(kind) {}

  bool ok_{true};
  ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind kind_{
      ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kNone};
};

} // namespace facebook::velox

// Returns a user error when `expr` holds, as the real macro does, and raises it
// so the host learns which row failed. The message arguments are discarded.
#define VELOX_USER_RETURN(expr, ...)                                         \
  do {                                                                       \
    if (static_cast<bool>(expr)) {                                           \
      ::facebook::velox::gpu_shadow_detail::useArgs(__VA_ARGS__);            \
      ::facebook::velox::cudf_velox::gpu_sfi::gpuRaise(                      \
          ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kUserError); \
      return ::facebook::velox::Status::UserError();                         \
    }                                                                        \
  } while (0)

// The comparison forms, which Spark's decimal and arithmetic headers use.
#define VELOX_GPU_SHADOW_USER_RETURN_OP(e1, e2, op, ...) \
  VELOX_USER_RETURN(!((e1)op(e2))__VA_OPT__(, ) __VA_ARGS__)

#define VELOX_USER_RETURN_EQ(e1, e2, ...) \
  VELOX_GPU_SHADOW_USER_RETURN_OP(e1, e2, == __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_USER_RETURN_NE(e1, e2, ...) \
  VELOX_GPU_SHADOW_USER_RETURN_OP(e1, e2, != __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_USER_RETURN_LT(e1, e2, ...) \
  VELOX_GPU_SHADOW_USER_RETURN_OP(e1, e2, < __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_USER_RETURN_LE(e1, e2, ...) \
  VELOX_GPU_SHADOW_USER_RETURN_OP(e1, e2, <= __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_USER_RETURN_GT(e1, e2, ...) \
  VELOX_GPU_SHADOW_USER_RETURN_OP(e1, e2, > __VA_OPT__(, ) __VA_ARGS__)
#define VELOX_USER_RETURN_GE(e1, e2, ...) \
  VELOX_GPU_SHADOW_USER_RETURN_OP(e1, e2, >= __VA_OPT__(, ) __VA_ARGS__)

// Returns `status` when `condition` holds, raising its error class.
#ifndef VELOX_RETURN_IF
#define VELOX_RETURN_IF(condition, status)                                    \
  do {                                                                        \
    if (static_cast<bool>(condition)) {                                       \
      ::facebook::velox::cudf_velox::gpu_sfi::gpuRaise((status).errorKind()); \
      return (status);                                                        \
    }                                                                         \
  } while (0)
#endif

// Status form only: there is no Result<T> on the device.
#ifndef VELOX_RETURN_NOT_OK
#define VELOX_RETURN_NOT_OK(status)                  \
  do {                                               \
    ::facebook::velox::Status _gpuStatus = (status); \
    VELOX_RETURN_IF(!_gpuStatus.ok(), _gpuStatus);   \
  } while (0)
#endif

// folly::Expected has no device form, so these raise and fall through,
// declining the row. They exist so that a header using them parses.
#ifndef VELOX_RETURN_UNEXPECTED_IF
#define VELOX_RETURN_UNEXPECTED_IF(condition, status)                        \
  do {                                                                       \
    if (static_cast<bool>(condition)) {                                      \
      ::facebook::velox::cudf_velox::gpu_sfi::gpuRaise(                      \
          ::facebook::velox::cudf_velox::gpu_sfi::GpuErrorKind::kUserError); \
    }                                                                        \
  } while (0)
#endif

#ifndef VELOX_RETURN_UNEXPECTED_NOT_OK
#define VELOX_RETURN_UNEXPECTED_NOT_OK(status)          \
  do {                                                  \
    ::facebook::velox::Status _gpuStatus = (status);    \
    if (!_gpuStatus.ok()) {                             \
      ::facebook::velox::cudf_velox::gpu_sfi::gpuRaise( \
          _gpuStatus.errorKind());                      \
    }                                                   \
  } while (0)
#endif

#ifndef VELOX_RETURN_UNEXPECTED
#define VELOX_RETURN_UNEXPECTED(expected)                                      \
  do {                                                                         \
    auto _gpuExpected = (expected);                                            \
    VELOX_RETURN_UNEXPECTED_IF(_gpuExpected.hasError(), _gpuExpected.error()); \
  } while (0)
#endif

#ifndef VELOX_USER_RETURN_NULL
#define VELOX_USER_RETURN_NULL(e, ...) \
  VELOX_USER_RETURN((e) == nullptr __VA_OPT__(, ) __VA_ARGS__)
#endif
#ifndef VELOX_USER_RETURN_NOT_NULL
#define VELOX_USER_RETURN_NOT_NULL(e, ...) \
  VELOX_USER_RETURN((e) != nullptr __VA_OPT__(, ) __VA_ARGS__)
#endif
