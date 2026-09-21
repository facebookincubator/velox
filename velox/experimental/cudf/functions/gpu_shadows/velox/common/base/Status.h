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

// GPU shadow for velox/common/base/Status.h.
//
// This exists to let a header *parse*, not to make Status work on GPU. Several
// function headers define a checked variant next to an unchecked one -- Spark's
// checked_add beside add, for instance -- and a device translation unit has to
// get through the whole header even when only the unchecked function is
// registered. The real Status is built on folly::Expected and carries a
// heap-allocated message, neither of which exists in device code.
//
// So the shadow is deliberately just enough to parse: a one-byte value with an
// ok flag. GpuUDFHolder still refuses to register a Status-returning call(),
// which is what keeps this from being mistaken for support.
//
// TODO(gpu-sfi-checks): a non-OK Status means "this row is a user error". The
// nearest GPU equivalent would be declining the row, which reports a null
// rather than an error -- a different answer, not a suppressed diagnostic. That
// distinction is why Status-returning functions wait for per-row error
// reporting rather than being mapped onto the bool convention.
#pragma once

#include "velox/experimental/cudf/types/GpuProxyTypes.cuh"

namespace facebook::velox {

class Status {
 public:
  GPU_HOST_DEVICE Status() = default;

  GPU_HOST_DEVICE static Status OK() {
    return Status{};
  }

  GPU_HOST_DEVICE bool ok() const {
    return ok_;
  }

 private:
  GPU_HOST_DEVICE explicit Status(bool ok) : ok_(ok) {}

  bool ok_{true};
};

} // namespace facebook::velox

// Reports the failure by returning a non-OK Status on the real path. Here the
// condition is evaluated -- it may have side effects, and discarding it would
// change behaviour -- and the arguments are referenced so they do not read as
// unused, matching what the Exceptions.h shadow does for the check macros.
#define VELOX_USER_RETURN(expr, ...)          \
  do {                                        \
    if (static_cast<bool>(expr)) {            \
      return ::facebook::velox::Status::OK(); \
    }                                         \
  } while (0)
