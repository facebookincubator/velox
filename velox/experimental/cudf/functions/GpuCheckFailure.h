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
#include <string>

/// What a failed VELOX_CHECK in a GPU-compiled Velox function body becomes:
/// its kind, which a kernel records per row, and the Velox error the host
/// raises for one. Names only standard types, so it is safe on both sides of
/// the shadow include path.
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

/// Throws the Velox error of the given kind with the message. The host side of
/// a shadow translation unit runs real function bodies in initialize(), and a
/// check they fail has to throw as the real macro would; this is defined on
/// the host, where VeloxException is reachable.
[[noreturn]] void throwCheckFailure(GpuErrorKind kind, const std::string& text);

} // namespace facebook::velox::cudf_velox::gpu_sfi
