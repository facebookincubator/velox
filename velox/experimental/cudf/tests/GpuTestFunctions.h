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

namespace facebook::velox::cudf_velox::gpu_sfi {

/// Registers the test-only GPU simple functions, which exercise argument
/// handling no production registration covers yet:
///   test_millis_utc(timestamp with time zone) -> bigint
///   test_initialize_constant(bigint, bigint) -> bigint, which returns the
///     second argument as initialize() saw it, -1 when it was not a constant.
///   test_constant_varchar_length(bigint, constant varchar) -> bigint, which
///     returns the length of the string initialize() saw.
/// Defined in a translation unit compiled with the registrations' shadow
/// include path, which a host test reaches only through this call.
void registerGpuTestFunctions();

} // namespace facebook::velox::cudf_velox::gpu_sfi
