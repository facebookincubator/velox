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

#include <string>

namespace facebook::velox::cudf_velox::gpu_sfi {

/// Registers the PrestoSQL simple functions that were compiled for GPU.
/// Defined in a .cu translation unit; declared here so host code can call it
/// without seeing anything behind the shadow boundary.
void registerPrestoGpuFunctions(const std::string& prefix);

/// Publishes every GPU-compiled simple function into the cuDF function
/// registry, so the existing FunctionExpression evaluator can dispatch to them
/// like any other cuDF function. Call after the register*GpuFunctions() above.
///
/// Ordering matters for names cuDF also implements natively. The cuDF registry
/// returns the first entry whose signature matches, so calling this before
/// registerBuiltinFunctions() makes the Velox-semantics kernel win for names
/// like plus and divide, and calling it after leaves cuDF's vectorized
/// implementation in front.
void publishGpuFunctionsToCudf();

} // namespace facebook::velox::cudf_velox::gpu_sfi
