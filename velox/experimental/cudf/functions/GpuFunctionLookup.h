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

#include "velox/experimental/cudf/functions/GpuFunctionRegistry.h"
#include "velox/expression/FunctionSignature.h"

#include <string>
#include <unordered_map>
#include <vector>

/// Host-side view of the GPU function registry.
///
/// GpuFunctionRegistry.h is the ABI that shadow-compiled .cu code shares, and so
/// may name nothing from Velox. That restriction does not apply here: no device
/// translation unit includes this header, which lets a registered entry hold a
/// real exec::FunctionSignature rather than the strings it crossed the boundary
/// as.
///
/// Holding the real thing is the point. Overload resolution then becomes
/// exec::SignatureBinder -- the matcher SimpleFunctionRegistry itself uses --
/// instead of a private reimplementation, and generic signatures, decimal
/// precision and scale, and the logical types all come along with it.
namespace facebook::velox::cudf_velox::gpu_sfi {

struct GpuFunctionEntry {
  /// Parsed once at registration from the strings the device side supplied.
  exec::FunctionSignaturePtr signature;
  GpuLaunchFn launch;
};

/// Every registration made so far, keyed by lowercased function name.
const std::unordered_map<std::string, std::vector<GpuFunctionEntry>>&
gpuFunctionRegistry();

/// Test support: drops all registrations.
void clearGpuFunctionRegistry();

} // namespace facebook::velox::cudf_velox::gpu_sfi
