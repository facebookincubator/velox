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

// Registry of GPU-compiled Velox simple functions.
//
// This is the boundary between shadow-compiled device code and real-Velox host
// code, and it is deliberately narrow. Registration happens in .cu translation
// units where the gpu_shadows/ include path is active and `StringView` resolves
// to a GPU substitute with a different layout; letting such a type reach a
// real-Velox translation unit would be an ODR violation with no diagnostic. So
// this header names only cudf, rmm, and standard library types, which are
// identical under both include paths, and carries no Velox include.
//
// A registered function crosses as a plain function pointer plus its signature
// as strings. Strings are how Velox describes signatures anyway --
// FunctionSignatureBuilder().returnType("double") -- so the host bridge can
// rebuild a real FunctionSignature without the device side ever naming one.

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace facebook::velox::cudf_velox::gpu_sfi {

/// Evaluates one registered function over whole columns. Instantiated behind
/// the shadow boundary, one per (function, argument types) combination, the way
/// SimpleFunctionAdapterFactoryImpl is instantiated per UDFHolder.
using GpuLaunchFn = std::unique_ptr<cudf::column> (*)(
    const std::vector<cudf::column_view>& inputs,
    cudf::data_type outputType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Argument and return types as Velox type names, lowercased, e.g. "double" or
/// "bigint". Derived at registration from SimpleTypeTrait<T>::name, the same
/// source Velox's own TypeAnalysis reads.
struct GpuFunctionSignature {
  std::string returnType;
  std::vector<std::string> argumentTypes;
};

struct GpuFunctionEntry {
  GpuFunctionSignature signature;
  GpuLaunchFn launch;
};

/// Registers `launch` under each alias.
///
/// Follows Velox's collision policy: an existing entry with the same name and
/// the same signature is replaced when `overwrite` is true, and left alone
/// otherwise, in which case this returns false. Entries differing in signature
/// coexist as overloads. Dialects therefore separate exactly as they do on the
/// CPU side -- by registered name, by prefix, and by who registers last.
bool registerGpuKernel(
    const std::vector<std::string>& aliases,
    GpuFunctionSignature signature,
    GpuLaunchFn launch,
    bool overwrite = true);

/// Every registration made so far, keyed by lowercased function name. The host
/// bridge walks this to publish each entry into the cuDF function registry.
const std::unordered_map<std::string, std::vector<GpuFunctionEntry>>&
gpuFunctionRegistry();

/// Test support: drops all registrations.
void clearGpuFunctionRegistry();

} // namespace facebook::velox::cudf_velox::gpu_sfi
