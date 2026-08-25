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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox::gpu_sfi {

/// One argument as the kernel sees it.
///
/// Modelled on how Velox feeds a simple function: SimpleFunctionAdapter reads
/// every argument through a DecodedVector, which maps a row to an index, so a
/// constant argument resolves to index 0 for every row and the call() body
/// never learns the difference. Carrying the same indirection here means a
/// literal argument costs no materialization and, unlike the column-in
/// column-out interface used elsewhere in this backend, needs no per-function
/// branch to handle.
struct GpuArgView {
  /// Device pointer to the first element.
  const void* data;
  /// Null mask, or nullptr when the argument cannot be null.
  const cudf::bitmask_type* nullMask;
  /// Added to the row index before reading, mirroring column_view::offset().
  cudf::size_type offset;
  /// When true every row reads element 0.
  bool isConstant;
};

/// Evaluates one registered function over a row range. Instantiated behind the
/// shadow boundary, one per (function, argument types) combination, the way
/// SimpleFunctionAdapterFactoryImpl is instantiated per UDFHolder.
using GpuLaunchFn = std::unique_ptr<cudf::column> (*)(
    const std::vector<GpuArgView>& arguments,
    cudf::size_type numRows,
    cudf::data_type outputType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

/// Argument and return types as Velox type names, lowercased, e.g. "double" or
/// "bigint". Derived at registration from SimpleTypeTrait<T>::name, the same
/// source Velox's own TypeAnalysis reads.
struct GpuFunctionSignature {
  std::string returnType;
  std::vector<std::string> argumentTypes;
  /// When true the last entry of argumentTypes is the element type of a
  /// variadic pack rather than a single argument, so a call matches with any
  /// number of trailing arguments of that type -- including none. Velox spells
  /// the same thing as FunctionSignatureBuilder::variableArity().
  bool variadicTail{false};
  /// Integer variables named by the type strings above, such as the i1 and i5
  /// in "decimal(i1,i5)". They have to be declared before the signature can be
  /// built; may contain duplicates.
  std::vector<std::string> integerVariables;
};

/// Registers `launch` under each alias.
///
/// Follows Velox's collision policy: an existing entry with the same name and
/// the same signature is replaced when `overwrite` is true, and left alone
/// otherwise, in which case this returns false. Entries differing in signature
/// coexist as overloads. Dialects therefore separate exactly as they do on the
/// CPU side -- by registered name, by prefix, and by who registers last.
/// The strings are parsed into an exec::FunctionSignature on the host side of
/// the boundary; see GpuFunctionLookup.h for the resulting entry, and for why
/// the registry is read through a separate header.
bool registerGpuKernel(
    const std::vector<std::string>& aliases,
    GpuFunctionSignature signature,
    GpuLaunchFn launch,
    bool overwrite = true);

/// Registers the PrestoSQL simple functions compiled for GPU. Defined in a .cu
/// translation unit; declared here so host code can call it without seeing
/// anything behind the shadow boundary.
void registerPrestoGpuFunctions(const std::string& prefix);

/// The SparkSQL counterpart. A separate translation unit because each dialect
/// instantiates its own Fn<GpuExec>, which is what lets the two disagree.
void registerSparkGpuFunctions(const std::string& prefix);

} // namespace facebook::velox::cudf_velox::gpu_sfi
