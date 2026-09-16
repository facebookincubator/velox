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

#include "velox/type/SimpleFunctionTags.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

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

/// An initialized function instance, as opaque bytes.
///
/// Velox calls initialize() once per compiled call site and the instance then
/// holds whatever it derived from the argument types -- decimal rescale
/// factors, a parsed date unit. The bytes cross this boundary rather than the
/// typed instance because only the shadow-compiled side can name the type.
/// `data` may be null for a function with no initialize(), which is the
/// common case and the reason the launcher tolerates an empty state.
struct GpuFunctionInstance {
  const void* data;
  int32_t size;
};

/// Evaluates one registered function over a row range. Instantiated behind the
/// shadow boundary, one per (function, argument types) combination, the way
/// SimpleFunctionAdapterFactoryImpl is instantiated per UDFHolder.
using GpuLaunchFn = std::unique_ptr<cudf::column> (*)(
    const std::vector<GpuArgView>& arguments,
    const GpuFunctionInstance& instance,
    cudf::size_type numRows,
    cudf::data_type outputType,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// Runs the function's own initialize() over `instance`, which the caller has
/// sized and aligned per the registration.
///
/// Also compiled behind the shadow boundary, deliberately: running it there
/// means the kernel and initialize() share one instantiation of the function
/// struct, rather than two under two include sets. It works because
/// initialize() only ever binds `*inputTypes[i]` to a `const Type&`, which
/// needs no complete type, and resolves getDecimalPrecisionScale at link time.
using GpuInitializeFn = void (*)(
    void* instance,
    const std::vector<TypePtr>& inputTypes,
    const core::QueryConfig& config);

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
  /// Constraints on those variables, as Velox spells them: a result precision
  /// or scale is not free, it is computed from the argument ones, e.g.
  /// "max(i2,i4)" for the scale of a decimal sum. Each entry names a variable
  /// and gives the expression SignatureBinder evaluates for it. A variable
  /// with no entry here is free, which is right for the argument variables.
  ///
  /// Without these a decimal registration declares its result precision and
  /// scale as unconstrained, and binding either fails or resolves them to
  /// something the kernel was not compiled for.
  std::vector<std::pair<std::string, std::string>> variableConstraints;
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
/// How much storage the function's instance needs, and how to set it up.
///
/// `initialize` is null for a function that has no initialize(); the instance
/// is then default-constructed and `size` is still meaningful, because an empty
/// struct occupies one byte and the launcher checks the size it was handed.
struct GpuFunctionInstanceSpec {
  GpuInitializeFn initialize;
  int32_t size;
  int32_t alignment;
};

bool registerGpuKernel(
    const std::vector<std::string>& aliases,
    GpuFunctionSignature signature,
    GpuLaunchFn launch,
    GpuFunctionInstanceSpec instanceSpec,
    bool overwrite = true);

/// Registers the PrestoSQL simple functions compiled for GPU. Defined in a .cu
/// translation unit; declared here so host code can call it without seeing
/// anything behind the shadow boundary.
void registerPrestoGpuFunctions(const std::string& prefix);

/// The SparkSQL counterpart. A separate translation unit because each dialect
/// instantiates its own Fn<GpuExec>, which is what lets the two disagree.
void registerSparkGpuFunctions(const std::string& prefix);

} // namespace facebook::velox::cudf_velox::gpu_sfi
