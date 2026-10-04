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

// Registry of GPU-compiled Velox simple functions, shared by shadow-compiled
// .cu code and real-Velox host code. It names only cudf, rmm and standard
// library types, which are identical under both include paths: a shadowed
// Velox type crossing here would be read with the wrong layout. A function
// crosses as a launch function pointer plus its signature as strings, from
// which the host rebuilds an exec::FunctionSignature.

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

/// One argument as the kernel sees it. Like a DecodedVector, it maps a row to
/// an element, so a constant argument is a one-element column that every row
/// reads at index 0.
struct GpuArgView {
  /// Device pointer to the first element.
  const void* data;
  /// Null mask, or nullptr when the argument cannot be null.
  const cudf::bitmask_type* nullMask;
  /// Added to the row index before reading, mirroring column_view::offset().
  cudf::size_type offset;
  /// When true every row reads element 0.
  bool isConstant;
  /// Ticks per second of a timestamp argument, which cuDF stores as one
  /// integer in the column's unit; zero otherwise.
  int64_t ticksPerSecond;
};

/// Ticks per second of a cuDF timestamp column, or zero for any other type.
/// Both sides of the shadow boundary convert between cuDF's one integer per
/// row and Velox's seconds and nanoseconds with it.
inline int64_t ticksPerSecond(cudf::data_type type) {
  switch (type.id()) {
    case cudf::type_id::TIMESTAMP_SECONDS:
      return 1;
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return 1'000;
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return 1'000'000;
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return 1'000'000'000;
    default:
      return 0;
  }
}

/// An initialized function instance, as opaque bytes: only the
/// shadow-compiled side can name its type. Holds what initialize() derived from
/// the argument types, such as decimal rescale factors. `data` is null for a
/// function with no initialize().
struct GpuFunctionInstance {
  const void* data;
  int32_t size;
};

/// Evaluates one registered function over a row range. Instantiated behind the
/// shadow boundary, once per function and argument types.
///
/// `declinedRows` is one byte per row, zeroed by the caller, where the launch
/// records a failed check as a GpuErrorKind; null turns collection off. A
/// declined row's value is meaningless and its validity bit is cleared.
using GpuLaunchFn = std::unique_ptr<cudf::column> (*)(
    const std::vector<GpuArgView>& arguments,
    const GpuFunctionInstance& instance,
    cudf::size_type numRows,
    cudf::data_type outputType,
    uint8_t* declinedRows,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

/// A constant argument's value for initialize(), as the bytes of its physical
/// representation in host memory: the native value of a fixed-width type, a
/// timestamp as its seconds then its nanoseconds, each an int64, and the
/// characters of a string. `data` is null for an argument that is not a
/// constant or is a null literal, which initialize() then sees as a null
/// pointer, as Velox passes it.
struct GpuConstantArgument {
  const void* data;
  int32_t size;
};

/// Runs the function's initialize() over `instance`, which the caller has
/// sized and aligned per the registration. Compiled behind the shadow
/// boundary, so the kernel and initialize() share one instantiation of the
/// function struct; initialize() binds `*inputTypes[i]` only to a
/// `const Type&`, which needs no complete type. `constants` holds one entry
/// per argument of the call, and need only cover the arguments the function
/// declares.
using GpuInitializeFn = void (*)(
    void* instance,
    const std::vector<TypePtr>& inputTypes,
    const core::QueryConfig& config,
    const std::vector<GpuConstantArgument>& constants);

/// Argument and return types as lowercase Velox type names, e.g. "double",
/// derived from SimpleTypeTrait<T>::name.
struct GpuFunctionSignature {
  std::string returnType;
  std::vector<std::string> argumentTypes;
  /// When true the last entry of argumentTypes is the element type of a
  /// variadic pack, matching any number of trailing arguments, including none.
  bool variadicTail{false};
  /// One flag per entry of argumentTypes: true where the function declared
  /// Constant<T>, so that only a literal binds and initialize() always
  /// receives its value.
  std::vector<bool> constantArguments;
  /// Integer variables named by the type strings, such as i1 and i5 in
  /// "decimal(i1,i5)". May contain duplicates.
  std::vector<std::string> integerVariables;
  /// Constraints on those variables, as Velox spells them: a variable and the
  /// expression SignatureBinder evaluates for it, such as "max(i2,i4)" for the
  /// scale of a decimal sum. A variable with no entry is free.
  std::vector<std::pair<std::string, std::string>> variableConstraints;

  /// The physical types the kernel was compiled for, which the type names
  /// cannot express: ShortDecimal<P,S> and LongDecimal<P,S> both render as
  /// decimal(i1,i5) while being int64 and int128.
  std::vector<TypeKind> argumentKinds;
  TypeKind returnKind{TypeKind::UNKNOWN};
};

/// Storage for a function's instance and how to initialize it. `initialize`
/// is null when the function has no initialize(); the instance is then
/// default-constructed.
struct GpuFunctionInstanceSpec {
  GpuInitializeFn initialize;
  int32_t size;
  int32_t alignment;
};

/// Registers `launch` under each alias, with Velox's collision policy: an entry
/// with the same name and signature is replaced when `overwrite` is true and
/// otherwise kept, returning false. Entries differing in signature coexist as
/// overloads. GpuFunctionLookup.h reads the result.
///
/// `dependsOnSessionTimeZone` marks a function whose result for a TIMESTAMP
/// argument depends on the session time zone, which initialize() reads; the
/// evaluator keeps such a call away from evaluators that read TIMESTAMP as UTC.
bool registerGpuKernel(
    const std::vector<std::string>& aliases,
    GpuFunctionSignature signature,
    GpuLaunchFn launch,
    GpuFunctionInstanceSpec instanceSpec,
    bool dependsOnSessionTimeZone,
    bool overwrite = true);

/// Registers the PrestoSQL simple functions compiled for GPU. Defined in a .cu
/// translation unit.
void registerPrestoGpuFunctions(const std::string& prefix);

/// Registers the SparkSQL simple functions compiled for GPU. Each dialect
/// instantiates its own Fn<GpuExec> in its own translation unit.
void registerSparkGpuFunctions(const std::string& prefix);

} // namespace facebook::velox::cudf_velox::gpu_sfi
