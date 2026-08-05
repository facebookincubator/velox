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

// Turns a Velox simple function into a CUDA kernel and registers it.
//
// Mirrors velox/functions/Registerer.h: `registerGpuFunction<Func, TReturn,
// TArgs...>` instantiates `Func<GpuExec>` at the call site, wraps it in a
// holder that resolves which of call/callNullable/callNullFree the function
// actually defines, and stores a type-erased launcher. Because instantiation
// happens where the function is named, a dialect registers its own
// implementation simply by naming its own type -- the same reason
// prestosql::DivideFunction and sparksql::DivideFunction can both be `divide`.
//
// Only includable from a translation unit compiled with the gpu_shadows/
// include path.

#include "velox/experimental/cudf/functions/GpuExec.h"
#include "velox/experimental/cudf/functions/GpuFunctionRegistry.h"

#include "velox/core/Metaprogramming.h"
#include "velox/type/TypeKind.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cctype>
#include <string>
#include <type_traits>
#include <utility>

namespace facebook::velox::cudf_velox::gpu_sfi {

namespace detail {

constexpr int kBlockSize = 256;

inline int gridSize(cudf::size_type numRows) {
  return static_cast<int>((numRows + kBlockSize - 1) / kBlockSize);
}

/// Velox lowercases SimpleTypeTrait<T>::name to form a signature string; this
/// is the same transformation, done without <boost/algorithm>.
inline std::string lowercase(const char* name) {
  std::string result(name);
  std::transform(
      result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
  return result;
}

} // namespace detail

/// Resolves which entry point a simple function defines and adapts them to one
/// device-callable signature. The CPU-side analogue is core::UDFHolder, which
/// is already generic over its Exec parameter but lives in a header that pulls
/// the signature and registry machinery, so only its dispatch shape is mirrored
/// here rather than the header reused.
///
/// Returning `false` marks the output row null, which is how a Velox simple
/// function signals "no result" -- the mechanism SparkSQL's divide uses for
/// division by zero.
template <typename Fn, typename TReturn, typename... TArgs>
struct GpuUDFHolder {
  using exec_return_type = typename gpu::GpuExec::resolver<TReturn>::out_type;

  template <typename T>
  using exec_arg_type = typename gpu::GpuExec::resolver<T>::in_type;

  template <typename T>
  using exec_null_free_arg_type =
      typename gpu::GpuExec::resolver<T>::null_free_in_type;

  DECLARE_METHOD_RESOLVER(call_resolver, call);
  DECLARE_METHOD_RESOLVER(call_nullable_resolver, callNullable);
  DECLARE_METHOD_RESOLVER(call_null_free_resolver, callNullFree);

  static constexpr bool hasCallVoid = util::has_method<
      Fn,
      call_resolver,
      void,
      exec_return_type&,
      const exec_arg_type<TArgs>&...>::value;

  static constexpr bool hasCallBool = util::has_method<
      Fn,
      call_resolver,
      bool,
      exec_return_type&,
      const exec_arg_type<TArgs>&...>::value;

  static constexpr bool hasCallNullFreeVoid = util::has_method<
      Fn,
      call_null_free_resolver,
      void,
      exec_return_type&,
      const exec_null_free_arg_type<TArgs>&...>::value;

  static constexpr bool hasCallNullFreeBool = util::has_method<
      Fn,
      call_null_free_resolver,
      bool,
      exec_return_type&,
      const exec_null_free_arg_type<TArgs>&...>::value;

  static constexpr bool hasCallNullableVoid = util::has_method<
      Fn,
      call_nullable_resolver,
      void,
      exec_return_type&,
      const exec_arg_type<TArgs>*...>::value;

  static constexpr bool hasCallNullableBool = util::has_method<
      Fn,
      call_nullable_resolver,
      bool,
      exec_return_type&,
      const exec_arg_type<TArgs>*...>::value;

  static constexpr bool hasCall = hasCallVoid || hasCallBool;
  static constexpr bool hasCallNullFree =
      hasCallNullFreeVoid || hasCallNullFreeBool;
  static constexpr bool hasCallNullable =
      hasCallNullableVoid || hasCallNullableBool;

  static_assert(
      hasCall || hasCallNullFree || hasCallNullable,
      "Function defines none of call(), callNullable() or callNullFree() with "
      "a signature matching the registered types. Note that Status-returning "
      "entry points are not supported on GPU: Status carries a heap-allocated "
      "message, and per-row error reporting from device code is not "
      "implemented yet.");

  /// True when nulls are handled by the framework rather than the function, in
  /// which case the caller can skip null rows entirely.
  static constexpr bool isDefaultNullBehavior = !hasCallNullable;

  /// True when the function cannot decline a row. Its output validity is then
  /// just the AND of the input masks, which cudf computes in one pass, so the
  /// launcher can skip building validity per row.
  static constexpr bool alwaysSucceeds = isDefaultNullBehavior &&
      !hasCallBool && !hasCallNullFreeBool;

  __device__ static bool invoke(
      exec_return_type& out,
      const exec_arg_type<TArgs>&... args) {
    Fn fn;
    if constexpr (hasCallBool) {
      return fn.call(out, args...);
    } else if constexpr (hasCallVoid) {
      fn.call(out, args...);
      return true;
    } else if constexpr (hasCallNullFreeBool) {
      return fn.callNullFree(out, args...);
    } else {
      fn.callNullFree(out, args...);
      return true;
    }
  }

  __device__ static bool invokeNullable(
      exec_return_type& out,
      const exec_arg_type<TArgs>*... args) {
    Fn fn;
    if constexpr (hasCallNullableBool) {
      return fn.callNullable(out, args...);
    } else {
      fn.callNullable(out, args...);
      return true;
    }
  }

  /// The signature Velox would derive from the same template arguments.
  static GpuFunctionSignature signature() {
    return GpuFunctionSignature{
        detail::lowercase(SimpleTypeTrait<TReturn>::name),
        {detail::lowercase(SimpleTypeTrait<TArgs>::name)...}};
  }
};

namespace detail {

/// Row-to-element mapping for one argument. A constant resolves to element 0
/// for every row, which is the same indirection DecodedVector performs on the
/// CPU side and the reason a literal argument needs no special case here.
__device__ inline cudf::size_type
argIndex(const GpuArgView& argument, cudf::size_type row) {
  return argument.isConstant ? 0 : row + argument.offset;
}

__device__ inline bool argIsNull(
    const GpuArgView& argument,
    cudf::size_type row) {
  return argument.nullMask != nullptr &&
      !cudf::bit_is_set(argument.nullMask, argIndex(argument, row));
}

template <typename T>
__device__ inline const T& argValue(
    const GpuArgView& argument,
    cudf::size_type row) {
  return static_cast<const T*>(argument.data)[argIndex(argument, row)];
}

/// Evaluates one row. `valid` is null only when no argument can be null and the
/// function cannot decline a row, in which case there is nothing to record.
template <typename Holder, typename TOut, typename... TIn, std::size_t... I>
__device__ void evaluateRow(
    TOut* out,
    bool* valid,
    const GpuArgView* arguments,
    cudf::size_type row,
    std::index_sequence<I...>) {
  TOut result{};

  if constexpr (Holder::isDefaultNullBehavior) {
    // call() and callNullFree() are never shown a null.
    if ((argIsNull(arguments[I], row) || ...)) {
      if (valid != nullptr) {
        valid[row] = false;
      }
      return;
    }
    bool const ok =
        Holder::invoke(result, argValue<TIn>(arguments[I], row)...);
    if (ok) {
      out[row] = result;
    }
    if (valid != nullptr) {
      valid[row] = ok;
    }
  } else {
    // callNullable() asked to see nulls, which arrive as null pointers.
    bool const ok = Holder::invokeNullable(
        result,
        (argIsNull(arguments[I], row) ? nullptr
                                      : &argValue<TIn>(arguments[I], row))...);
    if (ok) {
      out[row] = result;
    }
    if (valid != nullptr) {
      valid[row] = ok;
    }
  }
}

template <typename Holder, typename TOut, typename... TIn>
__global__ void simpleFunctionKernel(
    TOut* out,
    bool* valid,
    const GpuArgView* arguments,
    cudf::size_type numRows) {
  auto const row = static_cast<cudf::size_type>(
      blockIdx.x * static_cast<unsigned>(blockDim.x) + threadIdx.x);
  if (row >= numRows) {
    return;
  }
  evaluateRow<Holder, TOut, TIn...>(
      out, valid, arguments, row, std::index_sequence_for<TIn...>{});
}

} // namespace detail

/// Evaluates one registered function over whole columns. The instantiation of
/// this is what the registry stores, playing the role
/// SimpleFunctionAdapterFactoryImpl plays on the CPU side.
template <typename Holder, typename TReturn, typename... TArgs>
struct GpuSimpleFunctionAdapter {
  using TOut = typename gpu::GpuExec::resolver<TReturn>::out_type;

  static std::unique_ptr<cudf::column> launch(
      const std::vector<GpuArgView>& arguments,
      cudf::size_type numRows,
      cudf::data_type outputType,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    auto out = cudf::make_fixed_width_column(
        outputType, numRows, cudf::mask_state::UNALLOCATED, stream, mr);
    if (numRows == 0) {
      return out;
    }

    auto deviceArguments = cudf::detail::make_device_uvector_async(
        arguments, stream, cudf::get_current_device_resource_ref());

    // Validity only has to be recorded when something can produce a null: an
    // argument that carries a mask, or a function that can decline a row.
    auto const anyNullable = std::any_of(
        arguments.begin(), arguments.end(), [](const GpuArgView& argument) {
          return argument.nullMask != nullptr;
        });
    auto const needsValidity = anyNullable || !Holder::alwaysSucceeds;

    rmm::device_uvector<bool> valid(
        needsValidity ? numRows : 0, stream, cudf::get_current_device_resource_ref());

    detail::simpleFunctionKernel<
        Holder,
        TOut,
        typename gpu::GpuExec::resolver<TArgs>::in_type...>
        <<<detail::gridSize(numRows),
           detail::kBlockSize,
           0,
           stream.value()>>>(
            out->mutable_view().template data<TOut>(),
            needsValidity ? valid.data() : nullptr,
            deviceArguments.data(),
            numRows);

    if (needsValidity) {
      auto validColumn = cudf::column_view(
          cudf::data_type{cudf::type_id::BOOL8},
          numRows,
          valid.data(),
          nullptr,
          0);
      auto [mask, nullCount] = cudf::bools_to_mask(validColumn, stream, mr);
      out->set_null_mask(std::move(*mask), nullCount);
    }
    return out;
  }
};

/// Registers a Velox simple function to run on GPU.
///
/// Mirrors velox/functions/Registerer.h. `Func<GpuExec>` is instantiated here,
/// at the call site, which is what lets each dialect register its own
/// implementation under a shared name by naming its own type.
template <template <class> typename Func, typename TReturn, typename... TArgs>
bool registerGpuFunction(
    const std::vector<std::string>& aliases,
    bool overwrite = true) {
  using Fn = Func<gpu::GpuExec>;
  using Holder = GpuUDFHolder<Fn, TReturn, TArgs...>;
  using Adapter = GpuSimpleFunctionAdapter<Holder, TReturn, TArgs...>;

  return registerGpuKernel(
      aliases, Holder::signature(), &Adapter::launch, overwrite);
}

} // namespace facebook::velox::cudf_velox::gpu_sfi
