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

// Turns a Velox simple function into a CUDA kernel and registers it, as
// velox/functions/Registerer.h does on the CPU. Include only from a translation
// unit compiled with the gpu_shadows/ include path.

#include "velox/experimental/cudf/functions/GpuErrorSink.cuh"
#include "velox/experimental/cudf/functions/GpuExec.h"
#include "velox/experimental/cudf/functions/GpuFunctionRegistry.h"
#include "velox/experimental/cudf/functions/GpuVariadicView.h"

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
#include <cstring>
#include <new>
#include <string>
#include <type_traits>
#include <utility>

namespace facebook::velox::cudf_velox::gpu_sfi {

namespace detail {

constexpr int kBlockSize = 256;

inline int gridSize(cudf::size_type numRows) {
  return static_cast<int>((numRows + kBlockSize - 1) / kBlockSize);
}

/// Lowercases SimpleTypeTrait<T>::name into a signature string, as Velox does.
inline std::string lowercase(const char* name) {
  std::string result(name);
  std::transform(
      result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
  return result;
}

template <typename T>
struct isVariadicArg : std::false_type {};

template <typename T>
struct isVariadicArg<Variadic<T>> : std::true_type {};

/// How one declared type appears in a signature, like Velox's TypeAnalysis. A
/// parameterised type such as ShortDecimal<P, S> spells out its parameters and
/// declares them as variables; its SimpleTypeTrait name would say "bigint".
template <typename T>
struct SignatureType {
  static std::string name() {
    return lowercase(SimpleTypeTrait<T>::name);
  }
  static void collectVariables(std::vector<std::string>&) {}
  /// The physical type the kernel is compiled for, which name() cannot
  /// always express.
  static constexpr TypeKind kind() {
    return SimpleTypeTrait<T>::typeKind;
  }
};

template <typename P, typename S>
struct SignatureType<ShortDecimal<P, S>> {
  static std::string name() {
    return "decimal(" + P::name() + "," + S::name() + ")";
  }
  static void collectVariables(std::vector<std::string>& variables) {
    variables.push_back(P::name());
    variables.push_back(S::name());
  }
  static constexpr TypeKind kind() {
    return TypeKind::BIGINT;
  }
};

template <typename P, typename S>
struct SignatureType<LongDecimal<P, S>> {
  static std::string name() {
    return "decimal(" + P::name() + "," + S::name() + ")";
  }
  static void collectVariables(std::vector<std::string>& variables) {
    variables.push_back(P::name());
    variables.push_back(S::name());
  }
  static constexpr TypeKind kind() {
    return TypeKind::HUGEINT;
  }
};

/// A variadic pack contributes its element type; the signature marks it with
/// variableArity.
template <typename T>
struct SignatureType<Variadic<T>> {
  static std::string name() {
    return SignatureType<T>::name();
  }
  static void collectVariables(std::vector<std::string>& variables) {
    SignatureType<T>::collectVariables(variables);
  }
  static constexpr TypeKind kind() {
    return SignatureType<T>::kind();
  }
};

/// Every variable named anywhere in the signature, in declaration order. May
/// contain duplicates; the host drops them.
template <typename... T>
std::vector<std::string> signatureVariables() {
  std::vector<std::string> variables;
  (SignatureType<T>::collectVariables(variables), ...);
  return variables;
}

} // namespace detail

/// Resolves which entry point a simple function defines and adapts it to one
/// device-callable signature, mirroring core::UDFHolder's dispatch. Returning
/// false marks the output row null, as a Velox simple function does.
template <typename Fn, typename TReturn, typename... TArgs>
struct GpuUDFHolder {
  /// The instance type, as UDFHolder::udf_struct_t names it.
  using udf_struct_t = Fn;

  using exec_return_type = typename gpu::GpuExec::resolver<TReturn>::out_type;

  template <typename T>
  using exec_arg_type = typename gpu::GpuExec::resolver<T>::in_type;

  template <typename T>
  using exec_null_free_arg_type =
      typename gpu::GpuExec::resolver<T>::null_free_in_type;

  /// How an argument reaches callNullable(): a scalar as a pointer, null when
  /// the input is null, and a variadic pack as its view.
  template <typename T>
  using exec_nullable_arg_type = std::conditional_t<
      isGpuVariadicView<exec_arg_type<T>>::value,
      exec_arg_type<T>,
      const exec_arg_type<T>*>;

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
      exec_nullable_arg_type<TArgs>...>::value;

  static constexpr bool hasCallNullableBool = util::has_method<
      Fn,
      call_nullable_resolver,
      bool,
      exec_return_type&,
      exec_nullable_arg_type<TArgs>...>::value;

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

  /// True when the function cannot decline a row, so output validity is the
  /// AND of the input masks.
  static constexpr bool alwaysSucceeds =
      isDefaultNullBehavior && !hasCallBool && !hasCallNullFreeBool;

  /// True when the struct declares the template initialize() Velox calls once
  /// per compiled call site, detected as core::UDFHolder detects it.
  template <typename U, typename = void>
  struct hasTemplateInitialize : std::false_type {};

  template <typename U>
  struct hasTemplateInitialize<
      U,
      std::void_t<decltype(std::declval<U>().initialize(
          std::declval<const std::vector<TypePtr>&>(),
          std::declval<const core::QueryConfig&>(),
          static_cast<const exec_arg_type<TArgs>*>(nullptr)...))>>
      : std::true_type {};

  static constexpr bool hasInitialize = hasTemplateInitialize<Fn>::value;

  // TODO(gpu-sfi-initialize): Detect the initialize() overload that takes a
  // memory::MemoryPool* after config, trying the pool-free one first as
  // UDFHolder does. A function using it today runs on a default-constructed
  // instance. None registered here does.

  /// Passes null for every constant argument value, as Velox does for a
  /// non-constant argument; no registered function reads them.
  /// TODO(gpu-sfi-initialize): Pass constant argument values through.
  static void initializeInstance(
      void* storage,
      const std::vector<TypePtr>& inputTypes,
      const core::QueryConfig& config) {
    auto* fn = new (storage) Fn{};
    if constexpr (hasInitialize) {
      fn->initialize(
          inputTypes,
          config,
          static_cast<const exec_arg_type<TArgs>*>(nullptr)...);
    }
  }

  /// The instance is trivially copyable and small, so it reaches the device by
  /// value as a kernel argument.
  static_assert(
      std::is_trivially_copyable_v<Fn>,
      "A GPU simple function's instance is memcpy'd to the device, so any "
      "state initialize() sets has to be trivially copyable. A member holding "
      "a pointer, a std::string or a std::optional of a non-trivial type "
      "cannot cross that boundary.");

  __device__ static bool
  invoke(Fn fn, exec_return_type& out, const exec_arg_type<TArgs>&... args) {
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
      Fn fn,
      exec_return_type& out,
      exec_nullable_arg_type<TArgs>... args) {
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
        detail::SignatureType<TReturn>::name(),
        {detail::SignatureType<TArgs>::name()...},
        (detail::isVariadicArg<TArgs>::value || ...),
        detail::signatureVariables<TReturn, TArgs...>(),
        /*variableConstraints=*/{},
        {detail::SignatureType<TArgs>::kind()...},
        detail::SignatureType<TReturn>::kind()};
  }
};

namespace detail {

/// True when the argument at slot I is null at this row. A variadic slot is
/// never null; the function reads nullity per element.
template <typename TIn>
__device__ inline bool
slotIsNull(const GpuArgView* arguments, std::size_t i, cudf::size_type row) {
  if constexpr (isGpuVariadicView<TIn>::value) {
    return false;
  } else {
    return argIsNull(arguments[i], row);
  }
}

/// The argument to pass for slot I to call() or callNullFree(). A variadic
/// pack is last in a signature, so it is the tail of the descriptor array. A
/// scalar comes back as a reference into the column, a view by value.
template <typename TIn>
__device__ inline decltype(auto) slotArg(
    const GpuArgView* arguments,
    int32_t numArgs,
    std::size_t i,
    cudf::size_type row) {
  if constexpr (isGpuVariadicView<TIn>::value) {
    return TIn{arguments + i, numArgs - static_cast<int32_t>(i), row};
  } else if constexpr (gpu::isGpuCustomTypeView<TIn>::value) {
    // Wrapped on the way out of the column, so the view need not share the
    // element's layout.
    return TIn{argValue<typename TIn::physical_type>(arguments[i], row)};
  } else {
    return argValue<TIn>(arguments[i], row);
  }
}

/// The argument to pass for slot I to callNullable(): a null scalar is a null
/// pointer, and a variadic pack is passed by value.
template <typename TIn>
__device__ inline auto slotNullableArg(
    const GpuArgView* arguments,
    int32_t numArgs,
    std::size_t i,
    cudf::size_type row) {
  // A null pointer stands for a null input here, and a wrapped custom-type
  // value has no storage in the column to point at.
  static_assert(
      !gpu::isGpuCustomTypeView<TIn>::value,
      "Custom type arguments to callNullable() are not supported yet");
  if constexpr (isGpuVariadicView<TIn>::value) {
    return TIn{arguments + i, numArgs - static_cast<int32_t>(i), row};
  } else {
    return argIsNull(arguments[i], row) ? static_cast<const TIn*>(nullptr)
                                        : &argValue<TIn>(arguments[i], row);
  }
}

/// Clears this thread's error byte; shared memory is not zero-initialized.
__device__ inline void clearRaisedError() {
  gpuErrorBytes[threadIdx.x] = static_cast<uint8_t>(GpuErrorKind::kNone);
}

/// What the body recorded for this thread, if anything; see GpuErrorSink.cuh.
__device__ inline GpuErrorKind raisedError() {
  return static_cast<GpuErrorKind>(gpuErrorBytes[threadIdx.x]);
}

/// Evaluates one row. `valid` is null only when no argument can be null and the
/// function cannot decline a row.
template <typename Holder, typename TOut, typename... TIn, std::size_t... I>
__device__ void evaluateRow(
    typename Holder::udf_struct_t fn,
    TOut* out,
    bool* valid,
    uint8_t* declinedRows,
    const GpuArgView* arguments,
    int32_t numArgs,
    cudf::size_type row,
    std::index_sequence<I...>) {
  TOut result{};

  bool ok;
  if constexpr (Holder::isDefaultNullBehavior) {
    // call() and callNullFree() are never shown a null.
    if ((slotIsNull<TIn>(arguments, I, row) || ...)) {
      if (valid != nullptr) {
        valid[row] = false;
      }
      return;
    }
    ok =
        Holder::invoke(fn, result, slotArg<TIn>(arguments, numArgs, I, row)...);
  } else {
    // callNullable() asked to see nulls, which arrive as null pointers.
    ok = Holder::invokeNullable(
        fn, result, slotNullableArg<TIn>(arguments, numArgs, I, row)...);
  }

  // A declined row's value came from data a check rejected, so it is not
  // written. Returning false means the function has no value for the row;
  // declining means the host still has to raise an error. Rows are declined
  // only when the caller collects; otherwise the launch keeps its result.
  if (declinedRows != nullptr) {
    auto const raised = raisedError();
    if (raised != GpuErrorKind::kNone) {
      declinedRows[row] = static_cast<uint8_t>(raised);
      ok = false;
    }
  }
  if (ok) {
    out[row] = result;
  }
  if (valid != nullptr) {
    valid[row] = ok;
  }
}

template <typename Holder, typename TOut, typename... TIn>
__global__ void simpleFunctionKernel(
    typename Holder::udf_struct_t fn,
    TOut* out,
    bool* valid,
    uint8_t* declinedRows,
    const GpuArgView* arguments,
    int32_t numArgs,
    cudf::size_type numRows) {
  // Before the bounds check: every thread of the block owns a byte, with or
  // without a row.
  clearRaisedError();
  auto const row = static_cast<cudf::size_type>(
      blockIdx.x * static_cast<unsigned>(blockDim.x) + threadIdx.x);
  if (row >= numRows) {
    return;
  }
  evaluateRow<Holder, TOut, TIn...>(
      fn,
      out,
      valid,
      declinedRows,
      arguments,
      numArgs,
      row,
      std::index_sequence_for<TIn...>{});
}

} // namespace detail

/// Evaluates one registered function over whole columns. The registry stores
/// its instantiations.
template <typename Holder, typename TReturn, typename... TArgs>
struct GpuSimpleFunctionAdapter {
  using TOut = typename gpu::GpuExec::resolver<TReturn>::out_type;

  static std::unique_ptr<cudf::column> launch(
      const std::vector<GpuArgView>& arguments,
      const GpuFunctionInstance& instance,
      cudf::size_type numRows,
      cudf::data_type outputType,
      uint8_t* declinedRows,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) {
    // Retypes the instance in the only translation unit that can name its
    // type. The registered size comes from the same instantiation.
    using Fn = typename Holder::udf_struct_t;
    Fn fn{};
    if (instance.data != nullptr && instance.size == sizeof(Fn)) {
      std::memcpy(&fn, instance.data, sizeof(Fn));
    }

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
    // A declined row is nulled, so collecting errors makes validity necessary
    // even for a function that can otherwise never produce one.
    auto const needsValidity =
        anyNullable || !Holder::alwaysSucceeds || declinedRows != nullptr;

    rmm::device_uvector<bool> valid(
        needsValidity ? numRows : 0,
        stream,
        cudf::get_current_device_resource_ref());

    detail::simpleFunctionKernel<
        Holder,
        TOut,
        typename gpu::GpuExec::resolver<TArgs>::in_type...>
        // One byte of dynamic shared memory per thread for the error sink,
        // requested whether or not this launch collects, since the check sites
        // cannot tell.
        <<<detail::gridSize(numRows),
           detail::kBlockSize,
           detail::kBlockSize * sizeof(uint8_t),
           stream.get()>>>(
            fn,
            out->mutable_view().template data<TOut>(),
            needsValidity ? valid.data() : nullptr,
            declinedRows,
            deviceArguments.data(),
            static_cast<int32_t>(arguments.size()),
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

/// Registers a Velox simple function to run on GPU. Func<GpuExec> is
/// instantiated here, so each dialect registers its own implementation under a
/// shared name by naming its own type.
///
/// `constraints` gives Velox's constraint on a signature variable, such as a
/// decimal result scale computed from the argument scales.
template <template <class> typename Func, typename TReturn, typename... TArgs>
bool registerGpuFunction(
    const std::vector<std::string>& aliases,
    std::vector<std::pair<std::string, std::string>> constraints = {},
    bool overwrite = true) {
  using Fn = Func<gpu::GpuExec>;
  using Holder = GpuUDFHolder<Fn, TReturn, TArgs...>;
  using Adapter = GpuSimpleFunctionAdapter<Holder, TReturn, TArgs...>;

  auto signature = Holder::signature();
  signature.variableConstraints = std::move(constraints);

  return registerGpuKernel(
      aliases,
      std::move(signature),
      &Adapter::launch,
      GpuFunctionInstanceSpec{
          Holder::hasInitialize ? &Holder::initializeInstance : nullptr,
          static_cast<int32_t>(sizeof(Fn)),
          static_cast<int32_t>(alignof(Fn))},
      overwrite);
}

} // namespace facebook::velox::cudf_velox::gpu_sfi
