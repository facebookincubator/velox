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

template <typename T>
struct isVariadicArg : std::false_type {};

template <typename T>
struct isVariadicArg<Variadic<T>> : std::true_type {};

/// How one declared type appears in a signature.
///
/// The device-side counterpart of Velox's TypeAnalysis, and specialised for the
/// same reason: for most types the signature string is just the type name, but
/// a parameterised type spells out its parameters and has to declare them as
/// variables alongside. Reading SimpleTypeTrait<T>::name unconditionally gets
/// that wrong precisely where it matters -- SimpleTypeTrait<ShortDecimal<P, S>>
/// inherits TypeTraits<BIGINT>, so a decimal function would register as taking
/// a bigint and never match a decimal call.
template <typename T>
struct SignatureType {
  static std::string name() {
    return lowercase(SimpleTypeTrait<T>::name);
  }
  static void collectVariables(std::vector<std::string>&) {}
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
};

/// A pack contributes its element type, not itself: Velox writes a variadic
/// signature as the element type plus a variableArity() flag, and matching then
/// compares the element type against however many arguments arrive.
template <typename T>
struct SignatureType<Variadic<T>> {
  static std::string name() {
    return SignatureType<T>::name();
  }
  static void collectVariables(std::vector<std::string>& variables) {
    SignatureType<T>::collectVariables(variables);
  }
};

/// Every variable named anywhere in the signature, in declaration order.
/// Duplicates are expected -- decimal(i1,i5) as both an argument and the return
/// type names i1 twice -- and are dropped on the host side, where the builder
/// rejects a redeclaration.
template <typename... T>
std::vector<std::string> signatureVariables() {
  std::vector<std::string> variables;
  (SignatureType<T>::collectVariables(variables), ...);
  return variables;
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
  /// Named so the kernel and launcher can spell the instance type without
  /// restating the template arguments, mirroring UDFHolder::udf_struct_t.
  using udf_struct_t = Fn;

  using exec_return_type = typename gpu::GpuExec::resolver<TReturn>::out_type;

  template <typename T>
  using exec_arg_type = typename gpu::GpuExec::resolver<T>::in_type;

  template <typename T>
  using exec_null_free_arg_type =
      typename gpu::GpuExec::resolver<T>::null_free_in_type;

  /// How an argument reaches callNullable(). A scalar arrives as a pointer so
  /// that null can be spelled; a variadic pack arrives as the view itself,
  /// which reports nullity per element and is never absent as a whole.
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

  /// True when the function cannot decline a row. Its output validity is then
  /// just the AND of the input masks, which cudf computes in one pass, so the
  /// launcher can skip building validity per row.
  static constexpr bool alwaysSucceeds =
      isDefaultNullBehavior && !hasCallBool && !hasCallNullFreeBool;

  /// True when the struct declares the template initialize() Velox calls once
  /// per compiled call site. Detected exactly as core::UDFHolder detects it, so
  /// a function whose initialize() Velox would run is not silently skipped
  /// here. The signature only has to parse, not to be instantiable, which is
  /// why SimpleFunctionTags.h declaring TypePtr and QueryConfig is enough.
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

  // TODO(gpu-sfi-initialize): Velox accepts a second initialize() shape taking
  // a memory::MemoryPool* after config, and core::UDFHolder dispatches between
  // the two. The check above only matches the pool-free shape, so a function
  // written the other way registers with hasInitialize false and runs against a
  // default-constructed instance rather than failing. Nothing registered here
  // uses it -- the 15 sites that do are HyperLogLog, KHyperLogLog, SetDigest
  // and SfmSketch, none of them GPU candidates yet -- but it is a silent hole
  // rather than a loud one. Note when closing it that UDFHolder deliberately
  // tries the pool-free overload first: a template initialize() matches any
  // signature by deduction, so handing one a pool binds the pool to its first
  // value argument.

  /// Velox passes the value of each constant argument so initialize() can
  /// specialise on it. Nothing registered here reads them yet, so they are
  /// passed as null -- the same thing Velox passes for a non-constant argument.
  /// TODO(gpu-sfi-initialize): thread constant argument values through the
  /// launch ABI so a function that reads them resolves them the way it would on
  /// the CPU path.
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

  /// The instance is passed by value: it is trivially copyable and small, so a
  /// kernel argument carries it to the device without a separate allocation.
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
        detail::signatureVariables<TReturn, TArgs...>()};
  }
};

namespace detail {

/// True when the argument at slot I is null at this row.
///
/// A variadic slot is never itself null -- the view always exists -- so it
/// answers false and leaves per-element nullity to the function, which is the
/// only party that knows what a null element means for it.
template <typename TIn>
__device__ inline bool
slotIsNull(const GpuArgView* arguments, std::size_t i, cudf::size_type row) {
  if constexpr (isGpuVariadicView<TIn>::value) {
    return false;
  } else {
    return argIsNull(arguments[i], row);
  }
}

/// The argument to pass for slot I to call() or callNullFree(), which take
/// references and are never shown a null.
///
/// Variadic is last in a Velox signature, so the pack is exactly the tail of
/// the descriptor array from this slot onward. That is why the count can be
/// recovered as numArgs - i rather than having to be carried separately.
///
/// decltype(auto) because the two branches return differently and both are
/// right: a scalar yields a reference into the column, while a view is a handle
/// built here and so must come back by value.
template <typename TIn>
__device__ inline decltype(auto) slotArg(
    const GpuArgView* arguments,
    int32_t numArgs,
    std::size_t i,
    cudf::size_type row) {
  if constexpr (isGpuVariadicView<TIn>::value) {
    return TIn{arguments + i, numArgs - static_cast<int32_t>(i), row};
  } else {
    return argValue<TIn>(arguments[i], row);
  }
}

/// The argument to pass for slot I to callNullable(), where a null scalar is a
/// null pointer. A variadic pack has no absent form to express, so it is passed
/// by value exactly as it is to call().
template <typename TIn>
__device__ inline auto slotNullableArg(
    const GpuArgView* arguments,
    int32_t numArgs,
    std::size_t i,
    cudf::size_type row) {
  if constexpr (isGpuVariadicView<TIn>::value) {
    return TIn{arguments + i, numArgs - static_cast<int32_t>(i), row};
  } else {
    return argIsNull(arguments[i], row) ? static_cast<const TIn*>(nullptr)
                                        : &argValue<TIn>(arguments[i], row);
  }
}

/// Evaluates one row. `valid` is null only when no argument can be null and the
/// function cannot decline a row, in which case there is nothing to record.
///
/// Both entry-point conventions go through slotValue(), which yields a pointer
/// for a scalar argument and a view for a variadic one. call() takes references
/// so its pointers are dereferenced on the way in; callNullable() takes
/// pointers and passes them through, a null pointer meaning a null input. A
/// view is passed by value in either case -- there is no "absent pack" for a
/// pointer to express.
template <typename Holder, typename TOut, typename... TIn, std::size_t... I>
__device__ void evaluateRow(
    typename Holder::udf_struct_t fn,
    TOut* out,
    bool* valid,
    const GpuArgView* arguments,
    int32_t numArgs,
    cudf::size_type row,
    std::index_sequence<I...>) {
  TOut result{};

  if constexpr (Holder::isDefaultNullBehavior) {
    // call() and callNullFree() are never shown a null.
    if ((slotIsNull<TIn>(arguments, I, row) || ...)) {
      if (valid != nullptr) {
        valid[row] = false;
      }
      return;
    }
    bool const ok =
        Holder::invoke(fn, result, slotArg<TIn>(arguments, numArgs, I, row)...);
    if (ok) {
      out[row] = result;
    }
    if (valid != nullptr) {
      valid[row] = ok;
    }
  } else {
    // callNullable() asked to see nulls, which arrive as null pointers.
    bool const ok = Holder::invokeNullable(
        fn, result, slotNullableArg<TIn>(arguments, numArgs, I, row)...);
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
    typename Holder::udf_struct_t fn,
    TOut* out,
    bool* valid,
    const GpuArgView* arguments,
    int32_t numArgs,
    cudf::size_type numRows) {
  auto const row = static_cast<cudf::size_type>(
      blockIdx.x * static_cast<unsigned>(blockDim.x) + threadIdx.x);
  if (row >= numRows) {
    return;
  }
  evaluateRow<Holder, TOut, TIn...>(
      fn,
      out,
      valid,
      arguments,
      numArgs,
      row,
      std::index_sequence_for<TIn...>{});
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
      const GpuFunctionInstance& instance,
      cudf::size_type numRows,
      cudf::data_type outputType,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) {
    // The instance is typed again here, in the only translation unit that can
    // name the type. Both the registered size and this sizeof come from the
    // same instantiation, so they cannot disagree; what needs guarding is that
    // the bytes are meaningful at all, which is the static_assert above.
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
    auto const needsValidity = anyNullable || !Holder::alwaysSucceeds;

    rmm::device_uvector<bool> valid(
        needsValidity ? numRows : 0,
        stream,
        cudf::get_current_device_resource_ref());

    detail::simpleFunctionKernel<
        Holder,
        TOut,
        typename gpu::GpuExec::resolver<TArgs>::in_type...>
        <<<detail::gridSize(numRows), detail::kBlockSize, 0, stream.get()>>>(
            fn,
            out->mutable_view().template data<TOut>(),
            needsValidity ? valid.data() : nullptr,
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

/// Registers a Velox simple function to run on GPU.
///
/// Mirrors velox/functions/Registerer.h. `Func<GpuExec>` is instantiated here,
/// at the call site, which is what lets each dialect register its own
/// implementation under a shared name by naming its own type.
///
/// `constraints` mirrors the parameter of the same name on Velox's
/// registerFunction: a decimal result precision and scale are computed from
/// the argument ones, and SignatureBinder needs the expression to do it.
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
