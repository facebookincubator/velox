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
#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <new>
#include <optional>
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

/// The descriptors of a call with a fixed number of arguments, passed to the
/// kernel by value: a parameter block holds 4 KiB and a descriptor 24 bytes, so
/// the launch copies nothing to the device. Each slot is indexed with a
/// constant once inlined, so it is read from parameter space. A call with a
/// variadic tail uploads its descriptors instead, since the tail's view indexes
/// them at run time, which would force the block into local memory.
template <std::size_t N>
struct GpuArgumentPack {
  GpuArgView views[N];

  __device__ const GpuArgView& operator[](std::size_t i) const {
    return views[i];
  }
};

/// True when the argument at slot I is null at this row. A variadic slot is
/// never null; the function reads nullity per element. `TArguments` is a
/// GpuArgumentPack or a device pointer to uploaded descriptors.
template <typename TIn, typename TArguments>
__device__ inline bool
slotIsNull(const TArguments& arguments, std::size_t i, cudf::size_type row) {
  if constexpr (isGpuVariadicView<TIn>::value) {
    return false;
  } else {
    return argIsNull(arguments[i], row);
  }
}

/// The argument to pass for slot I to call() or callNullFree(). A variadic
/// pack is last in a signature, so it is the tail of the descriptor array. A
/// scalar comes back as a reference into the column, a view by value.
template <typename TIn, typename TArguments>
__device__ inline decltype(auto) slotArg(
    const TArguments& arguments,
    int32_t numArgs,
    std::size_t i,
    cudf::size_type row) {
  if constexpr (isGpuVariadicView<TIn>::value) {
    return TIn{arguments + i, numArgs - static_cast<int32_t>(i), row};
  } else {
    return argValue<TIn>(arguments[i], row);
  }
}

/// The argument to pass for slot I to callNullable(): a null scalar is a null
/// pointer, and a variadic pack is passed by value.
template <typename TIn, typename TArguments>
__device__ inline auto slotNullableArg(
    const TArguments& arguments,
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

/// Clears this thread's error byte; shared memory is not zero-initialized.
__device__ inline void clearRaisedError() {
  gpuErrorBytes[threadIdx.x] = static_cast<uint8_t>(GpuErrorKind::kNone);
}

/// What the body recorded for this thread, if anything; see GpuErrorSink.cuh.
__device__ inline GpuErrorKind raisedError() {
  return static_cast<GpuErrorKind>(gpuErrorBytes[threadIdx.x]);
}

/// Evaluates one row and returns whether it has a value. A null input under
/// default null behavior and a body returning false leave the row without one.
/// When `collecting`, so does a failed check: the value came from data the
/// check rejected, and the owner raises the error from the CPU. Without a
/// collector the launch keeps the value, since nobody can act on the failure.
template <
    typename Holder,
    typename TOut,
    typename TArguments,
    typename... TIn,
    std::size_t... I>
__device__ bool evaluateRow(
    typename Holder::udf_struct_t fn,
    TOut* out,
    bool collecting,
    const TArguments& arguments,
    int32_t numArgs,
    cudf::size_type row,
    std::index_sequence<I...>) {
  TOut result{};

  bool ok;
  if constexpr (Holder::isDefaultNullBehavior) {
    // call() and callNullFree() are never shown a null.
    if ((slotIsNull<TIn>(arguments, I, row) || ...)) {
      return false;
    }
    ok =
        Holder::invoke(fn, result, slotArg<TIn>(arguments, numArgs, I, row)...);
  } else {
    // callNullable() asked to see nulls, which arrive as null pointers.
    ok = Holder::invokeNullable(
        fn, result, slotNullableArg<TIn>(arguments, numArgs, I, row)...);
  }

  if (collecting && raisedError() != GpuErrorKind::kNone) {
    return false;
  }
  if (ok) {
    out[row] = result;
  }
  return ok;
}

/// Every lane takes part in the warp votes below: no thread returns before
/// them, and the block size is a multiple of the warp size.
constexpr unsigned kFullWarpMask = 0xffff'ffffu;
constexpr int kWarpSize = 32;
static_assert(
    kBlockSize % kWarpSize == 0,
    "Each warp has to cover one whole validity word");

/// Raises the evaluation's error word to the worst kind any row of this warp
/// hit. Every lane votes, so a clean warp costs one vote and a failing warp one
/// atomic, rather than one per failed row. Both votes are warp-uniform.
__device__ inline void recordDeclines(int32_t* worstKind) {
  auto const raised = raisedError();
  auto const anyRaised =
      __ballot_sync(kFullWarpMask, raised != GpuErrorKind::kNone);
  if (anyRaised == 0) {
    return;
  }
  auto const anyRuntime =
      __ballot_sync(kFullWarpMask, raised == GpuErrorKind::kRuntimeError);
  if (threadIdx.x % kWarpSize == 0) {
    atomicMax(
        worstKind,
        static_cast<int32_t>(
            anyRuntime != 0 ? GpuErrorKind::kRuntimeError
                            : GpuErrorKind::kUserError));
  }
}

/// Writes this warp's validity bits as one mask word and adds the block's null
/// count to the column's. A warp covers exactly one word because the block
/// size is a multiple of the warp size; the bits past the last row stay clear,
/// as cudf::detail::valid_if leaves them.
__device__ inline void recordValidity(
    cudf::bitmask_type* validity,
    cudf::size_type* nullCount,
    bool valid,
    bool hasRow,
    cudf::size_type row) {
  auto const word = __ballot_sync(kFullWarpMask, valid);
  // Lane 0 holds the word's first row, so its row exists iff the word does.
  if (threadIdx.x % kWarpSize == 0 && hasRow) {
    validity[cudf::word_index(row)] = word;
  }
  auto const numNulls = __syncthreads_count(hasRow && !valid);
  if (threadIdx.x == 0 && numNulls > 0) {
    atomicAdd(nullCount, numNulls);
  }
}

/// Evaluates every row. `validity` and `nullCount` are null together, when no
/// row can be null; `worstKind` is null when the caller does not collect. A
/// thread past the last row runs the whole kernel rather than returning: the
/// warp votes and the block count below need every thread of the block.
template <typename Holder, typename TOut, typename TArguments, typename... TIn>
__global__ void simpleFunctionKernel(
    typename Holder::udf_struct_t fn,
    TOut* out,
    cudf::bitmask_type* validity,
    cudf::size_type* nullCount,
    int32_t* worstKind,
    TArguments arguments,
    int32_t numArgs,
    cudf::size_type numRows) {
  // Before the bounds check: every thread of the block owns a byte, with or
  // without a row.
  clearRaisedError();
  auto const row = static_cast<cudf::size_type>(
      blockIdx.x * static_cast<unsigned>(blockDim.x) + threadIdx.x);
  auto const hasRow = row < numRows;
  bool valid = false;
  if (hasRow) {
    valid = evaluateRow<Holder, TOut, TArguments, TIn...>(
        fn,
        out,
        worstKind != nullptr,
        arguments,
        numArgs,
        row,
        std::index_sequence_for<TIn...>{});
  }
  if (worstKind != nullptr) {
    recordDeclines(worstKind);
  }
  if (validity != nullptr) {
    recordValidity(validity, nullCount, valid, hasRow, row);
  }
}

} // namespace detail

/// Evaluates one registered function over whole columns. The registry stores
/// its instantiations.
template <typename Holder, typename TReturn, typename... TArgs>
struct GpuSimpleFunctionAdapter {
  using TOut = typename gpu::GpuExec::resolver<TReturn>::out_type;

  /// The physical type the kernel reads an argument as.
  template <typename T>
  using TIn = typename gpu::GpuExec::resolver<T>::in_type;

  /// Whether the signature ends in a variadic pack, which decides how the
  /// argument descriptors reach the kernel.
  static constexpr bool kHasVariadicTail =
      (isGpuVariadicView<TIn<TArgs>>::value || ...);

  /// Queues the kernel, with the descriptors as a GpuArgumentPack or as a
  /// device pointer.
  template <typename TArguments>
  static void launchKernel(
      const typename Holder::udf_struct_t& fn,
      TOut* out,
      cudf::bitmask_type* validity,
      cudf::size_type* nullCount,
      int32_t* worstKind,
      const TArguments& arguments,
      int32_t numArgs,
      cudf::size_type numRows,
      cuda::stream_ref stream) {
    detail::simpleFunctionKernel<Holder, TOut, TArguments, TIn<TArgs>...>
        // One byte of dynamic shared memory per thread for the error sink,
        // requested whether or not this launch collects, since the check sites
        // cannot tell.
        <<<detail::gridSize(numRows),
           detail::kBlockSize,
           detail::kBlockSize * sizeof(uint8_t),
           stream.get()>>>(
            fn,
            out,
            validity,
            nullCount,
            worstKind,
            arguments,
            numArgs,
            numRows);
  }

  static std::unique_ptr<cudf::column> launch(
      const std::vector<GpuArgView>& arguments,
      const GpuFunctionInstance& instance,
      cudf::size_type numRows,
      cudf::data_type outputType,
      int32_t* worstKind,
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

    // Validity is recorded only when a row can be null for an answer: an
    // argument carries nulls, or the function can return false. A declined row
    // is not a third source, although the kernel leaves it without a value: the
    // owner discards the whole evaluation and re-runs it on the CPU, so the
    // only reader of that row is a parent node of the same discarded
    // evaluation. Recording validity would cost every node a stream
    // synchronization for the null count.
    auto const anyNullable = std::any_of(
        arguments.begin(), arguments.end(), [](const GpuArgView& argument) {
          return argument.nullMask != nullptr;
        });
    auto const needsValidity = anyNullable || !Holder::alwaysSucceeds;

    rmm::device_buffer validity;
    std::optional<rmm::device_scalar<cudf::size_type>> nullCount;
    if (needsValidity) {
      validity = cudf::create_null_mask(
          numRows, cudf::mask_state::UNINITIALIZED, stream, mr);
      nullCount.emplace(stream, cudf::get_current_device_resource_ref());
      nullCount->set_value_to_zero_async(stream);
    }

    auto* const outData = out->mutable_view().template data<TOut>();
    auto* const validityData = needsValidity
        ? static_cast<cudf::bitmask_type*>(validity.data())
        : nullptr;
    auto* const nullCountData = needsValidity ? nullCount->data() : nullptr;
    auto const numArgs = static_cast<int32_t>(arguments.size());
    if constexpr (kHasVariadicTail) {
      // Freed in stream order, so it outlives the kernel.
      auto deviceArguments = cudf::detail::make_device_uvector_async(
          arguments, stream, cudf::get_current_device_resource_ref());
      launchKernel<const GpuArgView*>(
          fn,
          outData,
          validityData,
          nullCountData,
          worstKind,
          deviceArguments.data(),
          numArgs,
          numRows,
          stream);
    } else {
      CUDF_EXPECTS(
          arguments.size() == sizeof...(TArgs),
          "GPU simple function launched with the wrong number of arguments");
      detail::GpuArgumentPack<std::max<std::size_t>(sizeof...(TArgs), 1)>
          pack{};
      std::copy(arguments.begin(), arguments.end(), pack.views);
      launchKernel(
          fn,
          outData,
          validityData,
          nullCountData,
          worstKind,
          pack,
          numArgs,
          numRows,
          stream);
    }

    if (needsValidity) {
      // Reading the count synchronizes the stream, the one synchronization a
      // nullable node pays. A column without nulls keeps no mask, so a parent
      // does not record validity for a mask that would be all ones.
      auto const numNulls = nullCount->value(stream);
      if (numNulls > 0) {
        out->set_null_mask(std::move(validity), numNulls);
      }
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
