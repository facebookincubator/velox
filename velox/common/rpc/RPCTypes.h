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

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include <folly/Expected.h>

#include "velox/common/base/Exceptions.h"
#include "velox/vector/TypeAliases.h"

namespace facebook::velox::rpc {

/// Streaming mode for RPC execution.
/// Controls how RPC results are emitted to downstream operators.
enum class RPCStreamingMode {
  /// Emit rows as they complete individually (default).
  /// Lower tail latency for high-variance workloads (e.g., LLM).
  kPerRow,

  /// Wait for all rows in batch before emitting.
  /// Lower overhead, useful for uniform-latency workloads.
  kBatch
};

/// Parse streaming mode from config string.
/// Returns kPerRow (default) unless explicitly set to "batch".
inline RPCStreamingMode parseStreamingMode(const std::string& value) {
  if (value == "batch") {
    return RPCStreamingMode::kBatch;
  }
  return RPCStreamingMode::kPerRow;
}

/// Typed cause of an RPC failure, carried alongside the human-readable error
/// string so consumers can classify failures without parsing message text.
///
/// The transports flatten backend-specific exceptions (rate-limit, timeout)
/// into the opaque 'error' string before a response reaches the framework,
/// which loses the signal a congestion controller needs. This enum preserves
/// it: the transport tags each failed response with why it failed, and a
/// congestion policy can then treat overload (kRateLimited / kTimeout)
/// differently from a user error (kNullInput) or a benign empty result.
enum class RPCErrorKind {
  /// Not an error, or cause not classified.
  kNone,
  /// Null primary input; a user error, not a backend problem.
  kNullInput,
  /// Backend rejected the call for rate limiting / quota (e.g. HTTP 429).
  kRateLimited,
  /// The call exceeded its deadline.
  kTimeout,
  /// Backend returned a non-overload error after retries.
  kBackendError,
  /// Backend returned successfully but with no usable result.
  kEmptyResponse,
  /// Backend rejected the request as invalid (e.g. malformed args, bad model).
  /// Non-retryable: the same request will fail again, so the transport fails
  /// fast rather than spending its retry budget.
  kInvalidRequest,
  /// A producer returned a response without setting an outcome. Not a backend
  /// condition: it yields a NULL row like any other failure, but is counted on
  /// its own and kept out of the congestion window so a producer bug cannot
  /// throttle a healthy backend.
  kUnset,
  /// A framework invariant tripped, or the process ran out of memory, while
  /// handling this row. The row degrades to NULL like any other failure; the
  /// kind is what keeps it out of the backend error rate and out of the
  /// congestion window, since the backend is not the thing that is broken.
  kInternalError,
};

/// Function-owned payload of a response: inline, move-only, exactly typed.
///
/// The framework moves a payload from the transport to the owning function's
/// buildOutput() and never looks inside it, so it needs type erasure. The
/// obvious spelling -- a shared_ptr to a polymorphic base -- costs a heap
/// allocation and a dynamic_cast on every successful row, which measured as
/// ~70% higher per-row CPU in rpc_framework_per_row. The payloads that
/// actually flow are handle-sized (a std::string, a std::vector<float>), so
/// they are stored in place instead and the type is recovered by comparing
/// type_info rather than by walking a class hierarchy.
///
/// Move-only on purpose: a response is handed along a chain of continuations
/// and never duplicated, so nothing needs to share one, and forbidding copies
/// keeps the storage a single owner with exactly one destruction.
class RpcPayload {
 public:
  /// Fits std::string (32 on libstdc++) and std::vector<float> (24). A payload
  /// that does not fit should hold a handle to its data rather than widen
  /// this: the whole point is that a response costs no allocation of its own.
  static constexpr size_t kMaxSize = 32;
  static constexpr size_t kMaxAlign = alignof(std::max_align_t);

  RpcPayload() = default;

  template <typename T, typename Decayed = std::decay_t<T>>
    requires(!std::is_same_v<Decayed, RpcPayload>)
  explicit RpcPayload(T&& value) {
    static_assert(
        sizeof(Decayed) <= kMaxSize,
        "RPC payload must fit inline; hold a handle to larger data instead");
    static_assert(
        alignof(Decayed) <= kMaxAlign, "RPC payload over-aligned for storage");
    static_assert(
        std::is_nothrow_move_constructible_v<Decayed>,
        "RPC payload must move without throwing: it is moved on completion "
        "paths that cannot report a failure");
    ::new (static_cast<void*>(storage_)) Decayed(std::forward<T>(value));
    vtable_ = &vtableFor<Decayed>();
  }

  RpcPayload(RpcPayload&& other) noexcept {
    moveFrom(other);
  }

  RpcPayload& operator=(RpcPayload&& other) noexcept {
    if (this != &other) {
      reset();
      moveFrom(other);
    }
    return *this;
  }

  RpcPayload(const RpcPayload&) = delete;
  RpcPayload& operator=(const RpcPayload&) = delete;

  ~RpcPayload() {
    reset();
  }

  /// True when nothing has been stored, or the value was moved out.
  bool empty() const {
    return vtable_ == nullptr;
  }

  template <typename T>
  bool holds() const {
    return vtable_ != nullptr && *vtable_->type == typeid(T);
  }

  /// The stored value. Throws if the payload holds a different type -- a
  /// transport helper shared by several functions writes one payload type for
  /// all of them, so a function does not always read back the type it wrote,
  /// and an unchecked cast would reinterpret one payload's bytes as another's.
  template <typename T>
  const T& get() const {
    VELOX_CHECK(
        vtable_ != nullptr, "RPC response payload is empty, nothing to read");
    VELOX_CHECK(
        *vtable_->type == typeid(T),
        "RPC response payload is not of the type this function produces: "
        "stored {}, requested {}",
        vtable_->type->name(),
        typeid(T).name());
    return *reinterpret_cast<const T*>(storage_);
  }

 private:
  struct VTable {
    const std::type_info* type;
    void (*destroy)(void*) noexcept;
    void (*moveTo)(void* destination, void* source) noexcept;
  };

  template <typename T>
  static const VTable& vtableFor() {
    static const VTable kVTable{
        &typeid(T),
        [](void* p) noexcept { static_cast<T*>(p)->~T(); },
        [](void* destination, void* source) noexcept {
          ::new (destination) T(std::move(*static_cast<T*>(source)));
          static_cast<T*>(source)->~T();
        }};
    return kVTable;
  }

  void reset() noexcept {
    if (vtable_ != nullptr) {
      vtable_->destroy(static_cast<void*>(storage_));
      vtable_ = nullptr;
    }
  }

  void moveFrom(RpcPayload& other) noexcept {
    if (other.vtable_ != nullptr) {
      other.vtable_->moveTo(
          static_cast<void*>(storage_), static_cast<void*>(other.storage_));
      vtable_ = other.vtable_;
      other.vtable_ = nullptr;
    }
  }

  alignas(kMaxAlign) std::byte storage_[kMaxSize]{};
  const VTable* vtable_{nullptr};
};

/// Why a request failed: the typed cause a congestion policy reads, and the
/// message a failed row carries when the on-error policy asks for one.
struct RpcError {
  RPCErrorKind kind{RPCErrorKind::kNone};
  std::string message;
};

/// Framework-visible part of a response: correlation and outcome. Everything a
/// backend actually returns lives in the function-owned payload.
///
/// A response is a payload or an error, never both and never neither. The
/// outcome is only reachable through setPayload()/setError(), so the state
/// where a producer set neither cannot be built -- a response that nothing has
/// filled in yet already reads as an error.
struct RPCResponse {
  /// Row ID for correlating response with the original request.
  ///
  /// Two meanings depending on context:
  ///   - In flushBatch() return values: the 0-based index of this response
  ///     within the flushed batch (set by the function). The operator uses
  ///     this to scatter responses into the correct positions before
  ///     stamping the global row ID.
  ///   - After operator processing: a globally unique ID assigned by the
  ///     operator for downstream result tracking.
  int64_t rowId{0};

  template <typename T>
  static RPCResponse ok(int64_t rowId, T&& payload) {
    RPCResponse response;
    response.rowId = rowId;
    response.setPayload(std::forward<T>(payload));
    return response;
  }

  static RPCResponse
  failed(int64_t rowId, RPCErrorKind kind, std::string message) {
    RPCResponse response;
    response.rowId = rowId;
    response.setError(kind, std::move(message));
    return response;
  }

  /// Stores the function's own value. A success with no payload is the third
  /// state this type exists to rule out: it reads as succeeded, feeds the
  /// congestion signal, and only fails later when a function tries to read it
  /// -- so there is no overload that sets an empty one.
  template <typename T, typename Decayed = std::decay_t<T>>
  void setPayload(T&& payload) {
    if constexpr (std::is_same_v<Decayed, RpcPayload>) {
      VELOX_CHECK(!payload.empty(), "RPC response payload must not be empty");
      result_ = std::forward<T>(payload);
    } else {
      result_ = RpcPayload{std::forward<T>(payload)};
    }
  }

  void setError(RPCErrorKind kind, std::string errorMessage) {
    // kNone means "not an error"; storing it as one makes hasError() and
    // errorKind() disagree, and the metric switches then drop the row without
    // counting it anywhere. A producer that means "no cause known" says
    // kBackendError.
    VELOX_CHECK(
        kind != RPCErrorKind::kNone,
        "RPC response error kind must name a cause, not kNone");
    result_ = folly::makeUnexpected(RpcError{kind, std::move(errorMessage)});
  }

  /// Returns true if this response represents an error.
  bool hasError() const {
    return result_.hasError();
  }

  /// The failure. Only valid when hasError().
  const RpcError& error() const {
    return result_.error();
  }

  /// Typed cause, or kNone on a successful response.
  RPCErrorKind errorKind() const {
    return result_.hasError() ? result_.error().kind : RPCErrorKind::kNone;
  }

  /// The function-owned payload. Only valid when !hasError().
  const RpcPayload& payload() const {
    return result_.value();
  }

 private:
  // Unfilled reads as an error so that "neither payload nor error" is not a
  // representable state, under a kind no backend can produce. It degrades to a
  // NULL row like any other failure; kUnset is what keeps it countable and out
  // of the congestion window. The message is kept short enough for the
  // small-string buffer -- every response is default-constructed before it is
  // filled, so a longer one would put a heap allocation on the per-row path.
  folly::Expected<RpcPayload, RpcError> result_{
      folly::makeUnexpected(RpcError{RPCErrorKind::kUnset, "unset response"})};
};

} // namespace facebook::velox::rpc
