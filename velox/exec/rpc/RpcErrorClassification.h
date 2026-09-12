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

#include <new>
#include <stdexcept>
#include <string>

#include <folly/ExceptionWrapper.h>
#include <folly/futures/Future.h> // folly::FutureTimeout

#include <thrift/lib/cpp/TApplicationException.h>
#include <thrift/lib/cpp/transport/TTransportException.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/rpc/RPCTypes.h"

namespace facebook::velox::exec::rpc {

/// Reports whether the failure carried by `ew` is a timeout.
///
/// Recognizes the three ways a timeout surfaces from a ServiceRouter/Thrift2
/// async call:
///   - a client-side processing/transport timeout
///     (`TTransportException::TIMED_OUT`, what `setProcessingTimeoutMs` fires);
///   - a server-reported application timeout
///     (`TApplicationException::TIMEOUT`);
///   - a folly `.within(deadline)` expiry (`folly::FutureTimeout`).
///
/// The transports tag such a failure as `RPCErrorKind::kTimeout` so the
/// congestion policy treats it as a hard-overload signal (immediate backoff),
/// rather than folding it into the generic-error majority fraction. Uses
/// `with_exception` (matches subclasses, never rethrows) so it is safe and
/// cheap in a `.thenTry` handler.
inline bool isTimeout(const folly::exception_wrapper& ew) {
  bool timedOut = false;
  if (ew.with_exception(
          [&](const apache::thrift::transport::TTransportException& e) {
            timedOut = e.getType() ==
                apache::thrift::transport::TTransportException::TIMED_OUT;
          })) {
    return timedOut;
  }
  if (ew.with_exception([&](const apache::thrift::TApplicationException& e) {
        timedOut =
            e.getType() == apache::thrift::TApplicationException::TIMEOUT;
      })) {
    return timedOut;
  }
  return ew.is_compatible_with<folly::FutureTimeout>();
}

/// Reports whether the failure carried by `ew` originated here rather than at
/// the backend: a framework invariant tripping (`VeloxRuntimeError` -- a
/// payload read back as the wrong type, a broken function contract) or the
/// process running out of memory.
///
/// The row still degrades to NULL like any other failure; what this changes is
/// the kind it is tagged with. `RPCErrorKind::kInternalError` keeps such a
/// failure countable on its own instead of inflating the backend error rate,
/// and keeps it out of the congestion window -- a bug in this process should
/// not throttle concurrency against a backend that is answering fine.
inline bool isInternalFailure(const folly::exception_wrapper& ew) {
  return ew.is_compatible_with<VeloxRuntimeError>() ||
      ew.is_compatible_with<std::bad_alloc>();
}

/// An exception that already knows what kind of failure it is.
///
/// A transport often classifies a failure precisely -- rate-limited, invalid
/// request, a polling deadline -- and then has to hand it up the future chain.
/// Propagating the raw exception throws that away: errorKindFor() can only
/// re-derive internal/timeout/backend from the type, so a rate-limit arrives
/// as a plain backend error and the congestion policy loses the immediate
/// backoff it should have triggered. Carrying the kind keeps the metric and
/// the row saying the same thing.
struct RpcClassifiedError : public std::runtime_error {
  RpcClassifiedError(velox::rpc::RPCErrorKind kind, const std::string& message)
      : std::runtime_error(message), kind(kind) {}

  velox::rpc::RPCErrorKind kind;
};

/// The `RPCErrorKind` an exception should be recorded as when it is turned
/// into a failed row.
///
/// Three outcomes, in priority order: a fault of ours (`kInternalError`), a
/// deadline (`kTimeout`, which the congestion policy reads as hard overload),
/// or the backend refusing (`kBackendError`). Shared by every site that
/// degrades an exception so the per-row and batch paths cannot disagree on
/// what a given failure was.
inline velox::rpc::RPCErrorKind errorKindFor(
    const folly::exception_wrapper& ew) {
  // A transport that already classified wins: it saw the backend's own
  // signal, which no amount of inspecting the exception type recovers.
  velox::rpc::RPCErrorKind classified{velox::rpc::RPCErrorKind::kNone};
  if (ew.with_exception(
          [&](const RpcClassifiedError& e) { classified = e.kind; })) {
    return classified;
  }
  if (isInternalFailure(ew)) {
    return velox::rpc::RPCErrorKind::kInternalError;
  }
  if (isTimeout(ew)) {
    return velox::rpc::RPCErrorKind::kTimeout;
  }
  return velox::rpc::RPCErrorKind::kBackendError;
}

} // namespace facebook::velox::exec::rpc
