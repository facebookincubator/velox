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

#include <folly/ExceptionWrapper.h>
#include <folly/futures/Future.h> // folly::FutureTimeout

#include <thrift/lib/cpp/TApplicationException.h>
#include <thrift/lib/cpp/transport/TTransportException.h>

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

} // namespace facebook::velox::exec::rpc
