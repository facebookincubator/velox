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

#include "velox/exec/rpc/RpcErrorClassification.h"

#include <new>

#include <folly/futures/Future.h>

#include <thrift/lib/cpp/TApplicationException.h>
#include <thrift/lib/cpp/transport/TTransportException.h>

namespace facebook::velox::exec::rpc {

bool isTimeout(const folly::exception_wrapper& exception) {
  bool timedOut{false};
  if (exception.with_exception(
          [&](const apache::thrift::transport::TTransportException& error) {
            timedOut = error.getType() ==
                apache::thrift::transport::TTransportException::TIMED_OUT;
          })) {
    return timedOut;
  }
  if (exception.with_exception(
          [&](const apache::thrift::TApplicationException& error) {
            timedOut = error.getType() ==
                apache::thrift::TApplicationException::TIMEOUT;
          })) {
    return timedOut;
  }
  return exception.is_compatible_with<folly::FutureTimeout>();
}

bool isInternalFailure(const folly::exception_wrapper& exception) {
  return exception.is_compatible_with<VeloxRuntimeError>() ||
      exception.is_compatible_with<std::bad_alloc>();
}

RpcClassifiedError::RpcClassifiedError(
    velox::rpc::RPCErrorKind kind,
    const std::string& message)
    : std::runtime_error(message), kind(kind) {}

velox::rpc::RPCErrorKind errorKindFor(
    const folly::exception_wrapper& exception) {
  velox::rpc::RPCErrorKind classified{velox::rpc::RPCErrorKind::kNone};
  if (exception.with_exception(
          [&](const RpcClassifiedError& error) { classified = error.kind; })) {
    return classified;
  }
  if (isInternalFailure(exception)) {
    return velox::rpc::RPCErrorKind::kInternalError;
  }
  if (isTimeout(exception)) {
    return velox::rpc::RPCErrorKind::kTimeout;
  }
  return velox::rpc::RPCErrorKind::kBackendError;
}

} // namespace facebook::velox::exec::rpc
