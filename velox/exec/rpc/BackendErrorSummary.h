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

#include <string>
#include <string_view>
#include <vector>

#include "velox/common/rpc/RPCTypes.h"

namespace facebook::velox::exec::rpc {

/// Upper bound, in bytes, on how much of a backend's error text is kept in one
/// row's error message. The prefix naming the failing path is added on top of
/// it.
constexpr size_t kMaxBackendErrorBytes{256};

/// Reduces a backend error message to the single line worth showing per row:
/// the text before the first newline, capped at 'maxLength' bytes including
/// the elision marker that is appended whenever anything was dropped.
/// Surrounding whitespace is dropped first, so a message that leads with a
/// newline still yields its first real line. The cut is backed off to a UTF-8
/// sequence boundary, so the result can be shorter than 'maxLength'.
///
/// A backend whose thrift method declares no `throws` clause surfaces a
/// rejection as an untyped TApplicationException, and its message is the
/// server's one useful sentence followed by the server-side stack trace: one
/// measured MetaGen rejection ran to 5,665 characters over 51 lines. A
/// whole-batch failure copies that message into every row, so without a cap it
/// lands in the output column and is allocated once per row.
std::string summarizeBackendError(std::string_view rawError, size_t maxLength);

/// Builds one errored response per row for a whole-batch failure, each
/// carrying 'errorPrefix' followed by the summarized backend text. Summarizing
/// once, before the fan-out, is what keeps the dropped trace from being
/// allocated per row. rowId is left at its default because callers stamp batch
/// positions differently.
std::vector<facebook::velox::rpc::RPCResponse> makeBatchErrorResponses(
    size_t numRows,
    std::string_view errorPrefix,
    std::string_view rawError);

} // namespace facebook::velox::exec::rpc
