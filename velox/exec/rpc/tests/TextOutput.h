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

#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/expression/rpc/AsyncRPCFunction.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec::rpc {

/// Builds a VARCHAR output vector from text payloads: errors become SQL NULL,
/// successes carry their text through.
///
/// For the demo and test functions, whose output is a bare VARCHAR with
/// nowhere to put a failure reason. It reads only whether a response failed,
/// so it drops both the message and the typed kind, and it applies no
/// row-failure policy.
///
/// A function that honours `meta_ai_on_error` cannot use this: see
/// FbLlmInferenceAsyncFunction::buildOutput(), which maps each response
/// through classifyRowOutcome() and, depending on the policy, emits the
/// message as the value, carries it in a sibling error column, or fails the
/// query.
inline VectorPtr buildTextOutput(
    const std::vector<RPCResponse>& responses,
    memory::MemoryPool* pool) {
  const auto numRows = static_cast<vector_size_t>(responses.size());
  auto result =
      BaseVector::create<FlatVector<StringView>>(VARCHAR(), numRows, pool);
  for (vector_size_t i = 0; i < numRows; ++i) {
    if (responses[i].hasError()) {
      result->setNull(i, true);
    } else {
      result->set(i, StringView(responseAs<TextPayload>(responses[i]).text));
    }
  }
  return result;
}

} // namespace facebook::velox::exec::rpc
