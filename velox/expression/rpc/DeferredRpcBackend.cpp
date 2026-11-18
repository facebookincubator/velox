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

#include "velox/expression/rpc/DeferredRpcBackend.h"

namespace facebook::velox::exec::rpc {

std::vector<std::pair<vector_size_t, folly::SemiFuture<RPCResponse>>>
DeferredRpcBackend::dispatchPerRow(
    const SelectivityVector&,
    const std::vector<VectorPtr>&) {
  VELOX_UNSUPPORTED("Deferred RPC backends require BATCH dispatch");
}

} // namespace facebook::velox::exec::rpc
