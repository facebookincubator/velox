/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/exec/rpc/RPCFlowControlPolicy.h"

#include <string>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec::rpc {

RPCFlowControlPolicy parseRPCFlowControlPolicy(std::string_view value) {
  std::string normalized{value};
  for (auto& character : normalized) {
    if (character >= 'a' && character <= 'z') {
      character = static_cast<char>(character - 'a' + 'A');
    }
  }
  if (normalized == "WINDOW_AIMD") {
    return RPCFlowControlPolicy::kWindowAimd;
  }
  if (normalized == "SMART_PROBING") {
    return RPCFlowControlPolicy::kSmartProbing;
  }
  VELOX_USER_FAIL(
      "RPC flow control policy must be WINDOW_AIMD or SMART_PROBING; received '{}'",
      value);
}

std::string_view rpcFlowControlPolicyName(RPCFlowControlPolicy policy) {
  switch (policy) {
    case RPCFlowControlPolicy::kWindowAimd:
      return "WINDOW_AIMD";
    case RPCFlowControlPolicy::kSmartProbing:
      return "SMART_PROBING";
  }
  VELOX_UNREACHABLE();
}

} // namespace facebook::velox::exec::rpc
