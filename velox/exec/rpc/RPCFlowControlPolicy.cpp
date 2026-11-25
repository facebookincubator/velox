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

#include "velox/common/EnumDefine.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec::rpc {

namespace {
const auto& rpcFlowControlPolicyNames() {
  static const folly::F14FastMap<RPCFlowControlPolicy, std::string_view>
      kNames = {
          {RPCFlowControlPolicy::kWindowAimd, "WINDOW_AIMD"},
          {RPCFlowControlPolicy::kCapacityProbing, "CAPACITY_PROBING"},
      };
  return kNames;
}
} // namespace

VELOX_DEFINE_ENUM_NAME(RPCFlowControlPolicy, rpcFlowControlPolicyNames);

RPCFlowControlPolicy parseRPCFlowControlPolicy(std::string_view value) {
  std::string normalized{value};
  for (auto& character : normalized) {
    if (character >= 'a' && character <= 'z') {
      character = static_cast<char>(character - 'a' + 'A');
    }
  }
  if (const auto policy =
          RPCFlowControlPolicyName::tryToRPCFlowControlPolicy(normalized)) {
    return *policy;
  }
  VELOX_USER_FAIL(
      "RPC flow control policy must be WINDOW_AIMD or CAPACITY_PROBING; received '{}'",
      value);
}

} // namespace facebook::velox::exec::rpc
