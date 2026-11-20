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

#pragma once

#include <fmt/format.h>

#include <string_view>

#include "velox/common/EnumDeclare.h"

namespace facebook::velox::exec::rpc {

/// Admission-window policy of a query's RPC rate limiters. WINDOW_AIMD is the
/// default. Experimental CAPACITY_PROBING is WINDOW_AIMD except on rejections
/// named as a lack of serving capacity: those keep 80% of the calls the backend
/// held, may cut below the floor, and are followed by growth back to that
/// level. Until the first overload, or until responses slow down, it doubles
/// the window every round.
enum class RPCFlowControlPolicy {
  kWindowAimd = 0,
  kCapacityProbing = 1,
};

VELOX_DECLARE_ENUM_NAME(RPCFlowControlPolicy);

/// Parses a policy name, ignoring ASCII case. Throws a user error for any
/// other value.
RPCFlowControlPolicy parseRPCFlowControlPolicy(std::string_view value);

} // namespace facebook::velox::exec::rpc

VELOX_ENUM_FORMATTER(facebook::velox::exec::rpc::RPCFlowControlPolicy);
