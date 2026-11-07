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

#include <string_view>

namespace facebook::velox::exec::rpc {

/// Query-stable policies: existing WINDOW_AIMD and the independent sending-rate
/// RATE_AIMD and CUBIC policies. Retry ownership is configured separately.
enum class RPCFlowControlPolicy { kWindowAimd = 0, kRateAimd = 1, kCubic = 2 };

/// Parses the ASCII case-insensitive WINDOW_AIMD/RATE_AIMD/CUBIC value set.
RPCFlowControlPolicy parseRPCFlowControlPolicy(std::string_view value);

/// Stable public policy name used by configuration and runtime reporting.
std::string_view rpcFlowControlPolicyName(RPCFlowControlPolicy policy);

/// Shared multiplicative backoff for independent sending-rate policies.
double rpcFlowControlBackoffRate(double previousRate);

/// Sending-rate recovery from the pre-cut peak and eligible admission-wait
/// time. WINDOW_AIMD (kWindowAimd) has no independent rate target and returns
/// zero. RATE_AIMD requires a positive service horizon whose square is finite
/// and nonzero.
double rpcFlowControlRecoveryRate(
    RPCFlowControlPolicy policy,
    double previousPeak,
    double demandSeconds,
    double serviceHorizonSeconds);

} // namespace facebook::velox::exec::rpc
