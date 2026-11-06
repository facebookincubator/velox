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

#include <cmath>
#include <string>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec::rpc {
namespace {
constexpr double kRateBackoffFactor{0.7};
constexpr double kCubicScale{0.4};
} // namespace

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
  if (normalized == "RATE_AIMD") {
    return RPCFlowControlPolicy::kRateAimd;
  }
  if (normalized == "CUBIC") {
    return RPCFlowControlPolicy::kCubic;
  }
  VELOX_USER_FAIL(
      "RPC flow control policy must be WINDOW_AIMD, RATE_AIMD, or CUBIC; received '{}'",
      value);
}

std::string_view rpcFlowControlPolicyName(RPCFlowControlPolicy policy) {
  switch (policy) {
    case RPCFlowControlPolicy::kWindowAimd:
      return "WINDOW_AIMD";
    case RPCFlowControlPolicy::kRateAimd:
      return "RATE_AIMD";
    case RPCFlowControlPolicy::kCubic:
      return "CUBIC";
  }
  VELOX_FAIL("Unknown RPC flow control policy");
}

double rpcFlowControlBackoffRate(double previousRate) {
  return kRateBackoffFactor * previousRate;
}

double rpcFlowControlRecoveryRate(
    RPCFlowControlPolicy policy,
    double previousPeak,
    double demandSeconds,
    double serviceHorizonSeconds) {
  switch (policy) {
    case RPCFlowControlPolicy::kWindowAimd:
      return 0.0;
    case RPCFlowControlPolicy::kRateAimd:
      return kRateBackoffFactor * previousPeak +
          demandSeconds / (serviceHorizonSeconds * serviceHorizonSeconds);
    case RPCFlowControlPolicy::kCubic: {
      const auto timeWindow =
          std::cbrt(previousPeak * (1.0 - kRateBackoffFactor) / kCubicScale);
      return kCubicScale * std::pow(demandSeconds - timeWindow, 3) +
          previousPeak;
    }
  }
  VELOX_FAIL("Unknown RPC flow control policy");
}

} // namespace facebook::velox::exec::rpc
