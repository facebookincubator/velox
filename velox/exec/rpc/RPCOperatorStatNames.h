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

namespace facebook::velox::exec::rpc {

/// Names of the runtime stats reported by RPCOperator.
struct RPCOperatorStatNames {
  static inline const std::string kRpcRequestsDispatched{
      "rpcRequestsDispatched"};

  static inline const std::string kRpcResponsesReceived{"rpcResponsesReceived"};

  static inline const std::string kRpcErrorCount{"rpcErrorCount"};

  static inline const std::string kRpcWaitWallNanos{"rpcWaitWallNanos"};

  static inline const std::string kRpcBackpressureWaitNanos{
      "rpcBackpressureWaitNanos"};

  static inline const std::string kRpcCongestionWindowFinal{
      "rpcCongestionWindowFinal"};

  static inline const std::string kRpcCongestionShrinks{"rpcCongestionShrinks"};

  static inline const std::string kRpcCongestionOverloadShrinks{
      "rpcCongestionOverloadShrinks"};

  /// Attempts rejected for capacity and admitted again for another attempt.
  static inline const std::string kRpcCapacityRetries{"rpcCapacityRetries"};

  /// Hinted rejections of a rate quota, such as requests or tokens per minute,
  /// whether or not they retried.
  static inline const std::string kRpcQuotaRejections{"rpcQuotaRejections"};

  /// Hinted rejections for lack of serving capacity, whether or not they
  /// retried.
  static inline const std::string kRpcCapacityRejections{
      "rpcCapacityRejections"};

  /// Capacity retries a hint's minRetries granted after its budget ran out.
  static inline const std::string kRpcCapacityRetryFloorRetries{
      "rpcCapacityRetryFloorRetries"};

  static inline const std::string kRpcBaselineRttNanos{"rpcBaselineRttNanos"};

  static inline const std::string kRpcPeakInFlight{"rpcPeakInFlight"};

  static inline const std::string kRpcRttMinWallNanos{"rpcRttMinWallNanos"};

  static inline const std::string kRpcRttMaxWallNanos{"rpcRttMaxWallNanos"};

  static inline const std::string kRpcRttCount{"rpcRttCount"};

  static inline const std::string kRpcStreamingMode{"rpcStreamingMode"};

  /// The requested flow-control policy, by RPCFlowControlPolicy value:
  /// WINDOW_AIMD=0, CAPACITY_PROBING=1. Reported per driver.
  static inline const std::string kRpcFlowControlRequestedPolicy{
      "rpcFlowControlRequestedPolicy"};

  /// The policy the limiter runs, by the same values. A CAPACITY_PROBING
  /// request that is not eligible reports WINDOW_AIMD here.
  static inline const std::string kRpcFlowControlEffectivePolicy{
      "rpcFlowControlEffectivePolicy"};

  static inline const std::string kRpcErrorKindRateLimited{
      "rpcErrorKindRateLimited"};

  static inline const std::string kRpcErrorKindTimeout{"rpcErrorKindTimeout"};

  static inline const std::string kRpcErrorKindBackendError{
      "rpcErrorKindBackendError"};

  static inline const std::string kRpcErrorKindInvalidRequest{
      "rpcErrorKindInvalidRequest"};

  static inline const std::string kRpcErrorKindInternal{"rpcErrorKindInternal"};

  /// Overloaded attempts the transport reported before each completed PER_ROW
  /// row's final attempt (RPCResponse::numOverloadedAttempts), failed rows
  /// included. BATCH completions are not counted.
  /// This and kRpcSuccessesAfterOverload are per-driver live counters that sum
  /// across drivers.
  static inline const std::string kRpcOverloadedAttempts{
      "rpcOverloadedAttempts"};

  /// Successes whose response reported at least one overloaded attempt.
  static inline const std::string kRpcSuccessesAfterOverload{
      "rpcSuccessesAfterOverload"};

  // Shared rate-limiter stats, refreshed on every stats() call.
  static inline const std::string kRpcRateLimiterCap{"rpcRateLimiterCap"};

  static inline const std::string kRpcRateLimiterHardLimit{
      "rpcRateLimiterHardLimit"};

  static inline const std::string kRpcRateLimiterPeakPending{
      "rpcRateLimiterPeakPending"};

  static inline const std::string kRpcRateLimiterMinCap{"rpcRateLimiterMinCap"};

  static inline const std::string kRpcRateLimiterLimitMilli{
      "rpcRateLimiterLimitMilli"};

  static inline const std::string kRpcRateLimiterMinLimitMilli{
      "rpcRateLimiterMinLimitMilli"};

  static inline const std::string kRpcRateLimiterServiceHorizonNanos{
      "rpcRateLimiterServiceHorizonNanos"};

  /// The floor the limiter enforces, in thousandths of an admission unit,
  /// present only when the limiter adapts and its floor is below its ceiling.
  static inline const std::string kRpcRateLimiterFloorMilli{
      "rpcRateLimiterFloorMilli"};

  /// The ceiling the function declares for its backend, present only when it
  /// declares one. Beside kRpcRateLimiterHardLimit, it shows whether a
  /// session limit changed the function's ceiling.
  static inline const std::string kRpcFunctionCeiling{"rpcFunctionCeiling"};

  /// The adaptive floor the function declares, in thousandths of an admission
  /// unit, present only when it declares one. Compare with
  /// kRpcRateLimiterFloorMilli.
  static inline const std::string kRpcFunctionFloorMilli{
      "rpcFunctionFloorMilli"};

  /// Present, with value one per driver, when the backend's shared limiter
  /// runs another policy and this operator runs under the limiter for its own.
  static inline const std::string kRpcRateLimiterPolicyScoped{
      "rpcRateLimiterPolicyScoped"};

  /// Time from this driver's first admission request until the stats sample
  /// or close. The pacing and window stats below cover the same span, and a
  /// driver that never requests admission reports none of the three. These
  /// three and rpcStaleSuccessUnits are per-driver counters that sum across
  /// drivers and tasks on Prestissimo and on Presto-on-Spark; the sums are
  /// driver time, not wall time. sum(PacingNanos) / sum(AttachedNanos) is the
  /// share of attached time during which the shared limiter paced. It bounds
  /// pacing delay from above, because it includes time a driver was idle or
  /// waiting on upstream. sum(LimitMilliUnitSeconds) / (sum(AttachedNanos) /
  /// 1e9) is the average window in milli-admission-units.
  static inline const std::string kRpcRateLimiterAttachedNanos{
      "rpcRateLimiterAttachedNanos"};

  /// Part of that span during which the shared limiter paced admission.
  static inline const std::string kRpcRateLimiterPacingNanos{
      "rpcRateLimiterPacingNanos"};

  /// The shared limiter's committed window, the one rpcRateLimiterLimitMilli
  /// shows, integrated over that span, in milli-admission-units times seconds.
  /// Recovery growth counts once the limiter commits it to the window.
  static inline const std::string kRpcRateLimiterLimitMilliUnitSeconds{
      "rpcRateLimiterLimitMilliUnitSeconds"};

  /// This driver's successful units whose recovery credit the shared limiter
  /// discarded because an overload, including one in the same completion,
  /// arrived after their admission. PER_ROW counts error-free rows; BATCH
  /// counts the admission units of a flush with at least one successful row
  /// as a whole, as the limiter credits them.
  static inline const std::string kRpcStaleSuccessUnits{"rpcStaleSuccessUnits"};

  /// Live liveness stats, refreshed on every stats() call rather than only at
  /// close(). Units dispatched but not yet completed.
  static inline const std::string kRpcInFlight{"rpcInFlight"};

  /// Monotonic count of completions signaled by the transport. Advances from
  /// the RPC executor threads, so it moves even while the driver is blocked.
  static inline const std::string kRpcCompletionsSignaled{
      "rpcCompletionsSignaled"};

  /// Monotonic count of retry attempts the function's transports have
  /// scheduled. A retry is neither a dispatch nor a completion, so a request
  /// climbing a doubling backoff schedule leaves both counters above frozen
  /// for as long as the ladder runs. Advances from the transport executor
  /// threads, so it moves even while the driver is blocked.
  static inline const std::string kRpcRetriesAttempted{"rpcRetriesAttempted"};
};

} // namespace facebook::velox::exec::rpc
