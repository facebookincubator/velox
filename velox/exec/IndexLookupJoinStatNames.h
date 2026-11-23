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

#include <string_view>

namespace facebook::velox::exec {

/// Names of the runtime stats reported by IndexLookupJoin.
struct IndexLookupJoinStatNames {
  /// The end-to-end walltime in nanoseconds that the index connector do the
  /// lookup.
  static constexpr std::string_view kConnectorLookupWallTime{
      "connectorLookupWallNanos"};

  /// The cpu time in nanoseconds that the index connector process response from
  /// storage client for followup processing by index join operator.
  static constexpr std::string_view kConnectorResultPrepareTime{
      "connectorResultPrepareCpuNanos"};

  /// The cpu time in nanoseconds that the storage client process request for
  /// remote storage lookup such as encoding the lookup input data into remotr
  /// storage request.
  static constexpr std::string_view kClientRequestProcessTime{
      "clientRequestProcessCpuNanos"};

  /// The walltime in nanoseconds that the storage client wait for the lookup
  /// from remote storage.
  static constexpr std::string_view kClientLookupWaitWallTime{
      "clientlookupWaitWallNanos"};

  /// The number of split requests sent to remote storage for a client lookup
  /// request.
  static constexpr std::string_view kClientNumStorageRequests{
      "clientNumStorageRequests"};

  /// The cpu time in nanoseconds that the storage client process response from
  /// remote storage lookup such as decoding the response data into velox
  /// vectors.
  static constexpr std::string_view kClientResultProcessTime{
      "clientResultProcessCpuNanos"};

  /// The byte size of the raw result received from the remote storage lookup.
  static constexpr std::string_view kClientLookupResultRawSize{
      "clientLookupResultRawSize"};

  /// The byte size of the result data in velox vectors that are decoded from
  /// the raw data received from the remote storage lookup.
  static constexpr std::string_view kClientLookupResultSize{
      "clientLookupResultSize"};

  /// The number of lookup results received from remote storage with error.
  static constexpr std::string_view kClientNumErrorResults{
      "clientNumErrorResults"};

  /// The number of index splits provided for index lookup.
  static constexpr std::string_view kNumIndexSplits{"numIndexSplits"};
};

} // namespace facebook::velox::exec
