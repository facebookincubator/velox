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

#include <fmt/format.h>
#include <string>

namespace facebook::velox::exec {

/// Define the internal execution state for hash/nested loop join probe. The
/// valid state transition is depicted as follows:
///
///                           +--------------------------------+
///                           ^                                |
///                           |                                V
///   kWaitForBuild -->  kRunning  -->  kWaitForPeers --> kFinish
///         ^                                |
///         |                                v
///         +--------------------------------+
///
enum class ProbeOperatorState {
  /// Wait for hash build operators to build the next hash table to join.
  kWaitForBuild = 0,
  /// The running state that join the probe input with the build table.
  kRunning = 1,
  /// Wait for all the peer probe operators to finish processing inputs. This
  /// state only applies when disk spilling is enabled. The last finished
  /// operator will notify the build operators to build the next hash table from
  /// the spilled data. Then all the peer probe operators will wait for the next
  /// hash table to build.
  kWaitForPeers = 2,
  /// The finishing state.
  kFinish = 3,
};

std::string probeOperatorStateName(ProbeOperatorState state);

} // namespace facebook::velox::exec
