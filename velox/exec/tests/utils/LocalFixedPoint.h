/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include <folly/Executor.h>

#include "velox/exec/FixedPointOptions.h"

namespace facebook::velox::exec::test {

/// Runs fixed point loops in one process, with producers named in the
/// "local://" scheme that LocalExchangeSource reads.
class LocalFixedPoint {
 public:
  /// Returns the options to pass to queries with FixedPointNodes.
  static const FixedPointOptions& options();

  /// Runs the iteration loops, kept off the query executor the sub-tasks'
  /// drivers run on. One thread serves every loop, nested or peer: a loop
  /// suspends while its sub-tasks run.
  static folly::Executor* orchestrationExecutor();

  /// Formats 'nesting', one "/{planNodeId}.{iteration}" per step.
  static std::string nestingSuffix(
      const std::vector<FixedPointOptions::NestingStep>& nesting);
};

} // namespace facebook::velox::exec::test
