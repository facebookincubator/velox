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

#include <gflags/gflags.h>
#include <string>

DECLARE_bool(fast);

namespace facebook::velox::tool::trace {

/// A utility class that prompts users to enter critical configs for trace
/// replay. It displays the current gflag setting and allows users to override
/// or confirm the value interactively.
///
/// When --fast flag is set, prompts are skipped and existing gflag values are
/// used directly.
class TraceReplayerConfigPrompt {
 public:
  TraceReplayerConfigPrompt() = default;
  ~TraceReplayerConfigPrompt() = default;

  /// Prompts the user to enter or confirm all critical configs for trace
  /// replay. This includes root_dir, query_id, task_id, and node_id.
  /// If a gflag is already set, it shows the current value and allows the user
  /// to press Enter to keep it or type a new value to override.
  ///
  /// If --fast flag is set, skips all prompts and uses existing gflag values.
  /// After configuration, prints the equivalent command with --fast for future
  /// use.
  void run();
};

} // namespace facebook::velox::tool::trace
