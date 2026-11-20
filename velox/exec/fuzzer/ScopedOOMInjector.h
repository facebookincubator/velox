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

#include <atomic>
#include <functional>
#include <string>

namespace facebook::velox::exec::test {

/// Injects OOMs into query execution while enabled. Relies on TestValue, so it
/// works only in debug builds, and only one instance can be enabled at a time.
class ScopedOOMInjector {
 public:
  /// Checks 'oomConditionChecker' at most once per 'oomCheckIntervalMs' on
  /// memory reservation and triggers an OOM when it returns true.
  ScopedOOMInjector(
      const std::function<bool()>& oomConditionChecker,
      uint64_t oomCheckIntervalMs);

  ~ScopedOOMInjector();

  /// Error message of the injected OOMs.
  inline static const std::string kErrorMessage = "Injected OOM";

  void enable();

 private:
  // If more than one instance of this class is enabled, they'll overwrite
  // each other, so make sure no one does this by accident.
  inline static std::atomic_bool enabled_{false};

  // When this function returns true, an OOM will be triggered.
  const std::function<bool()> oomConditionChecker_;
  // The interval between each check of the condition.
  const uint64_t oomCheckIntervalMs_;

  std::atomic<size_t> lastOomCheckTime_{0};
};

} // namespace facebook::velox::exec::test
