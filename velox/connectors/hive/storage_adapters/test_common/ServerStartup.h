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

#include <chrono>
#include <functional>
#include <string_view>

namespace facebook::velox::test {

class ServerStartup {
 public:
  /// Default for the 'timeout' argument of wait().
  static constexpr std::chrono::seconds kDefaultTimeout{30};

  /// Calls 'isReady' every 100 ms until it returns true. Throws if
  /// 'isRunning' returns false first, or if 'timeout' passes.
  static void wait(
      std::string_view serverName,
      const std::function<bool()>& isReady,
      const std::function<bool()>& isRunning,
      std::chrono::seconds timeout = kDefaultTimeout);
};

} // namespace facebook::velox::test
