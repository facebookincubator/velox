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
#include <vector>

namespace facebook::velox::exec::test {

class PortUtil {
 public:
  static constexpr std::chrono::seconds kWaitForPortTimeout{30};

  /// Find one or more free ports.
  ///
  /// Of course, there's no guarantee it won't be snatched by the time you
  /// actually want to use it, but it is safer than hard coding a port or
  /// randomly choosing one.
  ///
  /// NOTE: If you need multiple free ports, use getFreePorts() rather than
  /// calling getFreePort() multiple times. The latter may give you a same
  /// port multiple times, unless you started listening to the provided port
  /// before calling the function again.
  static int getFreePort();
  static std::vector<int> getFreePorts(std::size_t numPorts);

  /// Wait until a server accepts connections on the given port of 127.0.0.1.
  /// Throws if 'isRunning' returns false first, or if 'timeout' passes.
  static void waitForPort(
      int port,
      std::string_view serverName,
      const std::function<bool()>& isRunning,
      std::chrono::seconds timeout = kWaitForPortTimeout);
};

/// Deprecated: use PortUtil::getFreePort().
inline int getFreePort() {
  return PortUtil::getFreePort();
}

/// Deprecated: use PortUtil::getFreePorts().
inline std::vector<int> getFreePorts(std::size_t numPorts) {
  return PortUtil::getFreePorts(numPorts);
}

} // namespace facebook::velox::exec::test
