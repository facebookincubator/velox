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

#include "velox/connectors/hive/storage_adapters/test_common/ServerStartup.h"

#include <thread>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::test {

void ServerStartup::wait(
    std::string_view serverName,
    const std::function<bool()>& isReady,
    const std::function<bool()>& isRunning,
    std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (!isRunning()) {
      VELOX_FAIL("{} exited before it was ready", serverName);
    }
    if (isReady()) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  VELOX_FAIL(
      "Timed out after {} seconds waiting for {} to start",
      timeout.count(),
      serverName);
}

} // namespace facebook::velox::test
