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

#include <cstddef>
#include <cstdint>
#include <vector>

namespace facebook::velox::exec::test {

class Ports {
 public:
  /// Returns a port that is free at the time of the call. Another process can
  /// take it before the caller binds it. Repeated calls can return the same
  /// port unless the caller binds each one first; use findFreePorts() to get
  /// several ports.
  static uint16_t findFreePort();

  /// Returns 'numPorts' distinct ports that are free at the time of the call.
  /// Another process can take them before the caller binds them.
  static std::vector<uint16_t> findFreePorts(size_t numPorts);

  /// Returns true if a server accepts connections on 'port' of 127.0.0.1.
  static bool isListening(uint16_t port);
};

} // namespace facebook::velox::exec::test
