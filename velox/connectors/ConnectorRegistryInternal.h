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

#include <memory>
#include <string>
#include <unordered_map>

namespace facebook::velox::connector {

class Connector;

// Internal helper shared by Connector.cpp and ConnectorRegistry.cpp.
// Not part of the public API. Do not include from outside velox/connectors/.
inline std::unordered_map<std::string, std::shared_ptr<Connector>>&
connectors() {
  static std::unordered_map<std::string, std::shared_ptr<Connector>> instance;
  return instance;
}

} // namespace facebook::velox::connector
