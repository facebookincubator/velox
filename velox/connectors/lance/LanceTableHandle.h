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

#include <string>

#include "velox/connectors/Connector.h"

namespace facebook::velox::connector::lance {

class LanceTableHandle : public ConnectorTableHandle {
 public:
  LanceTableHandle(
      const std::string& connectorId,
      const std::string& datasetPath)
      : ConnectorTableHandle(connectorId), datasetPath_(datasetPath) {}

  const std::string& name() const override {
    return datasetPath_;
  }

  const std::string& datasetPath() const {
    return datasetPath_;
  }

  std::string toString() const override {
    return fmt::format("LanceTableHandle [path: {}]", datasetPath_);
  }

 private:
  const std::string datasetPath_;
};

} // namespace facebook::velox::connector::lance
