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

#include "velox/connectors/Connector.h"

namespace facebook::velox::connector::clp {

struct ClpConnectorSplit : public connector::ConnectorSplit {
  ClpConnectorSplit(
      const std::string& connectorId,
      const std::string& path,
      std::shared_ptr<std::string> kqlQuery)
      : connector::ConnectorSplit(connectorId),
        path_(path),
        kqlQuery_(kqlQuery) {}

  [[nodiscard]] std::string toString() const override {
    return fmt::format(
        "CLP Split: path: {}, kqlQuery: {}",
        path_,
        kqlQuery_ ? *kqlQuery_ : "<null>");
  }

  const std::string path_;
  std::shared_ptr<std::string> kqlQuery_;
};

} // namespace facebook::velox::connector::clp
