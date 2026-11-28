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
#include <vector>

#include "velox/connectors/Connector.h"

namespace facebook::velox::connector::lance {

/// Split representing a Lance dataset to scan.
struct LanceConnectorSplit : public connector::ConnectorSplit {
  const std::string datasetPath;
  /// Fragment IDs to scan. Empty means scan all fragments.
  const std::vector<uint64_t> fragmentIds;

  /// Create a split that scans all fragments.
  explicit LanceConnectorSplit(
      const std::string& connectorId,
      const std::string& datasetPath)
      : ConnectorSplit(connectorId), datasetPath(datasetPath) {}

  /// Create a split that scans specific fragments.
  LanceConnectorSplit(
      const std::string& connectorId,
      const std::string& datasetPath,
      std::vector<uint64_t> fragmentIds)
      : ConnectorSplit(connectorId),
        datasetPath(datasetPath),
        fragmentIds(std::move(fragmentIds)) {}

  std::string toString() const override {
    return fmt::format(
        "[LanceConnectorSplit: connectorId {}, path {}, fragments {}]",
        connectorId,
        datasetPath,
        fragmentIds.size());
  }
};

} // namespace facebook::velox::connector::lance
