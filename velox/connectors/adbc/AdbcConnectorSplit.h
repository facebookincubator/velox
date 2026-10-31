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

namespace facebook::velox::connector::adbc {

/// A remote database is not splittable the way files are, so each ADBC table
/// scan produces exactly one split. The split carries no information beyond
/// the connector id; it exists to trigger DataSource::addSplit.
struct AdbcConnectorSplit : public connector::ConnectorSplit {
  explicit AdbcConnectorSplit(const std::string& connectorId)
      : ConnectorSplit(connectorId, /*_splitWeight=*/0, /*_cacheable=*/false) {}

  std::string toString() const override;

  folly::dynamic serialize() const override;

  static std::shared_ptr<AdbcConnectorSplit> create(const folly::dynamic& obj);

  static void registerSerDe();

  VELOX_DEFINE_CLASS_NAME(AdbcConnectorSplit)
};

} // namespace facebook::velox::connector::adbc
