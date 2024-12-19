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
#include "velox/optimizer/connectors/ConnectorMetadata.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::runner {

/// Translates from table name to table. 'defaultConnector is used
/// to look up the name if the name has no catalog. 'defaultSchema'
/// is used to fill in the schema if the name has no schema.
class Schema {
 public:
  virtual ~Schema() = default;

  Schema(
      const std::shared_ptr<connector::Connector>& defaultConnector,
      const std::string& defaultSchema)
      : defaultConnector_(defaultConnector), defaultSchema_(defaultSchema) {}

  virtual const connector::Table* findTable(const std::string& name);

 private:
  // Connector to use if name does not specify a catalog.
  const std::shared_ptr<connector::Connector> defaultConnector_;
  const std::string defaultSchema_;
};

} // namespace facebook::velox::runner
