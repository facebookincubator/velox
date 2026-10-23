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
#include "velox/connectors/adbc/AdbcConnectorSplit.h"

namespace facebook::velox::connector::adbc {

std::string AdbcConnectorSplit::toString() const {
  return fmt::format("[split: adbc, connector id {}]", connectorId);
}

folly::dynamic AdbcConnectorSplit::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = AdbcConnectorSplit::getClassName();
  obj["connectorId"] = connectorId;
  return obj;
}

// static
std::shared_ptr<AdbcConnectorSplit> AdbcConnectorSplit::create(
    const folly::dynamic& obj) {
  return std::make_shared<AdbcConnectorSplit>(obj["connectorId"].asString());
}

// static
void AdbcConnectorSplit::registerSerDe() {
  registerDeserializer<AdbcConnectorSplit>();
}

} // namespace facebook::velox::connector::adbc
