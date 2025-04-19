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

#include "velox/connectors/thrift/ThriftConnector.h"

#include <velox/core/QueryConfig.h>

namespace facebook::velox::connector::thrift {

std::shared_ptr<Connector> ThriftConnectorFactory::newConnector(
    const std::string& id,
    std::shared_ptr<const config::ConfigBase> /* unused */,
    folly::Executor* /* unused */,
    folly::Executor* /* unused */) {
  return std::make_shared<ThriftConnector>(
      id, std::make_shared<ThriftClientImpl>());
}

} // namespace facebook::velox::connector::thrift
