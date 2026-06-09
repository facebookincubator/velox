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

#include "velox/connectors/hive/paimon/PaimonConfig.h"

#include "velox/connectors/hive/paimon/PaimonConnector.h"

namespace facebook::velox::connector::hive::paimon {

PaimonConfig::PaimonConfig(std::shared_ptr<const config::ConfigBase> config)
    : FileConfig(
          std::move(config),
          PaimonConnectorFactory::kPaimonConnectorName) {}

} // namespace facebook::velox::connector::hive::paimon
