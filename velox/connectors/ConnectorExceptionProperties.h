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

#include "velox/common/base/VeloxException.h"

namespace facebook::velox {

/// Structured context for an error thrown while a connector reads data.
struct ConnectorExceptionProperties : public ExceptionContextProperties {
  /// The owner reported by the connector; empty when none is declared.
  std::string owner;
  /// The id of the connector that threw, e.g. "hive".
  std::string connectorId;
  /// The table the connector was reading (ConnectorTableHandle::name()).
  std::string tableName;
};

} // namespace facebook::velox
