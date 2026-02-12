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

#include "velox/exec/tests/utils/TpcdsQueryBuilder.h"

#include <folly/executors/IOThreadPoolExecutor.h>

namespace facebook::velox::cudf_velox::exec::test {

/// CuDF-enabled extension of TpcdsQueryBuilder.
/// Registers CudfHiveConnector instead of plain HiveConnector and
/// calls cudf_velox::registerCudf() / unregisterCudf() for GPU operator
/// replacements.
class CudfTpcdsQueryBuilder
    : public facebook::velox::exec::test::TpcdsQueryBuilder {
 public:
  explicit CudfTpcdsQueryBuilder(
      dwio::common::FileFormat format = dwio::common::FileFormat::PARQUET,
      folly::Executor* ioExecutor = nullptr);

  /// Enable CuDF GPU operator replacements.
  /// Must be called before getQueryPlan().
  void enableCudf();

  /// Unregisters cuDF operators and connectors.
  void shutdown() override;

 protected:
  /// Registers CudfHiveConnector instead of HiveConnector.
  void registerHiveConnector(
      const std::string& connectorId,
      folly::Executor* ioExecutor = nullptr) override;

 private:
  bool cudfEnabled_{false};
  folly::Executor* ioExecutor_{nullptr};
};

} // namespace facebook::velox::cudf_velox::exec::test
