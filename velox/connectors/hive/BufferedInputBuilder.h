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
#include <folly/Executor.h>

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Reader.h"

namespace facebook::velox::connector::hive {

/// Registering a different implementation of BufferedInput is allowed using
/// 'registerBuilder' API.
class BufferedInputBuilder {
 public:
  virtual ~BufferedInputBuilder() = default;

  static const std::shared_ptr<BufferedInputBuilder>& getInstance() {
    VELOX_CHECK_NOT_NULL(builder_, "Builder is not registered");
    return builder_;
  }

  static void registerBuilder(std::shared_ptr<BufferedInputBuilder> builder) {
    VELOX_CHECK_NOT_NULL(builder);
    builder_ = std::move(builder);
  }

  virtual std::unique_ptr<dwio::common::BufferedInput> create(
      const FileHandle& fileHandle,
      const dwio::common::ReaderOptions& readerOpts,
      const ConnectorQueryCtx* connectorQueryCtx,
      std::shared_ptr<io::IoStatistics> ioStats,
      std::shared_ptr<filesystems::File::IoStats> fsStats,
      folly::Executor* executor,
      const folly::F14FastMap<std::string, std::string>& fileReadOps = {}) = 0;

 private:
  static std::shared_ptr<BufferedInputBuilder> builder_;
};

} // namespace facebook::velox::connector::hive
