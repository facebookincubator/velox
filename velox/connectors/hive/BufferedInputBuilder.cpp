/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"

namespace facebook::velox::connector::hive {

class DefaultBufferInputBuilder : public BufferedInputBuilder {
 public:
  std::unique_ptr<dwio::common::BufferedInput> create(
      const FileHandle& fileHandle,
      const dwio::common::ReaderOptions& readerOpts,
      const ConnectorQueryCtx* connectorQueryCtx,
      std::shared_ptr<io::IoStatistics> ioStatistics,
      std::shared_ptr<IoStats> ioStats,
      folly::Executor* executor,
      const folly::F14FastMap<std::string, std::string>& fileReadOps) override {
    return createBufferedInput(
        fileHandle,
        readerOpts,
        connectorQueryCtx,
        ioStatistics,
        ioStats,
        executor,
        fileReadOps);
  }
};

// static
std::shared_ptr<BufferedInputBuilder> BufferedInputBuilder::builder_ =
    std::make_shared<DefaultBufferInputBuilder>();

} // namespace facebook::velox::connector::hive
