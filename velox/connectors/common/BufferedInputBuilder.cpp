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

#include "velox/connectors/common/BufferedInputBuilder.h"
#include "velox/dwio/common/CachedBufferedInput.h"
#include "velox/dwio/common/DirectBufferedInput.h"

namespace facebook::velox::connector {

namespace {

class DefaultBufferedInputBuilder : public BufferedInputBuilder {
 public:
  std::unique_ptr<dwio::common::BufferedInput> create(
      const FileHandle& fileHandle,
      const dwio::common::ReaderOptions& readerOpts,
      const ConnectorQueryCtx* connectorQueryCtx,
      std::shared_ptr<io::IoStatistics> ioStatistics,
      std::shared_ptr<IoStats> ioStats,
      folly::Executor* executor,
      const folly::F14FastMap<std::string, std::string>& fileReadOps) override {
    return createDefault(
        fileHandle,
        readerOpts,
        connectorQueryCtx,
        ioStatistics,
        ioStats,
        executor,
        fileReadOps);
  }
};

} // namespace

// static
std::unique_ptr<dwio::common::BufferedInput>
BufferedInputBuilder::createDefault(
    const FileHandle& fileHandle,
    const dwio::common::ReaderOptions& readerOpts,
    const ConnectorQueryCtx* connectorQueryCtx,
    std::shared_ptr<io::IoStatistics> ioStatistics,
    std::shared_ptr<IoStats> ioStats,
    folly::Executor* executor,
    const folly::F14FastMap<std::string, std::string>& fileReadOps) {
  if (connectorQueryCtx->cache()) {
    return std::make_unique<dwio::common::CachedBufferedInput>(
        fileHandle.file,
        dwio::common::MetricsLog::voidLog(),
        fileHandle.uuid,
        connectorQueryCtx->cache(),
        Connector::getTracker(
            connectorQueryCtx->scanId(), readerOpts.loadQuantum()),
        fileHandle.groupId,
        ioStatistics,
        std::move(ioStats),
        executor,
        readerOpts,
        fileReadOps);
  }
  if (readerOpts.fileFormat() == dwio::common::FileFormat::NIMBLE &&
      !readerOpts.nimbleDirectBufferedInputEnabled()) {
    return std::make_unique<dwio::common::BufferedInput>(
        fileHandle.file,
        readerOpts.memoryPool(),
        dwio::common::MetricsLog::voidLog(),
        ioStatistics.get(),
        ioStats.get(),
        dwio::common::BufferedInput::kMaxMergeDistance,
        std::nullopt,
        fileReadOps);
  }
  return std::make_unique<dwio::common::DirectBufferedInput>(
      fileHandle.file,
      dwio::common::MetricsLog::voidLog(),
      fileHandle.uuid,
      Connector::getTracker(
          connectorQueryCtx->scanId(), readerOpts.loadQuantum()),
      fileHandle.groupId,
      std::move(ioStatistics),
      std::move(ioStats),
      executor,
      readerOpts,
      fileReadOps);
}

// static
std::shared_ptr<BufferedInputBuilder> BufferedInputBuilder::builder_ =
    std::make_shared<DefaultBufferedInputBuilder>();

} // namespace facebook::velox::connector
