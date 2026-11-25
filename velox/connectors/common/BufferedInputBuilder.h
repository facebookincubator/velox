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

#include "velox/common/caching/FileHandle.h"
#include "velox/connectors/Connector.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Reader.h"

namespace facebook::velox::connector {

/// Creates the BufferedInput for reading a file. A single builder is registered
/// for the process. The builder registered by default creates the input with
/// 'createDefault'. 'registerBuilder' replaces it.
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
      std::shared_ptr<io::IoStatistics> ioStatistics,
      std::shared_ptr<IoStats> ioStats,
      folly::Executor* executor,
      const folly::F14FastMap<std::string, std::string>& fileReadOps = {}) = 0;

  /// Creates:
  /// - a CachedBufferedInput when the query has a data cache;
  /// - a BufferedInput for Nimble files without direct buffered input;
  /// - a DirectBufferedInput otherwise.
  ///
  /// @param fileHandle The file to read.
  /// @param readerOpts Reader options. The file format and the Nimble direct
  /// buffered input flag choose the input. CachedBufferedInput and
  /// DirectBufferedInput get all the options; BufferedInput gets only the
  /// memory pool.
  /// @param connectorQueryCtx Supplies the data cache and the scan ID that
  /// keys the access tracker.
  /// @param ioStatistics Typed counters the input updates: bytes read, cache
  /// hits and I/O latencies.
  /// @param ioStats Free-form counters passed to each file read, for the file
  /// system to record its own metrics.
  /// @param executor Runs asynchronous loads. Not used for Nimble files
  /// without direct buffered input.
  /// @param fileReadOps File-system-specific options passed to each file read.
  static std::unique_ptr<dwio::common::BufferedInput> createDefault(
      const FileHandle& fileHandle,
      const dwio::common::ReaderOptions& readerOpts,
      const ConnectorQueryCtx* connectorQueryCtx,
      std::shared_ptr<io::IoStatistics> ioStatistics,
      std::shared_ptr<IoStats> ioStats,
      folly::Executor* executor,
      const folly::F14FastMap<std::string, std::string>& fileReadOps);

 private:
  static std::shared_ptr<BufferedInputBuilder> builder_;
};

} // namespace facebook::velox::connector
