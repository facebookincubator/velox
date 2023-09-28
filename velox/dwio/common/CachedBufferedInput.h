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

#include "velox/common/io/CachedBufferedInput.h"
#include "velox/dwio/common/Options.h"

namespace facebook::velox::dwio::common {
class CachedBufferedInput : public facebook::velox::io::CachedBufferedInput {
 public:
  CachedBufferedInput(
      std::shared_ptr<ReadFile> readFile,
      const MetricsLogPtr& metricsLog,
      uint64_t fileNum,
      cache::AsyncDataCache* FOLLY_NONNULL cache,
      std::shared_ptr<cache::ScanTracker> tracker,
      uint64_t groupId,
      std::shared_ptr<IoStatistics> ioStats,
      folly::Executor* FOLLY_NULLABLE executor,
      const ReaderOptions& readerOptions)
      : facebook::velox::io::CachedBufferedInput(
            readFile,
            fileNum,
            cache,
            tracker,
            groupId,
            ioStats,
            readerOptions.loadQuantum(),
            readerOptions.maxCoalesceDistance(),
            executor,
            &readerOptions.getMemoryPool()) {}

  CachedBufferedInput(
      std::shared_ptr<ReadFileInputStream> input,
      uint64_t fileNum,
      cache::AsyncDataCache* FOLLY_NONNULL cache,
      std::shared_ptr<cache::ScanTracker> tracker,
      uint64_t groupId,
      std::shared_ptr<IoStatistics> ioStats,
      folly::Executor* FOLLY_NULLABLE executor,
      const ReaderOptions& readerOptions)
      : facebook::velox::io::CachedBufferedInput(
            input,
            fileNum,
            cache,
            tracker,
            groupId,
            ioStats,
            readerOptions.loadQuantum(),
            readerOptions.maxCoalesceDistance(),
            executor,
            &readerOptions.getMemoryPool()) {}
};
} // namespace facebook::velox::dwio::common