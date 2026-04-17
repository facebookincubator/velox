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

#include <functional>
#include <memory>

#include "velox/common/io/IoStatistics.h"
#include "velox/connectors/hive/HiveWriterTypes.h"
#include "velox/connectors/hive/PartitionWriterInterface.h"
#include "velox/dwio/common/Writer.h"

namespace facebook::velox::connector::hive {

/// Decorates a dwio::common::Writer with file size-based rotation.
/// When the current file exceeds maxTargetFileBytes, closes it and
/// opens a new one via the writerCreator callback. Tracks all written
/// files and their metadata (HiveFileInfo) in HiveWriterInfo.
class RotationWriter : public dwio::common::Writer,
                       public PartitionWriterInterface {
 public:
  /// Callback to create a new underlying writer on rotation or lazy creation.
  using WriterCreator = std::function<std::unique_ptr<dwio::common::Writer>()>;

  /// @param writer Initial underlying writer (may be nullptr for lazy
  /// creation).
  /// @param writerInfo Tracks per-writer state (parameters, stats, files).
  /// @param ioStats IO statistics for the underlying writer.
  /// @param maxTargetFileBytes File size threshold for rotation. 0 disables
  /// rotation.
  /// @param canRotate Whether rotation is allowed (false for bucketed/sorted
  /// writes).
  /// @param writerCreator Factory to create new underlying writers on rotation.
  RotationWriter(
      std::unique_ptr<dwio::common::Writer> writer,
      std::shared_ptr<HiveWriterInfo> writerInfo,
      std::unique_ptr<io::IoStatistics> ioStats,
      uint64_t maxTargetFileBytes,
      bool canRotate,
      WriterCreator writerCreator);

  void write(const VectorPtr& data) override;
  void flush() override;
  bool finish() override;
  void close() override;
  void abort() override;

  /// Access writer info for commit message generation.
  const std::shared_ptr<HiveWriterInfo>& writerInfo() const override {
    return writerInfo_;
  }

  /// Access IO statistics.
  io::IoStatistics* ioStats() const override {
    return ioStats_.get();
  }

 private:
  void ensureWriter();
  void rotateWriter();
  void finalizeWriterFile();
  uint64_t getCurrentFileBytes() const;

  // writerInfo_ and ioStats_ must be declared before writer_ so they outlive
  // it. The inner writer (e.g., SortingWriter) holds raw pointers to pools
  // owned by writerInfo_, so writer_ must be destroyed first.
  std::shared_ptr<HiveWriterInfo> writerInfo_;
  std::unique_ptr<io::IoStatistics> ioStats_;
  std::unique_ptr<dwio::common::Writer> writer_;
  uint64_t maxTargetFileBytes_;
  bool canRotate_;
  WriterCreator writerCreator_;
};

} // namespace facebook::velox::connector::hive
