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

#include "velox/connectors/hive/RotationWriter.h"

#include "velox/common/memory/MemoryArbitrator.h"

namespace facebook::velox::connector::hive {

RotationWriter::RotationWriter(
    std::unique_ptr<dwio::common::Writer> writer,
    std::shared_ptr<HiveWriterInfo> writerInfo,
    std::unique_ptr<io::IoStatistics> ioStats,
    uint64_t maxTargetFileBytes,
    bool canRotate,
    WriterCreator writerCreator)
    : writerInfo_(std::move(writerInfo)),
      ioStats_(std::move(ioStats)),
      writer_(std::move(writer)),
      maxTargetFileBytes_(maxTargetFileBytes),
      canRotate_(canRotate),
      writerCreator_(std::move(writerCreator)) {
  VELOX_CHECK_NOT_NULL(writerInfo_);
  VELOX_CHECK_NOT_NULL(ioStats_);
  VELOX_CHECK_NOT_NULL(writerCreator_);
}

void RotationWriter::write(const VectorPtr& data) {
  memory::NonReclaimableSectionGuard nonReclaimableGuard(
      writerInfo_->nonReclaimableSectionHolder.get());

  ensureWriter();

  writer_->write(data);
  const auto numRows = data->size();
  writerInfo_->numWrittenRows += numRows;
  writerInfo_->currentFileWrittenRows += numRows;
  writerInfo_->inputSizeInBytes += data->estimateFlatSize();

  if (!canRotate_ || maxTargetFileBytes_ == 0) {
    return;
  }

  const auto currentFileBytes = getCurrentFileBytes();
  if (currentFileBytes >= maxTargetFileBytes_) {
    rotateWriter();
  }
}

void RotationWriter::flush() {
  if (writer_ != nullptr) {
    writer_->flush();
  }
}

bool RotationWriter::finish() {
  if (writer_ == nullptr) {
    return true;
  }
  memory::NonReclaimableSectionGuard nonReclaimableGuard(
      writerInfo_->nonReclaimableSectionHolder.get());
  return writer_->finish();
}

void RotationWriter::close() {
  if (writer_ != nullptr) {
    memory::NonReclaimableSectionGuard nonReclaimableGuard(
        writerInfo_->nonReclaimableSectionHolder.get());
    writer_->close();
    finalizeWriterFile();
  }
}

void RotationWriter::abort() {
  if (writer_ != nullptr) {
    memory::NonReclaimableSectionGuard nonReclaimableGuard(
        writerInfo_->nonReclaimableSectionHolder.get());
    writer_->abort();
  }
}

void RotationWriter::ensureWriter() {
  if (writer_ == nullptr) {
    writer_ = writerCreator_();
    VELOX_CHECK_NOT_NULL(writer_);
  }
}

void RotationWriter::rotateWriter() {
  VELOX_CHECK_NOT_NULL(writer_);

  // Close the writer first to flush all data including footer.
  writer_->close();

  // Finalize the current file state.
  finalizeWriterFile();

  // Release old writer's memory pools. The new writer will be created lazily
  // on the next write to avoid creating empty files.
  writer_.reset();

  ++writerInfo_->fileSequenceNumber;
}

void RotationWriter::finalizeWriterFile() {
  // Capture current file stats AFTER close to include footer bytes.
  const auto currentFileBytes = getCurrentFileBytes();

  // Finalize the current file into writtenFiles using the stored names.
  if (currentFileBytes > 0) {
    HiveFileInfo fileInfo;
    fileInfo.writeFileName = writerInfo_->currentWriteFileName;
    fileInfo.targetFileName = writerInfo_->currentTargetFileName;
    fileInfo.fileSize = currentFileBytes;
    fileInfo.numRows = writerInfo_->currentFileWrittenRows;
    // Reset for next file.
    writerInfo_->currentFileWrittenRows = 0;
    writerInfo_->writtenFiles.push_back(std::move(fileInfo));
  }

  // Update cumulative stats as a snapshot of total stats so far.
  // This becomes the baseline for the next file.
  writerInfo_->cumulativeWrittenBytes = ioStats_->rawBytesWritten();
}

uint64_t RotationWriter::getCurrentFileBytes() const {
  const auto totalBytes = ioStats_->rawBytesWritten();
  const auto baselineBytes = writerInfo_->cumulativeWrittenBytes;
  // Sanity check: total should always be >= baseline since ioStats is
  // never reset and cumulative is a snapshot of rawBytesWritten at rotation.
  VELOX_DCHECK_GE(totalBytes, baselineBytes);
  return totalBytes - baselineBytes;
}

} // namespace facebook::velox::connector::hive
