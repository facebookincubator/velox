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

#include <folly/io/IOBuf.h>
#include <cstdint>
#include <memory>
#include "velox/common/file/File.h"
#include "velox/common/file/FileInputStream.h"
#include "velox/serializers/VectorStream.h"

namespace facebook::velox::serializer {

/// Represents a file for writing the serialized pages into a disk file.
class SerializedPageFile {
 public:
  /// FileInfo struct containing the essential file information.
  struct FileInfo {
    uint32_t id;
    std::string path;
    uint64_t size;
  };

  static std::unique_ptr<SerializedPageFile> create(
      uint32_t id,
      const std::string& pathPrefix,
      const std::string& fileCreateConfig,
      IoStats* ioStats);

  uint32_t id() const {
    return id_;
  }

  uint64_t size();

  std::string path() const {
    return path_;
  }

  FileInfo fileInfo() const {
    return {id_, path_, (file_ != nullptr ? file_->size() : size_)};
  }

  uint64_t write(std::unique_ptr<folly::IOBuf> iobuf);

  WriteFile* file() {
    return file_.get();
  }

  /// Finishes writing and flushes any unwritten data.
  void finish();

 private:
  static inline std::atomic_int32_t ordinalCounter_{0};

  SerializedPageFile(
      uint32_t id,
      const std::string& pathPrefix,
      const std::string& fileCreateConfig,
      IoStats* ioStats);

  const uint32_t id_;

  const std::string path_;

  uint64_t size_{0};

  std::unique_ptr<WriteFile> file_;
};

/// Used to write 'RowVector' as serialized page data to a sequence of files.
class SerializedPageFileWriter {
 public:
  /// 'compressionKind' specifies the compression to use. 'pathPrefix' is a
  /// file path prefix. 'targetFileSize' is the target byte size of a single
  /// file. 'writeBufferSize' specifies the size limit of the buffered data
  /// before write to file. 'fileCreateConfig' specifies the file layout on
  /// remote storage which is storage system specific. 'serdeOptions' specifies
  /// the serialization options to use. 'serde' specifies the VectorSerde
  /// instance to use. 'pool' is used for buffering. 'ioStats' is used
  /// to collect filesystem I/O stats.
  SerializedPageFileWriter(
      const std::string& pathPrefix,
      uint64_t targetFileSize,
      uint64_t writeBufferSize,
      const std::string& fileCreateConfig,
      std::unique_ptr<VectorSerde::Options> serdeOptions,
      VectorSerde* serde,
      memory::MemoryPool* pool,
      IoStats* ioStats);

  // TODO(jtan6): Remove after other dependencies switch to the new ctor.
  SerializedPageFileWriter(
      const std::string& pathPrefix,
      uint64_t targetFileSize,
      uint64_t writeBufferSize,
      const std::string& fileCreateConfig,
      std::unique_ptr<VectorSerde::Options> serdeOptions,
      VectorSerde* serde,
      memory::MemoryPool* pool)
      : SerializedPageFileWriter(
            pathPrefix,
            targetFileSize,
            writeBufferSize,
            fileCreateConfig,
            std::move(serdeOptions),
            serde,
            pool,
            nullptr) {}

  virtual ~SerializedPageFileWriter() = default;

  /// Adds 'rows' for the positions in 'indices' into 'this'.
  /// Returns the size to write.
  uint64_t write(
      const RowVectorPtr& rows,
      const folly::Range<IndexRange*>& indices);

  /// Closes the current output file if any. Subsequent calls to write will
  /// start a new one.
  void finishFile();

  /// Returns the number of current finished files.
  size_t numFinishedFiles() const;

  /// Finishes this file writer and returns the written file info.
  /// NOTE: we don't allow write to a file writer after finish.
  std::vector<SerializedPageFile::FileInfo> finish();

 protected:
  // Invoked upon each vector append.
  virtual void updateAppendStats(
      uint64_t /* numRows */,
      uint64_t /* serializationTimeNs */) {}

  // Invoked upon each serialized page write to disk.
  virtual void updateWriteStats(
      uint64_t /* writtenBytes */,
      uint64_t /* flushTimeNs */,
      uint64_t /* fileWriteTimeNs */) {}

  // Invoked upon each file close.
  virtual void updateFileStats(const SerializedPageFile::FileInfo& /* file */) {
  }

  // Closes the current open file pointed by 'currentFile_'.
  virtual void closeFile();

  // Writes data from 'batch_' to the current output file. Returns the actual
  // written size.
  virtual uint64_t flush();

  FOLLY_ALWAYS_INLINE void checkNotFinished() const {
    VELOX_CHECK(!finished_, "SerializedPageFileWriter has finished");
  }

  // Returns an open file for write. If there is no open file, then
  // the function creates a new one. If the current open file exceeds the
  // target file size limit, then it first closes the current one and then
  // creates a new one. 'currentFile_' points to the current open file.
  SerializedPageFile* ensureFile();

  const std::string pathPrefix_;
  const uint64_t targetFileSize_;
  const uint64_t writeBufferSize_;
  const std::string fileCreateConfig_;

  const std::unique_ptr<VectorSerde::Options> serdeOptions_;
  memory::MemoryPool* const pool_;
  VectorSerde* const serde_;
  IoStats* const ioStats_;

  bool finished_{false};
  uint32_t nextFileId_{0};
  std::unique_ptr<VectorStreamGroup> batch_;
  std::unique_ptr<SerializedPageFile> currentFile_;
  std::vector<SerializedPageFile::FileInfo> finishedFiles_;
};

/// Used to read the serialized page data from a single file generated by
/// 'SerializedPageFileWriter'.
class SerializedPageFileReader {
 public:
  /// 'path' is the file path to read from. 'bufferSize' is the read buffer
  /// size. 'type' is the row type of the data. 'serde' is the VectorSerde
  /// instance to use. 'readOptions' specifies the deserialization options.
  /// 'pool' is used for buffering. 'ioStats' is used to collect
  /// filesystem I/O stats such as wsServiceTime.
  SerializedPageFileReader(
      const std::string& path,
      uint64_t bufferSize,
      const RowTypePtr& type,
      VectorSerde* serde,
      std::unique_ptr<VectorSerde::Options> readOptions,
      memory::MemoryPool* pool,
      IoStats* ioStats);

  virtual ~SerializedPageFileReader() = default;

  bool nextBatch(RowVectorPtr& rowVector);

 protected:
  virtual void updateFinalStats() {}

  virtual void updateSerializationTimeStats(uint64_t /* timeNs */) {}

  const std::unique_ptr<VectorSerde::Options> readOptions_;

  memory::MemoryPool* const pool_;

  VectorSerde* const serde_;

  const RowTypePtr type_;

  std::unique_ptr<common::FileInputStream> input_;
};

} // namespace facebook::velox::serializer
