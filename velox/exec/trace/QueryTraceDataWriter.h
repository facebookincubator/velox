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

#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"
#include "velox/vector/VectorStream.h"

namespace facebook::velox::exec {

/// Represents a trace file for writing the serialized input data into it.
class TraceWriteFile {
 public:
  static std::unique_ptr<TraceWriteFile> create(const std::string& path);

  uint64_t write(std::unique_ptr<folly::IOBuf> iobuf) const;

  void close();

  explicit TraceWriteFile(std::string path);

  const std::string path_;
  std::unique_ptr<WriteFile> file_;
};

/// Used to write the trace input vectors to a file.
class TraceWriter {
 public:
  explicit TraceWriter(
      const std::string& path,
      const uint64_t writeBufferSize,
      memory::MemoryPool* pool)
      : path_(path), writeBufferSize_(writeBufferSize), pool_(pool) {}

  /// Serializes rows to 'batch_'.
  void write(const RowVectorPtr& rows);

  /// Flushes 'batch_' to an output buffer, and uses 'file_' to write it.
  uint64_t flush();

  /// Flushes remaining buffered input and closes 'file_'.
  void finish();

  /// Flush the trace data summaries to the disk.
  ///
  /// TODO: Only row type is flushed, ddd more summaries, such as rows, size.
  void writeSummary() const;

 private:
  const std::string path_;
  const uint64_t writeBufferSize_;
  memory::MemoryPool* const pool_;
  std::unique_ptr<VectorStreamGroup> batch_;
  std::unique_ptr<TraceWriteFile> file_;
  TypePtr type_;
};

} // namespace facebook::velox::exec
