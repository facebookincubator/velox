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

#include "velox/common/file/FileSystems.h"
#include "velox/exec/Split.h"

#include <re2/re2.h>

namespace facebook::velox::exec::trace {

class QuerySplitTracer {
 public:
  explicit QuerySplitTracer(std::string traceDir);

  /// Serializes and writes out each split. Each serialized split is immediately
  /// flushed to a separate file to ensure that we can still replay a traced
  /// operator even if a crash occurs during execution.
  void write(const exec::Split& split);

  /// Lists the split info files and deserializes the splits. The splits are
  /// sorted by the file index generated during the tracing process, allowing us
  /// to replay the execution in the same order as the original split
  /// processing.
  std::vector<exec::Split> read() const;

 private:
  static int32_t extractFileIndex(const std::string& str);

  const std::string traceDir_;
  const std::shared_ptr<filesystems::FileSystem> fs_;
  int32_t fileId_{0};

  static inline RE2 kFileRegExp{std::string(R"(.+\.split\.(\d+)$)")};
};
} // namespace facebook::velox::exec::trace
