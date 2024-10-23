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
/// Used to write the input splits during the execution of a traced 'TableScan'
/// operator.
///
/// Currently, it only works with 'HiveConnectorSplit'. In the future, it will
/// be extended to handle more types of splits, such as
/// 'IcebergHiveConnectorSplit'.
class QuerySplitWriter {
 public:
  explicit QuerySplitWriter(std::string traceDir);

  /// Serializes and writes out each split. Each serialized split is immediately
  /// flushed to ensure that we can still replay a traced operator even if a
  /// crash occurs during execution.
  void write(const exec::Split& split) const;

  void finish();

 private:
  static std::unique_ptr<folly::IOBuf> serialize(const std::string& split);

  const std::string traceDir_;
  const std::shared_ptr<filesystems::FileSystem> fs_;
  std::unique_ptr<WriteFile> splitFile_;
  bool finished_{false};
};
} // namespace facebook::velox::exec::trace
