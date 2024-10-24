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

#include <re2/re2.h>
#include "velox/common/file/FileInputStream.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/Split.h"

namespace facebook::velox::exec::trace {
/// Used to load the input splits from a set of traced 'TableScan' operators for
/// replay.
///
/// Currently, it only works with 'HiveConnectorSplit'. In the future, it will
/// be extended to handle more types of splits, such as
/// 'IcebergHiveConnectorSplit'.
class QuerySplitReader {
 public:
  /// 'traceDirs' provides a list of directories with each one containing the
  /// traced split info file for one table scan operator.
  explicit QuerySplitReader(
      std::vector<std::string> traceDirs,
      memory::MemoryPool* pool);

  /// Reads from 'splitInfoStream_' and deserializes to 'splitInfos'. Returns
  /// all the collected tracing splits.
  std::vector<exec::Split> read() const;

 private:
  static std::vector<std::string> getSplitInfos(
      common::FileInputStream* stream);

  std::unique_ptr<common::FileInputStream> getSplitInputStream(
      const std::string& traceDir) const;

  const std::vector<std::string> traceDirs_;
  const std::shared_ptr<filesystems::FileSystem> fs_;
  memory::MemoryPool* const pool_;
};
} // namespace facebook::velox::exec::trace
