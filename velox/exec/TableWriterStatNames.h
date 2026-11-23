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

#include <string_view>

namespace facebook::velox::exec {

/// Names of the runtime stats reported by TableWriter.
struct TableWriterStatNames {
  /// The running wall time of a writer operator from creation to close.
  static constexpr std::string_view kRunningWallNanos{"runningWallNanos"};

  /// The number of files written by this writer operator.
  static constexpr std::string_view kNumWrittenFiles{"numWrittenFiles"};

  /// The file write IO walltime.
  static constexpr std::string_view kWriteIOTime{"writeIOWallNanos"};

  /// The walltime spend on file write data recoding.
  static constexpr std::string_view kWriteRecodeTime{"writeRecodeWallNanos"};

  /// The walltime spent on file write data compression.
  static constexpr std::string_view kWriteCompressionTime{
      "writeCompressionWallNanos"};
};

} // namespace facebook::velox::exec
