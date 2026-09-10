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

#include <cstdint>
#include <memory>
#include <optional>

#include <folly/container/F14Map.h>
#include <folly/dynamic.h>

namespace facebook::velox {

namespace io {
class IoStatistics;
}

struct FileProperties {
  std::optional<int64_t> fileSize;
  std::optional<int64_t> modificationTime;
  std::optional<int64_t> readRangeHint{std::nullopt};
  std::shared_ptr<std::string> extraFileInfo{nullptr};
  folly::F14FastMap<std::string, std::string> fileReadOps{};

  /// Non-owning pointer to the statistics of the data source that opened the
  /// file. Set locally by the reader at open time, so 'serialize()' skips it.
  io::IoStatistics* ioStatistics{nullptr};

  folly::dynamic serialize() const;

  /// Reads what 'serialize()' wrote. Absent keys fall back to the member
  /// default.
  static FileProperties create(const folly::dynamic& obj);
};

} // namespace facebook::velox
