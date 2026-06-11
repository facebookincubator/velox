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
#include <string>
#include <vector>

namespace facebook::velox::connector::hive::iceberg {

enum class ChangelogOperation { INSERT, DELETE, UPDATE_BEFORE, UPDATE_AFTER };

struct ChangelogSplitInfo {
  ChangelogOperation operation;
  int64_t ordinal;
  int64_t snapshotId;
  std::vector<std::string> dataColumnNames;

  ChangelogSplitInfo(
      ChangelogOperation op,
      int64_t ord,
      int64_t snapId,
      std::vector<std::string> colNames)
      : operation(op),
        ordinal(ord),
        snapshotId(snapId),
        dataColumnNames(std::move(colNames)) {}
};

} // namespace facebook::velox::connector::hive::iceberg
