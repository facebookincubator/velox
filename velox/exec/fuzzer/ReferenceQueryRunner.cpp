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
#include <unordered_map>

#include <gflags/gflags.h>

#include "velox/core/PlanNode.h"
#include "velox/exec/fuzzer/PrestoSql.h"
#include "velox/exec/fuzzer/ReferenceQueryRunner.h"

DEFINE_string(
    table_name_prefix,
    "",
    "Prefix for temporary table names created by fuzzer reference query "
    "runners. Use to avoid collisions when running multiple fuzzer instances "
    "against the same database server.");

namespace facebook::velox::exec::test {

std::string ReferenceQueryRunner::getTableName(
    const core::ValuesNode& valuesNode) {
  if (FLAGS_table_name_prefix.empty()) {
    return fmt::format("t_{}", valuesNode.id());
  }
  return fmt::format("{}_t_{}", FLAGS_table_name_prefix, valuesNode.id());
}

std::string ReferenceQueryRunner::getWriteTableName() {
  if (FLAGS_table_name_prefix.empty()) {
    return "tmp_write";
  }
  return fmt::format("{}_tmp_write", FLAGS_table_name_prefix);
}

std::unordered_map<std::string, std::vector<velox::RowVectorPtr>>
ReferenceQueryRunner::getAllTables(const core::PlanNodePtr& plan) {
  std::unordered_map<std::string, std::vector<velox::RowVectorPtr>> result;
  if (const auto valuesNode =
          std::dynamic_pointer_cast<const core::ValuesNode>(plan)) {
    result.insert({getTableName(*valuesNode), valuesNode->values()});
  } else {
    for (const auto& source : plan->sources()) {
      auto tablesAndNames = getAllTables(source);
      result.insert(tablesAndNames.begin(), tablesAndNames.end());
    }
  }
  return result;
}
} // namespace facebook::velox::exec::test
