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
#include <optional>
#include <string>
#include <vector>

#include "velox/connectors/hive/hudi/HudiLogBlock.h"

namespace facebook::velox::connector::hive::hudi {

/// One record key removed by a Hudi delete log block.
struct HudiDeleteRecord {
  /// The deleted row's Hudi record key. A null Avro value decodes to an empty
  /// string; Hudi requires non-null record keys in practice.
  std::string recordKey;

  /// Partition the deleted row belongs to. A null Avro value decodes to an
  /// empty string, indistinguishable from an unpartitioned table's empty
  /// partition path.
  std::string partitionPath;

  /// The delete's ordering (precombine) value, used to decide whether the
  /// delete supersedes a base/log record for the same key. Empty when the
  /// source value was null. Only integral ordering values are decoded for now.
  std::optional<int64_t> orderingValue;
};

/// Decodes the record keys carried by a Hudi delete log block. The block
/// content is a HoodieDeleteRecordList Avro datum: an array of
/// {recordKey, partitionPath, orderingVal} records. Throws if `block` is not a
/// delete block or carries an unsupported (non-integral) ordering value type.
std::vector<HudiDeleteRecord> decodeHudiDeleteBlock(const HudiLogBlock& block);

} // namespace facebook::velox::connector::hive::hudi
