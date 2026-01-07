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

#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/parquet/reader/SemanticVersion.h"
#include "velox/dwio/parquet/thrift/ParquetThriftTypes.h"

namespace facebook::velox::parquet {

struct ParquetStatsContext : dwio::common::StatsContext {
 public:
  ParquetStatsContext() = default;

  ParquetStatsContext(const std::optional<SemanticVersion>& version)
      : parquetVersion(version) {}

  bool shouldIgnoreStatistics(thrift::Type::type type) const {
    // Follow parquet-java community's approach: check type first, then
    // version.
    // https://github.com/apache/parquet-java/blob/312a15f53a011d1dc4863df196c0169bdf6db629/parquet-column/src/main/java/org/apache/parquet/CorruptStatistics.java#L57
    if (type != thrift::Type::BYTE_ARRAY &&
        type != thrift::Type::FIXED_LEN_BYTE_ARRAY) {
      return false;
    }

    if (!parquetVersion.has_value()) {
      return true;
    }
    return parquetVersion->shouldIgnoreStatistics(type);
  }

 private:
  std::optional<SemanticVersion> parquetVersion;
};

} // namespace facebook::velox::parquet
