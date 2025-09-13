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

#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

using namespace facebook::velox::exec::test;

namespace facebook::velox::connector::hive::iceberg {
class IcebergReadTestBase : public HiveConnectorTestBase {
 public:
  IcebergReadTestBase()
      : config_{std::make_shared<facebook::velox::dwrf::Config>()} {
    // Make the writers flush per batch so that we can create non-aligned
    // RowGroups between the base data files and delete files
    flushPolicyFactory_ = []() {
      return std::make_unique<dwrf::LambdaFlushPolicy>([]() { return true; });
    };
  }

 protected:
  RowTypePtr rowType_{ROW({"c0"}, {BIGINT()})};
  dwio::common::FileFormat fileFormat_{dwio::common::FileFormat::DWRF};
  static constexpr int rowCount_ = 20000;
  std::shared_ptr<dwrf::Config> config_;
  std::function<std::unique_ptr<dwrf::DWRFFlushPolicy>()> flushPolicyFactory_;

  std::vector<int64_t> makeRandomDeleteValues(int32_t maxRowNumber);

  template <class T>
  std::vector<T> makeSequenceValues(int32_t numRows, int8_t repeat = 1);

  template <TypeKind KIND>
  std::string makeNotInList(
      const std::vector<typename TypeTraits<KIND>::NativeType>& deleteValues);

  core::PlanNodePtr tableScanNode(const RowTypePtr& outputRowType) const;

  std::vector<std::shared_ptr<ConnectorSplit>> makeIcebergSplits(
      const std::string& dataFilePath,
      const std::vector<IcebergDeleteFile>& deleteFiles = {},
      const std::unordered_map<std::string, std::optional<std::string>>&
          partitionKeys = {},
      const uint32_t splitCount = 1);
};
} // namespace facebook::velox::connector::hive::iceberg
