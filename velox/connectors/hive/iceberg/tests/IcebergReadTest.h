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

#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

#include <string>

namespace facebook::velox::connector::hive::iceberg {

class IcebergSplitBuilder : public exec::test::HiveConnectorSplitBuilder {
 public:
  const std::string kHiveConnectorId = "test-hive";

  IcebergSplitBuilder(std::string filePath)
      : HiveConnectorSplitBuilder{std::move(filePath)} {}

  IcebergSplitBuilder& deleteFiles(
      const std::vector<IcebergDeleteFile>& deleteFiles) {
    deleteFiles_ = deleteFiles;
    return *this;
  }

  IcebergSplitBuilder& deleteFile(const IcebergDeleteFile& deleteFile) {
    deleteFiles_.push_back(std::move(deleteFile));
    return *this;
  }

  std::shared_ptr<HiveIcebergSplit> build() const {
    return std::make_shared<HiveIcebergSplit>(
        kHiveConnectorId,
        "file:" + filePath_,
        fileFormat_,
        start_,
        length_,
        partitionKeys_,
        tableBucketNumber_,
        {},
        {},
        deleteFiles_);
  }

 private:
  std::vector<IcebergDeleteFile> deleteFiles_;
};

} // namespace facebook::velox::connector::hive::iceberg
