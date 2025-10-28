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

#include "velox/connectors/hive/HivePartitionUtil.h"
#include "velox/connectors/hive/iceberg/PartitionSpec.h"

namespace facebook::velox::connector::hive::iceberg {

// Converts a partition value to its string representation for use in
// partition directory path. The format follows the Iceberg specification
// for partition path encoding.
class IcebergPartitionPath : public HivePartitionUtil {
 public:
  explicit IcebergPartitionPath(TransformType transformType)
      : transformType_(transformType) {}

  ~IcebergPartitionPath() override = default;

  using HivePartitionUtil::toPartitionString;

  std::string toPartitionString(int32_t value, const TypePtr& type)
      const override;

  std::string toPartitionString(Timestamp value, const TypePtr& type)
      const override;

  std::string toPartitionString(StringView value, const TypePtr& type)
      const override;

 private:
  const TransformType transformType_;
};

using IcebergPartitionPathPtr = std::shared_ptr<const IcebergPartitionPath>;

} // namespace facebook::velox::connector::hive::iceberg
