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

#include <string>
#include <vector>

#include "velox/connectors/Connector.h"
#include "velox/type/Type.h"

#include <cudf/types.hpp>

namespace facebook::velox::cudf_velox::connector::parquet {

using namespace facebook::velox::connector;

// Parquet column handle only needs the column name (all columns are generated
// in the same way).
class ParquetColumnHandle : public ColumnHandle {
 public:
  explicit ParquetColumnHandle(
      const std::string& name,
      const TypePtr type,
      const cudf::data_type cudfDataType,
      const std::vector<ParquetColumnHandle>& children)
      : name_(name),
        type_(type),
        cudfDataType_(cudfDataType),
        children_(children) {}

  const std::string& name() const {
    return name_;
  }

  const TypePtr& type() const {
    return type_;
  }

  const cudf::data_type cudfDataType() const {
    return cudfDataType_;
  }

  const std::vector<ParquetColumnHandle>& children() const {
    return children_;
  }

  std::string toString() const;

 private:
  const std::string name_;
  const TypePtr type_;
  const cudf::data_type cudfDataType_;
  const std::vector<ParquetColumnHandle> children_;
};

class ParquetTableHandle : public ConnectorTableHandle {
 public:
  ParquetTableHandle(
      std::string connectorId,
      const std::string& tableName,
      bool filterPushdownEnabled,
      const RowTypePtr& dataColumns = nullptr);

  const std::string& tableName() const {
    return tableName_;
  }

  bool isFilterPushdownEnabled() const {
    return filterPushdownEnabled_;
  }

  // Schema of the table.  Need this for reading TEXTFILE.
  const RowTypePtr& dataColumns() const {
    return dataColumns_;
  }

  std::string toString() const override;

  static ConnectorTableHandlePtr create(
      const folly::dynamic& obj,
      void* context);

 private:
  const std::string tableName_;
  const bool filterPushdownEnabled_;
  const RowTypePtr dataColumns_;
};

} // namespace facebook::velox::cudf_velox::connector::parquet
