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

#include "velox/common/config/Config.h"
#include "velox/connectors/Connector.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetDataSource.h"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>

#include <vector>

// Parquet column handle only needs the column name (all columns are generated
// in the same way).
class ParquetColumnHandle : public ColumnHandle {
 public:
  explicit ParquetColumnHandle(
      const std::string& name,
      const cudf::data_type type,
      const std::vector<ParquetColumnHandle>& children)
      : name_(name), type_(type), children_(children) {}

  const std::string& name() const {
    return name_;
  }

  const cudf::data_type type() const {
    return type_;
  }

  const std::vector<ParquetColumnHandle>& children() const {
    return children_;
  }

 private:
  const std::string name_;
  const cudf::data_type type_;
  const std::vector<ParquetColumnHandle> children_;
};

class ParquetTableHandle : public ConnectorTableHandle {
 public:
  ParquetTableHandle(
      std::string connectorId,
      const std::string& tableName,
      bool filterPushdownEnabled,
      SubfieldFilters subfieldFilters,
      const core::TypedExprPtr& remainingFilter,
      const RowTypePtr& dataColumns = nullptr,
      const std::unordered_map<std::string, std::string>& tableParameters = {});

  const std::string& tableName() const {
    return tableName_;
  }

  bool isFilterPushdownEnabled() const {
    return filterPushdownEnabled_;
  }

  const core::TypedExprPtr& remainingFilter() const {
    return remainingFilter_;
  }

  // Schema of the table.  Need this for reading TEXTFILE.
  const RowTypePtr& dataColumns() const {
    return dataColumns_;
  }

  const std::unordered_map<std::string, std::string>& tableParameters() const {
    return tableParameters_;
  }

  std::string toString() const override;

  static ConnectorTableHandlePtr create(
      const folly::dynamic& obj,
      void* context);

 private:
  const std::string tableName_;
  const bool filterPushdownEnabled_;
  const core::TypedExprPtr remainingFilter_;
  const RowTypePtr dataColumns_;
  const std::unordered_map<std::string, std::string> tableParameters_;
};
