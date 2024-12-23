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

#include "velox/optimizer/connectors/hive/HiveConnectorMetadata.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/expression/ExprToSubfieldFilter.h"
#include "velox/expression/FieldReference.h"

namespace facebook::velox::connector::hive {

namespace {
HiveColumnHandle::ColumnType columnType(
    const HiveTableLayout& layout,
    const std::string& columnName) {
  auto& columns = layout.hivePartitionColumns();
  for (auto& c : columns) {
    if (c->name() == columnName) {
      return HiveColumnHandle::ColumnType::kPartitionKey;
    }
  }
  // TODO recognize special names like $path, $bucket etc.
  return HiveColumnHandle::ColumnType::kRegular;
}
} // namespace

ColumnHandlePtr HiveConnectorMetadata::createColumnHandle(
    const TableLayout& layout,
    const std::string& columnName,
    std::vector<common::Subfield> subfields,
    std::optional<TypePtr> castToType,
    SubfieldMapping subfieldMapping) {
  // castToType and subfieldMapping are not yet supported.
  VELOX_CHECK(subfieldMapping.empty());
  VELOX_CHECK(!castToType.has_value());
  auto* hiveLayout = reinterpret_cast<const HiveTableLayout*>(&layout);
  auto* column = hiveLayout->findColumn(columnName);
  auto handle = std::make_shared<HiveColumnHandle>(
      columnName,
      columnType(*hiveLayout, columnName),
      column->type(),
      column->type(),
      std::move(subfields));
  return std::dynamic_pointer_cast<const ColumnHandle>(handle);
}

ConnectorTableHandlePtr HiveConnectorMetadata::createTableHandle(
    const TableLayout& layout,
    std::vector<ColumnHandlePtr> columnHandles,
    velox::core::ExpressionEvaluator& evaluator,
    std::vector<core::TypedExprPtr> filters,
    std::vector<core::TypedExprPtr>& rejectedFilters,
    std::optional<LookupKeys> lookupKeys) {
  VELOX_CHECK(!lookupKeys.has_value(), "Hive does not support lookup keys");
  auto* hiveLayout = dynamic_cast<const HiveTableLayout*>(&layout);

  std::vector<std::string> names;
  std::vector<TypePtr> types;
  for (auto& columnHandle : columnHandles) {
    auto* hiveColumn =
        reinterpret_cast<const HiveColumnHandle*>(columnHandle.get());
    names.push_back(hiveColumn->name());
    types.push_back(hiveColumn->dataType());
  }
  auto dataColumns = ROW(std::move(names), std::move(types));
  std::vector<core::TypedExprPtr> remainingConjuncts;
  SubfieldFilters subfieldFilters;
  for (auto& typedExpr : filters) {
    try {
      auto pair = velox::exec::toSubfieldFilter(typedExpr, &evaluator);
      if (!pair.second) {
        remainingConjuncts.push_back(std::move(typedExpr));
        continue;
      }
      subfieldFilters[std::move(pair.first)] = std::move(pair.second);
    } catch (const std::exception& e) {
      remainingConjuncts.push_back(std::move(typedExpr));
    }
  }
  core::TypedExprPtr remainingFilter;
  for (auto conjunct : remainingConjuncts) {
    if (!remainingFilter) {
      remainingFilter = conjunct;
    } else {
      remainingFilter = std::make_shared<core::CallTypedExpr>(
          BOOLEAN(),
          std::vector<core::TypedExprPtr>{remainingFilter, conjunct},
          "and");
    }
  }
  return std::dynamic_pointer_cast<const ConnectorTableHandle>(
      std::make_shared<HiveTableHandle>(
          hiveConnector_->connectorId(),
          hiveLayout->table()->name(),
          true,
          std::move(subfieldFilters),
          remainingFilter,
          std::move(dataColumns)));
}

} // namespace facebook::velox::connector::hive
