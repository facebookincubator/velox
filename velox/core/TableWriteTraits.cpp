/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/core/TableWriteTraits.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/ConstantVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::core {

// static
RowVectorPtr TableWriteTraits::createAggregationStatsOutput(
    RowTypePtr outputType,
    RowVectorPtr aggregationOutput,
    StringView tableCommitContext,
    velox::memory::MemoryPool* pool) {
  // TODO: record aggregation stats output time.
  if (aggregationOutput == nullptr) {
    return nullptr;
  }
  VELOX_CHECK_GT(aggregationOutput->childrenSize(), 0);
  const vector_size_t numOutputRows = aggregationOutput->childAt(0)->size();
  std::vector<VectorPtr> columns;
  for (int channel = 0; channel < outputType->size(); channel++) {
    if (channel < TableWriteTraits::kContextChannel) {
      // 1. Set null rows column.
      // 2. Set null fragments column.
      columns.push_back(
          BaseVector::createNullConstant(
              outputType->childAt(channel), numOutputRows, pool));
      continue;
    }
    if (channel == TableWriteTraits::kContextChannel) {
      // 3. Set commitcontext column.
      columns.push_back(
          std::make_shared<ConstantVector<StringView>>(
              pool,
              numOutputRows,
              false /*isNull*/,
              VARBINARY(),
              // Note that we move tableCommitContext here, so ensure this
              // branch is only executed once in the loop.
              std::move(tableCommitContext)));
      continue;
    }
    // 4. Set statistics columns.
    columns.push_back(
        aggregationOutput->childAt(channel - TableWriteTraits::kStatsChannel));
  }
  return std::make_shared<RowVector>(
      pool, outputType, nullptr, numOutputRows, columns);
}

std::string TableWriteTraits::rowCountColumnName() {
  static const std::string kRowCountName = "rows";
  return kRowCountName;
}

std::string TableWriteTraits::fragmentColumnName() {
  static const std::string kFragmentName = "fragments";
  return kFragmentName;
}

std::string TableWriteTraits::contextColumnName() {
  static const std::string kContextName = "commitcontext";
  return kContextName;
}

const TypePtr& TableWriteTraits::rowCountColumnType() {
  static const TypePtr kRowCountType = BIGINT();
  return kRowCountType;
}

const TypePtr& TableWriteTraits::fragmentColumnType() {
  static const TypePtr kFragmentType = VARBINARY();
  return kFragmentType;
}

const TypePtr& TableWriteTraits::contextColumnType() {
  static const TypePtr kContextType = VARBINARY();
  return kContextType;
}

// static.
RowTypePtr TableWriteTraits::outputType(
    const std::optional<core::ColumnStatsSpec>& columnStatsSpec) {
  static const auto kOutputTypeWithoutStats =
      ROW({rowCountColumnName(), fragmentColumnName(), contextColumnName()},
          {rowCountColumnType(), fragmentColumnType(), contextColumnType()});
  if (!columnStatsSpec.has_value()) {
    return kOutputTypeWithoutStats;
  }
  return kOutputTypeWithoutStats->unionWith(columnStatsSpec->outputType());
}

folly::dynamic TableWriteTraits::getTableCommitContext(
    const RowVectorPtr& input) {
  VELOX_CHECK_GT(input->size(), 0);
  auto* contextVector =
      input->childAt(kContextChannel)->as<SimpleVector<StringView>>();
  return folly::parseJson(
      std::string_view(contextVector->valueAt(input->size() - 1)));
}

int64_t TableWriteTraits::getRowCount(const RowVectorPtr& output) {
  VELOX_CHECK_GT(output->size(), 0);
  auto rowCountVector =
      output->childAt(kRowCountChannel)->asFlatVector<int64_t>();
  VELOX_CHECK_NOT_NULL(rowCountVector);
  int64_t rowCount{0};
  for (int i = 0; i < output->size(); ++i) {
    if (!rowCountVector->isNullAt(i)) {
      rowCount += rowCountVector->valueAt(i);
    }
  }
  return rowCount;
}

} // namespace facebook::velox::core
