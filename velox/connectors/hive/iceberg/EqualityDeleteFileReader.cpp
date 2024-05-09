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

#include "velox/connectors/hive/iceberg/EqualityDeleteFileReader.h"

#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/iceberg/FilterUtil.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/dwio/common/TypeUtils.h"

using namespace facebook::velox::common;
using namespace facebook::velox::core;
using namespace facebook::velox::exec;

namespace facebook::velox::connector::hive::iceberg {

static constexpr const int kMaxBatchRows = 10'000;

EqualityDeleteFileReader::EqualityDeleteFileReader(
    const IcebergDeleteFile& deleteFile,
    std::shared_ptr<const dwio::common::TypeWithId> baseFileSchema,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const HiveConfig> hiveConfig,
    std::shared_ptr<io::IoStatistics> ioStats,
    dwio::common::RuntimeStatistics& runtimeStats,
    const std::string& connectorId)
    : deleteFile_(deleteFile),
      baseFileSchema_(baseFileSchema),
      connectorQueryCtx_(connectorQueryCtx),
      hiveConfig_(hiveConfig),
      deleteSplit_(nullptr),
      fileHandleFactory_(fileHandleFactory),
      executor_(executor),
      pool_(connectorQueryCtx->memoryPool()),
      ioStats_(ioStats),
      deleteRowReader_(nullptr) {
  VELOX_CHECK(deleteFile_.content == FileContent::kEqualityDeletes);

  if (deleteFile_.recordCount == 0) {
    return;
  }

  std::unordered_set<int32_t> equalityFieldIds(
      deleteFile_.equalityFieldIds.begin(), deleteFile_.equalityFieldIds.end());
  auto deleteFieldSelector = [&equalityFieldIds](size_t index) {
    return equalityFieldIds.find(static_cast<int32_t>(index)) !=
        equalityFieldIds.end();
  };
  auto deleteFileSchema = dwio::common::typeutils::buildSelectedType(
      baseFileSchema_, deleteFieldSelector);

  rowType_ = std::static_pointer_cast<const RowType>(deleteFileSchema->type());

  // TODO: push down filter if previous delete file contains this one. E.g.
  // previous equality delete file has a=1, and this file also contains
  // columns a, then a!=1 can be pushed as a filter when reading this delete
  // file.

  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  scanSpec->addAllChildFields(rowType_->asRow());

  deleteSplit_ = std::make_shared<HiveConnectorSplit>(
      connectorId,
      deleteFile_.filePath,
      deleteFile_.fileFormat,
      0,
      deleteFile_.fileSizeInBytes);

  // Create the Reader and RowReader

  dwio::common::ReaderOptions deleteReaderOpts(pool_);
  configureReaderOptions(
      deleteReaderOpts,
      hiveConfig_,
      connectorQueryCtx_->sessionProperties(),
      rowType_,
      deleteSplit_);

  auto deleteFileHandle =
      fileHandleFactory_->generate(deleteFile_.filePath).second;
  auto deleteFileInput = createBufferedInput(
      *deleteFileHandle,
      deleteReaderOpts,
      connectorQueryCtx_,
      ioStats_,
      executor_);

  auto deleteReader =
      dwio::common::getReaderFactory(deleteReaderOpts.getFileFormat())
          ->createReader(std::move(deleteFileInput), deleteReaderOpts);

  dwio::common::RowReaderOptions deleteRowReaderOpts;
  configureRowReaderOptions(
      deleteRowReaderOpts, {}, scanSpec, nullptr, rowType_, deleteSplit_);

  deleteRowReader_.reset();
  deleteRowReader_ = deleteReader->createRowReader(deleteRowReaderOpts);
}

void EqualityDeleteFileReader::readDeleteValues(
    SubfieldFilters& subfieldFilters,
    std::vector<core::TypedExprPtr>& expressionInputs) {
  VELOX_CHECK(deleteRowReader_);
  VELOX_CHECK(deleteSplit_);

  if (!deleteValuesOutput_) {
    deleteValuesOutput_ = BaseVector::create(rowType_, 0, pool_);
  }

  // TODO:: verfiy if the field is a sub-field. Velox currently doesn't support
  // pushing down filters to sub-fields
  if (rowType_->size() == 1) {
    // Construct the IN list filter that can be pushed down to the base file
    // readers, then update the baseFileScanSpec.
    readSingleColumnDeleteValues(subfieldFilters);
  } else {
    readMultipleColumnDeleteValues(expressionInputs);
  }

  deleteSplit_.reset();
}

void EqualityDeleteFileReader::readSingleColumnDeleteValues(
    SubfieldFilters& subfieldFilters) {
  std::unique_ptr<Filter> filter = std::make_unique<AlwaysTrue>();
  while (deleteRowReader_->next(kMaxBatchRows, deleteValuesOutput_)) {
    if (deleteValuesOutput_->size() == 0) {
      continue;
    }

    deleteValuesOutput_->loadedVector();
    auto vector =
        std::dynamic_pointer_cast<RowVector>(deleteValuesOutput_)->childAt(0);
    auto name = rowType_->nameOf(0);

    auto typeKind = vector->type()->kind();
    VELOX_CHECK(
        typeKind != TypeKind::DOUBLE && typeKind != TypeKind::REAL,
        "Iceberg does not allow DOUBLE or REAL columns as the equality delete columns: {} : {}",
        name,
        typeKind);

    auto notExistsFilter =
        createNotExistsFilter(vector, 0, deleteValuesOutput_->size(), typeKind);
    filter = filter->mergeWith(notExistsFilter.get());
  }

  if (filter->kind() != FilterKind::kAlwaysTrue) {
    if (subfieldFilters.find(common::Subfield(rowType_->nameOf(0))) !=
        subfieldFilters.end()) {
      subfieldFilters[common::Subfield(rowType_->nameOf(0))] =
          subfieldFilters[common::Subfield(rowType_->nameOf(0))]->mergeWith(
              filter.get());
    } else {
      subfieldFilters[common::Subfield(rowType_->nameOf(0))] =
          std::move(filter);
    }
  }
}

void EqualityDeleteFileReader::readMultipleColumnDeleteValues(
    std::vector<core::TypedExprPtr>& expressionInputs) {
  auto numDeleteFields = rowType_->size();
  VELOX_CHECK_GT(
      numDeleteFields,
      0,
      "Iceberg equality delete file should have at least one field.");

  // TODO: logical expression simplifications
  while (deleteRowReader_->next(kMaxBatchRows, deleteValuesOutput_)) {
    if (deleteValuesOutput_->size() == 0) {
      continue;
    }

    deleteValuesOutput_->loadedVector();
    auto rowVector = std::dynamic_pointer_cast<RowVector>(deleteValuesOutput_);
    auto numDeletedValues = rowVector->childAt(0)->size();

    for (int i = 0; i < numDeletedValues; i++) {
      std::vector<core::TypedExprPtr> disconjunctInputs;

      for (int j = 0; j < numDeleteFields; j++) {
        auto type = rowType_->childAt(j);
        auto name = rowType_->nameOf(j);
        auto value = BaseVector::wrapInConstant(1, i, rowVector->childAt(j));

        std::vector<core::TypedExprPtr> isNotEqualInputs;
        isNotEqualInputs.push_back(
            std::make_shared<FieldAccessTypedExpr>(type, name));
        isNotEqualInputs.push_back(std::make_shared<ConstantTypedExpr>(value));
        auto isNotEqualExpr =
            std::make_shared<CallTypedExpr>(BOOLEAN(), isNotEqualInputs, "neq");

        disconjunctInputs.push_back(isNotEqualExpr);
      }

      auto disconjunctNotEqualExpr =
          std::make_shared<CallTypedExpr>(BOOLEAN(), disconjunctInputs, "or");
      expressionInputs.push_back(disconjunctNotEqualExpr);
    }
  }
}

} // namespace facebook::velox::connector::hive::iceberg