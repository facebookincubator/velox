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

#include "velox/connectors/lakehouse/iceberg/EqualityDeleteFileReader.h"

#include "velox/connectors/lakehouse/common/HiveConnectorUtil.h"
#include "velox/connectors/lakehouse/iceberg/FilterUtil.h"
#include "velox/connectors/lakehouse/iceberg/IcebergDeleteFile.h"
#include "velox/core/Expressions.h"
#include "velox/dwio/common/ReaderFactory.h"

using namespace facebook::velox::common;
using namespace facebook::velox::core;
using namespace facebook::velox::exec;

namespace facebook::velox::connector::lakehouse::iceberg {

static constexpr const int kMaxBatchRows = 10'000;

EqualityDeleteFileReader::EqualityDeleteFileReader(
    const IcebergDeleteFile& deleteFile,
    std::shared_ptr<const dwio::common::TypeWithId> baseFileSchema,
    common::FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<const common::HiveConfig>& hiveConfig,
    const std::shared_ptr<io::IoStatistics> ioStats,
    const std::shared_ptr<filesystems::File::IoStats>& fsStats,
    const std::string& connectorId)
    : deleteFile_(deleteFile),
      baseFileSchema_(baseFileSchema),
      fileHandleFactory_(fileHandleFactory),
      pool_(connectorQueryCtx->memoryPool()),
      deleteSplit_(nullptr),
      deleteRowReader_(nullptr) {
  VELOX_CHECK_EQ(deleteFile_.content, FileContent::kEqualityDeletes);

  if (deleteFile_.recordCount == 0) {
    return;
  }

  // TODO: push down filter if previous delete file contains this one. E.g.
  // previous equality delete file has a=1, and this file also contains
  // columns a, then a!=1 can be pushed as a filter when reading this delete
  // file.

  deleteSplit_ = std::make_shared<common::HiveConnectorSplit>(
      connectorId,
      deleteFile_.filePath,
      deleteFile_.fileFormat,
      0,
      deleteFile_.fileSizeInBytes);

  // Create the Reader and RowReader for the equality delete file

  dwio::common::ReaderOptions deleteReaderOpts(pool_);
  configureReaderOptions(
      hiveConfig,
      connectorQueryCtx,
      nullptr,
      deleteSplit_,
      {},
      deleteReaderOpts);

  const common::FileHandleKey fileHandleKey{
      .filename = deleteFile_.filePath,
      .tokenProvider = connectorQueryCtx->fsTokenProvider()};
  auto deleteFileHandleCachePtr = fileHandleFactory_->generate(fileHandleKey);
  auto deleteFileInput = common::createBufferedInput(
      *deleteFileHandleCachePtr,
      deleteReaderOpts,
      connectorQueryCtx,
      ioStats,
      fsStats,
      executor);

  auto deleteReader =
      dwio::common::getReaderFactory(deleteReaderOpts.fileFormat())
          ->createReader(std::move(deleteFileInput), deleteReaderOpts);

  // For now, we assume only the delete columns are written in the delete file
  deleteFileRowType_ = deleteReader->rowType();
  auto scanSpec = std::make_shared<velox::common::ScanSpec>("<root>");
  scanSpec->addAllChildFields(deleteFileRowType_->asRow());

  dwio::common::RowReaderOptions deleteRowReaderOpts;
  configureRowReaderOptions(
      {},
      scanSpec,
      nullptr,
      deleteFileRowType_,
      deleteSplit_,
      hiveConfig,
      connectorQueryCtx->sessionProperties(),
      deleteRowReaderOpts);

  deleteRowReader_.reset();
  deleteRowReader_ = deleteReader->createRowReader(deleteRowReaderOpts);
}

void EqualityDeleteFileReader::readDeleteValues(
    SubfieldFilters& subfieldFilters,
    std::vector<core::TypedExprPtr>& expressionInputs) {
  VELOX_CHECK(deleteRowReader_);
  VELOX_CHECK(deleteSplit_);

  if (!deleteValuesOutput_) {
    deleteValuesOutput_ = BaseVector::create(deleteFileRowType_, 0, pool_);
  }

  // TODO: verfiy if the field is an Iceberg RowId. Velox currently doesn't
  // support pushing down filters to non-RowId types, i.e. sub-fields of Array
  // or Map
  if (deleteFileRowType_->size() == 1) {
    // Construct the IN list filter that can be pushed down to the base file
    // readers, then update the baseFileScanSpec.
    buildDomainFilter(subfieldFilters);
  } else {
    // Build the filter functions that will be evaluated after all base file
    // read is done
    buildFilterFunctions(expressionInputs);
  }

  deleteSplit_.reset();
}

void EqualityDeleteFileReader::buildDomainFilter(
    SubfieldFilters& subfieldFilters) {
  std::unique_ptr<Filter> filter = std::make_unique<AlwaysTrue>();
  auto name = baseFileSchema_->childByFieldId(deleteFile_.equalityFieldIds[0])
                  ->fullName();
  while (deleteRowReader_->next(kMaxBatchRows, deleteValuesOutput_)) {
    if (deleteValuesOutput_->size() == 0) {
      continue;
    }

    deleteValuesOutput_->loadedVector();
    auto vector =
        std::dynamic_pointer_cast<RowVector>(deleteValuesOutput_)->childAt(0);

    auto typeKind = vector->type()->kind();
    VELOX_CHECK(
        typeKind != TypeKind::DOUBLE && typeKind != TypeKind::REAL,
        "Iceberg does not allow DOUBLE or REAL columns as the equality delete columns: {} : {}",
        name,
        typeKind);

    auto notExistsFilter =
        createNotInFilter(vector, 0, deleteValuesOutput_->size(), typeKind);
    filter = filter->mergeWith(notExistsFilter.get());
  }

  if (filter->kind() != FilterKind::kAlwaysTrue) {
    if (subfieldFilters.find(velox::common::Subfield(name)) != subfieldFilters.end()) {
      subfieldFilters[velox::common::Subfield(name)] =
          subfieldFilters[velox::common::Subfield(name)]->mergeWith(filter.get());
    } else {
      subfieldFilters[velox::common::Subfield(name)] = std::move(filter);
    }
  }
}

void EqualityDeleteFileReader::buildFilterFunctions(
    std::vector<core::TypedExprPtr>& expressionInputs) {
  auto numDeleteFields = deleteFileRowType_->size();
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
        auto type = deleteFileRowType_->childAt(j);
        auto name =
            baseFileSchema_->childByFieldId(deleteFile_.equalityFieldIds[j])
                ->fullName();
        auto value = BaseVector::wrapInConstant(1, i, rowVector->childAt(j));

        std::vector<core::TypedExprPtr> isNotEqualInputs;
        isNotEqualInputs.push_back(
            std::make_shared<FieldAccessTypedExpr>(type, name));
        isNotEqualInputs.push_back(std::make_shared<ConstantTypedExpr>(value));
        // TODO: generalize this to support different engines. Currently, only
        // Presto "neq" is supported. Spark does not register the "neq" function
        // but does support "not" and "equalto" functions.
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

} // namespace facebook::velox::connector::lakehouse::iceberg
