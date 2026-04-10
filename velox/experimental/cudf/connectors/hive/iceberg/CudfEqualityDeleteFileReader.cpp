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

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfEqualityDeleteFileReader.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergDeletionHelpers.h"
#include "velox/experimental/cudf/expression/AstUtils.h"

#include "velox/common/base/Exceptions.h"
#include "velox/connectors/hive/BufferedInputBuilder.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/iceberg/IcebergMetadataColumns.h"
#include "velox/dwio/common/ReaderFactory.h"

#include <cudf/ast/ast_operator.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/transform.hpp>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace {

/// Creates a cudf scalar and AST literal from a Velox vector element.
/// Reuses makeScalarFromValue from AstUtils.h which handles stream
/// synchronization and all supported Velox types.
template <TypeKind kind>
cudf::ast::literal makeScalarAndLiteralFromVector(
    const VectorPtr& vector,
    vector_size_t rowIndex,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars) {
  using T = typename TypeTraits<kind>::NativeType;
  const auto& type = vector->type();
  bool isNull = vector->isNullAt(rowIndex);

  if constexpr (
      cudf::is_fixed_width<T>() || std::is_same_v<T, StringView> ||
      std::is_same_v<T, std::string_view>) {
    T value{};
    if (!isNull) {
      value = vector->as<SimpleVector<T>>()->valueAt(rowIndex);
    }
    auto scalar = cudf_velox::makeScalarFromValue(type, value, isNull);
    scalars.push_back(std::move(scalar));
    return cudf_velox::makeLiteralFromScalar<T>(*scalars.back(), type);
  }
  VELOX_NYI(
      "Equality delete literal not implemented for type: {}", type->toString());
}

} // namespace

CudfEqualityDeleteFileReader::CudfEqualityDeleteFileReader(
    const velox_iceberg::IcebergDeleteFile& deleteFile,
    const std::vector<std::string>& equalityColumnNames,
    const std::vector<TypePtr>& equalityColumnTypes,
    const std::string& /*baseFilePath*/,
    FileHandleFactory* fileHandleFactory,
    const ConnectorQueryCtx* connectorQueryCtx,
    folly::Executor* executor,
    const std::shared_ptr<const velox_hive::HiveConfig>& hiveConfig,
    const std::shared_ptr<io::IoStatistics>& ioStatistics,
    const std::shared_ptr<IoStats>& ioStats,
    dwio::common::RuntimeStatistics& runtimeStats,
    const std::string& connectorId)
    : equalityColumnNames_(equalityColumnNames),
      equalityColumnTypes_(equalityColumnTypes),
      pool_(connectorQueryCtx->memoryPool()) {
  VELOX_CHECK(
      deleteFile.content == velox_iceberg::FileContent::kEqualityDeletes,
      "Expected equality delete file but got content type: {}",
      static_cast<int>(deleteFile.content));
  VELOX_CHECK_GT(deleteFile.recordCount, 0, "Empty equality delete file.");
  VELOX_CHECK(
      !equalityColumnNames_.empty(),
      "Equality delete file must specify at least one column.");
  VELOX_CHECK_EQ(
      equalityColumnNames_.size(),
      equalityColumnTypes_.size(),
      "Equality column names and types must have the same size.");

  auto deleteFileSchema =
      ROW(std::vector<std::string>(equalityColumnNames_),
          std::vector<TypePtr>(equalityColumnTypes_));

  auto scanSpec = std::make_shared<common::ScanSpec>("<root>");
  for (size_t i = 0; i < equalityColumnNames_.size(); ++i) {
    scanSpec->addField(equalityColumnNames_[i], static_cast<int>(i));
  }

  auto deleteSplit = std::make_shared<velox_hive::HiveConnectorSplit>(
      connectorId,
      deleteFile.filePath,
      deleteFile.fileFormat,
      0,
      deleteFile.fileSizeInBytes);

  dwio::common::ReaderOptions deleteReaderOpts(pool_);
  velox_hive::configureReaderOptions(
      hiveConfig,
      connectorQueryCtx,
      deleteFileSchema,
      deleteSplit,
      /*tableParameters=*/{},
      deleteReaderOpts);

  const FileHandleKey fileHandleKey{
      .filename = deleteFile.filePath,
      .tokenProvider = connectorQueryCtx->fsTokenProvider()};
  auto deleteFileHandleCachePtr = fileHandleFactory->generate(fileHandleKey);
  auto deleteFileInput =
      velox_hive::BufferedInputBuilder::getInstance()->create(
          *deleteFileHandleCachePtr,
          deleteReaderOpts,
          connectorQueryCtx,
          ioStatistics,
          ioStats,
          executor);

  auto deleteReader =
      dwio::common::getReaderFactory(deleteReaderOpts.fileFormat())
          ->createReader(std::move(deleteFileInput), deleteReaderOpts);

  if (!velox_hive::testFilters(
          scanSpec.get(),
          deleteReader.get(),
          deleteSplit->filePath,
          deleteSplit->partitionKeys,
          {},
          hiveConfig->readTimestampPartitionValueAsLocalTime(
              connectorQueryCtx->sessionProperties()))) {
    runtimeStats.skippedSplitBytes += static_cast<int64_t>(deleteSplit->length);
    return;
  }

  dwio::common::RowReaderOptions deleteRowReaderOpts;
  velox_hive::configureRowReaderOptions(
      {},
      scanSpec,
      nullptr,
      deleteFileSchema,
      deleteSplit,
      nullptr,
      nullptr,
      nullptr,
      deleteRowReaderOpts);

  auto deleteRowReader = deleteReader->createRowReader(deleteRowReaderOpts);

  VectorPtr output;
  output = BaseVector::create(deleteFileSchema, 0, pool_);

  while (true) {
    auto rowsRead = deleteRowReader->next(
        std::max(static_cast<uint64_t>(1'000), deleteFile.recordCount), output);
    if (rowsRead == 0) {
      break;
    }

    auto numRows = output->size();
    if (numRows == 0) {
      continue;
    }

    output->loadedVector();
    auto rowOutput = std::dynamic_pointer_cast<RowVector>(output);
    VELOX_CHECK_NOT_NULL(rowOutput);

    numDeleteKeys_ += static_cast<size_t>(numRows);
    deleteRows_.push_back(rowOutput);

    output = BaseVector::create(deleteFileSchema, 0, pool_);
  }
}

void CudfEqualityDeleteFileReader::applyDeletes(
    cudf::table_view table,
    const std::vector<std::string>& inputColumnNames,
    std::shared_ptr<rmm::device_buffer> rowMask,
    rmm::cuda_stream_view stream) {
  if (empty() || table.num_rows() == 0) {
    return;
  }

  if (!astTree_) {
    buildAstTree(inputColumnNames);
  }

  auto deleteColumn = cudf::compute_column(table, astTree_->back(), stream);

  applyEqualityDeleteColumn(
      deleteColumn->view(), rowMask, table.num_rows(), stream);
}

size_t CudfEqualityDeleteFileReader::numDeleteKeys() const {
  return numDeleteKeys_;
}

bool CudfEqualityDeleteFileReader::empty() const {
  return numDeleteKeys_ == 0;
}

void CudfEqualityDeleteFileReader::buildAstTree(
    const std::vector<std::string>& inputColumnNames) {
  astTree_ = std::make_unique<cudf::ast::tree>();
  ownedScalars_.clear();

  // Map equality column names to indices in the input table.
  std::vector<cudf::size_type> inputColumnIndices;
  inputColumnIndices.reserve(equalityColumnNames_.size());
  for (const auto& colName : equalityColumnNames_) {
    bool found = false;
    for (size_t i = 0; i < inputColumnNames.size(); ++i) {
      if (inputColumnNames[i] == colName) {
        inputColumnIndices.push_back(static_cast<cudf::size_type>(i));
        found = true;
        break;
      }
    }
    VELOX_CHECK(
        found,
        "Equality delete column '{}' not found in input table columns.",
        colName);
  }

  // Resolve column indices within each delete row batch. All batches share
  // the same schema, so resolve once from the first batch.
  std::vector<column_index_t> deleteColumnIndices;
  {
    const auto& rowType = deleteRows_[0]->type()->asRow();
    for (const auto& colName : equalityColumnNames_) {
      auto idx = rowType.getChildIdx(colName);
      deleteColumnIndices.push_back(static_cast<column_index_t>(idx));
    }
  }

  // Build one predicate per delete row, then OR them all together.
  const cudf::ast::expression* rootExpr = nullptr;

  for (const auto& batch : deleteRows_) {
    for (vector_size_t row = 0; row < batch->size(); ++row) {
      // Build AND chain for this delete row: (col0 NULL_EQUAL v0) AND ...
      const cudf::ast::expression* rowExpr = nullptr;

      for (size_t c = 0; c < equalityColumnNames_.size(); ++c) {
        auto& colRef = astTree_->emplace<cudf::ast::column_reference>(
            inputColumnIndices[c]);

        auto lit = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            makeScalarAndLiteralFromVector,
            equalityColumnTypes_[c]->kind(),
            batch->childAt(deleteColumnIndices[c]),
            row,
            ownedScalars_);
        auto& litRef = astTree_->push(std::move(lit));

        auto& cmpExpr = astTree_->emplace<cudf::ast::operation>(
            cudf::ast::ast_operator::NULL_EQUAL, colRef, litRef);

        if (rowExpr == nullptr) {
          rowExpr = &cmpExpr;
        } else {
          rowExpr = &astTree_->emplace<cudf::ast::operation>(
              cudf::ast::ast_operator::LOGICAL_AND, *rowExpr, cmpExpr);
        }
      }

      VELOX_CHECK_NOT_NULL(rowExpr);

      if (rootExpr == nullptr) {
        rootExpr = rowExpr;
      } else {
        rootExpr = &astTree_->emplace<cudf::ast::operation>(
            cudf::ast::ast_operator::LOGICAL_OR, *rootExpr, *rowExpr);
      }
    }
  }

  VELOX_CHECK_NOT_NULL(
      rootExpr, "Expected at least one delete row to build AST tree.");
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
