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
#include "velox/dwio/nimble/reader/NimbleReaderFactory.h"

#include <utility>

#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/reader/FeatureSelection.h"
#include "velox/dwio/nimble/reader/FieldIdResolver.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"

namespace facebook::velox::nimble {

namespace {

// Rebuilds a struct vector to match the field order and selection described by
// 'spec' (its projected children, in channel order, matched by field name).
// Reorders and drops fields relative to the on-disk file struct, fills
// constants for fields the connector marked constant, and recurses into nested
// structs. Needed for Iceberg field-id schema evolution, where a struct's
// fields may be reordered or dropped relative to the file; the shared
// RowReader::projectColumns only reshapes top-level columns.
VectorPtr projectStructToSpec(
    const VectorPtr& input,
    const common::ScanSpec& spec,
    memory::MemoryPool* pool) {
  auto loaded = BaseVector::loadedVectorShared(input);
  auto* inputRow = loaded->as<RowVector>();
  if (inputRow == nullptr) {
    return loaded;
  }
  const auto& inputType = loaded->type()->asRow();

  column_index_t numColumns = 0;
  for (const auto& childSpec : spec.children()) {
    if (childSpec->projectOut()) {
      numColumns =
          std::max<column_index_t>(numColumns, childSpec->channel() + 1);
    }
  }

  std::vector<std::string> names(numColumns);
  std::vector<TypePtr> types(numColumns);
  std::vector<VectorPtr> children(numColumns);
  for (const auto& childSpec : spec.children()) {
    if (!childSpec->projectOut()) {
      continue;
    }
    const auto i = childSpec->channel();
    VectorPtr child;
    if (childSpec->isConstant()) {
      child = BaseVector::wrapInConstant(
          loaded->size(), 0, childSpec->constantValue());
    } else {
      const auto idx = inputType.getChildIdxIfExists(childSpec->fieldName());
      VELOX_CHECK(
          idx.has_value(),
          "NIMBLE Iceberg field-id projection: struct field '{}' is absent "
          "from the file and is not a constant; nested added columns are not "
          "supported.",
          childSpec->fieldName());
      child = inputRow->childAt(*idx);
      if (!childSpec->isFlatMapAsStruct() && !childSpec->children().empty()) {
        child = projectStructToSpec(child, *childSpec, pool);
      }
    }
    names[i] = childSpec->fieldName();
    types[i] = child->type();
    children[i] = std::move(child);
  }
  return std::make_shared<RowVector>(
      pool,
      ROW(std::move(names), std::move(types)),
      loaded->nulls(),
      loaded->size(),
      std::move(children));
}

// Reshapes only the struct-typed top-level columns of 'row' to their requested
// nested shape (see projectStructToSpec), leaving every top-level column in
// place. Top-level reordering/selection is handled afterwards by
// RowReader::projectColumns; this pass exists solely to fix nested struct field
// order/selection under Iceberg field-id schema evolution. Returns 'row'
// unchanged when no nested struct needed reshaping.
VectorPtr reshapeNestedStructs(
    const VectorPtr& row,
    const common::ScanSpec& spec,
    memory::MemoryPool* pool) {
  auto* inputRow = row->as<RowVector>();
  if (inputRow == nullptr) {
    return row;
  }
  const auto& rowType = row->type()->asRow();
  auto children = inputRow->children();
  auto types = rowType.children();
  bool changed = false;
  for (const auto& childSpec : spec.children()) {
    if (!childSpec->projectOut() || childSpec->isConstant() ||
        childSpec->isFlatMapAsStruct() || childSpec->children().empty()) {
      continue;
    }
    const auto idx = rowType.getChildIdxIfExists(childSpec->fieldName());
    if (!idx.has_value() || children[*idx] == nullptr) {
      continue;
    }
    auto loadedChild = BaseVector::loadedVectorShared(children[*idx]);
    if (loadedChild->type()->kind() != TypeKind::ROW) {
      continue;
    }
    auto reshaped = projectStructToSpec(loadedChild, *childSpec, pool);
    if (reshaped.get() != children[*idx].get()) {
      children[*idx] = std::move(reshaped);
      types[*idx] = children[*idx]->type();
      changed = true;
    }
  }
  if (!changed) {
    return row;
  }
  const auto& nameRefs = rowType.names();
  std::vector<std::string> names(nameRefs.begin(), nameRefs.end());
  return std::make_shared<RowVector>(
      pool,
      ROW(std::move(names), std::move(types)),
      row->nulls(),
      row->size(),
      std::move(children));
}

class NimbleRowReader : public dwio::common::RowReader {
 public:
  NimbleRowReader(
      std::unique_ptr<facebook::nimble::BatchReader> reader,
      const std::shared_ptr<common::ScanSpec>& scanSpec,
      bool projectNestedStructs)
      : reader_(std::move(reader)),
        scanSpec_(scanSpec),
        projectNestedStructs_(projectNestedStructs) {
    reader_->loadStripeIfAny();
  }

  int64_t nextRowNumber() override {
    // File-relative row number of the next row to be read, or kAtEnd. The
    // underlying BatchReader reads the whole file (one split per NIMBLE file),
    // so the readable range ends at tabletRowCount(). getRowNumber() returns
    // the end row number once exhausted, so compare against the total.
    const auto current = reader_->getRowNumber();
    if (current >= reader_->tabletReader().tabletRowCount()) {
      return dwio::common::RowReader::kAtEnd;
    }
    return static_cast<int64_t>(current);
  }

  int64_t nextReadSize(uint64_t size) override {
    VELOX_DCHECK_GT(size, 0);
    const auto rowNumber = nextRowNumber();
    if (rowNumber == dwio::common::RowReader::kAtEnd) {
      return dwio::common::RowReader::kAtEnd;
    }
    // BatchReader::next() never crosses a stripe boundary in a single call, so
    // cap the requested size at the rows left in the current stripe to keep the
    // prediction consistent with what next() will actually read.
    const auto& tablet = reader_->tabletReader();
    const auto stripe = tablet.rowToStripe(rowNumber);
    const auto stripeEndRow =
        tablet.stripeStartRow(stripe) + tablet.stripeRowCount(stripe);
    const auto rowsRemainingInStripe =
        static_cast<int64_t>(stripeEndRow) - rowNumber;
    return std::min<int64_t>(size, rowsRemainingInStripe);
  }

  uint64_t next(
      uint64_t size,
      VectorPtr& result,
      const dwio::common::Mutation* mutation) override {
    TypePtr resultType;
    VectorPtr rawResult;
    if (result) {
      resultType = result->type();
      rawResult = std::move(rawVectorForBatchReader(*result));
      result.reset();
    }
    if (!reader_->next(size, rawResult)) {
      if (rawResult) {
        result = BaseVector::create(resultType, 0, &reader_->memoryPool());
        rawVectorForBatchReader(*result) = std::move(rawResult);
      }
      return 0;
    }
    auto scanned = rawResult->size();
    if (projectNestedStructs_) {
      result = projectColumns(
          reshapeNestedStructs(rawResult, *scanSpec_, &reader_->memoryPool()),
          *scanSpec_,
          mutation);
    } else {
      result = projectColumns(rawResult, *scanSpec_, mutation);
    }
    rawVectorForBatchReader(*result) = std::move(rawResult);
    return scanned;
  }

  void updateRuntimeStats(
      dwio::common::RuntimeStats& /*stats*/) const override {
    // No-op for non-selective reader.
  }

  void resetFilterCaches() override {
    // No-op for non-selective reader.
  }

  std::optional<size_t> estimatedRowSize() const override {
    return std::optional(reader_->estimatedRowSize());
  }

 private:
  std::unique_ptr<facebook::nimble::BatchReader> reader_;
  std::shared_ptr<common::ScanSpec> scanSpec_;
  // When true, reshape nested struct columns to the requested field order /
  // selection (Iceberg field-id schema evolution). Off for normal NIMBLE reads
  // to preserve existing behavior.
  bool projectNestedStructs_;

  static VectorPtr& rawVectorForBatchReader(BaseVector& vector) {
    auto* rowVector = vector.as<RowVector>();
    VELOX_CHECK_NOT_NULL(rowVector);
    return rowVector->rawVectorForBatchReader();
  }
};

class NimbleReader : public dwio::common::Reader {
 public:
  NimbleReader(
      const dwio::common::ReaderOptions& options,
      const std::shared_ptr<ReadFile>& readFile,
      const FileIoContext& fileIoContext)
      : options_(options), readFile_(readFile), tabletReader_([&]() {
          auto tabletOptions =
              facebook::nimble::TabletReader::configureOptions(options);
          tabletOptions.fileIoContext = fileIoContext;
          return facebook::nimble::TabletReader::create(
              readFile_, &options.memoryPool(), tabletOptions);
        }()) {
    // Field-id column resolution: when the caller requests
    // `kFieldId` mode, walk the file's NIMBLE Type tree in
    // lockstep with the requested file schema + per-column field-id
    // tree, then rewrite `options_.fileSchema()` to use the file's
    // column names at every nested-Row level. The downstream
    // name-based selector path then resolves each requested column to
    // its file column even after column renames at top-level, struct
    // fields, array element fields, and map key/value fields. See
    // `FieldIdResolver.h` for the recursion contract.
    //
    // This walk only fires when the table's per-column field-id tree
    // was set on ReaderOptions and a field-id attribute key is provided;
    // otherwise it is a no-op and the existing position/name-based
    // behavior is preserved.
    if (options_.columnMappingMode() ==
        dwio::common::ColumnMappingMode::kFieldId) {
      facebook::nimble::BatchReader tmpReader(
          tabletReader_, options.memoryPool());
      if (!options_.fileSchema()) {
        options_.setFileSchema(tmpReader.type());
      }
      const auto& fileSchemaType = tmpReader.schema();
      const auto& fieldIdAttributeKey = options_.fieldIdAttributeKey();
      if (fileSchemaType != nullptr && !options_.fieldIds().empty() &&
          !fieldIdAttributeKey.empty()) {
        auto resolution = resolveSchema(
            options_.fileSchema(),
            options_.fieldIds(),
            *fileSchemaType,
            fieldIdAttributeKey);
        if (resolution.has_value()) {
          tableNamedSchema_ = resolution->tableNamedSchema;
          options_.setFileSchema(resolution->fileNamedSchema);
        }
      }
    } else if (!options_.fileSchema()) {
      facebook::nimble::BatchReader tmpReader(
          tabletReader_, options.memoryPool());
      options_.setFileSchema(tmpReader.type());
    }
  }

  std::optional<uint64_t> numberOfRows() const override {
    return tabletReader_->tabletRowCount();
  }

  std::unique_ptr<dwio::common::ColumnStatistics> columnStatistics(
      uint32_t /*index*/) const override {
    // TODO: Implement using ReaderBase::fileColumnStats(), same as
    // SelectiveNimbleReader::columnStatistics().
    return nullptr;
  }

  const RowTypePtr& rowType() const override {
    // When field-id resolution ran, the exposed (table-named) schema carries
    // the requested column names (file columns renamed by matched field id),
    // while options_.fileSchema() holds the on-disk (file-named) view used for
    // the physical read. The caller does by-name matching / null-fill against
    // rowType(), so it must see the table-named schema.
    return tableNamedSchema_ != nullptr ? tableNamedSchema_
                                        : options_.fileSchema();
  }

  const std::shared_ptr<const dwio::common::TypeWithId>& typeWithId()
      const override {
    if (!typeWithId_) {
      typeWithId_ = dwio::common::TypeWithId::create(rowType());
    }
    return typeWithId_;
  }

  std::unique_ptr<dwio::common::RowReader> createRowReader(
      const dwio::common::RowReaderOptions& options) const override {
    facebook::nimble::BatchReadParams params;
    params.fileRangeStartOffset = options.offset();
    params.fileRangeEndOffset = options.limit();
    params.decodingExecutor = options.decodingExecutor();
    params.encodingFactory =
        [](velox::memory::MemoryPool& pool,
           std::string_view data,
           std::function<void*(uint32_t)> stringBufferFactory)
        -> std::unique_ptr<facebook::nimble::Encoding> {
      const facebook::nimble::EncodingFactory factory;
      return factory.create(pool, data, std::move(stringBufferFactory));
    };

    auto selector = options.selector();
    if (!selector) {
      selector = dwio::common::ColumnSelector::fromScanSpec(
          *options.scanSpec(), rowType());
    }
    facebook::nimble::populateFeatureSelector(
        *selector, options.mapColumnIdAsStruct(), params);
    auto reader = std::make_unique<facebook::nimble::BatchReader>(
        tabletReader_,
        options_.memoryPool(),
        std::move(selector),
        std::move(params));
    return std::make_unique<NimbleRowReader>(
        std::move(reader),
        options.scanSpec(),
        /*projectNestedStructs=*/tableNamedSchema_ != nullptr);
  }

 private:
  dwio::common::ReaderOptions options_;
  std::shared_ptr<ReadFile> readFile_;
  std::shared_ptr<facebook::nimble::TabletReader> tabletReader_;
  mutable std::shared_ptr<const dwio::common::TypeWithId> typeWithId_;
  // Set when field-id resolution renamed file columns to requested (table)
  // names; gates nested-struct projection in createRowReader().
  RowTypePtr tableNamedSchema_;
};

} // namespace

namespace detail {

std::unique_ptr<dwio::common::Reader> createBatchReader(
    const dwio::common::ReaderOptions& options,
    const std::shared_ptr<ReadFile>& readFile,
    const FileIoContext& fileIoContext) {
  return std::make_unique<NimbleReader>(options, readFile, fileIoContext);
}

} // namespace detail

} // namespace facebook::velox::nimble
