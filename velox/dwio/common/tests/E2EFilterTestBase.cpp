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

#include "velox/dwio/common/tests/E2EFilterTestBase.h"

#include "velox/dwio/common/tests/utils/DataSetBuilder.h"

DEFINE_int32(timing_repeats, 0, "Count of repeats for timing filter tests");

namespace facebook::velox::dwio::common {

using namespace facebook::velox::test;
using namespace facebook::velox::dwio::type::fbhive;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox;
using namespace facebook::velox::common;

using dwio::common::MemoryInputStream;
using dwio::common::MemorySink;
using velox::common::Subfield;

std::vector<RowVectorPtr> E2EFilterTestBase::makeDataset(
    std::function<void()> customize,
    bool forRowGroupSkip) {
  if (!dataSetBuilder_) {
    dataSetBuilder_ = std::make_unique<DataSetBuilder>(*pool_, 0);
  }

  dataSetBuilder_->makeDataset(rowType_, kBatchCount, kBatchSize);

  if (forRowGroupSkip) {
    dataSetBuilder_->withRowGroupSpecificData(kRowsInGroup);
  }

  if (customize) {
    customize();
  }

  std::vector<RowVectorPtr> batches = *dataSetBuilder_->build();
  return batches;
}

void E2EFilterTestBase::makeAllNulls(const std::string& fieldName) {
  dataSetBuilder_->withAllNullsForField(Subfield(fieldName));
}

void E2EFilterTestBase::makeStringDistribution(
    const std::string& fieldName,
    int cardinality,
    bool keepNulls,
    bool addOneOffs) {
  auto field = Subfield(fieldName);
  dataSetBuilder_->withStringDistributionForField(
      field, cardinality, keepNulls, addOneOffs);
}

void E2EFilterTestBase::makeStringUnique(const std::string& fieldName) {
  dataSetBuilder_->withUniqueStringsForField(Subfield(fieldName));
}

void E2EFilterTestBase::makeNotNull(int32_t firstRow) {
  dataSetBuilder_->withNoNullsAfter(firstRow);
}

void E2EFilterTestBase::readWithoutFilter(
    std::shared_ptr<ScanSpec> spec,
    const std::vector<RowVectorPtr>& batches,
    uint64_t& time) {
  auto input = std::make_unique<MemoryInputStream>(
      sinkPtr_->getData(), sinkPtr_->size());

  dwio::common::ReaderOptions readerOpts;
  dwio::common::RowReaderOptions rowReaderOpts;
  auto reader = makeReader(readerOpts, std::move(input));

  // The spec must stay live over the lifetime of the reader.
  rowReaderOpts.setScanSpec(spec);
  OwnershipChecker ownershipChecker;
  auto rowReader = reader->createRowReader(rowReaderOpts);

  auto batchIndex = 0;
  auto rowIndex = 0;
  auto batch = BaseVector::create(rowType_, 1, pool_.get());
  while (true) {
    bool hasData;
    {
      MicrosecondTimer timer(&time);
      hasData = rowReader->next(1000, batch);
    }
    if (!hasData) {
      break;
    }

    ownershipChecker.check(batch);
    for (int32_t i = 0; i < batch->size(); ++i) {
      ASSERT_TRUE(batch->equalValueAt(batches[batchIndex].get(), i, rowIndex))
          << "Content mismatch at batch " << batchIndex << " at index "
          << rowIndex
          << ": expected: " << batches[batchIndex]->toString(rowIndex)
          << " actual: " << batch->toString(i);

      if (++rowIndex == batches[batchIndex]->size()) {
        rowIndex = 0;
        ++batchIndex;
      }
    }
  }
  ASSERT_EQ(batchIndex, batches.size());
  ASSERT_EQ(rowIndex, 0);
}

void E2EFilterTestBase::readWithFilter(
    std::shared_ptr<ScanSpec> spec,
    const std::vector<RowVectorPtr>& batches,
    const std::vector<uint32_t>& hitRows,
    uint64_t& time,
    bool useValueHook,
    bool skipCheck) {
  auto input = std::make_unique<MemoryInputStream>(
      sinkPtr_->getData(), sinkPtr_->size());

  dwio::common::ReaderOptions readerOpts;
  dwio::common::RowReaderOptions rowReaderOpts;
  auto reader = makeReader(readerOpts, std::move(input));
  // The  spec must stay live over the lifetime of the reader.
  rowReaderOpts.setScanSpec(spec);
  OwnershipChecker ownershipChecker;
  auto rowReader = reader->createRowReader(rowReaderOpts);
  runtimeStats_ = dwio::common::RuntimeStatistics();
  auto rowIndex = 0;
  auto batch = BaseVector::create(rowType_, 1, pool_.get());
  resetReadBatchSizes();
  int32_t clearCnt = 0;
  while (true) {
    {
      MicrosecondTimer timer(&time);
      if (++clearCnt % 17 == 0) {
        rowReader->resetFilterCaches();
      }
      bool hasData = rowReader->next(nextReadBatchSize(), batch);
      if (!hasData) {
        break;
      }
      if (batch->size() == 0) {
        // No hits in the last batch of rows.
        continue;
      }
      if (useValueHook) {
        auto rowVector = reinterpret_cast<RowVector*>(batch.get());
        for (int32_t i = 0; i < rowVector->childrenSize(); ++i) {
          auto child = rowVector->childAt(i);
          if (child->encoding() == VectorEncoding::Simple::LAZY &&
              typeKindSupportsValueHook(child->typeKind())) {
            ASSERT_TRUE(
                loadWithHook(batches, rowVector, i, child, hitRows, rowIndex));
          }
        }
        rowIndex += batch->size();
        continue;
      }
      // Load eventual LazyVectors inside the timed section.
      auto rowVector = batch->asUnchecked<RowVector>();
      for (auto i = 0; i < rowVector->childrenSize(); ++i) {
        rowVector->childAt(i)->loadedVector();
      }
      if (skipCheck) {
        // Fetch next batch inside timed section.
        continue;
      }
    }
    // Outside of timed section.
    for (int32_t i = 0; i < batch->size(); ++i) {
      uint32_t hit = hitRows[rowIndex++];
      ASSERT_TRUE(batch->equalValueAt(
          batches[batchNumber(hit)].get(), i, batchRow(hit)))
          << "Content mismatch at " << rowIndex - 1 << ": expected: "
          << batches[batchNumber(hit)]->toString(batchRow(hit))
          << " actual: " << batch->toString(i);
    }
    // Check no overwrites after all LazyVectors are loaded.
    ownershipChecker.check(batch);
  }
  if (!skipCheck) {
    ASSERT_EQ(rowIndex, hitRows.size());
  }
  rowReader->updateRuntimeStats(runtimeStats_);
}

bool E2EFilterTestBase::loadWithHook(
    const std::vector<RowVectorPtr>& batches,
    RowVector* batch,
    int32_t columnIndex,
    VectorPtr child,
    const std::vector<uint32_t>& hitRows,
    int32_t rowIndex) {
  auto kind = child->typeKind();
  if (kind == TypeKind::ROW || kind == TypeKind::ARRAY ||
      kind == TypeKind::MAP) {
    return true;
  }
  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      checkLoadWithHook,
      kind,
      batches,
      batch,
      columnIndex,
      child,
      hitRows,
      rowIndex);
}

void E2EFilterTestBase::testFilterSpecs(
    const std::vector<RowVectorPtr>& batches,
    const std::vector<FilterSpec>& filterSpecs) {
  std::vector<uint32_t> hitRows;
  auto filters =
      filterGenerator_->makeSubfieldFilters(filterSpecs, batches, hitRows);
  auto spec = filterGenerator_->makeScanSpec(std::move(filters));
  uint64_t timeWithFilter = 0;
  readWithFilter(spec, batches, hitRows, timeWithFilter, false);

  if (FLAGS_timing_repeats) {
    for (auto i = 0; i < FLAGS_timing_repeats; ++i) {
      readWithFilter(spec, batches, hitRows, timeWithFilter, false, true);
    }
  }
  // Redo the test with LazyVectors for non-filtered columns.
  timeWithFilter = 0;
  for (auto& childSpec : spec->children()) {
    childSpec->setExtractValues(false);
  }
  readWithFilter(spec, batches, hitRows, timeWithFilter, false);
  timeWithFilter = 0;
  readWithFilter(spec, batches, hitRows, timeWithFilter, true);
}

void E2EFilterTestBase::testNoRowGroupSkip(
    const std::vector<RowVectorPtr>& batches,
    const std::vector<std::string>& filterable,
    int32_t numCombinations) {
  // TODO: Seed was hard coded as 1 to make it behave the same as before.
  // Change to use random seed (like current timestamp).
  filterGenerator_ = std::make_unique<FilterGenerator>(rowType_, 1);
  auto spec = filterGenerator_->makeScanSpec(SubfieldFilters{});

  uint64_t timeWithNoFilter = 0;
  readWithoutFilter(spec, batches, timeWithNoFilter);

  for (auto i = 0; i < numCombinations; ++i) {
    std::vector<FilterSpec> specs =
        filterGenerator_->makeRandomSpecs(filterable, 125);
    testFilterSpecs(batches, specs);
  }
}

void E2EFilterTestBase::testRowGroupSkip(
    const std::vector<RowVectorPtr>& batches,
    const std::vector<std::string>& filterable) {
  std::vector<FilterSpec> specs;
  // Makes a row group skipping filter for the first bigint column.
  for (auto& field : filterable) {
    VectorPtr child = getChildBySubfield(batches[0].get(), Subfield(field));
    if (child->typeKind() == TypeKind::BIGINT ||
        child->typeKind() == TypeKind::VARCHAR) {
      specs.emplace_back();
      specs.back().field = field;
      specs.back().isForRowGroupSkip = true;
      break;
    }
  }
  if (specs.empty()) {
    // No suitable column.
    return;
  }

  testFilterSpecs(batches, specs);
  EXPECT_LT(0, runtimeStats_.skippedStrides);
}

void E2EFilterTestBase::testSenario(
    const std::string& columns,
    std::function<void()> customize,
    bool wrapInStruct,
    const std::vector<std::string>& filterable,
    int32_t numCombinations) {
  rowType_ = DataSetBuilder::makeRowType(columns, wrapInStruct);

  auto batches = makeDataset(customize, false);
  writeToMemory(rowType_, batches, false);
  testNoRowGroupSkip(batches, filterable, numCombinations);

  batches = makeDataset(customize, true);
  writeToMemory(rowType_, batches, true);
  testRowGroupSkip(batches, filterable);
}

void OwnershipChecker::check(const VectorPtr& batch) {
  // Check the 6 first pairs of previous, next batch to see that
  // fetching the next does not overwrite parts reachable from a
  // retained reference to the previous one.
  if (batchCounter_ > 11) {
    return;
  }
  if (batchCounter_ % 2 == 0) {
    previousBatch_ = std::make_shared<RowVector>(
        batch->pool(),
        batch->type(),
        BufferPtr(nullptr),
        batch->size(),
        batch->as<RowVector>()->children());
    previousBatchCopy_ = BaseVector::copy(*batch);
  }
  if (batchCounter_ % 2 == 1) {
    for (auto i = 0; i < previousBatch_->size(); ++i) {
      ASSERT_TRUE(previousBatch_->equalValueAt(previousBatchCopy_.get(), i, i))
          << "Retained reference of a batch has been overwritten by the next "
          << "index " << i << " batch " << previousBatch_->toString(i)
          << " original " << previousBatchCopy_->toString(i);
    }
  }
  ++batchCounter_;
}

} // namespace facebook::velox::dwio::common
