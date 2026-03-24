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

#include "velox/exec/TableWriteMerge.h"

#include "velox/exec/OperatorType.h"
#include "velox/exec/TableWriter.h"

namespace facebook::velox::exec {

TableWriteMerge::TableWriteMerge(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::TableWriteMergeNode>& tableWriteMergeNode)
    : Operator(
          driverCtx,
          tableWriteMergeNode->outputType(),
          operatorId,
          tableWriteMergeNode->id(),
          OperatorType::kTableWriteMerge) {
  if (tableWriteMergeNode->outputType()->size() == 1) {
    VELOX_USER_CHECK(!tableWriteMergeNode->hasColumnStatsSpec());
  } else {
    VELOX_USER_CHECK(tableWriteMergeNode->outputType()->equivalent(*(
        TableWriteTraits::outputType(tableWriteMergeNode->columnStatsSpec()))));
  }
  if (tableWriteMergeNode->hasColumnStatsSpec()) {
    statsCollector_ = std::make_unique<ColumnStatsCollector>(
        tableWriteMergeNode->columnStatsSpec().value(),
        tableWriteMergeNode->sources()[0]->outputType(),
        &operatorCtx_->driverCtx()->queryConfig(),
        operatorCtx_->pool(),
        &nonReclaimableSection_);
  }
}

void TableWriteMerge::initialize() {
  Operator::initialize();
  if (statsCollector_ != nullptr) {
    statsCollector_->initialize();
  }
}

namespace {
// Creates a RowVector containing only the rows at the given indices from
// 'input'. Each child vector is wrapped in a dictionary to avoid copying.
RowVectorPtr
selectRows(const RowVectorPtr& input, BufferPtr indices, vector_size_t size) {
  std::vector<VectorPtr> children(input->childrenSize());
  for (auto i = 0; i < input->childrenSize(); ++i) {
    children[i] =
        BaseVector::wrapInDictionary(nullptr, indices, size, input->childAt(i));
  }
  return std::make_shared<RowVector>(
      input->pool(), input->type(), nullptr, size, std::move(children));
}
} // namespace

void TableWriteMerge::addInput(RowVectorPtr input) {
  VELOX_CHECK(!noMoreInput_);
  VELOX_CHECK_GT(input->size(), 0);

  // Possibly mixed batch: split into stats and data using dictionary wrapping.
  auto statsIndices = allocateIndices(input->size(), pool());
  auto dataIndices = allocateIndices(input->size(), pool());
  auto* rawStatsIndices = statsIndices->asMutable<vector_size_t>();
  auto* rawDataIndices = dataIndices->asMutable<vector_size_t>();
  vector_size_t numStats{0};
  vector_size_t numData{0};
  for (vector_size_t i = 0; i < input->size(); ++i) {
    if (TableWriteTraits::isStatisticsRow(input, i)) {
      rawStatsIndices[numStats++] = i;
    } else {
      rawDataIndices[numData++] = i;
    }
  }

  if (numStats > 0) {
    if (numData == 0) {
      addStatisticsInput(input);
    } else {
      addStatisticsInput(selectRows(input, statsIndices, numStats));
    }
  }

  if (numData > 0) {
    if (numStats == 0) {
      addDataInput(input);
    } else {
      addDataInput(selectRows(input, dataIndices, numData));
    }
  }
}

void TableWriteMerge::addStatisticsInput(const RowVectorPtr& input) {
  VELOX_CHECK_NOT_NULL(statsCollector_);
  statsCollector_->addInput(input);
}

void TableWriteMerge::addDataInput(const RowVectorPtr& input) {
  numRows_ += TableWriteTraits::getRowCount(input);

  // Validate commit strategy consistency. TaskId may differ in cross-worker
  // merge (coordinator merging output from multiple workers).
  auto commitContext = TableWriteTraits::getTableCommitContext(input);
  if (lastCommitContext_ != nullptr) {
    VELOX_CHECK_EQ(
        lastCommitContext_[TableWriteTraits::kCommitStrategyContextKey]
            .asString(),
        commitContext[TableWriteTraits::kCommitStrategyContextKey].asString(),
        "Mismatched commit strategy in commit context");
  }
  lastCommitContext_ = commitContext;

  // Buffer non-null fragments for early emission to free memory. The input
  // may contain a mix of fragment rows (non-null) and summary rows (null
  // fragment) when the TableWriter produces them in a single multi-row
  // output.
  auto fragmentVector = input->childAt(TableWriteTraits::kFragmentChannel);
  if (!fragmentVector->mayHaveNulls()) {
    fragmentVectors_.push(fragmentVector);
  } else {
    auto indices = allocateIndices(fragmentVector->size(), pool());
    auto* rawIndices = indices->asMutable<vector_size_t>();
    vector_size_t numNonNull{0};
    for (vector_size_t i = 0; i < fragmentVector->size(); ++i) {
      if (!fragmentVector->isNullAt(i)) {
        rawIndices[numNonNull++] = i;
      }
    }
    if (numNonNull > 0) {
      fragmentVectors_.push(
          BaseVector::wrapInDictionary(
              nullptr, indices, numNonNull, fragmentVector));
    }
  }
}

void TableWriteMerge::noMoreInput() {
  Operator::noMoreInput();
  if (statsCollector_ != nullptr) {
    statsCollector_->noMoreInput();
  }
}

void TableWriteMerge::close() {
  if (statsCollector_ != nullptr) {
    statsCollector_->close();
  }
  Operator::close();
}

RowVectorPtr TableWriteMerge::getOutput() {
  // Passes through fragment pages first to avoid using extra memory.
  if (!fragmentVectors_.empty()) {
    return createFragmentsOutput();
  }

  if (!noMoreInput_ || finished_) {
    return nullptr;
  }

  if (statsCollector_ != nullptr && !statsCollector_->finished()) {
    const std::string commitContext = createTableCommitContext(false);
    return TableWriteTraits::createAggregationStatsOutput(
        outputType_,
        statsCollector_->getOutput(),
        StringView(commitContext),
        pool());
  }
  finished_ = true;
  return createLastOutput();
}

RowVectorPtr TableWriteMerge::createFragmentsOutput() {
  VELOX_CHECK(!fragmentVectors_.empty());

  auto outputFragmentVector = fragmentVectors_.front();
  fragmentVectors_.pop();
  const int numOutputRows = outputFragmentVector->size();
  std::vector<VectorPtr> outputColumns(outputType_->size());
  for (int outputChannel = 0; outputChannel < outputType_->size();
       ++outputChannel) {
    if (outputChannel == TableWriteTraits::kFragmentChannel) {
      outputColumns[outputChannel] = std::move(outputFragmentVector);
    } else if (outputChannel == TableWriteTraits::kContextChannel) {
      const std::string commitContext = createTableCommitContext(false);
      outputColumns[outputChannel] =
          std::make_shared<ConstantVector<StringView>>(
              pool(),
              numOutputRows,
              false /*isNull*/,
              outputType_->childAt(outputChannel),
              StringView(commitContext));
    } else {
      outputColumns[outputChannel] = BaseVector::createNullConstant(
          outputType_->childAt(outputChannel), numOutputRows, pool());
    }
  }
  return std::make_shared<RowVector>(
      pool(), outputType_, nullptr, numOutputRows, outputColumns);
}

std::string TableWriteMerge::createTableCommitContext(bool lastOutput) const {
  folly::dynamic commitContext = lastCommitContext_;
  commitContext[TableWriteTraits::klastPageContextKey] = lastOutput;
  return folly::toJson(commitContext);
}

RowVectorPtr TableWriteMerge::createLastOutput() {
  VELOX_CHECK(
      lastCommitContext_[TableWriteTraits::klastPageContextKey].asBool(),
      "unexpected last table commit context: {}",
      lastCommitContext_.asString());

  auto output = BaseVector::create<RowVector>(outputType_, 1, pool());
  output->resize(1);
  for (int outputChannel = 0; outputChannel < outputType_->size();
       ++outputChannel) {
    if (outputChannel == TableWriteTraits::kRowCountChannel) {
      auto* rowCounterVector =
          output->childAt(outputChannel)->asFlatVector<int64_t>();
      rowCounterVector->resize(1);
      rowCounterVector->set(0, numRows_);
    } else if (outputChannel == TableWriteTraits::kContextChannel) {
      auto* contextVector =
          output->childAt(outputChannel)->asFlatVector<StringView>();
      contextVector->resize(1);
      const std::string lastCommitContext = createTableCommitContext(true);
      contextVector->set(0, StringView(lastCommitContext));
    } else {
      // All the fragments and statistics shall have already been outputted.
      VELOX_CHECK(fragmentVectors_.empty());
      output->childAt(outputChannel) = BaseVector::createNullConstant(
          outputType_->childAt(outputChannel), 1, pool());
    }
  }
  return output;
}

} // namespace facebook::velox::exec
