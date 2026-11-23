/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include "velox/experimental/cudf/exec/CudfVectorSearchJoinProbe.h"
#include "velox/experimental/cudf/exec/CudfVectorSearchJoinProbeKernels.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/vectorindex/FaissLocalVectorIndex.h"
#include "velox/exec/Task.h"
#include "velox/exec/VectorSearchJoinBridge.h"
#include "velox/exec/VectorSearchJoinProbe.h"

#include <cudf/copying.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/search.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <limits>
#include <numeric>

namespace facebook::velox::cudf_velox {
namespace {

using common::vectorindex::FaissLocalVectorIndex;
using common::vectorindex::LocalVectorIndex;
using common::vectorindex::VectorIndexKind;
using Size = cudf::size_type;

// Scratch and compacted output can each hold one copy of a candidate.
constexpr uint64_t kCandidateBytes =
    2 * (sizeof(int64_t) + sizeof(float) + 2 * sizeof(Size) + sizeof(bool));

template <typename T>
cudf::column_view view(
    cudf::type_id type,
    const rmm::device_uvector<T>& data,
    std::optional<Size> count = std::nullopt) {
  return {
      cudf::data_type{type},
      count.value_or(static_cast<Size>(data.size())),
      data.data(),
      nullptr,
      0};
}

} // namespace

struct CudfVectorSearchJoinProbe::State {
  State(
      cuda::stream_ref stream,
      uint64_t budget,
      size_t dimension,
      int64_t maxK)
      : stream{stream},
        mr{cudf::get_current_device_resource_ref()},
        maxCandidates(
            std::clamp<uint64_t>(
                budget / kCandidateBytes,
                maxK,
                std::numeric_limits<Size>::max())),
        maxMatchPairs(
            std::clamp<uint64_t>(
                budget / (8 * sizeof(Size)),
                1,
                std::numeric_limits<Size>::max())),
        maxSliceRows(
            std::clamp<uint64_t>(
                budget /
                    (2 * sizeof(float) * dimension + maxK * kCandidateBytes +
                     3 * sizeof(Size)),
                1,
                std::min<int64_t>(maxCandidates / maxK, maxMatchPairs))),
        queryValues{0, stream, mr},
        groupValues{0, stream, mr},
        groupRows{0, stream, mr},
        groupCounts{0, stream, mr},
        inputRows{0, stream, mr},
        ids{0, stream, mr},
        scores{0, stream, mr},
        rowMap{0, stream, mr},
        buildMap{0, stream, mr},
        selected{0, stream, mr},
        hasResults{0, stream, mr},
        validation{1, stream, mr} {}

  const cuda::stream_ref stream;
  const rmm::device_async_resource_ref mr;
  const Size maxCandidates;
  const Size maxMatchPairs;
  const Size maxSliceRows;
  CudaEvent inputReady{cudaEventDisableTiming};
  rmm::device_uvector<float> queryValues;
  rmm::device_uvector<float> groupValues;
  rmm::device_uvector<Size> groupRows;
  rmm::device_uvector<Size> groupCounts;
  rmm::device_uvector<Size> inputRows;
  rmm::device_uvector<int64_t> ids;
  rmm::device_uvector<float> scores;
  rmm::device_uvector<Size> rowMap;
  rmm::device_uvector<Size> buildMap;
  rmm::device_uvector<bool> selected;
  rmm::device_uvector<int32_t> hasResults;
  rmm::device_uvector<int32_t> validation;
  std::unique_ptr<cudf::table> buildColumns;
  cudf::table_view buildKeys;
  std::unique_ptr<cudf::filtered_join> keyFilter;
  std::unique_ptr<cudf::column> eligible;
  std::unique_ptr<cudf::table> eligibleKeys;
  std::unique_ptr<rmm::device_uvector<Size>> matchedQueries;
  std::vector<Size> counts;
  Size tileStart{0};
  Size nextGroup{0};
  Size nextMatch{0};
  std::unique_ptr<cudf::table> pending;
  bool pendingMatched{false};
  Size pendingOffset{0};
  Size nextRow{0};
  Size sliceStart{0};
  Size sliceSize{0};
  Size maxOutputRows{1};
  Size nextIndex{0};
  bool unmatchedDone{false};

  cudf::column_view eligibleRows() const {
    return eligible ? eligible->view() : view(cudf::type_id::INT32, inputRows);
  }
};

CudfVectorSearchJoinProbe::CudfVectorSearchJoinProbe(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    core::VectorSearchJoinNodePtr node)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          node->outputType(),
          node->id(),
          "CudfVectorSearchJoinProbe"),
      node_{std::move(node)} {
  const auto& probe = node_->sources()[0]->outputType();
  embeddingChannel_ = probe->getChildIdx(node_->probeEmbedding()->name());
  for (const auto& key : node_->leftKeys()) {
    probeKeyChannels_.push_back(probe->getChildIdx(key->name()));
  }
  Size buildChannel = node_->rightKeys().size();
  for (const auto& name : outputType_->names()) {
    if (name == node_->idName() || name == node_->scoreName()) {
      continue;
    }
    if (const auto channel = probe->getChildIdxIfExists(name)) {
      probeColumns_.push_back(*channel);
    } else {
      buildColumns_.push_back(buildChannel++);
    }
  }
}

CudfVectorSearchJoinProbe::~CudfVectorSearchJoinProbe() = default;

exec::BlockingReason CudfVectorSearchJoinProbe::isBlocked(
    ContinueFuture* future) {
  if (table_ != nullptr) {
    return fallback_ ? fallback_->isBlocked(future)
                     : exec::BlockingReason::kNotBlocked;
  }
  auto table = operatorCtx_->task()
                   ->getVectorSearchJoinBridge(splitGroupId(), planNodeId())
                   ->dataOrFuture(future);
  if (!table.has_value()) {
    return exec::BlockingReason::kWaitForJoinBuild;
  }
  table_ = std::move(*table);
  for (Size row = 0; row < table_->rows()->size(); ++row) {
    const auto* index =
        dynamic_cast<const FaissLocalVectorIndex*>(&table_->indexAt(row));
    const auto k = table_->numResults(row, node_->k());
    const auto* deviceIndex = index &&
            (index->kind() == VectorIndexKind::kFlat ||
             index->kind() == VectorIndexKind::kIvfFlat)
        ? index->deviceIndexForProbe()
        : nullptr;
    if (!deviceIndex || !deviceIndex->supportsDeviceSearch(k)) {
      // The driver has already wired GPU input and output. Run the original
      // CPU probe between conversions for this driver.
      fallback_ = std::make_unique<exec::VectorSearchJoinProbe>(
          operatorId(),
          operatorCtx_->driverCtx(),
          node_,
          "VectorSearchJoinProbeFallback");
      fallback_->initialize();
      addRuntimeStat("vectorSearchDeviceFallback", RuntimeCounter(1));
      if (noMoreInput_) {
        fallback_->noMoreInput();
      }
      return fallback_->isBlocked(future);
    }
    maxK_ = std::max(maxK_, k);
  }
  return exec::BlockingReason::kNotBlocked;
}

bool CudfVectorSearchJoinProbe::needsInput() const {
  return fallback_ ? fallback_->needsInput() : input_ == nullptr;
}

bool CudfVectorSearchJoinProbe::isFinished() {
  return fallback_ ? fallback_->isFinished()
                   : noMoreInput_ && input_ == nullptr;
}

void CudfVectorSearchJoinProbe::doAddInput(RowVectorPtr input) {
  auto gpuInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(gpuInput);
  if (fallback_) {
    fallback_->addInput(
        with_arrow::toVeloxColumn(
            gpuInput->getTableView(),
            pool(),
            input->type(),
            gpuInput->stream(),
            cudf::get_current_device_resource_ref()));
    return;
  }
  if (input->size() == 0) {
    return;
  }
  if (!state_) {
    VELOX_CHECK(
        node_->metric() == common::vectorindex::VectorIndexMetric::kCosine);
    VELOX_CHECK_GT(maxK_, 0);
    state_ = std::make_unique<State>(
        gpuInput->stream(),
        operatorCtx_->driverCtx()->queryConfig().preferredOutputBatchBytes(),
        table_->dimension(),
        maxK_);
    if (table_->rows()->size() > 0 && table_->rows()->childrenSize() > 0) {
      state_->buildColumns = with_arrow::toCudfTable(
          table_->rows(), pool(), state_->stream, state_->mr);
      const auto keyCount = probeKeyChannels_.size();
      if (keyCount > 0) {
        // The build operator stores the equality-key columns first.
        std::vector<Size> buildKeyChannels(keyCount);
        std::iota(buildKeyChannels.begin(), buildKeyChannels.end(), Size{0});
        state_->buildKeys =
            state_->buildColumns->view().select(buildKeyChannels);
      }
      if (keyCount > 1) {
        state_->keyFilter = std::make_unique<cudf::filtered_join>(
            state_->buildKeys,
            cudf::null_equality::UNEQUAL,
            state_->stream,
            state_->mr);
      }
    }
  }
  auto& state = *state_;
  if (gpuInput->stream() != state.stream) {
    state.inputReady.recordFrom(gpuInput->stream()).waitOn(state.stream);
  }
  state.nextRow = 0;
  state.sliceSize = 0;
  const auto probeBytes = input->estimateFlatSize() / input->size();
  const auto& build = table_->rows();
  const auto buildBytes =
      build->size() > 0 ? build->estimateFlatSize() / build->size() : 0;
  state.maxOutputRows = outputBatchRows(
      probeBytes + buildBytes + sizeof(int64_t) + sizeof(float));
  input_ = std::move(input);
}

void CudfVectorSearchJoinProbe::prepareSlice() {
  auto& state = *state_;
  state.sliceStart = state.nextRow;
  state.sliceSize =
      std::min(state.maxSliceRows, input_->size() - state.nextRow);
  state.nextRow += state.sliceSize;
  state.nextIndex = 0;
  state.counts.clear();
  state.nextGroup = 0;
  state.matchedQueries.reset();
  state.eligibleKeys.reset();
  state.inputRows.resize(0, state.stream);
  state.unmatchedDone = false;
  state.eligible.reset();
  if (node_->joinType() == core::JoinType::kLeft) {
    state.hasResults.resize(state.sliceSize, state.stream);
    CUDF_CUDA_TRY(cudaMemsetAsync(
        state.hasResults.data(),
        0,
        state.hasResults.size() * sizeof(int32_t),
        state.stream.get()));
  }
  if (table_->rows()->size() == 0) {
    return;
  }

  const auto input = static_cast<CudfVector&>(*input_).getTableView();
  const auto embeddingColumn = input.column(embeddingChannel_);
  std::unique_ptr<cudf::column> matches;
  std::unique_ptr<rmm::device_uvector<Size>> matchedRows;
  if (probeKeyChannels_.size() == 1) {
    const auto keys =
        cudf::slice(
            input.column(probeKeyChannels_.front()),
            {state.sliceStart, state.sliceStart + state.sliceSize},
            state.stream)
            .front();
    matches =
        cudf::contains(state.buildKeys.column(0), keys, state.stream, state.mr);
  } else if (state.keyFilter) {
    // Only validate embeddings that will be searched. A semi join finds
    // matching queries once, even when several indexes share the same key.
    const auto keys =
        cudf::slice(
            input.select(probeKeyChannels_),
            {state.sliceStart, state.sliceStart + state.sliceSize},
            state.stream)
            .front();
    matchedRows = state.keyFilter->semi_join(keys, state.stream, state.mr);
    if (matchedRows->is_empty()) {
      return;
    }
  }
  const Size rowCount = matchedRows ? matchedRows->size() : state.sliceSize;
  const bool filter = matches || embeddingColumn.has_nulls();
  state.inputRows.resize(rowCount, state.stream);
  state.selected.resize(filter ? rowCount : 0, state.stream);
  vector_search::selectRows(
      rowCount,
      state.sliceStart,
      matchedRows ? matchedRows->data() : nullptr,
      embeddingColumn.offset(),
      embeddingColumn.null_mask(),
      matches ? matches->view().data<bool>() : nullptr,
      matches ? matches->view().null_mask() : nullptr,
      nullptr,
      state.inputRows.data(),
      filter ? state.selected.data() : nullptr,
      state.stream.get());
  if (filter) {
    auto selected =
        cudf::apply_retention_mask(
            cudf::table_view{{view(cudf::type_id::INT32, state.inputRows)}},
            view(cudf::type_id::BOOL8, state.selected),
            state.stream,
            state.mr)
            ->release();
    state.eligible = std::move(selected.front());
  }
  const auto eligible = state.eligibleRows();
  if (eligible.size() == 0) {
    return;
  }
  if (!probeKeyChannels_.empty()) {
    state.eligibleKeys = cudf::gather(
        input.select(probeKeyChannels_),
        eligible,
        cudf::out_of_bounds_policy::DONT_CHECK,
        state.stream,
        state.mr);
  }

  const cudf::lists_column_view lists(embeddingColumn);
  const auto data = lists.child();
  const auto dimension = table_->dimension();
  state.queryValues.resize(
      static_cast<size_t>(eligible.size()) * dimension, state.stream);
  CUDF_CUDA_TRY(cudaMemsetAsync(
      state.validation.data(), 0, sizeof(int32_t), state.stream.get()));
  vector_search::normalize(
      eligible.size(),
      eligible.data<Size>(),
      lists.offsets().data<Size>() + embeddingColumn.offset(),
      data.data<float>(),
      data.null_mask(),
      data.offset(),
      dimension,
      state.queryValues.data(),
      state.validation.data(),
      state.stream.get());
  int32_t validation = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &validation,
      state.validation.data(),
      sizeof(validation),
      cudaMemcpyDeviceToHost,
      state.stream.get()));
  state.stream.sync();
  VELOX_USER_CHECK(
      (validation & vector_search::kInvalidLength) == 0,
      "Query embedding length must match the vector index's");
  VELOX_USER_CHECK(
      (validation & vector_search::kNullElement) == 0,
      "Query embedding must not contain nulls");
}

void CudfVectorSearchJoinProbe::prepareMatchTile() {
  auto& state = *state_;
  state.tileStart = state.nextIndex;
  // Bound the worst case where every probe key matches every build row.
  const Size tileRows = std::min(
      table_->rows()->size() - state.tileStart,
      state.maxMatchPairs / state.eligibleRows().size());
  state.nextIndex += tileRows;
  state.nextGroup = 0;
  state.nextMatch = 0;
  state.counts.assign(tileRows, 0);
  const auto right =
      cudf::slice(
          state.buildKeys, {state.tileStart, state.nextIndex}, state.stream)
          .front();
  auto [probe, build] = cudf::inner_join(
      state.eligibleKeys->view(),
      right,
      cudf::null_equality::UNEQUAL,
      state.stream,
      state.mr);
  state.matchedQueries = std::move(probe);
  if (build->is_empty()) {
    return;
  }
  state.groupCounts.resize(tileRows, state.stream);
  CUDF_CUDA_TRY(cudaMemsetAsync(
      state.groupCounts.data(),
      0,
      tileRows * sizeof(Size),
      state.stream.get()));
  vector_search::groupMatches(
      build->size(),
      build->data(),
      state.matchedQueries->data(),
      state.groupCounts.data(),
      state.stream,
      state.mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      state.counts.data(),
      state.groupCounts.data(),
      tileRows * sizeof(Size),
      cudaMemcpyDeviceToHost,
      state.stream.get()));
  state.stream.sync();
}

bool CudfVectorSearchJoinProbe::fillResults() {
  auto& state = *state_;
  const auto eligible = state.eligibleRows();
  if (eligible.size() == 0) {
    return false;
  }
  const Size indexCount = table_->rows()->size();
  const Size capacity = std::min<int64_t>(
      state.maxCandidates, int64_t{eligible.size()} * maxK_ * indexCount);
  // Accumulate current groups before compacting or gathering result columns.
  state.ids.resize(capacity, state.stream);
  state.scores.resize(capacity, state.stream);
  state.rowMap.resize(capacity, state.stream);
  state.buildMap.resize(capacity, state.stream);
  state.selected.resize(capacity, state.stream);
  const bool hasKeys = !probeKeyChannels_.empty();
  Size slots = 0;
  while (slots < capacity) {
    Size buildRow = state.nextIndex;
    Size queryCount = eligible.size();
    if (hasKeys) {
      while (state.nextGroup >= state.counts.size() &&
             state.nextIndex < indexCount) {
        prepareMatchTile();
      }
      if (state.nextGroup >= state.counts.size()) {
        break;
      }
      buildRow = state.tileStart + state.nextGroup;
      queryCount = state.counts[state.nextGroup];
      if (queryCount == 0) {
        ++state.nextGroup;
        continue;
      }
    } else if (buildRow >= indexCount) {
      break;
    }
    const auto k = table_->numResults(buildRow, node_->k());
    const Size count = queryCount * k;
    if (count > capacity - slots) {
      break;
    }
    const float* values = state.queryValues.data();
    const Size* rows = eligible.data<Size>();
    if (hasKeys) {
      const auto dimension = table_->dimension();
      state.groupValues.resize(
          static_cast<size_t>(queryCount) * dimension, state.stream);
      state.groupRows.resize(queryCount, state.stream);
      vector_search::gatherQueries(
          queryCount,
          dimension,
          state.matchedQueries->data() + state.nextMatch,
          rows,
          values,
          state.groupRows.data(),
          state.groupValues.data(),
          state.stream.get());
      rows = state.groupRows.data();
      values = state.groupValues.data();
      ++state.nextGroup;
      state.nextMatch += queryCount;
    } else {
      ++state.nextIndex;
    }
    const auto& index =
        static_cast<const FaissLocalVectorIndex&>(table_->indexAt(buildRow));
    const auto* deviceIndex = index.deviceIndexForProbe();
    VELOX_CHECK(deviceIndex->searchDevice(
        values,
        queryCount,
        k,
        state.ids.data() + slots,
        state.scores.data() + slots,
        state.stream.get()));
    vector_search::mapResults(
        count,
        k,
        state.sliceStart,
        buildRow,
        rows,
        deviceIndex->deviceIdMap(),
        index.size(),
        LocalVectorIndex::kPaddingId,
        state.ids.data() + slots,
        state.rowMap.data() + slots,
        state.buildMap.data() + slots,
        state.selected.data() + slots,
        state.hasResults.is_empty() ? nullptr : state.hasResults.data(),
        state.stream.get());
    slots += count;
  }
  if (slots == 0) {
    return false;
  }
  state.pending = cudf::apply_retention_mask(
      cudf::table_view{
          {view(cudf::type_id::INT32, state.rowMap, slots),
           view(cudf::type_id::INT32, state.buildMap, slots),
           view(cudf::type_id::INT64, state.ids, slots),
           view(cudf::type_id::FLOAT32, state.scores, slots)}},
      view(cudf::type_id::BOOL8, state.selected, slots),
      state.stream,
      state.mr);
  state.pendingMatched = true;
  return true;
}

void CudfVectorSearchJoinProbe::collectUnmatched() {
  auto& state = *state_;
  state.unmatchedDone = true;
  state.rowMap.resize(state.sliceSize, state.stream);
  state.selected.resize(state.sliceSize, state.stream);
  vector_search::selectRows(
      state.sliceSize,
      state.sliceStart,
      nullptr,
      0,
      nullptr,
      nullptr,
      nullptr,
      state.hasResults.data(),
      state.rowMap.data(),
      state.selected.data(),
      state.stream.get());
  state.pending = cudf::apply_retention_mask(
      cudf::table_view{{view(cudf::type_id::INT32, state.rowMap)}},
      view(cudf::type_id::BOOL8, state.selected),
      state.stream,
      state.mr);
  state.pendingMatched = false;
}

RowVectorPtr CudfVectorSearchJoinProbe::makeOutput() {
  auto& state = *state_;
  const auto rowCount = std::min(
      state.maxOutputRows, state.pending->num_rows() - state.pendingOffset);
  const auto selected =
      cudf::slice(
          state.pending->view(),
          {state.pendingOffset, state.pendingOffset + rowCount},
          state.stream)
          .front();
  const auto& probeType = node_->sources()[0]->outputType();
  const auto probe = static_cast<CudfVector&>(*input_).getTableView();
  const auto outputMr = get_output_mr();
  const cudf::memory_resources resources{outputMr, state.mr};
  std::vector<std::unique_ptr<cudf::column>> probeColumns;
  if (!probeColumns_.empty()) {
    probeColumns = cudf::gather(
                       probe.select(probeColumns_),
                       selected.column(0),
                       cudf::out_of_bounds_policy::DONT_CHECK,
                       state.stream,
                       resources)
                       ->release();
  }
  std::vector<std::unique_ptr<cudf::column>> buildColumns;
  if (state.pendingMatched && !buildColumns_.empty()) {
    buildColumns = cudf::gather(
                       state.buildColumns->view().select(buildColumns_),
                       selected.column(1),
                       cudf::out_of_bounds_policy::DONT_CHECK,
                       state.stream,
                       resources)
                       ->release();
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(outputType_->size());
  Size probeOffset = 0;
  Size buildOffset = 0;
  for (Size i = 0; i < outputType_->size(); ++i) {
    const auto& name = outputType_->nameOf(i);
    if (name == node_->idName() || name == node_->scoreName()) {
      columns.push_back(
          state.pendingMatched
              ? std::make_unique<cudf::column>(
                    selected.column(name == node_->idName() ? 2 : 3),
                    state.stream,
                    outputMr)
              : makeAllNullColumn(
                    outputType_->childAt(i), rowCount, state.stream, outputMr));
    } else if (probeType->containsChild(name)) {
      columns.push_back(std::move(probeColumns[probeOffset++]));
    } else {
      columns.push_back(
          state.pendingMatched
              ? std::move(buildColumns[buildOffset++])
              : makeAllNullColumn(
                    outputType_->childAt(i), rowCount, state.stream, outputMr));
    }
  }
  state.pendingOffset += rowCount;
  return std::make_shared<CudfVector>(
      pool(),
      outputType_,
      rowCount,
      std::make_unique<cudf::table>(std::move(columns)),
      state.stream);
}

RowVectorPtr CudfVectorSearchJoinProbe::doGetOutput() {
  if (fallback_) {
    auto output = fallback_->getOutput();
    if (output == nullptr) {
      return nullptr;
    }
    const auto stream = getDefaultStreamForCurrentThread();
    auto table =
        with_arrow::toCudfTable(output, pool(), stream, get_output_mr());
    return std::make_shared<CudfVector>(
        pool(), outputType_, output->size(), std::move(table), stream);
  }
  while (input_) {
    auto& state = *state_;
    if (state.pending && state.pendingOffset < state.pending->num_rows()) {
      return makeOutput();
    }
    state.pending.reset();
    state.pendingOffset = 0;
    if (state.sliceSize == 0) {
      if (state.nextRow == input_->size()) {
        releaseInput();
        return nullptr;
      }
      prepareSlice();
    }
    if (fillResults()) {
      continue;
    } else if (
        !state.unmatchedDone && node_->joinType() == core::JoinType::kLeft) {
      collectUnmatched();
    } else {
      state.sliceSize = 0;
    }
  }
  return nullptr;
}

void CudfVectorSearchJoinProbe::releaseInput() {
  if (input_) {
    const auto inputStream = static_cast<const CudfVector&>(*input_).stream();
    if (inputStream != state_->stream) {
      state_->inputReady.recordFrom(state_->stream).waitOn(inputStream);
    }
    input_.reset();
  }
}

void CudfVectorSearchJoinProbe::doNoMoreInput() {
  Operator::noMoreInput();
  if (fallback_) {
    fallback_->noMoreInput();
  }
}

void CudfVectorSearchJoinProbe::doClose() {
  if (fallback_) {
    fallback_->close();
    fallback_.reset();
  }
  releaseInput();
  if (state_) {
    // FAISS owns the index and the ID map. Retain them until all searches
    // queued by this driver on its execution stream have completed.
    state_->inputReady.recordFrom(state_->stream).synchronize();
    state_.reset();
  }
  table_.reset();
  Operator::close();
}

} // namespace facebook::velox::cudf_velox
