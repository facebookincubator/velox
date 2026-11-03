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

#include "velox/experimental/cudf/exec/CudfVectorSearchJoin.h"
#include "velox/experimental/cudf/exec/CudfVectorSearchKernels.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/vectorindex/GpuLocalVectorIndex.h"
#include "velox/exec/Task.h"
#include "velox/exec/VectorSearchJoinBridge.h"
#include "velox/vector/DecodedVector.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/join/join.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include "velox/experimental/cudf/CudfNoDefaults.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <numeric>

namespace facebook::velox::cudf_velox {
namespace {

using common::vectorindex::LocalVectorIndex;
using common::vectorindex::LocalVectorIndexAccelerator;

// Loads every local index onto the current device, with no CPU search left
// behind.
std::shared_ptr<const LocalVectorIndexAccelerator> deviceAccelerator() {
  static std::once_flag once;
  static std::shared_ptr<const LocalVectorIndexAccelerator> accelerator;
  std::call_once(once, [] {
    int device;
    CUDF_CUDA_TRY(cudaGetDevice(&device));
    accelerator = common::vectorindex::makeGpuLocalVectorIndexAccelerator(
        {.device = device, .requireDevice = true});
  });
  return accelerator;
}

template <typename T>
std::unique_ptr<cudf::column> valuesColumn(
    rmm::device_uvector<T>&& values,
    cuda::stream_ref stream) {
  const auto size = static_cast<cudf::size_type>(values.size());
  return std::make_unique<cudf::column>(
      std::move(values),
      cudf::create_null_mask(
          size,
          cudf::mask_state::UNALLOCATED,
          stream,
          cudf::memory_resources{get_temp_mr(), get_temp_mr()}),
      0);
}

std::unique_ptr<cudf::column> sizeTypeColumn(
    cudf::size_type value,
    cudf::size_type size,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  const cudf::numeric_scalar<cudf::size_type> scalar(value, true, stream, mr);
  return cudf::make_column_from_scalar(scalar, size, stream, mr);
}

} // namespace

CudfVectorSearchJoinTable::CudfVectorSearchJoinTable(
    std::unique_ptr<cudf::table> rows,
    column_index_t numKeys,
    std::vector<Index> indexes,
    VectorPtr packedIndexes,
    cuda::stream_ref stream)
    : stream_{stream},
      rows_{std::move(rows)},
      packedIndexes_{std::move(packedIndexes)},
      indexes_{std::move(indexes)} {
  VELOX_CHECK_EQ(rows_->num_rows(), static_cast<int64_t>(indexes_.size()));
  if (!indexes_.empty()) {
    dimension_ = indexes_[0]->dimension();
  }
  for (const auto& index : indexes_) {
    VELOX_USER_CHECK_EQ(
        index->dimension(),
        dimension_,
        "Vector search join indexes must have the same dimension");
    VELOX_CHECK_NOT_NULL(
        index->deviceIndex(), "Vector search join index is not on the GPU");
  }
  if (numKeys > 0 && rows_->num_rows() > 0) {
    std::vector<cudf::size_type> keys(numKeys);
    std::iota(keys.begin(), keys.end(), 0);
    hashJoin_ = std::make_unique<cudf::hash_join>(
        rows_->view().select(keys),
        cudf::null_equality::UNEQUAL,
        stream,
        get_temp_mr());
  }
  readyEvent_ = std::make_shared<CudaEvent>(cudaEventDisableTiming);
  readyEvent_->recordFrom(stream);
}

CudfVectorSearchJoinTable::~CudfVectorSearchJoinTable() = default;

void CudfVectorSearchJoinTable::waitUntilReady(cuda::stream_ref stream) const {
  readyEvent_->waitOn(stream);
}

void CudfVectorSearchJoinTable::recordRead(cuda::stream_ref stream) const {
  CudaEvent event(cudaEventDisableTiming);
  event.recordFrom(stream).waitOn(stream_);
}

CudfVectorSearchJoinBuild::CudfVectorSearchJoinBuild(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::VectorSearchJoinNode> joinNode)
    : CudfJoinBuild(
          operatorId,
          driverCtx,
          joinNode,
          "CudfVectorSearchJoinBuild",
          NvtxMethodFlag::kAddInput | NvtxMethodFlag::kNoMoreInput),
      joinNode_{std::move(joinNode)},
      searchOptions_{common::vectorindex::LocalVectorIndexOptions::parseSearch(
          {joinNode_->searchOptions().begin(),
           joinNode_->searchOptions().end()})} {
  const auto& buildType = joinNode_->sources()[1]->outputType();
  for (const auto& key : joinNode_->rightKeys()) {
    channels_.push_back(buildType->getChildIdx(key->name()));
  }
  numKeys_ = channels_.size();
  for (const auto& name : joinNode_->outputType()->names()) {
    if (const auto channel = buildType->getChildIdxIfExists(name)) {
      channels_.push_back(*channel);
    }
  }
  indexChannel_ = buildType->getChildIdx(joinNode_->buildIndex()->name());
}

void CudfVectorSearchJoinBuild::buildAndPublish(
    std::vector<CudfVectorPtr> inputs) {
  auto stream = cudfGlobalStreamPool().get_stream();
  const auto& buildType = joinNode_->sources()[1]->outputType();
  auto input = getConcatenatedTable(
      std::move(inputs), buildType, stream, get_output_mr());

  // A build row with a null key or index matches nothing.
  std::vector<cudf::size_type> required(
      channels_.begin(), channels_.begin() + numKeys_);
  required.push_back(indexChannel_);
  auto kept =
      cudf::drop_nulls(input->view(), required, stream, get_output_mr());
  input.reset();

  auto rows = std::make_unique<cudf::table>(
      kept->view().select(channels_), stream, get_output_mr());

  // FAISS parses the packed indexes on the host. Each becomes a device index
  // that the probe side searches with device queries.
  const auto indexType = ROW({"index"}, {buildType->childAt(indexChannel_)});
  auto packed = with_arrow::toVeloxColumn(
      cudf::table_view({kept->view().column(indexChannel_)}),
      pool(),
      indexType,
      "",
      stream,
      get_temp_mr());
  stream.sync();
  kept.reset();
  auto packedIndexes = packed->childAt(0);
  DecodedVector decoded(*packedIndexes);
  const auto accelerator = deviceAccelerator();
  std::vector<exec::VectorSearchJoinTable::Index> indexes;
  indexes.reserve(packedIndexes->size());
  for (vector_size_t row = 0; row < packedIndexes->size(); ++row) {
    const auto bytes = decoded.valueAt<StringView>(row);
    indexes.push_back(
        std::make_unique<LocalVectorIndex>(
            std::string_view(bytes.data(), bytes.size()),
            joinNode_->metric(),
            searchOptions_,
            accelerator));
  }

  operatorCtx_->task()
      ->getVectorSearchJoinBridge(
          operatorCtx_->driverCtx()->splitGroupId, planNodeId())
      ->setData(
          std::make_shared<const CudfVectorSearchJoinTable>(
              std::move(rows),
              numKeys_,
              std::move(indexes),
              std::move(packedIndexes),
              stream));
}

CudfVectorSearchJoinProbe::CudfVectorSearchJoinProbe(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::VectorSearchJoinNode> joinNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          joinNode->outputType(),
          joinNode->id(),
          "CudfVectorSearchJoinProbe",
          nvtx3::rgb{0, 128, 128},
          NvtxMethodFlag::kGetOutput,
          std::nullopt,
          joinNode),
      isLeftJoin_{joinNode->joinType() == core::JoinType::kLeft},
      metric_{joinNode->metric()},
      k_{joinNode->k()} {
  const auto& probeType = joinNode->sources()[0]->outputType();
  const auto& buildType = joinNode->sources()[1]->outputType();
  for (const auto& key : joinNode->leftKeys()) {
    keyChannels_.push_back(probeType->getChildIdx(key->name()));
  }
  embeddingChannel_ =
      probeType->getChildIdx(joinNode->probeEmbedding()->name());

  // The table's rows hold the keys, then the build columns of the output in
  // order.
  auto buildChannel = static_cast<cudf::size_type>(keyChannels_.size());
  const auto& outputType = joinNode->outputType();
  for (column_index_t i = 0; i < outputType->size(); ++i) {
    const auto& name = outputType->nameOf(i);
    if (name == joinNode->idName()) {
      idChannel_ = i;
    } else if (name == joinNode->scoreName()) {
      scoreChannel_ = i;
    } else if (const auto channel = probeType->getChildIdxIfExists(name)) {
      probeProjections_.emplace_back(*channel, i);
    } else {
      VELOX_CHECK(buildType->containsChild(name));
      buildProjections_.emplace_back(buildChannel++, i);
    }
  }
}

exec::BlockingReason CudfVectorSearchJoinProbe::isBlocked(
    ContinueFuture* future) {
  if (table_ != nullptr) {
    return exec::BlockingReason::kNotBlocked;
  }
  auto table = operatorCtx_->task()
                   ->getVectorSearchJoinBridge(splitGroupId(), planNodeId())
                   ->dataOrFuture(future);
  if (!table.has_value()) {
    return exec::BlockingReason::kWaitForJoinBuild;
  }
  table_ = std::dynamic_pointer_cast<const CudfVectorSearchJoinTable>(
      std::move(table.value()));
  VELOX_CHECK_NOT_NULL(
      table_, "The vector search join build side did not run on the GPU");
  return exec::BlockingReason::kNotBlocked;
}

void CudfVectorSearchJoinProbe::doAddInput(RowVectorPtr input) {
  input_ = std::move(input);
}

std::unique_ptr<cudf::table> CudfVectorSearchJoinProbe::pairsToSearch(
    cudf::table_view probe,
    cuda::stream_ref stream) {
  const auto mr = get_temp_mr();
  std::unique_ptr<cudf::table> pairs;
  if (const auto* hashJoin = table_->hashJoin()) {
    auto [probeRows, buildRows] =
        hashJoin->inner_join(
            probe.select(keyChannels_), std::nullopt, stream, mr);
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(valuesColumn(std::move(*probeRows), stream));
    columns.push_back(valuesColumn(std::move(*buildRows), stream));
    pairs = std::make_unique<cudf::table>(std::move(columns));
  } else {
    // With no keys, every probe row searches every build row.
    const cudf::numeric_scalar<cudf::size_type> zero(0, true, stream, mr);
    auto probeRows = cudf::sequence(probe.num_rows(), zero, stream, mr);
    auto buildRows = cudf::sequence(table_->numRows(), zero, stream, mr);
    pairs = cudf::cross_join(
        cudf::table_view({probeRows->view()}),
        cudf::table_view({buildRows->view()}),
        stream,
        mr);
  }

  const auto embeddings = probe.column(embeddingChannel_);
  if (embeddings.has_nulls() && pairs->num_rows() > 0) {
    auto valid = cudf::is_valid(embeddings, stream, mr);
    auto pairValid = cudf::gather(
        cudf::table_view({valid->view()}),
        pairs->get_column(0).view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        cudf::memory_resources{mr, get_temp_mr()});
    pairs = cudf::apply_boolean_mask(
        pairs->view(), pairValid->get_column(0).view(), stream, mr);
  }

  return cudf::stable_sort_by_key(
      pairs->view(),
      cudf::table_view({pairs->get_column(1).view()}),
      {},
      {},
      stream,
      mr);
}

std::unique_ptr<cudf::table> CudfVectorSearchJoinProbe::search(
    cudf::table_view probe,
    cuda::stream_ref stream) {
  const auto mr = get_temp_mr();
  const auto numProbeRows = probe.num_rows();
  std::unique_ptr<cudf::column> hasResults;
  if (isLeftJoin_) {
    const cudf::numeric_scalar<bool> none(false, true, stream, mr);
    hasResults =
        cudf::make_column_from_scalar(none, numProbeRows, stream, mr);
  }

  std::vector<std::unique_ptr<cudf::table>> results;
  if (table_->numRows() > 0 && numProbeRows > 0) {
    auto pairs = pairsToSearch(probe, stream);
    const auto numPairs = pairs->num_rows();
    if (numPairs > 0) {
      const auto* pairProbeRows =
          pairs->get_column(0).view().data<cudf::size_type>();
      const auto starts = vector_search::runStarts(
          pairs->get_column(1).view().data<cudf::size_type>(),
          numPairs,
          table_->numRows(),
          stream);

      // The queries, one per pair in order, as one device matrix.
      auto gathered = cudf::gather(
          cudf::table_view({probe.column(embeddingChannel_)}),
          pairs->get_column(0).view(),
          cudf::out_of_bounds_policy::DONT_CHECK,
          cudf::negative_index_policy::NOT_ALLOWED,
          stream,
          cudf::memory_resources{mr, get_temp_mr()});
      auto lists = std::move(gathered->release()[0]);
      const auto dimension = table_->embeddingDimension();
      {
        const cudf::lists_column_view view(lists->view());
        VELOX_USER_CHECK_EQ(
            vector_search::countWrongLengths(
                view.offsets().data<cudf::size_type>(),
                numPairs,
                dimension,
                stream),
            0,
            "Query embedding length must match the vector index's");
        VELOX_USER_CHECK_EQ(
            view.child().null_count(),
            0,
            "Query embedding must not contain nulls");
      }
      auto contents = lists->release();
      auto& values =
          contents.children[cudf::lists_column_view::child_column_index];
      auto* queries = values->mutable_view().data<float>();
      switch (metric_) {
        case common::vectorindex::VectorIndexMetric::kCosine:
          vector_search::normalizeRows(queries, numPairs, dimension, stream);
          break;
      }

      auto* rawHasResults =
          hasResults ? hasResults->mutable_view().data<bool>() : nullptr;
      for (cudf::size_type buildRow = 0; buildRow < table_->numRows();
           ++buildRow) {
        const auto& index = table_->index(buildRow);
        const auto* device = index.deviceIndex();
        const int64_t k = std::min(k_, index.size());
        const int64_t maxQueries = std::max<int64_t>(
            1, std::numeric_limits<cudf::size_type>::max() / k);
        for (int64_t start = starts[buildRow]; start < starts[buildRow + 1];
             start += maxQueries) {
          const int64_t numQueries =
              std::min<int64_t>(maxQueries, starts[buildRow + 1] - start);
          const auto numResults = numQueries * k;
          rmm::device_uvector<int64_t> ids(numResults, stream, mr);
          rmm::device_uvector<float> scores(numResults, stream, mr);
          rmm::device_uvector<cudf::size_type> probeRows(
              numResults, stream, mr);
          rmm::device_uvector<bool> found(numResults, stream, mr);
          VELOX_CHECK(
              device->searchDevice(
                  queries + start * dimension,
                  numQueries,
                  k,
                  ids.data(),
                  scores.data(),
                  stream.get()),
              "The GPU cannot search a vector index for k = {}",
              k);
          vector_search::emitResults(
              ids.data(),
              numQueries,
              k,
              device->deviceIdMap(),
              index.size(),
              pairProbeRows + start,
              probeRows.data(),
              found.data(),
              rawHasResults,
              stream);
          std::vector<std::unique_ptr<cudf::column>> columns;
          columns.push_back(valuesColumn(std::move(probeRows), stream));
          columns.push_back(sizeTypeColumn(
              buildRow, static_cast<cudf::size_type>(numResults), stream, mr));
          columns.push_back(valuesColumn(std::move(ids), stream));
          columns.push_back(valuesColumn(std::move(scores), stream));
          const cudf::table candidates(std::move(columns));
          auto foundColumn = valuesColumn(std::move(found), stream);
          results.push_back(cudf::apply_boolean_mask(
              candidates.view(), foundColumn->view(), stream, mr));
        }
      }
    }
  }

  if (isLeftJoin_ && numProbeRows > 0) {
    // Probe rows with no result join a null build row.
    auto noResults = cudf::unary_operation(
        hasResults->view(), cudf::unary_operator::NOT, stream, mr);
    const cudf::numeric_scalar<cudf::size_type> zero(0, true, stream, mr);
    auto sequence = cudf::sequence(numProbeRows, zero, stream, mr);
    auto unmatched = cudf::apply_boolean_mask(
        cudf::table_view({sequence->view()}), noResults->view(), stream, mr);
    const auto numUnmatched = unmatched->num_rows();
    if (numUnmatched > 0) {
      std::vector<std::unique_ptr<cudf::column>> columns;
      columns.push_back(std::move(unmatched->release()[0]));
      columns.push_back(sizeTypeColumn(-1, numUnmatched, stream, mr));
      columns.push_back(cudf::make_numeric_column(
          cudf::data_type{cudf::type_id::INT64},
          numUnmatched,
          cudf::mask_state::ALL_NULL,
          stream,
          mr));
      columns.push_back(cudf::make_numeric_column(
          cudf::data_type{cudf::type_id::FLOAT32},
          numUnmatched,
          cudf::mask_state::ALL_NULL,
          stream,
          mr));
      results.push_back(std::make_unique<cudf::table>(std::move(columns)));
    }
  }

  if (results.empty()) {
    return nullptr;
  }
  if (results.size() == 1) {
    return std::move(results[0]);
  }
  std::vector<cudf::table_view> views;
  views.reserve(results.size());
  for (const auto& result : results) {
    views.push_back(result->view());
  }
  return cudf::concatenate(views, stream, mr);
}

RowVectorPtr CudfVectorSearchJoinProbe::doGetOutput() {
  if (input_ == nullptr) {
    return nullptr;
  }
  auto input = std::dynamic_pointer_cast<CudfVector>(std::move(input_));
  input_.reset();
  VELOX_CHECK_NOT_NULL(input);
  const auto stream = input->stream();
  table_->waitUntilReady(stream);
  const auto probe = input->getTableView();

  auto results = search(probe, stream);
  if (results == nullptr || results->num_rows() == 0) {
    table_->recordRead(stream);
    return nullptr;
  }

  const auto numRows = results->num_rows();
  const auto mr = get_output_mr();
  std::vector<std::unique_ptr<cudf::column>> outputs(outputType_->size());
  if (!probeProjections_.empty()) {
    std::vector<cudf::size_type> channels;
    for (const auto& [channel, _] : probeProjections_) {
      channels.push_back(channel);
    }
    auto columns = cudf::gather(
                       probe.select(channels),
                       results->get_column(0).view(),
                       cudf::out_of_bounds_policy::DONT_CHECK,
                       cudf::negative_index_policy::NOT_ALLOWED,
                       stream,
                       cudf::memory_resources{mr, get_temp_mr()})
                       ->release();
    for (size_t i = 0; i < probeProjections_.size(); ++i) {
      outputs[probeProjections_[i].second] = std::move(columns[i]);
    }
  }
  if (!buildProjections_.empty()) {
    std::vector<cudf::size_type> channels;
    for (const auto& [channel, _] : buildProjections_) {
      channels.push_back(channel);
    }
    auto columns = cudf::gather(
                       table_->deviceRows().select(channels),
                       results->get_column(1).view(),
                       cudf::out_of_bounds_policy::NULLIFY,
                       cudf::negative_index_policy::NOT_ALLOWED,
                       stream,
                       cudf::memory_resources{mr, get_temp_mr()})
                       ->release();
    for (size_t i = 0; i < buildProjections_.size(); ++i) {
      outputs[buildProjections_[i].second] = std::move(columns[i]);
    }
  }
  table_->recordRead(stream);

  auto resultColumns = results->release();
  if (idChannel_.has_value()) {
    outputs[*idChannel_] = std::move(resultColumns[2]);
  }
  if (scoreChannel_.has_value()) {
    outputs[*scoreChannel_] = std::move(resultColumns[3]);
  }
  return std::make_shared<CudfVector>(
      pool(),
      outputType_,
      numRows,
      std::make_unique<cudf::table>(std::move(outputs)),
      stream);
}

void CudfVectorSearchJoinProbe::doClose() {
  table_.reset();
  Operator::close();
}

} // namespace facebook::velox::cudf_velox
