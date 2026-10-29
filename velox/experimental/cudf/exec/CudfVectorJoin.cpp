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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfVectorJoin.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/serialization/DeserializationRegistry.h"
#include "velox/exec/Task.h"
#include "velox/type/Type.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <limits>
#include <utility>

namespace facebook::velox::cudf_velox {

[[noreturn]] void failRuntime(const std::string& reason) {
  VELOX_FAIL("{}", reason);
}

[[noreturn]] void failUser(const std::string& reason) {
  VELOX_USER_FAIL("{}", reason);
}

namespace {

std::string metricName(VectorJoinMetric metric) {
  switch (metric) {
    case VectorJoinMetric::kL2Squared:
      return "l2";
    case VectorJoinMetric::kInnerProduct:
      return "dot";
    case VectorJoinMetric::kCosine:
      return "cosine";
  }
  VELOX_UNREACHABLE("Unknown vector join metric");
}

VectorJoinMetric metricFromName(const std::string& name) {
  if (name == "l2") {
    return VectorJoinMetric::kL2Squared;
  }
  if (name == "dot") {
    return VectorJoinMetric::kInnerProduct;
  }
  if (name == "cosine") {
    return VectorJoinMetric::kCosine;
  }
  VELOX_USER_FAIL("Unknown vector join metric: {}", name);
}

int32_t channelOf(const RowTypePtr& type, const std::string& name) {
  const auto channel = type->getChildIdxIfExists(name);
  VELOX_USER_CHECK(
      channel.has_value(), "Vector join field {} is not in the input", name);
  return static_cast<int32_t>(channel.value());
}

int32_t positiveOr(int32_t value, int32_t fallback) {
  return value > 0 ? value : fallback;
}

std::unique_ptr<cudf::column> intColumn(
    const std::vector<int32_t>& values,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto column = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32},
      static_cast<cudf::size_type>(values.size()),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  if (!values.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        column->mutable_view().data<int32_t>(),
        values.data(),
        sizeof(int32_t) * values.size(),
        cudaMemcpyHostToDevice,
        stream.get()));
  }
  return column;
}

int32_t batchOf(const std::vector<int64_t>& rowStarts, int64_t row) {
  const auto next = std::upper_bound(rowStarts.begin(), rowStarts.end(), row);
  VELOX_CHECK(
      next != rowStarts.begin(), "Vector join row {} has no batch", row);
  return static_cast<int32_t>(std::distance(rowStarts.begin(), next) - 1);
}

std::unique_ptr<cudf::column> gatherOne(
    const cudf::column_view& column,
    const cudf::column_view& gatherMap,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  cudf::table_view table{{column}};
  auto gathered = cudf::gather(
      table,
      gatherMap,
      cudf::out_of_bounds_policy::DONT_CHECK,
      stream,
      cudf::memory_resources{mr, mr});
  return std::move(gathered->release()[0]);
}

std::unique_ptr<cudf::column> gatherBuildColumn(
    const VectorJoinBuildData& build,
    int32_t channel,
    const std::vector<int32_t>& batches,
    const std::vector<int32_t>& localRows,
    const std::vector<int32_t>& outputSlots,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  std::vector<std::vector<int32_t>> rowsByBatch(build.batches.size());
  std::vector<std::vector<int32_t>> slotsByBatch(build.batches.size());
  for (size_t i = 0; i < batches.size(); ++i) {
    rowsByBatch[batches[i]].push_back(localRows[i]);
    slotsByBatch[batches[i]].push_back(outputSlots[i]);
  }
  std::vector<std::unique_ptr<cudf::column>> pieces;
  std::vector<cudf::column_view> views;
  std::vector<int32_t> reorder(batches.size());
  int32_t cursor = 0;
  for (size_t batch = 0; batch < rowsByBatch.size(); ++batch) {
    if (rowsByBatch[batch].empty()) {
      continue;
    }
    auto map = intColumn(rowsByBatch[batch], stream, mr);
    auto piece = gatherOne(
        build.batches[batch]->getTableView().column(channel),
        map->view(),
        stream,
        mr);
    for (int32_t slot : slotsByBatch[batch]) {
      reorder[slot] = cursor++;
    }
    views.push_back(piece->view());
    pieces.push_back(std::move(piece));
  }
  std::unique_ptr<cudf::column> concatenated;
  if (pieces.size() == 1) {
    concatenated = std::move(pieces[0]);
  } else {
    concatenated = cudf::concatenate(views, stream, mr);
  }
  auto order = intColumn(reorder, stream, mr);
  return gatherOne(concatenated->view(), order->view(), stream, mr);
}

std::unique_ptr<cudf::column> scoreColumn(
    const std::vector<float>& scores,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
  auto column = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::FLOAT32},
      static_cast<cudf::size_type>(scores.size()),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  if (!scores.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        column->mutable_view().data<float>(),
        scores.data(),
        sizeof(float) * scores.size(),
        cudaMemcpyHostToDevice,
        stream.get()));
  }
  return column;
}

} // namespace

class VectorJoinAdapter : public OperatorAdapter {
 public:
  VectorJoinAdapter() : OperatorAdapter("CudfVectorJoin") {}

  bool canHandle(const exec::Operator* op) const override {
    return op->is<CudfVectorJoinProbe>() || op->is<CudfVectorJoinBuild>();
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/) const override {
    return true;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {};
  }

  bool keepOperator() const override {
    return true;
  }
};

CudfVectorJoinNode::CudfVectorJoinNode(
    const core::PlanNodeId& id,
    std::string probeVector,
    std::string buildVector,
    std::string buildId,
    std::string score,
    VectorJoinMetric metric,
    int32_t k,
    int32_t candidateMultiplier,
    int32_t nLists,
    int32_t nProbes,
    int32_t pqBits,
    int32_t pqDim,
    core::PlanNodePtr probe,
    core::PlanNodePtr build,
    RowTypePtr outputType)
    : PlanNode(id),
      probeVector_(std::move(probeVector)),
      buildVector_(std::move(buildVector)),
      buildId_(std::move(buildId)),
      score_(std::move(score)),
      metric_(metric),
      k_(k),
      candidateMultiplier_(candidateMultiplier),
      nLists_(nLists),
      nProbes_(nProbes),
      pqBits_(pqBits),
      pqDim_(pqDim),
      sources_{std::move(probe), std::move(build)},
      outputType_(std::move(outputType)) {
  VELOX_USER_CHECK_GT(k_, 0, "Vector join k must be positive");
  VELOX_USER_CHECK_GE(candidateMultiplier_, 0);
  VELOX_USER_CHECK_GE(nLists_, 0);
  VELOX_USER_CHECK_GE(nProbes_, 0);
  VELOX_USER_CHECK_GE(pqBits_, 0);
  VELOX_USER_CHECK_GE(pqDim_, 0);
  const auto probeType = sources_[0]->outputType();
  const auto buildType = sources_[1]->outputType();
  const auto probeVectors =
      probeType->childAt(channelOf(probeType, probeVector_));
  const auto buildVectors =
      buildType->childAt(channelOf(buildType, buildVector_));
  VELOX_USER_CHECK(
      probeVectors->kind() == TypeKind::ARRAY &&
          probeVectors->childAt(0)->kind() == TypeKind::REAL,
      "Vector join probe column {} must be ARRAY(REAL)",
      probeVector_);
  VELOX_USER_CHECK(
      buildVectors->kind() == TypeKind::ARRAY &&
          buildVectors->childAt(0)->kind() == TypeKind::REAL,
      "Vector join build column {} must be ARRAY(REAL)",
      buildVector_);
  VELOX_USER_CHECK(
      buildType->childAt(channelOf(buildType, buildId_))->kind() ==
          TypeKind::BIGINT,
      "Vector join build id {} must be BIGINT",
      buildId_);
  VELOX_USER_CHECK(
      outputType_->childAt(channelOf(outputType_, score_))->kind() ==
          TypeKind::REAL,
      "Vector join score {} must be REAL",
      score_);
  for (const auto& name : outputType_->names()) {
    if (name == score_) {
      continue;
    }
    VELOX_USER_CHECK(
        probeType->containsChild(name) || buildType->containsChild(name),
        "Vector join output column {} is not in the probe or build input",
        name);
  }
}

void CudfVectorJoinNode::addDetails(std::stringstream& stream) const {
  stream << metricName(metric_) << ", k=" << k_;
}

folly::dynamic CudfVectorJoinNode::serialize() const {
  auto obj = PlanNode::serialize();
  obj["probeVector"] = probeVector_;
  obj["buildVector"] = buildVector_;
  obj["buildId"] = buildId_;
  obj["score"] = score_;
  obj["metric"] = metricName(metric_);
  obj["k"] = static_cast<int64_t>(k_);
  obj["candidateMultiplier"] = static_cast<int64_t>(candidateMultiplier_);
  obj["nLists"] = static_cast<int64_t>(nLists_);
  obj["nProbes"] = static_cast<int64_t>(nProbes_);
  obj["pqBits"] = static_cast<int64_t>(pqBits_);
  obj["pqDim"] = static_cast<int64_t>(pqDim_);
  obj["outputType"] = outputType_->serialize();
  return obj;
}

core::PlanNodePtr CudfVectorJoinNode::create(
    const folly::dynamic& obj,
    void* context) {
  auto sources = ISerializable::deserialize<std::vector<core::PlanNode>>(
      obj["sources"], context);
  VELOX_CHECK_EQ(sources.size(), 2);
  return std::make_shared<CudfVectorJoinNode>(
      obj["id"].asString(),
      obj["probeVector"].asString(),
      obj["buildVector"].asString(),
      obj["buildId"].asString(),
      obj["score"].asString(),
      metricFromName(obj["metric"].asString()),
      static_cast<int32_t>(obj["k"].asInt()),
      static_cast<int32_t>(obj["candidateMultiplier"].asInt()),
      static_cast<int32_t>(obj["nLists"].asInt()),
      static_cast<int32_t>(obj["nProbes"].asInt()),
      static_cast<int32_t>(obj["pqBits"].asInt()),
      static_cast<int32_t>(obj["pqDim"].asInt()),
      sources[0],
      sources[1],
      ISerializable::deserialize<RowType>(obj["outputType"]));
}

void CudfVectorJoinNode::accept(
    const core::PlanNodeVisitor& visitor,
    core::PlanNodeVisitorContext& context) const {
  visitor.visit(static_cast<const core::PlanNode&>(*this), context);
}

void CudfVectorJoinBridge::setData(std::optional<VectorJoinBuildData> data) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> guard{mutex_};
    VELOX_CHECK(!data_.has_value(), "Vector join bridge already has data");
    data_ = std::move(data);
    promises = std::move(promises_);
  }
  notify(std::move(promises));
}

std::optional<VectorJoinBuildData> CudfVectorJoinBridge::dataOrFuture(
    ContinueFuture* future) {
  std::lock_guard<std::mutex> guard{mutex_};
  VELOX_CHECK(
      !cancelled_, "Getting vector join data after the build side is aborted");
  if (data_.has_value()) {
    return data_;
  }
  promises_.emplace_back("CudfVectorJoinBridge::dataOrFuture");
  *future = promises_.back().getSemiFuture();
  return std::nullopt;
}

void CudfVectorJoinBridge::setBuildReadyEvent(
    std::shared_ptr<CudaEvent> buildReadyEvent) {
  std::lock_guard<std::mutex> guard{mutex_};
  buildReadyEvent_ = std::move(buildReadyEvent);
}

std::shared_ptr<CudaEvent> CudfVectorJoinBridge::getBuildReadyEvent() {
  std::lock_guard<std::mutex> guard{mutex_};
  return buildReadyEvent_;
}

void CudfVectorJoinBridge::setBuildStream(cuda::stream_ref buildStream) {
  std::lock_guard<std::mutex> guard{mutex_};
  buildStream_ = buildStream;
}

std::optional<cuda::stream_ref> CudfVectorJoinBridge::getBuildStream() {
  std::lock_guard<std::mutex> guard{mutex_};
  return buildStream_;
}

CudfVectorJoinBuild::CudfVectorJoinBuild(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const CudfVectorJoinNode> joinNode)
    : CudfJoinBuild(
          operatorId,
          driverCtx,
          joinNode,
          "CudfVectorJoinBuild",
          NvtxMethodFlag::kNoMoreInput),
      joinNode_(std::move(joinNode)) {}

void CudfVectorJoinBuild::buildAndPublish(std::vector<CudfVectorPtr> inputs) {
  auto stream = cudfGlobalStreamPool().get_stream();
  for (const auto& input : inputs) {
    CudaEvent ready(cudaEventDisableTiming);
    ready.recordFrom(input->stream()).waitOn(stream);
  }
  for (const auto& input : inputs) {
    VELOX_CHECK(
        input->rebindStream(stream),
        "Vector join build input cannot move onto the build stream");
  }

  const auto& config = CudfConfig::getInstance();
  int64_t rows = 0;
  uint64_t bytes = 0;
  for (const auto& input : inputs) {
    rows += input->size();
    bytes += input->estimateFlatSize();
  }
  const auto maxRows = operatorCtx_->driverCtx()->queryConfig().get<int64_t>(
      CudfConfig::kVectorJoinMaxShardRows, config.vectorJoinMaxShardRows);
  const auto maxBytes = operatorCtx_->driverCtx()->queryConfig().get<int64_t>(
      CudfConfig::kVectorJoinMaxShardBytes, config.vectorJoinMaxShardBytes);
  VELOX_USER_CHECK_LE(
      rows,
      maxRows,
      "Vector join build shard of {} rows exceeds {}. Repartition the build side",
      rows,
      maxRows);
  VELOX_USER_CHECK_LE(
      bytes,
      static_cast<uint64_t>(maxBytes),
      "Vector join build shard of {} bytes exceeds {}. Repartition the build side",
      bytes,
      maxBytes);

  const auto buildType = joinNode_->sources()[1]->outputType();
  const auto vectorChannel = channelOf(buildType, joinNode_->buildVector());
  const auto idChannel = channelOf(buildType, joinNode_->buildId());
  std::vector<RuntimeVectorIndex::Batch> views;
  std::vector<int64_t> rowStarts;
  int64_t rowStart = 0;
  views.reserve(inputs.size());
  for (const auto& input : inputs) {
    rowStarts.push_back(rowStart);
    views.push_back(
        RuntimeVectorIndex::Batch{
            input->getTableView().column(vectorChannel),
            input->getTableView().column(idChannel),
            rowStart});
    rowStart += input->size();
  }
  RuntimeVectorIndex::Options options;
  options.nLists = positiveOr(
      joinNode_->nLists(),
      operatorCtx_->driverCtx()->queryConfig().get<int32_t>(
          CudfConfig::kVectorJoinNLists, config.vectorJoinNLists));
  options.pqBits = positiveOr(
      joinNode_->pqBits(),
      operatorCtx_->driverCtx()->queryConfig().get<int32_t>(
          CudfConfig::kVectorJoinPqBits, config.vectorJoinPqBits));
  options.pqDim = joinNode_->pqDim() > 0
      ? joinNode_->pqDim()
      : operatorCtx_->driverCtx()->queryConfig().get<int32_t>(
            CudfConfig::kVectorJoinPqDim, config.vectorJoinPqDim);
  options.trainRows = std::max<int32_t>(
      1,
      operatorCtx_->driverCtx()->queryConfig().get<int32_t>(
          CudfConfig::kVectorJoinTrainRows, config.vectorJoinTrainRows));
  options.cosine = joinNode_->metric() == VectorJoinMetric::kCosine;
  options.largerIsBetter = joinNode_->metric() != VectorJoinMetric::kL2Squared;

  VectorJoinBuildData data;
  data.index = buildRuntimeVectorIndex(views, options, stream, get_temp_mr());
  data.batches = std::move(inputs);
  data.rowStarts = std::move(rowStarts);
  data.buildVectorChannel = vectorChannel;
  data.buildIdChannel = idChannel;
  data.k = joinNode_->k();
  data.candidateMultiplier = positiveOr(
      joinNode_->candidateMultiplier(),
      operatorCtx_->driverCtx()->queryConfig().get<int32_t>(
          CudfConfig::kVectorJoinCandidateMultiplier,
          config.vectorJoinCandidateMultiplier));
  data.nProbes = positiveOr(
      joinNode_->nProbes(),
      operatorCtx_->driverCtx()->queryConfig().get<int32_t>(
          CudfConfig::kVectorJoinNProbes, config.vectorJoinNProbes));

  auto ready = std::make_shared<CudaEvent>(cudaEventDisableTiming);
  ready->recordFrom(stream);
  auto bridge = std::dynamic_pointer_cast<CudfVectorJoinBridge>(
      operatorCtx_->task()->getCustomJoinBridge(
          operatorCtx_->driverCtx()->splitGroupId, planNodeId()));
  VELOX_CHECK_NOT_NULL(bridge);
  bridge->setBuildReadyEvent(std::move(ready));
  bridge->setBuildStream(stream);
  bridge->setData(std::move(data));
}

CudfVectorJoinProbe::CudfVectorJoinProbe(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const CudfVectorJoinNode> joinNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          joinNode->outputType(),
          joinNode->id(),
          "CudfVectorJoinProbe",
          nvtx3::rgb{70, 130, 180},
          NvtxMethodFlag::kAddInput | NvtxMethodFlag::kGetOutput |
              NvtxMethodFlag::kClose,
          std::nullopt,
          joinNode),
      joinNode_(std::move(joinNode)) {
  probeVectorChannel_ = channelOf(
      joinNode_->sources()[0]->outputType(), joinNode_->probeVector());
  probeBatchRows_ = std::max<int32_t>(
      1,
      operatorCtx_->driverCtx()->queryConfig().get<int32_t>(
          CudfConfig::kVectorJoinProbeBatchRows,
          CudfConfig::getInstance().vectorJoinProbeBatchRows));
  const auto outputRows =
      static_cast<int64_t>(probeBatchRows_) * joinNode_->k();
  VELOX_USER_CHECK_LE(
      outputRows,
      std::numeric_limits<int32_t>::max(),
      "Vector join probe batch of {} rows and k {} overflows the output batch",
      probeBatchRows_,
      joinNode_->k());
}

bool CudfVectorJoinProbe::needsInput() const {
  return !noMoreInput_ && input_ == nullptr && pending_.empty();
}

exec::BlockingReason CudfVectorJoinProbe::isBlocked(ContinueFuture* future) {
  if (buildData_.has_value()) {
    return exec::BlockingReason::kNotBlocked;
  }
  auto bridge = std::dynamic_pointer_cast<CudfVectorJoinBridge>(
      operatorCtx_->task()->getCustomJoinBridge(
          operatorCtx_->driverCtx()->splitGroupId, planNodeId()));
  VELOX_CHECK_NOT_NULL(bridge);
  buildData_ = bridge->dataOrFuture(future);
  if (!buildData_.has_value()) {
    return exec::BlockingReason::kWaitForJoinBuild;
  }
  buildReadyEvent_ = bridge->getBuildReadyEvent();
  buildStream_ = bridge->getBuildStream();
  return exec::BlockingReason::kNotBlocked;
}

bool CudfVectorJoinProbe::isFinished() {
  return noMoreInput_ && input_ == nullptr && pending_.empty();
}

void CudfVectorJoinProbe::doAddInput(RowVectorPtr input) {
  input_ = std::move(input);
}

void CudfVectorJoinProbe::waitForBuildReady(cuda::stream_ref probeStream) {
  if (buildReadyEvent_) {
    buildReadyEvent_->waitOn(probeStream);
  }
}

void CudfVectorJoinProbe::recordReadCompletion(cuda::stream_ref probeStream) {
  if (!buildStream_.has_value()) {
    return;
  }
  if (!readEvent_) {
    readEvent_ = std::make_unique<CudaEvent>(cudaEventDisableTiming);
  }
  readEvent_->recordFrom(probeStream).waitOn(*buildStream_);
}

RowVectorPtr CudfVectorJoinProbe::searchBatch(
    const cudf::table_view& input,
    vector_size_t size,
    cuda::stream_ref stream) {
  const auto mr = get_temp_mr();
  if (size == 0 || buildData_->index->indexedRows() == 0) {
    return nullptr;
  }
  auto packed = packQueryVectors(
      input.column(probeVectorChannel_),
      buildData_->index->dimension(),
      joinNode_->metric() == VectorJoinMetric::kCosine,
      stream,
      mr);
  if (packed.count == 0) {
    return nullptr;
  }
  const auto candidateK = static_cast<int32_t>(std::min<int64_t>(
      buildData_->index->indexedRows(),
      std::max<int64_t>(
          buildData_->k,
          static_cast<int64_t>(buildData_->k) *
              buildData_->candidateMultiplier)));
  auto hits = buildData_->index->search(
      packed.values->data(),
      packed.rowIds->data(),
      packed.count,
      buildData_->k,
      std::max(candidateK, buildData_->k),
      buildData_->nProbes,
      stream,
      mr);
  std::vector<VectorJoinHit> kept;
  kept.reserve(hits.size());
  for (const auto& hit : hits) {
    if (hit.buildRow >= 0) {
      kept.push_back(hit);
    }
  }
  if (kept.empty()) {
    return nullptr;
  }

  const auto probeType = joinNode_->sources()[0]->outputType();
  const auto buildType = joinNode_->sources()[1]->outputType();
  std::vector<int32_t> probeRows;
  std::vector<int32_t> buildBatches;
  std::vector<int32_t> localRows;
  std::vector<int32_t> slots;
  std::vector<float> scores;
  probeRows.reserve(kept.size());
  for (int32_t i = 0; i < static_cast<int32_t>(kept.size()); ++i) {
    probeRows.push_back(kept[i].probeRow);
    const auto batch = batchOf(buildData_->rowStarts, kept[i].buildRow);
    buildBatches.push_back(batch);
    localRows.push_back(
        static_cast<int32_t>(kept[i].buildRow - buildData_->rowStarts[batch]));
    slots.push_back(i);
    scores.push_back(kept[i].score);
  }
  auto probeMap = intColumn(probeRows, stream, mr);
  auto gatheredProbe = cudf::gather(
      input,
      probeMap->view(),
      cudf::out_of_bounds_policy::DONT_CHECK,
      stream,
      cudf::memory_resources{mr, mr});
  auto probeColumns = gatheredProbe->release();

  std::vector<std::unique_ptr<cudf::column>> output(outputType_->size());
  for (int32_t i = 0; i < outputType_->size(); ++i) {
    const auto& name = outputType_->nameOf(i);
    if (name == joinNode_->score()) {
      output[i] = scoreColumn(scores, stream, mr);
    } else if (probeType->containsChild(name)) {
      output[i] = std::move(probeColumns[channelOf(probeType, name)]);
    } else {
      output[i] = gatherBuildColumn(
          *buildData_,
          channelOf(buildType, name),
          buildBatches,
          localRows,
          slots,
          stream,
          mr);
    }
  }
  auto table = std::make_unique<cudf::table>(std::move(output));
  return std::make_shared<CudfVector>(
      pool(),
      outputType_,
      static_cast<vector_size_t>(kept.size()),
      std::move(table),
      stream);
}

RowVectorPtr CudfVectorJoinProbe::doGetOutput() {
  if (!pending_.empty()) {
    auto output = std::move(pending_.front());
    pending_.erase(pending_.begin());
    return output;
  }
  if (input_ == nullptr) {
    return nullptr;
  }
  auto input = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(input, "Vector join probe expects cuDF input");
  auto stream = cudfGlobalStreamPool().get_stream();
  CudaEvent inputReady(cudaEventDisableTiming);
  inputReady.recordFrom(input->stream()).waitOn(stream);
  VELOX_CHECK(
      input->rebindStream(stream),
      "Vector join probe input cannot move onto the probe stream");
  waitForBuildReady(stream);

  const auto rows = input->size();
  for (vector_size_t offset = 0; offset < rows;) {
    const auto end = static_cast<vector_size_t>(std::min<int64_t>(
        static_cast<int64_t>(offset) + probeBatchRows_, rows));
    auto slices = cudf::slice(
        input->getTableView(),
        {static_cast<cudf::size_type>(offset),
         static_cast<cudf::size_type>(end)},
        stream);
    if (auto output = searchBatch(slices[0], end - offset, stream)) {
      pending_.push_back(std::move(output));
    }
    offset = end;
  }
  recordReadCompletion(stream);
  input_.reset();
  if (pending_.empty()) {
    return nullptr;
  }
  auto output = std::move(pending_.front());
  pending_.erase(pending_.begin());
  return output;
}

void CudfVectorJoinProbe::doClose() {
  input_.reset();
  pending_.clear();
  buildData_.reset();
  Operator::close();
}

std::unique_ptr<exec::Operator> CudfVectorJoinBridgeTranslator::toOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  if (auto join = std::dynamic_pointer_cast<const CudfVectorJoinNode>(node)) {
    return std::make_unique<CudfVectorJoinProbe>(id, ctx, join);
  }
  return nullptr;
}

std::unique_ptr<exec::JoinBridge> CudfVectorJoinBridgeTranslator::toJoinBridge(
    const core::PlanNodePtr& node) {
  if (std::dynamic_pointer_cast<const CudfVectorJoinNode>(node)) {
    return std::make_unique<CudfVectorJoinBridge>();
  }
  return nullptr;
}

exec::OperatorSupplier CudfVectorJoinBridgeTranslator::toOperatorSupplier(
    const core::PlanNodePtr& node) {
  if (auto join = std::dynamic_pointer_cast<const CudfVectorJoinNode>(node)) {
    return [join](int32_t operatorId, exec::DriverCtx* ctx) {
      return std::make_unique<CudfVectorJoinBuild>(operatorId, ctx, join);
    };
  }
  return nullptr;
}

void registerCudfVectorJoin() {
  exec::Operator::registerOperator(
      std::make_unique<CudfVectorJoinBridgeTranslator>());
  DeserializationWithContextRegistryForSharedPtr().Register(
      "CudfVectorJoinNode", CudfVectorJoinNode::create);
  OperatorAdapterRegistry::getInstance().registerAdapter(
      std::make_unique<VectorJoinAdapter>());
}

} // namespace facebook::velox::cudf_velox
