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
#pragma once

#include "velox/experimental/cudf/exec/CudfJoinBuild.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/RuntimeVectorIndex.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/JoinBridge.h"
#include "velox/exec/Operator.h"

#include <string>

namespace facebook::velox::cudf_velox {

class CudaEvent;

/// Approximate top-k inner join between a probe vector column and a build
/// vector column. This is an experimental plan node registered by cuDF. It is
/// not a core Velox operator: the translator is installed from registerCudf(),
/// and builds without VELOX_ENABLE_FAISS_GPU do not contain it.
///
/// Each task indexes only the build rows it receives. The index is a cuVS
/// IVF-PQ index trained on a bounded sample and extended one batch at a time,
/// or an exact scan when the shard is too small to cluster. Search reranks
/// candidates against the original vectors, so scores from different tasks can
/// be merged. A shard that exceeds cudf.vector_join_max_shard_rows or
/// cudf.vector_join_max_shard_bytes fails instead of growing without a bound.
///
/// Distributed top-k is the planner's job. It assigns a globally unique probe
/// id, places each build row in exactly one shard, and delivers every probe to
/// every shard that should see it (broadcast, or a coarse routing index). Each
/// task emits its local top-k. The planner then hash-partitions those rows by
/// probe id and applies TopNRowNumber(k), ordered by score and build id.
/// L2 orders by ascending score. Inner product and cosine order by descending
/// score. Equal scores keep the smaller build id.
enum class VectorJoinMetric {
  kL2Squared,
  kInnerProduct,
  kCosine,
};

class CudfVectorJoinNode : public core::PlanNode {
 public:
  CudfVectorJoinNode(
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
      RowTypePtr outputType);

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  std::string_view name() const override {
    return "CudfVectorJoin";
  }

  const std::string& probeVector() const {
    return probeVector_;
  }

  const std::string& buildVector() const {
    return buildVector_;
  }

  const std::string& buildId() const {
    return buildId_;
  }

  const std::string& score() const {
    return score_;
  }

  VectorJoinMetric metric() const {
    return metric_;
  }

  int32_t k() const {
    return k_;
  }

  int32_t candidateMultiplier() const {
    return candidateMultiplier_;
  }

  int32_t nLists() const {
    return nLists_;
  }

  int32_t nProbes() const {
    return nProbes_;
  }

  int32_t pqBits() const {
    return pqBits_;
  }

  int32_t pqDim() const {
    return pqDim_;
  }

  folly::dynamic serialize() const override;

  static core::PlanNodePtr create(const folly::dynamic& obj, void* context);

  void accept(
      const core::PlanNodeVisitor& visitor,
      core::PlanNodeVisitorContext& context) const override;

 private:
  void addDetails(std::stringstream& stream) const override;

  const std::string probeVector_;
  const std::string buildVector_;
  const std::string buildId_;
  const std::string score_;
  const VectorJoinMetric metric_;
  const int32_t k_;
  const int32_t candidateMultiplier_;
  const int32_t nLists_;
  const int32_t nProbes_;
  const int32_t pqBits_;
  const int32_t pqDim_;
  const std::vector<core::PlanNodePtr> sources_;
  const RowTypePtr outputType_;
};

struct VectorJoinBuildData {
  std::shared_ptr<RuntimeVectorIndex> index;
  std::vector<CudfVectorPtr> batches;
  std::vector<int64_t> rowStarts;
  int32_t buildVectorChannel{0};
  int32_t buildIdChannel{0};
  int32_t k{1};
  int32_t candidateMultiplier{4};
  int32_t nProbes{1};
};

class CudfVectorJoinBridge : public exec::JoinBridge {
 public:
  void setData(std::optional<VectorJoinBuildData> data);
  std::optional<VectorJoinBuildData> dataOrFuture(ContinueFuture* future);
  void setBuildReadyEvent(std::shared_ptr<CudaEvent> buildReadyEvent);
  std::shared_ptr<CudaEvent> getBuildReadyEvent();
  void setBuildStream(cuda::stream_ref buildStream);
  std::optional<cuda::stream_ref> getBuildStream();

 private:
  std::optional<VectorJoinBuildData> data_;
  std::shared_ptr<CudaEvent> buildReadyEvent_;
  std::optional<cuda::stream_ref> buildStream_;
};

class CudfVectorJoinBuild : public CudfJoinBuild {
 public:
  CudfVectorJoinBuild(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const CudfVectorJoinNode> joinNode);

 protected:
  void buildAndPublish(std::vector<CudfVectorPtr> inputs) override;

 private:
  std::shared_ptr<const CudfVectorJoinNode> joinNode_;
};

class CudfVectorJoinProbe : public CudfOperatorBase {
 public:
  CudfVectorJoinProbe(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const CudfVectorJoinNode> joinNode);

  bool needsInput() const override;
  exec::BlockingReason isBlocked(ContinueFuture* future) override;
  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doClose() override;

 private:
  RowVectorPtr searchBatch(
      const cudf::table_view& input,
      vector_size_t size,
      cuda::stream_ref stream);
  void waitForBuildReady(cuda::stream_ref probeStream);
  void recordReadCompletion(cuda::stream_ref probeStream);

  std::shared_ptr<const CudfVectorJoinNode> joinNode_;
  std::optional<VectorJoinBuildData> buildData_;
  std::shared_ptr<CudaEvent> buildReadyEvent_;
  std::optional<cuda::stream_ref> buildStream_;
  std::unique_ptr<CudaEvent> readEvent_;
  RowVectorPtr input_;
  std::vector<RowVectorPtr> pending_;
  int32_t probeVectorChannel_{0};
  int32_t probeBatchRows_{2048};
};

class CudfVectorJoinBridgeTranslator
    : public exec::Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<exec::Operator> toOperator(
      exec::DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override;

  std::unique_ptr<exec::JoinBridge> toJoinBridge(
      const core::PlanNodePtr& node) override;

  exec::OperatorSupplier toOperatorSupplier(
      const core::PlanNodePtr& node) override;
};

void registerCudfVectorJoin();

} // namespace facebook::velox::cudf_velox
