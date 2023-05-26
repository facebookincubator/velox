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

#include "velox/exec/NestedLoopJoinBuild.h"
#include "velox/exec/Operator.h"
#include "velox/exec/ProbeOperatorState.h"

namespace facebook::velox::exec {
class NestedLoopJoinProbe : public Operator {
 public:
  NestedLoopJoinProbe(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::NestedLoopJoinNode>& joinNode);

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  bool needsInput() const override {
    return state_ == ProbeOperatorState::kRunning && input_ == nullptr &&
        !noMoreInput_;
  }

  void noMoreInput() override;

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override {
    return state_ == ProbeOperatorState::kFinish;
  }

  void close() override;

 private:
  // TODO: maybe consolidate initializeFilter routine across Operators like
  // HashProbe and MergeJoin
  void initializeFilter(
      const core::TypedExprPtr& filter,
      const RowTypePtr& leftType,
      const RowTypePtr& rightType);

  bool getBuildData(ContinueFuture* future);

  // Get probe side rows count to process on next call to getCrossProduct()
  vector_size_t getProbeCnt() const;

  // Get cross product of next 'probeCnt' rows of input_ and all rows of
  // buildData_.value()[buildIndex].
  // 'outputType' specifies the type of output.
  // Projections from input_ and buildData_ to the output are specified by
  // 'probeProjections' and 'buildProjections', respectively. Caller is
  // responsible for ensuring all columns in outputType is contained in either
  // projections.
  // TODO: consider consolidate the routine of producing cartesian product that
  // can be reused at MergeJoin::addToOutput
  RowVectorPtr getCrossProduct(
      vector_size_t probeCnt,
      const RowTypePtr& outputType,
      const std::vector<IdentityProjection>& probeProjections,
      const std::vector<IdentityProjection>& buildProjections);

  // Evaluates joinCondition against the output of getCrossProduct(probeCnt),
  // returns the result that passed joinCondition, updates probeMatched_,
  // buildMatched_ accordingly.
  RowVectorPtr doMatching(vector_size_t probeCnt);

  // Update probeRow_ and buildIndex_ by advance probeRow_ by probeCnt.
  // Return true if buildIndex_ is points to the end of buildData_.
  bool advanceIndex(vector_size_t probeCnt);

  bool hasProbedAllBuildData() const {
    return (buildIndex_ == buildData_.value().size());
  }

  // Wraps rows of 'data' that are not selected in 'matched' and projects
  // to the children of output according to 'projections'. Creates null vector
  // as output's children according to 'nullProjections'. 'mapping' is used as
  // buffer for recording mismatched row numbers.
  RowVectorPtr getMismatchedOutput(
      RowVectorPtr data,
      const SelectivityVector& matched,
      BufferPtr mapping,
      const std::vector<IdentityProjection>& projections,
      const std::vector<IdentityProjection>& nullProjections);

  void finishMatchAndProbeMismatch();

  // When doing right/full joins, all but last probe operators that finished
  // matching and probe-side mismatch output, will block on kWaitForPeers state.
  // The last finishing operator will gather buildMatched from all probe
  // operators to emit output for mismatched build side rows, and notify other
  // probe operators to finish.
  void beginBuildMismatch();

  void setState(ProbeOperatorState s) {
    state_ = s;
  }

 private:
  // Maximum number of rows in the output batch.
  const uint32_t outputBatchSize_;
  const core::JoinType joinType_;

  ProbeOperatorState state_{ProbeOperatorState::kWaitForBuild};
  ContinueFuture future_{ContinueFuture::makeEmpty()};

  // Join condition-related state
  std::unique_ptr<ExprSet> joinCondition_;
  RowTypePtr filterInputType_;
  SelectivityVector filterInputRows_;

  // Probe side state
  // Input row to process on next call to getOutput().
  vector_size_t probeRow_{0};
  // probeCnt used in the last call to getCrossProduct(). Gets reset upon every
  // buildIndex_ updates.
  vector_size_t prevProbeCnt_{0};
  bool lastProbe_{false};
  // Represents whether probe side rows have been matched.
  SelectivityVector probeMatched_;
  std::vector<IdentityProjection> filterProbeProjections_;
  BufferPtr probeOutMapping_;
  BufferPtr probeIndices_;

  // Build side state
  std::optional<std::vector<RowVectorPtr>> buildData_;
  bool buildSideEmpty_{false};
  // Index into buildData_ for the build side vector to process on next call to
  // getOutput().
  size_t buildIndex_{0};
  std::vector<IdentityProjection> buildProjections_;
  BufferPtr buildIndices_;

  // Represents whether probe build rows have been matched.
  std::vector<SelectivityVector> buildMatched_;
  std::vector<IdentityProjection> filterBuildProjections_;
  BufferPtr buildOutMapping_;
};

} // namespace facebook::velox::exec
