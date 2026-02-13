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
#pragma once

#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"
#include "velox/exec/ProbeOperatorState.h"
#include "velox/exec/SpatialIndex.h"

namespace facebook::velox::exec {

class SpatialJoinOutputBuilder {
 public:
  SpatialJoinOutputBuilder(
      vector_size_t outputBatchSize,
      RowTypePtr outputType,
      std::vector<IdentityProjection> probeProjections,
      std::vector<IdentityProjection> buildProjections,
      const OperatorCtx& operatorCtx)
      : outputBatchSize_{outputBatchSize},
        outputType_{std::move(outputType)},
        probeProjections_{std::move(probeProjections)},
        buildProjections_{std::move(buildProjections)},
        operatorCtx_{operatorCtx} {
    VELOX_CHECK_GT(outputBatchSize_, 0);
  }

  void initializeOutput(const RowVectorPtr& input, memory::MemoryPool* pool);

  bool isOutputFull() const {
    return outputRow_ >= outputBatchSize_;
  }

  void addOutputRow(vector_size_t probeRow, vector_size_t buildRow);

  /// Checks if it is required to add a probe mismatch row, and does it if
  /// needed. The caller needs to ensure there is available space in `output_`
  /// for the new record, which has nulled out build projections.
  void addProbeMismatchRow(vector_size_t probeRow);

  void copyBuildValues(const RowVectorPtr& buildVector);

  RowVectorPtr takeOutput();

 private:
  // Initialization parameters
  const vector_size_t outputBatchSize_;
  const RowTypePtr outputType_;
  const std::vector<IdentityProjection> probeProjections_;
  const std::vector<IdentityProjection> buildProjections_;
  const OperatorCtx& operatorCtx_;

  // Output state
  RowVectorPtr output_;
  vector_size_t outputRow_{0};
  // Dictionary indices for probe columns for output vector.
  BufferPtr probeOutputIndices_;
  // Mutable pointer to probeOutputIndices_
  vector_size_t* rawProbeOutputIndices_{};

  // Stores the ranges of build values to be copied to the output vector (we
  // batch them and copy once, instead of copying them row-by-row).
  std::vector<BaseVector::CopyRange> buildCopyRanges_{};
};

/// Implements a Spatial Join between records from the probe (input_)
/// and build (SpatialJoinBridge) sides. It supports inner and left joins.
///
/// This class is designed to evaluate spatial join conditions (e.g.
/// ST_INTERSECTS, ST_CONTAINS, ST_WITHIN) between geometric data types. It
/// can also implement spatial cross-join semantics if joinCondition is
/// nullptr.
///
/// The output follows the order of the probe side rows (for inner and left
/// joins). All build vectors are materialized upfront (check buildVectors_),
/// but probe batches are processed one-by-one as a stream.
///
/// To produce output, the operator processes each probe record from probe
/// input, using the following steps:
///
/// 1. Materialize a cross-product batch across probe and build.
/// 2. Evaluate the spatial join condition.
/// 3. Add spatial matches to the output.
/// 4. Once all build vectors are processed for a particular probe row, check
/// if
///    a probe mismatch is needed (only for left and full outer joins).
/// 5. Once all probe and build inputs are processed, check if build
/// mismatches
///    are needed (only for right and full outer joins).
/// 6. If so, signal other peer operators; only a single operator instance
/// will
///    collect all build matches at the end, and emit any records that haven't
///    been matched by any of the peers.
///
/// Spatial joins typically use spatial indexing for performance optimization,
/// but this implementation follows the nested loop pattern for compatibility
/// with the existing join framework.
class SpatialJoinProbe : public Operator {
 public:
  SpatialJoinProbe(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::SpatialJoinNode>& joinNode);

  void initialize() override;

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
  void checkStateTransition(ProbeOperatorState state);

  void setState(ProbeOperatorState state) {
    checkStateTransition(state);
    state_ = state;
  }

  // Initialize spatial filter for evaluating spatial join conditions.
  void initializeFilter(
      const core::TypedExprPtr& filter,
      const RowTypePtr& leftType,
      const RowTypePtr& rightType);

  // Materializes build data from spatial join bridge into `buildVectors_`.
  // Returns whether the data has been materialized and is ready for use.
  // Spatial join requires all build data to be materialized and available in
  // `buildVectors_` before it can produce output.
  bool getBuildData(ContinueFuture* future);

  // Produce as much output as possible for the current input.
  RowVectorPtr generateOutput();

  // Returns true if the input is exhausted or the output is full.
  bool isOutputDone() const {
    return probeRow_ >= input_->size() || outputBuilder_.isOutputFull();
  }

  // Called when we are done processing the current probe batch, to signal we
  // are ready for the next one.
  //
  // If this is the last probe batch (and this is a right or full outer join),
  // change the operator state to signal peers.
  void finishProbeInput();

  // Add the output for a single probe row.  This will return early if the
  // output vector is full.
  void addProbeRowOutput();

  // Returns true if all output for the current probe row has been produced.
  bool isProbeRowDone() const {
    return candidateIndex_ >= candidateBuildRows_.size() ||
        buildVectorIndex_ >= buildVectors_.value().size();
  }

  // Increment probeRow_ and reset associated fields
  void advanceProbeRow() {
    ++probeRow_;
    probeHasMatch_ = false;
    buildVectorIndex_ = 0;
    candidateIndex_ = 0;
    candidateOffsetForCurrentBuildVector_ = 0;
    buildRowOffset_ = 0;
    needsFilterEvaluated_ = true;
  }

  // Add the output for a single build vector for a single probe row.  This will
  // return early if the output vector is full.
  void addBuildVectorOutput(const RowVectorPtr& buildVector);

  // Returns true if all the rows for the current build vector have been
  // processed, or the output is full.
  bool isBuildVectorDone(const RowVectorPtr& buildVector) const {
    // Note that candidateBuildRows_ entries are row numbers across
    // all build vectors.
    return candidateIndex_ >= candidateBuildRows_.size() ||
        relativeBuildRow(candidateIndex_) >= buildVector->size() ||
        outputBuilder_.isOutputFull();
  }

  // Increment buildVectorIndex_ and reset associated fields
  void advanceBuildVector() {
    VELOX_CHECK(buildVectors_.has_value());

    buildRowOffset_ += buildVectors_.value()[buildVectorIndex_]->size();
    ++buildVectorIndex_;
    needsFilterEvaluated_ = true;
    candidateOffsetForCurrentBuildVector_ = candidateIndex_;
  }

  // Calculate candidate build rows from spatialIndex_ for the current probe
  // row. This should be done each time the probe is advanced.
  std::vector<int32_t> querySpatialIndex();

  // Evaluates the spatial joinCondition for a given build vector. This method
  // sets `filterOutput_` and `decodedFilterResult_`, which will be ready to
  // be used by `isSpatialJoinConditionMatch()` below.
  // This only evaluates rows that are in the candidateBuildRows_, restricted to
  // those in the current build vector. Thus we must index into this with
  // candidateIndex_.
  void evaluateJoinFilter(const RowVectorPtr& buildVector);

  // Checks if the spatial join condition matched for a particular row.
  bool isJoinConditionMatch(vector_size_t candidateIndex) const {
    vector_size_t relativeIndex =
        candidateIndex - candidateOffsetForCurrentBuildVector_;
    VELOX_CHECK_GT(decodedFilterResult_.size(), relativeIndex);
    return (
        !decodedFilterResult_.isNullAt(relativeIndex) &&
        decodedFilterResult_.valueAt<bool>(relativeIndex));
  }

  // Generates the next batch of a cross product between probe and build using
  // the supplied projections. It uses the current probe row as constant, and
  // flat copied data for build records.
  RowVectorPtr getNextJoinBatch(
      const RowVectorPtr& buildVector,
      const RowTypePtr& outputType,
      const std::vector<IdentityProjection>& probeProjections,
      const std::vector<IdentityProjection>& buildProjections,
      BufferPtr candidateRows) const;

  // Given a candidate index, return the row index into the current build
  // vector.  For example, if we have candidates [2, 50, 81] and have processed
  // two build vectors with size 30 and 40, then `relativeBuildRow(2) == 11`
  // (81 - (30 + 40)).
  vector_size_t relativeBuildRow(vector_size_t candidateRow) const {
    return candidateBuildRows_[candidateRow] - buildRowOffset_;
  }

  // Make the indices of build vector candidates suitable for creating a
  // DictionaryVector.
  BufferPtr makeBuildVectorIndices(vector_size_t vectorSize);

  /////////
  // SETUP
  // Variables set during operator setup that are used during execution.
  // These should not be modified after the operator is initialized.

  const core::JoinType joinType_;

  // Maximum number of rows in the output batch.
  const vector_size_t outputBatchSize_;

  // Join metadata and state.
  std::shared_ptr<const core::SpatialJoinNode> joinNode_;

  // Spatial join condition expression.
  // Must not be null
  std::unique_ptr<ExprSet> joinCondition_;

  // Input type for the spatial join condition expression.
  RowTypePtr filterInputType_;

  // List of output projections from the build side. Note that the list of
  // projections from the probe side is available at `identityProjections_`.
  std::vector<IdentityProjection> buildProjections_;

  // Projections needed as input to the filter to evaluation spatial join
  // filter conditions. Note that if this is a cross-join, filter projections
  // are the same as output projections.
  std::vector<IdentityProjection> filterProbeProjections_;
  std::vector<IdentityProjection> filterBuildProjections_;

  // Stores the build spatial index for the join
  std::optional<std::shared_ptr<SpatialIndex>> spatialIndex_;
  // Stores the data for build vectors (right side of the join).
  std::optional<std::vector<RowVectorPtr>> buildVectors_;

  // Channel of geometry variable used to probe spatial index
  column_index_t probeGeometryChannel_;

  //////////////////
  // OPERATOR STATE
  // Variables used to track the general operator state during exection.
  // These will change throughout setup and execution.

  ProbeOperatorState state_{ProbeOperatorState::kWaitForBuild};
  ContinueFuture future_{ContinueFuture::makeEmpty()};

  // The information needed to produce an output RowVectorPtr. It is stored
  // for all execution, but is reset on each output batch.
  SpatialJoinOutputBuilder outputBuilder_;

  // Count of output batches produced (1-indexed). Primarily for debugging.
  size_t outputCount_{0};

  // This is always set to all true, but we need it for eval/etc. Reuse between
  // evaluations.
  SelectivityVector filterInputRows_;
  // The output result of the join condition evaluation on the **current**
  // build vector. We must index into this with
  // `candidateIndex_ - candidateOffsetForCurrentBuildVector_`.
  VectorPtr filterOutput_;
  // Decoded filterOutput: remove recursive dictionary/etc encodings.
  // Like filterOutput_, this is only for the current build vector and we
  // must index into this with
  // `candidateIndex_ - candidateOffsetForCurrentBuildVector_`.
  DecodedVector decodedFilterResult_;

  // Decoded geometry vector.  Must be reset whenever input_ is changed (it
  // maintains a pointer to input_).
  DecodedVector decodedGeometryCol_{};

  ///////////////
  // PROBE STATE
  // Variables used to track the probe-side state state during exection.
  // These will change throughout setup and execution.

  // Count of probe batches added (1-indexed). Primarily for debugging.
  size_t probeCount_{0};

  // Probe row being currently processed (related to `input_`).
  vector_size_t probeRow_{0};

  // Whether the current probeRow_ has found a match.  Needed for left join.
  bool probeHasMatch_{false};

  ///////////////
  // BUILD STATE
  // Variables used to track the build-side state state during exection.
  // These will change throughout setup and execution.
  //
  // The build rows are stored in a vector of RowVectorPtrs.  These are
  // conceptually indexed by an absolute build row, which indexes into a
  // flattened vector of rows.  buildVectorIndex_ is the index to the current
  // RowVectorPtr in buildVectors_, buildRowOffset_ is the sum of the sizes
  // of the previous build vectors and should be subtracted from buildRow
  // to index into the current build vector.
  //
  // We primarily use candidateBuildRows_, which is a vector of (absolute)
  // build rows.  candidateIndex_ indexes the entry in candidateBuildRows_,
  // so candidateBuildrows_[candidiateIndex_] is the absolute build row
  // of the current candidate.

  // Whether we need to evaluate the join filter on this build vector.  It
  // should be done once per build vector/probe row pair.
  bool needsFilterEvaluated_{true};

  // Index into `buildVectors_` for the build vector being currently
  // processed.
  size_t buildVectorIndex_{0};

  // Keep track of how many build rows we've traversed in previous build
  // RowVectors. Subtract this from the current element in candidateBuildRows_
  // to index into the current build RowVector.
  vector_size_t buildRowOffset_{0};

  // Build rows returned from the spatial index.
  // The value is the row number over all build vectors, so if the have two
  // build vectors of size 100 and 200, candidate row 50 is the 50th entry of
  // the first vector, and 101 is the 2nd entry of the second vector.
  std::vector<vector_size_t> candidateBuildRows_{};

  // Index of candidate currently being processed from
  // `buildVectors_[buildIndex_]`.
  vector_size_t candidateIndex_{0};

  // How many candidates were in previous build vectors.
  // This is important because for each build vector, we calculate a
  // decodedFilterResult_ with only the rows from from the candidates in
  // that build vector.  candidateIndex_ indexes over _all_ candidates, so
  // we must substract candidateOffsetForCurrentBuildVector_ to index into the
  // candidates for this build vector.
  vector_size_t candidateOffsetForCurrentBuildVector_{0};
};

} // namespace facebook::velox::exec
