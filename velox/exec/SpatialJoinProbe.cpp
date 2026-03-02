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
#include "velox/exec/SpatialJoinProbe.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/SpatialJoinBuild.h"
#include "velox/exec/Task.h"
#include "velox/expression/FieldReference.h"

namespace facebook::velox::exec {
namespace {

bool needsProbeMismatch(core::JoinType joinType) {
  return isLeftJoin(joinType);
}

std::vector<IdentityProjection> extractProjections(
    const RowTypePtr& srcType,
    const RowTypePtr& destType) {
  std::vector<IdentityProjection> projections;
  for (auto i = 0; i < srcType->size(); ++i) {
    auto name = srcType->nameOf(i);
    auto outIndex = destType->getChildIdxIfExists(name);
    if (outIndex.has_value()) {
      projections.emplace_back(i, outIndex.value());
    }
  }
  return projections;
}

} // namespace

//////////////////
// OUTPUT BUILDER

void SpatialJoinOutputBuilder::initializeOutput(
    const RowVectorPtr& input,
    memory::MemoryPool* pool) {
  if (output_ == nullptr) {
    output_ =
        BaseVector::create<RowVector>(outputType_, outputBatchSize_, pool);
  } else {
    VectorPtr outputVector = std::move(output_);
    BaseVector::prepareForReuse(outputVector, outputBatchSize_);
    output_ = std::static_pointer_cast<RowVector>(outputVector);
  }
  probeOutputIndices_ = allocateIndices(outputBatchSize_, pool);
  rawProbeOutputIndices_ = probeOutputIndices_->asMutable<vector_size_t>();

  // Add probe side projections as dictionary vectors
  for (const auto& projection : probeProjections_) {
    output_->childAt(projection.outputChannel) = wrapChild(
        outputBatchSize_,
        probeOutputIndices_,
        input->childAt(projection.inputChannel));
  }

  // Add build side projections as uninitialized vectors
  for (const auto& projection : buildProjections_) {
    auto child = output_->childAt(projection.outputChannel);
    if (child == nullptr) {
      child = BaseVector::create(
          outputType_->childAt(projection.outputChannel),
          outputBatchSize_,
          operatorCtx_.pool());
    }
  }
}

void SpatialJoinOutputBuilder::addOutputRow(
    vector_size_t probeRow,
    vector_size_t buildRow) {
  VELOX_CHECK_NOT_NULL(probeOutputIndices_);
  // Probe side is always a dictionary; just populate the index.
  rawProbeOutputIndices_[outputRow_] = probeRow;

  // For the build side, we accumulate the ranges to copy, then copy all of
  // them at once. Consecutive records are copied in one memcpy.
  if (!buildCopyRanges_.empty() &&
      (buildCopyRanges_.back().sourceIndex + buildCopyRanges_.back().count) ==
          buildRow) {
    ++buildCopyRanges_.back().count;
  } else {
    buildCopyRanges_.push_back({buildRow, outputRow_, 1});
  }
  ++outputRow_;
}

void SpatialJoinOutputBuilder::copyBuildValues(
    const RowVectorPtr& buildVector) {
  if (buildCopyRanges_.empty()) {
    return;
  }

  VELOX_CHECK_NOT_NULL(output_);

  for (const auto& projection : buildProjections_) {
    const auto& buildChild = buildVector->childAt(projection.inputChannel);
    const auto& outputChild = output_->childAt(projection.outputChannel);
    outputChild->copyRanges(buildChild.get(), buildCopyRanges_);
  }
  buildCopyRanges_.clear();
}

void SpatialJoinOutputBuilder::addProbeMismatchRow(vector_size_t probeRow) {
  VELOX_CHECK_NOT_NULL(output_);

  // Probe side is always a dictionary; just populate the index.
  rawProbeOutputIndices_[outputRow_] = probeRow;

  // Null out build projections.
  for (const auto& projection : buildProjections_) {
    const auto& outputChild = output_->childAt(projection.outputChannel);
    outputChild->setNull(outputRow_, true);
  }
  ++outputRow_;
}

RowVectorPtr SpatialJoinOutputBuilder::takeOutput() {
  VELOX_CHECK(buildCopyRanges_.empty());
  if (outputRow_ == 0 || !output_) {
    return nullptr;
  }
  RowVectorPtr output = std::move(output_);
  output->resize(outputRow_);
  output_ = nullptr;
  outputRow_ = 0;
  return output;
}

////////////////////
// SpatialJoinProbe

SpatialJoinProbe::SpatialJoinProbe(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::SpatialJoinNode>& joinNode)
    : Operator(
          driverCtx,
          joinNode->outputType(),
          operatorId,
          joinNode->id(),
          "SpatialJoinProbe"),
      joinType_(joinNode->joinType()),
      outputBatchSize_{outputBatchRows()},
      joinNode_(joinNode),
      buildProjections_(extractProjections(
          joinNode_->rightNode()->outputType(),
          outputType_)),
      outputBuilder_{
          outputBatchSize_,
          outputType_,
          extractProjections(
              joinNode_->leftNode()->outputType(),
              outputType_), // these are the identity Projections
          buildProjections_,
          *operatorCtx_} {
  auto probeType = joinNode_->leftNode()->outputType();
  identityProjections_ = extractProjections(probeType, outputType_);
  probeGeometryChannel_ =
      probeType->getChildIdx(joinNode_->probeGeometry()->name());
  VELOX_CHECK_EQ(
      probeType->childAt(probeGeometryChannel_),
      joinNode_->probeGeometry()->type());
}

/////////
// SETUP

void SpatialJoinProbe::initialize() {
  Operator::initialize();

  VELOX_CHECK_NOT_NULL(joinNode_);
  if (joinNode_->joinCondition() != nullptr) {
    initializeFilter(
        joinNode_->joinCondition(),
        joinNode_->leftNode()->outputType(),
        joinNode_->rightNode()->outputType());
  }

  joinNode_.reset();
}

void SpatialJoinProbe::initializeFilter(
    const core::TypedExprPtr& filter,
    const RowTypePtr& probeType,
    const RowTypePtr& buildType) {
  VELOX_CHECK_NULL(joinCondition_);

  std::vector<core::TypedExprPtr> filters = {filter};
  joinCondition_ =
      std::make_unique<ExprSet>(std::move(filters), operatorCtx_->execCtx());

  column_index_t filterChannel = 0;
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  const auto numFields = joinCondition_->expr(0)->distinctFields().size();
  names.reserve(numFields);
  types.reserve(numFields);

  for (const auto& field : joinCondition_->expr(0)->distinctFields()) {
    const auto& name = field->field();
    auto channel = probeType->getChildIdxIfExists(name);
    if (channel.has_value()) {
      auto channelValue = channel.value();
      filterProbeProjections_.emplace_back(channelValue, filterChannel++);
      names.emplace_back(probeType->nameOf(channelValue));
      types.emplace_back(probeType->childAt(channelValue));
      continue;
    }
    channel = buildType->getChildIdxIfExists(name);
    if (channel.has_value()) {
      auto channelValue = channel.value();
      filterBuildProjections_.emplace_back(channelValue, filterChannel++);
      names.emplace_back(buildType->nameOf(channelValue));
      types.emplace_back(buildType->childAt(channelValue));
      continue;
    }
    VELOX_FAIL(
        "Spatial join filter field {} not in probe or build input, filter: {}",
        field->toString(),
        filter->toString());
  }

  filterInputType_ = ROW(std::move(names), std::move(types));
}

BlockingReason SpatialJoinProbe::isBlocked(ContinueFuture* future) {
  switch (state_) {
    case ProbeOperatorState::kRunning:
      [[fallthrough]];
    case ProbeOperatorState::kFinish:
      return BlockingReason::kNotBlocked;
    case ProbeOperatorState::kWaitForPeers:
      if (future_.valid()) {
        *future = std::move(future_);
        return BlockingReason::kWaitForJoinProbe;
      }
      setState(ProbeOperatorState::kFinish);
      return BlockingReason::kNotBlocked;
    case ProbeOperatorState::kWaitForBuild: {
      VELOX_CHECK(!buildVectors_.has_value());
      if (!getBuildData(future)) {
        return BlockingReason::kWaitForJoinBuild;
      }
      VELOX_CHECK(buildVectors_.has_value());
      setState(ProbeOperatorState::kRunning);
      return BlockingReason::kNotBlocked;
    }
    default:
      VELOX_UNREACHABLE(probeOperatorStateName(state_));
  }
}

void SpatialJoinProbe::close() {
  if (joinCondition_ != nullptr) {
    joinCondition_->clear();
  }
  buildVectors_.reset();
  spatialIndex_.reset();
  Operator::close();
}

void SpatialJoinProbe::noMoreInput() {
  Operator::noMoreInput();
  if (state_ == ProbeOperatorState::kRunning && input_ == nullptr) {
    setState(ProbeOperatorState::kFinish);
  }
}

bool SpatialJoinProbe::getBuildData(ContinueFuture* future) {
  VELOX_CHECK(!buildVectors_.has_value());

  auto buildData =
      operatorCtx_->task()
          ->getSpatialJoinBridge(
              operatorCtx_->driverCtx()->splitGroupId, planNodeId())
          ->dataOrFuture(future);
  if (!buildData.has_value()) {
    return false;
  }

  buildVectors_ = buildData.value().buildVectors;
  spatialIndex_ = buildData.value().spatialIndex;
  return true;
}

void SpatialJoinProbe::checkStateTransition(ProbeOperatorState state) {
  VELOX_CHECK_NE(state_, state);
  switch (state) {
    case ProbeOperatorState::kRunning:
      VELOX_CHECK_EQ(state_, ProbeOperatorState::kWaitForBuild);
      break;
    case ProbeOperatorState::kWaitForBuild:
      [[fallthrough]];
    case ProbeOperatorState::kFinish:
      VELOX_CHECK_EQ(state_, ProbeOperatorState::kRunning);
      break;
    default:
      VELOX_UNREACHABLE(probeOperatorStateName(state_));
      break;
  }
}

////////////////
// INPUT/OUTPUT

void SpatialJoinProbe::addInput(RowVectorPtr input) {
  VELOX_CHECK_NULL(input_);
  VELOX_CHECK_EQ(probeRow_, 0);
  VELOX_CHECK(!probeHasMatch_);
  VELOX_CHECK_EQ(buildVectorIndex_, 0);
  VELOX_CHECK_EQ(candidateIndex_, 0);

  // In getOutput(), we are going to wrap input in dictionaries a few rows at a
  // time. Since lazy vectors cannot be wrapped in different dictionaries, we
  // are going to load them here.
  for (auto& child : input->children()) {
    child->loadedVector();
  }
  input_ = std::move(input);
  decodedGeometryCol_.decode(*input_->childAt(probeGeometryChannel_)
                                  ->asChecked<SimpleVector<StringView>>());
  ++probeCount_;
}

RowVectorPtr SpatialJoinProbe::getOutput() {
  if (state_ == ProbeOperatorState::kFinish ||
      state_ == ProbeOperatorState::kWaitForPeers) {
    return nullptr;
  }

  RowVectorPtr output{nullptr};
  while (output == nullptr) {
    // Need more input.
    if (input_ == nullptr) {
      break;
    }

    // If the task owning this operator isn't running, there is no point
    // to continue executing this procedure, which may be long in degenerate
    // cases. Exit the working loop and let the Driver handle exiting
    // gracefully in its own loop.
    if (!operatorCtx_->task()->isRunning()) {
      break;
    }

    if (shouldYield()) {
      break;
    }

    // Generate actual join output by processing probe and build matches, and
    // probe mismaches (for left joins).
    output = generateOutput();
  }

  if (output != nullptr) {
    ++outputCount_;
  }
  return output;
}

RowVectorPtr SpatialJoinProbe::generateOutput() {
  VELOX_CHECK_NOT_NULL(input_);
  VELOX_CHECK_GT(input_->size(), probeRow_);
  outputBuilder_.initializeOutput(input_, pool());

  while (!isOutputDone()) {
    // Fill output_ with the results from one row.  This may produce too
    // much output and only partially complete.  If so, the next time we
    // call this we'll get the next chunk.
    //
    // addProbeRowOutput is responsible for advancing probeRow_.
    addProbeRowOutput();
  }

  // If we've exhausted the input, release it.
  if (probeRow_ >= input_->size()) {
    finishProbeInput();
  }

  return outputBuilder_.takeOutput();
}

// Return true if adding output stops early because output is full.
void SpatialJoinProbe::addProbeRowOutput() {
  VELOX_CHECK(buildVectors_.has_value());
  VELOX_CHECK(!outputBuilder_.isOutputFull());

  // Find the candidates for each probe row from the spatial index.  Only do
  // this at the start for each row.
  if (buildVectorIndex_ == 0 && candidateIndex_ == 0) {
    candidateBuildRows_ = querySpatialIndex();
  }

  while (!isProbeRowDone()) {
    addBuildVectorOutput(buildVectors_.value()[buildVectorIndex_]);
    if (outputBuilder_.isOutputFull()) {
      // If full, don't advance buildVectorIndex_ because we may not have
      // exhausted the current vector.  Return instead of breaking so that we
      // can add a mismatch row later if necessary.
      return;
    }
    advanceBuildVector();
  }

  // Now that we have finished the probe row, check if we need to add a probe
  // mismatch record.
  if (!probeHasMatch_ && needsProbeMismatch(joinType_)) {
    outputBuilder_.addProbeMismatchRow(probeRow_);
  }
  // Advance here instead of the loop in generateOutput so that early return on
  // full doesn't advance the probe.
  advanceProbeRow();
}

void SpatialJoinProbe::addBuildVectorOutput(const RowVectorPtr& buildVector) {
  if (FOLLY_UNLIKELY(needsFilterEvaluated_)) {
    // Evaluate join filter for the whole vector just once.
    evaluateJoinFilter(buildVector);
    needsFilterEvaluated_ = false;
  }

  // Start where we left off: after the last buildRow_ that was processed.
  while (!isBuildVectorDone(buildVector)) {
    vector_size_t buildRow = relativeBuildRow(candidateIndex_);
    if (isJoinConditionMatch(candidateIndex_)) {
      outputBuilder_.addOutputRow(probeRow_, buildRow);
      probeHasMatch_ = true;
    }

    // Advance candidateIndex_ even if full, since we're finished with this row.
    ++candidateIndex_;
  }

  // Since we are copying from the current buildVector, we must copy here.
  outputBuilder_.copyBuildValues(buildVector);
}

std::vector<int32_t> SpatialJoinProbe::querySpatialIndex() {
  VELOX_CHECK(spatialIndex_.has_value());
  VELOX_CHECK_NOT_NULL(spatialIndex_.value());

  if (decodedGeometryCol_.isNullAt(probeRow_)) {
    return std::vector<int32_t>{};
  }

  // Always apply radius to build side, not probe side.
  Envelope envelope = SpatialJoinBuild::readEnvelope(
      decodedGeometryCol_.valueAt<StringView>(probeRow_), 0 /* radius */);
  std::vector<int32_t> candidates = spatialIndex_.value()->query(envelope);
  std::sort(candidates.begin(), candidates.end());

  return candidates;
}

BufferPtr SpatialJoinProbe::makeBuildVectorIndices(vector_size_t vectorSize) {
  // Find the slice of candidates that are in this build vector.
  vector_size_t endIndex = candidateIndex_;
  for (; endIndex < candidateBuildRows_.size(); ++endIndex) {
    if (relativeBuildRow(endIndex) >= vectorSize) {
      break;
    }
  }

  // Make an index vector to fit the candidates.  Populate each entry with its
  // relative build row.
  vector_size_t indexCount =
      static_cast<vector_size_t>(endIndex - candidateIndex_);
  auto rowIndices = allocateIndices(indexCount, operatorCtx_->pool());
  auto rawIndices = rowIndices->asMutable<vector_size_t>();
  for (vector_size_t idx = 0; idx < indexCount; ++idx) {
    rawIndices[idx] = relativeBuildRow(idx + candidateIndex_);
  }

  return rowIndices;
}

void SpatialJoinProbe::evaluateJoinFilter(const RowVectorPtr& buildVector) {
  // Get the indices of the rows in the build vector that are candidates.
  auto candidateRowsBuffer = makeBuildVectorIndices(buildVector->size());

  // Now get the input for the spatial join filter, one row per candidate.
  auto filterInput = getNextJoinBatch(
      buildVector,
      filterInputType_,
      filterProbeProjections_,
      filterBuildProjections_,
      candidateRowsBuffer);

  if (filterInputRows_.size() != filterInput->size()) {
    filterInputRows_.resizeFill(filterInput->size(), true);
  }
  VELOX_CHECK(filterInputRows_.isAllSelected());

  std::vector<VectorPtr> filterResult;
  EvalCtx evalCtx(
      operatorCtx_->execCtx(), joinCondition_.get(), filterInput.get());
  joinCondition_->eval(0, 1, true, filterInputRows_, evalCtx, filterResult);
  VELOX_CHECK_GT(filterResult.size(), 0);
  filterOutput_ = filterResult[0];
  decodedFilterResult_.decode(*filterOutput_, filterInputRows_);
}

RowVectorPtr SpatialJoinProbe::getNextJoinBatch(
    const RowVectorPtr& buildVector,
    const RowTypePtr& outputType,
    const std::vector<IdentityProjection>& probeProjections,
    const std::vector<IdentityProjection>& buildProjections,
    BufferPtr candidateRows) const {
  VELOX_CHECK_GT(buildVector->size(), 0);
  // candidateRows is a buffer of vector_size_t indices into buildVector
  const vector_size_t numOutputRows =
      candidateRows->size() / sizeof(vector_size_t);
  if (numOutputRows == 0) {
    return RowVector::createEmpty(outputType, pool());
  }

  std::vector<VectorPtr> projectedChildren(outputType->size());
  // Project columns from the build side.
  projectChildren(
      projectedChildren,
      buildVector,
      buildProjections,
      numOutputRows,
      candidateRows);

  // Wrap projections from the probe side as constants.
  for (const auto [inputChannel, outputChannel] : probeProjections) {
    projectedChildren[outputChannel] = BaseVector::wrapInConstant(
        numOutputRows, probeRow_, input_->childAt(inputChannel));
  }

  return std::make_shared<RowVector>(
      pool(), outputType, nullptr, numOutputRows, std::move(projectedChildren));
}

void SpatialJoinProbe::finishProbeInput() {
  VELOX_CHECK_NOT_NULL(input_);
  input_.reset();
  probeRow_ = 0;

  if (noMoreInput_) {
    setState(ProbeOperatorState::kFinish);
  }
}

} // namespace facebook::velox::exec
