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
#include "velox/exec/NestedLoopJoinProbe.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/Task.h"
#include "velox/expression/FieldReference.h"

namespace facebook::velox::exec {

namespace {
bool needsProbeMismatch(core::JoinType joinType) {
  return isLeftJoin(joinType) || isFullJoin(joinType);
}

bool needsBuildMismatch(core::JoinType joinType) {
  return isRightJoin(joinType) || isFullJoin(joinType);
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

NestedLoopJoinProbe::NestedLoopJoinProbe(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::NestedLoopJoinNode>& joinNode)
    : Operator(
          driverCtx,
          joinNode->outputType(),
          operatorId,
          joinNode->id(),
          "NestedLoopJoinProbe"),
      outputBatchSize_{outputBatchRows()},
      joinType_(joinNode->joinType()) {
  auto probeType = joinNode->sources()[0]->outputType();
  auto buildType = joinNode->sources()[1]->outputType();
  identityProjections_ = extractProjections(probeType, outputType_);
  buildProjections_ = extractProjections(buildType, outputType_);

  if (joinNode->joinCondition() != nullptr) {
    initializeFilter(
        joinNode->joinCondition(),
        joinNode->sources()[0]->outputType(),
        joinNode->sources()[1]->outputType());
  }
}

BlockingReason NestedLoopJoinProbe::isBlocked(ContinueFuture* future) {
  switch (state_) {
    case ProbeOperatorState::kRunning:
      FOLLY_FALLTHROUGH;
    case ProbeOperatorState::kFinish:
      return BlockingReason::kNotBlocked;
    case ProbeOperatorState::kWaitForBuild: {
      VELOX_CHECK(buildData_.has_value() == false);
      if (!getBuildData(future)) {
        return BlockingReason::kWaitForJoinBuild;
      }
      VELOX_CHECK(buildData_.has_value());

      if (needsBuildMismatch(joinType_)) {
        buildMatched_.resize(buildData_->size());
        for (auto i = 0; i < buildData_->size(); ++i) {
          buildMatched_[i].resizeFill(buildData_.value()[i]->size(), false);
        }
      }

      setState(ProbeOperatorState::kRunning);
      return BlockingReason::kNotBlocked;
    }
    case ProbeOperatorState::kWaitForPeers:
      if (future_.valid()) {
        *future = std::move(future_);
        return BlockingReason::kWaitForJoinProbe;
      }
      if (!lastProbe_) {
        setState(ProbeOperatorState::kFinish);
      }
      return BlockingReason::kNotBlocked;
    default:
      VELOX_UNREACHABLE(probeOperatorStateName(state_));
  }
}

void NestedLoopJoinProbe::close() {
  if (joinCondition_ != nullptr) {
    joinCondition_->clear();
  }
  buildData_.reset();
  Operator::close();
}

void NestedLoopJoinProbe::addInput(RowVectorPtr input) {
  // In getOutput(), we are going to wrap input in dictionaries a few rows at a
  // time. Since lazy vectors cannot be wrapped in different dictionaries, we
  // are going to load them here.
  for (auto& child : input->children()) {
    child->loadedVector();
  }
  input_ = std::move(input);
  buildIndex_ = 0;
  if (needsProbeMismatch(joinType_)) {
    probeMatched_.resizeFill(input_->size(), false);
  }
}

RowVectorPtr NestedLoopJoinProbe::getOutput() {
  while (true) {
    switch (state_) {
      case ProbeOperatorState::kFinish:
        return nullptr;
      case ProbeOperatorState::kWaitForPeers: {
        VELOX_CHECK(lastProbe_);
        RowVectorPtr output{nullptr};
        while (!output && !hasProbedAllBuildData()) {
          output = getMismatchedOutput(
              buildData_.value()[buildIndex_],
              buildMatched_[buildIndex_],
              buildOutMapping_,
              buildProjections_,
              identityProjections_);
          buildIndex_++;
        }
        if (hasProbedAllBuildData()) {
          setState(ProbeOperatorState::kFinish);
        }
        return output;
      }
      case ProbeOperatorState::kRunning: {
        if (input_ == nullptr) {
          return nullptr;
        }
        // When input_ is not null but buildIndex_ is at the end, it means the
        // matching of input_ and buildData_ has finished. For left/full joins,
        // the next step is to emit output for mismatched probe side rows.
        if (hasProbedAllBuildData()) {
          auto output = needsProbeMismatch(joinType_)
              ? getMismatchedOutput(
                    input_,
                    probeMatched_,
                    probeOutMapping_,
                    identityProjections_,
                    buildProjections_)
              : nullptr;
          finishMatchAndProbeMismatch();
          return output;
        }

        vector_size_t probeCnt = getProbeCnt();
        auto output = doMatching(probeCnt);
        if (advanceIndex(probeCnt)) {
          if (!needsProbeMismatch(joinType_)) {
            finishMatchAndProbeMismatch();
          }
        }
        if (output) {
          return output;
        }
        break;
      }
      default:
        VELOX_UNREACHABLE(probeOperatorStateName(state_));
    }
  }
  VELOX_UNREACHABLE();
}

void NestedLoopJoinProbe::initializeFilter(
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
  auto numFields = joinCondition_->expr(0)->distinctFields().size();
  names.reserve(numFields);
  types.reserve(numFields);
  for (auto& field : joinCondition_->expr(0)->distinctFields()) {
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
        "Join filter field {} not in probe or build input, filter: {}",
        field->toString(),
        filter->toString());
  }

  filterInputType_ = ROW(std::move(names), std::move(types));
}

RowVectorPtr NestedLoopJoinProbe::getMismatchedOutput(
    RowVectorPtr data,
    const SelectivityVector& matched,
    BufferPtr mapping,
    const std::vector<IdentityProjection>& projections,
    const std::vector<IdentityProjection>& nullProjections) {
  if (matched.isAllSelected()) {
    return nullptr;
  }

  auto rawMapping = initializeRowNumberMapping(mapping, data->size(), pool());
  int32_t numUnmatched{0};
  for (auto i = 0; i < data->size(); ++i) {
    if (!matched.isValid(i)) {
      rawMapping[numUnmatched++] = i;
    }
  }
  VELOX_CHECK_GT(numUnmatched, 0);

  auto output =
      BaseVector::create<RowVector>(outputType_, numUnmatched, pool());
  projectChildren(output, data, projections, numUnmatched, mapping);
  for (auto projection : nullProjections) {
    output->childAt(projection.outputChannel) = BaseVector::createNullConstant(
        outputType_->childAt(projection.outputChannel), output->size(), pool());
  }
  return output;
}

void NestedLoopJoinProbe::finishMatchAndProbeMismatch() {
  input_.reset();
  if (!noMoreInput_) {
    return;
  }
  if (!needsBuildMismatch(joinType_)) {
    setState(ProbeOperatorState::kFinish);
    return;
  }
  beginBuildMismatch();
}

void NestedLoopJoinProbe::noMoreInput() {
  Operator::noMoreInput();
  if (state_ != ProbeOperatorState::kRunning || input_ != nullptr) {
    return;
  }
  if (!needsBuildMismatch(joinType_)) {
    setState(ProbeOperatorState::kFinish);
    return;
  }
  beginBuildMismatch();
}

void NestedLoopJoinProbe::beginBuildMismatch() {
  VELOX_CHECK(needsBuildMismatch(joinType_));

  setState(ProbeOperatorState::kWaitForPeers);

  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<Driver>> peers;
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    VELOX_CHECK(future_.valid());
    return;
  }

  lastProbe_ = true;
  // From now on, buildIndex_ is used to indexing into buildMismatched_
  buildIndex_ = 0;
  for (auto& peer : peers) {
    auto op = peer->findOperator(planNodeId());
    auto* probe = dynamic_cast<NestedLoopJoinProbe*>(op);
    VELOX_CHECK(probe);
    for (auto i = 0; i < buildMatched_.size(); ++i) {
      buildMatched_[i].select(probe->buildMatched_[i]);
    }
  }
  peers.clear();
  for (auto i = 0; i < buildMatched_.size(); ++i) {
    buildMatched_[i].updateBounds();
  }
  for (auto& promise : promises) {
    promise.setValue();
  }
}

bool NestedLoopJoinProbe::getBuildData(ContinueFuture* future) {
  VELOX_CHECK(!buildData_.has_value());

  auto buildData =
      operatorCtx_->task()
          ->getNestedLoopJoinBridge(
              operatorCtx_->driverCtx()->splitGroupId, planNodeId())
          ->dataOrFuture(future);
  if (!buildData.has_value()) {
    return false;
  }

  buildData_ = std::move(buildData);
  if (buildData_->empty()) {
    // Build side is empty. Return empty set of rows and  terminate the pipeline
    // early.
    buildSideEmpty_ = true;
  }
  return true;
}

vector_size_t NestedLoopJoinProbe::getProbeCnt() const {
  VELOX_CHECK_NOT_NULL(input_);
  VELOX_CHECK(!hasProbedAllBuildData());

  const auto inputSize = input_->size();
  auto buildSize = buildData_.value()[buildIndex_]->size();
  vector_size_t probeCnt;
  if (buildSize > outputBatchSize_) {
    probeCnt = 1;
  } else {
    numProbeRows = std::min(
        (vector_size_t)outputBatchSize_ / numBuildRows, inputSize - probeRow_);
  }
  return probeCnt;
}

RowVectorPtr NestedLoopJoinProbe::getCrossProduct(
    vector_size_t probeCnt,
    const RowTypePtr& outputType,
    const std::vector<IdentityProjection>& probeProjections,
    const std::vector<IdentityProjection>& buildProjections) {
  VELOX_CHECK_GT(probeCnt, 0);
  VELOX_CHECK(!hasProbedAllBuildData());

  const auto buildSize = buildData_.value()[buildIndex_]->size();
  const auto batchSize = probeCnt * buildSize;
  const bool probeCntChanged = (probeCnt != prevProbeCnt_);
  prevProbeCnt_ = probeCnt;
  auto output = BaseVector::create<RowVector>(outputType, batchSize, pool());

  auto rawProbeIndices =
      initializeRowNumberMapping(probeIndices_, batchSize, pool());
  for (auto i = 0; i < probeCnt; ++i) {
    std::fill(
        rawProbeIndices.begin() + i * buildSize,
        rawProbeIndices.begin() + (i + 1) * buildSize,
        probeRow_ + i);
  }

  if (probeCntChanged) {
    auto rawBuildIndices_ =
        initializeRowNumberMapping(buildIndices_, batchSize, pool());
    for (auto i = 0; i < probeCnt; ++i) {
      std::iota(
          rawBuildIndices_.begin() + i * buildSize,
          rawBuildIndices_.begin() + (i + 1) * buildSize,
          0);
    }
  }

  projectChildren(output, input_, probeProjections, batchSize, probeIndices_);
  projectChildren(
      output,
      buildData_.value()[buildIndex_],
      buildProjections,
      batchSize,
      buildIndices_);
  return output;
}

bool NestedLoopJoinProbe::advanceIndex(vector_size_t probeCnt) {
  probeRow_ += probeCnt;
  if (probeRow_ < input_->size()) {
    return false;
  }
  probeRow_ = 0;
  prevProbeCnt_ = 0;
  do {
    ++buildIndex_;
  } while (!hasProbedAllBuildData() &&
           !buildData_.value()[buildIndex_]->size());
  return hasProbedAllBuildData();
}

RowVectorPtr NestedLoopJoinProbe::doMatching(vector_size_t probeCnt) {
  VELOX_CHECK_NOT_NULL(input_);
  VELOX_CHECK(!hasProbedAllBuildData());

  if (joinCondition_ == nullptr) {
    return getCrossProduct(
        probeCnt, outputType_, identityProjections_, buildProjections_);
  }

  auto filterInput = getCrossProduct(
      probeCnt,
      filterInputType_,
      filterProbeProjections_,
      filterBuildProjections_);

  if (filterInputRows_.size() != filterInput->size()) {
    filterInputRows_.resizeFill(filterInput->size(), true);
  }
  VELOX_CHECK(filterInputRows_.isAllSelected());

  std::vector<VectorPtr> filterResult;
  EvalCtx evalCtx(
      operatorCtx_->execCtx(), joinCondition_.get(), filterInput.get());
  joinCondition_->eval(0, 1, true, filterInputRows_, evalCtx, filterResult);
  DecodedVector decodedFilterResult;
  decodedFilterResult.decode(*filterResult[0], filterInputRows_);

  const vector_size_t resultSize = decodedFilterResult.size();
  auto rawProbeOutMapping =
      initializeRowNumberMapping(probeOutMapping_, resultSize, pool());
  auto rawBuildOutMapping =
      initializeRowNumberMapping(buildOutMapping_, resultSize, pool());
  auto probeIndices = probeIndices_->asMutable<vector_size_t>();
  auto buildIndices = buildIndices_->asMutable<vector_size_t>();
  int32_t outputSize{0};
  for (auto i = 0; i < resultSize; ++i) {
    if (!decodedFilterResult.isNullAt(i) &&
        decodedFilterResult.valueAt<bool>(i)) {
      rawProbeOutMapping[outputSize] = probeIndices[i];
      rawBuildOutMapping[outputSize] = buildIndices[i];
      outputSize++;
    }
  }
  if (needsProbeMismatch(joinType_)) {
    for (auto i = 0; i < outputSize; ++i) {
      probeMatched_.setValid(rawProbeOutMapping[i], true);
    }
    probeMatched_.updateBounds();
  }
  if (needsBuildMismatch(joinType_)) {
    for (auto i = 0; i < outputSize; ++i) {
      buildMatched_[buildIndex_].setValid(rawBuildOutMapping[i], true);
    }
  }

  if (!outputSize) {
    return nullptr;
  }
  auto output = BaseVector::create<RowVector>(outputType_, outputSize, pool());
  projectChildren(
      output, input_, identityProjections_, outputSize, probeOutMapping_);
  projectChildren(
      output,
      buildData_.value()[buildIndex_],
      buildProjections_,
      outputSize,
      buildOutMapping_);
  return output;
}

} // namespace facebook::velox::exec
