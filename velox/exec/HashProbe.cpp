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

#include "velox/exec/HashProbe.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/Task.h"
#include "velox/expression/ControlExpr.h"

namespace facebook::velox::exec {

namespace {
static constexpr ChannelIndex kNoChannel = ~0;

static ChannelIndex childIndex(const RowType* type, const std::string& name) {
  for (auto i = 0; i < type->size(); ++i) {
    if (type->nameOf(i) == name) {
      return i;
    }
  }
  return kNoChannel;
}

// Returns the type for the hash table row. Build side keys first,
// then dependent build side columns.
static std::shared_ptr<const RowType> makeTableType(
    const RowType* type,
    const std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>&
        keys) {
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  std::unordered_set<ChannelIndex> keyChannels(keys.size());
  names.reserve(type->size());
  types.reserve(type->size());
  for (auto key : keys) {
    auto channel = type->getChildIdx(key->name());
    names.emplace_back(type->nameOf(channel));
    types.emplace_back(type->childAt(channel));
    keyChannels.insert(channel);
  }
  for (auto i = 0; i < type->size(); ++i) {
    if (keyChannels.find(i) == keyChannels.end()) {
      names.emplace_back(type->nameOf(i));
      types.emplace_back(type->childAt(i));
    }
  }
  return std::make_shared<RowType>(std::move(names), std::move(types));
}
} // namespace

HashProbe::HashProbe(
    int32_t operatorId,
    DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode)
    : Operator(
          driverCtx,
          joinNode->outputType(),
          operatorId,
          joinNode->id(),
          "HashProbe"),
      joinType_{joinNode->joinType()},
      filterResult_(1),
      outputRows_(kOutputBatchSize) {
  VELOX_CHECK(
      joinNode->isInnerJoin() || joinNode->isSemiJoin() ||
      joinNode->isAntiJoin());
  auto probeType = joinNode->sources()[0]->outputType();
  auto numKeys = joinNode->leftKeys().size();
  keyChannels_.reserve(numKeys);
  hashers_.reserve(numKeys);
  for (auto& key : joinNode->leftKeys()) {
    auto channel = exprToChannel(key.get(), probeType);
    keyChannels_.emplace_back(channel);
    hashers_.push_back(
        std::make_unique<VectorHasher>(probeType->childAt(channel), channel));
  }
  lookup_ = std::make_unique<HashLookup>(hashers_);
  auto buildType = joinNode->sources()[1]->outputType();
  auto tableType = makeTableType(buildType.get(), joinNode->rightKeys());
  if (joinNode->filter()) {
    initializeFilter(joinNode->filter(), probeType, tableType);
  }

  bool isIdentityProjection = true;
  for (auto i = 0; i < probeType->size(); ++i) {
    auto input = probeType->childAt(i);
    auto name = probeType->nameOf(i);
    auto outIndex = childIndex(outputType_.get(), name);
    if (outIndex != kNoChannel) {
      identityProjections_.emplace_back(i, outIndex);
      if (outIndex != i) {
        isIdentityProjection = false;
      }
    }
  }

  for (ChannelIndex i = 0; i < outputType_->size(); ++i) {
    auto tableChannel = childIndex(tableType.get(), outputType_->nameOf(i));
    if (tableChannel != kNoChannel) {
      tableResultProjections_.emplace_back(tableChannel, i);
    }
  }

  if (isIdentityProjection && tableResultProjections_.empty()) {
    isIdentityProjection_ = true;
  }
}

void HashProbe::initializeFilter(
    const std::shared_ptr<const core::ITypedExpr>& filter,
    const RowTypePtr& probeType,
    const RowTypePtr& tableType) {
  std::vector<std::shared_ptr<const core::ITypedExpr>> filters = {filter};
  filter_ =
      std::make_unique<ExprSet>(std::move(filters), operatorCtx_->execCtx());
  ChannelIndex filterChannel = 0;
  for (auto& field : filter_->expr(0)->distinctFields()) {
    const auto& name = field->field();
    auto channel = childIndex(probeType.get(), name);
    if (channel != kNoChannel) {
      filterProbeInputs_.emplace_back(channel, filterChannel++);
      continue;
    }
    channel = childIndex(tableType.get(), name);
    if (channel != kNoChannel) {
      filterBuildInputs_.emplace_back(channel, filterChannel++);
      continue;
    }
    VELOX_FAIL(
        "Join filter field {} not in probe or build input", field->toString());
  }
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  auto numFields = filterProbeInputs_.size() + filterBuildInputs_.size();
  names.reserve(numFields);
  types.reserve(numFields);
  for (auto projection : filterProbeInputs_) {
    names.emplace_back(probeType->nameOf(projection.inputChannel));
    types.emplace_back(probeType->childAt(projection.inputChannel));
  }
  for (auto projection : filterBuildInputs_) {
    names.emplace_back(tableType->nameOf(projection.inputChannel));
    types.emplace_back(tableType->childAt(projection.inputChannel));
  }
  filterInputType_ = ROW(std::move(names), std::move(types));
}

BlockingReason HashProbe::isBlocked(ContinueFuture* future) {
  if (table_) {
    return BlockingReason::kNotBlocked;
  }

  auto hashBuildResult = operatorCtx_->task()
                             ->findOrCreateJoinBridge(planNodeId())
                             ->tableOrFuture(future);
  if (!hashBuildResult.has_value()) {
    return BlockingReason::kWaitForJoinBuild;
  }

  if (hashBuildResult->antiJoinHashNullKeys) {
    // Anti join with null keys on the build side always returns nothing.
    VELOX_CHECK(joinType_ == core::JoinType::kAnti);
    isFinishing_ = true;
  } else {
    table_ = hashBuildResult->table;
    if (table_->numDistinct() == 0) {
      // Build side is empty. Inner and semi joins return nothing in this case,
      // hence, we can terminate the pipeline early.
      if (joinType_ == core::JoinType::kInner ||
          joinType_ == core::JoinType::kSemi) {
        isFinishing_ = true;
      }
    } else if (
        joinType_ == core::JoinType::kInner &&
        table_->hashMode() != BaseHashTable::HashMode::kHash) {
      // Find out whether there are any upstream operators that can accept
      // dynamic filters on all or a subset of the join keys. Setup dynamic
      // filter builders to track join selectivity for these keys and generate
      // dynamic filters to push down.
      const auto& buildHashers = table_->hashers();
      auto channels = operatorCtx_->driverCtx()->driver->canPushdownFilters(
          this, keyChannels_);
      dynamicFilterBuilders_.resize(keyChannels_.size());
      for (auto i = 0; i < keyChannels_.size(); i++) {
        auto it = channels.find(keyChannels_[i]);
        if (it != channels.end()) {
          dynamicFilterBuilders_[i].emplace(DynamicFilterBuilder(
              *(buildHashers[i].get()), keyChannels_[i], dynamicFilters_));
        }
      }
    }
  }

  return BlockingReason::kNotBlocked;
}

void HashProbe::clearDynamicFilters() {
  // The join can be completely replaced with a pushed down
  // filter when the following conditions are met:
  //  * hash table has a single key with unique values,
  //  * build side has no dependent columns.
  if (keyChannels_.size() == 1 && !table_->hasDuplicateKeys() &&
      tableResultProjections_.empty() && !filter_ && !dynamicFilters_.empty()) {
    canReplaceWithDynamicFilter_ = true;
  }

  Operator::clearDynamicFilters();
}

void HashProbe::addInput(RowVectorPtr input) {
  input_ = std::move(input);

  if (canReplaceWithDynamicFilter_) {
    replacedWithDynamicFilter_ = true;
    return;
  }

  if (table_->numDistinct() == 0) {
    // Build side is empty. This state is valid only for anti join which returns
    // all probe rows.
    VELOX_CHECK(joinType_ == core::JoinType::kAnti);
    return;
  }

  nonNullRows_.resize(input_->size());
  nonNullRows_.setAll();
  deselectRowsWithNulls(*input_, keyChannels_, nonNullRows_);

  auto getDynamicFilterBuilder = [&](auto i) -> DynamicFilterBuilder* {
    if (!dynamicFilterBuilders_.empty()) {
      auto& builder = dynamicFilterBuilders_[i];
      if (builder.has_value() && builder->isActive()) {
        return &(builder.value());
      }
    }
    return nullptr;
  };

  activeRows_ = nonNullRows_;
  lookup_->hashes.resize(input_->size());
  auto mode = table_->hashMode();
  auto& buildHashers = table_->hashers();
  for (auto i = 0; i < keyChannels_.size(); ++i) {
    auto key = input_->loadedChildAt(keyChannels_[i]);
    if (mode != BaseHashTable::HashMode::kHash) {
      auto* dynamicFilterBuilder = getDynamicFilterBuilder(i);
      if (dynamicFilterBuilder) {
        dynamicFilterBuilder->addInput(activeRows_.countSelected());
      }

      valueIdDecoder_.decode(*key, activeRows_);
      buildHashers[i]->lookupValueIds(
          valueIdDecoder_, activeRows_, deduppedHashes_, &lookup_->hashes);

      if (dynamicFilterBuilder) {
        dynamicFilterBuilder->addOutput(activeRows_.countSelected());
      }
    } else {
      hashers_[i]->hash(*key, activeRows_, i > 0, &lookup_->hashes);
    }
  }
  lookup_->rows.clear();
  if (activeRows_.isAllSelected()) {
    lookup_->rows.resize(activeRows_.size());
    std::iota(lookup_->rows.begin(), lookup_->rows.end(), 0);
  } else {
    bits::forEachSetBit(
        activeRows_.asRange().bits(),
        0,
        activeRows_.size(),
        [&](vector_size_t row) { lookup_->rows.push_back(row); });
  }
  if (lookup_->rows.empty()) {
    if (joinType_ != core::JoinType::kAnti) {
      input_ = nullptr;
    }
    return;
  }
  lookup_->hits.resize(lookup_->rows.back() + 1);
  table_->joinProbe(*lookup_);
  results_.reset(*lookup_);
}

namespace {
// Copies values from 'rows' of 'table' according to 'projections' in
// 'result'. Reuses 'result' children where possible.
void extractColumns(
    BaseHashTable* table,
    folly::Range<char**> rows,
    folly::Range<const IdentityProjection*> projections,
    memory::MemoryPool* pool,
    RowVectorPtr result) {
  result->resize(rows.size());
  for (auto projection : projections) {
    auto& child = result->childAt(projection.outputChannel);
    // TODO: Consider reuse of complex types.
    if (!child || !BaseVector::isReusableFlatVector(child)) {
      child = BaseVector::create(
          result->type()->childAt(projection.outputChannel), rows.size(), pool);
    }
    child->resize(rows.size());
    table->rows()->extractColumn(
        rows.data(), rows.size(), projection.inputChannel, child);
  }
}

folly::Range<vector_size_t*> initializeRowNumberMapping(
    BufferPtr& mapping,
    vector_size_t size,
    memory::MemoryPool* pool) {
  if (!mapping || !mapping->unique() ||
      mapping->size() < sizeof(vector_size_t) * size) {
    mapping = AlignedBuffer::allocate<vector_size_t>(size, pool);
  }
  return folly::Range(mapping->asMutable<vector_size_t>(), size);
}
} // namespace

RowVectorPtr HashProbe::getOutput() {
  clearIdentityProjectedOutput();
  if (!input_) {
    return nullptr;
  }

  const auto inputSize = input_->size();

  if (replacedWithDynamicFilter_) {
    stats_.addRuntimeStat("replacedWithDynamicFilterRows", inputSize);
    auto output = fillOutput(inputSize, nullptr);
    input_ = nullptr;
    return output;
  }

  const bool isSemiOrAntiJoin =
      joinType_ == core::JoinType::kSemi || joinType_ == core::JoinType::kAnti;

  // Semi and anti joins are always cardinality reducing, e.g. for a given row
  // of input they produce zero or 1 row of output. Therefore, we can process
  // each batch of input in one go.
  auto mapping = initializeRowNumberMapping(
      rowNumberMapping_,
      isSemiOrAntiJoin ? inputSize : kOutputBatchSize,
      pool());

  if (isSemiOrAntiJoin) {
    outputRows_.resize(inputSize);
  }

  for (;;) {
    int numOut = 0;
    if (joinType_ == core::JoinType::kAnti) {
      if (table_->numDistinct() == 0) {
        // When build side is empty, anti join returns all probe side rows,
        // including ones with null join keys.
        std::iota(mapping.begin(), mapping.end(), 0);
        numOut = inputSize;
      } else {
        // When build side is not empty, anti join returns probe rows with no
        // nulls in the join key and no match in the build side.
        for (auto i = 0; i < inputSize; i++) {
          if (nonNullRows_.isValid(i) &&
              (!activeRows_.isValid(i) || !lookup_->hits[i])) {
            mapping[numOut] = i;
            ++numOut;
          }
        }
      }
    } else {
      numOut = table_->listJoinResults(
          results_,
          mapping,
          folly::Range(outputRows_.data(), outputRows_.size()));
    }
    if (!numOut) {
      input_ = nullptr;
      return nullptr;
    }
    VELOX_CHECK_LE(numOut, outputRows_.size());
    numOut = evalFilter(numOut);
    if (!numOut) {
      // the filter was false on all rows.
      if (isSemiOrAntiJoin) {
        input_ = nullptr;
        return nullptr;
      }
      continue;
    }
    VectorPtr outputAsBase = std::move(output_);
    BaseVector::ensureWritable(
        SelectivityVector::empty(),
        outputType_,
        operatorCtx_->pool(),
        &outputAsBase);
    output_ = std::static_pointer_cast<RowVector>(outputAsBase);
    extractColumns(
        table_.get(),
        folly::Range<char**>(outputRows_.data(), numOut),
        tableResultProjections_,
        operatorCtx_->pool(),
        output_);

    for (auto projection : identityProjections_) {
      // Load input vector if it is being split into multiple batches. It is not
      // safe to wrap unloaded LazyVector into two different dictionaries.
      auto inputChild = numOut == outputRows_.size()
          ? input_->loadedChildAt(projection.inputChannel)
          : input_->childAt(projection.inputChannel);

      output_->childAt(projection.outputChannel) =
          wrapChild(numOut, rowNumberMapping_, inputChild);
    }
    if (isSemiOrAntiJoin) {
      input_ = nullptr;
    }
    return output_;
  }
}

int32_t HashProbe::evalFilter(int32_t numRows) {
  if (!filter_) {
    return numRows;
  }
  if (!filterInput_) {
    filterInput_ = std::static_pointer_cast<RowVector>(
        BaseVector::create(filterInputType_, 1, operatorCtx_->pool()));
  }
  filterInput_->resize(numRows);
  for (auto projection : filterProbeInputs_) {
    filterInput_->childAt(projection.outputChannel) = wrapChild(
        numRows, rowNumberMapping_, input_->childAt(projection.inputChannel));
  }
  extractColumns(
      table_.get(),
      folly::Range<char**>(outputRows_.data(), numRows),
      filterBuildInputs_,
      operatorCtx_->pool(),
      filterInput_);
  filterRows_.resize(numRows);
  filterRows_.setAll();
  EvalCtx evalCtx(operatorCtx_->execCtx(), filter_.get(), filterInput_.get());
  filter_->eval(0, 1, true, filterRows_, &evalCtx, &filterResult_);
  decodedFilterResult_.decode(*filterResult_[0], filterRows_);
  int32_t numPassed = 0;
  auto rawMapping = rowNumberMapping_->asMutable<vector_size_t>();
  for (auto i = 0; i < numRows; ++i) {
    if (decodedFilterResult_.valueAt<bool>(i)) {
      outputRows_[numPassed] = outputRows_[i];
      rawMapping[numPassed++] = rawMapping[i];
    }
  }
  return numPassed;
}

} // namespace facebook::velox::exec
