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
#include "velox/exec/DistinctAggregations.h"
#include "velox/exec/SetAccumulator.h"

namespace facebook::velox::exec {

namespace {

// Handles distinct aggregations where all inputs are constants.
// The distinct set of a constant tuple is always either empty or a single
// element, so it only needs a boolean flag per group indicating whether any
// row was seen.
class ConstantDistinctAggregations : public DistinctAggregations {
 public:
  ConstantDistinctAggregations(
      std::vector<AggregateInfo*> aggregates,
      memory::MemoryPool* pool)
      : pool_{pool}, aggregates_{std::move(aggregates)} {
    for (const auto& aggregate : aggregates_) {
      for (size_t i = 0; i < aggregate->inputs.size(); ++i) {
        VELOX_DCHECK_NOT_NULL(aggregate->constantInputs[i]);
      }
    }
  }

  Accumulator accumulator() const override {
    return {/*isFixedSize=*/true,
            sizeof(bool),
            /*usesExternalMemory=*/false,
            /*alignment=*/1,
            BOOLEAN(),
            /*spillExtractFunction=*/
            [this](folly::Range<char**> groups, VectorPtr& result) {
              extractForSpill(groups, result);
            },
            /*destroyFunction=*/nullptr};
  }

  void addInput(
      char** groups,
      const RowVectorPtr& /*input*/,
      const SelectivityVector& rows) override {
    rows.applyToSelected([&](vector_size_t i) { value(groups[i]) = true; });
  }

  void addSingleGroupInput(
      char* group,
      const RowVectorPtr& /*input*/,
      const SelectivityVector& /*rows*/) override {
    value(group) = true;
  }

  void extractValues(folly::Range<char**> groups, const RowVectorPtr& result)
      override {
    raw_vector<int32_t> indices(pool_);
    for (const auto& aggregate : aggregates_) {
      // All inputs are constant, so the aggregate result is identical for every
      // group that saw rows. Compute it once and broadcast.
      // Check whether any group is non-empty.
      bool hasNonEmpty = false;
      for (vector_size_t i = 0; i < groups.size(); ++i) {
        if (value(groups[i])) {
          hasNonEmpty = true;
          break;
        }
      }

      if (groups.size() < 2 || !hasNonEmpty) {
        // With 0 or 1 groups no broadcasting is needed.  If all groups
        // are empty, there is no need to add input, so just extract directly.
        if (hasNonEmpty) {
          VELOX_CHECK_EQ(groups.size(), 1);
          const SelectivityVector rows(1);
          aggregate->function->addSingleGroupRawInput(
              groups[0], rows, aggregate->constantInputs, false);
        }
        aggregate->function->extractValues(
            groups.data(), groups.size(), &result->childAt(aggregate->output));
      } else {
        // Use groups[0] as the non-empty representative and groups[1] as the
        // empty representative. Extract from these two into a 2-row vector,
        // then wrap the output in a dictionary that maps each group to the
        // appropriate row.
        const SelectivityVector rows(1);
        aggregate->function->addSingleGroupRawInput(
            groups[0], rows, aggregate->constantInputs, false);

        std::array<char*, 2> representatives = {groups[0], groups[1]};
        auto extracted =
            BaseVector::create(aggregate->function->resultType(), 2, pool_);
        aggregate->function->extractValues(
            representatives.data(), 2, &extracted);

        auto dictIndices = allocateIndices(groups.size(), pool_);
        auto* rawIndices = dictIndices->asMutable<vector_size_t>();
        for (vector_size_t i = 0; i < groups.size(); ++i) {
          // Index 0 = non-empty value, index 1 = empty/default value.
          rawIndices[i] = value(groups[i]) ? 0 : 1;
        }
        result->childAt(aggregate->output) = BaseVector::wrapInDictionary(
            nullptr,
            std::move(dictIndices),
            groups.size(),
            std::move(extracted));
      }

      aggregate->function->destroy(groups);
      aggregate->function->initializeNewGroups(
          groups.data(),
          folly::Range<const int32_t*>(
              iota(groups.size(), indices), groups.size()));
    }
  }

  void addSingleGroupSpillInput(
      char* group,
      const VectorPtr& input,
      vector_size_t index) override {
    if (input->as<FlatVector<bool>>()->valueAt(index)) {
      value(group) = true;
    }
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    for (auto index : indices) {
      groups[index][nullByte_] |= nullMask_;
      value(groups[index]) = false;
    }

    for (const auto& aggregate : aggregates_) {
      aggregate->function->initializeNewGroups(groups, indices);
    }
  }

 private:
  bool& value(char* group) const {
    return *reinterpret_cast<bool*>(group + offset_);
  }

  void extractForSpill(folly::Range<char**> groups, VectorPtr& result) const {
    auto* flatResult = result->asFlatVector<bool>();
    flatResult->resize(groups.size());
    for (auto i = 0; i < groups.size(); ++i) {
      flatResult->set(i, value(groups[i]));
    }
  }

  memory::MemoryPool* const pool_;
  const std::vector<AggregateInfo*> aggregates_;
};

template <
    typename T,
    typename AccumulatorType = aggregate::prestosql::SetAccumulator<T>>
class TypedDistinctAggregations : public DistinctAggregations {
 public:
  TypedDistinctAggregations(
      std::vector<AggregateInfo*> aggregates,
      const RowTypePtr& inputType,
      std::vector<column_index_t> nonConstantInputs,
      memory::MemoryPool* pool)
      : pool_{pool},
        aggregates_{std::move(aggregates)},
        nonConstantInputs_{std::move(nonConstantInputs)},
        inputType_(makeInputTypeForAccumulator(inputType, nonConstantInputs_)),
        spillType_(ARRAY(inputType_)),
        singleNonConstantInput_(nonConstantInputs_.size() == 1) {}

  /// Returns metadata about the accumulator used to store unique inputs.
  Accumulator accumulator() const override {
    return {/*isFixedSize=*/false,
            sizeof(AccumulatorType),
            /*usesExternalMemory=*/false,
            /*alignment=*/1,
            spillType_,
            /*spillExtractFunction=*/
            [this](folly::Range<char**> groups, VectorPtr& result) {
              extractForSpill(groups, result);
            },
            /*destroyFunction=*/
            [this](folly::Range<char**> groups) {
              for (auto* group : groups) {
                if (!isInitialized(group)) {
                  continue;
                }
                auto* accumulator =
                    reinterpret_cast<AccumulatorType*>(group + offset_);
                accumulator->free(*allocator_);
              }
            }};
  }

  void addInput(
      char** groups,
      const RowVectorPtr& input,
      const SelectivityVector& rows) override {
    decodeInput(input, rows);

    rows.applyToSelected([&](vector_size_t i) {
      auto* group = groups[i];
      auto* accumulator = reinterpret_cast<AccumulatorType*>(group + offset_);

      RowSizeTracker<char, uint32_t> tracker(
          group[rowSizeOffset_], *allocator_);
      accumulator->addValue(decodedInput_, i, allocator_);
    });

    inputForAccumulator_.reset();
  }

  void addSingleGroupInput(
      char* group,
      const RowVectorPtr& input,
      const SelectivityVector& rows) override {
    decodeInput(input, rows);

    auto* accumulator = reinterpret_cast<AccumulatorType*>(group + offset_);
    RowSizeTracker<char, uint32_t> tracker(group[rowSizeOffset_], *allocator_);
    rows.applyToSelected([&](vector_size_t i) {
      accumulator->addValue(decodedInput_, i, allocator_);
    });

    inputForAccumulator_.reset();
  }

  void extractValues(folly::Range<char**> groups, const RowVectorPtr& result)
      override {
    SelectivityVector rows;
    for (auto i = 0; i < aggregates_.size(); ++i) {
      const auto& aggregate = *aggregates_[i];

      // For each group, add distinct inputs to aggregate.
      for (auto* group : groups) {
        auto* accumulator = reinterpret_cast<AccumulatorType*>(group + offset_);

        // TODO Process group rows in batches to avoid creating very large input
        // vectors.
        auto data = BaseVector::create(inputType_, accumulator->size(), pool_);
        if constexpr (std::is_same_v<T, ComplexType>) {
          accumulator->extractValues(*data, 0);
        } else {
          accumulator->extractValues(*(data->template as<FlatVector<T>>()), 0);
        }

        if (data->size() > 0) {
          rows.resize(data->size());
          std::vector<VectorPtr> inputForAggregation =
              makeInputForAggregation(data, aggregate);
          aggregate.function->addSingleGroupRawInput(
              group, rows, inputForAggregation, false);
        }
      }

      aggregate.function->extractValues(
          groups.data(), groups.size(), &result->childAt(aggregate.output));

      // Release memory back to HashStringAllocator to allow next
      // aggregate to re-use it.
      aggregate.function->destroy(groups);

      // Overwrite empty groups over the destructed groups to keep the container
      // in a well formed state.
      raw_vector<int32_t> indices(pool_);
      aggregate.function->initializeNewGroups(
          groups.data(),
          folly::Range<const int32_t*>(
              iota(groups.size(), indices), groups.size()));
    }
  }

  void addSingleGroupSpillInput(
      char* group,
      const VectorPtr& input,
      vector_size_t index) override {
    auto* elementArray = input->asChecked<ArrayVector>();
    decodedInput_.decode(*elementArray->elements());

    auto* accumulator = reinterpret_cast<AccumulatorType*>(group + offset_);
    RowSizeTracker<char, uint32_t> tracker(group[rowSizeOffset_], *allocator_);
    accumulator->addValues(*elementArray, index, decodedInput_, allocator_);
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    for (auto index : indices) {
      groups[index][nullByte_] |= nullMask_;
      new (groups[index] + offset_) AccumulatorType(inputType_, allocator_);
    }

    for (auto i = 0; i < aggregates_.size(); ++i) {
      const auto& aggregate = *aggregates_[i];
      aggregate.function->initializeNewGroups(groups, indices);
    }
  }

 private:
  void decodeInput(const RowVectorPtr& input, const SelectivityVector& rows) {
    inputForAccumulator_ = makeInputForAccumulator(input);
    decodedInput_.decode(*inputForAccumulator_, rows);
  }

  static TypePtr makeInputTypeForAccumulator(
      const RowTypePtr& rowType,
      const std::vector<column_index_t>& inputChannels) {
    const auto numInputChannels = inputChannels.size();
    if (numInputChannels == 1) {
      return rowType->childAt(inputChannels[0]);
    }

    // Otherwise, synthesize a ROW(distinct_channels[0..N])
    std::vector<TypePtr> types;
    types.reserve(numInputChannels);
    std::vector<std::string> names;
    names.reserve(numInputChannels);
    for (column_index_t inputChannel : inputChannels) {
      names.emplace_back(rowType->nameOf(inputChannel));
      types.emplace_back(rowType->childAt(inputChannel));
    }
    return ROW(std::move(names), std::move(types));
  }

  VectorPtr makeInputForAccumulator(const RowVectorPtr& input) const {
    if (singleNonConstantInput_) {
      return input->childAt(nonConstantInputs_[0]);
    }

    std::vector<VectorPtr> newChildren(nonConstantInputs_.size());
    for (size_t i = 0; i < nonConstantInputs_.size(); ++i) {
      newChildren[i] = input->childAt(nonConstantInputs_[i]);
    }
    return std::make_shared<RowVector>(
        pool_, inputType_, nullptr, input->size(), newChildren);
  }

  /// Build the full input vector list for the aggregate function from the
  /// extracted distinct values, splicing constant inputs back in at the
  /// correct positions.
  std::vector<VectorPtr> makeInputForAggregation(
      VectorPtr& data,
      const AggregateInfo& aggregate) const {
    const auto& inputs = aggregate.inputs;
    const auto& constants = aggregate.constantInputs;
    std::vector<VectorPtr> result(inputs.size());

    std::vector<VectorPtr> distinctColumns;
    if (singleNonConstantInput_) {
      distinctColumns.push_back(data);
    } else {
      distinctColumns = data->template asUnchecked<RowVector>()->children();
    }

    size_t nonConstantIndex = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i] == kConstantChannel) {
        result[i] = constants[i];
      } else {
        result[i] = distinctColumns[nonConstantIndex++];
      }
    }
    VELOX_DCHECK_EQ(nonConstantIndex, distinctColumns.size());
    return result;
  }

  void extractForSpill(folly::Range<char**> groups, VectorPtr& result) const {
    auto* arrayVector = result->asChecked<ArrayVector>();
    arrayVector->resize(groups.size());

    auto* rawOffsets = arrayVector->offsets()->asMutable<vector_size_t>();
    auto* rawSizes = arrayVector->sizes()->asMutable<vector_size_t>();

    vector_size_t offset = 0;
    for (auto i = 0; i < groups.size(); ++i) {
      auto* accumulator =
          reinterpret_cast<AccumulatorType*>(groups[i] + offset_);

      const auto numDistinct = accumulator->size();
      VELOX_DCHECK_GT(numDistinct, 0);

      rawSizes[i] = numDistinct;
      rawOffsets[i] = offset;

      offset += numDistinct;
    }

    auto& elementsVector = arrayVector->elements();
    elementsVector->resize(offset);

    offset = 0;
    for (const auto group : groups) {
      auto* accumulator = reinterpret_cast<AccumulatorType*>(group + offset_);
      if constexpr (std::is_same_v<T, ComplexType>) {
        offset += accumulator->extractValues(*elementsVector, offset);
      } else {
        offset += accumulator->extractValues(
            *(elementsVector->template as<FlatVector<T>>()), offset);
      }
    }
  }

  memory::MemoryPool* const pool_;
  const std::vector<AggregateInfo*> aggregates_;
  const std::vector<column_index_t> nonConstantInputs_;
  const TypePtr inputType_;
  const TypePtr spillType_;
  const bool singleNonConstantInput_;

  DecodedVector decodedInput_;
  VectorPtr inputForAccumulator_;
};

template <TypeKind Kind>
std::unique_ptr<DistinctAggregations>
createDistinctAggregationsWithCustomCompare(
    const std::vector<AggregateInfo*>& aggregates,
    const RowTypePtr& inputType,
    std::vector<column_index_t> nonConstantInputs,
    memory::MemoryPool* pool) {
  return std::make_unique<TypedDistinctAggregations<
      typename TypeTraits<Kind>::NativeType,
      aggregate::prestosql::CustomComparisonSetAccumulator<Kind>>>(
      aggregates, inputType, std::move(nonConstantInputs), pool);
}
} // namespace

// static
std::unique_ptr<DistinctAggregations> DistinctAggregations::create(
    std::vector<AggregateInfo*> aggregates,
    const RowTypePtr& inputType,
    memory::MemoryPool* pool) {
  VELOX_CHECK_EQ(aggregates.size(), 1);
  VELOX_CHECK(!aggregates[0]->inputs.empty());

  // Collect non-constant input channels to determine the type for the
  // set accumulator. Constant inputs are not deduplicated — they are
  // spliced back during extraction.
  std::vector<column_index_t> nonConstantInputs;
  for (auto i = 0; i < aggregates[0]->inputs.size(); ++i) {
    if (aggregates[0]->inputs[i] != kConstantChannel) {
      nonConstantInputs.push_back(aggregates[0]->inputs[i]);
    }
  }

  if (nonConstantInputs.empty()) {
    return std::make_unique<ConstantDistinctAggregations>(
        std::move(aggregates), pool);
  }

  if (nonConstantInputs.size() > 1) {
    return std::make_unique<TypedDistinctAggregations<ComplexType>>(
        std::move(aggregates), inputType, std::move(nonConstantInputs), pool);
  }

  const auto type = inputType->childAt(nonConstantInputs[0]);

  if (type->providesCustomComparison()) {
    return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        createDistinctAggregationsWithCustomCompare,
        type->kind(),
        aggregates,
        inputType,
        std::move(nonConstantInputs),
        pool);
  }

  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      return std::make_unique<TypedDistinctAggregations<bool>>(
          aggregates, inputType, std::move(nonConstantInputs), pool);
    case TypeKind::TINYINT:
      return std::make_unique<TypedDistinctAggregations<int8_t>>(
          aggregates, inputType, std::move(nonConstantInputs), pool);
    case TypeKind::SMALLINT:
      return std::make_unique<TypedDistinctAggregations<int16_t>>(
          aggregates, inputType, std::move(nonConstantInputs), pool);
    case TypeKind::INTEGER:
      return std::make_unique<TypedDistinctAggregations<int32_t>>(
          aggregates, inputType, std::move(nonConstantInputs), pool);
    case TypeKind::BIGINT:
      return std::make_unique<TypedDistinctAggregations<int64_t>>(
          aggregates, inputType, std::move(nonConstantInputs), pool);
    case TypeKind::HUGEINT:
      return std::make_unique<TypedDistinctAggregations<int128_t>>(
          aggregates, inputType, std::move(nonConstantInputs), pool);
    case TypeKind::REAL:
      return std::make_unique<TypedDistinctAggregations<float>>(
          aggregates, inputType, std::move(nonConstantInputs), pool);
    case TypeKind::DOUBLE:
      return std::make_unique<TypedDistinctAggregations<double>>(
          aggregates, inputType, std::move(nonConstantInputs), pool);
    case TypeKind::TIMESTAMP:
      return std::make_unique<TypedDistinctAggregations<Timestamp>>(
          aggregates, inputType, std::move(nonConstantInputs), pool);
    case TypeKind::VARBINARY:
      [[fallthrough]];
    case TypeKind::VARCHAR:
      return std::make_unique<TypedDistinctAggregations<StringView>>(
          aggregates, inputType, std::move(nonConstantInputs), pool);
    case TypeKind::ARRAY:
    case TypeKind::MAP:
    case TypeKind::ROW:
      return std::make_unique<TypedDistinctAggregations<ComplexType>>(
          aggregates, inputType, std::move(nonConstantInputs), pool);
    case TypeKind::UNKNOWN:
      return std::make_unique<TypedDistinctAggregations<UnknownValue>>(
          aggregates, inputType, std::move(nonConstantInputs), pool);
    default:
      VELOX_UNREACHABLE("Unexpected type {}", type->toString());
  }
}

} // namespace facebook::velox::exec
