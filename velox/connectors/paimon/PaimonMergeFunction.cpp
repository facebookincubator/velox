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
#include "velox/connectors/paimon/PaimonMergeFunction.h"

#include "velox/connectors/paimon/PaimonFieldAggregator.h"

namespace facebook::velox::connector::paimon {
namespace {

// Paimon Java's AggregateMergeFunction: each column aggregates the column of
// the rows of a key in merge order. INSERT and UPDATE_AFTER rows add their
// values, and DELETE and UPDATE_BEFORE rows retract them. With
// 'aggregation.remove-record-on-delete', a DELETE row instead replaces the
// merged row with its own values, and the merged row is a DELETE if that row
// comes last. 'ignore-delete' does not apply here.
//
// Paimon Java also rejects a DELETE row with a null in a NOT NULL column,
// which Velox types cannot tell apart.
class AggregateMergeFunction final : public PaimonMergeFunction {
 public:
  AggregateMergeFunction(
      std::vector<column_index_t> channels,
      std::vector<std::unique_ptr<PaimonFieldAggregator>> aggregators,
      bool removeRecordOnDelete)
      : channels_(std::move(channels)),
        aggregators_(std::move(aggregators)),
        removeRecordOnDelete_(removeRecordOnDelete) {
    VELOX_CHECK_EQ(channels_.size(), aggregators_.size());
  }

  void reset() override {
    for (auto& aggregator : aggregators_) {
      aggregator->reset();
    }
    deleted_ = false;
  }

  void add(const RowVector& input, vector_size_t row, PaimonRowKind kind)
      override {
    deleted_ = removeRecordOnDelete_ && kind == PaimonRowKind::kDelete;
    if (deleted_) {
      // The DELETE row does not retract. Each column takes the value of the
      // DELETE row, and the functions keep their state, as in Paimon Java. A
      // later row of the key merges into these values, e.g. 'sum' over 10
      // (INSERT), 5 (DELETE) and 7 (INSERT) is 12. If no row follows,
      // resultKind() is DELETE and the reader drops the key.
      for (size_t i = 0; i < channels_.size(); ++i) {
        aggregators_[i]->set(*input.childAt(channels_[i]), row);
      }
      return;
    }
    const bool retract = isRetract(kind);
    for (size_t i = 0; i < channels_.size(); ++i) {
      const auto& column = *input.childAt(channels_[i]);
      if (retract) {
        aggregators_[i]->retract(column, row);
      } else {
        aggregators_[i]->add(column, row);
      }
    }
  }

  PaimonRowKind resultKind() const override {
    return deleted_ ? PaimonRowKind::kDelete : PaimonRowKind::kInsert;
  }

  void write(vector_size_t row, RowVector& output) const override {
    for (size_t i = 0; i < channels_.size(); ++i) {
      aggregators_[i]->write(row, *output.childAt(channels_[i]));
    }
  }

 private:
  const std::vector<column_index_t> channels_;
  const std::vector<std::unique_ptr<PaimonFieldAggregator>> aggregators_;
  // 'aggregation.remove-record-on-delete': whether a DELETE row replaces the
  // merged row with its own values instead of retracting them.
  const bool removeRecordOnDelete_;

  // Whether the last row was a DELETE that replaced the merged row.
  bool deleted_{false};
};

std::unique_ptr<PaimonMergeFunction> createAggregateMergeFunction(
    const RowTypePtr& type,
    const std::vector<column_index_t>& channels,
    const PaimonTableHandle::MergeOptions& options) {
  std::vector<std::unique_ptr<PaimonFieldAggregator>> aggregators;
  aggregators.reserve(channels.size());
  for (const auto channel : channels) {
    const auto& name = type->nameOf(channel);
    const auto it = options.fieldAggregations.find(name);
    VELOX_CHECK(
        it != options.fieldAggregations.end(),
        "Paimon aggregation table has no aggregate function for column {}",
        name);
    const auto& aggregation = it->second;
    if (aggregation.ignoreRetract) {
      VELOX_USER_CHECK(
          !options.aggregationRemoveRecordOnDelete,
          "Paimon table options '{}' and '{}' cannot both be true",
          PaimonTableOptions::kAggregationRemoveRecordOnDelete,
          PaimonTableOptions::fieldIgnoreRetract(name));
    }
    aggregators.emplace_back(
        PaimonFieldAggregator::create(
            aggregation.function,
            name,
            type->childAt(channel),
            aggregation.ignoreRetract));
  }
  return std::make_unique<AggregateMergeFunction>(
      channels,
      std::move(aggregators),
      options.aggregationRemoveRecordOnDelete);
}

} // namespace

// static
std::unique_ptr<PaimonMergeFunction> PaimonMergeFunction::create(
    const RowTypePtr& type,
    const std::vector<column_index_t>& channels,
    const PaimonTableHandle::MergeOptions& options) {
  switch (options.mergeEngine) {
    case PaimonMergeEngine::kAggregation:
      return createAggregateMergeFunction(type, channels, options);
    case PaimonMergeEngine::kDeduplicate:
    case PaimonMergeEngine::kPartialUpdate:
    case PaimonMergeEngine::kFirstRow:
      VELOX_NYI(
          "Paimon merge function does not support the merge engine: {}",
          options.mergeEngine);
    default:
      VELOX_FAIL(
          "Unknown PaimonMergeEngine: {}",
          static_cast<int>(options.mergeEngine));
  }
}

} // namespace facebook::velox::connector::paimon
