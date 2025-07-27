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
#include "velox/common/base/IOUtils.h"
#include "velox/exec/Aggregate.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/prestosql/aggregates/AggregateNames.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>
#include "velox/exec/SimpleAggregateAdapter.h"

using namespace facebook::velox::exec;
namespace facebook::velox::aggregate::prestosql {

namespace {

// todo(wangke): move it to functions/lib
class NumericHistogram {
 private:
  class Entry {
   public:
    int id_;
    double value_;
    double weight_;
    double penalty_;
    Entry* right_;
    Entry* left_;
    mutable bool valid_;

    Entry(
        int id,
        double value,
        double weight,
        Entry* right = nullptr,
        Entry* left = nullptr)
        : id_(id),
          value_(value),
          weight_(weight),
          right_(right),
          left_(left),
          valid_(true) {
      if (right != nullptr) {
        penalty_ = computePenalty(value, weight, right->value_, right->weight_);
      } else {
        penalty_ = std::numeric_limits<double>::infinity();
      }
      if (left != nullptr) {
        left->right_ = this;
      }
      if (right != nullptr) {
        right->left_ = this;
      }
    }

    struct Compare {
      bool operator()(const Entry* current, const Entry* other) const {
        if (current->penalty_ == other->penalty_) {
          return current->id_ > other->id_;
        }
        return current->penalty_ > other->penalty_;
      }
    };

    void invalidate() {
      valid_ = false;
    }
  };

  double static computePenalty(
      double value1,
      double weight1,
      double value2,
      double weight2) {
    double weight = weight1 + weight2;
    double squaredDifference = (value1 - value2) * (value1 - value2);
    double proportionsProduct =
        (weight1 * weight2) / ((weight1 + weight2) * (weight1 + weight2));
    return weight * squaredDifference * proportionsProduct;
  }

  int maxBuckets_;
  int nextIndex_;
  std::vector<double, StlAllocator<double>> values_;
  std::vector<double, StlAllocator<double>> weights_;
  static const int ENTRY_BUFFER_SIZE = 100;

 public:
  NumericHistogram(int maxBuckets, HashStringAllocator* allocator)
      : maxBuckets_(maxBuckets),
        nextIndex_(0),
        values_{StlAllocator<double>(allocator)},
        weights_{StlAllocator<double>(allocator)} {
    VELOX_USER_CHECK_GE(maxBuckets, 2);
    values_.reserve(maxBuckets_ + ENTRY_BUFFER_SIZE);
    weights_.reserve(maxBuckets_ + ENTRY_BUFFER_SIZE);
  }

  NumericHistogram(common::InputByteStream& in, HashStringAllocator* allocator)
      : values_{StlAllocator<double>(allocator)},
        weights_{StlAllocator<double>(allocator)} {
    maxBuckets_ = in.read<int>();
    values_.reserve(maxBuckets_ + ENTRY_BUFFER_SIZE);
    weights_.reserve(maxBuckets_ + ENTRY_BUFFER_SIZE);
    nextIndex_ = in.read<int>();
    for (int i = 0; i < nextIndex_; ++i) {
      values_.push_back(in.read<double>());
      weights_.push_back(in.read<double>());
    }
  }

  size_t serializedSize() const {
    size_t size = sizeof(maxBuckets_) + sizeof(nextIndex_) +
        nextIndex_ * sizeof(double) // size of values_
        + nextIndex_ * sizeof(double); // size of weights_
    return size;
  }

  void serialize(velox::common::OutputByteStream& out) {
    out.appendOne(maxBuckets_);
    out.appendOne(nextIndex_);
    for (int i = 0; i < nextIndex_; ++i) {
      out.appendOne(values_[i]);
      out.appendOne(weights_[i]);
    }
  }

  // Add a value to the histogram
  void add(double value, double weight = 1.0) {
    if (nextIndex_ >= maxBuckets_ + ENTRY_BUFFER_SIZE) {
      compact();
    }
    if (nextIndex_ < values_.size()) {
      values_[nextIndex_] = value;
      weights_[nextIndex_] = weight;
    } else {
      values_.push_back(value);
      weights_.push_back(weight);
    }
    // after compact, nextIndex_ should be equal to maxbuckets

    ++nextIndex_;
  }

  void mergeWith(NumericHistogram& other) {
    for (auto i = 0; i < other.nextIndex_; ++i) {
      auto value = other.values_[i];
      auto weight = other.weights_[i];
      auto iter = std::find(this->values_.begin(), this->values_.end(), value);
      if (iter != this->values_.end()) {
        weights_[iter - this->values_.begin()] += weight;
      } else {
        add(value, weight);
      }
    }

    // todo: we don't need to sort here, since we don't merge same buckets
    sortValuesAndWeights();
    mergeAndReduceBuckets();
  }

  // what compact does is:
  // 1. merge the same buckets - may or may not reduce nextIndex (if there are
  // no elements with same value)
  // 2. if 1 did not reduce it enough, mergeAndReduceBuckets will
  // merge and reduce it until nextIndex is smaller than maxBuckets
  void compact() {
    mergeSameBuckets();
    mergeAndReduceBuckets();
  }

  // Initialize a priority queue with Entries for each value and weight
  std::priority_queue<Entry*, std::vector<Entry*>, Entry::Compare> initQueue(
      std::vector<Entry*>& allocatedEntries) {
    std::priority_queue<Entry*, std::vector<Entry*>, Entry::Compare> queue;

    Entry* right = new Entry(
        nextIndex_ - 1, values_[nextIndex_ - 1], weights_[nextIndex_ - 1]);
    allocatedEntries.push_back(right);
    queue.push(right);
    for (int i = nextIndex_ - 2; i >= 0; i--) {
      Entry* current = new Entry(i, values_[i], weights_[i], right);
      queue.push(current);
      allocatedEntries.push_back(current);
      right = current;
    }

    return queue;
  }

  // Merge and reduce buckets to max number of buckets
  void mergeAndReduceBuckets() {
    if (nextIndex_ <= maxBuckets_) {
      return;
    }

    // add check for targetcount

    std::vector<Entry*> allocatedEntries;
    auto queue = initQueue(allocatedEntries);

    // merge and reduce entries in queue until the count is equal to targetCount
    while (nextIndex_ > maxBuckets_) {
      Entry* current = queue.top();
      queue.pop();
      if (!current->valid_) {
        // already replaced, move on
        continue;
      }
      nextIndex_--;
      current->invalidate();

      // right is guaranteed to exist because we set the penalty of the last
      // bucket to infinity so the first current in the queue can never be the
      // last bucket
      Entry* right = current->right_;
      if (right == nullptr) {
        LOG(ERROR) << "current id = " << current->id_
                   << "value = " << current->value_
                   << "weight = " << current->weight_
                   << "penalty = " << current->penalty_ << std::endl;
        throw std::runtime_error("Right entry is null");
      }
      if (!right->valid_) {
        throw std::runtime_error("Right entry is not valid");
      }

      right->invalidate();

      // merge "current" with "right" and mark "right" as invalid so we don't
      // visit it again
      double newWeight = current->weight_ + right->weight_;
      double newValue = (current->value_ * current->weight_ +
                         right->value_ * right->weight_) /
          newWeight;

      Entry* merged =
          new Entry(current->id_, newValue, newWeight, right->right_);

      allocatedEntries.push_back(merged);
      queue.push(merged);

      // update left's penalty
      Entry* left = current->left_;
      if (left != nullptr) {
        if (!left->valid_) {
          throw std::runtime_error("Left entry is not valid");
        }
        left->invalidate();
        Entry* newLeft = new Entry(
            left->id_, left->value_, left->weight_, merged, left->left_);
        allocatedEntries.push_back(newLeft);
        queue.push(newLeft);
      }
    }
    // re populate values_ and weights_ with the reduced queue
    nextIndex_ = 0;
    while (!queue.empty()) {
      Entry* entry = queue.top();
      queue.pop();
      if (entry->valid_) {
        values_[nextIndex_] = entry->value_;
        weights_[nextIndex_] = entry->weight_;
        ++nextIndex_;
      }
    }

    sortValuesAndWeights();
    for (Entry* entry : allocatedEntries) {
      delete entry;
    }
  }

  // Merge same buckets in place and update nextIndex_
  void mergeSameBuckets() {
    sortValuesAndWeights();

    int current = 0;
    for (int i = 1; i < nextIndex_; ++i) {
      // todo: double equality should be replaced with a tolerance
      if (values_[current] == values_[i]) {
        weights_[current] += weights_[i];
      } else {
        ++current;
        values_[current] = values_[i];
        weights_[current] = weights_[i];
      }
    }

    nextIndex_ = current + 1;
  }

  // Sort the values and weights
  void sortValuesAndWeights() {
    std::vector<std::pair<double, double>> pairs(nextIndex_);
    for (int i = 0; i < nextIndex_; ++i) {
      pairs[i] = {values_[i], weights_[i]};
    }
    std::sort(pairs.begin(), pairs.end());
    for (int i = 0; i < nextIndex_; ++i) {
      values_[i] = pairs[i].first;
      weights_[i] = pairs[i].second;
    }
  }

  void getBuckets(exec::out_type<Map<double, double>>& out) {
    compact();
    for (int i = 0; i < nextIndex_; ++i) {
      auto [keyWriter, valueWriter] = out.add_item();
      keyWriter = values_[i];
      valueWriter = weights_[i];
    }
  }

  void getBuckets(exec::out_type<Map<float, float>>& out) {
    compact();
    for (int i = 0; i < nextIndex_; ++i) {
      auto [keyWriter, valueWriter] = out.add_item();
      keyWriter = values_[i];
      valueWriter = weights_[i];
    }
  }
};

template <typename T, typename Func>
struct NumericHistogramAccumulator {
  static constexpr bool is_fixed_size_ = false;
  static constexpr bool use_external_memory_ = true;
  static constexpr bool is_aligned_ = true;

  std::optional<NumericHistogram> histogram_;

  NumericHistogramAccumulator() = delete;

  explicit NumericHistogramAccumulator(
      HashStringAllocator* /*allocator*/,
      Func* /*fn*/) {}

  bool addInput(
      HashStringAllocator* allocator,
      exec::arg_type<int64_t> buckets,
      exec::arg_type<double> value) {
    if (!histogram_.has_value()) {
      histogram_.emplace(buckets, allocator);
    }
    histogram_.value().add(value, 1.0);
    return true;
  }

  bool addInput(
      HashStringAllocator* allocator,
      exec::arg_type<int64_t> buckets,
      exec::arg_type<float> value) {
    if (!histogram_.has_value()) {
      histogram_.emplace(buckets, allocator);
    }
    histogram_.value().add(value, 1.0);
    return true;
  }

  bool addInput(
      HashStringAllocator* allocator,
      exec::arg_type<int64_t> buckets,
      exec::arg_type<double> value,
      exec::arg_type<double> weight) {
    if (!histogram_.has_value()) {
      histogram_.emplace(buckets, allocator);
    }
    histogram_.value().add(value, weight);
    return true;
  }

  bool addInput(
      HashStringAllocator* allocator,
      exec::arg_type<int64_t> buckets,
      exec::arg_type<float> value,
      exec::arg_type<double> weight) {
    if (!histogram_.has_value()) {
      histogram_.emplace(buckets, allocator);
    }
    histogram_.value().add(value, weight);
    return true;
  }

  void combine(
      HashStringAllocator* allocator,
      exec::arg_type<Varbinary> other) {
    common::InputByteStream stream((other.data()));
    auto otherHistogram = NumericHistogram(stream, allocator);

    if (!histogram_.has_value()) {
      histogram_.emplace(otherHistogram);
    } else {
      histogram_.value().mergeWith(otherHistogram);
    }
  }

  bool writeIntermediateResult(exec::out_type<Varbinary>& out) {
    if (histogram_.has_value()) {
      const auto serializedSize = histogram_.value().serializedSize();
      out.reserve(serializedSize);
      common::OutputByteStream stream(out.data());
      // histogram_.value().compact();
      histogram_.value().serialize(stream);

      out.resize(serializedSize);
      return true;
    } else {
      return false;
    }
  }

  bool writeFinalResult(exec::out_type<Map<T, T>>& out) {
    if (histogram_.has_value()) {
      histogram_.value().getBuckets(out);
    }

    return true;
  }
};

template <typename T, int NumArgs>
class NumericHistogramAggregate {};

template <typename T>
class NumericHistogramAggregate<T, 2> {
 public:
  using InputType = Row<int64_t, T>;

  using IntermediateType = Varbinary;

  using OutputType = Map<T, T>;

  using AccumulatorType =
      NumericHistogramAccumulator<T, NumericHistogramAggregate<T, 2>>;
};

template <typename T>
class NumericHistogramAggregate<T, 3> {
 public:
  using InputType = Row<int64_t, T, double>;

  using IntermediateType = Varbinary;

  using OutputType = Map<T, T>;

  using AccumulatorType =
      NumericHistogramAccumulator<T, NumericHistogramAggregate<T, 3>>;
};
} // namespace

void registerNumericHistogramAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  const auto valueTypes = {"REAL", "DOUBLE"};
  const auto weightTypes = {"DOUBLE"};
  for (const auto& valueType : valueTypes) {
    const auto returnType = fmt::format("map({}, {})", valueType, valueType);
    signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                             .returnType(returnType)
                             .intermediateType("varbinary")
                             .argumentType("bigint")
                             .argumentType(valueType)
                             .build());
    for (const auto& weightType : weightTypes) {
      signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                               .returnType(returnType)
                               .intermediateType("varbinary")
                               .argumentType("bigint")
                               .argumentType(valueType)
                               .argumentType(weightType)
                               .build());
    }
  }
  auto name = prefix + kNumericHistogram;

  exec::registerAggregateFunction(
      name,
      signatures,
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_USER_CHECK_GE(
            argTypes.size(), 2, "{} takes at least two arguments", name);
        VELOX_USER_CHECK_LE(
            argTypes.size(), 3, "{} takes at most three arguments", name);
        if (argTypes[0]->kind() != TypeKind::BIGINT) {
          VELOX_NYI(
              "aggregation {}: Buckets must be bigint {}, but is {}",
              name,
              argTypes[0]->kindName()); // check java error message
        }
        if (argTypes[1]->kind() != TypeKind::REAL &&
            argTypes[1]->kind() != TypeKind::DOUBLE) {
          VELOX_NYI(
              "aggregation {}: Value must be REAL or DOUBLE {}, but is {}",
              name,
              argTypes[1]->kindName()); // check java error message
        }
        if (argTypes.size() == 2) {
          switch (argTypes[1]->kind()) {
            case TypeKind::REAL:
              return std::make_unique<exec::SimpleAggregateAdapter<
                  NumericHistogramAggregate<float, 2>>>(
                  step, argTypes, resultType);
            case TypeKind::DOUBLE:
              return std::make_unique<exec::SimpleAggregateAdapter<
                  NumericHistogramAggregate<double, 2>>>(
                  step, argTypes, resultType);

            default:
              VELOX_NYI("Unknown input type for {} aggregation {}", name);
          }
        } else {
          switch (argTypes[1]->kind()) {
            case TypeKind::REAL:
              return std::make_unique<exec::SimpleAggregateAdapter<
                  NumericHistogramAggregate<float, 3>>>(
                  step, argTypes, resultType);
            case TypeKind::DOUBLE:
              return std::make_unique<exec::SimpleAggregateAdapter<
                  NumericHistogramAggregate<double, 3>>>(
                  step, argTypes, resultType);

            default:
              VELOX_NYI("Unknown input type for {} aggregation {}", name);
          }
        }
      },
      withCompanionFunctions,
      overwrite);
}
} // namespace facebook::velox::aggregate::prestosql
