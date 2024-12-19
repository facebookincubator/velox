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

#include "velox/optimizer/RelationOp.h"

namespace facebook::velox::optimizer {

/// Interface to historical query cost and cardinality
/// information. There is one long lived instance per
/// process. Public functions are thread safe since multiple
/// concurrent Optimizations may access and update the same History.
class History {
 public:
  virtual ~History() = default;

  /// Returns a historical cost for 'op' or nullopt ir if no data. Matches the
  /// full 'op' tree to history. The exactness of the match depends on the
  /// History implementation. Implementations may extrapolate  from approximate
  /// matches. A Cost from this will be used over a non-historical cost if
  /// available. Not const since some extrapolated data can be kept.
  virtual std::optional<Cost> findCost(RelationOp& op) = 0;

  /// Records that the cost and cardinality of 'op' was 'cost' as observed from
  /// execution.
  virtual void recordCost(const RelationOp& op, Cost cost) = 0;

  /// Sets 'filterSelectivity' of 'baseTable' from historical data. Considers
  /// filters only and does not return a cost since the cost depends on the
  /// columns extracted. This is used first for coming up with join orders. The
  /// plan candidates are then made and findCost() is used to access historical
  /// cost and plan cardinality.
  virtual bool setLeafSelectivity(BaseTable& baseTable) = 0;

  virtual void recordLeafSelectivity(
      const std::string& handle,
      float selectivity,
      bool overwrite = true) {
    std::lock_guard<std::mutex> l(mutex_);
    if (!overwrite &&
        leafSelectivities_.find(handle) != leafSelectivities_.end()) {
      return;
    }
    leafSelectivities_[handle] = selectivity;
  }

 protected:
  // serializes access to all data members.
  std::mutex mutex_;

  /// Memo for selectivity keyed on ConnectorTableHandle::toString().
  /// Values between 0 and 1.
  std::unordered_map<std::string, float> leafSelectivities_;
};

float shuffleCost(const ColumnVector& columns);

float shuffleCost(const ExprVector& columns);

} // namespace facebook::velox::optimizer
