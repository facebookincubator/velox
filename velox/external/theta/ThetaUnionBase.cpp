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

// Adapted from Apache DataSketches

#ifndef THETA_UNION_BASE_CPP
#define THETA_UNION_BASE_CPP

#include <algorithm>

#include "ConditionalForward.h"
#include "ThetaUnionBase.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common::theta {

template <
    typename EN,
    typename EK,
    typename P,
    typename S,
    typename CS,
    typename A>
ThetaUnionBase<EN, EK, P, S, CS, A>::ThetaUnionBase(
    uint8_t lg_cur_size,
    uint8_t lg_nom_size,
    resizeFactor rf,
    float p,
    uint64_t theta,
    uint64_t seed,
    const P& policy,
    const A& allocator)
    : policy_(policy),
      table_(lg_cur_size, lg_nom_size, rf, p, theta, seed, allocator),
      union_theta_(table_.theta_) {}

template <
    typename EN,
    typename EK,
    typename P,
    typename S,
    typename CS,
    typename A>
template <typename SS>
void ThetaUnionBase<EN, EK, P, S, CS, A>::update(SS&& sketch) {
  if (sketch.isEmpty())
    return;
  if (sketch.getSeedHash() != compute_seed_hash(table_.seed_)) {
    throw VeloxRuntimeError(
        __FILE__,
        __LINE__,
        __FUNCTION__,
        "",
        "seed hash mismatch",
        error_source::kErrorSourceRuntime,
        error_code::kUnknown,
        false /*retriable*/);
  }
  table_.isEmpty_ = false;
  union_theta_ = std::min(union_theta_, sketch.getTheta64());
  for (auto&& entry : sketch) {
    const uint64_t hash = EK()(entry);
    if (hash < union_theta_ && hash < table_.theta_) {
      auto result = table_.find(hash);
      if (!result.second) {
        table_.insert(result.first, conditionalForward<SS>(entry));
      } else {
        policy_(*result.first, conditionalForward<SS>(entry));
      }
    } else {
      if (sketch.isOrdered())
        break; // early stop
    }
  }
  union_theta_ = std::min(union_theta_, table_.theta_);
}

template <
    typename EN,
    typename EK,
    typename P,
    typename S,
    typename CS,
    typename A>
CS ThetaUnionBase<EN, EK, P, S, CS, A>::getResult(bool ordered) const {
  std::vector<EN, A> entries(table_.allocator_);
  if (table_.isEmpty_)
    return CS(
        true,
        true,
        compute_seed_hash(table_.seed_),
        union_theta_,
        std::move(entries));
  entries.reserve(table_.numEntries_);
  uint64_t theta = std::min(union_theta_, table_.theta_);
  const uint32_t nominal_num = 1 << table_.lgNomSize_;
  if (union_theta_ >= table_.theta_) {
    std::copy_if(
        table_.begin(),
        table_.end(),
        std::back_inserter(entries),
        keyNotZero<EN, EK>());
  } else {
    std::copy_if(
        table_.begin(),
        table_.end(),
        std::back_inserter(entries),
        keyNotZeroLessThan<uint64_t, EN, EK>(theta));
  }
  if (entries.size() > nominal_num) {
    std::nth_element(
        entries.begin(),
        entries.begin() + nominal_num,
        entries.end(),
        comparator());
    theta = EK()(entries[nominal_num]);
    entries.erase(entries.begin() + nominal_num, entries.end());
    entries.shrink_to_fit();
  }
  if (ordered)
    std::sort(entries.begin(), entries.end(), comparator());
  return CS(
      table_.isEmpty_,
      ordered,
      compute_seed_hash(table_.seed_),
      theta,
      std::move(entries));
}

template <
    typename EN,
    typename EK,
    typename P,
    typename S,
    typename CS,
    typename A>
const P& ThetaUnionBase<EN, EK, P, S, CS, A>::getPolicy() const {
  return policy_;
}

template <
    typename EN,
    typename EK,
    typename P,
    typename S,
    typename CS,
    typename A>
void ThetaUnionBase<EN, EK, P, S, CS, A>::reset() {
  table_.reset();
  union_theta_ = table_.theta_;
}

} // namespace facebook::velox::common::theta

#endif
