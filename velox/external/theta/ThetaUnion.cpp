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

#ifndef THETA_UNION_CPP
#define THETA_UNION_CPP

#include "ThetaUnion.h"

namespace facebook::velox::common::theta {

template <typename A>
ThetaUnionAlloc<A>::ThetaUnionAlloc(
    uint8_t lgCurSize,
    uint8_t lgNomSize,
    resizeFactor rf,
    float p,
    uint64_t theta,
    uint64_t seed,
    const A& allocator)
    : state_(
          lgCurSize,
          lgNomSize,
          rf,
          p,
          theta,
          seed,
          nop_policy(),
          allocator) {}

template <typename A>
template <typename FwdSketch>
void ThetaUnionAlloc<A>::update(FwdSketch&& sketch) {
  state_.update(std::forward<FwdSketch>(sketch));
}

template <typename A>
auto ThetaUnionAlloc<A>::getResult(bool ordered) const -> CompactSketch {
  return state_.getResult(ordered);
}

template <typename A>
void ThetaUnionAlloc<A>::reset() {
  state_.reset();
}

template <typename A>
ThetaUnionAlloc<A>::builder::builder(const A& allocator)
    : ThetaBaseBuilder<builder, A>(allocator) {}

template <typename A>
auto ThetaUnionAlloc<A>::builder::build() const -> ThetaUnionAlloc {
  return ThetaUnionAlloc(
      this->startingLgSize(),
      this->lg_k_,
      this->rf_,
      this->p_,
      this->startingTheta(),
      this->seed_,
      this->allocator_);
}

} // namespace facebook::velox::common::theta

#endif
