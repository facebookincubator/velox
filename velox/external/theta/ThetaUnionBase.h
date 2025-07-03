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

#pragma once

#include "ThetaUpdateSketchBase.h"

namespace facebook::velox::common::theta {

template <
    typename Entry,
    typename ExtractKey,
    typename Policy,
    typename Sketch,
    typename CompactSketch,
    typename Allocator>
class ThetaUnionBase {
 public:
  using hashTable = ThetaUpdateSketchBase<Entry, ExtractKey, Allocator>;
  using resizeFactor = typename hashTable::resizeFactor;
  using comparator = compareByKey<ExtractKey>;

  ThetaUnionBase(
      uint8_t lg_cur_size,
      uint8_t lg_nom_size,
      resizeFactor rf,
      float p,
      uint64_t theta,
      uint64_t seed,
      const Policy& policy,
      const Allocator& allocator);

  template <typename FwdSketch>
  void update(FwdSketch&& sketch);

  CompactSketch getResult(bool ordered = true) const;

  const Policy& getPolicy() const;

  void reset();

 private:
  Policy policy_;
  hashTable table_;
  uint64_t union_theta_;
};

} // namespace facebook::velox::common::theta

#include "ThetaUnionBase.cpp"
