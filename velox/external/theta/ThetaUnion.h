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

#include "Serde.h"
#include "ThetaSketch.h"
#include "ThetaUnionBase.h"

namespace facebook::velox::common::theta {

// forward declaration
template <typename A>
class ThetaUnionAlloc;

// alias with default allocator for convenience
using ThetaUnion = ThetaUnionAlloc<std::allocator<uint64_t>>;

/**
 * Theta Union.
 * Computes union of Theta sketches. There is no constructor. Use builder
 * instead.
 */
template <typename Allocator = std::allocator<uint64_t>>
class ThetaUnionAlloc {
 public:
  using Entry = uint64_t;
  using ExtractKey = trivialExtractKey;
  using Sketch = ThetaSketchAlloc<Allocator>;
  using CompactSketch = CompactThetaSketchAlloc<Allocator>;
  using resizeFactor = ThetaConstants::resizeFactor;

  // there is no payload in Theta sketch entry
  struct nop_policy {
    void operator()(uint64_t internal_entry, uint64_t incoming_entry) const {
      unused(internal_entry);
      unused(incoming_entry);
    }
  };
  using State = ThetaUnionBase<
      Entry,
      ExtractKey,
      nop_policy,
      Sketch,
      CompactSketch,
      Allocator>;

  // No constructor here. Use builder instead.
  class builder;

  /**
   * Update the union with a given sketch
   * @param sketch to update the union with
   */
  template <typename FwdSketch>
  void update(FwdSketch&& sketch);

  /**
   * Produces a copy of the current state of the union as a compact sketch.
   * @param ordered optional flag to specify if an ordered sketch should be
   * produced
   * @return the result of the union
   */
  CompactSketch getResult(bool ordered = true) const;

  /// Reset the union to the initial empty state
  void reset();

 private:
  State state_;

  // for builder
  ThetaUnionAlloc(
      uint8_t lg_cur_size,
      uint8_t lg_nom_size,
      resizeFactor rf,
      float p,
      uint64_t theta,
      uint64_t seed,
      const Allocator& allocator);
};

/// Theta union builder
template <typename A>
class ThetaUnionAlloc<A>::builder : public ThetaBaseBuilder<builder, A> {
 public:
  builder(const A& allocator = A());

  /**
   * Create an instance of the union with predefined parameters.
   * @return an instance of the union
   */
  ThetaUnionAlloc<A> build() const;
};

} // namespace facebook::velox::common::theta

#include "ThetaUnion.cpp"
