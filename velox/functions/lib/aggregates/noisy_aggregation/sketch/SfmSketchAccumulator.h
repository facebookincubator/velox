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

#include "velox/common/memory/HashStringAllocator.h"
#include "velox/functions/lib/aggregates/noisy_aggregation/sketch/SfmSketch.h"

namespace facebook::velox::functions::aggregate {

class SfmSketchAccumulator {
  using SfmSketch = facebook::velox::functions::aggregate::SfmSketch;
  const uint32_t DEFAULTBUCKETS = 4096;
  const uint32_t DEFAULTPRECISION = 24;

 public:
  explicit SfmSketchAccumulator(HashStringAllocator* allocator)
      : sketch_(
            SfmSketch::create(DEFAULTBUCKETS, DEFAULTPRECISION, allocator)) {}

  // Constructor for deserialization.
  explicit SfmSketchAccumulator(
      double epsilon,
      bool isSketchSet,
      SfmSketch sketch)
      : sketch_(std::move(sketch)) {
    epsilon_ = epsilon;
    isSketchSet_ = isSketchSet;
  }

  // Get the sketch from the accumulator, access and update the sketch.
  SfmSketch& getSketch() {
    return sketch_;
  }

  // Get the epsilon from the state.
  double getEpsilon() const {
    return epsilon_;
  }

  void setEpsilon(double epsilon) {
    epsilon_ = epsilon;
  }

  // If epsilon is set, the group is null or input is empty.
  bool isValid() const {
    return epsilon_ > 0;
  }

  size_t serializedSize() const {
    return sizeof(epsilon_) + sizeof(bool) + sketch_.serializedSize();
  }

  void serialize(char* outputBuffer) const {
    common::OutputByteStream stream(outputBuffer);
    stream.appendOne(epsilon_);
    stream.appendOne(isSketchSet_);
    sketch_.serialize(outputBuffer + stream.offset());
  }

  static SfmSketchAccumulator deserialize(
      const char* inputBuffer,
      HashStringAllocator* allocator) {
    common::InputByteStream stream(inputBuffer);
    auto epsilon = stream.read<double>();
    auto isSketchSet = stream.read<bool>();
    auto sketch =
        SfmSketch::deserialize(inputBuffer + stream.offset(), allocator);
    return SfmSketchAccumulator{epsilon, isSketchSet, sketch};
  }

 private:
  double epsilon_{-1.0};
  bool isSketchSet_{false};
  SfmSketch sketch_;
};

} // namespace facebook::velox::functions::aggregate
