/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include <folly/Random.h>
#include "velox/core/Expressions.h"
#include "velox/vector/ComplexVector.h"

#pragma once

namespace facebook::velox::exec::test {

struct FeatureOptions {
  int32_t numFloat{10};
  int32_t numInt{10};
  int32_t numIdList{10};
  int32_t idListMaxCard{1000};
  int32_t idListMinCard{10};
  int32_t idListMaxDistinct{1000};
  int32_t numIdScoreList{5};

  /// Structs for use in reading the features. One field for each
  /// key. Filled in by makeFeatures().
  velox::RowTypePtr floatStruct;
  velox::RowTypePtr idListStruct;
  velox::RowTypePtr idScoreListStruct;

  // Parameters for generating test exprs.
  /// Number of projections for float one feature.
  int32_t floatExprsPct{130};
  int32_t idListExprPct{0};
  int32_t idScoreListExprPct{0};

  /// Percentage of projections that depend on multiple features.
  float multiColumnPct{20};

  /// Percentage of exprs with a rand.
  int32_t randomPct{20};

  /// Percentage of uid dependent exprs.
  int32_t uidPct{20};

  /// percentage of extra  + 1's.
  int32_t plusOnePct{20};

  mutable folly::Random::DefaultGenerator rng;

  bool coinToss(int32_t pct) const {
    return folly::Random::rand32(rng) % 100u < pct;
  }
};

std::vector<velox::RowVectorPtr> makeFeatures(
    int32_t numBatches,
    int32_t batchSize,
    FeatureOptions& opts,
    velox::memory::MemoryPool* pool);

void makeExprs(
    const FeatureOptions& opts,
    std::vector<std::string>& names,
    std::vector<velox::core::TypedExprPtr>& exprs);


} // namespace facebook::axiom::optimizer::test
