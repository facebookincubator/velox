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
#include "velox/exec/Aggregate.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/prestosql/aggregates/AggregateNames.h"
#include "velox/functions/prestosql/aggregates/SimpleNumericAggregate.h"
#include "velox/functions/prestosql/aggregates/SingleValueAccumulator.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::exec {

class MaxSizeForStatsEstimator {
 public:
  // Estimate the total size of elements in the range [offset, offset+length).
  // Adds the total size to size_out.
  // Recursively travers source if it's ComplexType vector.
  void estimateSizeOfVectorElements(
      const BaseVector& source,
      vector_size_t offset,
      vector_size_t length,
      size_t& size_out) const;

  static const MaxSizeForStatsEstimator& instance();
};

} // namespace facebook::velox::exec