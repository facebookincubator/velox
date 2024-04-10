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

#include <exec/SimpleAggregateAdapter.h>
#include <type/SimpleFunctionApi.h>

namespace facebook::velox::functions::aggregate {
class MinMaxByAggregate {
 public:
  // Type(s) of input vector(s) wrapped in Row.
  using InputType = Row<Generic<T1>, Orderable<Generic<T2>>>;
  using IntermediateType = Row<Generic<T1>, Orderable<Generic<T2>>>;
  using OutputType = Array<Generic<T1>>;

  struct AccumulatorType {};
};
} // namespace facebook::velox::functions::aggregate
