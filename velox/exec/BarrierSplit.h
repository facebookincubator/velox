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

#include <cstdint>

namespace facebook::velox::exec {

/// A special split used by task barrier processing to signal output drain
/// processing. When task barrier processing is triggered, one barrier split is
/// added to each leaf source node. Once a source node receives the barrier
/// split, it will produce output and propagate the barrier down to the root
/// node of a pipeline (typically exchange or partitioned output).
struct BarrierSplit {
  /// The number of drivers in the source pipeline that share this barrier
  /// split. It is used to deduplicate barrier processing at the root node of
  /// the pipeline which blocks the barrier processing until all the pipelines
  /// drivers have reached it.
  uint32_t numDrivers;
};

} // namespace facebook::velox::exec
