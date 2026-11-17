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

#include <functional>

#include "velox/core/MultiStagePlan.h"
#include "velox/exec/Split.h"

namespace facebook::velox::exec {

class Task;

/// Wires the exchanges between the tasks running the stages of a
/// core::MultiStagePlan.  Every runtime wires them the same way -- a query
/// runner running the stages of a query, or a fixed point running the stages of
/// its body on its workers each iteration; only the split that reaches a
/// producer task differs.
///
/// For each consumer task, the runtime calls addInputSplits(); for each task of
/// a producer, updateOutputBuffers() with the number of tasks of its consumer.
class ExchangeWiring {
 public:
  /// Tells 'producer' how many tasks read its output when that output, rooted
  /// at 'producerRoot', broadcasts or partitions arbitrarily: such an output
  /// cannot learn the number from the plan, and it finishes only once every
  /// destination has drained it.  Does nothing for any other output.
  static void updateOutputBuffers(
      Task& producer,
      const core::PlanNode& producerRoot,
      int32_t numConsumerTasks);

  /// Gives 'task', a task running 'consumer', one split per producer task for
  /// each of 'consumer's inputs, then marks each exchange complete, once the
  /// splits of every stage it reads are added.
  /// 'numProducerTasks' returns how many tasks run the stage with the given
  /// id, and 'split' returns the split reaching one of them.
  static void addInputSplits(
      Task& task,
      const core::Stage& consumer,
      const std::function<int32_t(int32_t producerStageId)>& numProducerTasks,
      const std::function<Split(int32_t producerStageId, int32_t producer)>&
          split);
};

} // namespace facebook::velox::exec
