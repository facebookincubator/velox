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

#include <memory>
#include <optional>
#include <string>

#include "velox/core/PlanNode.h"
#include "velox/exec/OutputBuffer.h"

namespace facebook::velox::exec {

class Task;

class IOutputBufferManager {
 public:
  virtual ~IOutputBufferManager() = default;

  virtual void initializeTask(
      std::shared_ptr<Task> task,
      core::PartitionedOutputNode::Kind kind,
      int numDestinations,
      int numDrivers) = 0;

  virtual bool updateOutputBuffers(
      const std::string& taskId,
      int numBuffers,
      bool noMoreBuffers) = 0;

  virtual void removeTask(const std::string& taskId) = 0;

  virtual std::optional<OutputBuffer::Stats> stats(
      const std::string& taskId) = 0;
};

} // namespace facebook::velox::exec
