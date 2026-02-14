/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/common/file/FileSystems.h"
#include "velox/core/PlanNode.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/trace/TraceWriter.h"

namespace facebook::velox::exec::trace {

class TaskTraceMetadataWriter : public TraceMetadataWriter {
 public:
  TaskTraceMetadataWriter(
      std::string traceDir,
      std::string traceNodeId,
      memory::MemoryPool* pool);

  void write(const core::QueryCtx& queryCtx, const core::PlanNode& planNode)
      override;

 private:
  const std::string traceDir_;
  const std::string traceNodeId_;
  const std::shared_ptr<filesystems::FileSystem> fs_;
  const std::string traceFilePath_;
  bool finished_{false};
};
} // namespace facebook::velox::exec::trace
