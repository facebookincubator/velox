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

#include "velox/common/io/IoStatistics.h"
#include "velox/connectors/hive/HiveWriterTypes.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::connector::hive {

/// Defines the per-partition writer contract used by PartitionWriter.
/// Implementations can provide different write behaviors (e.g. rolling,
/// sorted, or non-rolling) without changing the routing logic.
class PartitionWriterInterface {
 public:
  virtual ~PartitionWriterInterface() = default;

  virtual void write(const VectorPtr& data) = 0;
  virtual bool finish() = 0;
  virtual void close() = 0;
  virtual void abort() = 0;

  virtual const std::shared_ptr<HiveWriterInfo>& writerInfo() const = 0;
  virtual io::IoStatistics* ioStats() const = 0;
};

} // namespace facebook::velox::connector::hive
