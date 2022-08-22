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

#include "velox/exec/Task.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec {

struct CommitAwareOperator {
  static void buildDMLFinishingFragments(
      std::vector<VectorPtr>& columns,
      velox::memory::MemoryPool* pool) {
    // TODO Refactor to build fragments using the information returned by
    // UpdatableDataSource::finish() or DataSink::finish
    auto fragments = std::dynamic_pointer_cast<FlatVector<StringView>>(
        BaseVector::create(VARBINARY(), 1, pool));
    fragments->setNull(0, true);
    columns.emplace_back(fragments);
  }

  static void buildDMLCommitContext(
      std::vector<VectorPtr>& columns,
      DriverCtx* driverCtx,
      velox::memory::MemoryPool* pool) {
    // clang-format off
    auto commitContextJson = folly::toJson(
        folly::dynamic::object
            ("lifespan", "TaskWide")
            ("taskId", driverCtx->task->taskId())
            ("pageSinkCommitStrategy", "NO_COMMIT")
            ("lastPage", false));
    // clang-format on
    auto commitContext = std::make_shared<ConstantVector<StringView>>(
        pool, 1, false, VARBINARY(), StringView(commitContextJson));
    columns.emplace_back(commitContext);
  }

  static void buildDMLOutput(
      vector_size_t numRowChanged,
      DriverCtx* driverCtx,
      velox::memory::MemoryPool* pool,
      std::vector<VectorPtr>& columns,
      bool commitAwareness) {
    auto rowsChanged = std::dynamic_pointer_cast<FlatVector<int64_t>>(
        BaseVector::create(BIGINT(), 1, pool));
    rowsChanged->set(0, numRowChanged);
    columns.emplace_back(rowsChanged);

    // TODO Refactor This logic is coupling to the PrestoDB's DML workflow
    if (commitAwareness) {
      buildDMLFinishingFragments(columns, pool);
      buildDMLCommitContext(columns, driverCtx, pool);
    }
  }
};
} // namespace facebook::velox::exec