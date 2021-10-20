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

#include "velox/vector/LazyVector.h"
#include <folly/ThreadLocal.h>
#include "velox/common/time/Timer.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox {

// Thread local stat writer, if set (not null) are used here to record how much
// time was spent on IO in lazy vectors.
static folly::ThreadLocalPtr<BaseRuntimeStatWriter> sRunTimeStatWriters;

void setRunTimeStatWriter(std::unique_ptr<BaseRuntimeStatWriter>&& ptr) {
  sRunTimeStatWriters.reset(std::move(ptr));
}

static void writeIOWallTimeStat(size_t ioTimeStartMicros) {
  if (BaseRuntimeStatWriter* pWriter = sRunTimeStatWriters.get()) {
    pWriter->addRuntimeStat(
        "dataSourceLazyWallNanos",
        (getCurrentTimeMicro() - ioTimeStartMicros) * 1'000);
  }
}

void VectorLoader::load(RowSet rows, ValueHook* hook, VectorPtr* result) {
  const auto ioTimeStartMicros = getCurrentTimeMicro();
  loadInternal(rows, hook, result);
  writeIOWallTimeStat(ioTimeStartMicros);
}

void VectorLoader::load(
    const SelectivityVector& rows,
    ValueHook* hook,
    VectorPtr* result) {
  const auto ioTimeStartMicros = getCurrentTimeMicro();
  loadInternal(rows, hook, result);
  writeIOWallTimeStat(ioTimeStartMicros);
}

void VectorLoader::loadInternal(
    const SelectivityVector& rows,
    ValueHook* hook,
    VectorPtr* result) {
  if (rows.isAllSelected()) {
    const auto& indices = DecodedVector::consecutiveIndices();
    assert(!indices.empty());
    if (rows.end() <= indices.size()) {
      load(
          RowSet(&indices[rows.begin()], rows.end() - rows.begin()),
          hook,
          result);
      return;
    }
  }
  std::vector<vector_size_t> positions(rows.countSelected());
  int index = 0;
  rows.applyToSelected([&](vector_size_t row) { positions[index++] = row; });
  load(positions, hook, result);
}

} // namespace facebook::velox
