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

#include "velox/common/base/ConcurrentRuntimeStatWriter.h"

#include "velox/common/base/Exceptions.h"

namespace facebook::velox {

void ConcurrentRuntimeStatWriter::addRuntimeStat(
    std::string_view name,
    const RuntimeCounter& value) {
  auto lockedStats = runtimeStats_.wlock();
  auto [it, unused] = lockedStats->try_emplace(std::string(name), value.unit);
  it->second.merge(value);
}

void ConcurrentRuntimeStatWriter::setRuntimeStat(
    std::string_view name,
    const RuntimeMetric& metric) {
  runtimeStats_.wlock()->insert_or_assign(std::string(name), metric);
}

std::unordered_map<std::string, RuntimeMetric>
ConcurrentRuntimeStatWriter::runtimeStats() const {
  return *runtimeStats_.rlock();
}

void ConcurrentRuntimeStatWriter::clear() {
  runtimeStats_.wlock()->clear();
}

} // namespace facebook::velox
