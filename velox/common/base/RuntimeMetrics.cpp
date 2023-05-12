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

#include <folly/ThreadLocal.h>

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/base/SuccinctPrinter.h"

namespace facebook::velox {

void RuntimeMetric::addValue(int64_t value) {
  sum += value;
  count++;
  min = std::min(min, value);
  max = std::max(max, value);
}

void RuntimeMetric::aggregate() {
  count = std::min(count, static_cast<int64_t>(1));
  min = max = sum;
}

void RuntimeMetric::merge(const RuntimeMetric& other) {
  VELOX_CHECK_EQ(unit, other.unit);
  sum += other.sum;
  count += other.count;
  min = std::min(min, other.min);
  max = std::max(max, other.max);
}

void RuntimeMetric::printMetric(std::stringstream& stream) const {
  switch (unit) {
    case RuntimeCounter::Unit::kNanos:
      stream << " sum: " << succinctNanos(sum) << ", count: " << count
             << ", min: " << succinctNanos(min)
             << ", max: " << succinctNanos(max);
      break;
    case RuntimeCounter::Unit::kBytes:
      stream << " sum: " << succinctBytes(sum) << ", count: " << count
             << ", min: " << succinctBytes(min)
             << ", max: " << succinctBytes(max);
      break;
    case RuntimeCounter::Unit::kNone:
    default:
      stream << " sum: " << sum << ", count: " << count << ", min: " << min
             << ", max: " << max;
  }
}

// Thread local runtime stat writers.
static thread_local BaseRuntimeStatWriter* localRuntimeStatWriter;

void setThreadLocalRunTimeStatWriter(BaseRuntimeStatWriter* writer) {
  localRuntimeStatWriter = writer;
}

BaseRuntimeStatWriter* getThreadLocalRunTimeStatWriter() {
  return localRuntimeStatWriter;
}

void addThreadLocalRuntimeStat(
    const std::string& name,
    const RuntimeCounter& value) {
  if (localRuntimeStatWriter) {
    localRuntimeStatWriter->addRuntimeStat(name, value);
  }
}

} // namespace facebook::velox
