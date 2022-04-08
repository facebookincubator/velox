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

#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/base/SuccinctPrinter.h"

namespace facebook::velox {

void RuntimeMetric::addValue(int64_t value, RuntimeMetricKind valueKind) {
  sum += value;
  count++;
  min = std::min(min, value);
  max = std::max(max, value);
  kind = valueKind;
}

void RuntimeMetric::merge(const RuntimeMetric& other) {
  sum += other.sum;
  count += other.count;
  min = std::min(min, other.min);
  max = std::max(max, other.max);
  kind = other.kind;
}

void RuntimeMetric::printMetric(std::stringstream& stream) const {
  switch (kind) {
    case kNanos:
      stream << " sum: " << succinctNanos(sum) << ", count: " << count
             << ", min: " << succinctNanos(min)
             << ", max: " << succinctNanos(max);
      break;
    case kByte:
      stream << " sum: " << succinctBytes(sum) << ", count: " << count
             << ", min: " << succinctBytes(min)
             << ", max: " << succinctBytes(max);
      break;
    case kNone:
    default:
      stream << " sum: " << sum << ", count: " << count << ", min: " << min
             << ", max: " << max;
  }
}
} // namespace facebook::velox
