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
#include <functional>

namespace facebook::velox::core {

/// Per-batch scan statistics event fired by TableScan after each batch.
struct ScanBatchEvent {
  virtual ~ScanBatchEvent() = default;

  /// Post-pushdown, pre-remaining-filter row count.
  uint64_t numRows{0};
  /// Wall time spent producing this batch in microseconds.
  uint64_t wallTimeMicros{0};
};

using ScanBatchCallback = std::function<void(const ScanBatchEvent&)>;

} // namespace facebook::velox::core
