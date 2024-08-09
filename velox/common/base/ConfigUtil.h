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

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::config {
enum class CapacityUnit {
  BYTE,
  KILOBYTE,
  MEGABYTE,
  GIGABYTE,
  TERABYTE,
  PETABYTE
};

double toBytesPerCapacityUnit(CapacityUnit unit);

CapacityUnit valueOfCapacityUnit(const std::string& unitStr);

/// Convert capacity string with unit to the capacity number in the specified
/// units
uint64_t toCapacity(const std::string& from, CapacityUnit to);

std::chrono::duration<double> toDuration(const std::string& str);
} // namespace facebook::velox::config
