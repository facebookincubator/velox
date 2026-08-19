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

#include <folly/container/F14Map.h>

#include "velox/dwio/parquet/common/PageIndex.h"

namespace facebook::velox::parquet {

/// Describes the optional page-index regions requested for one column chunk.
struct ColumnPageIndexInformation {
  int64_t offsetIndexOffset{0};
  int32_t offsetIndexLength{0};
  int64_t columnIndexOffset{0};
  int32_t columnIndexLength{0};
  bool readColumnIndex{false};
  bool readOffsetIndex{true};
  PageBoundsCapability boundsCapability{PageBoundsCapability::kNone};
};

using PageIndexInfoMap =
    folly::F14FastMap<uint32_t, ColumnPageIndexInformation>;

} // namespace facebook::velox::parquet
