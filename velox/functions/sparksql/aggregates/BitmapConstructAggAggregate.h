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
#include <string>

#include "velox/exec/AggregateUtil.h"

namespace facebook::velox::functions::aggregate::sparksql {

// Fixed bitmap size matching Spark's BitmapExpressionUtils.NUM_BYTES (4096
// bytes = 32768 bits). Used by both bitmap_construct_agg and bitmap_or_agg.
// See org.apache.spark.sql.catalyst.expressions.BitmapConstructAgg and
// org.apache.spark.sql.catalyst.expressions.BitmapOrAgg.
constexpr int32_t kBitmapNumBytes = 4096;
constexpr int64_t kBitmapNumBits = static_cast<int64_t>(kBitmapNumBytes) * 8;

exec::AggregateRegistrationResult registerBitmapConstructAggAggregate(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite);

} // namespace facebook::velox::functions::aggregate::sparksql
