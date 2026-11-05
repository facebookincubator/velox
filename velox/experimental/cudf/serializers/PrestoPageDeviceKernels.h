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

// The device side of PrestoPageDevice.h for the functions whose arguments the
// host checks first: PrestoPageDevice.cpp validates and forwards here. Not for
// other callers.

#include "velox/experimental/cudf/serializers/PrestoPageDevice.h"

namespace facebook::velox::cudf_velox::detail::device {

/// prestoPageBoundaries() without the argument checks.
std::vector<int64_t> prestoPageBoundaries(
    int32_t numRows,
    int64_t fixedBitsPerRow,
    cudf::host_span<const StringOffsets> strings,
    int64_t pageBytes,
    int64_t maxRows,
    cuda::stream_ref stream);

/// fillPrestoStringOffsets() without the offsets type check.
bool fillPrestoStringOffsets(
    cudf::host_span<const StringOffsetsTask> tasks,
    cuda::stream_ref stream);

/// gatherStringOffsets() without the offsets type check.
std::vector<int64_t> gatherStringOffsets(
    const StringOffsets& offsets,
    cudf::host_span<const int64_t> rows,
    cuda::stream_ref stream);

/// timestampsToMillis() with the scale resolved: dst[i] = floor(src[i] * mul /
/// div), where one of 'mul' and 'div' is 1 and both are positive.
bool timestampsToMillis(
    const int64_t* src,
    int32_t numRows,
    int64_t mul,
    int64_t div,
    int64_t* dst,
    cuda::stream_ref stream);

/// decimalsToJava() without the alignment checks.
void decimalsToJava(
    const __int128_t* src,
    int32_t numRows,
    __int128_t* dst,
    cuda::stream_ref stream);

} // namespace facebook::velox::cudf_velox::detail::device
