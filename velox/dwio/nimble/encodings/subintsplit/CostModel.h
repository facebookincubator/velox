/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <cstddef>

#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionMetrics.h"

// Per-section cost models for the SubIntSplit DP selector.
//
// Each function estimates the compressed size in bits for storing `numValues`
// items of a bit-range sub-stream of logical width `bitWidth`. The models
// correspond to nimble's present encodings and are derived from the same
// assumptions used in EncodingSizeEstimation.h (common prefix 6 bytes, nested
// encoding overhead, etc.).
//
// Deliberately avoids HLL cardinality estimation, entropy, and frame-residual
// tracking: the simplified SectionMetrics provides enough signal for the DP to
// make directionally correct split decisions.

namespace facebook::nimble::subintsplit {

/// Union of the metrics every cost model below reads.
MetricFlags allCostModelRequiredFlags() noexcept;

/// Stores each value at its native storage width.
double trivialCostBits(
    const SectionMetrics& metrics,
    size_t numValues,
    int bitWidth) noexcept;

/// Bit-packs using the observed range. SubIntSplit encodes nested sections with
/// fixedBitWidthUseExactBits, so there is no byte-boundary rounding: a 12-bit
/// section costs 12 bits/value, not 16.
double fixedBitWidthCostBits(
    const SectionMetrics& metrics,
    size_t numValues,
    int bitWidth) noexcept;

/// Infinite unless every value is equal.
double constantCostBits(
    const SectionMetrics& metrics,
    size_t numValues,
    int bitWidth) noexcept;

/// Stores one dominant value, a SparseBool mask marking the exception rows, and
/// the exception values as a FixedBitWidth child.
double mainlyConstantCostBits(
    const SectionMetrics& metrics,
    size_t numValues,
    int bitWidth) noexcept;

/// Unique value table plus bit-packed indices, using the observed unique count
/// directly (no HLL blending).
double dictionaryCostBits(
    const SectionMetrics& metrics,
    size_t numValues,
    int bitWidth) noexcept;

/// Run values plus bit-packed run lengths.
double rleCostBits(
    const SectionMetrics& metrics,
    size_t numValues,
    int bitWidth) noexcept;

/// Variable-length integer storage. Only worth considering for >=32-bit
/// sections.
double varintCostBits(
    const SectionMetrics& metrics,
    size_t numValues,
    int bitWidth) noexcept;

/// Evaluates every cost model and returns the minimum cost in bits, writing the
/// winning encoding into `bestEncoding`.
double bestCostBits(
    const SectionMetrics& metrics,
    size_t numValues,
    int bitWidth,
    EncodingType& bestEncoding) noexcept;

} // namespace facebook::nimble::subintsplit
