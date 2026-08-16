/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "velox/dwio/nimble/ML_ID_Compression/SelectiveTraceGen.h"

namespace facebook::nimble::mlidc {

enum class GapModel { UniformDeterministic, Geometric };

inline std::string gapModelName(GapModel m) {
    return m == GapModel::Geometric ? "geometric" : "uniform";
}

struct GatherAccessParams {
    size_t   start{0};
    size_t   span{0};
    double   selectivity{1.0};
    size_t   runLength{8};
    GapModel gapModel{GapModel::UniformDeterministic};
    uint64_t seed{42};
    size_t   maxRanges{0};
};

struct GatherTrace {
    RowRangeList ranges;
    size_t selectedRows{0};
    size_t rangeCountNominal{0};
    size_t rangeCount{0};
    size_t runLengthActual{0};
    size_t gapLength{0};
    double selectivityAchieved{0.0};
    bool   clamped{false};

    void expandToRows(std::vector<int32_t>& out) const {
        out.clear();
        out.reserve(selectedRows);
        for (const auto& r : ranges)
            for (size_t i = r.begin; i < r.end; ++i)
                out.push_back(static_cast<int32_t>(i));
    }
};

inline size_t impliedRangeCount(size_t span, double selectivity, size_t runLength) {
    if (span == 0) return 0;
    const size_t rl = std::max<size_t>(1, runLength);
    const size_t selected = static_cast<size_t>(
        std::floor(std::clamp(selectivity, 0.0, 1.0) * static_cast<double>(span)));
    return std::max<size_t>(1, selected / rl);
}

namespace detail {

inline void mergeAdjacent(RowRangeList& ranges) {
    if (ranges.size() < 2) return;
    RowRangeList merged;
    merged.reserve(ranges.size());
    merged.push_back(ranges.front());
    for (size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].begin <= merged.back().end)
            merged.back().end = std::max(merged.back().end, ranges[i].end);
        else
            merged.push_back(ranges[i]);
    }
    ranges.swap(merged);
}

inline void finalize(GatherTrace& t, size_t span) {
    mergeAdjacent(t.ranges);
    t.rangeCount = t.ranges.size();
    t.selectedRows = 0;
    for (const auto& r : t.ranges) t.selectedRows += r.size();
    t.selectivityAchieved = span > 0
        ? static_cast<double>(t.selectedRows) / static_cast<double>(span)
        : 0.0;
}

}  // namespace detail

inline GatherTrace buildGatherTrace(size_t streamLength, const GatherAccessParams& p) {
    GatherTrace t;
    if (streamLength == 0 || p.start >= streamLength) return t;
    const size_t start = p.start;
    const size_t span = std::min(p.span, streamLength - start);
    if (span == 0) return t;
    const double sigma = std::clamp(p.selectivity, 0.0, 1.0);
    if (sigma <= 0.0) return t;
    const size_t runLength = std::max<size_t>(1, p.runLength);

    if (p.gapModel == GapModel::Geometric) {
        SelectiveTraceParams sp;
        sp.selectivity = sigma;
        sp.meanRunLength = static_cast<double>(runLength);
        sp.seed = p.seed;
        sp.maxRanges = p.maxRanges;
        t.ranges = makeSelectiveTrace(span, sp);
        for (auto& r : t.ranges) { r.begin += start; r.end += start; }
        t.rangeCountNominal = impliedRangeCount(span, sigma, runLength);
        t.runLengthActual = runLength;
        t.gapLength = static_cast<size_t>(
            std::llround(static_cast<double>(runLength) * (1.0 / std::max(sigma, 1e-9) - 1.0)));
        detail::finalize(t, span);
        t.clamped = p.maxRanges != 0 && t.rangeCount >= p.maxRanges;
        return t;
    }

    // Uniform deterministic model
    const size_t selectedTarget = static_cast<size_t>(
        std::floor(sigma * static_cast<double>(span)));
    if (selectedTarget == 0) return t;
    size_t k = std::max<size_t>(1, selectedTarget / runLength);
    t.rangeCountNominal = k;
    if (p.maxRanges != 0 && k > p.maxRanges) { k = p.maxRanges; t.clamped = true; }
    const size_t gapTotal = span - selectedTarget;
    auto selBefore = [&](size_t i) { return selectedTarget * i / k; };
    auto gapBefore = [&](size_t i) { return k > 1 ? gapTotal * i / (k - 1) : size_t{0}; };
    t.ranges.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        const size_t begin = start + selBefore(i) + gapBefore(i);
        const size_t end = start + selBefore(i + 1) + gapBefore(i);
        if (end > begin) t.ranges.push_back({begin, end});
    }
    detail::finalize(t, span);
    t.runLengthActual = t.rangeCount > 0 ? t.selectedRows / t.rangeCount : 0;
    t.gapLength = k > 1 ? gapTotal / (k - 1) : 0;
    if (t.rangeCount == 1 && t.selectedRows == selectedTarget) t.clamped = false;
    return t;
}

}  // namespace facebook::nimble::mlidc
