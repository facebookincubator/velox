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

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/SubIntSplitCostModels.h"
#include "velox/dwio/nimble/encodings/SubIntSplitMetrics.h"
#include "velox/dwio/nimble/encodings/SubIntSplitSelector.h"

namespace facebook::nimble::mlidc {

enum class AccessClass : uint8_t {
  PureRA = 0,
  HybridRA = 1,
  BoundedSeq = 2,
  PureSeq = 3,
};

inline const char* accessClassName(AccessClass c) {
  switch (c) {
    case AccessClass::PureRA: return "PureRA";
    case AccessClass::HybridRA: return "HybridRA";
    case AccessClass::BoundedSeq: return "BoundedSeq";
    case AccessClass::PureSeq: return "PureSeq";
  }
  return "Unknown";
}

struct EncodingInfo {
  EncodingType type;
  std::string name;
  AccessClass accessClass;
  bool hasCostModel;
};

inline const std::vector<EncodingInfo>& encodingInventory() {
  static const std::vector<EncodingInfo> inv = {
      {EncodingType::Trivial, "Trivial", AccessClass::PureRA, true},
      {EncodingType::FixedBitWidth, "FixedBitWidth", AccessClass::PureRA, true},
      {EncodingType::Constant, "Constant", AccessClass::PureRA, true},
      {EncodingType::Dictionary, "Dictionary", AccessClass::PureRA, true},
      {EncodingType::MainlyConstant, "MainlyConstant", AccessClass::PureSeq,
       true},
      {EncodingType::RLE, "RLE", AccessClass::PureSeq, true},
      {EncodingType::Varint, "Varint", AccessClass::PureSeq, true},
  };
  return inv;
}

inline AccessClass accessClassOf(EncodingType t) {
  for (const auto& e : encodingInventory()) {
    if (e.type == t) return e.accessClass;
  }
  return AccessClass::PureSeq;
}

struct AblationRung {
  std::string name;
  std::unordered_set<EncodingType> allowed;
  AccessClass worstAllowed;
  bool costModelConsistent;
};

inline std::vector<AblationRung> combinedLadder() {
  std::vector<AblationRung> rungs;

  rungs.push_back(
      {"trivial_only",
       {EncodingType::Trivial},
       AccessClass::PureRA,
       true});

  rungs.push_back(
      {"pure_ra",
       {EncodingType::Trivial, EncodingType::FixedBitWidth,
        EncodingType::Constant, EncodingType::Dictionary},
       AccessClass::PureRA,
       true});

  rungs.push_back(
      {"full_set",
       {EncodingType::Trivial, EncodingType::FixedBitWidth,
        EncodingType::Constant, EncodingType::Dictionary,
        EncodingType::MainlyConstant, EncodingType::RLE, EncodingType::Varint},
       AccessClass::PureSeq,
       true});

  return rungs;
}

namespace detail_ablation {
using namespace facebook::nimble::detail::subintsplit;

inline double bestCostBitsRestricted(
    const SegmentMetrics& m,
    size_t numValues,
    int bitWidth,
    EncodingType& bestEncoding,
    const std::unordered_set<EncodingType>& allowed) noexcept {
  double best = std::numeric_limits<double>::infinity();
  auto consider = [&](double cost, EncodingType type) noexcept {
    if (allowed.count(type) && cost < best) {
      best = cost;
      bestEncoding = type;
    }
  };

  consider(trivialCostBits(m, numValues, bitWidth), EncodingType::Trivial);
  consider(
      fixedBitWidthCostBits(m, numValues, bitWidth),
      EncodingType::FixedBitWidth);
  consider(constantCostBits(m, numValues, bitWidth), EncodingType::Constant);
  consider(
      mainlyConstantCostBits(m, numValues, bitWidth),
      EncodingType::MainlyConstant);
  consider(rleCostBits(m, numValues, bitWidth), EncodingType::RLE);
  consider(varintCostBits(m, numValues, bitWidth), EncodingType::Varint);
  if (allowed.count(EncodingType::Dictionary) && m.uniqueCount > 0 &&
      (m.uniqueCountCapped || m.uniqueCount < numValues / 2)) {
    consider(
        dictionaryCostBits(m, numValues, bitWidth), EncodingType::Dictionary);
  }
  return best;
}

inline SelectorResult selectSplitsRestricted(
    const std::vector<uint64_t>& samples,
    int kBits,
    size_t fullCount,
    const SelectorConfig& cfg,
    const std::unordered_set<EncodingType>& allowed) {
  return selectSplitsImpl(
      samples, kBits, fullCount, cfg,
      [&allowed](
          const SegmentMetrics& m,
          size_t numValues,
          int bitWidth,
          EncodingType& bestEnc) noexcept {
        return bestCostBitsRestricted(m, numValues, bitWidth, bestEnc, allowed);
      });
}

} // namespace detail_ablation

} // namespace facebook::nimble::mlidc

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
