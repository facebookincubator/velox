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
#include "velox/dwio/nimble/encodings/subintsplit/CostModel.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionMetrics.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitSelector.h"

namespace facebook::nimble::mlidc {

enum class AccessClass : uint8_t {
  PureRA = 0,
  HybridRA = 1,
  BoundedSeq = 2,
  PureSeq = 3,
};

inline const char* accessClassName(AccessClass c) {
  switch (c) {
    case AccessClass::PureRA:
      return "PureRA";
    case AccessClass::HybridRA:
      return "HybridRA";
    case AccessClass::BoundedSeq:
      return "BoundedSeq";
    case AccessClass::PureSeq:
      return "PureSeq";
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
      {EncodingType::MainlyConstant,
       "MainlyConstant",
       AccessClass::PureSeq,
       true},
      {EncodingType::RLE, "RLE", AccessClass::PureSeq, true},
      {EncodingType::Varint, "Varint", AccessClass::PureSeq, true},
  };
  return inv;
}

inline AccessClass accessClassOf(EncodingType t) {
  for (const auto& e : encodingInventory()) {
    if (e.type == t)
      return e.accessClass;
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
      {"trivial_only", {EncodingType::Trivial}, AccessClass::PureRA, true});

  rungs.push_back(
      {"pure_ra",
       {EncodingType::Trivial,
        EncodingType::FixedBitWidth,
        EncodingType::Constant,
        EncodingType::Dictionary},
       AccessClass::PureRA,
       true});

  rungs.push_back(
      {"full_set",
       {EncodingType::Trivial,
        EncodingType::FixedBitWidth,
        EncodingType::Constant,
        EncodingType::Dictionary,
        EncodingType::MainlyConstant,
        EncodingType::RLE,
        EncodingType::Varint},
       AccessClass::PureSeq,
       true});

  return rungs;
}

namespace detail_ablation {
using namespace facebook::nimble::subintsplit;

// The cost-model dispatch lives in subintsplit/CostModel.h and is shared with
// the selector. An earlier version of this file kept its own copy, which went
// stale twice over: it missed every encoding added since, and it did not follow
// the cost function's signature when segment values were threaded through.
inline SelectorResult selectSplitsRestricted(
    const std::vector<uint64_t>& samples,
    int kBits,
    size_t fullCount,
    const SelectorConfig& cfg,
    const std::unordered_set<EncodingType>& allowed) {
  return facebook::nimble::subintsplit::selectSplitsRestricted(
      samples, kBits, fullCount, allowed, cfg);
}

} // namespace detail_ablation

} // namespace facebook::nimble::mlidc

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
