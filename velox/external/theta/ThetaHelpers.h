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

// Adapted from Apache DataSketches

#pragma once

#include <stdexcept>
#include <string>

#include "ThetaConstants.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common::theta {

template <typename T>
static void checkValue(T actual, T expected, const char* description) {
  if (actual != expected) {
    auto msg = std::string(description) + " mismatch: expected " +
        std::to_string(expected) + ", actual " + std::to_string(actual);
    throw VeloxUserError(
        __FILE__,
        __LINE__,
        __FUNCTION__,
        "",
        msg,
        error_source::kErrorSourceUser,
        error_code::kInvalidArgument,
        false /*retriable*/);
  }
}

template <bool dummy>
class checker {
 public:
  static void checkSerialVersion(uint8_t actual, uint8_t expected) {
    checkValue(actual, expected, "serial version");
  }
  static void checkSketchFamily(uint8_t actual, uint8_t expected) {
    checkValue(actual, expected, "sketch family");
  }
  static void checkSketchType(uint8_t actual, uint8_t expected) {
    checkValue(actual, expected, "sketch type");
  }
  static void checkSeedHash(uint16_t actual, uint16_t expected) {
    checkValue(actual, expected, "seed hash");
  }
};

template <bool dummy>
class ThetaBuildHelper {
 public:
  // consistent way of initializing theta from p
  // avoids multiplication if p == 1 since it might not yield MAX_THETA exactly
  static uint64_t startingThetaFromP(float p) {
    if (p < 1)
      return static_cast<uint64_t>(
          static_cast<double>(ThetaConstants::MAX_THETA) * p);
    return ThetaConstants::MAX_THETA;
  }

  static uint8_t
  startingSubMultiple(uint8_t lg_tgt, uint8_t lg_min, uint8_t lg_rf) {
    return (lg_tgt <= lg_min) ? lg_min
        : (lg_rf == 0)        ? lg_tgt
                              : ((lg_tgt - lg_min) % lg_rf) + lg_min;
  }
};

} // namespace facebook::velox::common::theta
