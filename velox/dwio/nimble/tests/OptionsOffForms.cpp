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
#include "velox/dwio/nimble/tests/OptionsOffForms.h"

#include <cstdint>
#include <limits>

namespace facebook::nimble::test::options_off {

std::vector<OffForm> offForms() {
  return {
      {"untouched", [](WriterOptions& /*options*/) {}},
      {"explicitlyOff",
       [](WriterOptions& options) {
         options.fixedBitWidthTypedLanes = false;
         options.fixedBitWidthTypedLanesMinValues = 256;
       }},
      {"floorOneWithoutLanes",
       [](WriterOptions& options) {
         options.fixedBitWidthTypedLanes = false;
         options.fixedBitWidthTypedLanesMinValues = 1;
       }},
      {"floorMaxWithoutLanes",
       [](WriterOptions& options) {
         options.fixedBitWidthTypedLanes = false;
         options.fixedBitWidthTypedLanesMinValues =
             std::numeric_limits<uint64_t>::max();
       }},
  };
}

} // namespace facebook::nimble::test::options_off
