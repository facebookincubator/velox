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

#include <string_view>

namespace facebook::velox::exec {

/// Names of the runtime stats reported by HashProbe.
struct HashProbeStatNames {
  /// Size of the bloom filter in bytes.
  static constexpr std::string_view kBloomFilterSize = "bloomFilterSize";

  /// Number of probe rows tested by the build-side Bloom filter locally.
  static constexpr std::string_view kBloomFilterTestedRows =
      "bloomFilterTestedRows";

  /// Number of local Bloom-filter tests that required a hash table lookup.
  static constexpr std::string_view kBloomFilterAcceptedRows =
      "bloomFilterAcceptedRows";

  /// Number of times local Bloom filter probing was bypassed after sampling.
  static constexpr std::string_view kBloomFilterBypassed =
      "bloomFilterBypassed";

  /// Number of rows bypassed via dynamic filter replacement.
  static constexpr std::string_view kReplacedWithDynamicFilterRows =
      "replacedWithDynamicFilterRows";
};

} // namespace facebook::velox::exec
