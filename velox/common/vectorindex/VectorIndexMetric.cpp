/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include "velox/common/vectorindex/VectorIndexMetric.h"

#include <folly/container/F14Map.h>

#include "velox/common/EnumDefine.h"

namespace facebook::velox::common::vectorindex {

namespace {

const auto& vectorIndexMetricNames() {
  static const folly::F14FastMap<VectorIndexMetric, std::string_view> kNames = {
      {VectorIndexMetric::kCosine, "cosine"},
  };
  return kNames;
}

} // namespace

VELOX_DEFINE_ENUM_NAME(VectorIndexMetric, vectorIndexMetricNames);

} // namespace facebook::velox::common::vectorindex
