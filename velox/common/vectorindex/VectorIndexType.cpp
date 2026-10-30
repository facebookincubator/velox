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

#include "velox/common/vectorindex/VectorIndexType.h"

#include <folly/container/F14Map.h>

#include "velox/common/EnumDefine.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common::vectorindex {

namespace {

const auto& vectorIndexTypeNames() {
  static const folly::F14FastMap<VectorIndexType, std::string_view> kNames = {
      {VectorIndexType::kFlat, "flat"},
      {VectorIndexType::kIvfFlat, "ivf_flat"},
      {VectorIndexType::kIvfSq, "ivf_sq"},
      {VectorIndexType::kHnswSq, "hnsw_sq"},
      {VectorIndexType::kCagra, "cagra"},
  };
  return kNames;
}

} // namespace

VELOX_DEFINE_ENUM_NAME(VectorIndexType, vectorIndexTypeNames);

VectorIndexType VectorIndexTypeValue::fromValue(int32_t value) {
  const auto type = static_cast<VectorIndexType>(value);
  switch (type) {
    case VectorIndexType::kFlat:
    case VectorIndexType::kIvfFlat:
    case VectorIndexType::kIvfSq:
    case VectorIndexType::kHnswSq:
    case VectorIndexType::kCagra:
      return type;
  }
  VELOX_USER_FAIL("Invalid vector index type value: {}", value);
}

} // namespace facebook::velox::common::vectorindex
