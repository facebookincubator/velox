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

#include "velox/common/vectorindex/VectorIndexKind.h"

#include <folly/container/F14Map.h>

#include "velox/common/EnumDefine.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::common::vectorindex {

namespace {

const auto& vectorIndexKindNames() {
  static const folly::F14FastMap<VectorIndexKind, std::string_view> kNames = {
      {VectorIndexKind::kFlat, "flat"},
      {VectorIndexKind::kIvfFlat, "ivf_flat"},
      {VectorIndexKind::kIvfSq, "ivf_sq"},
      {VectorIndexKind::kHnswSq, "hnsw_sq"},
  };
  return kNames;
}

} // namespace

VELOX_DEFINE_ENUM_NAME(VectorIndexKind, vectorIndexKindNames);

VectorIndexKind VectorIndexKindValue::fromValue(int32_t value) {
  const auto kind = static_cast<VectorIndexKind>(value);
  switch (kind) {
    case VectorIndexKind::kFlat:
    case VectorIndexKind::kIvfFlat:
    case VectorIndexKind::kIvfSq:
    case VectorIndexKind::kHnswSq:
      return kind;
  }
  VELOX_USER_FAIL("Invalid vector index kind value: {}", value);
}

} // namespace facebook::velox::common::vectorindex
