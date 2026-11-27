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

#include "velox/common/caching/EvictionPolicy.h"

#include "velox/common/EnumDefine.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/caching/ApproxLrfuEvictionPolicy.h"

namespace facebook::velox::cache {

namespace {

const auto& evictionPolicyKindNames() {
  static const folly::F14FastMap<EvictionPolicyKind, std::string_view> kNames =
      {
          {EvictionPolicyKind::kApproxLrfu, "sampled-lrfu"},
      };
  return kNames;
}

} // namespace

VELOX_DEFINE_ENUM_NAME(EvictionPolicyKind, evictionPolicyKindNames);

std::unique_ptr<EvictionPolicy> EvictionPolicy::create(
    EvictionPolicyKind kind) {
  switch (kind) {
    case EvictionPolicyKind::kApproxLrfu:
      return std::make_unique<ApproxLrfuEvictionPolicy>();
  }
  VELOX_FAIL("Unknown eviction policy kind: {}", static_cast<int>(kind));
}

} // namespace facebook::velox::cache
