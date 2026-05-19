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

#include "velox/common/memory/CustomMemoryResource.h"

#include <utility>

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/MemoryArbitrator.h"

namespace facebook::velox::memory {

CustomMemoryResource::CustomMemoryResource(
    std::string tag,
    std::shared_ptr<MemoryAllocator> allocator,
    std::shared_ptr<MemoryArbitrator> arbitrator,
    ReclaimerFactory reclaimerFactory,
    int64_t maxCapacity)
    : tag_(std::move(tag)),
      maxCapacity_(maxCapacity),
      allocator_(std::move(allocator)),
      arbitrator_(std::move(arbitrator)),
      reclaimerFactory_(std::move(reclaimerFactory)) {
  VELOX_USER_CHECK(!tag_.empty(), "CustomMemoryResource tag is empty");
  VELOX_USER_CHECK_NOT_NULL(
      allocator_,
      "CustomMemoryResource allocator is null for tag: {}",
      tag_);
  VELOX_USER_CHECK_NOT_NULL(
      arbitrator_,
      "CustomMemoryResource arbitrator is null for tag: {}",
      tag_);
  VELOX_USER_CHECK(
      reclaimerFactory_ != nullptr,
      "CustomMemoryResource reclaimerFactory is null for tag: {}",
      tag_);
}

std::unique_ptr<MemoryReclaimer> CustomMemoryResource::newReclaimer() const {
  return reclaimerFactory_();
}

} // namespace facebook::velox::memory
