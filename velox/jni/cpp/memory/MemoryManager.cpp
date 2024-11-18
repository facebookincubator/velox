/*
* Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "MemoryManager.h"

#include <core/QueryCtx.h>

#include "velox/jni/cpp/utils/JsonUtils.h"

namespace facebook::velox::sdk::memory {

std::shared_ptr<MemoryManager> MemoryManager::get() {
  static std::shared_ptr<MemoryManager> memoryManager =
      std::make_shared<MemoryManager>();
  return memoryManager;
}

std::shared_ptr<velox::memory::MemoryPool>
MemoryManager::rootVeloxMemoryPool() {
  static auto veloxAlloc =
      facebook::velox::memory::deprecatedDefaultMemoryManager().addRootPool(
          "root", FLAGS_max_root_memory_bytes);
  return veloxAlloc;
}

std::shared_ptr<velox::memory::MemoryPool>
MemoryManager::defaultLeafVeloxMemoryPool() {
  static std::shared_ptr<velox::memory::MemoryPool> defaultPool =
      rootVeloxMemoryPool()->addLeafChild("default_leaf");
  return defaultPool;
}

std::shared_ptr<velox::memory::MemoryPool> MemoryManager::createQueryPool(
    const std::string& queryId, int64_t bytes) {
  return facebook::velox::memory::deprecatedDefaultMemoryManager().addRootPool(
      velox::core::QueryCtx::generatePoolName(queryId),
      bytes);
}

std::shared_ptr<velox::memory::MemoryPool> MemoryManager::planMemoryPool() {
  static std::shared_ptr<velox::memory::MemoryPool> defaultPool =
      rootVeloxMemoryPool()->addLeafChild("plan");
  return defaultPool;
}




folly::dynamic MemoryManager::toJsonString(
    std::shared_ptr<velox::memory::MemoryPool> memoryPool) {
  auto stats = memoryPool->stats();
  folly::dynamic obj = folly::dynamic::object();
  obj["capacity"] = memoryPool->capacity();
  obj["reservedBytes"] = stats.reservedBytes;
  obj["peakBytes"] = stats.peakBytes;
  obj["cumulativeBytes"] = stats.cumulativeBytes;
  obj["numAllocs"] = stats.numAllocs;
  obj["numFrees"] = stats.numFrees;
  obj["numReserves"] = stats.numReserves;
  obj["numReleases"] = stats.numReleases;
  obj["numShrinks"] = stats.numShrinks;
  obj["numReclaims"] = stats.numReclaims;
  obj["numCollisions"] = stats.numCollisions;
  obj["numCapacityGrowths"] = stats.numCapacityGrowths;
  return obj;
}

std::string MemoryManager::memoryStatics() {
  folly::dynamic obj = folly::dynamic::object();
  obj["root"] = toJsonString(rootVeloxMemoryPool());
  obj["plan"] = toJsonString(planMemoryPool());
  for (auto pool :
       velox::memory::deprecatedDefaultMemoryManager().getAlivePools()) {
    obj[pool->name()] = toJsonString(pool);
  }
  return utils::JsonUtils::toSortedJson(obj);
}

} // namespace facebook::velox::sdk::memory