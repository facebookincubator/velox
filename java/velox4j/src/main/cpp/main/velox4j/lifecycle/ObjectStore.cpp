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
#include "velox4j/lifecycle/ObjectStore.h"

#include <glog/logging.h>

namespace facebook::velox4j {

ObjectStore* ObjectStore::global() {
  static std::unique_ptr<ObjectStore> globalStore = create();
  return globalStore.get();
}

std::unique_ptr<ObjectStore> ObjectStore::create() {
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  const StoreHandle nextId = safeCast<StoreHandle>(stores().nextId());
  auto store = std::unique_ptr<ObjectStore>(new ObjectStore(nextId));
  const StoreHandle storeId =
      safeCast<StoreHandle>(stores().insert(store.get()));
  VELOX_CHECK(storeId == nextId, "Store ID mismatched, this should not happen");
  return store;
}

void ObjectStore::release(ObjectHandle handle) {
  const ResourceHandle storeId =
      safeCast<ResourceHandle>(handle >> (sizeof(ResourceHandle) * 8));
  const ResourceHandle resourceId = safeCast<ResourceHandle>(
      handle & std::numeric_limits<ResourceHandle>::max());
  auto store = stores().lookup(storeId);
  store->releaseInternal(resourceId);
}

ResourceMap<ObjectStore*>& ObjectStore::stores() {
  static ResourceMap<ObjectStore*> stores;
  return stores;
}

ObjectHandle ObjectStore::toObjHandle(ResourceHandle rh) const {
  const ObjectHandle prefix = static_cast<ObjectHandle>(storeId_)
      << (sizeof(ResourceHandle) * 8);
  const ObjectHandle objHandle = prefix + rh;
  return objHandle;
}

ObjectStore::~ObjectStore() {
  // destructing in reversed order (the last added object destructed first)
  const std::lock_guard<std::mutex> lock(mtx_);
  for (auto itr = aliveObjects_.rbegin(); itr != aliveObjects_.rend(); ++itr) {
    const std::string_view description = (*itr).second;
    const ResourceHandle handle = (*itr).first;
    LOG(WARNING)
        << "Unclosed object [" << "Store ID: " << storeId_
        << ", Resource handle ID: " << handle
        << ", Description: " << description
        << "] is found when object store is closing. Velox4J will"
           " destroy it automatically but it's recommended to manually close"
           " the object through the Java API CppObject#close() after use,"
           " to minimize peak memory pressure of the application.";
    store_.erase(handle);
  }
  stores().erase(storeId_);
}

void ObjectStore::releaseInternal(ResourceHandle handle) {
  const std::lock_guard<std::mutex> lock(mtx_);
  store_.erase(handle);
  aliveObjects_.erase(handle);
}
} // namespace facebook::velox4j
