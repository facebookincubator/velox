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

#include <map>
#include "ResourceMap.h"

namespace velox4j {

// ObjectHandle is a signed int64 consisting of:
// 1. 1 - 32 bits is a signed int32 as the object store's ID;
// 2. 1st bit is always zero to be compatible with jlong;
// 3. 33 - 64 bits is an unsigned int32 as the object's ID;
//
// When the object is tended to be retrieved with its ObjectHandle,
// the program first finds its resident object store, then looks up
// for the object in the store.
using StoreHandle = int32_t;
using ObjectHandle = int64_t;
constexpr static ObjectHandle kInvalidObjectHandle = -1;

// A store for caching shared-ptrs and enlarging lifecycles of the ptrs to match
// lifecycle of the store itself by default, and also serving release calls to
// release a ptr in advance. This is typically used in JNI scenario to bind a
// shared-ptr's lifecycle to a Java-side object or some kind of resource
// manager.
class ObjectStore {
 public:
  static ObjectStore* global() {
    static std::unique_ptr<ObjectStore> globalStore = create();
    return globalStore.get();
  }

  static std::unique_ptr<ObjectStore> create() {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    StoreHandle nextId = safeCast<StoreHandle>(stores().nextId());
    auto store = std::unique_ptr<ObjectStore>(new ObjectStore(nextId));
    StoreHandle storeId = safeCast<StoreHandle>(stores().insert(store.get()));
    VELOX_CHECK(
        storeId == nextId, "Store ID mismatched, this should not happen");
    return store;
  }

  static void release(ObjectHandle handle) {
    ResourceHandle storeId =
        safeCast<ResourceHandle>(handle >> (sizeof(ResourceHandle) * 8));
    ResourceHandle resourceId = safeCast<ResourceHandle>(
        handle & std::numeric_limits<ResourceHandle>::max());
    auto store = stores().lookup(storeId);
    store->releaseInternal(resourceId);
  }

  template <typename T>
  static std::shared_ptr<T> retrieve(ObjectHandle handle) {
    ResourceHandle storeId =
        safeCast<ResourceHandle>(handle >> (sizeof(ResourceHandle) * 8));
    ResourceHandle resourceId = safeCast<ResourceHandle>(
        handle & std::numeric_limits<ResourceHandle>::max());
    auto store = stores().lookup(storeId);
    return store->retrieveInternal<T>(resourceId);
  }

  virtual ~ObjectStore();

  StoreHandle id() const {
    return storeId_;
  }

  template <typename T>
  ObjectHandle save(std::shared_ptr<T> obj) {
    const std::lock_guard<std::mutex> lock(mtx_);
    const std::string_view description = typeid(T).name();
    ResourceHandle handle = store_.insert(std::move(obj));
    aliveObjects_.emplace(handle, description);
    return toObjHandle(handle);
  }

 private:
  static ResourceMap<ObjectStore*>& stores();

  ObjectHandle toObjHandle(ResourceHandle rh) const {
    ObjectHandle prefix = static_cast<ObjectHandle>(storeId_)
        << (sizeof(ResourceHandle) * 8);
    ObjectHandle objHandle = prefix + rh;
    return objHandle;
  }

  template <typename T>
  std::shared_ptr<T> retrieveInternal(ResourceHandle handle) {
    const std::lock_guard<std::mutex> lock(mtx_);
    std::shared_ptr<void> object = store_.lookup(handle);
    // Programming carefully. This will lead to ub if wrong typename T was
    // passed in.
    auto casted = std::static_pointer_cast<T>(object);
    return casted;
  }

  void releaseInternal(ResourceHandle handle);

  explicit ObjectStore(StoreHandle storeId) : storeId_(storeId){};
  StoreHandle storeId_;
  ResourceMap<std::shared_ptr<void>> store_;
  // Preserves handles of objects in the store in order, with the text
  // descriptions associated with them.
  std::map<ResourceHandle, std::string_view> aliveObjects_{};
  std::mutex mtx_;
};
} // namespace velox4j
