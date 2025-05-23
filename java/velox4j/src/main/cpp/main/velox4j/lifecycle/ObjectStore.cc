#include "ObjectStore.h"
#include <glog/logging.h>

namespace velox4j {
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

ResourceMap<ObjectStore*>& ObjectStore::stores() {
  static ResourceMap<ObjectStore*> stores;
  return stores;
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
} // namespace velox4j
