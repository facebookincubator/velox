#include "Cache.h"

namespace facebook::velox::parquet {

void Cache::set(const CacheableEncryptedKeyVersion& key, const std::string& value, int ttlSeconds) {
  auto now = std::chrono::steady_clock::now();
  auto expiry_time = now + std::chrono::seconds(ttlSeconds);

  std::lock_guard<std::mutex> lock(mutex_);
  cache_[key] = { value, expiry_time };
}

std::optional<std::string> Cache::get(const CacheableEncryptedKeyVersion& key) {
  auto now = std::chrono::steady_clock::now();

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = cache_.find(key);

  if (it != cache_.end()) {
    if (it->second.expiry_time > now) {
      return it->second.value;
    } else {
      cache_.erase(it);
    }
  }

  return std::nullopt;
}

unsigned long Cache::size() {
  std::lock_guard<std::mutex> lock(mutex_);
  return cache_.size();
}

void Cache::cleanup() {
  auto now = std::chrono::steady_clock::now();

  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = cache_.begin(); it != cache_.end(); ) {
    if (it->second.expiry_time <= now) {
      it = cache_.erase(it);
    } else {
      ++it;
    }
  }
}
}
