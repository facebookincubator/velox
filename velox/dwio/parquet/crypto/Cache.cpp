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
#include "velox/dwio/parquet/crypto/Cache.h"

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
