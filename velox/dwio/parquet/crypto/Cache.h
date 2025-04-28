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

#include <iostream>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <string>
#include <optional>
#include <thread>
#include "velox/dwio/parquet/crypto/EncryptionKey.h"

namespace facebook::velox::parquet {

struct CacheableEncryptedKeyVersionHash {
  std::size_t operator()(const CacheableEncryptedKeyVersion& k) const {
    return k.hash();
  }
};

class Cache {
 public:

  // cleanupIntervalSeconds specifies how often the cleanup thread runs, it is not the TTL for the cached items
  explicit Cache(int cleanupIntervalSeconds): cleanupIntervalSeconds_(cleanupIntervalSeconds) {
    cleanupThread_ = std::thread([this]() {
      while (running_) {
        std::this_thread::sleep_for(std::chrono::seconds(cleanupIntervalSeconds_));
        cleanup();
      }
    });
  }

  ~Cache() {
    running_ = false;
    if (cleanupThread_.joinable()) {
      cleanupThread_.join();
    }
  }

  void set(const CacheableEncryptedKeyVersion& key, const std::string& value, int ttlSeconds);

  std::optional<std::string> get(const CacheableEncryptedKeyVersion& key);

  unsigned long size();

 private:
  struct CacheEntry {
    std::string value;
    std::chrono::steady_clock::time_point expiry_time;
  };

  void cleanup();

  int cleanupIntervalSeconds_;
  std::unordered_map<CacheableEncryptedKeyVersion, CacheEntry, CacheableEncryptedKeyVersionHash> cache_;
  std::mutex mutex_;
  std::thread cleanupThread_;
  bool running_ = true;
};

}
