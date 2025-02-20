#pragma once

#include <iostream>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <string>
#include <optional>
#include <thread>
#include "EncryptionKey.h"

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
